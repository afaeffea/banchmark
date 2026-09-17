"""Базовый набор задач на tool-calling: проверяет, что механика вообще работает.

Задачи намеренно простые — это проверка «умеет ли модель в тулкол в принципе».
Для различения моделей между собой см. tasks_hard.py.
"""
from __future__ import annotations

from .task_types import (Step, Task, ToolCallCheck, contains_any, fail, names_of, ok,
                          require_call, require_no_call, require_text)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Получить текущую погоду в указанном городе",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Название города"}},
            "required": ["city"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Выполнить арифметическую операцию над двумя числами",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
            },
            "required": ["a", "b", "operation"],
        },
    },
}

CURRENCY_TOOL = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Конвертировать сумму из одной валюты в другую",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "Код валюты, напр. USD"},
                "to_currency": {"type": "string", "description": "Код валюты, напр. EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}

REMINDER_TOOL = {
    "type": "function",
    "function": {
        "name": "set_reminder",
        "description": "Поставить напоминание",
        "parameters": {
            "type": "object",
            "properties": {
                "datetime": {"type": "string", "description": "Дата и время в свободном или ISO8601 формате"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "text": {"type": "string"},
            },
            "required": ["datetime", "priority", "text"],
        },
    },
}


def _city_is(args: dict, variants: list[str]) -> ToolCallCheck:
    city = args.get("city", "")
    if not contains_any(city, variants):
        return fail(f"city={city!r}, ожидался один из {variants}")
    return ok()


def _check_calculator(args: dict) -> ToolCallCheck:
    a, b, op = args.get("a"), args.get("b"), args.get("operation")
    if {a, b} != {823, 47}:
        return fail(f"a/b={a}/{b}, ожидались 823 и 47 (в любом порядке)")
    if op != "multiply":
        return fail(f"operation={op!r}, ожидалось multiply")
    return ok()


def _check_currency(args: dict) -> ToolCallCheck:
    if args.get("amount") != 100:
        return fail(f"amount={args.get('amount')}, ожидалось 100")
    if not contains_any(args.get("from_currency", ""), ["usd", "доллар"]):
        return fail(f"from_currency={args.get('from_currency')!r}, ожидался USD")
    if not contains_any(args.get("to_currency", ""), ["eur", "евро"]):
        return fail(f"to_currency={args.get('to_currency')!r}, ожидался EUR")
    return ok()


def _check_reminder(args: dict) -> ToolCallCheck:
    priority, dt = args.get("priority"), str(args.get("datetime", ""))
    if priority != "high":
        return fail(f"priority={priority!r}, ожидалось ровно 'high' (строгое значение enum)")
    if not (("15:00" in dt or "15-09" in dt or "09-15" in dt or "15.09" in dt) and "2026" in dt):
        return fail(f"datetime={dt!r} не похоже на 15 сентября 2026, 15:00")
    return ok()


def _check_parallel_weather(calls, text) -> ToolCallCheck:
    cities = [str((c["arguments"] or {}).get("city", "")).lower()
              for c in calls if c["name"] == "get_weather"]
    has_moscow = any("моск" in c or "moscow" in c for c in cities)
    has_tokyo = any("токио" in c or "tokyo" in c for c in cities)
    if has_moscow and has_tokyo:
        return ok("ok (оба города за один ход)")
    if len(cities) == 1:
        return fail(f"запрошен только один город из двух: {cities}")
    return fail(f"не найдено вызовов get_weather для Москвы и Токио: {cities or names_of(calls)}")


def _final_mentions_18(text: str) -> ToolCallCheck:
    if "18" in text:
        return ok()
    return fail(f"в финальном ответе нет температуры (18) из результата инструмента: {text[:200]!r}")


def build_tasks() -> list[Task]:
    return [
        Task(
            id="weather_single",
            description="Простой единичный вызов инструмента с одним аргументом",
            prompt="Какая погода сейчас в Москве?",
            tools=[WEATHER_TOOL],
            steps=[Step(require_call("get_weather", lambda a: _city_is(a, ["москва", "moscow"])))],
        ),
        Task(
            id="calc_arithmetic",
            description="Вызов с несколькими типизированными аргументами + enum",
            prompt="Сколько будет 823 умножить на 47? Посчитай через инструмент calculator.",
            tools=[CALCULATOR_TOOL],
            steps=[Step(require_call("calculator", _check_calculator))],
        ),
        Task(
            id="currency_conversion",
            description="Три обязательных аргумента разных типов",
            prompt="Переведи 100 долларов США в евро.",
            tools=[CURRENCY_TOOL],
            steps=[Step(require_call("convert_currency", _check_currency))],
        ),
        Task(
            id="no_tool_needed",
            description="Проверка ложных срабатываний: инструменты доступны, но не нужны",
            prompt="Привет! Как у тебя дела?",
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            steps=[Step(require_no_call("это обычный small-talk, инструменты не нужны"))],
            no_call_expected=True,
        ),
        Task(
            id="enum_strict",
            description="Строгое соответствие enum-значению + составной аргумент даты",
            prompt="Поставь напоминание 'позвонить врачу' на 15:00 15 сентября 2026 года с высоким приоритетом.",
            tools=[REMINDER_TOOL],
            steps=[Step(require_call("set_reminder", _check_reminder))],
        ),
        Task(
            id="parallel_calls",
            description="Параллельные вызовы одного инструмента за один ход",
            prompt="Узнай, пожалуйста, погоду и в Москве, и в Токио.",
            tools=[WEATHER_TOOL],
            steps=[Step(_check_parallel_weather)],
        ),
        Task(
            id="multi_turn_followup",
            description="Двухходовая задача: результат инструмента должен попасть в финальный ответ",
            prompt="Какая сейчас погода в Париже?",
            tools=[WEATHER_TOOL],
            steps=[
                Step(require_call("get_weather", lambda a: _city_is(a, ["париж", "paris"])),
                     tool_result=lambda calls: '{"city": "Paris", "temp_c": 18, "condition": "cloudy"}'),
                Step(require_text(_final_mentions_18)),
            ],
        ),
    ]
