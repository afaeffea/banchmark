"""Тяжёлый набор задач на tool-calling.

Базовый набор (tasks.py) проверяет саму способность вызвать инструмент — его проходят все.
Здесь собрано то, на чём модели реально расходятся: выбор нужного инструмента из десятка
похожих, вложенные объекты и массивы в аргументах, цепочка из двух зависимых вызовов,
обработка ошибки инструмента и — самое показательное — отказ вызывать инструмент,
когда обязательных данных в запросе нет.
"""
from __future__ import annotations

import json
import re

from .task_types import (Step, Task, ToolCallCheck, as_number, contains_any, fail, names_of,
                          norm, ok, require_call, require_no_call, require_text)

# ---------- разбор даты и времени ----------

def _extract_date(value: str) -> tuple[int, int, int] | None:
    """Возвращает (год, месяц, день) из ISO или из привычного dd.mm.yyyy."""
    text = str(value)
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text)
    if m:
        return int(m.group(3)), int(m.group(2)), int(m.group(1))
    return None


_ISO_DATE_RE = r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
_DMY_DATE_RE = r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"


def _extract_time(value: str) -> tuple[int, int] | None:
    # дату вырезаем заранее: иначе «17.09.2026» читается как время 17:09
    text = re.sub(_ISO_DATE_RE, " ", str(value))
    text = re.sub(_DMY_DATE_RE, " ", text)
    m = re.search(r"(\d{1,2})[:.](\d{2})", text)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    low = text.lower()
    if "pm" in low and hour < 12:
        hour += 12
    if "am" in low and hour == 12:
        hour = 0
    return hour, minute


def _check_datetime(when_values: list, date_expected: tuple[int, int, int],
                     time_expected: tuple[int, int], label: str) -> ToolCallCheck:
    """Дата и время могут лежать в разных полях или быть слиты в одну ISO-строку —
    поэтому ищем их по всем переданным значениям."""
    joined = " ".join(str(v) for v in when_values if v is not None)
    got_date = _extract_date(joined)
    got_time = _extract_time(joined)
    if got_date != date_expected:
        return fail(f"{label}: дата {got_date}, ожидалась {date_expected} (из {joined!r})")
    if got_time != time_expected:
        return fail(f"{label}: время {got_time}, ожидалось {time_expected} (из {joined!r})")
    return ok()


# ---------- каталог инструментов ----------

SEARCH_FLIGHTS = {
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Найти авиарейсы между городами на дату",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                            "date": {"type": "string", "description": "Дата вылета, ISO8601"}},
            "required": ["origin", "destination", "date"],
        },
    },
}

SEARCH_TRAINS = {
    "type": "function",
    "function": {
        "name": "search_trains",
        "description": "Найти железнодорожные рейсы между городами на дату",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"},
                            "date": {"type": "string", "description": "Дата отправления, ISO8601"}},
            "required": ["origin", "destination", "date"],
        },
    },
}

BOOK_TAXI = {
    "type": "function",
    "function": {
        "name": "book_taxi",
        "description": "Заказать такси от адреса до адреса",
        "parameters": {
            "type": "object",
            "properties": {"pickup": {"type": "string"}, "dropoff": {"type": "string"},
                            "when": {"type": "string"}},
            "required": ["pickup", "dropoff", "when"],
        },
    },
}

SEARCH_HOTELS = {
    "type": "function",
    "function": {
        "name": "search_hotels",
        "description": "Найти отели в городе на даты заезда и выезда",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "checkin": {"type": "string"},
                            "checkout": {"type": "string"}, "guests": {"type": "integer"}},
            "required": ["city", "checkin", "checkout", "guests"],
        },
    },
}

WEATHER_BY_DATE = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Прогноз погоды в городе на дату",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
            "required": ["city", "date"],
        },
    },
}

CONVERT_CURRENCY = {
    "type": "function",
    "function": {
        "name": "convert_currency",
        "description": "Конвертировать сумму из одной валюты в другую",
        "parameters": {
            "type": "object",
            "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"},
                            "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
}

CONVERT_UNITS = {
    "type": "function",
    "function": {
        "name": "convert_units",
        "description": "Перевести величину из одной единицы измерения в другую",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "enum": ["kg", "g", "lb", "oz", "km", "m", "mi", "ft"]},
                "to_unit": {"type": "string", "enum": ["kg", "g", "lb", "oz", "km", "m", "mi", "ft"]},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
}

FIND_USER = {
    "type": "function",
    "function": {
        "name": "find_user",
        "description": "Найти пользователя по email и получить его user_id",
        "parameters": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
}

BLOCK_USER = {
    "type": "function",
    "function": {
        "name": "block_user",
        "description": "Заблокировать пользователя по его user_id",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "Идентификатор из find_user"},
                            "reason": {"type": "string"}},
            "required": ["user_id", "reason"],
        },
    },
}

CREATE_EVENT = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Создать событие в календаре",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "when": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Дата в формате YYYY-MM-DD"},
                        "time": {"type": "string", "description": "Время в формате HH:MM"},
                        "timezone": {"type": "string", "description": "Например Europe/Moscow"},
                    },
                    "required": ["date", "time", "timezone"],
                },
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                        "required": ["name", "email"],
                    },
                },
            },
            "required": ["title", "when", "attendees"],
        },
    },
}

SEND_INVOICE = {
    "type": "function",
    "function": {
        "name": "send_invoice",
        "description": "Выставить счёт клиенту",
        "parameters": {
            "type": "object",
            "properties": {
                "client_email": {"type": "string"},
                "currency": {"type": "string", "enum": ["RUB", "USD", "EUR"]},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "quantity": {"type": "number"},
                            "unit_price": {"type": "number"},
                        },
                        "required": ["description", "quantity", "unit_price"],
                    },
                },
            },
            "required": ["client_email", "currency", "line_items"],
        },
    },
}

BOOK_TABLE = {
    "type": "function",
    "function": {
        "name": "book_table",
        "description": "Забронировать столик в ресторане",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant": {"type": "string"},
                "datetime": {"type": "string", "description": "Дата и время брони, ISO8601"},
                "people": {"type": "integer", "description": "Количество гостей"},
            },
            "required": ["restaurant", "datetime", "people"],
        },
    },
}

ALL_TOOLS = [SEARCH_FLIGHTS, SEARCH_TRAINS, BOOK_TAXI, SEARCH_HOTELS, WEATHER_BY_DATE,
             CONVERT_CURRENCY, CONVERT_UNITS, FIND_USER, BLOCK_USER, CREATE_EVENT,
             SEND_INVOICE, BOOK_TABLE]


# ---------- чекеры ----------

def _check_train_search(args: dict) -> ToolCallCheck:
    if not contains_any(args.get("origin", ""), ["москв", "moscow"]):
        return fail(f"origin={args.get('origin')!r}, ожидалась Москва")
    if not contains_any(args.get("destination", ""), ["петербург", "spb", "saint", "petersburg", "питер"]):
        return fail(f"destination={args.get('destination')!r}, ожидался Санкт-Петербург")
    got = _extract_date(args.get("date", ""))
    if got != (2026, 9, 16):
        return fail(f"date={args.get('date')!r} -> {got}, ожидалось 16.09.2026")
    return ok()


def _check_nested_event(args: dict) -> ToolCallCheck:
    when = args.get("when")
    if not isinstance(when, dict):
        return fail(f"when={when!r} — ожидался вложенный объект с полями date/time/timezone")
    verdict = _check_datetime([when.get("date"), when.get("time")], (2026, 10, 20), (14, 0), "when")
    if not verdict.passed:
        return verdict
    if not contains_any(when.get("timezone", ""), ["moscow", "msk", "+03", "мск"]):
        return fail(f"when.timezone={when.get('timezone')!r}, ожидался Europe/Moscow")
    attendees = args.get("attendees")
    if not isinstance(attendees, list) or len(attendees) != 2:
        return fail(f"attendees={attendees!r} — ожидался массив ровно из двух участников")
    emails = norm(json.dumps(attendees, ensure_ascii=False))
    for expected in ("anna@example.com", "boris@example.com"):
        if expected not in emails:
            return fail(f"в attendees нет {expected}: {attendees!r}")
    if not contains_any(args.get("title", ""), ["релиз"]):
        return fail(f"title={args.get('title')!r}, ожидалось упоминание релиза")
    return ok()


def _check_invoice(args: dict) -> ToolCallCheck:
    if not contains_any(args.get("client_email", ""), ["billing@acme.com"]):
        return fail(f"client_email={args.get('client_email')!r}, ожидался billing@acme.com")
    if args.get("currency") != "EUR":
        return fail(f"currency={args.get('currency')!r}, ожидалось ровно 'EUR' (значение enum)")
    items = args.get("line_items")
    if not isinstance(items, list) or len(items) != 3:
        return fail(f"line_items — ожидался массив из трёх позиций, получено {items!r}")
    quantities = sorted(filter(None, (as_number(i.get("quantity")) for i in items)))
    prices = sorted(filter(None, (as_number(i.get("unit_price")) for i in items)))
    if quantities != [1.0, 2.0, 3.0]:
        return fail(f"quantity по позициям: {quantities}, ожидались 3, 2 и 1")
    if prices != [300.0, 450.0, 1200.0]:
        return fail(f"unit_price по позициям: {prices}, ожидались 450, 300 и 1200")
    return ok()


def _check_block_uses_returned_id(args: dict) -> ToolCallCheck:
    user_id = str(args.get("user_id", "")).strip()
    if user_id != "u_8823":
        return fail(f"user_id={user_id!r}, ожидался ровно 'u_8823' из ответа find_user "
                     f"(модель не должна выдумывать идентификатор)")
    if not str(args.get("reason", "")).strip():
        return fail("reason пустой, хотя обязателен по схеме")
    return ok()


def _check_asks_for_missing_info(text: str) -> ToolCallCheck:
    if not text.strip():
        return fail("пустой ответ: модель не вызвала инструмент, но и не спросила недостающие данные")
    asks = "?" in text or contains_any(text, ["уточн", "подскаж", "сколько", "когда", "на какое",
                                                "какую дату", "какое время", "нужны"])
    if not asks:
        return fail(f"модель не запросила недостающие дату и число гостей: {text[:200]!r}")
    return ok()


def _check_relative_datetime(args: dict) -> ToolCallCheck:
    when = args.get("when")
    if not isinstance(when, dict):
        return fail(f"when={when!r} — ожидался вложенный объект")
    return _check_datetime([when.get("date"), when.get("time")], (2026, 9, 17), (19, 30), "when")


def _check_reports_tool_error(calls, text) -> ToolCallCheck:
    if any(c["name"] == "block_user" for c in calls):
        return fail("модель пытается заблокировать пользователя, которого find_user не нашёл")
    if calls:
        return fail(f"вместо сообщения о неудаче модель вызвала {names_of(calls)}")
    if not contains_any(text, ["не найден", "не найдено", "не существует", "not found", "отсутству",
                                "нет пользователя", "не удалось найти", "не зарегистрирован"]):
        return fail(f"в ответе нет признания, что пользователь не найден: {text[:200]!r}")
    return ok()


def _check_three_cities(calls, text) -> ToolCallCheck:
    cities = [norm(str((c["arguments"] or {}).get("city", "")))
              for c in calls if c["name"] == "get_weather"]
    wanted = {"москв|moscow": False, "токио|tokyo": False, "лондон|london": False}
    for key in wanted:
        wanted[key] = any(re.search(key, c) for c in cities)
    missing = [k.split("|")[0] for k, found in wanted.items() if not found]
    if missing:
        return fail(f"не запрошены города: {missing} (вызовы: {cities or names_of(calls)})")
    return ok(f"ok (все три города, вызовов: {len(cities)})")


def _check_unit_enum(args: dict) -> ToolCallCheck:
    value = as_number(args.get("value"))
    if value != 5.5:
        return fail(f"value={args.get('value')!r}, ожидалось 5.5 («пять с половиной»)")
    if args.get("from_unit") != "kg":
        return fail(f"from_unit={args.get('from_unit')!r}, ожидалось ровно 'kg'")
    if args.get("to_unit") != "lb":
        return fail(f"to_unit={args.get('to_unit')!r}, ожидалось ровно 'lb'")
    return ok()


def build_hard_tasks() -> list[Task]:
    return [
        Task(
            id="tool_selection_among_many",
            description="Выбор одного нужного инструмента из двенадцати, среди которых есть похожие",
            prompt="Мне нужно добраться из Москвы в Санкт-Петербург 16 сентября 2026 года на поезде. "
                   "Найди варианты.",
            tools=ALL_TOOLS,
            steps=[Step(require_call("search_trains", _check_train_search))],
        ),
        Task(
            id="nested_object_args",
            description="Вложенный объект и массив объектов в аргументах",
            prompt="Создай встречу «Созвон по релизу» на 20 октября 2026 года в 14:00 по часовому поясу "
                   "Europe/Moscow. Участники: Анна (anna@example.com) и Борис (boris@example.com).",
            tools=[CREATE_EVENT],
            steps=[Step(require_call("create_calendar_event", _check_nested_event))],
        ),
        Task(
            id="array_of_objects",
            description="Массив из трёх объектов с числовыми полями + строгий enum валюты",
            prompt="Выставь счёт на billing@acme.com в евро по трём позициям: "
                   "«Разработка макетов» — 3 штуки по 450, «Вёрстка» — 2 штуки по 300, "
                   "«Поддержка» — 1 штука по 1200.",
            tools=[SEND_INVOICE],
            steps=[Step(require_call("send_invoice", _check_invoice))],
        ),
        Task(
            id="chain_find_then_block",
            description="Цепочка: результат первого вызова — обязательный аргумент второго",
            prompt="Заблокируй пользователя с почтой petrov@example.com за рассылку спама.",
            tools=[FIND_USER, BLOCK_USER],
            steps=[
                Step(require_call("find_user",
                                   lambda a: ok() if contains_any(a.get("email", ""), ["petrov@example.com"])
                                   else fail(f"email={a.get('email')!r}, ожидался petrov@example.com")),
                     tool_result=lambda calls: json.dumps(
                         {"user_id": "u_8823", "name": "Пётр Петров", "status": "active"}, ensure_ascii=False)),
                Step(require_call("block_user", _check_block_uses_returned_id)),
            ],
        ),
        Task(
            id="missing_info_must_ask",
            description="Обязательных данных нет — правильно переспросить, а не выдумать аргументы",
            prompt="Забронируй столик в ресторане «Пушкинъ».",
            tools=ALL_TOOLS,
            steps=[Step(require_no_call(
                "в запросе не указаны ни дата со временем, ни число гостей — оба поля обязательны",
                _check_asks_for_missing_info))],
            no_call_expected=True,
        ),
        Task(
            id="relative_date_and_time",
            description="Относительная дата и разговорное время: «послезавтра в половине восьмого вечера»",
            prompt="Сегодня вторник, 15 сентября 2026 года. Поставь в календарь встречу «Ретро» "
                   "на послезавтра в половине восьмого вечера, часовой пояс Europe/Moscow. "
                   "Участник один: Анна (anna@example.com).",
            tools=[CREATE_EVENT],
            steps=[Step(require_call("create_calendar_event", _check_relative_datetime))],
        ),
        Task(
            id="tool_error_handling",
            description="Инструмент вернул ошибку — нельзя делать вид, что всё получилось",
            prompt="Заблокируй пользователя с почтой nobody@example.com за нарушение правил.",
            tools=[FIND_USER, BLOCK_USER],
            steps=[
                Step(require_call("find_user",
                                   lambda a: ok() if contains_any(a.get("email", ""), ["nobody@example.com"])
                                   else fail(f"email={a.get('email')!r}, ожидался nobody@example.com")),
                     tool_result=lambda calls: json.dumps(
                         {"error": "user_not_found",
                          "message": "Пользователь с таким email не зарегистрирован"}, ensure_ascii=False)),
                Step(_check_reports_tool_error),
            ],
        ),
        Task(
            id="parallel_three_cities",
            description="Три параллельных вызова с разными аргументами за один ход",
            prompt="Сравни погоду в Москве, Токио и Лондоне на завтра, 16 сентября 2026 года.",
            tools=[WEATHER_BY_DATE],
            steps=[Step(_check_three_cities)],
        ),
        Task(
            id="enum_unit_mapping",
            description="Сопоставление разговорной формулировки со значениями enum и дробным числом",
            prompt="Сколько будет пять с половиной килограммов в фунтах?",
            tools=[CONVERT_UNITS],
            steps=[Step(require_call("convert_units", _check_unit_enum))],
        ),
    ]
