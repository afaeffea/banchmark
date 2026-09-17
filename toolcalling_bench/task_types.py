"""Общие типы и хелперы для наборов задач tool-calling.

Задача — это диалог из одного или нескольких шагов. На каждом шаге модель получает
историю сообщений и список инструментов, а её ответ проверяет функция шага. Если у шага
задан tool_result, ответ модели и результат инструмента подклеиваются в историю,
и диалог продолжается следующим шагом — так описываются и цепочки вызовов,
и обработка ошибок инструмента.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

ToolCall = dict  # {"name": str, "arguments": dict | None, "raw_arguments": str}


@dataclass
class ToolCallCheck:
    passed: bool
    detail: str


# (вызовы инструментов, текст ответа) -> вердикт
StepCheck = Callable[[list[ToolCall], str], ToolCallCheck]


@dataclass
class Step:
    check: StepCheck
    # что вернуть модели на её вызовы инструментов; None — диалог заканчивается на этом шаге
    tool_result: Optional[Callable[[list[ToolCall]], str]] = None


@dataclass
class Task:
    id: str
    description: str
    prompt: str
    tools: list[dict]
    steps: list[Step]
    # задача проверяет отказ от вызова: по ней считается частота ложных срабатываний
    no_call_expected: bool = False


# ---------- хелперы для чекеров ----------

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def contains_any(haystack: str, needles: list[str]) -> bool:
    low = norm(str(haystack))
    return any(norm(n) in low for n in needles)


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def as_number(value) -> float | None:
    """Числа иногда приходят строкой ('5.5', '100') — приводим к float, где это возможно."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def names_of(calls: list[ToolCall]) -> str:
    return str([c["name"] for c in calls]) if calls else "ни одного (ответ текстом)"


def require_call(name: str, validate: Callable[[dict], ToolCallCheck]) -> StepCheck:
    """Ровно один вызов инструмента name с валидным JSON, дальше проверяет validate."""
    def check(calls: list[ToolCall], text: str) -> ToolCallCheck:
        if len(calls) != 1 or calls[0]["name"] != name:
            return ToolCallCheck(False, f"ожидался ровно один вызов {name}, получено {names_of(calls)}")
        args = calls[0]["arguments"]
        if args is None:
            return ToolCallCheck(False, f"аргументы не распарсились как JSON: {calls[0]['raw_arguments']!r}")
        return validate(args)
    return check


def require_no_call(reason: str, text_validate: Optional[Callable[[str], ToolCallCheck]] = None) -> StepCheck:
    """Модель не должна вызывать инструменты; опционально проверяем сам текст ответа."""
    def check(calls: list[ToolCall], text: str) -> ToolCallCheck:
        if calls:
            return ToolCallCheck(False, f"вызваны инструменты {names_of(calls)}, хотя {reason}")
        if text_validate:
            return text_validate(text)
        return ToolCallCheck(True, "ok")
    return check


def require_text(validate: Callable[[str], ToolCallCheck]) -> StepCheck:
    def check(calls: list[ToolCall], text: str) -> ToolCallCheck:
        return validate(text)
    return check


def ok(detail: str = "ok") -> ToolCallCheck:
    return ToolCallCheck(True, detail)


def fail(detail: str) -> ToolCallCheck:
    return ToolCallCheck(False, detail)
