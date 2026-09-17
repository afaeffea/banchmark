"""Прогон одной задачи через одну модель.

Задача — диалог из шагов. На каждом шаге модель получает всю историю и список инструментов;
её ответ проверяет чекер шага. Если шаг задаёт tool_result, ответ модели вместе с результатом
инструмента подклеивается в историю и диалог идёт дальше. Первый непройденный шаг завершает задачу.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from .providers import RateLimiter, call_with_retry
from .task_types import Task, ToolCall


@dataclass
class TaskResult:
    task_id: str
    model_id: str
    passed: bool
    detail: str
    latency_s: float
    json_valid: bool | None  # None — вызовов не было, проверять нечего
    api_error: str | None = None


def _parse_tool_calls(message) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in (getattr(message, "tool_calls", None) or []):
        raw_args = tc.function.arguments or ""
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args = None
        calls.append({"name": tc.function.name, "arguments": args, "raw_arguments": raw_args})
    return calls


def _assistant_message(message) -> dict:
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [{"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in (message.tool_calls or [])],
    }


def run_task(client: OpenAI, model: str, task: Task, limiter: RateLimiter,
              max_retries: int = 0) -> TaskResult:
    messages: list[dict] = [{"role": "user", "content": task.prompt}]
    total_latency = 0.0
    json_valid: bool | None = None
    notify = lambda msg: print(f"    {msg}", flush=True)
    multi_step = len(task.steps) > 1

    for index, step in enumerate(task.steps, 1):
        try:
            resp, latency = call_with_retry(
                lambda: client.chat.completions.create(
                    model=model, messages=messages, tools=task.tools, tool_choice="auto"),
                limiter, max_retries, notify)
        except Exception as exc:  # провайдеры кидают разные классы исключений — фиксируем как есть
            return TaskResult(task.id, model, False, "", total_latency, json_valid,
                               api_error=str(exc)[:300])
        total_latency += latency

        message = resp.choices[0].message
        calls = _parse_tool_calls(message)
        if calls:
            step_valid = all(c["arguments"] is not None for c in calls)
            json_valid = step_valid if json_valid is None else (json_valid and step_valid)

        verdict = step.check(calls, message.content or "")
        if not verdict.passed:
            detail = f"шаг {index}/{len(task.steps)}: {verdict.detail}" if multi_step else verdict.detail
            return TaskResult(task.id, model, False, detail, total_latency, json_valid)

        # без вызовов инструментов подставлять их результат некуда — диалог закончен
        if step.tool_result is None or not calls:
            break

        messages.append(_assistant_message(message))
        tool_output = step.tool_result(calls)
        for tc in (message.tool_calls or []):
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_output})

    return TaskResult(task.id, model, True, "ok", total_latency, json_valid)
