"""Реестр наборов задач: basic — проверка механики, hard — то, что различает модели."""
from __future__ import annotations

from .task_types import Task
from .tasks import build_tasks
from .tasks_hard import build_hard_tasks

SUITES = {
    "basic": ("Базовый набор", build_tasks),
    "hard": ("Тяжёлый набор", build_hard_tasks),
    "all": ("Оба набора", lambda: build_tasks() + build_hard_tasks()),
}


def load_suite(name: str) -> list[Task]:
    if name not in SUITES:
        raise ValueError(f"неизвестный набор задач {name!r}; доступны: {', '.join(SUITES)}")
    return SUITES[name][1]()


def suite_title(name: str) -> str:
    return SUITES[name][0] if name in SUITES else name
