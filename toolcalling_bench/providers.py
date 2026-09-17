"""Реестр моделей: у всех пяти провайдеров есть OpenAI-совместимый chat/completions
эндпоинт, поэтому единый openai.Client с разными base_url покрывает все случаи —
не нужен отдельный клиент под каждого вендора."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import yaml
from openai import (APIConnectionError, APITimeoutError, InternalServerError, OpenAI,
                     RateLimitError)

# Сбои, которые не говорят ничего о качестве модели: лимит частоты, таймаут, обрыв связи,
# 5xx на стороне провайдера. Их повторяем, иначе инфраструктурный шум попадёт в сравнение.
TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

from common.env import get_env

_MODELS_YAML = Path(__file__).parent / "models.yaml"

T = TypeVar("T")


@dataclass
class ModelConfig:
    id: str
    display_name: str
    model: str
    base_url: str
    api_key_env: str


@dataclass
class RateLimit:
    requests_per_minute: int = 0
    max_retries: int = 4


def load_model_configs() -> list[ModelConfig]:
    data = yaml.safe_load(_MODELS_YAML.read_text(encoding="utf-8"))
    return [ModelConfig(**entry) for entry in data["models"]]


def load_rate_limit() -> RateLimit:
    data = yaml.safe_load(_MODELS_YAML.read_text(encoding="utf-8")) or {}
    return RateLimit(**(data.get("rate_limit") or {}))


class RateLimiter:
    """Выдерживает паузу между запросами. Общий на весь прогон: при работе через роутер
    лимит считается по ключу, поэтому делить его между моделями нельзя."""

    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        delay = self._last + self.min_interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()


def call_with_retry(fn: Callable[[], T], limiter: RateLimiter, max_retries: int,
                     on_retry: Callable[[str], None] | None = None) -> tuple[T, float]:
    """Возвращает (результат, длительность последней попытки).

    Пауза троттлинга и неудачные попытки в длительность не входят — иначе метрика
    латентности показывала бы наше собственное ожидание, а не скорость модели.
    """
    for attempt in range(max_retries + 1):
        limiter.wait()
        started = time.perf_counter()
        try:
            return fn(), time.perf_counter() - started
        except TRANSIENT_ERRORS as exc:
            if attempt >= max_retries:
                raise
            pause = min(60.0, 15.0 * (attempt + 1))
            if on_retry:
                on_retry(f"{type(exc).__name__} — пауза {pause:.0f}s, "
                          f"попытка {attempt + 2}/{max_retries + 1}")
            time.sleep(pause)
    raise RuntimeError("недостижимо")


def build_client(cfg: ModelConfig) -> OpenAI | None:
    api_key = get_env(cfg.api_key_env)
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=cfg.base_url)
