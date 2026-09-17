"""Единый интерфейс для STT/TTS провайдеров, чтобы бенчмарк не знал деталей каждого API."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STTResult:
    text: str
    latency_s: float
    raw_error: str | None = None


@dataclass
class TTSResult:
    audio_bytes: bytes
    audio_format: str  # "wav" | "ogg_opus" | ...
    latency_s: float
    raw_error: str | None = None


class STTProvider:
    name: str = "base"

    def recognize(self, audio_bytes: bytes, sample_rate_hz: int = 16000) -> STTResult:
        """Принимает mono 16-bit PCM WAV (или собственный формат провайдера) и возвращает распознанный текст."""
        raise NotImplementedError


class TTSProvider:
    name: str = "base"

    def synthesize(self, text: str) -> TTSResult:
        raise NotImplementedError
