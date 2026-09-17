"""Мелкие хелперы для работы с сырым PCM16 и WAV-контейнером.

Все провайдеры внутри бенчмарка обмениваются raw PCM16LE mono (без заголовка) —
это единственный формат, который принимают оба STT API без дополнительного парсинга.
WAV нужен только на границах: при чтении датасета и при сохранении сэмплов для ручного прослушивания.
"""
from __future__ import annotations

import io
import wave


def wav_bytes_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"ожидается 16-bit PCM WAV, получено sampwidth={wf.getsampwidth()}")
        pcm = wf.readframes(wf.getnframes())
        return pcm, wf.getframerate()


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate_hz: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
