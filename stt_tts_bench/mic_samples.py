"""Записи с микрофона: пишет веб-панель, читают оба бенчмарка.

Одна запись даёт сразу пару тестов: аудио уходит в STT (эталон — текст, который вы
ввели вручную), а тот же текст уходит в TTS. То есть распознавание и синтез проверяются
на вашем материале, а не только на дикторской начитке из корпуса.

Лежит в data/mic_samples/ — каталог в .gitignore, ваши записи никуда не уедут.
"""
from __future__ import annotations

import json
import re
import time
import wave
from pathlib import Path

from common.env import DATA_DIR
from .audio_utils import wav_bytes_to_pcm16
from .dataset import STTSample

MIC_DIR = DATA_DIR / "mic_samples"
_MANIFEST = MIC_DIR / "manifest.json"


def _load() -> list[dict]:
    if not _MANIFEST.is_file():
        return []
    try:
        data = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _save(items: list[dict]) -> None:
    MIC_DIR.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def list_samples() -> list[dict]:
    return _load()


def add_sample(wav_bytes: bytes, reference_text: str) -> dict:
    """Возвращает описание записи. Валидирует WAV сразу: лучше отказать при сохранении,
    чем уронить прогон бенчмарка через час."""
    reference_text = re.sub(r"\s+", " ", reference_text).strip()
    if not reference_text:
        raise ValueError("нужен эталонный текст — что именно было сказано")
    try:
        pcm16, sample_rate = wav_bytes_to_pcm16(wav_bytes)
    except (wave.Error, EOFError) as exc:
        # wave кидает свой класс ошибки, а наружу нужен ValueError — иначе панель
        # покажет голый HTTP 500 вместо объяснения
        raise ValueError(f"файл не распознан как WAV: {exc}") from exc
    duration_s = len(pcm16) / 2 / sample_rate
    if duration_s < 0.3:
        raise ValueError("запись слишком короткая")
    if duration_s > 60:
        raise ValueError("запись длиннее минуты — синхронные API её не примут")

    MIC_DIR.mkdir(parents=True, exist_ok=True)
    items = _load()
    sample_id = f"mic_{int(time.time() * 1000)}"
    wav_name = f"{sample_id}.wav"
    (MIC_DIR / wav_name).write_bytes(wav_bytes)

    entry = {"sample_id": sample_id, "reference_text": reference_text, "wav": wav_name,
              "sample_rate": sample_rate, "duration_s": round(duration_s, 2),
              "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    items.append(entry)
    _save(items)
    return entry


def delete_sample(sample_id: str) -> bool:
    items = _load()
    kept = [i for i in items if i["sample_id"] != sample_id]
    if len(kept) == len(items):
        return False
    removed = next(i for i in items if i["sample_id"] == sample_id)
    (MIC_DIR / removed["wav"]).unlink(missing_ok=True)
    _save(kept)
    return True


def sample_path(sample_id: str) -> Path | None:
    entry = next((i for i in _load() if i["sample_id"] == sample_id), None)
    return (MIC_DIR / entry["wav"]) if entry else None


def load_mic_samples(n: int | None = None) -> list[STTSample]:
    samples = []
    for entry in _load():
        pcm16, sample_rate = wav_bytes_to_pcm16((MIC_DIR / entry["wav"]).read_bytes())
        samples.append(STTSample(entry["sample_id"], entry["reference_text"], pcm16,
                                  sample_rate, entry["duration_s"]))
        if n is not None and len(samples) >= n:
            break
    return samples


def load_mic_texts(n: int | None = None) -> list[str]:
    texts = [entry["reference_text"] for entry in _load()]
    return texts[:n] if n is not None else texts
