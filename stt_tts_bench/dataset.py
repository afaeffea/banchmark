"""Тестовые данные для STT/TTS: строки берём через datasets-server API, аудио качаем пофайлово.

Почему не библиотека datasets: у многих аудио-датасетов (google/fleurs — характерный пример)
parquet упакован гигантскими row-group без индекса страниц, поэтому ради одной строки
скачивается весь шард на сотни мегабайт, и это рвётся по таймауту. Здесь скачивается ровно
столько файлов, сколько нужно, и всё складывается в локальный кэш — повторные прогоны мгновенные.

Источник данных настраивается в dataset.yaml.
"""
from __future__ import annotations

import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import yaml

from common.env import DATA_DIR, get_env

_CONFIG_PATH = Path(__file__).parent / "dataset.yaml"
_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_PAGE_SIZE = 100  # ограничение datasets-server на length
CACHE_DIR = DATA_DIR / "cache"


@dataclass
class STTSample:
    sample_id: str
    reference_text: str
    pcm16: bytes
    sample_rate: int
    duration_s: float


def load_config() -> dict:
    return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


def _auth_headers() -> dict:
    token = get_env("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _cache_dir(section: str, cfg: dict) -> Path:
    """Ключ кэша намеренно без n: набор пополняется, а --n просто берёт первые N.
    Иначе прогоны с разным --n качали бы одни и те же записи заново."""
    slug = re.sub(r"[^a-z0-9]+", "_",
                   f"{cfg['dataset']}_{cfg['split']}_{cfg.get('offset', 0)}".lower()).strip("_")
    return CACHE_DIR / f"{section}_{slug}"


def _load_state(cache: Path) -> dict:
    """rows_scanned — сколько строк датасета уже просмотрено (включая отсеянные фильтрами);
    без него докачка начиналась бы с начала и заново отбраковывала те же строки."""
    manifest = cache / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "items" in data:
                return data
        except json.JSONDecodeError:
            pass
    return {"rows_scanned": 0, "items": []}


def _save_state(cache: Path, state: dict) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.json").write_text(json.dumps(state, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


def _fetch_rows(cfg: dict, offset: int, length: int) -> list[dict]:
    resp = requests.get(_ROWS_URL, params={
        "dataset": cfg["dataset"], "config": cfg["config"], "split": cfg["split"],
        "offset": offset, "length": length,
    }, headers=_auth_headers(), timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"datasets-server вернул HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    if "rows" not in payload:
        raise RuntimeError(f"datasets-server не отдал строки: {str(payload)[:300]}")
    return [item["row"] for item in payload["rows"]]


def _iter_rows(cfg: dict, start_offset: int, limit_pages: int = 40):
    offset = start_offset
    for _ in range(limit_pages):
        rows = _fetch_rows(cfg, offset, _PAGE_SIZE)
        if not rows:
            return
        yield from rows
        offset += len(rows)


def _clean_text(row: dict, cfg: dict) -> str:
    return re.sub(r"\s+", " ", str(row.get(cfg["text_field"], "")).strip())


def _to_pcm16(array: np.ndarray) -> bytes:
    return (np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _download_audio(row: dict) -> tuple[np.ndarray, int]:
    url = row["audio"][0]["src"]
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    data, sample_rate = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=True)
    return data[:, 0], sample_rate  # первый канал: оба STT API принимают только моно


# ---------- публичный API ----------

def load_stt_samples(n: int = 20, refresh: bool = False) -> list[STTSample]:
    from .audio_utils import pcm16_to_wav_bytes, wav_bytes_to_pcm16

    cfg = load_config()["stt"]
    cache = _cache_dir("stt", cfg)
    if refresh:
        shutil.rmtree(cache, ignore_errors=True)
    state = _load_state(cache)

    if len(state["items"]) < n:
        needed = n - len(state["items"])
        print(f"  в кэше {len(state['items'])}, докачиваю ещё {needed}", flush=True)
        start = int(cfg.get("offset", 0)) + state["rows_scanned"]
        for row in _iter_rows(cfg, start):
            state["rows_scanned"] += 1
            text = _clean_text(row, cfg)
            if not (cfg["min_chars"] <= len(text) <= cfg["max_chars"]):
                continue
            array, sample_rate = _download_audio(row)
            duration_s = len(array) / sample_rate
            if duration_s > cfg["max_duration_s"]:
                continue
            wav_name = f"{len(state['items']):03d}.wav"
            cache.mkdir(parents=True, exist_ok=True)
            (cache / wav_name).write_bytes(pcm16_to_wav_bytes(_to_pcm16(array), sample_rate))
            state["items"].append({"sample_id": f"{cfg['split']}_{len(state['items'])}",
                                    "reference_text": text, "wav": wav_name,
                                    "sample_rate": sample_rate, "duration_s": round(duration_s, 3)})
            print(f"  скачано {len(state['items'])}/{n} ({duration_s:.1f}s)", flush=True)
            if len(state["items"]) >= n:
                break
        _save_state(cache, state)

    samples = []
    for entry in state["items"][:n]:
        pcm16, sample_rate = wav_bytes_to_pcm16((cache / entry["wav"]).read_bytes())
        samples.append(STTSample(entry["sample_id"], entry["reference_text"], pcm16,
                                  sample_rate, entry["duration_s"]))
    return samples


def load_tts_texts(n: int = 10, refresh: bool = False) -> list[str]:
    cfg = load_config()["tts"]
    cache = _cache_dir("tts", cfg)
    if refresh:
        shutil.rmtree(cache, ignore_errors=True)
    state = _load_state(cache)

    if len(state["items"]) < n:
        seen = {t.lower() for t in state["items"]}
        start = int(cfg.get("offset", 0)) + state["rows_scanned"]
        for row in _iter_rows(cfg, start):
            state["rows_scanned"] += 1
            text = _clean_text(row, cfg)
            if not (cfg["min_chars"] <= len(text) <= cfg["max_chars"]) or text.lower() in seen:
                continue
            seen.add(text.lower())
            state["items"].append(text)
            if len(state["items"]) >= n:
                break
        _save_state(cache, state)

    return state["items"][:n]
