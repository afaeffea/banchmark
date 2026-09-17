"""WER/CER и агрегация задержек."""
from __future__ import annotations

import re
import statistics as stats
from dataclasses import dataclass

import jiwer

_PUNCT_RE = re.compile(r"[^\w\sа-яёa-z0-9]", re.IGNORECASE | re.UNICODE)


def normalize_ru(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_ru(reference), normalize_ru(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_ru(reference), normalize_ru(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.cer(ref, hyp)


@dataclass
class LatencyStats:
    n: int
    mean_s: float
    median_s: float
    p95_s: float
    min_s: float
    max_s: float


def latency_stats(values: list[float]) -> LatencyStats:
    if not values:
        return LatencyStats(0, 0, 0, 0, 0, 0)
    sorted_vals = sorted(values)
    p95_idx = min(len(sorted_vals) - 1, int(round(0.95 * (len(sorted_vals) - 1))))
    return LatencyStats(
        n=len(values),
        mean_s=stats.mean(values),
        median_s=stats.median(values),
        p95_s=sorted_vals[p95_idx],
        min_s=min(values),
        max_s=max(values),
    )
