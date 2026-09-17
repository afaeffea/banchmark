"""CLI: бенчмарк STT — WER/CER + латентность.

Провайдеры перечислены в build_providers(); сейчас настроен только Yandex SpeechKit.

Запуск:
    python -m stt_tts_bench.run_stt --n 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time

from tabulate import tabulate

from common.env import RESULTS_DIR
from .audio_utils import pcm16_to_wav_bytes
from .dataset import load_stt_samples
from .mic_samples import load_mic_samples
from .metrics import cer, latency_stats, wer
from .providers.base import STTProvider
from .providers.yandex import YandexSTT


def build_providers() -> dict[str, STTProvider]:
    """Провайдер без ключей пропускается, а не роняет прогон: так можно гонять
    бенчмарк на части провайдеров, пока настраиваются остальные."""
    providers: dict[str, STTProvider] = {}
    try:
        providers["yandex"] = YandexSTT()
    except RuntimeError as exc:
        print(f"[пропуск] yandex: {exc}", file=sys.stderr)
    return providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк распознавания речи")
    parser.add_argument("--n", type=int, default=20, help="Число тестовых аудио (источник — stt_tts_bench/dataset.yaml)")
    parser.add_argument("--source", choices=["dataset", "mic"], default="dataset",
                        help="dataset — записи из корпуса, mic — ваши записи с микрофона из панели")
    parser.add_argument("--refresh", action="store_true",
                        help="Перекачать записи заново, игнорируя кэш")
    args = parser.parse_args()

    providers = build_providers()
    if not providers:
        print("Нет ни одного настроенного провайдера — заполните .env", file=sys.stderr)
        sys.exit(1)

    if args.source == "mic":
        print("Беру записи с микрофона (вкладка «Микрофон» в панели)...")
        samples = load_mic_samples(n=args.n)
        if not samples:
            print("Записей с микрофона нет — запишите их в панели на вкладке «Микрофон»",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Загружаю {args.n} сэмплов из датасета (см. stt_tts_bench/dataset.yaml)...")
        samples = load_stt_samples(n=args.n, refresh=args.refresh)
    print(f"Загружено {len(samples)} сэмплов.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    samples_dir = RESULTS_DIR / f"stt_samples_{ts}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    manifest_items = []
    per_provider_latencies: dict[str, list[float]] = {name: [] for name in providers}
    per_provider_wer: dict[str, list[float]] = {name: [] for name in providers}
    per_provider_cer: dict[str, list[float]] = {name: [] for name in providers}
    per_provider_errors: dict[str, int] = {name: 0 for name in providers}

    for i, sample in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {sample.sample_id}: {sample.reference_text[:60]!r}")

        # исходное аудио кладём рядом с результатами, чтобы можно было послушать и сверить
        wav_name = f"{i:03d}.wav"
        (samples_dir / wav_name).write_bytes(pcm16_to_wav_bytes(sample.pcm16, sample.sample_rate))
        item = {"id": i, "audio": wav_name, "reference": sample.reference_text,
                 "duration_s": round(sample.duration_s, 2), "hypotheses": {}}

        row = {"sample_id": sample.sample_id, "reference": sample.reference_text,
                "audio_path": str(samples_dir / wav_name), "duration_s": round(sample.duration_s, 2)}
        for name, provider in providers.items():
            result = provider.recognize(sample.pcm16, sample_rate_hz=sample.sample_rate)
            row[f"{name}_hypothesis"] = result.text
            row[f"{name}_latency_s"] = round(result.latency_s, 3)
            row[f"{name}_error"] = result.raw_error or ""
            per_provider_latencies[name].append(result.latency_s)
            if result.raw_error:
                per_provider_errors[name] += 1
                row[f"{name}_wer"] = ""
                row[f"{name}_cer"] = ""
                item["hypotheses"][name] = {"text": "", "wer": None, "cer": None,
                                             "latency_s": round(result.latency_s, 3),
                                             "error": result.raw_error}
            else:
                w, c = wer(sample.reference_text, result.text), cer(sample.reference_text, result.text)
                row[f"{name}_wer"] = round(w, 4)
                row[f"{name}_cer"] = round(c, 4)
                per_provider_wer[name].append(w)
                per_provider_cer[name].append(c)
                item["hypotheses"][name] = {"text": result.text, "wer": round(w, 4), "cer": round(c, 4),
                                             "latency_s": round(result.latency_s, 3), "error": ""}
        per_sample_rows.append(row)
        manifest_items.append(item)

    (samples_dir / "manifest.json").write_text(json.dumps(
        {"kind": "stt", "created": ts, "providers": list(providers), "items": manifest_items},
        ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = RESULTS_DIR / f"stt_bench_{ts}.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(f"\nПодробные результаты: {out_csv}")
    print(f"Исходные аудиозаписи (можно прослушать и сверить): {samples_dir}")

    summary_rows = []
    for name in providers:
        lat = latency_stats(per_provider_latencies[name])
        mean_wer = sum(per_provider_wer[name]) / len(per_provider_wer[name]) if per_provider_wer[name] else float("nan")
        mean_cer = sum(per_provider_cer[name]) / len(per_provider_cer[name]) if per_provider_cer[name] else float("nan")
        summary_rows.append([
            name, f"{mean_wer:.1%}" if per_provider_wer[name] else "—",
            f"{mean_cer:.1%}" if per_provider_cer[name] else "—",
            f"{lat.mean_s:.2f}s", f"{lat.median_s:.2f}s", f"{lat.p95_s:.2f}s",
            per_provider_errors[name],
        ])

    print()
    print(tabulate(summary_rows, headers=["provider", "mean WER", "mean CER", "avg latency", "median latency",
                                           "p95 latency", "errors"], tablefmt="github"))


if __name__ == "__main__":
    main()
