"""CLI: бенчмарк TTS — латентность + автоматическая "разборчивость" (WER синтеза,
прогнанного через STT) + подготовка листа для ручной оценки.

Запуск:
    python -m stt_tts_bench.run_tts --n 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time

from tabulate import tabulate

from common.env import RESULTS_DIR
from .audio_utils import pcm16_to_wav_bytes, wav_bytes_to_pcm16
from .dataset import load_tts_texts
from .mic_samples import load_mic_texts
from .metrics import latency_stats, wer
from .providers.base import TTSProvider
from .providers.yandex import YandexSTT, YandexTTS


def build_tts_providers() -> dict[str, TTSProvider]:
    providers: dict[str, TTSProvider] = {}
    try:
        providers["yandex"] = YandexTTS()
    except RuntimeError as exc:
        print(f"[пропуск TTS] yandex: {exc}", file=sys.stderr)
    return providers


def build_stt_checkers() -> dict[str, YandexSTT]:
    """Движки, которыми проверяется разборчивость синтеза.

    ВАЖНО: пока здесь только Yandex, синтез Яндекса проверяется его же распознаванием.
    Это self-scoring: общий фронтенд и одинаковые акустические соглашения систематически
    занижают WER, поэтому цифру нельзя считать объективной оценкой разборчивости.
    Как появится второй независимый STT — добавьте его сюда, и метрика станет честной.
    """
    checkers = {}
    try:
        checkers["yandex"] = YandexSTT()
    except RuntimeError as exc:
        print(f"[пропуск STT-проверки] yandex: {exc}", file=sys.stderr)
    return checkers


def to_pcm16(audio_bytes: bytes, audio_format: str, fallback_rate: int = 16000) -> tuple[bytes, int]:
    if audio_format == "pcm16":
        return audio_bytes, fallback_rate
    return wav_bytes_to_pcm16(audio_bytes)  # wav16 и т.п. — контейнер с заголовком


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк синтеза речи")
    parser.add_argument("--n", type=int, default=10, help="Число тестовых фраз (источник — stt_tts_bench/dataset.yaml)")
    parser.add_argument("--source", choices=["dataset", "mic"], default="dataset",
                        help="dataset — фразы из корпуса, mic — эталонные тексты ваших записей")
    parser.add_argument("--refresh", action="store_true",
                        help="Перезагрузить тексты заново, игнорируя кэш")
    args = parser.parse_args()

    tts_providers = build_tts_providers()
    if not tts_providers:
        print("Нет ни одного настроенного TTS-провайдера — заполните .env", file=sys.stderr)
        sys.exit(1)

    stt_checkers = build_stt_checkers()
    if set(stt_checkers) <= set(tts_providers):
        print("ВНИМАНИЕ: разборчивость считается тем же вендором, что и синтезировал. "
              "WER будет занижен — сравнивайте его только с ручной оценкой на слух.",
              file=sys.stderr)

    if args.source == "mic":
        print("Беру эталонные тексты ваших записей с микрофона...")
        texts = load_mic_texts(n=args.n)
        if not texts:
            print("Записей с микрофона нет — запишите их в панели на вкладке «Микрофон»",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Загружаю {args.n} фраз из датасета (см. stt_tts_bench/dataset.yaml)...")
        texts = load_tts_texts(n=args.n, refresh=args.refresh)
    print(f"Загружено {len(texts)} фраз.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    samples_dir = RESULTS_DIR / f"tts_samples_{ts}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    review_rows = []
    manifest_items = []
    per_provider_latencies: dict[str, list[float]] = {name: [] for name in tts_providers}
    per_provider_wer: dict[str, list[float]] = {name: [] for name in tts_providers}
    per_provider_errors: dict[str, int] = {name: 0 for name in tts_providers}

    for i, text in enumerate(texts, 1):
        print(f"[{i}/{len(texts)}] {text[:70]!r}")
        row = {"idx": i, "text": text}
        item = {"id": i, "text": text, "providers": {}}
        for name, provider in tts_providers.items():
            result = provider.synthesize(text)
            row[f"{name}_latency_s"] = round(result.latency_s, 3)
            row[f"{name}_error"] = result.raw_error or ""
            per_provider_latencies[name].append(result.latency_s)

            audio_path = ""
            entry = {"audio": "", "latency_s": round(result.latency_s, 3),
                      "error": result.raw_error or "", "heard": {}}
            if result.raw_error or not result.audio_bytes:
                per_provider_errors[name] += 1
            else:
                pcm, rate = to_pcm16(result.audio_bytes, result.audio_format)
                provider_dir = samples_dir / name
                provider_dir.mkdir(exist_ok=True)
                rel_audio = f"{name}/{i:03d}.wav"
                audio_path = str(samples_dir / rel_audio)
                with open(audio_path, "wb") as f:
                    f.write(pcm16_to_wav_bytes(pcm, rate))
                entry["audio"] = rel_audio

                # Разборчивость: прогоняем синтез через STT и сверяем с исходным текстом
                wers_for_sample = []
                for checker_name, checker in stt_checkers.items():
                    stt_result = checker.recognize(pcm, sample_rate_hz=rate)
                    if not stt_result.raw_error:
                        w = wer(text, stt_result.text)
                        row[f"{name}_wer_by_{checker_name}"] = round(w, 4)
                        wers_for_sample.append(w)
                        entry["heard"][checker_name] = {"text": stt_result.text, "wer": round(w, 4)}
                    else:
                        row[f"{name}_wer_by_{checker_name}"] = ""
                if wers_for_sample:
                    per_provider_wer[name].extend(wers_for_sample)

            item["providers"][name] = entry
            row[f"{name}_audio_path"] = audio_path
            review_rows.append({
                "provider": name, "sample_id": i, "text": text, "audio_path": audio_path,
                "naturalness_1to5": "", "clarity_1to5": "", "notes": "",
            })
        rows.append(row)
        manifest_items.append(item)

    (samples_dir / "manifest.json").write_text(json.dumps(
        {"kind": "tts", "created": ts, "providers": list(tts_providers), "items": manifest_items},
        ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = RESULTS_DIR / f"tts_bench_{ts}.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nПодробные результаты: {out_csv}")
    print(f"Аудиосэмплы: {samples_dir}")

    review_csv = RESULTS_DIR / f"tts_manual_review_{ts}.csv"
    with review_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Лист для ручной оценки (прослушайте и заполните naturalness/clarity 1-5): {review_csv}")

    summary_rows = []
    for name in tts_providers:
        lat = latency_stats(per_provider_latencies[name])
        mean_wer = (sum(per_provider_wer[name]) / len(per_provider_wer[name])) if per_provider_wer[name] else float("nan")
        summary_rows.append([
            name, f"{mean_wer:.1%}" if per_provider_wer[name] else "—",
            f"{lat.mean_s:.2f}s", f"{lat.median_s:.2f}s", f"{lat.p95_s:.2f}s",
            per_provider_errors[name],
        ])

    print()
    print(tabulate(summary_rows, headers=["provider", "mean cross-WER (intelligibility)", "avg latency",
                                           "median latency", "p95 latency", "errors"], tablefmt="github"))


if __name__ == "__main__":
    main()
