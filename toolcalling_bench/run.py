"""CLI: бенчмарк tool-calling для qwen/glm/mistral/deepseek/minimax.

Запуск:
    python -m toolcalling_bench.run
    python -m toolcalling_bench.run --models qwen,deepseek   # только часть моделей
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict

from tabulate import tabulate

from common.env import RESULTS_DIR
from .evaluator import run_task
from .providers import RateLimiter, build_client, load_model_configs, load_rate_limit
from .suites import SUITES, load_suite, suite_title


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-calling бенчмарк для текстовых моделей")
    parser.add_argument("--models", type=str, default="", help="Список id моделей через запятую (по умолчанию — все из models.yaml)")
    parser.add_argument("--suite", type=str, default="basic", choices=list(SUITES),
                        help="Набор задач: basic — проверка механики, hard — различающий модели")
    args = parser.parse_args()

    all_configs = load_model_configs()
    wanted = {m.strip() for m in args.models.split(",") if m.strip()} or None
    configs = [c for c in all_configs if wanted is None or c.id in wanted]

    tasks = load_suite(args.suite)
    print(f"{suite_title(args.suite)}: {len(tasks)} задач")
    rate_limit = load_rate_limit()
    limiter = RateLimiter(rate_limit.requests_per_minute)
    if rate_limit.requests_per_minute:
        print(f"Ограничение частоты: {rate_limit.requests_per_minute} запр./мин "
              f"(~{limiter.min_interval:.1f}s между запросами), повторов при 429: {rate_limit.max_retries}")

    no_call_task_ids = {t.id for t in tasks if t.no_call_expected}
    rows = []
    per_model_results = defaultdict(list)

    for cfg in configs:
        client = build_client(cfg)
        if client is None:
            print(f"[пропуск] {cfg.id}: нет ключа в {cfg.api_key_env}", file=sys.stderr)
            continue
        print(f"\n=== {cfg.display_name} ({cfg.model}) ===")
        for task in tasks:
            result = run_task(client, cfg.model, task, limiter, rate_limit.max_retries)
            status = "OK" if result.passed else "FAIL"
            note = result.api_error or result.detail
            print(f"  [{status}] {task.id} ({result.latency_s:.2f}s) — {note}")
            rows.append({
                "suite": args.suite, "model_id": cfg.id,
                "model_display_name": cfg.display_name, "task_id": task.id,
                "passed": result.passed, "json_valid": result.json_valid,
                "latency_s": round(result.latency_s, 3), "detail": result.detail,
                "api_error": result.api_error or "",
            })
            per_model_results[cfg.id].append(result)

    if not rows:
        print("Ни одна модель не была прогнана — проверьте .env / --models", file=sys.stderr)
        sys.exit(1)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"toolcalling_{args.suite}_{ts}.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nПодробные результаты: {out_csv}")

    summary_rows = []
    for cfg in configs:
        results = per_model_results.get(cfg.id)
        if not results:
            continue
        n = len(results)
        passed = sum(1 for r in results if r.passed)
        api_errors = sum(1 for r in results if r.api_error)
        avg_latency = sum(r.latency_s for r in results) / n

        # JSON-валидность и ложные срабатывания осмысленны только там, где модель реально ответила
        answered = [r for r in results if r.json_valid is not None]
        json_cell = f"{sum(1 for r in answered if r.json_valid)}/{len(answered)}" if answered else "—"
        no_call_checked = [r for r in results if r.task_id in no_call_task_ids and not r.api_error]
        if not no_call_checked:
            false_positive = "—"
        else:
            failed = sum(1 for r in no_call_checked if not r.passed)
            false_positive = f"{failed}/{len(no_call_checked)}"

        summary_rows.append([
            cfg.display_name, f"{passed}/{n}", json_cell, f"{avg_latency:.2f}s",
            api_errors, false_positive,
        ])

    print()
    print(tabulate(summary_rows, headers=["model", "passed tasks", "valid JSON args", "avg latency",
                                           "API errors", "лишних вызовов"], tablefmt="github"))

    failures = [r for r in rows if not r["passed"] and not r["api_error"]]
    if failures:
        print("\nРазбор провалов:")
        for row in failures:
            print(f"  {row['model_display_name']} / {row['task_id']}: {row['detail']}")


if __name__ == "__main__":
    main()
