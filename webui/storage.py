"""Хранилище артефактов прогона: логи, CSV-результаты, аудио, записи с микрофона.

Две реализации за одним интерфейсом:
- FileStorage — каталоги results/ и data/, как было изначально; работает без базы;
- PostgresStorage — то же самое в БД, для облака, где файловая система эфемерная
  и всё написанное исчезает при следующем деплое.

Выбор по наличию DATABASE_URL. Сами бенчмарки про хранилище не знают: они как и раньше
пишут файлы, а сервер после завершения прогона забирает появившееся в базу.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common.env import DATA_DIR, RESULTS_DIR
from stt_tts_bench import mic_samples
from . import db

LOGS_DIR = RESULTS_DIR / "logs"


def _csv_to_table(text: str) -> dict:
    rows = list(csv.reader(io.StringIO(text)))
    return {"header": rows[0] if rows else [], "rows": rows[1:]}


def _safe_name(name: str) -> str:
    return Path(name).name


class FileStorage:
    kind = "files"

    # ---------- логи ----------
    def list_logs(self) -> list[dict]:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"name": p.name, "size": p.stat().st_size,
                  "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
                 for p in files]

    def read_log(self, name: str) -> dict | None:
        path = LOGS_DIR / _safe_name(name)
        if not path.is_file():
            return None
        return {"name": path.name, "content": path.read_text(encoding="utf-8", errors="replace")}

    # ---------- результаты ----------
    def list_results(self) -> list[dict]:
        files = sorted(RESULTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"name": p.name, "size": p.stat().st_size,
                  "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
                 for p in files]

    def read_result(self, name: str) -> dict | None:
        path = RESULTS_DIR / _safe_name(name)
        if not path.is_file():
            return None
        table = _csv_to_table(path.read_text(encoding="utf-8-sig"))
        return {"name": path.name, **table}

    # ---------- аудио ----------
    def list_audio_sets(self) -> list[dict]:
        sets = []
        for path in sorted(RESULTS_DIR.glob("*_samples_*"), key=lambda p: p.name, reverse=True):
            manifest = path / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            sets.append({"name": path.name, "kind": data.get("kind", "?"),
                          "created": data.get("created", ""),
                          "providers": data.get("providers", []),
                          "count": len(data.get("items", []))})
        return sets

    def read_audio_set(self, name: str) -> dict | None:
        manifest = RESULTS_DIR / _safe_name(name) / "manifest.json"
        if not manifest.is_file():
            return None
        return json.loads(manifest.read_text(encoding="utf-8"))

    def read_audio(self, set_name: str, relative: str) -> bytes | None:
        root = (RESULTS_DIR / _safe_name(set_name)).resolve()
        candidate = (root / relative).resolve()
        if not candidate.is_file() or root not in candidate.parents:
            return None
        return candidate.read_bytes()

    # ---------- микрофон ----------
    def list_mic(self) -> list[dict]:
        return mic_samples.list_samples()

    def add_mic(self, wav_bytes: bytes, reference_text: str) -> dict:
        return mic_samples.add_sample(wav_bytes, reference_text)

    def delete_mic(self, sample_id: str) -> bool:
        return mic_samples.delete_sample(sample_id)

    def read_mic_audio(self, sample_id: str) -> bytes | None:
        path = mic_samples.sample_path(_safe_name(sample_id))
        return path.read_bytes() if path and path.is_file() else None

    # ---------- жизненный цикл прогона ----------
    def snapshot(self) -> set:
        return set()

    def ingest_run(self, run_id, benchmark, command, log, exit_code, before) -> None:
        """Файлы уже лежат на диске — переносить некуда."""

    def prepare_mic_for_run(self) -> None:
        """Записи уже на диске."""


class PostgresStorage:
    kind = "postgres"

    def __init__(self):
        db.init_schema()

    # ---------- логи ----------
    def list_logs(self) -> list[dict]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT run_id, benchmark, started_at, exit_code, length(log) AS size "
                "FROM runs ORDER BY started_at DESC LIMIT 200").fetchall()
        return [{"name": f"{r['benchmark']}_{r['run_id']}.log", "size": r["size"] or 0,
                  "modified": r["started_at"].strftime("%Y-%m-%d %H:%M:%S")} for r in rows]

    def read_log(self, name: str) -> dict | None:
        run_id = Path(name).stem.split("_", 1)[-1]
        with db.connect() as conn:
            row = conn.execute("SELECT log FROM runs WHERE run_id = %s", (run_id,)).fetchone()
        return {"name": name, "content": row["log"]} if row else None

    # ---------- результаты ----------
    def list_results(self) -> list[dict]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT name, created_at, length(content) AS size "
                "FROM result_files ORDER BY created_at DESC LIMIT 200").fetchall()
        return [{"name": r["name"], "size": r["size"],
                  "modified": r["created_at"].strftime("%Y-%m-%d %H:%M:%S")} for r in rows]

    def read_result(self, name: str) -> dict | None:
        with db.connect() as conn:
            row = conn.execute("SELECT name, content FROM result_files WHERE name = %s",
                                (name,)).fetchone()
        if not row:
            return None
        return {"name": row["name"], **_csv_to_table(row["content"])}

    # ---------- аудио ----------
    def list_audio_sets(self) -> list[dict]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT name, kind, created_at, manifest FROM audio_sets "
                "ORDER BY created_at DESC LIMIT 200").fetchall()
        return [{"name": r["name"], "kind": r["kind"],
                  "created": r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
                  "providers": r["manifest"].get("providers", []),
                  "count": len(r["manifest"].get("items", []))} for r in rows]

    def read_audio_set(self, name: str) -> dict | None:
        with db.connect() as conn:
            row = conn.execute("SELECT manifest FROM audio_sets WHERE name = %s", (name,)).fetchone()
        return row["manifest"] if row else None

    def read_audio(self, set_name: str, relative: str) -> bytes | None:
        with db.connect() as conn:
            row = conn.execute("SELECT data FROM audio_files WHERE set_name = %s AND path = %s",
                                (set_name, relative)).fetchone()
        return bytes(row["data"]) if row else None

    # ---------- микрофон ----------
    def list_mic(self) -> list[dict]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT sample_id, reference_text, sample_rate, duration_s, created_at "
                "FROM mic_samples ORDER BY created_at").fetchall()
        return [{"sample_id": r["sample_id"], "reference_text": r["reference_text"],
                  "sample_rate": r["sample_rate"], "duration_s": round(r["duration_s"], 2),
                  "created": r["created_at"].strftime("%Y-%m-%d %H:%M:%S")} for r in rows]

    def add_mic(self, wav_bytes: bytes, reference_text: str) -> dict:
        entry = mic_samples.add_sample(wav_bytes, reference_text)  # валидация + запись на диск
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO mic_samples (sample_id, reference_text, sample_rate, duration_s, data) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (sample_id) DO NOTHING",
                (entry["sample_id"], entry["reference_text"], entry["sample_rate"],
                 entry["duration_s"], wav_bytes))
            conn.commit()
        return entry

    def delete_mic(self, sample_id: str) -> bool:
        mic_samples.delete_sample(sample_id)
        with db.connect() as conn:
            deleted = conn.execute("DELETE FROM mic_samples WHERE sample_id = %s",
                                    (sample_id,)).rowcount
            conn.commit()
        return deleted > 0

    def read_mic_audio(self, sample_id: str) -> bytes | None:
        with db.connect() as conn:
            row = conn.execute("SELECT data FROM mic_samples WHERE sample_id = %s",
                                (sample_id,)).fetchone()
        return bytes(row["data"]) if row else None

    # ---------- жизненный цикл прогона ----------
    def snapshot(self) -> set:
        """Какие артефакты лежали до запуска — всё появившееся потом принадлежит этому прогону."""
        existing = set(RESULTS_DIR.glob("*.csv")) | set(RESULTS_DIR.glob("*_samples_*"))
        return {p.name for p in existing}

    def ingest_run(self, run_id, benchmark, command, log, exit_code, before) -> None:
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, benchmark, command, finished_at, exit_code, log) "
                "VALUES (%s, %s, %s, now(), %s, %s) "
                "ON CONFLICT (run_id) DO UPDATE SET finished_at = now(), "
                "exit_code = EXCLUDED.exit_code, log = EXCLUDED.log",
                (run_id, benchmark, command, exit_code, log))

            new_csv = [p for p in RESULTS_DIR.glob("*.csv") if p.name not in before]
            for path in new_csv:
                conn.execute(
                    "INSERT INTO result_files (name, run_id, content) VALUES (%s, %s, %s) "
                    "ON CONFLICT (name) DO UPDATE SET content = EXCLUDED.content",
                    (path.name, run_id, path.read_text(encoding="utf-8-sig")))

            new_sets = [p for p in RESULTS_DIR.glob("*_samples_*")
                        if p.is_dir() and p.name not in before and (p / "manifest.json").is_file()]
            for directory in new_sets:
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                manifest["providers"] = manifest.get("providers", [])
                conn.execute(
                    "INSERT INTO audio_sets (name, kind, manifest) VALUES (%s, %s, %s) "
                    "ON CONFLICT (name) DO UPDATE SET manifest = EXCLUDED.manifest",
                    (directory.name, manifest.get("kind", "?"), json.dumps(manifest, ensure_ascii=False)))
                for wav in directory.rglob("*.wav"):
                    conn.execute(
                        "INSERT INTO audio_files (set_name, path, data) VALUES (%s, %s, %s) "
                        "ON CONFLICT (set_name, path) DO UPDATE SET data = EXCLUDED.data",
                        (directory.name, wav.relative_to(directory).as_posix(), wav.read_bytes()))
            conn.commit()

        # диск в контейнере всё равно эфемерный: чистим только то, что уже уехало в базу
        for path in new_csv:
            path.unlink(missing_ok=True)
        for directory in new_sets:
            shutil.rmtree(directory, ignore_errors=True)

    def prepare_mic_for_run(self) -> None:
        """Раннеры читают записи с диска, поэтому перед прогоном выкладываем их из базы."""
        entries = []
        mic_samples.MIC_DIR.mkdir(parents=True, exist_ok=True)
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT sample_id, reference_text, sample_rate, duration_s, data "
                "FROM mic_samples ORDER BY created_at").fetchall()
        for row in rows:
            wav_name = f"{row['sample_id']}.wav"
            (mic_samples.MIC_DIR / wav_name).write_bytes(bytes(row["data"]))
            entries.append({"sample_id": row["sample_id"], "reference_text": row["reference_text"],
                             "wav": wav_name, "sample_rate": row["sample_rate"],
                             "duration_s": row["duration_s"], "created": ""})
        (mic_samples.MIC_DIR / "manifest.json").write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def build_storage() -> FileStorage | PostgresStorage:
    return PostgresStorage() if db.is_enabled() else FileStorage()
