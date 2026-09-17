"""Веб-панель: запуск бенчмарков, выбор моделей, просмотр логов, результатов и записей.

Два режима, переключаются наличием DATABASE_URL:

- **локальный** — слушает 127.0.0.1, хранит всё в файлах, креды в .env можно править прямо
  из панели. Пароль не нужен: снаружи никто не достучится.
- **облачный (Railway)** — слушает 0.0.0.0:$PORT, хранит логи и аудио в Postgres (файловая
  система контейнера эфемерна), креды берутся из переменных окружения Railway и в панели
  только отображаются как «задано/не задано» — значения наружу не отдаются никогда.

Просмотр результатов открыт всем, а всё, что тратит деньги или меняет конфигурацию,
закрыто паролем PANEL_PASSWORD.
"""
from __future__ import annotations

import hmac
import io
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_file, send_from_directory, session

from common.env import RESULTS_DIR
from . import db
from .storage import LOGS_DIR, build_storage

ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
MODELS_YAML = ROOT / "toolcalling_bench" / "models.yaml"
BENCHMARKS_YAML = WEBUI_DIR / "benchmarks.yaml"

RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FIELDS = ("id", "display_name", "model", "base_url", "api_key_env")
_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# какие переменные показывать в облаке как «креды проекта»
_CRED_HINTS = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY", "FOLDER_ID")

app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("PANEL_SECRET") or secrets.token_hex(32)

STORAGE = build_storage()
CLOUD = db.is_enabled()

_lock = threading.Lock()
_run = {"running": False, "run_id": None, "benchmark": None, "started_at": None,
         "exit_code": None, "log_name": None, "lines": [], "proc": None, "before": set(),
         "command": ""}


# ---------- доступ ----------

def password_required() -> bool:
    return bool(os.environ.get("PANEL_PASSWORD", "").strip())


def authenticated() -> bool:
    return not password_required() or session.get("auth") is True


def protected(view):
    """Закрывает всё, что тратит деньги или меняет конфигурацию."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not authenticated():
            return jsonify({"error": "нужен вход: операция защищена паролем"}), 401
        return view(*args, **kwargs)
    return wrapper


@app.post("/api/login")
def api_login():
    expected = os.environ.get("PANEL_PASSWORD", "")
    supplied = (request.json or {}).get("password", "")
    if not expected:
        return jsonify({"ok": True, "authenticated": True})
    if not hmac.compare_digest(supplied, expected):
        return jsonify({"error": "неверный пароль"}), 401
    session["auth"] = True
    return jsonify({"ok": True, "authenticated": True})


@app.post("/api/logout")
def api_logout():
    session.pop("auth", None)
    return jsonify({"ok": True})


@app.get("/api/session")
def api_session():
    return jsonify({"authenticated": authenticated(), "password_required": password_required(),
                     "cloud": CLOUD, "storage": STORAGE.kind})


# ---------- конфиги ----------

def read_benchmarks() -> list[dict]:
    return yaml.safe_load(BENCHMARKS_YAML.read_text(encoding="utf-8"))["benchmarks"]


def read_models_file() -> dict:
    return yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}


def write_models(models: list[dict], rate_limit: dict | None = None) -> None:
    cleaned = []
    for entry in models:
        row = {field: str(entry.get(field, "")).strip() for field in MODEL_FIELDS}
        if not row["id"]:
            raise ValueError("у каждой модели должен быть непустой id")
        cleaned.append(row)
    ids = [row["id"] for row in cleaned]
    if len(ids) != len(set(ids)):
        raise ValueError("id моделей должны быть уникальными")

    payload = read_models_file()
    payload["models"] = cleaned
    if rate_limit is not None:
        try:
            payload["rate_limit"] = {
                "requests_per_minute": max(0, int(rate_limit.get("requests_per_minute", 0))),
                "max_retries": max(0, int(rate_limit.get("max_retries", 0))),
            }
        except (TypeError, ValueError):
            raise ValueError("лимит частоты должен задаваться целыми числами")

    header = (
        "# Реестр моделей для бенчмарка tool-calling.\n"
        "# Файл редактируется из веб-панели (webui), можно править и вручную.\n"
        "# base_url — любой OpenAI-совместимый эндпоинт: прямой API вендора или шлюз-роутер.\n"
        "# rate_limit — общий на весь прогон: при работе через роутер лимит считается по ключу.\n\n"
    )
    MODELS_YAML.write_text(header + yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")


def read_env_vars() -> list[dict]:
    if not ENV_PATH.exists():
        return []
    out = []
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        m = _ENV_LINE_RE.match(line)
        if m:
            out.append({"name": m.group(1), "value": m.group(2).strip()})
    return out


def write_env_vars(updates: dict[str, str], deleted: list[str]) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        m = _ENV_LINE_RE.match(line)
        if not m:
            result.append(line)
            continue
        name = m.group(1)
        if name in deleted:
            continue
        if name in updates:
            result.append(f"{name}={updates[name]}")
            seen.add(name)
        else:
            result.append(line)
    for name, value in updates.items():
        if name not in seen:
            result.append(f"{name}={value}")
    ENV_PATH.write_text("\n".join(result) + "\n", encoding="utf-8")


# ---------- запуск бенчмарков ----------

def _reader_thread(proc: subprocess.Popen, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in proc.stdout:
            line = line.rstrip("\n")
            with _lock:
                _run["lines"].append(line)
            log_file.write(line + "\n")
            log_file.flush()
    proc.wait()
    with _lock:
        _run["running"] = False
        _run["exit_code"] = proc.returncode
        _run["proc"] = None
        snapshot = dict(_run)
    try:
        STORAGE.ingest_run(snapshot["run_id"], snapshot["benchmark"], snapshot["command"],
                            "\n".join(snapshot["lines"]), snapshot["exit_code"], snapshot["before"])
    except Exception as exc:  # падение переноса не должно ронять поток чтения
        print(f"[storage] не удалось сохранить прогон: {exc}", file=sys.stderr)


def start_run(benchmark_id: str, models: list[str], args: dict) -> str:
    benchmarks = {b["id"]: b for b in read_benchmarks()}
    bench = benchmarks.get(benchmark_id)
    if bench is None:
        raise ValueError(f"неизвестный бенчмарк: {benchmark_id}")

    cmd = [sys.executable, "-m", bench["module"]]
    if bench.get("supports_models") and models:
        cmd += ["--models", ",".join(models)]
    uses_mic = False
    for arg_spec in bench.get("args", []):
        raw = args.get(arg_spec["name"], arg_spec.get("default"))
        if raw in (None, ""):
            continue
        if arg_spec.get("type") == "number":
            value = str(int(raw))
        else:
            value = str(raw)
            # значение из запроса не должно попадать в командную строку без проверки
            allowed = [str(o["value"]) for o in arg_spec.get("options", [])]
            if allowed and value not in allowed:
                raise ValueError(f"недопустимое значение {arg_spec['name']}={value!r}; "
                                  f"ожидалось одно из {allowed}")
            if arg_spec["name"] == "source" and value == "mic":
                uses_mic = True
        cmd += [arg_spec["flag"], value]

    if uses_mic:
        STORAGE.prepare_mic_for_run()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{benchmark_id}_{run_id}.log"
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
    )
    with _lock:
        _run.update({"running": True, "run_id": run_id, "benchmark": benchmark_id,
                      "started_at": time.time(), "exit_code": None, "log_name": log_name,
                      "lines": [f"$ {' '.join(cmd)}", ""], "proc": proc,
                      "before": STORAGE.snapshot(), "command": " ".join(cmd)})
    threading.Thread(target=_reader_thread, args=(proc, LOGS_DIR / log_name), daemon=True).start()
    return run_id


# ---------- API: только чтение, открыто всем ----------

@app.get("/")
def index():
    return send_from_directory(WEBUI_DIR, "index.html")


@app.get("/api/benchmarks")
def api_benchmarks():
    return jsonify(read_benchmarks())


@app.get("/api/models")
def api_get_models():
    data = read_models_file()
    return jsonify({"models": data.get("models", []), "rate_limit": data.get("rate_limit", {})})


@app.get("/api/logs")
def api_logs():
    return jsonify(STORAGE.list_logs())


@app.get("/api/logs/<name>")
def api_log_content(name: str):
    found = STORAGE.read_log(name)
    return jsonify(found) if found else (jsonify({"error": "лог не найден"}), 404)


@app.get("/api/results")
def api_results():
    return jsonify(STORAGE.list_results())


@app.get("/api/results/<name>")
def api_result_content(name: str):
    found = STORAGE.read_result(name)
    return jsonify(found) if found else (jsonify({"error": "файл не найден"}), 404)


@app.get("/api/audio-sets")
def api_audio_sets():
    return jsonify(STORAGE.list_audio_sets())


@app.get("/api/audio-sets/<name>")
def api_audio_set(name: str):
    found = STORAGE.read_audio_set(name)
    return jsonify(found) if found else (jsonify({"error": "набор не найден"}), 404)


@app.get("/api/audio/<name>/<path:relative>")
def api_audio(name: str, relative: str):
    data = STORAGE.read_audio(name, relative)
    if data is None:
        return jsonify({"error": "файл не найден"}), 404
    return send_file(io.BytesIO(data), mimetype="audio/wav")


@app.get("/api/mic")
def api_mic_list():
    return jsonify(STORAGE.list_mic())


@app.get("/api/mic/<sample_id>/audio")
def api_mic_audio(sample_id: str):
    data = STORAGE.read_mic_audio(sample_id)
    if data is None:
        return jsonify({"error": "запись не найдена"}), 404
    return send_file(io.BytesIO(data), mimetype="audio/wav")


@app.get("/api/run/log")
def api_run_log():
    offset = request.args.get("offset", default=0, type=int)
    with _lock:
        lines = _run["lines"][offset:]
        # offset возвращаем эхом: клиент по нему отбрасывает ответ на устаревший запрос
        return jsonify({"offset": offset, "lines": lines, "next_offset": offset + len(lines),
                         "running": _run["running"], "exit_code": _run["exit_code"],
                         "benchmark": _run["benchmark"], "log_name": _run["log_name"]})


# ---------- API: креды ----------

@app.get("/api/env")
def api_get_env():
    """В облаке значения не отдаются никогда — только факт, что переменная задана."""
    referenced = sorted({m.get("api_key_env", "") for m in read_models_file().get("models", [])
                          if m.get("api_key_env")})
    if CLOUD:
        names = sorted({n for n in os.environ if any(h in n for h in _CRED_HINTS)} | set(referenced))
        return jsonify({
            "readonly": True,
            "vars": [{"name": n, "is_set": bool(os.environ.get(n, "").strip())} for n in names],
            "missing": [n for n in referenced if not os.environ.get(n, "").strip()],
        })

    if not authenticated():
        return jsonify({"error": "нужен вход: креды защищены паролем"}), 401
    existing = {v["name"] for v in read_env_vars()}
    return jsonify({"readonly": False, "vars": read_env_vars(),
                     "missing": [n for n in referenced if n not in existing]})


@app.put("/api/env")
@protected
def api_put_env():
    if CLOUD:
        return jsonify({"error": "в облаке креды задаются переменными окружения Railway, "
                                  "а не через панель"}), 403
    payload = request.json or {}
    updates = {}
    for name, value in (payload.get("vars") or {}).items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return jsonify({"error": f"недопустимое имя переменной: {name}"}), 400
        updates[name] = str(value).replace("\n", " ").replace("\r", " ").strip()
    write_env_vars(updates, payload.get("deleted") or [])
    return jsonify({"ok": True})


# ---------- API: изменяющие операции ----------

@app.put("/api/models")
@protected
def api_put_models():
    payload = request.json or {}
    try:
        write_models(payload.get("models", []), payload.get("rate_limit"))
    except (ValueError, yaml.YAMLError) as exc:
        return jsonify({"error": str(exc)}), 400
    data = read_models_file()
    return jsonify({"ok": True, "models": data.get("models", []),
                     "rate_limit": data.get("rate_limit", {})})


@app.post("/api/run")
@protected
def api_run():
    with _lock:
        if _run["running"]:
            return jsonify({"error": "прогон уже идёт — дождитесь окончания или остановите его"}), 409
    payload = request.json or {}
    try:
        run_id = start_run(payload.get("benchmark", ""), payload.get("models") or [],
                            payload.get("args") or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "run_id": run_id})


@app.post("/api/run/stop")
@protected
def api_run_stop():
    with _lock:
        proc = _run["proc"]
    if proc is None:
        return jsonify({"error": "нет активного прогона"}), 400
    proc.terminate()
    return jsonify({"ok": True})


@app.post("/api/mic")
@protected
def api_mic_add():
    reference = (request.form.get("reference_text") or "").strip()
    upload = request.files.get("audio")
    if upload is None:
        return jsonify({"error": "нет файла записи"}), 400
    try:
        entry = STORAGE.add_mic(upload.read(), reference)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"ok": True, "sample": entry})


@app.delete("/api/mic/<sample_id>")
@protected
def api_mic_delete(sample_id: str):
    if not STORAGE.delete_mic(Path(sample_id).name):
        return jsonify({"error": "запись не найдена"}), 404
    return jsonify({"ok": True})


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    # 0.0.0.0 только когда платформа сама назначила порт: локально панель остаётся
    # доступной лишь с этой машины
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    print(f"Панель бенчмарков: http://{host}:{port}")
    print(f"Хранилище: {STORAGE.kind}; пароль {'включён' if password_required() else 'не задан'}")
    if host == "0.0.0.0" and not password_required():
        print("ВНИМАНИЕ: панель доступна снаружи без пароля — задайте PANEL_PASSWORD",
              file=sys.stderr)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
