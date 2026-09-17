"""Подключение к Postgres и схема.

База включается наличием DATABASE_URL (Railway подставляет её сам). Без неё панель
работает как раньше — с файлами на диске, см. storage.py. Так локальный запуск
не требует ни базы, ни докера.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    benchmark   TEXT NOT NULL,
    command     TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    exit_code   INTEGER,
    log         TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS result_files (
    name       TEXT PRIMARY KEY,
    run_id     TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    content    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_sets (
    name       TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    manifest   JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_files (
    set_name TEXT NOT NULL REFERENCES audio_sets(name) ON DELETE CASCADE,
    path     TEXT NOT NULL,
    data     BYTEA NOT NULL,
    PRIMARY KEY (set_name, path)
);

CREATE TABLE IF NOT EXISTS mic_samples (
    sample_id      TEXT PRIMARY KEY,
    reference_text TEXT NOT NULL,
    sample_rate    INTEGER NOT NULL,
    duration_s     REAL NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    data           BYTEA NOT NULL
);
"""


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url or None


def is_enabled() -> bool:
    return database_url() is not None


@contextmanager
def connect():
    url = database_url()
    if url is None:
        raise RuntimeError("DATABASE_URL не задан")
    with psycopg.connect(url, row_factory=dict_row) as conn:
        yield conn


def init_schema() -> None:
    with connect() as conn:
        conn.execute(_SCHEMA)
        conn.commit()
