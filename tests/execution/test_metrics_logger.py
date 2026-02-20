from __future__ import annotations

import pathlib
import sqlite3
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.metrics_logger as metrics_logger


def _patch_db_settings(tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
    db_path = tmp_path / "metrics.db"
    monkeypatch.setattr(metrics_logger, "_DB_PATH", db_path)
    monkeypatch.setattr(metrics_logger, "_SCHEMA_READY", False)
    monkeypatch.setattr(metrics_logger, "_DB_BUSY_TIMEOUT_MS", 120)
    monkeypatch.setattr(metrics_logger, "_DB_WRITE_RETRIES", 2)
    monkeypatch.setattr(metrics_logger, "_DB_RETRY_BASE_SLEEP_SEC", 0.0)
    monkeypatch.setattr(metrics_logger, "_DB_RETRY_MAX_SLEEP_SEC", 0.0)
    return db_path


def test_log_metric_writes_when_db_available(tmp_path: pathlib.Path, monkeypatch) -> None:
    db_path = _patch_db_settings(tmp_path, monkeypatch)

    ok = metrics_logger.log_metric(
        "range_mode_active",
        1.0,
        tags={"source": "test"},
    )

    assert ok is True
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            "SELECT metric, value FROM metrics ORDER BY rowid DESC LIMIT 1",
        ).fetchone()
    finally:
        con.close()
    assert row == ("range_mode_active", 1.0)


def test_log_metric_returns_false_when_db_is_locked(tmp_path: pathlib.Path, monkeypatch) -> None:
    db_path = _patch_db_settings(tmp_path, monkeypatch)

    lock_con = sqlite3.connect(str(db_path))
    lock_con.execute("PRAGMA journal_mode=WAL;")
    lock_con.execute(
        "CREATE TABLE IF NOT EXISTS metrics (ts TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL, tags TEXT)",
    )
    lock_con.commit()
    lock_con.execute("BEGIN IMMEDIATE;")
    try:
        ok = metrics_logger.log_metric(
            "range_mode_active",
            0.0,
            tags={"source": "lock-test"},
        )
    finally:
        lock_con.rollback()
        lock_con.close()

    assert ok is False
