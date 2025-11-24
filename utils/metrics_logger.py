from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

_DB_PATH = Path("logs/metrics.db")
_LOCK = threading.Lock()


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            ts TEXT NOT NULL,
            metric TEXT NOT NULL,
            value REAL NOT NULL,
            tags TEXT
        )
        """
    )


def log_metric(
    metric: str,
    value: float,
    *,
    tags: Optional[Mapping[str, object]] = None,
    ts: Optional[datetime] = None,
) -> None:
    payload = {
        "metric": metric,
        "value": float(value),
        "ts": (ts or datetime.utcnow()).isoformat(),
        "tags": json.dumps(tags or {}, ensure_ascii=True),
    }
    attempts = 0
    with _LOCK:
        while attempts < 2:
            attempts += 1
            con = sqlite3.connect(_DB_PATH, timeout=1.5)
            try:
                con.execute("PRAGMA busy_timeout=1500;")
                con.execute("PRAGMA journal_mode=WAL;")
                _ensure_schema(con)
                con.execute(
                    "INSERT INTO metrics(ts, metric, value, tags) VALUES (:ts, :metric, :value, :tags)",
                    payload,
                )
                con.commit()
                return
            except sqlite3.OperationalError as exc:
                if attempts >= 2:
                    logging.debug("[metrics] drop metric=%s due to lock: %s", metric, exc)
            except Exception as exc:  # pragma: no cover - defensive
                logging.debug("[metrics] drop metric=%s due to error: %s", metric, exc)
                return
            finally:
                con.close()
