import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("logs/signals.db")
_LOCK = threading.Lock()
_FALSEY = {"", "0", "false", "no"}


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    conn.execute("PRAGMA journal_mode=WAL;")


def enqueue(signal: Dict[str, Any]) -> None:
    """Store a signal dict for arbiter consumption."""
    # Safety: enqueuing when the gate is disabled will stall orders (workers will return early).
    # Require SIGNAL_GATE_ENABLED=1 for queue usage.
    if os.getenv("SIGNAL_GATE_ENABLED", "0").strip().lower() in _FALSEY:
        raise RuntimeError("signal_gate_disabled")
    ts_ms = int(time.time() * 1000)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, sqlite3.connect(DB_PATH) as conn:
        _ensure_table(conn)
        conn.execute(
            "INSERT INTO signals (ts_ms, payload) VALUES (?, ?)",
            (ts_ms, json.dumps(signal)),
        )
        conn.commit()


def fetch(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch up to N signals (oldest first) and remove them."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, sqlite3.connect(DB_PATH) as conn:
        _ensure_table(conn)
        # Drop stale signals to avoid acting on old intent after downtime.
        try:
            max_age_sec = float(os.getenv("SIGNAL_BUS_MAX_AGE_SEC", "60"))
        except Exception:
            max_age_sec = 60.0
        if max_age_sec > 0:
            cutoff_ms = int(time.time() * 1000 - max_age_sec * 1000)
            conn.execute("DELETE FROM signals WHERE ts_ms < ?", (cutoff_ms,))
            conn.commit()
        rows = conn.execute(
            "SELECT id, payload FROM signals ORDER BY id ASC LIMIT ?",
            (limit,),
        ).fetchall()
        ids = [row[0] for row in rows]
        signals: List[Dict[str, Any]] = []
        for _, payload in rows:
            try:
                signals.append(json.loads(payload))
            except Exception:
                continue
        if ids:
            conn.execute(
                f"DELETE FROM signals WHERE id IN ({','.join('?' for _ in ids)})",
                ids,
            )
            conn.commit()
        return signals
