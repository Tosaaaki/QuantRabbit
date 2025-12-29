import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

DB_PATH = Path("logs/signals.db")
_LOCK = threading.Lock()


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
