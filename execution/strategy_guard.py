"""Shared accessors for per-strategy cooldown state."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

_DB_PATH = Path("logs/stage_state.db")
_CONN: Optional[sqlite3.Connection] = None
_LOCK = Lock()


def _ensure_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    with _LOCK:
        if _CONN is None:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(_DB_PATH)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_cooldown (
                    strategy TEXT PRIMARY KEY,
                    reason TEXT,
                    cooldown_until TEXT NOT NULL
                )
                """
            )
            _CONN = conn
    return _CONN  # type: ignore[return-value]


def _as_naive_utc(dt: datetime) -> datetime:
    """Normalize datetime to naive UTC to avoid aware/naive comparison issues."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def set_block(strategy: str, seconds: int, reason: str) -> None:
    """Set a cooldown for the strategy with a given duration (in seconds)."""
    if not strategy or seconds <= 0:
        return
    conn = _ensure_conn()
    until = _as_naive_utc(datetime.utcnow()) + timedelta(seconds=max(1, seconds))
    with _LOCK:
        conn.execute(
            """
            INSERT INTO strategy_cooldown(strategy, reason, cooldown_until)
            VALUES(?,?,?)
            ON CONFLICT(strategy) DO UPDATE SET reason=excluded.reason, cooldown_until=excluded.cooldown_until
            """,
            (strategy, reason, until.isoformat()),
        )
        conn.commit()


def is_blocked(strategy: str, now: Optional[datetime] = None) -> Tuple[bool, Optional[int], Optional[str]]:
    """Return (blocked, remaining_seconds, reason) for the strategy."""
    if not strategy:
        return False, None, None
    conn = _ensure_conn()
    row = conn.execute(
        "SELECT reason, cooldown_until FROM strategy_cooldown WHERE strategy=?",
        (strategy,),
    ).fetchone()
    if not row:
        return False, None, None
    limit = _as_naive_utc(datetime.fromisoformat(row[1]))
    current = _as_naive_utc(now or datetime.utcnow())
    if current >= limit:
        with _LOCK:
            conn.execute("DELETE FROM strategy_cooldown WHERE strategy=?", (strategy,))
            conn.commit()
        return False, None, None
    remaining = int((limit - current).total_seconds())
    return True, max(1, remaining), row[0] or ""


def clear_expired(now: Optional[datetime] = None) -> None:
    conn = _ensure_conn()
    current = _as_naive_utc(now or datetime.utcnow()).isoformat()
    with _LOCK:
        conn.execute("DELETE FROM strategy_cooldown WHERE cooldown_until <= ?", (current,))
        conn.commit()
