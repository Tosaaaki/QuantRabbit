"""Shared accessors for per-strategy cooldown state."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

_DB_PATH = Path("logs/stage_state.db")
_CONN: Optional[sqlite3.Connection] = None
_LOCK = Lock()
_BUSY_TIMEOUT_MS = max(
    1000,
    int(os.getenv("STRATEGY_GUARD_DB_BUSY_TIMEOUT_MS", "12000")),
)
_LOCK_RETRY = max(
    1,
    int(os.getenv("STRATEGY_GUARD_DB_LOCK_RETRY", "3")),
)
_LOCK_RETRY_SLEEP_SEC = max(
    0.0,
    float(os.getenv("STRATEGY_GUARD_DB_LOCK_RETRY_SLEEP_SEC", "0.08")),
)


def _is_locked_error(exc: Exception) -> bool:
    msg = str(exc).strip().lower()
    return "locked" in msg or "busy" in msg


def _safe_execute(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...] = (),
) -> sqlite3.Cursor:
    for attempt in range(_LOCK_RETRY):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if not _is_locked_error(exc) or attempt + 1 >= _LOCK_RETRY:
                raise
            sleep_sec = _LOCK_RETRY_SLEEP_SEC * float(attempt + 1)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    raise sqlite3.OperationalError("database is locked")


def _ensure_conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is not None:
        return _CONN
    with _LOCK:
        if _CONN is None:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                _DB_PATH,
                timeout=_BUSY_TIMEOUT_MS / 1000.0,
                check_same_thread=False,
                isolation_level=None,
            )
            try:
                conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            except sqlite3.Error:
                pass
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            _safe_execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS strategy_cooldown (
                    strategy TEXT PRIMARY KEY,
                    reason TEXT,
                    cooldown_until TEXT NOT NULL
                )
                """,
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
    try:
        conn = _ensure_conn()
        until = _as_naive_utc(datetime.utcnow()) + timedelta(seconds=max(1, seconds))
        with _LOCK:
            _safe_execute(
                conn,
                """
                INSERT INTO strategy_cooldown(strategy, reason, cooldown_until)
                VALUES(?,?,?)
                ON CONFLICT(strategy) DO UPDATE SET reason=excluded.reason, cooldown_until=excluded.cooldown_until
                """,
                (strategy, reason, until.isoformat()),
            )
    except sqlite3.Error as exc:
        if _is_locked_error(exc):
            logging.debug("[strategy_guard] set_block lock: %s", exc)
            return
        logging.warning("[strategy_guard] set_block failed: %s", exc)


def is_blocked(strategy: str, now: Optional[datetime] = None) -> Tuple[bool, Optional[int], Optional[str]]:
    """Return (blocked, remaining_seconds, reason) for the strategy."""
    if not strategy:
        return False, None, None
    try:
        conn = _ensure_conn()
        with _LOCK:
            row = _safe_execute(
                conn,
                "SELECT reason, cooldown_until FROM strategy_cooldown WHERE strategy=?",
                (strategy,),
            ).fetchone()
    except sqlite3.Error as exc:
        if _is_locked_error(exc):
            logging.debug("[strategy_guard] is_blocked lock: %s", exc)
            return False, None, None
        logging.warning("[strategy_guard] is_blocked failed: %s", exc)
        return False, None, None
    if not row:
        return False, None, None
    limit = _as_naive_utc(datetime.fromisoformat(row[1]))
    current = _as_naive_utc(now or datetime.utcnow())
    if current >= limit:
        try:
            with _LOCK:
                _safe_execute(
                    conn,
                    "DELETE FROM strategy_cooldown WHERE strategy=?",
                    (strategy,),
                )
        except sqlite3.Error as exc:
            if _is_locked_error(exc):
                logging.debug("[strategy_guard] delete expired lock: %s", exc)
            else:
                logging.warning("[strategy_guard] delete expired failed: %s", exc)
        return False, None, None
    remaining = int((limit - current).total_seconds())
    return True, max(1, remaining), row[0] or ""


def clear_expired(now: Optional[datetime] = None) -> None:
    try:
        conn = _ensure_conn()
        current = _as_naive_utc(now or datetime.utcnow()).isoformat()
        with _LOCK:
            _safe_execute(
                conn,
                "DELETE FROM strategy_cooldown WHERE cooldown_until <= ?",
                (current,),
            )
    except sqlite3.Error as exc:
        if _is_locked_error(exc):
            logging.debug("[strategy_guard] clear_expired lock: %s", exc)
            return
        logging.warning("[strategy_guard] clear_expired failed: %s", exc)
