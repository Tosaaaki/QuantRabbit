"""
execution.micro_guard
~~~~~~~~~~~~~~~~~~~~~~

Centralised guardrails for micro pocket re-entries.
Keeps SQLite lookups cached so strategies and risk guard
can enforce loss-driven cool-downs without duplicating logic.
"""

from __future__ import annotations

import datetime
import os
import pathlib
import sqlite3
import time
from typing import Any, Dict

_DB_PATH = pathlib.Path("logs/trades.db")

try:
    _CON = sqlite3.connect(_DB_PATH, check_same_thread=False)
except Exception:
    _CON = None

_CACHE: Dict[str, Dict[str, Any]] = {
    "recent": {"ts": 0.0, "value": False},
    "cooldown": {"ts": 0.0, "value": False},
}

_RECENT_GUARD_PIPS = float(os.getenv("MICRO_RECENT_GUARD_PIPS", "12.0"))
_RECENT_GUARD_MINUTES = float(os.getenv("MICRO_RECENT_GUARD_MINUTES", "45.0"))
_COOLDOWN_LOSS_PIPS = float(os.getenv("MICRO_LOSS_COOLDOWN_PIPS", "8.0"))
_COOLDOWN_MINUTES = float(os.getenv("MICRO_LOSS_COOLDOWN_MINUTES", "10.0"))


def _parse_ts(raw: Any) -> datetime.datetime | None:
    if not raw:
        return None
    try:
        txt = str(raw)
        return datetime.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except Exception:
        return None


def _query_rows(limit: int = 8) -> list[tuple[float, str]]:
    if _CON is None:
        return []
    try:
        cur = _CON.cursor()
        rows = cur.execute(
            """
            SELECT pl_pips, close_time
            FROM trades
            WHERE pocket='micro' AND state='CLOSED' AND pl_pips IS NOT NULL
            ORDER BY datetime(close_time) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [(float(pl), ts) for pl, ts in rows if pl is not None]
    except Exception:
        return []


def micro_recent_loss_guard() -> bool:
    """Return True when recent closed losses warrant pausing entries."""

    now = time.time()
    cache = _CACHE["recent"]
    if now - cache["ts"] < 30.0:
        return bool(cache["value"])

    rows = _query_rows(limit=10)
    guard = False
    if rows:
        losses = [pl for pl, _ in rows if pl <= 0.0]
        if sum(losses) <= -_RECENT_GUARD_PIPS:
            guard = True
        else:
            cutoff = now - (_RECENT_GUARD_MINUTES * 60.0)
            for pl, ts in rows:
                if pl > -_COOLDOWN_LOSS_PIPS:
                    continue
                dt = _parse_ts(ts)
                if dt and dt.timestamp() >= cutoff:
                    guard = True
                    break

    cache.update({"ts": now, "value": guard})
    return guard


def micro_loss_cooldown_active() -> bool:
    """Return True when the latest loss breaches the cool-down threshold."""

    now = time.time()
    cache = _CACHE["cooldown"]
    if now - cache["ts"] < 15.0:
        return bool(cache["value"])

    if _CON is None:
        cache.update({"ts": now, "value": False})
        return False

    try:
        row = _CON.execute(
            """
            SELECT pl_pips, close_time
            FROM trades
            WHERE pocket='micro' AND state='CLOSED' AND pl_pips IS NOT NULL
            ORDER BY datetime(close_time) DESC
            LIMIT 1
            """
        ).fetchone()
    except Exception:
        row = None

    active = False
    if row:
        pl, ts = row
        try:
            pl_val = float(pl)
        except (TypeError, ValueError):
            pl_val = 0.0
        if pl_val <= -_COOLDOWN_LOSS_PIPS:
            dt = _parse_ts(ts)
            if dt:
                if now - dt.timestamp() <= _COOLDOWN_MINUTES * 60.0:
                    active = True

    cache.update({"ts": now, "value": active})
    return active


def is_micro_entry_blocked() -> bool:
    """Composite helper used by strategy launcher."""

    if micro_loss_cooldown_active():
        return True
    return micro_recent_loss_guard()

