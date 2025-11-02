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
import time
from typing import Any, Dict, List, Tuple

from utils.trade_repository import get_trade_repository

_TRADE_REPO = get_trade_repository()

_CACHE: Dict[str, Dict[str, Any]] = {
    "recent": {"ts": 0.0, "value": False},
    "cooldown": {"ts": 0.0, "value": False},
}

_RECENT_GUARD_PIPS = float(os.getenv("MICRO_RECENT_GUARD_PIPS", "9.0"))
_RECENT_GUARD_MINUTES = float(os.getenv("MICRO_RECENT_GUARD_MINUTES", "35.0"))
_COOLDOWN_LOSS_PIPS = float(os.getenv("MICRO_LOSS_COOLDOWN_PIPS", "6.0"))
_COOLDOWN_MINUTES = float(os.getenv("MICRO_LOSS_COOLDOWN_MINUTES", "15.0"))


def _parse_ts(raw: Any) -> datetime.datetime | None:
    if not raw:
        return None
    try:
        txt = str(raw)
        return datetime.datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except Exception:
        return None


def _query_rows(limit: int = 8) -> List[Tuple[float, str]]:
    records = _TRADE_REPO.recent_trades("micro", limit=limit, closed_only=True)
    result: List[Tuple[float, str]] = []
    for rec in records:
        if rec.pl_pips is None or rec.close_time is None:
            continue
        result.append((float(rec.pl_pips), rec.close_time.isoformat()))
    return result


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

    last_trade = _TRADE_REPO.last_closed_trade("micro")

    active = False
    if last_trade and last_trade.pl_pips is not None and last_trade.close_time is not None:
        if last_trade.pl_pips <= -_COOLDOWN_LOSS_PIPS:
            if now - last_trade.close_time.timestamp() <= _COOLDOWN_MINUTES * 60.0:
                active = True

    cache.update({"ts": now, "value": active})
    return active


def is_micro_entry_blocked() -> bool:
    """Composite helper used by strategy launcher."""

    if micro_loss_cooldown_active():
        return True
    return micro_recent_loss_guard()
