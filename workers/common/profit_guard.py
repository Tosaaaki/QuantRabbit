"""
Profit giveback guard.
Blocks new entries when recent realized PnL gives back too much from the peak.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

LOG = logging.getLogger(__name__)

_DB = pathlib.Path("logs/trades.db")

_RAW_ENABLED = os.getenv("PROFIT_GUARD_ENABLED", "1")
_ENABLED = str(_RAW_ENABLED).strip().lower() not in {"", "0", "false", "no"}
_MODE = os.getenv("PROFIT_GUARD_MODE", "block").strip().lower()
_LOOKBACK_MIN = max(10, int(float(os.getenv("PROFIT_GUARD_LOOKBACK_MIN", "180"))))
_MIN_PEAK_PIPS = float(os.getenv("PROFIT_GUARD_MIN_PEAK_PIPS", "6"))
_MAX_GIVEBACK_PIPS = float(os.getenv("PROFIT_GUARD_MAX_GIVEBACK_PIPS", "3"))
_MIN_PEAK_JPY = float(os.getenv("PROFIT_GUARD_MIN_PEAK_JPY", "0"))
_MAX_GIVEBACK_JPY = float(os.getenv("PROFIT_GUARD_MAX_GIVEBACK_JPY", "0"))
_TTL_SEC = max(10.0, float(os.getenv("PROFIT_GUARD_TTL_SEC", "30")))
_RAW_POCKETS = os.getenv("PROFIT_GUARD_POCKETS", "scalp").strip().lower()

_ALL_POCKETS = _RAW_POCKETS in {"*", "all"}
_POCKETS = {item.strip() for item in _RAW_POCKETS.split(",") if item.strip()} if not _ALL_POCKETS else set()

_cache: dict[str, tuple[float, "ProfitDecision"]] = {}


def _threshold(name: str, pocket: str, default: float) -> float:
    pocket_key = (pocket or "").strip().upper()
    if pocket_key:
        raw = os.getenv(f"{name}_{pocket_key}")
        if raw is not None:
            try:
                return float(raw)
            except ValueError:
                pass
    raw = os.getenv(name)
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _should_apply(pocket: str) -> bool:
    if not pocket:
        return False
    if _ALL_POCKETS:
        return True
    if not _POCKETS:
        return False
    return pocket.strip().lower() in _POCKETS


@dataclass(frozen=True, slots=True)
class ProfitDecision:
    allowed: bool
    reason: str
    peak_pips: float = 0.0
    current_pips: float = 0.0
    peak_jpy: float = 0.0
    current_jpy: float = 0.0


def _query_guard(pocket: str) -> ProfitDecision:
    if not _DB.exists():
        return ProfitDecision(True, "no_db")
    if _LOOKBACK_MIN <= 0:
        return ProfitDecision(True, "lookback_disabled")

    try:
        con = sqlite3.connect(f"file:{_DB}?mode=ro", uri=True, timeout=2.0)
        con.row_factory = sqlite3.Row
    except Exception:
        return ProfitDecision(True, "db_open_failed")

    cutoff = f"-{_LOOKBACK_MIN} minutes"
    try:
        rows = con.execute(
            """
            SELECT id, pl_pips, realized_pl
            FROM trades
            WHERE pocket = ?
              AND close_time IS NOT NULL
              AND datetime(close_time) >= datetime('now', ?)
            ORDER BY datetime(close_time) ASC, id ASC
            """,
            (pocket, cutoff),
        ).fetchall()
    except Exception as exc:
        LOG.debug("[PROFIT_GUARD] query failed pocket=%s err=%s", pocket, exc)
        return ProfitDecision(True, "query_failed")
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not rows:
        return ProfitDecision(True, "no_recent_trades")

    current_pips = 0.0
    current_jpy = 0.0
    peak_pips = 0.0
    peak_jpy = 0.0
    for row in rows:
        try:
            current_pips += float(row["pl_pips"] or 0.0)
            current_jpy += float(row["realized_pl"] or 0.0)
        except Exception:
            continue
        peak_pips = max(peak_pips, current_pips)
        peak_jpy = max(peak_jpy, current_jpy)

    min_peak_pips = _threshold("PROFIT_GUARD_MIN_PEAK_PIPS", pocket, _MIN_PEAK_PIPS)
    min_peak_jpy = _threshold("PROFIT_GUARD_MIN_PEAK_JPY", pocket, _MIN_PEAK_JPY)
    max_giveback_pips = _threshold("PROFIT_GUARD_MAX_GIVEBACK_PIPS", pocket, _MAX_GIVEBACK_PIPS)
    max_giveback_jpy = _threshold("PROFIT_GUARD_MAX_GIVEBACK_JPY", pocket, _MAX_GIVEBACK_JPY)

    if min_peak_pips <= 0 and min_peak_jpy <= 0:
        return ProfitDecision(True, "min_peak_disabled", peak_pips, current_pips, peak_jpy, current_jpy)
    if max_giveback_pips <= 0 and max_giveback_jpy <= 0:
        return ProfitDecision(True, "giveback_disabled", peak_pips, current_pips, peak_jpy, current_jpy)

    if peak_pips < max(0.0, min_peak_pips) and peak_jpy < max(0.0, min_peak_jpy):
        return ProfitDecision(True, "peak_below_min", peak_pips, current_pips, peak_jpy, current_jpy)

    giveback_pips = max(0.0, peak_pips - current_pips)
    giveback_jpy = max(0.0, peak_jpy - current_jpy)
    hit_pips = max_giveback_pips > 0 and giveback_pips >= max_giveback_pips
    hit_jpy = max_giveback_jpy > 0 and giveback_jpy >= max_giveback_jpy
    if hit_pips or hit_jpy:
        if hit_pips and hit_jpy:
            reason = (
                f"giveback={giveback_pips:.2f}p/{giveback_jpy:.0f}jpy "
                f"peak={peak_pips:.2f}p/{peak_jpy:.0f}jpy "
                f"curr={current_pips:.2f}p/{current_jpy:.0f}jpy "
                f"win={_LOOKBACK_MIN}m"
            )
        elif hit_pips:
            reason = (
                f"giveback={giveback_pips:.2f}p peak={peak_pips:.2f}p "
                f"curr={current_pips:.2f}p win={_LOOKBACK_MIN}m"
            )
        else:
            reason = (
                f"giveback={giveback_jpy:.0f}jpy peak={peak_jpy:.0f}jpy "
                f"curr={current_jpy:.0f}jpy win={_LOOKBACK_MIN}m"
            )
        return ProfitDecision(False, reason, peak_pips, current_pips, peak_jpy, current_jpy)

    return ProfitDecision(True, "ok", peak_pips, current_pips, peak_jpy, current_jpy)


def is_allowed(pocket: str, *, strategy_tag: Optional[str] = None) -> ProfitDecision:
    """
    Evaluate whether new entries are allowed based on recent profit giveback.
    strategy_tag is currently unused (reserved for future per-strategy scope).
    """
    if not _ENABLED or _MODE in {"off", "disabled", "false", "no"}:
        return ProfitDecision(True, "disabled")
    if not _should_apply(pocket):
        return ProfitDecision(True, "skip_pocket")

    key = pocket.strip().lower()
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] <= _TTL_SEC:
        return cached[1]

    decision = _query_guard(pocket)
    if _MODE != "block":
        decision = ProfitDecision(
            True,
            f"warn:{decision.reason}",
            decision.peak_pips,
            decision.current_pips,
            decision.peak_jpy,
            decision.current_jpy,
        )
    _cache[key] = (now, decision)
    return decision
