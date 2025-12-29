"""
Lightweight performance guard for workers.
Checks recent PF / win-rate per strategy_tag+pocket (and optional hour) and blocks entries when stats are poor.
Designed to be cheap: uses SQLite with short TTL caching.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)

_DB = pathlib.Path("logs/trades.db")

_LOOKBACK_DAYS = max(1, int(float(os.getenv("PERF_GUARD_LOOKBACK_DAYS", "3"))))
_MIN_TRADES = max(5, int(float(os.getenv("PERF_GUARD_MIN_TRADES", "20"))))
_PF_MIN = float(os.getenv("PERF_GUARD_PF_MIN", "0.9") or 0.9)
_WIN_MIN = float(os.getenv("PERF_GUARD_WIN_MIN", "0.48") or 0.48)
_TTL_SEC = max(30.0, float(os.getenv("PERF_GUARD_TTL_SEC", "120")) or 120.0)
_HOURLY_ENABLED = os.getenv("PERF_GUARD_HOURLY", "1").strip().lower() not in {"", "0", "false", "no"}
_HOURLY_MIN_TRADES = max(5, int(float(os.getenv("PERF_GUARD_HOURLY_MIN_TRADES", "12"))))

_cache: dict[tuple[str, str, Optional[int]], tuple[float, bool, str, int]] = {}


@dataclass(frozen=True, slots=True)
class PerfDecision:
    allowed: bool
    reason: str
    sample: int


def _query_perf(tag: str, pocket: str, hour: Optional[int]) -> Tuple[bool, str, int]:
    if not _DB.exists():
        return True, "no_db", 0
    try:
        con = sqlite3.connect(_DB)
        con.row_factory = sqlite3.Row
    except Exception:
        return True, "db_open_failed", 0
    try:
        row = con.execute(
            """
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN pl_pips > 0 THEN pl_pips ELSE 0 END) AS profit,
              SUM(CASE WHEN pl_pips < 0 THEN ABS(pl_pips) ELSE 0 END) AS loss,
              SUM(CASE WHEN pl_pips > 0 THEN 1 ELSE 0 END) AS win
            FROM trades
            WHERE strategy_tag = ?
              AND pocket = ?
              AND close_time >= datetime('now', ?)
              AND (strftime('%H', close_time) = ? OR ? IS NULL)
            """,
            (tag, pocket, f"-{_LOOKBACK_DAYS} day", f"{hour:02d}" if hour is not None else None, f"{hour:02d}" if hour is not None else None),
        ).fetchone()
    except Exception as exc:
        LOG.debug("[PERF_GUARD] query failed tag=%s pocket=%s err=%s", tag, pocket, exc)
        return True, "query_failed", 0
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not row:
        return True, "no_rows", 0
    n = int(row["n"] or 0)
    min_trades = _HOURLY_MIN_TRADES if hour is not None else _MIN_TRADES
    if n < min_trades:
        return True, f"warmup_n={n}", n
    profit = float(row["profit"] or 0.0)
    loss = float(row["loss"] or 0.0)
    win = float(row["win"] or 0.0)
    pf = profit / loss if loss > 0 else float("inf")
    win_rate = win / n if n > 0 else 0.0

    if pf < _PF_MIN or win_rate < _WIN_MIN:
        return False, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n
    return True, f"pf={pf:.2f} win={win_rate:.2f} n={n}", n


def is_allowed(tag: str, pocket: str, *, hour: Optional[int] = None) -> PerfDecision:
    """
    hour: optional UTC hour string (0-23) to apply hourly PF filter. When None, aggregate filter only.
    """
    key = (tag, pocket, hour if _HOURLY_ENABLED else None)
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] <= _TTL_SEC:
        _, ok, reason, sample = cached
        return PerfDecision(ok, reason, sample)

    # hourly check first (stricter). If enabled and blocked, return immediately.
    if _HOURLY_ENABLED and hour is not None:
        ok_hour, reason_hour, sample_hour = _query_perf(tag, pocket, hour)
        if not ok_hour:
            _cache[key] = (now, ok_hour, f"hour{hour}:{reason_hour}", sample_hour)
            return PerfDecision(ok_hour, f"hour{hour}:{reason_hour}", sample_hour)

    ok, reason, sample = _query_perf(tag, pocket, None)
    _cache[key] = (now, ok, reason, sample)
    return PerfDecision(ok, reason, sample)
