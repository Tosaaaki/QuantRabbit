"""Configuration for the squeeze-break S5 worker."""

from __future__ import annotations

import os
from typing import Set

PIP_VALUE = 0.01


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_hours(key: str, default: str) -> Set[int]:
    raw = os.getenv(key, default)
    hours: Set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            try:
                start = int(float(left))
                end = int(float(right))
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for hour in range(start, end + 1):
                if 0 <= hour <= 23:
                    hours.add(hour)
            continue
        try:
            hour_val = int(float(token))
        except ValueError:
            continue
        if 0 <= hour_val <= 23:
            hours.add(hour_val)
    return hours


LOG_PREFIX = "[SQUEEZE-S5]"
# Hard stop: keep squeeze break worker disabled regardless of environment.
ENABLED: bool = False
LOOP_INTERVAL_SEC: float = max(0.2, _float("SQUEEZE_BREAK_S5_LOOP_INTERVAL_SEC", 0.45))

ACTIVE_HOURS_UTC = frozenset(
    _parse_hours("SQUEEZE_BREAK_S5_ACTIVE_HOURS", "1,3,7-11,14-18")
)

WINDOW_SEC: float = max(60.0, _float("SQUEEZE_BREAK_S5_WINDOW_SEC", 360.0))
BUCKET_SECONDS: float = max(1.0, _float("SQUEEZE_BREAK_S5_BUCKET_SECONDS", 5.0))
FAST_BUCKETS: int = max(24, _int("SQUEEZE_BREAK_S5_FAST_BUCKETS", 48))
SLOW_BUCKETS: int = max(FAST_BUCKETS + 12, _int("SQUEEZE_BREAK_S5_SLOW_BUCKETS", 96))
MIN_BUCKETS: int = max(SLOW_BUCKETS + 4, _int("SQUEEZE_BREAK_S5_MIN_BUCKETS", 30))

MAX_SPREAD_PIPS: float = max(0.1, _float("SQUEEZE_BREAK_S5_MAX_SPREAD_PIPS", 0.8))
MIN_DENSITY_TICKS: int = max(10, _int("SQUEEZE_BREAK_S5_MIN_DENSITY_TICKS", 60))
ENTRY_UNITS: int = max(1000, _int("SQUEEZE_BREAK_S5_ENTRY_UNITS", 5000))
MAX_ACTIVE_TRADES: int = max(1, _int("SQUEEZE_BREAK_S5_MAX_ACTIVE_TRADES", 1))
COOLDOWN_SEC: float = max(10.0, _float("SQUEEZE_BREAK_S5_COOLDOWN_SEC", 60.0))
POST_EXIT_COOLDOWN_SEC: float = max(20.0, _float("SQUEEZE_BREAK_S5_POST_EXIT_COOLDOWN_SEC", 180.0))

BBW_WINDOW: int = max(10, _int("SQUEEZE_BREAK_S5_BBW_WINDOW", 40))
BBW_THRESHOLD: float = max(0.05, _float("SQUEEZE_BREAK_S5_BBW_THRESHOLD", 0.24))
BREAK_BUFFER_PIPS: float = max(0.02, _float("SQUEEZE_BREAK_S5_BREAK_BUFFER_PIPS", 0.1))
RETEST_MAX_SEC: float = max(5.0, _float("SQUEEZE_BREAK_S5_RETEST_MAX_SEC", 30.0))

RSI_PERIOD: int = max(5, _int("SQUEEZE_BREAK_S5_RSI_PERIOD", 14))
RSI_LONG_MAX: float = _float("SQUEEZE_BREAK_S5_RSI_LONG_MAX", 78.0)
RSI_SHORT_MIN: float = _float("SQUEEZE_BREAK_S5_RSI_SHORT_MIN", 22.0)

MIN_ATR_PIPS: float = max(0.0, _float("SQUEEZE_BREAK_S5_MIN_ATR_PIPS", 0.2))
TP_ATR_MULT: float = max(0.0, _float("SQUEEZE_BREAK_S5_TP_ATR_MULT", 1.2))
TP_MIN_PIPS: float = max(1.0, _float("SQUEEZE_BREAK_S5_TP_MIN_PIPS", 1.8))
SL_ATR_MULT: float = max(0.1, _float("SQUEEZE_BREAK_S5_SL_ATR_MULT", 1.0))
SL_MIN_PIPS: float = max(0.5, _float("SQUEEZE_BREAK_S5_SL_MIN_PIPS", 1.6))
TIMEOUT_SEC: float = max(30.0, _float("SQUEEZE_BREAK_S5_TIMEOUT_SEC", 150.0))

STAGE_MIN_DELTA_PIPS: float = max(0.05, _float("SQUEEZE_BREAK_S5_STAGE_MIN_DELTA_PIPS", 0.24))

BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("SQUEEZE_BREAK_S5_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)
NEWS_BLOCK_MINUTES: float = max(0.0, _float("SQUEEZE_BREAK_S5_NEWS_BLOCK_MINUTES", 20.0))
NEWS_BLOCK_MIN_IMPACT: int = max(1, _int("SQUEEZE_BREAK_S5_NEWS_BLOCK_MIN_IMPACT", 2))
LOSS_STREAK_MAX: int = max(0, _int("SQUEEZE_BREAK_S5_MAX_CONSEC_LOSSES", 2))
LOSS_STREAK_COOLDOWN_MIN: float = max(0.0, _float("SQUEEZE_BREAK_S5_LOSS_COOLDOWN_MIN", 20.0))

SPREAD_P50_LIMIT: float = max(0.0, _float("SQUEEZE_BREAK_S5_SPREAD_P50_LIMIT", 0.2))
RETURN_WINDOW_SEC: float = max(5.0, _float("SQUEEZE_BREAK_S5_RETURN_WINDOW_SEC", 18.0))
RETURN_PIPS_LIMIT: float = max(0.0, _float("SQUEEZE_BREAK_S5_RETURN_PIPS_LIMIT", 1.3))
INSTANT_MOVE_PIPS_LIMIT: float = max(0.0, _float("SQUEEZE_BREAK_S5_INSTANT_MOVE_PIPS_LIMIT", 1.3))
TICK_GAP_MS_LIMIT: float = max(0.0, _float("SQUEEZE_BREAK_S5_TICK_GAP_MS_LIMIT", 150.0))
TICK_GAP_MOVE_PIPS: float = max(0.0, _float("SQUEEZE_BREAK_S5_TICK_GAP_MOVE_PIPS", 0.6))
