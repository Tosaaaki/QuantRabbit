"""Configuration for the impulse-retest S5 worker."""

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


LOG_PREFIX = "[IMP-RETEST-S5]"
ENABLED: bool = _bool("IMPULSE_RETEST_S5_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("IMPULSE_RETEST_S5_LOOP_INTERVAL_SEC", 0.5))

ACTIVE_HOURS_UTC = frozenset(range(24))

WINDOW_SEC: float = max(60.0, _float("IMPULSE_RETEST_S5_WINDOW_SEC", 360.0))
BUCKET_SECONDS: float = max(1.0, _float("IMPULSE_RETEST_S5_BUCKET_SECONDS", 5.0))
MIN_BUCKETS: int = max(60, _int("IMPULSE_RETEST_S5_MIN_BUCKETS", 120))

MAX_SPREAD_PIPS: float = max(0.1, _float("IMPULSE_RETEST_S5_MAX_SPREAD_PIPS", 1.2))
ENTRY_UNITS: int = max(500, _int("IMPULSE_RETEST_S5_ENTRY_UNITS", 6000))
MAX_ACTIVE_TRADES: int = max(1, _int("IMPULSE_RETEST_S5_MAX_ACTIVE_TRADES", 2))
COOLDOWN_SEC: float = max(60.0, _float("IMPULSE_RETEST_S5_COOLDOWN_SEC", 180.0))
POST_EXIT_COOLDOWN_SEC: float = max(20.0, _float("IMPULSE_RETEST_S5_POST_EXIT_COOLDOWN_SEC", 240.0))
NEWS_BLOCK_MINUTES: float = _float("IMPULSE_RETEST_S5_NEWS_BLOCK_MINUTES", 0.0)

IMPULSE_LOOKBACK: int = max(30, _int("IMPULSE_RETEST_S5_IMPULSE_LOOKBACK", 72))
IMPULSE_MIN_PIPS: float = max(0.5, _float("IMPULSE_RETEST_S5_IMPULSE_MIN_PIPS", 2.8))
FIB_LOWER: float = max(0.0, min(1.0, _float("IMPULSE_RETEST_S5_FIB_LOWER", 0.42)))
FIB_UPPER: float = max(FIB_LOWER, min(1.0, _float("IMPULSE_RETEST_S5_FIB_UPPER", 0.58)))
RETEST_MAX_SEC: float = max(10.0, _float("IMPULSE_RETEST_S5_RETEST_MAX_SEC", 45.0))

RSI_PERIOD: int = max(5, _int("IMPULSE_RETEST_S5_RSI_PERIOD", 14))
RSI_LONG_MAX: float = _float("IMPULSE_RETEST_S5_RSI_LONG_MAX", 70.0)
RSI_SHORT_MIN: float = _float("IMPULSE_RETEST_S5_RSI_SHORT_MIN", 30.0)

MIN_ATR_PIPS: float = max(0.0, _float("IMPULSE_RETEST_S5_MIN_ATR_PIPS", 0.8))
TP_RATIO: float = max(0.1, _float("IMPULSE_RETEST_S5_TP_RATIO", 0.45))
SL_BUFFER_PIPS: float = max(0.05, _float("IMPULSE_RETEST_S5_SL_BUFFER_PIPS", 0.2))
TIMEOUT_SEC: float = max(30.0, _float("IMPULSE_RETEST_S5_TIMEOUT_SEC", 180.0))

STAGE_MIN_DELTA_PIPS: float = max(0.05, _float("IMPULSE_RETEST_S5_STAGE_MIN_DELTA_PIPS", 0.24))

BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("IMPULSE_RETEST_S5_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)
LOSS_STREAK_MAX: int = max(0, _int("IMPULSE_RETEST_S5_MAX_CONSEC_LOSSES", 2))
LOSS_STREAK_COOLDOWN_MIN: float = max(0.0, _float("IMPULSE_RETEST_S5_LOSS_COOLDOWN_MIN", 20.0))
