"""Configuration for the impulse momentum S5 worker."""

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


LOG_PREFIX = "[IMPULSE-MOMO]"
ENABLED: bool = _bool("IMPULSE_MOMENTUM_S5_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("IMPULSE_MOMENTUM_S5_LOOP_INTERVAL_SEC", 0.4))

ALLOWED_HOURS_UTC = frozenset(range(24))
BLOCKED_WEEKDAYS = tuple(
    token for token in os.getenv("IMPULSE_MOMENTUM_S5_BLOCKED_WEEKDAYS", "").split(",") if token.strip()
)

WINDOW_SEC: float = max(120.0, _float("IMPULSE_MOMENTUM_S5_WINDOW_SEC", 360.0))
BUCKET_SECONDS: float = max(1.0, _float("IMPULSE_MOMENTUM_S5_BUCKET_SECONDS", 5.0))
FAST_BUCKETS: int = max(36, _int("IMPULSE_MOMENTUM_S5_FAST_BUCKETS", 60))
SLOW_BUCKETS: int = max(FAST_BUCKETS + 12, _int("IMPULSE_MOMENTUM_S5_SLOW_BUCKETS", 96))
MIN_BUCKETS: int = max(FAST_BUCKETS + 6, _int("IMPULSE_MOMENTUM_S5_MIN_BUCKETS", 84))

MAX_SPREAD_PIPS: float = max(0.1, _float("IMPULSE_MOMENTUM_S5_MAX_SPREAD_PIPS", 1.2))
INSTANT_MOVE_PIPS_LIMIT: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_INSTANT_MOVE_PIPS_LIMIT", 0.9))
BACKOFF_AFTER_SPIKE_SEC: float = max(10.0, _float("IMPULSE_MOMENTUM_S5_BACKOFF_AFTER_SPIKE", 45.0))

ENTRY_UNITS: int = max(500, _int("IMPULSE_MOMENTUM_S5_ENTRY_UNITS", 2000))
MAX_ACTIVE_TRADES: int = max(1, _int("IMPULSE_MOMENTUM_S5_MAX_ACTIVE_TRADES", 2))
COOLDOWN_SEC: float = max(60.0, _float("IMPULSE_MOMENTUM_S5_COOLDOWN_SEC", 180.0))
POST_EXIT_COOLDOWN_SEC: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_POST_EXIT_COOLDOWN_SEC", 120.0))

MIN_ATR_PIPS: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_MIN_ATR_PIPS", 1.2))
ATR_FAST_PERIOD: int = max(6, _int("IMPULSE_MOMENTUM_S5_ATR_FAST_PERIOD", 18))
ATR_SLOW_PERIOD: int = max(ATR_FAST_PERIOD + 4, _int("IMPULSE_MOMENTUM_S5_ATR_SLOW_PERIOD", 36))

FAST_Z_LONG_MIN: float = _float("IMPULSE_MOMENTUM_S5_FAST_Z_LONG_MIN", 0.8)
FAST_Z_SHORT_MAX: float = _float("IMPULSE_MOMENTUM_S5_FAST_Z_SHORT_MAX", -0.8)
SLOW_Z_LONG_MIN: float = _float("IMPULSE_MOMENTUM_S5_SLOW_Z_LONG_MIN", 0.35)
SLOW_Z_SHORT_MAX: float = _float("IMPULSE_MOMENTUM_S5_SLOW_Z_SHORT_MAX", -0.35)

RSI_PERIOD: int = max(5, _int("IMPULSE_MOMENTUM_S5_RSI_PERIOD", 14))
RSI_LONG_MAX: float = _float("IMPULSE_MOMENTUM_S5_RSI_LONG_MAX", 80.0)
RSI_SHORT_MIN: float = _float("IMPULSE_MOMENTUM_S5_RSI_SHORT_MIN", 20.0)

MIN_BREAKOUT_PIPS: float = max(0.2, _float("IMPULSE_MOMENTUM_S5_MIN_BREAKOUT_PIPS", 0.9))
RETRACE_CONFIRMED_PIPS: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_RETRACE_CONFIRMED_PIPS", 0.25))
MIN_RETRACE_GAP_PIPS: float = RETRACE_CONFIRMED_PIPS

TREND_ALIGN_BUFFER_PIPS: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_TREND_ALIGN_BUFFER_PIPS", 0.4))
TREND_ADX_MIN: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_TREND_ADX_MIN", 20.0))
TREND_SLOPE_MIN_PIPS: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_TREND_SLOPE_MIN_PIPS", 0.12))
TREND_SIZE_MULT: float = max(1.0, _float("IMPULSE_MOMENTUM_S5_TREND_SIZE_MULT", 1.4))

DAILY_PNL_STOP_PIPS: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_DAILY_PNL_STOP_PIPS", 9.0))
MAX_CONSEC_LOSSES: int = max(0, _int("IMPULSE_MOMENTUM_S5_MAX_CONSEC_LOSSES", 2))
PERFORMANCE_REFRESH_SEC: float = max(30.0, _float("IMPULSE_MOMENTUM_S5_PERF_REFRESH_SEC", 120.0))
NEWS_BLOCK_MINUTES: float = _float("IMPULSE_MOMENTUM_S5_NEWS_BLOCK_MINUTES", 0.0)

BE_TRIGGER_PIPS: float = max(0.1, _float("IMPULSE_MOMENTUM_S5_BE_TRIGGER_PIPS", 1.4))
BE_OFFSET_PIPS: float = _float("IMPULSE_MOMENTUM_S5_BE_OFFSET_PIPS", 0.2)
TRAIL_TRIGGER_PIPS: float = max(0.2, _float("IMPULSE_MOMENTUM_S5_TRAIL_TRIGGER_PIPS", 1.8))
TRAIL_BACKOFF_PIPS: float = max(0.1, _float("IMPULSE_MOMENTUM_S5_TRAIL_BACKOFF_PIPS", 0.7))
TRAIL_STEP_PIPS: float = max(0.1, _float("IMPULSE_MOMENTUM_S5_TRAIL_STEP_PIPS", 0.6))
TRAIL_COOLDOWN_SEC: float = max(5.0, _float("IMPULSE_MOMENTUM_S5_TRAIL_COOLDOWN_SEC", 40.0))
MANAGED_POLL_SEC: float = max(1.0, _float("IMPULSE_MOMENTUM_S5_MANAGED_POLL_SEC", 2.0))

TP_ATR_MULT: float = max(0.0, _float("IMPULSE_MOMENTUM_S5_TP_ATR_MULT", 1.8))
TP_ATR_MIN_PIPS: float = max(0.5, _float("IMPULSE_MOMENTUM_S5_TP_ATR_MIN_PIPS", 1.6))
TP_ATR_MAX_PIPS: float = max(TP_ATR_MIN_PIPS, _float("IMPULSE_MOMENTUM_S5_TP_ATR_MAX_PIPS", 3.2))
SL_ATR_MULT: float = max(0.1, _float("IMPULSE_MOMENTUM_S5_SL_ATR_MULT", 0.9))
SL_ATR_MIN_PIPS: float = max(0.6, _float("IMPULSE_MOMENTUM_S5_SL_ATR_MIN_PIPS", 1.5))

BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("IMPULSE_MOMENTUM_S5_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)
)
)
LOSS_STREAK_MAX: int = max(0, _int("IMPULSE_MOMENTUM_S5_MAX_CONSEC_LOSSES", 2))
LOSS_STREAK_COOLDOWN_MIN: float = max(
    0.0, _float("IMPULSE_MOMENTUM_S5_LOSS_COOLDOWN_MIN", 20.0)
)
