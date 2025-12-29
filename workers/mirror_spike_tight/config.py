"""Configuration for the mirror spike tight S5 worker."""

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


LOG_PREFIX = "[MIRROR-TIGHT]"
ENABLED: bool = _bool("MIRROR_SPIKE_TIGHT_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.25, _float("MIRROR_SPIKE_TIGHT_LOOP_INTERVAL_SEC", 0.5))

ALLOWED_HOURS_UTC = frozenset(range(24))
BLOCKED_WEEKDAYS = tuple(
    token
    for token in os.getenv("MIRROR_SPIKE_TIGHT_BLOCKED_WEEKDAYS", "5").split(",")
    if token.strip()
)

WINDOW_SEC: float = max(60.0, _float("MIRROR_SPIKE_TIGHT_WINDOW_SEC", 240.0))
BUCKET_SECONDS: float = max(1.0, _float("MIRROR_SPIKE_TIGHT_BUCKET_SECONDS", 5.0))
MIN_BUCKETS: int = max(24, _int("MIRROR_SPIKE_TIGHT_MIN_BUCKETS", 48))

MAX_SPREAD_PIPS: float = max(0.1, _float("MIRROR_SPIKE_TIGHT_MAX_SPREAD_PIPS", 0.28))
SPREAD_P50_LIMIT: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_SPREAD_P50_LIMIT", 0.16))
RETURN_WINDOW_SEC: float = max(5.0, _float("MIRROR_SPIKE_TIGHT_RETURN_WINDOW_SEC", 15.0))
RETURN_PIPS_LIMIT: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_RETURN_PIPS_LIMIT", 0.3))
INSTANT_MOVE_PIPS_LIMIT: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_INSTANT_MOVE_PIPS_LIMIT", 0.8))
TICK_GAP_MS_LIMIT: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_TICK_GAP_MS_LIMIT", 120.0))
TICK_GAP_MOVE_PIPS: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_TICK_GAP_MOVE_PIPS", 0.5))
MIN_DENSITY_TICKS: int = max(10, _int("MIRROR_SPIKE_TIGHT_MIN_DENSITY_TICKS", 30))

MIRROR_LOOKBACK_BUCKETS: int = max(12, _int("MIRROR_SPIKE_TIGHT_LOOKBACK_BUCKETS", 36))
SPIKE_THRESHOLD_PIPS: float = max(0.2, _float("MIRROR_SPIKE_TIGHT_SPIKE_THRESHOLD_PIPS", 1.1))
CONFIRM_RANGE_PIPS: float = max(0.05, _float("MIRROR_SPIKE_TIGHT_CONFIRM_RANGE_PIPS", 0.35))
COOLDOWN_SEC: float = max(30.0, _float("MIRROR_SPIKE_TIGHT_COOLDOWN_SEC", 180.0))
POST_EXIT_COOLDOWN_SEC: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_POST_EXIT_COOLDOWN_SEC", 180.0))

ENTRY_UNITS: int = max(1000, _int("MIRROR_SPIKE_TIGHT_ENTRY_UNITS", 5000))
MAX_ACTIVE_TRADES: int = max(1, _int("MIRROR_SPIKE_TIGHT_MAX_ACTIVE_TRADES", 1))

TP_PIPS: float = max(0.5, _float("MIRROR_SPIKE_TIGHT_TP_PIPS", 1.6))
SL_PIPS: float = max(0.5, _float("MIRROR_SPIKE_TIGHT_SL_PIPS", 0.8))
BE_TRIGGER_PIPS: float = max(0.1, _float("MIRROR_SPIKE_TIGHT_BE_TRIGGER_PIPS", 1.1))
BE_OFFSET_PIPS: float = _float("MIRROR_SPIKE_TIGHT_BE_OFFSET_PIPS", 0.2)

LOSS_STREAK_MAX: int = max(0, _int("MIRROR_SPIKE_TIGHT_MAX_CONSEC_LOSSES", 3))
LOSS_STREAK_COOLDOWN_MIN: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_LOSS_COOLDOWN_MIN", 45.0))
DAILY_PNL_STOP_PIPS: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_DAILY_PNL_STOP_PIPS", 10.0))
PERFORMANCE_REFRESH_SEC: float = max(30.0, _float("MIRROR_SPIKE_TIGHT_PERF_REFRESH_SEC", 120.0))

TREND_ALIGN_BUFFER_PIPS: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_TREND_ALIGN_BUFFER_PIPS", 0.35))
TREND_ADX_MIN: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_TREND_ADX_MIN", 25.0))
TREND_SLOPE_MIN_PIPS: float = max(0.0, _float("MIRROR_SPIKE_TIGHT_TREND_SLOPE_MIN_PIPS", 0.17))

)
)
BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("MIRROR_SPIKE_TIGHT_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)

MAX_TRADES_PER_DAY: int = max(0, _int("MIRROR_SPIKE_TIGHT_MAX_TRADES_PER_DAY", 220))
