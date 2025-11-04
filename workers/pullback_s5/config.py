"""Configuration for the S5 pullback worker."""

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
            bounds = token.split("-", 1)
            try:
                start = int(float(bounds[0]))
                end = int(float(bounds[1]))
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


def _parse_float_list(key: str, default: str) -> tuple[float, ...]:
    raw = os.getenv(key, default)
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = float(token)
        except ValueError:
            continue
        if val <= 0.0:
            continue
        values.append(val)
    if not values:
        values = [1.0]
    return tuple(values)


LOG_PREFIX = "[PULLBACK-S5]"
ENABLED: bool = _bool("PULLBACK_S5_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("PULLBACK_S5_LOOP_INTERVAL_SEC", 0.45))

ACTIVE_HOURS_UTC = frozenset(
    _parse_hours("PULLBACK_S5_ACTIVE_HOURS", "6,8,10-12,15,19,22-23")
)
ALLOWED_HOURS_UTC = frozenset(
    _parse_hours("PULLBACK_S5_ALLOWED_HOURS", "6,8,10-12,15,19,22-23")
)
BLOCKED_WEEKDAYS = tuple(
    day.strip()
    for day in os.getenv("PULLBACK_S5_BLOCKED_WEEKDAYS", "4").split(",")
    if day.strip()
)

ALLOW_LONG: bool = _bool("PULLBACK_S5_ALLOW_LONG", True)
ALLOW_SHORT: bool = _bool("PULLBACK_S5_ALLOW_SHORT", False)

WINDOW_SEC: float = max(60.0, _float("PULLBACK_S5_WINDOW_SEC", 360.0))
BUCKET_SECONDS: float = max(1.0, _float("PULLBACK_S5_BUCKET_SECONDS", 5.0))
FAST_BUCKETS: int = max(24, _int("PULLBACK_S5_FAST_BUCKETS", 48))
SLOW_BUCKETS: int = max(FAST_BUCKETS + 8, _int("PULLBACK_S5_SLOW_BUCKETS", 72))
MIN_BUCKETS: int = max(FAST_BUCKETS, _int("PULLBACK_S5_MIN_BUCKETS", 60))

MAX_SPREAD_PIPS: float = max(0.1, _float("PULLBACK_S5_MAX_SPREAD_PIPS", 0.85))
ENTRY_UNITS: int = max(1000, _int("PULLBACK_S5_ENTRY_UNITS", 10000))
ENTRY_STAGE_RATIOS: tuple[float, ...] = _parse_float_list(
    "PULLBACK_S5_ENTRY_STAGE_RATIOS", "1.0"
)
ALLOW_DUPLICATE_ENTRIES: bool = _bool(
    "PULLBACK_S5_ALLOW_DUPLICATE_ENTRIES", False
)
MAX_ACTIVE_TRADES: int = 1
STAGE_MIN_DELTA_PIPS: float = max(0.05, _float("PULLBACK_S5_STAGE_MIN_DELTA_PIPS", 0.22))
COOLDOWN_SEC: float = max(20.0, _float("PULLBACK_S5_COOLDOWN_SEC", 120.0))

FAST_Z_MIN: float = _float("PULLBACK_S5_FAST_Z_MIN", 0.18)
FAST_Z_MAX: float = _float("PULLBACK_S5_FAST_Z_MAX", 0.55)
SLOW_Z_SHORT_MAX: float = _float("PULLBACK_S5_SLOW_Z_SHORT_MAX", 0.22)
SLOW_Z_LONG_MIN: float = _float("PULLBACK_S5_SLOW_Z_LONG_MIN", -0.22)

RSI_PERIOD: int = max(5, _int("PULLBACK_S5_RSI_PERIOD", 14))
RSI_SHORT_RANGE = (
    float(_float("PULLBACK_S5_RSI_SHORT_MIN", 46.0)),
    float(_float("PULLBACK_S5_RSI_SHORT_MAX", 68.0)),
)
RSI_LONG_RANGE = (
    float(_float("PULLBACK_S5_RSI_LONG_MIN", 32.0)),
    float(_float("PULLBACK_S5_RSI_LONG_MAX", 54.0)),
)

TP_PIPS: float = max(1.0, _float("PULLBACK_S5_TP_PIPS", 2.2))
MIN_SL_PIPS: float = max(1.0, _float("PULLBACK_S5_MIN_SL_PIPS", 3.0))
SL_ATR_MULT: float = max(0.5, _float("PULLBACK_S5_SL_ATR_MULT", 1.2))
MAX_SL_PIPS: float = max(1.0, _float("PULLBACK_S5_MAX_SL_PIPS", 2.5))

TREND_ALIGN_BUFFER_PIPS: float = max(
    0.0, _float("PULLBACK_S5_TREND_ALIGN_BUFFER_PIPS", 0.6)
)
TREND_ADX_MIN: float = max(0.0, _float("PULLBACK_S5_TREND_ADX_MIN", 18.0))
TP_ATR_MULT: float = max(0.0, _float("PULLBACK_S5_TP_ATR_MULT", 1.8))
TP_ATR_MIN_PIPS: float = max(0.5, _float("PULLBACK_S5_TP_ATR_MIN_PIPS", 1.6))
TP_ATR_MAX_PIPS: float = max(
    TP_ATR_MIN_PIPS,
    _float("PULLBACK_S5_TP_ATR_MAX_PIPS", 3.4),
)

MIN_ATR_PIPS: float = max(0.0, _float("PULLBACK_S5_MIN_ATR_PIPS", 0.8))
MIN_DENSITY_TICKS: int = max(10, _int("PULLBACK_S5_MIN_DENSITY_TICKS", 50))

# --- quality gates ---
BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("PULLBACK_S5_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)
NEWS_BLOCK_MINUTES: float = max(
    0.0, _float("PULLBACK_S5_NEWS_BLOCK_MINUTES", 25.0)
)
NEWS_BLOCK_MIN_IMPACT: int = max(
    1, _int("PULLBACK_S5_NEWS_BLOCK_MIN_IMPACT", 3)
)
LOSS_STREAK_MAX: int = max(0, _int("PULLBACK_S5_MAX_CONSEC_LOSSES", 3))
LOSS_STREAK_COOLDOWN_MIN: float = max(
    0.0, _float("PULLBACK_S5_LOSS_COOLDOWN_MIN", 20.0)
)

# Tokyo時間帯は板スプレッドが0.8p付近で推移するため、既定を広げて環境ガードを通過しやすくする
SPREAD_P50_LIMIT: float = max(0.0, _float("PULLBACK_S5_SPREAD_P50_LIMIT", 0.9))
RETURN_WINDOW_SEC: float = max(5.0, _float("PULLBACK_S5_RETURN_WINDOW_SEC", 18.0))
RETURN_PIPS_LIMIT: float = max(0.0, _float("PULLBACK_S5_RETURN_PIPS_LIMIT", 1.2))
INSTANT_MOVE_PIPS_LIMIT: float = max(0.0, _float("PULLBACK_S5_INSTANT_MOVE_PIPS_LIMIT", 0.8))
TICK_GAP_MS_LIMIT: float = max(0.0, _float("PULLBACK_S5_TICK_GAP_MS_LIMIT", 150.0))
TICK_GAP_MOVE_PIPS: float = max(0.0, _float("PULLBACK_S5_TICK_GAP_MOVE_PIPS", 0.6))

DAILY_PNL_STOP_PIPS: float = max(0.0, _float("PULLBACK_S5_DAILY_PNL_STOP_PIPS", 12.0))
MAX_CONSEC_LOSSES: int = max(0, _int("PULLBACK_S5_MAX_CONSEC_LOSSES", 3))
PERFORMANCE_REFRESH_SEC: float = max(10.0, _float("PULLBACK_S5_PERF_REFRESH_SEC", 60.0))
INSTANT_MOVE_PIPS_LIMIT: float = max(0.0, _float("PULLBACK_S5_INSTANT_MOVE_PIPS_LIMIT", 1.2))
TICK_GAP_MS_LIMIT: float = max(0.0, _float("PULLBACK_S5_TICK_GAP_MS_LIMIT", 150.0))
TICK_GAP_MOVE_PIPS: float = max(0.0, _float("PULLBACK_S5_TICK_GAP_MOVE_PIPS", 0.6))
