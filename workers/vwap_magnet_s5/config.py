"""Configuration for the VWAP magnet S5 worker."""

from __future__ import annotations

import os
from typing import Set

PIP_VALUE = 0.01
ENV_PREFIX = "VWAP_MAGNET_S5"


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


LOG_PREFIX = "[VWAP-S5]"
ENABLED: bool = _bool("VWAP_MAGNET_S5_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("VWAP_MAGNET_S5_LOOP_INTERVAL_SEC", 0.45))

ACTIVE_HOURS_UTC = frozenset(range(24))
ALLOWED_HOURS_UTC = ACTIVE_HOURS_UTC
BLOCKED_WEEKDAYS = tuple(
    day.strip()
    for day in os.getenv("VWAP_MAGNET_S5_BLOCKED_WEEKDAYS", "4").split(",")
    if day.strip()
)

WINDOW_SEC: float = max(60.0, _float("VWAP_MAGNET_S5_WINDOW_SEC", 600.0))
BUCKET_SECONDS: float = max(1.0, _float("VWAP_MAGNET_S5_BUCKET_SECONDS", 5.0))
VWAP_WINDOW_BUCKETS: int = max(30, _int("VWAP_MAGNET_S5_VWAP_BUCKETS", 96))
# Require at least the VWAP window, but not an extra margin, to ease warmup after restart.
MIN_BUCKETS: int = max(VWAP_WINDOW_BUCKETS, _int("VWAP_MAGNET_S5_MIN_BUCKETS", VWAP_WINDOW_BUCKETS))
# Warmup allowance: start計算できる最小バケット数（再起動直後用）
WARMUP_MIN_BUCKETS: int = max(20, _int("VWAP_MAGNET_S5_WARMUP_MIN_BUCKETS", 48))

MAX_SPREAD_PIPS: float = max(0.1, _float("VWAP_MAGNET_S5_MAX_SPREAD_PIPS", 1.2))
MIN_DENSITY_TICKS: int = max(10, _int("VWAP_MAGNET_S5_MIN_DENSITY_TICKS", 60))
ENTRY_UNITS: int = max(1000, _int("VWAP_MAGNET_S5_ENTRY_UNITS", 5000))
ENTRY_STAGE_RATIOS: tuple[float, ...] = _parse_float_list(
    "VWAP_MAGNET_S5_ENTRY_STAGE_RATIOS", "1.0"
)
ALLOW_DUPLICATE_ENTRIES: bool = _bool(
    "VWAP_MAGNET_S5_ALLOW_DUPLICATE_ENTRIES", False
)
MAX_ACTIVE_TRADES: int = max(1, _int("VWAP_MAGNET_S5_MAX_ACTIVE_TRADES", len(ENTRY_STAGE_RATIOS)))
MAX_ACTIVE_TRADES = min(MAX_ACTIVE_TRADES, len(ENTRY_STAGE_RATIOS))
COOLDOWN_SEC: float = max(20.0, _float("VWAP_MAGNET_S5_COOLDOWN_SEC", 120.0))
POST_EXIT_COOLDOWN_SEC: float = max(20.0, _float("VWAP_MAGNET_S5_POST_EXIT_COOLDOWN_SEC", 240.0))

Z_ENTRY_SIGMA: float = max(0.5, _float("VWAP_MAGNET_S5_Z_ENTRY_SIGMA", 2.5))
RSI_PERIOD: int = max(5, _int("VWAP_MAGNET_S5_RSI_PERIOD", 14))
RSI_SHORT_RANGE = (
    float(_float("VWAP_MAGNET_S5_RSI_SHORT_MIN", 48.0)),
    float(_float("VWAP_MAGNET_S5_RSI_SHORT_MAX", 66.0)),
)
RSI_LONG_RANGE = (
    float(_float("VWAP_MAGNET_S5_RSI_LONG_MIN", 34.0)),
    float(_float("VWAP_MAGNET_S5_RSI_LONG_MAX", 52.0)),
)
MIN_ATR_PIPS: float = max(0.0, _float("VWAP_MAGNET_S5_MIN_ATR_PIPS", 0.6))

TP_PIPS: float = max(1.0, _float("VWAP_MAGNET_S5_TP_PIPS", 1.4))
SL_ATR_MULT: float = max(0.2, _float("VWAP_MAGNET_S5_SL_ATR_MULT", 0.7))
SL_MIN_PIPS: float = max(0.5, _float("VWAP_MAGNET_S5_SL_MIN_PIPS", 1.2))

STAGE_MIN_DELTA_PIPS: float = max(0.05, _float("VWAP_MAGNET_S5_STAGE_MIN_DELTA_PIPS", 0.24))
STAGE_FAVORABLE_PIPS: float = max(0.0, _float("VWAP_MAGNET_S5_STAGE_FAVORABLE_PIPS", 0.25))

BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("VWAP_MAGNET_S5_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)
LOSS_STREAK_MAX: int = max(0, _int("VWAP_MAGNET_S5_MAX_CONSEC_LOSSES", 3))
LOSS_STREAK_COOLDOWN_MIN: float = max(0.0, _float("VWAP_MAGNET_S5_LOSS_COOLDOWN_MIN", 20.0))

SPREAD_P50_LIMIT: float = max(0.0, _float("VWAP_MAGNET_S5_SPREAD_P50_LIMIT", 0.14))
RETURN_WINDOW_SEC: float = max(5.0, _float("VWAP_MAGNET_S5_RETURN_WINDOW_SEC", 18.0))
RETURN_PIPS_LIMIT: float = max(0.0, _float("VWAP_MAGNET_S5_RETURN_PIPS_LIMIT", 0.3))

TREND_ALIGN_BUFFER_PIPS: float = max(0.0, _float("VWAP_MAGNET_S5_TREND_ALIGN_BUFFER_PIPS", 0.45))
TREND_ADX_MIN: float = max(0.0, _float("VWAP_MAGNET_S5_TREND_ADX_MIN", 16.0))
MA_FAST_BUCKETS: int = max(10, _int("VWAP_MAGNET_S5_MA_FAST_BUCKETS", 36))
MA_SLOW_BUCKETS: int = max(MA_FAST_BUCKETS + 4, _int("VWAP_MAGNET_S5_MA_SLOW_BUCKETS", 96))
MA_DIFF_PIPS: float = max(0.0, _float("VWAP_MAGNET_S5_MA_DIFF_PIPS", 0.8))
INSTANT_MOVE_PIPS_LIMIT: float = max(0.0, _float("VWAP_MAGNET_S5_INSTANT_MOVE_PIPS_LIMIT", 1.5))
TICK_GAP_MS_LIMIT: float = max(0.0, _float("VWAP_MAGNET_S5_TICK_GAP_MS_LIMIT", 150.0))
TICK_GAP_MOVE_PIPS: float = max(0.0, _float("VWAP_MAGNET_S5_TICK_GAP_MOVE_PIPS", 0.6))

DAILY_PNL_STOP_PIPS: float = max(0.0, _float("VWAP_MAGNET_S5_DAILY_PNL_STOP_PIPS", 8.0))
MAX_CONSEC_LOSSES: int = max(0, _int("VWAP_MAGNET_S5_MAX_CONSEC_LOSSES", 3))
PERFORMANCE_REFRESH_SEC: float = max(10.0, _float("VWAP_MAGNET_S5_PERF_REFRESH_SEC", 60.0))
