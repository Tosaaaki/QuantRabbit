"""Configuration for the VWAP magnet S5 worker."""

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

ACTIVE_HOURS_UTC = frozenset(
    _parse_hours("VWAP_MAGNET_S5_ACTIVE_HOURS", "1,4-8,12,17,19")
)

WINDOW_SEC: float = max(60.0, _float("VWAP_MAGNET_S5_WINDOW_SEC", 360.0))
BUCKET_SECONDS: float = max(1.0, _float("VWAP_MAGNET_S5_BUCKET_SECONDS", 5.0))
VWAP_WINDOW_BUCKETS: int = max(30, _int("VWAP_MAGNET_S5_VWAP_BUCKETS", 96))
MIN_BUCKETS: int = max(VWAP_WINDOW_BUCKETS + 4, _int("VWAP_MAGNET_S5_MIN_BUCKETS", 120))

MAX_SPREAD_PIPS: float = max(0.1, _float("VWAP_MAGNET_S5_MAX_SPREAD_PIPS", 0.8))
MIN_DENSITY_TICKS: int = max(10, _int("VWAP_MAGNET_S5_MIN_DENSITY_TICKS", 60))
ENTRY_UNITS: int = max(1000, _int("VWAP_MAGNET_S5_ENTRY_UNITS", 10000))
ENTRY_STAGE_RATIOS: tuple[float, ...] = _parse_float_list(
    "VWAP_MAGNET_S5_ENTRY_STAGE_RATIOS", "0.4,0.35,0.25"
)
ALLOW_DUPLICATE_ENTRIES: bool = _bool(
    "VWAP_MAGNET_S5_ALLOW_DUPLICATE_ENTRIES", True
)
MAX_ACTIVE_TRADES: int = max(1, _int("VWAP_MAGNET_S5_MAX_ACTIVE_TRADES", 9))
COOLDOWN_SEC: float = max(20.0, _float("VWAP_MAGNET_S5_COOLDOWN_SEC", 120.0))
POST_EXIT_COOLDOWN_SEC: float = max(20.0, _float("VWAP_MAGNET_S5_POST_EXIT_COOLDOWN_SEC", 240.0))

Z_ENTRY_SIGMA: float = max(0.5, _float("VWAP_MAGNET_S5_Z_ENTRY_SIGMA", 1.25))
RSI_PERIOD: int = max(5, _int("VWAP_MAGNET_S5_RSI_PERIOD", 14))
RSI_SHORT_RANGE = (
    float(_float("VWAP_MAGNET_S5_RSI_SHORT_MIN", 46.0)),
    float(_float("VWAP_MAGNET_S5_RSI_SHORT_MAX", 68.0)),
)
RSI_LONG_RANGE = (
    float(_float("VWAP_MAGNET_S5_RSI_LONG_MIN", 32.0)),
    float(_float("VWAP_MAGNET_S5_RSI_LONG_MAX", 54.0)),
)
MIN_ATR_PIPS: float = max(0.0, _float("VWAP_MAGNET_S5_MIN_ATR_PIPS", 0.5))

TP_PIPS: float = max(1.0, _float("VWAP_MAGNET_S5_TP_PIPS", 1.8))
SL_ATR_MULT: float = max(0.2, _float("VWAP_MAGNET_S5_SL_ATR_MULT", 1.2))
SL_MIN_PIPS: float = max(0.5, _float("VWAP_MAGNET_S5_SL_MIN_PIPS", 2.8))

STAGE_MIN_DELTA_PIPS: float = max(0.05, _float("VWAP_MAGNET_S5_STAGE_MIN_DELTA_PIPS", 0.24))

BLOCK_REGIMES = tuple(
    regime.strip()
    for regime in os.getenv("VWAP_MAGNET_S5_BLOCK_REGIMES", "Event").split(",")
    if regime.strip()
)
NEWS_BLOCK_MINUTES: float = max(0.0, _float("VWAP_MAGNET_S5_NEWS_BLOCK_MINUTES", 20.0))
NEWS_BLOCK_MIN_IMPACT: int = max(1, _int("VWAP_MAGNET_S5_NEWS_BLOCK_MIN_IMPACT", 2))
LOSS_STREAK_MAX: int = max(0, _int("VWAP_MAGNET_S5_MAX_CONSEC_LOSSES", 3))
LOSS_STREAK_COOLDOWN_MIN: float = max(0.0, _float("VWAP_MAGNET_S5_LOSS_COOLDOWN_MIN", 20.0))
