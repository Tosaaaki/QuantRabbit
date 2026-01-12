"""Configuration for the pullback scalp worker."""

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


LOG_PREFIX = "[PULLBACK-SCALP]"
ENABLED: bool = _bool("PULLBACK_SCALP_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.15, _float("PULLBACK_SCALP_LOOP_INTERVAL_SEC", 0.30))

ACTIVE_HOURS_UTC = frozenset(range(24))

MAX_SPREAD_PIPS: float = max(0.1, _float("PULLBACK_SCALP_MAX_SPREAD_PIPS", 1.0))
ENTRY_UNITS: int = max(1000, _int("PULLBACK_SCALP_ENTRY_UNITS", 7000))
MAX_ACTIVE_TRADES: int = max(1, _int("PULLBACK_SCALP_MAX_ACTIVE_TRADES", 2))
STAGE_MIN_DELTA_PIPS: float = max(0.05, _float("PULLBACK_SCALP_STAGE_MIN_DELTA_PIPS", 0.14))
COOLDOWN_SEC: float = max(10.0, _float("PULLBACK_SCALP_COOLDOWN_SEC", 45.0))

M1_WINDOW_SEC: float = max(20.0, _float("PULLBACK_SCALP_M1_WINDOW_SEC", 60.0))
M5_WINDOW_SEC: float = max(60.0, _float("PULLBACK_SCALP_M5_WINDOW_SEC", 300.0))
M1_Z_MIN: float = _float("PULLBACK_SCALP_M1_Z_MIN", 0.16)
M1_Z_MAX: float = _float("PULLBACK_SCALP_M1_Z_MAX", 0.70)
M1_Z_TRIGGER: float = _float("PULLBACK_SCALP_M1_Z_TRIGGER", 0.0)
M5_Z_SHORT_MAX: float = _float("PULLBACK_SCALP_M5_Z_SHORT_MAX", 0.35)
M5_Z_LONG_MIN: float = _float("PULLBACK_SCALP_M5_Z_LONG_MIN", -0.35)

RSI_PERIOD: int = max(5, _int("PULLBACK_SCALP_RSI_PERIOD", 14))
RSI_SHORT_RANGE = (
    float(_float("PULLBACK_SCALP_RSI_SHORT_MIN", 44.0)),
    float(_float("PULLBACK_SCALP_RSI_SHORT_MAX", 70.0)),
)
RSI_LONG_RANGE = (
    float(_float("PULLBACK_SCALP_RSI_LONG_MIN", 30.0)),
    float(_float("PULLBACK_SCALP_RSI_LONG_MAX", 56.0)),
)

MIN_ATR_PIPS: float = max(0.0, _float("PULLBACK_SCALP_MIN_ATR_PIPS", 0.40))

TP_PIPS: float = max(1.0, _float("PULLBACK_SCALP_TP_PIPS", 2.3))
USE_INITIAL_SL: bool = _bool("PULLBACK_SCALP_USE_INITIAL_SL", True)
SL_ATR_MULT: float = max(0.5, _float("PULLBACK_SCALP_SL_ATR_MULT", 1.15))
SL_MIN_FLOOR_PIPS: float = max(0.2, _float("PULLBACK_SCALP_SL_MIN_FLOOR_PIPS", 0.7))
ATR_SLOW_WEIGHT: float = max(0.0, min(1.0, _float("PULLBACK_SCALP_ATR_SLOW_WEIGHT", 0.6)))
ATR_FAST_WINDOW_SEC: float = max(10.0, _float("PULLBACK_SCALP_ATR_FAST_WINDOW_SEC", 45.0))
ATR_SLOW_WINDOW_SEC: float = max(
    ATR_FAST_WINDOW_SEC, _float("PULLBACK_SCALP_ATR_SLOW_WINDOW_SEC", 120.0)
)
HIGH_VOL_ATR_PIPS: float = max(0.0, _float("PULLBACK_SCALP_HIGH_VOL_ATR_PIPS", 1.5))
HIGH_VOL_TP_PIPS: float = max(0.6, _float("PULLBACK_SCALP_HIGH_VOL_TP_PIPS", 1.4))
HIGH_VOL_SL_CAP_PIPS: float = max(0.8, _float("PULLBACK_SCALP_HIGH_VOL_SL_CAP_PIPS", 1.8))
HIGH_VOL_UNIT_FACTOR: float = max(0.2, min(1.0, _float("PULLBACK_SCALP_HIGH_VOL_UNIT_FACTOR", 0.6)))

# pullback touch counter
TOUCH_ENABLED: bool = _bool("PULLBACK_SCALP_TOUCH_ENABLED", True)
TOUCH_WINDOW_SEC: float = max(20.0, _float("PULLBACK_SCALP_TOUCH_WINDOW_SEC", 120.0))
TOUCH_PULLBACK_ATR_MULT: float = max(0.1, _float("PULLBACK_SCALP_TOUCH_PULLBACK_ATR_MULT", 0.85))
TOUCH_PULLBACK_MIN_PIPS: float = max(0.1, _float("PULLBACK_SCALP_TOUCH_PULLBACK_MIN_PIPS", 0.45))
TOUCH_PULLBACK_MAX_PIPS: float = max(
    TOUCH_PULLBACK_MIN_PIPS,
    _float("PULLBACK_SCALP_TOUCH_PULLBACK_MAX_PIPS", 1.8),
)
TOUCH_TREND_ATR_MULT: float = max(0.1, _float("PULLBACK_SCALP_TOUCH_TREND_ATR_MULT", 1.5))
TOUCH_TREND_MIN_PIPS: float = max(0.2, _float("PULLBACK_SCALP_TOUCH_TREND_MIN_PIPS", 0.9))
TOUCH_TREND_MAX_PIPS: float = max(
    TOUCH_TREND_MIN_PIPS,
    _float("PULLBACK_SCALP_TOUCH_TREND_MAX_PIPS", 3.8),
)
TOUCH_RESET_RATIO: float = max(0.1, min(0.9, _float("PULLBACK_SCALP_TOUCH_RESET_RATIO", 0.45)))
TOUCH_SOFT_COUNT: int = max(1, _int("PULLBACK_SCALP_TOUCH_SOFT_COUNT", 2))
TOUCH_HARD_COUNT: int = max(TOUCH_SOFT_COUNT + 1, _int("PULLBACK_SCALP_TOUCH_HARD_COUNT", 4))
TOUCH_UNIT_FACTOR: float = max(0.1, min(1.0, _float("PULLBACK_SCALP_TOUCH_UNIT_FACTOR", 0.7)))
TOUCH_CONF_PENALTY: int = max(0, _int("PULLBACK_SCALP_TOUCH_CONF_PENALTY", 8))
