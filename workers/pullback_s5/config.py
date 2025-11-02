"""Configuration for the S5 pullback worker."""

from __future__ import annotations

import os

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


LOG_PREFIX = "[PULLBACK-S5]"
ENABLED: bool = _bool("PULLBACK_S5_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("PULLBACK_S5_LOOP_INTERVAL_SEC", 0.45))

WINDOW_SEC: float = max(60.0, _float("PULLBACK_S5_WINDOW_SEC", 360.0))
BUCKET_SECONDS: float = max(1.0, _float("PULLBACK_S5_BUCKET_SECONDS", 5.0))
FAST_BUCKETS: int = max(24, _int("PULLBACK_S5_FAST_BUCKETS", 48))
SLOW_BUCKETS: int = max(FAST_BUCKETS + 8, _int("PULLBACK_S5_SLOW_BUCKETS", 72))
MIN_BUCKETS: int = max(FAST_BUCKETS, _int("PULLBACK_S5_MIN_BUCKETS", 60))

MAX_SPREAD_PIPS: float = max(0.1, _float("PULLBACK_S5_MAX_SPREAD_PIPS", 0.85))
ENTRY_UNITS: int = max(1000, _int("PULLBACK_S5_ENTRY_UNITS", 6000))
MAX_ACTIVE_TRADES: int = max(1, _int("PULLBACK_S5_MAX_ACTIVE_TRADES", 1))
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

MIN_ATR_PIPS: float = max(0.0, _float("PULLBACK_S5_MIN_ATR_PIPS", 0.4))
MIN_DENSITY_TICKS: int = max(10, _int("PULLBACK_S5_MIN_DENSITY_TICKS", 50))

