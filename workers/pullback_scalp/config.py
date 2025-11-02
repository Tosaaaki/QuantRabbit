"""Configuration for the pullback scalp worker."""

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


LOG_PREFIX = "[PULLBACK-SCALP]"
ENABLED: bool = _bool("PULLBACK_SCALP_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.15, _float("PULLBACK_SCALP_LOOP_INTERVAL_SEC", 0.35))

MAX_SPREAD_PIPS: float = max(0.1, _float("PULLBACK_SCALP_MAX_SPREAD_PIPS", 0.85))
ENTRY_UNITS: int = max(1000, _int("PULLBACK_SCALP_ENTRY_UNITS", 8000))
MAX_ACTIVE_TRADES: int = max(1, _int("PULLBACK_SCALP_MAX_ACTIVE_TRADES", 1))
STAGE_MIN_DELTA_PIPS: float = max(0.05, _float("PULLBACK_SCALP_STAGE_MIN_DELTA_PIPS", 0.24))
COOLDOWN_SEC: float = max(10.0, _float("PULLBACK_SCALP_COOLDOWN_SEC", 120.0))

M1_WINDOW_SEC: float = max(20.0, _float("PULLBACK_SCALP_M1_WINDOW_SEC", 75.0))
M5_WINDOW_SEC: float = max(60.0, _float("PULLBACK_SCALP_M5_WINDOW_SEC", 330.0))
M1_Z_MIN: float = _float("PULLBACK_SCALP_M1_Z_MIN", 0.18)
M1_Z_MAX: float = _float("PULLBACK_SCALP_M1_Z_MAX", 0.60)
M1_Z_TRIGGER: float = _float("PULLBACK_SCALP_M1_Z_TRIGGER", 0.0)
M5_Z_SHORT_MAX: float = _float("PULLBACK_SCALP_M5_Z_SHORT_MAX", 0.25)
M5_Z_LONG_MIN: float = _float("PULLBACK_SCALP_M5_Z_LONG_MIN", -0.25)

RSI_PERIOD: int = max(5, _int("PULLBACK_SCALP_RSI_PERIOD", 14))
RSI_SHORT_RANGE = (
    float(_float("PULLBACK_SCALP_RSI_SHORT_MIN", 46.0)),
    float(_float("PULLBACK_SCALP_RSI_SHORT_MAX", 68.0)),
)
RSI_LONG_RANGE = (
    float(_float("PULLBACK_SCALP_RSI_LONG_MIN", 32.0)),
    float(_float("PULLBACK_SCALP_RSI_LONG_MAX", 54.0)),
)

MIN_ATR_PIPS: float = max(0.0, _float("PULLBACK_SCALP_MIN_ATR_PIPS", 0.0))

TP_PIPS: float = max(1.0, _float("PULLBACK_SCALP_TP_PIPS", 2.3))
USE_INITIAL_SL: bool = _bool("PULLBACK_SCALP_USE_INITIAL_SL", True)
MIN_SL_PIPS: float = max(1.0, _float("PULLBACK_SCALP_MIN_SL_PIPS", 2.7))
SL_ATR_MULT: float = max(0.5, _float("PULLBACK_SCALP_SL_ATR_MULT", 1.15))
