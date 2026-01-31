"""Configuration for Micro Pullback-Fib worker."""

from __future__ import annotations

import os

PIP_VALUE = 0.01
LOG_PREFIX = "[MICRO-PULLBACK-FIB]"
STRATEGY_TAG = "MicroPullbackFib"
PROFILE_TAG = "micro_pullback_fib"


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


ENABLED: bool = _bool("MICRO_PULLBACK_FIB_ENABLED", False)
LOOP_INTERVAL_SEC: float = max(0.5, _float("MICRO_PULLBACK_FIB_LOOP_INTERVAL_SEC", 6.0))
POCKET: str = "micro"
MODE: str = "pullback"

BASE_ENTRY_UNITS: int = max(1000, _int("MICRO_PULLBACK_FIB_BASE_ENTRY_UNITS", 20000))
MIN_UNITS: int = max(500, _int("MICRO_PULLBACK_FIB_MIN_UNITS", 1500))
CONFIDENCE_FLOOR: int = max(0, _int("MICRO_PULLBACK_FIB_CONFIDENCE_FLOOR", 40))
CONFIDENCE_CEIL: int = max(CONFIDENCE_FLOOR + 1, _int("MICRO_PULLBACK_FIB_CONFIDENCE_CEIL", 92))

CAP_MIN: float = max(0.05, _float("MICRO_PULLBACK_FIB_CAP_MIN", 0.15))
CAP_MAX: float = max(CAP_MIN, _float("MICRO_PULLBACK_FIB_CAP_MAX", 0.95))
MAX_FACTOR_AGE_SEC: float = max(10.0, _float("MICRO_PULLBACK_FIB_MAX_FACTOR_AGE_SEC", 90.0))

TREND_ADX_MIN: float = max(0.0, _float("MICRO_PULLBACK_FIB_TREND_ADX_MIN", 16.0))
TREND_SCORE_MIN: float = max(0.0, _float("MICRO_PULLBACK_FIB_TREND_SCORE_MIN", 0.55))
PULLBACK_RSI_MIN: float = max(0.0, _float("MICRO_PULLBACK_FIB_RSI_MIN", 40.0))
PULLBACK_RSI_MAX: float = max(PULLBACK_RSI_MIN + 1.0, _float("MICRO_PULLBACK_FIB_RSI_MAX", 58.0))
PULLBACK_MA_BUFFER_PIPS: float = max(0.0, _float("MICRO_PULLBACK_FIB_MA_BUFFER_PIPS", 0.6))

SL_MIN_PIPS: float = max(1.0, _float("MICRO_PULLBACK_FIB_SL_MIN_PIPS", 6.0))
SL_MAX_PIPS: float = max(SL_MIN_PIPS, _float("MICRO_PULLBACK_FIB_SL_MAX_PIPS", 18.0))
SL_ATR_MULT: float = max(0.2, _float("MICRO_PULLBACK_FIB_SL_ATR_MULT", 1.6))
TP_MIN_PIPS: float = max(1.0, _float("MICRO_PULLBACK_FIB_TP_MIN_PIPS", 10.0))
TP_MAX_PIPS: float = max(TP_MIN_PIPS, _float("MICRO_PULLBACK_FIB_TP_MAX_PIPS", 30.0))
TP_RR: float = max(0.6, _float("MICRO_PULLBACK_FIB_TP_RR", 1.5))
SPREAD_FLOOR_PIPS: float = max(0.0, _float("MICRO_PULLBACK_FIB_SPREAD_FLOOR_PIPS", 10.0))

EXIT_LOOP_INTERVAL_SEC: float = max(0.4, _float("MICRO_PULLBACK_FIB_EXIT_LOOP_INTERVAL_SEC", 0.9))
EXIT_MIN_HOLD_SEC: float = max(10.0, _float("MICRO_PULLBACK_FIB_EXIT_MIN_HOLD_SEC", 120.0))
EXIT_MAX_HOLD_SEC: float = max(EXIT_MIN_HOLD_SEC, _float("MICRO_PULLBACK_FIB_EXIT_MAX_HOLD_SEC", 1800.0))
EXIT_PROFIT_PIPS: float = max(0.8, _float("MICRO_PULLBACK_FIB_EXIT_PROFIT_PIPS", 8.0))
EXIT_TRAIL_START_PIPS: float = max(1.0, _float("MICRO_PULLBACK_FIB_EXIT_TRAIL_START_PIPS", 10.0))
EXIT_TRAIL_BACKOFF_PIPS: float = max(0.2, _float("MICRO_PULLBACK_FIB_EXIT_TRAIL_BACKOFF_PIPS", 3.5))
EXIT_LOCK_BUFFER_PIPS: float = max(0.2, _float("MICRO_PULLBACK_FIB_EXIT_LOCK_BUFFER_PIPS", 2.8))
EXIT_LOCK_TRIGGER_PIPS: float = max(0.5, _float("MICRO_PULLBACK_FIB_EXIT_LOCK_TRIGGER_PIPS", 5.0))
EXIT_REVERSAL_WINDOW_SEC: float = max(2.0, _float("MICRO_PULLBACK_FIB_EXIT_REV_WINDOW_SEC", 45.0))
EXIT_REVERSAL_PIPS: float = max(0.5, _float("MICRO_PULLBACK_FIB_EXIT_REV_PIPS", 7.0))
EXIT_ATR_SPIKE_PIPS: float = max(1.0, _float("MICRO_PULLBACK_FIB_EXIT_ATR_SPIKE_PIPS", 12.0))
EXIT_HARD_STOP_PIPS: float = max(1.0, _float("MICRO_PULLBACK_FIB_EXIT_HARD_STOP_PIPS", 12.0))
EXIT_TP_HINT_RATIO: float = max(0.2, _float("MICRO_PULLBACK_FIB_EXIT_TP_HINT_RATIO", 0.75))

TECH_FAILOPEN: bool = _bool("MICRO_PULLBACK_FIB_TECH_FAILOPEN", False)
TECH_CONF_BOOST: float = max(0.0, _float("MICRO_PULLBACK_FIB_TECH_CONF_BOOST", 18.0))
TECH_CONF_PENALTY: float = max(0.0, _float("MICRO_PULLBACK_FIB_TECH_CONF_PENALTY", 10.0))
