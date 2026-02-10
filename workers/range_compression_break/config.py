"""Configuration for Range Compression Break worker."""

from __future__ import annotations

import os

PIP_VALUE = 0.01
ENV_PREFIX = "RANGE_COMPRESSION_BREAK"
LOG_PREFIX = "[RANGE-COMP-BREAK]"
STRATEGY_TAG = "RangeCompressionBreak"
PROFILE_TAG = "range_compression_break"


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


ENABLED: bool = _bool("RANGE_COMPRESSION_BREAK_ENABLED", False)
LOOP_INTERVAL_SEC: float = max(0.5, _float("RANGE_COMPRESSION_BREAK_LOOP_INTERVAL_SEC", 6.0))
POCKET: str = "micro"
MODE: str = "trend"

BASE_ENTRY_UNITS: int = max(1000, _int("RANGE_COMPRESSION_BREAK_BASE_ENTRY_UNITS", 20000))
MIN_UNITS: int = max(500, _int("RANGE_COMPRESSION_BREAK_MIN_UNITS", 1500))
CONFIDENCE_FLOOR: int = max(0, _int("RANGE_COMPRESSION_BREAK_CONFIDENCE_FLOOR", 40))
CONFIDENCE_CEIL: int = max(CONFIDENCE_FLOOR + 1, _int("RANGE_COMPRESSION_BREAK_CONFIDENCE_CEIL", 92))

CAP_MIN: float = max(0.05, _float("RANGE_COMPRESSION_BREAK_CAP_MIN", 0.15))
CAP_MAX: float = max(CAP_MIN, _float("RANGE_COMPRESSION_BREAK_CAP_MAX", 0.95))
MAX_FACTOR_AGE_SEC: float = max(10.0, _float("RANGE_COMPRESSION_BREAK_MAX_FACTOR_AGE_SEC", 90.0))

BBW_MAX: float = max(0.0001, _float("RANGE_COMPRESSION_BREAK_BBW_MAX", 0.0018))
KC_WIDTH_MAX: float = max(0.0001, _float("RANGE_COMPRESSION_BREAK_KC_WIDTH_MAX", 0.012))
DONCHIAN_WIDTH_MAX: float = max(0.0001, _float("RANGE_COMPRESSION_BREAK_DONCHIAN_WIDTH_MAX", 0.012))
ADX_MAX: float = max(0.0, _float("RANGE_COMPRESSION_BREAK_ADX_MAX", 18.0))

NWAVE_MIN_QUALITY: float = max(0.0, _float("RANGE_COMPRESSION_BREAK_NWAVE_MIN_QUALITY", 0.2))
NWAVE_MIN_LEG_PIPS: float = max(0.5, _float("RANGE_COMPRESSION_BREAK_NWAVE_MIN_LEG_PIPS", 3.0))
DONCHIAN_LOOKBACK: int = max(8, _int("RANGE_COMPRESSION_BREAK_DONCHIAN_LOOKBACK", 20))
DONCHIAN_NEAR_PIPS: float = max(0.5, _float("RANGE_COMPRESSION_BREAK_DONCHIAN_NEAR_PIPS", 4.0))

SL_MIN_PIPS: float = max(1.0, _float("RANGE_COMPRESSION_BREAK_SL_MIN_PIPS", 6.0))
SL_MAX_PIPS: float = max(SL_MIN_PIPS, _float("RANGE_COMPRESSION_BREAK_SL_MAX_PIPS", 16.0))
SL_ATR_MULT: float = max(0.2, _float("RANGE_COMPRESSION_BREAK_SL_ATR_MULT", 1.8))
TP_MIN_PIPS: float = max(1.0, _float("RANGE_COMPRESSION_BREAK_TP_MIN_PIPS", 12.0))
TP_MAX_PIPS: float = max(TP_MIN_PIPS, _float("RANGE_COMPRESSION_BREAK_TP_MAX_PIPS", 32.0))
TP_RR: float = max(0.6, _float("RANGE_COMPRESSION_BREAK_TP_RR", 1.6))
SPREAD_FLOOR_PIPS: float = max(0.0, _float("RANGE_COMPRESSION_BREAK_SPREAD_FLOOR_PIPS", 10.0))

EXIT_LOOP_INTERVAL_SEC: float = max(0.4, _float("RANGE_COMPRESSION_BREAK_EXIT_LOOP_INTERVAL_SEC", 0.9))
EXIT_MIN_HOLD_SEC: float = max(10.0, _float("RANGE_COMPRESSION_BREAK_EXIT_MIN_HOLD_SEC", 120.0))
EXIT_MAX_HOLD_SEC: float = max(EXIT_MIN_HOLD_SEC, _float("RANGE_COMPRESSION_BREAK_EXIT_MAX_HOLD_SEC", 1800.0))
EXIT_PROFIT_PIPS: float = max(0.8, _float("RANGE_COMPRESSION_BREAK_EXIT_PROFIT_PIPS", 10.0))
EXIT_TRAIL_START_PIPS: float = max(1.0, _float("RANGE_COMPRESSION_BREAK_EXIT_TRAIL_START_PIPS", 12.0))
EXIT_TRAIL_BACKOFF_PIPS: float = max(0.2, _float("RANGE_COMPRESSION_BREAK_EXIT_TRAIL_BACKOFF_PIPS", 4.0))
EXIT_LOCK_BUFFER_PIPS: float = max(0.2, _float("RANGE_COMPRESSION_BREAK_EXIT_LOCK_BUFFER_PIPS", 3.0))
EXIT_LOCK_TRIGGER_PIPS: float = max(0.5, _float("RANGE_COMPRESSION_BREAK_EXIT_LOCK_TRIGGER_PIPS", 6.0))
EXIT_REVERSAL_WINDOW_SEC: float = max(2.0, _float("RANGE_COMPRESSION_BREAK_EXIT_REV_WINDOW_SEC", 45.0))
EXIT_REVERSAL_PIPS: float = max(0.5, _float("RANGE_COMPRESSION_BREAK_EXIT_REV_PIPS", 8.0))
EXIT_ATR_SPIKE_PIPS: float = max(1.0, _float("RANGE_COMPRESSION_BREAK_EXIT_ATR_SPIKE_PIPS", 12.0))
EXIT_HARD_STOP_PIPS: float = max(1.0, _float("RANGE_COMPRESSION_BREAK_EXIT_HARD_STOP_PIPS", 12.0))
EXIT_TP_HINT_RATIO: float = max(0.2, _float("RANGE_COMPRESSION_BREAK_EXIT_TP_HINT_RATIO", 0.75))

TECH_FAILOPEN: bool = _bool("RANGE_COMPRESSION_BREAK_TECH_FAILOPEN", False)
TECH_CONF_BOOST: float = max(0.0, _float("RANGE_COMPRESSION_BREAK_TECH_CONF_BOOST", 18.0))
TECH_CONF_PENALTY: float = max(0.0, _float("RANGE_COMPRESSION_BREAK_TECH_CONF_PENALTY", 10.0))
