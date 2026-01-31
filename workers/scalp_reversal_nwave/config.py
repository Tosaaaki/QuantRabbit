"""Configuration for Scalp Reversal-NWave worker."""

from __future__ import annotations

import os

PIP_VALUE = 0.01
LOG_PREFIX = "[SCALP-REV-NWAVE]"
STRATEGY_TAG = "ScalpReversalNWave"
PROFILE_TAG = "scalp_reversal_nwave"


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


ENABLED: bool = _bool("SCALP_REVERSAL_NWAVE_ENABLED", False)
LOOP_INTERVAL_SEC: float = max(0.1, _float("SCALP_REVERSAL_NWAVE_LOOP_INTERVAL_SEC", 0.8))
POCKET: str = "scalp"
MODE: str = "reversal"

BASE_ENTRY_UNITS: int = max(500, _int("SCALP_REVERSAL_NWAVE_BASE_ENTRY_UNITS", 6000))
MIN_UNITS: int = max(500, _int("SCALP_REVERSAL_NWAVE_MIN_UNITS", 900))
CONFIDENCE_FLOOR: int = max(0, _int("SCALP_REVERSAL_NWAVE_CONFIDENCE_FLOOR", 38))
CONFIDENCE_CEIL: int = max(CONFIDENCE_FLOOR + 1, _int("SCALP_REVERSAL_NWAVE_CONFIDENCE_CEIL", 90))

CAP_MIN: float = max(0.05, _float("SCALP_REVERSAL_NWAVE_CAP_MIN", 0.12))
CAP_MAX: float = max(CAP_MIN, _float("SCALP_REVERSAL_NWAVE_CAP_MAX", 0.85))
MAX_FACTOR_AGE_SEC: float = max(5.0, _float("SCALP_REVERSAL_NWAVE_MAX_FACTOR_AGE_SEC", 45.0))

VWAP_GAP_MIN_PIPS: float = max(0.5, _float("SCALP_REVERSAL_NWAVE_VWAP_GAP_MIN_PIPS", 3.5))
RSI_MIN_LONG: float = max(1.0, _float("SCALP_REVERSAL_NWAVE_RSI_MIN_LONG", 32.0))
RSI_MAX_SHORT: float = max(RSI_MIN_LONG + 1.0, _float("SCALP_REVERSAL_NWAVE_RSI_MAX_SHORT", 68.0))

SL_MIN_PIPS: float = max(0.5, _float("SCALP_REVERSAL_NWAVE_SL_MIN_PIPS", 2.0))
SL_MAX_PIPS: float = max(SL_MIN_PIPS, _float("SCALP_REVERSAL_NWAVE_SL_MAX_PIPS", 5.0))
SL_ATR_MULT: float = max(0.1, _float("SCALP_REVERSAL_NWAVE_SL_ATR_MULT", 1.1))
TP_MIN_PIPS: float = max(0.6, _float("SCALP_REVERSAL_NWAVE_TP_MIN_PIPS", 3.0))
TP_MAX_PIPS: float = max(TP_MIN_PIPS, _float("SCALP_REVERSAL_NWAVE_TP_MAX_PIPS", 7.0))
TP_RR: float = max(0.5, _float("SCALP_REVERSAL_NWAVE_TP_RR", 1.3))
SPREAD_FLOOR_PIPS: float = max(0.0, _float("SCALP_REVERSAL_NWAVE_SPREAD_FLOOR_PIPS", 4.0))

EXIT_LOOP_INTERVAL_SEC: float = max(0.1, _float("SCALP_REVERSAL_NWAVE_EXIT_LOOP_INTERVAL_SEC", 0.5))
EXIT_MIN_HOLD_SEC: float = max(2.0, _float("SCALP_REVERSAL_NWAVE_EXIT_MIN_HOLD_SEC", 12.0))
EXIT_MAX_HOLD_SEC: float = max(EXIT_MIN_HOLD_SEC, _float("SCALP_REVERSAL_NWAVE_EXIT_MAX_HOLD_SEC", 180.0))
EXIT_PROFIT_PIPS: float = max(0.6, _float("SCALP_REVERSAL_NWAVE_EXIT_PROFIT_PIPS", 3.0))
EXIT_TRAIL_START_PIPS: float = max(0.6, _float("SCALP_REVERSAL_NWAVE_EXIT_TRAIL_START_PIPS", 3.8))
EXIT_TRAIL_BACKOFF_PIPS: float = max(0.2, _float("SCALP_REVERSAL_NWAVE_EXIT_TRAIL_BACKOFF_PIPS", 1.2))
EXIT_LOCK_BUFFER_PIPS: float = max(0.1, _float("SCALP_REVERSAL_NWAVE_EXIT_LOCK_BUFFER_PIPS", 0.9))
EXIT_LOCK_TRIGGER_PIPS: float = max(0.5, _float("SCALP_REVERSAL_NWAVE_EXIT_LOCK_TRIGGER_PIPS", 1.8))
EXIT_REVERSAL_WINDOW_SEC: float = max(1.0, _float("SCALP_REVERSAL_NWAVE_EXIT_REV_WINDOW_SEC", 6.0))
EXIT_REVERSAL_PIPS: float = max(0.5, _float("SCALP_REVERSAL_NWAVE_EXIT_REV_PIPS", 3.2))
EXIT_ATR_SPIKE_PIPS: float = max(1.0, _float("SCALP_REVERSAL_NWAVE_EXIT_ATR_SPIKE_PIPS", 6.0))
EXIT_HARD_STOP_PIPS: float = max(1.0, _float("SCALP_REVERSAL_NWAVE_EXIT_HARD_STOP_PIPS", 6.0))
EXIT_TP_HINT_RATIO: float = max(0.2, _float("SCALP_REVERSAL_NWAVE_EXIT_TP_HINT_RATIO", 0.75))

TECH_FAILOPEN: bool = _bool("SCALP_REVERSAL_NWAVE_TECH_FAILOPEN", False)
TECH_CONF_BOOST: float = max(0.0, _float("SCALP_REVERSAL_NWAVE_TECH_CONF_BOOST", 16.0))
TECH_CONF_PENALTY: float = max(0.0, _float("SCALP_REVERSAL_NWAVE_TECH_CONF_PENALTY", 10.0))
