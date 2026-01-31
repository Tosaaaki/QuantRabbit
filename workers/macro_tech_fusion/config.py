"""Configuration for Macro Tech Fusion worker."""

from __future__ import annotations

import os

PIP_VALUE = 0.01
LOG_PREFIX = "[MACRO-TECH-FUSION]"
STRATEGY_TAG = "MacroTechFusion"
PROFILE_TAG = "macro_tech_fusion"


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


ENABLED: bool = _bool("MACRO_TECH_FUSION_ENABLED", False)
LOOP_INTERVAL_SEC: float = max(1.0, _float("MACRO_TECH_FUSION_LOOP_INTERVAL_SEC", 12.0))
POCKET: str = "macro"
MODE: str = "trend"

BASE_ENTRY_UNITS: int = max(1000, _int("MACRO_TECH_FUSION_BASE_ENTRY_UNITS", 40000))
MIN_UNITS: int = max(1000, _int("MACRO_TECH_FUSION_MIN_UNITS", 2000))
CONFIDENCE_FLOOR: int = max(0, _int("MACRO_TECH_FUSION_CONFIDENCE_FLOOR", 45))
CONFIDENCE_CEIL: int = max(CONFIDENCE_FLOOR + 1, _int("MACRO_TECH_FUSION_CONFIDENCE_CEIL", 92))

CAP_MIN: float = max(0.05, _float("MACRO_TECH_FUSION_CAP_MIN", 0.2))
CAP_MAX: float = max(CAP_MIN, _float("MACRO_TECH_FUSION_CAP_MAX", 0.95))
MAX_FACTOR_AGE_SEC: float = max(30.0, _float("MACRO_TECH_FUSION_MAX_FACTOR_AGE_SEC", 180.0))

TREND_ADX_MIN: float = max(0.0, _float("MACRO_TECH_FUSION_TREND_ADX_MIN", 20.0))
TREND_SCORE_MIN: float = max(0.0, _float("MACRO_TECH_FUSION_TREND_SCORE_MIN", 0.55))
RANGE_BLOCK: bool = _bool("MACRO_TECH_FUSION_RANGE_BLOCK", True)

SL_MIN_PIPS: float = max(1.0, _float("MACRO_TECH_FUSION_SL_MIN_PIPS", 20.0))
SL_MAX_PIPS: float = max(SL_MIN_PIPS, _float("MACRO_TECH_FUSION_SL_MAX_PIPS", 60.0))
SL_ATR_MULT: float = max(0.2, _float("MACRO_TECH_FUSION_SL_ATR_MULT", 2.8))
TP_MIN_PIPS: float = max(1.0, _float("MACRO_TECH_FUSION_TP_MIN_PIPS", 40.0))
TP_MAX_PIPS: float = max(TP_MIN_PIPS, _float("MACRO_TECH_FUSION_TP_MAX_PIPS", 120.0))
TP_RR: float = max(0.6, _float("MACRO_TECH_FUSION_TP_RR", 1.6))
SPREAD_FLOOR_PIPS: float = max(0.0, _float("MACRO_TECH_FUSION_SPREAD_FLOOR_PIPS", 16.0))

EXIT_LOOP_INTERVAL_SEC: float = max(0.4, _float("MACRO_TECH_FUSION_EXIT_LOOP_INTERVAL_SEC", 1.2))
EXIT_MIN_HOLD_SEC: float = max(30.0, _float("MACRO_TECH_FUSION_EXIT_MIN_HOLD_SEC", 300.0))
EXIT_MAX_HOLD_SEC: float = max(EXIT_MIN_HOLD_SEC, _float("MACRO_TECH_FUSION_EXIT_MAX_HOLD_SEC", 21600.0))
EXIT_PROFIT_PIPS: float = max(1.0, _float("MACRO_TECH_FUSION_EXIT_PROFIT_PIPS", 20.0))
EXIT_TRAIL_START_PIPS: float = max(1.0, _float("MACRO_TECH_FUSION_EXIT_TRAIL_START_PIPS", 28.0))
EXIT_TRAIL_BACKOFF_PIPS: float = max(0.2, _float("MACRO_TECH_FUSION_EXIT_TRAIL_BACKOFF_PIPS", 10.0))
EXIT_LOCK_BUFFER_PIPS: float = max(0.2, _float("MACRO_TECH_FUSION_EXIT_LOCK_BUFFER_PIPS", 8.0))
EXIT_LOCK_TRIGGER_PIPS: float = max(0.5, _float("MACRO_TECH_FUSION_EXIT_LOCK_TRIGGER_PIPS", 14.0))
EXIT_REVERSAL_WINDOW_SEC: float = max(5.0, _float("MACRO_TECH_FUSION_EXIT_REV_WINDOW_SEC", 120.0))
EXIT_REVERSAL_PIPS: float = max(1.0, _float("MACRO_TECH_FUSION_EXIT_REV_PIPS", 18.0))
EXIT_ATR_SPIKE_PIPS: float = max(1.0, _float("MACRO_TECH_FUSION_EXIT_ATR_SPIKE_PIPS", 28.0))
EXIT_HARD_STOP_PIPS: float = max(1.0, _float("MACRO_TECH_FUSION_EXIT_HARD_STOP_PIPS", 24.0))
EXIT_TP_HINT_RATIO: float = max(0.2, _float("MACRO_TECH_FUSION_EXIT_TP_HINT_RATIO", 0.75))

TECH_FAILOPEN: bool = _bool("MACRO_TECH_FUSION_TECH_FAILOPEN", False)
TECH_CONF_BOOST: float = max(0.0, _float("MACRO_TECH_FUSION_TECH_CONF_BOOST", 16.0))
TECH_CONF_PENALTY: float = max(0.0, _float("MACRO_TECH_FUSION_TECH_CONF_PENALTY", 10.0))
