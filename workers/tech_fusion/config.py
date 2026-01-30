"""Configuration for Tech Fusion worker."""

from __future__ import annotations

import os

PIP_VALUE = 0.01
LOG_PREFIX = "[TECH-FUSION]"
STRATEGY_TAG = "TechFusion"
PROFILE_TAG = "tech_fusion"


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


def _str(key: str, default: str) -> str:
    raw = os.getenv(key)
    if raw is None:
        return str(default)
    return str(raw).strip() or str(default)


ENABLED: bool = _bool("TECH_FUSION_ENABLED", False)
LOOP_INTERVAL_SEC: float = max(0.2, _float("TECH_FUSION_LOOP_INTERVAL_SEC", 8.0))
POCKET: str = _str("TECH_FUSION_POCKET", "micro")
MODE: str = _str("TECH_FUSION_MODE", "both")  # trend/range/both

BASE_ENTRY_UNITS: int = max(500, _int("TECH_FUSION_BASE_ENTRY_UNITS", 20000))
MIN_UNITS: int = max(500, _int("TECH_FUSION_MIN_UNITS", 1500))
CONFIDENCE_FLOOR: int = max(0, _int("TECH_FUSION_CONFIDENCE_FLOOR", 40))
CONFIDENCE_CEIL: int = max(CONFIDENCE_FLOOR + 1, _int("TECH_FUSION_CONFIDENCE_CEIL", 92))

CAP_MIN: float = max(0.05, _float("TECH_FUSION_CAP_MIN", 0.15))
CAP_MAX: float = max(CAP_MIN, _float("TECH_FUSION_CAP_MAX", 0.95))
MAX_FACTOR_AGE_SEC: float = max(10.0, _float("TECH_FUSION_MAX_FACTOR_AGE_SEC", 90.0))
RANGE_ONLY_SCORE: float = max(0.0, _float("TECH_FUSION_RANGE_ONLY_SCORE", 0.45))

TECH_FAILOPEN: bool = _bool("TECH_FUSION_TECH_FAILOPEN", False)
TECH_CONF_BOOST: float = max(0.0, _float("TECH_FUSION_TECH_CONF_BOOST", 18.0))
TECH_CONF_PENALTY: float = max(0.0, _float("TECH_FUSION_TECH_CONF_PENALTY", 10.0))

TREND_SCORE_MIN: float = max(0.0, _float("TECH_FUSION_TREND_SCORE_MIN", 0.6))
RANGE_SCORE_MIN: float = max(0.0, _float("TECH_FUSION_RANGE_SCORE_MIN", 0.7))
TREND_ADX_MIN: float = max(0.0, _float("TECH_FUSION_TREND_ADX_MIN", 18.0))
RANGE_ADX_MAX: float = max(0.0, _float("TECH_FUSION_RANGE_ADX_MAX", 18.0))
RANGE_RSI_LONG: float = max(1.0, _float("TECH_FUSION_RANGE_RSI_LONG", 35.0))
RANGE_RSI_SHORT: float = max(RANGE_RSI_LONG + 1.0, _float("TECH_FUSION_RANGE_RSI_SHORT", 65.0))
RANGE_BB_RATIO: float = max(0.05, _float("TECH_FUSION_RANGE_BB_RATIO", 0.18))
RANGE_BB_MIN_PIPS: float = max(0.3, _float("TECH_FUSION_RANGE_BB_MIN_PIPS", 1.2))
RANGE_BBW_MAX: float = max(0.0001, _float("TECH_FUSION_RANGE_BBW_MAX", 0.0018))

SL_MIN_TREND: float = max(1.0, _float("TECH_FUSION_SL_MIN_TREND", 8.0))
SL_MAX_TREND: float = max(SL_MIN_TREND, _float("TECH_FUSION_SL_MAX_TREND", 35.0))
SL_ATR_MULT_TREND: float = max(0.2, _float("TECH_FUSION_SL_ATR_MULT_TREND", 2.0))
TP_MIN_TREND: float = max(1.0, _float("TECH_FUSION_TP_MIN_TREND", 12.0))
TP_MAX_TREND: float = max(TP_MIN_TREND, _float("TECH_FUSION_TP_MAX_TREND", 60.0))
TP_RR_TREND: float = max(0.6, _float("TECH_FUSION_TP_RR_TREND", 1.4))

SL_MIN_RANGE: float = max(1.0, _float("TECH_FUSION_SL_MIN_RANGE", 4.0))
SL_MAX_RANGE: float = max(SL_MIN_RANGE, _float("TECH_FUSION_SL_MAX_RANGE", 18.0))
SL_ATR_MULT_RANGE: float = max(0.2, _float("TECH_FUSION_SL_ATR_MULT_RANGE", 1.3))
TP_MIN_RANGE: float = max(1.0, _float("TECH_FUSION_TP_MIN_RANGE", 6.0))
TP_MAX_RANGE: float = max(TP_MIN_RANGE, _float("TECH_FUSION_TP_MAX_RANGE", 24.0))
TP_RR_RANGE: float = max(0.5, _float("TECH_FUSION_TP_RR_RANGE", 1.0))

SPREAD_FLOOR_PIPS: float = max(0.0, _float("TECH_FUSION_SPREAD_FLOOR_PIPS", 12.0))

EXIT_LOOP_INTERVAL_SEC: float = max(0.2, _float("TECH_FUSION_EXIT_LOOP_INTERVAL_SEC", 0.8))
EXIT_MIN_HOLD_SEC: float = max(5.0, _float("TECH_FUSION_EXIT_MIN_HOLD_SEC", 90.0))
EXIT_MAX_HOLD_SEC: float = max(EXIT_MIN_HOLD_SEC, _float("TECH_FUSION_EXIT_MAX_HOLD_SEC", 1800.0))
EXIT_PROFIT_PIPS: float = max(0.8, _float("TECH_FUSION_EXIT_PROFIT_PIPS", 10.0))
EXIT_TRAIL_START_PIPS: float = max(1.0, _float("TECH_FUSION_EXIT_TRAIL_START_PIPS", 12.0))
EXIT_TRAIL_BACKOFF_PIPS: float = max(0.2, _float("TECH_FUSION_EXIT_TRAIL_BACKOFF_PIPS", 4.0))
EXIT_LOCK_BUFFER_PIPS: float = max(0.1, _float("TECH_FUSION_EXIT_LOCK_BUFFER_PIPS", 3.0))
EXIT_LOCK_TRIGGER_PIPS: float = max(0.5, _float("TECH_FUSION_EXIT_LOCK_TRIGGER_PIPS", 6.0))
EXIT_REVERSAL_WINDOW_SEC: float = max(2.0, _float("TECH_FUSION_EXIT_REV_WINDOW_SEC", 30.0))
EXIT_REVERSAL_PIPS: float = max(0.5, _float("TECH_FUSION_EXIT_REV_PIPS", 8.0))
EXIT_ATR_SPIKE_PIPS: float = max(1.0, _float("TECH_FUSION_EXIT_ATR_SPIKE_PIPS", 12.0))
EXIT_HARD_STOP_PIPS: float = max(1.0, _float("TECH_FUSION_EXIT_HARD_STOP_PIPS", 14.0))
EXIT_TP_HINT_RATIO: float = max(0.2, _float("TECH_FUSION_EXIT_TP_HINT_RATIO", 0.75))


