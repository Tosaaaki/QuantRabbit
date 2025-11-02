"""Configuration for the S5-based mirror spike worker."""

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


LOG_PREFIX = "[MIRROR-S5]"
ENABLED: bool = _bool("MIRROR_SPIKE_S5_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("MIRROR_SPIKE_S5_LOOP_INTERVAL_SEC", 0.45))

MAX_SPREAD_PIPS: float = max(0.1, _float("MIRROR_SPIKE_S5_MAX_SPREAD_PIPS", 0.9))

WINDOW_SEC: float = max(60.0, _float("MIRROR_SPIKE_S5_WINDOW_SEC", 360.0))
BUCKET_SECONDS: float = max(1.0, _float("MIRROR_SPIKE_S5_BUCKET_SECONDS", 5.0))
MIN_BUCKETS: int = max(40, _int("MIRROR_SPIKE_S5_MIN_BUCKETS", 60))
LOOKBACK_BUCKETS: int = max(MIN_BUCKETS, _int("MIRROR_SPIKE_S5_LOOKBACK_BUCKETS", 72))
PEAK_WINDOW_BUCKETS: int = max(6, _int("MIRROR_SPIKE_S5_PEAK_WINDOW_BUCKETS", 18))

SPIKE_THRESHOLD_PIPS: float = max(2.0, _float("MIRROR_SPIKE_S5_THRESHOLD_PIPS", 4.0))
RETRACE_TRIGGER_PIPS: float = max(0.3, _float("MIRROR_SPIKE_S5_RETRACE_PIPS", 1.1))
MIN_RETRACE_PIPS: float = max(0.2, _float("MIRROR_SPIKE_S5_MIN_RETRACE_PIPS", 0.6))

RSI_PERIOD: int = max(5, _int("MIRROR_SPIKE_S5_RSI_PERIOD", 14))
RSI_OVERBOUGHT: float = _float("MIRROR_SPIKE_S5_RSI_OVERBOUGHT", 72.0)
RSI_OVERSOLD: float = _float("MIRROR_SPIKE_S5_RSI_OVERSOLD", 28.0)

ENTRY_UNITS: int = max(1000, _int("MIRROR_SPIKE_S5_ENTRY_UNITS", 5000))
MAX_ACTIVE_TRADES: int = max(1, _int("MIRROR_SPIKE_S5_MAX_ACTIVE_TRADES", 1))
STAGE_MIN_DELTA_PIPS: float = max(0.1, _float("MIRROR_SPIKE_S5_STAGE_MIN_DELTA_PIPS", 1.8))

TP_PIPS: float = max(1.0, _float("MIRROR_SPIKE_S5_TP_PIPS", 5.5))
SL_PIPS: float = max(0.5, _float("MIRROR_SPIKE_S5_SL_PIPS", 2.8))

MIN_ATR_PIPS: float = max(0.0, _float("MIRROR_SPIKE_S5_MIN_ATR_PIPS", 0.5))

COOLDOWN_SEC: float = max(30.0, _float("MIRROR_SPIKE_S5_COOLDOWN_SEC", 360.0))
POST_EXIT_COOLDOWN_SEC: float = max(30.0, _float("MIRROR_SPIKE_S5_POST_EXIT_COOLDOWN_SEC", 900.0))
