"""
Configuration values for the manual spike-reversal worker.
"""

from __future__ import annotations

import os

PIP_VALUE = 0.01


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


SPREAD_MAX_PIPS: float = max(0.1, _float_env("MANUAL_SPIKE_MAX_SPREAD_PIPS", 1.2))
LOOKBACK_SEC: float = max(10.0, _float_env("MANUAL_SPIKE_LOOKBACK_SEC", 90.0))
PEAK_WINDOW_SEC: float = max(5.0, min(LOOKBACK_SEC, _float_env("MANUAL_SPIKE_PEAK_WINDOW_SEC", 30.0)))
SPIKE_THRESHOLD_PIPS: float = max(2.0, _float_env("MANUAL_SPIKE_THRESHOLD_PIPS", 7.0))
RETRACE_TRIGGER_PIPS: float = max(0.3, _float_env("MANUAL_SPIKE_RETRACE_PIPS", 1.6))
MIN_RETRACE_PIPS: float = max(0.2, _float_env("MANUAL_SPIKE_MIN_RETRACE_PIPS", 0.8))
RSI_OVERBOUGHT: float = _float_env("MANUAL_SPIKE_RSI_OVERBOUGHT", 64.0)
RSI_OVERSOLD: float = _float_env("MANUAL_SPIKE_RSI_OVERSOLD", 36.0)
MIN_TICK_COUNT: int = max(20, _int_env("MANUAL_SPIKE_MIN_TICK_COUNT", 50))
ENTRY_UNITS: int = max(1000, _int_env("MANUAL_SPIKE_ENTRY_UNITS", 5000))
TP_PIPS: float = max(2.0, _float_env("MANUAL_SPIKE_TP_PIPS", 15.0))
COOLDOWN_SEC: float = max(30.0, _float_env("MANUAL_SPIKE_COOLDOWN_SEC", 240.0))
POST_EXIT_COOLDOWN_SEC: float = max(30.0, _float_env("MANUAL_SPIKE_POST_EXIT_COOLDOWN_SEC", 180.0))
MAX_ACTIVE_TRADES: int = max(1, _int_env("MANUAL_SPIKE_MAX_ACTIVE_TRADES", 2))
STAGE_MIN_DELTA_PIPS: float = max(0.2, _float_env("MANUAL_SPIKE_STAGE_MIN_DELTA_PIPS", 1.0))
LOG_PREFIX = "[MANUAL-SPIKE]"
