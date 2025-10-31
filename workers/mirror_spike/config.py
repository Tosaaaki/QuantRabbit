"""
Configuration values for the mirror spike-reversal worker.
"""

from __future__ import annotations

import os

PIP_VALUE = 0.01


def _read_env(key: str, legacy_key: str | None) -> str | None:
    raw = os.getenv(key)
    if raw is None and legacy_key:
        raw = os.getenv(legacy_key)
    return raw


def _float_env(key: str, default: float, *, legacy_key: str | None = None) -> float:
    raw = _read_env(key, legacy_key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(key: str, default: int, *, legacy_key: str | None = None) -> int:
    raw = _read_env(key, legacy_key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


SPREAD_MAX_PIPS: float = max(
    0.1,
    _float_env(
        "MIRROR_SPIKE_MAX_SPREAD_PIPS",
        1.2,
        legacy_key="MANUAL_SPIKE_MAX_SPREAD_PIPS",
    ),
)
LOOKBACK_SEC: float = max(
    10.0,
    _float_env(
        "MIRROR_SPIKE_LOOKBACK_SEC",
        90.0,
        legacy_key="MANUAL_SPIKE_LOOKBACK_SEC",
    ),
)
PEAK_WINDOW_SEC: float = max(
    5.0,
    min(
        LOOKBACK_SEC,
        _float_env(
            "MIRROR_SPIKE_PEAK_WINDOW_SEC",
            30.0,
            legacy_key="MANUAL_SPIKE_PEAK_WINDOW_SEC",
        ),
    ),
)
SPIKE_THRESHOLD_PIPS: float = max(
    2.0,
    _float_env(
        "MIRROR_SPIKE_THRESHOLD_PIPS",
        4.5,
        legacy_key="MANUAL_SPIKE_THRESHOLD_PIPS",
    ),
)
RETRACE_TRIGGER_PIPS: float = max(
    0.3,
    _float_env(
        "MIRROR_SPIKE_RETRACE_PIPS",
        1.0,
        legacy_key="MANUAL_SPIKE_RETRACE_PIPS",
    ),
)
MIN_RETRACE_PIPS: float = max(
    0.2,
    _float_env(
        "MIRROR_SPIKE_MIN_RETRACE_PIPS",
        0.6,
        legacy_key="MANUAL_SPIKE_MIN_RETRACE_PIPS",
    ),
)
RSI_OVERBOUGHT: float = _float_env(
    "MIRROR_SPIKE_RSI_OVERBOUGHT",
    60.0,
    legacy_key="MANUAL_SPIKE_RSI_OVERBOUGHT",
)
RSI_OVERSOLD: float = _float_env(
    "MIRROR_SPIKE_RSI_OVERSOLD",
    40.0,
    legacy_key="MANUAL_SPIKE_RSI_OVERSOLD",
)
MIN_TICK_COUNT: int = max(
    20,
    _int_env(
        "MIRROR_SPIKE_MIN_TICK_COUNT",
        40,
        legacy_key="MANUAL_SPIKE_MIN_TICK_COUNT",
    ),
)
ENTRY_UNITS: int = max(
    1000,
    _int_env(
        "MIRROR_SPIKE_ENTRY_UNITS",
        5000,
        legacy_key="MANUAL_SPIKE_ENTRY_UNITS",
    ),
)
TP_PIPS: float = max(
    2.0,
    _float_env(
        "MIRROR_SPIKE_TP_PIPS",
        12.0,
        legacy_key="MANUAL_SPIKE_TP_PIPS",
    ),
)
COOLDOWN_SEC: float = max(
    30.0,
    _float_env(
        "MIRROR_SPIKE_COOLDOWN_SEC",
        150.0,
        legacy_key="MANUAL_SPIKE_COOLDOWN_SEC",
    ),
)
POST_EXIT_COOLDOWN_SEC: float = max(
    30.0,
    _float_env(
        "MIRROR_SPIKE_POST_EXIT_COOLDOWN_SEC",
        120.0,
        legacy_key="MANUAL_SPIKE_POST_EXIT_COOLDOWN_SEC",
    ),
)
MAX_ACTIVE_TRADES: int = max(
    1,
    _int_env(
        "MIRROR_SPIKE_MAX_ACTIVE_TRADES",
        2,
        legacy_key="MANUAL_SPIKE_MAX_ACTIVE_TRADES",
    ),
)
STAGE_MIN_DELTA_PIPS: float = max(
    0.2,
    _float_env(
        "MIRROR_SPIKE_STAGE_MIN_DELTA_PIPS",
        0.6,
        legacy_key="MANUAL_SPIKE_STAGE_MIN_DELTA_PIPS",
    ),
)
LOG_PREFIX = "[MIRROR-SPIKE]"
