"""
Configuration values for the mirror spike-reversal worker.
"""

from __future__ import annotations

import os
from typing import Set

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


def _bool_env(key: str, default: bool, *, legacy_key: str | None = None) -> bool:
    raw = _read_env(key, legacy_key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _parse_hours(key: str, default: str) -> Set[int]:
    raw = _read_env(key, None) or default
    hours: Set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", 1)
            try:
                start = int(float(bounds[0]))
                end = int(float(bounds[1]))
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            for hour in range(start, end + 1):
                if 0 <= hour <= 23:
                    hours.add(hour)
            continue
        try:
            hour_val = int(float(token))
        except ValueError:
            continue
        if 0 <= hour_val <= 23:
            hours.add(hour_val)
    return hours


SPREAD_MAX_PIPS: float = max(
    0.1,
    _float_env(
        "MIRROR_SPIKE_MAX_SPREAD_PIPS",
        0.70,
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
            24.0,
            legacy_key="MANUAL_SPIKE_PEAK_WINDOW_SEC",
        ),
    ),
)
SPIKE_THRESHOLD_PIPS: float = max(
    2.0,
    _float_env(
        "MIRROR_SPIKE_THRESHOLD_PIPS",
        3.8,
        legacy_key="MANUAL_SPIKE_THRESHOLD_PIPS",
    ),
)
RETRACE_TRIGGER_PIPS: float = max(
    0.3,
    _float_env(
        "MIRROR_SPIKE_RETRACE_PIPS",
        1.5,
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
    72.0,
    legacy_key="MANUAL_SPIKE_RSI_OVERBOUGHT",
)
RSI_OVERSOLD: float = _float_env(
    "MIRROR_SPIKE_RSI_OVERSOLD",
    28.0,
    legacy_key="MANUAL_SPIKE_RSI_OVERSOLD",
)
MIN_TICK_COUNT: int = max(
    20,
    _int_env(
        "MIRROR_SPIKE_MIN_TICK_COUNT",
        60,
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
        6.0,
        legacy_key="MANUAL_SPIKE_TP_PIPS",
    ),
)
SL_PIPS: float = max(
    0.5,
    _float_env(
        "MIRROR_SPIKE_SL_PIPS",
        2.6,
        legacy_key=None,
    ),
)
SHORT_SL_PIPS: float = max(
    0.5,
    _float_env(
        "MIRROR_SPIKE_SHORT_SL_PIPS",
        SL_PIPS,
        legacy_key=None,
    ),
)
MIN_ATR_PIPS: float = max(
    0.0,
    _float_env(
        "MIRROR_SPIKE_MIN_ATR_PIPS",
        0.5,
        legacy_key=None,
    ),
)
MIN_TICK_RATE: float = max(
    0.0,
    _float_env(
        "MIRROR_SPIKE_MIN_TICK_RATE",
        1.2,
        legacy_key=None,
    ),
)
COOLDOWN_SEC: float = max(
    30.0,
    _float_env(
        "MIRROR_SPIKE_COOLDOWN_SEC",
        480.0,
        legacy_key="MANUAL_SPIKE_COOLDOWN_SEC",
    ),
)
POST_EXIT_COOLDOWN_SEC: float = max(
    30.0,
    _float_env(
        "MIRROR_SPIKE_POST_EXIT_COOLDOWN_SEC",
        900.0,
        legacy_key="MANUAL_SPIKE_POST_EXIT_COOLDOWN_SEC",
    ),
)
SHORT_MIN_SPIKE_PIPS: float = max(
    SPIKE_THRESHOLD_PIPS,
    _float_env(
        "MIRROR_SPIKE_SHORT_MIN_SPIKE_PIPS",
        5.0,
        legacy_key=None,
    ),
)
SHORT_MAX_DOWNSLOPE_PIPS: float = max(
    0.0,
    _float_env(
        "MIRROR_SPIKE_SHORT_MAX_DOWNSLOPE_PIPS",
        2.0,
        legacy_key=None,
    ),
)
SHORT_TP_PIPS: float = max(
    1.0,
    _float_env(
        "MIRROR_SPIKE_SHORT_TP_PIPS",
        TP_PIPS,
        legacy_key=None,
    ),
)
LONG_MIN_SPIKE_PIPS: float = max(
    SPIKE_THRESHOLD_PIPS,
    _float_env(
        "MIRROR_SPIKE_LONG_MIN_SPIKE_PIPS",
        5.0,
        legacy_key=None,
    ),
)
LONG_MAX_UPSLOPE_PIPS: float = max(
    0.0,
    _float_env(
        "MIRROR_SPIKE_LONG_MAX_UPSLOPE_PIPS",
        2.0,
        legacy_key=None,
    ),
)
MAX_ACTIVE_TRADES: int = max(
    1,
    _int_env(
        "MIRROR_SPIKE_MAX_ACTIVE_TRADES",
        1,
        legacy_key="MANUAL_SPIKE_MAX_ACTIVE_TRADES",
    ),
)
STAGE_MIN_DELTA_PIPS: float = max(
    0.2,
    _float_env(
        "MIRROR_SPIKE_STAGE_MIN_DELTA_PIPS",
        2.0,
        legacy_key="MANUAL_SPIKE_STAGE_MIN_DELTA_PIPS",
    ),
)
MAX_HOLD_SEC: float = max(
    30.0,
    _float_env(
        "MIRROR_SPIKE_MAX_HOLD_SEC",
        75.0,
        legacy_key=None,
    ),
)
ENABLED: bool = _bool_env(
    "MIRROR_SPIKE_ENABLED",
    True,
    legacy_key="MANUAL_SPIKE_ENABLED",
)
ACTIVE_HOURS_UTC = frozenset(range(24))
LOG_PREFIX = "[MIRROR-SPIKE]"
