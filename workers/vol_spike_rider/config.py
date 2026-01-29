"""Configuration for Volatility Spike Rider worker."""

from __future__ import annotations

import os
from typing import Dict

PIP_VALUE = 0.01
LOG_PREFIX = "[VOL-SPIKE]"
STRATEGY_TAG = "VolSpikeRider"
PROFILE_TAG = "vol_spike_rider"


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


def _parse_map(key: str) -> Dict[str, float]:
    raw = os.getenv(key, "")
    mapping: Dict[str, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        name, value = token.split(":", 1)
        name = name.strip()
        if not name:
            continue
        try:
            mapping[name] = float(value)
        except (TypeError, ValueError):
            continue
    return mapping


ENABLED: bool = _bool("VOL_SPIKE_RIDER_ENABLED", False)
LOOP_INTERVAL_SEC: float = max(0.05, _float("VOL_SPIKE_RIDER_LOOP_INTERVAL_SEC", 0.25))

ENTRY_WINDOW_SEC: float = max(1.0, _float("VOL_SPIKE_RIDER_WINDOW_SEC", 6.0))
ENTRY_MIN_MOVE_PIPS: float = max(0.5, _float("VOL_SPIKE_RIDER_MIN_MOVE_PIPS", 4.0))
ENTRY_MIN_SPEED_PPS: float = max(0.1, _float("VOL_SPIKE_RIDER_MIN_SPEED_PPS", 0.7))
ENTRY_MIN_TICKS: int = max(6, _int("VOL_SPIKE_RIDER_MIN_TICKS", 18))
ENTRY_MAX_TICKS: int = max(20, _int("VOL_SPIKE_RIDER_MAX_TICKS", 240))
ENTRY_MAX_SPREAD_PIPS: float = max(0.1, _float("VOL_SPIKE_RIDER_MAX_SPREAD_PIPS", 0.9))
ENTRY_MIN_ATR_PIPS: float = max(0.0, _float("VOL_SPIKE_RIDER_MIN_ATR_PIPS", 2.0))
ENTRY_SIZE_MIN_MULT: float = max(0.5, _float("VOL_SPIKE_RIDER_SIZE_MIN_MULT", 0.85))
ENTRY_SIZE_MAX_MULT: float = max(1.0, _float("VOL_SPIKE_RIDER_SIZE_MAX_MULT", 1.6))

COOLDOWN_SEC: float = max(2.0, _float("VOL_SPIKE_RIDER_COOLDOWN_SEC", 20.0))
POST_EXIT_COOLDOWN_SEC: float = max(0.0, _float("VOL_SPIKE_RIDER_POST_EXIT_COOLDOWN_SEC", 15.0))
MAX_ACTIVE_TRADES: int = max(1, _int("VOL_SPIKE_RIDER_MAX_ACTIVE_TRADES", 1))

MIN_UNITS: int = max(500, _int("VOL_SPIKE_RIDER_MIN_UNITS", 1000))
MAX_LOT: float = max(0.05, _float("VOL_SPIKE_RIDER_MAX_LOT", 6.0))
RISK_PCT_OVERRIDE: float = max(0.0, _float("VOL_SPIKE_RIDER_RISK_PCT", 0.0))

SL_MIN_PIPS: float = max(0.5, _float("VOL_SPIKE_RIDER_SL_MIN_PIPS", 2.0))
SL_MAX_PIPS: float = max(SL_MIN_PIPS, _float("VOL_SPIKE_RIDER_SL_MAX_PIPS", 10.0))
SL_ATR_MULT: float = max(0.1, _float("VOL_SPIKE_RIDER_SL_ATR_MULT", 0.9))
SL_MOVE_MULT: float = max(0.05, _float("VOL_SPIKE_RIDER_SL_MOVE_MULT", 0.35))

TP_MIN_PIPS: float = max(0.6, _float("VOL_SPIKE_RIDER_TP_MIN_PIPS", 3.0))
TP_MAX_PIPS: float = max(TP_MIN_PIPS, _float("VOL_SPIKE_RIDER_TP_MAX_PIPS", 16.0))
TP_RR: float = max(1.0, _float("VOL_SPIKE_RIDER_TP_RR", 1.6))

EXIT_LOOP_INTERVAL_SEC: float = max(0.2, _float("VOL_SPIKE_RIDER_EXIT_LOOP_INTERVAL_SEC", 0.6))
EXIT_MIN_HOLD_SEC: float = max(3.0, _float("VOL_SPIKE_RIDER_EXIT_MIN_HOLD_SEC", 12.0))
EXIT_MAX_HOLD_SEC: float = max(20.0, _float("VOL_SPIKE_RIDER_EXIT_MAX_HOLD_SEC", 180.0))
EXIT_PROFIT_PIPS: float = max(0.8, _float("VOL_SPIKE_RIDER_EXIT_PROFIT_PIPS", 3.2))
EXIT_TRAIL_START_PIPS: float = max(1.0, _float("VOL_SPIKE_RIDER_EXIT_TRAIL_START_PIPS", 4.0))
EXIT_TRAIL_BACKOFF_PIPS: float = max(0.2, _float("VOL_SPIKE_RIDER_EXIT_TRAIL_BACKOFF_PIPS", 1.0))
EXIT_LOCK_BUFFER_PIPS: float = max(0.1, _float("VOL_SPIKE_RIDER_EXIT_LOCK_BUFFER_PIPS", 0.8))
EXIT_LOCK_TRIGGER_PIPS: float = max(0.5, _float("VOL_SPIKE_RIDER_EXIT_LOCK_TRIGGER_PIPS", 1.6))

EXIT_REVERSAL_WINDOW_SEC: float = max(1.0, _float("VOL_SPIKE_RIDER_EXIT_REV_WINDOW_SEC", 4.0))
EXIT_REVERSAL_PIPS: float = max(0.5, _float("VOL_SPIKE_RIDER_EXIT_REV_PIPS", 3.5))
EXIT_ATR_SPIKE_PIPS: float = max(1.0, _float("VOL_SPIKE_RIDER_EXIT_ATR_SPIKE_PIPS", 6.0))
EXIT_HARD_STOP_PIPS: float = max(1.0, _float("VOL_SPIKE_RIDER_EXIT_HARD_STOP_PIPS", 6.5))
EXIT_TP_HINT_RATIO: float = max(0.2, _float("VOL_SPIKE_RIDER_EXIT_TP_HINT_RATIO", 0.75))

EXIT_ATR_BY_TAG = _parse_map("VOL_SPIKE_RIDER_EXIT_ATR_BY_TAG")
EXIT_REV_BY_TAG = _parse_map("VOL_SPIKE_RIDER_EXIT_REV_BY_TAG")
