"""Configuration for the manual swing replication worker."""

from __future__ import annotations

import os
from typing import List, Set

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


def _parse_hours(key: str, default: str) -> Set[int]:
    raw = os.getenv(key, default)
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


def _parse_int_list(key: str, default: str) -> List[int]:
    raw = os.getenv(key, default)
    parts = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(int(float(token)))
        except ValueError:
            continue
    return [value for value in parts if value != 0]


LOG_PREFIX = "[MANUAL-SWING]"
ENABLED: bool = _bool("MANUAL_SWING_ENABLED", True)

LOOP_INTERVAL_SEC: float = max(5.0, _float("MANUAL_SWING_LOOP_INTERVAL_SEC", 15.0))

POCKET: str = os.getenv("MANUAL_SWING_POCKET", "macro")
ALLOW_LONG: bool = _bool("MANUAL_SWING_ALLOW_LONG", True)
ALLOW_SHORT: bool = _bool("MANUAL_SWING_ALLOW_SHORT", True)

ALLOWED_HOURS_UTC = frozenset(
    _parse_hours("MANUAL_SWING_ALLOWED_HOURS", "0-23")
)
BLOCKED_WEEKDAYS = tuple(
    day.strip()
    for day in os.getenv("MANUAL_SWING_BLOCKED_WEEKDAYS", "").split(",")
    if day.strip()
)

NEWS_BLOCK_MINUTES: float = max(0.0, _float("MANUAL_SWING_NEWS_BLOCK_MINUTES", 30.0))
NEWS_BLOCK_MIN_IMPACT: int = max(1, _int("MANUAL_SWING_NEWS_IMPACT", 2))

LOSS_STREAK_MAX: int = max(0, _int("MANUAL_SWING_MAX_CONSEC_LOSSES", 2))
LOSS_STREAK_COOLDOWN_MIN: float = max(
    0.0, _float("MANUAL_SWING_LOSS_COOLDOWN_MINUTES", 60.0)
)

REFERENCE_EQUITY: float = max(1.0, _float("MANUAL_SWING_REFERENCE_EQUITY", 500000.0))
MAX_ACTIVE_STAGES: int = max(1, _int("MANUAL_SWING_MAX_STAGES", 4))
STAGE_UNITS_BASE = _parse_int_list(
    "MANUAL_SWING_STAGE_UNITS_BASE",
    "5000,5000,5000,5000",
)
if not STAGE_UNITS_BASE:
    STAGE_UNITS_BASE = [12000]
MIN_STAGE_UNITS: int = max(1000, _int("MANUAL_SWING_MIN_STAGE_UNITS", 4000))
if len(STAGE_UNITS_BASE) < MAX_ACTIVE_STAGES:
    last_val = STAGE_UNITS_BASE[-1]
    while len(STAGE_UNITS_BASE) < MAX_ACTIVE_STAGES:
        STAGE_UNITS_BASE.append(last_val)
STAGE_ADD_TRIGGER_PIPS: float = max(
    5.0, _float("MANUAL_SWING_STAGE_ADD_TRIGGER_PIPS", 18.0)
)
STAGE_COOLDOWN_MINUTES: float = max(
    0.0, _float("MANUAL_SWING_STAGE_COOLDOWN_MINUTES", 120.0)
)

MAX_HOLD_HOURS: float = max(1.0, _float("MANUAL_SWING_MAX_HOLD_HOURS", 96.0))
TARGET_HOLD_HOURS: float = max(1.0, _float("MANUAL_SWING_TARGET_HOLD_HOURS", 36.0))

MIN_FREE_MARGIN_RATIO: float = _float("MANUAL_SWING_MIN_FREE_MARGIN", 0.18)
MARGIN_HEALTH_EXIT: float = _float("MANUAL_SWING_EXIT_HEALTH_BUFFER", 0.05)

RISK_PCT_OVERRIDE: float = _float("MANUAL_SWING_RISK_PCT", 0.035)
RISK_FREE_MARGIN_FRACTION: float = max(
    0.01, _float("MANUAL_SWING_RISK_FREE_MARGIN_FRACTION", 0.35)
)

SL_ATR_MULT: float = max(0.2, _float("MANUAL_SWING_SL_ATR_MULT", 1.1))
TP_ATR_MULT: float = max(0.2, _float("MANUAL_SWING_TP_ATR_MULT", 2.0))
MIN_SL_PIPS: float = max(5.0, _float("MANUAL_SWING_MIN_SL_PIPS", 20.0))
MIN_TP_PIPS: float = max(10.0, _float("MANUAL_SWING_MIN_TP_PIPS", 120.0))

PROFIT_TRIGGER_PIPS: float = max(
    40.0, _float("MANUAL_SWING_PROFIT_TRIGGER_PIPS", 110.0)
)
TRAIL_TRIGGER_PIPS: float = max(
    30.0, _float("MANUAL_SWING_TRAIL_TRIGGER", 80.0)
)
TRAIL_BACKOFF_PIPS: float = max(5.0, _float("MANUAL_SWING_TRAIL_BACKOFF", 25.0))

H4_GAP_MIN: float = _float("MANUAL_SWING_H4_GAP_MIN", 0.0015)
H1_GAP_MIN: float = _float("MANUAL_SWING_H1_GAP_MIN", 0.0006)
ADX_MIN: float = _float("MANUAL_SWING_ADX_MIN", 22.0)
ATR_MIN_PIPS: float = _float("MANUAL_SWING_ATR_MIN_PIPS", 20.0)

REVERSAL_GAP_EXIT: float = _float("MANUAL_SWING_REVERSAL_GAP_EXIT", 0.0015)
MAX_DRAWDOWN_PIPS: float = max(
    30.0, _float("MANUAL_SWING_MAX_DRAWDOWN_PIPS", 60.0)
)

SPREAD_MAX_PIPS: float = max(0.3, _float("MANUAL_SWING_SPREAD_MAX_PIPS", 1.2))
SPREAD_RECOVERY_PIPS: float = max(
    0.1, _float("MANUAL_SWING_SPREAD_RECOVERY_PIPS", SPREAD_MAX_PIPS - 0.1)
)

PERF_SYNC_INTERVAL_SEC: float = max(
    120.0, _float("MANUAL_SWING_PERF_SYNC_INTERVAL_SEC", 900.0)
)

LOG_SKIP_REASON: bool = _bool("MANUAL_SWING_LOG_SKIP_REASON", True)
