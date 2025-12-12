"""Configuration for the London Momentum session worker."""

from __future__ import annotations

import os


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


LOG_PREFIX = "[LONDON-MOMO]"
ENABLED = _bool("LONDON_MOMENTUM_ENABLED", True)
POCKET = os.getenv("LONDON_MOMENTUM_POCKET", "macro")
LOOP_INTERVAL_SEC = max(0.5, _float("LONDON_MOMENTUM_LOOP_SEC", 1.5))
RISK_PCT = max(0.001, _float("LONDON_MOMENTUM_RISK_PCT", 0.015))
MIN_UNITS = int(max(1000.0, _float("LONDON_MOMENTUM_MIN_UNITS", 5000)))
MAX_SPREAD_PIPS = _float("LONDON_MOMENTUM_MAX_SPREAD", 1.4)
MIN_ATR_PIPS = _float("LONDON_MOMENTUM_MIN_ATR", 3.5)
TREND_GAP_MIN = _float("LONDON_MOMENTUM_TREND_GAP", 0.05)
MOMENTUM_MIN = _float("LONDON_MOMENTUM_MOMENTUM_MIN", 0.012)
TP_PIPS = _float("LONDON_MOMENTUM_TP_PIPS", 7.0)
SL_PIPS = _float("LONDON_MOMENTUM_SL_PIPS", 4.5)
COOLDOWN_SEC = int(max(30.0, _float("LONDON_MOMENTUM_COOLDOWN_SEC", 120.0)))
SESSION_START_UTC = os.getenv("LONDON_MOMENTUM_START_UTC", "06:45")
SESSION_END_UTC = os.getenv("LONDON_MOMENTUM_END_UTC", "11:30")
