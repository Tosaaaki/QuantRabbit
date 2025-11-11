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
ENABLED = _bool("LONDON_MOMENTUM_ENABLED", False)
POCKET = os.getenv("LONDON_MOMENTUM_POCKET", "macro")
LOOP_INTERVAL_SEC = max(0.5, _float("LONDON_MOMENTUM_LOOP_SEC", 1.5))
RISK_PCT = max(0.001, _float("LONDON_MOMENTUM_RISK_PCT", 0.015))
MIN_UNITS = int(max(1000.0, _float("LONDON_MOMENTUM_MIN_UNITS", 5000)))
MAX_SPREAD_PIPS = _float("LONDON_MOMENTUM_MAX_SPREAD", 1.4)
MIN_ATR_PIPS = _float("LONDON_MOMENTUM_MIN_ATR", 5.0)
TREND_GAP_MIN = _float("LONDON_MOMENTUM_TREND_GAP", 0.08)
MOMENTUM_MIN = _float("LONDON_MOMENTUM_MOMENTUM_MIN", 0.02)
TP_PIPS = _float("LONDON_MOMENTUM_TP_PIPS", 8.0)
SL_PIPS = _float("LONDON_MOMENTUM_SL_PIPS", 5.0)
COOLDOWN_SEC = int(max(30.0, _float("LONDON_MOMENTUM_COOLDOWN_SEC", 150.0)))
SESSION_START_UTC = os.getenv("LONDON_MOMENTUM_START_UTC", "06:45")
SESSION_END_UTC = os.getenv("LONDON_MOMENTUM_END_UTC", "11:30")
