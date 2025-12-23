"""Configuration for the micro exit worker."""

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


LOG_PREFIX = "[MICRO-EXIT]"
ENABLED: bool = _bool("MICRO_EXIT_ENABLED", False)
LOOP_INTERVAL_SEC: float = max(0.5, _float("MICRO_EXIT_LOOP_INTERVAL_SEC", 1.0))
CONFIDENCE_BIAS: float = max(0.0, min(1.0, _float("MICRO_EXIT_CONFIDENCE_BIAS", 0.75)))
CONFIDENCE_NEUTRAL: float = max(0.0, min(1.0, _float("MICRO_EXIT_CONFIDENCE_NEUTRAL", 0.6)))
RSI_LONG: float = _float("MICRO_EXIT_RSI_LONG", 55.0)
RSI_SHORT: float = _float("MICRO_EXIT_RSI_SHORT", 45.0)
MA_GAP_MIN_PIPS: float = max(0.0, _float("MICRO_EXIT_MA_GAP_MIN_PIPS", 0.35))
