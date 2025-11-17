"""Configuration for the micro core worker."""

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


LOG_PREFIX = "[MICRO-CORE]"
ENABLED: bool = _bool("MICRO_CORE_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.5, _float("MICRO_CORE_LOOP_INTERVAL_SEC", 1.0))
PLAN_TIMEOUT_SEC: float = max(5.0, _float("MICRO_CORE_PLAN_TIMEOUT_SEC", 90.0))
COOLDOWN_BASE_SEC: int = int(max(10.0, _float("MICRO_CORE_COOLDOWN_SEC", 120.0)))
COOLDOWN_ATR_REF_PIPS: float = max(0.1, _float("MICRO_CORE_COOLDOWN_ATR_REF_PIPS", 6.0))
COOLDOWN_ATR_MIN_FACTOR: float = max(0.1, _float("MICRO_CORE_COOLDOWN_ATR_MIN_FACTOR", 0.5))
COOLDOWN_ATR_MAX_FACTOR: float = max(
    COOLDOWN_ATR_MIN_FACTOR, _float("MICRO_CORE_COOLDOWN_ATR_MAX_FACTOR", 1.5)
)
COOLDOWN_RELEASE_RSI_DELTA: float = max(1.0, _float("MICRO_CORE_COOLDOWN_RELEASE_RSI", 8.0))
COOLDOWN_RELEASE_MIN_REMAIN_SEC: int = int(
    max(5.0, _float("MICRO_CORE_COOLDOWN_RELEASE_MIN_SEC", 30.0))
)
COOLDOWN_RELEASE_MAX_SPREAD: float = max(
    0.5, _float("MICRO_CORE_COOLDOWN_RELEASE_MAX_SPREAD", 1.1)
)
DYNAMIC_LOG_ENABLED: bool = _bool("MICRO_CORE_DYNAMIC_LOG", False)
