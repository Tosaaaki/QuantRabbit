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
