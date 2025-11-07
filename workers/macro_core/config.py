"""Configuration for macro core worker."""

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


LOG_PREFIX = "[MACRO-CORE]"
ENABLED: bool = _bool("MACRO_CORE_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.5, _float("MACRO_CORE_LOOP_INTERVAL_SEC", 1.0))
PLAN_STALE_SEC: float = max(1.0, _float("MACRO_CORE_PLAN_STALE_SEC", 5.0))
