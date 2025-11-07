"""Configuration for scalp core worker."""

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


LOG_PREFIX = "[SCALP-CORE]"
ENABLED: bool = _bool("SCALP_CORE_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("SCALP_CORE_LOOP_INTERVAL_SEC", 0.5))
PLAN_STALE_SEC: float = max(0.5, _float("SCALP_CORE_PLAN_STALE_SEC", 3.0))
