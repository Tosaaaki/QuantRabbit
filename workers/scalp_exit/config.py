"""Configuration for scalp exit manager."""

from __future__ import annotations

import os


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


LOG_PREFIX = "[SCALP-EXIT]"
ENABLED: bool = _bool("SCALP_EXIT_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.2, _float("SCALP_EXIT_LOOP_INTERVAL_SEC", 0.5))

BASE_PROFIT_PIPS: float = max(1.0, _float("SCALP_EXIT_BASE_PROFIT_PIPS", 3.5))
HARD_PROFIT_PIPS: float = max(BASE_PROFIT_PIPS + 1.0, _float("SCALP_EXIT_HARD_PROFIT_PIPS", 9.0))
PROFIT_Z_THRESHOLD: float = _float("SCALP_EXIT_PROFIT_Z", 1.5)
RSI_EXIT_SHORT: float = _float("SCALP_EXIT_RSI_SHORT", 34.0)
RSI_EXIT_LONG: float = _float("SCALP_EXIT_RSI_LONG", 66.0)

TRAIL_START_PIPS: float = max(1.0, _float("SCALP_EXIT_TRAIL_START_PIPS", 3.0))
TRAIL_BACKOFF_PIPS: float = max(0.3, _float("SCALP_EXIT_TRAIL_BACKOFF_PIPS", 1.2))
LOCK_AT_PROFIT_PIPS: float = max(1.0, _float("SCALP_EXIT_LOCK_AT_PROFIT_PIPS", 2.2))
LOCK_BUFFER_PIPS: float = max(0.1, _float("SCALP_EXIT_LOCK_BUFFER_PIPS", 0.6))

HARD_STOP_PIPS: float = max(2.0, _float("SCALP_EXIT_HARD_STOP_PIPS", 7.0))
NEGATIVE_HOLD_TIMEOUT_SEC: float = max(60.0, _float("SCALP_EXIT_NEGATIVE_HOLD_TIMEOUT_SEC", 300.0))
MAX_HOLD_SEC: float = max(120.0, _float("SCALP_EXIT_MAX_HOLD_SEC", 1800.0))

STRATEGY_WHITELIST = {
    tag.strip()
    for tag in os.getenv("SCALP_EXIT_STRATEGIES", "pullback_scalp,spike_scalp,fast_scalp").split(",")
    if tag.strip()
}
