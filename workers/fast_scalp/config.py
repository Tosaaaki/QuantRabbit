"""
Configuration helpers for the FastScalp worker.
"""

from __future__ import annotations

import os

PIP_VALUE = 0.01


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


FAST_SCALP_ENABLED: bool = _bool_env("FAST_SCALP_ENABLED", True)
LOOP_INTERVAL_SEC: float = max(0.1, float(os.getenv("FAST_SCALP_LOOP_INTERVAL_SEC", "0.25")))
TP_BASE_PIPS: float = max(0.2, float(os.getenv("FAST_SCALP_TP_BASE_PIPS", "1.0")))
TP_SPREAD_BUFFER_PIPS: float = max(0.05, float(os.getenv("FAST_SCALP_SPREAD_BUFFER_PIPS", "0.2")))
SL_PIPS: float = max(5.0, float(os.getenv("FAST_SCALP_SL_PIPS", "30.0")))
MAX_SPREAD_PIPS: float = max(0.1, float(os.getenv("FAST_SCALP_MAX_SPREAD_PIPS", "0.35")))
ENTRY_THRESHOLD_PIPS: float = max(0.1, float(os.getenv("FAST_SCALP_ENTRY_MOM_PIPS", "0.65")))
ENTRY_SHORT_THRESHOLD_PIPS: float = max(
    0.05, float(os.getenv("FAST_SCALP_ENTRY_SHORT_MOM_PIPS", "0.35"))
)
ENTRY_RANGE_FLOOR_PIPS: float = max(0.1, float(os.getenv("FAST_SCALP_RANGE_FLOOR_PIPS", "0.7")))
ENTRY_COOLDOWN_SEC: float = max(1.0, float(os.getenv("FAST_SCALP_ENTRY_COOLDOWN_SEC", "5.0")))
MAX_ORDERS_PER_MINUTE: int = max(1, int(float(os.getenv("FAST_SCALP_MAX_ORDERS_PER_MIN", "24"))))
MIN_ORDER_SPACING_SEC: float = max(
    0.5, float(os.getenv("FAST_SCALP_MIN_ORDER_SPACING_SEC", "2.5"))
)
MAX_LOT: float = max(0.001, float(os.getenv("FAST_SCALP_MAX_LOT", "0.05")))
SYNC_INTERVAL_SEC: float = max(5.0, float(os.getenv("FAST_SCALP_SYNC_INTERVAL_SEC", "45.0")))
TIMEOUT_SEC: float = max(10.0, float(os.getenv("FAST_SCALP_TIMEOUT_SEC", "65.0")))
TIMEOUT_MIN_GAIN_PIPS: float = float(os.getenv("FAST_SCALP_TIMEOUT_MIN_GAIN_PIPS", "0.4"))
MAX_DRAWDOWN_CLOSE_PIPS: float = max(
    0.5, float(os.getenv("FAST_SCALP_MAX_DRAWDOWN_CLOSE_PIPS", "5.0"))
)
FAST_SHARE_HINT: float = max(0.0, min(1.0, float(os.getenv("FAST_SCALP_SHARE_HINT", "0.35"))))
SHORT_WINDOW_SEC: float = max(0.2, float(os.getenv("FAST_SCALP_SHORT_WINDOW_SEC", "1.4")))
LONG_WINDOW_SEC: float = max(
    SHORT_WINDOW_SEC + 0.2, float(os.getenv("FAST_SCALP_LONG_WINDOW_SEC", "5.5"))
)
MIN_TICK_COUNT: int = max(5, int(float(os.getenv("FAST_SCALP_MIN_TICK_COUNT", "15"))))
TP_SAFE_MARGIN_PIPS: float = max(
    0.1, float(os.getenv("FAST_SCALP_TP_SAFE_MARGIN_PIPS", "0.4"))
)
JST_OFF_HOURS_START: int = min(23, max(0, int(float(os.getenv("FAST_SCALP_OFF_HOURS_START_JST", "3")))))
JST_OFF_HOURS_END: int = min(23, max(0, int(float(os.getenv("FAST_SCALP_OFF_HOURS_END_JST", "5")))))
LOG_PREFIX_TICK = "[SCALP-TICK]"
MIN_UNITS: int = max(1000, int(float(os.getenv("FAST_SCALP_MIN_UNITS", "10000"))))
