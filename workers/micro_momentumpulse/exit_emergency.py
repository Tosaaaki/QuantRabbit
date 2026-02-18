from __future__ import annotations

import os
import time

from utils.metrics_logger import log_metric
from utils.oanda_account import get_account_snapshot

_ALLOW_NEGATIVE = os.getenv("EXIT_EMERGENCY_ALLOW_NEGATIVE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_HEALTH_BUFFER = max(0.0, float(os.getenv("EXIT_EMERGENCY_HEALTH_BUFFER", "0.07")))
_MARGIN_USAGE_RATIO = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_MARGIN_USAGE_RATIO", "0.92"))
)
_FREE_MARGIN_RATIO = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_FREE_MARGIN_RATIO", "0.12"))
)
_UNREALIZED_DD_RATIO = max(
    0.0, float(os.getenv("EXIT_EMERGENCY_UNREALIZED_DD_RATIO", "0.06"))
)
_CACHE_TTL_SEC = max(0.5, float(os.getenv("EXIT_EMERGENCY_CACHE_TTL_SEC", "2.0")))
_STALE_GRACE_SEC = max(2.0, float(os.getenv("EXIT_EMERGENCY_STALE_GRACE_SEC", "6.0")))

_LAST_CHECK_TS: float | None = None
_LAST_ALLOW: bool = False
_LAST_LOG_TS: float = 0.0


def should_allow_negative_close() -> bool:
    if not _ALLOW_NEGATIVE:
        return False
    now = time.monotonic()
    if _LAST_CHECK_TS is not None and now - _LAST_CHECK_TS < _CACHE_TTL_SEC:
        return _LAST_ALLOW
    try:
        snapshot = get_account_snapshot(cache_ttl_sec=_CACHE_TTL_SEC)
        hb = snapshot.health_buffer
        free_ratio = snapshot.free_margin_ratio
        nav = snapshot.nav or 0.0
        margin_used = snapshot.margin_used
        unrealized = snapshot.unrealized_pl
    except Exception:
        if _LAST_ALLOW and _LAST_CHECK_TS is not None and now - _LAST_CHECK_TS < _STALE_GRACE_SEC:
            return True
        return False
    allow = False
    reasons = {}
    if hb is not None and _HEALTH_BUFFER > 0 and hb <= _HEALTH_BUFFER:
        allow = True
        reasons["health_buffer"] = hb
    if free_ratio is not None and _FREE_MARGIN_RATIO > 0 and free_ratio <= _FREE_MARGIN_RATIO:
        allow = True
        reasons["free_margin_ratio"] = free_ratio
    if nav > 0 and _MARGIN_USAGE_RATIO > 0:
        usage_ratio = (margin_used or 0.0) / nav
        if usage_ratio >= _MARGIN_USAGE_RATIO:
            allow = True
            reasons["margin_usage_ratio"] = usage_ratio
    if nav > 0 and _UNREALIZED_DD_RATIO > 0 and unrealized < 0:
        dd_ratio = abs(unrealized) / nav
        if dd_ratio >= _UNREALIZED_DD_RATIO:
            allow = True
            reasons["unrealized_dd_ratio"] = dd_ratio
    if allow:
        global _LAST_LOG_TS
        if now - _LAST_LOG_TS >= 30.0:
            log_metric(
                "exit_emergency_allow_negative",
                float(hb) if hb is not None else -1.0,
                tags={
                    "threshold": _HEALTH_BUFFER,
                    "reason": next(iter(reasons.keys()), "unknown"),
                },
            )
            _LAST_LOG_TS = now
    globals()["_LAST_ALLOW"] = allow
    globals()["_LAST_CHECK_TS"] = now
    return allow
