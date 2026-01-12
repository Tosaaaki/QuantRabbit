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
_CACHE_TTL_SEC = max(0.5, float(os.getenv("EXIT_EMERGENCY_CACHE_TTL_SEC", "2.0")))
_STALE_GRACE_SEC = max(2.0, float(os.getenv("EXIT_EMERGENCY_STALE_GRACE_SEC", "6.0")))

_LAST_CHECK_TS: float | None = None
_LAST_ALLOW: bool = False
_LAST_LOG_TS: float = 0.0


def should_allow_negative_close() -> bool:
    if not _ALLOW_NEGATIVE or _HEALTH_BUFFER <= 0:
        return False
    now = time.monotonic()
    if _LAST_CHECK_TS is not None and now - _LAST_CHECK_TS < _CACHE_TTL_SEC:
        return _LAST_ALLOW
    try:
        snapshot = get_account_snapshot(cache_ttl_sec=_CACHE_TTL_SEC)
        hb = snapshot.health_buffer
    except Exception:
        if _LAST_ALLOW and _LAST_CHECK_TS is not None and now - _LAST_CHECK_TS < _STALE_GRACE_SEC:
            return True
        return False
    allow = hb is not None and hb <= _HEALTH_BUFFER
    if allow:
        global _LAST_LOG_TS
        if now - _LAST_LOG_TS >= 30.0:
            log_metric(
                "exit_emergency_allow_negative",
                float(hb),
                tags={"threshold": _HEALTH_BUFFER},
            )
            _LAST_LOG_TS = now
    globals()["_LAST_ALLOW"] = allow
    globals()["_LAST_CHECK_TS"] = now
    return allow
