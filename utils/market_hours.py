"""Utility helpers to infer market activity from cached candle data."""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Tuple

_DEFAULT_THRESHOLD_SEC = 600  # 10 minutes
_JST = timezone(timedelta(hours=9))


def _resolve_threshold(custom: int | None) -> int:
    if custom is not None and custom > 0:
        return custom
    env_val = os.environ.get("MARKET_STALE_AFTER_SEC")
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return _DEFAULT_THRESHOLD_SEC


def latest_candle_time(factors: Dict[str, Any] | None) -> datetime | None:
    """Extract the timestamp of the freshest candle from a factor payload."""

    if not isinstance(factors, dict):
        return None
    candles = factors.get("candles")
    if not isinstance(candles, list) or not candles:
        return None
    latest = candles[-1]
    ts = latest.get("timestamp")
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if isinstance(ts, str):
        norm = ts.strip()
        if not norm:
            return None
        if norm.endswith("Z") or norm.endswith("z"):
            norm = norm[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(norm)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def is_market_open(
    factors: Dict[str, Any] | None,
    *,
    now: datetime | None = None,
    stale_after_sec: int | None = None,
) -> Tuple[bool, float | None]:
    """Infer whether the market feed is active based on candle recency."""

    last_ts = latest_candle_time(factors)
    if not last_ts:
        return False, None

    now_dt = now if now is not None else datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    age_sec = (now_dt - last_ts).total_seconds()
    if age_sec < 0:
        age_sec = 0.0

    threshold = float(_resolve_threshold(stale_after_sec))
    if age_sec > threshold:
        return False, age_sec
    return True, age_sec


def is_scalp_weekend(now: datetime | None = None) -> bool:
    """Return True during the weekend shutdown window for scalping (JST Sat 05:00 → Mon 07:00)."""

    now_dt = now if now is not None else datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    now_jst = now_dt.astimezone(_JST)
    dow = now_jst.weekday()  # Monday=0
    minutes = now_jst.hour * 60 + now_jst.minute

    if dow == 5:  # Saturday
        return minutes >= 5 * 60
    if dow == 6:  # Sunday
        return True
    if dow == 0:  # Monday before 07:00
        return minutes < 7 * 60
    return False


def is_scalp_session_open(now: datetime | None = None) -> bool:
    """Return True when scalping is allowed (outside the weekend window)."""

    return not is_scalp_weekend(now=now)
