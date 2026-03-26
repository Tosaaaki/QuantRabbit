from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional


def parse_rollout_start_ts(value: object, *, default: float = 0.0) -> float:
    """Parse rollout start timestamp.

    Accepted formats:
    - Unix seconds (int/float/string)
    - Unix milliseconds (auto-detected when value is very large)
    - ISO8601 datetime string (naive values are treated as UTC)
    """
    fallback = max(0.0, float(default))
    if value is None:
        return fallback

    ts: Optional[float] = None
    if isinstance(value, (int, float)):
        ts = float(value)
    else:
        text = str(value).strip()
        if not text:
            return fallback
        try:
            ts = float(text)
        except ValueError:
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
            except Exception:
                return fallback

    if ts is None:
        return fallback
    if ts > 1_000_000_000_000:  # milliseconds
        ts /= 1000.0
    if ts <= 0:
        return 0.0
    return float(ts)


def parse_trade_open_ts(open_time: object) -> Optional[float]:
    if open_time is None:
        return None
    if isinstance(open_time, datetime):
        dt = open_time
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(open_time, (int, float)):
        ts = float(open_time)
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        return ts if ts > 0 else None

    text = str(open_time).strip()
    if not text:
        return None
    try:
        ts = float(text)
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        return ts if ts > 0 else None
    except ValueError:
        pass
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def trade_passes_rollout(
    open_time: object,
    start_ts: float,
    *,
    unknown_is_new: bool = False,
) -> bool:
    start = max(0.0, float(start_ts))
    if start <= 0.0:
        return True
    ts = parse_trade_open_ts(open_time)
    if ts is None:
        return bool(unknown_is_new)
    return float(ts) >= start


def load_rollout_start_ts(env_name: str, *, default: float = 0.0) -> float:
    return parse_rollout_start_ts(os.getenv(env_name), default=default)
