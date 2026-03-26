"""
analysis.mtf_utils
~~~~~~~~~~~~~~~~~~
Lightweight helpers to build lower-timeframe aggregates from M1 candles for
multi-timeframe confluence without changing the trading timeframe.
"""

from __future__ import annotations

from typing import Iterable, List, Dict
from datetime import datetime, timezone


def _parse_ts(ts: str) -> datetime:
    s = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Fallback: trim fractional seconds
        if "." in s:
            head, tail = s.split(".", 1)
            # keep timezone if present
            tz = ""
            if "+" in tail:
                tail, tz = tail.split("+", 1)
                tz = "+" + tz
            elif "-" in tail[6:]:
                tail, tz = tail.split("-", 1)
                tz = "-" + tz
            s = head + tz
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)


def resample_candles_from_m1(
    m1_candles: Iterable[Dict[str, float]], minutes: int, max_bars: int = 240
) -> List[Dict[str, float]]:
    """
    Aggregate M1 candles into N-minute OHLC bars.

    Parameters
    ----------
    m1_candles: iterable of dict with keys: timestamp, open, high, low, close
    minutes: aggregation size in minutes (e.g., 5, 10, 60)
    max_bars: optional cap on returned bar count (most recent)
    """
    if minutes <= 1:
        out = [
            {
                "timestamp": c.get("timestamp"),
                "open": float(c.get("open", 0.0)),
                "high": float(c.get("high", 0.0)),
                "low": float(c.get("low", 0.0)),
                "close": float(c.get("close", 0.0)),
            }
            for c in m1_candles
            if c
        ]
        return out[-max_bars:]

    buckets: Dict[datetime, Dict[str, float]] = {}
    order: List[datetime] = []
    for c in m1_candles:
        try:
            ts = _parse_ts(str(c["timestamp"]))
        except Exception:
            continue
        # floor to bucket
        minute = (ts.minute // minutes) * minutes
        bstart = ts.replace(minute=minute, second=0, microsecond=0)
        # normalize to UTC and naive ISO later
        if bstart.tzinfo is None:
            bstart = bstart.replace(tzinfo=timezone.utc)
        if bstart not in buckets:
            buckets[bstart] = {
                "timestamp": bstart.isoformat().replace("+00:00", "Z"),
                "open": float(c.get("open", c.get("close", 0.0))),
                "high": float(c.get("high", c.get("close", 0.0))),
                "low": float(c.get("low", c.get("close", 0.0))),
                "close": float(c.get("close", 0.0)),
            }
            order.append(bstart)
        else:
            agg = buckets[bstart]
            h = float(c.get("high", agg["high"]))
            l = float(c.get("low", agg["low"]))
            agg["high"] = max(agg["high"], h)
            agg["low"] = min(agg["low"], l)
            agg["close"] = float(c.get("close", agg["close"]))

    order.sort()
    out = [buckets[k] for k in order]
    # Ensure strictly increasing timestamps and cap
    return out[-max_bars:]
