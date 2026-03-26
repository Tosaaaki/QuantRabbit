"""
analysis.range_model
~~~~~~~~~~~~~~~~~~~~
レンジ帯の共通計算ロジック。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence


@dataclass(slots=True)
class RangeSnapshot:
    high: float
    low: float
    mid: float
    method: str
    lookback: int
    hi_pct: float
    lo_pct: float
    end_time: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "high": self.high,
            "low": self.low,
            "mid": self.mid,
            "method": self.method,
            "lookback": self.lookback,
            "hi_pct": self.hi_pct,
            "lo_pct": self.lo_pct,
            "end_time": self.end_time,
        }


def _percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if pct <= 0.0:
        return vals[0]
    if pct >= 100.0:
        return vals[-1]
    k = (len(vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(len(vals) - 1, f + 1)
    if f == c:
        return vals[f]
    d = k - f
    return vals[f] * (1.0 - d) + vals[c] * d


def compute_range_snapshot(
    candles: Sequence[Dict[str, object]],
    *,
    lookback: int,
    method: str = "percentile",
    hi_pct: float = 95.0,
    lo_pct: float = 5.0,
) -> Optional[RangeSnapshot]:
    if not candles or lookback <= 0:
        return None
    window = list(candles[-lookback:]) if len(candles) > lookback else list(candles)
    highs = [float(c.get("high") or 0.0) for c in window if c.get("high") is not None]
    lows = [float(c.get("low") or 0.0) for c in window if c.get("low") is not None]
    if not highs or not lows:
        return None
    method_key = (method or "").lower().strip()
    if method_key == "percentile":
        high = _percentile(highs, hi_pct)
        low = _percentile(lows, lo_pct)
    elif method_key == "donchian":
        high = max(highs)
        low = min(lows)
    else:
        return None
    mid = 0.5 * (high + low)
    end_time = window[-1].get("timestamp") if window else None
    return RangeSnapshot(
        high=high,
        low=low,
        mid=mid,
        method=method_key,
        lookback=len(window),
        hi_pct=float(hi_pct),
        lo_pct=float(lo_pct),
        end_time=end_time if isinstance(end_time, str) else None,
    )
