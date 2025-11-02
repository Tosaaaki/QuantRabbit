from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

PIP = 0.01
POINT = 0.001


def to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    """Convert arbitrary numeric input to float with fallback."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def atr_pips(factors: Dict[str, object], *, default: float = 0.0) -> float:
    """Resolve ATR in pip units with graceful fallback."""
    direct = to_float(factors.get("atr_pips"))
    if direct is not None:
        return float(direct)
    atr_raw = to_float(factors.get("atr"))
    if atr_raw is None:
        return float(default)
    return float(max(0.0, atr_raw) * 100.0)


def latest_candles(factors: Dict[str, object], count: int) -> Tuple[Dict[str, float], ...]:
    """Return the last N candles as tuple to avoid accidental mutation."""
    candles = factors.get("candles") or []
    if not isinstance(candles, Iterable):
        return tuple()
    tail = []
    for candle in candles[-count:]:
        if isinstance(candle, dict):
            tail.append(candle)
    return tuple(tail)


def candle_close(candle: Dict[str, object]) -> Optional[float]:
    return to_float(candle.get("close"))


def candle_high(candle: Dict[str, object]) -> Optional[float]:
    return to_float(candle.get("high"))


def candle_low(candle: Dict[str, object]) -> Optional[float]:
    return to_float(candle.get("low"))


def candle_body_pips(candle: Dict[str, object]) -> Optional[float]:
    open_px = to_float(candle.get("open"))
    close_px = to_float(candle.get("close"))
    if open_px is None or close_px is None:
        return None
    return (close_px - open_px) / PIP


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def price_delta_pips(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0
    return float((a - b) / PIP)


def typical_price(candle: Dict[str, object]) -> Optional[float]:
    high = candle_high(candle)
    low = candle_low(candle)
    close = candle_close(candle)
    if None in (high, low, close):
        return None
    return (high + low + close) / 3.0
