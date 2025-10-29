from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

_PIP = 0.01


def _to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pip_delta(a: float, b: float) -> float:
    return abs(a - b) / _PIP


@dataclass(frozen=True)
class PivotPoint:
    index: int
    timestamp: Optional[str]
    price: float
    kind: Literal["high", "low"]


@dataclass(frozen=True)
class NWaveStructure:
    direction: Literal["long", "short"]
    pivots: Tuple[PivotPoint, PivotPoint, PivotPoint, PivotPoint]
    amplitude_pips: float
    pullback_pips: float
    quality: float
    entry_price: float
    invalidation_price: float
    breakout_price: float


def extract_pivots(
    candles: Iterable[dict],
    *,
    window: int = 3,
    min_pips: float = 1.2,
    max_points: Optional[int] = 120,
) -> List[PivotPoint]:
    buffer: List[dict] = list(candles or [])
    if not buffer:
        return []
    if max_points and len(buffer) > max_points:
        buffer = buffer[-max_points:]

    pivots: List[PivotPoint] = []
    last_price: Optional[float] = None
    last_kind: Optional[str] = None
    size = len(buffer)

    for idx in range(window, size - window):
        candle = buffer[idx]
        high = _to_float(candle.get("high"))
        low = _to_float(candle.get("low"))
        if high is None or low is None:
            continue

        is_pivot_high = True
        is_pivot_low = True
        for off in range(idx - window, idx + window + 1):
            if off == idx:
                continue
            other = buffer[off]
            other_high = _to_float(other.get("high"))
            other_low = _to_float(other.get("low"))
            if other_high is not None and other_high > high:
                is_pivot_high = False
            if other_low is not None and other_low < low:
                is_pivot_low = False
            if not is_pivot_high and not is_pivot_low:
                break

        pivot: Optional[PivotPoint] = None
        if is_pivot_high:
            pivot = PivotPoint(
                index=idx,
                timestamp=candle.get("timestamp"),
                price=high,
                kind="high",
            )
        elif is_pivot_low:
            pivot = PivotPoint(
                index=idx,
                timestamp=candle.get("timestamp"),
                price=low,
                kind="low",
            )

        if not pivot:
            continue

        if last_price is not None and pivot.kind != last_kind:
            if _pip_delta(pivot.price, last_price) < min_pips:
                continue

        if pivots and pivots[-1].kind == pivot.kind:
            previous = pivots[-1]
            if pivot.kind == "high" and pivot.price > previous.price:
                pivots[-1] = pivot
                last_price = pivot.price
                last_kind = pivot.kind
            elif pivot.kind == "low" and pivot.price < previous.price:
                pivots[-1] = pivot
                last_price = pivot.price
                last_kind = pivot.kind
            continue

        pivots.append(pivot)
        last_price = pivot.price
        last_kind = pivot.kind

    return pivots


def detect_latest_n_wave(
    candles: Iterable[dict],
    *,
    window: int = 3,
    min_leg_pips: float = 3.0,
    min_quality: float = 0.15,
    max_points: Optional[int] = 120,
) -> Optional[NWaveStructure]:
    pivots = extract_pivots(candles, window=window, min_pips=1.0, max_points=max_points)
    if len(pivots) < 4:
        return None

    pip = _PIP

    for start in range(len(pivots) - 4, -1, -1):
        seq = pivots[start : start + 4]
        kinds = [p.kind for p in seq]
        a, b, c, d = seq

        if kinds == ["low", "high", "low", "high"]:
            if c.price <= a.price or d.price <= b.price:
                continue
            leg1 = (b.price - a.price) / pip
            leg2 = (d.price - c.price) / pip
            pullback = (b.price - c.price) / pip
            amplitude = (d.price - a.price) / pip
            if leg1 < min_leg_pips or leg2 < min_leg_pips * 0.6:
                continue
            quality = min(leg1, leg2) / max(pullback, 0.5)
            if quality < min_quality:
                continue
            entry_price = c.price + 0.2 * pip
            invalidation = min(a.price, c.price) - 1.8 * pip
            breakout_price = b.price + 0.2 * pip
            return NWaveStructure(
                direction="long",
                pivots=(a, b, c, d),
                amplitude_pips=amplitude,
                pullback_pips=pullback,
                quality=quality,
                entry_price=entry_price,
                invalidation_price=invalidation,
                breakout_price=breakout_price,
            )

        if kinds == ["high", "low", "high", "low"]:
            if c.price >= a.price or d.price >= b.price:
                continue
            leg1 = (a.price - b.price) / pip
            leg2 = (c.price - d.price) / pip
            pullback = (c.price - b.price) / pip
            amplitude = (a.price - d.price) / pip
            if leg1 < min_leg_pips or leg2 < min_leg_pips * 0.6:
                continue
            quality = min(leg1, leg2) / max(pullback, 0.5)
            if quality < min_quality:
                continue
            entry_price = c.price - 0.2 * pip
            invalidation = max(a.price, c.price) + 1.8 * pip
            breakout_price = b.price - 0.2 * pip
            return NWaveStructure(
                direction="short",
                pivots=(a, b, c, d),
                amplitude_pips=amplitude,
                pullback_pips=pullback,
                quality=quality,
                entry_price=entry_price,
                invalidation_price=invalidation,
                breakout_price=breakout_price,
            )

    return None
