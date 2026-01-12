"""Pullback touch counter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(slots=True)
class PullbackTouchResult:
    count: int
    last_touch_ts: Optional[float]
    last_touch_price: Optional[float]
    extreme_price: Optional[float]
    trend_pips: Optional[float]


def _clean_series(
    prices: Sequence[float],
    timestamps: Optional[Sequence[float]],
) -> Tuple[list[float], Optional[list[float]]]:
    if timestamps is not None and len(timestamps) == len(prices):
        clean_prices: list[float] = []
        clean_times: list[float] = []
        for price, ts in zip(prices, timestamps):
            try:
                clean_prices.append(float(price))
                clean_times.append(float(ts))
            except (TypeError, ValueError):
                continue
        return clean_prices, clean_times if clean_times else None
    clean_prices = []
    for price in prices:
        try:
            clean_prices.append(float(price))
        except (TypeError, ValueError):
            continue
    return clean_prices, None


def count_pullback_touches(
    prices: Sequence[float],
    side: str,
    pullback_pips: float,
    trend_confirm_pips: float,
    *,
    reset_pips: Optional[float] = None,
    pip_value: float = 0.01,
    timestamps: Optional[Sequence[float]] = None,
) -> PullbackTouchResult:
    clean_prices, clean_times = _clean_series(prices, timestamps)
    if len(clean_prices) < 3:
        return PullbackTouchResult(0, None, None, None, None)
    side = (side or "").lower()
    if side not in {"long", "short"}:
        return PullbackTouchResult(0, None, None, None, None)
    if pullback_pips <= 0.0 or trend_confirm_pips <= 0.0:
        return PullbackTouchResult(0, None, None, None, None)

    pip = max(1e-6, float(pip_value))
    reset_pips = max(0.05, float(reset_pips) if reset_pips is not None else pullback_pips * 0.45)
    direction = 1.0 if side == "long" else -1.0

    anchor = clean_prices[0] * direction
    extreme = anchor
    trend_confirmed = False
    in_pullback = False
    count = 0
    last_touch_idx: Optional[int] = None

    for idx, price in enumerate(clean_prices[1:], start=1):
        signed_price = price * direction

        if signed_price < anchor:
            anchor = signed_price
            extreme = signed_price
            trend_confirmed = False
            in_pullback = False
            count = 0
            last_touch_idx = None
            continue

        if signed_price > extreme:
            extreme = signed_price
            in_pullback = False

        if not trend_confirmed:
            if extreme - anchor >= trend_confirm_pips * pip:
                trend_confirmed = True
            else:
                continue

        pullback_level = extreme - pullback_pips * pip
        if not in_pullback and signed_price <= pullback_level:
            count += 1
            in_pullback = True
            last_touch_idx = idx

        if in_pullback:
            reset_level = extreme - reset_pips * pip
            if signed_price >= reset_level:
                in_pullback = False

    last_touch_ts = None
    if last_touch_idx is not None and clean_times:
        last_touch_ts = clean_times[last_touch_idx]

    last_touch_price = clean_prices[last_touch_idx] if last_touch_idx is not None else None
    extreme_price = extreme * direction if clean_prices else None
    trend_pips = (extreme - anchor) / pip if clean_prices else None
    if not trend_confirmed:
        trend_pips = None

    return PullbackTouchResult(
        count=count,
        last_touch_ts=last_touch_ts,
        last_touch_price=last_touch_price,
        extreme_price=extreme_price,
        trend_pips=trend_pips,
    )
