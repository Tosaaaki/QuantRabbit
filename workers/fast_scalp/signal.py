"""
Helpers to derive scalp signals from recent tick data.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Optional

from market_data import tick_window

from . import config


@dataclass(frozen=True)
class SignalFeatures:
    latest_mid: float
    spread_pips: float
    momentum_pips: float
    short_momentum_pips: float
    range_pips: float
    tick_count: int
    span_seconds: float


def _as_pips(delta: float) -> float:
    return delta / config.PIP_VALUE


def _window_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return mean(values)


def extract_features(spread_pips: float) -> Optional[SignalFeatures]:
    ticks = tick_window.recent_ticks(seconds=config.LONG_WINDOW_SEC, limit=180)
    if len(ticks) < config.MIN_TICK_COUNT:
        return None

    mids = [float(t["mid"]) for t in ticks]
    latest_mid = mids[-1]
    long_mean = _window_mean(mids)

    short_window = max(5, int(len(mids) * config.SHORT_WINDOW_SEC / config.LONG_WINDOW_SEC))
    short_slice = mids[-short_window:]
    short_mean = _window_mean(short_slice)

    high_mid = max(mids)
    low_mid = min(mids)

    span_seconds = float(ticks[-1]["epoch"] - ticks[0]["epoch"])

    momentum = _as_pips(latest_mid - long_mean)
    short_momentum = _as_pips(latest_mid - short_mean)
    range_pips = _as_pips(high_mid - low_mid)

    return SignalFeatures(
        latest_mid=latest_mid,
        spread_pips=spread_pips,
        momentum_pips=momentum,
        short_momentum_pips=short_momentum,
        range_pips=range_pips,
        tick_count=len(ticks),
        span_seconds=span_seconds,
    )


def evaluate_signal(features: SignalFeatures) -> Optional[str]:
    if features.range_pips < config.ENTRY_RANGE_FLOOR_PIPS:
        return None
    if abs(features.momentum_pips) < config.ENTRY_THRESHOLD_PIPS:
        return None
    if abs(features.short_momentum_pips) < config.ENTRY_SHORT_THRESHOLD_PIPS:
        return None
    if features.tick_count < config.MIN_TICK_COUNT:
        return None
    if features.span_seconds <= 0.0:
        return None

    if features.momentum_pips > 0:
        return "OPEN_LONG"
    if features.momentum_pips < 0:
        return "OPEN_SHORT"
    return None

