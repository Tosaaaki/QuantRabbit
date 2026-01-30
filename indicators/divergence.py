"""Divergence detection utilities for price vs oscillator series."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Iterable, Sequence

import numpy as np

PIP_DEFAULT = 0.01


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


DEFAULT_PIVOT_WINDOW = _env_int("DIV_PIVOT_WINDOW", 3)
DEFAULT_MIN_SEP = _env_int("DIV_MIN_SEP", 3)
DEFAULT_LOOKBACK_BARS = _env_int("DIV_LOOKBACK_BARS", 120)
DEFAULT_MAX_AGE_BARS = _env_int("DIV_MAX_AGE_BARS", 30)
DEFAULT_MIN_PRICE_PIPS = _env_float("DIV_MIN_PRICE_PIPS", 0.8)
DEFAULT_MIN_RSI = _env_float("DIV_MIN_RSI", 4.0)
DEFAULT_MIN_MACD_PIPS = _env_float("DIV_MIN_MACD_PIPS", 0.2)


@dataclass
class DivergenceResult:
    kind: int = 0
    score: float = 0.0
    age_bars: int = 0
    strength: float = 0.0
    price_delta_pips: float = 0.0
    osc_delta: float = 0.0
    pivot_gap: int = 0


def _to_array(values: Sequence[float] | Iterable[float]) -> np.ndarray:
    try:
        arr = np.asarray(list(values), dtype=float)
    except Exception:
        arr = np.asarray([], dtype=float)
    return arr


def _pivot_indices(
    series: np.ndarray,
    window: int,
    *,
    kind: str,
    min_sep: int,
) -> list[int]:
    n = len(series)
    if n < window * 2 + 1:
        return []
    indices: list[int] = []
    for i in range(window, n - window):
        val = series[i]
        if not math.isfinite(val):
            continue
        segment = series[i - window : i + window + 1]
        if not np.isfinite(segment).all():
            continue
        if kind == "high":
            if np.argmax(segment) != window:
                continue
        else:
            if np.argmin(segment) != window:
                continue
        if indices and i - indices[-1] < min_sep:
            prev_idx = indices[-1]
            prev_val = series[prev_idx]
            if kind == "high":
                if val >= prev_val:
                    indices[-1] = i
            else:
                if val <= prev_val:
                    indices[-1] = i
            continue
        indices.append(i)
    return indices


def _select_recent(pivots: list[int], *, total: int, lookback: int) -> list[int]:
    if not pivots:
        return []
    if lookback <= 0 or lookback >= total:
        return pivots[-2:]
    cutoff = max(0, total - lookback)
    recent = [idx for idx in pivots if idx >= cutoff]
    if len(recent) >= 2:
        return recent[-2:]
    return pivots[-2:] if len(pivots) >= 2 else []


def _strength(
    price_delta_pips: float,
    osc_delta: float,
    *,
    min_price_pips: float,
    min_osc: float,
) -> float:
    price_scale = max(min_price_pips * 2.0, 1e-6)
    osc_scale = max(min_osc * 2.0, 1e-6)
    price_term = min(1.0, abs(price_delta_pips) / price_scale)
    osc_term = min(1.0, abs(osc_delta) / osc_scale)
    return max(0.0, min(1.0, price_term * osc_term))


def _calc_divergence(
    price_series: np.ndarray,
    osc: np.ndarray,
    pivots: list[int],
    *,
    is_low: bool,
    pip_value: float,
    min_price_pips: float,
    min_osc: float,
) -> DivergenceResult | None:
    if len(pivots) < 2:
        return None
    i1, i2 = pivots[-2], pivots[-1]
    if i1 < 0 or i2 < 0 or i2 >= len(price_series) or i2 >= len(osc):
        return None
    p1 = price_series[i1]
    p2 = price_series[i2]
    o1 = osc[i1]
    o2 = osc[i2]
    if not all(map(math.isfinite, (p1, p2, o1, o2))):
        return None
    price_delta_pips = (p2 - p1) / max(pip_value, 1e-9)
    osc_delta = o2 - o1
    reg = False
    hidden = False
    if is_low:
        reg = price_delta_pips <= -min_price_pips and osc_delta >= min_osc
        hidden = price_delta_pips >= min_price_pips and osc_delta <= -min_osc
    else:
        reg = price_delta_pips >= min_price_pips and osc_delta <= -min_osc
        hidden = price_delta_pips <= -min_price_pips and osc_delta >= min_osc
    if not reg and not hidden:
        return None
    kind = 0
    if reg and is_low:
        kind = 1
    elif hidden and is_low:
        kind = 2
    elif reg:
        kind = -1
    elif hidden:
        kind = -2
    strength = _strength(
        price_delta_pips,
        osc_delta,
        min_price_pips=min_price_pips,
        min_osc=min_osc,
    )
    base = 1.0 if abs(kind) == 1 else 0.6
    score = base * (1.0 if kind > 0 else -1.0) * strength
    age = (len(price_series) - 1) - i2
    return DivergenceResult(
        kind=kind,
        score=score,
        age_bars=age,
        strength=strength,
        price_delta_pips=price_delta_pips,
        osc_delta=osc_delta,
        pivot_gap=i2 - i1,
    )


def compute_divergence(
    *,
    price_high: Sequence[float] | Iterable[float],
    price_low: Sequence[float] | Iterable[float],
    osc: Sequence[float] | Iterable[float],
    pivot_window: int | None = None,
    min_sep: int | None = None,
    lookback_bars: int | None = None,
    min_price_pips: float | None = None,
    min_osc: float | None = None,
    max_age_bars: int | None = None,
    pip_value: float | None = None,
) -> DivergenceResult:
    high_arr = _to_array(price_high)
    low_arr = _to_array(price_low)
    osc_arr = _to_array(osc)
    n = min(len(high_arr), len(low_arr), len(osc_arr))
    if n <= 0:
        return DivergenceResult()
    high_arr = high_arr[-n:]
    low_arr = low_arr[-n:]
    osc_arr = osc_arr[-n:]

    pivot_window = DEFAULT_PIVOT_WINDOW if pivot_window is None else max(1, pivot_window)
    min_sep = DEFAULT_MIN_SEP if min_sep is None else max(1, min_sep)
    lookback_bars = DEFAULT_LOOKBACK_BARS if lookback_bars is None else max(1, lookback_bars)
    min_price_pips = DEFAULT_MIN_PRICE_PIPS if min_price_pips is None else max(0.0, min_price_pips)
    min_osc = DEFAULT_MIN_RSI if min_osc is None else max(0.0, min_osc)
    max_age_bars = DEFAULT_MAX_AGE_BARS if max_age_bars is None else max(0, max_age_bars)
    pip_value = PIP_DEFAULT if pip_value is None else max(pip_value, 1e-9)

    if n < pivot_window * 2 + 3:
        return DivergenceResult()

    lows = _pivot_indices(low_arr, pivot_window, kind="low", min_sep=min_sep)
    highs = _pivot_indices(high_arr, pivot_window, kind="high", min_sep=min_sep)

    lows = _select_recent(lows, total=n, lookback=lookback_bars)
    highs = _select_recent(highs, total=n, lookback=lookback_bars)

    bull = _calc_divergence(
        low_arr,
        osc_arr,
        lows,
        is_low=True,
        pip_value=pip_value,
        min_price_pips=min_price_pips,
        min_osc=min_osc,
    )
    bear = _calc_divergence(
        high_arr,
        osc_arr,
        highs,
        is_low=False,
        pip_value=pip_value,
        min_price_pips=min_price_pips,
        min_osc=min_osc,
    )

    candidates = [c for c in (bull, bear) if c is not None]
    if not candidates:
        return DivergenceResult()
    best = sorted(candidates, key=lambda c: (c.age_bars, -c.strength))[0]
    if max_age_bars >= 0 and best.age_bars > max_age_bars:
        return DivergenceResult()
    return best

