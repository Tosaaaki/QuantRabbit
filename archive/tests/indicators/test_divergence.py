from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indicators.divergence import compute_divergence  # noqa: E402


def test_regular_bullish_divergence():
    price = [10, 9, 8, 9, 10, 9, 7, 8, 9, 10]
    osc = [30, 28, 25, 30, 32, 29, 35, 36, 38, 40]
    res = compute_divergence(
        price_high=price,
        price_low=price,
        osc=osc,
        pivot_window=1,
        min_sep=2,
        lookback_bars=20,
        min_price_pips=1.0,
        min_osc=3.0,
        max_age_bars=10,
        pip_value=1.0,
    )
    assert res.kind == 1
    assert res.score > 0


def test_hidden_bullish_divergence():
    price = [10, 9, 8, 9, 10, 10, 9, 10, 11, 12]
    osc = [40, 38, 42, 39, 41, 37, 30, 32, 34, 36]
    res = compute_divergence(
        price_high=price,
        price_low=price,
        osc=osc,
        pivot_window=1,
        min_sep=2,
        lookback_bars=20,
        min_price_pips=1.0,
        min_osc=5.0,
        max_age_bars=10,
        pip_value=1.0,
    )
    assert res.kind == 2
    assert res.score > 0


def test_regular_bearish_divergence():
    price = [10, 11, 12, 11, 10, 11, 13, 12, 11, 10]
    osc = [50, 55, 60, 55, 50, 55, 52, 50, 48, 46]
    res = compute_divergence(
        price_high=price,
        price_low=price,
        osc=osc,
        pivot_window=1,
        min_sep=2,
        lookback_bars=20,
        min_price_pips=1.0,
        min_osc=4.0,
        max_age_bars=10,
        pip_value=1.0,
    )
    assert res.kind == -1
    assert res.score < 0
