from __future__ import annotations

from strategies.scalping.impulse_retrace import ImpulseRetraceScalp
from strategies.scalping.range_fader import RangeFader


def test_impulse_retrace_relaxes_atr_on_low_spread():
    fac = {
        "close": 156.50,
        "ema20": 156.52,
        "ema10": 156.53,
        "rsi": 35,
        "atr_pips": 0.72,
        "spread_pips": 0.8,
        "vol_5m": 0.95,
        "momentum": -0.012,
    }
    signal = ImpulseRetraceScalp.check(fac)
    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["sl_pips"] > 0
    assert signal["tp_pips"] > 0


def test_range_fader_allows_low_atr_with_good_spread():
    fac = {
        "close": 156.50,
        "ema20": 156.52,
        "rsi": 42,
        "atr_pips": 0.72,
        "vol_5m": 1.05,
        "adx": 18.0,
        "bbw": 0.22,
        "bbw_squeeze_eta_min": 4.0,
        "spread_pips": 0.8,
    }
    signal = RangeFader.check(fac)
    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["sl_pips"] > 0


def test_range_fader_buy_supportive_context_extends_long_gate() -> None:
    signal = RangeFader.check(
        {
            "close": 156.50,
            "ema20": 156.515,
            "rsi": 46.5,
            "atr_pips": 1.9,
            "vol_5m": 1.15,
            "adx": 24.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "plus_di": 21.0,
            "minus_di": 19.5,
            "ema_slope_10": -0.0004,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "RangeFader-buy-supportive"
    assert signal["confidence"] == 51


def test_range_fader_buy_supportive_context_without_cluster_falls_back_to_neutral() -> None:
    fac = {
        "close": 156.50,
        "ema20": 156.515,
        "rsi": 46.5,
        "atr_pips": 1.9,
        "vol_5m": 1.15,
        "adx": 24.0,
        "bbw": 0.20,
        "bbw_squeeze_eta_min": 4.0,
        "spread_pips": 0.8,
        "plus_di": 16.0,
        "minus_di": 22.0,
        "ema_slope_10": -0.0016,
    }

    signal = RangeFader.check(fac)

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "RangeFader-neutral-fade"


def test_range_fader_short_headwind_blocks_weak_short_fade() -> None:
    signal = RangeFader.check(
        {
            "close": 157.93,
            "ema20": 157.90,
            "rsi": 60.0,
            "atr_pips": 2.9,
            "vol_5m": 1.30,
            "adx": 37.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "plus_di": 28.0,
            "minus_di": 14.0,
            "ema_slope_10": 0.0060,
            "range_score": 0.24,
        }
    )

    assert signal is None


def test_range_fader_extreme_short_stretch_still_allows_short_fade() -> None:
    signal = RangeFader.check(
        {
            "close": 157.96,
            "ema20": 157.90,
            "rsi": 68.0,
            "atr_pips": 2.9,
            "vol_5m": 1.30,
            "adx": 37.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "plus_di": 28.0,
            "minus_di": 14.0,
            "ema_slope_10": 0.0060,
            "range_score": 0.24,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert signal["tag"] == "RangeFader-sell-fade"


def test_range_fader_long_headwind_blocks_weak_long_fade() -> None:
    signal = RangeFader.check(
        {
            "close": 157.87,
            "ema20": 157.90,
            "rsi": 40.0,
            "atr_pips": 2.9,
            "vol_5m": 1.30,
            "adx": 37.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "plus_di": 14.0,
            "minus_di": 28.0,
            "ema_slope_10": -0.0060,
            "range_score": 0.24,
        }
    )

    assert signal is None
