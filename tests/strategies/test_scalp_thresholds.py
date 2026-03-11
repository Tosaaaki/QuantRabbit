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


def test_range_fader_blocks_shallow_buy_fade_probe_in_range_regime() -> None:
    signal = RangeFader.check(
        {
            "close": 156.500,
            "ema20": 156.515,
            "rsi": 43.8,
            "atr_pips": 1.9,
            "vol_5m": 1.05,
            "adx": 18.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "range_score": 0.31,
            "plus_di": 16.0,
            "minus_di": 20.0,
            "ema_slope_10": -0.0007,
        }
    )

    assert signal is None


def test_range_fader_blocks_shallow_neutral_fade_probe_in_range_regime() -> None:
    signal = RangeFader.check(
        {
            "close": 156.500,
            "ema20": 156.514,
            "rsi": 49.5,
            "atr_pips": 1.9,
            "vol_5m": 1.05,
            "adx": 18.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "range_score": 0.31,
            "plus_di": 17.0,
            "minus_di": 18.0,
            "ema_slope_10": -0.0004,
        }
    )

    assert signal is None


def test_range_fader_buy_supportive_survives_shallow_probe_guard_in_range_regime() -> None:
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
            "range_score": 0.31,
            "plus_di": 21.0,
            "minus_di": 19.5,
            "ema_slope_10": -0.0004,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "RangeFader-buy-supportive"
    assert signal["flow_regime"] == "range_fade"
    assert signal["setup_fingerprint"] == "RangeFader|long|buy-supportive|range_fade|p0"


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


def test_range_fader_blocks_fragile_transition_short_sell_fade() -> None:
    signal = RangeFader.check(
        {
            "close": 157.93,
            "ema20": 157.915,
            "ma10": 157.929,
            "ma20": 157.918,
            "rsi": 56.4,
            "atr_pips": 2.4,
            "vol_5m": 1.08,
            "adx": 20.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "plus_di": 22.0,
            "minus_di": 18.0,
            "ema_slope_10": 0.0008,
            "range_score": 0.20,
        }
    )

    assert signal is None


def test_range_fader_blocks_fragile_range_p1_short_sell_fade() -> None:
    signal = RangeFader.check(
        {
            "close": 157.94,
            "ema20": 157.92,
            "ma10": 157.937,
            "ma20": 157.923,
            "rsi": 56.1,
            "atr_pips": 2.4,
            "vol_5m": 1.05,
            "adx": 25.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "plus_di": 22.0,
            "minus_di": 18.0,
            "ema_slope_10": 0.0010,
            "range_score": 0.31,
        }
    )

    assert signal is None


def test_range_fader_extreme_short_stretch_still_allows_short_fade() -> None:
    signal = RangeFader.check(
        {
            "close": 157.96,
            "ema20": 157.90,
            "ma10": 157.94,
            "ma20": 157.90,
            "rsi": 73.0,
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
            "candles": [
                {"open": 157.88, "close": 157.90, "high": 157.91, "low": 157.87},
                {"open": 157.90, "close": 157.92, "high": 157.93, "low": 157.89},
                {"open": 157.92, "close": 157.94, "high": 157.95, "low": 157.91},
                {"open": 157.94, "close": 157.96, "high": 157.97, "low": 157.93},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert signal["tag"] == "RangeFader-sell-fade"
    assert signal["continuation_pressure"] == 2
    assert signal["flow_regime"] == "trend_long"
    assert 0.0 <= float(signal["setup_quality"]) <= 1.0
    assert 0.55 <= float(signal["setup_size_mult"]) <= 1.10
    assert signal["setup_fingerprint"] == "RangeFader|short|sell-fade|trend_long|p2"


def test_range_fader_blocks_fragile_neutral_long_range_probe() -> None:
    signal = RangeFader.check(
        {
            "close": 157.89,
            "ema20": 157.902,
            "ma10": 157.894,
            "ma20": 157.899,
            "rsi": 49.4,
            "atr_pips": 2.2,
            "vol_5m": 1.04,
            "adx": 19.0,
            "bbw": 0.20,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "plus_di": 18.0,
            "minus_di": 20.0,
            "ema_slope_10": -0.0004,
            "range_score": 0.48,
        }
    )

    assert signal is None


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


def test_range_fader_extreme_long_stretch_still_allows_long_fade() -> None:
    signal = RangeFader.check(
        {
            "close": 157.84,
            "ema20": 157.90,
            "ma10": 157.86,
            "ma20": 157.90,
            "rsi": 27.0,
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
            "candles": [
                {"open": 157.92, "close": 157.90, "high": 157.93, "low": 157.89},
                {"open": 157.90, "close": 157.88, "high": 157.91, "low": 157.87},
                {"open": 157.88, "close": 157.86, "high": 157.89, "low": 157.85},
                {"open": 157.86, "close": 157.84, "high": 157.87, "low": 157.83},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "RangeFader-buy-fade"
    assert signal["continuation_pressure"] == 2
    assert signal["flow_regime"] == "trend_short"
    assert signal["setup_fingerprint"] == "RangeFader|long|buy-fade|trend_short|p2"


def test_range_fader_blocks_thin_neutral_fade_setup() -> None:
    signal = RangeFader.check(
        {
            "close": 156.507,
            "ema20": 156.500,
            "rsi": 54.0,
            "atr_pips": 2.4,
            "vol_5m": 1.0,
            "adx": 21.0,
            "bbw": 0.18,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "range_score": 0.22,
            "plus_di": 18.0,
            "minus_di": 17.0,
            "ema_slope_10": 0.0005,
        }
    )

    assert signal is None


def test_range_fader_blocks_fragile_neutral_short_range_lane() -> None:
    signal = RangeFader.check(
        {
            "close": 156.5079,
            "ema20": 156.5000,
            "ma10": 156.5079,
            "ma20": 156.5000,
            "rsi": 54.0,
            "atr_pips": 1.63,
            "vol_5m": 1.0,
            "adx": 18.0,
            "bbw": 0.18,
            "bbw_squeeze_eta_min": 4.0,
            "spread_pips": 0.8,
            "range_score": 0.486,
            "plus_di": 18.0,
            "minus_di": 17.0,
            "ema_slope_10": 0.0004,
        }
    )

    assert signal is None
