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
