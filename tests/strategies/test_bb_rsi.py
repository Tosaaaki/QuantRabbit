from __future__ import annotations

from strategies.mean_reversion.bb_rsi import BBRsi


def test_bb_rsi_sl_tp_ratio_expanded():
    fac = {
        "rsi": 25,
        "bbw": 0.002,
        "ma20": 153.0,
        "ma10": 153.001,
        "close": 152.82,
        "atr_pips": 1.5,
        "adx": 18.0,
        "vol_5m": 1.2,
        "rsi_eta_upper_min": 4.0,
        "bbw_squeeze_eta_min": 3.0,
    }
    signal = BBRsi.check(fac)
    assert signal is not None
    assert signal["sl_pips"] >= 1.0
    assert signal["tp_pips"] >= signal["sl_pips"] * 1.4
    assert signal["profile"] == "bb_range_reversion"
    assert signal["target_tp_pips"] == signal["tp_pips"]


def test_bb_rsi_skips_trending_env():
    fac = {
        "rsi": 25,
        "bbw": 0.002,
        "ma20": 153.0,
        "ma10": 153.2,
        "close": 152.90,
        "atr_pips": 2.0,
        "adx": 28.0,
        "vol_5m": 0.9,
    }
    signal = BBRsi.check(fac)
    assert signal is None
