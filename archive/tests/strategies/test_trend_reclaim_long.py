from __future__ import annotations

from strategies.scalping.trend_reclaim_long import TrendReclaimLong


def _candle(o: float, h: float, l: float, c: float) -> dict:
    return {"open": o, "high": h, "low": l, "close": c}


def _base_fac() -> dict:
    candles = [
        _candle(152.93, 152.96, 152.92, 152.95),
        _candle(152.95, 152.98, 152.94, 152.97),
        _candle(152.97, 153.00, 152.96, 152.99),
        _candle(152.99, 153.02, 152.98, 153.01),
        _candle(153.01, 153.03, 153.00, 153.02),
        _candle(153.02, 153.04, 153.01, 153.03),
        _candle(153.03, 153.04, 153.00, 153.01),
        _candle(153.01, 153.02, 152.99, 153.00),
        _candle(153.00, 153.01, 152.99, 153.00),
        _candle(153.00, 153.02, 152.99, 153.01),
        _candle(153.01, 153.01, 153.00, 153.00),
        _candle(153.00, 153.03, 153.00, 153.02),
    ]
    return {
        "close": 153.02,
        "ema10": 153.01,
        "ema20": 152.99,
        "ema50": 152.94,
        "adx": 23.0,
        "rsi": 61.0,
        "atr_pips": 1.2,
        "vol_5m": 0.9,
        "h1_close": 153.05,
        "h1_ema20": 152.99,
        "h1_adx": 25.0,
        "m5_close": 153.03,
        "m5_ema20": 152.99,
        "bb_upper": 153.08,
        "candles": candles,
    }


def test_trend_reclaim_long_emits_signal() -> None:
    fac = _base_fac()
    signal = TrendReclaimLong.check(fac)
    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tp_pips"] > signal["sl_pips"]
    assert str(signal["tag"]).startswith("TrendReclaimLong-")


def test_trend_reclaim_long_blocks_weak_h1() -> None:
    fac = _base_fac()
    fac["h1_close"] = fac["h1_ema20"]  # no H1 gap
    signal = TrendReclaimLong.check(fac)
    assert signal is None


def test_trend_reclaim_long_blocks_overheat() -> None:
    fac = _base_fac()
    fac["rsi"] = 78.0
    signal = TrendReclaimLong.check(fac)
    assert signal is None
