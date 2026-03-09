from __future__ import annotations

from strategies.micro.level_reactor import MicroLevelReactor


def test_breakout_long_rejects_bearish_breakout_candle() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 158.09,
            "open": 158.11,
            "high": 158.10,
            "low": 158.089,
            "ema20": 158.00,
            "atr_pips": 3.2,
            "rsi": 58.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is None


def test_breakout_long_allows_lower_wick_rejection() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 158.09,
            "open": 158.07,
            "high": 158.10,
            "low": 158.05,
            "ema20": 158.00,
            "atr_pips": 3.2,
            "rsi": 58.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "MicroLevelReactor-breakout-long"


def test_bounce_long_rejects_when_candle_is_still_bearish() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.93,
            "open": 157.95,
            "high": 157.96,
            "low": 157.92,
            "ema20": 158.00,
            "atr_pips": 3.2,
            "rsi": 45.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is None
