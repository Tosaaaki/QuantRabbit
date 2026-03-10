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


def test_bounce_long_rejects_countertrend_probe_without_lower_wick() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.93,
            "open": 157.91,
            "high": 157.93,
            "low": 157.91,
            "ma10": 157.97,
            "ma20": 157.98,
            "ema20": 158.00,
            "atr_pips": 2.2,
            "rsi": 45.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is None


def test_bounce_long_allows_countertrend_reclaim_with_clear_lower_wick() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.935,
            "open": 157.918,
            "high": 157.939,
            "low": 157.902,
            "ma10": 157.97,
            "ma20": 157.98,
            "ema20": 158.00,
            "atr_pips": 2.2,
            "rsi": 45.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "MicroLevelReactor-bounce-lower"


def test_bounce_long_keeps_body_only_reclaim_when_local_trend_is_not_down() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.93,
            "open": 157.91,
            "high": 157.93,
            "low": 157.91,
            "ma10": 157.98,
            "ma20": 157.979,
            "ema20": 158.00,
            "atr_pips": 2.2,
            "rsi": 45.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "MicroLevelReactor-bounce-lower"


def test_bounce_long_rejects_tiny_lower_wick_under_strong_down_di_pressure() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.956,
            "open": 157.951,
            "high": 157.956,
            "low": 157.95,
            "ma10": 157.981,
            "ma20": 157.98,
            "ema20": 158.00,
            "atr_pips": 1.6,
            "rsi": 35.0,
            "adx": 23.0,
            "plus_di": 14.0,
            "minus_di": 37.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is None


def test_bounce_long_allows_clear_lower_wick_under_same_down_di_pressure() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.956,
            "open": 157.943,
            "high": 157.957,
            "low": 157.938,
            "ma10": 157.981,
            "ma20": 157.98,
            "ema20": 158.00,
            "atr_pips": 1.6,
            "rsi": 35.0,
            "adx": 23.0,
            "plus_di": 14.0,
            "minus_di": 37.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "MicroLevelReactor-bounce-lower"


def test_bounce_long_rejects_small_reclaim_body_under_strong_continuation_pressure() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.956,
            "open": 157.95,
            "high": 157.957,
            "low": 157.944,
            "ma10": 157.981,
            "ma20": 157.98,
            "ema20": 158.00,
            "atr_pips": 1.5,
            "rsi": 35.0,
            "adx": 24.0,
            "plus_di": 12.0,
            "minus_di": 38.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is None


def test_bounce_long_keeps_tiny_lower_wick_when_di_pressure_is_not_strong() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.956,
            "open": 157.951,
            "high": 157.956,
            "low": 157.95,
            "ma10": 157.981,
            "ma20": 157.98,
            "ema20": 158.00,
            "atr_pips": 1.6,
            "rsi": 35.0,
            "adx": 19.0,
            "plus_di": 18.0,
            "minus_di": 27.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "MicroLevelReactor-bounce-lower"


def test_bounce_long_rejects_clear_wick_when_strong_down_continuation_keeps_wide_ma_gap() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.956,
            "open": 157.95,
            "high": 157.957,
            "low": 157.936,
            "ma10": 157.95,
            "ma20": 157.98,
            "ema20": 158.00,
            "atr_pips": 1.6,
            "rsi": 35.0,
            "adx": 37.0,
            "plus_di": 12.0,
            "minus_di": 40.0,
            "spread_pips": 0.8,
        }
    )

    assert signal is None


def test_bounce_long_rejects_recent_three_bar_selloff_without_clear_reclaim() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.956,
            "open": 157.952,
            "high": 157.958,
            "low": 157.946,
            "ema20": 158.00,
            "atr_pips": 1.5,
            "rsi": 35.0,
            "spread_pips": 0.8,
            "candles": [
                {"open": 157.984, "high": 157.985, "low": 157.978, "close": 157.980},
                {"open": 157.980, "high": 157.982, "low": 157.968, "close": 157.970},
                {"open": 157.968, "high": 157.972, "low": 157.954, "close": 157.958},
                {"open": 157.952, "high": 157.958, "low": 157.946, "close": 157.956},
            ],
        }
    )

    assert signal is None


def test_bounce_long_allows_recent_three_bar_selloff_with_strong_body_reclaim() -> None:
    signal = MicroLevelReactor.check(
        {
            "close": 157.956,
            "open": 157.944,
            "high": 157.957,
            "low": 157.938,
            "ema20": 158.00,
            "atr_pips": 1.5,
            "rsi": 35.0,
            "spread_pips": 0.8,
            "candles": [
                {"open": 157.984, "high": 157.985, "low": 157.978, "close": 157.980},
                {"open": 157.980, "high": 157.982, "low": 157.968, "close": 157.970},
                {"open": 157.968, "high": 157.972, "low": 157.954, "close": 157.958},
                {"open": 157.944, "high": 157.957, "low": 157.938, "close": 157.956},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["tag"] == "MicroLevelReactor-bounce-lower"
