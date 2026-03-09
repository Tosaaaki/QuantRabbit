from __future__ import annotations

from strategies.micro.momentum_burst import MomentumBurstMicro


def test_long_signal_passes_when_indicator_quality_is_clean() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.58,
            "ma10": 158.55,
            "ma20": 158.50,
            "ema20": 158.535,
            "adx": 30.2,
            "atr_pips": 3.8,
            "vol_5m": 2.1,
            "rsi": 62.0,
            "plus_di": 28.0,
            "minus_di": 15.0,
            "roc5": 0.031,
            "ema_slope_10": 0.0014,
            "trend_snapshot": {
                "tf": "H4",
                "direction": "long",
                "gap_pips": 21.0,
                "adx": 30.0,
            },
            "candles": [
                {"high": 158.50, "low": 158.46, "close": 158.48},
                {"high": 158.54, "low": 158.49, "close": 158.52},
                {"high": 158.57, "low": 158.51, "close": 158.55},
                {"high": 158.60, "low": 158.54, "close": 158.58},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"


def test_long_rejects_overextended_indicator_state() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.621,
            "ma10": 158.584,
            "ma20": 158.548,
            "ema20": 158.54,
            "adx": 31.6,
            "atr_pips": 3.9,
            "vol_5m": 2.4,
            "rsi": 71.0,
            "plus_di": 23.0,
            "minus_di": 15.0,
            "roc5": 0.022,
            "ema_slope_10": 0.0008,
            "candles": [
                {"high": 158.470, "low": 158.452, "close": 158.461},
                {"high": 158.508, "low": 158.482, "close": 158.501},
                {"high": 158.544, "low": 158.516, "close": 158.539},
                {"high": 158.586, "low": 158.552, "close": 158.578},
                {"high": 158.624, "low": 158.594, "close": 158.621},
            ],
        }
    )

    assert signal is None


def test_long_rejects_strong_opposing_higher_tf_snapshot() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.58,
            "ma10": 158.55,
            "ma20": 158.50,
            "ema20": 158.535,
            "adx": 30.2,
            "atr_pips": 3.8,
            "vol_5m": 2.1,
            "rsi": 62.0,
            "plus_di": 28.0,
            "minus_di": 15.0,
            "roc5": 0.031,
            "ema_slope_10": 0.0014,
            "trend_snapshot": {
                "tf": "H4",
                "direction": "short",
                "gap_pips": -24.0,
                "adx": 31.0,
            },
            "candles": [
                {"high": 158.50, "low": 158.46, "close": 158.48},
                {"high": 158.54, "low": 158.49, "close": 158.52},
                {"high": 158.57, "low": 158.51, "close": 158.55},
                {"high": 158.60, "low": 158.54, "close": 158.58},
            ],
        }
    )

    assert signal is None


def test_short_reacceleration_break_can_fire_before_ma_cross() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.425,
            "ma10": 158.4623,
            "ma20": 158.4385,
            "ema20": 158.47319192374422,
            "adx": 32.39824580670669,
            "atr_pips": 4.037,
            "vol_5m": 2.86,
            "rsi": 39.294536393835585,
            "plus_di": 15.820554588018584,
            "minus_di": 27.183731946604937,
            "roc5": -0.03975064358183733,
            "ema_slope_10": -0.0013553459222743114,
            "candles": [
                {"high": 158.488, "low": 158.438, "close": 158.476},
                {"high": 158.478, "low": 158.465, "close": 158.478},
                {"high": 158.492, "low": 158.461, "close": 158.476},
                {"high": 158.476, "low": 158.422, "close": 158.425},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert signal["sl_pips"] == 5.05
    assert signal["tp_pips"] == 8.48


def test_short_reacceleration_requires_real_breakdown() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.476,
            "ma10": 158.4663,
            "ma20": 158.4411,
            "ema20": 158.47826475782256,
            "adx": 32.857852422943516,
            "atr_pips": 3.9322,
            "vol_5m": 1.94,
            "rsi": 46.36116118004843,
            "plus_di": 17.491785514424404,
            "minus_di": 22.42601517956522,
            "roc5": -0.0044168775199859844,
            "ema_slope_10": -0.001485892701959113,
            "candles": [
                {"high": 158.488, "low": 158.438, "close": 158.438},
                {"high": 158.488, "low": 158.438, "close": 158.476},
                {"high": 158.478, "low": 158.465, "close": 158.478},
                {"high": 158.492, "low": 158.461, "close": 158.476},
            ],
        }
    )

    assert signal is None


def test_short_reacceleration_allows_modest_break_after_pullback() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.434,
            "ma10": 158.458,
            "ma20": 158.447,
            "ema20": 158.455,
            "adx": 28.4,
            "atr_pips": 3.8,
            "vol_5m": 2.1,
            "rsi": 42.0,
            "plus_di": 17.0,
            "minus_di": 23.5,
            "roc5": -0.025,
            "ema_slope_10": -0.0012,
            "candles": [
                {"high": 158.488, "low": 158.456, "close": 158.466},
                {"high": 158.486, "low": 158.460, "close": 158.478},
                {"high": 158.492, "low": 158.448, "close": 158.458},
                {"high": 158.447, "low": 158.430, "close": 158.434},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"


def test_stretched_short_rejects_when_indicator_quality_is_not_strong_enough() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.421,
            "ma10": 158.447,
            "ma20": 158.459,
            "ema20": 158.455,
            "adx": 31.6,
            "atr_pips": 3.9,
            "vol_5m": 2.4,
            "rsi": 31.0,
            "plus_di": 16.0,
            "minus_di": 23.0,
            "roc5": -0.022,
            "ema_slope_10": -0.0008,
            "candles": [
                {"high": 158.565, "low": 158.548, "close": 158.552},
                {"high": 158.534, "low": 158.514, "close": 158.518},
                {"high": 158.5, "low": 158.48, "close": 158.484},
                {"high": 158.466, "low": 158.448, "close": 158.452},
                {"high": 158.43, "low": 158.418, "close": 158.421},
            ],
        }
    )

    assert signal is None


def test_stretched_short_still_allows_with_exceptional_indicator_impulse() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.421,
            "ma10": 158.447,
            "ma20": 158.459,
            "ema20": 158.455,
            "adx": 33.0,
            "atr_pips": 3.9,
            "vol_5m": 2.5,
            "rsi": 31.0,
            "plus_di": 12.0,
            "minus_di": 24.0,
            "roc5": -0.034,
            "ema_slope_10": -0.0013,
            "candles": [
                {"high": 158.565, "low": 158.548, "close": 158.552},
                {"high": 158.534, "low": 158.514, "close": 158.518},
                {"high": 158.5, "low": 158.48, "close": 158.484},
                {"high": 158.466, "low": 158.448, "close": 158.452},
                {"high": 158.43, "low": 158.418, "close": 158.421},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"


def test_stretched_long_rejects_when_indicator_quality_is_not_strong_enough() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.619,
            "ma10": 158.603,
            "ma20": 158.592,
            "ema20": 158.588,
            "adx": 31.2,
            "atr_pips": 3.8,
            "vol_5m": 2.3,
            "rsi": 69.0,
            "plus_di": 24.0,
            "minus_di": 17.0,
            "roc5": 0.022,
            "ema_slope_10": 0.0008,
            "candles": [
                {"high": 158.592, "low": 158.556, "close": 158.57},
                {"high": 158.602, "low": 158.568, "close": 158.585},
                {"high": 158.612, "low": 158.578, "close": 158.596},
                {"high": 158.625, "low": 158.592, "close": 158.619},
            ],
        }
    )

    assert signal is None


def test_stretched_long_still_allows_with_exceptional_indicator_impulse() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.619,
            "ma10": 158.603,
            "ma20": 158.592,
            "ema20": 158.588,
            "adx": 33.4,
            "atr_pips": 3.8,
            "vol_5m": 2.4,
            "rsi": 69.0,
            "plus_di": 28.0,
            "minus_di": 14.0,
            "roc5": 0.034,
            "ema_slope_10": 0.0013,
            "candles": [
                {"high": 158.592, "low": 158.556, "close": 158.57},
                {"high": 158.602, "low": 158.568, "close": 158.585},
                {"high": 158.612, "low": 158.578, "close": 158.596},
                {"high": 158.625, "low": 158.592, "close": 158.619},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"


def test_short_price_action_allows_one_noisy_bar() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.53,
            "ma10": 158.5,
            "ma20": 158.57,
            "ema20": 158.545,
            "adx": 30.4,
            "atr_pips": 3.7,
            "vol_5m": 1.8,
            "rsi": 31.0,
            "plus_di": 14.0,
            "minus_di": 23.0,
            "roc5": -0.018,
            "ema_slope_10": -0.0018,
            "candles": [
                {"high": 158.62, "low": 158.58, "close": 158.60},
                {"high": 158.60, "low": 158.55, "close": 158.57},
                {"high": 158.59, "low": 158.51, "close": 158.54},
                {"high": 158.57, "low": 158.52, "close": 158.53},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"


def test_short_rejects_oversold_indicator_exhaustion() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.53,
            "ma10": 158.5,
            "ma20": 158.57,
            "ema20": 158.545,
            "adx": 30.4,
            "atr_pips": 3.7,
            "vol_5m": 1.8,
            "rsi": 28.0,
            "plus_di": 14.0,
            "minus_di": 23.0,
            "roc5": -0.018,
            "ema_slope_10": -0.0018,
            "candles": [
                {"high": 158.62, "low": 158.58, "close": 158.60},
                {"high": 158.60, "low": 158.55, "close": 158.57},
                {"high": 158.59, "low": 158.51, "close": 158.54},
                {"high": 158.57, "low": 158.52, "close": 158.53},
            ],
        }
    )

    assert signal is None


def test_short_price_action_still_rejects_when_majority_is_not_directional() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.557,
            "ma10": 158.53,
            "ma20": 158.6,
            "ema20": 158.575,
            "adx": 29.8,
            "atr_pips": 3.6,
            "vol_5m": 1.7,
            "rsi": 31.0,
            "plus_di": 14.0,
            "minus_di": 19.0,
            "roc5": -0.028,
            "ema_slope_10": -0.0015,
            "candles": [
                {"high": 158.62, "low": 158.57, "close": 158.6},
                {"high": 158.6, "low": 158.55, "close": 158.58},
                {"high": 158.61, "low": 158.56, "close": 158.59},
                {"high": 158.63, "low": 158.55, "close": 158.55},
            ],
        }
    )

    assert signal is None


def test_short_rejects_strong_opposing_higher_tf_snapshot() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.434,
            "ma10": 158.458,
            "ma20": 158.447,
            "ema20": 158.455,
            "adx": 28.4,
            "atr_pips": 3.8,
            "vol_5m": 2.1,
            "rsi": 42.0,
            "plus_di": 17.0,
            "minus_di": 23.5,
            "roc5": -0.025,
            "ema_slope_10": -0.0012,
            "trend_snapshot": {
                "tf": "H4",
                "direction": "long",
                "gap_pips": 33.692,
                "adx": 31.52,
            },
            "candles": [
                {"high": 158.488, "low": 158.456, "close": 158.466},
                {"high": 158.486, "low": 158.460, "close": 158.478},
                {"high": 158.492, "low": 158.448, "close": 158.458},
                {"high": 158.447, "low": 158.430, "close": 158.434},
            ],
        }
    )

    assert signal is None


def test_context_tilt_reduces_confidence_in_chop_without_reaccel() -> None:
    base_fac = {
        "close": 158.53,
        "ma10": 158.5,
        "ma20": 158.57,
        "ema20": 158.545,
        "adx": 30.4,
        "atr_pips": 3.7,
        "vol_5m": 1.8,
        "rsi": 31.0,
        "plus_di": 14.0,
        "minus_di": 23.0,
        "roc5": -0.018,
        "ema_slope_10": -0.0018,
        "candles": [
            {"high": 158.62, "low": 158.58, "close": 158.60},
            {"high": 158.60, "low": 158.55, "close": 158.57},
            {"high": 158.59, "low": 158.51, "close": 158.54},
            {"high": 158.57, "low": 158.52, "close": 158.53},
        ],
    }
    plain = MomentumBurstMicro.check(dict(base_fac))
    tilted = MomentumBurstMicro.check(
        {
            **base_fac,
            "range_score": 0.52,
            "micro_chop_active": True,
            "micro_chop_score": 0.72,
        }
    )

    assert plain is not None
    assert tilted is not None
    assert tilted["action"] == "OPEN_SHORT"
    assert tilted["confidence"] < plain["confidence"]
    assert tilted["entry_probability"] < plain["confidence"] / 100.0
    assert tilted["notes"]["context_tilt"]["chop_active"] is True


def test_strong_chop_range_context_blocks_non_reaccel_signal() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.53,
            "ma10": 158.5,
            "ma20": 158.57,
            "ema20": 158.545,
            "adx": 30.4,
            "atr_pips": 3.7,
            "vol_5m": 1.8,
            "rsi": 31.0,
            "plus_di": 14.0,
            "minus_di": 23.0,
            "roc5": -0.018,
            "ema_slope_10": -0.0018,
            "range_active": True,
            "range_score": 0.95,
            "micro_chop_active": True,
            "micro_chop_score": 0.96,
            "candles": [
                {"high": 158.62, "low": 158.58, "close": 158.60},
                {"high": 158.60, "low": 158.55, "close": 158.57},
                {"high": 158.59, "low": 158.51, "close": 158.54},
                {"high": 158.57, "low": 158.52, "close": 158.53},
            ],
        }
    )

    assert signal is None


def test_reacceleration_still_passes_even_when_chop_is_active() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.425,
            "ma10": 158.4623,
            "ma20": 158.4385,
            "ema20": 158.47319192374422,
            "adx": 32.39824580670669,
            "atr_pips": 4.037,
            "vol_5m": 2.86,
            "rsi": 39.294536393835585,
            "plus_di": 15.820554588018584,
            "minus_di": 27.183731946604937,
            "roc5": -0.03975064358183733,
            "ema_slope_10": -0.0013553459222743114,
            "range_active": True,
            "range_score": 0.92,
            "micro_chop_active": True,
            "micro_chop_score": 0.96,
            "candles": [
                {"high": 158.488, "low": 158.438, "close": 158.476},
                {"high": 158.478, "low": 158.465, "close": 158.478},
                {"high": 158.492, "low": 158.461, "close": 158.476},
                {"high": 158.476, "low": 158.422, "close": 158.425},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
