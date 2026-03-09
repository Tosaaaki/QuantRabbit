from __future__ import annotations

from strategies.micro.momentum_burst import MomentumBurstMicro


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
