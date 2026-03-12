from __future__ import annotations

import importlib

import strategies.micro.momentum_burst as momentum_burst_module
from strategies.micro.momentum_burst import MomentumBurstMicro
from workers.micro_runtime import worker as micro_runtime_worker


def _synthetic_candles(*, start: float, drift: float, n: int = 40) -> list[dict[str, object]]:
    candles: list[dict[str, object]] = []
    close = start
    for idx in range(n):
        open_ = close
        close = start + drift * float(idx + 1)
        high = max(open_, close) + 0.01
        low = min(open_, close) - 0.01
        candles.append(
            {
                "timestamp": f"2026-01-01T00:{idx:02d}:00+00:00",
                "open": round(open_, 3),
                "high": round(high, 3),
                "low": round(low, 3),
                "close": round(close, 3),
            }
        )
    return candles


def _softly_contra_transition_long_fixture(*, projection_score: float) -> dict:
    return {
        "close": 158.582,
        "ma10": 158.556,
        "ma20": 158.528,
        "ema20": 158.544,
        "adx": 19.6,
        "atr_pips": 4.0,
        "vol_5m": 1.9,
        "rsi": 52.6,
        "plus_di": 22.5,
        "minus_di": 15.8,
        "roc5": 0.026,
        "ema_slope_10": 0.0012,
        "range_score": 0.24,
        "micro_chop_score": 0.54,
        "projection": {"score": projection_score},
        "trend_snapshot": {
            "tf": "H4",
            "direction": "short",
            "gap_pips": -7.0,
            "adx": 19.0,
        },
        "candles": [
            {"high": 158.53, "low": 158.49, "close": 158.50},
            {"high": 158.55, "low": 158.50, "close": 158.53},
            {"high": 158.586, "low": 158.52, "close": 158.55},
            {"high": 158.584, "low": 158.548, "close": 158.582},
        ],
    }


def _h4_tiebreak_transition_long_fixture(
    *,
    h1_gap_pips: float = -2.8,
    h1_adx: float = 16.0,
    h4_direction: str = "long",
) -> dict:
    h4_is_long = h4_direction == "long"
    return {
        "close": 158.582,
        "ma10": 158.556,
        "ma20": 158.528,
        "ema20": 158.544,
        "adx": 19.6,
        "atr_pips": 4.0,
        "vol_5m": 1.9,
        "rsi": 52.6,
        "plus_di": 22.5,
        "minus_di": 15.8,
        "roc5": 0.026,
        "ema_slope_10": 0.0012,
        "range_score": 0.24,
        "micro_chop_score": 0.54,
        "trend_snapshot": {
            "tf": "H4",
            "direction": h4_direction,
            "gap_pips": 18.0 if h4_is_long else -18.0,
            "adx": 19.0,
        },
        "mtf": {
            "candles_m5": _synthetic_candles(start=158.10, drift=0.018),
            "candles_h1": _synthetic_candles(start=158.80, drift=-0.014),
            "candles_h4": _synthetic_candles(start=157.90, drift=0.060 if h4_is_long else -0.060),
        },
        "mtf_context": {
            "m5": {"gap_pips": 3.6, "adx": 23.7, "direction": "long"},
            "h1": {"gap_pips": h1_gap_pips, "adx": h1_adx, "direction": "short"},
            "h4": {
                "gap_pips": 35.2 if h4_is_long else -35.2,
                "adx": 19.2,
                "direction": h4_direction,
            },
        },
        "candles": [
            {"high": 158.53, "low": 158.49, "close": 158.50},
            {"high": 158.55, "low": 158.50, "close": 158.53},
            {"high": 158.586, "low": 158.52, "close": 158.55},
            {"high": 158.584, "low": 158.548, "close": 158.582},
        ],
    }


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
    assert signal["sl_pips"] == 4.37
    assert signal["tp_pips"] == 7.6


def test_transition_long_allows_mid_rsi_when_higher_tf_impulse_is_strong() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.582,
            "ma10": 158.556,
            "ma20": 158.528,
            "ema20": 158.544,
            "adx": 19.6,
            "atr_pips": 4.0,
            "vol_5m": 1.9,
            "rsi": 52.6,
            "plus_di": 22.5,
            "minus_di": 15.8,
            "roc5": 0.026,
            "ema_slope_10": 0.0012,
            "range_score": 0.24,
            "micro_chop_score": 0.54,
            "trend_snapshot": {
                "tf": "H4",
                "direction": "long",
                "gap_pips": 18.0,
                "adx": 26.0,
            },
            "candles": [
                {"high": 158.53, "low": 158.49, "close": 158.50},
                {"high": 158.55, "low": 158.50, "close": 158.53},
                {"high": 158.586, "low": 158.52, "close": 158.55},
                {"high": 158.584, "low": 158.548, "close": 158.582},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"


def test_transition_long_keeps_rsi_floor_in_range_chop_context() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.582,
            "ma10": 158.556,
            "ma20": 158.528,
            "ema20": 158.544,
            "adx": 19.6,
            "atr_pips": 4.0,
            "vol_5m": 1.9,
            "rsi": 52.6,
            "plus_di": 22.5,
            "minus_di": 15.8,
            "roc5": 0.026,
            "ema_slope_10": 0.0012,
            "range_active": True,
            "range_score": 0.52,
            "micro_chop_active": True,
            "micro_chop_score": 0.72,
            "trend_snapshot": {
                "tf": "H4",
                "direction": "long",
                "gap_pips": 18.0,
                "adx": 26.0,
            },
            "candles": [
                {"high": 158.53, "low": 158.49, "close": 158.50},
                {"high": 158.55, "low": 158.50, "close": 158.53},
                {"high": 158.586, "low": 158.52, "close": 158.55},
                {"high": 158.584, "low": 158.548, "close": 158.582},
            ],
        }
    )

    assert signal is None


def test_transition_long_keeps_rsi_floor_under_softly_contra_snapshot_without_projection_override() -> None:
    module = importlib.reload(momentum_burst_module)

    signal = module.MomentumBurstMicro.check(
        _softly_contra_transition_long_fixture(projection_score=0.22)
    )

    assert signal is None


def test_transition_long_allows_projection_backed_mid_rsi_under_softly_contra_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOMENTUMBURST_TRANSITION_LONG_PROJECTION_SCORE_MIN", "0.18")
    module = importlib.reload(momentum_burst_module)

    try:
        signal = module.MomentumBurstMicro.check(
            _softly_contra_transition_long_fixture(projection_score=0.22)
        )
        assert signal is not None
        assert signal["action"] == "OPEN_LONG"
    finally:
        monkeypatch.delenv("MOMENTUMBURST_TRANSITION_LONG_PROJECTION_SCORE_MIN", raising=False)
        importlib.reload(momentum_burst_module)


def test_transition_long_rejects_weak_projection_override_under_softly_contra_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MOMENTUMBURST_TRANSITION_LONG_PROJECTION_SCORE_MIN", "0.18")
    module = importlib.reload(momentum_burst_module)

    try:
        signal = module.MomentumBurstMicro.check(
            _softly_contra_transition_long_fixture(projection_score=0.12)
        )
        assert signal is None
    finally:
        monkeypatch.delenv("MOMENTUMBURST_TRANSITION_LONG_PROJECTION_SCORE_MIN", raising=False)
        importlib.reload(momentum_burst_module)


def test_transition_long_allows_h4_tiebreak_when_h1_countertrend_is_shallow() -> None:
    signal = MomentumBurstMicro.check(_h4_tiebreak_transition_long_fixture())

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"


def test_transition_long_keeps_h1_countertrend_block_when_h1_headwind_is_strong() -> None:
    signal = MomentumBurstMicro.check(
        _h4_tiebreak_transition_long_fixture(h1_gap_pips=-6.2, h1_adx=20.0)
    )

    assert signal is None


def test_transition_long_keeps_h1_countertrend_block_when_h4_disagrees() -> None:
    signal = MomentumBurstMicro.check(
        _h4_tiebreak_transition_long_fixture(h4_direction="short")
    )

    assert signal is None


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


def test_long_bull_run_allows_strong_followthrough_before_extreme_impulse_threshold() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.590,
            "ma10": 158.562,
            "ma20": 158.530,
            "ema20": 158.545,
            "adx": 21.0,
            "atr_pips": 4.0,
            "vol_5m": 1.8,
            "rsi": 69.4,
            "plus_di": 25.0,
            "minus_di": 15.8,
            "roc5": 0.025,
            "ema_slope_10": 0.0009,
            "range_score": 0.22,
            "micro_chop_score": 0.50,
            "trend_snapshot": {
                "tf": "H4",
                "direction": "long",
                "gap_pips": 16.0,
                "adx": 24.0,
            },
            "candles": [
                {"high": 158.51, "low": 158.48, "close": 158.49},
                {"high": 158.54, "low": 158.50, "close": 158.525},
                {"high": 158.57, "low": 158.53, "close": 158.558},
                {"high": 158.592, "low": 158.566, "close": 158.590},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"


def test_long_bull_run_keeps_block_when_context_is_choppy() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.590,
            "ma10": 158.562,
            "ma20": 158.530,
            "ema20": 158.545,
            "adx": 21.0,
            "atr_pips": 4.0,
            "vol_5m": 1.8,
            "rsi": 69.4,
            "plus_di": 25.0,
            "minus_di": 15.8,
            "roc5": 0.025,
            "ema_slope_10": 0.0009,
            "range_score": 0.32,
            "micro_chop_score": 0.61,
            "trend_snapshot": {
                "tf": "H4",
                "direction": "long",
                "gap_pips": 16.0,
                "adx": 24.0,
            },
            "candles": [
                {"high": 158.51, "low": 158.48, "close": 158.49},
                {"high": 158.54, "low": 158.50, "close": 158.525},
                {"high": 158.57, "low": 158.53, "close": 158.558},
                {"high": 158.592, "low": 158.566, "close": 158.590},
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
    assert signal["sl_pips"] == 4.64
    assert signal["tp_pips"] == 8.07
    assert signal["notes"]["momentum_burst"] == {
        "direction": "short",
        "entry_mode": "reaccel",
        "reaccel": True,
    }
    assert signal["metadata"]["momentum_burst"] == signal["notes"]["momentum_burst"]


def test_entry_sl_keeps_floor_when_atr_is_small() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.58,
            "ma10": 158.55,
            "ma20": 158.50,
            "ema20": 158.535,
            "adx": 30.2,
            "atr_pips": 1.95,
            "vol_5m": 2.1,
            "rsi": 62.0,
            "plus_di": 28.0,
            "minus_di": 15.0,
            "roc5": 0.031,
            "ema_slope_10": 0.0014,
            "candles": [
                {"high": 158.50, "low": 158.46, "close": 158.48},
                {"high": 158.54, "low": 158.49, "close": 158.52},
                {"high": 158.57, "low": 158.51, "close": 158.55},
                {"high": 158.60, "low": 158.54, "close": 158.58},
            ],
        }
    )

    assert signal is not None
    assert signal["sl_pips"] == 2.4
    assert signal["tp_pips"] == 4.06


def test_momentumburst_reaccel_shortens_only_reaccel_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(micro_runtime_worker.config, "STRATEGY_COOLDOWN_SEC", 90.0)
    monkeypatch.setattr(
        micro_runtime_worker.config,
        "MOMENTUMBURST_REACCEL_COOLDOWN_SEC",
        45.0,
    )
    monkeypatch.setattr(micro_runtime_worker.config, "DYN_ALLOC_ENABLED", False)
    monkeypatch.setattr(
        micro_runtime_worker,
        "_STRATEGY_PARTICIPATION_ALLOC_ENABLED",
        False,
        raising=False,
    )
    micro_runtime_worker._STRATEGY_LAST_TS.clear()
    micro_runtime_worker._STRATEGY_LAST_TS["MomentumBurst"] = 100.0

    reaccel_signal = {
        "notes": {
            "momentum_burst": {
                "direction": "short",
                "entry_mode": "reaccel",
                "reaccel": True,
            }
        }
    }
    normal_signal = {
        "notes": {
            "momentum_burst": {
                "direction": "short",
                "entry_mode": "trend",
                "reaccel": False,
            }
        }
    }

    assert micro_runtime_worker._strategy_cooldown_active(
        "MomentumBurst",
        130.0,
        reaccel_signal,
    )
    assert not micro_runtime_worker._strategy_cooldown_active(
        "MomentumBurst",
        146.0,
        reaccel_signal,
    )
    assert micro_runtime_worker._strategy_cooldown_active(
        "MomentumBurst",
        146.0,
        normal_signal,
    )


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
            "rsi": 40.0,
            "plus_di": 17.0,
            "minus_di": 24.5,
            "roc5": -0.029,
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


def test_long_reacceleration_rejects_flat_upper_wick_breakout() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.582,
            "ma10": 158.555,
            "ma20": 158.548,
            "ema20": 158.554,
            "adx": 30.5,
            "atr_pips": 3.6,
            "vol_5m": 2.2,
            "rsi": 62.0,
            "plus_di": 28.0,
            "minus_di": 16.0,
            "roc5": 0.032,
            "ema_slope_10": 0.0013,
            "candles": [
                {"open": 158.53, "high": 158.55, "low": 158.52, "close": 158.545},
                {"open": 158.545, "high": 158.56, "low": 158.535, "close": 158.552},
                {"open": 158.552, "high": 158.57, "low": 158.545, "close": 158.562},
                {"open": 158.578, "high": 158.608, "low": 158.572, "close": 158.582},
            ],
        }
    )

    assert signal is None


def test_long_reacceleration_allows_clean_followthrough_breakout() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.584,
            "ma10": 158.556,
            "ma20": 158.548,
            "ema20": 158.554,
            "adx": 30.8,
            "atr_pips": 3.6,
            "vol_5m": 2.2,
            "rsi": 62.0,
            "plus_di": 28.0,
            "minus_di": 16.0,
            "roc5": 0.032,
            "ema_slope_10": 0.0013,
            "candles": [
                {"open": 158.53, "high": 158.55, "low": 158.52, "close": 158.545},
                {"open": 158.545, "high": 158.56, "low": 158.535, "close": 158.552},
                {"open": 158.552, "high": 158.57, "low": 158.545, "close": 158.562},
                {"open": 158.568, "high": 158.587, "low": 158.566, "close": 158.584},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_LONG"
    assert signal["notes"]["momentum_burst"]["reaccel"] is True


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
            "rsi": 36.0,
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
            "rsi": 36.0,
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


def test_short_rejects_marginal_impulse_that_only_passes_generic_thresholds() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.418,
            "ma10": 158.445,
            "ma20": 158.457,
            "ema20": 158.454,
            "adx": 31.8,
            "atr_pips": 3.9,
            "vol_5m": 2.2,
            "rsi": 36.5,
            "plus_di": 16.5,
            "minus_di": 22.0,
            "roc5": -0.018,
            "ema_slope_10": -0.0010,
            "candles": [
                {"high": 158.565, "low": 158.548, "close": 158.552},
                {"high": 158.534, "low": 158.514, "close": 158.518},
                {"high": 158.500, "low": 158.480, "close": 158.484},
                {"high": 158.466, "low": 158.448, "close": 158.452},
                {"high": 158.430, "low": 158.416, "close": 158.418},
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
        "rsi": 36.0,
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


def test_tight_short_context_rejects_without_clear_downside_drift() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.432,
            "ma10": 158.468,
            "ma20": 158.481,
            "ema20": 158.462,
            "adx": 28.5,
            "atr_pips": 3.1,
            "vol_5m": 1.7,
            "rsi": 39.5,
            "plus_di": 14.0,
            "minus_di": 22.0,
            "roc5": -0.021,
            "ema_slope_10": -0.0009,
            "drift_pips_15m": -0.08,
            "micro_chop_score": 0.62,
            "candles": [
                {"high": 158.51, "low": 158.47, "close": 158.49},
                {"high": 158.49, "low": 158.45, "close": 158.46},
                {"high": 158.47, "low": 158.44, "close": 158.45},
                {"high": 158.45, "low": 158.43, "close": 158.432},
            ],
        }
    )

    assert signal is None


def test_clean_trend_short_rejects_late_breakdown_chase() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.402,
            "ma10": 158.454,
            "ma20": 158.470,
            "ema20": 158.462,
            "adx": 30.8,
            "atr_pips": 3.6,
            "vol_5m": 2.2,
            "rsi": 34.2,
            "plus_di": 11.0,
            "minus_di": 25.0,
            "roc5": -0.022,
            "ema_slope_10": -0.0012,
            "drift_pips_15m": -0.31,
            "range_score": 0.18,
            "candles": [
                {"open": 158.51, "high": 158.52, "low": 158.47, "close": 158.49},
                {"open": 158.49, "high": 158.49, "low": 158.45, "close": 158.46},
                {"open": 158.46, "high": 158.47, "low": 158.43, "close": 158.44},
                {"open": 158.44, "high": 158.44, "low": 158.40, "close": 158.402},
            ],
        }
    )

    assert signal is None


def test_low_range_score_alone_does_not_block_short_reaccel() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.424,
            "ma10": 158.468,
            "ma20": 158.482,
            "ema20": 158.458,
            "adx": 31.2,
            "atr_pips": 3.1,
            "vol_5m": 1.6,
            "rsi": 39.0,
            "plus_di": 15.5,
            "minus_di": 23.5,
            "roc5": -0.031,
            "ema_slope_10": -0.0012,
            "drift_pips_15m": -0.31,
            "range_score": 0.18,
            "candles": [
                {"open": 158.52, "high": 158.52, "low": 158.47, "close": 158.49},
                {"open": 158.49, "high": 158.49, "low": 158.45, "close": 158.46},
                {"open": 158.47, "high": 158.47, "low": 158.43, "close": 158.44},
                {"open": 158.44, "high": 158.44, "low": 158.42, "close": 158.424},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"
    assert signal["notes"]["momentum_burst"]["reaccel"] is True


def test_tight_short_context_allows_when_impulse_is_clean() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.424,
            "ma10": 158.468,
            "ma20": 158.482,
            "ema20": 158.458,
            "adx": 31.2,
            "atr_pips": 3.1,
            "vol_5m": 1.6,
            "rsi": 39.0,
            "plus_di": 11.0,
            "minus_di": 23.5,
            "roc5": -0.031,
            "ema_slope_10": -0.0014,
            "drift_pips_15m": -0.28,
            "range_score": 0.38,
            "candles": [
                {"high": 158.52, "low": 158.47, "close": 158.49},
                {"high": 158.49, "low": 158.45, "close": 158.46},
                {"high": 158.47, "low": 158.43, "close": 158.44},
                {"high": 158.44, "low": 158.42, "close": 158.424},
            ],
        }
    )

    assert signal is not None
    assert signal["action"] == "OPEN_SHORT"


def test_tight_short_context_rejects_oversold_breakdown_chase() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.402,
            "ma10": 158.454,
            "ma20": 158.470,
            "ema20": 158.470,
            "adx": 30.8,
            "atr_pips": 3.2,
            "vol_5m": 1.5,
            "rsi": 33.0,
            "plus_di": 11.0,
            "minus_di": 24.0,
            "roc5": -0.031,
            "ema_slope_10": -0.0014,
            "drift_pips_15m": -0.34,
            "range_score": 0.36,
            "candles": [
                {"open": 158.51, "high": 158.52, "low": 158.47, "close": 158.49},
                {"open": 158.49, "high": 158.49, "low": 158.45, "close": 158.46},
                {"open": 158.46, "high": 158.47, "low": 158.43, "close": 158.44},
                {"open": 158.44, "high": 158.44, "low": 158.40, "close": 158.402},
            ],
        }
    )

    assert signal is None


def test_tight_short_context_rejects_oversold_rebound_squeeze() -> None:
    signal = MomentumBurstMicro.check(
        {
            "close": 158.468,
            "ma10": 158.496,
            "ma20": 158.508,
            "ema20": 158.525,
            "adx": 29.6,
            "atr_pips": 3.1,
            "vol_5m": 1.6,
            "rsi": 33.4,
            "plus_di": 12.0,
            "minus_di": 23.0,
            "roc5": -0.026,
            "ema_slope_10": -0.0013,
            "drift_pips_15m": -0.27,
            "range_score": 0.37,
            "candles": [
                {"open": 158.56, "high": 158.56, "low": 158.52, "close": 158.53},
                {"open": 158.505, "high": 158.54, "low": 158.50, "close": 158.535},
                {"open": 158.485, "high": 158.52, "low": 158.48, "close": 158.515},
                {"open": 158.49, "high": 158.50, "low": 158.46, "close": 158.468},
            ],
        }
    )

    assert signal is None
