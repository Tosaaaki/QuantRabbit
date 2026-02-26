from __future__ import annotations

import math

import numpy as np
import pandas as pd

from workers.common import forecast_gate


def _synthetic_candles(*, n: int, freq: str, drift: float) -> list[dict[str, object]]:
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    base = 150.0
    # keep both trend and short-term pullbacks so indicators are non-trivial
    wave = 0.04 * np.sin(np.linspace(0.0, 8.0 * math.pi, n))
    trend = drift * np.arange(n)
    close = base + trend + wave
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    out: list[dict[str, object]] = []
    for ts, o, h, l, c in zip(idx, open_, high, low, close):
        out.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
            }
        )
    return out


def test_technical_prediction_direction_changes_with_trend() -> None:
    up = _synthetic_candles(n=450, freq="5min", drift=0.0028)
    down = _synthetic_candles(n=450, freq="5min", drift=-0.0028)

    up_row = forecast_gate._technical_prediction_for_horizon(up, horizon="1h", step_bars=12)
    down_row = forecast_gate._technical_prediction_for_horizon(down, horizon="1h", step_bars=12)

    assert isinstance(up_row, dict)
    assert isinstance(down_row, dict)
    assert up_row["source"] == "technical"
    assert down_row["source"] == "technical"
    assert float(up_row["p_up"]) > 0.5
    assert float(down_row["p_up"]) < 0.5


def test_technical_prediction_projection_score_tracks_direction() -> None:
    up = _synthetic_candles(n=520, freq="5min", drift=0.0032)
    down = _synthetic_candles(n=520, freq="5min", drift=-0.0032)

    up_row = forecast_gate._technical_prediction_for_horizon(up, horizon="1h", step_bars=12)
    down_row = forecast_gate._technical_prediction_for_horizon(down, horizon="1h", step_bars=12)

    assert isinstance(up_row, dict)
    assert isinstance(down_row, dict)
    assert float(up_row.get("projection_score") or 0.0) > 0.0
    assert float(down_row.get("projection_score") or 0.0) < 0.0
    assert 0.0 <= float(up_row.get("projection_confidence") or 0.0) <= 1.0
    assert 0.0 <= float(down_row.get("projection_confidence") or 0.0) <= 1.0


def test_technical_prediction_exposes_trendline_and_sr_context() -> None:
    up = _synthetic_candles(n=520, freq="5min", drift=0.0032)
    down = _synthetic_candles(n=520, freq="5min", drift=-0.0032)

    up_row = forecast_gate._technical_prediction_for_horizon(up, horizon="1h", step_bars=12)
    down_row = forecast_gate._technical_prediction_for_horizon(down, horizon="1h", step_bars=12)

    assert isinstance(up_row, dict)
    assert isinstance(down_row, dict)
    for key in (
        "trend_slope_pips_20",
        "trend_slope_pips_50",
        "trend_accel_pips",
        "sr_balance_20",
        "breakout_bias_20",
        "rebound_signal_20",
        "rebound_drop_score_20",
        "rebound_oversold_score_20",
        "rebound_decel_score_20",
        "rebound_wick_score_20",
        "rebound_weight",
        "squeeze_score_20",
        "range_low_pips",
        "range_high_pips",
        "range_sigma_pips",
        "range_low_price",
        "range_high_price",
    ):
        assert key in up_row
        assert key in down_row
    assert float(up_row["trend_slope_pips_20"]) > 0.0
    assert float(down_row["trend_slope_pips_20"]) < 0.0
    assert float(up_row["breakout_bias_20"]) > float(down_row["breakout_bias_20"])
    assert 0.0 <= float(up_row["squeeze_score_20"]) <= 1.0
    assert 0.0 <= float(down_row["squeeze_score_20"]) <= 1.0
    assert float(up_row["range_low_pips"]) < float(up_row["range_high_pips"])
    assert float(down_row["range_low_pips"]) < float(down_row["range_high_pips"])
    assert float(up_row["range_sigma_pips"]) > 0.0
    assert float(down_row["range_sigma_pips"]) > 0.0


def test_rebound_bias_signal_rewards_lower_wick_rejection() -> None:
    reject_signal, reject_components = forecast_gate._rebound_bias_signal(
        ret1=-1.8,
        ret3=-2.2,
        ret12=-2.6,
        rsi=-0.9,
        range_pos=-0.8,
        sr_balance=-0.7,
        trend_accel=0.35,
        trend_pullback=-0.5,
        breakout_bias=-0.4,
        trend_strength=0.55,
        last_candle={
            "open": 153.70,
            "high": 153.72,
            "low": 153.58,
            "close": 153.715,
        },
    )
    continuation_signal, continuation_components = forecast_gate._rebound_bias_signal(
        ret1=-1.8,
        ret3=-2.2,
        ret12=-2.6,
        rsi=-0.9,
        range_pos=-0.8,
        sr_balance=-0.7,
        trend_accel=0.35,
        trend_pullback=-0.5,
        breakout_bias=-0.4,
        trend_strength=0.55,
        last_candle={
            "open": 153.70,
            "high": 153.71,
            "low": 153.58,
            "close": 153.62,
        },
    )

    assert 0.0 <= float(reject_signal) <= 1.0
    assert 0.0 <= float(continuation_signal) <= 1.0
    assert float(reject_components["wick_score"]) > float(continuation_components["wick_score"])
    assert float(reject_signal) > float(continuation_signal)


def test_technical_prediction_uses_latest_finite_feature_row(monkeypatch) -> None:
    from analysis import forecast_sklearn

    candles = _synthetic_candles(n=120, freq="1min", drift=0.0005)
    idx = pd.date_range("2026-01-02", periods=2, freq="1min", tz="UTC")
    valid_row = {
        "atr_pips_14": 1.2,
        "vol_pips_20": 1.0,
        "ret_pips_1": 0.4,
        "ret_pips_3": 0.8,
        "ret_pips_12": 1.6,
        "ma_gap_pips_10_20": 0.6,
        "close_ma20_pips": 0.3,
        "close_ma50_pips": 0.2,
        "rsi_14": 55.0,
        "range_pos": 0.55,
        "trend_slope_pips_20": 0.2,
        "trend_slope_pips_50": 0.1,
        "trend_accel_pips": 0.05,
        "sr_balance_20": 0.1,
        "breakout_up_pips_20": 0.4,
        "breakout_down_pips_20": 0.2,
        "donchian_width_pips_20": 2.0,
        "range_compression_20": 0.3,
        "trend_pullback_norm_20": 0.1,
    }
    invalid_latest = dict(valid_row)
    invalid_latest["atr_pips_14"] = float("nan")
    frame = pd.DataFrame([valid_row, invalid_latest], index=idx)

    monkeypatch.setattr(forecast_sklearn, "compute_feature_frame", lambda _: frame)

    row = forecast_gate._technical_prediction_for_horizon(
        candles,
        horizon="1m",
        step_bars=1,
        timeframe="M1",
    )
    assert isinstance(row, dict)
    assert row.get("status") == "ready"
    assert bool(row.get("forecast_ready")) is True
    assert row.get("feature_ts") == idx[0].isoformat()


def test_decide_includes_range_band_fields(monkeypatch) -> None:
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "1d": {
                "p_up": 0.38,
                "expected_pips": -2.4,
                "source": "technical",
                "trend_strength": 0.62,
                "range_pressure": 0.38,
                "range_low_pips": -4.8,
                "range_high_pips": -0.6,
                "range_sigma_pips": 1.2,
                "range_low_price": 149.82,
                "range_high_price": 149.86,
                "rebound_signal_20": 0.73,
            }
        },
    )

    decision = forecast_gate.decide(
        strategy_tag="TrendMA",
        pocket="macro",
        side="buy",
        units=20_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "edge_block"
    assert decision.range_low_pips == -4.8
    assert decision.range_high_pips == -0.6
    assert decision.range_sigma_pips == 1.2
    assert decision.range_low_price == 149.82
    assert decision.range_high_price == 149.86
    assert decision.rebound_probability == 0.73
    assert decision.target_reach_prob is not None
    assert 0.0 <= float(decision.target_reach_prob) <= 1.0


def test_decide_blocks_opposite_side_when_only_technical_source_available(monkeypatch) -> None:
    candles_m5 = _synthetic_candles(n=600, freq="5min", drift=0.0025)
    candles_h1 = _synthetic_candles(n=520, freq="1h", drift=0.018)
    candles_d1 = _synthetic_candles(n=240, freq="1d", drift=0.07)

    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_fetch_candles_by_tf",
        lambda: {"M5": candles_m5, "H1": candles_h1, "D1": candles_d1},
    )
    forecast_gate._PRED_CACHE = None
    forecast_gate._PRED_CACHE_TS = 0.0

    decision = forecast_gate.decide(
        strategy_tag="TrendMA",
        pocket="macro",
        side="sell",
        units=-20_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "edge_block"
    assert decision.source == "technical"


def test_decide_blocks_trend_style_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "1d": {
                "p_up": 0.66,
                "expected_pips": 3.2,
                "source": "technical",
                "trend_strength": 0.21,
                "range_pressure": 0.79,
            }
        },
    )

    decision = forecast_gate.decide(
        strategy_tag="TrendMA",
        pocket="macro",
        side="buy",
        units=20_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "style_mismatch_trend"
    assert decision.style == "trend"


def test_decide_blocks_range_style_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "8h": {
                "p_up": 0.44,
                "expected_pips": -1.8,
                "source": "technical",
                "trend_strength": 0.78,
                "range_pressure": 0.22,
            }
        },
    )

    decision = forecast_gate.decide(
        strategy_tag="BB_RSI",
        pocket="micro",
        side="sell",
        units=-15_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "style_mismatch_range"
    assert decision.style == "range"


def test_decide_returns_explicit_allow_when_scale_is_full(monkeypatch) -> None:
    monkeypatch.setattr(forecast_gate, "_ENABLED", True)
    monkeypatch.setattr(forecast_gate, "_POCKET_ALLOWLIST", set())
    monkeypatch.setattr(forecast_gate, "_STRATEGY_ALLOWLIST", set())
    monkeypatch.setattr(forecast_gate, "_EDGE_BAD", 0.45)
    monkeypatch.setattr(forecast_gate, "_EDGE_REF", 0.55)
    monkeypatch.setattr(forecast_gate, "_SCALE_MIN", 0.5)
    monkeypatch.setattr(forecast_gate, "_EXPECTED_PIPS_GUARD_ENABLED", False)
    monkeypatch.setattr(forecast_gate, "_TARGET_REACH_GUARD_ENABLED", False)
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "1m": {
                "p_up": 0.86,
                "expected_pips": 1.1,
                "source": "technical",
                "trend_strength": 0.66,
                "range_pressure": 0.34,
            }
        },
    )

    decision = forecast_gate.decide(
        strategy_tag="scalp_ping_5s_c_live",
        pocket="scalp_fast",
        side="buy",
        units=1_200,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is True
    assert decision.scale == 1.0
    assert decision.reason == "edge_allow"
    assert decision.p_up == 0.86


def test_decide_uses_strategy_specific_edge_override(monkeypatch) -> None:
    monkeypatch.setenv("FORECAST_GATE_EDGE_BLOCK_TREND_STRATEGY_TRENDMA", "0.80")
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "1d": {
                "p_up": 0.70,  # edge=0.70 for buy
                "expected_pips": 2.5,
                "source": "technical",
                "trend_strength": 0.82,
                "range_pressure": 0.18,
            }
        },
    )
    decision = forecast_gate.decide(
        strategy_tag="TrendMA",
        pocket="macro",
        side="buy",
        units=12_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "edge_block"
    assert decision.style == "trend"


def test_decide_uses_strategy_specific_style_override(monkeypatch) -> None:
    monkeypatch.setenv("FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_BBRSI", "0.90")
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "8h": {
                "p_up": 0.45,
                "expected_pips": -1.2,
                "source": "technical",
                "trend_strength": 0.30,
                "range_pressure": 0.72,
            }
        },
    )
    decision = forecast_gate.decide(
        strategy_tag="BB_RSI",
        pocket="micro",
        side="sell",
        units=-9_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "style_mismatch_range"
    assert decision.style == "range"


def test_decide_uses_strategy_specific_edge_override_with_underscore_suffix(monkeypatch) -> None:
    monkeypatch.setenv("FORECAST_GATE_EDGE_BLOCK_STRATEGY_SCALP_PING_5S_B_LIVE", "0.76")
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "1m": {
                "p_up": 0.72,
                "expected_pips": 0.3,
                "source": "technical",
                "trend_strength": 0.62,
                "range_pressure": 0.38,
            }
        },
    )
    decision = forecast_gate.decide(
        strategy_tag="scalp_ping_5s_b_live-labc123",
        pocket="scalp_fast",
        side="buy",
        units=1_200,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "edge_block"


def test_decide_blocks_when_expected_pips_guard_fails(monkeypatch) -> None:
    monkeypatch.setenv(
        "FORECAST_GATE_EXPECTED_PIPS_GUARD_ENABLED_STRATEGY_MICROLEVELREACTOR",
        "1",
    )
    monkeypatch.setenv(
        "FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_MICROLEVELREACTOR",
        "0.18",
    )
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "10m": {
                "p_up": 0.79,
                "expected_pips": 0.05,
                "source": "technical",
                "trend_strength": 0.40,
                "range_pressure": 0.66,
                "range_sigma_pips": 0.45,
            }
        },
    )
    decision = forecast_gate.decide(
        strategy_tag="MicroLevelReactor-fade-upper",
        pocket="micro",
        side="buy",
        units=3_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "expected_pips_low"


def test_decide_blocks_when_target_reach_guard_fails(monkeypatch) -> None:
    monkeypatch.setenv(
        "FORECAST_GATE_TARGET_REACH_GUARD_ENABLED_STRATEGY_MICROLEVELREACTOR",
        "1",
    )
    monkeypatch.setenv(
        "FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_MICROLEVELREACTOR",
        "0.30",
    )
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "10m": {
                "p_up": 0.78,
                "expected_pips": 0.6,
                "source": "technical",
                "trend_strength": 0.42,
                "range_pressure": 0.64,
                "range_sigma_pips": 0.55,
                "tp_pips_hint": 2.6,
            }
        },
    )
    decision = forecast_gate.decide(
        strategy_tag="MicroLevelReactor-bounce-lower",
        pocket="micro",
        side="buy",
        units=3_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "target_reach_prob_low"


def test_decide_projection_penalty_blocks_borderline_edge(monkeypatch) -> None:
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "1d": {
                "p_up": 0.46,
                "expected_pips": -0.4,
                "source": "technical",
                "trend_strength": 0.78,
                "range_pressure": 0.22,
                "projection_score": -1.0,
                "projection_confidence": 0.9,
            }
        },
    )
    decision = forecast_gate.decide(
        strategy_tag="TrendMA",
        pocket="macro",
        side="buy",
        units=20_000,
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "edge_block"


def test_decide_tf_confluence_penalty_blocks_misaligned_higher_tf(monkeypatch) -> None:
    monkeypatch.setattr(forecast_gate, "_TF_CONFLUENCE_ENABLED", True)
    monkeypatch.setattr(forecast_gate, "_TF_CONFLUENCE_BONUS", 0.0)
    monkeypatch.setattr(forecast_gate, "_TF_CONFLUENCE_PENALTY", 0.2)
    monkeypatch.setattr(forecast_gate, "_TF_CONFLUENCE_MIN_CONFIRM", 1)
    monkeypatch.setattr(forecast_gate, "_load_bundle_cached", lambda: None)
    monkeypatch.setattr(
        forecast_gate,
        "_ensure_predictions",
        lambda bundle: {
            "1d": {
                "p_up": 0.44,
                "expected_pips": -0.2,
                "source": "technical",
                "trend_strength": 0.82,
                "range_pressure": 0.18,
            },
            "8h": {
                "p_up": 0.18,
                "expected_pips": -2.8,
                "source": "technical",
                "trend_strength": 0.88,
                "range_pressure": 0.12,
            },
        },
    )
    decision = forecast_gate.decide(
        strategy_tag="TrendMA",
        pocket="macro",
        side="buy",
        units=18_000,
        entry_thesis={"forecast_support_horizons": ["8h"]},
        meta={"instrument": "USD_JPY"},
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.reason == "edge_block"
    assert decision.tf_confluence_count == 1
    assert decision.tf_confluence_score is not None
    assert float(decision.tf_confluence_score) < 0.0
    assert decision.tf_confluence_horizons == "8h"


def test_horizon_for_strategy_tag_prefers_micro_10m() -> None:
    horizon = forecast_gate._horizon_for(
        "micro",
        "MicroRangeBreak",
        entry_thesis=None,
        meta=None,
    )
    assert horizon == "10m"


def test_horizon_for_strategy_tag_prefers_scalp_ping_1m() -> None:
    horizon = forecast_gate._horizon_for(
        "scalp_fast",
        "scalp_ping_5s_live",
        entry_thesis=None,
        meta=None,
    )
    assert horizon == "1m"


def test_horizon_for_unknown_strategy_uses_pocket_default() -> None:
    horizon = forecast_gate._horizon_for(
        "micro",
        "BB_RSI",
        entry_thesis=None,
        meta=None,
    )
    assert horizon == forecast_gate._HORIZON_MICRO


def test_estimate_directional_skill_positive() -> None:
    signal = [1.0] * 200
    target = [1.0] * 200
    skill, hit_rate, samples = forecast_gate._estimate_directional_skill(
        signal_values=signal,
        target_values=target,
        min_samples=40,
        lookback=120,
    )
    assert samples == 120
    assert hit_rate == 1.0
    assert skill > 0.9


def test_estimate_directional_skill_negative() -> None:
    signal = [1.0] * 200
    target = [-1.0] * 200
    skill, hit_rate, samples = forecast_gate._estimate_directional_skill(
        signal_values=signal,
        target_values=target,
        min_samples=40,
        lookback=120,
    )
    assert samples == 120
    assert hit_rate == 0.0
    assert skill < -0.9


def test_estimate_session_hour_bias_positive() -> None:
    idx = pd.date_range("2026-01-01", periods=240, freq="1h", tz="UTC")
    targets: list[float] = []
    for ts in idx:
        hour_jst = (int(ts.hour) + 9) % 24
        targets.append(1.2 if hour_jst == 10 else -0.1)
    current_ts = next(ts for ts in reversed(idx) if ((int(ts.hour) + 9) % 24) == 10)
    bias, mean_move, samples, hour_jst = forecast_gate._estimate_session_hour_bias(
        timestamp_values=list(idx),
        target_values=targets,
        current_timestamp=current_ts,
        min_samples=8,
        lookback=240,
    )
    assert hour_jst == 10
    assert samples >= 8
    assert mean_move > 0.0
    assert bias > 0.0


def test_estimate_session_hour_bias_negative() -> None:
    idx = pd.date_range("2026-01-01", periods=240, freq="1h", tz="UTC")
    targets: list[float] = []
    for ts in idx:
        hour_jst = (int(ts.hour) + 9) % 24
        targets.append(-1.1 if hour_jst == 3 else 0.12)
    current_ts = next(ts for ts in reversed(idx) if ((int(ts.hour) + 9) % 24) == 3)
    bias, mean_move, samples, hour_jst = forecast_gate._estimate_session_hour_bias(
        timestamp_values=list(idx),
        target_values=targets,
        current_timestamp=current_ts,
        min_samples=8,
        lookback=240,
    )
    assert hour_jst == 3
    assert samples >= 8
    assert mean_move < 0.0
    assert bias < 0.0
