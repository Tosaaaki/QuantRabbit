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
