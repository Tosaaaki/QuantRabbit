from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.forecast_sklearn import (
    HorizonSpec,
    build_direction_dataset,
    compute_feature_frame,
    predict_latest,
    train_bundle,
    train_forecast_model,
)


def _synthetic_candles(*, n: int, freq: str) -> list[dict[str, object]]:
    # Build a smooth oscillating series so both up/down labels exist.
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    base = 150.0
    amp = 0.22  # ~22 pips
    wave = amp * np.sin(np.linspace(0.0, 12.0 * math.pi, n))
    drift = np.linspace(0.0, 0.06, n)  # ~6 pips drift
    close = base + wave + drift
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


def _trend_candles(*, n: int, freq: str, drift: float) -> list[dict[str, object]]:
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    base = 150.0
    wave = 0.01 * np.sin(np.linspace(0.0, 6.0 * math.pi, n))
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


def test_compute_feature_frame_columns() -> None:
    candles = _synthetic_candles(n=200, freq="5min")
    feats = compute_feature_frame(candles)
    assert not feats.empty
    for col in (
        "ret_pips_1",
        "ma_gap_pips_10_20",
        "atr_pips_14",
        "rsi_14",
        "range_pos",
        "trend_slope_pips_20",
        "trend_slope_pips_50",
        "trend_accel_pips",
        "sr_balance_20",
        "breakout_up_pips_20",
        "breakout_down_pips_20",
        "hour_sin",
        "dow_cos",
    ):
        assert col in feats.columns


def test_compute_feature_frame_trendline_features_follow_trend() -> None:
    up = _trend_candles(n=260, freq="5min", drift=0.0035)
    down = _trend_candles(n=260, freq="5min", drift=-0.0035)

    up_last = compute_feature_frame(up).dropna().iloc[-1]
    down_last = compute_feature_frame(down).dropna().iloc[-1]

    assert float(up_last["trend_slope_pips_20"]) > 0.0
    assert float(down_last["trend_slope_pips_20"]) < 0.0
    assert float(up_last["breakout_up_pips_20"]) > float(down_last["breakout_up_pips_20"])
    assert float(down_last["breakout_down_pips_20"]) > float(up_last["breakout_down_pips_20"])


def test_build_direction_dataset_shapes() -> None:
    candles = _synthetic_candles(n=400, freq="5min")
    spec = HorizonSpec("test", timeframe="M5", step_bars=12, min_move_pips=1.0)
    X, y, fut, names, ts = build_direction_dataset(candles, horizon=spec)
    assert X.ndim == 2
    assert X.shape[0] == y.shape[0] == fut.shape[0] == len(ts)
    assert X.shape[1] == len(names)
    assert set(np.unique(y)).issuperset({0, 1})


def test_train_forecast_model_smoke() -> None:
    candles = _synthetic_candles(n=600, freq="5min")
    spec = HorizonSpec("test", timeframe="M5", step_bars=12, min_move_pips=1.0)
    model, metrics = train_forecast_model(candles, horizon=spec)
    assert model.horizon.name == "test"
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["logloss"] >= 0.0


def test_predict_latest_includes_quantile_range_band() -> None:
    candles = _synthetic_candles(n=760, freq="5min")
    horizon = HorizonSpec("1h_test", timeframe="M5", step_bars=12, min_move_pips=1.0)
    bundle, _ = train_bundle(
        "USD_JPY",
        {"M5": candles},
        horizons=(horizon,),
    )
    rows = predict_latest(bundle, {"M5": candles})
    row = rows["1h_test"]

    for key in (
        "q10_pips",
        "q50_pips",
        "q90_pips",
        "range_low_pips",
        "range_high_pips",
        "range_sigma_pips",
        "dispersion_pips",
    ):
        assert key in row

    assert float(row["range_low_pips"]) < float(row["range_high_pips"])
    assert float(row["range_sigma_pips"]) > 0.0
