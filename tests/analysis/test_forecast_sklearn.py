from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.forecast_sklearn import (
    HorizonSpec,
    build_direction_dataset,
    compute_feature_frame,
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
        "hour_sin",
        "dow_cos",
    ):
        assert col in feats.columns


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

