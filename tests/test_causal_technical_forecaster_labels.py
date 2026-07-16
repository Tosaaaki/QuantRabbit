from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:  # optional research dependency
    raise unittest.SkipTest(f"causal label research dependencies unavailable: {exc.name}") from exc

from scripts.train_causal_technical_forecaster import MODEL_FEATURES, _label_frame


def _source_frame() -> pd.DataFrame:
    index = pd.date_range("2026-07-01", periods=15, freq="5min", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["pair"] = "EUR_USD"
    for feature in MODEL_FEATURES:
        frame[feature] = 1.0
    frame["lookback_span_hours"] = 24.0
    frame["bid_open"] = 1.1000 + np.arange(len(frame)) * 0.0001
    frame["ask_open"] = frame["bid_open"] + 0.0002
    frame["bid_close"] = frame["bid_open"] + 0.00005
    frame["ask_close"] = frame["bid_close"] + 0.0004
    frame["mid_close"] = (frame["bid_close"] + frame["ask_close"]) / 2.0
    return frame


def test_label_horizon_is_measured_from_executable_entry() -> None:
    labelled = _label_frame(_source_frame(), horizon_min=60, np=np, pd=pd)
    row = labelled.iloc[0]
    assert row["entry_timestamp_utc"] - row.name == pd.Timedelta(minutes=5)
    assert (
        row["future_timestamp_utc"] - row["entry_timestamp_utc"]
        == pd.Timedelta(minutes=60)
    )
    assert row["future_timestamp_utc"] - row.name == pd.Timedelta(minutes=65)


def test_executable_returns_decompose_to_mid_move_minus_one_roundtrip_cost() -> None:
    labelled = _label_frame(_source_frame(), horizon_min=60, np=np, pd=pd)
    row = labelled.iloc[0]
    assert abs(row["entry_spread_pips"] - 2.0) < 1e-9
    assert abs(row["exit_spread_pips"] - 2.0) < 1e-9
    assert abs(row["roundtrip_spread_cost_pips"] - 2.0) < 1e-9
    assert abs(
        row["long_pips"]
        - (row["target_mid_pips"] - row["roundtrip_spread_cost_pips"])
    ) < 1e-9
    assert abs(
        row["short_pips"]
        - (-row["target_mid_pips"] - row["roundtrip_spread_cost_pips"])
    ) < 1e-9


def test_exit_uses_quote_at_recorded_horizon_not_later_candle_close() -> None:
    source = _source_frame()
    labelled = _label_frame(source, horizon_min=60, np=np, pd=pd)
    row = labelled.iloc[0]
    exit_row = source.loc[row["future_timestamp_utc"]]
    entry_row = source.loc[row["entry_timestamp_utc"]]
    expected_long = (exit_row["bid_open"] - entry_row["ask_open"]) * 10_000
    close_based_long = (exit_row["bid_close"] - entry_row["ask_open"]) * 10_000
    assert abs(row["long_pips"] - expected_long) < 1e-9
    assert abs(row["long_pips"] - close_based_long) > 1e-6
