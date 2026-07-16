from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:  # optional research dependency
    raise unittest.SkipTest(f"market-phase research dependencies unavailable: {exc.name}") from exc

from scripts.audit_market_phase_technical_selector import (
    _merge_non_overlapping_rows,
    _phase_and_rule_frame,
    _selected_rows,
)


def _frame(rows: int = 2400) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    x = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "atr_14": 2.0 + (x % 11) / 20.0,
            "atr_48": 2.5 + (x % 17) / 20.0,
            "atr_288": 3.0 + (x % 23) / 20.0,
            "ema_gap_5_24": np.sin(x / 30.0) * 2.0,
            "ema_gap_12_48": np.sin(x / 60.0) * 3.0,
            "return_3": np.sin(x / 15.0),
            "return_12": np.sin(x / 60.0) * 2.0,
            "return_48": np.sin(x / 90.0) * 3.0,
            "range_location_12": (x % 12) / 11.0,
            "range_location_48": (x % 48) / 47.0,
            "rsi_14": 50.0 + np.sin(x / 20.0) * 20.0,
            "body_atr_ratio": np.sin(x / 8.0),
        },
        index=index,
    )


def test_future_mutation_cannot_change_earlier_phase_or_rule_scores() -> None:
    source = _frame()
    baseline = _phase_and_rule_frame(source, np=np, pd=pd)
    changed = source.copy()
    changed.iloc[-100:, changed.columns.get_loc("ema_gap_12_48")] = 10_000.0
    changed.iloc[-100:, changed.columns.get_loc("atr_14")] = 10_000.0
    replayed = _phase_and_rule_frame(changed, np=np, pd=pd)
    cutoff = source.index[-101]
    columns = ["market_phase", "trend_fast", "mean_revert_fast"]
    pd.testing.assert_frame_equal(
        baseline.loc[:cutoff, columns],
        replayed.loc[:cutoff, columns],
    )


def test_phase_output_is_finite_four_state_taxonomy_after_warmup() -> None:
    result = _phase_and_rule_frame(_frame(), np=np, pd=pd)
    phases = set(result["market_phase"].dropna())
    assert phases <= {"PRE_RANGE", "RANGE", "PRE_TREND", "TREND"}
    assert phases
    assert set(result["utc_session_bucket"]) == {
        "UTC_00_08",
        "UTC_08_13",
        "UTC_13_17",
        "UTC_17_22",
        "UTC_22_24",
    }


def test_selection_is_pair_local_non_overlapping_and_uses_executable_side() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {
            "pair": ["EUR_USD", "EUR_USD", "EUR_USD"],
            "market_phase": ["TREND"] * 3,
            "utc_session_bucket": ["UTC_00_08"] * 3,
            "entry_timestamp_utc": index + pd.Timedelta(minutes=5),
            "future_timestamp_utc": index + pd.Timedelta(minutes=60),
            "target_mid_pips": [3.0, 4.0, 5.0],
            "long_pips": [2.0, 3.0, 4.0],
            "short_pips": [-3.0, -4.0, -5.0],
            "entry_spread_pips": [1.0] * 3,
            "exit_spread_pips": [1.0] * 3,
            "roundtrip_spread_cost_pips": [1.0] * 3,
            "trend_fast": [1.0, 2.0, 3.0],
        },
        index=index,
    )
    selected = _selected_rows(
        frame,
        rule="trend_fast",
        orientation=1,
        threshold=0.0,
        horizon_min=60,
    )
    assert [row["timestamp_utc"] for row in selected] == [
        index[0].isoformat(),
        index[2].isoformat(),
    ]
    assert [row["executed_pips"] for row in selected] == [2.0, 4.0]


def test_selection_accepts_duplicate_timestamps_across_pairs() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "pair": ["EUR_USD", "GBP_USD"],
            "market_phase": ["TREND", "PRE_TREND"],
            "utc_session_bucket": ["UTC_00_08", "UTC_00_08"],
            "entry_timestamp_utc": [timestamp + pd.Timedelta(minutes=5)] * 2,
            "future_timestamp_utc": [timestamp + pd.Timedelta(minutes=60)] * 2,
            "target_mid_pips": [3.0, 3.0],
            "long_pips": [2.0, 3.0],
            "short_pips": [-3.0, -4.0],
            "entry_spread_pips": [1.0, 1.0],
            "exit_spread_pips": [1.0, 1.0],
            "roundtrip_spread_cost_pips": [1.0, 1.0],
            "trend_fast": [1.0, -2.0],
        },
        index=[timestamp, timestamp],
    )
    selected = _selected_rows(
        frame,
        rule="trend_fast",
        orientation=1,
        threshold=0.0,
        horizon_min=60,
    )
    assert [(row["pair"], row["executed_pips"]) for row in selected] == [
        ("EUR_USD", 2.0),
        ("GBP_USD", -4.0),
    ]


def test_phase_merge_reapplies_pair_flatness() -> None:
    rows = [
        {"pair": "EUR_USD", "timestamp_utc": "2026-01-01T00:00:00+00:00"},
        {"pair": "EUR_USD", "timestamp_utc": "2026-01-01T00:30:00+00:00"},
        {"pair": "GBP_USD", "timestamp_utc": "2026-01-01T00:30:00+00:00"},
        {"pair": "EUR_USD", "timestamp_utc": "2026-01-01T01:00:00+00:00"},
    ]
    selected = _merge_non_overlapping_rows(rows, horizon_min=60)
    assert [(row["pair"], row["timestamp_utc"]) for row in selected] == [
        ("EUR_USD", "2026-01-01T00:00:00+00:00"),
        ("GBP_USD", "2026-01-01T00:30:00+00:00"),
        ("EUR_USD", "2026-01-01T01:00:00+00:00"),
    ]
