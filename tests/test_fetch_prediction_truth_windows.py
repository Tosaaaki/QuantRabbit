from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "fetch_prediction_truth_windows.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fetch_prediction_truth_windows",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


truth_fetch = _load_module()


class FetchPredictionTruthWindowsTest(unittest.TestCase):
    def test_overlapping_prediction_windows_are_merged_with_candle_pad(self) -> None:
        rows = [
            {
                "pair": "EUR_USD",
                "entry_timestamp_utc": "2026-06-01T00:00:00+00:00",
                "future_timestamp_utc": "2026-06-02T00:00:00+00:00",
            },
            {
                "pair": "EUR_USD",
                "entry_timestamp_utc": "2026-06-02T00:00:00+00:00",
                "future_timestamp_utc": "2026-06-03T00:00:00+00:00",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            windows = truth_fetch._prediction_windows([path], granularity="S5")

        self.assertEqual(len(windows["EUR_USD"]), 1)
        start, end = windows["EUR_USD"][0]
        self.assertEqual(start.isoformat(), "2026-06-01T00:00:00+00:00")
        self.assertEqual(end.isoformat(), "2026-06-03T00:00:05+00:00")

    def test_entry_relative_override_covers_long_vehicle(self) -> None:
        entry = datetime(2026, 7, 1, 0, 5, tzinfo=timezone.utc)
        row = {
            "pair": "EUR_USD",
            "entry_timestamp_utc": entry.isoformat(),
            "future_timestamp_utc": (entry + timedelta(minutes=60)).isoformat(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            windows = truth_fetch._prediction_windows(
                [path],
                granularity="S5",
                window_minutes_from_entry=1440.0,
            )

        self.assertEqual(
            windows["EUR_USD"],
            [(entry, entry + timedelta(days=1, seconds=5))],
        )

    def test_pair_filter_fetches_only_requested_truth(self) -> None:
        rows = [
            {
                "pair": pair,
                "entry_timestamp_utc": "2026-07-01T00:00:00+00:00",
                "future_timestamp_utc": "2026-07-01T01:00:00+00:00",
            }
            for pair in ("AUD_JPY", "EUR_USD")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            windows = truth_fetch._prediction_windows(
                [path],
                granularity="S5",
                selected_pairs={"AUD_JPY"},
            )

        self.assertEqual(set(windows), {"AUD_JPY"})

    def test_exit_grace_extends_sparse_truth_for_next_executable_quote(self) -> None:
        entry = datetime(2026, 7, 1, tzinfo=timezone.utc)
        row = {
            "pair": "USD_CHF",
            "entry_timestamp_utc": entry.isoformat(),
            "future_timestamp_utc": (entry + timedelta(hours=4)).isoformat(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            windows = truth_fetch._prediction_windows(
                [path],
                granularity="S5",
                end_grace_seconds=60.0,
            )

        assert windows["USD_CHF"] == [
            (entry, entry + timedelta(hours=4, seconds=60))
        ]


if __name__ == "__main__":
    unittest.main()
