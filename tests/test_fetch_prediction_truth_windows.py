from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
