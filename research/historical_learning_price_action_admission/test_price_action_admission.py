from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SPEC = importlib.util.spec_from_file_location("price_action_admission", ROOT / "run_price_action_admission.py")
PA = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = PA
SPEC.loader.exec_module(PA)


def bucket(count: int = 60) -> dict:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "start": start,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "timestamps": [start + timedelta(seconds=5 * index) for index in range(count)],
        "spread_bps": [1.0] * count,
    }


class PriceActionAdmissionTests(unittest.TestCase):
    def test_complete_m5_requires_all_sixty_s5_timestamps(self) -> None:
        self.assertIsNotNone(PA.finish_bucket(bucket(60)))
        self.assertIsNone(PA.finish_bucket(bucket(59)))

    def test_duplicate_s5_timestamp_is_rejected(self) -> None:
        value = bucket(60)
        value["timestamps"][-1] = value["timestamps"][-2]
        self.assertIsNone(PA.finish_bucket(value))

    def test_structure_features_reject_any_m5_gap(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [PA.Bar(start + timedelta(minutes=5 * index), 100, 101, 99, 100 + index / 100, 1.0) for index in range(48)]
        self.assertIsNotNone(PA.structure_features(bars))
        bars[20] = PA.Bar(bars[20].start + timedelta(minutes=5), 100, 101, 99, 100, 1.0)
        self.assertIsNone(PA.structure_features(bars))

    def test_source_hashes_match_preregister(self) -> None:
        prereg = json.loads((ROOT / "preregister_v1.json").read_text())
        for source in prereg["s5_bidask_inputs"]:
            self.assertEqual(PA.sha256(REPO / source["path"]), source["sha256"])

    def test_report_fails_closed_without_interpolation(self) -> None:
        report = json.loads((ROOT / "report_v1.json").read_text())
        self.assertEqual(report["feature_coverage"]["available"], 0)
        self.assertEqual(report["overall_decision"], "REJECT")
        self.assertFalse(report["holdout_used"])
        self.assertEqual(report["multidimensional_sweep"], "NOT_OPENED_FIXED_FEATURE_ADMISSION_FAILED")


if __name__ == "__main__":
    unittest.main()
