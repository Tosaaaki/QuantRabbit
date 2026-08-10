from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))

import build_system_admission as admission


class SystemAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((HERE / "SYSTEM_ADMISSION_V1.json").read_text())

    def test_regeneration_is_byte_identical(self) -> None:
        expected = (HERE / "SYSTEM_ADMISSION_V1.json").read_text()
        self.assertEqual(admission.canonical_json(admission.build_report()), expected)

    def test_preregistration_hash_is_bound(self) -> None:
        self.assertEqual(
            self.report["contract"]["preregistration"]["sha256"],
            "9b2b2196cbed5e28c435518ef46d6c05ace974b5201c07a33f21b0e92c9b9460",
        )

    def test_calendar_cannot_establish_first_publication(self) -> None:
        row = self.report["source_inventory"]["economic_calendar"]
        self.assertEqual(row["actual_non_null_count"], 0)
        self.assertFalse(row["provider_receipt_timestamp_field"])
        self.assertFalse(row["revision_lineage"])
        self.assertFalse(row["append_only_history"])

    def test_cross_asset_history_is_not_executable(self) -> None:
        row = self.report["source_inventory"]["context_asset_charts"]
        self.assertEqual(row["snapshot_count"], 1)
        self.assertFalse(row["all_required_views_have_bid_ask"])
        self.assertTrue(all(set(view["recent_candle_counts"]) == {30} for view in row["view_rows"]))

    def test_inherited_execution_evidence_remains_zero(self) -> None:
        row = self.report["source_inventory"]["inherited_execution_coverage"]
        self.assertEqual(row["episode_count"], 251)
        self.assertEqual(row["strict_eligible"], 0)
        self.assertEqual(row["overall_stage_coverage"]["slippage_fee_financing"], 0)
        self.assertEqual(row["overall_stage_coverage"]["margin_exposure_concurrency"], 0)
        self.assertEqual(row["overall_stage_coverage"]["exit_unwind"], 0)

    def test_replay_stops_before_outcomes(self) -> None:
        self.assertFalse(self.report["replay"]["started"])
        self.assertEqual(self.report["replay"]["grid_points_evaluated"], 0)
        self.assertTrue(all(value is None for value in self.report["replay"]["metrics"].values()))

    def test_fail_closed_classification(self) -> None:
        self.assertEqual(self.report["classification"], "NOT_EVALUABLE")
        self.assertEqual(self.report["parent_target_status"], "TARGET_PATH_NOT_YET_PROVEN")
        self.assertGreaterEqual(len(self.report["failed_admission_gates"]), 6)

    def test_no_holdout_or_external_execution(self) -> None:
        boundary = self.report["inspection_boundary"]
        self.assertFalse(boundary["holdout_read"])
        self.assertFalse(boundary["network_data_acquisition"])
        self.assertFalse(boundary["live_paper_broker_order_deploy"])

    def test_independent_oracle_passes(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / "verify_system_admission.py"),
                "--check",
                str(HERE / "INDEPENDENT_ORACLE_V1.json"),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["all_pass"])
        self.assertEqual(payload["classification"], "NOT_EVALUABLE")


if __name__ == "__main__":
    unittest.main()
