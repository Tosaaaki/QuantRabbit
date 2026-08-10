from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from math import isclose, log
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "reclassify_existing_evidence", HERE / "reclassify_existing_evidence.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReclassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = MODULE.main()

    def test_contract_is_fixed_before_readback(self) -> None:
        contract = json.loads((HERE / "MONTHLY_2X_DIRECT_PROOF_V1.json").read_text())
        self.assertEqual(contract["contract_id"], "MONTHLY_2X_DIRECT_PROOF_V1")
        self.assertEqual(contract["scope"]["change_from_prior_contract"], "target_multiple_only: 3.0 -> 2.0")
        self.assertTrue(contract["scope"]["all_other_conditions_unchanged"])

    def test_all_source_hashes_match(self) -> None:
        self.assertTrue(self.result["source_hashes_verified"])
        self.assertEqual(set(self.result["source_hashes"]), set(MODULE.EXPECTED_SOURCES))

    def test_linear_gap_independent_fraction_oracle(self) -> None:
        baseline = Fraction(10476891546051672, 10**16)
        target = Fraction(2, 1)
        expected = target - baseline
        self.assertEqual(float(expected), self.result["gap"]["linear_multiple_gap"])
        self.assertEqual(float(expected * 200000), self.result["gap"]["linear_equity_gap_jpy"])

    def test_log_gap_independent_ratio_oracle(self) -> None:
        baseline = self.result["gap"]["baseline_rolling_30d_multiple"]
        expected = log(2.0 / baseline)
        self.assertTrue(isclose(expected, self.result["gap"]["log_growth_gap"], rel_tol=0, abs_tol=1e-15))
        self.assertGreater(expected, 0)

    def test_no_family_is_proven(self) -> None:
        self.assertNotIn("PROVEN", {row["status"] for row in self.result["families"]})
        self.assertEqual(self.result["overall_status"], "TARGET_PATH_NOT_YET_PROVEN")

    def test_missing_evidence_is_not_zero(self) -> None:
        by_family = {row["family"]: row for row in self.result["families"]}
        self.assertEqual(by_family["trailing_break_even_partial_take_profit"]["status"], "NOT_EVALUABLE")
        self.assertEqual(by_family["x_derived_methods"]["status"], "NOT_EVALUABLE")

    def test_cap_compliant_growth_grid_has_no_2x_row(self) -> None:
        by_family = {row["family"]: row for row in self.result["families"]}
        evidence = by_family["dynamic_lot_inventory_exposure"]["evidence"]
        self.assertEqual(evidence["cap_compliant_2x_rows"], 0)
        self.assertLess(evidence["cap_compliant_best_rolling_30d_multiple"], 2.0)

    def test_no_prohibited_external_state(self) -> None:
        self.assertFalse(self.result["holdout_read"])
        self.assertFalse(self.result["live_paper_broker_order_deploy_touched"])


if __name__ == "__main__":
    unittest.main()
