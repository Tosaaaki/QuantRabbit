from __future__ import annotations

import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


class IntegratedProfitCubeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((HERE / "report_v1.json").read_text())
        cls.cube = [json.loads(line) for line in (HERE / "canonical_decision_cube_v1.jsonl").read_text().splitlines()]
        cls.grid = [json.loads(line) for line in (HERE / "candidate_grid_v1.jsonl").read_text().splitlines()]

    def test_v2_baseline_is_bound(self) -> None:
        self.assertAlmostEqual(self.report["v2_baseline"]["raw_64d_validation_after_cost_net_jpy"], 11706.0523, places=6)
        self.assertAlmostEqual(self.report["v2_baseline"]["raw_64d_validation_profit_factor"], 1.4469329373747661, places=12)

    def test_missing_cells_remain_null(self) -> None:
        self.assertTrue(all(row["missing_not_zero"] == (row["value"] is None) for row in self.cube))
        self.assertTrue(self.report["xarray"]["null_preserved"])

    def test_exit_and_hedge_are_not_monetized(self) -> None:
        for row in self.cube:
            if row["stage"] in {"EXIT", "HEDGE"} and row["candidate_actual_after_cost_net_jpy"] is None:
                self.assertIsNone(row["value"])
        self.assertEqual(self.report["hedge_status"], "NOT_EVALUABLE_DUAL_LEG_COST_MARGIN_UNWIND_MISSING")

    def test_validation_did_not_select_the_champion(self) -> None:
        champion = self.report["train_only_champion"]
        self.assertTrue(any(row["parameter_id"] == champion and row["split"] == "TRAIN" and row["train_only_champion"] for row in self.grid))
        self.assertTrue(all(row["parameter_id"] != champion or row["train_only_champion"] for row in self.grid))

    def test_no_false_adoption(self) -> None:
        self.assertEqual(self.report["strict_pass_candidates"], [])
        self.assertEqual(self.report["conclusion"], "BASELINE_POSITIVE_INTEGRATED_IMPROVEMENT_NOT_YET_ADMISSIBLE")

    def test_oss_consumers_executed(self) -> None:
        self.assertGreater(self.report["salib"]["samples"], 0)
        self.assertGreater(self.report["pymoo"]["pareto_candidate_count"], 0)
        self.assertEqual(self.report["mapie"]["validation_n"], 101)


if __name__ == "__main__":
    unittest.main()
