import json
import math
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class GrowthEngineArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preregister = json.loads((HERE / "preregister_v1.json").read_text())
        cls.report = json.loads((HERE / "growth_report_v1.json").read_text())
        cls.grid = [json.loads(line) for line in (HERE / "growth_grid_v1.jsonl").read_text().splitlines()]
        cls.multipliers = [json.loads(line) for line in (HERE / "decision_multipliers_v1.jsonl").read_text().splitlines()]

    def test_holdout_and_side_effects_remain_disabled(self):
        self.assertFalse(self.report["holdout_used"])
        permissions = self.preregister["permissions"]
        for key in ("live", "paper", "broker_order", "deploy", "holdout"):
            self.assertFalse(permissions[key])

    def test_target_math_is_200k_to_600k(self):
        objective = self.preregister["objective"]
        self.assertEqual(objective["starting_equity_jpy"], 200000.0)
        self.assertEqual(objective["target_ending_equity_jpy"], 600000.0)
        row = next(value for value in self.report["target_math"] if value["monthly_trades"] == 200)
        self.assertEqual(row["fixed_jpy_expectancy_required"], 2000.0)
        self.assertAlmostEqual(row["equal_compound_return_per_trade"], 3 ** (1 / 200) - 1, places=15)

    def test_decision_rows_never_use_current_outcome(self):
        self.assertTrue(self.multipliers)
        self.assertTrue(all(row["actual_outcome_used_for_this_decision"] is False for row in self.multipliers))

    def test_size_modifiers_are_bounded_and_never_skip(self):
        values = [row["decision_multiplier"] for row in self.multipliers]
        self.assertGreaterEqual(min(values), 0.5)
        self.assertLessEqual(max(values), 1.5)

    def test_equity_cashflow_conservation(self):
        for row in self.grid:
            self.assertAlmostEqual(row["ending_equity_jpy"], 200000.0 + row["after_cost_net_jpy"], places=6)

    def test_profit_factor_rejects_impossible_negative_value(self):
        for row in self.grid:
            value = row["profit_factor"]
            if value is not None and math.isfinite(value):
                self.assertGreaterEqual(value, 0.0)

    def test_validation_admission_is_train_frozen(self):
        train_keys = {
            (row["window"], row["policy"], row["risk_scale"], row["margin_cap_fraction"])
            for row in self.grid
            if row["split"] == "TRAIN" and row["train_connected_plateau"]
        }
        for row in self.grid:
            if row["validation_admission_candidate"]:
                self.assertIn((row["window"], row["policy"], row["risk_scale"], row["margin_cap_fraction"]), train_keys)

    def test_corrected_v2_baseline_is_not_old_15144_label(self):
        row = self.report["corrected_64d_validation_baseline_at_1x_75pct_cap"]
        self.assertNotAlmostEqual(row["after_cost_net_jpy"], 15144.4802, places=3)
        self.assertGreater(row["after_cost_net_jpy"], 0.0)


if __name__ == "__main__":
    unittest.main()
