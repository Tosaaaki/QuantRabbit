import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class StrategyExpansionArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((HERE / "strategy_expansion_report_v1.json").read_text())
        cls.rows = [json.loads(line) for line in (HERE / "strategy_expansion_grid_v1.jsonl").read_text().splitlines()]
        cls.exit_report = json.loads((HERE / "strategy_exit_report_v1.json").read_text())
        cls.exit_rows = [json.loads(line) for line in (HERE / "strategy_exit_grid_v1.jsonl").read_text().splitlines()]
        cls.reason = json.loads((HERE / "profit_reason_ledger_v1.json").read_text())

    def test_holdout_is_not_used(self):
        self.assertFalse(self.report["holdout_used"])
        self.assertFalse(self.exit_report["holdout_used"])
        self.assertFalse(self.reason["holdout_used"])

    def test_primary_strategy_selection_has_cost_bound(self):
        for row in self.rows:
            if row["train_connected_plateau"]:
                self.assertEqual(row["slippage_spread_multiple"], 0.5)

    def test_validation_never_self_selects(self):
        for row in self.rows + self.exit_rows:
            if row["validation_pass"]:
                self.assertTrue(row["train_connected_plateau"])

    def test_strategy_pnl_is_jpy_per_1000_units(self):
        for row in self.rows + self.exit_rows:
            if row["trades"]:
                self.assertAlmostEqual(
                    row["net_jpy_per_1000u"],
                    row["expectancy_jpy_per_1000u"] * row["trades"],
                    places=6,
                )

    def test_exit_terminals_conserve_trade_count(self):
        for row in self.exit_rows:
            self.assertEqual(sum(row["terminal_counts"].values()), row["trades"])

    def test_eurusd_short_validation_reason_is_actual_positive(self):
        reason = next(row for row in self.reason["verified_reasons_can_work"] if row["reason"] == "EURUSD_SHORT_PERSISTS_ACROSS_SPLIT")
        self.assertEqual(reason["validation"]["trades"], 5)
        self.assertEqual(reason["validation"]["wins"], 5)
        self.assertGreater(reason["validation"]["net_jpy"], 0)
        self.assertGreater(reason["validation"]["paired_bootstrap_lcb_expectancy_jpy"], 0)

    def test_exact_limit_tp_evidence_is_positive_but_not_live_permission(self):
        evidence = self.reason["exact_limit_attached_tp_evidence"]
        self.assertTrue(evidence["available"])
        self.assertEqual(evidence["sample_count"], 4)
        self.assertEqual(evidence["wins"], 4)
        self.assertEqual(evidence["losses"], 0)
        self.assertGreater(evidence["net_jpy"], 0)
        self.assertGreater(evidence["expectancy_after_bidask_jpy"], 0)
        self.assertFalse(evidence["live_permission_allowed"])


if __name__ == "__main__":
    unittest.main()
