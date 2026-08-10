from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("operator_alpha_replay", ROOT / "run_operator_alpha_replay.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OperatorAlphaReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        MODULE.run()
        cls.report = json.loads((ROOT / "comparison_report_v1.json").read_text(encoding="utf-8"))
        cls.reconstruction = json.loads((ROOT / "trade_reconstruction_v1.json").read_text(encoding="utf-8"))
        cls.receipts = [json.loads(line) for line in (ROOT / "arm_receipts_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]

    def test_four_wins_reproduce_broker_total_and_return(self) -> None:
        summary = self.reconstruction["four_win_summary"]
        self.assertAlmostEqual(summary["after_cost_net_jpy"], 5052.0833, places=7)
        self.assertAlmostEqual(summary["return_fraction"], 0.019873737485045204, places=12)

    def test_every_frozen_fill_has_its_exact_market_order(self) -> None:
        packet = json.loads((ROOT / "source_transactions_v1.json").read_text(encoding="utf-8"))
        by_id = {row["id"]: row for row in packet["transactions"]}
        for trade in self.reconstruction["trades"]:
            for fill_id in (trade["entry_fill_id"], trade["close_fill_id"]):
                fill = by_id[fill_id]
                order = by_id[fill["orderID"]]
                self.assertEqual(order["type"], "MARKET_ORDER")
                self.assertEqual(order["time"], fill["time"])

    def test_same_six_rows_are_used_by_every_arm(self) -> None:
        cohorts = [tuple(value["cohort_ids"]) for value in self.report["arms"].values()]
        self.assertEqual(len(cohorts), 4)
        self.assertEqual(len(set(cohorts)), 1)
        self.assertEqual(len(cohorts[0]), 6)

    def test_failure_shape_is_bounded_not_relabelled(self) -> None:
        baseline = self.report["arms"]["BASELINE_ACTUAL"]["metrics"]
        operator = self.report["arms"]["OPERATOR_ALPHA"]["metrics"]
        self.assertEqual(baseline["max_drawdown_jpy"], 76200.0)
        self.assertLess(operator["max_drawdown_jpy"], baseline["max_drawdown_jpy"])
        self.assertLessEqual(operator["holding_time_max_seconds"], self.report["operator_parameters"]["max_hold_seconds"])

    def test_valid_profit_shape_is_still_allowed(self) -> None:
        operator = [row for row in self.receipts if row["arm"] == "OPERATOR_ALPHA"]
        self.assertTrue(any(row["exit_reason"] == "PROFIT_HARVEST" and row["after_cost_net_jpy"] > 0 for row in operator))

    def test_x_structure_is_not_promoted_as_edge(self) -> None:
        x_arm = self.report["arms"]["X_STRUCTURE"]["metrics"]
        self.assertEqual(x_arm["executed_or_retained"], 2)
        self.assertLess(x_arm["after_cost_net_jpy"], 0)
        x_contract = json.loads((ROOT / "x_claims_contract_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(len(x_contract["rejected_as_strategy_evidence"]), 3)

    def test_open_manual_unknown_boundary_is_no_touch(self) -> None:
        boundary = self.reconstruction["open_boundary"]
        self.assertEqual(boundary["entry_fill_id"], "473207")
        self.assertEqual(boundary["action"], "NO_TOUCH")

    def test_live_permission_is_never_granted(self) -> None:
        self.assertFalse(self.report["live_permission"])
        fusion = json.loads((ROOT / "fusion_table_v1.json").read_text(encoding="utf-8"))
        self.assertTrue(all(not row["live_permission"] for row in fusion["rows"]))

    def test_counterfactual_uses_after_cost_executable_side(self) -> None:
        operator = [row for row in self.receipts if row["arm"] == "OPERATOR_ALPHA"]
        self.assertEqual(len(operator), 6)
        self.assertTrue(all(row["exit_time_utc"] and row["receipt_sha256"] for row in operator))
        self.assertEqual(self.report["operator_parameters"]["slippage_stress"], "additional adverse half of observed S5 closing spread")

    def test_target_scenarios_are_not_guarantees(self) -> None:
        target = json.loads((ROOT / "target_arithmetic_v1.json").read_text(encoding="utf-8"))
        self.assertFalse(target["guarantee"])
        self.assertGreater(target["break_conditions"]["combined_loss_as_four_win_batches"], 15.0)


if __name__ == "__main__":
    unittest.main()
