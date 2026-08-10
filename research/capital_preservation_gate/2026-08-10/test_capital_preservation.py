from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("capital_preservation", HERE / "capital_preservation.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
DecisionInput = MODULE.DecisionInput
RiskPolicy = MODULE.RiskPolicy
evaluate = MODULE.evaluate
REQUIRED_STAGES = MODULE.REQUIRED_STAGES


def good(**changes):
    values = dict(
        decision_id="fixture-1",
        decision_time="2026-08-10T00:00:00Z",
        source_sha="a" * 64,
        stage_coverage={stage: True for stage in REQUIRED_STAGES},
        equity_jpy=200_000.0,
        peak_equity_jpy=200_000.0,
        daily_gross_loss_spent_jpy=0.0,
        candidate_loss_bound_jpy=400.0,
        expected_after_cost_lcb_jpy=50.0,
    )
    values.update(changes)
    return DecisionInput(**values)


class CapitalPreservationTest(unittest.TestCase):
    def test_legacy_missing_margin_cost_and_unwind_fail_closed(self):
        coverage = {stage: True for stage in REQUIRED_STAGES}
        for stage in ("slippage_fee_financing", "margin_exposure_concurrency", "exit_unwind"):
            coverage[stage] = False
        receipt = evaluate(good(stage_coverage=coverage))
        self.assertEqual(receipt["action"], "WAIT")
        self.assertFalse(receipt["new_exposure_permitted"])

    def test_fully_evidenced_positive_bounded_shape_is_allowed(self):
        receipt = evaluate(good())
        self.assertEqual(receipt["action"], "TRADE")
        self.assertTrue(receipt["new_exposure_permitted"])

    def test_nonpositive_edge_is_skip_not_trade(self):
        receipt = evaluate(good(expected_after_cost_lcb_jpy=0.0))
        self.assertEqual(receipt["action"], "SKIP")

    def test_missing_loss_bound_is_wait(self):
        receipt = evaluate(good(candidate_loss_bound_jpy=None))
        self.assertEqual(receipt["action"], "WAIT")

    def test_daily_gross_loss_budget_is_non_refillable(self):
        receipt = evaluate(good(daily_gross_loss_spent_jpy=2_000.0))
        self.assertEqual(receipt["action"], "WAIT")
        self.assertIn("DAILY_GROSS_LOSS_BUDGET_EXHAUSTED", receipt["reason_codes"])

    def test_candidate_loss_above_effective_cap_is_wait(self):
        receipt = evaluate(good(candidate_loss_bound_jpy=501.0))
        self.assertEqual(receipt["action"], "WAIT")

    def test_drawdown_lock_blocks_new_exposure(self):
        receipt = evaluate(good(equity_jpy=189_999.0, peak_equity_jpy=200_000.0))
        self.assertEqual(receipt["action"], "WAIT")
        self.assertIn("DRAWDOWN_LOCK_REACHED", receipt["reason_codes"])

    def test_receipt_is_deterministic_and_has_no_outcome_input(self):
        first = evaluate(good())
        second = evaluate(good())
        self.assertEqual(first, second)
        self.assertFalse(first["realized_outcome_used"])
        self.assertNotIn("realized_pl", first)

    def test_existing_position_routes_to_manage(self):
        receipt = evaluate(good(existing_position=True))
        self.assertEqual(receipt["action"], "MANAGE")


if __name__ == "__main__":
    unittest.main()
