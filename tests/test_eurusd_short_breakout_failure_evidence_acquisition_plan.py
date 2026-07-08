from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data" / "eurusd_short_breakout_failure_evidence_acquisition_plan.json"
DOC_PATH = ROOT / "docs" / "eurusd_short_breakout_failure_evidence_acquisition_plan.md"


class EurUsdShortBreakoutFailureEvidenceAcquisitionPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.doc = DOC_PATH.read_text(encoding="utf-8")

    def test_required_schema_and_safety_invariants(self) -> None:
        allowed_statuses = {
            "EVIDENCE_PLAN_READY",
            "EVIDENCE_PLAN_BLOCKED_STALE_ARTIFACTS",
            "EVIDENCE_PLAN_BLOCKED_NO_SOURCE",
            "EVIDENCE_PLAN_BLOCKED_OPERATOR_ONLY",
        }
        self.assertIn(self.plan["status"], allowed_statuses)
        self.assertEqual(self.plan["target_shape"], "EUR_USD|SHORT|BREAKOUT_FAILURE")
        self.assertTrue(self.plan["read_only"])
        self.assertEqual(self.plan["live_side_effects"], [])
        self.assertFalse(self.plan["live_permission_allowed"])
        self.assertEqual(
            self.plan["current_tp_proof"],
            {
                "wins": 20,
                "losses": 0,
                "proof_floor": 20,
                "remaining_samples": 0,
                "proof_floor_reached": True,
                "net_jpy": 12865.8232,
                "expectancy_jpy_per_trade": 643.2912,
                "source": "data/payoff_shape_diagnosis.json canonical_proof_reconciliation + data/eurusd_short_breakout_failure_proof_floor_update.json",
            },
        )
        self.assertGreaterEqual(self.plan["current_tp_proof"]["proof_floor"], 20)
        self.assertEqual(
            self.plan["canonical_proof_reconciliation"]["legacy_limit_trade_ids"],
            ["469278", "469427", "469898"],
        )
        self.assertEqual(self.plan["canonical_proof_reconciliation"]["scope"], "broad_take_profit_order_proof_only_not_exact_limit_sample_floor")
        self.assertEqual(self.plan["exact_limit_harvest_proof"]["samples"], 4)
        self.assertEqual(self.plan["exact_limit_harvest_proof"]["remaining_limit_samples"], 16)
        self.assertFalse(self.plan["exact_limit_harvest_proof"]["live_grade_candidate"])

        forbidden_phrases = (
            "--send",
            "--confirm-live",
            "QR_LIVE_ENABLED=1",
            "run-autotrade-live.sh",
            "stage-live-order",
            "autotrade-cycle",
            " launchctl ",
        )
        for command in self.plan["next_read_only_commands"]:
            text = command["command"]
            for phrase in forbidden_phrases:
                self.assertNotIn(phrase, text)
            self.assertEqual(command["live_side_effects"], [])

        do_not_do = "\n".join(self.plan["do_not_do"]).lower()
        for required in (
            "live orders",
            "cancel",
            "close",
            "launchd",
            "lower the proof floor below 20",
            "negative expectancy",
            "remaining_to_4x",
            "operator decision",
            "secrets",
        ):
            self.assertIn(required, do_not_do)

    def test_missing_evidence_answers_are_explicit(self) -> None:
        missing_by_code = {item["code"]: item for item in self.plan["missing_evidence"]}
        for code in (
            "BROAD_TP_SAMPLE_GAP_CLEARED",
            "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
            "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
            "MARKET_CLOSE_LEAK_PRESENT",
            "NEGATIVE_EXPECTANCY_ACTIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
        ):
            self.assertIn(code, missing_by_code)

        sample_plan = self.plan["sample_acquisition_plan"]
        self.assertIn("legacy rows 469278 / 469427 / 469898", sample_plan["question_1_existing_history_available"])
        self.assertIn("No live-SCOUTless method", sample_plan["question_2_if_existing_history_is_insufficient"])
        self.assertTrue(missing_by_code["BROAD_TP_SAMPLE_GAP_CLEARED"]["existing_history_currently_proves_gap_closed"])
        self.assertFalse(missing_by_code["BROAD_TP_SAMPLE_GAP_CLEARED"]["can_create_new_true_samples_without_live_scout"])

        spread_plan = self.plan["spread_proof_plan"]
        self.assertEqual(spread_plan["current_status"], "LIMIT_S5_BIDASK_REPLAY_PASSED_STILL_BLOCKED")
        self.assertTrue(
            any("overwrite runtime rules" in item.lower() or "relax live gates" in item.lower() for item in spread_plan["failure_condition"])
        )

    def test_month_scale_market_close_and_transition_do_not_grant_permission(self) -> None:
        month = self.plan["month_scale_replay_plan"]
        self.assertEqual(month["current_evidence"]["global_status"], "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE")
        self.assertLess(month["current_evidence"]["current_residual_pl_jpy"], 0)
        self.assertFalse(month["current_evidence"]["direct_target_shape_blocker"])

        leak = self.plan["market_close_leak_plan"]
        self.assertEqual(leak["current_evidence"]["market_close_losses"], 10)
        self.assertLess(leak["current_evidence"]["market_close_net_jpy"], 0)
        self.assertTrue(leak["current_evidence"]["partial_tp_runner_is_diagnostic_only"])

        transition = self.plan["success_condition"]["question_6_transition"]
        self.assertEqual(transition["from"], "BROAD_TP_PROOF_RECONCILED_EXACT_LIMIT_UNDERSAMPLED")
        self.assertEqual(transition["evidence_only_next_state"], "EVIDENCE_ACQUISITION_LIMIT_SAMPLE_MINING_AND_TOUCH_LAG_RECONCILIATION")
        self.assertEqual(
            transition["operator_review_vocabulary_if_explicit_clearance_also_present"],
            "SCOUT_APPROVE_RECOMMENDED_ONLY_AFTER_NON_EVIDENCE_GATES_CLEAR",
        )
        self.assertFalse(transition["live_permission_after_transition"])

    def test_markdown_covers_required_questions(self) -> None:
        for phrase in (
            "Existing read-only history for the old broad TP sample gap",
            "If exact LIMIT history is insufficient",
            "Exact LIMIT S5 bid/ask replay",
            "Market-close leak split",
            "Month-scale negative",
            "Transition after broad proof reconciliation",
            "not live permission",
        ):
            self.assertIn(phrase, self.doc)


if __name__ == "__main__":
    unittest.main()
