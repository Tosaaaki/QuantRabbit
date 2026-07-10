from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relpath: str) -> dict:
    return json.loads((ROOT / relpath).read_text(encoding="utf-8"))


class ScoutModeContractTests(unittest.TestCase):
    def test_contract_is_approval_bound_limit_only_and_non_executable(self) -> None:
        payload = load_json("data/operator_approved_scout_mode_contract.json")

        self.assertEqual(
            payload["candidate"]["lane_id"],
            "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
        )
        self.assertEqual(payload["mode"], "proof-collection scout")
        self.assertFalse(payload["active"])
        self.assertTrue(payload["historical_only"])
        self.assertEqual(payload["superseded_by_schema_v2"], "config/predictive_scout_policy.json")
        self.assertEqual(payload["order_contract"]["allowed_order_types"], ["LIMIT"])
        self.assertIn("MARKET", payload["order_contract"]["prohibited_order_types"])
        self.assertTrue(payload["order_contract"]["no_market_chase"])
        self.assertTrue(payload["order_contract"]["one_order_only"])
        self.assertEqual(payload["order_contract"]["default_units"], 1000)
        self.assertTrue(payload["operator_approval"]["required"])
        self.assertFalse(payload["operator_approval"]["present"])
        self.assertFalse(payload["current_state_invariants"]["order_allowed_now"])
        self.assertFalse(payload["current_state_invariants"]["execution_flags_enabled"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_contract_requires_fresh_loss_recalculation_and_exact_approval_text(self) -> None:
        payload = load_json("data/operator_approved_scout_mode_contract.json")

        max_loss = payload["order_contract"]["max_loss_jpy_cap"]
        self.assertTrue(max_loss["must_recalculate_from_fresh_quote"])
        self.assertEqual(max_loss["executable_cap_jpy"], 200)
        self.assertEqual(max_loss["reference_estimate_jpy_at_1000u"], 125.0)
        self.assertTrue(max_loss["reference_estimate_is_not_permission"])
        self.assertEqual(
            payload["operator_approval"]["exact_required_text"],
            "I approve one AUD_JPY SHORT BREAKOUT_FAILURE LIMIT proof-collection scout, max loss 200 JPY, units 1000, this run only.",
        )
        self.assertEqual(payload["operator_approval"]["approved_max_loss_jpy_cap"], 200)
        self.assertEqual(payload["operator_approval"]["approved_units_cap"], 1000)
        self.assertTrue(max_loss["fresh_recalculation_must_be_less_than_or_equal_to_executable_cap_jpy"])

    def test_contract_preflight_stop_conditions_and_manual_protection_are_complete(self) -> None:
        payload = load_json("data/operator_approved_scout_mode_contract.json")

        preflight = {item["name"] for item in payload["required_preflight"]}
        self.assertEqual(
            {
                "fresh broker snapshot",
                "fresh quote",
                "RiskEngine pass",
                "LiveOrderGateway pass",
                "GPT verifier pass",
                "guardian/operator-review pass",
                "profitability blockers acknowledged",
                "manual EUR_USD 472987 protected",
            },
            preflight,
        )
        for stop in (
            "stale quote",
            "spread too wide",
            "forecast mismatch",
            "RiskEngine fail",
            "Gateway fail",
            "GPT verifier fail",
            "guardian/operator review missing",
            "execution flags not explicitly approved",
            "broker last_transaction_id unexpected",
        ):
            self.assertIn(stop, payload["stop_conditions"])
        manual = payload["manual_protection"]
        self.assertEqual(manual["trade_id"], "472987")
        self.assertEqual(manual["protected_take_profit_order_id"], "472998")
        self.assertTrue(manual["do_not_touch_trade"])
        self.assertTrue(manual["do_not_modify_tp"])
        self.assertTrue(manual["do_not_add_sl"])
        self.assertTrue(manual["do_not_close"])

    def test_readiness_remains_not_approved_and_blocked(self) -> None:
        payload = load_json("data/scout_mode_readiness_check.json")

        self.assertEqual(
            payload["classification_values"],
            [
                "NOT_APPROVED",
                "STALE_CURRENT_PACKET",
                "REJECTED_CURRENT_PACKET",
                "APPROVAL_REQUIRED",
                "STALE_NOT_READY_FOR_OPERATOR_APPROVAL",
                "BLOCKED",
            ],
        )
        self.assertEqual(payload["readiness"]["current"], "STALE_CURRENT_PACKET")
        self.assertEqual(payload["readiness"]["contract_state"], "STALE_NOT_READY_FOR_OPERATOR_APPROVAL")
        self.assertEqual(payload["readiness"]["next_required"], "SELECT_NEXT_CANDIDATE_OR_COLLECT_NEW_PROOF")
        self.assertEqual(payload["readiness"]["execution_state"], "BLOCKED")
        self.assertFalse(payload["approval"]["present"])
        self.assertFalse(payload["approval"]["detected_as_operator_approval"])
        self.assertEqual(payload["scout_execution_decision"]["order_sent"], False)
        self.assertEqual(payload["scout_execution_decision"]["reason"], "STALE_CANDIDATE_AND_APPROVAL_MISSING")
        self.assertTrue(payload["fresh_quote_reference_only"]["reference_is_stale_for_execution"])
        self.assertTrue(payload["fresh_quote_reference_only"]["must_recalculate_from_fresh_quote"])
        self.assertTrue(payload["expected_outcome"]["no_order"])
        self.assertTrue(payload["expected_outcome"]["no_execution_flags"])
        self.assertTrue(payload["expected_outcome"]["no_LIVE_READY"])
        self.assertTrue(payload["expected_outcome"]["normal_routing_remains_BLOCKED"])
        self.assertFalse(payload["active"])
        self.assertTrue(payload["historical_only"])
        self.assertEqual(payload["superseded_by_schema_v2"], "config/predictive_scout_policy.json")

    def test_markdown_reports_approval_boundary(self) -> None:
        contract = (ROOT / "docs/operator_approved_scout_mode_contract.md").read_text(encoding="utf-8")
        readiness = (ROOT / "docs/scout_mode_readiness_check.md").read_text(encoding="utf-8")

        self.assertIn("Order type: `LIMIT` only.", contract)
        self.assertIn("No market chase", contract)
        self.assertIn("No auto retry", contract)
        self.assertIn("I approve one AUD_JPY SHORT BREAKOUT_FAILURE LIMIT proof-collection scout", contract)
        self.assertIn("EUR_USD `472987` protected", contract)
        self.assertIn("Current readiness: `STALE_CURRENT_PACKET`", readiness)
        self.assertIn("Candidate is absent from refreshed live order_intents", readiness)
        self.assertIn("Execution state: `BLOCKED`", readiness)
        self.assertIn("No order.", readiness)

    def test_scout_execution_receipt_blocks_without_approval(self) -> None:
        payload = load_json("data/scout_execution_receipt.json")

        self.assertEqual(
            payload["candidate_id"],
            "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT",
        )
        self.assertFalse(payload["approval"]["detected"])
        self.assertEqual(payload["execution_decision"]["execution_state"], "BLOCKED")
        self.assertFalse(payload["execution_decision"]["order_sent"])
        self.assertEqual(payload["execution_decision"]["rejection_reason"], "STALE_CANDIDATE_AND_APPROVAL_MISSING")
        self.assertFalse(payload["execution_decision"]["normal_routing_created"])
        self.assertFalse(payload["execution_decision"]["execution_flags_enabled"])
        self.assertTrue(payload["safety_constraints"]["limit_only"])
        self.assertFalse(payload["safety_constraints"]["market_order_allowed"])
        self.assertFalse(payload["active"])
        self.assertTrue(payload["historical_only"])
        self.assertEqual(payload["superseded_by_schema_v2"], "config/predictive_scout_policy.json")
        self.assertEqual(payload["preflight"]["candidate_id_status"], "ABSENT_FROM_CURRENT_ORDER_INTENTS")
        self.assertEqual(payload["preflight"]["last_transaction_id_expected"], "472998")
        self.assertEqual(payload["local_broker_evidence"]["tp_472998"]["state"], "PENDING")

    def test_active_eurusd_scout_plan_uses_schema_v2_current_nav_integer_sizing(self) -> None:
        payload = load_json("data/eurusd_short_breakout_failure_scout_plan.json")

        self.assertEqual(payload["schema_version"], "eurusd_short_breakout_failure_scout_plan_v2")
        self.assertIsNone(payload["max_loss_jpy_cap"])
        self.assertEqual(payload["max_loss_jpy_cap_mode"], "DYNAMIC_CURRENT_NAV_EXACT_VEHICLE_TIER")
        sizing = payload["risk_sizing_contract"]
        self.assertEqual(sizing["schema_version"], 2)
        self.assertEqual(sizing["min_production_lot_units"], 1)
        self.assertTrue(sizing["positive_integer_units_allowed"])
        self.assertTrue(sizing["sub_1000_units_allowed"])
        self.assertTrue(sizing["fresh_regeneration_required"])
        self.assertIsNone(sizing["current_executable_units"])
        min_lot = payload["min_lot_feasibility"]
        self.assertEqual(min_lot["min_production_lot_units"], 1)
        self.assertEqual(min_lot["units_mode"], "DYNAMIC_CURRENT_NAV_SL_RISK")
        self.assertIsNone(min_lot["proposed_scout_units"])
        historical = payload["historical_reference"]
        self.assertEqual(historical["legacy_proposed_scout_units"], 3000)
        self.assertEqual(historical["legacy_min_production_lot_units"], 1000)
        self.assertFalse(historical["execution_eligible"])


if __name__ == "__main__":
    unittest.main()
