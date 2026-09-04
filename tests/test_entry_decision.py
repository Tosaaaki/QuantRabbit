from __future__ import annotations

import copy
import math
import unittest
from datetime import datetime, timedelta, timezone

from quant_rabbit.entry_decision import (
    EntryDecisionError,
    build_entry_decision,
    compute_dynamic_units,
    decision_id_for,
    validate_entry_decision,
)


NOW = datetime(2026, 9, 4, 3, 0, tzinfo=timezone.utc)


def sizing(**overrides):
    values = {
        "daily_remaining": 10_000.0,
        "portfolio_allowance": 8_000.0,
        "nav_risk_ceiling": 12_000.0,
        "calibration_factor": 0.8,
        "drawdown_factor": 0.6,
        "correlation_factor": 0.9,
        "net_edge_factor": 0.7,
        "loss_per_unit_at_stop": 2.0,
        "margin_max_units": 10_000,
        "correlation_max_units": 9_000,
        "broker_max_units": 20_000,
        "exposures": (),
    }
    values.update(overrides)
    return compute_dynamic_units(**values)


def proposal(receipt=None):
    resolved = receipt or sizing()
    return {
        "pair": "EUR_USD",
        "side": "LONG",
        "method": "BREAKOUT",
        "vehicle": "STOP-ENTRY",
        "entry_price": 1.101,
        "stop_loss": 1.099,
        "take_profit": 1.105,
        "units": resolved["final_units"],
        "resource_claims": ["entry:cycle-1:EUR_USD"],
        "sizing_receipt": resolved,
    }


class DynamicSizingTests(unittest.TestCase):
    def test_exact_formula_and_receipt(self):
        receipt = sizing()
        # floor(min(10000, 8000, 12000) * min(.8,.6,.9,.7) / 2)
        self.assertEqual(receipt["risk_formula_units"], 2400)
        self.assertEqual(receipt["final_units"], 2400)
        self.assertEqual(receipt["base_risk_limiting_reasons"], ["portfolio_allowance"])
        self.assertEqual(receipt["factor_limiting_reasons"], ["drawdown_factor"])
        self.assertEqual(receipt["unit_limiting_reasons"], ["risk_formula_units"])
        self.assertIsNone(receipt["numeric_policy"]["maximum_units"])
        self.assertNotIn("target_trade_count_divisor", receipt["numeric_policy"])
        self.assertNotIn("allocation_multiplier", receipt["numeric_policy"])

    def test_caps_apply_after_formula_without_one_thousand_cap(self):
        large = sizing(
            daily_remaining=100_000,
            portfolio_allowance=100_000,
            nav_risk_ceiling=100_000,
            calibration_factor=1,
            drawdown_factor=1,
            correlation_factor=1,
            net_edge_factor=1,
            loss_per_unit_at_stop=1,
            margin_max_units=80_000,
            correlation_max_units=70_000,
            broker_max_units=60_000,
        )
        self.assertEqual(large["final_units"], 60_000)
        self.assertEqual(large["unit_limiting_reasons"], ["broker_max_units"])
        micro = sizing(
            daily_remaining=5,
            portfolio_allowance=5,
            nav_risk_ceiling=5,
            calibration_factor=1,
            drawdown_factor=1,
            correlation_factor=1,
            net_edge_factor=1,
            loss_per_unit_at_stop=2,
        )
        self.assertEqual(micro["final_units"], 2)

    def test_manual_and_unknown_exposure_reduce_caps_and_are_no_touch(self):
        receipt = sizing(
            margin_max_units=5_000,
            correlation_max_units=5_000,
            exposures=(
                {
                    "reference": "manual-1",
                    "owner": "manual",
                    "margin_units_equivalent": 1_500,
                    "correlation_units_equivalent": 200,
                },
                {
                    "reference": "tagless-1",
                    "owner": "mystery",
                    "margin_units_equivalent": 500,
                    "correlation_units_equivalent": 2_700,
                },
            ),
        )
        self.assertEqual(receipt["exposure_totals"]["margin_units_equivalent"], 2_000)
        self.assertEqual(
            receipt["exposure_totals"]["correlation_units_equivalent"], 2_900
        )
        self.assertEqual(receipt["effective_unit_caps"]["margin_max_units"], 3_000)
        self.assertEqual(
            receipt["effective_unit_caps"]["correlation_max_units"], 2_100
        )
        self.assertEqual(receipt["final_units"], 2_100)
        self.assertEqual(
            [row["owner_class"] for row in receipt["exposures"]],
            ["MANUAL", "UNKNOWN"],
        )
        self.assertTrue(
            all(row["management_action"] == "NO_TOUCH" for row in receipt["exposures"])
        )

    def test_rejects_nonpositive_nonfinite_and_sub_one_capacity(self):
        for field, value in (
            ("daily_remaining", 0),
            ("loss_per_unit_at_stop", -1),
            ("broker_max_units", math.inf),
            ("net_edge_factor", math.nan),
        ):
            with self.subTest(field=field):
                with self.assertRaises(EntryDecisionError):
                    sizing(**{field: value})
        with self.assertRaisesRegex(EntryDecisionError, "at least one unit"):
            sizing(
                daily_remaining=0.5,
                portfolio_allowance=0.5,
                nav_risk_ceiling=0.5,
                calibration_factor=1,
                drawdown_factor=1,
                correlation_factor=1,
                net_edge_factor=1,
                loss_per_unit_at_stop=1,
            )


class EntryDecisionTests(unittest.TestCase):
    def build(self, **overrides):
        values = {
            "action": "ENTER",
            "cycle_id": "cycle-1",
            "broker_epoch": "broker-epoch-7",
            "evidence_observed_at_utc": NOW - timedelta(seconds=10),
            "created_at_utc": NOW,
            "ttl_seconds": 600,
            "proposal": proposal(),
            "reasons": ("positive net edge",),
        }
        values.update(overrides)
        return build_entry_decision(**values)

    def test_enter_is_content_addressed_and_bound(self):
        decision = self.build()
        self.assertTrue(decision["decision_id"].startswith("qre_"))
        self.assertEqual(decision["decision_id"], decision_id_for(decision))
        self.assertEqual(len(decision["proposals"]), 1)
        self.assertEqual(
            validate_entry_decision(
                decision,
                expected_cycle_id="cycle-1",
                expected_broker_epoch="broker-epoch-7",
                now_utc=NOW + timedelta(seconds=5),
            ),
            decision,
        )

    def test_decision_with_manual_exposure_revalidates_exact_sizing(self):
        receipt = sizing(
            exposures=(
                {
                    "reference": "manual-1",
                    "owner": "manual",
                    "margin_units_equivalent": 100,
                    "correlation_units_equivalent": 200,
                },
            )
        )
        decision = self.build(proposal=proposal(receipt))
        validate_entry_decision(
            decision,
            expected_cycle_id="cycle-1",
            expected_broker_epoch="broker-epoch-7",
            now_utc=NOW,
        )

    def test_canonical_id_is_key_order_independent_and_binds_claims(self):
        decision = self.build()
        reordered = dict(reversed(list(decision.items())))
        self.assertEqual(decision_id_for(reordered), decision["decision_id"])
        changed = copy.deepcopy(decision)
        changed["proposals"][0]["resource_claims"] = ["entry:other:EUR_USD"]
        self.assertNotEqual(decision_id_for(changed), decision["decision_id"])

    def test_wait_and_request_evidence_have_no_proposal(self):
        wait = self.build(action="WAIT", proposal=None, reasons=("no edge",))
        self.assertEqual(wait["proposals"], [])
        request = self.build(
            action="REQUEST_EVIDENCE",
            proposal=None,
            requested_evidence=("fresh M5 close",),
        )
        self.assertEqual(request["proposals"], [])
        self.assertEqual(request["requested_evidence"], ["fresh M5 close"])
        for decision in (wait, request):
            validate_entry_decision(
                decision,
                expected_cycle_id="cycle-1",
                expected_broker_epoch="broker-epoch-7",
                now_utc=NOW + timedelta(seconds=5),
            )

    def test_action_and_proposal_cardinality_fail_closed(self):
        with self.assertRaisesRegex(EntryDecisionError, "unsupported entry action"):
            self.build(action="CLOSE")
        with self.assertRaisesRegex(EntryDecisionError, "exactly one"):
            self.build(proposal=None)
        with self.assertRaisesRegex(EntryDecisionError, "zero proposals"):
            self.build(action="WAIT")

        forged = self.build()
        forged["proposals"].append(copy.deepcopy(forged["proposals"][0]))
        forged["decision_id"] = decision_id_for(forged)
        with self.assertRaisesRegex(EntryDecisionError, "zero or one"):
            validate_entry_decision(
                forged,
                expected_cycle_id="cycle-1",
                expected_broker_epoch="broker-epoch-7",
                now_utc=NOW,
            )

    def test_rejects_id_cycle_epoch_and_freshness_mismatch(self):
        decision = self.build()
        tampered = copy.deepcopy(decision)
        tampered["proposals"][0]["units"] += 1
        with self.assertRaisesRegex(EntryDecisionError, "content address"):
            validate_entry_decision(
                tampered,
                expected_cycle_id="cycle-1",
                expected_broker_epoch="broker-epoch-7",
                now_utc=NOW,
            )
        with self.assertRaisesRegex(EntryDecisionError, "another cycle"):
            validate_entry_decision(
                decision,
                expected_cycle_id="cycle-2",
                expected_broker_epoch="broker-epoch-7",
                now_utc=NOW,
            )
        with self.assertRaisesRegex(EntryDecisionError, "another broker epoch"):
            validate_entry_decision(
                decision,
                expected_cycle_id="cycle-1",
                expected_broker_epoch="broker-epoch-8",
                now_utc=NOW,
            )
        with self.assertRaisesRegex(EntryDecisionError, "stale"):
            validate_entry_decision(
                decision,
                expected_cycle_id="cycle-1",
                expected_broker_epoch="broker-epoch-7",
                now_utc=NOW + timedelta(seconds=601),
            )

    def test_rejects_rehashed_but_invalid_sizing_and_forbidden_fields(self):
        forged = self.build()
        forged["proposals"][0]["sizing_receipt"]["final_units"] += 1
        forged["proposals"][0]["units"] += 1
        forged["decision_id"] = decision_id_for(forged)
        with self.assertRaisesRegex(EntryDecisionError, "does not reproduce"):
            validate_entry_decision(
                forged,
                expected_cycle_id="cycle-1",
                expected_broker_epoch="broker-epoch-7",
                now_utc=NOW,
            )
        with self.assertRaisesRegex(EntryDecisionError, "forbidden sizing field"):
            self.build(proposal={**proposal(), "allocation_multiplier": 0.5})


if __name__ == "__main__":
    unittest.main()
