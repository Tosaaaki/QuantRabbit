from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from quant_rabbit.decision_adjudication import (
    ADJUDICATION_ID_PREFIX,
    EXIT_ID_PREFIX,
    DecisionAdjudicationError,
    adjudicate_decisions,
    canonical_content_id,
    verify_adjudication_receipt,
)
from quant_rabbit.entry_decision import build_entry_decision, compute_dynamic_units
from quant_rabbit.exit_decision import ExitDecision, OwnerBinding


NOW = datetime(2026, 9, 4, 3, 0, tzinfo=timezone.utc)
CYCLE = "cycle-20260904T0300Z"
EPOCH = "broker-transaction-4200"


class DecisionAdjudicationTests(unittest.TestCase):
    def test_emergency_exit_precedes_other_exit_and_entry(self) -> None:
        entry = _entry(pair="EUR_USD", campaign_id="campaign-a")
        ordinary = _exit(
            action="REPLACE_TP",
            trade_id="trade-2",
            pair="GBP_USD",
            campaign_id="campaign-b",
        )
        emergency = _exit(
            action="REDUCE",
            trade_id="trade-1",
            pair="USD_JPY",
            campaign_id="campaign-c",
            emergency_eligible=True,
        )

        receipt = _adjudicate(entries=[entry], exits=[ordinary, emergency])

        self.assertEqual(receipt["selected_proposal_id"], emergency["decision_id"])
        self.assertEqual(receipt["selected_action"], "REDUCE")
        self.assertEqual(receipt["mutation_count"], 1)
        self.assertTrue(receipt["require_fresh_broker_readback_after_mutation"])
        self.assertEqual(
            {row["proposal_id"] for row in receipt["rejected_proposals"]},
            {entry["decision_id"], ordinary["decision_id"]},
        )

    def test_other_exit_mutation_precedes_entry(self) -> None:
        entry = _entry(pair="EUR_USD")
        exit_decision = _exit(action="TIGHTEN_SL", pair="GBP_USD")

        receipt = _adjudicate(entries=[entry], exits=[exit_decision])

        self.assertEqual(receipt["selected_proposal_id"], exit_decision["decision_id"])
        self.assertEqual(receipt["selected_kind"], "EXIT")

    def test_same_trade_pair_or_campaign_claims_serialize(self) -> None:
        cases = (
            (
                _exit(trade_id="trade-shared", pair="EUR_USD", campaign_id="campaign-a"),
                _exit(trade_id="trade-shared", pair="GBP_USD", campaign_id="campaign-b"),
                "position:trade-shared",
            ),
            (
                _exit(trade_id="trade-a", pair="EUR_USD", campaign_id="campaign-a"),
                _exit(trade_id="trade-b", pair="EUR_USD", campaign_id="campaign-b"),
                "pair:EUR_USD",
            ),
            (
                _exit(trade_id="trade-a", pair="EUR_USD", campaign_id="campaign-shared"),
                _exit(trade_id="trade-b", pair="GBP_USD", campaign_id="campaign-shared"),
                "campaign:campaign-shared",
            ),
        )
        for first, second, expected_claim in cases:
            with self.subTest(expected_claim=expected_claim):
                receipt = _adjudicate(exits=[second, first])

                self.assertEqual(
                    receipt["selected_proposal_id"],
                    min(first["decision_id"], second["decision_id"]),
                )
                rejected = receipt["rejected_proposals"][0]
                self.assertEqual(rejected["reason"], "RESOURCE_CONFLICT")
                self.assertIn(expected_claim, rejected["conflicting_claims"])

    def test_deterministic_order_and_identical_input_hash(self) -> None:
        first = _entry(pair="EUR_USD", campaign_id="campaign-a")
        second = _entry(pair="GBP_USD", campaign_id="campaign-b")

        left = _adjudicate(entries=[first, second])
        right = _adjudicate(entries=[second, first])

        self.assertEqual(left, right)
        self.assertTrue(left["adjudication_id"].startswith(ADJUDICATION_ID_PREFIX))
        verify_adjudication_receipt(left)

    def test_mismatched_epoch_and_cycle_fail_closed(self) -> None:
        wrong_epoch = _entry(broker_epoch="other-epoch")
        with self.assertRaises(DecisionAdjudicationError) as epoch_error:
            _adjudicate(entries=[wrong_epoch])
        self.assertEqual(epoch_error.exception.code, "BROKER_EPOCH_MISMATCH")

        wrong_exit_epoch = _exit(broker_epoch="other-epoch")
        with self.assertRaises(DecisionAdjudicationError) as exit_epoch_error:
            _adjudicate(exits=[wrong_exit_epoch])
        self.assertEqual(exit_epoch_error.exception.code, "BROKER_EPOCH_MISMATCH")

        wrong_cycle = _entry(cycle_id="other-cycle")
        with self.assertRaises(DecisionAdjudicationError) as cycle_error:
            _adjudicate(entries=[wrong_cycle])
        self.assertEqual(cycle_error.exception.code, "CYCLE_ID_MISMATCH")

    def test_stale_proposal_fails_closed(self) -> None:
        stale = _entry(
            created_at=NOW - timedelta(minutes=11),
            ttl_seconds=600,
        )

        with self.assertRaises(DecisionAdjudicationError) as caught:
            _adjudicate(entries=[stale])

        self.assertEqual(caught.exception.code, "PROPOSAL_STALE")

        expired_exit = _exit(
            created_at=NOW - timedelta(minutes=5),
            expires_at=NOW,
        )
        with self.assertRaises(DecisionAdjudicationError) as exit_caught:
            _adjudicate(exits=[expired_exit])
        self.assertEqual(exit_caught.exception.code, "PROPOSAL_STALE")

    def test_duplicate_sealed_decision_fails_closed(self) -> None:
        decision = _entry()

        with self.assertRaises(DecisionAdjudicationError) as caught:
            _adjudicate(entries=[decision, decision])

        self.assertEqual(caught.exception.code, "DUPLICATE_PROPOSAL_ID")

    def test_two_entries_select_one_by_content_id(self) -> None:
        first = _entry(pair="EUR_USD", campaign_id="campaign-a")
        second = _entry(pair="GBP_USD", campaign_id="campaign-b")

        receipt = _adjudicate(entries=[second, first])

        self.assertEqual(
            receipt["selected_proposal_id"],
            min(first["decision_id"], second["decision_id"]),
        )
        self.assertEqual(receipt["mutation_count"], 1)
        self.assertEqual(len(receipt["rejected_proposals"]), 1)
        self.assertEqual(
            receipt["rejected_proposals"][0]["reason"],
            "MAX_ONE_MUTATION_PER_CYCLE",
        )

    def test_close_and_reverse_entry_cannot_share_cycle(self) -> None:
        close = _exit(
            action="CLOSE_ALL",
            pair="EUR_USD",
        )
        reverse = _entry(pair="EUR_USD", side="SHORT")

        receipt = _adjudicate(entries=[reverse], exits=[close])

        self.assertEqual(receipt["selected_proposal_id"], close["decision_id"])
        rejected = next(
            row for row in receipt["rejected_proposals"] if row["proposal_id"] == reverse["decision_id"]
        )
        self.assertEqual(
            rejected["reason"],
            "REVERSE_ENTRY_REQUIRES_FRESH_BROKER_READBACK",
        )
        self.assertTrue(receipt["require_fresh_broker_readback_after_mutation"])

    def test_wait_and_request_evidence_yield_no_mutation(self) -> None:
        wait = _entry(action="WAIT", nested_proposal=None)
        request = _exit(action="REQUEST_EVIDENCE")

        receipt = _adjudicate(entries=[wait], exits=[request])

        self.assertIsNone(receipt["selected_proposal_id"])
        self.assertEqual(receipt["resource_claims"], [])
        self.assertEqual(receipt["mutation_count"], 0)
        self.assertFalse(receipt["require_fresh_broker_readback_after_mutation"])
        self.assertEqual(
            {row["reason"] for row in receipt["rejected_proposals"]},
            {"NON_MUTATING_ACTION"},
        )

    def test_manual_and_unknown_exit_claims_are_no_touch(self) -> None:
        manual = _exit(owner_kind="MANUAL", trade_id="manual-1", pair="USD_JPY")
        unknown = _exit(owner_kind="UNKNOWN", trade_id="unknown-1", pair="GBP_JPY")

        receipt = _adjudicate(exits=[manual, unknown])

        self.assertIsNone(receipt["selected_proposal_id"])
        self.assertEqual(receipt["mutation_count"], 0)
        self.assertEqual(
            {row["reason"] for row in receipt["rejected_proposals"]},
            {"NO_TOUCH_OWNER"},
        )

    def test_tampered_content_id_and_claim_binding_fail_closed(self) -> None:
        tampered = _entry()
        tampered["proposals"][0]["units"] = 999
        with self.assertRaises(DecisionAdjudicationError) as id_error:
            _adjudicate(entries=[tampered])
        self.assertEqual(id_error.exception.code, "PROPOSAL_ID_MISMATCH")

        bad_claim = _entry(resource_claims=["pair:GBP_USD"])
        with self.assertRaises(DecisionAdjudicationError) as claim_error:
            _adjudicate(entries=[bad_claim])
        self.assertEqual(claim_error.exception.code, "CLAIM_BINDING_MISMATCH")

    def test_tampered_adjudication_receipt_is_rejected(self) -> None:
        receipt = _adjudicate(entries=[_entry()])
        changed = copy.deepcopy(receipt)
        changed["selected_action"] = "WAIT"

        with self.assertRaises(DecisionAdjudicationError) as caught:
            verify_adjudication_receipt(changed)

        self.assertEqual(caught.exception.code, "ADJUDICATION_ID_MISMATCH")


def _adjudicate(
    *,
    entries: list[dict[str, Any]] | None = None,
    exits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return adjudicate_decisions(
        cycle_id=CYCLE,
        broker_epoch=EPOCH,
        entry_proposals=entries or [],
        exit_proposals=exits or [],
        now=NOW,
    )


def _entry(
    *,
    action: str = "ENTER",
    cycle_id: str = CYCLE,
    broker_epoch: str = EPOCH,
    pair: str = "EUR_USD",
    side: str = "LONG",
    campaign_id: str | None = "campaign-a",
    resource_claims: list[str] | None = None,
    nested_proposal: dict[str, Any] | None | object = ...,
    created_at: datetime | None = None,
    ttl_seconds: int = 600,
) -> dict[str, Any]:
    resolved_created_at = created_at or NOW
    if nested_proposal is ...:
        sizing_receipt = compute_dynamic_units(
            daily_remaining=100.0,
            portfolio_allowance=100.0,
            nav_risk_ceiling=100.0,
            calibration_factor=1.0,
            drawdown_factor=1.0,
            correlation_factor=1.0,
            net_edge_factor=1.0,
            loss_per_unit_at_stop=1.0,
            margin_max_units=100.0,
            correlation_max_units=100.0,
            broker_max_units=100.0,
        )
        nested_proposal = {
            "pair": pair,
            "side": side,
            "units": 100,
            "sizing_receipt": sizing_receipt,
            "campaign_id": campaign_id,
            "resource_claims": resource_claims
            if resource_claims is not None
            else [f"entry:{cycle_id}:{pair}"],
        }
    return build_entry_decision(
        action=action,
        cycle_id=cycle_id,
        broker_epoch=broker_epoch,
        evidence_observed_at_utc=resolved_created_at,
        proposal=nested_proposal,
        requested_evidence=("fresh evidence",) if action == "REQUEST_EVIDENCE" else (),
        reasons=("adjudication integration test",),
        ttl_seconds=ttl_seconds,
        created_at_utc=resolved_created_at,
    )


def _exit(
    *,
    action: str = "REPLACE_TP",
    cycle_id: str = CYCLE,
    broker_epoch: str = EPOCH,
    trade_id: str = "trade-1",
    pair: str = "EUR_USD",
    campaign_id: str = "campaign-a",
    owner_kind: str = "AI_SYSTEM",
    emergency_eligible: bool = False,
    created_at: datetime = NOW,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    geometry: dict[str, Any] = {}
    if action == "REDUCE":
        geometry["units"] = 25
    elif action == "TIGHTEN_SL":
        geometry["stop_loss"] = "1.095"
    elif action == "REPLACE_TP":
        geometry["take_profit"] = "1.125"

    decision = ExitDecision.create(
        action=action,
        cycle_id=cycle_id,
        broker_epoch=broker_epoch,
        position_revision=f"revision-{trade_id}",
        trade_id=trade_id,
        instrument=pair,
        owner_binding=OwnerBinding(
            "AI_SYSTEM",
            "ai-trader",
            f"client-{trade_id}",
            campaign_id,
        ),
        created_at_utc=created_at,
        expires_at_utc=expires_at or created_at + timedelta(minutes=10),
        emergency_eligible=emergency_eligible,
        reason="sealed exit integration test",
        **geometry,
    ).to_dict()
    if owner_kind != "AI_SYSTEM":
        # The Exit builder itself refuses NO_TOUCH ownership.  This negative
        # fixture changes only that sealed field and recomputes its public qrx
        # content address so the adjudicator must reject it as ineligible.
        decision["owner_binding"]["owner_kind"] = owner_kind
        decision["decision_id"] = canonical_content_id(decision, prefix=EXIT_ID_PREFIX)
    return decision


if __name__ == "__main__":
    unittest.main()
