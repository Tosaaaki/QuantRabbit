from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from quant_rabbit.ai_live_gateway import (
    AILiveGatewayError,
    broker_epoch_from_evidence_packet,
    build_live_gateway_artifacts,
    execute_ai_trade_candidate,
)
from quant_rabbit.entry_decision import (
    EntryDecisionError,
    build_entry_decision,
    compute_dynamic_units,
    decision_id_for,
)
from quant_rabbit.market_read_overlay import canonical_json_sha256


NOW = datetime(2026, 9, 4, 3, 0, tzinfo=timezone.utc)


class AILiveGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "guardian_action_receipt.json").write_text(
            json.dumps({"action": "NO_ACTION", "selected_pair": "EUR_USD"})
        )
        # A directory deliberately occupies the legacy path.  Any attempted
        # read of old bot intents therefore fails the test immediately.
        (self.root / "data" / "order_intents.json").mkdir()
        self.metrics = {
            "trades": 20,
            "wins": 20,
            "losses": 0,
            "net_jpy": 2000.0,
            "expectancy_jpy_per_trade": 100.0,
            "avg_win_jpy": 100.0,
            "avg_loss_jpy": 0.0,
            "unresolved_realized_trades": 0,
            "unresolved_realized_net_jpy": 0.0,
        }
        self.packet = self._packet()
        self.cost_floor = {"status": "PASSED", "proof_sha256": "c" * 64}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _packet(self, *, now: datetime = NOW) -> dict:
        sources = {
            "broker_snapshot": {
                "required": True,
                "status": "READY",
                "issue_code": None,
                "sha256": "a" * 64,
                "as_of_utc": now.isoformat(),
                "stale_after_utc": (now + timedelta(minutes=5)).isoformat(),
            },
            "capture_economics": {
                "required": True,
                "status": "READY",
                "issue_code": None,
                "sha256": "b" * 64,
                "as_of_utc": now.isoformat(),
                "stale_after_utc": (now + timedelta(hours=24)).isoformat(),
            },
        }
        segment = {
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "BREAKOUT",
            **{key: self.metrics[key] for key in (
                "trades", "wins", "losses", "net_jpy",
                "expectancy_jpy_per_trade", "avg_win_jpy", "avg_loss_jpy",
            )},
        }
        body = {
            "contract": "QR_AI_EVIDENCE_PACKET_V1",
            "schema_version": 1,
            "status": "READY",
            "sources": sources,
            "source_set_sha256": canonical_json_sha256(sources),
            "broker_epoch": {
                "as_of_utc": now.isoformat(),
                "source_sha256": "a" * 64,
                "last_transaction_id": "9001",
            },
            "broker": {
                "home_conversions": {"USD": 150.0},
                "exposure": {"positions": [], "pending_orders": []},
            },
            "portfolio": {
                "margin": {"nav_jpy": 100_000.0},
                "daily_target": {
                    "status": "ACTIVE",
                    "remaining_risk_budget_jpy": 10_000.0,
                },
            },
            "costs": {
                "spread_slippage_latency_swap_facts": [
                    {"field": "average_slippage_pips", "value": 0.2}
                ]
            },
            "net_edge_inputs": {"status": "OK", "segments": [segment]},
        }
        return {**body, "packet_sha256": canonical_json_sha256(body)}

    def _decision(self, *, packet: dict | None = None, units: int | None = None) -> dict:
        bound_packet = packet or self.packet
        sizing = compute_dynamic_units(
            daily_remaining=10_000,
            portfolio_allowance=8_000,
            nav_risk_ceiling=12_000,
            calibration_factor=0.8,
            drawdown_factor=0.6,
            correlation_factor=0.9,
            net_edge_factor=0.7,
            loss_per_unit_at_stop=0.3,
            margin_max_units=2_400,
            correlation_max_units=9_000,
            broker_max_units=20_000,
        )
        if units is not None:
            sizing["final_units"] = units
        selected_units = sizing["final_units"]
        proof = {
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "BREAKOUT",
            "vehicle": "STOP-ENTRY",
            "source_sha256": "b" * 64,
            **self.metrics,
            "net_edge_after_cost_jpy": 100.0,
        }
        proposal = {
            "pair": "EUR_USD",
            "side": "LONG",
            "method": "BREAKOUT",
            "vehicle": "STOP-ENTRY",
            "entry_price": 1.101,
            "stop_loss": 1.099,
            "take_profit": 1.105,
            "units": selected_units,
            "resource_claims": ["entry:cycle-1:EUR_USD"],
            "sizing_receipt": sizing,
            "evidence_binding": {
                "packet_sha256": bound_packet["packet_sha256"],
                "source_set_sha256": bound_packet["source_set_sha256"],
                "broker_epoch": broker_epoch_from_evidence_packet(bound_packet),
            },
            "net_edge_proof": proof,
            "cost_proof": {
                "packet_costs_sha256": canonical_json_sha256(bound_packet["costs"]),
                "execution_cost_floor_sha256": "c" * 64,
            },
            "rationale": "sealed positive edge after execution costs",
        }
        return build_entry_decision(
            action="ENTER",
            cycle_id="cycle-1",
            broker_epoch=broker_epoch_from_evidence_packet(bound_packet),
            evidence_observed_at_utc=NOW,
            created_at_utc=NOW,
            ttl_seconds=600,
            proposal=proposal,
            reasons=("positive net edge",),
        )

    def _build(self, decision: dict | None = None, packet: dict | None = None) -> dict:
        resolved_packet = packet or self.packet
        with (
            patch("quant_rabbit.ai_live_gateway.read_exact_vehicle_allocation_surface", return_value={"parse_status": "VALID"}),
            patch("quant_rabbit.ai_live_gateway.exact_vehicle_metrics_from_surface", return_value={("EUR_USD", "LONG", "BREAKOUT", "STOP"): self.metrics}),
            patch("quant_rabbit.ai_live_gateway.execution_cost_floor_from_surface", return_value=self.cost_floor),
        ):
            return build_live_gateway_artifacts(
                repo_root=self.root,
                entry_decision=decision or self._decision(packet=resolved_packet),
                evidence_packet=resolved_packet,
                run_id="cycle-1",
                generated_at=NOW,
                candidate_context={"model": "gpt-5.6-luna", "reasoning_effort": "max"},
            )

    def test_exact_ai_units_are_preserved_and_order_intents_are_not_read(self) -> None:
        built = self._build()
        intent = built["intents"]["results"][0]["intent"]
        self.assertEqual(intent["units"], 2400)
        self.assertEqual(intent["entry"], 1.101)
        self.assertEqual(intent["tp"], 1.105)
        self.assertEqual(intent["sl"], 1.099)
        allocation = built["verified_decision"]["decision"]["capital_allocation"]
        self.assertEqual(allocation["selected_units"], 2400)
        self.assertEqual(allocation["size_multiple"], 1.0)
        self.assertNotIn("allocation_multiplier", json.dumps(built))
        metadata = intent["metadata"]
        self.assertTrue(metadata["forecast_cycle_id"].startswith("pre-entry-forecast-refresh:"))
        self.assertEqual(metadata["forecast_target_price"], 1.105)
        self.assertEqual(metadata["forecast_invalidation_price"], 1.099)

    def test_old_candidate_and_allocation_multiplier_are_rejected(self) -> None:
        with self.assertRaisesRegex(AILiveGatewayError, "entry_decision"):
            execute_ai_trade_candidate(
                repo_root=self.root,
                state_root=self.root / "state",
                receipt={
                    "run_id": "old",
                    "decision": {
                        "action": "TRADE",
                        "orders": [{"allocation_multiplier": 0.75}],
                    },
                },
            )
        valid = self._decision()
        legacy = copy.deepcopy(valid)
        legacy["proposals"][0]["allocation_multiplier"] = 0.75
        # A legacy field cannot be assigned a valid qre id by the source contract.
        with self.assertRaises(EntryDecisionError):
            decision_id_for(legacy)

    def test_forged_sizing_receipt_is_blocked_even_when_qre_is_rehashed(self) -> None:
        forged = copy.deepcopy(self._decision())
        forged["proposals"][0]["sizing_receipt"]["final_units"] += 1
        forged["proposals"][0]["units"] += 1
        forged["decision_id"] = decision_id_for(forged)
        with self.assertRaisesRegex(AILiveGatewayError, "does not reproduce"):
            self._build(decision=forged)

    def test_forged_stop_loss_value_is_blocked_by_sealed_conversion(self) -> None:
        forged = copy.deepcopy(self._decision())
        forged["proposals"][0]["sizing_receipt"] = compute_dynamic_units(
            daily_remaining=10_000,
            portfolio_allowance=8_000,
            nav_risk_ceiling=12_000,
            calibration_factor=0.8,
            drawdown_factor=0.6,
            correlation_factor=0.9,
            net_edge_factor=0.7,
            loss_per_unit_at_stop=0.01,
            margin_max_units=1_000_000,
            correlation_max_units=1_000_000,
            broker_max_units=1_000_000,
        )
        forged["proposals"][0]["units"] = forged["proposals"][0]["sizing_receipt"][
            "final_units"
        ]
        forged["decision_id"] = decision_id_for(forged)
        with self.assertRaisesRegex(AILiveGatewayError, "entry-stop geometry"):
            self._build(decision=forged)

    def test_sealed_zero_daily_risk_blocks_live_artifact(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["portfolio"]["daily_target"].update(
            {"status": "RISK_BUDGET_EXHAUSTED", "remaining_risk_budget_jpy": 0.0}
        )
        body = {key: value for key, value in packet.items() if key != "packet_sha256"}
        packet["packet_sha256"] = canonical_json_sha256(body)
        decision = self._decision(packet=packet)
        with self.assertRaisesRegex(AILiveGatewayError, "capacity is exhausted"):
            self._build(decision=decision, packet=packet)

    def test_net_edge_source_mismatch_and_nonpositive_after_cost_fail_closed(self) -> None:
        for field, value, message in (
            ("source_sha256", "d" * 64, "source does not match"),
            ("net_edge_after_cost_jpy", 0.0, "must be positive"),
        ):
            with self.subTest(field=field):
                decision = copy.deepcopy(self._decision())
                decision["proposals"][0]["net_edge_proof"][field] = value
                decision["decision_id"] = decision_id_for(decision)
                with self.assertRaisesRegex(AILiveGatewayError, message):
                    self._build(decision=decision)

    def test_forged_net_edge_metrics_and_stale_packet_are_blocked(self) -> None:
        forged = copy.deepcopy(self._decision())
        forged["proposals"][0]["net_edge_proof"]["net_jpy"] = 3000.0
        forged["decision_id"] = decision_id_for(forged)
        with self.assertRaisesRegex(AILiveGatewayError, "not present"):
            self._build(decision=forged)

        stale_packet = self._packet(now=NOW - timedelta(days=2))
        stale_decision = self._decision(packet=stale_packet)
        with self.assertRaisesRegex(AILiveGatewayError, "stale"):
            self._build(decision=stale_decision, packet=stale_packet)

    def test_cost_proof_and_broker_epoch_must_match_current_evidence(self) -> None:
        bad_cost = copy.deepcopy(self._decision())
        bad_cost["proposals"][0]["cost_proof"]["execution_cost_floor_sha256"] = "d" * 64
        bad_cost["decision_id"] = decision_id_for(bad_cost)
        with self.assertRaisesRegex(AILiveGatewayError, "cost proof is stale or mismatched"):
            self._build(decision=bad_cost)

        bad_epoch = copy.deepcopy(self._decision())
        bad_epoch["broker_epoch"] = "9002"
        bad_epoch["proposals"][0]["evidence_binding"]["broker_epoch"] = "9002"
        bad_epoch["decision_id"] = decision_id_for(bad_epoch)
        with self.assertRaisesRegex(AILiveGatewayError, "another broker epoch"):
            self._build(decision=bad_epoch)

    def test_non_entry_is_validated_but_never_invokes_subprocess(self) -> None:
        current = datetime.now(timezone.utc)
        packet = self._packet(now=current)
        decision = build_entry_decision(
            action="WAIT",
            cycle_id="cycle-wait",
            broker_epoch=broker_epoch_from_evidence_packet(packet),
            evidence_observed_at_utc=current,
            created_at_utc=current,
            ttl_seconds=600,
            reasons=("no positive edge",),
        )
        with patch("quant_rabbit.ai_live_gateway.subprocess.run") as run:
            result = execute_ai_trade_candidate(
                repo_root=self.root,
                state_root=self.root / "state",
                receipt={
                    "run_id": "cycle-wait",
                    "entry_decision": decision,
                    "evidence_packet": packet,
                },
            )
        run.assert_not_called()
        self.assertEqual(result["status"], "NO_BROKER_ACTION")
        self.assertEqual(result["broker_order_posts"], 0)
        self.assertFalse(result["broker_mutation_allowed"])


if __name__ == "__main__":
    unittest.main()
