from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.fast_bot import REGIME_CONTRACT, _seal, build_fast_bot_shadow
from quant_rabbit.fast_bot_promotion import (
    EXTERNAL_MUTATION_GATEWAY,
    FORWARD_ADMISSION_CONTRACT,
    RISK_CONTRACT,
    SIZING_RECEIPT_CONTRACT,
    SUPERVISION_RECEIPT_CONTRACT,
    _canonical_sha,
    build_fast_bot_promotion,
    build_sizing_receipt,
    dispatch_promotion_once,
    seal_forward_admission,
    seal_risk_contract,
    seal_sizing_receipt,
    seal_supervision_receipt,
)
from quant_rabbit.inventory_controller import InventoryController


class RecordingGateway:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def run(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.fail:
            raise TimeoutError("ambiguous broker boundary")
        return {"status": "STAGED", "sent": False, "sent_count": 0}


class FastBotPromotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
        regime = _seal(
            {
                "contract": REGIME_CONTRACT,
                "schema_version": 1,
                "generated_at_utc": self.now.isoformat(),
                "rows": [
                    {
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "method": "RANGE_ROTATION",
                        "state": "GO",
                        "execution_enabled": True,
                        "score": 5.0,
                        "m1_closed_candle_utc": self.now.isoformat(),
                        "m5_atr_pips": 5.0,
                    }
                ],
            }
        )
        shadow = build_fast_bot_shadow(
            regime,
            broker_snapshot={
                "quotes": {
                    "EUR_USD": {
                        "bid": 1.16410,
                        "ask": 1.16418,
                        "timestamp_utc": self.now.isoformat(),
                    }
                }
            },
            now_utc=self.now,
        )
        self.signal = shadow["signals"][0]
        self.software_sha = "a" * 64
        self.feature_sha = "b" * 64
        self.live_campaign_id = "live-fb-test-forward-v1"
        self.live_strategy_id = f"live-{self.signal['strategy_id']}"
        self.supervision = seal_supervision_receipt({
            "contract": SUPERVISION_RECEIPT_CONTRACT,
            "receipt_id": "allow-current-regime",
            "event_id": "regime-ready",
            "dedupe_key": "PORTFOLIO|REGIME_READY",
            "feature_snapshot_sha256": self.feature_sha,
            "signal_sha256": self.signal["signal_sha256"],
            "regime_contract_sha256": self.signal["regime_contract_sha256"],
            "decision": "ALLOW",
            "regime": "RANGE",
            "allowed_strategy_ids": [self.live_strategy_id],
            "risk_budget_cap_jpy": 500.0,
            "max_positions_cap": 2,
            "generated_at_utc": self.now.isoformat(),
            "expires_at_utc": (self.now + timedelta(minutes=5)).isoformat(),
        })
        self.inventory = InventoryController.open(
            self.root / "inventory.json",
            campaign_id=self.live_campaign_id,
            now_utc=self.now,
        )
        applied = self.inventory.apply_supervision_receipt(
            event={
                "event_id": self.supervision["event_id"],
                "dedupe_key": self.supervision["dedupe_key"],
            },
            receipt=self.supervision,
            now_utc=self.now,
        )
        self.assertEqual(applied, "APPLIED_ALLOW")
        self.forward = seal_forward_admission(
            {
                "contract": FORWARD_ADMISSION_CONTRACT,
                "status": "ADMITTED",
                "promotion_allowed": True,
                "live_permission": True,
                "external_mutation_gateway": EXTERNAL_MUTATION_GATEWAY,
                "software_version_sha256": self.software_sha,
                "allowed_strategy_ids": [self.signal["strategy_id"]],
                "allowed_pairs": [self.signal["pair"]],
                "resolved_fills": 100,
                "active_days": 10,
                "profit_factor": 1.30,
                "one_sided_95_expectancy_lower_pips": 0.01,
                "spread_anomaly_rate": 0.01,
                "after_cost_net_pips": 4.0,
                "leftover_inventory_units": 0,
                "paper_broker_mutation_count": 0,
                "maximum_drawdown_within_predeclared_limit": True,
                "tail_loss_within_predeclared_limit": True,
                "margin_stress_passed": True,
                "independent_readback_verified": True,
            }
        )
        self.risk = seal_risk_contract(
            {
                "contract": RISK_CONTRACT,
                "status": "ACCEPTED",
                "accepted_by_user": True,
                "acceptance_source": "EXPLICIT_USER_DECISION",
                "acceptance_id": "test-user-risk-acceptance",
                "accepted_at_utc": self.now.isoformat(),
                "software_version_sha256": self.software_sha,
                "forward_admission_sha256": self.forward["admission_sha256"],
                "live_campaign_id": self.live_campaign_id,
                "max_loss_per_order_jpy": 500.0,
                "stop_drawdown_jpy": 5_000.0,
                "minimum_margin_buffer_jpy": 50_000.0,
                "max_post_entry_current_mcp": 0.85,
                "max_post_entry_stress_mcp": 0.90,
                "max_currency_factor_nav_multiple": 3.0,
                "max_bot_positions": 2,
                "mode_hysteresis_mcp": 0.03,
                "stress_pips": 25.0,
                "max_account_snapshot_age_seconds": 20.0,
            }
        )
        self.sizing = seal_sizing_receipt({
            "contract": SIZING_RECEIPT_CONTRACT,
            "signal_sha256": self.signal["signal_sha256"],
            "forward_admission_sha256": self.forward["admission_sha256"],
            "risk_contract_sha256": self.risk["risk_contract_sha256"],
            "software_version_sha256": self.software_sha,
            "mode": "THROTTLED_LIVE",
            "mutation_allowed": True,
            "calculated_units": 10,
            "broker_minimum_units": 1,
            "planned_loss_jpy": 20.0,
            "post_entry_current_mcp": 0.80,
            "post_entry_stress_mcp": 0.86,
            "post_entry_margin_available_jpy": 75_000.0,
            "post_entry_max_currency_factor_nav_multiple": 2.5,
            "campaign_drawdown_jpy": 100.0,
            "account_snapshot_age_seconds": 1.0,
            "quote_age_seconds": 1.0,
            "account_snapshot_sha256": "c" * 64,
            "quote_snapshot_sha256": "d" * 64,
            "calculated_at_utc": self.now.isoformat(),
            "account_scope_includes_manual_and_tagless_positions": True,
            "manual_tagless_mutation_count": 0,
            "spread_gate_passed": True,
        })

    def tearDown(self) -> None:
        self.temp.cleanup()

    def build(self, **overrides):
        values = {
            "signal": self.signal,
            "supervision_receipt": self.supervision,
            "sizing_receipt": self.sizing,
            "forward_admission": self.forward,
            "risk_contract": self.risk,
            "software_version_sha256": self.software_sha,
            "expected_feature_snapshot_sha256": self.feature_sha,
            "inventory": self.inventory,
            "now_utc": self.now,
        }
        values.update(overrides)
        return build_fast_bot_promotion(**values)

    def test_exact_admission_emits_one_gateway_compatible_intent(self) -> None:
        result = self.build()

        self.assertEqual(result["status"], "ADMITTED")
        self.assertTrue(result["live_permission"])
        self.assertEqual(result["external_mutation_gateway"], "LiveOrderGateway")
        rows = result["intents_payload"]["results"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["intent"]["units"], 10)
        self.assertEqual(rows[0]["intent"]["side"], self.signal["side"])
        self.assertEqual(rows[0]["intent"]["metadata"]["campaign_id"], self.live_campaign_id)
        self.assertEqual(rows[0]["intent"]["metadata"]["strategy_id"], self.live_strategy_id)
        self.assertNotEqual(
            rows[0]["intent"]["metadata"]["campaign_id"], self.signal["campaign_id"]
        )
        self.assertNotEqual(
            rows[0]["intent"]["metadata"]["strategy_id"], self.signal["strategy_id"]
        )
        self.assertEqual(
            rows[0]["intent"]["metadata"]["signal_sha256"],
            self.signal["signal_sha256"],
        )

    def test_unsealed_or_mismatched_evidence_never_emits_intent(self) -> None:
        forged = dict(self.forward)
        forged["resolved_fills"] = 101
        result = self.build(forward_admission=forged)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIsNone(result["intents_payload"])
        self.assertIn("FORWARD_ADMISSION_INVALID_OR_UNPROVEN", result["blocking_reasons"])

        wrong_version = self.build(software_version_sha256="c" * 64)
        self.assertEqual(wrong_version["status"], "BLOCKED")
        self.assertIsNone(wrong_version["intents_payload"])

    def test_progressive_micro_live_does_not_require_fixed_sample_wait(self) -> None:
        progressive = seal_forward_admission(
            {
                "contract": FORWARD_ADMISSION_CONTRACT,
                "status": "ADMITTED",
                "promotion_allowed": True,
                "live_permission": True,
                "external_mutation_gateway": EXTERNAL_MUTATION_GATEWAY,
                "software_version_sha256": self.software_sha,
                "allowed_strategy_ids": [self.signal["strategy_id"]],
                "allowed_pairs": [self.signal["pair"]],
                "admission_mode": "PROGRESSIVE_MICRO_LIVE",
                "progressive_live_user_authorized": True,
                "authorization_source": "EXPLICIT_USER_DECISION",
                "authorization_id": "progressive-live-policy-20260828",
                "resident_shadow_required": True,
                "resident_shadow_status": "RUNNING",
                "resident_shadow_execution_authority": "NONE",
                "resident_shadow_broker_mutation_count": 0,
                "resident_shadow_external_order_attempts": 0,
                "resident_shadow_external_orders": 0,
                "scorecard_monitoring_active": True,
                "scorecard_can_force_demotion": True,
                "fixed_sample_wait_required_for_micro_live": False,
                "micro_live_only": True,
                "independent_readback_verified": True,
                "resolved_fills": 0,
                "active_days": 0,
            }
        )
        risk = dict(self.risk)
        risk["forward_admission_sha256"] = progressive["admission_sha256"]
        risk = seal_risk_contract(risk)
        sizing = dict(self.sizing)
        sizing["forward_admission_sha256"] = progressive["admission_sha256"]
        sizing["risk_contract_sha256"] = risk["risk_contract_sha256"]
        sizing = seal_sizing_receipt(sizing)
        result = self.build(
            forward_admission=progressive,
            risk_contract=risk,
            sizing_receipt=sizing,
        )
        self.assertEqual(result["status"], "ADMITTED")
        self.assertTrue(result["live_permission"])

        stopped = dict(progressive)
        stopped["resident_shadow_status"] = "STOPPED"
        stopped = seal_forward_admission(stopped)
        blocked = self.build(forward_admission=stopped)
        self.assertIn(
            "FORWARD_ADMISSION_INVALID_OR_UNPROVEN",
            blocked["blocking_reasons"],
        )

    def test_account_wide_mode_receipt_binds_directly_to_promotion(self) -> None:
        sizing = build_sizing_receipt(
            mode_receipt={
                "mode": "THROTTLED_LIVE",
                "mutation_allowed": True,
                "calculated_units": 1,
                "safe_unit_capacity": 250,
                "broker_minimum_units": 1,
                "planned_loss_jpy": 2.0,
                "post_entry_current_mcp": 0.80,
                "post_entry_stress_mcp": 0.86,
                "post_entry_margin_available_jpy": 75_000.0,
                "post_entry_max_currency_factor_nav_multiple": 2.5,
            },
            signal_sha256=self.signal["signal_sha256"],
            forward_admission_sha256=self.forward["admission_sha256"],
            risk_contract_sha256=self.risk["risk_contract_sha256"],
            software_version_sha256=self.software_sha,
            account_snapshot_sha256="c" * 64,
            quote_snapshot_sha256="d" * 64,
            campaign_drawdown_jpy=100.0,
            account_snapshot_age_seconds=1.0,
            quote_age_seconds=1.0,
            calculated_at_utc=self.now,
            spread_gate_passed=True,
        )
        result = self.build(sizing_receipt=sizing)
        self.assertEqual(result["status"], "ADMITTED")
        self.assertEqual(result["intents_payload"]["results"][0]["intent"]["units"], 1)

    def test_stale_signal_and_stale_supervision_fail_closed(self) -> None:
        stale_signal = copy.deepcopy(self.signal)
        stale_signal["quote_timestamp_utc"] = (self.now - timedelta(minutes=2)).isoformat()
        body = {key: value for key, value in stale_signal.items() if key != "signal_sha256"}
        stale_signal["signal_sha256"] = _canonical_sha(body)
        stale_sizing = dict(self.sizing)
        stale_sizing["signal_sha256"] = stale_signal["signal_sha256"]
        stale_sizing = seal_sizing_receipt(stale_sizing)
        stale_supervision = dict(self.supervision)
        stale_supervision["signal_sha256"] = stale_signal["signal_sha256"]
        stale_supervision = seal_supervision_receipt(stale_supervision)
        result = self.build(
            signal=stale_signal,
            sizing_receipt=stale_sizing,
            supervision_receipt=stale_supervision,
        )
        self.assertIn("SIGNAL_STALE_OR_FUTURE", result["blocking_reasons"])

        expired = dict(self.supervision)
        expired["expires_at_utc"] = (self.now - timedelta(seconds=1)).isoformat()
        expired = seal_supervision_receipt(expired)
        result = self.build(supervision_receipt=expired)
        self.assertIn("SUPERVISION_RECEIPT_INVALID_OR_STALE", result["blocking_reasons"])

    def test_llm_cannot_choose_units_or_order_fields(self) -> None:
        receipt = dict(self.supervision)
        receipt["Units"] = 999
        receipt = seal_supervision_receipt(receipt)
        result = self.build(supervision_receipt=receipt)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("LLM_RECEIPT_CONTAINS_ORDER_FIELDS", result["blocking_reasons"])
        self.assertIsNone(result["intents_payload"])

    def test_account_wide_sizing_gates_fail_closed(self) -> None:
        unsafe_values = {
            "post_entry_margin_available_jpy": 49_999.0,
            "post_entry_max_currency_factor_nav_multiple": 3.01,
            "campaign_drawdown_jpy": 5_000.0,
            "account_snapshot_age_seconds": 20.01,
            "quote_age_seconds": 20.01,
            "account_scope_includes_manual_and_tagless_positions": False,
            "manual_tagless_mutation_count": 1,
            "spread_gate_passed": False,
        }
        for field, value in unsafe_values.items():
            with self.subTest(field=field):
                receipt = dict(self.sizing)
                receipt[field] = value
                result = self.build(sizing_receipt=seal_sizing_receipt(receipt))
                self.assertEqual(result["status"], "BLOCKED")
                self.assertIn(
                    "SIZING_RECEIPT_INVALID_OR_BLOCKED", result["blocking_reasons"]
                )

    def test_malformed_supervision_is_blocked_without_exception(self) -> None:
        receipt = dict(self.supervision)
        receipt["max_positions_cap"] = "not-an-integer"
        result = self.build(supervision_receipt=seal_supervision_receipt(receipt))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIsNone(result["intents_payload"])

    def test_sizing_inventory_and_readd_guards_fail_closed(self) -> None:
        blocked_sizing = dict(self.sizing)
        blocked_sizing["mode"] = "SHADOW_ONLY"
        blocked_sizing["mutation_allowed"] = False
        blocked_sizing["calculated_units"] = 0
        blocked_sizing = seal_sizing_receipt(blocked_sizing)
        result = self.build(sizing_receipt=blocked_sizing)
        self.assertIn("SIZING_RECEIPT_INVALID_OR_BLOCKED", result["blocking_reasons"])

        self.inventory.freeze_new(reason="TEST_HARD_GUARD", now_utc=self.now)
        result = self.build()
        self.assertIn("INVENTORY_NOT_RUNNING_OR_COOLDOWN_ACTIVE", result["blocking_reasons"])

    def test_existing_gateway_is_invoked_once_in_stage_only_mode(self) -> None:
        promotion = self.build()
        gateway = RecordingGateway()
        result = dispatch_promotion_once(
            promotion=promotion,
            gateway=gateway,
            intents_path=self.root / "gateway_intents.json",
            dispatch_ledger_path=self.root / "dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
            send=False,
            confirm_live=False,
        )
        duplicate = dispatch_promotion_once(
            promotion=promotion,
            gateway=gateway,
            intents_path=self.root / "gateway_intents.json",
            dispatch_ledger_path=self.root / "dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
            send=False,
            confirm_live=False,
        )

        self.assertEqual(result["status"], "GATEWAY_RETURNED")
        self.assertEqual(result["live_order_gateway_invocation_count"], 1)
        self.assertFalse(result["broker_mutation_performed"])
        self.assertEqual(len(gateway.calls), 1)
        self.assertFalse(gateway.calls[0]["send"])
        self.assertEqual(duplicate["status"], "DUPLICATE_BLOCKED")
        self.assertEqual(len(gateway.calls), 1)

    def test_ambiguous_gateway_result_consumes_reservation_without_retry(self) -> None:
        promotion = self.build()
        gateway = RecordingGateway(fail=True)
        first = dispatch_promotion_once(
            promotion=promotion,
            gateway=gateway,
            intents_path=self.root / "gateway_intents.json",
            dispatch_ledger_path=self.root / "dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
            send=False,
        )
        second = dispatch_promotion_once(
            promotion=promotion,
            gateway=gateway,
            intents_path=self.root / "gateway_intents.json",
            dispatch_ledger_path=self.root / "dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
            send=False,
        )
        self.assertEqual(first["status"], "UNKNOWN_GATEWAY_RESULT_NO_RETRY")
        self.assertEqual(second["status"], "DUPLICATE_BLOCKED")
        self.assertEqual(len(gateway.calls), 1)

    def test_inventory_change_after_promotion_blocks_gateway_invocation(self) -> None:
        promotion = self.build()
        self.inventory.freeze_new(reason="ACCOUNT_GATE_WORSENED", now_utc=self.now)
        gateway = RecordingGateway()
        result = dispatch_promotion_once(
            promotion=promotion,
            gateway=gateway,
            intents_path=self.root / "gateway_intents.json",
            dispatch_ledger_path=self.root / "dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
            send=False,
        )
        self.assertEqual(result["status"], "BLOCKED_INVENTORY_CHANGED_AFTER_PROMOTION")
        self.assertEqual(gateway.calls, [])

    def test_send_requires_explicit_confirmation_even_when_admitted(self) -> None:
        promotion = self.build()
        gateway = RecordingGateway()
        result = dispatch_promotion_once(
            promotion=promotion,
            gateway=gateway,
            intents_path=self.root / "gateway_intents.json",
            dispatch_ledger_path=self.root / "dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
            send=True,
            confirm_live=False,
        )
        self.assertEqual(result["status"], "BLOCKED_LIVE_CONFIRMATION_REQUIRED")
        self.assertEqual(gateway.calls, [])


if __name__ == "__main__":
    unittest.main()
