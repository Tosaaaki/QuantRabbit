from __future__ import annotations

import copy
import hashlib
import json
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
from quant_rabbit.fast_bot_shock_guard import (
    RECEIPT_CONTRACT,
    load_config as load_shock_config,
    seal as seal_guard,
    structure_exit_plan,
)
from quant_rabbit.inventory_controller import InventoryController
from quant_rabbit.broker.execution import _intent_from_json, _progressive_promotion_live_send_issues


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
        shock_config, _ = load_shock_config(
            Path(__file__).resolve().parents[1] / "config" / "fast_bot_shock_guard_v1.json"
        )
        catastrophe_width = 18.0
        catastrophe_price = float(self.signal["entry"]) - catastrophe_width / 10_000.0
        self.signal["stop_loss"] = catastrophe_price
        self.signal["stop_loss_pips"] = catastrophe_width
        self.signal["reward_risk"] = round(
            float(self.signal["take_profit_pips"]) / catastrophe_width, 6
        )
        stop = dict(self.signal["protective_stop"])
        stop.update(
            geometry_id="CONSERVATIVE_CATASTROPHE",
            stop_loss=catastrophe_price,
            stop_loss_pips=catastrophe_width,
            server_side_catastrophic_stop=True,
            normal_exit_policy="RAW_STRUCTURE_SHOCK_TIME_EXIT_FIRST",
            atr_role="AUXILIARY_NORMALIZATION_AND_UPPER_BOUND_ONLY",
            live_candidate_eligible=True,
        )
        self.signal["protective_stop"] = seal_guard(stop)
        self.signal["structure_exit_plan"] = structure_exit_plan(
            pair=self.signal["pair"],
            side=self.signal["side"],
            observed_at_utc=self.now,
            config=shock_config,
        )
        self.signal["shock_guard"] = seal_guard(
            {
                "contract": RECEIPT_CONTRACT,
                "schema_version": 1,
                "event_id": None,
                "state": "NORMAL",
                "resolution": None,
                "shock_direction": None,
                "observed_at_utc": self.now.isoformat(),
                "expires_at_utc": (self.now + timedelta(minutes=2)).isoformat(),
                "config_sha256": "f" * 64,
                "fail_closed_reason": None,
                "short_term_reversal": False,
                "higher_timeframe_continuation": False,
                "timeframe_alignment": {},
                "automatic_reversal_allowed": False,
                "execution_authority": "NONE",
                "broker_mutation_allowed": False,
                "external_order_attempts": 0,
                "external_orders": 0,
            }
        )
        self.signal["signal_sha256"] = _canonical_sha(
            {key: value for key, value in self.signal.items() if key != "signal_sha256"}
        )
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
            "protective_stop_loss_pips": self.signal["stop_loss_pips"],
            "loss_jpy_per_unit": 2.0,
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
        self.assertEqual(
            rows[0]["intent"]["market_context"]["method"],
            self.signal["method"],
        )
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
            protective_stop_loss_pips=float(self.signal["stop_loss_pips"]),
            loss_jpy_per_unit=2.0,
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

    def test_gateway_progressive_authority_binds_exact_fresh_promotion(self) -> None:
        promotion = self.build()
        quote_at = datetime.now(timezone.utc)
        promotion["bindings"]["signal_quote_timestamp_utc"] = quote_at.isoformat()
        promotion["bindings"]["signal_entry_ttl_seconds"] = 60
        promotion["expires_at_utc"] = (quote_at + timedelta(minutes=1)).isoformat()
        metadata = promotion["intents_payload"]["results"][0]["intent"]["metadata"]
        metadata["signal_quote_timestamp_utc"] = quote_at.isoformat()
        metadata["signal_entry_ttl_seconds"] = 60
        promotion["promotion_sha256"] = _canonical_sha(
            {key: value for key, value in promotion.items() if key != "promotion_sha256"}
        )
        path = self.root / "promotion.json"
        path.write_text(json.dumps(promotion), encoding="utf-8")
        promoted = promotion["intents_payload"]["results"][0]["intent"]
        intent = _intent_from_json(promoted)
        request = {
            "instrument": intent.pair,
            "type": "LIMIT",
            "units": str(intent.units),
        }

        issues = _progressive_promotion_live_send_issues(
            path,
            selected_lane_id=promotion["lane_id"],
            intents_payload=promotion["intents_payload"],
            send=True,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            intent=intent,
            final_units=intent.units,
            order_request=request,
            conflicting_verified_decision_path=None,
        )
        self.assertEqual(issues, [])

        changed = dict(promotion["intents_payload"])
        changed["results"] = copy.deepcopy(changed["results"])
        changed["results"][0]["intent"]["units"] += 1
        blocked = _progressive_promotion_live_send_issues(
            path,
            selected_lane_id=promotion["lane_id"],
            intents_payload=changed,
            send=True,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            intent=intent,
            final_units=intent.units,
            order_request=request,
            conflicting_verified_decision_path=None,
        )
        self.assertIn("FAST_BOT_PROMOTION_INTENTS_MISMATCH", {item["code"] for item in blocked})

    def test_full_live_sizing_is_not_an_automatic_promotion_mode(self) -> None:
        sizing = dict(self.sizing)
        sizing["mode"] = "FULL_LIVE"
        result = self.build(sizing_receipt=seal_sizing_receipt(sizing))
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("SIZING_RECEIPT_INVALID_OR_BLOCKED", result["blocking_reasons"])

    def test_missing_stale_or_mispriced_protective_stop_blocks_before_gateway(self) -> None:
        missing = copy.deepcopy(self.signal)
        missing.pop("protective_stop")
        missing["signal_sha256"] = _canonical_sha(
            {key: value for key, value in missing.items() if key != "signal_sha256"}
        )
        result = self.build(signal=missing)
        self.assertIn("PROTECTIVE_STOP_MISSING_OR_UNSEALED", result["blocking_reasons"])
        self.assertIsNone(result["intents_payload"])

        stale = copy.deepcopy(self.signal)
        stop = dict(stale["protective_stop"])
        stop["observed_at_utc"] = (self.now - timedelta(minutes=5)).isoformat()
        stale["protective_stop"] = seal_guard(stop)
        stale["signal_sha256"] = _canonical_sha(
            {key: value for key, value in stale.items() if key != "signal_sha256"}
        )
        result = self.build(signal=stale)
        self.assertIn("PROTECTIVE_STOP_STALE_OR_FUTURE", result["blocking_reasons"])

        wrong = copy.deepcopy(self.signal)
        stop = dict(wrong["protective_stop"])
        stop["stop_loss"] = float(wrong["entry"]) + 0.001
        wrong["stop_loss"] = stop["stop_loss"]
        wrong["protective_stop"] = seal_guard(stop)
        wrong["signal_sha256"] = _canonical_sha(
            {key: value for key, value in wrong.items() if key != "signal_sha256"}
        )
        result = self.build(signal=wrong)
        self.assertIn("PROTECTIVE_STOP_PRICE_INVALID", result["blocking_reasons"])

    def test_no_sl_shadow_arm_is_never_promotable_and_gateway_is_not_invoked(self) -> None:
        no_sl = copy.deepcopy(self.signal)
        stop = dict(no_sl["protective_stop"])
        stop.update(
            geometry_id="NO_SL_SHADOW_ONLY",
            stop_loss=None,
            stop_loss_pips=None,
            attached_required=False,
            server_side_catastrophic_stop=False,
            live_candidate_eligible=False,
            shadow_comparison_controls={
                "campaign_loss_cap_pips": 50.0,
                "holding_time_cap_minutes": 60,
                "inventory_position_cap": 1,
                "margin_usage_proxy_cap": 1.0,
            },
        )
        no_sl["stop_loss"] = None
        no_sl["stop_loss_pips"] = None
        no_sl["attached_stop_loss_required"] = False
        no_sl["protective_stop"] = seal_guard(stop)
        no_sl["signal_sha256"] = _canonical_sha(
            {key: value for key, value in no_sl.items() if key != "signal_sha256"}
        )
        promotion = self.build(signal=no_sl)
        self.assertIn(
            "PROTECTIVE_STOP_NOT_CATASTROPHIC_LIVE_CANDIDATE",
            promotion["blocking_reasons"],
        )
        gateway = RecordingGateway()
        result = dispatch_promotion_once(
            promotion=promotion,
            gateway=gateway,
            intents_path=self.root / "no_sl_intents.json",
            dispatch_ledger_path=self.root / "no_sl_dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
            send=False,
        )
        self.assertEqual(result["status"], "BLOCKED_NOT_ADMITTED")
        self.assertEqual(gateway.calls, [])

    def test_stop_width_and_units_risk_receipt_are_bound(self) -> None:
        sizing = dict(self.sizing)
        sizing["protective_stop_loss_pips"] = float(self.signal["stop_loss_pips"]) + 1.0
        result = self.build(sizing_receipt=seal_sizing_receipt(sizing))
        self.assertIn("SIZING_RECEIPT_INVALID_OR_BLOCKED", result["blocking_reasons"])

        sizing = dict(self.sizing)
        sizing["planned_loss_jpy"] = 19.0
        result = self.build(sizing_receipt=seal_sizing_receipt(sizing))
        self.assertIn("SIZING_RECEIPT_INVALID_OR_BLOCKED", result["blocking_reasons"])

    def test_shock_freeze_blocks_before_gateway(self) -> None:
        signal = copy.deepcopy(self.signal)
        receipt = dict(signal["shock_guard"])
        receipt["state"] = "SHOCK_FREEZE"
        receipt["event_id"] = "qrs:test"
        signal["shock_guard"] = seal_guard(receipt)
        signal["signal_sha256"] = _canonical_sha(
            {key: value for key, value in signal.items() if key != "signal_sha256"}
        )
        result = self.build(signal=signal)
        self.assertIn("SHOCK_GUARD_SHOCK_FREEZE", result["blocking_reasons"])
        gateway = RecordingGateway()
        dispatch = dispatch_promotion_once(
            promotion=result,
            gateway=gateway,
            intents_path=self.root / "guarded_intents.json",
            dispatch_ledger_path=self.root / "guarded_dispatch.json",
            inventory_state_path=self.root / "inventory.json",
            now_utc=self.now,
        )
        self.assertEqual(dispatch["live_order_gateway_invocation_count"], 0)
        self.assertEqual(gateway.calls, [])


if __name__ == "__main__":
    unittest.main()
