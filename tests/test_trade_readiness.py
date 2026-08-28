from __future__ import annotations

import unittest
from datetime import datetime, timezone

from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.trade_readiness import (
    ExplicitRiskLimits,
    RuntimeMode,
    SignalSizingInput,
    screen_trade_readiness,
    size_signal_for_runtime_mode,
)


NOW = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)
LIMITS = ExplicitRiskLimits(
    max_loss_per_order_jpy=500.0,
    stop_drawdown_jpy=5_000.0,
    minimum_margin_buffer_jpy=50_000.0,
    max_post_entry_current_mcp=0.85,
    max_post_entry_stress_mcp=0.90,
    max_currency_factor_nav_multiple=3.0,
    max_bot_positions=2,
    mode_hysteresis_mcp=0.03,
    forward_proof_sha256="a" * 64,
    risk_contract_sha256="b" * 64,
)


class TradeReadinessTest(unittest.TestCase):
    def test_manual_positions_and_margin_are_read_only_waiting_external_state(self) -> None:
        positions = (
            BrokerPosition("1", "USD_JPY", Side.SHORT, 40_000, 147.0, take_profit=146.0, owner=Owner.UNKNOWN),
            BrokerPosition("2", "EUR_USD", Side.SHORT, 1_000, 1.17, owner=Owner.UNKNOWN),
            BrokerPosition("3", "EUR_USD", Side.SHORT, 1_000, 1.17, take_profit=1.16, owner=Owner.UNKNOWN),
        )
        snapshot = BrokerSnapshot(
            fetched_at_utc=NOW,
            positions=positions,
            quotes={
                "EUR_USD": Quote("EUR_USD", 1.17000, 1.17008, NOW),
                "USD_JPY": Quote("USD_JPY", 147.000, 147.008, NOW),
            },
        )
        result = screen_trade_readiness(
            snapshot=snapshot,
            raw_account={
                "account": {
                    "NAV": "283908.7",
                    "marginUsed": "270292.6",
                    "marginAvailable": "13792.5",
                    "marginCloseoutPercent": "0.95145",
                }
            },
            limits=LIMITS,
            software_ready=True,
            now_utc=NOW,
        )
        self.assertEqual(result["status"], "ready_waiting_for_margin")
        self.assertEqual(result["lifecycle"], "waiting_external_state")
        self.assertEqual(result["orders_sent"], 0)
        self.assertFalse(result["broker_write_performed"])
        self.assertEqual(result["account"]["no_touch_position_count"], 3)
        self.assertEqual(result["account"]["attached_tp_count"], 2)
        self.assertIn("EXISTING_NO_TOUCH_POSITIONS", result["observations"])
        self.assertIn("CURRENCY_FACTOR_CONCENTRATION_ABOVE_BUDGET", result["blockers"])
        self.assertIn("MINIMUM_MARGIN_BUFFER_NOT_MET", result["blockers"])
        self.assertIn("MARGIN_CLOSEOUT_PERCENT_ABOVE_RISK_CONTRACT", result["blockers"])
        self.assertIn("STRESS_MARGIN_CLOSEOUT_PERCENT_ABOVE_RISK_CONTRACT", result["blockers"])
        self.assertLess(result["currency_exposure"]["USD"], 0.0)

    def test_flat_account_with_three_limits_and_fresh_quotes_reaches_final_screen(self) -> None:
        snapshot = BrokerSnapshot(
            fetched_at_utc=NOW,
            quotes={
                "EUR_USD": Quote("EUR_USD", 1.17000, 1.17008, NOW),
                "USD_JPY": Quote("USD_JPY", 147.000, 147.008, NOW),
            },
        )
        result = screen_trade_readiness(
            snapshot=snapshot,
            raw_account={
                "account": {
                    "NAV": "283908.7",
                    "marginUsed": "0",
                    "marginAvailable": "283908.7",
                    "marginCloseoutPercent": "0",
                }
            },
            limits=LIMITS,
            software_ready=True,
            now_utc=NOW,
        )
        self.assertEqual(result["status"], "ready_for_final_screen")
        self.assertEqual(result["blockers"], [])

    def test_missing_three_limits_is_not_order_ready(self) -> None:
        snapshot = BrokerSnapshot(
            fetched_at_utc=NOW,
            quotes={
                "EUR_USD": Quote("EUR_USD", 1.17000, 1.17008, NOW),
                "USD_JPY": Quote("USD_JPY", 147.000, 147.008, NOW),
            },
        )
        result = screen_trade_readiness(
            snapshot=snapshot,
            raw_account={"account": {"marginCloseoutPercent": "0", "marginAvailable": "100000"}},
            limits=ExplicitRiskLimits(),
            software_ready=True,
            now_utc=NOW,
        )
        self.assertEqual(result["status"], "ready_waiting_for_risk_limits")
        self.assertEqual(result["lifecycle"], "needs_user_decision")
        self.assertEqual(result["orders_sent"], 0)

    def test_mode_hysteresis_and_broker_minimum_keep_unsafe_signal_shadow_only(self) -> None:
        signal = SignalSizingInput(
            requested_units=1_000,
            broker_minimum_units=1,
            margin_jpy_per_unit=5.0,
            closeout_margin_jpy_per_unit=2.5,
            stress_closeout_margin_jpy_per_unit=3.0,
            loss_jpy_per_unit=1.0,
            factor_delta_jpy_per_unit={"USD": 159.0},
        )
        limits = LIMITS
        unsafe = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.SHADOW_ONLY,
            inventory_state="RUNNING",
            has_bot_inventory=False,
            nav_jpy=282_890.0,
            margin_available_jpy=12_500.0,
            current_mcp=0.9558,
            stress_baseline_mcp=1.02,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=0,
            cooldown_elapsed=True,
            factor_exposure_jpy={"USD": -6_000_000.0},
            limits=limits,
            software_ready=True,
            signal=signal,
        )
        self.assertEqual(unsafe["mode"], "SHADOW_ONLY")
        self.assertEqual(unsafe["calculated_units"], 0)
        self.assertFalse(unsafe["mutation_allowed"])

        promote = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.SHADOW_ONLY,
            inventory_state="RUNNING",
            has_bot_inventory=False,
            nav_jpy=300_000.0,
            margin_available_jpy=100_000.0,
            current_mcp=0.80,
            stress_baseline_mcp=0.83,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=0,
            cooldown_elapsed=True,
            factor_exposure_jpy={"USD": 0.0},
            limits=limits,
            software_ready=True,
            signal=SignalSizingInput(
                requested_units=1_000,
                broker_minimum_units=1,
                margin_jpy_per_unit=1.0,
                closeout_margin_jpy_per_unit=0.1,
                stress_closeout_margin_jpy_per_unit=0.2,
                loss_jpy_per_unit=1.0,
                factor_delta_jpy_per_unit={"USD": 159.0},
            ),
        )
        self.assertEqual(promote["mode"], "THROTTLED_LIVE")
        self.assertTrue(promote["mutation_allowed"])

        retained = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.THROTTLED_LIVE,
            inventory_state="RUNNING",
            has_bot_inventory=False,
            nav_jpy=300_000.0,
            margin_available_jpy=100_000.0,
            current_mcp=0.84,
            stress_baseline_mcp=0.88,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=0,
            cooldown_elapsed=True,
            factor_exposure_jpy={"USD": 0.0},
            limits=limits,
            software_ready=True,
            signal=SignalSizingInput(
                requested_units=1_000,
                broker_minimum_units=1,
                margin_jpy_per_unit=1.0,
                closeout_margin_jpy_per_unit=0.1,
                stress_closeout_margin_jpy_per_unit=0.2,
                loss_jpy_per_unit=1.0,
                factor_delta_jpy_per_unit={"USD": 159.0},
            ),
        )
        self.assertEqual(retained["mode"], "THROTTLED_LIVE")

    def test_gate_deterioration_with_bot_inventory_freezes_then_draining_overrides(self) -> None:
        signal = SignalSizingInput(10, 1, 1.0, 1.0, 1.0, 1.0, {"USD": 1.0})
        limits = LIMITS
        frozen = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.THROTTLED_LIVE,
            inventory_state="RUNNING",
            has_bot_inventory=True,
            nav_jpy=100_000.0,
            margin_available_jpy=1_000.0,
            current_mcp=0.95,
            stress_baseline_mcp=1.0,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=1,
            cooldown_elapsed=True,
            factor_exposure_jpy={},
            limits=limits,
            software_ready=True,
            signal=signal,
        )
        self.assertEqual(frozen["mode"], "FREEZE_NEW")
        self.assertFalse(frozen["mutation_allowed"])
        draining = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.FREEZE_NEW,
            inventory_state="DRAINING",
            has_bot_inventory=True,
            nav_jpy=100_000.0,
            margin_available_jpy=1_000.0,
            current_mcp=0.95,
            stress_baseline_mcp=1.0,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=1,
            cooldown_elapsed=True,
            factor_exposure_jpy={},
            limits=limits,
            software_ready=True,
            signal=signal,
        )
        self.assertEqual(draining["mode"], "DRAINING")

    def test_live_mutation_requires_forward_seal_and_enforces_loss_drawdown_and_stress(self) -> None:
        signal = SignalSizingInput(
            requested_units=1_000,
            broker_minimum_units=1,
            margin_jpy_per_unit=1.0,
            closeout_margin_jpy_per_unit=0.1,
            stress_closeout_margin_jpy_per_unit=0.2,
            loss_jpy_per_unit=2.0,
            factor_delta_jpy_per_unit={"USD": 159.0},
        )
        unsealed = ExplicitRiskLimits(
            **{
                name: getattr(LIMITS, name)
                for name in LIMITS.__dataclass_fields__
                if name != "forward_proof_sha256"
            },
            forward_proof_sha256=None,
        )
        result = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.SHADOW_ONLY,
            inventory_state="RUNNING",
            has_bot_inventory=False,
            nav_jpy=300_000.0,
            margin_available_jpy=200_000.0,
            current_mcp=0.50,
            stress_baseline_mcp=0.60,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=0,
            cooldown_elapsed=True,
            factor_exposure_jpy={"USD": 0.0, "EUR": 0.0, "JPY": 0.0},
            limits=unsealed,
            software_ready=True,
            signal=signal,
        )
        self.assertEqual(result["mode"], "SHADOW_ONLY")
        self.assertFalse(result["mutation_allowed"])

        admitted = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.SHADOW_ONLY,
            inventory_state="RUNNING",
            has_bot_inventory=False,
            nav_jpy=300_000.0,
            margin_available_jpy=200_000.0,
            current_mcp=0.50,
            stress_baseline_mcp=0.60,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=0,
            cooldown_elapsed=True,
            factor_exposure_jpy={"USD": 0.0, "EUR": 0.0, "JPY": 0.0},
            limits=LIMITS,
            software_ready=True,
            signal=signal,
        )
        self.assertEqual(admitted["calculated_units"], 250)
        self.assertEqual(admitted["mode"], "THROTTLED_LIVE")
        self.assertTrue(admitted["mutation_allowed"])

        stopped = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.THROTTLED_LIVE,
            inventory_state="RUNNING",
            has_bot_inventory=True,
            nav_jpy=300_000.0,
            margin_available_jpy=200_000.0,
            current_mcp=0.50,
            stress_baseline_mcp=0.60,
            campaign_drawdown_jpy=5_000.0,
            current_bot_position_count=1,
            cooldown_elapsed=True,
            factor_exposure_jpy={"USD": 0.0, "EUR": 0.0, "JPY": 0.0},
            limits=LIMITS,
            software_ready=True,
            signal=signal,
        )
        self.assertEqual(stopped["mode"], "FREEZE_NEW")
        self.assertEqual(stopped["transition_reason"], "STOP_DRAWDOWN_REACHED")

    def test_factor_reduction_cannot_cross_zero_into_opposite_concentration(self) -> None:
        result = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.SHADOW_ONLY,
            inventory_state="RUNNING",
            has_bot_inventory=False,
            nav_jpy=100_000.0,
            margin_available_jpy=500_000.0,
            current_mcp=0.10,
            stress_baseline_mcp=0.20,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=0,
            cooldown_elapsed=True,
            factor_exposure_jpy={"USD": -250_000.0},
            limits=LIMITS,
            software_ready=True,
            signal=SignalSizingInput(
                requested_units=10_000,
                broker_minimum_units=1,
                margin_jpy_per_unit=1.0,
                closeout_margin_jpy_per_unit=0.001,
                stress_closeout_margin_jpy_per_unit=0.001,
                loss_jpy_per_unit=0.01,
                factor_delta_jpy_per_unit={"USD": 100.0},
            ),
        )
        self.assertEqual(result["calculated_units"], 5_500)
        self.assertEqual(result["mode"], "THROTTLED_LIVE")

    def test_non_finite_runtime_inputs_fail_closed(self) -> None:
        result = size_signal_for_runtime_mode(
            previous_mode=RuntimeMode.SHADOW_ONLY,
            inventory_state="RUNNING",
            has_bot_inventory=False,
            nav_jpy=100_000.0,
            margin_available_jpy=float("nan"),
            current_mcp=0.10,
            stress_baseline_mcp=0.20,
            campaign_drawdown_jpy=0.0,
            current_bot_position_count=0,
            cooldown_elapsed=True,
            factor_exposure_jpy={},
            limits=LIMITS,
            software_ready=True,
            signal=SignalSizingInput(10, 1, 1.0, 1.0, 1.0, 1.0, {}),
        )
        self.assertEqual(result["mode"], "SHADOW_ONLY")
        self.assertFalse(result["mutation_allowed"])


if __name__ == "__main__":
    unittest.main()
