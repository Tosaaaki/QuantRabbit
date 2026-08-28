from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.broker.execution import _oanda_order_request
from quant_rabbit.inventory_controller import (
    InventoryController,
    InventoryState,
    LotIdentity,
    broker_position_identity,
)
from quant_rabbit.models import BrokerPosition, OrderIntent, OrderType, Owner, Side


NOW = datetime(2026, 8, 28, 10, 0, tzinfo=timezone.utc)


def _identity(lot_id: str, *, strategy_id: str = "range_rotation") -> LotIdentity:
    return LotIdentity(
        campaign_id="campaign-20260828",
        strategy_id=strategy_id,
        lot_id=lot_id,
    )


class InventoryControllerTest(unittest.TestCase):
    def test_gateway_binds_all_ownership_ids_and_tagless_manual_is_no_touch(self) -> None:
        identity = _identity("lot-001")
        intent = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=10,
            entry=1.10000,
            tp=1.10100,
            sl=1.09900,
            thesis="paper identity binding",
            metadata={**identity.to_metadata(), "lane_id": "lane:EUR_USD:LONG"},
        )

        order = _oanda_order_request(intent)

        self.assertEqual(order["clientExtensions"]["tag"], Owner.TRADER.value)
        self.assertEqual(order["clientExtensions"]["id"], identity.broker_client_id())
        self.assertEqual(
            order["tradeClientExtensions"]["id"], identity.broker_client_id()
        )
        tagged = BrokerPosition(
            trade_id="1",
            pair="EUR_USD",
            side=Side.LONG,
            units=10,
            entry_price=1.1,
            owner=Owner.TRADER,
            raw={"tradeClientExtensions": order["tradeClientExtensions"]},
        )
        self.assertEqual(broker_position_identity(tagged), identity)
        forged_or_legacy = BrokerPosition(
            trade_id="legacy-1",
            pair="EUR_USD",
            side=Side.LONG,
            units=10,
            entry_price=1.1,
            owner=Owner.TRADER,
            raw={
                "tradeClientExtensions": {
                    "id": identity.broker_client_id(),
                    "comment": "qr-vnext",
                }
            },
        )
        self.assertIsNone(broker_position_identity(forged_or_legacy))
        manual = BrokerPosition(
            trade_id="manual-1",
            pair="EUR_USD",
            side=Side.SHORT,
            units=1_000,
            entry_price=1.1,
            owner=Owner.UNKNOWN,
            raw={},
        )
        self.assertIsNone(broker_position_identity(manual))

        malformed = OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            units=10,
            entry=1.10000,
            tp=1.10100,
            sl=1.09900,
            thesis="missing lot identity",
            metadata={"campaign_id": "campaign-20260828"},
        )
        with self.assertRaisesRegex(ValueError, "require campaign_id"):
            _oanda_order_request(malformed)

    def test_all_owned_lots_partial_scale_then_terminal_flat_and_durable_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "inventory.json"
            controller = InventoryController.open(
                state_path,
                campaign_id="campaign-20260828",
                now_utc=NOW,
            )
            controller.register_pending_entry("pending-1", now_utc=NOW)
            controller.register_fill(
                identity=_identity("lot-001"),
                pair="EUR_USD",
                side=Side.LONG,
                units=10,
                entry_price=1.1000,
                now_utc=NOW,
            )
            controller.register_fill(
                identity=_identity("lot-002", strategy_id="trend_continuation"),
                pair="USD_JPY",
                side=Side.SHORT,
                units=7,
                entry_price=150.000,
                now_utc=NOW,
            )
            controller.mark_lot(
                "lot-001", executable_price=1.1010, now_utc=NOW + timedelta(minutes=1)
            )
            controller.mark_lot(
                "lot-001", executable_price=1.1004, now_utc=NOW + timedelta(minutes=2)
            )
            self.assertEqual(controller.lots["lot-001"].mfe_pips, 10.0)
            self.assertEqual(controller.lots["lot-001"].giveback_pips, 6.0)
            controller.update_unwind_economics(
                "lot-002",
                estimated_margin_relief_jpy=28_739,
                estimated_close_loss_and_cost_jpy=38,
                currency_factor="USD",
                now_utc=NOW + timedelta(minutes=2),
            )

            controller.freeze_new(reason="MARGIN_BUFFER", now_utc=NOW + timedelta(minutes=3))
            controller.begin_draining(now_utc=NOW + timedelta(minutes=3))
            self.assertEqual(controller.state, InventoryState.DRAINING)
            actions = controller.unwind_actions(
                now_utc=NOW + timedelta(minutes=3),
                terminal_deadline_utc=NOW + timedelta(minutes=10),
            )
            self.assertEqual(
                {action.lot_id for action in actions if action.action == "REDUCE_BOT_LOT"},
                {"lot-001", "lot-002"},
            )
            self.assertEqual(
                next(action for action in actions if action.lot_id == "lot-001").units,
                5,
            )
            self.assertEqual(
                next(action for action in actions if action.lot_id == "lot-002").units,
                4,
            )
            controller.record_pending_cancel("pending-1", now_utc=NOW + timedelta(minutes=4))
            controller.record_unwind_fill(
                "lot-001", units=5, realized_after_cost_jpy=12.0, now_utc=NOW + timedelta(minutes=4)
            )
            controller.record_unwind_fill(
                "lot-002", units=4, realized_after_cost_jpy=-38.0, now_utc=NOW + timedelta(minutes=4)
            )
            terminal = controller.unwind_actions(
                now_utc=NOW + timedelta(minutes=10),
                terminal_deadline_utc=NOW + timedelta(minutes=10),
            )
            for action in terminal:
                controller.record_unwind_fill(
                    str(action.lot_id),
                    units=int(action.units or 0),
                    realized_after_cost_jpy=0.0,
                    now_utc=NOW + timedelta(minutes=10),
                )
            self.assertEqual(controller.state, InventoryState.FLAT)
            self.assertTrue(all(lot.remaining_units == 0 for lot in controller.lots.values()))
            controller.stop(now_utc=NOW + timedelta(minutes=10), cooldown=timedelta(minutes=30))
            self.assertEqual(controller.state, InventoryState.STOPPED)

            reloaded = InventoryController.open(
                state_path,
                campaign_id="campaign-20260828",
                now_utc=NOW + timedelta(minutes=11),
            )
            self.assertEqual(reloaded.state, InventoryState.STOPPED)
            with self.assertRaisesRegex(RuntimeError, "cooldown"):
                InventoryController.open(
                    state_path,
                    campaign_id="campaign-next",
                    now_utc=NOW + timedelta(minutes=39),
                )
            next_campaign = InventoryController.open(
                state_path,
                campaign_id="campaign-next",
                now_utc=NOW + timedelta(minutes=41),
            )
            self.assertEqual(next_campaign.state, InventoryState.RUNNING)

    def test_concurrent_stale_writer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "inventory.json"
            first = InventoryController.open(
                state_path,
                campaign_id="campaign-20260828",
                now_utc=NOW,
            )
            stale = InventoryController.open(
                state_path,
                campaign_id="campaign-20260828",
                now_utc=NOW,
            )
            first.register_pending_entry("pending-1", now_utc=NOW)
            with self.assertRaisesRegex(RuntimeError, "changed concurrently"):
                stale.register_pending_entry("pending-2", now_utc=NOW)

    def test_campaign_profit_lock_uses_start_nav_floor_and_forces_owned_flat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "inventory.json"
            controller = InventoryController.open(
                state_path,
                campaign_id="campaign-20260828",
                now_utc=NOW,
            )
            controller.configure_profit_lock(
                cycle_start_nav_jpy=100_000,
                now_utc=NOW,
            )
            controller.register_fill(
                identity=_identity("lot-profit-lock"),
                pair="EUR_USD",
                side=Side.LONG,
                units=10,
                entry_price=1.1,
                now_utc=NOW,
            )
            self.assertEqual(
                controller.evaluate_profit_lock(
                    current_nav_jpy=110_000,
                    now_utc=NOW + timedelta(minutes=1),
                ),
                "DRAIN_50_PERCENT_AFTER_TARGET",
            )
            self.assertEqual(controller.state, InventoryState.DRAINING)
            self.assertTrue(controller.profit_lock_triggered)
            self.assertEqual(
                controller.evaluate_profit_lock(
                    current_nav_jpy=107_400,
                    now_utc=NOW + timedelta(minutes=2),
                ),
                "DRAIN_75_PERCENT_APPROACHING_RETAINED_FLOOR",
            )
            action = next(
                item
                for item in controller.unwind_actions(
                    now_utc=NOW + timedelta(minutes=2),
                    terminal_deadline_utc=NOW + timedelta(hours=1),
                )
                if item.action == "REDUCE_BOT_LOT"
            )
            self.assertEqual(action.units, 8)
            controller.record_unwind_fill(
                "lot-profit-lock",
                units=8,
                realized_after_cost_jpy=7_600,
                execution_cost_jpy=25,
                now_utc=NOW + timedelta(minutes=2),
            )
            self.assertEqual(
                controller.evaluate_profit_lock(
                    current_nav_jpy=105_000,
                    now_utc=NOW + timedelta(minutes=3),
                ),
                "FORCE_FLAT_AT_CYCLE_START_PLUS_5_PERCENT",
            )
            force = controller.unwind_actions(
                now_utc=NOW + timedelta(minutes=3),
                terminal_deadline_utc=NOW + timedelta(hours=1),
            )
            self.assertEqual(len(force), 1)
            self.assertEqual(force[0].units, 2)
            self.assertEqual(force[0].reason, "PROFIT_FLOOR_FORCE_FLAT_ALL_REMAINING")
            controller.record_unwind_fill(
                "lot-profit-lock",
                units=2,
                realized_after_cost_jpy=-100,
                execution_cost_jpy=10,
                now_utc=NOW + timedelta(minutes=3),
            )
            self.assertEqual(controller.state, InventoryState.FLAT)
            self.assertEqual(controller.cycle_start_nav_jpy, 100_000)
            self.assertEqual(controller.cycle_retained_return, 0.05)
            self.assertEqual(controller.cycle_giveback_jpy, 5_000)
            self.assertEqual(controller.cycle_execution_cost_jpy, 35)
            self.assertEqual(controller.cycle_count, 1)
            reloaded = InventoryController.open(
                state_path,
                campaign_id="campaign-20260828",
                now_utc=NOW + timedelta(minutes=4),
            )
            self.assertEqual(reloaded.cycle_retained_return, 0.05)
            self.assertTrue(reloaded.profit_floor_breached)
            reloaded.stop(
                now_utc=NOW + timedelta(minutes=4),
                cooldown=timedelta(minutes=5),
            )
            next_cycle = InventoryController.open(
                state_path,
                campaign_id="campaign-next",
                now_utc=NOW + timedelta(minutes=10),
            )
            next_cycle.configure_profit_lock(
                cycle_start_nav_jpy=105_000,
                now_utc=NOW + timedelta(minutes=10),
            )
            self.assertEqual(next_cycle.cycle_count, 2)
            self.assertEqual(next_cycle.cycle_start_nav_jpy, 105_000)

    def test_hard_limit_overrides_profit_lock_and_enters_draining(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = InventoryController.open(
                Path(temp_dir) / "inventory.json",
                campaign_id="campaign-hard-limit",
                now_utc=NOW,
            )
            controller.configure_profit_lock(cycle_start_nav_jpy=100_000, now_utc=NOW)
            controller.register_fill(
                identity=LotIdentity(
                    campaign_id="campaign-hard-limit",
                    strategy_id="range_rotation",
                    lot_id="lot-hard-limit",
                ),
                pair="USD_JPY",
                side=Side.SHORT,
                units=10,
                entry_price=147.0,
                now_utc=NOW,
            )
            action = controller.evaluate_profit_lock(
                current_nav_jpy=101_000,
                hard_limit_reason="STRESS_MCP",
                now_utc=NOW + timedelta(seconds=1),
            )
            self.assertEqual(action, "HARD_LIMIT_DRAINING")
            self.assertEqual(controller.state, InventoryState.DRAINING)
            self.assertFalse(controller.profit_lock_triggered)
            self.assertEqual(controller.stop_reason, "HARD_LIMIT:STRESS_MCP")

    def test_event_receipt_is_bound_deduped_and_expiry_or_timeout_freezes_new(self) -> None:
        event = {"event_id": "event-1", "dedupe_key": "EUR_USD|REGIME_FLIP"}
        receipt = {
            "receipt_id": "receipt-1",
            "event_id": "event-1",
            "dedupe_key": "EUR_USD|REGIME_FLIP",
            "feature_snapshot_sha256": "a" * 64,
            "decision": "UNWIND",
            "regime": "MIXED",
            "allowed_strategy_ids": [],
            "risk_budget_cap_jpy": 0.0,
            "max_positions_cap": 0,
            "generated_at_utc": NOW.isoformat(),
            "expires_at_utc": (NOW + timedelta(minutes=5)).isoformat(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = InventoryController.open(
                Path(temp_dir) / "inventory.json",
                campaign_id="campaign-20260828",
                now_utc=NOW,
            )
            controller.register_fill(
                identity=_identity("lot-001"),
                pair="EUR_USD",
                side=Side.LONG,
                units=2,
                entry_price=1.1,
                now_utc=NOW,
            )
            self.assertEqual(
                controller.apply_supervision_receipt(
                    event=event, receipt=receipt, now_utc=NOW + timedelta(minutes=1)
                ),
                "APPLIED_UNWIND",
            )
            self.assertEqual(controller.state, InventoryState.DRAINING)
            self.assertEqual(
                controller.apply_supervision_receipt(
                    event=event, receipt=receipt, now_utc=NOW + timedelta(minutes=2)
                ),
                "DUPLICATE_IGNORED",
            )

        for label, invalid_receipt in (
            ("timeout", None),
            (
                "expired",
                {
                    **receipt,
                    "receipt_id": "receipt-expired",
                    "expires_at_utc": (NOW - timedelta(seconds=1)).isoformat(),
                },
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                controller = InventoryController.open(
                    Path(temp_dir) / "inventory.json",
                    campaign_id="campaign-20260828",
                    now_utc=NOW,
                )
                result = controller.apply_supervision_receipt(
                    event=event,
                    receipt=invalid_receipt,
                    now_utc=NOW,
                )
                self.assertTrue(result.startswith("FREEZE_NEW"))
                self.assertEqual(controller.state, InventoryState.FREEZE_NEW)


if __name__ == "__main__":
    unittest.main()
