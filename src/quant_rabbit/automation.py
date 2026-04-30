from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.broker.oanda import OandaExecutionClient
from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT_REPORT,
    DEFAULT_TRADER_DECISION,
    DEFAULT_TRADER_DECISION_REPORT,
    ROOT,
)
from quant_rabbit.strategy.intent_generator import IntentGenerator
from quant_rabbit.strategy.position_manager import PositionManager
from quant_rabbit.strategy.trader_brain import ACTION_SEND_ENTRY, TraderBrain


DEFAULT_AUTOTRADE_REPORT = ROOT / "docs" / "autotrade_cycle_report.md"


@dataclass(frozen=True)
class AutoTradeCycleSummary:
    status: str
    report_path: Path
    snapshot_path: Path
    intents_path: Path
    selected_lane_id: str | None
    sent: bool
    positions: int
    orders: int
    live_ready: int
    canceled_orders: tuple[str, ...] = ()
    position_management_action: str | None = None


class AutoTradeCycle:
    """One safe automated trading cycle.

    The cycle never stacks entries. Existing positions or pending orders turn the
    cycle into monitor-only until a later position-management layer exists.
    """

    def __init__(
        self,
        *,
        client: OandaExecutionClient | None = None,
        snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        decision_path: Path = DEFAULT_TRADER_DECISION,
        decision_report_path: Path = DEFAULT_TRADER_DECISION_REPORT,
        position_management_path: Path = DEFAULT_POSITION_MANAGEMENT,
        position_management_report_path: Path = DEFAULT_POSITION_MANAGEMENT_REPORT,
        report_path: Path = DEFAULT_AUTOTRADE_REPORT,
        live_enabled: bool = False,
    ) -> None:
        self.client = client or OandaExecutionClient()
        self.snapshot_path = snapshot_path
        self.intents_path = intents_path
        self.decision_path = decision_path
        self.decision_report_path = decision_report_path
        self.position_management_path = position_management_path
        self.position_management_report_path = position_management_report_path
        self.report_path = report_path
        self.live_enabled = live_enabled

    def run(self, *, send: bool = False) -> AutoTradeCycleSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        pairs = ("AUD_JPY", "AUD_USD", "EUR_JPY", "EUR_USD", "GBP_JPY", "GBP_USD", "USD_JPY")
        snapshot = self.client.snapshot(pairs)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
        positions = len(snapshot.positions)
        orders = len(snapshot.orders)
        intent_summary = IntentGenerator(output_path=self.intents_path).run(snapshot_path=self.snapshot_path)
        if positions or orders:
            decision = self._brain().run(snapshot)
            position_decision = self._position_manager().run(snapshot)
            canceled_orders: list[str] = []
            status = "MONITOR_ONLY_EXPOSURE_OPEN"
            if send and positions == 0 and decision.pending_cancel_order_ids:
                for order_id in decision.pending_cancel_order_ids:
                    self.client.cancel_order(order_id)
                    canceled_orders.append(order_id)
                status = "CANCELED_CONTAMINATED_PENDING"
            summary = AutoTradeCycleSummary(
                status=status,
                report_path=self.report_path,
                snapshot_path=self.snapshot_path,
                intents_path=self.intents_path,
                selected_lane_id=None,
                sent=False,
                positions=positions,
                orders=orders,
                live_ready=intent_summary.live_ready,
                canceled_orders=tuple(canceled_orders),
                position_management_action=position_decision.action,
            )
            self._write_report(summary, generated_at)
            return summary

        decision = self._brain().run(snapshot)
        selected_lane_id = decision.selected_lane_id if decision.action == ACTION_SEND_ENTRY else None
        if selected_lane_id is None:
            summary = AutoTradeCycleSummary(
                status=decision.action if decision.action != ACTION_SEND_ENTRY else "NO_LIVE_READY_INTENT",
                report_path=self.report_path,
                snapshot_path=self.snapshot_path,
                intents_path=self.intents_path,
                selected_lane_id=None,
                sent=False,
                positions=positions,
                orders=orders,
                live_ready=intent_summary.live_ready,
            )
            self._write_report(summary, generated_at)
            return summary

        order_summary = LiveOrderGateway(client=self.client, live_enabled=self.live_enabled).run(
            intents_path=self.intents_path,
            lane_id=selected_lane_id,
            send=send,
            confirm_live=send,
        )
        summary = AutoTradeCycleSummary(
            status=order_summary.status,
            report_path=self.report_path,
            snapshot_path=self.snapshot_path,
            intents_path=self.intents_path,
            selected_lane_id=selected_lane_id,
            sent=order_summary.sent,
            positions=positions,
            orders=orders,
            live_ready=intent_summary.live_ready,
        )
        self._write_report(summary, generated_at)
        return summary

    def _write_report(self, summary: AutoTradeCycleSummary, generated_at: str) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Autotrade Cycle Report",
            "",
            f"- Generated at UTC: `{generated_at}`",
            f"- Status: `{summary.status}`",
            f"- Positions: `{summary.positions}`",
            f"- Orders: `{summary.orders}`",
            f"- Live-ready intents: `{summary.live_ready}`",
            f"- Selected lane: `{summary.selected_lane_id}`",
            f"- Sent: `{summary.sent}`",
            f"- Canceled orders: `{', '.join(summary.canceled_orders) if summary.canceled_orders else 'none'}`",
            f"- Position management: `{summary.position_management_action or 'none'}`",
            "",
            "## Cycle Contract",
            "",
            "- If any open position or pending order exists, the cycle is monitor-only and sends no fresh entry.",
            "- If a pending entry came from a now-vetoed lane, the cycle may cancel it before waiting for the next cycle.",
            "- If flat, the cycle refreshes broker truth, regenerates intents, asks TraderBrain to compare lanes, and sends only the selected lane when live mode is explicitly enabled.",
        ]
        self.report_path.write_text("\n".join(lines) + "\n")

    def _brain(self) -> TraderBrain:
        return TraderBrain(
            intents_path=self.intents_path,
            output_path=self.decision_path,
            report_path=self.decision_report_path,
        )

    def _position_manager(self) -> PositionManager:
        return PositionManager(
            trader_decision_path=self.decision_path,
            output_path=self.position_management_path,
            report_path=self.position_management_report_path,
        )


def _snapshot_to_json(snapshot) -> str:
    payload = {
        "fetched_at_utc": snapshot.fetched_at_utc.isoformat(),
        "positions": [
            {
                "trade_id": pos.trade_id,
                "pair": pos.pair,
                "side": pos.side.value,
                "units": pos.units,
                "entry_price": pos.entry_price,
                "unrealized_pl_jpy": pos.unrealized_pl_jpy,
                "take_profit": pos.take_profit,
                "stop_loss": pos.stop_loss,
                "owner": pos.owner.value,
            }
            for pos in snapshot.positions
        ],
        "orders": [
            {
                "order_id": order.order_id,
                "pair": order.pair,
                "order_type": order.order_type,
                "trade_id": order.trade_id,
                "price": order.price,
                "state": order.state,
                "units": order.units,
                "owner": order.owner.value,
            }
            for order in snapshot.orders
        ],
        "quotes": {
            pair: {
                "bid": quote.bid,
                "ask": quote.ask,
                "timestamp_utc": quote.timestamp_utc.isoformat(),
            }
            for pair, quote in snapshot.quotes.items()
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
