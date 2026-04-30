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
    ROOT,
)
from quant_rabbit.strategy.intent_generator import IntentGenerator


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
        report_path: Path = DEFAULT_AUTOTRADE_REPORT,
        live_enabled: bool = False,
    ) -> None:
        self.client = client or OandaExecutionClient()
        self.snapshot_path = snapshot_path
        self.intents_path = intents_path
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
        if positions or orders:
            summary = AutoTradeCycleSummary(
                status="MONITOR_ONLY_EXPOSURE_OPEN",
                report_path=self.report_path,
                snapshot_path=self.snapshot_path,
                intents_path=self.intents_path,
                selected_lane_id=None,
                sent=False,
                positions=positions,
                orders=orders,
                live_ready=0,
            )
            self._write_report(summary, generated_at)
            return summary

        intent_summary = IntentGenerator(output_path=self.intents_path).run(snapshot_path=self.snapshot_path)
        intents_payload = json.loads(self.intents_path.read_text())
        selected_lane_id = _first_live_ready_lane(intents_payload)
        if selected_lane_id is None:
            summary = AutoTradeCycleSummary(
                status="NO_LIVE_READY_INTENT",
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
            "",
            "## Cycle Contract",
            "",
            "- If any open position or pending order exists, the cycle is monitor-only and sends no fresh entry.",
            "- If flat, the cycle refreshes broker truth, regenerates intents, selects the first live-ready lane, and sends only when live mode is explicitly enabled.",
        ]
        self.report_path.write_text("\n".join(lines) + "\n")


def _first_live_ready_lane(payload: dict) -> str | None:
    for item in payload.get("results", []) or []:
        if isinstance(item, dict) and item.get("status") == "LIVE_READY":
            return str(item.get("lane_id") or "")
    return None


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
