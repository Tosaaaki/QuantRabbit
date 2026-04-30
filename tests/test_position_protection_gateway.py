from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.broker.position_execution import PositionProtectionGateway
from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.strategy.position_manager import ACTION_PROFIT_PROTECT, ACTION_REVIEW_EXIT, ManagedPosition, PositionManagementDecision


class PositionProtectionGatewayTest(unittest.TestCase):
    def test_stages_break_even_stop_without_sending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_PROFIT_PROTECT, stop=1.1729), snapshot=_snapshot(), send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.dependent_orders, [])
            self.assertIn('"price": "1.17290"', (root / "exec.md").read_text())

    def test_sends_break_even_stop_when_live_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_PROFIT_PROTECT, stop=1.1729), snapshot=_snapshot(), send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertTrue(summary.sent)
            self.assertEqual(client.dependent_orders[0][1]["stopLoss"]["price"], "1.17290")

    def test_blocks_stop_widening(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_PROFIT_PROTECT, stop=1.1710), snapshot=_snapshot(), send=True)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.dependent_orders, [])
            self.assertIn("SL_NOT_TIGHTER", (root / "exec.md").read_text())

    def test_closes_contradicted_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_REVIEW_EXIT, stop=None), snapshot=_snapshot(), send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.closed, [("1", "ALL")])


class FakePositionClient:
    def __init__(self) -> None:
        self.dependent_orders: list[tuple[str, dict[str, Any]]] = []
        self.closed: list[tuple[str, str]] = []

    def replace_trade_dependent_orders(self, trade_id: str, order_request: dict[str, Any]) -> dict[str, Any]:
        self.dependent_orders.append((trade_id, order_request))
        return {"relatedTransactionIDs": ["10"]}

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]:
        self.closed.append((trade_id, units))
        return {"relatedTransactionIDs": ["20"]}


def _decision(action: str, *, stop: float | None) -> PositionManagementDecision:
    return PositionManagementDecision(
        generated_at_utc="2026-05-01T00:00:00+00:00",
        action=action,
        positions=(
            ManagedPosition(
                trade_id="1",
                pair="EUR_USD",
                side="LONG",
                units=1000,
                action=action,
                unrealized_pl_jpy=90.0,
                remaining_risk_jpy=125.6,
                remaining_reward_jpy=188.4,
                same_direction_score=160.0,
                opposite_direction_score=120.0,
                recommended_stop_loss=stop,
                recommended_take_profit=None,
                reasons=("test",),
            ),
        ),
    )


def _snapshot() -> BrokerSnapshot:
    now = datetime.now(timezone.utc)
    return BrokerSnapshot(
        fetched_at_utc=now,
        positions=(
            BrokerPosition(
                trade_id="1",
                pair="EUR_USD",
                side=Side.LONG,
                units=1000,
                entry_price=1.1729,
                unrealized_pl_jpy=90.0,
                take_profit=1.1741,
                stop_loss=1.1721,
                owner=Owner.TRADER,
            ),
        ),
        quotes={"EUR_USD": Quote("EUR_USD", bid=1.1738, ask=1.1739, timestamp_utc=now)},
    )


if __name__ == "__main__":
    unittest.main()
