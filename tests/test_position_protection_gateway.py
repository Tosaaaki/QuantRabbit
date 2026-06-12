from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from quant_rabbit.broker.position_execution import PositionProtectionGateway
from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.strategy.position_manager import (
    ACTION_BREAK_EVEN_STOP,
    ACTION_PROFIT_PROTECT,
    ACTION_REPAIR_TAKE_PROFIT,
    ACTION_REVIEW_EXIT,
    ACTION_TAKE_PROFIT_MARKET,
    ManagedPosition,
    PositionManagementDecision,
)


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

    def test_sends_sl_free_break_even_stop_when_live_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(ACTION_BREAK_EVEN_STOP, stop=1.1729),
                snapshot=_snapshot(stop_loss=None),
                send=True,
            )

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

    def test_blocks_plain_review_exit_when_auto_close_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {"QR_DISABLE_AUTO_CLOSE": "1"},
            clear=False,
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_REVIEW_EXIT, stop=None), snapshot=_snapshot(), send=True)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.closed, [])
            self.assertIn("REVIEW_EXIT_GATE_AB_REQUIRED", (root / "exec.md").read_text())

    def test_allows_gpt_verified_review_exit_when_auto_close_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {"QR_DISABLE_AUTO_CLOSE": "1"},
            clear=False,
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(
                    ACTION_REVIEW_EXIT,
                    stop=None,
                    reasons=(
                        "gpt-close: accepted gpt_trader CLOSE receipt passed Gate A/B; "
                        "execute only through PositionProtectionGateway",
                    ),
                ),
                snapshot=_snapshot(),
                send=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.closed, [("1", "ALL")])

    def test_blocks_gpt_verified_review_exit_when_close_spread_too_wide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "QR_DISABLE_AUTO_CLOSE": "1",
                "QR_POSITION_CLOSE_SPREAD_OVERRIDE": "",
            },
            clear=False,
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(
                    ACTION_REVIEW_EXIT,
                    stop=None,
                    reasons=(
                        "gpt-close: accepted gpt_trader CLOSE receipt passed Gate A/B; "
                        "execute only through PositionProtectionGateway",
                    ),
                ),
                snapshot=_snapshot(unrealized_pl_jpy=-90.0, quote_bid=1.1700, quote_ask=1.1720),
                send=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.closed, [])
            self.assertIn("POSITION_CLOSE_SPREAD_TOO_WIDE", (root / "exec.md").read_text())

    def test_allows_wide_spread_close_with_explicit_operator_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "QR_DISABLE_AUTO_CLOSE": "1",
                "QR_POSITION_CLOSE_SPREAD_OVERRIDE": "1",
            },
            clear=False,
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(
                    ACTION_REVIEW_EXIT,
                    stop=None,
                    reasons=(
                        "gpt-close: accepted gpt_trader CLOSE receipt passed Gate A/B; "
                        "execute only through PositionProtectionGateway",
                    ),
                ),
                snapshot=_snapshot(unrealized_pl_jpy=-90.0, quote_bid=1.1700, quote_ask=1.1720),
                send=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.closed, [("1", "ALL")])

    def test_blocks_structural_review_exit_without_explicit_opt_in_when_auto_close_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "QR_DISABLE_AUTO_CLOSE": "1",
                "QR_ALLOW_STRUCTURAL_AUTO_CLOSE": "",
            },
            clear=False,
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(
                    ACTION_REVIEW_EXIT,
                    stop=None,
                    reasons=(
                        "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-90 JPY)",
                        "next-generation entry thesis ledger present → structural loss-cut remains executable under QR_DISABLE_AUTO_CLOSE=1",
                    ),
                ),
                snapshot=_snapshot(unrealized_pl_jpy=-90.0),
                send=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.closed, [])
            self.assertIn("QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1", (root / "exec.md").read_text())

    def test_allows_structural_review_exit_with_explicit_opt_in_when_auto_close_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "QR_DISABLE_AUTO_CLOSE": "1",
                "QR_ALLOW_STRUCTURAL_AUTO_CLOSE": "1",
            },
            clear=False,
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(
                    ACTION_REVIEW_EXIT,
                    stop=None,
                    reasons=(
                        "loss-cut: structural OB broken across 2 TFs (M15@1.17000, H1@1.17100) (-90 JPY)",
                        "next-generation entry thesis ledger present → structural loss-cut remains executable under QR_DISABLE_AUTO_CLOSE=1",
                    ),
                ),
                snapshot=_snapshot(unrealized_pl_jpy=-90.0),
                send=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.closed, [("1", "ALL")])

    def test_blocks_confidence_collapse_review_exit_even_with_structural_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "QR_DISABLE_AUTO_CLOSE": "1",
                "QR_ALLOW_STRUCTURAL_AUTO_CLOSE": "1",
            },
            clear=False,
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(
                    ACTION_REVIEW_EXIT,
                    stop=None,
                    reasons=(
                        "loss-cut: entry thesis confidence collapse: entry UP conf=0.62 → latest UP conf=0.31 "
                        "(< 0.50× entry); technical invalidation confirmed against LONG: M15 MACD-; M30 MACD-; M5 cloud- (-109 JPY)",
                        "next-generation entry thesis ledger present → structural loss-cut remains executable under QR_DISABLE_AUTO_CLOSE=1",
                    ),
                ),
                snapshot=_snapshot(unrealized_pl_jpy=-90.0),
                send=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.closed, [])
            self.assertIn("REVIEW_EXIT_GATE_AB_REQUIRED", (root / "exec.md").read_text())

    def test_closes_profitable_position_with_take_profit_market_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_TAKE_PROFIT_MARKET, stop=None), snapshot=_snapshot(), send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.closed, [("1", "ALL")])

    def test_blocks_take_profit_market_when_position_is_not_profitable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(ACTION_TAKE_PROFIT_MARKET, stop=None),
                snapshot=_snapshot(unrealized_pl_jpy=-1.0),
                send=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.closed, [])
            self.assertIn("PROFIT_MARKET_CLOSE_NOT_PROFITABLE", (root / "exec.md").read_text())

    def test_manual_position_allows_take_profit_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(ACTION_REPAIR_TAKE_PROFIT, stop=None, take_profit=1.1750),
                snapshot=_snapshot(owner=Owner.UNKNOWN, take_profit=None, stop_loss=None),
                send=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.dependent_orders[0][1]["takeProfit"]["price"], "1.17500")
            self.assertNotIn("stopLoss", client.dependent_orders[0][1])

    def test_manual_position_blocks_stop_loss_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(ACTION_PROFIT_PROTECT, stop=1.1729),
                snapshot=_snapshot(owner=Owner.MANUAL, stop_loss=None),
                send=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.dependent_orders, [])
            self.assertIn("MANUAL_POSITION_STOP_LOSS_FORBIDDEN", (root / "exec.md").read_text())

    def test_manual_position_blocks_market_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(ACTION_REVIEW_EXIT, stop=None),
                snapshot=_snapshot(owner=Owner.UNKNOWN),
                send=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(client.closed, [])
            self.assertIn("MANUAL_POSITION_CLOSE_FORBIDDEN", (root / "exec.md").read_text())

    def test_successful_close_receipt_survives_later_broker_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FailingProtectionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=PositionManagementDecision(
                    generated_at_utc="2026-05-01T00:00:00+00:00",
                    snapshot_fetched_at_utc="2026-05-01T00:00:00+00:00",
                    action="MIXED",
                    positions=(
                        ManagedPosition(
                            trade_id="1",
                            pair="EUR_USD",
                            side="LONG",
                            units=1000,
                            action=ACTION_REVIEW_EXIT,
                            unrealized_pl_jpy=-100.0,
                            remaining_risk_jpy=100.0,
                            remaining_reward_jpy=0.0,
                            same_direction_score=None,
                            opposite_direction_score=None,
                            recommended_stop_loss=None,
                            recommended_take_profit=None,
                            reasons=("test",),
                            owner="trader",
                        ),
                        ManagedPosition(
                            trade_id="2",
                            pair="EUR_USD",
                            side="LONG",
                            units=1000,
                            action=ACTION_REPAIR_TAKE_PROFIT,
                            unrealized_pl_jpy=80.0,
                            remaining_risk_jpy=50.0,
                            remaining_reward_jpy=150.0,
                            same_direction_score=None,
                            opposite_direction_score=None,
                            recommended_stop_loss=None,
                            recommended_take_profit=1.1760,
                            reasons=("test",),
                            owner="trader",
                        ),
                    ),
                ),
                snapshot=_two_position_snapshot(),
                send=True,
            )

            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "PARTIAL_SENT_WITH_BLOCKS")
        self.assertTrue(summary.sent)
        self.assertEqual(client.closed, [("1", "ALL")])
        self.assertTrue(payload["actions"][0]["sent"])
        self.assertEqual(payload["actions"][0]["response"]["relatedTransactionIDs"], ["20"])
        self.assertFalse(payload["actions"][1]["sent"])
        self.assertEqual(payload["actions"][1]["issues"][0]["code"], "POSITION_PROTECTION_SEND_FAILED")


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


class FailingProtectionClient(FakePositionClient):
    def replace_trade_dependent_orders(self, trade_id: str, order_request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("dependent order replace failed")


def _decision(
    action: str,
    *,
    stop: float | None,
    take_profit: float | None = None,
    reasons: tuple[str, ...] = ("test",),
) -> PositionManagementDecision:
    return PositionManagementDecision(
        generated_at_utc="2026-05-01T00:00:00+00:00",
        snapshot_fetched_at_utc="2026-05-01T00:00:00+00:00",
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
                recommended_take_profit=take_profit,
                reasons=reasons,
            ),
        ),
    )


def _snapshot(
    *,
    owner: Owner = Owner.TRADER,
    take_profit: float | None = 1.1741,
    stop_loss: float | None = 1.1721,
    unrealized_pl_jpy: float = 90.0,
    quote_bid: float = 1.1738,
    quote_ask: float = 1.1739,
) -> BrokerSnapshot:
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
                unrealized_pl_jpy=unrealized_pl_jpy,
                take_profit=take_profit,
                stop_loss=stop_loss,
                owner=owner,
            ),
        ),
        quotes={"EUR_USD": Quote("EUR_USD", bid=quote_bid, ask=quote_ask, timestamp_utc=now)},
    )


def _two_position_snapshot() -> BrokerSnapshot:
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
                unrealized_pl_jpy=-100.0,
                take_profit=1.1741,
                stop_loss=1.1721,
                owner=Owner.TRADER,
            ),
            BrokerPosition(
                trade_id="2",
                pair="EUR_USD",
                side=Side.LONG,
                units=1000,
                entry_price=1.1729,
                unrealized_pl_jpy=80.0,
                take_profit=1.1741,
                stop_loss=1.1721,
                owner=Owner.TRADER,
            ),
        ),
        quotes={"EUR_USD": Quote("EUR_USD", bid=1.1738, ask=1.1739, timestamp_utc=now)},
    )


if __name__ == "__main__":
    unittest.main()
