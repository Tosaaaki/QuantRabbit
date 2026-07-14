from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from quant_rabbit.broker.position_execution import (
    PositionProtectionGateway as _PositionProtectionGateway,
)
from quant_rabbit.models import BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.operator_manual import OPERATOR_MANUAL_POSITION_PACKET
from quant_rabbit.position_execution_evidence import (
    POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME,
    POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD,
    POSITION_EXECUTION_SNAPSHOT_EVIDENCE_MAX_BYTES,
    POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA,
    load_position_execution_snapshot_evidence,
    persist_position_execution_snapshot_evidence,
)
from quant_rabbit.strategy.position_manager import (
    ACTION_BREAK_EVEN_STOP,
    ACTION_PROFIT_PROTECT,
    ACTION_REPAIR_TAKE_PROFIT,
    ACTION_REVIEW_EXIT,
    ACTION_TAKE_PROFIT_MARKET,
    ManagedPosition,
    PositionManagementDecision,
)


TEST_SNAPSHOT_AT = datetime.now(timezone.utc)


class PositionProtectionGateway(_PositionProtectionGateway):
    """Keep the gateway clock aligned with the module's fixed test fixtures."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("clock", lambda: TEST_SNAPSHOT_AT)
        super().__init__(*args, **kwargs)


class PositionProtectionGatewayTest(unittest.TestCase):
    def test_fixture_clock_remains_consistent_after_discovery_delay(self) -> None:
        delayed_fixture_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
        with (
            patch(f"{__name__}.TEST_SNAPSHOT_AT", delayed_fixture_at),
            tempfile.TemporaryDirectory() as tmp,
        ):
            root = Path(tmp)
            summary = PositionProtectionGateway(
                client=FakePositionClient(),
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(ACTION_PROFIT_PROTECT, stop=1.1729),
                snapshot=_snapshot(),
                send=False,
            )

        self.assertEqual(summary.status, "STAGED")

    def test_predictive_scout_rejects_stale_exit_management_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(
                decision=_decision(ACTION_PROFIT_PROTECT, stop=1.1729, take_profit=1.1750),
                snapshot=_snapshot(
                    raw={
                        "tradeClientExtensions": {
                            "comment": "qr-vnext role=BIDASK_REPLAY_CONTRARIAN_SCOUT vehicle=psv-test"
                        }
                    }
                ),
                send=True,
            )

            report = (root / "exec.md").read_text()

        self.assertEqual(summary.status, "BLOCKED")
        self.assertFalse(summary.sent)
        self.assertEqual(client.dependent_orders, [])
        self.assertEqual(client.closed, [])
        self.assertIn("PREDICTIVE_SCOUT_EXIT_GEOMETRY_FROZEN", report)

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
            payload = json.loads((root / "exec.json").read_text())
            self.assertFalse(payload["actions"][0]["broker_post_attempted"])
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
            payload = json.loads((root / "exec.json").read_text())
            self.assertTrue(payload["actions"][0]["broker_post_attempted"])
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

    def test_close_uses_position_protection_provenance_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"QR_DISABLE_AUTO_CLOSE": ""}, clear=False):
            root = Path(tmp)
            client = ProvenancePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_REVIEW_EXIT, stop=None), snapshot=_snapshot(), send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.closed_with_provenance, [("1", "ALL", "position_protection_gateway")])
            payload = json.loads((root / "exec.json").read_text())
            self.assertEqual(payload["actions"][0]["request"]["provenance"], "position_protection_gateway")

    def test_close_blocks_raw_close_fallback_when_provenance_method_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"QR_DISABLE_AUTO_CLOSE": ""}, clear=False):
            root = Path(tmp)
            client = RawOnlyPositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=_decision(ACTION_REVIEW_EXIT, stop=None), snapshot=_snapshot(), send=True)

            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertFalse(summary.sent)
        self.assertEqual(client.closed, [])
        self.assertEqual(payload["actions"][0]["issues"][0]["code"], "POSITION_CLOSE_SEND_FAILED")
        self.assertIn("close_trade_with_provenance", payload["actions"][0]["issues"][0]["message"])

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

    def test_allows_gpt_verified_review_exit_when_session_cap_covers_spread(self) -> None:
        snapshot_at = datetime(2026, 6, 12, 16, 0, tzinfo=timezone.utc)
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
                clock=lambda: snapshot_at,
            ).run(
                decision=_decision(
                    ACTION_REVIEW_EXIT,
                    stop=None,
                    snapshot_fetched_at_utc=snapshot_at.isoformat(),
                    reasons=(
                        "gpt-close: accepted gpt_trader CLOSE receipt passed Gate A/B; "
                        "execute only through PositionProtectionGateway",
                    ),
                ),
                snapshot=_snapshot(
                    unrealized_pl_jpy=-90.0,
                    quote_bid=1.17000,
                    quote_ask=1.17017,
                    fetched_at_utc=snapshot_at,
                ),
                send=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.closed, [("1", "ALL")])

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
            payload = json.loads((root / "exec.json").read_text())
            action = payload["actions"][0]
            self.assertEqual(action["pair"], "EUR_USD")
            self.assertEqual(action["owner"], "unknown")
            self.assertEqual(
                action["broker_position_identity"],
                {
                    "snapshot_fetched_at_utc": action["broker_position_identity"][
                        "snapshot_fetched_at_utc"
                    ],
                    "trade_id": "1",
                    "pair": "EUR_USD",
                    "owner": "unknown",
                },
            )

    def test_blocks_managed_pair_that_contradicts_broker_position_identity(self) -> None:
        decision = _decision(
            ACTION_REPAIR_TAKE_PROFIT,
            stop=None,
            take_profit=1.1750,
        )
        managed = decision.positions[0]
        mismatched = PositionManagementDecision(
            generated_at_utc=decision.generated_at_utc,
            snapshot_fetched_at_utc=decision.snapshot_fetched_at_utc,
            action=decision.action,
            positions=(
                ManagedPosition(
                    trade_id=managed.trade_id,
                    pair="GBP_USD",
                    side=managed.side,
                    units=managed.units,
                    action=managed.action,
                    unrealized_pl_jpy=managed.unrealized_pl_jpy,
                    remaining_risk_jpy=managed.remaining_risk_jpy,
                    remaining_reward_jpy=managed.remaining_reward_jpy,
                    same_direction_score=managed.same_direction_score,
                    opposite_direction_score=managed.opposite_direction_score,
                    recommended_stop_loss=managed.recommended_stop_loss,
                    recommended_take_profit=managed.recommended_take_profit,
                    reasons=managed.reasons,
                    owner=managed.owner,
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
            ).run(decision=mismatched, snapshot=_snapshot(), send=True)
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertFalse(summary.sent)
        self.assertEqual(client.dependent_orders, [])
        self.assertEqual(payload["actions"][0]["pair"], "EUR_USD")
        self.assertEqual(
            payload["actions"][0]["issues"][0]["code"],
            "BROKER_POSITION_PAIR_MISMATCH",
        )

    def test_blocks_decision_bound_to_a_different_broker_snapshot(self) -> None:
        decision = _decision(ACTION_REPAIR_TAKE_PROFIT, stop=None, take_profit=1.1750)
        mismatched = PositionManagementDecision(
            generated_at_utc=decision.generated_at_utc,
            snapshot_fetched_at_utc=(TEST_SNAPSHOT_AT - timedelta(seconds=1)).isoformat(),
            action=decision.action,
            positions=decision.positions,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(decision=mismatched, snapshot=_snapshot(), send=True)
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertEqual(
            payload["actions"][0]["issues"][0]["code"],
            "BROKER_SNAPSHOT_DECISION_MISMATCH",
        )

    def test_blocks_stale_broker_snapshot_before_post(self) -> None:
        stale_at = TEST_SNAPSHOT_AT - timedelta(minutes=6)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(
                decision=_decision(
                    ACTION_REPAIR_TAKE_PROFIT,
                    stop=None,
                    take_profit=1.1750,
                    snapshot_fetched_at_utc=stale_at.isoformat(),
                ),
                snapshot=_snapshot(fetched_at_utc=stale_at),
                send=True,
            )
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertEqual(
            payload["actions"][0]["issues"][0]["code"],
            "BROKER_SNAPSHOT_STALE",
        )

    def test_operator_manual_auto_tp_modify_false_blocks_take_profit_replace(self) -> None:
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
                snapshot=_snapshot(
                    owner=Owner.OPERATOR_MANUAL,
                    take_profit=1.1741,
                    stop_loss=None,
                    raw={
                        "operator_manual_position": {
                            "packet_type": OPERATOR_MANUAL_POSITION_PACKET,
                            "classification": "OPERATOR_MANUAL",
                            "management_intent": "KEEP",
                            "auto_tp_modify_allowed": False,
                            "auto_sl_attach_allowed": False,
                            "loss_side_auto_close_allowed": False,
                            "same_theme_auto_add_allowed": False,
                        }
                    },
                ),
                send=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.dependent_orders, [])
            report = (root / "exec.md").read_text()
            self.assertIn("OPERATOR_MANUAL_TP_MODIFY_FORBIDDEN", report)
            self.assertIn("auto_tp_modify_allowed=false", report)

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
                    generated_at_utc=TEST_SNAPSHOT_AT.isoformat(),
                    snapshot_fetched_at_utc=TEST_SNAPSHOT_AT.isoformat(),
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

    def test_rechecks_snapshot_freshness_before_each_broker_post(self) -> None:
        clock_now = [TEST_SNAPSHOT_AT]

        class AdvancingClient(FakePositionClient):
            def replace_trade_dependent_orders(
                self,
                trade_id: str,
                order_request: dict[str, Any],
            ) -> dict[str, Any]:
                response = super().replace_trade_dependent_orders(
                    trade_id,
                    order_request,
                )
                clock_now[0] = clock_now[0] + timedelta(seconds=301)
                return response

        decision = PositionManagementDecision(
            generated_at_utc=TEST_SNAPSHOT_AT.isoformat(),
            snapshot_fetched_at_utc=TEST_SNAPSHOT_AT.isoformat(),
            action="MIXED",
            positions=tuple(
                ManagedPosition(
                    trade_id=trade_id,
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
                    recommended_take_profit=take_profit,
                    reasons=("test",),
                    owner="trader",
                )
                for trade_id, take_profit in (("1", 1.1750), ("2", 1.1760))
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = AdvancingClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: clock_now[0],
            ).run(
                decision=decision,
                snapshot=_two_position_snapshot(),
                send=True,
            )
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "PARTIAL_SENT_WITH_BLOCKS")
        self.assertEqual([trade_id for trade_id, _ in client.dependent_orders], ["1"])
        self.assertTrue(payload["actions"][0]["broker_post_attempted"])
        self.assertFalse(payload["actions"][1]["broker_post_attempted"])
        self.assertEqual(
            payload["actions"][0]["send_boundary_checked_at_utc"],
            TEST_SNAPSHOT_AT.isoformat(),
        )
        self.assertEqual(
            payload["actions"][1]["send_boundary_checked_at_utc"],
            (TEST_SNAPSHOT_AT + timedelta(seconds=301)).isoformat(),
        )
        self.assertEqual(
            payload["actions"][1]["issues"][0]["code"],
            "BROKER_SNAPSHOT_STALE",
        )

    def test_rejects_duplicate_decision_trade_ids_before_any_broker_post(self) -> None:
        original = _decision(
            ACTION_REPAIR_TAKE_PROFIT,
            stop=None,
            take_profit=1.1750,
        )
        duplicate = PositionManagementDecision(
            generated_at_utc=original.generated_at_utc,
            snapshot_fetched_at_utc=original.snapshot_fetched_at_utc,
            action="MIXED",
            positions=(original.positions[0], original.positions[0]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(decision=duplicate, snapshot=_snapshot(), send=True)
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertTrue(
            all(
                "POSITION_DECISION_TRADE_ID_DUPLICATE"
                in {issue["code"] for issue in action["issues"]}
                for action in payload["actions"]
            )
        )

    def test_rejects_duplicate_snapshot_trade_ids_before_any_broker_post(self) -> None:
        base = _snapshot()
        duplicate_snapshot = BrokerSnapshot(
            fetched_at_utc=base.fetched_at_utc,
            positions=(base.positions[0], base.positions[0]),
            orders=base.orders,
            quotes=base.quotes,
            account=base.account,
            home_conversions=base.home_conversions,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(
                decision=_decision(
                    ACTION_REPAIR_TAKE_PROFIT,
                    stop=None,
                    take_profit=1.1750,
                ),
                snapshot=duplicate_snapshot,
                send=True,
            )
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertIn(
            "BROKER_SNAPSHOT_TRADE_ID_DUPLICATE",
            {issue["code"] for issue in payload["actions"][0]["issues"]},
        )

    def test_snapshot_evidence_persistence_failure_blocks_before_post(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch(
            "quant_rabbit.broker.position_execution."
            "persist_position_execution_snapshot_evidence",
            side_effect=OSError("evidence disk unavailable"),
        ):
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(
                decision=_decision(
                    ACTION_REPAIR_TAKE_PROFIT,
                    stop=None,
                    take_profit=1.1750,
                ),
                snapshot=_snapshot(),
                send=True,
            )
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertIsNone(payload["pre_send_broker_snapshot_evidence"])
        self.assertIn(
            "BROKER_SNAPSHOT_EVIDENCE_PERSIST_FAILED",
            {issue["code"] for issue in payload["actions"][0]["issues"]},
        )

    def test_invalid_snapshot_evidence_body_blocks_before_post(self) -> None:
        base = _snapshot()
        invalid_snapshot = replace(
            base,
            positions=(
                replace(
                    base.positions[0],
                    unrealized_pl_jpy="100",
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(
                decision=_decision(
                    ACTION_REPAIR_TAKE_PROFIT,
                    stop=None,
                    take_profit=1.1750,
                ),
                snapshot=invalid_snapshot,
                send=True,
            )
            payload = json.loads((root / "exec.json").read_text())
            evidence_dir_exists = (
                root / POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME
            ).exists()

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertIsNone(payload["pre_send_broker_snapshot_evidence"])
        self.assertFalse(evidence_dir_exists)
        self.assertIn(
            "BROKER_SNAPSHOT_EVIDENCE_PERSIST_FAILED",
            {issue["code"] for issue in payload["actions"][0]["issues"]},
        )

    def test_quote_key_pair_mismatch_blocks_before_post(self) -> None:
        base = _snapshot()
        mismatched_snapshot = replace(
            base,
            quotes={
                "EUR_USD": Quote(
                    "GBP_USD",
                    bid=1.1738,
                    ask=1.1739,
                    timestamp_utc=TEST_SNAPSHOT_AT,
                )
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(
                decision=_decision(
                    ACTION_REPAIR_TAKE_PROFIT,
                    stop=None,
                    take_profit=1.1750,
                ),
                snapshot=mismatched_snapshot,
                send=True,
            )
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertIn(
            "BROKER_SNAPSHOT_EVIDENCE_PERSIST_FAILED",
            {issue["code"] for issue in payload["actions"][0]["issues"]},
        )

    def test_noncanonical_home_conversion_blocks_before_post(self) -> None:
        invalid_snapshot = replace(
            _snapshot(),
            home_conversions={"usd": 1.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakePositionClient()
            summary = PositionProtectionGateway(
                client=client,
                output_path=root / "exec.json",
                report_path=root / "exec.md",
                live_enabled=True,
                clock=lambda: TEST_SNAPSHOT_AT,
            ).run(
                decision=_decision(
                    ACTION_REPAIR_TAKE_PROFIT,
                    stop=None,
                    take_profit=1.1750,
                ),
                snapshot=invalid_snapshot,
                send=True,
            )
            payload = json.loads((root / "exec.json").read_text())

        self.assertEqual(summary.status, "BLOCKED")
        self.assertEqual(client.dependent_orders, [])
        self.assertIn(
            "BROKER_SNAPSHOT_EVIDENCE_PERSIST_FAILED",
            {issue["code"] for issue in payload["actions"][0]["issues"]},
        )

    def test_snapshot_evidence_rejects_content_addressed_truncated_body(self) -> None:
        body = {
            "evidence_schema": POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA,
        }
        raw = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt_path = root / "position_execution.json"
            evidence_path = (
                root
                / POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME
                / f"{digest}.json"
            )
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_bytes(raw)
            receipt = {
                POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD: {
                    "schema": POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA,
                    "sha256": digest,
                    "byte_count": len(raw),
                    "path": (
                        f"{POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME}/"
                        f"{digest}.json"
                    ),
                }
            }

            with self.assertRaisesRegex(ValueError, "body fields are invalid"):
                load_position_execution_snapshot_evidence(
                    receipt_payload=receipt,
                    receipt_path=receipt_path,
                )

    def test_snapshot_evidence_directory_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "redirected"
            actual.mkdir()
            evidence_dir = root / POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME
            evidence_dir.symlink_to(actual, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                persist_position_execution_snapshot_evidence(
                    snapshot=_snapshot(),
                    receipt_path=root / "position_execution.json",
                )

    def test_snapshot_evidence_size_is_checked_before_bounded_read(self) -> None:
        digest = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence_dir = root / POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME
            evidence_dir.mkdir()
            evidence_path = evidence_dir / f"{digest}.json"
            with evidence_path.open("wb") as handle:
                handle.truncate(POSITION_EXECUTION_SNAPSHOT_EVIDENCE_MAX_BYTES + 1)
            receipt = {
                POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD: {
                    "schema": POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA,
                    "sha256": digest,
                    "byte_count": 1,
                    "path": (
                        f"{POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME}/"
                        f"{digest}.json"
                    ),
                }
            }

            with self.assertRaisesRegex(ValueError, "digest/size mismatch"):
                load_position_execution_snapshot_evidence(
                    receipt_payload=receipt,
                    receipt_path=root / "position_execution.json",
                )

    def test_snapshot_evidence_huge_integer_is_normalized_to_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt_path = root / "position_execution.json"
            proof = persist_position_execution_snapshot_evidence(
                snapshot=_snapshot(),
                receipt_path=receipt_path,
            )
            original_path = root / str(proof["path"])
            body = json.loads(original_path.read_text())
            body["positions"][0]["unrealized_pl_jpy"] = 10**1000
            raw = json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            altered_path = (
                root
                / POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME
                / f"{digest}.json"
            )
            altered_path.write_bytes(raw)
            receipt = {
                POSITION_EXECUTION_SNAPSHOT_EVIDENCE_FIELD: {
                    "schema": POSITION_EXECUTION_SNAPSHOT_EVIDENCE_SCHEMA,
                    "sha256": digest,
                    "byte_count": len(raw),
                    "path": (
                        f"{POSITION_EXECUTION_SNAPSHOT_EVIDENCE_DIRNAME}/"
                        f"{digest}.json"
                    ),
                }
            }

            with self.assertRaisesRegex(ValueError, "is not finite"):
                load_position_execution_snapshot_evidence(
                    receipt_payload=receipt,
                    receipt_path=receipt_path,
                )


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

    def close_trade_with_provenance(self, trade_id: str, units: str = "ALL", *, provenance: str) -> dict[str, Any]:
        self.closed.append((trade_id, units))
        return {"relatedTransactionIDs": ["20"], "provenance": provenance}


class ProvenancePositionClient(FakePositionClient):
    def __init__(self) -> None:
        super().__init__()
        self.closed_with_provenance: list[tuple[str, str, str]] = []

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]:
        raise AssertionError("position gateway must use close_trade_with_provenance when available")

    def close_trade_with_provenance(self, trade_id: str, units: str = "ALL", *, provenance: str) -> dict[str, Any]:
        self.closed_with_provenance.append((trade_id, units, provenance))
        return {"relatedTransactionIDs": ["20"]}


class RawOnlyPositionClient:
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
    snapshot_fetched_at_utc: str | None = None,
) -> PositionManagementDecision:
    return PositionManagementDecision(
        generated_at_utc=TEST_SNAPSHOT_AT.isoformat(),
        snapshot_fetched_at_utc=(
            snapshot_fetched_at_utc or TEST_SNAPSHOT_AT.isoformat()
        ),
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
    fetched_at_utc: datetime | None = None,
    raw: dict[str, Any] | None = None,
) -> BrokerSnapshot:
    now = fetched_at_utc or TEST_SNAPSHOT_AT
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
                raw=raw or {},
            ),
        ),
        quotes={"EUR_USD": Quote("EUR_USD", bid=quote_bid, ask=quote_ask, timestamp_utc=now)},
    )


def _two_position_snapshot() -> BrokerSnapshot:
    now = TEST_SNAPSHOT_AT
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
