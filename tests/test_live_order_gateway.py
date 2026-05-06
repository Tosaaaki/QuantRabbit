from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import quant_rabbit.broker.execution as execution_module
from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.models import AccountSummary, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.risk import OANDA_JP_RETAIL_FX_MARGIN_RATE


class LiveOrderGatewayTest(unittest.TestCase):
    def test_stages_oanda_stop_order_without_sending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=_intents(root))

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            order = payload["order_request"]
            self.assertEqual(order["type"], "STOP")
            self.assertEqual(order["instrument"], "EUR_USD")
            self.assertEqual(order["units"], "1000")
            self.assertEqual(order["price"], "1.17330")
            self.assertEqual(order["takeProfitOnFill"]["price"], "1.17450")
            self.assertEqual(order["stopLossOnFill"]["price"], "1.17250")

    def test_stages_oanda_market_order_without_entry_price(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=_intents(root, order_type="MARKET"), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            order = payload["order_request"]
            self.assertEqual(order["type"], "MARKET")
            self.assertEqual(order["timeInForce"], "FOK")
            self.assertNotIn("price", order)

    def test_send_requires_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            self.assertIn("LIVE_CONFIRMATION_REQUIRED", (root / "report.md").read_text())

    def test_send_posts_only_after_live_validation_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(
                intents_path=_intents(root),
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertTrue(summary.sent)
            self.assertEqual(len(client.orders), 1)

    def test_batch_send_posts_multiple_live_ready_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root)
            payload = json.loads(intents.read_text())
            second = json.loads(json.dumps(payload["results"][0]))
            second["lane_id"] = "lane:EUR_USD:LONG:reload"
            second["intent"]["entry"] = 1.17360
            second["intent"]["tp"] = 1.17480
            second["intent"]["sl"] = 1.17280
            payload["results"].append(second)
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:reload"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 2)
            self.assertTrue(summary.sent)
            self.assertEqual(len(client.orders), 2)
            result = json.loads((root / "request.json").read_text())
            self.assertEqual(len(result["orders"]), 2)

    def test_batch_send_does_not_double_count_sent_margin_from_fresh_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = MutatingExecutionClient(margin_used_jpy=168_000.0)
            intents = _intents(root, order_type="MARKET")
            payload = json.loads(intents.read_text())
            second = json.loads(json.dumps(payload["results"][0]))
            second["lane_id"] = "lane:EUR_USD:LONG:reload"
            second["intent"]["tp"] = 1.17480
            payload["results"].append(second)
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:reload"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 2)
            result = json.loads((root / "request.json").read_text())
            second_result = result["orders"][1]
            self.assertEqual(second_result["scaled_units"], 1000)
            self.assertNotIn(
                "BASKET_DOWNSIZED_FOR_CAPACITY",
                {issue["code"] for issue in second_result["risk_issues"]},
            )

    def test_default_risk_cap_reads_daily_target_state_before_policy_literal(self) -> None:
        original_reader = execution_module._per_trade_risk_from_state
        execution_module._per_trade_risk_from_state = lambda: 100.0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

                self.assertEqual(summary.status, "BLOCKED")
                payload = json.loads((root / "request.json").read_text())
                self.assertIn("LOSS_CAP_EXCEEDED", {issue["code"] for issue in payload["risk_issues"]})
        finally:
            execution_module._per_trade_risk_from_state = original_reader

    def test_send_blocks_when_candidate_exceeds_margin_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = client.snapshot_value.fetched_at_utc
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=AccountSummary(
                    nav_jpy=220_145.7765,
                    balance_jpy=208_945.7765,
                    margin_used_jpy=156_414.0,
                    margin_available_jpy=63_831.7765,
                    fetched_at_utc=now,
                ),
            )
            intents = _intents(root, order_type="MARKET")
            payload = json.loads(intents.read_text())
            payload["results"][0]["intent"]["units"] = 13_000
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                max_loss_jpy=2_000.0,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            self.assertIn("MARGIN_UTILIZATION_CAP_EXCEEDED", {issue["code"] for issue in result["risk_issues"]})

    def test_hedge_intent_uses_open_only_position_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "range_trader",
                        "campaign_role": "BACKUP_OR_RELOAD",
                        "position_intent": "HEDGE",
                    },
                )
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["order_request"]["positionFill"], "OPEN_ONLY")

    def test_existing_protected_position_blocks_when_portfolio_budget_exceeded(self) -> None:
        # Per AGENT_CONTRACT §3.5: portfolio cap is the WHOLE-DAY risk budget,
        # not the per-trade slice. We assert the gateway blocks new entries
        # only when open_risk + candidate_risk exceeds the day's budget — set
        # an explicit small portfolio cap to force that condition without
        # relying on the per-trade resolver.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=client.snapshot_value.fetched_at_utc,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=5000,
                        entry_price=1.1710,
                        take_profit=1.1750,
                        stop_loss=1.1690,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=client.snapshot_value.account,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                portfolio_loss_cap_jpy=500.0,
            ).run(
                intents_path=_intents(root),
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("PORTFOLIO_LOSS_CAP_EXCEEDED", {issue["code"] for issue in payload["risk_issues"]})

    def test_existing_break_even_trader_position_allows_portfolio_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=client.snapshot_value.fetched_at_utc,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=3000,
                        entry_price=1.1710,
                        take_profit=1.1750,
                        stop_loss=1.1710,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=client.snapshot_value.account,
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            self.assertNotIn("OPEN_POSITION_EXISTS", {issue["code"] for issue in payload["risk_issues"]})

    def test_non_live_ready_intent_is_not_staged_even_if_fresh_risk_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(
                intents_path=_intents(root, status="DRY_RUN_BLOCKED"),
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("INTENT_NOT_LIVE_READY", {issue["code"] for issue in payload["risk_issues"]})

    def test_invalid_pending_entry_writes_blocked_receipt_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root)
            payload = json.loads(intents.read_text())
            payload["results"][0]["intent"]["entry"] = None
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(
                intents_path=intents,
                lane_id="lane:EUR_USD:LONG",
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            result = json.loads((root / "request.json").read_text())
            self.assertIsNone(result["order_request"])
            self.assertIn("ORDER_REQUEST_INVALID", {issue["code"] for issue in result["risk_issues"]})


class FakeExecutionClient:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.snapshot_value = BrokerSnapshot(
            fetched_at_utc=now,
            positions=(),
            orders=(),
            quotes={
                "EUR_USD": Quote("EUR_USD", bid=1.17298, ask=1.17306, timestamp_utc=now),
                "USD_JPY": Quote("USD_JPY", bid=157.0, ask=157.01, timestamp_utc=now),
            },
            account=AccountSummary(
                nav_jpy=200_000.0,
                balance_jpy=200_000.0,
                margin_used_jpy=0.0,
                margin_available_jpy=200_000.0,
                fetched_at_utc=now,
            ),
        )
        self.orders: list[dict[str, Any]] = []

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        return self.snapshot_value

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        self.orders.append(order_request)
        return {"orderCreateTransaction": {"id": "1"}, "relatedTransactionIDs": ["1"]}


class MutatingExecutionClient(FakeExecutionClient):
    def __init__(self, *, margin_used_jpy: float) -> None:
        super().__init__()
        account = self.snapshot_value.account
        assert account is not None
        self.snapshot_value = BrokerSnapshot(
            fetched_at_utc=self.snapshot_value.fetched_at_utc,
            positions=self.snapshot_value.positions,
            orders=self.snapshot_value.orders,
            quotes=self.snapshot_value.quotes,
            account=AccountSummary(
                nav_jpy=account.nav_jpy,
                balance_jpy=account.balance_jpy,
                margin_used_jpy=margin_used_jpy,
                margin_available_jpy=max(0.0, account.nav_jpy - margin_used_jpy),
                fetched_at_utc=account.fetched_at_utc,
            ),
        )

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        response = super().post_order_json(order_request)
        units = int(order_request["units"])
        quote = self.snapshot_value.quotes["EUR_USD"]
        entry = quote.ask if units > 0 else quote.bid
        margin = abs(units) * entry * self.snapshot_value.quotes["USD_JPY"].bid * OANDA_JP_RETAIL_FX_MARGIN_RATE
        account = self.snapshot_value.account
        assert account is not None
        self.snapshot_value = BrokerSnapshot(
            fetched_at_utc=datetime.now(timezone.utc),
            positions=(
                *self.snapshot_value.positions,
                BrokerPosition(
                    trade_id=str(len(self.orders)),
                    pair="EUR_USD",
                    side=Side.LONG if units > 0 else Side.SHORT,
                    units=abs(units),
                    entry_price=entry,
                    take_profit=float(order_request["takeProfitOnFill"]["price"]),
                    stop_loss=float(order_request["stopLossOnFill"]["price"]),
                    owner=Owner.TRADER,
                ),
            ),
            orders=self.snapshot_value.orders,
            quotes=self.snapshot_value.quotes,
            account=AccountSummary(
                nav_jpy=account.nav_jpy,
                balance_jpy=account.balance_jpy,
                margin_used_jpy=account.margin_used_jpy + margin,
                margin_available_jpy=max(0.0, account.margin_available_jpy - margin),
                fetched_at_utc=datetime.now(timezone.utc),
            ),
        )
        return response


def _profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "required_fix": "eligible",
                    }
                ]
            }
        )
    )
    return path


def _intents(
    root: Path,
    *,
    status: str = "LIVE_READY",
    metadata: dict[str, str] | None = None,
    order_type: str = "STOP-ENTRY",
) -> Path:
    path = root / "intents.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": "lane:EUR_USD:LONG",
                        "status": status,
                        "risk_allowed": True,
                        "intent": {
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "order_type": order_type,
                            "units": 1000,
                            "entry": 1.17306 if order_type == "MARKET" else 1.17330,
                            "tp": 1.17450,
                            "sl": 1.17250,
                            "thesis": "trend continuation",
                            "owner": "trader",
                            "market_context": {
                                "regime": "TREND_CONTINUATION campaign lane",
                                "narrative": "trend continuation pressure",
                                "chart_story": "trend staircase",
                                "method": "TREND_CONTINUATION",
                                "invalidation": "SL trades",
                            },
                            "metadata": metadata
                            or {
                                "desk": "trend_trader",
                                "campaign_role": "NOW",
                            },
                        },
                    }
                ]
            }
        )
    )
    return path


if __name__ == "__main__":
    unittest.main()
