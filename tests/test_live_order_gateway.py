from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import quant_rabbit.broker.execution as execution_module
from quant_rabbit.broker.execution import LiveOrderGateway
from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.risk import OANDA_JP_RETAIL_FX_MARGIN_RATE


class LiveOrderGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_per_trade_reader = execution_module._per_trade_risk_from_state
        self._original_daily_budget_reader = execution_module._daily_risk_budget_from_state
        self._original_target_trades_reader = execution_module._target_trades_per_day_from_state
        execution_module._per_trade_risk_from_state = lambda: None
        execution_module._daily_risk_budget_from_state = lambda path=None: None
        execution_module._target_trades_per_day_from_state = lambda path=None: None

    def tearDown(self) -> None:
        execution_module._per_trade_risk_from_state = self._original_per_trade_reader
        execution_module._daily_risk_budget_from_state = self._original_daily_budget_reader
        execution_module._target_trades_per_day_from_state = self._original_target_trades_reader

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
            self.assertTrue(order["clientExtensions"]["id"].startswith("qrv1-EURUSD-L-"))
            self.assertEqual(order["clientExtensions"]["tag"], "trader")
            self.assertIn("lane=lane:EUR_USD:LONG", order["clientExtensions"]["comment"])
            self.assertEqual(order["tradeClientExtensions"]["tag"], "trader")
            self.assertIn("lane=lane:EUR_USD:LONG", order["tradeClientExtensions"]["comment"])

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

    def test_runner_intent_omits_broker_take_profit_on_fill(self) -> None:
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
                        "desk": "trend_trader",
                        "campaign_role": "RUNNER",
                        "attach_take_profit_on_fill": False,
                        "tp_execution_mode": "RUNNER_NO_BROKER_TP",
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            order = payload["order_request"]
            self.assertNotIn("takeProfitOnFill", order)

    def test_stage_receipt_persists_market_context_evidence(self) -> None:
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
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "market_context_matrix_ref": "matrix:EUR_USD:LONG",
                        "matrix_support_count": 4,
                        "matrix_reject_count": 0,
                        "matrix_support_layers": ["chart", "cross_asset", "context_asset_chart", "flow"],
                        "matrix_support_refs": [
                            "matrix:EUR_USD:LONG",
                            "cross:XAU_USD",
                            "context_asset:WTICO_USD",
                            "news:macro_event",
                        ],
                        "news_digest_ref": "news:macro_event",
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            evidence = payload["context_evidence"]
            self.assertEqual(evidence["market_context_matrix_ref"], "matrix:EUR_USD:LONG")
            self.assertEqual(evidence["matrix_support_layers"], ["chart", "cross_asset", "context_asset_chart", "flow"])
            self.assertIn("cross:XAU_USD", evidence["context_asset_refs"])
            self.assertIn("context_asset:WTICO_USD", evidence["context_asset_refs"])
            self.assertIn("news:macro_event", evidence["evidence_refs"])

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

    def test_self_improvement_p0_blocks_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 50 consecutive audit runs",
                            }
                        ]
                    }
                )
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

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

    def test_batch_blocks_duplicate_parent_lane_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root)
            payload = json.loads(intents.read_text())
            market = json.loads(json.dumps(payload["results"][0]))
            market["lane_id"] = "lane:EUR_USD:LONG:MARKET"
            market["intent"]["order_type"] = "MARKET"
            market["intent"]["entry"] = 1.17306
            payload["results"].append(market)
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:MARKET"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "PARTIAL_SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(len(client.orders), 1)
            result = json.loads((root / "request.json").read_text())
            second = result["orders"][1]
            self.assertEqual(second["status"], "BLOCKED")
            self.assertIn("BASKET_DUPLICATE_PARENT_LANE", {issue["code"] for issue in second["risk_issues"]})

    def test_batch_blocks_same_pair_opposite_side_without_explicit_hedge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root, order_type="MARKET")
            payload = json.loads(intents.read_text())
            short = json.loads(json.dumps(payload["results"][0]))
            short["lane_id"] = "lane:EUR_USD:SHORT"
            short["intent"]["side"] = "SHORT"
            short["intent"]["entry"] = 1.17298
            short["intent"]["tp"] = 1.17180
            short["intent"]["sl"] = 1.17360
            short["intent"]["thesis"] = "opposite side should not share basket"
            payload["results"].append(short)
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=intents,
                lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:SHORT"),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "PARTIAL_SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(len(client.orders), 1)
            result = json.loads((root / "request.json").read_text())
            second = result["orders"][1]
            self.assertEqual(second["status"], "BLOCKED")
            self.assertIn("BASKET_OPPOSING_PAIR_SIDE", {issue["code"] for issue in second["risk_issues"]})

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

    def test_capacity_downsize_floors_units_instead_of_rounding_up(self) -> None:
        scaled, issues = execution_module._scaled_units(4000, 0.28849)

        self.assertEqual(scaled, 1153)
        self.assertEqual(issues, [])

    def test_score_size_multiple_keeps_min_lot_floor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=_intents(root),
                lane_ids=("lane:EUR_USD:LONG",),
                size_multiples={"lane:EUR_USD:LONG": 0.95},
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            order_result = payload["orders"][0]
            self.assertEqual(order_result["requested_units"], 1000)
            self.assertEqual(order_result["scaled_units"], 1000)
            self.assertEqual(order_result["order_request"]["units"], "1000")
            issue_codes = {issue["code"] for issue in order_result["risk_issues"]}
            self.assertIn("SIZE_MULTIPLE_CLAMPED_TO_MIN_LOT", issue_codes)
            self.assertNotIn("MIN_LOT_VIOLATION", issue_codes)

    def test_capacity_downsize_below_min_lot_blocks_lane(self) -> None:
        scaled, issues = execution_module._scaled_units(
            1000,
            0.95,
            sub_min_lot_mode="block",
        )

        self.assertIsNone(scaled)
        self.assertEqual(["BASKET_CAPACITY_BELOW_MIN_LOT"], [issue.code for issue in issues])

    def test_portfolio_position_cap_scales_with_target_trade_pace(self) -> None:
        execution_module._target_trades_per_day_from_state = lambda path=None: 12
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "daily_target_state.json"
            state.write_text(json.dumps({"target_trades_per_day": 12}))

            cap = execution_module._portfolio_position_cap_from_state(
                state,
                policy=SimpleNamespace(max_portfolio_positions=2),
            )

            self.assertEqual(cap, 4)

    def test_capacity_downsize_leaves_integer_margin_headroom(self) -> None:
        now = datetime.now(timezone.utc)
        intent = execution_module._intent_from_json(
            {
                "pair": "EUR_USD",
                "side": "LONG",
                "order_type": "MARKET",
                "units": 4000,
                "entry": 1.17306,
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
            }
        )
        snapshot = BrokerSnapshot(
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
                margin_used_jpy=175_510.0,
                margin_available_jpy=200_000.0,
                fetched_at_utc=now,
            ),
        )
        metrics = SimpleNamespace(risk_jpy=400.0, estimated_margin_jpy=29_465.0)

        scale, issue = execution_module._basket_size_multiple(
            intent=intent,
            snapshot=snapshot,
            metrics=metrics,
            portfolio_loss_cap=None,
            cumulative_risk_jpy=0.0,
            cumulative_margin_jpy=0.0,
        )
        scaled, scale_issues = execution_module._scaled_units(intent.units, scale)

        self.assertIsNotNone(issue)
        self.assertEqual(issue.code, "BASKET_DOWNSIZED_FOR_CAPACITY")
        self.assertEqual(scale_issues, [])
        self.assertLessEqual((metrics.estimated_margin_jpy / intent.units) * scaled, 8_490.0)
        self.assertLess(scaled, 1153)

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

    def test_send_allows_same_pair_hedge_when_margin_cap_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = client.snapshot_value.fetched_at_utc
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=22_000,
                        entry_price=1.16688,
                        take_profit=1.17100,
                        stop_loss=1.16600,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=AccountSummary(
                    nav_jpy=175_988.7367,
                    balance_jpy=192_275.8359,
                    margin_used_jpy=162_740.16,
                    margin_available_jpy=13_436.9823,
                    hedging_enabled=True,
                    fetched_at_utc=now,
                ),
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "position_intent": "HEDGE",
                    "position_fill": "OPEN_ONLY",
                    "hedge_timing_class": "OPPOSITE_EXPOSURE",
                    "hedge_unwind_plan_required": True,
                    "hedge_review_trigger": "next_m15_close_or_structure_change",
                },
            )
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:EUR_USD:SHORT"
            intent = result["intent"]
            intent["side"] = "SHORT"
            intent["entry"] = 1.17270
            intent["tp"] = 1.17120
            intent["sl"] = 1.17350
            intent["thesis"] = "same-pair hedge against existing long"
            intent["market_context"] = {
                "regime": "BREAKOUT_FAILURE reject/retest current",
                "narrative": "short hedge against trapped long exposure",
                "chart_story": "failed reclaim and rejection below trigger",
                "method": "BREAKOUT_FAILURE",
                "invalidation": "SL trades",
            }
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT"),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
                max_loss_jpy=2_000.0,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:SHORT", send=True, confirm_live=True)

            self.assertEqual(summary.status, "SENT")
            self.assertTrue(summary.sent)
            self.assertEqual(len(client.orders), 1)
            self.assertEqual(client.orders[0]["units"], "-1000")
            self.assertEqual(client.orders[0]["positionFill"], "OPEN_ONLY")
            request = json.loads((root / "request.json").read_text())
            self.assertEqual(request["risk_metrics"]["estimated_margin_jpy"], 0.0)
            self.assertNotIn("MARGIN_UTILIZATION_CAP_EXCEEDED", {issue["code"] for issue in request["risk_issues"]})

    def test_hedge_intent_uses_open_only_position_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = client.snapshot_value.fetched_at_utc
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="101",
                        pair="EUR_USD",
                        side=Side.SHORT,
                        units=22_000,
                        entry_price=1.17000,
                        take_profit=1.16400,
                        stop_loss=1.17300,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes=client.snapshot_value.quotes,
                account=AccountSummary(
                    nav_jpy=200_000.0,
                    balance_jpy=200_000.0,
                    margin_used_jpy=50_000.0,
                    margin_available_jpy=150_000.0,
                    hedging_enabled=True,
                    fetched_at_utc=now,
                ),
            )
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
                        "hedge_timing_class": "OPPOSITE_EXPOSURE",
                        "hedge_unwind_plan_required": True,
                        "hedge_review_trigger": "next_m15_close_or_structure_change",
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

    def test_batch_position_cap_scales_with_target_trades_per_day(self) -> None:
        execution_module._target_trades_per_day_from_state = lambda path=None: 30
        prior_pair_cap = os.environ.get("QR_MAX_SAME_PAIR_TRADER_POSITIONS")
        os.environ["QR_MAX_SAME_PAIR_TRADER_POSITIONS"] = "10"
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                client = FakeExecutionClient()
                now = client.snapshot_value.fetched_at_utc
                client.snapshot_value = BrokerSnapshot(
                    fetched_at_utc=now,
                    positions=tuple(
                        BrokerPosition(
                            trade_id=str(200 + index),
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=1000,
                            entry_price=1.1710 + index * 0.0001,
                            take_profit=1.1750,
                            stop_loss=1.1710 + index * 0.0001,
                            owner=Owner.TRADER,
                        )
                        for index in range(4)
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
                ).run_batch(
                    intents_path=_intents(root, order_type="MARKET"),
                    lane_ids=("lane:EUR_USD:LONG",),
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["portfolio_position_cap"], 10)
                issue_codes = {issue["code"] for issue in payload["risk_issues"]}
                self.assertNotIn("BASKET_PORTFOLIO_POSITION_LIMIT", issue_codes)
                self.assertNotIn("PORTFOLIO_POSITION_LIMIT", issue_codes)
            finally:
                if prior_pair_cap is None:
                    os.environ.pop("QR_MAX_SAME_PAIR_TRADER_POSITIONS", None)
                else:
                    os.environ["QR_MAX_SAME_PAIR_TRADER_POSITIONS"] = prior_pair_cap

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

    def test_expired_projection_pending_blocks_stale_live_ready_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            emitted = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
            (root / "projection_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_emitted_utc": emitted,
                        "pair": "EUR_USD",
                        "signal_name": "directional_forecast",
                        "direction": "UP",
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                        "cycle_id": "expired-cycle",
                    }
                )
                + "\n"
            )

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

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn(
                "TELEMETRY_PROJECTION_PENDING_EXPIRED_FOR_LIVE",
                {issue["code"] for issue in payload["risk_issues"]},
            )

    def test_expired_projection_pending_blocks_batch_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            emitted = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
            (root / "projection_ledger.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp_emitted_utc": emitted,
                        "pair": "EUR_USD",
                        "signal_name": "directional_forecast",
                        "direction": "UP",
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                        "cycle_id": "expired-cycle",
                    }
                )
                + "\n"
            )

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                live_enabled=True,
            ).run_batch(
                intents_path=_intents(root, order_type="MARKET"),
                lane_ids=("lane:EUR_USD:LONG",),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["blocked_count"], 1)
            self.assertIn(
                "TELEMETRY_PROJECTION_PENDING_EXPIRED_FOR_LIVE",
                {issue["code"] for issue in payload["risk_issues"]},
            )

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

    def test_sl_free_pending_order_without_broker_sl_has_synthetic_basket_risk(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                client.snapshot_value = BrokerSnapshot(
                    fetched_at_utc=client.snapshot_value.fetched_at_utc,
                    positions=(),
                    orders=(
                        BrokerOrder(
                            order_id="471248",
                            pair="EUR_USD",
                            order_type="LIMIT",
                            price=1.16512,
                            units=1000,
                            owner=Owner.TRADER,
                            raw={
                                "id": "471248",
                                "instrument": "EUR_USD",
                                "type": "LIMIT_ORDER",
                                "price": "1.16512",
                                "units": "1000",
                                "takeProfitOnFill": {"price": "1.16826"},
                            },
                        ),
                    ),
                    quotes=client.snapshot_value.quotes,
                    account=client.snapshot_value.account,
                )

                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run_batch(
                    intents_path=_intents(root),
                    lane_ids=("lane:EUR_USD:LONG",),
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                issue_codes = {issue["code"] for issue in payload["risk_issues"]}
                self.assertNotIn("PENDING_RISK_UNKNOWN", issue_codes)
                self.assertNotIn("stopLossOnFill", payload["orders"][0]["order_request"])
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl


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


def _profile(root: Path, *, direction: str = "LONG") -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": direction,
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
    metadata: dict[str, Any] | None = None,
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
