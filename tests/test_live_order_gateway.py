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


TEST_PER_TRADE_RISK_JPY = 10_000.0
TEST_DAILY_RISK_BUDGET_JPY = 50_000.0


class LiveOrderGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_per_trade_reader = execution_module._per_trade_risk_from_state
        self._original_daily_budget_reader = execution_module._daily_risk_budget_from_state
        self._original_target_trades_reader = execution_module._target_trades_per_day_from_state
        execution_module._per_trade_risk_from_state = lambda: TEST_PER_TRADE_RISK_JPY
        execution_module._daily_risk_budget_from_state = lambda path=None: TEST_DAILY_RISK_BUDGET_JPY
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

    def test_guardian_wake_hourly_schedule_alone_cannot_stage_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "guardian_action_receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "action": "TRADE",
                        "new_information": True,
                        "event_id": "event-hourly",
                        "pair": "EUR_USD",
                        "thesis": "trend continuation",
                        "thesis_state": "ALIVE",
                        "reason": "scheduled hour arrived, so place the trade",
                        "invalidation": "break back below support",
                        "harvest_trigger": "upper rail",
                        "gateway_required": True,
                    }
                )
            )

            summary = LiveOrderGateway(
                client=FakeExecutionClient(),
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                guardian_action_receipt_path=receipt,
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "guardian_event_id": "event-hourly",
                        "guardian_event_wake": True,
                    },
                    order_type="MARKET",
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("GUARDIAN_ACTION_SCHEDULE_ONLY", {issue["code"] for issue in payload["risk_issues"]})

    def test_guardian_wake_intent_requires_action_receipt_before_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = LiveOrderGateway(
                client=FakeExecutionClient(),
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                guardian_action_receipt_path=root / "missing_guardian_action_receipt.json",
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "guardian_event_id": "event-missing",
                        "guardian_event_wake": True,
                    },
                    order_type="MARKET",
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("GUARDIAN_ACTION_RECEIPT_REQUIRED", {issue["code"] for issue in payload["risk_issues"]})

    def test_sl_lint_blocks_jpy_major_figure_battle_zone_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = datetime.now(timezone.utc)
            account = client.snapshot_value.account
            assert account is not None
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(),
                orders=(),
                quotes={
                    "USD_JPY": Quote("USD_JPY", bid=161.884, ask=161.894, timestamp_utc=now),
                },
                account=account,
            )
            intents = _intents(
                root,
                order_type="MARKET",
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "tp_atr_pips": 6.0,
                    "level_cluster_radius_pips": 8.0,
                    "nearest_levels_above": [
                        {"price": 162.0, "source": "levels:round_number"}
                    ],
                    "event_risk": "JPY intervention risk near 162.00",
                },
            )
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:USD_JPY:SHORT"
            intent = result["intent"]
            intent.update(
                {
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "entry": 161.884,
                    "tp": 161.720,
                    "sl": 161.941,
                    "thesis": "JPY strength reversal fade below 162.00",
                    "market_context": {
                        "regime": "TREND_CONTINUATION campaign lane",
                        "narrative": "JPY intervention risk and reversal pressure near 162.00",
                        "chart_story": "fade major figure stop run",
                        "method": "TREND_CONTINUATION",
                        "invalidation": "clean acceptance above the major figure",
                    },
                }
            )
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT", pair="USD_JPY"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id="lane:USD_JPY:SHORT")

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            self.assertEqual(staged["sl_lint"]["status"], "BLOCK")
            codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("SL_LINT_MAJOR_FIGURE_BATTLE_ZONE", codes)
            self.assertIn("SL_LINT_EVENT_INTERVENTION_ZONE", codes)
            self.assertEqual(client.orders, [])

    def test_sl_lint_blocks_same_jpy_theme_without_theme_level_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            now = datetime.now(timezone.utc)
            account = client.snapshot_value.account
            assert account is not None
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="usd-jpy-short",
                        pair="USD_JPY",
                        side=Side.SHORT,
                        units=1000,
                        entry_price=161.800,
                        unrealized_pl_jpy=-120.0,
                        stop_loss=162.200,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(),
                quotes={
                    "EUR_JPY": Quote("EUR_JPY", bid=185.500, ask=185.510, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", bid=161.884, ask=161.894, timestamp_utc=now),
                },
                account=account,
            )
            intents = _intents(
                root,
                order_type="MARKET",
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "tp_atr_pips": 6.0,
                    "level_cluster_radius_pips": 8.0,
                },
            )
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:EUR_JPY:SHORT"
            intent = result["intent"]
            intent.update(
                {
                    "pair": "EUR_JPY",
                    "side": "SHORT",
                    "entry": 185.500,
                    "tp": 185.200,
                    "sl": 185.610,
                    "thesis": "same JPY strength reversal theme on EUR_JPY",
                    "market_context": {
                        "regime": "TREND_CONTINUATION campaign lane",
                        "narrative": "JPY strength reversal theme already active",
                        "chart_story": "JPY crosses fading together",
                        "method": "TREND_CONTINUATION",
                        "invalidation": "theme acceptance against JPY strength",
                    },
                }
            )
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT", pair="EUR_JPY"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id="lane:EUR_JPY:SHORT")

            self.assertEqual(summary.status, "BLOCKED")
            staged = json.loads((root / "request.json").read_text())
            self.assertEqual(staged["sl_lint"]["theme_group"], "JPY_STRENGTH_REVERSAL")
            codes = {issue["code"] for issue in staged["risk_issues"]}
            self.assertIn("SL_LINT_JPY_THEME_INVALIDATION_REQUIRED", codes)
            self.assertEqual(client.orders, [])

    def test_long_limit_crossed_favorably_reprices_passive_instead_of_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=_intents(root, order_type="LIMIT"), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["order_request"]["type"], "LIMIT")
            self.assertEqual(payload["order_request"]["price"], "1.17305")
            issue_codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("LIMIT_ENTRY_REPRICED_PASSIVE", issue_codes)
            self.assertNotIn("LIMIT_ENTRY_NOT_BELOW_MARKET", issue_codes)

    def test_short_limit_crossed_favorably_reprices_passive_instead_of_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            intents = _intents(root, order_type="LIMIT")
            payload = json.loads(intents.read_text())
            result = payload["results"][0]
            result["lane_id"] = "lane:EUR_USD:SHORT"
            intent = result["intent"]
            intent["side"] = "SHORT"
            intent["entry"] = 1.17290
            intent["tp"] = 1.17140
            intent["sl"] = 1.17380
            intent["thesis"] = "bear trend continuation"
            intent["market_context"] = {
                "regime": "TREND_CONTINUATION campaign lane",
                "narrative": "downtrend continuation pressure",
                "chart_story": "lower highs and continuation pressure",
                "method": "TREND_CONTINUATION",
                "invalidation": "SL trades",
            }
            intents.write_text(json.dumps(payload))

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root, direction="SHORT"),
                output_path=root / "request.json",
                report_path=root / "report.md",
            ).run(intents_path=intents, lane_id="lane:EUR_USD:SHORT")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["order_request"]["type"], "LIMIT")
            self.assertEqual(payload["order_request"]["units"], "-1000")
            self.assertEqual(payload["order_request"]["price"], "1.17299")
            issue_codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("LIMIT_ENTRY_REPRICED_PASSIVE", issue_codes)
            self.assertNotIn("LIMIT_ENTRY_NOT_ABOVE_MARKET", issue_codes)

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

    def test_report_surfaces_loss_asymmetry_sizing_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                max_loss_jpy=1000.0,
            ).run(
                intents_path=_intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "capture_economics_status": "NEGATIVE_EXPECTANCY",
                        "capture_avg_win_jpy": 600.0,
                        "capture_avg_loss_jpy": 1100.0,
                        "loss_asymmetry_guard_active": True,
                        "loss_asymmetry_guard_loss_cap_jpy": 600.0,
                        "loss_asymmetry_guard_base_max_loss_jpy": 1000.0,
                        "loss_asymmetry_guard_effective_max_loss_jpy": 600.0,
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            self.assertEqual(payload["sizing_evidence"]["loss_asymmetry_guard_loss_cap_jpy"], 600.0)
            report = (root / "report.md").read_text()
            self.assertIn("sizing guard: `LOSS_ASYMMETRY`", report)
            self.assertIn("units=`1000`", report)
            self.assertIn("cap=`600.0 JPY`", report)
            self.assertIn("avg_win/avg_loss=`600.0 JPY`/`1100.0 JPY`", report)

    def test_sl_free_disaster_stop_reports_attached_tail_risk_separately(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
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
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["order_request"]["stopLossOnFill"]["price"], "1.17000")
                self.assertEqual(payload["attached_stop_risk_metrics"]["basis"], "DISASTER_SL")
                self.assertGreater(
                    payload["attached_stop_risk_metrics"]["risk_jpy"],
                    payload["risk_metrics"]["risk_jpy"],
                )
                report = (root / "report.md").read_text()
                self.assertIn("intent risk", report)
                self.assertIn("attached broker SL: `DISASTER_SL`", report)
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_sl_free_firepower_route_attaches_intent_stop_for_measured_risk(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
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
                            "campaign_role": "OANDA_FIREPOWER_ROUTE",
                            "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["order_request"]["stopLossOnFill"]["price"], "1.17250")
                self.assertEqual(payload["attached_stop_risk_metrics"]["basis"], "INTENT_SL")
                self.assertAlmostEqual(
                    payload["attached_stop_risk_metrics"]["risk_jpy"],
                    payload["risk_metrics"]["risk_jpy"],
                )
                self.assertAlmostEqual(
                    payload["attached_stop_risk_metrics"]["loss_delta_pips"],
                    0.0,
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_sl_free_disaster_stop_does_not_block_on_per_trade_tail_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    max_loss_jpy=300.0,
                    live_enabled=True,
                ).run(
                    intents_path=_intents(
                        root,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["order_request"]["stopLossOnFill"]["price"], "1.17000")
                self.assertGreater(payload["attached_stop_risk_metrics"]["risk_jpy"], 300.0)
                self.assertNotIn(
                    "ATTACHED_STOP_LOSS_CAP_BELOW_MIN_LOT",
                    {issue["code"] for issue in payload["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_sl_free_disaster_stop_does_not_clip_units_to_attached_tail_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    max_loss_jpy=1100.0,
                ).run(
                    intents_path=_intents(
                        root,
                        units=3000,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                )

                self.assertEqual(summary.status, "STAGED")
                payload = json.loads((root / "request.json").read_text())
                scaled_units = int(payload["order_request"]["units"])
                self.assertEqual(scaled_units, 3000)
                self.assertEqual(payload["scaled_units"], scaled_units)
                self.assertGreater(payload["attached_stop_risk_metrics"]["risk_jpy"], 1100.0)
                self.assertEqual(payload["attached_stop_risk_metrics"]["basis"], "DISASTER_SL")
                self.assertNotIn(
                    "SIZE_MULTIPLE_CLIPPED_TO_ATTACHED_STOP_CAP",
                    {issue["code"] for issue in payload["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_attached_tail_risk_counts_against_portfolio_remaining_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    max_loss_jpy=10_000.0,
                    portfolio_loss_cap_jpy=300.0,
                ).run(
                    intents_path=_intents(
                        root,
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                self.assertLessEqual(payload["risk_metrics"]["risk_jpy"], 300.0)
                self.assertGreater(payload["attached_stop_risk_metrics"]["risk_jpy"], 300.0)
                self.assertIn(
                    "DISASTER_STOP_PORTFOLIO_CAP_EXCEEDED",
                    {issue["code"] for issue in payload["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_batch_counts_attached_tail_risk_in_cumulative_portfolio_cap(self) -> None:
        prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_initial_sl = os.environ.get("QR_NEW_ENTRY_INITIAL_SL")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                client = FakeExecutionClient()
                intents = _intents(
                    root,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "disaster_sl": 1.17000,
                    },
                )
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
                    max_loss_jpy=10_000.0,
                    portfolio_loss_cap_jpy=800.0,
                ).run_batch(
                    intents_path=intents,
                    lane_ids=("lane:EUR_USD:LONG", "lane:EUR_USD:LONG:reload"),
                )

                self.assertEqual(summary.status, "STAGED")
                self.assertEqual(summary.sent_count, 0)
                result = json.loads((root / "request.json").read_text())
                self.assertEqual(result["staged_count"], 1)
                self.assertEqual(result["blocked_count"], 1)
                first, second = result["orders"]
                self.assertEqual(first["status"], "STAGED")
                self.assertGreater(
                    first["attached_stop_risk_metrics"]["risk_jpy"],
                    first["risk_metrics"]["risk_jpy"],
                )
                self.assertEqual(second["status"], "BLOCKED")
                self.assertLessEqual(second["risk_metrics"]["risk_jpy"], 800.0)
                self.assertIn(
                    "DISASTER_STOP_PORTFOLIO_CAP_EXCEEDED",
                    {issue["code"] for issue in second["risk_issues"]},
                )
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_gateway_clips_units_to_intent_metadata_loss_cap(self) -> None:
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
                    units=3000,
                    metadata={
                        "desk": "trend_trader",
                        "campaign_role": "NOW",
                        "max_loss_jpy": 250.0,
                    },
                ),
                lane_id="lane:EUR_USD:LONG",
            )

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            scaled_units = int(payload["order_request"]["units"])
            self.assertGreaterEqual(scaled_units, 1000)
            self.assertLess(scaled_units, 3000)
            self.assertEqual(payload["scaled_units"], scaled_units)
            self.assertLessEqual(payload["risk_metrics"]["risk_jpy"], 250.0)
            self.assertIn(
                "SIZE_MULTIPLE_CLIPPED_TO_LOSS_CAP",
                {issue["code"] for issue in payload["risk_issues"]},
            )

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

    def test_live_send_retries_stale_quote_before_blocking(self) -> None:
        prior_attempts = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS")
        prior_sleep = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS")
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = "2"
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime.now(timezone.utc)
                stale = now - timedelta(seconds=45)
                client = SequenceExecutionClient(
                    (
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=stale),
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=now),
                    )
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

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                self.assertGreaterEqual(len(client.snapshot_calls), 2)
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["quote_refresh_attempts"], 1)
                self.assertNotIn("STALE_QUOTE", {issue["code"] for issue in payload["risk_issues"]})
        finally:
            if prior_attempts is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = prior_attempts
            if prior_sleep is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = prior_sleep

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

    def test_self_improvement_p0_allows_verified_repair_lane_staging(self) -> None:
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
            intents = _intents(
                root,
                metadata={
                    "desk": "range_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_pending_cancel_review_p0_requires_verified_trade_cancel_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                "message": "1 trader-owned pending entry order(s) need cancel review",
                                "evidence": {"cancel_review_order_ids": ["pending-1"]},
                            }
                        ],
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_pending_cancel_review_p0_allows_verified_trade_replacement_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (now - timedelta(minutes=1)).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
                                "message": "1 trader-owned pending entry order(s) need cancel review",
                                "evidence": {"cancel_review_order_ids": ["pending-1"]},
                            }
                        ],
                    }
                )
            )
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "TRADE",
                            "selected_lane_id": "lane:EUR_USD:LONG",
                            "selected_lane_ids": ["lane:EUR_USD:LONG"],
                            "cancel_order_ids": ["pending-1"],
                        },
                        "verification_issues": [],
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
                verified_decision_path=verified,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_profit_capture_miss_p0_allows_tp_harvest_repair_lane_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "status": "SELF_IMPROVEMENT_BLOCKED",
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "execution_quality",
                                "code": "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                                "message": (
                                    "13 losing close(s) had production-gate replay proof "
                                    "that TP-progress capture was executable before closing red"
                                ),
                            }
                        ],
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "range_trader",
                    "campaign_role": "NOW",
                    "positive_rotation_mode": "TP_PROOF_COLLECTION_HARVEST",
                    "positive_rotation_pessimistic_expectancy_jpy": 215.6,
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_p0_repair_blocker_code": (
                        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"
                    ),
                    "capture_take_profit_trades": 6,
                    "capture_take_profit_wins": 6,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 992.7,
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_self_improvement_p0_blocks_underpowered_oanda_repair_lane_staging(self) -> None:
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
            intents = _intents(
                root,
                metadata={
                    "desk": "range_trader",
                    "campaign_role": "OANDA_FIREPOWER_ROUTE",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                    "positive_rotation_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
                    "positive_rotation_minimum_floor_reachable": False,
                    "positive_rotation_minimum_floor_reach_basis": (
                        "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED"
                    ),
                    "positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable": False,
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_self_improvement_p0_rejects_repair_lane_on_named_worst_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "layer": "profitability",
                                "code": "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                                "message": "profitability discipline has failed for 62 consecutive audit runs",
                                "evidence": {
                                    "system_defect_evidence": {
                                        "worst_segments": [
                                            {
                                                "pair": "EUR_USD",
                                                "side": "LONG",
                                                "method": "TREND_CONTINUATION",
                                                "trades": 2,
                                                "net_jpy": -1937.49,
                                            }
                                        ]
                                    }
                                },
                            }
                        ]
                    }
                )
            )
            intents = _intents(
                root,
                metadata={
                    "desk": "trend_trader",
                    "campaign_role": "NOW",
                    "self_improvement_p0_repair_live_ready": True,
                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                },
            )
            client = FakeExecutionClient()
            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_forecast_adverse_path_blocks_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 16,
                                "supporting_codes": [
                                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                                ],
                                "metrics": {
                                    "directional_hit_rate": 0.261,
                                    "invalidation_first_rate": 0.739,
                                    "profit_factor": 0.891,
                                },
                            }
                        },
                        "findings": [
                            {
                                "priority": "P1",
                                "layer": "forecast",
                                "code": "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                "message": "directional_forecast HIT rate is weak",
                            }
                        ],
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
            self.assertIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", payload["risk_issues"][-1]["message"])

    def test_forecast_adverse_path_allows_tp_proven_repair_lane_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "root_cause_focus": {
                            "primary": {
                                "family": "FORECAST_ADVERSE_PATH",
                                "confidence": "HIGH",
                                "priority": "P1",
                                "process_loop_streak": 16,
                                "supporting_codes": [
                                    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                                    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                                ],
                            }
                        },
                    }
                )
            )
            intents = _intents(
                root,
                order_type="LIMIT",
                metadata={
                    "desk": "failure_trader",
                    "campaign_role": "NOW",
                    "forecast_direction": "RANGE",
                    "forecast_confidence": 0.62,
                    "attach_take_profit_on_fill": True,
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                    "opportunity_mode": "HARVEST",
                    "positive_rotation_live_ready": True,
                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                    "positive_rotation_pessimistic_expectancy_jpy": 180.0,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                    "capture_take_profit_scope_key": (
                        "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER"
                    ),
                    "capture_take_profit_trades": 20,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 591.5,
                    "self_improvement_forecast_adverse_path_repair_live_ready": True,
                    "self_improvement_forecast_adverse_path_repair_mode": "TP_HARVEST_REPAIR",
                    "self_improvement_forecast_adverse_path_repair_blocker_code": (
                        "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
                    ),
                },
            )
            payload = json.loads(intents.read_text())
            intent = payload["results"][0]["intent"]
            intent["thesis"] = "tp-proven failed-break fade"
            intent["market_context"]["method"] = "BREAKOUT_FAILURE"
            intent["market_context"]["regime"] = "RANGE current; BREAKOUT_FAILURE campaign lane"
            intents.write_text(json.dumps(payload))
            client = FakeExecutionClient()

            summary = LiveOrderGateway(
                client=client,
                strategy_profile=_profile(root),
                output_path=root / "request.json",
                report_path=root / "report.md",
                self_improvement_audit=audit,
            ).run(intents_path=intents, lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_stale_prior_gpt_decision_p0_does_not_block_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
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

            self.assertEqual(summary.status, "STAGED")
            self.assertFalse(summary.sent)
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_persistent_stale_prior_gpt_decision_p0_blocks_live_order_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 2},
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
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_persistent_stale_decision_p0_exempted_when_verified_receipt_postdates_audit(self) -> None:
        # Mirrors gpt_trader._self_improvement_trade_blockers: an ACCEPTED
        # verification produced AFTER the audit ran proves the stale-decision
        # finding is already repaired. Without this, the 20-minute audit
        # cadence re-blocks the first staging attempt of every fresh receipt
        # whenever the decision cadence is slower than two audit runs.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (now - timedelta(minutes=10)).isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 21},
                            }
                        ],
                    }
                )
            )
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [],
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
                verified_decision_path=verified,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "STAGED")
            payload = json.loads((root / "request.json").read_text())
            codes = {issue["code"] for issue in payload["risk_issues"]}
            self.assertNotIn("SELF_IMPROVEMENT_P0_BLOCKS_LIVE_ORDER", codes)

    def test_persistent_stale_decision_p0_still_blocks_when_verification_predates_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            audit = root / "self_improvement.json"
            audit.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "findings": [
                            {
                                "priority": "P0",
                                "code": "LATEST_GPT_DECISION_STALE",
                                "message": "latest GPT decision receipt predates the current broker snapshot",
                                "evidence": {"current_streak": 21},
                            }
                        ],
                    }
                )
            )
            verified = root / "gpt_decision.json"
            verified.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (now - timedelta(minutes=10)).isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "TRADE"},
                        "verification_issues": [],
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
                verified_decision_path=verified,
            ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG")

            self.assertEqual(summary.status, "BLOCKED")
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

    def test_target_path_live_send_defaults_to_disabled_even_when_gateway_live_is_enabled(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ.pop("QR_TARGET_PATH_LIVE_ENABLED", None)
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
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="A")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn("TARGET_PATH_LIVE_DISABLED", {issue["code"] for issue in result["risk_issues"]})
                self.assertFalse(result["target_path_receipt"]["live_order_sent"])
                self.assertFalse(result["target_path_receipt"]["target_path_live_enabled"])
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

    def test_target_path_live_send_blocks_b0_even_with_explicit_flag(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ["QR_TARGET_PATH_LIVE_ENABLED"] = "1"
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
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="B0", valid="NO")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn("TARGET_PATH_GRADE_TOO_LOW", {issue["code"] for issue in result["risk_issues"]})
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

    def test_target_path_live_send_allows_a_grade_with_receipt(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ["QR_TARGET_PATH_LIVE_ENABLED"] = "1"
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
                ).run(
                    intents_path=_intents(root, metadata=_target_path_metadata(grade="A")),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                receipt = result["target_path_receipt"]
                self.assertTrue(receipt["live_order_sent"])
                self.assertTrue(receipt["target_path_live_enabled"])
                self.assertEqual(receipt["target_path_live_mode"], "LIVE_LEARNING")
                self.assertEqual(receipt["daily_target_mode"], "ATTACK")
                self.assertEqual(receipt["five_pct_path_role"], "HERO")
                self.assertEqual(receipt["attack_stack_slot"], "NOW")
                self.assertEqual(receipt["grade"], "A")
                self.assertTrue(str(receipt["live_order_gateway_receipt_id"]).startswith("qrv1-EURUSD-L-"))
                report = (root / "report.md").read_text()
                self.assertIn("target-path receipt", report)
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

    def test_target_path_live_send_allows_b_plus_support_reload(self) -> None:
        prior = os.environ.get("QR_TARGET_PATH_LIVE_ENABLED")
        os.environ["QR_TARGET_PATH_LIVE_ENABLED"] = "1"
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
                ).run(
                    intents_path=_intents(
                        root,
                        metadata=_target_path_metadata(grade="B+", role="SUPPORT", slot="RELOAD"),
                    ),
                    lane_id="lane:EUR_USD:LONG",
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                self.assertEqual(result["target_path_receipt"]["five_pct_path_role"], "SUPPORT")
                self.assertEqual(result["target_path_receipt"]["attack_stack_slot"], "RELOAD")
                self.assertEqual(result["target_path_receipt"]["grade"], "B+")
        finally:
            _restore_env("QR_TARGET_PATH_LIVE_ENABLED", prior)

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

    def test_send_blocks_when_required_position_guardian_is_inactive(self) -> None:
        prior_required = os.environ.get("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE")
        prior_active = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")
        os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "1"
        os.environ["QR_POSITION_GUARDIAN_ACTIVE"] = "0"
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
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            _restore_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", prior_required)
            _restore_env("QR_POSITION_GUARDIAN_ACTIVE", prior_active)

    def test_send_position_guardian_requirement_has_explicit_operator_override(self) -> None:
        prior_required = os.environ.get("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE")
        prior_active = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")
        os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "0"
        os.environ["QR_POSITION_GUARDIAN_ACTIVE"] = "0"
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
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                self.assertNotIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            _restore_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", prior_required)
            _restore_env("QR_POSITION_GUARDIAN_ACTIVE", prior_active)

    def test_direct_send_fallback_blocks_when_loaded_guardian_lacks_fresh_heartbeat(self) -> None:
        env_keys = (
            "PATH",
            "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_LABEL",
            "QR_POSITION_GUARDIAN_PLIST",
            "QR_POSITION_GUARDIAN_EXECUTION",
            "QR_POSITION_GUARDIAN_HEARTBEAT",
            "QR_POSITION_GUARDIAN_INTERVAL",
            "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS",
            "QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT",
            "QR_FAKE_POSITION_GUARDIAN_LOADED",
        )
        prior = {key: os.environ.get(key) for key in env_keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plist = _write_guardian_plist(root)
                _install_fake_launchctl(root, loaded=True)
                os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "1"
                os.environ.pop("QR_POSITION_GUARDIAN_ACTIVE", None)
                os.environ["QR_POSITION_GUARDIAN_LABEL"] = "com.quantrabbit.position-guardian"
                os.environ["QR_POSITION_GUARDIAN_PLIST"] = str(plist)
                os.environ["QR_POSITION_GUARDIAN_EXECUTION"] = str(root / "missing_execution.json")
                os.environ["QR_POSITION_GUARDIAN_HEARTBEAT"] = str(root / "missing_heartbeat.json")
                os.environ["QR_POSITION_GUARDIAN_INTERVAL"] = "30"
                os.environ["QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT"] = "1"

                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "BLOCKED")
                self.assertFalse(summary.sent)
                self.assertEqual(client.orders, [])
                result = json.loads((root / "request.json").read_text())
                self.assertIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            for key, value in prior.items():
                _restore_env(key, value)

    def test_direct_send_fallback_allows_loaded_guardian_with_fresh_heartbeat(self) -> None:
        env_keys = (
            "PATH",
            "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_ACTIVE",
            "QR_POSITION_GUARDIAN_LABEL",
            "QR_POSITION_GUARDIAN_PLIST",
            "QR_POSITION_GUARDIAN_EXECUTION",
            "QR_POSITION_GUARDIAN_HEARTBEAT",
            "QR_POSITION_GUARDIAN_INTERVAL",
            "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS",
            "QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT",
            "QR_FAKE_POSITION_GUARDIAN_LOADED",
        )
        prior = {key: os.environ.get(key) for key in env_keys}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plist = _write_guardian_plist(root)
                heartbeat = _write_guardian_heartbeat(root)
                _install_fake_launchctl(root, loaded=True)
                os.environ["QR_REQUIRE_POSITION_GUARDIAN_ACTIVE"] = "1"
                os.environ.pop("QR_POSITION_GUARDIAN_ACTIVE", None)
                os.environ["QR_POSITION_GUARDIAN_LABEL"] = "com.quantrabbit.position-guardian"
                os.environ["QR_POSITION_GUARDIAN_PLIST"] = str(plist)
                os.environ["QR_POSITION_GUARDIAN_EXECUTION"] = str(heartbeat)
                os.environ["QR_POSITION_GUARDIAN_HEARTBEAT"] = str(root / "missing_heartbeat.json")
                os.environ["QR_POSITION_GUARDIAN_INTERVAL"] = "30"
                os.environ["QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT"] = "1"

                client = FakeExecutionClient()
                summary = LiveOrderGateway(
                    client=client,
                    strategy_profile=_profile(root),
                    output_path=root / "request.json",
                    report_path=root / "report.md",
                    live_enabled=True,
                ).run(intents_path=_intents(root), lane_id="lane:EUR_USD:LONG", send=True, confirm_live=True)

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                result = json.loads((root / "request.json").read_text())
                self.assertNotIn(
                    "POSITION_GUARDIAN_INACTIVE_FOR_SEND",
                    {issue["code"] for issue in result["risk_issues"]},
                )
        finally:
            for key, value in prior.items():
                _restore_env(key, value)

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

    def test_batch_send_retries_stale_quote_before_blocking(self) -> None:
        prior_attempts = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS")
        prior_sleep = os.environ.get("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS")
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = "2"
        os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = "0"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime.now(timezone.utc)
                stale = now - timedelta(seconds=45)
                client = SequenceExecutionClient(
                    (
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=now),
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=stale),
                        _gateway_snapshot(fetched_at=now, eur_usd_quote_time=now),
                    )
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
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "SENT")
                self.assertTrue(summary.sent)
                self.assertEqual(len(client.orders), 1)
                payload = json.loads((root / "request.json").read_text())
                self.assertEqual(payload["orders"][0]["quote_refresh_attempts"], 1)
                self.assertNotIn("STALE_QUOTE", {issue["code"] for issue in payload["risk_issues"]})
        finally:
            if prior_attempts is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_ATTEMPTS"] = prior_attempts
            if prior_sleep is None:
                os.environ.pop("QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS", None)
            else:
                os.environ["QR_GATEWAY_STALE_QUOTE_RETRY_SLEEP_SECONDS"] = prior_sleep

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

    def test_batch_blocks_existing_pending_geometry_with_disaster_stop(self) -> None:
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
                            order_id="471249",
                            pair="EUR_USD",
                            order_type="LIMIT",
                            price=1.17330,
                            units=1000,
                            owner=Owner.TRADER,
                            raw={
                                "id": "471249",
                                "instrument": "EUR_USD",
                                "type": "LIMIT_ORDER",
                                "price": "1.17330",
                                "units": "1000",
                                "takeProfitOnFill": {"price": "1.17450"},
                                "stopLossOnFill": {"price": "1.17000"},
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
                    intents_path=_intents(
                        root,
                        order_type="LIMIT",
                        metadata={
                            "desk": "trend_trader",
                            "campaign_role": "NOW",
                            "disaster_sl": 1.17000,
                        },
                    ),
                    lane_ids=("lane:EUR_USD:LONG",),
                    send=True,
                    confirm_live=True,
                )

                self.assertEqual(summary.status, "BLOCKED")
                self.assertEqual(summary.sent_count, 0)
                self.assertEqual(client.orders, [])
                payload = json.loads((root / "request.json").read_text())
                self.assertIn("BASKET_DUPLICATE_GEOMETRY", {issue["code"] for issue in payload["risk_issues"]})
        finally:
            if prior_sl_free is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free
            if prior_initial_sl is None:
                os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
            else:
                os.environ["QR_NEW_ENTRY_INITIAL_SL"] = prior_initial_sl

    def test_batch_blocks_existing_pending_parent_lane_with_drifted_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeExecutionClient()
            client.snapshot_value = BrokerSnapshot(
                fetched_at_utc=client.snapshot_value.fetched_at_utc,
                positions=(),
                orders=(
                    BrokerOrder(
                        order_id="472480",
                        pair="EUR_USD",
                        order_type="LIMIT",
                        price=1.17336,
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "id": "472480",
                            "instrument": "EUR_USD",
                            "type": "LIMIT_ORDER",
                            "price": "1.17336",
                            "units": "1000",
                            "clientExtensions": {
                                "comment": "qr-vnext lane=lane:EUR_USD:LONG desk=trend_trader role=NOW",
                                "id": "qrv1-EURUSD-L-existing",
                                "tag": "trader",
                            },
                            "tradeClientExtensions": {
                                "comment": "qr-vnext lane=lane:EUR_USD:LONG desk=trend_trader role=NOW",
                                "id": "qrv1-EURUSD-L-existing-trade",
                                "tag": "trader",
                            },
                            "takeProfitOnFill": {"price": "1.17456"},
                            "stopLossOnFill": {"price": "1.17244"},
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
                intents_path=_intents(root, order_type="LIMIT"),
                lane_ids=("lane:EUR_USD:LONG",),
                send=True,
                confirm_live=True,
            )

            self.assertEqual(summary.status, "BLOCKED")
            self.assertEqual(summary.sent_count, 0)
            self.assertEqual(client.orders, [])
            payload = json.loads((root / "request.json").read_text())
            self.assertIn("BASKET_DUPLICATE_PARENT_LANE", {issue["code"] for issue in payload["risk_issues"]})


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


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _write_guardian_plist(root: Path) -> Path:
    plist = root / "com.quantrabbit.position-guardian.plist"
    plist.write_text("<plist><dict><key>Label</key><string>com.quantrabbit.position-guardian</string></dict></plist>\n")
    return plist


def _write_guardian_heartbeat(root: Path, *, generated_at: datetime | None = None) -> Path:
    path = root / "position_guardian_execution.json"
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": (generated_at or datetime.now(timezone.utc)).isoformat(),
                "status": "NO_ACTION",
                "sent": False,
            }
        )
        + "\n"
    )
    return path


def _install_fake_launchctl(root: Path, *, loaded: bool) -> None:
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "launchctl"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cmd=\"${1:-}\"",
                "label=\"${2:-}\"",
                "if [[ \"$cmd\" == \"list\" && \"$label\" == \"com.quantrabbit.position-guardian\" ]]; then",
                "  if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-0}\" == \"1\" ]]; then",
                "    printf '123\\t0\\tcom.quantrabbit.position-guardian\\n'",
                "    exit 0",
                "  fi",
                "  exit 113",
                "fi",
                "if [[ \"$cmd\" == \"print\" && \"$label\" == gui/*/com.quantrabbit.position-guardian ]]; then",
                "  if [[ \"${QR_FAKE_POSITION_GUARDIAN_LOADED:-0}\" == \"1\" ]]; then",
                "    printf 'com.quantrabbit.position-guardian = { active = 1 }\\n'",
                "    exit 0",
                "  fi",
                "  exit 113",
                "fi",
                "printf 'unsupported fake launchctl command: %s %s\\n' \"$cmd\" \"$label\" >&2",
                "exit 64",
            ]
        )
        + "\n"
    )
    script.chmod(0o755)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["QR_FAKE_POSITION_GUARDIAN_LOADED"] = "1" if loaded else "0"


class SequenceExecutionClient(FakeExecutionClient):
    def __init__(self, snapshots: tuple[BrokerSnapshot, ...]) -> None:
        self.snapshots = snapshots
        self.snapshot_calls: list[tuple[str, ...]] = []
        self.orders: list[dict[str, Any]] = []

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_calls.append(tuple(pairs))
        index = min(len(self.snapshot_calls) - 1, len(self.snapshots) - 1)
        return self.snapshots[index]


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


def _gateway_snapshot(*, fetched_at: datetime, eur_usd_quote_time: datetime) -> BrokerSnapshot:
    return BrokerSnapshot(
        fetched_at_utc=fetched_at,
        positions=(),
        orders=(),
        quotes={
            "EUR_USD": Quote("EUR_USD", bid=1.17298, ask=1.17306, timestamp_utc=eur_usd_quote_time),
            "USD_JPY": Quote("USD_JPY", bid=157.0, ask=157.01, timestamp_utc=fetched_at),
        },
        account=AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            margin_used_jpy=0.0,
            margin_available_jpy=200_000.0,
            fetched_at_utc=fetched_at,
        ),
        home_conversions={"USD": 157.0},
    )


def _profile(root: Path, *, direction: str = "LONG", pair: str = "EUR_USD") -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": pair,
                        "direction": direction,
                        "status": "CANDIDATE",
                        "required_fix": "eligible",
                    }
                ]
            }
        )
    )
    return path


def _target_path_metadata(*, grade: str, role: str = "HERO", slot: str = "NOW", valid: str = "YES") -> dict[str, Any]:
    return {
        "desk": "trend_trader",
        "campaign_role": slot,
        "daily_target_mode": "ATTACK",
        "remaining_to_5pct_yen": 3000.0,
        "remaining_to_10pct_yen": 8000.0,
        "target_path_role": role,
        "path_board_slot": role,
        "path_board_available": True,
        "five_pct_path_available": True,
        "attack_stack_available": True,
        "attack_stack_slot": slot,
        "maps_to_attack_stack": True,
        "conviction_grade": grade,
        "valid_as_target_path": valid,
        "suggested_units": 1000,
        "risk_yen": 87.92,
        "risk_pct": 0.04,
        "target_yen": 226.08,
        "contribution_to_5pct": 226.08,
        "extension_gate": "NO",
        "exact_pretrade_passed": True,
        "spread_guard_passed": True,
        "pricing_probe_passed": True,
        "fill_guard_passed": True,
        "same_thesis_lost_recently": False,
        "vehicle_unchanged_after_loss": False,
        "target_path_live_mode": "LIVE_LEARNING",
    }


def _intents(
    root: Path,
    *,
    status: str = "LIVE_READY",
    metadata: dict[str, Any] | None = None,
    order_type: str = "STOP-ENTRY",
    units: int = 1000,
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
                            "units": units,
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


class DisasterStopOrderRequestTest(unittest.TestCase):
    """§3.5-K disaster stop attach in _oanda_order_request (2026-06-11)."""

    SL_KEYS = ("QR_NEW_ENTRY_INITIAL_SL", "QR_TRADER_DISABLE_SL_REPAIR")

    def setUp(self) -> None:
        self._prior = {k: os.environ.get(k) for k in self.SL_KEYS}

    def tearDown(self) -> None:
        for k, v in self._prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    @staticmethod
    def _intent(metadata: dict | None = None):
        from quant_rabbit.models import OrderIntent, OrderType, Side

        return OrderIntent(
            pair="EUR_USD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            units=5000,
            tp=1.1620,
            sl=1.1470,
            thesis="test",
            entry=1.1500,
            metadata=metadata or {},
        )

    def test_sl_free_mode_attaches_disaster_stop_from_metadata(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(
            self._intent({"disaster_sl": 1.1380, "disaster_sl_pips": 120.0})
        )
        self.assertEqual(order["stopLossOnFill"]["price"], "1.13800")

    def test_sl_free_mode_without_metadata_stays_sl_free(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(self._intent({"disaster_sl_missing": "H4_ATR_MISSING"}))
        self.assertNotIn("stopLossOnFill", order)

    def test_initial_sl_mode_keeps_intent_sl_over_disaster(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_NEW_ENTRY_INITIAL_SL"] = "1"
        order = execution_module._oanda_order_request(self._intent({"disaster_sl": 1.1380}))
        self.assertEqual(order["stopLossOnFill"]["price"], "1.14700")

    def test_firepower_route_keeps_intent_sl_over_disaster(self) -> None:
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(
            self._intent(
                {
                    "campaign_role": "OANDA_FIREPOWER_ROUTE",
                    "positive_rotation_oanda_campaign_firepower_vehicle_match": True,
                    "disaster_sl": 1.1380,
                }
            )
        )
        self.assertEqual(order["stopLossOnFill"]["price"], "1.14700")

    def test_normal_sl_mode_unchanged(self) -> None:
        os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
        os.environ.pop("QR_NEW_ENTRY_INITIAL_SL", None)
        order = execution_module._oanda_order_request(self._intent({"disaster_sl": 1.1380}))
        self.assertEqual(order["stopLossOnFill"]["price"], "1.14700")


if __name__ == "__main__":
    unittest.main()
