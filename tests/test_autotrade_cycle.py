from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.automation import AutoTradeCycle, _snapshot_to_json
from quant_rabbit.gpt_trader import StaticTraderProvider
from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side


class AutoTradeCycleTest(unittest.TestCase):
    def test_existing_pending_order_turns_cycle_monitor_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    orders=(
                        BrokerOrder(
                            order_id="1",
                            pair="AUD_JPY",
                            order_type="STOP",
                            price=112.576,
                            state="PENDING",
                            units=1000,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={"AUD_JPY": Quote("AUD_JPY", 112.49, 112.50, timestamp_utc=now)},
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                report_path=root / "report.md",
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "MONITOR_ONLY_EXPOSURE_OPEN")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertIn("monitor-only", (root / "report.md").read_text())
            self.assertTrue((root / "decision.json").exists())

    def test_protected_position_can_cancel_contaminated_pending_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    positions=(
                        BrokerPosition(
                            trade_id="t-protected",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=1000,
                            entry_price=1.1700,
                            unrealized_pl_jpy=40.0,
                            take_profit=1.1730,
                            stop_loss=1.1700,
                            owner=Owner.TRADER,
                        ),
                    ),
                    orders=(
                        BrokerOrder(
                            order_id="stale-pending",
                            pair="EUR_USD",
                            order_type="STOP",
                            price=1.1800,
                            state="PENDING",
                            units=1000,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.1710, 1.17108, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=True)

            self.assertEqual(summary.status, "CANCELED_CONTAMINATED_PENDING")
            self.assertEqual(summary.canceled_orders, ("stale-pending",))
            self.assertEqual(client.orders_canceled, ["stale-pending"])
            self.assertEqual(client.orders_sent, [])
            self.assertEqual(summary.position_management_action, "HOLD_PROTECTED")
            self.assertFalse((root / "live_order.json").exists())

    def test_flat_cycle_promotes_repair_receipt_before_trader_brain_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )
            profile = _repair_profile(root)

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=profile,
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            payload = json.loads(profile.read_text())
            statuses = {(item["pair"], item["direction"]): item["status"] for item in payload["profiles"]}
            self.assertEqual(statuses[("EUR_USD", "LONG")], "CANDIDATE")

    def test_flat_cycle_refreshes_quotes_after_market_story_before_pricing_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            stale = now - timedelta(seconds=90)
            client = SequenceCycleClient(
                (
                    BrokerSnapshot(
                        fetched_at_utc=stale,
                        quotes={
                            "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=stale),
                            "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=stale),
                        },
                    ),
                    BrokerSnapshot(
                        fetched_at_utc=now,
                        quotes={
                            "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                            "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                        },
                    ),
                )
            )
            news_root = root / "news"
            news_root.mkdir()

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                market_news_root=news_root,
                refresh_market_story=True,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertGreaterEqual(len(client.snapshot_calls), 3)
            intents = json.loads((root / "intents.json").read_text())
            self.assertEqual(intents["results"][0]["status"], "LIVE_READY")
            snapshot = json.loads((root / "snapshot.json").read_text())
            self.assertEqual(snapshot["quotes"]["EUR_USD"]["timestamp_utc"], now.isoformat())

    def test_protected_profitable_position_is_managed_before_new_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    positions=(
                        BrokerPosition(
                            trade_id="t-profit",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=1000,
                            entry_price=1.1700,
                            unrealized_pl_jpy=500.0,
                            take_profit=1.1730,
                            stop_loss=1.1690,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.1710, 1.17108, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "POSITION_ACTION_STAGED")
            self.assertEqual(summary.position_management_action, "PROFIT_PROTECT_REQUIRED")
            self.assertEqual(summary.position_execution_status, "STAGED")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())
            position_execution = json.loads((root / "pe.json").read_text())
            self.assertEqual(position_execution["actions"][0]["request"]["type"], "DEPENDENT_ORDER_REPLACE")
            self.assertIn('"stopLoss"', (root / "pe.md").read_text())

    def test_flat_cycle_does_not_enter_after_daily_target_reached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            target_state = root / "target.json"
            target_state.write_text(
                json.dumps(
                    {
                        "start_balance_jpy": 100_000,
                        "target_return_pct": 10.0,
                        "realized_pl_jpy": 10_250,
                        "daily_risk_budget_jpy": 500,
                    }
                )
            )
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                target_state_path=target_state,
                target_report_path=root / "target.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=True)

            self.assertEqual(summary.status, "TARGET_REACHED_PROTECT")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())
            self.assertFalse((root / "decision.json").exists())
            self.assertIn("protection-first no-send", (root / "report.md").read_text())

    def test_trade_attached_protection_orders_do_not_block_flat_entry_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    orders=(
                        BrokerOrder(
                            order_id="sl-1",
                            pair=None,
                            order_type="STOP_LOSS",
                            trade_id="closed-or-attached-trade",
                            price=1.17100,
                            state="PENDING",
                            owner=Owner.UNKNOWN,
                        ),
                    ),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.orders, 1)
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertFalse(summary.sent)
            self.assertFalse((root / "pm.json").exists())

    def test_operator_manual_position_does_not_stop_fresh_entry_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    positions=(
                        BrokerPosition(
                            trade_id="manual-470201",
                            pair="USD_JPY",
                            side=Side.LONG,
                            units=25000,
                            entry_price=155.962,
                            owner=Owner.UNKNOWN,
                        ),
                    ),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.positions, 1)
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertFalse((root / "pm.json").exists())
            self.assertTrue((root / "live_order.json").exists())

    def test_operator_manual_pending_order_does_not_stop_fresh_entry_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    orders=(
                        BrokerOrder(
                            order_id="manual-pending",
                            pair="USD_JPY",
                            order_type="STOP",
                            price=160.0,
                            state="PENDING",
                            units=25000,
                            owner=Owner.UNKNOWN,
                        ),
                    ),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.orders, 1)
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertTrue((root / "live_order.json").exists())

    def test_flat_cycle_uses_accepted_gpt_trade_before_gateway(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(_gpt_trade_decision()),
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.decision_source, "gpt_trader")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertIn("GPT trader", (root / "report.md").read_text())

    def test_gpt_batch_trade_sends_multiple_verified_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            market_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"
            _write_two_live_ready_intents(intents_path)
            target_state = _open_target_state(root)
            client = FakeCycleClient(snapshot)

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                gpt_target_state_path=target_state,
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(
                    _gpt_batch_trade_decision(
                        [
                            "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                            market_lane,
                        ]
                    )
                ),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertTrue(summary.sent)
            self.assertEqual(summary.sent_count, 2)
            self.assertEqual(
                summary.selected_lane_ids,
                ("trend_trader:EUR_USD:LONG:TREND_CONTINUATION", market_lane),
            )
            self.assertEqual(len(client.orders_sent), 2)
            result = json.loads((root / "live_order.json").read_text())
            self.assertEqual(len(result["orders"]), 2)

    def test_live_fresh_entry_requires_gpt_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            target_state = _open_target_state(root)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=True)

            self.assertEqual(summary.status, "GPT_REQUIRED_FOR_LIVE_SEND")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())
            self.assertIn("requires `--use-gpt-trader", (root / "report.md").read_text())

    def test_gpt_rejection_blocks_prefiltered_live_ready_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )
            decision = _gpt_trade_decision()
            decision["evidence_refs"] = ["broker:snapshot", "legacy:invented"]

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(decision),
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "GPT_REJECTED")
            self.assertEqual(summary.deterministic_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertIsNone(summary.selected_lane_id)
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())
            payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertIn("UNKNOWN_EVIDENCE_REF", {issue["code"] for issue in payload["verification_issues"]})

    def test_campaign_exposure_recovers_from_gpt_wait_when_flat_target_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            target_state = _open_target_state(root)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                gpt_target_state_path=target_state,
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(_gpt_wait_decision()),
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.decision_source, "campaign_exposure_recovery")
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertTrue(summary.campaign_exposure_required)
            self.assertIn("CAMPAIGN_EXPOSURE_RECOVERY", summary.gpt_recovery_source or "")
            self.assertTrue((root / "live_order.json").exists())
            self.assertIn("Campaign exposure required: `True`", (root / "report.md").read_text())

    def test_campaign_exposure_recovers_from_invalid_gpt_trade_when_flat_target_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            target_state = _open_target_state(root)
            decision = _gpt_trade_decision()
            decision["evidence_refs"] = ["broker:snapshot", "legacy:invented"]
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                gpt_target_state_path=target_state,
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(decision),
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.decision_source, "campaign_exposure_recovery")
            self.assertEqual(summary.gpt_status, "REJECTED")
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertTrue(summary.campaign_exposure_required)

    def test_gpt_can_select_prefiltered_discretionary_penalty_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "AUD_JPY": Quote("AUD_JPY", 113.100, 113.108, timestamp_utc=now),
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
            )
            rejected_lane = "failure_trader:AUD_JPY:LONG:BREAKOUT_FAILURE"

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_two_lane_campaign(root),
                strategy_profile_path=_two_lane_profile(root),
                market_story_profile_path=_two_lane_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(
                    _gpt_trade_decision(lane_id=rejected_lane, pair="AUD_JPY", method="BREAKOUT_FAILURE")
                ),
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.deterministic_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET")
            self.assertEqual(summary.selected_lane_id, rejected_lane)
            self.assertEqual(summary.decision_source, "gpt_trader")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(client.orders_sent, [])
            self.assertTrue((root / "live_order.json").exists())

    def test_reuse_market_artifacts_pins_gpt_packet_and_skips_repricing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.000, 157.004, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            _write_live_ready_intents(intents_path)
            target_state = root / "target.json"
            target_state.write_text(
                json.dumps(
                    {
                        "start_balance_jpy": 100_000,
                        "target_return_pct": 10.0,
                        "daily_risk_budget_jpy": 2_000,
                        "target_trades_per_day": 10,
                    }
                )
            )
            client = FakeCycleClient(snapshot)

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                gpt_target_state_path=target_state,
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(_gpt_trade_decision()),
                reuse_market_artifacts=True,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(summary.selected_lane_id, "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertEqual(len(client.snapshot_calls), 1)
            self.assertIn("reuse_existing", (root / "report.md").read_text())

    def test_reuse_market_artifacts_does_not_promote_then_reprice_stale_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            stale = now - timedelta(seconds=180)
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(
                _snapshot_to_json(
                    BrokerSnapshot(
                        fetched_at_utc=stale,
                        quotes={
                            "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=stale),
                            "USD_JPY": Quote("USD_JPY", 157.000, 157.004, timestamp_utc=stale),
                        },
                    )
                )
                + "\n"
            )
            intents_path = root / "intents.json"
            _write_live_ready_intents(intents_path)
            pinned_intents = intents_path.read_text()
            profile = _repair_then_candidate_profile(root)
            target_state = root / "target.json"
            target_state.write_text(
                json.dumps(
                    {
                        "start_balance_jpy": 100_000,
                        "target_return_pct": 10.0,
                        "daily_risk_budget_jpy": 2_000,
                        "target_trades_per_day": 10,
                    }
                )
            )
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.000, 157.004, timestamp_utc=now),
                    },
                )
            )

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                strategy_profile_path=profile,
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                gpt_target_state_path=target_state,
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(_gpt_trade_decision()),
                reuse_market_artifacts=True,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.receipt_promotions, 0)
            self.assertEqual(intents_path.read_text(), pinned_intents)
            profile_payload = json.loads(profile.read_text())
            self.assertEqual(profile_payload["profiles"][0]["status"], "RISK_REPAIR_CANDIDATE")
            self.assertFalse((root / "promotion.md").exists())


class FakeCycleClient:
    def __init__(self, snapshot: BrokerSnapshot) -> None:
        self.snapshot_value = _with_account(snapshot)
        self.snapshot_calls: list[tuple[str, ...]] = []
        self.orders_sent: list[dict[str, Any]] = []
        self.orders_canceled: list[str] = []

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_calls.append(pairs)
        return self.snapshot_value

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        self.orders_sent.append(order_request)
        return {"orderCreateTransaction": {"id": "1"}}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.orders_canceled.append(order_id)
        return {"orderCancelTransaction": {"id": "2", "orderID": order_id}}


class SequenceCycleClient(FakeCycleClient):
    def __init__(self, snapshots: tuple[BrokerSnapshot, ...]) -> None:
        snapshots_with_account = tuple(_with_account(snapshot) for snapshot in snapshots)
        super().__init__(snapshots_with_account[-1])
        self.snapshots = snapshots_with_account

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_calls.append(pairs)
        index = min(len(self.snapshot_calls) - 1, len(self.snapshots) - 1)
        return self.snapshots[index]


def _with_account(snapshot: BrokerSnapshot) -> BrokerSnapshot:
    if snapshot.account is not None:
        return snapshot
    now = snapshot.fetched_at_utc
    return replace(
        snapshot,
        account=AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            margin_used_jpy=0.0,
            margin_available_jpy=200_000.0,
            fetched_at_utc=now,
        ),
    )


def _open_target_state(root: Path) -> Path:
    path = root / "target.json"
    path.write_text(
        json.dumps(
            {
                "start_balance_jpy": 100_000,
                "target_return_pct": 10.0,
                "daily_risk_budget_jpy": 2_000,
                "target_trades_per_day": 10,
                "status": "PURSUE_TARGET",
                "remaining_target_jpy": 10_000,
            }
        )
    )
    return path


def _campaign(root: Path) -> Path:
    path = root / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "TREND_CONTINUATION",
                        "adoption": "RISK_REPAIR_DRY_RUN",
                        "campaign_role": "NOW_IF_REPAIRED",
                        "reason": "RISK_REPAIR_CANDIDATE; pretrade_net=5000, live_net=800, worst=-798",
                        "required_receipt": "prove current loss cap repair",
                        "blockers": ["old sizing broke the loss cap"],
                        "story_examples": [
                            "news_digest: EUR_USD trend-bull macro continuation",
                            "quality_audit: EUR_USD trend-bull staircase continuation",
                        ],
                    }
                ]
            }
        )
    )
    return path


def _two_lane_campaign(root: Path) -> Path:
    path = root / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "failure_trader",
                        "pair": "AUD_JPY",
                        "direction": "LONG",
                        "method": "BREAKOUT_FAILURE",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW_IF_CLEAN",
                        "reason": "positive legacy evidence, but JPY intervention narrative must veto weak longs",
                        "required_receipt": "live-ready failure receipt",
                        "blockers": [],
                        "story_examples": [
                            "news_digest: JPY intervention risk and rate check; WAIT on crosses",
                            "quality_audit: AUD_JPY trend-bull but intervention-sensitive",
                        ],
                    },
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "method": "TREND_CONTINUATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW_IF_CLEAN",
                        "reason": "EUR_USD trend-bull continuation pressure",
                        "required_receipt": "live-ready continuation receipt",
                        "blockers": [],
                        "story_examples": [
                            "news_digest: EUR_USD trend-bull macro continuation",
                            "quality_audit: EUR_USD trend-bull staircase continuation",
                        ],
                    },
                ]
            }
        )
    )
    return path


def _repair_profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "RISK_REPAIR_CANDIDATE",
                        "required_fix": "old sizing broke the loss cap",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 800,
                        "live_worst_jpy": -798,
                        "positive_evidence_n": 120,
                        "positive_tail_jpy": 1200,
                        "positive_best_jpy": 2200,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 5,
                    }
                ]
            }
        )
    )
    return path


def _candidate_profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "required_fix": "eligible after receipt promotion",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 800,
                        "live_worst_jpy": -350,
                        "positive_evidence_n": 120,
                        "positive_tail_jpy": 1200,
                        "positive_best_jpy": 2200,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 5,
                    }
                ]
            }
        )
    )
    return path


def _repair_then_candidate_profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "RISK_REPAIR_CANDIDATE",
                        "required_fix": "would be promoted if reuse mode mutated pinned artifacts",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 800,
                        "live_worst_jpy": -798,
                    },
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "required_fix": "eligible after prior repair",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 800,
                        "live_worst_jpy": -350,
                        "positive_evidence_n": 120,
                        "positive_tail_jpy": 1200,
                        "positive_best_jpy": 2200,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 5,
                    },
                ]
            }
        )
    )
    return path


def _two_lane_profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "AUD_JPY",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "required_fix": "eligible but narrative-sensitive",
                        "pretrade_net_jpy": 3000,
                        "live_net_jpy": 2000,
                        "live_worst_jpy": -300,
                        "positive_evidence_n": 80,
                        "positive_tail_jpy": 900,
                        "positive_best_jpy": 1500,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 4,
                    },
                    {
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "status": "CANDIDATE",
                        "required_fix": "eligible",
                        "pretrade_net_jpy": 5000,
                        "live_net_jpy": 2500,
                        "live_worst_jpy": -350,
                        "positive_evidence_n": 120,
                        "positive_tail_jpy": 1200,
                        "positive_best_jpy": 2200,
                        "seat_discovered": 10,
                        "seat_orderable": 8,
                        "seat_captured": 5,
                    },
                ]
            }
        )
    )
    return path


def _stories(root: Path) -> Path:
    path = root / "stories.json"
    path.write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 20},
                        "themes": {"momentum": 6},
                        "examples": [
                            "news_digest: EUR_USD trend-bull macro continuation",
                            "quality_audit: EUR_USD trend-bull staircase continuation",
                        ],
                    }
                ]
            }
        )
    )
    return path


def _two_lane_stories(root: Path) -> Path:
    path = root / "stories.json"
    path.write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "AUD_JPY",
                        "methods": {"BREAKOUT_FAILURE": 30},
                        "themes": {"breakout_failure": 4, "intervention": 3},
                        "examples": [
                            "news_digest: JPY intervention risk and rate check; WAIT on crosses",
                            "quality_audit: AUD_JPY trend-bull but intervention-sensitive",
                        ],
                    },
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 35},
                        "themes": {"momentum": 5},
                        "examples": [
                            "news_digest: EUR_USD trend-bull macro continuation",
                            "quality_audit: EUR_USD trend-bull staircase continuation",
                        ],
                    },
                ]
            }
        )
    )
    return path


def _write_live_ready_intents(path: Path) -> None:
    lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": lane_id,
                        "status": "LIVE_READY",
                        "intent": {
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "order_type": "STOP-ENTRY",
                            "units": 10_000,
                            "entry": 1.1735,
                            "tp": 1.175,
                            "sl": 1.1728,
                            "thesis": "Repaired EUR_USD continuation entry.",
                            "reason": "campaign lane is live-ready",
                            "owner": "trader",
                            "market_context": {
                                "regime": "TREND_CONTINUATION",
                                "narrative": "EUR_USD continuation pressure remains intact.",
                                "chart_story": "M5 trend staircase continuation above support.",
                                "method": "TREND_CONTINUATION",
                                "invalidation": "support shelf breaks before trigger",
                                "event_risk": "none",
                                "session": "test",
                            },
                            "metadata": {"max_loss_jpy": 1_500},
                        },
                        "risk_metrics": {
                            "risk_jpy": 1_099,
                            "reward_jpy": 2_355,
                            "reward_risk": 2.14,
                            "spread_pips": 0.8,
                        },
                        "risk_issues": [],
                        "strategy_issues": [],
                        "live_blockers": [],
                    }
                ]
            }
        )
        + "\n"
    )


def _write_two_live_ready_intents(path: Path) -> None:
    lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
    market_lane = f"{lane_id}:MARKET"
    _write_live_ready_intents(path)
    payload = json.loads(path.read_text())
    market = json.loads(json.dumps(payload["results"][0]))
    market["lane_id"] = market_lane
    market["intent"]["order_type"] = "MARKET"
    market["intent"]["entry"] = 1.17306
    market["intent"]["tp"] = 1.17436
    market["intent"]["sl"] = 1.17246
    market["risk_metrics"] = {
        "risk_jpy": 94.2,
        "reward_jpy": 204.1,
        "reward_risk": 2.17,
        "spread_pips": 0.8,
    }
    payload["results"].append(market)
    path.write_text(json.dumps(payload) + "\n")


def _gpt_trade_decision(
    *,
    lane_id: str = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
    pair: str = "EUR_USD",
    method: str = "TREND_CONTINUATION",
) -> dict:
    return {
        "action": "TRADE",
        "selected_lane_id": lane_id,
        "confidence": "HIGH",
        "thesis": "The live-ready EUR_USD continuation lane has current story and repaired risk geometry.",
        "method": method,
        "narrative": "Momentum and campaign role align with a controlled stop-entry.",
        "chart_story": "Higher lows press into the trigger shelf.",
        "invalidation": "Do not trade if the shelf fails before entry or the SL level trades.",
        "rejected_alternatives": ["WAIT rejected because a verified lane exists under the loss cap."],
        "risk_notes": ["Use only the verified lane units, TP, and SL."],
        "evidence_refs": [
            "broker:snapshot",
            "target:daily",
            f"intent:{lane_id}",
            f"campaign:{lane_id}",
            f"strategy:{pair}:LONG",
            f"story:{pair}",
        ],
        "operator_summary": "Accept the verified EUR_USD continuation lane.",
    }


def _gpt_batch_trade_decision(lane_ids: list[str]) -> dict:
    decision = _gpt_trade_decision(lane_id=lane_ids[0])
    decision["selected_lane_ids"] = lane_ids
    refs = list(decision["evidence_refs"])
    for lane_id in lane_ids[1:]:
        refs.extend([f"intent:{lane_id}", f"campaign:{lane_id}"])
    decision["evidence_refs"] = refs
    decision["operator_summary"] = "Accept the verified EUR_USD continuation basket."
    return decision


def _gpt_wait_decision() -> dict:
    return {
        "action": "WAIT",
        "selected_lane_id": None,
        "confidence": "MEDIUM",
        "thesis": "Wait despite a live-ready lane because discretionary timing is not clean enough.",
        "method": "EVENT_RISK",
        "narrative": "The lane is executable, but the operator asks for patience this cycle.",
        "chart_story": "The trigger shelf exists, but confirmation has not printed yet.",
        "invalidation": "Reconsider if the shelf holds and spread remains inside the receipt.",
        "rejected_alternatives": [
            "trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET rejected for timing only."
        ],
        "risk_notes": ["No trader exposure is open, so waiting would leave the campaign flat."],
        "evidence_refs": [
            "broker:snapshot",
            "target:daily",
            "intent:trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
            "campaign:trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET",
            "strategy:EUR_USD:LONG",
            "story:EUR_USD",
        ],
        "operator_summary": "Wait even though a live-ready lane exists.",
    }


if __name__ == "__main__":
    unittest.main()
