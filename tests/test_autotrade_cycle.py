from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from quant_rabbit.automation import (
    AutoTradeCycle,
    AutoTradeCycleSummary,
    GptHandoffSummary,
    _gpt_lanes_pass_prefilter_or_recovery,
    _passes_gpt_prefilter,
    _snapshot_to_json,
)
from quant_rabbit.broker.position_execution import PositionExecutionSummary
from quant_rabbit.gpt_trader import StaticTraderProvider
from quant_rabbit.models import AccountSummary, BrokerOrder, BrokerPosition, BrokerSnapshot, Owner, Quote, Side
from quant_rabbit.strategy.trader_brain import ACTION_NO_TRADE, ACTION_SEND_ENTRY, LaneScore, TraderDecision


class AutoTradeCycleTest(unittest.TestCase):
    def test_target_state_refreshes_ai_backtest_before_pace_recalculation(self) -> None:
        prior = os.environ.get("QR_REFRESH_AI_BACKTEST_IN_TESTS")
        os.environ["QR_REFRESH_AI_BACKTEST_IN_TESTS"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                history_db = root / "legacy_history.db"
                sqlite3.connect(history_db).close()
                ai_backtest = root / "ai_test_bot_backtest.json"
                ai_report = root / "ai_test_bot_backtest.md"
                target_state = root / "target.json"
                target_state.write_text(
                    json.dumps(
                        {
                            "generated_at_utc": "2026-05-11T12:00:00+00:00",
                            "campaign_day_jst": "2026-05-11",
                            "start_balance_jpy": 100_000,
                            "target_return_pct": 10.0,
                            "daily_risk_budget_jpy": 2_000,
                            "target_trades_per_day": 10,
                            "target_trades_per_day_source": "ai_test_bot_required_trades",
                            "status": "PURSUE_TARGET",
                            "remaining_target_jpy": 10_000,
                        }
                    )
                    + "\n"
                )
                now = datetime(2026, 5, 11, 12, 5, tzinfo=timezone.utc)
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={},
                    account=AccountSummary(
                        nav_jpy=100_000,
                        balance_jpy=100_000,
                        unrealized_pl_jpy=0.0,
                        fetched_at_utc=now,
                    ),
                )
                calls: list[tuple[float, float]] = []

                def fake_backtest_run(self, *, start_balance_jpy: float, target_return_pct: float, max_validation_days=None):
                    calls.append((start_balance_jpy, target_return_pct))
                    Path(self.output_path).write_text(
                        json.dumps(
                            {
                                "target_return_pct": 10.0,
                                "firepower": {
                                    "required_trades_per_day_at_observed_expectancy": 80,
                                },
                                "target_band": {
                                    "floor_return_pct": 5.0,
                                    "stretch_return_pct": 10.0,
                                    "selected_attainable_return_pct": 5.0,
                                    "bands": [
                                        {
                                            "return_pct": 5.0,
                                            "required_trades_per_day_at_observed_expectancy": 20,
                                        },
                                        {
                                            "return_pct": 6.0,
                                            "required_trades_per_day_at_observed_expectancy": 23,
                                        },
                                    ],
                                },
                            }
                        )
                        + "\n"
                    )
                    return object()

                with (
                    mock.patch("quant_rabbit.automation.DEFAULT_HISTORY_DB", history_db),
                    mock.patch("quant_rabbit.automation.DEFAULT_AI_TEST_BOT_BACKTEST", ai_backtest),
                    mock.patch("quant_rabbit.automation.DEFAULT_AI_TEST_BOT_BACKTEST_REPORT", ai_report),
                    mock.patch("quant_rabbit.automation.AITestBotBacktester.run", fake_backtest_run),
                ):
                    summary = AutoTradeCycle(
                        client=object(),
                        target_state_path=target_state,
                        target_report_path=root / "target.md",
                        execution_ledger_db_path=root / "execution_ledger.db",
                    )._update_target_state(snapshot)

                payload = json.loads(target_state.read_text())
                self.assertIsNotNone(summary)
                self.assertEqual(calls, [(100_000.0, 10.0)])
                self.assertEqual(payload["target_trades_per_day"], 23)
                self.assertEqual(
                    payload["target_trades_per_day_source"],
                    # 1.0% min_per_trade_risk_pct floor (2026-06-11) lifts the
                    # backtest pace slice and annotates the source.
                    "ai_test_bot_target_band_6pct_required_trades_floored_by_min_per_trade_pct",
                )
                self.assertEqual(payload["target_trades_per_day_basis_return_pct"], 6.0)
        finally:
            if prior is None:
                os.environ.pop("QR_REFRESH_AI_BACKTEST_IN_TESTS", None)
            else:
                os.environ["QR_REFRESH_AI_BACKTEST_IN_TESTS"] = prior

    def test_projection_preflight_resolves_expired_pending_before_intents(self) -> None:
        prior = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE")
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir()
                now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
                emitted = now - timedelta(minutes=10)
                (data_root / "projection_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_emitted_utc": emitted.isoformat().replace("+00:00", "Z"),
                            "pair": "EUR_USD",
                            "signal_name": "directional_forecast",
                            "direction": "UP",
                            "lead_time_min": 1,
                            "confidence": 0.8,
                            "entry_price": 1.1000,
                            "predicted_target_price": 1.1010,
                            "predicted_invalidation_price": 1.0990,
                            "resolution_window_min": 1,
                            "resolution_status": "PENDING",
                            "cycle_id": "stale-cycle",
                        }
                    )
                    + "\n"
                )
                pair_charts = root / "pair_charts.json"
                pair_charts.write_text(
                    json.dumps(
                        {
                            "charts": [
                                {
                                    "pair": "EUR_USD",
                                    "views": [
                                        {"granularity": "H1", "indicators": {"atr_pips": 10.0}},
                                    ],
                                }
                            ]
                        }
                    )
                )
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.1020, 1.1022, timestamp_utc=now),
                    },
                )

                AutoTradeCycle(
                    client=object(),
                    intents_path=data_root / "order_intents.json",
                    pair_charts_path=pair_charts,
                )._verify_projection_preflight(snapshot)

                row = json.loads((data_root / "projection_ledger.jsonl").read_text())
        finally:
            if prior is None:
                os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = prior

        self.assertEqual(row["resolution_status"], "HIT")

    def test_projection_preflight_uses_m5_when_m1_window_missing(self) -> None:
        prior_required = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE")
        prior_m1 = os.environ.get("QR_PROJECTION_VERIFY_M1_COUNT")
        prior_m5 = os.environ.get("QR_PROJECTION_VERIFY_M5_COUNT")
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        os.environ["QR_PROJECTION_VERIFY_M1_COUNT"] = "2"
        os.environ["QR_PROJECTION_VERIFY_M5_COUNT"] = "2"

        class CandleClient:
            def __init__(self, emitted_at: datetime) -> None:
                self.emitted_at = emitted_at
                self.calls: list[tuple[str, str]] = []

            def get_json(self, _path: str, params: dict[str, str]) -> dict[str, Any]:
                granularity = params["granularity"]
                self.calls.append((granularity, params["count"]))
                if granularity == "M1":
                    return {"candles": []}
                return {
                    "candles": [
                        {
                            "time": (self.emitted_at + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
                            "mid": {"o": "1.1000", "h": "1.1012", "l": "1.1001", "c": "1.1008"},
                            "complete": True,
                        }
                    ]
                }

        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir()
                now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
                emitted = now - timedelta(minutes=30)
                (data_root / "projection_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_emitted_utc": emitted.isoformat().replace("+00:00", "Z"),
                            "pair": "EUR_USD",
                            "signal_name": "directional_forecast",
                            "direction": "UP",
                            "lead_time_min": 5,
                            "confidence": 0.8,
                            "entry_price": 1.1000,
                            "predicted_target_price": 1.1010,
                            "predicted_invalidation_price": 1.0990,
                            "resolution_window_min": 10,
                            "resolution_status": "PENDING",
                            "cycle_id": "m5-fallback-cycle",
                        }
                    )
                    + "\n"
                )
                pair_charts = root / "pair_charts.json"
                pair_charts.write_text(json.dumps({"charts": []}))
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.1004, 1.1006, timestamp_utc=now),
                    },
                )
                client = CandleClient(emitted)

                summary = AutoTradeCycle(
                    client=client,
                    intents_path=data_root / "order_intents.json",
                    pair_charts_path=pair_charts,
                )._verify_projection_preflight(snapshot)

                row = json.loads((data_root / "projection_ledger.jsonl").read_text())
        finally:
            if prior_required is None:
                os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = prior_required
            if prior_m1 is None:
                os.environ.pop("QR_PROJECTION_VERIFY_M1_COUNT", None)
            else:
                os.environ["QR_PROJECTION_VERIFY_M1_COUNT"] = prior_m1
            if prior_m5 is None:
                os.environ.pop("QR_PROJECTION_VERIFY_M5_COUNT", None)
            else:
                os.environ["QR_PROJECTION_VERIFY_M5_COUNT"] = prior_m5

        self.assertEqual(row["resolution_status"], "HIT")
        self.assertEqual(client.calls, [("M1", "2"), ("M5", "2")])
        self.assertEqual(summary["candle_granularity_counts"], {"EUR_USD": {"M1": 0, "M5": 1}})

    def test_projection_preflight_retries_truth_missing_timeout_with_m5(self) -> None:
        prior_required = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE")
        prior_m1 = os.environ.get("QR_PROJECTION_VERIFY_M1_COUNT")
        prior_m5 = os.environ.get("QR_PROJECTION_VERIFY_M5_COUNT")
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        os.environ["QR_PROJECTION_VERIFY_M1_COUNT"] = "0"
        os.environ["QR_PROJECTION_VERIFY_M5_COUNT"] = "2"

        class CandleClient:
            def __init__(self, emitted_at: datetime) -> None:
                self.emitted_at = emitted_at
                self.calls: list[tuple[str, str]] = []

            def get_json(self, _path: str, params: dict[str, str]) -> dict[str, Any]:
                granularity = params["granularity"]
                self.calls.append((granularity, params["count"]))
                return {
                    "candles": [
                        {
                            "time": (self.emitted_at + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
                            "mid": {"o": "1.1000", "h": "1.1012", "l": "1.1001", "c": "1.1008"},
                            "complete": True,
                        }
                    ]
                }

        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir()
                now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
                emitted = now - timedelta(minutes=30)
                (data_root / "projection_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_emitted_utc": emitted.isoformat().replace("+00:00", "Z"),
                            "pair": "EUR_USD",
                            "signal_name": "directional_forecast",
                            "direction": "UP",
                            "lead_time_min": 5,
                            "confidence": 0.8,
                            "entry_price": 1.1000,
                            "predicted_target_price": 1.1010,
                            "predicted_invalidation_price": 1.0990,
                            "resolution_window_min": 10,
                            "resolution_status": "TIMEOUT",
                            "resolution_evidence": "no M1 candle truth for projection window",
                            "cycle_id": "retry-timeout-cycle",
                        }
                    )
                    + "\n"
                )
                pair_charts = root / "pair_charts.json"
                pair_charts.write_text(json.dumps({"charts": []}))
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.1004, 1.1006, timestamp_utc=now),
                    },
                )
                client = CandleClient(emitted)

                summary = AutoTradeCycle(
                    client=client,
                    intents_path=data_root / "order_intents.json",
                    pair_charts_path=pair_charts,
                )._verify_projection_preflight(snapshot)

                row = json.loads((data_root / "projection_ledger.jsonl").read_text())
        finally:
            if prior_required is None:
                os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = prior_required
            if prior_m1 is None:
                os.environ.pop("QR_PROJECTION_VERIFY_M1_COUNT", None)
            else:
                os.environ["QR_PROJECTION_VERIFY_M1_COUNT"] = prior_m1
            if prior_m5 is None:
                os.environ.pop("QR_PROJECTION_VERIFY_M5_COUNT", None)
            else:
                os.environ["QR_PROJECTION_VERIFY_M5_COUNT"] = prior_m5

        self.assertEqual(row["resolution_status"], "HIT")
        self.assertEqual(client.calls, [("M5", "2")])
        self.assertEqual(summary["pending_pairs"], 0)
        self.assertEqual(summary["retryable_timeout_pairs"], 1)
        self.assertEqual(summary["candle_granularity_counts"], {"EUR_USD": {"M5": 1}})

    def test_reuse_market_artifacts_still_resolves_expired_projection_preflight(self) -> None:
        prior = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE")
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
                emitted = now - timedelta(minutes=10)
                (root / "projection_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_emitted_utc": emitted.isoformat().replace("+00:00", "Z"),
                            "pair": "EUR_USD",
                            "signal_name": "directional_forecast",
                            "direction": "UP",
                            "lead_time_min": 1,
                            "confidence": 0.8,
                            "entry_price": 1.1000,
                            "predicted_target_price": 1.1010,
                            "predicted_invalidation_price": 1.0990,
                            "resolution_window_min": 1,
                            "resolution_status": "PENDING",
                            "cycle_id": "reuse-stale-cycle",
                        }
                    )
                    + "\n"
                )
                pair_charts = root / "pair_charts.json"
                pair_charts.write_text(
                    json.dumps(
                        {
                            "charts": [
                                {
                                    "pair": "EUR_USD",
                                    "views": [
                                        {"granularity": "H1", "indicators": {"atr_pips": 10.0}},
                                    ],
                                }
                            ]
                        }
                    )
                )
                pending = BrokerOrder(
                    order_id="pending-1",
                    pair="EUR_USD",
                    order_type="STOP",
                    price=1.1735,
                    state="PENDING",
                    units=1000,
                    owner=Owner.TRADER,
                )
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    orders=(pending,),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.1020, 1.1022, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
                snapshot_path = root / "snapshot.json"
                snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
                intents_path = root / "intents.json"
                _write_no_live_ready_intents(intents_path)
                target_state = _open_target_state(root)

                summary = AutoTradeCycle(
                    client=FakeCycleClient(snapshot),
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    gpt_decision_path=root / "gpt_decision.json",
                    gpt_decision_report_path=root / "gpt_decision.md",
                    gpt_attack_advice_path=root / "attack_missing.json",
                    position_management_path=root / "pm.json",
                    position_management_report_path=root / "pm.md",
                    position_execution_path=root / "pe.json",
                    position_execution_report_path=root / "pe.md",
                    live_order_output_path=root / "live_order.json",
                    live_order_report_path=root / "live_order.md",
                    report_path=root / "report.md",
                    campaign_plan_path=_campaign(root),
                    pair_charts_path=pair_charts,
                    strategy_profile_path=_candidate_profile(root),
                    market_story_profile_path=_stories(root),
                    receipt_promotion_report_path=root / "promotion.md",
                    target_state_path=target_state,
                    target_report_path=root / "target.md",
                    gpt_target_state_path=target_state,
                    use_gpt_trader=True,
                    gpt_provider=StaticTraderProvider(_gpt_cancel_pending_decision(["pending-1"])),
                    reuse_market_artifacts=True,
                    refresh_market_story=False,
                    live_enabled=True,
                    max_loss_jpy=1_500,
                ).run(send=True)

                rows = [
                    json.loads(line)
                    for line in (root / "projection_ledger.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                row = next(item for item in rows if item.get("cycle_id") == "reuse-stale-cycle")
                report = (root / "report.md").read_text()
        finally:
            if prior is None:
                os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
            else:
                os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = prior

        self.assertEqual(summary.status, "CANCELED_GPT_PENDING")
        self.assertEqual(summary.gpt_action, "CANCEL_PENDING")
        self.assertEqual(row["resolution_status"], "HIT")
        self.assertIn("Projection preflight: status=`OK`", report)

    def test_gpt_cancel_pending_runs_when_no_live_ready_basket_exists(self) -> None:
        # Regression for 2026-06-12 AUD_CAD 472367: verifier accepted a
        # CANCEL_PENDING receipt, but the pending-entry branch skipped
        # _run_gpt_handoff whenever the current basket prefilter was empty.
        # The stale GTC order survived with live_ready=0.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 6, 12, 14, 32, tzinfo=timezone.utc)
            pending = BrokerOrder(
                order_id="pending-1",
                pair="AUD_CAD",
                order_type="LIMIT",
                price=0.98494,
                state="PENDING",
                units=-8000,
                owner=Owner.TRADER,
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={
                    "AUD_CAD": Quote("AUD_CAD", 0.98520, 0.98535, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            _write_no_live_ready_intents(intents_path)
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
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_provider=StaticTraderProvider(_gpt_cancel_pending_decision(["pending-1"])),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

        self.assertEqual(summary.status, "CANCELED_GPT_PENDING")
        self.assertEqual(summary.gpt_action, "CANCEL_PENDING")
        self.assertEqual(summary.canceled_orders, ("pending-1",))
        self.assertEqual(client.orders_canceled, ["pending-1"])
        self.assertEqual(client.orders_sent, [])

    def test_report_summarizes_harvest_and_runner_opportunity_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            data_root.mkdir()
            intents_path = data_root / "order_intents.json"
            intents_path.write_text(json.dumps({"results": []}))
            (data_root / "coverage_optimization.json").write_text(
                json.dumps(
                    {
                        "opportunity_modes": {
                            "HARVEST": {
                                "lanes": 7,
                                "live_ready_lanes": 2,
                                "reward_jpy": 1234.5,
                                "top_issue_codes": [
                                    {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 4}
                                ],
                                "top_live_blocker_codes": [
                                    {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 3}
                                ],
                            },
                            "RUNNER": {
                                "lanes": 0,
                                "live_ready_lanes": 0,
                                "diagnostic_candidate_lanes": 3,
                                "demoted_to_harvest_lanes": 3,
                                "runner_qualified_lanes": 0,
                                "reward_jpy": 0.0,
                            },
                        },
                        "runner_candidate_diagnostics": {
                            "top_issue_codes": [
                                {"code": "FORECAST_WATCH_ONLY", "count": 3}
                            ],
                            "top_live_blocker_codes": [
                                {"code": "RUNNER_REGIME_NOT_CLEAN", "count": 2}
                            ],
                            "top_demotion_reasons": [
                                {"reason": "RANGE regime is not a clean runner trend", "count": 2},
                                {"reason": "ADX below trend threshold", "count": 1},
                            ]
                        },
                        "perspective_alignment_diagnostics": {
                            "status": "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED",
                            "range_forecast_method_mismatch_lanes": 4,
                            "range_forecast_method_mismatch_top": [
                                {
                                    "pair": "AUD_JPY",
                                    "direction": "LONG",
                                    "method_mismatch_lanes": 3,
                                    "range_rotation_lanes": 0,
                                    "range_rotation_other_side_directions": [
                                        {"code": "SHORT", "count": 1}
                                    ],
                                    "range_rotation_other_side_top_live_blocker_codes": [
                                        {"code": "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE", "count": 1}
                                    ],
                                }
                            ],
                        },
                    }
                )
            )
            report_path = root / "report.md"
            summary = AutoTradeCycleSummary(
                status="NO_LIVE_READY_INTENT",
                report_path=report_path,
                snapshot_path=root / "snapshot.json",
                intents_path=intents_path,
                selected_lane_id=None,
                deterministic_lane_id=None,
                sent=False,
                positions=0,
                orders=0,
                live_ready=0,
            )

            AutoTradeCycle(
                client=object(),
                intents_path=intents_path,
                report_path=report_path,
                refresh_market_story=False,
            )._write_report(summary, "2026-06-16T00:00:00+00:00")

            report_text = report_path.read_text()
            self.assertIn(
                "- Opportunity modes: `HARVEST lanes=7 live_ready=2 reward_jpy=1234.5`; "
                "`RUNNER lanes=0 live_ready=0 diagnostic_candidates=3 demoted_to_harvest=3 "
                "runner_qualified=0 reward_jpy=0`",
                report_text,
            )
            self.assertIn(
                "- Opportunity issue codes: HARVEST=`FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE:4`; "
                "RUNNER=`FORECAST_WATCH_ONLY:3`",
                report_text,
            )
            self.assertIn(
                "- Opportunity live blocker codes: HARVEST=`FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE:3`; "
                "RUNNER=`RUNNER_REGIME_NOT_CLEAN:2`",
                report_text,
            )
            self.assertIn(
                "- Runner demotions: `RANGE regime is not a clean runner trend:2, "
                "ADX below trend threshold:1`",
                report_text,
            )
            self.assertIn(
                "- Perspective alignment: status=RANGE_METHOD_MISMATCH_REPAIR_REQUIRED "
                "range_mismatch_lanes=4; top=`AUD_JPY LONG mismatch=3 range_lanes=0 "
                "other_rail=SHORT:1 other_blockers=FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE:1`",
                report_text,
            )

    def test_rejected_gpt_close_ids_do_not_call_broker(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.closed: list[tuple[str, str]] = []

            def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]:
                self.closed.append((trade_id, units))
                return {"ok": True}

        client = Client()
        cycle = AutoTradeCycle(client=client, live_enabled=True)
        summary = GptHandoffSummary(
            status="REJECTED",
            action="CLOSE",
            selected_lane_id=None,
            allowed=False,
            issues=1,
            close_trade_ids=("471232",),
        )
        snapshot = BrokerSnapshot(fetched_at_utc=datetime.now(timezone.utc))

        execution = cycle._close_gpt_trades(summary, snapshot=snapshot, send=True)

        self.assertEqual(execution.status, "NO_ACTION")
        self.assertFalse(execution.sent)
        self.assertEqual(client.closed, [])

    def test_accepted_gpt_close_routes_through_position_gateway_receipt(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.closed: list[tuple[str, str]] = []
                self.snapshot_payload: BrokerSnapshot | None = None
                self.snapshot_calls: list[tuple[str, ...]] = []

            def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
                self.snapshot_calls.append(pairs)
                if self.snapshot_payload is None:
                    raise AssertionError("missing snapshot payload")
                return self.snapshot_payload

            def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]:
                self.closed.append((trade_id, units))
                return {"relatedTransactionIDs": ["20"]}

            def replace_trade_dependent_orders(self, trade_id: str, order_request: dict[str, Any]) -> dict[str, Any]:
                raise AssertionError("GPT CLOSE must not replace dependent orders")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="471232",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1729,
                        unrealized_pl_jpy=-250.0,
                        take_profit=1.1740,
                        stop_loss=None,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"EUR_USD": Quote("EUR_USD", 1.1710, 1.1711, timestamp_utc=now)},
            )
            client = Client()
            client.snapshot_payload = snapshot
            cycle = AutoTradeCycle(
                client=client,
                live_enabled=True,
                snapshot_path=root / "broker.json",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
            )
            summary = GptHandoffSummary(
                status="ACCEPTED",
                action="CLOSE",
                selected_lane_id=None,
                allowed=True,
                issues=0,
                close_trade_ids=("471232",),
            )

            execution = cycle._close_gpt_trades(summary, snapshot=snapshot, send=True)

            self.assertEqual(execution.status, "SENT")
            self.assertTrue(execution.sent)
            self.assertEqual(client.closed, [("471232", "ALL")])
            self.assertEqual(len(client.snapshot_calls), 1)
            self.assertIn("EUR_USD", client.snapshot_calls[0])
            payload = json.loads((root / "pe.json").read_text())
            self.assertEqual(payload["status"], "SENT")
            self.assertEqual(payload["actions"][0]["request"]["type"], "CLOSE")
            self.assertEqual(payload["actions"][0]["trade_id"], "471232")
            self.assertIn("CLOSE", (root / "pe.md").read_text())

    def test_accepted_gpt_close_ledger_receipt_survives_position_execution_overwrite(self) -> None:
        class Client:
            def __init__(self, snapshot_payload: BrokerSnapshot) -> None:
                self.snapshot_payload = snapshot_payload
                self.closed: list[tuple[str, str]] = []

            def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
                return self.snapshot_payload

            def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]:
                self.closed.append((trade_id, units))
                return {"relatedTransactionIDs": ["20"], "closedTradeID": trade_id}

            def replace_trade_dependent_orders(self, trade_id: str, order_request: dict[str, Any]) -> dict[str, Any]:
                raise AssertionError("GPT CLOSE must not replace dependent orders")

            def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
                return AccountSummary(
                    nav_jpy=200_000.0,
                    balance_jpy=200_000.0,
                    margin_available_jpy=200_000.0,
                    last_transaction_id="100",
                    fetched_at_utc=now_utc or datetime.now(timezone.utc),
                )

            def transactions_since_id(self, transaction_id: str) -> dict[str, Any]:
                return {"lastTransactionID": transaction_id, "transactions": []}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="471232",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1729,
                        unrealized_pl_jpy=-250.0,
                        take_profit=1.1740,
                        stop_loss=None,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"EUR_USD": Quote("EUR_USD", 1.1710, 1.1711, timestamp_utc=now)},
            )
            ledger_path = root / "execution_ledger.db"
            gpt_decision_path = root / "gpt_decision.json"
            gpt_decision_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": "2026-06-08T00:00:00+00:00",
                        "status": "ACCEPTED",
                        "decision": {
                            "action": "CLOSE",
                            "selected_lane_id": None,
                            "close_trade_ids": ["471232"],
                        },
                        "verification_issues": [],
                    }
                )
                + "\n"
            )
            cycle = AutoTradeCycle(
                client=Client(snapshot),
                live_enabled=True,
                snapshot_path=root / "broker.json",
                gpt_decision_path=gpt_decision_path,
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                execution_ledger_db_path=ledger_path,
                execution_ledger_report_path=root / "execution_ledger.md",
            )
            summary = GptHandoffSummary(
                status="ACCEPTED",
                action="CLOSE",
                selected_lane_id=None,
                allowed=True,
                issues=0,
                close_trade_ids=("471232",),
            )

            execution = cycle._close_gpt_trades(summary, snapshot=snapshot, send=True)
            self.assertEqual(execution.status, "SENT")

            overwritten = {
                "generated_at_utc": (now + timedelta(minutes=1)).isoformat(),
                "status": "NO_ACTION",
                "send_requested": True,
                "sent": False,
                "actions": [
                    {
                        "trade_id": "471232",
                        "pair": "EUR_USD",
                        "owner": "trader",
                        "management_action": "HOLD_SL_FREE",
                        "request": None,
                        "issues": [],
                        "sent": False,
                        "response": None,
                    }
                ],
            }
            (root / "pe.json").write_text(json.dumps(overwritten, indent=2) + "\n")
            cycle._record_execution_ledger_receipts()

            with sqlite3.connect(ledger_path) as conn:
                events = conn.execute(
                    "SELECT event_type, trade_id, exit_reason FROM execution_events ORDER BY rowid"
                ).fetchall()

        self.assertIn(("GATEWAY_TRADE_CLOSE_SENT", "471232", "GPT_CLOSE"), events)
        self.assertIn(("GATEWAY_GPT_CLOSE_ACCEPTED", "471232", "GPT_CLOSE_ACCEPTED"), events)
        self.assertIn(("GATEWAY_POSITION_NO_ACTION", "471232", "HOLD_SL_FREE"), events)

    def test_accepted_gpt_close_uses_refreshed_snapshot_before_send(self) -> None:
        class Client:
            def __init__(self, snapshot_payload: BrokerSnapshot) -> None:
                self.snapshot_payload = snapshot_payload
                self.snapshot_calls: list[tuple[str, ...]] = []
                self.closed: list[tuple[str, str]] = []

            def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
                self.snapshot_calls.append(pairs)
                return self.snapshot_payload

            def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]:
                self.closed.append((trade_id, units))
                return {"relatedTransactionIDs": ["20"]}

            def replace_trade_dependent_orders(self, trade_id: str, order_request: dict[str, Any]) -> dict[str, Any]:
                raise AssertionError("GPT CLOSE must not replace dependent orders")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            stale_snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="471232",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.1729,
                        unrealized_pl_jpy=-250.0,
                        take_profit=1.1740,
                        stop_loss=None,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={"EUR_USD": Quote("EUR_USD", 1.1710, 1.1711, timestamp_utc=now)},
            )
            refreshed_snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(),
                quotes={"EUR_USD": Quote("EUR_USD", 1.1712, 1.1713, timestamp_utc=now)},
            )
            client = Client(refreshed_snapshot)
            cycle = AutoTradeCycle(
                client=client,
                live_enabled=True,
                snapshot_path=root / "broker.json",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
            )
            summary = GptHandoffSummary(
                status="ACCEPTED",
                action="CLOSE",
                selected_lane_id=None,
                allowed=True,
                issues=0,
                close_trade_ids=("471232",),
            )

            execution = cycle._close_gpt_trades(summary, snapshot=stale_snapshot, send=True)

            self.assertEqual(execution.status, "STALE_CLOSE_SATISFIED")
            self.assertFalse(execution.sent)
            self.assertEqual(client.closed, [])
            self.assertEqual(len(client.snapshot_calls), 1)
            payload = json.loads((root / "pe.json").read_text())
            self.assertEqual(payload["status"], "STALE_CLOSE_SATISFIED")
            self.assertEqual(payload["actions"][0]["issues"][0]["code"], "STALE_CLOSE_ALREADY_ABSENT")
            self.assertIn("STALE_CLOSE_SATISFIED", (root / "pe.md").read_text())

    def test_stale_gpt_close_satisfied_defers_reentry_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class Cycle(AutoTradeCycle):
                def __init__(self) -> None:
                    super().__init__(
                        client=object(),
                        report_path=root / "cycle.md",
                        snapshot_path=root / "broker.json",
                        intents_path=root / "intents.json",
                    )
                    self.archived = False
                    self.resumed_depth: int | None = None

                def _archive_gpt_close_receipt_for_reentry(self) -> None:
                    self.archived = True

                def _run(self, *, send: bool = False, _close_reentry_depth: int = 0) -> AutoTradeCycleSummary:
                    self.resumed_depth = _close_reentry_depth
                    return AutoTradeCycleSummary(
                        status="STAGED",
                        report_path=root / "cycle.md",
                        snapshot_path=root / "broker.json",
                        intents_path=root / "intents.json",
                        selected_lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        deterministic_lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                        sent=False,
                        positions=0,
                        orders=0,
                        live_ready=1,
                        decision_source="gpt_trader",
                        position_management_action=None,
                    )

            cycle = Cycle()
            close_execution = PositionExecutionSummary(
                status="STALE_CLOSE_SATISFIED",
                output_path=root / "pe.json",
                report_path=root / "pe.md",
                sent=False,
                actions=0,
                blocked=0,
            )
            close_summary = GptHandoffSummary(
                status="ACCEPTED",
                action="CLOSE",
                selected_lane_id=None,
                allowed=True,
                issues=0,
                close_trade_ids=("471232",),
            )

            summary = cycle._continue_after_gpt_close(
                generated_at=datetime.now(timezone.utc).isoformat(),
                send=True,
                close_execution=close_execution,
                close_gpt_summary=close_summary,
                positions=1,
                orders=0,
                live_ready=0,
                deterministic_lane_id=None,
                target_summary=None,
            )

            self.assertFalse(cycle.archived)
            self.assertIsNone(cycle.resumed_depth)
            self.assertEqual(summary.status, "POSITION_ACTION_SATISFIED")
            self.assertEqual(summary.decision_source, "gpt_trader")
            self.assertEqual(summary.position_execution_status, "STALE_CLOSE_SATISFIED")
            self.assertEqual(summary.gpt_recovery_source, "POST_CLOSE_REENTRY_DEFERRED")

    def test_forecast_blocker_is_not_gpt_prefilter_eligible(self) -> None:
        score = LaneScore(
            lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
            pair="EUR_USD",
            direction="LONG",
            method="TREND_CONTINUATION",
            order_type="STOP-ENTRY",
            entry=1.16522,
            tp=1.16878,
            sl=1.16401,
            status="LIVE_READY",
            score=42.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast DOWN opposes LONG; rationale: pair chart shifted lower",),
            rationale=(),
            size_multiple=1.0,
            estimated_margin_jpy=7_400.0,
        )

        self.assertFalse(_passes_gpt_prefilter(score))

    def test_low_confidence_range_limit_forecast_blocker_remains_gpt_prefilter_eligible(self) -> None:
        score = LaneScore(
            lane_id="range_trader:EUR_USD:SHORT:RANGE_ROTATION",
            pair="EUR_USD",
            direction="SHORT",
            method="RANGE_ROTATION",
            order_type="LIMIT",
            entry=1.16522,
            tp=1.16378,
            sl=1.16601,
            status="LIVE_READY",
            score=42.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast confidence 0.23 < 0.55 threshold",),
            rationale=(),
            size_multiple=1.0,
            estimated_margin_jpy=0.0,
        )

        self.assertTrue(_passes_gpt_prefilter(score))

    def test_low_confidence_trend_forecast_blocker_is_not_gpt_prefilter_eligible(self) -> None:
        score = LaneScore(
            lane_id="trend_trader:EUR_USD:SHORT:TREND_CONTINUATION",
            pair="EUR_USD",
            direction="SHORT",
            method="TREND_CONTINUATION",
            order_type="STOP-ENTRY",
            entry=1.16522,
            tp=1.16378,
            sl=1.16601,
            status="LIVE_READY",
            score=42.0,
            action=ACTION_NO_TRADE,
            blockers=("forecast confidence 0.23 < 0.55 threshold",),
            rationale=(),
            size_multiple=1.0,
            estimated_margin_jpy=0.0,
        )

        self.assertFalse(_passes_gpt_prefilter(score))

    def test_live_ready_advisory_history_blockers_remain_gpt_prefilter_eligible(self) -> None:
        score = LaneScore(
            lane_id="trend_trader:NZD_CHF:LONG:TREND_CONTINUATION",
            pair="NZD_CHF",
            direction="LONG",
            method="TREND_CONTINUATION",
            order_type="STOP-ENTRY",
            entry=0.46789,
            tp=0.46935,
            sl=0.46699,
            status="LIVE_READY",
            score=20.45,
            action=ACTION_NO_TRADE,
            blockers=(
                "missing strategy profile",
                "campaign lane is not executable: missing",
                "no positive mined or repaired edge evidence",
                "market story does not support the selected method",
            ),
            rationale=(),
            size_multiple=0.9,
            estimated_margin_jpy=3_820.0,
        )

        self.assertTrue(_passes_gpt_prefilter(score))

    def test_recovery_hedge_gpt_selection_can_bypass_empty_prefilter(self) -> None:
        lane_id = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
        payload = {
            "results": [
                {
                    "lane_id": lane_id,
                    "status": "LIVE_READY",
                    "intent": {
                        "metadata": {
                            "position_intent": "HEDGE",
                            "hedge_recovery": True,
                        }
                    },
                }
            ]
        }

        allowed, recovery_bypass = _gpt_lanes_pass_prefilter_or_recovery(
            intents_payload=payload,
            gpt_lane_ids=(lane_id,),
            prefiltered_lane_ids=set(),
        )

        self.assertTrue(allowed)
        self.assertTrue(recovery_bypass)

        payload["results"][0]["intent"]["metadata"]["hedge_recovery"] = False
        allowed, recovery_bypass = _gpt_lanes_pass_prefilter_or_recovery(
            intents_payload=payload,
            gpt_lane_ids=(lane_id,),
            prefiltered_lane_ids=set(),
        )

        self.assertFalse(allowed)
        self.assertFalse(recovery_bypass)

    def test_expanded_gpt_basket_recovers_from_stale_selected_lane(self) -> None:
        current_lane = LaneScore(
            lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
            pair="EUR_USD",
            direction="LONG",
            method="BREAKOUT_FAILURE",
            order_type="STOP-ENTRY",
            entry=1.16522,
            tp=1.16878,
            sl=1.16401,
            status="LIVE_READY",
            score=110.0,
            action=ACTION_SEND_ENTRY,
            blockers=(),
            rationale=(),
            size_multiple=1.0,
            estimated_margin_jpy=7_400.0,
        )
        stale_gpt_lane = LaneScore(
            lane_id="trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
            pair="EUR_USD",
            direction="LONG",
            method="TREND_CONTINUATION",
            order_type="STOP-ENTRY",
            entry=1.16522,
            tp=1.16878,
            sl=1.16401,
            status="DRY_RUN_BLOCKED",
            score=120.0,
            action=ACTION_NO_TRADE,
            blockers=("intent status is DRY_RUN_BLOCKED",),
            rationale=(),
            size_multiple=1.0,
            estimated_margin_jpy=7_400.0,
        )
        decision = TraderDecision(
            action=ACTION_SEND_ENTRY,
            selected_lane_id=current_lane.lane_id,
            selected_lane_score=current_lane.score,
            selected_lane_size_multiple=current_lane.size_multiple,
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            reason="current prefilter selected the breakout-failure repair lane",
            scores=(current_lane, stale_gpt_lane),
            positions=2,
            orders=1,
        )

        lane_ids, size_multiples = AutoTradeCycle._expanded_gpt_basket_plan(
            decision=decision,
            gpt_lane_ids=(stale_gpt_lane.lane_id,),
            margin_room_jpy=10_000.0,
        )

        self.assertEqual(lane_ids, (current_lane.lane_id,))
        self.assertEqual(size_multiples, {current_lane.lane_id: 1.0})

    def test_direct_send_cycle_refuses_existing_live_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_dir = root / "lock"
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(str(os.getpid()) + "\n")
            old_lock = os.environ.get("QR_AUTOTRADE_LOCK_DIR")
            old_held = os.environ.get("QR_AUTOTRADE_LOCK_HELD")
            os.environ["QR_AUTOTRADE_LOCK_DIR"] = str(lock_dir)
            os.environ.pop("QR_AUTOTRADE_LOCK_HELD", None)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={"EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now)},
                    account=AccountSummary(
                        nav_jpy=110_000,
                        balance_jpy=110_000,
                        unrealized_pl_jpy=0.0,
                        fetched_at_utc=now,
                    ),
                )
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "another autotrade cycle is already running"):
                    AutoTradeCycle(
                        client=client,
                        snapshot_path=root / "snapshot.json",
                        intents_path=root / "intents.json",
                        intent_report_path=root / "intents.md",
                        decision_path=root / "decision.json",
                        decision_report_path=root / "decision.md",
                        report_path=root / "report.md",
                    ).run(send=True)
            finally:
                if old_lock is None:
                    os.environ.pop("QR_AUTOTRADE_LOCK_DIR", None)
                else:
                    os.environ["QR_AUTOTRADE_LOCK_DIR"] = old_lock
                if old_held is None:
                    os.environ.pop("QR_AUTOTRADE_LOCK_HELD", None)
                else:
                    os.environ["QR_AUTOTRADE_LOCK_HELD"] = old_held

            self.assertEqual(client.snapshot_calls, [])
            self.assertEqual(client.orders_sent, [])

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

    def test_no_live_ready_pending_cleanup_cancels_contaminated_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            pending = BrokerOrder(
                order_id="pending-1",
                pair="EUR_USD",
                order_type="STOP",
                price=1.1735,
                state="PENDING",
                units=1000,
                owner=Owner.TRADER,
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            _write_no_live_ready_intents(intents_path)
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
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_provider=StaticTraderProvider(_gpt_cancel_pending_decision(["pending-1"])),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

            self.assertEqual(summary.status, "CANCELED_GPT_PENDING")
            self.assertEqual(summary.gpt_action, "CANCEL_PENDING")
            self.assertEqual(summary.canceled_orders, ("pending-1",))
            self.assertEqual(client.orders_canceled, ["pending-1"])
            self.assertEqual(client.orders_sent, [])

    def test_trader_pending_order_can_add_verified_basket_when_risk_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(
                    BrokerOrder(
                        order_id="kept-pending",
                        pair="EUR_USD",
                        order_type="STOP",
                        price=1.1735,
                        state="PENDING",
                        units=1000,
                        owner=Owner.TRADER,
                        raw={
                            "takeProfitOnFill": {"price": "1.17550"},
                            "stopLossOnFill": {"price": "1.17290"},
                        },
                    ),
                ),
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
                gpt_attack_advice_path=root / "attack_missing.json",
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
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(client.orders_canceled, [])
            self.assertEqual(len(client.orders_sent), 1)
            self.assertEqual(summary.orders, 1)

    def test_pending_entry_dry_probe_runs_gpt_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(
                    BrokerOrder(
                        order_id="kept-pending",
                        pair="EUR_CHF",
                        order_type="LIMIT",
                        price=0.91653,
                        state="PENDING",
                        units=3600,
                        owner=Owner.TRADER,
                        raw={"takeProfitOnFill": {"price": "0.91810"}},
                    ),
                ),
                quotes={
                    "EUR_CHF": Quote("EUR_CHF", 0.91650, 0.91665, timestamp_utc=now),
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "CHF_JPY": Quote("CHF_JPY", 170.0, 170.02, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            _write_live_ready_intents(intents_path)
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
                gpt_attack_advice_path=root / "attack_missing.json",
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
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=False)

            self.assertIn(summary.status, {"STAGED", "BLOCKED"})
            self.assertEqual(summary.decision_source, "gpt_trader")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(summary.gpt_action, "TRADE")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])

    def test_gpt_trade_preserves_pending_cancel_candidate_when_cancel_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(
                    BrokerOrder(
                        order_id="stale-but-gpt-preserved",
                        pair="EUR_CHF",
                        order_type="LIMIT",
                        price=0.91653,
                        state="PENDING",
                        units=3600,
                        owner=Owner.TRADER,
                    ),
                ),
                quotes={
                    "EUR_CHF": Quote("EUR_CHF", 0.91650, 0.91665, timestamp_utc=now),
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "CHF_JPY": Quote("CHF_JPY", 170.0, 170.02, timestamp_utc=now),
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

            prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                summary = AutoTradeCycle(
                    client=client,
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    gpt_decision_path=root / "gpt_decision.json",
                    gpt_decision_report_path=root / "gpt_decision.md",
                    gpt_attack_advice_path=root / "attack_missing.json",
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
            finally:
                if prior_sl_free is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.decision_source, "gpt_trader")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(summary.canceled_orders, ())
            self.assertEqual(client.orders_canceled, [])
            self.assertEqual(len(client.orders_sent), 1)
            decision_payload = json.loads((root / "decision.json").read_text())
            self.assertEqual(decision_payload["pending_cancel_order_ids"], ["stale-but-gpt-preserved"])

    def test_gpt_trade_cancels_named_pending_orders_before_capacity_checked_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            pending_orders = tuple(
                BrokerOrder(
                    order_id=f"pending-{idx}",
                    pair="EUR_USD",
                    order_type="STOP",
                    price=1.1735 + idx * 0.0001,
                    state="PENDING",
                    units=1000,
                    owner=Owner.TRADER,
                    raw={
                        "takeProfitOnFill": {"price": "1.17550"},
                        "stopLossOnFill": {"price": "1.17290"},
                    },
                )
                for idx in range(4)
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=pending_orders,
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
            gpt_decision = _gpt_batch_trade_decision(
                [
                    market_lane,
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                ]
            )
            gpt_decision["cancel_order_ids"] = [order.order_id for order in pending_orders]

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_provider=StaticTraderProvider(gpt_decision),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(summary.canceled_orders, tuple(order.order_id for order in pending_orders))
            self.assertEqual(client.orders_canceled, [order.order_id for order in pending_orders])
            self.assertEqual(len(client.orders_sent), 1)

    def test_gpt_trade_not_prefiltered_still_cancels_named_pending_orders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            pending = BrokerOrder(
                order_id="stale-gpt-pending",
                pair="GBP_CHF",
                order_type="STOP",
                price=1.06824,
                state="PENDING",
                units=3600,
                owner=Owner.TRADER,
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                orders=(pending,),
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            _write_two_live_ready_intents(intents_path)
            payload = json.loads(intents_path.read_text())
            gpt_lane = "range_trader:NZD_CAD:SHORT:RANGE_ROTATION:MARKET"
            gpt_intent = json.loads(json.dumps(payload["results"][0]))
            gpt_intent["lane_id"] = gpt_lane
            gpt_intent["intent"]["pair"] = "NZD_CAD"
            gpt_intent["intent"]["side"] = "SHORT"
            gpt_intent["intent"]["order_type"] = "MARKET"
            gpt_intent["intent"]["units"] = 10_000
            gpt_intent["intent"]["entry"] = 0.81346
            gpt_intent["intent"]["tp"] = 0.81166
            gpt_intent["intent"]["sl"] = 0.81426
            gpt_intent["intent"]["thesis"] = "NZD_CAD range rotation is live-ready but outside the deterministic prefilter."
            gpt_intent["intent"]["market_context"]["regime"] = "RANGE_ROTATION"
            gpt_intent["intent"]["market_context"]["narrative"] = "NZD_CAD range rejection can rotate lower."
            gpt_intent["intent"]["market_context"]["chart_story"] = "M5 rejection at the range cap."
            gpt_intent["intent"]["market_context"]["method"] = "RANGE_ROTATION"
            gpt_intent["intent"]["market_context"]["invalidation"] = "range cap breaks higher"
            gpt_intent["risk_metrics"] = {
                "risk_jpy": 1_020.0,
                "reward_jpy": 2_142.0,
                "reward_risk": 2.1,
                "spread_pips": 0.9,
            }
            payload["results"].append(gpt_intent)
            intents_path.write_text(json.dumps(payload) + "\n")
            target_state = _open_target_state(root)
            deterministic_lane = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
            decision = TraderDecision(
                action=ACTION_SEND_ENTRY,
                selected_lane_id=deterministic_lane,
                selected_lane_score=90.0,
                selected_lane_size_multiple=1.0,
                generated_at_utc=now.isoformat(),
                reason="deterministic prefilter chose the stop-entry lane",
                scores=(
                    LaneScore(
                        lane_id=deterministic_lane,
                        pair="EUR_USD",
                        direction="LONG",
                        method="TREND_CONTINUATION",
                        order_type="STOP-ENTRY",
                        entry=1.1735,
                        tp=1.1750,
                        sl=1.1728,
                        status="LIVE_READY",
                        score=90.0,
                        action=ACTION_SEND_ENTRY,
                        blockers=(),
                        rationale=(),
                        size_multiple=1.0,
                        estimated_margin_jpy=5000.0,
                    ),
                ),
                positions=0,
                orders=1,
            )

            class StaticBrain:
                def run(self, snapshot):
                    return decision

            class PrefilterMismatchCycle(AutoTradeCycle):
                def _brain(self):
                    return StaticBrain()

            gpt_decision = _gpt_trade_decision(
                lane_id=gpt_lane,
                pair="NZD_CAD",
                method="RANGE_ROTATION",
                direction="SHORT",
            )
            gpt_decision["selected_lane_ids"] = [gpt_lane]
            gpt_decision["cancel_order_ids"] = [pending.order_id]
            client = FakeCycleClient(snapshot)

            summary = PrefilterMismatchCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_provider=StaticTraderProvider(gpt_decision),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

            self.assertEqual(summary.status, "GPT_DECISION_NOT_PREFILTERED")
            self.assertFalse(summary.sent)
            self.assertEqual(summary.sent_count, 0)
            self.assertEqual(summary.canceled_orders, (pending.order_id,))
            self.assertEqual(client.orders_canceled, [pending.order_id])
            self.assertEqual(client.orders_sent, [])

    def test_protected_position_with_pending_can_replace_pending_before_capacity_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            pending_orders = tuple(
                BrokerOrder(
                    order_id=f"pending-{idx}",
                    pair="EUR_USD",
                    order_type="STOP",
                    price=1.1735 + idx * 0.0001,
                    state="PENDING",
                    units=1000,
                    owner=Owner.TRADER,
                    raw={
                        "takeProfitOnFill": {"price": "1.17550"},
                        "stopLossOnFill": {"price": "1.17290"},
                    },
                )
                for idx in range(3)
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="protected-gbp",
                        pair="GBP_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.3600,
                        take_profit=1.3620,
                        stop_loss=1.3590,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=pending_orders,
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "GBP_USD": Quote("GBP_USD", 1.3602, 1.3603, timestamp_utc=now),
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
            gpt_decision = _gpt_batch_trade_decision(
                [
                    market_lane,
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                ]
            )
            gpt_decision["cancel_order_ids"] = [order.order_id for order in pending_orders]

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_provider=StaticTraderProvider(gpt_decision),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

            self.assertEqual(summary.position_management_action, "HOLD_PROTECTED")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(summary.canceled_orders, tuple(order.order_id for order in pending_orders))
            self.assertEqual(client.orders_canceled, [order.order_id for order in pending_orders])
            self.assertEqual(len(client.orders_sent), 1)

    def test_protected_position_gpt_trade_preserves_pending_cancel_candidate_when_cancel_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            pending = BrokerOrder(
                order_id="protected-stale-but-gpt-preserved",
                pair="EUR_USD",
                order_type="STOP",
                price=1.1800,
                state="PENDING",
                units=1000,
                owner=Owner.TRADER,
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="protected-gbp",
                        pair="GBP_USD",
                        side=Side.LONG,
                        units=1000,
                        entry_price=1.3600,
                        take_profit=1.3620,
                        stop_loss=1.3590,
                        owner=Owner.TRADER,
                    ),
                ),
                orders=(pending,),
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "GBP_USD": Quote("GBP_USD", 1.3602, 1.3603, timestamp_utc=now),
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
            gpt_decision = _gpt_batch_trade_decision(
                [
                    market_lane,
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                ]
            )
            gpt_decision["cancel_order_ids"] = []

            prior_sl_free = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            try:
                summary = AutoTradeCycle(
                    client=client,
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    gpt_decision_path=root / "gpt_decision.json",
                    gpt_decision_report_path=root / "gpt_decision.md",
                    gpt_attack_advice_path=root / "attack_missing.json",
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
                    gpt_provider=StaticTraderProvider(gpt_decision),
                    reuse_market_artifacts=True,
                    refresh_market_story=False,
                    live_enabled=True,
                    max_loss_jpy=1_500,
                ).run(send=True)
            finally:
                if prior_sl_free is None:
                    os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
                else:
                    os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl_free

            self.assertEqual(summary.position_management_action, "HOLD_PROTECTED")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(summary.canceled_orders, ())
            self.assertEqual(client.orders_canceled, [])
            self.assertEqual(len(client.orders_sent), 1)
            decision_payload = json.loads((root / "decision.json").read_text())
            self.assertEqual(decision_payload["pending_cancel_order_ids"], [pending.order_id])

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
                execution_ledger_db_path=root / "execution_ledger.db",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=True)

            self.assertEqual(summary.status, "TARGET_REACHED_PROTECT")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())
            self.assertFalse((root / "decision.json").exists())
            self.assertIn("protection-first no-send", (root / "report.md").read_text())

    def test_target_reached_cancels_trader_pending_entry(self) -> None:
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
            pending = BrokerOrder(
                order_id="target-hit-pending",
                pair="EUR_USD",
                order_type="STOP",
                price=1.1735,
                state="PENDING",
                units=1000,
                owner=Owner.TRADER,
            )
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    orders=(pending,),
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
                execution_ledger_db_path=root / "execution_ledger.db",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=True)

            self.assertEqual(summary.status, "CANCELED_TARGET_REACHED_PENDING")
            self.assertEqual(summary.canceled_orders, ("target-hit-pending",))
            self.assertEqual(client.orders_canceled, ["target-hit-pending"])
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())

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

    def test_monitor_cycle_clears_stale_live_order_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={"EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now)},
                )
            )
            target_state = _open_target_state(root)
            target_payload = json.loads(target_state.read_text())
            target_payload["campaign_day_jst"] = (
                now.astimezone(timezone(timedelta(hours=9))) - timedelta(hours=9)
            ).date().isoformat()
            target_payload["target_jpy"] = 10_000
            target_payload["realized_pl_jpy"] = 10_000
            target_payload["remaining_target_jpy"] = 0
            target_state.write_text(json.dumps(target_payload))
            live_order_path = root / "live_order.json"
            live_order_report = root / "live_order.md"
            live_order_path.write_text(json.dumps({"status": "SENT", "sent": True, "send_requested": True}))
            live_order_report.write_text("# Live Order Stage Report\n\n- Status: `SENT`\n")

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                live_order_output_path=live_order_path,
                live_order_report_path=live_order_report,
                report_path=root / "report.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                # Isolate from the repo's real data/execution_ledger.db: a
                # freshly synced dev ledger would override this fixture's
                # same-day realized_pl with ledger truth (0) and reopen the
                # target.
                execution_ledger_db_path=root / "execution_ledger.db",
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=True)

            self.assertEqual(summary.status, "TARGET_REACHED_PROTECT")
            result = json.loads(live_order_path.read_text())
            self.assertEqual(result["status"], "NO_ACTION")
            self.assertFalse(result["sent"])
            self.assertFalse(result["send_requested"])
            self.assertIn("NO_ACTION", live_order_report.read_text())

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
                gpt_attack_advice_path=root / "attack_missing.json",
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

    def test_gpt_close_defers_fresh_trade_until_next_cycle(self) -> None:
        prior_override = os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE")
        os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime.now(timezone.utc)
                target_state = _open_target_state(root)
                target_payload = json.loads(target_state.read_text())
                target_payload["start_balance_jpy"] = 400_000
                target_payload["remaining_target_jpy"] = 40_000
                target_payload["daily_risk_budget_jpy"] = 8_000
                target_state.write_text(json.dumps(target_payload) + "\n")
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    positions=(
                        BrokerPosition(
                            trade_id="close-me",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=1000,
                            entry_price=1.1740,
                            unrealized_pl_jpy=-120.0,
                            take_profit=1.1760,
                            stop_loss=1.1715,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                    account=AccountSummary(
                        nav_jpy=400_000,
                        balance_jpy=400_000,
                        margin_available_jpy=400_000,
                        fetched_at_utc=now,
                    ),
                )
                client = FakeCycleClient(snapshot)
                campaign_path = _campaign(root)
                pair_charts_path = root / "pair_charts.json"
                pair_charts_payload = json.loads(pair_charts_path.read_text())
                pair_charts_payload["charts"][0]["chart_story"] = (
                    "EUR_USD RANGE; "
                    "M15(RANGE, ADX=20 RSI=40 ATR=3.0p struct=BOS_DOWN@1.1720); "
                    "H4(RANGE, ADX=20 RSI=45 ATR=8.0p struct=CHOCH_DOWN@1.1700)"
                )
                pair_charts_path.write_text(json.dumps(pair_charts_payload) + "\n")
                settings_path = root / "settings.json"
                settings_path.write_text(
                    json.dumps({"size_by_score": {"enabled": False}}) + "\n"
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
                    gpt_attack_advice_path=root / "attack_missing.json",
                    position_management_path=root / "pm.json",
                    position_management_report_path=root / "pm.md",
                    position_execution_path=root / "pe.json",
                    position_execution_report_path=root / "pe.md",
                    live_order_output_path=root / "live_order.json",
                    live_order_report_path=root / "live_order.md",
                    report_path=root / "report.md",
                    campaign_plan_path=campaign_path,
                    pair_charts_path=pair_charts_path,
                    strategy_profile_path=_candidate_profile(root),
                    market_story_profile_path=_stories(root),
                    trader_settings_path=settings_path,
                    receipt_promotion_report_path=root / "promotion.md",
                    target_state_path=target_state,
                    target_report_path=root / "target.md",
                    gpt_target_state_path=target_state,
                    use_gpt_trader=True,
                    gpt_provider=SequenceTraderProvider(
                        _gpt_close_decision(["close-me"]),
                        _gpt_trade_decision(),
                    ),
                    refresh_market_story=False,
                    live_enabled=True,
                    max_loss_jpy=1_500,
                ).run(send=True)

                self.assertEqual(summary.status, "POSITION_ACTION_SENT")
                self.assertEqual(summary.decision_source, "gpt_trader")
                self.assertEqual(summary.position_execution_status, "SENT")
                self.assertTrue(summary.position_execution_sent)
                self.assertEqual(summary.gpt_action, "CLOSE")
                self.assertEqual(summary.gpt_recovery_source, "POST_CLOSE_REENTRY_DEFERRED")
                self.assertEqual(client.trades_closed, [("close-me", "ALL")])
                self.assertEqual(len(client.orders_sent), 0)
                self.assertIsNone(summary.selected_lane_id)
                self.assertFalse((root / "gpt_decision.close_reentry.json").exists())
                current_receipt = json.loads((root / "gpt_decision.json").read_text())
                self.assertEqual(current_receipt["decision"]["action"], "CLOSE")
        finally:
            if prior_override is None:
                os.environ.pop("QR_OPERATOR_CLOSE_OVERRIDE", None)
            else:
                os.environ["QR_OPERATOR_CLOSE_OVERRIDE"] = prior_override

    def test_reuse_market_artifacts_refreshes_positions_before_management(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_struct = os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir()
                now = datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc)
                stale_snapshot = BrokerSnapshot(
                    fetched_at_utc=now - timedelta(minutes=20),
                    positions=(
                        BrokerPosition(
                            trade_id="472071",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=6000,
                            entry_price=1.34702,
                            unrealized_pl_jpy=-50.0,
                            take_profit=1.34853,
                            stop_loss=None,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={"EUR_USD": Quote("EUR_USD", 1.34710, 1.34720, timestamp_utc=now - timedelta(minutes=20))},
                )
                fresh_snapshot = replace(
                    stale_snapshot,
                    fetched_at_utc=now,
                    positions=(replace(stale_snapshot.positions[0], unrealized_pl_jpy=-2981.9),),
                    quotes={"EUR_USD": Quote("EUR_USD", 1.34392, 1.34404, timestamp_utc=now)},
                )
                snapshot_path = root / "broker_snapshot.json"
                snapshot_path.write_text(_snapshot_to_json(stale_snapshot) + "\n")
                intents_path = root / "order_intents.json"
                intents_path.write_text(json.dumps({"results": []}) + "\n")
                (data_root / "entry_thesis_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-06-05T11:19:50Z",
                            "trade_id": "472071",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "entry_price": 1.34702,
                            "forecast_direction": "UP",
                            "forecast_confidence": 0.58,
                            "regime": "RANGE",
                            "invalidation_price": 1.34000,
                            "target_price": 1.34853,
                            "key_drivers": ["failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"],
                        }
                    )
                    + "\n"
                )
                pair_charts_path = _entry_invalidation_pair_charts(root)
                client = FakeCycleClient(fresh_snapshot)

                class CycleWithStaticBrain(AutoTradeCycle):
                    def _brain(self):  # type: ignore[override]
                        class Brain:
                            def run(self, snapshot: BrokerSnapshot) -> TraderDecision:
                                (root / "decision.json").write_text(json.dumps({"scores": []}) + "\n")
                                return TraderDecision(
                                    action=ACTION_NO_TRADE,
                                    selected_lane_id=None,
                                    generated_at_utc=now.isoformat(),
                                    reason="no entry while position management handles invalidation",
                                    scores=(),
                                    positions=len(snapshot.positions),
                                    orders=len(snapshot.orders),
                                )

                        return Brain()

                summary = CycleWithStaticBrain(
                    client=client,
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    position_management_path=data_root / "position_management.json",
                    position_management_report_path=root / "position_management.md",
                    position_execution_path=root / "position_execution.json",
                    position_execution_report_path=root / "position_execution.md",
                    live_order_output_path=root / "live_order.json",
                    live_order_report_path=root / "live_order.md",
                    report_path=root / "autotrade.md",
                    pair_charts_path=pair_charts_path,
                    target_state_path=None,
                    refresh_market_story=False,
                    reuse_market_artifacts=True,
                    live_enabled=True,
                ).run(send=True)

                self.assertEqual(summary.status, "POSITION_ACTION_SENT")
                self.assertEqual(summary.position_execution_status, "SENT")
                self.assertEqual(client.trades_closed, [("472071", "ALL")])
                self.assertGreaterEqual(len(client.snapshot_calls), 1)
                self.assertEqual(json.loads(snapshot_path.read_text())["fetched_at_utc"], stale_snapshot.fetched_at_utc.isoformat())
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_struct is None:
                os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
            else:
                os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = prior_struct

    def test_reuse_market_artifacts_keeps_structural_close_advisory_without_opt_in(self) -> None:
        prior_close = os.environ.get("QR_DISABLE_AUTO_CLOSE")
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        prior_struct = os.environ.get("QR_ALLOW_STRUCTURAL_AUTO_CLOSE")
        os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data_root = root / "data"
                data_root.mkdir()
                now = datetime(2026, 6, 5, 16, 0, tzinfo=timezone.utc)
                stale_snapshot = BrokerSnapshot(
                    fetched_at_utc=now - timedelta(minutes=20),
                    positions=(
                        BrokerPosition(
                            trade_id="472071",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=6000,
                            entry_price=1.34702,
                            unrealized_pl_jpy=-50.0,
                            take_profit=1.34853,
                            stop_loss=None,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={"EUR_USD": Quote("EUR_USD", 1.34710, 1.34720, timestamp_utc=now - timedelta(minutes=20))},
                )
                fresh_snapshot = replace(
                    stale_snapshot,
                    fetched_at_utc=now,
                    positions=(replace(stale_snapshot.positions[0], unrealized_pl_jpy=-2981.9),),
                    quotes={"EUR_USD": Quote("EUR_USD", 1.34392, 1.34404, timestamp_utc=now)},
                )
                snapshot_path = root / "broker_snapshot.json"
                snapshot_path.write_text(_snapshot_to_json(stale_snapshot) + "\n")
                intents_path = root / "order_intents.json"
                intents_path.write_text(json.dumps({"results": []}) + "\n")
                (data_root / "entry_thesis_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_utc": "2026-06-05T11:19:50Z",
                            "trade_id": "472071",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "entry_price": 1.34702,
                            "forecast_direction": "UP",
                            "forecast_confidence": 0.58,
                            "regime": "RANGE",
                            "invalidation_price": 1.34000,
                            "target_price": 1.34853,
                            "key_drivers": ["failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE"],
                        }
                    )
                    + "\n"
                )
                pair_charts_path = _entry_invalidation_pair_charts(root)
                client = FakeCycleClient(fresh_snapshot)

                class CycleWithStaticBrain(AutoTradeCycle):
                    def _brain(self):  # type: ignore[override]
                        class Brain:
                            def run(self, snapshot: BrokerSnapshot) -> TraderDecision:
                                (root / "decision.json").write_text(json.dumps({"scores": []}) + "\n")
                                return TraderDecision(
                                    action=ACTION_NO_TRADE,
                                    selected_lane_id=None,
                                    generated_at_utc=now.isoformat(),
                                    reason="no entry while position management handles invalidation",
                                    scores=(),
                                    positions=len(snapshot.positions),
                                    orders=len(snapshot.orders),
                                )

                        return Brain()

                summary = CycleWithStaticBrain(
                    client=client,
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    position_management_path=data_root / "position_management.json",
                    position_management_report_path=root / "position_management.md",
                    position_execution_path=root / "position_execution.json",
                    position_execution_report_path=root / "position_execution.md",
                    live_order_output_path=root / "live_order.json",
                    live_order_report_path=root / "live_order.md",
                    report_path=root / "autotrade.md",
                    pair_charts_path=pair_charts_path,
                    target_state_path=None,
                    refresh_market_story=False,
                    reuse_market_artifacts=True,
                    live_enabled=True,
                ).run(send=True)

                self.assertEqual(summary.position_management_action, "HOLD_PROTECTED")
                self.assertEqual(summary.position_execution_status, "NO_ACTION")
                self.assertEqual(client.trades_closed, [])
                self.assertIn("QR_ALLOW_STRUCTURAL_AUTO_CLOSE=1", (root / "position_management.md").read_text())
        finally:
            if prior_close is None:
                os.environ.pop("QR_DISABLE_AUTO_CLOSE", None)
            else:
                os.environ["QR_DISABLE_AUTO_CLOSE"] = prior_close
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl
            if prior_struct is None:
                os.environ.pop("QR_ALLOW_STRUCTURAL_AUTO_CLOSE", None)
            else:
                os.environ["QR_ALLOW_STRUCTURAL_AUTO_CLOSE"] = prior_struct

    def test_gpt_handoff_runs_learning_audit_before_decision_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            lane_id = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION"
            attack_advice_path = root / "ai_attack_advice.json"
            _write_learning_audit_artifacts(root, attack_advice_path=attack_advice_path, lane_id=lane_id)
            decision = _gpt_trade_decision(lane_id=lane_id)
            decision["evidence_refs"].extend(
                ["attack:advice", f"attack:lane:{lane_id}", "learning:audit", f"learning:lane:{lane_id}"]
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
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=attack_advice_path,
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

            self.assertEqual(summary.gpt_status, "ACCEPTED")
            audit_payload = json.loads((root / "learning_audit.json").read_text())
            self.assertEqual(audit_payload["learning_influence"]["influenced_lanes"], 1)
            gpt_payload = json.loads((root / "gpt_decision.json").read_text())
            self.assertEqual(gpt_payload["input_packet"]["learning_audit"]["status"], audit_payload["status"])
            self.assertIn("learning:audit", gpt_payload["input_packet"]["allowed_evidence_refs"])
            verification_payload = json.loads((root / "verification_ledger.json").read_text())
            self.assertEqual(
                gpt_payload["input_packet"]["verification_ledger"]["status"],
                verification_payload["status"],
            )
            self.assertIn("verification:ledger", gpt_payload["input_packet"]["allowed_evidence_refs"])
            with sqlite3.connect(root / "execution_ledger.db") as conn:
                rows = conn.execute("SELECT COUNT(*) FROM verification_observations").fetchone()[0]
            self.assertGreater(rows, 0)

    def test_gpt_single_trade_dedupes_prefiltered_parent_variants(self) -> None:
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
                gpt_attack_advice_path=root / "attack_missing.json",
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
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(
                summary.selected_lane_ids,
                ("trend_trader:EUR_USD:LONG:TREND_CONTINUATION",),
            )
            self.assertEqual(len(client.orders_sent), 1)

    def test_gpt_batch_trade_dedupes_same_parent_variants(self) -> None:
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
                gpt_attack_advice_path=root / "attack_missing.json",
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
            self.assertEqual(summary.sent_count, 1)
            self.assertEqual(
                summary.selected_lane_ids,
                ("trend_trader:EUR_USD:LONG:TREND_CONTINUATION",),
            )
            self.assertEqual(len(client.orders_sent), 1)
            result = json.loads((root / "live_order.json").read_text())
            self.assertEqual(result["lane_id"], "trend_trader:EUR_USD:LONG:TREND_CONTINUATION")
            self.assertNotIn("orders", result)

    def test_cycle_appends_jsonl_audit_entry_per_run(self) -> None:
        """Regression: each cycle must append one JSONL line to the trader
        journal. AGENT_CONTRACT §6 / §11 require an append-only audit trail
        of decisions and execution outcomes; latest-state files like
        `data/live_order_request.json` are overwritten each cycle and
        cannot serve as a long-term record.
        """
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
            journal_path = root / "logs" / "trader_journal.jsonl"

            cycle = AutoTradeCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                trader_journal_path=journal_path,
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
            )
            summary = cycle.run(send=True)
            self.assertEqual(summary.status, "SENT")
            self.assertTrue(journal_path.exists())
            entries = [
                json.loads(line)
                for line in journal_path.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["status"], "SENT")
            self.assertTrue(entry["sent"])
            self.assertEqual(entry["sent_count"], 1)
            self.assertEqual(
                entry["selected_lane_ids"],
                [
                    "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
                ],
            )
            self.assertIn("ts", entry)
            # Run a second cycle to confirm the journal is append-only.
            cycle.run(send=True)
            entries = [
                json.loads(line)
                for line in journal_path.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(entries), 2)

    def test_injected_client_does_not_write_default_trader_journal_without_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            journal_path = root / "should_not_write.jsonl"
            cycle = AutoTradeCycle(client=object())
            cycle.trader_journal_path = journal_path

            cycle._append_trader_journal_entry(
                AutoTradeCycleSummary(
                    status="NO_ACTION",
                    report_path=root / "report.md",
                    snapshot_path=root / "snapshot.json",
                    intents_path=root / "intents.json",
                    selected_lane_id=None,
                    deterministic_lane_id=None,
                    sent=False,
                    positions=0,
                    orders=0,
                    live_ready=0,
                )
            )

            self.assertFalse(journal_path.exists())

    def test_cycle_syncs_execution_ledger_around_live_send(self) -> None:
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
            _write_live_ready_intents(intents_path)
            target_state = _open_target_state(root)
            client = LedgerCycleClient(snapshot)
            ledger_path = root / "execution_ledger.db"

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                trader_journal_path=root / "trader_journal.jsonl",
                execution_ledger_db_path=ledger_path,
                execution_ledger_report_path=root / "execution_ledger.md",
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
                refresh_market_story=False,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=True)

            self.assertEqual(summary.status, "SENT")
            self.assertEqual(client.transaction_sync_calls, ["100"])
            with sqlite3.connect(ledger_path) as conn:
                event_types = {
                    row[0] for row in conn.execute("SELECT event_type FROM execution_events").fetchall()
                }
                tx_count = conn.execute("SELECT COUNT(*) FROM oanda_transactions").fetchone()[0]
                receipt_count = conn.execute("SELECT COUNT(*) FROM gateway_receipts").fetchone()[0]
                last_id = conn.execute(
                    "SELECT value FROM sync_state WHERE key='last_oanda_transaction_id'"
                ).fetchone()[0]

            self.assertIn("GATEWAY_ORDER_SENT", event_types)
            self.assertIn("ORDER_FILLED", event_types)
            self.assertEqual(tx_count, 1)
            self.assertGreaterEqual(receipt_count, 1)
            self.assertEqual(last_id, "101")

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
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_attack_advice_path=root / "attack_missing.json",
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

    def test_gpt_wait_without_live_ready_does_not_regenerate_intents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            target_state = _open_target_state(root)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(_with_account(snapshot)) + "\n")
            intents_path = root / "intents.json"
            intents_path.write_text(json.dumps({"results": []}) + "\n")
            campaign_path = _campaign(root)
            wait_decision = _gpt_wait_decision()
            wait_decision["evidence_refs"] = ["broker:snapshot", "target:daily"]
            wait_decision["twenty_minute_plan"]["evidence_refs"] = ["broker:snapshot", "target:daily"]

            class NoRetryCycle(AutoTradeCycle):
                def _intent_generator(self, max_loss_jpy: float | None = None):  # type: ignore[override]
                    raise AssertionError("WAIT without LIVE_READY must not regenerate intents")

            client = FakeCycleClient(snapshot)
            summary = NoRetryCycle(
                client=client,
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=campaign_path,
                pair_charts_path=root / "pair_charts.json",
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                gpt_target_state_path=target_state,
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(wait_decision),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "GPT_WAIT")
            self.assertEqual(summary.live_ready, 0)
            self.assertEqual(summary.gpt_wait_retries, 0)
            self.assertFalse(summary.campaign_exposure_required)
            self.assertEqual(client.snapshot_calls, [])
            self.assertFalse((root / "live_order.json").exists())
            self.assertIn("GPT wait recovery attempts: `0`", (root / "report.md").read_text())

    def test_reused_verified_gpt_wait_requires_fresh_receipt_before_gateway(self) -> None:
        prior_telemetry = os.environ.get("QR_REQUIRE_TELEMETRY_FOR_LIVE")
        os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            try:
                root = Path(tmp)
                now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
                target_state = _open_target_state(root)
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                )
                snapshot_path = root / "snapshot.json"
                snapshot_path.write_text(_snapshot_to_json(_with_account(snapshot)) + "\n")
                intents_path = root / "intents.json"
                intents_path.write_text(json.dumps({"generated_at_utc": now.isoformat(), "results": []}) + "\n")
                pair_charts_path = root / "pair_charts.json"
                pair_charts_path.write_text(json.dumps({"charts": []}) + "\n")
                emitted = now - timedelta(minutes=90)
                (root / "projection_ledger.jsonl").write_text(
                    json.dumps(
                        {
                            "timestamp_emitted_utc": emitted.isoformat().replace("+00:00", "Z"),
                            "pair": "EUR_USD",
                            "signal_name": "directional_forecast",
                            "direction": "UP",
                            "lead_time_min": 30,
                            "confidence": 0.8,
                            "entry_price": 1.1720,
                            "predicted_target_price": 1.1728,
                            "predicted_invalidation_price": 1.1710,
                            "resolution_window_min": 60,
                            "resolution_status": "PENDING",
                            "cycle_id": "stale-wait-cycle",
                        }
                    )
                    + "\n"
                )
                response_path = root / "codex_trader_decision_response.json"
                wait_decision = _gpt_wait_decision()
                response_path.write_text(json.dumps(wait_decision) + "\n")
                gpt_decision_path = root / "gpt_decision.json"
                gpt_decision_path.write_text(
                    json.dumps(
                        {
                            "generated_at_utc": now.isoformat(),
                            "status": "ACCEPTED",
                            "decision": {"action": "WAIT"},
                            "verification_issues": [],
                        }
                    )
                    + "\n"
                )
                os.utime(snapshot_path, (100.0, 100.0))
                os.utime(intents_path, (100.0, 100.0))
                os.utime(response_path, (101.0, 101.0))
                os.utime(gpt_decision_path, (102.0, 102.0))

                summary = AutoTradeCycle(
                    client=FakeCycleClient(snapshot),
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    gpt_decision_path=gpt_decision_path,
                    gpt_decision_report_path=root / "gpt_decision.md",
                    gpt_attack_advice_path=root / "attack_missing.json",
                    position_management_path=root / "pm.json",
                    position_management_report_path=root / "pm.md",
                    position_execution_path=root / "pe.json",
                    position_execution_report_path=root / "pe.md",
                    live_order_output_path=root / "live_order.json",
                    live_order_report_path=root / "live_order.md",
                    report_path=root / "report.md",
                    campaign_plan_path=_campaign(root),
                    pair_charts_path=pair_charts_path,
                    strategy_profile_path=_candidate_profile(root),
                    market_story_profile_path=_stories(root),
                    receipt_promotion_report_path=root / "promotion.md",
                    target_state_path=target_state,
                    target_report_path=root / "target.md",
                    gpt_target_state_path=target_state,
                    use_gpt_trader=True,
                    gpt_provider=StaticTraderProvider(wait_decision, source_path=response_path),
                    reuse_market_artifacts=True,
                    refresh_market_story=False,
                    live_enabled=True,
                ).run(send=True)

                projection_row = json.loads((root / "projection_ledger.jsonl").read_text())
                # §2/§8: a consumed/already-verified receipt degrades the cycle
                # to deterministic continuation instead of stopping it. With
                # no current LIVE_READY lane, the root status must name the
                # executable opportunity gap while retaining stale GPT evidence.
                self.assertEqual(summary.status, "NO_LIVE_READY_INTENT")
                self.assertEqual(summary.gpt_status, "STALE_DECISION")
                self.assertEqual(projection_row["resolution_status"], "HIT")
                self.assertFalse((root / "live_order.json").exists())
                self.assertIn("already verified as ACCEPTED WAIT", summary.gpt_error or "")
                report_text = (root / "report.md").read_text()
                self.assertIn("NO_LIVE_READY_INTENT", report_text)
            finally:
                if prior_telemetry is None:
                    os.environ.pop("QR_REQUIRE_TELEMETRY_FOR_LIVE", None)
                else:
                    os.environ["QR_REQUIRE_TELEMETRY_FOR_LIVE"] = prior_telemetry

    def test_stale_gpt_receipt_degrades_to_deterministic_position_cycle(self) -> None:
        # Regression for the 2026-06-11 cycle-poisoning bug: 13 of 25 live
        # cycles ended as full-stop STALE_GPT_DECISION_REFRESH_REQUIRED no-ops
        # because an already-verified ACCEPTED WAIT receipt blocked the whole
        # run before position management. A stale receipt must degrade the
        # cycle to deterministic continuation (position management still
        # runs) without ever handing the stale receipt to the gateway.
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
            response_path = root / "codex_trader_decision_response.json"
            wait_decision = _gpt_wait_decision()
            response_path.write_text(json.dumps(wait_decision) + "\n")
            gpt_decision_path = root / "gpt_decision.json"
            gpt_decision_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "status": "ACCEPTED",
                        "decision": {"action": "WAIT"},
                        "verification_issues": [],
                    }
                )
                + "\n"
            )
            os.utime(response_path, (100.0, 100.0))
            os.utime(gpt_decision_path, (101.0, 101.0))

            summary = AutoTradeCycle(
                client=client,
                snapshot_path=root / "snapshot.json",
                intents_path=root / "intents.json",
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=gpt_decision_path,
                gpt_decision_report_path=root / "gpt_decision.md",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(wait_decision, source_path=response_path),
                live_enabled=True,
            ).run(send=False)

            self.assertEqual(summary.status, "MONITOR_ONLY_EXPOSURE_OPEN")
            self.assertTrue((root / "pm.json").exists())
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())

    def test_reused_verified_gpt_close_reaches_position_gateway(self) -> None:
        # The verifier-to-gateway bridge must cover accepted CLOSE receipts,
        # not just TRADE. Otherwise hard Gate A exits are repeatedly reported
        # as accepted but never reach PositionProtectionGateway, blocking fresh
        # entries behind the still-open position.
        prior_sl = os.environ.get("QR_TRADER_DISABLE_SL_REPAIR")
        os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                now = datetime.now(timezone.utc)
                target_state = _open_target_state(root)
                snapshot = BrokerSnapshot(
                    fetched_at_utc=now,
                    positions=(
                        BrokerPosition(
                            trade_id="close-me",
                            pair="EUR_USD",
                            side=Side.LONG,
                            units=1000,
                            entry_price=1.1740,
                            unrealized_pl_jpy=-120.0,
                            take_profit=1.1760,
                            stop_loss=None,
                            owner=Owner.TRADER,
                        ),
                    ),
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                        "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                    },
                    account=AccountSummary(
                        nav_jpy=400_000,
                        balance_jpy=400_000,
                        margin_available_jpy=400_000,
                        fetched_at_utc=now,
                    ),
                )
                snapshot_path = root / "snapshot.json"
                snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
                intents_path = root / "intents.json"
                _write_no_live_ready_intents(intents_path)
                pair_charts_path = root / "pair_charts.json"
                pair_charts_path.write_text(json.dumps({"charts": []}) + "\n")
                response_path = root / "codex_trader_decision_response.json"
                close_decision = _gpt_close_decision(["close-me"])
                response_path.write_text(json.dumps(close_decision) + "\n")
                gpt_decision_path = root / "gpt_decision.json"
                gpt_decision_path.write_text(
                    json.dumps(
                        {
                            "generated_at_utc": now.isoformat(),
                            "status": "ACCEPTED",
                            "decision": close_decision,
                            "verification_issues": [],
                        }
                    )
                    + "\n"
                )
                live_order_path = root / "live_order.json"
                live_order_path.write_text(json.dumps({"status": "REJECTED"}) + "\n")
                position_execution_path = root / "pe.json"
                position_execution_path.write_text(
                    json.dumps(
                        {
                            "generated_at_utc": now.isoformat(),
                            "status": "NO_ACTION",
                            "send_requested": True,
                            "sent": False,
                            "actions": [
                                {
                                    "trade_id": "close-me",
                                    "pair": "EUR_USD",
                                    "owner": "trader",
                                    "management_action": "HOLD_PROTECTED",
                                    "reasons": ["deterministic review did not send"],
                                    "request": None,
                                    "issues": [],
                                    "sent": False,
                                    "response": None,
                                }
                            ],
                        }
                    )
                    + "\n"
                )
                os.utime(snapshot_path, (100.0, 100.0))
                os.utime(intents_path, (100.0, 100.0))
                os.utime(response_path, (101.0, 101.0))
                os.utime(gpt_decision_path, (102.0, 102.0))
                os.utime(live_order_path, (103.0, 103.0))
                os.utime(position_execution_path, (104.0, 104.0))
                client = FakeCycleClient(snapshot)

                summary = AutoTradeCycle(
                    client=client,
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    gpt_decision_path=gpt_decision_path,
                    gpt_decision_report_path=root / "gpt_decision.md",
                    gpt_attack_advice_path=root / "attack_missing.json",
                    position_management_path=root / "pm.json",
                    position_management_report_path=root / "pm.md",
                    position_execution_path=root / "pe.json",
                    position_execution_report_path=root / "pe.md",
                    live_order_output_path=live_order_path,
                    live_order_report_path=root / "live_order.md",
                    report_path=root / "report.md",
                    campaign_plan_path=_campaign(root),
                    pair_charts_path=pair_charts_path,
                    strategy_profile_path=_candidate_profile(root),
                    market_story_profile_path=_stories(root),
                    receipt_promotion_report_path=root / "promotion.md",
                    target_state_path=target_state,
                    target_report_path=root / "target.md",
                    gpt_target_state_path=target_state,
                    use_gpt_trader=True,
                    gpt_provider=StaticTraderProvider(close_decision, source_path=response_path),
                    reuse_market_artifacts=True,
                    refresh_market_story=False,
                    live_enabled=True,
                ).run(send=True)
        finally:
            if prior_sl is None:
                os.environ.pop("QR_TRADER_DISABLE_SL_REPAIR", None)
            else:
                os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = prior_sl

        self.assertEqual(summary.status, "POSITION_ACTION_SENT")
        self.assertEqual(summary.decision_source, "gpt_trader")
        self.assertEqual(summary.gpt_status, "ACCEPTED")
        self.assertEqual(summary.gpt_action, "CLOSE")
        self.assertEqual(summary.position_execution_status, "SENT")
        self.assertEqual(client.trades_closed, [("close-me", "ALL")])
        self.assertEqual(client.orders_sent, [])

    def test_gpt_decision_response_older_than_snapshot_requires_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            target_state = _open_target_state(root)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.0, 157.01, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(_with_account(snapshot)) + "\n")
            intents_path = root / "intents.json"
            intents_path.write_text(json.dumps({"generated_at_utc": now.isoformat(), "results": []}) + "\n")
            response_path = root / "codex_trader_decision_response.json"
            wait_decision = _gpt_wait_decision()
            response_path.write_text(json.dumps(wait_decision) + "\n")
            os.utime(response_path, (100.0, 100.0))
            os.utime(snapshot_path, (101.0, 101.0))
            os.utime(intents_path, (99.0, 99.0))

            summary = AutoTradeCycle(
                client=FakeCycleClient(snapshot),
                snapshot_path=snapshot_path,
                intents_path=intents_path,
                intent_report_path=root / "intents.md",
                decision_path=root / "decision.json",
                decision_report_path=root / "decision.md",
                gpt_decision_path=root / "gpt_decision.json",
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
                position_management_path=root / "pm.json",
                position_management_report_path=root / "pm.md",
                position_execution_path=root / "pe.json",
                position_execution_report_path=root / "pe.md",
                live_order_output_path=root / "live_order.json",
                live_order_report_path=root / "live_order.md",
                report_path=root / "report.md",
                campaign_plan_path=_campaign(root),
                pair_charts_path=root / "pair_charts.json",
                strategy_profile_path=_candidate_profile(root),
                market_story_profile_path=_stories(root),
                receipt_promotion_report_path=root / "promotion.md",
                target_state_path=target_state,
                target_report_path=root / "target.md",
                gpt_target_state_path=target_state,
                use_gpt_trader=True,
                gpt_provider=StaticTraderProvider(wait_decision, source_path=response_path),
                reuse_market_artifacts=True,
                refresh_market_story=False,
                live_enabled=True,
            ).run(send=True)

            # §2/§8: a receipt that predates current market artifacts degrades
            # the cycle to deterministic continuation. Since there is no
            # current LIVE_READY lane, the cycle reports the opportunity gap
            # instead of making stale GPT look like the entry blocker.
            self.assertEqual(summary.status, "NO_LIVE_READY_INTENT")
            self.assertEqual(summary.gpt_status, "STALE_DECISION")
            self.assertFalse((root / "live_order.json").exists())
            self.assertIn("predates broker snapshot", summary.gpt_error or "")

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
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_attack_advice_path=root / "attack_missing.json",
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

    def test_gpt_recovery_hedge_can_bypass_open_position_prefilter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=(
                    BrokerPosition(
                        trade_id="trapped-long",
                        pair="EUR_USD",
                        side=Side.LONG,
                        units=22_000,
                        entry_price=1.16688,
                        take_profit=None,
                        stop_loss=None,
                        owner=Owner.TRADER,
                        unrealized_pl_jpy=-22_000.0,
                    ),
                ),
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.16072, 1.16080, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 159.30, 159.31, timestamp_utc=now),
                },
                account=AccountSummary(
                    nav_jpy=170_000.0,
                    balance_jpy=192_000.0,
                    unrealized_pl_jpy=-22_000.0,
                    margin_used_jpy=162_000.0,
                    margin_available_jpy=8_000.0,
                    hedging_enabled=True,
                    fetched_at_utc=now,
                ),
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            hedge_lane = "trend_trader:EUR_USD:SHORT:TREND_CONTINUATION"
            _write_recovery_hedge_intents(intents_path, hedge_lane)
            target_state = _open_target_state(root)
            target_payload = json.loads(target_state.read_text())
            target_payload["daily_risk_budget_jpy"] = 50_000
            target_state.write_text(json.dumps(target_payload) + "\n")
            client = FakeCycleClient(snapshot)
            old_env = {name: os.environ.get(name) for name in ("QR_TRADER_DISABLE_SL_REPAIR", "QR_DISABLE_AUTO_CLOSE")}
            os.environ["QR_TRADER_DISABLE_SL_REPAIR"] = "1"
            os.environ["QR_DISABLE_AUTO_CLOSE"] = "1"
            try:
                summary = AutoTradeCycle(
                    client=client,
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    gpt_decision_path=root / "gpt_decision.json",
                    gpt_decision_report_path=root / "gpt_decision.md",
                    gpt_attack_advice_path=root / "attack_missing.json",
                    position_management_path=root / "pm.json",
                    position_management_report_path=root / "pm.md",
                    position_execution_path=root / "pe.json",
                    position_execution_report_path=root / "pe.md",
                    live_order_output_path=root / "live_order.json",
                    live_order_report_path=root / "live_order.md",
                    report_path=root / "report.md",
                    campaign_plan_path=_short_hedge_campaign(root),
                    strategy_profile_path=_short_candidate_profile(root),
                    market_story_profile_path=_short_stories(root),
                    receipt_promotion_report_path=root / "promotion.md",
                    target_state_path=target_state,
                    target_report_path=root / "target.md",
                    gpt_target_state_path=target_state,
                    use_gpt_trader=True,
                    gpt_provider=StaticTraderProvider(
                        _gpt_trade_decision(
                            lane_id=hedge_lane,
                            pair="EUR_USD",
                            method="TREND_CONTINUATION",
                            direction="SHORT",
                        )
                    ),
                    reuse_market_artifacts=True,
                    refresh_market_story=False,
                    live_enabled=True,
                    max_loss_jpy=10_000,
                ).run(send=False)
            finally:
                for name, value in old_env.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.selected_lane_id, hedge_lane)
            self.assertEqual(client.orders_sent, [])
            payload = json.loads((root / "live_order.json").read_text())
            self.assertEqual(payload["order_request"]["units"], "-22000")

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
                gpt_attack_advice_path=root / "attack_missing.json",
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

    def test_reuse_market_artifacts_reuses_accepted_gpt_packet_after_snapshot_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            pinned_ts = now - timedelta(seconds=30)
            intents_ts = pinned_ts + timedelta(seconds=10)
            decision_ts = pinned_ts + timedelta(seconds=20)
            refreshed_ts = now
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(
                _snapshot_to_json(
                    BrokerSnapshot(
                        fetched_at_utc=pinned_ts,
                        quotes={
                            "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=pinned_ts),
                            "USD_JPY": Quote("USD_JPY", 157.000, 157.004, timestamp_utc=pinned_ts),
                        },
                    )
                )
                + "\n"
            )
            intents_path = root / "intents.json"
            _write_live_ready_intents(intents_path)
            intents_payload = json.loads(intents_path.read_text())
            intents_payload["generated_at_utc"] = intents_ts.isoformat()
            intents_path.write_text(json.dumps(intents_payload) + "\n")
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
            response_path = root / "codex_trader_decision_response.json"
            decision = _gpt_trade_decision()
            decision["generated_at_utc"] = decision_ts.isoformat()
            response_path.write_text(json.dumps(decision) + "\n")
            gpt_decision_path = root / "gpt_decision.json"
            gpt_decision_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": (decision_ts + timedelta(seconds=1)).isoformat(),
                        "status": "ACCEPTED",
                        "decision": decision,
                        "verification_issues": [],
                        "input_packet": {
                            "artifact_timestamps": {
                                "order_intents_generated_at_utc": intents_ts.isoformat()
                            },
                            "broker_snapshot": {"fetched_at_utc": pinned_ts.isoformat()},
                        },
                    }
                )
                + "\n"
            )
            os.utime(response_path, (100.0, 100.0))
            os.utime(gpt_decision_path, (101.0, 101.0))
            snapshot_path.write_text(
                _snapshot_to_json(
                    BrokerSnapshot(
                        fetched_at_utc=refreshed_ts,
                        quotes={
                            "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=refreshed_ts),
                            "USD_JPY": Quote("USD_JPY", 157.000, 157.004, timestamp_utc=refreshed_ts),
                        },
                    )
                )
                + "\n"
            )
            os.utime(snapshot_path, (102.0, 102.0))
            client = FakeCycleClient(
                BrokerSnapshot(
                    fetched_at_utc=refreshed_ts,
                    quotes={
                        "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=refreshed_ts),
                        "USD_JPY": Quote("USD_JPY", 157.000, 157.004, timestamp_utc=refreshed_ts),
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
                gpt_decision_path=gpt_decision_path,
                gpt_decision_report_path=root / "gpt_decision.md",
                gpt_attack_advice_path=root / "attack_missing.json",
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
                gpt_provider=StaticTraderProvider(decision, source_path=response_path),
                reuse_market_artifacts=True,
                live_enabled=True,
                max_loss_jpy=1_500,
            ).run(send=False)

            self.assertEqual(summary.status, "STAGED")
            self.assertEqual(summary.gpt_status, "ACCEPTED")
            self.assertEqual(summary.gpt_action, "TRADE")
            payload = json.loads(gpt_decision_path.read_text())
            self.assertEqual(payload["status"], "ACCEPTED")
            self.assertEqual(
                payload["input_packet"]["broker_snapshot"]["fetched_at_utc"],
                pinned_ts.isoformat(),
            )

    def test_gpt_trade_at_position_capacity_stays_monitor_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc)
            positions = tuple(
                BrokerPosition(
                    trade_id=f"protected-{idx}",
                    pair="EUR_USD",
                    side=Side.LONG,
                    units=1000,
                    entry_price=1.1740 + idx * 0.0001,
                    take_profit=1.1800 + idx * 0.0001,
                    stop_loss=1.1600,
                    owner=Owner.TRADER,
                )
                for idx in range(4)
            )
            snapshot = BrokerSnapshot(
                fetched_at_utc=now,
                positions=positions,
                quotes={
                    "EUR_USD": Quote("EUR_USD", 1.17298, 1.17306, timestamp_utc=now),
                    "USD_JPY": Quote("USD_JPY", 157.000, 157.004, timestamp_utc=now),
                },
            )
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(_snapshot_to_json(snapshot) + "\n")
            intents_path = root / "intents.json"
            _write_live_ready_intents(intents_path)
            target_state = _open_target_state(root)
            client = FakeCycleClient(snapshot)
            old_cap = os.environ.get("QR_MAX_PORTFOLIO_POSITIONS")
            os.environ["QR_MAX_PORTFOLIO_POSITIONS"] = "4"
            try:
                summary = AutoTradeCycle(
                    client=client,
                    snapshot_path=snapshot_path,
                    intents_path=intents_path,
                    intent_report_path=root / "intents.md",
                    decision_path=root / "decision.json",
                    decision_report_path=root / "decision.md",
                    gpt_decision_path=root / "gpt_decision.json",
                    gpt_decision_report_path=root / "gpt_decision.md",
                    gpt_attack_advice_path=root / "attack_missing.json",
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
                ).run(send=True)
            finally:
                if old_cap is None:
                    os.environ.pop("QR_MAX_PORTFOLIO_POSITIONS", None)
                else:
                    os.environ["QR_MAX_PORTFOLIO_POSITIONS"] = old_cap

            self.assertEqual(summary.status, "MONITOR_ONLY_EXPOSURE_OPEN")
            self.assertEqual(summary.gpt_action, "TRADE")
            self.assertFalse(summary.sent)
            self.assertEqual(client.orders_sent, [])
            self.assertFalse((root / "live_order.json").exists())

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
                gpt_attack_advice_path=root / "attack_missing.json",
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
        self.trades_closed: list[tuple[str, str]] = []

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_calls.append(pairs)
        return self.snapshot_value

    def post_order_json(self, order_request: dict[str, Any]) -> dict[str, Any]:
        self.orders_sent.append(order_request)
        return {"orderCreateTransaction": {"id": "1"}}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.orders_canceled.append(order_id)
        self.snapshot_value = replace(
            self.snapshot_value,
            orders=tuple(order for order in self.snapshot_value.orders if order.order_id != order_id),
        )
        return {"orderCancelTransaction": {"id": "2", "orderID": order_id}}

    def close_trade(self, trade_id: str, units: str = "ALL") -> dict[str, Any]:
        self.trades_closed.append((trade_id, units))
        self.snapshot_value = replace(
            self.snapshot_value,
            positions=tuple(position for position in self.snapshot_value.positions if position.trade_id != trade_id),
        )
        return {"relatedTransactionIDs": ["3"], "closedTradeID": trade_id}


class SequenceCycleClient(FakeCycleClient):
    def __init__(self, snapshots: tuple[BrokerSnapshot, ...]) -> None:
        snapshots_with_account = tuple(_with_account(snapshot) for snapshot in snapshots)
        super().__init__(snapshots_with_account[-1])
        self.snapshots = snapshots_with_account

    def snapshot(self, pairs: tuple[str, ...]) -> BrokerSnapshot:
        self.snapshot_calls.append(pairs)
        index = min(len(self.snapshot_calls) - 1, len(self.snapshots) - 1)
        return self.snapshots[index]


class SequenceTraderProvider:
    def __init__(self, *decisions: dict[str, Any]) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def decide(self, input_packet: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        index = min(self.calls, len(self.decisions) - 1)
        self.calls += 1
        return dict(self.decisions[index])


class LedgerCycleClient(FakeCycleClient):
    def __init__(self, snapshot: BrokerSnapshot) -> None:
        super().__init__(snapshot)
        self.transaction_sync_calls: list[str] = []

    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary:
        return AccountSummary(
            nav_jpy=200_000.0,
            balance_jpy=200_000.0,
            margin_available_jpy=200_000.0,
            last_transaction_id="100",
            fetched_at_utc=now_utc or datetime.now(timezone.utc),
        )

    def transactions_since_id(self, transaction_id: str) -> dict[str, Any]:
        self.transaction_sync_calls.append(str(transaction_id))
        if not self.orders_sent:
            return {"lastTransactionID": transaction_id, "transactions": []}
        return {
            "lastTransactionID": "101",
            "transactions": [
                {
                    "id": "101",
                    "time": "2026-05-06T00:00:02.000000000Z",
                    "type": "ORDER_FILL",
                    "orderID": "1",
                    "instrument": "EUR_USD",
                    "units": "1000",
                    "price": "1.17306",
                    "reason": "MARKET_ORDER",
                    "tradeOpened": {"tradeID": "200", "units": "1000", "price": "1.17306"},
                }
            ],
        }


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
                "target_trades_per_day_source": "cli",
                "status": "PURSUE_TARGET",
                "remaining_target_jpy": 10_000,
            }
        )
    )
    return path


def _pair_charts(root: Path, pairs: tuple[str, ...] = ("EUR_USD",)) -> Path:
    path = root / "pair_charts.json"
    charts = []
    for pair in pairs:
        atr_pips = 3.0 if pair.endswith("_USD") else 5.0
        charts.append(
            {
                "pair": pair,
                "dominant_regime": "TREND_UP",
                "long_score": 0.80,
                "short_score": 0.20,
                "session": {"current_tag": "NY_AM_KILLZONE"},
                "views": [
                    {
                        "granularity": "M5",
                        "regime": "TREND_UP",
                        "indicators": {"atr_pips": atr_pips},
                        "regime_reading": {"confidence": 0.75, "atr_percentile": 0.40},
                    },
                    {
                        "granularity": "M15",
                        "regime": "TREND_UP",
                        "indicators": {"atr_pips": atr_pips * 1.5},
                        "regime_reading": {"confidence": 0.75, "atr_percentile": 0.40},
                    },
                    {
                        "granularity": "H1",
                        "regime": "TREND_UP",
                        "indicators": {"atr_pips": atr_pips * 3.0},
                        "regime_reading": {"confidence": 0.75, "atr_percentile": 0.40},
                    },
                ],
            }
        )
    path.write_text(json.dumps({"charts": charts}) + "\n")
    return path


def _entry_invalidation_pair_charts(root: Path) -> Path:
    path = root / "pair_charts_entry_invalidation.json"
    adverse_indicators = {
        "atr_pips": 1.0,
        "rsi_14": 39.0,
        "macd_hist": -0.0001,
        "supertrend_dir": -1,
        "ichimoku_cloud_pos": -1,
        "plus_di_14": 13.0,
        "minus_di_14": 24.0,
    }
    path.write_text(
        json.dumps(
            {
                "charts": [
                    {
                        "pair": "EUR_USD",
                        "chart_story": (
                            "M5(TREND_DOWN,ADX=22,ST=-,struct=NONE) "
                            "M15(TREND_DOWN,ADX=23,ST=-,struct=NONE) "
                            "M30(TREND_DOWN,ADX=21,ST=-,struct=NONE) "
                            "H1(TREND_DOWN,ADX=24,ST=-,struct=BOS_DOWN@1.3439) "
                            "H4(TREND_DOWN,ADX=25,ST=-,struct=BOS_DOWN@1.3439)"
                        ),
                        "session": {"current_tag": "LONDON_KILLZONE"},
                        "views": [
                            {
                                "granularity": "M5",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                            },
                            {
                                "granularity": "M15",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                            },
                            {
                                "granularity": "M30",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                            },
                            {
                                "granularity": "H1",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                                "structure": {
                                    "structure_events": [
                                        {"kind": "BOS_DOWN", "close_confirmed": True},
                                    ]
                                },
                            },
                            {
                                "granularity": "H4",
                                "regime": "TREND_DOWN",
                                "indicators": dict(adverse_indicators),
                                "structure": {
                                    "structure_events": [
                                        {"kind": "BOS_DOWN", "close_confirmed": True},
                                    ]
                                },
                            },
                        ],
                    }
                ]
            }
        )
        + "\n"
    )
    return path


def _campaign(root: Path) -> Path:
    _pair_charts(root)
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


def _short_hedge_campaign(root: Path) -> Path:
    _pair_charts(root)
    path = root / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "desk": "trend_trader",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "method": "TREND_CONTINUATION",
                        "adoption": "ORDER_INTENT_REQUIRED",
                        "campaign_role": "NOW_OR_BACKUP",
                        "reason": "recovery hedge for trapped EUR_USD long exposure",
                        "required_receipt": "live-ready recovery hedge receipt",
                        "blockers": [],
                        "story_examples": [
                            "quality_audit: EUR_USD downside hedge recovery",
                        ],
                    }
                ]
            }
        )
    )
    return path


def _two_lane_campaign(root: Path) -> Path:
    _pair_charts(root, ("AUD_JPY", "EUR_USD"))
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


def _short_candidate_profile(root: Path) -> Path:
    path = root / "profile.json"
    path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "status": "CANDIDATE",
                        "required_fix": "eligible recovery hedge",
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


def _short_stories(root: Path) -> Path:
    path = root / "stories.json"
    path.write_text(
        json.dumps(
            {
                "pair_profiles": [
                    {
                        "pair": "EUR_USD",
                        "methods": {"TREND_CONTINUATION": 20},
                        "themes": {"recovery_hedge": 6},
                        "examples": [
                            "quality_audit: EUR_USD downside hedge recovery",
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


def _write_no_live_ready_intents(path: Path) -> None:
    _write_live_ready_intents(path)
    payload = json.loads(path.read_text())
    for result in payload.get("results", []):
        result["status"] = "DRY_RUN_BLOCKED"
        result["live_blockers"] = ["forecast no longer backs this pending entry"]
    path.write_text(json.dumps(payload) + "\n")


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


def _write_recovery_hedge_intents(path: Path, lane_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "lane_id": lane_id,
                        "status": "LIVE_READY",
                        "risk_allowed": True,
                        "intent": {
                            "pair": "EUR_USD",
                            "side": "SHORT",
                            "order_type": "STOP-ENTRY",
                            "units": 22_000,
                            "entry": 1.16056,
                            "tp": 1.15813,
                            "sl": 1.16118,
                            "thesis": "Recovery hedge for trapped EUR_USD long exposure.",
                            "reason": "same-pair opposite hedge is live-ready",
                            "owner": "trader",
                            "market_context": {
                                "regime": "TREND_CONTINUATION recovery hedge",
                                "narrative": "EUR_USD downside continuation monetizes trapped long exposure.",
                                "chart_story": "M5 trend continuation below support.",
                                "method": "TREND_CONTINUATION",
                                "invalidation": "stop level trades",
                                "event_risk": "none",
                                "session": "test",
                            },
                            "metadata": {
                                "desk": "trend_trader",
                                "campaign_role": "NOW_OR_BACKUP",
                                "position_intent": "HEDGE",
                                "hedge_recovery": True,
                                "hedge_timing_class": "REVERSAL",
                                "hedge_unwind_plan_required": True,
                                "hedge_review_trigger": "h1_close_or_reversal_structure_failure",
                                "hedge_recovery_reason": "opposing_same_pair_underwater",
                                "hedge_recovery_units": 22_000,
                                "hedge_recovery_size_scale": 1.0,
                                "estimated_margin_jpy": 0.0,
                                "max_loss_jpy": 10_000,
                            },
                        },
                        "risk_metrics": {
                            "risk_jpy": 2_174,
                            "reward_jpy": 8_520,
                            "reward_risk": 3.92,
                            "spread_pips": 0.8,
                            "estimated_margin_jpy": 0.0,
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


def _gpt_trade_decision(
    *,
    lane_id: str = "trend_trader:EUR_USD:LONG:TREND_CONTINUATION",
    pair: str = "EUR_USD",
    method: str = "TREND_CONTINUATION",
    direction: str = "LONG",
) -> dict:
    return {
        "generated_at_utc": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
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
            f"strategy:{pair}:{direction}",
            f"story:{pair}",
            f"chart:{pair}:M5",
            f"chart:{pair}:M15",
        ],
        "twenty_minute_plan": _gpt_twenty_minute_plan(lane_ids=[lane_id], pair=pair),
        "operator_summary": "Accept the verified EUR_USD continuation lane.",
    }


def _gpt_batch_trade_decision(lane_ids: list[str]) -> dict:
    decision = _gpt_trade_decision(lane_id=lane_ids[0])
    decision["selected_lane_ids"] = lane_ids
    refs = list(decision["evidence_refs"])
    for lane_id in lane_ids[1:]:
        refs.extend([f"intent:{lane_id}", f"campaign:{lane_id}"])
    decision["evidence_refs"] = refs
    decision["twenty_minute_plan"] = _gpt_twenty_minute_plan(lane_ids=lane_ids)
    decision["operator_summary"] = "Accept the verified EUR_USD continuation basket."
    return decision


def _gpt_twenty_minute_plan(*, lane_ids: list[str] | None = None, pair: str = "EUR_USD") -> dict:
    refs = [f"chart:{pair}:M5", f"chart:{pair}:M15"]
    for lane_id in lane_ids or []:
        refs.append(f"intent:{lane_id}")
    return {
        "horizon_minutes": 20,
        "primary_path": f"{pair} should hold the operating shelf and follow through toward the selected trigger before the next cycle.",
        "failure_path": "A close through the shelf or a newly named deterministic blocker makes the decision wrong.",
        "entry_or_hold_trigger": "Use only the current LIVE_READY intent trigger; do not chase if price leaves the planned structure.",
        "invalidation_or_cancel_trigger": "Cancel or wait if the invalidation structure breaks or the selected lane leaves LIVE_READY.",
        "counterargument": "The move can still fade on M15; the receipt wins only while the cited M5/M15 structure remains intact.",
        "next_cycle_check": "Refresh broker truth, selected lane status, and M5/M15 structure before extending or replacing the thesis.",
        "evidence_refs": refs,
    }


def _gpt_close_decision(trade_ids: list[str]) -> dict:
    return {
        "generated_at_utc": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        "action": "CLOSE",
        "selected_lane_id": None,
        "selected_lane_ids": [],
        "cancel_order_ids": [],
        "close_trade_ids": trade_ids,
        "confidence": "HIGH",
        "thesis": "The open trader position has printed close-confirmed invalidation against its side.",
        "method": "POSITION_MANAGEMENT",
        "narrative": "Close the broken position before refreshing broker truth for any new entry.",
        "chart_story": "M15/H4 structure invalidates the active thesis.",
        "invalidation": "Do not close if the named trade is no longer open or Gate A/B does not pass.",
        "rejected_alternatives": ["WAIT rejected because the active thesis is invalidated."],
        "risk_notes": ["CLOSE reduces current exposure; any re-entry needs a fresh receipt."],
        "evidence_refs": [
            "broker:snapshot",
            "target:daily",
            "story:EUR_USD",
            *(f"position:thesis:{trade_id}" for trade_id in trade_ids),
        ],
        "operator_summary": "Close the invalidated trader-owned EUR_USD position.",
    }


def _write_learning_audit_artifacts(root: Path, *, attack_advice_path: Path, lane_id: str) -> None:
    attack_advice_path.write_text(
        json.dumps(
            {
                "status": "ATTACK_PARTIAL",
                "read_only": True,
                "live_permission": False,
                "recommended_now_lane_ids": [lane_id],
                "lanes": [
                    {
                        "lane_id": lane_id,
                        "score": 44.0,
                        "learning_influences": ["ai_backtest_research_positive_edge"],
                        "learning_score_delta": 8.0,
                        "learning_influence_details": [
                            {
                                "influence": "ai_backtest_research_positive_edge",
                                "source": "ai_backtest",
                                "reason": "profitable research edge, reduced weight",
                                "score_delta": 8.0,
                            }
                        ],
                    }
                ],
            }
        )
    )
    (root / "ai_test_bot_backtest.json").write_text(
        json.dumps(
            {
                "status": "RESEARCH_PROFITABLE_NOT_CERTIFIED",
                "read_only": True,
                "live_permission": False,
                "blockers": [],
                "summary": {
                    "selected_trades": 30,
                    "total_managed_net_jpy": 1_200.0,
                    "profit_factor": 1.2,
                },
            }
        )
    )
    (root / "outcome_mart.json").write_text(
        json.dumps(
            {
                "read_only": True,
                "live_permission": False,
                "condition_validation": {"validated_outcomes": 30},
            }
        )
    )
    (root / "post_trade_learning.json").write_text(
        json.dumps(
            {
                "status": "NO_UPDATES",
                "blockers": [],
                "profile_update_candidates": [],
            }
        )
    )


def _gpt_wait_decision() -> dict:
    return {
        "generated_at_utc": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
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
            "chart:EUR_USD:M5",
        ],
        "twenty_minute_plan": _gpt_twenty_minute_plan(
            lane_ids=["trend_trader:EUR_USD:LONG:TREND_CONTINUATION:MARKET"]
        ),
        "operator_summary": "Wait even though a live-ready lane exists.",
    }


def _gpt_cancel_pending_decision(cancel_order_ids: list[str]) -> dict:
    return {
        "generated_at_utc": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        "action": "CANCEL_PENDING",
        "selected_lane_id": None,
        "cancel_order_ids": cancel_order_ids,
        "confidence": "HIGH",
        "thesis": "The pending entry is stale relative to current broker truth and should be cleared before new risk.",
        "method": "POSITION_MANAGEMENT",
        "narrative": "A pending order blocks clean discretionary comparison.",
        "chart_story": "The original trigger has drifted away from the current executable lane.",
        "invalidation": "Do not cancel if the order id is not present in current broker truth.",
        "rejected_alternatives": ["TRADE rejected until pending exposure is resolved."],
        "risk_notes": ["Canceling a pending entry reduces possible future exposure."],
        "evidence_refs": ["broker:snapshot", "target:daily"],
        "operator_summary": "Clear the stale pending order before considering another entry.",
    }


class MarginAwareBasketTest(unittest.TestCase):
    """Coverage for C-4 margin-aware basket truncation (2026-05-12).

    `_basket_lane_plan` and `_expanded_gpt_basket_plan` track cumulative
    `LaneScore.estimated_margin_jpy` and stop adding lanes once
    effective margin room × `MARGIN_AWARE_BASKET_BUFFER` would be
    breached. This prevents the gateway from rejecting every basket
    candidate at staging when margin is already tight (2026-05-12 14:00
    UTC scenario: 12-lane basket, 40k margin headroom, all 12 rejected
    with `BASKET_MARGIN_UTILIZATION_CAP_EXCEEDED`).
    """

    @staticmethod
    def _score(lane_id: str, *, pair: str, direction: str, score: float,
               margin_jpy: float | None, action: str = "SEND_ENTRY",
               size_multiple: float = 1.0):
        from quant_rabbit.strategy.trader_brain import LaneScore
        return LaneScore(
            lane_id=lane_id,
            pair=pair,
            direction=direction,
            method="TREND_CONTINUATION",
            order_type="MARKET",
            entry=1.0,
            tp=1.01,
            sl=None,
            status="LIVE_READY",
            score=score,
            action=action,
            blockers=(),
            rationale=(),
            size_multiple=size_multiple,
            estimated_margin_jpy=margin_jpy,
        )

    def _decision(self, scores: list):
        from quant_rabbit.strategy.trader_brain import TraderDecision
        return TraderDecision(
            action="SEND_ENTRY",
            selected_lane_id=scores[0].lane_id if scores else None,
            generated_at_utc="2026-05-12T00:00:00+00:00",
            reason="test",
            scores=tuple(scores),
            positions=0,
            orders=0,
            loss_cap_jpy=None,
            loss_cap_source=None,
            pending_cancel_order_ids=(),
            selected_lane_score=scores[0].score if scores else None,
            selected_lane_size_multiple=scores[0].size_multiple if scores else None,
        )

    def test_basket_truncates_when_cumulative_margin_exceeds_budget(self) -> None:
        from quant_rabbit.automation import AutoTradeCycle
        scores = [
            self._score("a:EUR_USD:SHORT:TREND_CONTINUATION", pair="EUR_USD",
                        direction="SHORT", score=200, margin_jpy=13000),
            self._score("b:EUR_USD:SHORT:BREAKOUT_FAILURE", pair="EUR_USD",
                        direction="SHORT", score=190, margin_jpy=13000),
            self._score("c:GBP_USD:SHORT:TREND_CONTINUATION", pair="GBP_USD",
                        direction="SHORT", score=180, margin_jpy=13000),
            self._score("d:AUD_JPY:SHORT:TREND_CONTINUATION", pair="AUD_JPY",
                        direction="SHORT", score=170, margin_jpy=13000),
        ]
        decision = self._decision(scores)
        lane_ids, _ = AutoTradeCycle._basket_lane_plan(
            decision=decision,
            primary_lane_id=None,
            primary_size_multiple=None,
            margin_available_jpy=40000.0,
        )
        # 40000 * 0.9 = 36000 budget. 13000 + 13000 = 26000 fits; next
        # add would push to 39000 > 36000 → truncate at 2 lanes.
        self.assertEqual(len(lane_ids), 2)
        self.assertEqual(
            lane_ids,
            ("a:EUR_USD:SHORT:TREND_CONTINUATION", "b:EUR_USD:SHORT:BREAKOUT_FAILURE"),
        )

    def test_basket_uses_effective_margin_room_before_raw_margin_available(self) -> None:
        from quant_rabbit.automation import AutoTradeCycle
        scores = [
            self._score("a:EUR_USD:LONG:TREND_CONTINUATION", pair="EUR_USD",
                        direction="LONG", score=200, margin_jpy=7400),
            self._score("b:GBP_USD:LONG:TREND_CONTINUATION", pair="GBP_USD",
                        direction="LONG", score=190, margin_jpy=8500),
        ]
        decision = self._decision(scores)
        lane_ids, _ = AutoTradeCycle._expanded_gpt_basket_plan(
            decision=decision,
            gpt_lane_ids=(scores[0].lane_id, scores[1].lane_id),
            margin_room_jpy=9300.0,
            margin_available_jpy=24600.0,
        )
        # The fixed live path passes RiskEngine-equivalent margin room.
        # 9300 * 0.9 = 8370, so only the first 7400 JPY lane fits even
        # though raw marginAvailable would have admitted both.
        self.assertEqual(lane_ids, (scores[0].lane_id,))

    def test_expanded_gpt_basket_does_not_append_unselected_prefilter_lanes(self) -> None:
        from quant_rabbit.automation import AutoTradeCycle
        scores = [
            self._score("a:EUR_USD:LONG:TREND_CONTINUATION", pair="EUR_USD",
                        direction="LONG", score=200, margin_jpy=7400),
            self._score("b:GBP_USD:SHORT:TREND_CONTINUATION", pair="GBP_USD",
                        direction="SHORT", score=190, margin_jpy=8500),
        ]
        decision = self._decision(scores)

        lane_ids, size_multiples = AutoTradeCycle._expanded_gpt_basket_plan(
            decision=decision,
            gpt_lane_ids=(scores[0].lane_id,),
            margin_available_jpy=40000.0,
        )

        self.assertEqual(lane_ids, (scores[0].lane_id,))
        self.assertEqual(size_multiples, {scores[0].lane_id: 1.0})

    def test_basket_unchanged_when_margin_available_not_passed(self) -> None:
        # Backwards-compat: callers that don't supply margin_available_jpy
        # get the legacy "fit everything" behavior. Protects existing
        # tests and ad-hoc smoke runs.
        from quant_rabbit.automation import AutoTradeCycle
        scores = [
            self._score(f"l{i}:P{i}_USD:LONG:TREND_CONTINUATION", pair=f"P{i}_USD",
                        direction="LONG", score=200 - i, margin_jpy=99999.0)
            for i in range(4)
        ]
        decision = self._decision(scores)
        lane_ids, _ = AutoTradeCycle._basket_lane_plan(
            decision=decision,
            primary_lane_id=None,
            primary_size_multiple=None,
        )
        self.assertEqual(len(lane_ids), 4)

    def test_basket_skips_lane_lacking_margin_but_continues_evaluating(self) -> None:
        # Lanes without estimated_margin_jpy are admitted without
        # counting (legacy fixtures may not supply the field).
        # Subsequent lanes still respect cumulative margin.
        from quant_rabbit.automation import AutoTradeCycle
        scores = [
            self._score("a:EUR_USD:SHORT:TREND_CONTINUATION", pair="EUR_USD",
                        direction="SHORT", score=200, margin_jpy=13000),
            self._score("b:EUR_USD:SHORT:BREAKOUT_FAILURE", pair="EUR_USD",
                        direction="SHORT", score=190, margin_jpy=None),
            self._score("c:GBP_USD:SHORT:TREND_CONTINUATION", pair="GBP_USD",
                        direction="SHORT", score=180, margin_jpy=13000),
        ]
        decision = self._decision(scores)
        lane_ids, _ = AutoTradeCycle._basket_lane_plan(
            decision=decision,
            primary_lane_id=None,
            primary_size_multiple=None,
            margin_available_jpy=20000.0,
        )
        # 20000 * 0.9 = 18000 budget. a (13000) → cumulative 13000. b
        # (None margin) → admitted without count. c (13000) → would push
        # known cumulative to 26000 > 18000 → reject.
        self.assertEqual(lane_ids[:2], (
            "a:EUR_USD:SHORT:TREND_CONTINUATION",
            "b:EUR_USD:SHORT:BREAKOUT_FAILURE",
        ))
        self.assertNotIn("c:GBP_USD:SHORT:TREND_CONTINUATION", lane_ids)

    def test_margin_buffer_is_documented_engineering_value(self) -> None:
        # Guard against silent buffer drift. The documented engineering
        # value is 0.9 = 10% drift headroom — changing it is an explicit
        # operator decision.
        from quant_rabbit.automation import MARGIN_AWARE_BASKET_BUFFER
        self.assertEqual(MARGIN_AWARE_BASKET_BUFFER, 0.9)


if __name__ == "__main__":
    unittest.main()
