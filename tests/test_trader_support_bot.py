from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from quant_rabbit.cli import main
from quant_rabbit.trader_support_bot import STATUS_BLOCKED, STATUS_READY, TraderSupportBot


class TraderSupportBotTest(unittest.TestCase):
    def test_blocks_when_guardian_is_inactive_and_profit_capture_was_missed(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            env = _guardian_env(root, active="0")
            with mock.patch.dict(os.environ, env, clear=False):
                summary = TraderSupportBot(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    position_management_path=files["position_management"],
                    position_guardian_management_path=files["guardian_management"],
                    position_guardian_execution_path=files["guardian_execution"],
                    position_guardian_heartbeat_path=files["guardian_heartbeat"],
                    self_improvement_audit_path=files["self_improvement"],
                    profitability_acceptance_path=files["profitability"],
                    execution_timing_audit_path=files["timing"],
                    profit_capture_bot_path=files["profit_capture_bot"],
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            self.assertEqual(summary.status, STATUS_BLOCKED)
            payload = json.loads(files["output"].read_text())
            codes = {item["code"] for item in payload["blockers"]}
            self.assertIn("POSITION_GUARDIAN_INACTIVE", codes)
            self.assertIn("LOSS_CLOSE_PROFIT_CAPTURE_MISSED", codes)
            self.assertFalse(payload["metrics"]["send_fresh_entries_allowed"])
            self.assertFalse(payload["metrics"]["repair_basket_send_allowed"])
            self.assertEqual(payload["metrics"]["repair_basket_guardian_recovery_lanes"], 0)
            self.assertEqual(payload["profit_capture"]["missed_loss_closes"], 2)
            self.assertIn("zero loss_closes_profit_capture_missed", payload["profit_capture"]["clearance_condition"])
            self.assertEqual(payload["current_profit_capture"]["watch_positions"], 1)
            self.assertEqual(payload["entry_readiness"]["guardian_blocked_lanes"], 2)
            self.assertEqual(payload["metrics"]["global_unlock_frontier_lanes"], 1)
            unlock = payload["entry_readiness"]["global_unlock_frontier"][0]
            self.assertEqual(unlock["lane_id"], "range_trader:NZD_CAD:LONG:RANGE_ROTATION")
            self.assertEqual(unlock["remaining_blocker_codes_after_global_unlock"], [])
            self.assertTrue(payload["profitability_acceptance"]["target_firepower"]["minimum_5pct_estimated_reachable"])
            self.assertEqual(payload["profitability_acceptance"]["target_firepower"]["best_bucket"], "high_precision")
            repair_plan = payload["profitability_acceptance"]["repair_plan"]
            self.assertEqual(repair_plan["p0_count"], 2)
            self.assertIn("Rerunning profitability-acceptance alone", repair_plan["loop_breaker"])
            self.assertEqual(
                [item["code"] for item in repair_plan["items"]],
                ["RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK", "LOSS_CLOSE_GATE_EVIDENCE_MISSING"],
            )
            self.assertEqual(
                repair_plan["items"][1]["evidence_summary"]["example_trade_ids"],
                ["472743"],
            )
            repair = payload["entry_readiness"]["repair_frontier"][0]
            self.assertEqual(
                repair["remaining_blocker_codes_after_guardian_and_repair_exemption"],
                ["FORECAST_CONTEXT_REQUIRED_FOR_LIVE"],
            )
            self.assertEqual(payload["metrics"]["repair_frontier_after_support_clear_lanes"], 0)
            self.assertEqual(payload["metrics"]["repair_frontier_after_support_blocked_lanes"], 1)
            self.assertEqual(
                payload["metrics"]["repair_frontier_after_support_top_blockers"],
                [
                    {
                        "code": "FORECAST_CONTEXT_REQUIRED_FOR_LIVE",
                        "count": 1,
                        "reward_jpy": 790.5,
                        "example_lane_ids": ["failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT"],
                    }
                ],
            )
            action_codes = {item["code"] for item in payload["operator_actions"]}
            self.assertIn("CHECK_POSITION_GUARDIAN_PREFLIGHT", action_codes)
            self.assertIn("FOLLOW_ACCEPTANCE_REPAIR_PLAN", action_codes)
            self.assertIn("VERIFY_CLOSE_GATE_EVIDENCE", action_codes)
            self.assertIn("RECHECK_LOSS_CLOSE_LEAK_WINDOW", action_codes)
            self.assertIn("WORK_GLOBAL_UNLOCK_FRONTIER", action_codes)
            self.assertIn("WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS", action_codes)
            self.assertIn("WORK_TARGET_FIREPOWER_BLOCKERS", action_codes)
            self.assertTrue(
                any(item["code"] == "LOAD_POSITION_GUARDIAN_ONLY_IF_APPROVED" and item["requires_explicit_operator_approval"]
                    for item in payload["operator_actions"])
            )
            report = files["report"].read_text()
            self.assertIn("Trader Support Bot Report", report)
            self.assertIn("Acceptance Repair Plan", report)
            self.assertIn("Repair Frontier Blockers After Support", report)
            self.assertIn("FORECAST_CONTEXT_REQUIRED_FOR_LIVE", report)
            self.assertIn("472792", report)

    def test_ready_when_guardian_heartbeat_is_fresh_and_live_lane_exists(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=False)
            env = _guardian_env(root, active="1")
            with mock.patch.dict(os.environ, env, clear=False):
                summary = TraderSupportBot(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    position_management_path=files["position_management"],
                    position_guardian_management_path=files["guardian_management"],
                    position_guardian_execution_path=files["guardian_execution"],
                    position_guardian_heartbeat_path=files["guardian_heartbeat"],
                    self_improvement_audit_path=files["self_improvement"],
                    profitability_acceptance_path=files["profitability"],
                    execution_timing_audit_path=files["timing"],
                    profit_capture_bot_path=files["profit_capture_bot"],
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(summary.status, STATUS_READY)
            self.assertTrue(payload["guardian"]["active"])
            self.assertTrue(payload["guardian"]["heartbeat_fresh"])
            self.assertTrue(payload["metrics"]["send_fresh_entries_allowed"])
            self.assertFalse(payload["metrics"]["repair_basket_send_allowed"])
            self.assertEqual(payload["entry_readiness"]["live_ready_lanes"], 1)
            self.assertIn("RUN_NEXT_TRADER_CYCLE", {item["code"] for item in payload["operator_actions"]})

    def test_repair_basket_allowed_even_when_acceptance_panel_is_blocked(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            _write_json(
                files["intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            "lane_id": "range_trader:GBP_JPY:SHORT:RANGE_ROTATION",
                            "status": "LIVE_READY",
                            "live_blocker_codes": [],
                            "intent": {
                                "pair": "GBP_JPY",
                                "side": "SHORT",
                                "order_type": "LIMIT",
                                "market_context": {"method": "RANGE_ROTATION"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "positive_rotation_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
                                    "sizing_actual_reward_jpy": 1152.0,
                                    "sizing_actual_risk_jpy": 1146.0,
                                },
                            },
                        }
                    ],
                },
            )
            env = _guardian_env(root, active="1")
            with mock.patch.dict(os.environ, env, clear=False):
                summary = TraderSupportBot(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    position_management_path=files["position_management"],
                    position_guardian_management_path=files["guardian_management"],
                    position_guardian_execution_path=files["guardian_execution"],
                    position_guardian_heartbeat_path=files["guardian_heartbeat"],
                    self_improvement_audit_path=files["self_improvement"],
                    profitability_acceptance_path=files["profitability"],
                    execution_timing_audit_path=files["timing"],
                    profit_capture_bot_path=files["profit_capture_bot"],
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertFalse(payload["metrics"]["send_fresh_entries_allowed"])
            self.assertTrue(payload["metrics"]["repair_basket_send_allowed"])
            self.assertEqual(payload["metrics"]["repair_live_ready_lanes"], 1)
            self.assertEqual(
                payload["metrics"]["repair_basket_lane_ids"],
                ["range_trader:GBP_JPY:SHORT:RANGE_ROTATION"],
            )
            self.assertEqual(payload["entry_readiness"]["repair_live_ready"][0]["repair_mode"], "TP_HARVEST_REPAIR")
            report = files["report"].read_text()
            self.assertIn("Repair basket send allowed", report)
            self.assertIn("Repair LIVE_READY lanes", report)

    def test_repair_basket_guardian_recovery_candidate_is_visible_before_loading_guardian(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            _write_json(
                files["intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            "lane_id": "range_trader:GBP_JPY:SHORT:RANGE_ROTATION",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": ["POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE"],
                            "intent": {
                                "pair": "GBP_JPY",
                                "side": "SHORT",
                                "order_type": "LIMIT",
                                "market_context": {"method": "RANGE_ROTATION"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "positive_rotation_mode": "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
                                    "sizing_actual_reward_jpy": 1362.0,
                                    "sizing_actual_risk_jpy": 1362.0,
                                },
                            },
                        }
                    ],
                },
            )
            env = _guardian_env(root, active="0")
            with mock.patch.dict(os.environ, env, clear=False):
                summary = TraderSupportBot(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    position_management_path=files["position_management"],
                    position_guardian_management_path=files["guardian_management"],
                    position_guardian_execution_path=files["guardian_execution"],
                    position_guardian_heartbeat_path=files["guardian_heartbeat"],
                    self_improvement_audit_path=files["self_improvement"],
                    profitability_acceptance_path=files["profitability"],
                    execution_timing_audit_path=files["timing"],
                    profit_capture_bot_path=files["profit_capture_bot"],
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(summary.status, STATUS_BLOCKED)
            self.assertFalse(payload["metrics"]["repair_basket_send_allowed"])
            self.assertEqual(payload["metrics"]["repair_basket_guardian_recovery_lanes"], 1)
            self.assertEqual(
                payload["metrics"]["repair_basket_guardian_recovery_lane_ids"],
                ["range_trader:GBP_JPY:SHORT:RANGE_ROTATION"],
            )
            candidate = payload["entry_readiness"]["repair_basket_guardian_recovery"][0]
            self.assertEqual(candidate["remaining_blocker_codes_after_guardian_and_repair_exemption"], [])
            report = files["report"].read_text()
            self.assertIn("Repair lanes after guardian recovery", report)

    def test_range_forecast_non_rotation_repair_is_superseded_by_range_counterpart(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            _write_json(
                files["intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            "lane_id": "range_trader:EUR_USD:LONG:RANGE_ROTATION",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                            ],
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "RANGE_ROTATION"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "sizing_actual_reward_jpy": 900.0,
                                    "sizing_actual_risk_jpy": 300.0,
                                },
                            },
                        },
                        {
                            "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                            ],
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "sizing_actual_reward_jpy": 1400.0,
                                    "sizing_actual_risk_jpy": 350.0,
                                },
                            },
                        },
                    ],
                },
            )
            env = _guardian_env(root, active="0")
            with mock.patch.dict(os.environ, env, clear=False):
                TraderSupportBot(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    position_management_path=files["position_management"],
                    position_guardian_management_path=files["guardian_management"],
                    position_guardian_execution_path=files["guardian_execution"],
                    position_guardian_heartbeat_path=files["guardian_heartbeat"],
                    self_improvement_audit_path=files["self_improvement"],
                    profitability_acceptance_path=files["profitability"],
                    execution_timing_audit_path=files["timing"],
                    profit_capture_bot_path=files["profit_capture_bot"],
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            payload = json.loads(files["output"].read_text())
            frontier_lane_ids = [item["lane_id"] for item in payload["entry_readiness"]["repair_frontier"]]
            remaining_codes = {
                item["code"] for item in payload["metrics"]["repair_frontier_after_support_top_blockers"]
            }
            superseded = payload["entry_readiness"]["repair_frontier_superseded_by_range_forecast"][0]

            self.assertEqual(frontier_lane_ids, ["range_trader:EUR_USD:LONG:RANGE_ROTATION"])
            self.assertNotIn("RANGE_FORECAST_REQUIRES_RANGE_ROTATION", remaining_codes)
            self.assertEqual(payload["metrics"]["repair_frontier_superseded_by_range_forecast_lanes"], 1)
            self.assertEqual(
                superseded["superseded_by_range_rotation_lane_id"],
                "range_trader:EUR_USD:LONG:RANGE_ROTATION",
            )
            report = files["report"].read_text()
            self.assertIn("RANGE Forecast Superseded Repair Lanes", report)
            self.assertIn("failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT", report)

    def test_range_forecast_non_rotation_repair_without_counterpart_is_coverage_gap(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            _write_json(
                files["intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                                "MATRIX_REPAIR_REJECT_CONTEXT",
                            ],
                            "intent": {
                                "pair": "GBP_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "matrix_repair_profile_status": "BLOCK_UNTIL_NEW_EVIDENCE",
                                    "sizing_actual_reward_jpy": 1000.0,
                                    "sizing_actual_risk_jpy": 400.0,
                                },
                            },
                        },
                    ],
                },
            )
            env = _guardian_env(root, active="0")
            with mock.patch.dict(os.environ, env, clear=False):
                TraderSupportBot(
                    broker_snapshot_path=files["broker"],
                    order_intents_path=files["intents"],
                    target_state_path=files["target"],
                    position_management_path=files["position_management"],
                    position_guardian_management_path=files["guardian_management"],
                    position_guardian_execution_path=files["guardian_execution"],
                    position_guardian_heartbeat_path=files["guardian_heartbeat"],
                    self_improvement_audit_path=files["self_improvement"],
                    profitability_acceptance_path=files["profitability"],
                    execution_timing_audit_path=files["timing"],
                    profit_capture_bot_path=files["profit_capture_bot"],
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            payload = json.loads(files["output"].read_text())
            missing = payload["entry_readiness"]["repair_frontier_missing_range_rotation_counterpart"][0]

            self.assertEqual(payload["entry_readiness"]["repair_frontier"], [])
            self.assertEqual(payload["entry_readiness"]["repair_frontier_superseded_by_range_forecast"], [])
            self.assertEqual(payload["metrics"]["repair_frontier_missing_range_rotation_counterpart_lanes"], 1)
            self.assertEqual(missing["missing_range_rotation_counterpart_for"], ["GBP_USD", "LONG"])
            self.assertIn(
                "RANGE_ROTATION_COUNTERPART_MISSING",
                missing["remaining_blocker_codes_after_guardian_and_repair_exemption"],
            )
            report = files["report"].read_text()
            self.assertIn("RANGE Forecast Missing Counterpart Repair Lanes", report)
            self.assertIn("failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT", report)

    def test_cli_writes_support_panel_and_returns_blocked_code(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            env = _guardian_env(root, active="0")
            stdout = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(stdout):
                code = main(
                    [
                        "trader-support-bot",
                        "--broker-snapshot",
                        str(files["broker"]),
                        "--order-intents",
                        str(files["intents"]),
                        "--target-state",
                        str(files["target"]),
                        "--position-management",
                        str(files["position_management"]),
                        "--position-guardian-management",
                        str(files["guardian_management"]),
                        "--position-guardian-execution",
                        str(files["guardian_execution"]),
                        "--position-guardian-heartbeat",
                        str(files["guardian_heartbeat"]),
                        "--self-improvement-audit",
                        str(files["self_improvement"]),
                        "--profitability-acceptance",
                        str(files["profitability"]),
                        "--execution-timing-audit",
                        str(files["timing"]),
                        "--profit-capture-bot",
                        str(files["profit_capture_bot"]),
                        "--output",
                        str(files["output"]),
                        "--report",
                        str(files["report"]),
                    ]
                )

            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], STATUS_BLOCKED)
            self.assertTrue(files["output"].exists())
            self.assertTrue(files["report"].exists())

    def test_cli_returns_error_code_distinct_from_blocked_diagnostic(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            files["broker"].write_text("{not-json", encoding="utf-8")
            env = _guardian_env(root, active="0")
            stdout = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(stdout):
                code = main(
                    [
                        "trader-support-bot",
                        "--broker-snapshot",
                        str(files["broker"]),
                        "--order-intents",
                        str(files["intents"]),
                        "--target-state",
                        str(files["target"]),
                        "--position-management",
                        str(files["position_management"]),
                        "--position-guardian-management",
                        str(files["guardian_management"]),
                        "--position-guardian-execution",
                        str(files["guardian_execution"]),
                        "--position-guardian-heartbeat",
                        str(files["guardian_heartbeat"]),
                        "--self-improvement-audit",
                        str(files["self_improvement"]),
                        "--profitability-acceptance",
                        str(files["profitability"]),
                        "--execution-timing-audit",
                        str(files["timing"]),
                        "--profit-capture-bot",
                        str(files["profit_capture_bot"]),
                        "--output",
                        str(files["output"]),
                        "--report",
                        str(files["report"]),
                    ]
                )

            self.assertEqual(code, 3)
            self.assertIn("error", json.loads(stdout.getvalue()))


def _write_fixture(root: Path, *, now: datetime, blocked: bool) -> dict[str, Path]:
    data = root / "data"
    docs = root / "docs"
    data.mkdir()
    docs.mkdir()
    files = {
        "broker": data / "broker_snapshot.json",
        "intents": data / "order_intents.json",
        "target": data / "daily_target_state.json",
        "position_management": data / "position_management.json",
        "guardian_management": data / "position_guardian_management.json",
        "guardian_execution": data / "position_guardian_execution.json",
        "guardian_heartbeat": data / "position_guardian.json",
        "self_improvement": data / "self_improvement_audit.json",
        "profitability": data / "profitability_acceptance.json",
        "timing": data / "execution_timing_audit.json",
        "profit_capture_bot": data / "profit_capture_bot.json",
        "output": data / "trader_support_bot.json",
        "report": docs / "trader_support_bot_report.md",
    }
    _write_json(
        files["broker"],
        {
            "fetched_at_utc": now.isoformat(),
            "account": {"balance_jpy": 173000.0, "nav_jpy": 172500.0, "margin_available_jpy": 120000.0},
            "positions": [
                {
                    "trade_id": "472792",
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "owner": "trader",
                    "units": 6300,
                    "unrealized_pl_jpy": -120.5,
                    "take_profit": 161.1,
                    "stop_loss": 161.9,
                }
            ],
            "orders": [],
        },
    )
    _write_json(
        files["target"],
        {
            "status": "PURSUE_TARGET",
            "campaign_day_jst": "2026-06-22",
            "remaining_target_jpy": 9000.0,
            "remaining_minimum_jpy": 1000.0,
            "progress_pct": -1.2,
            "minimum_progress_pct": -2.4,
            "target_trades_per_day": 30,
        },
    )
    if blocked:
        results = [
            {
                "lane_id": "range_trader:NZD_CAD:LONG:RANGE_ROTATION",
                "status": "DRY_RUN_BLOCKED",
                "live_blocker_codes": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                    "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                ],
                "intent": {
                    "pair": "NZD_CAD",
                    "side": "LONG",
                    "order_type": "LIMIT",
                    "market_context": {"method": "RANGE_ROTATION"},
                    "metadata": {
                        "sizing_actual_reward_jpy": 910.0,
                        "sizing_actual_risk_jpy": 300.0,
                    },
                },
            },
            {
                "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                "status": "DRY_RUN_BLOCKED",
                "live_blocker_codes": [
                    "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                    "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                    "FORECAST_CONTEXT_REQUIRED_FOR_LIVE",
                ],
                "intent": {
                    "pair": "GBP_USD",
                    "side": "LONG",
                    "order_type": "LIMIT",
                    "market_context": {"method": "BREAKOUT_FAILURE"},
                    "metadata": {
                        "self_improvement_p0_repair_live_ready": True,
                        "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                        "sizing_actual_reward_jpy": 790.5,
                        "sizing_actual_risk_jpy": 260.0,
                    },
                },
            }
        ]
    else:
        results = [
            {
                "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                "status": "LIVE_READY",
                "live_blocker_codes": [],
                "intent": {
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "order_type": "LIMIT",
                    "market_context": {"method": "RANGE_ROTATION"},
                    "metadata": {},
                },
            }
        ]
    _write_json(files["intents"], {"generated_at_utc": now.isoformat(), "results": results})
    _write_json(
        files["position_management"],
        {
            "generated_at_utc": now.isoformat(),
            "action": "HOLD_PROTECTED",
            "positions": [{"trade_id": "472792", "pair": "USD_JPY", "side": "SHORT", "action": "HOLD_PROTECTED"}],
        },
    )
    _write_json(
        files["guardian_management"],
        {
            "generated_at_utc": (now - timedelta(seconds=40)).isoformat(),
            "action": "HOLD_PROTECTED",
            "positions": [{"trade_id": "472792", "pair": "USD_JPY", "side": "SHORT", "action": "HOLD_PROTECTED"}],
        },
    )
    _write_json(
        files["guardian_execution"],
        {"generated_at_utc": (now - timedelta(seconds=30)).isoformat(), "status": "NO_ACTION", "actions": []},
    )
    _write_json(files["guardian_heartbeat"], {"generated_at_utc": (now - timedelta(seconds=30)).isoformat()})
    if blocked:
        findings = [
            {
                "priority": "P0",
                "code": "LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
                "message": "2 losing close(s) had positive TP-progress capture opportunity before closing red",
                "next_action": "repair fast profit capture",
                "evidence": {
                    "loss_closes_profit_capture_missed": 2,
                    "loss_close_estimated_capture_gap_jpy": 646.489,
                    "top_profit_capture_misses": [{"trade_id": "472792", "pair": "USD_JPY"}],
                },
            },
            {
                "priority": "P0",
                "code": "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                "message": "position guardian is required but inactive",
                "next_action": "run guardian preflight",
                "evidence": {"guardian": {"active": False, "required": True}},
            },
        ]
        self_improvement_status = "SELF_IMPROVEMENT_BLOCKED"
        profitability_status = "PROFITABILITY_ACCEPTANCE_BLOCKED"
        profitability_blockers = [
            "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK: 1 loss-side close remains inside the window",
            "LOSS_CLOSE_GATE_EVIDENCE_MISSING: 1 close lacks evidence",
        ]
        profitability_findings = [
            {
                "priority": "P0",
                "code": "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                "message": "1 loss-side gateway market close remains inside the 7-day window",
                "next_action": "recheck timing leak window",
                "evidence": {
                    "recent_leak_loss_closes": 1,
                    "recent_leak_loss_net_jpy": -1380.8,
                    "latest_loss_close_ts_utc": (now - timedelta(days=2)).isoformat(),
                    "recent_loss_timing_label_counts": {"LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE": 1},
                    "examples": [
                        {
                            "trade_id": "472743",
                            "pair": "NZD_USD",
                            "side": "LONG",
                            "ts_utc": (now - timedelta(days=2)).isoformat(),
                        }
                    ],
                },
            },
            {
                "priority": "P0",
                "code": "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                "message": "1 recent GPT loss-side market close lacks passing durable close_gate_evidence",
                "next_action": "persist close gate evidence",
                "evidence": {
                    "recent_close_gate_unverified_loss_closes": 1,
                    "recent_close_gate_unverified_loss_net_jpy": -1380.8,
                    "examples": [
                        {
                            "trade_id": "472743",
                            "pair": "NZD_USD",
                            "side": "LONG",
                            "ts_utc": (now - timedelta(days=2)).isoformat(),
                        }
                    ],
                },
            },
        ]
    else:
        findings = []
        self_improvement_status = "SELF_IMPROVEMENT_OK"
        profitability_status = "PROFITABILITY_ACCEPTANCE_PASSED"
        profitability_blockers = []
        profitability_findings = []
    _write_json(files["self_improvement"], {"status": self_improvement_status, "findings": findings})
    _write_json(
        files["profitability"],
        {
            "status": profitability_status,
            "blockers": profitability_blockers,
            "findings": profitability_findings,
            "metrics": {
                "capture_economics": {},
                "oanda_campaign_firepower": {
                    "status": "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
                    "target_open": True,
                    "minimum_return_pct": 5.0,
                    "target_return_pct": 10.0,
                    "per_trade_risk_pct_lens": 1.0,
                    "high_precision": {
                        "estimated_return_pct_per_active_day_at_observed_frequency": 12.5,
                        "weighted_return_pct_per_trade_at_risk_lens": 0.625,
                        "observed_attempts_per_active_day": 20.0,
                        "trades_needed_for_minimum_5pct_at_weighted_expectancy": 8,
                        "trades_needed_for_target_10pct_at_weighted_expectancy": 16,
                        "pair_count": 4,
                        "unique_vehicle_count": 9,
                        "top_vehicle_keys": ["NZD_CAD|LONG|range_reversion|tp1_sl1"],
                    },
                    "evidence_queue": {
                        "estimated_return_pct_per_active_day_at_observed_frequency": 4.0,
                        "weighted_return_pct_per_trade_at_risk_lens": 0.4,
                        "observed_attempts_per_active_day": 10.0,
                        "trades_needed_for_minimum_5pct_at_weighted_expectancy": 13,
                        "trades_needed_for_target_10pct_at_weighted_expectancy": 25,
                        "pair_count": 2,
                        "unique_vehicle_count": 3,
                        "top_vehicle_keys": ["USD_JPY|LONG|range_reversion|tp1_sl1"],
                    },
                },
            },
        },
    )
    _write_json(
        files["timing"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "OK",
            "summary": {"loss_closes_profit_capture_missed": 2 if blocked else 0},
        },
    )
    _write_json(
        files["profit_capture_bot"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "PROFIT_CAPTURE_BLOCKED" if blocked else "PROFIT_CAPTURE_WATCH",
            "metrics": {
                "open_trader_positions": 1,
                "bankable_positions": 0,
                "watch_positions": 1,
                "blocked_positions": 0,
                "historical_missed_loss_closes": 2 if blocked else 0,
            },
            "positions": [
                {
                    "trade_id": "472792",
                    "pair": "USD_JPY",
                    "side": "SHORT",
                    "gate_status": "WATCH_NOT_PROFITABLE",
                    "tp_progress": None,
                    "capture_trigger": {"quote_side": "ask", "comparator": "<=", "price": 161.674},
                }
            ],
        },
    )
    return files


def _guardian_env(root: Path, *, active: str) -> dict[str, str]:
    return {
        "QR_REQUIRE_POSITION_GUARDIAN_ACTIVE": "1",
        "QR_POSITION_GUARDIAN_ACTIVE": active,
        "QR_POSITION_GUARDIAN_INTERVAL": "30",
        "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS": "120",
        "QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT": "1",
        "QR_POSITION_GUARDIAN_PLIST": str(root / "com.quantrabbit.position-guardian.plist"),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
