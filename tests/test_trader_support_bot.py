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
from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    TP_PROGRESS_REPAIR_REPLAY_FIELD,
)
from quant_rabbit.trader_support_bot import (
    STATUS_BLOCKED,
    STATUS_READY,
    TraderSupportBot,
    _acceptance_clearance_for_code,
)


class TraderSupportBotTest(unittest.TestCase):
    def test_acceptance_plan_breaks_loop_when_tp_progress_repair_is_not_deployed(self) -> None:
        condition, command, summary = _acceptance_clearance_for_code(
            "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
            {
                "guardian_profit_capture_inactive": True,
                "loss_closes_profit_capture_missed": 2,
                "loss_closes_repair_replay_triggered": 1,
                "repair_replay_contract": TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                "top_repair_replay_triggers": [{"trade_id": "472792"}],
                "clearance_condition": "guardian active then replay clean",
            },
            {},
        )

        self.assertIn("position guardian is proven active", condition)
        self.assertEqual(command, "scripts/install-position-guardian.sh --status")
        self.assertTrue(summary["guardian_profit_capture_inactive"])
        self.assertEqual(summary["example_trade_ids"], ["472792"])

    def test_acceptance_plan_requires_month_scale_loss_close_replay(self) -> None:
        condition, command, summary = _acceptance_clearance_for_code(
            "MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED",
            {
                "take_profit_net_jpy": 48804.0,
                "market_close_net_jpy": -81147.0,
                "window_lookback_hours": 168.0,
                "required_lookback_hours": 720.0,
                "repair_replay_contract": TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                "repair_replay_contract_present": True,
            },
            {},
        )

        self.assertIn("at least 720 hours", condition)
        self.assertIn("--lookback-hours 744", command)
        self.assertEqual(summary["take_profit_net_jpy"], 48804.0)
        self.assertEqual(summary["market_close_net_jpy"], -81147.0)
        self.assertEqual(summary["window_lookback_hours"], 168.0)

    def test_acceptance_plan_exposes_month_scale_residual_loss_groups(self) -> None:
        residual_groups = [
            {
                "pair": "GBP_USD",
                "side": "LONG",
                "method": "BREAKOUT_FAILURE",
                "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                "loss_closes": 1,
                "repair_replay_pl_jpy": -2981.8961,
                "block_reasons": {"BELOW_TP_PROGRESS_GATE": 1},
            }
        ]
        method_rollups = [
            {
                "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
                "method": "BREAKOUT_FAILURE",
                "pair_count": 2,
                "pairs": ["EUR_GBP", "GBP_USD"],
                "side_count": 2,
                "sides": ["LONG", "SHORT"],
                "loss_closes": 2,
                "repair_replay_pl_jpy": -3872.9794,
            }
        ]
        condition, command, summary = _acceptance_clearance_for_code(
            "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
            {
                "window_lookback_hours": 744.0,
                "loss_closes_profit_capture_missed": 14,
                "loss_closes_repair_replay_triggered": 13,
                "repair_replay_counterfactual_pl_jpy": -13824.5957,
                "active_counterfactual_profit_capture_pl_jpy": -13824.5957,
                "counterfactual_profit_capture_delta_jpy": 18775.1646,
                "top_repair_replay_residual_groups": residual_groups,
                "top_repair_replay_residual_method_rollups": method_rollups,
                "top_entry_quality_residual_method_rollups": method_rollups,
            },
            {},
        )

        self.assertIn("month-scale production-gate replay is non-negative", condition)
        self.assertIn("--lookback-hours 744", command)
        self.assertEqual(summary["repair_replay_counterfactual_pl_jpy"], -13824.596)
        self.assertEqual(summary["top_repair_replay_residual_groups"], residual_groups)
        self.assertEqual(summary["top_repair_replay_residual_method_rollups"], method_rollups)
        self.assertEqual(summary["top_entry_quality_residual_method_rollups"], method_rollups)

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
            self.assertEqual(payload["metrics"]["profit_capture_counterfactual_delta_jpy"], 1054.02)
            self.assertEqual(payload["metrics"]["profit_capture_repair_replay_triggered"], 1)
            self.assertTrue(payload["metrics"]["profit_capture_repair_replay_contract_present"])
            self.assertEqual(payload["metrics"]["profit_capture_repair_replay_delta_jpy"], 466.2)
            self.assertEqual(payload["profit_capture"]["top_misses"][0]["profit_capture_counterfactual_jpy"], 105.84)
            self.assertIn("zero loss_closes_repair_replay_triggered", payload["profit_capture"]["clearance_condition"])
            self.assertEqual(payload["current_profit_capture"]["watch_positions"], 1)
            self.assertEqual(payload["entry_readiness"]["guardian_blocked_lanes"], 2)
            self.assertEqual(payload["metrics"]["global_unlock_frontier_lanes"], 1)
            unlock = payload["entry_readiness"]["global_unlock_frontier"][0]
            self.assertEqual(unlock["lane_id"], "range_trader:NZD_CAD:LONG:RANGE_ROTATION")
            self.assertEqual(unlock["remaining_blocker_codes_after_global_unlock"], [])
            self.assertTrue(payload["profitability_acceptance"]["target_firepower"]["minimum_5pct_estimated_reachable"])
            self.assertFalse(
                payload["profitability_acceptance"]["target_firepower"][
                    "operational_minimum_5pct_reachable"
                ]
            )
            self.assertIn(
                "POSITION_GUARDIAN_INACTIVE",
                payload["profitability_acceptance"]["target_firepower"]["operational_blocker_codes"],
            )
            self.assertIn(
                "FRESH_ENTRY_SEND_NOT_ALLOWED",
                payload["profitability_acceptance"]["target_firepower"]["operational_blocker_codes"],
            )
            self.assertEqual(payload["profitability_acceptance"]["target_firepower"]["best_bucket"], "high_precision")
            repair_plan = payload["profitability_acceptance"]["repair_plan"]
            self.assertEqual(repair_plan["p0_count"], 3)
            self.assertIn("Rerunning profitability-acceptance alone", repair_plan["loop_breaker"])
            self.assertEqual(payload["metrics"]["acceptance_evidence_collection_count"], 1)
            self.assertEqual(payload["metrics"]["repair_request_count"], 5)
            self.assertEqual(
                payload["metrics"]["repair_request_codes"],
                [
                    "REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE",
                    "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                    "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                    "COLLECT_BIDASK_REPLAY_EVIDENCE",
                    "REPAIR_FRONTIER_LANE_BLOCKER",
                ],
            )
            repair_requests = payload["repair_requests"]
            self.assertEqual(
                [item["code"] for item in repair_requests],
                payload["metrics"]["repair_request_codes"],
            )
            close_gate_request = repair_requests[0]
            self.assertEqual(close_gate_request["status"], "READY_FOR_CODE_REPAIR")
            self.assertEqual(
                close_gate_request["source_findings"],
                ["LOSS_CLOSE_GATE_EVIDENCE_MISSING", "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK"],
            )
            self.assertEqual(close_gate_request["live_side_effects"], [])
            self.assertEqual(
                close_gate_request["automation_contract"]["live_side_effects_allowed"],
                [],
            )
            self.assertIn(
                "position_close",
                close_gate_request["automation_contract"]["requires_explicit_operator_approval_for"],
            )
            self.assertIn(
                "model_api_call_from_quantrabbit_code",
                close_gate_request["automation_contract"]["forbidden_direct_actions"],
            )
            guardian_request = next(
                item for item in repair_requests if item["code"] == "RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT"
            )
            self.assertTrue(guardian_request["requires_explicit_operator_approval"])
            self.assertEqual(guardian_request["status"], "OPERATOR_APPROVAL_REQUIRED")
            self.assertIn("launchd_load", guardian_request["automation_contract"]["requires_explicit_operator_approval_for"])
            evidence_request = next(
                item for item in repair_requests if item["code"] == "COLLECT_BIDASK_REPLAY_EVIDENCE"
            )
            self.assertEqual(evidence_request["status"], "BIDASK_REPLAY_WAITING_FOR_FORECAST_SAMPLE_COVERAGE")
            self.assertNotIn("oanda_history_fetch.py", " ".join(evidence_request["verification_commands"]))
            self.assertFalse(evidence_request["evidence_summary"]["price_truth_fetch_required"])
            self.assertTrue(evidence_request["evidence_summary"]["stale_history_fetch_command_suppressed"])
            self.assertEqual(evidence_request["evidence_summary"]["under_sampled_pair_direction_count"], 48)
            self.assertEqual(evidence_request["evidence_summary"]["under_sampled_missing_evaluated_samples"], 1121)
            self.assertEqual(
                repair_plan["evidence_collection_items"][0]["code"],
                "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE",
            )
            self.assertEqual(
                repair_plan["evidence_collection_items"][0]["evidence_summary"]["rank_only_support_rules"],
                2,
            )
            self.assertEqual(
                repair_plan["evidence_collection_items"][0]["evidence_summary"]["rank_only_examples"][0][
                    "daily_stability_gap"
                ]["missing_positive_days_at_current_requirement"],
                2,
            )
            self.assertEqual(
                [item["code"] for item in repair_plan["items"]],
                [
                    "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                    "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                    "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
                ],
            )
            self.assertEqual(
                repair_plan["items"][1]["evidence_summary"]["example_trade_ids"],
                ["472743"],
            )
            self.assertEqual(
                repair_plan["items"][2]["evidence_summary"]["example_trade_ids"],
                ["472792"],
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
            self.assertNotIn("FETCH_BIDASK_REPLAY_HISTORY", action_codes)
            self.assertNotIn("VALIDATE_BIDASK_REPLAY_HISTORY", action_codes)
            self.assertIn("VERIFY_CLOSE_GATE_EVIDENCE", action_codes)
            self.assertIn("RECHECK_LOSS_CLOSE_LEAK_WINDOW", action_codes)
            self.assertIn("VERIFY_TP_PROGRESS_REPLAY_REPAIR", action_codes)
            self.assertIn("WORK_GLOBAL_UNLOCK_FRONTIER", action_codes)
            self.assertIn("WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS", action_codes)
            self.assertIn("WORK_TARGET_FIREPOWER_BLOCKERS", action_codes)
            self.assertTrue(
                any(item["code"] == "LOAD_POSITION_GUARDIAN_ONLY_IF_APPROVED" and item["requires_explicit_operator_approval"]
                    for item in payload["operator_actions"])
            )
            report = files["report"].read_text()
            self.assertIn("Trader Support Bot Report", report)
            self.assertIn("Counterfactual profit-capture delta JPY", report)
            self.assertIn("Repair Requests", report)
            self.assertIn("REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE", report)
            self.assertIn("RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT", report)
            self.assertIn("Acceptance Repair Plan", report)
            self.assertIn("Acceptance Evidence Collection", report)
            self.assertIn("BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE", report)
            self.assertIn("Repair Frontier Blockers After Support", report)
            self.assertIn("FORECAST_CONTEXT_REQUIRED_FOR_LIVE", report)
            self.assertIn("472792", report)

    def test_close_gate_block_evidence_does_not_request_persistence_repair(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            _write_json(
                files["profitability"],
                {
                    "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                    "blockers": [
                        "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING: 1 close has BLOCK evidence",
                    ],
                    "findings": [
                        {
                            "priority": "P0",
                            "code": "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING",
                            "message": "1 recent GPT loss-side market close has durable close_gate_evidence, but no PASS close_gate_evidence",
                            "next_action": "wait for clean window or future PASS evidence",
                            "evidence": {
                                "recent_close_gate_unverified_loss_closes": 1,
                                "recent_close_gate_unverified_loss_net_jpy": -1380.8,
                                "recent_close_gate_not_passing_loss_closes": 1,
                                "recent_close_gate_not_passing_loss_net_jpy": -1380.8,
                                "recent_close_gate_missing_loss_closes": 0,
                                "examples": [
                                    {
                                        "trade_id": "472743",
                                        "pair": "NZD_USD",
                                        "side": "LONG",
                                        "ts_utc": (now - timedelta(days=2)).isoformat(),
                                        "has_close_gate_evidence": True,
                                        "has_passing_close_gate_evidence": False,
                                    }
                                ],
                            },
                        }
                    ],
                    "metrics": {
                        "capture_economics": {},
                        "oanda_campaign_firepower": {},
                    },
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
        codes = payload["metrics"]["repair_request_codes"]
        self.assertNotIn("REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE", codes)
        self.assertIn("REVIEW_CLOSE_GATE_EVIDENCE_FAILURES", codes)
        request = next(
            item
            for item in payload["repair_requests"]
            if item["code"] == "REVIEW_CLOSE_GATE_EVIDENCE_FAILURES"
        )
        self.assertEqual(request["status"], "HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE")
        self.assertEqual(request["source_findings"], ["LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING"])
        self.assertEqual(request["evidence_summary"]["not_passing_close_gate_evidence"], 1)
        self.assertEqual(request["evidence_summary"]["missing_close_gate_evidence"], 0)

    def test_month_scale_residual_repair_waits_when_current_intents_are_already_blocked(self) -> None:
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
                                "MONTH_SCALE_ENTRY_QUALITY_RESIDUAL_BLOCKED",
                            ],
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "RANGE_ROTATION"},
                                "metadata": {
                                    "month_scale_residual_loss_group": {
                                        "pair": "EUR_USD",
                                        "side": "LONG",
                                        "method": "RANGE_ROTATION",
                                        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
                                        "repair_replay_pl_jpy": -2333.8215,
                                    }
                                },
                            },
                        }
                    ],
                },
            )
            _write_json(
                files["profitability"],
                {
                    "status": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                    "blockers": [
                        "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE: replay remains negative",
                    ],
                    "findings": [
                        {
                            "priority": "P0",
                            "code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                            "message": "month-scale replay remains negative after TP-progress repair",
                            "next_action": "wait for replay evidence after residual blocks are active",
                            "evidence": {
                                "window_lookback_hours": 744.0,
                                "loss_closes_profit_capture_missed": 14,
                                "loss_closes_repair_replay_triggered": 13,
                                "repair_replay_counterfactual_pl_jpy": -13824.5957,
                                "top_repair_replay_residual_groups": [
                                    {
                                        "pair": "EUR_USD",
                                        "side": "LONG",
                                        "method": "RANGE_ROTATION",
                                        "residual_scope": "ENTRY_QUALITY_OR_CLOSE_RESIDUAL",
                                        "repair_replay_pl_jpy": -2333.8215,
                                    }
                                ],
                            },
                        }
                    ],
                    "metrics": {
                        "capture_economics": {},
                        "oanda_campaign_firepower": {},
                    },
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
        self.assertEqual(payload["metrics"]["month_scale_residual_blocked_intent_count"], 1)
        self.assertEqual(payload["entry_readiness"]["month_scale_residual_blocked_intent_count"], 1)
        request = next(
            item
            for item in payload["repair_requests"]
            if item["code"] == "REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY"
        )
        self.assertEqual(
            request["status"],
            "RESIDUAL_GROUPS_ALREADY_BLOCKED_WAITING_FOR_REPLAY",
        )
        block_status = request["evidence_summary"]["current_residual_block_status"]
        self.assertEqual(block_status["current_residual_blocked_intents_count"], 1)
        self.assertEqual(block_status["top_residual_groups_with_current_blocked_intent"], 1)
        self.assertEqual(
            block_status["status"],
            "CURRENT_INTENTS_BLOCK_RESIDUAL_GROUPS_WAIT_FOR_744H_REPLAY",
        )

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
            self.assertEqual(payload["metrics"]["repair_request_count"], 0)
            self.assertEqual(payload["repair_requests"], [])
            self.assertFalse(payload["metrics"]["repair_basket_send_allowed"])
            self.assertEqual(payload["entry_readiness"]["live_ready_lanes"], 1)
            self.assertIn("RUN_NEXT_TRADER_CYCLE", {item["code"] for item in payload["operator_actions"]})

    def test_bidask_partial_price_truth_evidence_exposes_exact_fetch_command(self) -> None:
        now = datetime(2026, 6, 22, 12, 15, tzinfo=timezone.utc)
        fetch_command = (
            "PYTHONPATH=src python3 scripts/oanda_history_fetch.py "
            "--pairs AUD_USD,USD_JPY --granularities S5 --price BA "
            "--from 2026-06-01T00:00:00Z --to 2026-06-02T00:00:00Z "
            "--output-dir logs/replay/oanda_history"
        )
        replay_command = (
            "PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py "
            "--forecast-history data/forecast_history.jsonl --granularity S5 "
            "--auto-history-min-days 30"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            _write_json(
                files["profitability"],
                {
                    "status": "PROFITABILITY_ACCEPTANCE_ACTION_REQUIRED",
                    "blockers": [],
                    "findings": [
                        {
                            "priority": "P1",
                            "code": "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL",
                            "message": "S5 bid/ask replay rules still have partial OANDA price-truth coverage",
                            "next_action": "fetch missing OANDA BA windows and rerun replay validation",
                            "evidence": {},
                        }
                    ],
                    "metrics": {
                        "capture_economics": {},
                        "bidask_replay_rules": {
                            "support_rules": 1,
                            "daily_stable_support_rules": 1,
                            "rank_only_support_rules": 0,
                            "edge_rules": 0,
                            "daily_stable_edge_rules": 0,
                            "rank_only_edge_rules": 0,
                            "contrarian_edge_rules": 1,
                            "daily_stable_contrarian_edge_rules": 1,
                            "rank_only_contrarian_edge_rules": 0,
                            "negative_rules": 0,
                            "price_truth_coverage": {
                                "status": "PARTIAL_PRICE_TRUTH",
                                "adoption_level": "PAIR_LOCAL_RANK_ONLY",
                                "evaluated_rows": 18502,
                                "missing_price_truth_samples": 21847,
                                "missing_price_window_group_count": 26,
                                "history_fetch_command_count": 26,
                                "history_fetch_command_mode": "WINDOWED",
                                "missing_pair_directions": ["AUD_USD:UP", "USD_JPY:DOWN"],
                            },
                            "daily_stability_requirements": {
                                "min_active_days": 3,
                                "max_daily_sample_share": 0.7,
                                "min_positive_day_rate": 2.0 / 3.0,
                            },
                            "history_fetch_command": fetch_command,
                            "replay_validation_command": replay_command,
                            "rank_only_examples": [],
                        },
                    },
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
            repair_plan = payload["profitability_acceptance"]["repair_plan"]
            self.assertEqual(payload["metrics"]["acceptance_evidence_collection_count"], 1)
            self.assertEqual(
                repair_plan["evidence_collection_items"][0]["code"],
                "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL",
            )
            evidence_summary = repair_plan["evidence_collection_items"][0]["evidence_summary"]
            self.assertEqual(evidence_summary["history_fetch_command"], fetch_command)
            self.assertEqual(evidence_summary["history_fetch_command_count"], 26)
            self.assertEqual(evidence_summary["history_fetch_command_mode"], "WINDOWED")
            self.assertEqual(evidence_summary["missing_price_window_group_count"], 26)
            action_by_code = {item["code"]: item for item in payload["operator_actions"]}
            self.assertEqual(action_by_code["FETCH_BIDASK_REPLAY_HISTORY"]["command"], fetch_command)
            self.assertEqual(action_by_code["VALIDATE_BIDASK_REPLAY_HISTORY"]["command"], replay_command)

    def test_forecast_frontier_waits_for_live_precision_evidence(self) -> None:
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
                            "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                                "EXHAUSTION_RANGE_CHASE",
                                "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                            ],
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "sizing_actual_reward_jpy": 1640.0,
                                    "sizing_actual_risk_jpy": 520.0,
                                    "forecast_direction": "UNCLEAR",
                                    "forecast_confidence": 0.046,
                                    "forecast_horizon_min": 0,
                                    "forecast_market_support_ok": False,
                                    "forecast_market_support_reason": (
                                        "forecast UNCLEAR has no executable direction; audited projection unselected"
                                    ),
                                    "forecast_market_support": {
                                        "ok": False,
                                        "reason": (
                                            "forecast UNCLEAR has no executable direction; audited projection unselected"
                                        ),
                                        "unselected_projection_count": 1,
                                        "best_unselected_hit_rate": 0.919,
                                        "best_unselected_samples": 74,
                                        "unselected_signals": [
                                            {
                                                "name": "macro_event_nowcast_inflation",
                                                "direction": "DOWN",
                                                "live_precision_ok": False,
                                                "lead_time_min": 3208.0,
                                                "hit_rate": 0.919,
                                                "calibration_samples": 74,
                                                "confidence": 0.79,
                                            }
                                        ],
                                    },
                                },
                            },
                            "risk_metrics": {"reward_jpy": 1640.0, "risk_jpy": 520.0},
                        }
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
            top = payload["metrics"]["repair_frontier_after_support_top_blockers"][0]
            request = next(
                item for item in payload["repair_requests"] if item["code"] == "REPAIR_FRONTIER_LANE_BLOCKER"
            )

            self.assertEqual(top["code"], "FORECAST_NOT_EXECUTABLE_FOR_LIVE")
            support = top["forecast_support_examples"][0]["forecast_support"]
            self.assertEqual(support["forecast_direction"], "UNCLEAR")
            self.assertFalse(support["top_unselected_signal"]["live_precision_ok"])
            self.assertEqual(
                request["status"],
                "FORECAST_FRONTIER_WAITING_FOR_LIVE_PRECISION_EVIDENCE",
            )
            self.assertIn(
                "oanda_history_replay_validate.py",
                " ".join(request["verification_commands"]),
            )

    def test_forecast_confidence_frontier_with_no_current_projection_waits_for_evidence(self) -> None:
        now = datetime(2026, 6, 23, 14, 5, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = _write_fixture(root, now=now, blocked=True)
            _write_json(
                files["intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                                "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                            ],
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "order_type": "STOP-ENTRY",
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "sizing_actual_reward_jpy": 1378.988,
                                    "sizing_actual_risk_jpy": 391.684,
                                    "forecast_direction": "DOWN",
                                    "forecast_confidence": 0.33,
                                    "forecast_horizon_min": 1440,
                                    "forecast_market_support_ok": False,
                                    "forecast_market_support_reason": (
                                        "no current projection clears audited support floors"
                                    ),
                                    "forecast_market_support": {
                                        "ok": False,
                                        "reason": "no current projection clears audited support floors",
                                        "unselected_projection_count": 0,
                                        "best_unselected_samples": 0,
                                        "unselected_signals": [],
                                    },
                                },
                            },
                            "risk_metrics": {"reward_jpy": 1378.988, "risk_jpy": 391.684},
                        }
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
            request = next(
                item for item in payload["repair_requests"] if item["code"] == "REPAIR_FRONTIER_LANE_BLOCKER"
            )

            self.assertEqual(
                request["status"],
                "FORECAST_FRONTIER_WAITING_FOR_LIVE_PRECISION_EVIDENCE",
            )
            self.assertEqual(request["source_findings"], ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"])
            reason = request["evidence_summary"]["forecast_support_examples"][0]["forecast_support"][
                "forecast_market_support_reason"
            ]
            self.assertIn("no current projection", reason)
            self.assertIn(
                "oanda_history_replay_validate.py",
                " ".join(request["verification_commands"]),
            )

    def test_frontier_reward_risk_guardrail_is_not_code_repair(self) -> None:
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
                            "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
                                "REWARD_RISK_TOO_LOW",
                                "BREAKOUT_FAILURE_STOP_CHASES_FAILED_SIDE",
                                "PATTERN_REVERSAL_CHASE",
                            ],
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "BREAKOUT_FAILURE"},
                                "metadata": {
                                    "self_improvement_p0_repair_live_ready": True,
                                    "self_improvement_p0_repair_mode": "TP_HARVEST_REPAIR",
                                    "sizing_actual_reward_jpy": 692.0,
                                    "sizing_actual_risk_jpy": 820.0,
                                },
                            },
                            "risk_metrics": {"reward_jpy": 692.0, "risk_jpy": 820.0},
                        }
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
            request = next(
                item for item in payload["repair_requests"] if item["code"] == "REPAIR_FRONTIER_LANE_BLOCKER"
            )

        self.assertEqual(request["status"], "FRONTIER_PROTECTIVE_GUARDRAIL_ACTIVE")
        self.assertIn("bad entry shape", request["why_now"])
        self.assertIn("Do not edit common entry gates", " ".join(request["clearance_conditions"]))

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
                                    "positive_rotation_mode": "TP_PROVEN_HARVEST",
                                    "positive_rotation_pessimistic_expectancy_jpy": 327.2788,
                                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                                    "capture_take_profit_scope_key": (
                                        "GBP_JPY|SHORT|RANGE_ROTATION|TAKE_PROFIT_ORDER"
                                    ),
                                    "capture_take_profit_trades": 20,
                                    "capture_take_profit_wins": 20,
                                    "capture_take_profit_losses": 0,
                                    "capture_take_profit_expectancy_jpy": 591.5,
                                },
                            },
                            "risk_metrics": {"reward_jpy": 1362.0, "risk_jpy": 1362.0},
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
            self.assertEqual(candidate["reward_jpy"], 1362.0)
            self.assertEqual(candidate["risk_jpy"], 1362.0)
            self.assertEqual(candidate["remaining_blocker_codes_after_guardian_and_repair_exemption"], [])
            self.assertEqual(candidate["tp_proof"]["capture_take_profit_trades"], 20)
            self.assertEqual(candidate["tp_proof"]["capture_take_profit_losses"], 0)
            self.assertEqual(candidate["tp_proof"]["capture_take_profit_expectancy_jpy"], 591.5)
            self.assertEqual(
                candidate["tp_proof"]["positive_rotation_pessimistic_expectancy_jpy"],
                327.2788,
            )
            unlock = payload["entry_readiness"]["global_unlock_frontier"][0]
            self.assertEqual(unlock["tp_proof"]["positive_rotation_mode"], "TP_PROVEN_HARVEST")
            report = files["report"].read_text()
            self.assertIn("Guardian Recovery Candidates", report)
            self.assertIn("trades=20 losses=0", report)
            self.assertIn("pess_exp_jpy=327.279", report)

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
            oanda_rotation = root / "data" / "oanda_rotation.json"
            _write_json(
                oanda_rotation,
                {
                    "campaign_firepower": {
                        "high_precision": {
                            "top_vehicles": [
                                {
                                    "vehicle_key": "GBP_JPY|LONG|range_reversion|tp1_sl1",
                                    "evidence_status": "HIGH_PRECISION_VALIDATED",
                                    "pair": "GBP_JPY",
                                    "side": "LONG",
                                    "validation_n": 15,
                                    "validation_win_rate": 0.8,
                                    "validation_win_wilson95_lower": 0.548141,
                                    "validation_profit_factor": 4.129446,
                                    "validation_avg_realized_pips": 4.2,
                                    "validation_expectancy_r": 0.6,
                                    "active_days": 6,
                                    "positive_day_rate": 0.833333,
                                    "estimated_return_pct_per_active_day_at_observed_frequency": 1.53987,
                                    "trades_needed_for_minimum_5pct": 9,
                                    "trades_needed_for_target_10pct": 18,
                                }
                            ]
                        },
                        "evidence_queue": {"top_vehicles": []},
                    }
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
                    oanda_rotation_mining_path=oanda_rotation,
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

    def test_oanda_audit_only_candidate_requires_local_tp_proof(self) -> None:
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
                            "lane_id": "range_trader:GBP_JPY:LONG:RANGE_ROTATION",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                                "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                            ],
                            "intent": {
                                "pair": "GBP_JPY",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "RANGE_ROTATION"},
                                "metadata": {
                                    "positive_rotation_oanda_campaign_audit_only": True,
                                    "positive_rotation_oanda_campaign_local_tp_proof_required": True,
                                    "positive_rotation_oanda_campaign_matching_vehicle_key": (
                                        "GBP_JPY|LONG|range_reversion|tp1_sl1"
                                    ),
                                    "capture_take_profit_scope": "MISSING_METHOD_SCOPE",
                                    "capture_take_profit_scope_key": (
                                        "GBP_JPY|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER"
                                    ),
                                    "sizing_actual_reward_jpy": 1020.0,
                                    "sizing_actual_risk_jpy": 820.0,
                                },
                            },
                        }
                    ],
                },
            )
            oanda_rotation = root / "data" / "oanda_rotation.json"
            _write_json(
                oanda_rotation,
                {
                    "campaign_firepower": {
                        "high_precision": {
                            "top_vehicles": [
                                {
                                    "vehicle_key": "GBP_JPY|LONG|range_reversion|tp1_sl1",
                                    "evidence_status": "HIGH_PRECISION_VALIDATED",
                                    "pair": "GBP_JPY",
                                    "side": "LONG",
                                    "validation_n": 15,
                                    "validation_win_rate": 0.8,
                                    "validation_win_wilson95_lower": 0.548141,
                                    "validation_profit_factor": 4.129446,
                                    "validation_avg_realized_pips": 4.2,
                                    "validation_expectancy_r": 0.6,
                                    "active_days": 6,
                                    "positive_day_rate": 0.833333,
                                    "estimated_return_pct_per_active_day_at_observed_frequency": 1.53987,
                                    "trades_needed_for_minimum_5pct": 9,
                                    "trades_needed_for_target_10pct": 18,
                                }
                            ]
                        },
                        "evidence_queue": {"top_vehicles": []},
                    }
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
                    oanda_rotation_mining_path=oanda_rotation,
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            payload = json.loads(files["output"].read_text())
            self.assertEqual(payload["metrics"]["oanda_audit_only_local_tp_proof_required_lanes"], 1)
            self.assertEqual(
                payload["metrics"]["oanda_audit_only_local_tp_proof_required_lane_ids"],
                ["range_trader:GBP_JPY:LONG:RANGE_ROTATION"],
            )
            candidate = payload["entry_readiness"]["oanda_audit_only_local_tp_proof_required"][0]
            self.assertEqual(candidate["capture_take_profit_scope"], "MISSING_METHOD_SCOPE")
            self.assertEqual(candidate["oanda_vehicle_key"], "GBP_JPY|LONG|range_reversion|tp1_sl1")
            self.assertEqual(candidate["oanda_replay_evidence_status"], "HIGH_PRECISION_VALIDATED")
            self.assertFalse(candidate["oanda_replay_live_permission"])
            self.assertEqual(candidate["oanda_replay_evidence"]["validation_n"], 15)
            self.assertEqual(candidate["oanda_replay_evidence"]["validation_win_rate"], 0.8)
            self.assertEqual(candidate["oanda_replay_evidence"]["validation_profit_factor"], 4.129446)
            self.assertEqual(candidate["oanda_replay_evidence"]["trades_needed_for_minimum_5pct"], 9)
            self.assertFalse(candidate["historical_replay_can_clear_local_tp_proof"])
            self.assertIn(
                "PAIR_SIDE_METHOD TAKE_PROFIT_ORDER",
                candidate["local_tp_proof_clearance_condition"],
            )
            self.assertEqual(payload["metrics"]["oanda_audit_only_with_replay_evidence_lanes"], 1)
            self.assertEqual(
                candidate["remaining_blocker_codes_after_guardian"],
                [
                    "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
                    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                ],
            )
            action_codes = {item["code"] for item in payload["operator_actions"]}
            self.assertIn("MINE_LOCAL_TP_PROOF_FOR_OANDA_AUDIT_ONLY", action_codes)
            self.assertIn("VALIDATE_OANDA_AUDIT_ONLY_BIDASK_REPLAY", action_codes)
            self.assertIn("MINE_OANDA_AUDIT_ONLY_CAMPAIGN_FIREPOWER", action_codes)
            self.assertIn("PACKAGE_OANDA_AUDIT_ONLY_FIREPOWER_RULES_AFTER_REVIEW", action_codes)
            self.assertIn("RERUN_INTENTS_AFTER_OANDA_AUDIT_ONLY_REPLAY", action_codes)
            action_by_code = {item["code"]: item for item in payload["operator_actions"]}
            self.assertIn(
                "--granularities S5,M5",
                action_by_code["MINE_LOCAL_TP_PROOF_FOR_OANDA_AUDIT_ONLY"]["command"],
            )
            self.assertIn(
                "scripts/oanda_history_replay_validate.py",
                action_by_code["VALIDATE_OANDA_AUDIT_ONLY_BIDASK_REPLAY"]["command"],
            )
            self.assertIn(
                "scripts/oanda_universal_rotation_miner.py",
                action_by_code["MINE_OANDA_AUDIT_ONLY_CAMPAIGN_FIREPOWER"]["command"],
            )
            self.assertIn(
                "--pairs GBP_JPY",
                action_by_code["MINE_OANDA_AUDIT_ONLY_CAMPAIGN_FIREPOWER"]["command"],
            )
            self.assertFalse(
                action_by_code["PACKAGE_OANDA_AUDIT_ONLY_FIREPOWER_RULES_AFTER_REVIEW"][
                    "requires_explicit_operator_approval"
                ]
            )
            self.assertIn(
                "test, commit, and sync",
                action_by_code["PACKAGE_OANDA_AUDIT_ONLY_FIREPOWER_RULES_AFTER_REVIEW"]["reason"],
            )
            report = files["report"].read_text()
            self.assertIn("OANDA Audit-Only Local TP Proof Required", report)
            self.assertIn("Historical replay can rank and mine these candidates", report)
            self.assertIn("MISSING_METHOD_SCOPE", report)
            self.assertIn("n=15", report)
            self.assertIn("pf=4.129446", report)
            self.assertIn("5%trades=9", report)

    def test_oanda_audit_only_candidate_reads_preserved_packaged_runtime_artifact(self) -> None:
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
                            "lane_id": "range_trader:AUD_USD:LONG:RANGE_ROTATION",
                            "status": "DRY_RUN_BLOCKED",
                            "live_blocker_codes": [
                                "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
                                "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
                            ],
                            "intent": {
                                "pair": "AUD_USD",
                                "side": "LONG",
                                "order_type": "LIMIT",
                                "market_context": {"method": "RANGE_ROTATION"},
                                "metadata": {
                                    "positive_rotation_oanda_campaign_audit_only": True,
                                    "positive_rotation_oanda_campaign_matching_vehicle_key": (
                                        "AUD_USD|LONG|range_reversion|tp1_sl1"
                                    ),
                                    "capture_take_profit_scope": "MISSING_METHOD_SCOPE",
                                    "capture_take_profit_scope_key": (
                                        "AUD_USD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER"
                                    ),
                                },
                            },
                        }
                    ],
                },
            )
            latest = (
                root
                / "logs"
                / "reports"
                / "forecast_improvement"
                / "oanda_universal_rotation_mining_latest.json"
            )
            packaged = root / "src" / "quant_rabbit" / "oanda_universal_rotation_precision_rules.json"
            latest.parent.mkdir(parents=True, exist_ok=True)
            packaged.parent.mkdir(parents=True, exist_ok=True)
            _write_json(
                latest,
                {
                    "generated_at_utc": "2026-06-23T10:09:16Z",
                    "campaign_firepower": {
                        "high_precision": {"top_vehicles": []},
                        "evidence_queue": {"top_vehicles": []},
                    },
                },
            )
            _write_json(
                packaged,
                {
                    "generated_at_utc": "2026-06-23T10:09:16Z",
                    "source_report": (
                        "logs/reports/forecast_improvement/"
                        "oanda_universal_rotation_mining_latest.json"
                    ),
                    "campaign_firepower_preserved_from_existing": True,
                    "campaign_firepower": {
                        "high_precision": {
                            "top_vehicles": [
                                {
                                    "vehicle_key": "AUD_USD|LONG|range_reversion|tp1_sl1",
                                    "evidence_status": "HIGH_PRECISION_VALIDATED",
                                    "pair": "AUD_USD",
                                    "side": "LONG",
                                    "validation_n": 21,
                                    "validation_win_rate": 0.714286,
                                    "validation_profit_factor": 3.04137,
                                    "validation_avg_realized_pips": 1.997279,
                                    "validation_expectancy_r": 0.465228,
                                    "active_days": 12,
                                    "positive_day_rate": 0.583333,
                                    "estimated_return_pct_per_active_day_at_observed_frequency": 0.814149,
                                    "trades_needed_for_minimum_5pct": 11,
                                    "trades_needed_for_target_10pct": 22,
                                }
                            ]
                        },
                        "evidence_queue": {"top_vehicles": []},
                    },
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
                    oanda_rotation_mining_path=latest,
                    oanda_rotation_packaged_path=packaged,
                    output_path=files["output"],
                    report_path=files["report"],
                    now_utc=now,
                ).run()

            payload = json.loads(files["output"].read_text())
            candidate = payload["entry_readiness"]["oanda_audit_only_local_tp_proof_required"][0]

            self.assertEqual(payload["artifact_paths"]["oanda_rotation_effective"], str(packaged))
            self.assertEqual(candidate["oanda_vehicle_key"], "AUD_USD|LONG|range_reversion|tp1_sl1")
            self.assertEqual(candidate["oanda_replay_evidence_status"], "HIGH_PRECISION_VALIDATED")
            self.assertEqual(candidate["oanda_replay_evidence"]["validation_n"], 21)
            self.assertEqual(payload["metrics"]["oanda_audit_only_with_replay_evidence_lanes"], 1)

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
                    "loss_close_actual_pl_jpy": -5188.197,
                    "loss_close_counterfactual_profit_capture_pl_jpy": -4134.177,
                    "loss_close_counterfactual_profit_capture_delta_jpy": 1054.02,
                    "loss_close_counterfactual_profit_capture_jpy": 474.341,
                    "loss_closes_repair_replay_triggered": 1,
                    "loss_close_repair_replay_delta_jpy": 466.2,
                    "loss_close_repair_replay_profit_capture_jpy": 126.0,
                    "top_profit_capture_misses": [
                        {
                            "trade_id": "472792",
                            "pair": "USD_JPY",
                            "profit_capture_counterfactual_jpy": 105.84,
                            "profit_capture_counterfactual_net_improvement_jpy": 446.04,
                        }
                    ],
                    "top_repair_replay_triggers": [
                        {
                            "trade_id": "472792",
                            "pair": "USD_JPY",
                            "side": "SHORT",
                            "repair_counterfactual_jpy": 126.0,
                            "repair_counterfactual_delta_jpy": 466.2,
                        }
                    ],
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
            {
                "priority": "P0",
                "code": "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
                "message": "2 loss close(s) have OANDA candle replay evidence for TP-progress capture",
                "next_action": "prove TP-progress capture repair clean",
                "evidence": {
                    "loss_closes_profit_capture_missed": 2,
                    "loss_closes_repair_replay_triggered": 1,
                    "counterfactual_profit_capture_delta_jpy": 1054.02,
                    "counterfactual_profit_capture_jpy": 474.341,
                    "clearance_condition": (
                        "execution-timing-audit must report zero loss_closes_repair_replay_triggered"
                    ),
                    "top_profit_capture_misses": [
                        {
                            "trade_id": "472792",
                            "pair": "USD_JPY",
                            "side": "SHORT",
                            "counterfactual_jpy": 105.84,
                            "counterfactual_delta_jpy": 446.04,
                        }
                    ],
                    "top_repair_replay_triggers": [
                        {
                            "trade_id": "472792",
                            "pair": "USD_JPY",
                            "side": "SHORT",
                            "repair_counterfactual_jpy": 126.0,
                            "repair_counterfactual_delta_jpy": 466.2,
                        }
                    ],
                },
            },
            {
                "priority": "P1",
                "code": "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE",
                "message": "2 S5 bid/ask replay support rules exist but remain rank-only",
                "next_action": "fetch more OANDA BA candles and rerun bid/ask replay",
                "evidence": {
                    "support_rules": 2,
                    "daily_stable_support_rules": 0,
                    "rank_only_support_rules": 2,
                    "edge_rules": 1,
                    "daily_stable_edge_rules": 0,
                    "rank_only_edge_rules": 1,
                    "contrarian_edge_rules": 1,
                    "daily_stable_contrarian_edge_rules": 0,
                    "rank_only_contrarian_edge_rules": 1,
                    "negative_rules": 1,
                    "price_truth_coverage": {
                        "status": "PRICE_TRUTH_OK",
                        "adoption_level": "FULL_REPLAY_READY",
                        "evaluated_rows": 650,
                        "missing_price_truth_samples": 0,
                        "under_sampled_pair_direction_count": 48,
                        "under_sampled_missing_evaluated_samples": 1121,
                    },
                    "daily_stability_requirements": {
                        "min_active_days": 3,
                        "max_daily_sample_share": 0.7,
                        "min_positive_day_rate": 2.0 / 3.0,
                    },
                    "history_fetch_command": (
                        "python3 scripts/oanda_history_fetch.py --pairs AUD_JPY --granularities S5 "
                        "--price BA --days 120 --output-dir logs/replay/oanda_history"
                    ),
                    "replay_validation_command": (
                        "python3 scripts/oanda_history_replay_validate.py "
                        "--forecast-history data/forecast_history.jsonl "
                        "--granularity S5 "
                        "--auto-history-min-days 30 --stable-min-active-days 3 "
                        "--stable-max-daily-sample-share 0.7 "
                        "--stable-min-positive-day-rate 0.6666666667"
                    ),
                    "rank_only_examples": [
                        {
                            "name": "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
                            "pair": "EUR_USD",
                            "granularity": "S5",
                            "forecast_direction": None,
                            "direction": "DOWN",
                            "samples": 226,
                            "active_days": 5,
                            "positive_day_rate": 0.4,
                            "daily_stability_status": "DAILY_SAMPLE_CONCENTRATED",
                            "optimized_profit_factor": 3.34,
                            "daily_stability_gap": {
                                "reasons": [
                                    "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION",
                                    "NEEDS_HIGHER_POSITIVE_DAY_RATE",
                                ],
                                "missing_active_days": 0,
                                "missing_positive_days_at_current_requirement": 2,
                            },
                        },
                        {
                            "name": "AUD_JPY_UP_FADE_TO_DOWN_RANK_ONLY",
                            "pair": "AUD_JPY",
                            "granularity": "S5",
                            "forecast_direction": "UP",
                            "direction": "DOWN",
                            "samples": 40,
                            "active_days": 2,
                            "positive_day_rate": 0.5,
                            "daily_stability_status": "INSUFFICIENT_ACTIVE_DAYS",
                            "optimized_profit_factor": 2.31,
                            "daily_stability_gap": {
                                "reasons": [
                                    "NEEDS_MORE_ACTIVE_DAYS",
                                    "NEEDS_DAILY_STABILITY_CONFIRMATION",
                                ],
                                "missing_active_days": 1,
                                "missing_positive_days_at_current_requirement": 1,
                            },
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
                "bidask_replay_rules": {
                    "support_rules": 2,
                    "daily_stable_support_rules": 0,
                    "rank_only_support_rules": 2,
                    "edge_rules": 1,
                    "daily_stable_edge_rules": 0,
                    "rank_only_edge_rules": 1,
                    "contrarian_edge_rules": 1,
                    "daily_stable_contrarian_edge_rules": 0,
                    "rank_only_contrarian_edge_rules": 1,
                    "negative_rules": 1,
                    "price_truth_coverage": {
                        "status": "PRICE_TRUTH_OK",
                        "adoption_level": "FULL_REPLAY_READY",
                        "evaluated_rows": 650,
                        "missing_price_truth_samples": 0,
                        "under_sampled_pair_direction_count": 48,
                        "under_sampled_missing_evaluated_samples": 1121,
                    },
                    "daily_stability_requirements": {
                        "min_active_days": 3,
                        "max_daily_sample_share": 0.7,
                        "min_positive_day_rate": 2.0 / 3.0,
                    },
                    "history_fetch_command": (
                        "python3 scripts/oanda_history_fetch.py --pairs AUD_JPY --granularities S5 "
                        "--price BA --days 120 --output-dir logs/replay/oanda_history"
                    ),
                    "replay_validation_command": (
                        "python3 scripts/oanda_history_replay_validate.py "
                        "--forecast-history data/forecast_history.jsonl "
                        "--granularity S5 "
                        "--auto-history-min-days 30 --stable-min-active-days 3 "
                        "--stable-max-daily-sample-share 0.7 "
                        "--stable-min-positive-day-rate 0.6666666667"
                    ),
                    "rank_only_examples": [
                        {
                            "name": "EUR_USD_DOWN_S5_BIDASK_HARVEST_TP5_SL7",
                            "pair": "EUR_USD",
                            "granularity": "S5",
                            "forecast_direction": None,
                            "direction": "DOWN",
                            "samples": 226,
                            "active_days": 5,
                            "positive_day_rate": 0.4,
                            "daily_stability_status": "DAILY_SAMPLE_CONCENTRATED",
                            "daily_stability_gap": {
                                "reasons": [
                                    "NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION",
                                    "NEEDS_HIGHER_POSITIVE_DAY_RATE",
                                ],
                                "missing_active_days": 0,
                                "missing_positive_days_at_current_requirement": 2,
                            },
                        },
                        {
                            "name": "AUD_JPY_UP_FADE_TO_DOWN_RANK_ONLY",
                            "pair": "AUD_JPY",
                            "granularity": "S5",
                            "forecast_direction": "UP",
                            "direction": "DOWN",
                            "samples": 40,
                            "active_days": 2,
                            "positive_day_rate": 0.5,
                            "daily_stability_status": "INSUFFICIENT_ACTIVE_DAYS",
                            "daily_stability_gap": {
                                "reasons": [
                                    "NEEDS_MORE_ACTIVE_DAYS",
                                    "NEEDS_DAILY_STABILITY_CONFIRMATION",
                                ],
                                "missing_active_days": 1,
                                "missing_positive_days_at_current_requirement": 1,
                            },
                        }
                    ],
                },
            },
        },
    )
    _write_json(
        files["timing"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "OK",
            "precision": {
                TP_PROGRESS_REPAIR_REPLAY_FIELD: TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
            },
            "summary": {
                "loss_closes_profit_capture_missed": 2 if blocked else 0,
                "loss_closes_repair_replay_triggered": 1 if blocked else 0,
                "loss_close_repair_replay_delta_jpy": 466.2 if blocked else 0.0,
                "loss_close_repair_replay_profit_capture_jpy": 126.0 if blocked else 0.0,
            },
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
                "historical_repair_replay_triggered": 1 if blocked else 0,
                "historical_repair_replay_delta_jpy": 466.2 if blocked else 0.0,
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
