from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from quant_rabbit.trader_goal_loop_orchestrator import TraderGoalLoopOrchestrator


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class TraderGoalLoopOrchestratorTest(unittest.TestCase):
    def test_scout_blocked_operator_review_selects_operator_review_report_only(self) -> None:
        now = datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_BLOCKED_OPERATOR_REVIEW")
            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report_text = paths["report"].read_text()

        self.assertEqual(summary.selected_next_work_type, "OPERATOR_REVIEW_REPORT")
        self.assertEqual(payload["selected_next_work_type"], "OPERATOR_REVIEW_REPORT")
        self.assertNotIn("requires_operator_approval", payload)
        self.assertFalse(payload["requires_operator_approval_for_this_report"])
        self.assertTrue(payload["requires_operator_review_before_scout_or_routing"])
        self.assertTrue(payload["operator_review_state"]["operator_review_required"])
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertFalse(payload["proof_state"]["proof_queue_count_is_live_permission"])
        self.assertEqual(payload["scout_state"]["status"], "SCOUT_BLOCKED_OPERATOR_REVIEW")
        self.assertFalse(payload["operator_review_state"]["normal_routing_allowed"])
        self.assertTrue(payload["edge_improvement_state"]["candidate_available"])
        self.assertIn("4xに近づく根本改善", payload["selected_next_prompt"])
        self.assertIn("SCOUTを許可/却下するための判断材料", payload["selected_next_prompt"])
        self.assertIn("live permission ではありません", payload["selected_next_prompt"])
        self.assertIn("EUR_USD|SHORT|BREAKOUT_FAILURE", payload["four_x_progress_hypothesis"])
        self.assertIn("live-grade HARVEST", payload["root_improvement_target"])
        self.assertIn("TP proof 17勝 / 0 TP負け", payload["expected_edge_improvement"])
        self.assertIn("Selected next work type: `OPERATOR_REVIEW_REPORT`", report_text)
        self.assertIn("Live permission allowed: `False`", report_text)
        self.assertIn("requires_operator_approval_for_this_report: `False`", report_text)
        self.assertIn("requires_operator_review_before_scout_or_routing: `True`", report_text)
        self.assertIn(
            "このreport生成自体は承認不要。ただしSCOUT/normal routing前にはoperator review必須",
            report_text,
        )

    def test_operator_review_clear_with_zero_proof_queue_routes_to_edge_experiment_not_live_permission(self) -> None:
        now = datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            _write_json(
                paths["board"],
                {
                    "generated_at_utc": now.isoformat(),
                    "normal_routing_status": "READY",
                    "routing_allowed": True,
                    "as_live_ready_path_exists": False,
                    "exact_blocker_preventing_live_ready": {"primary": "PROOF_QUEUE_EMPTY"},
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["scout"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "SCOUT_DIAGNOSIS_COMPLETE",
                    "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                    "scout_mode_allowed": True,
                    "operator_approval_required": False,
                    "max_loss_jpy_cap": 418.0,
                    "min_lot_feasibility": {
                        "status": "MIN_LOT_NUMERICALLY_FEASIBLE_BUT_OTHER_GATES_BLOCK",
                        "feasible_if_all_non_lot_gates_clear": True,
                    },
                    "proof_queue_entry_blockers": [],
                    "live_side_effects": [],
                },
            )
            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "EDGE_IMPROVEMENT_EXPERIMENT")
        self.assertEqual(payload["selected_next_work_type"], "EDGE_IMPROVEMENT_EXPERIMENT")
        self.assertNotIn("requires_operator_approval", payload)
        self.assertFalse(payload["requires_operator_approval_for_this_report"])
        self.assertFalse(payload["requires_operator_review_before_scout_or_routing"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["proof_state"]["proof_queue_count"], 0)
        self.assertIn("entry / exit / payoff / sampling", payload["selected_next_prompt"])
        self.assertIn("NO_TRADE", payload["expected_edge_improvement"])

    def test_scout_blocked_status_wins_even_if_routing_artifact_claims_ready(self) -> None:
        now = datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_BLOCKED_OPERATOR_REVIEW")
            _write_json(
                paths["board"],
                {
                    "generated_at_utc": now.isoformat(),
                    "normal_routing_status": "READY",
                    "routing_allowed": True,
                    "as_live_ready_path_exists": False,
                    "exact_blocker_preventing_live_ready": {"primary": "SCOUT_OPERATOR_REVIEW_STILL_REQUIRED"},
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["scout"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "SCOUT_BLOCKED_OPERATOR_REVIEW",
                    "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                    "scout_mode_allowed": False,
                    "operator_approval_required": True,
                    "max_loss_jpy_cap": 418.0,
                    "min_lot_feasibility": {
                        "status": "MIN_LOT_NUMERICALLY_FEASIBLE_BUT_OTHER_GATES_BLOCK",
                        "feasible_if_all_non_lot_gates_clear": True,
                    },
                    "proof_queue_entry_blockers": [
                        {
                            "code": "CONTRADICTED_ROUTING_ARTIFACT",
                            "evidence": {"normal_routing_allowed": True},
                        }
                    ],
                    "live_side_effects": [],
                },
            )
            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "OPERATOR_REVIEW_REPORT")
        self.assertEqual(payload["selected_next_work_type"], "OPERATOR_REVIEW_REPORT")
        self.assertFalse(payload["requires_operator_approval_for_this_report"])
        self.assertTrue(payload["requires_operator_review_before_scout_or_routing"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertIn("SCOUT_BLOCKED_OPERATOR_REVIEW", payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"])

    def test_raw_guardian_clear_overrides_stale_scout_operator_review_summary(self) -> None:
        now = datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_BLOCKED_OPERATOR_REVIEW")
            _write_json(
                paths["guardian_consumption"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED",
                    "normal_routing_allowed": True,
                    "unresolved_issue_count": 0,
                    "classifications": [],
                },
            )
            _write_json(
                paths["guardian_review"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED",
                    "normal_routing_allowed": True,
                    "unresolved_review_count": 0,
                    "classifications": [],
                },
            )
            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "EDGE_IMPROVEMENT_EXPERIMENT")
        self.assertEqual(payload["selected_next_work_type"], "EDGE_IMPROVEMENT_EXPERIMENT")
        self.assertFalse(payload["requires_operator_review_before_scout_or_routing"])
        self.assertTrue(payload["operator_review_state"]["normal_routing_allowed"])
        self.assertTrue(payload["operator_review_state"]["guardian_clear"])
        self.assertEqual(payload["operator_review_state"]["source"], "raw_guardian_receipt_artifacts")
        self.assertIn(
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
            payload["operator_review_state"]["stale_guardian_blocker_codes_suppressed"],
        )

    def test_active_contract_evidence_prompt_overrides_generic_payoff_work(self) -> None:
        now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "selected_active_path": "EVIDENCE_ACQUISITION",
                    "selected_active_path_reason": "board top plus non-EUR frontier identify the shortest read-only evidence path.",
                    "target_shape": "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET",
                    "four_x_progress_hypothesis": (
                        "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET is the current active path; "
                        "advance lane-local evidence instead of legacy EUR_USD work."
                    ),
                    "root_improvement_target": "Advance USD_CAD|LONG|BREAKOUT_FAILURE|MARKET from the active board.",
                    "expected_edge_improvement": "Expected improvement is USD_CAD active-path proof collection.",
                    "next_prompt": (
                        "Implement EVIDENCE_ACQUISITION for "
                        "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET plus frontier evidence "
                        "USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT as read-only work."
                    ),
                    "next_trade_enabling_action": (
                        "Use the latest board top lane and frontier evidence lane as one "
                        "USD_CAD unblock plan without sending an order."
                    ),
                    "current_state": {
                        "active_opportunity_board": {
                            "top_lane": {
                                "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET",
                                "pair": "USD_CAD",
                                "direction": "LONG",
                                "strategy_family": "BREAKOUT_FAILURE",
                                "vehicle": "MARKET",
                                "status": "EVIDENCE_ACQUISITION",
                            }
                        },
                        "non_eurusd_live_grade_frontier": {
                            "next_evidence_lane": {
                                "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                                "pair": "USD_CAD",
                                "direction": "LONG",
                                "strategy_family": "BREAKOUT_FAILURE",
                                "vehicle": "LIMIT",
                                "status": "EVIDENCE_ACQUISITION",
                            }
                        },
                    },
                    "remaining_blockers": [
                        {"code": "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"},
                        {"code": "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH"},
                    ],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )

            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report_text = paths["report"].read_text()

        self.assertEqual(summary.selected_next_work_type, "ACTIVE_TRADER_CONTRACT_EVIDENCE")
        self.assertEqual(payload["selected_next_work_type"], "ACTIVE_TRADER_CONTRACT_EVIDENCE")
        self.assertEqual(payload["current_phase"], "ACTIVE_CONTRACT_EVIDENCE_ACQUISITION")
        self.assertEqual(payload["success_condition_evaluation"]["status"], "MET")
        self.assertTrue(payload["active_contract_state"]["active_prompt_available"])
        self.assertEqual(
            payload["active_contract_state"]["top_lane_id"],
            "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET",
        )
        self.assertEqual(
            payload["active_contract_state"]["frontier_lane_id"],
            "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
        )
        self.assertIn("active_trader_contract next_prompt", payload["selected_next_prompt"])
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|MARKET", payload["selected_next_prompt"])
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT", payload["selected_next_prompt"])
        self.assertIn("同じ unblock plan", payload["selected_next_prompt"])
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|MARKET", payload["four_x_progress_hypothesis"])
        self.assertIn("Advance USD_CAD|LONG|BREAKOUT_FAILURE|MARKET", payload["root_improvement_target"])
        self.assertIn("USD_CAD active-path proof collection", payload["expected_edge_improvement"])
        self.assertNotIn("EUR_USD|SHORT|BREAKOUT_FAILURE", payload["four_x_progress_hypothesis"])
        self.assertIn("ACTIVE_CONTRACT:EVIDENCE_ACQUISITION", payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"])
        self.assertIn("Active contract prompt available: `True`", report_text)

    def test_repair_orchestrator_waiting_evidence_waits_when_artifacts_are_current(self) -> None:
        now = datetime(2026, 7, 9, 12, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "ORCHESTRATOR_BLOCKED",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 4,
                    "repair_request_count": 4,
                    "read_only": True,
                    "live_side_effects": [],
                    "next_evidence_actions": [
                        {
                            "action_id": "refresh_lane_board_after_input_evidence_changes",
                            "read_only": True,
                            "live_side_effects": [],
                            "success_condition": {
                                "schema_version": "success_condition_v1",
                                "mode": "all",
                                "checks": [
                                    {"field": "proof_normal_routing_status", "operator": "neq", "value": "BLOCKED"},
                                    {"field": "can_create_live_permission_count", "operator": "gt", "value": 0},
                                ],
                            },
                        }
                    ],
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "selected_active_path": "EVIDENCE_ACQUISITION",
                    "selected_active_path_reason": "active board still has a lane-specific evidence prompt.",
                    "target_shape": "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT",
                    "four_x_progress_hypothesis": "Active contract evidence remains available but is not the next loop action.",
                    "root_improvement_target": "Do not repeat active contract while repair orchestrator is waiting for evidence.",
                    "expected_edge_improvement": "Read-only evidence refresh must change the proof state first.",
                    "next_prompt": "Repeat active contract evidence for EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT.",
                    "next_trade_enabling_action": "Collect exact local TP proof; do not send.",
                    "current_state": {
                        "active_opportunity_board": {
                            "top_lane": {
                                "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "strategy_family": "BREAKOUT_FAILURE",
                                "vehicle": "LIMIT",
                                "status": "EVIDENCE_ACQUISITION",
                            }
                        }
                    },
                    "remaining_blockers": [{"code": "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"}],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )

            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "NO_ACTION_WAIT")
        self.assertEqual(payload["selected_next_work_type"], "NO_ACTION_WAIT")
        self.assertEqual(payload["current_phase"], "WAITING_FOR_EVIDENCE_OR_MARKET_TRIGGER")
        self.assertTrue(payload["active_contract_state"]["active_prompt_available"])
        self.assertTrue(payload["repair_loop_state"]["waiting_for_evidence"])
        self.assertEqual(payload["repair_loop_state"]["actionable_request_count"], 0)
        self.assertEqual(payload["repair_loop_state"]["waiting_request_count"], 4)
        self.assertIn("trader_repair_orchestrator reports ORCHESTRATOR_BLOCKED", payload["selection_reason"])
        self.assertIn("artifact health is already clear", payload["selection_reason"])
        self.assertNotIn(
            "REPAIR_ORCHESTRATOR_WAITING_FOR_EVIDENCE",
            payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"],
        )
        self.assertNotIn("trader-repair-orchestrator", "\n".join(payload["next_allowed_commands"]))
        self.assertFalse(payload["live_permission_allowed"])

    def test_pending_active_lane_dispatch_is_preserved_with_exact_commands(self) -> None:
        now = datetime(2026, 7, 10, 0, 55, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            dispatch_id = "a" * 64
            commands = [
                "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                (
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-repair-orchestrator "
                    f"--ack-active-lane-dispatch {dispatch_id}"
                ),
            ]
            command_steps = [
                {"command": commands[0], "required": True, "ok_rcs": [0, 2]},
                {"command": commands[1], "required": True, "ok_rcs": [0]},
            ]
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "NO_REPAIR_REQUESTS",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 0,
                    "repair_request_count": 0,
                    "codex_work_order": {
                        "status": "READ_ONLY_EVIDENCE_WORK",
                        "action_code": "EXACT_TP_PROOF_COLLECTION",
                        "material_digest": dispatch_id,
                        "pending_material_digest": dispatch_id,
                        "pending_dispatch_id": dispatch_id,
                        "dispatch_allowed": True,
                        "execution_pending": True,
                        "new_dispatch_issued": True,
                        "suggested_commands": commands,
                        "suggested_command_steps": command_steps,
                    },
                    "read_only": True,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": (now + timedelta(seconds=5)).isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "selected_active_path": "EVIDENCE_ACQUISITION",
                    "next_prompt": "Do not replace the pending repair dispatch.",
                    "next_trade_enabling_action": "Collect exact local TP proof; do not send.",
                    "current_state": {},
                    "remaining_blockers": [],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "ACTIVE_LANE_EVIDENCE_DISPATCH")
        self.assertTrue(payload["repair_loop_state"]["active_lane_execution_pending"])
        self.assertEqual(payload["next_allowed_commands"], commands)
        self.assertEqual(payload["next_allowed_command_steps"], command_steps)
        self.assertIn(dispatch_id, payload["selected_next_prompt"])
        self.assertIn("末尾の exact-digest acknowledgement", payload["selected_next_prompt"])
        self.assertEqual(payload["success_condition_evaluation"]["status"], "NOT_MET")
        self.assertIn(
            f"ACTIVE_LANE_EVIDENCE_DISPATCH:{dispatch_id}",
            payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"],
        )
        self.assertFalse(payload["live_permission_allowed"])

    def test_concurrent_stale_goal_writer_cannot_overwrite_fresh_repair_wait(self) -> None:
        now = datetime(2026, 7, 10, 0, 57, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            dispatch_id = "b" * 64
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "NO_REPAIR_REQUESTS",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 0,
                    "repair_request_count": 0,
                    "codex_work_order": {
                        "status": "READ_ONLY_EVIDENCE_WORK",
                        "action_code": "EXACT_TP_PROOF_COLLECTION",
                        "material_digest": dispatch_id,
                        "pending_material_digest": dispatch_id,
                        "pending_dispatch_id": dispatch_id,
                        "dispatch_allowed": True,
                        "execution_pending": True,
                        "suggested_commands": ["read-only-command-a"],
                    },
                    "read_only": True,
                    "live_side_effects": [],
                },
            )
            kwargs = {
                "trader_repair_orchestrator_path": paths["repair"],
                "active_trader_contract_path": paths["active_contract"],
                "payoff_shape_diagnosis_path": paths["payoff"],
                "harvest_live_grade_path": paths["harvest"],
                "scout_plan_path": paths["scout"],
                "as_proof_pack_queue_path": paths["proof"],
                "as_lane_candidate_board_path": paths["board"],
                "portfolio_4x_path_planner_path": paths["portfolio"],
                "guardian_receipt_consumption_path": paths["guardian_consumption"],
                "guardian_receipt_operator_review_path": paths["guardian_review"],
                "live_order_request_path": paths["live_order"],
                "broker_snapshot_path": paths["broker"],
                "output_path": paths["output"],
                "report_path": paths["report"],
                "now_utc": now,
            }

            import quant_rabbit.trader_goal_loop_orchestrator as goal_module

            original_load = goal_module._load_artifact
            stale_repair_loaded = threading.Event()
            release_stale_writer = threading.Event()
            errors: list[BaseException] = []

            def controlled_load(path: Path):
                payload = original_load(path)
                if (
                    threading.current_thread().name == "stale-goal-run"
                    and path == paths["repair"]
                ):
                    stale_repair_loaded.set()
                    if not release_stale_writer.wait(timeout=3):
                        raise TimeoutError("test did not release stale goal writer")
                return payload

            def execute() -> None:
                try:
                    TraderGoalLoopOrchestrator(**kwargs).run()
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            stale_thread = threading.Thread(name="stale-goal-run", target=execute)
            fresh_thread = threading.Thread(name="fresh-goal-run", target=execute)
            with patch(
                "quant_rabbit.trader_goal_loop_orchestrator._load_artifact",
                side_effect=controlled_load,
            ):
                stale_thread.start()
                self.assertTrue(stale_repair_loaded.wait(timeout=2))
                _write_json(
                    paths["repair"],
                    {
                        "generated_at_utc": (now + timedelta(seconds=1)).isoformat(),
                        "status": "NO_REPAIR_REQUESTS",
                        "selected_request_code": None,
                        "actionable_request_count": 0,
                        "approval_required_request_count": 0,
                        "waiting_request_count": 0,
                        "repair_request_count": 0,
                        "codex_work_order": {
                            "status": "WAITING_FOR_MATERIAL_EVIDENCE",
                            "reason_code": "MATERIAL_EVIDENCE_UNCHANGED",
                            "action_code": "EXACT_TP_PROOF_COLLECTION",
                            "material_digest": dispatch_id,
                            "dispatch_allowed": False,
                            "repeat_suppressed": True,
                            "suggested_commands": [],
                        },
                        "read_only": True,
                        "live_side_effects": [],
                    },
                )
                fresh_thread.start()
                release_stale_writer.set()
                stale_thread.join(timeout=3)
                fresh_thread.join(timeout=3)

            final_payload = json.loads(paths["output"].read_text())

        self.assertFalse(stale_thread.is_alive())
        self.assertFalse(fresh_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(final_payload["selected_next_work_type"], "NO_ACTION_WAIT")
        self.assertTrue(final_payload["repair_loop_state"]["material_evidence_wait"])
        self.assertNotIn("read-only-command-a", final_payload["next_allowed_commands"])

    def test_goal_report_publish_failure_preserves_last_valid_json(self) -> None:
        now = datetime(2026, 7, 10, 0, 58, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            kwargs = {
                "trader_repair_orchestrator_path": paths["repair"],
                "active_trader_contract_path": paths["active_contract"],
                "payoff_shape_diagnosis_path": paths["payoff"],
                "harvest_live_grade_path": paths["harvest"],
                "scout_plan_path": paths["scout"],
                "as_proof_pack_queue_path": paths["proof"],
                "as_lane_candidate_board_path": paths["board"],
                "portfolio_4x_path_planner_path": paths["portfolio"],
                "guardian_receipt_consumption_path": paths["guardian_consumption"],
                "guardian_receipt_operator_review_path": paths["guardian_review"],
                "live_order_request_path": paths["live_order"],
                "broker_snapshot_path": paths["broker"],
                "output_path": paths["output"],
                "report_path": paths["report"],
                "now_utc": now,
            }
            TraderGoalLoopOrchestrator(**kwargs).run()
            previous_json = paths["output"].read_bytes()
            dispatch_id = "c" * 64
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": (now + timedelta(seconds=1)).isoformat(),
                    "status": "NO_REPAIR_REQUESTS",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 0,
                    "repair_request_count": 0,
                    "codex_work_order": {
                        "status": "READ_ONLY_EVIDENCE_WORK",
                        "action_code": "EXACT_TP_PROOF_COLLECTION",
                        "material_digest": dispatch_id,
                        "pending_material_digest": dispatch_id,
                        "pending_dispatch_id": dispatch_id,
                        "dispatch_allowed": True,
                        "execution_pending": True,
                        "suggested_commands": ["read-only-command-c"],
                    },
                    "read_only": True,
                    "live_side_effects": [],
                },
            )

            import quant_rabbit.trader_goal_loop_orchestrator as goal_module

            original_replace = goal_module._replace_prepared_text

            def fail_report_replace(temp_path: Path, destination: Path) -> None:
                if destination == paths["report"]:
                    raise OSError("simulated goal report ENOSPC")
                original_replace(temp_path, destination)

            with patch(
                "quant_rabbit.trader_goal_loop_orchestrator._replace_prepared_text",
                side_effect=fail_report_replace,
            ):
                with self.assertRaisesRegex(OSError, "simulated goal report ENOSPC"):
                    TraderGoalLoopOrchestrator(**kwargs).run()

            self.assertEqual(paths["output"].read_bytes(), previous_json)
            TraderGoalLoopOrchestrator(**kwargs).run()
            recovered = json.loads(paths["output"].read_text())

        self.assertEqual(recovered["selected_next_work_type"], "ACTIVE_LANE_EVIDENCE_DISPATCH")
        self.assertEqual(
            recovered["repair_loop_state"]["active_lane_pending_dispatch_id"],
            dispatch_id,
        )

    def test_material_evidence_wait_blocks_goal_loop_redispatch_of_same_active_prompt(self) -> None:
        now = datetime(2026, 7, 10, 1, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "NO_REPAIR_REQUESTS",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 0,
                    "repair_request_count": 0,
                    "codex_work_order": {
                        "status": "WAITING_FOR_MATERIAL_EVIDENCE",
                        "reason_code": "MATERIAL_EVIDENCE_UNCHANGED",
                        "action_code": "EXACT_TP_PROOF_COLLECTION",
                        "material_digest": "material-digest-1",
                        "dispatch_allowed": False,
                        "repeat_suppressed": True,
                        "suggested_commands": [],
                    },
                    "read_only": True,
                    "live_side_effects": [],
                },
            )
            # A timestamp-only refresh must not bypass the material watermark.
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": (now + timedelta(seconds=5)).isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "selected_active_path": "EVIDENCE_ACQUISITION",
                    "target_shape": "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT",
                    "next_prompt": "Collect the same exact local TP proof again.",
                    "next_trade_enabling_action": "Collect exact local TP proof; do not send.",
                    "current_state": {
                        "active_opportunity_board": {
                            "top_lane": {
                                "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "strategy_family": "BREAKOUT_FAILURE",
                                "vehicle": "LIMIT",
                                "status": "EVIDENCE_ACQUISITION",
                            }
                        }
                    },
                    "remaining_blockers": [
                        {"code": "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"}
                    ],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )

            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "NO_ACTION_WAIT")
        self.assertEqual(payload["selected_next_work_type"], "NO_ACTION_WAIT")
        self.assertTrue(payload["repair_loop_state"]["waiting_for_evidence"])
        self.assertTrue(payload["repair_loop_state"]["material_evidence_wait"])
        self.assertFalse(payload["repair_loop_state"]["queued_evidence_wait"])
        self.assertIn("suppressed an unchanged active-lane", payload["selection_reason"])
        self.assertIn(
            "REPAIR_MATERIAL_EVIDENCE_WAIT:EXACT_TP_PROOF_COLLECTION:material-digest-1",
            payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"],
        )
        self.assertNotIn(
            "active-trader-contract",
            "\n".join(payload["next_allowed_commands"]),
        )
        self.assertFalse(payload["live_permission_allowed"])

    def test_unmapped_active_lane_action_routes_to_code_repair_not_prompt_redispatch(self) -> None:
        now = datetime(2026, 7, 10, 1, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "NO_REPAIR_REQUESTS",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 0,
                    "repair_request_count": 0,
                    "codex_work_order": {
                        "status": "ACTIVE_LANE_ACTION_MAPPING_REQUIRED",
                        "reason_code": "ACTION_STAGE_MAPPING_REQUIRED",
                        "action_code": "METHOD_SCOPED_PROFILE_PROMOTION",
                        "material_digest": "mapping-gap-digest",
                        "dispatch_allowed": False,
                        "suggested_commands": [],
                    },
                    "read_only": True,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": (now + timedelta(seconds=5)).isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "selected_active_path": "EVIDENCE_ACQUISITION",
                    "next_prompt": "Repeat METHOD_SCOPED_PROFILE_PROMOTION.",
                    "next_trade_enabling_action": (
                        "The next safe tuning action is METHOD_SCOPED_PROFILE_PROMOTION."
                    ),
                    "current_state": {},
                    "remaining_blockers": [],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "CODE_REPAIR")
        self.assertTrue(payload["repair_loop_state"]["action_mapping_required"])
        self.assertTrue(payload["schema_state"]["active_lane_action_mapping_required"])
        self.assertEqual(payload["success_condition_evaluation"]["status"], "NOT_MET")
        self.assertIn("no approved command mapping", payload["selection_reason"])
        self.assertIn(
            "ACTIVE_LANE_ACTION_MAPPING_REQUIRED:METHOD_SCOPED_PROFILE_PROMOTION",
            payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"],
        )
        self.assertFalse(payload["live_permission_allowed"])

    def test_selected_actionable_repair_outranks_newer_active_contract_prompt(self) -> None:
        now = datetime(2026, 7, 10, 1, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            selected_code = "REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY"
            targeted = "PYTHONPATH=src python3 -m unittest tests.test_profit_capture_bot -v"
            verification = "python3 -m json.tool data/profit_capture_bot.json >/dev/null"
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "READY_FOR_CODEX_REPAIR",
                    "selected_request_code": selected_code,
                    "actionable_request_count": 1,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 0,
                    "repair_request_count": 1,
                    "codex_work_order": {
                        "status": "READY_FOR_CODEX_IMPLEMENTATION",
                        "selected_request_code": selected_code,
                        "objective": "Repair the TP-progress profit-capture replay gap.",
                        "automation_prompt": "Implement the selected repair and preserve every live gate.",
                        "suggested_files": ["src/quant_rabbit/profit_capture_bot.py"],
                        "required_tests": ["loss leak remains blocked until replay proves it"],
                        "targeted_test_commands": [targeted],
                        "verification_commands": [verification],
                        "final_verification_commands": [targeted, verification],
                    },
                    "read_only": True,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": (now + timedelta(seconds=5)).isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "selected_active_path": "EVIDENCE_ACQUISITION",
                    "next_prompt": "This newer active-contract prompt must not hide the selected repair.",
                    "next_trade_enabling_action": "Collect exact local TP proof; do not send.",
                    "current_state": {},
                    "remaining_blockers": [],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )

            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "CODE_REPAIR")
        self.assertTrue(payload["repair_loop_state"]["actionable_repair_selected"])
        self.assertEqual(payload["repair_loop_state"]["selected_request_code"], selected_code)
        self.assertEqual(payload["next_allowed_commands"], [targeted, verification])
        self.assertIn(selected_code, payload["current_phase"])
        self.assertIn(selected_code, payload["selected_next_prompt"])
        self.assertIn("Repair the TP-progress profit-capture replay gap", payload["selected_next_prompt"])
        self.assertIn("src/quant_rabbit/profit_capture_bot.py", payload["selected_next_prompt"])
        self.assertIn(
            f"TRADER_REPAIR_REQUEST:{selected_code}",
            payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"],
        )
        self.assertEqual(payload["success_condition_evaluation"]["status"], "NOT_MET")
        self.assertFalse(payload["live_permission_allowed"])

    def test_repair_orchestrator_waiting_evidence_refreshes_stale_artifacts_once(self) -> None:
        now = datetime(2026, 7, 9, 12, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "ORCHESTRATOR_BLOCKED",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 2,
                    "repair_request_count": 2,
                    "read_only": True,
                    "live_side_effects": [],
                    "next_evidence_actions": [
                        {
                            "action_id": "refresh_lane_board_after_input_evidence_changes",
                            "read_only": True,
                            "live_side_effects": [],
                        }
                    ],
                },
            )
            _write_json(
                paths["board"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "CONTRADICTED",
                    "normal_routing_status": "BLOCKED",
                    "routing_allowed": False,
                    "as_live_ready_path_exists": False,
                    "live_side_effects": [],
                },
            )

            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "READ_ONLY_EVIDENCE_REFRESH")
        self.assertEqual(payload["selected_next_work_type"], "READ_ONLY_EVIDENCE_REFRESH")
        self.assertTrue(payload["artifact_health"]["has_stale_or_contradicted_artifact"])
        self.assertIn("stale or contradicted", payload["selection_reason"])
        self.assertIn("trader-repair-orchestrator", "\n".join(payload["next_allowed_commands"]))
        self.assertFalse(payload["live_permission_allowed"])

    def test_newer_active_contract_prompt_supersedes_repair_waiting_refresh_loop(self) -> None:
        now = datetime(2026, 7, 9, 12, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_DIAGNOSIS_COMPLETE")
            _write_json(
                paths["repair"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "ORCHESTRATOR_BLOCKED",
                    "selected_request_code": None,
                    "actionable_request_count": 0,
                    "approval_required_request_count": 0,
                    "waiting_request_count": 3,
                    "repair_request_count": 3,
                    "read_only": True,
                    "live_side_effects": [],
                    "next_evidence_actions": [
                        {
                            "action_id": "refresh_lane_board_after_input_evidence_changes",
                            "read_only": True,
                            "live_side_effects": [],
                        }
                    ],
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": (now + timedelta(seconds=5)).isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
                    "selected_active_path": "EVIDENCE_ACQUISITION",
                    "selected_active_path_reason": "terminal active contract consumed the latest board/frontier.",
                    "target_shape": "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT",
                    "four_x_progress_hypothesis": (
                        "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT plus GBP_USD frontier evidence."
                    ),
                    "root_improvement_target": "Advance the terminal lane-specific evidence plan.",
                    "expected_edge_improvement": "Exact TP proof collection for the active lane and frontier.",
                    "next_prompt": (
                        "Implement EVIDENCE_ACQUISITION for EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT "
                        "plus frontier evidence GBP_USD|LONG|RANGE_ROTATION|LIMIT."
                    ),
                    "next_trade_enabling_action": (
                        "Collect exact local TP proof for EUR_USD and build exact TP-proven "
                        "rotation proof for GBP_USD; do not send."
                    ),
                    "current_state": {
                        "active_opportunity_board": {
                            "top_lane": {
                                "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                                "pair": "EUR_USD",
                                "direction": "LONG",
                                "strategy_family": "BREAKOUT_FAILURE",
                                "vehicle": "LIMIT",
                                "status": "EVIDENCE_ACQUISITION",
                            }
                        },
                        "non_eurusd_live_grade_frontier": {
                            "next_evidence_lane": {
                                "lane_id": "range_trader:GBP_USD:LONG:RANGE_ROTATION",
                                "pair": "GBP_USD",
                                "direction": "LONG",
                                "strategy_family": "RANGE_ROTATION",
                                "vehicle": "LIMIT",
                                "status": "EVIDENCE_ACQUISITION",
                            }
                        },
                    },
                    "remaining_blockers": [{"code": "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"}],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )

            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                active_trader_contract_path=paths["active_contract"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "ACTIVE_TRADER_CONTRACT_EVIDENCE")
        self.assertEqual(payload["selected_next_work_type"], "ACTIVE_TRADER_CONTRACT_EVIDENCE")
        self.assertTrue(payload["repair_loop_state"]["waiting_for_evidence"])
        self.assertTrue(payload["active_contract_state"]["active_prompt_available"])
        self.assertEqual(
            payload["active_contract_state"]["frontier_lane_id"],
            "range_trader:GBP_USD:LONG:RANGE_ROTATION",
        )
        self.assertIn("already-satisfied generic artifact refresh", payload["selection_reason"])
        self.assertIn("GBP_USD|LONG|RANGE_ROTATION|LIMIT", payload["selected_next_prompt"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_repeat_guard_does_not_override_scout_blocked_classification(self) -> None:
        now = datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_BLOCKED_OPERATOR_REVIEW")
            kwargs = {
                "trader_repair_orchestrator_path": paths["repair"],
                "active_trader_contract_path": paths["active_contract"],
                "payoff_shape_diagnosis_path": paths["payoff"],
                "harvest_live_grade_path": paths["harvest"],
                "scout_plan_path": paths["scout"],
                "as_proof_pack_queue_path": paths["proof"],
                "as_lane_candidate_board_path": paths["board"],
                "portfolio_4x_path_planner_path": paths["portfolio"],
                "guardian_receipt_consumption_path": paths["guardian_consumption"],
                "guardian_receipt_operator_review_path": paths["guardian_review"],
                "live_order_request_path": paths["live_order"],
                "broker_snapshot_path": paths["broker"],
                "output_path": paths["output"],
                "report_path": paths["report"],
                "now_utc": now,
            }
            TraderGoalLoopOrchestrator(**kwargs).run()
            TraderGoalLoopOrchestrator(**kwargs).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["selected_next_work_type"], "OPERATOR_REVIEW_REPORT")
        self.assertFalse(payload["repeat_loop_guard"]["repeat_allowed"])
        self.assertIn("repeat_loop_guard", payload["selection_reason"])


def _write_base_artifacts(root: Path, *, now: datetime, scout_status: str) -> dict[str, Path]:
    paths = {
        "repair": root / "data" / "trader_repair_orchestrator.json",
        "active_contract": root / "data" / "active_trader_contract.json",
        "payoff": root / "data" / "payoff_shape_diagnosis.json",
        "harvest": root / "data" / "harvest_live_grade_path.json",
        "scout": root / "data" / "eurusd_short_breakout_failure_scout_plan.json",
        "proof": root / "data" / "as_proof_pack_queue.json",
        "board": root / "data" / "as_lane_candidate_board.json",
        "portfolio": root / "data" / "portfolio_4x_path_planner.json",
        "guardian_consumption": root / "data" / "guardian_receipt_consumption.json",
        "guardian_review": root / "data" / "guardian_receipt_operator_review.json",
        "live_order": root / "data" / "live_order_request.json",
        "broker": root / "data" / "broker_snapshot.json",
        "output": root / "data" / "trader_goal_loop_orchestrator.json",
        "report": root / "docs" / "trader_goal_loop_orchestrator_report.md",
    }
    _write_json(
        paths["repair"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "READY_FOR_CODEX_REPAIR",
            "read_only": True,
            "live_side_effects": [],
            "next_evidence_actions": [
                {
                    "action_id": "collect_exact_tp_or_live_grade_harvest_evidence",
                    "read_only": True,
                    "live_side_effects": [],
                    "success_condition": {
                        "schema_version": "success_condition_v1",
                        "mode": "any",
                        "checks": [
                            {"field": "proof_queue_count", "operator": "gt", "value": 0},
                        ],
                    },
                }
            ],
        },
    )
    _write_json(
        paths["active_contract"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "ACTIVE_PATH_SELECTED_REPLAY_PASSED_STILL_BLOCKED",
            "selected_active_path": "NO_TRADE_WITH_CAUSE",
            "next_prompt": "",
            "next_trade_enabling_action": "",
            "current_state": {},
            "remaining_blockers": [],
            "live_permission_allowed": False,
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["payoff"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "OK",
            "overall_payoff_shape_verdict": {
                "classification": "MIXED_HARVEST_PRIMARY",
                "capture_economics_status": "NEGATIVE_EXPECTANCY",
                "live_promotion_allowed": False,
                "month_scale_blocker": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                "live_promotion_blockers": ["NEGATIVE_EXPECTANCY", "MONTH_SCALE_REPLAY_NEGATIVE"],
            },
            "runner_candidates": [],
            "harvest_candidates": [{"shape_key": f"shape-{idx}"} for idx in range(12)],
            "missing_source_artifacts": [],
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["harvest"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "DIAGNOSIS_COMPLETE_BLOCKED_NO_LIVE_PERMISSION",
            "live_promotion_allowed": False,
            "closest_harvest_candidate": {
                "candidate_id": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                "shape_key": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                "pair": "EUR_USD",
                "side": "SHORT",
                "method": "BREAKOUT_FAILURE",
                "actual_proof_queue_member": False,
                "planner_can_enter_proof_pack": False,
                "can_create_live_permission": False,
                "live_promotion_allowed": False,
                "promotion_blockers": ["GUARDIAN_OPERATOR_REVIEW_BLOCK", "SAMPLE_GAP"],
                "tp_proof": {
                    "take_profit_trades": 17,
                    "take_profit_losses": 0,
                    "proof_floor_trades": 20,
                    "proof_gap_trades": 3,
                    "take_profit_expectancy_jpy": 613.2,
                },
                "current_intent_best": {
                    "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                    "status": "DRY_RUN_BLOCKED",
                    "order_type": "LIMIT",
                    "risk_allowed": False,
                    "risk_jpy": 418.0,
                    "units": 3000,
                    "live_blocker_codes": ["GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"],
                },
            },
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["scout"],
        {
            "generated_at_utc": now.isoformat(),
            "status": scout_status,
            "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE",
            "scout_mode_allowed": scout_status != "SCOUT_BLOCKED_OPERATOR_REVIEW",
            "operator_approval_required": scout_status == "SCOUT_BLOCKED_OPERATOR_REVIEW",
            "max_loss_jpy_cap": 418.0,
            "min_lot_feasibility": {
                "status": "MIN_LOT_NUMERICALLY_FEASIBLE_BUT_OTHER_GATES_BLOCK",
                "feasible_if_all_non_lot_gates_clear": True,
            },
            "proof_queue_entry_blockers": [
                {
                    "code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                    "evidence": {
                        "normal_routing_allowed": False,
                        "guardian_receipt_consumption_status": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                    },
                }
            ],
            "evidence_success_conditions": [],
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["proof"],
        {
            "generated_at_utc": now.isoformat(),
            "summary": {
                "queue_count": 0,
                "proof_ready_count": 0,
                "can_create_live_permission_count": 0,
                "rejected_candidate_count": 1,
                "as_live_ready_path_exists": False,
            },
            "queue": [],
            "rejected_candidates": [{"lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"}],
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["board"],
        {
            "generated_at_utc": now.isoformat(),
            "normal_routing_status": "BLOCKED",
            "routing_allowed": False,
            "as_live_ready_path_exists": False,
            "exact_blocker_preventing_live_ready": {
                "primary": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                "global_blockers": [
                    "NEGATIVE_EXPECTANCY_ACTIVE",
                    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                ],
                "p0_rows": ["NEGATIVE_EXPECTANCY_ACTIVE"],
            },
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["portfolio"],
        {
            "generated_at_utc": now.isoformat(),
            "portfolio_status": "NO_LIVE_READY_PORTFOLIO",
            "normal_routing_status": "BLOCKED",
            "can_reach_4x_now": False,
            "summary": {"can_create_live_permission": False},
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["live_order"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "NO_ACTION",
            "send_requested": False,
            "sent": False,
        },
    )
    _write_json(
        paths["broker"],
        {
            "fetched_at_utc": now.isoformat(),
            "account": {},
            "positions": [],
            "orders": [],
            "quotes": {},
        },
    )
    return paths


if __name__ == "__main__":
    unittest.main()
