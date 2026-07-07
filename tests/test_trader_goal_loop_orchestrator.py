from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
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
        self.assertFalse(payload["requires_operator_approval"])
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
            summary = TraderGoalLoopOrchestrator(
                trader_repair_orchestrator_path=paths["repair"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "EDGE_IMPROVEMENT_EXPERIMENT")
        self.assertEqual(payload["selected_next_work_type"], "EDGE_IMPROVEMENT_EXPERIMENT")
        self.assertFalse(payload["requires_operator_approval"])
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
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                as_proof_pack_queue_path=paths["proof"],
                as_lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(summary.selected_next_work_type, "OPERATOR_REVIEW_REPORT")
        self.assertEqual(payload["selected_next_work_type"], "OPERATOR_REVIEW_REPORT")
        self.assertFalse(payload["live_permission_allowed"])
        self.assertIn("SCOUT_BLOCKED_OPERATOR_REVIEW", payload["repeat_loop_guard"]["current_fingerprint"]["key_blocker"])

    def test_repeat_guard_does_not_override_scout_blocked_classification(self) -> None:
        now = datetime(2026, 7, 7, 13, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now, scout_status="SCOUT_BLOCKED_OPERATOR_REVIEW")
            kwargs = {
                "trader_repair_orchestrator_path": paths["repair"],
                "payoff_shape_diagnosis_path": paths["payoff"],
                "harvest_live_grade_path": paths["harvest"],
                "scout_plan_path": paths["scout"],
                "as_proof_pack_queue_path": paths["proof"],
                "as_lane_candidate_board_path": paths["board"],
                "portfolio_4x_path_planner_path": paths["portfolio"],
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
        "payoff": root / "data" / "payoff_shape_diagnosis.json",
        "harvest": root / "data" / "harvest_live_grade_path.json",
        "scout": root / "data" / "eurusd_short_breakout_failure_scout_plan.json",
        "proof": root / "data" / "as_proof_pack_queue.json",
        "board": root / "data" / "as_lane_candidate_board.json",
        "portfolio": root / "data" / "portfolio_4x_path_planner.json",
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
