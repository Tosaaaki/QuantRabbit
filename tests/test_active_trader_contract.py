from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.active_trader_contract import ActiveTraderContract


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class ActiveTraderContractTest(unittest.TestCase):
    def test_replay_passed_still_selects_evidence_acquisition_not_no_action(self) -> None:
        now = datetime(2026, 7, 8, 6, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)

            summary = ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        self.assertEqual(summary.selected_active_path, "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["target_shape"], "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST")
        self.assertTrue(payload["active_path_required"])
        self.assertFalse(payload["no_action_allowed"])
        self.assertFalse(payload["no_action_contract"]["no_action_allowed"])
        self.assertIn("HARVEST_CANDIDATE_PRESENT", payload["no_action_contract"]["blocked_by"])
        self.assertIn("SCOUT_CANDIDATE_PRESENT", payload["no_action_contract"]["blocked_by"])
        self.assertIn("MAX_LOSS_CAP_DEFINED", payload["no_action_contract"]["blocked_by"])
        self.assertIn("MIN_LOT_FEASIBLE", payload["no_action_contract"]["blocked_by"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertEqual(payload["contract_goal"], "monthly_funding_adjusted_equity_4x")
        self.assertEqual(payload["current_state"]["proof_floor"]["wins"], 20)
        self.assertEqual(
            payload["current_state"]["daily_target"]["uncapped_required_trades_per_day"],
            173,
        )
        self.assertEqual(
            payload["current_state"]["daily_target"][
                "selected_basis_uncapped_required_trades_per_day"
            ],
            87,
        )
        self.assertEqual(
            payload["current_state"]["daily_target"]["trade_pace_feasibility"],
            "INFEASIBLE_AT_OPERATING_PACE",
        )
        self.assertTrue(payload["current_state"]["daily_target"]["advisory_only"])
        self.assertTrue(payload["current_state"]["proof_floor"]["proof_floor_reached"])
        self.assertEqual(payload["current_state"]["limit_s5_bidask_replay"]["replay_wins"], 4)
        self.assertTrue(payload["current_state"]["limit_s5_bidask_replay"]["passed"])
        self.assertEqual(
            payload["current_state"]["limit_sample_mining"]["additional_acceptable_local_samples_found"],
            0,
        )
        self.assertEqual(payload["current_state"]["limit_sample_mining"]["remaining_exact_limit_samples"], 16)
        self.assertIn("Canonicalize", payload["next_trade_enabling_action"])
        self.assertIn("0 new acceptable samples", payload["next_trade_enabling_action"])
        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertIn("LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY", blocker_codes)
        self.assertIn("LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED", blocker_codes)
        self.assertIn("S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION", blocker_codes)
        self.assertIn("PROOF_QUEUE_COUNT_ZERO_NOT_PERMISSION", blocker_codes)
        self.assertIn("NEGATIVE_EXPECTANCY_ACTIVE", blocker_codes)
        self.assertIn("MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE", blocker_codes)
        self.assertIn("Selected active path: `EVIDENCE_ACQUISITION`", report)

    def test_non_empty_proof_queue_filters_stale_queue_empty_blockers(self) -> None:
        now = datetime(2026, 7, 8, 6, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["proof"],
                {
                    "generated_at_utc": now.isoformat(),
                    "summary": {
                        "queue_count": 2,
                        "proof_ready_count": 0,
                        "can_create_live_permission_count": 0,
                        "rejected_candidate_count": 4,
                    },
                    "queue": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "proof_classification": "EVIDENCE_GAP",
                            "can_create_live_permission": False,
                            "can_enter_proof_pack": True,
                        }
                    ],
                    "live_side_effects": [],
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        replay_blockers = set(payload["current_state"]["limit_s5_bidask_replay"]["blocker_codes"])
        self.assertEqual(payload["current_state"]["proof"]["proof_queue_count"], 2)
        self.assertIn("PROOF_QUEUE_HAS_CANDIDATES", payload["active_deployment_gap"]["active_path_triggers"])
        self.assertNotIn("NOT_IN_PROOF_QUEUE", replay_blockers)
        self.assertNotIn("NOT_IN_PROOF_QUEUE", blocker_codes)
        self.assertNotIn("PROOF_QUEUE_EMPTY_NO_LIVE_PERMISSION", blocker_codes)
        self.assertNotIn("PROOF_QUEUE_COUNT_ZERO_NOT_PERMISSION", blocker_codes)
        self.assertIn("LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY", blocker_codes)
        self.assertIn("S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION", blocker_codes)

    def test_proof_floor_and_positive_replay_filter_stale_scout_sample_gap(self) -> None:
        now = datetime(2026, 7, 8, 6, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["harvest"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "DIAGNOSIS_COMPLETE_BLOCKED_NO_LIVE_PERMISSION",
                    "live_promotion_allowed": False,
                    "closest_harvest_candidate": {
                        "candidate_id": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                        "shape_key": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                        "actual_proof_queue_member": True,
                        "planner_can_enter_proof_pack": True,
                        "can_create_live_permission": False,
                        "live_promotion_allowed": False,
                        "promotion_blockers": [
                            "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
                            "PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY",
                            "S5_BIDASK_SPREAD_INCLUDED_REPLAY_MISSING",
                        ],
                        "tp_proof": {
                            "take_profit_trades": 20,
                            "take_profit_losses": 0,
                            "proof_floor_trades": 20,
                            "proof_gap_trades": 0,
                            "take_profit_expectancy_jpy": 613.2,
                        },
                    },
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
                        {"code": "SAMPLE_GAP"},
                        {"code": "POSITIVE_SPREAD_SLIPPAGE_PROOF_MISSING"},
                        {"code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"},
                    ],
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["proof"],
                {
                    "generated_at_utc": now.isoformat(),
                    "summary": {
                        "queue_count": 2,
                        "proof_ready_count": 0,
                        "can_create_live_permission_count": 0,
                        "rejected_candidate_count": 4,
                    },
                    "queue": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "proof_classification": "EVIDENCE_GAP",
                            "can_create_live_permission": False,
                            "can_enter_proof_pack": True,
                        }
                    ],
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["proof_floor"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "PROOF_FLOOR_REACHED_STILL_BLOCKED",
                    "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                    "post_update_tp_proof": {
                        "wins": 20,
                        "losses": 0,
                        "proof_floor": 20,
                        "remaining_samples": 0,
                        "proof_floor_reached": True,
                    },
                    "remaining_blockers": [
                        {"code": "SPREAD_SLIPPAGE_PROOF_MISSING"},
                        {"code": "NEGATIVE_EXPECTANCY_ACTIVE"},
                    ],
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        scout_blockers = set(payload["current_state"]["scout"]["blocker_codes"])
        proof_floor_blockers = set(payload["current_state"]["proof_floor"]["blocker_codes"])
        self.assertEqual(payload["current_state"]["harvest"]["tp_proof"]["proof_gap_trades"], 0)
        self.assertTrue(payload["current_state"]["limit_s5_bidask_replay"]["passed"])
        self.assertNotIn("SAMPLE_GAP", blocker_codes)
        self.assertNotIn("POSITIVE_SPREAD_SLIPPAGE_PROOF_MISSING", blocker_codes)
        self.assertNotIn("S5_BIDASK_SPREAD_INCLUDED_REPLAY_MISSING", blocker_codes)
        self.assertNotIn("SPREAD_SLIPPAGE_PROOF_MISSING", blocker_codes)
        self.assertNotIn("SAMPLE_GAP", scout_blockers)
        self.assertNotIn("POSITIVE_SPREAD_SLIPPAGE_PROOF_MISSING", scout_blockers)
        self.assertNotIn("SPREAD_SLIPPAGE_PROOF_MISSING", proof_floor_blockers)
        self.assertIn("LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY", blocker_codes)
        self.assertIn("PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY", blocker_codes)

    def test_missing_exact_replay_is_evidence_acquisition_blocker(self) -> None:
        now = datetime(2026, 7, 8, 6, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            paths["replay"].unlink()

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertFalse(payload["no_action_allowed"])
        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertIn("EXACT_LIMIT_S5_BIDASK_REPLAY_MISSING", blocker_codes)
        self.assertIn("REPLAY_OR_PROOF_ACTION_AVAILABLE", payload["active_deployment_gap"]["active_path_triggers"])
        self.assertEqual(
            payload["next_trade_enabling_action"],
            "Generate exact EUR_USD SHORT BREAKOUT_FAILURE LIMIT HARVEST S5 bid/ask replay artifact.",
        )

    def test_consumes_previous_board_failed_stop_replay_without_repeating_it(self) -> None:
        now = datetime(2026, 7, 8, 9, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            board = json.loads(paths["board"].read_text())
            board["exact_blocker_preventing_live_ready"]["global_blockers"].append(
                "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
            )
            _write_json(paths["board"], board)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 93,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 17,
                        "pairs_scanned": ["EUR_USD", "AUD_JPY"],
                        "vehicles_scanned": ["LIMIT", "STOP", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "replay_status": "SPREAD_SLIPPAGE_PROOF_INCOMPLETE",
                        "next_action": "Acquire exact LIMIT proof without mixing vehicles.",
                        "blockers": ["LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY"],
                    },
                    "ranked_active_lanes": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "pair": "EUR_USD",
                            "direction": "SHORT",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "status": "EVIDENCE_ACQUISITION",
                            "replay_status": "SPREAD_SLIPPAGE_PROOF_INCOMPLETE",
                            "next_action": "Acquire exact LIMIT proof without mixing vehicles.",
                            "blockers": ["LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY"],
                        },
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE",
                            "pair": "EUR_USD",
                            "direction": "SHORT",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "STOP",
                            "status": "NO_TRADE_WITH_CAUSE",
                            "replay_status": "STOP_HARVEST_EXACT_S5_BIDASK_REPLAY_FAILED_BLOCKED",
                            "next_action": "Do not repeat the same exact replay.",
                            "blockers": [
                                "STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED",
                                "S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS",
                            ],
                        },
                    ],
                    "next_active_path": "EVIDENCE_ACQUISITION: failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT is the closest read-only path.",
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        board_state = payload["current_state"]["active_opportunity_board"]
        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(board_state["total_lanes"], 93)
        self.assertTrue(board_state["stop_harvest_failed_replay_consumed"])
        self.assertEqual(board_state["failed_exact_replay_consumed_count"], 1)
        self.assertIn("S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS", blocker_codes)
        self.assertIn("STOP exact replay is already consumed", payload["next_trade_enabling_action"])
        self.assertNotIn(
            "STOP_HARVEST_EXACT_REPLAY_BEFORE_SCOUT_OR_OPERATOR_REVIEW",
            payload["next_trade_enabling_action"],
        )
        self.assertIn("Failed exact replay consumed count: `1`", report)

    def test_previous_board_guardian_blocker_overrides_stale_evidence_status(self) -> None:
        now = datetime(2026, 7, 8, 10, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 93,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 17,
                        "pairs_scanned": ["EUR_USD", "AUD_JPY"],
                        "vehicles_scanned": ["LIMIT", "STOP", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": "Acquire exact LIMIT proof without mixing vehicles.",
                        "blockers": [
                            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                            "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
                        ],
                    },
                    "ranked_active_lanes": [
                        {
                            "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                            "pair": "EUR_USD",
                            "direction": "SHORT",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "status": "EVIDENCE_ACQUISITION",
                            "next_action": "Acquire exact LIMIT proof without mixing vehicles.",
                            "blockers": [
                                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                                "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
                            ],
                        }
                    ],
                    "next_active_path": "EVIDENCE_ACQUISITION: stale previous board path.",
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        board_state = payload["current_state"]["active_opportunity_board"]
        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(board_state["top_lane"]["status"], "OPERATOR_REVIEW_REQUIRED")
        self.assertEqual(payload["selected_active_path"], "OPERATOR_REVIEW_REPORT")
        self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", blocker_codes)
        self.assertIn("top lane failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT is OPERATOR_REVIEW_REQUIRED", payload["selected_active_path_reason"])
        self.assertIn("operator/guardian review evidence", payload["next_trade_enabling_action"])

    def test_operator_review_path_preserves_after_review_edge_repair_action(self) -> None:
        now = datetime(2026, 7, 8, 10, 20, tzinfo=timezone.utc)
        lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_OPERATOR_REVIEW_REQUIRED_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 95,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 0,
                        "operator_review_required_count": 81,
                        "pairs_scanned": ["EUR_USD", "GBP_USD"],
                        "vehicles_scanned": ["LIMIT", "STOP", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "OPERATOR_REVIEW_REQUIRED",
                        "edge_improvement_candidate": True,
                        "edge_improvement_target": "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT",
                        "next_action": (
                            "Package guardian receipt operator-review evidence for "
                            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT; do not infer approval. "
                            "After review clears, run read-only EDGE_IMPROVEMENT_EXPERIMENT for "
                            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT; preserve bid/ask, forecast, risk, "
                            "and profitability blockers, rerank, and do not send."
                        ),
                        "blockers": [
                            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                            "NEGATIVE_EXPECTANCY_ACTIVE",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "OPERATOR_REVIEW_REQUIRED: failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT "
                        "is top-ranked."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["selected_active_path"], "OPERATOR_REVIEW_REPORT")
        self.assertIn("After review clears", payload["next_trade_enabling_action"])
        self.assertIn("EDGE_IMPROVEMENT_EXPERIMENT", payload["next_trade_enabling_action"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_active_board_top_blockers_override_stale_hardcoded_target_blockers(self) -> None:
        now = datetime(2026, 7, 8, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_OPERATOR_REVIEW_REQUIRED_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 93,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 0,
                        "operator_review_required_count": 79,
                        "no_trade_count": 14,
                        "pairs_scanned": ["AUD_USD", "EUR_USD"],
                        "vehicles_scanned": ["LIMIT", "STOP", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": "failure_trader:AUD_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "pair": "AUD_USD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "OPERATOR_REVIEW_REQUIRED",
                        "next_action": "Package guardian receipt operator-review evidence.",
                        "blockers": [
                            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": "OPERATOR_REVIEW_REQUIRED: AUD_USD top lane.",
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "OPERATOR_REVIEW_REPORT")
        self.assertIn("AUD_USD:LONG:BREAKOUT_FAILURE:LIMIT", payload["selected_active_path_reason"])
        self.assertEqual(
            blocker_codes,
            {
                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            },
        )
        self.assertNotIn("LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY", blocker_codes)
        self.assertNotIn("PORTFOLIO_PLANNER_CANNOT_CREATE_LIVE_PERMISSION", blocker_codes)
        self.assertNotIn("NO_LIVE_ORDER_REQUEST", blocker_codes)

    def test_active_board_bidask_refresh_path_is_evidence_acquisition_not_no_trade(self) -> None:
        now = datetime(2026, 7, 8, 11, 55, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 142,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 1,
                        "operator_review_required_count": 0,
                        "no_trade_count": 141,
                        "pairs_scanned": ["EUR_USD", "USD_CHF"],
                        "vehicles_scanned": ["LIMIT", "STOP", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "replay_status": "NEGATIVE_EVIDENCE_REFRESH_REQUIRED",
                        "next_action": (
                            "Refresh exact S5 bid/ask replay evidence for "
                            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT."
                        ),
                        "blockers": [
                            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                            "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT "
                        "is the closest read-only path."
                    ),
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ALL_FRONTIER_BLOCKED_BY_NEGATIVE_EXPECTANCY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "scanned_pairs": ["EUR_USD", "USD_CAD"],
                    "scanned_intents": 2,
                    "top_lane": {},
                    "top_non_eurusd_lane": {
                        "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "NO_TRADE_WITH_CAUSE",
                        "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_SPREAD_FORECAST",
                        "blockers": [
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "SPREAD_TOO_WIDE",
                        ],
                        "next_action": "Do not override the active board evidence lane.",
                    },
                    "required_checks": {
                        "next_evidence_lane": {
                            "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "pair": "USD_CAD",
                            "direction": "LONG",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "status": "NO_TRADE_WITH_CAUSE",
                            "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_SPREAD_FORECAST",
                            "blockers": [
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "SPREAD_TOO_WIDE",
                            ],
                        }
                    },
                    "next_active_path": "EVIDENCE_ACQUISITION: USD_CAD read-only frontier.",
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertIn(
            "top lane failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT is EVIDENCE_ACQUISITION",
            payload["selected_active_path_reason"],
        )
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", blocker_codes)
        self.assertIn("BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED", blocker_codes)
        refresh = next(
            row
            for row in payload["remaining_blockers"]
            if row["code"] == "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED"
        )
        self.assertEqual(refresh["status"], "BLOCKING_EVIDENCE_REFRESH")
        self.assertIn("Refresh exact S5 bid/ask replay evidence", payload["next_trade_enabling_action"])
        self.assertIn(
            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT",
            payload["next_prompt"],
        )
        self.assertNotIn("USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT", payload["next_prompt"])
        self.assertNotIn("Use non_eurusd_live_grade_frontier", payload["next_trade_enabling_action"])
        self.assertNotIn("SPREAD_TOO_WIDE", blocker_codes)
        self.assertNotIn(
            "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST",
            payload["next_prompt"],
        )

    def test_eurusd_board_path_keeps_distinct_non_eurusd_frontier_parallel_prompt(self) -> None:
        now = datetime(2026, 7, 9, 17, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 127,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 2,
                        "operator_review_required_count": 0,
                        "no_trade_count": 125,
                        "pairs_scanned": ["EUR_USD", "AUD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": (
                            "Collect exact local TAKE_PROFIT_ORDER proof for "
                            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT|TAKE_PROFIT_ORDER."
                        ),
                        "blockers": [
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT "
                        "is the closest read-only path."
                    ),
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "NON_EURUSD_FRONTIER_FOUND",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "scanned_pairs": ["EUR_USD", "AUD_CAD"],
                    "scanned_intents": 113,
                    "top_lane": {
                        "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "pair": "EUR_USD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                    },
                    "top_non_eurusd_lane": {
                        "lane_id": "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                        "pair": "AUD_CAD",
                        "direction": "SHORT",
                        "strategy_family": "RANGE_ROTATION",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_BIDASK_REPLAY_TP_PROOF_FLOOR",
                        "bidask_status": "NEGATIVE",
                        "spread_status": "PASS",
                        "forecast_status": "PASS",
                        "loss_budget_status": "PASS",
                        "tp_proof_count": 0,
                        "tp_proof_floor": 20,
                        "blockers": [
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                            "LOCAL_TP_PROOF_ZERO_TRADES",
                        ],
                        "next_action": (
                            "Repair bid/ask-negative pattern or vehicle shape for "
                            "range_trader:AUD_CAD:SHORT:RANGE_ROTATION; do not repeat replay until the lane inputs change."
                        ),
                    },
                    "required_checks": {
                        "non_eurusd_closer_than_eurusd": False,
                        "spread_too_wide_not_ignored": True,
                        "bidask_negative_not_ignored": True,
                        "next_evidence_lane": {
                            "lane_id": "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                            "pair": "AUD_CAD",
                            "direction": "SHORT",
                            "strategy_family": "RANGE_ROTATION",
                            "vehicle": "LIMIT",
                            "status": "EVIDENCE_ACQUISITION",
                            "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_BIDASK_REPLAY_TP_PROOF_FLOOR",
                            "bidask_status": "NEGATIVE",
                            "spread_status": "PASS",
                            "forecast_status": "PASS",
                            "loss_budget_status": "PASS",
                            "blockers": [
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                                "LOCAL_TP_PROOF_ZERO_TRADES",
                            ],
                            "next_action": (
                                "Repair bid/ask-negative pattern or vehicle shape for "
                                "range_trader:AUD_CAD:SHORT:RANGE_ROTATION; do not repeat replay until the lane inputs change."
                            ),
                        },
                    },
                    "next_active_path": (
                        "BIDASK_NEGATIVE_PATTERN_REPAIR: current exact bid/ask replay is negative for "
                        "range_trader:AUD_CAD:SHORT:RANGE_ROTATION; repair pattern/vehicle selection or lane-local TP proof before rerunning replay. Do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertIn("distinct read-only evidence lane", payload["selected_active_path_reason"])
        self.assertIn("range_trader:AUD_CAD:SHORT:RANGE_ROTATION", payload["next_trade_enabling_action"])
        self.assertIn("Parallel non_eurusd_live_grade_frontier", payload["next_trade_enabling_action"])
        self.assertIn("AUD_CAD|SHORT|RANGE_ROTATION|LIMIT", payload["next_prompt"])
        self.assertIn("EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT", payload["next_prompt"])
        self.assertIn("Non-EUR frontier blockers", payload["next_prompt"])
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", payload["next_prompt"])
        self.assertIn("Non-EUR frontier blockers", payload["next_trade_enabling_action"])
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", payload["next_trade_enabling_action"])
        self.assertIn("NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", blocker_codes)
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", blocker_codes)
        self.assertEqual(
            payload["target_shape"],
            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT plus frontier evidence AUD_CAD|SHORT|RANGE_ROTATION|LIMIT",
        )
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_parallel_non_eurusd_frontier_uses_latest_matching_forecast_artifact(self) -> None:
        now = datetime(2026, 7, 9, 18, 15, tzinfo=timezone.utc)
        gbp_lane_id = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 127,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 2,
                        "operator_review_required_count": 0,
                        "no_trade_count": 125,
                        "pairs_scanned": ["EUR_USD", "GBP_USD"],
                        "vehicles_scanned": ["LIMIT", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "RANGE_ROTATION",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": "Wait for EUR_USD range rail recheck; do not send.",
                        "blockers": [
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": "EVIDENCE_ACQUISITION: EUR_USD range rail.",
                },
            )
            frontier_lane = {
                "lane_id": gbp_lane_id,
                "pair": "GBP_USD",
                "direction": "LONG",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "distance_to_live_ready": "2_CLOSE_BUT_BLOCKED_BY_NEGATIVE_EXPECTANCY_AND_TP_PROOF_FLOOR",
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                    "EXHAUSTION_RANGE_CHASE",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
                "next_action": "Run entry-frequency recovery analysis for GBP_USD|LONG|RANGE_ROTATION|LIMIT; do not send.",
            }
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "NON_EURUSD_FRONTIER_FOUND",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "scanned_pairs": ["EUR_USD", "GBP_USD"],
                    "scanned_intents": 117,
                    "top_non_eurusd_lane": frontier_lane,
                    "required_checks": {"next_evidence_lane": frontier_lane},
                    "next_active_path": "ENTRY_FREQUENCY_RECOVERY: Run entry-frequency recovery analysis for GBP_USD; do not send.",
                },
            )
            _write_json(
                paths["entry_recovery"],
                {
                    "schema_version": "entry_frequency_recovery_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": frontier_lane,
                    "next_contract_prompt": "Consume data/entry_frequency_recovery.json for range_trader:GBP_USD:LONG:RANGE_ROTATION; do not send.",
                },
            )
            _write_json(
                paths["forecast_pattern"],
                {
                    "schema_version": "forecast_pattern_refresh_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "FORECAST_PATTERN_REFRESH_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": {
                        **frontier_lane,
                        "forecast_range_box": {"status": "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL"},
                        "range_rotation_counterpart": {
                            "status": "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL"
                        },
                    },
                    "next_actions": [
                        {
                            "priority": 1,
                            "lane_id": gbp_lane_id,
                            "action_type": "RANGE_RAIL_GEOMETRY_REPAIR",
                            "description": "wait for broad discount/lower-half rail for LONG; do not chase.",
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/forecast_pattern_refresh.json for "
                        "range_trader:GBP_USD:LONG:RANGE_ROTATION: next safe action is "
                        "RANGE_RAIL_GEOMETRY_REPAIR; do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                entry_frequency_recovery_path=paths["entry_recovery"],
                forecast_pattern_refresh_path=paths["forecast_pattern"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertTrue(
            payload["next_trade_enabling_action"].startswith(
                "Advance non_eurusd_live_grade_frontier as the current next action"
            )
        )
        self.assertIn("Consume data/forecast_pattern_refresh.json", payload["next_trade_enabling_action"])
        self.assertIn(gbp_lane_id, payload["next_trade_enabling_action"])
        self.assertIn("RANGE_RAIL_GEOMETRY_REPAIR", payload["next_trade_enabling_action"])
        self.assertNotIn("Consume data/entry_frequency_recovery.json", payload["next_trade_enabling_action"])
        self.assertIn("Keep active board lane range_trader:EUR_USD:SHORT:RANGE_ROTATION visible", payload["next_trade_enabling_action"])
        self.assertTrue(
            payload["next_prompt"].startswith(
                "Advance non_eurusd_live_grade_frontier evidence as the current next work"
            )
        )
        self.assertIn("Consume data/forecast_pattern_refresh.json", payload["next_prompt"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_board_all_no_trade_with_guardian_clear_overrides_stale_single_lane_evidence(self) -> None:
        now = datetime(2026, 7, 8, 10, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_NO_TRADE_WITH_CAUSE",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "global_safety": {
                        "guardian_receipt_normal_routing_allowed": True,
                        "live_permission_allowed": False,
                    },
                    "coverage_summary": {
                        "total_lanes": 112,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 0,
                        "operator_review_required_count": 0,
                        "no_trade_count": 112,
                        "pairs_scanned": ["USD_CHF", "EUR_USD"],
                        "vehicles_scanned": ["LIMIT", "STOP", "MARKET"],
                    },
                    "top_lane": {
                        "lane_id": "trend_trader:USD_CHF:LONG:TREND_CONTINUATION",
                        "pair": "USD_CHF",
                        "direction": "LONG",
                        "strategy_family": "TREND_CONTINUATION",
                        "vehicle": "STOP",
                        "status": "NO_TRADE_WITH_CAUSE",
                        "next_action": "No trade for USD_CHF|LONG|TREND_CONTINUATION|STOP; preserve blocker cause.",
                        "blockers": [
                            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING",
                            "GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING",
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                        ],
                        "stale_source_blockers": [
                            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                        ],
                    },
                    "stale_source_reasons": [
                        {
                            "code": "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
                            "count": 30,
                            "example_lane_ids": ["failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"],
                        }
                    ],
                    "ranked_active_lanes": [],
                    "next_active_path": "NO_TRADE_WITH_CAUSE: all lanes blocked by current profitability/replay causes.",
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "NO_TRADE_WITH_CAUSE")
        self.assertIn("all lanes as NO_TRADE_WITH_CAUSE", payload["selected_active_path_reason"])
        self.assertNotIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", blocker_codes)
        self.assertNotIn("GUARDIAN_RECEIPT_CONSUMPTION_BLOCKS_NORMAL_ROUTING", blocker_codes)
        self.assertNotIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING", blocker_codes)
        self.assertNotIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", blocker_codes)
        self.assertNotIn("NEGATIVE_EXPECTANCY_ACTIVE", blocker_codes)
        self.assertNotIn("PROOF_QUEUE_COUNT_ZERO_NOT_PERMISSION", blocker_codes)
        self.assertNotIn("PORTFOLIO_PLANNER_CANNOT_CREATE_LIVE_PERMISSION", blocker_codes)
        self.assertNotIn("NO_LIVE_ORDER_REQUEST", blocker_codes)
        self.assertIn("NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", blocker_codes)
        self.assertNotIn(
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING",
            payload["current_state"]["active_opportunity_board"]["top_lane"]["blockers"],
        )
        self.assertIn(
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_BLOCKS_NORMAL_ROUTING",
            payload["current_state"]["active_opportunity_board"]["top_lane"]["stale_source_blockers"],
        )

    def test_non_eurusd_frontier_turns_all_no_trade_board_into_evidence_action(self) -> None:
        now = datetime(2026, 7, 9, 1, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_NO_TRADE_WITH_CAUSE",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 129,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 0,
                        "operator_review_required_count": 0,
                        "no_trade_count": 129,
                        "pairs_scanned": ["EUR_USD", "USD_CAD", "AUD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": {
                        "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "RANGE_ROTATION",
                        "vehicle": "LIMIT",
                        "status": "NO_TRADE_WITH_CAUSE",
                        "next_action": "No trade; preserve negative expectancy.",
                        "blockers": ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": "NO_TRADE_WITH_CAUSE: all lanes blocked.",
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ALL_FRONTIER_BLOCKED_BY_NEGATIVE_EXPECTANCY",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "scanned_pairs": ["AUD_CAD", "EUR_USD", "USD_CAD"],
                    "scanned_intents": 129,
                    "top_lane": {
                        "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "RANGE_ROTATION",
                        "vehicle": "LIMIT",
                        "status": "NO_TRADE_WITH_CAUSE",
                        "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY",
                        "bidask_status": "PASS",
                        "spread_status": "PASS",
                        "forecast_status": "PASS",
                        "loss_budget_status": "BLOCKED",
                        "blockers": ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                        "next_action": "Preserve negative expectancy and collect exact TP proof.",
                    },
                    "top_non_eurusd_lane": {
                        "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "NO_TRADE_WITH_CAUSE",
                        "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_SPREAD_FORECAST",
                        "bidask_status": "PASS",
                        "spread_status": "BLOCKED",
                        "forecast_status": "BLOCKED",
                        "loss_budget_status": "PASS",
                        "blockers": [
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "SPREAD_TOO_WIDE",
                            "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                        ],
                        "next_action": "Build exact TP-proven rotation proof for USD_CAD; do not hide negative expectancy.",
                    },
                    "required_checks": {
                        "non_eurusd_closer_than_eurusd": False,
                        "spread_too_wide_not_ignored": True,
                        "bidask_negative_not_ignored": False,
                        "next_evidence_lane": {
                            "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "pair": "USD_CAD",
                            "direction": "LONG",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "status": "NO_TRADE_WITH_CAUSE",
                            "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_SPREAD_FORECAST",
                            "bidask_status": "PASS",
                            "spread_status": "BLOCKED",
                            "forecast_status": "BLOCKED",
                            "loss_budget_status": "PASS",
                            "blockers": [
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "SPREAD_TOO_WIDE",
                                "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                            ],
                            "next_action": "Build exact TP-proven rotation proof for USD_CAD; do not hide negative expectancy.",
                        },
                        "usd_cad_long_breakout_failure_blocker_breakdown": [
                            {"lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"}
                        ],
                    },
                    "ranked_frontier_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: preserve negative expectancy and rebuild exact "
                        "TP/bidask proof for failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT."
                    ),
                    "do_not_do": ["do_not_send_live_order"],
                },
            )
            _write_json(
                paths["range_rail"],
                {
                    "schema_version": "range_rail_geometry_repair_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "RANGE_RAIL_RECHECK_BUILT",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "top_lane": {
                        "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "RANGE_ROTATION",
                        "vehicle": "LIMIT",
                        "status": "NO_TRADE_WITH_CAUSE",
                    },
                    "next_contract_prompt": (
                        "Consume data/range_rail_geometry_repair.json for "
                        "range_trader:EUR_USD:SHORT:RANGE_ROTATION: "
                        "next safe action is WAIT_FOR_RANGE_RAIL_RECHECK; do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                range_rail_geometry_repair_path=paths["range_rail"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        frontier = payload["current_state"]["non_eurusd_live_grade_frontier"]
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertIn("non_eurusd_live_grade_frontier", payload["selected_active_path_reason"])
        self.assertIn("NON_EURUSD_LIVE_GRADE_FRONTIER_AVAILABLE", payload["active_deployment_gap"]["active_path_triggers"])
        self.assertEqual(frontier["next_evidence_lane"]["lane_id"], "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT")
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT", payload["next_prompt"])
        self.assertNotIn("WAIT_FOR_RANGE_RAIL_RECHECK", payload["next_prompt"])
        self.assertIn("Use non_eurusd_live_grade_frontier", payload["next_trade_enabling_action"])
        self.assertIn("NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", blocker_codes)
        self.assertIn("SPREAD_TOO_WIDE", blocker_codes)
        self.assertIn("FORECAST_NOT_EXECUTABLE_FOR_LIVE", blocker_codes)
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertIn("Next evidence lane: `failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT`", report)

    def test_local_tp_proof_gap_preserves_evidence_acquisition_status(self) -> None:
        now = datetime(2026, 7, 8, 16, 55, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 121,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 1,
                        "operator_review_required_count": 0,
                        "pairs_scanned": ["USD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": {
                        "lane_id": "range_trader:USD_CAD:LONG:RANGE_ROTATION",
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "RANGE_ROTATION",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": (
                            "Collect exact local TAKE_PROFIT_ORDER proof for "
                            "USD_CAD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER; require positive "
                            "expectancy, zero TP losses, and positive Wilson-stressed expectancy "
                            "before reranking. Do not send."
                        ),
                        "local_tp_proof": {
                            "attach_take_profit_on_fill": True,
                            "capture_take_profit_expectancy_jpy": 658.9,
                            "capture_take_profit_losses": 0,
                            "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                            "capture_take_profit_scope_key": (
                                "USD_CAD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER"
                            ),
                            "capture_take_profit_trades": 1,
                            "capture_take_profit_wins": 1,
                            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                            "tp_target_intent": "HARVEST",
                        },
                        "blockers": [
                            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                            "SPREAD_TOO_WIDE",
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: range_trader:USD_CAD:LONG:RANGE_ROTATION "
                        "needs local TP proof."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        top_lane = payload["current_state"]["active_opportunity_board"]["top_lane"]
        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(top_lane["status"], "EVIDENCE_ACQUISITION")
        self.assertEqual(
            top_lane["local_tp_proof"]["capture_take_profit_scope_key"],
            "USD_CAD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
        )
        self.assertIn("NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", blocker_codes)
        self.assertIn("LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR", blocker_codes)
        self.assertIn("(LIMIT, EVIDENCE_ACQUISITION)", payload["next_trade_enabling_action"])
        self.assertIn(
            "USD_CAD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
            payload["next_trade_enabling_action"],
        )
        self.assertNotIn("NO_TRADE_WITH_CAUSE", payload["next_trade_enabling_action"])

    def test_entry_recovery_board_lane_preserves_evidence_status_despite_negative_blockers(self) -> None:
        now = datetime(2026, 7, 9, 1, 15, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 102,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 1,
                        "operator_review_required_count": 0,
                        "pairs_scanned": ["USD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "MARKET",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": (
                            "Run entry-frequency recovery analysis for "
                            "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET; historical accepted=2, "
                            "fills=2, closed_pl_jpy=664.0852 (exact_lane) but recent entries are zero. "
                            "Do not send."
                        ),
                        "entry_recovery_candidate": True,
                        "entry_recovery_history": {
                            "accepted_before_recent": 2,
                            "fills_before_recent": 2,
                            "closed_trades": 2,
                            "closed_pl_jpy": 664.0852,
                            "profit_source": "exact_lane",
                        },
                        "blockers": [
                            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                            "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET "
                        "needs entry-frequency recovery analysis."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        top_lane = payload["current_state"]["active_opportunity_board"]["top_lane"]
        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(top_lane["status"], "EVIDENCE_ACQUISITION")
        self.assertIn("NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", blocker_codes)
        self.assertIn("ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH", blocker_codes)
        self.assertIn("(MARKET, EVIDENCE_ACQUISITION)", payload["next_trade_enabling_action"])
        self.assertIn("entry-frequency recovery analysis", payload["next_trade_enabling_action"])
        self.assertNotIn("(MARKET, NO_TRADE_WITH_CAUSE)", payload["next_trade_enabling_action"])

    def test_entry_frequency_recovery_artifact_replaces_generic_drought_prompt(self) -> None:
        now = datetime(2026, 7, 9, 2, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 134,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 1,
                        "operator_review_required_count": 0,
                        "pairs_scanned": ["USD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "MARKET",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": (
                            "Run entry-frequency recovery analysis for "
                            "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET; historical accepted=2, "
                            "fills=2, closed_pl_jpy=664.0852 (exact_lane) but recent entries are zero. "
                            "Do not send."
                        ),
                        "entry_recovery_candidate": True,
                        "blockers": [
                            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                            "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET "
                        "needs entry-frequency recovery analysis."
                    ),
                },
            )
            _write_json(
                paths["entry_recovery"],
                {
                    "schema_version": "entry_frequency_recovery_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "target_lane_count": 1,
                    "target_lanes": [lane_id],
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "MARKET",
                        "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                        "forecast_audit": {
                            "status": "FORECAST_PATTERN_REFRESH_REQUIRED",
                            "latest_direction": "RANGE",
                            "blockers": ["RANGE_FORECAST_REQUIRES_RANGE_ROTATION"],
                        },
                        "strategy_profile_audit": {
                            "status": "METHOD_PROFILE_MISSING",
                            "blockers": ["METHOD_SCOPED_STRATEGY_PROFILE_MISSING"],
                        },
                        "tp_proof_audit": {
                            "status": "TP_PROOF_COLLECTION_REQUIRED",
                            "tp_proof_count": 1,
                            "tp_proof_floor": 20,
                            "remaining_samples": 19,
                        },
                    },
                    "forecast_pattern_tuning_queue": [
                        {
                            "priority": 1,
                            "lane_id": lane_id,
                            "action_type": "FORECAST_PATTERN_REFRESH",
                            "description": (
                                "Retune USD_CAD LONG BREAKOUT_FAILURE MARKET around RANGE "
                                "forecast or promote a RANGE_ROTATION path; do not force MARKET."
                            ),
                            "preserve_blockers": [
                                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            ],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/entry_frequency_recovery.json for "
                        "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET: next safe tuning "
                        "action is FORECAST_PATTERN_REFRESH plus METHOD_SCOPED_PROFILE_PROMOTION "
                        "and EXACT_TP_PROOF_COLLECTION; do not send."
                    ),
                    "do_not_do": ["do_not_send_live_order"],
                },
            )
            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                entry_frequency_recovery_path=paths["entry_recovery"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        recovery_state = payload["current_state"]["entry_frequency_recovery"]
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(recovery_state["artifact_status"], "present")
        self.assertEqual(recovery_state["top_lane"]["lane_id"], lane_id)
        self.assertEqual(recovery_state["tuning_queue"][0]["action_type"], "FORECAST_PATTERN_REFRESH")
        self.assertIn("Consume data/entry_frequency_recovery.json", payload["next_prompt"])
        self.assertIn("FORECAST_PATTERN_REFRESH", payload["next_prompt"])
        self.assertIn("METHOD_SCOPED_PROFILE_PROMOTION", payload["next_prompt"])
        self.assertIn("EXACT_TP_PROOF_COLLECTION", payload["next_prompt"])
        self.assertIn("entry_frequency_recovery artifact", payload["next_trade_enabling_action"])
        self.assertNotIn("Run entry-frequency recovery analysis", payload["next_trade_enabling_action"])
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|MARKET", payload["root_improvement_target"])
        self.assertNotIn("Implement EVIDENCE_ACQUISITION", payload["next_prompt"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_forecast_pattern_refresh_artifact_replaces_entry_recovery_prompt(self) -> None:
        now = datetime(2026, 7, 9, 2, 30, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 134,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 1,
                        "operator_review_required_count": 0,
                        "pairs_scanned": ["USD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": "Consume forecast-pattern refresh before repeating drought work. Do not send.",
                        "entry_recovery_candidate": True,
                        "blockers": [
                            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                            "SPREAD_TOO_WIDE",
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                            "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT "
                        "needs forecast-pattern refresh consumption."
                    ),
                },
            )
            _write_json(
                paths["entry_recovery"],
                {
                    "schema_version": "entry_frequency_recovery_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "target_lane_count": 1,
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                    },
                    "forecast_pattern_tuning_queue": [
                        {
                            "priority": 1,
                            "lane_id": lane_id,
                            "action_type": "FORECAST_PATTERN_REFRESH",
                            "description": "Retune around RANGE forecast.",
                            "preserve_blockers": ["RANGE_FORECAST_REQUIRES_RANGE_ROTATION"],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/entry_frequency_recovery.json for "
                        "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT: next safe tuning "
                        "action is FORECAST_PATTERN_REFRESH; do not send."
                    ),
                },
            )
            _write_json(
                paths["forecast_pattern"],
                {
                    "schema_version": "forecast_pattern_refresh_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "FORECAST_PATTERN_REFRESH_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                        "range_rotation_counterpart": {
                            "status": "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL"
                        },
                        "forecast_range_box": {"status": "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL"},
                        "projection_trigger_audit": {
                            "status": "TRIGGER_PROJECTIONS_EXPIRED_PENDING_VERIFICATION_REQUIRED"
                        },
                    },
                    "next_actions": [
                        {
                            "priority": 1,
                            "lane_id": lane_id,
                            "action_type": "RANGE_RAIL_GEOMETRY_REPAIR",
                            "description": "Do not chase the current RANGE midpoint.",
                            "preserve_blockers": ["RANGE_FORECAST_REQUIRES_RANGE_ROTATION"],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/forecast_pattern_refresh.json for "
                        "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT: next safe action is "
                        "RANGE_RAIL_GEOMETRY_REPAIR; do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                entry_frequency_recovery_path=paths["entry_recovery"],
                forecast_pattern_refresh_path=paths["forecast_pattern"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        pattern_state = payload["current_state"]["forecast_pattern_refresh"]
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(pattern_state["artifact_status"], "present")
        self.assertEqual(pattern_state["top_lane"]["lane_id"], lane_id)
        self.assertEqual(pattern_state["top_lane"]["forecast_box_status"], "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL")
        self.assertEqual(pattern_state["next_actions"][0]["action_type"], "RANGE_RAIL_GEOMETRY_REPAIR")
        self.assertIn("Consume data/forecast_pattern_refresh.json", payload["next_prompt"])
        self.assertIn("RANGE_RAIL_GEOMETRY_REPAIR", payload["next_prompt"])
        self.assertNotIn("Consume data/entry_frequency_recovery.json", payload["next_prompt"])
        self.assertIn("forecast_pattern_refresh artifact", payload["next_trade_enabling_action"])
        self.assertNotIn("Consume data/entry_frequency_recovery.json", payload["next_trade_enabling_action"])
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT", payload["root_improvement_target"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_range_rail_geometry_repair_artifact_replaces_forecast_pattern_prompt(self) -> None:
        now = datetime(2026, 7, 9, 2, 45, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            board_top = {
                "lane_id": lane_id,
                "pair": "USD_CAD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "next_action": "Consume range rail repair before repeating forecast-pattern refresh. Do not send.",
                "blockers": [
                    "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                    "SPREAD_TOO_WIDE",
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                ],
            }
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {"evidence_acquisition_count": 1},
                    "top_lane": board_top,
                    "ranked_active_lanes": [],
                },
            )
            _write_json(
                paths["forecast_pattern"],
                {
                    "schema_version": "forecast_pattern_refresh_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "FORECAST_PATTERN_REFRESH_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": {
                        **board_top,
                        "range_rotation_counterpart": {
                            "status": "RANGE_ROTATION_COUNTERPART_BLOCKED_BY_RANGE_RAIL"
                        },
                        "forecast_range_box": {"status": "RANGE_BOX_NOT_AT_EXECUTABLE_RAIL"},
                    },
                    "next_actions": [
                        {
                            "priority": 1,
                            "lane_id": lane_id,
                            "action_type": "RANGE_RAIL_GEOMETRY_REPAIR",
                            "description": "Repair rail geometry.",
                            "preserve_blockers": ["RANGE_FORECAST_REQUIRES_RANGE_ROTATION"],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/forecast_pattern_refresh.json for "
                        f"{lane_id}: next safe action is RANGE_RAIL_GEOMETRY_REPAIR; do not send."
                    ),
                },
            )
            _write_json(
                paths["range_rail"],
                {
                    "schema_version": "range_rail_geometry_repair_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "RANGE_RAIL_RECHECK_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": {
                        **board_top,
                        "range_box": {
                            "rail_status": "RANGE_RAIL_NOT_REACHED",
                            "box_position": 0.7601,
                            "required_zone": "LONG_DISCOUNT_LOWER_RAIL",
                        },
                        "counterpart_geometry": {
                            "status": "COUNTERPART_PRICE_GEOMETRY_INCOMPLETE",
                            "geometry_ready": False,
                        },
                    },
                    "next_actions": [
                        {
                            "priority": 1,
                            "lane_id": lane_id,
                            "action_type": "WAIT_FOR_RANGE_RAIL_RECHECK",
                            "description": "Wait for lower rail.",
                            "preserve_blockers": ["RANGE_RAIL_NOT_REACHED"],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/range_rail_geometry_repair.json for "
                        f"{lane_id}: next safe action is WAIT_FOR_RANGE_RAIL_RECHECK; do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                forecast_pattern_refresh_path=paths["forecast_pattern"],
                range_rail_geometry_repair_path=paths["range_rail"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        rail_state = payload["current_state"]["range_rail_geometry_repair"]
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["target_shape"], "USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT")
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT", payload["four_x_progress_hypothesis"])
        self.assertNotIn("EUR_USD|SHORT|BREAKOUT_FAILURE", payload["four_x_progress_hypothesis"])
        self.assertIn("Target shape: `USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT`", report)
        self.assertEqual(rail_state["artifact_status"], "present")
        self.assertEqual(rail_state["top_lane"]["lane_id"], lane_id)
        self.assertEqual(rail_state["top_lane"]["rail_status"], "RANGE_RAIL_NOT_REACHED")
        self.assertIn("Consume data/range_rail_geometry_repair.json", payload["next_prompt"])
        self.assertIn("WAIT_FOR_RANGE_RAIL_RECHECK", payload["next_prompt"])
        self.assertNotIn("Consume data/forecast_pattern_refresh.json", payload["next_prompt"])
        self.assertIn("range_rail_geometry_repair artifact", payload["next_trade_enabling_action"])
        self.assertNotIn("Consume data/forecast_pattern_refresh.json", payload["next_trade_enabling_action"])
        self.assertNotIn(
            "Consume range rail repair before repeating forecast-pattern refresh",
            payload["next_trade_enabling_action"],
        )
        self.assertIn("USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT", payload["root_improvement_target"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_latest_not_range_rail_repair_rotates_to_non_eurusd_frontier(self) -> None:
        now = datetime(2026, 7, 9, 4, 10, tzinfo=timezone.utc)
        board_lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
        frontier_lane_id = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            board_top = {
                "lane_id": board_lane_id,
                "pair": "EUR_USD",
                "direction": "SHORT",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "next_action": "Run entry-frequency recovery again; do not send.",
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
            }
            frontier_lane = {
                "lane_id": frontier_lane_id,
                "pair": "GBP_USD",
                "direction": "LONG",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "distance_to_live_ready": "2_CLOSE_BUT_BLOCKED_BY_TP_PROOF_FLOOR",
                "tp_proof_count": 4,
                "tp_proof_floor": 20,
                "blockers": [
                    "BROAD_TP_PROOF_NOT_EXACT_VEHICLE",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "SOURCE_EVIDENCE_NOT_EXACT_VEHICLE_ENTRY_TYPE",
                ],
                "next_action": (
                    "Collect 16 exact TP proof sample(s) for "
                    "range_trader:GBP_USD:LONG:RANGE_ROTATION."
                ),
            }
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 126,
                        "evidence_acquisition_count": 17,
                        "live_ready_count": 0,
                    },
                    "top_lane": board_top,
                    "ranked_active_lanes": [board_top],
                },
            )
            _write_json(
                paths["range_rail"],
                {
                    "schema_version": "range_rail_geometry_repair_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "RANGE_RAIL_GEOMETRY_DATA_INCOMPLETE",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": {
                        **board_top,
                        "range_box": {
                            "latest": {"direction": "UP", "confidence": 0.24},
                            "rail_status": "LATEST_FORECAST_NOT_RANGE",
                            "box_position": None,
                            "required_zone": "UNKNOWN",
                        },
                        "counterpart_geometry": {
                            "status": "COUNTERPART_BOX_GEOMETRY_INCOMPLETE",
                            "geometry_ready": False,
                        },
                    },
                    "next_actions": [
                        {
                            "priority": 1,
                            "lane_id": board_lane_id,
                            "action_type": "REFRESH_FORECAST_RANGE_BOX",
                            "description": "Refresh forecast range box.",
                            "preserve_blockers": ["LATEST_FORECAST_NOT_RANGE"],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/range_rail_geometry_repair.json for "
                        f"{board_lane_id}: next safe action is REFRESH_FORECAST_RANGE_BOX; do not send."
                    ),
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "NON_EURUSD_FRONTIER_FOUND",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "scanned_pairs": ["EUR_USD", "GBP_USD"],
                    "scanned_intents": 126,
                    "top_non_eurusd_lane": frontier_lane,
                    "required_checks": {"next_evidence_lane": frontier_lane},
                    "next_active_path": (
                        "TP_PROOF_COLLECTION: collect exact TAKE_PROFIT_ORDER proof for "
                        "range_trader:GBP_USD:LONG:RANGE_ROTATION."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                range_rail_geometry_repair_path=paths["range_rail"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["target_shape"], "GBP_USD|LONG|RANGE_ROTATION|LIMIT")
        self.assertIn("latest forecast is no longer RANGE", payload["next_trade_enabling_action"])
        self.assertIn("Use non_eurusd_live_grade_frontier", payload["next_trade_enabling_action"])
        self.assertIn(frontier_lane_id, payload["next_trade_enabling_action"])
        self.assertNotIn("range_rail_geometry_repair artifact", payload["next_trade_enabling_action"])
        self.assertNotIn("REFRESH_FORECAST_RANGE_BOX", payload["next_trade_enabling_action"])
        self.assertIn("latest forecast is no longer RANGE", payload["next_prompt"])
        self.assertIn(frontier_lane_id, payload["next_prompt"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_latest_not_range_frontier_consumes_entry_recovery_artifact(self) -> None:
        now = datetime(2026, 7, 10, 4, 40, tzinfo=timezone.utc)
        board_lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
        frontier_lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            board_top = {
                "lane_id": board_lane_id,
                "pair": "EUR_USD",
                "direction": "SHORT",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "next_action": "Run entry-frequency recovery again; do not send.",
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
            }
            frontier_lane = {
                "lane_id": frontier_lane_id,
                "pair": "USD_CAD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "MARKET",
                "status": "EVIDENCE_ACQUISITION",
                "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_FORECAST_TP_PROOF_FLOOR",
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
                "next_action": (
                    "Run entry-frequency recovery analysis for "
                    "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET; do not send."
                ),
            }
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {"total_lanes": 116, "evidence_acquisition_count": 14, "live_ready_count": 0},
                    "top_lane": board_top,
                    "ranked_active_lanes": [board_top],
                },
            )
            _write_json(
                paths["range_rail"],
                {
                    "schema_version": "range_rail_geometry_repair_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "RANGE_RAIL_GEOMETRY_DATA_INCOMPLETE",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": {
                        **board_top,
                        "range_box": {
                            "latest": {"direction": "UP", "confidence": 0.24},
                            "rail_status": "LATEST_FORECAST_NOT_RANGE",
                        },
                    },
                    "next_contract_prompt": (
                        "Consume data/range_rail_geometry_repair.json for "
                        f"{board_lane_id}: next safe action is REFRESH_FORECAST_RANGE_BOX; do not send."
                    ),
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "NON_EURUSD_FRONTIER_FOUND",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_non_eurusd_lane": frontier_lane,
                    "required_checks": {"next_evidence_lane": frontier_lane},
                    "next_active_path": "ENTRY_FREQUENCY_RECOVERY: run USD_CAD recovery; do not send.",
                },
            )
            _write_json(
                paths["entry_recovery"],
                {
                    "schema_version": "entry_frequency_recovery_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "target_lane_count": 1,
                    "top_lane": {
                        **frontier_lane,
                        "forecast_audit": {"status": "SIDE_SUPPORTED_BY_LATEST_FORECAST"},
                        "strategy_profile_audit": {"status": "PAIR_SIDE_PROFILE_PRESENT_METHOD_PROFILE_MISSING"},
                        "tp_proof_audit": {"status": "TP_PROOF_FLOOR_GAP"},
                    },
                    "forecast_pattern_tuning_queue": [
                        {
                            "priority": 2,
                            "lane_id": frontier_lane_id,
                            "action_type": "TRIGGER_PROJECTION_TO_LIMIT_PROOF",
                            "description": "verify expired projections, then retune the pattern signal",
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/entry_frequency_recovery.json for "
                        f"{frontier_lane_id}: next safe action is TRIGGER_PROJECTION_TO_LIMIT_PROOF, "
                        "METHOD_SCOPED_PROFILE_PROMOTION, and EXACT_TP_PROOF_COLLECTION; do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                entry_frequency_recovery_path=paths["entry_recovery"],
                range_rail_geometry_repair_path=paths["range_rail"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["target_shape"], "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET")
        self.assertIn("latest forecast is no longer RANGE", payload["next_trade_enabling_action"])
        self.assertIn("Consume data/entry_frequency_recovery.json", payload["next_trade_enabling_action"])
        self.assertIn("TRIGGER_PROJECTION_TO_LIMIT_PROOF", payload["next_trade_enabling_action"])
        self.assertNotIn("Run entry-frequency recovery analysis", payload["next_trade_enabling_action"])
        self.assertIn("Consume data/entry_frequency_recovery.json", payload["next_prompt"])
        self.assertNotIn("REFRESH_FORECAST_RANGE_BOX", payload["next_trade_enabling_action"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_guardian_range_rail_trigger_advances_wait_recheck_prompt(self) -> None:
        now = datetime(2026, 7, 9, 3, 5, tzinfo=timezone.utc)
        lane_id = "range_trader:EUR_JPY:SHORT:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            board_top = {
                "lane_id": lane_id,
                "pair": "EUR_JPY",
                "direction": "SHORT",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "next_action": "Build exact TP-proven rotation proof; do not send.",
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "FORECAST_WATCH_ONLY",
                    "LOCAL_TP_PROOF_ZERO_TRADES",
                ],
            }
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {"total_lanes": 93, "evidence_acquisition_count": 1},
                    "top_lane": board_top,
                    "ranked_active_lanes": [],
                },
            )
            _write_json(
                paths["range_rail"],
                {
                    "schema_version": "range_rail_geometry_repair_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "RANGE_RAIL_RECHECK_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "top_lane": {
                        **board_top,
                        "range_box": {
                            "rail_status": "RANGE_RAIL_NOT_REACHED",
                            "box_position": 0.5707,
                            "required_zone": "SHORT_PREMIUM_UPPER_RAIL",
                        },
                        "counterpart_geometry": {
                            "status": "COUNTERPART_GEOMETRY_READY",
                            "geometry_ready": True,
                        },
                    },
                    "next_actions": [
                        {
                            "priority": 1,
                            "lane_id": lane_id,
                            "action_type": "WAIT_FOR_RANGE_RAIL_RECHECK",
                            "description": "Wait for upper rail.",
                            "preserve_blockers": ["RANGE_RAIL_NOT_REACHED"],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/range_rail_geometry_repair.json for "
                        f"{lane_id}: next safe action is WAIT_FOR_RANGE_RAIL_RECHECK; do not send."
                    ),
                },
            )
            _write_json(
                paths["guardian_events"],
                {
                    "schema_version": 1,
                    "generated_at_utc": now.isoformat(),
                    "events": [
                        {
                            "event_id": "evt-range-fired",
                            "event_type": "CONTRACT_ADD_TRIGGER",
                            "pair": "EUR_JPY",
                            "direction": "SHORT",
                            "thesis": "EUR_JPY short range rail rotation",
                            "price_zone": "mid >= 185.6567 fired with actual=185.7125",
                            "severity": "P1",
                            "recommended_review_type": "ADD_REVIEW",
                            "dedupe_key": "EUR_JPY|RANGE_RAIL|CONTRACT_ADD_TRIGGER|ADD",
                            "action_hint": "ADD",
                            "details": {
                                "contract_trigger": {
                                    "kind": "range_rail_recheck",
                                    "lane_id": lane_id,
                                    "pair": "EUR_JPY",
                                    "side": "SHORT",
                                    "action_hint": "ADD",
                                    "live_permission_allowed": False,
                                    "contract_triggers_do_not_execute": True,
                                    "condition": {"metric": "mid", "operator": ">=", "value": 185.6567},
                                    "preserve_blockers": [
                                        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                        "FORECAST_WATCH_ONLY",
                                    ],
                                }
                            },
                        }
                    ],
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                range_rail_geometry_repair_path=paths["range_rail"],
                guardian_events_path=paths["guardian_events"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertIn("CONTRACT_ADD_TRIGGER fired", payload["next_prompt"])
        self.assertIn("Do not repeat WAIT_FOR_RANGE_RAIL_RECHECK", payload["next_prompt"])
        self.assertIn("guardian_events artifact", payload["next_trade_enabling_action"])
        self.assertIn("reprice the RANGE_ROTATION counterpart", payload["next_trade_enabling_action"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_guardian_range_rail_trigger_advances_parallel_frontier_wait(self) -> None:
        now = datetime(2026, 7, 10, 0, 50, tzinfo=timezone.utc)
        board_lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
        frontier_lane_id = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            board_top = {
                "lane_id": board_lane_id,
                "pair": "EUR_USD",
                "direction": "SHORT",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "next_action": "Collect exact EUR_USD TP proof; do not send.",
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                ],
            }
            frontier_lane = {
                "lane_id": frontier_lane_id,
                "pair": "GBP_USD",
                "direction": "LONG",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_FORECAST_TP_PROOF_FLOOR",
                "next_action": "Wait for the GBP_USD lower range rail; do not send.",
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "RANGE_ROTATION_BROADER_LOCATION_CHASE",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                ],
            }
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 129,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 2,
                    },
                    "top_lane": board_top,
                    "ranked_active_lanes": [board_top],
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "NON_EURUSD_FRONTIER_FOUND",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "scanned_pairs": ["EUR_USD", "GBP_USD"],
                    "scanned_intents": 129,
                    "top_lane": board_top,
                    "top_non_eurusd_lane": frontier_lane,
                    "required_checks": {
                        "non_eurusd_closer_than_eurusd": False,
                        "spread_too_wide_not_ignored": False,
                        "bidask_negative_not_ignored": False,
                        "next_evidence_lane": frontier_lane,
                    },
                    "next_active_path": "WAIT_FOR_RANGE_RAIL_RECHECK for GBP_USD; do not send.",
                },
            )
            _write_json(
                paths["range_rail"],
                {
                    "schema_version": "range_rail_geometry_repair_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "RANGE_RAIL_RECHECK_BUILT",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "top_lane": {
                        **frontier_lane,
                        "range_box": {
                            "rail_status": "RANGE_RAIL_NOT_REACHED",
                            "box_position": 0.6667,
                            "required_zone": "LONG_DISCOUNT_LOWER_RAIL",
                        },
                        "counterpart_geometry": {
                            "status": "COUNTERPART_PRICE_GEOMETRY_INCOMPLETE",
                            "geometry_ready": False,
                        },
                    },
                    "next_actions": [
                        {
                            "priority": 1,
                            "lane_id": frontier_lane_id,
                            "action_type": "WAIT_FOR_RANGE_RAIL_RECHECK",
                            "description": "Wait for the lower rail.",
                            "preserve_blockers": ["RANGE_RAIL_NOT_REACHED"],
                        }
                    ],
                    "next_contract_prompt": (
                        "Consume data/range_rail_geometry_repair.json for "
                        f"{frontier_lane_id}: next safe action is WAIT_FOR_RANGE_RAIL_RECHECK; do not send."
                    ),
                },
            )
            _write_json(
                paths["guardian_events"],
                {
                    "schema_version": 1,
                    "generated_at_utc": now.isoformat(),
                    "events": [
                        {
                            "event_id": "evt-gbp-range-fired",
                            "event_type": "CONTRACT_ADD_TRIGGER",
                            "pair": "GBP_USD",
                            "direction": "LONG",
                            "thesis": "GBP_USD long range rail rotation",
                            "price_zone": "mid <= 1.3406055 fired with actual=1.3405",
                            "severity": "P1",
                            "recommended_review_type": "ADD_REVIEW",
                            "dedupe_key": "GBP_USD|RANGE_RAIL|CONTRACT_ADD_TRIGGER|ADD",
                            "action_hint": "ADD",
                            "details": {
                                "contract_trigger": {
                                    "kind": "range_rail_recheck",
                                    "lane_id": frontier_lane_id,
                                    "pair": "GBP_USD",
                                    "side": "LONG",
                                    "action_hint": "ADD",
                                    "live_permission_allowed": False,
                                    "contract_triggers_do_not_execute": True,
                                    "condition": {"metric": "mid", "operator": "<=", "value": 1.3406055},
                                    "preserve_blockers": [
                                        "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                        "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                                    ],
                                }
                            },
                        }
                    ],
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                range_rail_geometry_repair_path=paths["range_rail"],
                guardian_events_path=paths["guardian_events"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        for field in ("next_prompt", "next_trade_enabling_action"):
            text = payload[field]
            self.assertIn(frontier_lane_id, text)
            self.assertIn("guardian_events", text)
            self.assertIn("CONTRACT_ADD_TRIGGER fired", text)
            self.assertIn("Do not repeat WAIT_FOR_RANGE_RAIL_RECHECK", text)
            self.assertIn("reprice the RANGE_ROTATION counterpart", text)
            self.assertIn("continue exact TP-proof collection", text)
            self.assertNotIn("next safe action is WAIT_FOR_RANGE_RAIL_RECHECK", text)
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_same_shape_non_eurusd_frontier_supplements_entry_recovery_board_lane(self) -> None:
        now = datetime(2026, 7, 9, 2, 5, tzinfo=timezone.utc)
        board_lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        frontier_lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 134,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 2,
                        "operator_review_required_count": 0,
                        "pairs_scanned": ["EUR_USD", "USD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": {
                        "lane_id": board_lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "MARKET",
                        "status": "EVIDENCE_ACQUISITION",
                        "next_action": (
                            "Run entry-frequency recovery analysis for "
                            "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET; historical accepted=2, "
                            "fills=2, closed_pl_jpy=664.0852 (exact_lane) but recent entries are zero. "
                            "Do not send."
                        ),
                        "entry_recovery_candidate": True,
                        "blockers": [
                            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "HARVEST_TP_STRUCTURE_MISSING",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                            "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                        ],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET "
                        "is the closest read-only path."
                    ),
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "ONLY_EURUSD_FRONTIER_FOUND",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "scanned_pairs": ["EUR_USD", "USD_CAD"],
                    "scanned_intents": 114,
                    "top_lane": {
                        "lane_id": "range_trader:EUR_USD:SHORT:RANGE_ROTATION",
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "RANGE_ROTATION",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "distance_to_live_ready": "2_CLOSE_BUT_BLOCKED_BY_NEGATIVE_EXPECTANCY_AND_TP_PROOF_FLOOR",
                        "bidask_status": "PASS",
                        "spread_status": "PASS",
                        "forecast_status": "PASS",
                        "loss_budget_status": "PASS",
                        "blockers": ["NEGATIVE_EXPECTANCY_ACTIVE"],
                    },
                    "top_non_eurusd_lane": {
                        "lane_id": frontier_lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_FORECAST_TP_PROOF_FLOOR",
                        "bidask_status": "PASS",
                        "spread_status": "PASS",
                        "forecast_status": "BLOCKED",
                        "loss_budget_status": "PASS",
                        "tp_proof_count": 1,
                        "tp_proof_floor": 20,
                        "blockers": [
                            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                        ],
                        "next_action": (
                            "Build exact TP-proven rotation proof for "
                            "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT; do not hide negative expectancy."
                        ),
                    },
                    "required_checks": {
                        "non_eurusd_closer_than_eurusd": False,
                        "spread_too_wide_not_ignored": False,
                        "bidask_negative_not_ignored": False,
                        "next_evidence_lane": {
                            "lane_id": frontier_lane_id,
                            "pair": "USD_CAD",
                            "direction": "LONG",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "status": "EVIDENCE_ACQUISITION",
                            "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_FORECAST_TP_PROOF_FLOOR",
                            "bidask_status": "PASS",
                            "spread_status": "PASS",
                            "forecast_status": "BLOCKED",
                            "loss_budget_status": "PASS",
                            "tp_proof_count": 1,
                            "tp_proof_floor": 20,
                            "blockers": [
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
                                "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                            ],
                            "next_action": (
                                "Build exact TP-proven rotation proof for "
                                "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT; do not hide negative expectancy."
                            ),
                        },
                    },
                    "next_active_path": (
                        "TP_PROOF_COLLECTION: collect exact TAKE_PROFIT_ORDER proof for "
                        "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT; do not mix market-close losses."
                    ),
                    "do_not_do": ["do_not_send_live_order"],
                },
            )
            _write_json(
                paths["range_rail"],
                {
                    "schema_version": "range_rail_geometry_repair_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "RANGE_RAIL_RECHECK_BUILT",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "top_lane": {
                        "lane_id": board_lane_id,
                        "pair": "USD_CAD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "MARKET",
                        "status": "EVIDENCE_ACQUISITION",
                    },
                    "next_contract_prompt": (
                        "Consume data/range_rail_geometry_repair.json for "
                        f"{board_lane_id}: next safe action is WAIT_FOR_RANGE_RAIL_RECHECK; do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                range_rail_geometry_repair_path=paths["range_rail"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertIn("Non-EUR frontier points to the same pair/side/family", payload["selected_active_path_reason"])
        self.assertIn(board_lane_id, payload["next_trade_enabling_action"])
        self.assertIn(frontier_lane_id, payload["next_trade_enabling_action"])
        self.assertIn("Pair this with non_eurusd_live_grade_frontier", payload["next_trade_enabling_action"])
        self.assertIn(board_lane_id, payload["next_prompt"])
        self.assertIn("WAIT_FOR_RANGE_RAIL_RECHECK", payload["next_prompt"])
        self.assertIn("frontier evidence USD_CAD|LONG|BREAKOUT_FAILURE|LIMIT", payload["next_prompt"])
        self.assertIn("FORECAST_NOT_EXECUTABLE_FOR_LIVE", blocker_codes)
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_parallel_non_eurusd_frontier_is_not_hidden_behind_eurusd_board_lane(self) -> None:
        now = datetime(2026, 7, 10, 5, 30, tzinfo=timezone.utc)
        board_lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        frontier_lane_id = "range_trader:AUD_CAD:SHORT:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            board_top = {
                "lane_id": board_lane_id,
                "pair": "EUR_USD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "expected_edge_jpy": 1146.7898,
                "next_action": (
                    "Collect exact local TAKE_PROFIT_ORDER proof for "
                    "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT|TAKE_PROFIT_ORDER; do not send."
                ),
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                ],
            }
            frontier_lane = {
                "lane_id": frontier_lane_id,
                "pair": "AUD_CAD",
                "direction": "SHORT",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "distance_to_live_ready": "3_MULTI_GATE_BLOCKED_NEGATIVE_EXPECTANCY_BIDASK_REPLAY_TP_PROOF_FLOOR",
                "expected_edge_jpy": 0,
                "bidask_status": "NEGATIVE",
                "spread_status": "PASS",
                "forecast_status": "PASS",
                "loss_budget_status": "PASS",
                "tp_proof_count": 0,
                "tp_proof_floor": 20,
                "blockers": [
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                    "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED",
                    "LOCAL_TP_PROOF_ZERO_TRADES",
                ],
                "next_action": (
                    "Repair bid/ask-negative pattern or vehicle shape for "
                    "range_trader:AUD_CAD:SHORT:RANGE_ROTATION; do not repeat replay until inputs change."
                ),
            }
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 127,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 21,
                        "pairs_scanned": ["EUR_USD", "AUD_CAD"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": board_top,
                    "ranked_active_lanes": [board_top],
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "schema_version": "non_eurusd_live_grade_frontier_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "NON_EURUSD_FRONTIER_FOUND",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "scanned_pairs": ["EUR_USD", "AUD_CAD"],
                    "scanned_intents": 113,
                    "top_lane": board_top,
                    "top_non_eurusd_lane": frontier_lane,
                    "required_checks": {
                        "non_eurusd_closer_than_eurusd": False,
                        "spread_too_wide_not_ignored": True,
                        "bidask_negative_not_ignored": True,
                        "next_evidence_lane": frontier_lane,
                    },
                    "next_active_path": (
                        "BIDASK_NEGATIVE_PATTERN_REPAIR: current exact bid/ask replay is negative "
                        "for range_trader:AUD_CAD:SHORT:RANGE_ROTATION; do not send."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blocker_codes = {row["code"] for row in payload["remaining_blockers"]}
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(
            payload["target_shape"],
            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT plus frontier evidence AUD_CAD|SHORT|RANGE_ROTATION|LIMIT",
        )
        self.assertIn("EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT", payload["four_x_progress_hypothesis"])
        self.assertIn("AUD_CAD|SHORT|RANGE_ROTATION|LIMIT", payload["four_x_progress_hypothesis"])
        self.assertIn("AUD_CAD|SHORT|RANGE_ROTATION|LIMIT", payload["root_improvement_target"])
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", blocker_codes)
        self.assertIn("LOCAL_TP_PROOF_ZERO_TRADES", blocker_codes)
        self.assertIn(frontier_lane_id, payload["next_trade_enabling_action"])
        self.assertIn("Parallel non_eurusd_live_grade_frontier", payload["next_trade_enabling_action"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_edge_improvement_board_lane_preserves_evidence_status_despite_negative_blockers(self) -> None:
        now = datetime(2026, 7, 8, 17, 20, tzinfo=timezone.utc)
        lane_id = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["active_board"],
                {
                    "schema_version": "active_opportunity_board_v1",
                    "generated_at_utc": now.isoformat(),
                    "status": "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY",
                    "read_only": True,
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                    "coverage_summary": {
                        "total_lanes": 118,
                        "live_ready_count": 0,
                        "harvest_ready_count": 0,
                        "scout_ready_count": 0,
                        "evidence_acquisition_count": 1,
                        "operator_review_required_count": 0,
                        "pairs_scanned": ["EUR_USD", "EUR_JPY"],
                        "vehicles_scanned": ["LIMIT", "MARKET", "STOP"],
                    },
                    "top_lane": {
                        "lane_id": lane_id,
                        "pair": "EUR_USD",
                        "direction": "SHORT",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "EVIDENCE_ACQUISITION",
                        "edge_improvement_candidate": True,
                        "edge_improvement_target": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                        "next_action": (
                            "Run read-only EDGE_IMPROVEMENT_EXPERIMENT for "
                            "EUR_USD|SHORT|BREAKOUT_FAILURE; preserve negative/month-scale "
                            "blockers, canonicalize proof/replay/sample gaps, rerank, and do not send."
                        ),
                        "blockers": [
                            "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
                            "PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY",
                            "NEGATIVE_EXPECTANCY_ACTIVE",
                            "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                            "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED",
                        ],
                        "stale_source_blockers": ["NOT_IN_PROOF_QUEUE"],
                    },
                    "ranked_active_lanes": [],
                    "next_active_path": (
                        "EVIDENCE_ACQUISITION: failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT "
                        "is the closest read-only path."
                    ),
                },
            )

            ActiveTraderContract(
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                scout_plan_path=paths["scout"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                daily_target_state_path=paths["daily"],
                proof_floor_update_path=paths["proof_floor"],
                limit_s5_bidask_replay_path=paths["replay"],
                limit_sample_mining_path=paths["mining"],
                active_opportunity_board_path=paths["active_board"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        top_lane = payload["current_state"]["active_opportunity_board"]["top_lane"]
        self.assertEqual(payload["selected_active_path"], "EVIDENCE_ACQUISITION")
        self.assertEqual(top_lane["status"], "EVIDENCE_ACQUISITION")
        self.assertTrue(top_lane["edge_improvement_candidate"])
        self.assertIn("top lane failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT is EVIDENCE_ACQUISITION", payload["selected_active_path_reason"])
        self.assertIn("(LIMIT, EVIDENCE_ACQUISITION)", payload["next_trade_enabling_action"])
        self.assertIn("EDGE_IMPROVEMENT_EXPERIMENT", payload["next_trade_enabling_action"])
        self.assertNotIn("(LIMIT, NO_TRADE_WITH_CAUSE)", payload["next_trade_enabling_action"])


def _write_base_artifacts(root: Path, *, now: datetime) -> dict[str, Path]:
    paths = {
        "goal_loop": root / "data" / "trader_goal_loop_orchestrator.json",
        "payoff": root / "data" / "payoff_shape_diagnosis.json",
        "harvest": root / "data" / "harvest_live_grade_path.json",
        "scout": root / "data" / "eurusd_short_breakout_failure_scout_plan.json",
        "proof": root / "data" / "as_proof_pack_queue.json",
        "board": root / "data" / "as_lane_candidate_board.json",
        "portfolio": root / "data" / "portfolio_4x_path_planner.json",
        "live_order": root / "data" / "live_order_request.json",
        "broker": root / "data" / "broker_snapshot.json",
        "daily": root / "data" / "daily_target_state.json",
        "proof_floor": root / "data" / "eurusd_short_breakout_failure_proof_floor_update.json",
        "replay": root / "data" / "eurusd_short_breakout_failure_limit_s5_bidask_replay.json",
        "mining": root / "data" / "eurusd_short_breakout_failure_limit_sample_mining.json",
        "active_board": root / "data" / "active_opportunity_board.json",
        "frontier": root / "data" / "non_eurusd_live_grade_frontier.json",
        "entry_recovery": root / "data" / "entry_frequency_recovery.json",
        "forecast_pattern": root / "data" / "forecast_pattern_refresh.json",
        "range_rail": root / "data" / "range_rail_geometry_repair.json",
        "guardian_events": root / "data" / "guardian_events.json",
        "output": root / "data" / "active_trader_contract.json",
        "report": root / "docs" / "active_trader_contract.md",
    }
    _write_json(
        paths["goal_loop"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "NEXT_WORK_SELECTED",
            "selected_next_work_type": "EDGE_IMPROVEMENT_EXPERIMENT",
            "four_x_progress_hypothesis": "existing goal loop",
            "root_improvement_target": "HARVEST live-grade",
            "expected_edge_improvement": "evidence quality",
            "live_permission_allowed": False,
        },
    )
    _write_json(paths["payoff"], {"generated_at_utc": now.isoformat(), "status": "OK"})
    _write_json(
        paths["harvest"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "DIAGNOSIS_COMPLETE_BLOCKED_NO_LIVE_PERMISSION",
            "live_promotion_allowed": False,
            "closest_harvest_candidate": {
                "candidate_id": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                "shape_key": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                "actual_proof_queue_member": False,
                "planner_can_enter_proof_pack": False,
                "can_create_live_permission": False,
                "live_promotion_allowed": False,
                "promotion_blockers": [
                    "SAMPLE_GAP",
                    "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED",
                    "GUARDIAN_OPERATOR_REVIEW_BLOCK",
                ],
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
                    "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                    "tp_target_intent": "HARVEST",
                },
            },
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
            "proof_queue_entry_blockers": [{"code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"}],
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
            },
            "queue": [],
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
                    "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                ],
            },
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["portfolio"],
        {
            "generated_at_utc": now.isoformat(),
            "portfolio_status": "NO_LIVE_READY_PORTFOLIO",
            "can_reach_4x_now": False,
            "summary": {"can_create_live_permission": False, "live_ready_lanes": 0},
            "live_side_effects": [],
        },
    )
    _write_json(paths["live_order"], {"generated_at_utc": now.isoformat(), "status": "NO_ACTION", "send_requested": False, "sent": False})
    _write_json(paths["broker"], {"fetched_at_utc": now.isoformat(), "account": {}, "positions": [], "orders": []})
    _write_json(
        paths["daily"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "PURSUE_TARGET",
            "target_jpy": 29_114.51,
            "funding_adjusted_equity": 100000,
            "target_trades_per_day": 30,
            "uncapped_required_trades_per_day": 173,
            "uncapped_required_trades_per_day_basis_return_pct": 10.0,
            "selected_basis_uncapped_required_trades_per_day": 87,
            "selected_basis_return_pct": 5.0,
            "operating_pace_trades_per_day": 30,
            "automated_operating_cap_trades_per_day": 30,
            "observed_trades_per_day": 4.7955,
            "observed_expectancy_jpy_per_trade": 168.6658,
            "frequency_multiple_required": 36.0755,
            "planned_reward_at_operating_pace_jpy": 5059.974,
            "stretch_required_minus_operating_gap_trades_per_day": 143,
            "selected_required_minus_operating_gap_trades_per_day": 57,
            "trade_pace_feasible_within_operating_pace": False,
            "trade_pace_feasibility": "INFEASIBLE_AT_OPERATING_PACE",
        },
    )
    _write_json(
        paths["proof_floor"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "PROOF_FLOOR_REACHED_STILL_BLOCKED",
            "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE",
            "post_update_tp_proof": {
                "wins": 20,
                "losses": 0,
                "proof_floor": 20,
                "remaining_samples": 0,
                "proof_floor_reached": True,
            },
            "remaining_blockers": [
                {"code": "NEGATIVE_EXPECTANCY_ACTIVE"},
                {"code": "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"},
            ],
            "live_permission_allowed": False,
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["replay"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "LIMIT_S5_BIDASK_REPLAY_PASSED_STILL_BLOCKED",
            "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST",
            "read_only": True,
            "live_permission_allowed": False,
            "live_side_effects": [],
            "s5_bidask_replay_status": "PASSED_WITH_S5_TOUCH_LAG_4_OF_4_LIMIT_SAMPLES_STILL_UNDERSAMPLED_AND_BLOCKED",
            "replay_sample_count": 4,
            "replay_wins": 4,
            "replay_losses": 0,
            "net_expectancy_after_bidask": 813.7734,
            "market_stop_samples_excluded": True,
            "market_close_excluded": True,
            "live_grade_candidate": False,
            "remaining_blockers": [
                {"code": "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY"},
                {"code": "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION"},
                {"code": "NOT_IN_PROOF_QUEUE"},
            ],
            "next_read_only_actions": [
                "Reconcile/import accepted legacy LIMIT rows.",
                "Mine additional exact LIMIT samples.",
            ],
        },
    )
    _write_json(
        paths["mining"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED_STILL_UNDERSAMPLED",
            "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST",
            "read_only": True,
            "live_permission_allowed": False,
            "live_side_effects": [],
            "sample_floor": {
                "required_exact_limit_samples": 20,
                "current_replayed_exact_limit_samples": 4,
                "additional_acceptable_local_samples_found": 0,
                "total_exact_limit_samples_after_local_mining": 4,
                "remaining_exact_limit_samples": 16,
                "floor_met": False,
            },
            "execution_ledger_coverage": {
                "summary": {
                    "accepted_current_replay": 1,
                    "acceptable_new_exact_limit_samples": 0,
                }
            },
            "legacy_history_coverage": {
                "summary": {
                    "accepted_current_replay": 3,
                    "acceptable_new_exact_limit_samples": 0,
                }
            },
            "remaining_blockers": [
                {"code": "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY"},
                {"code": "LOCAL_LIMIT_SAMPLE_COVERAGE_EXHAUSTED"},
            ],
        },
    )
    return paths


if __name__ == "__main__":
    unittest.main()
