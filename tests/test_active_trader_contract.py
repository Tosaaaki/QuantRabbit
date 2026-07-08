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
    _write_json(paths["daily"], {"generated_at_utc": now.isoformat(), "funding_adjusted_equity": 100000})
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
