from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.active_opportunity_board import ActiveOpportunityBoard


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class ActiveOpportunityBoardTest(unittest.TestCase):
    def test_board_scans_multi_pair_multi_vehicle_and_selects_evidence_path(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)

            summary = ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        self.assertEqual(summary.status, "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY")
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertEqual(summary.top_lane_id, "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT")
        self.assertEqual(payload["top_lane"]["status"], "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["top_lane"]["vehicle"], "LIMIT")
        self.assertIn("EUR_USD", payload["coverage_summary"]["pairs_scanned"])
        self.assertIn("AUD_JPY", payload["coverage_summary"]["pairs_scanned"])
        self.assertIn("GBP_USD", payload["coverage_summary"]["pairs_scanned"])
        self.assertIn("LIMIT", payload["coverage_summary"]["vehicles_scanned"])
        self.assertIn("STOP", payload["coverage_summary"]["vehicles_scanned"])
        self.assertIn("MARKET", payload["coverage_summary"]["vehicles_scanned"])
        self.assertGreaterEqual(payload["coverage_summary"]["total_lanes"], 4)
        self.assertGreaterEqual(payload["coverage_summary"]["evidence_acquisition_count"], 1)
        self.assertIn("failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT", payload["next_active_path"])
        self.assertIn("Active Opportunity Board", report)

    def test_operator_review_precedes_evidence_when_guardian_blocks_lane(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            order_intents = json.loads(paths["order_intents"].read_text())
            for row in order_intents["results"]:
                row["live_blocker_codes"].append("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED")
            _write_json(paths["order_intents"], order_intents)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["status"], "OPERATOR_REVIEW_REQUIRED")
        self.assertEqual(payload["top_lane"]["operator_review_status"], "REQUIRED")
        self.assertGreaterEqual(payload["coverage_summary"]["operator_review_required_count"], 4)
        self.assertIn("OPERATOR_REVIEW_REQUIRED", payload["next_active_path"])

    def test_stale_order_intent_guardian_blocker_is_suppressed_after_consumption_refresh(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            order_intents = json.loads(paths["order_intents"].read_text())
            order_intents["generated_at_utc"] = (now - timedelta(minutes=15)).isoformat()
            for row in order_intents["results"]:
                row["live_blocker_codes"].append("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED")
            _write_json(paths["order_intents"], order_intents)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["status"], "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["top_lane"]["operator_review_status"], "NOT_REQUIRED")
        self.assertNotIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", payload["top_lane"]["blockers"])
        self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", payload["top_lane"]["stale_source_blockers"])

    def test_cleared_guardian_receipt_suppresses_stale_planner_guardian_blocker(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            portfolio = json.loads(paths["portfolio"].read_text())
            portfolio["candidate_rankings"][0]["current_blockers"] = [
                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION",
            ]
            _write_json(paths["portfolio"], portfolio)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        lane = next(
            row
            for row in payload["ranked_active_lanes"]
            if row["lane_id"] == "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
        )
        self.assertEqual(lane["status"], "EVIDENCE_ACQUISITION")
        self.assertNotIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", lane["blockers"])
        self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", lane["stale_source_blockers"])

    def test_current_intent_owned_self_improvement_blocker_is_stale_when_absent_from_current_intents(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            portfolio = json.loads(paths["portfolio"].read_text())
            portfolio["candidate_rankings"][0]["current_blockers"] = [
                "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH",
                "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION",
            ]
            _write_json(paths["portfolio"], portfolio)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        lane = next(
            row
            for row in payload["ranked_active_lanes"]
            if row["lane_id"] == "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
        )
        self.assertEqual(lane["status"], "EVIDENCE_ACQUISITION")
        self.assertNotIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", lane["blockers"])
        self.assertIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", lane["stale_source_blockers"])
        self.assertIn("S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION", lane["blockers"])
        stale_reasons = {row["code"]: row for row in payload["stale_source_reasons"]}
        self.assertIn("SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH", stale_reasons)

    def test_order_intent_warn_issues_do_not_become_board_blockers(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            order_intents = json.loads(paths["order_intents"].read_text())
            order_intents["results"][0]["risk_issues"].append(
                {
                    "code": "SELF_IMPROVEMENT_P0_PROFITABILITY_REPAIR_MODE",
                    "message": "repair marker is diagnostic and must not block board ranking",
                    "severity": "WARN",
                }
            )
            _write_json(paths["order_intents"], order_intents)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        lane = next(
            row
            for row in payload["ranked_active_lanes"]
            if row["lane_id"] == "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
        )
        self.assertNotIn("SELF_IMPROVEMENT_P0_PROFITABILITY_REPAIR_MODE", lane["blockers"])

    def test_no_trade_ranking_prefers_current_intent_with_fewest_hard_blockers(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            close_lane = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            far_lane = "trend_trader:USD_CHF:LONG:TREND_CONTINUATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            close_lane,
                            "EUR_USD",
                            "LONG",
                            "LIMIT",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        ),
                        _intent_row(
                            far_lane,
                            "USD_CHF",
                            "LONG",
                            "STOP-ENTRY",
                            blockers=[
                                "LOSS_BUDGET_TOO_THIN_FOR_MIN_LOT",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "EXHAUSTION_RANGE_CHASE",
                                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                                "STRATEGY_NOT_ELIGIBLE",
                                "BAD_UNITS",
                                "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                                "spread_included_bidask_replay_negative_for_exact_lane",
                                "packaged_bidask_rule_live_block_negative_expectancy",
                            ],
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(
                paths["portfolio"],
                {
                    "candidate_rankings": [
                        {
                            "lane_id": far_lane,
                            "pair": "USD_CHF",
                            "side": "LONG",
                            "method": "TREND_CONTINUATION",
                            "order_type": "STOP-ENTRY",
                            "expected_jpy_per_trade": 2263.4,
                            "rank_score": 200.0,
                            "current_blockers": [],
                            "math_exclusion_reasons": [],
                        }
                    ],
                    "summary": {"can_create_live_permission": False},
                },
            )
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(
                paths["harvest"],
                {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False},
            )

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], close_lane)
        self.assertEqual(payload["top_lane"]["blockers"], ["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"])

    def test_no_trade_ranking_prefers_current_non_negative_unlock_before_bidask_negative(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            bidask_lane = "range_trader:EUR_JPY:SHORT:RANGE_ROTATION"
            unlock_lane = "range_trader:GBP_USD:SHORT:RANGE_ROTATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            bidask_lane,
                            "EUR_JPY",
                            "SHORT",
                            "LIMIT",
                            blockers=[
                                "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
                                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                            ],
                        ),
                        _intent_row(
                            unlock_lane,
                            "GBP_USD",
                            "SHORT",
                            "LIMIT",
                            blockers=[
                                "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED",
                                "RANGE_FORMING_HTF_TREND_CONFLICT",
                                "REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE",
                            ],
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], unlock_lane)
        self.assertEqual(payload["top_lane"]["status"], "NO_TRADE_WITH_CAUSE")
        self.assertNotIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", payload["top_lane"]["blockers"])
        self.assertEqual(payload["ranked_active_lanes"][1]["lane_id"], bidask_lane)

    def test_live_ready_lane_keeps_current_intent_over_stale_no_trade_shape(self) -> None:
        now = datetime(2026, 7, 8, 16, 20, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            lane_id = "range_trader:GBP_USD:SHORT:RANGE_ROTATION"
            row = _intent_row(lane_id, "GBP_USD", "SHORT", "LIMIT")
            row["status"] = "LIVE_READY"
            row["risk_allowed"] = True
            row["live_blocker_codes"] = []
            _write_json(paths["order_intents"], {"generated_at_utc": now.isoformat(), "results": [row]})
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(
                paths["payoff"],
                {
                    "generated_at_utc": (now - timedelta(hours=8)).isoformat(),
                    "harvest_candidates": [],
                    "no_trade_shapes": [
                        {
                            "shape_key": "GBP_USD|SHORT|RANGE_ROTATION",
                            "pair": "GBP_USD",
                            "side": "SHORT",
                            "method": "RANGE_ROTATION",
                            "reason_code": "REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE",
                            "current_intent_blockers": [
                                {
                                    "lane_id": "range_trader:GBP_USD:SHORT:RANGE_ROTATION:MARKET",
                                    "code": "REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE",
                                }
                            ],
                        }
                    ],
                    "live_side_effects": [],
                },
            )
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], "BOARD_BUILT_LIVE_READY_DIAGNOSTIC_ONLY")
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["coverage_summary"]["live_ready_count"], 1)
        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_lane"]["status"], "LIVE_READY")
        self.assertEqual(payload["top_lane"]["blockers"], [])
        self.assertIn("REALIZED_NEGATIVE_NO_POSITIVE_TP_SHAPE", payload["top_lane"]["stale_source_blockers"])

    def test_verification_lane_blockers_uses_concrete_codes_not_generic_check_name(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            lane_id = "range_trader:CAD_JPY:SHORT:RANGE_ROTATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "CAD_JPY",
                            "SHORT",
                            "LIMIT",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                        )
                    ],
                },
            )
            _write_json(
                paths["verification"],
                {
                    "blocking_evidence": [
                        {
                            "check_name": "lane_blockers",
                            "status": "BLOCK",
                            "severity": "BLOCK",
                            "subject_id": lane_id,
                            "subject_type": "lane",
                            "evidence": {
                                "blockers": [
                                    {
                                        "code": "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                        "severity": "BLOCK",
                                    },
                                    "negative expectancy diagnostic text without a code",
                                ]
                            },
                        }
                    ],
                    "learning_evidence": [
                        {
                            "check_name": "read_only_learning",
                            "status": "PASS",
                            "severity": "INFO",
                            "subject_id": lane_id,
                            "subject_type": "lane",
                        }
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_lane"]["blockers"], ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"])
        self.assertNotIn("lane_blockers", payload["top_lane"]["blockers"])
        self.assertNotIn("read_only_learning", payload["top_lane"]["blockers"])

    def test_verification_lane_blockers_preserves_concrete_codes_when_intent_omits_them(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            lane_id = "range_trader:CHF_JPY:SHORT:RANGE_ROTATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [_intent_row(lane_id, "CHF_JPY", "SHORT", "LIMIT", blockers=[])],
                },
            )
            _write_json(
                paths["verification"],
                {
                    "blocking_evidence": [
                        {
                            "check_name": "lane_blockers",
                            "status": "BLOCK",
                            "severity": "BLOCK",
                            "subject_id": lane_id,
                            "subject_type": "lane",
                            "evidence": {
                                "blockers": [
                                    {"code": "SPREAD_TOO_WIDE", "severity": "BLOCK"},
                                    {"code": "FORECAST_WATCH_ONLY", "severity": "WARN"},
                                    "spread diagnostic text without a code",
                                ]
                            },
                        }
                    ],
                    "learning_evidence": [],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertIn("SPREAD_TOO_WIDE", payload["top_lane"]["blockers"])
        self.assertNotIn("FORECAST_WATCH_ONLY", payload["top_lane"]["blockers"])
        self.assertNotIn("lane_blockers", payload["top_lane"]["blockers"])

    def test_stale_bidask_negative_evidence_becomes_evidence_acquisition(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "EUR_USD",
                            "LONG",
                            "LIMIT",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                            metadata={
                                "forecast_direction": "RANGE",
                                "bidask_replay_precision_negative": {
                                    "name": "EUR_USD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
                                    "pair": "EUR_USD",
                                    "side": "LONG",
                                    "direction": "UP",
                                    "granularity": "S5",
                                    "samples": 1383,
                                    "active_days": 32,
                                    "last_day": "2026-07-03",
                                    "directional_hit_rate": 0.29,
                                    "avg_final_pips": -2.8936,
                                    "avg_mae_pips": 7.6197,
                                    "positive_day_rate": 0.0,
                                    "audit_report": "logs/reports/forecast_improvement/missing_bidask_report.json",
                                    "rule_set_generated_at_utc": "2026-07-03T14:52:18.653002Z",
                                },
                            },
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_lane"]["status"], "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["top_lane"]["replay_status"], "NEGATIVE_EVIDENCE_REFRESH_REQUIRED")
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", payload["top_lane"]["blockers"])
        self.assertIn("BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED", payload["top_lane"]["blockers"])
        self.assertIn("BIDASK_REPLAY_AUDIT_REPORT_MISSING", payload["top_lane"]["evidence_refresh_reasons"])
        self.assertIn("Refresh exact S5 bid/ask replay evidence", payload["top_lane"]["next_action"])
        self.assertEqual(payload["coverage_summary"]["evidence_acquisition_count"], 1)
        self.assertFalse(payload["live_permission_allowed"])

    def test_bidask_last_day_stale_but_price_truth_ok_stays_no_trade_cause(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            audit_report = Path(tmp) / "current_bidask_report.json"
            audit_report.write_text(
                json.dumps(
                    {
                        "generated_at_utc": now.isoformat(),
                        "price_truth_coverage": {
                            "status": "PRICE_TRUTH_OK",
                            "missing_price_truth_samples": 0,
                            "missing_price_window_group_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            lane_id = "trend_trader:GBP_USD:LONG:TREND_CONTINUATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "GBP_USD",
                            "LONG",
                            "STOP",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                            metadata={
                                "forecast_direction": "UP",
                                "bidask_replay_precision_negative": {
                                    "name": "GBP_USD_UP_S5_BIDASK_NEGATIVE_EXPECTANCY",
                                    "pair": "GBP_USD",
                                    "side": "LONG",
                                    "direction": "UP",
                                    "granularity": "S5",
                                    "samples": 1426,
                                    "active_days": 34,
                                    "last_day": "2026-07-04",
                                    "avg_final_pips": -5.894,
                                    "audit_report": str(audit_report),
                                    "rule_set_generated_at_utc": now.isoformat(),
                                    "price_truth_coverage": {
                                        "status": "PRICE_TRUTH_OK",
                                        "missing_price_truth_samples": 0,
                                        "missing_price_window_group_count": 0,
                                    },
                                },
                            },
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_lane"]["status"], "NO_TRADE_WITH_CAUSE")
        self.assertEqual(payload["top_lane"]["replay_status"], "NEGATIVE")
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", payload["top_lane"]["blockers"])
        self.assertNotIn("BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED", payload["top_lane"]["blockers"])
        self.assertNotIn("evidence_refresh_reasons", payload["top_lane"])
        self.assertTrue(
            payload["top_lane"]["bidask_negative_evidence"]["last_day_refresh_bypassed_by_price_truth_coverage"]
        )
        self.assertEqual(payload["coverage_summary"]["evidence_acquisition_count"], 0)
        self.assertGreaterEqual(payload["coverage_summary"]["no_trade_count"], 1)
        self.assertFalse(payload["live_permission_allowed"])

    def test_stale_harvest_grade_blockers_do_not_override_fresh_order_intent(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "EUR_USD",
                            "LONG",
                            "LIMIT",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        )
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(
                paths["harvest"],
                {
                    "generated_at_utc": (now - timedelta(minutes=30)).isoformat(),
                    "ranked_harvest_candidates": [
                        {
                            "shape_key": "EUR_USD|LONG|BREAKOUT_FAILURE",
                            "pair": "EUR_USD",
                            "side": "LONG",
                            "method": "BREAKOUT_FAILURE",
                            "classification": "HARVEST_POSITIVE_TP_PROVEN",
                            "rank_score": 8.4,
                            "proof_gap_trades": 0,
                            "current_intent_count": 0,
                            "current_intent_best": {"lane_id": None, "status": None, "live_blocker_codes": []},
                            "promotion_blockers": [
                                "MARKET_CLOSE_LEAK_PRESENT",
                                "NOT_IN_PROOF_QUEUE",
                                "NEGATIVE_EXPECTANCY_ACTIVE",
                                "PROFITABILITY_ACCEPTANCE_BLOCKED",
                            ],
                            "tp_proof": {"take_profit_expectancy_jpy": 591.5},
                        }
                    ],
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                },
            )

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_lane"]["blockers"], ["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"])
        self.assertIn("MARKET_CLOSE_LEAK_PRESENT", payload["top_lane"]["stale_source_blockers"])
        self.assertIn("PROFITABILITY_ACCEPTANCE_BLOCKED", payload["top_lane"]["stale_source_blockers"])

    def test_operator_manual_overlap_blocker_is_not_operator_review_required(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "EUR_USD",
                            "SHORT",
                            "LIMIT",
                            blockers=["OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED"],
                        )
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_lane"]["status"], "NO_TRADE_WITH_CAUSE")
        self.assertEqual(payload["top_lane"]["operator_review_status"], "NOT_REQUIRED")
        self.assertIn("OPERATOR_MANUAL_SAME_THEME_ADD_BLOCKED", payload["top_lane"]["blockers"])

    def test_negative_replay_lane_is_no_trade_with_cause(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        lanes = {row["lane_id"]: row for row in payload["ranked_active_lanes"]}
        aud = lanes["trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION"]
        self.assertEqual(aud["status"], "NO_TRADE_WITH_CAUSE")
        self.assertEqual(aud["replay_status"], "NEGATIVE")
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", aud["blockers"])
        reasons = {row["code"] for row in payload["no_trade_reasons"]}
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", reasons)

    def test_min_lot_feasible_capped_candidate_can_surface_as_scout_ready_without_permission(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now, scout_only=True)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["status"], "SCOUT_READY")
        self.assertEqual(payload["coverage_summary"]["scout_ready_count"], 1)
        self.assertFalse(payload["top_lane"]["live_permission_allowed"])
        self.assertIn("SCOUT_READY", payload["next_active_path"])

    def test_failed_stop_exact_replay_is_consumed_as_no_trade_not_replayed(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)

            ActiveOpportunityBoard(
                active_trader_contract_path=paths["active_contract"],
                trader_goal_loop_path=paths["goal_loop"],
                payoff_shape_diagnosis_path=paths["payoff"],
                harvest_live_grade_path=paths["harvest"],
                proof_pack_queue_path=paths["proof"],
                lane_candidate_board_path=paths["board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                live_order_request_path=paths["live_order"],
                broker_snapshot_path=paths["broker"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                strategy_profile_path=paths["strategy"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[paths["limit_replay"], paths["stop_replay"]],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        lanes = {row["lane_id"]: row for row in payload["ranked_active_lanes"]}
        stop = lanes["failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE"]
        self.assertEqual(stop["vehicle"], "STOP")
        self.assertEqual(stop["status"], "NO_TRADE_WITH_CAUSE")
        self.assertEqual(stop["replay_status"], "STOP_HARVEST_EXACT_S5_BIDASK_REPLAY_FAILED_BLOCKED")
        self.assertIn("S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS", stop["blockers"])
        self.assertIn("do not repeat the same exact replay", stop["next_action"])
        self.assertNotEqual(payload["top_lane"]["lane_id"], stop["lane_id"])


def _write_base_artifacts(root: Path, *, now: datetime, scout_only: bool = False) -> dict[str, Path]:
    paths = {
        "active_contract": root / "data" / "active_trader_contract.json",
        "goal_loop": root / "data" / "trader_goal_loop_orchestrator.json",
        "payoff": root / "data" / "payoff_shape_diagnosis.json",
        "harvest": root / "data" / "harvest_live_grade_path.json",
        "proof": root / "data" / "as_proof_pack_queue.json",
        "board": root / "data" / "as_lane_candidate_board.json",
        "portfolio": root / "data" / "portfolio_4x_path_planner.json",
        "live_order": root / "data" / "live_order_request.json",
        "broker": root / "data" / "broker_snapshot.json",
        "order_intents": root / "data" / "order_intents.json",
        "verification": root / "data" / "verification_ledger.json",
        "execution_db": root / "data" / "execution_ledger.db",
        "strategy": root / "data" / "strategy_profile.json",
        "guardian_consumption": root / "data" / "guardian_receipt_consumption.json",
        "guardian_operator_review": root / "data" / "guardian_receipt_operator_review.json",
        "limit_replay": root / "data" / "eurusd_short_breakout_failure_limit_s5_bidask_replay.json",
        "stop_replay": root / "data" / "eurusd_short_breakout_failure_stop_harvest_replay.json",
        "output": root / "data" / "active_opportunity_board.json",
        "report": root / "docs" / "active_opportunity_board.md",
    }
    _write_json(
        paths["active_contract"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "ACTIVE_PATH_SELECTED_REPLAY_REQUIRED",
            "selected_active_path": "EVIDENCE_ACQUISITION",
            "four_x_progress_hypothesis": "existing contract hypothesis",
            "root_improvement_target": "exact LIMIT proof import",
            "expected_edge_improvement": "evidence quality",
            "live_permission_allowed": False,
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["goal_loop"],
        {
            "generated_at_utc": now.isoformat(),
            "selected_next_work_type": "EDGE_IMPROVEMENT_EXPERIMENT",
            "four_x_progress_hypothesis": "goal loop hypothesis",
            "root_improvement_target": "portfolio evidence",
            "expected_edge_improvement": "positive HARVEST proof",
            "live_permission_allowed": False,
            "live_side_effects": [],
        },
    )
    _write_json(paths["live_order"], {"status": "NO_ACTION", "send_requested": False, "sent": False})
    _write_json(paths["broker"], {"fetched_at_utc": now.isoformat(), "quotes": {"EUR_USD": {}, "AUD_JPY": {}}})
    _write_json(paths["verification"], {"blocking_evidence": [], "learning_evidence": [], "status": "OK"})
    _write_json(paths["strategy"], {"profiles": []})
    _write_json(
        paths["guardian_consumption"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED",
            "normal_routing_allowed": True,
            "current_p0_p1_blocks_routing": False,
            "classifications": [],
            "live_side_effects": [],
            "read_only": True,
        },
    )
    _write_json(
        paths["guardian_operator_review"],
        {
            "generated_at_utc": now.isoformat(),
            "status": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED",
            "normal_routing_allowed": True,
            "unresolved_review_count": 0,
            "classifications": [],
            "live_side_effects": [],
            "read_only": True,
            "no_live_side_effects": True,
        },
    )

    if scout_only:
        _write_json(
            paths["order_intents"],
            {
                "generated_at_utc": now.isoformat(),
                "results": [
                    {
                        "lane_id": "failure_trader:USD_JPY:LONG:BREAKOUT_FAILURE:LIMIT",
                        "status": "DRY_RUN_BLOCKED",
                        "risk_allowed": False,
                        "live_blocker_codes": [],
                        "risk_issues": [],
                        "strategy_issues": [],
                        "risk_metrics": {"risk_jpy": 120.0, "reward_jpy": 300.0, "spread_pips": 0.8},
                        "intent": {
                            "pair": "USD_JPY",
                            "side": "LONG",
                            "order_type": "LIMIT",
                            "units": 1000,
                            "metadata": {"opportunity_mode": "HARVEST"},
                        },
                    }
                ],
            },
        )
        _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
        _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
        _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
        _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
        _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
        return paths

    _write_json(
        paths["order_intents"],
        {
            "generated_at_utc": now.isoformat(),
            "results": [
                _intent_row("failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT", "EUR_USD", "SHORT", "LIMIT"),
                _intent_row("failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE", "EUR_USD", "SHORT", "STOP-ENTRY"),
                _intent_row(
                    "trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION",
                    "AUD_JPY",
                    "SHORT",
                    "STOP-ENTRY",
                    blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                ),
                _intent_row(
                    "range_trader:GBP_USD:LONG:RANGE_ROTATION:MARKET",
                    "GBP_USD",
                    "LONG",
                    "MARKET",
                    blockers=["SPREAD_TOO_WIDE"],
                ),
            ],
        },
    )
    _write_json(
        paths["proof"],
        {
            "summary": {"queue_count": 1, "proof_ready_count": 0, "can_create_live_permission_count": 0},
            "queue": [
                {
                    "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "method": "BREAKOUT_FAILURE",
                    "order_type": "LIMIT",
                    "exit_shape": "TP_PROOF_COLLECTION_HARVEST",
                    "proof_classification": "EVIDENCE_GAP",
                    "can_enter_proof_pack": True,
                    "can_create_live_permission": False,
                    "proof_distance": 2,
                    "current_blockers": ["S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION"],
                }
            ],
            "rejected_candidates": [],
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["portfolio"],
        {
            "candidate_rankings": [
                {
                    "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "method": "BREAKOUT_FAILURE",
                    "order_type": "LIMIT",
                    "proof_classification": "EVIDENCE_GAP",
                    "can_enter_proof_pack": True,
                    "can_create_live_permission": False,
                    "expected_jpy_per_trade": 304.0,
                    "rank_score": 10.0,
                    "proof_distance": 2,
                    "current_blockers": [],
                }
            ],
            "summary": {"can_create_live_permission": False},
        },
    )
    _write_json(
        paths["board"],
        {
            "closest_candidate_to_proof_pack": {
                "lane_id": "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT",
                "pair": "EUR_USD",
                "side": "SHORT",
                "method": "BREAKOUT_FAILURE",
                "order_type": "LIMIT",
                "proof_classification": "EVIDENCE_GAP",
                "can_enter_proof_pack": True,
                "can_create_live_permission": False,
                "proof_distance": 2,
            },
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["payoff"],
        {
            "harvest_candidates": [
                {
                    "shape_key": "AUD_JPY|SHORT|BREAKOUT_FAILURE",
                    "pair": "AUD_JPY",
                    "side": "SHORT",
                    "method": "BREAKOUT_FAILURE",
                    "classification": "HARVEST_POSITIVE_THIN_SAMPLE",
                    "take_profit_expectancy_jpy": 992.7,
                    "proof_gap_trades": 14,
                    "live_promotion_allowed": False,
                }
            ],
            "no_trade_shapes": [],
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["harvest"],
        {
            "ranked_harvest_candidates": [
                {
                    "shape_key": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                    "pair": "EUR_USD",
                    "side": "SHORT",
                    "method": "BREAKOUT_FAILURE",
                    "classification": "HARVEST_PROOF_FLOOR_REACHED_EVIDENCE_ONLY",
                    "rank_score": 100.0,
                    "proof_gap_trades": 0,
                    "live_promotion_allowed": False,
                    "can_create_live_permission": False,
                    "promotion_blockers": ["LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY"],
                    "tp_proof": {"take_profit_expectancy_jpy": 643.0},
                }
            ],
            "live_side_effects": [],
            "live_permission_allowed": False,
        },
    )
    _write_json(
        paths["limit_replay"],
        {
            "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE|LIMIT|HARVEST",
            "status": "LIMIT_REPLAY_PASSED_STILL_BLOCKED",
            "s5_bidask_replay_status": "PASSED_STILL_BLOCKED",
            "replay_sample_count": 4,
            "net_expectancy_after_bidask": 120.0,
            "remaining_blockers": [{"code": "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION"}],
            "live_permission_allowed": False,
            "live_side_effects": [],
        },
    )
    _write_json(
        paths["stop_replay"],
        {
            "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE|STOP|HARVEST",
            "status": "STOP_HARVEST_EXACT_S5_BIDASK_REPLAY_FAILED_BLOCKED",
            "bidask_replay_status": "S5_TRIGGER_OR_TP_PATH_INCOMPLETE_STILL_BLOCKED",
            "replay_sample_count": 7,
            "net_expectancy_after_bidask_slippage": 901.0337,
            "remaining_blockers": [
                {"code": "STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED"},
                {"code": "S5_TP_PATH_DOES_NOT_RECONSTRUCT_OBSERVED_TP_FILLS"},
                {"code": "STOP_SAMPLE_COUNT_THIN_FOR_LIVE_GRADE"},
                {"code": "STOP_TRIGGER_INVALIDATION_NOT_SCOUT_READY"},
            ],
            "scout_candidate_after_replay": False,
            "live_permission_allowed": False,
            "live_side_effects": [],
        },
    )
    return paths


def _intent_row(
    lane_id: str,
    pair: str,
    side: str,
    order_type: str,
    blockers: list[str] | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = blockers or []
    intent_metadata = {"opportunity_mode": "HARVEST"}
    if metadata:
        intent_metadata.update(metadata)
    return {
        "lane_id": lane_id,
        "status": "DRY_RUN_BLOCKED",
        "risk_allowed": False,
        "live_blocker_codes": blockers,
        "risk_issues": [{"code": code, "severity": "BLOCK"} for code in blockers if code == "SPREAD_TOO_WIDE"],
        "strategy_issues": [],
        "risk_metrics": {"risk_jpy": 120.0, "reward_jpy": 300.0, "spread_pips": 0.8},
        "intent": {
            "pair": pair,
            "side": side,
            "order_type": order_type,
            "units": 1000,
            "metadata": intent_metadata,
        },
    }
