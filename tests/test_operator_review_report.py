from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.operator_review_report import (
    OperatorReviewReport,
    STATUS_DATA_INCOMPLETE,
    STATUS_STILL_BLOCKED,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


class OperatorReviewReportTests(unittest.TestCase):
    def test_packages_current_board_top_lane_without_inferred_approval(self) -> None:
        now = datetime(2026, 7, 9, 8, 30, tzinfo=timezone.utc)
        lane = _lane("range_trader:USD_JPY:SHORT:RANGE_ROTATION", "USD_JPY", "SHORT", "RANGE_ROTATION", "LIMIT")
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["contract"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "ACTIVE_PATH_SELECTED_STILL_BLOCKED",
                    "selected_active_path": "OPERATOR_REVIEW_REPORT",
                    "live_permission_allowed": False,
                    "next_trade_enabling_action": "Package operator/guardian review evidence for USD_JPY.",
                    "remaining_blockers": [
                        {"code": "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", "status": "BLOCKING_LIVE_PERMISSION"},
                        {"code": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", "status": "BLOCKING_ROUTING_OR_REVIEW"},
                    ],
                    "current_state": {"active_opportunity_board": {"top_lane": lane}},
                },
            )
            _write_json(paths["board"], {"generated_at_utc": now.isoformat(), "top_lane": lane, "ranked_active_lanes": [lane]})
            _write_json(
                paths["frontier"],
                {
                    "status": "ALL_FRONTIER_BLOCKED_BY_NEGATIVE_EXPECTANCY",
                    "top_non_eurusd_lane": {
                        "lane_id": "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "pair": "GBP_USD",
                        "direction": "LONG",
                        "strategy_family": "BREAKOUT_FAILURE",
                        "vehicle": "LIMIT",
                        "status": "OPERATOR_REVIEW_REQUIRED",
                    },
                    "next_active_path": "EVIDENCE_ACQUISITION",
                    "live_permission_allowed": False,
                },
            )
            _write_json(paths["mapper"], {"status": "NON_EURUSD_EVIDENCE_PATH_FOUND"})
            _write_json(
                paths["consumption"],
                {
                    "status": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                    "normal_routing_allowed": False,
                    "unresolved_issue_count": 1,
                    "classifications": [
                        {
                            "classification": "NEEDS_OPERATOR_REVIEW",
                            "receipt_event_id": "832d2908eeb84b2f",
                            "receipt_action": "REDUCE",
                            "receipt_lifecycle": "EXPIRED",
                            "normal_routing_allowed": False,
                            "operator_review_status": "OPERATOR_REVIEW_STALE",
                            "operator_review_reason": "operator review row is expired",
                        }
                    ],
                },
            )
            guardian_review_payload = {
                "status": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                "normal_routing_allowed": False,
                "classifications": [
                    {
                        "receipt_event_id": "832d2908eeb84b2f",
                        "receipt_action": "REDUCE",
                        "receipt_lifecycle": "EXPIRED",
                        "operator_decision": "OPERATOR_REQUESTS_KEEP_BLOCKED",
                        "normal_routing_allowed": False,
                    }
                ],
            }
            _write_json(paths["guardian_review"], guardian_review_payload)
            guardian_review_before = paths["guardian_review"].read_text()
            _write_json(paths["watchdog"], {"status": "OK", "runtime_status": "OK", "issues": []})
            _write_json(paths["broker"], {"generated_at_utc": now.isoformat(), "positions": [], "orders": []})

            summary = OperatorReviewReport(
                active_trader_contract_path=paths["contract"],
                active_opportunity_board_path=paths["board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                guardian_receipt_consumption_path=paths["consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                qr_trader_run_watchdog_path=paths["watchdog"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            guardian_review_after = paths["guardian_review"].read_text()
            report = paths["report"].read_text()

        self.assertEqual(summary.status, STATUS_STILL_BLOCKED)
        self.assertEqual(payload["target_shape"], "USD_JPY|SHORT|RANGE_ROTATION|LIMIT")
        self.assertEqual(payload["guardian_receipt_id"], "832d2908eeb84b2f")
        self.assertTrue(payload["operator_review_required"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertTrue(payload["success_condition_evaluation"]["target_shape_matches_active_board_top"])
        self.assertFalse(payload["success_condition_evaluation"]["guardian_consumption_normal_routing_allowed"])
        self.assertTrue(payload["success_condition_evaluation"]["explicit_operator_review_not_inferred"])
        self.assertEqual(guardian_review_after, guardian_review_before)
        self.assertIn("USD_JPY", report)
        self.assertIn("Live permission allowed: `False`", report)

    def test_missing_contract_or_board_is_data_incomplete(self) -> None:
        now = datetime(2026, 7, 9, 8, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(paths["frontier"], {})
            _write_json(paths["mapper"], {})
            _write_json(paths["consumption"], {})
            _write_json(paths["guardian_review"], {})
            _write_json(paths["watchdog"], {})
            _write_json(paths["broker"], {})

            OperatorReviewReport(
                active_trader_contract_path=paths["contract"],
                active_opportunity_board_path=paths["board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                guardian_receipt_consumption_path=paths["consumption"],
                guardian_receipt_operator_review_path=paths["guardian_review"],
                qr_trader_run_watchdog_path=paths["watchdog"],
                broker_snapshot_path=paths["broker"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_DATA_INCOMPLETE)
        self.assertFalse(payload["live_permission_allowed"])


def _paths(root: Path) -> dict[str, Path]:
    return {
        "contract": root / "data" / "active_trader_contract.json",
        "board": root / "data" / "active_opportunity_board.json",
        "frontier": root / "data" / "non_eurusd_live_grade_frontier.json",
        "mapper": root / "data" / "non_eurusd_proof_lane_mapper.json",
        "consumption": root / "data" / "guardian_receipt_consumption.json",
        "guardian_review": root / "data" / "guardian_receipt_operator_review.json",
        "watchdog": root / "data" / "qr_trader_run_watchdog.json",
        "broker": root / "data" / "broker_snapshot.json",
        "output": root / "data" / "operator_review_report.json",
        "report": root / "docs" / "operator_review_report.md",
    }


def _lane(lane_id: str, pair: str, side: str, strategy: str, vehicle: str) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "pair": pair,
        "direction": side,
        "strategy_family": strategy,
        "vehicle": vehicle,
        "status": "OPERATOR_REVIEW_REQUIRED",
        "blockers": [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
            "LOCAL_TP_PROOF_ZERO_TRADES",
        ],
        "next_action": f"Package guardian receipt operator-review evidence for {pair}|{side}|{strategy}|{vehicle}.",
        "risk_status": "MIN_LOT_FEASIBLE_RISK_NOT_LIVE_READY",
        "replay_status": "NEGATIVE_EVIDENCE_REFRESH_REQUIRED",
        "spread_status": "OBSERVED_NOT_BLOCKED",
        "guardian_status": "BLOCKED",
        "operator_review_status": "REQUIRED",
        "local_tp_proof": {
            "capture_take_profit_trades": 0,
            "capture_take_profit_wins": 0,
            "capture_take_profit_losses": 0,
            "capture_take_profit_expectancy_jpy": 0.0,
            "capture_take_profit_proof_floor": 20,
            "capture_take_profit_scope_key": f"{pair}|{side}|{strategy}|TAKE_PROFIT_ORDER",
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
        },
    }
