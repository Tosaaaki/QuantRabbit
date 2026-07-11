from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from quant_rabbit.active_opportunity_board import (
    ActiveOpportunityBoard,
    _attach_capture_economics_local_tp,
    _exact_vehicle_take_profit_metrics,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_execution_db(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            create table execution_events (
                ts_utc text,
                event_type text,
                lane_id text,
                order_id text,
                trade_id text,
                pair text,
                side text,
                units real,
                realized_pl_jpy real,
                financing_jpy real,
                exit_reason text,
                raw_json text
            )
            """
        )
        con.execute("create table verification_observations (ts_utc text)")
        con.execute(
            "create table sync_state (key text primary key, value text, updated_at_utc text)"
        )
        con.execute(
            "insert into sync_state values ('oanda_transaction_coverage_start_utc', ?, ?)",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00"),
        )
        for row in rows:
            raw_json = row.get("raw_json")
            con.execute(
                """
                insert into execution_events
                    (ts_utc,event_type,lane_id,order_id,trade_id,pair,side,units,
                     realized_pl_jpy,financing_jpy,exit_reason,raw_json)
                values (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.get("ts_utc"),
                    row.get("event_type"),
                    row.get("lane_id", ""),
                    row.get("order_id"),
                    row.get("trade_id"),
                    row.get("pair"),
                    row.get("side"),
                    row.get("units"),
                    row.get("realized_pl_jpy"),
                    row.get("financing_jpy", 0.0),
                    row.get("exit_reason"),
                    json.dumps(raw_json) if isinstance(raw_json, dict) else raw_json,
                ),
            )
        con.commit()
    finally:
        con.close()


def _exact_tp_rows(
    *,
    lane_id: str,
    pair: str,
    side: str,
    entry_reason: str,
    count: int,
    pl_jpy: float,
    start_ts: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    day = start_ts[:10]
    for idx in range(count):
        trade_id = f"{lane_id}:{idx}"
        entry_ts = f"{day}T00:{idx % 60:02d}:00+00:00" if "T" in start_ts else start_ts
        close_ts = f"{day}T01:{idx % 60:02d}:00+00:00" if "T" in start_ts else start_ts
        signed_units = 1000 if side == "LONG" else -1000
        rows.append(
            {
                "ts_utc": entry_ts,
                "event_type": "ORDER_FILLED",
                "lane_id": lane_id,
                "order_id": f"entry-{idx}",
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "units": signed_units,
                "realized_pl_jpy": 0.0,
                "exit_reason": entry_reason,
                "raw_json": {
                    "type": "ORDER_FILL",
                    "reason": entry_reason,
                    "time": entry_ts,
                    "instrument": pair,
                    "orderID": f"entry-{idx}",
                    "units": str(signed_units),
                    "tradeOpened": {
                        "tradeID": trade_id,
                        "units": str(signed_units),
                    },
                },
            }
        )
        rows.append(
            {
                "ts_utc": close_ts,
                "event_type": "TRADE_CLOSED",
                "lane_id": "",
                "order_id": f"close-{idx}",
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "realized_pl_jpy": pl_jpy,
                "exit_reason": "TAKE_PROFIT_ORDER",
                "raw_json": {
                    "type": "ORDER_FILL",
                    "reason": "TAKE_PROFIT_ORDER",
                    "time": close_ts,
                    "commission": "0.0",
                    "guaranteedExecutionFee": "0.0",
                    "tradesClosed": [
                        {
                            "tradeID": trade_id,
                            "realizedPL": str(pl_jpy),
                            "financing": "0.0",
                        }
                    ],
                },
            }
        )
    return rows


class ActiveOpportunityBoardTest(unittest.TestCase):
    def test_board_unreadable_exact_audit_never_falls_back_to_broad_tp_as_vehicle_proof(self) -> None:
        lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        lanes = {
            lane_id: {
                "lane_id": lane_id,
                "pair": "EUR_USD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "LIMIT",
                "blockers": ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                "local_tp_proof": {"attach_take_profit_on_fill": True},
            }
        }
        capture = _capture_payload(
            "EUR_USD",
            "LONG",
            "BREAKOUT_FAILURE",
            trades=20,
            wins=20,
            losses=0,
            expectancy=500.0,
            avg_win=500.0,
            avg_loss=0.0,
        )
        capture["_artifact_status"] = "present"

        _attach_capture_economics_local_tp(
            lanes,
            capture,
            exact_vehicle_tp_metrics=None,
        )

        proof = lanes[lane_id]["local_tp_proof"]
        self.assertEqual(proof["capture_take_profit_evidence_status"], "EVIDENCE_UNREADABLE")
        self.assertEqual(proof["capture_take_profit_trades"], 0)
        self.assertEqual(proof["broad_capture_take_profit_trades"], 20)
        self.assertIn("EVIDENCE_UNREADABLE", proof["capture_take_profit_metrics_source"])

    def test_board_exact_tp_reader_never_uses_close_lane_for_manual_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "execution.db"
            trade_id = "manual-entry"
            _write_execution_db(
                db,
                [
                    {
                        "ts_utc": "2026-07-01T00:00:00+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": "",
                        "order_id": "manual-order",
                        "trade_id": trade_id,
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "units": 1000,
                        "realized_pl_jpy": 0.0,
                        "exit_reason": "LIMIT_ORDER",
                        "raw_json": {
                            "type": "ORDER_FILL",
                            "reason": "LIMIT_ORDER",
                            "time": "2026-07-01T00:00:00+00:00",
                            "instrument": "EUR_USD",
                            "orderID": "manual-order",
                            "units": "1000",
                            "tradeOpened": {
                                "tradeID": trade_id,
                                "units": "1000",
                            },
                        },
                    },
                    {
                        "ts_utc": "2026-07-01T01:00:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                        "order_id": "tp-order",
                        "trade_id": trade_id,
                        "pair": "EUR_USD",
                        "side": "LONG",
                        "realized_pl_jpy": 500.0,
                        "exit_reason": "TAKE_PROFIT_ORDER",
                        "raw_json": {
                            "type": "ORDER_FILL",
                            "reason": "TAKE_PROFIT_ORDER",
                            "time": "2026-07-01T01:00:00+00:00",
                            "commission": "0.0",
                            "guaranteedExecutionFee": "0.0",
                            "tradesClosed": [
                                {
                                    "tradeID": trade_id,
                                    "realizedPL": "500.0",
                                    "financing": "0.0",
                                }
                            ],
                        },
                    },
                ],
            )

            self.assertEqual(_exact_vehicle_take_profit_metrics(db), {})

    def test_board_exact_tp_reader_excludes_market_reduction_before_final_tp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "execution.db"
            trade_id = "mixed-lifecycle"
            lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
            rows = _exact_tp_rows(
                lane_id=lane_id,
                pair="EUR_USD",
                side="LONG",
                entry_reason="LIMIT_ORDER",
                count=1,
                pl_jpy=500.0,
                start_ts="2026-07-01T00:00:00+00:00",
            )
            rows[0]["trade_id"] = trade_id
            rows[0]["raw_json"]["tradeOpened"]["tradeID"] = trade_id
            rows[1]["trade_id"] = trade_id
            rows[1]["raw_json"]["tradesClosed"][0]["tradeID"] = trade_id
            rows.insert(
                1,
                {
                    "ts_utc": "2026-07-01T00:30:00+00:00",
                    "event_type": "TRADE_REDUCED",
                    "lane_id": "",
                    "order_id": "reduce-order",
                    "trade_id": trade_id,
                    "pair": "EUR_USD",
                    "side": "LONG",
                    "realized_pl_jpy": 50.0,
                    "exit_reason": "MARKET_ORDER_TRADE_CLOSE",
                    "raw_json": {
                        "type": "ORDER_FILL",
                        "reason": "MARKET_ORDER_TRADE_CLOSE",
                        "time": "2026-07-01T00:30:00+00:00",
                        "commission": "0.0",
                        "guaranteedExecutionFee": "0.0",
                        "tradeReduced": {
                            "tradeID": trade_id,
                            "realizedPL": "50.0",
                            "financing": "0.0",
                        },
                    },
                },
            )
            _write_execution_db(db, rows)

            self.assertEqual(_exact_vehicle_take_profit_metrics(db), {})

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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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

    def test_durable_consumption_suppresses_stale_guardian_blocker_even_when_legacy_review_top_level_false(
        self,
    ) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            order_intents = json.loads(paths["order_intents"].read_text())
            order_intents["generated_at_utc"] = (now - timedelta(minutes=15)).isoformat()
            for row in order_intents["results"]:
                row["live_blocker_codes"].append("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED")
            _write_json(paths["order_intents"], order_intents)
            _write_json(
                paths["guardian_consumption"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED",
                    "normal_routing_allowed": True,
                    "current_p0_p1_blocks_routing": False,
                    "unresolved_issue_count": 0,
                    "classifications": [
                        {
                            "issue_code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
                            "receipt_event_id": "832d2908eeb84b2f",
                            "receipt_action": "REDUCE",
                            "receipt_lifecycle": "EXPIRED",
                            "classification": "HISTORICAL_ONLY",
                            "operator_review_required": True,
                            "operator_review_status": "OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT",
                            "normal_routing_allowed": True,
                        }
                    ],
                    "live_side_effects": [],
                    "read_only": True,
                },
            )
            _write_json(
                paths["guardian_operator_review"],
                {
                    "generated_at_utc": (now - timedelta(days=6)).isoformat(),
                    "status": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING",
                    "normal_routing_allowed": False,
                    "unresolved_review_count": 0,
                    "classifications": [
                        {
                            "receipt_event_id": "832d2908eeb84b2f",
                            "operator_decision": "OPERATOR_ACKNOWLEDGED_HISTORICAL",
                            "normal_routing_allowed": True,
                            "no_live_side_effects": True,
                        }
                    ],
                    "live_side_effects": [],
                    "read_only": True,
                    "no_live_side_effects": True,
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
                capture_economics_path=paths["capture"],
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

        self.assertTrue(payload["global_safety"]["guardian_receipt_normal_routing_allowed"])
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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

    def test_tp_proven_rotation_blocker_becomes_local_tp_proof_acquisition_for_attached_limit(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "MISSING_METHOD_SCOPE",
                "capture_take_profit_scope_key": "USD_CAD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "range_trader:USD_CAD:LONG:RANGE_ROTATION",
                            "USD_CAD",
                            "LONG",
                            "LIMIT",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                            metadata=metadata,
                        ),
                        _intent_row(
                            "range_trader:USD_CAD:LONG:RANGE_ROTATION:MARKET",
                            "USD_CAD",
                            "LONG",
                            "MARKET",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                            metadata=metadata,
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_json(
                paths["capture"],
                _capture_payload(
                    "USD_CAD",
                    "LONG",
                    "RANGE_ROTATION",
                    trades=5,
                    wins=5,
                    losses=0,
                    expectancy=556.2,
                    avg_win=556.2,
                    avg_loss=0.0,
                ),
            )
            _write_execution_db(
                paths["execution_db"],
                _exact_tp_rows(
                    lane_id="range_trader:USD_CAD:LONG:RANGE_ROTATION",
                    pair="USD_CAD",
                    side="LONG",
                    entry_reason="LIMIT_ORDER",
                    count=5,
                    pl_jpy=556.2,
                    start_ts="2026-07-08T00:00:00+00:00",
                ),
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
                capture_economics_path=paths["capture"],
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

        self.assertEqual(payload["top_lane"]["lane_id"], "range_trader:USD_CAD:LONG:RANGE_ROTATION")
        self.assertEqual(payload["top_lane"]["status"], "EVIDENCE_ACQUISITION")
        self.assertEqual(payload["top_lane"]["blockers"], ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"])
        self.assertIn(
            "USD_CAD|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
            payload["top_lane"]["next_action"],
        )
        self.assertIn("positive Wilson-stressed expectancy", payload["top_lane"]["next_action"])
        self.assertEqual(
            payload["top_lane"]["local_tp_proof"]["capture_take_profit_scope"],
            "PAIR_SIDE_METHOD_VEHICLE",
        )
        self.assertEqual(payload["top_lane"]["local_tp_proof"]["capture_take_profit_trades"], 5)
        self.assertEqual(payload["coverage_summary"]["evidence_acquisition_count"], 1)
        self.assertEqual(payload["status"], "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY")
        market = next(
            row
            for row in payload["ranked_active_lanes"]
            if row["lane_id"] == "range_trader:USD_CAD:LONG:RANGE_ROTATION:MARKET"
        )
        self.assertEqual(market["status"], "NO_TRADE_WITH_CAUSE")
        self.assertFalse(payload["live_permission_allowed"])

    def test_below_floor_positive_local_tp_proof_stays_evidence_acquisition(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "USD_CAD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "USD_CAD",
                            "LONG",
                            "LIMIT",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                            metadata=metadata,
                        )
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_json(
                paths["capture"],
                _capture_payload(
                    "USD_CAD",
                    "LONG",
                    "BREAKOUT_FAILURE",
                    trades=1,
                    wins=1,
                    losses=0,
                    expectancy=658.9,
                    avg_win=658.9,
                    avg_loss=0.0,
                ),
            )
            _write_execution_db(
                paths["execution_db"],
                _exact_tp_rows(
                    lane_id="failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                    pair="USD_CAD",
                    side="LONG",
                    entry_reason="LIMIT_ORDER",
                    count=1,
                    pl_jpy=658.9,
                    start_ts="2026-07-08T00:00:00+00:00",
                ),
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT")
        self.assertEqual(top["status"], "EVIDENCE_ACQUISITION")
        self.assertIn("LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR", top["blockers"])
        self.assertIn("Collect exact local TAKE_PROFIT_ORDER proof", top["next_action"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_broad_method_tp_proof_does_not_make_limit_vehicle_floor_met(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            tp_metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "EUR_USD",
                            "LONG",
                            "LIMIT",
                            blockers=[
                                "NEGATIVE_EXPECTANCY_ACTIVE",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            ],
                            metadata=tp_metadata,
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_json(
                paths["capture"],
                _capture_payload(
                    "EUR_USD",
                    "LONG",
                    "BREAKOUT_FAILURE",
                    trades=20,
                    wins=20,
                    losses=0,
                    expectancy=591.5,
                    avg_win=591.5,
                    avg_loss=0.0,
                ),
            )
            _write_execution_db(
                paths["execution_db"],
                _exact_tp_rows(
                    lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:STOP",
                    pair="EUR_USD",
                    side="LONG",
                    entry_reason="STOP_ORDER",
                    count=20,
                    pl_jpy=591.5,
                    start_ts="2026-07-01T00:00:00+00:00",
                ),
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
                capture_economics_path=paths["capture"],
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

        lane = next(
            row
            for row in payload["ranked_active_lanes"]
            if row["lane_id"] == "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        )
        self.assertNotIn("edge_improvement_candidate", lane)
        self.assertIn("BROAD_TP_PROOF_NOT_EXACT_VEHICLE", lane["blockers"])
        self.assertIn("LOCAL_TP_PROOF_ZERO_TRADES", lane["blockers"])
        self.assertEqual(lane["local_tp_proof"]["capture_take_profit_scope"], "PAIR_SIDE_METHOD_VEHICLE")
        self.assertEqual(
            lane["local_tp_proof"]["capture_take_profit_scope_key"],
            "EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT|TAKE_PROFIT_ORDER",
        )
        self.assertEqual(lane["local_tp_proof"]["capture_take_profit_trades"], 0)
        self.assertEqual(lane["local_tp_proof"]["broad_capture_take_profit_trades"], 20)
        self.assertTrue(lane["local_tp_proof"]["broad_capture_take_profit_not_used_as_exact_vehicle_proof"])
        self.assertIn("exact local TAKE_PROFIT_ORDER proof", lane["next_action"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_exact_vehicle_tp_proof_uses_fill_order_type_when_lane_id_has_no_vehicle(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            tp_metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "GBP_USD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            lane_id = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "GBP_USD",
                            "LONG",
                            "LIMIT",
                            blockers=[
                                "NEGATIVE_EXPECTANCY_ACTIVE",
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                            ],
                            metadata=tp_metadata,
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_json(
                paths["capture"],
                _capture_payload(
                    "GBP_USD",
                    "LONG",
                    "RANGE_ROTATION",
                    trades=3,
                    wins=3,
                    losses=0,
                    expectancy=331.6,
                    avg_win=331.6,
                    avg_loss=0.0,
                ),
            )
            _write_execution_db(
                paths["execution_db"],
                _exact_tp_rows(
                    lane_id=lane_id,
                    pair="GBP_USD",
                    side="LONG",
                    entry_reason="LIMIT_ORDER",
                    count=2,
                    pl_jpy=148.5524,
                    start_ts="2026-07-01T00:00:00+00:00",
                ),
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
                capture_economics_path=paths["capture"],
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

        lane = next(row for row in payload["ranked_active_lanes"] if row["lane_id"] == lane_id)
        proof = lane["local_tp_proof"]
        self.assertEqual(proof["capture_take_profit_scope"], "PAIR_SIDE_METHOD_VEHICLE")
        self.assertEqual(proof["capture_take_profit_scope_key"], "GBP_USD|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER")
        self.assertEqual(proof["capture_take_profit_trades"], 2)
        self.assertEqual(proof["capture_take_profit_wins"], 2)
        self.assertEqual(proof["capture_take_profit_losses"], 0)
        self.assertEqual(proof["capture_take_profit_expectancy_jpy"], 148.5524)
        self.assertIn("LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR", lane["blockers"])
        self.assertNotIn("LOCAL_TP_PROOF_ZERO_TRADES", lane["blockers"])
        self.assertIn("exact_tp_proof=2/20", lane["proof_status"])
        self.assertIn("exact_proof_gap=18", lane["proof_status"])
        self.assertNotIn(";proof_gap=", lane["proof_status"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_tp_floor_met_local_proof_becomes_blocker_repair_not_more_collection(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            tp_metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            below_floor_metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "EUR_JPY|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "range_trader:EUR_JPY:LONG:RANGE_ROTATION:LIMIT",
                            "EUR_JPY",
                            "LONG",
                            "LIMIT",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                            metadata=below_floor_metadata,
                        ),
                        _intent_row(
                            "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "EUR_USD",
                            "LONG",
                            "LIMIT",
                            blockers=[
                                "NEGATIVE_EXPECTANCY_ACTIVE",
                                "MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
                                "HARVEST_TP_STRUCTURE_MISSING",
                            ],
                            metadata=tp_metadata,
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            capture = _capture_payload(
                "EUR_USD",
                "LONG",
                "BREAKOUT_FAILURE",
                trades=20,
                wins=20,
                losses=0,
                expectancy=591.5,
                avg_win=591.5,
                avg_loss=0.0,
            )
            capture["by_pair_side_method_exit_reason"]["EUR_JPY"] = {
                "LONG": {
                    "RANGE_ROTATION": {
                        "TAKE_PROFIT_ORDER": {
                            "trades": 1,
                            "wins": 1,
                            "losses": 0,
                            "expectancy_jpy_per_trade": 655.2,
                            "avg_win_jpy": 655.2,
                            "avg_loss_jpy": 0.0,
                        }
                    }
                }
            }
            _write_json(paths["capture"], capture)
            _write_execution_db(
                paths["execution_db"],
                _exact_tp_rows(
                    lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                    pair="EUR_USD",
                    side="LONG",
                    entry_reason="LIMIT_ORDER",
                    count=20,
                    pl_jpy=591.5,
                    start_ts="2026-07-01T00:00:00+00:00",
                )
                + _exact_tp_rows(
                    lane_id="range_trader:EUR_JPY:LONG:RANGE_ROTATION:LIMIT",
                    pair="EUR_JPY",
                    side="LONG",
                    entry_reason="LIMIT_ORDER",
                    count=1,
                    pl_jpy=655.2,
                    start_ts="2026-07-02T00:00:00+00:00",
                ),
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT")
        self.assertEqual(top["status"], "EVIDENCE_ACQUISITION")
        self.assertTrue(top["edge_improvement_candidate"])
        self.assertEqual(top["tp_proven_harvest_repair_target"], "TP_PROVEN_HARVEST_BLOCKER_REPAIR_REQUIRED")
        self.assertEqual(top["local_tp_proof"]["capture_take_profit_trades"], 20)
        self.assertNotIn("LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR", top["blockers"])
        self.assertNotIn("Collect exact local TAKE_PROFIT_ORDER proof", top["next_action"])
        self.assertIn("EDGE_IMPROVEMENT_EXPERIMENT", top["next_action"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_tp_floor_met_operator_lane_outranks_zero_proof_operator_lane(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            tp_metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                "capture_take_profit_scope_key": "EUR_USD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            zero_metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "MISSING_METHOD_EXIT",
                "capture_take_profit_scope_key": "USD_JPY|SHORT|RANGE_ROTATION|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "range_trader:USD_JPY:SHORT:RANGE_ROTATION:LIMIT",
                            "USD_JPY",
                            "SHORT",
                            "LIMIT",
                            blockers=[
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                            ],
                            metadata=zero_metadata,
                        ),
                        _intent_row(
                            "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "EUR_USD",
                            "LONG",
                            "LIMIT",
                            blockers=[
                                "NEGATIVE_EXPECTANCY_ACTIVE",
                                "MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
                                "HARVEST_TP_STRUCTURE_MISSING",
                                "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                            ],
                            metadata=tp_metadata,
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_json(
                paths["capture"],
                _capture_payload(
                    "EUR_USD",
                    "LONG",
                    "BREAKOUT_FAILURE",
                    trades=20,
                    wins=20,
                    losses=0,
                    expectancy=591.5,
                    avg_win=591.5,
                    avg_loss=0.0,
                ),
            )
            _write_execution_db(
                paths["execution_db"],
                _exact_tp_rows(
                    lane_id="failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT",
                    pair="EUR_USD",
                    side="LONG",
                    entry_reason="LIMIT_ORDER",
                    count=20,
                    pl_jpy=591.5,
                    start_ts="2026-07-01T00:00:00+00:00",
                ),
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT")
        self.assertEqual(top["status"], "OPERATOR_REVIEW_REQUIRED")
        self.assertTrue(top["edge_improvement_candidate"])
        self.assertIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", top["blockers"])
        self.assertIn("After review clears", top["next_action"])
        self.assertIn("EDGE_IMPROVEMENT_EXPERIMENT", top["next_action"])
        self.assertEqual(payload["coverage_summary"]["operator_review_required_count"], 2)
        self.assertFalse(payload["live_permission_allowed"])

    def test_zero_local_tp_proof_is_no_trade_not_evidence_acquisition(self) -> None:
        now = datetime(2026, 7, 8, 11, 45, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            metadata = {
                "attach_take_profit_on_fill": True,
                "capture_take_profit_scope": "MISSING_METHOD_SCOPE",
                "capture_take_profit_scope_key": "USD_CAD|LONG|RANGE_ROTATION|TAKE_PROFIT_ORDER",
                "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                "tp_target_intent": "HARVEST",
            }
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "range_trader:USD_CAD:LONG:RANGE_ROTATION",
                            "USD_CAD",
                            "LONG",
                            "LIMIT",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                            metadata=metadata,
                        )
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_json(paths["capture"], _capture_payload("USD_CAD", "LONG", "TREND_CONTINUATION"))
            _write_execution_db(paths["execution_db"], [])

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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], "range_trader:USD_CAD:LONG:RANGE_ROTATION")
        self.assertEqual(top["status"], "NO_TRADE_WITH_CAUSE")
        self.assertIn("LOCAL_TP_PROOF_ZERO_TRADES", top["blockers"])
        self.assertEqual(
            top["local_tp_proof"]["capture_take_profit_scope"],
            "PAIR_SIDE_METHOD_VEHICLE",
        )
        self.assertEqual(top["local_tp_proof"]["capture_take_profit_trades"], 0)
        self.assertIn("0/20", top["next_action"])
        self.assertEqual(payload["coverage_summary"]["evidence_acquisition_count"], 0)
        self.assertEqual(payload["status"], "BOARD_BUILT_NO_TRADE_WITH_CAUSE")
        self.assertFalse(payload["live_permission_allowed"])

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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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

    def test_current_packaged_bidask_evidence_replaces_stale_intent_evidence(self) -> None:
        now = datetime(2026, 7, 9, 9, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now)
            audit_report = root / "current_euraud_bidask_report.json"
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
            rules_path = root / "bidask_replay_precision_rules.json"
            _write_json(
                rules_path,
                {
                    "generated_at_utc": now.isoformat(),
                    "generated_from": "test_packaged_rules",
                    "price_truth_coverage": {
                        "status": "PRICE_TRUTH_OK",
                        "missing_price_truth_samples": 0,
                        "missing_price_window_group_count": 0,
                    },
                    "negative_rules": [
                        {
                            "name": "EUR_AUD_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY",
                            "pair": "EUR_AUD",
                            "side": "SHORT",
                            "direction": "DOWN",
                            "granularity": "S5",
                            "samples": 396,
                            "active_days": 27,
                            "directional_hit_rate": 0.3157,
                            "avg_final_pips": -5.551,
                            "optimized_profit_factor": 0.0,
                            "positive_day_rate": 0.0,
                            "last_day": "2026-07-08",
                            "blocks_live_support": True,
                            "audit_report": str(audit_report),
                        }
                    ],
                },
            )
            lane_id = "range_trader:EUR_AUD:SHORT:RANGE_ROTATION"
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "EUR_AUD",
                            "SHORT",
                            "LIMIT",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                            metadata={
                                "forecast_direction": "DOWN",
                                "bidask_replay_precision_negative": {
                                    "name": "EUR_AUD_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY",
                                    "pair": "EUR_AUD",
                                    "side": "SHORT",
                                    "direction": "DOWN",
                                    "granularity": "S5",
                                    "samples": 396,
                                    "active_days": 27,
                                    "last_day": "2026-07-08",
                                    "directional_hit_rate": 0.3157,
                                    "avg_final_pips": -5.551,
                                    "positive_day_rate": 0.0,
                                    "audit_report": "logs/reports/forecast_improvement/missing_euraud_bidask.json",
                                    "rule_set_generated_at_utc": "2026-07-08T12:36:54Z",
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

            with patch("quant_rabbit.active_opportunity_board.BIDASK_REPLAY_PRECISION_RULES_PATH", rules_path):
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
                    capture_economics_path=paths["capture"],
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

        lane = payload["top_lane"]
        self.assertEqual(lane["lane_id"], lane_id)
        self.assertEqual(lane["status"], "NO_TRADE_WITH_CAUSE")
        self.assertEqual(lane["replay_status"], "NEGATIVE")
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", lane["blockers"])
        self.assertNotIn("BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED", lane["blockers"])
        self.assertNotIn("evidence_refresh_reasons", lane)
        evidence = lane["bidask_negative_evidence"]
        self.assertTrue(evidence["packaged_pair_side_supplement"])
        self.assertEqual(evidence["audit_report"], str(audit_report))
        self.assertEqual(evidence["audit_report_exists"], True)
        self.assertEqual(
            evidence["replaced_intent_bidask_negative_evidence"]["audit_report"],
            "logs/reports/forecast_improvement/missing_euraud_bidask.json",
        )
        self.assertEqual(payload["coverage_summary"]["evidence_acquisition_count"], 0)
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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

    def test_packaged_pair_side_negative_blocks_vehicle_when_intent_omits_negative(self) -> None:
        now = datetime(2026, 7, 9, 9, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_base_artifacts(root, now=now)
            rules_path = root / "bidask_replay_precision_rules.json"
            _write_json(
                rules_path,
                {
                    "generated_at_utc": now.isoformat(),
                    "generated_from": "test_packaged_rules",
                    "price_truth_coverage": {"status": "PRICE_TRUTH_OK", "missing_price_truth_samples": 0},
                    "negative_rules": [
                        {
                            "name": "AUD_JPY_DOWN_S5_BIDASK_NEGATIVE_EXPECTANCY",
                            "pair": "AUD_JPY",
                            "side": "SHORT",
                            "direction": "DOWN",
                            "granularity": "S5",
                            "samples": 1369,
                            "active_days": 42,
                            "directional_hit_rate": 0.3477,
                            "avg_final_pips": -3.3925,
                            "optimized_profit_factor": 0.0,
                            "positive_day_rate": 0.0,
                            "last_day": "2026-07-08",
                            "blocks_live_support": True,
                            "audit_report": "logs/reports/forecast_improvement/audjpy.json",
                        }
                    ],
                },
            )
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE",
                            "AUD_JPY",
                            "SHORT",
                            "STOP-ENTRY",
                            blockers=[
                                "MATRIX_REPAIR_REJECT_CONTEXT",
                                "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
                            ],
                            metadata={"forecast_direction": "UP"},
                        )
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})

            with patch("quant_rabbit.active_opportunity_board.BIDASK_REPLAY_PRECISION_RULES_PATH", rules_path):
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
                    capture_economics_path=paths["capture"],
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

        lane = payload["top_lane"]
        self.assertEqual(lane["lane_id"], "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE")
        self.assertEqual(lane["status"], "NO_TRADE_WITH_CAUSE")
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", lane["blockers"])
        self.assertTrue(lane["bidask_negative_evidence"]["packaged_pair_side_supplement"])
        self.assertEqual(lane["bidask_negative_evidence"]["samples"], 1369)

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
                capture_economics_path=paths["capture"],
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
                capture_economics_path=paths["capture"],
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

    def test_proof_queue_member_suppresses_stale_not_in_proof_queue_blocker(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            replay = json.loads(paths["limit_replay"].read_text())
            replay["remaining_blockers"] = [
                {"code": "NOT_IN_PROOF_QUEUE"},
                {"code": "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION"},
            ]
            _write_json(paths["limit_replay"], replay)

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
                capture_economics_path=paths["capture"],
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
        self.assertNotIn("NOT_IN_PROOF_QUEUE", lane["blockers"])
        self.assertIn("NOT_IN_PROOF_QUEUE", lane["stale_source_blockers"])
        self.assertIn("S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION", lane["blockers"])

    def test_goal_loop_edge_target_with_negative_blockers_surfaces_read_only_evidence_path(self) -> None:
        now = datetime(2026, 7, 8, 7, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:EUR_USD:SHORT:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            goal_loop = json.loads(paths["goal_loop"].read_text())
            goal_loop["edge_improvement_state"] = {
                "target_shape": "EUR_USD|SHORT|BREAKOUT_FAILURE",
                "experiments": [{"target": "EUR_USD|SHORT|BREAKOUT_FAILURE"}],
            }
            _write_json(paths["goal_loop"], goal_loop)
            order_intents = json.loads(paths["order_intents"].read_text())
            for row in order_intents["results"]:
                if row["lane_id"] != lane_id:
                    continue
                row["live_blocker_codes"] = [
                    "NEGATIVE_EXPECTANCY_ACTIVE",
                    "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY",
                    "PROOF_QUEUE_MEMBER_BUT_NOT_PROOF_READY",
                ]
                row["intent"]["metadata"].update(
                    {
                        "attach_take_profit_on_fill": True,
                        "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
                        "tp_target_intent": "HARVEST",
                    }
                )
            _write_json(paths["order_intents"], order_intents)
            _write_json(
                paths["capture"],
                _capture_payload(
                    "EUR_USD",
                    "SHORT",
                    "BREAKOUT_FAILURE",
                    trades=20,
                    wins=20,
                    losses=0,
                    expectancy=643.2912,
                    avg_win=643.2912,
                    avg_loss=0.0,
                ),
            )
            replay = json.loads(paths["limit_replay"].read_text())
            replay["remaining_blockers"] = [
                {"code": "LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY"},
                {"code": "S5_TOUCH_LAG_REQUIRES_CANONICAL_FILL_RECONCILIATION"},
                {"code": "NEGATIVE_EXPECTANCY_ACTIVE"},
            ]
            _write_json(paths["limit_replay"], replay)

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
                capture_economics_path=paths["capture"],
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

        self.assertEqual(payload["top_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_lane"]["status"], "EVIDENCE_ACQUISITION")
        self.assertTrue(payload["top_lane"]["edge_improvement_candidate"])
        self.assertIn("NEGATIVE_EXPECTANCY_ACTIVE", payload["top_lane"]["blockers"])
        self.assertIn("LIMIT_SAMPLE_FLOOR_NOT_MET_BY_LIMIT_ONLY", payload["top_lane"]["blockers"])
        self.assertIn("EDGE_IMPROVEMENT_EXPERIMENT", payload["top_lane"]["next_action"])
        self.assertEqual(payload["status"], "BOARD_BUILT_ACTIVE_PATH_AVAILABLE_READ_ONLY")
        self.assertFalse(payload["live_permission_allowed"])

    def test_entry_drought_recovery_surfaces_historic_profitable_lane_without_permission(self) -> None:
        now = datetime(2026, 7, 9, 0, 30, tzinfo=timezone.utc)
        historic_lane = "range_trader:USD_JPY:SHORT:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            "range_trader:AUD_CAD:SHORT:RANGE_ROTATION",
                            "AUD_CAD",
                            "SHORT",
                            "LIMIT",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        ),
                        _intent_row(
                            historic_lane,
                            "USD_JPY",
                            "SHORT",
                            "LIMIT",
                            blockers=[
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
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
            _write_execution_db(
                paths["execution_db"],
                [
                    {
                        "ts_utc": "2026-06-11T00:10:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": historic_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:20:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": historic_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:30:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": historic_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:40:00+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": historic_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:50:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "realized_pl_jpy": 900.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:50:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "realized_pl_jpy": 600.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                ],
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], historic_lane)
        self.assertEqual(top["status"], "EVIDENCE_ACQUISITION")
        self.assertTrue(top["entry_recovery_candidate"])
        self.assertIn("ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH", top["blockers"])
        self.assertIn("ENTRY_DROUGHT_PAIR_SIDE_FALLBACK_REQUIRES_LANE_MAPPING", top["blockers"])
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", top["blockers"])
        self.assertIn("Map historical pair/side recovery profit", top["next_action"])
        self.assertEqual(top["entry_recovery_history"]["accepted_before_recent"], 3)
        self.assertEqual(top["entry_recovery_history"]["closed_pl_jpy"], 1500)
        self.assertEqual(top["entry_recovery_history"]["profit_source"], "pair_side_fallback")
        self.assertEqual(payload["coverage_summary"]["entry_recovery_candidate_count"], 1)
        self.assertEqual(payload["entry_recovery_summary"]["candidate_count"], 1)
        self.assertFalse(payload["live_permission_allowed"])

    def test_tiny_entry_drought_profit_is_not_evidence_path(self) -> None:
        now = datetime(2026, 7, 9, 0, 30, tzinfo=timezone.utc)
        lane_id = "range_trader:EUR_JPY:SHORT:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane_id,
                            "EUR_JPY",
                            "SHORT",
                            "LIMIT",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                        )
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_execution_db(
                paths["execution_db"],
                [
                    {
                        "ts_utc": "2026-06-25T15:51:14+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": lane_id,
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-25T15:52:14+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": lane_id,
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-25T15:53:14+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": lane_id,
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-25T16:01:14+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": lane_id,
                        "trade_id": "thin-1",
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-26T16:01:14+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": lane_id,
                        "trade_id": "thin-2",
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-27T16:01:14+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": lane_id,
                        "trade_id": "thin-3",
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-27T16:20:14+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": lane_id,
                        "trade_id": "thin-1",
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "realized_pl_jpy": 6.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-28T16:20:14+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": lane_id,
                        "trade_id": "thin-2",
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "realized_pl_jpy": 5.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-29T16:20:14+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": lane_id,
                        "trade_id": "thin-3",
                        "pair": "EUR_JPY",
                        "side": "SHORT",
                        "realized_pl_jpy": -1.2,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                ],
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], lane_id)
        self.assertEqual(top["status"], "NO_TRADE_WITH_CAUSE")
        self.assertNotIn("ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH", top["blockers"])
        self.assertFalse(top.get("entry_recovery_candidate", False))
        self.assertEqual(payload["coverage_summary"]["entry_recovery_candidate_count"], 0)

    def test_entry_recovery_prefers_limit_over_market_for_same_shape(self) -> None:
        now = datetime(2026, 7, 9, 0, 30, tzinfo=timezone.utc)
        limit_lane = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        market_lane = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            market_lane,
                            "USD_CAD",
                            "LONG",
                            "MARKET",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                        ),
                        _intent_row(
                            limit_lane,
                            "USD_CAD",
                            "LONG",
                            "LIMIT",
                            blockers=["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_execution_db(
                paths["execution_db"],
                [
                    {
                        "ts_utc": "2026-06-12T08:15:56+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": limit_lane,
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-12T08:30:56+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": limit_lane,
                        "trade_id": "limit-1",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-13T08:30:56+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": limit_lane,
                        "trade_id": "limit-2",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-14T08:30:56+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": limit_lane,
                        "trade_id": "limit-1",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "realized_pl_jpy": 330.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-14T09:30:56+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": limit_lane,
                        "trade_id": "limit-2",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "realized_pl_jpy": 330.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-12T09:43:50+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": market_lane,
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "raw_json": {"type": "MARKET_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-12T09:44:50+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": market_lane,
                        "trade_id": "market-1",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "raw_json": {"type": "MARKET_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-13T09:44:50+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": market_lane,
                        "trade_id": "market-2",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "raw_json": {"type": "MARKET_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-14T10:44:50+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": market_lane,
                        "trade_id": "market-1",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "realized_pl_jpy": 330.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-14T11:44:50+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": market_lane,
                        "trade_id": "market-2",
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "realized_pl_jpy": 330.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                ],
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
                capture_economics_path=paths["capture"],
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

        self.assertEqual(payload["top_lane"]["lane_id"], limit_lane)
        self.assertEqual(payload["top_lane"]["vehicle"], "LIMIT")
        self.assertTrue(payload["top_lane"]["entry_recovery_candidate"])

    def test_trade_closed_without_lane_uses_filled_trade_lane_for_exact_entry_recovery(self) -> None:
        now = datetime(2026, 7, 9, 0, 30, tzinfo=timezone.utc)
        lane = "range_trader:AUD_CAD:SHORT:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            lane,
                            "AUD_CAD",
                            "SHORT",
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
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_execution_db(
                paths["execution_db"],
                [
                    {
                        "ts_utc": "2026-06-11T00:10:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": lane,
                        "pair": "AUD_CAD",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:20:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": lane,
                        "pair": "AUD_CAD",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:30:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": lane,
                        "pair": "AUD_CAD",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:40:00+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": lane,
                        "trade_id": "471883",
                        "pair": "AUD_CAD",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:50:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "",
                        "trade_id": "471883",
                        "pair": "AUD_CAD",
                        "side": "SHORT",
                        "realized_pl_jpy": 160.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:40:00+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": lane,
                        "trade_id": "472667",
                        "pair": "AUD_CAD",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:50:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "",
                        "trade_id": "472667",
                        "pair": "AUD_CAD",
                        "side": "SHORT",
                        "realized_pl_jpy": 180.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                ],
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], lane)
        self.assertTrue(top["entry_recovery_candidate"])
        self.assertEqual(top["entry_recovery_history"]["profit_source"], "exact_lane")
        self.assertEqual(top["entry_recovery_history"]["closed_trades"], 2)
        self.assertEqual(top["entry_recovery_history"]["closed_pl_jpy"], 340)
        self.assertIn("ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH", top["blockers"])
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", top["blockers"])
        self.assertNotIn("ENTRY_DROUGHT_PAIR_SIDE_FALLBACK_REQUIRES_LANE_MAPPING", top["blockers"])
        self.assertNotIn("Map historical pair/side recovery profit", top["next_action"])
        self.assertFalse(payload["live_permission_allowed"])

    def test_exact_entry_recovery_ranks_ahead_of_pair_side_fallback_profit(self) -> None:
        now = datetime(2026, 7, 9, 0, 30, tzinfo=timezone.utc)
        exact_lane = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
        fallback_lane = "range_trader:USD_JPY:SHORT:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent_row(
                            fallback_lane,
                            "USD_JPY",
                            "SHORT",
                            "LIMIT",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        ),
                        _intent_row(
                            exact_lane,
                            "GBP_USD",
                            "LONG",
                            "LIMIT",
                            blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        ),
                    ],
                },
            )
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_execution_db(
                paths["execution_db"],
                [
                    {
                        "ts_utc": "2026-06-11T00:10:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": fallback_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:20:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": fallback_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:30:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": fallback_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:40:00+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": fallback_lane,
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:50:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "SHORT",
                        "realized_pl_jpy": 1500.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:10:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": exact_lane,
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:20:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": exact_lane,
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:30:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": exact_lane,
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:40:00+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": exact_lane,
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "raw_json": {"type": "LIMIT_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-12T00:50:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": exact_lane,
                        "pair": "GBP_USD",
                        "side": "LONG",
                        "realized_pl_jpy": 500.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                ],
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], exact_lane)
        self.assertEqual(top["entry_recovery_history"]["profit_source"], "exact_lane")
        fallback = next(row for row in payload["ranked_active_lanes"] if row["lane_id"] == fallback_lane)
        self.assertEqual(fallback["entry_recovery_history"]["profit_source"], "pair_side_fallback")
        self.assertIn("ENTRY_DROUGHT_PAIR_SIDE_FALLBACK_REQUIRES_LANE_MAPPING", fallback["blockers"])

    def test_pair_side_fallback_without_current_intent_is_not_active_evidence_path(self) -> None:
        now = datetime(2026, 7, 9, 0, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_base_artifacts(Path(tmp), now=now)
            _write_json(paths["order_intents"], {"generated_at_utc": now.isoformat(), "results": []})
            _write_json(paths["proof"], {"summary": {"queue_count": 0}, "queue": [], "rejected_candidates": []})
            _write_json(paths["portfolio"], {"candidate_rankings": [], "summary": {"can_create_live_permission": False}})
            _write_json(paths["board"], {"closest_candidate_to_proof_pack": {}, "live_side_effects": []})
            _write_json(paths["payoff"], {"harvest_candidates": [], "no_trade_shapes": [], "live_side_effects": []})
            _write_json(paths["harvest"], {"ranked_harvest_candidates": [], "live_side_effects": [], "live_permission_allowed": False})
            _write_execution_db(
                paths["execution_db"],
                [
                    {
                        "ts_utc": "2026-06-11T00:10:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "raw_json": {"type": "MARKET_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:20:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "raw_json": {"type": "MARKET_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:30:00+00:00",
                        "event_type": "ORDER_ACCEPTED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "raw_json": {"type": "MARKET_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:40:00+00:00",
                        "event_type": "ORDER_FILLED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "raw_json": {"type": "MARKET_ORDER"},
                    },
                    {
                        "ts_utc": "2026-06-11T00:50:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": "",
                        "pair": "USD_JPY",
                        "side": "LONG",
                        "realized_pl_jpy": 1200.0,
                        "raw_json": {"type": "ORDER_FILL"},
                    },
                ],
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
                capture_economics_path=paths["capture"],
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

        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], "entry_recovery:USD_JPY:LONG:UNKNOWN:MARKET")
        self.assertEqual(top["status"], "NO_TRADE_WITH_CAUSE")
        self.assertEqual(top["entry_recovery_history"]["profit_source"], "pair_side_fallback")
        self.assertIn("ENTRY_DROUGHT_RECOVERY_REQUIRES_CURRENT_INTENT", top["blockers"])
        self.assertIn("ENTRY_DROUGHT_PAIR_SIDE_FALLBACK_REQUIRES_LANE_MAPPING", top["blockers"])


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
        "capture": root / "data" / "capture_economics.json",
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


def _capture_payload(
    pair: str,
    side: str,
    method: str,
    *,
    trades: int = 1,
    wins: int = 1,
    losses: int = 0,
    expectancy: float = 500.0,
    avg_win: float = 500.0,
    avg_loss: float = 0.0,
) -> dict[str, Any]:
    return {
        "generated_at_utc": "2026-07-08T11:00:00+00:00",
        "status": "NEGATIVE_EXPECTANCY",
        "min_sample_for_verdict": 20,
        "segment_repair_priorities": {
            "scoped_tp_proof_min_exit_trades": 20,
            "items": [],
        },
        "by_pair_side_method_exit_reason": {
            pair: {
                side: {
                    method: {
                        "TAKE_PROFIT_ORDER": {
                            "trades": trades,
                            "wins": wins,
                            "losses": losses,
                            "expectancy_jpy_per_trade": expectancy,
                            "avg_win_jpy": avg_win,
                            "avg_loss_jpy": avg_loss,
                        }
                    }
                }
            }
        },
        "by_pair_side_exit_reason": {},
        "live_side_effects": [],
    }


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
