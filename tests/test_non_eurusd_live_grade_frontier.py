from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.non_eurusd_live_grade_frontier import (
    NonEurusdLiveGradeFrontier,
    STATUS_ALL_NEGATIVE,
    STATUS_DATA_INCOMPLETE,
    STATUS_NON_EURUSD_FOUND,
    _active_board_evidence_tp_sort_bucket,
    _artifact_is_current,
    _frontier_sort_key,
    _jsonl_summary,
    _next_action,
    _next_active_path,
    _proof_mapper_is_current,
    _select_top_non_eurusd_lane,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name in {"active_opportunity_board.json", "non_eurusd_proof_lane_mapper.json"} and not payload.get(
        "generated_at_utc"
    ):
        order_intents_path = path.with_name("order_intents.json")
        if order_intents_path.exists():
            order_intents = json.loads(order_intents_path.read_text())
            generated_at_utc = order_intents.get("generated_at_utc") if isinstance(order_intents, dict) else None
            if generated_at_utc:
                payload = {**payload, "generated_at_utc": generated_at_utc}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_execution_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute(
            """
            create table execution_events (
                ts_utc text,
                event_type text,
                lane_id text,
                pair text,
                side text,
                realized_pl_jpy real,
                exit_reason text
            )
            """
        )
        con.commit()
    finally:
        con.close()


class NonEurusdLiveGradeFrontierTests(unittest.TestCase):
    def test_jsonl_summary_stream_preserves_legacy_decode_and_splitline_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projection_ledger.jsonl"
            raw = b"".join(
                (
                    b"\n",
                    b'{"timestamp_utc":"2026-07-01T00:00:00Z","note":"\xff"}\r\n',
                    b"not-json\r[]\n",
                    b'{"timestamp_emitted_utc":"2026-07-02T00:00:00Z"}',
                    "\u2028".encode("utf-8"),
                    b'{"generated_at_utc":"2026-07-03T00:00:00Z"}',
                )
            )
            path.write_bytes(raw)

            summary = _jsonl_summary(path)

        self.assertEqual(
            summary,
            {
                "path": str(path),
                "status": "present",
                "rows": 5,
                "malformed_rows": 1,
                "latest_timestamp_utc": "2026-07-03T00:00:00Z",
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        )

    def test_jsonl_summary_trailing_newline_changes_only_raw_sha(self) -> None:
        row = b'{"timestamp_utc":"2026-07-03T00:00:00Z"}'
        summaries = []
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (b"", b"\n"):
                path = Path(tmp) / f"projection_{len(suffix)}.jsonl"
                raw = row + suffix
                path.write_bytes(raw)
                summaries.append((_jsonl_summary(path), raw))

        for summary, raw in summaries:
            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["malformed_rows"], 0)
            self.assertEqual(summary["latest_timestamp_utc"], "2026-07-03T00:00:00Z")
            self.assertEqual(summary["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertNotEqual(summaries[0][0]["sha256"], summaries[1][0]["sha256"])

    def test_ranks_non_eurusd_frontier_and_keeps_blocker_breakdown(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        aud_jpy = "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT"
        usd_cad = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        eur_usd = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        aud_cad = "range_trader:AUD_CAD:LONG:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent(eur_usd, "EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]),
                        _intent(aud_jpy, "AUD_JPY", "SHORT", "BREAKOUT_FAILURE", "LIMIT", []),
                        _intent(
                            usd_cad,
                            "USD_CAD",
                            "LONG",
                            "BREAKOUT_FAILURE",
                            "LIMIT",
                            [
                                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                                "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
                            ],
                        ),
                        _intent(aud_cad, "AUD_CAD", "LONG", "RANGE_ROTATION", "LIMIT", ["SPREAD_TOO_WIDE"]),
                    ],
                },
            )
            _write_json(
                paths["active_board"],
                {
                    "status": "BOARD_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "ranked_active_lanes": [
                        _board_lane(eur_usd, "EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]),
                        _board_lane(aud_jpy, "AUD_JPY", "SHORT", "BREAKOUT_FAILURE", "LIMIT", []),
                        _board_lane(
                            usd_cad,
                            "USD_CAD",
                            "LONG",
                            "BREAKOUT_FAILURE",
                            "LIMIT",
                            ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        ),
                        _board_lane(aud_cad, "AUD_CAD", "LONG", "RANGE_ROTATION", "LIMIT", ["SPREAD_TOO_WIDE"]),
                    ],
                },
            )
            _write_json(
                paths["mapper"],
                {
                    "status": "NON_EURUSD_EVIDENCE_PATH_FOUND",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "mapped_lanes": [
                        {
                            "lane_id": aud_jpy,
                            "pair": "AUD_JPY",
                            "side": "SHORT",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "promotion_assessment": "EVIDENCE_ACQUISITION_CANDIDATE",
                            "proof_floor": {
                                "current_tp_trades": 20,
                                "required_tp_trades": 20,
                                "remaining_tp_trades": 0,
                                "met": True,
                            },
                        }
                    ],
                },
            )
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_NON_EURUSD_FOUND)
        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], aud_jpy)
        self.assertEqual(payload["next_evidence_lane"]["lane_id"], aud_jpy)
        self.assertTrue(payload["required_checks"]["non_eurusd_closer_than_eurusd"])
        self.assertTrue(payload["required_checks"]["spread_too_wide_not_ignored"])
        self.assertTrue(payload["required_checks"]["bidask_negative_not_ignored"])
        self.assertEqual(
            payload["required_checks"]["usd_cad_long_breakout_failure_blocker_breakdown"][0]["lane_id"],
            usd_cad,
        )
        self.assertEqual(payload["bidask_replay_gaps"][0]["lane_id"], usd_cad)
        self.assertEqual(payload["spread_gaps"][0]["lane_id"], aud_cad)
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_frontier_prefers_mapper_exact_tp_count_over_broad_intent_method_count(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        usd_cad = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            **_intent(
                                usd_cad,
                                "USD_CAD",
                                "LONG",
                                "BREAKOUT_FAILURE",
                                "LIMIT",
                                ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                            ),
                            "intent": {
                                **_intent(
                                    usd_cad,
                                    "USD_CAD",
                                    "LONG",
                                    "BREAKOUT_FAILURE",
                                    "LIMIT",
                                    ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                                )["intent"],
                                "metadata": {
                                    "capture_take_profit_trades": 1,
                                    "capture_take_profit_wins": 1,
                                    "capture_take_profit_losses": 0,
                                    "capture_take_profit_expectancy_jpy": 658.9,
                                    "capture_take_profit_proof_floor": 20,
                                    "capture_take_profit_scope": "PAIR_SIDE_METHOD",
                                    "capture_take_profit_scope_key": "USD_CAD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                                },
                            },
                        }
                    ],
                },
            )
            _write_json(
                paths["active_board"],
                {
                    "status": "BOARD_BUILT",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "ranked_active_lanes": [
                        _board_lane(
                            usd_cad,
                            "USD_CAD",
                            "LONG",
                            "BREAKOUT_FAILURE",
                            "LIMIT",
                            ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
                        )
                    ],
                },
            )
            _write_json(
                paths["mapper"],
                {
                    "status": "NON_EURUSD_EVIDENCE_PATH_FOUND",
                    "read_only": True,
                    "live_side_effects": [],
                    "live_permission_allowed": False,
                    "mapped_lanes": [
                        {
                            "lane_id": usd_cad,
                            "pair": "USD_CAD",
                            "side": "LONG",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "promotion_assessment": "MAPPING_GAPS_REMAIN",
                            "mapping_gaps": ["BROAD_TP_PROOF_NOT_EXACT_VEHICLE"],
                            "proof_floor": {
                                "current_tp_trades": 0,
                                "required_tp_trades": 20,
                                "remaining_tp_trades": 20,
                                "met": False,
                                "broad_method_tp_trades": 1,
                                "broad_method_not_used_as_exact_vehicle_proof": True,
                            },
                        }
                    ],
                },
            )
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        top = payload["top_non_eurusd_lane"]
        self.assertEqual(top["lane_id"], usd_cad)
        self.assertEqual(top["tp_proof_count"], 0)
        self.assertEqual(top["tp_proof_floor"], 20)
        self.assertIn("BROAD_TP_PROOF_NOT_EXACT_VEHICLE", top["blockers"])

    def test_frontier_keeps_zero_proof_no_trade_below_positive_exact_tp_evidence(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        zero_proof_id = "range_trader:CAD_JPY:SHORT:RANGE_ROTATION:LIMIT"
        zero_repair_id = "range_trader:AUD_CAD:SHORT:RANGE_ROTATION:LIMIT"
        positive_proof_id = "range_trader:EUR_JPY:LONG:RANGE_ROTATION:LIMIT"
        zero_blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "LOCAL_TP_PROOF_ZERO_TRADES",
        ]
        positive_blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
            "RANGE_COUNTERTREND_RR_TOO_LOW",
            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
        ]
        zero_repair_blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "LOCAL_TP_PROOF_ZERO_TRADES",
            "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED",
        ]
        zero_next_action = (
            "No trade for CAD_JPY|SHORT|RANGE_ROTATION|LIMIT; exact local TAKE_PROFIT_ORDER proof is 0/20. "
            "Wait for new local TP receipts or an explicitly approved proof-collection scout, then rerank."
        )
        positive_next_action = (
            "Collect exact local TAKE_PROFIT_ORDER proof for "
            "EUR_JPY|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER; preserve every current blocker and do not send."
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            zero_intent = _intent(
                zero_proof_id,
                "CAD_JPY",
                "SHORT",
                "RANGE_ROTATION",
                "LIMIT",
                zero_blockers,
            )
            zero_intent["risk_metrics"]["expected_edge_jpy"] = 0.0
            positive_intent = _intent(
                positive_proof_id,
                "EUR_JPY",
                "LONG",
                "RANGE_ROTATION",
                "LIMIT",
                positive_blockers,
            )
            positive_intent["risk_metrics"]["expected_edge_jpy"] = 655.2
            positive_intent["intent"]["metadata"].update(
                {
                    "capture_take_profit_trades": 1,
                    "capture_take_profit_wins": 1,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 655.2,
                    "capture_take_profit_proof_floor": 20,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                    "capture_take_profit_scope_key": "EUR_JPY|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
                }
            )
            zero_repair_intent = _intent(
                zero_repair_id,
                "AUD_CAD",
                "SHORT",
                "RANGE_ROTATION",
                "LIMIT",
                zero_repair_blockers,
            )
            zero_repair_intent["risk_metrics"]["expected_edge_jpy"] = 0.0
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [zero_intent, zero_repair_intent, positive_intent],
                },
            )

            zero_board = _board_lane(
                zero_proof_id,
                "CAD_JPY",
                "SHORT",
                "RANGE_ROTATION",
                "LIMIT",
                zero_blockers,
            )
            zero_board["expected_edge_jpy"] = 0.0
            zero_board["next_action"] = zero_next_action
            positive_board = _board_lane(
                positive_proof_id,
                "EUR_JPY",
                "LONG",
                "RANGE_ROTATION",
                "LIMIT",
                positive_blockers,
            )
            positive_board["status"] = "EVIDENCE_ACQUISITION"
            positive_board["expected_edge_jpy"] = 655.2
            positive_board["next_action"] = positive_next_action
            positive_board["local_tp_proof"].update(
                {
                    "capture_take_profit_trades": 1,
                    "capture_take_profit_wins": 1,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 655.2,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                    "capture_take_profit_scope_key": "EUR_JPY|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
                }
            )
            zero_repair_board = _board_lane(
                zero_repair_id,
                "AUD_CAD",
                "SHORT",
                "RANGE_ROTATION",
                "LIMIT",
                zero_repair_blockers,
            )
            zero_repair_board["status"] = "EVIDENCE_ACQUISITION"
            zero_repair_board["expected_edge_jpy"] = 0.0
            _write_json(
                paths["active_board"],
                {"ranked_active_lanes": [positive_board, zero_repair_board, zero_board]},
            )
            _write_json(
                paths["mapper"],
                {
                    "mapped_lanes": [
                        {
                            "lane_id": positive_proof_id,
                            "pair": "EUR_JPY",
                            "side": "LONG",
                            "strategy_family": "RANGE_ROTATION",
                            "vehicle": "LIMIT",
                            "promotion_assessment": "EVIDENCE_ACQUISITION_CANDIDATE",
                            "proof_floor": {
                                "current_tp_trades": 1,
                                "required_tp_trades": 20,
                                "remaining_tp_trades": 19,
                                "met": False,
                            },
                        }
                    ]
                },
            )
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], positive_proof_id)
        self.assertEqual(payload["next_evidence_lane"]["lane_id"], positive_proof_id)
        self.assertEqual(payload["top_non_eurusd_lane"]["next_action"], positive_next_action)
        self.assertIn("TP_PROOF_COLLECTION", payload["next_active_path"])
        ranked_by_id = {lane["lane_id"]: lane for lane in payload["ranked_frontier_lanes"]}
        self.assertEqual(ranked_by_id[positive_proof_id]["status"], "EVIDENCE_ACQUISITION")
        self.assertEqual(ranked_by_id[zero_repair_id]["status"], "EVIDENCE_ACQUISITION")
        self.assertEqual(ranked_by_id[zero_repair_id]["tp_proof_count"], 0)
        self.assertEqual(ranked_by_id[zero_proof_id]["status"], "NO_TRADE_WITH_CAUSE")
        self.assertEqual(ranked_by_id[zero_proof_id]["next_action"], zero_next_action)
        for blocker in positive_blockers:
            self.assertIn(blocker, payload["top_non_eurusd_lane"]["blockers"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_frontier_keeps_zero_distance_live_ready_ahead_of_positive_evidence(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        live_id = "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT"
        evidence_id = "range_trader:EUR_USD:LONG:RANGE_ROTATION:LIMIT"
        evidence_blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            live_intent = _intent(live_id, "AUD_JPY", "SHORT", "BREAKOUT_FAILURE", "LIMIT", [])
            live_intent["status"] = "LIVE_READY"
            evidence_intent = _intent(
                evidence_id,
                "EUR_USD",
                "LONG",
                "RANGE_ROTATION",
                "LIMIT",
                evidence_blockers,
            )
            evidence_intent["intent"]["metadata"].update(
                {
                    "capture_take_profit_trades": 1,
                    "capture_take_profit_wins": 1,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 500.0,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                    "capture_take_profit_scope_key": "EUR_USD|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
                }
            )
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [evidence_intent, live_intent]},
            )
            evidence_board = _board_lane(
                evidence_id,
                "EUR_USD",
                "LONG",
                "RANGE_ROTATION",
                "LIMIT",
                evidence_blockers,
            )
            evidence_board["status"] = "EVIDENCE_ACQUISITION"
            evidence_board["local_tp_proof"].update(
                {
                    "capture_take_profit_trades": 1,
                    "capture_take_profit_wins": 1,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 500.0,
                    "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                    "capture_take_profit_scope_key": "EUR_USD|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
                }
            )
            _write_json(
                paths["active_board"],
                {"ranked_active_lanes": [evidence_board]},
            )
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["lane_id"], live_id)
        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], live_id)
        self.assertEqual(payload["top_lane"]["status"], "LIVE_READY")
        self.assertEqual(payload["top_lane"]["distance_to_live_ready"], "0_LIVE_READY_VERIFIER_GATE_ONLY")
        self.assertTrue(payload["required_checks"]["non_eurusd_closer_than_eurusd"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_proof_materiality_only_breaks_ties_within_same_distance_band(self) -> None:
        base = {
            "active_board_status": "EVIDENCE_ACQUISITION",
            "status": "EVIDENCE_ACQUISITION",
            "pair": "EUR_JPY",
            "direction": "LONG",
            "strategy_family": "RANGE_ROTATION",
            "vehicle": "LIMIT",
            "order_intent_index": 0,
            "board_rank_index": 0,
        }
        exact_proof = {
            "capture_take_profit_trades": 1,
            "capture_take_profit_losses": 0,
            "capture_take_profit_expectancy_jpy": 500.0,
            "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
            "capture_take_profit_scope_key": "EUR_JPY|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
        }
        near_positive = {**base, "lane_id": "near-positive", "distance_score": 55, "tp_proof_count": 1, "local_tp_proof": exact_proof}
        near_zero = {
            **base,
            "lane_id": "near-zero",
            "distance_score": 51,
            "tp_proof_count": 0,
            "local_tp_proof": {
                **exact_proof,
                "capture_take_profit_trades": 0,
                "capture_take_profit_expectancy_jpy": 0.0,
            },
        }
        far_positive = {**near_positive, "lane_id": "far-positive", "distance_score": 65}
        boundary_positive = {**near_positive, "lane_id": "boundary-positive", "distance_score": 50}
        boundary_zero = {**near_zero, "lane_id": "boundary-zero", "distance_score": 49}

        self.assertLess(_frontier_sort_key(near_positive), _frontier_sort_key(near_zero))
        self.assertLess(_frontier_sort_key(near_zero), _frontier_sort_key(far_positive))
        self.assertLess(_frontier_sort_key(boundary_positive), _frontier_sort_key(boundary_zero))

    def test_positive_tp_bucket_requires_exact_vehicle_scope_and_known_zero_losses(self) -> None:
        lane = {
            "active_board_status": "EVIDENCE_ACQUISITION",
            "pair": "EUR_JPY",
            "direction": "LONG",
            "strategy_family": "RANGE_ROTATION",
            "vehicle": "LIMIT",
            "tp_proof_count": 1,
            "local_tp_proof": {
                "capture_take_profit_trades": 1,
                "capture_take_profit_losses": 0,
                "capture_take_profit_expectancy_jpy": 500.0,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                "capture_take_profit_scope_key": "EUR_JPY|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
            },
        }
        self.assertEqual(_active_board_evidence_tp_sort_bucket(lane), 0)
        broad = {**lane, "local_tp_proof": {**lane["local_tp_proof"], "capture_take_profit_scope": "PAIR_SIDE_METHOD"}}
        missing_losses = {
            **lane,
            "local_tp_proof": {
                key: value
                for key, value in lane["local_tp_proof"].items()
                if key != "capture_take_profit_losses"
            },
        }
        wrong_vehicle = {
            **lane,
            "local_tp_proof": {
                **lane["local_tp_proof"],
                "capture_take_profit_scope_key": "EUR_JPY|LONG|RANGE_ROTATION|MARKET|TAKE_PROFIT_ORDER",
            },
        }
        self.assertEqual(_active_board_evidence_tp_sort_bucket(broad), 1)
        self.assertEqual(_active_board_evidence_tp_sort_bucket(missing_losses), 1)
        self.assertEqual(_active_board_evidence_tp_sort_bucket(wrong_vehicle), 1)

    def test_floor_met_exact_tp_evidence_routes_to_edge_improvement_not_more_proof(self) -> None:
        edge_action = (
            "Run read-only EDGE_IMPROVEMENT_EXPERIMENT for EUR_JPY|LONG|RANGE_ROTATION|LIMIT; "
            "preserve negative/month-scale blockers, rerank, and do not send."
        )
        lane = {
            "lane_id": "range_trader:EUR_JPY:LONG:RANGE_ROTATION:LIMIT",
            "pair": "EUR_JPY",
            "direction": "LONG",
            "strategy_family": "RANGE_ROTATION",
            "vehicle": "LIMIT",
            "status": "EVIDENCE_ACQUISITION",
            "active_board_status": "EVIDENCE_ACQUISITION",
            "active_board_next_action": edge_action,
            "edge_improvement_candidate": True,
            "edge_improvement_target": "EUR_JPY|LONG|RANGE_ROTATION|LIMIT",
            "tp_proof_count": 20,
            "tp_proof_floor": 20,
            "tp_proof_remaining": 0,
            "bidask_status": "PASS",
            "spread_status": "PASS",
            "forecast_status": "BLOCKED",
            "loss_budget_status": "PASS",
            "blockers": ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"],
            "local_tp_proof": {
                "capture_take_profit_trades": 20,
                "capture_take_profit_losses": 0,
                "capture_take_profit_expectancy_jpy": 500.0,
                "capture_take_profit_scope": "PAIR_SIDE_METHOD_VEHICLE",
                "capture_take_profit_scope_key": "EUR_JPY|LONG|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
            },
        }

        self.assertEqual(_next_action(lane), edge_action)
        self.assertEqual(
            _next_active_path(STATUS_NON_EURUSD_FOUND, lane),
            "EDGE_IMPROVEMENT_EXPERIMENT: " + edge_action,
        )
        self.assertNotIn("proof", _next_action(lane).lower())

    def test_failed_exact_replay_no_trade_preserves_board_action_instead_of_rebuilding_proof(self) -> None:
        failed_action = (
            "No trade for EUR_JPY|SHORT|BREAKOUT_FAILURE|STOP; consume the failed exact replay as not "
            "SCOUT-ready, wait for independent trigger/TP-path evidence, and do not repeat the same exact replay."
        )
        lane = {
            "lane_id": "failure_trader:EUR_JPY:SHORT:BREAKOUT_FAILURE:STOP",
            "pair": "EUR_JPY",
            "direction": "SHORT",
            "strategy_family": "BREAKOUT_FAILURE",
            "vehicle": "STOP",
            "status": "NO_TRADE_WITH_CAUSE",
            "active_board_status": "NO_TRADE_WITH_CAUSE",
            "active_board_next_action": failed_action,
            "tp_proof_count": 1,
            "tp_proof_floor": 20,
            "tp_proof_remaining": 19,
            "bidask_status": "PASS",
            "spread_status": "PASS",
            "forecast_status": "BLOCKED",
            "loss_budget_status": "PASS",
            "blockers": [
                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                "STOP_S5_TRIGGER_OR_TP_PATH_REPLAY_FAILED",
            ],
        }

        self.assertEqual(_next_action(lane), failed_action)
        self.assertEqual(
            _next_active_path(STATUS_NON_EURUSD_FOUND, lane),
            "NO_TRADE_WITH_CAUSE: " + failed_action,
        )
        self.assertNotIn("Build exact TP-proven", _next_action(lane))

    def test_market_proof_collection_lane_yields_to_non_market_frontier(self) -> None:
        market = {
            "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET",
            "pair": "USD_CAD",
            "direction": "LONG",
            "strategy_family": "BREAKOUT_FAILURE",
            "vehicle": "MARKET",
            "active_board_status": "EVIDENCE_ACQUISITION",
            "distance_score": 55,
            "tp_proof_remaining": 19,
            "blockers": ["LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR"],
        }
        limit = {
            **market,
            "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
            "vehicle": "LIMIT",
        }
        unrelated = {
            **limit,
            "lane_id": "range_trader:NZD_CHF:SHORT:RANGE_ROTATION:LIMIT",
            "pair": "NZD_CHF",
            "direction": "SHORT",
            "strategy_family": "RANGE_ROTATION",
        }
        self.assertEqual(_select_top_non_eurusd_lane([market, limit])["lane_id"], limit["lane_id"])
        self.assertEqual(_select_top_non_eurusd_lane([market, unrelated])["lane_id"], market["lane_id"])

    def test_stale_active_board_does_not_downgrade_fresh_live_ready_intent(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            intent = _intent(lane_id, "AUD_JPY", "SHORT", "BREAKOUT_FAILURE", "LIMIT", [])
            intent["status"] = "LIVE_READY"
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [intent]},
            )
            stale_board = _board_lane(
                lane_id,
                "AUD_JPY",
                "SHORT",
                "BREAKOUT_FAILURE",
                "LIMIT",
                ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION", "LOCAL_TP_PROOF_ZERO_TRADES"],
            )
            _write_json(
                paths["active_board"],
                {
                    "generated_at_utc": (now - timedelta(minutes=5)).isoformat(),
                    "ranked_active_lanes": [stale_board],
                },
            )
            _write_json(
                paths["mapper"],
                {
                    "generated_at_utc": (now - timedelta(minutes=4)).isoformat(),
                    "mapped_lanes": [
                        {
                            "lane_id": lane_id,
                            "pair": "AUD_JPY",
                            "side": "SHORT",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "blockers": ["BROAD_TP_PROOF_NOT_EXACT_VEHICLE"],
                            "proof_floor": {
                                "current_tp_trades": 0,
                                "required_tp_trades": 20,
                            },
                        }
                    ],
                },
            )
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_lane"]["status"], "LIVE_READY")
        self.assertEqual(payload["top_lane"]["blockers"], [])
        self.assertEqual(payload["top_lane"]["distance_to_live_ready"], "0_LIVE_READY_VERIFIER_GATE_ONLY")

    def test_artifact_freshness_fails_closed_when_current_intents_have_no_comparable_source_time(self) -> None:
        current_intents = {"generated_at_utc": "2026-07-09T00:00:00+00:00"}
        self.assertFalse(_artifact_is_current(artifact={}, order_intents=current_intents))
        self.assertFalse(_artifact_is_current(artifact={}, order_intents={}))
        self.assertFalse(
            _artifact_is_current(
                artifact={"generated_at_utc": "not-a-timestamp"},
                order_intents=current_intents,
            )
        )
        self.assertTrue(
            _artifact_is_current(
                artifact={"generated_at_utc": "2026-07-09T00:01:00+00:00"},
                order_intents=current_intents,
            )
        )
        board = {"generated_at_utc": "2026-07-09T00:02:00+00:00"}
        self.assertFalse(
            _proof_mapper_is_current(
                proof_mapper={"generated_at_utc": "2026-07-09T00:01:00+00:00"},
                active_board=board,
                order_intents=current_intents,
            )
        )
        self.assertTrue(
            _proof_mapper_is_current(
                proof_mapper={"generated_at_utc": "2026-07-09T00:03:00+00:00"},
                active_board=board,
                order_intents=current_intents,
            )
        )

    def test_reports_non_eurusd_found_even_when_eurusd_is_closer(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        eur_usd = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
        eur_jpy = "range_trader:EUR_JPY:SHORT:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent(eur_usd, "EUR_USD", "SHORT", "RANGE_ROTATION", "LIMIT", []),
                        _intent(eur_jpy, "EUR_JPY", "SHORT", "RANGE_ROTATION", "LIMIT", ["FORECAST_WATCH_ONLY"]),
                    ],
                },
            )
            _write_json(
                paths["active_board"],
                {
                    "ranked_active_lanes": [
                        _board_lane(eur_usd, "EUR_USD", "SHORT", "RANGE_ROTATION", "LIMIT", []),
                        _board_lane(eur_jpy, "EUR_JPY", "SHORT", "RANGE_ROTATION", "LIMIT", ["FORECAST_WATCH_ONLY"]),
                    ],
                },
            )
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_NON_EURUSD_FOUND)
        self.assertEqual(payload["top_lane"]["lane_id"], eur_usd)
        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], eur_jpy)
        self.assertFalse(payload["required_checks"]["non_eurusd_closer_than_eurusd"])

    def test_current_negative_bidask_replay_does_not_repeat_refresh_action(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(lane_id, "GBP_USD", "LONG", "RANGE_ROTATION", "LIMIT", [])]},
            )
            board_lane = _board_lane(lane_id, "GBP_USD", "LONG", "RANGE_ROTATION", "LIMIT", [])
            board_lane["replay_status"] = "NEGATIVE"
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_non_eurusd_lane"]["bidask_status"], "NEGATIVE")
        self.assertIn("BIDASK_NEGATIVE_PATTERN_REPAIR", payload["next_active_path"])
        self.assertNotIn("BIDASK_REPLAY_REFRESH", payload["next_active_path"])
        self.assertIn("Repair bid/ask-negative pattern", payload["top_non_eurusd_lane"]["next_action"])
        self.assertNotIn("Refresh exact S5 bid/ask replay", payload["top_non_eurusd_lane"]["next_action"])

    def test_negative_bidask_replay_refresh_gap_requests_fresh_evidence(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "range_trader:USD_CAD:SHORT:RANGE_ROTATION"
        blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE",
            "BIDASK_REPLAY_EVIDENCE_REFRESH_REQUIRED",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(lane_id, "USD_CAD", "SHORT", "RANGE_ROTATION", "LIMIT", blockers)]},
            )
            board_lane = _board_lane(lane_id, "USD_CAD", "SHORT", "RANGE_ROTATION", "LIMIT", blockers)
            board_lane["replay_status"] = "NEGATIVE_EVIDENCE_REFRESH_REQUIRED"
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_non_eurusd_lane"]["bidask_status"], "REFRESH_REQUIRED")
        self.assertIn("BIDASK_REPLAY_REFRESH", payload["next_active_path"])
        self.assertNotIn("BIDASK_NEGATIVE_PATTERN_REPAIR", payload["next_active_path"])
        self.assertIn("Refresh exact S5 bid/ask replay", payload["top_non_eurusd_lane"]["next_action"])
        self.assertNotIn("Repair bid/ask-negative pattern", payload["top_non_eurusd_lane"]["next_action"])

    def test_refresh_required_bidask_replay_still_requests_replay_refresh(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(lane_id, "GBP_USD", "LONG", "RANGE_ROTATION", "LIMIT", [])]},
            )
            board_lane = _board_lane(lane_id, "GBP_USD", "LONG", "RANGE_ROTATION", "LIMIT", [])
            board_lane["replay_status"] = "REFRESH_REQUIRED"
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_non_eurusd_lane"]["bidask_status"], "REFRESH_REQUIRED")
        self.assertIn("BIDASK_REPLAY_REFRESH", payload["next_active_path"])
        self.assertIn("Refresh exact S5 bid/ask replay", payload["top_non_eurusd_lane"]["next_action"])

    def test_entry_drought_frontier_preserves_board_recovery_action(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "range_trader:GBP_USD:LONG:RANGE_ROTATION"
        blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
            "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
        ]
        board_next_action = (
            "Run entry-frequency recovery analysis for GBP_USD|LONG|RANGE_ROTATION|LIMIT; "
            "historical accepted=3, fills=2, closed_pl_jpy=992.6099 (exact_lane) but recent entries are zero. "
            "Re-tune forecast/pattern selection and bid/ask or local-TP proof for this lane while preserving every current blocker. "
            "Do not send."
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [_intent(lane_id, "GBP_USD", "LONG", "RANGE_ROTATION", "LIMIT", blockers)],
                },
            )
            board_lane = _board_lane(lane_id, "GBP_USD", "LONG", "RANGE_ROTATION", "LIMIT", blockers)
            board_lane["status"] = "EVIDENCE_ACQUISITION"
            board_lane["replay_status"] = "UNKNOWN"
            board_lane["entry_recovery_candidate"] = True
            board_lane["entry_recovery_history"] = {
                "accepted_before_recent": 3,
                "fills_before_recent": 2,
                "closed_pl_jpy": 992.6099,
                "profit_source": "exact_lane",
                "recent_accepted": 0,
                "recent_fills": 0,
            }
            board_lane["next_action"] = board_next_action
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_NON_EURUSD_FOUND)
        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], lane_id)
        self.assertEqual(payload["top_non_eurusd_lane"]["next_action"], board_next_action)
        self.assertIn("ENTRY_FREQUENCY_RECOVERY", payload["next_active_path"])
        self.assertIn("negative expectancy", payload["next_active_path"])
        self.assertNotIn("Build exact TP-proven rotation proof", payload["top_non_eurusd_lane"]["next_action"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_market_tp_proof_gap_routes_to_non_market_counterpart_instead_of_collecting(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
            "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
        ]
        stale_board_action = "Collect 19 exact TP proof sample(s) for failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET."
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [_intent(lane_id, "USD_CAD", "LONG", "BREAKOUT_FAILURE", "MARKET", blockers)],
                },
            )
            board_lane = _board_lane(lane_id, "USD_CAD", "LONG", "BREAKOUT_FAILURE", "MARKET", blockers)
            board_lane["status"] = "EVIDENCE_ACQUISITION"
            board_lane["entry_recovery_candidate"] = True
            board_lane["entry_recovery_history"] = {
                "accepted_before_recent": 2,
                "fills_before_recent": 2,
                "closed_pl_jpy": 664.0852,
                "profit_source": "exact_lane",
            }
            board_lane["local_tp_proof"] = {
                "capture_take_profit_trades": 1,
                "capture_take_profit_losses": 0,
                "capture_take_profit_expectancy_jpy": 658.9,
                "capture_take_profit_proof_floor": 20,
            }
            board_lane["next_action"] = stale_board_action
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        top = payload["top_non_eurusd_lane"]
        self.assertEqual(top["lane_id"], lane_id)
        self.assertIn("MARKET cannot use the TP-proof-collection live exception", top["next_action"])
        self.assertIn("NON_MARKET_TP_PROOF_ROUTE_REQUIRED", payload["next_active_path"])
        self.assertNotIn("Collect 19 exact TP proof", top["next_action"])
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])

    def test_range_forecast_mismatch_prefers_range_rotation_counterpart(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        trend_id = "trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION"
        range_id = "range_trader:AUD_JPY:SHORT:RANGE_ROTATION"
        eur_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION"
        trend_blockers = [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "MATRIX_REPAIR_REJECT_CONTEXT",
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "CHART_DIRECTION_CONFLICT",
            "STRATEGY_NOT_ELIGIBLE",
            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
        ]
        range_blockers = [
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "REWARD_RISK_TOO_LOW",
            "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        _intent(eur_id, "EUR_USD", "SHORT", "RANGE_ROTATION", "LIMIT", ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]),
                        _intent(trend_id, "AUD_JPY", "SHORT", "TREND_CONTINUATION", "STOP", trend_blockers),
                        _intent(range_id, "AUD_JPY", "SHORT", "RANGE_ROTATION", "LIMIT", range_blockers),
                    ],
                },
            )
            _write_json(
                paths["active_board"],
                {
                    "ranked_active_lanes": [
                        _board_lane(eur_id, "EUR_USD", "SHORT", "RANGE_ROTATION", "LIMIT", ["FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE"]),
                        _board_lane(trend_id, "AUD_JPY", "SHORT", "TREND_CONTINUATION", "STOP", trend_blockers),
                        _board_lane(range_id, "AUD_JPY", "SHORT", "RANGE_ROTATION", "LIMIT", range_blockers),
                    ]
                },
            )
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_NON_EURUSD_FOUND)
        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], range_id)
        self.assertEqual(payload["next_evidence_lane"]["lane_id"], range_id)
        self.assertIn(range_id, payload["next_active_path"])
        self.assertNotIn(trend_id, payload["next_active_path"])
        trend_lane = next(lane for lane in payload["ranked_frontier_lanes"] if lane["lane_id"] == trend_id)
        self.assertIn("Route current RANGE forecast mismatch", trend_lane["next_action"])
        self.assertIn(range_id, trend_lane["next_action"])
        self.assertNotIn("Build exact TP-proven rotation proof", trend_lane["next_action"])

    def test_range_forecast_mismatch_without_counterpart_does_not_request_trend_tp_proof(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        trend_id = "trend_trader:AUD_JPY:SHORT:TREND_CONTINUATION"
        blockers = [
            "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
            "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
            "CHART_DIRECTION_CONFLICT",
            "STRATEGY_NOT_ELIGIBLE",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(trend_id, "AUD_JPY", "SHORT", "TREND_CONTINUATION", "STOP", blockers)]},
            )
            _write_json(
                paths["active_board"],
                {"ranked_active_lanes": [_board_lane(trend_id, "AUD_JPY", "SHORT", "TREND_CONTINUATION", "STOP", blockers)]},
            )
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        top = payload["top_non_eurusd_lane"]
        self.assertEqual(top["lane_id"], trend_id)
        self.assertIn("Route current RANGE forecast mismatch away", top["next_action"])
        self.assertIn("RANGE_FORECAST_COUNTERPART_HANDOFF", payload["next_active_path"])
        self.assertNotIn("Build exact TP-proven rotation proof", top["next_action"])

    def test_frontier_does_not_reactivate_active_board_stale_guardian_blocker(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(lane_id, "GBP_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", [])]},
            )
            board_lane = _board_lane(lane_id, "GBP_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", [])
            board_lane["stale_source_blockers"] = ["GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"]
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], lane_id)
        self.assertNotIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", payload["top_non_eurusd_lane"]["blockers"])

    def test_durable_guardian_consumption_suppresses_stale_order_intent_guardian_blocker(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": (now.replace(minute=0) - timedelta(minutes=15)).isoformat(),
                    "results": [
                        _intent(
                            lane_id,
                            "GBP_USD",
                            "LONG",
                            "BREAKOUT_FAILURE",
                            "LIMIT",
                            ["GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", "BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                        )
                    ],
                },
            )
            board_lane = _board_lane(
                lane_id,
                "GBP_USD",
                "LONG",
                "BREAKOUT_FAILURE",
                "LIMIT",
                ["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
            )
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)
            _write_json(
                paths["guardian_consumption"],
                {
                    "generated_at_utc": now.isoformat(),
                    "status": "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED",
                    "normal_routing_allowed": True,
                    "unresolved_issue_count": 0,
                    "classifications": [
                        {
                            "issue_code": "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
                            "receipt_event_id": "receipt-expired",
                            "receipt_action": "REDUCE",
                            "receipt_lifecycle": "EXPIRED",
                            "classification": "HISTORICAL_ONLY",
                            "operator_review_required": True,
                            "operator_review_status": "OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT",
                            "normal_routing_allowed": True,
                        }
                    ],
                },
            )
            _write_json(
                paths["guardian_operator_review"],
                {
                    "generated_at_utc": (now.replace(day=8)).isoformat(),
                    "status": "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING",
                    "normal_routing_allowed": False,
                    "classifications": [],
                },
            )

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                guardian_receipt_consumption_path=paths["guardian_consumption"],
                guardian_receipt_operator_review_path=paths["guardian_operator_review"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        blockers = payload["top_non_eurusd_lane"]["blockers"]
        self.assertNotIn("GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED", blockers)
        self.assertIn("BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE", blockers)

    def test_non_eurusd_frontier_presence_takes_priority_over_all_negative_status(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:CAD_JPY:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            blocker = ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"]
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(lane_id, "CAD_JPY", "LONG", "BREAKOUT_FAILURE", "LIMIT", blocker)]},
            )
            _write_json(
                paths["active_board"],
                {
                    "ranked_active_lanes": [
                        _board_lane(lane_id, "CAD_JPY", "LONG", "BREAKOUT_FAILURE", "LIMIT", blocker)
                    ]
                },
            )
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_NON_EURUSD_FOUND)
        self.assertEqual(payload["top_non_eurusd_lane"]["lane_id"], lane_id)
        self.assertIn("negative expectancy", payload["next_active_path"])

    def test_only_eurusd_negative_frontier_reports_negative_status(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:EUR_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            blocker = [
                "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                "LOCAL_TP_PROOF_ZERO_TRADES",
            ]
            no_trade_action = (
                "No trade for EUR_USD|LONG|BREAKOUT_FAILURE|LIMIT; exact local TAKE_PROFIT_ORDER proof is 0/20. "
                "Wait for new local TP receipts or an explicitly approved proof-collection scout, then rerank."
            )
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(lane_id, "EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", blocker)]},
            )
            board_lane = _board_lane(lane_id, "EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", blocker)
            board_lane["next_action"] = no_trade_action
            _write_json(paths["active_board"], {"ranked_active_lanes": [board_lane]})
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_ALL_NEGATIVE)
        self.assertEqual(payload["top_non_eurusd_lane"], {})
        self.assertEqual(payload["top_lane"]["next_action"], no_trade_action)
        self.assertEqual(payload["next_active_path"], "NO_TRADE_WITH_CAUSE: " + no_trade_action)

    def test_missing_current_artifacts_reports_data_incomplete(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _paths(Path(tmp))
            _write_json(paths["mapper"], {"mapped_lanes": []})
            _write_support_files(paths)

            NonEurusdLiveGradeFrontier(
                active_opportunity_board_path=paths["active_board"],
                order_intents_path=paths["order_intents"],
                non_eurusd_proof_lane_mapper_path=paths["mapper"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                execution_ledger_db_path=paths["execution_db"],
                verification_ledger_path=paths["verification"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_DATA_INCOMPLETE)
        self.assertEqual(payload["ranked_frontier_lanes"], [])


def _paths(root: Path) -> dict[str, Path]:
    return {
        "active_board": root / "data" / "active_opportunity_board.json",
        "order_intents": root / "data" / "order_intents.json",
        "mapper": root / "data" / "non_eurusd_proof_lane_mapper.json",
        "payoff": root / "data" / "payoff_shape_diagnosis.json",
        "proof_queue": root / "data" / "as_proof_pack_queue.json",
        "portfolio": root / "data" / "portfolio_4x_path_planner.json",
        "verification": root / "data" / "verification_ledger.json",
        "guardian_consumption": root / "data" / "guardian_receipt_consumption.json",
        "guardian_operator_review": root / "data" / "guardian_receipt_operator_review.json",
        "execution_db": root / "data" / "execution_ledger.db",
        "forecast_history": root / "data" / "forecast_history.jsonl",
        "projection_ledger": root / "data" / "projection_ledger.jsonl",
        "output": root / "data" / "non_eurusd_live_grade_frontier.json",
        "report": root / "docs" / "non_eurusd_live_grade_frontier.md",
    }


def _write_support_files(paths: dict[str, Path]) -> None:
    _write_json(paths["payoff"], {})
    _write_json(paths["proof_queue"], {"queue": [], "rejected_candidates": []})
    _write_json(paths["portfolio"], {"candidate_rankings": []})
    _write_json(paths["verification"], {"blocking_evidence": []})
    _write_jsonl(paths["forecast_history"], [{"timestamp_utc": "2026-07-09T00:00:00+00:00", "pair": "AUD_JPY"}])
    _write_jsonl(
        paths["projection_ledger"],
        [{"timestamp_emitted_utc": "2026-07-09T00:00:00+00:00", "pair": "AUD_JPY", "resolution_status": "PENDING"}],
    )
    _write_execution_db(paths["execution_db"])


def _intent(
    lane_id: str,
    pair: str,
    side: str,
    method: str,
    vehicle: str,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "status": "DRY_RUN_BLOCKED" if blockers else "DRY_RUN_PASSED",
        "risk_allowed": not blockers,
        "live_blocker_codes": blockers,
        "risk_metrics": {"expected_edge_jpy": 321.0, "spread_pips": 1.2},
        "intent": {
            "pair": pair,
            "side": side,
            "method": method,
            "order_type": vehicle,
            "units": 1000,
            "metadata": {
                "capture_take_profit_trades": 20 if not blockers else 0,
                "capture_take_profit_proof_floor": 20,
            },
        },
    }


def _board_lane(
    lane_id: str,
    pair: str,
    side: str,
    method: str,
    vehicle: str,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "lane_id": lane_id,
        "pair": pair,
        "direction": side,
        "strategy_family": method,
        "vehicle": vehicle,
        "status": "NO_TRADE_WITH_CAUSE" if blockers else "EVIDENCE_ACQUISITION",
        "spread_status": "BLOCKED" if any("SPREAD" in code for code in blockers) else "OBSERVED_NOT_BLOCKED",
        "replay_status": "NEGATIVE" if any("BIDASK" in code for code in blockers) else "UNKNOWN",
        "expected_edge_jpy": 321.0,
        "blockers": blockers,
        "local_tp_proof": {
            "capture_take_profit_trades": 20 if not blockers else 0,
            "capture_take_profit_proof_floor": 20,
        },
    }


if __name__ == "__main__":
    unittest.main()
