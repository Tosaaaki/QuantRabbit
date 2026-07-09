from __future__ import annotations

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
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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

    def test_negative_bidask_replay_takes_priority_over_refresh_gap(self) -> None:
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

        self.assertEqual(payload["top_non_eurusd_lane"]["bidask_status"], "NEGATIVE")
        self.assertIn("BIDASK_NEGATIVE_PATTERN_REPAIR", payload["next_active_path"])
        self.assertNotIn("BIDASK_REPLAY_REFRESH", payload["next_active_path"])
        self.assertIn("Repair bid/ask-negative pattern", payload["top_non_eurusd_lane"]["next_action"])
        self.assertNotIn("Refresh exact S5 bid/ask replay", payload["top_non_eurusd_lane"]["next_action"])

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
            blocker = ["NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION"]
            _write_json(
                paths["order_intents"],
                {"generated_at_utc": now.isoformat(), "results": [_intent(lane_id, "EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", blocker)]},
            )
            _write_json(
                paths["active_board"],
                {
                    "ranked_active_lanes": [
                        _board_lane(lane_id, "EUR_USD", "LONG", "BREAKOUT_FAILURE", "LIMIT", blocker)
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

        self.assertEqual(payload["status"], STATUS_ALL_NEGATIVE)
        self.assertEqual(payload["top_non_eurusd_lane"], {})
        self.assertIn("negative expectancy", payload["next_active_path"])

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
