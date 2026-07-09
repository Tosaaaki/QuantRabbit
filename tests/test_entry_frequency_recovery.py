from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.entry_frequency_recovery import EntryFrequencyRecovery


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


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
                trade_id text,
                pair text,
                side text,
                realized_pl_jpy real
            )
            """
        )
        rows = [
            ("2026-06-12T09:43:50+00:00", "ORDER_ACCEPTED", "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET", "", "USD_CAD", "LONG", None),
            ("2026-06-12T09:44:50+00:00", "ORDER_FILLED", "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET", "471100", "USD_CAD", "LONG", None),
            ("2026-06-12T10:44:50+00:00", "TRADE_CLOSED", "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET", "471100", "USD_CAD", "LONG", 320.0),
        ]
        con.executemany("insert into execution_events values (?,?,?,?,?,?,?)", rows)
        con.commit()
    finally:
        con.close()


class EntryFrequencyRecoveryTest(unittest.TestCase):
    def test_usdcad_entry_drought_becomes_concrete_forecast_pattern_tuning_queue(self) -> None:
        now = datetime(2026, 7, 9, 10, 30, tzinfo=timezone.utc)
        lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _paths(root)
            board_lane = {
                "lane_id": lane_id,
                "pair": "USD_CAD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "MARKET",
                "status": "EVIDENCE_ACQUISITION",
                "expected_edge_jpy": 658.9,
                "entry_recovery_candidate": True,
                "entry_recovery_history": {
                    "accepted_before_recent": 2,
                    "fills_before_recent": 2,
                    "recent_accepted": 0,
                    "recent_fills": 0,
                    "closed_trades": 2,
                    "closed_pl_jpy": 664.0852,
                    "profit_source": "exact_lane",
                },
                "local_tp_proof": {
                    "capture_take_profit_scope_key": "USD_CAD|LONG|BREAKOUT_FAILURE|TAKE_PROFIT_ORDER",
                    "capture_take_profit_trades": 1,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 658.9,
                    "capture_take_profit_proof_floor": 20,
                },
                "blockers": [
                    "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
            }
            _write_json(
                paths["active_board"],
                {
                    "generated_at_utc": now.isoformat(),
                    "top_lane": board_lane,
                    "ranked_active_lanes": [board_lane],
                    "entry_recovery_summary": {"top_candidates": [board_lane]},
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": now.isoformat(),
                    "current_state": {"active_opportunity_board": {"top_lane": board_lane}},
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["frontier"],
                {
                    "required_checks": {
                        "next_evidence_lane": {
                            "lane_id": "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT",
                            "pair": "USD_CAD",
                            "direction": "LONG",
                            "strategy_family": "BREAKOUT_FAILURE",
                            "vehicle": "LIMIT",
                            "tp_proof_count": 1,
                            "tp_proof_floor": 20,
                        }
                    },
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            "lane_id": lane_id,
                            "status": "DRY_RUN_BLOCKED",
                            "risk_allowed": False,
                            "live_blocker_codes": board_lane["blockers"],
                            "intent": {
                                "pair": "USD_CAD",
                                "side": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "order_type": "MARKET",
                                "market_context": {
                                    "method": "BREAKOUT_FAILURE",
                                    "event_risk": "range forecast blocks non-range entry",
                                    "narrative": "candidate needs pattern refresh",
                                    "chart_story": "M5 CHOCH_DOWN trap fade UP",
                                },
                            },
                        }
                    ],
                },
            )
            _write_json(
                paths["strategy_profile"],
                {
                    "profiles": [
                        {
                            "pair": "USD_CAD",
                            "direction": "LONG",
                            "method": None,
                            "status": "CANDIDATE",
                            "live_n": 8,
                            "live_net_jpy": 1019.9,
                            "positive_evidence_n": 5,
                            "positive_tail_jpy": 777.6,
                            "required_fix": "pair-side positive; method scope missing",
                        }
                    ]
                },
            )
            _append_jsonl(
                paths["forecast_history"],
                [
                    {
                        "timestamp_utc": "2026-07-09T10:14:56.370426Z",
                        "cycle_id": "cycle-1",
                        "pair": "USD_CAD",
                        "direction": "RANGE",
                        "confidence": 1.0,
                        "raw_confidence": 0.86,
                        "calibration_multiplier": 1.332,
                        "horizon_min": 120,
                        "up_score": 84.0,
                        "down_score": 67.5,
                        "range_score": 12.8,
                        "rationale_summary": "contested direction inside RANGE_FORMING",
                    }
                ],
            )
            _append_jsonl(
                paths["projection_ledger"],
                [
                    {
                        "timestamp_emitted_utc": "2026-07-09T10:14:56.370426Z",
                        "pair": "USD_CAD",
                        "signal_name": "directional_forecast",
                        "direction": "RANGE",
                        "confidence": 1.0,
                        "resolution_window_min": 120,
                        "resolution_status": "PENDING",
                    },
                    {
                        "timestamp_emitted_utc": "2026-07-09T10:14:56.370426Z",
                        "pair": "USD_CAD",
                        "signal_name": "liquidity_sweep_low",
                        "direction": "UP",
                        "confidence": 0.99,
                        "resolution_window_min": 30,
                        "resolution_status": "PENDING",
                    },
                ],
            )
            _write_execution_db(paths["execution_db"])

            summary = EntryFrequencyRecovery(
                active_trader_contract_path=paths["active_contract"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                order_intents_path=paths["order_intents"],
                strategy_profile_path=paths["strategy_profile"],
                execution_ledger_db_path=paths["execution_db"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())
            report = paths["report"].read_text()

        self.assertEqual(summary.status, "ENTRY_FREQUENCY_RECOVERY_ANALYSIS_BUILT")
        top = payload["top_lane"]
        self.assertEqual(top["lane_id"], lane_id)
        self.assertEqual(top["forecast_audit"]["status"], "RANGE_FORECAST_BLOCKS_NON_RANGE_ENTRY")
        self.assertEqual(top["strategy_profile_audit"]["status"], "PAIR_SIDE_PROFILE_PRESENT_METHOD_PROFILE_MISSING")
        self.assertEqual(top["tp_proof_audit"]["remaining_tp_trades"], 19)
        actions = {row["action_type"] for row in payload["forecast_pattern_tuning_queue"]}
        self.assertIn("FORECAST_PATTERN_REFRESH", actions)
        self.assertIn("TRIGGER_PROJECTION_TO_LIMIT_PROOF", actions)
        self.assertIn("METHOD_SCOPED_PROFILE_PROMOTION", actions)
        self.assertIn("EXACT_TP_PROOF_COLLECTION", actions)
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertIn("Do not send", payload["next_contract_prompt"])
        self.assertIn("Entry Frequency Recovery", report)

    def test_next_contract_prompt_uses_top_lane_action_not_global_first_action(self) -> None:
        now = datetime(2026, 7, 9, 10, 30, tzinfo=timezone.utc)
        top_lane_id = "range_trader:EUR_USD:SHORT:RANGE_ROTATION:LIMIT"
        other_lane_id = "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:MARKET"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _paths(root)
            top_lane = {
                "lane_id": top_lane_id,
                "pair": "EUR_USD",
                "direction": "SHORT",
                "strategy_family": "RANGE_ROTATION",
                "vehicle": "LIMIT",
                "status": "EVIDENCE_ACQUISITION",
                "expected_edge_jpy": 457.9,
                "entry_recovery_candidate": True,
                "entry_recovery_history": {
                    "accepted_before_recent": 26,
                    "fills_before_recent": 12,
                    "recent_accepted": 0,
                    "recent_fills": 0,
                    "closed_pl_jpy": 1554.3,
                    "profit_source": "exact_lane",
                },
                "local_tp_proof": {
                    "capture_take_profit_scope_key": "EUR_USD|SHORT|RANGE_ROTATION|LIMIT|TAKE_PROFIT_ORDER",
                    "capture_take_profit_trades": 8,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 457.9,
                    "capture_take_profit_proof_floor": 20,
                },
                "blockers": [
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
            }
            other_lane = {
                "lane_id": other_lane_id,
                "pair": "USD_CAD",
                "direction": "LONG",
                "strategy_family": "BREAKOUT_FAILURE",
                "vehicle": "MARKET",
                "status": "EVIDENCE_ACQUISITION",
                "expected_edge_jpy": 658.9,
                "entry_recovery_candidate": True,
                "entry_recovery_history": {
                    "accepted_before_recent": 2,
                    "fills_before_recent": 2,
                    "recent_accepted": 0,
                    "recent_fills": 0,
                    "closed_pl_jpy": 664.0,
                    "profit_source": "exact_lane",
                },
                "local_tp_proof": {
                    "capture_take_profit_scope_key": "USD_CAD|LONG|BREAKOUT_FAILURE|MARKET|TAKE_PROFIT_ORDER",
                    "capture_take_profit_trades": 1,
                    "capture_take_profit_losses": 0,
                    "capture_take_profit_expectancy_jpy": 658.9,
                    "capture_take_profit_proof_floor": 20,
                },
                "blockers": [
                    "RANGE_FORECAST_REQUIRES_RANGE_ROTATION",
                    "LOCAL_TP_PROOF_BELOW_COLLECTION_FLOOR",
                    "ENTRY_DROUGHT_RECOVERY_REQUIRES_PATTERN_REFRESH",
                ],
            }
            _write_json(
                paths["active_board"],
                {
                    "generated_at_utc": now.isoformat(),
                    "top_lane": top_lane,
                    "ranked_active_lanes": [top_lane, other_lane],
                    "entry_recovery_summary": {"top_candidates": [top_lane, other_lane]},
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            _write_json(
                paths["active_contract"],
                {
                    "generated_at_utc": now.isoformat(),
                    "current_state": {"active_opportunity_board": {"top_lane": top_lane}},
                    "live_permission_allowed": False,
                    "live_side_effects": [],
                },
            )
            _write_json(paths["frontier"], {"live_permission_allowed": False, "live_side_effects": []})
            _write_json(
                paths["order_intents"],
                {
                    "generated_at_utc": now.isoformat(),
                    "results": [
                        {
                            "lane_id": top_lane_id,
                            "status": "DRY_RUN_BLOCKED",
                            "risk_allowed": False,
                            "live_blocker_codes": top_lane["blockers"],
                            "intent": {
                                "pair": "EUR_USD",
                                "side": "SHORT",
                                "method": "RANGE_ROTATION",
                                "order_type": "LIMIT",
                            },
                        },
                        {
                            "lane_id": other_lane_id,
                            "status": "DRY_RUN_BLOCKED",
                            "risk_allowed": False,
                            "live_blocker_codes": other_lane["blockers"],
                            "intent": {
                                "pair": "USD_CAD",
                                "side": "LONG",
                                "method": "BREAKOUT_FAILURE",
                                "order_type": "MARKET",
                            },
                        },
                    ],
                },
            )
            _write_json(
                paths["strategy_profile"],
                {
                    "profiles": [
                        {
                            "pair": "EUR_USD",
                            "direction": "SHORT",
                            "method": "RANGE_ROTATION",
                            "status": "CANDIDATE",
                            "live_n": 12,
                            "live_net_jpy": 1554.3,
                        },
                        {
                            "pair": "USD_CAD",
                            "direction": "LONG",
                            "method": "BREAKOUT_FAILURE",
                            "status": "CANDIDATE",
                            "live_n": 2,
                            "live_net_jpy": 664.0,
                        },
                    ]
                },
            )
            _append_jsonl(
                paths["forecast_history"],
                [
                    {
                        "timestamp_utc": "2026-07-09T10:14:56.370426Z",
                        "pair": "EUR_USD",
                        "direction": "DOWN",
                    },
                    {
                        "timestamp_utc": "2026-07-09T10:14:56.370426Z",
                        "pair": "USD_CAD",
                        "direction": "RANGE",
                    },
                ],
            )
            _append_jsonl(paths["projection_ledger"], [])
            _write_execution_db(paths["execution_db"])

            EntryFrequencyRecovery(
                active_trader_contract_path=paths["active_contract"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                order_intents_path=paths["order_intents"],
                strategy_profile_path=paths["strategy_profile"],
                execution_ledger_db_path=paths["execution_db"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        prompt = payload["next_contract_prompt"]
        self.assertEqual(payload["top_lane"]["lane_id"], top_lane_id)
        self.assertEqual(payload["forecast_pattern_tuning_queue"][0]["lane_id"], other_lane_id)
        self.assertIn(top_lane_id, prompt)
        self.assertIn("EXACT_TP_PROOF_COLLECTION", prompt)
        self.assertNotIn("retune USD_CAD", prompt)

    def test_no_entry_drought_target_stays_read_only_noop(self) -> None:
        now = datetime(2026, 7, 9, 10, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _paths(root)
            for key in ("active_contract", "active_board", "frontier", "order_intents", "strategy_profile"):
                _write_json(paths[key], {"generated_at_utc": now.isoformat(), "results": []})
            _append_jsonl(paths["forecast_history"], [])
            _append_jsonl(paths["projection_ledger"], [])
            _write_execution_db(paths["execution_db"])

            EntryFrequencyRecovery(
                active_trader_contract_path=paths["active_contract"],
                active_opportunity_board_path=paths["active_board"],
                non_eurusd_live_grade_frontier_path=paths["frontier"],
                order_intents_path=paths["order_intents"],
                strategy_profile_path=paths["strategy_profile"],
                execution_ledger_db_path=paths["execution_db"],
                forecast_history_path=paths["forecast_history"],
                projection_ledger_path=paths["projection_ledger"],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], "NO_ENTRY_FREQUENCY_RECOVERY_TARGETS")
        self.assertEqual(payload["target_lanes"], [])
        self.assertFalse(payload["live_permission_allowed"])


def _paths(root: Path) -> dict[str, Path]:
    return {
        "active_contract": root / "data" / "active_trader_contract.json",
        "active_board": root / "data" / "active_opportunity_board.json",
        "frontier": root / "data" / "non_eurusd_live_grade_frontier.json",
        "order_intents": root / "data" / "order_intents.json",
        "strategy_profile": root / "data" / "strategy_profile.json",
        "execution_db": root / "data" / "execution_ledger.db",
        "forecast_history": root / "data" / "forecast_history.jsonl",
        "projection_ledger": root / "data" / "projection_ledger.jsonl",
        "output": root / "data" / "entry_frequency_recovery.json",
        "report": root / "docs" / "entry_frequency_recovery_report.md",
    }


if __name__ == "__main__":
    unittest.main()
