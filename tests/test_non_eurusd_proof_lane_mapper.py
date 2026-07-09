from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.non_eurusd_proof_lane_mapper import (
    NonEurusdProofLaneMapper,
    STATUS_EVIDENCE_PATH,
    STATUS_MAPPING_GAPS,
    STATUS_NO_VALID_PATH,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_execution_db(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            con.execute(
                """
                insert into execution_events
                    (ts_utc,event_type,lane_id,pair,side,realized_pl_jpy,exit_reason)
                values (?,?,?,?,?,?,?)
                """,
                (
                    row.get("ts_utc"),
                    row.get("event_type"),
                    row.get("lane_id"),
                    row.get("pair"),
                    row.get("side"),
                    row.get("realized_pl_jpy"),
                    row.get("exit_reason"),
                ),
            )
        con.commit()
    finally:
        con.close()


class NonEurusdProofLaneMapperTests(unittest.TestCase):
    def test_pair_side_only_take_profit_stays_unmapped_not_exact_proof(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            paths = _base_paths(Path(tmp))
            _write_base_json(paths, lanes=[_lane("failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT")])
            _write_execution_db(
                paths["execution_db"],
                [
                    {
                        "ts_utc": "2026-07-01T00:00:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": None,
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "realized_pl_jpy": 500.0,
                        "exit_reason": "TAKE_PROFIT_ORDER",
                    },
                    {
                        "ts_utc": "2026-07-02T00:00:00+00:00",
                        "event_type": "TRADE_CLOSED",
                        "lane_id": None,
                        "pair": "USD_CAD",
                        "side": "LONG",
                        "realized_pl_jpy": 616.0,
                        "exit_reason": "TAKE_PROFIT_ORDER",
                    },
                ],
            )

            NonEurusdProofLaneMapper(
                active_opportunity_board_path=paths["active_board"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                lane_candidate_board_path=paths["lane_board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_MAPPING_GAPS)
        self.assertEqual(payload["mapped_lanes"], [])
        self.assertEqual(payload["unmapped_profit_evidence"][0]["evidence"]["evidence_type"], "EXECUTION_TP_PROOF")
        self.assertFalse(payload["unmapped_profit_evidence"][0]["mapping_allowed_as_tp_proof"])
        self.assertEqual(payload["unmapped_profit_evidence"][0]["possible_current_lanes"][0]["lane_id"], "failure_trader:USD_CAD:LONG:BREAKOUT_FAILURE:LIMIT")

    def test_exact_positive_tp_with_bidask_negative_is_evidence_path_not_live_permission(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:AUD_JPY:SHORT:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _base_paths(Path(tmp))
            lane = _lane(
                lane_id,
                blockers=["BIDASK_REPLAY_NEGATIVE_EXPECTANCY_FOR_LIVE"],
                can_enter_proof_pack=True,
                local_tp_trades=6,
                local_tp_expectancy=992.7,
            )
            lane["replay_status"] = "NEGATIVE_EVIDENCE_REFRESH_REQUIRED"
            _write_base_json(paths, lanes=[lane])
            _write_execution_db(paths["execution_db"], [])

            NonEurusdProofLaneMapper(
                active_opportunity_board_path=paths["active_board"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                lane_candidate_board_path=paths["lane_board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_EVIDENCE_PATH)
        self.assertFalse(payload["live_permission_allowed"])
        self.assertEqual(payload["live_side_effects"], [])
        self.assertIn(lane_id, payload["next_active_path"])
        mapped = payload["mapped_lanes"][0]
        self.assertEqual(mapped["promotion_assessment"], "EVIDENCE_ACQUISITION_CANDIDATE")
        self.assertTrue(mapped["bidask_replay_negative"])
        self.assertEqual(payload["bidask_replay_gaps"][0]["lane_id"], lane_id)
        self.assertEqual(payload["proof_floor_gaps"][0]["remaining_tp_trades"], 14)

    def test_spread_too_wide_blocks_live_ready_assessment(self) -> None:
        now = datetime(2026, 7, 9, 0, 0, tzinfo=timezone.utc)
        lane_id = "failure_trader:GBP_USD:LONG:BREAKOUT_FAILURE:LIMIT"
        with tempfile.TemporaryDirectory() as tmp:
            paths = _base_paths(Path(tmp))
            lane = _lane(
                lane_id,
                status="LIVE_READY",
                blockers=["SPREAD_TOO_WIDE"],
                local_tp_trades=20,
                local_tp_expectancy=523.3,
            )
            lane["spread_status"] = "BLOCKED"
            _write_base_json(paths, lanes=[lane])
            _write_execution_db(paths["execution_db"], [])

            NonEurusdProofLaneMapper(
                active_opportunity_board_path=paths["active_board"],
                payoff_shape_diagnosis_path=paths["payoff"],
                proof_pack_queue_path=paths["proof_queue"],
                lane_candidate_board_path=paths["lane_board"],
                portfolio_4x_path_planner_path=paths["portfolio"],
                order_intents_path=paths["order_intents"],
                verification_ledger_path=paths["verification"],
                execution_ledger_db_path=paths["execution_db"],
                replay_artifact_paths=[],
                output_path=paths["output"],
                report_path=paths["report"],
                now_utc=now,
            ).run()
            payload = json.loads(paths["output"].read_text())

        self.assertEqual(payload["status"], STATUS_NO_VALID_PATH)
        self.assertEqual(payload["mapped_lanes"][0]["promotion_assessment"], "BLOCKED_SPREAD_GAP_VISIBLE")
        self.assertEqual(payload["spread_gaps"][0]["lane_id"], lane_id)
        self.assertNotIn("NON_EURUSD_LIVE_READY_FOUND", payload["next_active_path"])


def _base_paths(root: Path) -> dict[str, Path]:
    return {
        "active_board": root / "data" / "active_opportunity_board.json",
        "payoff": root / "data" / "payoff_shape_diagnosis.json",
        "proof_queue": root / "data" / "as_proof_pack_queue.json",
        "lane_board": root / "data" / "as_lane_candidate_board.json",
        "portfolio": root / "data" / "portfolio_4x_path_planner.json",
        "order_intents": root / "data" / "order_intents.json",
        "verification": root / "data" / "verification_ledger.json",
        "execution_db": root / "data" / "execution_ledger.db",
        "output": root / "data" / "non_eurusd_proof_lane_mapper.json",
        "report": root / "docs" / "non_eurusd_proof_lane_mapper.md",
    }


def _write_base_json(paths: dict[str, Path], *, lanes: list[dict[str, Any]]) -> None:
    _write_json(
        paths["active_board"],
        {
            "status": "BOARD_BUILT",
            "read_only": True,
            "live_side_effects": [],
            "live_permission_allowed": False,
            "ranked_active_lanes": lanes,
        },
    )
    _write_json(paths["payoff"], {"harvest_candidates": [], "partial_tp_runner_candidates": [], "runner_candidates": []})
    _write_json(paths["proof_queue"], {"queue": [], "rejected_candidates": []})
    _write_json(paths["lane_board"], {"closest_candidate_to_proof_pack": {}})
    _write_json(paths["portfolio"], {"candidate_rankings": []})
    _write_json(paths["order_intents"], {"results": []})
    _write_json(paths["verification"], {"blocking_evidence": [], "learning_evidence": []})


def _lane(
    lane_id: str,
    *,
    status: str = "OPERATOR_REVIEW_REQUIRED",
    blockers: list[str] | None = None,
    can_enter_proof_pack: bool = False,
    local_tp_trades: int | None = None,
    local_tp_expectancy: float | None = None,
) -> dict[str, Any]:
    parts = lane_id.split(":")
    pair = parts[1]
    side = parts[2]
    method = parts[3]
    vehicle = parts[4] if len(parts) > 4 else "STOP"
    lane = {
        "lane_id": lane_id,
        "pair": pair,
        "direction": side,
        "strategy_family": method,
        "vehicle": vehicle,
        "status": status,
        "rank_score": 100.0,
        "proof_status": "UNKNOWN",
        "spread_status": "OBSERVED_NOT_BLOCKED",
        "replay_status": "UNKNOWN",
        "blockers": blockers or [],
        "source_refs": ["data/order_intents.json"],
        "can_enter_proof_pack": can_enter_proof_pack,
        "live_permission_allowed": False,
    }
    if local_tp_trades is not None:
        lane["local_tp_proof"] = {
            "attach_take_profit_on_fill": True,
            "capture_take_profit_proof_floor": 20,
            "capture_take_profit_scope": "PAIR_SIDE_METHOD",
            "capture_take_profit_scope_key": f"{pair}|{side}|{method}|TAKE_PROFIT_ORDER",
            "capture_take_profit_trades": local_tp_trades,
            "capture_take_profit_wins": local_tp_trades,
            "capture_take_profit_losses": 0,
            "capture_take_profit_expectancy_jpy": local_tp_expectancy,
            "capture_take_profit_avg_win_jpy": local_tp_expectancy,
            "capture_take_profit_avg_loss_jpy": 0.0,
            "tp_execution_mode": "ATTACHED_TECHNICAL_TP",
            "tp_target_intent": "HARVEST",
        }
    return lane


if __name__ == "__main__":
    unittest.main()
