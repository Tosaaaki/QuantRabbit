#!/usr/bin/env python3
"""Policy arithmetic and saved sparse-cube invariants."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import build_path_metrics as bpm
import run_exit_replay as replay


HERE = Path(__file__).resolve().parent


class ExitReplayTests(unittest.TestCase):
    def test_price_pnl_side_orientation(self) -> None:
        self.assertEqual(replay.price_pnl(100, 101, 1000, "LONG"), 1000)
        self.assertEqual(replay.price_pnl(100, 99, 1000, "SHORT"), 1000)

    def test_adverse_market_fill_uses_side_correct_open_and_half_spread(self) -> None:
        candle = {"bid": {"o": "99"}, "ask": {"o": "101"}}
        self.assertEqual(replay.adverse_market_price(candle, "LONG"), (98.0, 1.0))
        self.assertEqual(replay.adverse_market_price(candle, "SHORT"), (102.0, 1.0))

    def test_stop_gap_is_conservative(self) -> None:
        candle = {"bid": {"o": "98"}, "ask": {"o": "100"}}
        self.assertEqual(replay.adverse_stop_price(candle, "LONG", 99), (97.0, 1.0))
        self.assertEqual(replay.adverse_stop_price(candle, "SHORT", 99), (101.0, 1.0))

    def test_changed_diagnostics_never_claim_actual_after_cost(self) -> None:
        rows = replay.read_jsonl(HERE / "exit_replay_rows_v1.jsonl")
        changed = [row for row in rows if row["changed"] is True]
        self.assertTrue(changed)
        self.assertTrue(all(row["candidate_actual_after_cost_net_jpy"] is None for row in changed))

    def test_action_times_are_causal_and_before_actual_close(self) -> None:
        rows = replay.read_jsonl(HERE / "exit_replay_rows_v1.jsonl")
        changed = [row for row in rows if row["changed"] is True]
        self.assertTrue(all(bpm.parse_ns(row["fill_at_utc"]) <= bpm.parse_ns(row["action_time"]) < bpm.parse_ns(row["close_at_utc"]) for row in changed))

    def test_corrected_baseline_is_preserved(self) -> None:
        rows = replay.read_jsonl(HERE / "exit_replay_rows_v1.jsonl")
        selected = [row for row in rows if row["window"] == "QUADRUPLE_64D" and row["split"] == "VALIDATION" and row["exit_policy"] == "BASELINE"]
        self.assertEqual(len(selected), 101)
        self.assertAlmostEqual(sum(row["candidate_actual_after_cost_net_jpy"] for row in selected), 11706.0523, places=7)

    def test_sparse_cube_keeps_null(self) -> None:
        rows = replay.read_jsonl(HERE / "exit_cube_long_v1.jsonl")
        self.assertEqual(len(rows), 8280)
        self.assertTrue(any(row["value"] is None for row in rows))
        self.assertFalse(any(row["value"] == 0 and row["admission_status"].startswith("NOT_EVALUABLE") for row in rows if row["metric"] == "actual_after_cost_net_jpy"))

    def test_no_interaction_or_pareto_runs_without_admitted_axis(self) -> None:
        report = json.loads((HERE / "exit_report_v1.json").read_text())
        self.assertEqual(report["next_cube_phase"], "NO_PROMISING_AXIS_ADMITTED_SO_TWO_AXIS_INTERACTIONS_AND_PARETO_NOT_RUN")
        self.assertFalse(report["holdout_used"])


if __name__ == "__main__":
    unittest.main()
