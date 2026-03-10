from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts import participation_allocator


def test_build_participation_alloc_trims_overused_loser_and_boosts_winner() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "RangeFader-neutral-fade": {
                "pocket": "scalp",
                "attempts": 120,
                "fills": 3,
                "filled_rate": 0.025,
                "attempt_share": 0.70,
                "fill_share": 0.15,
                "share_gap": 0.55,
                "terminal_status_counts": {"perf_block": 45, "filled": 3},
            },
            "MomentumBurst": {
                "pocket": "micro",
                "attempts": 35,
                "fills": 14,
                "filled_rate": 0.40,
                "attempt_share": 0.20,
                "fill_share": 0.55,
                "share_gap": -0.35,
                "terminal_status_counts": {"filled": 14},
            },
        },
    }
    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "RangeFader-neutral-fade": -1200.0,
            "MomentumBurst": 950.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    loser = payload["strategies"]["RangeFader-neutral-fade"]
    winner = payload["strategies"]["MomentumBurst"]

    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 1.0
    assert loser["hard_block_rate"] > 0.0
    assert winner["action"] == "boost_participation"
    assert winner["lot_multiplier"] > 1.0
    assert winner["probability_boost"] > 0.0
    assert winner["cadence_floor"] > 1.0


def test_build_participation_alloc_boosts_profitable_small_sample_winner() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "PrecisionLowVol": {
                "pocket": "scalp",
                "attempts": 10,
                "fills": 8,
                "filled_rate": 0.80,
                "attempt_share": 0.05,
                "fill_share": 0.18,
                "share_gap": -0.13,
                "terminal_status_counts": {"filled": 8},
            },
            "RangeFader-neutral-fade": {
                "pocket": "scalp",
                "attempts": 120,
                "fills": 3,
                "filled_rate": 0.025,
                "attempt_share": 0.70,
                "fill_share": 0.15,
                "share_gap": 0.55,
                "terminal_status_counts": {"perf_block": 45, "filled": 3},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "PrecisionLowVol": 220.0,
            "RangeFader-neutral-fade": -1200.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    winner = payload["strategies"]["PrecisionLowVol"]

    assert winner["action"] == "boost_participation"
    assert winner["lot_multiplier"] > 1.0
    assert winner["probability_boost"] > 0.0
    assert winner["cadence_floor"] > 1.0


def test_build_participation_alloc_boosts_four_trade_session_breakout_winner() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "session_open_breakout": {
                "pocket": "micro",
                "attempts": 4,
                "fills": 4,
                "filled_rate": 1.0,
                "attempt_share": 0.002,
                "fill_share": 0.009,
                "share_gap": -0.007,
                "terminal_status_counts": {"filled": 4},
            },
            "RangeFader-buy-fade": {
                "pocket": "scalp",
                "attempts": 120,
                "fills": 10,
                "filled_rate": 0.0833,
                "attempt_share": 0.50,
                "fill_share": 0.20,
                "share_gap": 0.30,
                "terminal_status_counts": {"perf_block": 40, "filled": 10},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "session_open_breakout": 12.0,
            "RangeFader-buy-fade": -120.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    winner = payload["strategies"]["session_open_breakout"]

    assert winner["action"] == "boost_participation"
    assert winner["lot_multiplier"] > 1.0
    assert winner["probability_boost"] > 0.0
    assert winner["cadence_floor"] > 1.0


def test_load_recent_realized_jpy_prefers_lane_tag_from_entry_thesis(tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                strategy_tag TEXT,
                strategy TEXT,
                entry_thesis TEXT,
                realized_pl REAL,
                close_time TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO trades(strategy_tag, strategy, entry_thesis, realized_pl, close_time)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "RangeFader",
                "RangeFader",
                '{"strategy":"RangeFader","strategy_tag":"RangeFader","strategy_tag_raw":"RangeFader-sell-fade"}',
                -220.0,
                "2026-03-10T00:00:00Z",
            ),
        )
        conn.commit()

    realized = participation_allocator._load_recent_realized_jpy(db_path, lookback_hours=48.0)

    assert realized["RangeFader-sell-fade"] == -220.0
    assert "RangeFader" not in realized
