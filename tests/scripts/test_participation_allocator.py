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


def test_build_participation_alloc_can_emit_safe_probability_trim_for_severe_loser() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "MicroTrendRetest-long": {
                "pocket": "micro",
                "attempts": 140,
                "fills": 18,
                "filled_rate": 0.1286,
                "attempt_share": 0.36,
                "fill_share": 0.11,
                "share_gap": 0.25,
                "terminal_status_counts": {"perf_block": 39, "filled": 18},
            },
            "MomentumBurst": {
                "pocket": "micro",
                "attempts": 35,
                "fills": 14,
                "filled_rate": 0.40,
                "attempt_share": 0.12,
                "fill_share": 0.34,
                "share_gap": -0.22,
                "terminal_status_counts": {"filled": 14},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "MicroTrendRetest-long": -420.0,
            "MomentumBurst": 180.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    loser = payload["strategies"]["MicroTrendRetest-long"]

    assert payload["allocation_policy"]["negative_probability_offsets_enabled"] is True
    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 1.0
    assert loser["probability_offset"] < 0.0
    assert loser["probability_multiplier"] == 1.0


def test_build_participation_alloc_keeps_mild_loser_on_units_trim_without_probability_offset() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "RangeFader-neutral-fade": {
                "pocket": "scalp",
                "attempts": 537,
                "fills": 35,
                "filled_rate": 0.0652,
                "attempt_share": 0.255228,
                "fill_share": 0.089059,
                "share_gap": 0.16617,
                "terminal_status_counts": {
                    "entry_probability_reject": 338,
                    "perf_block": 502,
                    "filled": 35,
                },
            },
            "MicroTrendRetest-long": {
                "pocket": "micro",
                "attempts": 31,
                "fills": 17,
                "filled_rate": 0.5484,
                "attempt_share": 0.10,
                "fill_share": 0.23,
                "share_gap": -0.13,
                "terminal_status_counts": {"filled": 17},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "RangeFader-neutral-fade": -4.218,
            "MicroTrendRetest-long": 42.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    loser = payload["strategies"]["RangeFader-neutral-fade"]

    assert payload["allocation_policy"]["negative_probability_offsets_enabled"] is False
    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 1.0
    assert loser["probability_offset"] == 0.0
    assert loser["probability_multiplier"] == 1.0


def test_build_participation_alloc_can_emit_probability_trim_for_high_reject_loser_with_small_loss() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "RangeFader-sell-fade": {
                "pocket": "scalp",
                "attempts": 578,
                "fills": 29,
                "filled_rate": 0.0502,
                "attempt_share": 0.274715,
                "fill_share": 0.073791,
                "share_gap": 0.200923,
                "terminal_status_counts": {
                    "entry_probability_reject": 891,
                    "perf_block": 549,
                    "filled": 29,
                },
            },
            "MicroTrendRetest-long": {
                "pocket": "micro",
                "attempts": 31,
                "fills": 17,
                "filled_rate": 0.5484,
                "attempt_share": 0.10,
                "fill_share": 0.23,
                "share_gap": -0.13,
                "terminal_status_counts": {"filled": 17},
            },
            "scalp_ping_5s_d_live": {
                "pocket": "scalp_fast",
                "attempts": 44,
                "fills": 29,
                "filled_rate": 0.6591,
                "attempt_share": 0.08,
                "fill_share": 0.18,
                "share_gap": -0.10,
                "terminal_status_counts": {"filled": 29},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "RangeFader-sell-fade": -7.866,
            "MicroTrendRetest-long": 42.0,
            "scalp_ping_5s_d_live": 9.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    loser = payload["strategies"]["RangeFader-sell-fade"]

    assert payload["allocation_policy"]["negative_probability_offsets_enabled"] is True
    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 1.0
    assert loser["probability_offset"] < 0.0
    assert loser["probability_multiplier"] == 1.0


def test_build_participation_alloc_keeps_buy_sell_probability_trim_but_exempts_neutral_fade() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "RangeFader-buy-fade": {
                "pocket": "scalp",
                "attempts": 668,
                "fills": 52,
                "filled_rate": 0.0778,
                "attempt_share": 0.31749,
                "fill_share": 0.132316,
                "share_gap": 0.185175,
                "terminal_status_counts": {"perf_block": 1800, "filled": 52},
            },
            "RangeFader-sell-fade": {
                "pocket": "scalp",
                "attempts": 578,
                "fills": 29,
                "filled_rate": 0.0502,
                "attempt_share": 0.274715,
                "fill_share": 0.073791,
                "share_gap": 0.200923,
                "terminal_status_counts": {"perf_block": 1438, "filled": 29},
            },
            "RangeFader-neutral-fade": {
                "pocket": "scalp",
                "attempts": 537,
                "fills": 35,
                "filled_rate": 0.0652,
                "attempt_share": 0.255228,
                "fill_share": 0.089059,
                "share_gap": 0.16617,
                "terminal_status_counts": {"perf_block": 840, "filled": 35},
            },
            "MomentumBurst": {
                "pocket": "micro",
                "attempts": 35,
                "fills": 14,
                "filled_rate": 0.4,
                "attempt_share": 0.12,
                "fill_share": 0.34,
                "share_gap": -0.22,
                "terminal_status_counts": {"filled": 14},
            },
            "PrecisionLowVol": {
                "pocket": "scalp",
                "attempts": 24,
                "fills": 10,
                "filled_rate": 0.4167,
                "attempt_share": 0.08,
                "fill_share": 0.18,
                "share_gap": -0.10,
                "terminal_status_counts": {"filled": 10},
            },
            "session_open_breakout": {
                "pocket": "micro",
                "attempts": 20,
                "fills": 9,
                "filled_rate": 0.45,
                "attempt_share": 0.05,
                "fill_share": 0.12,
                "share_gap": -0.07,
                "terminal_status_counts": {"filled": 9},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "RangeFader-buy-fade": -71.598,
            "RangeFader-sell-fade": -7.866,
            "RangeFader-neutral-fade": -4.218,
            "MomentumBurst": 180.0,
            "PrecisionLowVol": 33.0,
            "session_open_breakout": 12.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    buy = payload["strategies"]["RangeFader-buy-fade"]
    sell = payload["strategies"]["RangeFader-sell-fade"]
    neutral = payload["strategies"]["RangeFader-neutral-fade"]

    assert buy["action"] == "trim_units"
    assert buy["probability_offset"] < 0.0
    assert sell["action"] == "trim_units"
    assert sell["probability_offset"] < 0.0
    assert neutral["action"] == "trim_units"
    assert neutral["probability_offset"] == 0.0
    assert payload["allocation_policy"]["negative_probability_offsets_enabled"] is True


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


def test_build_participation_alloc_boosts_profitable_probe_lane_with_probability_rejects() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "scalp_ping_5s_c_live": {
                "pocket": "scalp_fast",
                "attempts": 1,
                "fills": 1,
                "filled_rate": 1.0,
                "attempt_share": 0.001,
                "fill_share": 0.005,
                "share_gap": -0.004,
                "terminal_status_counts": {
                    "entry_probability_reject": 42,
                    "strategy_control_entry_disabled": 8,
                    "filled": 1,
                },
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
            "scalp_ping_5s_c_live": 6.0,
            "RangeFader-buy-fade": -120.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    probe = payload["strategies"]["scalp_ping_5s_c_live"]

    assert probe["action"] == "boost_participation"
    assert probe["lot_multiplier"] > 1.0
    assert probe["probability_boost"] > 0.0
    assert probe["cadence_floor"] > 1.0
    assert 0.0 < probe["hard_block_rate"] < 1.0


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
