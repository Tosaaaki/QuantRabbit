from __future__ import annotations

import json
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


def test_build_participation_alloc_trims_underused_high_fill_loser() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "RangeFader-neutral-fade": {
                "pocket": "scalp",
                "attempts": 367,
                "fills": 66,
                "filled_rate": 0.1798,
                "attempt_share": 0.150225,
                "fill_share": 0.176,
                "share_gap": -0.025775,
                "terminal_status_counts": {
                    "entry_probability_reject": 116,
                    "filled": 66,
                    "perf_block": 237,
                },
            },
            "MomentumBurst-open_long": {
                "pocket": "micro",
                "attempts": 8,
                "fills": 8,
                "filled_rate": 1.0,
                "attempt_share": 0.006,
                "fill_share": 0.026,
                "share_gap": -0.02,
                "terminal_status_counts": {"filled": 8},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "RangeFader-neutral-fade": -46.15,
            "MomentumBurst-open_long": 185.32,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.18,
        max_prob_boost=0.08,
    )

    loser = payload["strategies"]["RangeFader-neutral-fade"]

    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 1.0
    assert loser["probability_offset"] < 0.0
    assert loser["cadence_floor"] < 1.0


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


def test_build_participation_alloc_expands_small_sample_winner_boost_with_higher_caps() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "MomentumBurst-open_long": {
                "pocket": "micro",
                "attempts": 5,
                "fills": 5,
                "filled_rate": 1.0,
                "attempt_share": 0.002,
                "fill_share": 0.013,
                "share_gap": -0.011,
                "terminal_status_counts": {"filled": 5},
            },
            "RangeFader-buy-fade": {
                "pocket": "scalp",
                "attempts": 240,
                "fills": 18,
                "filled_rate": 0.075,
                "attempt_share": 0.40,
                "fill_share": 0.12,
                "share_gap": 0.28,
                "terminal_status_counts": {"perf_block": 88, "filled": 18},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "MomentumBurst-open_long": 185.32,
            "RangeFader-buy-fade": -74.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.18,
        max_prob_boost=0.08,
    )

    winner = payload["strategies"]["MomentumBurst-open_long"]

    assert winner["action"] == "boost_participation"
    assert winner["lot_multiplier"] > 1.14
    assert winner["probability_boost"] >= 0.05
    assert winner["cadence_floor"] > 1.17


def test_build_participation_alloc_boosts_two_trade_fast_winner_lane() -> None:
    summary = {
        "lookback_hours": 6.0,
        "strategies": {
            "MomentumBurst-open_long": {
                "pocket": "micro",
                "attempts": 2,
                "fills": 2,
                "filled_rate": 1.0,
                "attempt_share": 0.0017,
                "fill_share": 0.0377,
                "share_gap": -0.0360,
                "terminal_status_counts": {"filled": 2},
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
            "MomentumBurst-open_long": 185.32,
            "RangeFader-buy-fade": -120.0,
        },
        min_attempts=12,
        max_units_cut=0.22,
        max_units_boost=0.24,
        max_prob_boost=0.10,
    )

    winner = payload["strategies"]["MomentumBurst-open_long"]

    assert winner["action"] == "boost_participation"
    assert winner["lot_multiplier"] > 1.18
    assert winner["probability_boost"] >= 0.06
    assert winner["cadence_floor"] > 1.17


def test_build_participation_alloc_trims_recent_loss_drag_lane_sooner() -> None:
    summary = {
        "lookback_hours": 6.0,
        "strategies": {
            "DroughtRevert": {
                "pocket": "scalp",
                "attempts": 13,
                "fills": 10,
                "filled_rate": 0.7692,
                "attempt_share": 0.0098,
                "fill_share": 0.0751,
                "share_gap": -0.0653,
                "terminal_status_counts": {"filled": 10},
            },
            "MomentumBurst-open_long": {
                "pocket": "micro",
                "attempts": 2,
                "fills": 2,
                "filled_rate": 1.0,
                "attempt_share": 0.0017,
                "fill_share": 0.0377,
                "share_gap": -0.0360,
                "terminal_status_counts": {"filled": 2},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "DroughtRevert": -33.301,
            "MomentumBurst-open_long": 185.32,
        },
        min_attempts=12,
        max_units_cut=0.22,
        max_units_boost=0.24,
        max_prob_boost=0.10,
    )

    loser = payload["strategies"]["DroughtRevert"]

    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 0.85
    assert loser["probability_offset"] < 0.0
    assert loser["cadence_floor"] < 1.0


def test_build_participation_alloc_emits_setup_overrides_for_loser_setup() -> None:
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
                "setups": {
                    "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression": {
                        "setup_fingerprint": "RangeFader-sell-fade|short|trend_long|tight_fast|rsi:overbought|atr:mid|gap:up_extended|volatility_compression",
                        "flow_regime": "trend_long",
                        "microstructure_bucket": "tight_fast",
                        "attempts": 244,
                        "fills": 8,
                        "filled_rate": 0.0328,
                        "attempt_share": 0.122,
                        "fill_share": 0.018,
                        "share_gap": 0.104,
                        "terminal_status_counts": {
                            "entry_probability_reject": 402,
                            "perf_block": 231,
                            "filled": 8,
                        },
                    },
                    "RangeFader-sell-fade|short|transition|normal_normal|rsi:mid|atr:mid|gap:down_flat|volatility_compression": {
                        "setup_fingerprint": "RangeFader-sell-fade|short|transition|normal_normal|rsi:mid|atr:mid|gap:down_flat|volatility_compression",
                        "flow_regime": "transition",
                        "microstructure_bucket": "normal_normal",
                        "attempts": 88,
                        "fills": 12,
                        "filled_rate": 0.1364,
                        "attempt_share": 0.041,
                        "fill_share": 0.038,
                        "share_gap": 0.003,
                        "terminal_status_counts": {
                            "entry_probability_reject": 53,
                            "filled": 12,
                        },
                    },
                },
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
            "RangeFader-sell-fade": -220.0,
            "MomentumBurst": 180.0,
        },
        min_attempts=20,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    lane = payload["strategies"]["RangeFader-sell-fade"]
    assert isinstance(lane.get("setup_overrides"), list)
    loser = next(
        item
        for item in lane["setup_overrides"]
        if item["match_dimension"] == "setup_fingerprint"
        and item["setup_fingerprint"].startswith("RangeFader-sell-fade|short|trend_long|tight_fast|")
    )
    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 1.0
    assert loser["flow_regime"] == "trend_long"
    assert loser["microstructure_bucket"] == "tight_fast"


def test_build_participation_alloc_emits_low_sample_precision_setup_overrides() -> None:
    summary = {
        "lookback_hours": 24.0,
        "strategies": {
            "PrecisionLowVol": {
                "pocket": "scalp",
                "attempts": 26,
                "fills": 23,
                "filled_rate": 0.8846,
                "attempt_share": 0.08,
                "fill_share": 0.18,
                "share_gap": -0.10,
                "terminal_status_counts": {"filled": 23},
                "setups": {
                    "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression": {
                        "setup_fingerprint": "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression",
                        "flow_regime": "range_fade",
                        "microstructure_bucket": "unknown",
                        "attempts": 15,
                        "fills": 3,
                        "filled_rate": 0.20,
                        "attempt_share": 0.009,
                        "fill_share": 0.062,
                        "share_gap": -0.053,
                        "terminal_status_counts": {"filled": 3},
                    },
                    "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:down_flat|volatility_compression": {
                        "setup_fingerprint": "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:down_flat|volatility_compression",
                        "flow_regime": "range_fade",
                        "microstructure_bucket": "unknown",
                        "attempts": 25,
                        "fills": 2,
                        "filled_rate": 0.08,
                        "attempt_share": 0.121,
                        "fill_share": 0.019,
                        "share_gap": 0.102,
                        "terminal_status_counts": {"perf_block": 12, "filled": 2},
                    },
                },
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={"PrecisionLowVol": -6.11},
        realized_by_setup={
            json.dumps(
                {
                    "strategy_key": "PrecisionLowVol",
                    "setup_fingerprint": "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression",
                    "flow_regime": "range_fade",
                    "microstructure_bucket": "unknown",
                },
                sort_keys=True,
                ensure_ascii=True,
            ): 51.03,
            json.dumps(
                {
                    "strategy_key": "PrecisionLowVol",
                    "setup_fingerprint": "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:down_flat|volatility_compression",
                    "flow_regime": "range_fade",
                    "microstructure_bucket": "unknown",
                },
                sort_keys=True,
                ensure_ascii=True,
            ): -57.14,
        },
        min_attempts=20,
        setup_min_attempts=4,
        max_units_cut=0.18,
        max_units_boost=0.12,
        max_prob_boost=0.05,
    )

    overrides = payload["strategies"]["PrecisionLowVol"]["setup_overrides"]
    winner = next(
        item
        for item in overrides
        if item.get("setup_fingerprint")
        == "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:up_lean|volatility_compression"
    )
    loser = next(
        item
        for item in overrides
        if item.get("setup_fingerprint")
        == "PrecisionLowVol|short|range_fade|unknown|rsi:overbought|atr:low|gap:down_flat|volatility_compression"
    )

    assert payload["allocation_policy"]["setup_min_attempts"] == 4
    assert winner["action"] == "boost_participation"
    assert winner["lot_multiplier"] > 1.0
    assert winner["max_units_boost"] == 0.12
    assert winner["max_probability_boost"] == 0.05


def test_build_participation_alloc_emits_two_attempt_loser_setup_override() -> None:
    summary = {
        "lookback_hours": 6.0,
        "strategies": {
            "DroughtRevert": {
                "pocket": "scalp",
                "attempts": 12,
                "fills": 6,
                "filled_rate": 0.5,
                "attempt_share": 0.05,
                "fill_share": 0.08,
                "share_gap": -0.03,
                "terminal_status_counts": {"filled": 6},
                "setups": {
                    "DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:down_flat|volatility_compression": {
                        "setup_fingerprint": "DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:down_flat|volatility_compression",
                        "flow_regime": "range_fade",
                        "microstructure_bucket": "unknown",
                        "attempts": 2,
                        "fills": 2,
                        "filled_rate": 1.0,
                        "attempt_share": 0.01,
                        "fill_share": 0.03,
                        "share_gap": -0.02,
                        "terminal_status_counts": {"filled": 2},
                    },
                },
            },
            "MomentumBurst-open_long": {
                "pocket": "micro",
                "attempts": 2,
                "fills": 2,
                "filled_rate": 1.0,
                "attempt_share": 0.0017,
                "fill_share": 0.0377,
                "share_gap": -0.0360,
                "terminal_status_counts": {"filled": 2},
            },
        },
    }

    payload = participation_allocator.build_participation_alloc(
        summary,
        realized_by_strategy={
            "DroughtRevert": -10.279,
            "MomentumBurst-open_long": 185.32,
        },
        realized_by_setup={
            json.dumps(
                {
                    "strategy_key": "DroughtRevert",
                    "setup_fingerprint": "DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:down_flat|volatility_compression",
                    "flow_regime": "range_fade",
                    "microstructure_bucket": "unknown",
                },
                sort_keys=True,
                ensure_ascii=True,
            ): -10.279,
        },
        min_attempts=12,
        setup_min_attempts=2,
        max_units_cut=0.22,
        max_units_boost=0.24,
        max_prob_boost=0.10,
    )

    overrides = payload["strategies"]["DroughtRevert"]["setup_overrides"]
    loser = next(
        item
        for item in overrides
        if item.get("setup_fingerprint")
        == "DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:down_flat|volatility_compression"
    )

    assert payload["allocation_policy"]["setup_min_attempts"] == 2
    assert loser["action"] == "trim_units"
    assert loser["lot_multiplier"] < 1.0
    assert loser["probability_offset"] < 0.0


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


def test_load_recent_realized_setup_jpy_derives_setup_key_from_technical_context(tmp_path: Path) -> None:
    db_path = tmp_path / "trades.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                strategy_tag TEXT,
                strategy TEXT,
                units INTEGER,
                entry_thesis TEXT,
                realized_pl REAL,
                close_time TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO trades(strategy_tag, strategy, units, entry_thesis, realized_pl, close_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "RangeFader-sell-fade",
                "RangeFader-sell-fade",
                -120,
                json.dumps(
                    {
                        "strategy_tag": "RangeFader-sell-fade",
                        "range_mode": "trend",
                        "range_score": 0.18,
                        "spread_pips": 0.8,
                        "technical_context": {
                            "ticks": {"spread_pips": 0.8, "tick_rate": 9.2},
                            "indicators": {
                                "M1": {
                                    "atr_pips": 2.4,
                                    "rsi": 67.0,
                                    "adx": 29.0,
                                    "plus_di": 31.0,
                                    "minus_di": 14.0,
                                    "ma10": 158.110,
                                    "ma20": 158.080,
                                }
                            },
                        },
                    },
                    ensure_ascii=True,
                ),
                -88.0,
                "2026-03-10T00:00:00Z",
            ),
        )
        conn.commit()

    realized = participation_allocator._load_recent_realized_setup_jpy(db_path, lookback_hours=48.0)

    key = next(iter(realized.keys()))
    assert "RangeFader-sell-fade" in key
    assert "trend_long" in key
    assert "tight_fast" in key
    assert realized[key] == -88.0
