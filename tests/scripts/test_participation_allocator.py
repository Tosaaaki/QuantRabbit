from __future__ import annotations

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
