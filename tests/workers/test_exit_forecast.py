from __future__ import annotations

import pytest

from workers.common.exit_forecast import (
    apply_exit_forecast_to_loss_cut,
    apply_exit_forecast_to_targets,
    build_exit_forecast_adjustment,
)


def test_build_adjustment_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXIT_FORECAST_ENABLED", "0")
    adj = build_exit_forecast_adjustment(side="long", entry_thesis={"forecast": {"p_up": 0.2}})
    assert adj.enabled is False
    assert adj.contra_score == 0.0
    assert adj.profit_take_mult == 1.0


def test_build_adjustment_contra_tightens_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXIT_FORECAST_ENABLED", raising=False)
    monkeypatch.setenv("EXIT_FORECAST_MIN_AGAINST_PROB", "0.54")
    adj = build_exit_forecast_adjustment(
        side="long",
        entry_thesis={
            "forecast": {
                "p_up": 0.18,
                "edge": 0.77,
                "reason": "edge_block",
            }
        },
    )
    assert adj.enabled is True
    assert adj.against_prob is not None and adj.against_prob > 0.8
    assert adj.contra_score > 0.4
    assert adj.loss_cut_mult < 1.0
    assert adj.profit_take_mult < 1.0
    assert adj.lock_buffer_mult > 1.0


def test_apply_adjustments_for_targets_and_loss_cut() -> None:
    adj = build_exit_forecast_adjustment(
        side="short",
        entry_thesis={
            "forecast": {"p_up": 0.85, "edge": 0.8},
        },
    )
    profit_take, trail_start, trail_backoff, lock_buffer = apply_exit_forecast_to_targets(
        profit_take=3.0,
        trail_start=2.0,
        trail_backoff=1.0,
        lock_buffer=0.5,
        adjustment=adj,
    )
    assert profit_take <= 3.0
    assert trail_start <= 2.0
    assert trail_backoff <= 1.0
    assert lock_buffer >= 0.5

    soft, hard, hold = apply_exit_forecast_to_loss_cut(
        soft_pips=10.0,
        hard_pips=16.0,
        max_hold_sec=1200.0,
        adjustment=adj,
    )
    assert soft <= 10.0
    assert hard <= 16.0
    assert hard >= soft
    assert hold is not None and hold <= 1200.0
