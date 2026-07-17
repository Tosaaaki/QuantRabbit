from __future__ import annotations

import pytest

from quant_rabbit.conviction_ladder import (
    allowed_risk_fraction,
    declared_condition_count,
)


def test_ladder_tiers_are_predeclared_and_capped() -> None:
    base = allowed_risk_fraction(
        conviction_conditions_met=0,
        today_nav_return_fraction=0.0,
        consecutive_losing_trades=0,
    )
    assert base.risk_fraction == pytest.approx(0.0025)
    assert base.multiplier == 1.0

    strong = allowed_risk_fraction(
        conviction_conditions_met=3,
        today_nav_return_fraction=0.0,
        consecutive_losing_trades=0,
    )
    assert strong.risk_fraction == pytest.approx(0.005)

    # Beyond four conditions the multiplier stays capped at 4x.
    maxed = allowed_risk_fraction(
        conviction_conditions_met=7,
        today_nav_return_fraction=0.0,
        consecutive_losing_trades=0,
    )
    assert maxed.risk_fraction == pytest.approx(0.01)
    assert maxed.multiplier == 4.0


def test_daily_stop_and_losing_streak_throttle() -> None:
    stopped = allowed_risk_fraction(
        conviction_conditions_met=4,
        today_nav_return_fraction=-0.031,
        consecutive_losing_trades=0,
    )
    assert stopped.risk_fraction == 0.0
    assert stopped.daily_stop_engaged is True

    halved = allowed_risk_fraction(
        conviction_conditions_met=4,
        today_nav_return_fraction=0.0,
        consecutive_losing_trades=3,
    )
    assert halved.multiplier == 2.0
    assert "HALVED_FOR_LOSING_STREAK" in halved.reason

    boosted = allowed_risk_fraction(
        conviction_conditions_met=0,
        today_nav_return_fraction=0.0,
        consecutive_losing_trades=0,
        prior_week_nav_return_fraction=0.06,
    )
    assert boosted.risk_fraction == pytest.approx(0.0025 * 1.25)


def test_condition_checklist_is_strict() -> None:
    assert (
        declared_condition_count(
            [
                ("REGIME_ALIGNED", True),
                ("SESSION_FAVORABLE", True),
                ("NO_EVENT_WINDOW", False),
            ]
        )
        == 2
    )
    with pytest.raises(ValueError, match="checklist"):
        declared_condition_count([])
    with pytest.raises(ValueError, match="strict boolean"):
        declared_condition_count([("REGIME_ALIGNED", 1)])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="non-negative"):
        allowed_risk_fraction(
            conviction_conditions_met=-1,
            today_nav_return_fraction=0.0,
            consecutive_losing_trades=0,
        )
