"""Pre-declared conviction ladder sizing (weakness ledger W14 / S4).

The operator's 2025 4x came from concentrating size on objectively strong
setups while surviving losing streaks.  This module freezes that behavior
into arithmetic declared before the trade: a base NAV risk fraction, an
escalation ladder driven only by a count of pre-declared objective
conditions, an equity-curve throttle, and a hard daily stop.  There is no
"judgement" input — anything discretionary must first become one of the
declared conditions.  Sizing output is a risk fraction; it grants no order
authority and never exceeds the declared cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

LADDER_POLICY = "PREDECLARED_CONVICTION_LADDER_V1"
BASE_RISK_FRACTION = 0.0025
LADDER_MULTIPLIERS = {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0, 4: 4.0}
MAX_MULTIPLIER = 4.0
DAILY_STOP_FRACTION = -0.03
LOSING_STREAK_HALVING_THRESHOLD = 3
WEEKLY_MOMENTUM_THRESHOLD = 0.05
WEEKLY_MOMENTUM_BASE_BOOST = 1.25


@dataclass(frozen=True, slots=True)
class SizingDecision:
    risk_fraction: float
    reason: str
    multiplier: float
    daily_stop_engaged: bool

    def payload(self) -> dict[str, Any]:
        return {
            "policy": LADDER_POLICY,
            "risk_fraction": self.risk_fraction,
            "reason": self.reason,
            "multiplier": self.multiplier,
            "daily_stop_engaged": self.daily_stop_engaged,
            "discretionary_inputs": False,
            "order_authority": "NONE",
        }


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def allowed_risk_fraction(
    *,
    conviction_conditions_met: int,
    today_nav_return_fraction: float,
    consecutive_losing_trades: int,
    prior_week_nav_return_fraction: float = 0.0,
    base_risk_fraction: float = BASE_RISK_FRACTION,
) -> SizingDecision:
    """Resolve today's per-trade risk fraction from declared inputs only."""

    if (
        isinstance(conviction_conditions_met, bool)
        or not isinstance(conviction_conditions_met, int)
        or conviction_conditions_met < 0
    ):
        raise ValueError("conviction_conditions_met must be a non-negative integer")
    if (
        isinstance(consecutive_losing_trades, bool)
        or not isinstance(consecutive_losing_trades, int)
        or consecutive_losing_trades < 0
    ):
        raise ValueError("consecutive_losing_trades must be a non-negative integer")
    base = _fraction(base_risk_fraction, "base_risk_fraction")
    if base <= 0:
        raise ValueError("base_risk_fraction must be positive")
    today = _fraction(today_nav_return_fraction, "today_nav_return_fraction")
    prior_week = _fraction(
        prior_week_nav_return_fraction, "prior_week_nav_return_fraction"
    )

    if today <= DAILY_STOP_FRACTION:
        return SizingDecision(
            risk_fraction=0.0,
            reason="DAILY_STOP_ENGAGED",
            multiplier=0.0,
            daily_stop_engaged=True,
        )

    tier = min(conviction_conditions_met, max(LADDER_MULTIPLIERS))
    multiplier = min(LADDER_MULTIPLIERS[tier], MAX_MULTIPLIER)
    reason = f"LADDER_TIER_{tier}"
    if consecutive_losing_trades >= LOSING_STREAK_HALVING_THRESHOLD:
        multiplier /= 2.0
        reason += "_HALVED_FOR_LOSING_STREAK"
    effective_base = base
    if prior_week >= WEEKLY_MOMENTUM_THRESHOLD:
        effective_base = base * WEEKLY_MOMENTUM_BASE_BOOST
        reason += "_WEEKLY_MOMENTUM_BASE"
    return SizingDecision(
        risk_fraction=round(effective_base * multiplier, 12),
        reason=reason,
        multiplier=multiplier,
        daily_stop_engaged=False,
    )


def declared_condition_count(conditions: Sequence[tuple[str, bool]]) -> int:
    """Count met conditions from an explicit named checklist.

    Forcing callers to pass named booleans keeps every escalation input
    auditable; an empty checklist is refused so "0 conditions" is always a
    deliberate statement, never a missing wire.
    """

    if not conditions:
        raise ValueError("a declared condition checklist is required")
    count = 0
    for name, met in conditions:
        if not str(name).strip():
            raise ValueError("condition name is required")
        if met.__class__ is not bool:
            raise ValueError(f"condition {name} must be a strict boolean")
        count += met
    return count
