"""Pre-entry close-distance gate (weakness ledger S1 / close-leak family).

The market-close leak family dominated both the live TP-edge diagnosis
(`MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE`) and the exact-S5 TRAIN sign flip
(60 weekend-crossing trades, -1,438.8 pips).  This module gives every lane
one shared, purely pre-entry admission decision: an entry is admitted only
when the FX market is open and the planned hold plus a safety margin ends
before the next market close.  It reads only the decision clock — never a
realized exit — so it can never delete a filled trade after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from quant_rabbit.analysis.market_status import compute_market_status

DEFAULT_SAFETY_MARGIN_MINUTES = 5
GATE_POLICY = "PRE_ENTRY_HOLD_PLUS_MARGIN_BEFORE_NEXT_FX_CLOSE_V1"


@dataclass(frozen=True, slots=True)
class CloseDistanceDecision:
    admitted: bool
    reason: str
    decision_utc: datetime
    hold_minutes: int
    safety_margin_minutes: int
    minutes_to_next_close: int | None

    def payload(self) -> dict[str, Any]:
        return {
            "policy": GATE_POLICY,
            "admitted": self.admitted,
            "reason": self.reason,
            "decision_utc": self.decision_utc.isoformat(),
            "hold_minutes": self.hold_minutes,
            "safety_margin_minutes": self.safety_margin_minutes,
            "minutes_to_next_close": self.minutes_to_next_close,
            "uses_post_entry_information": False,
        }


def evaluate_close_distance_gate(
    decision_utc: datetime,
    *,
    hold_minutes: int,
    safety_margin_minutes: int = DEFAULT_SAFETY_MARGIN_MINUTES,
) -> CloseDistanceDecision:
    """Admit an entry only when hold+margin fits before the next FX close."""

    if decision_utc.tzinfo is None:
        raise ValueError("decision clock must be timezone-aware")
    if isinstance(hold_minutes, bool) or not isinstance(hold_minutes, int):
        raise ValueError("hold_minutes must be an integer")
    if hold_minutes <= 0:
        raise ValueError("hold_minutes must be positive")
    if (
        isinstance(safety_margin_minutes, bool)
        or not isinstance(safety_margin_minutes, int)
        or safety_margin_minutes < 0
    ):
        raise ValueError("safety_margin_minutes must be a non-negative integer")

    status = compute_market_status(decision_utc)
    if not status.is_fx_open:
        return CloseDistanceDecision(
            admitted=False,
            reason="FX_MARKET_CLOSED",
            decision_utc=decision_utc,
            hold_minutes=hold_minutes,
            safety_margin_minutes=safety_margin_minutes,
            minutes_to_next_close=status.minutes_to_next_close,
        )
    remaining = status.minutes_to_next_close
    if remaining is None or remaining <= hold_minutes + safety_margin_minutes:
        return CloseDistanceDecision(
            admitted=False,
            reason="HOLD_WOULD_CROSS_NEXT_FX_CLOSE",
            decision_utc=decision_utc,
            hold_minutes=hold_minutes,
            safety_margin_minutes=safety_margin_minutes,
            minutes_to_next_close=remaining,
        )
    return CloseDistanceDecision(
        admitted=True,
        reason="HOLD_FITS_BEFORE_NEXT_FX_CLOSE",
        decision_utc=decision_utc,
        hold_minutes=hold_minutes,
        safety_margin_minutes=safety_margin_minutes,
        minutes_to_next_close=remaining,
    )


def max_admissible_hold_minutes(
    decision_utc: datetime,
    *,
    safety_margin_minutes: int = DEFAULT_SAFETY_MARGIN_MINUTES,
) -> int:
    """Largest hold that the gate would admit now (0 when market is closed).

    Lets a thesis-driven lane shorten its horizon to fit before the close
    instead of skipping the trade entirely.
    """

    status = compute_market_status(decision_utc)
    if not status.is_fx_open or status.minutes_to_next_close is None:
        return 0
    return max(0, status.minutes_to_next_close - safety_margin_minutes - 1)
