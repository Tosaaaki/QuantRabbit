"""Pre-declared high-cost time-window mask (weakness ledger W10).

The cost gate rejects expensive quotes one at a time, but the scheduler
still spends decisions inside hours that are reliably expensive (rollover
21-22 UTC, late Friday).  This mask lets a lane pre-declare avoided UTC
windows measured from sealed spread history; admission then skips those
windows entirely.  The mask must be declared before use and never widens
itself from live prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

MASK_POLICY = "PREDECLARED_UTC_HIGH_COST_WINDOW_MASK_V1"
# Default from long-observed FX microstructure; lanes may override with a
# sealed, spread-history-derived declaration.
DEFAULT_MASKED_WINDOWS: tuple[tuple[int, int], ...] = ((21 * 60, 22 * 60),)


@dataclass(frozen=True, slots=True)
class CostWindowDecision:
    admitted: bool
    reason: str
    minute_of_day_utc: int

    def payload(self) -> dict[str, Any]:
        return {
            "policy": MASK_POLICY,
            "admitted": self.admitted,
            "reason": self.reason,
            "minute_of_day_utc": self.minute_of_day_utc,
            "derived_from_live_prices": False,
        }


def _validated_windows(
    windows: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for window in windows:
        start, end = window
        for value in (start, end):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("mask minutes must be integers")
        if not 0 <= start < end <= 24 * 60:
            raise ValueError("mask window must be inside one UTC day and positive")
        result.append((start, end))
    return tuple(result)


def evaluate_cost_window(
    decision_utc: datetime,
    *,
    masked_windows: Sequence[tuple[int, int]] = DEFAULT_MASKED_WINDOWS,
) -> CostWindowDecision:
    """Admit a decision only outside the declared high-cost UTC windows."""

    if decision_utc.tzinfo is None:
        raise ValueError("decision clock must be timezone-aware")
    utc = decision_utc.astimezone(timezone.utc)
    minute = utc.hour * 60 + utc.minute
    for start, end in _validated_windows(masked_windows):
        if start <= minute < end:
            return CostWindowDecision(
                admitted=False,
                reason="INSIDE_DECLARED_HIGH_COST_WINDOW",
                minute_of_day_utc=minute,
            )
    return CostWindowDecision(
        admitted=True,
        reason="OUTSIDE_DECLARED_HIGH_COST_WINDOWS",
        minute_of_day_utc=minute,
    )
