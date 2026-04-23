"""Shared history scoping for the live discretionary trader runtime."""

from __future__ import annotations

from datetime import date, timedelta

LIVE_DISCRETIONARY_RESET = date(2026, 4, 17)
LIVE_DISCRETIONARY_RESET_STR = LIVE_DISCRETIONARY_RESET.isoformat()


def live_history_start(lookback_days: int | None, *, today: date | None = None) -> str:
    """Floor runtime learning windows at the discretionary-only reset date."""
    today = today or date.today()
    if lookback_days is None:
        return LIVE_DISCRETIONARY_RESET_STR
    return max(today - timedelta(days=lookback_days), LIVE_DISCRETIONARY_RESET).isoformat()


def live_history_scope_label(lookback_days: int | None, *, today: date | None = None) -> str:
    start = live_history_start(lookback_days, today=today)
    if lookback_days is None:
        return f"since {start} discretionary reset"
    if start == LIVE_DISCRETIONARY_RESET_STR:
        return f"{lookback_days}d window, floored at {LIVE_DISCRETIONARY_RESET_STR} reset"
    return f"{lookback_days}d window"
