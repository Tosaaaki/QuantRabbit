#!/usr/bin/env python3
"""
Market State — Shared module for detecting market tradeability.

Used by profit_check.py, quality_audit.py, and other tools to determine
whether the market is open, closed, or in maintenance.

States:
    OPEN       — Normal trading hours. All actions allowed.
    ROLLOVER   — Daily OANDA maintenance (5 PM ET +/- 20 min). No orders.
    CLOSED     — Weekend (Fri 5PM ET -> Sun 5PM ET). No orders.

Design principle:
    Detection is TIME-BASED ONLY, never spread-based.
    Wide spreads during OPEN hours (news, intervention, BOJ) = trading opportunity.
    Blocking on spread would miss the best trades.
"""
from datetime import datetime, timezone


def _is_us_dst(dt: datetime) -> bool:
    """Check if US is in Daylight Saving Time (EDT) at the given UTC datetime."""
    year = dt.year
    # 2nd Sunday of March at 07:00 UTC (= 02:00 EST -> 03:00 EDT)
    mar1 = dt.replace(month=3, day=1, hour=0, minute=0, second=0, microsecond=0)
    days_to_sun = (6 - mar1.weekday()) % 7
    dst_start = mar1.replace(day=1 + days_to_sun + 7, hour=7)

    # 1st Sunday of November at 06:00 UTC (= 02:00 EDT -> 01:00 EST)
    nov1 = dt.replace(month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
    days_to_sun = (6 - nov1.weekday()) % 7
    dst_end = nov1.replace(day=1 + days_to_sun, hour=6)

    return dst_start <= dt < dst_end


def get_market_state(now=None) -> tuple:
    """Determine current market tradeability based on time only.

    Args:
        now: Override current time (for testing). Defaults to UTC now.

    Returns:
        (state, reason)
        state: "OPEN" | "ROLLOVER" | "CLOSED"
        reason: Human-readable explanation
    """
    if now is None:
        now = datetime.now(timezone.utc)

    weekday = now.weekday()  # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    rollover_hour = 21 if _is_us_dst(now) else 22  # 5 PM ET in UTC
    rollover_time = now.replace(hour=rollover_hour, minute=0, second=0, microsecond=0)
    delta_min = (rollover_time - now).total_seconds() / 60

    # --- CLOSED: Weekend ---
    # FX market: Fri 5 PM ET close -> Sun 5 PM ET open
    # Friday after rollover passed (>15 min past)
    if weekday == 4 and delta_min < -15:
        return "CLOSED", f"Weekend (market closed since Fri {rollover_hour}:00 UTC)"
    # All Saturday
    if weekday == 5:
        return "CLOSED", "Weekend (Saturday — market closed)"
    # Sunday before market open (5 PM ET)
    if weekday == 6:
        if delta_min > 15:
            h = int(delta_min // 60)
            m = int(delta_min % 60)
            return "CLOSED", f"Weekend (market opens Sun {rollover_hour}:00 UTC, in {h}h{m:02d}m)"
        # Sunday within rollover window = market opening
        # Fall through to rollover check below

    # --- ROLLOVER: Daily OANDA maintenance (5 PM ET +/- 20 min) ---
    if -15 <= delta_min <= 20:
        if delta_min > 0:
            return "ROLLOVER", f"OANDA maintenance in {int(delta_min)} min ({rollover_hour}:00 UTC). Spread spike imminent."
        else:
            return "ROLLOVER", f"OANDA maintenance window ({int(-delta_min)} min ago). Spread still normalizing."

    # --- OPEN ---
    return "OPEN", "Market open"


def is_tradeable(now=None) -> bool:
    """Quick check: can we execute orders right now?"""
    state, _ = get_market_state(now)
    return state == "OPEN"


if __name__ == "__main__":
    state, reason = get_market_state()
    print(f"Market state: {state}")
    print(f"Reason: {reason}")
    print(f"Tradeable: {is_tradeable()}")
