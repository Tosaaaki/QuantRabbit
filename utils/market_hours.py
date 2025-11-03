"""Utilities for determining FX market open/close windows."""

from __future__ import annotations

import datetime as _dt

# OANDA weekend trading halt:
# Friday 21:55 UTC → Sunday 21:35 UTC (per ops spec)
_FRIDAY = 4
_SATURDAY = 5
_SUNDAY = 6
_CLOSE_START = _dt.time(hour=21, minute=55)
_REOPEN_TIME = _dt.time(hour=21, minute=35)


def is_market_open(now: _dt.datetime | None = None) -> bool:
    """Return True when the market is considered open for trading."""
    now = now or _dt.datetime.utcnow()
    if now.tzinfo is not None:
        now = now.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    weekday = now.weekday()
    current_time = now.time()

    if weekday == _FRIDAY and current_time >= _CLOSE_START:
        return False
    if weekday == _SATURDAY:
        return False
    if weekday == _SUNDAY and current_time < _REOPEN_TIME:
        return False
    return True


def seconds_until_open(now: _dt.datetime | None = None) -> float:
    """Return seconds until the next open window (0 when already open)."""
    now = now or _dt.datetime.utcnow()
    if now.tzinfo is not None:
        now = now.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    if is_market_open(now):
        return 0.0

    # compute next open
    weekday = now.weekday()
    current_time = now.time()

    if weekday == _FRIDAY and current_time >= _CLOSE_START:
        days_ahead = (_SUNDAY - weekday) % 7
        reopen_date = (now + _dt.timedelta(days=days_ahead)).date()
        reopen_dt = _dt.datetime.combine(reopen_date, _REOPEN_TIME)
        return (reopen_dt - now).total_seconds()

    if weekday == _SATURDAY:
        reopen_date = (now + _dt.timedelta(days=( _SUNDAY - weekday) % 7)).date()
        reopen_dt = _dt.datetime.combine(reopen_date, _REOPEN_TIME)
        return (reopen_dt - now).total_seconds()

    if weekday == _SUNDAY and current_time < _REOPEN_TIME:
        reopen_dt = _dt.datetime.combine(now.date(), _REOPEN_TIME)
        return (reopen_dt - now).total_seconds()

    # Default fallback (should not hit due to above checks)
    return 0.0


__all__ = ["is_market_open", "seconds_until_open"]
