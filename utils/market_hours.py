"""Utilities for determining FX market open/close windows."""

from __future__ import annotations

import datetime as _dt

# Weekend trading halt (JST基準: 土曜07:00〜月曜07:00)
# JST+9 に換算すると、UTCでは金曜22:00〜日曜22:00
_FRIDAY = 4
_SATURDAY = 5
_SUNDAY = 6
_CLOSE_START = _dt.time(hour=22, minute=0)  # Friday 22:00 UTC
_REOPEN_TIME = _dt.time(hour=22, minute=0)  # Sunday 22:00 UTC


def is_market_open(now: _dt.datetime | None = None) -> bool:
    """Return True when the market is considered open for trading."""
    now = now or _dt.datetime.utcnow()
    if now.tzinfo is not None:
        now = now.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    weekday = now.weekday()
    current_time = now.time()

    if weekday == _FRIDAY and current_time >= _CLOSE_START:
        return False  # 金曜22:00 UTC 以降はクローズ
    if weekday == _SATURDAY:
        return False  # 土曜は終日クローズ
    if weekday == _SUNDAY and current_time < _REOPEN_TIME:
        return False  # 日曜22:00 UTC までクローズ
    return True  # それ以外の時間はオープン


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
        # Friday after 22:00 UTC -> next open is Sunday 22:00 UTC
        days_ahead = (_SUNDAY - weekday) % 7
        reopen_date = (now + _dt.timedelta(days=days_ahead)).date()
        reopen_dt = _dt.datetime.combine(reopen_date, _REOPEN_TIME)
        return (reopen_dt - now).total_seconds()

    if weekday == _SATURDAY:
        # Saturday -> next open Sunday 22:00 UTC
        reopen_date = (now + _dt.timedelta(days=(_SUNDAY - weekday) % 7)).date()
        reopen_dt = _dt.datetime.combine(reopen_date, _REOPEN_TIME)
        return (reopen_dt - now).total_seconds()

    if weekday == _SUNDAY and current_time < _REOPEN_TIME:
        # Sunday before 22:00 UTC -> wait until 22:00 UTC
        reopen_dt = _dt.datetime.combine(now.date(), _REOPEN_TIME)
        return (reopen_dt - now).total_seconds()

    # Default fallback (should not hit due to above checks)
    return 0.0


__all__ = ["is_market_open", "seconds_until_open"]
