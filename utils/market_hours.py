"""Utilities for determining FX market open/close windows."""

from __future__ import annotations

import datetime as _dt
import os

# Weekend trading halt (JST基準: 土曜07:00〜月曜07:00)
# JST+9 に換算すると、UTCでは金曜22:00〜日曜22:00
_FRIDAY = 4
_SATURDAY = 5
_SUNDAY = 6
_CLOSE_START = _dt.time(hour=22, minute=0)  # Friday 22:00 UTC
_REOPEN_TIME = _dt.time(hour=22, minute=0)  # Sunday 22:00 UTC


def _parse_hhmm(value: str, default: _dt.time) -> _dt.time:
    raw = (value or "").strip()
    if not raw:
        return default
    parts = raw.split(":")
    if len(parts) < 2:
        return default
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) >= 3 else 0
    except Exception:
        return default
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return default
    return _dt.time(hour=hour, minute=minute, second=second)


_DAILY_HALT_ENABLED = os.getenv("DAILY_HALT_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
    "",
}
_DAILY_HALT_START = _parse_hhmm(
    os.getenv("DAILY_HALT_START_UTC", "21:58"), _dt.time(hour=21, minute=58)
)
_DAILY_HALT_END = _parse_hhmm(
    os.getenv("DAILY_HALT_END_UTC", "22:07"), _dt.time(hour=22, minute=7)
)


def _in_daily_halt(now: _dt.datetime) -> bool:
    if not _DAILY_HALT_ENABLED:
        return False
    if _DAILY_HALT_START == _DAILY_HALT_END:
        return False
    t = now.time()
    if _DAILY_HALT_START < _DAILY_HALT_END:
        return _DAILY_HALT_START <= t < _DAILY_HALT_END
    # Window crosses midnight
    return t >= _DAILY_HALT_START or t < _DAILY_HALT_END


def _daily_halt_end_dt(now: _dt.datetime) -> _dt.datetime:
    if _DAILY_HALT_START == _DAILY_HALT_END:
        return now
    if _DAILY_HALT_START < _DAILY_HALT_END:
        end_date = now.date()
        if now.time() >= _DAILY_HALT_END:
            end_date = end_date + _dt.timedelta(days=1)
        return _dt.datetime.combine(end_date, _DAILY_HALT_END)
    # Window crosses midnight
    if now.time() >= _DAILY_HALT_START:
        end_date = now.date() + _dt.timedelta(days=1)
    else:
        end_date = now.date()
    return _dt.datetime.combine(end_date, _DAILY_HALT_END)


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
    if _in_daily_halt(now):
        return False  # 日次メンテナンス窓
    return True  # それ以外の時間はオープン


def seconds_until_open(now: _dt.datetime | None = None) -> float:
    """Return seconds until the next open window (0 when already open)."""
    now = now or _dt.datetime.utcnow()
    if now.tzinfo is not None:
        now = now.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    if is_market_open(now):
        return 0.0

    # Weekend close has priority over daily halt
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

    if _in_daily_halt(now):
        end_dt = _daily_halt_end_dt(now)
        return max(0.0, (end_dt - now).total_seconds())

    # Default fallback (should not hit due to above checks)
    return 0.0


__all__ = ["is_market_open", "seconds_until_open"]
