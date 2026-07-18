"""Versioned OANDA FX session slots used by prospective DOJO sources.

OANDA publishes all non-TRY FX hours in New York time: Sunday 17:05 through
Friday 16:59, with a daily six-minute break from 16:59 through 17:04.  OANDA's
aligned candle response can label the close boundary and the bucket ending at
the reopen boundary.  The functions below encode that versioned candle-label
policy and let ``zoneinfo`` apply the New York DST boundary.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


OANDA_FX_HOURS_POLICY = (
    "OANDA_US_FX_SUN1705_FRI1659_ALIGNED_CANDLE_BOUNDARIES_NY_V2"
)
OANDA_FX_HOURS_SOURCE = "https://www.oanda.com/us-en/trading/hours-of-operation/"
_NY = ZoneInfo("America/New_York")


def oanda_fx_candle_open_is_expected(
    value: datetime, *, step: timedelta = timedelta(minutes=1)
) -> bool:
    """Return whether OANDA may label an aligned non-TRY FX candle here."""

    instant = _utc(value)
    if step <= timedelta(0) or step % timedelta(minutes=1):
        raise ValueError("OANDA candle step must be positive whole minutes")
    step_minutes = int(step / timedelta(minutes=1))
    local = instant.astimezone(_NY)
    weekday = local.weekday()  # Monday=0, Sunday=6.
    minute = local.hour * 60 + local.minute
    close_minute = 16 * 60 + 59
    reopen_minute = 17 * 60 + 5
    if weekday == 5:
        return False
    reopen_bucket_minute = reopen_minute - step_minutes
    if weekday == 6:
        return minute >= reopen_bucket_minute
    if weekday == 4:
        return minute <= close_minute
    return minute <= close_minute or minute >= reopen_bucket_minute


def expected_oanda_fx_slots(
    start: datetime,
    end: datetime,
    *,
    step: timedelta,
) -> list[datetime]:
    """Return expected candle opens in one aligned half-open UTC window."""

    cursor = _utc(start)
    stop = _utc(end)
    if step <= timedelta(0) or cursor >= stop:
        raise ValueError("OANDA slot window and step must be positive")
    step_seconds = step.total_seconds()
    if (
        not step_seconds.is_integer()
        or int(step_seconds) % 60
        or cursor.second
        or cursor.microsecond
        or int(cursor.timestamp()) % int(step_seconds)
    ):
        raise ValueError("OANDA slot window is not on the absolute candle grid")
    slots: list[datetime] = []
    while cursor < stop:
        if oanda_fx_candle_open_is_expected(cursor, step=step):
            slots.append(cursor)
        cursor += step
    if cursor != stop:
        raise ValueError("OANDA slot window is not aligned to the requested step")
    return slots


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("OANDA slot timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "OANDA_FX_HOURS_POLICY",
    "OANDA_FX_HOURS_SOURCE",
    "expected_oanda_fx_slots",
    "oanda_fx_candle_open_is_expected",
]
