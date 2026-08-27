"""Europe/London session chronology for completed M5 paper data.

UTC is the only accepted timestamp input.  Local wall-clock time is derived
with ``zoneinfo`` so GMT/BST changes cannot silently shift a named London
event.  This module supplies chronology only and creates no trading edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from run_causal_min_spread_representative_v27 import parse_utc_nanoseconds


LONDON = ZoneInfo("Europe/London")
M5_SECONDS = 300


class ChronologyError(RuntimeError):
    pass


@dataclass(frozen=True)
class LondonInstant:
    utc_time: str
    epoch_nanoseconds: int
    local_date: str
    local_clock: str
    utc_offset_seconds: int
    timezone_name: str
    fold: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionSpec:
    name: str
    rail_start_local_minute: int
    rail_end_local_minute: int
    decision_local_minute: int
    bar_seconds: int = M5_SECONDS

    def __post_init__(self) -> None:
        if not self.name:
            raise ChronologyError("session name is required")
        values = (
            self.rail_start_local_minute,
            self.rail_end_local_minute,
            self.decision_local_minute,
        )
        if any(not isinstance(value, int) or not 0 <= value < 24 * 60 for value in values):
            raise ChronologyError("session minutes must be integer local wall-clock minutes")
        if self.bar_seconds != M5_SECONDS:
            raise ChronologyError("this contract accepts completed M5 bars only")
        if self.rail_start_local_minute > self.rail_end_local_minute:
            raise ChronologyError("rail must be forward ordered")
        if self.rail_end_local_minute + self.bar_seconds // 60 > self.decision_local_minute:
            raise ChronologyError("rail must end no later than t-1 before decision t")
        if any(value % (self.bar_seconds // 60) for value in values):
            raise ChronologyError("session wall-clock minutes must align to M5")


def _datetime_from_epoch_ns(value: int) -> datetime:
    seconds, _nanoseconds = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


def utc_to_london(value: str) -> LondonInstant:
    """Convert canonical UTC to London; local/offset timestamp inputs are rejected."""
    try:
        parsed = parse_utc_nanoseconds(value)
    except ValueError as error:
        raise ChronologyError(f"timestamp must be canonical UTC Z input: {value}") from error
    local = _datetime_from_epoch_ns(parsed.value).astimezone(LONDON)
    offset = local.utcoffset()
    if offset is None:
        raise ChronologyError("Europe/London offset unavailable")
    return LondonInstant(
        utc_time=value,
        epoch_nanoseconds=parsed.value,
        local_date=local.date().isoformat(),
        local_clock=local.strftime("%H:%M:%S"),
        utc_offset_seconds=int(offset.total_seconds()),
        timezone_name=local.tzname() or "",
        fold=int(local.fold),
    )


def _bar_time(bar: Any) -> str:
    if isinstance(bar, dict):
        value = bar.get("time")
    else:
        value = getattr(bar, "time", None)
    if not isinstance(value, str):
        raise ChronologyError("bar lacks a canonical UTC time")
    return value


def _local_minute(instant: LondonInstant) -> int:
    hour, minute, second = (int(value) for value in instant.local_clock.split(":"))
    if second != 0:
        raise ChronologyError("M5 bar timestamp has nonzero local seconds")
    return hour * 60 + minute


def resolve_completed_m5_session(
    bars: Sequence[Any],
    local_date: str,
    spec: SessionSpec,
) -> dict[str, Any]:
    """Resolve rail<=t-1, decision t close, and the next M5 executable open."""
    try:
        datetime.strptime(local_date, "%Y-%m-%d")
    except ValueError as error:
        raise ChronologyError(f"invalid local session date: {local_date}") from error
    entries: list[tuple[int, LondonInstant, str]] = []
    seen_utc: set[int] = set()
    previous_ns: int | None = None
    for bar in bars:
        stamp = _bar_time(bar)
        instant = utc_to_london(stamp)
        if instant.epoch_nanoseconds in seen_utc:
            raise ChronologyError(f"duplicate UTC bar: {stamp}")
        if previous_ns is not None and instant.epoch_nanoseconds <= previous_ns:
            raise ChronologyError("input bars are not strictly increasing in UTC")
        previous_ns = instant.epoch_nanoseconds
        seen_utc.add(instant.epoch_nanoseconds)
        if instant.local_date == local_date:
            entries.append((_local_minute(instant), instant, stamp))

    fill_minute = spec.decision_local_minute + spec.bar_seconds // 60
    if fill_minute >= 24 * 60:
        raise ChronologyError("session fill crosses local date boundary")
    expected_rail = list(range(
        spec.rail_start_local_minute,
        spec.rail_end_local_minute + 1,
        spec.bar_seconds // 60,
    ))
    expected = expected_rail + [spec.decision_local_minute, fill_minute]
    relevant = [item for item in entries if item[0] in set(expected)]
    by_minute: dict[int, tuple[LondonInstant, str]] = {}
    for minute, instant, stamp in relevant:
        if minute in by_minute:
            raise ChronologyError(
                f"ambiguous duplicated London wall-clock bar at minute {minute}"
            )
        by_minute[minute] = (instant, stamp)
    missing = [minute for minute in expected if minute not in by_minute]
    if missing:
        raise ChronologyError(f"missing completed London-session M5 bars: {missing}")

    ordered = [by_minute[minute][0].epoch_nanoseconds for minute in expected]
    expected_delta = spec.bar_seconds * 1_000_000_000
    if any(right - left != expected_delta for left, right in zip(ordered, ordered[1:])):
        raise ChronologyError("London-session bars are not a continuous causal M5 chain")
    rail_times = [by_minute[minute][1] for minute in expected_rail]
    decision_time = by_minute[spec.decision_local_minute][1]
    fill_time = by_minute[fill_minute][1]
    decision_available_ns = parse_utc_nanoseconds(decision_time).value + expected_delta
    if decision_available_ns != parse_utc_nanoseconds(fill_time).value:
        raise ChronologyError("decision close is not available exactly at next M5 fill open")
    if any(parse_utc_nanoseconds(value).value >= parse_utc_nanoseconds(decision_time).value
           for value in rail_times):
        raise ChronologyError("rail contains decision or future bars")
    return {
        "session": spec.name,
        "local_date": local_date,
        "rail_bar_times_utc": rail_times,
        "rail_last_local_minute": spec.rail_end_local_minute,
        "decision_bar_time_utc": decision_time,
        "decision_local_minute": spec.decision_local_minute,
        "decision_close_available_at_utc": fill_time,
        "fill_next_m5_open_time_utc": fill_time,
        "utc_offset_seconds": by_minute[spec.decision_local_minute][0].utc_offset_seconds,
        "timezone_name": by_minute[spec.decision_local_minute][0].timezone_name,
        "completed_data_only": True,
        "rail_latest_bar_relation": "RAIL_LE_T_MINUS_1",
        "decision_relation": "DECISION_T_COMPLETED_CLOSE",
        "fill_relation": "FILL_T_PLUS_1_EXECUTABLE_OPEN",
        "fixed_utc_hour_used_as_edge_definition": False,
    }


def utc_string(epoch_seconds: int) -> str:
    """Fixture helper returning canonical zero-fraction UTC."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_epoch_seconds(value: str) -> int:
    parsed = parse_utc_nanoseconds(value).value
    if parsed % 1_000_000_000:
        raise ChronologyError("fixture grid helper requires whole seconds")
    return parsed // 1_000_000_000


def consecutive_utc_m5(start: str, count: int) -> list[dict[str, str]]:
    if count <= 0:
        raise ChronologyError("fixture count must be positive")
    start_seconds = utc_epoch_seconds(start)
    return [
        {"time": utc_string(start_seconds + index * M5_SECONDS)}
        for index in range(count)
    ]
