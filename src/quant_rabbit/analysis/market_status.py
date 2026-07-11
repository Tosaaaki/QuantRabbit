from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


FX_OPEN_WEEKDAY = 6  # Sunday
FX_CLOSE_WEEKDAY = 4  # Friday
FX_MARKET_TIMEZONE_NAME = "America/New_York"
FX_MARKET_TIMEZONE = ZoneInfo(FX_MARKET_TIMEZONE_NAME)
FX_WEEK_BOUNDARY_LOCAL_TIME = time(17, 0)

SESSION_WINDOWS_UTC: tuple[tuple[str, int, int], ...] = (
    ("Sydney", 22, 7),
    ("Tokyo", 0, 9),
    ("London", 7, 16),
    ("New_York", 12, 21),
)


@dataclass(frozen=True)
class MarketStatus:
    generated_at_utc: str
    weekday: str
    weekday_index: int
    is_fx_open: bool
    closed_reason: str | None
    active_sessions: tuple[str, ...]
    minutes_to_next_open: int | None
    minutes_to_next_close: int | None
    most_recent_open_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "evidence_ref": "market:status",
            "weekday": self.weekday,
            "weekday_index": self.weekday_index,
            "is_fx_open": self.is_fx_open,
            "closed_reason": self.closed_reason,
            "active_sessions": list(self.active_sessions),
            "minutes_to_next_open": self.minutes_to_next_open,
            "minutes_to_next_close": self.minutes_to_next_close,
            "most_recent_open_utc": self.most_recent_open_utc,
            "contract": {
                "calendar_source": "deterministic_fx_week",
                "market_timezone": FX_MARKET_TIMEZONE_NAME,
                "sunday_open_local": "17:00",
                "friday_close_local": "17:00",
                "utc_boundary_is_dst_aware": True,
                "live_permission": False,
                "must_not_override_broker_truth": True,
            },
        }


def compute_market_status(now_utc: datetime | None = None) -> MarketStatus:
    now = _as_utc(now_utc or datetime.now(timezone.utc))
    is_open = _is_fx_market_open(now)
    next_open = _next_market_open(now) if not is_open else None
    next_close = _next_market_close(now) if is_open else None
    most_recent_open = _most_recent_market_open(now)
    return MarketStatus(
        generated_at_utc=now.isoformat(),
        weekday=now.strftime("%A"),
        weekday_index=now.weekday(),
        is_fx_open=is_open,
        closed_reason=None if is_open else _closed_reason(now),
        active_sessions=_active_sessions(now) if is_open else (),
        minutes_to_next_open=_minutes_until(now, next_open),
        minutes_to_next_close=_minutes_until(now, next_close),
        most_recent_open_utc=most_recent_open.isoformat(),
    )


def write_snapshot(status: MarketStatus, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_report(status: MarketStatus, report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    data = status.to_dict()
    lines = [
        "# Market Status Report",
        "",
        f"- Generated at UTC: `{data['generated_at_utc']}`",
        f"- FX open: `{data['is_fx_open']}`",
        f"- Weekday: `{data['weekday']}` (`{data['weekday_index']}`)",
        f"- Closed reason: `{data['closed_reason'] or 'none'}`",
        f"- Active sessions: `{', '.join(data['active_sessions']) or 'none'}`",
        f"- Minutes to next open: `{data['minutes_to_next_open']}`",
        f"- Minutes to next close: `{data['minutes_to_next_close']}`",
        f"- Most recent weekly open UTC: `{data['most_recent_open_utc']}`",
        "",
        "## Contract",
        "",
        "- Evidence ref: `market:status`.",
        "- This packet is deterministic calendar evidence only; broker snapshot remains authoritative for prices, positions, and tradability.",
        "- The weekly boundary is Sunday/Friday 17:00 America/New_York and follows New York daylight-saving time.",
    ]
    report.write_text("\n".join(lines) + "\n")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_fx_market_open(now: datetime) -> bool:
    market_now = now.astimezone(FX_MARKET_TIMEZONE)
    weekday = market_now.weekday()
    clock = market_now.time()
    if weekday == FX_OPEN_WEEKDAY:
        return clock >= FX_WEEK_BOUNDARY_LOCAL_TIME
    if weekday == FX_CLOSE_WEEKDAY:
        return clock < FX_WEEK_BOUNDARY_LOCAL_TIME
    return weekday in {0, 1, 2, 3}


def _closed_reason(now: datetime) -> str:
    market_now = now.astimezone(FX_MARKET_TIMEZONE)
    weekday = market_now.weekday()
    clock = market_now.time()
    if weekday == FX_OPEN_WEEKDAY and clock < FX_WEEK_BOUNDARY_LOCAL_TIME:
        return "SUNDAY_PRE_OPEN"
    if weekday == 5:
        return "SATURDAY_CLOSED"
    if weekday == FX_CLOSE_WEEKDAY and clock >= FX_WEEK_BOUNDARY_LOCAL_TIME:
        return "FRIDAY_AFTER_CLOSE"
    return "WEEKEND_CLOSED"


def _active_sessions(now: datetime) -> tuple[str, ...]:
    hour = now.hour
    sessions: list[str] = []
    for name, start_hour, end_hour in SESSION_WINDOWS_UTC:
        if start_hour < end_hour:
            active = start_hour <= hour < end_hour
        else:
            active = hour >= start_hour or hour < end_hour
        if active:
            sessions.append(name)
    return tuple(sessions)


def _next_market_open(now: datetime) -> datetime:
    market_now = now.astimezone(FX_MARKET_TIMEZONE)
    for days_ahead in range(8):
        day = market_now.date() + timedelta(days=days_ahead)
        candidate = datetime.combine(
            day,
            FX_WEEK_BOUNDARY_LOCAL_TIME,
            tzinfo=FX_MARKET_TIMEZONE,
        )
        if candidate.weekday() == FX_OPEN_WEEKDAY and candidate > market_now:
            return candidate.astimezone(timezone.utc)
    raise RuntimeError("next market open not found")


def _most_recent_market_open(now: datetime) -> datetime:
    market_now = now.astimezone(FX_MARKET_TIMEZONE)
    for days_back in range(8):
        day = market_now.date() - timedelta(days=days_back)
        candidate = datetime.combine(
            day,
            FX_WEEK_BOUNDARY_LOCAL_TIME,
            tzinfo=FX_MARKET_TIMEZONE,
        )
        if candidate.weekday() == FX_OPEN_WEEKDAY and candidate <= market_now:
            return candidate.astimezone(timezone.utc)
    raise RuntimeError("most recent market open not found")


def _next_market_close(now: datetime) -> datetime:
    market_now = now.astimezone(FX_MARKET_TIMEZONE)
    for days_ahead in range(8):
        day = market_now.date() + timedelta(days=days_ahead)
        candidate = datetime.combine(
            day,
            FX_WEEK_BOUNDARY_LOCAL_TIME,
            tzinfo=FX_MARKET_TIMEZONE,
        )
        if candidate.weekday() == FX_CLOSE_WEEKDAY and candidate > market_now:
            return candidate.astimezone(timezone.utc)
    raise RuntimeError("next market close not found")


def _minutes_until(now: datetime, target: datetime | None) -> int | None:
    if target is None:
        return None
    return max(0, int((target - now).total_seconds() // 60))
