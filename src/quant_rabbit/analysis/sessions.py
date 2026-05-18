"""Session microstructure tagging (killzones, Judas Swing, Silver Bullet, JP holidays).

This module classifies a UTC bar timestamp into a session-microstructure tag
anchored to **NY local time** (DST-aware via ``zoneinfo``), and computes the
``SessionContext`` the trader needs to reason about ICT/SMC-style intraday
geometry: the NY Midnight Open (the day's reference price), the previous
Asian range (and whether its extreme has been swept during the Judas window),
the London and NY-AM ranges, plus the next upcoming killzone.

Sources:
- ``docs/research/01-smc-ict.md`` §2 (session microstructure).
- ``docs/research/04-intermarket-macro.md`` §7 (JST holidays).
- Contract: ``docs/AGENT_CONTRACT.md`` §3.5 — every numeric constant has
  an (a)/(b)/(c) docstring stating: (a) what market reality it represents,
  (b) why it is constant rather than market-derived, (c) what should
  replace it if it ever needs to be changed.

Stdlib-only. No I/O. ``tag_bar`` is pure; ``build_session_context`` walks
candles. JP holidays are encoded explicitly because the optional
``jpholiday`` package is not a hard dependency for this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

from quant_rabbit.analysis.candles import Candle


# --- Time-zone constants -----------------------------------------------------

# (a) NY local time governs every ICT/SMC killzone reference in the literature
#     (London Killzone, NY AM Killzone, Silver Bullet, NY PM, Judas Swing,
#     Asian Range start/end).
# (b) Constant because the IANA timezone database itself encodes DST
#     transitions; we just resolve through ``zoneinfo``. Hard-coding
#     ``UTC-5`` would silently break twice a year.
# (c) If exchange / venue conventions ever shift to an alternative anchor
#     (e.g. London local time), update both this constant and the killzone
#     bounds below in lock-step.
_NY_TZ = ZoneInfo("America/New_York")

# (a) JST anchors the JP holiday calendar (Children's Day, Golden Week
#     range, Obon, year-end break). All §7 JST holidays are evaluated as
#     calendar dates in ``Asia/Tokyo``.
# (b) Constant: JST has no DST, so ``UTC+9`` is stable, but we still go
#     through ``zoneinfo`` for symmetry with NY and to avoid a future
#     surprise if Japan ever changes the rule.
# (c) Replace only if Japan adopts DST or shifts the IANA name.
_JST_TZ = ZoneInfo("Asia/Tokyo")


# --- Session tag enumeration -------------------------------------------------


class SessionTag(Enum):
    """Microstructure tag for a single bar.

    The tags are mutually exclusive and exhaustive: every UTC timestamp
    maps to exactly one ``SessionTag`` after conversion to NY local time.
    """

    # Coarse archive/outcome bucket alias. Live bar tagging emits the
    # more explicit TOKYO_KILLZONE during the Asian range so downstream
    # session-aware risk code can distinguish Tokyo liquidity from generic
    # off-hours while legacy outcome marts can still normalize to ASIA.
    ASIA = "ASIA"
    TOKYO_KILLZONE = "TOKYO_KILLZONE"
    LONDON_KILLZONE = "LONDON_KILLZONE"
    NY_AM_KILLZONE = "NY_AM_KILLZONE"
    SILVER_BULLET = "SILVER_BULLET"
    NY_PM_KILLZONE = "NY_PM_KILLZONE"
    JUDAS_WINDOW = "JUDAS_WINDOW"
    OFF_HOURS = "OFF_HOURS"


# --- Killzone window definitions (NY local hour, 24h) ------------------------
#
# Each window is `[start_hour, end_hour)` in **NY local time**. Hours are
# stored as floats so half-hour boundaries (e.g. NY PM 13:30) can be expressed
# without lossy rounding.
#
# (a) These are the canonical ICT killzones referenced in
#     ``docs/research/01-smc-ict.md`` §2: where institutional order flow
#     concentrates intraday.
# (b) Constant because they are *human-conventional* market session anchors,
#     not market-derived statistics. They define the regime classification;
#     they are not optimisation knobs.
# (c) If the research file is updated with refined empirical bounds, change
#     these bounds and re-run ``tests/test_sessions.py`` to confirm the
#     classification still matches expected fixtures.
_LONDON_KILLZONE = (2.0, 5.0)        # 02:00–05:00 NY
_NY_AM_KILLZONE = (7.0, 10.0)        # 07:00–10:00 NY
_SILVER_BULLET = (10.0, 11.0)        # 10:00–11:00 NY
_NY_PM_KILLZONE = (13.5, 16.0)       # 13:30–16:00 NY
_JUDAS_WINDOW = (0.0, 5.0)           # 00:00–05:00 NY (overlaps London KZ; precedence below)

# Asian/Tokyo range: 19:00–24:00 NY of the previous calendar day, and 00:00 NY
# is the day boundary that resets the "True Day Open" reference price.
# (a) The Asian session in NY-local terms; per ICT literature the Asian
#     range is built between 19:00 NY (prior day) and 00:00 NY (current day).
#     In JST this is the morning Tokyo liquidity window in US DST months,
#     so the live tag is named TOKYO_KILLZONE while the analytics range
#     remains the Asian range.
# (b) Constant for the same reason as the killzones.
# (c) If the trader ever moves the day-boundary anchor (e.g. to London open),
#     update ``_NY_DAY_RESET_HOUR`` and ``_ASIAN_RANGE_NY_HOURS`` together.
_NY_DAY_RESET_HOUR = 0.0
_ASIAN_RANGE_NY_HOURS = (19.0, 24.0)  # prior calendar day 19:00 NY → 00:00 NY


# --- JP holiday calendar -----------------------------------------------------
#
# (a) JP-side risk events that historically thin liquidity on JPY pairs
#     and have produced losing trades when SLs were placed too tightly
#     (see ``feedback_no_tight_sl_thin_market.md``: 4/3 Children's-Day-week
#     -984 JPY loss).
# (b) Constant because the calendar is *legally fixed* by Japanese statute
#     for the current and following year. We encode 2026 + 2027 explicitly;
#     the operator must extend this table before the next year begins.
# (c) If ``jpholiday`` is later added as a dependency, replace the dict
#     lookup with ``jpholiday.is_holiday(d)`` and ``jpholiday.holiday_name(d)``.
#     Until then, extend ``_FIXED_JP_HOLIDAYS`` and ``_RANGE_JP_HOLIDAYS``.
_FIXED_JP_HOLIDAYS: dict[tuple[int, int, int], str] = {
    # 2026
    (2026, 1, 1): "New Year's Day",
    (2026, 1, 12): "Coming of Age Day",
    (2026, 2, 11): "National Foundation Day",
    (2026, 2, 23): "Emperor's Birthday",
    (2026, 3, 20): "Vernal Equinox Day",
    (2026, 4, 29): "Showa Day",
    (2026, 5, 3): "Constitution Memorial Day",
    (2026, 5, 4): "Greenery Day",
    (2026, 5, 5): "Children's Day",
    (2026, 5, 6): "Golden Week (substitute holiday)",
    (2026, 7, 20): "Marine Day",
    (2026, 8, 11): "Mountain Day",
    (2026, 9, 21): "Respect for the Aged Day",
    (2026, 9, 23): "Autumnal Equinox Day",
    (2026, 10, 12): "Health and Sports Day",
    (2026, 11, 3): "Culture Day",
    (2026, 11, 23): "Labor Thanksgiving Day",
    # 2027
    (2027, 1, 1): "New Year's Day",
    (2027, 1, 11): "Coming of Age Day",
    (2027, 2, 11): "National Foundation Day",
    (2027, 2, 23): "Emperor's Birthday",
    (2027, 3, 21): "Vernal Equinox Day",
    (2027, 3, 22): "Vernal Equinox Day (substitute)",
    (2027, 4, 29): "Showa Day",
    (2027, 5, 3): "Constitution Memorial Day",
    (2027, 5, 4): "Greenery Day",
    (2027, 5, 5): "Children's Day",
    (2027, 7, 19): "Marine Day",
    (2027, 8, 11): "Mountain Day",
    (2027, 9, 20): "Respect for the Aged Day",
    (2027, 9, 23): "Autumnal Equinox Day",
    (2027, 11, 3): "Culture Day",
    (2027, 11, 23): "Labor Thanksgiving Day",
}

# Soft-flag ranges: not statutory holidays but liquidity sinks. Each entry
# is ``((year, month_start, day_start), (year, month_end, day_end), label)``
# inclusive on both ends.
#
# (a) Golden Week extends de-facto through May 6 even on years where the
#     calendar formally ends on May 5; Obon is a customary corporate
#     vacation week (mid-August); Year-end / New-Year markets thin from
#     Dec 28 through Jan 3.
# (b) Constant because they are cultural / corporate conventions that do
#     not change year-to-year in a way the trader cares about.
# (c) Refine boundaries if a future market-story review shows liquidity
#     restored earlier or later.
_RANGE_JP_HOLIDAYS: tuple[tuple[tuple[int, int, int], tuple[int, int, int], str], ...] = (
    ((2026, 4, 29), (2026, 5, 6), "Golden Week"),
    ((2026, 8, 13), (2026, 8, 16), "Obon"),
    ((2026, 12, 28), (2027, 1, 3), "Year-end / New Year"),
    ((2027, 4, 29), (2027, 5, 6), "Golden Week"),
    ((2027, 8, 13), (2027, 8, 16), "Obon"),
    ((2027, 12, 28), (2028, 1, 3), "Year-end / New Year"),
)


def jp_holiday_calendar_for(year_min: int, year_max: int) -> Mapping[date, str]:
    """Return ``{date: label}`` for every JP holiday between ``year_min`` and ``year_max`` (inclusive).

    The result merges fixed-date statutory holidays and the soft-flag
    liquidity ranges (Golden Week, Obon, Year-end). When a date is in both
    a statutory holiday and a soft-flag range, the soft-flag label wins
    (so Golden Week dominates Children's Day for trader prose) — this
    matches how the operator narrates the period.

    (a) Encodes the JP holiday risk surface for ``§7`` of the macro research.
    (b) Constant per the calendar above.
    (c) Replace with ``jpholiday`` if/when the package is added.
    """
    if year_max < year_min:
        raise ValueError("year_max must be >= year_min")

    out: dict[date, str] = {}
    for (yyyy, mm, dd), label in _FIXED_JP_HOLIDAYS.items():
        if year_min <= yyyy <= year_max:
            out[date(yyyy, mm, dd)] = label
    for start, end, label in _RANGE_JP_HOLIDAYS:
        start_d = date(*start)
        end_d = date(*end)
        # Only include if either endpoint touches the requested year window.
        if end_d.year < year_min or start_d.year > year_max:
            continue
        cur = start_d
        while cur <= end_d:
            if year_min <= cur.year <= year_max:
                out[cur] = label  # soft-flag label wins
            cur += timedelta(days=1)
    return out


def _resolve_holiday(
    ts_utc: datetime,
    calendar: Mapping[date, str] | None,
) -> tuple[bool, str | None]:
    """Return ``(jp_holiday, label)`` for the JST calendar date of ``ts_utc``."""
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    jst_date = ts_utc.astimezone(_JST_TZ).date()
    cal = calendar or jp_holiday_calendar_for(jst_date.year, jst_date.year + 1)
    label = cal.get(jst_date)
    return (label is not None, label)


# --- Bar-level tagging -------------------------------------------------------


@dataclass(frozen=True)
class SessionMarker:
    """The session-microstructure classification of a single bar."""

    timestamp_utc: datetime
    ny_local_hour: float          # 0..24, fractional (e.g. 13.5 = 13:30 NY)
    tag: SessionTag
    jp_holiday: bool
    holiday_name: str | None      # e.g. "Children's Day", "Golden Week"


def _ny_local_hour(ts_utc: datetime) -> float:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ny = ts_utc.astimezone(_NY_TZ)
    return ny.hour + ny.minute / 60.0 + ny.second / 3600.0


def _classify_hour(hour: float) -> SessionTag:
    """Map an NY-local hour to a session tag.

    Precedence:
      1. SILVER_BULLET (10:00–11:00) wins over NY_AM_KILLZONE.
      2. LONDON_KILLZONE (02:00–05:00) wins over JUDAS_WINDOW
         (which spans 00:00–05:00 — Judas explicitly only fires before
         London opens).
      3. TOKYO_KILLZONE covers 19:00–24:00 (the Asian-range build).
      4. Everything else is OFF_HOURS.
    """
    if _SILVER_BULLET[0] <= hour < _SILVER_BULLET[1]:
        return SessionTag.SILVER_BULLET
    if _NY_AM_KILLZONE[0] <= hour < _NY_AM_KILLZONE[1]:
        return SessionTag.NY_AM_KILLZONE
    if _NY_PM_KILLZONE[0] <= hour < _NY_PM_KILLZONE[1]:
        return SessionTag.NY_PM_KILLZONE
    if _LONDON_KILLZONE[0] <= hour < _LONDON_KILLZONE[1]:
        return SessionTag.LONDON_KILLZONE
    if _JUDAS_WINDOW[0] <= hour < _JUDAS_WINDOW[1]:
        # Only the pre-London slice (00:00–02:00) reaches here, because
        # 02:00–05:00 was already claimed by LONDON_KILLZONE above.
        return SessionTag.JUDAS_WINDOW
    if _ASIAN_RANGE_NY_HOURS[0] <= hour < _ASIAN_RANGE_NY_HOURS[1]:
        return SessionTag.TOKYO_KILLZONE
    return SessionTag.OFF_HOURS


def tag_bar(
    timestamp_utc: datetime,
    *,
    jp_holiday_calendar: Mapping[date, str] | None = None,
) -> SessionMarker:
    """Classify a single UTC bar timestamp.

    Pure function. ``jp_holiday_calendar`` may be supplied for tests; when
    omitted the calendar for the bar's JST year +1 is built on demand.
    """
    if timestamp_utc.tzinfo is None:
        timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
    hour = _ny_local_hour(timestamp_utc)
    tag = _classify_hour(hour)
    is_holiday, label = _resolve_holiday(timestamp_utc, jp_holiday_calendar)
    return SessionMarker(
        timestamp_utc=timestamp_utc,
        ny_local_hour=hour,
        tag=tag,
        jp_holiday=is_holiday,
        holiday_name=label,
    )


# --- Session context ---------------------------------------------------------


@dataclass(frozen=True)
class SessionContext:
    """Aggregated session microstructure for the *current* moment.

    All ranges are ``(high, low)`` tuples. ``judas_armed`` is ``True`` when
    the prior Asian-range high or low has been wicked-through during the
    current day's Judas window — the ICT precondition for a London-session
    reversal play.
    """

    current: SessionMarker
    ny_midnight_open_utc: datetime
    ny_midnight_open_price: float | None
    asian_range: tuple[float, float] | None
    london_range: tuple[float, float] | None
    ny_am_range: tuple[float, float] | None
    judas_armed: bool
    next_killzone: SessionTag | None
    minutes_to_next_killzone: int | None


def _ny_midnight_utc_for(ts_utc: datetime) -> datetime:
    """Return the UTC instant of 00:00 NY of the trading day containing ``ts_utc``.

    The "trading day" is anchored at NY local 00:00. If the bar itself is
    pre-midnight NY (e.g. 22:00 NY), it still belongs to the *next*-NY-day's
    reference open; we resolve that by taking the NY-local date of the bar
    and constructing 00:00 NY of that date.
    """
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    ny = ts_utc.astimezone(_NY_TZ)
    midnight_ny = ny.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight_ny.astimezone(timezone.utc)


def ny_midnight_open(
    candles_m5: Sequence[Candle],
    *,
    now_utc: datetime | None = None,
) -> tuple[datetime, float | None]:
    """Return ``(ny_midnight_utc, price_or_none)`` for the current trading day.

    Selection rule: closest M5 candle whose timestamp is at-or-after
    00:00 NY of the trading day containing ``now_utc`` (or the last
    candle's timestamp). If no candle reaches 00:00 NY yet, returns
    ``price=None`` and the resolved UTC anchor.
    """
    if not candles_m5:
        anchor = _ny_midnight_utc_for(now_utc or datetime.now(timezone.utc))
        return anchor, None
    reference = now_utc or candles_m5[-1].timestamp_utc
    anchor = _ny_midnight_utc_for(reference)
    # First candle at-or-after the anchor.
    candidate: Candle | None = None
    for c in candles_m5:
        if c.timestamp_utc >= anchor:
            candidate = c
            break
    if candidate is None:
        return anchor, None
    return anchor, candidate.close


def _range_in_window(
    candles: Sequence[Candle],
    *,
    start_utc: datetime,
    end_utc: datetime,
) -> tuple[float, float] | None:
    """Return ``(high, low)`` over candles whose timestamp is in ``[start, end)``."""
    highs: list[float] = []
    lows: list[float] = []
    for c in candles:
        ts = c.timestamp_utc
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if start_utc <= ts < end_utc:
            highs.append(c.high)
            lows.append(c.low)
    if not highs:
        return None
    return (max(highs), min(lows))


def _prev_asian_window_utc(now_utc: datetime) -> tuple[datetime, datetime]:
    """The 19:00–24:00 NY window of the *previous* NY date (relative to ``now_utc``).

    Because the Asian range is built before the day's NY-midnight reset, we
    look at the NY-date of ``now_utc - 1 day`` and take 19:00 NY → 00:00 NY
    of the next NY-date as the window. Both endpoints converted to UTC.
    """
    ny = now_utc.astimezone(_NY_TZ)
    prev_ny_date = (ny - timedelta(days=1)).date()
    start_ny = datetime.combine(prev_ny_date, datetime.min.time(), tzinfo=_NY_TZ).replace(hour=19)
    end_ny = start_ny + timedelta(hours=5)  # 19:00 → 00:00 next day
    return start_ny.astimezone(timezone.utc), end_ny.astimezone(timezone.utc)


def _killzone_window_today_utc(
    now_utc: datetime,
    bounds: tuple[float, float],
) -> tuple[datetime, datetime]:
    """``[start_utc, end_utc)`` for a killzone with NY-local bounds, on today's NY date."""
    ny = now_utc.astimezone(_NY_TZ)
    today = ny.date()
    start_h, start_min = _split_hour(bounds[0])
    end_h, end_min = _split_hour(bounds[1])
    start_ny = datetime.combine(today, datetime.min.time(), tzinfo=_NY_TZ).replace(hour=start_h, minute=start_min)
    if end_h >= 24:
        end_ny = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=_NY_TZ)
    else:
        end_ny = datetime.combine(today, datetime.min.time(), tzinfo=_NY_TZ).replace(hour=end_h, minute=end_min)
    return start_ny.astimezone(timezone.utc), end_ny.astimezone(timezone.utc)


def _split_hour(h: float) -> tuple[int, int]:
    hour = int(h)
    minute = int(round((h - hour) * 60))
    if minute == 60:
        hour += 1
        minute = 0
    return hour, minute


def _next_killzone(now_utc: datetime) -> tuple[SessionTag | None, int | None]:
    """Return the next upcoming killzone tag and minutes-until from ``now_utc``."""
    candidates: list[tuple[datetime, SessionTag]] = []
    for bounds, tag in (
        (_ASIAN_RANGE_NY_HOURS, SessionTag.TOKYO_KILLZONE),
        (_LONDON_KILLZONE, SessionTag.LONDON_KILLZONE),
        (_NY_AM_KILLZONE, SessionTag.NY_AM_KILLZONE),
        (_SILVER_BULLET, SessionTag.SILVER_BULLET),
        (_NY_PM_KILLZONE, SessionTag.NY_PM_KILLZONE),
    ):
        start_today, _end_today = _killzone_window_today_utc(now_utc, bounds)
        if start_today > now_utc:
            candidates.append((start_today, tag))
        else:
            # Roll to tomorrow.
            start_tomorrow = start_today + timedelta(days=1)
            candidates.append((start_tomorrow, tag))
    if not candidates:
        return None, None
    candidates.sort()
    next_start, tag = candidates[0]
    delta = next_start - now_utc
    minutes = int(delta.total_seconds() // 60)
    return tag, minutes


def build_session_context(
    candles: Sequence[Candle],
    *,
    now_utc: datetime | None = None,
    jp_holiday_calendar: Mapping[date, str] | None = None,
) -> SessionContext:
    """Build the full session microstructure context.

    ``candles`` should be a chronologically-sorted M5 (or finer) sequence
    spanning at least the previous Asian session through the current bar.
    ``now_utc`` defaults to the last candle timestamp (preferred for
    deterministic tests); pass an explicit value to align the context with
    a wall-clock different from the latest candle.
    """
    if candles:
        reference = now_utc or candles[-1].timestamp_utc
    else:
        reference = now_utc or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    current_marker = tag_bar(reference, jp_holiday_calendar=jp_holiday_calendar)

    midnight_utc, midnight_price = ny_midnight_open(candles, now_utc=reference)

    # Asian range = previous NY day's 19:00 → today's 00:00 NY.
    asian_start, asian_end = _prev_asian_window_utc(reference)
    asian_range = _range_in_window(candles, start_utc=asian_start, end_utc=asian_end)

    # London Killzone window today.
    london_start, london_end = _killzone_window_today_utc(reference, _LONDON_KILLZONE)
    london_range = _range_in_window(
        candles,
        start_utc=london_start,
        end_utc=min(london_end, reference + timedelta(microseconds=1)),
    )
    if london_range is not None and london_start > reference:
        london_range = None

    # NY AM Killzone window today.
    am_start, am_end = _killzone_window_today_utc(reference, _NY_AM_KILLZONE)
    ny_am_range = _range_in_window(
        candles,
        start_utc=am_start,
        end_utc=min(am_end, reference + timedelta(microseconds=1)),
    )
    if ny_am_range is not None and am_start > reference:
        ny_am_range = None

    # Judas: did price wick through the Asian-range extreme during 00:00–05:00 NY?
    judas_armed = False
    if asian_range is not None:
        asian_high, asian_low = asian_range
        judas_start, judas_end = _killzone_window_today_utc(reference, _JUDAS_WINDOW)
        # Clip end at "now".
        judas_end_eff = min(judas_end, reference + timedelta(microseconds=1))
        for c in candles:
            ts = c.timestamp_utc if c.timestamp_utc.tzinfo else c.timestamp_utc.replace(tzinfo=timezone.utc)
            if judas_start <= ts < judas_end_eff:
                if c.high > asian_high or c.low < asian_low:
                    judas_armed = True
                    break

    next_kz, minutes_to_next = _next_killzone(reference)

    return SessionContext(
        current=current_marker,
        ny_midnight_open_utc=midnight_utc,
        ny_midnight_open_price=midnight_price,
        asian_range=asian_range,
        london_range=london_range,
        ny_am_range=ny_am_range,
        judas_armed=judas_armed,
        next_killzone=next_kz,
        minutes_to_next_killzone=minutes_to_next,
    )


__all__ = [
    "SessionTag",
    "SessionMarker",
    "SessionContext",
    "tag_bar",
    "build_session_context",
    "jp_holiday_calendar_for",
    "ny_midnight_open",
]
