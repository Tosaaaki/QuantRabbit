"""Economic calendar feed + event-window gating.

Pulls the public ForexFactory weekly XML feed
(https://nfs.faireconomy.media/ff_calendar_thisweek.xml) — a free,
publicly-mirrored source widely used by retail FX automation. We parse it,
keep only High/Medium impact events for currencies relevant to the trader,
and emit:

- A list of upcoming events with timestamp, currency, impact, title, forecast.
- A per-pair "in window" flag indicating whether either component currency is
  inside a configurable pre/post-event quiet zone.

If the feed cannot be reached we emit `MISSING_FOREX_FACTORY_FEED` issue —
the trader should treat the calendar as unavailable rather than as "all clear".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

DEFAULT_CURRENCIES: tuple[str, ...] = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")


@dataclass(frozen=True)
class CalendarEvent:
    timestamp_utc: str
    currency: str
    impact: str  # "High" / "Medium" / "Low" / "Holiday"
    title: str
    forecast: str | None
    previous: str | None
    actual: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "currency": self.currency,
            "impact": self.impact,
            "title": self.title,
            "forecast": self.forecast,
            "previous": self.previous,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class PairWindow:
    pair: str
    in_window: bool
    reason: str  # description of why
    next_event: CalendarEvent | None

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "in_window": self.in_window,
            "reason": self.reason,
            "next_event": self.next_event.to_dict() if self.next_event else None,
        }


@dataclass(frozen=True)
class CalendarSnapshot:
    generated_at_utc: str
    source_url: str
    events: tuple[CalendarEvent, ...]
    pair_windows: tuple[PairWindow, ...]
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "source_url": self.source_url,
            "events": [e.to_dict() for e in self.events],
            "pair_windows": [w.to_dict() for w in self.pair_windows],
            "issues": list(self.issues),
        }


def fetch_calendar_xml(url: str = FOREX_FACTORY_URL, *, timeout: int = 15) -> bytes:
    """Fetch the raw XML feed bytes."""
    req = Request(url, headers={"User-Agent": "QuantRabbit/1.0 (research)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_calendar_xml(payload: bytes) -> tuple[CalendarEvent, ...]:
    """Parse the ForexFactory weekly XML into `CalendarEvent` records."""
    root = ET.fromstring(payload)
    events: list[CalendarEvent] = []
    for item in root.findall(".//event"):
        title = (item.findtext("title") or "").strip()
        currency = (item.findtext("country") or "").strip()
        impact = (item.findtext("impact") or "").strip()
        date_text = (item.findtext("date") or "").strip()
        time_text = (item.findtext("time") or "").strip()
        forecast = (item.findtext("forecast") or "").strip() or None
        previous = (item.findtext("previous") or "").strip() or None
        actual = (item.findtext("actual") or "").strip() or None
        ts_utc = _parse_ff_datetime(date_text, time_text)
        if not ts_utc or not currency:
            continue
        events.append(CalendarEvent(
            timestamp_utc=ts_utc.isoformat(),
            currency=currency,
            impact=impact,
            title=title,
            forecast=forecast,
            previous=previous,
            actual=actual,
        ))
    events.sort(key=lambda e: e.timestamp_utc)
    return tuple(events)


def build_calendar_snapshot(
    *,
    pairs: Sequence[str],
    pre_minutes: int = 30,
    post_minutes: int = 30,
    impact_filter: tuple[str, ...] = ("High", "Medium"),
    url: str = FOREX_FACTORY_URL,
    now_utc: datetime | None = None,
    fetch: bool = True,
) -> CalendarSnapshot:
    """Pull the feed and compute per-pair event-window flags.

    `pairs` are FX pairs like 'USD_JPY' — we split into base/quote to look up
    relevant events. `pre_minutes` / `post_minutes` define the quiet zone
    around each High / Medium event.
    """

    issues: list[str] = []
    events: tuple[CalendarEvent, ...] = tuple()
    if fetch:
        try:
            xml_bytes = fetch_calendar_xml(url)
            events = parse_calendar_xml(xml_bytes)
        except (URLError, ET.ParseError, ValueError, OSError) as exc:
            issues.append(f"MISSING_FOREX_FACTORY_FEED: {exc}")

    now = now_utc or datetime.now(timezone.utc)
    pair_windows: list[PairWindow] = []
    calendar_unavailable = fetch and bool(issues) and not events
    for pair in pairs:
        if calendar_unavailable:
            pair_windows.append(PairWindow(
                pair=pair,
                in_window=True,
                reason=f"calendar unavailable: {issues[0]}",
                next_event=None,
            ))
            continue
        base, _, quote = pair.upper().partition("_")
        relevant = [
            e for e in events
            if (e.currency in (base, quote)) and (e.impact in impact_filter)
        ]
        in_window = False
        reason = "no relevant events"
        next_event: CalendarEvent | None = None
        for e in relevant:
            try:
                ts = datetime.fromisoformat(e.timestamp_utc)
            except ValueError:
                continue
            window_start = ts - timedelta(minutes=pre_minutes)
            window_end = ts + timedelta(minutes=post_minutes)
            if window_start <= now <= window_end:
                in_window = True
                reason = f"{e.currency} {e.impact} '{e.title}' at {e.timestamp_utc} (±{pre_minutes}m)"
                next_event = e
                break
            if ts > now and (next_event is None or ts < datetime.fromisoformat(next_event.timestamp_utc)):
                next_event = e
        if not in_window and next_event is not None:
            try:
                ts = datetime.fromisoformat(next_event.timestamp_utc)
                minutes_ahead = (ts - now).total_seconds() / 60.0
                reason = f"next event in {minutes_ahead:.0f}min: {next_event.currency} {next_event.impact} '{next_event.title}'"
            except ValueError:
                pass
        pair_windows.append(PairWindow(
            pair=pair, in_window=in_window, reason=reason, next_event=next_event,
        ))

    return CalendarSnapshot(
        generated_at_utc=now.isoformat(),
        source_url=url,
        events=events,
        pair_windows=tuple(pair_windows),
        issues=tuple(issues),
    )


def _parse_ff_datetime(date_text: str, time_text: str) -> datetime | None:
    """ForexFactory feed publishes dates like 'MM-DD-YYYY' and times like '8:30am'.

    The public `nfs.faireconomy.media` mirror publishes normalized UTC clock
    times (for example US NFP appears as 12:30pm, not 8:30am New York time).
    Keep the parsed clock as UTC so high-impact event windows line up with
    broker timestamps.
    """
    if not date_text or not time_text:
        return None
    if time_text.lower() in ("all day", "tentative"):
        return None
    # Date format from feed: MM-DD-YYYY
    try:
        m, d, y = date_text.split("-")
        date_part = datetime(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None
    # Time: '8:30am' / '11:00pm'
    t = time_text.lower().strip()
    pm = t.endswith("pm")
    am = t.endswith("am")
    if not (pm or am):
        return None
    body = t[:-2].strip()
    if ":" in body:
        try:
            hh, mm = body.split(":")
            hh_i = int(hh) % 12
            mm_i = int(mm)
        except (ValueError, TypeError):
            return None
    else:
        try:
            hh_i = int(body) % 12
            mm_i = 0
        except (ValueError, TypeError):
            return None
    if pm:
        hh_i += 12
    naive = date_part.replace(hour=hh_i, minute=mm_i)
    return naive.replace(tzinfo=timezone.utc)
