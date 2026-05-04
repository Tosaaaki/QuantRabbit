"""Economic-calendar gating with tier S/A/B/C scoring.

The trader uses an event tier (and minutes-to-event) to decide whether the
next cycle should pause new entries, tighten exposure, or proceed normally.

Tier mapping (per ``docs/research/04-intermarket-macro.md`` §6):
  * **S** — top-tier macro shocks: NFP, FOMC (rate decision / minutes
    release), CPI (US), BOJ rate decision.
  * **A** — high-impact regional drivers: ECB / BoE rate decision, Core PCE,
    flash PMIs.
  * **B** — second-tier US macro: Retail Sales, ADP Employment, JOLTS.
  * **C** — third-tier macro: Trade Balance, Housing Starts, Consumer
    Confidence, Industrial Production.
  * **none** — anything not in the above buckets.

Contract:
  ``docs/AGENT_CONTRACT.md`` §3.5 — every numeric constant carries an
  (a)/(b)/(c) docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable


# ---------------------------------------------------------------------------
# §3.5 documented constants
# ---------------------------------------------------------------------------

# (a) Default "in-window" half-width in minutes — the trader treats an event
#     as imminent when it is within 30 minutes either side.
# (b) Constant because §6 of the research note picks a fixed pre-event
#     blackout that matches typical liquidity-thin behavior; tuning per
#     pair would invite ad-hoc gates (forbidden by §3.5).
# (c) Replace if the research note widens the blackout for tier-S events
#     specifically — implement as a tier-keyed dict, not as a new global.
_DEFAULT_WINDOW_MINUTES = 30

# (a) Event-name keywords mapped to tiers — case-insensitive substring match.
# (b) Constant because the tier mapping is a research-note artifact (§6),
#     not a runtime knob; unrecognised events fall through to "none".
# (c) Replace by editing the dict below; do not introduce per-trader
#     overrides — that would silently break consistency across cycles.
_TIER_KEYWORDS: tuple[tuple[str, str], ...] = (
    # S — top-tier macro shocks.
    ("FOMC", "S"),
    ("FED RATE", "S"),
    ("FEDERAL FUNDS", "S"),
    ("FOMC MINUTES", "S"),
    ("NFP", "S"),
    ("NON-FARM", "S"),
    ("NONFARM", "S"),
    ("US CPI", "S"),
    ("CPI Y/Y", "S"),
    ("BOJ", "S"),
    ("BANK OF JAPAN", "S"),
    # A — high-impact regional drivers.
    ("ECB", "A"),
    ("EUROPEAN CENTRAL BANK", "A"),
    ("BOE", "A"),
    ("BANK OF ENGLAND", "A"),
    ("CORE PCE", "A"),
    ("FLASH PMI", "A"),
    ("PMI FLASH", "A"),
    ("MANUFACTURING PMI FLASH", "A"),
    # B — second-tier US macro.
    ("RETAIL SALES", "B"),
    ("ADP", "B"),
    ("JOLTS", "B"),
    # C — third-tier macro.
    ("TRADE BALANCE", "C"),
    ("HOUSING STARTS", "C"),
    ("CONSUMER CONFIDENCE", "C"),
    ("INDUSTRIAL PRODUCTION", "C"),
)


@dataclass(frozen=True)
class CalendarEvent:
    """Single economic-calendar event."""

    name: str
    currency: str  # e.g. "USD", "JPY"
    scheduled_at_utc: datetime
    importance: str = ""  # raw provider importance label, optional

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "currency": self.currency,
            "scheduled_at_utc": self.scheduled_at_utc.isoformat(),
            "importance": self.importance,
        }


def event_tier(event: CalendarEvent) -> str:
    """Return the tier label for ``event`` ("S"/"A"/"B"/"C"/"none")."""

    name = (event.name or "").upper()
    for keyword, tier in _TIER_KEYWORDS:
        if keyword in name:
            return tier
    return "none"


@dataclass(frozen=True)
class PairWindow:
    """Window decision for a single pair against the next event.

    ``in_window`` is preserved for backwards compatibility with callers that
    only checked the boolean. ``tier`` and ``minutes_to_event`` are the new
    fields driving tier-aware behavior.
    """

    pair: str
    in_window: bool
    tier: str = "none"
    minutes_to_event: int | None = None
    event_name: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "in_window": self.in_window,
            "tier": self.tier,
            "minutes_to_event": self.minutes_to_event,
            "event_name": self.event_name,
        }


@dataclass(frozen=True)
class CalendarSnapshot:
    """Tier-aware calendar snapshot for the current cycle."""

    fetched_at_utc: datetime
    events: tuple[CalendarEvent, ...]
    pair_windows: tuple[PairWindow, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "fetched_at_utc": self.fetched_at_utc.isoformat(),
            "events": [e.to_dict() for e in self.events],
            "pair_windows": [w.to_dict() for w in self.pair_windows],
        }


def build_pair_windows(
    *,
    pairs: Iterable[str],
    events: Iterable[CalendarEvent],
    now: datetime,
    window_minutes: int = _DEFAULT_WINDOW_MINUTES,
) -> tuple[PairWindow, ...]:
    """Compute one ``PairWindow`` per requested pair.

    Per pair, pick the highest-tier event whose currency appears in the pair
    symbol AND that lies within ``window_minutes`` of ``now``.
    """

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    tier_rank = {"S": 4, "A": 3, "B": 2, "C": 1, "none": 0}

    out: list[PairWindow] = []
    for pair in pairs:
        symbol = pair.upper()
        best: tuple[int, CalendarEvent, str, int] | None = None
        for evt in events:
            cur = (evt.currency or "").upper()
            if cur and cur not in symbol:
                continue
            evt_at = evt.scheduled_at_utc
            if evt_at.tzinfo is None:
                evt_at = evt_at.replace(tzinfo=timezone.utc)
            delta_min = int((evt_at - now).total_seconds() // 60)
            if abs(delta_min) > window_minutes:
                continue
            tier = event_tier(evt)
            rank = tier_rank.get(tier, 0)
            if best is None or rank > best[0]:
                best = (rank, evt, tier, delta_min)
        if best is None:
            out.append(PairWindow(pair=pair, in_window=False))
        else:
            _, evt, tier, delta_min = best
            out.append(
                PairWindow(
                    pair=pair,
                    in_window=tier != "none",
                    tier=tier,
                    minutes_to_event=delta_min,
                    event_name=evt.name,
                )
            )
    return tuple(out)


__all__ = [
    "CalendarEvent",
    "CalendarSnapshot",
    "PairWindow",
    "event_tier",
    "build_pair_windows",
]
