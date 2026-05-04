"""Pivot levels, session ranges, and round-number magnets.

For each pair we compute:
- Standard / Camarilla / Fibonacci / DeMark daily pivots from the prior day
  OHLC.
- Prior day high/low/close (PDH/PDL/PDC) and weekly/monthly opens.
- Asian / London / NY session high/low for the current day.
- Nearest round-number levels (e.g. USD/JPY 150.00, EUR/USD 1.1700).

Sessions (UTC):
- Tokyo:   00:00–08:00
- London:  07:00–16:00
- NY:      12:00–21:00
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Iterable, Sequence

from quant_rabbit.analysis.candles import Candle, fetch_candles_via_client
from quant_rabbit.broker.oanda import OandaReadOnlyClient


@dataclass(frozen=True)
class PivotSet:
    style: str  # "STANDARD" / "CAMARILLA" / "FIBONACCI" / "DEMARK"
    pp: float | None
    r1: float | None
    r2: float | None
    r3: float | None
    r4: float | None
    s1: float | None
    s2: float | None
    s3: float | None
    s4: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "style": self.style,
            "pp": self.pp, "r1": self.r1, "r2": self.r2, "r3": self.r3, "r4": self.r4,
            "s1": self.s1, "s2": self.s2, "s3": self.s3, "s4": self.s4,
        }


@dataclass(frozen=True)
class SessionRange:
    name: str  # "ASIA" / "LONDON" / "NY"
    high: float | None
    low: float | None
    range_pips: float | None

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "high": self.high, "low": self.low, "range_pips": self.range_pips}


@dataclass(frozen=True)
class RoundNumber:
    price: float
    distance_pips: float

    def to_dict(self) -> dict[str, object]:
        return {"price": self.price, "distance_pips": self.distance_pips}


@dataclass(frozen=True)
class LevelsReading:
    pair: str
    pdh: float | None
    pdl: float | None
    pdc: float | None
    pdo: float | None  # prior day open
    daily_open: float | None
    weekly_open: float | None
    monthly_open: float | None
    pivots: tuple[PivotSet, ...]
    sessions: tuple[SessionRange, ...]
    round_numbers: tuple[RoundNumber, ...]
    last_close: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "pair": self.pair,
            "pdh": self.pdh, "pdl": self.pdl, "pdc": self.pdc, "pdo": self.pdo,
            "daily_open": self.daily_open,
            "weekly_open": self.weekly_open,
            "monthly_open": self.monthly_open,
            "pivots": [p.to_dict() for p in self.pivots],
            "sessions": [s.to_dict() for s in self.sessions],
            "round_numbers": [r.to_dict() for r in self.round_numbers],
            "last_close": self.last_close,
        }


@dataclass(frozen=True)
class LevelsSnapshot:
    generated_at_utc: str
    pairs: tuple[LevelsReading, ...] = field(default_factory=tuple)
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "pairs": [p.to_dict() for p in self.pairs],
            "issues": list(self.issues),
        }


def build_levels_snapshot(
    *,
    client: OandaReadOnlyClient,
    pairs: Sequence[str],
) -> LevelsSnapshot:
    issues: list[str] = []
    out: list[LevelsReading] = []
    for pair in pairs:
        try:
            reading = build_levels_reading(client, pair)
            out.append(reading)
        except Exception as exc:
            issues.append(f"MISSING_LEVELS_{pair}: {exc}")
    return LevelsSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        pairs=tuple(out),
        issues=tuple(issues),
    )


def build_levels_reading(client: OandaReadOnlyClient, pair: str) -> LevelsReading:
    pip = 0.01 if pair.upper().endswith("_JPY") else 0.0001

    # Daily candles for PDH/PDL/PDC + pivots
    daily = fetch_candles_via_client(client, pair, "D", count=60)
    if len(daily) < 2:
        raise RuntimeError("not enough daily candles")
    prior = daily[-2]
    today = daily[-1]
    pdh, pdl, pdc, pdo = prior.high, prior.low, prior.close, prior.open
    daily_open = today.open

    pivots = (
        _pivots_standard(pdh, pdl, pdc),
        _pivots_camarilla(pdh, pdl, pdc),
        _pivots_fibonacci(pdh, pdl, pdc),
        _pivots_demark(pdh, pdl, pdc, pdo, today.open),
    )

    # Weekly/monthly opens
    weekly = fetch_candles_via_client(client, pair, "D", count=10)  # need ~7 daily candles
    weekly_open = _resolve_weekly_open(weekly)
    monthly = fetch_candles_via_client(client, pair, "D", count=40)
    monthly_open = _resolve_monthly_open(monthly)

    # Session ranges from M15 candles for today (UTC)
    m15 = fetch_candles_via_client(client, pair, "M15", count=200)
    sessions = _session_ranges(m15, pip)

    # Round numbers near current price
    last_close = m15[-1].close if m15 else daily_open
    round_levels = _round_numbers_near(last_close, pip)

    return LevelsReading(
        pair=pair,
        pdh=pdh, pdl=pdl, pdc=pdc, pdo=pdo,
        daily_open=daily_open,
        weekly_open=weekly_open,
        monthly_open=monthly_open,
        pivots=pivots,
        sessions=sessions,
        round_numbers=round_levels,
        last_close=last_close,
    )


# ---------------------------------------------------------------------------
# Pivot calculations
# ---------------------------------------------------------------------------


def _pivots_standard(h: float, l: float, c: float) -> PivotSet:
    pp = (h + l + c) / 3.0
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    r3 = h + 2 * (pp - l)
    s3 = l - 2 * (h - pp)
    return PivotSet(style="STANDARD", pp=pp, r1=r1, r2=r2, r3=r3, r4=None, s1=s1, s2=s2, s3=s3, s4=None)


def _pivots_camarilla(h: float, l: float, c: float) -> PivotSet:
    rng = h - l
    return PivotSet(
        style="CAMARILLA",
        pp=(h + l + c) / 3.0,
        r1=c + rng * 1.1 / 12,
        r2=c + rng * 1.1 / 6,
        r3=c + rng * 1.1 / 4,
        r4=c + rng * 1.1 / 2,
        s1=c - rng * 1.1 / 12,
        s2=c - rng * 1.1 / 6,
        s3=c - rng * 1.1 / 4,
        s4=c - rng * 1.1 / 2,
    )


def _pivots_fibonacci(h: float, l: float, c: float) -> PivotSet:
    pp = (h + l + c) / 3.0
    rng = h - l
    return PivotSet(
        style="FIBONACCI", pp=pp,
        r1=pp + 0.382 * rng, r2=pp + 0.618 * rng, r3=pp + rng, r4=None,
        s1=pp - 0.382 * rng, s2=pp - 0.618 * rng, s3=pp - rng, s4=None,
    )


def _pivots_demark(h: float, l: float, c: float, o: float | None, today_open: float | None) -> PivotSet:
    """DeMark uses prior open vs close to choose X."""
    if o is None or today_open is None:
        x = h + l + 2 * c
    elif c < o:
        x = h + 2 * l + c
    elif c > o:
        x = 2 * h + l + c
    else:
        x = h + l + 2 * c
    pp = x / 4.0
    r1 = x / 2.0 - l
    s1 = x / 2.0 - h
    return PivotSet(style="DEMARK", pp=pp, r1=r1, r2=None, r3=None, r4=None, s1=s1, s2=None, s3=None, s4=None)


# ---------------------------------------------------------------------------
# Sessions / weekly opens / round numbers
# ---------------------------------------------------------------------------


def _resolve_weekly_open(daily: Sequence[Candle]) -> float | None:
    if not daily:
        return None
    today = daily[-1]
    target_weekday = 0  # Monday
    base = today.timestamp_utc
    monday = base - timedelta(days=base.weekday())
    for c in daily:
        if c.timestamp_utc.date() == monday.date():
            return c.open
    return None


def _resolve_monthly_open(daily: Sequence[Candle]) -> float | None:
    if not daily:
        return None
    today = daily[-1]
    first_of_month = today.timestamp_utc.replace(day=1)
    for c in daily:
        if c.timestamp_utc.date() == first_of_month.date():
            return c.open
    return None


def _session_ranges(m15: Sequence[Candle], pip: float) -> tuple[SessionRange, ...]:
    if not m15:
        return tuple()
    today = m15[-1].timestamp_utc.date()
    # Filter today's candles
    today_candles = [c for c in m15 if c.timestamp_utc.date() == today]
    sessions = (
        ("ASIA", 0, 8),
        ("LONDON", 7, 16),
        ("NY", 12, 21),
    )
    out: list[SessionRange] = []
    for name, start_h, end_h in sessions:
        bucket = [c for c in today_candles if start_h <= c.timestamp_utc.hour < end_h]
        if not bucket:
            out.append(SessionRange(name=name, high=None, low=None, range_pips=None))
            continue
        hi = max(c.high for c in bucket)
        lo = min(c.low for c in bucket)
        out.append(SessionRange(name=name, high=hi, low=lo, range_pips=(hi - lo) / pip))
    return tuple(out)


def _round_numbers_near(price: float | None, pip: float, *, count: int = 4) -> tuple[RoundNumber, ...]:
    if price is None:
        return tuple()
    # Round-number step depends on the pair: JPY pairs at 0.50 / 1.00, others at 0.0050 / 0.0100
    step = 0.50 if pip == 0.01 else 0.0050
    base = round(price / step) * step
    levels: list[float] = []
    for k in range(-count, count + 1):
        levels.append(base + k * step)
    levels = sorted(set(round(l, 5) for l in levels))
    out = []
    for l in levels:
        out.append(RoundNumber(price=l, distance_pips=(l - price) / pip))
    return tuple(out)
