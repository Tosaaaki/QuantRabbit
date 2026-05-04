"""OANDA-native flow snapshot — order book, position book, spread time series.

OANDA exposes two endpoints that show where retail orders cluster:

- `/v3/instruments/{pair}/orderBook` — pending orders by price (stop-magnet
  zones above/below current price).
- `/v3/instruments/{pair}/positionBook` — currently open positions distribution
  (where retail is long vs short).

These are read-only public-facing endpoints and require the same OANDA token
the trader already uses. We snapshot them per pair and write a compact summary
the trader can cite: "57 % of retail SHORT at AUD/JPY 113.10, biggest cluster
of stops at 113.40 — sweep risk to the upside".

We also build a spread time-series proxy from M1 candle ranges + the latest
quote spread, so the trader can see how spread looks vs the recent average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

from quant_rabbit.analysis.candles import Candle, fetch_candles_via_client
from quant_rabbit.broker.oanda import OandaReadOnlyClient


@dataclass(frozen=True)
class BookBucket:
    price: float
    long_pct: float
    short_pct: float

    def to_dict(self) -> dict[str, object]:
        return {"price": self.price, "long_pct": self.long_pct, "short_pct": self.short_pct}


@dataclass(frozen=True)
class BookSnapshot:
    instrument: str
    kind: str  # "ORDER" or "POSITION"
    timestamp_utc: str | None
    price: float | None  # OANDA-supplied last price at snapshot time
    bucket_width: float | None
    long_total_pct: float
    short_total_pct: float
    top_long_clusters: tuple[BookBucket, ...]  # highest long_pct
    top_short_clusters: tuple[BookBucket, ...]  # highest short_pct
    issue: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "kind": self.kind,
            "timestamp_utc": self.timestamp_utc,
            "price": self.price,
            "bucket_width": self.bucket_width,
            "long_total_pct": self.long_total_pct,
            "short_total_pct": self.short_total_pct,
            "top_long_clusters": [b.to_dict() for b in self.top_long_clusters],
            "top_short_clusters": [b.to_dict() for b in self.top_short_clusters],
            "issue": self.issue,
        }


@dataclass(frozen=True)
class SpreadStat:
    instrument: str
    current_pips: float | None
    median_pips: float | None
    p90_pips: float | None
    max_pips: float | None
    sample_size: int
    stress_flag: str  # "NORMAL" / "ELEVATED" / "STRESSED"

    def to_dict(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "current_pips": self.current_pips,
            "median_pips": self.median_pips,
            "p90_pips": self.p90_pips,
            "max_pips": self.max_pips,
            "sample_size": self.sample_size,
            "stress_flag": self.stress_flag,
        }


@dataclass(frozen=True)
class FlowSnapshot:
    generated_at_utc: str
    order_books: tuple[BookSnapshot, ...] = field(default_factory=tuple)
    position_books: tuple[BookSnapshot, ...] = field(default_factory=tuple)
    spreads: tuple[SpreadStat, ...] = field(default_factory=tuple)
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "order_books": [ob.to_dict() for ob in self.order_books],
            "position_books": [pb.to_dict() for pb in self.position_books],
            "spreads": [s.to_dict() for s in self.spreads],
            "issues": list(self.issues),
        }


def build_flow_snapshot(
    *,
    client: OandaReadOnlyClient,
    pairs: Sequence[str],
    top_n: int = 5,
    spread_lookback_minutes: int = 60,
) -> FlowSnapshot:
    """Fetch order/position books + recent spread stats for each pair."""

    issues: list[str] = []
    order_books: list[BookSnapshot] = []
    position_books: list[BookSnapshot] = []
    spreads: list[SpreadStat] = []

    for pair in pairs:
        # Order book
        try:
            payload = client.get_json(f"/v3/instruments/{pair}/orderBook")
            order_books.append(_book_from_payload(pair, payload, kind="ORDER", top_n=top_n))
        except Exception as exc:
            issues.append(f"MISSING_ORDERBOOK_{pair}: {exc}")
            order_books.append(BookSnapshot(
                instrument=pair, kind="ORDER", timestamp_utc=None, price=None,
                bucket_width=None, long_total_pct=0.0, short_total_pct=0.0,
                top_long_clusters=tuple(), top_short_clusters=tuple(),
                issue=f"{exc}",
            ))

        # Position book
        try:
            payload = client.get_json(f"/v3/instruments/{pair}/positionBook")
            position_books.append(_book_from_payload(pair, payload, kind="POSITION", top_n=top_n))
        except Exception as exc:
            issues.append(f"MISSING_POSITIONBOOK_{pair}: {exc}")
            position_books.append(BookSnapshot(
                instrument=pair, kind="POSITION", timestamp_utc=None, price=None,
                bucket_width=None, long_total_pct=0.0, short_total_pct=0.0,
                top_long_clusters=tuple(), top_short_clusters=tuple(),
                issue=f"{exc}",
            ))

        # Spread time-series proxy: pull current quote + M1 candles
        try:
            spread_stat = _spread_stat(client, pair, lookback_minutes=spread_lookback_minutes)
            spreads.append(spread_stat)
        except Exception as exc:
            issues.append(f"MISSING_SPREAD_{pair}: {exc}")

    return FlowSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        order_books=tuple(order_books),
        position_books=tuple(position_books),
        spreads=tuple(spreads),
        issues=tuple(issues),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _book_from_payload(pair: str, payload: dict, *, kind: str, top_n: int) -> BookSnapshot:
    book_key = "orderBook" if kind == "ORDER" else "positionBook"
    book = payload.get(book_key) or {}
    timestamp = book.get("time")
    price = _opt_float(book.get("price"))
    bucket_width = _opt_float(book.get("bucketWidth"))
    raw_buckets = book.get("buckets") or []
    buckets: list[BookBucket] = []
    long_total = 0.0
    short_total = 0.0
    for b in raw_buckets:
        p = _opt_float(b.get("price"))
        lp = _opt_float(b.get("longCountPercent"))
        sp = _opt_float(b.get("shortCountPercent"))
        if p is None or lp is None or sp is None:
            continue
        buckets.append(BookBucket(price=p, long_pct=lp, short_pct=sp))
        long_total += lp
        short_total += sp

    top_long = tuple(sorted(buckets, key=lambda b: b.long_pct, reverse=True)[:top_n])
    top_short = tuple(sorted(buckets, key=lambda b: b.short_pct, reverse=True)[:top_n])
    return BookSnapshot(
        instrument=pair, kind=kind, timestamp_utc=timestamp, price=price,
        bucket_width=bucket_width, long_total_pct=long_total, short_total_pct=short_total,
        top_long_clusters=top_long, top_short_clusters=top_short,
    )


def _spread_stat(client: OandaReadOnlyClient, pair: str, *, lookback_minutes: int = 60) -> SpreadStat:
    """Build spread stats: current bid/ask spread + recent M1 candle ranges as proxy."""

    # 1. Current quote — bid/ask spread in pips
    pricing = client.get_json(
        f"/v3/accounts/{client.account_id}/pricing",
        {"instruments": pair},
    )
    pip = 0.01 if pair.upper().endswith("_JPY") else 0.0001
    current_pips = None
    for entry in pricing.get("prices") or []:
        if entry.get("instrument") != pair:
            continue
        bids = entry.get("bids") or []
        asks = entry.get("asks") or []
        if bids and asks:
            current_pips = (float(asks[0]["price"]) - float(bids[0]["price"])) / pip

    # 2. Recent M1 candles using bid/ask
    samples_pips: list[float] = []
    try:
        ba_payload = client.get_json(
            f"/v3/instruments/{pair}/candles",
            {"granularity": "M1", "count": str(int(lookback_minutes)), "price": "BA"},
        )
        for entry in ba_payload.get("candles") or []:
            bid = (entry.get("bid") or {}).get("c")
            ask = (entry.get("ask") or {}).get("c")
            if bid is None or ask is None:
                continue
            try:
                samples_pips.append((float(ask) - float(bid)) / pip)
            except (TypeError, ValueError):
                continue
    except Exception:
        pass

    sample_size = len(samples_pips)
    med = median(samples_pips) if samples_pips else None
    p90 = _percentile(samples_pips, 0.9) if samples_pips else None
    mx = max(samples_pips) if samples_pips else None

    flag = "NORMAL"
    if current_pips is not None and med is not None and med > 0:
        ratio = current_pips / med
        if ratio >= 2.5:
            flag = "STRESSED"
        elif ratio >= 1.5:
            flag = "ELEVATED"

    return SpreadStat(
        instrument=pair, current_pips=current_pips, median_pips=med,
        p90_pips=p90, max_pips=mx, sample_size=sample_size, stress_flag=flag,
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _opt_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
