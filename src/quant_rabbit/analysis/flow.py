"""OANDA-native flow snapshot — spread time series plus optional book data.

OANDA exposes two endpoints that show where retail orders cluster:

- `/v3/instruments/{pair}/orderBook` — pending orders by price (stop-magnet
  zones above/below current price).
- `/v3/instruments/{pair}/positionBook` — currently open positions distribution
  (where retail is long vs short).

NOTE on access: these endpoints require OANDA account/feed entitlement for
book data. Many live retail accounts return 401 Invalid authentication for
them even though `/candles` and `/pricing` work with the same credential.
The production default therefore does not call book endpoints at all. Enable
book fetch only after the entitlement/account tier is confirmed, or plug in
an alternate book feed.

We also build a spread time-series proxy from M1 bid/ask candle ranges + the
latest quote spread, so the trader can see how spread looks vs the recent
average. This part works on every account.
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
    book_fetch_enabled: bool = False
    book_fetch_reason: str | None = None
    order_books: tuple[BookSnapshot, ...] = field(default_factory=tuple)
    position_books: tuple[BookSnapshot, ...] = field(default_factory=tuple)
    spreads: tuple[SpreadStat, ...] = field(default_factory=tuple)
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "book_fetch_enabled": self.book_fetch_enabled,
            "book_fetch_reason": self.book_fetch_reason,
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
    include_books: bool = False,
) -> FlowSnapshot:
    """Fetch recent spread stats, and optionally order/position books.

    OANDA book endpoints are disabled by default because unavailable book
    entitlement returns a persistent 401 and adds latency without evidence.
    """

    issues: list[str] = []
    order_books: list[BookSnapshot] = []
    position_books: list[BookSnapshot] = []
    spreads: list[SpreadStat] = []
    order_book_unavailable: str | None = None
    position_book_unavailable: str | None = None

    for pair in pairs:
        if include_books:
            # Order book
            if order_book_unavailable is not None:
                order_books.append(_missing_book_snapshot(pair, "ORDER", order_book_unavailable))
            else:
                try:
                    payload = client.get_json(f"/v3/instruments/{pair}/orderBook")
                    order_books.append(_book_from_payload(pair, payload, kind="ORDER", top_n=top_n))
                except Exception as exc:
                    auth_issue = _book_authorization_issue("ORDERBOOK", exc)
                    if auth_issue is not None:
                        order_book_unavailable = auth_issue
                        issues.append(auth_issue)
                        order_books.append(_missing_book_snapshot(pair, "ORDER", auth_issue))
                    else:
                        issue = f"MISSING_ORDERBOOK_{pair}: {exc}"
                        issues.append(issue)
                        order_books.append(_missing_book_snapshot(pair, "ORDER", issue))

            # Position book
            if position_book_unavailable is not None:
                position_books.append(_missing_book_snapshot(pair, "POSITION", position_book_unavailable))
            else:
                try:
                    payload = client.get_json(f"/v3/instruments/{pair}/positionBook")
                    position_books.append(_book_from_payload(pair, payload, kind="POSITION", top_n=top_n))
                except Exception as exc:
                    auth_issue = _book_authorization_issue("POSITIONBOOK", exc)
                    if auth_issue is not None:
                        position_book_unavailable = auth_issue
                        issues.append(auth_issue)
                        position_books.append(_missing_book_snapshot(pair, "POSITION", auth_issue))
                    else:
                        issue = f"MISSING_POSITIONBOOK_{pair}: {exc}"
                        issues.append(issue)
                        position_books.append(_missing_book_snapshot(pair, "POSITION", issue))

        # Spread time-series proxy: pull current quote + M1 candles
        try:
            spread_stat = _spread_stat(client, pair, lookback_minutes=spread_lookback_minutes)
            spreads.append(spread_stat)
        except Exception as exc:
            issues.append(f"MISSING_SPREAD_{pair}: {exc}")

    return FlowSnapshot(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        book_fetch_enabled=include_books,
        book_fetch_reason=None if include_books else "disabled_by_default_enable_with_include_books",
        order_books=tuple(order_books),
        position_books=tuple(position_books),
        spreads=tuple(spreads),
        issues=tuple(issues),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _book_authorization_issue(feed: str, exc: Exception) -> str | None:
    text = str(exc)
    lowered = text.lower()
    if "401" not in lowered and "unauthorized" not in lowered and "invalid authentication" not in lowered:
        return None
    return (
        f"{feed}_FEED_UNAUTHORIZED: OANDA book endpoint returned authorization error; "
        "spread stats remain usable, but book clusters are unavailable until the "
        "account/feed entitlement is enabled or an alternate book feed is plugged in"
    )


def _missing_book_snapshot(pair: str, kind: str, issue: str) -> BookSnapshot:
    return BookSnapshot(
        instrument=pair,
        kind=kind,
        timestamp_utc=None,
        price=None,
        bucket_width=None,
        long_total_pct=0.0,
        short_total_pct=0.0,
        top_long_clusters=tuple(),
        top_short_clusters=tuple(),
        issue=issue,
    )


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
