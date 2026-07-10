"""OANDA OHLC candle fetcher.

The trader needs current candles to read the chart. This module returns
plain-Python tuples of `Candle` records so downstream indicator code stays
dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Sequence

from quant_rabbit.broker.oanda import OandaReadOnlyClient


SUPPORTED_GRANULARITIES: frozenset[str] = frozenset({"M1", "M5", "M15", "M30", "H1", "H4", "D"})


@dataclass(frozen=True)
class Candle:
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    complete: bool = True


def fetch_candles(
    pair: str,
    granularity: str,
    *,
    count: int = 200,
    price: str = "M",
    client: OandaReadOnlyClient | None = None,
) -> tuple[Candle, ...]:
    """Fetch up to `count` recent candles for a pair-granularity.

    `price="M"` returns mid candles (default). Use `"BA"` for bid/ask if needed
    by callers — the parser pulls whichever block is present, preferring mid.
    """

    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"unsupported granularity {granularity!r}; expected one of {sorted(SUPPORTED_GRANULARITIES)}")
    client = client or OandaReadOnlyClient()
    payload = client.get_json(
        f"/v3/instruments/{pair}/candles",
        {"granularity": granularity, "count": str(int(count)), "price": price},
    )
    return _candles_from_payload(payload)


def fetch_candles_between(
    pair: str,
    granularity: str,
    *,
    time_from: datetime,
    time_to: datetime,
    price: str = "M",
    client: OandaReadOnlyClient | None = None,
) -> tuple[Candle, ...]:
    """Fetch complete candles inside an explicit UTC time window.

    This is read-only historical evidence plumbing. Callers that need long
    windows are responsible for chunking the request; OANDA caps candle result
    sizes, and hiding pagination here would make long evidence jobs opaque.
    """

    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"unsupported granularity {granularity!r}; expected one of {sorted(SUPPORTED_GRANULARITIES)}")
    client = client or OandaReadOnlyClient()
    payload = client.get_json(
        f"/v3/instruments/{pair}/candles",
        {
            "granularity": granularity,
            "from": _format_oanda_time(time_from),
            "to": _format_oanda_time(time_to),
            "price": price,
            "includeFirst": "true",
        },
    )
    return _candles_from_payload(payload)


def fetch_candles_via_client(
    client: OandaReadOnlyClient,
    pair: str,
    granularity: str,
    *,
    count: int = 200,
    price: str = "M",
) -> tuple[Candle, ...]:
    """Same as `fetch_candles` but with the client passed positionally for tests."""

    return fetch_candles(pair, granularity, count=count, price=price, client=client)


def _candles_from_payload(payload: dict) -> tuple[Candle, ...]:
    candles: list[Candle] = []
    for entry in payload.get("candles") or []:
        # Strategy indicators and structure must never repaint from OANDA's
        # still-forming tail candle. Live bid/ask quotes are consumed through
        # broker snapshots, so excluding an incomplete bar does not make the
        # guardian blind to price displacement; it keeps the technical state
        # anchored to closed evidence as downstream pattern code expects.
        if entry.get("complete") is False:
            continue
        block = entry.get("mid") or entry.get("ask") or entry.get("bid") or {}
        try:
            timestamp = _parse_oanda_time(entry.get("time"))
            o = float(block.get("o"))
            h = float(block.get("h"))
            low_value = float(block.get("l"))
            c = float(block.get("c"))
        except (TypeError, ValueError):
            continue
        if timestamp is None:
            continue
        candles.append(
            Candle(
                timestamp_utc=timestamp,
                open=o,
                high=h,
                low=low_value,
                close=c,
                volume=int(entry.get("volume") or 0),
                complete=bool(entry.get("complete", True)),
            )
        )
    return tuple(candles)


def _parse_oanda_time(value: object) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        core = text[:-1]
        if "." in core:
            head, frac = core.split(".", 1)
            text = f"{head}.{frac[:6]}+00:00"
        else:
            text = f"{core}+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _format_oanda_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def closes(candles: Sequence[Candle]) -> tuple[float, ...]:
    return tuple(c.close for c in candles)


def highs(candles: Sequence[Candle]) -> tuple[float, ...]:
    return tuple(c.high for c in candles)


def lows(candles: Sequence[Candle]) -> tuple[float, ...]:
    return tuple(c.low for c in candles)


def volumes(candles: Sequence[Candle]) -> tuple[int, ...]:
    return tuple(c.volume for c in candles)
