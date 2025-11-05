"""
market_data.orderbook_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orderbook snapshot cache shared across maker-type workers.

This module intentionally keeps the interface lightweight so that
future connectors (FIX/REST/WebSocket) can push level-2 snapshots
without coupling to worker internals.  Consumers can query the latest
snapshot, inspect aggregate depth, or derive imbalance metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(slots=True, frozen=True)
class OrderBookLevel:
    """Single price level in the orderbook."""

    price: float
    size: float


@dataclass(slots=True, frozen=True)
class OrderBookSnapshot:
    """Normalized orderbook snapshot for USD/JPY."""

    epoch_ts: float  # seconds since epoch (UTC)
    bid_levels: Tuple[OrderBookLevel, ...]
    ask_levels: Tuple[OrderBookLevel, ...]
    provider: str | None = None
    latency_ms: float | None = None
    seq: int | None = None  # LP provided sequence number

    @property
    def spread(self) -> float:
        if not self.bid_levels or not self.ask_levels:
            return 0.0
        return max(0.0, self.ask_levels[0].price - self.bid_levels[0].price)

    @property
    def mid(self) -> float:
        if not self.bid_levels or not self.ask_levels:
            return 0.0
        return (self.bid_levels[0].price + self.ask_levels[0].price) * 0.5

    def aggregate_depth(self, depth: int = 1) -> tuple[float, float]:
        depth = max(1, depth)
        bid_total = sum(level.size for level in self.bid_levels[:depth])
        ask_total = sum(level.size for level in self.ask_levels[:depth])
        return float(bid_total), float(ask_total)


_LOCK = Lock()
_SNAPSHOT: OrderBookSnapshot | None = None
_MONOTONIC_TS: float = 0.0


def _normalize_levels(levels: Iterable[Tuple[float, float]]) -> Tuple[OrderBookLevel, ...]:
    normalized = []
    for price, size in levels:
        try:
            price_f = float(price)
            size_f = float(size)
        except (TypeError, ValueError):
            continue
        normalized.append(OrderBookLevel(price=price_f, size=size_f))
    return tuple(normalized)


def update_snapshot(
    *,
    epoch_ts: float,
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
    provider: str | None = None,
    latency_ms: float | None = None,
    seq: int | None = None,
) -> None:
    """Store the latest orderbook snapshot."""

    bid_levels = _normalize_levels(bids)
    ask_levels = _normalize_levels(asks)
    if not bid_levels or not ask_levels:
        return
    snapshot = OrderBookSnapshot(
        epoch_ts=float(epoch_ts),
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        provider=provider,
        latency_ms=float(latency_ms) if latency_ms is not None else None,
        seq=seq,
    )
    monotonic_now = time.monotonic()
    with _LOCK:
        global _SNAPSHOT, _MONOTONIC_TS
        _SNAPSHOT = snapshot
        _MONOTONIC_TS = monotonic_now


def get_latest(max_age_ms: Optional[float] = None) -> Optional[OrderBookSnapshot]:
    """Return the latest snapshot when present and fresh enough."""

    with _LOCK:
        snapshot = _SNAPSHOT
        age_ms = (time.monotonic() - _MONOTONIC_TS) * 1000 if snapshot else None
    if snapshot is None:
        return None
    if max_age_ms is not None and (age_ms or 0.0) > max_age_ms:
        return None
    return snapshot


def latest_age_ms() -> Optional[float]:
    with _LOCK:
        if _SNAPSHOT is None:
            return None
        return max(0.0, (time.monotonic() - _MONOTONIC_TS) * 1000)


def queue_imbalance(snapshot: OrderBookSnapshot, depth: int = 1) -> Optional[float]:
    """Compute queue imbalance (bid - ask) / (bid + ask) for a given depth."""

    if depth <= 0:
        depth = 1
    bid_total, ask_total = snapshot.aggregate_depth(depth)
    denom = bid_total + ask_total
    if denom <= 0.0:
        return None
    return (bid_total - ask_total) / denom


def describe(snapshot: OrderBookSnapshot) -> dict[str, object]:
    """Return a JSON-serialisable summary of the snapshot."""

    bid_levels = [
        {"price": lvl.price, "size": lvl.size} for lvl in snapshot.bid_levels[:5]
    ]
    ask_levels = [
        {"price": lvl.price, "size": lvl.size} for lvl in snapshot.ask_levels[:5]
    ]
    return {
        "epoch_ts": snapshot.epoch_ts,
        "provider": snapshot.provider,
        "latency_ms": snapshot.latency_ms,
        "spread": snapshot.spread,
        "mid": snapshot.mid,
        "bid_levels": bid_levels,
        "ask_levels": ask_levels,
    }


def has_sufficient_depth(
    snapshot: OrderBookSnapshot,
    *,
    depth: int = 1,
    min_size: float = 100000.0,
) -> bool:
    """Return True when both sides have at least ``depth`` levels with size >= ``min_size``."""

    depth = max(1, depth)
    if len(snapshot.bid_levels) < depth or len(snapshot.ask_levels) < depth:
        return False
    for lvl in snapshot.bid_levels[:depth]:
        if lvl.size < min_size:
            return False
    for lvl in snapshot.ask_levels[:depth]:
        if lvl.size < min_size:
            return False
    return True
