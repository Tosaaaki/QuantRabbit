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

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
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
_BASE_DIR = Path(__file__).resolve().parents[1]
_SNAPSHOT_PATH = (_BASE_DIR / "logs" / "orderbook_snapshot.json").resolve()
_FLUSH_INTERVAL_SEC = 1.0
_LOGGER = logging.getLogger(__name__)
_last_flush_ts: float = 0.0
_cache_mtime: float = 0.0
_last_persist_error_ts: float = 0.0


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
    _persist_snapshot(snapshot)


def get_latest(max_age_ms: Optional[float] = None) -> Optional[OrderBookSnapshot]:
    """Return the latest snapshot when present and fresh enough."""

    _reload_snapshot_if_updated()
    with _LOCK:
        snapshot = _SNAPSHOT
        age_ms = (time.monotonic() - _MONOTONIC_TS) * 1000 if snapshot else None
    if snapshot is None:
        return None
    if max_age_ms is not None and (age_ms or 0.0) > max_age_ms:
        return None
    return snapshot


def latest_age_ms() -> Optional[float]:
    _reload_snapshot_if_updated()
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


def _persist_snapshot(snapshot: OrderBookSnapshot) -> None:
    """Persist the latest snapshot for cross-process consumers."""

    global _last_flush_ts, _last_persist_error_ts, _cache_mtime
    now = time.time()
    if now - _last_flush_ts < _FLUSH_INTERVAL_SEC:
        return
    _last_flush_ts = now
    try:
        payload = {
            "epoch_ts": snapshot.epoch_ts,
            "provider": snapshot.provider,
            "latency_ms": snapshot.latency_ms,
            "seq": snapshot.seq,
            "bid_levels": [[lvl.price, lvl.size] for lvl in snapshot.bid_levels],
            "ask_levels": [[lvl.price, lvl.size] for lvl in snapshot.ask_levels],
        }
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _SNAPSHOT_PATH.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(_SNAPSHOT_PATH)
        try:
            _cache_mtime = float(_SNAPSHOT_PATH.stat().st_mtime)
        except Exception:
            pass
    except Exception as exc:  # noqa: BLE001
        if now - _last_persist_error_ts >= 30.0:
            _LOGGER.warning("[ORDERBOOK] persist failed: %s", exc)
            _last_persist_error_ts = now


def _reload_snapshot_if_updated() -> None:
    """Reload snapshot from disk when another process updated it."""

    global _SNAPSHOT, _MONOTONIC_TS, _cache_mtime
    try:
        stat = _SNAPSHOT_PATH.stat()
    except FileNotFoundError:
        return
    except Exception:
        return
    mtime = float(stat.st_mtime)
    if mtime <= _cache_mtime:
        return
    try:
        payload = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        bid_levels = _normalize_levels(payload.get("bid_levels", ()))
        ask_levels = _normalize_levels(payload.get("ask_levels", ()))
        if not bid_levels or not ask_levels:
            return
        epoch_ts = float(payload.get("epoch_ts", 0.0))
        provider = payload.get("provider")
        latency_ms = payload.get("latency_ms")
        seq = payload.get("seq")
    except Exception:
        return

    snapshot = OrderBookSnapshot(
        epoch_ts=epoch_ts,
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        provider=str(provider) if provider is not None else None,
        latency_ms=float(latency_ms) if latency_ms is not None else None,
        seq=int(seq) if seq is not None else None,
    )
    with _LOCK:
        _SNAPSHOT = snapshot
        age_sec = max(0.0, time.time() - epoch_ts)
        _MONOTONIC_TS = time.monotonic() - age_sec
        _cache_mtime = mtime
