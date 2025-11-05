"""
replay_orderbook_support
~~~~~~~~~~~~~~~~~~~~~~~~
Read tick JSONL with depth information and emit synthetic L2 snapshots
compatible with market_data.orderbook_state for shadow testing.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Iterable, List, Tuple

from market_data import orderbook_state


def _load_ticks(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield payload


def push_snapshots(path: Path) -> int:
    count = 0
    for tick in _load_ticks(path):
        bids: List[Tuple[float, float]] = []
        for entry in tick.get("bids") or []:
            if isinstance(entry, dict):
                price = entry.get("price")
                size = entry.get("liquidity") or entry.get("size")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                price, size = entry[0], entry[1]
            else:
                continue
            try:
                bids.append((float(price), float(size)))
            except (TypeError, ValueError):
                continue

        asks: List[Tuple[float, float]] = []
        for entry in tick.get("asks") or []:
            if isinstance(entry, dict):
                price = entry.get("price")
                size = entry.get("liquidity") or entry.get("size")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                price, size = entry[0], entry[1]
            else:
                continue
            try:
                asks.append((float(price), float(size)))
            except (TypeError, ValueError):
                continue
        if not bids or not asks:
            continue
        epoch = tick.get("epoch")
        if epoch is None:
            ts = tick.get("ts") or tick.get("time")
            if isinstance(ts, str):
                try:
                    parsed = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    try:
                        parsed = datetime.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
                    except ValueError:
                        parsed = None
                if parsed is not None:
                    epoch = parsed.timestamp()
        if epoch is None:
            continue
        try:
            orderbook_state.update_snapshot(
                epoch_ts=float(epoch),
                bids=tuple(bids),
                asks=tuple(asks),
                provider="replay",
                latency_ms=None,
            )
            count += 1
        except Exception:
            continue
    return count


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay depth snapshots")
    ap.add_argument("ticks", type=Path, help="Tick JSONL with bids/asks arrays")
    args = ap.parse_args()
    pushed = push_snapshots(args.ticks)
    print(f"pushed {pushed} snapshots from {args.ticks}")
