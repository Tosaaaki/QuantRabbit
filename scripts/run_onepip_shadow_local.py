#!/usr/bin/env python3
"""
Run onepip_maker_s1 in shadow mode locally by pushing synthetic orderbook
snapshots into market_data.orderbook_state.

This avoids external dependencies and produces logs/onepip_maker_s1_shadow.jsonl
for quick inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from typing import Tuple
from pathlib import Path
import sys

# Ensure repository root is on sys.path for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))


def _set_env() -> None:
    # Enable the worker in shadow mode, disable strict range gating for offline
    os.environ.setdefault("ONEPIP_MAKER_S1_ENABLED", "true")
    os.environ.setdefault("ONEPIP_MAKER_S1_SHADOW_MODE", "true")
    os.environ.setdefault("ONEPIP_MAKER_S1_RANGE_ONLY", "false")
    # Keep cost guard permissive but realistic; bootstrap from logs if present
    os.environ.setdefault("ONEPIP_MAKER_S1_MAX_COST_PIPS", "1.2")
    os.environ.setdefault("ONEPIP_MAKER_S1_COST_BOOTSTRAP_FILES", "4")
    os.environ.setdefault("ONEPIP_MAKER_S1_COST_BOOTSTRAP_LINES", "800")
    os.environ.setdefault("ONEPIP_MAKER_S1_MIN_COST_SAMPLES", "1")
    # Allow spread up to 0.3 pips for synthetic snapshots
    os.environ.setdefault("ONEPIP_MAKER_S1_MAX_SPREAD_PIPS", "0.30")


async def _push_snapshots(duration_sec: float, hz: float, base: float, spread_pips: float) -> None:
    from market_data import orderbook_state

    period = 1.0 / max(1.0, hz)
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        # Small mean-reverting random walk around base price
        base_noise = random.uniform(-0.002, 0.002)
        base += base_noise * 0.5
        # Spread in price units (1 pip = 0.01)
        spr = max(0.0005, spread_pips * 0.01)
        mid = base
        bid = round(mid - spr / 2.0, 3)
        ask = round(mid + spr / 2.0, 3)
        # Alternate queue imbalance to trigger entries
        if int((end - time.monotonic()) // period) % 2 == 0:
            bid_sz, ask_sz = 1_300_000.0, 600_000.0  # BUY imbalance ~0.368
        else:
            bid_sz, ask_sz = 600_000.0, 1_300_000.0  # SELL imbalance ~-0.368
        bids = (
            (bid, bid_sz),
            (round(bid - 0.001, 3), bid_sz * 0.7),
        )
        asks = (
            (ask, ask_sz),
            (round(ask + 0.001, 3), ask_sz * 0.7),
        )
        orderbook_state.update_snapshot(
            epoch_ts=time.time(),
            bids=bids,
            asks=asks,
            provider="synthetic",
            latency_ms=5.0,
        )
        await asyncio.sleep(period)


async def _run_worker(duration_sec: float) -> None:
    # Import after env so config picks up values
    from workers.onepip_maker_s1 import onepip_maker_s1_worker

    task = asyncio.create_task(onepip_maker_s1_worker())
    try:
        await asyncio.sleep(duration_sec)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def main() -> None:
    ap = argparse.ArgumentParser(description="Run onepip maker shadow locally")
    ap.add_argument("--duration", type=float, default=60.0, help="Run seconds (default 60s)")
    ap.add_argument("--hz", type=float, default=4.0, help="Snapshots per second (default 4)")
    ap.add_argument("--base", type=float, default=150.000, help="Base mid price (default 150.000)")
    ap.add_argument("--spread-pips", type=float, default=0.18, help="Synthetic spread in pips (default 0.18)")
    args = ap.parse_args()

    _set_env()
    # Run pusher and worker concurrently
    await asyncio.gather(
        _push_snapshots(args.duration, args.hz, args.base, args.spread_pips),
        _run_worker(args.duration + 0.5),
    )


if __name__ == "__main__":
    asyncio.run(main())
