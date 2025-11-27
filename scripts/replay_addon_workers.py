#!/usr/bin/env python3
"""Replay harness for the QuantRabbit addon workers bundle."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from addons import AddonSystem, ReplayBroker, ReplayDataFeed
from workers import (
    MM_LITE_CONFIG,
    SESSION_OPEN_CONFIG,
    STOP_RUN_REVERSAL_CONFIG,
    VOL_SQUEEZE_CONFIG,
)


def _parse_epoch(ts: str) -> float:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc).timestamp()


def load_ticks(path: Path) -> List[Dict[str, float]]:
    ticks: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            ts = data.get("ts") or data.get("time")
            if not ts:
                continue
            epoch = _parse_epoch(str(ts))
            mid = data.get("mid")
            if mid is None:
                bid = data.get("bid")
                ask = data.get("ask")
                if bid is None or ask is None:
                    continue
                mid = (float(bid) + float(ask)) * 0.5
            ticks.append(
                {
                    "epoch": epoch,
                    "mid": float(mid),
                    "bid": float(data.get("bid", mid or 0.0) or 0.0),
                    "ask": float(data.get("ask", mid or 0.0) or 0.0),
                }
            )
    return ticks


def build_time_bars(ticks: Iterable[Dict[str, float]], interval_sec: int) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = {}
    for tick in ticks:
        epoch = float(tick["epoch"])
        price = float(tick["mid"])
        bucket = int(math.floor(epoch / interval_sec))
        bucket_end = (bucket + 1) * interval_sec
        bar = buckets.get(bucket)
        if bar is None:
            bar = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "timestamp": float(bucket_end),
                "volume": 1,
            }
            buckets[bucket] = bar
        else:
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["volume"] += 1
    ordered = [buckets[idx] for idx in sorted(buckets)]
    return ordered


def _default_sessions() -> List[Dict[str, Any]]:
    return [
        {"tz": "Asia/Tokyo", "start": "00:00", "build_minutes": 25, "hold_minutes": 240},
        {"tz": "Europe/London", "start": "07:00", "build_minutes": 20, "hold_minutes": 240},
        {"tz": "America/New_York", "start": "12:30", "build_minutes": 20, "hold_minutes": 240},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, required=True, help="Tick JSONL file (mid/bid/ask).")
    parser.add_argument("--symbol", default="USD_JPY", help="Instrument symbol for replay.")
    parser.add_argument("--out", type=Path, default=Path("tmp/replay_addon_workers.json"), help="Output JSON path.")
    parser.add_argument("--spread-bp", type=float, default=1.2, help="Synthetic spread in basis points.")
    parser.add_argument("--warmup-bars", type=int, default=200, help="Bars to skip at start for warmup.")
    parser.add_argument("--max-steps", type=int, default=720, help="Maximum replay steps after warmup.")
    args = parser.parse_args()

    ticks = load_ticks(args.ticks)
    if not ticks:
        raise SystemExit(f"No ticks parsed from {args.ticks}")

    bars_1m = build_time_bars(ticks, 60)
    bars_5m = build_time_bars(ticks, 300)

    feed = ReplayDataFeed(spread_bp=args.spread_bp)
    feed.add_bars(args.symbol, "1m", bars_1m)
    feed.add_bars(args.symbol, "5m", bars_5m)

    broker = ReplayBroker()

    session_cfg = dict(SESSION_OPEN_CONFIG)
    session_cfg.update(
        {
            "universe": [args.symbol],
            "sessions": _default_sessions(),
            "place_orders": False,
            "filters": {"max_spread_bp": 12, "min_bars": max(240, args.warmup_bars)},
            "exit": {"stop_atr": 1.5, "tp_atr": 2.4, "breakeven_mult": 0.6},
        }
    )

    squeeze_cfg = dict(VOL_SQUEEZE_CONFIG)
    squeeze_cfg.update(
        {
            "universe": [args.symbol],
            "place_orders": False,
            "allow_long": False,
            "allow_short": True,
            "ema_slope_min": 0.0008,
            "cooldown_bars": 3,
            "exit": {"stop_atr": 0.5, "tp_atr": 0.6, "breakeven_mult": 0.35},
        }
    )

    stop_run_cfg = False  # disabled for now; re-enable via CLI overrides when retuned

    mm_cfg = dict(MM_LITE_CONFIG)
    mm_cfg.update(
        {"universe": [args.symbol], "place_orders": False, "tick_size": 0.001, "disable_on_event": False}
    )

    system = AddonSystem(
        feed,
        broker,
        session_open_cfg=session_cfg,
        vol_squeeze_cfg=squeeze_cfg,
        stop_run_cfg=stop_run_cfg,
        mm_lite_cfg=mm_cfg,
    )

    timeline = [bar["timestamp"] for bar in bars_1m]
    timeline = timeline[args.warmup_bars :]
    if args.max_steps > 0:
        timeline = timeline[: args.max_steps]

    outcome = system.run(timeline)
    summary = system.summarize(outcome.steps)
    trade_stats = system.summarize_trades(outcome.trades)
    budgets = system.allocate_budgets(summary) if summary else {}

    events = [
        {"ts": step.ts, "iso": step.iso, "signals": step.signals}
        for step in outcome.steps
    ]
    trades = [asdict(trade) for trade in outcome.trades]

    payload = {
        "symbol": args.symbol,
        "source": str(args.ticks),
        "tick_count": len(ticks),
        "bars": {"1m": len(bars_1m), "5m": len(bars_5m)},
        "warmup_bars": args.warmup_bars,
        "steps_simulated": len(timeline),
        "summary": summary,
        "budgets_bps": budgets,
        "trade_pnl": trade_stats,
        "trades": trades,
        "events": events,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"[addon-replay] wrote {args.out} with {len(events)} signal events and {len(trades)} trades")
    if summary:
        for name, stats in summary.items():
            print(
                f"  - {name}: signals={stats['signals']} long={stats['long']} short={stats['short']} budget={budgets.get(name, 0.0):.2f}bps"
            )
    if trade_stats:
        for name, stats in trade_stats.items():
            pnl = stats["pnl_pips"]
            print(
                f"    > {name}: trades={stats['trades']} wins={stats['wins']} losses={stats['losses']} pnl={pnl:.2f} pips"
            )


if __name__ == "__main__":
    main()
