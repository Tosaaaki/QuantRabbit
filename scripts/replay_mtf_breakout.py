#!/usr/bin/env python3
"""Replay harness for the MTF breakout worker."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workers.mtf_breakout import MtfBreakoutWorker
from workers.mtf_breakout.config import DEFAULT_CONFIG

PIP_SIZE = 0.01

@dataclass
class Tick:
    epoch: float
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


def load_ticks(path: Path) -> List[Tick]:
    ticks: List[Tick] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            data = json.loads(payload)
            ts = data.get("timestamp") or data.get("ts") or data.get("time")
            if ts is None:
                continue
            if isinstance(ts, (int, float)):
                epoch = float(ts)
                if epoch > 10**11:  # assume milliseconds
                    epoch /= 1000.0
            else:
                epoch = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
            bid = float(data.get("bid", 0.0))
            ask = float(data.get("ask", 0.0))
            ticks.append(Tick(epoch=epoch, bid=bid, ask=ask))
    ticks.sort(key=lambda t: t.epoch)
    return ticks


def build_bars(ticks: Sequence[Tick], seconds: int) -> List[Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {}
    for tick in ticks:
        bucket_id = int(math.floor(tick.epoch / seconds))
        price = tick.mid
        bar = buckets.get(bucket_id)
        if bar is None:
            start = bucket_id * seconds
            bar = {
                "start": float(start),
                "end": float(start + seconds),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
            buckets[bucket_id] = bar
        else:
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
    ordered: List[Dict[str, float]] = []
    for bucket in sorted(buckets):
        ordered.append(buckets[bucket])
    return ordered


class SliceDataFeed:
    """Replay data feed that exposes bars up to the current cursor."""

    def __init__(self, symbol: str, bars_m5: List[Dict[str, float]], bars_h1: List[Dict[str, float]]) -> None:
        self.symbol = symbol
        self.bars_m5 = bars_m5
        self.bars_h1 = bars_h1
        self.cursor = len(bars_m5) - 1

    def set_cursor(self, index: int) -> None:
        self.cursor = max(0, min(index, len(self.bars_m5) - 1))

    def get_bars(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, float]]:
        if symbol != self.symbol:
            raise ValueError(f"unsupported symbol {symbol}")
        tf = timeframe.upper()
        if tf in {"M5", "5M"}:
            data = self.bars_m5[: self.cursor + 1]
        elif tf in {"H1", "1H"}:
            current_end = self.bars_m5[self.cursor]["end"]
            idx = -1
            for pos, bar in enumerate(self.bars_h1):
                if bar["end"] <= current_end + 1e-6:
                    idx = pos
                else:
                    break
            if idx < 0:
                return []
            data = self.bars_h1[: idx + 1]
        else:
            raise ValueError(f"unsupported timeframe {timeframe}")
        if count <= 0 or count >= len(data):
            return list(data)
        return data[-count:]

    def last(self, symbol: str) -> float:
        if symbol != self.symbol:
            raise ValueError(f"unsupported symbol {symbol}")
        return self.bars_m5[self.cursor]["close"]

def _simulate_trade(
    bars: List[Dict[str, float]],
    start_idx: int,
    intent,
    exit_cfg: Dict[str, object],
    max_hold_bars: int = 48,
) -> Dict[str, object]:
    entry_px = float(intent.entry_px or bars[start_idx]["close"])
    atr = float(intent.meta.get("atr") or 0.0)
    min_atr = float(exit_cfg.get("min_atr_price", 0.003))
    stop_mult = float(exit_cfg.get("stop_atr", 1.8))
    tp_mult = float(exit_cfg.get("tp_atr", 3.0))
    atr_price = max(atr, min_atr)

    if intent.side == "long":
        stop_price = entry_px - atr_price * stop_mult
        take_price = entry_px + atr_price * tp_mult
    else:
        stop_price = entry_px + atr_price * stop_mult
        take_price = entry_px - atr_price * tp_mult

    exit_idx = start_idx
    exit_reason = "timeout"
    exit_price = entry_px

    for ahead, idx in enumerate(range(start_idx + 1, len(bars))):
        bar = bars[idx]
        high, low = bar["high"], bar["low"]
        hit_stop = low <= stop_price if intent.side == "long" else high >= stop_price
        hit_take = high >= take_price if intent.side == "long" else low <= take_price

        if hit_stop and hit_take:
            dist_stop = abs(entry_px - stop_price)
            dist_take = abs(entry_px - take_price)
            choose_stop = dist_stop <= dist_take
            if choose_stop:
                exit_reason = "stop"
                exit_price = stop_price
            else:
                exit_reason = "take_profit"
                exit_price = take_price
            exit_idx = idx
            break
        if hit_stop:
            exit_reason = "stop"
            exit_price = stop_price
            exit_idx = idx
            break
        if hit_take:
            exit_reason = "take_profit"
            exit_price = take_price
            exit_idx = idx
            break
        exit_price = bar["close"]
        exit_idx = idx
        if ahead + 1 >= max_hold_bars:
            exit_reason = "timeout"
            break
    else:
        exit_price = bars[-1]["close"]
        exit_idx = len(bars) - 1
        exit_reason = "timeout"

    pnl_pips = (exit_price - entry_px) / PIP_SIZE if intent.side == "long" else (entry_px - exit_price) / PIP_SIZE

    exit_dt = datetime.fromtimestamp(bars[exit_idx]["end"], tz=timezone.utc)
    return {
        "exit_time": exit_dt.isoformat(),
        "exit_px": round(exit_price, 5),
        "exit_reason": exit_reason,
        "hold_bars": max(0, exit_idx - start_idx),
        "pnl_pips": round(pnl_pips, 3),
    }

def replay_worker(
    ticks: Sequence[Tick],
    symbol: str,
    overrides: Dict[str, object],
) -> Dict[str, object]:
    bars_m5 = build_bars(ticks, seconds=300)
    bars_h1 = build_bars(ticks, seconds=3600)
    if len(bars_m5) < 20 or len(bars_h1) < 5:
        return {"summary": {"signals": 0, "reason": "insufficient_bars"}}

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)
    cfg["universe"] = [symbol]
    cfg["place_orders"] = False

    datafeed = SliceDataFeed(symbol, bars_m5, bars_h1)
    worker = MtfBreakoutWorker(cfg, broker=None, datafeed=datafeed)

    trades: List[Dict[str, object]] = []
    counts = defaultdict(int)
    totals = {"pips": 0.0, "wins": 0, "losses": 0}
    exit_cfg = dict(cfg.get("exit") or {})

    for idx in range(len(bars_m5)):
        datafeed.set_cursor(idx)
        intent = worker.edge(symbol)
        if not intent:
            continue
        dt = datetime.fromtimestamp(bars_m5[idx]["end"], tz=timezone.utc)
        counts[intent.side] += 1
        trade = {
            "time": dt.isoformat(),
            "side": intent.side,
            "strength": round(intent.strength, 3),
            "entry_px": round(float(intent.entry_px or bars_m5[idx]["close"]), 5),
            "meta": intent.meta,
        }
        max_hold = int(exit_cfg.get("max_hold_bars", 48))
        filled = _simulate_trade(bars_m5, idx, intent, exit_cfg, max_hold_bars=max_hold)
        trade.update(filled)
        trades.append(trade)
        totals["pips"] += filled["pnl_pips"]
        if filled["pnl_pips"] > 0:
            totals["wins"] += 1
        elif filled["pnl_pips"] < 0:
            totals["losses"] += 1

    summary = {
        "signals": len(trades),
        "long_signals": counts["long"],
        "short_signals": counts["short"],
    }
    if trades:
        summary["first_signal"] = trades[0]["time"]
        summary["last_signal"] = trades[-1]["time"]
        summary["profit_pips"] = round(totals["pips"], 3)
        summary["win_rate"] = round(
            (totals["wins"] / len(trades)) * 100.0, 1
        ) if trades else 0.0
        summary["avg_pnl_pips"] = round(
            totals["pips"] / len(trades), 3
        )

    return {"summary": summary, "trades": trades}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the MTF breakout worker against recorded ticks.")
    parser.add_argument("--ticks", required=True, help="Path to tick JSONL file (timestamp/bid/ask).")
    parser.add_argument("--symbol", default="USD_JPY", help="Instrument symbol; default USD_JPY.")
    parser.add_argument("--out", help="Optional path to save the replay result JSON.")
    parser.add_argument("--lookback", type=int, help="Override breakout lookback.")
    parser.add_argument("--pullback", type=int, help="Override pullback_min_bars.")
    parser.add_argument("--cooldown", type=int, help="Override cooldown_bars.")
    parser.add_argument("--edge", type=float, help="Override edge_threshold.")
    parser.add_argument("--budget", type=float, help="Override budget_bps.")
    parser.add_argument("--trend-min", dest="trend_min", type=int, help="Override minimum H1 bars required before signalling.")
    parser.add_argument("--stop-atr", dest="stop_atr", type=float, help="Override stop ATR multiplier.")
    parser.add_argument("--tp-atr", dest="tp_atr", type=float, help="Override take-profit ATR multiplier.")
    parser.add_argument("--max-hold", dest="max_hold", type=int, help="Override maximum hold bars before timeout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ticks_path = Path(args.ticks)
    if not ticks_path.exists():
        raise SystemExit(f"ticks file not found: {ticks_path}")
    ticks = load_ticks(ticks_path)
    overrides: Dict[str, object] = {}
    if args.lookback is not None:
        overrides["breakout_lookback"] = max(5, args.lookback)
    if args.pullback is not None:
        overrides["pullback_min_bars"] = max(0, args.pullback)
    if args.cooldown is not None:
        overrides["cooldown_bars"] = max(0, args.cooldown)
    if args.edge is not None:
        overrides["edge_threshold"] = max(0.0, min(1.0, args.edge))
    if args.budget is not None:
        overrides["budget_bps"] = max(0.0, args.budget)
    if args.trend_min is not None:
        overrides["trend_min_bars"] = max(10, args.trend_min)
    if args.stop_atr is not None or args.tp_atr is not None or args.max_hold is not None:
        exit_cfg: Dict[str, object] = dict(DEFAULT_CONFIG.get("exit", {}))
        if args.stop_atr is not None:
            exit_cfg["stop_atr"] = max(0.1, args.stop_atr)
        if args.tp_atr is not None:
            exit_cfg["tp_atr"] = max(0.1, args.tp_atr)
        if args.max_hold is not None:
            exit_cfg["max_hold_bars"] = max(1, args.max_hold)
        overrides["exit"] = exit_cfg

    result = replay_worker(ticks, args.symbol, overrides)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
