#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastScalp worker backtest (synthetic tick replay based on M1 candles).

Usage:
  python scripts/backtest_fast_scalp.py \
      --candles logs/candles_M1_20251001_20251022.json \
      --spread-pips 0.3

The script approximates intra-minute ticks by interpolating each candle's
open/high/low/close path, runs the FastScalp signal logic, and reports
aggregate performance using 10k-unit entries (TP ≈ 1 pip, SL 30 pips).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Deque, Iterable, List, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workers.fast_scalp import config
from workers.fast_scalp.signal import SignalFeatures, evaluate_signal

PIP_VALUE = config.PIP_VALUE


def parse_time(ts: str) -> datetime:
    ts = ts.rstrip("Z")
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000")[:6]
        ts = f"{head}.{frac}+00:00"
    else:
        ts = ts + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class Tick:
    epoch: float
    mid: float


@dataclass
class Trade:
    direction: str  # "long" or "short"
    entry_price: float
    entry_epoch: float
    entry_time: datetime
    tp_price: float
    sl_price: float
    units: int


def load_candles(path: Path) -> List[Candle]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candles = payload.get("candles", payload)
    out: List[Candle] = []
    for c in candles:
        mid = c.get("mid") or {}
        out.append(
            Candle(
                time=parse_time(c["time"]),
                open=float(mid.get("o", mid.get("open", 0.0))),
                high=float(mid.get("h", mid.get("high", 0.0))),
                low=float(mid.get("l", mid.get("low", 0.0))),
                close=float(mid.get("c", mid.get("close", 0.0))),
            )
        )
    out.sort(key=lambda x: x.time)
    return out


def interpolate_path(prev_close: float, candle: Candle, steps: int = 60) -> List[Tick]:
    """Generate synthetic ticks within the candle minute."""
    if steps < 4:
        steps = 4
    values = []
    for idx in range(steps):
        frac = (idx + 1) / steps
        base = prev_close + (candle.close - prev_close) * frac
        values.append(base)
    high_idx = 15 if candle.close >= candle.open else 45
    low_idx = 42 if candle.close >= candle.open else 18
    high_idx = max(1, min(steps - 2, high_idx))
    low_idx = max(1, min(steps - 2, low_idx))
    values[1] = candle.open
    values[high_idx] = candle.high
    values[high_idx + 1] = (values[high_idx] + candle.close) / 2
    values[low_idx] = candle.low
    values[low_idx + 1] = (values[low_idx] + candle.close) / 2
    ticks: List[Tick] = []
    base_epoch = candle.time.timestamp()
    for idx, price in enumerate(values):
        epoch = base_epoch + idx
        ticks.append(Tick(epoch=epoch, mid=float(price)))
    return ticks


class FastScalpBacktester:
    def __init__(
        self,
        *,
        spread_pips: float,
        min_units: int,
        max_units: int,
    ) -> None:
        self.spread_pips = spread_pips
        self.min_units = min_units
        self.max_units = max_units
        self.ticks: Deque[Tick] = deque()
        self.window_seconds = config.LONG_WINDOW_SEC
        self.trade: Optional[Trade] = None
        self.trades: List[dict] = []

    def _update_window(self, tick: Tick) -> None:
        self.ticks.append(tick)
        cutoff = tick.epoch - self.window_seconds
        while self.ticks and self.ticks[0].epoch < cutoff:
            self.ticks.popleft()

    def _features(self) -> Optional[SignalFeatures]:
        if len(self.ticks) < config.MIN_TICK_COUNT:
            return None
        mids = [t.mid for t in self.ticks]
        latest = mids[-1]
        long_mean = mean(mids)
        short_len = max(5, int(len(mids) * config.SHORT_WINDOW_SEC / config.LONG_WINDOW_SEC))
        short_slice = mids[-short_len:]
        short_mean = mean(short_slice)
        high_mid = max(mids)
        low_mid = min(mids)
        span_seconds = self.ticks[-1].epoch - self.ticks[0].epoch
        momentum = (latest - long_mean) / PIP_VALUE
        short_momentum = (latest - short_mean) / PIP_VALUE
        range_pips = (high_mid - low_mid) / PIP_VALUE
        return SignalFeatures(
            latest_mid=latest,
            spread_pips=self.spread_pips,
            momentum_pips=momentum,
            short_momentum_pips=short_momentum,
            range_pips=range_pips,
            tick_count=len(mids),
            span_seconds=span_seconds,
        )

    def _close_trade(self, tick: Tick, reason: str) -> None:
        if not self.trade:
            return
        entry = self.trade.entry_price
        exit_price = tick.mid
        if self.trade.direction == "long":
            pnl_pips = (exit_price - entry) / PIP_VALUE
        else:
            pnl_pips = (entry - exit_price) / PIP_VALUE
        pnl_jpy = pnl_pips * self.trade.units * PIP_VALUE
        self.trades.append(
            {
                "entry_time": self.trade.entry_time.isoformat(),
                "exit_time": datetime.fromtimestamp(tick.epoch, tz=timezone.utc).isoformat(),
                "direction": self.trade.direction,
                "entry_price": round(entry, 5),
                "exit_price": round(exit_price, 5),
                "tp_price": round(self.trade.tp_price, 5),
                "sl_price": round(self.trade.sl_price, 5),
                "units": self.trade.units,
                "pnl_pips": round(pnl_pips, 3),
                "pnl_jpy": round(pnl_jpy, 2),
                "reason": reason,
            }
        )
        self.trade = None

    def _ensure_tp_sl(self, entry_price: float, direction: str) -> tuple[float, float]:
        tp_pips = config.TP_BASE_PIPS + max(self.spread_pips, config.TP_SPREAD_BUFFER_PIPS)
        sl_pips = config.SL_PIPS
        if direction == "long":
            tp_price = entry_price + tp_pips * PIP_VALUE
            sl_price = entry_price - sl_pips * PIP_VALUE
        else:
            tp_price = entry_price - tp_pips * PIP_VALUE
            sl_price = entry_price + sl_pips * PIP_VALUE
        return round(tp_price, 5), round(sl_price, 5)

    def _maybe_open(self, tick: Tick, signal: str) -> None:
        direction = "long" if signal == "OPEN_LONG" else "short"
        entry_price = tick.mid
        tp_price, sl_price = self._ensure_tp_sl(entry_price, direction)
        units = max(self.min_units, min(self.max_units, config.MIN_UNITS))
        entry_time = datetime.fromtimestamp(tick.epoch, tz=timezone.utc)
        self.trade = Trade(
            direction=direction,
            entry_price=entry_price,
            entry_epoch=tick.epoch,
            entry_time=entry_time,
            tp_price=tp_price,
            sl_price=sl_price,
            units=units,
        )

    def _check_open_trade(self, tick: Tick) -> None:
        if not self.trade:
            return
        price = tick.mid
        direction = self.trade.direction
        if direction == "long":
            gain_pips = (price - self.trade.entry_price) / PIP_VALUE
            if price >= self.trade.tp_price:
                self._close_trade(tick, "tp_hit")
                return
            if price <= self.trade.sl_price:
                self._close_trade(tick, "sl_hit")
                return
        else:
            gain_pips = (self.trade.entry_price - price) / PIP_VALUE
            if price <= self.trade.tp_price:
                self._close_trade(tick, "tp_hit")
                return
            if price >= self.trade.sl_price:
                self._close_trade(tick, "sl_hit")
                return

        elapsed = tick.epoch - self.trade.entry_epoch
        if gain_pips <= -config.MAX_DRAWDOWN_CLOSE_PIPS:
            self._close_trade(tick, "drawdown_stop")
            return
        if elapsed >= config.TIMEOUT_SEC and gain_pips < config.TIMEOUT_MIN_GAIN_PIPS:
            self._close_trade(tick, "timeout_close")

    def run(self, ticks: Iterable[Tick]) -> None:
        for tick in ticks:
            self._update_window(tick)
            self._check_open_trade(tick)
            if self.trade:
                continue
            features = self._features()
            if not features:
                continue
            signal = evaluate_signal(features)
            if signal:
                self._maybe_open(tick, signal)

    def summary(self) -> dict:
        total_pips = sum(t["pnl_pips"] for t in self.trades)
        total_jpy = sum(t["pnl_jpy"] for t in self.trades)
        wins = [t for t in self.trades if t["pnl_pips"] > 0]
        losses = [t for t in self.trades if t["pnl_pips"] < 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        gross_win = sum(t["pnl_pips"] for t in wins)
        gross_loss = abs(sum(t["pnl_pips"] for t in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else math.inf
        return {
            "trades": len(self.trades),
            "total_pnl_pips": round(total_pips, 3),
            "total_pnl_jpy": round(total_jpy, 2),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 3) if math.isfinite(profit_factor) else float("inf"),
        }


def generate_ticks(candles: List[Candle], *, spread_pips: float) -> List[Tick]:
    ticks: List[Tick] = []
    prev_close = candles[0].open if candles else 0.0
    for candle in candles:
        ticks.extend(interpolate_path(prev_close, candle, steps=60))
        prev_close = candle.close
    return ticks


def main() -> None:
    parser = argparse.ArgumentParser(description="FastScalp synthetic backtest")
    parser.add_argument("--candles", type=Path, required=True, help="Path to M1 candle JSON")
    parser.add_argument("--spread-pips", type=float, default=0.3, help="Assumed spread in pips (default 0.3)")
    parser.add_argument("--min-units", type=int, default=10000, help="Minimum trade units (default 10000)")
    parser.add_argument("--max-units", type=int, default=10000, help="Maximum trade units (default 10000)")
    parser.add_argument("--json-out", type=Path, help="Optional path to write JSON results")
    args = parser.parse_args()

    candles = load_candles(args.candles)
    ticks = generate_ticks(candles, spread_pips=args.spread_pips)
    backtester = FastScalpBacktester(
        spread_pips=args.spread_pips,
        min_units=args.min_units,
        max_units=args.max_units,
    )
    backtester.run(ticks)
    summary = backtester.summary()
    print("----- FastScalp Backtest Summary -----")
    print(f"candles: {args.candles}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.json_out:
        payload = {"summary": summary, "trades": backtester.trades}
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved details -> {args.json_out}")


if __name__ == "__main__":
    main()
