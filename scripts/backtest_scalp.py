#!/usr/bin/env python3
"""
Quick-and-dirty M1 candle replay for scalp strategies.
Loads a candle JSON dump (logs/candles_*.json), reconstructs key indicators,
runs the registered scalp strategies, and reports aggregated performance metrics.

Example:
  python scripts/backtest_scalp.py --candles logs/candles_M1_20251022.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.scalping.m1_scalper import M1Scalper
from strategies.scalping.range_fader import RangeFader
from strategies.scalping.pulse_break import PulseBreak


PIP_VALUE = 0.01
MAX_HOLD_MINUTES = 30
SCALP_STRATEGIES = [M1Scalper, RangeFader, PulseBreak]


def parse_time(ts: str) -> datetime:
    # Truncate nanoseconds to microseconds so fromisoformat can parse.
    ts = ts.rstrip("Z")
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000")[:6]
        ts = f"{head}.{frac}+00:00"
    else:
        ts = ts + "+00:00"
    return datetime.fromisoformat(ts)


def load_candles(path: Path) -> pd.DataFrame:
    with path.open() as f:
        payload = json.load(f)

    candles = payload.get("candles", payload)
    rows = []
    for c in candles:
        mid = c.get("mid") or {}
        rows.append(
            {
                "time": parse_time(c["time"]),
                "open": float(mid.get("o", mid.get("open", 0.0))),
                "high": float(mid.get("h", mid.get("high", 0.0))),
                "low": float(mid.get("l", mid.get("low", 0.0))),
                "close": float(mid.get("c", mid.get("close", 0.0))),
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    prices = df["close"]
    df["ema20"] = prices.ewm(span=20, adjust=False, min_periods=20).mean()
    df["ema50"] = prices.ewm(span=50, adjust=False, min_periods=50).mean()
    df["ma20"] = prices.rolling(window=20, min_periods=20).mean()

    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"].fillna(50.0, inplace=True)

    prev_close = prices.shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().fillna(0.0)
    df["atr_pips"] = df["atr"] / PIP_VALUE

    pip_move = prices.diff().abs() / PIP_VALUE
    df["vol_5m"] = pip_move.rolling(window=5, min_periods=5).mean().fillna(0.0)

    return df


@dataclass
class Trade:
    strategy: str
    side: str
    entry_index: int
    entry_time: datetime
    entry_price: float
    tp_price: float
    sl_price: float
    outcome: Optional[str] = None
    exit_index: Optional[int] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0


def simulate(df: pd.DataFrame) -> Dict[str, List[Trade]]:
    open_trades: Dict[str, List[Trade]] = defaultdict(list)
    closed_trades: Dict[str, List[Trade]] = defaultdict(list)
    fac_keys = [
        "close",
        "ema20",
        "ema50",
        "ma20",
        "rsi",
        "atr",
        "atr_pips",
        "vol_5m",
    ]

    for idx, row in df.iterrows():
        # First evaluate open trades (skip bar of entry)
        for strat_name, trades in list(open_trades.items()):
            remaining = []
            for trd in trades:
                if idx <= trd.entry_index:
                    remaining.append(trd)
                    continue
                candle_high = float(row["high"])
                candle_low = float(row["low"])
                exit_price = None
                exit_reason = None

                if trd.side == "LONG":
                    if candle_high >= trd.tp_price:
                        exit_price = trd.tp_price
                        exit_reason = "TP"
                    elif candle_low <= trd.sl_price:
                        exit_price = trd.sl_price
                        exit_reason = "SL"
                else:
                    if candle_low <= trd.tp_price:
                        exit_price = trd.tp_price
                        exit_reason = "TP"
                    elif candle_high >= trd.sl_price:
                        exit_price = trd.sl_price
                        exit_reason = "SL"

                if exit_price is None:
                    if (row["time"] - trd.entry_time) >= timedelta(minutes=MAX_HOLD_MINUTES):
                        exit_price = float(row["close"])
                        exit_reason = "TIME"

                if exit_price is None:
                    remaining.append(trd)
                    continue

                trd.exit_index = idx
                trd.exit_time = row["time"]
                trd.exit_price = exit_price
                trd.outcome = exit_reason
                direction = 1 if trd.side == "LONG" else -1
                trd.pnl_pips = (exit_price - trd.entry_price) * direction / PIP_VALUE
                closed_trades[strat_name].append(trd)
            open_trades[strat_name] = remaining

        # Build factor dict for current bar
        fac = {k: float(row[k]) for k in fac_keys if pd.notna(row[k])}
        fac["atr_pips"] = float(row["atr_pips"])
        fac["vol_5m"] = float(row.get("vol_5m", 0.0))

        for cls in SCALP_STRATEGIES:
            strat_name = cls.name
            signal = cls.check(fac)
            if not signal:
                continue
            if open_trades[strat_name]:
                # Simple rule: only one simultaneous trade per strategy.
                continue

            action = signal.get("action")
            if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                continue
            sl = float(signal.get("sl_pips") or 0.0)
            tp = float(signal.get("tp_pips") or 0.0)
            if sl <= 0 or tp <= 0:
                continue

            entry_price = float(row["close"])
            if action == "OPEN_LONG":
                sl_price = entry_price - sl * PIP_VALUE
                tp_price = entry_price + tp * PIP_VALUE
                side = "LONG"
            else:
                sl_price = entry_price + sl * PIP_VALUE
                tp_price = entry_price - tp * PIP_VALUE
                side = "SHORT"

            trade = Trade(
                strategy=strat_name,
                side=side,
                entry_index=idx,
                entry_time=row["time"],
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
            )
            open_trades[strat_name].append(trade)

    # Force close any leftover trades at final close
    final_time = df.iloc[-1]["time"]
    final_close = float(df.iloc[-1]["close"])
    for strat, trades in open_trades.items():
        for trd in trades:
            trd.exit_index = len(df) - 1
            trd.exit_time = final_time
            trd.exit_price = final_close
            trd.outcome = "EOD"
            direction = 1 if trd.side == "LONG" else -1
            trd.pnl_pips = (final_close - trd.entry_price) * direction / PIP_VALUE
            closed_trades[strat].append(trd)

    return closed_trades


def summarise(trades: Dict[str, List[Trade]]) -> None:
    for cls in SCALP_STRATEGIES:
        strat = cls.name
        items = trades.get(strat, [])
        if not items:
            print(f"{strat}: no trades")
            continue

        pnl = [tr.pnl_pips for tr in items]
        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p <= 0]
        win_rate = (len(wins) / len(pnl)) * 100 if pnl else 0.0
        avg = np.mean(pnl) if pnl else 0.0
        med = np.median(pnl) if pnl else 0.0
        total = np.sum(pnl)

        print(f"--- {strat} ---")
        print(f"trades={len(pnl)} win_rate={win_rate:.1f}% total={total:.1f} pips")
        print(f"avg={avg:.2f} median={med:.2f} best={max(pnl):.2f} worst={min(pnl):.2f}")
        outcome_counts = defaultdict(int)
        for tr in items:
            outcome_counts[tr.outcome] += 1
        print("outcomes:", dict(outcome_counts))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay scalp strategies on candle data.")
    parser.add_argument(
        "--candles",
        type=Path,
        required=True,
        help="Path to candle JSON (e.g. logs/candles_M1_*.json)",
    )
    args = parser.parse_args()

    df = load_candles(args.candles)
    df = compute_indicators(df)
    trades = simulate(df)
    summarise(trades)


if __name__ == "__main__":
    main()
