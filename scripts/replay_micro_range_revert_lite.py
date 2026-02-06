#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tick replay for micro_range_revert_lite (entry + exit)."""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from indicators.calc_core import IndicatorEngine
from analysis.range_guard import detect_range_mode
from strategies.micro_lowvol.range_revert_lite import RangeRevertLite
from execution.risk_guard import clamp_sl_tp
from workers.micro_range_revert_lite import exit_worker as rrl_exit

PIP = 0.01
M1_WINDOW = 160
H4_WINDOW = 160


@dataclass
class Tick:
    epoch: float
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, tz=timezone.utc)


@dataclass
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class Trade:
    trade_id: str
    side: str
    entry_time: datetime
    entry_price: float
    units: int
    sl_price: Optional[float]
    tp_price: Optional[float]
    tag: str
    remaining_units: int
    max_pnl: float = 0.0
    partial_done: bool = False


class CandleBuilder:
    def __init__(self, minutes: int) -> None:
        self.minutes = minutes
        self.bucket_start: Optional[datetime] = None
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = 0.0
        self.close: float = 0.0

    def update(self, ts: datetime, price: float) -> Optional[Candle]:
        bucket = ts.replace(second=0, microsecond=0)
        if self.minutes > 1:
            minute = (bucket.minute // self.minutes) * self.minutes
            bucket = bucket.replace(minute=minute)
        if self.bucket_start is None:
            self.bucket_start = bucket
            self.open = price
            self.high = price
            self.low = price
            self.close = price
            return None
        if bucket != self.bucket_start:
            candle = Candle(
                time=self.bucket_start,
                open=self.open,
                high=self.high,
                low=self.low,
                close=self.close,
            )
            self.bucket_start = bucket
            self.open = price
            self.high = price
            self.low = price
            self.close = price
            return candle
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        return None


@dataclass
class ReplayResult:
    summary: Dict
    trades: List[Dict]


def iter_ticks(path: Path) -> Iterable[Tick]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            ts_raw = payload.get("timestamp") or payload.get("ts") or payload.get("time")
            if ts_raw is None:
                continue
            if isinstance(ts_raw, (int, float)):
                epoch = float(ts_raw)
                if epoch > 10**11:
                    epoch /= 1000.0
            else:
                epoch = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
            bid = float(payload.get("bid", 0.0))
            ask = float(payload.get("ask", 0.0))
            yield Tick(epoch=epoch, bid=bid, ask=ask)


def _calc_sl_tp(side: str, price: float, sl_pips: float, tp_pips: float) -> tuple[float, float]:
    if side == "long":
        sl_price = price - sl_pips * PIP
        tp_price = price + tp_pips * PIP
    else:
        sl_price = price + sl_pips * PIP
        tp_price = price - tp_pips * PIP
    return sl_price, tp_price


def _close_price(side: str, tick: Tick) -> float:
    return tick.bid if side == "long" else tick.ask


def _trade_pips(entry: float, exit_price: float, side: str) -> float:
    if side == "long":
        return (exit_price - entry) / PIP
    return (entry - exit_price) / PIP


def _build_factors(candles: Deque[Candle]) -> Dict[str, float]:
    df = pd.DataFrame([{"open": c.open, "high": c.high, "low": c.low, "close": c.close} for c in candles])
    factors = IndicatorEngine.compute(df)
    last = candles[-1]
    tail = list(candles)[-60:]
    factors.update(
        {
            "open": last.open,
            "high": last.high,
            "low": last.low,
            "close": last.close,
            "timestamp": last.time.isoformat(),
            "candles": [
                {
                    "timestamp": c.time.isoformat(),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                }
                for c in tail
            ],
        }
    )
    return factors


def replay(
    *,
    ticks: Iterable[Tick],
    units: int,
    no_hard_sl: bool,
    no_hard_tp: bool,
    exclude_end_of_replay: bool,
) -> ReplayResult:
    m1_builder = CandleBuilder(1)
    h4_builder = CandleBuilder(240)
    m1_candles: Deque[Candle] = deque(maxlen=M1_WINDOW)
    h4_candles: Deque[Candle] = deque(maxlen=H4_WINDOW)
    fac_m1: Dict[str, float] = {}
    fac_h4: Dict[str, float] = {}
    open_trades: List[Trade] = []
    trade_records: List[Dict] = []
    trade_id_seq = 0
    last_exit_eval = 0.0
    last_tick: Optional[Tick] = None

    def _record_close(trade: Trade, close_units: int, exit_price: float, ts: datetime, reason: str) -> None:
        pl_pips = _trade_pips(trade.entry_price, exit_price, trade.side)
        trade_records.append(
            {
                "trade_id": trade.trade_id,
                "entry_time": trade.entry_time.isoformat(),
                "exit_time": ts.isoformat(),
                "direction": trade.side,
                "entry_price": trade.entry_price,
                "exit_price": exit_price,
                "units": close_units,
                "pl_pips": pl_pips,
                "pl_jpy": round(pl_pips * abs(close_units) * PIP, 2),
                "reason": reason,
                "tag": trade.tag,
            }
        )

    for tick in ticks:
        last_tick = tick
        ts = tick.dt

        # Update candle builders
        m1_candle = m1_builder.update(ts, tick.mid)
        h4_candle = h4_builder.update(ts, tick.mid)

        if h4_candle:
            h4_candles.append(h4_candle)
            if len(h4_candles) >= 20:
                fac_h4 = _build_factors(h4_candles)

        if m1_candle:
            m1_candles.append(m1_candle)
            if len(m1_candles) >= 20:
                fac_m1 = _build_factors(m1_candles)
                range_ctx = detect_range_mode(fac_m1, fac_h4)
                fac_m1 = dict(fac_m1)
                fac_m1["range_active"] = bool(range_ctx.active)
                fac_m1["range_score"] = float(range_ctx.score or 0.0)

                signal = RangeRevertLite.check(fac_m1)
                if signal:
                    trade_id_seq += 1
                    side = "long" if signal.get("action") == "OPEN_LONG" else "short"
                    sl_pips = float(signal.get("sl_pips") or 0.0)
                    tp_pips = float(signal.get("tp_pips") or 0.0)
                    if sl_pips > 0 and tp_pips > 0:
                        sl_price, tp_price = _calc_sl_tp(side, m1_candle.close, sl_pips, tp_pips)
                        sl_price, tp_price = clamp_sl_tp(m1_candle.close, sl_price, tp_price, side == "long")
                        tag = signal.get("tag", RangeRevertLite.name)
                        trade = Trade(
                            trade_id=f"sim-{trade_id_seq}",
                            side=side,
                            entry_time=m1_candle.time,
                            entry_price=m1_candle.close,
                            units=units if side == "long" else -units,
                            sl_price=sl_price,
                            tp_price=tp_price,
                            tag=tag,
                            remaining_units=units if side == "long" else -units,
                        )
                        open_trades.append(trade)

        # Hard TP/SL handling
        if open_trades:
            for trade in list(open_trades):
                if trade.remaining_units == 0:
                    open_trades.remove(trade)
                    continue
                exit_price = _close_price(trade.side, tick)
                if not no_hard_tp and trade.tp_price is not None:
                    if trade.side == "long" and exit_price >= trade.tp_price:
                        _record_close(trade, trade.remaining_units, trade.tp_price, ts, "hard_tp")
                        trade.remaining_units = 0
                        open_trades.remove(trade)
                        continue
                    if trade.side == "short" and exit_price <= trade.tp_price:
                        _record_close(trade, trade.remaining_units, trade.tp_price, ts, "hard_tp")
                        trade.remaining_units = 0
                        open_trades.remove(trade)
                        continue
                if not no_hard_sl and trade.sl_price is not None:
                    if trade.side == "long" and exit_price <= trade.sl_price:
                        _record_close(trade, trade.remaining_units, trade.sl_price, ts, "hard_sl")
                        trade.remaining_units = 0
                        open_trades.remove(trade)
                        continue
                    if trade.side == "short" and exit_price >= trade.sl_price:
                        _record_close(trade, trade.remaining_units, trade.sl_price, ts, "hard_sl")
                        trade.remaining_units = 0
                        open_trades.remove(trade)
                        continue

        # Exit worker logic (evaluate every ~1s)
        if open_trades and fac_m1 and (tick.epoch - last_exit_eval) >= 1.0:
            last_exit_eval = tick.epoch
            range_active = False
            try:
                range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
            except Exception:
                range_active = False
            for trade in list(open_trades):
                trade_dict = {
                    "trade_id": trade.trade_id,
                    "units": trade.remaining_units,
                    "price": trade.entry_price,
                    "open_time": trade.entry_time.isoformat(),
                    "client_order_id": trade.trade_id,
                    "entry_thesis": {"strategy_tag": RangeRevertLite.name},
                    "strategy_tag": RangeRevertLite.name,
                }
                state = rrl_exit._TradeState(max_pnl=trade.max_pnl, partial_done=trade.partial_done)
                decision = rrl_exit.evaluate_exit(
                    trade_dict,
                    now=ts,
                    mid=tick.mid,
                    range_active=range_active,
                    fac=fac_m1,
                    state=state,
                )
                trade.max_pnl = state.max_pnl
                trade.partial_done = state.partial_done
                if decision is None:
                    continue
                close_units, reason = decision
                if close_units == 0:
                    continue
                close_units = int(close_units)
                if abs(close_units) >= abs(trade.remaining_units):
                    _record_close(trade, trade.remaining_units, _close_price(trade.side, tick), ts, reason)
                    trade.remaining_units = 0
                    open_trades.remove(trade)
                else:
                    _record_close(trade, close_units, _close_price(trade.side, tick), ts, reason)
                    trade.remaining_units -= close_units

    if not exclude_end_of_replay and last_tick is not None:
        for trade in list(open_trades):
            exit_price = _close_price(trade.side, last_tick)
            _record_close(trade, trade.remaining_units, exit_price, last_tick.dt, "end_of_replay")
            trade.remaining_units = 0
            open_trades.remove(trade)

    wins = [t for t in trade_records if t["pl_pips"] > 0]
    losses = [t for t in trade_records if t["pl_pips"] <= 0]
    sum_pips = sum(t["pl_pips"] for t in trade_records)
    sum_jpy = sum(t["pl_jpy"] for t in trade_records)
    win_rate = (len(wins) / len(trade_records) * 100.0) if trade_records else 0.0
    gain = sum(t["pl_pips"] for t in wins)
    loss = abs(sum(t["pl_pips"] for t in losses))
    pf = (gain / loss) if loss > 0 else None

    summary = {
        "total_trades": len(trade_records),
        "win_rate": round(win_rate, 2),
        "sum_pips": round(sum_pips, 2),
        "sum_jpy": round(sum_jpy, 2),
        "profit_factor": round(pf, 3) if pf is not None else None,
    }
    return ReplayResult(summary=summary, trades=trade_records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", type=int, default=10000)
    ap.add_argument("--no-hard-sl", action="store_true")
    ap.add_argument("--no-hard-tp", action="store_true")
    ap.add_argument("--exclude-end-of-replay", action="store_true")
    args = ap.parse_args()

    ticks_path = Path(args.ticks)
    out_path = Path(args.out)
    result = replay(
        ticks=iter_ticks(ticks_path),
        units=args.units,
        no_hard_sl=bool(args.no_hard_sl),
        no_hard_tp=bool(args.no_hard_tp),
        exclude_end_of_replay=bool(args.exclude_end_of_replay),
    )
    payload = {"summary": result.summary, "trades": result.trades}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
