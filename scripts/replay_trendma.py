#!/usr/bin/env python3
"""
Replay TrendMA (macro) strategy on recorded USD/JPY tick files.

Reads JSONL tick logs (as produced by tmp/ticks_USDJPY_*.jsonl),
builds M1/H4 candles, runs the MovingAverageCross strategy with
partial reductions and exit manager, and emits aggregate metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.exit_manager import ExitManager
from execution.order_manager import plan_partial_reductions, _PARTIAL_STAGE
from strategies.trend.ma_cross import MovingAverageCross


UTC = timezone.utc


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
        return datetime.fromtimestamp(self.epoch, tz=UTC)


def load_ticks(path: Path) -> List[Tick]:
    ticks: List[Tick] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            ts_raw = payload.get("timestamp") or payload.get("ts")
            if ts_raw is None:
                continue
            if isinstance(ts_raw, (int, float)):
                epoch = float(ts_raw)
                if epoch > 10**11:  # assume milliseconds
                    epoch /= 1000.0
            else:
                ts_text = str(ts_raw).replace("Z", "+00:00")
                epoch = datetime.fromisoformat(ts_text).timestamp()
            bid = float(payload.get("bid", 0.0))
            ask = float(payload.get("ask", 0.0))
            ticks.append(Tick(epoch=epoch, bid=bid, ask=ask))
    ticks.sort(key=lambda t: t.epoch)
    return ticks


def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def floor_4h(dt: datetime) -> datetime:
    hour = (dt.hour // 4) * 4
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0)


def build_m1_candles(ticks: Sequence[Tick]) -> List[Dict[str, object]]:
    candles: List[Dict[str, object]] = []
    current_key: Optional[datetime] = None
    current: Optional[Dict[str, object]] = None
    for tk in ticks:
        dt = tk.dt
        key = floor_minute(dt)
        price = tk.mid
        if current_key is None or key != current_key:
            if current:
                candles.append(current)
            current = {
                "time": dt,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
            current_key = key
        else:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price
            current["time"] = dt
    if current:
        candles.append(current)
    return candles


class H4Aggregator:
    def __init__(self) -> None:
        self._current_key: Optional[datetime] = None
        self._current: Optional[Dict[str, object]] = None
        self.history: List[Dict[str, object]] = []

    def update(self, candle: Dict[str, object]) -> None:
        dt = candle["time"]
        if not isinstance(dt, datetime):
            raise ValueError("candle['time'] must be datetime")
        key = floor_4h(dt)
        if self._current_key is None:
            self._current_key = key
            self._current = dict(candle)
            return
        if key != self._current_key:
            if self._current:
                self.history.append(dict(self._current))
            self._current_key = key
            self._current = dict(candle)
        else:
            cur = self._current or dict(candle)
            cur["high"] = max(cur["high"], candle["high"])
            cur["low"] = min(cur["low"], candle["low"])
            cur["close"] = candle["close"]
            cur["time"] = candle["time"]
            self._current = cur

    def finalize(self) -> None:
        if self._current:
            self.history.append(dict(self._current))
            self._current = None
            self._current_key = None


def compute_factors(
    candles: List[Dict[str, object]], *, min_bars: int = 20
) -> Optional[Dict[str, object]]:
    if len(candles) < min_bars:
        return None
    df = pd.DataFrame(
        [
            {
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
            }
            for c in candles
        ]
    )
    from indicators.calc_core import IndicatorEngine

    factors = IndicatorEngine.compute(df)
    last = candles[-1]
    factors.update(
        {
            "close": last["close"],
            "open": last["open"],
            "high": last["high"],
            "low": last["low"],
            "timestamp": last["time"],
            "candles": [{"close": c["close"]} for c in candles[-300:]],
        }
    )
    factors["atr_pips"] = factors.get("atr", 0.0) * 100.0
    return factors


def pips_between(entry: float, exit_price: float, side: str) -> float:
    if side == "long":
        return (exit_price - entry) * 100.0
    return (entry - exit_price) * 100.0


STAGE_RATIOS_MACRO = (0.30,)


class TrendMAReplayer:
    BASE_POCKET_LOT = 0.0035  # mirror live macro sizing cap (~0.0035 lot total)
    MIN_UNITS = 80

    def __init__(self) -> None:
        self.exit_manager = ExitManager()
        self.reset()

    def reset(self) -> None:
        self.m1_history: List[Dict[str, object]] = []
        self.h4_history: List[Dict[str, object]] = []
        self.h4_agg = H4Aggregator()
        self.open_positions = {
            "macro": {
                "units": 0,
                "avg_price": 0.0,
                "long_units": 0,
                "long_avg_price": 0.0,
                "short_units": 0,
                "short_avg_price": 0.0,
                "open_trades": [],
            },
            "__net__": {"units": 0},
        }
        self.stage_state = {"long": 0, "short": 0}
        self.trade_seq = 1
        self.closed_trades: List[Dict[str, object]] = []
        _PARTIAL_STAGE.clear()
        self.latest_fac_h4: Optional[Dict[str, object]] = None

    def run_on_ticks(self, ticks: Sequence[Tick]) -> None:
        self.reset()
        m1_candles = build_m1_candles(ticks)
        for candle in m1_candles:
            self._on_m1_candle(candle)
        # close leftover trades at last price
        if self.open_positions["macro"]["open_trades"]:
            last_price = self.m1_history[-1]["close"]
            end_time = self.m1_history[-1]["time"]
            self._close_all(last_price, end_time)
        # finalize h4 aggregator (optional)
        self.h4_agg.finalize()

    def _on_m1_candle(self, candle: Dict[str, object]) -> None:
        price = candle["close"]
        ctime = candle["time"]
        self.m1_history.append(candle)
        self.h4_agg.update(candle)
        if self.h4_agg.history:
            self.h4_history = self.h4_agg.history[-500:]
            self.latest_fac_h4 = compute_factors(self.h4_history, min_bars=5)

        if len(self.m1_history) < 40 or self.latest_fac_h4 is None:
            return
        fac_m1 = compute_factors(self.m1_history[-2000:], min_bars=40)
        if fac_m1 is None:
            return
        fac_h4 = self.latest_fac_h4
        fac_m1["atr_pips"] = fac_m1.get("atr", 0.0) * 100.0
        fac_h4["atr_pips"] = fac_h4.get("atr", 0.0) * 100.0

        self._update_unrealized(price)

        partials = plan_partial_reductions(
            self.open_positions,
            fac_m1,
            fac_h4,
            range_mode=False,
            now=ctime,
        )
        if partials:
            for _, trade_id, reduce_units in partials:
                self._apply_partial(trade_id, reduce_units, price, ctime)

        raw_signal = MovingAverageCross.check(fac_m1)
        signals: List[Dict[str, object]] = []
        if raw_signal:
            signals.append(
                {
                    "strategy": "TrendMA",
                    "pocket": "macro",
                    "action": raw_signal["action"],
                    "confidence": raw_signal.get("confidence", 50),
                    "sl_pips": raw_signal.get("sl_pips"),
                    "tp_pips": raw_signal.get("tp_pips"),
                    "tag": raw_signal.get("tag", "TrendMA"),
                }
            )
            self._maybe_enter(raw_signal, price, ctime, fac_m1)

        exit_decisions = self.exit_manager.plan_closures(
            self.open_positions,
            signals,
            fac_m1,
            fac_h4,
            event_soon=False,
            range_mode=False,
            now=ctime,
        )
        if exit_decisions:
            self._process_exit_decisions(exit_decisions, price, ctime)

    def _maybe_enter(
        self,
        raw_signal: Dict[str, object],
        price: float,
        ts: datetime,
        fac_m1: Dict[str, object],
    ) -> None:
        action = raw_signal.get("action")
        if action not in {"OPEN_LONG", "OPEN_SHORT"}:
            return
        direction = "long" if action == "OPEN_LONG" else "short"
        if direction == "long":
            if self.open_positions["macro"]["short_units"] > 0:
                return
        else:
            if self.open_positions["macro"]["long_units"] > 0:
                return
        if not self._macro_direction_allowed(action, fac_m1):
            return
        stage_idx = self.stage_state[direction]
        if stage_idx >= len(STAGE_RATIOS_MACRO):
            return
        next_fraction = STAGE_RATIOS_MACRO[stage_idx]
        lot = self.BASE_POCKET_LOT * next_fraction
        units = int(round(lot * 100000))
        if units < self.MIN_UNITS:
            return
        if direction == "short":
            units = -units
        sl_pips = raw_signal.get("sl_pips", 20.0)
        tp_pips = raw_signal.get("tp_pips", 30.0)
        trade_id = f"T{self.trade_seq:05d}"
        self.trade_seq += 1
        trade = {
            "trade_id": trade_id,
            "units": units,
            "price": price,
            "side": direction,
            "open_time": ts.isoformat(),
            "stage_index": stage_idx,
            "entry_sl_pips": sl_pips,
            "entry_tp_pips": tp_pips,
            "sl_price": round(price - sl_pips * 0.01, 3)
            if direction == "long"
            else round(price + sl_pips * 0.01, 3),
            "tp_price": round(price + tp_pips * 0.01, 3)
            if direction == "long"
            else round(price - tp_pips * 0.01, 3),
        }
        self.open_positions["macro"]["open_trades"].append(trade)
        self.stage_state[direction] += 1
        self._recalc_position()

    def _macro_direction_allowed(self, action: str, fac_m1: Dict[str, object]) -> bool:
        fac_h4 = self.latest_fac_h4 or {}
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        if ma10_h4 is None or ma20_h4 is None:
            return True
        close = fac_m1.get("close")
        ema20 = fac_m1.get("ema20") or fac_m1.get("ma20")
        if close is None or ema20 is None:
            return True
        if action == "OPEN_LONG":
            if ma10_h4 <= ma20_h4 - 0.0002:
                return False
            return close >= ema20 - 0.002
        if ma10_h4 >= ma20_h4 + 0.0002:
            return False
        return close <= ema20 + 0.002

    def _apply_partial(
        self, trade_id: str, reduce_units: int, price: float, ts: datetime
    ) -> None:
        trade = self._find_trade(trade_id)
        if not trade:
            return
        signed_units = int(reduce_units)
        quantity = abs(signed_units)
        if quantity == 0:
            return
        self._close_units(trade, quantity, price, ts, reason="partial_reduction")
        self._cleanup_trades()

    def _process_exit_decisions(self, decisions, price: float, ts: datetime) -> None:
        for decision in decisions:
            remaining = abs(decision.units)
            target_side = "long" if decision.units < 0 else "short"
            trades = [
                t
                for t in self.open_positions["macro"]["open_trades"]
                if t["side"] == target_side
            ]
            for trade in trades:
                if remaining <= 0:
                    break
                units_available = abs(trade["units"])
                close_qty = min(remaining, units_available)
                self._close_units(trade, close_qty, price, ts, decision.reason)
                remaining -= close_qty
            if remaining > 0:
                pass
        self._cleanup_trades()

    def _close_units(
        self,
        trade: Dict[str, object],
        quantity: int,
        price: float,
        ts: datetime,
        reason: str,
    ) -> None:
        side = trade["side"]
        entry_price = trade["price"]
        pnl_pips = pips_between(entry_price, price, side)
        pnl_jpy = pnl_pips * (quantity / 100.0)
        record = {
            "trade_id": trade["trade_id"],
            "side": side,
            "closed_units": quantity,
            "pnl_pips": pnl_pips,
            "pnl_jpy": pnl_jpy,
            "entry_time": trade["open_time"],
            "exit_time": ts.isoformat(),
            "entry_price": entry_price,
            "exit_price": price,
            "reason": reason,
            "stage_index": trade.get("stage_index", 0),
        }
        self.closed_trades.append(record)
        signed = quantity if trade["units"] > 0 else -quantity
        trade["units"] = int(trade["units"]) - signed
        if abs(trade["units"]) < 1:
            trade["units"] = 0

    def _cleanup_trades(self) -> None:
        trades = []
        for trade in self.open_positions["macro"]["open_trades"]:
            if trade["units"] != 0:
                trades.append(trade)
        self.open_positions["macro"]["open_trades"] = trades
        self._recalc_position()

    def _recalc_position(self) -> None:
        info = self.open_positions["macro"]
        long_trades = [t for t in info["open_trades"] if t["units"] > 0]
        short_trades = [t for t in info["open_trades"] if t["units"] < 0]
        long_units = sum(int(t["units"]) for t in long_trades)
        short_units = sum(-int(t["units"]) for t in short_trades)
        info["long_units"] = long_units
        info["short_units"] = short_units
        net_units = long_units - short_units
        info["units"] = net_units
        info["avg_price"] = (
            np.average(
                [t["price"] for t in info["open_trades"]],
                weights=[abs(t["units"]) for t in info["open_trades"]],
            )
            if info["open_trades"]
            else 0.0
        )
        if long_units > 0:
            info["long_avg_price"] = (
                np.average(
                    [t["price"] for t in long_trades],
                    weights=[t["units"] for t in long_trades],
                )
                if long_trades
                else 0.0
            )
        else:
            info["long_avg_price"] = 0.0
            self.stage_state["long"] = 0
        if short_units > 0:
            info["short_avg_price"] = (
                np.average(
                    [t["price"] for t in short_trades],
                    weights=[-t["units"] for t in short_trades],
                )
                if short_trades
                else 0.0
            )
        else:
            info["short_avg_price"] = 0.0
            self.stage_state["short"] = 0
        self.open_positions["__net__"]["units"] = net_units

    def _update_unrealized(self, price: float) -> None:
        for trade in self.open_positions["macro"]["open_trades"]:
            side = trade["side"]
            entry = trade["price"]
            pnl_pips = pips_between(entry, price, side)
            trade["unrealized_pl_pips"] = pnl_pips
            trade["unrealized_pl"] = pnl_pips * (abs(trade["units"]) / 100.0)

    def _find_trade(self, trade_id: str) -> Optional[Dict[str, object]]:
        for trade in self.open_positions["macro"]["open_trades"]:
            if trade["trade_id"] == trade_id:
                return trade
        return None

    def _close_all(self, price: float, ts: datetime) -> None:
        for trade in list(self.open_positions["macro"]["open_trades"]):
            quantity = abs(trade["units"])
            self._close_units(trade, quantity, price, ts, reason="end_of_data")
        self._cleanup_trades()


def summarize(trades: Iterable[Dict[str, object]]) -> Dict[str, object]:
    trades_list = list(trades)
    if not trades_list:
        return {"trades": 0, "pnl_pips": 0.0, "pnl_jpy": 0.0, "win_rate": 0.0}
    total_pips = sum(t["pnl_pips"] for t in trades_list)
    total_jpy = sum(t["pnl_jpy"] for t in trades_list)
    wins = sum(1 for t in trades_list if t["pnl_pips"] > 0)
    return {
        "trades": len(trades_list),
        "pnl_pips": round(total_pips, 3),
        "pnl_jpy": round(total_jpy, 2),
        "win_rate": round((wins / len(trades_list)) * 100.0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay TrendMA on recorded ticks.")
    parser.add_argument(
        "--ticks-glob",
        required=True,
        help="Glob pattern for tick files (e.g. tmp/ticks_USDJPY_202510*.jsonl)",
    )
    parser.add_argument("--out", required=True, help="Output JSON file path.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.ticks_glob))
    if not files:
        raise SystemExit(f"No tick files matched pattern: {args.ticks_glob}")

    replayer = TrendMAReplayer()
    all_trades: List[Dict[str, object]] = []
    per_file_summary: Dict[str, Dict[str, object]] = {}

    for path_str in files:
        path = Path(path_str)
        ticks = load_ticks(path)
        if not ticks:
            continue
        replayer.run_on_ticks(ticks)
        trades = [
            dict(t, source_file=path.name) for t in replayer.closed_trades
        ]
        all_trades.extend(trades)
        per_file_summary[path.name] = summarize(trades)

    output = {
        "summary": summarize(all_trades),
        "files": per_file_summary,
        "trades": all_trades,
    }
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
