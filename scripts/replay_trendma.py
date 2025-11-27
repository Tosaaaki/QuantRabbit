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
from collections import Counter

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.exit_manager import ExitManager
from execution.order_manager import plan_partial_reductions, _PARTIAL_STAGE
from strategies.trend.ma_cross import MovingAverageCross
from analysis.regime_classifier import classify
from workers.trend_h1 import config as trend_cfg
from workers.trend_h1.worker import _confidence_scale


UTC = timezone.utc
PIP = 0.01


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


class H1Aggregator:
    def __init__(self) -> None:
        self._current_key: Optional[datetime] = None
        self._current: Optional[Dict[str, object]] = None
        self.history: List[Dict[str, object]] = []

    def update(self, candle: Dict[str, object]) -> bool:
        dt = candle["time"]
        if not isinstance(dt, datetime):
            raise ValueError("candle['time'] must be datetime")
        key = dt.replace(minute=0, second=0, microsecond=0)
        new_closed = False
        if self._current_key is None:
            self._current_key = key
            self._current = dict(candle)
            return False
        if key != self._current_key:
            if self._current:
                self.history.append(dict(self._current))
                self.history = self.history[-900:]
                new_closed = True
            self._current_key = key
            self._current = dict(candle)
        else:
            cur = self._current or dict(candle)
            cur["high"] = max(cur["high"], candle["high"])
            cur["low"] = min(cur["low"], candle["low"])
            cur["close"] = candle["close"]
            cur["time"] = candle["time"]
            self._current = cur
        return new_closed

    def finalize(self) -> None:
        if self._current:
            self.history.append(dict(self._current))
            self.history = self.history[-900:]
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


STAGE_RATIOS_MACRO = trend_cfg.STAGE_RATIOS


class TrendMAReplayer:
    BASE_POCKET_LOT = 0.003  # mirror live macro sizing cap (~0.003 lot total)
    MIN_UNITS = 70

    def __init__(self) -> None:
        self.exit_manager = ExitManager()
        self.reset()

    def reset(self) -> None:
        self.m1_history: List[Dict[str, object]] = []
        self.h4_history: List[Dict[str, object]] = []
        self.h1_history: List[Dict[str, object]] = []
        self.h4_agg = H4Aggregator()
        self.h1_agg = H1Aggregator()
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
        epoch0 = datetime(1970, 1, 1, tzinfo=UTC)
        self.cooldown_until = epoch0
        self.direction_block_until = {"long": epoch0, "short": epoch0}
        self.recent_signal_gate: Dict[str, datetime] = {}
        self.last_signal: Optional[Dict[str, object]] = None
        self.latest_fac_h1: Optional[Dict[str, object]] = None
        self.trade_seq = 1
        self.closed_trades: List[Dict[str, object]] = []
        _PARTIAL_STAGE.clear()
        self.latest_fac_h4: Optional[Dict[str, object]] = None
        self.equity = 10000.0
        self.debug_counts: Counter[str] = Counter()

    def run_on_ticks(self, ticks: Sequence[Tick]) -> None:
        self.reset()
        self._process_ticks(ticks, allow_entries=True, finalize=True)

    def run_with_warmup(
        self,
        warmup_ticks: Sequence[Tick],
        main_ticks: Sequence[Tick],
    ) -> None:
        """Run evaluation ticks after seeding factor state with warmup ticks."""
        self.reset()
        if warmup_ticks:
            self._process_ticks(warmup_ticks, allow_entries=False, finalize=False)
        self._process_ticks(main_ticks, allow_entries=True, finalize=True)

    def _process_ticks(
        self,
        ticks: Sequence[Tick],
        *,
        allow_entries: bool,
        finalize: bool,
    ) -> None:
        if not ticks:
            return
        m1_candles = build_m1_candles(ticks)
        for candle in m1_candles:
            self._on_m1_candle(candle, allow_entries=allow_entries)
        if finalize and self.m1_history:
            last_price = self.m1_history[-1]["close"]
            end_time = self.m1_history[-1]["time"]
            if self.open_positions["macro"]["open_trades"]:
                self._close_all(last_price, end_time)
            self.h4_agg.finalize()
            self.h1_agg.finalize()

    def _on_m1_candle(
        self,
        candle: Dict[str, object],
        *,
        allow_entries: bool = True,
    ) -> None:
        price = candle["close"]
        ctime = candle["time"]
        self.m1_history.append(candle)
        self.h4_agg.update(candle)
        h1_closed = self.h1_agg.update(candle)
        if h1_closed and self.h1_agg.history:
            self.h1_history = self.h1_agg.history[-900:]
            fac_h1 = compute_factors(
                self.h1_history,
                min_bars=max(trend_cfg.MIN_CANDLES, 40),
            )
            if fac_h1:
                fac_h1["atr_pips"] = fac_h1.get("atr", 0.0) * 100.0
                self.latest_fac_h1 = fac_h1
                latest_h1 = self.h1_history[-1]
                self._handle_h1_signal(
                    fac_h1,
                    latest_h1["time"],
                    latest_h1["close"],
                    allow_entries=allow_entries,
                )
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

        signals: List[Dict[str, object]] = []
        if self.last_signal:
            signals.append(self.last_signal)

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

        exit_decisions = self.exit_manager.plan_closures(
            self.open_positions,
            signals,
            fac_m1,
            fac_h4,
            event_soon=False,
            range_mode=False,
            now=ctime,
            stage_tracker=None,
        )
        if exit_decisions:
            self._process_exit_decisions(exit_decisions, price, ctime)

    def _handle_h1_signal(
        self,
        fac_h1: Dict[str, object],
        ts: datetime,
        price: float,
        *,
        allow_entries: bool = True,
    ) -> None:
        regime = classify(fac_h1, "H1")
        decision = MovingAverageCross.check(fac_h1)
        if decision is None:
            self.debug_counts["no_signal"] += 1
        else:
            self.debug_counts["raw_signal"] += 1
        if decision:
            self.last_signal = {
                "strategy": "TrendMA",
                "pocket": "macro",
                "action": decision["action"],
                "confidence": decision.get("confidence", 50),
                "sl_pips": decision.get("sl_pips"),
                "tp_pips": decision.get("tp_pips"),
                "tag": decision.get("tag", "TrendMA"),
                "_meta": decision.get("_meta"),
            }
        else:
            self.last_signal = None

        if decision is None:
            return

        confidence = int(decision.get("confidence", 0))
        if confidence < trend_cfg.MIN_CONFIDENCE:
            self.debug_counts["confidence_low"] += 1
            return

        atr_pips = fac_h1.get("atr_pips") or fac_h1.get("atr", 0.0) * 100.0
        if (
            atr_pips < trend_cfg.MIN_ATR_PIPS
            or atr_pips > trend_cfg.MAX_ATR_PIPS
        ):
            self.debug_counts["atr_block"] += 1
            return

        if trend_cfg.REQUIRE_REGIME and regime not in trend_cfg.REQUIRE_REGIME:
            self.debug_counts[f"regime_block:{regime or 'none'}"] += 1
            return
        if trend_cfg.BLOCK_REGIME and regime in trend_cfg.BLOCK_REGIME:
            self.debug_counts[f"block_regime:{regime}"] += 1
            return

        action = decision.get("action")
        if action not in {"OPEN_LONG", "OPEN_SHORT"}:
            self.debug_counts["unsupported_action"] += 1
            return

        direction = "long" if action == "OPEN_LONG" else "short"
        allowed_dirs = {d.lower() for d in trend_cfg.ALLOWED_DIRECTIONS} or {"long", "short"}
        if direction not in allowed_dirs:
            self.debug_counts[f"direction_block:{direction}"] += 1
            return

        if ts < self.cooldown_until:
            self.debug_counts["global_cooldown"] += 1
            return
        if ts < self.direction_block_until[direction]:
            self.debug_counts[f"direction_cooldown:{direction}"] += 1
            return

        tag = decision.get("tag") or "TrendMA"
        gate_key = f"{tag}:{direction}"
        last_ts = self.recent_signal_gate.get(gate_key)
        if last_ts and (ts - last_ts).total_seconds() < trend_cfg.REPEAT_BLOCK_SEC:
            self.debug_counts["repeat_gate"] += 1
            return

        if allow_entries:
            self._maybe_enter(decision, price, ts, fac_h1)
        self.recent_signal_gate[gate_key] = ts

    def _maybe_enter(
        self,
        raw_signal: Dict[str, object],
        price: float,
        ts: datetime,
        fac_h1: Dict[str, object],
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
        meta = raw_signal.get("_meta") or {}
        fast_gap = float(meta.get("price_to_fast_pips", 0.0) or 0.0)
        if direction == "long" and fast_gap < trend_cfg.MIN_FAST_GAP_PIPS:
            return
        if direction == "short" and fast_gap > -trend_cfg.MIN_FAST_GAP_PIPS:
            return
        if not self._macro_direction_allowed(action, fac_h1):
            return
        stage_idx = self.stage_state[direction]
        if stage_idx >= len(STAGE_RATIOS_MACRO):
            return
        confidence = int(raw_signal.get("confidence", 0))
        sl_pips = float(raw_signal.get("sl_pips", 0.0) or 0.0)
        tp_pips = float(raw_signal.get("tp_pips", 0.0) or 0.0)
        if sl_pips <= 0.0 or tp_pips <= 0.0:
            return
        if (ts - self.direction_block_until[direction]).total_seconds() < trend_cfg.REENTRY_COOLDOWN_SEC:
            return
        base_lot = self._risk_lot(sl_pips)
        lot = max(trend_cfg.MIN_LOT, min(trend_cfg.MAX_LOT, base_lot))
        lot *= _confidence_scale(confidence)
        lot = max(trend_cfg.MIN_LOT, min(trend_cfg.MAX_LOT, lot))
        next_fraction = STAGE_RATIOS_MACRO[stage_idx]
        lot *= next_fraction
        units = int(round(lot * 100000))
        if units < self.MIN_UNITS:
            return
        if direction == "short":
            units = -units
        atr_pips = float(fac_h1.get("atr_pips") or fac_h1.get("atr", 0.0) * 100.0)
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
            "entry_meta": raw_signal.get("_meta"),
            "signal_confidence": confidence,
            "signal_atr_pips": atr_pips,
            "signal_gap_pips": (raw_signal.get("_meta") or {}).get("gap_pips"),
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
        cooldown = max(trend_cfg.ENTRY_COOLDOWN_SEC, 60.0)
        self.cooldown_until = ts + timedelta(seconds=cooldown)
        self.direction_block_until[direction] = ts + timedelta(seconds=trend_cfg.REENTRY_COOLDOWN_SEC)

    def _macro_direction_allowed(self, action: str, fac_h1: Dict[str, object]) -> bool:
        fac_h4 = self.latest_fac_h4 or {}
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        if ma10_h4 is None or ma20_h4 is None:
            return True
        gap_h4 = float(ma10_h4) - float(ma20_h4)
        bias_buffer = 0.00018
        if action == "OPEN_LONG" and gap_h4 < -bias_buffer:
            ma10_h1 = fac_h1.get("ma10")
            ma20_h1 = fac_h1.get("ma20")
            atr_pips = float(fac_h1.get("atr_pips") or 0.0)
            if ma10_h1 is None or ma20_h1 is None:
                return False
            gap_h1_pips = (float(ma10_h1) - float(ma20_h1)) / PIP
            if not (
                gap_h1_pips >= trend_cfg.H1_OVERRIDE_GAP_PIPS
                and atr_pips >= trend_cfg.H1_OVERRIDE_ATR_PIPS
            ):
                return False
        elif action == "OPEN_SHORT" and gap_h4 > bias_buffer:
            ma10_h1 = fac_h1.get("ma10")
            ma20_h1 = fac_h1.get("ma20")
            atr_pips = float(fac_h1.get("atr_pips") or 0.0)
            if ma10_h1 is None or ma20_h1 is None:
                return False
            gap_h1_pips = (float(ma10_h1) - float(ma20_h1)) / PIP
            if not (
                gap_h1_pips <= -trend_cfg.H1_OVERRIDE_GAP_PIPS
                and atr_pips >= trend_cfg.H1_OVERRIDE_ATR_PIPS
            ):
                return False
        ma10_h1 = fac_h1.get("ma10")
        ma20_h1 = fac_h1.get("ma20")
        if ma10_h1 is None or ma20_h1 is None:
            return True
        gap_h1 = float(ma10_h1) - float(ma20_h1)
        micro_buffer = 0.00006
        if action == "OPEN_LONG" and gap_h1 < -micro_buffer:
            return False
        if action == "OPEN_SHORT" and gap_h1 > micro_buffer:
            return False
        return True

    def _risk_lot(self, sl_pips: float) -> float:
        if sl_pips <= 0.0:
            return 0.0
        risk_amount = self.equity * trend_cfg.RISK_PCT
        return max(0.0, risk_amount / (sl_pips * 1000.0))

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
            "signal_confidence": trade.get("signal_confidence"),
            "signal_atr_pips": trade.get("signal_atr_pips"),
            "signal_gap_pips": trade.get("signal_gap_pips"),
            "entry_meta": trade.get("entry_meta"),
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
    parser.add_argument(
        "--warmup-files",
        type=int,
        default=0,
        help="Number of preceding files to use as warmup (entries disabled).",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.ticks_glob))
    if not files:
        raise SystemExit(f"No tick files matched pattern: {args.ticks_glob}")

    replayer = TrendMAReplayer()
    all_trades: List[Dict[str, object]] = []
    per_file_summary: Dict[str, Dict[str, object]] = {}
    per_file_debug: Dict[str, Dict[str, int]] = {}
    tick_cache: Dict[str, List[Tick]] = {}

    def _get_ticks(path_str: str) -> List[Tick]:
        cached = tick_cache.get(path_str)
        if cached is not None:
            return cached
        payload = load_ticks(Path(path_str))
        tick_cache[path_str] = payload
        return payload

    warmup_span = max(0, args.warmup_files)

    for idx, path_str in enumerate(files):
        path = Path(path_str)
        eval_ticks = _get_ticks(path_str)
        if not eval_ticks:
            continue
        warmup_ticks: List[Tick] = []
        if warmup_span > 0 and idx > 0:
            warmup_paths = files[max(0, idx - warmup_span) : idx]
            for warm_path in warmup_paths:
                warmup_ticks.extend(_get_ticks(warm_path))
        if warmup_ticks:
            replayer.run_with_warmup(warmup_ticks, eval_ticks)
        else:
            replayer.run_on_ticks(eval_ticks)
        trades = [
            dict(t, source_file=path.name) for t in replayer.closed_trades
        ]
        all_trades.extend(trades)
        per_file_summary[path.name] = summarize(trades)
        per_file_debug[path.name] = dict(replayer.debug_counts)

    output = {
        "summary": summarize(all_trades),
        "files": per_file_summary,
        "debug": per_file_debug,
        "trades": all_trades,
    }
    Path(args.out).write_text(json.dumps(output, indent=2, default=str))
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
