"""DOJO lab bot: one parameterized worker for the declared experiment grid.

Config via env var DOJO_BOT_CONFIG (JSON):
  signal        "burst" | "pullback_limit" | "range_fade_limit"
  tp_pips       float
  sl_pips       float or null  (null = SL-free, the operator's philosophy)
  ceiling_min   int    hard exit after N minutes (the cage; always on)
  max_concurrent int
  per_pos_lev   float  NAV-proportional exposure per position
  atr_floor_pips float  minimum ATR to trade (dead-market guard)
  pull_atr      float  (pullback_limit) limit distance in ATRs below/above close
  fade_atr      float  (range_fade_limit) band distance in ATRs
  eff_max       float  (range_fade_limit) only fade when 6h efficiency below this

All indicators are incremental (no pandas): 24h trend via a 1441-bar
ring, Wilder ATR(14), 6h efficiency ratio.  Trades only USD_JPY.
"""

from __future__ import annotations

import json
import os
from collections import deque

from quant_rabbit.virtual_broker import VirtualBroker, VirtualBrokerError

def _pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


class _PairState:
    def __init__(self):
        self.closes: deque[float] = deque(maxlen=1441)
        self.diffs_6h: deque[float] = deque(maxlen=360)
        self.highs3: deque[float] = deque(maxlen=3)
        self.lows3: deque[float] = deque(maxlen=3)
        self.atr: float | None = None
        self.prev_close: float | None = None
        self.my_trades: dict[str, float] = {}
        self.my_orders: list[str] = []


class Bot:
    def __init__(self, broker: VirtualBroker):
        self.broker = broker
        cfg = json.loads(os.environ["DOJO_BOT_CONFIG"])
        self.pairs = cfg.get("pairs", ["USD_JPY"])
        self.signal = cfg["signal"]
        self.tp_pips = float(cfg.get("tp_pips", 0) or 0)
        self.tp_atr = cfg.get("tp_atr")  # scale-free take: TP = tp_atr x ATR
        self.sl_pips = cfg.get("sl_pips")
        self.ceiling_s = int(cfg["ceiling_min"]) * 60
        self.max_concurrent = int(cfg.get("max_concurrent", 3))
        self.per_pos_lev = float(cfg.get("per_pos_lev", 4.3))
        self.atr_floor = float(cfg.get("atr_floor_pips", 1.0))
        self.pull_atr = float(cfg.get("pull_atr", 0.6))
        self.fade_atr = float(cfg.get("fade_atr", 1.2))
        self.eff_max = float(cfg.get("eff_max", 0.2))
        self.global_max = int(cfg.get("global_max_concurrent",
                                      self.max_concurrent * len(self.pairs)))
        self.state: dict[str, _PairState] = {p: _PairState() for p in self.pairs}
        self._owner: dict[str, str] = {}  # trade_id -> pair

    # ---- incremental indicators -----------------------------------------
    def _update(self, st: "_PairState", bar: dict) -> None:
        mid_c = (bar["bid_c"] + bar["ask_c"]) / 2
        mid_h = (bar["bid_h"] + bar["ask_h"]) / 2
        mid_l = (bar["bid_l"] + bar["ask_l"]) / 2
        if st.prev_close is not None:
            tr = max(mid_h - mid_l, abs(mid_h - st.prev_close),
                     abs(mid_l - st.prev_close))
            st.atr = tr if st.atr is None else st.atr + (tr - st.atr) / 14.0
            st.diffs_6h.append(abs(mid_c - st.prev_close))
        st.prev_close = mid_c
        st.closes.append(mid_c)
        st.highs3.append(mid_h)
        st.lows3.append(mid_l)

    @staticmethod
    def _trend(st: "_PairState") -> str | None:
        if len(st.closes) < 1441:
            return None
        return "LONG" if st.closes[-1] > st.closes[0] else "SHORT"

    @staticmethod
    def _efficiency_6h(st: "_PairState") -> float | None:
        if len(st.diffs_6h) < 360 or len(st.closes) < 361:
            return None
        path = sum(st.diffs_6h)
        if path <= 0:
            return None
        return abs(st.closes[-1] - st.closes[-361]) / path

    # ---- lifecycle -------------------------------------------------------
    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        st = self.state.get(pair)
        if st is None:
            return
        pip = _pip(pair)
        prior_h3 = max(st.highs3) if len(st.highs3) == 3 else None
        prior_l3 = min(st.lows3) if len(st.lows3) == 3 else None
        self._update(st, bar)

        # adopt limit fills into this pair's book; ceiling exits
        for trade_id, pos in list(self.broker.positions.items()):
            if trade_id not in self._owner and pos.pair == pair:
                self._owner[trade_id] = pair
                st.my_trades[trade_id] = epoch
        for trade_id in list(st.my_trades):
            if trade_id not in self.broker.positions:
                del st.my_trades[trade_id]
                self._owner.pop(trade_id, None)
            elif epoch - st.my_trades[trade_id] >= self.ceiling_s:
                try:
                    self.broker.close_trade(trade_id)
                except VirtualBrokerError:
                    pass
                st.my_trades.pop(trade_id, None)
                self._owner.pop(trade_id, None)

        trend = self._trend(st)
        if trend is None or st.atr is None:
            return
        atr_pips = st.atr / pip
        if atr_pips < self.atr_floor:
            return
        total_open = sum(len(s.my_trades) for s in self.state.values())
        if total_open >= self.global_max:
            open_n = self.max_concurrent  # treat as full
        else:
            open_n = len(st.my_trades)
        mid_c = (bar["bid_c"] + bar["ask_c"]) / 2

        def units_for(price: float) -> float:
            try:
                equity = self.broker.account()["equity_jpy"]
                jpy_per_unit = price * self.broker._jpy_per_quote_unit(pair)
            except VirtualBrokerError:
                return 0.0
            if jpy_per_unit <= 0:
                return 0.0
            return max(equity, 0.0) * self.per_pos_lev / jpy_per_unit

        digits = 3 if pair.endswith("JPY") else 5
        tp_pips = (float(self.tp_atr) * atr_pips) if self.tp_atr else self.tp_pips
        spread_pips = (bar["ask_c"] - bar["bid_c"]) / pip
        if tp_pips <= 0 or spread_pips > tp_pips * 0.35:
            return  # habitat gate: cost must stay well under the take

        if self.signal == "burst":
            if open_n >= self.max_concurrent or prior_h3 is None:
                return
            triggered = (trend == "LONG" and mid_c > prior_h3) or (
                trend == "SHORT" and mid_c < prior_l3)
            if not triggered:
                return
            units = units_for(mid_c)
            if units <= 0:
                return
            try:
                tid = self.broker.market_order(
                    pair, trend, units, tp_pips=tp_pips, sl_pips=self.sl_pips)
                st.my_trades[tid] = epoch
                self._owner[tid] = pair
            except VirtualBrokerError:
                pass

        elif self.signal == "pullback_limit":
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            if open_n >= self.max_concurrent:
                return
            dist = self.pull_atr * st.atr
            price = mid_c - dist if trend == "LONG" else mid_c + dist
            units = units_for(price)
            if units <= 0:
                return
            try:
                oid = self.broker.limit_order(
                    pair, trend, units, price=round(price, digits),
                    tp_pips=tp_pips, sl_pips=self.sl_pips)
                st.my_orders = [oid]
            except VirtualBrokerError:
                pass

        elif self.signal == "range_fade_limit":
            for oid in st.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            st.my_orders = []
            eff = self._efficiency_6h(st)
            if eff is None or eff > self.eff_max or open_n >= self.max_concurrent:
                return
            dist = self.fade_atr * st.atr
            units = units_for(mid_c)
            if units <= 0:
                return
            for side, price in (("LONG", mid_c - dist), ("SHORT", mid_c + dist)):
                try:
                    oid = self.broker.limit_order(
                        pair, side, units, price=round(price, digits),
                        tp_pips=tp_pips, sl_pips=self.sl_pips)
                    st.my_orders.append(oid)
                except VirtualBrokerError:
                    pass
