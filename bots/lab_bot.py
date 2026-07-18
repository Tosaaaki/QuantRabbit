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

PAIR = "USD_JPY"
PIP = 0.01


class Bot:
    def __init__(self, broker: VirtualBroker):
        self.broker = broker
        cfg = json.loads(os.environ["DOJO_BOT_CONFIG"])
        self.signal = cfg["signal"]
        self.tp_pips = float(cfg["tp_pips"])
        self.sl_pips = cfg.get("sl_pips")
        self.ceiling_s = int(cfg["ceiling_min"]) * 60
        self.max_concurrent = int(cfg.get("max_concurrent", 3))
        self.per_pos_lev = float(cfg.get("per_pos_lev", 4.3))
        self.atr_floor = float(cfg.get("atr_floor_pips", 1.0))
        self.pull_atr = float(cfg.get("pull_atr", 0.6))
        self.fade_atr = float(cfg.get("fade_atr", 1.2))
        self.eff_max = float(cfg.get("eff_max", 0.2))

        self.closes: deque[float] = deque(maxlen=1441)  # 24h of M1 mids
        self.diffs_6h: deque[float] = deque(maxlen=360)
        self.highs3: deque[float] = deque(maxlen=3)
        self.lows3: deque[float] = deque(maxlen=3)
        self.atr: float | None = None
        self.prev_close: float | None = None
        self.my_trades: dict[str, float] = {}
        self.my_orders: list[str] = []

    # ---- incremental indicators -----------------------------------------
    def _update(self, bar: dict) -> None:
        mid_c = (bar["bid_c"] + bar["ask_c"]) / 2
        mid_h = (bar["bid_h"] + bar["ask_h"]) / 2
        mid_l = (bar["bid_l"] + bar["ask_l"]) / 2
        if self.prev_close is not None:
            tr = max(mid_h - mid_l, abs(mid_h - self.prev_close),
                     abs(mid_l - self.prev_close))
            self.atr = tr if self.atr is None else self.atr + (tr - self.atr) / 14.0
            self.diffs_6h.append(abs(mid_c - self.prev_close))
        self.prev_close = mid_c
        self.closes.append(mid_c)
        self.highs3.append(mid_h)
        self.lows3.append(mid_l)

    def _trend(self) -> str | None:
        if len(self.closes) < 1441:
            return None
        return "LONG" if self.closes[-1] > self.closes[0] else "SHORT"

    def _efficiency_6h(self) -> float | None:
        if len(self.diffs_6h) < 360 or len(self.closes) < 361:
            return None
        path = sum(self.diffs_6h)
        if path <= 0:
            return None
        return abs(self.closes[-1] - self.closes[-361]) / path

    # ---- lifecycle -------------------------------------------------------
    def on_bar_closed(self, pair: str, bar: dict, epoch: int) -> None:
        if pair != PAIR:
            return
        prior_h3 = max(self.highs3) if len(self.highs3) == 3 else None
        prior_l3 = min(self.lows3) if len(self.lows3) == 3 else None
        self._update(bar)

        # adopt limit fills into my book; ceiling exits
        for trade_id in list(self.broker.positions):
            if trade_id not in self.my_trades:
                self.my_trades[trade_id] = epoch
        for trade_id in list(self.my_trades):
            if trade_id not in self.broker.positions:
                del self.my_trades[trade_id]
            elif epoch - self.my_trades[trade_id] >= self.ceiling_s:
                try:
                    self.broker.close_trade(trade_id)
                except VirtualBrokerError:
                    pass
                self.my_trades.pop(trade_id, None)

        trend = self._trend()
        if trend is None or self.atr is None:
            return
        atr_pips = self.atr / PIP
        if atr_pips < self.atr_floor:
            return
        open_n = len(self.my_trades)
        mid_c = (bar["bid_c"] + bar["ask_c"]) / 2

        def units_for(price: float) -> float:
            try:
                equity = self.broker.account()["equity_jpy"]
            except VirtualBrokerError:
                return 0.0
            return max(equity, 0.0) * self.per_pos_lev / price

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
                    PAIR, trend, units, tp_pips=self.tp_pips, sl_pips=self.sl_pips)
                self.my_trades[tid] = epoch
            except VirtualBrokerError:
                pass

        elif self.signal == "pullback_limit":
            # one resting trend-side limit, re-priced every bar
            for oid in self.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            self.my_orders = []
            if open_n >= self.max_concurrent:
                return
            dist = self.pull_atr * self.atr
            price = mid_c - dist if trend == "LONG" else mid_c + dist
            units = units_for(price)
            if units <= 0:
                return
            try:
                oid = self.broker.limit_order(
                    PAIR, trend, units, price=round(price, 3),
                    tp_pips=self.tp_pips, sl_pips=self.sl_pips)
                self.my_orders = [oid]
            except VirtualBrokerError:
                pass

        elif self.signal == "range_fade_limit":
            for oid in self.my_orders:
                try:
                    self.broker.cancel_order(oid)
                except VirtualBrokerError:
                    pass
            self.my_orders = []
            eff = self._efficiency_6h()
            if eff is None or eff > self.eff_max or open_n >= self.max_concurrent:
                return
            dist = self.fade_atr * self.atr
            units = units_for(mid_c)
            if units <= 0:
                return
            for side, price in (("LONG", mid_c - dist), ("SHORT", mid_c + dist)):
                try:
                    oid = self.broker.limit_order(
                        PAIR, side, units, price=round(price, 3),
                        tp_pips=self.tp_pips, sl_pips=self.sl_pips)
                    self.my_orders.append(oid)
                except VirtualBrokerError:
                    pass
