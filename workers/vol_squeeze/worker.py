from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
import math
import asyncio
import logging

class DataFeed(Protocol):
    def get_bars(self, symbol: str, tf: str, n: int) -> Any: ...
    def last(self, symbol: str) -> float: ...

class Broker(Protocol):
    def send(self, order: Dict[str, Any]) -> Any: ...

from workers.common.exit_adapter import build_exit_manager

LOG = logging.getLogger(__name__)

def _as_bars(bars: Any):
    if not bars: return []
    if isinstance(bars, list) and bars and isinstance(bars[0], dict):
        return [{"open":float(x.get("open", x.get("o",0))),
                 "high":float(x.get("high", x.get("h",0))),
                 "low": float(x.get("low",  x.get("l",0))),
                 "close":float(x.get("close",x.get("c",0)))} for x in bars]
    return []

def _ema(vals, n):
    k = 2/(n+1)
    e = vals[0]
    out = []
    for v in vals:
        e = v*k + e*(1-k)
        out.append(e)
    return out

def _atr(b, n=14):
    if len(b) < n+1: return 0.0
    trs = []
    for i in range(1, n+1):
        h, l = b[-i]["high"], b[-i]["low"]
        pc = b[-i-1]["close"]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return sum(trs)/len(trs)

def _stdev(vals):
    if len(vals) < 2: return 0.0
    m = sum(vals)/len(vals)
    return (sum((v-m)**2 for v in vals)/len(vals))**0.5

class VolSqueezeWorker:
    """
    Detect 'squeeze' (BB width at low percentile) then trade a Keltner breakout.
    """
    def __init__(self, cfg: Dict[str, any], broker: Optional[Broker], datafeed: DataFeed, logger=None):
        self.c = cfg; self.b = broker; self.d = datafeed; self.log = logger
        self.tf = self.c.get("timeframe", "5m")
        self.place_orders = bool(self.c.get("place_orders", False))
        self.exit_cfg = self.c.get("exit", {})
        self.k = float(self.c.get("keltner_mult", 1.5))
        self.bb_len = int(self.c.get("bb_len", 20))
        self.ema_len = int(self.c.get("ema_len", 20))
        self.atr_len = int(self.c.get("atr_len", 14))
        self.pct = float(self.c.get("squeeze_pctile", 0.2))
        self.allow_long = bool(self.c.get("allow_long", True))
        self.allow_short = bool(self.c.get("allow_short", True))
        self.ema_slope_min = float(self.c.get("ema_slope_min", 0.0))
        self.cooldown_bars = int(self.c.get("cooldown_bars", 2))
        self._last_entry_idx: Dict[str, int] = {}

        self.exit_mgr = build_exit_manager(self.exit_cfg)

    def run_once(self):
        intents = []
        for sym in self.c.get("universe", []):
            ti = self.edge(sym)
            if ti:
                intents.append(ti)
                if self.place_orders and self.b is not None:
                    order = self._mk_order(sym, ti)
                    if self.exit_mgr:
                        order = self.exit_mgr.attach(order)
                    self.b.send(order)
        return intents

    def edge(self, sym: str):
        bars = _as_bars(self.d.get_bars(sym, self.tf, max(self.bb_len*6, 200)))
        if len(bars) < self.bb_len*3: return None
        closes = [x["close"] for x in bars]
        last_idx = len(closes) - 1
        prev_idx = max(0, last_idx - 1)

        ema = _ema(closes, self.ema_len)
        atr = _atr(bars, n=self.atr_len)
        mid = ema[-1]
        up = mid + self.k*atr
        dn = mid - self.k*atr
        slope = 0.0
        if len(ema) >= 2:
            slope = ema[-1] - ema[-2]

        # BB width percentile
        widths = []
        for i in range(self.bb_len, len(closes)+1):
            seg = closes[i-self.bb_len:i]
            sd = _stdev(seg)
            widths.append(2*sd)
        if not widths: return None
        cur_w = widths[-1]
        sorted_w = sorted(widths[-self.bb_len*5:])  # local window
        rank = sum(1 for w in sorted_w if w <= cur_w)/len(sorted_w)

        last = closes[-1]
        squeeze = (rank <= self.pct)
        if not squeeze: return None

        last_entry = self._last_entry_idx.get(sym, -999)
        if last_idx - last_entry < self.cooldown_bars:
            return None

        if self.allow_long and last > up and slope >= self.ema_slope_min:
            self._last_entry_idx[sym] = last_idx
            return {
                "symbol": sym,
                "side": "long",
                "px": last,
                "meta": {
                    "mid": mid,
                    "up": up,
                    "dn": dn,
                    "width_rank": rank,
                    "atr": atr,
                },
            }
        elif self.allow_short and last < dn and slope <= -self.ema_slope_min:
            self._last_entry_idx[sym] = last_idx
            return {
                "symbol": sym,
                "side": "short",
                "px": last,
                "meta": {
                    "mid": mid,
                    "up": up,
                    "dn": dn,
                    "width_rank": rank,
                    "atr": atr,
                },
            }
        return None

    def _mk_order(self, sym: str, intent: Dict[str, any]):
        side = "buy" if intent["side"] == "long" else "sell"
        size = max(0.0, float(self.c.get("budget_bps", 30))/10000.0)
        return {"symbol": sym, "side": side, "type": "market", "size": size, "meta": {"worker_id": self.c.get("id"), "intent": intent}}


async def _idle_loop() -> None:
    """Keep systemd service alive when run as a module without a runner."""
    LOG.info("vol_squeeze worker idle loop started (no live wiring)")
    try:
        while True:
            await asyncio.sleep(3600.0)
    except asyncio.CancelledError:  # pragma: no cover
        LOG.info("vol_squeeze worker idle loop cancelled")
        raise


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_idle_loop())
