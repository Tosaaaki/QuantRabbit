from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol

class DataFeed(Protocol):
    def get_bars(self, symbol: str, tf: str, n: int) -> Any: ...
    def last(self, symbol: str) -> float: ...

class Broker(Protocol):
    def send(self, order: Dict[str, Any]) -> Any: ...

from workers.common.exit_adapter import build_exit_manager

def _as_bars(bars: Any):
    if not bars: return []
    if isinstance(bars, list) and bars and isinstance(bars[0], dict):
        return [{"open":float(x.get("open", x.get("o",0))),
                 "high":float(x.get("high", x.get("h",0))),
                 "low": float(x.get("low",  x.get("l",0))),
                 "close":float(x.get("close",x.get("c",0)))} for x in bars]
    return []

def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0: return 0.0
    if n % 2: return s[n//2]
    return 0.5*(s[n//2 - 1] + s[n//2])


def _atr(bars, n=14):
    if len(bars) < n + 1:
        return 0.0
    trs = []
    for i in range(1, n + 1):
        h = bars[-i]["high"]
        l = bars[-i]["low"]
        pc = bars[-i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs)

class StopRunReversalWorker:
    """
    Detects a stop-run candle (long wick & outsized range), waits for failure to extend,
    then enters reversal.
    """
    def __init__(self, cfg: Dict[str, any], broker: Optional[Broker], datafeed: DataFeed, logger=None):
        self.c = cfg; self.b = broker; self.d = datafeed; self.log = logger
        self.tf = self.c.get("timeframe", "5m")
        self.place_orders = bool(self.c.get("place_orders", False))
        self.exit_cfg = self.c.get("exit", {})
        self.exit_mgr = build_exit_manager(self.exit_cfg)

    def run_once(self):
        intents = []
        for sym in self.c.get("universe", []):
            ti = self.edge(sym)
            if ti:
                intents.append(ti)
                if self.place_orders and self.b is not None:
                    order = {"symbol": sym, "side": "sell" if ti["side"]=="short" else "buy", "type": "market",
                             "size": max(0.0, float(self.c.get("budget_bps", 25))/10000.0),
                             "meta": {"worker_id": self.c.get("id"), "intent": ti}}
                    if self.exit_mgr:
                        order = self.exit_mgr.attach(order)
                    self.b.send(order)
        return intents

    def edge(self, sym: str):
        bars = _as_bars(self.d.get_bars(sym, self.tf, 200))
        if len(bars) < 50: return None
        atr = _atr(bars, n=int(self.c.get("atr_len", 14)))

        rngs = [b["high"] - b["low"] for b in bars[-50:]]
        med_rng = _median(rngs) or 1e-9
        k = float(self.c.get("min_range_mult", 1.8))
        wick_ratio = float(self.c.get("wick_ratio", 0.6))
        confirm = int(self.c.get("confirm_bars", 2))

        # last complete bar as trigger candle
        t = bars[-2]
        rng = t["high"] - t["low"]
        body = abs(t["close"] - t["open"])
        up_wick = t["high"] - max(t["open"], t["close"])
        dn_wick = min(t["open"], t["close"]) - t["low"]

        # outsized range
        if rng < k * med_rng:
            return None

        # bull trap: long upper wick + small body near lows
        if up_wick / (rng + 1e-9) >= wick_ratio and body <= 0.5 * rng:
            # failure to extend: next bars do not make new highs, then break prior low
            nxt = bars[-confirm:]
            if max(x["high"] for x in nxt) <= t["high"] and bars[-1]["close"] < t["low"]:
                return {
                    "symbol": sym,
                    "side": "short",
                    "px": bars[-1]["close"],
                    "meta": {
                        "trap": "bull",
                        "trigger_high": t["high"],
                        "trigger_low": t["low"],
                        "atr": atr,
                    },
                }

        # bear trap: long lower wick + small body near highs
        if dn_wick / (rng + 1e-9) >= wick_ratio and body <= 0.5 * rng:
            nxt = bars[-confirm:]
            if min(x["low"] for x in nxt) >= t["low"] and bars[-1]["close"] > t["high"]:
                return {
                    "symbol": sym,
                    "side": "long",
                    "px": bars[-1]["close"],
                    "meta": {
                        "trap": "bear",
                        "trigger_high": t["high"],
                        "trigger_low": t["low"],
                        "atr": atr,
                    },
                }

        return None
