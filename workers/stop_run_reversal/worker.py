from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
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
        if len(bars) < 50:
            return None
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


async def _idle_loop() -> None:
    LOG.info("stop_run_reversal idle loop started (no live runner configured)")
    try:
        while True:
            await asyncio.sleep(3600.0)
    except asyncio.CancelledError:  # pragma: no cover
        LOG.info("stop_run_reversal idle loop cancelled")
        raise


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    from .config import DEFAULT_CONFIG
    from workers.common import addon_live

    cfg = addon_live.apply_env_overrides(
        "STOP_RUN_REVERSAL",
        DEFAULT_CONFIG,
        default_universe=["USD_JPY"],
        default_pocket="micro",
        default_loop=30.0,
    )
    if not cfg.get("live_enabled"):
        LOG.info("stop_run_reversal idle mode (set STOP_RUN_REVERSAL_LIVE=1 or ADDON_LIVE_MODE=1 to enable)")
        asyncio.run(_idle_loop())
        raise SystemExit(0)

    datafeed = addon_live.LiveDataFeed(default_timeframe=str(cfg.get("timeframe", "M5")), logger=LOG)
    broker = addon_live.AddonLiveBroker(
        worker_id=str(cfg.get("id", "stop_run_reversal")),
        pocket=str(cfg.get("pocket", "micro")),
        datafeed=datafeed,
        exit_cfg=cfg.get("exit"),
        atr_len=int(cfg.get("atr_len", 14)),
        atr_timeframe=str(cfg.get("timeframe", "M5")),
        default_budget_bps=cfg.get("budget_bps"),
        ttl_ms=float(cfg.get("ttl_ms", 800.0)),
        require_passive=bool(cfg.get("require_passive", True)),
        logger=LOG,
    )
    worker = StopRunReversalWorker(cfg, broker=broker, datafeed=datafeed, logger=LOG)
    addon_live.run_loop(
        worker,
        loop_interval_sec=float(cfg.get("loop_interval_sec", 30.0)),
        pocket=str(cfg.get("pocket", "micro")),
        logger=LOG,
    )
