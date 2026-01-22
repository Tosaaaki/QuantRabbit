from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, Tuple
import math, time, asyncio, logging

class DataFeed(Protocol):
    def get_bars(self, symbol: str, tf: str, n: int) -> Any: ...
    def last(self, symbol: str) -> float: ...
    def best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]: ...  # optional

class Broker(Protocol):
    def send(self, order: Dict[str, Any]) -> Any: ...
    def cancel(self, order_id: str) -> Any: ...

class EventFlagSource(Protocol):
    def is_event_now(self) -> bool: ...  # e.g., macro event window

def _as_bars(bars: Any):
    if not bars: return []
    if isinstance(bars, list) and bars and isinstance(bars[0], dict):
        return [{"open":float(x.get("open", x.get("o",0))),
                 "high":float(x.get("high", x.get("h",0))),
                 "low": float(x.get("low",  x.get("l",0))),
                 "close":float(x.get("close",x.get("c",0)))} for x in bars]
    return []

def _atr(b, n=14):
    if len(b) < n+1: return 0.0
    trs = []
    for i in range(1, n+1):
        h, l = b[-i]["high"], b[-i]["low"]
        pc = b[-i-1]["close"]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return sum(trs)/len(trs)

def _round_tick(px: float, tick: float) -> float:
    if tick and tick > 0:
        return round(px / tick) * tick
    return px

LOG = logging.getLogger(__name__)

class MMLiteWorker:
    """
    Extremely simple maker that places symmetric quotes around mid and manages small inventory.
    - Spread: base_spread_bp + spread_k_atr * (ATR/price)*1e4
    - Inventory: bounded by +/- inventory_r (R is risk unit defined by user)
    - Event window: optional kill switch via external flag
    NOTE: This is a placeholder that assumes presence of a broker that supports limit GTC orders.
    """
    def __init__(self, cfg: Dict[str, Any], broker: Broker, datafeed: DataFeed,
                 event_flags: Optional[EventFlagSource] = None, logger=None):
        self.c = cfg; self.b = broker; self.d = datafeed; self.ev = event_flags; self.log = logger
        self.tf = str(self.c.get("timeframe", "5m"))
        self.orders = {}       # sym -> {"bid_id":..., "ask_id":...}
        self.inv = {}          # sym -> position size (signed)
        self.risk_unit = 1.0   # user-defined; for demo we treat size in notional fractions
        self.place_orders = bool(self.c.get("place_orders", False))

    def run_once(self):
        intents = []
        for sym in self.c.get("universe", []):
            it = self._quote(sym)
            if it: intents.append(it)
        return intents

    def _quote(self, sym: str):
        if self.c.get("disable_on_event") and self.ev and self.ev.is_event_now():
            # cancel existing
            pair = self.orders.get(sym, {})
            for oid in [pair.get("bid_id"), pair.get("ask_id")]:
                if oid: 
                    try: self.b.cancel(oid)
                    except Exception: pass
            self.orders[sym] = {}
            return None

        bars = _as_bars(self.d.get_bars(sym, self.tf, 100))
        if not bars: return None
        last = self.d.last(sym) if hasattr(self.d, "last") else bars[-1]["close"]
        atr = _atr(bars, n=int(self.c.get("atr_len", 14)))
        atr_bp = (atr / max(1e-9, last)) * 1e4

        base_bp = float(self.c.get("base_spread_bp", 2.0))
        add_bp = float(self.c.get("spread_k_atr", 0.6)) * atr_bp
        half_bp = 0.5 * (base_bp + add_bp)

        mid = last
        bid = mid * (1 - half_bp/1e4)
        ask = mid * (1 + half_bp/1e4)
        tick = float(self.c.get("tick_size", 0.0))
        bid = _round_tick(bid, tick); ask = _round_tick(ask, tick)

        # inventory skew
        pos = float(self.inv.get(sym, 0.0))
        inv_r = float(self.c.get("inventory_r", 1.0))
        skew = max(-inv_r, min(inv_r, -pos))  # negative pos -> skew bid up to attract buys, etc.
        bid *= (1 + 0.05 * skew / max(1e-9, inv_r))
        ask *= (1 - 0.05 * skew / max(1e-9, inv_r))

        notional = float(self.c.get("size_bps", 10))/1e4  # toy sizing
        if self.place_orders:
            # Cancel & replace strategy (simplified)
            pair = self.orders.get(sym, {})
            for oid in [pair.get("bid_id"), pair.get("ask_id")]:
                if oid:
                    try: self.b.cancel(oid)
                    except Exception: pass
            bid_id = f"{sym}-bid-{time.time()}"
            ask_id = f"{sym}-ask-{time.time()}"
            self.b.send({"id": bid_id, "symbol": sym, "side": "buy", "type": "limit", "price": bid, "size": notional})
            self.b.send({"id": ask_id, "symbol": sym, "side": "sell","type": "limit", "price": ask, "size": notional})
            self.orders[sym] = {"bid_id": bid_id, "ask_id": ask_id}

        return {"symbol": sym, "bid": bid, "ask": ask, "mid": mid, "atr_bp": atr_bp, "half_spread_bp": half_bp}


async def _idle_loop() -> None:
    """Keep systemd unit alive when run via -m workers.mm_lite.worker."""
    LOG.info("mm_lite idle loop started (no live broker/datafeed wiring)")
    try:
        while True:
            await asyncio.sleep(3600.0)
    except asyncio.CancelledError:  # pragma: no cover
        LOG.info("mm_lite idle loop cancelled")
        raise


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    from .config import DEFAULT_CONFIG
    from workers.common import addon_live

    cfg = addon_live.apply_env_overrides(
        "MM_LITE",
        DEFAULT_CONFIG,
        default_universe=["USD_JPY"],
        default_pocket="scalp",
        default_loop=5.0,
        default_exit={"stop_pips": 2.0, "tp_pips": 3.0},
    )
    if not cfg.get("live_enabled"):
        LOG.info("mm_lite idle mode (set MM_LITE_LIVE=1 or ADDON_LIVE_MODE=1 to enable)")
        asyncio.run(_idle_loop())
        raise SystemExit(0)

    datafeed = addon_live.LiveDataFeed(default_timeframe=cfg.get("timeframe", "M5"), logger=LOG)
    broker = addon_live.AddonLiveBroker(
        worker_id=str(cfg.get("id", "mm_lite")),
        pocket=str(cfg.get("pocket", "scalp")),
        datafeed=datafeed,
        exit_cfg=cfg.get("exit"),
        atr_len=int(cfg.get("atr_len", 14)),
        atr_timeframe=cfg.get("timeframe", "M5"),
        default_budget_bps=cfg.get("budget_bps"),
        default_size_bps=cfg.get("size_bps"),
        ttl_ms=float(cfg.get("ttl_ms", 800.0)),
        require_passive=bool(cfg.get("require_passive", True)),
        logger=LOG,
    )
    worker = MMLiteWorker(cfg, broker=broker, datafeed=datafeed, logger=LOG)
    addon_live.run_loop(
        worker,
        loop_interval_sec=float(cfg.get("loop_interval_sec", 5.0)),
        pocket=str(cfg.get("pocket", "scalp")),
        logger=LOG,
    )
