from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, Tuple
from dataclasses import dataclass
import time, math, datetime as dt
from zoneinfo import ZoneInfo
import asyncio
import logging

# ---- minimal protocols ----
class DataFeed(Protocol):
    def get_bars(self, symbol: str, tf: str, n: int) -> Any: ...
    # bar dicts should include 'timestamp' (epoch seconds) when possible
    def last(self, symbol: str) -> float: ...
    def best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]: ...  # optional

class Broker(Protocol):
    def send(self, order: Dict[str, Any]) -> Any: ...

from workers.common.exit_adapter import build_exit_manager

def _as_bars(bars: Any) -> List[Dict[str, float]]:
    out = []
    if not bars: return out
    if isinstance(bars, list) and bars and isinstance(bars[0], dict):
        for b in bars:
            d = {
                "open": float(b.get("open", b.get("o", 0))),
                "high": float(b.get("high", b.get("h", 0))),
                "low": float(b.get("low", b.get("l", 0))),
                "close": float(b.get("close", b.get("c", 0))),
            }
            ts = b.get("timestamp") or b.get("ts") or b.get("time")
            d["timestamp"] = float(ts) if ts is not None else float("nan")
            out.append(d)
        return out
    # fallback: cannot parse structure
    return out

def _atr(b: List[Dict[str, float]], n: int = 14) -> float:
    if len(b) < n + 1: return 0.0
    trs = []
    for i in range(1, n + 1):
        h, l = b[-i]["high"], b[-i]["low"]
        pc = b[-i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / len(trs)

def _initial_range(bars: List[Dict[str, float]], t0: float, t1: float) -> Optional[Dict[str, float]]:
    seg = [x for x in bars if not math.isnan(x["timestamp"]) and t0 <= x["timestamp"] <= t1]
    if len(seg) < 2: 
        return None
    return {
        "high": max(x["high"] for x in seg),
        "low":  min(x["low"]  for x in seg),
        "n": len(seg)
    }

def _today_session_start(now: float, start_hhmm: str, tz: str) -> float:
    tzinfo = ZoneInfo(tz)
    dtnow = dt.datetime.fromtimestamp(now, tzinfo)
    hh, mm = map(int, start_hhmm.split(":"))
    session = dtnow.replace(hour=hh, minute=mm, second=0, microsecond=0)
    # if already past next day (unlikely), adjust – also handle if we are before start (then use today's start)
    return session.timestamp()

def _bps(a: float, b: float) -> float:
    if b == 0: return 0.0
    return (a / b - 1.0) * 1e4

class SessionOpenWorker:
    """
    Build an initial range during the first X minutes of each configured session window.
    Trade a breakout of that range with small padding in bps.
    """
    def __init__(self, cfg: Dict[str, Any], broker: Optional[Broker], datafeed: DataFeed, logger: Any = None):
        self.c = cfg
        self.b = broker
        self.d = datafeed
        self.log = logger
        self._last_entry_bar = {}  # symbol -> idx

        self.tf = self.c.get("timeframe_entry", "1m")
        self.cooldown = int(self.c.get("cooldown_bars", 3))
        self.pad_bp = float(self.c.get("pad_bp", 2.0))
        self.place_orders = bool(self.c.get("place_orders", False))
        self.exit_cfg = self.c.get("exit", {})
        self.filters = self.c.get("filters", {})
        self.sessions = self.c.get("sessions", [])

        self.exit_mgr = build_exit_manager(self.exit_cfg)

    def run_once(self, now: Optional[float] = None):
        now = time.time() if now is None else float(now)
        intents = []
        for sym in self.c.get("universe", []):
            ti = self.edge(sym, now)
            if not ti: 
                continue
            intents.append(ti)
            if self.place_orders and self.b is not None:
                order = self._mk_order(sym, ti)
                if self.exit_mgr:
                    order = self.exit_mgr.attach(order)
                self.b.send(order)
        return intents

    def edge(self, sym: str, now: float) -> Optional[Dict[str, Any]]:
        # bars for entry tf
        bars = _as_bars(self.d.get_bars(sym, self.tf, max(self.filters.get("min_bars", 60), 200)))
        if len(bars) < self.filters.get("min_bars", 60):
            return None

        # best bid/ask for spread filter (optional)
        ba = None
        try:
            ba = self.d.best_bid_ask(sym)
        except Exception:
            pass
        if ba and self.filters.get("max_spread_bp") is not None:
            bid, ask = ba
            if bid and ask and bid > 0:
                sp_bp = (ask / bid - 1.0) * 1e4
                if sp_bp > float(self.filters["max_spread_bp"]):
                    return None

        last = bars[-1]["close"]
        atr = _atr(bars, n=14)
        min_atr = float(self.filters.get("min_atr", 0.0) or 0.0)
        if atr < min_atr:
            return None

        for ses in self.sessions:
            t0 = _today_session_start(now, ses["start"], ses.get("tz", "UTC"))
            t_build_end = t0 + ses.get("build_minutes", 15) * 60
            t_hold_end = t0 + ses.get("hold_minutes", 120) * 60

            # use today's session if we are within build..hold window
            if not (t_build_end <= now <= t_hold_end):
                continue

            rng = _initial_range(bars, t0, t_build_end)
            if not rng or rng["n"] < 2:
                continue

            hi = rng["high"]
            lo = rng["low"]
            up_trig = hi * (1 + self.pad_bp / 1e4)
            dn_trig = lo * (1 - self.pad_bp / 1e4)

            # cooldown on bar index edge:
            if len(bars) - 1 == self._last_entry_bar.get(sym, -999):
                continue

            if last > up_trig:
                self._last_entry_bar[sym] = len(bars) - 1
                return {"symbol": sym, "side": "long", "px": last, "meta": {"atr": atr, "rng": (lo, hi), "session": ses}}
            if last < dn_trig:
                self._last_entry_bar[sym] = len(bars) - 1
                return {"symbol": sym, "side": "short", "px": last, "meta": {"atr": atr, "rng": (lo, hi), "session": ses}}

        return None

    def _mk_order(self, sym: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        side = "buy" if intent["side"] == "long" else "sell"
        px = float(intent.get("px") or self.d.last(sym))
        size = max(0.0, float(self.c.get("budget_bps", 25)) / 10000.0)
        return {
            "symbol": sym, "side": side, "type": "market", "size": size,
            "meta": {"worker_id": self.c.get("id", "session_open_breakout"), "intent": intent}
        }


async def _idle_loop() -> None:
    """
    Placeholder loop to keep the service alive when no datafeed/broker wiring exists.
    SessionOpenWorker expects external feed/broker; until接続するまでは idle で待機させる。
    """
    log = logging.getLogger("session_open")
    while True:
        log.info("[SESSION_OPEN] inactive (no datafeed/broker configured); sleeping 300s")
        await asyncio.sleep(300)


if __name__ == "__main__":  # pragma: no cover - service entrypoint
    try:
        asyncio.run(_idle_loop())
    except KeyboardInterrupt:
        pass
