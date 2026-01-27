from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, Tuple
from dataclasses import dataclass
import time, math, datetime as dt
from zoneinfo import ZoneInfo
import asyncio
import logging
import os
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from indicators.factor_cache import all_factors, get_candles_snapshot

# ---- minimal protocols ----
class DataFeed(Protocol):
    def get_bars(self, symbol: str, tf: str, n: int) -> Any: ...
    # bar dicts should include 'timestamp' (epoch seconds) when possible
    def last(self, symbol: str) -> float: ...
    def best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]: ...  # optional

class Broker(Protocol):
    def send(self, order: Dict[str, Any]) -> Any: ...

from workers.common.exit_adapter import build_exit_manager


_BB_ENTRY_ENABLED = os.getenv("BB_ENTRY_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
_BB_ENTRY_REVERT_PIPS = float(os.getenv("BB_ENTRY_REVERT_PIPS", "2.4"))
_BB_ENTRY_REVERT_RATIO = float(os.getenv("BB_ENTRY_REVERT_RATIO", "0.22"))
_BB_ENTRY_TREND_EXT_PIPS = float(os.getenv("BB_ENTRY_TREND_EXT_PIPS", "3.5"))
_BB_ENTRY_TREND_EXT_RATIO = float(os.getenv("BB_ENTRY_TREND_EXT_RATIO", "0.40"))
_BB_ENTRY_SCALP_REVERT_PIPS = float(os.getenv("BB_ENTRY_SCALP_REVERT_PIPS", "2.0"))
_BB_ENTRY_SCALP_REVERT_RATIO = float(os.getenv("BB_ENTRY_SCALP_REVERT_RATIO", "0.20"))
_BB_ENTRY_SCALP_EXT_PIPS = float(os.getenv("BB_ENTRY_SCALP_EXT_PIPS", "2.4"))
_BB_ENTRY_SCALP_EXT_RATIO = float(os.getenv("BB_ENTRY_SCALP_EXT_RATIO", "0.30"))
_BB_PIP = 0.01


def _bb_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bb_levels(fac):
    if not fac:
        return None
    upper = _bb_float(fac.get("bb_upper"))
    lower = _bb_float(fac.get("bb_lower"))
    mid = _bb_float(fac.get("bb_mid")) or _bb_float(fac.get("ma20"))
    bbw = _bb_float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span / _BB_PIP


def _bb_entry_allowed(style, side, price, fac_m1, *, range_active=None):
    if not _BB_ENTRY_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac_m1)
    if not levels:
        return True
    upper, mid, lower, span, span_pips = levels
    side_key = str(side or "").lower()
    if side_key in {"buy", "long", "open_long"}:
        direction = "long"
    else:
        direction = "short"
    orig_style = style
    if style == "scalp" and range_active:
        style = "reversion"
    if style == "reversion":
        base_pips = _BB_ENTRY_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_ENTRY_REVERT_PIPS
        base_ratio = _BB_ENTRY_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_ENTRY_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist <= threshold
    if direction == "long":
        if price < mid:
            return False
        ext = max(0.0, price - upper) / _BB_PIP
    else:
        if price > mid:
            return False
        ext = max(0.0, lower - price) / _BB_PIP
    max_ext = max(_BB_ENTRY_TREND_EXT_PIPS, span_pips * _BB_ENTRY_TREND_EXT_RATIO)
    if orig_style == "scalp":
        max_ext = max(_BB_ENTRY_SCALP_EXT_PIPS, span_pips * _BB_ENTRY_SCALP_EXT_RATIO)
    return ext <= max_ext

BB_STYLE = "trend"

_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}


def _projection_mode(pocket, mode_override=None):
    if mode_override:
        return mode_override
    if globals().get("IS_RANGE"):
        return "range"
    if globals().get("IS_PULLBACK"):
        return "pullback"
    if pocket in {"scalp", "scalp_fast"}:
        return "scalp"
    return "trend"


def _projection_tfs(pocket, mode):
    if pocket == "macro":
        return ("H4", "H1")
    if pocket == "micro":
        return ("M5", "M1")
    if pocket in {"scalp", "scalp_fast"}:
        return ("M1",)
    return ("M5", "M1")


def _projection_candles(tfs):
    for tf in tfs:
        candles = get_candles_snapshot(tf, limit=120)
        if candles and len(candles) >= 30:
            return tf, list(candles)
    return None, None


def _score_ma(ma, side, opp_block_bars):
    if ma is None:
        return None
    align = ma.gap_pips >= 0 if side == "long" else ma.gap_pips <= 0
    cross_soon = ma.projected_cross_bars is not None and ma.projected_cross_bars <= opp_block_bars
    if align and not cross_soon:
        return 0.7
    if align and cross_soon:
        return -0.4
    if cross_soon:
        return -0.8
    return -0.5


def _score_rsi(rsi, side, long_target, short_target, overheat_bars):
    if rsi is None:
        return None
    score = 0.0
    if side == "long":
        if rsi.rsi >= long_target and rsi.slope_per_bar > 0:
            score = 0.4
        elif rsi.rsi <= (long_target - 8) and rsi.slope_per_bar < 0:
            score = -0.4
        if rsi.eta_upper_bars is not None and rsi.eta_upper_bars <= overheat_bars:
            score -= 0.2
    else:
        if rsi.rsi <= short_target and rsi.slope_per_bar < 0:
            score = 0.4
        elif rsi.rsi >= (short_target + 8) and rsi.slope_per_bar > 0:
            score = -0.4
        if rsi.eta_lower_bars is not None and rsi.eta_lower_bars <= overheat_bars:
            score -= 0.2
    return score


def _score_adx(adx, trend_mode, threshold):
    if adx is None:
        return None
    if trend_mode:
        if adx.adx >= threshold and adx.slope_per_bar >= 0:
            return 0.4
        if adx.adx <= threshold and adx.slope_per_bar < 0:
            return -0.4
        return 0.0
    if adx.adx >= threshold and adx.slope_per_bar > 0:
        return -0.5
    if adx.adx <= threshold and adx.slope_per_bar < 0:
        return 0.3
    return 0.0


def _score_bbw(bbw, threshold):
    if bbw is None:
        return None
    if bbw.bbw <= threshold and bbw.slope_per_bar <= 0:
        return 0.5
    if bbw.bbw > threshold and bbw.slope_per_bar > 0:
        return -0.5
    return 0.0


def _projection_decision(side, pocket, mode_override=None):
    mode = _projection_mode(pocket, mode_override=mode_override)
    tfs = _projection_tfs(pocket, mode)
    tf, candles = _projection_candles(tfs)
    if not candles:
        return True, 1.0, {}
    minutes = _PROJ_TF_MINUTES.get(tf, 1.0)

    if mode == "trend":
        params = {
            "adx_threshold": 20.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 5.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.45, "rsi": 0.25, "adx": 0.30},
            "block_score": -0.6,
            "size_scale": 0.18,
        }
    elif mode == "pullback":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 4.0,
            "long_target": 50.0,
            "short_target": 50.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.40, "rsi": 0.40, "adx": 0.20},
            "block_score": -0.55,
            "size_scale": 0.15,
        }
    elif mode == "scalp":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 3.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 2.0,
            "weights": {"ma": 0.50, "rsi": 0.30, "adx": 0.20},
            "block_score": -0.6,
            "size_scale": 0.12,
        }
    else:
        params = {
            "adx_threshold": 16.0,
            "bbw_threshold": 0.14,
            "opp_block_bars": 4.0,
            "long_target": 45.0,
            "short_target": 55.0,
            "overheat_bars": 3.0,
            "weights": {"bbw": 0.40, "rsi": 0.35, "adx": 0.25},
            "block_score": -0.5,
            "size_scale": 0.15,
        }

    ma = compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
    rsi = compute_rsi_projection(candles, timeframe_minutes=minutes)
    adx = compute_adx_projection(candles, timeframe_minutes=minutes, trend_threshold=params["adx_threshold"])
    bbw = None
    if mode == "range":
        bbw = compute_bbw_projection(candles, timeframe_minutes=minutes, squeeze_threshold=params["bbw_threshold"])

    scores = {}
    ma_score = _score_ma(ma, side, params["opp_block_bars"])
    if ma_score is not None and "ma" in params["weights"]:
        scores["ma"] = ma_score
    rsi_score = _score_rsi(rsi, side, params["long_target"], params["short_target"], params["overheat_bars"])
    if rsi_score is not None and "rsi" in params["weights"]:
        scores["rsi"] = rsi_score
    adx_score = _score_adx(adx, mode != "range", params["adx_threshold"])
    if adx_score is not None and "adx" in params["weights"]:
        scores["adx"] = adx_score
    bbw_score = _score_bbw(bbw, params["bbw_threshold"])
    if bbw_score is not None and "bbw" in params["weights"]:
        scores["bbw"] = bbw_score

    weight_sum = 0.0
    score_sum = 0.0
    for key, score in scores.items():
        weight = params["weights"].get(key, 0.0)
        weight_sum += weight
        score_sum += weight * score
    score = score_sum / weight_sum if weight_sum > 0 else 0.0

    allow = score > params["block_score"]
    size_mult = 1.0 + max(0.0, score) * params["size_scale"]
    size_mult = max(0.8, min(1.35, size_mult))

    detail = {
        "mode": mode,
        "tf": tf,
        "score": round(score, 3),
        "size_mult": round(size_mult, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
    return allow, size_mult, detail
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
            pocket = str(self.c.get("pocket", "micro"))
            proj_allow, proj_mult, proj_detail = _projection_decision(ti["side"], pocket)
            if not proj_allow:
                continue
            if proj_detail:
                ti.setdefault("meta", {})["projection"] = proj_detail
            intents.append(ti)
            if self.place_orders and self.b is not None:
                order = self._mk_order(sym, ti, proj_mult)
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
                fac_m1 = all_factors().get("M1") or {}
                if not _bb_entry_allowed(BB_STYLE, "long", last, fac_m1):
                    continue
                self._last_entry_bar[sym] = len(bars) - 1
                return {"symbol": sym, "side": "long", "px": last, "meta": {"atr": atr, "rng": (lo, hi), "session": ses}}
            if last < dn_trig:
                fac_m1 = all_factors().get("M1") or {}
                if not _bb_entry_allowed(BB_STYLE, "short", last, fac_m1):
                    continue
                self._last_entry_bar[sym] = len(bars) - 1
                return {"symbol": sym, "side": "short", "px": last, "meta": {"atr": atr, "rng": (lo, hi), "session": ses}}

        return None

    def _mk_order(self, sym: str, intent: Dict[str, Any], size_mult: float = 1.0) -> Dict[str, Any]:
        side = "buy" if intent["side"] == "long" else "sell"
        px = float(intent.get("px") or self.d.last(sym))
        size = max(0.0, float(self.c.get("budget_bps", 25)) / 10000.0)
        if size_mult > 1.0:
            size *= size_mult
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    from .config import DEFAULT_CONFIG
    from workers.common import addon_live

    cfg = addon_live.apply_env_overrides(
        "SESSION_OPEN",
        DEFAULT_CONFIG,
        default_universe=["USD_JPY"],
        default_pocket="micro",
        default_loop=20.0,
    )
    if not cfg.get("live_enabled"):
        LOG = logging.getLogger("session_open")
        LOG.info("session_open idle mode (set SESSION_OPEN_LIVE=1 or ADDON_LIVE_MODE=1 to enable)")
        try:
            asyncio.run(_idle_loop())
        except KeyboardInterrupt:
            pass
        raise SystemExit(0)

    datafeed = addon_live.LiveDataFeed(
        default_timeframe=str(cfg.get("timeframe_entry", "M1")),
        logger=logging.getLogger("session_open"),
    )
    broker = addon_live.AddonLiveBroker(
        worker_id=str(cfg.get("id", "session_open")),
        pocket=str(cfg.get("pocket", "micro")),
        datafeed=datafeed,
        exit_cfg=cfg.get("exit"),
        atr_len=int(cfg.get("atr_len", 14)),
        atr_timeframe=str(cfg.get("timeframe_entry", "M1")),
        default_budget_bps=cfg.get("budget_bps"),
        ttl_ms=float(cfg.get("ttl_ms", 800.0)),
        require_passive=bool(cfg.get("require_passive", True)),
        logger=logging.getLogger("session_open"),
    )
    worker = SessionOpenWorker(cfg, broker=broker, datafeed=datafeed, logger=logging.getLogger("session_open"))
    addon_live.run_loop(
        worker,
        loop_interval_sec=float(cfg.get("loop_interval_sec", 20.0)),
        pocket=str(cfg.get("pocket", "micro")),
        pass_now=True,
        logger=logging.getLogger("session_open"),
    )
