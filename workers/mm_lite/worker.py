from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, Tuple
import math, time, asyncio, logging, os
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from indicators.factor_cache import all_factors, get_candles_snapshot

class DataFeed(Protocol):
    def get_bars(self, symbol: str, tf: str, n: int) -> Any: ...
    def last(self, symbol: str) -> float: ...
    def best_bid_ask(self, symbol: str) -> Optional[Tuple[float, float]]: ...  # optional

class Broker(Protocol):
    def send(self, order: Dict[str, Any]) -> Any: ...
    def cancel(self, order_id: str) -> Any: ...

class EventFlagSource(Protocol):
    def is_event_now(self) -> bool: ...  # e.g., macro event window


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

BB_STYLE = "reversion"

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
        self.refresh_sec = float(self.c.get("refresh_sec", 15.0))
        self.replace_bp = float(self.c.get("replace_bp", 0.4))
        self.max_quote_age_sec = float(self.c.get("max_quote_age_sec", 90.0))

    def run_once(self):
        intents = []
        for sym in self.c.get("universe", []):
            it = self._quote(sym)
            if it: intents.append(it)
        return intents

    def _quote(self, sym: str):
        now = time.time()
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
            pair = self.orders.get(sym, {})
            last_ts = float(pair.get("ts", 0.0) or 0.0)
            age = now - last_ts if last_ts > 0 else 1e9
            if pair.get("bid_id") or pair.get("ask_id"):
                if age < self.refresh_sec:
                    return {"symbol": sym, "bid": bid, "ask": ask, "mid": mid, "atr_bp": atr_bp, "half_spread_bp": half_bp}
                try:
                    old_bid = float(pair.get("bid_px") or 0.0)
                    old_ask = float(pair.get("ask_px") or 0.0)
                except Exception:
                    old_bid = 0.0
                    old_ask = 0.0
                if old_bid > 0 and old_ask > 0 and age < self.max_quote_age_sec:
                    bid_diff = abs(bid / old_bid - 1.0) * 1e4
                    ask_diff = abs(ask / old_ask - 1.0) * 1e4
                    if bid_diff < self.replace_bp and ask_diff < self.replace_bp:
                        return {"symbol": sym, "bid": bid, "ask": ask, "mid": mid, "atr_bp": atr_bp, "half_spread_bp": half_bp}

            # Cancel & replace strategy (throttled)
            for oid in [pair.get("bid_id"), pair.get("ask_id")]:
                if oid:
                    try: self.b.cancel(oid)
                    except Exception: pass
            bid_id = f"{sym}-bid-{time.time()}"
            ask_id = f"{sym}-ask-{time.time()}"
            long_allow, long_mult, _ = _projection_decision("long", "scalp", mode_override="range")
            short_allow, short_mult, _ = _projection_decision("short", "scalp", mode_override="range")
            bid_size = notional * long_mult if long_allow else 0.0
            ask_size = notional * short_mult if short_allow else 0.0
            fac_m1 = all_factors().get("M1") or {}
            if bid_size > 0.0 and not _bb_entry_allowed(BB_STYLE, "long", bid, fac_m1):
                bid_size = 0.0
            if ask_size > 0.0 and not _bb_entry_allowed(BB_STYLE, "short", ask, fac_m1):
                ask_size = 0.0
            if bid_size > 0.0:
                self.b.send({"id": bid_id, "symbol": sym, "side": "buy", "type": "limit", "price": bid, "size": bid_size})
            if ask_size > 0.0:
                self.b.send({"id": ask_id, "symbol": sym, "side": "sell", "type": "limit", "price": ask, "size": ask_size})
            self.orders[sym] = {"bid_id": bid_id if bid_size > 0.0 else None, "ask_id": ask_id if ask_size > 0.0 else None, "bid_px": bid, "ask_px": ask, "ts": now}

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
