from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
import asyncio
import logging
import os
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.ma_projection import score_ma_for_side
from indicators.factor_cache import all_factors, get_candles_snapshot

class DataFeed(Protocol):
    def get_bars(self, symbol: str, tf: str, n: int) -> Any: ...
    def last(self, symbol: str) -> float: ...

class Broker(Protocol):
    def send(self, order: Dict[str, Any]) -> Any: ...

from workers.common.exit_adapter import build_exit_manager

from . import config
from utils.env_utils import env_bool, env_float

_BB_ENV_PREFIX = getattr(config, "ENV_PREFIX", "")
_BB_ENTRY_ENABLED = env_bool("BB_ENTRY_ENABLED", True, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_PIPS = env_float("BB_ENTRY_REVERT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_RATIO = env_float("BB_ENTRY_REVERT_RATIO", 0.22, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_PIPS = env_float("BB_ENTRY_TREND_EXT_PIPS", 3.5, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_RATIO = env_float("BB_ENTRY_TREND_EXT_RATIO", 0.40, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_PIPS = env_float("BB_ENTRY_SCALP_REVERT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_RATIO = env_float("BB_ENTRY_SCALP_REVERT_RATIO", 0.20, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_PIPS = env_float("BB_ENTRY_SCALP_EXT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_RATIO = env_float("BB_ENTRY_SCALP_EXT_RATIO", 0.30, prefix=_BB_ENV_PREFIX)
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

LOG = logging.getLogger(__name__)


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
    return score_ma_for_side(ma, side, opp_block_bars)


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
                pocket = str(self.c.get("pocket", "micro"))
                proj_allow, proj_mult, proj_detail = _projection_decision(
                    ti["side"],
                    pocket,
                    mode_override="range",
                )
                if not proj_allow:
                    continue
                if proj_detail:
                    ti.setdefault("meta", {})["projection"] = proj_detail
                intents.append(ti)
                if self.place_orders and self.b is not None:
                    size = max(0.0, float(self.c.get("budget_bps", 25)) / 10000.0)
                    if proj_mult > 1.0:
                        size *= proj_mult
                    order = {"symbol": sym, "side": "sell" if ti["side"]=="short" else "buy", "type": "market",
                             "size": size,
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
                fac_m1 = all_factors().get("M1") or {}
                if not _bb_entry_allowed(BB_STYLE, "short", bars[-1]["close"], fac_m1):
                    return None
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
                fac_m1 = all_factors().get("M1") or {}
                if not _bb_entry_allowed(BB_STYLE, "long", bars[-1]["close"], fac_m1):
                    return None
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
