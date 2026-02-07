"""Entry loop for MicroPullbackEMA (micro pocket)."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection



import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis import perf_monitor
from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors, get_candles_snapshot, refresh_cache_from_disk
from market_data import tick_window
from strategies.micro.pullback_ema import MicroPullbackEMA
from utils.divergence import apply_divergence_confidence, divergence_bias, divergence_snapshot
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common.dyn_cap import compute_cap
from workers.common import perf_guard

from workers.common.size_utils import scale_base_units

from . import config

import os
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

BB_STYLE = "trend"

LOG = logging.getLogger(__name__)

MR_RANGE_LOOKBACK = 20
MR_RANGE_HI_PCT = 95.0
MR_RANGE_LO_PCT = 5.0

STRATEGY = MicroPullbackEMA
IS_TREND = True
IS_PULLBACK = True
IS_RANGE = False



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
def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=10.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_val = tick.get("mid")
        if mid_val is not None:
            try:
                return float(mid_val)
            except Exception:
                pass
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except Exception:
                return fallback
    return fallback


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "micro"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-micro-{sanitized}{digest}"


def _confidence_scale(conf: int, *, lo: int, hi: int) -> float:
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _compute_cap(*, cap_min: float, cap_max: float, **kwargs) -> Tuple[float, Dict[str, float]]:
    kwargs.setdefault("env_prefix", config.ENV_PREFIX)
    res = compute_cap(cap_min=cap_min, cap_max=cap_max, **kwargs)
    return res.cap, res.reasons


def _is_mr_signal(tag: str) -> bool:
    tag_str = (tag or "").strip()
    if not tag_str:
        return False
    base_tag = tag_str.split("-", 1)[0]
    if base_tag in {"MicroVWAPBound", "BB_RSI"}:
        return True
    lower = tag_str.lower()
    return lower.startswith("mlr-fade") or lower.startswith("mlr-bounce")


def _build_mr_entry_thesis(
    signal: Dict,
    *,
    strategy_tag: str,
    atr_entry: float,
    entry_mean: Optional[float],
) -> Dict:
    thesis: Dict[str, object] = {
        "strategy_tag": strategy_tag,
        "profile": signal.get("profile"),
        "confidence": signal.get("confidence", 0),
        "env_tf": "H1",
        "struct_tf": "M5",
        "range_method": "percentile",
        "range_lookback": MR_RANGE_LOOKBACK,
        "range_hi_pct": MR_RANGE_HI_PCT,
        "range_lo_pct": MR_RANGE_LO_PCT,
        "atr_entry": atr_entry,
        "structure_break": {"buffer_atr": 0.10, "confirm_closes": 2},
        "tp_mode": "soft_zone",
        "tp_target": "entry_mean",
        "tp_pad_atr": 0.05,
        "reversion_failure": {
            "z_ext": 0.55,
            "contraction_min": 0.50,
            "bars_budget": {"k_per_z": 3.5, "min": 2, "max": 12},
            "trend_takeover": {"require_env_trend_bars": 2},
        },
    }
    if entry_mean is not None and entry_mean > 0:
        thesis["entry_mean"] = float(entry_mean)
    candles = get_candles_snapshot("H1", limit=MR_RANGE_LOOKBACK)
    snapshot = compute_range_snapshot(
        candles,
        lookback=MR_RANGE_LOOKBACK,
        method="percentile",
        hi_pct=MR_RANGE_HI_PCT,
        lo_pct=MR_RANGE_LO_PCT,
    )
    if snapshot:
        thesis["range_snapshot"] = snapshot.to_dict()
        thesis.setdefault("entry_mean", snapshot.mid)
    return thesis


def _factor_age_seconds(factors: Dict[str, float]) -> float:
    ts_raw = factors.get("timestamp") if isinstance(factors, dict) else None
    if not ts_raw:
        return float("inf")
    try:
        if isinstance(ts_raw, (int, float)):
            ts_val = float(ts_raw)
            ts_dt = datetime.datetime.utcfromtimestamp(ts_val).replace(tzinfo=datetime.timezone.utc)
        else:
            ts_txt = str(ts_raw)
            if ts_txt.endswith("Z"):
                ts_txt = ts_txt.replace("Z", "+00:00")
            ts_dt = datetime.datetime.fromisoformat(ts_txt)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return float("inf")
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (now - ts_dt).total_seconds())


async def micro_pullbackema_worker() -> None:
    enabled = getattr(config, "ENABLED", True)
    strategy_name = getattr(STRATEGY, "name", STRATEGY.__name__)
    log_prefix = getattr(config, "LOG_PREFIX", f"[{strategy_name}]")
    if not enabled:
        LOG.info("%s disabled", log_prefix)
        return
    loop_interval = float(getattr(config, "LOOP_INTERVAL_SEC", 8.0))
    pocket = getattr(config, "POCKET", "micro")
    min_units = int(getattr(config, "MIN_UNITS", 1500))
    base_units_cfg = int(getattr(config, "BASE_ENTRY_UNITS", 20000))
    conf_floor = int(getattr(config, "CONFIDENCE_FLOOR", 35))
    conf_ceil = int(getattr(config, "CONFIDENCE_CEIL", 90))
    cap_min = float(getattr(config, "CAP_MIN", 0.15))
    cap_max = float(getattr(config, "CAP_MAX", 0.95))
    range_only_score = float(getattr(config, "RANGE_ONLY_SCORE", 0.45))
    max_factor_age = float(getattr(config, "MAX_FACTOR_AGE_SEC", 90.0))
    trend_block_hours = getattr(config, "TREND_BLOCK_HOURS_UTC", frozenset())

    LOG.info("%s worker start (interval=%.1fs)", log_prefix, loop_interval)
    last_stale_log = 0.0
    last_perf_block_log = 0.0

    while True:
        await asyncio.sleep(loop_interval)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(pocket):
            continue
        if IS_TREND and trend_block_hours and now.hour in trend_block_hours:
            continue

        try:
            refresh_cache_from_disk()
        except Exception:
            pass
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        fac_m5 = factors.get("M5") or {}
        age_m1 = _factor_age_seconds(fac_m1)
        if age_m1 > max_factor_age:
            if time.time() - last_stale_log > 30.0:
                LOG.warning(
                    "%s stale factors age=%.1fs limit=%.1fs (proceeding)",
                    log_prefix,
                    age_m1,
                    max_factor_age,
                )
                last_stale_log = time.time()

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        range_only = range_ctx.active or range_score >= range_only_score
        if range_only and not IS_RANGE:
            continue
        fac_m1 = dict(fac_m1)
        fac_m1["range_active"] = bool(range_ctx.active)
        fac_m1["range_score"] = range_score
        fac_m1["range_reason"] = range_ctx.reason
        fac_m1["range_mode"] = range_ctx.mode

        signal = STRATEGY.check(fac_m1)
        if not signal:
            continue
        div_bias = divergence_bias(
            fac_m1,
            signal.get("action") or "",
            mode="trend",
            max_age_bars=12,
        )
        if div_bias:
            base_conf = int(signal.get("confidence", 0) or 0)
            signal["confidence"] = apply_divergence_confidence(
                base_conf,
                div_bias,
                max_bonus=6.0,
                max_penalty=8.0,
                floor=40.0,
                ceil=95.0,
            )
        div_meta = divergence_snapshot(fac_m1, max_age_bars=12)
        perf_decision = perf_guard.is_allowed(strategy_name, pocket, env_prefix=config.ENV_PREFIX)
        if not perf_decision.allowed:
            now_mono = time.monotonic()
            if now_mono - last_perf_block_log > 120.0:
                LOG.info(
                    "%s perf_block tag=%s reason=%s",
                    log_prefix,
                    strategy_name,
                    perf_decision.reason,
                )
                last_perf_block_log = now_mono
            continue

        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(pocket) or {}).get("pf"))
        except Exception:
            pf = None

        snap = get_account_snapshot()
        equity = float(snap.nav or snap.balance or 0.0)

        balance = float(snap.balance or snap.nav or 0.0)
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
        try:
            atr_m5 = float(fac_m5.get("atr_pips") or 0.0)
        except Exception:
            atr_m5 = 0.0
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            micro_pos = open_positions.get("micro") or {}
            pos_bias = abs(float(micro_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
        except Exception:
            pos_bias = 0.0

        cap, cap_reason = _compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=range_ctx.active,
            perf_pf=pf,
            pos_bias=pos_bias,
            cap_min=cap_min,
            cap_max=cap_max,
        )
        if cap <= 0.0:
            continue

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        if price <= 0.0:
            continue

        long_units = 0.0
        short_units = 0.0
        try:
            long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
        except Exception:
            long_units, short_units = 0.0, 0.0

        side = "long" if signal.get("action") == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if sl_pips <= 0.0:
            continue
        # BB_ENTRY_GUARD
        if not _bb_entry_allowed(BB_STYLE, side, price, fac_m1, range_active=range_ctx.active):
            continue

        tp_scale = 10.0 / max(1.0, tp_pips)
        tp_scale = max(0.4, min(1.1, tp_scale))
        base_units = int(
            round(
                scale_base_units(
                    base_units_cfg,
                    equity=balance if balance > 0 else equity,
                    ref_equity=balance,
                    env_prefix=config.ENV_PREFIX,
                )
                * tp_scale
            )
        )
        conf_scale = _confidence_scale(int(signal.get("confidence", 50)), lo=conf_floor, hi=conf_ceil)
        signal_tag = signal.get("tag", strategy_name)
        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=pocket,
            side=side,
            open_long_units=long_units,
            open_short_units=short_units,
            strategy_tag=signal_tag,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units_risk = int(round(lot * 100000))
        units = int(round(base_units * conf_scale))
        units = min(units, units_risk)
        units = int(round(units * cap))
        if units < min_units:
            continue
        if side == "short":
            units = -abs(units)

        if side == "long":
            sl_price = round(price - sl_pips * 0.01, 3)
            tp_price = round(price + tp_pips * 0.01, 3) if tp_pips > 0 else None
        else:
            sl_price = round(price + sl_pips * 0.01, 3)
            tp_price = round(price - tp_pips * 0.01, 3) if tp_pips > 0 else None
        sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=side == "long")

        client_id = _client_order_id(signal_tag)
        entry_thesis: Dict[str, object] = {
            "strategy_tag": signal_tag,
            "env_prefix": config.ENV_PREFIX,
            "profile": signal.get("profile"),
            "confidence": signal.get("confidence", 0),
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "hard_stop_pips": sl_pips,
            "range_active": bool(range_ctx.active),
            "range_score": round(range_score, 3),
            "range_reason": range_ctx.reason,
            "range_mode": range_ctx.mode,
        }
        if div_meta:
            entry_thesis["divergence"] = div_meta
        if IS_TREND:
            entry_thesis["entry_tf"] = "M5"
        if IS_PULLBACK:
            entry_thesis["entry_guard_pullback"] = True
            entry_thesis["entry_guard_pullback_only"] = True
        if _is_mr_signal(signal_tag):
            entry_mean = None
            base_tag = signal_tag.split("-", 1)[0] if signal_tag else ""
            if base_tag == "MicroVWAPBound":
                notes = signal.get("notes") or {}
                if isinstance(notes, dict):
                    try:
                        entry_mean = float(notes.get("vwap"))
                    except Exception:
                        entry_mean = None
            elif base_tag == "BB_RSI":
                try:
                    entry_mean = float(fac_m1.get("ma20") or fac_m1.get("ma10") or 0.0) or None
                except Exception:
                    entry_mean = None
            entry_thesis.update(
                _build_mr_entry_thesis(
                    signal,
                    strategy_tag=signal_tag,
                    atr_entry=atr_m5 or atr_pips or 1.0,
                    entry_mean=entry_mean,
                )
            )
            if base_tag == "MicroVWAPBound":
                notes = signal.get("notes") or {}
                z_val = None
                if isinstance(notes, dict):
                    try:
                        z_val = abs(float(notes.get("z")))
                    except Exception:
                        z_val = None
                rf = entry_thesis.get("reversion_failure")
                if isinstance(rf, dict) and z_val is not None:
                    bars_budget = rf.get("bars_budget")
                    if not isinstance(bars_budget, dict):
                        bars_budget = {}
                        rf["bars_budget"] = bars_budget
                    if z_val >= 2.5:
                        bars_budget["k_per_z"] = 4.0
                        bars_budget["max"] = 14
                    elif z_val <= 1.4:
                        bars_budget["k_per_z"] = 3.0
                        bars_budget["max"] = 10

        proj_allow, proj_mult, proj_detail = _projection_decision(side, pocket)
        if not proj_allow:
            continue
        if proj_detail:
            entry_thesis["projection"] = proj_detail
        if proj_mult > 1.0:
            sign = 1 if units > 0 else -1
            units = int(round(abs(units) * proj_mult)) * sign

        candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
        if not candle_allow:
            continue
        if candle_mult != 1.0:
            sign = 1 if units > 0 else -1
            units = int(round(abs(units) * candle_mult)) * sign
        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=pocket,
            client_order_id=client_id,
            strategy_tag=signal_tag,
            confidence=int(signal.get("confidence", 0)),
            entry_thesis=entry_thesis,
        )
        LOG.info(
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
            log_prefix,
            units,
            side,
            price,
            sl_price or 0.0,
            tp_price or 0.0,
            signal.get("confidence", 0),
            cap,
            cap_reason,
            res or "none",
        )





_CANDLE_PIP = 0.01
_CANDLE_MIN_CONF = 0.35
_CANDLE_ENTRY_BLOCK = -0.7
_CANDLE_ENTRY_SCALE = 0.2
_CANDLE_ENTRY_MIN = 0.8
_CANDLE_ENTRY_MAX = 1.2
_CANDLE_WORKER_NAME = (__file__.replace("\\", "/").split("/")[-2] if "/" in __file__ else "").lower()


def _candle_tf_for_worker() -> str:
    name = _CANDLE_WORKER_NAME
    if "macro" in name or "trend_h1" in name or "manual" in name:
        return "H1"
    if "scalp" in name or "s5" in name or "fast" in name:
        return "M1"
    return "M5"


def _extract_candles(raw):
    candles = []
    for candle in raw or []:
        try:
            o = float(candle.get("open", candle.get("o")))
            h = float(candle.get("high", candle.get("h")))
            l = float(candle.get("low", candle.get("l")))
            c = float(candle.get("close", candle.get("c")))
        except Exception:
            continue
        if h <= 0 or l <= 0:
            continue
        candles.append((o, h, l, c))
    return candles


def _detect_candlestick_pattern(candles):
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, _CANDLE_PIP * 0.1)
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1

    if body1 <= range1 * 0.1:
        return {
            "type": "doji",
            "confidence": round(min(1.0, (range1 - body1) / range1), 3),
            "bias": None,
        }

    if (
        c1 > o1
        and c0 < o0
        and c1 >= max(o0, c0)
        and o1 <= min(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bullish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "up",
        }
    if (
        c1 < o1
        and c0 > o0
        and o1 >= min(o0, c0)
        and c1 <= max(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bearish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "down",
        }
    if lower_wick > body1 * 2.5 and upper_wick <= body1 * 0.6:
        return {
            "type": "hammer" if c1 >= o1 else "inverted_hammer",
            "confidence": round(min(1.0, lower_wick / range1 + 0.25), 3),
            "bias": "up",
        }
    if upper_wick > body1 * 2.5 and lower_wick <= body1 * 0.6:
        return {
            "type": "shooting_star" if c1 <= o1 else "hanging_man",
            "confidence": round(min(1.0, upper_wick / range1 + 0.25), 3),
            "bias": "down",
        }
    return None


def _score_candle(*, candles, side, min_conf):
    pattern = _detect_candlestick_pattern(_extract_candles(candles))
    if not pattern:
        return None, {}
    bias = pattern.get("bias")
    conf = float(pattern.get("confidence") or 0.0)
    if conf < min_conf:
        return None, {"type": pattern.get("type"), "confidence": round(conf, 3)}
    if bias is None:
        return 0.0, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": None}
    match = (side == "long" and bias == "up") or (side == "short" and bias == "down")
    score = conf if match else -conf * 0.7
    score = max(-1.0, min(1.0, score))
    return score, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": bias}


def _entry_candle_guard(side):
    tf = _candle_tf_for_worker()
    candles = get_candles_snapshot(tf, limit=4)
    if not candles:
        return True, 1.0
    score, _detail = _score_candle(candles=candles, side=side, min_conf=_CANDLE_MIN_CONF)
    if score is None:
        return True, 1.0
    if score <= _CANDLE_ENTRY_BLOCK:
        return False, 0.0
    mult = 1.0 + score * _CANDLE_ENTRY_SCALE
    mult = max(_CANDLE_ENTRY_MIN, min(_CANDLE_ENTRY_MAX, mult))
    return True, mult

if __name__ == "__main__":
    asyncio.run(micro_pullbackema_worker())
