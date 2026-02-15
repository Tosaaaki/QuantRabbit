"""H1Momentum dedicated macro worker with dynamic cap by market state."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.technique_engine import evaluate_entry_techniques
from analysis.ma_projection import score_ma_for_side

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors, get_candles_snapshot
from execution.strategy_entry import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.trend.h1_momentum import H1MomentumSwing
from utils.divergence import apply_divergence_confidence, divergence_bias, divergence_snapshot
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.dyn_cap import compute_cap
from workers.common.quality_gate import current_regime
from analysis import perf_monitor

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
_REGIME_GUARD_ENABLED = os.getenv("H1M_REGIME_GUARD_ENABLED", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_H1M_REQUIRE_MACRO_TREND = os.getenv("H1M_REQUIRE_MACRO_TREND", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_H1M_BLOCK_MICRO_RANGE = os.getenv("H1M_BLOCK_MICRO_RANGE", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
}
_H1M_MIN_ATR_PIPS = float(os.getenv("H1M_MIN_ATR_PIPS", "1.2"))


def _apply_regime_bias(conf: int, macro_regime: str | None, micro_regime: str | None):
    if not _REGIME_GUARD_ENABLED:
        return conf, {"macro": macro_regime, "micro": micro_regime, "bonus": 0}
    macro = macro_regime or "Mixed"
    micro = micro_regime or "Mixed"
    if macro == "Range":
        return None, {"macro": macro, "micro": micro, "decision": "skip_macro_range"}
    bonus = 0
    if macro == "Trend":
        bonus += 4
    elif macro == "Breakout":
        bonus += 3
    elif macro == "Mixed":
        bonus -= 2
    if micro == "Trend":
        bonus += 2
    elif micro == "Breakout":
        bonus += 2
    elif micro == "Range":
        bonus -= 3
    elif micro == "Mixed":
        bonus -= 1
    adj = max(0, min(100, int(round(conf + bonus))))
    return adj, {"macro": macro, "micro": micro, "bonus": bonus}


def _quality_guard(strategy_tag: str | None, macro_regime: str | None, micro_regime: str | None, atr_pips: float):
    macro = macro_regime or "NA"
    micro = micro_regime or "NA"
    if _H1M_REQUIRE_MACRO_TREND and macro not in {"Trend", "Breakout"}:
        return False, "macro_not_trend"
    if _H1M_BLOCK_MICRO_RANGE and micro == "Range":
        return False, "micro_range"
    if _H1M_MIN_ATR_PIPS > 0 and atr_pips < _H1M_MIN_ATR_PIPS:
        return False, "atr_low"
    return True, None


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
PIP = 0.01



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
def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=15.0, limit=1)
    if ticks:
        tick = ticks[-1]
        for key in ("mid",):
            val = tick.get(key)
            if val is not None:
                try:
                    return float(val)
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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "h1m"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-macro-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.6
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.6 + span * 0.4


def _compute_cap(
    *args,
    **kwargs,
) -> Tuple[float, Dict[str, float]]:
    kwargs.setdefault("env_prefix", config.ENV_PREFIX)
    res = compute_cap(
        cap_min=config.CAP_MIN,
        cap_max=config.CAP_MAX,
        *args,
        **kwargs,
    )
    return res.cap, res.reasons


async def h1momentum_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled via config", config.LOG_PREFIX)
        return

    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            LOG.debug("%s skip: market closed", config.LOG_PREFIX)
            continue
        if not can_trade(config.POCKET):
            LOG.debug("%s skip: pocket guard", config.LOG_PREFIX)
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        fac_h1 = factors.get("H1") or {}
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None

        signal = H1MomentumSwing.check(fac_m1)
        if not signal:
            continue
        div_bias = divergence_bias(
            fac_h1,
            signal.get("action") or "",
            mode="trend",
            max_age_bars=6,
        )
        if div_bias:
            base_conf = int(signal.get("confidence", 0) or 0)
            signal["confidence"] = apply_divergence_confidence(
                base_conf,
                div_bias,
                max_bonus=6.0,
                max_penalty=10.0,
                floor=30.0,
                ceil=90.0,
            )

        macro_regime = current_regime("H4", event_mode=False) or current_regime("H1", event_mode=False)
        micro_regime = current_regime("M1", event_mode=False)
        conf = int(signal.get("confidence", 50))
        conf_adj, regime_meta = _apply_regime_bias(conf, macro_regime, micro_regime)
        if conf_adj is None:
            LOG.info(
                "%s skip: regime_guard macro=%s micro=%s",
                config.LOG_PREFIX,
                macro_regime,
                micro_regime,
            )
            continue
        if conf_adj != conf:
            signal["confidence"] = conf_adj
            conf = conf_adj
        strategy_tag = signal.get("tag", H1MomentumSwing.name)

        snap = get_account_snapshot()
        equity = float(snap.nav or snap.balance or 0.0)

        balance = float(snap.balance or snap.nav or 0.0)
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        atr_pips = float(fac_h1.get("atr_pips") or 0.0)
        q_allow, q_reason = _quality_guard(strategy_tag, macro_regime, micro_regime, atr_pips)
        if not q_allow:
            LOG.info(
                "%s skip: quality_guard tag=%s macro=%s micro=%s atr=%.2f reason=%s",
                config.LOG_PREFIX,
                strategy_tag,
                macro_regime,
                micro_regime,
                atr_pips,
                q_reason,
            )
            continue
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            macro_pos = open_positions.get("macro") or {}
            pos_bias = abs(float(macro_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
        except Exception:
            pos_bias = 0.0

        cap, cap_reason = _compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=range_ctx.active,
            perf_pf=pf,
            pos_bias=pos_bias,
        )
        if cap <= 0.0:
            LOG.info("%s cap=0 stop (atr=%.2f fmr=%.3f range=%s)", config.LOG_PREFIX, atr_pips, free_ratio, range_ctx.active)
            continue

        # Entry sizing
        try:
            price = float(fac_h1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        side = "long" if signal["action"] == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if price <= 0.0 or sl_pips <= 0.0:
            LOG.debug("%s skip: bad price/sl price=%.5f sl=%.2f", config.LOG_PREFIX, price, sl_pips)
            continue
        if not _bb_entry_allowed(BB_STYLE, side, price, fac_m1, range_active=range_ctx.active):
            continue

        # 長めのTPはロットを薄く、短めはやや厚めにする
        tp_scale = 14.0 / max(1.0, tp_pips)
        tp_scale = max(0.35, min(1.1, tp_scale))
        base_units = int(
            round(
                scale_base_units(
                    config.BASE_ENTRY_UNITS,
                    equity=balance if balance > 0 else equity,
                    ref_equity=balance,
                    env_prefix=config.ENV_PREFIX,
                )
                * tp_scale
            )
        )
        conf_scale = _confidence_scale(int(signal.get("confidence", 50)))
        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=config.POCKET,
            side=side,
            strategy_tag=strategy_tag,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units_risk = int(round(lot * 100000))
        units = int(round(base_units * conf_scale))
        units = min(units, units_risk)
        units = int(round(units * cap))
        if units < config.MIN_UNITS:
            LOG.debug("%s skip: units too small (%s)", config.LOG_PREFIX, units)
            continue
        if side == "short":
            units = -abs(units)

        if side == "long":
            sl_price = round(price - sl_pips * 0.01, 3)
            tp_price = round(price + tp_pips * 0.01, 3) if tp_pips > 0 else None
        else:
            sl_price = round(price + sl_pips * 0.01, 3)
            tp_price = round(price - tp_pips * 0.01, 3) if tp_pips > 0 else None

        sl_price, tp_price = clamp_sl_tp(
            price=price,
            sl=sl_price,
            tp=tp_price,
            is_buy=side == "long",
        )
        client_id = _client_order_id(strategy_tag)
        entry_thesis = {
            "strategy_tag": strategy_tag,
            "env_prefix": config.ENV_PREFIX,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "hard_stop_pips": sl_pips,
            "confidence": int(signal.get("confidence", 0) or 0),
        }
        if regime_meta:
            entry_thesis["regime_bias"] = regime_meta
        div_meta = divergence_snapshot(fac_h1, max_age_bars=6)
        if div_meta:
            entry_thesis["divergence"] = div_meta

        proj_allow, proj_mult, proj_detail = _projection_decision(side, config.POCKET)
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
        entry_thesis_ctx = None
        for _name in ("entry_thesis", "thesis"):
            _candidate = locals().get(_name)
            if isinstance(_candidate, dict):
                entry_thesis_ctx = _candidate
                break
        if entry_thesis_ctx is None:
            entry_thesis_ctx = {}

        _tech_pocket = str(locals().get("pocket", config.POCKET))
        _tech_side_raw = str(locals().get("side", locals().get("direction", "long"))).lower()
        if _tech_side_raw in {"long", "short"}:
            _tech_side = _tech_side_raw
        else:
            _tech_side = "long"
        _tech_entry_price = locals().get("price")
        if not isinstance(_tech_entry_price, (int, float)):
            _tech_entry_price = locals().get("entry_price")
        if not isinstance(_tech_entry_price, (int, float)):
            _tech_entry_price = 0.0
        try:
            _tech_entry_price = float(_tech_entry_price)
        except (TypeError, ValueError):
            _tech_entry_price = 0.0

        _tech_signal_tag = str(
            locals().get("signal_tag")
            or locals().get("strategy_tag")
            or locals().get("STRATEGY_TAG")
            or getattr(config, "STRATEGY_TAG", "")
        )

        entry_thesis_ctx.setdefault(
            "tech_tfs",
            {"fib": ["H1", "M5"], "median": ["H1", "M5"], "nwave": ["M1", "M5"], "candle": ["M1", "M5"]},
        )
        entry_thesis_ctx.setdefault("technical_context_tfs", ["M1", "M5", "H1", "H4"])
        entry_thesis_ctx.setdefault(
            "technical_context_fields",
            [
                "ma10",
                "ma20",
                "rsi",
                "atr",
                "atr_pips",
                "adx",
                "macd",
                "macd_hist",
                "plus_di",
                "minus_di",
                "bbw",
                "kc_width",
                "vwap",
                "ema20",
                "ema24",
            ],
        )
        entry_thesis_ctx.setdefault("technical_context_ticks", ["latest_bid", "latest_ask", "latest_mid", "spread_pips"])
        entry_thesis_ctx.setdefault("technical_context_candle_counts", {"M1": 120, "M5": 80, "H1": 70, "H4": 60})
        entry_thesis_ctx.setdefault("tech_allow_candle", True)
        entry_thesis_ctx.setdefault(
            "tech_policy",
            {
                "mode": "balanced",
                "min_score": 0.0,
                "min_coverage": 0.0,
                "weight_fib": 0.25,
                "weight_median": 0.25,
                "weight_nwave": 0.25,
                "weight_candle": 0.25,
                "require_fib": False,
                "require_median": False,
                "require_nwave": False,
                "require_candle": False,
                "size_scale": 0.15,
                "size_min": 0.6,
                "size_max": 1.25,
            },
        )
        entry_thesis_ctx.setdefault("tech_policy_locked", False)
        entry_thesis_ctx.setdefault("env_tf", "M1")
        entry_thesis_ctx.setdefault("struct_tf", "M1")
        entry_thesis_ctx.setdefault("entry_tf", "M1")

        tech_decision = evaluate_entry_techniques(
            entry_price=_tech_entry_price,
            side=_tech_side,
            pocket=_tech_pocket,
            strategy_tag=_tech_signal_tag,
            entry_thesis=entry_thesis_ctx,
            allow_candle=bool(entry_thesis_ctx.get("tech_allow_candle", False)),
        )
        if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
            continue

        entry_thesis_ctx["tech_score"] = round(tech_decision.score, 3) if tech_decision.score is not None else None
        entry_thesis_ctx["tech_coverage"] = (
            round(tech_decision.coverage, 3) if tech_decision.coverage is not None else None
        )
        entry_thesis_ctx["tech_entry"] = tech_decision.debug
        entry_thesis_ctx["tech_reason"] = tech_decision.reason
        entry_thesis_ctx["tech_decision_allowed"] = bool(tech_decision.allowed)

        _tech_units_raw = locals().get("units")
        if isinstance(_tech_units_raw, (int, float)):
            _tech_units = int(round(abs(float(_tech_units_raw)) * tech_decision.size_mult))
            if _tech_units <= 0:
                continue
            units = _tech_units if _tech_side == "long" else -_tech_units
            entry_thesis_ctx["entry_units_intent"] = abs(int(units))

        _tech_conf = locals().get("conf")
        if isinstance(_tech_conf, (int, float)):
            _tech_conf = float(_tech_conf)
            if tech_decision.score is not None:
                if tech_decision.score >= 0:
                    _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_BOOST", 0.0)
                else:
                    _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_PENALTY", 0.0)
            conf = _tech_conf


        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=config.POCKET,
            client_order_id=client_id,
            strategy_tag=strategy_tag,
            confidence=int(signal.get("confidence", 0)),
            entry_thesis=entry_thesis,
        )
        LOG.info(
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
            config.LOG_PREFIX,
            units,
            side,
            price,
            sl_price,
            tp_price,
            signal.get("confidence", 0),
            cap,
            {**cap_reason, "tp_scale": round(tp_scale, 3)},
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
    asyncio.run(h1momentum_worker())
