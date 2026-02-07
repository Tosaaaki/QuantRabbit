"""Tech Fusion worker combining multi-technical entry filters."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis import perf_monitor
from analysis.range_guard import detect_range_mode
from analysis.technique_engine import evaluate_entry_techniques
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors, refresh_cache_from_disk
from market_data import tick_window
from utils.divergence import apply_divergence_confidence, divergence_bias, divergence_snapshot
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common.dyn_cap import compute_cap
from workers.common import perf_guard

from workers.common.size_utils import scale_base_units

from . import config

LOG = logging.getLogger(__name__)
PIP = 0.01


def _float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=6.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid = _float(tick.get("mid"))
        if mid is not None and mid > 0:
            return mid
        bid = _float(tick.get("bid"))
        ask = _float(tick.get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    return fallback


def _bb_levels(fac: Dict[str, object]) -> Optional[Tuple[float, float, float, float, float]]:
    upper = _float(fac.get("bb_upper"))
    lower = _float(fac.get("bb_lower"))
    mid = _float(fac.get("bb_mid")) or _float(fac.get("ma20"))
    bbw = _float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span / PIP


def _confidence_scale(conf: float, lo: float, hi: float) -> float:
    if conf <= lo:
        return 0.0
    if conf >= hi:
        return 1.0
    return (conf - lo) / max(1.0, (hi - lo))


def _trend_signal(fac: Dict[str, object]) -> Optional[Tuple[str, float, Dict[str, float]]]:
    ma10 = _float(fac.get("ma10"))
    ma20 = _float(fac.get("ma20"))
    macd_hist = _float(fac.get("macd_hist"))
    rsi = _float(fac.get("rsi"))
    adx = _float(fac.get("adx"))
    plus_di = _float(fac.get("plus_di"))
    minus_di = _float(fac.get("minus_di"))

    score_long = 0.0
    score_short = 0.0
    detail: Dict[str, float] = {}

    if ma10 is not None and ma20 is not None and ma10 != ma20:
        if ma10 > ma20:
            score_long += 0.35
            detail["ma"] = 0.35
        else:
            score_short += 0.35
            detail["ma"] = -0.35
    if macd_hist is not None:
        if macd_hist > 0:
            score_long += 0.25
            detail["macd"] = 0.25
        elif macd_hist < 0:
            score_short += 0.25
            detail["macd"] = -0.25
    if rsi is not None:
        if rsi >= 52:
            score_long += 0.2
            detail["rsi"] = 0.2
        elif rsi <= 48:
            score_short += 0.2
            detail["rsi"] = -0.2
    if plus_di is not None and minus_di is not None:
        diff = plus_di - minus_di
        if diff >= 5:
            score_long += 0.15
            detail["dmi"] = 0.15
        elif diff <= -5:
            score_short += 0.15
            detail["dmi"] = -0.15
    if adx is not None and adx < config.TREND_ADX_MIN:
        score_long -= 0.15
        score_short -= 0.15
        detail["adx_cut"] = -0.15

    side = "long" if score_long >= score_short else "short"
    raw = max(score_long, score_short)
    if raw < config.TREND_SCORE_MIN:
        return None
    base_conf = 45.0 + raw * 40.0
    if adx is not None and adx >= 25:
        base_conf += 4.0
    if rsi is not None and (rsi >= 60 or rsi <= 40):
        base_conf += 2.0
    detail["score"] = raw
    return side, base_conf, detail


def _range_signal(fac: Dict[str, object], price: float) -> Optional[Tuple[str, float, Dict[str, float]]]:
    levels = _bb_levels(fac)
    if not levels:
        return None
    upper, mid, lower, span, span_pips = levels
    rsi = _float(fac.get("rsi"))
    adx = _float(fac.get("adx"))
    bbw = _float(fac.get("bbw"))

    dist_long = (price - lower) / PIP
    dist_short = (upper - price) / PIP
    threshold = max(config.RANGE_BB_MIN_PIPS, span_pips * config.RANGE_BB_RATIO)

    score_long = 0.0
    score_short = 0.0
    detail: Dict[str, float] = {
        "dist_long": round(dist_long, 2),
        "dist_short": round(dist_short, 2),
    }

    if dist_long <= threshold:
        score_long += 0.5
    if dist_short <= threshold:
        score_short += 0.5
    if rsi is not None:
        if rsi <= config.RANGE_RSI_LONG:
            score_long += 0.3
        if rsi >= config.RANGE_RSI_SHORT:
            score_short += 0.3
    if adx is not None and adx <= config.RANGE_ADX_MAX:
        score_long += 0.2
        score_short += 0.2
    if bbw is not None and bbw <= config.RANGE_BBW_MAX:
        score_long += 0.15
        score_short += 0.15

    side = "long" if score_long >= score_short else "short"
    raw = max(score_long, score_short)
    if raw < config.RANGE_SCORE_MIN:
        return None
    base_conf = 40.0 + raw * 35.0
    detail["score"] = raw
    return side, base_conf, detail


def _targets(mode: str, atr_pips: float, spread_pips: float) -> Tuple[float, float]:
    if mode == "range":
        sl = max(config.SL_MIN_RANGE, atr_pips * config.SL_ATR_MULT_RANGE)
        sl = min(sl, config.SL_MAX_RANGE)
        tp = max(config.TP_MIN_RANGE, sl * config.TP_RR_RANGE)
        tp = min(tp, config.TP_MAX_RANGE)
    else:
        sl = max(config.SL_MIN_TREND, atr_pips * config.SL_ATR_MULT_TREND)
        sl = min(sl, config.SL_MAX_TREND)
        tp = max(config.TP_MIN_TREND, sl * config.TP_RR_TREND)
        tp = min(tp, config.TP_MAX_TREND)

    if spread_pips > 0:
        sl = max(sl, spread_pips * 2.2 + config.SPREAD_FLOOR_PIPS)
        tp = max(tp, sl * 0.8 + spread_pips)

    return round(sl, 2), round(tp, 2)


def _client_order_id(tag: str, pocket: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "tech"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-{pocket}-{sanitized}{digest}"


def _factor_age_seconds(fac: Dict[str, object]) -> float:
    ts_raw = fac.get("timestamp")
    if not ts_raw:
        return float("inf")
    try:
        if isinstance(ts_raw, (int, float)):
            ts_val = float(ts_raw)
            if ts_val > 1e12:
                ts_val /= 1000.0
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


async def tech_fusion_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    loop_interval = float(config.LOOP_INTERVAL_SEC)
    pocket = str(config.POCKET)
    mode_cfg = str(config.MODE).lower()
    log_prefix = config.LOG_PREFIX

    LOG.info("%s worker start (interval=%.1fs pocket=%s mode=%s)", log_prefix, loop_interval, pocket, mode_cfg)
    last_stale_log = 0.0
    last_perf_block_log = 0.0

    while True:
        await asyncio.sleep(loop_interval)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(pocket):
            continue

        try:
            refresh_cache_from_disk()
        except Exception:
            pass

        factors = all_factors()
        fac_m1 = dict(factors.get("M1") or {})
        fac_m5 = factors.get("M5") or {}
        fac_h4 = factors.get("H4") or {}

        age_m1 = _factor_age_seconds(fac_m1)
        if age_m1 > config.MAX_FACTOR_AGE_SEC and time.time() - last_stale_log > 30.0:
            LOG.warning(
                "%s stale factors age=%.1fs limit=%.1fs (proceeding)",
                log_prefix,
                age_m1,
                config.MAX_FACTOR_AGE_SEC,
            )
            last_stale_log = time.time()

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = float(range_ctx.score or 0.0) if range_ctx else 0.0
        range_only = bool(range_ctx.active) or range_score >= config.RANGE_ONLY_SCORE

        price = _float(fac_m1.get("close")) or 0.0
        price = _latest_mid(price)
        if price <= 0:
            continue

        mode = mode_cfg
        if mode == "both":
            mode = "range" if range_only else "trend"
        if mode not in {"trend", "range"}:
            mode = "trend"

        signal = None
        if mode == "range":
            signal = _range_signal(fac_m1, price)
        else:
            signal = _trend_signal(fac_m1)
        if not signal:
            continue

        side, base_conf, side_detail = signal
        spread_pips = _float(fac_m1.get("spread_pips")) or 0.0
        atr_pips = _float(fac_m1.get("atr_pips")) or 0.0
        if atr_pips <= 0:
            atr_pips = _float(fac_m5.get("atr_pips")) or 0.0
        if atr_pips <= 0:
            atr_pips = 6.0

        sl_pips, tp_pips = _targets(mode, atr_pips, spread_pips)
        if sl_pips <= 0 or tp_pips <= 0:
            continue

        signal_tag = f"{config.STRATEGY_TAG}-{mode}"
        tech_decision = evaluate_entry_techniques(
            entry_price=price,
            side=side,
            pocket=pocket,
            strategy_tag=signal_tag,
            entry_thesis={"tech_allow_candle": True},
            allow_candle=True,
        )
        if not tech_decision.allowed:
            if config.TECH_FAILOPEN is False:
                continue

        conf = float(base_conf)
        if tech_decision.score is not None:
            if tech_decision.score >= 0:
                conf += tech_decision.score * config.TECH_CONF_BOOST
            else:
                conf += tech_decision.score * config.TECH_CONF_PENALTY
        conf = max(config.CONFIDENCE_FLOOR, min(config.CONFIDENCE_CEIL, conf))

        div_bias = divergence_bias(
            fac_m1,
            "OPEN_LONG" if side == "long" else "OPEN_SHORT",
            mode="trend" if mode == "trend" else "range",
            max_age_bars=10,
        )
        if div_bias:
            conf = apply_divergence_confidence(
                int(conf),
                div_bias,
                max_bonus=6.0,
                max_penalty=10.0,
                floor=float(config.CONFIDENCE_FLOOR),
                ceil=float(config.CONFIDENCE_CEIL),
            )

        div_meta = divergence_snapshot(fac_m1, max_age_bars=10)

        perf_decision = perf_guard.is_allowed(config.STRATEGY_TAG, pocket, env_prefix=config.ENV_PREFIX)
        if not perf_decision.allowed:
            now_mono = time.monotonic()
            if now_mono - last_perf_block_log > 120.0:
                LOG.info(
                    "%s perf_block tag=%s reason=%s",
                    log_prefix,
                    config.STRATEGY_TAG,
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
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            pocket_pos = open_positions.get(pocket) or {}
            pos_bias = abs(float(pocket_pos.get("units", 0.0) or 0.0)) / max(
                1.0, float(snap.nav or 1.0)
            )
        except Exception:
            pos_bias = 0.0

        cap_res = compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=range_only,
            perf_pf=pf,
            pos_bias=pos_bias,
            cap_min=config.CAP_MIN,
            cap_max=config.CAP_MAX,
            env_prefix=config.ENV_PREFIX,
        )
        cap = cap_res.cap
        if cap <= 0.0:
            continue

        long_units = 0.0
        short_units = 0.0
        try:
            long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
        except Exception:
            long_units, short_units = 0.0, 0.0

        tp_scale = 10.0 / max(1.0, tp_pips)
        tp_scale = max(0.4, min(1.1, tp_scale))
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
        conf_scale = _confidence_scale(conf, config.CONFIDENCE_FLOOR, config.CONFIDENCE_CEIL)

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
        if tech_decision.size_mult > 1.0:
            units = int(round(units * tech_decision.size_mult))
        if units < config.MIN_UNITS:
            continue
        if side == "short":
            units = -abs(units)

        if side == "long":
            sl_price = round(price - sl_pips * PIP, 3)
            tp_price = round(price + tp_pips * PIP, 3)
        else:
            sl_price = round(price + sl_pips * PIP, 3)
            tp_price = round(price - tp_pips * PIP, 3)
        sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=side == "long")

        entry_thesis: Dict[str, object] = {
            "strategy_tag": signal_tag,
            "env_prefix": config.ENV_PREFIX,
            "profile": config.PROFILE_TAG,
            "confidence": int(conf),
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "hard_stop_pips": sl_pips,
            "mode": mode,
            "range_active": bool(range_ctx.active),
            "range_score": round(range_score, 3),
            "range_reason": range_ctx.reason,
            "range_mode": range_ctx.mode,
            "tech_score": round(tech_decision.score, 3) if tech_decision.score is not None else None,
            "tech_coverage": round(tech_decision.coverage, 3)
            if tech_decision.coverage is not None
            else None,
            "tech_entry": tech_decision.debug,
            "tech_allow_candle": True,
            "side_detail": side_detail,
        }
        if div_meta:
            entry_thesis["divergence"] = div_meta

        client_id = _client_order_id(signal_tag, pocket)
        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=pocket,
            client_order_id=client_id,
            strategy_tag=signal_tag,
            confidence=int(conf),
            entry_thesis=entry_thesis,
        )
        LOG.info(
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f mode=%s tech=%s res=%s",
            log_prefix,
            units,
            side,
            price,
            sl_price or 0.0,
            tp_price or 0.0,
            conf,
            cap,
            mode,
            tech_decision.reason or "ok",
            res or "none",
        )


if __name__ == "__main__":
    asyncio.run(tech_fusion_worker())
