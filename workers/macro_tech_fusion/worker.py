"""Macro Tech Fusion worker (trend-only)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

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
from analysis import perf_monitor

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
    ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
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


def _confidence_scale(conf: float, lo: float, hi: float) -> float:
    if conf <= lo:
        return 0.0
    if conf >= hi:
        return 1.0
    return (conf - lo) / max(1.0, (hi - lo))


def _trend_signal(fac_h1: Dict[str, object], fac_h4: Dict[str, object]) -> Optional[Tuple[str, float, Dict[str, float]]]:
    ma10_h1 = _float(fac_h1.get("ma10"))
    ma20_h1 = _float(fac_h1.get("ma20"))
    ma10_h4 = _float(fac_h4.get("ma10"))
    ma20_h4 = _float(fac_h4.get("ma20"))
    if ma10_h1 is None or ma20_h1 is None or ma10_h4 is None or ma20_h4 is None:
        return None

    if ma10_h1 > ma20_h1 and ma10_h4 > ma20_h4:
        side = "long"
    elif ma10_h1 < ma20_h1 and ma10_h4 < ma20_h4:
        side = "short"
    else:
        return None

    adx = _float(fac_h1.get("adx")) or 0.0
    if adx < config.TREND_ADX_MIN:
        return None

    macd_hist = _float(fac_h1.get("macd_hist")) or 0.0
    rsi = _float(fac_h1.get("rsi")) or 50.0
    plus_di = _float(fac_h1.get("plus_di")) or 0.0
    minus_di = _float(fac_h1.get("minus_di")) or 0.0

    score = 0.55
    if side == "long" and macd_hist > 0:
        score += 0.15
    if side == "short" and macd_hist < 0:
        score += 0.15
    if side == "long" and rsi >= 55:
        score += 0.1
    if side == "short" and rsi <= 45:
        score += 0.1
    if side == "long" and plus_di - minus_di >= 5:
        score += 0.1
    if side == "short" and minus_di - plus_di >= 5:
        score += 0.1

    if score < config.TREND_SCORE_MIN:
        return None

    base_conf = 52.0 + min(18.0, adx * 0.35) + (score - 0.5) * 40.0
    detail = {"score": round(score, 3), "adx": round(adx, 2), "rsi": round(rsi, 2)}
    return side, base_conf, detail


def _targets(atr_pips: float, spread_pips: float) -> Tuple[float, float]:
    sl = max(config.SL_MIN_PIPS, atr_pips * config.SL_ATR_MULT)
    sl = min(sl, config.SL_MAX_PIPS)
    tp = max(config.TP_MIN_PIPS, sl * config.TP_RR)
    tp = min(tp, config.TP_MAX_PIPS)

    if spread_pips > 0:
        sl = max(sl, spread_pips * 2.2 + config.SPREAD_FLOOR_PIPS)
    if tp < sl * config.TP_RR:
        tp = max(tp, sl * config.TP_RR)
        tp = min(tp, config.TP_MAX_PIPS)
    if tp < sl * config.TP_RR:
        sl = max(config.SL_MIN_PIPS, tp / max(config.TP_RR, 0.1))
    return round(sl, 2), round(tp, 2)


def _client_order_id(tag: str, pocket: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "macro"
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


async def macro_tech_fusion_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    loop_interval = float(config.LOOP_INTERVAL_SEC)
    pocket = config.POCKET
    log_prefix = config.LOG_PREFIX

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

        try:
            refresh_cache_from_disk()
        except Exception:
            pass
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h1 = factors.get("H1") or {}
        fac_h4 = factors.get("H4") or {}

        age_h1 = _factor_age_seconds(fac_h1)
        if age_h1 > config.MAX_FACTOR_AGE_SEC and time.time() - last_stale_log > 30.0:
            LOG.warning(
                "%s stale H1 factors age=%.1fs limit=%.1fs (proceeding)",
                log_prefix,
                age_h1,
                config.MAX_FACTOR_AGE_SEC,
            )
            last_stale_log = time.time()

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        if config.RANGE_BLOCK and range_ctx.active:
            continue

        signal = _trend_signal(fac_h1, fac_h4)
        if not signal:
            continue
        side, base_conf, side_detail = signal

        price = _float(fac_m1.get("close")) or _float(fac_h1.get("close")) or 0.0
        price = _latest_mid(price)
        if price <= 0:
            continue

        spread_pips = _float(fac_m1.get("spread_pips")) or 0.0
        atr_pips = _float(fac_h1.get("atr_pips")) or 0.0
        if atr_pips <= 0:
            atr_pips = _float(fac_h4.get("atr_pips")) or 0.0
        if atr_pips <= 0:
            atr_pips = 12.0

        sl_pips, tp_pips = _targets(atr_pips, spread_pips)
        if sl_pips <= 0 or tp_pips <= 0:
            continue

        signal_tag = f"{config.STRATEGY_TAG}-trend"
        tech_thesis = {
            "tech_allow_candle": True,
            "tech_tfs": {
                "fib": ["H4", "H1"],
                "median": ["H4", "H1"],
                "nwave": ["H1", "M5"],
                "candle": ["H1"],
            },
            "tech_policy": {
                "mode": "trend",
                "min_score": 0.18,
                "min_coverage": 0.6,
                "weight_fib": 0.3,
                "weight_median": 0.25,
                "weight_nwave": 0.25,
                "weight_candle": 0.2,
                "require_fib": True,
                "require_median": True,
                "require_nwave": True,
                "require_candle": True,
                "size_scale": 0.2,
                "size_min": 0.7,
                "size_max": 1.25,
                "nwave_min_quality": 0.22,
                "nwave_min_leg_pips": 6.0,
                "mid_distance_pips": 8.0,
            },
            "env_tf": "H4",
            "struct_tf": "H1",
            "entry_tf": "H1",
        }
        tech_decision = evaluate_entry_techniques(
            entry_price=price,
            side=side,
            pocket=pocket,
            strategy_tag=signal_tag,
            entry_thesis=tech_thesis,
            allow_candle=True,
        )
        if not tech_decision.allowed and not config.TECH_FAILOPEN:
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
            mode="trend",
            max_age_bars=12,
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
        div_meta = divergence_snapshot(fac_m1, max_age_bars=12)

        perf_decision = perf_guard.is_allowed(config.STRATEGY_TAG, pocket)
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
            range_active=range_ctx.active,
            perf_pf=pf,
            pos_bias=pos_bias,
            cap_min=config.CAP_MIN,
            cap_max=config.CAP_MAX,
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

        tp_scale = 12.0 / max(1.0, tp_pips)
        tp_scale = max(0.4, min(1.1, tp_scale))
        base_units = int(round(scale_base_units(config.BASE_ENTRY_UNITS, equity=balance if balance > 0 else equity, ref_equity=balance) * tp_scale))
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
            "profile": config.PROFILE_TAG,
            "confidence": int(conf),
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "hard_stop_pips": sl_pips,
            "mode": "trend",
            "range_active": bool(range_ctx.active),
            "range_score": round(range_ctx.score or 0.0, 3),
            "range_reason": range_ctx.reason,
            "range_mode": range_ctx.mode,
            "tech_score": round(tech_decision.score, 3) if tech_decision.score is not None else None,
            "tech_coverage": round(tech_decision.coverage, 3)
            if tech_decision.coverage is not None
            else None,
            "tech_entry": tech_decision.debug,
            "tech_allow_candle": True,
            "side_detail": side_detail,
            "tech_policy": tech_thesis.get("tech_policy"),
            "tech_tfs": tech_thesis.get("tech_tfs"),
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
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f tech=%s res=%s",
            log_prefix,
            units,
            side,
            price,
            sl_price or 0.0,
            tp_price or 0.0,
            conf,
            cap,
            tech_decision.reason or "ok",
            res or "none",
        )


if __name__ == "__main__":
    asyncio.run(macro_tech_fusion_worker())
