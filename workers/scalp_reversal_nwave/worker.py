"""Scalp Reversal-NWave worker."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

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

from . import config

LOG = logging.getLogger(__name__)
PIP = 0.01


def _float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=4.0, limit=1)
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


def _reversal_signal(fac_m1: Dict[str, object]) -> Optional[Tuple[str, float, Dict[str, float]]]:
    vwap_gap = _float(fac_m1.get("vwap_gap")) or 0.0
    rsi = _float(fac_m1.get("rsi")) or 50.0

    if abs(vwap_gap) < config.VWAP_GAP_MIN_PIPS:
        return None

    if vwap_gap > 0:
        side = "short"
        if rsi < config.RSI_MAX_SHORT:
            return None
    else:
        side = "long"
        if rsi > config.RSI_MIN_LONG:
            return None

    strength = min(1.0, abs(vwap_gap) / max(1.0, config.VWAP_GAP_MIN_PIPS))
    base_conf = 38.0 + strength * 35.0
    detail = {"vwap_gap": round(vwap_gap, 2), "rsi": round(rsi, 2), "strength": round(strength, 2)}
    return side, base_conf, detail


def _targets(atr_pips: float, spread_pips: float) -> Tuple[float, float]:
    sl = max(config.SL_MIN_PIPS, atr_pips * config.SL_ATR_MULT)
    sl = min(sl, config.SL_MAX_PIPS)
    tp = max(config.TP_MIN_PIPS, sl * config.TP_RR)
    tp = min(tp, config.TP_MAX_PIPS)

    if spread_pips > 0:
        sl = max(sl, spread_pips * 1.6 + config.SPREAD_FLOOR_PIPS)
    if tp < sl * config.TP_RR:
        tp = max(tp, sl * config.TP_RR)
        tp = min(tp, config.TP_MAX_PIPS)
    if tp < sl * config.TP_RR:
        sl = max(config.SL_MIN_PIPS, tp / max(config.TP_RR, 0.1))
    return round(sl, 2), round(tp, 2)


def _client_order_id(tag: str, pocket: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "scalp"
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


async def scalp_reversal_nwave_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    loop_interval = float(config.LOOP_INTERVAL_SEC)
    pocket = config.POCKET
    log_prefix = config.LOG_PREFIX

    LOG.info("%s worker start (interval=%.2fs)", log_prefix, loop_interval)
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

        age_m1 = _factor_age_seconds(fac_m1)
        if age_m1 > config.MAX_FACTOR_AGE_SEC and time.time() - last_stale_log > 20.0:
            LOG.warning(
                "%s stale M1 factors age=%.1fs limit=%.1fs (proceeding)",
                log_prefix,
                age_m1,
                config.MAX_FACTOR_AGE_SEC,
            )
            last_stale_log = time.time()

        signal = _reversal_signal(fac_m1)
        if not signal:
            continue
        side, base_conf, side_detail = signal

        price = _float(fac_m1.get("close")) or 0.0
        price = _latest_mid(price)
        if price <= 0:
            continue

        spread_pips = _float(fac_m1.get("spread_pips")) or 0.0
        atr_pips = _float(fac_m1.get("atr_pips")) or 0.0
        if atr_pips <= 0:
            atr_pips = 2.6

        sl_pips, tp_pips = _targets(atr_pips, spread_pips)
        if sl_pips <= 0 or tp_pips <= 0:
            continue

        signal_tag = f"{config.STRATEGY_TAG}-reversal"
        tech_thesis = {
            "tech_allow_candle": True,
            "tech_tfs": {
                "fib": ["M1"],
                "median": ["M1"],
                "nwave": ["M1"],
                "candle": ["M1"],
            },
            "tech_policy": {
                "mode": "reversal",
                "min_score": 0.08,
                "min_coverage": 0.6,
                "weight_fib": 0.15,
                "weight_median": 0.2,
                "weight_nwave": 0.4,
                "weight_candle": 0.25,
                "require_fib": False,
                "require_median": False,
                "require_nwave": True,
                "require_candle": True,
                "size_scale": 0.18,
                "size_min": 0.6,
                "size_max": 1.2,
                "nwave_min_quality": 0.18,
                "nwave_min_leg_pips": 2.0,
                "mid_distance_pips": 1.2,
            },
            "env_tf": "M1",
            "struct_tf": "M1",
            "entry_tf": "M1",
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
            mode="reversal",
            max_age_bars=6,
        )
        if div_bias:
            conf = apply_divergence_confidence(
                int(conf),
                div_bias,
                max_bonus=4.0,
                max_penalty=8.0,
                floor=float(config.CONFIDENCE_FLOOR),
                ceil=float(config.CONFIDENCE_CEIL),
            )
        div_meta = divergence_snapshot(fac_m1, max_age_bars=6)

        perf_decision = perf_guard.is_allowed(config.STRATEGY_TAG, pocket)
        if not perf_decision.allowed:
            now_mono = time.monotonic()
            if now_mono - last_perf_block_log > 60.0:
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
            range_active=False,
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
            long_units, short_units = get_position_summary("USD_JPY", timeout=2.0)
        except Exception:
            long_units, short_units = 0.0, 0.0

        tp_scale = 6.0 / max(1.0, tp_pips)
        tp_scale = max(0.45, min(1.1, tp_scale))
        base_units = int(round(config.BASE_ENTRY_UNITS * tp_scale))
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
            fac_h4=factors.get("H4") or {},
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
            "mode": "reversal",
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
    asyncio.run(scalp_reversal_nwave_worker())
