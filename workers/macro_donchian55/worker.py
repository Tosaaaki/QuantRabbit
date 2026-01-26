"""Donchian55 dedicated macro worker with dynamic cap."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import time
from typing import Dict, Optional, Tuple

from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from indicators.factor_cache import all_factors
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.breakout.donchian55 import Donchian55
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.dyn_cap import compute_cap
from analysis import perf_monitor

from . import config

LOG = logging.getLogger(__name__)
PIP = 0.01


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no"}


def _entry_guard(
    *,
    side: str,
    price: float,
    adx: float,
    fac_h4: Dict[str, object],
    fac_h1: Dict[str, object],
    is_addon: bool,
) -> bool:
    if not _env_bool("ENTRY_GUARD_ENABLED_DONCHIAN55", True):
        return True
    if not is_addon and not _env_bool("ENTRY_GUARD_PRIMARY_DONCHIAN55", True):
        return True

    candles = fac_h4.get("candles") or fac_h1.get("candles") or []
    if not candles:
        return True
    lookback = _env_int("ENTRY_GUARD_LOOKBACK_DONCHIAN55", 60)
    hi_pct = _env_float("ENTRY_GUARD_HI_PCT_DONCHIAN55", 95.0)
    lo_pct = _env_float("ENTRY_GUARD_LO_PCT_DONCHIAN55", 5.0)
    axis = compute_range_snapshot(
        candles,
        lookback=max(20, lookback),
        method="percentile",
        hi_pct=hi_pct,
        lo_pct=lo_pct,
    )
    if axis is None:
        return True
    range_span = max(0.0, float(axis.high) - float(axis.low))
    if range_span <= 0.0:
        return True
    range_pips = range_span / PIP
    min_range_pips = _env_float("ENTRY_GUARD_MIN_RANGE_PIPS_DONCHIAN55", 20.0)
    if range_pips < min_range_pips:
        return True

    fib_extreme = _env_float("ENTRY_GUARD_FIB_EXTREME_DONCHIAN55", 0.214)
    fib_extreme = max(0.05, min(0.45, fib_extreme))
    mid_distance_pips = _env_float("ENTRY_GUARD_MID_DISTANCE_PIPS_DONCHIAN55", 20.0)
    mid_distance_frac = _env_float("ENTRY_GUARD_MID_DISTANCE_FRAC_DONCHIAN55", 0.2)
    mid_distance_pips = max(mid_distance_pips, range_pips * mid_distance_frac)
    adx_bypass = _env_float("ENTRY_GUARD_ADX_BYPASS_DONCHIAN55", 30.0)
    if adx >= adx_bypass:
        return True

    upper_guard = float(axis.high) - range_span * fib_extreme
    lower_guard = float(axis.low) + range_span * fib_extreme
    mid = float(axis.mid)
    mid_distance = abs(price - mid) / PIP

    if side == "long":
        if price >= upper_guard:
            LOG.info("%s entry_guard_extreme_long price=%.3f upper=%.3f", config.LOG_PREFIX, price, upper_guard)
            return False
        if price > mid and mid_distance >= mid_distance_pips:
            LOG.info("%s entry_guard_mid_far_long price=%.3f mid=%.3f dist=%.1f", config.LOG_PREFIX, price, mid, mid_distance)
            return False
    else:
        if price <= lower_guard:
            LOG.info("%s entry_guard_extreme_short price=%.3f lower=%.3f", config.LOG_PREFIX, price, lower_guard)
            return False
        if price < mid and mid_distance >= mid_distance_pips:
            LOG.info("%s entry_guard_mid_far_short price=%.3f mid=%.3f dist=%.1f", config.LOG_PREFIX, price, mid, mid_distance)
            return False
    return True


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=15.0, limit=1)
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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "don55"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-macro-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _confidence_fraction(conf: int) -> float:
    """0.35〜1.10で信頼度に応じて段階的に増加。"""
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    norm = (conf - lo) / max(1.0, hi - lo)
    norm = max(0.0, min(1.0, norm))
    return max(0.35, min(1.10, 0.35 + norm * 0.75))


def _incremental_units(target_units: int, existing_units: int, min_units: int) -> int:
    """長期の積み増し用に、既存ポジとの差分の一部だけを積む。"""
    if target_units == 0:
        return 0
    same_side = target_units * existing_units > 0
    if same_side:
        gap = abs(target_units) - abs(existing_units)
        if gap <= max(1, int(min_units * 0.25)):
            return 0
        step = int(round(gap * 0.4))
        step = max(min_units, min(abs(gap), step))
        return step if target_units > 0 else -step
    # 反対サイドだったらそのまま（ネットで調整）
    return target_units


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


async def donchian55_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
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

        signal = Donchian55.check(fac_h1, range_active=range_ctx.active)
        if not signal:
            continue

        snap = get_account_snapshot()
        free_ratio_raw = snap.free_margin_ratio
        free_ratio = float(free_ratio_raw or 0.0) if free_ratio_raw is not None else 0.0
        usage_ratio = None
        if free_ratio_raw is not None:
            usage_ratio = max(0.0, min(1.0, 1.0 - free_ratio))
        else:
            total_margin = float((snap.margin_available or 0.0) + (snap.margin_used or 0.0))
            if total_margin > 0.0:
                usage_ratio = float(snap.margin_used or 0.0) / total_margin
        if free_ratio_raw is not None and free_ratio_raw <= config.MIN_FREE_MARGIN_RATIO:
            LOG.info(
                "%s skip: free_margin_low ratio=%.3f limit=%.3f",
                config.LOG_PREFIX,
                free_ratio_raw,
                config.MIN_FREE_MARGIN_RATIO,
            )
            continue
        if usage_ratio is not None and usage_ratio >= config.MAX_MARGIN_USAGE:
            LOG.info(
                "%s skip: margin_usage_high usage=%.3f limit=%.3f",
                config.LOG_PREFIX,
                usage_ratio,
                config.MAX_MARGIN_USAGE,
            )
            continue
        try:
            atr_pips = float(fac_h1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
        try:
            adx = float(fac_h1.get("adx") or 0.0)
        except Exception:
            adx = 0.0
        conf = int(signal.get("confidence", 50))
        min_conf = (
            config.CONFIDENCE_MIN_HIGH_VOL
            if atr_pips >= config.CONFIDENCE_MIN_ATR_PIPS
            else config.CONFIDENCE_MIN_BASE
        )
        if conf < min_conf:
            LOG.debug(
                "%s skip: confidence_low conf=%s min=%s atr=%.2f",
                config.LOG_PREFIX,
                conf,
                min_conf,
                atr_pips,
            )
            continue
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            macro_pos = open_positions.get("macro") or {}
            pos_bias = abs(float(macro_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
            existing_units = int(macro_pos.get("units") or 0)
        except Exception:
            pos_bias = 0.0
            existing_units = 0

        cap, cap_reason = _compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=range_ctx.active,
            perf_pf=pf,
            pos_bias=pos_bias,
        )
        if cap <= 0.0:
            continue

        try:
            price = float(fac_h1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        side = "long" if signal["action"] == "OPEN_LONG" else "short"
        same_side = (existing_units > 0 and side == "long") or (existing_units < 0 and side == "short")
        is_addon = same_side and abs(existing_units) > 0
        if not _entry_guard(
            side=side,
            price=price,
            adx=adx,
            fac_h4=fac_h4,
            fac_h1=fac_h1,
            is_addon=is_addon,
        ):
            continue
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if price <= 0.0 or sl_pips <= 0.0:
            continue

        tp_scale = 20.0 / max(1.0, tp_pips)
        tp_scale = max(0.25, min(1.05, tp_scale))
        base_units = int(round(config.BASE_ENTRY_UNITS * tp_scale))

        conf_scale = _confidence_scale(conf)
        conf_frac = _confidence_fraction(conf)
        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=config.POCKET,
        )
        units_risk = int(round(lot * 100000))
        target_units = int(round(base_units * conf_frac))
        target_units = min(target_units, units_risk)
        target_units = int(round(target_units * cap))
        if side == "short":
            target_units = -abs(target_units)

        units = _incremental_units(target_units, existing_units, config.MIN_UNITS)
        if abs(units) < config.MIN_UNITS:
            continue

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
        strategy_tag = signal.get("tag", Donchian55.name)
        client_id = _client_order_id(strategy_tag)
        entry_thesis = {
            "strategy_tag": strategy_tag,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "hard_stop_pips": sl_pips,
            "confidence": conf,
        }

        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=config.POCKET,
            client_order_id=client_id,
            strategy_tag=strategy_tag,
            confidence=conf,
            entry_thesis=entry_thesis,
        )
        LOG.info(
            "%s sent units=%s target=%s existing=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
            config.LOG_PREFIX,
            units,
            target_units,
            existing_units,
            side,
            price,
            sl_price,
            tp_price,
            conf,
            cap,
            {**cap_reason, "tp_scale": round(tp_scale, 3)},
            res or "none",
        )


if __name__ == "__main__":
    asyncio.run(donchian55_worker())
