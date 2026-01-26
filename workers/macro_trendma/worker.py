"""TrendMA dedicated macro worker with dynamic cap."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.trend.ma_cross import MovingAverageCross
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.dyn_cap import compute_cap
from analysis import perf_monitor

from . import config

LOG = logging.getLogger(__name__)
PIP = 0.01


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


def _recent_extremes(raw_candles: object, window: int) -> Tuple[float, float]:
    if not isinstance(raw_candles, list) or not raw_candles:
        return 0.0, 0.0
    window = max(1, int(window))
    highs = []
    lows = []
    for candle in raw_candles[-window:]:
        if not isinstance(candle, dict):
            continue
        try:
            high = float(candle.get("high"))
            low = float(candle.get("low"))
        except Exception:
            continue
        highs.append(high)
        lows.append(low)
    if not highs or not lows:
        return 0.0, 0.0
    return max(highs), min(lows)


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "trendma"
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
    res = compute_cap(
        cap_min=config.CAP_MIN,
        cap_max=config.CAP_MAX,
        *args,
        **kwargs,
    )
    return res.cap, res.reasons


async def trendma_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
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

        # レンジ判定が出ているときは TrendMA での新規を抑止
        if range_ctx.active:
            LOG.debug("%s skip: range_active", config.LOG_PREFIX)
            continue

        signal = MovingAverageCross.check(fac_m1)
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
        atr_pips = float(fac_h1.get("atr_pips") or 0.0)
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
            continue

        try:
            price = float(fac_m1.get("close") or fac_h1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        side = "long" if signal["action"] == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if price <= 0.0 or sl_pips <= 0.0:
            continue

        # TP をさらに短くクランプ（レンジ警戒）
        tp_cap = max(10.0, min(22.0, atr_pips * 3.0 + 6.0))
        if tp_pips > tp_cap:
            tp_pips = tp_cap

        h4_recent_high, h4_recent_low = _recent_extremes(
            fac_h4.get("candles"),
            config.H4_EXTREME_WINDOW,
        )
        if side == "long" and h4_recent_high > 0:
            high_gap_pips = (h4_recent_high - price) / PIP
            if 0.0 <= high_gap_pips <= config.H4_NEAR_HIGH_PIPS:
                LOG.debug(
                    "%s skip: near_h4_high price=%.3f recent_high=%.3f gap_pips=%.1f",
                    config.LOG_PREFIX,
                    price,
                    h4_recent_high,
                    high_gap_pips,
                )
                continue
        if side == "short" and h4_recent_low > 0:
            low_gap_pips = (price - h4_recent_low) / PIP
            if 0.0 <= low_gap_pips <= config.H4_NEAR_LOW_PIPS:
                LOG.debug(
                    "%s skip: near_h4_low price=%.3f recent_low=%.3f gap_pips=%.1f",
                    config.LOG_PREFIX,
                    price,
                    h4_recent_low,
                    low_gap_pips,
                )
                continue

        # TP を手前にクランプ（レンジ気味のため）
        tp_cap = max(12.0, min(20.0, atr_pips * 3.5 + 4.0))
        if tp_pips > tp_cap:
            tp_pips = tp_cap

        tp_scale = 14.0 / max(1.0, tp_pips)
        tp_scale = max(0.35, min(1.1, tp_scale))
        base_units = int(round(config.BASE_ENTRY_UNITS * tp_scale))

        conf_scale = _confidence_scale(conf)
        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=config.POCKET,
        )
        units_risk = int(round(lot * 100000))
        units = int(round(base_units * conf_scale))
        units = min(units, units_risk)
        units = int(round(units * cap))
        if units < config.MIN_UNITS:
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
        strategy_tag = signal.get("tag", MovingAverageCross.name)
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
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
            config.LOG_PREFIX,
            units,
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
    asyncio.run(trendma_worker())
