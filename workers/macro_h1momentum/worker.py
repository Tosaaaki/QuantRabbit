"""H1Momentum dedicated macro worker with dynamic cap by market state."""

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
from strategies.trend.h1_momentum import H1MomentumSwing
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.quality_gate import news_block_active

from . import config

LOG = logging.getLogger(__name__)
PIP = 0.01


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
    *,
    atr_pips: float,
    free_ratio: float,
    range_active: bool,
    perf_pf: Optional[float],
    pos_bias: float,
) -> Tuple[float, Dict[str, float]]:
    cap = 0.7
    reasons: Dict[str, float] = {}

    # ATR
    if atr_pips >= 3.0:
        cap *= 1.2
        reasons["atr_boost"] = atr_pips
    elif atr_pips <= 1.2:
        cap *= 0.75
        reasons["atr_cut"] = atr_pips

    # Free margin ratio
    if free_ratio < 0.03:
        return 0.0, {"free_ratio": free_ratio}
    if free_ratio < 0.15:
        cap *= 0.65
        reasons["fmr_tight"] = free_ratio
    elif free_ratio < 0.25:
        cap *= 0.85
        reasons["fmr_soft"] = free_ratio
    else:
        cap *= 1.05
        reasons["fmr_ok"] = free_ratio

    # Perf
    if perf_pf is not None:
        if perf_pf >= 1.3:
            cap *= 1.15
            reasons["pf_boost"] = perf_pf
        elif perf_pf <= 0.9:
            cap *= 0.7
            reasons["pf_cut"] = perf_pf

    # Range guard
    if range_active:
        cap = min(cap, 0.4)
        reasons["range_cap"] = 1.0

    # Existing position bias (same side reduces headroom)
    if pos_bias > 0.5:
        cap *= 0.7
        reasons["pos_bias_cut"] = pos_bias

    cap = max(config.CAP_MIN, min(config.CAP_MAX, cap))
    reasons["cap"] = cap
    return cap, reasons


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
        if config.NEWS_BLOCK_MINUTES > 0 and news_block_active(
            config.NEWS_BLOCK_MINUTES, min_impact=config.NEWS_BLOCK_MIN_IMPACT
        ):
            LOG.debug("%s skip: news block", config.LOG_PREFIX)
            continue
        if not can_trade(config.POCKET):
            LOG.debug("%s skip: pocket guard", config.LOG_PREFIX)
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        fac_h1 = factors.get("H1") or {}
        range_ctx = detect_range_mode(fac_m1, fac_h4)

        signal = H1MomentumSwing.check(fac_m1)
        if not signal:
            continue

        snap = get_account_snapshot()
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        atr_pips = float(fac_h1.get("atr_pips") or 0.0)
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
            perf_pf=None,  # hook up to perf cache later
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

        # 長めのTPはロットを薄く、短めはやや厚めにする
        tp_scale = 14.0 / max(1.0, tp_pips)
        tp_scale = max(0.35, min(1.1, tp_scale))
        base_units = int(round(config.BASE_ENTRY_UNITS * tp_scale))
        conf_scale = _confidence_scale(int(signal.get("confidence", 50)))
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
            LOG.debug("%s skip: units too small (%s)", config.LOG_PREFIX, units)
            continue
        if side == "short":
            units = -abs(units)

        sl_price, tp_price = clamp_sl_tp(
            price=price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            side="BUY" if side == "long" else "SELL",
        )
        client_id = _client_order_id(signal.get("tag", H1MomentumSwing.name))

        res = market_order(
            units=units,
            sl=sl_price,
            tp=tp_price,
            client_order_id=client_id,
            pocket=config.POCKET,
            tag=signal.get("tag", H1MomentumSwing.name),
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
            res.status if res else "none",
        )
