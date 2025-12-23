"""Micro multi-strategy worker with dynamic cap."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, List, Tuple

from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.micro.momentum_burst import MomentumBurstMicro
from strategies.micro.momentum_stack import MicroMomentumStack
from strategies.micro.pullback_ema import MicroPullbackEMA
from strategies.micro.level_reactor import MicroLevelReactor
from strategies.micro.range_break import MicroRangeBreak
from strategies.micro.vwap_bound_revert import MicroVWAPBound
from strategies.micro.trend_momentum import TrendMomentumMicro
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.quality_gate import news_block_active
from workers.common.dyn_cap import compute_cap
from analysis import perf_monitor

from . import config

LOG = logging.getLogger(__name__)


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


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


def _strategy_list() -> List:
    return [
        MomentumBurstMicro,
        MicroMomentumStack,
        MicroPullbackEMA,
        MicroLevelReactor,
        MicroRangeBreak,
        MicroVWAPBound,
        TrendMomentumMicro,
    ]


async def micro_multi_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if config.NEWS_BLOCK_MINUTES > 0 and news_block_active(
            config.NEWS_BLOCK_MINUTES, min_impact=config.NEWS_BLOCK_MIN_IMPACT
        ):
            continue
        if not can_trade(config.POCKET):
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None

        signal = None
        strategy_name = ""
        for strat in _strategy_list():
            cand = strat.check(fac_m1)
            if cand:
                signal = cand
                strategy_name = getattr(strat, "name", strat.__name__)
                break
        if not signal:
            continue

        snap = get_account_snapshot()
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
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
        )
        if cap <= 0.0:
            continue

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        side = "long" if signal["action"] == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if price <= 0.0 or sl_pips <= 0.0:
            continue

        tp_scale = 10.0 / max(1.0, tp_pips)
        tp_scale = max(0.4, min(1.1, tp_scale))
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
        client_id = _client_order_id(signal.get("tag", strategy_name))

        res = market_order(
            units=units,
            sl=sl_price,
            tp=tp_price,
            client_order_id=client_id,
            pocket=config.POCKET,
            tag=signal.get("tag", strategy_name),
        )
        LOG.info(
            "%s strat=%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
            config.LOG_PREFIX,
            strategy_name,
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


if __name__ == "__main__":
    asyncio.run(micro_multi_worker())
