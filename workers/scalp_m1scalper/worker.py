"""M1Scalper dedicated worker with dynamic cap."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Tuple

from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.scalping.m1_scalper import M1Scalper
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.dyn_cap import compute_cap
from workers.common import perf_guard
from analysis import perf_monitor

from . import config

LOG = logging.getLogger(__name__)


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=6.0, limit=1)
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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "m1scalp"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.5
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.5 + span * 0.5


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


async def scalp_m1_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    last_block_log = 0.0
    last_cap_log = 0.0

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
            continue
        if now.hour in config.BLOCK_HOURS_UTC:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 300.0:
                LOG.info(
                    "%s blocked by hour hour=%02d block_hours=%s",
                    config.LOG_PREFIX,
                    now.hour,
                    sorted(config.BLOCK_HOURS_UTC),
                )
                last_block_log = now_mono
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

        signal = M1Scalper.check(fac_m1)
        if not signal:
            continue

        perf_decision = perf_guard.is_allowed(M1Scalper.name, config.POCKET)
        if not perf_decision.allowed:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s perf_block tag=%s reason=%s",
                    config.LOG_PREFIX,
                    M1Scalper.name,
                    perf_decision.reason,
                )
                last_block_log = now_mono
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
            scalp_pos = open_positions.get("scalp") or {}
            pos_bias = abs(float(scalp_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
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

        now_mono = time.monotonic()
        if (
            cap >= config.CAP_MAX * 0.85
            or free_ratio < 0.12
            or pos_bias > 0.25
            or range_ctx.active
        ) and now_mono - last_cap_log > 90.0:
            LOG.info(
                "%s cap=%.3f reasons=%s free_ratio=%.3f pos_bias=%.3f pf=%s range=%s",
                config.LOG_PREFIX,
                cap,
                cap_reason,
                free_ratio,
                pos_bias,
                pf,
                range_ctx.active,
            )
            last_cap_log = now_mono

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

        tp_scale = 5.0 / max(1.0, tp_pips)
        tp_scale = max(0.45, min(1.15, tp_scale))
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

        sl_price, tp_price = clamp_sl_tp(
            price=price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            side="BUY" if side == "long" else "SELL",
        )
        client_id = _client_order_id(signal.get("tag", M1Scalper.name))

        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=config.POCKET,
            client_order_id=client_id,
            strategy_tag=signal.get("tag", M1Scalper.name),
            entry_thesis={"strategy_tag": signal.get("tag", M1Scalper.name), "confidence": signal.get("confidence", 0)},
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
            res if res else "none",
        )


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    asyncio.run(scalp_m1_worker())
