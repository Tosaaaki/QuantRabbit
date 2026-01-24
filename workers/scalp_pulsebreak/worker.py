from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Tuple

from analysis.range_guard import detect_range_mode
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors
from market_data import tick_window
from strategies.scalping.pulse_break import PulseBreak
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.dyn_cap import compute_cap

from . import config

LOG = logging.getLogger(__name__)


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "scalp"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-{sanitized}{digest}"


def _confidence_scale(conf: int, *, lo: int, hi: int) -> float:
    if conf <= lo:
        return 0.5
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.5 + span * 0.5


def _compute_cap(*, cap_min: float, cap_max: float, **kwargs) -> Tuple[float, Dict[str, float]]:
    res = compute_cap(cap_min=cap_min, cap_max=cap_max, **kwargs)
    return res.cap, res.reasons


async def scalp_pulsebreak_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            return

    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    pos_manager = PositionManager()
    last_entry_ts = 0.0

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now = datetime.datetime.utcnow()
            if not is_market_open(now):
                continue
            if not can_trade(config.POCKET):
                continue
            if config.COOLDOWN_SEC > 0.0 and time.monotonic() - last_entry_ts < config.COOLDOWN_SEC:
                continue

            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            range_ctx = detect_range_mode(fac_m1, fac_h4)
            range_score = 0.0
            try:
                range_score = float(range_ctx.score or 0.0)
            except Exception:
                range_score = 0.0
            range_only = range_ctx.active or range_score >= config.RANGE_ONLY_SCORE
            if range_only:
                continue

            if config.MAX_OPEN_TRADES > 0:
                try:
                    positions = pos_manager.get_open_positions()
                    scalp_info = positions.get(config.POCKET) or {}
                    open_trades = len(scalp_info.get("open_trades") or [])
                    if open_trades >= config.MAX_OPEN_TRADES:
                        continue
                except Exception:
                    pass

            fac_m1 = dict(fac_m1)
            fac_m1["range_active"] = bool(range_ctx.active)
            fac_m1["range_score"] = range_score
            fac_m1["range_reason"] = range_ctx.reason
            fac_m1["range_mode"] = range_ctx.mode

            signal = PulseBreak.check(fac_m1)
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
                scalp_pos = open_positions.get(config.POCKET) or {}
                pos_bias = abs(float(scalp_pos.get("units", 0.0) or 0.0)) / max(
                    1.0, float(snap.nav or 1.0)
                )
            except Exception:
                pos_bias = 0.0

            cap, cap_reason = _compute_cap(
                atr_pips=atr_pips,
                free_ratio=free_ratio,
                range_active=range_ctx.active,
                perf_pf=None,
                pos_bias=pos_bias,
                cap_min=config.CAP_MIN,
                cap_max=config.CAP_MAX,
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

            side = "long" if signal.get("action") == "OPEN_LONG" else "short"
            sl_pips = float(signal.get("sl_pips") or 0.0)
            tp_pips = float(signal.get("tp_pips") or 0.0)
            if sl_pips <= 0.0:
                continue

            tp_scale = 4.0 / max(1.0, tp_pips)
            tp_scale = max(0.4, min(1.2, tp_scale))
            base_units = int(round(config.BASE_ENTRY_UNITS * tp_scale))

            conf_scale = _confidence_scale(int(signal.get("confidence", 50)), lo=config.CONFIDENCE_FLOOR, hi=config.CONFIDENCE_CEIL)
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

            sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=side == "long")
            signal_tag = (signal.get("tag") or "").strip() or PulseBreak.name
            client_id = _client_order_id(signal_tag)
            entry_thesis: Dict[str, object] = {
                "strategy_tag": signal_tag,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
                "hard_stop_pips": sl_pips,
                "confidence": signal.get("confidence", 0),
                "range_active": bool(range_ctx.active),
                "range_score": round(range_score, 3),
                "range_reason": range_ctx.reason,
                "range_mode": range_ctx.mode,
                "entry_guard_trend": True,
            }

            res = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=client_id,
                strategy_tag=signal_tag,
                confidence=int(signal.get("confidence", 0)),
                entry_thesis=entry_thesis,
            )
            last_entry_ts = time.monotonic()
            LOG.info(
                "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f reasons=%s res=%s",
                config.LOG_PREFIX,
                units,
                side,
                price,
                sl_price or 0.0,
                tp_price or 0.0,
                signal.get("confidence", 0),
                cap,
                {**cap_reason, "tp_scale": round(tp_scale, 3)},
                res or "none",
            )
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)


if __name__ == "__main__":
    asyncio.run(scalp_pulsebreak_worker())
