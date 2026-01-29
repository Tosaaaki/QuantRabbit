"""Volatility spike rider entry worker (scalp pocket)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis.range_guard import detect_range_mode
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, clamp_sl_tp, can_trade
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common import perf_guard

from . import config

LOG = logging.getLogger(__name__)


def _float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_tick() -> Optional[dict]:
    ticks = tick_window.recent_ticks(seconds=max(2.0, config.ENTRY_WINDOW_SEC), limit=1)
    if not ticks:
        return None
    return ticks[-1]


def _mid_from_tick(tick: dict, fallback: float) -> float:
    mid = _float(tick.get("mid"))
    if mid is not None and mid > 0:
        return mid
    bid = _float(tick.get("bid"))
    ask = _float(tick.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return fallback


def _recent_move(window_sec: float) -> Tuple[float, float, float, float]:
    ticks = tick_window.recent_ticks(seconds=window_sec, limit=int(window_sec * 15) + 5)
    if len(ticks) < 2:
        return 0.0, 0.0, 0.0, 0.0
    first = ticks[0]
    last = ticks[-1]
    first_mid = _mid_from_tick(first, _float(first.get("mid")) or 0.0)
    last_mid = _mid_from_tick(last, _float(last.get("mid")) or first_mid)
    span = max(0.001, float(last.get("epoch") or 0.0) - float(first.get("epoch") or 0.0))
    move_pips = (last_mid - first_mid) / config.PIP_VALUE
    speed = abs(move_pips) / span if span > 0 else 0.0
    return move_pips, abs(move_pips), span, speed


def _tick_range(ticks: list[dict]) -> tuple[float, float] | None:
    if not ticks:
        return None
    highs = []
    lows = []
    for tick in ticks:
        mid = _float(tick.get("mid"))
        if mid is None:
            bid = _float(tick.get("bid"))
            ask = _float(tick.get("ask"))
            if bid is None or ask is None:
                continue
            mid = (bid + ask) / 2.0
        highs.append(mid)
        lows.append(mid)
    if not highs or not lows:
        return None
    return max(highs), min(lows)


def _spread_pips(tick: Optional[dict]) -> float:
    state = spread_monitor.get_state() or {}
    spread = _float(state.get("spread_pips"))
    if spread is not None and spread > 0:
        return spread
    if tick:
        bid = _float(tick.get("bid"))
        ask = _float(tick.get("ask"))
        if bid is not None and ask is not None and ask > bid:
            return (ask - bid) / config.PIP_VALUE
    return 0.0


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch for ch in tag if ch.isalnum())[:10] or "spike"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-{sanitized}-{digest}"


def _build_thesis(
    *,
    direction: str,
    move_pips: float,
    min_move_pips: float,
    recent_move_pips: float,
    span_sec: float,
    speed_pps: float,
    spread_pips: float,
    atr_pips: Optional[float],
    range_active: bool,
    trend_dir: Optional[str],
    trend_gap_pips: Optional[float],
    trend_adx: Optional[float],
    body_pips: Optional[float],
    tp_pips: float,
    sl_pips: float,
    confidence: int,
) -> Dict[str, object]:
    return {
        "strategy_tag": config.STRATEGY_TAG,
        "profile": config.PROFILE_TAG,
        "direction": direction,
        "move_pips": round(move_pips, 3),
        "min_move_pips": round(min_move_pips, 3),
        "recent_move_pips": round(recent_move_pips, 3),
        "window_sec": round(span_sec, 3),
        "speed_pps": round(speed_pps, 3),
        "spread_pips": round(spread_pips, 3),
        "atr_pips": None if atr_pips is None else round(atr_pips, 3),
        "range_active": bool(range_active),
        "trend_dir": trend_dir,
        "trend_gap_pips": None if trend_gap_pips is None else round(trend_gap_pips, 3),
        "trend_adx": None if trend_adx is None else round(trend_adx, 3),
        "body_pips": None if body_pips is None else round(body_pips, 3),
        "tp_pips": round(tp_pips, 3),
        "sl_pips": round(sl_pips, 3),
        "hard_stop_pips": round(sl_pips, 3),
        "confidence": confidence,
        "entry_tf": "M1",
        "env_tf": "M5",
        "struct_tf": "M1",
    }


def _calc_sl_tp(abs_move: float, atr_pips: float) -> Tuple[float, float]:
    sl_from_atr = atr_pips * config.SL_ATR_MULT
    sl_from_move = abs_move * config.SL_MOVE_MULT
    sl_pips = max(config.SL_MIN_PIPS, min(config.SL_MAX_PIPS, max(sl_from_atr, sl_from_move)))
    tp_pips = max(config.TP_MIN_PIPS, min(config.TP_MAX_PIPS, sl_pips * config.TP_RR))
    return sl_pips, tp_pips


def _size_mult(abs_move: float, speed_pps: float, min_move_pips: float) -> float:
    base = abs_move / max(0.1, min_move_pips)
    speed = speed_pps / max(0.1, config.ENTRY_MIN_SPEED_PPS)
    mult = 0.6 * base + 0.4 * speed
    return max(config.ENTRY_SIZE_MIN_MULT, min(config.ENTRY_SIZE_MAX_MULT, mult))


def _trend_state(factors: Dict[str, float], fallback_price: float) -> tuple[str, float, float] | None:
    ma10 = _float(factors.get("ma10"))
    ma20 = _float(factors.get("ma20"))
    adx = _float(factors.get("adx")) or 0.0
    price = _float(factors.get("close")) or fallback_price
    if not ma10 or not ma20 or not price:
        return None
    gap_pips = (ma10 - ma20) / config.PIP_VALUE
    direction = "long" if ma10 > ma20 else "short"
    return direction, gap_pips, adx


async def vol_spike_rider_worker() -> None:
    LOG.info("%s worker starting (interval=%.2fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    post_exit_until = 0.0
    try:
        while True:
            now = time.monotonic()
            if not config.ENABLED:
                await asyncio.sleep(max(1.0, config.LOOP_INTERVAL_SEC))
                continue
            if not is_market_open():
                await asyncio.sleep(1.0)
                continue
            if not can_trade("scalp"):
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if now < cooldown_until or now < post_exit_until:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            positions = pos_manager.get_open_positions()
            scalp = positions.get("scalp") or {}
            open_trades = scalp.get("open_trades") or []
            active = [
                tr
                for tr in open_trades
                if (tr.get("entry_thesis") or {}).get("strategy_tag") == config.STRATEGY_TAG
            ]
            if len(active) >= config.MAX_ACTIVE_TRADES:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            ticks = tick_window.recent_ticks(
                seconds=config.ENTRY_WINDOW_SEC,
                limit=config.ENTRY_MAX_TICKS,
            )
            if len(ticks) < config.ENTRY_MIN_TICKS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            move_pips, abs_move, span_sec, speed_pps = _recent_move(config.ENTRY_WINDOW_SEC)

            direction = "long" if move_pips > 0 else "short"

            tick = ticks[-1]
            spread_pips = _spread_pips(tick)
            if spread_pips > config.ENTRY_MAX_SPREAD_PIPS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            fac = all_factors()
            fac_m1 = fac.get("M1") or {}
            fac_m5 = fac.get("M5") or {}
            fac_h4 = fac.get("H4") or {}
            atr_pips = _float(fac_m1.get("atr_pips"))
            if atr_pips is None:
                atr_val = _float(fac_m1.get("atr"))
                atr_pips = atr_val * 100.0 if atr_val is not None else None

            if atr_pips is None or atr_pips < config.ENTRY_MIN_ATR_PIPS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            min_move_pips = max(config.ENTRY_MIN_MOVE_PIPS, atr_pips * config.ENTRY_MIN_ATR_MULT)
            if abs_move < min_move_pips or speed_pps < config.ENTRY_MIN_SPEED_PPS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if abs_move > config.ENTRY_MAX_MOVE_PIPS or abs_move > atr_pips * config.ENTRY_MAX_ATR_MULT:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            recent_move, recent_abs, _, _ = _recent_move(config.ENTRY_RECENT_WINDOW_SEC)
            if abs_move > 0 and recent_abs / abs_move < config.ENTRY_RECENT_MIN_RATIO:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if config.ENTRY_RECENT_DIR_CONFIRM and recent_move and move_pips and recent_move * move_pips <= 0:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            latest_mid = _mid_from_tick(tick, _float(fac_m1.get("close")) or 0.0)
            if latest_mid <= 0:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            tick_range = _tick_range(ticks)
            if tick_range:
                high, low = tick_range
                if direction == "long":
                    retrace_pips = (high - latest_mid) / config.PIP_VALUE
                else:
                    retrace_pips = (latest_mid - low) / config.PIP_VALUE
                retrace_limit = max(config.ENTRY_RETRACE_PIPS, abs_move * config.ENTRY_RETRACE_RATIO)
                if retrace_pips >= retrace_limit:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

            rsi = _float(fac_m1.get("rsi"))
            if rsi is not None:
                if direction == "long" and rsi >= config.ENTRY_RSI_MAX_LONG:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue
                if direction == "short" and rsi <= config.ENTRY_RSI_MIN_SHORT:
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

            high = _float(fac_m1.get("high"))
            low = _float(fac_m1.get("low"))
            open_ = _float(fac_m1.get("open"))
            close_ = _float(fac_m1.get("close"))
            body_pips = None
            if open_ is not None and close_ is not None:
                body_pips = (close_ - open_) / config.PIP_VALUE
            if None not in (high, low, open_, close_):
                range_pips = (high - low) / config.PIP_VALUE
                if range_pips >= config.ENTRY_WICK_MIN_RANGE_PIPS and range_pips > 0:
                    upper_wick = (high - max(open_, close_)) / config.PIP_VALUE
                    lower_wick = (min(open_, close_) - low) / config.PIP_VALUE
                    if direction == "long":
                        wick_ratio = upper_wick / range_pips
                    else:
                        wick_ratio = lower_wick / range_pips
                    if wick_ratio >= config.ENTRY_MAX_WICK_RATIO:
                        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                        continue
                if body_pips is not None and config.ENTRY_BODY_COUNTER_MAX_PIPS > 0:
                    if direction == "long" and body_pips <= -config.ENTRY_BODY_COUNTER_MAX_PIPS:
                        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                        continue
                    if direction == "short" and body_pips >= config.ENTRY_BODY_COUNTER_MAX_PIPS:
                        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                        continue

            trend_dir = None
            trend_gap_pips = None
            trend_adx = None
            if config.ENTRY_TREND_BLOCK_COUNTER:
                trend_fac = fac_m5 if config.ENTRY_TREND_USE_M5 else fac_m1
                trend = _trend_state(trend_fac, latest_mid)
                if trend:
                    trend_dir, trend_gap_pips, trend_adx = trend
                    if (
                        abs(trend_gap_pips) >= config.ENTRY_TREND_GAP_PIPS
                        and trend_adx >= config.ENTRY_TREND_ADX_MIN
                        and direction != trend_dir
                    ):
                        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                        continue

            range_active = False
            try:
                range_active = bool(detect_range_mode(fac_m1, fac_h4).active)
            except Exception:
                range_active = False
            if range_active:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            perf = perf_guard.is_allowed(config.STRATEGY_TAG, "scalp")
            if not perf.allowed:
                LOG.info("%s perf_guard blocked: %s", config.LOG_PREFIX, perf.reason)
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            sl_pips, tp_pips = _calc_sl_tp(abs_move, atr_pips)
            if direction == "long":
                sl_price = latest_mid - sl_pips * config.PIP_VALUE
                tp_price = latest_mid + tp_pips * config.PIP_VALUE
            else:
                sl_price = latest_mid + sl_pips * config.PIP_VALUE
                tp_price = latest_mid - tp_pips * config.PIP_VALUE
            sl_price, tp_price = clamp_sl_tp(latest_mid, sl_price, tp_price, direction == "long")

            try:
                snap = get_account_snapshot()
            except Exception:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            equity = float(snap.nav or snap.balance or 0.0)
            if equity <= 0:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            risk_override = config.RISK_PCT_OVERRIDE if config.RISK_PCT_OVERRIDE > 0 else None
            lot = allowed_lot(
                equity,
                sl_pips=sl_pips,
                margin_available=float(snap.margin_available or 0.0),
                price=latest_mid,
                margin_rate=float(snap.margin_rate or 0.0),
                risk_pct_override=risk_override,
                pocket="scalp",
                strategy_tag=config.STRATEGY_TAG,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
            )
            if lot <= 0:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            lot = min(lot, config.MAX_LOT)
            units = int(round(lot * 100000))
            if units < config.MIN_UNITS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            size_mult = _size_mult(abs_move, speed_pps, min_move_pips)
            units = int(round(units * size_mult))
            if units < config.MIN_UNITS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            sign = 1 if direction == "long" else -1
            units *= sign

            confidence = int(min(95, max(55, 55 + (abs_move - config.ENTRY_MIN_MOVE_PIPS) * 4)))
            client_id = _client_order_id(config.STRATEGY_TAG)
            thesis = _build_thesis(
                direction=direction,
                move_pips=abs_move,
                min_move_pips=min_move_pips,
                recent_move_pips=recent_move,
                span_sec=span_sec,
                speed_pps=speed_pps,
                spread_pips=spread_pips,
                atr_pips=atr_pips,
                range_active=range_active,
                trend_dir=trend_dir,
                trend_gap_pips=trend_gap_pips,
                trend_adx=trend_adx,
                body_pips=body_pips,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                confidence=confidence,
            )

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=client_id,
                    strategy_tag=config.STRATEGY_TAG,
                    entry_thesis=thesis,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s order error dir=%s exc=%s", config.LOG_PREFIX, direction, exc)
                cooldown_until = time.monotonic() + config.COOLDOWN_SEC
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            if trade_id:
                LOG.info(
                    "%s entry trade=%s dir=%s units=%s price=%.3f move=%.2fp speed=%.2f",
                    config.LOG_PREFIX,
                    trade_id,
                    direction,
                    units,
                    latest_mid,
                    abs_move,
                    speed_pps,
                )
                cooldown_until = time.monotonic() + config.COOLDOWN_SEC
                post_exit_until = time.monotonic() + config.POST_EXIT_COOLDOWN_SEC
            else:
                cooldown_until = time.monotonic() + config.COOLDOWN_SEC

            await asyncio.sleep(config.LOOP_INTERVAL_SEC)

    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:  # noqa: BLE001
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)


if __name__ == "__main__":
    asyncio.run(vol_spike_rider_worker())
