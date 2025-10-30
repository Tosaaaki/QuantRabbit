"""
Async worker that drives ultra-short-term scalping based on tick data.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from execution.order_manager import close_trade, market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from execution.stage_tracker import StageTracker
from execution.position_manager import PositionManager
from market_data import spread_monitor
from utils.metrics_logger import log_metric

from . import config
from .rate_limiter import SlidingWindowRateLimiter
from .signal import evaluate_signal, extract_features
from .state import FastScalpState


@dataclass
class ActiveTrade:
    trade_id: str
    side: str
    units: int
    entry_price: float
    opened_at: datetime
    opened_monotonic: float
    client_order_id: str


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_off_hours(now_utc: datetime) -> bool:
    jst = now_utc + timedelta(hours=9)
    start = config.JST_OFF_HOURS_START
    end = config.JST_OFF_HOURS_END
    if start <= end:
        return start <= jst.hour < end
    return jst.hour >= start or jst.hour < end


def _build_client_order_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-fast-{ts_ms}-{side[0]}{digest}"


def _pips(delta_price: float) -> float:
    return delta_price / config.PIP_VALUE


async def fast_scalp_worker(shared_state: FastScalpState) -> None:
    logger = logging.getLogger(__name__)
    if not config.FAST_SCALP_ENABLED:
        logger.info("%s disabled via env, worker idling.", config.LOG_PREFIX_TICK)
        while True:
            await asyncio.sleep(30.0)

    rate_limiter = SlidingWindowRateLimiter(
        config.MAX_ORDERS_PER_MINUTE, config.MIN_ORDER_SPACING_SEC
    )
    stage_tracker = StageTracker()
    pos_manager = PositionManager()
    active: Optional[ActiveTrade] = None
    last_sync = time.monotonic()
    spread_block_logged = False
    dd_block_logged = False
    off_hours_logged = False
    order_backoff: float = 0.0
    next_order_after: float = 0.0

    try:
        while True:
            loop_start = time.monotonic()
            now = _now_utc()

            if _is_off_hours(now):
                if not off_hours_logged:
                    logger.info("%s pause during JST off-hours window.", config.LOG_PREFIX_TICK)
                    off_hours_logged = True
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            off_hours_logged = False

            if not can_trade("scalp_fast"):
                if not dd_block_logged:
                    logger.warning(
                        "%s drawdown guard prevents trading; waiting.",
                        config.LOG_PREFIX_TICK,
                    )
                    dd_block_logged = True
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            dd_block_logged = False

            blocked, remain, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = spread_state["spread_pips"] if spread_state else 0.0
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if not spread_block_logged:
                    logger.info(
                        "%s skip due to spread %.2fp (remain=%ss reason=%s)",
                        config.LOG_PREFIX_TICK,
                        spread_pips,
                        remain,
                        spread_reason or "guard_active",
                    )
                    log_metric(
                        "fast_scalp_skip",
                        float(spread_pips),
                        tags={"reason": "spread", "guard": spread_reason or ""},
                        ts=now,
                    )
                    spread_block_logged = True
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            spread_block_logged = False

            snapshot = shared_state.snapshot()

            features = extract_features(spread_pips)
            if not features:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            # Manage existing active trade (time-stop / protection)
            if active:
                pips_gain = _pips(features.latest_mid - active.entry_price)
                if active.side == "short":
                    pips_gain = -pips_gain
                elapsed = time.monotonic() - active.opened_monotonic
                if pips_gain <= -config.MAX_DRAWDOWN_CLOSE_PIPS or (
                    elapsed >= config.TIMEOUT_SEC and pips_gain < config.TIMEOUT_MIN_GAIN_PIPS
                ):
                    if rate_limiter.allow():
                        logger.info(
                            "%s close trade=%s side=%s reason=%s pnl=%.2fp elapsed=%.1fs",
                            config.LOG_PREFIX_TICK,
                            active.trade_id,
                            active.side,
                            "drawdown"
                            if pips_gain <= -config.MAX_DRAWDOWN_CLOSE_PIPS
                            else "timeout",
                            pips_gain,
                            elapsed,
                        )
                        rate_limiter.record()
                        try:
                            await close_trade(active.trade_id)
                        finally:
                            stage_tracker.set_cooldown(
                                "scalp_fast",
                                active.side,
                                reason="manual_exit",
                                seconds=int(config.ENTRY_COOLDOWN_SEC),
                                now=now,
                            )
                            active = None
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

            # Periodic reconciliation with live positions
            now_monotonic = time.monotonic()
            if now_monotonic - last_sync >= config.SYNC_INTERVAL_SEC:
                last_sync = now_monotonic
                try:
                    positions = pos_manager.get_open_positions()
                except Exception as exc:
                    logger.warning("%s sync positions failed: %s", config.LOG_PREFIX_TICK, exc)
                else:
                    pocket = positions.get("scalp_fast")
                    open_trades = (pocket or {}).get("open_trades") or []
                    if not open_trades:
                        active = None
                    else:
                        first = open_trades[0]
                        trade_id = str(first.get("trade_id"))
                        if not active or active.trade_id != trade_id:
                            direction = first.get("side") or ("long" if first.get("units", 0) > 0 else "short")
                            active = ActiveTrade(
                                trade_id=trade_id,
                                side=direction,
                                units=int(first.get("units", 0)),
                                entry_price=float(first.get("price", features.latest_mid)),
                                opened_at=_now_utc(),
                                opened_monotonic=time.monotonic(),
                                client_order_id=str(first.get("client_id") or ""),
                            )

            # If trade still active skip new entries
            if active:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            action = evaluate_signal(features)
            if not action:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            direction = "long" if action == "OPEN_LONG" else "short"
            cooldown = stage_tracker.get_cooldown("scalp_fast", direction, now=now)
            if cooldown:
                logger.debug(
                    "%s cooldown active pocket=scalp_fast dir=%s until=%s reason=%s",
                    config.LOG_PREFIX_TICK,
                    direction,
                    cooldown.cooldown_until,
                    cooldown.reason,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            monotonic_now = time.monotonic()
            if monotonic_now < next_order_after:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            if not rate_limiter.allow():
                logger.info(
                    "%s skip signal due to rate limit side=%s mom=%.2fp",
                    config.LOG_PREFIX_TICK,
                    direction,
                    features.momentum_pips,
                )
                log_metric(
                    "fast_scalp_skip",
                    features.momentum_pips,
                    tags={"reason": "rate_limit", "side": direction},
                    ts=now,
                )
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            lot = allowed_lot(
                snapshot.account_equity,
                sl_pips=config.SL_PIPS,
                margin_available=snapshot.margin_available,
                price=features.latest_mid,
                margin_rate=snapshot.margin_rate,
                risk_pct_override=snapshot.risk_pct_override,
            )
            lot = min(lot, config.MAX_LOT)
            if lot <= 0.0:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            min_lot = config.MIN_UNITS / 100000.0
            adjusted_lot = False
            if 0.0 < lot < min_lot:
                lot = min(config.MAX_LOT, min_lot)
                adjusted_lot = True
            units = int(round(lot * 100000.0))
            if units < config.MIN_UNITS:
                units = config.MIN_UNITS
                adjusted_lot = True
            if direction == "short":
                units = -units
            if abs(units) < config.MIN_UNITS:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue
            if adjusted_lot:
                logger.debug(
                    "%s adjusted lot to %.3f units=%d (min=%d)",
                    config.LOG_PREFIX_TICK,
                    lot,
                    units,
                    config.MIN_UNITS,
                )

            client_id = _build_client_order_id(direction)

            spread_padding = max(features.spread_pips, config.TP_SPREAD_BUFFER_PIPS)
            tp_margin = max(config.TP_SAFE_MARGIN_PIPS, features.spread_pips * 0.5)
            tp_pips = config.TP_BASE_PIPS + spread_padding + tp_margin
            sl_pips = config.SL_PIPS
            entry_price = features.latest_mid
            if direction == "long":
                sl_price = entry_price - sl_pips * config.PIP_VALUE
                tp_price = entry_price + tp_pips * config.PIP_VALUE
            else:
                sl_price = entry_price + sl_pips * config.PIP_VALUE
                tp_price = entry_price - tp_pips * config.PIP_VALUE
            sl_price, tp_price = clamp_sl_tp(entry_price, sl_price, tp_price, direction == "long")

            thesis = {
                "momentum_pips": round(features.momentum_pips, 3),
                "short_momentum_pips": round(features.short_momentum_pips, 3),
                "range_pips": round(features.range_pips, 3),
                "spread_pips": round(features.spread_pips, 3),
                "tick_count": features.tick_count,
                "weight_scalp": snapshot.weight_scalp,
                "tp_pips": round(tp_pips, 3),
            }

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price,
                    tp_price,
                    "scalp_fast",
                    client_order_id=client_id,
                    entry_thesis=thesis,
                )
            except Exception as exc:
                logger.error(
                    "%s order error side=%s exc=%s",
                    config.LOG_PREFIX_TICK,
                    direction,
                    exc,
                )
                order_backoff = max(order_backoff * 1.8, 0.3) if order_backoff else 0.3
                next_order_after = monotonic_now + order_backoff
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            if not trade_id:
                order_backoff = max(order_backoff * 1.8, 0.3) if order_backoff else 0.3
                next_order_after = monotonic_now + order_backoff
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                continue

            order_backoff = 0.0
            rate_limiter.record()
            stage_tracker.set_cooldown(
                "scalp_fast",
                direction,
                reason="entry",
                seconds=int(config.ENTRY_COOLDOWN_SEC),
                now=now,
            )
            active = ActiveTrade(
                trade_id=trade_id,
                side=direction,
                units=units,
                entry_price=entry_price,
                opened_at=now,
                opened_monotonic=monotonic_now,
                client_order_id=client_id,
            )
            log_metric(
                "fast_scalp_signal",
                features.momentum_pips,
                tags={"side": direction, "range_pips": f"{features.range_pips:.2f}"},
                ts=now,
            )
            logger.info(
                "%s open trade=%s side=%s units=%d tp=%.3f sl=%.3f mom=%.2fp range=%.2fp spread=%.2fp",
                config.LOG_PREFIX_TICK,
                trade_id,
                direction,
                units,
                tp_price or 0.0,
                sl_price or 0.0,
                features.momentum_pips,
                features.range_pips,
                features.spread_pips,
            )

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.05, config.LOOP_INTERVAL_SEC - elapsed))
    finally:
        stage_tracker.close()
        pos_manager.close()
        logger.info("%s worker shutdown", config.LOG_PREFIX_TICK)
