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
from typing import Dict, Optional

import httpx

from execution.order_manager import close_trade, market_order, set_trade_protections
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from execution.stage_tracker import StageTracker
from execution.position_manager import PositionManager
from market_data import spread_monitor, tick_window
from utils.metrics_logger import log_metric
from utils.secrets import get_secret

from . import config
from .rate_limiter import SlidingWindowRateLimiter
from .signal import SignalFeatures, evaluate_signal, extract_features
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
    sl_price: float
    tp_price: float


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    iso = value.replace("Z", "+00:00")
    if "." not in iso:
        return datetime.fromisoformat(iso)
    head, frac_and_tz = iso.split(".", 1)
    tz = "+00:00"
    if "+" in frac_and_tz:
        frac, tz_tail = frac_and_tz.split("+", 1)
        tz = "+" + tz_tail
    elif "-" in frac_and_tz[6:]:
        frac, tz_tail = frac_and_tz.split("-", 1)
        tz = "-" + tz_tail
    else:
        frac = frac_and_tz
    frac = (''.join(ch for ch in frac if ch.isdigit())[:6]).ljust(6, "0")
    return datetime.fromisoformat(f"{head}.{frac}{tz}")


class _SnapshotTick:
    __slots__ = ("bid", "ask", "time")

    def __init__(self, bid: float, ask: float, time_val: datetime) -> None:
        self.bid = bid
        self.ask = ask
        self.time = time_val


try:
    _OANDA_TOKEN = get_secret("oanda_token")
    _OANDA_ACCOUNT = get_secret("oanda_account_id")
    try:
        _OANDA_PRACTICE = get_secret("oanda_practice").lower() == "true"
    except KeyError:
        _OANDA_PRACTICE = False
except Exception as exc:  # pragma: no cover - secrets must be present
    logging.error("%s failed to load OANDA secrets: %s", config.LOG_PREFIX_TICK, exc)
    _OANDA_TOKEN = ""
    _OANDA_ACCOUNT = ""
    _OANDA_PRACTICE = False

_PRICING_HOST = (
    "https://api-fxpractice.oanda.com" if _OANDA_PRACTICE else "https://api-fxtrade.oanda.com"
)
_PRICING_URL = f"{_PRICING_HOST}/v3/accounts/{_OANDA_ACCOUNT}/pricing"
_PRICING_HEADERS = {"Authorization": f"Bearer {_OANDA_TOKEN}"} if _OANDA_TOKEN else {}


def _is_off_hours(now_utc: datetime) -> bool:
    jst = now_utc + timedelta(hours=9)
    start = config.JST_OFF_HOURS_START
    end = config.JST_OFF_HOURS_END
    if start <= end:
        return start <= jst.hour < end
    return jst.hour >= start or jst.hour < end


async def _fetch_price_snapshot(logger: logging.Logger) -> Optional[_SnapshotTick]:
    if not _OANDA_TOKEN or not _OANDA_ACCOUNT:
        return None
    params = {"instruments": "USD_JPY"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(_PRICING_URL, headers=_PRICING_HEADERS, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s pricing snapshot failed: %s", config.LOG_PREFIX_TICK, exc)
        return None
    prices = payload.get("prices") or []
    if not prices:
        return None
    price = prices[0]
    bids = price.get("bids") or []
    asks = price.get("asks") or []
    if not bids or not asks:
        return None
    try:
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        ts = _parse_time(price.get("time", datetime.utcnow().isoformat() + "Z"))
    except Exception as exc:
        logger.warning("%s pricing snapshot parse error: %s", config.LOG_PREFIX_TICK, exc)
        return None
    tick = _SnapshotTick(bid=bid, ask=ask, time_val=ts)
    logger.warning(
        "%s snapshot bid=%.3f ask=%.3f", config.LOG_PREFIX_TICK, bid, ask
    )
    try:
        spread_monitor.update_from_tick(tick)
        tick_window.record(tick)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s failed to record snapshot tick: %s", config.LOG_PREFIX_TICK, exc)
    return tick


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
    active_trades: dict[str, ActiveTrade] = {}
    last_sync = time.monotonic()
    spread_block_logged = False
    dd_block_logged = False
    off_hours_logged = False
    order_backoff: float = 0.0
    next_order_after: float = 0.0
    last_snapshot_fetch: float = 0.0

    loop_counter = 0
    try:
        while True:
            loop_start = time.monotonic()
            now = _now_utc()
            loop_counter += 1
            if loop_counter % 200 == 0:
                logger.warning(
                    "%s loop=%d active=%d", config.LOG_PREFIX_TICK, loop_counter, len(active_trades)
                )

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

            snapshot_needed = False
            age_ms = None
            if spread_state:
                age_ms = spread_state.get("age_ms")
                if spread_state.get("stale"):
                    snapshot_needed = True
                elif age_ms is not None and age_ms > config.STALE_TICK_MAX_SEC * 1000:
                    snapshot_needed = True
            else:
                snapshot_needed = True

            monotonic_now = time.monotonic()
            if snapshot_needed and monotonic_now - last_snapshot_fetch >= config.SNAPSHOT_MIN_INTERVAL_SEC:
                fetched_tick = await _fetch_price_snapshot(logger)
                last_snapshot_fetch = monotonic_now
                if fetched_tick:
                    spread_state = spread_monitor.get_state()
                    spread_pips = (
                        spread_state.get("spread_pips") if spread_state else spread_pips
                    )

            features = extract_features(spread_pips)
            if not features:
                recent = tick_window.recent_ticks(10.0, limit=5)
                if recent:
                    latest = recent[-1]
                    first = recent[0]
                    latest_mid = float(latest.get("mid") or 0.0)
                    first_mid = float(first.get("mid") or latest_mid)
                    span = float(latest.get("epoch", 0.0)) - float(first.get("epoch", 0.0))
                    momentum = (latest_mid - first_mid) / config.PIP_VALUE if span != 0 else 0.0
                    range_pips = abs(latest_mid - first_mid) / config.PIP_VALUE
                    features = SignalFeatures(
                        latest_mid=latest_mid,
                        spread_pips=spread_pips,
                        momentum_pips=momentum,
                        short_momentum_pips=momentum,
                        range_pips=range_pips,
                        tick_count=len(recent),
                        span_seconds=span,
                    )
                if not features:
                    if loop_counter % 40 == 0:
                        logger.debug(
                            "%s insufficient features spread=%.3f ticks=%d",
                            config.LOG_PREFIX_TICK,
                            spread_pips,
                            len(tick_window.recent_ticks(config.LONG_WINDOW_SEC, limit=10)),
                        )
                    await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                    continue

            skip_new_entry = False
            for trade_id, active in list(active_trades.items()):
                pips_gain = _pips(features.latest_mid - active.entry_price)
                if active.side == "short":
                    pips_gain = -pips_gain
                elapsed = time.monotonic() - active.opened_monotonic
                if pips_gain <= -config.MAX_DRAWDOWN_CLOSE_PIPS or (
                    elapsed >= config.TIMEOUT_SEC and pips_gain < config.TIMEOUT_MIN_GAIN_PIPS
                ):
                    if rate_limiter.allow():
                        reason_txt = (
                            "drawdown"
                            if pips_gain <= -config.MAX_DRAWDOWN_CLOSE_PIPS
                            else "timeout"
                        )
                        logger.info(
                            "%s close trade=%s side=%s reason=%s pnl=%.2fp elapsed=%.1fs",
                            config.LOG_PREFIX_TICK,
                            trade_id,
                            active.side,
                            reason_txt,
                            pips_gain,
                            elapsed,
                        )
                        rate_limiter.record()
                        try:
                            await close_trade(trade_id)
                        finally:
                            stage_tracker.set_cooldown(
                                "scalp_fast",
                                active.side,
                                reason="manual_exit",
                                seconds=int(config.ENTRY_COOLDOWN_SEC),
                                now=now,
                            )
                            active_trades.pop(trade_id, None)
                        skip_new_entry = True
                    else:
                        skip_new_entry = True
            if skip_new_entry:
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
                    updated: dict[str, ActiveTrade] = {}
                    for tr in open_trades:
                        trade_id = str(tr.get("trade_id"))
                        if not trade_id:
                            continue
                        direction = tr.get("side") or ("long" if tr.get("units", 0) > 0 else "short")
                        units = int(tr.get("units", 0) or 0)
                        entry_price = float(tr.get("price", features.latest_mid))
                        client_id = str(tr.get("client_id") or "")
                        existing = active_trades.get(trade_id)
                        opened_monotonic = existing.opened_monotonic if existing else time.monotonic()
                        opened_at = existing.opened_at if existing else _now_utc()
                        updated[trade_id] = ActiveTrade(
                            trade_id=trade_id,
                            side=direction,
                            units=units,
                            entry_price=entry_price,
                            opened_at=opened_at,
                            opened_monotonic=opened_monotonic,
                            client_order_id=client_id,
                            sl_price=existing.sl_price if existing else entry_price,
                            tp_price=existing.tp_price if existing else entry_price,
                        )
                    active_trades = updated

            # Max concurrent trades guard
            if len(active_trades) >= config.MAX_ACTIVE_TRADES:
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

            state = spread_monitor.get_state()
            bid_quote = float(state.get("bid") or 0.0) if state else 0.0
            ask_quote = float(state.get("ask") or 0.0) if state else 0.0
            last_ticks = tick_window.recent_ticks(3.0, limit=1)
            if last_ticks:
                last_tick = last_ticks[-1]
                bid_quote = float(last_tick.get("bid") or bid_quote or features.latest_mid)
                ask_quote = float(last_tick.get("ask") or ask_quote or features.latest_mid)
            expected_entry_price = (
                ask_quote if direction == "long" else bid_quote
            ) or features.latest_mid
            spread_padding = max(features.spread_pips, config.TP_SPREAD_BUFFER_PIPS)
            tp_margin = max(config.TP_SAFE_MARGIN_PIPS, features.spread_pips * 0.5)
            tp_pips = config.TP_BASE_PIPS + spread_padding + tp_margin
            sl_pips = config.SL_PIPS
            entry_price = expected_entry_price
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
                "entry_price_expect": round(entry_price, 5),
                "sl_price_initial": round(sl_price, 5),
                "tp_price_initial": round(tp_price, 5),
            }

            try:
                trade_id, executed_price = await market_order(
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
            actual_entry_price = executed_price if executed_price is not None else entry_price
            sl_adjust_pips = config.SL_PIPS + config.SL_POST_ADJUST_BUFFER_PIPS
            if direction == "long":
                actual_sl_price = round(actual_entry_price - sl_adjust_pips * config.PIP_VALUE, 3)
                actual_tp_price = round(actual_entry_price + tp_pips * config.PIP_VALUE, 3)
            else:
                actual_sl_price = round(actual_entry_price + sl_adjust_pips * config.PIP_VALUE, 3)
                actual_tp_price = round(actual_entry_price - tp_pips * config.PIP_VALUE, 3)
            set_ok = await set_trade_protections(
                trade_id,
                sl_price=actual_sl_price,
                tp_price=actual_tp_price,
            )
            if not set_ok:
                logger.warning(
                    "%s protection update failed trade=%s", config.LOG_PREFIX_TICK, trade_id
                )
            active_trades[trade_id] = ActiveTrade(
                trade_id=trade_id,
                side=direction,
                units=units,
                entry_price=actual_entry_price,
                opened_at=now,
                opened_monotonic=monotonic_now,
                client_order_id=client_id,
                sl_price=actual_sl_price,
                tp_price=actual_tp_price,
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
