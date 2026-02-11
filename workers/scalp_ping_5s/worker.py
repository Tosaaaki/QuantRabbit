"""Entry-only 5s ping scalp worker.

This worker focuses on short-horizon entries and leaves exits to broker TP/SL
(or separate exit workers) by design.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Sequence

import httpx

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import spread_monitor, tick_window
from market_data.tick_fetcher import _parse_time
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from utils.secrets import get_secret
from workers.common.size_utils import scale_base_units
from workers.fast_scalp.rate_limiter import SlidingWindowRateLimiter

from . import config


LOG = logging.getLogger(__name__)


try:
    _OANDA_TOKEN = get_secret("oanda_token")
    _OANDA_ACCOUNT = get_secret("oanda_account_id")
    try:
        _OANDA_PRACTICE = get_secret("oanda_practice").lower() == "true"
    except Exception:
        _OANDA_PRACTICE = False
except Exception:
    _OANDA_TOKEN = ""
    _OANDA_ACCOUNT = ""
    _OANDA_PRACTICE = False

_PRICING_HOST = (
    "https://api-fxpractice.oanda.com"
    if _OANDA_PRACTICE
    else "https://api-fxtrade.oanda.com"
)
_PRICING_URL = f"{_PRICING_HOST}/v3/accounts/{_OANDA_ACCOUNT}/pricing"
_PRICING_HEADERS = {"Authorization": f"Bearer {_OANDA_TOKEN}"} if _OANDA_TOKEN else {}


def _mask_value(value: str, *, head: int = 4, tail: int = 2) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if len(text) <= head + tail:
        return "*" * len(text)
    return f"{text[:head]}***{text[-tail:]}"


@dataclass(slots=True)
class TickSignal:
    side: str
    confidence: int
    momentum_pips: float
    trigger_pips: float
    imbalance: float
    tick_rate: float
    span_sec: float
    tick_age_ms: float
    spread_pips: float
    bid: float
    ask: float
    mid: float


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _quotes_from_row(row: dict, fallback_mid: float = 0.0) -> tuple[float, float, float]:
    bid = _safe_float(row.get("bid"), 0.0)
    ask = _safe_float(row.get("ask"), 0.0)
    mid = _safe_float(row.get("mid"), 0.0)
    if mid <= 0.0 and bid > 0.0 and ask > 0.0:
        mid = (bid + ask) * 0.5
    if mid <= 0.0:
        mid = fallback_mid
    return bid, ask, mid


def _latest_spread_from_ticks(rows: Sequence[dict]) -> float:
    if not rows:
        return 0.0
    bid, ask, _ = _quotes_from_row(rows[-1])
    if bid <= 0.0 or ask <= 0.0 or ask < bid:
        return 0.0
    return (ask - bid) / config.PIP_VALUE


def _build_tick_signal(rows: Sequence[dict], spread_pips: float) -> Optional[TickSignal]:
    if len(rows) < config.MIN_TICKS:
        return None

    latest_epoch = _safe_float(rows[-1].get("epoch"), 0.0)
    if latest_epoch <= 0.0:
        return None

    tick_age_ms = max(0.0, (time.time() - latest_epoch) * 1000.0)
    if tick_age_ms > config.MAX_TICK_AGE_MS:
        return None

    window_cutoff = latest_epoch - config.SIGNAL_WINDOW_SEC
    signal_rows = [r for r in rows if _safe_float(r.get("epoch"), 0.0) >= window_cutoff]
    if len(signal_rows) < config.MIN_SIGNAL_TICKS:
        return None

    mids: list[float] = []
    epochs: list[float] = []
    fallback_mid = 0.0
    for row in signal_rows:
        bid, ask, mid = _quotes_from_row(row, fallback_mid=fallback_mid)
        if mid <= 0.0:
            continue
        epoch = _safe_float(row.get("epoch"), 0.0)
        if epoch <= 0.0:
            continue
        mids.append(mid)
        epochs.append(epoch)
        fallback_mid = mid

    if len(mids) < config.MIN_SIGNAL_TICKS:
        return None

    span_sec = max(0.001, epochs[-1] - epochs[0])
    momentum_pips = (mids[-1] - mids[0]) / config.PIP_VALUE

    up = 0
    down = 0
    for i in range(1, len(mids)):
        diff = mids[i] - mids[i - 1]
        if diff > 0:
            up += 1
        elif diff < 0:
            down += 1
    directional = max(1, up + down)
    imbalance = max(up, down) / directional
    tick_rate = max(0.0, (len(mids) - 1) / span_sec)

    bid, ask, mid = _quotes_from_row(signal_rows[-1], fallback_mid=mids[-1])
    if spread_pips <= 0.0:
        if bid > 0.0 and ask > 0.0 and ask >= bid:
            spread_pips = (ask - bid) / config.PIP_VALUE
        else:
            spread_pips = 0.0

    trigger_pips = max(
        config.MOMENTUM_TRIGGER_PIPS,
        spread_pips * config.MOMENTUM_SPREAD_MULT,
    )

    side: Optional[str] = None
    if (
        momentum_pips >= trigger_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.MIN_TICK_RATE
    ):
        side = "long"
    elif (
        momentum_pips <= -trigger_pips
        and imbalance >= config.IMBALANCE_MIN
        and tick_rate >= config.MIN_TICK_RATE
    ):
        side = "short"

    if side is None:
        return None

    strength = abs(momentum_pips) / max(0.01, trigger_pips)
    confidence = config.CONFIDENCE_FLOOR
    confidence += int(min(22.0, strength * 8.0))
    confidence += int(min(10.0, max(0.0, (imbalance - 0.5) * 25.0)))
    confidence += int(min(8.0, tick_rate))
    confidence = max(config.CONFIDENCE_FLOOR, min(config.CONFIDENCE_CEIL, confidence))

    return TickSignal(
        side=side,
        confidence=int(confidence),
        momentum_pips=momentum_pips,
        trigger_pips=trigger_pips,
        imbalance=imbalance,
        tick_rate=tick_rate,
        span_sec=span_sec,
        tick_age_ms=tick_age_ms,
        spread_pips=max(0.0, spread_pips),
        bid=bid,
        ask=ask,
        mid=mid,
    )


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.65
    if conf >= hi:
        return 1.15
    ratio = (conf - lo) / max(1.0, hi - lo)
    return 0.65 + ratio * 0.5


def _compute_targets(*, spread_pips: float, momentum_pips: float) -> tuple[float, float]:
    tp_pips = max(config.TP_BASE_PIPS, spread_pips + config.TP_NET_MIN_PIPS)
    tp_bonus_raw = max(0.0, abs(momentum_pips) - tp_pips)
    tp_pips += min(config.TP_MOMENTUM_BONUS_MAX, tp_bonus_raw * 0.25)
    tp_pips = min(config.TP_MAX_PIPS, tp_pips)

    sl_floor = spread_pips * config.SL_SPREAD_MULT + config.SL_SPREAD_BUFFER_PIPS
    sl_pips = max(config.SL_MIN_PIPS, config.SL_BASE_PIPS, sl_floor)
    if abs(momentum_pips) > tp_pips * 1.4:
        sl_pips *= 1.08
    sl_pips = max(config.SL_MIN_PIPS, min(config.SL_MAX_PIPS, sl_pips))

    return tp_pips, sl_pips


def _strategy_trade_counts(pocket_info: dict, strategy_tag: str) -> tuple[int, int, int]:
    open_trades = pocket_info.get("open_trades") if isinstance(pocket_info, dict) else None
    if not isinstance(open_trades, list):
        return 0, 0, 0

    tag_key = (strategy_tag or "").strip().lower()
    total = 0
    long_count = 0
    short_count = 0

    for tr in open_trades:
        if not isinstance(tr, dict):
            continue
        raw_tag = tr.get("strategy_tag")
        if not raw_tag:
            thesis = tr.get("entry_thesis")
            if isinstance(thesis, dict):
                raw_tag = thesis.get("strategy_tag") or thesis.get("strategy")
        if tag_key and str(raw_tag or "").strip().lower() != tag_key:
            continue
        units = int(_safe_float(tr.get("units"), 0.0))
        if units == 0:
            continue
        total += 1
        if units > 0:
            long_count += 1
        else:
            short_count += 1

    return total, long_count, short_count


def _client_order_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{config.STRATEGY_TAG}-{side}-{ts_ms}".encode("utf-8")).hexdigest()[:8]
    return f"qr-{ts_ms}-{config.STRATEGY_TAG[:12]}-{side[0]}{digest}"


async def _fetch_price_snapshot(logger: logging.Logger) -> bool:
    if not _OANDA_TOKEN or not _OANDA_ACCOUNT:
        return False

    params = {"instruments": "USD_JPY"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(_PRICING_URL, headers=_PRICING_HEADERS, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning("%s snapshot fetch failed: %s", config.LOG_PREFIX, exc)
        return False

    prices = payload.get("prices") or []
    if not prices:
        return False

    price = prices[0]
    bids = price.get("bids") or []
    asks = price.get("asks") or []
    if not bids or not asks:
        return False

    try:
        bid = float(bids[0]["price"])
        ask = float(asks[0]["price"])
        ts = _parse_time(price.get("time", datetime.datetime.utcnow().isoformat() + "Z"))
    except Exception as exc:
        logger.warning("%s snapshot parse failed: %s", config.LOG_PREFIX, exc)
        return False

    tick = SimpleNamespace(bid=bid, ask=ask, time=ts)
    try:
        spread_monitor.update_from_tick(tick)
        tick_window.record(tick)
    except Exception as exc:
        logger.warning("%s snapshot cache update failed: %s", config.LOG_PREFIX, exc)
        return False

    return True


async def scalp_ping_5s_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        while True:
            await asyncio.sleep(30.0)

    env_name = "practice" if _OANDA_PRACTICE else "live"
    LOG.info(
        "%s env=%s require_practice=%s account=%s",
        config.LOG_PREFIX,
        env_name,
        int(config.REQUIRE_PRACTICE),
        _mask_value(_OANDA_ACCOUNT),
    )
    if config.REQUIRE_PRACTICE and not _OANDA_PRACTICE:
        LOG.error(
            "%s blocked: practice account required (set SCALP_PING_5S_REQUIRE_PRACTICE=0 to bypass)",
            config.LOG_PREFIX,
        )
        while True:
            await asyncio.sleep(30.0)

    LOG.info(
        "%s start loop=%.2fs pocket=%s tag=%s",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        config.POCKET,
        config.STRATEGY_TAG,
    )

    pos_manager = PositionManager()
    rate_limiter = SlidingWindowRateLimiter(
        config.MAX_ORDERS_PER_MINUTE,
        config.MIN_ORDER_SPACING_SEC,
    )
    last_entry_mono = 0.0
    last_snapshot_fetch = 0.0

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_utc = datetime.datetime.utcnow()
            if not is_market_open(now_utc):
                continue
            if not can_trade(config.POCKET):
                continue

            blocked, remain, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = _safe_float((spread_state or {}).get("spread_pips"), 0.0)
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if blocked and remain > 0 and remain % 5 == 0:
                    LOG.info(
                        "%s spread block remain=%ss spread=%.2fp reason=%s",
                        config.LOG_PREFIX,
                        remain,
                        spread_pips,
                        spread_reason or "guard",
                    )
                continue

            ticks = tick_window.recent_ticks(
                seconds=config.WINDOW_SEC,
                limit=int(config.WINDOW_SEC * 25) + 50,
            )

            if spread_pips <= 0.0:
                spread_pips = _latest_spread_from_ticks(ticks)

            if len(ticks) < config.MIN_TICKS:
                now_mono = time.monotonic()
                if (
                    config.SNAPSHOT_FALLBACK_ENABLED
                    and now_mono - last_snapshot_fetch >= config.SNAPSHOT_MIN_INTERVAL_SEC
                ):
                    if await _fetch_price_snapshot(LOG):
                        last_snapshot_fetch = now_mono
                continue

            signal = _build_tick_signal(ticks, spread_pips)
            if signal is None:
                continue
            if signal.spread_pips > config.MAX_SPREAD_PIPS:
                continue

            now_mono = time.monotonic()
            if now_mono - last_entry_mono < config.ENTRY_COOLDOWN_SEC:
                continue
            if not rate_limiter.allow(now_mono):
                continue

            positions = pos_manager.get_open_positions()
            pocket_info = positions.get(config.POCKET) or {}
            active_total, active_long, active_short = _strategy_trade_counts(
                pocket_info,
                config.STRATEGY_TAG,
            )
            if active_total >= config.MAX_ACTIVE_TRADES:
                continue
            if signal.side == "long" and active_long >= config.MAX_PER_DIRECTION:
                continue
            if signal.side == "short" and active_short >= config.MAX_PER_DIRECTION:
                continue

            long_units = _safe_float(pocket_info.get("long_units"), 0.0)
            short_units = _safe_float(pocket_info.get("short_units"), 0.0)
            if config.NO_HEDGE_ENTRY:
                if signal.side == "long" and short_units > 0.0:
                    continue
                if signal.side == "short" and long_units > 0.0:
                    continue

            snap = get_account_snapshot(cache_ttl_sec=0.5)
            nav = max(_safe_float(snap.nav, 0.0), 1.0)
            balance = max(_safe_float(snap.balance, 0.0), nav)
            margin_available = max(_safe_float(snap.margin_available, 0.0), 0.0)
            margin_rate = _safe_float(snap.margin_rate, 0.0)
            free_ratio = _safe_float(snap.free_margin_ratio, 0.0)

            if free_ratio > 0.0 and free_ratio < config.MIN_FREE_MARGIN_RATIO:
                continue

            tp_pips, sl_pips = _compute_targets(
                spread_pips=signal.spread_pips,
                momentum_pips=signal.momentum_pips,
            )
            conf_mult = _confidence_scale(signal.confidence)
            strength_mult = max(
                0.75,
                min(1.25, abs(signal.momentum_pips) / max(0.1, signal.trigger_pips)),
            )
            base_units = int(
                round(
                    scale_base_units(
                        config.BASE_ENTRY_UNITS,
                        equity=balance,
                        ref_equity=balance,
                        min_units=config.MIN_UNITS,
                        max_units=config.MAX_UNITS,
                        env_prefix=config.ENV_PREFIX,
                    )
                )
            )
            units = int(round(base_units * conf_mult * strength_mult))

            lot = allowed_lot(
                nav,
                sl_pips,
                margin_available=margin_available,
                margin_rate=margin_rate,
                price=signal.mid,
                pocket=config.POCKET,
                side=signal.side,
                open_long_units=long_units,
                open_short_units=short_units,
                strategy_tag=config.STRATEGY_TAG,
            )
            units_risk = int(round(max(0.0, lot) * 100000))
            units = min(units, units_risk, config.MAX_UNITS)
            if units < config.MIN_UNITS:
                continue
            units = abs(units)
            if signal.side == "short":
                units = -units

            entry_price = signal.ask if signal.side == "long" else signal.bid
            if entry_price <= 0.0:
                entry_price = signal.mid
            if entry_price <= 0.0:
                continue

            if signal.side == "long":
                sl_price = (
                    entry_price - sl_pips * config.PIP_VALUE if config.USE_SL else None
                )
                tp_price = entry_price + tp_pips * config.PIP_VALUE
            else:
                sl_price = (
                    entry_price + sl_pips * config.PIP_VALUE if config.USE_SL else None
                )
                tp_price = entry_price - tp_pips * config.PIP_VALUE

            sl_price, tp_price = clamp_sl_tp(
                entry_price,
                sl_price,
                tp_price,
                is_buy=signal.side == "long",
                strategy_tag=config.STRATEGY_TAG,
                pocket=config.POCKET,
            )

            entry_thesis = {
                "strategy_tag": config.STRATEGY_TAG,
                "env_prefix": config.ENV_PREFIX,
                "signal_window_sec": config.SIGNAL_WINDOW_SEC,
                "window_sec": config.WINDOW_SEC,
                "momentum_pips": round(signal.momentum_pips, 3),
                "trigger_pips": round(signal.trigger_pips, 3),
                "imbalance": round(signal.imbalance, 3),
                "tick_rate": round(signal.tick_rate, 3),
                "tick_age_ms": round(signal.tick_age_ms, 1),
                "spread_pips": round(signal.spread_pips, 3),
                "confidence": int(signal.confidence),
                "tp_pips": round(tp_pips, 3),
                "sl_pips": round(sl_pips, 3),
                "entry_mode": "market_ping_5s",
            }

            client_order_id = _client_order_id(signal.side)
            result = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=client_order_id,
                strategy_tag=config.STRATEGY_TAG,
                confidence=int(signal.confidence),
                entry_thesis=entry_thesis,
            )
            if result:
                pos_manager.register_open_trade(str(result), config.POCKET, client_order_id)
            last_entry_mono = now_mono
            rate_limiter.record(now_mono)

            LOG.info(
                "%s open side=%s units=%s conf=%d mom=%.2fp trig=%.2fp sp=%.2fp tp=%.2fp sl=%.2fp res=%s",
                config.LOG_PREFIX,
                signal.side,
                units,
                signal.confidence,
                signal.momentum_pips,
                signal.trigger_pips,
                signal.spread_pips,
                tp_pips,
                sl_pips,
                result or "none",
            )

    except asyncio.CancelledError:
        LOG.info("%s cancelled", config.LOG_PREFIX)
        raise
    finally:
        pos_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(scalp_ping_5s_worker())
