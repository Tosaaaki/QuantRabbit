"""Pullback strategy operating on 5-second synthetic candles."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from market_data import spread_monitor, tick_window

from . import config

LOG = logging.getLogger(__name__)


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-pullback-s5-{ts_ms}-{side[0]}{digest}"


def _z_score(values: Sequence[float]) -> Optional[float]:
    if len(values) < 20:
        return None
    sample = values[-20:]
    mean_val = sum(sample) / len(sample)
    variance = sum((v - mean_val) ** 2 for v in sample) / max(len(sample) - 1, 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (sample[-1] - mean_val) / std


def _rsi(values: Sequence[float], period: int) -> Optional[float]:
    if len(values) <= 1:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    period = min(period, len(gains))
    if period <= 0:
        return 50.0
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0.0:
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_from_closes(values: Sequence[float], period: int) -> float:
    if len(values) <= 1:
        return 0.0
    true_ranges = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    period = min(period, len(true_ranges))
    if period <= 0:
        return 0.0
    return (sum(true_ranges[-period:]) / period) / config.PIP_VALUE


def _bucket_ticks(ticks: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {}
    bucket_span = config.BUCKET_SECONDS
    for tick in ticks:
        mid = tick.get("mid")
        epoch = tick.get("epoch")
        if mid is None or epoch is None:
            continue
        bucket_id = int(epoch // bucket_span)
        candle = buckets.get(bucket_id)
        if candle is None:
            candle = {
                "start": bucket_id * bucket_span,
                "end": (bucket_id + 1) * bucket_span,
                "open": float(mid),
                "high": float(mid),
                "low": float(mid),
                "close": float(mid),
            }
            buckets[bucket_id] = candle
        else:
            price = float(mid)
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
    ordered = [buckets[idx] for idx in sorted(buckets)]
    return ordered


async def pullback_s5_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    last_spread_log = 0.0
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_monotonic = time.monotonic()
            if now_monotonic < cooldown_until:
                continue

            blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = float((spread_state or {}).get("spread_pips", 0.0) or 0.0)
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if now_monotonic - last_spread_log > 30.0:
                    LOG.info(
                        "%s spread gate active spread=%.2fp reason=%s",
                        config.LOG_PREFIX,
                        spread_pips,
                        spread_reason or "guard",
                    )
                    last_spread_log = now_monotonic
                continue

            ticks = tick_window.recent_ticks(seconds=config.WINDOW_SEC, limit=3600)
            if len(ticks) < config.MIN_BUCKETS:
                continue

            candles = _bucket_ticks(ticks)
            if len(candles) < config.MIN_BUCKETS:
                continue

            closes = [c["close"] for c in candles]
            fast_series = closes[-config.FAST_BUCKETS :]
            slow_series = closes[-config.SLOW_BUCKETS :]
            z_fast = _z_score(fast_series)
            z_slow = _z_score(slow_series)
            if z_fast is None or z_slow is None:
                continue
            atr_fast = _atr_from_closes(fast_series, config.RSI_PERIOD)
            if atr_fast < config.MIN_ATR_PIPS:
                continue
            rsi_fast = _rsi(fast_series, config.RSI_PERIOD)

            side: Optional[str] = None
            if config.FAST_Z_MIN <= z_fast <= config.FAST_Z_MAX and z_slow <= config.SLOW_Z_SHORT_MAX:
                if rsi_fast is None or config.RSI_SHORT_RANGE[0] <= rsi_fast <= config.RSI_SHORT_RANGE[1]:
                    side = "short"
            elif -config.FAST_Z_MAX <= z_fast <= -config.FAST_Z_MIN and z_slow >= config.SLOW_Z_LONG_MIN:
                if rsi_fast is None or config.RSI_LONG_RANGE[0] <= rsi_fast <= config.RSI_LONG_RANGE[1]:
                    side = "long"
            if side is None:
                continue

            pockets = pos_manager.get_open_positions()
            scalp_pos = pockets.get("scalp") or {}
            tagged = []
            for trade in scalp_pos.get("open_trades") or []:
                thesis = trade.get("entry_thesis") or {}
                if thesis.get("strategy_tag") == "pullback_s5":
                    tagged.append(trade)
            if tagged:
                last_price = float(tagged[-1].get("price") or 0.0)
                if last_price:
                    delta = abs(last_price - closes[-1]) / config.PIP_VALUE
                    if delta < config.STAGE_MIN_DELTA_PIPS:
                        continue
            if len(tagged) >= config.MAX_ACTIVE_TRADES:
                continue

            latest_tick = ticks[-1]
            try:
                bid = float(latest_tick.get("bid") or closes[-1])
                ask = float(latest_tick.get("ask") or closes[-1])
            except (TypeError, ValueError):
                bid = closes[-1]
                ask = closes[-1]

            entry_price = ask if side == "long" else bid
            tp_price = round(
                entry_price + config.TP_PIPS * config.PIP_VALUE
                if side == "long"
                else entry_price - config.TP_PIPS * config.PIP_VALUE,
                3,
            )
            sl_base = max(config.MIN_SL_PIPS, atr_fast * config.SL_ATR_MULT)
            sl_price = round(
                entry_price - sl_base * config.PIP_VALUE
                if side == "long"
                else entry_price + sl_base * config.PIP_VALUE,
                3,
            )

            units = config.ENTRY_UNITS if side == "long" else -config.ENTRY_UNITS
            thesis = {
                "strategy_tag": "pullback_s5",
                "z_fast": round(z_fast, 2),
                "z_slow": round(z_slow, 2),
                "rsi_fast": None if rsi_fast is None else round(rsi_fast, 1),
                "atr_fast_pips": round(atr_fast, 2),
                "spread_pips": round(spread_pips, 2),
            }

            try:
                trade_id, _ = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=_client_id(side),
                    entry_thesis=thesis,
                )
            except Exception as exc:  # pragma: no cover - network path
                LOG.error("%s order error side=%s exc=%s", config.LOG_PREFIX, side, exc)
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue

            if trade_id:
                LOG.info(
                    "%s entry trade=%s side=%s units=%s tp=%.3f sl=%.3f z_fast=%.2f z_slow=%.2f",
                    config.LOG_PREFIX,
                    trade_id,
                    side,
                    units,
                    tp_price,
                    sl_price,
                    z_fast,
                    z_slow,
                )
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
            else:
                cooldown_until = now_monotonic + 10.0
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:  # pragma: no cover - defensive
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)
