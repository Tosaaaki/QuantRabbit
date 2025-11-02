"\"\"Mirror-spike style strategy operating on S5 aggregated data.\"\"\""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import deque
from typing import Dict, List, Optional

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from market_data import spread_monitor, tick_window

from . import config

LOG = logging.getLogger(__name__)


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-mirror-s5-{ts_ms}-{side[0]}{digest}"


def _bucket_ticks(ticks: List[Dict[str, float]]) -> List[Dict[str, float]]:
    span = config.BUCKET_SECONDS
    buckets: Dict[int, Dict[str, float]] = {}
    for tick in ticks:
        mid = tick.get("mid")
        epoch = tick.get("epoch")
        if mid is None or epoch is None:
            continue
        bucket_id = int(epoch // span)
        candle = buckets.get(bucket_id)
        price = float(mid)
        if candle is None:
            candle = {
                "start": bucket_id * span,
                "high": price,
                "low": price,
                "close": price,
            }
            buckets[bucket_id] = candle
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
    ordered = [buckets[idx] for idx in sorted(buckets)]
    return ordered


def _atr_from_closes(closes: List[float], period: int) -> float:
    if len(closes) <= 1:
        return 0.0
    tr = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    period = min(period, len(tr))
    if period <= 0:
        return 0.0
    return sum(tr[-period:]) / period / config.PIP_VALUE


def _z_score(series: List[float], window: int) -> Optional[float]:
    if len(series) < window:
        return None
    sample = series[-window:]
    mean_val = sum(sample) / len(sample)
    variance = sum((v - mean_val) ** 2 for v in sample) / max(len(sample) - 1, 1)
    std = variance ** 0.5
    if std == 0.0:
        return 0.0
    return (sample[-1] - mean_val) / std


def _rsi(values: List[float], period: int) -> Optional[float]:
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


async def mirror_spike_s5_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    post_exit_until = 0.0
    last_spread_log = 0.0
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now = time.monotonic()
            if now < cooldown_until or now < post_exit_until:
                continue

            blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = float((spread_state or {}).get("spread_pips", 0.0) or 0.0)
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if now - last_spread_log > 30.0:
                    LOG.info(
                        "%s spread gate active spread=%.2fp reason=%s",
                        config.LOG_PREFIX,
                        spread_pips,
                        spread_reason or "guard",
                    )
                    last_spread_log = now
                continue

            ticks = tick_window.recent_ticks(seconds=config.WINDOW_SEC, limit=3600)
            if len(ticks) < config.MIN_BUCKETS:
                continue
            candles = _bucket_ticks(ticks)
            if len(candles) < config.MIN_BUCKETS:
                continue

            closes = [c["close"] for c in candles]
            highs = [c["high"] for c in candles]
            lows = [c["low"] for c in candles]

            lookback = candles[-config.LOOKBACK_BUCKETS :]
            peak_idx = max(range(len(lookback)), key=lambda i: lookback[i]["high"])
            trough_idx = min(range(len(lookback)), key=lambda i: lookback[i]["low"])
            peak = lookback[peak_idx]
            trough = lookback[trough_idx]
            latest = candles[-1]

            direction: Optional[str] = None
            spike_height = 0.0
            retrace = 0.0
            if peak["high"] - trough["low"] >= config.SPIKE_THRESHOLD_PIPS * config.PIP_VALUE:
                if peak_idx >= len(lookback) - config.PEAK_WINDOW_BUCKETS:
                    spike_height = (peak["high"] - trough["low"]) / config.PIP_VALUE
                    retrace = (peak["high"] - latest["close"]) / config.PIP_VALUE
                    if retrace >= config.RETRACE_TRIGGER_PIPS and retrace >= config.MIN_RETRACE_PIPS:
                        direction = "short"
            if direction is None and peak["high"] - trough["low"] >= config.SPIKE_THRESHOLD_PIPS * config.PIP_VALUE:
                if trough_idx >= len(lookback) - config.PEAK_WINDOW_BUCKETS:
                    spike_height = (peak["high"] - trough["low"]) / config.PIP_VALUE
                    retrace = (latest["close"] - trough["low"]) / config.PIP_VALUE
                    if retrace >= config.RETRACE_TRIGGER_PIPS and retrace >= config.MIN_RETRACE_PIPS:
                        direction = "long"

            if direction is None:
                continue

            atr = _atr_from_closes(closes[-config.LOOKBACK_BUCKETS :], config.RSI_PERIOD)
            if atr < config.MIN_ATR_PIPS:
                continue
            fast_z = _z_score(closes, config.PEAK_WINDOW_BUCKETS + 6)
            slow_z = _z_score(closes, config.LOOKBACK_BUCKETS)
            if fast_z is None or slow_z is None:
                continue
            rsi = _rsi(closes[-config.PEAK_WINDOW_BUCKETS :], config.RSI_PERIOD)
            if direction == "short" and rsi is not None and rsi < config.RSI_OVERBOUGHT:
                continue
            if direction == "long" and rsi is not None and rsi > config.RSI_OVERSOLD:
                continue

            pockets = pos_manager.get_open_positions()
            scalp_pos = pockets.get("scalp") or {}
            tagged = []
            for trade in scalp_pos.get("open_trades") or []:
                thesis = trade.get("entry_thesis") or {}
                if thesis.get("strategy_tag") == "mirror_spike_s5":
                    tagged.append(trade)
            if tagged:
                last_price = float(tagged[-1].get("price") or latest["close"])
                if abs(last_price - latest["close"]) / config.PIP_VALUE < config.STAGE_MIN_DELTA_PIPS:
                    continue
            if len(tagged) >= config.MAX_ACTIVE_TRADES:
                continue

            entry_price = latest["close"]
            if direction == "long":
                entry_price = float(ticks[-1].get("ask") or entry_price)
                tp_price = round(entry_price + config.TP_PIPS * config.PIP_VALUE, 3)
                sl_price = round(entry_price - config.SL_PIPS * config.PIP_VALUE, 3)
            else:
                entry_price = float(ticks[-1].get("bid") or entry_price)
                tp_price = round(entry_price - config.TP_PIPS * config.PIP_VALUE, 3)
                sl_price = round(entry_price + config.SL_PIPS * config.PIP_VALUE, 3)

            entry_thesis = {
                "strategy_tag": "mirror_spike_s5",
                "spike_height_pips": round(spike_height, 2),
                "retrace_pips": round(retrace, 2),
                "peak_high": round(peak["high"], 5),
                "trough_low": round(trough["low"], 5),
                "fast_z": 0.0 if fast_z is None else round(fast_z, 2),
                "slow_z": 0.0 if slow_z is None else round(slow_z, 2),
                "rsi": None if rsi is None else round(rsi, 1),
                "spread_pips": round(spread_pips, 2),
                "atr_pips": round(atr, 2),
            }

            units = config.ENTRY_UNITS if direction == "long" else -config.ENTRY_UNITS
            try:
                trade_id, executed_price = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=_client_id(direction),
                    entry_thesis=entry_thesis,
                )
            except Exception as exc:  # pragma: no cover - network path
                LOG.error("%s order error side=%s exc=%s", config.LOG_PREFIX, direction, exc)
                cooldown_until = now + config.COOLDOWN_SEC
                continue

            if trade_id:
                LOG.info(
                    "%s entry trade=%s side=%s units=%s exec=%.3f tp=%.3f sl=%.3f spike=%.2f retrace=%.2f",
                    config.LOG_PREFIX,
                    trade_id,
                    direction,
                    units,
                    executed_price if executed_price is not None else entry_price,
                    tp_price,
                    sl_price,
                    spike_height,
                    retrace,
                )
                cooldown_until = now + config.COOLDOWN_SEC
                post_exit_until = now + config.POST_EXIT_COOLDOWN_SEC
            else:
                cooldown_until = now + 10.0
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:  # pragma: no cover - defensive
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)
