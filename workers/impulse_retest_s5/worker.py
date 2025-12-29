"""Impulse-retest continuation worker on S5 candles."""

from __future__ import annotations

import datetime
import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Sequence

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import loss_cooldown_status
from market_data import spread_monitor, tick_window
from workers.common.quality_gate import current_regime, news_block_active
from utils.market_hours import is_market_open

from . import config

LOG = logging.getLogger(__name__)


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-imp-retest-{ts_ms}-{side[0]}{digest}"


def _bucket_ticks(ticks: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    span = config.BUCKET_SECONDS
    buckets: Dict[int, Dict[str, float]] = {}
    for tick in ticks:
        mid = tick.get("mid")
        epoch = tick.get("epoch")
        if mid is None or epoch is None:
            continue
        bucket_id = int(float(epoch) // span)
        price = float(mid)
        bucket = buckets.get(bucket_id)
        if bucket is None:
            buckets[bucket_id] = {
                "high": price,
                "low": price,
                "close": price,
                "epoch": float(epoch),
            }
        else:
            bucket["high"] = max(bucket["high"], price)
            bucket["low"] = min(bucket["low"], price)
            bucket["close"] = price
            bucket["epoch"] = float(epoch)
    return [buckets[idx] for idx in sorted(buckets)]


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
    period = max(1, min(period, len(gains)))
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
    trs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    if not trs:
        return 0.0
    period = max(1, min(period, len(trs)))
    return sum(trs[-period:]) / period / config.PIP_VALUE


async def impulse_retest_s5_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    post_exit_until = 0.0
    last_spread_log = 0.0
    last_hour_log = 0.0
    last_market_log = 0.0
    news_block_logged = False
    regime_block_logged: Optional[str] = None
    loss_block_logged = False
    candidate: Optional[Dict[str, float]] = None
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_monotonic = time.monotonic()
            if now_monotonic < cooldown_until or now_monotonic < post_exit_until:
                continue

            if config.ACTIVE_HOURS_UTC:
                current_hour = time.gmtime().tm_hour
                if current_hour not in config.ACTIVE_HOURS_UTC:
                    if now_monotonic - last_hour_log > 300.0:
                        LOG.info("%s outside active hours hour=%02d", config.LOG_PREFIX, current_hour)
                        last_hour_log = now_monotonic
                    continue
                last_hour_log = now_monotonic

            if not is_market_open(datetime.datetime.utcnow()):
                if now_monotonic - last_market_log > 300.0:
                    LOG.info("%s market closed (weekend window). Skipping iteration.", config.LOG_PREFIX)
                    last_market_log = now_monotonic
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

            if config.NEWS_BLOCK_MINUTES > 0 and news_block_active(
                config.NEWS_BLOCK_MINUTES, min_impact=config.NEWS_BLOCK_MIN_IMPACT
            ):
                if not news_block_logged:
                    LOG.info(
                        "%s paused by news guard (impact≥%s / %.0f min)",
                        config.LOG_PREFIX,
                        config.NEWS_BLOCK_MIN_IMPACT,
                        config.NEWS_BLOCK_MINUTES,
                    )
                    news_block_logged = True
                continue
            news_block_logged = False

            regime_label = current_regime("M1", event_mode=False)
            if regime_label and regime_label in config.BLOCK_REGIMES:
                if regime_block_logged != regime_label:
                    LOG.info("%s blocked by regime=%s", config.LOG_PREFIX, regime_label)
                    regime_block_logged = regime_label
                continue
            regime_block_logged = None

            if config.LOSS_STREAK_MAX > 0 and config.LOSS_STREAK_COOLDOWN_MIN > 0:
                loss_block, remain_sec = loss_cooldown_status(
                    "scalp",
                    max_losses=config.LOSS_STREAK_MAX,
                    cooldown_minutes=config.LOSS_STREAK_COOLDOWN_MIN,
                )
                if loss_block:
                    if not loss_block_logged:
                        LOG.warning(
                            "%s cooling down after %d consecutive losses (%.0fs remain)",
                            config.LOG_PREFIX,
                            config.LOSS_STREAK_MAX,
                            remain_sec,
                        )
                        loss_block_logged = True
                    continue
            loss_block_logged = False

            ticks = tick_window.recent_ticks(seconds=config.WINDOW_SEC, limit=3600)
            if len(ticks) < config.MIN_BUCKETS:
                continue
            candles = _bucket_ticks(ticks)
            if len(candles) < config.MIN_BUCKETS:
                continue

            closes = [c["close"] for c in candles]
            atr = _atr_from_closes(closes, max(4, config.IMPULSE_LOOKBACK // 4))
            if atr < config.MIN_ATR_PIPS:
                continue
            rsi = _rsi(closes[-config.IMPULSE_LOOKBACK :], config.RSI_PERIOD)
            if rsi is None:
                continue

            latest = candles[-1]
            lookback_slice = candles[-config.IMPULSE_LOOKBACK :]
            highs = [c["high"] for c in lookback_slice]
            lows = [c["low"] for c in lookback_slice]
            high_val = max(highs)
            low_val = min(lows)
            impulse_pips = (high_val - low_val) / config.PIP_VALUE

            now_epoch = latest.get("epoch")
            if now_epoch is None and ticks:
                try:
                    now_epoch = float(ticks[-1].get("epoch"))
                except (TypeError, ValueError, AttributeError):
                    now_epoch = None
            if now_epoch is None:
                continue
            now_epoch = float(now_epoch)

            if candidate and now_epoch > candidate.get("expire", 0):
                candidate = None

            if candidate is None:
                if impulse_pips >= config.IMPULSE_MIN_PIPS:
                    if latest["close"] >= high_val - 0.2 * config.PIP_VALUE:
                        candidate = {
                            "dir": "long",
                            "start": low_val,
                            "end": high_val,
                            "expire": now_epoch + config.RETEST_MAX_SEC,
                        }
                    elif latest["close"] <= low_val + 0.2 * config.PIP_VALUE:
                        candidate = {
                            "dir": "short",
                            "start": high_val,
                            "end": low_val,
                            "expire": now_epoch + config.RETEST_MAX_SEC,
                        }
                continue

            direction = candidate.get("dir")
            fib_low = candidate["start"] + (candidate["end"] - candidate["start"]) * config.FIB_LOWER
            fib_high = candidate["start"] + (candidate["end"] - candidate["start"]) * config.FIB_UPPER
            in_zone = False
            if direction == "long" and fib_low <= latest["close"] <= fib_high and rsi <= config.RSI_LONG_MAX:
                in_zone = True
            elif direction == "short" and fib_low >= latest["close"] >= fib_high and rsi >= config.RSI_SHORT_MIN:
                in_zone = True

            if not in_zone:
                continue

            position_info = pos_manager.get_open_positions()
            scalp_pos = position_info.get("scalp") or {}
            tagged = [
                tr
                for tr in (scalp_pos.get("open_trades") or [])
                if (tr.get("entry_thesis") or {}).get("strategy_tag") == "impulse_retest_s5"
            ]
            if tagged:
                last_price = float(tagged[-1].get("price") or 0.0)
                delta = abs(last_price - latest["close"]) / config.PIP_VALUE
                if delta < config.STAGE_MIN_DELTA_PIPS:
                    continue
            if len(tagged) >= config.MAX_ACTIVE_TRADES:
                continue

            latest_tick = ticks[-1]
            try:
                last_bid = float(latest_tick.get("bid") or latest["close"])
                last_ask = float(latest_tick.get("ask") or latest["close"])
                last_mid = float(latest_tick.get("mid") or latest["close"])
            except (TypeError, ValueError):
                last_bid = last_ask = last_mid = latest["close"]

            zone_mid = (fib_low + fib_high) / 2.0
            if direction == "long" and last_mid < zone_mid:
                continue
            if direction == "short" and last_mid > zone_mid:
                continue

            prev_close = candles[-2]["close"] if len(candles) >= 2 else latest["close"]
            if direction == "long" and latest["close"] < prev_close:
                continue
            if direction == "short" and latest["close"] > prev_close:
                continue

            if direction == "long":
                entry_price = last_ask
                tp_price = entry_price + (candidate["end"] - candidate["start"]) * config.TP_RATIO
                sl_price = fib_low - config.SL_BUFFER_PIPS * config.PIP_VALUE
                units = config.ENTRY_UNITS
            else:
                entry_price = last_bid
                tp_price = entry_price - (candidate["start"] - candidate["end"]) * config.TP_RATIO
                sl_price = fib_low + config.SL_BUFFER_PIPS * config.PIP_VALUE
                units = -config.ENTRY_UNITS

            thesis = {
                "strategy_tag": "impulse_retest_s5",
                "impulse_pips": round(impulse_pips, 3),
                "fib_low": round(fib_low, 5),
                "fib_high": round(fib_high, 5),
                "atr_pips": round(atr, 3),
                "rsi": round(rsi, 2),
                "spread_pips": round(spread_pips, 3),
            }

            client_id = _client_id(direction)
            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=round(sl_price, 3),
                    tp_price=round(tp_price, 3),
                    pocket="scalp",
                    client_order_id=client_id,
                    strategy_tag="impulse_retest_s5",
                    entry_thesis=thesis,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s order error side=%s exc=%s", config.LOG_PREFIX, direction, exc)
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                candidate = None
                continue

            candidate = None
            if trade_id:
                LOG.info(
                    "%s entry trade=%s side=%s units=%s price=%.3f",
                    config.LOG_PREFIX,
                    trade_id,
                    direction,
                    units,
                    entry_price,
                )
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                post_exit_until = now_monotonic + config.POST_EXIT_COOLDOWN_SEC
            else:
                cooldown_until = now_monotonic + config.COOLDOWN_SEC

    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:  # noqa: BLE001
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    LOG.info("%s worker boot (loop %.2fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    asyncio.run(impulse_retest_s5_worker())
