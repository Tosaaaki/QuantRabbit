"\"\"Mirror-spike style strategy operating on S5 aggregated data.\"\"\""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import deque
from typing import Dict, List, Optional

from execution.order_manager import market_order
from workers.common.dyn_size import compute_units
from utils.oanda_account import get_account_snapshot
from execution.position_manager import PositionManager
from execution.risk_guard import loss_cooldown_status
from market_data import spread_monitor, tick_window
from workers.common import env_guard
from workers.common.quality_gate import current_regime, news_block_active

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
    news_block_logged = False
    regime_block_logged: Optional[str] = None
    loss_block_logged = False
    last_hour_log = 0.0
    env_block_logged = False
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now = time.monotonic()
            if now < cooldown_until or now < post_exit_until:
                continue

            if config.ACTIVE_HOURS_UTC:
                current_hour = time.gmtime().tm_hour
                if current_hour not in config.ACTIVE_HOURS_UTC:
                    if now - last_hour_log > 300.0:
                        LOG.info(
                            "%s outside active hours hour=%02d",
                            config.LOG_PREFIX,
                            current_hour,
                        )
                        last_hour_log = now
                    continue
                last_hour_log = now

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
                    LOG.info(
                        "%s blocked by regime=%s",
                        config.LOG_PREFIX,
                        regime_label,
                    )
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

            allowed_env, env_reason = env_guard.mean_reversion_allowed(
                spread_p50_limit=config.SPREAD_P50_LIMIT,
                return_pips_limit=config.RETURN_PIPS_LIMIT,
                return_window_sec=config.RETURN_WINDOW_SEC,
                instant_move_limit=config.INSTANT_MOVE_PIPS_LIMIT,
                tick_gap_ms_limit=config.TICK_GAP_MS_LIMIT,
                tick_gap_move_pips=config.TICK_GAP_MOVE_PIPS,
                ticks=ticks,
            )
            if not allowed_env:
                if not env_block_logged:
                    LOG.info("%s env guard blocked (%s)", config.LOG_PREFIX, env_reason)
                    env_block_logged = True
                continue
            env_block_logged = False

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

            prev_close = closes[-2] if len(closes) >= 2 else closes[-1]
            body_pips = abs(latest["close"] - prev_close) / config.PIP_VALUE
            if body_pips < 0.1:
                body_pips = 0.1
            if direction == "short":
                wick_pips = (peak["high"] - latest["close"]) / config.PIP_VALUE
            else:
                wick_pips = (latest["close"] - trough["low"]) / config.PIP_VALUE
            if wick_pips <= 0.0 or (wick_pips / body_pips) < config.MIN_WICK_RATIO:
                continue

            atr = _atr_from_closes(closes[-config.LOOKBACK_BUCKETS :], config.RSI_PERIOD)
            if atr < config.MIN_ATR_PIPS:
                continue
            if spread_pips > max(0.16, atr * 0.35):
                continue
            fast_z = _z_score(closes, config.PEAK_WINDOW_BUCKETS + 6)
            slow_z = _z_score(closes, config.LOOKBACK_BUCKETS)
            if fast_z is None or slow_z is None:
                continue
            rsi = _rsi(closes[-config.PEAK_WINDOW_BUCKETS :], config.RSI_PERIOD)
            # ショート側はより強いオーバーボートを要求
            if direction == "short" and rsi is not None and rsi < max(config.SELL_RSI_OVERBOUGHT, config.RSI_OVERBOUGHT):
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
            tp_pips = config.TP_PIPS
            sl_pips = config.SL_PIPS
            if atr > 0.0:
                tp_pips = min(
                    config.TP_MAX_PIPS,
                    max(config.TP_MIN_PIPS, atr * config.TP_ATR_MULT),
                )
                base_sl = max(config.SL_PIPS, atr * config.SL_ATR_MULT)
                if direction == "short":
                    # ショートはSLを少しタイトに、かつ上限を設ける
                    base_sl = min(base_sl * config.SELL_SL_ATR_BIAS, config.SELL_SL_MAX_PIPS)
                sl_pips = base_sl
            if direction == "long":
                entry_price = float(ticks[-1].get("ask") or entry_price)
                tp_price = round(entry_price + tp_pips * config.PIP_VALUE, 3)
                sl_price = round(entry_price - sl_pips * config.PIP_VALUE, 3)
            else:
                entry_price = float(ticks[-1].get("bid") or entry_price)
                tp_price = round(entry_price - tp_pips * config.PIP_VALUE, 3)
                sl_price = round(entry_price + sl_pips * config.PIP_VALUE, 3)

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
                "wick_to_body": round(wick_pips / body_pips, 2),
                "tp_pips": round(tp_pips, 2),
                "sl_pips": round(sl_pips, 2),
            }

            # 柔軟サイズ決定（スプレッド/ATR/余力/シグナル強度）
            sig_strength = None
            try:
                sig_strength = min(1.0, max(0.0, abs(float(fast_z)) / 2.0))
            except Exception:
                sig_strength = None
            sizing = compute_units(
                entry_price=float(entry_price),
                sl_pips=float(sl_pips),
                base_entry_units=int(config.ENTRY_UNITS),
                min_units=int(config.MIN_UNITS),
                max_margin_usage=float(config.MAX_MARGIN_USAGE),
                spread_pips=float(spread_pips),
                spread_soft_cap=float(config.MAX_SPREAD_PIPS),
                adx=None,
                signal_score=sig_strength,
            )
            if sizing.units <= 0:
                continue
            units = sizing.units if direction == "long" else -sizing.units
            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=_client_id(direction),
                    entry_thesis=entry_thesis,
                    meta={"entry_price": float(entry_price)},
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
                    entry_price,
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
