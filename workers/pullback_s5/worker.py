"""Pullback strategy operating on 5-second synthetic candles."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

from execution.order_manager import market_order
from workers.common.dyn_size import compute_units
from execution.position_manager import PositionManager
from market_data import spread_monitor, tick_window
from indicators.factor_cache import all_factors
from execution.risk_guard import loss_cooldown_status
from workers.common import env_guard
from workers.common.quality_gate import current_regime, news_block_active

from . import config

LOG = logging.getLogger(__name__)


def _stage_ratio(idx: int) -> float:
    ratios = config.ENTRY_STAGE_RATIOS
    if idx < len(ratios):
        return ratios[idx]
    return ratios[-1]


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


def _atr_from_candles(candles: Sequence[Dict[str, float]], period: int) -> float:
    """Compute a responsive ATR (in pips) from bucketed candles."""
    if len(candles) <= 1:
        return 0.0

    true_ranges: List[float] = []
    prev_close = float(candles[0]["close"])
    for candle in candles[1:]:
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
        prev_close = close

    if not true_ranges:
        return 0.0

    period = min(period, len(true_ranges))
    if period <= 0:
        return 0.0

    recent = true_ranges[-period:]
    return (sum(recent) / len(recent)) / config.PIP_VALUE


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
    news_block_logged = False
    regime_block_logged: Optional[str] = None
    loss_block_logged = False
    last_density_log = 0.0
    env_block_logged = False
    blocked_weekdays = {
        int(day)
        for day in config.BLOCKED_WEEKDAYS
        if day.strip().isdigit() and 0 <= int(day) <= 6
    }
    kill_switch_triggered = False
    kill_switch_reason = ""
    last_perf_sync = 0.0
    last_kill_log = 0.0
    managed_day: Optional[datetime.date] = None
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_monotonic = time.monotonic()
            if now_monotonic < cooldown_until:
                continue

            now_utc = datetime.datetime.utcnow()
            current_day = now_utc.date()
            if managed_day != current_day:
                kill_switch_triggered = False
                kill_switch_reason = ""
                managed_day = current_day

            if config.ALLOWED_HOURS_UTC and now_utc.hour not in config.ALLOWED_HOURS_UTC:
                continue
            if blocked_weekdays and now_utc.weekday() in blocked_weekdays:
                continue

            if config.ACTIVE_HOURS_UTC and now_utc.hour not in config.ACTIVE_HOURS_UTC:
                continue

            if not kill_switch_triggered and now_monotonic - last_perf_sync >= config.PERFORMANCE_REFRESH_SEC:
                last_perf_sync = now_monotonic
                try:
                    pos_manager.sync_trades()
                except Exception as exc:  # pragma: no cover
                    LOG.debug("%s sync_trades error: %s", config.LOG_PREFIX, exc)
                try:
                    summary = pos_manager.get_performance_summary()
                except Exception as exc:  # pragma: no cover
                    LOG.debug("%s performance summary error: %s", config.LOG_PREFIX, exc)
                    summary = {}
                daily = summary.get("daily", {}) if isinstance(summary, dict) else {}
                daily_pips = float(daily.get("pips", 0.0) or 0.0)
                if config.DAILY_PNL_STOP_PIPS > 0.0 and daily_pips <= -config.DAILY_PNL_STOP_PIPS:
                    kill_switch_triggered = True
                    kill_switch_reason = f"daily_pnl={daily_pips:.1f}"
                elif config.MAX_CONSEC_LOSSES > 0:
                    try:
                        recent = pos_manager.fetch_recent_trades(limit=config.MAX_CONSEC_LOSSES)
                    except Exception as exc:  # pragma: no cover
                        LOG.debug("%s fetch_recent_trades error: %s", config.LOG_PREFIX, exc)
                        recent = []
                    consecutive_losses = 0
                    for row in recent:
                        try:
                            pl_val = float(row.get("pl_pips") or 0.0)
                        except (TypeError, ValueError):
                            pl_val = 0.0
                        if pl_val < 0:
                            consecutive_losses += 1
                        else:
                            break
                    if consecutive_losses >= config.MAX_CONSEC_LOSSES:
                        kill_switch_triggered = True
                        kill_switch_reason = f"consecutive_losses={consecutive_losses}"

            if kill_switch_triggered:
                if now_monotonic - last_kill_log > 60.0:
                    LOG.info(
                        "%s kill switch active (reason=%s day=%s)",
                        config.LOG_PREFIX,
                        kill_switch_reason or "unknown",
                        managed_day,
                    )
                    last_kill_log = now_monotonic
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

            if getattr(config, "MIN_DENSITY_TICKS", 0):
                density_ticks = tick_window.recent_ticks(seconds=30.0, limit=1200)
                if len(density_ticks) < config.MIN_DENSITY_TICKS:
                    if now_monotonic - last_density_log > 30.0:
                        LOG.info(
                            "%s density gate active ticks=%d",
                            config.LOG_PREFIX,
                            len(density_ticks),
                        )
                        last_density_log = now_monotonic
                    continue
                last_density_log = now_monotonic

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
            fast_candles = candles[-config.FAST_BUCKETS :]
            fast_series = closes[-config.FAST_BUCKETS :]
            slow_series = closes[-config.SLOW_BUCKETS :]
            z_fast = _z_score(fast_series)
            z_slow = _z_score(slow_series)
            if z_fast is None or z_slow is None:
                continue
            atr_fast = _atr_from_candles(fast_candles, config.RSI_PERIOD)
            if atr_fast < config.MIN_ATR_PIPS:
                continue
            if spread_pips > max(0.18, atr_fast * 0.35):
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

            if side == "short" and not config.ALLOW_SHORT:
                continue
            if side == "long" and not config.ALLOW_LONG:
                continue

            factors = all_factors()
            trend_bias: Optional[str] = None
            adx_value: Optional[float] = None
            for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                ma10 = fac.get("ma10")
                ma20 = fac.get("ma20")
                if ma10 is not None and ma20 is not None:
                    diff = float(ma10) - float(ma20)
                    diff_pips = abs(diff) / config.PIP_VALUE
                    if diff_pips >= config.TREND_ALIGN_BUFFER_PIPS:
                        trend_bias = "long" if diff > 0 else "short"
                        break
            for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                adx = fac.get("adx")
                if adx is not None:
                    adx_value = float(adx)
                    break
            if trend_bias and trend_bias != side:
                continue
            if adx_value is not None and adx_value < config.TREND_ADX_MIN:
                continue

            pockets = pos_manager.get_open_positions()
            scalp_pos = pockets.get("scalp") or {}
            tagged = []
            for trade in scalp_pos.get("open_trades") or []:
                thesis = trade.get("entry_thesis") or {}
                if thesis.get("strategy_tag") == "pullback_s5":
                    tagged.append(trade)
            if tagged and not config.ALLOW_DUPLICATE_ENTRIES:
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
            tp_pips = config.TP_PIPS
            sl_base = max(config.MIN_SL_PIPS, atr_fast * config.SL_ATR_MULT)
            sl_base = min(sl_base, config.MAX_SL_PIPS)
            if atr_fast > 0.0:
                tp_pips = min(
                    config.TP_ATR_MAX_PIPS,
                    max(config.TP_ATR_MIN_PIPS, atr_fast * config.TP_ATR_MULT),
                )
            tp_price = round(
                entry_price + tp_pips * config.PIP_VALUE
                if side == "long"
                else entry_price - tp_pips * config.PIP_VALUE,
                3,
            )
            sl_price = round(
                entry_price - sl_base * config.PIP_VALUE
                if side == "long"
                else entry_price + sl_base * config.PIP_VALUE,
                3,
            )

            stage_idx = len(tagged)
            stage_ratio = _stage_ratio(stage_idx)
            base_units = int(round(config.ENTRY_UNITS * stage_ratio))
            # シグナル強度を 0..1 に正規化
            sig_strength = None
            try:
                span = max(1e-6, config.FAST_Z_MAX - config.FAST_Z_MIN)
                if side == "short":
                    sig_strength = (z_fast - config.FAST_Z_MIN) / span
                else:
                    sig_strength = ((-z_fast) - config.FAST_Z_MIN) / span
                sig_strength = max(0.0, min(1.0, sig_strength))
            except Exception:
                sig_strength = None

            # 柔軟なユニット計算（リスク/スプレッド/ADX/余力）
            sizing = compute_units(
                entry_price=entry_price,
                sl_pips=float(sl_base),
                base_entry_units=base_units,
                min_units=int(config.MIN_UNITS),
                max_margin_usage=float(config.MAX_MARGIN_USAGE),
                spread_pips=float(spread_pips),
                spread_soft_cap=float(config.MAX_SPREAD_PIPS),
                adx=adx_value,
                signal_score=sig_strength,
            )
            if sizing.units <= 0:
                continue
            units = sizing.units if side == "long" else -sizing.units
            thesis = {
                "strategy_tag": "pullback_s5",
                "z_fast": round(z_fast, 2),
                "z_slow": round(z_slow, 2),
                "rsi_fast": None if rsi_fast is None else round(rsi_fast, 1),
                "atr_fast_pips": round(atr_fast, 2),
                "spread_pips": round(spread_pips, 2),
                "trend_bias": trend_bias,
                "trend_adx": None if adx_value is None else round(adx_value, 1),
                "tp_pips": round(tp_pips, 2),
                "sl_pips": round(sl_base, 2),
                "stage_index": stage_idx + 1,
                "stage_ratio": round(stage_ratio, 3),
            }

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=_client_id(side),
                    entry_thesis=thesis,
                    meta={"entry_price": entry_price},
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
