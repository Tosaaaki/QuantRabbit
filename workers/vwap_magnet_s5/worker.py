"""VWAP magnet strategy operating on S5 aggregated data."""

from __future__ import annotations

import datetime
import asyncio
import hashlib
import logging
import math
import time
from typing import Dict, List, Optional, Sequence

from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import loss_cooldown_status
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from workers.common import env_guard
from workers.common.quality_gate import current_regime, news_block_active
from utils.market_hours import is_market_open

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
    return f"qr-vwap-s5-{ts_ms}-{side[0]}{digest}"


def _bucket_ticks_with_counts(ticks: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
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
                "count": 1.0,
            }
        else:
            bucket["high"] = max(bucket["high"], price)
            bucket["low"] = min(bucket["low"], price)
            bucket["close"] = price
            bucket["count"] += 1.0
    ordered = [buckets[idx] for idx in sorted(buckets)]
    return ordered


def _wma(values: Sequence[float], weights: Sequence[float]) -> float:
    total_w = sum(weights)
    if total_w <= 0.0:
        return values[-1]
    return sum(v * w for v, w in zip(values, weights)) / total_w


def _z_dev(closes: Sequence[float], weights: Sequence[float], win: int) -> Optional[float]:
    if len(closes) < win:
        return None
    window_closes = closes[-win:]
    window_weights = weights[-win:]
    vwap = _wma(window_closes, window_weights)
    mean_val = sum(window_closes) / win
    var = sum((x - mean_val) ** 2 for x in window_closes) / max(win - 1, 1)
    if var <= 0.0:
        return 0.0
    sigma = math.sqrt(var)
    if sigma == 0:
        return 0.0
    return (window_closes[-1] - vwap) / sigma


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


async def vwap_magnet_s5_worker() -> None:
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
            if now_monotonic < cooldown_until or now_monotonic < post_exit_until:
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

            if config.ACTIVE_HOURS_UTC:
                current_hour = now_utc.hour
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

            if not kill_switch_triggered and now_monotonic - last_perf_sync >= config.PERFORMANCE_REFRESH_SEC:
                last_perf_sync = now_monotonic
                try:
                    pos_manager.sync_trades()
                except Exception as exc:  # pragma: no cover - defensive network path
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
            if len(ticks) < config.MIN_DENSITY_TICKS:
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

            candles = _bucket_ticks_with_counts(ticks)
            if len(candles) < config.MIN_BUCKETS:
                continue

            closes = [c["close"] for c in candles]
            counts = [c["count"] for c in candles]
            weights = counts
            z_dev = _z_dev(closes, weights, config.VWAP_WINDOW_BUCKETS)
            if z_dev is None:
                continue
            atr = _atr_from_closes(closes, config.VWAP_WINDOW_BUCKETS // 2)
            if atr < config.MIN_ATR_PIPS:
                continue
            rsi = _rsi(closes[-config.VWAP_WINDOW_BUCKETS :], config.RSI_PERIOD)
            if rsi is None:
                continue

            position_info = pos_manager.get_open_positions()
            scalp_pos = position_info.get("scalp") or {}
            trades = scalp_pos.get("open_trades") or []
            current_tagged = [
                tr
                for tr in trades
                if (tr.get("entry_thesis") or {}).get("strategy_tag") == "vwap_magnet_s5"
            ]
            stage_idx = len(current_tagged)
            if stage_idx >= config.MAX_ACTIVE_TRADES or stage_idx >= len(config.ENTRY_STAGE_RATIOS):
                continue
            if current_tagged:
                last_price = float(current_tagged[-1].get("price") or 0.0)
                latest_close = closes[-1]
                delta = abs(latest_close - last_price) / config.PIP_VALUE
                if delta < config.STAGE_MIN_DELTA_PIPS:
                    continue
                if not config.ALLOW_DUPLICATE_ENTRIES and stage_idx > 0:
                    continue

            latest_close = closes[-1]
            prev_vwap = _wma(closes[-(config.VWAP_WINDOW_BUCKETS + 1):-1], counts[-(config.VWAP_WINDOW_BUCKETS + 1):-1])
            slope = latest_close - prev_vwap

            side: Optional[str] = None
            if (
                z_dev >= config.Z_ENTRY_SIGMA
                and slope <= 0.0
                and config.RSI_SHORT_RANGE[0] <= rsi <= config.RSI_SHORT_RANGE[1]
            ):
                side = "short"
            elif (
                z_dev <= -config.Z_ENTRY_SIGMA
                and slope >= 0.0
                and config.RSI_LONG_RANGE[0] <= rsi <= config.RSI_LONG_RANGE[1]
            ):
                side = "long"
            if side is None:
                continue

            if side == "long" and not config.ALLOW_LONG:
                continue
            if side == "short" and not config.ALLOW_SHORT:
                continue

            fast_len = min(len(closes), config.MA_FAST_BUCKETS)
            slow_len = min(len(closes), config.MA_SLOW_BUCKETS)
            if fast_len < config.MA_FAST_BUCKETS or slow_len < config.MA_SLOW_BUCKETS:
                continue
            fast_ma = sum(closes[-fast_len:]) / fast_len
            slow_ma = sum(closes[-slow_len:]) / slow_len
            ma_diff_pips = (fast_ma - slow_ma) / config.PIP_VALUE
            if side == "long" and ma_diff_pips < config.MA_DIFF_PIPS:
                continue
            if side == "short" and ma_diff_pips > -config.MA_DIFF_PIPS:
                continue

            trend_bias: Optional[str] = None
            trend_adx: Optional[float] = None
            try:
                factors = all_factors()
            except Exception:
                factors = {}
            for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                ma10 = fac.get("ma10")
                ma20 = fac.get("ma20")
                if ma10 is None or ma20 is None:
                    continue
                diff = float(ma10) - float(ma20)
                diff_pips = abs(diff) / config.PIP_VALUE
                if diff_pips >= config.TREND_ALIGN_BUFFER_PIPS:
                    trend_bias = "long" if diff > 0 else "short"
                    break
            for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                adx = fac.get("adx")
                if adx is not None:
                    trend_adx = float(adx)
                    break

            if trend_bias and trend_bias != side:
                continue
            if trend_adx is not None and trend_adx < config.TREND_ADX_MIN:
                continue

            if stage_idx > 0 and config.STAGE_FAVORABLE_PIPS > 0:
                last_price = float(current_tagged[-1].get("price") or latest_close)
                move_pips = (latest_close - last_price) / config.PIP_VALUE
                if side == "long":
                    if move_pips < config.STAGE_FAVORABLE_PIPS:
                        continue
                else:
                    if move_pips > -config.STAGE_FAVORABLE_PIPS:
                        continue

            latest_tick = ticks[-1]
            try:
                last_bid = float(latest_tick.get("bid") or latest_close)
                last_ask = float(latest_tick.get("ask") or latest_close)
            except (TypeError, ValueError):
                last_bid = last_ask = latest_close

            entry_price = last_ask if side == "long" else last_bid
            tp_price = (
                entry_price + config.TP_PIPS * config.PIP_VALUE
                if side == "long"
                else entry_price - config.TP_PIPS * config.PIP_VALUE
            )
            sl_pips = max(config.SL_MIN_PIPS, atr * config.SL_ATR_MULT)
            sl_pips = min(sl_pips, config.TP_PIPS * 1.1)
            sl_price = (
                entry_price - sl_pips * config.PIP_VALUE
                if side == "long"
                else entry_price + sl_pips * config.PIP_VALUE
            )
            stage_ratio = _stage_ratio(stage_idx)
            staged_units = int(round(config.ENTRY_UNITS * stage_ratio))
            if staged_units < 1000:
                staged_units = 1000
            units = staged_units if side == "long" else -staged_units

            thesis = {
                "strategy_tag": "vwap_magnet_s5",
                "z_dev": round(z_dev, 3),
                "atr_pips": round(atr, 3),
                "rsi": round(rsi, 2),
                "slope": round(slope, 5),
                "spread_pips": round(spread_pips, 3),
                "stage_index": stage_idx + 1,
                "stage_ratio": round(stage_ratio, 3),
                "trend_bias": trend_bias,
                "trend_adx": None if trend_adx is None else round(trend_adx, 1),
                "ma_diff_pips": round(ma_diff_pips, 3),
            }
            client_id = _client_id(side)

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=client_id,
                    entry_thesis=thesis,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s order error side=%s exc=%s", config.LOG_PREFIX, side, exc)
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue

            if trade_id:
                fill_price = entry_price
                LOG.info(
                    "%s entry trade=%s side=%s units=%s price=%.3f tp=%.3f sl=%.3f",
                    config.LOG_PREFIX,
                    trade_id,
                    side,
                    units,
                    fill_price,
                    tp_price,
                    sl_price,
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
