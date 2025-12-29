"""Impulse momentum continuation worker operating on S5 tick aggregates."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from execution.order_manager import market_order, set_trade_protections
from execution.position_manager import PositionManager
from execution.risk_guard import loss_cooldown_status
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from workers.common.quality_gate import current_regime, news_block_active

from . import config

LOG = logging.getLogger(__name__)


@dataclass
class ManagedTradeState:
    be_applied: bool = False
    last_trail_sl: Optional[float] = None
    last_update: float = field(default_factory=time.monotonic)


def _client_id(direction: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{direction}".encode("utf-8")).hexdigest()[:6]
    return f"qr-imp-momo-{ts_ms}-{direction[0]}{digest}"


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
        candle = buckets.get(bucket_id)
        if candle is None:
            buckets[bucket_id] = {
                "start": bucket_id * span,
                "end": (bucket_id + 1) * span,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
    return [buckets[idx] for idx in sorted(buckets)]


def _z_score(values: Sequence[float], window: int) -> Optional[float]:
    if len(values) < window or window <= 0:
        return None
    sample = values[-window:]
    mean_val = sum(sample) / len(sample)
    var = sum((v - mean_val) ** 2 for v in sample) / max(len(sample) - 1, 1)
    if var <= 0.0:
        return 0.0
    return (sample[-1] - mean_val) / (var ** 0.5)


def _atr_from_closes(values: Sequence[float], period: int) -> float:
    if len(values) <= 1 or period <= 0:
        return 0.0
    true_ranges = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    if not true_ranges:
        return 0.0
    period = max(1, min(period, len(true_ranges)))
    return sum(true_ranges[-period:]) / period / config.PIP_VALUE


def _extract_protection(trade: dict, node: str) -> Optional[float]:
    data = trade.get(node) or {}
    price = data.get("price")
    try:
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None


async def impulse_momentum_s5_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    post_exit_until = 0.0
    spike_block_until = 0.0
    last_spread_log = 0.0
    news_block_logged = False
    regime_block_logged: Optional[str] = None
    loss_block_logged = False
    last_manage = 0.0
    managed_state: Dict[str, ManagedTradeState] = {}
    blocked_weekdays = {
        int(day)
        for day in config.BLOCKED_WEEKDAYS
        if day.strip().isdigit() and 0 <= int(day) <= 6
    }
    kill_switch_triggered = False
    kill_switch_reason = ""
    last_perf_sync = 0.0
    perf_cached_day: Optional[datetime.date] = None
    last_kill_log = 0.0
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_monotonic = time.monotonic()
            if now_monotonic < cooldown_until or now_monotonic < post_exit_until:
                continue
            if now_monotonic < spike_block_until:
                continue

            now_utc = datetime.datetime.utcnow()
            if config.ALLOWED_HOURS_UTC and now_utc.hour not in config.ALLOWED_HOURS_UTC:
                continue
            if blocked_weekdays and now_utc.weekday() in blocked_weekdays:
                continue

            current_day = now_utc.date()
            if perf_cached_day != current_day:
                kill_switch_triggered = False
                kill_switch_reason = ""
                perf_cached_day = current_day

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
                            pnl = float(row.get("pl_pips") or 0.0)
                        except (TypeError, ValueError):
                            pnl = 0.0
                        if pnl < 0:
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
                        perf_cached_day,
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
            if len(ticks) < config.MIN_BUCKETS:
                continue

            if config.INSTANT_MOVE_PIPS_LIMIT > 0.0 and len(ticks) >= 2:
                latest_tick = ticks[-1]
                prior_tick = ticks[-2]
                try:
                    latest_mid = float(latest_tick.get("mid") or latest_tick.get("ask") or latest_tick.get("bid"))
                    prior_mid = float(prior_tick.get("mid") or prior_tick.get("ask") or prior_tick.get("bid") or latest_mid)
                except (TypeError, ValueError):
                    latest_mid = prior_mid = 0.0
                move_pips = abs(latest_mid - prior_mid) / config.PIP_VALUE
                if move_pips >= config.INSTANT_MOVE_PIPS_LIMIT:
                    spike_block_until = now_monotonic + config.BACKOFF_AFTER_SPIKE_SEC
                    LOG.info(
                        "%s spike guard triggered move=%.2fp block=%.0fs",
                        config.LOG_PREFIX,
                        move_pips,
                        config.BACKOFF_AFTER_SPIKE_SEC,
                    )
                    continue

            candles = _bucket_ticks(ticks)
            if len(candles) < config.MIN_BUCKETS:
                continue

            closes = [c["close"] for c in candles]
            fast_z = _z_score(closes, config.FAST_BUCKETS)
            slow_z = _z_score(closes, config.SLOW_BUCKETS)
            atr_fast = _atr_from_closes(closes, config.ATR_FAST_PERIOD)
            atr_slow = _atr_from_closes(closes, config.ATR_SLOW_PERIOD)
            atr = max(atr_fast, atr_slow)
            if atr < config.MIN_ATR_PIPS:
                continue

            latest = candles[-1]
            prev_close = candles[-2]["close"] if len(candles) >= 2 else latest["close"]
            recent_window = candles[-(config.FAST_BUCKETS + 6) : -1]
            if not recent_window:
                continue
            recent_high = max(c["high"] for c in recent_window)
            recent_low = min(c["low"] for c in recent_window)

            direction: Optional[str] = None
            breakout_gap = config.MIN_BREAKOUT_PIPS * config.PIP_VALUE
            retrace = config.RETRACE_CONFIRMED_PIPS * config.PIP_VALUE
            latest_close = latest["close"]

            if (
                latest_close >= recent_high + breakout_gap
                and fast_z is not None
                and slow_z is not None
                and fast_z >= config.FAST_Z_LONG_MIN
                and slow_z >= config.SLOW_Z_LONG_MIN
                and latest_close - prev_close >= retrace
            ):
                direction = "long"
            elif (
                latest_close <= recent_low - breakout_gap
                and fast_z is not None
                and slow_z is not None
                and fast_z <= config.FAST_Z_SHORT_MAX
                and slow_z <= config.SLOW_Z_SHORT_MAX
                and prev_close - latest_close >= retrace
            ):
                direction = "short"

            if direction is None:
                continue

            position_info = pos_manager.get_open_positions()
            scalp_pos = position_info.get("scalp") or {}
            open_trades = scalp_pos.get("open_trades") or []
            momentum_trades = [
                tr
                for tr in open_trades
                if (tr.get("entry_thesis") or {}).get("strategy_tag") == "impulse_momentum_s5"
            ]
            active_ids = {
                str(tr.get("trade_id"))
                for tr in momentum_trades
                if tr.get("trade_id")
            }
            previous_state = managed_state
            managed_state = {
                tid: state
                for tid, state in previous_state.items()
                if tid in active_ids
            }

            if momentum_trades and now_monotonic - last_manage >= config.MANAGED_POLL_SEC:
                try:
                    manage_price = float(momentum_trades[-1].get("price") or latest_close)
                except (TypeError, ValueError):
                    manage_price = latest_close
                await _manage_open_trades(
                    momentum_trades,
                    manage_price,
                    managed_state,
                    now_monotonic,
                )
                last_manage = now_monotonic
            if len(momentum_trades) >= config.MAX_ACTIVE_TRADES:
                continue
            if momentum_trades:
                existing_side = float(momentum_trades[-1].get("units") or 0.0)
                if existing_side > 0 and direction != "long":
                    continue
                if existing_side < 0 and direction != "short":
                    continue

            factors = all_factors()
            trend_bias: Optional[str] = None
            trend_adx: Optional[float] = None
            ma_slope_pips: Optional[float] = None

            for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                ma10 = fac.get("ma10")
                ma20 = fac.get("ma20")
                if ma10 is None or ma20 is None:
                    continue
                diff = float(ma10) - float(ma20)
                diff_pips = diff / config.PIP_VALUE
                if abs(diff_pips) >= config.TREND_ALIGN_BUFFER_PIPS:
                    trend_bias = "long" if diff_pips > 0 else "short"
                    if trend_bias == "long":
                        ma_slope_pips = diff_pips
                    else:
                        ma_slope_pips = -diff_pips
                    break

            for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                adx = fac.get("adx")
                if adx is not None:
                    trend_adx = float(adx)
                    break

            trend_slope_ok = False
            if ma_slope_pips is not None:
                if direction == "long" and ma_slope_pips >= config.TREND_SLOPE_MIN_PIPS:
                    trend_slope_ok = True
                if direction == "short" and ma_slope_pips >= config.TREND_SLOPE_MIN_PIPS:
                    trend_slope_ok = True
            if not trend_slope_ok and ma_slope_pips is None:
                # fallback: use recent price slope
                window_for_slope = closes[-config.FAST_BUCKETS :]
                if len(window_for_slope) >= 2:
                    slope = (window_for_slope[-1] - window_for_slope[0]) / config.PIP_VALUE
                    if direction == "long" and slope >= config.TREND_SLOPE_MIN_PIPS:
                        trend_slope_ok = True
                    if direction == "short" and slope <= -config.TREND_SLOPE_MIN_PIPS:
                        trend_slope_ok = True

            if trend_bias and trend_bias != direction:
                continue
            if not trend_slope_ok:
                continue
            if trend_adx is not None and trend_adx < config.TREND_ADX_MIN:
                continue

            latest_tick = ticks[-1]
            try:
                last_bid = float(latest_tick.get("bid") or latest_close)
                last_ask = float(latest_tick.get("ask") or latest_close)
            except (TypeError, ValueError):
                last_bid = last_ask = latest_close
            entry_price = last_ask if direction == "long" else last_bid

            tp_pips = min(
                config.TP_ATR_MAX_PIPS,
                max(config.TP_ATR_MIN_PIPS, atr * config.TP_ATR_MULT),
            )
            sl_pips = max(config.SL_ATR_MIN_PIPS, atr * config.SL_ATR_MULT)

            size_scale = 1.0
            if trend_bias == direction and (trend_adx is None or trend_adx >= config.TREND_ADX_MIN):
                size_scale = config.TREND_SIZE_MULT
            staged_units = int(round(config.ENTRY_UNITS * size_scale))
            if staged_units < 1000:
                staged_units = 1000
            units = staged_units if direction == "long" else -staged_units

            tp_price = round(
                entry_price + tp_pips * config.PIP_VALUE if direction == "long" else entry_price - tp_pips * config.PIP_VALUE,
                3,
            )
            sl_price = round(
                entry_price - sl_pips * config.PIP_VALUE if direction == "long" else entry_price + sl_pips * config.PIP_VALUE,
                3,
            )

            client_id = _client_id(direction)
            thesis = {
                "strategy_tag": "impulse_momentum_s5",
                "direction": direction,
                "fast_z": None if fast_z is None else round(fast_z, 3),
                "slow_z": None if slow_z is None else round(slow_z, 3),
                "atr_pips": round(atr, 3),
                "atr_fast": round(atr_fast, 3),
                "atr_slow": round(atr_slow, 3),
                "recent_high": round(recent_high, 5),
                "recent_low": round(recent_low, 5),
                "spread_pips": round(spread_pips, 3),
                "trend_bias": trend_bias,
                "trend_adx": None if trend_adx is None else round(trend_adx, 1),
                "ma_slope_pips": None if ma_slope_pips is None else round(ma_slope_pips, 3),
                "unit_scale": round(size_scale, 2),
            }

            try:
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=client_id,
                    strategy_tag="impulse_momentum_s5",
                    entry_thesis=thesis,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s order error direction=%s exc=%s", config.LOG_PREFIX, direction, exc)
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue

            if trade_id:
                LOG.info(
                    "%s entry trade=%s dir=%s units=%s price=%.3f tp=%.3f sl=%.3f",
                    config.LOG_PREFIX,
                    trade_id,
                    direction,
                    units,
                    entry_price,
                    tp_price,
                    sl_price,
                )
                managed_state[str(trade_id)] = ManagedTradeState()
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                post_exit_until = now_monotonic + config.POST_EXIT_COOLDOWN_SEC
            else:
                cooldown_until = now_monotonic + config.COOLDOWN_SEC

            if now_monotonic < spike_block_until:
                spike_block_until = now_monotonic

            # manage trailing when there is at least one trade
            now_monotonic = time.monotonic()
            if momentum_trades and now_monotonic - last_manage >= config.MANAGED_POLL_SEC:
                # refresh latest price for management after execution
                refreshed_positions = pos_manager.get_open_positions()
                refreshed_trades = [
                    tr
                    for tr in (refreshed_positions.get("scalp") or {}).get("open_trades") or []
                    if (tr.get("entry_thesis") or {}).get("strategy_tag") == "impulse_momentum_s5"
                ]
                if refreshed_trades:
                    latest_price = latest_close
                    try:
                        latest_price = float(refreshed_trades[-1].get("price") or latest_price)
                    except (TypeError, ValueError):
                        pass
                    await _manage_open_trades(
                        refreshed_trades,
                        latest_price,
                        managed_state,
                        now_monotonic,
                    )
                    last_manage = now_monotonic

    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:  # noqa: BLE001
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)


async def _manage_open_trades(
    trades: List[dict],
    latest_price: float,
    managed: Dict[str, ManagedTradeState],
    now_monotonic: float,
) -> None:
    for tr in trades:
        trade_id = str(tr.get("trade_id"))
        if not trade_id:
            continue
        units = tr.get("units", 0)
        try:
            units = float(units)
        except (TypeError, ValueError):
            continue
        if units == 0:
            continue
        side = "long" if units > 0 else "short"
        try:
            entry_price = float(tr.get("price"))
        except (TypeError, ValueError):
            continue

        pip_value = config.PIP_VALUE
        profit_pips = (
            (latest_price - entry_price) / pip_value if side == "long" else (entry_price - latest_price) / pip_value
        )

        state = managed.setdefault(trade_id, ManagedTradeState())
        tp_price = _extract_protection(tr, "take_profit")
        current_sl = _extract_protection(tr, "stop_loss")

        if not state.be_applied and profit_pips >= config.BE_TRIGGER_PIPS:
            if side == "long":
                desired_sl = entry_price + config.BE_OFFSET_PIPS * pip_value
                should_update = current_sl is None or desired_sl > current_sl + 1e-6
            else:
                desired_sl = entry_price - config.BE_OFFSET_PIPS * pip_value
                should_update = current_sl is None or desired_sl < current_sl - 1e-6
            if should_update:
                ok = await set_trade_protections(trade_id, sl_price=round(desired_sl, 3), tp_price=tp_price)
                if ok:
                    state.be_applied = True
                    state.last_trail_sl = desired_sl
                    state.last_update = now_monotonic
                    current_sl = desired_sl

        if profit_pips >= config.TRAIL_TRIGGER_PIPS and (now_monotonic - state.last_update) >= config.TRAIL_COOLDOWN_SEC:
            if side == "long":
                desired_sl = latest_price - config.TRAIL_BACKOFF_PIPS * pip_value
                step_ok = state.last_trail_sl is None or desired_sl - state.last_trail_sl >= config.TRAIL_STEP_PIPS * pip_value
                should_update = desired_sl > (current_sl or -float("inf")) + 1e-6 and step_ok
            else:
                desired_sl = latest_price + config.TRAIL_BACKOFF_PIPS * pip_value
                step_ok = state.last_trail_sl is None or state.last_trail_sl - desired_sl >= config.TRAIL_STEP_PIPS * pip_value
                should_update = desired_sl < (current_sl or float("inf")) - 1e-6 and step_ok
            if should_update:
                ok = await set_trade_protections(trade_id, sl_price=round(desired_sl, 3), tp_price=tp_price)
                if ok:
                    state.last_trail_sl = desired_sl
                    state.last_update = now_monotonic


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    LOG.info("%s worker boot (loop %.2fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    asyncio.run(impulse_momentum_s5_worker())
