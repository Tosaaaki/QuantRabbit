"""Impulse-break continuation worker operating on S5 synthetic candles."""

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


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-impulse-s5-{ts_ms}-{side[0]}{digest}"


def _bucket_ticks(ticks: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {}
    span = config.BUCKET_SECONDS
    for tick in ticks:
        mid = tick.get("mid")
        epoch = tick.get("epoch")
        if mid is None or epoch is None:
            continue
        bucket_id = int(float(epoch) // span)
        price = float(mid)
        candle = buckets.get(bucket_id)
        if candle is None:
            candle = {
                "start": bucket_id * span,
                "end": (bucket_id + 1) * span,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
            }
            buckets[bucket_id] = candle
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
    if len(values) <= 1:
        return 0.0
    true_ranges = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    if not true_ranges:
        return 0.0
    period = max(1, min(period, len(true_ranges)))
    return sum(true_ranges[-period:]) / period / config.PIP_VALUE


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


def _extract_protection(trade: dict, key: str) -> Optional[float]:
    node = trade.get(key) or {}
    price = node.get("price")
    try:
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None


async def impulse_break_s5_worker() -> None:
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

            now_utc = datetime.datetime.utcnow()
            current_day = now_utc.date()
            if perf_cached_day != current_day:
                kill_switch_triggered = False
                kill_switch_reason = ""
                perf_cached_day = current_day

            if config.ALLOWED_HOURS_UTC and now_utc.hour not in config.ALLOWED_HOURS_UTC:
                continue
            if blocked_weekdays and now_utc.weekday() in blocked_weekdays:
                continue

            if not kill_switch_triggered and now_monotonic - last_perf_sync >= config.PERFORMANCE_REFRESH_SEC:
                last_perf_sync = now_monotonic
                try:
                    pos_manager.sync_trades()
                except Exception as exc:  # pragma: no cover - network operations
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
                        pl_val = row.get("pl_pips")
                        try:
                            pl_pips = float(pl_val)
                        except (TypeError, ValueError):
                            pl_pips = 0.0
                        if pl_pips < 0:
                            consecutive_losses += 1
                        else:
                            break
                    if consecutive_losses >= config.MAX_CONSEC_LOSSES:
                        kill_switch_triggered = True
                        kill_switch_reason = f"consecutive_losses={consecutive_losses}"

            if kill_switch_triggered:
                if now_monotonic - last_kill_log > 60.0:
                    LOG.info(
                        "%s kill switch active (reason=%s, day=%s)",
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
            candles = _bucket_ticks(ticks)
            if len(candles) < config.MIN_BUCKETS:
                continue

            closes = [c["close"] for c in candles]
            latest = candles[-1]
            prev_close = candles[-2]["close"] if len(candles) >= 2 else latest["close"]

            fast_z = _z_score(closes, config.FAST_BUCKETS)
            slow_z = _z_score(closes, config.SLOW_BUCKETS)
            atr = _atr_from_closes(closes, config.FAST_BUCKETS // 2)
            rsi = _rsi(closes[-config.FAST_BUCKETS :], config.RSI_PERIOD)
            if atr < config.MIN_ATR_PIPS or fast_z is None or slow_z is None:
                continue

            position_info = pos_manager.get_open_positions()
            scalp_pos = position_info.get("scalp") or {}
            open_trades = scalp_pos.get("open_trades") or []
            impulse_trades = [
                tr
                for tr in open_trades
                if (tr.get("entry_thesis") or {}).get("strategy_tag") == "impulse_break_s5"
            ]
            active_ids = {str(tr.get("trade_id")) for tr in impulse_trades if tr.get("trade_id")}
            managed_state = {tid: managed_state[tid] for tid in managed_state if tid in active_ids}

            latest_close = latest["close"]
            allow_manage = now_monotonic - last_manage >= config.MANAGED_POLL_SEC
            if allow_manage and impulse_trades:
                await _manage_open_trades(
                    impulse_trades,
                    latest_close,
                    managed_state,
                    now_monotonic,
                )
                last_manage = now_monotonic

            existing_side: Optional[str] = None
            if impulse_trades:
                impulse_trades.sort(key=lambda tr: tr.get("open_time") or "", reverse=True)
                units_val = impulse_trades[0].get("units", 0)
                try:
                    units_val = float(units_val)
                except (TypeError, ValueError):
                    units_val = 0.0
                existing_side = "long" if units_val > 0 else "short"
                if len(impulse_trades) >= config.MAX_ACTIVE_TRADES:
                    continue

            recent_high = max(c["high"] for c in candles[-(config.FAST_BUCKETS + 6) : -1])
            recent_low = min(c["low"] for c in candles[-(config.FAST_BUCKETS + 6) : -1])

            direction: Optional[str] = None
            breakout_gap = config.MIN_BREAKOUT_PIPS * config.PIP_VALUE
            retrace_gap = config.MIN_RETRACE_GAP_PIPS * config.PIP_VALUE

            if (
                latest_close >= recent_high + breakout_gap
                and fast_z >= config.FAST_Z_LONG_MIN
                and slow_z >= config.SLOW_Z_LONG_MIN
                and (rsi is None or rsi <= config.RSI_LONG_MAX)
                and latest_close - prev_close >= retrace_gap
            ):
                direction = "long"
            elif (
                latest_close <= recent_low - breakout_gap
                and fast_z <= config.FAST_Z_SHORT_MAX
                and slow_z <= config.SLOW_Z_SHORT_MAX
                and (rsi is None or rsi >= config.RSI_SHORT_MIN)
                and prev_close - latest_close >= retrace_gap
            ):
                direction = "short"

            if direction is None:
                continue
            if direction == "long" and not config.ALLOW_LONG:
                continue
            if direction == "short" and not config.ALLOW_SHORT:
                continue
            if existing_side and existing_side != direction:
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

            latest_tick = ticks[-1]
            try:
                last_bid = float(latest_tick.get("bid") or latest_close)
                last_ask = float(latest_tick.get("ask") or latest_close)
            except (TypeError, ValueError):
                last_bid = last_ask = latest_close

            entry_price = last_ask if direction == "long" else last_bid

            tp_pips = max(config.TP_ATR_MIN_PIPS, atr * config.TP_ATR_MULT)
            tp_pips = min(tp_pips, config.TP_ATR_MAX_PIPS)
            sl_pips = max(config.SL_ATR_MIN_PIPS, atr * config.SL_ATR_MULT)

            size_scale = 1.0
            if trend_bias == direction and (
                trend_adx is None or trend_adx >= config.TREND_ADX_MIN
            ):
                size_scale = config.TREND_SIZE_MULT
            scaled_units = int(round(config.ENTRY_UNITS * size_scale))
            if scaled_units < 1000:
                scaled_units = 1000

            if direction == "long":
                tp_price = round(entry_price + tp_pips * config.PIP_VALUE, 3)
                sl_price = round(entry_price - sl_pips * config.PIP_VALUE, 3)
                units = scaled_units
            else:
                tp_price = round(entry_price - tp_pips * config.PIP_VALUE, 3)
                sl_price = round(entry_price + sl_pips * config.PIP_VALUE, 3)
                units = -scaled_units

            client_id = _client_id(direction)
            entry_thesis = {
                "strategy_tag": "impulse_break_s5",
                "direction": direction,
                "fast_z": round(fast_z, 3) if fast_z is not None else None,
                "slow_z": round(slow_z, 3) if slow_z is not None else None,
                "atr_pips": round(atr, 3),
                "breakout_gap_pips": round(breakout_gap / config.PIP_VALUE, 3),
                "recent_high": round(recent_high, 5),
                "recent_low": round(recent_low, 5),
                "spread_pips": round(spread_pips, 3),
                "trend_bias": trend_bias,
                "trend_adx": None if trend_adx is None else round(trend_adx, 1),
                "unit_scale": round(size_scale, 2),
            }

            try:
                trade_id, executed_price = await market_order(
                    "USD_JPY",
                    units,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    pocket="scalp",
                    client_order_id=client_id,
                    entry_thesis=entry_thesis,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error("%s order error side=%s exc=%s", config.LOG_PREFIX, direction, exc)
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue

            if trade_id:
                LOG.info(
                    "%s entry trade=%s side=%s units=%s price=%.3f tp=%.3f sl=%.3f",
                    config.LOG_PREFIX,
                    trade_id,
                    direction,
                    units,
                    executed_price if executed_price is not None else entry_price,
                    tp_price,
                    sl_price,
                )
                managed_state[str(trade_id)] = ManagedTradeState()
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
        if side == "long":
            profit_pips = (latest_price - entry_price) / pip_value
        else:
            profit_pips = (entry_price - latest_price) / pip_value

        state = managed.setdefault(trade_id, ManagedTradeState())
        tp_price = _extract_protection(tr, "take_profit")
        current_sl = _extract_protection(tr, "stop_loss")

        # Break-even move
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

        # Trailing stop management
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
