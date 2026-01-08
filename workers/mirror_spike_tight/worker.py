"""Mirror spike tight strategy operating on S5 aggregated data."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from execution.order_manager import market_order, set_trade_protections
from execution.position_manager import PositionManager
from execution.risk_guard import loss_cooldown_status
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from workers.common import env_guard
from workers.common.quality_gate import current_regime
from analysis import policy_bus

from . import config

LOG = logging.getLogger(__name__)


@dataclass
class ManagedTradeState:
    be_applied: bool = False
    last_update: float = field(default_factory=time.monotonic)


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-mirror-tight-{ts_ms}-{side[0]}{digest}"


def _bucket_ticks(ticks: List[Dict[str, float]]) -> List[Dict[str, float]]:
    span = config.BUCKET_SECONDS
    buckets: Dict[int, Dict[str, float]] = {}
    for tick in ticks:
        mid = tick.get("mid")
        epoch = tick.get("epoch")
        if mid is None or epoch is None:
            continue
        bucket_id = int(epoch // span)
        price = float(mid)
        candle = buckets.get(bucket_id)
        if candle is None:
            buckets[bucket_id] = {
                "start": bucket_id * span,
                "end": (bucket_id + 1) * span,
                "high": price,
                "low": price,
                "close": price,
            }
        else:
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
    return [buckets[idx] for idx in sorted(buckets)]


async def mirror_spike_tight_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    pos_manager = PositionManager()
    cooldown_until = 0.0
    post_exit_until = 0.0
    last_spread_log = 0.0
    last_env_log = 0.0
    regime_block_logged: Optional[str] = None
    loss_block_logged = False
    last_perf_sync = 0.0
    perf_cached_day: Optional[datetime.date] = None
    kill_switch_triggered = False
    kill_switch_reason = ""
    last_kill_log = 0.0
    managed_state: Dict[str, ManagedTradeState] = {}
    daily_trades = 0
    daily_limit_logged = False
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_monotonic = time.monotonic()
            if now_monotonic < cooldown_until or now_monotonic < post_exit_until:
                continue

            now_utc = datetime.datetime.utcnow()
            if config.ALLOWED_HOURS_UTC and now_utc.hour not in config.ALLOWED_HOURS_UTC:
                continue

            blocked_weekdays = {
                int(token)
                for token in config.BLOCKED_WEEKDAYS
                if token.strip().isdigit()
            }
            if blocked_weekdays and now_utc.weekday() in blocked_weekdays:
                continue

            current_day = now_utc.date()
            if perf_cached_day != current_day:
                kill_switch_triggered = False
                kill_switch_reason = ""
                perf_cached_day = current_day
                daily_trades = 0
                daily_limit_logged = False

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
                elif config.LOSS_STREAK_MAX > 0:
                    try:
                        recent = pos_manager.fetch_recent_trades(limit=config.LOSS_STREAK_MAX)
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
                    if consecutive_losses >= config.LOSS_STREAK_MAX:
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

            plan_snapshot = policy_bus.latest()
            plan_scalp = {}
            if plan_snapshot and getattr(plan_snapshot, "pockets", None):
                plan_scalp = plan_snapshot.pockets.get("scalp") or {}
            entry_plan = plan_scalp.get("entry_gates", {}) if isinstance(plan_scalp, dict) else {}
            allow_new_plan = bool(entry_plan.get("allow_new", True))
            policy_units_cap = plan_scalp.get("units_cap") if isinstance(plan_scalp, dict) else None
            try:
                policy_units_cap = int(policy_units_cap) if policy_units_cap is not None else None
            except (TypeError, ValueError):
                policy_units_cap = None
            current_units_plan = plan_scalp.get("current_units") if isinstance(plan_scalp, dict) else None
            try:
                current_units_plan = abs(int(current_units_plan)) if current_units_plan is not None else 0
            except (TypeError, ValueError):
                current_units_plan = 0

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

            if (
                config.MAX_TRADES_PER_DAY > 0
                and daily_trades >= config.MAX_TRADES_PER_DAY
            ):
                if not daily_limit_logged:
                    LOG.info(
                        "%s daily trade limit reached (%d)",
                        config.LOG_PREFIX,
                        config.MAX_TRADES_PER_DAY,
                    )
                    daily_limit_logged = True
                continue

            if not allow_new_plan:
                await _manage_trades(pos_manager, managed_state, time.monotonic())
                continue

            allowed_env, reason = env_guard.mean_reversion_allowed(
                spread_p50_limit=config.SPREAD_P50_LIMIT,
                return_pips_limit=config.RETURN_PIPS_LIMIT,
                return_window_sec=config.RETURN_WINDOW_SEC,
                instant_move_limit=config.INSTANT_MOVE_PIPS_LIMIT,
                tick_gap_ms_limit=config.TICK_GAP_MS_LIMIT,
                tick_gap_move_pips=config.TICK_GAP_MOVE_PIPS,
                ticks=ticks,
            )
            if not allowed_env:
                if now_monotonic - last_env_log > 60.0:
                    LOG.info("%s env guard blocked (%s)", config.LOG_PREFIX, reason)
                    last_env_log = now_monotonic
                continue
            last_env_log = now_monotonic

            candles = _bucket_ticks(ticks)
            if len(candles) < config.MIN_BUCKETS:
                continue

            lookback = candles[-config.MIRROR_LOOKBACK_BUCKETS :]
            if len(lookback) < config.MIRROR_LOOKBACK_BUCKETS:
                continue

            latest = candles[-1]
            prev_close = candles[-2]["close"] if len(candles) >= 2 else latest["close"]
            high_val = max(c["high"] for c in lookback)
            low_val = min(c["low"] for c in lookback)
            range_pips = (high_val - low_val) / config.PIP_VALUE
            if range_pips < config.SPIKE_THRESHOLD_PIPS:
                continue

            upper_trigger = high_val - config.CONFIRM_RANGE_PIPS * config.PIP_VALUE
            lower_trigger = low_val + config.CONFIRM_RANGE_PIPS * config.PIP_VALUE

            direction: Optional[str] = None
            if latest["close"] >= upper_trigger and prev_close < high_val:
                direction = "short"
            elif latest["close"] <= lower_trigger and prev_close > low_val:
                direction = "long"

            if direction is None:
                continue

            factors = all_factors()
            trend_bias: Optional[str] = None
            trend_source: Optional[str] = None
            slope_pips: Optional[float] = None

            for timeframe in ("H4", "H1", "M1"):
                fac = factors.get(timeframe) or {}
                ma10 = fac.get("ma10")
                ma20 = fac.get("ma20")
                if ma10 is None or ma20 is None:
                    continue
                diff = float(ma10) - float(ma20)
                diff_pips = diff / config.PIP_VALUE
                if abs(diff_pips) >= config.TREND_ALIGN_BUFFER_PIPS:
                    trend_bias = "long" if diff_pips > 0 else "short"
                    slope_pips = abs(diff_pips)
                    trend_source = timeframe
                    break

            if trend_bias is None or trend_bias != direction or trend_source is None:
                continue
            if trend_source == "M1":
                continue
            if slope_pips is None or slope_pips < config.TREND_SLOPE_MIN_PIPS:
                continue

            trend_adx: Optional[float] = None
            for timeframe in ("H4", "H1"):
                fac = factors.get(timeframe) or {}
                adx = fac.get("adx")
                if adx is not None:
                    trend_adx = float(adx)
                    break
            if trend_adx is not None and trend_adx < config.TREND_ADX_MIN:
                continue

            # prevent duplicate entries
            positions = pos_manager.get_open_positions()
            scalp_pos = positions.get("scalp") or {}
            open_trades = scalp_pos.get("open_trades") or []
            tight_trades = [
                tr
                for tr in open_trades
                if (tr.get("entry_thesis") or {}).get("strategy_tag") == "mirror_spike_tight"
            ]
            active_ids = {
                str(tr.get("trade_id"))
                for tr in tight_trades
                if tr.get("trade_id")
            }
            managed_state = {
                tid: state for tid, state in managed_state.items() if tid in active_ids
            }
            if tight_trades:
                continue

            latest_tick = ticks[-1]
            try:
                bid = float(latest_tick.get("bid") or latest["close"])
                ask = float(latest_tick.get("ask") or latest["close"])
            except (TypeError, ValueError):
                bid = ask = latest["close"]

            entry_price = ask if direction == "long" else bid
            tp_price = round(
                entry_price + config.TP_PIPS * config.PIP_VALUE
                if direction == "long"
                else entry_price - config.TP_PIPS * config.PIP_VALUE,
                3,
            )
            sl_price = round(
                entry_price - config.SL_PIPS * config.PIP_VALUE
                if direction == "long"
                else entry_price + config.SL_PIPS * config.PIP_VALUE,
                3,
            )

            units_abs = config.ENTRY_UNITS
            if policy_units_cap is not None:
                remaining_cap = max(0, policy_units_cap - current_units_plan)
                units_abs = min(units_abs, remaining_cap)
            if units_abs <= 0:
                await _manage_trades(pos_manager, managed_state, time.monotonic())
                continue
            units = units_abs if direction == "long" else -units_abs
            client_id = _client_id(direction)
            atr_entry = None
            try:
                fac_m1 = (factors.get("M1") or {}) if isinstance(factors, dict) else {}
                atr_entry = fac_m1.get("atr_pips")
                atr_entry = float(atr_entry) if atr_entry is not None else None
            except Exception:
                atr_entry = None
            entry_mean = 0.5 * (high_val + low_val)
            thesis = {
                "strategy_tag": "mirror_spike_tight",
                "direction": direction,
                "range_high": round(high_val, 5),
                "range_low": round(low_val, 5),
                "range_pips": round(range_pips, 3),
                "spread_pips": round(spread_pips, 3),
                "trend_bias": trend_bias,
                "trend_adx": None if trend_adx is None else round(trend_adx, 1),
                "trend_slope_pips": round(slope_pips, 3) if slope_pips is not None else None,
                "tp_pips": round(config.TP_PIPS, 2),
                "sl_pips": round(config.SL_PIPS, 2),
                "env_tf": "M5",
                "struct_tf": "M1",
                "entry_mean": round(entry_mean, 5),
                "atr_entry": None if atr_entry is None else round(atr_entry, 3),
                "range_method": "bucket_range",
                "range_lookback": int(config.MIRROR_LOOKBACK_BUCKETS),
                "range_hi_pct": 100.0,
                "range_lo_pct": 0.0,
                "tp_mode": "soft_zone",
                "tp_target": "entry_mean",
                "tp_pad_atr": 0.05,
                "range_snapshot": {
                    "high": round(high_val, 5),
                    "low": round(low_val, 5),
                    "mid": round(entry_mean, 5),
                    "method": "bucket_range",
                    "lookback": int(config.MIRROR_LOOKBACK_BUCKETS),
                    "hi_pct": 100.0,
                    "lo_pct": 0.0,
                },
                "structure_break": {"buffer_atr": 0.10, "confirm_closes": 0},
                "reversion_failure": {
                    "z_ext": 0.45,
                    "contraction_min": 0.45,
                    "bars_budget": {"k_per_z": 2.5, "min": 2, "max": 8},
                    "trend_takeover": {"require_env_trend_bars": 2},
                },
            }
            rf = thesis.get("reversion_failure")
            if isinstance(rf, dict):
                bars_budget = rf.get("bars_budget")
                if not isinstance(bars_budget, dict):
                    bars_budget = {}
                    rf["bars_budget"] = bars_budget
                if range_pips >= 10.0:
                    bars_budget["k_per_z"] = 3.0
                    bars_budget["max"] = 10
                elif range_pips <= 6.0:
                    bars_budget["k_per_z"] = 2.0
                    bars_budget["max"] = 6

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
                LOG.error("%s order error direction=%s exc=%s", config.LOG_PREFIX, direction, exc)
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue

            if trade_id:
                LOG.info(
                    "%s entry trade=%s side=%s units=%s price=%.3f tp=%.3f sl=%.3f",
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
                daily_trades += 1
                daily_limit_logged = False
            else:
                cooldown_until = now_monotonic + config.COOLDOWN_SEC

            await _manage_trades(pos_manager, managed_state, time.monotonic())

    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    finally:
        try:
            pos_manager.close()
        except Exception:  # noqa: BLE001
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)


async def _manage_trades(
    pos_manager: PositionManager,
    managed_state: Dict[str, ManagedTradeState],
    now_monotonic: float,
) -> None:
    if not managed_state:
        return
    positions = pos_manager.get_open_positions()
    scalp_pos = positions.get("scalp") or {}
    trades = [
        tr
        for tr in scalp_pos.get("open_trades") or []
        if (tr.get("entry_thesis") or {}).get("strategy_tag") == "mirror_spike_tight"
    ]
    for tr in trades:
        trade_id = str(tr.get("trade_id"))
        if not trade_id or trade_id not in managed_state:
            continue
        state = managed_state[trade_id]
        try:
            units = float(tr.get("units") or 0.0)
            entry_price = float(tr.get("price"))
            latest_price = float(tr.get("unrealized_price") or tr.get("price"))
        except (TypeError, ValueError):
            continue
        side = "long" if units > 0 else "short"
        if units == 0:
            continue
        profit_pips = (
            (latest_price - entry_price) / config.PIP_VALUE
            if side == "long"
            else (entry_price - latest_price) / config.PIP_VALUE
        )
        if not state.be_applied and profit_pips >= config.BE_TRIGGER_PIPS:
            if side == "long":
                desired_sl = entry_price + config.BE_OFFSET_PIPS * config.PIP_VALUE
            else:
                desired_sl = entry_price - config.BE_OFFSET_PIPS * config.PIP_VALUE
            ok = await set_trade_protections(
                trade_id,
                sl_price=round(desired_sl, 3),
                tp_price=(tr.get("take_profit") or {}).get("price"),
            )
            if ok:
                state.be_applied = True
                state.last_update = now_monotonic


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    LOG.info("%s worker boot (loop %.2fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    asyncio.run(mirror_spike_tight_worker())
