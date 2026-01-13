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

from analysis import plan_bus
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from workers.common import env_guard
from workers.common.pocket_plan import PocketPlan
from workers.common.quality_gate import current_regime
from workers.common.pullback_touch import count_pullback_touches
from utils.metrics_logger import log_metric

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
    cooldown_until = 0.0
    last_spread_log = 0.0
    regime_block_logged: Optional[str] = None
    loss_block_logged = False
    last_density_log = 0.0
    last_touch_block_log = 0.0
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

            # loss streak / DDは core_executor 側で統合管理

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
            # Spread gate: allow through up to the configured cap (defaults 1.2p)
            if spread_pips > config.MAX_SPREAD_PIPS:
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
            if adx_value is not None and adx_value >= config.TREND_ADX_MIN:
                if trend_bias and trend_bias != side:
                    continue
            elif adx_value is not None and adx_value < config.TREND_ADX_MIN:
                # レンジ寄りは方向バイアスを緩める
                pass

            touch_stats = None
            touch_pullback_pips = None
            touch_trend_pips = None
            touch_reset_pips = None
            touch_last_age_sec = None
            if config.TOUCH_ENABLED:
                touch_candles = candles
                if config.TOUCH_WINDOW_SEC > 0:
                    try:
                        cutoff = float(candles[-1].get("end") or 0.0) - config.TOUCH_WINDOW_SEC
                    except (TypeError, ValueError):
                        cutoff = None
                    if cutoff is not None:
                        touch_candles = [
                            c
                            for c in candles
                            if float(c.get("end") or c.get("start") or 0.0) >= cutoff
                        ]
                touch_prices = [float(c["close"]) for c in touch_candles]
                touch_times = [
                    float(c.get("end") or c.get("start") or 0.0) for c in touch_candles
                ]
                if len(touch_prices) >= 4:
                    atr_ref = atr_fast if atr_fast > 0.0 else config.TOUCH_PULLBACK_MIN_PIPS
                    touch_pullback_pips = max(
                        config.TOUCH_PULLBACK_MIN_PIPS,
                        min(config.TOUCH_PULLBACK_MAX_PIPS, atr_ref * config.TOUCH_PULLBACK_ATR_MULT),
                    )
                    touch_trend_pips = max(
                        config.TOUCH_TREND_MIN_PIPS,
                        min(config.TOUCH_TREND_MAX_PIPS, atr_ref * config.TOUCH_TREND_ATR_MULT),
                    )
                    touch_reset_pips = max(
                        config.TOUCH_PULLBACK_MIN_PIPS * 0.5,
                        touch_pullback_pips * config.TOUCH_RESET_RATIO,
                    )
                    touch_stats = count_pullback_touches(
                        touch_prices,
                        side,
                        pullback_pips=touch_pullback_pips,
                        trend_confirm_pips=touch_trend_pips,
                        reset_pips=touch_reset_pips,
                        pip_value=config.PIP_VALUE,
                        timestamps=touch_times,
                    )
                    if touch_stats.count >= config.TOUCH_HARD_COUNT:
                        if now_monotonic - last_touch_block_log > 30.0:
                            LOG.info(
                                "%s touch block side=%s count=%d pullback=%.2f trend=%.2f",
                                config.LOG_PREFIX,
                                side,
                                touch_stats.count,
                                touch_pullback_pips,
                                touch_trend_pips,
                            )
                            log_metric(
                                "pullback_touch_count",
                                float(touch_stats.count),
                                tags={
                                    "strategy": "pullback_s5",
                                    "side": side,
                                    "event": "block",
                                },
                            )
                            last_touch_block_log = now_monotonic
                        continue
                    if touch_stats.last_touch_ts is not None:
                        try:
                            touch_last_age_sec = max(
                                0.0,
                                float(touch_times[-1]) - float(touch_stats.last_touch_ts),
                            )
                        except (TypeError, ValueError):
                            touch_last_age_sec = None

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

            stage_idx = 0  # stage管理は executor に委任
            stage_ratio = _stage_ratio(stage_idx)
            base_units = int(round(config.ENTRY_UNITS * stage_ratio))
            if base_units < config.MIN_UNITS:
                base_units = config.MIN_UNITS
            confidence = 80
            if touch_stats and touch_stats.count >= config.TOUCH_SOFT_COUNT:
                orig_units = base_units
                base_units = int(round(base_units * config.TOUCH_UNIT_FACTOR))
                if base_units < config.MIN_UNITS:
                    base_units = config.MIN_UNITS
                if base_units < orig_units:
                    confidence = max(1, confidence - config.TOUCH_CONF_PENALTY)
            lot = abs(base_units) / (100000.0 * (confidence / 100.0))
            try:
                factors = all_factors()
            except Exception:
                factors = {}
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            signal = {
                "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
                "pocket": "scalp",
                "strategy": "pullback_s5",
                "tag": "pullback_s5",
                "entry_thesis": {
                    "strategy_tag": "pullback_s5",
                    "entry_guard_pullback": True,
                },
                "tp_pips": round(tp_pips, 2),
                "sl_pips": round(sl_base, 2),
                "confidence": confidence,
                "min_hold_sec": 90,
                "loss_guard_pips": None,
                "target_tp_pips": round(tp_pips, 2),
                "meta": {
                    "z_fast": round(z_fast, 2),
                    "z_slow": round(z_slow, 2),
                    "rsi_fast": None if rsi_fast is None else round(rsi_fast, 1),
                    "atr_fast_pips": round(atr_fast, 2),
                    "spread_pips": round(spread_pips, 2),
                    "trend_bias": trend_bias,
                    "trend_adx": None if adx_value is None else round(adx_value, 1),
                    "stage_index": stage_idx + 1,
                    "stage_ratio": round(stage_ratio, 3),
                    "touch_count": None if touch_stats is None else touch_stats.count,
                    "touch_pullback_pips": None
                    if touch_pullback_pips is None
                    else round(touch_pullback_pips, 2),
                    "touch_trend_pips": None
                    if touch_trend_pips is None
                    else round(touch_trend_pips, 2),
                    "touch_reset_pips": None
                    if touch_reset_pips is None
                    else round(touch_reset_pips, 2),
                    "touch_last_age_sec": None
                    if touch_last_age_sec is None
                    else round(touch_last_age_sec, 1),
                },
            }
            plan = PocketPlan(
                generated_at=datetime.datetime.utcnow(),
                pocket="scalp",
                focus_tag="hybrid",
                focus_pockets=["scalp"],
                range_active=False,
                range_soft_active=False,
                range_ctx={},
                event_soon=False,
                spread_gate_active=False,
                spread_gate_reason="",
                spread_log_context="pullback_s5",
                lot_allocation=lot,
                risk_override=0.0,
                weight_macro=0.0,
                scalp_share=1.0,
                signals=[signal],
                perf_snapshot={},
                factors_m1=fac_m1,
                factors_h4=fac_h4,
                notes={},
            )
            plan_bus.publish(plan)
            touch_count = "n/a" if touch_stats is None else str(touch_stats.count)
            touch_pullback = (
                "n/a" if touch_pullback_pips is None else f"{touch_pullback_pips:.2f}"
            )
            touch_trend = "n/a" if touch_trend_pips is None else f"{touch_trend_pips:.2f}"
            touch_age = (
                "n/a" if touch_last_age_sec is None else f"{touch_last_age_sec:.1f}"
            )
            LOG.info(
                "%s publish plan side=%s units=%s tp=%.2f sl=%.2f z_fast=%.2f z_slow=%.2f touch=%s pullback=%s trend=%s age=%s",
                config.LOG_PREFIX,
                side,
                base_units if side == "long" else -base_units,
                tp_pips,
                sl_base,
                z_fast,
                z_slow,
                touch_count,
                touch_pullback,
                touch_trend,
                touch_age,
            )
            if touch_stats:
                log_metric(
                    "pullback_touch_count",
                    float(touch_stats.count),
                    tags={
                        "strategy": "pullback_s5",
                        "side": side,
                        "event": "entry",
                    },
                )
            cooldown_until = now_monotonic + config.COOLDOWN_SEC
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
if __name__ == "__main__":  # pragma: no cover
    asyncio.run(pullback_s5_worker())
