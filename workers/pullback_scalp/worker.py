"""Pullback continuation scalp worker."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from datetime import datetime
from typing import Optional

from analysis import plan_bus
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from workers.common.pocket_plan import PocketPlan
from workers.common.pullback_touch import count_pullback_touches
from utils.metrics_logger import log_metric

from . import config

LOG = logging.getLogger(__name__)


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-pullback-{ts_ms}-{side[0]}{digest}"


def _z_score(values: list[float]) -> Optional[float]:
    if len(values) < 20:
        return None
    sample = values[-20:]
    mean_val = sum(sample) / len(sample)
    var = sum((v - mean_val) ** 2 for v in sample) / max(len(sample) - 1, 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (sample[-1] - mean_val) / std


def _rsi(values: list[float], period: int) -> Optional[float]:
    if len(values) <= 1:
        return None
    gains = []
    losses = []
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


def _atr_from_ticks(ticks: list[dict], window_seconds: float) -> float:
    if len(ticks) < 2:
        return 0.0
    now_epoch = float(ticks[-1].get("epoch") or 0.0)
    cutoff = now_epoch - max(10.0, window_seconds)
    filtered: list[float] = []
    for row in reversed(ticks):
        try:
            epoch = float(row.get("epoch") or 0.0)
            mid = float(row.get("mid") or 0.0)
        except (TypeError, ValueError):
            continue
        if epoch < cutoff and filtered:
            break
        filtered.append(mid)
    filtered.reverse()
    if len(filtered) < 2:
        filtered = [float(ticks[-2].get("mid") or 0.0), float(ticks[-1].get("mid") or 0.0)]
    tr_sum = 0.0
    prev = filtered[0]
    for mid in filtered[1:]:
        tr_sum += abs(mid - prev) / config.PIP_VALUE
        prev = mid
    return tr_sum / max(len(filtered) - 1, 1)


async def pullback_scalp_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    cooldown_until = 0.0
    last_spread_log = 0.0
    last_density_log = 0.0
    last_hour_log = 0.0
    last_touch_block_log = 0.0
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_monotonic = time.monotonic()
            if now_monotonic < cooldown_until:
                continue

            if config.ACTIVE_HOURS_UTC:
                current_hour = time.gmtime().tm_hour
                if current_hour not in config.ACTIVE_HOURS_UTC:
                    if now_monotonic - last_hour_log > 300.0:
                        LOG.info(
                            "%s outside active hours hour=%02d",
                            config.LOG_PREFIX,
                            current_hour,
                        )
                        last_hour_log = now_monotonic
                    continue
                last_hour_log = now_monotonic

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

            ticks_m1 = tick_window.recent_ticks(seconds=config.M1_WINDOW_SEC, limit=900)
            ticks_m5 = tick_window.recent_ticks(seconds=config.M5_WINDOW_SEC, limit=1800)
            if len(ticks_m1) < 20 or len(ticks_m5) < 40:
                continue

            mids_m1 = [float(t.get("mid")) for t in ticks_m1]
            mids_m5 = [float(t.get("mid")) for t in ticks_m5]
            z_m1 = _z_score(mids_m1)
            z_m5 = _z_score(mids_m5)
            if z_m1 is None or z_m5 is None:
                continue
            if getattr(config, "M1_Z_TRIGGER", 0.0) > 0.0 and abs(z_m1) < config.M1_Z_TRIGGER:
                continue
            rsi_m1 = _rsi(mids_m1, config.RSI_PERIOD)
            atr_fast = _atr_from_ticks(ticks_m1, config.ATR_FAST_WINDOW_SEC)
            atr_slow = _atr_from_ticks(ticks_m5, config.ATR_SLOW_WINDOW_SEC)
            atr_ref = max(atr_fast, atr_slow * config.ATR_SLOW_WEIGHT)
            if config.MIN_ATR_PIPS > 0.0 and atr_ref < config.MIN_ATR_PIPS:
                continue

            side: Optional[str] = None
            if config.M1_Z_MIN <= z_m1 <= config.M1_Z_MAX and z_m5 <= config.M5_Z_SHORT_MAX:
                if rsi_m1 is None or config.RSI_SHORT_RANGE[0] <= rsi_m1 <= config.RSI_SHORT_RANGE[1]:
                    side = "short"
            elif -config.M1_Z_MAX <= z_m1 <= -config.M1_Z_MIN and z_m5 >= config.M5_Z_LONG_MIN:
                if rsi_m1 is None or config.RSI_LONG_RANGE[0] <= rsi_m1 <= config.RSI_LONG_RANGE[1]:
                    side = "long"
            if side is None:
                continue

            touch_stats = None
            touch_pullback_pips = None
            touch_trend_pips = None
            touch_reset_pips = None
            touch_last_age_sec = None
            if config.TOUCH_ENABLED:
                touch_ticks = tick_window.recent_ticks(
                    seconds=config.TOUCH_WINDOW_SEC,
                    limit=1800,
                )
                touch_prices = []
                touch_times = []
                for row in touch_ticks:
                    mid = row.get("mid")
                    epoch = row.get("epoch")
                    if mid is None or epoch is None:
                        continue
                    try:
                        touch_prices.append(float(mid))
                        touch_times.append(float(epoch))
                    except (TypeError, ValueError):
                        continue
                if len(touch_prices) >= 4:
                    touch_atr_ref = atr_ref if atr_ref > 0.0 else config.TOUCH_PULLBACK_MIN_PIPS
                    touch_pullback_pips = max(
                        config.TOUCH_PULLBACK_MIN_PIPS,
                        min(
                            config.TOUCH_PULLBACK_MAX_PIPS,
                            touch_atr_ref * config.TOUCH_PULLBACK_ATR_MULT,
                        ),
                    )
                    touch_trend_pips = max(
                        config.TOUCH_TREND_MIN_PIPS,
                        min(config.TOUCH_TREND_MAX_PIPS, touch_atr_ref * config.TOUCH_TREND_ATR_MULT),
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
                                    "strategy": "pullback_scalp",
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

            latest_tick = ticks_m1[-1]
            entry_price = float(latest_tick.get("ask")) if side == "long" else float(latest_tick.get("bid"))

            high_vol = bool(
                config.HIGH_VOL_ATR_PIPS > 0.0 and atr_ref >= config.HIGH_VOL_ATR_PIPS
            )
            tp_pips = config.TP_PIPS
            if high_vol:
                tp_pips = min(tp_pips, config.HIGH_VOL_TP_PIPS)
            tp_price = round(
                entry_price + tp_pips * config.PIP_VALUE if side == "long" else entry_price - tp_pips * config.PIP_VALUE,
                3,
            )
            sl_price = None
            if config.USE_INITIAL_SL:
                sl_pips = max(config.SL_MIN_FLOOR_PIPS, atr_ref * config.SL_ATR_MULT)
                if high_vol:
                    sl_pips = min(sl_pips, config.HIGH_VOL_SL_CAP_PIPS)
                sl_price = round(
                    entry_price - sl_pips * config.PIP_VALUE if side == "long" else entry_price + sl_pips * config.PIP_VALUE,
                    3,
                )
            else:
                sl_pips = None

            # Safety: avoid inverted TP/SL before publishing
            if tp_price is not None:
                if side == "long" and tp_price <= entry_price:
                    continue
                if side == "short" and tp_price >= entry_price:
                    continue
            if sl_price is not None:
                if side == "long" and sl_price >= entry_price:
                    sl_price = None
                elif side == "short" and sl_price <= entry_price:
                    sl_price = None

            base_units = config.ENTRY_UNITS
            if high_vol and config.HIGH_VOL_UNIT_FACTOR < 0.999:
                scaled = int(round(base_units * config.HIGH_VOL_UNIT_FACTOR / 1000.0) * 1000)
                base_units = max(1000, scaled)
            # Plan publish (scalp pocket)
            confidence = 85
            if touch_stats and touch_stats.count >= config.TOUCH_SOFT_COUNT:
                orig_units = base_units
                base_units = int(round(base_units * config.TOUCH_UNIT_FACTOR))
                base_units = max(1000, base_units)
                if base_units < orig_units:
                    confidence = max(1, confidence - config.TOUCH_CONF_PENALTY)
            total_lot = base_units / (100000.0 * (confidence / 100.0))
            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            signal = {
                "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
                "pocket": "scalp",
                "strategy": "pullback_scalp",
                "tag": "pullback_scalp",
                "entry_thesis": {
                    "strategy_tag": "pullback_scalp",
                    "entry_guard_pullback": True,
                    "entry_guard_pullback_only": True,
                    "tp_pips": round(tp_pips, 2),
                    "sl_pips": None if sl_pips is None else round(sl_pips, 2),
                    "hard_stop_pips": None if sl_pips is None else round(sl_pips, 2),
                },
                "tp_pips": round(tp_pips, 2),
                "sl_pips": None if sl_price is None or sl_pips is None else round(sl_pips, 2),
                "hard_stop_pips": None if sl_price is None or sl_pips is None else round(sl_pips, 2),
                "confidence": confidence,
                "min_hold_sec": 90,
                "loss_guard_pips": None,
                "target_tp_pips": round(tp_pips, 2),
                "meta": {
                    "z_m1": None if z_m1 is None else round(z_m1, 2),
                    "z_m5": None if z_m5 is None else round(z_m5, 2),
                    "rsi_m1": None if rsi_m1 is None else round(rsi_m1, 1),
                    "atr_fast_pips": round(atr_fast, 2),
                    "atr_slow_pips": round(atr_slow, 2),
                    "atr_ref_pips": round(atr_ref, 2),
                    "spread_pips": round(spread_pips, 2),
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
                generated_at=datetime.utcnow(),
                pocket="scalp",
                focus_tag="hybrid",
                focus_pockets=["scalp"],
                range_active=False,
                range_soft_active=False,
                range_ctx={},
                event_soon=False,
                spread_gate_active=False,
                spread_gate_reason="",
                spread_log_context="pullback_scalp",
                lot_allocation=total_lot,
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
                "%s publish plan side=%s units=%s tp=%.2fp z1=%.2f z5=%.2f rsi=%.1f atr=%.2f touch=%s pullback=%s trend=%s age=%s",
                config.LOG_PREFIX,
                side,
                base_units if side == "long" else -base_units,
                tp_pips,
                z_m1 or 0.0,
                z_m5 or 0.0,
                rsi_m1 or -1.0,
                atr_ref,
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
                        "strategy": "pullback_scalp",
                        "side": side,
                        "event": "entry",
                    },
                )
            cooldown_until = now_monotonic + config.COOLDOWN_SEC
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(pullback_scalp_worker())
