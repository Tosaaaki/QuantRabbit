"""Pullback strategy operating on 5-second synthetic candles."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.technique_engine import evaluate_entry_techniques
from analysis.ma_projection import score_ma_for_side
from analysis.mtf_heat import evaluate_mtf_heat
from analysis.range_guard import detect_range_mode

import asyncio
import datetime
import hashlib
import logging
import math
import time
from collections import deque
from typing import Dict, List, Optional, Sequence

from analysis import plan_bus
from indicators.factor_cache import all_factors, get_candles_snapshot
from execution.strategy_entry import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import clamp_sl_tp
from market_data import spread_monitor, tick_window
from workers.common import env_guard
from workers.common.pocket_plan import PocketPlan
from workers.common.quality_gate import current_regime
from workers.common.pullback_touch import count_pullback_touches
from utils.metrics_logger import log_metric
from utils.divergence import apply_divergence_confidence, divergence_bias, divergence_snapshot

from . import config

import os
from utils.env_utils import env_bool, env_float

_BB_ENV_PREFIX = getattr(config, "ENV_PREFIX", "")
_BB_ENTRY_ENABLED = env_bool("BB_ENTRY_ENABLED", True, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_PIPS = env_float("BB_ENTRY_REVERT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_REVERT_RATIO = env_float("BB_ENTRY_REVERT_RATIO", 0.22, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_PIPS = env_float("BB_ENTRY_TREND_EXT_PIPS", 3.5, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_TREND_EXT_RATIO = env_float("BB_ENTRY_TREND_EXT_RATIO", 0.40, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_PIPS = env_float("BB_ENTRY_SCALP_REVERT_PIPS", 2.0, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_REVERT_RATIO = env_float("BB_ENTRY_SCALP_REVERT_RATIO", 0.20, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_PIPS = env_float("BB_ENTRY_SCALP_EXT_PIPS", 2.4, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_SCALP_EXT_RATIO = env_float("BB_ENTRY_SCALP_EXT_RATIO", 0.30, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_RANGE_FORCE_REVERT = env_bool("BB_ENTRY_RANGE_FORCE_REVERT", True, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_RANGE_BREAK_PIPS = env_float("BB_ENTRY_RANGE_BREAK_PIPS", 1.2, prefix=_BB_ENV_PREFIX)
_BB_ENTRY_RANGE_BREAK_RATIO = env_float("BB_ENTRY_RANGE_BREAK_RATIO", 0.12, prefix=_BB_ENV_PREFIX)
_BB_PIP = 0.01


def _bb_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bb_levels(fac):
    if not fac:
        return None
    upper = _bb_float(fac.get("bb_upper"))
    lower = _bb_float(fac.get("bb_lower"))
    mid = _bb_float(fac.get("bb_mid")) or _bb_float(fac.get("ma20"))
    bbw = _bb_float(fac.get("bbw")) or 0.0
    if upper is None or lower is None:
        if mid is None or bbw <= 0:
            return None
        half = abs(mid) * bbw / 2.0
        upper = mid + half
        lower = mid - half
    span = upper - lower
    if span <= 0:
        return None
    return upper, mid if mid is not None else (upper + lower) / 2.0, lower, span, span / _BB_PIP


def _bb_entry_allowed(style, side, price, fac_m1, *, range_active=None):
    if not _BB_ENTRY_ENABLED:
        return True
    if price is None or price <= 0:
        return True
    levels = _bb_levels(fac_m1)
    if not levels:
        return True
    upper, mid, lower, span, span_pips = levels
    side_key = str(side or "").lower()
    if side_key in {"buy", "long", "open_long"}:
        direction = "long"
    else:
        direction = "short"
    orig_style = style
    if range_active and style != "reversion":
        if _BB_ENTRY_RANGE_FORCE_REVERT:
            break_threshold = max(0.0, _BB_ENTRY_RANGE_BREAK_PIPS, span_pips * max(0.0, _BB_ENTRY_RANGE_BREAK_RATIO))
            break_px = break_threshold * _BB_PIP
            breakout = (
                (direction == "long" and price >= upper + break_px)
                or (direction == "short" and price <= lower - break_px)
            )
            if not breakout:
                style = "reversion"
        elif style == "scalp":
            style = "reversion"
    if style == "reversion":
        base_pips = _BB_ENTRY_SCALP_REVERT_PIPS if orig_style == "scalp" else _BB_ENTRY_REVERT_PIPS
        base_ratio = _BB_ENTRY_SCALP_REVERT_RATIO if orig_style == "scalp" else _BB_ENTRY_REVERT_RATIO
        threshold = max(base_pips, span_pips * base_ratio)
        if direction == "long":
            dist = (price - lower) / _BB_PIP
        else:
            dist = (upper - price) / _BB_PIP
        return dist <= threshold
    if direction == "long":
        if price < mid:
            return False
        ext = max(0.0, price - upper) / _BB_PIP
    else:
        if price > mid:
            return False
        ext = max(0.0, lower - price) / _BB_PIP
    max_ext = max(_BB_ENTRY_TREND_EXT_PIPS, span_pips * _BB_ENTRY_TREND_EXT_RATIO)
    if orig_style == "scalp":
        max_ext = max(_BB_ENTRY_SCALP_EXT_PIPS, span_pips * _BB_ENTRY_SCALP_EXT_RATIO)
    return ext <= max_ext

BB_STYLE = "trend"

LOG = logging.getLogger(__name__)



_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}


def _projection_mode(pocket, mode_override=None):
    if mode_override:
        return mode_override
    if globals().get("IS_RANGE"):
        return "range"
    if globals().get("IS_PULLBACK"):
        return "pullback"
    if pocket in {"scalp", "scalp_fast"}:
        return "scalp"
    return "trend"


def _projection_tfs(pocket, mode):
    if pocket == "macro":
        return ("H4", "H1")
    if pocket == "micro":
        return ("M5", "M1")
    if pocket in {"scalp", "scalp_fast"}:
        return ("M1",)
    return ("M5", "M1")


def _projection_candles(tfs):
    for tf in tfs:
        candles = get_candles_snapshot(tf, limit=120)
        if candles and len(candles) >= 30:
            return tf, list(candles)
    return None, None


def _score_ma(ma, side, opp_block_bars):
    if ma is None:
        return None
    return score_ma_for_side(ma, side, opp_block_bars)


def _score_rsi(rsi, side, long_target, short_target, overheat_bars):
    if rsi is None:
        return None
    score = 0.0
    if side == "long":
        if rsi.rsi >= long_target and rsi.slope_per_bar > 0:
            score = 0.4
        elif rsi.rsi <= (long_target - 8) and rsi.slope_per_bar < 0:
            score = -0.4
        if rsi.eta_upper_bars is not None and rsi.eta_upper_bars <= overheat_bars:
            score -= 0.2
    else:
        if rsi.rsi <= short_target and rsi.slope_per_bar < 0:
            score = 0.4
        elif rsi.rsi >= (short_target + 8) and rsi.slope_per_bar > 0:
            score = -0.4
        if rsi.eta_lower_bars is not None and rsi.eta_lower_bars <= overheat_bars:
            score -= 0.2
    return score


def _score_adx(adx, trend_mode, threshold):
    if adx is None:
        return None
    if trend_mode:
        if adx.adx >= threshold and adx.slope_per_bar >= 0:
            return 0.4
        if adx.adx <= threshold and adx.slope_per_bar < 0:
            return -0.4
        return 0.0
    if adx.adx >= threshold and adx.slope_per_bar > 0:
        return -0.5
    if adx.adx <= threshold and adx.slope_per_bar < 0:
        return 0.3
    return 0.0


def _score_bbw(bbw, threshold):
    if bbw is None:
        return None
    if bbw.bbw <= threshold and bbw.slope_per_bar <= 0:
        return 0.5
    if bbw.bbw > threshold and bbw.slope_per_bar > 0:
        return -0.5
    return 0.0


def _projection_decision(side, pocket, mode_override=None):
    mode = _projection_mode(pocket, mode_override=mode_override)
    tfs = _projection_tfs(pocket, mode)
    tf, candles = _projection_candles(tfs)
    if not candles:
        return True, 1.0, {}
    minutes = _PROJ_TF_MINUTES.get(tf, 1.0)

    if mode == "trend":
        params = {
            "adx_threshold": 20.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 5.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.45, "rsi": 0.25, "adx": 0.30},
            "block_score": -0.6,
            "size_scale": 0.18,
        }
    elif mode == "pullback":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 4.0,
            "long_target": 50.0,
            "short_target": 50.0,
            "overheat_bars": 3.0,
            "weights": {"ma": 0.40, "rsi": 0.40, "adx": 0.20},
            "block_score": -0.55,
            "size_scale": 0.15,
        }
    elif mode == "scalp":
        params = {
            "adx_threshold": 18.0,
            "bbw_threshold": 0.16,
            "opp_block_bars": 3.0,
            "long_target": 52.0,
            "short_target": 48.0,
            "overheat_bars": 2.0,
            "weights": {"ma": 0.50, "rsi": 0.30, "adx": 0.20},
            "block_score": -0.6,
            "size_scale": 0.12,
        }
    else:
        params = {
            "adx_threshold": 16.0,
            "bbw_threshold": 0.14,
            "opp_block_bars": 4.0,
            "long_target": 45.0,
            "short_target": 55.0,
            "overheat_bars": 3.0,
            "weights": {"bbw": 0.40, "rsi": 0.35, "adx": 0.25},
            "block_score": -0.5,
            "size_scale": 0.15,
        }

    ma = compute_ma_projection({"candles": candles}, timeframe_minutes=minutes)
    rsi = compute_rsi_projection(candles, timeframe_minutes=minutes)
    adx = compute_adx_projection(candles, timeframe_minutes=minutes, trend_threshold=params["adx_threshold"])
    bbw = None
    if mode == "range":
        bbw = compute_bbw_projection(candles, timeframe_minutes=minutes, squeeze_threshold=params["bbw_threshold"])

    scores = {}
    ma_score = _score_ma(ma, side, params["opp_block_bars"])
    if ma_score is not None and "ma" in params["weights"]:
        scores["ma"] = ma_score
    rsi_score = _score_rsi(rsi, side, params["long_target"], params["short_target"], params["overheat_bars"])
    if rsi_score is not None and "rsi" in params["weights"]:
        scores["rsi"] = rsi_score
    adx_score = _score_adx(adx, mode != "range", params["adx_threshold"])
    if adx_score is not None and "adx" in params["weights"]:
        scores["adx"] = adx_score
    bbw_score = _score_bbw(bbw, params["bbw_threshold"])
    if bbw_score is not None and "bbw" in params["weights"]:
        scores["bbw"] = bbw_score

    weight_sum = 0.0
    score_sum = 0.0
    for key, score in scores.items():
        weight = params["weights"].get(key, 0.0)
        weight_sum += weight
        score_sum += weight * score
    score = score_sum / weight_sum if weight_sum > 0 else 0.0

    allow = score > params["block_score"]
    size_mult = 1.0 + max(0.0, score) * params["size_scale"]
    size_mult = max(0.8, min(1.35, size_mult))

    detail = {
        "mode": mode,
        "tf": tf,
        "score": round(score, 3),
        "size_mult": round(size_mult, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
    return allow, size_mult, detail
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
            sl_price, tp_price = clamp_sl_tp(
                price=entry_price,
                sl=sl_price,
                tp=tp_price,
                is_buy=side == "long",
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
            try:
                factors = all_factors()
            except Exception:
                factors = {}
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            range_ctx = None
            try:
                range_ctx = detect_range_mode(fac_m1, fac_h4, env_tf="M1", macro_tf="H4")
            except Exception:
                range_ctx = None
            range_active = bool(range_ctx.active) if range_ctx else False
            if not _bb_entry_allowed(
                BB_STYLE,
                side,
                entry_price,
                fac_m1,
                range_active=range_active,
            ):
                continue
            heat_decision = evaluate_mtf_heat(
                side,
                factors,
                price=entry_price,
                env_prefix=config.ENV_PREFIX,
                short_tf="M1",
                mid_tf="M5",
                long_tf="H1",
                macro_tf="H4",
                pivot_tfs=("H1", "H4"),
            )
            base_units = int(round(base_units * heat_decision.lot_mult))
            if base_units < config.MIN_UNITS:
                base_units = config.MIN_UNITS
            tp_pips = max(
                config.TP_ATR_MIN_PIPS,
                min(config.TP_ATR_MAX_PIPS, tp_pips * heat_decision.tp_mult),
            )
            tp_price = round(
                entry_price + tp_pips * config.PIP_VALUE
                if side == "long"
                else entry_price - tp_pips * config.PIP_VALUE,
                3,
            )
            sl_price, tp_price = clamp_sl_tp(
                price=entry_price,
                sl=sl_price,
                tp=tp_price,
                is_buy=side == "long",
            )
            div_bias = divergence_bias(
                fac_m1,
                "OPEN_LONG" if side == "long" else "OPEN_SHORT",
                mode="trend",
                max_age_bars=10,
            )
            if div_bias:
                confidence = apply_divergence_confidence(
                    confidence,
                    div_bias,
                    max_bonus=5.0,
                    max_penalty=8.0,
                    floor=40.0,
                    ceil=95.0,
                )
            confidence = int(max(1, min(99, round(float(confidence) + heat_decision.confidence_delta))))
            lot = abs(base_units) / (100000.0 * (max(1.0, confidence) / 100.0))
            div_meta = divergence_snapshot(fac_m1, max_age_bars=10)
            signal = {
                "action": "OPEN_LONG" if side == "long" else "OPEN_SHORT",
                "pocket": "scalp",
                "strategy": "pullback_s5",
                "tag": "pullback_s5",
                "entry_thesis": {
                    "strategy_tag": "pullback_s5",
                    "entry_price": round(entry_price, 5),
                    "entry_price_source": "bidask",
                    "tp_pips": round(tp_pips, 2),
                    "sl_pips": round(sl_base, 2),
                    "hard_stop_pips": round(sl_base, 2),
                    "range_active": range_active,
                    "range_mode": None if range_ctx is None else range_ctx.mode,
                    "range_reason": None if range_ctx is None else range_ctx.reason,
                    "range_score": None if range_ctx is None else round(float(range_ctx.score or 0.0), 3),
                    "mtf_heat_score": round(heat_decision.score, 3),
                    "mtf_heat_conf_delta": round(heat_decision.confidence_delta, 2),
                    "mtf_heat_lot_mult": round(heat_decision.lot_mult, 3),
                    "mtf_heat_tp_mult": round(heat_decision.tp_mult, 3),
                    "mtf_heat": heat_decision.debug,
                },
                "tp_pips": round(tp_pips, 2),
                "sl_pips": round(sl_base, 2),
                "hard_stop_pips": round(sl_base, 2),
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
                    "mtf_heat_score": round(heat_decision.score, 3),
                },
            }
            if div_meta:
                signal["entry_thesis"]["divergence"] = div_meta
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
            entry_thesis = signal.get("entry_thesis")
            if not isinstance(entry_thesis, dict):
                entry_thesis = {}
            entry_thesis = dict(entry_thesis)
            entry_thesis.setdefault("strategy_tag", "pullback_s5")
            entry_thesis.setdefault("env_prefix", config.ENV_PREFIX)
            proj_allow, proj_mult, proj_detail = _projection_decision(
                side,
                "scalp",
                mode_override="pullback",
            )
            if not proj_allow:
                continue
            if proj_detail:
                entry_thesis["projection"] = proj_detail
            units = base_units if side == "long" else -base_units
            if proj_mult > 1.0:
                sign = 1 if units > 0 else -1
                units = int(round(abs(units) * proj_mult)) * sign
            client_id = _client_id(side)
            candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
            if not candle_allow:
                continue
            if candle_mult != 1.0:
                sign = 1 if units > 0 else -1
                units = int(round(abs(units) * candle_mult)) * sign
            entry_thesis_ctx = None
            for _name in ("entry_thesis", "thesis"):
                _candidate = locals().get(_name)
                if isinstance(_candidate, dict):
                    entry_thesis_ctx = _candidate
                    break
            if entry_thesis_ctx is None:
                entry_thesis_ctx = {}

            _tech_pocket = str(locals().get("pocket", config.POCKET))
            _tech_side_raw = str(locals().get("side", locals().get("direction", "long"))).lower()
            if _tech_side_raw in {"long", "short"}:
                _tech_side = _tech_side_raw
            else:
                _tech_side = "long"
            _tech_entry_price = locals().get("price")
            if not isinstance(_tech_entry_price, (int, float)):
                _tech_entry_price = locals().get("entry_price")
            if not isinstance(_tech_entry_price, (int, float)):
                _tech_entry_price = 0.0
            try:
                _tech_entry_price = float(_tech_entry_price)
            except (TypeError, ValueError):
                _tech_entry_price = 0.0

            _tech_signal_tag = str(
                locals().get("signal_tag")
                or locals().get("strategy_tag")
                or locals().get("STRATEGY_TAG")
                or getattr(config, "STRATEGY_TAG", "")
            )

            entry_thesis_ctx.setdefault(
                "tech_tfs",
                {"fib": ["H1", "M5"], "median": ["H1", "M5"], "nwave": ["M1", "M5"], "candle": ["M1", "M5"]},
            )
            entry_thesis_ctx.setdefault("technical_context_tfs", ["M1", "M5", "H1", "H4"])
            entry_thesis_ctx.setdefault(
                "technical_context_fields",
                [
                    "ma10",
                    "ma20",
                    "rsi",
                    "atr",
                    "atr_pips",
                    "adx",
                    "macd",
                    "macd_hist",
                    "plus_di",
                    "minus_di",
                    "bbw",
                    "kc_width",
                    "vwap",
                    "ema20",
                    "ema24",
                ],
            )
            entry_thesis_ctx.setdefault("technical_context_ticks", ["latest_bid", "latest_ask", "latest_mid", "spread_pips"])
            entry_thesis_ctx.setdefault("technical_context_candle_counts", {"M1": 120, "M5": 80, "H1": 70, "H4": 60})
            entry_thesis_ctx.setdefault("tech_allow_candle", True)
            entry_thesis_ctx.setdefault(
                "tech_policy",
                {
                    "mode": "balanced",
                    "min_score": 0.0,
                    "min_coverage": 0.0,
                    "weight_fib": 0.25,
                    "weight_median": 0.25,
                    "weight_nwave": 0.25,
                    "weight_candle": 0.25,
                    "require_fib": False,
                    "require_median": False,
                    "require_nwave": False,
                    "require_candle": False,
                    "size_scale": 0.15,
                    "size_min": 0.6,
                    "size_max": 1.25,
                },
            )
            entry_thesis_ctx.setdefault("tech_policy_locked", False)
            entry_thesis_ctx.setdefault("env_tf", "M1")
            entry_thesis_ctx.setdefault("struct_tf", "M1")
            entry_thesis_ctx.setdefault("entry_tf", "M1")

            tech_decision = evaluate_entry_techniques(
                entry_price=_tech_entry_price,
                side=_tech_side,
                pocket=_tech_pocket,
                strategy_tag=_tech_signal_tag,
                entry_thesis=entry_thesis_ctx,
                allow_candle=bool(entry_thesis_ctx.get("tech_allow_candle", False)),
            )
            if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
                continue

            entry_thesis_ctx["tech_score"] = round(tech_decision.score, 3) if tech_decision.score is not None else None
            entry_thesis_ctx["tech_coverage"] = (
                round(tech_decision.coverage, 3) if tech_decision.coverage is not None else None
            )
            entry_thesis_ctx["tech_entry"] = tech_decision.debug
            entry_thesis_ctx["tech_reason"] = tech_decision.reason
            entry_thesis_ctx["tech_decision_allowed"] = bool(tech_decision.allowed)

            _tech_units_raw = locals().get("units")
            if isinstance(_tech_units_raw, (int, float)):
                _tech_units = int(round(abs(float(_tech_units_raw)) * tech_decision.size_mult))
                if _tech_units <= 0:
                    continue
                units = _tech_units if _tech_side == "long" else -_tech_units
                entry_thesis_ctx["entry_units_intent"] = abs(int(units))

            _tech_conf = locals().get("conf")
            if isinstance(_tech_conf, (int, float)):
                _tech_conf = float(_tech_conf)
                if tech_decision.score is not None:
                    if tech_decision.score >= 0:
                        _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_BOOST", 0.0)
                    else:
                        _tech_conf += tech_decision.score * getattr(config, "TECH_CONF_PENALTY", 0.0)
                conf = _tech_conf


            res = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket="scalp",
                client_order_id=client_id,
                strategy_tag="pullback_s5",
                confidence=confidence,
                entry_thesis=entry_thesis,
            )
            touch_count = "n/a" if touch_stats is None else str(touch_stats.count)
            touch_pullback = (
                "n/a" if touch_pullback_pips is None else f"{touch_pullback_pips:.2f}"
            )
            touch_trend = "n/a" if touch_trend_pips is None else f"{touch_trend_pips:.2f}"
            touch_age = (
                "n/a" if touch_last_age_sec is None else f"{touch_last_age_sec:.1f}"
            )
            LOG.info(
                "%s publish plan side=%s units=%s tp=%.2f sl=%.2f z_fast=%.2f z_slow=%.2f heat=%.2f touch=%s pullback=%s trend=%s age=%s",
                config.LOG_PREFIX,
                side,
                base_units if side == "long" else -base_units,
                tp_pips,
                sl_base,
                z_fast,
                z_slow,
                heat_decision.score,
                touch_count,
                touch_pullback,
                touch_trend,
                touch_age,
            )
            if res:
                LOG.info(
                    "%s entry sent id=%s units=%s sl=%.3f tp=%.3f",
                    config.LOG_PREFIX,
                    res,
                    units,
                    sl_price or 0.0,
                    tp_price or 0.0,
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


_CANDLE_PIP = 0.01
_CANDLE_MIN_CONF = 0.35
_CANDLE_ENTRY_BLOCK = -0.7
_CANDLE_ENTRY_SCALE = 0.2
_CANDLE_ENTRY_MIN = 0.8
_CANDLE_ENTRY_MAX = 1.2
_CANDLE_WORKER_NAME = (__file__.replace("\\", "/").split("/")[-2] if "/" in __file__ else "").lower()


def _candle_tf_for_worker() -> str:
    name = _CANDLE_WORKER_NAME
    if "macro" in name or "trend_h1" in name or "manual" in name:
        return "H1"
    if "scalp" in name or "s5" in name or "fast" in name:
        return "M1"
    return "M5"


def _extract_candles(raw):
    candles = []
    for candle in raw or []:
        try:
            o = float(candle.get("open", candle.get("o")))
            h = float(candle.get("high", candle.get("h")))
            l = float(candle.get("low", candle.get("l")))
            c = float(candle.get("close", candle.get("c")))
        except Exception:
            continue
        if h <= 0 or l <= 0:
            continue
        candles.append((o, h, l, c))
    return candles


def _detect_candlestick_pattern(candles):
    if len(candles) < 2:
        return None
    o0, h0, l0, c0 = candles[-2]
    o1, h1, l1, c1 = candles[-1]
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range1 = max(h1 - l1, _CANDLE_PIP * 0.1)
    upper_wick = h1 - max(o1, c1)
    lower_wick = min(o1, c1) - l1

    if body1 <= range1 * 0.1:
        return {
            "type": "doji",
            "confidence": round(min(1.0, (range1 - body1) / range1), 3),
            "bias": None,
        }

    if (
        c1 > o1
        and c0 < o0
        and c1 >= max(o0, c0)
        and o1 <= min(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bullish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "up",
        }
    if (
        c1 < o1
        and c0 > o0
        and o1 >= min(o0, c0)
        and c1 <= max(o0, c0)
        and body1 > body0
    ):
        return {
            "type": "bearish_engulfing",
            "confidence": round(min(1.0, body1 / range1 + 0.3), 3),
            "bias": "down",
        }
    if lower_wick > body1 * 2.5 and upper_wick <= body1 * 0.6:
        return {
            "type": "hammer" if c1 >= o1 else "inverted_hammer",
            "confidence": round(min(1.0, lower_wick / range1 + 0.25), 3),
            "bias": "up",
        }
    if upper_wick > body1 * 2.5 and lower_wick <= body1 * 0.6:
        return {
            "type": "shooting_star" if c1 <= o1 else "hanging_man",
            "confidence": round(min(1.0, upper_wick / range1 + 0.25), 3),
            "bias": "down",
        }
    return None


def _score_candle(*, candles, side, min_conf):
    pattern = _detect_candlestick_pattern(_extract_candles(candles))
    if not pattern:
        return None, {}
    bias = pattern.get("bias")
    conf = float(pattern.get("confidence") or 0.0)
    if conf < min_conf:
        return None, {"type": pattern.get("type"), "confidence": round(conf, 3)}
    if bias is None:
        return 0.0, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": None}
    match = (side == "long" and bias == "up") or (side == "short" and bias == "down")
    score = conf if match else -conf * 0.7
    score = max(-1.0, min(1.0, score))
    return score, {"type": pattern.get("type"), "confidence": round(conf, 3), "bias": bias}


def _entry_candle_guard(side):
    tf = _candle_tf_for_worker()
    candles = get_candles_snapshot(tf, limit=4)
    if not candles:
        return True, 1.0
    score, _detail = _score_candle(candles=candles, side=side, min_conf=_CANDLE_MIN_CONF)
    if score is None:
        return True, 1.0
    if score <= _CANDLE_ENTRY_BLOCK:
        return False, 0.0
    mult = 1.0 + score * _CANDLE_ENTRY_SCALE
    mult = max(_CANDLE_ENTRY_MIN, min(_CANDLE_ENTRY_MAX, mult))
    return True, mult

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(pullback_s5_worker())
