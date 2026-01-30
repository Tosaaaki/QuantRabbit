"""Pullback strategy (S5) with profit runner: extend TP as conditions improve.

Entry logic mirrors pullback_s5; post-entry management moves SL to BE and
extends TP in steps when momentum/trend supports letting profits run.
"""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.range_guard import detect_range_mode

import asyncio
import datetime
import hashlib
import json
import logging
import math
import time
from typing import Dict, List, Optional, Sequence

from execution.order_manager import market_order, set_trade_protections
from workers.common.dyn_size import compute_units
from execution.position_manager import PositionManager
from market_data import spread_monitor, tick_window
from indicators.factor_cache import all_factors, get_candles_snapshot
from workers.common import env_guard
from workers.common.quality_gate import current_regime
from workers.common.pullback_touch import count_pullback_touches
from utils.oanda_account import get_account_snapshot
from utils.metrics_logger import log_metric

from . import config

import os
_BB_ENTRY_ENABLED = os.getenv("BB_ENTRY_ENABLED", "1").strip().lower() not in {"", "0", "false", "no"}
_BB_ENTRY_REVERT_PIPS = float(os.getenv("BB_ENTRY_REVERT_PIPS", "2.4"))
_BB_ENTRY_REVERT_RATIO = float(os.getenv("BB_ENTRY_REVERT_RATIO", "0.22"))
_BB_ENTRY_TREND_EXT_PIPS = float(os.getenv("BB_ENTRY_TREND_EXT_PIPS", "3.5"))
_BB_ENTRY_TREND_EXT_RATIO = float(os.getenv("BB_ENTRY_TREND_EXT_RATIO", "0.40"))
_BB_ENTRY_SCALP_REVERT_PIPS = float(os.getenv("BB_ENTRY_SCALP_REVERT_PIPS", "2.0"))
_BB_ENTRY_SCALP_REVERT_RATIO = float(os.getenv("BB_ENTRY_SCALP_REVERT_RATIO", "0.20"))
_BB_ENTRY_SCALP_EXT_PIPS = float(os.getenv("BB_ENTRY_SCALP_EXT_PIPS", "2.4"))
_BB_ENTRY_SCALP_EXT_RATIO = float(os.getenv("BB_ENTRY_SCALP_EXT_RATIO", "0.30"))
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
    if style == "scalp" and range_active:
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
    align = ma.gap_pips >= 0 if side == "long" else ma.gap_pips <= 0
    cross_soon = ma.projected_cross_bars is not None and ma.projected_cross_bars <= opp_block_bars
    if align and not cross_soon:
        return 0.7
    if align and cross_soon:
        return -0.4
    if cross_soon:
        return -0.8
    return -0.5


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
    ratios = (1.0,)
    if idx < len(ratios):
        return ratios[idx]
    return ratios[-1]


def _client_id(side: str) -> str:
    ts_ms = int(time.time() * 1000)
    digest = hashlib.sha1(f"{ts_ms}-{side}".encode("utf-8")).hexdigest()[:6]
    return f"qr-pullrun-s5-{ts_ms}-{side[0]}{digest}"


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


async def _runner_loop() -> None:
    """Single iteration loop body with defensive guards."""
    pos_manager = PositionManager()
    cooldown_until = 0.0
    last_spread_log = 0.0
    regime_block_logged: Optional[str] = None
    range_block_logged: Optional[str] = None
    loss_block_logged = False
    last_touch_block_log = 0.0
    last_extreme_log = 0.0
    managed_state: Dict[str, float] = {}  # trade_id -> last_update_monotonic
    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_mono = time.monotonic()
            if now_mono < cooldown_until:
                continue

            # Spread gate
            blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = float((spread_state or {}).get("spread_pips", 0.0) or 0.0)
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if now_mono - last_spread_log > 30.0:
                    LOG.info("%s spread gate active spread=%.2fp reason=%s", config.LOG_PREFIX, spread_pips, spread_reason or "guard")
                    last_spread_log = now_mono
                continue

            # Regime guard
            regime_label = current_regime("M1", event_mode=False)
            if regime_label and regime_label in (os.environ.get("PULLBACK_RUNNER_S5_BLOCK_REGIMES", "Event").split(",")):
                if regime_block_logged != regime_label:
                    LOG.info("%s blocked by regime=%s", config.LOG_PREFIX, regime_label)
                    regime_block_logged = regime_label
                continue
            regime_block_logged = None

            # Range mode guard (block entries during compression)
            if config.BLOCK_RANGE_MODE:
                fac_m1 = all_factors().get("M1") or {}
                fac_h4 = all_factors().get("H4") or {}
                range_ctx = detect_range_mode(fac_m1, fac_h4, env_tf="M1", macro_tf="H4")
                if range_ctx.active:
                    if range_block_logged != range_ctx.reason:
                        LOG.info(
                            "%s blocked by range_mode reason=%s score=%.2f",
                            config.LOG_PREFIX,
                            range_ctx.reason,
                            range_ctx.score,
                        )
                        range_block_logged = range_ctx.reason
                    continue
                range_block_logged = None

            # Build S5 buckets
            ticks = tick_window.recent_ticks(seconds=config.WINDOW_SEC, limit=3600)
            if len(ticks) < config.MIN_BUCKETS:
                continue
            candles: List[Dict[str, float]] = []
            buckets: Dict[int, Dict[str, float]] = {}
            for tick in ticks:
                mid = tick.get("mid")
                epoch = tick.get("epoch")
                if mid is None or epoch is None:
                    continue
                price = float(mid)
                b_id = int(float(epoch) // config.BUCKET_SECONDS)
                c = buckets.get(b_id)
                if c is None:
                    c = {
                        "start": b_id * config.BUCKET_SECONDS,
                        "end": (b_id + 1) * config.BUCKET_SECONDS,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                    }
                    buckets[b_id] = c
                else:
                    c["high"] = max(c["high"], price)
                    c["low"] = min(c["low"], price)
                    c["close"] = price
            candles = [buckets[idx] for idx in sorted(buckets)]
            if len(candles) < config.MIN_BUCKETS:
                continue
            closes = [c["close"] for c in candles]
            fast_c = candles[-config.FAST_BUCKETS :]
            fast_series = closes[-config.FAST_BUCKETS :]
            slow_series = closes[-config.SLOW_BUCKETS :]
            z_fast = _z_score(fast_series)
            z_slow = _z_score(slow_series)
            if z_fast is None or z_slow is None:
                continue
            atr_fast = _atr_from_candles(fast_c, config.RSI_PERIOD)
            if atr_fast < config.TP_ATR_MIN_PIPS * 0.4:  # require some activity
                continue
            rsi_fast = _rsi(fast_series, config.RSI_PERIOD)

            # Entry select (same logic as pullback_s5)
            side: Optional[str] = None
            if config.FAST_Z_MIN <= z_fast <= config.FAST_Z_MAX and z_slow <= config.SLOW_Z_SHORT_MAX:
                if rsi_fast is None or config.RSI_SHORT_RANGE[0] <= rsi_fast <= config.RSI_SHORT_RANGE[1]:
                    side = "short"
            elif -config.FAST_Z_MAX <= z_fast <= -config.FAST_Z_MIN and z_slow >= config.SLOW_Z_LONG_MIN:
                if rsi_fast is None or config.RSI_LONG_RANGE[0] <= rsi_fast <= config.RSI_LONG_RANGE[1]:
                    side = "long"
            if side is None:
                pass  # no entry this loop
            else:
                # Trend/ADX alignment
                factors = all_factors()
                trend_bias: Optional[str] = None
                adx_value: Optional[float] = None
                for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                    ma10 = fac.get("ma10")
                    ma20 = fac.get("ma20")
                    if ma10 is not None and ma20 is not None:
                        diff = float(ma10) - float(ma20)
                        if abs(diff) / config.PIP_VALUE >= 0.6:
                            trend_bias = "long" if diff > 0 else "short"
                            break
                for fac in (factors.get("H4") or {}, factors.get("M1") or {}):
                    adx = fac.get("adx")
                    if adx is not None:
                        adx_value = float(adx)
                        break
                if trend_bias and trend_bias != side:
                    side = None
                if adx_value is not None and adx_value < config.ADX_TREND_MIN:
                    side = None

            touch_stats = None
            touch_pullback_pips = None
            touch_trend_pips = None
            touch_reset_pips = None
            touch_last_age_sec = None
            if side and config.TOUCH_ENABLED:
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
                        if now_mono - last_touch_block_log > 30.0:
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
                                    "strategy": "pullback_runner_s5",
                                    "side": side,
                                    "event": "block",
                                },
                            )
                            last_touch_block_log = now_mono
                        side = None
                    if touch_stats.last_touch_ts is not None and side:
                        try:
                            touch_last_age_sec = max(
                                0.0,
                                float(touch_times[-1]) - float(touch_stats.last_touch_ts),
                            )
                        except (TypeError, ValueError):
                            touch_last_age_sec = None

            if side:
                latest_tick = ticks[-1]
                bid = float(latest_tick.get("bid") or closes[-1])
                ask = float(latest_tick.get("ask") or closes[-1])
                entry_price = ask if side == "long" else bid
                if config.EXTREME_GUARD_ENABLED and entry_price > 0:
                    lookback = int(config.EXTREME_LOOKBACK_SEC / config.BUCKET_SECONDS)
                    lookback = max(4, min(len(candles), lookback))
                    recent = candles[-lookback:] if lookback > 0 else candles
                    if recent:
                        recent_high = max(float(c["high"]) for c in recent)
                        recent_low = min(float(c["low"]) for c in recent)
                        guard_pips = max(config.EXTREME_MIN_PIPS, atr_fast * config.EXTREME_ATR_MULT)
                        guard_pips = min(guard_pips, config.EXTREME_MAX_PIPS)
                        guard_price = guard_pips * config.PIP_VALUE
                        block = False
                        # Avoid catching exact bottoms/tops: require a small bounce
                        if side == "long" and entry_price <= recent_low + guard_price:
                            block = True
                        elif side == "short" and entry_price >= recent_high - guard_price:
                            block = True
                        if block:
                            if now_mono - last_extreme_log > 30.0:
                                LOG.info(
                                    "%s extreme guard block side=%s entry=%.3f hi=%.3f lo=%.3f guard=%.2fp",
                                    config.LOG_PREFIX,
                                    side,
                                    entry_price,
                                    recent_high,
                                    recent_low,
                                    guard_pips,
                                )
                                last_extreme_log = now_mono
                            side = None
                if side:
                    fac_m1 = all_factors().get("M1") or {}
                    if not _bb_entry_allowed(BB_STYLE, side, entry_price, fac_m1):
                        side = None
            if side:

                # SL/TP baseline
                tp_pips = config.TP_PIPS
                sl_base = max(config.MIN_SL_PIPS, atr_fast * config.SL_ATR_MULT)
                sl_base = min(sl_base, config.MAX_SL_PIPS)
                spread_floor = spread_pips * config.SL_SPREAD_MULT + config.SL_SPREAD_MIN_PIPS
                sl_base = max(sl_base, spread_floor)
                if atr_fast > 0.0:
                    tp_pips = min(config.TP_ATR_MAX_PIPS, max(config.TP_ATR_MIN_PIPS, atr_fast * config.TP_ATR_MULT))
                min_tp = sl_base * config.MIN_RR + spread_pips * config.TP_SPREAD_BUFFER_PIPS
                tp_pips = max(tp_pips, min_tp)
                tp_price = round(entry_price + (tp_pips * config.PIP_VALUE if side == "long" else -tp_pips * config.PIP_VALUE), 3)
                sl_price = round(entry_price - (sl_base * config.PIP_VALUE if side == "long" else -sl_base * config.PIP_VALUE), 3)

                # 柔軟サイズ決定
                entry_units = int(config.ENTRY_UNITS)
                if touch_stats and touch_stats.count >= config.TOUCH_SOFT_COUNT:
                    entry_units = int(round(entry_units * config.TOUCH_UNIT_FACTOR))
                    entry_units = max(entry_units, int(config.MIN_UNITS))
                sizing = compute_units(
                    entry_price=float(entry_price),
                    sl_pips=float(sl_base),
                    base_entry_units=entry_units,
                    min_units=int(config.MIN_UNITS),
                    max_margin_usage=float(config.MAX_MARGIN_USAGE),
                    spread_pips=float(spread_pips),
                    spread_soft_cap=float(config.MAX_SPREAD_PIPS),
                    adx=adx_value,
                    signal_score=None,
                    pocket="scalp",
                    strategy_tag="pullback_runner_s5",
                )
                if sizing.units >= int(config.MIN_UNITS):
                    units = sizing.units if side == "long" else -sizing.units
                    thesis = {
                        "strategy_tag": "pullback_runner_s5",
                        "entry_price": round(entry_price, 5),
                        "entry_price_source": "bidask",
                        "z_fast": round(z_fast, 2),
                        "z_slow": round(z_slow, 2),
                        "rsi_fast": None if rsi_fast is None else round(rsi_fast, 1),
                        "atr_fast_pips": round(atr_fast, 2),
                        "spread_pips": round(spread_pips, 2),
                        "tp_pips": round(tp_pips, 2),
                        "sl_pips": round(sl_base, 2),
                        "hard_stop_pips": round(sl_base, 2),
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
                    }
                    proj_allow, proj_mult, proj_detail = _projection_decision(
                        side,
                        "scalp",
                        mode_override="pullback",
                    )
                    if not proj_allow:
                        cooldown_until = now_mono + config.COOLDOWN_SEC
                        continue
                    if proj_detail:
                        thesis["projection"] = proj_detail
                    proj_score = None
                    try:
                        if isinstance(proj_detail, dict) and proj_detail.get("score") is not None:
                            proj_score = float(proj_detail.get("score"))
                    except Exception:
                        proj_score = None
                    if proj_mult > 1.0:
                        sign = 1 if units > 0 else -1
                        units = int(round(abs(units) * proj_mult)) * sign
                    if proj_score is not None and proj_score <= config.NEG_SCORE_THRESHOLD:
                        sign = 1 if units > 0 else -1
                        reduced = int(round(abs(units) * config.NEG_SCORE_UNIT_FACTOR))
                        reduced = max(int(config.MIN_UNITS), reduced)
                        if reduced != abs(units):
                            units = reduced * sign
                            thesis["neg_score_unit_factor"] = round(config.NEG_SCORE_UNIT_FACTOR, 3)
                    try:
                        candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
                        if not candle_allow:
                            continue
                        if candle_mult != 1.0:
                            sign = 1 if units > 0 else -1
                            units = int(round(abs(units) * candle_mult)) * sign
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
                    except Exception as exc:  # network
                        LOG.error("%s order error side=%s exc=%s", config.LOG_PREFIX, side, exc)
                        cooldown_until = now_mono + config.COOLDOWN_SEC
                        trade_id = None

                    if trade_id:
                        LOG.info(
                            "%s entry trade=%s side=%s units=%s tp=%.3f sl=%.3f zf=%.2f zs=%.2f atr=%.2f touch=%s pullback=%s trend=%s",
                            config.LOG_PREFIX,
                            trade_id,
                            side,
                            units,
                            tp_price,
                            sl_price,
                            z_fast,
                            z_slow,
                            atr_fast,
                            "n/a" if touch_stats is None else str(touch_stats.count),
                            "n/a"
                            if touch_pullback_pips is None
                            else f"{touch_pullback_pips:.2f}",
                            "n/a"
                            if touch_trend_pips is None
                            else f"{touch_trend_pips:.2f}",
                        )
                        if touch_stats:
                            log_metric(
                                "pullback_touch_count",
                                float(touch_stats.count),
                                tags={
                                    "strategy": "pullback_runner_s5",
                                    "side": side,
                                    "event": "entry",
                                },
                            )
                        cooldown_until = now_mono + config.COOLDOWN_SEC
                    else:
                        cooldown_until = now_mono + 10.0

                # Post-entry management: extend TP and set BE
                try:
                    pockets = pos_manager.get_open_positions()
                except Exception as exc:
                    LOG.debug("%s get_open_positions error: %s", config.LOG_PREFIX, exc)
                    continue
                scalp_pos = pockets.get("scalp") or {}
                open_trades = (scalp_pos.get("open_trades") or [])
                if not open_trades:
                    continue
                # Use latest mid for profit calc
                last_tick = tick_window.recent_ticks(1.0, limit=1)
                latest_mid = None
                if last_tick:
                    t = last_tick[-1]
                    bid = t.get("bid") or 0.0
                    ask = t.get("ask") or 0.0
                    try:
                        latest_mid = float((bid + ask) / 2.0) if (bid or ask) else None
                    except Exception:
                        latest_mid = None

                for tr in open_trades:
                    trade_id = str(tr.get("trade_id") or "")
                    if not trade_id:
                        continue
                    client_id = tr.get("client_id") or ""
                    if "qr-pullrun-s5-" not in str(client_id):
                        continue
                    try:
                        units = int(tr.get("units") or 0)
                        entry = float(tr.get("price") or 0.0)
                    except Exception:
                        continue
                    if units == 0 or entry == 0.0:
                        continue
                    side = "long" if units > 0 else "short"
                    price = latest_mid
                    if price is None:
                        try:
                            price = float(tr.get("price"))
                        except Exception:
                            continue
                    pip = config.PIP_VALUE
                    profit_pips = (price - entry) / pip if side == "long" else (entry - price) / pip

                    # Break-even move
                    last_upd = managed_state.get(trade_id, 0.0)
                    if profit_pips >= config.BE_TRIGGER_PIPS and (now_mono - last_upd) >= 2.0:
                        be = entry + config.BE_OFFSET_PIPS * pip if side == "long" else entry - config.BE_OFFSET_PIPS * pip
                        ok = await set_trade_protections(trade_id, sl_price=round(be, 3), tp_price=None)
                        if ok:
                            managed_state[trade_id] = now_mono

                    # Extend TP when in profit
                    if profit_pips >= config.EXTEND_TRIGGER_PIPS and (now_mono - managed_state.get(trade_id, 0.0)) >= config.EXTEND_COOLDOWN_SEC:
                        # Simple policy: set TP to (profit + step), capped by EXTEND_MAX_PIPS
                        extra = min(config.EXTEND_STEP_PIPS, max(0.6, atr_fast * 0.5))
                        target = profit_pips + extra
                        target = min(max(target, config.TP_ATR_MIN_PIPS), config.EXTEND_MAX_PIPS)
                        tp_price = entry + target * pip if side == "long" else entry - target * pip
                        ok = await set_trade_protections(trade_id, sl_price=None, tp_price=round(tp_price, 3))
                        if ok:
                            managed_state[trade_id] = now_mono
    finally:
        try:
            pos_manager.close()
        except Exception:
            LOG.exception("%s failed to close PositionManager", config.LOG_PREFIX)


async def pullback_runner_s5_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    LOG.info("%s worker starting", config.LOG_PREFIX)
    while True:
        try:
            await _runner_loop()
        except asyncio.CancelledError:
            LOG.info("%s worker cancelled", config.LOG_PREFIX)
            raise
        except Exception:
            LOG.exception("%s loop error; continuing", config.LOG_PREFIX)
            await asyncio.sleep(1.5)




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
    asyncio.run(pullback_runner_s5_worker())
