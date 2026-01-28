"""Mirror-spike style strategy operating on S5 aggregated data."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.range_guard import detect_range_mode
from indicators.factor_cache import all_factors, get_candles_snapshot

import asyncio
import hashlib
import logging
import os
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
from workers.common.quality_gate import current_regime

from . import config

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

BB_STYLE = "reversion"

LOG = logging.getLogger(__name__)



_PROJ_TF_MINUTES = {"M1": 1.0, "M5": 5.0, "H1": 60.0, "H4": 240.0, "D1": 1440.0}


def _htf_trend_state(fac_h1: Dict) -> tuple[str, float, float] | None:
    adx = _bb_float(fac_h1.get("adx"))
    close = _bb_float(fac_h1.get("close"))
    ma20 = _bb_float(fac_h1.get("ma20")) or _bb_float(fac_h1.get("ema20"))
    if adx is None or close is None or ma20 is None or ma20 <= 0:
        return None
    gap_pips = (close - ma20) / _BB_PIP
    if abs(gap_pips) < config.HTF_GAP_PIPS or adx < config.HTF_ADX_MIN:
        return None
    trend_dir = "long" if gap_pips > 0 else "short"
    return trend_dir, gap_pips, adx


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

            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            fac_h1 = factors.get("H1") or {}
            range_ctx = detect_range_mode(fac_m1, fac_h4)
            if config.RANGE_ONLY and not range_ctx.active:
                continue

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

            htf = _htf_trend_state(fac_h1)
            if htf and config.HTF_BLOCK_COUNTER:
                htf_dir, htf_gap, htf_adx = htf
                if direction != htf_dir:
                    LOG.info(
                        "%s htf_block side=%s h1_dir=%s gap=%.2fp adx=%.1f",
                        config.LOG_PREFIX,
                        direction,
                        htf_dir,
                        htf_gap,
                        htf_adx,
                    )
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
            if config.SLOW_Z_BLOCK > 0 and abs(slow_z) >= config.SLOW_Z_BLOCK:
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

            fac_m1 = all_factors().get("M1") or {}
            if not _bb_entry_allowed(BB_STYLE, direction, entry_price, fac_m1):
                continue

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
                "hard_stop_pips": round(sl_pips, 2),
            }
            entry_mean = 0.5 * (peak["high"] + trough["low"])
            entry_thesis.update(
                {
                    "env_tf": "M5",
                    "struct_tf": "M1",
                    "entry_mean": round(entry_mean, 5),
                    "atr_entry": round(atr, 3),
                    "range_method": "bucket_range",
                    "range_lookback": int(config.LOOKBACK_BUCKETS),
                    "range_hi_pct": 100.0,
                    "range_lo_pct": 0.0,
                    "tp_mode": "soft_zone",
                    "tp_target": "entry_mean",
                    "tp_pad_atr": 0.05,
                    "range_snapshot": {
                        "high": round(peak["high"], 5),
                        "low": round(trough["low"], 5),
                        "mid": round(entry_mean, 5),
                        "method": "bucket_range",
                        "lookback": int(config.LOOKBACK_BUCKETS),
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
            )
            rf = entry_thesis.get("reversion_failure")
            if isinstance(rf, dict):
                bars_budget = rf.get("bars_budget")
                if not isinstance(bars_budget, dict):
                    bars_budget = {}
                    rf["bars_budget"] = bars_budget
                if spike_height >= 7.0:
                    bars_budget["k_per_z"] = 3.0
                    bars_budget["max"] = 10
                elif spike_height <= 4.0:
                    bars_budget["k_per_z"] = 2.0
                    bars_budget["max"] = 6

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
                pocket="scalp",
            )
            if sizing.units <= 0:
                continue
            units = sizing.units if direction == "long" else -sizing.units
            proj_allow, proj_mult, proj_detail = _projection_decision(
                direction,
                "scalp",
                mode_override="range",
            )
            if not proj_allow:
                cooldown_until = now + config.COOLDOWN_SEC
                continue
            if proj_detail:
                entry_thesis["projection"] = proj_detail
            if proj_mult > 1.0:
                sign = 1 if units > 0 else -1
                units = int(round(abs(units) * proj_mult)) * sign
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    LOG.info("%s worker boot (loop %.2fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    asyncio.run(mirror_spike_s5_worker())
