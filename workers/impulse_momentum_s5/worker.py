"""Impulse momentum continuation worker operating on S5 tick aggregates."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.technique_engine import evaluate_entry_techniques
from analysis.ma_projection import score_ma_for_side

import asyncio
import datetime
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from execution.strategy_entry import market_order, set_trade_protections
from execution.position_manager import PositionManager
from execution.risk_guard import loss_cooldown_status
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor, tick_window
from workers.common.quality_gate import current_regime

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


@dataclass
class ManagedTradeState:
    be_applied: bool = False
    last_trail_sl: Optional[float] = None
    last_update: float = field(default_factory=time.monotonic)



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
            fac_m1 = factors.get("M1") or {}
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
            if not _bb_entry_allowed(BB_STYLE, direction, entry_price, fac_m1):
                continue

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
                "env_prefix": config.ENV_PREFIX,
                "entry_price": round(entry_price, 5),
                "entry_price_source": "bidask",
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
                "tp_pips": round(tp_pips, 2),
                "sl_pips": round(sl_pips, 2),
                "hard_stop_pips": round(sl_pips, 2),
            }

            proj_allow, proj_mult, proj_detail = _projection_decision(direction, "scalp")
            if not proj_allow:
                cooldown_until = now_monotonic + config.COOLDOWN_SEC
                continue
            if proj_detail:
                thesis["projection"] = proj_detail
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
    asyncio.run(impulse_momentum_s5_worker())
