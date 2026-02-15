"""Replicate the July 2025 manual swing behaviour with automated guardrails."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.ma_projection import score_ma_for_side
from analysis.mtf_heat import evaluate_mtf_heat
from analysis.range_guard import detect_range_mode

import asyncio
import datetime as dt
import logging
import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

from analysis.technique_engine import evaluate_entry_techniques
from execution.strategy_entry import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import can_trade, clamp_sl_tp, loss_cooldown_status
from indicators.factor_cache import all_factors, get_candles_snapshot
from market_data import spread_monitor, tick_window
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
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
STRATEGY_TAG = "manual_swing"



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


def _evaluate_entry_techniques_local(
    *,
    entry_price: float,
    side: str,
    pocket: str,
    strategy_tag: str,
    entry_thesis: Dict[str, object],
) -> tuple:
    thesis_ctx = dict(entry_thesis)
    thesis_ctx.setdefault("technical_context_tfs", {
        "fib": ["H4", "H1", "M5", "M1"],
        "median": ["H4", "H1", "M5", "M1"],
        "nwave": ["H1", "M1"],
        "candle": ["M1", "M5"],
    })
    thesis_ctx.setdefault(
        "technical_context_ticks",
        ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
    )
    thesis_ctx.setdefault(
        "technical_context_candle_counts",
        {"M1": 120, "M5": 80, "H1": 60, "H4": 40},
    )
    thesis_ctx.setdefault(
        "tech_policy",
        {
            "require_fib": False,
            "require_median": False,
            "require_nwave": False,
            "require_candle": False,
        },
    )
    thesis_ctx.setdefault("tech_allow_candle", True)
    thesis_ctx.setdefault("tech_policy_locked", False)
    thesis_ctx.setdefault("env_tf", "M1")
    thesis_ctx.setdefault("struct_tf", "M5")
    thesis_ctx.setdefault("entry_tf", "M1")

    tech_decision = evaluate_entry_techniques(
        entry_price=entry_price,
        side=side,
        pocket=pocket,
        strategy_tag=strategy_tag,
        entry_thesis=thesis_ctx,
        allow_candle=bool(thesis_ctx.get("tech_allow_candle", False)),
    )

    thesis_ctx["tech_score"] = round(tech_decision.score, 3) if tech_decision.score is not None else None
    if tech_decision.coverage is not None:
        thesis_ctx["tech_coverage"] = round(float(tech_decision.coverage), 3)
    thesis_ctx["tech_entry"] = tech_decision.debug
    thesis_ctx["tech_reason"] = tech_decision.reason
    thesis_ctx["tech_decision_allowed"] = bool(tech_decision.allowed)
    if tech_decision.score is None:
        thesis_ctx["entry_probability"] = 0.5
    else:
        thesis_ctx["entry_probability"] = max(0.0, min(1.0, 0.5 + (tech_decision.score / 2.0)))

    return tech_decision, thesis_ctx


def _client_order_id(suffix: str) -> str:
    ts_ms = int(time.time() * 1000)
    return f"qr-manual-{ts_ms}-{suffix}"


def _latest_mid(fallback: float) -> float:
    """Fallback-safe mid price from recent tick."""
    ticks = tick_window.recent_ticks(seconds=15.0, limit=1)
    if ticks:
        tick = ticks[-1]
        try:
            bid = float(tick.get("bid"))
            ask = float(tick.get("ask"))
            if math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0:
                return (bid + ask) * 0.5
        except (TypeError, ValueError):
            pass
        mid = tick.get("mid")
        if mid is not None:
            try:
                mid_f = float(mid)
                if math.isfinite(mid_f) and mid_f > 0:
                    return mid_f
            except (TypeError, ValueError):
                pass
    return fallback


def _stage_thresholds(stages: Iterable[int]) -> List[int]:
    thresholds: List[int] = []
    total = 0
    for units in stages:
        total += abs(int(units))
        thresholds.append(total)
    return thresholds


def _scaled_stage_units(equity: float) -> List[int]:
    if equity <= 0:
        equity = config.REFERENCE_EQUITY
    scale = equity / config.REFERENCE_EQUITY if config.REFERENCE_EQUITY else 1.0
    units: List[int] = []
    for base in config.STAGE_UNITS_BASE[: config.MAX_ACTIVE_STAGES]:
        scaled = int(round(base * scale))
        units.append(max(config.MIN_STAGE_UNITS, scaled))
    return units


def _parse_iso(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    try:
        parsed = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _calc_sl_tp(
    side: str,
    price: float,
    sl_pips: float,
    tp_pips: float,
) -> Tuple[Optional[float], Optional[float]]:
    pip = config.PIP_VALUE
    if side == "long":
        sl = price - sl_pips * pip
        tp = price + tp_pips * pip
    else:
        sl = price + sl_pips * pip
        tp = price - tp_pips * pip
    return round(sl, 3), round(tp, 3)


def _trend_bias(
    fac_h1: Dict[str, float],
    fac_h4: Dict[str, float],
) -> Tuple[Optional[str], Dict[str, float]]:
    features: Dict[str, float] = {}
    try:
        ma10_h4 = float(fac_h4.get("ma10"))
        ma20_h4 = float(fac_h4.get("ma20"))
        features["ma_gap_h4"] = ma10_h4 - ma20_h4
    except (TypeError, ValueError):
        features["ma_gap_h4"] = 0.0
    try:
        ma10_h1 = float(fac_h1.get("ma10"))
        ma20_h1 = float(fac_h1.get("ma20"))
        features["ma_gap_h1"] = ma10_h1 - ma20_h1
    except (TypeError, ValueError):
        features["ma_gap_h1"] = 0.0
    try:
        adx = float(fac_h1.get("adx"))
    except (TypeError, ValueError):
        adx = 0.0
    features["adx"] = adx
    direction: Optional[str] = None
    if (
        features["ma_gap_h4"] >= config.H4_GAP_MIN
        and features["ma_gap_h1"] >= config.H1_GAP_MIN
        and adx >= config.ADX_MIN
    ):
        direction = "long"
    elif (
        features["ma_gap_h4"] <= -config.H4_GAP_MIN
        and features["ma_gap_h1"] <= -config.H1_GAP_MIN
        and adx >= config.ADX_MIN
    ):
        direction = "short"
    return direction, features


def _open_position_summary(
    pocket_state: Dict[str, object]
) -> Tuple[int, Optional[dt.datetime]]:
    net_units = int(pocket_state.get("units") or 0)
    open_trades: List[Dict[str, object]] = pocket_state.get("open_trades") or []
    if open_trades:
        oldest = min(
            (_parse_iso(trade.get("open_time")) for trade in open_trades),
            key=lambda x: x or dt.datetime.now(dt.timezone.utc),
        )
    else:
        oldest = None
    return net_units, oldest


def _hold_hours(opened_at: Optional[dt.datetime], now: dt.datetime) -> Optional[float]:
    if not opened_at:
        return None
    delta = now - opened_at
    return delta.total_seconds() / 3600.0


async def manual_swing_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return

    if config.POCKET not in {"micro", "macro", "scalp"}:
        LOG.error(
            "%s invalid pocket=%s; allowed only micro/macro/scalp",
            config.LOG_PREFIX,
            config.POCKET,
        )
        return

    LOG.info("%s worker starting (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    pos_manager = PositionManager()
    stage_cooldown_until: Dict[str, float] = {"long": 0.0, "short": 0.0}
    last_sync_perf = 0.0
    skip_state: Dict[str, float] = {"ts": 0.0, "reason": ""}
    position_tracker: Dict[str, object] = {
        "side": None,
        "best_pips": 0.0,
        "last_stage_price": None,
        "stage_count": 0,
    }

    try:
        while True:
            await asyncio.sleep(config.LOOP_INTERVAL_SEC)
            now_mono = time.monotonic()
            now_utc = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

            if config.ALLOWED_HOURS_UTC and now_utc.hour not in config.ALLOWED_HOURS_UTC:
                continue
            if config.BLOCKED_WEEKDAYS:
                if str(now_utc.weekday()) in config.BLOCKED_WEEKDAYS:
                    continue

            if not is_market_open(now_utc):
                continue

            if not can_trade(config.POCKET):
                continue

            loss_block, loss_remain = loss_cooldown_status(
                config.POCKET,
                max_losses=config.LOSS_STREAK_MAX,
                cooldown_minutes=config.LOSS_STREAK_COOLDOWN_MIN,
            )
            if loss_block:
                if config.LOG_SKIP_REASON and now_mono - skip_state.get("ts", 0.0) > 120.0:
                    LOG.info(
                        "%s cooldown in effect pocket=%s remaining=%.0fs",
                        config.LOG_PREFIX,
                        config.POCKET,
                        loss_remain,
                    )
                    skip_state["ts"] = now_mono
                continue

            blocked, recovery, spread_state, reason = spread_monitor.is_blocked()
            spread_pips = float((spread_state or {}).get("spread_pips") or 0.0)
            if blocked or spread_pips > config.SPREAD_MAX_PIPS:
                continue
            if recovery and spread_pips > config.SPREAD_RECOVERY_PIPS:
                continue

            regime = current_regime()
            if regime and regime.lower() in {"event", "halt"}:
                continue

            factors = all_factors()
            fac_h1 = factors.get("H1")
            fac_h4 = factors.get("H4")
            if not fac_h1 or not fac_h4:
                continue
            range_ctx = None
            try:
                range_ctx = detect_range_mode(
                    fac_h1,
                    fac_h4,
                    env_tf="H1",
                    macro_tf="H4",
                )
            except Exception:
                range_ctx = None
            range_active = bool(range_ctx.active) if range_ctx else False
            direction, features = _trend_bias(fac_h1, fac_h4)
            if direction is None:
                continue

            candles_h1 = fac_h1.get("candles") or []
            try:
                last_close = float(candles_h1[-1]["close"])
            except Exception:
                last_close = float(fac_h1.get("ma10") or 0.0)
            price = _latest_mid(last_close or 150.0)

            atr_pips = float(fac_h1.get("atr_pips") or 0.0)
            if atr_pips < config.ATR_MIN_PIPS:
                continue
            if not _bb_entry_allowed(
                BB_STYLE,
                direction,
                price,
                fac_h1,
                range_active=range_active,
            ):
                continue
            heat_decision = evaluate_mtf_heat(
                direction,
                factors,
                price=price,
                env_prefix=config.ENV_PREFIX,
                short_tf="H1",
                mid_tf="H4",
                long_tf="H4",
                macro_tf="D1",
                pivot_tfs=("H1", "H4"),
            )

            snapshot = get_account_snapshot(timeout=6.0)
            if (
                snapshot.free_margin_ratio is not None
                and snapshot.free_margin_ratio < config.MIN_FREE_MARGIN_RATIO
            ):
                LOG.warning(
                    "%s free margin %.2f below limit %.2f – skipping entries",
                    config.LOG_PREFIX,
                    snapshot.free_margin_ratio,
                    config.MIN_FREE_MARGIN_RATIO,
                )
                direction = None

            pocket_positions = pos_manager.get_open_positions()
            pocket_state: Dict[str, object] = pocket_positions.get(config.POCKET, {})
            net_units, oldest_trade = _open_position_summary(pocket_state)
            sign = 1 if net_units >= 0 else -1
            hold_hours = _hold_hours(oldest_trade, now_utc)

            if net_units == 0:
                position_tracker["side"] = None
                position_tracker["best_pips"] = 0.0
                position_tracker["last_stage_price"] = None
                position_tracker["stage_count"] = 0

            stage_units = _scaled_stage_units(snapshot.nav or snapshot.balance or config.REFERENCE_EQUITY)
            if not stage_units:
                continue
            max_stage_count = min(config.MAX_ACTIVE_STAGES, len(stage_units))

            # Flatten if regime flips or hold exceeds max
            should_exit = False
            exit_reason = ""
            if net_units != 0:
                active_side = "long" if net_units > 0 else "short"
                if direction and active_side != direction:
                    should_exit = True
                    exit_reason = "trend_flip"
                elif hold_hours and hold_hours >= config.MAX_HOLD_HOURS:
                    should_exit = True
                    exit_reason = "max_hold"
                elif (
                    snapshot.health_buffer is not None
                    and snapshot.health_buffer < config.MARGIN_HEALTH_EXIT
                ):
                    should_exit = True
                    exit_reason = "margin_health"
                else:
                    # drawdown guard: compare price vs entry avg
                    avg_price = pocket_state.get("avg_price")
                    if avg_price:
                        try:
                            avg_price_val = float(avg_price)
                            favourable_pips = (
                                (price - avg_price_val) / config.PIP_VALUE
                                if net_units > 0
                                else (avg_price_val - price) / config.PIP_VALUE
                            )
                            if favourable_pips < -config.MAX_DRAWDOWN_PIPS:
                                should_exit = True
                                exit_reason = "max_drawdown"
                            else:
                                current_side = "long" if net_units > 0 else "short"
                                if position_tracker.get("side") != current_side:
                                    position_tracker["side"] = current_side
                                    position_tracker["best_pips"] = 0.0
                                best_pips = float(position_tracker.get("best_pips") or 0.0)
                                if favourable_pips > best_pips:
                                    position_tracker["best_pips"] = favourable_pips
                                    best_pips = favourable_pips
                                if best_pips >= config.PROFIT_TRIGGER_PIPS:
                                    should_exit = True
                                    exit_reason = "profit_lock"
                                elif (
                                    best_pips >= config.TRAIL_TRIGGER_PIPS
                                    and favourable_pips
                                    <= best_pips - config.TRAIL_BACKOFF_PIPS
                                ):
                                    should_exit = True
                                    exit_reason = "trail_backoff"
                                elif (
                                    features.get("ma_gap_h1") is not None
                                    and (
                                        (
                                            net_units > 0
                                            and features["ma_gap_h1"]
                                            < -config.REVERSAL_GAP_EXIT
                                        )
                                        or (
                                            net_units < 0
                                            and features["ma_gap_h1"]
                                            > config.REVERSAL_GAP_EXIT
                                        )
                                    )
                                ):
                                    should_exit = True
                                    exit_reason = "gap_reversal"
                        except Exception:
                            pass

            if should_exit and net_units != 0:
                units_to_close = -net_units
                LOG.info(
                    "%s closing position units=%s reason=%s",
                    config.LOG_PREFIX,
                    units_to_close,
                    exit_reason,
                )
                candle_allow, candle_mult = _entry_candle_guard("long" if net_units > 0 else "short")
                if not candle_allow:
                    continue
                if candle_mult != 1.0:
                    sign = 1 if units_to_close > 0 else -1
                    units_to_close = int(round(abs(units_to_close) * candle_mult)) * sign
                await market_order(
                    "USD_JPY",
                    units_to_close,
                    sl_price=None,
                    tp_price=None,
                    pocket=config.POCKET,  # type: ignore[arg-type]
                    reduce_only=True,
                    client_order_id=_client_order_id("close"),
                    strategy_tag=STRATEGY_TAG,
                    entry_thesis={
                        "exit_reason": exit_reason,
                        "strategy_tag": STRATEGY_TAG,
                        "env_prefix": config.ENV_PREFIX,
                    },
                )
                stage_cooldown_until["long"] = now_mono + 300.0
                stage_cooldown_until["short"] = now_mono + 300.0
                position_tracker["side"] = None
                position_tracker["best_pips"] = 0.0
                position_tracker["last_stage_price"] = None
                position_tracker["stage_count"] = 0
                continue

            if direction is None:
                continue

            # If in cooldown for this direction, skip new entries
            if now_mono < stage_cooldown_until[direction]:
                continue

            current_stage_count = int(position_tracker.get("stage_count") or 0)
            if net_units != 0:
                position_tracker["side"] = "long" if net_units > 0 else "short"
                inferred_stage = 0
                cumulative = 0
                abs_units = abs(net_units)
                for size in stage_units:
                    cumulative += size
                    if abs_units >= cumulative * 0.6:
                        inferred_stage += 1
                if inferred_stage > current_stage_count:
                    current_stage_count = min(inferred_stage, max_stage_count)
                    position_tracker["stage_count"] = current_stage_count
                    if position_tracker.get("last_stage_price") is None:
                        try:
                            position_tracker["last_stage_price"] = float(pocket_state.get("avg_price"))
                        except Exception:
                            position_tracker["last_stage_price"] = price

            if current_stage_count >= max_stage_count:
                continue

            stage_size = stage_units[current_stage_count]
            stage_size = int(round(stage_size * heat_decision.lot_mult))
            if stage_size <= 0:
                continue

            if current_stage_count > 0 and position_tracker.get("side") == direction:
                anchor_price = position_tracker.get("last_stage_price")
                if anchor_price is None:
                    anchor_price = pocket_state.get("avg_price") or price
                try:
                    anchor_price_val = float(anchor_price)
                    favourable_move = (
                        (price - anchor_price_val) / config.PIP_VALUE
                        if direction == "long"
                        else (anchor_price_val - price) / config.PIP_VALUE
                    )
                except Exception:
                    favourable_move = 0.0
                if favourable_move < config.STAGE_ADD_TRIGGER_PIPS:
                    continue
            elif net_units != 0:
                continue

            sl_pips = max(config.MIN_SL_PIPS, atr_pips * config.SL_ATR_MULT)
            tp_pips = max(config.MIN_TP_PIPS, atr_pips * config.TP_ATR_MULT)
            tp_pips = max(config.MIN_TP_PIPS, tp_pips * heat_decision.tp_mult)

            # Determine units based on free margin usage
            margin_available = snapshot.margin_available
            margin_rate = snapshot.margin_rate
            if margin_available <= 0 or margin_rate <= 0:
                continue
            leverage_budget = margin_available * config.RISK_FREE_MARGIN_FRACTION
            units_budget = int(leverage_budget / (price * margin_rate))
            units_budget = max(units_budget, 0)
            if units_budget <= 0:
                continue
            incremental_units = min(stage_size, units_budget)
            if incremental_units < config.MIN_STAGE_UNITS:
                continue

            side = direction
            units_to_send = incremental_units if side == "long" else -incremental_units
            sl_price, tp_price = _calc_sl_tp(side, price, sl_pips, tp_pips)
            sl_price, tp_price = clamp_sl_tp(price, sl_price, tp_price, side.upper())

            entry_meta = {
                "stage": current_stage_count + 1,
                "atr_pips": atr_pips,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "hard_stop_pips": sl_pips,
                "stage_units": stage_size,
                "ma_gap_h1": features["ma_gap_h1"],
                "ma_gap_h4": features["ma_gap_h4"],
                "adx": features["adx"],
                "range_active": range_active,
                "range_mode": None if range_ctx is None else range_ctx.mode,
                "range_reason": None if range_ctx is None else range_ctx.reason,
                "range_score": None if range_ctx is None else round(float(range_ctx.score or 0.0), 3),
                "mtf_heat_score": round(heat_decision.score, 3),
                "mtf_heat_conf_delta": round(heat_decision.confidence_delta, 2),
                "mtf_heat_lot_mult": round(heat_decision.lot_mult, 3),
                "mtf_heat_tp_mult": round(heat_decision.tp_mult, 3),
                "mtf_heat": heat_decision.debug,
            }
            LOG.info(
                "%s opening stage=%s units=%s dir=%s price=%.3f",
                config.LOG_PREFIX,
                current_stage_count + 1,
                units_to_send,
                side,
                price,
            )
            entry_thesis = {**entry_meta, "strategy_tag": STRATEGY_TAG, "env_prefix": config.ENV_PREFIX}
            proj_allow, proj_mult, proj_detail = _projection_decision(side, config.POCKET)
            if not proj_allow:
                continue
            if proj_detail:
                entry_thesis["projection"] = proj_detail
            if proj_mult > 1.0:
                sign = 1 if units_to_send > 0 else -1
                units_to_send = int(round(abs(units_to_send) * proj_mult)) * sign
            tech_decision, entry_thesis = _evaluate_entry_techniques_local(
                entry_price=price,
                side=side,
                pocket=config.POCKET,
                strategy_tag=STRATEGY_TAG,
                entry_thesis=entry_thesis,
            )
            if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
                continue
            candle_allow, candle_mult = _entry_candle_guard("long" if units_to_send > 0 else "short")
            if not candle_allow:
                continue
            if candle_mult != 1.0:
                sign = 1 if units_to_send > 0 else -1
                units_to_send = int(round(abs(units_to_send) * candle_mult)) * sign
            entry_thesis["entry_units_intent"] = abs(int(units_to_send))
            await market_order(
                "USD_JPY",
                units_to_send,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,  # type: ignore[arg-type]
                client_order_id=_client_order_id(f"{side}-stage{current_stage_count+1}"),
                reduce_only=False,
                strategy_tag=STRATEGY_TAG,
                entry_thesis=entry_thesis,
            )
            stage_cooldown_until[direction] = now_mono + config.STAGE_COOLDOWN_MINUTES * 60.0
            position_tracker["side"] = side
            position_tracker["best_pips"] = float(position_tracker.get("best_pips") or 0.0)
            position_tracker["last_stage_price"] = price
            if current_stage_count == 0:
                position_tracker["best_pips"] = 0.0
            position_tracker["stage_count"] = current_stage_count + 1

            if now_mono - last_sync_perf >= config.PERF_SYNC_INTERVAL_SEC:
                try:
                    pos_manager.sync_trades()
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("%s sync failed: %s", config.LOG_PREFIX, exc)
                last_sync_perf = now_mono
    finally:
        pos_manager.close()




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
    asyncio.run(manual_swing_worker())
