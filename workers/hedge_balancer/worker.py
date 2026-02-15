"""
Margin-driven hedge balancer.

マージン使用率が高まったときに逆方向の reduce-only を直接発注し、
ネットエクスポージャを軽くしつつ余力を回復する。
"""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection
from analysis.ma_projection import score_ma_for_side
from analysis.range_guard import detect_range_mode
from analysis.technique_engine import evaluate_entry_techniques
from indicators.factor_cache import all_factors, get_candles_snapshot

import asyncio
import datetime
import logging
import time
from typing import Dict, Optional, Tuple

from market_data import tick_window
from execution.order_ids import build_client_order_id
from execution.strategy_entry import market_order
from execution.position_manager import PositionManager
from utils.market_hours import is_market_open
from utils.oanda_account import AccountSnapshot, get_account_snapshot, get_position_summary

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

_HEDGE_ALLOWED_TAGS = {"hedgebalancer", "hedgelock"}
_HEDGE_ALLOWED_CLIENT_HINTS = ("-hedge-", "hedgebalancer", "hedgelock")


def _is_hedge_trade(entry: dict, pocket: str) -> bool:
    if pocket == "manual":
        return False
    tag = str(entry.get("strategy_tag") or "").lower()
    if any(t in tag for t in _HEDGE_ALLOWED_TAGS):
        return True
    client_id = str(entry.get("client_order_id") or entry.get("client_id") or "").lower()
    return any(hint in client_id for hint in _HEDGE_ALLOWED_CLIENT_HINTS)


def _has_foreign_trades(pos_manager: PositionManager) -> bool:
    positions = pos_manager.get_open_positions(include_unknown=True)
    if not positions:
        return False
    for pocket, info in positions.items():
        pocket_key = str(pocket or "").lower()
        # Manual trades are an explicit part of the account's exposure we must manage;
        # they should not disable hedge balancing.
        if pocket_key == "manual":
            continue
        if not isinstance(info, dict):
            continue
        trades = info.get("open_trades")
        if not trades:
            continue
        for tr in trades:
            if not _is_hedge_trade(tr, pocket):
                return True
    return False



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
    thesis_ctx.setdefault("technical_context_tfs", {"fib": ["M5", "M1"], "median": ["M5", "M1"], "nwave": ["M1", "M5"], "candle": ["M1", "M5"]})
    thesis_ctx.setdefault(
        "technical_context_ticks",
        ["latest_bid", "latest_ask", "latest_mid", "spread_pips"],
    )
    thesis_ctx.setdefault(
        "technical_context_candle_counts",
        {"M1": 120, "M5": 80, "H1": 60, "H4": 40},
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
def _latest_mid(fallback: Optional[float] = None) -> Optional[float]:
    ticks = tick_window.recent_ticks(seconds=5.0, limit=5)
    if not ticks:
        return fallback
    mids = []
    for t in ticks:
        try:
            mids.append(float(t.get("mid") or 0.0))
        except Exception:
            continue
    if mids:
        try:
            return sum(mids) / len(mids)
        except Exception:
            return fallback
    return fallback


def _margin_usage(snapshot: AccountSnapshot) -> Tuple[Optional[float], Optional[float]]:
    try:
        total_margin = float(snapshot.margin_available + snapshot.margin_used)
        if total_margin <= 0:
            return None, None
        usage = float(snapshot.margin_used) / total_margin
        return usage, total_margin
    except Exception:
        return None, None


def _plan_units(
    snapshot: AccountSnapshot,
    net_units: int,
    price: float,
) -> Tuple[Optional[int], Optional[str], Optional[float]]:
    usage, total_margin = _margin_usage(snapshot)
    if usage is None or total_margin is None:
        return None, None, usage
    margin_rate = snapshot.margin_rate
    if margin_rate <= 0:
        return None, None, usage
    margin_per_unit = price * margin_rate
    if margin_per_unit <= 0:
        return None, None, usage
    target_usage = min(config.TARGET_MARGIN_USAGE, config.TRIGGER_MARGIN_USAGE)
    target_used = total_margin * target_usage
    target_net_units = target_used / margin_per_unit

    desired_reduction = abs(net_units) - target_net_units
    if desired_reduction <= 0:
        desired_reduction = abs(net_units) * 0.25
    max_reduce = abs(net_units) * config.MAX_REDUCTION_FRACTION
    capped = min(desired_reduction, max_reduce, config.MAX_HEDGE_UNITS)
    final_units = int(max(config.MIN_HEDGE_UNITS, capped))
    if final_units <= 0:
        return None, None, usage
    reason = "margin_usage_high"
    if snapshot.free_margin_ratio is not None and snapshot.free_margin_ratio <= config.TRIGGER_FREE_MARGIN_RATIO:
        reason = "free_margin_low"
    return final_units, reason, usage


async def hedge_balancer_worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if not config.ENABLED:
        LOG.info("%s disabled via config.", config.LOG_PREFIX)
        return
    LOG.info(
        "%s start interval=%.1fs trigger_usage=%.2f target_usage=%.2f free_margin<=%.3f pocket=%s",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        config.TRIGGER_MARGIN_USAGE,
        config.TARGET_MARGIN_USAGE,
        config.TRIGGER_FREE_MARGIN_RATIO,
        config.POCKET,
    )
    last_action_ts = 0.0
    last_lock_ts = 0.0
    last_foreign_log_ts = 0.0
    pos_manager = PositionManager()

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue

        price_hint = _latest_mid()
        if price_hint is None or price_hint < config.MIN_PRICE:
            continue

        try:
            snapshot = get_account_snapshot(cache_ttl_sec=2.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s snapshot fetch failed: %s", config.LOG_PREFIX, exc)
            continue
        try:
            long_units, short_units = get_position_summary(timeout=4.0)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("%s position fetch failed: %s", config.LOG_PREFIX, exc)
            continue

        net_units = int(round(long_units - short_units))
        abs_net = abs(net_units)
        gross_units = int(round(long_units + short_units))
        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        range_ctx = None
        try:
            range_ctx = detect_range_mode(fac_m1, fac_h4)
        except Exception:
            range_ctx = None
        range_active = bool(range_ctx.active) if range_ctx is not None else False

        foreign_present = False
        if config.SKIP_FOREIGN_TRADES:
            try:
                foreign_present = _has_foreign_trades(pos_manager)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("%s foreign trade check failed: %s", config.LOG_PREFIX, exc)
                foreign_present = False
            if foreign_present and (time.monotonic() - last_foreign_log_ts) >= config.FOREIGN_TRADE_LOG_SEC:
                LOG.info("%s skip hedge: foreign open trades detected", config.LOG_PREFIX)
                last_foreign_log_ts = time.monotonic()

        if foreign_present:
            continue

        # Hedge lock unwind: both-side positions with small net exposure
        if (
            config.LOCK_ENABLED
            and gross_units >= config.LOCK_GROSS_MIN_UNITS
            and abs_net <= config.LOCK_NET_MAX_UNITS
            and long_units >= config.LOCK_MIN_SIDE_UNITS
            and short_units >= config.LOCK_MIN_SIDE_UNITS
            and (time.monotonic() - last_lock_ts) >= config.LOCK_COOLDOWN_SEC
        ):
            lock_mode = "range" if range_active else "scalp"
            allow_long, _, detail_long = _projection_decision("long", "scalp", mode_override=lock_mode)
            allow_short, _, detail_short = _projection_decision("short", "scalp", mode_override=lock_mode)
            score_long = float((detail_long or {}).get("score", 0.0))
            score_short = float((detail_short or {}).get("score", 0.0))
            score_gap = score_long - score_short
            min_score = config.LOCK_SCORE_MIN
            min_gap = config.LOCK_SCORE_GAP
            if range_active and not config.LOCK_ALLOW_RANGE:
                min_score += 0.2
                min_gap += 0.2
            direction = None
            if allow_long and score_gap >= min_gap and score_long >= min_score:
                direction = "long"
            elif allow_short and (-score_gap) >= min_gap and score_short >= min_score:
                direction = "short"

            if direction:
                reduce_side_units = short_units if direction == "long" else long_units
                planned = min(
                    float(reduce_side_units) * config.LOCK_MAX_REDUCTION_FRACTION,
                    config.LOCK_MAX_UNITS,
                )
                reduce_units = int(max(config.LOCK_MIN_UNITS, planned))
                reduce_units = int(min(reduce_units, reduce_side_units))
                if reduce_units > 0:
                    units = abs(reduce_units) if direction == "long" else -abs(reduce_units)
                    if not _bb_entry_allowed(BB_STYLE, "long" if units > 0 else "short", price_hint, fac_m1, range_active=range_active):
                        direction = None
                    else:
                        candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
                        if candle_allow and candle_mult != 1.0:
                            sign = 1 if units > 0 else -1
                            units = int(round(abs(units) * candle_mult)) * sign
                        if candle_allow:
                            entry_thesis = {
                                "strategy_tag": "HedgeLock",
                                "env_prefix": config.ENV_PREFIX,
                                "reduce_only": True,
                                "hedge_lock": True,
                                "range_active": range_active,
                                "score_long": round(score_long, 3),
                                "score_short": round(score_short, 3),
                                "score_gap": round(score_gap, 3),
                                "direction": direction,
                            }
                            tech_decision, entry_thesis = _evaluate_entry_techniques_local(
                                entry_price=float(price_hint),
                                side="long" if units > 0 else "short",
                                pocket=config.POCKET,
                                strategy_tag="HedgeLock",
                                entry_thesis=entry_thesis,
                            )
                            if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
                                continue
                            client_id = build_client_order_id("hedge", "HedgeLock")
                            res = await market_order(
                                instrument="USD_JPY",
                                units=units,
                                sl_price=None,
                                tp_price=None,
                                pocket=config.POCKET,
                                client_order_id=client_id,
                                strategy_tag="HedgeLock",
                                reduce_only=True,
                                entry_thesis=entry_thesis,
                                confidence=config.CONFIDENCE,
                                arbiter_final=True,
                            )
                            last_lock_ts = time.monotonic()
                            LOG.info(
                                "%s hedge_lock dir=%s units=%d gross=%d net=%d range=%s scoreL=%.2f scoreS=%.2f res=%s",
                                config.LOG_PREFIX,
                                direction,
                                units,
                                gross_units,
                                net_units,
                                range_active,
                                score_long,
                                score_short,
                                res or "none",
                            )
                            continue

        if abs_net < config.MIN_NET_UNITS:
            continue

        hedge_units, reason, usage = _plan_units(snapshot, net_units, price_hint)
        if hedge_units is None or hedge_units <= 0:
            continue

        trigger_usage = usage is not None and usage >= config.TRIGGER_MARGIN_USAGE
        trigger_free = (
            snapshot.free_margin_ratio is not None
            and snapshot.free_margin_ratio <= config.TRIGGER_FREE_MARGIN_RATIO
        )
        if not (trigger_usage or trigger_free):
            continue

        if time.monotonic() - last_action_ts < config.COOLDOWN_SEC:
            continue

        direction = "OPEN_SHORT" if net_units > 0 else "OPEN_LONG"
        proposed_units = min(hedge_units, abs_net)
        units = -abs(proposed_units) if net_units > 0 else abs(proposed_units)
        fac_m1 = fac_m1 or {}
        if not _bb_entry_allowed(BB_STYLE, "long" if units > 0 else "short", price_hint, fac_m1):
            continue
        client_id = build_client_order_id("hedge", "HedgeBalancer")
        entry_thesis = {
            "strategy_tag": "HedgeBalancer",
            "env_prefix": config.ENV_PREFIX,
            "reduce_only": True,
            "margin_usage": usage,
            "free_margin_ratio": snapshot.free_margin_ratio,
            "target_usage": config.TARGET_MARGIN_USAGE,
            "reason": reason,
        }
        tech_decision, entry_thesis = _evaluate_entry_techniques_local(
            entry_price=float(price_hint),
            side="long" if units > 0 else "short",
            pocket=config.POCKET,
            strategy_tag="HedgeBalancer",
            entry_thesis=entry_thesis,
        )
        if not tech_decision.allowed and not getattr(config, "TECH_FAILOPEN", True):
            continue
        candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
        if not candle_allow:
            continue
        if candle_mult != 1.0:
            sign = 1 if units > 0 else -1
            units = int(round(abs(units) * candle_mult)) * sign
        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=None,
            tp_price=None,
            pocket=config.POCKET,
            client_order_id=client_id,
            strategy_tag="HedgeBalancer",
            reduce_only=True,
            entry_thesis=entry_thesis,
            confidence=config.CONFIDENCE,
            arbiter_final=True,
        )
        last_action_ts = time.monotonic()
        LOG.info(
            "%s hedge dir=%s units=%d net=%d usage=%.3f free=%.3f reason=%s res=%s",
            config.LOG_PREFIX,
            direction,
            units,
            net_units,
            usage if usage is not None else -1.0,
            snapshot.free_margin_ratio if snapshot.free_margin_ratio is not None else -1.0,
            reason or "unknown",
            res or "none",
        )




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

if __name__ == "__main__":
    asyncio.run(hedge_balancer_worker())
