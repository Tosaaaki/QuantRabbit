"""H1 trend-following worker leveraging the MovingAverageCross strategy."""

from __future__ import annotations
from analysis.ma_projection import compute_adx_projection, compute_bbw_projection, compute_ma_projection, compute_rsi_projection

import asyncio
import datetime
import hashlib
import logging
import os
import time
from typing import Dict, Optional

from indicators.factor_cache import all_factors, get_candles_snapshot
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    loss_cooldown_status,
)
from market_data import spread_monitor, tick_window
from strategies.trend.ma_cross import MovingAverageCross
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
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

BB_STYLE = "trend"

LOG = logging.getLogger(__name__)
PIP = 0.01



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
def _parse_iso8601(value: str) -> Optional[datetime.datetime]:
    """Decode ISO8601 timestamps produced by factor cache."""
    try:
        dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _latest_mid(fallback: float) -> float:
    """Pick the most recent mid price from tick cache, fallback to provided value."""
    ticks = tick_window.recent_ticks(seconds=20.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_val = tick.get("mid")
        if mid_val is not None:
            try:
                return float(mid_val)
            except (TypeError, ValueError):
                pass
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except (TypeError, ValueError):
                return fallback
    return fallback


def _confidence_scale(confidence: int) -> float:
    """Map confidence score (0-100) into a lot multiplier."""
    floor = config.CONFIDENCE_FLOOR
    ceil = config.CONFIDENCE_CEIL
    if ceil <= floor:
        return config.MAX_CONFIDENCE_SCALE
    if confidence <= floor:
        return config.MIN_CONFIDENCE_SCALE
    if confidence >= ceil:
        return config.MAX_CONFIDENCE_SCALE
    ratio = (confidence - floor) / float(ceil - floor)
    span = config.MAX_CONFIDENCE_SCALE - config.MIN_CONFIDENCE_SCALE
    return config.MIN_CONFIDENCE_SCALE + max(0.0, min(1.0, ratio)) * span


def _client_order_id(tag: str) -> str:
    """Generate a quasi-unique client order id that follows QuantRabbit conventions."""
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "trend"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-macro-{sanitized}{digest}"


def _log_skip(reason: str, state: Dict[str, float]) -> None:
    """Throttle skip logging so the worker does not spam the logs."""
    if not config.LOG_SKIP_REASON:
        return
    now = time.monotonic()
    last_reason = state.get("reason", "")
    last_ts = state.get("ts", 0.0)
    if reason == last_reason and now - last_ts < 60.0:
        return
    LOG.debug("%s skip: %s", config.LOG_PREFIX, reason)
    state["reason"] = reason
    state["ts"] = now


def _direction_allowed(
    fac_h1: Dict[str, float],
    fac_h4: Optional[Dict[str, float]],
    direction: str,
    atr_pips: float,
) -> bool:
    """Require higher timeframe alignment before taking the trade."""
    if not fac_h4:
        return True
    ma10_h4 = fac_h4.get("ma10")
    ma20_h4 = fac_h4.get("ma20")
    if ma10_h4 is None or ma20_h4 is None:
        return True
    gap_h4 = float(ma10_h4) - float(ma20_h4)
    bias_buffer = 0.00018
    override_allowed = False
    if direction == "long" and gap_h4 < -bias_buffer:
        ma10_h1 = fac_h1.get("ma10")
        ma20_h1 = fac_h1.get("ma20")
        if ma10_h1 is not None and ma20_h1 is not None:
            gap_h1 = (float(ma10_h1) - float(ma20_h1)) / PIP
            if (
                gap_h1 >= config.H1_OVERRIDE_GAP_PIPS
                and atr_pips >= config.H1_OVERRIDE_ATR_PIPS
            ):
                override_allowed = True
        if not override_allowed:
            return False
    if direction == "short" and gap_h4 > bias_buffer:
        ma10_h1 = fac_h1.get("ma10")
        ma20_h1 = fac_h1.get("ma20")
        if ma10_h1 is not None and ma20_h1 is not None:
            gap_h1 = (float(ma10_h1) - float(ma20_h1)) / PIP
            if (
                gap_h1 <= -config.H1_OVERRIDE_GAP_PIPS
                and atr_pips >= config.H1_OVERRIDE_ATR_PIPS
            ):
                override_allowed = True
        if not override_allowed:
            return False
    ma10_h1 = fac_h1.get("ma10")
    ma20_h1 = fac_h1.get("ma20")
    if ma10_h1 is None or ma20_h1 is None:
        return True
    gap_h1 = float(ma10_h1) - float(ma20_h1)
    micro_buffer = 0.00006
    if direction == "long" and gap_h1 < -micro_buffer:
        return False
    if direction == "short" and gap_h1 > micro_buffer:
        return False
    return True


async def trend_h1_worker() -> None:
    """Async loop that scans H1 factors and places macro-pocket trend trades."""
    if not config.ENABLED:
        LOG.info("%s disabled via configuration", config.LOG_PREFIX)
        return

    LOG.info(
        "%s worker starting (interval=%.1fs)",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
    )
    pos_manager = PositionManager()
    cooldown_until = 0.0
    last_direction_entry: Dict[str, float] = {"long": 0.0, "short": 0.0}
    recent_signal_gate: Dict[str, float] = {}
    skip_state: Dict[str, float] = {"reason": "", "ts": 0.0}

    try:
        while True:
            try:
                await asyncio.sleep(config.LOOP_INTERVAL_SEC)
                now_mono = time.monotonic()
                if now_mono < cooldown_until:
                    continue

                now_utc = datetime.datetime.utcnow()
                if not is_market_open(now_utc):
                    _log_skip("market_closed", skip_state)
                    continue

                if not can_trade(config.POCKET):
                    _log_skip("pocket_drawdown_guard", skip_state)
                    continue

                loss_block, loss_remain = loss_cooldown_status(
                    config.POCKET,
                    max_losses=config.LOSS_STREAK_MAX,
                    cooldown_minutes=config.LOSS_STREAK_COOLDOWN_MIN,
                )
                if loss_block:
                    _log_skip(f"loss_cooldown {loss_remain:.0f}s", skip_state)
                    continue

                blocked, _, spread_state, spread_reason = spread_monitor.is_blocked()
                spread_pips = float((spread_state or {}).get("spread_pips") or 0.0)
                if blocked:
                    _log_skip(f"spread_block {spread_reason or ''}".strip(), skip_state)
                    continue
                if config.SPREAD_MAX_PIPS > 0.0 and spread_pips > config.SPREAD_MAX_PIPS:
                    _log_skip(f"spread {spread_pips:.2f}p>limit", skip_state)
                    continue

                factors = all_factors()
                fac_h1 = factors.get("H1")
                fac_h4 = factors.get("H4")
                if not fac_h1:
                    _log_skip("missing_h1_factors", skip_state)
                    continue

                candles = fac_h1.get("candles")
                if not isinstance(candles, list) or len(candles) < config.MIN_CANDLES:
                    _log_skip("insufficient_candles", skip_state)
                    continue

                timestamp_raw = fac_h1.get("timestamp")
                if isinstance(timestamp_raw, str):
                    ts = _parse_iso8601(timestamp_raw)
                    if ts:
                        age = (
                            now_utc.replace(tzinfo=datetime.timezone.utc) - ts
                        ).total_seconds()
                        if config.DATA_STALE_SECONDS > 0 and age > config.DATA_STALE_SECONDS:
                            _log_skip(f"stale_data age={age:.0f}s", skip_state)
                            continue

                fac_signal = dict(fac_h1)
                atr_val = fac_signal.get("atr_pips")
                if atr_val is None:
                    atr_raw = fac_signal.get("atr")
                    if isinstance(atr_raw, (int, float)):
                        atr_val = float(atr_raw) * 100.0
                        fac_signal["atr_pips"] = atr_val
                try:
                    atr_pips = float(atr_val) if atr_val is not None else None
                except (TypeError, ValueError):
                    atr_pips = None
                if atr_pips is None:
                    _log_skip("atr_missing", skip_state)
                    continue
                if atr_pips < config.MIN_ATR_PIPS:
                    _log_skip(f"atr_low {atr_pips:.1f}p", skip_state)
                    continue
                if atr_pips > config.MAX_ATR_PIPS:
                    _log_skip(f"atr_high {atr_pips:.1f}p", skip_state)
                    continue

                close_price = fac_signal.get("close")
                vwap = fac_signal.get("vwap")
                adx_h1 = float(fac_signal.get("adx") or 0.0)
                bbw_h1 = float(fac_signal.get("bbw") or 0.0)
                try:
                    close_f = float(close_price) if close_price is not None else 0.0
                except (TypeError, ValueError):
                    close_f = 0.0
                if bbw_h1 > 0.0 and bbw_h1 < 0.0010 and adx_h1 < 16.0:
                    _log_skip(f"range_guard adx={adx_h1:.1f} bbw={bbw_h1:.4f}", skip_state)
                    continue
                if vwap is not None and close_price is not None:
                    try:
                        vwap_gap = abs(close_f - float(vwap)) / PIP
                        if vwap_gap < 1.0:
                            _log_skip(f"vwap_gap {vwap_gap:.2f}p", skip_state)
                            continue
                    except Exception:
                        pass

                regime = current_regime("H1")
                if config.REQUIRE_REGIME and regime and regime not in config.REQUIRE_REGIME:
                    _log_skip(f"regime_guard {regime}", skip_state)
                    continue
                if config.BLOCK_REGIME and regime and regime in config.BLOCK_REGIME:
                    _log_skip(f"blocked_regime {regime}", skip_state)
                    continue

                decision = MovingAverageCross.check(fac_signal)
                if not decision:
                    _log_skip("no_signal", skip_state)
                    continue

                confidence = int(decision.get("confidence", 0))
                if confidence < config.MIN_CONFIDENCE:
                    _log_skip(f"confidence_low {confidence}", skip_state)
                    continue

                action = decision.get("action")
                if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                    _log_skip(f"unsupported_action {action}", skip_state)
                    continue

                direction = "long" if action == "OPEN_LONG" else "short"
                if (
                    config.ALLOWED_DIRECTIONS
                    and direction.title() not in config.ALLOWED_DIRECTIONS
                ):
                    _log_skip(f"direction_block {direction}", skip_state)
                    continue
                meta = decision.get("_meta") or {}
                fast_gap = float(meta.get("price_to_fast_pips", 0.0) or 0.0)
                if direction == "long" and fast_gap < config.MIN_FAST_GAP_PIPS:
                    _log_skip("fast_gap_insufficient", skip_state)
                    continue
                if direction == "short" and fast_gap > -config.MIN_FAST_GAP_PIPS:
                    _log_skip("fast_gap_insufficient", skip_state)
                    continue
                if not _direction_allowed(fac_signal, fac_h4, direction, atr_pips):
                    _log_skip("direction_mismatch", skip_state)
                    continue

                sig_key = f"{decision.get('tag')}:{direction}"
                prev_ts = recent_signal_gate.get(sig_key, 0.0)
                if config.REPEAT_BLOCK_SEC > 0 and now_mono - prev_ts < config.REPEAT_BLOCK_SEC:
                    _log_skip("repeat_signal_block", skip_state)
                    continue

                open_positions = pos_manager.get_open_positions()
                macro_positions = open_positions.get(config.POCKET, {})
                trades = macro_positions.get("open_trades", [])
                if len(trades) >= config.MAX_ACTIVE_TRADES:
                    _log_skip("max_active_trades", skip_state)
                    continue
                directional_trades = [
                    tr for tr in trades if (tr.get("side") or "").lower() == direction
                ]
                if len(directional_trades) >= config.MAX_DIRECTIONAL_TRADES:
                    _log_skip("directional_trade_cap", skip_state)
                    continue
                directional_units = sum(
                    abs(int(tr.get("units", 0) or 0)) for tr in directional_trades
                )
                if directional_units >= config.MAX_DIRECTIONAL_UNITS:
                    _log_skip("directional_units_cap", skip_state)
                    continue
                stage_idx = len(directional_trades)
                if stage_idx >= len(config.STAGE_RATIOS):
                    _log_skip("stage_limit", skip_state)
                    continue

                last_entry_ts = last_direction_entry.get(direction, 0.0)
                if now_mono - last_entry_ts < config.REENTRY_COOLDOWN_SEC:
                    _log_skip("directional_cooldown", skip_state)
                    continue

                try:
                    snapshot = get_account_snapshot()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("%s account snapshot failed: %s", config.LOG_PREFIX, exc)
                    _log_skip("account_snapshot_failed", skip_state)
                    continue

                equity = float(getattr(snapshot, "nav", 0.0) or getattr(snapshot, "balance", 0.0))
                if equity <= 0.0:
                    _log_skip("equity_zero", skip_state)
                    continue

                try:
                    sl_pips = float(decision.get("sl_pips") or 0.0)
                    tp_pips = float(decision.get("tp_pips") or 0.0)
                except (TypeError, ValueError):
                    _log_skip("invalid_sl_tp", skip_state)
                    continue
                if sl_pips <= 0.0 or tp_pips <= 0.0:
                    _log_skip("invalid_sl_tp", skip_state)
                    continue

                price_hint = float(fac_signal.get("close") or 0.0)
                entry_price = _latest_mid(price_hint)
                if not _bb_entry_allowed(BB_STYLE, direction, entry_price, fac_h1):
                    _log_skip("bb_entry_block", skip_state)
                    continue

                strategy_tag = decision.get("tag") or "trend_h1"
                lot = allowed_lot(
                    equity,
                    sl_pips,
                    margin_available=getattr(snapshot, "margin_available", None),
                    price=entry_price,
                    margin_rate=getattr(snapshot, "margin_rate", None),
                    risk_pct_override=config.RISK_PCT,
                    pocket=config.POCKET,
                    side=direction,
                    strategy_tag=strategy_tag,
                    fac_m1=fac_signal,
                    fac_h4=fac_h4,
                )
                if lot <= 0.0:
                    _log_skip("lot_zero", skip_state)
                    continue

                # 市況に応じた lot/TP の動的調整
                if atr_pips < 6.0:
                    lot *= 0.92
                    tp_pips = max(6.0, tp_pips * 0.9)
                elif atr_pips > 22.0:
                    lot *= 1.06
                    tp_pips = min(36.0, tp_pips * 1.05)
                try:
                    vwap_gap = abs(entry_price - float(vwap)) / PIP if vwap is not None else None
                except Exception:
                    vwap_gap = None
                if vwap_gap is not None:
                    if vwap_gap >= 2.0:
                        tp_pips = min(38.0, tp_pips + 1.0)
                    elif vwap_gap <= 1.0:
                        tp_pips = max(5.0, tp_pips * 0.9)

                lot *= _confidence_scale(confidence)
                lot = max(config.MIN_LOT, min(config.MAX_LOT, lot))
                stage_ratio = config.STAGE_RATIOS[stage_idx]
                lot *= max(0.01, stage_ratio)
                units = int(round(lot * 100000))
                if units < config.MIN_UNITS:
                    _log_skip("units_below_min", skip_state)
                    continue
                if direction == "short":
                    units = -units

                sl_price = (
                    entry_price - sl_pips * PIP if direction == "long" else entry_price + sl_pips * PIP
                )
                tp_price = (
                    entry_price + tp_pips * PIP if direction == "long" else entry_price - tp_pips * PIP
                )
                sl_price, tp_price = clamp_sl_tp(entry_price, sl_price, tp_price, direction == "long")

                client_id = _client_order_id(strategy_tag or "trend")
                thesis = {
                    "strategy_tag": strategy_tag,
                    "confidence": confidence,
                    "atr_pips": atr_pips,
                    "regime": regime,
                    "stage": stage_idx + 1,
                    "tp_pips": tp_pips,
                    "sl_pips": sl_pips,
                    "hard_stop_pips": sl_pips,
                }
                entry_meta = decision.get("_meta") or {}
                entry_meta["stage_index"] = stage_idx

                proj_allow, proj_mult, proj_detail = _projection_decision(direction, config.POCKET)
                if not proj_allow:
                    _log_skip("projection_block", skip_state)
                    continue
                if proj_detail:
                    thesis["projection"] = proj_detail
                if proj_mult > 1.0:
                    sign = 1 if units > 0 else -1
                    units = int(round(abs(units) * proj_mult)) * sign

                LOG.info(
                    "%s signal=%s dir=%s conf=%d lot=%.4f units=%d sl=%.2fp tp=%.2fp price=%.3f atr=%.1fp",
                    config.LOG_PREFIX,
                    decision.get("tag"),
                    direction,
                    confidence,
                    lot,
                    units,
                    sl_pips,
                    tp_pips,
                    entry_price,
                    atr_pips,
                )

                candle_allow, candle_mult = _entry_candle_guard("long" if units > 0 else "short")
                if not candle_allow:
                    continue
                if candle_mult != 1.0:
                    sign = 1 if units > 0 else -1
                    units = int(round(abs(units) * candle_mult)) * sign
                ticket_id = await market_order(
                    "USD_JPY",
                    units,
                    sl_price,
                    tp_price,
                    config.POCKET,
                    client_order_id=client_id,
                    entry_thesis=thesis,
                    meta=entry_meta,
                )
                if ticket_id:
                    pos_manager.register_open_trade(ticket_id, config.POCKET, client_id)
                    last_direction_entry[direction] = now_mono
                    recent_signal_gate[sig_key] = now_mono
                    cooldown_until = now_mono + config.ENTRY_COOLDOWN_SEC
                    skip_state["reason"] = ""
                    skip_state["ts"] = 0.0
                    LOG.info(
                        "%s order filled ticket=%s units=%d sl=%.3f tp=%.3f",
                        config.LOG_PREFIX,
                        ticket_id,
                        units,
                        sl_price,
                        tp_price,
                    )
                else:
                    cooldown_until = now_mono + min(config.ENTRY_COOLDOWN_SEC, 60.0)
                    _log_skip("order_failed", skip_state)
            except asyncio.CancelledError:
                LOG.info("%s worker cancelled", config.LOG_PREFIX)
                raise
            except Exception as exc:  # noqa: BLE001
                LOG.exception("%s loop error: %s", config.LOG_PREFIX, exc)
                cooldown_until = time.monotonic() + max(config.LOOP_INTERVAL_SEC, 15.0)
                continue
    except asyncio.CancelledError:
        LOG.info("%s worker cancelled", config.LOG_PREFIX)
        raise
    except Exception as exc:  # noqa: BLE001
        LOG.exception("%s worker crashed: %s", config.LOG_PREFIX, exc)
    finally:
        pos_manager.close()
        LOG.info("%s worker stopped", config.LOG_PREFIX)




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
    asyncio.run(trend_h1_worker())
