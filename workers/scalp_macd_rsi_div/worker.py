"""Entry-only worker for MACD divergence + RSI exhaustion reclaim on M1."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional

from analysis.range_guard import detect_range_mode
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common import perf_guard
from workers.common.dyn_cap import compute_cap
from workers.common.dynamic_alloc import load_strategy_profile
from workers.common.size_utils import scale_base_units

from . import config

LOG = logging.getLogger(__name__)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
    if ticks:
        tick = ticks[-1]
        mid_val = tick.get("mid")
        if mid_val is not None:
            try:
                return float(mid_val)
            except Exception:
                pass
        bid = tick.get("bid")
        ask = tick.get("ask")
        if bid is not None and ask is not None:
            try:
                return (float(bid) + float(ask)) / 2.0
            except Exception:
                return fallback
    return fallback


def _latest_spread_pips(fallback: float = 0.0) -> float:
    ticks = tick_window.recent_ticks(seconds=8.0, limit=1)
    if ticks:
        tick = ticks[-1]
        try:
            bid = float(tick.get("bid") or 0.0)
            ask = float(tick.get("ask") or 0.0)
            if bid > 0.0 and ask > 0.0 and ask >= bid:
                return (ask - bid) / config.PIP_VALUE
        except Exception:
            pass
    return max(0.0, fallback)


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:12] or "macdrsidiv"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:8]
    return f"qr-{ts_ms}-scalp-{sanitized}{digest}"


def _confidence_scale(confidence: int) -> float:
    lo = float(config.CONFIDENCE_FLOOR)
    hi = float(config.CONFIDENCE_CEIL)
    if confidence <= lo:
        return 0.55
    if confidence >= hi:
        return 1.0
    span = (float(confidence) - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _to_probability(value: object) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


def _compute_cap(*, atr_pips: float, free_ratio: float, range_active: bool, perf_pf: Optional[float]) -> tuple[float, Dict[str, float]]:
    res = compute_cap(
        atr_pips=atr_pips,
        free_ratio=free_ratio,
        range_active=range_active,
        perf_pf=perf_pf,
        pos_bias=0.0,
        cap_min=config.CAP_MIN,
        cap_max=config.CAP_MAX,
        env_prefix=config.ENV_PREFIX,
    )
    return res.cap, res.reasons


def _signal_side(
    *,
    prev_rsi: Optional[float],
    rsi: float,
    long_armed: bool,
    short_armed: bool,
    div_kind: int,
    div_score: float,
    div_strength: float,
    div_age_bars: float,
) -> tuple[Optional[str], Dict[str, float]]:
    meta = {
        "div_kind": float(div_kind),
        "div_score": float(div_score),
        "div_strength": float(div_strength),
        "div_age_bars": float(div_age_bars),
    }
    if prev_rsi is None:
        return None, meta
    if div_age_bars > float(config.MAX_DIV_AGE_BARS):
        return None, meta
    if abs(div_score) < float(config.MIN_DIV_SCORE):
        return None, meta
    if float(div_strength) < float(config.MIN_DIV_STRENGTH):
        return None, meta

    long_kinds = {1, 2} if config.ALLOW_HIDDEN_DIVERGENCE else {1}
    short_kinds = {-1, -2} if config.ALLOW_HIDDEN_DIVERGENCE else {-1}

    cross_up = prev_rsi <= config.RSI_LONG_ENTRY and rsi > config.RSI_LONG_ENTRY
    cross_down = prev_rsi >= config.RSI_SHORT_ENTRY and rsi < config.RSI_SHORT_ENTRY
    if long_armed and cross_up and div_kind in long_kinds and div_score > 0:
        return "long", meta
    if short_armed and cross_down and div_kind in short_kinds and div_score < 0:
        return "short", meta
    return None, meta


def _compute_confidence(*, side: str, rsi: float, div_score: float, div_strength: float, range_score: float) -> int:
    div_term = min(20.0, abs(div_score) * 40.0)
    strength_term = min(12.0, max(0.0, div_strength) * 18.0)
    range_term = min(9.0, max(0.0, range_score) * 12.0)
    heat_term = min(8.0, abs(rsi - 50.0) * 0.20)

    directional_term = 0.0
    if side == "long" and rsi <= 45.0:
        directional_term = min(6.0, (45.0 - rsi) * 0.35)
    elif side == "short" and rsi >= 55.0:
        directional_term = min(6.0, (rsi - 55.0) * 0.35)

    confidence = 52.0 + div_term + strength_term + range_term + heat_term + directional_term
    confidence = max(float(config.CONFIDENCE_FLOOR), min(float(config.CONFIDENCE_CEIL), confidence))
    return int(round(confidence))


def _compute_targets(atr_pips: float) -> tuple[float, float]:
    atr = max(0.1, float(atr_pips))
    sl_pips = atr * config.SL_ATR_MULT
    tp_pips = atr * config.TP_ATR_MULT
    sl_pips = max(config.MIN_SL_PIPS, min(config.MAX_SL_PIPS, sl_pips))
    tp_pips = max(config.MIN_TP_PIPS, min(config.MAX_TP_PIPS, tp_pips))
    tp_pips = max(tp_pips, sl_pips * config.MIN_TP_RR)
    return tp_pips, sl_pips


def _passes_open_trades_guard(pos_manager: PositionManager, strategy_tag: str) -> bool:
    if config.MAX_OPEN_TRADES <= 0 and config.MAX_OPEN_TRADES_GLOBAL <= 0:
        return True
    try:
        positions = pos_manager.get_open_positions()
    except Exception:
        return True
    pocket_info = positions.get(config.POCKET) or {}
    open_trades_all = pocket_info.get("open_trades") or []
    if config.MAX_OPEN_TRADES_GLOBAL > 0 and len(open_trades_all) >= config.MAX_OPEN_TRADES_GLOBAL:
        return False
    open_trades = open_trades_all
    if config.OPEN_TRADES_SCOPE == "tag":
        tag_lower = str(strategy_tag or "").strip().lower()
        open_trades = [
            tr
            for tr in open_trades_all
            if str(tr.get("strategy_tag") or "").strip().lower() == tag_lower
        ]
    if config.MAX_OPEN_TRADES > 0 and len(open_trades) >= config.MAX_OPEN_TRADES:
        return False
    return True


async def scalp_macd_rsi_div_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            return

    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    pos_manager = PositionManager()

    last_entry_mono = 0.0
    prev_rsi: Optional[float] = None
    long_arm_until = 0.0
    short_arm_until = 0.0
    last_block_log_mono = 0.0
    last_spread_log_mono = 0.0
    last_gate_log_mono = 0.0
    last_margin_log_mono = 0.0

    try:
        while True:
            await asyncio.sleep(max(0.2, config.LOOP_INTERVAL_SEC))

            now_utc = datetime.datetime.utcnow()
            now_mono = time.monotonic()
            if not is_market_open(now_utc):
                continue
            if not can_trade(config.POCKET):
                continue

            blocked, remain, spread_state, spread_reason = spread_monitor.is_blocked()
            spread_pips = _safe_float((spread_state or {}).get("spread_pips"), 0.0)
            if spread_pips <= 0.0:
                spread_pips = _latest_spread_pips(spread_pips)
            if blocked or spread_pips > config.MAX_SPREAD_PIPS:
                if now_mono - last_spread_log_mono > 60.0:
                    LOG.info(
                        "%s spread_block spread=%.2f stale=%s remain=%ss reason=%s",
                        config.LOG_PREFIX,
                        spread_pips,
                        bool((spread_state or {}).get("stale")),
                        remain,
                        spread_reason or "guard",
                    )
                    last_spread_log_mono = now_mono
                continue

            factors = all_factors()
            fac_m1 = factors.get("M1") or {}
            fac_h4 = factors.get("H4") or {}
            if not fac_m1:
                continue

            rsi = _safe_float(fac_m1.get("rsi"), 50.0)
            if rsi <= config.RSI_LONG_ARM:
                long_arm_until = now_mono + max(30.0, config.RSI_ARM_TTL_SEC)
            if rsi >= config.RSI_SHORT_ARM:
                short_arm_until = now_mono + max(30.0, config.RSI_ARM_TTL_SEC)

            if config.COOLDOWN_SEC > 0.0 and (now_mono - last_entry_mono) < config.COOLDOWN_SEC:
                prev_rsi = rsi
                continue

            range_ctx = detect_range_mode(fac_m1, fac_h4)
            range_score = _safe_float(range_ctx.score, 0.0)
            range_active = bool(range_ctx.active)
            if config.REQUIRE_RANGE_ACTIVE and not range_active:
                if now_mono - last_gate_log_mono > 60.0:
                    LOG.info(
                        "%s gate_block range_active_required range_active=%s score=%.3f min=%.3f",
                        config.LOG_PREFIX,
                        range_active,
                        range_score,
                        config.RANGE_MIN_SCORE,
                    )
                    last_gate_log_mono = now_mono
                prev_rsi = rsi
                continue
            if range_score < config.RANGE_MIN_SCORE:
                if now_mono - last_gate_log_mono > 60.0:
                    LOG.info(
                        "%s gate_block range_score score=%.3f min=%.3f mode=%s",
                        config.LOG_PREFIX,
                        range_score,
                        config.RANGE_MIN_SCORE,
                        str(range_ctx.mode or "-"),
                    )
                    last_gate_log_mono = now_mono
                prev_rsi = rsi
                continue

            adx = _safe_float(fac_m1.get("adx"), 0.0)
            if adx > config.MAX_ADX:
                if now_mono - last_gate_log_mono > 60.0:
                    LOG.info(
                        "%s gate_block adx adx=%.2f max=%.2f",
                        config.LOG_PREFIX,
                        adx,
                        config.MAX_ADX,
                    )
                    last_gate_log_mono = now_mono
                prev_rsi = rsi
                continue

            if not _passes_open_trades_guard(pos_manager, config.STRATEGY_TAG):
                prev_rsi = rsi
                continue

            div_kind = int(_safe_float(fac_m1.get("div_macd_kind"), 0.0))
            div_score = _safe_float(fac_m1.get("div_macd_score"), 0.0)
            div_strength = _safe_float(fac_m1.get("div_macd_strength"), 0.0)
            div_age_bars = _safe_float(fac_m1.get("div_macd_age"), 99.0)
            side, div_meta = _signal_side(
                prev_rsi=prev_rsi,
                rsi=rsi,
                long_armed=long_arm_until > now_mono,
                short_armed=short_arm_until > now_mono,
                div_kind=div_kind,
                div_score=div_score,
                div_strength=div_strength,
                div_age_bars=div_age_bars,
            )
            if not side:
                if now_mono - last_gate_log_mono > 60.0:
                    LOG.info(
                        "%s gate_wait signal prev_rsi=%.2f rsi=%.2f long_armed=%s short_armed=%s div_kind=%s div_score=%.3f div_strength=%.3f div_age=%.1f",
                        config.LOG_PREFIX,
                        prev_rsi if prev_rsi is not None else -1.0,
                        rsi,
                        long_arm_until > now_mono,
                        short_arm_until > now_mono,
                        div_kind,
                        div_score,
                        div_strength,
                        div_age_bars,
                    )
                    last_gate_log_mono = now_mono
                prev_rsi = rsi
                continue

            perf = perf_guard.is_allowed(
                config.STRATEGY_TAG,
                config.POCKET,
                hour=now_utc.hour,
                side=side,
                env_prefix=config.ENV_PREFIX,
            )
            if not perf.allowed:
                if now_mono - last_block_log_mono > 60.0:
                    LOG.info(
                        "%s perf_block reason=%s sample=%s",
                        config.LOG_PREFIX,
                        perf.reason,
                        perf.sample,
                    )
                    last_block_log_mono = now_mono
                prev_rsi = rsi
                continue

            snap = get_account_snapshot()
            nav = _safe_float(snap.nav or snap.balance, 0.0)
            balance = _safe_float(snap.balance or snap.nav, 0.0)
            margin_available = _safe_float(snap.margin_available, 0.0)
            margin_rate = _safe_float(snap.margin_rate, 0.0)
            free_ratio = _safe_float(snap.free_margin_ratio, 0.0)
            margin_used = _safe_float(snap.margin_used, 0.0)
            margin_usage = margin_used / nav if nav > 0 else 0.0
            if free_ratio <= config.MIN_FREE_MARGIN_RATIO_HARD or margin_usage >= config.MARGIN_USAGE_HARD:
                if now_mono - last_margin_log_mono > 60.0:
                    LOG.info(
                        "%s gate_block margin free_ratio=%.4f min=%.4f usage=%.4f max=%.4f",
                        config.LOG_PREFIX,
                        free_ratio,
                        config.MIN_FREE_MARGIN_RATIO_HARD,
                        margin_usage,
                        config.MARGIN_USAGE_HARD,
                    )
                    last_margin_log_mono = now_mono
                prev_rsi = rsi
                continue

            atr_pips = _safe_float(fac_m1.get("atr_pips"), 0.0)
            if atr_pips <= 0.0:
                atr_pips = max(0.8, spread_pips * 2.0)
            cap, cap_reason = _compute_cap(
                atr_pips=atr_pips,
                free_ratio=free_ratio,
                range_active=range_active,
                perf_pf=None,
            )
            if cap <= 0.0:
                prev_rsi = rsi
                continue

            price = _latest_mid(_safe_float(fac_m1.get("close"), 0.0))
            if price <= 0.0:
                prev_rsi = rsi
                continue

            confidence = _compute_confidence(
                side=side,
                rsi=rsi,
                div_score=div_score,
                div_strength=div_strength,
                range_score=range_score,
            )
            tp_pips, sl_pips = _compute_targets(atr_pips)

            base_units = int(
                round(
                    scale_base_units(
                        config.BASE_ENTRY_UNITS,
                        equity=balance if balance > 0 else nav,
                        ref_equity=balance if balance > 0 else nav,
                        env_prefix=config.ENV_PREFIX,
                    )
                )
            )
            units = int(round(base_units * _confidence_scale(confidence)))
            lot = allowed_lot(
                nav,
                sl_pips,
                margin_available=margin_available,
                price=price,
                margin_rate=margin_rate,
                pocket=config.POCKET,
                side=side,
                strategy_tag=config.STRATEGY_TAG,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
            )
            units_risk = int(round(max(0.0, lot) * 100000))
            units = min(units, units_risk)
            units = int(round(units * cap))

            dyn_mult = 1.0
            dyn_profile: Dict[str, object] = {}
            if config.DYN_ALLOC_ENABLED:
                dyn_profile = load_strategy_profile(
                    config.STRATEGY_TAG,
                    config.POCKET,
                    path=config.DYN_ALLOC_PATH,
                    ttl_sec=config.DYN_ALLOC_TTL_SEC,
                )
                if config.DYN_ALLOC_LOSER_BLOCK and bool(dyn_profile.get("found")):
                    dyn_trades = int(dyn_profile.get("trades", 0) or 0)
                    dyn_score = _safe_float(dyn_profile.get("score"), 0.0)
                    if dyn_trades >= config.DYN_ALLOC_MIN_TRADES and dyn_score <= config.DYN_ALLOC_LOSER_SCORE:
                        prev_rsi = rsi
                        continue
                if bool(dyn_profile.get("found")):
                    dyn_mult = _safe_float(dyn_profile.get("lot_multiplier"), 1.0)
                    dyn_mult = max(config.DYN_ALLOC_MULT_MIN, min(config.DYN_ALLOC_MULT_MAX, dyn_mult))
            units = int(round(units * dyn_mult))
            if units < config.MIN_UNITS:
                prev_rsi = rsi
                continue
            if side == "short":
                units = -abs(units)

            if side == "long":
                sl_price = price - sl_pips * config.PIP_VALUE
                tp_price = price + tp_pips * config.PIP_VALUE
            else:
                sl_price = price + sl_pips * config.PIP_VALUE
                tp_price = price - tp_pips * config.PIP_VALUE
            sl_price, tp_price = clamp_sl_tp(
                price=price,
                sl=round(sl_price, 3),
                tp=round(tp_price, 3),
                is_buy=side == "long",
            )

            entry_thesis: Dict[str, object] = {
                "strategy_tag": config.STRATEGY_TAG,
                "env_prefix": config.ENV_PREFIX,
                "pattern_gate_opt_in": bool(config.PATTERN_GATE_OPT_IN),
                "confidence": confidence,
                "entry_probability": _to_probability(confidence),
                "sl_pips": round(sl_pips, 3),
                "tp_pips": round(tp_pips, 3),
                "hard_stop_pips": round(sl_pips, 3),
                "rsi": round(rsi, 3),
                "prev_rsi": round(prev_rsi, 3) if prev_rsi is not None else None,
                "range_active": range_active,
                "range_score": round(range_score, 3),
                "range_mode": str(range_ctx.mode or ""),
                "adx": round(adx, 3),
                "atr_pips": round(atr_pips, 3),
                "divergence": div_meta,
            }
            if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
                entry_thesis["dynamic_alloc"] = {
                    "strategy_key": dyn_profile.get("strategy_key"),
                    "score": round(_safe_float(dyn_profile.get("score"), 0.0), 3),
                    "trades": int(dyn_profile.get("trades", 0) or 0),
                    "lot_multiplier": round(dyn_mult, 3),
                }

            res = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=_client_order_id(config.STRATEGY_TAG),
                strategy_tag=config.STRATEGY_TAG,
                entry_thesis=dict(entry_thesis, entry_units_intent=abs(units)),
                )
            last_entry_mono = now_mono
            if side == "long":
                long_arm_until = 0.0
            else:
                short_arm_until = 0.0
            LOG.info(
                "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f rsi=%.2f div=%.3f conf=%s cap=%.2f dyn=%.2f reasons=%s res=%s",
                config.LOG_PREFIX,
                units,
                side,
                price,
                sl_price,
                tp_price,
                rsi,
                div_score,
                confidence,
                cap,
                dyn_mult,
                cap_reason,
                res if res else "none",
            )
            prev_rsi = rsi
    except asyncio.CancelledError:
        LOG.info("%s cancelled", config.LOG_PREFIX)
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    asyncio.run(scalp_macd_rsi_div_worker())
