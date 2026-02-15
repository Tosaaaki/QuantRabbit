"""Entry loop for TrendReclaimLong (scalp pocket)."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Tuple

from analysis import perf_monitor
from analysis.range_guard import detect_range_mode
from analysis.technique_engine import evaluate_entry_techniques
from execution.strategy_entry import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from indicators.factor_cache import all_factors
from market_data import spread_monitor, tick_window
from strategies.scalping.trend_reclaim_long import TrendReclaimLong
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot, get_position_summary
from workers.common import perf_guard
from workers.common.dyn_cap import compute_cap
from workers.common.dynamic_alloc import load_strategy_profile
from workers.common.size_utils import scale_base_units

from . import config

LOG = logging.getLogger(__name__)


def _latest_mid(fallback: float) -> float:
    ticks = tick_window.recent_ticks(seconds=6.0, limit=1)
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


def _client_order_id(tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "trr"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.55
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.55 + span * 0.45


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    kwargs.setdefault("env_prefix", config.ENV_PREFIX)
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


def _build_strategy_input(fac_m1: Dict, fac_m5: Dict, fac_h1: Dict) -> Dict:
    fac = dict(fac_m1)
    fac["h1_close"] = fac_h1.get("close")
    fac["h1_ema20"] = fac_h1.get("ema20") or fac_h1.get("ma20")
    fac["h1_adx"] = fac_h1.get("adx")
    fac["m5_close"] = fac_m5.get("close")
    fac["m5_ema20"] = fac_m5.get("ema20") or fac_m5.get("ma20")
    return fac


async def scalp_trend_reclaim_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled", config.LOG_PREFIX)
        return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)

    last_entry_ts = 0.0
    last_block_log = 0.0
    last_spread_log = 0.0

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now_utc = datetime.datetime.utcnow()
        if not is_market_open(now_utc):
            continue
        if not can_trade(config.POCKET):
            continue

        blocked, remain, spread_state, spread_reason = spread_monitor.is_blocked()
        spread_pips = float(spread_state.get("spread_pips") or 0.0) if spread_state else 0.0
        spread_stale = bool(spread_state.get("stale")) if spread_state else False
        if blocked or spread_stale or spread_pips > config.MAX_SPREAD_PIPS:
            now_mono = time.monotonic()
            if now_mono - last_spread_log > 60.0:
                LOG.info(
                    "%s spread_block spread=%.2f stale=%s remain=%ss reason=%s",
                    config.LOG_PREFIX,
                    spread_pips,
                    spread_stale,
                    remain,
                    spread_reason or "guard",
                )
                last_spread_log = now_mono
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_m5 = factors.get("M5") or {}
        fac_h1 = factors.get("H1") or {}
        fac_h4 = factors.get("H4") or {}
        if not fac_m1:
            continue

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        if bool(range_ctx.active) and range_score >= config.RANGE_BLOCK_SCORE:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s range_block score=%.3f mode=%s reason=%s",
                    config.LOG_PREFIX,
                    range_score,
                    str(range_ctx.mode or "-"),
                    str(range_ctx.reason or "-"),
                )
                last_block_log = now_mono
            continue

        strategy_input = _build_strategy_input(fac_m1, fac_m5, fac_h1)
        signal = TrendReclaimLong.check(strategy_input)
        if not signal:
            continue
        signal_tag = str(signal.get("tag") or TrendReclaimLong.name)

        conf_val = int(signal.get("confidence", 0) or 0)
        if conf_val < config.MIN_ENTRY_CONF:
            continue

        now_ts = time.time()
        if (now_ts - last_entry_ts) < config.COOLDOWN_SEC:
            continue

        if config.BLOCK_IF_LONG_OPEN:
            try:
                long_units, _ = get_position_summary("USD_JPY", timeout=3.0)
            except Exception:
                long_units = 0.0
            if long_units > 0.0:
                continue

        perf_decision = perf_guard.is_allowed(signal_tag, config.POCKET, env_prefix=config.ENV_PREFIX)
        if not perf_decision.allowed:
            now_mono = time.monotonic()
            if now_mono - last_block_log > 120.0:
                LOG.info(
                    "%s perf_block tag=%s reason=%s",
                    config.LOG_PREFIX,
                    signal_tag,
                    perf_decision.reason,
                )
                last_block_log = now_mono
            continue

        dyn_profile: Dict[str, object] = {}
        if config.DYN_ALLOC_ENABLED:
            dyn_profile = load_strategy_profile(
                signal_tag,
                config.POCKET,
                path=config.DYN_ALLOC_PATH,
                ttl_sec=config.DYN_ALLOC_TTL_SEC,
            )
            if not bool(dyn_profile.get("found")):
                dyn_profile = load_strategy_profile(
                    TrendReclaimLong.name,
                    config.POCKET,
                    path=config.DYN_ALLOC_PATH,
                    ttl_sec=config.DYN_ALLOC_TTL_SEC,
                )
            if config.DYN_ALLOC_LOSER_BLOCK and bool(dyn_profile.get("found")):
                dyn_trades = int(dyn_profile.get("trades", 0) or 0)
                dyn_score = float(dyn_profile.get("score", 0.0) or 0.0)
                if dyn_trades >= config.DYN_ALLOC_MIN_TRADES and dyn_score <= config.DYN_ALLOC_LOSER_SCORE:
                    continue

        snap = get_account_snapshot()
        equity = float(snap.nav or snap.balance or 0.0)
        balance = float(snap.balance or snap.nav or 0.0)
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0

        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None
        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0

        cap, cap_reason = _compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=bool(range_ctx.active),
            perf_pf=pf,
            pos_bias=0.0,
        )
        if cap <= 0.0:
            continue

        price = _latest_mid(float(fac_m1.get("close") or 0.0))
        if price <= 0.0:
            continue
        side = "long" if str(signal.get("action") or "") == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if sl_pips <= 0.0 or tp_pips <= 0.0:
            continue

        base_units = int(
            round(
                scale_base_units(
                    config.BASE_ENTRY_UNITS,
                    equity=balance if balance > 0 else equity,
                    ref_equity=balance,
                    env_prefix=config.ENV_PREFIX,
                )
            )
        )

        conf_scale = _confidence_scale(conf_val)
        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=config.POCKET,
            side=side,
            strategy_tag=signal_tag,
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units_risk = int(round(lot * 100000))
        units = int(round(base_units * conf_scale))
        units = min(units, units_risk)
        units = int(round(units * cap))

        dyn_mult = 1.0
        dyn_score = 0.0
        dyn_trades = 0
        if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
            dyn_mult = float(dyn_profile.get("lot_multiplier", 1.0) or 1.0)
            dyn_mult = max(config.DYN_ALLOC_MULT_MIN, min(config.DYN_ALLOC_MULT_MAX, dyn_mult))
            dyn_score = float(dyn_profile.get("score", 0.0) or 0.0)
            dyn_trades = int(dyn_profile.get("trades", 0) or 0)
        units = int(round(units * dyn_mult))

        if units < config.MIN_UNITS:
            continue
        if side == "short":
            units = -abs(units)

        if side == "long":
            sl_price = price - sl_pips * 0.01
            tp_price = price + tp_pips * 0.01
        else:
            sl_price = price + sl_pips * 0.01
            tp_price = price - tp_pips * 0.01
        sl_price, tp_price = clamp_sl_tp(
            price=price,
            sl=round(sl_price, 3),
            tp=round(tp_price, 3),
            is_buy=side == "long",
        )

        entry_thesis = {
            "strategy_tag": signal_tag,
            "env_prefix": config.ENV_PREFIX,
            "profile": signal.get("profile"),
            "confidence": conf_val,
            "sl_pips": round(sl_pips, 2),
            "tp_pips": round(tp_pips, 2),
            "hard_stop_pips": round(sl_pips, 2),
            "range_active": bool(range_ctx.active),
            "range_score": round(range_score, 3),
        }
        notes = signal.get("notes")
        if isinstance(notes, dict) and notes:
            entry_thesis["notes"] = notes
        if config.DYN_ALLOC_ENABLED and bool(dyn_profile.get("found")):
            entry_thesis["dynamic_alloc"] = {
                "strategy_key": dyn_profile.get("strategy_key"),
                "score": round(dyn_score, 3),
                "trades": dyn_trades,
                "lot_multiplier": round(dyn_mult, 3),
            }

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
            pocket=config.POCKET,
            client_order_id=_client_order_id(signal_tag),
            strategy_tag=signal_tag,
            entry_thesis=entry_thesis,
        )
        last_entry_ts = now_ts
        LOG.info(
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%s cap=%.2f dyn=%.2f reasons=%s res=%s",
            config.LOG_PREFIX,
            units,
            side,
            price,
            sl_price,
            tp_price,
            conf_val,
            cap,
            dyn_mult,
            cap_reason,
            res if res else "none",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    asyncio.run(scalp_trend_reclaim_worker())
