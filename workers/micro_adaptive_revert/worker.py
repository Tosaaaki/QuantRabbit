"""MicroAdaptiveRevert dedicated worker."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import time
from typing import Dict, Optional, Tuple

from analysis.range_guard import detect_range_mode
from analysis import perf_monitor
from indicators.factor_cache import all_factors
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from market_data import tick_window
from strategies.micro_lowvol.micro_adaptive_revert import MicroAdaptiveRevert
from utils.divergence import apply_divergence_confidence, divergence_bias, divergence_snapshot
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.air_state import adjust_signal, evaluate_air
from workers.common.dyn_cap import compute_cap
from workers.common.size_utils import scale_base_units

from . import config

LOG = logging.getLogger(__name__)


def _factor_age_seconds(factors: Dict[str, float]) -> float:
    ts_raw = factors.get("timestamp") if isinstance(factors, dict) else None
    if not ts_raw:
        return float("inf")
    try:
        if isinstance(ts_raw, (int, float)):
            ts_val = float(ts_raw)
            ts_dt = datetime.datetime.utcfromtimestamp(ts_val).replace(tzinfo=datetime.timezone.utc)
        else:
            ts_txt = str(ts_raw)
            if ts_txt.endswith("Z"):
                ts_txt = ts_txt.replace("Z", "+00:00")
            ts_dt = datetime.datetime.fromisoformat(ts_txt)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return float("inf")
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(0.0, (now - ts_dt).total_seconds())


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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:10] or "mar"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-mar-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.54
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.54 + span * 0.46


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    kwargs.setdefault("env_prefix", config.ENV_PREFIX)
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


def _calc_sl_tp(side: str, price: float, sl_pips: float, tp_pips: float) -> Tuple[float, Optional[float]]:
    if side == "long":
        sl_price = price - sl_pips * 0.01
        tp_price = price + tp_pips * 0.01 if tp_pips > 0 else None
    else:
        sl_price = price + sl_pips * 0.01
        tp_price = price - tp_pips * 0.01 if tp_pips > 0 else None
    return sl_price, tp_price


def _build_entry_thesis(signal: Dict, *, range_ctx, atr_entry: float, notes: Dict[str, object]) -> Dict:
    tag = signal.get("tag", MicroAdaptiveRevert.name)
    thesis: Dict[str, object] = {
        "strategy_tag": tag,
        "env_prefix": config.ENV_PREFIX,
        "profile": signal.get("profile"),
        "confidence": signal.get("confidence", 0),
        "tp_pips": signal.get("tp_pips"),
        "sl_pips": signal.get("sl_pips"),
        "hard_stop_pips": signal.get("sl_pips"),
        "range_active": bool(getattr(range_ctx, "active", False)),
        "range_score": float(getattr(range_ctx, "score", 0.0) or 0.0),
        "range_reason": getattr(range_ctx, "reason", None),
        "range_mode": getattr(range_ctx, "mode", None),
        "atr_entry": atr_entry,
        "pattern_gate_opt_in": bool(config.PATTERN_GATE_OPT_IN),
    }
    if notes:
        thesis["notes"] = notes
    return thesis


async def micro_adaptive_revert_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s worker disabled", config.LOG_PREFIX)
        return
    LOG.info("%s worker start (interval=%.1fs)", config.LOG_PREFIX, config.LOOP_INTERVAL_SEC)
    last_stale_log = 0.0
    max_factor_age = float(getattr(config, "MAX_FACTOR_AGE_SEC", 90.0))

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)

        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        fac_m5 = factors.get("M5") or {}
        if not fac_m1:
            continue
        age_m1 = _factor_age_seconds(fac_m1)
        if age_m1 > max_factor_age:
            if time.time() - last_stale_log > 30.0:
                LOG.warning(
                    "%s stale factors age=%.1fs limit=%.1fs (proceeding)",
                    config.LOG_PREFIX,
                    age_m1,
                    max_factor_age,
                )
                last_stale_log = time.time()

        range_ctx = detect_range_mode(fac_m1, fac_h4)
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        if not range_ctx.active and range_score < config.RANGE_SCORE_MIN:
            continue

        fac_m1 = dict(fac_m1)
        fac_m1["range_active"] = bool(range_ctx.active)
        fac_m1["range_score"] = range_score
        fac_m1["range_reason"] = range_ctx.reason
        fac_m1["range_mode"] = range_ctx.mode

        air = evaluate_air(fac_m1, fac_h4, range_ctx=range_ctx, tag=MicroAdaptiveRevert.name)
        if air.enabled and not air.allow_entry:
            continue

        signal = MicroAdaptiveRevert.check(fac_m1)
        if not signal:
            continue

        signal = adjust_signal(signal, air)
        if not signal:
            continue

        div_bias = divergence_bias(fac_m1, signal.get("action") or "", mode="reversion", max_age_bars=18)
        if div_bias:
            base_conf = int(signal.get("confidence", 0) or 0)
            signal["confidence"] = apply_divergence_confidence(
                base_conf,
                div_bias,
                max_bonus=8.0,
                max_penalty=8.5,
                floor=config.CONFIDENCE_FLOOR,
                ceil=config.CONFIDENCE_CEIL,
            )

        conf_val = int(signal.get("confidence", 0) or 0)
        if conf_val < config.MIN_ENTRY_CONF:
            continue

        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None

        snap = get_account_snapshot()
        equity = float(snap.nav or snap.balance or 0.0)
        balance = float(snap.balance or snap.nav or 0.0)
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0

        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
        try:
            atr_m5 = float(fac_m5.get("atr_pips") or 0.0)
        except Exception:
            atr_m5 = 0.0

        cap, _ = _compute_cap(atr_pips=atr_pips, free_ratio=free_ratio, range_active=range_ctx.active, perf_pf=pf)
        if cap <= 0.0:
            continue

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        if price <= 0.0:
            continue

        side = "long" if signal.get("action") == "OPEN_LONG" else "short"
        sl_pips = float(signal.get("sl_pips") or 0.0)
        tp_pips = float(signal.get("tp_pips") or 0.0)
        if sl_pips <= 0.0:
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
        lot_mult = 1.0
        notes = {}
        raw_notes = signal.get("notes")
        if isinstance(raw_notes, dict):
            notes = dict(raw_notes)
            hist_mult = notes.get("history_lot_mult")
            if isinstance(hist_mult, (int, float)) and hist_mult > 0:
                lot_mult = float(hist_mult)

        lot = allowed_lot(
            float(snap.nav or 0.0),
            sl_pips,
            margin_available=float(snap.margin_available or 0.0),
            price=price,
            margin_rate=float(snap.margin_rate or 0.0),
            pocket=config.POCKET,
            side=side,
            strategy_tag=signal.get("tag", MicroAdaptiveRevert.name),
            fac_m1=fac_m1,
            fac_h4=fac_h4,
        )
        units_risk = int(round(lot * 100000))

        units = int(round(base_units * conf_scale * lot_mult))
        units = min(units, units_risk)
        units = int(round(units * cap))
        if units < config.MIN_UNITS:
            continue
        if side == "short":
            units = -abs(units)

        sl_price, tp_price = _calc_sl_tp(side, price, sl_pips, tp_pips)
        sl_price, tp_price = clamp_sl_tp(price=price, sl=sl_price, tp=tp_price, is_buy=side == "long")

        entry_thesis = _build_entry_thesis(
            signal,
            range_ctx=range_ctx,
            atr_entry=atr_m5 or atr_pips or 1.0,
            notes=notes,
        )
        div_meta = divergence_snapshot(fac_m1, max_age_bars=18)
        if div_meta:
            entry_thesis["divergence"] = div_meta

        if notes:
            entry_thesis.setdefault("notes", notes)

        client_id = _client_order_id(signal.get("tag", MicroAdaptiveRevert.name))

        res = await market_order(
            instrument="USD_JPY",
            units=units,
            sl_price=sl_price,
            tp_price=tp_price,
            pocket=config.POCKET,
            client_order_id=client_id,
            strategy_tag=signal.get("tag", MicroAdaptiveRevert.name),
            confidence=conf_val,
            entry_thesis=entry_thesis,
        )
        LOG.info(
            "%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%s cap=%.2f res=%s",
            config.LOG_PREFIX,
            units,
            side,
            price,
            sl_price,
            tp_price if tp_price is not None else 0.0,
            conf_val,
            cap,
            res if res else "none",
        )


if __name__ == "__main__":
    asyncio.run(micro_adaptive_revert_worker())
