"""Scalp multi-strategy worker with dynamic cap."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from autotune.scalp_trainer import AUTO_INTERVAL_SEC, start_background_autotune
from analysis.range_guard import detect_range_mode
from analysis.range_model import compute_range_snapshot
from indicators.factor_cache import all_factors, get_candles_snapshot
from execution.order_manager import market_order
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from execution.position_manager import PositionManager
from market_data import tick_window
from strategies.scalping.range_fader import RangeFader
from strategies.scalping.pulse_break import PulseBreak
from strategies.scalping.impulse_retrace import ImpulseRetraceScalp
from strategies.scalping.m1_scalper import M1Scalper
from utils.market_hours import is_market_open
from utils.oanda_account import get_account_snapshot
from workers.common.dyn_cap import compute_cap
from analysis import perf_monitor

from . import config

LOG = logging.getLogger(__name__)
PM = PositionManager()
_LAST_ENTRY_TS: float = 0.0
MR_RANGE_LOOKBACK = 20
MR_RANGE_HI_PCT = 95.0
MR_RANGE_LO_PCT = 5.0
_STRATEGY_LAST_TS: Dict[str, float] = {}
_TREND_STRATEGIES = {
    PulseBreak.name,
    ImpulseRetraceScalp.name,
}
_PULLBACK_STRATEGIES = {
    ImpulseRetraceScalp.name,
    M1Scalper.name,
}
_RANGE_STRATEGIES = {
    RangeFader.name,
}


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
    sanitized = "".join(ch.lower() for ch in tag if ch.isalnum())[:8] or "scalp"
    digest = hashlib.sha1(f"{ts_ms}-{tag}".encode("utf-8")).hexdigest()[:9]
    return f"qr-{ts_ms}-scalp-{sanitized}{digest}"


def _confidence_scale(conf: int) -> float:
    lo = config.CONFIDENCE_FLOOR
    hi = config.CONFIDENCE_CEIL
    if conf <= lo:
        return 0.5
    if conf >= hi:
        return 1.0
    span = (conf - lo) / max(1.0, hi - lo)
    return 0.5 + span * 0.5


def _compute_cap(*args, **kwargs) -> Tuple[float, Dict[str, float]]:
    res = compute_cap(cap_min=config.CAP_MIN, cap_max=config.CAP_MAX, *args, **kwargs)
    return res.cap, res.reasons


def _is_mr_signal(tag: str) -> bool:
    tag_str = (tag or "").strip()
    if not tag_str:
        return False
    base_tag = tag_str.split("-", 1)[0]
    return base_tag == "RangeFader"


def _build_mr_entry_thesis(
    signal: Dict,
    *,
    strategy_tag: str,
    atr_entry: float,
    entry_mean: Optional[float],
) -> Dict:
    thesis: Dict[str, object] = {
        "strategy_tag": strategy_tag,
        "confidence": signal.get("confidence", 0),
        "env_tf": "M5",
        "struct_tf": "M1",
        "range_method": "percentile",
        "range_lookback": MR_RANGE_LOOKBACK,
        "range_hi_pct": MR_RANGE_HI_PCT,
        "range_lo_pct": MR_RANGE_LO_PCT,
        "atr_entry": atr_entry,
        "structure_break": {"buffer_atr": 0.10, "confirm_closes": 2},
        "tp_mode": "soft_zone",
        "tp_target": "entry_mean",
        "tp_pad_atr": 0.05,
        "reversion_failure": {
            "z_ext": 0.45,
            "contraction_min": 0.45,
            "bars_budget": {"k_per_z": 2.5, "min": 2, "max": 8},
            "trend_takeover": {"require_env_trend_bars": 2},
        },
    }
    if entry_mean is not None and entry_mean > 0:
        thesis["entry_mean"] = float(entry_mean)
    candles = get_candles_snapshot("M5", limit=MR_RANGE_LOOKBACK)
    snapshot = compute_range_snapshot(
        candles,
        lookback=MR_RANGE_LOOKBACK,
        method="percentile",
        hi_pct=MR_RANGE_HI_PCT,
        lo_pct=MR_RANGE_LO_PCT,
    )
    if snapshot:
        thesis["range_snapshot"] = snapshot.to_dict()
        thesis.setdefault("entry_mean", snapshot.mid)
    return thesis


def _diversity_bonus(strategy_name: str, now_ts: float) -> float:
    if not config.DIVERSITY_ENABLED:
        return 0.0
    last_ts = _STRATEGY_LAST_TS.get(strategy_name)
    if last_ts is None:
        return config.DIVERSITY_MAX_BONUS
    idle = max(0.0, now_ts - last_ts)
    if idle < config.DIVERSITY_IDLE_SEC:
        return 0.0
    scale = max(1.0, config.DIVERSITY_SCALE_SEC)
    bonus = (idle - config.DIVERSITY_IDLE_SEC) / scale * config.DIVERSITY_MAX_BONUS
    return min(config.DIVERSITY_MAX_BONUS, bonus)


def _strategy_list() -> List:
    """Select strategies based on env allowlist (comma-separated names)."""
    allow_raw = (os.getenv("SCALP_STRATEGY_ALLOWLIST") or "").strip()
    allowlist = {token.strip() for token in allow_raw.split(",") if token.strip()}
    catalog = {
        "RangeFader": RangeFader,
        "PulseBreak": PulseBreak,
        "ImpulseRetrace": ImpulseRetraceScalp,
        "ImpulseRetraceScalp": ImpulseRetraceScalp,
        "M1Scalper": M1Scalper,
    }
    if not allowlist:
        return list(catalog.values())

    selected = []
    for name in allowlist:
        strat = catalog.get(name)
        if strat:
            selected.append(strat)
        else:
            LOG.warning("%s unknown strategy in allowlist: %s", config.LOG_PREFIX, name)

    if not selected:
        LOG.warning("%s allowlist provided but no strategies matched; skipping all", config.LOG_PREFIX)
    return selected


async def scalp_multi_worker() -> None:
    if not config.ENABLED:
        LOG.info("%s disabled (idle)", config.LOG_PREFIX)
        try:
            while True:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            return
    strategies = _strategy_list()
    LOG.info(
        "%s worker start (interval=%.1fs) strategies=%s",
        config.LOG_PREFIX,
        config.LOOP_INTERVAL_SEC,
        [getattr(s, "name", s.__name__) for s in strategies],
    )
    if config.AUTOTUNE_ENABLED:
        start_background_autotune()
        LOG.info(
            "%s scalp_autotune enabled interval_sec=%s",
            config.LOG_PREFIX,
            AUTO_INTERVAL_SEC,
        )

    while True:
        await asyncio.sleep(config.LOOP_INTERVAL_SEC)
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            continue
        if not can_trade(config.POCKET):
            continue
        # クールダウン: 直近エントリーから一定時間はスキップ
        now_ts = time.time()
        if now_ts - _LAST_ENTRY_TS < config.COOLDOWN_SEC:
            continue

        factors = all_factors()
        fac_m1 = factors.get("M1") or {}
        fac_h4 = factors.get("H4") or {}
        range_ctx = detect_range_mode(fac_m1, fac_h4)
        range_score = 0.0
        try:
            range_score = float(range_ctx.score or 0.0)
        except Exception:
            range_score = 0.0
        range_only = range_ctx.active or range_score >= config.RANGE_ONLY_SCORE
        range_bias = range_score >= config.RANGE_BIAS_SCORE
        fac_m1 = dict(fac_m1)
        fac_m1["range_active"] = bool(range_ctx.active)
        fac_m1["range_score"] = range_score
        fac_m1["range_reason"] = range_ctx.reason
        fac_m1["range_mode"] = range_ctx.mode
        perf = perf_monitor.snapshot()
        pf = None
        try:
            pf = float((perf.get(config.POCKET) or {}).get("pf"))
        except Exception:
            pf = None

        candidates: List[Tuple[float, int, Dict, str]] = []
        for strat in strategies:
            strategy_name = getattr(strat, "name", strat.__name__)
            if range_only and strategy_name not in _RANGE_STRATEGIES:
                continue
            cand = strat.check(fac_m1)
            if not cand:
                continue
            base_conf = int(cand.get("confidence", 0) or 0)
            bonus = _diversity_bonus(strategy_name, now_ts)
            score = base_conf + bonus
            if range_bias:
                if strategy_name in _RANGE_STRATEGIES:
                    score += config.RANGE_STRATEGY_BONUS * range_score
                else:
                    score -= config.RANGE_TREND_PENALTY * range_score
            candidates.append((score, base_conf, cand, strategy_name))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        snap = get_account_snapshot()
        # 同ポケットのオープントレード数制限
        available_slots = None
        try:
            positions = PM.get_open_positions()
            scalp_info = positions.get("scalp") or {}
            open_trades = len(scalp_info.get("open_trades") or [])
            available_slots = max(0, config.MAX_OPEN_TRADES - open_trades)
            if available_slots <= 0:
                continue
        except Exception:
            available_slots = None
        max_signals = max(1, int(config.MAX_SIGNALS_PER_CYCLE))
        if available_slots is not None:
            max_signals = max(1, min(max_signals, available_slots))
        selected = candidates[:max_signals]
        free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0
        try:
            atr_pips = float(fac_m1.get("atr_pips") or 0.0)
        except Exception:
            atr_pips = 0.0
        pos_bias = 0.0
        try:
            open_positions = snap.positions or {}
            scalp_pos = open_positions.get("scalp") or {}
            pos_bias = abs(float(scalp_pos.get("units", 0.0) or 0.0)) / max(1.0, float(snap.nav or 1.0))
        except Exception:
            pos_bias = 0.0

        cap, cap_reason = _compute_cap(
            atr_pips=atr_pips,
            free_ratio=free_ratio,
            range_active=range_ctx.active,
            perf_pf=pf,
            pos_bias=pos_bias,
        )
        if cap <= 0.0:
            continue
        multi_scale = max(
            float(config.MULTI_SIGNAL_MIN_SCALE),
            1.0 / max(1, len(selected)),
        )

        try:
            price = float(fac_m1.get("close") or 0.0)
        except Exception:
            price = 0.0
        price = _latest_mid(price)
        for _, _, signal, strategy_name in selected:
            signal_tag = (signal.get("tag") or "").strip() or strategy_name
            side = "long" if signal["action"] == "OPEN_LONG" else "short"
            sl_pips = float(signal.get("sl_pips") or 0.0)
            tp_pips = float(signal.get("tp_pips") or 0.0)
            if price <= 0.0 or sl_pips <= 0.0:
                continue

            tp_scale = 4.0 / max(1.0, tp_pips)
            tp_scale = max(0.4, min(1.2, tp_scale))
            base_units = int(round(config.BASE_ENTRY_UNITS * tp_scale))

            conf_scale = _confidence_scale(int(signal.get("confidence", 50)))
            lot = allowed_lot(
                float(snap.nav or 0.0),
                sl_pips,
                margin_available=float(snap.margin_available or 0.0),
                price=price,
                margin_rate=float(snap.margin_rate or 0.0),
                pocket=config.POCKET,
                strategy_tag=signal.get("tag"),
                fac_m1=fac_m1,
                fac_h4=fac_h4,
            )
            units_risk = int(round(lot * 100000))
            units = int(round(base_units * conf_scale))
            units = min(units, units_risk)
            units = int(round(units * cap))
            units = int(round(units * multi_scale))
            if units < config.MIN_UNITS:
                continue
            if side == "short":
                units = -abs(units)

            if side == "long":
                sl_price = round(price - sl_pips * 0.01, 3)
                tp_price = round(price + tp_pips * 0.01, 3) if tp_pips > 0 else None
            else:
                sl_price = round(price + sl_pips * 0.01, 3)
                tp_price = round(price - tp_pips * 0.01, 3) if tp_pips > 0 else None

            sl_price, tp_price = clamp_sl_tp(
                price=price,
                sl=sl_price,
                tp=tp_price,
                is_buy=side == "long",
            )
            client_id = _client_order_id(signal_tag)
            entry_thesis = {
                "strategy_tag": signal_tag,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
                "hard_stop_pips": sl_pips,
                "confidence": signal.get("confidence", 0),
                "range_active": bool(range_ctx.active),
                "range_score": round(range_score, 3),
                "range_reason": range_ctx.reason,
                "range_mode": range_ctx.mode,
            }
            if strategy_name in _TREND_STRATEGIES:
                entry_thesis["entry_guard_trend"] = True
            if strategy_name in _PULLBACK_STRATEGIES:
                entry_thesis["entry_guard_pullback"] = True
                entry_thesis["entry_guard_pullback_only"] = True
            if strategy_name == M1Scalper.name:
                entry_thesis["entry_guard_trend"] = True
            if _is_mr_signal(signal_tag):
                entry_thesis["entry_guard_trend"] = False
                entry_mean = None
                try:
                    entry_mean = float(
                        fac_m1.get("ema20") or fac_m1.get("ma20") or fac_m1.get("ma10") or 0.0
                    ) or None
                except Exception:
                    entry_mean = None
                entry_thesis.update(
                    _build_mr_entry_thesis(
                        signal,
                        strategy_tag=signal_tag,
                        atr_entry=atr_pips or 1.0,
                        entry_mean=entry_mean,
                    )
                )
                rf = entry_thesis.get("reversion_failure")
                if isinstance(rf, dict):
                    bars_budget = rf.get("bars_budget")
                    if not isinstance(bars_budget, dict):
                        bars_budget = {}
                        rf["bars_budget"] = bars_budget
                    if atr_pips >= 6.0:
                        bars_budget["k_per_z"] = 3.0
                        bars_budget["max"] = 10
                    elif 0.0 < atr_pips <= 3.0:
                        bars_budget["k_per_z"] = 2.0
                        bars_budget["max"] = 6

            res = await market_order(
                instrument="USD_JPY",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
                pocket=config.POCKET,
                client_order_id=client_id,
                strategy_tag=signal_tag,
                confidence=int(signal.get("confidence", 0)),
                entry_thesis=entry_thesis,
            )
            _STRATEGY_LAST_TS[strategy_name] = time.time()
            LOG.info(
                "%s strat=%s sent units=%s side=%s price=%.3f sl=%.3f tp=%.3f conf=%.0f cap=%.2f multi=%.2f reasons=%s res=%s",
                config.LOG_PREFIX,
                strategy_name,
                units,
                side,
                price,
                sl_price,
                tp_price,
                signal.get("confidence", 0),
                cap,
                multi_scale,
                {**cap_reason, "tp_scale": round(tp_scale, 3)},
                res or "none",
            )


if __name__ == "__main__":
    asyncio.run(scalp_multi_worker())
