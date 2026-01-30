from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from execution.risk_guard import allowed_lot
from workers.common import perf_guard
from utils.oanda_account import get_account_snapshot


@dataclass
class SizingContext:
    units: int
    capped_by_margin: bool
    cap_units: int
    risk_pct_applied: float
    factors: dict


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def compute_units(
    *,
    entry_price: float,
    sl_pips: float,
    base_entry_units: int,
    min_units: int,
    max_margin_usage: float,
    spread_pips: float,
    spread_soft_cap: float,
    adx: Optional[float] = None,
    signal_score: Optional[float] = None,  # 0..1 (confidence-like)
    pocket: Optional[str] = None,
    strategy_tag: Optional[str] = None,
) -> SizingContext:
    """Flexible unit sizing with risk/volatility/margin awareness.

    Returns a SizingContext including final units and diagnostics for logging.
    """
    snap = get_account_snapshot()
    equity = snap.nav or snap.balance or 0.0
    margin_avail = float(snap.margin_available or 0.0)
    margin_rate = float(snap.margin_rate or 0.0)
    free_ratio = float(snap.free_margin_ratio or 0.0) if snap.free_margin_ratio is not None else 0.0

    # 1) Dynamic risk based on free margin ratio
    #    Very low free margin -> scale down aggressively
    if free_ratio < 0.03:
        return SizingContext(0, False, 0, 0.0, {"reason": "low_free_margin"})
    if free_ratio < 0.05:
        free_scale = 0.35
    elif free_ratio < 0.08:
        free_scale = 0.55
    elif free_ratio < 0.15:
        free_scale = 0.8
    elif free_ratio < 0.25:
        free_scale = 1.0
    else:
        free_scale = 1.2

    base_risk_pct = max(0.0005, _env_float("DYN_SIZE_BASE_RISK_PCT", 0.01))
    min_risk_pct = max(0.0005, _env_float("DYN_SIZE_MIN_RISK_PCT", 0.002))
    max_risk_pct = max(min_risk_pct, _env_float("DYN_SIZE_MAX_RISK_PCT", 0.03))
    risk_pct = _clamp(base_risk_pct * free_scale, min_risk_pct, max_risk_pct)

    # 2) Allowed lot from risk math (equity/sl)
    lot = allowed_lot(
        equity,
        sl_pips=max(1.0, sl_pips),
        margin_available=margin_avail,
        price=entry_price,
        margin_rate=margin_rate,
        risk_pct_override=risk_pct,
        pocket=pocket,
    )
    units_risk = int(round(lot * 100000))

    # 3) Spread penalty
    if spread_pips <= 0.0:
        s_spread = 1.0
    elif spread_pips <= spread_soft_cap:
        s_spread = 1.0
    else:
        excess = max(0.0, spread_pips - spread_soft_cap)
        s_spread = _clamp(1.0 - excess / max(0.2, spread_soft_cap), 0.35, 1.0)

    # 4) ADX scaling (trend quality)
    s_adx = 1.0
    if adx is not None:
        if adx >= 30.0:
            s_adx = 1.18
        elif adx >= 26.0:
            s_adx = 1.08
        elif adx >= 22.0:
            s_adx = 1.0
        elif adx >= 18.0:
            s_adx = 0.88
        else:
            s_adx = 0.8

    # 5) Signal strength scaling (optional 0..1)
    s_sig = 1.0
    if signal_score is not None:
        s_sig = _clamp(0.8 + 0.4 * signal_score, 0.6, 1.3)

    # 6) Aggregate scaling
    scaled_units = int(round(units_risk * s_spread * s_adx * s_sig))
    perf_mult = 1.0
    if strategy_tag and pocket:
        try:
            perf = perf_guard.perf_scale(strategy_tag, pocket)
            perf_mult = float(perf.multiplier or 1.0)
            if perf_mult > 1.0:
                scaled_units = int(round(scaled_units * perf_mult))
        except Exception:
            perf_mult = 1.0
    # Soft floor: prefer base_entry_units but allow going below when conditions are poor
    if scaled_units < base_entry_units:
        # Blend towards base by 50%
        blended = int(round(0.5 * base_entry_units + 0.5 * scaled_units))
        scaled_units = max(min_units, blended)

    # 7) Margin cap (usage budget)
    cap_units = 0
    if margin_rate > 0.0 and entry_price > 0.0:
        per_unit = entry_price * margin_rate
        if per_unit > 0.0:
            cap_units = int((margin_avail * max(0.1, min(1.0, max_margin_usage))) / per_unit)
    final_units = scaled_units
    capped = False
    if cap_units > 0 and final_units > cap_units:
        final_units = cap_units
        capped = True

    if final_units < min_units:
        return SizingContext(0, capped, cap_units, risk_pct, {"reason": "below_min_units", "units_risk": units_risk})

    return SizingContext(
        final_units,
        capped,
        cap_units,
        risk_pct,
        {
            "units_risk": units_risk,
            "s_spread": s_spread,
            "s_adx": s_adx,
            "s_sig": s_sig,
            "free_ratio": free_ratio,
            "perf_mult": perf_mult,
        },
    )
