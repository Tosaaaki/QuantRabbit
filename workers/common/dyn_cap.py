from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from utils.env_utils import env_float


@dataclass
class CapResult:
    cap: float
    reasons: Dict[str, float]


def compute_cap(
    *,
    atr_pips: float,
    free_ratio: float,
    range_active: bool,
    perf_pf: Optional[float],
    pos_bias: float,
    cap_min: float = 0.25,
    cap_max: float = 0.95,
    target_margin_usage: Optional[float] = None,
    env_prefix: Optional[str] = None,
) -> CapResult:
    """Dynamic cap based on market/position state."""
    pf_boost_min = env_float("DYN_CAP_PF_BOOST_MIN", 1.3, prefix=env_prefix)
    pf_boost_mult = env_float("DYN_CAP_PF_BOOST_MULT", 1.15, prefix=env_prefix)
    pf_cut_max = env_float("DYN_CAP_PF_CUT_MAX", 0.9, prefix=env_prefix)
    pf_cut_mult = env_float("DYN_CAP_PF_CUT_MULT", 0.8, prefix=env_prefix)
    range_cap = env_float("DYN_CAP_RANGE_CAP", 0.55, prefix=env_prefix)

    # Aggressive: start near the upper bound to target high utilization
    cap = cap_max * 0.9
    reasons: Dict[str, float] = {}

    # Aim for target margin usage (e.g., 0.9 = use 90% of equity)
    if target_margin_usage is None:
        target_usage = env_float("TARGET_MARGIN_USAGE", 0.85, prefix=env_prefix)
    else:
        target_usage = float(target_margin_usage)
    target_usage = max(0.1, min(0.99, target_usage))
    current_usage = max(0.0, min(1.0, 1.0 - free_ratio))
    if current_usage > 0:
        adj = target_usage / current_usage
        # clamp adjustment to avoid extreme swings
        adj = max(0.8, min(1.2, adj))
        cap *= adj
        reasons["usage_adj"] = round(adj, 3)
        reasons["usage_now"] = round(current_usage, 3)

    # ATR / volatility
    if atr_pips >= 3.0:
        cap *= 1.1
        reasons["atr_boost"] = atr_pips
    elif atr_pips <= 1.2:
        cap *= 0.85
        reasons["atr_cut"] = atr_pips

    # Free margin ratio
    if free_ratio < 0.02:
        return CapResult(0.0, {"free_ratio": free_ratio})
    if free_ratio < 0.10:
        cap *= 0.7
        reasons["fmr_tight"] = free_ratio
    elif free_ratio < 0.20:
        cap *= 0.9
        reasons["fmr_soft"] = free_ratio
    else:
        cap *= 1.02
        reasons["fmr_ok"] = free_ratio

    # Performance (PF)
    if perf_pf is not None:
        if perf_pf >= pf_boost_min:
            cap *= pf_boost_mult
            reasons["pf_boost"] = perf_pf
        elif perf_pf <= pf_cut_max:
            cap *= pf_cut_mult
            reasons["pf_cut"] = perf_pf

    # Range guard
    if range_active:
        if range_cap > 0:
            cap = min(cap, range_cap)
            reasons["range_cap"] = round(range_cap, 3)

    # Position bias (same side exposure)
    if pos_bias > 0.5:
        cap *= 0.8
        reasons["pos_bias_cut"] = pos_bias

    cap = max(cap_min, min(cap_max, cap))
    reasons["cap"] = cap
    return CapResult(cap=cap, reasons=reasons)
