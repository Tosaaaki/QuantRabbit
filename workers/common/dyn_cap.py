from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


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
) -> CapResult:
    """Dynamic cap based on market/position state."""
    cap = 0.7
    reasons: Dict[str, float] = {}

    # ATR / volatility
    if atr_pips >= 3.0:
        cap *= 1.2
        reasons["atr_boost"] = atr_pips
    elif atr_pips <= 1.2:
        cap *= 0.75
        reasons["atr_cut"] = atr_pips

    # Free margin ratio
    if free_ratio < 0.03:
        return CapResult(0.0, {"free_ratio": free_ratio})
    if free_ratio < 0.15:
        cap *= 0.65
        reasons["fmr_tight"] = free_ratio
    elif free_ratio < 0.25:
        cap *= 0.85
        reasons["fmr_soft"] = free_ratio
    else:
        cap *= 1.05
        reasons["fmr_ok"] = free_ratio

    # Performance (PF)
    if perf_pf is not None:
        if perf_pf >= 1.3:
            cap *= 1.15
            reasons["pf_boost"] = perf_pf
        elif perf_pf <= 0.9:
            cap *= 0.7
            reasons["pf_cut"] = perf_pf

    # Range guard
    if range_active:
        cap = min(cap, 0.4)
        reasons["range_cap"] = 1.0

    # Position bias (same side exposure)
    if pos_bias > 0.5:
        cap *= 0.7
        reasons["pos_bias_cut"] = pos_bias

    cap = max(cap_min, min(cap_max, cap))
    reasons["cap"] = cap
    return CapResult(cap=cap, reasons=reasons)
