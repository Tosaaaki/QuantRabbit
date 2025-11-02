"""
Utility helpers for order sizing safeguards.

Currently used by Cloud Run trader service to enforce a floor on the
calculated base units without violating the risk-based ceiling.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional


def apply_min_units_floor(
    base_units: int,
    min_units_cfg: int,
    units_by_risk: int,
    units_by_margin: int,
) -> Tuple[int, Optional[Dict[str, object]]]:
    """Enforce a minimum order size while respecting risk and margin limits.

    Returns the adjusted base units together with metadata describing how the
    floor was applied. The metadata is ``None`` when no adjustment is needed.
    """

    base = max(int(base_units), 0)
    configured_floor = max(int(min_units_cfg), 0)
    risk_cap = max(int(units_by_risk), 0)
    margin_cap = max(int(units_by_margin), 0)

    if configured_floor <= 0 or margin_cap <= 0 or risk_cap <= 0:
        return base, None

    effective_floor = min(configured_floor, risk_cap)

    info: Dict[str, object] = {
        "configured_floor": configured_floor,
        "effective_floor": effective_floor,
        "base_units_before": base,
        "risk_cap": risk_cap,
        "margin_cap": margin_cap,
    }

    if configured_floor > risk_cap:
        info["clipped_by"] = "risk"

    if base >= effective_floor:
        if info.get("clipped_by"):
            info["applied"] = False
            return base, info
        return base, None

    adjusted = min(effective_floor, margin_cap)
    info["applied"] = True
    info["adjusted_units"] = adjusted
    if adjusted < effective_floor:
        info["clipped_by"] = "margin"

    return adjusted, info
