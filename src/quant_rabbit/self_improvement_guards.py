"""Shared self-improvement gates for fresh entry risk.

These helpers convert audit diagnoses into executable no-new-risk blockers.
They do not relax any P0 gate; they only hard-block a repeated, high-confidence
root-cause diagnosis before it can leak through a lower layer.
"""

from __future__ import annotations

from typing import Any


# Three repeated audit runs is the smallest persistence count that distinguishes
# a system-level loop from a single noisy diagnosis. This mirrors the
# self-improvement audit's repeated-repair-loop threshold and is not a market
# parameter.
PERSISTENT_ROOT_CAUSE_STREAK_MIN = 3

FORECAST_ADVERSE_PATH_BLOCKER_CODE = "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
FORECAST_ADVERSE_PATH_FAMILY = "FORECAST_ADVERSE_PATH"
FORECAST_ADVERSE_SUPPORT_CODES = frozenset(
    {
        "DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK",
        "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
        "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
    }
)


def forecast_adverse_path_new_risk_blocker(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a fresh-entry blocker for a persistent forecast adverse path.

    The audit can rank `FORECAST_ADVERSE_PATH` as P1 because the hard safety
    gates are intact, but a high-confidence repeated adverse-path diagnosis is
    still a code-owned new-risk problem: the system must not keep advertising
    or staging fresh entries from a forecast family whose recent hit path is
    dominated by invalidation-first outcomes.
    """

    if not isinstance(payload, dict):
        return None
    root_focus = payload.get("root_cause_focus")
    if not isinstance(root_focus, dict):
        return None
    primary = root_focus.get("primary")
    if not isinstance(primary, dict):
        return None
    if str(primary.get("family") or "").upper() != FORECAST_ADVERSE_PATH_FAMILY:
        return None
    if str(primary.get("confidence") or "").upper() != "HIGH":
        return None
    streak = _optional_int(primary.get("process_loop_streak"))
    if streak is None or streak < PERSISTENT_ROOT_CAUSE_STREAK_MIN:
        return None
    supporting_codes = {str(code or "") for code in primary.get("supporting_codes", []) or []}
    if not supporting_codes.intersection(FORECAST_ADVERSE_SUPPORT_CODES):
        return None

    metrics = primary.get("metrics") if isinstance(primary.get("metrics"), dict) else {}
    details: list[str] = [f"streak={streak}"]
    directional_hit_rate = _optional_float(metrics.get("directional_hit_rate"))
    invalidation_first_rate = _optional_float(metrics.get("invalidation_first_rate"))
    profit_factor = _optional_float(metrics.get("profit_factor"))
    if directional_hit_rate is not None:
        details.append(f"directional_hit_rate={directional_hit_rate:.3f}")
    if invalidation_first_rate is not None:
        details.append(f"invalidation_first_rate={invalidation_first_rate:.3f}")
    if profit_factor is not None:
        details.append(f"PF={profit_factor:.3f}")
    suffix = f" ({', '.join(details)})"
    next_action = str(primary.get("next_action") or "").strip()
    message = (
        "persistent high-confidence forecast adverse path blocks new entry risk; "
        "repair directional forecast buckets/range-location priors before expanding exposure"
        f"{suffix}"
    )
    return {
        "code": FORECAST_ADVERSE_PATH_BLOCKER_CODE,
        "layer": "forecast",
        "message": message,
        "next_action": next_action,
        "current_streak": streak,
        "directional_hit_rate": directional_hit_rate,
        "invalidation_first_rate": invalidation_first_rate,
        "profit_factor": profit_factor,
        "supporting_codes": sorted(supporting_codes.intersection(FORECAST_ADVERSE_SUPPORT_CODES)),
        "evidence_ref": f"self_improvement:root_cause:{FORECAST_ADVERSE_PATH_FAMILY}",
    }


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
