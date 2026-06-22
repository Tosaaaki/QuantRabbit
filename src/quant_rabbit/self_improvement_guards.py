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

PENDING_EXECUTION_LIFECYCLE_BLOCKER_CODE = "SELF_IMPROVEMENT_PENDING_EXECUTION_LIFECYCLE"
PENDING_EXECUTION_LIFECYCLE_FAMILY = "EXECUTION_LIFECYCLE"
PENDING_EXECUTION_LIFECYCLE_SUPPORT_CODES = frozenset(
    {
        "PENDING_ENTRY_CANCEL_RATE_HIGH",
        "PENDING_ENTRY_FILL_RATE_WEAK",
    }
)

PROFITABILITY_DISCIPLINE_BLOCKED_CODE = "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"
OANDA_CAMPAIGN_FIREPOWER_REPAIR_MODE = "OANDA_CAMPAIGN_FIREPOWER_HARVEST"
OANDA_CAMPAIGN_CURRENT_RISK_UNDERPOWERED_BASIS = (
    "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED"
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


def pending_execution_lifecycle_new_risk_blocker(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a fresh-pending-entry blocker for repeated fill/cancel churn.

    The audit may keep `PENDING_ENTRY_CANCEL_RATE_HIGH` at P1 because existing
    gateway safety gates are intact. Once the root-cause ranker sees the same
    execution lifecycle loop across repeated audit runs, fresh pending entries
    are the code-owned leak: they keep creating broker-anchored orders that are
    soon canceled before thesis invalidation is separated from TTL/entry drift.
    """

    if not isinstance(payload, dict):
        return None
    root_focus = payload.get("root_cause_focus")
    if not isinstance(root_focus, dict):
        return None
    candidates: list[dict[str, Any]] = []
    primary = root_focus.get("primary")
    if isinstance(primary, dict):
        candidates.append(primary)
    candidates.extend(item for item in root_focus.get("candidates", []) or [] if isinstance(item, dict))
    for candidate in candidates:
        if str(candidate.get("family") or "").upper() != PENDING_EXECUTION_LIFECYCLE_FAMILY:
            continue
        if str(candidate.get("confidence") or "").upper() != "HIGH":
            continue
        streak = _optional_int(candidate.get("process_loop_streak"))
        if streak is None or streak < PERSISTENT_ROOT_CAUSE_STREAK_MIN:
            continue
        supporting_codes = {str(code or "") for code in candidate.get("supporting_codes", []) or []}
        matched_codes = sorted(supporting_codes.intersection(PENDING_EXECUTION_LIFECYCLE_SUPPORT_CODES))
        if not matched_codes:
            continue
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        details: list[str] = [f"streak={streak}"]
        cancel_rate = _optional_float(metrics.get("pending_cancel_before_fill_rate"))
        fill_rate = _optional_float(metrics.get("pending_fill_rate"))
        profit_factor = _optional_float(metrics.get("profit_factor"))
        if cancel_rate is not None:
            details.append(f"cancel_before_fill_rate={cancel_rate:.3f}")
        if fill_rate is not None:
            details.append(f"fill_rate={fill_rate:.3f}")
        if profit_factor is not None:
            details.append(f"PF={profit_factor:.3f}")
        suffix = f" ({', '.join(details)})"
        next_action = str(candidate.get("next_action") or "").strip()
        churn_lane_keys = _pending_churn_lane_keys(payload)
        return {
            "code": PENDING_EXECUTION_LIFECYCLE_BLOCKER_CODE,
            "layer": "execution_quality",
            "message": (
                "persistent pending-entry fill/cancel churn blocks fresh pending-entry generation; "
                "repair entry distance/TTL versus thesis-invalidation separation before arming more orders"
                f"{suffix}"
            ),
            "next_action": next_action,
            "current_streak": streak,
            "pending_cancel_before_fill_rate": cancel_rate,
            "pending_fill_rate": fill_rate,
            "profit_factor": profit_factor,
            "supporting_codes": matched_codes,
            "cancel_churn_lane_keys": churn_lane_keys,
            "evidence_ref": f"self_improvement:root_cause:{PENDING_EXECUTION_LIFECYCLE_FAMILY}",
        }
    return None


def profitability_p0_worst_segment(payload: dict[str, Any] | None) -> dict[str, str] | None:
    """Return the current profitability-P0 worst segment, if the audit names one."""

    if not isinstance(payload, dict):
        return None
    candidates: list[dict[str, Any]] = []
    for finding in payload.get("findings", []) or []:
        if not isinstance(finding, dict):
            continue
        if str(finding.get("priority") or "").upper() != "P0":
            continue
        if str(finding.get("code") or "") != PROFITABILITY_DISCIPLINE_BLOCKED_CODE:
            continue
        if str(finding.get("layer") or "") not in {"", "profitability"}:
            continue
        evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
        system = (
            evidence.get("system_defect_evidence")
            if isinstance(evidence.get("system_defect_evidence"), dict)
            else {}
        )
        candidates.extend(item for item in system.get("worst_segments", []) or [] if isinstance(item, dict))
    for blocker in payload.get("profitability_blockers", []) or []:
        if not isinstance(blocker, dict):
            continue
        if str(blocker.get("code") or "") != PROFITABILITY_DISCIPLINE_BLOCKED_CODE:
            continue
        candidates.extend(item for item in blocker.get("worst_segments", []) or [] if isinstance(item, dict))
    for segment in candidates:
        normalized = _normalize_profitability_segment(segment)
        if normalized is not None:
            return normalized
    return None


def intent_matches_profitability_worst_segment(
    intent_or_lane: dict[str, Any] | None,
    worst_segment: dict[str, str] | None,
) -> bool:
    """True when an intent/lane targets the audit's named bad pair/side/method."""

    if not isinstance(intent_or_lane, dict) or not isinstance(worst_segment, dict):
        return False
    pair = str(intent_or_lane.get("pair") or "").strip()
    side = str(intent_or_lane.get("side") or intent_or_lane.get("direction") or "").strip().upper()
    method = str(intent_or_lane.get("method") or "").strip().upper()
    context = intent_or_lane.get("market_context")
    if not method and isinstance(context, dict):
        method = str(context.get("method") or "").strip().upper()
    if not pair or not side or not method:
        return False
    return (
        pair == worst_segment.get("pair")
        and side == worst_segment.get("side")
        and method == worst_segment.get("method")
    )


def oanda_firepower_repair_current_risk_reaches_minimum(
    metadata: dict[str, Any] | None,
) -> bool:
    """Require current-risk 5% firepower before OANDA-only P0 repair escapes.

    Local TP-proven repair lanes are judged by realized broker TP evidence.
    OANDA campaign firepower is historical audit evidence, so when a live
    daily-target state lets us scale that audit lens to the executable order
    risk, an underpowered result must not bypass a profitability P0.
    """

    if not isinstance(metadata, dict):
        return True
    mode = str(metadata.get("positive_rotation_mode") or "").strip().upper()
    if mode != OANDA_CAMPAIGN_FIREPOWER_REPAIR_MODE:
        return True
    if metadata.get("positive_rotation_minimum_floor_reachable") is not True:
        return False
    basis = str(metadata.get("positive_rotation_minimum_floor_reach_basis") or "").strip().upper()
    if basis == OANDA_CAMPAIGN_CURRENT_RISK_UNDERPOWERED_BASIS:
        return False
    current_risk_reachable = metadata.get(
        "positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable"
    )
    if current_risk_reachable is False:
        return False
    return True


def _pending_churn_lane_keys(payload: dict[str, Any]) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for finding in payload.get("findings", []) or []:
        if not isinstance(finding, dict):
            continue
        if str(finding.get("code") or "") not in PENDING_EXECUTION_LIFECYCLE_SUPPORT_CODES:
            continue
        evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
        for group_name in (
            "canceled_before_fill_orphan_groups",
            "canceled_before_fill_replaced_groups",
        ):
            for group in evidence.get(group_name, []) or []:
                if not isinstance(group, dict):
                    continue
                pair = str(group.get("pair") or "").strip()
                side = str(group.get("side") or "").strip().upper()
                method = str(group.get("method") or "").strip().upper()
                if not pair or not side or not method:
                    continue
                key = (pair, side, method)
                if key in seen:
                    continue
                seen.add(key)
                keys.append(
                    {
                        "pair": pair,
                        "side": side,
                        "method": method,
                        "count": _optional_int(group.get("count")),
                        "source": group_name,
                    }
                )
    return keys


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_profitability_segment(segment: dict[str, Any]) -> dict[str, str] | None:
    pair = str(segment.get("pair") or "").strip()
    side = str(segment.get("side") or "").strip().upper()
    method = str(segment.get("method") or "").strip().upper()
    if not pair or not side or not method:
        return None
    out = {"pair": pair, "side": side, "method": method}
    label_parts = [f"pair={pair}", f"side={side}", f"method={method}"]
    trades = segment.get("trades")
    if trades is not None:
        label_parts.append(f"trades={trades}")
    net = _optional_float(segment.get("net_jpy"))
    if net is not None:
        label_parts.append(f"net={net:.2f} JPY")
    trade_ids = [str(item) for item in segment.get("trade_ids", []) or [] if str(item)]
    if trade_ids:
        label_parts.append(f"trade_ids={','.join(trade_ids[:5])}")
    out["label"] = "data/execution_ledger.db worst_segment[" + ", ".join(label_parts) + "]"
    return out


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
