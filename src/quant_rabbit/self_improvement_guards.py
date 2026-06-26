"""Shared self-improvement gates for fresh entry risk.

These helpers convert audit diagnoses into executable no-new-risk blockers.
They also centralize the narrow TP_HARVEST_REPAIR exceptions so the verifier,
gateway, and support surfaces do not disagree about which P0 a repair basket
is allowed to address.
"""

from __future__ import annotations

import math
from typing import Any


# Three repeated audit runs is the smallest persistence count that distinguishes
# a system-level loop from a single noisy diagnosis. This mirrors the
# self-improvement audit's repeated-repair-loop threshold and is not a market
# parameter.
PERSISTENT_ROOT_CAUSE_STREAK_MIN = 3

FORECAST_ADVERSE_PATH_BLOCKER_CODE = "SELF_IMPROVEMENT_FORECAST_ADVERSE_PATH"
FORECAST_ADVERSE_PATH_FAMILY = "FORECAST_ADVERSE_PATH"
FORECAST_ADVERSE_PATH_REPAIR_MODE = "TP_HARVEST_REPAIR"
TP_HARVEST_REPAIR_MIN_TRADES = 20
FORECAST_ADVERSE_SUPPORT_CODES = frozenset(
    {
        "DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK",
        "DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
        "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
        "PROJECTION_ECONOMIC_PRECISION_WEAK",
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
PROFIT_CAPTURE_MISS_CODE = "LOSS_CLOSE_PROFIT_CAPTURE_MISSED"
TP_HARVEST_REPAIR_EXEMPT_P0_CODES = frozenset(
    {
        PROFITABILITY_DISCIPLINE_BLOCKED_CODE,
        PROFIT_CAPTURE_MISS_CODE,
    }
)
OANDA_CAMPAIGN_FIREPOWER_REPAIR_MODE = "OANDA_CAMPAIGN_FIREPOWER_HARVEST"
OANDA_CAMPAIGN_CURRENT_RISK_UNDERPOWERED_BASIS = (
    "OANDA_CAMPAIGN_FIREPOWER_CURRENT_RISK_UNDERPOWERED"
)
OANDA_CAMPAIGN_NORMAL_CAP_WEIGHTED_PACE_BASIS = (
    "OANDA_CAMPAIGN_FIREPOWER_NORMAL_CAP_WEIGHTED_PACE"
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
    projection_gap_count = _optional_int(metrics.get("projection_economic_precision_gap_count"))
    projection_worst_lower = _optional_float(metrics.get("projection_worst_economic_wilson_lower"))
    projection_timeout_rate = _optional_float(metrics.get("projection_worst_timeout_rate"))
    profit_factor = _optional_float(metrics.get("profit_factor"))
    if directional_hit_rate is not None:
        details.append(f"directional_hit_rate={directional_hit_rate:.3f}")
    if invalidation_first_rate is not None:
        details.append(f"invalidation_first_rate={invalidation_first_rate:.3f}")
    if projection_gap_count is not None:
        details.append(f"projection_economic_precision_gap_count={projection_gap_count}")
    if projection_worst_lower is not None:
        details.append(f"projection_worst_economic_wilson_lower={projection_worst_lower:.3f}")
    if projection_timeout_rate is not None:
        details.append(f"projection_worst_timeout_rate={projection_timeout_rate:.3f}")
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
        "projection_economic_precision_gap_count": projection_gap_count,
        "projection_worst_economic_wilson_lower": projection_worst_lower,
        "projection_worst_timeout_rate": projection_timeout_rate,
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
    normal_cap_reachable = _oanda_firepower_normal_cap_weighted_pace_reaches_minimum(
        metadata,
        basis=basis,
    )
    if basis == OANDA_CAMPAIGN_CURRENT_RISK_UNDERPOWERED_BASIS and not normal_cap_reachable:
        return False
    current_risk_reachable = metadata.get(
        "positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable"
    )
    if current_risk_reachable is False and not normal_cap_reachable:
        return False
    return True


def p0_code_exempted_by_tp_harvest_repair(code: str, *, p0_repair_selected: bool) -> bool:
    if not p0_repair_selected:
        return False
    return str(code or "").strip() in TP_HARVEST_REPAIR_EXEMPT_P0_CODES


def tp_harvest_forecast_adverse_path_repair_shape(
    intent_or_lane: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Return true for the exact TP-proven HARVEST lane that may repair forecast P1.

    This does not bypass market-structure, spread, risk, broker-truth, or gateway
    gates. It only identifies the already broker-TP-proven non-market shape whose
    forecast-adverse-path blocker can be downgraded while those gates still decide
    live eligibility.
    """

    if not isinstance(intent_or_lane, dict):
        return False
    intent_payload = _intent_payload(intent_or_lane)
    meta = _metadata_from_intent_or_lane(intent_payload, metadata)
    if str(intent_payload.get("order_type") or "").strip().upper() == "MARKET":
        return False
    pair = str(intent_payload.get("pair") or "").strip()
    side = str(
        intent_payload.get("side")
        or intent_payload.get("direction")
        or ""
    ).strip().upper()
    method = _intent_or_lane_method(intent_payload)
    if not pair or not side or method != "BREAKOUT_FAILURE":
        return False
    forecast_direction = str(meta.get("forecast_direction") or "").strip().upper()
    if forecast_direction and forecast_direction != "RANGE":
        return False
    if str(meta.get("position_intent") or "NEW").strip().upper() == "HEDGE":
        return False
    if meta.get("attach_take_profit_on_fill") is not True:
        return False
    if str(meta.get("tp_execution_mode") or "").strip().upper() != "ATTACHED_TECHNICAL_TP":
        return False
    if str(meta.get("tp_target_intent") or "").strip().upper() != "HARVEST":
        return False
    if str(meta.get("opportunity_mode") or "").strip().upper() != "HARVEST":
        return False
    if str(meta.get("positive_rotation_mode") or "").strip().upper() != "TP_PROVEN_HARVEST":
        return False
    if meta.get("positive_rotation_live_ready") is not True:
        return False
    if str(meta.get("capture_take_profit_scope") or "").strip().upper() != "PAIR_SIDE_METHOD":
        return False
    expected_scope = f"{pair}|{side}|{method}|TAKE_PROFIT_ORDER".upper()
    if str(meta.get("capture_take_profit_scope_key") or "").strip().upper() != expected_scope:
        return False
    tp_trades = _optional_int(meta.get("capture_take_profit_trades")) or 0
    if tp_trades < TP_HARVEST_REPAIR_MIN_TRADES:
        return False
    tp_losses = _optional_int(meta.get("capture_take_profit_losses")) or 0
    if tp_losses != 0:
        return False
    tp_expectancy = _optional_float(meta.get("capture_take_profit_expectancy_jpy"))
    if tp_expectancy is None or tp_expectancy <= 0:
        return False
    pessimistic = _optional_float(meta.get("positive_rotation_pessimistic_expectancy_jpy"))
    if pessimistic is None or pessimistic <= 0:
        return False
    return True


def forecast_adverse_path_exempted_by_tp_harvest_repair(
    intent_or_lane: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    meta = _metadata_from_intent_or_lane(intent_or_lane, metadata)
    if meta.get("self_improvement_forecast_adverse_path_repair_live_ready") is not True:
        return False
    if (
        str(meta.get("self_improvement_forecast_adverse_path_repair_mode") or "").strip()
        != FORECAST_ADVERSE_PATH_REPAIR_MODE
    ):
        return False
    if (
        str(meta.get("self_improvement_forecast_adverse_path_repair_blocker_code") or "").strip()
        != FORECAST_ADVERSE_PATH_BLOCKER_CODE
    ):
        return False
    return tp_harvest_forecast_adverse_path_repair_shape(intent_or_lane, meta)


def _oanda_firepower_normal_cap_weighted_pace_reaches_minimum(
    metadata: dict[str, Any],
    *,
    basis: str,
) -> bool:
    if basis != OANDA_CAMPAIGN_NORMAL_CAP_WEIGHTED_PACE_BASIS:
        return False
    if (
        metadata.get("positive_rotation_oanda_campaign_normal_cap_minimum_floor_reachable")
        is not True
    ):
        return False
    required = _optional_float(
        metadata.get("positive_rotation_oanda_campaign_normal_cap_required_minimum_trades")
    )
    target = _optional_float(
        metadata.get("positive_rotation_oanda_campaign_normal_cap_target_trades_per_day")
    )
    observed = _optional_float(
        metadata.get("positive_rotation_oanda_campaign_normal_cap_observed_attempts_per_day")
    )
    weighted_return = _optional_float(
        metadata.get("positive_rotation_oanda_campaign_normal_cap_weighted_return_pct_per_trade")
    )
    if (
        required is None
        or target is None
        or observed is None
        or weighted_return is None
        or required < 0
        or target <= 0
        or observed < 0
        or weighted_return <= 0
    ):
        return False
    return required == 0 or (required <= target and required <= math.floor(observed))


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


def _metadata_from_intent_or_lane(
    intent_or_lane: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata
    if not isinstance(intent_or_lane, dict):
        return {}
    intent_payload = _intent_payload(intent_or_lane)
    raw = intent_payload.get("metadata")
    if isinstance(raw, dict):
        return raw
    raw = intent_payload.get("self_improvement")
    if isinstance(raw, dict):
        return raw
    return {}


def _intent_payload(intent_or_lane: dict[str, Any]) -> dict[str, Any]:
    raw = intent_or_lane.get("intent")
    if isinstance(raw, dict):
        return raw
    return intent_or_lane


def _intent_or_lane_method(intent_or_lane: dict[str, Any]) -> str:
    intent_payload = _intent_payload(intent_or_lane)
    method = str(intent_payload.get("method") or "").strip().upper()
    if method:
        return method
    context = intent_payload.get("market_context")
    if isinstance(context, dict):
        return str(context.get("method") or "").strip().upper()
    return ""


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
