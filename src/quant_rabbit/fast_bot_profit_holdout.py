"""Pre-registered, zero-authority profit holdout for fast-bot signals.

The raw fast-bot ledger intentionally retains every diagnostic ``GO`` row.
This module builds a separate prospective cohort containing at most one
pre-registered, non-overlapping signal per cycle.  It never creates units,
order intents, broker requests, live permission, or automatic adoption.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.fast_bot import SHADOW_CONTRACT
from quant_rabbit.fast_bot_profitability_gate import (
    DEFAULT_THRESHOLDS,
    assess_profitability_evidence,
    build_profitability_evidence,
)
from quant_rabbit.fast_bot_truth import (
    OUTCOME_CONTRACT,
    SCORECARD_CONTRACT as TRUTH_SCORECARD_CONTRACT,
    _fast_bot_outcome_valid_for_signal,
    _fast_bot_signal_valid,
    build_fast_bot_scorecard,
)


POLICY_CONTRACT_V1 = "QR_FAST_BOT_PROFIT_HOLDOUT_POLICY_V1"
POLICY_CONTRACT_V2 = "QR_FAST_BOT_PROFIT_HOLDOUT_POLICY_V2"
POLICY_CONTRACT_V3 = "QR_FAST_BOT_PROFIT_HOLDOUT_POLICY_V3"
DECISION_CONTRACT_V1 = "QR_FAST_BOT_PROFIT_HOLDOUT_DECISION_V1"
DECISION_CONTRACT_V2 = "QR_FAST_BOT_PROFIT_HOLDOUT_DECISION_V2"
DECISION_CONTRACT_V3 = "QR_FAST_BOT_PROFIT_HOLDOUT_DECISION_V3"
SCORECARD_CONTRACT_V1 = "QR_FAST_BOT_PROFIT_HOLDOUT_SCORECARD_V1"
SCORECARD_CONTRACT_V2 = "QR_FAST_BOT_PROFIT_HOLDOUT_SCORECARD_V2"
SCORECARD_CONTRACT_V3 = "QR_FAST_BOT_PROFIT_HOLDOUT_SCORECARD_V3"

# Backward-compatible public aliases.  V1 remains readable as immutable history;
# the active resident explicitly selects the separately sealed V2 policy.
POLICY_CONTRACT = POLICY_CONTRACT_V1
DECISION_CONTRACT = DECISION_CONTRACT_V1
SCORECARD_CONTRACT = SCORECARD_CONTRACT_V1
SELECTION_POLICY = "ONE_PRECOMMITTED_NONOVERLAPPING_LANE_PER_CYCLE"
NO_ACTIVE_CANDIDATE_SELECTION_POLICY = "NO_ACTIVE_CANDIDATE_FAIL_CLOSED"
SIGNAL_FILTER_POLICY_V2 = "PASSIVE_NEAR_SIDE_M5_ATR_FLOOR_V1"
# This is a frozen, post-hoc research hypothesis threshold, not a risk limit or
# profitability proof.  It may change only in another versioned future cohort.
V2_MINIMUM_M5_ATR_PIPS = 5.0
MAX_SOURCE_SIGNALS = 128
MAX_PENDING_CYCLES_PER_RUN = 64
LANE_FIELDS = ("pair", "side", "method", "horizon_lane")


def canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": canonical_sha(body)}


def sealed_valid(value: Mapping[str, Any], contract: str) -> bool:
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return value.get("contract") == contract and stored == canonical_sha(body)


def load_policy(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profit holdout policy must be a JSON object")
    validate_policy(value)
    if value.get("contract") == POLICY_CONTRACT_V3:
        _validate_v3_audit_artifact(value)
    return value, canonical_sha(value)


def validate_policy(policy: Mapping[str, Any]) -> None:
    contract = policy.get("contract")
    schema_version = policy.get("schema_version")
    if isinstance(schema_version, bool) or (contract, schema_version) not in {
        (POLICY_CONTRACT_V1, 1),
        (POLICY_CONTRACT_V2, 2),
        (POLICY_CONTRACT_V3, 3),
    }:
        raise ValueError("profit holdout policy contract mismatch")
    authority = policy.get("authority")
    if not isinstance(authority, Mapping) or (
        authority.get("execution_authority") != "NONE"
        or authority.get("broker_http_methods_allowed") != ["GET"]
        or authority.get("broker_mutation_allowed") is not False
        or authority.get("automatic_adoption_allowed") is not False
        or authority.get("promotion_allowed") is not False
        or authority.get("live_permission") is not False
        or authority.get("external_order_attempts") != 0
        or authority.get("external_orders") != 0
        or authority.get("manual_tagless_policy") != "NO_TOUCH"
    ):
        raise ValueError("profit holdout authority boundary mismatch")
    selection = policy.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("profit holdout selection contract mismatch")
    if contract == POLICY_CONTRACT_V3:
        if (
            selection.get("selection_policy")
            != NO_ACTIVE_CANDIDATE_SELECTION_POLICY
            or selection.get("maximum_selected_per_cycle") != 0
            or selection.get("maximum_concurrent_per_pair_horizon") != 0
            or selection.get("reservation_seconds") != 0
            or selection.get("maximum_selection_delay_seconds") != 45
            or selection.get("unknown_lane_policy") != "REJECT"
            or selection.get("equal_priority_policy") != "REJECT_ALL_TIED_TOP_PRIORITY"
            or selection.get("opposite_side_policy")
            != "REJECT_CYCLE_ON_SAME_PAIR_HORIZON_OPPOSITE_GO"
            or selection.get("post_outcome_reranking_allowed") is not False
            or selection.get("candidate_admission_status")
            != "NO_ADMISSIBLE_CANDIDATE"
            or selection.get("new_candidate_requires_new_policy") is not True
        ):
            raise ValueError("V3 no-candidate selection contract mismatch")
    elif (
        selection.get("selection_policy") != SELECTION_POLICY
        or selection.get("maximum_selected_per_cycle") != 1
        or selection.get("maximum_concurrent_per_pair_horizon") != 1
        or selection.get("reservation_seconds") != 990
        or selection.get("maximum_selection_delay_seconds") != 45
        or selection.get("unknown_lane_policy") != "REJECT"
        or selection.get("equal_priority_policy") != "REJECT_ALL_TIED_TOP_PRIORITY"
        or selection.get("opposite_side_policy")
        != "REJECT_CYCLE_ON_SAME_PAIR_HORIZON_OPPOSITE_GO"
        or selection.get("post_outcome_reranking_allowed") is not False
    ):
        raise ValueError("profit holdout selection contract mismatch")
    lanes = selection.get("allowed_lanes")
    expected_lane_count = 0 if contract == POLICY_CONTRACT_V3 else 1
    if not isinstance(lanes, list) or len(lanes) != expected_lane_count:
        raise ValueError(
            "V3 profit holdout requires no active lane"
            if contract == POLICY_CONTRACT_V3
            else "profit holdout requires exactly one precommitted lane"
        )
    lane_ids: set[tuple[str, str, str, str]] = set()
    for row in lanes:
        if not isinstance(row, Mapping):
            raise ValueError("profit holdout lane must be an object")
        lane = _lane(row)
        priority = row.get("priority")
        if (
            any(not item for item in lane)
            or lane in lane_ids
            or isinstance(priority, bool)
            or not isinstance(priority, int)
            or priority <= 0
            or row.get("candidate_status") != "UNPROVEN_PROSPECTIVE_CANDIDATE"
        ):
            raise ValueError("profit holdout lane identity or priority mismatch")
        lane_ids.add(lane)
    if contract == POLICY_CONTRACT_V1:
        if "signal_filter" in selection or "supersession" in policy:
            raise ValueError("V1 profit holdout cannot contain V2 fields")
    elif contract == POLICY_CONTRACT_V2:
        _validate_v2_signal_filter(selection.get("signal_filter"))
        supersession = policy.get("supersession")
        if not isinstance(supersession, Mapping) or (
            supersession.get("supersedes_policy_id")
            != "usdjpy-short-range-rotation-prospective-v1"
            or supersession.get("prior_policy_status")
            != "RETIRED_ZERO_ELIGIBLE_SELECTIONS"
            or supersession.get("prior_rows_admitted") is not False
            or supersession.get("replacement_reason")
            != "ZERO_ELIGIBLE_SELECTIONS_AND_NEGATIVE_POST_CUTOFF_DIAGNOSTIC_LANE"
            or supersession.get("single_factor_changed")
            != "M5_ATR_PIPS_MINIMUM"
            or supersession.get("reward_risk_changed") is not False
        ):
            raise ValueError("V2 profit holdout supersession contract mismatch")
        screened = supersession.get("prior_policy_decisions_screened")
        diagnostic_fills = supersession.get(
            "post_cutoff_diagnostic_lane_filled_signals"
        )
        if (
            isinstance(screened, bool)
            or not isinstance(screened, int)
            or screened <= 0
            or supersession.get("prior_policy_selected_signals") != 0
            or supersession.get("prior_policy_resolved_signals") != 0
            or not _valid_sha(
                str(supersession.get("prior_policy_last_decision_sha256") or "")
            )
            or not _valid_sha(
                str(supersession.get("prior_policy_source_bundle_sha256") or "")
            )
            or not _valid_commit(
                str(supersession.get("prior_policy_source_commit") or "")
            )
            or not isinstance(diagnostic_fills, int)
            or isinstance(diagnostic_fills, bool)
            or diagnostic_fills <= 0
            or supersession.get("post_cutoff_diagnostic_rows_admitted") is not False
            or float(supersession.get("post_cutoff_diagnostic_lane_net_pips") or 0.0)
            >= 0.0
            or float(
                supersession.get("post_cutoff_diagnostic_lane_profit_factor")
                or 0.0
            )
            >= 1.0
        ):
            raise ValueError("V2 profit holdout retirement evidence mismatch")
        _parse_utc(supersession.get("prior_policy_observed_at_utc"))
    else:
        _validate_v3_no_candidate_policy(policy)
    holdout = policy.get("holdout")
    training = policy.get("training_evidence")
    if not isinstance(holdout, Mapping) or not isinstance(training, Mapping):
        raise ValueError("profit holdout timing and training evidence are required")
    frozen_at = _parse_utc(holdout.get("frozen_at_utc"))
    eligible_after = _parse_utc(holdout.get("eligible_after_utc"))
    training_at = _parse_utc(training.get("generated_at_utc"))
    expected_claim = (
        "NO_ADMISSIBLE_CANDIDATE"
        if contract == POLICY_CONTRACT_V3
        else "UNPROVEN"
    )
    if (
        holdout.get("cohort_policy") != "STRICTLY_AFTER_ELIGIBLE_AFTER_UTC"
        or holdout.get("retroactive_signal_admission_allowed") is not False
        or not training_at <= frozen_at <= eligible_after
        or not _valid_sha(str(training.get("source_scorecard_contract_sha256") or ""))
        or not _valid_sha(str(training.get("source_scorecard_file_sha256") or ""))
        or training.get("forward_evidence_passed") is not False
        or training.get("profitability_claim") != expected_claim
    ):
        raise ValueError("profit holdout evidence boundary mismatch")
    thresholds = policy.get("acceptance_thresholds")
    if not isinstance(thresholds, Mapping) or any(
        thresholds.get(key) != value for key, value in DEFAULT_THRESHOLDS.items()
    ):
        raise ValueError("profit holdout acceptance thresholds mismatch")


def _validate_v3_no_candidate_policy(policy: Mapping[str, Any]) -> None:
    supersession = policy.get("supersession")
    admission = policy.get("candidate_admission")
    if not isinstance(supersession, Mapping) or (
        supersession.get("supersedes_policy_id")
        != "usdjpy-short-range-rotation-m5-atr-gte-5-prospective-v2"
        or supersession.get("prior_policy_status")
        != "RETIRED_INADMISSIBLE_POST_HOC_SLICE"
        or supersession.get("prior_rows_admitted") is not False
        or supersession.get("replacement_reason")
        != "CROSS_RESIDENT_AUDIT_FOUND_NO_ADMISSIBLE_CANDIDATE"
        or supersession.get("prior_policy_selected_signals") != 0
        or supersession.get("prior_policy_resolved_signals") != 0
        or supersession.get("automatic_replacement_candidate_allowed") is not False
        or not _valid_sha(str(supersession.get("prior_policy_sha256") or ""))
        or not _valid_sha(str(supersession.get("prior_scorecard_sha256") or ""))
    ):
        raise ValueError("V3 profit holdout supersession contract mismatch")
    _parse_utc(supersession.get("prior_policy_observed_at_utc"))
    if not isinstance(admission, Mapping) or (
        admission.get("audit_contract")
        != "QR_FAST_BOT_RESIDENT_PROFIT_CANDIDATE_AUDIT_V1"
        or admission.get("audit_status") != "NO_ADMISSIBLE_CANDIDATE"
        or admission.get("source_integrity_passed") is not True
        or admission.get("research_lead_count") != 0
        or admission.get("automatic_candidate_activation_allowed") is not False
        or admission.get("historical_rows_admitted") is not False
        or not _valid_sha(str(admission.get("audit_contract_sha256") or ""))
        or not _valid_sha(str(admission.get("audit_file_sha256") or ""))
        or not _valid_sha(str(admission.get("source_bundle_sha256") or ""))
        or not Path(str(admission.get("source_artifact") or "")).is_absolute()
    ):
        raise ValueError("V3 candidate admission audit mismatch")
    candidate_count = admission.get("candidate_count")
    unique_signals = admission.get("unique_sealed_signals")
    valid_outcomes = admission.get("unique_valid_outcomes")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
        or isinstance(unique_signals, bool)
        or not isinstance(unique_signals, int)
        or unique_signals <= 0
        or isinstance(valid_outcomes, bool)
        or not isinstance(valid_outcomes, int)
        or not 0 < valid_outcomes <= unique_signals
    ):
        raise ValueError("V3 candidate admission counts mismatch")
    v2 = admission.get("v2_candidate_reassessment")
    if not isinstance(v2, Mapping) or (
        v2.get("filled_signals") != 3
        or v2.get("active_days") != 1
        or float(v2.get("net_pips") or 0.0) != 2.2
        or float(v2.get("maximum_daily_sample_share") or 0.0) != 1.0
        or float(v2.get("pessimistic_expectancy_pips") or 0.0) >= 0.0
        or v2.get("admission_passed") is not False
    ):
        raise ValueError("V3 prior candidate reassessment mismatch")


def _validate_v3_audit_artifact(policy: Mapping[str, Any]) -> None:
    admission = policy["candidate_admission"]
    path = Path(str(admission["source_artifact"]))
    try:
        data = path.read_bytes()
        audit = json.loads(data)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("V3 candidate audit artifact is unavailable") from exc
    audit_v2 = audit.get("v2_candidate_reassessment") if isinstance(audit, Mapping) else None
    audit_v2_metrics = (
        audit_v2.get("metrics") if isinstance(audit_v2, Mapping) else None
    )
    configured_v2 = admission["v2_candidate_reassessment"]
    near = (
        (audit.get("aggregate_entry_arms") or {}).get("PASSIVE_NEAR_SIDE")
        if isinstance(audit, Mapping)
        else None
    )
    training = policy["training_evidence"]
    if not isinstance(audit, Mapping) or (
        hashlib.sha256(data).hexdigest() != admission["audit_file_sha256"]
        or not sealed_valid(
            audit,
            "QR_FAST_BOT_RESIDENT_PROFIT_CANDIDATE_AUDIT_V1",
        )
        or audit.get("contract_sha256") != admission["audit_contract_sha256"]
        or audit.get("source_bundle_sha256") != admission["source_bundle_sha256"]
        or audit.get("status") != admission["audit_status"]
        or audit.get("source_integrity_passed")
        != admission["source_integrity_passed"]
        or audit.get("research_lead_count") != admission["research_lead_count"]
        or audit.get("unique_sealed_signals")
        != admission["unique_sealed_signals"]
        or audit.get("unique_valid_outcomes")
        != admission["unique_valid_outcomes"]
        or (audit.get("candidate_universe") or {}).get("candidate_count")
        != admission["candidate_count"]
        or audit.get("automatic_candidate_activation_allowed") is not False
        or audit.get("execution_authority") != "NONE"
        or audit.get("broker_mutation_allowed") is not False
        or audit.get("live_permission") is not False
        or audit.get("external_order_attempts") != 0
        or audit.get("external_orders") != 0
        or not isinstance(audit_v2, Mapping)
        or not isinstance(audit_v2_metrics, Mapping)
        or audit_v2.get("admission_passed")
        != configured_v2["admission_passed"]
        or any(
            audit_v2_metrics.get(key) != configured_v2[key]
            for key in (
                "filled_signals",
                "active_days",
                "resolved_signals",
                "wins",
                "losses",
                "net_pips",
                "profit_factor",
                "pessimistic_expectancy_pips",
                "maximum_daily_sample_share",
            )
        )
        or not isinstance(near, Mapping)
        or near.get("filled_signals") != training["near_side_filled_signals"]
        or near.get("wins") != training["near_side_wins"]
        or near.get("losses") != training["near_side_losses"]
        or near.get("net_pips") != training["near_side_net_pips"]
        or near.get("profit_factor") != training["near_side_profit_factor"]
        or near.get("pessimistic_expectancy_pips")
        != training["near_side_pessimistic_expectancy_pips"]
        or near.get("positive_day_rate")
        != training["near_side_positive_day_rate"]
    ):
        raise ValueError("V3 candidate audit artifact binding mismatch")


def _validate_v2_signal_filter(value: Any) -> None:
    if not isinstance(value, Mapping) or (
        value.get("filter_policy") != SIGNAL_FILTER_POLICY_V2
        or value.get("entry_reference") != "PASSIVE_NEAR_SIDE"
        or value.get("m5_atr_pips_operator") != "GREATER_THAN_OR_EQUAL"
        or value.get("m5_atr_pips_minimum") != V2_MINIMUM_M5_ATR_PIPS
        or value.get("units") != "PIPS"
        or value.get("missing_or_invalid_policy") != "REJECT"
        or value.get("threshold_role") != "POST_HOC_HYPOTHESIS_ONLY"
        or value.get("historical_rows_admitted") is not False
    ):
        raise ValueError("V2 profit holdout signal filter mismatch")


def _policy_contracts(policy: Mapping[str, Any]) -> tuple[str, str, int]:
    if policy.get("contract") == POLICY_CONTRACT_V1:
        return DECISION_CONTRACT_V1, SCORECARD_CONTRACT_V1, 1
    if policy.get("contract") == POLICY_CONTRACT_V2:
        return DECISION_CONTRACT_V2, SCORECARD_CONTRACT_V2, 2
    if policy.get("contract") == POLICY_CONTRACT_V3:
        return DECISION_CONTRACT_V3, SCORECARD_CONTRACT_V3, 3
    raise ValueError("unknown profit holdout policy contract")


def _decision_contract_valid(
    decision: Mapping[str, Any],
    *,
    expected_contract: str | None = None,
) -> bool:
    contract = str(decision.get("contract") or "")
    if contract not in {
        DECISION_CONTRACT_V1,
        DECISION_CONTRACT_V2,
        DECISION_CONTRACT_V3,
    }:
        return False
    expected_schema = {
        DECISION_CONTRACT_V1: 1,
        DECISION_CONTRACT_V2: 2,
        DECISION_CONTRACT_V3: 3,
    }[contract]
    if decision.get("schema_version") != expected_schema:
        return False
    return (expected_contract is None or contract == expected_contract) and sealed_valid(
        decision,
        contract,
    )


def _signal_filter_reasons(
    signal: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[str]:
    if policy.get("contract") == POLICY_CONTRACT_V1:
        return []
    if policy.get("contract") == POLICY_CONTRACT_V3:
        return ["NO_ADMISSIBLE_CANDIDATE"]
    signal_filter = policy["selection"]["signal_filter"]
    reasons: list[str] = []
    if signal.get("entry_reference") != signal_filter["entry_reference"]:
        reasons.append("ENTRY_REFERENCE_NOT_PRECOMMITTED")
    raw_atr = signal.get("m5_atr_pips")
    if isinstance(raw_atr, bool):
        reasons.append("M5_ATR_INVALID_FOR_PRECOMMITTED_FILTER")
        return reasons
    try:
        atr = float(raw_atr)
    except (TypeError, ValueError, OverflowError):
        reasons.append("M5_ATR_INVALID_FOR_PRECOMMITTED_FILTER")
        return reasons
    if not math.isfinite(atr) or atr <= 0.0:
        reasons.append("M5_ATR_INVALID_FOR_PRECOMMITTED_FILTER")
    elif atr < float(signal_filter["m5_atr_pips_minimum"]):
        reasons.append("M5_ATR_BELOW_PRECOMMITTED_MINIMUM")
    return reasons


def build_holdout_decision(
    raw_shadow: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    policy_sha256: str,
    selected_history: Sequence[Mapping[str, Any]] = (),
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Select at most one strictly prospective signal without mutating input."""

    validate_policy(policy)
    if policy_sha256 != canonical_sha(policy):
        raise ValueError("profit holdout policy SHA-256 mismatch")
    decision_contract, _, schema_version = _policy_contracts(policy)
    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    cutoff = _parse_utc(policy["holdout"]["eligible_after_utc"])
    source_errors = _source_shadow_errors(raw_shadow)
    history_errors = _selected_history_errors(selected_history)
    signals = raw_shadow.get("signals") if isinstance(raw_shadow, Mapping) else None
    source_signals = list(signals) if isinstance(signals, list) else []
    if len(source_signals) > MAX_SOURCE_SIGNALS:
        source_errors.append("SOURCE_SIGNAL_LIMIT_EXCEEDED")

    allowed = {
        _lane(row): int(row["priority"])
        for row in policy["selection"]["allowed_lanes"]
    }
    opposite_conflicts = {
        lane
        for lane in allowed
        if any(
            isinstance(signal, Mapping)
            and _fast_bot_signal_valid(signal)
            and signal.get("pair") == lane[0]
            and signal.get("horizon_lane") == lane[3]
            and signal.get("side") in {"LONG", "SHORT"}
            and signal.get("side") != lane[1]
            for signal in source_signals
        )
    }
    rows: list[dict[str, Any]] = []
    eligible: list[tuple[int, int, Mapping[str, Any]]] = []
    source_seals: set[str] = set()
    for index, signal in enumerate(source_signals):
        reasons: list[str] = []
        if not isinstance(signal, Mapping) or not _fast_bot_signal_valid(signal):
            reasons.append("SOURCE_SIGNAL_INVALID")
            signal = signal if isinstance(signal, Mapping) else {}
        seal_value = str(signal.get("signal_sha256") or "")
        if seal_value in source_seals:
            reasons.append("DUPLICATE_SOURCE_SIGNAL")
        elif seal_value:
            source_seals.add(seal_value)
        lane = _lane(signal)
        priority = allowed.get(lane)
        generated: datetime | None = None
        try:
            generated = _parse_utc(signal.get("generated_at_utc"))
        except (TypeError, ValueError):
            reasons.append("SIGNAL_TIME_INVALID")
        if generated is not None:
            if generated <= cutoff:
                reasons.append("IN_SAMPLE_OR_PRE_POLICY_SIGNAL")
            if generated > now:
                reasons.append("SIGNAL_GENERATED_IN_FUTURE")
            elif (now - generated).total_seconds() > int(
                policy["selection"]["maximum_selection_delay_seconds"]
            ):
                reasons.append("SIGNAL_SELECTION_WINDOW_EXPIRED")
        if priority is None:
            reasons.append("LANE_NOT_PRECOMMITTED")
        reasons.extend(_signal_filter_reasons(signal, policy))
        if (
            lane in opposite_conflicts
            or any(
                lane[0] == conflict[0]
                and lane[3] == conflict[3]
                and lane[1] != conflict[1]
                for conflict in opposite_conflicts
            )
        ):
            reasons.append("OPPOSITE_SIDE_GO_AMBIGUITY")
        overlap = _overlapping_history(signal, selected_history)
        if overlap is not None:
            reasons.append(
                "ALREADY_SELECTED"
                if overlap.get("signal_sha256") == signal.get("signal_sha256")
                else "PAIR_HORIZON_RESERVED_BY_PRIOR_SELECTION"
            )
        status = "ELIGIBLE_PRECOMMITTED_CANDIDATE" if not reasons else "REJECTED"
        rows.append(
            {
                "source_index": index,
                "signal_id": signal.get("signal_id"),
                "signal_sha256": signal.get("signal_sha256"),
                "lane": dict(zip(LANE_FIELDS, lane)),
                "precommitted_priority": priority,
                "status": status,
                "reasons": sorted(set(reasons)),
            }
        )
        if not reasons and priority is not None:
            eligible.append((priority, index, signal))

    selected: list[Mapping[str, Any]] = []
    if not source_errors and not history_errors and eligible:
        highest_priority = max(item[0] for item in eligible)
        top = [item for item in eligible if item[0] == highest_priority]
        if len(top) == 1:
            _, selected_index, selected_signal = top[0]
            selected.append(selected_signal)
            for row in rows:
                if row["source_index"] == selected_index:
                    row["status"] = "SELECTED_PROSPECTIVE_HOLDOUT"
                elif row["status"] == "ELIGIBLE_PRECOMMITTED_CANDIDATE":
                    row["status"] = "REJECTED"
                    row["reasons"] = ["LOWER_PRECOMMITTED_PRIORITY"]
        else:
            top_indexes = {item[1] for item in top}
            for row in rows:
                if row["source_index"] in top_indexes:
                    row["status"] = "REJECTED"
                    row["reasons"] = ["AMBIGUOUS_TOP_PRIORITY"]

    status = _decision_status(
        source_errors=source_errors,
        history_errors=history_errors,
        rows=rows,
        selected=selected,
    )
    source_sha = raw_shadow.get("contract_sha256") if isinstance(raw_shadow, Mapping) else None
    source_cycle_sha = _source_cycle_sha256(
        source_signals,
        fallback_generated_at=raw_shadow.get("generated_at_utc"),
    )
    decision_identity = canonical_sha([policy_sha256, source_cycle_sha])
    body = {
        "contract": decision_contract,
        "schema_version": schema_version,
        "generated_at_utc": now.isoformat(),
        "status": status,
        "selection_policy": policy["selection"]["selection_policy"],
        "policy_id": policy.get("policy_id"),
        "policy_sha256": policy_sha256,
        "decision_identity_sha256": decision_identity,
        "eligible_after_utc": cutoff.isoformat(),
        "source_shadow_contract_sha256": source_sha,
        "source_cycle_sha256": source_cycle_sha,
        "source_signal_count": len(source_signals),
        "selected_signal_count": len(selected),
        "selected_signal_sha256s": [str(item["signal_sha256"]) for item in selected],
        "selected_signals": [dict(item) for item in selected],
        "selection_rows": rows,
        "source_integrity_errors": sorted(set(source_errors)),
        "history_integrity_errors": sorted(set(history_errors)),
        "candidate_status": _candidate_status(policy),
        "profitability_claim": _profitability_claim(policy),
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "shadow_only": True,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "gateway_invocations": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    if policy.get("contract") == POLICY_CONTRACT_V2:
        body["signal_filter"] = dict(policy["selection"]["signal_filter"])
    if policy.get("contract") == POLICY_CONTRACT_V3:
        body["candidate_admission"] = dict(policy["candidate_admission"])
    return seal(body)


def build_holdout_scorecard(
    *,
    policy: Mapping[str, Any],
    policy_sha256: str,
    decisions: Sequence[Mapping[str, Any]],
    raw_signals: Sequence[Mapping[str, Any]],
    selected_signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    truth_scorecard: Mapping[str, Any],
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Validate the untouched cohort and assess exact bid/ask profitability."""

    validate_policy(policy)
    if policy_sha256 != canonical_sha(policy):
        raise ValueError("profit holdout policy SHA-256 mismatch")
    decision_contract, scorecard_contract, schema_version = _policy_contracts(policy)
    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    cutoff = _parse_utc(policy["holdout"]["eligible_after_utc"])
    allowed = {_lane(row) for row in policy["selection"]["allowed_lanes"]}
    invalid: list[str] = []

    raw_cycle_by_sha: dict[str, list[Mapping[str, Any]]] = {}
    try:
        raw_cycles = _group_raw_signal_cycles(raw_signals)
    except (TypeError, ValueError):
        raw_cycles = []
        invalid.append("RAW_SIGNAL_LEDGER_INTEGRITY_FAILURE")
    for group in raw_cycles:
        generated = _parse_utc(group[0]["generated_at_utc"])
        if generated < cutoff:
            continue
        if generated > now:
            invalid.append("FUTURE_RAW_SIGNAL_CYCLE")
        cycle_sha = _source_cycle_sha256(group)
        if cycle_sha in raw_cycle_by_sha:
            invalid.append("DUPLICATE_RAW_SIGNAL_CYCLE")
        raw_cycle_by_sha[cycle_sha] = group

    signal_by_sha: dict[str, Mapping[str, Any]] = {}
    for signal in selected_signals:
        if not isinstance(signal, Mapping) or not _fast_bot_signal_valid(signal):
            invalid.append("SELECTED_SIGNAL_INVALID")
            continue
        signal_sha = str(signal.get("signal_sha256") or "")
        if signal_sha in signal_by_sha:
            invalid.append("DUPLICATE_SELECTED_SIGNAL")
            continue
        try:
            generated = _parse_utc(signal.get("generated_at_utc"))
        except (TypeError, ValueError):
            invalid.append("SELECTED_SIGNAL_TIME_INVALID")
            continue
        if generated <= cutoff:
            invalid.append("IN_SAMPLE_SIGNAL_IN_HOLDOUT_LEDGER")
        if generated > now:
            invalid.append("FUTURE_SIGNAL_IN_HOLDOUT_LEDGER")
        if _lane(signal) not in allowed:
            invalid.append("UNPRECOMMITTED_LANE_IN_HOLDOUT_LEDGER")
        invalid.extend(
            f"SELECTED_SIGNAL_{reason}"
            for reason in _signal_filter_reasons(signal, policy)
        )
        signal_by_sha[signal_sha] = signal

    selected_by_decision: defaultdict[str, int] = defaultdict(int)
    decisions_by_raw_cycle: defaultdict[str, int] = defaultdict(int)
    decision_rows_by_raw_cycle: defaultdict[
        str, list[Mapping[str, Any]]
    ] = defaultdict(list)
    for decision in decisions:
        if (
            not isinstance(decision, Mapping)
            or not _decision_contract_valid(
                decision,
                expected_contract=decision_contract,
            )
            or decision.get("schema_version") != schema_version
            or decision.get("policy_sha256") != policy_sha256
            or not _valid_sha(str(decision.get("source_cycle_sha256") or ""))
            or decision.get("decision_identity_sha256")
            != canonical_sha([policy_sha256, decision.get("source_cycle_sha256")])
            or decision.get("policy_id") != policy.get("policy_id")
            or decision.get("selection_policy")
            != policy["selection"]["selection_policy"]
            or decision.get("eligible_after_utc") != cutoff.isoformat()
            or (
                decision.get("source_shadow_contract_sha256") is not None
                and not _valid_sha(
                    str(decision.get("source_shadow_contract_sha256") or "")
                )
            )
            or decision.get("execution_authority") != "NONE"
            or decision.get("broker_http_methods_allowed") != ["GET"]
            or decision.get("broker_mutation_allowed") is not False
            or decision.get("automatic_adoption_allowed") is not False
            or decision.get("promotion_allowed") is not False
            or decision.get("live_permission") is not False
            or decision.get("external_order_attempts") != 0
            or decision.get("external_orders") != 0
            or decision.get("gateway_invocations") != 0
            or decision.get("manual_tagless_policy") != "NO_TOUCH"
            or decision.get("candidate_status") != _candidate_status(policy)
            or decision.get("profitability_claim") != _profitability_claim(policy)
            or (
                policy.get("contract") == POLICY_CONTRACT_V2
                and decision.get("signal_filter")
                != policy["selection"]["signal_filter"]
            )
            or (
                policy.get("contract") == POLICY_CONTRACT_V3
                and decision.get("candidate_admission")
                != policy["candidate_admission"]
            )
        ):
            invalid.append("DECISION_LEDGER_INTEGRITY_FAILURE")
            continue
        source_cycle_sha = str(decision["source_cycle_sha256"])
        if int(decision.get("source_signal_count") or 0) > 0:
            raw_group = raw_cycle_by_sha.get(source_cycle_sha)
            if raw_group is None:
                invalid.append("DECISION_SOURCE_CYCLE_NOT_IN_RAW_LEDGER")
            else:
                decisions_by_raw_cycle[source_cycle_sha] += 1
                decision_rows_by_raw_cycle[source_cycle_sha].append(decision)
                audit_rows = decision.get("selection_rows")
                audit_shas = {
                    str(row.get("signal_sha256") or "")
                    for row in audit_rows
                    if isinstance(row, Mapping)
                } if isinstance(audit_rows, list) else set()
                raw_shas = {str(row["signal_sha256"]) for row in raw_group}
                if (
                    decision.get("source_signal_count") != len(raw_group)
                    or audit_shas != raw_shas
                    or not isinstance(audit_rows, list)
                    or len(audit_rows) != len(raw_group)
                ):
                    invalid.append("DECISION_RAW_CYCLE_AUDIT_INCOMPLETE")
        selected_shas = decision.get("selected_signal_sha256s")
        selected_rows = decision.get("selected_signals")
        if (
            not isinstance(selected_shas, list)
            or not isinstance(selected_rows, list)
            or len(selected_shas) != len(selected_rows)
            or len(selected_shas) > 1
            or decision.get("selected_signal_count") != len(selected_shas)
        ):
            invalid.append("DECISION_SELECTION_CARDINALITY_INVALID")
            continue
        for signal_sha, signal in zip(selected_shas, selected_rows):
            if (
                not isinstance(signal_sha, str)
                or not isinstance(signal, Mapping)
                or signal.get("signal_sha256") != signal_sha
                or not _fast_bot_signal_valid(signal)
                or _signal_filter_reasons(signal, policy)
            ):
                invalid.append("DECISION_SELECTED_SIGNAL_INVALID")
                continue
            try:
                decision_at = _parse_utc(decision.get("generated_at_utc"))
                signal_at = _parse_utc(signal.get("generated_at_utc"))
                selection_delay = (decision_at - signal_at).total_seconds()
                if selection_delay < 0.0:
                    invalid.append("DECISION_PRECEDES_SELECTED_SIGNAL")
                    continue
                if selection_delay > int(
                    policy["selection"]["maximum_selection_delay_seconds"]
                ):
                    invalid.append("DECISION_SELECTION_WINDOW_EXPIRED")
                    continue
            except (TypeError, ValueError):
                invalid.append("DECISION_TIME_INVALID")
                continue
            selection_rows = decision.get("selection_rows")
            if not isinstance(selection_rows, list) or sum(
                isinstance(row, Mapping)
                and row.get("signal_sha256") == signal_sha
                and row.get("status") == "SELECTED_PROSPECTIVE_HOLDOUT"
                for row in selection_rows
            ) != 1:
                invalid.append("DECISION_SELECTION_AUDIT_BINDING_INVALID")
                continue
            selected_by_decision[signal_sha] += 1

    for signal_sha in signal_by_sha:
        if selected_by_decision[signal_sha] != 1:
            invalid.append("SELECTED_SIGNAL_DECISION_BINDING_INVALID")
    for signal_sha in selected_by_decision:
        if signal_sha not in signal_by_sha:
            invalid.append("SELECTED_DECISION_MISSING_LEDGER_SIGNAL")
    for cycle_sha in raw_cycle_by_sha:
        if decisions_by_raw_cycle[cycle_sha] != 1:
            invalid.append("RAW_SIGNAL_CYCLE_DECISION_COVERAGE_INVALID")

    replay_history: list[Mapping[str, Any]] = []
    for raw_group in raw_cycles:
        generated = _parse_utc(raw_group[0]["generated_at_utc"])
        if generated < cutoff:
            continue
        cycle_sha = _source_cycle_sha256(raw_group)
        cycle_decisions = decision_rows_by_raw_cycle.get(cycle_sha, [])
        if len(cycle_decisions) != 1:
            continue
        actual_decision = cycle_decisions[0]
        try:
            decision_at = _parse_utc(actual_decision.get("generated_at_utc"))
            expected_decision = build_holdout_decision(
                _synthetic_shadow(raw_group),
                policy=policy,
                policy_sha256=policy_sha256,
                selected_history=replay_history,
                now_utc=decision_at,
            )
        except (TypeError, ValueError):
            invalid.append("DECISION_SEMANTIC_REPLAY_FAILED")
            continue
        if _decision_selection_semantics(actual_decision) != (
            _decision_selection_semantics(expected_decision)
        ):
            invalid.append("DECISION_SELECTION_SEMANTICS_MISMATCH")
        replay_history.extend(expected_decision.get("selected_signals") or [])

    invalid.extend(_cohort_overlap_errors(list(signal_by_sha.values())))

    valid_outcomes: list[Mapping[str, Any]] = []
    seen_outcomes: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            invalid.append("OUTCOME_LEDGER_NON_OBJECT")
            continue
        signal_sha = str(outcome.get("signal_sha256") or "")
        signal = signal_by_sha.get(signal_sha)
        outcome_sha = str(outcome.get("contract_sha256") or "")
        if (
            signal is None
            or not sealed_valid(outcome, OUTCOME_CONTRACT)
            or not _fast_bot_outcome_valid_for_signal(outcome, signal)
        ):
            invalid.append("OUTCOME_LEDGER_INTEGRITY_FAILURE")
            continue
        if outcome_sha in seen_outcomes:
            invalid.append("DUPLICATE_VALID_OUTCOME")
            continue
        seen_outcomes.add(outcome_sha)
        valid_outcomes.append(outcome)

    if not sealed_valid(truth_scorecard, TRUTH_SCORECARD_CONTRACT):
        invalid.append("TRUTH_SCORECARD_INTEGRITY_FAILURE")
    else:
        try:
            recomputed = build_fast_bot_scorecard(
                list(signal_by_sha.values()),
                valid_outcomes,
                as_of_utc=_parse_utc(truth_scorecard.get("generated_at_utc")),
            )
            if recomputed.get("contract_sha256") != truth_scorecard.get("contract_sha256"):
                invalid.append("TRUTH_SCORECARD_RECOMPUTE_MISMATCH")
        except (TypeError, ValueError):
            invalid.append("TRUTH_SCORECARD_RECOMPUTE_FAILED")

    invalid = sorted(set(invalid))
    evidence: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    if not invalid and policy.get("contract") != POLICY_CONTRACT_V3:
        metrics = _profitability_metrics(
            list(signal_by_sha.values()),
            valid_outcomes,
            truth_scorecard,
        )
        evidence_end = max(
            (
                _parse_utc(signal["generated_at_utc"])
                for signal in signal_by_sha.values()
            ),
            default=cutoff,
        )
        lane = policy["selection"]["allowed_lanes"][0]
        evidence = build_profitability_evidence(
            lane_id=_profitability_lane_id(policy),
            pair=str(lane["pair"]),
            side=str(lane["side"]),
            method=str(lane["method"]),
            order_type="LIMIT",
            metrics=metrics,
            source_artifact_sha256=str(truth_scorecard["contract_sha256"]),
            generated_at_utc=now,
            evidence_end_utc=evidence_end,
            rank_only=False,
        )
        gate = assess_profitability_evidence(
            evidence,
            thresholds=policy["acceptance_thresholds"],
        )

    if invalid:
        status = "REJECT_INVALID_HOLDOUT_COHORT"
        blockers = invalid
    elif policy.get("contract") == POLICY_CONTRACT_V3:
        status = "NO_ADMISSIBLE_PROFIT_CANDIDATE"
        blockers = [
            "NO_ADMISSIBLE_PROFIT_CANDIDATE",
            "NEW_CANDIDATE_REQUIRES_SEPARATE_PRECOMMITTED_POLICY",
        ]
    elif gate is None:
        status = "REJECT_INVALID_HOLDOUT_COHORT"
        blockers = ["PROFITABILITY_GATE_NOT_BUILT"]
    elif gate["status"] == "SHADOW_FORWARD_OBSERVATION_READY":
        status = "SHADOW_PROFITABILITY_EVIDENCE_PASSED"
        blockers = ["SEPARATE_LIVE_PROMOTION_CONTRACT_REQUIRED"]
    else:
        status = str(gate["status"])
        blockers = list(gate.get("blockers") or [])

    body = {
        "contract": scorecard_contract,
        "schema_version": schema_version,
        "generated_at_utc": now.isoformat(),
        "status": status,
        "policy_id": policy.get("policy_id"),
        "policy_sha256": policy_sha256,
        "eligible_after_utc": cutoff.isoformat(),
        "selected_signal_count": len(signal_by_sha),
        "resolved_signal_count": len(valid_outcomes),
        "cohort_integrity_passed": not invalid,
        "cohort_integrity_errors": invalid,
        "truth_scorecard_sha256": truth_scorecard.get("contract_sha256"),
        "truth_metrics": {
            key: truth_scorecard.get(key)
            for key in (
                "filled_signals",
                "active_days",
                "wins",
                "losses",
                "net_pips",
                "mean_pips_per_fill",
                "profit_factor",
                "one_sided_95_daily_mean_lower_pips",
                "forward_evidence_passed",
            )
        },
        "profitability_evidence": evidence,
        "profitability_gate": gate,
        "blockers": sorted(set(blockers)),
        "candidate_status": _candidate_status(policy),
        "profitability_claim": (
            "PROSPECTIVE_SHADOW_EVIDENCE_PASSED"
            if status == "SHADOW_PROFITABILITY_EVIDENCE_PASSED"
            else _profitability_claim(policy)
        ),
        "execution_authority": "NONE",
        "broker_http_methods_allowed": ["GET"],
        "broker_mutation_allowed": False,
        "shadow_only": True,
        "automatic_adoption_allowed": False,
        "promotion_allowed": False,
        "live_permission": False,
        "external_order_attempts": 0,
        "external_orders": 0,
        "gateway_invocations": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    if policy.get("contract") == POLICY_CONTRACT_V2:
        body["signal_filter"] = dict(policy["selection"]["signal_filter"])
    if policy.get("contract") == POLICY_CONTRACT_V3:
        body["candidate_admission"] = dict(policy["candidate_admission"])
    return seal(body)


def run_selection(
    *,
    raw_shadow_path: Path,
    raw_signal_ledger_path: Path,
    policy_path: Path,
    selected_ledger_path: Path,
    decision_ledger_path: Path,
    output_path: Path,
    report_path: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    policy, policy_sha = load_policy(policy_path)
    decision_contract, _, _ = _policy_contracts(policy)
    shadow = _read_object(raw_shadow_path)
    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    raw_signals = load_jsonl(raw_signal_ledger_path)
    raw_errors = _selected_history_errors(raw_signals)
    if raw_errors:
        raise ValueError("raw signal ledger integrity failure: " + ",".join(raw_errors))
    current_signals = [
        row
        for row in shadow.get("signals", []) or []
        if isinstance(row, Mapping)
    ]
    durable_shas = {str(row.get("signal_sha256") or "") for row in raw_signals}
    if any(str(row.get("signal_sha256") or "") not in durable_shas for row in current_signals):
        raise ValueError("current shadow signal is not durable in the raw ledger")

    existing_decisions = load_jsonl(decision_ledger_path)
    signals_appended = 0
    for existing in existing_decisions:
        if (
            not _decision_contract_valid(
                existing,
                expected_contract=decision_contract,
            )
            or existing.get("policy_sha256") != policy_sha
            or existing.get("decision_identity_sha256")
            != canonical_sha([policy_sha, existing.get("source_cycle_sha256")])
        ):
            raise ValueError("decision ledger integrity failure")
        # Decision-first persistence is intentional.  Replaying its immutable
        # selected payload repairs an interruption before the signal append.
        signals_appended += append_selected_signals_once(
            selected_ledger_path,
            existing,
        )

    history = load_jsonl(selected_ledger_path)
    cutoff = _parse_utc(policy["holdout"]["eligible_after_utc"])
    cycle_groups = _group_raw_signal_cycles(raw_signals)
    processed_cycles = {
        str(row.get("source_cycle_sha256") or "") for row in existing_decisions
    }
    pending = [
        group
        for group in cycle_groups
        if _parse_utc(group[0]["generated_at_utc"]) >= cutoff
        and _source_cycle_sha256(group) not in processed_cycles
    ]
    pending.sort(
        key=lambda group: (
            _parse_utc(group[0]["generated_at_utc"]),
            _source_cycle_sha256(group),
        )
    )
    decision_appended = 0
    decisions_screened = 0
    latest_decision: Mapping[str, Any] | None = (
        existing_decisions[-1] if existing_decisions else None
    )
    current_shas = {
        str(row.get("signal_sha256") or "") for row in current_signals
    }
    for group in pending[:MAX_PENDING_CYCLES_PER_RUN]:
        group_shas = {str(row.get("signal_sha256") or "") for row in group}
        packet = shadow if group_shas == current_shas else _synthetic_shadow(group)
        decision = build_holdout_decision(
            packet,
            policy=policy,
            policy_sha256=policy_sha,
            selected_history=history,
            now_utc=now,
        )
        expected_cycle_sha = _source_cycle_sha256(group)
        if decision.get("source_cycle_sha256") != expected_cycle_sha:
            raise ValueError("selection decision source-cycle binding mismatch")
        decision_appended += append_decision_once(decision_ledger_path, decision)
        signals_appended += append_selected_signals_once(
            selected_ledger_path,
            decision,
        )
        history = load_jsonl(selected_ledger_path)
        latest_decision = decision
        decisions_screened += 1

    if latest_decision is None:
        latest_decision = build_holdout_decision(
            shadow,
            policy=policy,
            policy_sha256=policy_sha,
            selected_history=history,
            now_utc=now,
        )
        decision_appended += append_decision_once(
            decision_ledger_path,
            latest_decision,
        )
        signals_appended += append_selected_signals_once(
            selected_ledger_path,
            latest_decision,
        )
        decisions_screened = 1

    _write_json_atomic(output_path, latest_decision)
    _write_text_atomic(report_path, render_selection_report(latest_decision))
    return {
        "status": latest_decision["status"],
        "policy_contract": policy["contract"],
        "policy_id": policy["policy_id"],
        "decision_sha256": latest_decision["contract_sha256"],
        "policy_sha256": policy_sha,
        "signal_filter": policy["selection"].get("signal_filter"),
        "selected_signal_count": latest_decision["selected_signal_count"],
        "selected_signals_appended": signals_appended,
        "decision_appended": decision_appended,
        "raw_cycles_screened": decisions_screened,
        "raw_cycle_backlog_remaining": max(
            0,
            len(pending) - MAX_PENDING_CYCLES_PER_RUN,
        ),
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def run_evaluation(
    *,
    policy_path: Path,
    raw_signal_ledger_path: Path,
    selected_ledger_path: Path,
    decision_ledger_path: Path,
    outcome_ledger_path: Path,
    truth_scorecard_path: Path,
    output_path: Path,
    report_path: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    policy, policy_sha = load_policy(policy_path)
    scorecard = build_holdout_scorecard(
        policy=policy,
        policy_sha256=policy_sha,
        decisions=load_jsonl(decision_ledger_path),
        raw_signals=load_jsonl(raw_signal_ledger_path),
        selected_signals=load_jsonl(selected_ledger_path),
        outcomes=load_jsonl(outcome_ledger_path),
        truth_scorecard=_read_object(truth_scorecard_path),
        now_utc=now_utc,
    )
    _write_json_atomic(output_path, scorecard)
    _write_text_atomic(report_path, render_scorecard_report(scorecard))
    return {
        "status": scorecard["status"],
        "policy_contract": policy["contract"],
        "policy_id": policy["policy_id"],
        "scorecard_sha256": scorecard["contract_sha256"],
        "policy_sha256": policy_sha,
        "signal_filter": policy["selection"].get("signal_filter"),
        "cohort_integrity_passed": scorecard["cohort_integrity_passed"],
        "selected_signal_count": scorecard["selected_signal_count"],
        "filled_signals": scorecard["truth_metrics"].get("filled_signals"),
        "active_days": scorecard["truth_metrics"].get("active_days"),
        "profit_factor": scorecard["truth_metrics"].get("profit_factor"),
        "net_pips": scorecard["truth_metrics"].get("net_pips"),
        "execution_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(value)
    return rows


def append_decision_once(path: Path, decision: Mapping[str, Any]) -> int:
    if not _decision_contract_valid(decision):
        raise ValueError("invalid profit holdout decision")
    decision_contract = str(decision["contract"])
    return _append_jsonl_once(
        path,
        decision,
        identity_field="decision_identity_sha256",
        validator=lambda row: _decision_contract_valid(
            row,
            expected_contract=decision_contract,
        ),
    )


def append_selected_signals_once(path: Path, decision: Mapping[str, Any]) -> int:
    if not _decision_contract_valid(decision):
        raise ValueError("invalid profit holdout decision")
    selected = decision.get("selected_signals")
    if not isinstance(selected, list) or len(selected) > 1:
        raise ValueError("invalid selected signal cardinality")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen: set[str] = set()
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed selected ledger row {number}") from exc
            if not isinstance(row, Mapping) or not _fast_bot_signal_valid(row):
                raise ValueError(f"invalid selected ledger row {number}")
            signal_sha = str(row["signal_sha256"])
            if signal_sha in seen:
                raise ValueError("duplicate selected signal in holdout ledger")
            seen.add(signal_sha)
        appended = 0
        handle.seek(0, os.SEEK_END)
        for signal in selected:
            if not isinstance(signal, Mapping) or not _fast_bot_signal_valid(signal):
                raise ValueError("invalid selected signal")
            signal_sha = str(signal["signal_sha256"])
            if signal_sha in seen:
                continue
            handle.write(json.dumps(dict(signal), ensure_ascii=False, sort_keys=True) + "\n")
            seen.add(signal_sha)
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return appended


def render_selection_report(decision: Mapping[str, Any]) -> str:
    selected = decision.get("selected_signals") or []
    selected_label = (
        ", ".join(
            f"{row.get('pair')} {row.get('side')} {row.get('method')}"
            for row in selected
            if isinstance(row, Mapping)
        )
        or "none"
    )
    signal_filter = decision.get("signal_filter")
    filter_label = (
        f"PASSIVE_NEAR_SIDE and M5 ATR >= {signal_filter.get('m5_atr_pips_minimum')} pips"
        if isinstance(signal_filter, Mapping)
        else (
            "none; no candidate admitted"
            if decision.get("candidate_status") == "NO_ADMISSIBLE_CANDIDATE"
            else "lane only (V1 historical contract)"
        )
    )
    return "\n".join(
        [
            "# Fast Bot Profit Holdout Selection",
            "",
            f"- Generated: `{decision.get('generated_at_utc')}`",
            f"- Status: `{decision.get('status')}`",
            f"- Policy: `{decision.get('policy_id')}` / `{decision.get('policy_sha256')}`",
            f"- Strictly prospective after: `{decision.get('eligible_after_utc')}`",
            f"- Precommitted filter: `{filter_label}`",
            f"- Selected: {selected_label}",
            f"- Profitability claim: `{decision.get('profitability_claim')}`",
            "- Execution authority: `NONE`",
            "- Broker mutation: `false`",
            "- Live permission: `false`",
            "",
        ]
    )


def render_scorecard_report(scorecard: Mapping[str, Any]) -> str:
    metrics = scorecard.get("truth_metrics") or {}
    blockers = scorecard.get("blockers") or []
    signal_filter = scorecard.get("signal_filter")
    filter_label = (
        f"PASSIVE_NEAR_SIDE and M5 ATR >= {signal_filter.get('m5_atr_pips_minimum')} pips"
        if isinstance(signal_filter, Mapping)
        else (
            "none; no candidate admitted"
            if scorecard.get("candidate_status") == "NO_ADMISSIBLE_CANDIDATE"
            else "lane only (V1 historical contract)"
        )
    )
    return "\n".join(
        [
            "# Fast Bot Profit Holdout Scorecard",
            "",
            f"- Generated: `{scorecard.get('generated_at_utc')}`",
            f"- Status: `{scorecard.get('status')}`",
            f"- Cohort integrity: `{scorecard.get('cohort_integrity_passed')}`",
            f"- Precommitted filter: `{filter_label}`",
            f"- Selected / filled / days: {scorecard.get('selected_signal_count')} / {metrics.get('filled_signals')} / {metrics.get('active_days')}",
            f"- Net pips / PF: {metrics.get('net_pips')} / {metrics.get('profit_factor')}",
            f"- Blockers: {', '.join(str(item) for item in blockers) or 'none'}",
            f"- Profitability claim: `{scorecard.get('profitability_claim')}`",
            "- Execution authority: `NONE`",
            "- Broker mutation: `false`",
            "- Live permission: `false`",
            "",
        ]
    )


def _source_shadow_errors(raw_shadow: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(raw_shadow, Mapping) or not sealed_valid(raw_shadow, SHADOW_CONTRACT):
        errors.append("SOURCE_SHADOW_INTEGRITY_FAILURE")
        return errors
    if (
        raw_shadow.get("shadow_only") is not True
        or raw_shadow.get("live_permission") is not False
        or raw_shadow.get("broker_mutation_allowed") is not False
    ):
        errors.append("SOURCE_SHADOW_AUTHORITY_INVALID")
    if not isinstance(raw_shadow.get("signals"), list):
        errors.append("SOURCE_SIGNAL_LIST_INVALID")
    return errors


def _selected_history_errors(history: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in history:
        if not isinstance(row, Mapping) or not _fast_bot_signal_valid(row):
            errors.append("SELECTED_HISTORY_INTEGRITY_FAILURE")
            continue
        signal_sha = str(row.get("signal_sha256") or "")
        if signal_sha in seen:
            errors.append("SELECTED_HISTORY_DUPLICATE")
        seen.add(signal_sha)
    return sorted(set(errors))


def _overlapping_history(
    signal: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    try:
        generated = _parse_utc(signal.get("generated_at_utc"))
    except (TypeError, ValueError):
        return None
    pair_horizon = (signal.get("pair"), signal.get("horizon_lane"))
    for prior in history:
        if not isinstance(prior, Mapping):
            continue
        if (prior.get("pair"), prior.get("horizon_lane")) != pair_horizon:
            continue
        try:
            prior_at = _parse_utc(prior.get("generated_at_utc"))
            reserved_until = prior_at + timedelta(
                seconds=int(prior.get("entry_ttl_seconds") or 0)
                + int(prior.get("max_hold_seconds") or 0)
            )
        except (TypeError, ValueError, OverflowError):
            return prior
        if prior_at <= generated < reserved_until or generated == prior_at:
            return prior
        if generated < prior_at:
            return prior
    return None


def _cohort_overlap_errors(signals: Sequence[Mapping[str, Any]]) -> list[str]:
    prior_until: dict[tuple[str, str], datetime] = {}
    errors: list[str] = []
    for signal in sorted(
        signals,
        key=lambda row: (_parse_utc(row["generated_at_utc"]), str(row["signal_sha256"])),
    ):
        key = (str(signal.get("pair") or ""), str(signal.get("horizon_lane") or ""))
        generated = _parse_utc(signal["generated_at_utc"])
        if key in prior_until and generated < prior_until[key]:
            errors.append("OVERLAPPING_PAIR_HORIZON_SIGNALS")
        prior_until[key] = max(
            prior_until.get(key, generated),
            generated
            + timedelta(
                seconds=int(signal["entry_ttl_seconds"])
                + int(signal["max_hold_seconds"])
            ),
        )
    return errors


def _decision_status(
    *,
    source_errors: Sequence[str],
    history_errors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> str:
    if source_errors:
        return "BLOCKED_SOURCE_INTEGRITY"
    if history_errors:
        return "BLOCKED_HISTORY_INTEGRITY"
    if selected:
        return "SELECTED_PROSPECTIVE_HOLDOUT"
    reasons = {
        str(reason)
        for row in rows
        for reason in row.get("reasons", [])
    }
    if "AMBIGUOUS_TOP_PRIORITY" in reasons:
        return "BLOCKED_AMBIGUOUS_TOP_PRIORITY"
    if "NO_ADMISSIBLE_CANDIDATE" in reasons:
        return "NO_ACTIVE_PROFIT_CANDIDATE"
    if "OPPOSITE_SIDE_GO_AMBIGUITY" in reasons:
        return "BLOCKED_OPPOSITE_SIDE_AMBIGUITY"
    if reasons & {"PAIR_HORIZON_RESERVED_BY_PRIOR_SELECTION", "ALREADY_SELECTED"}:
        return "NO_SELECTION_LANE_RESERVED"
    return "NO_ELIGIBLE_PRECOMMITTED_SIGNAL"


def _decision_selection_semantics(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    rows = decision.get("selection_rows")
    normalized_rows = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                normalized_rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "source_index"
                    }
                )
            else:
                normalized_rows.append(row)
    normalized_rows.sort(
        key=lambda row: (
            str(row.get("signal_sha256") or "")
            if isinstance(row, Mapping)
            else canonical_sha(row)
        )
    )
    return {
        "status": decision.get("status"),
        "selection_policy": decision.get("selection_policy"),
        "policy_id": decision.get("policy_id"),
        "policy_sha256": decision.get("policy_sha256"),
        "eligible_after_utc": decision.get("eligible_after_utc"),
        "source_cycle_sha256": decision.get("source_cycle_sha256"),
        "source_signal_count": decision.get("source_signal_count"),
        "selected_signal_count": decision.get("selected_signal_count"),
        "selected_signal_sha256s": decision.get("selected_signal_sha256s"),
        "selected_signals": decision.get("selected_signals"),
        "selection_rows": normalized_rows,
        "source_integrity_errors": decision.get("source_integrity_errors"),
        "history_integrity_errors": decision.get("history_integrity_errors"),
        "candidate_status": decision.get("candidate_status"),
        "profitability_claim": decision.get("profitability_claim"),
        "signal_filter": decision.get("signal_filter"),
    }


def _profitability_metrics(
    signals: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    truth_scorecard: Mapping[str, Any],
) -> dict[str, Any]:
    signal_by_sha = {str(row["signal_sha256"]): row for row in signals}
    daily_values: defaultdict[str, list[float]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.get("filled") is not True:
            continue
        signal = signal_by_sha[str(outcome["signal_sha256"])]
        day = _parse_utc(signal["generated_at_utc"]).date().isoformat()
        daily_values[day].append(float(outcome["realized_pips"]))
    filled = sum(len(values) for values in daily_values.values())
    day_means = [statistics.fmean(values) for values in daily_values.values()]
    positive_day_rate = (
        sum(value > 0.0 for value in day_means) / len(day_means)
        if day_means
        else 0.0
    )
    maximum_daily_share = (
        max(len(values) for values in daily_values.values()) / filled
        if filled
        else 1.0
    )
    return {
        "sample_count": int(truth_scorecard.get("filled_signals") or 0),
        "active_days": int(truth_scorecard.get("active_days") or 0),
        "profit_factor": truth_scorecard.get("profit_factor") or 0.0,
        "net_pl_pips": float(truth_scorecard.get("net_pips") or 0.0),
        "expectancy_pips": float(truth_scorecard.get("mean_pips_per_fill") or 0.0),
        "pessimistic_expectancy_pips": truth_scorecard.get(
            "one_sided_95_daily_mean_lower_pips"
        ),
        "positive_day_rate": round(positive_day_rate, 12),
        "max_daily_sample_share": round(maximum_daily_share, 12),
        "spread_included": True,
    }


def _profitability_lane_id(policy: Mapping[str, Any]) -> str:
    lane = policy["selection"]["allowed_lanes"][0]
    base = ":".join(str(lane[field]) for field in LANE_FIELDS)
    if policy.get("contract") == POLICY_CONTRACT_V1:
        return base
    minimum = policy["selection"]["signal_filter"]["m5_atr_pips_minimum"]
    return f"{base}:PASSIVE_NEAR_SIDE:M5_ATR_GTE_{minimum:g}"


def _candidate_status(policy: Mapping[str, Any]) -> str:
    return (
        "NO_ADMISSIBLE_CANDIDATE"
        if policy.get("contract") == POLICY_CONTRACT_V3
        else "UNPROVEN_PROSPECTIVE_CANDIDATE"
    )


def _profitability_claim(policy: Mapping[str, Any]) -> str:
    return (
        "NO_ADMISSIBLE_CANDIDATE"
        if policy.get("contract") == POLICY_CONTRACT_V3
        else "UNPROVEN"
    )


def _append_jsonl_once(
    path: Path,
    value: Mapping[str, Any],
    *,
    identity_field: str,
    validator: Any,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen: set[str] = set()
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL row {number}") from exc
            if not isinstance(row, Mapping) or not validator(row):
                raise ValueError(f"invalid JSONL row {number}")
            identity = str(row.get(identity_field) or "")
            if identity in seen:
                raise ValueError("duplicate JSONL identity")
            seen.add(identity)
        identity = str(value.get(identity_field) or "")
        if identity in seen:
            return 0
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return 1


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _lane(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(value.get(field) or "").upper() for field in LANE_FIELDS)  # type: ignore[return-value]


def _source_cycle_sha256(
    signals: Sequence[Mapping[str, Any]],
    *,
    fallback_generated_at: Any = None,
) -> str:
    generated_values = {str(row.get("generated_at_utc") or "") for row in signals}
    regime_values = {str(row.get("regime_contract_sha256") or "") for row in signals}
    if signals and (len(generated_values) != 1 or len(regime_values) != 1):
        raise ValueError("source cycle signals must share generation and regime identity")
    generated_at = (
        next(iter(generated_values))
        if generated_values
        else str(fallback_generated_at or "")
    )
    if not generated_at:
        raise ValueError("source cycle generation time is required")
    _parse_utc(generated_at)
    return canonical_sha(
        {
            "generated_at_utc": generated_at,
            "regime_contract_sha256": (
                next(iter(regime_values)) if regime_values else None
            ),
            "signal_sha256s": sorted(
                str(row.get("signal_sha256") or "") for row in signals
            ),
        }
    )


def _group_raw_signal_cycles(
    signals: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for signal in signals:
        if not isinstance(signal, Mapping) or not _fast_bot_signal_valid(signal):
            raise ValueError("invalid signal in raw signal ledger")
        signal_sha = str(signal["signal_sha256"])
        if signal_sha in seen:
            raise ValueError("duplicate signal in raw signal ledger")
        seen.add(signal_sha)
        groups[
            (
                str(signal["generated_at_utc"]),
                str(signal["regime_contract_sha256"]),
            )
        ].append(signal)
    cycle_groups = [
        sorted(rows, key=lambda row: str(row["signal_sha256"]))
        for rows in groups.values()
    ]
    cycle_groups.sort(
        key=lambda rows: (
            _parse_utc(rows[0]["generated_at_utc"]),
            _source_cycle_sha256(rows),
        )
    )
    return cycle_groups


def _synthetic_shadow(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not signals:
        raise ValueError("cannot reconstruct an empty raw signal cycle")
    generated_at = str(signals[0]["generated_at_utc"])
    body = {
        "contract": SHADOW_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": generated_at,
        "status": "EMITTED",
        "signals": [dict(row) for row in signals],
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "reconstructed_from_append_only_raw_signal_ledger": True,
    }
    return seal(body)


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("aware UTC timestamp is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _aware_utc(parsed)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _valid_sha(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _valid_commit(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)
