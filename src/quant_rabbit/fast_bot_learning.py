"""Diagnostic shadow cohort for all eligible non-primary fast-bot candidates.

This module deliberately does not feed order intents, RiskEngine, the primary
fast-bot scorecard, or any broker mutation path.  It records every eligible
side-by-method cost-blocked, CAUTION, and GO-control hypothesis in a
deterministic order for later exact bid/ask truth scoring.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor


REGIME_CONTRACT = "QR_HIERARCHICAL_BOT_REGIME_V1"
LEARNING_SHADOW_CONTRACT = "QR_FAST_BOT_LEARNING_SHADOW_V1"
LEARNING_SEAT_CONTRACT = "QR_FAST_BOT_LEARNING_SEAT_V1"
LEARNING_CANDIDATE_CONTRACT = "QR_FAST_BOT_LEARNING_CANDIDATE_V1"
LEARNING_SELECTION_POLICY_V2 = "PAIR_UTC_10M_ALL_ELIGIBLE_CELLS_V2"
LEARNING_SELECTION_POLICY = "PAIR_UTC_10M_ALL_VALID_INPUT_CELLS_V3"
SUPPORTED_LEARNING_SELECTION_POLICIES = (
    LEARNING_SELECTION_POLICY_V2,
    LEARNING_SELECTION_POLICY,
)
LEARNING_ARM_POLICY_V1 = "QR_FAST_BOT_LEARNING_OFAT_ARMS_V1"
LEARNING_ARM_POLICY = LEARNING_ARM_POLICY_V1

METHODS = ("BREAKOUT_FAILURE", "RANGE_ROTATION", "TREND_CONTINUATION")
SIDES = ("LONG", "SHORT")
CELL_ORDER = tuple((side, method) for method in METHODS for side in SIDES)
CANDIDATE_CLASSES_V2 = ("COST_BLOCKED", "CAUTION_TECHNICAL", "GO_CONTROL")
CANDIDATE_CLASSES = (
    "COST_BLOCKED",
    "REJECTED_TECHNICAL",
    "SUPERVISOR_BLOCKED",
    "CAUTION_TECHNICAL",
    "GO_CONTROL",
)
INVALID_INPUT_BLOCKER_PREFIXES = (
    "FAST_CHART_PACKET_STALE",
    "BROKER_SNAPSHOT_OR_QUOTES_STALE",
    "PAIR_QUOTE_STALE_OR_FUTURE",
    "TECHNICAL_INPUT_STALE",
    "FAST_TECHNICAL_CANDLE_INTEGRITY_BLOCKED",
    "FAST_TIMEFRAME_EVIDENCE_MISSING",
    "FAST_CLOSED_CANDLE_STALE_OR_FUTURE",
    "SPREAD_OR_M5_ATR_UNAVAILABLE",
)
TECHNICAL_BLOCKERS = frozenset(
    {
        "M5_FAILED_BREAK_DIRECTION_NOT_BOUND_TO_SIDE",
        "OPERATING_RANGE_PHASE_NOT_CONFIRMED",
        "RANGE_EDGE_LOCATION_DOES_NOT_SUPPORT_SIDE",
        "M1_EXECUTION_DIRECTION_NOT_ALIGNED",
        "OPERATING_DIRECTION_NOT_ALIGNED",
        "FAST_TREND_PHASE_NOT_CONFIRMED",
    }
)
COST_BLOCKER = "SPREAD_ANOMALY"
SUPERVISOR_BLOCKER = "AI_REGIME_SUPERVISOR_STOP"
MAX_CANDIDATES_PER_SEAT = len(CELL_ORDER)
SAMPLING_BUCKET_SECONDS = 10 * 60
QUOTE_MAX_AGE_SECONDS = 45
REGIME_MAX_AGE_SECONDS = 180
MAX_SEATS_PER_UTC_DAY = len(DEFAULT_TRADER_PAIRS) * (24 * 60 // 10)
MAX_CANDIDATES_PER_UTC_DAY = MAX_SEATS_PER_UTC_DAY * MAX_CANDIDATES_PER_SEAT
HOT_LEDGER_RETENTION_POLICY = "PENDING_NOT_IMPLEMENTED"
LEARNING_OUTCOME_AGGREGATION_STATUS = "PENDING_NOT_IMPLEMENTED"
STORAGE_GROWTH_DISCLOSURE = (
    "APPEND_ONLY_UNBOUNDED_UNTIL_RETENTION_AND_AGGREGATION_ARE_IMPLEMENTED"
)
PROMOTION_BLOCKERS = (
    "HOT_LEDGER_RETENTION_POLICY_NOT_IMPLEMENTED",
    "LEARNING_OUTCOME_AGGREGATION_NOT_IMPLEMENTED",
    "SEPARATE_FRESH_FORWARD_PROMOTION_CONTRACT_REQUIRED",
)

BASE_ENTRY_TTL_SECONDS = 90
BASE_MAX_HOLD_SECONDS = 15 * 60
LEARNING_ARM_SPECS_V1 = (
    ("BASE", 0.0, BASE_ENTRY_TTL_SECONDS, BASE_MAX_HOLD_SECONDS, 1.0, 1.0, "BASELINE"),
    (
        "ENTRY25",
        0.25,
        BASE_ENTRY_TTL_SECONDS,
        BASE_MAX_HOLD_SECONDS,
        1.0,
        1.0,
        "ENTRY_FRACTION",
    ),
    (
        "ENTRY50",
        0.50,
        BASE_ENTRY_TTL_SECONDS,
        BASE_MAX_HOLD_SECONDS,
        1.0,
        1.0,
        "ENTRY_FRACTION",
    ),
    (
        "ENTRY75",
        0.75,
        BASE_ENTRY_TTL_SECONDS,
        BASE_MAX_HOLD_SECONDS,
        1.0,
        1.0,
        "ENTRY_FRACTION",
    ),
    ("TTL180", 0.0, 180, BASE_MAX_HOLD_SECONDS, 1.0, 1.0, "ENTRY_TTL"),
    ("HOLD1800", 0.0, BASE_ENTRY_TTL_SECONDS, 30 * 60, 1.0, 1.0, "HOLD_HORIZON"),
    (
        "TP125",
        0.0,
        BASE_ENTRY_TTL_SECONDS,
        BASE_MAX_HOLD_SECONDS,
        1.25,
        1.0,
        "TAKE_PROFIT",
    ),
    (
        "SL075",
        0.0,
        BASE_ENTRY_TTL_SECONDS,
        BASE_MAX_HOLD_SECONDS,
        1.0,
        0.75,
        "STOP_LOSS",
    ),
)
LEARNING_ARM_SPECS = LEARNING_ARM_SPECS_V1


def learning_arm_specs_for_policy(
    arm_policy: str,
) -> tuple[tuple[Any, ...], ...]:
    if arm_policy == LEARNING_ARM_POLICY_V1:
        return LEARNING_ARM_SPECS_V1
    raise ValueError("unsupported learning arm policy")


def learning_candidate_classes_for_selection_policy(
    selection_policy: str,
) -> tuple[str, ...]:
    if selection_policy == LEARNING_SELECTION_POLICY_V2:
        return CANDIDATE_CLASSES_V2
    if selection_policy == LEARNING_SELECTION_POLICY:
        return CANDIDATE_CLASSES
    raise ValueError("unsupported learning selection policy")


def build_fast_bot_learning_shadow(
    regime_contract: Mapping[str, Any],
    broker_snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build diagnostic seats without changing the sealed regime decision."""

    now = _aware_utc(now_utc or datetime.now(timezone.utc))
    base = {
        "contract": LEARNING_SHADOW_CONTRACT,
        "schema_version": 1,
        "generated_at_utc": now.isoformat(),
        "selection_policy": LEARNING_SELECTION_POLICY,
        "arm_policy": LEARNING_ARM_POLICY,
        "lifecycle": "PERMANENT_ALWAYS_ON_COUNTERFACTUAL_SHADOW",
        "always_on_counterfactual_shadow": True,
        "order_authority": "NONE",
        "top_one_selection_assumption": False,
        "future_parallel_go_policy": (
            "ALL_ELIGIBLE_GO_CONCURRENT_SUBJECT_TO_PORTFOLIO_GATES"
        ),
        "future_parallel_go_portfolio_gates": [
            "CURRENCY_EXPOSURE",
            "CORRELATION",
            "BROKER_MARGIN",
        ],
        "horizon_lane_policy": "EXACT_PAIR_SIDE_METHOD_HOLD_SECONDS_V1",
        "scorecard_aggregation_policy": (
            "EXACT_PAIR_SIDE_METHOD_HORIZON_LANE_NO_NETTING"
        ),
        "static_pair_correlation_exclusion_allowed": False,
        "pair_direction_netting_allowed": False,
        "portfolio_exposure_aggregation_scope": "SIMULTANEOUS_HOLD_INTERVAL_ONLY",
        "future_parallel_go_policy_is_live_permission": False,
        "primary_artifacts_mutated": False,
        "current_risk_gate_changed": False,
        "pair_universe": "CANONICAL_G8_28",
        "pair_universe_size": len(DEFAULT_TRADER_PAIRS),
        "maximum_seats_per_utc_day": MAX_SEATS_PER_UTC_DAY,
        "maximum_candidates_per_seat": MAX_CANDIDATES_PER_SEAT,
        "maximum_candidates_per_utc_day": MAX_CANDIDATES_PER_UTC_DAY,
        "all_eligible_side_method_cells_emitted": True,
        "valid_input_rejected_cells_retained": True,
        "candidate_classes": list(CANDIDATE_CLASSES),
        "candidate_blocker_facets": [
            "cost_blocked",
            "technical_blocked",
            "supervisor_blocked",
        ],
        "source_timeframe_votes_frozen": True,
        "paired_direction_proof_requires_complete_six_cell_seat": True,
        "future_truth_fetch_unit": "ONE_BID_ASK_PATH_PER_PAIR_10M_SEAT",
        "hot_ledger_retention_policy": HOT_LEDGER_RETENTION_POLICY,
        "learning_outcome_aggregation_status": LEARNING_OUTCOME_AGGREGATION_STATUS,
        "storage_growth_disclosure": STORAGE_GROWTH_DISCLOSURE,
        "promotion_allowed": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "diagnostic_only": True,
        "primary_promotion_eligible": False,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
    }
    if not _sealed_contract_valid(regime_contract, REGIME_CONTRACT):
        return _seal({**base, "status": "INVALID_REGIME_CONTRACT", "seats": []})
    if not _regime_snapshot_binding_valid(regime_contract, broker_snapshot):
        return _seal({**base, "status": "REGIME_SNAPSHOT_BINDING_INVALID", "seats": []})
    regime_generated = _parse_utc(regime_contract.get("generated_at_utc"))
    if not _timestamp_current(now, regime_generated, REGIME_MAX_AGE_SECONDS):
        return _seal({**base, "status": "REGIME_CONTRACT_STALE_OR_FUTURE", "seats": []})

    quotes = (
        broker_snapshot.get("quotes")
        if isinstance(broker_snapshot.get("quotes"), Mapping)
        else {}
    )
    rows_by_pair: dict[str, list[Mapping[str, Any]]] = {}
    outside_universe = 0
    for row in regime_contract.get("rows", []) or []:
        if isinstance(row, Mapping):
            pair = str(row.get("pair") or "").strip().upper()
            if pair in DEFAULT_TRADER_PAIRS:
                rows_by_pair.setdefault(pair, []).append(row)
            elif pair:
                outside_universe += 1

    seats: list[dict[str, Any]] = []
    excluded_pair_counts: dict[str, int] = (
        {"PAIR_OUTSIDE_CANONICAL_G8_28": outside_universe} if outside_universe else {}
    )
    for pair, rows in sorted(rows_by_pair.items()):
        quote = quotes.get(pair) if isinstance(quotes.get(pair), Mapping) else {}
        quote_context = _validated_quote_context(
            pair,
            quote,
            rows=rows,
            now=now,
        )
        if quote_context is None:
            excluded_pair_counts["QUOTE_OR_ATR_INVALID"] = (
                excluded_pair_counts.get("QUOTE_OR_ATR_INVALID", 0) + 1
            )
            continue
        m1_values = {
            str(row.get("m1_closed_candle_utc") or "")
            for row in rows
            if str(row.get("m1_closed_candle_utc") or "")
        }
        if len(m1_values) != 1:
            excluded_pair_counts["M1_IDENTITY_INVALID"] = (
                excluded_pair_counts.get("M1_IDENTITY_INVALID", 0) + 1
            )
            continue
        m1_closed_text = next(iter(m1_values))
        m1_closed = _parse_utc(m1_closed_text)
        if m1_closed is None or m1_closed > now:
            excluded_pair_counts["M1_IDENTITY_INVALID"] = (
                excluded_pair_counts.get("M1_IDENTITY_INVALID", 0) + 1
            )
            continue
        bucket = _sampling_bucket(m1_closed)
        candidates_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
        eligible_counts = {name: 0 for name in CANDIDATE_CLASSES}
        duplicate_cell = False
        for row in rows:
            side = str(row.get("side") or "").upper()
            method = str(row.get("method") or "").upper()
            if (
                side not in SIDES
                or method not in METHODS
                or str(row.get("m1_closed_candle_utc") or "") != m1_closed_text
            ):
                continue
            candidate_class = _candidate_class(row)
            if candidate_class is None or not _row_cost_context_matches(
                row,
                quote_context,
            ):
                continue
            if (side, method) in candidates_by_cell:
                duplicate_cell = True
                break
            eligible_counts[candidate_class] += 1
            candidates_by_cell[(side, method)] = {
                "row": row,
                "candidate_class": candidate_class,
            }
        if duplicate_cell:
            excluded_pair_counts["DUPLICATE_SIDE_METHOD_CELL"] = (
                excluded_pair_counts.get("DUPLICATE_SIDE_METHOD_CELL", 0) + 1
            )
            continue
        if not candidates_by_cell:
            continue

        rotation_offset = _rotation_offset(pair, bucket)
        rotated_cells = [
            CELL_ORDER[(rotation_offset + index) % len(CELL_ORDER)]
            for index in range(len(CELL_ORDER))
        ]
        selected_cells = [cell for cell in rotated_cells if cell in candidates_by_cell]

        seat_identity = {
            "selection_policy": LEARNING_SELECTION_POLICY,
            "pair": pair,
            "sampling_bucket_utc": bucket.isoformat(),
            "m1_closed_candle_utc": m1_closed.isoformat(),
        }
        seat_id = _canonical_sha(seat_identity)[:24]
        cost_context = _cost_context(
            spread_pips=quote_context["regime_spread_pips"],
            spread_to_m5_atr=quote_context["spread_to_m5_atr"],
        )
        candidates: list[dict[str, Any]] = []
        selected_counts = {name: 0 for name in CANDIDATE_CLASSES}
        for cell in selected_cells:
            side, method = cell
            source = candidates_by_cell[cell]
            candidate_class = str(source["candidate_class"])
            row = source["row"]
            selected_counts[candidate_class] += 1
            arms = _learning_arms(
                pair=pair,
                side=side,
                method=method,
                bid=quote_context["bid"],
                ask=quote_context["ask"],
                spread_pips=quote_context["executable_spread_pips"],
                m5_atr_pips=quote_context["m5_atr_pips"],
            )
            candidate_identity = {
                "seat_id": seat_id,
                "side": side,
                "method": method,
                "candidate_class": candidate_class,
            }
            hard_blockers = sorted(
                {str(item).upper() for item in row.get("hard_blockers", []) or []}
            )
            caution_reasons = sorted(
                {str(item) for item in row.get("caution_reasons", []) or []}
            )
            blocker_facets = _blocker_facets(hard_blockers)
            source_regime_evidence = {
                "state": str(row.get("state") or "").upper(),
                "execution_enabled": row.get("execution_enabled") is True,
                "score": _finite_number(row.get("score")),
                "hard_blockers": hard_blockers,
                "caution_reasons": caution_reasons,
                "failed_break_direction": str(row.get("failed_break_direction") or ""),
                "ai_supervision": (
                    dict(row["ai_supervision"])
                    if isinstance(row.get("ai_supervision"), Mapping)
                    else {}
                ),
                "timeframe_votes": _normalized_timeframe_votes(row),
            }
            candidate_body = {
                "contract": LEARNING_CANDIDATE_CONTRACT,
                "schema_version": 1,
                "candidate_id": _canonical_sha(candidate_identity)[:24],
                **candidate_identity,
                "counterfactual_comparison_group_id": seat_id,
                "horizon_lane": "M1_EXECUTION_FACTORIZED_HOLD_LANES",
                "horizon_lane_policy": "EXACT_PAIR_SIDE_METHOD_HOLD_SECONDS_V1",
                "comparison_role": (
                    "ELIGIBLE_GO_CONTROL"
                    if candidate_class == "GO_CONTROL"
                    else "DISCARDED_COUNTERFACTUAL"
                ),
                "top_one_selection_assumption": False,
                "state_at_emission": str(row.get("state") or ""),
                "hard_blockers": hard_blockers,
                "caution_reasons": caution_reasons,
                **blocker_facets,
                "regime_score": _finite_number(row.get("score")),
                "failed_break_direction": str(row.get("failed_break_direction") or ""),
                "source_regime_evidence": source_regime_evidence,
                "source_regime_evidence_sha256": _canonical_sha(source_regime_evidence),
                "arm_policy": LEARNING_ARM_POLICY,
                "arms": arms,
                "frozen_bid_ask_truth_path_required": True,
                "lifecycle": "PERMANENT_ALWAYS_ON_COUNTERFACTUAL_SHADOW",
                "order_authority": "NONE",
                "diagnostic_only": True,
                "primary_promotion_eligible": False,
                "shadow_only": True,
                "live_permission": False,
                "broker_mutation_allowed": False,
            }
            candidates.append(
                {
                    **candidate_body,
                    "candidate_sha256": _canonical_sha(candidate_body),
                }
            )

        unselected_counts = {
            name: eligible_counts[name] - selected_counts[name]
            for name in CANDIDATE_CLASSES
        }
        eligible_cells = [
            {
                "side": side,
                "method": method,
                "candidate_class": str(
                    candidates_by_cell[(side, method)]["candidate_class"]
                ),
            }
            for side, method in rotated_cells
            if (side, method) in candidates_by_cell
        ]
        unselected_cells = [
            item
            for item in eligible_cells
            if (str(item["side"]), str(item["method"])) not in selected_cells
        ]
        complete_six_cell_seat = len(candidates) == MAX_CANDIDATES_PER_SEAT
        seat_body = {
            "contract": LEARNING_SEAT_CONTRACT,
            "schema_version": 1,
            "seat_id": seat_id,
            **seat_identity,
            "arm_policy": LEARNING_ARM_POLICY,
            "generated_at_utc": now.isoformat(),
            "regime_contract_sha256": str(regime_contract.get("contract_sha256") or ""),
            "broker_snapshot_sha256": _canonical_sha(broker_snapshot),
            "quote_timestamp_utc": quote_context["quote_timestamp_utc"],
            "quote_bid": quote_context["bid"],
            "quote_ask": quote_context["ask"],
            "executable_spread_pips": quote_context["executable_spread_pips"],
            "regime_spread_pips": quote_context["regime_spread_pips"],
            "m5_atr_pips": quote_context["m5_atr_pips"],
            "spread_to_m5_atr": quote_context["spread_to_m5_atr"],
            "cost_context": cost_context,
            "rotation_offset": rotation_offset,
            "rotation_order": [f"{side}:{method}" for side, method in rotated_cells],
            "eligible_counts": eligible_counts,
            "selected_counts": selected_counts,
            "eligible_but_unselected_counts": unselected_counts,
            "eligible_cells": eligible_cells,
            "eligible_but_unselected_cells": unselected_cells,
            "selected_candidate_count": len(candidates),
            "candidates": candidates,
            "top_one_selection_assumption": False,
            "future_parallel_go_policy": (
                "ALL_ELIGIBLE_GO_CONCURRENT_SUBJECT_TO_PORTFOLIO_GATES"
            ),
            "future_parallel_go_portfolio_gates": [
                "CURRENCY_EXPOSURE",
                "CORRELATION",
                "BROKER_MARGIN",
            ],
            "counterfactual_comparison_group_id": seat_id,
            "horizon_lane": "M1_EXECUTION_FACTORIZED_HOLD_LANES",
            "horizon_lane_policy": "EXACT_PAIR_SIDE_METHOD_HOLD_SECONDS_V1",
            "scorecard_aggregation_policy": (
                "EXACT_PAIR_SIDE_METHOD_HORIZON_LANE_NO_NETTING"
            ),
            "static_pair_correlation_exclusion_allowed": False,
            "pair_direction_netting_allowed": False,
            "portfolio_exposure_aggregation_scope": ("SIMULTANEOUS_HOLD_INTERVAL_ONLY"),
            "frozen_bid_ask_truth_path_required": True,
            "future_truth_fetch_unit": "ONE_BID_ASK_PATH_PER_PAIR_10M_SEAT",
            "future_metrics_required": [
                "FILL",
                "POST_COST_PNL",
                "MFE",
                "MAE",
            ],
            "lifecycle": "PERMANENT_ALWAYS_ON_COUNTERFACTUAL_SHADOW",
            "always_on_counterfactual_shadow": True,
            "order_authority": "NONE",
            "all_eligible_side_method_cells_emitted": True,
            "valid_input_rejected_cells_retained": True,
            "candidate_classes": list(CANDIDATE_CLASSES),
            "candidate_blocker_facets": [
                "cost_blocked",
                "technical_blocked",
                "supervisor_blocked",
            ],
            "source_timeframe_votes_frozen": True,
            "paired_direction_proof_requires_complete_six_cell_seat": True,
            "complete_six_cell_seat": complete_six_cell_seat,
            "paired_direction_proof_eligible": complete_six_cell_seat,
            "maximum_candidates_per_seat": MAX_CANDIDATES_PER_SEAT,
            "hot_ledger_retention_policy": HOT_LEDGER_RETENTION_POLICY,
            "learning_outcome_aggregation_status": (
                LEARNING_OUTCOME_AGGREGATION_STATUS
            ),
            "storage_growth_disclosure": STORAGE_GROWTH_DISCLOSURE,
            "promotion_allowed": False,
            "promotion_blockers": list(PROMOTION_BLOCKERS),
            "diagnostic_only": True,
            "primary_promotion_eligible": False,
            "shadow_only": True,
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
        seats.append(_seal(seat_body))

    return _seal(
        {
            **base,
            "status": "EMITTED" if seats else "NO_ELIGIBLE_LEARNING_SEAT",
            "seat_count": len(seats),
            "candidate_count": sum(
                int(seat.get("selected_candidate_count") or 0) for seat in seats
            ),
            "complete_six_cell_seat_count": sum(
                seat.get("complete_six_cell_seat") is True for seat in seats
            ),
            "partial_valid_input_seat_count": sum(
                seat.get("complete_six_cell_seat") is False for seat in seats
            ),
            "excluded_pair_counts": excluded_pair_counts,
            "seats": seats,
            "regime_contract_sha256": str(regime_contract.get("contract_sha256") or ""),
            "broker_snapshot_sha256": _canonical_sha(broker_snapshot),
        }
    )


def run_fast_bot_learning_shadow(
    *,
    regime_contract_path: Path,
    broker_snapshot_path: Path,
    output_path: Path,
    ledger_path: Path,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Build, append once per pair/bucket, and atomically publish latest."""

    regime = _read_object(regime_contract_path)
    snapshot = _read_object(broker_snapshot_path)
    shadow = build_fast_bot_learning_shadow(regime, snapshot, now_utc=now_utc)
    try:
        append_result = _append_learning_seats_once(
            ledger_path,
            [
                seat
                for seat in shadow.get("seats", []) or []
                if isinstance(seat, Mapping)
            ],
        )
        latest = _seal(
            {
                **{
                    key: value
                    for key, value in shadow.items()
                    if key != "contract_sha256"
                },
                "ledger_status": "APPENDED",
                "ledger_appended": append_result["appended"],
                "bucket_duplicates_suppressed": append_result[
                    "bucket_duplicates_suppressed"
                ],
                "ledger_path": str(ledger_path),
            }
        )
    except ValueError as exc:
        latest = _seal(
            {
                **{
                    key: value
                    for key, value in shadow.items()
                    if key != "contract_sha256"
                },
                "status": "LEARNING_LEDGER_INVALID",
                "ledger_status": "INVALID_FAIL_CLOSED",
                "ledger_appended": 0,
                "bucket_duplicates_suppressed": 0,
                "ledger_error": str(exc)[:320],
                "ledger_path": str(ledger_path),
            }
        )
    _write_json_atomic(output_path, latest)
    return {
        "status": latest["status"],
        "seat_count": latest.get("seat_count", 0),
        "candidate_count": latest.get("candidate_count", 0),
        "ledger_appended": latest["ledger_appended"],
        "bucket_duplicates_suppressed": latest["bucket_duplicates_suppressed"],
        "diagnostic_only": True,
        "lifecycle": "PERMANENT_ALWAYS_ON_COUNTERFACTUAL_SHADOW",
        "always_on_counterfactual_shadow": True,
        "order_authority": "NONE",
        "primary_promotion_eligible": False,
        "promotion_allowed": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "hot_ledger_retention_policy": HOT_LEDGER_RETENTION_POLICY,
        "learning_outcome_aggregation_status": LEARNING_OUTCOME_AGGREGATION_STATUS,
        "storage_growth_disclosure": STORAGE_GROWTH_DISCLOSURE,
        "shadow_only": True,
        "live_permission": False,
        "broker_mutation_allowed": False,
        "output": str(output_path),
        "ledger": str(ledger_path),
    }


def _candidate_class(row: Mapping[str, Any]) -> str | None:
    state = str(row.get("state") or "").upper()
    hard = {str(item).upper() for item in row.get("hard_blockers", []) or []}
    if not _timeframe_votes_complete(row):
        return None
    if any(
        any(item.startswith(prefix) for prefix in INVALID_INPUT_BLOCKER_PREFIXES)
        for item in hard
    ):
        return None
    known = {*TECHNICAL_BLOCKERS, COST_BLOCKER, SUPERVISOR_BLOCKER}
    if hard - known:
        return None
    if state == "STOP":
        if SUPERVISOR_BLOCKER in hard:
            return "SUPERVISOR_BLOCKED"
        if COST_BLOCKER in hard:
            return "COST_BLOCKED"
        # Input/quote validity is proved separately before a seat can be
        # emitted.  A deterministic technical veto is therefore evidence to
        # retain, not a reason to erase the opposite side or discarded method
        # from the counterfactual cohort.
        return "REJECTED_TECHNICAL"
    if state == "CAUTION" and not hard:
        return "CAUTION_TECHNICAL"
    if state == "GO" and not hard and row.get("execution_enabled") is True:
        return "GO_CONTROL"
    return None


def _normalized_timeframe_votes(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = row.get("timeframe_votes")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(name): dict(vote)
        for name, vote in sorted(raw.items())
        if isinstance(vote, Mapping)
    }


def _timeframe_votes_complete(row: Mapping[str, Any]) -> bool:
    votes = _normalized_timeframe_votes(row)
    required = {"M1", "M5", "M15", "M30", "H1", "H4", "D"}
    return bool(
        set(votes) == required
        and all(vote.get("evidence_complete") is True for vote in votes.values())
    )


def _blocker_facets(hard_blockers: Sequence[str]) -> dict[str, bool]:
    hard = {str(item).upper() for item in hard_blockers}
    return {
        "cost_blocked": COST_BLOCKER in hard,
        "technical_blocked": bool(hard & TECHNICAL_BLOCKERS),
        "supervisor_blocked": SUPERVISOR_BLOCKER in hard,
    }


def _validated_quote_context(
    pair: str,
    quote: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    now: datetime,
) -> dict[str, Any] | None:
    bid_raw = _positive_number(quote.get("bid"))
    ask_raw = _positive_number(quote.get("ask"))
    quote_at = _parse_utc(quote.get("timestamp_utc"))
    if (
        bid_raw is None
        or ask_raw is None
        or ask_raw <= bid_raw
        or not _timestamp_current(now, quote_at, QUOTE_MAX_AGE_SECONDS)
    ):
        return None
    row_atr = {
        float(row["m5_atr_pips"])
        for row in rows
        if _positive_number(row.get("m5_atr_pips")) is not None
    }
    row_spread = {
        float(row["spread_pips"])
        for row in rows
        if _positive_number(row.get("spread_pips")) is not None
    }
    row_ratio = {
        float(row["spread_to_m5_atr"])
        for row in rows
        if _positive_number(row.get("spread_to_m5_atr")) is not None
    }
    if len(row_atr) != 1 or len(row_spread) != 1 or len(row_ratio) != 1:
        return None
    atr = next(iter(row_atr))
    spread = next(iter(row_spread))
    ratio = next(iter(row_ratio))
    try:
        pip_factor = float(instrument_pip_factor(pair))
    except (KeyError, TypeError, ValueError):
        return None
    observed_spread = (ask_raw - bid_raw) * pip_factor
    if not math.isclose(
        spread, observed_spread, rel_tol=0.0, abs_tol=1e-6
    ) or not math.isclose(ratio, spread / atr, rel_tol=0.0, abs_tol=1e-6):
        return None
    bid = _price(pair, bid_raw)
    ask = _price(pair, ask_raw)
    if ask <= bid:
        return None
    executable_spread = round(
        (ask - bid) * pip_factor,
        6,
    )
    return {
        "bid": bid,
        "ask": ask,
        "quote_timestamp_utc": quote_at.isoformat(),
        "executable_spread_pips": executable_spread,
        "regime_spread_pips": round(spread, 6),
        "m5_atr_pips": round(atr, 6),
        "spread_to_m5_atr": round(ratio, 6),
    }


def _row_cost_context_matches(
    row: Mapping[str, Any],
    quote_context: Mapping[str, Any],
) -> bool:
    atr = _positive_number(row.get("m5_atr_pips"))
    spread = _positive_number(row.get("spread_pips"))
    ratio = _positive_number(row.get("spread_to_m5_atr"))
    if atr is None or spread is None or ratio is None:
        return False
    return bool(
        math.isclose(
            atr,
            float(quote_context["m5_atr_pips"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            spread,
            float(quote_context["regime_spread_pips"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        and math.isclose(
            ratio,
            float(quote_context["spread_to_m5_atr"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )


def _learning_arms(
    *,
    pair: str,
    side: str,
    method: str,
    bid: float,
    ask: float,
    spread_pips: float,
    m5_atr_pips: float,
    arm_policy: str = LEARNING_ARM_POLICY,
) -> list[dict[str, Any]]:
    base_tp, base_sl = _shadow_geometry_pips(
        method,
        spread=spread_pips,
        m5_atr=m5_atr_pips,
    )
    arms: list[dict[str, Any]] = []
    for (
        arm_id,
        entry_fraction,
        ttl_seconds,
        hold_seconds,
        tp_multiplier,
        sl_multiplier,
        changed_axis,
    ) in learning_arm_specs_for_policy(arm_policy):
        tp_pips = min(15.0, base_tp * tp_multiplier)
        sl_pips = min(30.0, base_sl * sl_multiplier)
        geometry = _rounded_passive_geometry(
            pair=pair,
            side=side,
            bid=bid,
            ask=ask,
            entry_fraction=entry_fraction,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
        )
        arms.append(
            {
                "arm_id": arm_id,
                "changed_axis": changed_axis,
                "horizon_lane": f"M1_EXECUTION_HOLD_{hold_seconds}S",
                "entry_fraction_toward_market": entry_fraction,
                "entry_ttl_seconds": ttl_seconds,
                "max_hold_seconds": hold_seconds,
                "tp_multiplier": tp_multiplier,
                "sl_multiplier": sl_multiplier,
                **geometry,
            }
        )
    return arms


def _rounded_passive_geometry(
    *,
    pair: str,
    side: str,
    bid: float,
    ask: float,
    entry_fraction: float,
    tp_pips: float,
    sl_pips: float,
) -> dict[str, Any]:
    if side not in SIDES or ask <= bid:
        raise ValueError("invalid passive learning geometry")
    price_tick = 10.0 ** (-(3 if pair.endswith("_JPY") else 5))
    width = ask - bid
    raw_entry = (
        bid + width * entry_fraction if side == "LONG" else ask - width * entry_fraction
    )
    rounded = _price(pair, raw_entry)
    entry = (
        max(bid, min(rounded, ask - price_tick))
        if side == "LONG"
        else min(ask, max(rounded, bid + price_tick))
    )
    entry = _price(pair, entry)
    pip_factor = float(instrument_pip_factor(pair))
    pip_size = 1.0 / pip_factor
    take_profit = _price(
        pair,
        entry + tp_pips * pip_size if side == "LONG" else entry - tp_pips * pip_size,
    )
    stop_loss = _price(
        pair,
        entry - sl_pips * pip_size if side == "LONG" else entry + sl_pips * pip_size,
    )
    return {
        "entry": entry,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "take_profit_pips": round(abs(take_profit - entry) * pip_factor, 6),
        "stop_loss_pips": round(abs(entry - stop_loss) * pip_factor, 6),
        "effective_entry_fraction_toward_market": round(
            (entry - bid) / width if side == "LONG" else (ask - entry) / width,
            6,
        ),
    }


def _shadow_geometry_pips(
    method: str,
    *,
    spread: float,
    m5_atr: float,
) -> tuple[float, float]:
    """Pure contract copy of the primary v1 method geometry."""

    if method == "TREND_CONTINUATION":
        tp = max(3.0, spread * 4.0, m5_atr * 0.8)
        sl = max(2.5, spread * 4.0, m5_atr * 0.6)
    else:
        tp = max(2.0, spread * 3.0, m5_atr * 0.6)
        sl = max(3.0, spread * 4.0, m5_atr * 0.8)
    return min(tp, 15.0), min(sl, 30.0)


def _cost_context(
    *,
    spread_pips: float,
    spread_to_m5_atr: float,
) -> dict[str, Any]:
    pressure = max(spread_pips / 3.0, spread_to_m5_atr / 0.35)
    return {
        "cost_pressure": round(pressure, 6),
        "cost_pressure_bucket": _bucket(
            pressure,
            (
                (0.50, "LE_0_50"),
                (0.75, "0_50_0_75"),
                (1.00, "0_75_1_00"),
                (1.25, "1_00_1_25"),
                (1.50, "1_25_1_50"),
            ),
            "GT_1_50",
        ),
        "spread_pips_bucket": _bucket(
            spread_pips,
            ((1.0, "LE_1"), (2.0, "1_2"), (3.0, "2_3"), (5.0, "3_5")),
            "GT_5",
        ),
        "spread_to_m5_atr_bucket": _bucket(
            spread_to_m5_atr,
            (
                (0.10, "LE_0_10"),
                (0.20, "0_10_0_20"),
                (0.35, "0_20_0_35"),
                (0.50, "0_35_0_50"),
                (0.75, "0_50_0_75"),
            ),
            "GT_0_75",
        ),
        "current_cost_gate_pass": pressure <= 1.0,
        "current_absolute_spread_cap_pips": 3.0,
        "current_spread_to_m5_atr_cap": 0.35,
    }


def _bucket(
    value: float,
    boundaries: Sequence[tuple[float, str]],
    overflow: str,
) -> str:
    for upper, label in boundaries:
        if value <= upper:
            return label
    return overflow


def _rotation_offset(pair: str, bucket: datetime) -> int:
    pair_offset = int(hashlib.sha256(pair.encode("utf-8")).hexdigest()[:8], 16)
    bucket_index = int(bucket.timestamp()) // SAMPLING_BUCKET_SECONDS
    return (pair_offset + bucket_index) % len(CELL_ORDER)


def _sampling_bucket(value: datetime) -> datetime:
    timestamp = int(_aware_utc(value).timestamp())
    start = timestamp - timestamp % SAMPLING_BUCKET_SECONDS
    return datetime.fromtimestamp(start, tz=timezone.utc)


def _append_learning_seats_once(
    path: Path,
    seats: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if not seats:
        return {"appended": 0, "bucket_duplicates_suppressed": 0}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        seen_ids: set[str] = set()
        seen_buckets: dict[tuple[str, str], str] = {}
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"malformed learning ledger row {line_number}"
                ) from exc
            if not isinstance(item, Mapping) or not _learning_seat_valid(item):
                raise ValueError(f"invalid learning ledger row {line_number}")
            seat_id = str(item["seat_id"])
            bucket_identity = _bucket_identity(item)
            prior = seen_buckets.get(bucket_identity)
            if seat_id in seen_ids or (prior is not None and prior != seat_id):
                raise ValueError(
                    f"duplicate learning ledger identity at row {line_number}"
                )
            seen_ids.add(seat_id)
            seen_buckets[bucket_identity] = seat_id

        appended = 0
        suppressed = 0
        handle.seek(0, os.SEEK_END)
        for seat in seats:
            if not _learning_seat_valid(seat):
                raise ValueError("attempted to append invalid learning seat")
            seat_id = str(seat["seat_id"])
            bucket_identity = _bucket_identity(seat)
            if seat_id in seen_ids or bucket_identity in seen_buckets:
                suppressed += 1
                continue
            handle.write(
                json.dumps(dict(seat), ensure_ascii=False, sort_keys=True) + "\n"
            )
            seen_ids.add(seat_id)
            seen_buckets[bucket_identity] = seat_id
            appended += 1
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {"appended": appended, "bucket_duplicates_suppressed": suppressed}


def _learning_seat_valid(value: Mapping[str, Any]) -> bool:
    if not _sealed_contract_valid(value, LEARNING_SEAT_CONTRACT):
        return False
    candidates = value.get("candidates")
    eligible_cells = value.get("eligible_cells")
    unselected_cells = value.get("eligible_but_unselected_cells")
    unselected_counts = value.get("eligible_but_unselected_counts")
    selection_policy = str(value.get("selection_policy") or "")
    arm_policy = _seat_arm_policy(value)
    try:
        candidate_classes = learning_candidate_classes_for_selection_policy(
            selection_policy
        )
        learning_arm_specs_for_policy(arm_policy)
    except ValueError:
        return False
    if (
        selection_policy not in SUPPORTED_LEARNING_SELECTION_POLICIES
        or value.get("diagnostic_only") is not True
        or value.get("lifecycle") != "PERMANENT_ALWAYS_ON_COUNTERFACTUAL_SHADOW"
        or value.get("always_on_counterfactual_shadow") is not True
        or value.get("order_authority") != "NONE"
        or value.get("primary_promotion_eligible") is not False
        or value.get("shadow_only") is not True
        or value.get("live_permission") is not False
        or value.get("broker_mutation_allowed") is not False
        or value.get("all_eligible_side_method_cells_emitted") is not True
        or value.get("maximum_candidates_per_seat") != MAX_CANDIDATES_PER_SEAT
        or value.get("future_truth_fetch_unit") != "ONE_BID_ASK_PATH_PER_PAIR_10M_SEAT"
        or value.get("hot_ledger_retention_policy") != HOT_LEDGER_RETENTION_POLICY
        or value.get("learning_outcome_aggregation_status")
        != LEARNING_OUTCOME_AGGREGATION_STATUS
        or value.get("storage_growth_disclosure") != STORAGE_GROWTH_DISCLOSURE
        or value.get("promotion_allowed") is not False
        or value.get("promotion_blockers") != list(PROMOTION_BLOCKERS)
        or not isinstance(candidates, list)
        or not isinstance(eligible_cells, list)
        or not isinstance(unselected_cells, list)
        or unselected_cells
        or not isinstance(unselected_counts, Mapping)
        or set(unselected_counts) != set(candidate_classes)
        or any(unselected_counts.get(name) != 0 for name in candidate_classes)
        or not 1 <= len(candidates) <= MAX_CANDIDATES_PER_SEAT
        or len(candidates) != len(eligible_cells)
        or value.get("selected_candidate_count") != len(candidates)
        or _parse_utc(value.get("sampling_bucket_utc")) is None
        or _parse_utc(value.get("m1_closed_candle_utc")) is None
    ):
        return False
    if selection_policy == LEARNING_SELECTION_POLICY:
        complete_six = len(candidates) == MAX_CANDIDATES_PER_SEAT
        if (
            value.get("paired_direction_proof_requires_complete_six_cell_seat")
            is not True
            or value.get("complete_six_cell_seat") is not complete_six
            or value.get("paired_direction_proof_eligible") is not complete_six
        ):
            return False
    bucket = _parse_utc(value.get("sampling_bucket_utc"))
    m1_closed = _parse_utc(value.get("m1_closed_candle_utc"))
    seat_identity = {
        "selection_policy": value.get("selection_policy"),
        "pair": value.get("pair"),
        "sampling_bucket_utc": value.get("sampling_bucket_utc"),
        "m1_closed_candle_utc": value.get("m1_closed_candle_utc"),
    }
    if (
        str(value.get("seat_id") or "") != _canonical_sha(seat_identity)[:24]
        or bucket != _sampling_bucket(m1_closed)
        or str(value.get("counterfactual_comparison_group_id") or "")
        != str(value.get("seat_id") or "")
    ):
        return False
    candidate_ids = [
        str(item.get("candidate_id") or "")
        for item in candidates
        if isinstance(item, Mapping)
    ]
    return bool(
        len(candidate_ids) == len(candidates)
        and len(set(candidate_ids)) == len(candidate_ids)
        and all(_learning_candidate_valid(item, seat=value) for item in candidates)
    )


def _learning_candidate_valid(value: Any, *, seat: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    seat_id = str(seat.get("seat_id") or "")
    body = {key: item for key, item in value.items() if key != "candidate_sha256"}
    arms = value.get("arms")
    identity = {
        "seat_id": seat_id,
        "side": value.get("side"),
        "method": value.get("method"),
        "candidate_class": value.get("candidate_class"),
    }
    selection_policy = str(seat.get("selection_policy") or "")
    arm_policy = _seat_arm_policy(seat)
    try:
        candidate_classes = learning_candidate_classes_for_selection_policy(
            selection_policy
        )
        arm_specs = learning_arm_specs_for_policy(arm_policy)
        expected_arms = _learning_arms(
            pair=str(seat["pair"]),
            side=str(value["side"]),
            method=str(value["method"]),
            bid=float(seat["quote_bid"]),
            ask=float(seat["quote_ask"]),
            spread_pips=float(seat["executable_spread_pips"]),
            m5_atr_pips=float(seat["m5_atr_pips"]),
            arm_policy=arm_policy,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    common_valid = bool(
        value.get("contract") == LEARNING_CANDIDATE_CONTRACT
        and value.get("seat_id") == seat_id
        and value.get("candidate_class") in candidate_classes
        and str(value.get("candidate_id") or "") == _canonical_sha(identity)[:24]
        and value.get("counterfactual_comparison_group_id") == seat_id
        and value.get("top_one_selection_assumption") is False
        and value.get("horizon_lane") == "M1_EXECUTION_FACTORIZED_HOLD_LANES"
        and value.get("diagnostic_only") is True
        and value.get("lifecycle") == "PERMANENT_ALWAYS_ON_COUNTERFACTUAL_SHADOW"
        and value.get("order_authority") == "NONE"
        and value.get("primary_promotion_eligible") is False
        and value.get("shadow_only") is True
        and value.get("live_permission") is False
        and value.get("broker_mutation_allowed") is False
        and isinstance(arms, list)
        and value.get("arm_policy") == arm_policy
        and [arm.get("arm_id") for arm in arms if isinstance(arm, Mapping)]
        == [spec[0] for spec in arm_specs]
        and arms == expected_arms
        and str(value.get("candidate_sha256") or "") == _canonical_sha(body)
    )
    if not common_valid:
        return False
    candidate_class = str(value.get("candidate_class") or "")
    state = str(value.get("state_at_emission") or "")
    hard = {str(item) for item in value.get("hard_blockers", []) or []}
    if selection_policy == LEARNING_SELECTION_POLICY_V2:
        return bool(
            (
                candidate_class == "COST_BLOCKED"
                and state == "STOP"
                and hard == {"SPREAD_ANOMALY"}
            )
            or (
                candidate_class == "CAUTION_TECHNICAL"
                and state == "CAUTION"
                and not hard
            )
            or (candidate_class == "GO_CONTROL" and state == "GO" and not hard)
        )
    source_evidence = value.get("source_regime_evidence")
    if not isinstance(source_evidence, Mapping):
        return False
    execution_enabled = source_evidence.get("execution_enabled") is True
    expected_source_evidence = {
        "state": state,
        "execution_enabled": execution_enabled,
        "score": value.get("regime_score"),
        "hard_blockers": sorted(hard),
        "caution_reasons": sorted(
            {str(item) for item in value.get("caution_reasons", []) or []}
        ),
        "failed_break_direction": str(value.get("failed_break_direction") or ""),
        "ai_supervision": (
            dict(source_evidence["ai_supervision"])
            if isinstance(source_evidence.get("ai_supervision"), Mapping)
            else {}
        ),
        "timeframe_votes": (
            {
                str(name): dict(vote)
                for name, vote in sorted(source_evidence["timeframe_votes"].items())
                if isinstance(vote, Mapping)
            }
            if isinstance(source_evidence.get("timeframe_votes"), Mapping)
            else {}
        ),
    }
    expected_facets = _blocker_facets(sorted(hard))
    if not bool(
        dict(source_evidence) == expected_source_evidence
        and value.get("source_regime_evidence_sha256")
        == _canonical_sha(expected_source_evidence)
        and all(
            value.get(name) is expected for name, expected in expected_facets.items()
        )
        and set(expected_source_evidence["timeframe_votes"])
        == {"M1", "M5", "M15", "M30", "H1", "H4", "D"}
        and all(
            vote.get("evidence_complete") is True
            for vote in expected_source_evidence["timeframe_votes"].values()
        )
    ):
        return False
    return bool(
        (
            candidate_class == "SUPERVISOR_BLOCKED"
            and state == "STOP"
            and not execution_enabled
            and SUPERVISOR_BLOCKER in hard
        )
        or (
            candidate_class == "COST_BLOCKED"
            and state == "STOP"
            and not execution_enabled
            and COST_BLOCKER in hard
            and SUPERVISOR_BLOCKER not in hard
        )
        or (
            candidate_class == "REJECTED_TECHNICAL"
            and state == "STOP"
            and not execution_enabled
            and COST_BLOCKER not in hard
            and SUPERVISOR_BLOCKER not in hard
            and bool(hard)
            and hard <= TECHNICAL_BLOCKERS
        )
        or (
            candidate_class == "CAUTION_TECHNICAL"
            and state == "CAUTION"
            and not execution_enabled
            and not hard
        )
        or (
            candidate_class == "GO_CONTROL"
            and state == "GO"
            and execution_enabled
            and not hard
        )
    )


def _bucket_identity(value: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(value.get("pair") or ""),
        str(value.get("sampling_bucket_utc") or ""),
    )


def _seat_arm_policy(value: Mapping[str, Any]) -> str:
    explicit = str(value.get("arm_policy") or "")
    if explicit:
        return explicit
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        return ""
    policies = {
        str(item.get("arm_policy") or "")
        for item in candidates
        if isinstance(item, Mapping) and str(item.get("arm_policy") or "")
    }
    return next(iter(policies)) if len(policies) == 1 else ""


def _regime_snapshot_binding_valid(
    regime_contract: Mapping[str, Any],
    broker_snapshot: Mapping[str, Any],
) -> bool:
    sources = (
        regime_contract.get("sources")
        if isinstance(regime_contract.get("sources"), Mapping)
        else {}
    )
    return str(sources.get("broker_snapshot_sha256") or "") == _canonical_sha(
        broker_snapshot
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return {**body, "contract_sha256": _canonical_sha(body)}


def _sealed_contract_valid(value: Mapping[str, Any], contract: str) -> bool:
    if not isinstance(value, Mapping) or value.get("contract") != contract:
        return False
    stored = str(value.get("contract_sha256") or "")
    body = {key: item for key, item in value.items() if key != "contract_sha256"}
    return bool(stored and stored == _canonical_sha(body))


def _canonical_sha(value: Any) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raw = b"INVALID"
    return hashlib.sha256(raw).hexdigest()


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp_current(
    now: datetime,
    then: datetime | None,
    max_age_seconds: int,
) -> bool:
    if then is None:
        return False
    age = (now - then).total_seconds()
    return 0.0 <= age <= max_age_seconds


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("_JPY") else 5)


__all__ = [
    "LEARNING_ARM_POLICY",
    "LEARNING_ARM_POLICY_V1",
    "LEARNING_ARM_SPECS",
    "LEARNING_ARM_SPECS_V1",
    "LEARNING_SEAT_CONTRACT",
    "LEARNING_SHADOW_CONTRACT",
    "build_fast_bot_learning_shadow",
    "learning_arm_specs_for_policy",
    "learning_candidate_classes_for_selection_policy",
    "run_fast_bot_learning_shadow",
]
