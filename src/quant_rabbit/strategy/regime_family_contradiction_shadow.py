"""Paired, read-only evidence for regime-family forecast contradictions.

This module is deliberately separate from :mod:`projection_ledger`.  The
ordinary projection ledger scores path-dependent ATR/target semantics and
feeds live forecast calibration.  A regime-family contradiction instead needs
one outcome-blind pair of hypotheses, evaluated at a fixed 60-minute terminal
quote.  Mixing those contracts would let a diagnostic arm silently affect a
``directional_forecast*`` consumer.

The public flow is:

* :func:`build_regime_family_contradiction_shadow` freezes the detector and
  family arms from one validated technical-context receipt;
* :func:`bind_regime_family_contradiction_emission` adds the immutable
  emission clock, cycle, non-overlap rule, and optional predeclared holdout;
* :func:`persist_regime_family_contradiction_emission` appends an idempotent,
  hash-chained event to its own JSONL ledger;
* :func:`resolve_due_regime_family_contradiction_trials` chooses the first
  complete OANDA-style M1 bid/ask close at or after the exact 60-minute
  terminal clock (bounded to less than one minute of observation lag);
* :func:`resolve_regime_family_contradiction_trial` computes executable LONG
  and SHORT returns, including both entry and exit spread plus any explicit
  non-spread cost supplied by the evaluator.

Every artifact is shadow-only.  It cannot grant live permission, change
sizing, relax a gate, or replace a forward holdout.  Digest validation proves
internal content integrity only; an adapter evaluating promotion must also
authenticate the referenced technical-context and candle source bytes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS, instrument_pip_factor
from quant_rabbit.strategy.forecast_technical_context import (
    verify_forecast_technical_context,
)
from quant_rabbit.strategy.regime_family_weighting import (
    verify_regime_family_weighting_context_binding,
    verify_regime_family_weighting_receipt,
)


SHADOW_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_SHADOW_V1"
TRIAL_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_TRIAL_V1"
RESULT_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_TERMINAL_RESULT_V1"
HOLDOUT_LOCK_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_HOLDOUT_LOCK_V1"
LEDGER_EVENT_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_LEDGER_EVENT_V1"
SELECTION_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_NON_OVERLAP_V1"
TERMINAL_EVALUATION_CONTRACT = "QR_REGIME_FAMILY_CONTRADICTION_FIXED_60M_BID_ASK_V1"
LEDGER_FILENAME = "regime_family_contradiction_shadow_ledger.jsonl"
LEDGER_LOCK_FILENAME = ".regime_family_contradiction_shadow_ledger.lock"
EVALUATION_HORIZON_MINUTES = 60
NON_OVERLAP_MINUTES = 60
M1_SECONDS = 60
MAX_TERMINAL_OBSERVATION_LAG_SECONDS = 60
# Forecast persistence normally follows quote capture in the same call stack.
# The broker quote freshness contract is twenty seconds; older decision clocks
# are stale samples and would permit outcome-aware backfilling.  Five seconds
# of future skew permits ordinary cross-process clock/serialization jitter.
MAX_EMISSION_AGE_SECONDS = 20
MAX_EMISSION_FUTURE_SKEW_SECONDS = 5
MAX_CYCLE_ID_CHARS = 256
MAX_HOLDOUT_ID_CHARS = 128
MAX_SOURCE_ROWS = 10_000_000_000

_PAIR_RE = re.compile(r"^[A-Z]{3}_[A-Z]{3}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_RFC3339_UTC_RE = re.compile(
    r"^(?P<seconds>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?:Z|\+00:00)$"
)
_DIRECTIONS = frozenset({"UP", "DOWN"})
_DETECTOR_SCORE_KEYS = frozenset({"UP", "DOWN", "RANGE", "EITHER"})
_SHADOW_BODY_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "pair_id",
        "pair",
        "entry_reference_price",
        "entry_bid",
        "entry_ask",
        "entry_spread_pips",
        "entry_quote_source",
        "pip_factor",
        "detector_arm",
        "family_arm",
        "detector_scores",
        "technical_context_sha256",
        "regime_family_weighting_receipt_sha256",
        "primary_regime",
        "dominant_regime",
        "session",
        "situation_label",
        "selected_method",
        "weighted_directional_score",
        "directional_coverage_weight",
        "selected_family_coverage_weight",
        "evaluation_contract",
        "evaluation_horizon_minutes",
        "terminal_price_semantics",
        "round_trip_spread_included",
        "independent_non_overlap_required",
        "non_overlap_minutes",
        "holdout_lock_required_for_promotion_review",
        "transition_or_method_none",
        "promotion_exclusion_reason",
        "read_only",
        "shadow_only",
        "live_permission",
        "sizing_permission",
        "gate_relaxation_allowed",
        "automatic_promotion_allowed",
        "source_binding_verified_by_builder",
    }
)
_SHADOW_KEYS = _SHADOW_BODY_KEYS | {"shadow_sha256"}
_ARM_KEYS = frozenset({"arm_id", "direction", "source"})
_TRIAL_BODY_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "trial_id",
        "pair_id",
        "shadow_sha256",
        "pair",
        "cycle_id",
        "emitted_at_utc",
        "emission_as_of_utc",
        "evaluation_due_at_utc",
        "entry_reference_price",
        "entry_bid",
        "entry_ask",
        "entry_spread_pips",
        "pip_factor",
        "detector_arm",
        "family_arm",
        "detector_scores",
        "technical_context_sha256",
        "regime_family_weighting_receipt_sha256",
        "primary_regime",
        "dominant_regime",
        "session",
        "situation_label",
        "selected_method",
        "weighted_directional_score",
        "directional_coverage_weight",
        "selected_family_coverage_weight",
        "evaluation_contract",
        "evaluation_horizon_minutes",
        "sampling_contract",
        "non_overlap_scope",
        "non_overlap_minutes",
        "outcome_blind_selection_required",
        "holdout_lock",
        "holdout_lock_sha256",
        "holdout_membership_declared",
        "proof_eligible",
        "holdout_eligibility_reason",
        "resolution_status_at_emission",
        "transition_or_method_none",
        "promotion_exclusion_reason",
        "read_only",
        "shadow_only",
        "live_permission",
        "sizing_permission",
        "gate_relaxation_allowed",
        "automatic_promotion_allowed",
    }
)
_TRIAL_KEYS = _TRIAL_BODY_KEYS | {"trial_sha256"}
_RESULT_ARM_KEYS = frozenset(
    {
        "arm_id",
        "direction",
        "gross_mid_pips",
        "round_trip_spread_cost_pips",
        "executable_pips_before_non_spread_cost",
        "non_spread_cost_pips",
        "post_cost_pips",
        "post_cost_outcome",
    }
)
_RESULT_BODY_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "trial_id",
        "trial_sha256",
        "pair_id",
        "pair",
        "cycle_id",
        "emitted_at_utc",
        "evaluation_due_at_utc",
        "resolved_as_of_utc",
        "terminal_interval_start_utc",
        "terminal_interval_end_utc",
        "terminal_observed_at_utc",
        "terminal_observation_lag_seconds",
        "terminal_source_sha256",
        "terminal_source_semantics",
        "entry_bid",
        "entry_ask",
        "entry_mid",
        "entry_spread_pips",
        "pip_factor",
        "non_spread_cost_pips",
        "terminal_bid",
        "terminal_ask",
        "terminal_mid",
        "terminal_spread_pips",
        "terminal_delta_mid_pips",
        "terminal_direction",
        "detector_result",
        "family_result",
        "post_cost_winner",
        "same_pair_time_entry_and_terminal_truth",
        "round_trip_spread_included",
        "read_only",
        "shadow_only",
        "live_permission",
        "sizing_permission",
        "gate_relaxation_allowed",
        "automatic_promotion_allowed",
        "proof_eligible",
        "source_artifact_authenticated_by_evaluator",
    }
)
_RESULT_KEYS = _RESULT_BODY_KEYS | {"result_sha256"}
_HOLDOUT_BODY_KEYS = frozenset(
    {
        "contract",
        "holdout_id",
        "locked_at_utc",
        "holdout_start_utc",
        "holdout_end_utc",
        "source_prefix_sha256",
        "source_row_count",
        "selection_policy_sha256",
        "evaluation_horizon_minutes",
        "non_overlap_scope",
        "non_overlap_minutes",
        "predeclared_before_outcome",
        "immutable",
        "read_only",
        "live_permission",
    }
)
_HOLDOUT_KEYS = _HOLDOUT_BODY_KEYS | {"lock_sha256"}


@dataclass(frozen=True, order=True)
class _UtcInstant:
    epoch_second: int
    nanosecond: int = 0

    def plus_seconds(self, seconds: int) -> "_UtcInstant":
        return _UtcInstant(self.epoch_second + seconds, self.nanosecond)

    def nanoseconds_since(self, other: "_UtcInstant") -> int:
        return (
            (self.epoch_second - other.epoch_second) * 1_000_000_000
            + self.nanosecond
            - other.nanosecond
        )

    def canonical(self) -> str:
        base = datetime.fromtimestamp(self.epoch_second, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        fraction = f".{self.nanosecond:09d}".rstrip("0") if self.nanosecond else ""
        return f"{base}{fraction}Z"


@dataclass(frozen=True)
class M1BidAskCandle:
    """One complete OANDA-style M1 candle used only for its terminal close."""

    pair: str
    timestamp_utc: str
    bid_close: float
    ask_close: float
    complete: bool = True
    source_sha256: str | None = None


def build_regime_family_contradiction_shadow(
    *,
    pair: str,
    current_price: float,
    detector_direction: str,
    detector_scores: Mapping[str, Any],
    technical_context_v1: Mapping[str, Any],
    entry_bid: float | None = None,
    entry_ask: float | None = None,
) -> dict[str, Any]:
    """Freeze one detector-vs-family contradiction without execution rights."""

    pair_text = _canonical_pair(pair)
    price = _finite_positive(current_price, "current_price")
    direction = _canonical_direction(detector_direction, "detector_direction")
    scores = _canonical_detector_scores(detector_scores, winner=direction)
    context = _snapshot_mapping(technical_context_v1, "technical_context_v1")
    context_valid, context_error = verify_forecast_technical_context(
        context,
        pair=pair_text,
        current_price=price,
    )
    if not context_valid:
        raise ValueError(context_error or "TECHNICAL_CONTEXT_INVALID")
    completeness = context.get("completeness")
    if (
        not isinstance(completeness, Mapping)
        or completeness.get("complete") is not True
    ):
        raise ValueError("TECHNICAL_CONTEXT_INCOMPLETE")
    receipt = context.get("regime_family_weighting")
    receipt_valid, receipt_error = verify_regime_family_weighting_receipt(
        receipt,
        pair=pair_text,
    )
    if not receipt_valid or not isinstance(receipt, Mapping):
        raise ValueError(receipt_error or "REGIME_FAMILY_WEIGHTING_INVALID")
    binding_valid, binding_error = verify_regime_family_weighting_context_binding(
        receipt,
        technical_context=context,
    )
    if not binding_valid:
        raise ValueError(
            binding_error or "REGIME_FAMILY_WEIGHTING_CONTEXT_BINDING_INVALID"
        )

    aggregate = receipt.get("aggregate")
    source = receipt.get("source_identity")
    execution = context.get("execution")
    if not all(isinstance(value, Mapping) for value in (aggregate, source, execution)):
        raise ValueError("REGIME_FAMILY_WEIGHTING_SHADOW_SOURCE_INVALID")
    family_direction = _canonical_direction(
        aggregate.get("direction"), "family_direction"
    )
    if family_direction == direction:
        raise ValueError("REGIME_FAMILY_DIRECTION_NOT_CONTRADICTORY")

    pip_factor = float(instrument_pip_factor(pair_text))
    spread_pips = _finite_nonnegative(
        execution.get("spread_pips"), "technical_context spread_pips"
    )
    normalized_bid, normalized_ask, quote_source = _entry_bid_ask(
        current_price=price,
        spread_pips=spread_pips,
        pip_factor=pip_factor,
        entry_bid=entry_bid,
        entry_ask=entry_ask,
    )
    executable_spread = _round_number(
        (normalized_ask - normalized_bid) * pip_factor,
        9,
    )
    entry_reference_price = _round_number(
        (normalized_bid + normalized_ask) / 2.0,
        12,
    )
    location = context.get("location")
    if not isinstance(location, Mapping):
        raise ValueError("TECHNICAL_CONTEXT_LOCATION_INVALID")
    context_reference_price = _round_number(
        _finite_positive(
            location.get("current_price"),
            "technical_context current_price",
        ),
        12,
    )
    if entry_reference_price != context_reference_price:
        raise ValueError("ENTRY_REFERENCE_TECHNICAL_CONTEXT_MISMATCH")
    if executable_spread != _round_number(spread_pips, 9):
        raise ValueError("ENTRY_SPREAD_TECHNICAL_CONTEXT_MISMATCH")
    weighted_score = _finite_number(
        aggregate.get("weighted_directional_score"),
        "weighted_directional_score",
    )
    directional_coverage = _bounded_fraction(
        aggregate.get("directional_coverage_weight"),
        "directional_coverage_weight",
    )
    selected_coverage = _bounded_fraction(
        aggregate.get("selected_family_coverage_weight"),
        "selected_family_coverage_weight",
    )
    primary_regime = _bounded_text(source.get("primary_regime"), "primary_regime", 64)
    dominant_regime = _bounded_text(
        source.get("dominant_regime"), "dominant_regime", 64
    )
    session = _optional_bounded_text(source.get("session"), "session", 64)
    situation_label = _bounded_text(
        source.get("situation_label"), "situation_label", 256
    )
    selected_method = _optional_bounded_text(
        source.get("selected_method"), "selected_method", 96
    )
    transition_or_method_none = (
        primary_regime in {"TRANSITION", "UNKNOWN"} or selected_method is None
    )
    exclusion = (
        "PRIMARY_REGIME_TRANSITION_OR_METHOD_NONE"
        if transition_or_method_none
        else None
    )
    context_sha = _canonical_sha256_text(
        context.get("context_sha256"), "technical_context_sha256"
    )
    receipt_sha = _canonical_sha256_text(
        receipt.get("receipt_sha256"),
        "regime_family_weighting_receipt_sha256",
    )
    detector_arm = {
        "arm_id": "DETECTOR",
        "direction": direction,
        "source": "PRE_VETO_DIRECTIONAL_DETECTOR_WINNER",
    }
    family_arm = {
        "arm_id": "REGIME_FAMILY",
        "direction": family_direction,
        "source": "CONTENT_ADDRESSED_REGIME_FAMILY_AGGREGATE",
    }
    pair_material = {
        "contract": SHADOW_CONTRACT,
        "pair": pair_text,
        "entry_reference_price": entry_reference_price,
        "entry_bid": normalized_bid,
        "entry_ask": normalized_ask,
        "detector_arm": detector_arm,
        "family_arm": family_arm,
        "detector_scores": scores,
        "technical_context_sha256": context_sha,
        "regime_family_weighting_receipt_sha256": receipt_sha,
        "weighted_directional_score": _round_number(weighted_score, 12),
        "directional_coverage_weight": _round_number(directional_coverage, 12),
        "selected_family_coverage_weight": _round_number(selected_coverage, 12),
        "evaluation_contract": TERMINAL_EVALUATION_CONTRACT,
        "evaluation_horizon_minutes": EVALUATION_HORIZON_MINUTES,
    }
    pair_id = _canonical_sha256(pair_material)
    body: dict[str, Any] = {
        "contract": SHADOW_CONTRACT,
        "schema_version": 1,
        "pair_id": pair_id,
        "pair": pair_text,
        "entry_reference_price": entry_reference_price,
        "entry_bid": normalized_bid,
        "entry_ask": normalized_ask,
        "entry_spread_pips": executable_spread,
        "entry_quote_source": quote_source,
        "pip_factor": pip_factor,
        "detector_arm": detector_arm,
        "family_arm": family_arm,
        "detector_scores": scores,
        "technical_context_sha256": context_sha,
        "regime_family_weighting_receipt_sha256": receipt_sha,
        "primary_regime": primary_regime,
        "dominant_regime": dominant_regime,
        "session": session,
        "situation_label": situation_label,
        "selected_method": selected_method,
        "weighted_directional_score": _round_number(weighted_score, 12),
        "directional_coverage_weight": _round_number(directional_coverage, 12),
        "selected_family_coverage_weight": _round_number(selected_coverage, 12),
        "evaluation_contract": TERMINAL_EVALUATION_CONTRACT,
        "evaluation_horizon_minutes": EVALUATION_HORIZON_MINUTES,
        "terminal_price_semantics": (
            "LONG_ENTRY_ASK_EXIT_BID__SHORT_ENTRY_BID_EXIT_ASK"
        ),
        "round_trip_spread_included": True,
        "independent_non_overlap_required": True,
        "non_overlap_minutes": NON_OVERLAP_MINUTES,
        "holdout_lock_required_for_promotion_review": True,
        "transition_or_method_none": transition_or_method_none,
        "promotion_exclusion_reason": exclusion,
        "read_only": True,
        "shadow_only": True,
        "live_permission": False,
        "sizing_permission": False,
        "gate_relaxation_allowed": False,
        "automatic_promotion_allowed": False,
        "source_binding_verified_by_builder": True,
    }
    sealed = {**body, "shadow_sha256": _canonical_sha256(body)}
    issues = validate_regime_family_contradiction_shadow(sealed)
    if issues:
        raise ValueError("INVALID_CONTRADICTION_SHADOW:" + ",".join(issues))
    return sealed


def validate_regime_family_contradiction_shadow(value: object) -> tuple[str, ...]:
    """Validate internal shadow integrity without authenticating source bytes."""

    if not isinstance(value, Mapping):
        return ("SHADOW_NOT_MAPPING",)
    try:
        shadow = _snapshot_mapping(value, "shadow")
    except ValueError:
        return ("SHADOW_SNAPSHOT_UNREADABLE",)
    issues: list[str] = []
    if set(shadow) != _SHADOW_KEYS:
        issues.append("SHADOW_SCHEMA_INVALID")
    if shadow.get("contract") != SHADOW_CONTRACT or shadow.get("schema_version") != 1:
        issues.append("SHADOW_CONTRACT_INVALID")
    _permission_issues(shadow, issues)
    try:
        pair = _canonical_pair(shadow.get("pair"))
        price = _finite_positive(shadow.get("entry_reference_price"), "entry")
        bid = _finite_positive(shadow.get("entry_bid"), "entry_bid")
        ask = _finite_positive(shadow.get("entry_ask"), "entry_ask")
        pip_factor = _finite_positive(shadow.get("pip_factor"), "pip_factor")
        if pip_factor != float(instrument_pip_factor(pair)):
            issues.append("SHADOW_PIP_FACTOR_MISMATCH")
        spread = _finite_nonnegative(shadow.get("entry_spread_pips"), "spread")
        if not bid < ask or not math.isclose(
            spread,
            (ask - bid) * pip_factor,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            issues.append("SHADOW_ENTRY_QUOTE_INVALID")
        if price != _round_number((bid + ask) / 2.0, 12):
            issues.append("SHADOW_ENTRY_REFERENCE_MISMATCH")
        if shadow.get("entry_quote_source") not in {
            "EXPLICIT_BID_ASK",
            "REFERENCE_MID_PLUS_CONTEXT_SPREAD",
        }:
            issues.append("SHADOW_ENTRY_QUOTE_SOURCE_INVALID")
        detector = _validated_arm(shadow.get("detector_arm"), "DETECTOR")
        family = _validated_arm(shadow.get("family_arm"), "REGIME_FAMILY")
        if detector["direction"] == family["direction"]:
            issues.append("SHADOW_ARMS_NOT_CONTRADICTORY")
        scores = _canonical_detector_scores(
            _snapshot_mapping(shadow.get("detector_scores"), "detector_scores"),
            winner=detector["direction"],
        )
        if scores != shadow.get("detector_scores"):
            issues.append("SHADOW_DETECTOR_SCORES_NON_CANONICAL")
        for field in (
            "technical_context_sha256",
            "regime_family_weighting_receipt_sha256",
            "pair_id",
            "shadow_sha256",
        ):
            _canonical_sha256_text(shadow.get(field), field)
        for field in (
            "weighted_directional_score",
            "directional_coverage_weight",
            "selected_family_coverage_weight",
        ):
            if field.endswith("coverage_weight"):
                _bounded_fraction(shadow.get(field), field)
            else:
                _finite_number(shadow.get(field), field)
        if pair != shadow.get("pair"):
            issues.append("SHADOW_PAIR_NON_CANONICAL")
        pair_material = {
            "contract": SHADOW_CONTRACT,
            "pair": pair,
            "entry_reference_price": price,
            "entry_bid": bid,
            "entry_ask": ask,
            "detector_arm": detector,
            "family_arm": family,
            "detector_scores": scores,
            "technical_context_sha256": shadow.get("technical_context_sha256"),
            "regime_family_weighting_receipt_sha256": shadow.get(
                "regime_family_weighting_receipt_sha256"
            ),
            "weighted_directional_score": shadow.get("weighted_directional_score"),
            "directional_coverage_weight": shadow.get("directional_coverage_weight"),
            "selected_family_coverage_weight": shadow.get(
                "selected_family_coverage_weight"
            ),
            "evaluation_contract": TERMINAL_EVALUATION_CONTRACT,
            "evaluation_horizon_minutes": EVALUATION_HORIZON_MINUTES,
        }
        if shadow.get("pair_id") != _canonical_sha256(pair_material):
            issues.append("SHADOW_PAIR_ID_MISMATCH")
    except (TypeError, ValueError, OverflowError):
        issues.append("SHADOW_FIELD_INVALID")
    if (
        shadow.get("evaluation_contract") != TERMINAL_EVALUATION_CONTRACT
        or shadow.get("evaluation_horizon_minutes") != EVALUATION_HORIZON_MINUTES
        or shadow.get("terminal_price_semantics")
        != "LONG_ENTRY_ASK_EXIT_BID__SHORT_ENTRY_BID_EXIT_ASK"
        or shadow.get("round_trip_spread_included") is not True
        or shadow.get("independent_non_overlap_required") is not True
        or shadow.get("non_overlap_minutes") != NON_OVERLAP_MINUTES
        or shadow.get("holdout_lock_required_for_promotion_review") is not True
        or shadow.get("source_binding_verified_by_builder") is not True
    ):
        issues.append("SHADOW_EVALUATION_CONTRACT_INVALID")
    transition = shadow.get("transition_or_method_none")
    selected_method = shadow.get("selected_method")
    expected_transition = (
        shadow.get("primary_regime") in {"TRANSITION", "UNKNOWN"}
        or selected_method is None
    )
    if transition is not expected_transition or shadow.get(
        "promotion_exclusion_reason"
    ) != ("PRIMARY_REGIME_TRANSITION_OR_METHOD_NONE" if expected_transition else None):
        issues.append("SHADOW_PROMOTION_EXCLUSION_INVALID")
    stored_sha = shadow.get("shadow_sha256")
    try:
        expected_sha = _canonical_sha256(
            {key: shadow.get(key) for key in sorted(_SHADOW_BODY_KEYS)}
        )
        if stored_sha != expected_sha:
            issues.append("SHADOW_SHA256_MISMATCH")
    except (TypeError, ValueError, OverflowError):
        issues.append("SHADOW_SHA256_INVALID")
    return tuple(dict.fromkeys(issues))


def verify_regime_family_contradiction_source_binding(
    shadow: object,
    *,
    technical_context_v1: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Revalidate the full technical source against a stored shadow digest."""

    issues = validate_regime_family_contradiction_shadow(shadow)
    if issues or not isinstance(shadow, Mapping):
        return False, issues[0] if issues else "SHADOW_INVALID"
    try:
        context = _snapshot_mapping(technical_context_v1, "technical_context_v1")
    except ValueError:
        return False, "TECHNICAL_CONTEXT_SNAPSHOT_INVALID"
    valid, error = verify_forecast_technical_context(
        context,
        pair=str(shadow.get("pair") or ""),
        current_price=float(shadow.get("entry_reference_price") or 0.0),
    )
    if not valid:
        return False, error or "TECHNICAL_CONTEXT_INVALID"
    receipt = context.get("regime_family_weighting")
    if not isinstance(receipt, Mapping):
        return False, "REGIME_FAMILY_WEIGHTING_MISSING"
    receipt_valid, receipt_error = verify_regime_family_weighting_receipt(
        receipt,
        pair=str(shadow.get("pair") or ""),
    )
    binding_valid, binding_error = verify_regime_family_weighting_context_binding(
        receipt,
        technical_context=context,
    )
    if not receipt_valid or not binding_valid:
        return False, receipt_error or binding_error or "SOURCE_BINDING_INVALID"
    if context.get("context_sha256") != shadow.get(
        "technical_context_sha256"
    ) or receipt.get("receipt_sha256") != shadow.get(
        "regime_family_weighting_receipt_sha256"
    ):
        return False, "SHADOW_SOURCE_DIGEST_MISMATCH"
    try:
        execution = _snapshot_mapping(context.get("execution"), "execution")
        location = _snapshot_mapping(context.get("location"), "location")
        aggregate = _snapshot_mapping(receipt.get("aggregate"), "aggregate")
        source = _snapshot_mapping(receipt.get("source_identity"), "source_identity")
        expected_reference_price = _round_number(
            _finite_positive(
                location.get("current_price"),
                "technical_context current_price",
            ),
            12,
        )
        expected_spread_pips = _round_number(
            _finite_nonnegative(
                execution.get("spread_pips"),
                "technical_context spread_pips",
            ),
            9,
        )
        family_direction = _canonical_direction(
            aggregate.get("direction"),
            "family_direction",
        )
        primary_regime = _bounded_text(
            source.get("primary_regime"),
            "primary_regime",
            64,
        )
        dominant_regime = _bounded_text(
            source.get("dominant_regime"),
            "dominant_regime",
            64,
        )
        session = _optional_bounded_text(source.get("session"), "session", 64)
        situation_label = _bounded_text(
            source.get("situation_label"),
            "situation_label",
            256,
        )
        selected_method = _optional_bounded_text(
            source.get("selected_method"),
            "selected_method",
            96,
        )
        transition_or_method_none = (
            primary_regime in {"TRANSITION", "UNKNOWN"} or selected_method is None
        )
        expected_fields: dict[str, Any] = {
            "entry_reference_price": expected_reference_price,
            "entry_spread_pips": expected_spread_pips,
            "family_arm": {
                "arm_id": "REGIME_FAMILY",
                "direction": family_direction,
                "source": "CONTENT_ADDRESSED_REGIME_FAMILY_AGGREGATE",
            },
            "primary_regime": primary_regime,
            "dominant_regime": dominant_regime,
            "session": session,
            "situation_label": situation_label,
            "selected_method": selected_method,
            "weighted_directional_score": _round_number(
                _finite_number(
                    aggregate.get("weighted_directional_score"),
                    "weighted_directional_score",
                ),
                12,
            ),
            "directional_coverage_weight": _round_number(
                _bounded_fraction(
                    aggregate.get("directional_coverage_weight"),
                    "directional_coverage_weight",
                ),
                12,
            ),
            "selected_family_coverage_weight": _round_number(
                _bounded_fraction(
                    aggregate.get("selected_family_coverage_weight"),
                    "selected_family_coverage_weight",
                ),
                12,
            ),
            "transition_or_method_none": transition_or_method_none,
            "promotion_exclusion_reason": (
                "PRIMARY_REGIME_TRANSITION_OR_METHOD_NONE"
                if transition_or_method_none
                else None
            ),
        }
    except (TypeError, ValueError, OverflowError):
        return False, "SHADOW_SOURCE_FIELDS_INVALID"
    for field, expected in expected_fields.items():
        if shadow.get(field) != expected:
            return False, f"SHADOW_SOURCE_BINDING_{field.upper()}_MISMATCH"
    return True, None


def seal_regime_family_contradiction_holdout_lock(
    *,
    holdout_id: str,
    locked_at_utc: str,
    holdout_start_utc: str,
    holdout_end_utc: str,
    source_prefix_sha256: str,
    source_row_count: int,
    selection_policy_sha256: str,
) -> dict[str, Any]:
    """Seal a pre-outcome holdout boundary for later promotion review."""

    lock_id = _bounded_id(holdout_id, "holdout_id", MAX_HOLDOUT_ID_CHARS)
    locked = _required_utc(locked_at_utc, "locked_at_utc")
    start = _required_utc(holdout_start_utc, "holdout_start_utc")
    end = _required_utc(holdout_end_utc, "holdout_end_utc")
    if locked > start or start >= end:
        raise ValueError("HOLDOUT_BOUNDARY_INVALID")
    if (
        source_row_count.__class__ is not int
        or not 0 <= source_row_count <= MAX_SOURCE_ROWS
    ):
        raise ValueError("HOLDOUT_SOURCE_ROW_COUNT_INVALID")
    body = {
        "contract": HOLDOUT_LOCK_CONTRACT,
        "holdout_id": lock_id,
        "locked_at_utc": locked.canonical(),
        "holdout_start_utc": start.canonical(),
        "holdout_end_utc": end.canonical(),
        "source_prefix_sha256": _canonical_sha256_text(
            source_prefix_sha256, "source_prefix_sha256"
        ),
        "source_row_count": source_row_count,
        "selection_policy_sha256": _canonical_sha256_text(
            selection_policy_sha256, "selection_policy_sha256"
        ),
        "evaluation_horizon_minutes": EVALUATION_HORIZON_MINUTES,
        "non_overlap_scope": "PAIR",
        "non_overlap_minutes": NON_OVERLAP_MINUTES,
        "predeclared_before_outcome": True,
        "immutable": True,
        "read_only": True,
        "live_permission": False,
    }
    return {**body, "lock_sha256": _canonical_sha256(body)}


def validate_regime_family_contradiction_holdout_lock(
    value: object,
) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("HOLDOUT_LOCK_NOT_MAPPING",)
    lock = _snapshot_mapping(value, "holdout_lock")
    issues: list[str] = []
    if set(lock) != _HOLDOUT_KEYS:
        issues.append("HOLDOUT_LOCK_SCHEMA_INVALID")
    try:
        _bounded_id(lock.get("holdout_id"), "holdout_id", MAX_HOLDOUT_ID_CHARS)
        locked = _required_utc(lock.get("locked_at_utc"), "locked_at_utc")
        start = _required_utc(lock.get("holdout_start_utc"), "holdout_start_utc")
        end = _required_utc(lock.get("holdout_end_utc"), "holdout_end_utc")
        if locked > start or start >= end:
            issues.append("HOLDOUT_BOUNDARY_INVALID")
        _canonical_sha256_text(lock.get("source_prefix_sha256"), "source_prefix")
        _canonical_sha256_text(lock.get("selection_policy_sha256"), "policy")
        count = lock.get("source_row_count")
        if count.__class__ is not int or not 0 <= count <= MAX_SOURCE_ROWS:
            issues.append("HOLDOUT_SOURCE_ROW_COUNT_INVALID")
    except (TypeError, ValueError, OverflowError):
        issues.append("HOLDOUT_LOCK_FIELD_INVALID")
    if (
        lock.get("contract") != HOLDOUT_LOCK_CONTRACT
        or lock.get("evaluation_horizon_minutes") != EVALUATION_HORIZON_MINUTES
        or lock.get("non_overlap_scope") != "PAIR"
        or lock.get("non_overlap_minutes") != NON_OVERLAP_MINUTES
        or lock.get("predeclared_before_outcome") is not True
        or lock.get("immutable") is not True
        or lock.get("read_only") is not True
        or lock.get("live_permission") is not False
    ):
        issues.append("HOLDOUT_LOCK_CONTRACT_INVALID")
    try:
        if lock.get("lock_sha256") != _canonical_sha256(
            {key: lock.get(key) for key in sorted(_HOLDOUT_BODY_KEYS)}
        ):
            issues.append("HOLDOUT_LOCK_SHA256_MISMATCH")
    except (TypeError, ValueError, OverflowError):
        issues.append("HOLDOUT_LOCK_SHA256_INVALID")
    return tuple(dict.fromkeys(issues))


def bind_regime_family_contradiction_emission(
    shadow: Mapping[str, Any],
    *,
    emitted_at_utc: str | datetime,
    cycle_id: str,
    holdout_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a decision-time shadow to one immutable emission trial."""

    issues = validate_regime_family_contradiction_shadow(shadow)
    if issues:
        raise ValueError("INVALID_CONTRADICTION_SHADOW:" + ",".join(issues))
    frozen = _snapshot_mapping(shadow, "shadow")
    emitted = _coerce_utc(emitted_at_utc, "emitted_at_utc")
    cycle = _bounded_id(cycle_id, "cycle_id", MAX_CYCLE_ID_CHARS)
    due = emitted.plus_seconds(EVALUATION_HORIZON_MINUTES * 60)
    frozen_lock: dict[str, Any] | None = None
    lock_sha: str | None = None
    if holdout_lock is not None:
        lock_issues = validate_regime_family_contradiction_holdout_lock(holdout_lock)
        if lock_issues:
            raise ValueError("INVALID_HOLDOUT_LOCK:" + ",".join(lock_issues))
        frozen_lock = deepcopy(dict(holdout_lock))
        lock_start = _required_utc(frozen_lock["holdout_start_utc"], "holdout_start")
        lock_end = _required_utc(frozen_lock["holdout_end_utc"], "holdout_end")
        locked_at = _required_utc(frozen_lock["locked_at_utc"], "locked_at")
        if not (locked_at <= lock_start <= emitted < lock_end):
            raise ValueError("EMISSION_OUTSIDE_PREDECLARED_HOLDOUT")
        lock_sha = str(frozen_lock["lock_sha256"])
    trial_identity = {
        "contract": TRIAL_CONTRACT,
        "pair_id": frozen["pair_id"],
        "pair": frozen["pair"],
        "emitted_at_utc": emitted.canonical(),
        "evaluation_due_at_utc": due.canonical(),
    }
    body: dict[str, Any] = {
        "contract": TRIAL_CONTRACT,
        "schema_version": 1,
        "trial_id": _canonical_sha256(trial_identity),
        "pair_id": frozen["pair_id"],
        "shadow_sha256": frozen["shadow_sha256"],
        "pair": frozen["pair"],
        "cycle_id": cycle,
        "emitted_at_utc": emitted.canonical(),
        "emission_as_of_utc": emitted.canonical(),
        "evaluation_due_at_utc": due.canonical(),
        "entry_reference_price": frozen["entry_reference_price"],
        "entry_bid": frozen["entry_bid"],
        "entry_ask": frozen["entry_ask"],
        "entry_spread_pips": frozen["entry_spread_pips"],
        "pip_factor": frozen["pip_factor"],
        "detector_arm": deepcopy(frozen["detector_arm"]),
        "family_arm": deepcopy(frozen["family_arm"]),
        "detector_scores": deepcopy(frozen["detector_scores"]),
        "technical_context_sha256": frozen["technical_context_sha256"],
        "regime_family_weighting_receipt_sha256": frozen[
            "regime_family_weighting_receipt_sha256"
        ],
        "primary_regime": frozen["primary_regime"],
        "dominant_regime": frozen["dominant_regime"],
        "session": frozen["session"],
        "situation_label": frozen["situation_label"],
        "selected_method": frozen["selected_method"],
        "weighted_directional_score": frozen["weighted_directional_score"],
        "directional_coverage_weight": frozen["directional_coverage_weight"],
        "selected_family_coverage_weight": frozen["selected_family_coverage_weight"],
        "evaluation_contract": TERMINAL_EVALUATION_CONTRACT,
        "evaluation_horizon_minutes": EVALUATION_HORIZON_MINUTES,
        "sampling_contract": SELECTION_CONTRACT,
        "non_overlap_scope": "PAIR",
        "non_overlap_minutes": NON_OVERLAP_MINUTES,
        "outcome_blind_selection_required": True,
        "holdout_lock": frozen_lock,
        "holdout_lock_sha256": lock_sha,
        "holdout_membership_declared": frozen_lock is not None,
        # A caller-supplied lock and timestamp are necessary metadata, but this
        # pure contract cannot authenticate the external emission clock or
        # source prefix.  Promotion review therefore remains fail-closed.
        "proof_eligible": False,
        "holdout_eligibility_reason": (
            "EXTERNAL_EMISSION_ANCHOR_AND_SOURCE_AUTHENTICATION_REQUIRED"
            if frozen_lock is not None
            else "HOLDOUT_LOCK_MISSING"
        ),
        "resolution_status_at_emission": "PENDING",
        "transition_or_method_none": frozen["transition_or_method_none"],
        "promotion_exclusion_reason": frozen["promotion_exclusion_reason"],
        "read_only": True,
        "shadow_only": True,
        "live_permission": False,
        "sizing_permission": False,
        "gate_relaxation_allowed": False,
        "automatic_promotion_allowed": False,
    }
    sealed = {**body, "trial_sha256": _canonical_sha256(body)}
    trial_issues = validate_regime_family_contradiction_trial(sealed)
    if trial_issues:
        raise ValueError("INVALID_CONTRADICTION_TRIAL:" + ",".join(trial_issues))
    return sealed


def validate_regime_family_contradiction_trial(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("TRIAL_NOT_MAPPING",)
    trial = _snapshot_mapping(value, "trial")
    issues: list[str] = []
    if set(trial) != _TRIAL_KEYS:
        issues.append("TRIAL_SCHEMA_INVALID")
    _permission_issues(trial, issues)
    try:
        pair = _canonical_pair(trial.get("pair"))
        _bounded_id(trial.get("cycle_id"), "cycle_id", MAX_CYCLE_ID_CHARS)
        emitted = _required_utc(trial.get("emitted_at_utc"), "emitted_at")
        due = _required_utc(trial.get("evaluation_due_at_utc"), "due_at")
        if trial.get("emission_as_of_utc") != emitted.canonical():
            issues.append("TRIAL_AS_OF_MISMATCH")
        if due != emitted.plus_seconds(EVALUATION_HORIZON_MINUTES * 60):
            issues.append("TRIAL_DUE_AT_MISMATCH")
        detector = _validated_arm(trial.get("detector_arm"), "DETECTOR")
        family = _validated_arm(trial.get("family_arm"), "REGIME_FAMILY")
        if detector["direction"] == family["direction"]:
            issues.append("TRIAL_ARMS_NOT_CONTRADICTORY")
        scores = _canonical_detector_scores(
            _snapshot_mapping(trial.get("detector_scores"), "detector_scores"),
            winner=detector["direction"],
        )
        for field in (
            "trial_id",
            "pair_id",
            "shadow_sha256",
            "technical_context_sha256",
            "regime_family_weighting_receipt_sha256",
            "trial_sha256",
        ):
            _canonical_sha256_text(trial.get(field), field)
        reference_price = _finite_positive(
            trial.get("entry_reference_price"),
            "entry_reference_price",
        )
        bid = _finite_positive(trial.get("entry_bid"), "entry_bid")
        ask = _finite_positive(trial.get("entry_ask"), "entry_ask")
        pip_factor = _finite_positive(trial.get("pip_factor"), "pip_factor")
        if pip_factor != float(instrument_pip_factor(pair)):
            issues.append("TRIAL_PIP_FACTOR_MISMATCH")
        spread = _finite_nonnegative(trial.get("entry_spread_pips"), "spread")
        if not bid < ask or not math.isclose(
            spread,
            (ask - bid) * pip_factor,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            issues.append("TRIAL_ENTRY_QUOTE_INVALID")
        if reference_price != _round_number((bid + ask) / 2.0, 12):
            issues.append("TRIAL_ENTRY_REFERENCE_MISMATCH")
        weighted_score = _finite_number(
            trial.get("weighted_directional_score"),
            "weighted_directional_score",
        )
        directional_coverage = _bounded_fraction(
            trial.get("directional_coverage_weight"),
            "directional_coverage_weight",
        )
        selected_coverage = _bounded_fraction(
            trial.get("selected_family_coverage_weight"),
            "selected_family_coverage_weight",
        )
        pair_material = {
            "contract": SHADOW_CONTRACT,
            "pair": pair,
            "entry_reference_price": reference_price,
            "entry_bid": bid,
            "entry_ask": ask,
            "detector_arm": detector,
            "family_arm": family,
            "detector_scores": scores,
            "technical_context_sha256": trial.get("technical_context_sha256"),
            "regime_family_weighting_receipt_sha256": trial.get(
                "regime_family_weighting_receipt_sha256"
            ),
            "weighted_directional_score": weighted_score,
            "directional_coverage_weight": directional_coverage,
            "selected_family_coverage_weight": selected_coverage,
            "evaluation_contract": TERMINAL_EVALUATION_CONTRACT,
            "evaluation_horizon_minutes": EVALUATION_HORIZON_MINUTES,
        }
        if trial.get("pair_id") != _canonical_sha256(pair_material):
            issues.append("TRIAL_PAIR_ID_MISMATCH")
        identity = {
            "contract": TRIAL_CONTRACT,
            "pair_id": trial.get("pair_id"),
            "pair": pair,
            "emitted_at_utc": emitted.canonical(),
            "evaluation_due_at_utc": due.canonical(),
        }
        if trial.get("trial_id") != _canonical_sha256(identity):
            issues.append("TRIAL_ID_MISMATCH")
    except (TypeError, ValueError, OverflowError):
        issues.append("TRIAL_FIELD_INVALID")
    lock = trial.get("holdout_lock")
    lock_sha = trial.get("holdout_lock_sha256")
    membership_declared = trial.get("holdout_membership_declared")
    if lock is None:
        if (
            lock_sha is not None
            or membership_declared is not False
            or trial.get("holdout_eligibility_reason") != "HOLDOUT_LOCK_MISSING"
        ):
            issues.append("TRIAL_HOLDOUT_STATE_INVALID")
    else:
        lock_issues = validate_regime_family_contradiction_holdout_lock(lock)
        if lock_issues or not isinstance(lock, Mapping):
            issues.append("TRIAL_HOLDOUT_LOCK_INVALID")
        elif (
            lock_sha != lock.get("lock_sha256")
            or membership_declared is not True
            or trial.get("holdout_eligibility_reason")
            != "EXTERNAL_EMISSION_ANCHOR_AND_SOURCE_AUTHENTICATION_REQUIRED"
        ):
            issues.append("TRIAL_HOLDOUT_BINDING_INVALID")
        else:
            try:
                emitted = _required_utc(trial.get("emitted_at_utc"), "emitted")
                start = _required_utc(lock.get("holdout_start_utc"), "start")
                end = _required_utc(lock.get("holdout_end_utc"), "end")
                locked = _required_utc(lock.get("locked_at_utc"), "locked")
                if not (locked <= start <= emitted < end):
                    issues.append("TRIAL_OUTSIDE_HOLDOUT")
            except ValueError:
                issues.append("TRIAL_HOLDOUT_TIME_INVALID")
    if (
        trial.get("contract") != TRIAL_CONTRACT
        or trial.get("schema_version") != 1
        or trial.get("evaluation_contract") != TERMINAL_EVALUATION_CONTRACT
        or trial.get("evaluation_horizon_minutes") != EVALUATION_HORIZON_MINUTES
        or trial.get("sampling_contract") != SELECTION_CONTRACT
        or trial.get("non_overlap_scope") != "PAIR"
        or trial.get("non_overlap_minutes") != NON_OVERLAP_MINUTES
        or trial.get("outcome_blind_selection_required") is not True
        or trial.get("resolution_status_at_emission") != "PENDING"
        or trial.get("proof_eligible") is not False
    ):
        issues.append("TRIAL_CONTRACT_INVALID")
    try:
        if trial.get("trial_sha256") != _canonical_sha256(
            {key: trial.get(key) for key in sorted(_TRIAL_BODY_KEYS)}
        ):
            issues.append("TRIAL_SHA256_MISMATCH")
    except (TypeError, ValueError, OverflowError):
        issues.append("TRIAL_SHA256_INVALID")
    return tuple(dict.fromkeys(issues))


def persist_regime_family_contradiction_emission(
    shadow: Mapping[str, Any] | None,
    *,
    technical_context_v1: Mapping[str, Any],
    emitted_at_utc: str | datetime,
    cycle_id: str,
    data_root: Path,
    holdout_lock: Mapping[str, Any] | None = None,
    replace_existing: bool = False,
) -> int:
    """Persist one cycle/pair trial; absent shadow is an intentional no-op.

    A malformed present shadow raises and fails closed.  Replacement appends a
    superseding event rather than rewriting history, and is forbidden after a
    resolution event has bound the current trial.
    """

    if shadow is None:
        return 0
    shadow_issues = validate_regime_family_contradiction_shadow(shadow)
    if shadow_issues:
        raise ValueError("INVALID_CONTRADICTION_SHADOW:" + ",".join(shadow_issues))
    source_valid, source_error = verify_regime_family_contradiction_source_binding(
        shadow,
        technical_context_v1=technical_context_v1,
    )
    if not source_valid:
        raise ValueError(
            "INVALID_CONTRADICTION_SHADOW_SOURCE_BINDING:" + (source_error or "UNKNOWN")
        )
    # A midpoint reconstructed from spread is useful diagnostic material but
    # is not an authenticated forward executable quote.  Do not let it enter
    # the persisted learning cohort.  The caller can rebuild the shadow with
    # an explicit bid/ask snapshot from the same forecast observation.
    if shadow.get("entry_quote_source") != "EXPLICIT_BID_ASK":
        return 0
    trial = bind_regime_family_contradiction_emission(
        shadow,
        emitted_at_utc=emitted_at_utc,
        cycle_id=cycle_id,
        holdout_lock=holdout_lock,
    )
    return _persist_trial_event(
        trial,
        data_root=Path(data_root),
        replace_existing=replace_existing,
    )


def resolve_regime_family_contradiction_trial(
    trial: Mapping[str, Any],
    *,
    terminal_bid: float,
    terminal_ask: float,
    terminal_interval_start_utc: str | datetime,
    terminal_observed_at_utc: str | datetime,
    resolved_as_of_utc: str | datetime,
    terminal_source_sha256: str,
    non_spread_cost_pips: float = 0.0,
) -> dict[str, Any]:
    """Score both arms at the same fixed terminal bid/ask observation."""

    issues = validate_regime_family_contradiction_trial(trial)
    if issues:
        raise ValueError("INVALID_CONTRADICTION_TRIAL:" + ",".join(issues))
    frozen = _snapshot_mapping(trial, "trial")
    bid = _finite_positive(terminal_bid, "terminal_bid")
    ask = _finite_positive(terminal_ask, "terminal_ask")
    if not bid < ask:
        raise ValueError("TERMINAL_BID_ASK_INVALID")
    interval_start = _coerce_utc(
        terminal_interval_start_utc, "terminal_interval_start_utc"
    )
    observed = _coerce_utc(terminal_observed_at_utc, "terminal_observed_at_utc")
    as_of = _coerce_utc(resolved_as_of_utc, "resolved_as_of_utc")
    due = _required_utc(frozen["evaluation_due_at_utc"], "evaluation_due_at_utc")
    if observed != interval_start.plus_seconds(M1_SECONDS):
        raise ValueError("TERMINAL_INTERVAL_MUST_BE_EXACT_COMPLETE_M1")
    if not _minute_aligned(interval_start):
        raise ValueError("TERMINAL_INTERVAL_START_NOT_OANDA_M1_ALIGNED")
    lag_ns = observed.nanoseconds_since(due)
    if lag_ns < 0 or lag_ns >= MAX_TERMINAL_OBSERVATION_LAG_SECONDS * 1_000_000_000:
        raise ValueError("TERMINAL_OBSERVATION_OUTSIDE_FIXED_60M_BOUND")
    if observed > as_of:
        raise ValueError("TERMINAL_OBSERVATION_AFTER_AS_OF")
    source_sha = _canonical_sha256_text(
        terminal_source_sha256, "terminal_source_sha256"
    )
    extra_cost = _finite_nonnegative(non_spread_cost_pips, "non_spread_cost_pips")
    pip_factor = float(frozen["pip_factor"])
    entry_bid = float(frozen["entry_bid"])
    entry_ask = float(frozen["entry_ask"])
    entry_mid = (entry_bid + entry_ask) / 2.0
    terminal_mid = (bid + ask) / 2.0
    terminal_delta_pips = (terminal_mid - entry_mid) * pip_factor
    terminal_direction = (
        "UP"
        if terminal_delta_pips > 0.0
        else "DOWN"
        if terminal_delta_pips < 0.0
        else "FLAT"
    )
    round_trip_spread_cost = ((entry_ask - entry_bid) + (ask - bid)) * pip_factor / 2.0

    def arm_result(arm: Mapping[str, Any]) -> dict[str, Any]:
        direction = str(arm["direction"])
        gross_mid = terminal_delta_pips if direction == "UP" else -terminal_delta_pips
        executable = (
            (bid - entry_ask) * pip_factor
            if direction == "UP"
            else (entry_bid - ask) * pip_factor
        )
        post_cost = executable - extra_cost
        stored_post_cost = _round_number(post_cost, 9)
        return {
            "arm_id": arm["arm_id"],
            "direction": direction,
            "gross_mid_pips": _round_number(gross_mid, 9),
            "round_trip_spread_cost_pips": _round_number(round_trip_spread_cost, 9),
            "executable_pips_before_non_spread_cost": _round_number(executable, 9),
            "non_spread_cost_pips": _round_number(extra_cost, 9),
            "post_cost_pips": stored_post_cost,
            "post_cost_outcome": (
                "WIN"
                if stored_post_cost > 0.0
                else "LOSS"
                if stored_post_cost < 0.0
                else "FLAT"
            ),
        }

    detector_result = arm_result(frozen["detector_arm"])
    family_result = arm_result(frozen["family_arm"])
    detector_net = float(detector_result["post_cost_pips"])
    family_net = float(family_result["post_cost_pips"])
    if detector_net > 0.0 and detector_net > family_net:
        winner = "DETECTOR"
    elif family_net > 0.0 and family_net > detector_net:
        winner = "REGIME_FAMILY"
    elif detector_net == family_net and detector_net > 0.0:
        winner = "TIE_POSITIVE"
    else:
        winner = "NEITHER_POST_COST_POSITIVE"
    body = {
        "contract": RESULT_CONTRACT,
        "schema_version": 1,
        "trial_id": frozen["trial_id"],
        "trial_sha256": frozen["trial_sha256"],
        "pair_id": frozen["pair_id"],
        "pair": frozen["pair"],
        "cycle_id": frozen["cycle_id"],
        "emitted_at_utc": frozen["emitted_at_utc"],
        "evaluation_due_at_utc": frozen["evaluation_due_at_utc"],
        "resolved_as_of_utc": as_of.canonical(),
        "terminal_interval_start_utc": interval_start.canonical(),
        "terminal_interval_end_utc": observed.canonical(),
        "terminal_observed_at_utc": observed.canonical(),
        "terminal_observation_lag_seconds": _round_number(lag_ns / 1e9, 9),
        "terminal_source_sha256": source_sha,
        "terminal_source_semantics": (
            "FIRST_COMPLETE_M1_BID_ASK_INTERVAL_END_AT_OR_AFTER_EXACT_DUE"
        ),
        "entry_bid": _round_number(entry_bid, 12),
        "entry_ask": _round_number(entry_ask, 12),
        "entry_mid": _round_number(entry_mid, 12),
        "entry_spread_pips": _round_number((entry_ask - entry_bid) * pip_factor, 9),
        "pip_factor": pip_factor,
        "non_spread_cost_pips": _round_number(extra_cost, 9),
        "terminal_bid": _round_number(bid, 12),
        "terminal_ask": _round_number(ask, 12),
        "terminal_mid": _round_number(terminal_mid, 12),
        "terminal_spread_pips": _round_number((ask - bid) * pip_factor, 9),
        "terminal_delta_mid_pips": _round_number(terminal_delta_pips, 9),
        "terminal_direction": terminal_direction,
        "detector_result": detector_result,
        "family_result": family_result,
        "post_cost_winner": winner,
        "same_pair_time_entry_and_terminal_truth": True,
        "round_trip_spread_included": True,
        "read_only": True,
        "shadow_only": True,
        "live_permission": False,
        "sizing_permission": False,
        "gate_relaxation_allowed": False,
        "automatic_promotion_allowed": False,
        "proof_eligible": False,
        "source_artifact_authenticated_by_evaluator": False,
    }
    return {**body, "result_sha256": _canonical_sha256(body)}


def resolve_due_regime_family_contradiction_trials(
    trials: Sequence[Mapping[str, Any]],
    candles: Sequence[M1BidAskCandle | Mapping[str, Any]],
    *,
    as_of_utc: str | datetime,
    already_resolved_trial_ids: Iterable[str] = (),
    non_spread_cost_pips: float = 0.0,
) -> dict[str, Any]:
    """Resolve due trials from local complete M1 bid/ask candle truth.

    This function performs no network or broker writes.  Candle timestamps are
    OANDA candle-open times; the observable terminal timestamp is therefore
    exactly one minute later.  The earliest complete close on/after the 60m
    due clock is selected before looking at its price.
    """

    as_of = _coerce_utc(as_of_utc, "as_of_utc")
    resolved = set(already_resolved_trial_ids)
    normalized_candles = _normalized_m1_candles(candles, as_of=as_of)
    by_pair: dict[str, list[tuple[_UtcInstant, M1BidAskCandle, str]]] = {}
    for close_at, candle, source_sha in normalized_candles:
        by_pair.setdefault(candle.pair, []).append((close_at, candle, source_sha))
    for rows in by_pair.values():
        rows.sort(key=lambda item: item[0])
    results: list[dict[str, Any]] = []
    pending_due_without_truth: list[str] = []
    not_due_count = 0
    invalid_trials: list[dict[str, Any]] = []
    for supplied in trials:
        issues = validate_regime_family_contradiction_trial(supplied)
        if issues:
            invalid_trials.append(
                {"trial_id": supplied.get("trial_id"), "issues": list(issues)}
            )
            continue
        trial = _snapshot_mapping(supplied, "trial")
        trial_id = str(trial["trial_id"])
        if trial_id in resolved:
            continue
        due = _required_utc(trial["evaluation_due_at_utc"], "due")
        if due > as_of:
            not_due_count += 1
            continue
        chosen: tuple[_UtcInstant, M1BidAskCandle, str] | None = None
        for candidate in by_pair.get(str(trial["pair"]), ()):  # outcome-blind order
            close_at = candidate[0]
            lag_ns = close_at.nanoseconds_since(due)
            if 0 <= lag_ns < MAX_TERMINAL_OBSERVATION_LAG_SECONDS * 1_000_000_000:
                chosen = candidate
                break
            if lag_ns >= MAX_TERMINAL_OBSERVATION_LAG_SECONDS * 1_000_000_000:
                break
        if chosen is None:
            pending_due_without_truth.append(trial_id)
            continue
        close_at, candle, source_sha = chosen
        results.append(
            resolve_regime_family_contradiction_trial(
                trial,
                terminal_bid=candle.bid_close,
                terminal_ask=candle.ask_close,
                terminal_interval_start_utc=candle.timestamp_utc,
                terminal_observed_at_utc=close_at.canonical(),
                resolved_as_of_utc=as_of.canonical(),
                terminal_source_sha256=source_sha,
                non_spread_cost_pips=non_spread_cost_pips,
            )
        )
    return {
        "contract": RESULT_CONTRACT,
        "status": "INVALID_TRIALS" if invalid_trials else "OK",
        "as_of_utc": as_of.canonical(),
        "resolved_results": results,
        "resolved_count": len(results),
        "already_resolved_count": len(resolved),
        "not_due_count": not_due_count,
        "pending_due_without_truth": pending_due_without_truth,
        "invalid_trials": invalid_trials,
        "read_only": True,
        "live_permission": False,
    }


def select_independent_regime_family_contradiction_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    as_of_utc: str | datetime | None = None,
    require_locked_holdout: bool = False,
    ledger_recorded_at_by_trial_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select pair-scoped, outcome-blind 60m non-overlapping trials."""

    as_of = _coerce_utc(as_of_utc, "as_of_utc") if as_of_utc is not None else None
    candidates: list[tuple[_UtcInstant, int, dict[str, Any]]] = []
    invalid: list[dict[str, Any]] = []
    excluded_after_as_of = 0
    excluded_unlocked = 0
    recording_anchor_missing = 0
    recording_anchor_invalid = 0
    recording_anchor_after_due = 0
    for index, supplied in enumerate(trials):
        issues = validate_regime_family_contradiction_trial(supplied)
        if issues:
            invalid.append({"index": index, "issues": list(issues)})
            continue
        frozen = _snapshot_mapping(supplied, "trial")
        emitted = _required_utc(frozen["emitted_at_utc"], "emitted")
        if as_of is not None and emitted > as_of:
            excluded_after_as_of += 1
            continue
        if require_locked_holdout and frozen["holdout_membership_declared"] is not True:
            excluded_unlocked += 1
            continue
        recorded_raw = (
            ledger_recorded_at_by_trial_id.get(str(frozen["trial_id"]))
            if isinstance(ledger_recorded_at_by_trial_id, Mapping)
            else None
        )
        if recorded_raw is None:
            recording_anchor_missing += 1
        else:
            try:
                recorded = _required_utc(recorded_raw, "ledger_recorded_at")
                due = _required_utc(frozen["evaluation_due_at_utc"], "due")
            except ValueError:
                recording_anchor_invalid += 1
                continue
            if recorded >= due:
                recording_anchor_after_due += 1
                continue
            if as_of is not None and recorded > as_of:
                excluded_after_as_of += 1
                continue
        candidates.append((emitted, index, frozen))
    candidates.sort(key=lambda item: (item[0], item[1]))
    pair_next_at: dict[str, _UtcInstant] = {}
    selected: list[dict[str, Any]] = []
    skipped_overlap = 0
    for emitted, _index, trial in candidates:
        pair = str(trial["pair"])
        if pair in pair_next_at and emitted < pair_next_at[pair]:
            skipped_overlap += 1
            continue
        selected.append(trial)
        pair_next_at[pair] = emitted.plus_seconds(NON_OVERLAP_MINUTES * 60)
    selection_material = {
        "contract": SELECTION_CONTRACT,
        "as_of_utc": as_of.canonical() if as_of is not None else None,
        "require_locked_holdout": require_locked_holdout,
        "selected_trial_ids": [trial["trial_id"] for trial in selected],
        "non_overlap_scope": "PAIR",
        "non_overlap_minutes": NON_OVERLAP_MINUTES,
    }
    return {
        **selection_material,
        "selection_sha256": _canonical_sha256(selection_material),
        "selected_trials": selected,
        "selected_count": len(selected),
        "skipped_overlapping_count": skipped_overlap,
        "excluded_after_as_of_count": excluded_after_as_of,
        "excluded_unlocked_count": excluded_unlocked,
        "recording_anchor_missing_count": recording_anchor_missing,
        "recording_anchor_invalid_count": recording_anchor_invalid,
        "recording_anchor_after_due_count": recording_anchor_after_due,
        "external_source_authentication_verified": False,
        "proof_eligible": False,
        "invalid_trials": invalid,
        "outcome_fields_read_by_selector": False,
        "read_only": True,
        "live_permission": False,
    }


def load_regime_family_contradiction_ledger(
    data_root: Path,
) -> dict[str, Any]:
    """Validate the hash chain and return current trials/results."""

    path = Path(data_root) / LEDGER_FILENAME
    lock_path = Path(data_root) / LEDGER_LOCK_FILENAME
    if not path.exists():
        return {
            "status": "MISSING",
            "events": [],
            "trials": [],
            "results": [],
            "historical_trial_ids": [],
            "ledger_recorded_at_by_trial_id": {},
            "last_event_sha256": None,
        }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
        try:
            with path.open("r", encoding="utf-8") as handle:
                events = _validated_ledger_events(handle)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    state = _current_ledger_state(events)
    return {
        "status": "VALID",
        "events": events,
        "trials": list(state["trials_by_key"].values()),
        "results": list(state["results_by_trial_id"].values()),
        "historical_trial_ids": sorted(state["historical_trial_ids"]),
        "ledger_recorded_at_by_trial_id": state["ledger_recorded_at_by_trial_id"],
        "last_event_sha256": events[-1]["event_sha256"] if events else None,
    }


def persist_regime_family_contradiction_results(
    results: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
) -> int:
    """Append internally valid, idempotent terminal result events."""

    appended = 0
    for result in results:
        appended += _persist_result_event(result, data_root=Path(data_root))
    return appended


def resolve_due_regime_family_contradiction_ledger(
    *,
    data_root: Path,
    candles: Sequence[M1BidAskCandle | Mapping[str, Any]],
    as_of_utc: str | datetime,
    non_spread_cost_pips: float = 0.0,
) -> dict[str, Any]:
    """Load, deterministically resolve, and append results to the shadow ledger."""

    loaded = load_regime_family_contradiction_ledger(data_root)
    if loaded["status"] == "MISSING":
        return {
            "contract": RESULT_CONTRACT,
            "status": "NO_LEDGER",
            "resolved_count": 0,
            "persisted_count": 0,
            "read_only": True,
            "live_permission": False,
        }
    resolution = resolve_due_regime_family_contradiction_trials(
        loaded["trials"],
        candles,
        as_of_utc=as_of_utc,
        already_resolved_trial_ids=(result["trial_id"] for result in loaded["results"]),
        non_spread_cost_pips=non_spread_cost_pips,
    )
    if resolution["status"] != "OK":
        resolution["persisted_count"] = 0
        return resolution
    resolution["persisted_count"] = persist_regime_family_contradiction_results(
        resolution["resolved_results"],
        data_root=data_root,
    )
    return resolution


def _persist_trial_event(
    trial: Mapping[str, Any],
    *,
    data_root: Path,
    replace_existing: bool,
) -> int:
    path = data_root / LEDGER_FILENAME
    lock_path = data_root / LEDGER_LOCK_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            events = []
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    events = _validated_ledger_events(handle)
            state = _current_ledger_state(events)
            key = (str(trial["cycle_id"]), str(trial["pair"]))
            existing = state["trials_by_key"].get(key)
            if trial.get("trial_id") in state["historical_trial_ids"]:
                # A superseded observation remains tombstoned forever.  A
                # later scheduler/cycle cannot resurrect A after A→B and turn
                # one old market fact into another statistical sample.
                return 0
            same_observation = next(
                (
                    candidate
                    for candidate in state["trials_by_key"].values()
                    if candidate.get("trial_id") == trial.get("trial_id")
                ),
                None,
            )
            # Runtime retries can mint a different cycle id for the identical
            # market observation.  Store it once; cycle identity is lineage,
            # not an independent statistical sample.
            if same_observation is not None:
                return 0
            if existing is not None:
                if existing.get("trial_sha256") == trial.get("trial_sha256"):
                    return 0
                if not replace_existing:
                    raise ValueError("CONTRADICTION_TRIAL_IDENTITY_COLLISION")
                if existing["trial_id"] in state["results_by_trial_id"]:
                    raise ValueError("RESOLVED_CONTRADICTION_TRIAL_CANNOT_BE_REPLACED")
                operation = "REPLACE_EMISSION"
                supersedes = existing["trial_sha256"]
            else:
                operation = "APPEND_EMISSION"
                supersedes = None
            event = _ledger_event(
                operation=operation,
                payload=dict(trial),
                sequence=len(events) + 1,
                previous_event_sha256=(events[-1]["event_sha256"] if events else None),
                supersedes_sha256=supersedes,
            )
            recorded_at = _validate_new_event_recorded_at(
                event,
                previous_events=events,
            )
            emitted_at = _required_utc(trial["emitted_at_utc"], "emitted_at_utc")
            future_skew_ns = emitted_at.nanoseconds_since(recorded_at)
            if future_skew_ns > MAX_EMISSION_FUTURE_SKEW_SECONDS * 1_000_000_000:
                raise ValueError("EMISSION_TIMESTAMP_EXCEEDS_LEDGER_FUTURE_SKEW")
            emission_age_ns = recorded_at.nanoseconds_since(emitted_at)
            if emission_age_ns > MAX_EMISSION_AGE_SECONDS * 1_000_000_000:
                raise ValueError("EMISSION_TIMESTAMP_EXCEEDS_LEDGER_MAX_AGE")
            if existing is not None:
                existing_due_at = _required_utc(
                    existing["evaluation_due_at_utc"],
                    "existing evaluation_due_at_utc",
                )
                if recorded_at >= existing_due_at:
                    raise ValueError(
                        "REPLACEMENT_RECORDED_AT_OR_AFTER_SUPERSEDED_OUTCOME_WINDOW"
                    )
            due_at = _required_utc(
                trial["evaluation_due_at_utc"], "evaluation_due_at_utc"
            )
            if recorded_at >= due_at:
                raise ValueError("EMISSION_RECORDED_AT_OR_AFTER_OUTCOME_WINDOW")
            _append_ledger_event(path, event)
            return 1
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _persist_result_event(result: Mapping[str, Any], *, data_root: Path) -> int:
    result_issues = validate_regime_family_contradiction_result(result)
    if result_issues:
        raise ValueError("INVALID_CONTRADICTION_RESULT:" + ",".join(result_issues))
    path = data_root / LEDGER_FILENAME
    lock_path = data_root / LEDGER_LOCK_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            events: list[dict[str, Any]] = []
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    events = _validated_ledger_events(handle)
            state = _current_ledger_state(events)
            trial_id = str(result["trial_id"])
            trial = next(
                (
                    value
                    for value in state["trials_by_key"].values()
                    if value["trial_id"] == trial_id
                ),
                None,
            )
            if trial is None or trial["trial_sha256"] != result["trial_sha256"]:
                raise ValueError("CONTRADICTION_RESULT_TRIAL_NOT_CURRENT")
            binding_issues = validate_regime_family_contradiction_result_binding(
                result,
                trial,
            )
            if binding_issues:
                raise ValueError(
                    "INVALID_CONTRADICTION_RESULT_TRIAL_BINDING:"
                    + ",".join(binding_issues)
                )
            existing = state["results_by_trial_id"].get(trial_id)
            if existing is not None:
                if existing.get("result_sha256") == result.get("result_sha256"):
                    return 0
                raise ValueError("CONTRADICTION_RESULT_IDENTITY_COLLISION")
            event = _ledger_event(
                operation="APPEND_RESULT",
                payload=dict(result),
                sequence=len(events) + 1,
                previous_event_sha256=(events[-1]["event_sha256"] if events else None),
                supersedes_sha256=None,
            )
            recorded_at = _validate_new_event_recorded_at(
                event,
                previous_events=events,
            )
            terminal_observed_at = _required_utc(
                result["terminal_observed_at_utc"],
                "terminal_observed_at_utc",
            )
            resolved_as_of = _required_utc(
                result["resolved_as_of_utc"],
                "resolved_as_of_utc",
            )
            if recorded_at < terminal_observed_at or recorded_at < resolved_as_of:
                raise ValueError("RESULT_RECORDED_BEFORE_RESOLUTION_CLOCK")
            _append_ledger_event(path, event)
            return 1
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def validate_regime_family_contradiction_result(
    value: object,
) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ("RESULT_NOT_MAPPING",)
    result = _snapshot_mapping(value, "result")
    issues: list[str] = []
    if set(result) != _RESULT_KEYS:
        issues.append("RESULT_SCHEMA_INVALID")
    if result.get("contract") != RESULT_CONTRACT or result.get("schema_version") != 1:
        issues.append("RESULT_CONTRACT_INVALID")
    for field in (
        "trial_id",
        "trial_sha256",
        "pair_id",
        "terminal_source_sha256",
        "result_sha256",
    ):
        try:
            _canonical_sha256_text(result.get(field), field)
        except ValueError:
            issues.append(f"RESULT_{field.upper()}_INVALID")
    _permission_issues(result, issues)
    if (
        result.get("same_pair_time_entry_and_terminal_truth") is not True
        or result.get("round_trip_spread_included") is not True
        or result.get("proof_eligible") is not False
        or result.get("source_artifact_authenticated_by_evaluator") is not False
        or result.get("terminal_source_semantics")
        != "FIRST_COMPLETE_M1_BID_ASK_INTERVAL_END_AT_OR_AFTER_EXACT_DUE"
    ):
        issues.append("RESULT_PAIRING_INVALID")
    try:
        pair = _canonical_pair(result.get("pair"))
        _bounded_id(result.get("cycle_id"), "cycle_id", MAX_CYCLE_ID_CHARS)
        emitted = _required_utc(result.get("emitted_at_utc"), "emitted_at")
        due = _required_utc(result.get("evaluation_due_at_utc"), "due_at")
        interval_start = _required_utc(
            result.get("terminal_interval_start_utc"), "interval_start"
        )
        interval_end = _required_utc(
            result.get("terminal_interval_end_utc"), "interval_end"
        )
        observed = _required_utc(result.get("terminal_observed_at_utc"), "observed_at")
        as_of = _required_utc(result.get("resolved_as_of_utc"), "resolved_as_of")
        if due != emitted.plus_seconds(EVALUATION_HORIZON_MINUTES * 60):
            issues.append("RESULT_DUE_AT_MISMATCH")
        expected_trial_id = _canonical_sha256(
            {
                "contract": TRIAL_CONTRACT,
                "pair_id": result.get("pair_id"),
                "pair": pair,
                "emitted_at_utc": emitted.canonical(),
                "evaluation_due_at_utc": due.canonical(),
            }
        )
        if result.get("trial_id") != expected_trial_id:
            issues.append("RESULT_TRIAL_ID_MISMATCH")
        if interval_end != interval_start.plus_seconds(M1_SECONDS):
            issues.append("RESULT_TERMINAL_INTERVAL_INVALID")
        if not _minute_aligned(interval_start):
            issues.append("RESULT_TERMINAL_INTERVAL_NOT_M1_ALIGNED")
        if observed != interval_end or observed > as_of:
            issues.append("RESULT_OBSERVATION_CLOCK_INVALID")
        lag_ns = observed.nanoseconds_since(due)
        if not 0 <= lag_ns < MAX_TERMINAL_OBSERVATION_LAG_SECONDS * 1_000_000_000:
            issues.append("RESULT_TERMINAL_LAG_INVALID")
        stored_lag = _finite_nonnegative(
            result.get("terminal_observation_lag_seconds"), "terminal lag"
        )
        if not math.isclose(stored_lag, lag_ns / 1e9, rel_tol=0.0, abs_tol=1e-9):
            issues.append("RESULT_TERMINAL_LAG_MISMATCH")

        pip_factor = _finite_positive(result.get("pip_factor"), "pip_factor")
        if pip_factor != float(instrument_pip_factor(pair)):
            issues.append("RESULT_PIP_FACTOR_MISMATCH")
        entry_bid = _finite_positive(result.get("entry_bid"), "entry_bid")
        entry_ask = _finite_positive(result.get("entry_ask"), "entry_ask")
        terminal_bid = _finite_positive(result.get("terminal_bid"), "terminal_bid")
        terminal_ask = _finite_positive(result.get("terminal_ask"), "terminal_ask")
        if not entry_bid < entry_ask or not terminal_bid < terminal_ask:
            issues.append("RESULT_BID_ASK_INVALID")
        entry_mid = (entry_bid + entry_ask) / 2.0
        terminal_mid = (terminal_bid + terminal_ask) / 2.0
        entry_spread = (entry_ask - entry_bid) * pip_factor
        terminal_spread = (terminal_ask - terminal_bid) * pip_factor
        terminal_delta = (terminal_mid - entry_mid) * pip_factor
        additional_cost = _finite_nonnegative(
            result.get("non_spread_cost_pips"), "non_spread_cost_pips"
        )
        number_expectations = {
            "entry_mid": entry_mid,
            "entry_spread_pips": entry_spread,
            "terminal_mid": terminal_mid,
            "terminal_spread_pips": terminal_spread,
            "terminal_delta_mid_pips": terminal_delta,
        }
        for field, expected_number in number_expectations.items():
            stored_number = _finite_number(result.get(field), field)
            if not math.isclose(
                stored_number,
                expected_number,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                issues.append(f"RESULT_{field.upper()}_MISMATCH")
        expected_terminal_direction = (
            "UP" if terminal_delta > 0.0 else "DOWN" if terminal_delta < 0.0 else "FLAT"
        )
        if result.get("terminal_direction") != expected_terminal_direction:
            issues.append("RESULT_TERMINAL_DIRECTION_MISMATCH")
        round_trip_spread = (entry_spread + terminal_spread) / 2.0

        arm_values: dict[str, tuple[str, float]] = {}
        for field, expected_id in (
            ("detector_result", "DETECTOR"),
            ("family_result", "REGIME_FAMILY"),
        ):
            arm = result.get(field)
            if not isinstance(arm, Mapping) or set(arm) != _RESULT_ARM_KEYS:
                issues.append(f"RESULT_{expected_id}_SCHEMA_INVALID")
                continue
            direction = _canonical_direction(
                arm.get("direction"), "result arm direction"
            )
            if arm.get("arm_id") != expected_id:
                issues.append(f"RESULT_{expected_id}_ID_INVALID")
            gross = terminal_delta if direction == "UP" else -terminal_delta
            executable = (
                (terminal_bid - entry_ask) * pip_factor
                if direction == "UP"
                else (entry_bid - terminal_ask) * pip_factor
            )
            post_cost = executable - additional_cost
            canonical_post_cost = _round_number(post_cost, 9)
            arm_expectations = {
                "gross_mid_pips": gross,
                "round_trip_spread_cost_pips": round_trip_spread,
                "executable_pips_before_non_spread_cost": executable,
                "non_spread_cost_pips": additional_cost,
                "post_cost_pips": canonical_post_cost,
            }
            for arm_field, expected_number in arm_expectations.items():
                stored_number = _finite_number(arm.get(arm_field), arm_field)
                if not math.isclose(
                    stored_number,
                    expected_number,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                ):
                    issues.append(f"RESULT_{expected_id}_{arm_field.upper()}_MISMATCH")
            expected_outcome = (
                "WIN"
                if canonical_post_cost > 0.0
                else "LOSS"
                if canonical_post_cost < 0.0
                else "FLAT"
            )
            if arm.get("post_cost_outcome") != expected_outcome:
                issues.append(f"RESULT_{expected_id}_OUTCOME_MISMATCH")
            arm_values[expected_id] = (direction, canonical_post_cost)
        if set(arm_values) == {"DETECTOR", "REGIME_FAMILY"}:
            if arm_values["DETECTOR"][0] == arm_values["REGIME_FAMILY"][0]:
                issues.append("RESULT_ARMS_NOT_CONTRADICTORY")
            detector_net = arm_values["DETECTOR"][1]
            family_net = arm_values["REGIME_FAMILY"][1]
            if detector_net > 0.0 and detector_net > family_net:
                expected_winner = "DETECTOR"
            elif family_net > 0.0 and family_net > detector_net:
                expected_winner = "REGIME_FAMILY"
            elif detector_net == family_net and detector_net > 0.0:
                expected_winner = "TIE_POSITIVE"
            else:
                expected_winner = "NEITHER_POST_COST_POSITIVE"
            if result.get("post_cost_winner") != expected_winner:
                issues.append("RESULT_POST_COST_WINNER_MISMATCH")
    except (TypeError, ValueError, OverflowError):
        issues.append("RESULT_ECONOMICS_OR_TIME_INVALID")
    try:
        expected = _canonical_sha256(
            {key: result.get(key) for key in sorted(_RESULT_BODY_KEYS)}
        )
        if result.get("result_sha256") != expected:
            issues.append("RESULT_SHA256_MISMATCH")
    except (TypeError, ValueError, OverflowError):
        issues.append("RESULT_SHA256_INVALID")
    return tuple(dict.fromkeys(issues))


def validate_regime_family_contradiction_result_binding(
    result_value: object,
    trial_value: object,
) -> tuple[str, ...]:
    """Bind a self-consistent result to the exact current emission trial.

    Result validation alone deliberately recomputes terminal economics from
    the result's own quotes and arm directions.  Without this second check, an
    attacker could swap DETECTOR and REGIME_FAMILY directions, recompute a
    perfectly self-consistent winner and digest, and still retain the original
    ``trial_id``/``trial_sha256``.  Persistence and ledger replay therefore
    compare every immutable field copied from the current trial.
    """

    result_issues = validate_regime_family_contradiction_result(result_value)
    if result_issues:
        return tuple(f"RESULT_INVALID:{issue}" for issue in result_issues)
    trial_issues = validate_regime_family_contradiction_trial(trial_value)
    if trial_issues:
        return tuple(f"TRIAL_INVALID:{issue}" for issue in trial_issues)
    assert isinstance(result_value, Mapping)
    assert isinstance(trial_value, Mapping)
    result = _snapshot_mapping(result_value, "result")
    trial = _snapshot_mapping(trial_value, "trial")
    issues: list[str] = []
    immutable_field_bindings = {
        "trial_id": "trial_id",
        "trial_sha256": "trial_sha256",
        "pair_id": "pair_id",
        "pair": "pair",
        "cycle_id": "cycle_id",
        "emitted_at_utc": "emitted_at_utc",
        "evaluation_due_at_utc": "evaluation_due_at_utc",
        "entry_bid": "entry_bid",
        "entry_ask": "entry_ask",
        "entry_spread_pips": "entry_spread_pips",
        "pip_factor": "pip_factor",
    }
    for result_field, trial_field in immutable_field_bindings.items():
        if result.get(result_field) != trial.get(trial_field):
            issues.append(f"RESULT_TRIAL_{result_field.upper()}_MISMATCH")

    detector_result = result.get("detector_result")
    family_result = result.get("family_result")
    detector_trial = trial.get("detector_arm")
    family_trial = trial.get("family_arm")
    if not isinstance(detector_result, Mapping) or not isinstance(
        detector_trial, Mapping
    ):
        issues.append("RESULT_TRIAL_DETECTOR_ARM_MISSING")
    elif detector_result.get("arm_id") != detector_trial.get(
        "arm_id"
    ) or detector_result.get("direction") != detector_trial.get("direction"):
        issues.append("RESULT_TRIAL_DETECTOR_ARM_MISMATCH")
    if not isinstance(family_result, Mapping) or not isinstance(family_trial, Mapping):
        issues.append("RESULT_TRIAL_FAMILY_ARM_MISSING")
    elif family_result.get("arm_id") != family_trial.get("arm_id") or family_result.get(
        "direction"
    ) != family_trial.get("direction"):
        issues.append("RESULT_TRIAL_FAMILY_ARM_MISMATCH")

    # Standalone result validation recomputes these from result.entry_bid/ask;
    # repeat their binding to the trial reference explicitly so a future result
    # schema extension cannot accidentally bypass the immutable entry quote.
    expected_entry_mid = _round_number(
        (float(trial["entry_bid"]) + float(trial["entry_ask"])) / 2.0,
        12,
    )
    if result.get("entry_mid") != expected_entry_mid:
        issues.append("RESULT_TRIAL_ENTRY_MID_MISMATCH")
    return tuple(dict.fromkeys(issues))


def _ledger_event(
    *,
    operation: str,
    payload: Mapping[str, Any],
    sequence: int,
    previous_event_sha256: str | None,
    supersedes_sha256: str | None,
) -> dict[str, Any]:
    payload_sha = str(payload.get("result_sha256") or payload.get("trial_sha256") or "")
    body = {
        "contract": LEDGER_EVENT_CONTRACT,
        "sequence": sequence,
        "operation": operation,
        "recorded_at_utc": _coerce_utc(
            _ledger_recorded_at_utc(), "recorded_at_utc"
        ).canonical(),
        "payload_sha256": payload_sha,
        "previous_event_sha256": previous_event_sha256,
        "supersedes_sha256": supersedes_sha256,
        "payload": deepcopy(dict(payload)),
        "read_only": True,
        "live_permission": False,
    }
    return {**body, "event_sha256": _canonical_sha256(body)}


def _ledger_recorded_at_utc() -> datetime:
    """Return the non-caller-controlled wall clock used by ledger events."""

    return datetime.now(timezone.utc)


def _validate_new_event_recorded_at(
    event: Mapping[str, Any],
    *,
    previous_events: Sequence[Mapping[str, Any]],
) -> _UtcInstant:
    recorded_at = _required_utc(
        event.get("recorded_at_utc"),
        "ledger recorded_at_utc",
    )
    if previous_events:
        previous_recorded_at = _required_utc(
            previous_events[-1].get("recorded_at_utc"),
            "previous ledger recorded_at_utc",
        )
        if recorded_at < previous_recorded_at:
            raise ValueError("LEDGER_RECORDED_AT_REGRESSION")
    return recorded_at


def _append_ledger_event(path: Path, event: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(event) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validated_ledger_events(handle: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: str | None = None
    previous_recorded_at: _UtcInstant | None = None
    for line_number, raw in enumerate(handle, 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"CONTRADICTION_LEDGER_JSON_INVALID:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"CONTRADICTION_LEDGER_EVENT_INVALID:{line_number}")
        body = {key: item for key, item in value.items() if key != "event_sha256"}
        if (
            value.get("contract") != LEDGER_EVENT_CONTRACT
            or value.get("sequence") != len(events) + 1
            or value.get("previous_event_sha256") != previous
            or value.get("read_only") is not True
            or value.get("live_permission") is not False
            or value.get("event_sha256") != _canonical_sha256(body)
        ):
            raise ValueError(f"CONTRADICTION_LEDGER_CHAIN_INVALID:{line_number}")
        payload = value.get("payload")
        operation = value.get("operation")
        if operation in {"APPEND_EMISSION", "REPLACE_EMISSION"}:
            issues = validate_regime_family_contradiction_trial(payload)
            expected_payload_sha = (
                payload.get("trial_sha256") if isinstance(payload, Mapping) else None
            )
        elif operation == "APPEND_RESULT":
            issues = validate_regime_family_contradiction_result(payload)
            expected_payload_sha = (
                payload.get("result_sha256") if isinstance(payload, Mapping) else None
            )
        else:
            raise ValueError(f"CONTRADICTION_LEDGER_OPERATION_INVALID:{line_number}")
        if issues or value.get("payload_sha256") != expected_payload_sha:
            raise ValueError(f"CONTRADICTION_LEDGER_PAYLOAD_INVALID:{line_number}")
        try:
            recorded_at = _required_utc(
                value.get("recorded_at_utc"), "ledger recorded_at_utc"
            )
        except ValueError as exc:
            raise ValueError(
                f"CONTRADICTION_LEDGER_RECORDED_AT_INVALID:{line_number}"
            ) from exc
        if previous_recorded_at is not None and recorded_at < previous_recorded_at:
            raise ValueError(
                f"CONTRADICTION_LEDGER_RECORDED_AT_REGRESSION:{line_number}"
            )
        if operation in {"APPEND_EMISSION", "REPLACE_EMISSION"}:
            assert isinstance(payload, Mapping)
            emitted_at = _required_utc(payload.get("emitted_at_utc"), "emitted_at")
            if (
                emitted_at.nanoseconds_since(recorded_at)
                > MAX_EMISSION_FUTURE_SKEW_SECONDS * 1_000_000_000
            ):
                raise ValueError(
                    f"CONTRADICTION_LEDGER_EMISSION_FUTURE_SKEW:{line_number}"
                )
            if (
                recorded_at.nanoseconds_since(emitted_at)
                > MAX_EMISSION_AGE_SECONDS * 1_000_000_000
            ):
                raise ValueError(f"CONTRADICTION_LEDGER_EMISSION_STALE:{line_number}")
            due_at = _required_utc(payload.get("evaluation_due_at_utc"), "due_at")
            if recorded_at >= due_at:
                raise ValueError(
                    f"CONTRADICTION_LEDGER_BACKFILLED_AFTER_DUE:{line_number}"
                )
        elif operation == "APPEND_RESULT":
            assert isinstance(payload, Mapping)
            terminal_observed_at = _required_utc(
                payload.get("terminal_observed_at_utc"),
                "terminal_observed_at_utc",
            )
            resolved_as_of = _required_utc(
                payload.get("resolved_as_of_utc"),
                "resolved_as_of_utc",
            )
            if recorded_at < terminal_observed_at or recorded_at < resolved_as_of:
                raise ValueError(f"CONTRADICTION_LEDGER_RESULT_PREMATURE:{line_number}")
        events.append(value)
        previous = str(value["event_sha256"])
        previous_recorded_at = recorded_at
    _current_ledger_state(events)  # validates operation semantics too
    return events


def _current_ledger_state(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trials_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    results_by_trial_id: dict[str, dict[str, Any]] = {}
    ledger_recorded_at_by_trial_id: dict[str, str] = {}
    historical_trial_ids: set[str] = set()
    for event in events:
        operation = event.get("operation")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("CONTRADICTION_LEDGER_PAYLOAD_INVALID")
        if operation in {"APPEND_EMISSION", "REPLACE_EMISSION"}:
            key = (str(payload["cycle_id"]), str(payload["pair"]))
            trial_id = str(payload["trial_id"])
            if trial_id in historical_trial_ids:
                raise ValueError(
                    "CONTRADICTION_LEDGER_HISTORICAL_OBSERVATION_RESURRECTED"
                )
            previous = trials_by_key.get(key)
            if operation == "APPEND_EMISSION" and previous is not None:
                raise ValueError("CONTRADICTION_LEDGER_DUPLICATE_APPEND")
            if operation == "REPLACE_EMISSION":
                if previous is None or event.get("supersedes_sha256") != previous.get(
                    "trial_sha256"
                ):
                    raise ValueError("CONTRADICTION_LEDGER_REPLACE_INVALID")
                if previous["trial_id"] in results_by_trial_id:
                    raise ValueError("CONTRADICTION_LEDGER_REPLACE_AFTER_RESULT")
                recorded_at = _required_utc(
                    event.get("recorded_at_utc"),
                    "replacement recorded_at_utc",
                )
                previous_due_at = _required_utc(
                    previous.get("evaluation_due_at_utc"),
                    "superseded evaluation_due_at_utc",
                )
                if recorded_at >= previous_due_at:
                    raise ValueError("CONTRADICTION_LEDGER_OUTCOME_AWARE_REPLACEMENT")
                ledger_recorded_at_by_trial_id.pop(str(previous["trial_id"]), None)
            duplicate_observation = next(
                (
                    trial
                    for other_key, trial in trials_by_key.items()
                    if other_key != key and trial["trial_id"] == payload["trial_id"]
                ),
                None,
            )
            if duplicate_observation is not None:
                raise ValueError("CONTRADICTION_LEDGER_DUPLICATE_OBSERVATION")
            trials_by_key[key] = deepcopy(dict(payload))
            historical_trial_ids.add(trial_id)
            ledger_recorded_at_by_trial_id[trial_id] = str(event["recorded_at_utc"])
        elif operation == "APPEND_RESULT":
            trial_id = str(payload["trial_id"])
            if trial_id in results_by_trial_id:
                raise ValueError("CONTRADICTION_LEDGER_DUPLICATE_RESULT")
            current = next(
                (
                    trial
                    for trial in trials_by_key.values()
                    if trial["trial_id"] == trial_id
                ),
                None,
            )
            if current is None or current["trial_sha256"] != payload["trial_sha256"]:
                raise ValueError("CONTRADICTION_LEDGER_RESULT_ORPHANED")
            binding_issues = validate_regime_family_contradiction_result_binding(
                payload,
                current,
            )
            if binding_issues:
                raise ValueError(
                    "CONTRADICTION_LEDGER_RESULT_TRIAL_BINDING_INVALID:"
                    + ",".join(binding_issues)
                )
            results_by_trial_id[trial_id] = deepcopy(dict(payload))
    return {
        "trials_by_key": trials_by_key,
        "results_by_trial_id": results_by_trial_id,
        "historical_trial_ids": historical_trial_ids,
        "ledger_recorded_at_by_trial_id": ledger_recorded_at_by_trial_id,
    }


def _normalized_m1_candles(
    values: Sequence[M1BidAskCandle | Mapping[str, Any]],
    *,
    as_of: _UtcInstant,
) -> list[tuple[_UtcInstant, M1BidAskCandle, str]]:
    # One broker interval is one economic fact.  Input order must never choose
    # between two prices for the same pair/open clock.  Exact/economically
    # identical duplicates are collapsed; conflicting duplicates fail the
    # whole resolver before any outcome is calculated or persisted.
    by_interval: dict[
        tuple[str, str],
        tuple[_UtcInstant, M1BidAskCandle, set[str]],
    ] = {}
    for supplied in values:
        candle, material = _coerce_m1_candle(supplied)
        if candle.complete is not True:
            continue
        pair = _canonical_pair(candle.pair)
        opened = _coerce_utc(candle.timestamp_utc, "candle timestamp")
        if not _minute_aligned(opened):
            raise ValueError("M1_CANDLE_TIMESTAMP_NOT_MINUTE_ALIGNED")
        close_at = opened.plus_seconds(M1_SECONDS)
        if close_at > as_of:
            continue
        bid = _finite_positive(candle.bid_close, "bid_close")
        ask = _finite_positive(candle.ask_close, "ask_close")
        if not bid < ask:
            raise ValueError("M1_CANDLE_BID_ASK_INVALID")
        source_sha = candle.source_sha256
        if source_sha is None:
            source_sha = _canonical_sha256(material)
        else:
            source_sha = _canonical_sha256_text(source_sha, "candle source_sha256")
        canonical = M1BidAskCandle(
            pair=pair,
            timestamp_utc=opened.canonical(),
            bid_close=_round_number(bid, 12),
            ask_close=_round_number(ask, 12),
            complete=True,
            source_sha256=source_sha,
        )
        interval_key = (pair, opened.canonical())
        existing = by_interval.get(interval_key)
        if existing is not None:
            _existing_close, existing_candle, source_shas = existing
            if (
                existing_candle.bid_close != canonical.bid_close
                or existing_candle.ask_close != canonical.ask_close
            ):
                raise ValueError("M1_CANDLE_DUPLICATE_INTERVAL_CONFLICT")
            source_shas.add(source_sha)
            continue
        by_interval[interval_key] = (close_at, canonical, {source_sha})

    rows: list[tuple[_UtcInstant, M1BidAskCandle, str]] = []
    for (pair, opened_at), (close_at, candle, source_shas) in sorted(
        by_interval.items()
    ):
        if len(source_shas) == 1:
            canonical_source_sha = next(iter(source_shas))
        else:
            canonical_source_sha = _canonical_sha256(
                {
                    "contract": "QR_M1_BID_ASK_DUPLICATE_SOURCE_SET_V1",
                    "pair": pair,
                    "interval_start_utc": opened_at,
                    "bid_close": candle.bid_close,
                    "ask_close": candle.ask_close,
                    "source_sha256s": sorted(source_shas),
                }
            )
        rows.append(
            (
                close_at,
                M1BidAskCandle(
                    pair=candle.pair,
                    timestamp_utc=candle.timestamp_utc,
                    bid_close=candle.bid_close,
                    ask_close=candle.ask_close,
                    complete=True,
                    source_sha256=canonical_source_sha,
                ),
                canonical_source_sha,
            )
        )
    return rows


def _coerce_m1_candle(
    value: M1BidAskCandle | Mapping[str, Any],
) -> tuple[M1BidAskCandle, dict[str, Any]]:
    if value.__class__ is M1BidAskCandle:
        candle = value
        material = {
            "pair": candle.pair,
            "timestamp_utc": candle.timestamp_utc,
            "bid_close": candle.bid_close,
            "ask_close": candle.ask_close,
            "complete": candle.complete,
        }
        return candle, material
    raw = _snapshot_mapping(value, "M1 candle")
    pair = raw.get("pair") or raw.get("instrument")
    timestamp = raw.get("timestamp_utc") or raw.get("time")
    bid_raw = raw.get("bid")
    ask_raw = raw.get("ask")
    bid_close = (
        bid_raw.get("c", bid_raw.get("close"))
        if isinstance(bid_raw, Mapping)
        else raw.get("bid_close")
    )
    ask_close = (
        ask_raw.get("c", ask_raw.get("close"))
        if isinstance(ask_raw, Mapping)
        else raw.get("ask_close")
    )
    complete = raw.get("complete") is True
    granularity = str(raw.get("granularity") or "M1").upper()
    if granularity != "M1":
        raise ValueError("CANDLE_GRANULARITY_MUST_BE_M1")
    candle = M1BidAskCandle(
        pair=str(pair or ""),
        timestamp_utc=str(timestamp or ""),
        # OANDA v20 serializes candle prices as decimal strings.  Parse only
        # at this adapter boundary; the frozen internal candle remains floats.
        bid_close=_external_finite_positive(bid_close, "bid_close"),
        ask_close=_external_finite_positive(ask_close, "ask_close"),
        complete=complete,
        source_sha256=(
            str(raw.get("source_sha256"))
            if raw.get("source_sha256") is not None
            else None
        ),
    )
    material = deepcopy(raw)
    return candle, material


def _entry_bid_ask(
    *,
    current_price: float,
    spread_pips: float,
    pip_factor: float,
    entry_bid: float | None,
    entry_ask: float | None,
) -> tuple[float, float, str]:
    if (entry_bid is None) is not (entry_ask is None):
        raise ValueError("ENTRY_BID_ASK_MUST_BE_PAIRED")
    if entry_bid is None:
        half_spread = spread_pips / pip_factor / 2.0
        bid = current_price - half_spread
        ask = current_price + half_spread
        source = "REFERENCE_MID_PLUS_CONTEXT_SPREAD"
    else:
        bid = _finite_positive(entry_bid, "entry_bid")
        ask = _finite_positive(entry_ask, "entry_ask")
        source = "EXPLICIT_BID_ASK"
        if not math.isclose(
            (bid + ask) / 2.0,
            current_price,
            rel_tol=0.0,
            abs_tol=1.0 / pip_factor / 10.0 + 1e-12,
        ):
            raise ValueError("ENTRY_MID_CURRENT_PRICE_MISMATCH")
        actual_spread = (ask - bid) * pip_factor
        if not math.isclose(actual_spread, spread_pips, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("ENTRY_SPREAD_CONTEXT_MISMATCH")
    if not 0.0 < bid < ask:
        raise ValueError("ENTRY_BID_ASK_INVALID")
    return _round_number(bid, 12), _round_number(ask, 12), source


def _canonical_detector_scores(
    value: Mapping[str, Any],
    *,
    winner: str,
) -> dict[str, float]:
    scores = _snapshot_mapping(value, "detector_scores")
    if set(scores) != _DETECTOR_SCORE_KEYS:
        raise ValueError("DETECTOR_SCORES_SCHEMA_INVALID")
    normalized = {
        direction: _round_number(
            _finite_nonnegative(scores[direction], f"detector_scores.{direction}"),
            12,
        )
        for direction in sorted(_DETECTOR_SCORE_KEYS)
    }
    winner_score = normalized[winner]
    directional_competitors = [
        normalized[direction]
        for direction in ("UP", "DOWN", "RANGE")
        if direction != winner
    ]
    if winner_score <= 0.0 or winner_score < max(directional_competitors):
        raise ValueError("DETECTOR_DIRECTION_NOT_SCORE_WINNER")
    return {key: normalized[key] for key in ("UP", "DOWN", "RANGE", "EITHER")}


def _validated_arm(value: object, expected_id: str) -> dict[str, str]:
    arm = _snapshot_mapping(value, "arm")
    if set(arm) != _ARM_KEYS or arm.get("arm_id") != expected_id:
        raise ValueError("ARM_SCHEMA_INVALID")
    direction = _canonical_direction(arm.get("direction"), "arm direction")
    source = _bounded_text(arm.get("source"), "arm source", 96)
    expected_source = {
        "DETECTOR": "PRE_VETO_DIRECTIONAL_DETECTOR_WINNER",
        "REGIME_FAMILY": "CONTENT_ADDRESSED_REGIME_FAMILY_AGGREGATE",
    }.get(expected_id)
    if expected_source is None or source != expected_source:
        raise ValueError("ARM_SOURCE_INVALID")
    return {"arm_id": expected_id, "direction": direction, "source": source}


def _permission_issues(value: Mapping[str, Any], issues: list[str]) -> None:
    if value.get("read_only") is not True or value.get("shadow_only") is not True:
        issues.append("SHADOW_READ_ONLY_REQUIRED")
    if value.get("live_permission") is not False:
        issues.append("LIVE_PERMISSION_MUST_BE_FALSE")
    if value.get("sizing_permission") is not False:
        issues.append("SIZING_PERMISSION_MUST_BE_FALSE")
    if value.get("gate_relaxation_allowed") is not False:
        issues.append("GATE_RELAXATION_MUST_BE_FALSE")
    if value.get("automatic_promotion_allowed") is not False:
        issues.append("AUTOMATIC_PROMOTION_MUST_BE_FALSE")


def _canonical_pair(value: Any) -> str:
    if value.__class__ is not str:
        raise ValueError("PAIR_INVALID")
    pair = value.strip().upper()
    if (
        pair != value
        or _PAIR_RE.fullmatch(pair) is None
        or pair not in DEFAULT_TRADER_PAIRS
    ):
        raise ValueError("PAIR_INVALID")
    return pair


def _canonical_direction(value: Any, label: str) -> str:
    if value.__class__ is not str:
        raise ValueError(f"{label} invalid")
    direction = value.strip().upper()
    if direction != value or direction not in _DIRECTIONS:
        raise ValueError(f"{label} invalid")
    return direction


def _snapshot_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    try:
        snapshot = dict(value)
    except Exception as exc:
        raise ValueError(f"{label} snapshot unreadable") from exc
    if any(key.__class__ is not str for key in snapshot):
        raise ValueError(f"{label} keys invalid")
    return snapshot


def _finite_number(value: Any, label: str) -> float:
    if value.__class__ not in {int, float}:
        raise ValueError(f"{label} invalid")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} invalid")
    return parsed


def _finite_positive(value: Any, label: str) -> float:
    parsed = _finite_number(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _external_finite_positive(value: Any, label: str) -> float:
    if value.__class__ is str:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} invalid") from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{label} invalid")
        return parsed
    return _finite_positive(value, label)


def _finite_nonnegative(value: Any, label: str) -> float:
    parsed = _finite_number(value, label)
    if parsed < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return parsed


def _bounded_fraction(value: Any, label: str) -> float:
    parsed = _finite_number(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} outside 0..1")
    return parsed


def _bounded_text(value: Any, label: str, maximum: int) -> str:
    if (
        value.__class__ is not str
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ValueError(f"{label} invalid")
    return value


def _optional_bounded_text(value: Any, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, label, maximum)


def _bounded_id(value: Any, label: str, maximum: int) -> str:
    text = _bounded_text(value, label, maximum)
    if _ID_RE.fullmatch(text) is None:
        raise ValueError(f"{label} invalid")
    return text


def _canonical_sha256_text(value: Any, label: str) -> str:
    if value.__class__ is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} invalid")
    return value


def _round_number(value: float, digits: int) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def _parse_utc(value: Any) -> _UtcInstant | None:
    if value.__class__ is not str:
        return None
    match = _RFC3339_UTC_RE.fullmatch(value)
    if match is None:
        return None
    try:
        base = datetime.strptime(match.group("seconds"), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return _UtcInstant(int(base.timestamp()), int(fraction or "0"))


def _required_utc(value: Any, label: str) -> _UtcInstant:
    parsed = _parse_utc(value)
    if parsed is None:
        raise ValueError(f"{label} invalid")
    if value != parsed.canonical():
        raise ValueError(f"{label} noncanonical")
    return parsed


def _coerce_utc(value: str | datetime, label: str) -> _UtcInstant:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{label} must be timezone-aware")
        value = value.astimezone(timezone.utc)
        canonical = value.isoformat().replace("+00:00", "Z")
        parsed = _parse_utc(canonical)
    else:
        parsed = _parse_utc(value)
    if parsed is None:
        raise ValueError(f"{label} invalid")
    return parsed


def _minute_aligned(value: _UtcInstant) -> bool:
    return value.nanosecond == 0 and value.epoch_second % M1_SECONDS == 0


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "EVALUATION_HORIZON_MINUTES",
    "HOLDOUT_LOCK_CONTRACT",
    "LEDGER_FILENAME",
    "MAX_EMISSION_AGE_SECONDS",
    "MAX_EMISSION_FUTURE_SKEW_SECONDS",
    "M1BidAskCandle",
    "RESULT_CONTRACT",
    "SELECTION_CONTRACT",
    "SHADOW_CONTRACT",
    "TERMINAL_EVALUATION_CONTRACT",
    "TRIAL_CONTRACT",
    "bind_regime_family_contradiction_emission",
    "build_regime_family_contradiction_shadow",
    "load_regime_family_contradiction_ledger",
    "persist_regime_family_contradiction_emission",
    "persist_regime_family_contradiction_results",
    "resolve_due_regime_family_contradiction_ledger",
    "resolve_due_regime_family_contradiction_trials",
    "resolve_regime_family_contradiction_trial",
    "seal_regime_family_contradiction_holdout_lock",
    "select_independent_regime_family_contradiction_trials",
    "validate_regime_family_contradiction_holdout_lock",
    "validate_regime_family_contradiction_result",
    "validate_regime_family_contradiction_result_binding",
    "validate_regime_family_contradiction_shadow",
    "validate_regime_family_contradiction_trial",
    "verify_regime_family_contradiction_source_binding",
]
