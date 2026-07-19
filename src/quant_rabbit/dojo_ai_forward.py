"""Two-stage prospective lifecycle for the registered DOJO AI prompt test.

The V1 phase manifest required hashes for all 30 future market days at once and
therefore could not be a genuine pre-response preregistration.  This lifecycle
first freezes the exact schedule and 90 cell identities, then seals one causal
three-variant request bundle per scheduled cutoff.  The final 90-cell index is
derived from the chain; it is not treated as the original precommit.

This module never invokes a model, opens answer material, mutates a broker, or
upgrades evidence beyond a self-attested diagnostic.  V3 binds a separate
operator-side one-shot CLI executor and its locally verifiable receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from quant_rabbit.dojo_ai_discretion import (
    DIAGNOSTIC_TIER,
    build_day_packet,
    build_provider_capability_manifest,
    canonical_sha256,
    prelock_prompt,
    prelock_scorer,
    seal_provider_model_manifest,
    seal_response,
)
from quant_rabbit.dojo_ai_execution import validate_execution_receipt
from quant_rabbit.dojo_prompt_phase import (
    LOCKED_EXPERIMENT_ID,
    LOCKED_PREREGISTRATION_SHA256,
    LOCKED_VARIANT_PROMPT_SHA256,
    assert_locked_preregistration,
    build_cell_assignment,
)
from quant_rabbit.dojo_market_calendar import (
    OANDA_FX_HOURS_POLICY,
    OANDA_FX_HOURS_SOURCE,
    expected_oanda_fx_slots,
)


PRECOMMIT_CONTRACT = "QR_DOJO_AI_FORWARD_PRECOMMIT_V3"
START_CONTRACT = "QR_DOJO_AI_FORWARD_START_V3"
SUPERSESSION_CONTRACT = "QR_DOJO_AI_FORWARD_SUPERSESSION_V1"
DAY_SEAL_CONTRACT = "QR_DOJO_AI_DAY_SOURCE_SEAL_V3"
REQUEST_CONTRACT = "QR_DOJO_AI_FORWARD_REQUEST_V3"
SOURCE_PROVENANCE_CONTRACT = "QR_DOJO_AI_OANDA_SOURCE_PROVENANCE_V3"
SOURCE_CAPTURE_CONTRACT = "QR_DOJO_AI_OANDA_SOURCE_CAPTURE_V3"
SOURCE_REQUEST_CONTRACT = "QR_DOJO_AI_OANDA_SOURCE_REQUEST_V3"
CELL_RESPONSE_CONTRACT = "QR_DOJO_AI_FORWARD_CELL_RESPONSE_V3"
CELL_EXECUTION_FAILURE_CONTRACT = "QR_DOJO_AI_FORWARD_CELL_EXECUTION_FAILURE_V3"
CELL_FAILURE_CONTRACT = "QR_DOJO_AI_FORWARD_CELL_FAILURE_V3"
PHASE_INDEX_CONTRACT = "QR_DOJO_AI_FORWARD_PHASE_INDEX_V3"
PHASE_ID = "phase_1_diagnostic"
PAIR = "USD_JPY"
SOURCE_CONTRACT = "QR_DOJO_AI_DAY_SOURCE_V1"
SOURCE_KIND = "M5_BA_CANDLE"
SOURCE_GRANULARITY = "M5"
SOURCE_PRICE = "BA"
OFFICIAL_OANDA_BASE_URL = "https://api-fxtrade.oanda.com"
SOURCE_LOOKBACK = timedelta(hours=24)
SOURCE_NOT_BEFORE_DELAY = timedelta(minutes=2)
SOURCE_SEAL_DEADLINE_DELAY = timedelta(minutes=30)
RESPONSE_DEADLINE_DELAY = timedelta(hours=6)
TRUTH_HORIZON = timedelta(hours=24)
TRUTH_MATURITY_DELAY = timedelta(minutes=2)
TRUTH_SEAL_DEADLINE_DELAY = timedelta(hours=6)
TRUTH_SLIPPAGE_PIPS_PER_FILL = Decimal("0.3")
TRUTH_FINANCING_PIPS_PER_DAY = Decimal("0.8")
TRUTH_NOTIONAL_MULTIPLE = Decimal("1.0")
ELIGIBLE_WEEKDAYS = (0, 1, 2, 3)  # Monday through Thursday.
DECISION_TIME_UTC = time(15, 0)
EXPECTED_DAY_COUNT = 30
EXPECTED_CELL_COUNT = 90
M5_MINIMUM_COVERAGE = (98, 100)
M5_MAX_CONTIGUOUS_GAP = timedelta(minutes=15)
M5_BOUNDARY_TOLERANCE = timedelta(minutes=15)
M5_COVERAGE_POLICY = "OANDA_M5_TRUTHFUL_SPARSE_FIXED_COVERAGE_V1"
LOCKED_REGISTRY_AT = datetime(2026, 7, 18, 18, 2, 4, tzinfo=timezone.utc)
_LIMITATIONS = [
    "EXTERNAL_MONOTONIC_WITNESS_ABSENT",
    "PROVIDER_MODEL_IDENTITY_ATTESTATION_ABSENT",
    "PROVIDER_BASE_PROMPT_AND_FULL_REQUEST_ATTESTATION_ABSENT",
    "LOCAL_EXECUTION_RECEIPT_IS_NOT_PROVIDER_ATTESTATION",
    "MARKET_TRUTH_AND_SCORING_SELF_ATTESTED_WITHOUT_EXTERNAL_WITNESS",
    "UPSTREAM_HTTP_STATUS_AND_HEADER_ATTESTATION_ABSENT",
    "SOURCE_COMPLETENESS_SELF_ATTESTED_FIXED_COVERAGE_GATE",
    "DOJO_HAS_NO_LIVE_AUTHORITY",
]
_HEX64 = re.compile(r"[0-9a-f]{64}")
_GIT_OID = re.compile(r"[0-9a-f]{40}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]+")
_OANDA_TIME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(0+))?Z")
_VARIANTS = tuple(LOCKED_VARIANT_PROMPT_SHA256)
REQUIRED_SOURCE_BINDING_FILES = frozenset(
    {
        "src/quant_rabbit/dojo_ai_forward.py",
        "src/quant_rabbit/dojo_ai_execution.py",
        "src/quant_rabbit/dojo_ai_validity.py",
        "src/quant_rabbit/dojo_ai_discretion.py",
        "src/quant_rabbit/dojo_prompt_phase.py",
        "src/quant_rabbit/dojo_market_calendar.py",
        "src/quant_rabbit/dojo_ai_truth.py",
        "src/quant_rabbit/broker/oanda.py",
        "scripts/run-dojo-ai-forward.py",
        "tools/run-dojo-ai-model-cell.py",
    }
)


class DojoAIForwardError(ValueError):
    """The prospective schedule, day chain, source, or request is invalid."""


def build_precommit(
    registry: Mapping[str, Any],
    prompt_texts: Mapping[str, str],
    spec: Mapping[str, Any],
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    """Freeze 30 exact future cutoffs and all 90 identities before day one."""

    prereg = assert_locked_preregistration(registry)
    source = _mapping(spec, "AI forward precommit spec")
    _exact_keys(
        source,
        {
            "first_cutoff_utc",
            "allocation_nonce",
            "model_policy",
            "source_bindings",
        },
        "AI forward precommit spec",
    )
    now = _utc(now_utc, "now_utc")
    first = _parse_utc(source["first_cutoff_utc"], "first_cutoff_utc")
    if first.time() != DECISION_TIME_UTC or first.weekday() not in ELIGIBLE_WEEKDAYS:
        raise DojoAIForwardError("first cutoff violates fixed weekday/time policy")
    if now >= first:
        raise DojoAIForwardError("AI forward precommit must precede first cutoff")
    nonce = _sha(source["allocation_nonce"], "allocation_nonce")
    model_policy = _validate_model_policy(source["model_policy"])
    source_bindings = _validate_source_bindings(source["source_bindings"])
    prompt_locks = _build_prompt_locks(prereg, prompt_texts)
    scorer_lock = prelock_scorer(locked_at_utc=now)
    schedule = _build_schedule(first, nonce)
    body = {
        "contract": PRECOMMIT_CONTRACT,
        "schema_version": 2,
        "validity_status": "VALID",
        "state": "PRECOMMITTED",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "created_at_utc": _iso(now),
        "preregistration_sha256": LOCKED_PREREGISTRATION_SHA256,
        "allocation_nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "allocation_policy": {
            "unique_day_count": EXPECTED_DAY_COUNT,
            "variant_ids": list(_VARIANTS),
            "allocated_cell_count": EXPECTED_CELL_COUNT,
            "eligible_weekdays_utc": list(ELIGIBLE_WEEKDAYS),
            "decision_time_utc": "15:00:00",
            "missing_day_replacement_allowed": False,
            "rank_or_cutoff_reselection_allowed": False,
            "all_three_variants_sealed_atomically_per_day": True,
            "one_fresh_context_per_cell": True,
        },
        "market_policy": {
            "pair": PAIR,
            "source_granularity": SOURCE_GRANULARITY,
            "source_price_component": SOURCE_PRICE,
            "source_lookback_seconds": int(SOURCE_LOOKBACK.total_seconds()),
            "source_kind": SOURCE_KIND,
            "market_hours_policy": OANDA_FX_HOURS_POLICY,
            "market_hours_source": OANDA_FX_HOURS_SOURCE,
            "source_not_before_delay_seconds": int(
                SOURCE_NOT_BEFORE_DELAY.total_seconds()
            ),
            "source_seal_deadline_delay_seconds": int(
                SOURCE_SEAL_DEADLINE_DELAY.total_seconds()
            ),
            "response_deadline_delay_seconds": int(
                RESPONSE_DEADLINE_DELAY.total_seconds()
            ),
            "truth_horizon_seconds": int(TRUTH_HORIZON.total_seconds()),
            "truth_maturity_delay_seconds": int(TRUTH_MATURITY_DELAY.total_seconds()),
            "truth_seal_deadline_delay_seconds": int(
                TRUTH_SEAL_DEADLINE_DELAY.total_seconds()
            ),
            "truth_semantics": "FIXED_24H_DIRECTION_AND_SIZE_ONLY",
            "truth_slippage_pips_per_fill": float(TRUTH_SLIPPAGE_PIPS_PER_FILL),
            "truth_financing_pips_per_day": float(TRUTH_FINANCING_PIPS_PER_DAY),
            "truth_notional_multiple": float(TRUTH_NOTIONAL_MULTIPLE),
            "source_coverage_policy": M5_COVERAGE_POLICY,
            "minimum_coverage_numerator": M5_MINIMUM_COVERAGE[0],
            "minimum_coverage_denominator": M5_MINIMUM_COVERAGE[1],
            "max_contiguous_gap_seconds": int(M5_MAX_CONTIGUOUS_GAP.total_seconds()),
            "boundary_tolerance_seconds": int(M5_BOUNDARY_TOLERANCE.total_seconds()),
            "missing_source_slots_retained": True,
            "market_derived_answer_key_required": True,
            "answer_key_before_response_seal_allowed": False,
        },
        "model_policy": model_policy,
        "prompt_locks": prompt_locks,
        "scorer_lock": scorer_lock,
        "source_bindings": source_bindings,
        "schedule": schedule,
        "evidence_tier": DIAGNOSTIC_TIER,
        "limitations": list(_LIMITATIONS),
        "authority": _authority(),
    }
    return _seal(body, "precommit_sha256")


def validate_precommit(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(value, "AI forward precommit")
    expected_keys = {
        "contract",
        "schema_version",
        "validity_status",
        "state",
        "experiment_id",
        "phase_id",
        "created_at_utc",
        "preregistration_sha256",
        "allocation_nonce_sha256",
        "allocation_policy",
        "market_policy",
        "model_policy",
        "prompt_locks",
        "scorer_lock",
        "source_bindings",
        "schedule",
        "evidence_tier",
        "limitations",
        "authority",
        "precommit_sha256",
    }
    _exact_keys(artifact, expected_keys, "AI forward precommit")
    _validate_seal(artifact, "precommit_sha256")
    if (
        artifact["contract"] != PRECOMMIT_CONTRACT
        or artifact["schema_version"] != 2
        or artifact["validity_status"] != "VALID"
        or artifact["state"] != "PRECOMMITTED"
        or artifact["experiment_id"] != LOCKED_EXPERIMENT_ID
        or artifact["phase_id"] != PHASE_ID
        or artifact["preregistration_sha256"] != LOCKED_PREREGISTRATION_SHA256
        or artifact["evidence_tier"] != DIAGNOSTIC_TIER
        or artifact["limitations"] != _LIMITATIONS
        or artifact["authority"] != _authority()
    ):
        raise DojoAIForwardError("AI forward precommit identity or authority drifted")
    created = _parse_utc(artifact["created_at_utc"], "created_at_utc")
    allocation = _mapping(artifact["allocation_policy"], "allocation_policy")
    expected_allocation = {
        "unique_day_count": EXPECTED_DAY_COUNT,
        "variant_ids": list(_VARIANTS),
        "allocated_cell_count": EXPECTED_CELL_COUNT,
        "eligible_weekdays_utc": list(ELIGIBLE_WEEKDAYS),
        "decision_time_utc": "15:00:00",
        "missing_day_replacement_allowed": False,
        "rank_or_cutoff_reselection_allowed": False,
        "all_three_variants_sealed_atomically_per_day": True,
        "one_fresh_context_per_cell": True,
    }
    if allocation != expected_allocation:
        raise DojoAIForwardError("AI forward allocation policy drifted")
    if artifact["market_policy"] != {
        "pair": PAIR,
        "source_granularity": SOURCE_GRANULARITY,
        "source_price_component": SOURCE_PRICE,
        "source_lookback_seconds": int(SOURCE_LOOKBACK.total_seconds()),
        "source_kind": SOURCE_KIND,
        "market_hours_policy": OANDA_FX_HOURS_POLICY,
        "market_hours_source": OANDA_FX_HOURS_SOURCE,
        "source_not_before_delay_seconds": int(SOURCE_NOT_BEFORE_DELAY.total_seconds()),
        "source_seal_deadline_delay_seconds": int(
            SOURCE_SEAL_DEADLINE_DELAY.total_seconds()
        ),
        "response_deadline_delay_seconds": int(RESPONSE_DEADLINE_DELAY.total_seconds()),
        "truth_horizon_seconds": int(TRUTH_HORIZON.total_seconds()),
        "truth_maturity_delay_seconds": int(TRUTH_MATURITY_DELAY.total_seconds()),
        "truth_seal_deadline_delay_seconds": int(
            TRUTH_SEAL_DEADLINE_DELAY.total_seconds()
        ),
        "truth_semantics": "FIXED_24H_DIRECTION_AND_SIZE_ONLY",
        "truth_slippage_pips_per_fill": float(TRUTH_SLIPPAGE_PIPS_PER_FILL),
        "truth_financing_pips_per_day": float(TRUTH_FINANCING_PIPS_PER_DAY),
        "truth_notional_multiple": float(TRUTH_NOTIONAL_MULTIPLE),
        "source_coverage_policy": M5_COVERAGE_POLICY,
        "minimum_coverage_numerator": M5_MINIMUM_COVERAGE[0],
        "minimum_coverage_denominator": M5_MINIMUM_COVERAGE[1],
        "max_contiguous_gap_seconds": int(M5_MAX_CONTIGUOUS_GAP.total_seconds()),
        "boundary_tolerance_seconds": int(M5_BOUNDARY_TOLERANCE.total_seconds()),
        "missing_source_slots_retained": True,
        "market_derived_answer_key_required": True,
        "answer_key_before_response_seal_allowed": False,
    }:
        raise DojoAIForwardError("AI forward market policy drifted")
    _validate_model_policy(artifact["model_policy"])
    _validate_source_bindings(artifact["source_bindings"])
    _sha(artifact["allocation_nonce_sha256"], "allocation_nonce_sha256")
    prompt_locks = artifact["prompt_locks"]
    if not isinstance(prompt_locks, Mapping) or set(prompt_locks) != set(_VARIANTS):
        raise DojoAIForwardError("AI forward prompt lock set drifted")
    for variant, lock in prompt_locks.items():
        lock_value = _mapping(lock, "prompt lock")
        try:
            rebuilt_lock = prelock_prompt(
                lock_value["prompt_text"],
                variant_id=variant,
                locked_at_utc=LOCKED_REGISTRY_AT,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DojoAIForwardError("AI forward prompt lock is invalid") from exc
        if (
            lock_value != rebuilt_lock
            or lock_value.get("prompt_sha256") != LOCKED_VARIANT_PROMPT_SHA256[variant]
        ):
            raise DojoAIForwardError("AI forward prompt lock drifted")
    scorer = _mapping(artifact["scorer_lock"], "scorer_lock")
    if scorer != prelock_scorer(locked_at_utc=created):
        raise DojoAIForwardError("AI forward scorer lock drifted")
    schedule = _validate_schedule(artifact["schedule"])
    if created >= _parse_utc(schedule[0]["decision_cutoff_utc"], "first cutoff"):
        raise DojoAIForwardError("AI forward precommit did not precede first cutoff")
    return _snapshot(artifact)


def build_start_receipt(
    precommit: Mapping[str, Any], *, now_utc: datetime
) -> dict[str, Any]:
    lock = validate_precommit(precommit)
    now = _utc(now_utc, "now_utc")
    created = _parse_utc(lock["created_at_utc"], "created_at_utc")
    first = _parse_utc(lock["schedule"][0]["decision_cutoff_utc"], "first cutoff")
    if now < created or now >= first:
        raise DojoAIForwardError(
            "AI forward start must be after precommit and before cutoff"
        )
    body = {
        "contract": START_CONTRACT,
        "schema_version": 2,
        "state": "STARTED",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "started_at_utc": _iso(now),
        "precommit_sha256": lock["precommit_sha256"],
        "previous_receipt_sha256": lock["precommit_sha256"],
        "scheduled_day_count": EXPECTED_DAY_COUNT,
        "allocated_cell_count": EXPECTED_CELL_COUNT,
        "authority": _authority(),
    }
    return _seal(body, "start_receipt_sha256")


def validate_start_receipt(
    value: Mapping[str, Any], precommit: Mapping[str, Any]
) -> dict[str, Any]:
    lock = validate_precommit(precommit)
    receipt = _mapping(value, "AI forward start receipt")
    _exact_keys(
        receipt,
        {
            "contract",
            "schema_version",
            "state",
            "experiment_id",
            "phase_id",
            "started_at_utc",
            "precommit_sha256",
            "previous_receipt_sha256",
            "scheduled_day_count",
            "allocated_cell_count",
            "authority",
            "start_receipt_sha256",
        },
        "AI forward start receipt",
    )
    _validate_seal(receipt, "start_receipt_sha256")
    expected = build_start_receipt(
        lock,
        now_utc=_parse_utc(receipt["started_at_utc"], "started_at_utc"),
    )
    if receipt != expected:
        raise DojoAIForwardError("AI forward start receipt drifted")
    return _snapshot(receipt)


def validate_supersession(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable terminal receipt for a zero-source stopped run."""

    lock = validate_precommit(precommit)
    start = validate_start_receipt(start_receipt, lock)
    receipt = _mapping(value, "AI forward supersession")
    _exact_keys(
        receipt,
        {
            "allocated_cell_count",
            "authority",
            "contract",
            "evidence",
            "executed_cell_count",
            "market_source_acquisition_count",
            "precommit_sha256",
            "reason_code",
            "schema_version",
            "start_receipt_sha256",
            "state",
            "successor_policy",
            "superseded_at_utc",
            "supersession_sha256",
        },
        "AI forward supersession",
    )
    _validate_seal(receipt, "supersession_sha256")
    superseded_at = _parse_utc(receipt["superseded_at_utc"], "superseded_at_utc")
    started_at = _parse_utc(start["started_at_utc"], "started_at_utc")
    first_cutoff = _parse_utc(
        lock["schedule"][0]["decision_cutoff_utc"], "first cutoff"
    )
    counts = (
        receipt["allocated_cell_count"],
        receipt["executed_cell_count"],
        receipt["market_source_acquisition_count"],
    )
    if any(isinstance(count, bool) or not isinstance(count, int) for count in counts):
        raise DojoAIForwardError("AI forward supersession counts are invalid")
    authority = _mapping(receipt["authority"], "supersession authority")
    _exact_keys(
        authority,
        {
            "broker_mutation_allowed",
            "live_permission",
            "order_authority",
            "promotion_eligible",
        },
        "supersession authority",
    )
    successor = _mapping(receipt["successor_policy"], "successor_policy")
    _exact_keys(
        successor,
        {
            "entry_full_context_requires_causal_multisource_contract",
            "next_experiment_class",
            "one_judgment_per_fresh_context",
        },
        "successor_policy",
    )
    if (
        receipt["contract"] != SUPERSESSION_CONTRACT
        or isinstance(receipt["schema_version"], bool)
        or receipt["schema_version"] != 1
        or receipt["state"] != "SUPERSEDED_BEFORE_SOURCE"
        or receipt["precommit_sha256"] != lock["precommit_sha256"]
        or receipt["start_receipt_sha256"] != start["start_receipt_sha256"]
        or counts != (EXPECTED_CELL_COUNT, 0, 0)
        or authority["broker_mutation_allowed"] is not False
        or authority["live_permission"] is not False
        or authority["order_authority"] != "NONE"
        or authority["promotion_eligible"] is not False
        or successor["entry_full_context_requires_causal_multisource_contract"]
        is not True
        or successor["next_experiment_class"] != "AI_EXIT_JUDGMENT"
        or successor["one_judgment_per_fresh_context"] is not True
        or not (started_at <= superseded_at < first_cutoff)
    ):
        raise DojoAIForwardError("AI forward supersession identity drifted")
    _identifier(receipt["reason_code"], "reason_code")
    evidence = receipt["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 16:
        raise DojoAIForwardError("AI forward supersession evidence is invalid")
    seen_evidence: set[str] = set()
    for index, raw in enumerate(evidence):
        row = _mapping(raw, f"supersession evidence {index}")
        _exact_keys(
            row,
            {"artifact_sha256", "path", "result"},
            f"supersession evidence {index}",
        )
        digest = _sha(row["artifact_sha256"], "artifact_sha256")
        path = row["path"]
        if (
            digest in seen_evidence
            or not isinstance(path, str)
            or not 1 <= len(path) <= 4096
            or "\x00" in path
        ):
            raise DojoAIForwardError("AI forward supersession evidence drifted")
        _identifier(row["result"], "evidence result")
        seen_evidence.add(digest)
    return _snapshot(receipt)


def build_day_requests(
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    oanda_response: Mapping[str, Any],
    *,
    ordinal: int,
    now_utc: datetime,
) -> dict[str, Any]:
    """Derive one causal OANDA source and seal all three model requests."""

    capture = build_source_capture(
        precommit,
        start_receipt,
        previous_day_seal,
        oanda_response,
        ordinal=ordinal,
        now_utc=now_utc,
    )
    return build_day_requests_from_capture(
        registry,
        precommit,
        start_receipt,
        previous_day_seal,
        capture,
    )


def build_source_capture(
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    oanda_response: Any,
    *,
    ordinal: int,
    now_utc: datetime,
) -> dict[str, Any]:
    """Validate and first-seal one OANDA response before derived requests."""

    now = _utc(now_utc, "now_utc")
    request = prepare_day_request(
        precommit,
        start_receipt,
        previous_day_seal,
        ordinal=ordinal,
        now_utc=now,
    )
    return build_source_capture_from_request(
        precommit,
        start_receipt,
        previous_day_seal,
        request,
        oanda_response,
        acquired_at_utc=now,
    )


def build_source_capture_from_request(
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    source_request: Mapping[str, Any],
    oanda_response: Any,
    *,
    acquired_at_utc: datetime,
) -> dict[str, Any]:
    """First-write any finite response and preserve request/acquisition ordering."""

    lock = validate_precommit(precommit)
    start = validate_start_receipt(start_receipt, lock)
    request = validate_source_request(source_request, lock, start, previous_day_seal)
    ordinal = request["ordinal"]
    acquired = _utc(acquired_at_utc, "acquired_at_utc")
    requested = _parse_utc(request["requested_at_utc"], "requested_at_utc")
    schedule, previous_sha = _day_context(
        lock, start, previous_day_seal, ordinal=ordinal, at_utc=acquired
    )
    deadline = _parse_utc(schedule["source_seal_deadline_utc"], "source deadline")
    if acquired < requested or acquired > deadline:
        raise DojoAIForwardError("AI source request/acquisition ordering is invalid")
    response = _snapshot(oanda_response)
    body = {
        "contract": SOURCE_CAPTURE_CONTRACT,
        "schema_version": 2,
        "state": "SOURCE_CAPTURED",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "ordinal": ordinal,
        "requested_at_utc": request["requested_at_utc"],
        "acquired_at_utc": _iso(acquired),
        "precommit_sha256": lock["precommit_sha256"],
        "start_receipt_sha256": start["start_receipt_sha256"],
        "previous_receipt_sha256": previous_sha,
        "request": request,
        "source_request_sha256": request["source_request_sha256"],
        "response": response,
        "response_sha256": canonical_sha256(response),
        "credentials_persisted": False,
        "late_response_reselection_allowed": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "source_capture_sha256")


def validate_source_capture(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifact = _mapping(value, "AI source capture")
    _validate_seal(artifact, "source_capture_sha256")
    expected = build_source_capture_from_request(
        precommit,
        start_receipt,
        previous_day_seal,
        artifact.get("request"),
        artifact.get("response"),
        acquired_at_utc=_parse_utc(artifact.get("acquired_at_utc"), "acquired_at_utc"),
    )
    if artifact != expected:
        raise DojoAIForwardError("AI source capture bytes or parents drifted")
    return _snapshot(artifact)


def build_day_requests_from_capture(
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    source_capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the three request cells from one already immutable capture."""

    assert_locked_preregistration(registry)
    lock = validate_precommit(precommit)
    start = validate_start_receipt(start_receipt, lock)
    capture = validate_source_capture(source_capture, lock, start, previous_day_seal)
    ordinal = capture["ordinal"]
    now = _parse_utc(capture["acquired_at_utc"], "acquired_at_utc")
    schedule, previous_sha = _day_context(
        lock, start, previous_day_seal, ordinal=ordinal, at_utc=now
    )
    source_value, source_provenance = _source_from_oanda_response(
        capture["response"],
        schedule=schedule,
        acquired_at_utc=now,
        source_capture=capture,
    )
    return _build_day_requests_from_source(
        lock,
        start,
        schedule,
        previous_sha,
        source_value,
        source_provenance,
        ordinal=ordinal,
        now=now,
        source_capture_sha256=capture["source_capture_sha256"],
    )


def prepare_day_request(
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    *,
    ordinal: int,
    now_utc: datetime,
) -> dict[str, Any]:
    """Validate chronology before any credential or network access is needed."""

    lock = validate_precommit(precommit)
    start = validate_start_receipt(start_receipt, lock)
    now = _utc(now_utc, "now_utc")
    schedule, previous_sha = _day_context(
        lock, start, previous_day_seal, ordinal=ordinal, at_utc=now
    )
    not_before = _parse_utc(schedule["source_not_before_utc"], "source_not_before")
    deadline = _parse_utc(schedule["source_seal_deadline_utc"], "source deadline")
    if now < not_before or now > deadline:
        raise DojoAIForwardError("AI day source is outside its causal seal window")
    body = {
        "contract": SOURCE_REQUEST_CONTRACT,
        "schema_version": 2,
        "state": "SOURCE_REQUESTED",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "ordinal": ordinal,
        "requested_at_utc": _iso(now),
        "precommit_sha256": lock["precommit_sha256"],
        "start_receipt_sha256": start["start_receipt_sha256"],
        "previous_receipt_sha256": previous_sha,
        "path": f"/v3/instruments/{PAIR}/candles",
        "base_url": OFFICIAL_OANDA_BASE_URL,
        "query": {
            "from": schedule["source_window_start_utc"],
            "granularity": SOURCE_GRANULARITY,
            "includeFirst": "true",
            "price": SOURCE_PRICE,
            "to": schedule["source_window_end_utc"],
        },
        "source_seal_deadline_utc": schedule["source_seal_deadline_utc"],
        "read_only": True,
        "authority": _authority(),
    }
    return _seal(body, "source_request_sha256")


def validate_source_request(
    value: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
) -> dict[str, Any]:
    request = _mapping(value, "AI source request")
    _validate_seal(request, "source_request_sha256")
    expected = prepare_day_request(
        precommit,
        start_receipt,
        previous_day_seal,
        ordinal=request.get("ordinal"),
        now_utc=_parse_utc(request.get("requested_at_utc"), "requested_at_utc"),
    )
    if request != expected:
        raise DojoAIForwardError("AI source request bytes or parents drifted")
    return _snapshot(request)


def _build_day_requests_from_source(
    lock: Mapping[str, Any],
    start: Mapping[str, Any],
    schedule: Mapping[str, Any],
    previous_sha: str,
    source_value: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    *,
    ordinal: int,
    now: datetime,
    source_capture_sha256: str,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    source_sha: str | None = None
    for variant in _VARIANTS:
        scheduled_cell = next(
            item for item in schedule["cells"] if item["variant_id"] == variant
        )
        context_id = scheduled_cell["context_id"]
        capability = build_provider_capability_manifest(
            context_id=context_id,
            generated_at_utc=now,
        )
        prompt = lock["prompt_locks"][variant]
        model_policy = lock["model_policy"]
        model = seal_provider_model_manifest(
            model_name=model_policy["model_name"],
            model_version=model_policy["model_version"],
            model_lineage=model_policy["model_lineage"],
            reasoning_effort=model_policy["reasoning_effort"],
            context_id=context_id,
            capability_manifest=capability,
            locked_at_utc=now,
        )
        packet = build_day_packet(
            source_value,
            prompt_lock=prompt,
            capability_manifest=capability,
            scorer_lock=lock["scorer_lock"],
        )
        if source_sha is None:
            source_sha = packet["source_sha256"]
        elif packet["source_sha256"] != source_sha:
            raise DojoAIForwardError("variant packets do not share one source")
        request_body = {
            "contract": REQUEST_CONTRACT,
            "schema_version": 2,
            "experiment_id": LOCKED_EXPERIMENT_ID,
            "phase_id": PHASE_ID,
            "ordinal": ordinal,
            "blind_day_rank": schedule["blind_day_rank"],
            "blind_day_id": schedule["blind_day_id"],
            "variant_id": variant,
            "cell_id": scheduled_cell["cell_id"],
            "context_id": context_id,
            "sealed_at_utc": _iso(now),
            "response_deadline_utc": schedule["response_deadline_utc"],
            "truth_not_before_utc": schedule["truth_not_before_utc"],
            "source_sha256": packet["source_sha256"],
            "packet_sha256": packet["packet_sha256"],
            "prompt_sha256": prompt["prompt_sha256"],
            "prompt_lock_sha256": prompt["prompt_lock_sha256"],
            "model_sha256": model["model_sha256"],
            "capability_manifest_sha256": capability["capability_manifest_sha256"],
            "scorer_lock_sha256": lock["scorer_lock"]["scorer_lock_sha256"],
            "model_api_invoked": False,
            "answer_key_present": False,
            "authority": _authority(),
        }
        request = _seal(request_body, "request_receipt_sha256")
        assignment = build_cell_assignment(
            phase_id=PHASE_ID,
            blind_day_rank=schedule["blind_day_rank"],
            blind_day_id=schedule["blind_day_id"],
            variant_id=variant,
            source_sha256=packet["source_sha256"],
            packet_sha256=packet["packet_sha256"],
            prompt_sha256=prompt["prompt_sha256"],
            prompt_lock_sha256=prompt["prompt_lock_sha256"],
            model_sha256=model["model_sha256"],
            capability_manifest_sha256=capability["capability_manifest_sha256"],
            request_receipt_sha256=request["request_receipt_sha256"],
            context_id=context_id,
        )
        if assignment["cell_id"] != scheduled_cell["cell_id"]:
            raise DojoAIForwardError("generated assignment differs from precommit cell")
        cells.append(
            {
                "variant_id": variant,
                "capability_manifest": capability,
                "model_manifest": model,
                "packet": packet,
                "request_receipt": request,
                "assignment": assignment,
            }
        )
    assert source_sha is not None
    body = {
        "contract": DAY_SEAL_CONTRACT,
        "schema_version": 2,
        "state": "REQUESTS_SEALED",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "ordinal": ordinal,
        "sealed_at_utc": _iso(now),
        "precommit_sha256": lock["precommit_sha256"],
        "start_receipt_sha256": start["start_receipt_sha256"],
        "previous_receipt_sha256": previous_sha,
        "source_capture_sha256": source_capture_sha256,
        "schedule": schedule,
        "source": source_value,
        "source_provenance": source_provenance,
        "source_sha256": source_sha,
        "cells": cells,
        "terminal_failures": [],
        "response_selection_allowed": False,
        "late_source_backfill_allowed": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "day_seal_sha256")


def build_missing_day_seal(
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    *,
    ordinal: int,
    now_utc: datetime,
    failed_source_capture_sha256: str | None = None,
) -> dict[str, Any]:
    """Preserve a missed scheduled day as three permanent denominator failures."""

    lock = validate_precommit(precommit)
    start = validate_start_receipt(start_receipt, lock)
    now = _utc(now_utc, "now_utc")
    schedule, previous_sha = _day_context(
        lock, start, previous_day_seal, ordinal=ordinal, at_utc=now
    )
    deadline = _parse_utc(schedule["source_seal_deadline_utc"], "source deadline")
    if now <= deadline:
        raise DojoAIForwardError("missing day cannot seal before source deadline")
    failed_capture_sha = (
        _sha(failed_source_capture_sha256, "failed_source_capture_sha256")
        if failed_source_capture_sha256 is not None
        else None
    )
    failures = [
        {
            "cell_id": cell["cell_id"],
            "variant_id": cell["variant_id"],
            "reason": "MISSING_SOURCE_DEADLINE",
            "economic_fallback": "SYNTHETIC_FLAT_ZERO_RETURN",
        }
        for cell in schedule["cells"]
    ]
    body = {
        "contract": DAY_SEAL_CONTRACT,
        "schema_version": 2,
        "state": "MISSING_SOURCE_DEADLINE",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "ordinal": ordinal,
        "sealed_at_utc": _iso(now),
        "precommit_sha256": lock["precommit_sha256"],
        "start_receipt_sha256": start["start_receipt_sha256"],
        "previous_receipt_sha256": previous_sha,
        "source_capture_sha256": failed_capture_sha,
        "schedule": schedule,
        "source": None,
        "source_provenance": None,
        "source_sha256": None,
        "cells": [],
        "terminal_failures": failures,
        "response_selection_allowed": False,
        "late_source_backfill_allowed": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "day_seal_sha256")


def validate_day_seal(
    value: Mapping[str, Any],
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    *,
    expected_ordinal: int,
) -> dict[str, Any]:
    artifact = _mapping(value, "AI day seal")
    _validate_seal(artifact, "day_seal_sha256")
    if artifact.get("ordinal") != expected_ordinal:
        raise DojoAIForwardError("AI day seal ordinal drifted")
    sealed = _parse_utc(artifact.get("sealed_at_utc"), "sealed_at_utc")
    if artifact.get("state") == "REQUESTS_SEALED":
        provenance = _mapping(artifact.get("source_provenance"), "AI source provenance")
        expected = build_day_requests_from_capture(
            registry,
            precommit,
            start_receipt,
            previous_day_seal,
            provenance.get("source_capture"),
        )
    elif artifact.get("state") == "MISSING_SOURCE_DEADLINE":
        expected = build_missing_day_seal(
            precommit,
            start_receipt,
            previous_day_seal,
            ordinal=expected_ordinal,
            now_utc=sealed,
            failed_source_capture_sha256=artifact.get("source_capture_sha256"),
        )
    else:
        raise DojoAIForwardError("AI day seal state is unsupported")
    if artifact != expected:
        raise DojoAIForwardError("AI day seal bytes or derived artifacts drifted")
    return _snapshot(artifact)


def build_cell_response_seal(
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    day_seal: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    cell_id: str,
    now_utc: datetime,
) -> dict[str, Any]:
    """Import a fixed-schema response as diagnostic-only, never as model execution."""

    day = validate_day_seal(
        day_seal,
        registry,
        precommit,
        start_receipt,
        previous_day_seal,
        expected_ordinal=int(day_seal.get("ordinal", 0)),
    )
    if day["state"] != "REQUESTS_SEALED":
        raise DojoAIForwardError("missing-source day cannot accept a response")
    now = _utc(now_utc, "now_utc")
    day_sealed = _parse_utc(day["sealed_at_utc"], "day sealed_at_utc")
    deadline = _parse_utc(day["schedule"]["response_deadline_utc"], "deadline")
    if now < day_sealed or now > deadline:
        raise DojoAIForwardError("AI response is outside its append-only deadline")
    cell = _find_day_cell(day, cell_id)
    try:
        response_receipt = seal_response(
            response,
            packet=cell["packet"],
            prompt_lock=precommit["prompt_locks"][cell["variant_id"]],
            model_manifest=cell["model_manifest"],
            capability_manifest=cell["capability_manifest"],
            sealed_at_utc=now,
        )
    except (TypeError, ValueError) as exc:
        raise DojoAIForwardError(
            "AI response schema or parent binding is invalid"
        ) from exc
    body = {
        "contract": CELL_RESPONSE_CONTRACT,
        "schema_version": 2,
        "state": "DIAGNOSTIC_IMPORTED_RESPONSE",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "ordinal": day["ordinal"],
        "blind_day_rank": day["schedule"]["blind_day_rank"],
        "blind_day_id": day["schedule"]["blind_day_id"],
        "variant_id": cell["variant_id"],
        "cell_id": cell_id,
        "sealed_at_utc": _iso(now),
        "response_deadline_utc": day["schedule"]["response_deadline_utc"],
        "truth_not_before_utc": day["schedule"]["truth_not_before_utc"],
        "precommit_sha256": precommit["precommit_sha256"],
        "day_seal_sha256": day["day_seal_sha256"],
        "request_receipt_sha256": cell["request_receipt"]["request_receipt_sha256"],
        "response_receipt": response_receipt,
        "answer_key_opened": False,
        "response_selection_allowed": False,
        "provider_execution_attestation_present": False,
        "local_execution_procedure_verified": False,
        "execution_receipt": None,
        "prompt_evaluation_eligible": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "cell_terminal_sha256")


def build_executed_cell_terminal(
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    day_seal: Mapping[str, Any],
    execution_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the first one-shot execution outcome as an immutable cell terminal."""

    day = validate_day_seal(
        day_seal,
        registry,
        precommit,
        start_receipt,
        previous_day_seal,
        expected_ordinal=int(day_seal.get("ordinal", 0)),
    )
    if day["state"] != "REQUESTS_SEALED":
        raise DojoAIForwardError("missing-source day cannot execute a model")
    receipt = validate_execution_receipt(execution_receipt)
    request = receipt["execution_request"]
    cell = _find_day_cell(day, str(request.get("cell_id")))
    if (
        request.get("precommit_sha256") != precommit.get("precommit_sha256")
        or request.get("day_seal_sha256") != day["day_seal_sha256"]
        or request.get("variant_id") != cell["variant_id"]
        or request.get("context_id") != cell["request_receipt"]["context_id"]
        or request.get("request_receipt_sha256")
        != cell["request_receipt"]["request_receipt_sha256"]
        or request.get("packet_sha256") != cell["packet"]["packet_sha256"]
        or request.get("prompt_lock_sha256")
        != precommit["prompt_locks"][cell["variant_id"]]["prompt_lock_sha256"]
        or request.get("model_sha256") != cell["model_manifest"]["model_sha256"]
        or request.get("capability_manifest_sha256")
        != cell["capability_manifest"]["capability_manifest_sha256"]
        or request.get("answer_key_present") is not False
    ):
        raise DojoAIForwardError("execution receipt parent binding drifted")
    completed = _parse_utc(receipt["completed_at_utc"], "execution completed")
    day_sealed = _parse_utc(day["sealed_at_utc"], "day sealed_at_utc")
    deadline = _parse_utc(day["schedule"]["response_deadline_utc"], "deadline")
    if completed < day_sealed or completed > deadline:
        raise DojoAIForwardError("execution receipt is outside its response window")
    common = {
        "schema_version": 3,
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "ordinal": day["ordinal"],
        "blind_day_rank": day["schedule"]["blind_day_rank"],
        "blind_day_id": day["schedule"]["blind_day_id"],
        "variant_id": cell["variant_id"],
        "cell_id": request["cell_id"],
        "sealed_at_utc": _iso(completed),
        "response_deadline_utc": day["schedule"]["response_deadline_utc"],
        "truth_not_before_utc": day["schedule"]["truth_not_before_utc"],
        "precommit_sha256": precommit["precommit_sha256"],
        "day_seal_sha256": day["day_seal_sha256"],
        "request_receipt_sha256": cell["request_receipt"]["request_receipt_sha256"],
        "execution_request_sha256": request["execution_request_sha256"],
        "execution_receipt": receipt,
        "execution_receipt_sha256": receipt["execution_receipt_sha256"],
        "answer_key_opened": False,
        "response_selection_allowed": False,
        "provider_execution_attestation_present": False,
        "provider_identity_status": "REQUESTED_MODEL_CLI_REPORTED_UNVERIFIED",
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    if receipt["state"] == "EXECUTION_SUCCEEDED":
        try:
            response_receipt = seal_response(
                receipt["response"],
                packet=cell["packet"],
                prompt_lock=precommit["prompt_locks"][cell["variant_id"]],
                model_manifest=cell["model_manifest"],
                capability_manifest=cell["capability_manifest"],
                sealed_at_utc=completed,
            )
        except (TypeError, ValueError) as exc:
            raise DojoAIForwardError(
                "successful execution response violates the locked schema"
            ) from exc
        body = {
            "contract": CELL_RESPONSE_CONTRACT,
            "state": "EXECUTED_RESPONSE_SEALED",
            **common,
            "response_receipt": response_receipt,
            "economic_fallback": None,
            "response_failure": False,
            "local_execution_procedure_verified": True,
            "prompt_evaluation_eligible": True,
        }
    else:
        body = {
            "contract": CELL_EXECUTION_FAILURE_CONTRACT,
            "state": "MODEL_EXECUTION_FAILED",
            **common,
            "response_receipt": None,
            "economic_fallback": "SYNTHETIC_FLAT_ZERO_RETURN",
            "response_failure": True,
            "failure_code": receipt["failure_code"],
            "late_response_backfill_allowed": False,
            "local_execution_procedure_verified": False,
            "prompt_evaluation_eligible": False,
        }
    return _seal(body, "cell_terminal_sha256")


def build_cell_response_failure(
    precommit: Mapping[str, Any],
    day_seal: Mapping[str, Any],
    *,
    cell_id: str,
    now_utc: datetime,
) -> dict[str, Any]:
    """Permanently retain one response deadline miss as economic FLAT."""

    lock = validate_precommit(precommit)
    day = _mapping(day_seal, "AI day seal")
    _validate_seal(day, "day_seal_sha256")
    ordinal = day.get("ordinal")
    if (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or ordinal < 1
        or ordinal > EXPECTED_DAY_COUNT
    ):
        raise DojoAIForwardError("response failure day ordinal is invalid")
    if (
        day.get("state") != "REQUESTS_SEALED"
        or day.get("precommit_sha256") != lock["precommit_sha256"]
        or day.get("schedule") != lock["schedule"][ordinal - 1]
    ):
        raise DojoAIForwardError("response failure day parent is invalid")
    now = _utc(now_utc, "now_utc")
    deadline = _parse_utc(day["schedule"]["response_deadline_utc"], "deadline")
    if now <= deadline:
        raise DojoAIForwardError("response failure cannot seal before deadline")
    cell = _find_day_cell(day, cell_id)
    body = {
        "contract": CELL_FAILURE_CONTRACT,
        "schema_version": 2,
        "state": "MISSING_RESPONSE_DEADLINE",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "ordinal": day["ordinal"],
        "blind_day_rank": day["schedule"]["blind_day_rank"],
        "blind_day_id": day["schedule"]["blind_day_id"],
        "variant_id": cell["variant_id"],
        "cell_id": cell_id,
        "sealed_at_utc": _iso(now),
        "response_deadline_utc": day["schedule"]["response_deadline_utc"],
        "precommit_sha256": lock["precommit_sha256"],
        "day_seal_sha256": day["day_seal_sha256"],
        "request_receipt_sha256": cell["request_receipt"]["request_receipt_sha256"],
        "response_receipt": None,
        "economic_fallback": "SYNTHETIC_FLAT_ZERO_RETURN",
        "response_failure": True,
        "late_response_backfill_allowed": False,
        "answer_key_opened": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "cell_terminal_sha256")


def validate_cell_terminal(
    value: Mapping[str, Any],
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    day_seal: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _mapping(value, "AI cell terminal")
    _validate_seal(artifact, "cell_terminal_sha256")
    sealed = _parse_utc(artifact.get("sealed_at_utc"), "sealed_at_utc")
    if artifact.get("state") == "DIAGNOSTIC_IMPORTED_RESPONSE":
        receipt = _mapping(artifact.get("response_receipt"), "response receipt")
        expected = build_cell_response_seal(
            registry,
            precommit,
            start_receipt,
            previous_day_seal,
            day_seal,
            receipt.get("response"),
            cell_id=str(artifact.get("cell_id")),
            now_utc=sealed,
        )
    elif artifact.get("state") in {
        "EXECUTED_RESPONSE_SEALED",
        "MODEL_EXECUTION_FAILED",
    }:
        expected = build_executed_cell_terminal(
            registry,
            precommit,
            start_receipt,
            previous_day_seal,
            day_seal,
            _mapping(artifact.get("execution_receipt"), "execution receipt"),
        )
    elif artifact.get("state") == "MISSING_RESPONSE_DEADLINE":
        expected = build_cell_response_failure(
            precommit,
            day_seal,
            cell_id=str(artifact.get("cell_id")),
            now_utc=sealed,
        )
    else:
        raise DojoAIForwardError("AI cell terminal state is unsupported")
    if artifact != expected:
        raise DojoAIForwardError("AI cell terminal bytes or parent binding drifted")
    return _snapshot(artifact)


def build_phase_index(
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    day_seals: Sequence[Mapping[str, Any]],
    cell_terminals: Sequence[Mapping[str, Any]],
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    """Derive the fixed 90-cell denominator after every truth horizon matures."""

    assert_locked_preregistration(registry)
    lock = validate_precommit(precommit)
    start = validate_start_receipt(start_receipt, lock)
    if (
        isinstance(day_seals, (str, bytes))
        or not isinstance(day_seals, Sequence)
        or len(day_seals) != EXPECTED_DAY_COUNT
    ):
        raise DojoAIForwardError("phase index requires exactly 30 day seals")
    if isinstance(cell_terminals, (str, bytes)) or not isinstance(
        cell_terminals, Sequence
    ):
        raise DojoAIForwardError("phase cell terminals must be a sequence")
    terminal_by_cell: dict[str, dict[str, Any]] = {}
    for raw in cell_terminals:
        terminal = _mapping(raw, "phase cell terminal")
        cell_id = _identifier(terminal.get("cell_id"), "cell_id")
        if cell_id in terminal_by_cell:
            raise DojoAIForwardError("phase index contains a duplicate cell terminal")
        terminal_by_cell[cell_id] = terminal

    now = _utc(now_utc, "now_utc")
    last_truth = max(
        _parse_utc(row["truth_not_before_utc"], "truth_not_before_utc")
        for row in lock["schedule"]
    )
    if now < last_truth:
        raise DojoAIForwardError("phase index cannot seal before all truth horizons")

    previous: dict[str, Any] | None = None
    valid_days: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for ordinal, raw_day in enumerate(day_seals, start=1):
        day = validate_day_seal(
            raw_day,
            registry,
            lock,
            start,
            previous,
            expected_ordinal=ordinal,
        )
        if _parse_utc(day["sealed_at_utc"], "day sealed_at_utc") > now:
            raise DojoAIForwardError("phase index predates a day seal")
        valid_days.append(day)
        if day["state"] == "MISSING_SOURCE_DEADLINE":
            for scheduled in day["schedule"]["cells"]:
                cell_id = scheduled["cell_id"]
                if cell_id in terminal_by_cell:
                    raise DojoAIForwardError(
                        "missing-source cell cannot also carry a response terminal"
                    )
                index.append(
                    {
                        "ordinal": ordinal,
                        "blind_day_rank": day["schedule"]["blind_day_rank"],
                        "blind_day_id": day["schedule"]["blind_day_id"],
                        "variant_id": scheduled["variant_id"],
                        "cell_id": cell_id,
                        "terminal_state": "MISSING_SOURCE_DEADLINE",
                        "terminal_artifact_sha256": day["day_seal_sha256"],
                        "economic_fallback": "SYNTHETIC_FLAT_ZERO_RETURN",
                    }
                )
        else:
            for scheduled in day["schedule"]["cells"]:
                cell_id = scheduled["cell_id"]
                raw_terminal = terminal_by_cell.get(cell_id)
                if raw_terminal is None:
                    raise DojoAIForwardError(
                        "phase index has an unsealed response cell"
                    )
                terminal = validate_cell_terminal(
                    raw_terminal,
                    registry,
                    lock,
                    start,
                    previous,
                    day,
                )
                if (
                    _parse_utc(terminal["sealed_at_utc"], "terminal sealed_at_utc")
                    > now
                ):
                    raise DojoAIForwardError("phase index predates a cell terminal")
                if terminal["state"] == "DIAGNOSTIC_IMPORTED_RESPONSE":
                    raise DojoAIForwardError(
                        "imported response cannot enter the V3 prompt phase"
                    )
                consumed.add(cell_id)
                index.append(
                    {
                        "ordinal": ordinal,
                        "blind_day_rank": day["schedule"]["blind_day_rank"],
                        "blind_day_id": day["schedule"]["blind_day_id"],
                        "variant_id": scheduled["variant_id"],
                        "cell_id": cell_id,
                        "terminal_state": terminal["state"],
                        "terminal_artifact_sha256": terminal["cell_terminal_sha256"],
                        "economic_fallback": (
                            terminal.get("economic_fallback")
                            if terminal["state"]
                            in {"MISSING_RESPONSE_DEADLINE", "MODEL_EXECUTION_FAILED"}
                            else None
                        ),
                    }
                )
        previous = day
    if set(terminal_by_cell) != consumed:
        raise DojoAIForwardError("phase index contains an unexpected cell terminal")
    if (
        len(index) != EXPECTED_CELL_COUNT
        or len({row["cell_id"] for row in index}) != 90
    ):
        raise DojoAIForwardError("phase index denominator is not exactly 90 cells")

    response_count = sum(
        row["terminal_state"] == "EXECUTED_RESPONSE_SEALED" for row in index
    )
    missing_response_count = sum(
        row["terminal_state"] == "MISSING_RESPONSE_DEADLINE" for row in index
    )
    execution_failure_count = sum(
        row["terminal_state"] == "MODEL_EXECUTION_FAILED" for row in index
    )
    missing_source_cell_count = sum(
        row["terminal_state"] == "MISSING_SOURCE_DEADLINE" for row in index
    )
    body = {
        "contract": PHASE_INDEX_CONTRACT,
        "schema_version": 2,
        "state": "RESPONSES_FIXED_AWAITING_MARKET_TRUTH",
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": PHASE_ID,
        "sealed_at_utc": _iso(now),
        "last_truth_not_before_utc": _iso(last_truth),
        "precommit_sha256": lock["precommit_sha256"],
        "start_receipt_sha256": start["start_receipt_sha256"],
        "day_seal_sha256s": [day["day_seal_sha256"] for day in valid_days],
        "scheduled_day_count": EXPECTED_DAY_COUNT,
        "allocated_cell_count": EXPECTED_CELL_COUNT,
        "response_sealed_count": response_count,
        "missing_response_cell_count": missing_response_count,
        "execution_failure_cell_count": execution_failure_count,
        "missing_source_cell_count": missing_source_cell_count,
        "cell_index": index,
        "answer_keys_opened": False,
        "truth_scoring_present": False,
        "promotion_eligible": False,
        "evidence_tier": DIAGNOSTIC_TIER,
        "authority": _authority(),
    }
    return _seal(body, "phase_index_sha256")


def validate_phase_index(
    value: Mapping[str, Any],
    registry: Mapping[str, Any],
    precommit: Mapping[str, Any],
    start_receipt: Mapping[str, Any],
    day_seals: Sequence[Mapping[str, Any]],
    cell_terminals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact = _mapping(value, "AI phase index")
    _validate_seal(artifact, "phase_index_sha256")
    expected = build_phase_index(
        registry,
        precommit,
        start_receipt,
        day_seals,
        cell_terminals,
        now_utc=_parse_utc(artifact.get("sealed_at_utc"), "sealed_at_utc"),
    )
    if artifact != expected:
        raise DojoAIForwardError("AI phase index bytes or parents drifted")
    return _snapshot(artifact)


def _find_day_cell(day: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    wanted = _identifier(cell_id, "cell_id")
    cells = day.get("cells")
    if not isinstance(cells, list):
        raise DojoAIForwardError("AI day cell set is invalid")
    matches = [
        _mapping(cell, "AI day cell")
        for cell in cells
        if isinstance(cell, Mapping)
        and isinstance(cell.get("assignment"), Mapping)
        and cell["assignment"].get("cell_id") == wanted
    ]
    if len(matches) != 1:
        raise DojoAIForwardError("cell_id is not uniquely assigned to this day")
    cell = matches[0]
    if cell.get("request_receipt", {}).get("cell_id") != wanted:
        raise DojoAIForwardError("cell request receipt identity drifted")
    return _snapshot(cell)


def _build_prompt_locks(
    registry: Mapping[str, Any], prompt_texts: Mapping[str, str]
) -> dict[str, Any]:
    if not isinstance(prompt_texts, Mapping) or set(prompt_texts) != set(_VARIANTS):
        raise DojoAIForwardError("exact three prompt texts are required")
    locked_at = _parse_utc(registry["locked_at_utc"], "registry locked_at_utc")
    if locked_at != LOCKED_REGISTRY_AT:
        raise DojoAIForwardError("registered prompt lock timestamp drifted")
    locks: dict[str, Any] = {}
    for variant in _VARIANTS:
        text = prompt_texts[variant]
        if not isinstance(text, str):
            raise DojoAIForwardError("prompt text must be text")
        lock = prelock_prompt(text, variant_id=variant, locked_at_utc=locked_at)
        if lock["prompt_sha256"] != LOCKED_VARIANT_PROMPT_SHA256[variant]:
            raise DojoAIForwardError("prompt bytes differ from preregistration")
        locks[variant] = lock
    return locks


def _build_schedule(first: datetime, allocation_nonce: str) -> list[dict[str, Any]]:
    cutoffs: list[datetime] = []
    cursor = first
    while len(cutoffs) < EXPECTED_DAY_COUNT:
        if cursor.weekday() in ELIGIBLE_WEEKDAYS and cursor.time() == DECISION_TIME_UTC:
            cutoffs.append(cursor)
        cursor += timedelta(days=1)
    schedule: list[dict[str, Any]] = []
    for rank, cutoff in enumerate(cutoffs, start=1):
        blind_nonce = canonical_sha256(
            {
                "allocation_nonce": allocation_nonce,
                "experiment_id": LOCKED_EXPERIMENT_ID,
                "purpose": "SOURCE_BLIND_NONCE",
                "rank": rank,
            }
        )
        blind_day_id = canonical_sha256(
            {
                "allocation_nonce": allocation_nonce,
                "experiment_id": LOCKED_EXPERIMENT_ID,
                "purpose": "BLIND_DAY_ID",
                "rank": rank,
            }
        )
        cells = []
        for variant in _VARIANTS:
            identity = {
                "experiment_id": LOCKED_EXPERIMENT_ID,
                "phase_id": PHASE_ID,
                "blind_day_rank": rank,
                "blind_day_id": blind_day_id,
                "variant_id": variant,
            }
            cells.append(
                {
                    "variant_id": variant,
                    "cell_id": "cell-" + canonical_sha256(identity)[:32],
                    "context_id": (
                        f"{LOCKED_EXPERIMENT_ID}:rank-{rank:02d}:{variant}:fresh"
                    ),
                }
            )
        schedule.append(
            {
                "ordinal": rank,
                "blind_day_rank": rank,
                "blind_day_id": blind_day_id,
                "blind_nonce": blind_nonce,
                "decision_cutoff_utc": _iso(cutoff),
                "source_window_start_utc": _iso(cutoff - SOURCE_LOOKBACK),
                "source_window_end_utc": _iso(cutoff),
                "source_not_before_utc": _iso(cutoff + SOURCE_NOT_BEFORE_DELAY),
                "source_seal_deadline_utc": _iso(cutoff + SOURCE_SEAL_DEADLINE_DELAY),
                "response_deadline_utc": _iso(cutoff + RESPONSE_DEADLINE_DELAY),
                "truth_not_before_utc": _iso(
                    cutoff + TRUTH_HORIZON + TRUTH_MATURITY_DELAY
                ),
                "cells": cells,
            }
        )
    return schedule


def _validate_schedule(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DojoAIForwardError("AI forward schedule must be a sequence")
    schedule = [_mapping(item, "schedule row") for item in value]
    if len(schedule) != EXPECTED_DAY_COUNT:
        raise DojoAIForwardError("AI forward schedule must have exactly 30 days")
    first = _parse_utc(schedule[0].get("decision_cutoff_utc"), "first cutoff")
    # We cannot recover the secret allocation nonce from its one-way commitment,
    # so validate every timing and identity relation directly.
    expected_cutoffs: list[datetime] = []
    cursor = first
    while len(expected_cutoffs) < EXPECTED_DAY_COUNT:
        if cursor.weekday() in ELIGIBLE_WEEKDAYS and cursor.time() == DECISION_TIME_UTC:
            expected_cutoffs.append(cursor)
        cursor += timedelta(days=1)
    seen_ids: set[str] = set()
    seen_contexts: set[str] = set()
    for rank, (row, cutoff) in enumerate(zip(schedule, expected_cutoffs), start=1):
        _exact_keys(
            row,
            {
                "ordinal",
                "blind_day_rank",
                "blind_day_id",
                "blind_nonce",
                "decision_cutoff_utc",
                "source_window_start_utc",
                "source_window_end_utc",
                "source_not_before_utc",
                "source_seal_deadline_utc",
                "response_deadline_utc",
                "truth_not_before_utc",
                "cells",
            },
            "schedule row",
        )
        if (
            row["ordinal"] != rank
            or row["blind_day_rank"] != rank
            or row["decision_cutoff_utc"] != _iso(cutoff)
            or row["source_window_start_utc"] != _iso(cutoff - SOURCE_LOOKBACK)
            or row["source_window_end_utc"] != _iso(cutoff)
            or row["source_not_before_utc"] != _iso(cutoff + SOURCE_NOT_BEFORE_DELAY)
            or row["source_seal_deadline_utc"]
            != _iso(cutoff + SOURCE_SEAL_DEADLINE_DELAY)
            or row["response_deadline_utc"] != _iso(cutoff + RESPONSE_DEADLINE_DELAY)
            or row["truth_not_before_utc"]
            != _iso(cutoff + TRUTH_HORIZON + TRUTH_MATURITY_DELAY)
        ):
            raise DojoAIForwardError("AI schedule timing drifted")
        blind_id = _sha(row["blind_day_id"], "blind_day_id")
        _sha(row["blind_nonce"], "blind_nonce")
        if blind_id in seen_ids:
            raise DojoAIForwardError("blind day identity is reused")
        seen_ids.add(blind_id)
        cells = row["cells"]
        if not isinstance(cells, list) or len(cells) != len(_VARIANTS):
            raise DojoAIForwardError("schedule day must have exactly three cells")
        if [item.get("variant_id") for item in cells] != list(_VARIANTS):
            raise DojoAIForwardError("scheduled variants drifted")
        for cell in cells:
            _exact_keys(cell, {"variant_id", "cell_id", "context_id"}, "cell")
            identity = {
                "experiment_id": LOCKED_EXPERIMENT_ID,
                "phase_id": PHASE_ID,
                "blind_day_rank": rank,
                "blind_day_id": blind_id,
                "variant_id": cell["variant_id"],
            }
            if cell["cell_id"] != "cell-" + canonical_sha256(identity)[:32]:
                raise DojoAIForwardError("scheduled cell identity drifted")
            context = _identifier(cell["context_id"], "context_id")
            if context in seen_contexts:
                raise DojoAIForwardError("scheduled fresh context is reused")
            seen_contexts.add(context)
    return _snapshot(schedule)


def _day_context(
    precommit: Mapping[str, Any],
    start: Mapping[str, Any],
    previous_day_seal: Mapping[str, Any] | None,
    *,
    ordinal: int,
    at_utc: datetime,
) -> tuple[dict[str, Any], str]:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        raise DojoAIForwardError("AI day ordinal must be an integer")
    if ordinal < 1 or ordinal > EXPECTED_DAY_COUNT:
        raise DojoAIForwardError("AI day ordinal is outside phase")
    schedule = _snapshot(precommit["schedule"][ordinal - 1])
    if ordinal == 1:
        if previous_day_seal is not None:
            raise DojoAIForwardError("AI day 1 cannot have a previous day")
        return schedule, start["start_receipt_sha256"]
    if previous_day_seal is None:
        raise DojoAIForwardError("AI day chain has a gap")
    prior = _mapping(previous_day_seal, "previous AI day seal")
    _validate_seal(prior, "day_seal_sha256")
    prior_schedule = precommit["schedule"][ordinal - 2]
    prior_sealed = _parse_utc(prior.get("sealed_at_utc"), "previous sealed_at_utc")
    if prior_sealed > at_utc:
        raise DojoAIForwardError("AI day chain time moved backwards")
    if (
        prior.get("ordinal") != ordinal - 1
        or prior.get("schedule") != prior_schedule
        or prior.get("start_receipt_sha256") != start["start_receipt_sha256"]
        or prior.get("experiment_id") != LOCKED_EXPERIMENT_ID
        or prior.get("phase_id") != PHASE_ID
        or prior.get("state") not in {"REQUESTS_SEALED", "MISSING_SOURCE_DEADLINE"}
        or prior.get("authority") != _authority()
    ):
        raise DojoAIForwardError("AI day chain previous ordinal drifted")
    if prior.get("precommit_sha256") != precommit["precommit_sha256"]:
        raise DojoAIForwardError("AI day chain precommit drifted")
    return schedule, prior["day_seal_sha256"]


def _expected_open_m5_slots(start: datetime, end: datetime) -> list[datetime]:
    cursor = _utc(start, "M5 window start")
    stop = _utc(end, "M5 window end")
    if stop - cursor != SOURCE_LOOKBACK:
        raise DojoAIForwardError("AI source window must be exactly 24 hours")
    return expected_oanda_fx_slots(
        cursor,
        stop,
        step=timedelta(minutes=5),
    )


def _source_from_oanda_response(
    value: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    acquired_at_utc: datetime,
    source_capture: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = _mapping(value, "OANDA M5 response")
    _exact_keys(
        response,
        {"instrument", "granularity", "candles"},
        "OANDA M5 response",
    )
    if response["instrument"] != PAIR or response["granularity"] != SOURCE_GRANULARITY:
        raise DojoAIForwardError("OANDA M5 response identity drifted")
    candles = response["candles"]
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        raise DojoAIForwardError("OANDA M5 candles must be a sequence")
    window_start = _parse_utc(
        schedule["source_window_start_utc"], "source window start"
    )
    cutoff = _parse_utc(schedule["decision_cutoff_utc"], "decision cutoff")
    expected_opens = _expected_open_m5_slots(window_start, cutoff)
    if not expected_opens:
        raise DojoAIForwardError("scheduled AI window has no open-market M5 slots")
    expected_open_set = set(expected_opens)
    observations: list[dict[str, Any]] = []
    actual_opens: list[datetime] = []
    previous_open: datetime | None = None
    for index, raw in enumerate(candles):
        candle = _mapping(raw, f"OANDA M5 candle {index}")
        _exact_keys(
            candle,
            {"complete", "volume", "time", "bid", "ask"},
            f"OANDA M5 candle {index}",
        )
        if candle["complete"] is not True:
            raise DojoAIForwardError("OANDA M5 candle is incomplete")
        opened = _parse_oanda_time(candle["time"], f"OANDA M5 candle {index} time")
        if (
            opened < window_start
            or opened + timedelta(minutes=5) > cutoff
            or opened.second != 0
            or opened.microsecond != 0
            or opened.minute % 5 != 0
            or opened not in expected_open_set
        ):
            raise DojoAIForwardError("OANDA M5 candle escapes the causal source window")
        if previous_open is not None and opened <= previous_open:
            raise DojoAIForwardError("OANDA M5 candles are duplicated or unsorted")
        previous_open = opened
        actual_opens.append(opened)
        payload = _validate_candle_payload(
            {
                "bid": candle["bid"],
                "ask": candle["ask"],
                "volume": candle["volume"],
            },
            index=index,
        )
        observations.append(
            {
                # OANDA candle timestamps are opens.  The model-visible source
                # uses the close timestamp so a 14:55 candle becomes observable
                # only at the 15:00 decision cutoff.
                "observed_at_utc": _iso(opened + timedelta(minutes=5)),
                "kind": SOURCE_KIND,
                "payload": payload,
            }
        )
    coverage = _validate_sparse_source_coverage(
        actual_opens,
        expected_opens,
        label="OANDA M5 response",
    )
    source = _validate_day_source(
        {
            "contract": SOURCE_CONTRACT,
            "blind_nonce": schedule["blind_nonce"],
            "pair": PAIR,
            "decision_cutoff_utc": schedule["decision_cutoff_utc"],
            "observations": observations,
        },
        schedule=schedule,
    )
    query = {
        "from": schedule["source_window_start_utc"],
        "granularity": SOURCE_GRANULARITY,
        "includeFirst": "true",
        "price": SOURCE_PRICE,
        "to": schedule["source_window_end_utc"],
    }
    response_value = _snapshot(response)
    capture_value = _mapping(source_capture, "OANDA source capture")
    capture_sha = _sha(
        capture_value.get("source_capture_sha256"), "source_capture_sha256"
    )
    provenance_body = {
        "contract": SOURCE_PROVENANCE_CONTRACT,
        "schema_version": 2,
        "acquired_at_utc": _iso(acquired_at_utc),
        "method": "GET",
        "base_url": OFFICIAL_OANDA_BASE_URL,
        "path": f"/v3/instruments/{PAIR}/candles",
        "query": query,
        "response": response_value,
        "response_sha256": canonical_sha256(response_value),
        "response_candle_count": len(candles),
        "normalized_observation_count": len(observations),
        "slot_coverage": coverage,
        "source_capture": _snapshot(capture_value),
        "source_capture_sha256": capture_sha,
        "credentials_persisted": False,
        "read_only": True,
        "authority": _authority(),
    }
    provenance = _seal(provenance_body, "source_provenance_sha256")
    return source, provenance


def _validate_day_source(
    value: Mapping[str, Any], *, schedule: Mapping[str, Any]
) -> dict[str, Any]:
    source = _mapping(value, "AI day source")
    _exact_keys(
        source,
        {"contract", "blind_nonce", "pair", "decision_cutoff_utc", "observations"},
        "AI day source",
    )
    if (
        source["contract"] != SOURCE_CONTRACT
        or source["blind_nonce"] != schedule["blind_nonce"]
        or source["pair"] != PAIR
        or source["decision_cutoff_utc"] != schedule["decision_cutoff_utc"]
    ):
        raise DojoAIForwardError("AI day source identity differs from precommit")
    observations = source["observations"]
    if not isinstance(observations, list):
        raise DojoAIForwardError("AI day source observations must be a list")
    window_start = _parse_utc(
        schedule["source_window_start_utc"], "source_window_start"
    )
    cutoff = _parse_utc(schedule["decision_cutoff_utc"], "decision_cutoff")
    expected_closes = [
        opened + timedelta(minutes=5)
        for opened in _expected_open_m5_slots(window_start, cutoff)
    ]
    expected_close_set = set(expected_closes)
    previous: datetime | None = None
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(observations):
        observation = _mapping(raw, f"source observation {index}")
        _exact_keys(
            observation,
            {"observed_at_utc", "kind", "payload"},
            f"source observation {index}",
        )
        observed = _parse_utc(
            observation["observed_at_utc"], f"observation {index} time"
        )
        if (
            observed <= window_start
            or observed > cutoff
            or observed.second != 0
            or observed.microsecond != 0
            or observed.minute % 5 != 0
            or observed not in expected_close_set
        ):
            raise DojoAIForwardError("AI observation is outside closed M5 window")
        if previous is not None and observed <= previous:
            raise DojoAIForwardError("AI observations are not strictly chronological")
        previous = observed
        if observation["kind"] != SOURCE_KIND:
            raise DojoAIForwardError("AI observation kind is not allowlisted")
        payload = _validate_candle_payload(observation["payload"], index=index)
        normalized.append(
            {
                "observed_at_utc": _iso(observed),
                "kind": SOURCE_KIND,
                "payload": payload,
            }
        )
    actual_closes = [
        _parse_utc(row["observed_at_utc"], "observation close") for row in normalized
    ]
    _validate_sparse_source_coverage(
        actual_closes,
        expected_closes,
        label="AI day source",
    )
    return {
        "contract": SOURCE_CONTRACT,
        "blind_nonce": schedule["blind_nonce"],
        "pair": PAIR,
        "decision_cutoff_utc": schedule["decision_cutoff_utc"],
        "observations": normalized,
    }


def _validate_sparse_source_coverage(
    actual: Sequence[datetime],
    expected: Sequence[datetime],
    *,
    label: str,
) -> dict[str, Any]:
    if not expected:
        raise DojoAIForwardError(f"{label} expected-slot set is empty")
    numerator, denominator = M5_MINIMUM_COVERAGE
    required = (len(expected) * numerator + denominator - 1) // denominator
    if len(actual) < required:
        raise DojoAIForwardError(f"{label} is below the fixed coverage floor")
    first_lag = actual[0] - expected[0]
    last_lead = expected[-1] - actual[-1]
    if first_lag > M5_BOUNDARY_TOLERANCE or last_lead > M5_BOUNDARY_TOLERANCE:
        raise DojoAIForwardError(f"{label} violates the fixed boundary tolerance")
    gaps = [right - left for left, right in zip(actual, actual[1:])]
    max_gap = max(gaps, default=timedelta(0))
    if max_gap > M5_MAX_CONTIGUOUS_GAP:
        raise DojoAIForwardError(f"{label} violates the fixed gap ceiling")
    actual_set = set(actual)
    missing = [_iso(slot) for slot in expected if slot not in actual_set]
    return {
        "coverage_policy": M5_COVERAGE_POLICY,
        "expected_open_slot_count": len(expected),
        "returned_slot_count": len(actual),
        "missing_slot_count": len(missing),
        "missing_slots_sha256": canonical_sha256(missing),
        "minimum_coverage_numerator": numerator,
        "minimum_coverage_denominator": denominator,
        "required_returned_slot_count": required,
        "boundary_tolerance_seconds": int(M5_BOUNDARY_TOLERANCE.total_seconds()),
        "first_boundary_lag_seconds": int(first_lag.total_seconds()),
        "last_boundary_lead_seconds": int(last_lead.total_seconds()),
        "max_contiguous_gap_seconds": int(M5_MAX_CONTIGUOUS_GAP.total_seconds()),
        "max_observed_gap_seconds": int(max_gap.total_seconds()),
        "coverage_passed": True,
    }


def _validate_candle_payload(value: Any, *, index: int) -> dict[str, Any]:
    source = _mapping(value, f"observation {index} payload")
    _exact_keys(source, {"bid", "ask", "volume"}, "M5 candle payload")
    if (
        isinstance(source["volume"], bool)
        or not isinstance(source["volume"], int)
        or source["volume"] < 0
    ):
        raise DojoAIForwardError("M5 candle volume is invalid")
    bid = _validate_ohlc(source["bid"], "bid")
    ask = _validate_ohlc(source["ask"], "ask")
    for key in ("o", "h", "l", "c"):
        if Decimal(ask[key]) < Decimal(bid[key]):
            raise DojoAIForwardError("M5 candle ask is below bid")
    return {"bid": bid, "ask": ask, "volume": source["volume"]}


def _validate_ohlc(value: Any, label: str) -> dict[str, str]:
    source = _mapping(value, f"M5 {label}")
    _exact_keys(source, {"o", "h", "l", "c"}, f"M5 {label}")
    numbers: dict[str, Decimal] = {}
    normalized: dict[str, str] = {}
    for key in ("o", "h", "l", "c"):
        raw = source[key]
        if not isinstance(raw, str) or _DECIMAL.fullmatch(raw) is None:
            raise DojoAIForwardError("M5 OHLC must use positive decimal strings")
        try:
            number = Decimal(raw)
        except InvalidOperation as exc:
            raise DojoAIForwardError("M5 OHLC decimal is invalid") from exc
        if not number.is_finite() or number <= 0:
            raise DojoAIForwardError("M5 OHLC must be finite and positive")
        numbers[key] = number
        normalized[key] = raw
    if numbers["h"] < max(numbers["o"], numbers["l"], numbers["c"]):
        raise DojoAIForwardError("M5 OHLC high geometry is invalid")
    if numbers["l"] > min(numbers["o"], numbers["h"], numbers["c"]):
        raise DojoAIForwardError("M5 OHLC low geometry is invalid")
    return normalized


def _validate_model_policy(value: Any) -> dict[str, str]:
    source = _mapping(value, "model_policy")
    _exact_keys(
        source,
        {"model_name", "model_version", "model_lineage", "reasoning_effort"},
        "model_policy",
    )
    return {
        key: _identifier(source[key], f"model_policy.{key}")
        for key in ("model_name", "model_version", "model_lineage", "reasoning_effort")
    }


def _validate_source_bindings(value: Any) -> dict[str, Any]:
    source = _mapping(value, "source_bindings")
    _exact_keys(source, {"git_commit", "files"}, "source_bindings")
    commit = source["git_commit"]
    if not isinstance(commit, str) or _GIT_OID.fullmatch(commit) is None:
        raise DojoAIForwardError("source binding git commit is invalid")
    files = _mapping(source["files"], "source binding files")
    if not files:
        raise DojoAIForwardError("source binding files are absent")
    normalized: dict[str, str] = {}
    for path, digest in files.items():
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in path.split("/")
        ):
            raise DojoAIForwardError("source binding path is unsafe")
        normalized[path] = _sha(digest, f"source binding {path}")
    return {"git_commit": commit, "files": dict(sorted(normalized.items()))}


def verify_source_bindings_against_repo(
    value: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Verify declared bytes both in the pinned commit and current checkout."""

    bindings = _validate_source_bindings(value)
    files = bindings["files"]
    missing = REQUIRED_SOURCE_BINDING_FILES - set(files)
    if missing:
        raise DojoAIForwardError(
            "AI operational source binding closure is incomplete: "
            + ",".join(sorted(missing))
        )
    root = repo_root.resolve()
    if not repo_root.is_dir() or repo_root.is_symlink():
        raise DojoAIForwardError("AI source binding repo root is unsafe")
    for relative, expected in files.items():
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or root not in path.resolve().parents
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise DojoAIForwardError(f"AI current source binding drifted: {relative}")
        completed = subprocess.run(
            ["git", "show", f"{bindings['git_commit']}:{relative}"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            completed.returncode != 0
            or hashlib.sha256(completed.stdout).hexdigest() != expected
        ):
            raise DojoAIForwardError(f"AI commit source binding drifted: {relative}")
    return bindings


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = _snapshot(value)
    return {**body, field: canonical_sha256(body)}


def _validate_seal(value: Mapping[str, Any], field: str) -> None:
    digest = _sha(value.get(field), field)
    body = {key: item for key, item in value.items() if key != field}
    if canonical_sha256(body) != digest:
        raise DojoAIForwardError(f"{field} digest mismatch")


def _authority() -> dict[str, Any]:
    return {
        "read_only": True,
        "ai_order_authority": "NONE",
        "broker_mutation_allowed": False,
        "live_permission": False,
        "positive_result_grants_live_permission": False,
    }


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoAIForwardError(f"{label} must be a mapping")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DojoAIForwardError(f"{label} keys are not exact")


def _snapshot(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoAIForwardError("artifact is not finite JSON") from exc


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise DojoAIForwardError(f"{label} must be lowercase SHA-256")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise DojoAIForwardError(f"{label} is not a bounded identifier")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoAIForwardError(f"{label} must be timezone-aware")
    result = value.astimezone(timezone.utc)
    if not math.isfinite(result.timestamp()):
        raise DojoAIForwardError(f"{label} is invalid")
    return result


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DojoAIForwardError(f"{label} must be exact UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DojoAIForwardError(f"{label} is invalid") from exc
    return _utc(parsed, label)


def _parse_oanda_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise DojoAIForwardError(f"{label} must be an aligned OANDA UTC timestamp")
    match = _OANDA_TIME.fullmatch(value)
    if match is None:
        raise DojoAIForwardError(f"{label} must be an aligned OANDA UTC timestamp")
    try:
        parsed = datetime.fromisoformat(match.group(1) + "+00:00")
    except ValueError as exc:
        raise DojoAIForwardError(f"{label} is invalid") from exc
    return _utc(parsed, label)


def _iso(value: datetime) -> str:
    parsed = _utc(value, "timestamp")
    if parsed.microsecond:
        return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "CELL_FAILURE_CONTRACT",
    "CELL_RESPONSE_CONTRACT",
    "DAY_SEAL_CONTRACT",
    "DojoAIForwardError",
    "OFFICIAL_OANDA_BASE_URL",
    "PHASE_INDEX_CONTRACT",
    "PRECOMMIT_CONTRACT",
    "SOURCE_CAPTURE_CONTRACT",
    "SOURCE_REQUEST_CONTRACT",
    "START_CONTRACT",
    "SUPERSESSION_CONTRACT",
    "build_cell_response_failure",
    "build_cell_response_seal",
    "build_day_requests",
    "build_day_requests_from_capture",
    "build_missing_day_seal",
    "build_phase_index",
    "build_precommit",
    "build_start_receipt",
    "build_source_capture",
    "build_source_capture_from_request",
    "prepare_day_request",
    "validate_cell_terminal",
    "validate_day_seal",
    "validate_phase_index",
    "validate_precommit",
    "validate_start_receipt",
    "validate_supersession",
    "validate_source_capture",
    "validate_source_request",
]
