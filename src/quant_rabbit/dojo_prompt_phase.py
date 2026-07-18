"""Immutable experiment-level scoring for the locked DOJO prompt comparison.

The older per-trial pilot scorer is deliberately not used here: combining A,
B, and C would compound mutually exclusive prompt choices into one fictitious
portfolio.  This module binds every allocated cell to the reviewed
preregistration, keeps response failures in the denominator as economic FLAT,
and reports separate variant paths plus paired within-day contrasts.

All artifacts remain self-attested diagnostics.  The module does not call a
model, read a broker, claim independent model samples, or grant live authority.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from quant_rabbit.dojo_ai_discretion import (
    DIAGNOSTIC_TIER,
    VALID,
    canonical_sha256,
    validate_score_receipt,
)


PREREGISTRATION_CONTRACT = "DOJO_PROMPT_EXPERIMENT_PREREGISTRATION"
PHASE_MANIFEST_CONTRACT = "QR_DOJO_PROMPT_PHASE_MANIFEST_V1"
PHASE_SCORE_CONTRACT = "QR_DOJO_PROMPT_PHASE_SCORE_V1"
LOCKED_EXPERIMENT_ID = "dojo-prompt-experiment-v1"
LOCKED_PREREGISTRATION_SHA256 = (
    "cee86c2060511389183591cf071ff91eeb20c45a9c29ba6481d1619f61cb0272"
)
LOCKED_SCORER_POLICY_SHA256 = (
    "76eff0705971fdd149bdf4364b240e9c60d61ac8fb61542e80345472008b8496"
)
LOCKED_VARIANT_PROMPT_SHA256 = {
    "A_FABLE_MINIMAL": (
        "b20c37aff8be4282110a05457c5434e6dc690fd70aa533e06c69e2c50a30cccf"
    ),
    "B_CALIBRATED_ABSTENTION": (
        "97b413e5d5ad978a7825d08e0cb390e8d23bbb72d88f9023a6e67abfc219c2c2"
    ),
    "C_STRUCTURAL_REGIME": (
        "02a204ade67002fae04a556b5a7f466a07cef28001620d1b7e6024c07bbb4fd5"
    ),
}
PHASE_RANKS = {
    "phase_1_diagnostic": tuple(range(1, 31)),
    "future_confirmation": tuple(range(31, 61)),
}
CELL_TERMINAL_FAILURES = frozenset(
    {
        "MISSING_RESPONSE_DEADLINE",
        "SCHEMA_INVALID_RESPONSE",
        "EXECUTOR_FAILURE",
        "INVALID_PARENT",
        "RESPONSE_FORK_DETECTED",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ASSIGNMENT_KEYS = frozenset(
    {
        "cell_id",
        "phase_id",
        "blind_day_rank",
        "blind_day_id",
        "variant_id",
        "source_sha256",
        "packet_sha256",
        "prompt_sha256",
        "prompt_lock_sha256",
        "model_sha256",
        "capability_manifest_sha256",
        "request_receipt_sha256",
        "context_id",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "contract",
        "schema_version",
        "validity_status",
        "experiment_id",
        "preregistration_sha256",
        "phase_id",
        "locked_at_utc",
        "expected_blind_day_ranks",
        "expected_variant_ids",
        "allocated_cell_count",
        "cells",
        "scorer_policy_sha256",
        "missing_response_economic_action",
        "missing_response_remains_failure",
        "responses_visible_before_phase_seal",
        "variant_portfolios_combined",
        "paired_contrast_policy",
        "phase_1_selection_allowed",
        "positive_result_grants_live_permission",
        "evidence_tier",
        "external_attestations_verified",
        "proof_eligible",
        "effective_independent_n",
        "read_only",
        "ai_order_authority",
        "live_permission",
        "broker_mutation_allowed",
        "phase_manifest_sha256",
    }
)
_DIAGNOSTIC_FIELDS = {
    "evidence_tier": DIAGNOSTIC_TIER,
    "external_attestations_verified": False,
    "proof_eligible": False,
    "effective_independent_n": 0,
    "read_only": True,
    "ai_order_authority": "NONE",
    "live_permission": False,
    "broker_mutation_allowed": False,
}


class DojoPromptPhaseError(ValueError):
    """The locked allocation or a cell result is inconsistent."""


def preregistration_sha256(registry: Mapping[str, Any]) -> str:
    return canonical_sha256(_snapshot(registry))


def assert_locked_preregistration(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact reviewed preregistration, not a caller look-alike."""

    value = _snapshot(registry)
    if value.get("contract") != PREREGISTRATION_CONTRACT:
        raise DojoPromptPhaseError("prompt preregistration contract is invalid")
    if value.get("experiment_id") != LOCKED_EXPERIMENT_ID:
        raise DojoPromptPhaseError("prompt experiment id is invalid")
    actual = canonical_sha256(value)
    if actual != LOCKED_PREREGISTRATION_SHA256:
        raise DojoPromptPhaseError("prompt preregistration bytes drifted")
    policy = value.get("score_policy")
    if not isinstance(policy, Mapping):
        raise DojoPromptPhaseError("registered scorer policy is missing")
    if (
        value.get("scorer_policy_sha256") != LOCKED_SCORER_POLICY_SHA256
        or canonical_sha256(policy) != LOCKED_SCORER_POLICY_SHA256
    ):
        raise DojoPromptPhaseError("registered scorer policy drifted")
    variants = value.get("variants")
    if not isinstance(variants, list):
        raise DojoPromptPhaseError("registered variants are missing")
    prompt_hashes = {
        item.get("variant_id"): item.get("prompt_sha256")
        for item in variants
        if isinstance(item, Mapping)
    }
    if prompt_hashes != LOCKED_VARIANT_PROMPT_SHA256:
        raise DojoPromptPhaseError("registered prompt hashes drifted")
    return value


def build_cell_assignment(
    *,
    phase_id: str,
    blind_day_rank: int,
    blind_day_id: str,
    variant_id: str,
    source_sha256: str,
    packet_sha256: str,
    prompt_sha256: str,
    prompt_lock_sha256: str,
    model_sha256: str,
    capability_manifest_sha256: str,
    request_receipt_sha256: str,
    context_id: str,
) -> dict[str, Any]:
    """Create one deterministic cell identity for an already sealed request."""

    if phase_id not in PHASE_RANKS:
        raise DojoPromptPhaseError("phase id is unsupported")
    if isinstance(blind_day_rank, bool) or blind_day_rank not in PHASE_RANKS[phase_id]:
        raise DojoPromptPhaseError("blind day rank is outside the phase")
    blind_id = _sha(blind_day_id, "blind_day_id")
    if variant_id not in LOCKED_VARIANT_PROMPT_SHA256:
        raise DojoPromptPhaseError("variant id is not preregistered")
    if prompt_sha256 != LOCKED_VARIANT_PROMPT_SHA256[variant_id]:
        raise DojoPromptPhaseError("cell prompt hash does not match preregistration")
    context = _text(context_id, "context_id", maximum=200)
    hashes = {
        "source_sha256": _sha(source_sha256, "source_sha256"),
        "packet_sha256": _sha(packet_sha256, "packet_sha256"),
        "prompt_sha256": _sha(prompt_sha256, "prompt_sha256"),
        "prompt_lock_sha256": _sha(prompt_lock_sha256, "prompt_lock_sha256"),
        "model_sha256": _sha(model_sha256, "model_sha256"),
        "capability_manifest_sha256": _sha(
            capability_manifest_sha256, "capability_manifest_sha256"
        ),
        "request_receipt_sha256": _sha(
            request_receipt_sha256, "request_receipt_sha256"
        ),
    }
    identity = {
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "phase_id": phase_id,
        "blind_day_rank": blind_day_rank,
        "blind_day_id": blind_id,
        "variant_id": variant_id,
    }
    return {
        "cell_id": "cell-" + canonical_sha256(identity)[:32],
        **{
            key: identity[key]
            for key in ("phase_id", "blind_day_rank", "blind_day_id", "variant_id")
        },
        **hashes,
        "context_id": context,
    }


def build_phase_manifest(
    registry: Mapping[str, Any],
    *,
    phase_id: str,
    assignments: Sequence[Mapping[str, Any]],
    locked_at_utc: datetime,
) -> dict[str, Any]:
    """Freeze the exact 30-day x 3-variant allocation before responses."""

    prereg = assert_locked_preregistration(registry)
    if phase_id not in PHASE_RANKS:
        raise DojoPromptPhaseError("phase id is unsupported")
    if isinstance(assignments, (str, bytes)) or not isinstance(assignments, Sequence):
        raise DojoPromptPhaseError("assignments must be a sequence")
    cells = [_validate_assignment(item, phase_id=phase_id) for item in assignments]
    expected_count = len(PHASE_RANKS[phase_id]) * len(LOCKED_VARIANT_PROMPT_SHA256)
    if len(cells) != expected_count:
        raise DojoPromptPhaseError(
            f"phase requires exactly {expected_count} allocated cells"
        )
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    context_ids: set[str] = set()
    request_ids: set[str] = set()
    for cell in cells:
        key = (cell["blind_day_rank"], cell["variant_id"])
        if key in by_key:
            raise DojoPromptPhaseError("phase contains a duplicate rank/variant cell")
        by_key[key] = cell
        if cell["context_id"] in context_ids:
            raise DojoPromptPhaseError("fresh model context is reused across cells")
        context_ids.add(cell["context_id"])
        if cell["request_receipt_sha256"] in request_ids:
            raise DojoPromptPhaseError("request receipt is reused across cells")
        request_ids.add(cell["request_receipt_sha256"])
    for rank in PHASE_RANKS[phase_id]:
        day_cells = [
            by_key[(rank, variant)] for variant in LOCKED_VARIANT_PROMPT_SHA256
        ]
        if len({cell["blind_day_id"] for cell in day_cells}) != 1:
            raise DojoPromptPhaseError("variants do not share one blind day id")
        if len({cell["source_sha256"] for cell in day_cells}) != 1:
            raise DojoPromptPhaseError("variants do not share one source commitment")
    cells.sort(key=lambda item: (item["blind_day_rank"], item["variant_id"]))
    body = {
        "contract": PHASE_MANIFEST_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "preregistration_sha256": canonical_sha256(prereg),
        "phase_id": phase_id,
        "locked_at_utc": _utc(locked_at_utc).isoformat(),
        "expected_blind_day_ranks": list(PHASE_RANKS[phase_id]),
        "expected_variant_ids": list(LOCKED_VARIANT_PROMPT_SHA256),
        "allocated_cell_count": expected_count,
        "cells": cells,
        "scorer_policy_sha256": LOCKED_SCORER_POLICY_SHA256,
        "missing_response_economic_action": "SYNTHETIC_FLAT_ZERO_RETURN",
        "missing_response_remains_failure": True,
        "responses_visible_before_phase_seal": False,
        "variant_portfolios_combined": False,
        "paired_contrast_policy": "WITHIN_BLIND_DAY_B_MINUS_A_AND_C_MINUS_A",
        "phase_1_selection_allowed": False,
        "positive_result_grants_live_permission": False,
        **_DIAGNOSTIC_FIELDS,
    }
    return _seal(body, "phase_manifest_sha256")


def score_prompt_phase(
    registry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    scored_cells: Mapping[str, Mapping[str, Any]],
    terminal_failures: Mapping[str, str],
    sealed_at_utc: datetime,
) -> dict[str, Any]:
    """Seal all allocated cells and score variants without cross-compounding."""

    prereg = assert_locked_preregistration(registry)
    phase = _validate_manifest(manifest)
    if phase["preregistration_sha256"] != canonical_sha256(prereg):
        raise DojoPromptPhaseError("phase manifest preregistration binding is stale")
    if not isinstance(scored_cells, Mapping) or not isinstance(
        terminal_failures, Mapping
    ):
        raise DojoPromptPhaseError("cell results must be mappings")
    if set(scored_cells).intersection(terminal_failures):
        raise DojoPromptPhaseError("a cell cannot be both scored and failed")
    assignment_by_id = {cell["cell_id"]: cell for cell in phase["cells"]}
    unknown = (set(scored_cells) | set(terminal_failures)) - set(assignment_by_id)
    if unknown:
        raise DojoPromptPhaseError("cell results contain an unallocated cell")

    terminal_rows: list[dict[str, Any]] = []
    seen_response_receipts: dict[str, str] = {}
    for cell_id, assignment in assignment_by_id.items():
        if cell_id in scored_cells:
            score = validate_score_receipt(scored_cells[cell_id])
            if score.get("validity_status") != VALID:
                raise DojoPromptPhaseError("invalidated score must be a failed cell")
            for score_key, assignment_key in (
                ("packet_sha256", "packet_sha256"),
                ("prompt_sha256", "prompt_sha256"),
                ("prompt_lock_sha256", "prompt_lock_sha256"),
                ("model_sha256", "model_sha256"),
            ):
                if score.get(score_key) != assignment[assignment_key]:
                    raise DojoPromptPhaseError(
                        f"score {score_key} does not match allocated cell"
                    )
            response_receipt = str(score["response_receipt_sha256"])
            prior_cell = seen_response_receipts.get(response_receipt)
            if prior_cell is not None:
                raise DojoPromptPhaseError(
                    f"response receipt is reused by {prior_cell} and {cell_id}"
                )
            seen_response_receipts[response_receipt] = cell_id
            terminal_rows.append(
                {
                    "cell_id": cell_id,
                    "blind_day_rank": assignment["blind_day_rank"],
                    "blind_day_id": assignment["blind_day_id"],
                    "variant_id": assignment["variant_id"],
                    "status": "VALID_RESPONSE",
                    "response_failure": False,
                    "economic_fallback": None,
                    "return_key": score["return_key"],
                    "net_return": score["net_return"],
                    "net_log_growth": score["net_log_growth"],
                    "score_receipt_sha256": score["score_receipt_sha256"],
                    "declared_model_lineage": score["declared_model_lineage"],
                }
            )
        else:
            reason = terminal_failures.get(cell_id, "MISSING_RESPONSE_DEADLINE")
            if reason not in CELL_TERMINAL_FAILURES:
                raise DojoPromptPhaseError("terminal failure reason is unsupported")
            terminal_rows.append(
                {
                    "cell_id": cell_id,
                    "blind_day_rank": assignment["blind_day_rank"],
                    "blind_day_id": assignment["blind_day_id"],
                    "variant_id": assignment["variant_id"],
                    "status": reason,
                    "response_failure": True,
                    "economic_fallback": "SYNTHETIC_FLAT_ZERO_RETURN",
                    "return_key": None,
                    "net_return": 0.0,
                    "net_log_growth": 0.0,
                    "score_receipt_sha256": None,
                    "declared_model_lineage": None,
                }
            )

    terminal_rows.sort(key=lambda row: (row["blind_day_rank"], row["variant_id"]))
    variants = {
        variant: _variant_summary(terminal_rows, variant)
        for variant in LOCKED_VARIANT_PROMPT_SHA256
    }
    by_key = {(row["blind_day_rank"], row["variant_id"]): row for row in terminal_rows}
    paired_rows = []
    for rank in PHASE_RANKS[phase["phase_id"]]:
        a = by_key[(rank, "A_FABLE_MINIMAL")]
        b = by_key[(rank, "B_CALIBRATED_ABSTENTION")]
        c = by_key[(rank, "C_STRUCTURAL_REGIME")]
        complete = not any(row["response_failure"] for row in (a, b, c))
        paired_rows.append(
            {
                "blind_day_rank": rank,
                "blind_day_id": a["blind_day_id"],
                "all_three_valid": complete,
                "b_minus_a_log_growth": b["net_log_growth"] - a["net_log_growth"],
                "c_minus_a_log_growth": c["net_log_growth"] - a["net_log_growth"],
                "inference_eligible": complete,
            }
        )
    failure_count = sum(row["response_failure"] for row in terminal_rows)
    complete_pairs = [row for row in paired_rows if row["inference_eligible"]]
    contrasts = {
        "complete_day_count": len(complete_pairs),
        "allocated_day_count": len(paired_rows),
        "b_minus_a_total_log_growth_all_days": sum(
            row["b_minus_a_log_growth"] for row in paired_rows
        ),
        "c_minus_a_total_log_growth_all_days": sum(
            row["c_minus_a_log_growth"] for row in paired_rows
        ),
        "complete_response_requirement_met": (
            failure_count == 0 and len(complete_pairs) == len(paired_rows)
        ),
        # External chronology/capability attestations are absent in V1.
        "confirmatory_inference_allowed": False,
        "holm_corrected_test_performed": False,
        "selection_status": "NO_SELECTION_DIAGNOSTIC_ONLY",
    }
    body = {
        "contract": PHASE_SCORE_CONTRACT,
        "schema_version": 1,
        "validity_status": VALID,
        "experiment_id": LOCKED_EXPERIMENT_ID,
        "preregistration_sha256": canonical_sha256(prereg),
        "phase_id": phase["phase_id"],
        "phase_manifest_sha256": phase["phase_manifest_sha256"],
        "sealed_at_utc": _utc(sealed_at_utc).isoformat(),
        "allocated_cell_count": len(terminal_rows),
        "valid_response_cell_count": len(terminal_rows) - failure_count,
        "response_failure_cell_count": failure_count,
        "terminal_cell_count": len(terminal_rows),
        "all_allocated_cells_terminal": True,
        "missing_responses_count_in_denominator": True,
        "missing_response_is_genuine_flat": False,
        "variant_portfolios_combined": False,
        "variant_summaries": variants,
        "paired_contrasts": contrasts,
        "cell_results": terminal_rows,
        "paired_day_results": paired_rows,
        "positive_superiority_claim_allowed": False,
        "prompt_selection_allowed": False,
        "independent_model_n_status": "UNATTESTED_ZERO",
        **_DIAGNOSTIC_FIELDS,
    }
    return _seal(body, "phase_score_sha256")


def _variant_summary(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant_id"] == variant]
    total_log = sum(float(row["net_log_growth"]) for row in selected)
    failures = sum(bool(row["response_failure"]) for row in selected)
    genuine_flat = sum(
        row["status"] == "VALID_RESPONSE" and row["return_key"] == "FLAT"
        for row in selected
    )
    return {
        "allocated_cell_count": len(selected),
        "valid_response_cell_count": len(selected) - failures,
        "response_failure_cell_count": failures,
        "genuine_flat_count": genuine_flat,
        "directional_response_count": len(selected) - failures - genuine_flat,
        "synthetic_flat_failure_count": failures,
        "total_log_growth": total_log,
        "compounded_net_return": math.expm1(total_log),
        "diagnostic_only": True,
    }


def _validate_assignment(value: Mapping[str, Any], *, phase_id: str) -> dict[str, Any]:
    item = _snapshot(value)
    if set(item) != _ASSIGNMENT_KEYS:
        raise DojoPromptPhaseError("cell assignment shape is invalid")
    rebuilt = build_cell_assignment(
        phase_id=item["phase_id"],
        blind_day_rank=item["blind_day_rank"],
        blind_day_id=item["blind_day_id"],
        variant_id=item["variant_id"],
        source_sha256=item["source_sha256"],
        packet_sha256=item["packet_sha256"],
        prompt_sha256=item["prompt_sha256"],
        prompt_lock_sha256=item["prompt_lock_sha256"],
        model_sha256=item["model_sha256"],
        capability_manifest_sha256=item["capability_manifest_sha256"],
        request_receipt_sha256=item["request_receipt_sha256"],
        context_id=item["context_id"],
    )
    if item != rebuilt or item["phase_id"] != phase_id:
        raise DojoPromptPhaseError("cell assignment identity is stale")
    return item


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    phase = _snapshot(value)
    if set(phase) != _MANIFEST_KEYS:
        raise DojoPromptPhaseError("phase manifest shape is invalid")
    if phase.get("contract") != PHASE_MANIFEST_CONTRACT:
        raise DojoPromptPhaseError("phase manifest contract is invalid")
    claimed = phase.get("phase_manifest_sha256")
    if not isinstance(claimed, str) or claimed != canonical_sha256(
        {key: item for key, item in phase.items() if key != "phase_manifest_sha256"}
    ):
        raise DojoPromptPhaseError("phase manifest seal is invalid")
    if phase.get("evidence_tier") != DIAGNOSTIC_TIER:
        raise DojoPromptPhaseError("phase manifest overstates evidence tier")
    if (
        phase.get("live_permission") is not False
        or phase.get("broker_mutation_allowed") is not False
        or phase.get("effective_independent_n") != 0
    ):
        raise DojoPromptPhaseError("phase manifest violates diagnostic boundary")
    cells = phase.get("cells")
    if not isinstance(cells, list):
        raise DojoPromptPhaseError("phase manifest cells are missing")
    phase_id = str(phase.get("phase_id"))
    if phase_id not in PHASE_RANKS:
        raise DojoPromptPhaseError("phase manifest phase id is unsupported")
    validated = [_validate_assignment(cell, phase_id=phase_id) for cell in cells]
    expected = len(PHASE_RANKS.get(phase_id, ())) * len(LOCKED_VARIANT_PROMPT_SHA256)
    if len(validated) != expected or phase.get("allocated_cell_count") != expected:
        raise DojoPromptPhaseError("phase manifest allocation is incomplete")
    keys = [(cell["blind_day_rank"], cell["variant_id"]) for cell in validated]
    if len(set(keys)) != expected:
        raise DojoPromptPhaseError("phase manifest contains duplicate cells")
    for rank in PHASE_RANKS[phase_id]:
        day = [cell for cell in validated if cell["blind_day_rank"] == rank]
        if len(day) != len(LOCKED_VARIANT_PROMPT_SHA256):
            raise DojoPromptPhaseError("phase manifest day is incomplete")
        if (
            len({cell["blind_day_id"] for cell in day}) != 1
            or len({cell["source_sha256"] for cell in day}) != 1
        ):
            raise DojoPromptPhaseError("phase manifest day sharing is invalid")
    if len({cell["context_id"] for cell in validated}) != expected:
        raise DojoPromptPhaseError("phase manifest reuses a model context")
    return phase


def _seal(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = _snapshot(body)
    value.pop(field, None)
    return {**value, field: canonical_sha256(value)}


def _snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DojoPromptPhaseError("artifact must be finite JSON") from exc
    if not isinstance(decoded, dict):
        raise DojoPromptPhaseError("artifact must be one JSON object")
    return decoded


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise DojoPromptPhaseError(f"{field} must be a lowercase sha256")
    return value


def _text(value: Any, field: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DojoPromptPhaseError(f"{field} is required and bounded")
    return value


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DojoPromptPhaseError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


__all__ = [
    "CELL_TERMINAL_FAILURES",
    "DojoPromptPhaseError",
    "LOCKED_EXPERIMENT_ID",
    "LOCKED_PREREGISTRATION_SHA256",
    "LOCKED_SCORER_POLICY_SHA256",
    "LOCKED_VARIANT_PROMPT_SHA256",
    "PHASE_MANIFEST_CONTRACT",
    "PHASE_RANKS",
    "PHASE_SCORE_CONTRACT",
    "assert_locked_preregistration",
    "build_cell_assignment",
    "build_phase_manifest",
    "preregistration_sha256",
    "score_prompt_phase",
]
