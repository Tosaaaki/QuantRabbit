"""Versioned, deterministic baseline revision for DOJO generation G2.

This module freezes a six-family, exact-28-pair research baseline without
reinterpreting or mutating any earlier sealed trainer artifact.  The registry
artifact produced from this contract is worn historical TRAIN input only: it
cannot grant proof, promotion, live permission, model authority, or broker
mutation.

Validation is intentionally exact.  A consumer may not widen one limit,
reorder one worker, add an undeclared field, or treat unused G3 proposal slots
as G2 capacity.  A later change requires a new contract and schema version.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Final

from quant_rabbit.dojo_bot_catalog import (
    AUTHORITY_INVARIANTS,
    CATALOG_CONTRACT,
    bot_config_sha256,
    catalog_manifest,
    validate_bot_config,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


CONTRACT: Final = "QR_DOJO_G2_BASELINE_REVISION_V1"
SCHEMA_VERSION: Final = 1
REVISION_ID: Final = "DOJO_G2_FIXED_SIX_FAMILY_28PAIR_V1"
REGISTRY_RELATIVE_PATH: Final = (
    "research/registries/dojo_g2_baseline_revision_v1.json"
)
EVIDENCE_TIER: Final = "WORN_HISTORICAL_TRAIN_ONLY"

G2_PAIRS: Final = tuple(sorted(DEFAULT_TRADER_PAIRS))
G2_PROPOSAL_SLOTS_BEFORE: Final = 4
G2_PLANNED_PROPOSAL_SLOTS: Final = 6
MAX_PROPOSAL_SLOTS: Final = 14
G3_PLANNED_PROPOSAL_SLOTS: Final = 4

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_ZERO_SHA256: Final = "0" * 64

_FAMILY_SPECS: Final = (
    ("g2-01-spike-fade-control", "spike_fade", "CONTROL", {}),
    ("g2-02-burst", "burst", "ACTIVE", {}),
    ("g2-03-pullback-limit", "pullback_limit", "ACTIVE", {"pull_atr": 0.6}),
    (
        "g2-04-prev-day-extreme-fade",
        "prev_day_extreme_fade",
        "ACTIVE",
        {},
    ),
    ("g2-05-round-number-fade", "round_number_fade", "ACTIVE", {}),
    (
        "g2-06-mean-revert-24h",
        "mean_revert_24h",
        "ACTIVE",
        {"fade_atr": 1.2},
    ),
)


class DojoG2BaselineError(ValueError):
    """The G2 baseline artifact is malformed or differs from sealed V1."""


def canonical_sha256(value: Any) -> str:
    """Return strict canonical JSON SHA-256 for a JSON-safe value."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoG2BaselineError("value is not strict canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _worker_config(family: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "signal": family,
        "pairs": list(G2_PAIRS),
        "tp_atr": 3.0,
        "sl_pips": 25.0,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 1,
        "per_pos_lev": 2.0,
        "atr_floor_pips": 0.5,
        "exit_policy": "FIXED",
        **dict(AUTHORITY_INVARIANTS),
        **dict(parameters),
    }
    return validate_bot_config(config)


def _catalog_sha256() -> str:
    return canonical_sha256(catalog_manifest())


def _expected_body() -> dict[str, Any]:
    workers: list[dict[str, Any]] = []
    catalog_sha256 = _catalog_sha256()
    for worker_id, family, role, parameters in _FAMILY_SPECS:
        config = _worker_config(family, parameters)
        workers.append(
            {
                "worker_id": worker_id,
                "family": family,
                "role": role,
                "execution_status": "PREREGISTERED_NOT_EXECUTED",
                "planned_proposal_slot_cost_if_executed": 1,
                "catalog_contract": CATALOG_CONTRACT,
                "catalog_sha256": catalog_sha256,
                "config": config,
                "config_sha256": bot_config_sha256(config),
            }
        )

    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "revision_id": REVISION_ID,
        "generation": "G2",
        "parent_generation": "G1",
        "execution_status": "PREREGISTERED_NOT_EXECUTED",
        "baseline_policy": {
            "kind": "DETERMINISTIC_VERSIONED_BASELINE",
            "mutates_prior_sealed_artifacts": False,
            "reinterprets_prior_results": False,
            "supersedes_prior_sealed_artifacts": False,
            "future_changes_require_new_contract_version": True,
        },
        "universe": {
            "pair_contract": "FORMAL_G8_EXACT_28",
            "pair_count": 28,
            "pairs": list(G2_PAIRS),
        },
        "workers": workers,
        "allocator": {
            "contract": "QR_DOJO_G2_FIXED_ALLOCATOR_V1",
            "simultaneous_slots": 4,
            "max_concurrent_per_pair": 1,
            "max_concurrent_per_family": 1,
            "worker_global_max_concurrent": 1,
            "per_position_leverage": 2.0,
            "maximum_gross_leverage": 8.0,
            "gross_leverage_formula": (
                "simultaneous_slots*per_position_leverage"
            ),
            "new_position_margin_admission_fraction_max": 0.45,
            "margin_closeout_fraction": 0.90,
            "portfolio_stop_risk_fraction": 0.10,
            "allocation_changes_require_new_version": True,
        },
        "search_budget": {
            "generation": "G2",
            "execution_status": "PREREGISTERED_NOT_EXECUTED",
            "observed_proposal_slots_consumed_before": G2_PROPOSAL_SLOTS_BEFORE,
            "actual_g2_model_invocations": 0,
            "actual_g2_reservation_events": 0,
            "actual_g2_reserved_proposal_slots": 0,
            "actual_g2_proposal_slots_consumed": 0,
            "planned_proposal_slots_for_g2": G2_PLANNED_PROPOSAL_SLOTS,
            "projected_proposal_slots_consumed_after_execution": (
                G2_PROPOSAL_SLOTS_BEFORE + G2_PLANNED_PROPOSAL_SLOTS
            ),
            "max_proposal_slots": MAX_PROPOSAL_SLOTS,
            "projected_proposal_slots_remaining_after_execution": (
                G3_PLANNED_PROPOSAL_SLOTS
            ),
            "next_planned_generation": "G3",
            "planned_g3_proposal_slots_after_g2_execution": (
                G3_PLANNED_PROPOSAL_SLOTS
            ),
            "g3_intended_comparison": "FIXED_VS_BREAKEVEN_VS_ATR_TRAILING",
            "planned_slots_may_not_be_reassigned_without_new_version": True,
        },
        "authority": {
            "classification": EVIDENCE_TIER,
            "diagnostic_train_only": True,
            "historical_only": True,
            "proof_eligible": False,
            "forward_proof_eligible": False,
            "promotion_eligible": False,
            "live_permission": False,
            "order_authority": "NONE",
            "broker_mutation_allowed": False,
            "trainer_may_change_live_configuration": False,
            "automatic_deployment_allowed": False,
        },
    }


def build_g2_baseline() -> dict[str, Any]:
    """Build the one canonical G2 V1 artifact, including its body digest."""

    if len(G2_PAIRS) != 28 or len(set(G2_PAIRS)) != 28:
        raise DojoG2BaselineError("formal G8 pair universe is no longer exact-28")
    if len(_FAMILY_SPECS) != G2_PLANNED_PROPOSAL_SLOTS:
        raise DojoG2BaselineError("G2 worker count no longer matches its slot budget")
    body = _expected_body()
    return {**body, "artifact_sha256": canonical_sha256(body)}


def validate_g2_baseline(value: Mapping[str, Any]) -> dict[str, Any]:
    """Require byte-semantically exact G2 V1 content and all cross-bindings."""

    if not isinstance(value, Mapping):
        raise DojoG2BaselineError("G2 baseline must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise DojoG2BaselineError("G2 baseline keys must be strings")
    try:
        candidate = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoG2BaselineError(
            "G2 baseline must contain only strict canonical JSON values"
        ) from exc

    expected = build_g2_baseline()
    if set(candidate) != set(expected):
        raise DojoG2BaselineError("G2 baseline top-level schema mismatch")
    digest = candidate.get("artifact_sha256")
    if (
        not isinstance(digest, str)
        or _SHA256_RE.fullmatch(digest) is None
        or digest == _ZERO_SHA256
    ):
        raise DojoG2BaselineError(
            "G2 baseline artifact_sha256 must be a non-zero lowercase SHA-256"
        )
    body = {key: item for key, item in candidate.items() if key != "artifact_sha256"}
    if canonical_sha256(body) != digest:
        raise DojoG2BaselineError("G2 baseline artifact SHA-256 mismatch")

    workers = candidate.get("workers")
    if not isinstance(workers, list) or len(workers) != G2_PLANNED_PROPOSAL_SLOTS:
        raise DojoG2BaselineError("G2 baseline must contain exactly six workers")
    for index, worker in enumerate(workers):
        if not isinstance(worker, Mapping):
            raise DojoG2BaselineError(f"workers[{index}] must be a JSON object")
        config = worker.get("config")
        if not isinstance(config, Mapping):
            raise DojoG2BaselineError(f"workers[{index}].config must be an object")
        try:
            normalized = validate_bot_config(config)
        except ValueError as exc:
            raise DojoG2BaselineError(
                f"workers[{index}].config is outside the reviewed catalog"
            ) from exc
        if normalized != config:
            raise DojoG2BaselineError(
                f"workers[{index}].config is not catalog-normalized"
            )
        if bot_config_sha256(config) != worker.get("config_sha256"):
            raise DojoG2BaselineError(f"workers[{index}] config SHA-256 mismatch")

    if candidate != expected:
        raise DojoG2BaselineError(
            "G2 baseline differs from the exact versioned V1 contract"
        )
    return expected


__all__ = [
    "CONTRACT",
    "DojoG2BaselineError",
    "EVIDENCE_TIER",
    "G2_PAIRS",
    "G2_PLANNED_PROPOSAL_SLOTS",
    "G3_PLANNED_PROPOSAL_SLOTS",
    "MAX_PROPOSAL_SLOTS",
    "REGISTRY_RELATIVE_PATH",
    "REVISION_ID",
    "SCHEMA_VERSION",
    "build_g2_baseline",
    "canonical_sha256",
    "validate_g2_baseline",
]
