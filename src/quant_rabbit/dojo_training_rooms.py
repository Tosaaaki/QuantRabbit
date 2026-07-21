"""Fail-closed isolation contracts for DOJO training rooms.

One room owns one strategy or research thesis.  Rooms may share only the exact
source, evaluator, cost-policy, and risk-policy bindings frozen by their common
registry.  Trainer lineage, search budget, artifact namespace, fixed TRAIN
denominator, parameters, and results remain room-local.

The common sparring handoff is the sole integration point.  It accepts at most
one terminal validated candidate from each room and preregisters a new
shared-capital, four-slot evaluation on its own fixed denominator.  A room's
TRAIN result is selection provenance only and is never reused as a sparring,
holdout, or forward result.

This module is pure and research-only: it performs no file, model, broker, or
runtime writes and grants no order or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final


ROOM_REGISTRY_CONTRACT: Final = "QR_DOJO_TRAINING_ROOM_REGISTRY_V1"
ROOM_RECEIPT_CONTRACT: Final = "QR_DOJO_TRAINING_ROOM_RECEIPT_V1"
QUEUE_BOUND_ROOM_RECEIPT_CONTRACT: Final = "QR_DOJO_TRAINING_ROOM_RECEIPT_V2"
ROOM_DENOMINATOR_CONTRACT: Final = "QR_DOJO_ROOM_FIXED_TRAIN_DENOMINATOR_V1"
COMMON_SPARRING_HANDOFF_CONTRACT: Final = "QR_DOJO_COMMON_SPARRING_HANDOFF_V1"
QUEUE_BOUND_COMMON_SPARRING_HANDOFF_CONTRACT: Final = (
    "QR_DOJO_COMMON_SPARRING_HANDOFF_V2"
)
COMMON_SPARRING_DENOMINATOR_CONTRACT: Final = (
    "QR_DOJO_COMMON_SPARRING_FIXED_DENOMINATOR_V1"
)
SCHEMA_VERSION: Final = 1
QUEUE_BOUND_SCHEMA_VERSION: Final = 2
TAXONOMY_REVISION: Final = "DOJO_ONE_THESIS_PER_ROOM_TAXONOMY_V1"
ROOM_ARTIFACT_ROOT: Final = "research/dojo/training_rooms"
COMMON_SPARRING_SLOTS: Final = 4

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_SHARED_BINDING_NAMES: Final = (
    "source",
    "evaluator",
    "cost_policy",
    "risk_policy",
)
_SEARCH_BUDGET_KEYS: Final = frozenset(
    {
        "max_attempts",
        "max_hypotheses",
        "max_parameter_revisions",
        "max_model_calls",
    }
)
_BUDGET_CONSUMED_KEYS: Final = frozenset(
    {"attempts", "hypotheses", "parameter_revisions", "model_calls"}
)
_DENOMINATOR_INPUT_KEYS: Final = frozenset(
    {"denominator_id", "coordinate_set_sha256", "expected_coordinate_count"}
)
_CANDIDATE_INPUT_KEYS: Final = frozenset(
    {
        "candidate_id",
        "hypothesis_id",
        "revision_id",
        "parameters_sha256",
        "terminal_result_sha256",
        "cross_room_parent",
    }
)
_CROSS_ROOM_PARENT_KEYS: Final = frozenset(
    {
        "source_room_id",
        "source_candidate_id",
        "source_hypothesis_id",
        "source_revision_id",
        "source_parameters_sha256",
        "source_result_sha256",
    }
)

_AUTHORITY: Final = {
    "research_only": True,
    "historical_train_is_proof": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "automatic_deployment_allowed": False,
}

_SHARING_POLICY: Final = {
    "allowed_shared_components": [
        "SOURCE_BINDING",
        "EVALUATOR_BINDING",
        "COST_POLICY_BINDING",
        "RISK_POLICY_BINDING",
    ],
    "room_local_components": [
        "STRATEGY_OR_RESEARCH_THESIS",
        "TRAINER_LINEAGE",
        "SEARCH_BUDGET",
        "ARTIFACT_NAMESPACE",
        "FIXED_TRAIN_DENOMINATOR",
        "PARAMETERS",
        "RESULTS",
    ],
    "cross_room_result_copy_allowed_without_new_hypothesis_revision": False,
    "cross_room_parameter_copy_allowed_without_new_hypothesis_revision": False,
    "cross_room_import_requires_new_hypothesis": True,
    "cross_room_import_requires_new_revision": True,
    "cross_room_result_never_satisfies_destination_denominator": True,
    "common_sparring_is_only_room_integration_point": True,
}

_STAGE_SEPARATION: Final = {
    "training_rooms": {
        "stage": "WORN_HISTORICAL_TRAIN",
        "room_local_search_only": True,
        "holdout_exam": False,
        "forward_arena": False,
    },
    "common_sparring": {
        "stage": "WORN_HISTORICAL_COMMON_SPARRING",
        "accepts_only_terminal_validated_room_candidates": True,
        "requires_new_fixed_denominator_reexecution": True,
        "holdout_exam": False,
        "forward_arena": False,
    },
    "holdout_exam": {
        "stage": "SEALED_HOLDOUT_EXAM",
        "separate_from_training_rooms": True,
        "separate_from_common_sparring": True,
        "train_or_sparring_result_reuse_allowed": False,
        "one_open_only_under_separate_preregistration": True,
    },
    "forward_arena": {
        "stage": "FIXED_VERSION_FORWARD_ARENA",
        "separate_from_training_rooms": True,
        "separate_from_common_sparring": True,
        "separate_from_holdout_exam": True,
        "historical_result_counts_as_forward": False,
    },
}


class DojoTrainingRoomError(ValueError):
    """A room, receipt, or sparring handoff violates isolation."""


@dataclass(frozen=True)
class TrainingRoomTaxonomy:
    room_id: str
    strategy_family: str
    thesis: str
    input_class: str
    decision_context_policy: str

    def as_dict(self) -> dict[str, str]:
        return {
            "room_id": self.room_id,
            "strategy_family": self.strategy_family,
            "thesis": self.thesis,
            "input_class": self.input_class,
            "decision_context_policy": self.decision_context_policy,
        }


INITIAL_ROOM_TAXONOMY: Final = (
    TrainingRoomTaxonomy(
        "room-g2-01",
        "spike_fade",
        "G2 spike-fade control; preserve the known baseline without mixing another thesis.",
        "CAUSAL_PRICE_STRUCTURE",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-g2-02",
        "burst",
        "Trade bounded directional burst continuation as one isolated G2 thesis.",
        "CAUSAL_PRICE_STRUCTURE",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-g2-03",
        "pullback_limit",
        "Enter causal pullbacks passively under the fixed G2 vehicle contract.",
        "CAUSAL_PRICE_STRUCTURE",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-g2-04",
        "prev_day_extreme_fade",
        "Fade prior-day extremes using only already closed daily structure.",
        "CAUSAL_CLOSED_DAILY_AND_EXECUTION_PRICE",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-g2-05",
        "round_number_fade",
        "Fade round-number rejection as one independent G2 research thesis.",
        "CAUSAL_PRICE_STRUCTURE",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-g2-06",
        "mean_revert_24h",
        "Test 24-hour mean reversion without borrowing another family's result.",
        "CAUSAL_ROLLING_24H_PRICE",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-01",
        "asia_sweep_reclaim_be",
        "Fade a London-session sweep only after price closes back inside the completed Asia range.",
        "CAUSAL_M5_SESSION_STRUCTURE",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-02",
        "h1_donchian_break_atr_trailing",
        "Follow a completed-H1 Donchian break with a separately tested ATR trailing exit.",
        "CAUSAL_M5_DERIVED_CLOSED_H1",
        "DETERMINISTIC_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-03",
        "g8_relative_strength_risk_budget",
        "Allocate four shared-risk seats to strongest-versus-weakest G8 pairs from closed H1 strength.",
        "CAUSAL_MULTI_PAIR_CLOSED_H1_AND_PORTFOLIO_STATE",
        "DETERMINISTIC_PORTFOLIO_WORKER_NO_MODEL_CONTEXT",
    ),
    TrainingRoomTaxonomy(
        "room-ai-01",
        "ai_discretionary_exit_capital_recycle",
        "Use AI only for exit and capital-recycle judgement under one decision per fresh context.",
        "FRESH_CAPABILITY_ISOLATED_FULL_CONTEXT_DECISION",
        "ONE_DECISION_ONE_FRESH_CONTEXT_NO_CROSS_DECISION_HISTORY",
    ),
)

ROOM_TAXONOMY_README: Final = """DOJO training-room taxonomy V1

One room owns exactly one strategy or research thesis.  G2 is six separate
rooms: room-g2-01 spike-fade control, room-g2-02 burst, room-g2-03 pullback,
room-g2-04 previous-day extreme fade, room-g2-05 round-number fade, and
room-g2-06 24-hour mean reversion.  New rooms are room-01 Asia sweep/reclaim,
room-02 H1 Donchian with ATR trailing, room-03 G8 relative-strength allocator,
and room-ai-01 AI discretionary exit/capital recycle.  room-ai-01 has one decision
per fresh context.  No room mixes theses.  Common sparring is the only
integration point and may accept at most one terminal validated candidate per
room.  Holdout examination and the fixed-version forward arena are separate
stages and never inherit TRAIN authority.
"""

_TAXONOMY_BY_ID: Final = {room.room_id: room for room in INITIAL_ROOM_TAXONOMY}
_TAXONOMY_ROOM_IDS: Final = tuple(room.room_id for room in INITIAL_ROOM_TAXONOMY)


def canonical_room_sha256(value: Any) -> str:
    """Return strict canonical JSON SHA-256."""

    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoTrainingRoomError("value is not strict canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def build_training_room_registry(
    *,
    registry_id: str,
    registry_revision: str,
    shared_bindings: Mapping[str, Mapping[str, Any]],
    room_controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the exact ten-room, one-thesis-per-room V1 registry."""

    normalized_shared = _normalize_shared_bindings(shared_bindings)
    if set(room_controls) != set(_TAXONOMY_ROOM_IDS):
        raise DojoTrainingRoomError(
            "room controls must cover the exact initial room taxonomy"
        )
    rooms: list[dict[str, Any]] = []
    lineage_ids: set[str] = set()
    namespaces: set[str] = set()
    denominator_ids: set[str] = set()
    for taxonomy in INITIAL_ROOM_TAXONOMY:
        control = _exact_mapping(
            room_controls[taxonomy.room_id],
            {
                "trainer_lineage_id",
                "search_budget",
                "artifact_namespace",
                "fixed_train_denominator",
            },
            field=f"room control {taxonomy.room_id}",
        )
        lineage_id = _room_scoped_identifier(
            control["trainer_lineage_id"],
            taxonomy.room_id,
            field="trainer_lineage_id",
        )
        namespace = _room_namespace(control["artifact_namespace"], taxonomy.room_id)
        budget = _normalize_search_budget(control["search_budget"])
        denominator = _build_room_denominator(
            taxonomy.room_id, control["fixed_train_denominator"]
        )
        if lineage_id in lineage_ids:
            raise DojoTrainingRoomError("trainer lineage is shared across rooms")
        if namespace in namespaces:
            raise DojoTrainingRoomError("artifact namespace is shared across rooms")
        if denominator["denominator_id"] in denominator_ids:
            raise DojoTrainingRoomError("TRAIN denominator id is shared across rooms")
        lineage_ids.add(lineage_id)
        namespaces.add(namespace)
        denominator_ids.add(denominator["denominator_id"])
        room_body = {
            **taxonomy.as_dict(),
            "trainer_lineage": {
                "lineage_id": lineage_id,
                "room_id": taxonomy.room_id,
                "cross_room_lineage_reuse_allowed": False,
            },
            "search_budget": budget,
            "artifact_namespace": namespace,
            "fixed_train_denominator": denominator,
            "shared_bindings_sha256": normalized_shared["shared_bindings_sha256"],
            "room_isolation": {
                "one_strategy_or_research_thesis": True,
                "mixed_strategy_family_allowed": False,
                "result_namespace_is_room_local": True,
                "parameter_namespace_is_room_local": True,
                "common_sparring_is_only_integration_point": True,
            },
        }
        rooms.append({**room_body, "room_sha256": canonical_room_sha256(room_body)})

    body = {
        "contract": ROOM_REGISTRY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "registry_id": _identifier(registry_id, field="registry_id"),
        "registry_revision": _identifier(registry_revision, field="registry_revision"),
        "taxonomy_revision": TAXONOMY_REVISION,
        "taxonomy_readme": ROOM_TAXONOMY_README,
        "taxonomy_readme_sha256": canonical_room_sha256(ROOM_TAXONOMY_README),
        "room_count": len(rooms),
        "shared_bindings": normalized_shared,
        "sharing_policy": _strict_clone(_SHARING_POLICY),
        "rooms": rooms,
        "stage_separation": _strict_clone(_STAGE_SEPARATION),
        "authority": _strict_clone(_AUTHORITY),
    }
    return {**body, "registry_sha256": canonical_room_sha256(body)}


def validate_training_room_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a registry from its declared inputs and require exact equality."""

    row = _strict_clone(value)
    _exact_mapping(
        row,
        {
            "contract",
            "schema_version",
            "registry_id",
            "registry_revision",
            "taxonomy_revision",
            "taxonomy_readme",
            "taxonomy_readme_sha256",
            "room_count",
            "shared_bindings",
            "sharing_policy",
            "rooms",
            "stage_separation",
            "authority",
            "registry_sha256",
        },
        field="room registry",
    )
    if row["contract"] != ROOM_REGISTRY_CONTRACT or row["schema_version"] != 1:
        raise DojoTrainingRoomError("room registry contract/version is invalid")
    rooms = _sequence(row["rooms"], field="rooms")
    controls: dict[str, dict[str, Any]] = {}
    for item in rooms:
        room = _mapping(item, field="room")
        room_id = _identifier(room.get("room_id"), field="room_id")
        if room_id in controls:
            raise DojoTrainingRoomError("room registry contains a duplicate room")
        denominator = _mapping(
            room.get("fixed_train_denominator"), field="fixed_train_denominator"
        )
        controls[room_id] = {
            "trainer_lineage_id": _mapping(
                room.get("trainer_lineage"), field="trainer_lineage"
            ).get("lineage_id"),
            "search_budget": room.get("search_budget"),
            "artifact_namespace": room.get("artifact_namespace"),
            "fixed_train_denominator": {
                key: denominator.get(key) for key in _DENOMINATOR_INPUT_KEYS
            },
        }
    expected = build_training_room_registry(
        registry_id=row["registry_id"],
        registry_revision=row["registry_revision"],
        shared_bindings={
            name: {
                "binding_id": row["shared_bindings"][name]["binding_id"],
                "artifact_sha256": row["shared_bindings"][name]["artifact_sha256"],
            }
            for name in _SHARED_BINDING_NAMES
        },
        room_controls=controls,
    )
    if row != expected:
        raise DojoTrainingRoomError("room registry differs from canonical V1")
    return row


def build_training_room_receipt(
    *,
    registry: Mapping[str, Any],
    room_id: str,
    candidate: Mapping[str, Any],
    terminal_status: str,
    candidate_gate_passed: bool,
    observed_coordinate_count: int,
    failed_coordinate_count: int,
    budget_consumed: Mapping[str, Any],
    artifact_relative_path: str,
    research_queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one room-local terminal TRAIN receipt.

    Passing ``research_queue`` opts into the queue-bound V2 contract.  V2 is
    required for queue candidates admitted to a queue-bound common sparring
    handoff; legacy V1 receipts remain readable but cannot satisfy that gate.
    """

    normalized_registry = validate_training_room_registry(registry)
    room = _registry_room(normalized_registry, room_id)
    if terminal_status not in {"TERMINAL_VALIDATED", "TERMINAL_REJECTED"}:
        raise DojoTrainingRoomError("room receipt status must be terminal")
    if candidate_gate_passed.__class__ is not bool:
        raise DojoTrainingRoomError("candidate_gate_passed must be a boolean")
    expected_count = room["fixed_train_denominator"]["expected_coordinate_count"]
    observed_count = _integer(
        observed_coordinate_count,
        field="observed_coordinate_count",
        minimum=0,
    )
    failed_count = _integer(
        failed_coordinate_count,
        field="failed_coordinate_count",
        minimum=0,
    )
    if observed_count != expected_count or failed_count > observed_count:
        raise DojoTrainingRoomError(
            "room terminal receipt must preserve the complete fixed denominator"
        )
    terminal_validated = candidate_gate_passed and failed_count == 0
    expected_status = (
        "TERMINAL_VALIDATED" if terminal_validated else "TERMINAL_REJECTED"
    )
    if terminal_status != expected_status:
        raise DojoTrainingRoomError(
            "room terminal status contradicts gate/failure evidence"
        )
    normalized_candidate = _normalize_candidate(
        candidate,
        room_id=room_id,
        registry=normalized_registry,
        room_denominator_sha256=room["fixed_train_denominator"][
            "room_denominator_sha256"
        ],
    )
    queue_binding: dict[str, Any] | None = None
    if research_queue is not None:
        from quant_rabbit.dojo_strategy_research_queue import (
            DojoStrategyResearchQueueError,
            resolve_queue_room_binding,
        )

        try:
            canonical_binding = resolve_queue_room_binding(
                research_queue, dojo_room_id=room_id
            )
        except DojoStrategyResearchQueueError as exc:
            raise DojoTrainingRoomError(
                "queue-bound receipt requires one canonical room candidate"
            ) from exc
        expected_local_candidate_id = (
            f"{room_id}:{canonical_binding['canonical_candidate_id']}"
        )
        if normalized_candidate["candidate_id"] != expected_local_candidate_id:
            raise DojoTrainingRoomError(
                "room candidate differs from the canonical queue candidate"
            )
        if room["strategy_family"] != canonical_binding["canonical_family"]:
            raise DojoTrainingRoomError(
                "room strategy family differs from the canonical queue family"
            )
        queue_binding = {
            **canonical_binding,
            "room_local_candidate_id": normalized_candidate["candidate_id"],
            "registry_strategy_family": room["strategy_family"],
        }
    consumed = _normalize_budget_consumed(budget_consumed, room["search_budget"])
    artifact_path = _artifact_path_in_namespace(
        artifact_relative_path, room["artifact_namespace"]
    )
    body = {
        "contract": (
            QUEUE_BOUND_ROOM_RECEIPT_CONTRACT
            if queue_binding is not None
            else ROOM_RECEIPT_CONTRACT
        ),
        "schema_version": (
            QUEUE_BOUND_SCHEMA_VERSION if queue_binding is not None else SCHEMA_VERSION
        ),
        "registry_id": normalized_registry["registry_id"],
        "registry_revision": normalized_registry["registry_revision"],
        "registry_sha256": normalized_registry["registry_sha256"],
        "room_id": room_id,
        "room_sha256": room["room_sha256"],
        "trainer_lineage_id": room["trainer_lineage"]["lineage_id"],
        "artifact_namespace": room["artifact_namespace"],
        "artifact_relative_path": artifact_path,
        "shared_bindings_sha256": normalized_registry["shared_bindings"][
            "shared_bindings_sha256"
        ],
        "fixed_train_denominator": _strict_clone(room["fixed_train_denominator"]),
        "terminal": {
            "status": terminal_status,
            "expected_coordinate_count": expected_count,
            "observed_coordinate_count": observed_count,
            "failed_coordinate_count": failed_count,
            "fixed_denominator_complete": True,
            "candidate_gate_passed": candidate_gate_passed,
            "eligible_for_common_sparring": terminal_validated,
        },
        "candidate": normalized_candidate,
        **(
            {"research_queue_binding": queue_binding}
            if queue_binding is not None
            else {}
        ),
        "budget_consumed": consumed,
        "stage": "WORN_HISTORICAL_TRAIN_ONLY",
        "holdout_exam_result": False,
        "forward_arena_result": False,
        "authority": _strict_clone(_AUTHORITY),
    }
    return {**body, "room_receipt_sha256": canonical_room_sha256(body)}


def validate_training_room_receipt(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    research_queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild one room receipt and reject every schema or binding drift."""

    normalized_registry = validate_training_room_registry(registry)
    row = _strict_clone(value)
    queue_bound = row.get("contract") == QUEUE_BOUND_ROOM_RECEIPT_CONTRACT
    if queue_bound != (research_queue is not None):
        raise DojoTrainingRoomError(
            "queue-bound V2 receipt and research queue must be supplied together"
        )
    terminal = _mapping(row.get("terminal"), field="terminal")
    candidate = _mapping(row.get("candidate"), field="candidate")
    cross_parent = candidate.get("cross_room_parent")
    candidate_input = {
        "candidate_id": candidate.get("candidate_id"),
        "hypothesis_id": candidate.get("hypothesis_id"),
        "revision_id": candidate.get("revision_id"),
        "parameters_sha256": candidate.get("parameters_sha256"),
        "terminal_result_sha256": candidate.get("terminal_result_sha256"),
        "cross_room_parent": cross_parent,
    }
    expected = build_training_room_receipt(
        registry=normalized_registry,
        room_id=row.get("room_id"),
        candidate=candidate_input,
        terminal_status=terminal.get("status"),
        candidate_gate_passed=terminal.get("candidate_gate_passed"),
        observed_coordinate_count=terminal.get("observed_coordinate_count"),
        failed_coordinate_count=terminal.get("failed_coordinate_count"),
        budget_consumed=row.get("budget_consumed"),
        artifact_relative_path=row.get("artifact_relative_path"),
        research_queue=research_queue,
    )
    if row != expected:
        raise DojoTrainingRoomError("room receipt differs from canonical V1")
    return row


def build_common_sparring_handoff(
    *,
    registry: Mapping[str, Any],
    handoff_id: str,
    handoff_revision: str,
    room_receipts: Sequence[Mapping[str, Any]],
    fixed_denominator: Mapping[str, Any],
    research_queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preregister common sparring from at most one validated candidate/room."""

    normalized_registry = validate_training_room_registry(registry)
    if isinstance(room_receipts, (str, bytes, bytearray)) or not room_receipts:
        raise DojoTrainingRoomError(
            "common sparring requires at least one room receipt"
        )
    queue_bindings: dict[str, dict[str, Any]] = {}
    if research_queue is not None:
        from quant_rabbit.dojo_strategy_research_queue import validate_research_queue

        queue = validate_research_queue(research_queue)
        queue_bindings = {
            candidate["dojo_room_id"]: {
                "canonical_candidate_id": candidate["candidate_id"],
                "canonical_candidate_sha256": candidate["candidate_sha256"],
                "canonical_family": candidate["family"],
            }
            for candidate in queue["candidates"]
        }
    receipts: list[dict[str, Any]] = []
    for receipt_value in room_receipts:
        receipt_row = _mapping(receipt_value, field="room receipt")
        receipt_room_id = receipt_row.get("room_id")
        receipt_queue = research_queue if receipt_room_id in queue_bindings else None
        receipts.append(
            validate_training_room_receipt(
                receipt_row,
                registry=normalized_registry,
                research_queue=receipt_queue,
            )
        )
    by_room: dict[str, dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    for receipt in receipts:
        room_id = receipt["room_id"]
        if room_id in by_room:
            raise DojoTrainingRoomError(
                "common sparring accepts at most one candidate per room"
            )
        if (
            receipt["terminal"]["status"] != "TERMINAL_VALIDATED"
            or receipt["terminal"]["eligible_for_common_sparring"] is not True
        ):
            raise DojoTrainingRoomError(
                "common sparring accepts only terminal validated candidates"
            )
        candidate_id = receipt["candidate"]["candidate_id"]
        if candidate_id in candidate_ids:
            raise DojoTrainingRoomError(
                "common sparring candidate identity is duplicated"
            )
        candidate_ids.add(candidate_id)
        by_room[room_id] = receipt
    denominator = _build_common_denominator(fixed_denominator)
    candidates: list[dict[str, Any]] = []
    for room_id in _TAXONOMY_ROOM_IDS:
        receipt = by_room.get(room_id)
        if receipt is None:
            continue
        candidate = receipt["candidate"]
        queue_binding = receipt.get("research_queue_binding")
        candidates.append(
            {
                "room_id": room_id,
                "room_receipt_sha256": receipt["room_receipt_sha256"],
                "candidate_id": candidate["candidate_id"],
                "hypothesis_id": candidate["hypothesis_id"],
                "revision_id": candidate["revision_id"],
                "parameters_sha256": candidate["parameters_sha256"],
                "room_train_result_sha256": candidate["terminal_result_sha256"],
                "room_train_result_is_selection_provenance_only": True,
                "common_sparring_denominator_reexecution_required": True,
                "common_sparring_result_sha256": None,
                **(
                    {
                        "research_queue_binding": _strict_clone(queue_binding),
                    }
                    if queue_binding is not None
                    else {}
                ),
            }
        )
    body = {
        "contract": (
            QUEUE_BOUND_COMMON_SPARRING_HANDOFF_CONTRACT
            if research_queue is not None
            else COMMON_SPARRING_HANDOFF_CONTRACT
        ),
        "schema_version": (
            QUEUE_BOUND_SCHEMA_VERSION if research_queue is not None else SCHEMA_VERSION
        ),
        "handoff_id": _identifier(handoff_id, field="handoff_id"),
        "handoff_revision": _identifier(handoff_revision, field="handoff_revision"),
        "registry_id": normalized_registry["registry_id"],
        "registry_revision": normalized_registry["registry_revision"],
        "registry_sha256": normalized_registry["registry_sha256"],
        "shared_bindings": _strict_clone(normalized_registry["shared_bindings"]),
        "fixed_denominator": denominator,
        "allocator": {
            "shared_capital": True,
            "simultaneous_slots": COMMON_SPARRING_SLOTS,
            "maximum_candidates_per_room": 1,
            "candidate_count_may_exceed_simultaneous_slots": True,
            "allocator_changes_require_new_handoff_revision": True,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        **(
            {
                "research_queue_binding": {
                    "queue_contract": queue["contract"],
                    "queue_id": queue["queue_id"],
                    "queue_artifact_sha256": queue["artifact_sha256"],
                    "bound_room_ids": sorted(queue_bindings),
                }
            }
            if research_queue is not None
            else {}
        ),
        "status": "PREREGISTERED_NOT_EXECUTED",
        "room_train_results_reused_as_sparring_results": False,
        "parameters_mutable_in_common_sparring": False,
        "holdout_exam_result": False,
        "forward_arena_result": False,
        "stage_separation": _strict_clone(_STAGE_SEPARATION),
        "authority": _strict_clone(_AUTHORITY),
    }
    return {**body, "handoff_sha256": canonical_room_sha256(body)}


def validate_common_sparring_handoff(
    value: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    room_receipts: Sequence[Mapping[str, Any]],
    research_queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a handoff from its receipts and fixed denominator core."""

    row = _strict_clone(value)
    denominator = _mapping(row.get("fixed_denominator"), field="fixed_denominator")
    expected = build_common_sparring_handoff(
        registry=registry,
        handoff_id=row.get("handoff_id"),
        handoff_revision=row.get("handoff_revision"),
        room_receipts=room_receipts,
        fixed_denominator={
            key: denominator.get(key) for key in _DENOMINATOR_INPUT_KEYS
        },
        research_queue=research_queue,
    )
    if row != expected:
        raise DojoTrainingRoomError("common sparring handoff differs from canonical V1")
    return row


def _normalize_shared_bindings(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    row = _exact_mapping(value, set(_SHARED_BINDING_NAMES), field="shared_bindings")
    normalized: dict[str, Any] = {}
    for name in _SHARED_BINDING_NAMES:
        binding = _exact_mapping(
            row[name], {"binding_id", "artifact_sha256"}, field=name
        )
        normalized[name] = {
            "binding_id": _identifier(binding["binding_id"], field=f"{name}.id"),
            "artifact_sha256": _sha256(
                binding["artifact_sha256"], field=f"{name}.artifact_sha256"
            ),
        }
    normalized["only_these_components_are_shareable"] = True
    normalized["shared_bindings_sha256"] = canonical_room_sha256(normalized)
    return normalized


def _normalize_search_budget(value: Any) -> dict[str, int]:
    row = _exact_mapping(value, _SEARCH_BUDGET_KEYS, field="search_budget")
    normalized = {
        key: _integer(row[key], field=f"search_budget.{key}", minimum=0)
        for key in sorted(_SEARCH_BUDGET_KEYS)
    }
    for key in ("max_attempts", "max_hypotheses", "max_parameter_revisions"):
        if normalized[key] == 0:
            raise DojoTrainingRoomError(f"search_budget.{key} must be positive")
    return normalized


def _normalize_budget_consumed(
    value: Any, maximum: Mapping[str, int]
) -> dict[str, int]:
    row = _exact_mapping(value, _BUDGET_CONSUMED_KEYS, field="budget_consumed")
    relation = {
        "attempts": "max_attempts",
        "hypotheses": "max_hypotheses",
        "parameter_revisions": "max_parameter_revisions",
        "model_calls": "max_model_calls",
    }
    normalized: dict[str, int] = {}
    for key, limit_key in relation.items():
        item = _integer(row[key], field=f"budget_consumed.{key}", minimum=0)
        if item > maximum[limit_key]:
            raise DojoTrainingRoomError(f"budget_consumed.{key} exceeds room budget")
        normalized[key] = item
    return normalized


def _build_room_denominator(room_id: str, value: Any) -> dict[str, Any]:
    row = _exact_mapping(
        value, _DENOMINATOR_INPUT_KEYS, field="fixed_train_denominator"
    )
    body = {
        "contract": ROOM_DENOMINATOR_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "room_id": room_id,
        "denominator_id": _room_scoped_identifier(
            row["denominator_id"], room_id, field="denominator_id"
        ),
        "coordinate_set_sha256": _sha256(
            row["coordinate_set_sha256"], field="coordinate_set_sha256"
        ),
        "expected_coordinate_count": _integer(
            row["expected_coordinate_count"],
            field="expected_coordinate_count",
            minimum=1,
        ),
        "fixed_before_room_search": True,
        "immutable_within_room_revision": True,
        "room_local_result_denominator": True,
        "train_only": True,
    }
    return {**body, "room_denominator_sha256": canonical_room_sha256(body)}


def _build_common_denominator(value: Any) -> dict[str, Any]:
    row = _exact_mapping(
        value, _DENOMINATOR_INPUT_KEYS, field="common fixed_denominator"
    )
    body = {
        "contract": COMMON_SPARRING_DENOMINATOR_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "denominator_id": _identifier(
            row["denominator_id"], field="common denominator_id"
        ),
        "coordinate_set_sha256": _sha256(
            row["coordinate_set_sha256"], field="coordinate_set_sha256"
        ),
        "expected_coordinate_count": _integer(
            row["expected_coordinate_count"],
            field="expected_coordinate_count",
            minimum=1,
        ),
        "fixed_before_candidate_handoff": True,
        "shared_capital": True,
        "simultaneous_slots": COMMON_SPARRING_SLOTS,
        "room_train_result_reuse_allowed": False,
    }
    return {**body, "denominator_sha256": canonical_room_sha256(body)}


def _normalize_candidate(
    value: Any,
    *,
    room_id: str,
    registry: Mapping[str, Any],
    room_denominator_sha256: str,
) -> dict[str, Any]:
    row = _exact_mapping(value, _CANDIDATE_INPUT_KEYS, field="candidate")
    candidate_id = _room_scoped_identifier(
        row["candidate_id"], room_id, field="candidate_id"
    )
    hypothesis_id = _room_scoped_identifier(
        row["hypothesis_id"], room_id, field="hypothesis_id"
    )
    revision_id = _room_scoped_identifier(
        row["revision_id"], room_id, field="revision_id"
    )
    cross_parent = row["cross_room_parent"]
    if cross_parent is None:
        normalized_parent = None
        copy_policy = {
            "kind": "ROOM_NATIVE_HYPOTHESIS",
            "new_hypothesis_declared": True,
            "new_revision_declared": True,
            "source_result_reused_as_room_result": False,
            "room_denominator_reexecution_required": True,
        }
    else:
        parent = _exact_mapping(
            cross_parent, _CROSS_ROOM_PARENT_KEYS, field="cross_room_parent"
        )
        source_room_id = _identifier(parent["source_room_id"], field="source_room_id")
        if source_room_id == room_id or source_room_id not in _TAXONOMY_BY_ID:
            raise DojoTrainingRoomError(
                "cross-room parent must name another registered room"
            )
        _registry_room(registry, source_room_id)
        normalized_parent = {
            "source_room_id": source_room_id,
            "source_candidate_id": _room_scoped_identifier(
                parent["source_candidate_id"],
                source_room_id,
                field="source_candidate_id",
            ),
            "source_hypothesis_id": _room_scoped_identifier(
                parent["source_hypothesis_id"],
                source_room_id,
                field="source_hypothesis_id",
            ),
            "source_revision_id": _room_scoped_identifier(
                parent["source_revision_id"],
                source_room_id,
                field="source_revision_id",
            ),
            "source_parameters_sha256": _sha256(
                parent["source_parameters_sha256"],
                field="source_parameters_sha256",
            ),
            "source_result_sha256": _sha256(
                parent["source_result_sha256"], field="source_result_sha256"
            ),
        }
        copy_policy = {
            "kind": "CROSS_ROOM_NEW_HYPOTHESIS_AND_REVISION",
            "new_hypothesis_declared": True,
            "new_revision_declared": True,
            "source_result_reused_as_room_result": False,
            "room_denominator_reexecution_required": True,
        }
    return {
        "candidate_id": candidate_id,
        "hypothesis_id": hypothesis_id,
        "revision_id": revision_id,
        "parameters_sha256": _sha256(
            row["parameters_sha256"], field="parameters_sha256"
        ),
        "terminal_result_sha256": _sha256(
            row["terminal_result_sha256"], field="terminal_result_sha256"
        ),
        "room_id": room_id,
        "room_denominator_sha256": _sha256(
            room_denominator_sha256, field="room_denominator_sha256"
        ),
        "room_denominator_reexecuted": True,
        "cross_room_parent": normalized_parent,
        "cross_room_copy_policy": copy_policy,
    }


def _registry_room(registry: Mapping[str, Any], room_id: Any) -> dict[str, Any]:
    normalized_id = _identifier(room_id, field="room_id")
    matches = [room for room in registry["rooms"] if room["room_id"] == normalized_id]
    if len(matches) != 1:
        raise DojoTrainingRoomError("room is absent from the registry")
    return matches[0]


def _room_namespace(value: Any, room_id: str) -> str:
    text = _identifier(value, field="artifact_namespace")
    expected = f"{ROOM_ARTIFACT_ROOT}/{room_id}"
    if text != expected:
        raise DojoTrainingRoomError(
            "artifact namespace must be the exact room-owned namespace"
        )
    return text


def _artifact_path_in_namespace(value: Any, namespace: str) -> str:
    text = _identifier(value, field="artifact_relative_path")
    path = PurePosixPath(text)
    namespace_path = PurePosixPath(namespace)
    if path.is_absolute() or ".." in path.parts or path == namespace_path:
        raise DojoTrainingRoomError("artifact path is not a room-owned file path")
    try:
        path.relative_to(namespace_path)
    except ValueError as exc:
        raise DojoTrainingRoomError("artifact path escapes the room namespace") from exc
    return path.as_posix()


def _room_scoped_identifier(value: Any, room_id: str, *, field: str) -> str:
    text = _identifier(value, field=field)
    if not text.startswith(f"{room_id}:"):
        raise DojoTrainingRoomError(f"{field} must be scoped to {room_id}")
    return text


def _strict_clone(value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise DojoTrainingRoomError("contract value must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise DojoTrainingRoomError("contract keys must be strings")
    try:
        return json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise DojoTrainingRoomError("contract must contain strict JSON values") from exc


def _exact_mapping(
    value: Any, keys: set[str] | frozenset[str], *, field: str
) -> dict[str, Any]:
    row = _mapping(value, field=field)
    if set(row) != set(keys):
        raise DojoTrainingRoomError(f"{field} schema is not exact")
    return row


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise DojoTrainingRoomError(f"{field} must be an object")
    return dict(value)


def _sequence(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DojoTrainingRoomError(f"{field} must be an array")
    return list(value)


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise DojoTrainingRoomError(f"{field} is not a bounded identifier")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DojoTrainingRoomError(f"{field} is not a SHA-256 digest")
    return value


def _integer(value: Any, *, field: str, minimum: int) -> int:
    if value.__class__ is not int or value < minimum:
        raise DojoTrainingRoomError(f"{field} must be an integer >= {minimum}")
    return value


__all__ = [
    "COMMON_SPARRING_DENOMINATOR_CONTRACT",
    "COMMON_SPARRING_HANDOFF_CONTRACT",
    "COMMON_SPARRING_SLOTS",
    "INITIAL_ROOM_TAXONOMY",
    "ROOM_ARTIFACT_ROOT",
    "ROOM_DENOMINATOR_CONTRACT",
    "ROOM_RECEIPT_CONTRACT",
    "ROOM_REGISTRY_CONTRACT",
    "ROOM_TAXONOMY_README",
    "SCHEMA_VERSION",
    "TAXONOMY_REVISION",
    "DojoTrainingRoomError",
    "TrainingRoomTaxonomy",
    "build_common_sparring_handoff",
    "build_training_room_receipt",
    "build_training_room_registry",
    "canonical_room_sha256",
    "validate_common_sparring_handoff",
    "validate_training_room_receipt",
    "validate_training_room_registry",
]
