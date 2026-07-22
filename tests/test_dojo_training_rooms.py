from __future__ import annotations

import copy

import pytest

from quant_rabbit.dojo_strategy_research_queue import build_research_queue
from quant_rabbit.dojo_training_rooms import (
    COMMON_SPARRING_SLOTS,
    INITIAL_ROOM_TAXONOMY,
    QUEUE_BOUND_COMMON_SPARRING_HANDOFF_CONTRACT,
    QUEUE_BOUND_ROOM_RECEIPT_CONTRACT,
    ROOM_TAXONOMY_README,
    ROOM_TAXONOMY_V2,
    ROOM_TAXONOMY_V2_README,
    DojoTrainingRoomError,
    build_common_sparring_handoff,
    build_training_room_receipt,
    build_training_room_registry,
    build_training_room_registry_v2,
    canonical_room_sha256,
    validate_common_sparring_handoff,
    validate_training_room_receipt,
    validate_training_room_registry,
)


def _sha(label: str) -> str:
    return canonical_room_sha256({"fixture": label})


@pytest.fixture(scope="module")
def shared_bindings() -> dict:
    return {
        name: {
            "binding_id": f"shared:{name}:v1",
            "artifact_sha256": _sha(name),
        }
        for name in ("source", "evaluator", "cost_policy", "risk_policy")
    }


@pytest.fixture(scope="module")
def room_controls() -> dict:
    return {
        taxonomy.room_id: {
            "trainer_lineage_id": f"{taxonomy.room_id}:lineage-v1",
            "search_budget": {
                "max_attempts": 3,
                "max_hypotheses": 4,
                "max_parameter_revisions": 8,
                "max_model_calls": 12 if taxonomy.room_id == "room-ai-01" else 0,
            },
            "artifact_namespace": (f"research/dojo/training_rooms/{taxonomy.room_id}"),
            "fixed_train_denominator": {
                "denominator_id": f"{taxonomy.room_id}:train-denominator-v1",
                "coordinate_set_sha256": _sha(f"coordinates:{taxonomy.room_id}"),
                "expected_coordinate_count": 96,
            },
        }
        for taxonomy in INITIAL_ROOM_TAXONOMY
    }


@pytest.fixture(scope="module")
def registry(shared_bindings: dict, room_controls: dict) -> dict:
    return build_training_room_registry(
        registry_id="dojo-room-registry-v1",
        registry_revision="revision-1",
        shared_bindings=shared_bindings,
        room_controls=room_controls,
    )


@pytest.fixture(scope="module")
def v2_room_controls() -> dict:
    return {
        taxonomy.room_id: {
            "trainer_lineage_id": f"{taxonomy.room_id}:lineage-v2",
            "search_budget": {
                "max_attempts": 3,
                "max_hypotheses": 4,
                "max_parameter_revisions": 8,
                "max_model_calls": 12 if taxonomy.room_id == "room-ai-01" else 0,
            },
            "artifact_namespace": f"research/dojo/training_rooms/{taxonomy.room_id}",
            "fixed_train_denominator": {
                "denominator_id": f"{taxonomy.room_id}:train-denominator-v2",
                "coordinate_set_sha256": _sha(f"coordinates-v2:{taxonomy.room_id}"),
                "expected_coordinate_count": 96,
            },
        }
        for taxonomy in ROOM_TAXONOMY_V2
    }


@pytest.fixture(scope="module")
def v2_registry(shared_bindings: dict, v2_room_controls: dict) -> dict:
    return build_training_room_registry_v2(
        registry_id="dojo-room-registry-v2",
        registry_revision="revision-2",
        shared_bindings=shared_bindings,
        room_controls=v2_room_controls,
    )


def _candidate(room_id: str, *, cross_room_parent=None) -> dict:
    return {
        "candidate_id": f"{room_id}:candidate-1",
        "hypothesis_id": f"{room_id}:hypothesis-1",
        "revision_id": f"{room_id}:revision-1",
        "parameters_sha256": _sha(f"parameters:{room_id}"),
        "terminal_result_sha256": _sha(f"result:{room_id}"),
        "cross_room_parent": cross_room_parent,
    }


def _queued_candidate(room_id: str) -> dict:
    queue = build_research_queue()
    queue_candidate = next(
        row for row in queue["candidates"] if row["dojo_room_id"] == room_id
    )
    candidate = _candidate(room_id)
    candidate["candidate_id"] = f"{room_id}:{queue_candidate['candidate_id']}"
    return candidate


def _receipt(
    registry: dict,
    room_id: str,
    *,
    validated: bool = True,
    candidate: dict | None = None,
) -> dict:
    return build_training_room_receipt(
        registry=registry,
        room_id=room_id,
        candidate=candidate or _candidate(room_id),
        terminal_status=("TERMINAL_VALIDATED" if validated else "TERMINAL_REJECTED"),
        candidate_gate_passed=validated,
        observed_coordinate_count=96,
        failed_coordinate_count=0,
        budget_consumed={
            "attempts": 1,
            "hypotheses": 1,
            "parameter_revisions": 1,
            "model_calls": 1 if room_id == "room-ai-01" else 0,
        },
        artifact_relative_path=(
            f"research/dojo/training_rooms/{room_id}/terminal.json"
        ),
    )


def _common_denominator() -> dict:
    return {
        "denominator_id": "common-sparring:denominator-v1",
        "coordinate_set_sha256": _sha("common-sparring-coordinates"),
        "expected_coordinate_count": 960,
    }


def test_taxonomy_and_readme_fix_one_thesis_per_room() -> None:
    expected = {
        "room-g2-01": "spike_fade",
        "room-g2-02": "burst",
        "room-g2-03": "pullback_limit",
        "room-g2-04": "prev_day_extreme_fade",
        "room-g2-05": "round_number_fade",
        "room-g2-06": "mean_revert_24h",
        "room-01": "asia_sweep_reclaim_be",
        "room-02": "h1_donchian_break_atr_trailing",
        "room-03": "g8_relative_strength_risk_budget",
        "room-ai-01": "ai_discretionary_exit_capital_recycle",
    }
    assert {
        room.room_id: room.strategy_family for room in INITIAL_ROOM_TAXONOMY
    } == expected
    assert len({room.thesis for room in INITIAL_ROOM_TAXONOMY}) == len(expected)
    assert all(room_id in ROOM_TAXONOMY_README for room_id in expected)
    ai = next(room for room in INITIAL_ROOM_TAXONOMY if room.room_id == "room-ai-01")
    assert ai.decision_context_policy == (
        "ONE_DECISION_ONE_FRESH_CONTEXT_NO_CROSS_DECISION_HISTORY"
    )
    assert "room-00" not in ROOM_TAXONOMY_README
    portfolio = next(
        room for room in INITIAL_ROOM_TAXONOMY if room.room_id == "room-03"
    )
    assert "H1" in portfolio.thesis
    assert portfolio.input_class == ("CAUSAL_MULTI_PAIR_CLOSED_H1_AND_PORTFOLIO_STATE")
    assert "room-meta-01" not in ROOM_TAXONOMY_README


def test_v2_taxonomy_separates_relative_strength_alpha_and_anomaly_admission() -> None:
    taxonomy = {room.room_id: room.strategy_family for room in ROOM_TAXONOMY_V2}
    assert taxonomy["room-03"] == "g8_relative_strength_alpha"
    assert taxonomy["room-meta-01"] == "anomaly_admission_controller"
    assert len(taxonomy) == 11
    admission = next(
        room for room in ROOM_TAXONOMY_V2 if room.room_id == "room-meta-01"
    )
    assert "without predicting direction" in admission.thesis
    assert admission.decision_context_policy == (
        "DETERMINISTIC_DIRECTION_NEUTRAL_ADMISSION_NO_MODEL_CONTEXT"
    )
    assert all(
        room_id in ROOM_TAXONOMY_V2_README
        for room_id in ("room-03", "room-meta-01", "room-ai-01")
    )


def test_new_room_ids_and_families_match_the_content_addressed_research_queue() -> None:
    queue = build_research_queue()
    queued = {row["dojo_room_id"]: row["family"] for row in queue["candidates"]}
    taxonomy = {room.room_id: room.strategy_family for room in INITIAL_ROOM_TAXONOMY}

    assert queued == {
        "room-01": "asia_sweep_reclaim_be",
        "room-02": "h1_donchian_break_atr_trailing",
        "room-03": "g8_relative_strength_risk_budget",
    }
    assert {room_id: taxonomy[room_id] for room_id in queued} == queued


def test_registry_isolates_lineage_budget_namespace_and_train_denominator(
    registry: dict,
) -> None:
    normalized = validate_training_room_registry(registry)
    rooms = normalized["rooms"]
    assert normalized["room_count"] == 10
    assert len({room["trainer_lineage"]["lineage_id"] for room in rooms}) == 10
    assert len({room["artifact_namespace"] for room in rooms}) == 10
    assert (
        len({room["fixed_train_denominator"]["denominator_id"] for room in rooms}) == 10
    )
    assert all(
        room["fixed_train_denominator"]["room_id"] == room["room_id"] for room in rooms
    )
    assert normalized["sharing_policy"]["allowed_shared_components"] == [
        "SOURCE_BINDING",
        "EVALUATOR_BINDING",
        "COST_POLICY_BINDING",
        "RISK_POLICY_BINDING",
    ]
    assert normalized["authority"]["live_permission"] is False
    assert normalized["authority"]["order_authority"] == "NONE"


def test_v2_registry_adds_meta_room_without_rewriting_v1(
    registry: dict,
    v2_registry: dict,
) -> None:
    normalized_v1 = validate_training_room_registry(registry)
    normalized_v2 = validate_training_room_registry(v2_registry)
    assert normalized_v1["contract"] == "QR_DOJO_TRAINING_ROOM_REGISTRY_V1"
    assert normalized_v1["registry_sha256"] == (
        "53e1d9eaffa28875015cb97cd02f551b7ca0563713d27baa13d8d5b4a2695379"
    )
    assert normalized_v1["room_count"] == 10
    assert normalized_v2["contract"] == "QR_DOJO_TRAINING_ROOM_REGISTRY_V2"
    assert normalized_v2["schema_version"] == 2
    assert normalized_v2["room_count"] == 11
    families = {
        room["room_id"]: room["strategy_family"] for room in normalized_v2["rooms"]
    }
    assert families["room-03"] == "g8_relative_strength_alpha"
    assert families["room-meta-01"] == "anomaly_admission_controller"
    assert normalized_v1["registry_sha256"] != normalized_v2["registry_sha256"]


def test_v2_meta_room_can_issue_receipt_and_enter_common_sparring(
    v2_registry: dict,
) -> None:
    alpha = _receipt(v2_registry, "room-03")
    admission = _receipt(v2_registry, "room-meta-01")
    assert (
        validate_training_room_receipt(admission, registry=v2_registry)["room_id"]
        == "room-meta-01"
    )
    handoff = build_common_sparring_handoff(
        registry=v2_registry,
        handoff_id="common-sparring-v2",
        handoff_revision="revision-2",
        room_receipts=[admission, alpha],
        fixed_denominator=_common_denominator(),
    )
    assert [row["room_id"] for row in handoff["candidates"]] == [
        "room-03",
        "room-meta-01",
    ]


def test_registry_rejects_shared_room_local_identity_and_room_override(
    shared_bindings: dict, room_controls: dict, registry: dict
) -> None:
    collided = copy.deepcopy(room_controls)
    collided["room-g2-02"]["trainer_lineage_id"] = collided["room-g2-01"][
        "trainer_lineage_id"
    ]
    with pytest.raises(DojoTrainingRoomError, match="scoped to room-g2-02"):
        build_training_room_registry(
            registry_id="dojo-room-registry-v1",
            registry_revision="revision-1",
            shared_bindings=shared_bindings,
            room_controls=collided,
        )

    tampered = copy.deepcopy(registry)
    tampered["rooms"][0]["source_override_sha256"] = _sha("override")
    with pytest.raises(DojoTrainingRoomError, match="canonical V1"):
        validate_training_room_registry(tampered)


def test_terminal_room_receipt_binds_room_and_has_no_authority(registry: dict) -> None:
    receipt = _receipt(registry, "room-01")
    normalized = validate_training_room_receipt(receipt, registry=registry)

    assert normalized["terminal"] == {
        "status": "TERMINAL_VALIDATED",
        "expected_coordinate_count": 96,
        "observed_coordinate_count": 96,
        "failed_coordinate_count": 0,
        "fixed_denominator_complete": True,
        "candidate_gate_passed": True,
        "eligible_for_common_sparring": True,
    }
    assert normalized["candidate"]["room_denominator_reexecuted"] is True
    assert normalized["holdout_exam_result"] is False
    assert normalized["forward_arena_result"] is False
    assert normalized["authority"]["promotion_eligible"] is False
    assert normalized["authority"]["order_authority"] == "NONE"


def test_queue_bound_receipt_and_sparring_reject_noncanonical_room_candidate(
    registry: dict,
) -> None:
    queue = build_research_queue()
    good = build_training_room_receipt(
        registry=registry,
        room_id="room-01",
        candidate=_queued_candidate("room-01"),
        terminal_status="TERMINAL_VALIDATED",
        candidate_gate_passed=True,
        observed_coordinate_count=96,
        failed_coordinate_count=0,
        budget_consumed={
            "attempts": 1,
            "hypotheses": 1,
            "parameter_revisions": 1,
            "model_calls": 0,
        },
        artifact_relative_path=(
            "research/dojo/training_rooms/room-01/queue-terminal.json"
        ),
        research_queue=queue,
    )
    assert good["contract"] == QUEUE_BOUND_ROOM_RECEIPT_CONTRACT
    assert good["research_queue_binding"]["canonical_candidate_id"] == (
        "asia_sweep_reclaim_be"
    )
    assert good["research_queue_binding"]["canonical_family"] == (
        "asia_sweep_reclaim_be"
    )
    assert (
        validate_training_room_receipt(good, registry=registry, research_queue=queue)
        == good
    )

    with pytest.raises(
        DojoTrainingRoomError, match="differs from the canonical queue candidate"
    ):
        build_training_room_receipt(
            registry=registry,
            room_id="room-01",
            candidate={
                **_queued_candidate("room-01"),
                "candidate_id": "room-01:not-the-queued-candidate",
            },
            terminal_status="TERMINAL_VALIDATED",
            candidate_gate_passed=True,
            observed_coordinate_count=96,
            failed_coordinate_count=0,
            budget_consumed={
                "attempts": 1,
                "hypotheses": 1,
                "parameter_revisions": 1,
                "model_calls": 0,
            },
            artifact_relative_path=(
                "research/dojo/training_rooms/room-01/forged-terminal.json"
            ),
            research_queue=queue,
        )

    unbound_v1 = _receipt(
        registry,
        "room-01",
        candidate={
            **_queued_candidate("room-01"),
            "candidate_id": "room-01:not-the-queued-candidate",
        },
    )
    with pytest.raises(DojoTrainingRoomError, match="V2 receipt"):
        build_common_sparring_handoff(
            registry=registry,
            handoff_id="queue-bound-common-sparring-v2",
            handoff_revision="revision-2",
            room_receipts=[unbound_v1],
            fixed_denominator=_common_denominator(),
            research_queue=queue,
        )

    handoff = build_common_sparring_handoff(
        registry=registry,
        handoff_id="queue-bound-common-sparring-v2",
        handoff_revision="revision-2",
        room_receipts=[good],
        fixed_denominator=_common_denominator(),
        research_queue=queue,
    )
    assert handoff["contract"] == QUEUE_BOUND_COMMON_SPARRING_HANDOFF_CONTRACT
    assert (
        handoff["candidates"][0]["research_queue_binding"]
        == good["research_queue_binding"]
    )


def test_cross_room_parameter_inspiration_requires_new_hypothesis_and_revision(
    registry: dict,
) -> None:
    parent = {
        "source_room_id": "room-01",
        "source_candidate_id": "room-01:candidate-1",
        "source_hypothesis_id": "room-01:hypothesis-1",
        "source_revision_id": "room-01:revision-1",
        "source_parameters_sha256": _sha("parameters:room-01"),
        "source_result_sha256": _sha("result:room-01"),
    }
    candidate = _candidate("room-02", cross_room_parent=parent)
    candidate["parameters_sha256"] = parent["source_parameters_sha256"]
    receipt = _receipt(registry, "room-02", candidate=candidate)
    policy = receipt["candidate"]["cross_room_copy_policy"]

    assert policy == {
        "kind": "CROSS_ROOM_NEW_HYPOTHESIS_AND_REVISION",
        "new_hypothesis_declared": True,
        "new_revision_declared": True,
        "source_result_reused_as_room_result": False,
        "room_denominator_reexecution_required": True,
    }
    assert receipt["candidate"]["hypothesis_id"].startswith("room-02:")
    assert receipt["candidate"]["revision_id"].startswith("room-02:")
    assert (
        receipt["candidate"]["terminal_result_sha256"] != parent["source_result_sha256"]
    )

    invalid = _candidate("room-02", cross_room_parent=parent)
    invalid["hypothesis_id"] = parent["source_hypothesis_id"]
    with pytest.raises(DojoTrainingRoomError, match="scoped to room-02"):
        _receipt(registry, "room-02", candidate=invalid)


def test_room_receipt_fails_closed_on_budget_namespace_and_denominator_drift(
    registry: dict,
) -> None:
    with pytest.raises(DojoTrainingRoomError, match="exceeds room budget"):
        build_training_room_receipt(
            registry=registry,
            room_id="room-g2-01",
            candidate=_candidate("room-g2-01"),
            terminal_status="TERMINAL_VALIDATED",
            candidate_gate_passed=True,
            observed_coordinate_count=96,
            failed_coordinate_count=0,
            budget_consumed={
                "attempts": 4,
                "hypotheses": 1,
                "parameter_revisions": 1,
                "model_calls": 0,
            },
            artifact_relative_path=(
                "research/dojo/training_rooms/room-g2-01/terminal.json"
            ),
        )

    with pytest.raises(DojoTrainingRoomError, match="escapes"):
        build_training_room_receipt(
            registry=registry,
            room_id="room-g2-01",
            candidate=_candidate("room-g2-01"),
            terminal_status="TERMINAL_VALIDATED",
            candidate_gate_passed=True,
            observed_coordinate_count=96,
            failed_coordinate_count=0,
            budget_consumed={
                "attempts": 1,
                "hypotheses": 1,
                "parameter_revisions": 1,
                "model_calls": 0,
            },
            artifact_relative_path=(
                "research/dojo/training_rooms/room-g2-02/stolen.json"
            ),
        )

    tampered = _receipt(registry, "room-g2-01")
    tampered["fixed_train_denominator"]["expected_coordinate_count"] = 95
    with pytest.raises(DojoTrainingRoomError, match="canonical V1"):
        validate_training_room_receipt(tampered, registry=registry)


def test_common_sparring_accepts_one_validated_candidate_per_room_on_four_slots(
    registry: dict,
) -> None:
    receipts = [
        _receipt(registry, "room-g2-01"),
        _receipt(registry, "room-01"),
        _receipt(registry, "room-02"),
        _receipt(registry, "room-03"),
        _receipt(registry, "room-ai-01"),
    ]
    handoff = build_common_sparring_handoff(
        registry=registry,
        handoff_id="common-sparring-v1",
        handoff_revision="revision-1",
        room_receipts=list(reversed(receipts)),
        fixed_denominator=_common_denominator(),
    )
    normalized = validate_common_sparring_handoff(
        handoff, registry=registry, room_receipts=receipts
    )

    assert normalized["candidate_count"] == 5
    assert normalized["allocator"]["simultaneous_slots"] == COMMON_SPARRING_SLOTS
    assert normalized["allocator"]["maximum_candidates_per_room"] == 1
    assert [row["room_id"] for row in normalized["candidates"]] == [
        "room-g2-01",
        "room-01",
        "room-02",
        "room-03",
        "room-ai-01",
    ]
    assert all(
        row["room_train_result_is_selection_provenance_only"] is True
        and row["common_sparring_result_sha256"] is None
        for row in normalized["candidates"]
    )
    assert normalized["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert normalized["authority"]["live_permission"] is False


def test_common_sparring_rejects_duplicate_room_and_nonvalidated_receipt(
    registry: dict,
) -> None:
    valid = _receipt(registry, "room-01")
    with pytest.raises(DojoTrainingRoomError, match="at most one"):
        build_common_sparring_handoff(
            registry=registry,
            handoff_id="common-sparring-v1",
            handoff_revision="revision-1",
            room_receipts=[valid, valid],
            fixed_denominator=_common_denominator(),
        )

    rejected = _receipt(registry, "room-02", validated=False)
    with pytest.raises(DojoTrainingRoomError, match="only terminal validated"):
        build_common_sparring_handoff(
            registry=registry,
            handoff_id="common-sparring-v1",
            handoff_revision="revision-1",
            room_receipts=[valid, rejected],
            fixed_denominator=_common_denominator(),
        )


def test_holdout_exam_and_forward_arena_are_explicitly_separate(
    registry: dict,
) -> None:
    separation = registry["stage_separation"]
    assert separation["holdout_exam"]["separate_from_training_rooms"] is True
    assert separation["holdout_exam"]["train_or_sparring_result_reuse_allowed"] is False
    assert separation["forward_arena"]["separate_from_holdout_exam"] is True
    assert separation["forward_arena"]["historical_result_counts_as_forward"] is False

    handoff = build_common_sparring_handoff(
        registry=registry,
        handoff_id="common-sparring-v1",
        handoff_revision="revision-1",
        room_receipts=[_receipt(registry, "room-01")],
        fixed_denominator=_common_denominator(),
    )
    assert handoff["holdout_exam_result"] is False
    assert handoff["forward_arena_result"] is False
    assert handoff["authority"]["order_authority"] == "NONE"
    assert handoff["authority"]["broker_mutation_allowed"] is False


def test_handoff_validation_rejects_candidate_or_denominator_tamper(
    registry: dict,
) -> None:
    receipts = [_receipt(registry, "room-01"), _receipt(registry, "room-02")]
    handoff = build_common_sparring_handoff(
        registry=registry,
        handoff_id="common-sparring-v1",
        handoff_revision="revision-1",
        room_receipts=receipts,
        fixed_denominator=_common_denominator(),
    )

    candidate_tamper = copy.deepcopy(handoff)
    candidate_tamper["candidates"][0]["parameters_sha256"] = _sha("tampered")
    with pytest.raises(DojoTrainingRoomError, match="canonical V1"):
        validate_common_sparring_handoff(
            candidate_tamper, registry=registry, room_receipts=receipts
        )

    denominator_tamper = copy.deepcopy(handoff)
    denominator_tamper["fixed_denominator"]["simultaneous_slots"] = 3
    with pytest.raises(DojoTrainingRoomError, match="canonical V1"):
        validate_common_sparring_handoff(
            denominator_tamper, registry=registry, room_receipts=receipts
        )
