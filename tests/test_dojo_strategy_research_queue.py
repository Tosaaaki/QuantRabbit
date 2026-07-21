from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_g2_baseline import build_g2_baseline
from quant_rabbit.dojo_strategy_research_queue import (
    DojoStrategyResearchQueueError,
    EVIDENCE_CLASS,
    REGISTRY_RELATIVE_PATH,
    ROOM_ISOLATION_CONTRACT,
    SELECTION_BASIS,
    TRIGGER_CONTRACT,
    build_initial_reservation_state,
    build_research_queue,
    canonical_sha256,
    load_research_queue,
    main,
    plan_reservation,
    validate_research_queue,
    validate_reservation_state,
    validate_trigger,
)


ROOT = Path(__file__).resolve().parents[1]


def _queue() -> dict[str, object]:
    value = json.loads((ROOT / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _trigger(
    seed: str,
    *,
    semantic_seed: str | None = None,
    candidate_id: str | None = None,
    kind: str = "TERMINAL_RESULT",
) -> dict[str, object]:
    terminal = kind == "TERMINAL_RESULT"
    return {
        "contract": TRIGGER_CONTRACT,
        "schema_version": 1,
        "trigger_kind": kind,
        "result_artifact_sha256": seed * 64,
        "semantic_result_sha256": (semantic_seed or seed) * 64,
        "result_candidate_id": candidate_id,
        "source_partition": "TRAIN",
        "evidence_class": EVIDENCE_CLASS,
        "terminal_result": terminal,
        "material_change": True,
        "holdout_opened": False,
        "prospective_window_opened": False,
        "global_untouched_holdout_claimed": False,
        "target_multiple_backsolve_used": False,
        "selection_basis": SELECTION_BASIS,
        "economics": {
            "recorded_bid_ask_costed": True,
            "slippage_costed": True,
            "financing_costed": True,
            "continuous_mtm_complete": True,
            "margin_replayed": True,
            "lopo_complete": True,
            "fixed_denominator_complete": True,
        },
    }


def _reseal_queue(value: dict[str, object]) -> None:
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    value["artifact_sha256"] = canonical_sha256(body)


def _reseal_candidate(value: dict[str, object]) -> None:
    body = {key: item for key, item in value.items() if key != "candidate_sha256"}
    value["candidate_sha256"] = canonical_sha256(body)


def test_registry_is_exact_content_addressed_v1_queue() -> None:
    expected = build_research_queue()
    registry = _queue()

    assert registry == expected
    assert validate_research_queue(registry) == expected
    assert load_research_queue(ROOT / REGISTRY_RELATIVE_PATH) == expected
    assert Path(REGISTRY_RELATIVE_PATH).stem.endswith(expected["artifact_sha256"])
    assert [row["candidate_id"] for row in registry["candidates"]] == [
        "asia_sweep_reclaim_be",
        "h1_donchian_break_atr_trailing",
        "g8_relative_strength_risk_budget",
    ]


def test_candidates_are_novel_and_bind_causal_falsifiable_economics() -> None:
    queue = validate_research_queue(_queue())
    g2_families = {worker["family"] for worker in build_g2_baseline()["workers"]}

    assert [candidate["dojo_room_id"] for candidate in queue["candidates"]] == [
        "room-01",
        "room-02",
        "room-03",
    ]
    room_roots: set[str] = set()
    for candidate in queue["candidates"]:
        assert candidate["family"] not in g2_families
        assert candidate["status"] == "DESIGN_BACKLOG_NOT_EXECUTED"
        assert candidate["causal_inputs"]
        assert candidate["forbidden_inputs"]
        assert candidate["novelty"]["not_a_g2_family"] is True
        assert candidate["implementation_dependencies"]
        assert candidate["falsification_conditions"]
        room = candidate["room_isolation"]
        assert room["contract"] == ROOM_ISOLATION_CONTRACT
        assert room["dojo_room_id"] == candidate["dojo_room_id"]
        assert room["strategy_identity"] == candidate["candidate_id"]
        assert room["search_budget"]["scope"] == "ROOM_ONLY"
        assert room["search_budget"]["proposal_slots_consumed"] == 0
        assert room["search_budget"]["borrow_from_other_rooms_allowed"] is False
        assert room["shared_contracts_only"] == [
            "EVALUATOR_CONTRACT",
            "COST_CONTRACT",
            "RISK_CONTRACT",
            "SOURCE_CONTRACT",
        ]
        assert room["artifact_root"] not in room_roots
        room_roots.add(room["artifact_root"])
        assert (
            candidate["minimum_train"]["train_is_hypothesis_generation_not_edge_proof"]
            is True
        )
        assert candidate["independent_window_requirement"] == {
            "required": True,
            "next_window_role": "SEPARATELY_PREREGISTERED_INDEPENDENT_TRAIN_DIAGNOSTIC",
            "chronological_non_overlap_required": True,
            "candidate_and_evaluator_sealed_before_window_read": True,
            "burn_registry_check_required": True,
            "global_untouched_holdout_claim_allowed": False,
            "historical_holdout_access_allowed": False,
            "prospective_data_access_allowed": False,
            "opened_window_must_be_burned_after_one_evaluation": True,
            "reason": (
                "The available 2024-2026 historical corpus is worn; independence is "
                "lineage-local TRAIN discipline and must not be relabeled as proof."
            ),
        }
        economics = candidate["economics_requirements"]
        assert economics["recorded_bid_ask_required"] is True
        assert economics["continuous_account_mtm_after_each_action_required"] is True
        assert economics["hedge_net_margin_replay_required"] is True
        assert economics["lopo_required"] is True
        assert economics["missing_cost_mtm_margin_or_lopo_is_terminal_reject"] is True
        assert candidate["authority"] == queue["authority"]
        assert candidate["authority"]["live_permission"] is False
        assert candidate["authority"]["order_authority"] == "NONE"

    isolation = queue["room_isolation_contract"]
    assert isolation["cross_room_result_or_parameter_copy_policy"] == {
        "copy_is_new_hypothesis": True,
        "new_candidate_id_required": True,
        "new_candidate_sha256_required": True,
        "destination_room_budget_debit_required": True,
        "source_room_result_inheritance_allowed": False,
        "silent_parameter_inheritance_allowed": False,
    }
    common = isolation["common_sparring_arena"]
    holdout = isolation["holdout_examination_arena"]
    forward = isolation["prospective_forward_arena"]
    assert common["shared_capital_slots"] == 4
    assert common["role"] == "WORN_HISTORICAL_TRAIN_PORTFOLIO_COMPARISON_ONLY"
    assert common["target_multiple_backsolve_allowed"] is False
    assert holdout["currently_open"] is False
    assert holdout["trainer_access_allowed"] is False
    assert forward["currently_open"] is False
    assert forward["trainer_access_allowed"] is False
    arena_roots = {
        common["artifact_root"],
        holdout["artifact_root"],
        forward["artifact_root"],
    }
    assert len(arena_roots) == 3
    assert arena_roots.isdisjoint(room_roots)


def test_mutated_or_g2_duplicate_queue_fails_even_if_resealed() -> None:
    queue = copy.deepcopy(_queue())
    first = queue["candidates"][0]
    first["family"] = "spike_fade"
    _reseal_candidate(first)
    _reseal_queue(queue)

    with pytest.raises(DojoStrategyResearchQueueError, match="duplicates G2"):
        validate_research_queue(queue)

    changed = copy.deepcopy(_queue())
    changed["candidates"][0]["minimum_train"]["chronological_months_min"] = 1
    _reseal_candidate(changed["candidates"][0])
    _reseal_queue(changed)
    with pytest.raises(DojoStrategyResearchQueueError, match="immutable V1"):
        validate_research_queue(changed)


def test_room_state_cannot_be_shared_or_merged_even_if_resealed() -> None:
    queue = copy.deepcopy(_queue())
    queue["candidates"][1]["dojo_room_id"] = "room-01"
    queue["candidates"][1]["room_isolation"]["dojo_room_id"] = "room-01"
    _reseal_candidate(queue["candidates"][1])
    _reseal_queue(queue)
    with pytest.raises(DojoStrategyResearchQueueError, match="dojo_room_id"):
        validate_research_queue(queue)

    shared_root = copy.deepcopy(_queue())
    shared_root["candidates"][1]["room_isolation"]["artifact_root"] = shared_root[
        "candidates"
    ][0]["room_isolation"]["artifact_root"]
    _reseal_candidate(shared_root["candidates"][1])
    _reseal_queue(shared_root)
    with pytest.raises(DojoStrategyResearchQueueError, match="artifact roots"):
        validate_research_queue(shared_root)


def test_initial_terminal_result_reserves_exactly_one_candidate() -> None:
    queue = _queue()
    state = build_initial_reservation_state(queue)
    decision = plan_reservation(
        queue=queue,
        trigger=_trigger("1"),
        previous_state=state,
    )

    assert decision["action"] == "RESERVE_ONE_CANDIDATE"
    assert decision["reservation"]["candidate_id"] == "asia_sweep_reclaim_be"
    assert decision["reservation"]["dojo_room_id"] == "room-01"
    next_state = validate_reservation_state(decision["next_state"], queue)
    assert next_state["completed_candidate_ids"] == []
    assert next_state["reservation_count"] == 1
    assert next_state["active_reservation"]["candidate_id"] == "asia_sweep_reclaim_be"
    assert next_state["authority"]["live_permission"] is False


def test_explicit_empty_queue_never_falls_back_to_builtin_contract() -> None:
    with pytest.raises(DojoStrategyResearchQueueError):
        build_initial_reservation_state({})
    with pytest.raises(DojoStrategyResearchQueueError):
        validate_reservation_state({}, {})


def test_unchanged_semantic_result_is_exact_state_noop() -> None:
    queue = _queue()
    first = plan_reservation(
        queue=queue,
        trigger=_trigger("2"),
        previous_state=build_initial_reservation_state(queue),
    )
    unchanged = plan_reservation(
        queue=queue,
        trigger=_trigger("3", semantic_seed="2"),
        previous_state=first["next_state"],
    )

    assert unchanged["action"] == "NO_OP_RESULT_UNCHANGED"
    assert unchanged["reservation"] is None
    assert unchanged["next_state"] == first["next_state"]


def test_changed_terminal_results_advance_in_order_then_exhaust() -> None:
    queue = _queue()
    decision1 = plan_reservation(
        queue=queue,
        trigger=_trigger("4"),
        previous_state=build_initial_reservation_state(queue),
    )
    decision2 = plan_reservation(
        queue=queue,
        trigger=_trigger("5", candidate_id="asia_sweep_reclaim_be"),
        previous_state=decision1["next_state"],
    )
    assert decision2["reservation"]["candidate_id"] == "h1_donchian_break_atr_trailing"
    assert decision2["reservation"]["dojo_room_id"] == "room-02"
    assert decision2["next_state"]["completed_candidate_ids"] == [
        "asia_sweep_reclaim_be"
    ]

    decision3 = plan_reservation(
        queue=queue,
        trigger=_trigger("6", candidate_id="h1_donchian_break_atr_trailing"),
        previous_state=decision2["next_state"],
    )
    assert (
        decision3["reservation"]["candidate_id"] == "g8_relative_strength_risk_budget"
    )
    assert decision3["reservation"]["dojo_room_id"] == "room-03"
    assert decision3["next_state"]["reservation_count"] == 3

    exhausted = plan_reservation(
        queue=queue,
        trigger=_trigger("7", candidate_id="g8_relative_strength_risk_budget"),
        previous_state=decision3["next_state"],
    )
    assert exhausted["action"] == "QUEUE_EXHAUSTED"
    assert exhausted["reservation"] is None
    assert exhausted["next_state"]["completed_candidate_ids"] == [
        "asia_sweep_reclaim_be",
        "h1_donchian_break_atr_trailing",
        "g8_relative_strength_risk_budget",
    ]
    assert exhausted["next_state"]["active_reservation"] is None


def test_unbound_result_cannot_replace_an_active_reservation() -> None:
    queue = _queue()
    first = plan_reservation(
        queue=queue,
        trigger=_trigger("8"),
        previous_state=build_initial_reservation_state(queue),
    )
    decision = plan_reservation(
        queue=queue,
        trigger=_trigger("9", candidate_id="h1_donchian_break_atr_trailing"),
        previous_state=first["next_state"],
    )

    assert decision["action"] == "NO_OP_ACTIVE_RESERVATION_RESULT_UNBOUND"
    assert decision["reservation"] is None
    assert decision["next_state"] == first["next_state"]


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("holdout_opened",), True, "holdout/prospective access is forbidden"),
        (
            ("prospective_window_opened",),
            True,
            "holdout/prospective access is forbidden",
        ),
        (
            ("global_untouched_holdout_claimed",),
            True,
            "holdout/prospective access is forbidden",
        ),
        (("target_multiple_backsolve_used",), True, "backsolving is forbidden"),
        (("economics", "continuous_mtm_complete"), False, "trigger economics"),
        (("economics", "lopo_complete"), False, "trigger economics"),
        (("material_change",), False, "terminal trigger"),
    ],
)
def test_trigger_fails_closed_on_holdout_target_or_incomplete_economics(
    path: tuple[str, ...], value: object, error: str
) -> None:
    trigger = _trigger("a")
    target: dict[str, object] = trigger
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(DojoStrategyResearchQueueError, match=error):
        validate_trigger(trigger)


def test_material_change_trigger_is_admissible_but_still_train_only() -> None:
    trigger = _trigger("b", kind="MATERIAL_RESULT_CHANGE")
    assert validate_trigger(trigger) == trigger
    queue = _queue()
    decision = plan_reservation(
        queue=queue,
        trigger=trigger,
        previous_state=build_initial_reservation_state(queue),
    )
    assert decision["action"] == "RESERVE_ONE_CANDIDATE"
    assert decision["authority"]["proof_eligible"] is False
    assert decision["authority"]["live_permission"] is False


def test_cli_validates_registry_without_writing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["validate", "--registry", str(ROOT / REGISTRY_RELATIVE_PATH)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "artifact_sha256": build_research_queue()["artifact_sha256"],
        "candidate_count": 3,
        "contract": "QR_DOJO_STRATEGY_RESEARCH_QUEUE_V1",
        "live_permission": False,
        "order_authority": "NONE",
        "status": "VALID",
    }
