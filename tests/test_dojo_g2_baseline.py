from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_g2_baseline import (
    DojoG2BaselineError,
    G2_PAIRS,
    REGISTRY_RELATIVE_PATH,
    build_g2_baseline,
    canonical_sha256,
    validate_g2_baseline,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict[str, object]:
    value = json.loads((ROOT / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_registry_is_exact_canonical_g2_v1_artifact() -> None:
    expected = build_g2_baseline()
    registry = _registry()

    assert registry == expected
    assert validate_g2_baseline(registry) == expected
    assert registry["execution_status"] == "PREREGISTERED_NOT_EXECUTED"
    assert tuple(registry["universe"]["pairs"]) == G2_PAIRS
    assert set(G2_PAIRS) == set(DEFAULT_TRADER_PAIRS)
    assert len(G2_PAIRS) == 28


def test_g2_workers_and_allocator_are_fixed_to_requested_envelope() -> None:
    artifact = validate_g2_baseline(_registry())
    workers = artifact["workers"]

    assert [(row["family"], row["role"]) for row in workers] == [
        ("spike_fade", "CONTROL"),
        ("burst", "ACTIVE"),
        ("pullback_limit", "ACTIVE"),
        ("prev_day_extreme_fade", "ACTIVE"),
        ("round_number_fade", "ACTIVE"),
        ("mean_revert_24h", "ACTIVE"),
    ]
    assert workers[2]["config"]["pull_atr"] == 0.6
    assert workers[5]["config"]["fade_atr"] == 1.2
    for row in workers:
        config = row["config"]
        assert row["execution_status"] == "PREREGISTERED_NOT_EXECUTED"
        assert row["planned_proposal_slot_cost_if_executed"] == 1
        assert config["pairs"] == list(G2_PAIRS)
        assert config["exit_policy"] == "FIXED"
        assert config["tp_atr"] == 3.0
        assert config["tp_pips"] is None
        assert config["sl_pips"] == 25.0
        assert config["ceiling_min"] == 60
        assert config["per_pos_lev"] == 2.0
        assert config["max_concurrent_per_pair"] == 1
        assert config["global_max_concurrent"] == 1
        assert config["atr_floor_pips"] == 0.5
        assert config["order_authority"] == "NONE"
        assert config["live_permission"] is False
        assert config["external_broker_mutation_allowed"] is False

    assert artifact["allocator"] == {
        "contract": "QR_DOJO_G2_FIXED_ALLOCATOR_V1",
        "simultaneous_slots": 4,
        "max_concurrent_per_pair": 1,
        "max_concurrent_per_family": 1,
        "worker_global_max_concurrent": 1,
        "per_position_leverage": 2.0,
        "maximum_gross_leverage": 8.0,
        "gross_leverage_formula": "simultaneous_slots*per_position_leverage",
        "new_position_margin_admission_fraction_max": 0.45,
        "margin_closeout_fraction": 0.90,
        "portfolio_stop_risk_fraction": 0.10,
        "allocation_changes_require_new_version": True,
    }


def test_g2_budget_is_preregistered_but_not_consumed_or_reserved() -> None:
    budget = validate_g2_baseline(_registry())["search_budget"]

    assert budget["execution_status"] == "PREREGISTERED_NOT_EXECUTED"
    assert budget["observed_proposal_slots_consumed_before"] == 4
    assert budget["actual_g2_model_invocations"] == 0
    assert budget["actual_g2_reservation_events"] == 0
    assert budget["actual_g2_reserved_proposal_slots"] == 0
    assert budget["actual_g2_proposal_slots_consumed"] == 0
    assert budget["planned_proposal_slots_for_g2"] == 6
    assert budget["projected_proposal_slots_consumed_after_execution"] == 10
    assert budget["max_proposal_slots"] == 14
    assert budget["projected_proposal_slots_remaining_after_execution"] == 4
    assert budget["next_planned_generation"] == "G3"
    assert budget["planned_g3_proposal_slots_after_g2_execution"] == 4


def test_g2_authority_is_research_only_and_does_not_rewrite_prior_seals() -> None:
    artifact = validate_g2_baseline(_registry())

    assert artifact["baseline_policy"] == {
        "kind": "DETERMINISTIC_VERSIONED_BASELINE",
        "mutates_prior_sealed_artifacts": False,
        "reinterprets_prior_results": False,
        "supersedes_prior_sealed_artifacts": False,
        "future_changes_require_new_contract_version": True,
    }
    assert artifact["authority"] == {
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
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
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("workers", 1, "config", "per_pos_lev"), 2.1),
        (("workers", 2, "config", "pull_atr"), 0.7),
        (("workers", 0, "role"), "ACTIVE"),
        (("allocator", "simultaneous_slots"), 5),
        (("allocator", "margin_closeout_fraction"), 0.95),
        (
            ("search_budget", "projected_proposal_slots_remaining_after_execution"),
            3,
        ),
        (("search_budget", "actual_g2_model_invocations"), 1),
        (("search_budget", "actual_g2_proposal_slots_consumed"), 6),
        (("authority", "live_permission"), True),
        (("baseline_policy", "mutates_prior_sealed_artifacts"), True),
    ],
)
def test_mutation_fails_closed_even_if_attacker_reseals_hash(
    path: tuple[object, ...], value: object
) -> None:
    artifact = copy.deepcopy(_registry())
    target: object = artifact
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    body = {key: item for key, item in artifact.items() if key != "artifact_sha256"}
    artifact["artifact_sha256"] = canonical_sha256(body)
    if body == {
        key: item
        for key, item in build_g2_baseline().items()
        if key != "artifact_sha256"
    }:
        raise AssertionError("test mutation did not change the body")

    with pytest.raises(DojoG2BaselineError):
        validate_g2_baseline(artifact)


def test_extra_worker_and_unknown_field_fail_closed() -> None:
    extra_worker = copy.deepcopy(_registry())
    extra_worker["workers"].append(copy.deepcopy(extra_worker["workers"][0]))
    with pytest.raises(DojoG2BaselineError):
        validate_g2_baseline(extra_worker)

    extra_field = copy.deepcopy(_registry())
    extra_field["model_authority"] = "NONE"
    with pytest.raises(DojoG2BaselineError):
        validate_g2_baseline(extra_field)
