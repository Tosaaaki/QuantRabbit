from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import quant_rabbit.dojo_trainer_generation_driver as driver
from quant_rabbit.dojo_bot_catalog import catalog_manifest
from quant_rabbit.dojo_candidate_lineage_registry import CandidateLineageSnapshot


NOW = "2026-07-20T00:00:00Z"


def _binding() -> dict:
    return {
        "registry_id": "qr-lineage",
        "lineage_prefix": "qr-",
        "attempt_ordinal": 1,
        "study_sha256": "1" * 64,
        "evaluation_sha256": "2" * 64,
        "evaluation_artifact_sha256": "3" * 64,
        "evaluation_artifact_size_bytes": 123,
        "result_event_sha256": "4" * 64,
        "result_event_sequence": 2,
        "lineage_tip_sha256": "4" * 64,
    }


def _lineage() -> CandidateLineageSnapshot:
    return CandidateLineageSnapshot(
        registry_id="qr-lineage",
        lineage_prefix="qr-",
        event_count=2,
        latest_sequence=2,
        latest_event_sha256="4" * 64,
        latest_event_at_utc=NOW,
        studies=(
            {
                "attempt_ordinal": 1,
                "study_sha256": "1" * 64,
                "candidate_count": 1,
            },
        ),
        results=(
            {
                "attempt_ordinal": 1,
                "study_sha256": "1" * 64,
                "evaluation_sha256": "2" * 64,
                "evaluation_artifact_sha256": "3" * 64,
                "evaluation_artifact_size_bytes": 123,
            },
        ),
        cumulative_unique_config_sha256s=("5" * 64,),
        cumulative_unique_proposal_sha256s=("6" * 64,),
        events=(
            {"event_type": "GENESIS", "event_sha256": "7" * 64, "sequence": 1},
            {
                "event_type": "RESULT_BOUND",
                "event_sha256": "4" * 64,
                "sequence": 2,
            },
        ),
    )


def _state() -> dict:
    return {
        "registry_id": "qr-lineage",
        "lineage_prefix": "qr-",
        "phase": "READY_FOR_MODEL",
        "state_sha256": "8" * 64,
        "last_terminal_result_binding": _binding(),
        "fixed_envelope_sha256": "9" * 64,
        "fixed_envelope": {
            "window_role": "TRAIN",
            "window": {
                "start_utc": "2025-01-01T00:00:00Z",
                "end_utc": "2025-02-01T00:00:00Z",
                "corpus_sha256": "a" * 64,
            },
            "initial_balance_jpy": 200_000.0,
            "trade_pairs": ["EUR_USD"],
            "feed_pairs": ["EUR_USD"],
            "intrabar_paths": ["OHLC", "OLHC"],
            "cost_arms": {"BASE": {}, "STRESS": {}},
            "thresholds": {"normal_mtm_drawdown_max": 0.1},
            "scorer_contract": "QR_DOJO_BOT_TRAINER_EVALUATION_V1",
            "source_digests": {
                "bots/lab_bot.py": "b" * 64,
                "src/quant_rabbit/dojo_bot_trainer.py": "c" * 64,
            },
            "source_bundle_sha256": "d" * 64,
        },
    }


def _packet(state: dict) -> dict:
    binding = state["last_terminal_result_binding"]
    return {
        "packet_sha256": "e" * 64,
        "source_bindings": {
            "registry_id": "qr-lineage",
            "lineage_prefix": "qr-",
            "attempt_ordinal": 1,
            "study_sha256": binding["study_sha256"],
            "run_sha256": "0" * 64,
            "evaluation_sha256": binding["evaluation_sha256"],
            "evaluation_artifact_sha256": binding[
                "evaluation_artifact_sha256"
            ],
            "lineage_result_event_sha256": binding["result_event_sha256"],
            "lineage_tip_sha256": binding["lineage_tip_sha256"],
            "tuning_state_sha256": state["state_sha256"],
            "fixed_envelope_sha256": state["fixed_envelope_sha256"],
            "external_witness_status": "ABSENT",
            "exact_result_binding_verified": True,
        },
        "fixed_environment": {
            **{
                key: copy.deepcopy(state["fixed_envelope"][key])
                for key in (
                    "window_role",
                    "window",
                    "initial_balance_jpy",
                    "trade_pairs",
                    "feed_pairs",
                    "intrabar_paths",
                    "cost_arms",
                    "thresholds",
                )
            },
            "catalog": catalog_manifest(),
        },
        "search_budget": {
            "phase": "READY_FOR_MODEL",
            "attempts_consumed": 1,
            "attempts_remaining": 2,
            "max_attempts": 3,
            "proposal_slots_consumed": 1,
            "proposal_slots_remaining": 13,
            "max_proposal_slots": 14,
            "invalid_proposal_count": 0,
            "duplicate_proposal_count": 0,
        },
        "current_run": {
            "status": "COMPLETE",
            "fixed_denominator": {
                "expected_cell_count": 4,
                "observed_cell_count": 4,
            },
            "candidate_ids": ["qr-a1"],
        },
        "cells": [
            {
                "candidate_id": "qr-a1",
                "intrabar": intrabar,
                "cost_arm": arm,
            }
            for intrabar in ("OHLC", "OLHC")
            for arm in ("BASE", "STRESS")
        ],
        "failed_coordinates": [],
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }


def _health() -> dict:
    return {
        "process_state": "IDLE",
    }


def _install_context(monkeypatch, state: dict, packet: dict) -> list[dict]:
    reserve_calls: list[dict] = []
    store = {
        "latest_state": state,
        "latest_event_sha256": "f" * 64,
        "event_count": 1,
        **driver._REQUIRED_STORE_BOUNDARY,
    }
    monkeypatch.setattr(driver, "verify_state_store", lambda path: copy.deepcopy(store))
    monkeypatch.setattr(driver, "verify_tuning_state", lambda value: copy.deepcopy(value))
    monkeypatch.setattr(driver, "verify_registry", lambda path, artifact_root: _lineage())
    monkeypatch.setattr(
        driver, "verify_trainer_packet", lambda value: copy.deepcopy(value)
    )
    def tuning_status(value):
        attempts = value.get("attempts", [])
        invocations = [
            invocation
            for attempt in attempts
            for invocation in attempt.get("invocations", [])
        ]
        submissions = [
            submission
            for invocation in invocations
            for submission in invocation.get("submissions", [])
        ]
        return {
            "attempts_consumed": 1 + len(attempts),
            "proposal_slots_consumed": 1
            + sum(invocation.get("proposal_slot_charge", 0) for invocation in invocations),
            "model_invocation_count": len(invocations),
            "invalid_proposal_count": sum(
                row.get("status") in {"INVALID", "BUDGET_REJECTED"}
                for row in submissions
            ),
            "duplicate_proposal_count": sum(
                row.get("status") == "DUPLICATE" for row in submissions
            ),
        }

    monkeypatch.setattr(driver, "status_artifact", tuning_status)

    def seal_manifest(**kwargs):
        return {
            "manifest_sha256": "0" * 64,
            "attempt_ordinal": 2,
            "provider_id": kwargs["provider_id"],
            "model_id": kwargs["model_id"],
        }

    monkeypatch.setattr(driver, "seal_model_invocation_manifest", seal_manifest)
    monkeypatch.setattr(
        driver,
        "plan_immutable_generation",
        lambda **kwargs: {
            "action": "RESERVE_NEXT_MODEL_INVOCATION",
            "decision_sha256": "1" * 64,
            "health_sha256": "2" * 64,
        },
    )

    def reserve(value, **kwargs):
        reserve_calls.append(kwargs)
        return {
            **copy.deepcopy(value),
            "phase": "MODEL_INVOCATION_RESERVED",
            "state_sha256": "3" * 64,
            "attempts": [
                {
                    "invocations": [
                        {
                            "invocation_id": kwargs["invocation_id"],
                            "request_sha256": kwargs["request_sha256"],
                        }
                    ]
                }
            ],
        }

    monkeypatch.setattr(driver, "reserve_model_invocation", reserve)
    return reserve_calls


def _plan(monkeypatch, *, adapter: bool) -> tuple[driver.TrainerGenerationPlan, list]:
    state = _state()
    packet = _packet(state)
    calls = _install_context(monkeypatch, state, packet)
    plan = driver.plan_terminal_to_next_generation(
        tuning_state_events_dir=Path("state"),
        lineage_events_dir=Path("lineage"),
        artifact_root=Path("artifacts"),
        trainer_packet=packet,
        prompt_text="Propose a bounded fixed-envelope candidate set.",
        provider_id="openai-api",
        model_id="gpt-fixed",
        health=_health(),
        prepared_at_utc=NOW,
        model_adapter_configured=adapter,
    )
    return plan, calls


def test_missing_adapter_stops_at_ready_for_model_without_budget_charge(
    monkeypatch,
) -> None:
    plan, calls = _plan(monkeypatch, adapter=False)

    assert calls == []
    assert plan.reservation_state is None
    assert plan.receipt["status"] == "READY_FOR_MODEL"
    assert plan.receipt["blockers"] == ["MODEL_ADAPTER_NOT_CONFIGURED"]
    assert plan.receipt["budget"]["reservation_charge_derived"] is False
    assert plan.receipt["permissions"] == driver._NO_AUTHORITY
    assert plan.receipt["capacity_gate"]["runner_start_eligible"] is False
    assert plan.receipt["invariants"]["partial_economics_allowed"] is False


def test_configured_adapter_only_derives_pre_call_reservation_transition(
    monkeypatch,
) -> None:
    plan, calls = _plan(monkeypatch, adapter=True)

    assert len(calls) == 1
    assert calls[0]["expected_parent_state_sha256"] == "8" * 64
    assert calls[0]["request_sha256"] == hashlib.sha256(
        plan.model_request_bytes
    ).hexdigest()
    assert plan.reservation_state is not None
    assert plan.receipt["status"] == "RESERVATION_TRANSITION_SEALED"
    assert plan.receipt["blockers"] == [
        "RESERVATION_STATE_NOT_DURABLY_APPENDED"
    ]
    assert plan.receipt["budget"]["reservation_attempt_charge"] == 1
    assert plan.receipt["budget"]["reservation_model_invocation_charge"] == 1
    assert plan.receipt["budget"]["reservation_charge_durable"] is False
    bindings = plan.receipt["bindings"]
    assert bindings["provider_id"] == "openai-api"
    assert bindings["model_id"] == "gpt-fixed"
    assert bindings["scorer_contract"] == "QR_DOJO_BOT_TRAINER_EVALUATION_V1"
    assert bindings["corpus_sha256"] == "a" * 64
    assert bindings["code_source_bundle_sha256"] == "d" * 64
    assert len(bindings["terminal_fixed_denominator_sha256"]) == 64


def test_partial_or_unbound_economics_cannot_reach_prompt_seal(monkeypatch) -> None:
    state = _state()
    packet = _packet(state)
    packet["current_run"]["status"] = "RUNNING"
    _install_context(monkeypatch, state, packet)

    with pytest.raises(
        driver.DojoTrainerGenerationDriverError,
        match="partial or non-terminal",
    ):
        driver.plan_terminal_to_next_generation(
            tuning_state_events_dir=Path("state"),
            lineage_events_dir=Path("lineage"),
            artifact_root=Path("artifacts"),
            trainer_packet=packet,
            prompt_text="candidate",
            provider_id="provider",
            model_id="model",
            health=_health(),
            prepared_at_utc=NOW,
        )

    packet = _packet(state)
    packet["source_bindings"]["lineage_tip_sha256"] = "9" * 64
    _install_context(monkeypatch, state, packet)
    with pytest.raises(
        driver.DojoTrainerGenerationDriverError,
        match="exact latest terminal",
    ):
        driver.plan_terminal_to_next_generation(
            tuning_state_events_dir=Path("state"),
            lineage_events_dir=Path("lineage"),
            artifact_root=Path("artifacts"),
            trainer_packet=packet,
            prompt_text="candidate",
            provider_id="provider",
            model_id="model",
            health=_health(),
            prepared_at_utc=NOW,
        )


def test_external_response_is_charged_and_duplicate_is_not_in_denominator(
    monkeypatch,
) -> None:
    plan, _ = _plan(monkeypatch, adapter=True)
    assert plan.reservation_state is not None
    response = b'[{"submission_id":"new"},{"submission_id":"duplicate"}]\n'

    response_state = copy.deepcopy(plan.reservation_state)
    response_state["state_sha256"] = "4" * 64
    response_state["attempts"][-1]["invocations"][-1].update(
        {
            "proposal_slot_charge": 2,
            "submissions": [
                {
                    "submission_id": "new",
                    "status": "ACCEPTED",
                    "candidate_id": "qr-a2-new",
                },
                {
                    "submission_id": "duplicate",
                    "status": "DUPLICATE",
                    "candidate_id": None,
                },
            ],
        }
    )
    monkeypatch.setattr(
        driver,
        "record_model_response",
        lambda value, **kwargs: copy.deepcopy(response_state),
    )
    captured: dict = {}

    def materialize(value, **kwargs):
        captured.update(kwargs)
        return {"study_sha256": "5" * 64, "study": {"candidates": ["new"]}}

    monkeypatch.setattr(driver, "materialize_next_study", materialize)

    reduced = driver.reduce_model_response_to_next_study(
        plan,
        response_bytes=response,
        submissions=[{"submission_id": "new"}, {"submission_id": "duplicate"}],
        recorded_at_utc="2026-07-20T00:01:00Z",
    )

    assert reduced.receipt["proposal_slot_charge"] == 2
    assert reduced.receipt["budget_after"]["attempts_consumed"] == 2
    assert reduced.receipt["budget_after"]["model_invocation_count"] == 1
    assert reduced.receipt["budget_after"]["proposal_slots_consumed"] == 3
    assert reduced.receipt["accepted_candidate_ids"] == ["qr-a2-new"]
    assert reduced.receipt["duplicate_submission_ids"] == ["duplicate"]
    assert reduced.receipt["fixed_candidate_denominator"] == 1
    assert reduced.receipt["capacity_gate"]["runner_start_eligible"] is False
    assert captured["request_artifacts"] == {
        plan.receipt["invocation_id"]: plan.model_request_bytes
    }
    assert captured["response_artifacts"] == {
        plan.receipt["invocation_id"]: response
    }


def test_raw_response_rows_cannot_be_omitted_from_caller_submissions(
    monkeypatch,
) -> None:
    plan, _ = _plan(monkeypatch, adapter=True)
    assert plan.reservation_state is not None
    raw_rows = [
        {"submission_id": "winner"},
        {"submission_id": "omitted-loser"},
    ]
    response = json.dumps(raw_rows, separators=(",", ":")).encode() + b"\n"
    captured: dict = {}
    response_state = copy.deepcopy(plan.reservation_state)
    response_state["state_sha256"] = "a" * 64
    response_state["attempts"][-1]["invocations"][-1].update(
        {
            "proposal_slot_charge": 2,
            "submissions": [
                {
                    "submission_id": "winner",
                    "status": "ACCEPTED",
                    "candidate_id": "qr-a2-winner",
                },
                {
                    "submission_id": "omitted-loser",
                    "status": "INVALID",
                    "candidate_id": None,
                },
            ],
        }
    )

    def record(value, **kwargs):
        captured.update(kwargs)
        return copy.deepcopy(response_state)

    monkeypatch.setattr(driver, "record_model_response", record)
    monkeypatch.setattr(
        driver,
        "materialize_next_study",
        lambda *args, **kwargs: pytest.fail(
            "mismatched caller view must not seal a study"
        ),
    )

    reduced = driver.reduce_model_response_to_next_study(
        plan,
        response_bytes=response,
        submissions=[raw_rows[0]],
        recorded_at_utc="2026-07-20T00:01:00Z",
    )

    assert captured["submissions"] == raw_rows
    assert (
        reduced.response_state["attempts"][-1]["invocations"][-1][
            "proposal_slot_charge"
        ]
        == 2
    )
    assert reduced.receipt["proposal_slot_charge"] == 2
    assert reduced.receipt["raw_response_row_count"] == 2
    assert reduced.receipt["caller_submissions_match_raw_response"] is False
    assert reduced.receipt["submissions_source"] == "STRICT_RAW_RESPONSE_BYTES"
    assert reduced.receipt["status"] == "REVIEW_REQUIRED"
    assert reduced.sealed_study is None


def test_empty_raw_response_consumes_one_slot_and_cannot_seal_study(
    monkeypatch,
) -> None:
    plan, _ = _plan(monkeypatch, adapter=True)
    assert plan.reservation_state is not None
    captured: dict = {}
    response_state = copy.deepcopy(plan.reservation_state)
    response_state["state_sha256"] = "b" * 64
    response_state["attempts"][-1]["invocations"][-1].update(
        {
            "proposal_slot_charge": 1,
            "submissions": [
                {
                    "submission_id": "invalid-empty-response",
                    "status": "INVALID",
                    "candidate_id": None,
                }
            ],
        }
    )

    def record(value, **kwargs):
        captured.update(kwargs)
        return copy.deepcopy(response_state)

    monkeypatch.setattr(driver, "record_model_response", record)
    monkeypatch.setattr(
        driver,
        "materialize_next_study",
        lambda *args, **kwargs: pytest.fail("empty response must not seal a study"),
    )

    reduced = driver.reduce_model_response_to_next_study(
        plan,
        response_bytes=b"",
        submissions=[],
        recorded_at_utc="2026-07-20T00:01:00Z",
    )

    assert captured["submissions"] == b""
    assert reduced.receipt["response_parse_status"] == "INVALID_STRICT_JSON"
    assert reduced.receipt["raw_response_row_count"] is None
    assert reduced.receipt["proposal_slot_charge"] == 1
    assert reduced.receipt["status"] == "REVIEW_REQUIRED"
    assert reduced.sealed_study is None


def test_response_without_adapter_or_safe_study_fails_closed(monkeypatch) -> None:
    no_adapter, _ = _plan(monkeypatch, adapter=False)
    with pytest.raises(
        driver.DojoTrainerGenerationDriverError,
        match="MODEL_ADAPTER_NOT_CONFIGURED",
    ):
        driver.reduce_model_response_to_next_study(
            no_adapter,
            response_bytes=b"[]\n",
            submissions=[],
            recorded_at_utc="2026-07-20T00:01:00Z",
        )

    plan, _ = _plan(monkeypatch, adapter=True)
    assert plan.reservation_state is not None
    charged_state = copy.deepcopy(plan.reservation_state)
    charged_state["state_sha256"] = "6" * 64
    charged_state["attempts"][-1]["invocations"][-1].update(
        {"proposal_slot_charge": 1, "submissions": []}
    )
    monkeypatch.setattr(
        driver,
        "record_model_response",
        lambda value, **kwargs: copy.deepcopy(charged_state),
    )
    monkeypatch.setattr(
        driver,
        "materialize_next_study",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("no accepted unique proposal")
        ),
    )
    rejected = driver.reduce_model_response_to_next_study(
        plan,
        response_bytes=b"[]\n",
        submissions=[],
        recorded_at_utc="2026-07-20T00:01:00Z",
    )
    assert rejected.sealed_study is None
    assert rejected.receipt["status"] == "REVIEW_REQUIRED"
    assert rejected.receipt["blockers"] == [
        "NEXT_STUDY_MATERIALIZATION_REJECTED"
    ]
    assert rejected.receipt["next_action"] == (
        "PERSIST_CHARGED_RESPONSE_STATE_THEN_REVIEW"
    )


def test_capacity_wrapper_never_turns_eligibility_into_runner_authority(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        driver,
        "plan_immutable_generation",
        lambda **kwargs: {
            "action": "MANUAL_START_ELIGIBLE",
            "permissions": {
                "runner_invocation_allowed": False,
                "automatic_runner_start_allowed": False,
                "live_permission": False,
                "order_authority": "NONE",
            },
        },
    )

    decision = driver.plan_reserved_generation_capacity(
        tuning_state_events_dir=Path("state"),
        lineage_events_dir=Path("lineage"),
        artifact_root=Path("artifacts"),
        health={"process_state": "RESERVED"},
        resource_gate_request={"contract": "request"},
        decision_at_utc=NOW,
    )

    assert decision["action"] == "MANUAL_START_ELIGIBLE"
    assert decision["permissions"]["runner_invocation_allowed"] is False

    monkeypatch.setattr(
        driver,
        "plan_immutable_generation",
        lambda **kwargs: {
            "action": "MONITOR_HEALTH_ONLY",
            "permissions": {
                "runner_invocation_allowed": False,
                "automatic_runner_start_allowed": False,
                "live_permission": False,
                "order_authority": "NONE",
            },
        },
    )
    with pytest.raises(
        driver.DojoTrainerGenerationDriverError,
        match="requires a lineage-sealed reserved run dispatch",
    ):
        driver.plan_reserved_generation_capacity(
            tuning_state_events_dir=Path("state"),
            lineage_events_dir=Path("lineage"),
            artifact_root=Path("artifacts"),
            health={"process_state": "RUNNING"},
            resource_gate_request={"contract": "request"},
            decision_at_utc=NOW,
        )
