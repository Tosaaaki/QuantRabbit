from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from quant_rabbit.dojo_ai_trainer_packet import (
    PACKET_CONTRACT,
    TRAINER_READBACK_KINDS,
    _ALLOWED_MUTATIONS,
    _EVALUATION_OBLIGATIONS,
    _FORBIDDEN_MUTATIONS,
    _LIMITATIONS,
    canonical_packet_bytes,
    canonical_packet_sha256,
)
from quant_rabbit.dojo_ai_tuning_state import (
    ENVELOPE_CONTRACT,
    STATE_CONTRACT,
    fixed_envelope_from_sealed_study,
    verify_tuning_state,
)
from quant_rabbit.dojo_bot_catalog import catalog_manifest
from quant_rabbit.dojo_bot_trainer import (
    EVALUATION_CONTRACT,
    PROPOSAL_CONTRACT,
    STUDY_CONTRACT,
    seal_candidate_proposal,
    seal_study,
)
from quant_rabbit.dojo_next_study_builder import (
    DojoNextStudyBuilderError,
    PROMPT_ARTIFACT_CONTRACT,
    UNVERIFIED_MODEL_CLAIM,
    UNVERIFIED_PROVIDER_ATTESTATION,
    materialize_next_study,
)


PAIRS = ["AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_JPY"]
SOURCES = {
    "bots/lab_bot.py": "a" * 64,
    "src/quant_rabbit/dojo_bot_trainer.py": "b" * 64,
}
STATE_AUTHORITY = {
    "automation_ready": False,
    "research_train_only": True,
    "holdout_access_allowed": False,
    "forward_access_allowed": False,
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}
SEALED_AUTHORITY = {
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _raw_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _prompt_bytes(*, packet_sha256: str, request_sha256: str) -> bytes:
    return _json_bytes(
        {
            "contract": PROMPT_ARTIFACT_CONTRACT,
            "schema_version": 1,
            "classification": "UNVERIFIED_CALLER_ARTIFACT",
            "trainer_packet_sha256": packet_sha256,
            "model_request_sha256": request_sha256,
            "prompt_text": "Propose one bounded next-attempt TRAIN candidate set.",
            "model_claim": UNVERIFIED_MODEL_CLAIM,
            "provider_attestation": UNVERIFIED_PROVIDER_ATTESTATION,
            **SEALED_AUTHORITY,
        }
    )


def _raw_proposal(
    candidate_id: str,
    seed: int,
    *,
    family: str = "spike_fade",
    config_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "signal": family,
        "pairs": PAIRS,
        "tp_atr": 3.0 + seed / 10,
        "sl_pips": 25.0,
        "ceiling_min": 60,
        "max_concurrent_per_pair": 1,
        "global_max_concurrent": 4,
        "per_pos_lev": 5.0,
        "atr_floor_pips": 0.5,
        "exit_policy": "FIXED",
        "order_authority": "NONE",
        "live_permission": False,
        "external_broker_mutation_allowed": False,
    }
    config.update(config_updates or {})
    return {
        "contract": PROPOSAL_CONTRACT,
        "schema_version": 1,
        "candidate_id": candidate_id,
        "family": family,
        "hypothesis": f"Bounded causal stale-capital candidate {seed}.",
        "config": config,
        "risk_increase": False,
    }


def _baseline_study() -> dict[str, Any]:
    proposal = seal_candidate_proposal(_raw_proposal("qr-a1-base", 1))
    return seal_study(
        {
            "contract": STUDY_CONTRACT,
            "schema_version": 1,
            "study_id": "qr-baseline-attempt-1",
            "window_role": "TRAIN",
            "initial_balance_jpy": 200_000.0,
            "trade_pairs": PAIRS,
            "feed_pairs": PAIRS,
            "candidates": [proposal],
            "window": {
                "start_utc": "2025-06-01T00:00:00Z",
                "end_utc": "2025-07-01T00:00:00Z",
                "corpus_id": "worn-train-2025-06",
                "corpus_sha256": "c" * 64,
                "evidence_tier": "WORN_TRAIN",
            },
            "cost_arms": {
                "BASE": {
                    "slippage_pips_per_fill": 0.0,
                    "financing_pips_per_day": 0.0,
                    "recorded_spread_multiplier": 1.0,
                },
                "STRESS": {
                    "slippage_pips_per_fill": 0.3,
                    "financing_pips_per_day": 0.8,
                    "recorded_spread_multiplier": 1.0,
                },
            },
            "proposer_evidence": {
                "prompt_sha256": "d" * 64,
                "input_sha256": "e" * 64,
                "raw_response_sha256": "f" * 64,
                "model_claim": "baseline",
                "provider_attestation": "UNVERIFIED",
            },
            "search_budget": {
                "attempt_ordinal": 1,
                "total_attempts_in_lineage": 3,
                "max_candidates": 1,
            },
            "thresholds": {
                "normal_mtm_drawdown_max": 0.10,
                "stress_mtm_drawdown_max": 0.15,
                "peak_margin_usage_max": 0.45,
                "margin_reject_rate_max": 0.10,
                "cost_retention_min": 0.50,
                "pair_positive_share_max": 0.50,
                "pair_hhi_max": 0.40,
            },
        },
        SOURCES,
    )


def _result_binding(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "registry_id": "qr-ai-tuning-lineage",
        "lineage_prefix": "qr-",
        "attempt_ordinal": 1,
        "study_sha256": baseline["study_sha256"],
        "evaluation_sha256": "1" * 64,
        "evaluation_artifact_sha256": "2" * 64,
        "evaluation_artifact_size_bytes": 123,
        "result_event_sha256": "3" * 64,
        "result_event_sequence": 2,
        "lineage_tip_sha256": "3" * 64,
    }


def _packet(baseline: dict[str, Any]) -> dict[str, Any]:
    envelope = fixed_envelope_from_sealed_study(baseline)
    prior = _result_binding(baseline)
    body = {
        "contract": PACKET_CONTRACT,
        "schema_version": 1,
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "source_bindings": {
            "registry_id": prior["registry_id"],
            "lineage_prefix": prior["lineage_prefix"],
            "attempt_ordinal": 1,
            "study_sha256": prior["study_sha256"],
            "run_sha256": "4" * 64,
            "evaluation_sha256": prior["evaluation_sha256"],
            "evaluation_artifact_sha256": prior["evaluation_artifact_sha256"],
            "lineage_result_event_sha256": prior["result_event_sha256"],
            "lineage_tip_sha256": prior["lineage_tip_sha256"],
            "tuning_state_sha256": "5" * 64,
            "fixed_envelope_sha256": envelope["envelope_sha256"],
            "external_witness_status": "NOT_CONFIGURED",
            "exact_result_binding_verified": True,
        },
        "fixed_environment": {
            "window_role": envelope["window_role"],
            "window": envelope["window"],
            "initial_balance_jpy": envelope["initial_balance_jpy"],
            "trade_pairs": envelope["trade_pairs"],
            "feed_pairs": envelope["feed_pairs"],
            "intrabar_paths": envelope["intrabar_paths"],
            "cost_arms": envelope["cost_arms"],
            "thresholds": envelope["thresholds"],
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
        "mutation_policy": {
            "allowed": list(_ALLOWED_MUTATIONS),
            "forbidden": list(_FORBIDDEN_MUTATIONS),
            "risk_increase_required_value": False,
            "catalog_validation_required": True,
            "fixed_envelope_mutation_allowed": False,
        },
        "evaluation_obligations": list(_EVALUATION_OBLIGATIONS),
        "current_run": {
            "status": "COMPLETE",
            "fixed_denominator": {"expected_cell_count": 4},
            "candidate_ids": ["qr-a1-base"],
            "diagnostic_ranking": [],
            "rank_eligible_candidate_ids": [],
            "unranked_candidate_ids": ["qr-a1-base"],
        },
        "candidates": [
            {
                "candidate_id": baseline["study"]["candidates"][0]["candidate_id"],
                "family": baseline["study"]["candidates"][0]["family"],
                "config_sha256": baseline["study"]["candidates"][0]["config_sha256"],
                "config": baseline["study"]["candidates"][0]["config"],
            }
        ],
        "cells": [
            {
                "candidate_id": "qr-a1-base",
                "intrabar": intrabar,
                "cost_arm": cost_arm,
            }
            for intrabar in ("OHLC", "OLHC")
            for cost_arm in ("BASE", "STRESS")
        ],
        "base_stress_comparisons": [],
        "ohlc_olhc_comparisons": [],
        "failed_coordinates": [],
        "previous_attempts": [],
        "drive_evidence_refs": sorted(
            [
                {
                    "artifact_kind": kind,
                    "drive_file_id": f"nextStudyDriveFile{index:02d}",
                    "drive_parent_id": "nextStudyDriveParent",
                    "content_sha256": f"{index}" * 64,
                    "content_size_bytes": 1000 + index,
                    "version": str(400 + index),
                    "head_revision_id": f"nextStudyRevision{index:02d}",
                    "readback_sha256": "9" * 64,
                    "remote_verified": True,
                }
                for index, kind in enumerate(TRAINER_READBACK_KINDS, start=1)
            ],
            key=lambda row: (row["artifact_kind"], row["drive_file_id"]),
        ),
        "limitations": list(_LIMITATIONS),
        **SEALED_AUTHORITY,
    }
    return {**body, "packet_sha256": canonical_packet_sha256(body)}


def _fixture(
    *,
    candidate_count: int = 2,
    proposals: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes], bytes, dict[str, Any]]:
    baseline = _baseline_study()
    packet = _packet(baseline)
    request_raw = canonical_packet_bytes(packet) + b"\n"
    candidate_proposals = proposals or [
        _raw_proposal(f"qr-a2-new-{index}", index + 2)
        for index in range(candidate_count)
    ]
    candidate_count = len(candidate_proposals)
    submissions = [
        {
            "submission_id": f"submission-{index}",
            "raw_proposal_sha256": _sha(proposal),
            "proposal": proposal,
            "validation_errors": [],
        }
        for index, proposal in enumerate(candidate_proposals)
    ]
    response_raw = _json_bytes(submissions)
    sealed_proposals = [
        seal_candidate_proposal(proposal) for proposal in candidate_proposals
    ]
    response_sha = _raw_sha(response_raw)
    invocation = {
        "invocation_id": "model-a2-1",
        "request_sha256": _raw_sha(request_raw),
        "reserved_at_utc": "2026-07-20T00:01:00Z",
        "response_sha256": response_sha,
        "response_input_sha256": _sha(
            {"response_sha256": response_sha, "submissions": submissions}
        ),
        "response_recorded_at_utc": "2026-07-20T00:02:00Z",
        "proposal_slot_charge": candidate_count,
        "submissions": [
            {
                "submission_id": raw["submission_id"],
                "raw_proposal_sha256": raw["raw_proposal_sha256"],
                "status": "ACCEPTED",
                "validation_errors": [],
                "candidate_id": sealed["candidate_id"],
                "proposal_sha256": sealed["proposal_sha256"],
                "executable_identity_sha256": sealed["config_sha256"],
                "sealed_proposal": sealed,
            }
            for raw, sealed in zip(submissions, sealed_proposals, strict=True)
        ],
    }
    prior = _result_binding(baseline)
    envelope = fixed_envelope_from_sealed_study(baseline)
    state_body = {
        "contract": STATE_CONTRACT,
        "schema_version": 1,
        "registry_id": prior["registry_id"],
        "lineage_prefix": prior["lineage_prefix"],
        "fixed_envelope": envelope,
        "fixed_envelope_sha256": envelope["envelope_sha256"],
        "revision": 2,
        "previous_state_sha256": "6" * 64,
        "initialized_at_utc": "2026-07-20T00:00:00Z",
        "last_transition_at_utc": "2026-07-20T00:02:00Z",
        "initial_attempts_consumed": 1,
        "initial_proposal_slots_consumed": 1,
        "attempts": [
            {
                "attempt_ordinal": 2,
                "prior_result_binding": prior,
                "envelope_sha256": envelope["envelope_sha256"],
                "phase": "COLLECTING_PROPOSALS",
                "invocations": [invocation],
                "dispatch": None,
                "terminal": None,
            }
        ],
        "max_attempts": 3,
        "max_proposal_slots": 14,
        "global_executable_identity_sha256s": sorted(
            [
                baseline["study"]["candidates"][0]["config_sha256"],
                *[row["config_sha256"] for row in sealed_proposals],
            ]
        ),
        "last_terminal_result_binding": prior,
        "phase": "COLLECTING_PROPOSALS",
        "terminal_reason": None,
        **STATE_AUTHORITY,
    }
    state = {**state_body, "state_sha256": _sha(state_body)}
    verify_tuning_state(state)
    prompt = _prompt_bytes(
        packet_sha256=packet["packet_sha256"],
        request_sha256=_raw_sha(request_raw),
    )
    return (
        state,
        {"model-a2-1": request_raw},
        {"model-a2-1": response_raw},
        prompt,
        packet,
    )


def _rehash_state(state: dict[str, Any]) -> None:
    state["state_sha256"] = _sha(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )


def _reseal_packet(packet: dict[str, Any]) -> bytes:
    packet["packet_sha256"] = canonical_packet_sha256(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    )
    return canonical_packet_bytes(packet) + b"\n"


def _rebind_packet_request(
    state: dict[str, Any],
    requests: dict[str, bytes],
    packet: dict[str, Any],
) -> bytes:
    request_raw = _reseal_packet(packet)
    requests["model-a2-1"] = request_raw
    state["attempts"][-1]["invocations"][0]["request_sha256"] = _raw_sha(request_raw)
    _rehash_state(state)
    return request_raw


def test_materializes_exact_accepted_denominator_and_fixed_envelope() -> None:
    state, requests, responses, prompt, _ = _fixture(candidate_count=2)

    sealed = materialize_next_study(
        state,
        request_artifacts=requests,
        response_artifacts=responses,
        prompt_artifact=prompt,
    )

    expected = sorted(
        (
            row["sealed_proposal"]
            for row in state["attempts"][-1]["invocations"][0]["submissions"]
        ),
        key=lambda row: row["candidate_id"],
    )
    assert sealed["study"]["candidates"] == expected
    assert fixed_envelope_from_sealed_study(sealed) == state["fixed_envelope"]
    assert sealed["study"]["search_budget"] == {
        "attempt_ordinal": 2,
        "total_attempts_in_lineage": 3,
        "max_candidates": 2,
    }
    assert sealed["study"]["proposer_evidence"] == {
        "prompt_sha256": _raw_sha(prompt),
        "input_sha256": _raw_sha(requests["model-a2-1"]),
        "raw_response_sha256": _raw_sha(responses["model-a2-1"]),
        "model_claim": UNVERIFIED_MODEL_CLAIM,
        "provider_attestation": UNVERIFIED_PROVIDER_ATTESTATION,
    }
    for key, expected_value in SEALED_AUTHORITY.items():
        assert sealed[key] == expected_value


def test_rejects_false_risk_increase_claim_against_same_family_parent() -> None:
    proposal = _raw_proposal("qr-a2-risk-lie", 2)
    proposal["config"].pop("max_concurrent_per_pair")
    proposal["config"]["max_concurrent"] = 2
    state, requests, responses, prompt, _ = _fixture(proposals=[proposal])

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="REVIEW_REQUIRED_RELATIVE_RISK_INCREASE",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_rejects_absolute_hard_risk_envelope_before_relative_comparison() -> None:
    proposal = _raw_proposal(
        "qr-a2-hard-risk",
        2,
        config_updates={"per_pos_lev": 6.0},
    )
    state, requests, responses, prompt, _ = _fixture(proposals=[proposal])

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="REVIEW_REQUIRED_ABSOLUTE_RISK_HARD_ENVELOPE_EXCEEDED",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_rejects_new_family_without_reviewed_prior_baseline() -> None:
    proposal = _raw_proposal("qr-a2-new-family", 2, family="burst")
    state, requests, responses, prompt, _ = _fixture(proposals=[proposal])

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="REVIEW_REQUIRED_NEW_FAMILY_BASELINE_REVISION_REQUIRED",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_rejects_dynamic_unbounded_initial_stop_before_parent_admission() -> None:
    proposal = _raw_proposal(
        "qr-a2-dynamic-stop",
        2,
        family="session_open_range_break",
        config_updates={
            "tp_atr": None,
            "sl_pips": None,
            "session_buffer_atr": 0.25,
            "session_tp_range": 1.5,
            "session_sl_range": 0.75,
        },
    )
    state, requests, responses, prompt, _ = _fixture(proposals=[proposal])

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="REVIEW_REQUIRED_BOUNDED_INITIAL_STOP_REQUIRED",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_accepts_componentwise_risk_decrease_against_same_family_parent() -> None:
    proposal = _raw_proposal(
        "qr-a2-safer",
        2,
        config_updates={
            "per_pos_lev": 4.0,
            "global_max_concurrent": 3,
            "sl_pips": 20.0,
        },
    )
    state, requests, responses, prompt, _ = _fixture(proposals=[proposal])

    sealed = materialize_next_study(
        state,
        request_artifacts=requests,
        response_artifacts=responses,
        prompt_artifact=prompt,
    )

    config = sealed["study"]["candidates"][0]["config"]
    assert config["per_pos_lev"] == 4.0
    assert config["max_concurrent_per_pair"] == 1
    assert config["global_max_concurrent"] == 3
    assert config["sl_pips"] == 20.0
    for key, expected_value in SEALED_AUTHORITY.items():
        assert sealed[key] == expected_value


def test_previous_attempt_candidate_is_never_a_relative_risk_parent() -> None:
    proposal = _raw_proposal("qr-a2-burst", 2, family="burst")
    state, requests, responses, prompt, packet = _fixture(proposals=[proposal])
    historical = seal_candidate_proposal(
        _raw_proposal("qr-historical-burst", 7, family="burst")
    )
    packet["previous_attempts"] = [
        {
            "attempt_ordinal": 0,
            "study_sha256": "7" * 64,
            "evaluation_sha256": "8" * 64,
            "candidate_count": 1,
            "candidates": [
                {
                    "candidate_id": historical["candidate_id"],
                    "family": historical["family"],
                    "proposal_sha256": historical["proposal_sha256"],
                    "config_sha256": historical["config_sha256"],
                    "config": historical["config"],
                    "evaluation": {},
                }
            ],
        }
    ]
    _rebind_packet_request(state, requests, packet)

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="REVIEW_REQUIRED_NEW_FAMILY_BASELINE_REVISION_REQUIRED",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_candidate_must_not_exceed_any_current_same_family_parent() -> None:
    proposal = _raw_proposal(
        "qr-a2-safe-except-ceiling",
        2,
        config_updates={
            "per_pos_lev": 4.0,
            "global_max_concurrent": 3,
            "sl_pips": 20.0,
            "ceiling_min": 60,
        },
    )
    state, requests, responses, prompt, packet = _fixture(proposals=[proposal])
    stricter_parent = seal_candidate_proposal(
        _raw_proposal(
            "qr-a1-stricter-ceiling",
            9,
            config_updates={
                "per_pos_lev": 4.0,
                "global_max_concurrent": 3,
                "sl_pips": 20.0,
                "ceiling_min": 50,
            },
        )
    )
    packet["candidates"].append(
        {
            "candidate_id": stricter_parent["candidate_id"],
            "family": stricter_parent["family"],
            "config_sha256": stricter_parent["config_sha256"],
            "config": stricter_parent["config"],
        }
    )
    packet["current_run"]["fixed_denominator"]["expected_cell_count"] = 8
    packet["current_run"]["candidate_ids"].append(stricter_parent["candidate_id"])
    packet["current_run"]["unranked_candidate_ids"].append(
        stricter_parent["candidate_id"]
    )
    packet["cells"].extend(
        {
            "candidate_id": stricter_parent["candidate_id"],
            "intrabar": intrabar,
            "cost_arm": cost_arm,
        }
        for intrabar in ("OHLC", "OLHC")
        for cost_arm in ("BASE", "STRESS")
    )
    _rebind_packet_request(state, requests, packet)

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="REVIEW_REQUIRED_RELATIVE_RISK_INCREASE",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("classification", "VERIFIED_PROVIDER_ARTIFACT"),
        ("model_claim", "gpt-5.6"),
        ("provider_attestation", "OPENAI_VERIFIED"),
        ("model_request_sha256", "9" * 64),
    ],
)
def test_rejects_prompt_provenance_or_request_binding_claims(
    field: str, value: str
) -> None:
    state, requests, responses, prompt, _ = _fixture(candidate_count=1)
    prompt_value = json.loads(prompt)
    prompt_value[field] = value

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="prompt artifact provenance or request binding drifted",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=_json_bytes(prompt_value),
        )


def test_caller_cannot_override_fixed_model_provenance() -> None:
    state, requests, responses, prompt, _ = _fixture(candidate_count=1)

    with pytest.raises(TypeError, match="model_claim"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
            model_claim="gpt-5.6",  # type: ignore[call-arg]
        )


def test_rejects_fixed_environment_or_budget_drift_even_when_rehashed() -> None:
    state, requests, responses, prompt, packet = _fixture()
    packet["fixed_environment"]["window"]["corpus_sha256"] = "9" * 64
    request_raw = _reseal_packet(packet)
    requests["model-a2-1"] = request_raw
    state["attempts"][-1]["invocations"][0]["request_sha256"] = _raw_sha(request_raw)
    _rehash_state(state)

    with pytest.raises(DojoNextStudyBuilderError, match="fixed environment"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )

    state, requests, responses, prompt, packet = _fixture()
    packet["search_budget"]["proposal_slots_remaining"] = 12
    request_raw = _reseal_packet(packet)
    requests["model-a2-1"] = request_raw
    state["attempts"][-1]["invocations"][0]["request_sha256"] = _raw_sha(request_raw)
    _rehash_state(state)
    with pytest.raises(DojoNextStudyBuilderError, match="search budget"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_rejects_request_response_and_accepted_membership_drift() -> None:
    state, requests, responses, prompt, _ = _fixture()
    requests["model-a2-1"] += b" "
    with pytest.raises(DojoNextStudyBuilderError, match="request SHA-256"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )

    state, requests, responses, prompt, _ = _fixture()
    raw = json.loads(responses["model-a2-1"])
    raw.pop()
    changed_response = _json_bytes(raw)
    responses["model-a2-1"] = changed_response
    invocation = state["attempts"][-1]["invocations"][0]
    invocation["response_sha256"] = _raw_sha(changed_response)
    _rehash_state(state)
    with pytest.raises(DojoNextStudyBuilderError, match="response denominator"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_rejects_omitted_raw_response_row_from_charged_denominator() -> None:
    state, requests, responses, prompt, _ = _fixture(candidate_count=1)
    raw = json.loads(responses["model-a2-1"])
    raw.append(
        {
            "submission_id": "omitted-loser",
            "raw_proposal_sha256": "0" * 64,
            "proposal": None,
            "validation_errors": ["MODEL_PROPOSAL_UNPARSEABLE"],
        }
    )
    changed_response = _json_bytes(raw)
    responses["model-a2-1"] = changed_response
    invocation = state["attempts"][-1]["invocations"][0]
    invocation["response_sha256"] = _raw_sha(changed_response)
    invocation["response_input_sha256"] = _sha(
        {
            "response_sha256": invocation["response_sha256"],
            "submissions": [
                {
                    "submission_id": "submission-0",
                    "raw_proposal_sha256": raw[0]["raw_proposal_sha256"],
                    "proposal": raw[0]["proposal"],
                    "validation_errors": [],
                }
            ],
        }
    )
    _rehash_state(state)

    with pytest.raises(DojoNextStudyBuilderError, match="response denominator"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_rejects_multiple_response_candidate_mix() -> None:
    state, requests, responses, prompt, _ = _fixture(candidate_count=2)
    first = state["attempts"][-1]["invocations"][0]
    first_raw = json.loads(responses.pop("model-a2-1"))
    moved_submission = first["submissions"].pop()
    moved_raw = first_raw.pop()
    first_response = _json_bytes(first_raw)
    first["response_sha256"] = _raw_sha(first_response)
    first["proposal_slot_charge"] = 1
    responses["model-a2-1"] = first_response
    second_response = _json_bytes([moved_raw])
    second = {
        "invocation_id": "model-a2-2",
        "request_sha256": first["request_sha256"],
        "reserved_at_utc": "2026-07-20T00:03:00Z",
        "response_sha256": _raw_sha(second_response),
        "response_input_sha256": "7" * 64,
        "response_recorded_at_utc": "2026-07-20T00:04:00Z",
        "proposal_slot_charge": 1,
        "submissions": [moved_submission],
    }
    state["attempts"][-1]["invocations"].append(second)
    state["revision"] = 4
    state["last_transition_at_utc"] = "2026-07-20T00:04:00Z"
    requests["model-a2-2"] = requests["model-a2-1"]
    responses["model-a2-2"] = second_response
    _rehash_state(state)

    with pytest.raises(
        DojoNextStudyBuilderError,
        match="AI tuning state is invalid",
    ):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_rejects_catalog_or_global_duplicate_forgery() -> None:
    state, requests, responses, prompt, _ = _fixture(candidate_count=1)
    accepted = state["attempts"][-1]["invocations"][0]["submissions"][0]
    accepted["sealed_proposal"]["risk_increase"] = True
    accepted["sealed_proposal"]["proposal_sha256"] = _sha(
        {
            key: value
            for key, value in accepted["sealed_proposal"].items()
            if key != "proposal_sha256"
        }
    )
    _rehash_state(state)
    with pytest.raises(DojoNextStudyBuilderError, match="tuning state is invalid"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )

    state, requests, responses, prompt, _ = _fixture(candidate_count=1)
    baseline_identity = _baseline_study()["study"]["candidates"][0]["config_sha256"]
    accepted = state["attempts"][-1]["invocations"][0]["submissions"][0]
    accepted["executable_identity_sha256"] = baseline_identity
    accepted["sealed_proposal"]["config_sha256"] = baseline_identity
    state["global_executable_identity_sha256s"] = [baseline_identity]
    _rehash_state(state)
    with pytest.raises(DojoNextStudyBuilderError, match="tuning state is invalid"):
        materialize_next_study(
            state,
            request_artifacts=requests,
            response_artifacts=responses,
            prompt_artifact=prompt,
        )


def test_contract_constants_stay_bound_to_expected_versions() -> None:
    baseline = _baseline_study()
    envelope = fixed_envelope_from_sealed_study(baseline)
    assert envelope["contract"] == ENVELOPE_CONTRACT
    assert envelope["scorer_contract"] == EVALUATION_CONTRACT
