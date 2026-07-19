from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_ai_tuning_state import (
    DojoAITuningStateError,
    abandon_incomplete_lineage,
    append_state_transition,
    bind_terminal_evaluation,
    fixed_envelope_from_sealed_study,
    initialize_state_store,
    initialize_tuning_state,
    mark_incomplete_run,
    mark_run_dispatched,
    record_model_response,
    reserve_model_invocation,
    reserve_run_dispatch,
    status_artifact,
    verify_state_store,
    verify_tuning_state,
)
from quant_rabbit.dojo_bot_catalog import bot_config_risk_vector
from quant_rabbit.dojo_bot_trainer import (
    EVALUATION_CONTRACT,
    PROPOSAL_CONTRACT,
    STUDY_CONTRACT,
    seal_candidate_proposal,
    seal_study,
)
from quant_rabbit.dojo_candidate_lineage_registry import (
    CandidateLineageSnapshot,
    bind_result,
    initialize_registry,
    seal_study_attempt,
)


PAIRS = ["AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_JPY"]
SOURCES = {
    "bots/lab_bot.py": "a" * 64,
    "src/quant_rabbit/dojo_bot_trainer.py": "b" * 64,
}


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _rehash(state: dict) -> None:
    state["state_sha256"] = _sha(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )


def _write_json(root: Path, relpath: str, value: object) -> Path:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _proposal(candidate_id: str, seed: int) -> dict:
    return seal_candidate_proposal(_raw_proposal(candidate_id, seed))


def _raw_proposal(candidate_id: str, seed: int) -> dict:
    return {
        "contract": PROPOSAL_CONTRACT,
        "schema_version": 1,
        "candidate_id": candidate_id,
        "family": "spike_fade",
        "hypothesis": f"Unique bounded spike-fade timing candidate {seed}.",
        "config": {
            "signal": "spike_fade",
            "pairs": PAIRS,
            "tp_atr": 3.0,
            "sl_pips": 25.0,
            "ceiling_min": 60 + seed,
            "max_concurrent_per_pair": 1,
            "global_max_concurrent": 4,
            "per_pos_lev": 5.0,
            "atr_floor_pips": 0.5,
            "exit_policy": "FIXED",
            "order_authority": "NONE",
            "live_permission": False,
            "external_broker_mutation_allowed": False,
        },
        "risk_increase": False,
    }


def _study(
    attempt: int, proposals: list[dict], *, stress_slippage: float = 0.3
) -> dict:
    study = {
        "contract": STUDY_CONTRACT,
        "schema_version": 1,
        "study_id": f"qr-ai-tuning-attempt-{attempt}",
        "window_role": "TRAIN",
        "initial_balance_jpy": 200_000.0,
        "trade_pairs": PAIRS,
        "feed_pairs": PAIRS,
        "candidates": sorted(proposals, key=lambda item: item["candidate_id"]),
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
                "slippage_pips_per_fill": stress_slippage,
                "financing_pips_per_day": 0.8,
                "recorded_spread_multiplier": 1.0,
            },
        },
        "proposer_evidence": {
            "prompt_sha256": "d" * 64,
            "input_sha256": "e" * 64,
            "raw_response_sha256": "f" * 64,
            "model_claim": "gpt-test",
            "provider_attestation": "UNVERIFIED",
        },
        "search_budget": {
            "attempt_ordinal": attempt,
            "total_attempts_in_lineage": 3,
            "max_candidates": len(proposals),
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
    }
    return seal_study(study, SOURCES)


def _evaluation(sealed: dict) -> dict:
    candidates = sealed["study"]["candidates"]
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    cell_count = len(candidate_ids) * 4
    body = {
        "contract": EVALUATION_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed["study_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "fixed_denominator": {
            "candidate_count": len(candidate_ids),
            "intrabar_paths": ["OHLC", "OLHC"],
            "cost_arms": ["BASE", "STRESS"],
            "expected_cell_count": cell_count,
            "observed_cell_count": cell_count,
            "coordinate_receipts_complete": True,
            "execution_success_complete": False,
        },
        "candidate_evaluations": [
            {
                "candidate_id": candidate["candidate_id"],
                "status": "TRAIN_REJECT",
                "diagnostic_rank_eligible": False,
                "diagnostic_score": None,
                "risk_policy_receipt": (
                    risk := bot_config_risk_vector(
                        candidate["config"],
                        stress_slippage_pips_per_fill=sealed["study"]["cost_arms"][
                            "STRESS"
                        ]["slippage_pips_per_fill"],
                    )
                ),
                "proposer_risk_claim_ignored": True,
                "gate_blockers": [
                    *risk["blocker_codes"],
                    "RUNNER_CELL_FAILURE",
                    "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE",
                    "NON_POSITIVE_COORDINATE_WORST_NET",
                    "NORMAL_DRAWDOWN_TOO_HIGH",
                    "STRESS_DRAWDOWN_TOO_HIGH",
                    "MARGIN_REJECT_RATE_TOO_HIGH",
                    "COST_RETENTION_TOO_LOW",
                    "PAIR_POSITIVE_CONTRIBUTION_TOO_CONCENTRATED",
                    "PAIR_CONTRIBUTION_HHI_TOO_HIGH",
                    "LEAVE_ONE_PAIR_OUT_NOT_POSITIVE",
                    "COUNTERFACTUAL_LOPO_INCOMPLETE",
                    "CAPITAL_LOCK_METRIC_INCOMPLETE",
                ],
                "failed_coordinates": ["OHLC:BASE:TEST_FAILURE"],
                "coordinate_worst": {
                    "terminal_net_jpy": 0.0,
                    "realized_max_drawdown_fraction": 0.0,
                    "normal_effective_drawdown_fraction": 1.0,
                    "stress_effective_drawdown_fraction": 1.0,
                    "peak_margin_usage_fraction": 0.0,
                    "margin_reject_rate": 1.0,
                    "cost_retention": None,
                    "pair_positive_share": 1.0,
                    "pair_hhi": 1.0,
                    "effective_positive_pairs": 1.0,
                    "leave_one_pair_out_net_jpy": 0.0,
                    "capital_productivity_per_margin_day": None,
                },
                "cost_retention_by_intrabar": {"OHLC": None, "OLHC": None},
                "capital_productivity_by_cell": {
                    "OHLC:BASE": None,
                    "OHLC:STRESS": None,
                    "OLHC:BASE": None,
                    "OLHC:STRESS": None,
                },
                "mtm_complete": False,
                "mtm_incomplete_uses_realized_dd_for_train_diagnostic_only": False,
                "lopo_replay_complete": False,
                "promotion_gate_passed": False,
                "promotion_blockers": [
                    "WORN_HISTORICAL_TRAIN_ONLY",
                    "PROSPECTIVE_FORWARD_EVIDENCE_REQUIRED",
                    "CONTINUOUS_MTM_EVIDENCE_INCOMPLETE",
                    "COUNTERFACTUAL_LOPO_INCOMPLETE",
                ],
                "proof_eligible": False,
                "promotion_eligible": False,
                "live_permission": False,
                "order_authority": "NONE",
            }
            for candidate in candidates
        ],
        "diagnostic_ranking": [],
        "rank_eligible_candidate_ids": [],
        "unranked_candidate_ids": candidate_ids,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "evaluation_sha256": _sha(body)}


def _previous_kwargs(snapshot: CandidateLineageSnapshot) -> dict[str, object]:
    result = snapshot.results[-1]
    return {
        "previous_evaluation_sha256": result["evaluation_sha256"],
        "previous_evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "previous_evaluation_artifact_size_bytes": result[
            "evaluation_artifact_size_bytes"
        ],
    }


def _baseline(
    root: Path, *, proposals: list[dict] | None = None
) -> tuple[Path, CandidateLineageSnapshot, dict, Path]:
    root.mkdir(parents=True, exist_ok=True)
    events = root / "lineage"
    snapshot = initialize_registry(
        events,
        artifact_root=root,
        registry_id="qr-ai-tuning-lineage",
        lineage_prefix="qr-",
        created_by="pytest",
        event_at_utc="2026-07-20T00:00:00Z",
    )
    sealed = _study(1, proposals or [_proposal("qr-a1-base", 1)])
    study_path = _write_json(root, "artifacts/study-1.json", sealed)
    snapshot = seal_study_attempt(
        events,
        artifact_root=root,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-20T00:00:01Z",
    )
    evaluation_path = _write_json(
        root, "artifacts/evaluation-1.json", _evaluation(sealed)
    )
    snapshot = bind_result(
        events,
        artifact_root=root,
        evaluation_path=evaluation_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-20T00:00:02Z",
    )
    return events, snapshot, sealed, evaluation_path


def _submission(
    submission_id: str, proposal: object, *, errors: list[str] | None = None
) -> dict:
    return {
        "submission_id": submission_id,
        "raw_proposal_sha256": _sha(proposal),
        "proposal": proposal,
        "validation_errors": errors or [],
    }


def _state_with_one_response(
    root: Path,
) -> tuple[dict, dict, Path, CandidateLineageSnapshot, dict]:
    events, snapshot, baseline, _ = _baseline(root)
    state = initialize_tuning_state(events, artifact_root=root, sealed_study=baseline)
    state = reserve_model_invocation(
        state,
        lineage_events_dir=events,
        artifact_root=root,
        expected_parent_state_sha256=state["state_sha256"],
        invocation_id="model-a2-1",
        request_sha256="1" * 64,
        event_at_utc="2026-07-20T00:01:00Z",
    )
    proposal = _raw_proposal("qr-a2-new", 2)
    state = record_model_response(
        state,
        expected_parent_state_sha256=state["state_sha256"],
        invocation_id="model-a2-1",
        response_sha256="2" * 64,
        submissions=[_submission("submission-a2-new", proposal)],
        event_at_utc="2026-07-20T00:02:00Z",
    )
    return state, proposal, events, snapshot, baseline


def _seal_pending_second(
    root: Path,
    events: Path,
    snapshot: CandidateLineageSnapshot,
    sealed: dict,
) -> CandidateLineageSnapshot:
    path = _write_json(root, "artifacts/study-2.json", sealed)
    return seal_study_attempt(
        events,
        artifact_root=root,
        sealed_study_path=path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-20T00:02:30Z",
        **_previous_kwargs(snapshot),
    )


def _reserve_second_dispatch(
    root: Path,
) -> tuple[dict, Path, CandidateLineageSnapshot, dict]:
    state, _, events, snapshot, _ = _state_with_one_response(root)
    sealed_proposal = state["attempts"][-1]["invocations"][0]["submissions"][0][
        "sealed_proposal"
    ]
    second = _study(2, [sealed_proposal])
    pending = _seal_pending_second(root, events, snapshot, second)
    state = reserve_run_dispatch(
        state,
        lineage_events_dir=events,
        artifact_root=root,
        expected_parent_state_sha256=state["state_sha256"],
        sealed_study=second,
        dispatch_id="run-a2",
        event_at_utc="2026-07-20T00:03:00Z",
    )
    return state, events, pending, second


def test_initialize_freezes_train_envelope_and_has_no_authority(tmp_path: Path) -> None:
    events, _, baseline, _ = _baseline(tmp_path)

    state = initialize_tuning_state(
        events, artifact_root=tmp_path, sealed_study=baseline
    )
    status = status_artifact(state)

    assert state["phase"] == "READY_FOR_MODEL"
    assert state["revision"] == 0
    assert state["previous_state_sha256"] is None
    assert state["fixed_envelope"] == fixed_envelope_from_sealed_study(baseline)
    assert status["attempts_consumed"] == 1
    assert status["proposal_slots_consumed"] == 1
    assert status["automation_ready"] is False
    assert status["holdout_access_allowed"] is False
    assert status["forward_access_allowed"] is False
    assert status["proof_eligible"] is False
    assert status["promotion_eligible"] is False
    assert status["live_permission"] is False
    assert status["order_authority"] == "NONE"
    assert status["broker_mutation_allowed"] is False


def test_public_lineage_apis_reverify_paths_and_bound_artifacts(tmp_path: Path) -> None:
    events, snapshot, baseline, evaluation_path = _baseline(tmp_path)
    with pytest.raises(DojoAITuningStateError, match="caller-constructed"):
        initialize_tuning_state(
            snapshot,  # type: ignore[arg-type]
            artifact_root=tmp_path,
            sealed_study=baseline,
        )

    state = initialize_tuning_state(
        events, artifact_root=tmp_path, sealed_study=baseline
    )
    evaluation_path.write_bytes(evaluation_path.read_bytes() + b" ")
    with pytest.raises(DojoAITuningStateError, match="lineage verification failed"):
        reserve_model_invocation(
            state,
            lineage_events_dir=events,
            artifact_root=tmp_path,
            expected_parent_state_sha256=state["state_sha256"],
            invocation_id="model-a2-1",
            request_sha256="1" * 64,
            event_at_utc="2026-07-20T00:01:00Z",
        )


def test_one_model_invocation_charges_all_rows_and_rejects_a_second_call(
    tmp_path: Path,
) -> None:
    events, _, baseline, _ = _baseline(tmp_path)
    state = initialize_tuning_state(
        events, artifact_root=tmp_path, sealed_study=baseline
    )
    state = reserve_model_invocation(
        state,
        lineage_events_dir=events,
        artifact_root=tmp_path,
        expected_parent_state_sha256=state["state_sha256"],
        invocation_id="model-a2-1",
        request_sha256="1" * 64,
        event_at_utc="2026-07-20T00:01:00Z",
    )
    parent = state["state_sha256"]
    submissions = [
        _submission("duplicate", _raw_proposal("qr-a2-duplicate", 1)),
        _submission("invalid", None, errors=["MISSING_JSON_OBJECT"]),
        _submission("accepted", _raw_proposal("qr-a2-new", 2)),
    ]
    state = record_model_response(
        state,
        expected_parent_state_sha256=parent,
        invocation_id="model-a2-1",
        response_sha256="2" * 64,
        submissions=submissions,
        event_at_utc="2026-07-20T00:02:00Z",
    )
    assert (
        record_model_response(
            state,
            expected_parent_state_sha256=parent,
            invocation_id="model-a2-1",
            response_sha256="2" * 64,
            submissions=submissions,
            event_at_utc="2026-07-20T00:02:00Z",
        )
        == state
    )

    with pytest.raises(
        DojoAITuningStateError,
        match="model invocation is not allowed from phase COLLECTING_PROPOSALS",
    ):
        reserve_model_invocation(
            state,
            lineage_events_dir=events,
            artifact_root=tmp_path,
            expected_parent_state_sha256=state["state_sha256"],
            invocation_id="model-a2-2",
            request_sha256="3" * 64,
            event_at_utc="2026-07-20T00:03:00Z",
        )
    status = status_artifact(state)
    assert status["attempts_consumed"] == 2
    assert status["model_invocation_count"] == 1
    assert status["proposal_slots_consumed"] == 4
    assert status["invalid_proposal_count"] == 1
    assert status["duplicate_proposal_count"] == 1
    assert status["current_accepted_candidate_ids"] == ["qr-a2-new"]

    forged = json.loads(json.dumps(state))
    extra_invocation = json.loads(json.dumps(forged["attempts"][-1]["invocations"][0]))
    extra_invocation["invocation_id"] = "forged-second-invocation"
    forged["attempts"][-1]["invocations"].append(extra_invocation)
    _rehash(forged)
    with pytest.raises(
        DojoAITuningStateError,
        match="exactly one model invocation",
    ):
        verify_tuning_state(forged)

    with pytest.raises(
        DojoAITuningStateError,
        match="not the exact immediate first invocation",
    ):
        reserve_model_invocation(
            state,
            lineage_events_dir=events,
            artifact_root=tmp_path,
            expected_parent_state_sha256=state["state_sha256"],
            invocation_id="model-a2-1",
            request_sha256="1" * 64,
            event_at_utc="2026-07-20T00:01:00Z",
        )


@pytest.mark.parametrize("output", [[], "not-an-array"])
def test_empty_or_malformed_single_model_response_consumes_one_slot(
    tmp_path: Path, output: object
) -> None:
    root = tmp_path / ("empty" if output == [] else "malformed")
    events, _, baseline, _ = _baseline(root)
    state = initialize_tuning_state(events, artifact_root=root, sealed_study=baseline)
    parent = state["state_sha256"]
    reserved = reserve_model_invocation(
        state,
        lineage_events_dir=events,
        artifact_root=root,
        expected_parent_state_sha256=parent,
        invocation_id="single-model-a2",
        request_sha256="1" * 64,
        event_at_utc="2026-07-20T00:01:00Z",
    )
    assert (
        reserve_model_invocation(
            reserved,
            lineage_events_dir=events,
            artifact_root=root,
            expected_parent_state_sha256=parent,
            invocation_id="single-model-a2",
            request_sha256="1" * 64,
            event_at_utc="2026-07-20T00:01:00Z",
        )
        == reserved
    )
    state = record_model_response(
        reserved,
        expected_parent_state_sha256=reserved["state_sha256"],
        invocation_id="single-model-a2",
        response_sha256="2" * 64,
        submissions=output,
        event_at_utc="2026-07-20T00:02:00Z",
    )
    status = status_artifact(state)
    assert status["model_invocation_count"] == 1
    assert status["proposal_slots_consumed"] == 2
    assert status["invalid_proposal_count"] == 1
    with pytest.raises(DojoAITuningStateError, match="COLLECTING_PROPOSALS"):
        reserve_model_invocation(
            state,
            lineage_events_dir=events,
            artifact_root=root,
            expected_parent_state_sha256=state["state_sha256"],
            invocation_id="forbidden-second-model-a2",
            request_sha256="3" * 64,
            event_at_utc="2026-07-20T00:03:00Z",
        )


def test_dispatch_crosschecks_study_and_accepted_submissions(tmp_path: Path) -> None:
    state, events, _, second = _reserve_second_dispatch(tmp_path)
    assert state["phase"] == "RUN_DISPATCH_RESERVED"
    assert (
        reserve_run_dispatch(
            state,
            lineage_events_dir=events,
            artifact_root=tmp_path,
            expected_parent_state_sha256=state["previous_state_sha256"],
            sealed_study=second,
            dispatch_id="run-a2",
            event_at_utc="2026-07-20T00:03:00Z",
        )
        == state
    )

    forged = json.loads(json.dumps(state))
    forged["attempts"][-1]["dispatch"]["candidate_ids"] = ["qr-forged"]
    _rehash(forged)
    with pytest.raises(DojoAITuningStateError, match="accepted submissions"):
        verify_tuning_state(forged)

    forged = json.loads(json.dumps(state))
    forged["attempts"][-1]["dispatch"]["prior_result_binding"]["evaluation_sha256"] = (
        "9" * 64
    )
    _rehash(forged)
    with pytest.raises(DojoAITuningStateError, match="prior-result"):
        verify_tuning_state(forged)


def test_prior_result_chain_fork_is_rejected_after_rehash(tmp_path: Path) -> None:
    state, _, _, _, _ = _state_with_one_response(tmp_path)
    forged = json.loads(json.dumps(state))
    forged["attempts"][-1]["prior_result_binding"]["evaluation_sha256"] = "9" * 64
    _rehash(forged)

    with pytest.raises(DojoAITuningStateError, match="last result diverges"):
        verify_tuning_state(forged)


def test_terminal_bind_requires_dispatch_and_is_idempotent(tmp_path: Path) -> None:
    reserved, events, pending, second = _reserve_second_dispatch(tmp_path)
    evaluation_path = _write_json(
        tmp_path, "artifacts/evaluation-2.json", _evaluation(second)
    )
    complete = bind_result(
        events,
        artifact_root=tmp_path,
        evaluation_path=evaluation_path,
        expected_tip_sha256=pending.latest_event_sha256,
        event_at_utc="2026-07-20T00:05:00Z",
    )
    with pytest.raises(DojoAITuningStateError, match="before replay dispatch"):
        bind_terminal_evaluation(
            reserved,
            lineage_events_dir=events,
            artifact_root=tmp_path,
            expected_parent_state_sha256=reserved["state_sha256"],
            event_at_utc="2026-07-20T00:06:00Z",
        )

    # Re-create the pending view by removing the result event is intentionally
    # impossible: public APIs always verify the current real registry.  The run
    # start therefore uses a separate fixture with its still-pending registry.
    other_root = tmp_path / "other"
    dispatched, other_events, other_pending, other_second = _reserve_second_dispatch(
        other_root
    )
    dispatched = mark_run_dispatched(
        dispatched,
        lineage_events_dir=other_events,
        artifact_root=other_root,
        expected_parent_state_sha256=dispatched["state_sha256"],
        dispatch_id="run-a2",
        event_at_utc="2026-07-20T00:04:00Z",
    )
    evaluation_path = _write_json(
        other_root, "artifacts/evaluation-2.json", _evaluation(other_second)
    )
    complete = bind_result(
        other_events,
        artifact_root=other_root,
        evaluation_path=evaluation_path,
        expected_tip_sha256=other_pending.latest_event_sha256,
        event_at_utc="2026-07-20T00:05:00Z",
    )
    parent = dispatched["state_sha256"]
    bound = bind_terminal_evaluation(
        dispatched,
        lineage_events_dir=other_events,
        artifact_root=other_root,
        expected_parent_state_sha256=parent,
        event_at_utc="2026-07-20T00:06:00Z",
    )
    assert bound["phase"] == "READY_FOR_MODEL"
    assert (
        bound["last_terminal_result_binding"]["result_event_sha256"]
        == (complete.events[-1]["event_sha256"])
    )
    assert (
        bind_terminal_evaluation(
            bound,
            lineage_events_dir=other_events,
            artifact_root=other_root,
            expected_parent_state_sha256=parent,
            event_at_utc="2026-07-20T00:06:00Z",
        )
        == bound
    )


def test_cas_and_backward_time_reject_parallel_or_reordered_transition(
    tmp_path: Path,
) -> None:
    events, _, baseline, _ = _baseline(tmp_path)
    genesis = initialize_tuning_state(
        events, artifact_root=tmp_path, sealed_study=baseline
    )
    child = reserve_model_invocation(
        genesis,
        lineage_events_dir=events,
        artifact_root=tmp_path,
        expected_parent_state_sha256=genesis["state_sha256"],
        invocation_id="model-child-a",
        request_sha256="1" * 64,
        event_at_utc="2026-07-20T00:01:00Z",
    )
    with pytest.raises(DojoAITuningStateError, match="previous-parent replay"):
        reserve_model_invocation(
            child,
            lineage_events_dir=events,
            artifact_root=tmp_path,
            expected_parent_state_sha256=genesis["state_sha256"],
            invocation_id="model-child-b",
            request_sha256="2" * 64,
            event_at_utc="2026-07-20T00:02:00Z",
        )
    with pytest.raises(DojoAITuningStateError, match="timestamp moved backward"):
        record_model_response(
            child,
            expected_parent_state_sha256=child["state_sha256"],
            invocation_id="model-child-a",
            response_sha256="3" * 64,
            submissions=[],
            event_at_utc="2026-07-20T00:00:30Z",
        )


def test_incomplete_run_consumes_attempt_and_review_transition(tmp_path: Path) -> None:
    events, _, baseline, _ = _baseline(tmp_path)
    state = initialize_tuning_state(
        events, artifact_root=tmp_path, sealed_study=baseline
    )
    state = reserve_model_invocation(
        state,
        lineage_events_dir=events,
        artifact_root=tmp_path,
        expected_parent_state_sha256=state["state_sha256"],
        invocation_id="crashed-model",
        request_sha256="1" * 64,
        event_at_utc="2026-07-20T00:01:00Z",
    )
    parent = state["state_sha256"]
    incomplete = mark_incomplete_run(
        state,
        expected_parent_state_sha256=parent,
        reason_code="MODEL_RESPONSE_LOST_AFTER_RESERVATION",
        event_at_utc="2026-07-20T00:02:00Z",
    )
    assert status_artifact(incomplete)["attempts_consumed"] == 2
    assert (
        mark_incomplete_run(
            incomplete,
            expected_parent_state_sha256=parent,
            reason_code="MODEL_RESPONSE_LOST_AFTER_RESERVATION",
            event_at_utc="2026-07-20T00:02:00Z",
        )
        == incomplete
    )
    abandoned = abandon_incomplete_lineage(
        incomplete,
        expected_parent_state_sha256=incomplete["state_sha256"],
        review_id="operator-review-1",
        rationale="No terminal evaluation exists; burn the attempt and lineage.",
        event_at_utc="2026-07-20T00:04:00Z",
    )
    assert abandoned["phase"] == "TERMINATED"
    assert status_artifact(abandoned)["live_permission"] is False


def test_append_only_store_cas_replay_and_corruption_detection(tmp_path: Path) -> None:
    lineage_root = tmp_path / "lineage-root"
    events, _, baseline, _ = _baseline(lineage_root)
    genesis = initialize_tuning_state(
        events, artifact_root=lineage_root, sealed_study=baseline
    )
    store = tmp_path / "state-store"
    stored = initialize_state_store(store, genesis)
    assert stored["automation_ready"] is True
    assert stored["latest_state"]["automation_ready"] is False

    child_a = reserve_model_invocation(
        genesis,
        lineage_events_dir=events,
        artifact_root=lineage_root,
        expected_parent_state_sha256=genesis["state_sha256"],
        invocation_id="model-child-a",
        request_sha256="1" * 64,
        event_at_utc="2026-07-20T00:01:00Z",
    )
    child_b = reserve_model_invocation(
        genesis,
        lineage_events_dir=events,
        artifact_root=lineage_root,
        expected_parent_state_sha256=genesis["state_sha256"],
        invocation_id="model-child-b",
        request_sha256="2" * 64,
        event_at_utc="2026-07-20T00:01:00Z",
    )
    parent_tip = stored["latest_event_sha256"]
    stored = append_state_transition(
        store,
        child_a,
        expected_tip_event_sha256=parent_tip,
        expected_parent_state_sha256=genesis["state_sha256"],
    )
    assert (
        append_state_transition(
            store,
            child_a,
            expected_tip_event_sha256=parent_tip,
            expected_parent_state_sha256=genesis["state_sha256"],
        )
        == stored
    )
    with pytest.raises(DojoAITuningStateError, match="stale or forked"):
        append_state_transition(
            store,
            child_b,
            expected_tip_event_sha256=parent_tip,
            expected_parent_state_sha256=genesis["state_sha256"],
        )

    event_path = store / "000001.json"
    raw = event_path.read_text(encoding="utf-8")
    event_path.write_text(
        raw.replace("model-child-a", "model-child-z"), encoding="utf-8"
    )
    with pytest.raises(DojoAITuningStateError, match="digest mismatch"):
        verify_state_store(store)


def test_self_signed_authority_upgrade_is_rejected_even_with_new_digest(
    tmp_path: Path,
) -> None:
    events, _, baseline, _ = _baseline(tmp_path)
    state = initialize_tuning_state(
        events, artifact_root=tmp_path, sealed_study=baseline
    )
    forged = json.loads(json.dumps(state))
    forged["live_permission"] = True
    _rehash(forged)

    with pytest.raises(DojoAITuningStateError, match="research authority"):
        verify_tuning_state(forged)
