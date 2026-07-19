from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import quant_rabbit.dojo_candidate_lineage_registry as registry_module
from quant_rabbit.dojo_bot_trainer import (
    EVALUATION_CONTRACT,
    PROPOSAL_CONTRACT,
    STUDY_CONTRACT,
    seal_candidate_proposal,
    seal_study,
)
from quant_rabbit.dojo_bot_catalog import bot_config_risk_vector
from quant_rabbit.dojo_candidate_lineage_registry import (
    CandidateLineageError,
    bind_result,
    initialize_registry,
    seal_study_attempt,
    status_artifact,
    verify_registry,
)


PAIRS = ["CAD_JPY", "EUR_USD", "GBP_USD", "USD_JPY"]
SOURCES = {
    "bots/lab_bot.py": "a" * 64,
    "src/quant_rabbit/dojo_bot_trainer.py": "b" * 64,
}


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(root: Path, relpath: str, value: object) -> Path:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _proposal(candidate_id: str, seed: int) -> dict:
    return seal_candidate_proposal(
        {
            "contract": PROPOSAL_CONTRACT,
            "schema_version": 1,
            "candidate_id": candidate_id,
            "family": "spike_fade",
            "hypothesis": f"Bounded candidate {seed} without risk increase.",
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
            },
            "risk_increase": False,
        }
    )


def _sealed_study(
    attempt: int,
    seeds: list[int],
    *,
    window: str = "march",
) -> dict:
    candidates = [
        _proposal(f"qr-a{attempt}-c{index:02d}", seed)
        for index, seed in enumerate(seeds)
    ]
    start, end, corpus = {
        "march": (
            "2025-03-01T00:00:00Z",
            "2025-04-01T00:00:00Z",
            "1" * 64,
        ),
        "april": (
            "2025-04-01T00:00:00Z",
            "2025-05-01T00:00:00Z",
            "2" * 64,
        ),
    }[window]
    study = {
        "contract": STUDY_CONTRACT,
        "schema_version": 1,
        "study_id": f"qr-lineage-attempt-{attempt}",
        "window_role": "TRAIN",
        "initial_balance_jpy": 200_000.0,
        "trade_pairs": PAIRS,
        "feed_pairs": PAIRS,
        "candidates": candidates,
        "window": {
            "start_utc": start,
            "end_utc": end,
            "corpus_id": f"worn-fx-{window}-2025",
            "corpus_sha256": corpus,
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
            "prompt_sha256": "3" * 64,
            "input_sha256": "4" * 64,
            "raw_response_sha256": "5" * 64,
            "model_claim": "gpt-5.6-sol",
            "provider_attestation": "UNVERIFIED",
        },
        "search_budget": {
            "attempt_ordinal": attempt,
            "total_attempts_in_lineage": 3,
            "max_candidates": len(candidates),
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
    return {**body, "evaluation_sha256": _canonical_sha(body)}


def _passing_evaluation(sealed: dict, scores: list[float]) -> dict:
    evaluation = _evaluation(sealed)
    evaluation["fixed_denominator"]["execution_success_complete"] = True
    for row, score in zip(evaluation["candidate_evaluations"], scores, strict=True):
        row.update(
            {
                "status": "TRAIN_DIAGNOSTIC_PASS",
                "diagnostic_rank_eligible": True,
                "diagnostic_score": score,
                "gate_blockers": [],
                "failed_coordinates": [],
                "coordinate_worst": {
                    "terminal_net_jpy": 10_000.0,
                    "realized_max_drawdown_fraction": 0.02,
                    "normal_effective_drawdown_fraction": 0.03,
                    "stress_effective_drawdown_fraction": 0.04,
                    "peak_margin_usage_fraction": 0.20,
                    "margin_reject_rate": 0.01,
                    "cost_retention": 0.80,
                    "pair_positive_share": 0.25,
                    "pair_hhi": 0.25,
                    "effective_positive_pairs": 4.0,
                    "leave_one_pair_out_net_jpy": 2_000.0,
                    "capital_productivity_per_margin_day": 1.0,
                },
                "cost_retention_by_intrabar": {"OHLC": 0.8, "OLHC": 0.8},
                "capital_productivity_by_cell": {
                    "OHLC:BASE": 1.0,
                    "OHLC:STRESS": 1.0,
                    "OLHC:BASE": 1.0,
                    "OLHC:STRESS": 1.0,
                },
                "mtm_complete": True,
                "lopo_replay_complete": True,
                "promotion_blockers": [
                    "WORN_HISTORICAL_TRAIN_ONLY",
                    "PROSPECTIVE_FORWARD_EVIDENCE_REQUIRED",
                ],
            }
        )
    ranked = [
        row["candidate_id"]
        for row in sorted(
            evaluation["candidate_evaluations"],
            key=lambda row: (-row["diagnostic_score"], row["candidate_id"]),
        )
    ]
    evaluation["diagnostic_ranking"] = list(ranked)
    evaluation["rank_eligible_candidate_ids"] = list(ranked)
    evaluation["unranked_candidate_ids"] = []
    body = {
        key: value for key, value in evaluation.items() if key != "evaluation_sha256"
    }
    evaluation["evaluation_sha256"] = _canonical_sha(body)
    return evaluation


def _init(root: Path, events: Path):
    return initialize_registry(
        events,
        artifact_root=root,
        registry_id="qr-lineage-test",
        lineage_prefix="qr-",
        created_by="pytest",
        event_at_utc="2026-07-19T00:00:00Z",
    )


def _previous_kwargs(snapshot) -> dict[str, object]:
    result = snapshot.results[-1]
    return {
        "previous_evaluation_sha256": result["evaluation_sha256"],
        "previous_evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "previous_evaluation_artifact_size_bytes": result[
            "evaluation_artifact_size_bytes"
        ],
    }


def _seal_and_bind(
    root: Path,
    events: Path,
    snapshot,
    *,
    attempt: int,
    seeds: list[int],
    second: int,
    window: str = "march",
):
    sealed = _sealed_study(attempt, seeds, window=window)
    study_path = _write_json(root, f"artifacts/study-{attempt}.json", sealed)
    kwargs = _previous_kwargs(snapshot) if attempt > 1 else {}
    snapshot = seal_study_attempt(
        events,
        artifact_root=root,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc=f"2026-07-19T00:00:{second:02d}Z",
        **kwargs,
    )
    evaluation_path = _write_json(
        root, f"artifacts/evaluation-{attempt}.json", _evaluation(sealed)
    )
    snapshot = bind_result(
        events,
        artifact_root=root,
        evaluation_path=evaluation_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc=f"2026-07-19T00:00:{second + 1:02d}Z",
    )
    return snapshot, sealed, evaluation_path


def test_full_three_attempt_chain_is_bounded_and_authority_free(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=1, seeds=[1, 2], second=1
    )
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=2, seeds=[3, 4], second=3
    )
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=3, seeds=[5, 6], second=5
    )

    assert [event["event_type"] for event in snapshot.events] == [
        "GENESIS",
        "STUDY_SEALED",
        "RESULT_BOUND",
        "NEXT_STUDY_SEALED",
        "RESULT_BOUND",
        "NEXT_STUDY_SEALED",
        "RESULT_BOUND",
    ]
    assert [path.name for path in sorted(events.iterdir())] == [
        f"{index:06d}.json" for index in range(7)
    ]
    assert len(snapshot.cumulative_unique_config_sha256s) == 6
    assert len(snapshot.cumulative_unique_proposal_sha256s) == 6
    assert snapshot.studies[1]["previous_attempt_evaluation_binding"] == (
        _previous_kwargs_for_body(snapshot.results[0])
    )
    status = status_artifact(events, artifact_root=tmp_path)
    assert status["state"] == "COMPLETE"
    assert status["external_witness_status"] == "ABSENT"
    assert status["proof_eligible"] is False
    assert status["promotion_eligible"] is False
    assert status["live_permission"] is False
    assert status["order_authority"] == "NONE"
    assert "LOCAL_OWNER_CAN_RECOMPUTE_A_REPLACEMENT_CHAIN" in status["limitations"]

    study_four = _write_json(tmp_path, "artifacts/study-4.json", _sealed_study(3, [7]))
    with pytest.raises(CandidateLineageError, match="attempt limit"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=study_four,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:07Z",
            **_previous_kwargs(snapshot),
        )


def _previous_kwargs_for_body(result) -> dict[str, object]:
    return {
        "attempt_ordinal": result["attempt_ordinal"],
        "evaluation_sha256": result["evaluation_sha256"],
        "evaluation_artifact_sha256": result["evaluation_artifact_sha256"],
        "evaluation_artifact_size_bytes": result["evaluation_artifact_size_bytes"],
    }


def test_next_attempt_requires_exact_previous_evaluation_triple(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=1, seeds=[1], second=1
    )
    study_path = _write_json(tmp_path, "artifacts/study-2.json", _sealed_study(2, [2]))
    with pytest.raises(CandidateLineageError, match="previous_evaluation_sha256"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=study_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:03Z",
        )
    wrong = _previous_kwargs(snapshot)
    wrong["previous_evaluation_artifact_size_bytes"] = (
        int(wrong["previous_evaluation_artifact_size_bytes"]) + 1
    )
    with pytest.raises(CandidateLineageError, match="immediately preceding"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=study_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:03Z",
            **wrong,
        )


def test_duplicate_config_same_prefix_window_is_rejected(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=1, seeds=[1], second=1
    )
    duplicate = _write_json(tmp_path, "artifacts/study-2.json", _sealed_study(2, [1]))
    with pytest.raises(CandidateLineageError, match="duplicate config"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=duplicate,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:03Z",
            **_previous_kwargs(snapshot),
        )


def test_window_label_or_corpus_reexport_cannot_bypass_duplicate_config(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=1, seeds=[1], second=1
    )
    changed_label = _sealed_study(2, [1])
    study = json.loads(json.dumps(changed_label["study"]))
    study["window"]["corpus_id"] = "same-window-reexported-under-another-name"
    study["window"]["corpus_sha256"] = "9" * 64
    changed_label = seal_study(study, SOURCES)
    duplicate = _write_json(tmp_path, "artifacts/study-2.json", changed_label)
    with pytest.raises(CandidateLineageError, match="duplicate config"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=duplicate,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:03Z",
            **_previous_kwargs(snapshot),
        )


def test_cumulative_unique_config_budget_cannot_exceed_fourteen(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=1, seeds=list(range(1, 9)), second=1
    )
    oversized = _write_json(
        tmp_path,
        "artifacts/study-2.json",
        _sealed_study(2, list(range(9, 16))),
    )
    with pytest.raises(CandidateLineageError, match="exceeds 14"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=oversized,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:03Z",
            **_previous_kwargs(snapshot),
        )


def test_cumulative_unique_proposal_budget_cannot_exceed_fourteen(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    snapshot, _, _ = _seal_and_bind(
        tmp_path, events, snapshot, attempt=1, seeds=list(range(1, 9)), second=1
    )
    # Reusing eight configs on a distinct window is allowed by the duplicate
    # rule, but seven new proposal seals would take the cross-study total to 15.
    oversized = _write_json(
        tmp_path,
        "artifacts/study-2.json",
        _sealed_study(2, list(range(1, 8)), window="april"),
    )
    with pytest.raises(CandidateLineageError, match="proposal budget exceeds 14"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=oversized,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:03Z",
            **_previous_kwargs(snapshot),
        )


def test_ordinal_gap_and_second_study_before_result_are_rejected(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    gap_path = _write_json(tmp_path, "artifacts/study-gap.json", _sealed_study(2, [1]))
    with pytest.raises(CandidateLineageError, match="ordinal"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=gap_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:01Z",
        )

    study_one = _sealed_study(1, [1])
    study_one_path = _write_json(tmp_path, "artifacts/study-1.json", study_one)
    snapshot = seal_study_attempt(
        events,
        artifact_root=tmp_path,
        sealed_study_path=study_one_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-19T00:00:01Z",
    )
    with pytest.raises(CandidateLineageError, match="bind its result"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=gap_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:02Z",
        )


def test_stale_tip_and_o_excl_collision_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    study_path = _write_json(tmp_path, "artifacts/study-1.json", _sealed_study(1, [1]))
    with pytest.raises(CandidateLineageError, match="stale lineage tip"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=study_path,
            expected_tip_sha256="0" * 64,
            event_at_utc="2026-07-19T00:00:01Z",
        )

    original = registry_module._write_event_exclusive

    def collide(directory_fd: int, name: str, event: object) -> None:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"{}\n")
        original(directory_fd, name, event)

    monkeypatch.setattr(registry_module, "_write_event_exclusive", collide)
    with pytest.raises(CandidateLineageError, match="slot already exists"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=study_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:01Z",
        )


def test_post_write_fsync_failure_never_reopens_consumed_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    study_path = _write_json(tmp_path, "artifacts/study-1.json", _sealed_study(1, [1]))
    real_fsync = registry_module.os.fsync
    fsync_calls = 0

    def fail_directory_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(registry_module.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="injected directory fsync failure"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=study_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:01Z",
        )
    monkeypatch.setattr(registry_module.os, "fsync", real_fsync)

    assert (events / "000001.json").is_file()
    advanced = verify_registry(events, artifact_root=tmp_path)
    assert advanced.latest_sequence == 1
    with pytest.raises(CandidateLineageError, match="stale lineage tip"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=study_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:01Z",
        )


def test_directory_open_detects_lstat_open_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact-root"
    replacement = tmp_path / "replacement"
    displaced = tmp_path / "displaced"
    root.mkdir()
    replacement.mkdir()
    real_open = registry_module.os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if not swapped and Path(path) == root:
            swapped = True
            root.rename(displaced)
            replacement.rename(root)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(registry_module.os, "open", swap_before_open)
    with pytest.raises(CandidateLineageError, match="changed while being opened"):
        registry_module._open_artifact_root(root, close=True)


def test_symlinked_registry_and_bound_artifact_are_rejected(tmp_path: Path) -> None:
    real_events = tmp_path / "real-events"
    real_events.mkdir()
    linked_events = tmp_path / "linked-events"
    linked_events.symlink_to(real_events, target_is_directory=True)
    with pytest.raises(CandidateLineageError, match="real directory"):
        initialize_registry(
            linked_events,
            artifact_root=tmp_path,
            registry_id="symlink-test",
            lineage_prefix="qr-",
            created_by="pytest",
            event_at_utc="2026-07-19T00:00:00Z",
        )

    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    real_study = _write_json(tmp_path, "real-study.json", _sealed_study(1, [1]))
    linked_study = tmp_path / "artifacts" / "study-1.json"
    linked_study.parent.mkdir()
    linked_study.symlink_to(real_study)
    with pytest.raises(CandidateLineageError, match="regular file"):
        seal_study_attempt(
            events,
            artifact_root=tmp_path,
            sealed_study_path=linked_study,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:01Z",
        )


def test_bound_artifact_rewrite_invalidates_registry(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    snapshot, _, evaluation_path = _seal_and_bind(
        tmp_path, events, snapshot, attempt=1, seeds=[1], second=1
    )
    evaluation_path.write_text(
        evaluation_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(CandidateLineageError, match="rewritten"):
        verify_registry(events, artifact_root=tmp_path)


@pytest.mark.parametrize("mutation", ["rewrite", "gap", "unexpected"])
def test_event_rewrite_gap_and_unexpected_fork_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    study_path = _write_json(tmp_path, "artifacts/study-1.json", _sealed_study(1, [1]))
    seal_study_attempt(
        events,
        artifact_root=tmp_path,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-19T00:00:01Z",
    )
    if mutation == "rewrite":
        event_path = events / "000001.json"
        event = json.loads(event_path.read_text(encoding="utf-8"))
        event["body"]["study_id"] = "rewritten-study"
        event_path.write_text(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    elif mutation == "gap":
        (events / "000001.json").rename(events / "000002.json")
    else:
        (events / "fork.txt").write_text("fork", encoding="utf-8")
    with pytest.raises(CandidateLineageError):
        verify_registry(events, artifact_root=tmp_path)


def test_whole_directory_recreation_is_disclosed_not_misrepresented(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events"
    original = _init(tmp_path, events)
    events.rename(tmp_path / "discarded-events")
    replacement = initialize_registry(
        events,
        artifact_root=tmp_path,
        registry_id="qr-lineage-test",
        lineage_prefix="qr-",
        created_by="pytest",
        event_at_utc="2026-07-19T00:00:01Z",
    )

    assert replacement.latest_event_sha256 != original.latest_event_sha256
    status = status_artifact(events, artifact_root=tmp_path)
    assert "LOCAL_OWNER_CAN_DELETE_OR_RECREATE_ENTIRE_LEDGER" in status["limitations"]
    assert status["external_witness_status"] == "ABSENT"
    assert status["proof_eligible"] is False


def test_forged_or_incomplete_evaluation_is_rejected(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    sealed = _sealed_study(1, [1])
    study_path = _write_json(tmp_path, "artifacts/study-1.json", sealed)
    snapshot = seal_study_attempt(
        events,
        artifact_root=tmp_path,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-19T00:00:01Z",
    )
    evaluation = _evaluation(sealed)
    evaluation["fixed_denominator"]["observed_cell_count"] = 0
    body = {
        key: value for key, value in evaluation.items() if key != "evaluation_sha256"
    }
    evaluation["evaluation_sha256"] = _canonical_sha(body)
    evaluation_path = _write_json(tmp_path, "artifacts/evaluation-1.json", evaluation)
    with pytest.raises(CandidateLineageError, match="fixed denominator"):
        bind_result(
            events,
            artifact_root=tmp_path,
            evaluation_path=evaluation_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:02Z",
        )


def test_evaluation_nested_shape_and_risk_receipt_are_verified(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    sealed = _sealed_study(1, [1])
    study_path = _write_json(tmp_path, "artifacts/study-1.json", sealed)
    snapshot = seal_study_attempt(
        events,
        artifact_root=tmp_path,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-19T00:00:01Z",
    )

    evaluation = _evaluation(sealed)
    evaluation["candidate_evaluations"][0]["risk_policy_receipt"]["rankable"] = False
    body = {
        key: value for key, value in evaluation.items() if key != "evaluation_sha256"
    }
    evaluation["evaluation_sha256"] = _canonical_sha(body)
    evaluation_path = _write_json(
        tmp_path, "artifacts/evaluation-risk.json", evaluation
    )
    with pytest.raises(CandidateLineageError, match="risk policy receipt"):
        bind_result(
            events,
            artifact_root=tmp_path,
            evaluation_path=evaluation_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:02Z",
        )

    evaluation = _evaluation(sealed)
    evaluation["candidate_evaluations"][0]["coordinate_worst"].pop("pair_hhi")
    body = {
        key: value for key, value in evaluation.items() if key != "evaluation_sha256"
    }
    evaluation["evaluation_sha256"] = _canonical_sha(body)
    evaluation_path = _write_json(
        tmp_path, "artifacts/evaluation-shape.json", evaluation
    )
    with pytest.raises(CandidateLineageError, match="coordinate_worst.*shape drifted"):
        bind_result(
            events,
            artifact_root=tmp_path,
            evaluation_path=evaluation_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:02Z",
        )


def test_evaluation_ranking_is_recomputed_not_caller_ordered(tmp_path: Path) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    sealed = _sealed_study(1, [1, 2])
    study_path = _write_json(tmp_path, "artifacts/study-1.json", sealed)
    snapshot = seal_study_attempt(
        events,
        artifact_root=tmp_path,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-19T00:00:01Z",
    )
    evaluation = _passing_evaluation(sealed, [77.666667, 77.666667])
    evaluation["diagnostic_ranking"].reverse()
    evaluation["rank_eligible_candidate_ids"].reverse()
    body = {
        key: value for key, value in evaluation.items() if key != "evaluation_sha256"
    }
    evaluation["evaluation_sha256"] = _canonical_sha(body)
    evaluation_path = _write_json(
        tmp_path, "artifacts/evaluation-rank.json", evaluation
    )
    with pytest.raises(CandidateLineageError, match="ranking"):
        bind_result(
            events,
            artifact_root=tmp_path,
            evaluation_path=evaluation_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:02Z",
        )


@pytest.mark.parametrize("mutation", ["score", "negative_summary"])
def test_evaluation_score_and_visible_gates_are_rebuilt(
    tmp_path: Path, mutation: str
) -> None:
    events = tmp_path / "events"
    snapshot = _init(tmp_path, events)
    sealed = _sealed_study(1, [1])
    study_path = _write_json(tmp_path, "artifacts/study-1.json", sealed)
    snapshot = seal_study_attempt(
        events,
        artifact_root=tmp_path,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-19T00:00:01Z",
    )
    evaluation = _passing_evaluation(sealed, [77.666667])
    row = evaluation["candidate_evaluations"][0]
    if mutation == "score":
        row["diagnostic_score"] = 100.0
    else:
        row["coordinate_worst"]["terminal_net_jpy"] = -1.0
    body = {
        key: value for key, value in evaluation.items() if key != "evaluation_sha256"
    }
    evaluation["evaluation_sha256"] = _canonical_sha(body)
    evaluation_path = _write_json(
        tmp_path, f"artifacts/evaluation-{mutation}.json", evaluation
    )
    with pytest.raises(CandidateLineageError, match="score|gate blocker"):
        bind_result(
            events,
            artifact_root=tmp_path,
            evaluation_path=evaluation_path,
            expected_tip_sha256=snapshot.latest_event_sha256,
            event_at_utc="2026-07-19T00:00:02Z",
        )


def test_cli_init_and_status_expose_local_only_limitations(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "run-dojo-candidate-lineage.py"
    events = tmp_path / "events"
    init = subprocess.run(
        [
            sys.executable,
            str(script),
            "init",
            "--events-dir",
            str(events),
            "--artifact-root",
            str(tmp_path),
            "--registry-id",
            "cli-lineage",
            "--lineage-prefix",
            "qr-",
            "--created-by",
            "pytest",
            "--event-at-utc",
            "2026-07-19T00:00:00Z",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert init.returncode == 0, init.stderr
    payload = json.loads(init.stdout)
    assert payload["state"] == "READY_FOR_STUDY"
    assert payload["proof_eligible"] is False

    status = subprocess.run(
        [
            sys.executable,
            str(script),
            "status",
            "--events-dir",
            str(events),
            "--artifact-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["external_witness_status"] == "ABSENT"
