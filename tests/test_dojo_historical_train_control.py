from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from quant_rabbit.dojo_historical_train_control import (
    CONTROL_CONTRACT,
    DojoHistoricalTrainControlError,
    _candidate_proposals,
    _milestone_status,
    _registry_artifact,
    _risk_envelope,
    _seal_source_failure,
    _verified_control,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = REPO_ROOT / "config" / "dojo_g2_historical_run_control_v1.json"
REGISTRY_PATH = REPO_ROOT / "research" / "registries" / "dojo_g2_baseline_revision_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reviewed_control_is_six_month_research_only() -> None:
    control = _verified_control(CONTROL_PATH, repo_root=REPO_ROOT)

    assert control["contract"] == CONTROL_CONTRACT
    assert control["trainer_milestones"]["m5_completed_months_per_review"] == 6
    assert control["trainer_milestones"]["partial_month_tuning_allowed"] is False
    assert control["trainer_milestones"]["parameter_change_applies_only_to_new_generation"] is True
    assert control["authority"]["historical_replay_process_start_allowed"] is True
    assert control["authority"]["broker_mutation_allowed"] is False
    assert control["authority"]["live_permission"] is False
    assert control["authority"]["order_authority"] == "NONE"


def test_control_rejects_live_or_parallel_authority(tmp_path: Path) -> None:
    control = _load(CONTROL_PATH)
    unsafe = copy.deepcopy(control)
    unsafe["authority"]["live_permission"] = True
    unsafe["execution"]["max_parallel_jobs"] = 2
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(unsafe), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError):
        _verified_control(path, repo_root=REPO_ROOT)


def test_g2_registry_produces_exact_six_sealed_workers() -> None:
    registry = _load(REGISTRY_PATH)
    assert _registry_artifact(registry) == registry["artifact_sha256"]

    proposals = _candidate_proposals(registry)

    assert len(proposals) == 6
    assert [row["candidate_id"] for row in proposals] == sorted(
        row["worker_id"] for row in registry["workers"]
    )
    assert all(row["risk_increase"] is False for row in proposals)
    assert {row["family"] for row in proposals} == {
        "burst",
        "mean_revert_24h",
        "prev_day_extreme_fade",
        "pullback_limit",
        "round_number_fade",
        "spike_fade",
    }
    assert _risk_envelope(registry)["maximum_gross_leverage"] == 8.0
    assert _risk_envelope(registry)["max_open_and_pending_total"] == 4


def test_read_only_milestone_status_does_not_publish(tmp_path: Path) -> None:
    status = _milestone_status(tmp_path, {"jobs": []}, publish=False)

    assert status["completed_m5_month_count"] == 0
    assert status["next_trainer_review_at_completed_m5_month_count"] == 6
    assert status["trainer_review_due"] is False
    assert not (tmp_path / "trainer-milestones").exists()


def test_source_failure_accepts_predecessor_failures_already_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[tuple[Path, dict]] = []
    recorded: list[dict] = []
    job = {
        "job_sha256": "1" * 64,
        "source_binding_id": "M5_EXACT28_2020_2026H1",
        "month": "2020-02",
        "intrabar_path": "OHLC",
        "coordinates": [
            {"coordinate_id": "a" * 64},
            {"coordinate_id": "b" * 64},
        ],
    }
    handoff = {
        "job": job,
        "claim": {"claim_sha256": "2" * 64},
        "recorded_coordinate_count": 1,
        "pending_coordinate_count": 1,
        "runnable_coordinate_ids": ["b" * 64],
        "predecessor_blocked_coordinate_count": 0,
    }

    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._write_once",
        lambda path, value: writes.append((path, value)),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.build_long_horizon_coordinate_result",
        lambda **kwargs: {
            "coordinate_id": kwargs["coordinate_id"],
            "status": kwargs["status"],
        },
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.record_long_horizon_coordinate_results",
        lambda *args, **kwargs: recorded.extend(kwargs["results"]),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.seal_long_horizon_attempt",
        lambda *args, **kwargs: {
            "terminal_manifest": {
                "terminal_sha256": "3" * 64,
                "cells": [
                    {"status": "FAILED"},
                    {"status": "FAILED"},
                ],
            }
        },
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._milestone_status",
        lambda *args, **kwargs: {"completed_m5_month_count": 0},
    )

    result = _seal_source_failure(
        root=tmp_path,
        schedule={},
        plan={},
        handoff=handoff,
        error=RuntimeError("missing source day"),
    )

    assert [row["coordinate_id"] for row in recorded] == ["b" * 64]
    assert result["coordinate_result_count"] == 2
    assert result["new_source_failure_coordinate_count"] == 1
    assert result["predecessor_failure_coordinate_count"] == 1
    assert result["failed_coordinate_count"] == 2
    assert len(writes) == 2
