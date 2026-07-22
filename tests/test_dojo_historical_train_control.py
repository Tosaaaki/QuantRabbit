from __future__ import annotations

import copy
import fcntl
import json
import os
from pathlib import Path

import pytest

from quant_rabbit.dojo_historical_train_control import (
    CONTROL_CONTRACT,
    DojoHistoricalTrainControlError,
    _candidate_proposals,
    _assert_dynamic_machine_capacity,
    _acquire_conflicting_run_locks,
    _baseline_raw_bytes,
    _conflicting_generation_statuses,
    _disk_capacity_snapshot,
    _g2_room_bindings,
    _milestone_status,
    _room_binding_sha256,
    _registry_artifact,
    _risk_envelope,
    _seal_source_failure,
    _verified_control,
    prepare_generation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = REPO_ROOT / "config" / "dojo_g2_historical_run_control_v1.json"
ROOM_CONTROL_PATH = (
    REPO_ROOT / "config" / "dojo_g2_parallel_rooms_run_control_v1.json"
)
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


def test_room_control_binds_exact_six_isolated_strategy_rooms() -> None:
    control = _verified_control(ROOM_CONTROL_PATH, repo_root=REPO_ROOT)

    assert control["fixed_inputs"]["generation"] == "G2_ROOM_V1"
    assert control["fixed_inputs"]["room_family_bindings_sha256"] == (
        _room_binding_sha256()
    )
    assert _g2_room_bindings() == [
        {"room_id": "room-g2-01", "family_id": "spike_fade"},
        {"room_id": "room-g2-02", "family_id": "burst"},
        {"room_id": "room-g2-03", "family_id": "pullback_limit"},
        {"room_id": "room-g2-04", "family_id": "prev_day_extreme_fade"},
        {"room_id": "room-g2-05", "family_id": "round_number_fade"},
        {"room_id": "room-g2-06", "family_id": "mean_revert_24h"},
    ]


def test_room_control_rejects_taxonomy_binding_drift(tmp_path: Path) -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["fixed_inputs"]["room_family_bindings_sha256"] = "1" * 64
    path = tmp_path / "drifted-room-control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="room taxonomy binding drifted",
    ):
        _verified_control(path, repo_root=REPO_ROOT)


def test_room_generation_prepares_twelve_stream_jobs_and_144_room_cells(
    tmp_path: Path,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["output_root"] = str(tmp_path / "room-run")
    control_path = tmp_path / "room-control.json"
    control_path.write_text(json.dumps(control), encoding="utf-8")

    prepared = prepare_generation(
        repo_root=REPO_ROOT,
        run_control_path=control_path,
    )
    schedule = _load(tmp_path / "room-run" / "schedule.json")
    manifest = _load(tmp_path / "room-run" / "control-manifest.json")

    assert prepared["stream_job_count"] == 12
    assert prepared["result_coordinate_count"] == 144
    assert manifest["generation"] == "G2_ROOM_V1"
    assert manifest["room_count"] == 6
    assert manifest["room_family_bindings_sha256"] == _room_binding_sha256()
    assert all(job["coordinate_count"] == 12 for job in schedule["jobs"])
    assert {
        coordinate["fold_label"]
        for coordinate in schedule["jobs"][0]["coordinates"]
    } == {row["room_id"] for row in _g2_room_bindings()}
    assert all(
        coordinate["active_worker_count"] == 1
        for coordinate in schedule["jobs"][0]["coordinates"]
    )
    assert [
        row["artifact_id"] for row in manifest["sealed_input_artifacts"]
    ] == [
        "IMPLEMENTATION_MANIFEST",
        "RUN_CONTROL",
        "SOURCE_MANIFEST",
        "STRATEGY_REGISTRY",
    ]
    assert all(
        (tmp_path / "room-run" / row["relative_path"]).is_file()
        for row in manifest["sealed_input_artifacts"]
    )


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


def test_allocator_values_are_generation_sealed_but_not_engine_literals() -> None:
    registry = _load(REGISTRY_PATH)
    tuned = copy.deepcopy(registry)
    tuned["allocator"]["per_position_leverage"] = 1.5
    tuned["allocator"]["maximum_gross_leverage"] = 6.0

    envelope = _risk_envelope(tuned)

    assert envelope["per_position_leverage"] == 1.5
    assert envelope["maximum_gross_leverage"] == 6.0
    unsafe = copy.deepcopy(tuned)
    unsafe["allocator"]["maximum_gross_leverage"] = 30.0
    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="safety relationships",
    ):
        _risk_envelope(unsafe)


def test_dynamic_machine_load_gate_blocks_another_heavy_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.cpu_count", lambda: 10
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.getloadavg",
        lambda: (21.0, 20.0, 19.0),
    )

    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="machine load is above",
    ):
        _assert_dynamic_machine_capacity(control)


def test_conflicting_runner_lock_is_held_for_the_caller_lifetime(
    tmp_path: Path,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    lock_path = tmp_path / "old-run.lock"
    lock_path.touch()
    control["execution"]["conflicting_run_lock_paths"] = [str(lock_path)]

    descriptors = _acquire_conflicting_run_locks(control)
    competing = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(competing)
        for descriptor in descriptors:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def test_capacity_baseline_scales_shared_source_and_per_coordinate_evidence() -> None:
    control = _load(ROOM_CONTROL_PATH)

    estimate = _baseline_raw_bytes(
        control=control,
        planned_coordinate_count=12,
    )

    baseline = control["execution"]["capacity_baseline"]
    expected = baseline["source_bytes_per_job"] + 12 * (
        (
            baseline["raw_bytes_per_job"]
            - baseline["source_bytes_per_job"]
            + baseline["coordinate_count"]
            - 1
        )
        // baseline["coordinate_count"]
    )
    assert estimate == expected
    assert estimate < baseline["raw_bytes_per_job"] * 6


def test_conflicting_generation_status_blocks_orphaned_active_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    conflict_root = tmp_path / "old-generation"
    conflict_root.mkdir()
    control["execution"]["conflicting_execution_roots"] = [str(conflict_root)]
    control["execution"]["conflicting_run_lock_paths"] = []
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.cpu_count", lambda: 10
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.getloadavg",
        lambda: (1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._conflicting_generation_statuses",
        lambda value: [
            {
                "output_root": str(conflict_root),
                "exists": True,
                "active_job_count": 1,
                "terminal_job_count": 3,
                "status_sha256": "1" * 64,
            }
        ],
    )

    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="active or orphaned claim",
    ):
        _assert_dynamic_machine_capacity(control)


def test_conflicting_generation_status_reports_absent_root() -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["conflicting_execution_roots"] = [
        "/definitely/absent/dojo-generation"
    ]

    assert _conflicting_generation_statuses(control) == [
        {
            "output_root": "/definitely/absent/dojo-generation",
            "exists": False,
            "active_job_count": 0,
            "terminal_job_count": 0,
        }
    ]


def test_disk_capacity_compares_shared_raw_and_archive_filesystem(
    tmp_path: Path,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["output_root"] = str(tmp_path / "run")
    control["execution"]["archive_root"] = str(tmp_path / "archive")

    snapshot = _disk_capacity_snapshot(
        root=tmp_path / "run",
        control=control,
        estimated_raw_bytes=1_000,
        estimated_peak_bytes=1_250,
    )

    assert snapshot["shared_filesystem"] is True
    assert snapshot["run_required_bytes"] == (
        control["execution"]["minimum_free_disk_bytes"] + 1_250
    )
    assert snapshot["archive_required_bytes"] == 0

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
