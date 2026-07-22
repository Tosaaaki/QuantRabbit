from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_rabbit.dojo_long_horizon_plan import canonical_sha256

from quant_rabbit.dojo_historical_train_control import (
    CONTROL_CONTRACT,
    DojoHistoricalTrainControlError,
    _acquire_conflicting_run_locks,
    _acquire_historical_operation_locks,
    _archive_runtime_options,
    _archive_runtime_binding,
    _assert_archive_staging_capacity,
    _assert_disk_capacity,
    _assert_dynamic_machine_capacity,
    _assert_historical_operation_lock_identities,
    _baseline_raw_bytes,
    _candidate_proposals,
    _conflicting_generation_statuses,
    _deep_verify_completed_job_custody,
    _disk_capacity_snapshot,
    _effective_archive_staging_fraction,
    _filesystem_capacity_reservations,
    _find_supersede_receipt_for_root,
    _g2_room_bindings,
    _milestone_status,
    _load_generation,
    _open_stable_lock_file,
    _release_historical_operation_locks,
    _release_one_historical_operation_lock,
    _room_binding_sha256,
    _registry_artifact,
    _risk_envelope,
    _seal_source_failure,
    _verified_control,
    _verify_supersede_receipt_chain,
    advance_one_historical_transition,
    archive_next_completed_job,
    evaluate_historical_lifecycle,
    generation_status,
    prepare_generation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = REPO_ROOT / "config" / "dojo_g2_historical_run_control_v1.json"
ROOM_CONTROL_PATH = REPO_ROOT / "config" / "dojo_g2_parallel_rooms_run_control_v1.json"
ROOM_CONTROL_V2_PATH = (
    REPO_ROOT / "config" / "dojo_g2_parallel_rooms_run_control_v2.json"
)
REGISTRY_PATH = (
    REPO_ROOT / "research" / "registries" / "dojo_g2_baseline_revision_v1.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_compact_signed_attestation(
    *,
    archive_root: Path,
    job_sha256: str,
    manifest_sha256: str,
    local_receipt_sha256: str,
    parent_id: str = "driveParent123456",
    attestation_id: str = "a" * 64,
) -> tuple[Path, Path]:
    public_key_hex = "12" * 32
    public_key_sha256 = hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()
    authority_body = {
        "contract": "QR_DOJO_HISTORICAL_JOB_DRIVE_ATTESTATION_PUBLIC_KEY_SEAL_V2",
        "schema_version": 2,
        "status": "OPERATOR_PUBLIC_KEY_ENROLLED_BEFORE_READBACK",
        "job_sha256": job_sha256,
        "manifest_sha256": manifest_sha256,
        "local_archive_receipt_sha256": local_receipt_sha256,
        "expected_drive_parent_id": parent_id,
        "algorithm": "ED25519",
        "public_key_hex": public_key_hex,
        "public_key_sha256": public_key_sha256,
        "enrolled_at_utc": "2026-07-23T00:00:00+00:00",
        "private_key_material_accepted": False,
        "historical_train_is_proof": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    authority = {
        **authority_body,
        "authority_seal_sha256": canonical_sha256(authority_body),
    }
    authority_path = (
        archive_root
        / "remote-authorities"
        / (
            f"key-job-{job_sha256}-{manifest_sha256}-"
            f"{authority['authority_seal_sha256']}.json"
        )
    )
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    body = {
        "contract": "QR_DOJO_HISTORICAL_JOB_DRIVE_ATTESTATION_BODY_V2",
        "schema_version": 2,
        "attestation_id": attestation_id,
        "provider": "GOOGLE_DRIVE",
        "verification_method": (
            "GOOGLE_DRIVE_V3_FILES_GET_REVISIONS_LIST_AND_"
            "INDEPENDENT_REVISION_READBACK"
        ),
        "issued_at_utc": "2026-07-23T00:02:00+00:00",
        "expires_at_utc": "2026-07-23T00:12:00+00:00",
        "job_sha256": job_sha256,
        "completion_sha256": "b" * 64,
        "bundle_kind": "SUCCESS_ECONOMIC",
        "manifest_sha256": manifest_sha256,
        "local_archive_receipt_sha256": local_receipt_sha256,
        "archive_sha256": "c" * 64,
        "archive_size_bytes": 100,
        "object_set_sha256": "d" * 64,
        "object_count": 1,
        "expected_drive_parent_id": parent_id,
        "drive_parent": {},
        "readback_at_utc": "2026-07-23T00:01:00+00:00",
        "objects": [],
        "files_get_before_after_verified": True,
        "revisions_list_head_present_unique": True,
        "independent_revision_readback_verified": True,
        "exact_revision_bytes_hashed": True,
        "download_bytes_match_local_objects": True,
        "drive_metadata_revision_bound": True,
        "drive_parents_bound": True,
        "drive_trashed_false": True,
        "external_readback_attested": True,
        "remote_verified": True,
        "raw_reclaim_eligible": True,
        "historical_train_is_proof": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    envelope_body = {
        "contract": "QR_DOJO_HISTORICAL_JOB_DRIVE_SIGNED_ATTESTATION_V2",
        "schema_version": 2,
        "algorithm": "ED25519",
        "public_key_sha256": public_key_sha256,
        "body": body,
        "signature_base64": "A" * 88,
    }
    envelope = {
        **envelope_body,
        "remote_receipt_sha256": canonical_sha256(envelope_body),
    }
    signed_path = (
        archive_root
        / "remote-receipts"
        / (
            f"signed-job-{job_sha256}-{manifest_sha256}-"
            f"{envelope['remote_receipt_sha256']}.json"
        )
    )
    signed_path.parent.mkdir(parents=True, exist_ok=True)
    signed_path.write_text(json.dumps(envelope), encoding="utf-8")
    return signed_path, authority_path


def test_reviewed_control_is_six_month_research_only() -> None:
    control = _verified_control(CONTROL_PATH, repo_root=REPO_ROOT)

    assert control["contract"] == CONTROL_CONTRACT
    assert control["trainer_milestones"]["m5_completed_months_per_review"] == 6
    assert control["trainer_milestones"]["partial_month_tuning_allowed"] is False
    assert (
        control["trainer_milestones"]["parameter_change_applies_only_to_new_generation"]
        is True
    )
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


def test_control_accepts_absolute_archive_runtime_paths(tmp_path: Path) -> None:
    control = _load(ROOM_CONTROL_PATH)
    zstd = tmp_path / "zstd"
    zstd.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    zstd.chmod(0o700)
    staging = tmp_path / "staging"
    control["execution"]["archive_local_staging_root"] = str(staging)
    control["execution"]["archive_zstd_executable"] = str(zstd)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    verified = _verified_control(path, repo_root=REPO_ROOT)

    assert _archive_runtime_options(verified) == {
        "local_staging_root": staging,
        "zstd_bin": str(zstd),
    }


def _new_room_archive_runtime_control(tmp_path: Path) -> tuple[dict, Path, Path]:
    control = _load(ROOM_CONTROL_PATH)
    v2_control = _load(ROOM_CONTROL_V2_PATH)
    control["fixed_inputs"]["strategy_generation_ordinal"] = 3
    control["fixed_inputs"]["study_profile_policy"] = copy.deepcopy(
        v2_control["fixed_inputs"]["study_profile_policy"]
    )
    control["trainer_milestones"].pop("non_overlapping_six_month_blocks_required")
    control["trainer_milestones"]["non_overlapping_review_blocks_required"] = True
    control["execution"]["capacity_baseline"] = copy.deepcopy(
        v2_control["execution"]["capacity_baseline"]
    )
    control["execution"]["lifecycle_barrier_policy"] = copy.deepcopy(
        v2_control["execution"]["lifecycle_barrier_policy"]
    )
    control["execution"]["output_root"] = str(tmp_path / "run")
    control["execution"]["archive_root"] = str(tmp_path / "archive")
    control["execution"]["allowed_command"] = (
        "scripts/run-dojo-historical-train-control.py step"
    )
    staging = tmp_path / "staging"
    staging.mkdir(parents=True)
    zstd = tmp_path / "zstd"
    zstd.write_text("#!/bin/sh\nprintf 'zstd fake 1.0\\n'\n", encoding="utf-8")
    zstd.chmod(0o700)
    raw = zstd.read_bytes()
    control["execution"].update(
        {
            "archive_local_staging_root": str(staging),
            "archive_zstd_executable": str(zstd),
            "archive_zstd_sha256": hashlib.sha256(raw).hexdigest(),
            "archive_zstd_size_bytes": len(raw),
            "archive_zstd_version": "zstd fake 1.0",
        }
    )
    return control, staging, zstd


def test_new_room_generation_seals_zstd_and_staging_runtime(tmp_path: Path) -> None:
    control, staging, zstd = _new_room_archive_runtime_control(tmp_path)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    verified = _verified_control(path, repo_root=REPO_ROOT)
    binding = _archive_runtime_binding(verified)

    assert binding == {
        "local_staging_root": str(staging),
        "local_staging_device": staging.stat().st_dev,
        "zstd_executable": str(zstd),
        "zstd_sha256": control["execution"]["archive_zstd_sha256"],
        "zstd_size_bytes": control["execution"]["archive_zstd_size_bytes"],
        "zstd_version": "zstd fake 1.0",
    }
    zstd.write_text("#!/bin/sh\nprintf 'zstd fake 2.0\\n'\n", encoding="utf-8")
    with pytest.raises(DojoHistoricalTrainControlError, match="sealed metadata"):
        _archive_runtime_binding(verified)


def test_new_room_generation_manifests_seal_archive_runtime(tmp_path: Path) -> None:
    control, _, _ = _new_room_archive_runtime_control(tmp_path)
    control["execution"]["global_heavy_lock_path"] = str(tmp_path / "global.lock")
    control["execution"]["conflicting_execution_roots"] = []
    control["execution"]["conflicting_run_lock_paths"] = []
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    prepare_generation(repo_root=REPO_ROOT, run_control_path=path)
    root = Path(control["execution"]["output_root"])
    manifest = _load(root / "control-manifest.json")
    implementation = _load(root / "sealed-inputs" / "implementation-manifest.json")

    assert (
        manifest["archive_runtime_binding"] == implementation["archive_runtime_binding"]
    )
    assert (
        manifest["archive_runtime_binding"]["zstd_sha256"]
        == control["execution"]["archive_zstd_sha256"]
    )


def test_new_room_custody_compatibility_is_separate_from_run_exactness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, _, _ = _new_room_archive_runtime_control(tmp_path)
    control["execution"]["global_heavy_lock_path"] = str(tmp_path / "global.lock")
    control["execution"]["conflicting_execution_roots"] = []
    control["execution"]["conflicting_run_lock_paths"] = []
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")
    prepare_generation(repo_root=REPO_ROOT, run_control_path=path)
    manifest = _load(
        Path(control["execution"]["output_root"])
        / "sealed-inputs"
        / "implementation-manifest.json"
    )
    sealed = manifest["custody_control_plane_binding"]
    compatible_current = {
        **sealed,
        "inventory": [{"relative_path": "updated-compatible-control-plane"}],
        "inventory_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._custody_control_plane_binding",
        lambda repo: compatible_current,
    )

    _load_generation(
        repo_root=REPO_ROOT,
        run_control_path=path,
        operation="custody",
    )
    with pytest.raises(DojoHistoricalTrainControlError, match="compatibility"):
        _load_generation(
            repo_root=REPO_ROOT,
            run_control_path=path,
            operation="run",
        )


def test_new_room_generation_rejects_cloud_storage_staging(
    tmp_path: Path,
) -> None:
    control, _, _ = _new_room_archive_runtime_control(tmp_path)
    staging = tmp_path / "Library" / "CloudStorage" / "staging"
    staging.mkdir(parents=True)
    control["execution"]["archive_local_staging_root"] = str(staging)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError, match="CloudStorage"):
        _verified_control(path, repo_root=REPO_ROOT)


def test_new_room_generation_requires_archive_runtime_binding(
    tmp_path: Path,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    v2_control = _load(ROOM_CONTROL_V2_PATH)
    control["fixed_inputs"]["strategy_generation_ordinal"] = 3
    control["fixed_inputs"]["study_profile_policy"] = copy.deepcopy(
        v2_control["fixed_inputs"]["study_profile_policy"]
    )
    control["trainer_milestones"].pop("non_overlapping_six_month_blocks_required")
    control["trainer_milestones"]["non_overlapping_review_blocks_required"] = True
    control["execution"]["capacity_baseline"] = copy.deepcopy(
        v2_control["execution"]["capacity_baseline"]
    )
    control["execution"]["lifecycle_barrier_policy"] = copy.deepcopy(
        v2_control["execution"]["lifecycle_barrier_policy"]
    )
    control["execution"]["allowed_command"] = (
        "scripts/run-dojo-historical-train-control.py step"
    )
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError, match="require"):
        _verified_control(path, repo_root=REPO_ROOT)


def test_new_room_generation_rejects_non_step_heartbeat_command(
    tmp_path: Path,
) -> None:
    control, _, _ = _new_room_archive_runtime_control(tmp_path)
    control["execution"]["allowed_command"] = (
        "scripts/run-dojo-historical-train-control.py run-next"
    )
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError, match="exact step"):
        _verified_control(path, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    "key",
    ["archive_local_staging_root", "archive_zstd_executable"],
)
def test_control_rejects_relative_archive_runtime_paths(
    tmp_path: Path,
    key: str,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"][key] = "relative/path"
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError, match="archive"):
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


def test_r8_control_uses_the_only_allowed_heartbeat_entrypoint() -> None:
    control = _verified_control(ROOM_CONTROL_V2_PATH, repo_root=REPO_ROOT)

    assert control["fixed_inputs"]["strategy_generation_ordinal"] == 3
    assert control["fixed_inputs"]["study_profile_policy"]["train_month_count"] == len(
        control["fixed_inputs"]["train_months"]
    )
    assert (
        control["fixed_inputs"]["study_profile_policy"]["review_cadence_months"]
        == control["trainer_milestones"]["m5_completed_months_per_review"]
    )
    assert control["execution"]["allowed_command"] == (
        "scripts/run-dojo-historical-train-control.py step"
    )


def _set_dynamic_room_train_window(
    control: dict, *, months: list[str], cadence: int
) -> None:
    control["fixed_inputs"]["train_months"] = months
    policy = control["fixed_inputs"]["study_profile_policy"]
    policy["train_month_count"] = len(months)
    policy["review_cadence_months"] = cadence
    policy["review_blocks"] = [
        {
            "review_block_id": f"review-block-{index + 1:04d}",
            "train_months": months[index * cadence : (index + 1) * cadence],
        }
        for index in range(len(months) // cadence)
    ]
    control["trainer_milestones"]["m5_completed_months_per_review"] = cadence


@pytest.mark.parametrize(
    "months",
    [
        [f"2025-{month:02d}" for month in range(7, 13)],
        [f"2025-{month:02d}" for month in range(1, 13)],
    ],
)
def test_r8_room_study_window_changes_by_sealed_config_only(
    tmp_path: Path, months: list[str]
) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    _set_dynamic_room_train_window(control, months=months, cadence=6)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    verified = _verified_control(path, repo_root=REPO_ROOT)

    assert verified["fixed_inputs"]["train_months"] == months
    assert (
        len(verified["fixed_inputs"]["study_profile_policy"]["review_blocks"])
        == len(months) // 6
    )


def test_r8_twelve_month_profile_preserves_economic_and_strategy_digests(
    tmp_path: Path,
) -> None:
    six_month_control, _, _ = _new_room_archive_runtime_control(tmp_path)
    six_month_control["execution"]["output_root"] = str(tmp_path / "run-six")
    six_month_control["execution"]["archive_root"] = str(tmp_path / "archive-six")
    six_month_path = tmp_path / "six-month.json"
    six_month_path.write_text(json.dumps(six_month_control), encoding="utf-8")
    prepare_generation(repo_root=REPO_ROOT, run_control_path=six_month_path)

    twelve_month_control = copy.deepcopy(six_month_control)
    twelve_month_control["execution"]["output_root"] = str(tmp_path / "run-twelve")
    twelve_month_control["execution"]["archive_root"] = str(tmp_path / "archive-twelve")
    months = [f"2025-{month:02d}" for month in range(1, 13)]
    _set_dynamic_room_train_window(
        twelve_month_control,
        months=months,
        cadence=6,
    )
    twelve_month_path = tmp_path / "twelve-month.json"
    twelve_month_path.write_text(json.dumps(twelve_month_control), encoding="utf-8")
    prepared = prepare_generation(
        repo_root=REPO_ROOT,
        run_control_path=twelve_month_path,
    )

    six_month_plan = _load(tmp_path / "run-six" / "plan.json")
    twelve_month_plan = _load(tmp_path / "run-twelve" / "plan.json")
    twelve_month_manifest = _load(tmp_path / "run-twelve" / "control-manifest.json")
    twelve_month_schedule = _load(tmp_path / "run-twelve" / "schedule.json")
    milestone = _milestone_status(
        tmp_path / "run-twelve", twelve_month_schedule, publish=False
    )
    assert prepared["stream_job_count"] == 24
    assert prepared["result_coordinate_count"] == 288
    assert twelve_month_plan["period"]["calendar_months"] == months
    assert twelve_month_manifest["study_profile_policy"]["train_month_count"] == 12
    assert milestone["trainer_review_month_count"] == 6
    assert milestone["next_trainer_review_at_completed_m5_month_count"] == 6
    assert milestone["non_overlapping_review_blocks_required"] is True
    assert (
        twelve_month_plan["implementation_binding"]
        == six_month_plan["implementation_binding"]
    )
    assert twelve_month_plan["portfolio"] == six_month_plan["portfolio"]


@pytest.mark.parametrize(
    "months",
    [
        ["2025-01", "2025-02", "2025-04", "2025-05", "2025-06", "2025-07"],
        ["2025-01", "2025-02", "2025-02", "2025-03", "2025-04", "2025-05"],
    ],
)
def test_r8_room_study_window_rejects_gaps_and_duplicates(
    tmp_path: Path, months: list[str]
) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    _set_dynamic_room_train_window(control, months=months, cadence=6)
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError, match="contiguous|duplicate"):
        _verified_control(path, repo_root=REPO_ROOT)


def test_r8_room_study_window_rejects_cadence_mismatch(tmp_path: Path) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    control["trainer_milestones"]["m5_completed_months_per_review"] = 3
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError, match="cadence differs"):
        _verified_control(path, repo_root=REPO_ROOT)


def test_r8_room_study_window_rejects_overlapping_review_blocks(
    tmp_path: Path,
) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    months = [f"2025-{month:02d}" for month in range(1, 13)]
    _set_dynamic_room_train_window(control, months=months, cadence=6)
    control["fixed_inputs"]["study_profile_policy"]["review_blocks"][1]["train_months"][
        0
    ] = "2025-06"
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DojoHistoricalTrainControlError, match="overlap"):
        _verified_control(path, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    (("raw_bytes_per_job", 100_000_000), ("coordinate_count", 2)),
)
def test_r8_rejects_self_reported_capacity_baseline_drift_before_claim(
    tmp_path: Path, field: str, value: int
) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    control["execution"]["capacity_baseline"][field] = value
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="claims differ from their sealed artifact",
    ):
        _verified_control(path, repo_root=REPO_ROOT)


def test_r8_rejects_tampered_capacity_baseline_artifact_before_claim(
    tmp_path: Path,
) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    source = (
        REPO_ROOT / control["execution"]["capacity_baseline"]["baseline_artifact_path"]
    )
    artifact = _load(source)
    artifact["measured_raw_bytes"] -= 1
    tampered = tmp_path / "capacity-baseline.json"
    tampered.write_text(json.dumps(artifact), encoding="utf-8")
    control["execution"]["capacity_baseline"]["baseline_artifact_path"] = str(tampered)
    control["execution"]["capacity_baseline"]["baseline_artifact_sha256"] = (
        hashlib.sha256(tampered.read_bytes()).hexdigest()
    )
    path = tmp_path / "control.json"
    path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="artifact schema or seal is invalid",
    ):
        _verified_control(path, repo_root=REPO_ROOT)


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
        coordinate["fold_label"] for coordinate in schedule["jobs"][0]["coordinates"]
    } == {row["room_id"] for row in _g2_room_bindings()}
    assert all(
        coordinate["active_worker_count"] == 1
        for coordinate in schedule["jobs"][0]["coordinates"]
    )
    assert [row["artifact_id"] for row in manifest["sealed_input_artifacts"]] == [
        "IMPLEMENTATION_MANIFEST",
        "RUN_CONTROL",
        "SOURCE_MANIFEST",
        "STRATEGY_REGISTRY",
    ]
    assert all(
        (tmp_path / "room-run" / row["relative_path"]).is_file()
        for row in manifest["sealed_input_artifacts"]
    )


def _prepare_archive_only_generation(tmp_path: Path) -> Path:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["output_root"] = str(tmp_path / "room-run")
    control["execution"]["archive_root"] = str(tmp_path / "archive")
    control["execution"]["global_heavy_lock_path"] = str(tmp_path / "global.lock")
    control["execution"]["conflicting_execution_roots"] = []
    control["execution"]["conflicting_run_lock_paths"] = []
    control_path = tmp_path / "room-control.json"
    control_path.write_text(json.dumps(control), encoding="utf-8")
    prepare_generation(repo_root=REPO_ROOT, run_control_path=control_path)
    return control_path


def test_archive_next_without_terminal_never_claims_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = _prepare_archive_only_generation(tmp_path)
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.cpu_count", lambda: 10
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.getloadavg",
        lambda: (1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.claim_next_long_horizon_job",
        lambda *args, **kwargs: pytest.fail("archive-next must never claim a job"),
    )

    result = archive_next_completed_job(
        repo_root=REPO_ROOT,
        run_control_path=control_path,
    )

    assert result["status"] == "ARCHIVE_NOT_PENDING"
    assert result["archive"] is None
    assert result["next_job_started"] is False


def test_archive_next_returns_one_recovery_without_claiming_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = _prepare_archive_only_generation(tmp_path)
    receipt = {"receipt_sha256": "1" * 64}
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.cpu_count", lambda: 10
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.getloadavg",
        lambda: (1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._archive_pending_completed_jobs",
        lambda **kwargs: [receipt],
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.evaluate_historical_lifecycle",
        lambda **kwargs: {
            "state": "COMPLETION_PUBLISHED",
            "next_transition": "ARCHIVE_NEXT",
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.claim_next_long_horizon_job",
        lambda *args, **kwargs: pytest.fail("archive-next must never claim a job"),
    )

    result = archive_next_completed_job(
        repo_root=REPO_ROOT,
        run_control_path=control_path,
    )

    assert result["status"] == "ARCHIVE_RECOVERED"
    assert result["archive"] == receipt
    assert result["next_job_started"] is False


@pytest.mark.parametrize(
    ("transition", "expected_operation"),
    [("ARCHIVE_NEXT", "archive"), ("CLAIM_NEXT_JOB", "run")],
)
def test_heartbeat_step_executes_exactly_one_selected_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
    expected_operation: str,
) -> None:
    root = tmp_path / "run"
    generation = ({}, root, {}, {}, {}, {}, {})
    calls: list[str] = []
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._load_generation",
        lambda **kwargs: generation,
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.evaluate_historical_lifecycle",
        lambda **kwargs: {
            "state": "COMPLETION_PUBLISHED"
            if transition == "ARCHIVE_NEXT"
            else "READY_TO_CLAIM",
            "next_transition": transition,
            "lifecycle_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.archive_next_completed_job",
        lambda **kwargs: calls.append("archive") or {"status": "ARCHIVED"},
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.run_next_job",
        lambda **kwargs: calls.append("run") or {"status": "RAN"},
    )

    result = advance_one_historical_transition(
        repo_root=REPO_ROOT,
        run_control_path=tmp_path / "control.json",
    )

    assert calls == [expected_operation]
    assert result["heartbeat_step"] == {
        "selected_transition": transition,
        "transition_execution_count": 1,
        "fallthrough_allowed": False,
        "lifecycle_before_sha256": "1" * 64,
        "operation_revalidated_under_lock": True,
    }


def test_heartbeat_step_does_not_fall_through_when_no_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._load_generation",
        lambda **kwargs: ({}, root, {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.evaluate_historical_lifecycle",
        lambda **kwargs: {
            "state": "RUNNING",
            "next_transition": "NONE",
            "lifecycle_sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.archive_next_completed_job",
        lambda **kwargs: pytest.fail("step must not fall through to archive"),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.run_next_job",
        lambda **kwargs: pytest.fail("step must not fall through to run"),
    )

    result = advance_one_historical_transition(
        repo_root=REPO_ROOT,
        run_control_path=tmp_path / "control.json",
    )

    assert result["status"] == "NO_LIFECYCLE_TRANSITION"
    assert result["transition_execution_count"] == 0
    assert result["fallthrough_allowed"] is False


def _fake_execution_status(
    *, terminal: int, ready: int, active: int = 0, exhausted: int = 0
) -> dict:
    return {
        "terminal_job_count": terminal,
        "ready_job_count": ready,
        "active_job_count": active,
        "exhausted_job_count": exhausted,
        "status_sha256": "f" * 64,
    }


def _publish_fake_terminal(root: Path, job_sha: str) -> str:
    terminal_sha = "b" * 64
    claim_path = root / "execution-state" / "claims" / job_sha / "attempt-0001.json"
    claim_path.parent.mkdir(parents=True)
    claim_path.write_text("{}", encoding="utf-8")
    path = (
        root
        / "execution-state"
        / "terminals"
        / job_sha
        / f"attempt-0001-{'a' * 64}-{terminal_sha}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return terminal_sha


def _publish_fake_active_claim(root: Path, job_sha: str, *, attempt: int = 1) -> None:
    path = root / "execution-state" / "claims" / job_sha / f"attempt-{attempt:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def _publish_fake_completion(root: Path, job_sha: str, terminal_sha: str) -> None:
    body = {
        "contract": "QR_DOJO_HISTORICAL_TRAIN_JOB_COMPLETION_V1",
        "schema_version": 1,
        "job_sha256": job_sha,
        "terminal_sha256": terminal_sha,
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    path = root / "jobs" / job_sha / "completion.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({**body, "completion_sha256": canonical_sha256(body)}),
        encoding="utf-8",
    )


def test_lifecycle_ready_terminal_and_completion_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "1" * 64
    control = {
        "fixed_inputs": {},
        "execution": {"archive_root": str(tmp_path / "archive")},
    }
    schedule = {"jobs": [{"job_sha256": job_sha}]}
    statuses = iter(
        (
            _fake_execution_status(terminal=0, ready=1),
            _fake_execution_status(terminal=1, ready=0),
            _fake_execution_status(terminal=1, ready=0),
        )
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: next(statuses),
    )

    ready = evaluate_historical_lifecycle(
        root=tmp_path, control=control, plan={}, schedule=schedule
    )
    assert (ready["state"], ready["next_transition"]) == (
        "READY_TO_CLAIM",
        "CLAIM_NEXT_JOB",
    )

    terminal_sha = _publish_fake_terminal(tmp_path, job_sha)
    unpublished = evaluate_historical_lifecycle(
        root=tmp_path, control=control, plan={}, schedule=schedule
    )
    assert unpublished["state"] == "TERMINAL_UNPUBLISHED"
    assert unpublished["next_transition"] == "NONE"
    assert unpublished["blockers"] == [f"INCOMPLETE_TERMINAL_PUBLICATION:{job_sha}"]

    _publish_fake_completion(tmp_path, job_sha, terminal_sha)
    completed = evaluate_historical_lifecycle(
        root=tmp_path, control=control, plan={}, schedule=schedule
    )
    assert (completed["state"], completed["next_transition"]) == (
        "COMPLETION_PUBLISHED",
        "ARCHIVE_NEXT",
    )


def test_lifecycle_rejects_completion_without_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "2" * 64
    _publish_fake_completion(tmp_path, job_sha, "b" * 64)
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=0, ready=1),
    )

    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {}},
        plan={},
        schedule={"jobs": [{"job_sha256": job_sha}]},
    )

    assert lifecycle["state"] == "BLOCKED"
    assert f"COMPLETION_WITHOUT_TERMINAL:{job_sha}" in lifecycle["blockers"]


def test_lifecycle_uses_latest_attempt_for_running_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "7" * 64
    _publish_fake_terminal(tmp_path, job_sha)
    _publish_fake_active_claim(tmp_path, job_sha, attempt=2)
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=0, ready=0, active=1),
    )

    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {}},
        plan={},
        schedule={"jobs": [{"job_sha256": job_sha}]},
    )

    assert lifecycle["state"] == "RUNNING"
    assert lifecycle["job_states"][0]["job_sha256"] == job_sha
    assert lifecycle["job_states"][0]["state"] == "RUNNING"
    assert lifecycle["job_states"][0]["custody_validation"] == "NOT_APPLICABLE"
    assert lifecycle["blockers"] == []


def test_lifecycle_blocks_running_with_unsettled_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = "3" * 64
    active = "4" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, predecessor)
    _publish_fake_completion(tmp_path, predecessor, terminal_sha)
    _publish_fake_active_claim(tmp_path, active)
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=0, active=1),
    )

    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {}},
        plan={},
        schedule={
            "jobs": [
                {"job_sha256": predecessor},
                {"job_sha256": active},
            ]
        },
    )

    assert lifecycle["state"] == "BLOCKED"
    assert "RUNNING_WITH_UNSETTLED_PREDECESSOR" in lifecycle["blockers"]


def test_lifecycle_remote_receipt_is_candidate_not_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "5" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, job_sha)
    _publish_fake_completion(tmp_path, job_sha, terminal_sha)
    archive = tmp_path / "archive"
    local = archive / "receipts" / f"job-{job_sha}-{'6' * 64}.json"
    remote = (
        archive / "remote-receipts" / f"remote-job-{job_sha}-{'7' * 64}-{'8' * 64}.json"
    )
    local.parent.mkdir(parents=True)
    remote.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    remote.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_archive_receipt",
        lambda **kwargs: {"job_sha256": job_sha, "manifest_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_remote_receipt",
        lambda **kwargs: {},
    )

    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {"archive_root": str(archive)}},
        plan={},
        schedule={"jobs": [{"job_sha256": job_sha}]},
    )

    assert lifecycle["state"] == "UNSIGNED_REMOTE_RECEIPT_CANDIDATE"
    assert lifecycle["remote_attestation_authorized"] is False


def test_lifecycle_recognizes_exact_one_signed_v2_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "1" * 64
    manifest_sha = "2" * 64
    local_sha = "3" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, job_sha)
    _publish_fake_completion(tmp_path, job_sha, terminal_sha)
    archive = tmp_path / "archive"
    local = archive / "receipts" / f"job-{job_sha}-{manifest_sha}.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    _publish_compact_signed_attestation(
        archive_root=archive,
        job_sha256=job_sha,
        manifest_sha256=manifest_sha,
        local_receipt_sha256=local_sha,
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_archive_receipt",
        lambda **kwargs: {
            "job_sha256": job_sha,
            "manifest_sha256": manifest_sha,
            "receipt_sha256": local_sha,
        },
    )
    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={
            "fixed_inputs": {},
            "execution": {
                "archive_root": str(archive),
                "archive_drive_readback_parent_id": "driveParent123456",
            },
        },
        plan={},
        schedule={"jobs": [{"job_sha256": job_sha}]},
    )

    assert lifecycle["state"] == "SIGNED_REMOTE_ATTESTATION_CANDIDATE"
    assert lifecycle["job_states"][0]["remote_attestation_verified"] is False
    assert lifecycle["deep_custody_verification_before_claim"] is True


def test_lifecycle_rejects_multiple_signed_v2_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "4" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, job_sha)
    _publish_fake_completion(tmp_path, job_sha, terminal_sha)
    archive = tmp_path / "archive"
    local = archive / "receipts" / f"job-{job_sha}-{'5' * 64}.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    remote_root = archive / "remote-receipts"
    remote_root.mkdir(parents=True)
    for suffix in ("6", "7"):
        (
            remote_root / f"signed-job-{job_sha}-{'5' * 64}-{suffix * 64}.json"
        ).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=0),
    )

    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {"archive_root": str(archive)}},
        plan={},
        schedule={"jobs": [{"job_sha256": job_sha}]},
    )

    assert lifecycle["state"] == "BLOCKED"
    assert f"DUPLICATE_LIFECYCLE_ARTIFACT:{job_sha}" in lifecycle["blockers"]


def test_lifecycle_rejects_signed_v2_candidate_for_unknown_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled = "8" * 64
    unknown = "9" * 64
    archive = tmp_path / "archive"
    remote_root = archive / "remote-receipts"
    remote_root.mkdir(parents=True)
    (remote_root / f"signed-job-{unknown}-{'a' * 64}-{'b' * 64}.json").write_text(
        "{}", encoding="utf-8"
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=0, ready=1),
    )

    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {"archive_root": str(archive)}},
        plan={},
        schedule={"jobs": [{"job_sha256": scheduled}]},
    )

    assert lifecycle["state"] == "BLOCKED"
    assert f"UNKNOWN_CUSTODY_JOB:{unknown}" in lifecycle["blockers"]


def test_signed_v2_candidate_is_deep_verified_before_next_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "c" * 64
    manifest_sha = "d" * 64
    local_sha = "e" * 64
    archive = tmp_path / "archive"
    completion = tmp_path / "jobs" / job_sha / "completion.json"
    completion.parent.mkdir(parents=True)
    completion.write_text("{}", encoding="utf-8")
    local = archive / "receipts" / f"job-{job_sha}-{manifest_sha}.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    signed, authority = _publish_compact_signed_attestation(
        archive_root=archive,
        job_sha256=job_sha,
        manifest_sha256=manifest_sha,
        local_receipt_sha256=local_sha,
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_archive_receipt",
        lambda **kwargs: {
            "job_sha256": job_sha,
            "manifest_sha256": manifest_sha,
            "receipt_sha256": local_sha,
        },
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.verify_historical_job_raw_reclaim",
        lambda **kwargs: calls.append(kwargs),
    )

    _deep_verify_completed_job_custody(
        root=tmp_path,
        control={
            "fixed_inputs": {},
            "execution": {
                "archive_root": str(archive),
                "archive_drive_readback_parent_id": "driveParent123456",
            },
        },
    )

    assert len(calls) == 1
    assert calls[0]["remote_receipt_path"] == signed
    assert calls[0]["attestation_authority_seal_path"] == authority


@pytest.mark.parametrize("unsigned_v1_present", (False, True))
def test_r8_claim_barrier_waits_for_signed_attestation_after_local_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsigned_v1_present: bool,
) -> None:
    completed = "1" * 64
    ready = "2" * 64
    manifest_sha = "3" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, completed)
    _publish_fake_completion(tmp_path, completed, terminal_sha)
    archive = tmp_path / "archive"
    local = archive / "receipts" / f"job-{completed}-{manifest_sha}.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    if unsigned_v1_present:
        remote = (
            archive
            / "remote-receipts"
            / (f"remote-job-{completed}-{manifest_sha}-{'4' * 64}.json")
        )
        remote.parent.mkdir(parents=True)
        remote.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "quant_rabbit.dojo_historical_train_control._compact_remote_receipt",
            lambda **kwargs: {},
        )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=1),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_archive_receipt",
        lambda **kwargs: {
            "job_sha256": completed,
            "manifest_sha256": manifest_sha,
            "receipt_sha256": "5" * 64,
        },
    )
    control = {
        "fixed_inputs": {},
        "execution": {
            "archive_root": str(archive),
            "lifecycle_barrier_policy": _load(ROOM_CONTROL_V2_PATH)["execution"][
                "lifecycle_barrier_policy"
            ],
        },
    }

    first = evaluate_historical_lifecycle(
        root=tmp_path,
        control=control,
        plan={},
        schedule={
            "jobs": [
                {"job_sha256": completed},
                {"job_sha256": ready},
            ]
        },
    )
    second = evaluate_historical_lifecycle(
        root=tmp_path,
        control=control,
        plan={},
        schedule={
            "jobs": [
                {"job_sha256": completed},
                {"job_sha256": ready},
            ]
        },
    )

    assert first["state"] == "WAIT_FOR_SIGNED_REMOTE_ATTESTATION"
    assert first["next_transition"] == "NONE"
    assert first["unreclaimed_terminal_job_count"] == 1
    assert first["lifecycle_sha256"] == second["lifecycle_sha256"]


def test_r8_claim_barrier_waits_for_exact_reclaim_then_allows_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = "6" * 64
    ready = "7" * 64
    manifest_sha = "8" * 64
    local_sha = "9" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, completed)
    _publish_fake_completion(tmp_path, completed, terminal_sha)
    archive = tmp_path / "archive"
    local = archive / "receipts" / f"job-{completed}-{manifest_sha}.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    _publish_compact_signed_attestation(
        archive_root=archive,
        job_sha256=completed,
        manifest_sha256=manifest_sha,
        local_receipt_sha256=local_sha,
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=1),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_archive_receipt",
        lambda **kwargs: {
            "job_sha256": completed,
            "manifest_sha256": manifest_sha,
            "receipt_sha256": local_sha,
        },
    )
    control = {
        "fixed_inputs": {},
        "execution": {
            "archive_root": str(archive),
            "archive_drive_readback_parent_id": "driveParent123456",
            "lifecycle_barrier_policy": _load(ROOM_CONTROL_V2_PATH)["execution"][
                "lifecycle_barrier_policy"
            ],
        },
    }
    schedule = {
        "jobs": [
            {"job_sha256": completed},
            {"job_sha256": ready},
        ]
    }

    waiting = evaluate_historical_lifecycle(
        root=tmp_path,
        control=control,
        plan={},
        schedule=schedule,
    )

    assert waiting["state"] == "WAIT_FOR_EXACT_V2_RAW_RECLAIM"
    assert waiting["next_transition"] == "NONE"
    reclaim = tmp_path / "reclaim-v2-receipts" / f"reclaim-{completed}-{'a' * 64}.json"
    reclaim.parent.mkdir(parents=True)
    reclaim.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_reclaim_receipt",
        lambda **kwargs: {},
    )

    ready_to_claim = evaluate_historical_lifecycle(
        root=tmp_path,
        control=control,
        plan={},
        schedule=schedule,
    )

    assert ready_to_claim["state"] == "READY_TO_CLAIM"
    assert ready_to_claim["next_transition"] == "CLAIM_NEXT_JOB"
    assert ready_to_claim["unreclaimed_terminal_job_count"] == 0


def test_lifecycle_local_archive_is_explicitly_compact_not_deep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "6" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, job_sha)
    _publish_fake_completion(tmp_path, job_sha, terminal_sha)
    archive = tmp_path / "archive"
    local = archive / "receipts" / f"job-{job_sha}-{'7' * 64}.json"
    local.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._compact_archive_receipt",
        lambda **kwargs: {"job_sha256": job_sha, "manifest_sha256": "8" * 64},
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.verify_existing_historical_job_archive",
        lambda **kwargs: pytest.fail("status lifecycle must remain compact-only"),
    )

    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {"archive_root": str(archive)}},
        plan={},
        schedule={"jobs": [{"job_sha256": job_sha}]},
    )

    assert lifecycle["state"] == "LOCAL_ARCHIVED"
    assert lifecycle["status_custody_validation"] == "COMPACT_ONLY"
    assert lifecycle["job_states"][0]["local_archive_deep_verified"] is False
    assert lifecycle["job_states"][0]["custody_validation"] == (
        "COMPACT_ONLY_DEEP_REQUIRED_BEFORE_CLAIM"
    )


def test_lifecycle_accepts_restore_receipt_as_compact_only_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_sha = "9" * 64
    terminal_sha = _publish_fake_terminal(tmp_path, job_sha)
    _publish_fake_completion(tmp_path, job_sha, terminal_sha)
    archive = tmp_path / "archive"
    local = archive / "receipts" / f"job-{job_sha}-{'a' * 64}.json"
    reclaim = tmp_path / "reclaim-receipts" / f"reclaim-{job_sha}-{'b' * 64}.json"
    local.parent.mkdir(parents=True)
    reclaim.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    reclaim.write_text("{}", encoding="utf-8")
    restore_body = {
        "contract": "QR_DOJO_HISTORICAL_JOB_RAW_RESTORE_V1",
        "schema_version": 1,
        "status": "RAW_RESTORED",
        "job_sha256": job_sha,
        "completion_sha256": "c" * 64,
        "bundle_kind": "SUCCESS",
        "manifest_sha256": "d" * 64,
        "local_archive_receipt_sha256": "e" * 64,
        "archive_sha256": "f" * 64,
        "reclaim_plan_sha256": "1" * 64,
        "reclaim_receipt_sha256": "2" * 64,
        "restored_at_utc": "2026-07-23T00:00:00+00:00",
        "restored_file_count": 1,
        "restored_files": [
            {
                "relative_path": "jobs/restored/evidence.json",
                "size_bytes": 123,
                "sha256": "3" * 64,
            }
        ],
        "restored_logical_bytes": 123,
        "published_file_count": 1,
        "preexisting_matching_file_count": 0,
        "local_archive_deep_verified": True,
        "retained_bytes_verified": True,
        "all_raw_targets_present": True,
        "remote_receipt_trusted": False,
        "historical_train_is_proof": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    restore_sha = canonical_sha256(restore_body)
    restore = tmp_path / "restore-receipts" / f"restore-{job_sha}-{restore_sha}.json"
    restore.parent.mkdir(parents=True)
    restore.write_text(
        json.dumps({**restore_body, "restore_receipt_sha256": restore_sha}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.long_horizon_execution_status",
        lambda *args, **kwargs: _fake_execution_status(terminal=1, ready=0),
    )
    lifecycle = evaluate_historical_lifecycle(
        root=tmp_path,
        control={"fixed_inputs": {}, "execution": {"archive_root": str(archive)}},
        plan={},
        schedule={"jobs": [{"job_sha256": job_sha}]},
    )

    assert lifecycle["state"] == "RESTORED"
    assert lifecycle["raw_reclaim_transition_allowed"] is False
    assert lifecycle["status_custody_validation"] == "COMPACT_ONLY"
    assert lifecycle["job_states"][0]["local_archive_deep_verified"] is False
    assert lifecycle["job_states"][0]["custody_validation"] == (
        "COMPACT_ONLY_DEEP_REQUIRED_BEFORE_CLAIM"
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
    conflict_root = tmp_path / "old-run"
    conflict_root.mkdir()
    lock_path = conflict_root / ".historical-train.lock"
    lock_path.touch()
    control["execution"]["conflicting_execution_roots"] = [str(conflict_root)]
    control["execution"]["conflicting_run_lock_paths"] = [str(lock_path)]

    descriptors = _acquire_conflicting_run_locks(control)
    competing = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(competing)
        _release_historical_operation_locks(
            run_lock_descriptor=None,
            global_lock_descriptor=None,
            conflicting_lock_descriptors=descriptors,
        )


def test_lock_release_rejects_lock_rename_and_recreate(tmp_path: Path) -> None:
    path = tmp_path / "operation.lock"
    descriptor = _open_stable_lock_file(path)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path.rename(tmp_path / "operation.lock.replaced")
    path.touch()

    with pytest.raises(DojoHistoricalTrainControlError, match="replaced"):
        _release_one_historical_operation_lock(descriptor)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_lock_release_rejects_parent_rename_and_recreate(tmp_path: Path) -> None:
    parent = tmp_path / "locks"
    parent.mkdir()
    path = parent / "operation.lock"
    descriptor = _open_stable_lock_file(path)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent.rename(tmp_path / "old-locks")
    parent.mkdir()
    path.touch()

    with pytest.raises(DojoHistoricalTrainControlError, match="replaced"):
        _release_one_historical_operation_lock(descriptor)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_mutation_guard_rejects_recreated_lock_before_state_write(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["global_heavy_lock_path"] = str(tmp_path / "global.lock")
    control["execution"]["conflicting_execution_roots"] = []
    control["execution"]["conflicting_run_lock_paths"] = []
    run_descriptor, global_descriptor, conflicts = _acquire_historical_operation_locks(
        root=root, control=control
    )
    lock = root / ".historical-train.lock"
    lock.rename(root / ".historical-train.lock.replaced")
    lock.touch()

    with pytest.raises(DojoHistoricalTrainControlError, match="replaced"):
        _assert_historical_operation_lock_identities(
            run_lock_descriptor=run_descriptor,
            global_lock_descriptor=global_descriptor,
            conflicting_lock_descriptors=conflicts,
        )
    with pytest.raises(DojoHistoricalTrainControlError, match="replaced"):
        _release_historical_operation_locks(
            run_lock_descriptor=run_descriptor,
            global_lock_descriptor=global_descriptor,
            conflicting_lock_descriptors=conflicts,
        )


def test_global_lock_conflict_releases_the_already_acquired_run_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    global_lock = tmp_path / "global.lock"
    global_lock.touch()
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["global_heavy_lock_path"] = str(global_lock)
    control["execution"]["conflicting_execution_roots"] = []
    control["execution"]["conflicting_run_lock_paths"] = []
    owner = os.open(global_lock, os.O_RDWR)
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(
            DojoHistoricalTrainControlError,
            match="machine-wide lease",
        ):
            _acquire_historical_operation_locks(root=root, control=control)
        competing = os.open(root / ".historical-train.lock", os.O_RDWR)
        try:
            fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fcntl.flock(competing, fcntl.LOCK_UN)
            os.close(competing)
    finally:
        fcntl.flock(owner, fcntl.LOCK_UN)
        os.close(owner)


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
    control["execution"]["conflicting_run_lock_paths"] = [
        str(conflict_root / ".historical-train.lock")
    ]
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.cpu_count", lambda: 10
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.getloadavg",
        lambda: (1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._conflicting_generation_statuses",
        lambda value, **kwargs: [
            {
                "output_root": str(conflict_root),
                "exists": True,
                "active_job_count": 1,
                "terminal_job_count": 3,
                "status_sha256": "1" * 64,
                "superseded_by_current_generation": False,
                "supersede_receipt_sha256": None,
            }
        ],
    )

    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="active or orphaned claim",
    ):
        _assert_dynamic_machine_capacity(control)


def test_verified_supersede_excludes_orphan_from_active_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.cpu_count", lambda: 10
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.os.getloadavg",
        lambda: (1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._conflicting_generation_statuses",
        lambda value, **kwargs: [
            {
                "output_root": "/old",
                "exists": True,
                "active_job_count": 1,
                "terminal_job_count": 3,
                "status_sha256": "1" * 64,
                "superseded_by_current_generation": True,
                "supersede_receipt_sha256": "2" * 64,
            }
        ],
    )

    _assert_dynamic_machine_capacity(control)


def test_supersede_receipt_lookup_accepts_v2_durable_pending_anchor(
    tmp_path: Path,
) -> None:
    current_root = tmp_path / "current"
    conflicting_root = tmp_path / "old"
    store = current_root / "transition-receipts"
    store.mkdir(parents=True)
    identity = "a" * 64
    digest = "b" * 64
    final = store / f"supersede-{identity}-{digest}.json"
    payload = {
        "contract": "QR_DOJO_HISTORICAL_GENERATION_SUPERSEDE_RECEIPT_V2",
        "old_generation": {"root": str(conflicting_root)},
    }
    final.write_text(json.dumps(payload), encoding="utf-8")
    os.link(final, store / f".{final.name}.pending")

    assert (
        _find_supersede_receipt_for_root(
            current_root=current_root,
            conflicting_root=conflicting_root,
        )
        == final
    )


def test_supersede_receipt_chain_reaches_current_through_unique_successors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "old"
    middle = tmp_path / "middle"
    current = tmp_path / "current"
    for root in (old, middle, current):
        root.mkdir()
    control = _load(ROOM_CONTROL_PATH)
    configured_roots = sorted((old, middle))
    control["execution"]["conflicting_execution_roots"] = [
        str(root) for root in configured_roots
    ]
    control["execution"]["conflicting_run_lock_paths"] = [
        str(root / ".historical-train.lock") for root in configured_roots
    ]
    first = middle / "first.json"
    second = current / "second.json"
    paths = {(middle, old): first, (current, middle): second}
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control."
        "_find_supersede_receipt_for_root",
        lambda *, current_root, conflicting_root: paths.get(
            (current_root, conflicting_root)
        ),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control."
        "verify_historical_supersede_receipt_file",
        lambda path, **kwargs: {
            "receipt_sha256": "a" * 64 if path == first else "b" * 64
        },
    )

    chain = _verify_supersede_receipt_chain(
        control,
        current_root=current,
        conflicting_root=old,
    )

    assert [row["receipt_sha256"] for row in chain] == ["a" * 64, "b" * 64]


def test_conflicting_generation_status_reports_absent_root() -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["conflicting_execution_roots"] = [
        "/definitely/absent/dojo-generation"
    ]
    control["execution"]["conflicting_run_lock_paths"] = [
        "/definitely/absent/dojo-generation/.historical-train.lock"
    ]

    assert _conflicting_generation_statuses(control) == [
        {
            "output_root": "/definitely/absent/dojo-generation",
            "exists": False,
            "active_job_count": 0,
            "terminal_job_count": 0,
            "superseded_by_current_generation": False,
            "supersede_receipt_sha256": None,
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
        control["execution"]["minimum_free_disk_bytes"] + 5_000
    )
    assert snapshot["archive_required_bytes"] == snapshot["run_required_bytes"]
    assert snapshot["compression_ratio_assumed"] is False


def test_r8_separate_filesystems_each_keep_the_25gib_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    control["execution"]["output_root"] = str(tmp_path / "run")
    control["execution"]["archive_local_staging_root"] = str(tmp_path / "staging")
    control["execution"]["archive_root"] = str(tmp_path / "archive")
    floor = control["execution"]["minimum_free_disk_bytes"]
    assert control["execution"]["bootstrap_job_working_set_bytes"] < floor

    class Probe:
        def __init__(self, name: str, device: int) -> None:
            self.name = name
            self.device = device

        def stat(self, *, follow_symlinks: bool) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_dev=self.device)

        def __fspath__(self) -> str:
            return f"/{self.name}"

    probes = {
        "run": Probe("run", 11),
        "staging": Probe("staging", 22),
        "archive": Probe("archive", 33),
    }
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._nearest_existing_parent",
        lambda path: probes[path.name],
    )
    free_by_role = {
        "run": floor + 1_100,
        "staging": floor + 2_100,
        "archive": floor + 6_100,
    }
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.shutil.disk_usage",
        lambda probe: SimpleNamespace(free=free_by_role[probe.name]),
    )

    snapshot = _disk_capacity_snapshot(
        root=tmp_path / "run",
        control=control,
        estimated_raw_bytes=1_000,
        estimated_archive_upper_bytes=2_000,
        estimated_peak_bytes=9_000,
    )

    assert snapshot["shared_filesystem"] is False
    by_role = {row["roles"][0]: row for row in snapshot["filesystem_reservations"]}
    assert {role: row["recovery_floor_bytes"] for role, row in by_role.items()} == {
        "run": floor,
        "local_staging": floor,
        "archive": floor,
    }
    assert by_role["run"]["required_bytes"] == floor + 1_000
    assert by_role["local_staging"]["required_bytes"] == floor + 2_000
    assert by_role["archive"]["required_bytes"] == floor + 6_000
    _assert_disk_capacity(snapshot)


def test_r8_separate_archive_filesystem_cannot_fall_back_to_bootstrap_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _load(ROOM_CONTROL_V2_PATH)
    control["execution"]["output_root"] = str(tmp_path / "run")
    control["execution"]["archive_local_staging_root"] = str(tmp_path / "staging")
    control["execution"]["archive_root"] = str(tmp_path / "archive")
    floor = control["execution"]["minimum_free_disk_bytes"]
    bootstrap = control["execution"]["bootstrap_job_working_set_bytes"]

    class Probe:
        def __init__(self, name: str, device: int) -> None:
            self.name = name
            self.device = device

        def stat(self, *, follow_symlinks: bool) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_dev=self.device)

        def __fspath__(self) -> str:
            return f"/{self.name}"

    probes = {
        "run": Probe("run", 11),
        "staging": Probe("staging", 22),
        "archive": Probe("archive", 33),
    }
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._nearest_existing_parent",
        lambda path: probes[path.name],
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.shutil.disk_usage",
        lambda probe: SimpleNamespace(
            free=(bootstrap + 6_001 if probe.name == "archive" else floor + 10_000)
        ),
    )

    snapshot = _disk_capacity_snapshot(
        root=tmp_path / "run",
        control=control,
        estimated_raw_bytes=1_000,
        estimated_archive_upper_bytes=2_000,
        estimated_peak_bytes=9_000,
    )

    archive_row = next(
        row
        for row in snapshot["filesystem_reservations"]
        if row["roles"] == ["archive"]
    )
    assert archive_row["recovery_floor_bytes"] == floor
    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="filesystem cannot cover conservative DOJO reservations",
    ):
        _assert_disk_capacity(snapshot)


def test_r8_status_reports_25gib_floor_for_each_separate_capacity_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, _, _ = _new_room_archive_runtime_control(tmp_path)
    control["execution"]["global_heavy_lock_path"] = str(tmp_path / "global.lock")
    control["execution"]["conflicting_execution_roots"] = []
    control["execution"]["conflicting_run_lock_paths"] = []
    control_path = tmp_path / "r8-control.json"
    control_path.write_text(json.dumps(control), encoding="utf-8")
    prepare_generation(repo_root=REPO_ROOT, run_control_path=control_path)
    floor = control["execution"]["minimum_free_disk_bytes"]

    class Probe:
        def __init__(self, name: str, device: int) -> None:
            self.name = name
            self.device = device

        def stat(self, *, follow_symlinks: bool) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_dev=self.device)

        def __fspath__(self) -> str:
            return f"/{self.name}"

    probes = {
        "run": Probe("run", 11),
        "staging": Probe("staging", 22),
        "archive": Probe("archive", 33),
    }
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._nearest_existing_parent",
        lambda path: probes[path.name],
    )

    def disk_usage(probe: object) -> SimpleNamespace:
        if isinstance(probe, Probe):
            return SimpleNamespace(free=floor + 100 * 1024**3)
        return SimpleNamespace(free=floor + 100 * 1024**3)

    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.shutil.disk_usage",
        disk_usage,
    )

    status = generation_status(
        repo_root=REPO_ROOT,
        run_control_path=control_path,
    )

    assert status["disk_capacity"]["shared_filesystem"] is False
    reservations = status["disk_capacity"]["filesystem_reservations"]
    assert {row["roles"][0]: row["recovery_floor_bytes"] for row in reservations} == {
        "run": floor,
        "local_staging": floor,
        "archive": floor,
    }


def test_r8_capacity_baseline_reserves_one_measured_working_set_not_twelve() -> None:
    control = _verified_control(ROOM_CONTROL_V2_PATH, repo_root=REPO_ROOT)
    baseline = control["execution"]["capacity_baseline"]

    assert (
        _baseline_raw_bytes(
            control=control,
            planned_coordinate_count=12,
        )
        == baseline["raw_bytes_per_job"]
    )


def test_archive_staging_uses_sealed_conservative_floor(tmp_path: Path) -> None:
    control = _load(ROOM_CONTROL_PATH)

    assert _effective_archive_staging_fraction(
        control=control,
        archive_root=tmp_path / "absent-archive",
    ) == pytest.approx(0.3)


def test_local_archive_staging_reserves_both_local_and_destination_bytes(
    tmp_path: Path,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["archive_root"] = str(tmp_path / "archive")
    control["execution"]["archive_local_staging_root"] = str(tmp_path / "staging")

    recovery_body = {
        "contract": "QR_DOJO_HISTORICAL_JOB_ARCHIVE_RECOVERY_CAPACITY_V1",
        "schema_version": 1,
        "manifest_sha256": "f" * 64,
        "total_source_bytes": 1_000,
        "archive_upper_bound_bytes": 1_300,
        "manifest_payload_bytes": 100,
        "tar_pax_upper_bound_bytes": 1_100,
        "zstd_framing_upper_bound_bytes": 200,
        "remaining_local_staging_bytes": 400,
        "remaining_archive_filesystem_bytes": 600,
        "validated_local_archive_pending_bytes": 900,
        "compression_ratio_assumed": False,
    }
    snapshot = _assert_archive_staging_capacity(
        root=tmp_path / "run",
        control=control,
        recovery_capacity={
            **recovery_body,
            "capacity_inspection_sha256": canonical_sha256(recovery_body),
        },
    )

    assert snapshot["shared_filesystem"] is True
    assert snapshot["remaining_local_staging_bytes"] == 400
    assert snapshot["remaining_archive_filesystem_bytes"] == 600
    floor = control["execution"]["minimum_free_disk_bytes"]
    assert snapshot["local_staging_required_bytes"] == floor + 400
    assert snapshot["archive_required_bytes"] == floor + 600
    assert snapshot["validated_existing_bytes"] == {
        "validated_local_archive_pending_bytes": 900
    }
    assert snapshot["compression_ratio_assumed"] is False


@pytest.mark.parametrize("shared_filesystem", (True, False))
def test_archive_recovery_rejects_free_space_below_remaining_plus_25gib_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_filesystem: bool,
) -> None:
    control = _load(ROOM_CONTROL_PATH)
    control["execution"]["archive_root"] = str(tmp_path / "archive")
    control["execution"]["archive_local_staging_root"] = str(tmp_path / "staging")
    floor = control["execution"]["minimum_free_disk_bytes"]

    class Probe:
        def __init__(self, name: str, device: int) -> None:
            self.name = name
            self.device = device

        def stat(self, *, follow_symlinks: bool) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_dev=self.device)

        def __fspath__(self) -> str:
            return f"/{self.name}"

    devices = {
        "run": 1,
        "staging": 1 if shared_filesystem else 2,
        "archive": 1 if shared_filesystem else 3,
    }
    probes = {name: Probe(name, device) for name, device in devices.items()}
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._nearest_existing_parent",
        lambda path: probes[path.name],
    )

    def disk_usage(probe: Probe) -> SimpleNamespace:
        if shared_filesystem:
            return SimpleNamespace(free=floor + 999)
        if probe.name == "archive":
            return SimpleNamespace(free=floor + 599)
        return SimpleNamespace(free=floor + 10_000)

    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.shutil.disk_usage", disk_usage
    )
    recovery_body = {
        "contract": "QR_DOJO_HISTORICAL_JOB_ARCHIVE_RECOVERY_CAPACITY_V1",
        "schema_version": 1,
        "manifest_sha256": "f" * 64,
        "total_source_bytes": 1_000,
        "archive_upper_bound_bytes": 1_300,
        "manifest_payload_bytes": 100,
        "tar_pax_upper_bound_bytes": 1_100,
        "zstd_framing_upper_bound_bytes": 200,
        "remaining_local_staging_bytes": 400,
        "remaining_archive_filesystem_bytes": 600,
        "compression_ratio_assumed": False,
    }

    with pytest.raises(
        DojoHistoricalTrainControlError,
        match="cannot cover conservative DOJO reservations",
    ):
        _assert_archive_staging_capacity(
            root=tmp_path / "run",
            control=control,
            recovery_capacity={
                **recovery_body,
                "capacity_inspection_sha256": canonical_sha256(recovery_body),
            },
        )


def test_capacity_reservations_keep_separate_filesystems_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Probe:
        def __init__(self, name: str, device: int) -> None:
            self.name = name
            self.device = device

        def stat(self, *, follow_symlinks: bool) -> SimpleNamespace:
            assert follow_symlinks is False
            return SimpleNamespace(st_dev=self.device)

        def __fspath__(self) -> str:
            return f"/{self.name}"

    probes = {
        "run": Probe("run", 1),
        "staging": Probe("staging", 2),
        "archive": Probe("archive", 3),
    }
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control._nearest_existing_parent",
        lambda path: probes[path.name],
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_train_control.shutil.disk_usage",
        lambda path: SimpleNamespace(free=10_000),
    )

    reservations = _filesystem_capacity_reservations(
        (
            ("run", Path("run"), 1_000, 100),
            ("local_staging", Path("staging"), 1_000, 200),
            ("archive", Path("archive"), 3_000, 200),
        )
    )

    assert [row["required_bytes"] for row in reservations["devices"]] == [
        1_100,
        1_200,
        3_200,
    ]


def test_read_only_milestone_status_does_not_publish(tmp_path: Path) -> None:
    (tmp_path / "control-manifest.json").write_text(
        json.dumps(
            {
                "trainer_milestone_policy": {
                    "m5_completed_months_per_review": 6,
                    "non_overlapping_six_month_blocks_required": True,
                }
            }
        ),
        encoding="utf-8",
    )
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
