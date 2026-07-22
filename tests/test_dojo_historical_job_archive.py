from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fcntl
import pytest

import quant_rabbit.dojo_historical_job_archive as archive_module
from quant_rabbit.dojo_historical_job_archive import (
    ARCHIVE_RECOVERY_CAPACITY_CONTRACT,
    ARCHIVE_MANIFEST_CONTRACT,
    ARCHIVE_SOURCE_INSPECTION_CONTRACT,
    FAILED_SOURCE_BUNDLE_KIND,
    IMPLEMENTATION_MANIFEST_CONTRACT,
    JOB_COMPLETION_CONTRACT,
    SUCCESS_BUNDLE_KIND,
    DojoHistoricalJobArchiveError,
    _VerifiedSource,
    _canonical_bytes,
    _canonical_sha256,
    _tar_info,
    _verify_archive,
    archive_completed_historical_job,
    inspect_historical_job_archive_recovery_capacity,
    inspect_historical_job_archive_source,
    verify_existing_historical_job_archive,
)


JOB_SHA256 = "a" * 64


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _sealed(body: dict, field: str) -> dict:
    return {**body, field: _canonical_sha256(body)}


def _terminal_run(
    tmp_path: Path,
    *,
    payload_bytes: int = 2 * 1024 * 1024,
    incompressible: bool = False,
) -> Path:
    root = tmp_path / "run"
    implementation_digests = {
        "execution_protocol_sha256": "1" * 64,
        "replay_engine_sha256": "2" * 64,
    }
    safe_authority = {
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    for relative in (
        "candidate-proposals.json",
        "resource-policy.json",
        "schedule.json",
        "tuned-runtime-seal.json",
        "worker-catalog.json",
        "execution-state/execution-manifest.json",
    ):
        _write_json(root / relative, {"artifact": relative})
    _write_json(
        root / "plan.json",
        {"implementation_binding": {"digests": implementation_digests}},
    )
    run_control = {
        "contract": "QR_DOJO_G2_HISTORICAL_RUN_CONTROL_V1",
        "schema_version": 1,
        "authority": {
            "historical_replay_process_start_allowed": True,
            "research_filesystem_write_allowed": True,
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
        },
    }
    source_manifest = _sealed(
        {
            "contract": "QR_DOJO_LONG_HORIZON_SOURCE_MANIFEST_V1",
            "schema_version": 1,
            "sources": [],
            "authority": safe_authority,
        },
        "source_manifest_sha256",
    )
    registry = _sealed(
        {
            "contract": "QR_DOJO_G2_STRATEGY_REGISTRY_V1",
            "schema_version": 1,
            "workers": [],
            "authority": safe_authority,
        },
        "artifact_sha256",
    )
    implementation = _sealed(
        {
            "contract": IMPLEMENTATION_MANIFEST_CONTRACT,
            "schema_version": 1,
            "implementation_digests": implementation_digests,
            "implementation_digests_sha256": _canonical_sha256(implementation_digests),
            "automatic_deployment_allowed": False,
            "broker_mutation_allowed": False,
            "live_permission": False,
            "order_authority": "NONE",
            "promotion_eligible": False,
        },
        "implementation_manifest_sha256",
    )
    sealed_inputs = {
        "IMPLEMENTATION_MANIFEST": (
            "sealed-inputs/implementation-manifest.json",
            implementation,
        ),
        "RUN_CONTROL": ("sealed-inputs/run-control.json", run_control),
        "SOURCE_MANIFEST": (
            "sealed-inputs/source-manifest.json",
            source_manifest,
        ),
        "STRATEGY_REGISTRY": (
            "sealed-inputs/strategy-registry.json",
            registry,
        ),
    }
    sealed_rows = []
    for artifact_id in sorted(sealed_inputs):
        relative, value = sealed_inputs[artifact_id]
        path = root / relative
        _write_json(path, value)
        raw = path.read_bytes()
        sealed_rows.append(
            {
                "artifact_id": artifact_id,
                "relative_path": relative,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "file_size_bytes": len(raw),
            }
        )
    control_body = {
        "contract": "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1",
        "schema_version": 1,
        "run_control_sha256": "3" * 64,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "registry_artifact_sha256": registry["artifact_sha256"],
        "sealed_input_artifacts": sealed_rows,
        "sealed_input_artifacts_sha256": _canonical_sha256(sealed_rows),
        "artifact_sha256": {
            "plan": hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
            "proposals": hashlib.sha256(
                (root / "candidate-proposals.json").read_bytes()
            ).hexdigest(),
            "resource_policy": hashlib.sha256(
                (root / "resource-policy.json").read_bytes()
            ).hexdigest(),
            "runtime_seal": hashlib.sha256(
                (root / "tuned-runtime-seal.json").read_bytes()
            ).hexdigest(),
            "schedule": hashlib.sha256(
                (root / "schedule.json").read_bytes()
            ).hexdigest(),
            "worker_catalog": hashlib.sha256(
                (root / "worker-catalog.json").read_bytes()
            ).hexdigest(),
        },
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    _write_json(
        root / "control-manifest.json",
        _sealed(control_body, "manifest_sha256"),
    )

    result_body = {
        "contract": "QR_DOJO_LONG_HORIZON_ECONOMIC_JOB_RESULT_V1",
        "schema_version": 1,
        "job_sha256": JOB_SHA256,
        "job_status": "COMPLETE",
    }
    result = _sealed(result_body, "economic_job_result_sha256")
    _write_json(root / "job-results" / f"{JOB_SHA256}.json", result)

    source_relative = f"M5/2025-01/OHLC-{JOB_SHA256}.jsonl"
    source = root / "source-slices" / source_relative
    source.parent.mkdir(parents=True, exist_ok=True)
    if incompressible:
        source.write_bytes(random.Random(7).randbytes(payload_bytes))
    else:
        source.write_bytes(
            (b'{"price":1}\n' * ((payload_bytes // 12) + 1))[:payload_bytes]
        )
    source_raw = source.read_bytes()
    source_body = {
        "contract": "QR_DOJO_SPARSE_MONTH_SOURCE_SLICE_V2",
        "schema_version": 2,
        "job_sha256": JOB_SHA256,
        "relative_path": source_relative,
        "file_sha256": hashlib.sha256(source_raw).hexdigest(),
        "file_size_bytes": len(source_raw),
    }
    job_root = root / "jobs" / JOB_SHA256
    _write_json(
        job_root / "source-slice-receipt.json",
        _sealed(source_body, "source_slice_receipt_sha256"),
    )
    _write_json(job_root / "runner-handoff.json", {"job_sha256": JOB_SHA256})
    evidence = job_root / "economic-evidence" / "transcript.jsonl"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_bytes(b'{"event":"fill"}\n' * 1024)

    for section in ("claims", "reducers", "terminals", "cells"):
        _write_json(
            root / "execution-state" / section / JOB_SHA256 / "attempt.json",
            {"section": section, "job_sha256": JOB_SHA256},
        )

    completion_body = {
        "contract": JOB_COMPLETION_CONTRACT,
        "schema_version": 1,
        "job_sha256": JOB_SHA256,
        "job_status": "COMPLETE",
        "economic_job_result_sha256": result["economic_job_result_sha256"],
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    _write_json(
        job_root / "completion.json",
        _sealed(completion_body, "completion_sha256"),
    )
    return root


def _failed_source_run(tmp_path: Path) -> Path:
    root = _terminal_run(tmp_path)
    job_root = root / "jobs" / JOB_SHA256
    (root / "job-results" / f"{JOB_SHA256}.json").unlink()
    (job_root / "source-slice-receipt.json").unlink()
    transcript = job_root / "economic-evidence" / "transcript.jsonl"
    transcript.unlink()
    transcript.parent.rmdir()
    claim_sha256 = "4" * 64
    failure_body = {
        "contract": "QR_DOJO_HISTORICAL_SOURCE_FAILURE_V1",
        "schema_version": 1,
        "job_sha256": JOB_SHA256,
        "claim_sha256": claim_sha256,
        "stage": "SPARSE_SOURCE_MATERIALIZATION",
        "error_type": "FileNotFoundError",
        "error": "source is unavailable",
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    _write_json(
        job_root / "source-failure.json",
        _sealed(failure_body, "failure_evidence_sha256"),
    )
    completion_body = {
        "contract": JOB_COMPLETION_CONTRACT,
        "schema_version": 1,
        "job_sha256": JOB_SHA256,
        "claim_sha256": claim_sha256,
        "coordinate_result_count": 4,
        "new_source_failure_coordinate_count": 4,
        "predecessor_failure_coordinate_count": 0,
        "complete_coordinate_count": 0,
        "failed_coordinate_count": 4,
        "job_status": "FAILED_SOURCE",
        "economic_job_result_sha256": None,
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    _write_json(
        job_root / "completion.json",
        _sealed(completion_body, "completion_sha256"),
    )
    return root


def _rewrite_receipt(path: Path, mutate) -> dict:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutate(receipt)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = _canonical_sha256(body)
    _write_json(path, receipt)
    return receipt


def test_archive_is_deterministic_streamed_and_idempotent(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    first_root = tmp_path / "drive-a"
    second_root = tmp_path / "drive-b"

    inspection = inspect_historical_job_archive_source(
        run_root=root,
        job_sha256=JOB_SHA256,
    )
    assert inspection["contract"] == ARCHIVE_SOURCE_INSPECTION_CONTRACT
    assert inspection["inspection_sha256"] == _canonical_sha256(
        {key: value for key, value in inspection.items() if key != "inspection_sha256"}
    )
    assert not first_root.exists()

    first = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=first_root,
    )
    repeated = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=first_root,
    )
    second = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=second_root,
    )

    assert repeated == first
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert second["archive_sha256"] == first["archive_sha256"]
    assert second["archive_size_bytes"] == first["archive_size_bytes"]
    assert first["source_deletion_allowed"] is False
    assert first["remote_verification"]["remote_verified"] is False
    assert first["bundle_kind"] == SUCCESS_BUNDLE_KIND
    assert inspection["bundle_kind"] == first["bundle_kind"]
    assert inspection["file_count"] == first["file_count"]
    assert inspection["total_source_bytes"] == first["total_source_bytes"]
    assert inspection["manifest_sha256"] == first["manifest_sha256"]
    assert (
        verify_existing_historical_job_archive(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=first_root,
        )
        == first
    )
    verified = _verify_archive(
        Path(first["archive_path"]),
        zstd_bin="zstd",
        expected_job_sha256=JOB_SHA256,
        expected_completion_sha256=first["completion_sha256"],
        expected_bundle_kind=SUCCESS_BUNDLE_KIND,
    )
    assert {
        "sealed-inputs/implementation-manifest.json",
        "sealed-inputs/run-control.json",
        "sealed-inputs/source-manifest.json",
        "sealed-inputs/strategy-registry.json",
    }.issubset({row["path"] for row in verified["files"]})


def test_recovery_capacity_includes_tar_pax_and_zstd_overhead(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path, payload_bytes=64 * 1024)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    staging_root.mkdir()

    capacity = inspect_historical_job_archive_recovery_capacity(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert capacity["contract"] == ARCHIVE_RECOVERY_CAPACITY_CONTRACT
    assert capacity["manifest_payload_bytes"] > 0
    assert capacity["tar_pax_upper_bound_bytes"] > capacity["total_source_bytes"]
    assert capacity["zstd_framing_upper_bound_bytes"] >= 128 * 1024
    assert capacity["archive_upper_bound_bytes"] == (
        capacity["tar_pax_upper_bound_bytes"]
        + capacity["zstd_framing_upper_bound_bytes"]
    )
    assert (
        capacity["remaining_local_staging_bytes"]
        == capacity["archive_upper_bound_bytes"]
    )
    assert (
        capacity["remaining_archive_final_bytes"]
        == capacity["archive_upper_bound_bytes"]
    )
    assert capacity["compression_ratio_assumed"] is False


def test_recovery_capacity_subtracts_deep_verified_crash_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path, payload_bytes=64 * 1024)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_rename = archive_module._atomic_rename_no_replace

    class SimulatedCrash(BaseException):
        pass

    def crash_before(source: Path, destination: Path) -> bool:
        if destination.name.endswith(".tar.zst"):
            raise SimulatedCrash
        return original_rename(source, destination)

    monkeypatch.setattr(archive_module, "_atomic_rename_no_replace", crash_before)
    with pytest.raises(SimulatedCrash):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )
    monkeypatch.setattr(archive_module, "_atomic_rename_no_replace", original_rename)

    capacity = inspect_historical_job_archive_recovery_capacity(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert capacity["validated_local_archive_pending_bytes"] > 0
    assert (
        capacity["validated_archive_pending_bytes"]
        == capacity["validated_local_archive_pending_bytes"]
    )
    assert capacity["remaining_local_staging_bytes"] == 0
    assert capacity["remaining_archive_final_bytes"] == 0
    assert capacity["remaining_archive_filesystem_bytes"] < (
        capacity["archive_upper_bound_bytes"] * 2
        + min(
            archive_module.ARCHIVE_PART_BYTES,
            capacity["archive_upper_bound_bytes"],
        )
    )


def test_archive_stages_locally_then_deep_verifies_drive_temp_and_final_without_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    local_staging_root = tmp_path / "local-staging"
    local_staging_root.mkdir()
    verified_paths: list[Path] = []
    original_verify = archive_module._verify_archive

    def record_verify(path: Path, **kwargs):
        verified_paths.append(Path(path))
        return original_verify(path, **kwargs)

    def reject_hardlink(*_args, **_kwargs):
        raise AssertionError("File Provider publication must not use hardlinks")

    monkeypatch.setattr(archive_module, "_verify_archive", record_verify)
    monkeypatch.setattr(archive_module.os, "link", reject_hardlink)

    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=local_staging_root,
    )

    assert len(verified_paths) == 2
    assert local_staging_root in verified_paths[0].parents
    assert verified_paths[1].parent == archive_root / "archives"
    assert verified_paths[1].name.endswith(".tar.zst.pending")
    assert not list(local_staging_root.glob(".*.local-pending"))
    retired = list(local_staging_root.glob(".retired-*.anchor"))
    assert len(retired) == 1
    assert retired[0].stat().st_size == 0
    assert not list((archive_root / "archives").glob("*.part"))
    assert not list((archive_root / "archives").glob(".*.pending"))
    assert receipt["remote_readback_objects"]["object_count"] == 1
    assert Path(
        receipt["remote_readback_objects"]["objects"][0]["relative_path"]
    ).parent == Path("readback-objects")


def test_drive_temporary_corruption_fails_before_final_or_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    original_copy = archive_module._prepare_checked_copy_pending

    def corrupt_temporary(**kwargs) -> Path:
        path = original_copy(**kwargs)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        path.write_bytes(raw)
        return path

    monkeypatch.setattr(
        archive_module,
        "_prepare_checked_copy_pending",
        corrupt_temporary,
    )

    with pytest.raises(DojoHistoricalJobArchiveError):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=tmp_path / "local-staging",
        )

    assert not list((archive_root / "archives").glob("*.tar.zst"))
    assert not list((archive_root / "archives").glob("*.part"))
    assert not list((archive_root / "receipts").glob("*.json"))


def test_existing_final_is_never_overwritten(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    inspection = inspect_historical_job_archive_source(
        run_root=root,
        job_sha256=JOB_SHA256,
    )
    archive_root = tmp_path / "drive"
    archives = archive_root / "archives"
    archives.mkdir(parents=True)
    final = archives / (f"job-{JOB_SHA256}-{inspection['manifest_sha256']}.tar.zst")
    hostile = b"pre-existing-unverified-file"
    final.write_bytes(hostile)

    with pytest.raises(DojoHistoricalJobArchiveError):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )

    assert final.read_bytes() == hostile
    assert not list((archive_root / "receipts").glob("*.json"))


def test_final_appearing_after_drive_temp_is_not_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    original_copy = archive_module._prepare_checked_copy_pending
    hostile = b"concurrent-final"
    published: dict[str, Path] = {}

    def race_final(**kwargs) -> Path:
        temporary = original_copy(**kwargs)
        pending = kwargs["pending_path"]
        final = pending.with_name(pending.name[1 : -len(".pending")])
        final.write_bytes(hostile)
        published["final"] = final
        return temporary

    monkeypatch.setattr(
        archive_module,
        "_prepare_checked_copy_pending",
        race_final,
    )

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="existing archive drifted",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )

    assert published["final"].read_bytes() == hostile
    assert not list((archive_root / "receipts").glob("*.json"))


def test_verify_archive_passes_open_fd_to_zstd_and_keeps_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = tmp_path / "invalid.tar.zst"
    invalid.write_bytes(b"this is not a zstd stream")
    real_popen = subprocess.Popen
    observed: dict[str, object] = {}

    def record_popen(command, *args, **kwargs):
        observed["command"] = command
        observed["stdin"] = kwargs.get("stdin")
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(archive_module.subprocess, "Popen", record_popen)

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match=r"zstd return code -?\d+; zstd stderr:",
    ):
        _verify_archive(invalid, zstd_bin="zstd")

    assert all("/dev/fd" not in str(argument) for argument in observed["command"])
    assert isinstance(observed["stdin"], int)


def test_crash_before_archive_publish_reuses_verified_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_rename = archive_module._atomic_rename_no_replace

    class SimulatedCrash(BaseException):
        pass

    def crash_before(source: Path, destination: Path) -> bool:
        if destination.name.endswith(".tar.zst"):
            raise SimulatedCrash
        return original_rename(source, destination)

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        crash_before,
    )
    with pytest.raises(SimulatedCrash):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    drive_pending = next((archive_root / "archives").glob(".*.pending"))
    local_pending = next(staging_root.glob(".*.local-pending"))
    assert not list((archive_root / "archives").glob("*.tar.zst"))
    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        original_rename,
    )

    def reject_rebuild(**_kwargs) -> None:
        raise AssertionError("verified local pending must be reused")

    monkeypatch.setattr(archive_module, "_write_archive", reject_rebuild)
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert Path(receipt["archive_path"]).is_file()
    assert not drive_pending.exists()
    assert not local_pending.exists()


def test_crash_after_archive_publish_recovers_complete_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_rename = archive_module._atomic_rename_no_replace

    class SimulatedCrash(BaseException):
        pass

    def crash_after(source: Path, destination: Path) -> bool:
        published = original_rename(source, destination)
        if destination.name.endswith(".tar.zst"):
            raise SimulatedCrash
        return published

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        crash_after,
    )
    with pytest.raises(SimulatedCrash):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    final = next((archive_root / "archives").glob("*.tar.zst"))
    assert not list((archive_root / "archives").glob(".*.pending"))
    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        original_rename,
    )

    def reject_rebuild(**_kwargs) -> None:
        raise AssertionError("complete final must be recovered without rebuilding")

    monkeypatch.setattr(archive_module, "_write_archive", reject_rebuild)
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert Path(receipt["archive_path"]) == final


def test_transient_drive_pending_deep_read_failure_keeps_verified_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_verify = archive_module._verify_archive
    failed = False

    def fail_one_drive_pending(path: Path, *args, **kwargs) -> None:
        nonlocal failed
        if path.parent == archive_root / "archives" and path.name.endswith(".pending"):
            if not failed:
                failed = True
                raise DojoHistoricalJobArchiveError(
                    "transient Drive pending materialization failure"
                )
        original_verify(path, *args, **kwargs)

    monkeypatch.setattr(archive_module, "_verify_archive", fail_one_drive_pending)
    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="transient Drive pending materialization failure",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    drive_pending = next((archive_root / "archives").glob(".*.pending"))
    pending_sha256, pending_size = archive_module._hash_file(drive_pending)
    local_pending = next(staging_root.glob(".*.local-pending"))
    assert archive_module._hash_file(local_pending) == (pending_sha256, pending_size)

    monkeypatch.setattr(archive_module, "_verify_archive", original_verify)

    def reject_recopy(**_kwargs) -> None:
        raise AssertionError("verified Drive pending must be reused")

    monkeypatch.setattr(
        archive_module,
        "_copy_regular_file_to_output",
        reject_recopy,
    )
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert Path(receipt["archive_path"]).is_file()
    assert not drive_pending.exists()


def test_corrupt_drive_pending_is_removed_after_deep_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_verify = archive_module._verify_archive

    def corrupt_drive_pending(path: Path, *args, **kwargs) -> None:
        if path.parent == archive_root / "archives" and path.name.endswith(".pending"):
            path.write_bytes(b"confirmed corrupt Drive pending")
            raise DojoHistoricalJobArchiveError("corrupt Drive pending")
        original_verify(path, *args, **kwargs)

    monkeypatch.setattr(archive_module, "_verify_archive", corrupt_drive_pending)
    with pytest.raises(DojoHistoricalJobArchiveError, match="corrupt Drive pending"):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    assert not list((archive_root / "archives").glob(".*.pending"))
    assert len(list(staging_root.glob(".*.local-pending"))) == 1


def test_unsupported_atomic_rename_fails_closed_with_reusable_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_rename = archive_module._atomic_rename_no_replace

    def unsupported(_source: Path, _destination: Path) -> bool:
        raise DojoHistoricalJobArchiveError("atomic no-replace rename is unsupported")

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        unsupported,
    )
    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="atomic no-replace rename is unsupported",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    assert not list((archive_root / "archives").glob("*.tar.zst"))
    assert not list((archive_root / "receipts").glob("*.json"))
    drive_pending = next((archive_root / "archives").glob(".*.pending"))
    assert len(list(staging_root.glob(".*.local-pending"))) == 1
    drive_pending.write_bytes(b"invalid pending")
    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        original_rename,
    )

    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert Path(receipt["archive_path"]).is_file()
    assert not drive_pending.exists()


def test_atomic_publish_has_no_unsupported_platform_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = tmp_path / ".artifact.pending"
    final = tmp_path / "artifact.bin"
    pending.write_bytes(b"verified")
    monkeypatch.setattr(archive_module.sys, "platform", "unsupported-os")

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="atomic no-replace rename is unsupported",
    ):
        archive_module._atomic_rename_no_replace(pending, final)

    assert pending.read_bytes() == b"verified"
    assert not final.exists()


def test_pending_cleanup_never_deletes_last_moment_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = tmp_path / ".archive.pending"
    replacement = tmp_path / "hostile-replacement"
    pending.write_bytes(b"expected pending")
    replacement.write_bytes(b"replacement must survive")
    expected = pending.stat(follow_symlinks=False)
    original_rename = archive_module._atomic_rename_at_no_replace
    replaced = False

    def replace_at_last_cleanup_transition(
        directory_fd: int,
        source_name: str,
        destination_name: str,
    ) -> bool:
        nonlocal replaced
        if source_name == pending.name and not replaced:
            replaced = True
            os.replace(replacement, pending)
        return original_rename(directory_fd, source_name, destination_name)

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_at_no_replace",
        replace_at_last_cleanup_transition,
    )

    def reject_unlink(*_args, **_kwargs) -> None:
        raise AssertionError("pending cleanup must never call unlink")

    monkeypatch.setattr(archive_module.os, "unlink", reject_unlink)

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="incomplete archive output was replaced concurrently",
    ):
        archive_module._unlink_if_same_inode(
            pending,
            device=expected.st_dev,
            inode=expected.st_ino,
        )

    assert replaced is True
    assert not pending.exists()
    anchors = list(tmp_path.glob(".retired-*.anchor"))
    assert len(anchors) == 1
    assert anchors[0].read_bytes() == b"replacement must survive"


def test_pending_cleanup_releases_payload_without_copying_anchor(
    tmp_path: Path,
) -> None:
    pending = tmp_path / ".archive.pending"
    pending.write_bytes(b"x" * (2 * 1024 * 1024))
    expected = pending.stat(follow_symlinks=False)

    archive_module._unlink_if_same_inode(
        pending,
        device=expected.st_dev,
        inode=expected.st_ino,
    )

    assert not pending.exists()
    anchors = list(tmp_path.glob(".retired-*.anchor"))
    assert len(anchors) == 1
    assert anchors[0].stat().st_size == 0


def test_receipt_pending_is_atomic_and_reused_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_rename = archive_module._atomic_rename_no_replace

    class SimulatedCrash(BaseException):
        pass

    def crash_before_receipt(source: Path, destination: Path) -> bool:
        if destination.name.endswith(".json"):
            raise SimulatedCrash
        return original_rename(source, destination)

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        crash_before_receipt,
    )
    with pytest.raises(SimulatedCrash):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    receipt_pending = next((archive_root / "receipts").glob(".*.pending"))
    assert not list((archive_root / "receipts").glob("*.json"))
    assert list((archive_root / "archives").glob("*.tar.zst"))
    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        original_rename,
    )

    def reject_rebuild(**_kwargs) -> None:
        raise AssertionError("receipt retry must not rebuild the archive")

    monkeypatch.setattr(archive_module, "_write_archive", reject_rebuild)
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert Path(receipt["archive_path"]).is_file()
    assert not receipt_pending.exists()
    assert len(list((archive_root / "receipts").glob("*.json"))) == 1


def test_crash_after_receipt_publish_cannot_leave_local_archive_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_rename = archive_module._atomic_rename_no_replace

    class SimulatedCrash(BaseException):
        pass

    def crash_after_receipt(source: Path, destination: Path) -> bool:
        published = original_rename(source, destination)
        if (
            destination.parent == archive_root / "receipts"
            and destination.suffix == ".json"
        ):
            raise SimulatedCrash
        return published

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        crash_after_receipt,
    )
    with pytest.raises(SimulatedCrash):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    assert len(list((archive_root / "receipts").glob("*.json"))) == 1
    assert not list(staging_root.glob(".*.local-pending"))

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        original_rename,
    )

    def reject_rebuild(**_kwargs) -> None:
        raise AssertionError("settled receipt retry must not rebuild the archive")

    monkeypatch.setattr(archive_module, "_write_archive", reject_rebuild)
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert Path(receipt["archive_path"]).is_file()
    assert not list(staging_root.glob(".*.local-pending"))


def test_zero_byte_receipt_pending_is_safely_replaced(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    pending = tmp_path / ".receipt.json.pending"
    pending.write_bytes(b"")

    archive_module._write_once(receipt, {"contract": "TEST", "schema_version": 1})

    assert json.loads(receipt.read_text(encoding="utf-8")) == {
        "contract": "TEST",
        "schema_version": 1,
    }
    assert not pending.exists()


def test_readback_part_pending_is_reused_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_rename = archive_module._atomic_rename_no_replace

    class SimulatedCrash(BaseException):
        pass

    def crash_before_part(source: Path, destination: Path) -> bool:
        if destination.suffix == ".bin":
            raise SimulatedCrash
        return original_rename(source, destination)

    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        crash_before_part,
    )
    with pytest.raises(SimulatedCrash):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
            local_staging_root=staging_root,
        )

    part_pending = next((archive_root / "readback-objects").glob(".*.pending"))
    assert not list((archive_root / "readback-objects").glob("*.bin"))
    monkeypatch.setattr(
        archive_module,
        "_atomic_rename_no_replace",
        original_rename,
    )
    original_prepare = archive_module._prepare_slice_pending
    observed_existing: list[bool] = []

    def record_reuse(**kwargs) -> Path:
        observed_existing.append(kwargs["pending_path"].exists())
        return original_prepare(**kwargs)

    monkeypatch.setattr(archive_module, "_prepare_slice_pending", record_reuse)

    def reject_rebuild(**_kwargs) -> None:
        raise AssertionError("part retry must not rebuild the archive")

    monkeypatch.setattr(archive_module, "_write_archive", reject_rebuild)
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
        local_staging_root=staging_root,
    )

    assert observed_existing == [True]
    assert Path(receipt["archive_path"]).is_file()
    assert not part_pending.exists()


def test_local_staging_rejects_cloudstorage_path_before_creation(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    cloud_staging = tmp_path / "Library" / "CloudStorage" / "staging"

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="must not use CloudStorage or File Provider",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=tmp_path / "drive",
            local_staging_root=cloud_staging,
        )

    assert not cloud_staging.exists()


def test_failed_source_is_archived_as_an_explicit_non_economic_bundle(
    tmp_path: Path,
) -> None:
    root = _failed_source_run(tmp_path)
    archive_root = tmp_path / "drive"
    inspection = inspect_historical_job_archive_source(
        run_root=root,
        job_sha256=JOB_SHA256,
    )

    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )

    assert receipt["bundle_kind"] == FAILED_SOURCE_BUNDLE_KIND
    assert inspection["bundle_kind"] == FAILED_SOURCE_BUNDLE_KIND
    assert inspection["manifest_sha256"] == receipt["manifest_sha256"]
    assert (
        verify_existing_historical_job_archive(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )
        == receipt
    )
    manifest = _verify_archive(
        Path(receipt["archive_path"]),
        zstd_bin="zstd",
        expected_job_sha256=JOB_SHA256,
        expected_completion_sha256=receipt["completion_sha256"],
        expected_bundle_kind=FAILED_SOURCE_BUNDLE_KIND,
    )
    paths = {row["path"] for row in manifest["files"]}
    assert f"jobs/{JOB_SHA256}/source-failure.json" in paths
    assert f"job-results/{JOB_SHA256}.json" not in paths
    assert f"jobs/{JOB_SHA256}/source-slice-receipt.json" not in paths


def test_generation_transition_receipt_is_bound_into_job_archive(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    transition = root / "transition-receipts" / "supersede-test.json"
    _write_json(transition, {"contract": "TEST_TRANSITION", "schema_version": 1})

    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=tmp_path / "drive",
    )
    manifest = _verify_archive(
        Path(receipt["archive_path"]),
        zstd_bin="zstd",
        expected_job_sha256=JOB_SHA256,
        expected_completion_sha256=receipt["completion_sha256"],
        expected_bundle_kind=SUCCESS_BUNDLE_KIND,
    )

    assert "transition-receipts/supersede-test.json" in {
        row["path"] for row in manifest["files"]
    }


def test_incomplete_economic_result_remains_an_archivable_economic_bundle(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    result_path = root / "job-results" / f"{JOB_SHA256}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["job_status"] = "INCOMPLETE_FAILED"
    result_body = {
        key: value
        for key, value in result.items()
        if key != "economic_job_result_sha256"
    }
    result["economic_job_result_sha256"] = _canonical_sha256(result_body)
    _write_json(result_path, result)
    completion_path = root / "jobs" / JOB_SHA256 / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["job_status"] = "INCOMPLETE_FAILED"
    completion["economic_job_result_sha256"] = result["economic_job_result_sha256"]
    completion_body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    completion["completion_sha256"] = _canonical_sha256(completion_body)
    _write_json(completion_path, completion)

    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=tmp_path / "drive",
    )

    assert receipt["bundle_kind"] == SUCCESS_BUNDLE_KIND


@pytest.mark.parametrize("mixed_artifact", ["result", "source_receipt"])
def test_failed_source_rejects_mixed_success_schema(
    tmp_path: Path,
    mixed_artifact: str,
) -> None:
    root = _failed_source_run(tmp_path)
    if mixed_artifact == "result":
        _write_json(
            root / "job-results" / f"{JOB_SHA256}.json",
            {"job_sha256": JOB_SHA256},
        )
    else:
        _write_json(
            root / "jobs" / JOB_SHA256 / "source-slice-receipt.json",
            {"job_sha256": JOB_SHA256},
        )

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="mixes economic success artifacts",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=tmp_path / "drive",
        )


def test_success_rejects_source_failure_schema(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    _write_json(
        root / "jobs" / JOB_SHA256 / "source-failure.json",
        {"job_sha256": JOB_SHA256},
    )

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="contains source-failure evidence",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=tmp_path / "drive",
        )


def test_read_only_verifier_does_not_create_missing_archive_root(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    missing = tmp_path / "missing-drive"

    with pytest.raises(
        DojoHistoricalJobArchiveError, match="archive root is unavailable"
    ):
        verify_existing_historical_job_archive(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=missing,
        )

    assert not missing.exists()


def test_source_inspection_rejects_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    original = archive_module._inventory
    call_count = 0

    def drifting_inventory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        files, total = original(*args, **kwargs)
        if call_count == 2:
            return files, total + 1
        return files, total

    monkeypatch.setattr(archive_module, "_inventory", drifting_inventory)

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="inventory changed while inspected",
    ):
        inspect_historical_job_archive_source(
            run_root=root,
            job_sha256=JOB_SHA256,
        )


def test_source_inspection_rejects_source_receipt_traversal(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    receipt_path = root / "jobs" / JOB_SHA256 / "source-slice-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["relative_path"] = "../outside.jsonl"
    receipt_body = {
        key: value
        for key, value in receipt.items()
        if key != "source_slice_receipt_sha256"
    }
    receipt["source_slice_receipt_sha256"] = _canonical_sha256(receipt_body)
    _write_json(receipt_path, receipt)

    with pytest.raises(DojoHistoricalJobArchiveError, match="member path is unsafe"):
        inspect_historical_job_archive_source(
            run_root=root,
            job_sha256=JOB_SHA256,
        )


def test_sealed_common_input_tamper_fails_closed(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    path = root / "sealed-inputs" / "strategy-registry.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["workers"] = [{"worker_id": "forged"}]
    _write_json(path, value)

    with pytest.raises(
        DojoHistoricalJobArchiveError, match="sealed input bytes drifted"
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=tmp_path / "drive",
        )


def test_inventory_rejects_symlinked_parent_outside_run_root(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    sealed = root / "sealed-inputs"
    outside = tmp_path / "outside-sealed-inputs"
    sealed.rename(outside)
    sealed.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="source path contains a symlink",
    ):
        inspect_historical_job_archive_source(
            run_root=root,
            job_sha256=JOB_SHA256,
        )
    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="source path contains a symlink",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=tmp_path / "drive",
        )


def test_concurrent_calls_publish_one_archive_and_one_receipt(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"

    def archive() -> dict | DojoHistoricalJobArchiveError:
        try:
            return archive_completed_historical_job(
                run_root=root,
                job_sha256=JOB_SHA256,
                archive_root=archive_root,
            )
        except DojoHistoricalJobArchiveError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: archive(), range(2)))

    receipts = [row for row in outcomes if isinstance(row, dict)]
    backpressure = [
        row for row in outcomes if isinstance(row, DojoHistoricalJobArchiveError)
    ]
    assert receipts
    assert all("archive lock is already held" in str(row) for row in backpressure)
    retry = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    assert all(row == retry for row in receipts)
    assert len(list((archive_root / "archives").glob("*.tar.zst"))) == 1
    assert len(list((archive_root / "receipts").glob("*.json"))) == 1
    assert not list((archive_root / "archives").glob("*.part"))


def test_job_archive_lock_contention_is_immediate_backpressure(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()

    with archive_module._job_archive_lock(receipts, JOB_SHA256) as guard:
        started = time.monotonic()
        with pytest.raises(
            DojoHistoricalJobArchiveError,
            match="archive lock is already held",
        ):
            with archive_module._job_archive_lock(receipts, JOB_SHA256):
                pytest.fail("a second writer must not enter the held job lock")
        elapsed = time.monotonic() - started
        guard()

    assert elapsed < 0.5
    assert not list(receipts.glob("*.pending"))
    assert not list(receipts.glob("*.json"))


def test_job_archive_lock_tolerates_metadata_only_timestamp_refresh(
    tmp_path: Path,
) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    lock_path = receipts / f".job-{JOB_SHA256}.lock"

    with archive_module._job_archive_lock(receipts, JOB_SHA256) as guard:
        before = lock_path.stat(follow_symlinks=False)
        os.utime(
            lock_path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            follow_symlinks=False,
        )
        after = lock_path.stat(follow_symlinks=False)
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        assert (after.st_mtime_ns, after.st_ctime_ns) != (
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        guard()


def test_job_archive_lock_rejects_hardlink_added_while_held(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    lock_path = receipts / f".job-{JOB_SHA256}.lock"
    alias_path = receipts / "lock-alias"

    with archive_module._job_archive_lock(receipts, JOB_SHA256) as guard:
        os.link(lock_path, alias_path)
        try:
            with pytest.raises(
                archive_module.DojoHistoricalJobArchiveError,
                match="archive lock pathname was replaced concurrently",
            ):
                guard()
        finally:
            alias_path.unlink()


def test_replaced_job_lock_stops_before_first_archive_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_inventory = archive_module._inventory
    second_descriptor: int | None = None
    second_entered = False

    def replace_lock_after_flock(*args, **kwargs):
        nonlocal second_descriptor, second_entered
        rows = original_inventory(*args, **kwargs)
        receipts = archive_root / "receipts"
        lock_path = receipts / f".job-{JOB_SHA256}.lock"
        displaced = receipts / ".displaced-first-writer.lock"
        lock_path.rename(displaced)
        second_descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        fcntl.flock(second_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second_entered = True
        return rows

    monkeypatch.setattr(archive_module, "_inventory", replace_lock_after_flock)

    def reject_archive_write(**_kwargs) -> None:
        raise AssertionError("replaced lock must fail before archive mutation")

    monkeypatch.setattr(archive_module, "_write_archive", reject_archive_write)
    try:
        with pytest.raises(
            DojoHistoricalJobArchiveError,
            match="archive lock pathname was replaced concurrently",
        ):
            archive_completed_historical_job(
                run_root=root,
                job_sha256=JOB_SHA256,
                archive_root=archive_root,
                local_staging_root=staging_root,
            )
    finally:
        if second_descriptor is not None:
            fcntl.flock(second_descriptor, fcntl.LOCK_UN)
            os.close(second_descriptor)

    assert second_entered is True
    assert not list((archive_root / "archives").iterdir())
    assert not list((archive_root / "receipts").glob("*.json"))
    assert not list((archive_root / "receipts").glob("*.pending"))
    assert not staging_root.exists()


def test_replaced_receipts_parent_stops_double_lock_before_archive_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    staging_root = tmp_path / "local-staging"
    original_inventory = archive_module._inventory
    second_descriptor: int | None = None
    second_entered = False

    def replace_receipts_parent_after_flock(*args, **kwargs):
        nonlocal second_descriptor, second_entered
        rows = original_inventory(*args, **kwargs)
        receipts = archive_root / "receipts"
        displaced = archive_root / "displaced-first-writer-receipts"
        receipts.rename(displaced)
        receipts.mkdir()
        second_descriptor = os.open(
            receipts / f".job-{JOB_SHA256}.lock",
            os.O_RDWR | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        fcntl.flock(second_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second_entered = True
        return rows

    monkeypatch.setattr(
        archive_module,
        "_inventory",
        replace_receipts_parent_after_flock,
    )

    def reject_archive_write(**_kwargs) -> None:
        raise AssertionError("replaced receipts parent must stop archive mutation")

    monkeypatch.setattr(archive_module, "_write_archive", reject_archive_write)
    try:
        with pytest.raises(
            DojoHistoricalJobArchiveError,
            match="archive lock directory was replaced concurrently",
        ):
            archive_completed_historical_job(
                run_root=root,
                job_sha256=JOB_SHA256,
                archive_root=archive_root,
                local_staging_root=staging_root,
            )
    finally:
        if second_descriptor is not None:
            fcntl.flock(second_descriptor, fcntl.LOCK_UN)
            os.close(second_descriptor)

    assert second_entered is True
    assert not list((archive_root / "archives").iterdir())
    assert not list((archive_root / "receipts").glob("*.json"))
    assert not list((archive_root / "receipts").glob("*.pending"))
    displaced = archive_root / "displaced-first-writer-receipts"
    assert not list(displaced.glob("*.json"))
    assert not list(displaced.glob("*.pending"))
    assert not staging_root.exists()


def test_large_archive_is_split_into_bounded_content_addressed_readback_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_job_archive.ARCHIVE_PART_BYTES",
        64 * 1024,
    )

    def reject_hardlink(*_args, **_kwargs):
        raise AssertionError("File Provider readback objects must not use hardlinks")

    monkeypatch.setattr(archive_module.os, "link", reject_hardlink)
    root = _terminal_run(
        tmp_path,
        payload_bytes=512 * 1024,
        incompressible=True,
    )
    archive_root = tmp_path / "drive"

    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    readback = receipt["remote_readback_objects"]

    assert readback["object_count"] > 1
    assert all(row["size_bytes"] <= 64 * 1024 for row in readback["objects"])
    assert all(
        Path(row["relative_path"]).name.endswith(f"-{row['sha256']}.bin")
        for row in readback["objects"]
    )
    assert all(
        Path(row["relative_path"]).parent == Path("readback-objects")
        for row in readback["objects"]
    )
    assert all(
        Path(row["relative_path"]).name.startswith(
            f"job-{JOB_SHA256}-{receipt['manifest_sha256']}-part-"
        )
        for row in readback["objects"]
    )
    assert (
        sum(row["size_bytes"] for row in readback["objects"])
        == receipt["archive_size_bytes"]
    )
    assert (
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )
        == receipt
    )

    first_part = archive_root / readback["objects"][0]["relative_path"]
    part_raw = bytearray(first_part.read_bytes())
    part_raw[0] ^= 0xFF
    first_part.write_bytes(part_raw)
    corrupted = first_part.read_bytes()
    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="remote readback object bytes drifted",
    ):
        verify_existing_historical_job_archive(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )
    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="remote readback object bytes drifted",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )
    assert first_part.read_bytes() == corrupted


def test_legacy_nested_readback_part_receipt_remains_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_job_archive.ARCHIVE_PART_BYTES",
        64 * 1024,
    )
    root = _terminal_run(
        tmp_path,
        payload_bytes=512 * 1024,
        incompressible=True,
    )
    archive_root = tmp_path / "drive"
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    archive_path = Path(receipt["archive_path"])
    stem = f"job-{JOB_SHA256}-{receipt['manifest_sha256']}"
    legacy_root = archive_root / "parts" / stem
    legacy_root.mkdir(parents=True)
    legacy = json.loads(json.dumps(receipt["remote_readback_objects"]))
    for row in legacy["objects"]:
        current = archive_root / row["relative_path"]
        legacy_path = legacy_root / (f"part-{row['index']:05d}-{row['sha256']}.bin")
        current.rename(legacy_path)
        row["relative_path"] = legacy_path.relative_to(archive_root).as_posix()
    legacy_body = {
        key: value for key, value in legacy.items() if key != "object_set_sha256"
    }
    legacy["object_set_sha256"] = _canonical_sha256(legacy_body)

    archive_module._verify_remote_readback_objects(
        legacy,
        destination=archive_root,
        archive_path=archive_path,
        archive_sha256=receipt["archive_sha256"],
        archive_size_bytes=receipt["archive_size_bytes"],
        stem=stem,
    )


def test_mixed_current_and_legacy_readback_layout_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_job_archive.ARCHIVE_PART_BYTES",
        64 * 1024,
    )
    root = _terminal_run(
        tmp_path,
        payload_bytes=512 * 1024,
        incompressible=True,
    )
    archive_root = tmp_path / "drive"
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    archive_path = Path(receipt["archive_path"])
    stem = f"job-{JOB_SHA256}-{receipt['manifest_sha256']}"
    mixed = json.loads(json.dumps(receipt["remote_readback_objects"]))
    first = mixed["objects"][0]
    current = archive_root / first["relative_path"]
    legacy_root = archive_root / "parts" / stem
    legacy_root.mkdir(parents=True)
    legacy = legacy_root / f"part-00000-{first['sha256']}.bin"
    current.rename(legacy)
    first["relative_path"] = legacy.relative_to(archive_root).as_posix()
    body = {key: value for key, value in mixed.items() if key != "object_set_sha256"}
    mixed["object_set_sha256"] = _canonical_sha256(body)

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="remote readback object layouts are mixed",
    ):
        archive_module._verify_remote_readback_objects(
            mixed,
            destination=archive_root,
            archive_path=archive_path,
            archive_sha256=receipt["archive_sha256"],
            archive_size_bytes=receipt["archive_size_bytes"],
            stem=stem,
        )


def test_existing_receipt_cannot_redirect_to_an_arbitrary_file(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    receipt_path = next((archive_root / "receipts").glob("*.json"))
    arbitrary = root / "plan.json"
    _rewrite_receipt(
        receipt_path,
        lambda receipt: receipt.__setitem__("archive_path", os.fspath(arbitrary)),
    )

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="existing archive receipt is invalid",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )


def test_remote_verified_claim_cannot_be_written_into_local_receipt(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    receipt_path = next((archive_root / "receipts").glob("*.json"))

    def forge(receipt: dict) -> None:
        receipt["remote_verification"] = {
            "status": "REMOTE_VERIFIED",
            "remote_verified": True,
            "metadata_receipt_sha256": "b" * 64,
        }

    _rewrite_receipt(receipt_path, forge)

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="existing archive receipt is invalid",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )


def test_resealed_receipt_does_not_hide_corrupt_archive(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    archive_path = Path(receipt["archive_path"])
    raw = bytearray(archive_path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    archive_path.write_bytes(raw)
    receipt_path = next((archive_root / "receipts").glob("*.json"))

    def reseal(value: dict) -> None:
        value["archive_sha256"] = hashlib.sha256(raw).hexdigest()
        value["archive_size_bytes"] = len(raw)

    _rewrite_receipt(receipt_path, reseal)

    with pytest.raises(DojoHistoricalJobArchiveError):
        verify_existing_historical_job_archive(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )
    with pytest.raises(DojoHistoricalJobArchiveError):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )


def test_read_only_verifier_requires_exactly_one_receipt(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    receipt = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    _write_json(
        archive_root / "receipts" / f"job-{JOB_SHA256}-{'f' * 64}.json",
        receipt,
    )

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="exactly one archive receipt",
    ):
        verify_existing_historical_job_archive(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )


def test_verifier_rejects_tar_traversal_member_without_extracting(
    tmp_path: Path,
) -> None:
    payload = b"x"
    files = [
        {
            "path": "safe.txt",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    manifest_body = {
        "contract": ARCHIVE_MANIFEST_CONTRACT,
        "schema_version": 1,
        "job_sha256": JOB_SHA256,
        "completion_sha256": "c" * 64,
        "bundle_kind": SUCCESS_BUNDLE_KIND,
        "file_count": 1,
        "total_source_bytes": 1,
        "files": files,
        "historical_train_is_proof": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    manifest = _sealed(manifest_body, "manifest_sha256")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        manifest_payload = _canonical_bytes(manifest)
        archive.addfile(
            _tar_info("MANIFEST.json", len(manifest_payload)),
            io.BytesIO(manifest_payload),
        )
        archive.addfile(_tar_info("../escape", 1), io.BytesIO(payload))
    compressed = subprocess.run(
        ["zstd", "-q", "-3", "-T1", "-c"],
        input=stream.getvalue(),
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    archive_path = tmp_path / "traversal.tar.zst"
    archive_path.write_bytes(compressed)

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="payload member header mismatch",
    ):
        _verify_archive(
            archive_path,
            manifest=manifest,
            zstd_bin="zstd",
            expected_job_sha256=JOB_SHA256,
            expected_completion_sha256="c" * 64,
            expected_bundle_kind=SUCCESS_BUNDLE_KIND,
        )
    assert not (tmp_path / "escape").exists()


def test_source_change_during_stream_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    with _VerifiedSource(
        path,
        expected_size=path.stat().st_size,
        expected_sha256=expected,
    ) as source:
        source.read(1024 * 1024)
        path.write_bytes(b"b" * (2 * 1024 * 1024))
        source.read()
        with pytest.raises(
            DojoHistoricalJobArchiveError,
            match="changed while streaming",
        ):
            source.verify()


def test_missing_economic_result_fails_closed(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    (root / "job-results" / f"{JOB_SHA256}.json").unlink()

    with pytest.raises(
        DojoHistoricalJobArchiveError,
        match="economic job result is unavailable",
    ):
        archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=tmp_path / "drive",
        )
