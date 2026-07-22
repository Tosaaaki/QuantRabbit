from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

import quant_rabbit.dojo_historical_raw_reclaim as reclaim_module
from quant_rabbit.dojo_historical_job_archive import (
    IMPLEMENTATION_MANIFEST_CONTRACT,
    JOB_COMPLETION_CONTRACT,
    _canonical_bytes,
    _canonical_sha256,
    archive_completed_historical_job,
)
from quant_rabbit.dojo_historical_raw_reclaim import (
    REMOTE_READBACK_RECEIPT_CONTRACT,
    DojoHistoricalRawReclaimError,
    reclaim_historical_job_raw,
    verify_existing_historical_job_raw_reclaim,
    verify_historical_job_raw_reclaim,
)


JOB_SHA256 = "a" * 64
DRIVE_PARENT_ID = "driveParent12345"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _sealed(body: dict, field: str) -> dict:
    return {**body, field: _canonical_sha256(body)}


def _terminal_run(
    tmp_path: Path,
    *,
    source_size: int = 256 * 1024,
    incompressible: bool = False,
) -> tuple[Path, list[Path]]:
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
            **safe_authority,
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
        **safe_authority,
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
        source.write_bytes(random.Random(19).randbytes(source_size))
    else:
        source.write_bytes((b'{"price":1}\n' * (source_size // 12 + 1))[:source_size])
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
    evidence = job_root / "economic-evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    transcripts = []
    for index in range(2):
        transcript = evidence / f"coordinate-{index}.economic.jsonl"
        transcript.write_bytes(
            (f'{{"event":"fill","coordinate":{index}}}\n'.encode()) * 1024
        )
        transcripts.append(transcript)
    _write_json(
        evidence / "coordinate-0.economic.attestation.json",
        {"transcript_sha256": hashlib.sha256(transcripts[0].read_bytes()).hexdigest()},
    )

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
    return root, [source, *transcripts]


def _remote_receipt(archive_root: Path, local: dict) -> Path:
    objects = []
    for expected in local["remote_readback_objects"]["objects"]:
        path = archive_root / expected["relative_path"]
        raw = path.read_bytes()
        md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        index = expected["index"]
        metadata = {
            "drive_file_id": f"driveFile{index:08d}",
            "drive_parent_id": DRIVE_PARENT_ID,
            "drive_file_name": path.name,
            "mime_type": "application/octet-stream",
            "content_size_bytes": len(raw),
            "md5_checksum": md5,
            "modified_time": "2026-07-22T11:55:00+00:00",
            "version": str(index + 1),
            "head_revision_id": f"headRevision{index:08d}",
            "trashed": False,
        }
        objects.append(
            {
                "index": index,
                "offset_bytes": expected["offset_bytes"],
                "relative_path": expected["relative_path"],
                "size_bytes": expected["size_bytes"],
                "sha256": expected["sha256"],
                "metadata_before": metadata,
                "metadata_after": dict(metadata),
                "downloaded": {
                    "content_size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "md5_checksum": md5,
                },
            }
        )
    body = {
        "contract": REMOTE_READBACK_RECEIPT_CONTRACT,
        "schema_version": 1,
        "status": "REMOTE_VERIFIED",
        "provider": "GOOGLE_DRIVE",
        "verification_method": "AUTHENTICATED_EXTERNAL_RAW_READBACK",
        "job_sha256": local["job_sha256"],
        "completion_sha256": local["completion_sha256"],
        "bundle_kind": local["bundle_kind"],
        "manifest_sha256": local["manifest_sha256"],
        "local_archive_receipt_sha256": local["receipt_sha256"],
        "archive_sha256": local["archive_sha256"],
        "archive_size_bytes": local["archive_size_bytes"],
        "object_set_sha256": local["remote_readback_objects"]["object_set_sha256"],
        "object_count": local["remote_readback_objects"]["object_count"],
        "expected_drive_parent_id": DRIVE_PARENT_ID,
        "drive_parent": {
            "drive_folder_id": DRIVE_PARENT_ID,
            "drive_folder_parent_id": "driveRoot123456",
            "drive_folder_name": "historical-job-readback",
            "mime_type": "application/vnd.google-apps.folder",
            "trashed": False,
        },
        "readback_at_utc": "2026-07-22T12:00:00+00:00",
        "objects": objects,
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
    receipt = {**body, "remote_receipt_sha256": _canonical_sha256(body)}
    path = (
        archive_root
        / "remote-receipts"
        / (
            f"remote-job-{local['job_sha256']}-{local['manifest_sha256']}-"
            f"{receipt['remote_receipt_sha256']}.json"
        )
    )
    _write_json(path, receipt)
    return path


def _setup(tmp_path: Path, **kwargs) -> tuple[Path, Path, Path, list[Path], dict]:
    root, raw_paths = _terminal_run(tmp_path, **kwargs)
    archive_root = tmp_path / "drive"
    local = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    archive_receipt = next((archive_root / "receipts").glob("*.json"))
    remote_receipt = _remote_receipt(archive_root, local)
    return root, archive_receipt, remote_receipt, raw_paths, local


def _reseal_remote(path: Path, mutate) -> Path:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    body = {key: item for key, item in value.items() if key != "remote_receipt_sha256"}
    value["remote_receipt_sha256"] = _canonical_sha256(body)
    renamed = path.with_name(
        f"remote-job-{value['job_sha256']}-{value['manifest_sha256']}-"
        f"{value['remote_receipt_sha256']}.json"
    )
    path.unlink()
    _write_json(renamed, value)
    return renamed


def test_verified_remote_readback_reclaims_only_allowlisted_raw_and_is_idempotent(
    tmp_path: Path,
) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    retained = (
        root
        / "jobs"
        / JOB_SHA256
        / "economic-evidence"
        / "coordinate-0.economic.attestation.json"
    )

    verified = verify_historical_job_raw_reclaim(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
    )

    assert verified["status"] == "RAW_RECLAIM_VERIFIED_NOT_EXECUTED"
    assert verified["plan"]["target_count"] == 3
    assert all(path.is_file() for path in raw_paths)

    receipt = reclaim_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
    )
    repeated = reclaim_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
    )

    assert receipt == repeated
    assert receipt["status"] == "RAW_RECLAIMED"
    assert receipt["deleted_file_count"] == 3
    assert all(not path.exists() for path in raw_paths)
    assert retained.is_file()
    assert (root / "jobs" / JOB_SHA256 / "completion.json").is_file()
    assert len(list((root / "reclaim-receipts").glob("plan-*.json"))) == 1
    assert len(list((root / "reclaim-receipts").glob("reclaim-*.json"))) == 1

    deep = verify_existing_historical_job_raw_reclaim(
        run_root=root,
        archive_root=local_path.parent.parent,
        job_sha256=JOB_SHA256,
        expected_drive_parent_id=DRIVE_PARENT_ID,
    )
    assert deep["status"] == "LOCALLY_ARCHIVED_AND_RAW_RECLAIMED"
    assert deep["archive_and_parts_verified"] is True
    assert deep["retained_bytes_verified"] is True
    assert deep["all_raw_targets_missing"] is True


def test_wrong_expected_drive_parent_fails_before_any_unlink(tmp_path: Path) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="lineage or authority is invalid",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id="anotherParent123",
        )

    assert all(path.is_file() for path in raw_paths)
    assert not list((root / "reclaim-receipts").glob("plan-*.json"))


def test_resealed_remote_download_hash_drift_fails_closed(tmp_path: Path) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    remote_path = _reseal_remote(
        remote_path,
        lambda value: value["objects"][0]["downloaded"].__setitem__("sha256", "f" * 64),
    )

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="Drive raw readback differs",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )

    assert all(path.is_file() for path in raw_paths)


def test_current_raw_drift_fails_before_reclaim_plan(tmp_path: Path) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    raw_paths[-1].write_bytes(b"drift")

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="does not match its full inventory",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )

    assert raw_paths[0].is_file()
    assert raw_paths[1].is_file()
    assert not list((root / "reclaim-receipts").glob("plan-*.json"))


def test_retained_job_file_drift_fails_before_raw_unlink(tmp_path: Path) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    retained = (
        root
        / "jobs"
        / JOB_SHA256
        / "economic-evidence"
        / "coordinate-0.economic.attestation.json"
    )
    retained.write_bytes(b"{}\n")

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="does not match its full inventory",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )

    assert all(path.is_file() for path in raw_paths)


def test_partial_unlink_crash_resumes_only_from_the_same_append_only_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    original = reclaim_module._unlink_target
    calls = 0

    def interrupt_after_first(root_path: Path, row: dict) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DojoHistoricalRawReclaimError("simulated interruption")
        return original(root_path, row)

    monkeypatch.setattr(reclaim_module, "_unlink_target", interrupt_after_first)
    with pytest.raises(DojoHistoricalRawReclaimError, match="simulated interruption"):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )

    assert len(list((root / "reclaim-receipts").glob("plan-*.json"))) == 1
    assert sum(path.exists() for path in raw_paths) == len(raw_paths) - 1
    assert not list((root / "reclaim-receipts").glob("reclaim-*.json"))

    monkeypatch.setattr(reclaim_module, "_unlink_target", original)
    rogue = (
        root / "jobs" / JOB_SHA256 / "economic-evidence" / "unmanifested.economic.jsonl"
    )
    rogue.write_bytes(b"rogue\n")
    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="unmanifested job evidence appeared",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )
    rogue.unlink()

    receipt = reclaim_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
    )

    assert receipt["status"] == "RAW_RECLAIMED"
    assert all(not path.exists() for path in raw_paths)


def test_split_remote_parts_are_all_required_and_revision_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_job_archive.ARCHIVE_PART_BYTES",
        64 * 1024,
    )
    root, local_path, remote_path, raw_paths, local = _setup(
        tmp_path,
        source_size=512 * 1024,
        incompressible=True,
    )
    assert local["remote_readback_objects"]["object_count"] > 1
    remote_path = _reseal_remote(
        remote_path,
        lambda value: value["objects"][1]["metadata_after"].__setitem__(
            "version", "99"
        ),
    )

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="Drive raw readback differs",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )

    assert all(path.is_file() for path in raw_paths)


def test_post_reclaim_verifier_rejects_reappeared_raw_and_ambiguous_plan(
    tmp_path: Path,
) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    restored_bytes = raw_paths[0].read_bytes()
    reclaim_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
    )
    raw_paths[0].write_bytes(restored_bytes)

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="raw target still present",
    ):
        verify_existing_historical_job_raw_reclaim(
            run_root=root,
            archive_root=local_path.parent.parent,
            job_sha256=JOB_SHA256,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )

    raw_paths[0].unlink()
    plan = next((root / "reclaim-receipts").glob("plan-*.json"))
    ambiguous = plan.with_name(f"plan-job-{JOB_SHA256}-{'f' * 64}.json")
    ambiguous.write_bytes(plan.read_bytes())
    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="exact-one plan and receipt",
    ):
        verify_existing_historical_job_raw_reclaim(
            run_root=root,
            archive_root=local_path.parent.parent,
            job_sha256=JOB_SHA256,
            expected_drive_parent_id=DRIVE_PARENT_ID,
        )
