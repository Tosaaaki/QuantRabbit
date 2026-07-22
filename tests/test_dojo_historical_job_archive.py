from __future__ import annotations

import hashlib
import io
import json
import os
import random
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import quant_rabbit.dojo_historical_job_archive as archive_module
from quant_rabbit.dojo_historical_job_archive import (
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

    def archive() -> dict:
        return archive_completed_historical_job(
            run_root=root,
            job_sha256=JOB_SHA256,
            archive_root=archive_root,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        receipts = list(executor.map(lambda _: archive(), range(2)))

    assert receipts[0] == receipts[1]
    assert len(list((archive_root / "archives").glob("*.tar.zst"))) == 1
    assert len(list((archive_root / "receipts").glob("*.json"))) == 1
    assert not list((archive_root / "archives").glob("*.part"))


def test_large_archive_is_split_into_bounded_content_addressed_readback_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    readback = receipt["remote_readback_objects"]

    assert readback["object_count"] > 1
    assert all(row["size_bytes"] <= 64 * 1024 for row in readback["objects"])
    assert all(
        Path(row["relative_path"]).name.endswith(f"-{row['sha256']}.bin")
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
