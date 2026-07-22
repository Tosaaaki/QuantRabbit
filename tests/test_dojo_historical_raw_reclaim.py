from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import json
import random
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import quant_rabbit.dojo_historical_raw_reclaim as reclaim_module
from quant_rabbit.dojo_historical_job_archive import (
    IMPLEMENTATION_MANIFEST_CONTRACT,
    JOB_COMPLETION_CONTRACT,
    _canonical_bytes,
    _canonical_sha256,
    archive_completed_historical_job,
)
from quant_rabbit.dojo_historical_raw_reclaim import (
    ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT,
    RAW_STORAGE_BOUNDARY_CONTRACT,
    REMOTE_READBACK_CANDIDATE_CONTRACT,
    REMOTE_READBACK_EVIDENCE_CONTRACT,
    REMOTE_READBACK_RECEIPT_CONTRACT,
    REMOTE_READBACK_ATTESTATION_BODY_CONTRACT,
    REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT,
    DojoHistoricalRawReclaimError,
    create_historical_job_remote_readback_receipt,
    enroll_historical_job_attestation_public_key,
    historical_raw_storage_boundary,
    publish_historical_job_signed_remote_readback_receipt,
    reclaim_historical_job_raw,
    restore_historical_job_raw,
    verify_existing_historical_job_raw_reclaim,
    verify_existing_historical_job_raw_restore,
    verify_historical_job_raw_reclaim,
)


JOB_SHA256 = "a" * 64
DRIVE_PARENT_ID = "driveParent12345"
ZSTD_BIN = str(Path(shutil.which("zstd") or "zstd").resolve())


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


def _remote_evidence_packet(
    tmp_path: Path,
    archive_root: Path,
    local: dict,
    *,
    metadata_parent_id: str = DRIVE_PARENT_ID,
) -> Path:
    readback_at = "2026-07-22T12:00:00+00:00"
    downloads = tmp_path / "authenticated-drive-downloads"
    downloads.mkdir(exist_ok=True)
    objects = []
    for expected in local["remote_readback_objects"]["objects"]:
        local_object = archive_root / expected["relative_path"]
        raw = local_object.read_bytes()
        downloaded = downloads / f"download-{expected['index']:05d}.bin"
        downloaded.write_bytes(raw)
        md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        index = expected["index"]
        metadata = {
            "drive_file_id": f"driveFile{index:08d}",
            "drive_parent_id": metadata_parent_id,
            "drive_file_name": local_object.name,
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
                "relative_path": expected["relative_path"],
                "readback_at_utc": readback_at,
                "metadata_before": metadata,
                "downloaded_local_path": str(downloaded.resolve()),
                "metadata_after": dict(metadata),
            }
        )
    packet = {
        "contract": REMOTE_READBACK_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "provider": "GOOGLE_DRIVE",
        "verification_method": "AUTHENTICATED_EXTERNAL_RAW_READBACK",
        "job_sha256": local["job_sha256"],
        "completion_sha256": local["completion_sha256"],
        "manifest_sha256": local["manifest_sha256"],
        "local_archive_receipt_sha256": local["receipt_sha256"],
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
        "readback_at_utc": readback_at,
        "objects": objects,
        "external_readback_attested": True,
        "source_deletion_requested": False,
    }
    path = tmp_path / "authenticated-drive-readback-evidence.json"
    _write_json(path, packet)
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


def _signed_remote_lineage(
    tmp_path: Path,
    *,
    mutate_body=None,
    mutate_envelope=None,
    private_key: Ed25519PrivateKey | None = None,
    signing_key: Ed25519PrivateKey | None = None,
    enrolled_at: datetime | None = None,
    readback_at: datetime | None = None,
    issued_at: datetime | None = None,
    expires_at: datetime | None = None,
    verification_now: datetime | None = None,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    list[Path],
    dict,
    Ed25519PrivateKey,
    datetime,
]:
    root, raw_paths = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    local = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    local_path = next((archive_root / "receipts").glob("job-*.json"))
    signer = private_key or Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)
    public_hex = (
        signer.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    enrolled = enrolled_at or datetime(2026, 7, 22, 11, 30, tzinfo=timezone.utc)
    with patch.object(reclaim_module, "_utc_now", return_value=enrolled):
        seal = enroll_historical_job_attestation_public_key(
            run_root=root,
            archive_receipt_path=local_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_public_key_hex=public_hex,
            zstd_bin=ZSTD_BIN,
        )
    assert seal["contract"] == ATTESTATION_PUBLIC_KEY_SEAL_CONTRACT
    seal_path = next((archive_root / "remote-authorities").glob("key-job-*.json"))
    observed = readback_at or datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    issued = issued_at or observed + timedelta(minutes=1)
    expires = expires_at or issued + timedelta(minutes=8)
    objects = []
    for expected in local["remote_readback_objects"]["objects"]:
        path = archive_root / expected["relative_path"]
        raw = path.read_bytes()
        md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        index = expected["index"]
        revision = f"revision+/{index}="
        metadata = {
            "drive_file_id": f"driveFile{index:08d}",
            "drive_parent_id": DRIVE_PARENT_ID,
            "drive_file_name": path.name,
            "mime_type": "application/octet-stream",
            "content_size_bytes": len(raw),
            "md5_checksum": md5,
            "modified_time": (observed - timedelta(minutes=1)).isoformat(),
            "version": str(index + 1),
            "head_revision_id": revision,
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
                "listed_revision_ids": [f"prior+/{index}=", revision],
                "drivefs_file_id_xattr": metadata["drive_file_id"],
                "drivefs_md5_field48": md5,
                "drivefs_version_field57": metadata["version"],
                "drivefs_current_revision_id_field78": revision,
            }
        )
    body = {
        "contract": REMOTE_READBACK_ATTESTATION_BODY_CONTRACT,
        "schema_version": 2,
        "attestation_id": hashlib.sha256(
            f"{JOB_SHA256}:{issued.isoformat()}".encode()
        ).hexdigest(),
        "provider": "GOOGLE_DRIVE",
        "verification_method": (
            "GOOGLE_DRIVE_V3_FILES_GET_REVISIONS_LIST_AND_"
            "INDEPENDENT_REVISION_READBACK"
        ),
        "issued_at_utc": issued.isoformat(),
        "expires_at_utc": expires.isoformat(),
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
        "readback_at_utc": observed.isoformat(),
        "objects": objects,
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
    if mutate_body is not None:
        mutate_body(body)
    signature = base64.b64encode(
        (signing_key or signer).sign(_canonical_bytes(body))
    ).decode("ascii")
    envelope_body = {
        "contract": REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT,
        "schema_version": 2,
        "algorithm": "ED25519",
        "public_key_sha256": seal["public_key_sha256"],
        "body": body,
        "signature_base64": signature,
    }
    envelope = {
        **envelope_body,
        "remote_receipt_sha256": _canonical_sha256(envelope_body),
    }
    if mutate_envelope is not None:
        mutate_envelope(envelope)
        envelope_body = {
            key: value
            for key, value in envelope.items()
            if key != "remote_receipt_sha256"
        }
        envelope["remote_receipt_sha256"] = _canonical_sha256(envelope_body)
    packet_path = tmp_path / "external-signed-drive-attestation.json"
    _write_json(packet_path, envelope)
    now = verification_now or issued + timedelta(minutes=1)
    remote = publish_historical_job_signed_remote_readback_receipt(
        run_root=root,
        archive_receipt_path=local_path,
        signed_attestation_path=packet_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        attestation_authority_seal_path=seal_path,
        zstd_bin=ZSTD_BIN,
        now_utc=now,
    )
    remote_path = (
        archive_root
        / "remote-receipts"
        / (
            f"signed-job-{JOB_SHA256}-{local['manifest_sha256']}-"
            f"{remote['remote_receipt_sha256']}.json"
        )
    )
    return (
        root,
        local_path,
        seal_path,
        remote_path,
        raw_paths,
        local,
        signer,
        now,
    )


def _historical_reclaim_lineage(
    *,
    root: Path,
    local_path: Path,
    remote_path: Path,
) -> tuple[Path, Path, tuple[dict, ...], dict[str, bytes]]:
    _, local, manifest = reclaim_module._validate_local_archive(
        run_root=root,
        archive_receipt_path=local_path,
        require_full_inventory=True,
        zstd_bin=ZSTD_BIN,
    )
    targets, retained = reclaim_module._reclaimable_rows(
        run_root=root,
        job_sha256=JOB_SHA256,
        manifest=manifest,
    )
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    plan_body = {
        "contract": reclaim_module.RECLAIM_PLAN_CONTRACT,
        "schema_version": 1,
        "job_sha256": JOB_SHA256,
        "completion_sha256": local["completion_sha256"],
        "bundle_kind": local["bundle_kind"],
        "manifest_sha256": local["manifest_sha256"],
        "local_archive_receipt_sha256": local["receipt_sha256"],
        "remote_receipt_sha256": remote["remote_receipt_sha256"],
        "object_set_sha256": local["remote_readback_objects"]["object_set_sha256"],
        "archive_sha256": local["archive_sha256"],
        "archive_size_bytes": local["archive_size_bytes"],
        "reclaim_mode": "UNLINK_EXACT_ALLOWLISTED_RAW",
        "target_count": len(targets),
        "target_bytes": sum(row["size_bytes"] for row in targets),
        "targets": list(targets),
        "retained_file_count": len(retained),
        "retained_bytes": sum(row["size_bytes"] for row in retained),
        "full_source_inventory_verified": True,
        "remote_raw_readback_verified": True,
        **reclaim_module._AUTHORITY,
    }
    plan = {**plan_body, "plan_sha256": _canonical_sha256(plan_body)}
    reclaim_root = root / "reclaim-receipts"
    plan_path = reclaim_root / (
        f"plan-job-{JOB_SHA256}-{plan['remote_receipt_sha256']}.json"
    )
    _write_json(plan_path, plan)
    receipt_body = {
        "contract": reclaim_module.RECLAIM_RECEIPT_CONTRACT,
        "schema_version": 1,
        "status": "RAW_RECLAIMED",
        "job_sha256": JOB_SHA256,
        "completion_sha256": plan["completion_sha256"],
        "bundle_kind": plan["bundle_kind"],
        "manifest_sha256": plan["manifest_sha256"],
        "local_archive_receipt_sha256": plan["local_archive_receipt_sha256"],
        "remote_receipt_sha256": plan["remote_receipt_sha256"],
        "reclaim_plan_sha256": plan["plan_sha256"],
        "completed_at_utc": "2026-07-22T12:30:00+00:00",
        "deleted_file_count": plan["target_count"],
        "deleted_files": plan["targets"],
        "reclaimed_logical_bytes": plan["target_bytes"],
        "reclaimed_allocated_bytes_observed": plan["target_bytes"],
        "free_disk_bytes_before": 1,
        "free_disk_bytes_after": 2,
        "retained_file_count": plan["retained_file_count"],
        "retained_bytes": plan["retained_bytes"],
        "restore_requires_verified_archive": True,
        **reclaim_module._AUTHORITY,
    }
    receipt = {
        **receipt_body,
        "reclaim_receipt_sha256": _canonical_sha256(receipt_body),
    }
    receipt_path = reclaim_root / (
        f"reclaim-{JOB_SHA256}-{receipt['reclaim_receipt_sha256']}.json"
    )
    _write_json(receipt_path, receipt)
    original = {row["path"]: (root / row["path"]).read_bytes() for row in targets}
    for row in targets:
        (root / row["path"]).unlink()
    return plan_path, receipt_path, targets, original


def _write_hidden_authority_fork(seal_path: Path) -> Path:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    second_key = Ed25519PrivateKey.from_private_bytes(b"\x59" * 32).public_key()
    public_raw = second_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    seal["public_key_hex"] = public_raw.hex()
    seal["public_key_sha256"] = hashlib.sha256(public_raw).hexdigest()
    body = {key: value for key, value in seal.items() if key != "authority_seal_sha256"}
    seal["authority_seal_sha256"] = _canonical_sha256(body)
    hidden = seal_path.parent / ".hidden-second-key.json"
    _write_json(hidden, seal)
    return hidden


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


def test_local_evidence_seals_candidate_only_without_deletion_authority(
    tmp_path: Path,
) -> None:
    root, raw_paths = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    local = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    local_path = next((archive_root / "receipts").glob("job-*.json"))
    evidence_path = _remote_evidence_packet(tmp_path, archive_root, local)

    receipt = create_historical_job_remote_readback_receipt(
        run_root=root,
        archive_receipt_path=local_path,
        evidence_packet_path=evidence_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=ZSTD_BIN,
    )
    candidate_path = (
        archive_root
        / "remote-candidates"
        / (
            f"candidate-job-{JOB_SHA256}-{local['manifest_sha256']}-"
            f"{receipt['candidate_sha256']}.json"
        )
    )
    assert receipt["contract"] == REMOTE_READBACK_CANDIDATE_CONTRACT
    assert receipt["status"] == "CANDIDATE_ONLY"
    assert receipt["remote_verified"] is False
    assert receipt["raw_reclaim_eligible"] is False
    assert receipt["trusted_provider_attestation_present"] is False
    assert candidate_path.is_file()
    assert all(path.is_file() for path in raw_paths)
    with pytest.raises(DojoHistoricalRawReclaimError, match="attestation|trusted"):
        verify_historical_job_raw_reclaim(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=candidate_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )

    repeated = create_historical_job_remote_readback_receipt(
        run_root=root,
        archive_receipt_path=local_path,
        evidence_packet_path=evidence_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=ZSTD_BIN,
    )
    assert repeated == receipt
    assert (
        len(list((archive_root / "remote-candidates").glob("candidate-job-*.json")))
        == 1
    )
    assert all(path.is_file() for path in raw_paths)


def test_remote_evidence_rejects_wrong_drive_parent_before_sealing(
    tmp_path: Path,
) -> None:
    root, raw_paths = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    local = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    local_path = next((archive_root / "receipts").glob("job-*.json"))
    evidence_path = _remote_evidence_packet(
        tmp_path,
        archive_root,
        local,
        metadata_parent_id="wrongDriveParent12345",
    )

    with pytest.raises(DojoHistoricalRawReclaimError, match="metadata is invalid"):
        create_historical_job_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            evidence_packet_path=evidence_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )
    assert not list((archive_root / "remote-receipts").glob("remote-job-*.json"))
    assert all(path.is_file() for path in raw_paths)


def test_remote_evidence_rejects_revision_drift_and_download_symlink(
    tmp_path: Path,
) -> None:
    root, raw_paths = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    local = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    local_path = next((archive_root / "receipts").glob("job-*.json"))
    evidence_path = _remote_evidence_packet(tmp_path, archive_root, local)
    packet = json.loads(evidence_path.read_text(encoding="utf-8"))
    packet["objects"][0]["metadata_after"]["version"] = "99"
    _write_json(evidence_path, packet)
    with pytest.raises(DojoHistoricalRawReclaimError, match="revision drifted"):
        create_historical_job_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            evidence_packet_path=evidence_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )

    evidence_path.unlink()
    evidence_path = _remote_evidence_packet(tmp_path, archive_root, local)
    packet = json.loads(evidence_path.read_text(encoding="utf-8"))
    downloaded = Path(packet["objects"][0]["downloaded_local_path"])
    downloaded.unlink()
    downloaded.symlink_to(
        archive_root / local["remote_readback_objects"]["objects"][0]["relative_path"]
    )
    with pytest.raises(DojoHistoricalRawReclaimError, match="symlink component"):
        create_historical_job_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            evidence_packet_path=evidence_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )
    assert not list((archive_root / "remote-receipts").glob("remote-job-*.json"))
    assert all(path.is_file() for path in raw_paths)


def test_remote_evidence_rejects_downloaded_byte_mismatch(
    tmp_path: Path,
) -> None:
    root, raw_paths = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    local = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    local_path = next((archive_root / "receipts").glob("job-*.json"))
    evidence_path = _remote_evidence_packet(tmp_path, archive_root, local)
    packet = json.loads(evidence_path.read_text(encoding="utf-8"))
    downloaded = Path(packet["objects"][0]["downloaded_local_path"])
    downloaded.write_bytes(downloaded.read_bytes() + b"tampered")

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="downloaded Drive bytes differ",
    ):
        create_historical_job_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            evidence_packet_path=evidence_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )
    assert not list((archive_root / "remote-receipts").glob("remote-job-*.json"))
    assert all(path.is_file() for path in raw_paths)


def test_remote_evidence_rejects_duplicate_drive_object_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_job_archive.ARCHIVE_PART_BYTES",
        64 * 1024,
    )
    root, raw_paths = _terminal_run(
        tmp_path,
        source_size=512 * 1024,
        incompressible=True,
    )
    archive_root = tmp_path / "drive"
    local = archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    assert local["remote_readback_objects"]["object_count"] > 1
    local_path = next((archive_root / "receipts").glob("job-*.json"))
    evidence_path = _remote_evidence_packet(tmp_path, archive_root, local)
    packet = json.loads(evidence_path.read_text(encoding="utf-8"))
    duplicate_id = packet["objects"][0]["metadata_after"]["drive_file_id"]
    for field in ("metadata_before", "metadata_after"):
        packet["objects"][1][field]["drive_file_id"] = duplicate_id
    _write_json(evidence_path, packet)

    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="reuses a Drive object or revision",
    ):
        create_historical_job_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            evidence_packet_path=evidence_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )
    assert not list((archive_root / "remote-receipts").glob("remote-job-*.json"))
    assert all(path.is_file() for path in raw_paths)


def test_handmade_remote_verified_packet_cannot_authorize_reclaim(
    tmp_path: Path,
) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)

    with pytest.raises(DojoHistoricalRawReclaimError, match="attestation|trusted"):
        verify_historical_job_raw_reclaim(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )
    with pytest.raises(DojoHistoricalRawReclaimError, match="attestation|trusted"):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
        )

    assert all(path.is_file() for path in raw_paths)
    assert not list((root / "reclaim-receipts").glob("plan-*.json"))
    assert not list((root / "reclaim-receipts").glob("reclaim-*.json"))


def test_signed_drive_attestation_reclaims_then_restores_exact_roundtrip(
    tmp_path: Path,
) -> None:
    (
        root,
        local_path,
        seal_path,
        remote_path,
        raw_paths,
        _,
        _,
        now,
    ) = _signed_remote_lineage(tmp_path)
    originals = {path: path.read_bytes() for path in raw_paths}
    verified = verify_historical_job_raw_reclaim(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        attestation_authority_seal_path=seal_path,
        zstd_bin=ZSTD_BIN,
        now_utc=now,
    )
    plan = verified["plan"]
    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="exact plan SHA/count/bytes",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_authority_seal_path=seal_path,
            confirmed_plan_sha256="0" * 64,
            confirmed_target_count=plan["target_count"],
            confirmed_target_bytes=plan["target_bytes"],
            zstd_bin=ZSTD_BIN,
            now_utc=now,
        )
    assert all(path.is_file() for path in raw_paths)

    reclaimed = reclaim_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        attestation_authority_seal_path=seal_path,
        confirmed_plan_sha256=plan["plan_sha256"],
        confirmed_target_count=plan["target_count"],
        confirmed_target_bytes=plan["target_bytes"],
        zstd_bin=ZSTD_BIN,
        now_utc=now,
    )
    assert reclaimed["status"] == "RAW_RECLAIMED"
    assert all(not path.exists() for path in raw_paths)
    for row in plan["targets"]:
        anchor = (root / row["path"]).parent / (
            reclaim_module._raw_retirement_anchor_name(row)
        )
        assert anchor.is_file()
        assert anchor.stat().st_size == 0

    existing = verify_existing_historical_job_raw_reclaim(
        run_root=root,
        archive_root=local_path.parent.parent,
        job_sha256=JOB_SHA256,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=ZSTD_BIN,
    )
    assert existing["remote_receipt_trusted"] is True
    assert existing["zero_byte_retirement_anchors_verified"] is True

    plan_path = next((root / "reclaim-v2-receipts").glob("plan-job-*.json"))
    reclaim_path = next((root / "reclaim-v2-receipts").glob("reclaim-*.json"))
    restored = restore_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        reclaim_plan_path=plan_path,
        reclaim_receipt_path=reclaim_path,
        zstd_bin=ZSTD_BIN,
    )
    assert restored["status"] == "RAW_RESTORED"
    assert {path: path.read_bytes() for path in raw_paths} == originals


def test_lock_replacement_cannot_create_a_second_authoritative_lock(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "authority.lock"
    displaced = tmp_path / "displaced.lock"
    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="path identity changed",
    ):
        with reclaim_module._exclusive_lock(
            lock_path,
            field="adversarial authority lock",
        ) as guard:
            lock_path.rename(displaced)
            lock_path.write_bytes(b"replacement")
            guard.assert_stable()

    assert displaced.is_file()
    assert lock_path.read_bytes() == b"replacement"


def test_reclaim_lock_replacement_stops_before_first_raw_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, local_path, seal_path, remote_path, raw_paths, _, _, now = (
        _signed_remote_lineage(tmp_path)
    )
    verified = verify_historical_job_raw_reclaim(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=ZSTD_BIN,
        attestation_authority_seal_path=seal_path,
        now_utc=now,
    )
    plan = verified["plan"]
    original_write = reclaim_module._write_once

    def replace_job_lock(path: Path, value: dict, *, field: str) -> None:
        original_write(path, value, field=field)
        if field != "raw reclaim plan":
            return
        lock_path = root / "reclaim-v2-receipts" / f".job-{JOB_SHA256}.lock"
        displaced = lock_path.with_name(f"{lock_path.name}.displaced")
        lock_path.rename(displaced)
        lock_path.write_bytes(b"replacement")

    monkeypatch.setattr(reclaim_module, "_write_once", replace_job_lock)
    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="path identity changed",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=ZSTD_BIN,
            attestation_authority_seal_path=seal_path,
            confirmed_plan_sha256=plan["plan_sha256"],
            confirmed_target_count=plan["target_count"],
            confirmed_target_bytes=plan["target_bytes"],
            now_utc=now,
        )
    assert all(path.is_file() for path in raw_paths)
    assert not list((root / "reclaim-v2-receipts").glob("reclaim-*.json"))


def test_hidden_second_key_fork_blocks_enroll_publish_verify_and_reclaim(
    tmp_path: Path,
) -> None:
    root, local_path, seal_path, remote_path, raw_paths, _, _, now = (
        _signed_remote_lineage(tmp_path)
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _write_hidden_authority_fork(seal_path)

    with patch.object(
        reclaim_module,
        "_utc_now",
        return_value=datetime.fromisoformat(seal["enrolled_at_utc"]),
    ):
        with pytest.raises(DojoHistoricalRawReclaimError, match="multiple.*forks"):
            enroll_historical_job_attestation_public_key(
                run_root=root,
                archive_receipt_path=local_path,
                expected_drive_parent_id=DRIVE_PARENT_ID,
                attestation_public_key_hex=seal["public_key_hex"],
                zstd_bin=ZSTD_BIN,
            )
    with pytest.raises(DojoHistoricalRawReclaimError, match="multiple.*forks"):
        publish_historical_job_signed_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            signed_attestation_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_authority_seal_path=seal_path,
            zstd_bin=ZSTD_BIN,
            now_utc=now,
        )
    with pytest.raises(DojoHistoricalRawReclaimError, match="multiple.*forks"):
        verify_historical_job_raw_reclaim(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_authority_seal_path=seal_path,
            zstd_bin=ZSTD_BIN,
            now_utc=now,
        )
    with pytest.raises(DojoHistoricalRawReclaimError, match="multiple.*forks"):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_authority_seal_path=seal_path,
            zstd_bin=ZSTD_BIN,
            now_utc=now,
        )
    assert all(path.is_file() for path in raw_paths)


def test_hidden_second_key_fork_blocks_signed_v2_restore(tmp_path: Path) -> None:
    root, local_path, seal_path, remote_path, raw_paths, _, _, now = (
        _signed_remote_lineage(tmp_path)
    )
    verified = verify_historical_job_raw_reclaim(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        attestation_authority_seal_path=seal_path,
        zstd_bin=ZSTD_BIN,
        now_utc=now,
    )
    plan = verified["plan"]
    reclaim_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        attestation_authority_seal_path=seal_path,
        confirmed_plan_sha256=plan["plan_sha256"],
        confirmed_target_count=plan["target_count"],
        confirmed_target_bytes=plan["target_bytes"],
        zstd_bin=ZSTD_BIN,
        now_utc=now,
    )
    _write_hidden_authority_fork(seal_path)
    plan_path = next((root / "reclaim-v2-receipts").glob("plan-job-*.json"))
    reclaim_path = next((root / "reclaim-v2-receipts").glob("reclaim-*.json"))
    with pytest.raises(DojoHistoricalRawReclaimError, match="multiple.*forks"):
        restore_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            reclaim_plan_path=plan_path,
            reclaim_receipt_path=reclaim_path,
            zstd_bin=ZSTD_BIN,
        )
    assert all(not path.exists() for path in raw_paths)


def test_reclaim_plan_binds_absolute_zstd_bytes_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, local_path, seal_path, remote_path, raw_paths, _, _, now = (
        _signed_remote_lineage(tmp_path)
    )
    wrapper = tmp_path / "sealed-zstd"

    def write_wrapper(version: str) -> None:
        wrapper.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = "--version" ]; then echo "{version}"; exit 0; fi\n'
            f'exec "{ZSTD_BIN}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o700)

    write_wrapper("sealed-zstd-v1")
    verified = verify_historical_job_raw_reclaim(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=str(wrapper),
        attestation_authority_seal_path=seal_path,
        now_utc=now,
    )
    plan = verified["plan"]
    assert plan["zstd_executable_path"] == str(wrapper.resolve())
    assert plan["zstd_version"] == "sealed-zstd-v1"
    assert plan["zstd_executable_size_bytes"] == wrapper.stat().st_size
    assert (
        plan["zstd_executable_sha256"]
        == hashlib.sha256(wrapper.read_bytes()).hexdigest()
    )

    write_wrapper("sealed-zstd-v2")
    with pytest.raises(
        DojoHistoricalRawReclaimError,
        match="exact plan SHA/count/bytes",
    ):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin=str(wrapper),
            attestation_authority_seal_path=seal_path,
            confirmed_plan_sha256=plan["plan_sha256"],
            confirmed_target_count=plan["target_count"],
            confirmed_target_bytes=plan["target_bytes"],
            now_utc=now,
        )
    assert all(path.is_file() for path in raw_paths)

    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "zstd").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (shadow / "zstd").chmod(0o700)
    monkeypatch.setenv("PATH", str(shadow))
    with pytest.raises(DojoHistoricalRawReclaimError, match="absolute path"):
        verify_historical_job_raw_reclaim(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            zstd_bin="zstd",
            attestation_authority_seal_path=seal_path,
            now_utc=now,
        )


def test_signed_attestation_rejects_stale_key_reenrollment_and_tamper(
    tmp_path: Path,
) -> None:
    with pytest.raises(DojoHistoricalRawReclaimError, match="stale"):
        _signed_remote_lineage(
            tmp_path / "stale",
            issued_at=datetime(2026, 7, 22, 12, 1, tzinfo=timezone.utc),
            expires_at=datetime(2026, 7, 22, 12, 1, 1, tzinfo=timezone.utc),
        )

    root, local_path, seal_path, remote_path, raw_paths, _, _, now = (
        _signed_remote_lineage(tmp_path / "valid")
    )
    wrong = Ed25519PrivateKey.from_private_bytes(b"\x29" * 32)
    wrong_public = (
        wrong.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    with patch.object(reclaim_module, "_utc_now", return_value=now):
        with pytest.raises(DojoHistoricalRawReclaimError, match="already enrolled"):
            enroll_historical_job_attestation_public_key(
                run_root=root,
                archive_receipt_path=local_path,
                expected_drive_parent_id=DRIVE_PARENT_ID,
                attestation_public_key_hex=wrong_public,
                zstd_bin=ZSTD_BIN,
            )

    receipt = json.loads(remote_path.read_text(encoding="utf-8"))
    tampered = tmp_path / "tampered.json"
    receipt["body"]["objects"][0]["drivefs_current_revision_id_field78"] = (
        "another+/revision="
    )
    envelope_body = {
        key: value for key, value in receipt.items() if key != "remote_receipt_sha256"
    }
    receipt["remote_receipt_sha256"] = _canonical_sha256(envelope_body)
    _write_json(tampered, receipt)
    with pytest.raises(DojoHistoricalRawReclaimError, match="signature"):
        publish_historical_job_signed_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            signed_attestation_path=tampered,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_authority_seal_path=seal_path,
            zstd_bin=ZSTD_BIN,
            now_utc=now,
        )
    assert all(path.is_file() for path in raw_paths)


def test_signed_attestation_rejects_wrong_key_future_ttl_and_base64(
    tmp_path: Path,
) -> None:
    wrong = Ed25519PrivateKey.from_private_bytes(b"\x39" * 32)
    with pytest.raises(DojoHistoricalRawReclaimError, match="signature"):
        _signed_remote_lineage(tmp_path / "wrong-key", signing_key=wrong)

    issued = datetime(2026, 7, 22, 12, 5, tzinfo=timezone.utc)
    with pytest.raises(DojoHistoricalRawReclaimError, match="stale"):
        _signed_remote_lineage(
            tmp_path / "future",
            issued_at=issued,
            verification_now=issued - timedelta(minutes=2),
        )
    with pytest.raises(DojoHistoricalRawReclaimError, match="stale"):
        _signed_remote_lineage(
            tmp_path / "ttl",
            issued_at=issued,
            expires_at=issued + timedelta(minutes=16),
        )

    def noncanonical_base64(envelope: dict) -> None:
        envelope["signature_base64"] += "\n"

    with pytest.raises(DojoHistoricalRawReclaimError, match="base64"):
        _signed_remote_lineage(
            tmp_path / "base64",
            mutate_envelope=noncanonical_base64,
        )


def test_signed_attestation_id_replay_is_rejected(tmp_path: Path) -> None:
    root, local_path, seal_path, remote_path, _, _, signer, now = (
        _signed_remote_lineage(tmp_path)
    )
    first = json.loads(remote_path.read_text(encoding="utf-8"))
    second_body = json.loads(json.dumps(first["body"]))
    second_body["expires_at_utc"] = (
        datetime.fromisoformat(second_body["expires_at_utc"]) + timedelta(seconds=1)
    ).isoformat()
    signature = base64.b64encode(signer.sign(_canonical_bytes(second_body))).decode(
        "ascii"
    )
    envelope_body = {
        "contract": REMOTE_READBACK_SIGNED_ATTESTATION_CONTRACT,
        "schema_version": 2,
        "algorithm": "ED25519",
        "public_key_sha256": first["public_key_sha256"],
        "body": second_body,
        "signature_base64": signature,
    }
    second = {
        **envelope_body,
        "remote_receipt_sha256": _canonical_sha256(envelope_body),
    }
    second_path = tmp_path / "second-same-id.json"
    _write_json(second_path, second)
    with pytest.raises(DojoHistoricalRawReclaimError, match="replayed"):
        publish_historical_job_signed_remote_readback_receipt(
            run_root=root,
            archive_receipt_path=local_path,
            signed_attestation_path=second_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_authority_seal_path=seal_path,
            zstd_bin=ZSTD_BIN,
            now_utc=now,
        )


def test_private_key_material_is_not_an_api_or_persistence_surface() -> None:
    source = inspect.getsource(reclaim_module)
    tree = ast.parse(source)
    function_args = {
        argument.arg
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
    }
    assert "private_key" not in function_args
    assert "Ed25519PrivateKey" not in source


def test_authority_enrollment_time_is_internal_clock_only(tmp_path: Path) -> None:
    assert (
        "enrolled_at_utc"
        not in inspect.signature(
            enroll_historical_job_attestation_public_key
        ).parameters
    )
    root, _ = _terminal_run(tmp_path)
    archive_root = tmp_path / "drive"
    archive_completed_historical_job(
        run_root=root,
        job_sha256=JOB_SHA256,
        archive_root=archive_root,
    )
    local_path = next((archive_root / "receipts").glob("job-*.json"))
    signer = Ed25519PrivateKey.from_private_bytes(b"\x69" * 32)
    public_hex = (
        signer.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    clock = datetime(2026, 7, 22, 11, 59, 30, tzinfo=timezone.utc)
    with patch.object(reclaim_module, "_utc_now", return_value=clock):
        seal = enroll_historical_job_attestation_public_key(
            run_root=root,
            archive_receipt_path=local_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_public_key_hex=public_hex,
            zstd_bin=ZSTD_BIN,
        )
    assert seal["enrolled_at_utc"] == clock.isoformat()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("trashed", "metadata is invalid"),
        ("revision_drift", "object or revision list drifted"),
        ("duplicate_revision", "object or revision list drifted"),
        ("drivefs_revision", "object or revision list drifted"),
        ("bool_index", "object or revision list drifted"),
    ],
)
def test_signed_attestation_rejects_provider_metadata_adversaries(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    def mutate(body: dict) -> None:
        row = body["objects"][0]
        if case == "trashed":
            row["metadata_before"]["trashed"] = True
            row["metadata_after"]["trashed"] = True
        elif case == "revision_drift":
            row["metadata_after"]["version"] = "99"
        elif case == "duplicate_revision":
            head = row["metadata_after"]["head_revision_id"]
            row["listed_revision_ids"] = [head, head]
        elif case == "drivefs_revision":
            row["drivefs_current_revision_id_field78"] = "wrong+/revision="
        elif case == "bool_index":
            row["index"] = False

    with pytest.raises(DojoHistoricalRawReclaimError, match=message):
        _signed_remote_lineage(tmp_path, mutate_body=mutate)


def test_expired_attestation_can_resume_only_its_existing_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, local_path, seal_path, remote_path, raw_paths, _, _, now = (
        _signed_remote_lineage(tmp_path)
    )
    verified = verify_historical_job_raw_reclaim(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        attestation_authority_seal_path=seal_path,
        zstd_bin=ZSTD_BIN,
        now_utc=now,
    )
    plan = verified["plan"]
    original = reclaim_module._retire_target
    calls = 0

    def crash_after_first(target_root: Path, row: dict) -> int:
        nonlocal calls
        released = original(target_root, row)
        calls += 1
        if calls == 1:
            raise DojoHistoricalRawReclaimError("simulated post-retirement crash")
        return released

    monkeypatch.setattr(reclaim_module, "_retire_target", crash_after_first)
    with pytest.raises(DojoHistoricalRawReclaimError, match="simulated"):
        reclaim_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            remote_receipt_path=remote_path,
            expected_drive_parent_id=DRIVE_PARENT_ID,
            attestation_authority_seal_path=seal_path,
            confirmed_plan_sha256=plan["plan_sha256"],
            confirmed_target_count=plan["target_count"],
            confirmed_target_bytes=plan["target_bytes"],
            zstd_bin=ZSTD_BIN,
            now_utc=now,
        )
    assert len(list((root / "reclaim-v2-receipts").glob("plan-job-*.json"))) == 1
    assert not list((root / "reclaim-v2-receipts").glob("reclaim-*.json"))
    assert sum(path.exists() for path in raw_paths) == len(raw_paths) - 1

    monkeypatch.setattr(reclaim_module, "_retire_target", original)
    receipt = reclaim_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        remote_receipt_path=remote_path,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        attestation_authority_seal_path=seal_path,
        confirmed_plan_sha256=plan["plan_sha256"],
        confirmed_target_count=plan["target_count"],
        confirmed_target_bytes=plan["target_bytes"],
        zstd_bin=ZSTD_BIN,
        now_utc=now + timedelta(days=1),
    )
    assert receipt["status"] == "RAW_RECLAIMED"
    assert all(not path.exists() for path in raw_paths)


def test_restore_is_deep_verified_content_addressed_and_idempotent(
    tmp_path: Path,
) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    plan_path, reclaim_path, targets, original = _historical_reclaim_lineage(
        root=root,
        local_path=local_path,
        remote_path=remote_path,
    )
    assert all(not path.exists() for path in raw_paths)

    receipt = restore_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        reclaim_plan_path=plan_path,
        reclaim_receipt_path=reclaim_path,
        zstd_bin=ZSTD_BIN,
    )
    repeated = restore_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        reclaim_plan_path=plan_path,
        reclaim_receipt_path=reclaim_path,
        zstd_bin=ZSTD_BIN,
    )

    assert repeated == receipt
    assert receipt["status"] == "RAW_RESTORED"
    assert receipt["remote_receipt_trusted"] is False
    assert receipt["restored_file_count"] == len(targets)
    assert len(list((root / "restore-receipts").glob("restore-*.json"))) == 1
    for row in targets:
        assert (root / row["path"]).read_bytes() == original[row["path"]]

    deep = verify_existing_historical_job_raw_restore(
        run_root=root,
        archive_root=local_path.parent.parent,
        job_sha256=JOB_SHA256,
        zstd_bin=ZSTD_BIN,
    )
    compact = verify_existing_historical_job_raw_reclaim(
        run_root=root,
        archive_root=local_path.parent.parent,
        job_sha256=JOB_SHA256,
        zstd_bin=ZSTD_BIN,
    )
    assert deep == compact
    assert deep["status"] == "LOCALLY_ARCHIVED_AND_RAW_RESTORED"
    assert deep["raw_reclaimed"] is False
    assert deep["raw_restored"] is True
    assert deep["remote_raw_readback_verified"] is False
    assert deep["all_raw_targets_present"] is True


def test_restore_resumes_after_publish_crash_without_replacing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    plan_path, reclaim_path, targets, original_bytes = _historical_reclaim_lineage(
        root=root,
        local_path=local_path,
        remote_path=remote_path,
    )
    original_publish = reclaim_module._publish_restore_target
    calls = 0

    def interrupt_after_first(**kwargs) -> bool:
        nonlocal calls
        published = original_publish(**kwargs)
        calls += 1
        if calls == 1:
            raise DojoHistoricalRawReclaimError("simulated interruption")
        return published

    monkeypatch.setattr(
        reclaim_module,
        "_publish_restore_target",
        interrupt_after_first,
    )
    with pytest.raises(DojoHistoricalRawReclaimError, match="simulated interruption"):
        restore_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            reclaim_plan_path=plan_path,
            reclaim_receipt_path=reclaim_path,
            zstd_bin=ZSTD_BIN,
        )

    assert sum(path.exists() for path in raw_paths) == 1
    assert not list((root / "restore-receipts").glob("restore-*.json"))

    monkeypatch.setattr(
        reclaim_module,
        "_publish_restore_target",
        original_publish,
    )
    receipt = restore_historical_job_raw(
        run_root=root,
        archive_receipt_path=local_path,
        reclaim_plan_path=plan_path,
        reclaim_receipt_path=reclaim_path,
        zstd_bin=ZSTD_BIN,
    )

    assert receipt["status"] == "RAW_RESTORED"
    assert receipt["published_file_count"] == len(targets) - 1
    assert receipt["preexisting_matching_file_count"] == 1
    for row in targets:
        assert (root / row["path"]).read_bytes() == original_bytes[row["path"]]


def test_restore_refuses_existing_wrong_bytes_without_overwrite(
    tmp_path: Path,
) -> None:
    root, local_path, remote_path, _, _ = _setup(tmp_path)
    plan_path, reclaim_path, targets, _ = _historical_reclaim_lineage(
        root=root,
        local_path=local_path,
        remote_path=remote_path,
    )
    wrong = root / targets[0]["path"]
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong.write_bytes(b"do-not-overwrite")

    with pytest.raises(DojoHistoricalRawReclaimError, match="bytes drifted"):
        restore_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            reclaim_plan_path=plan_path,
            reclaim_receipt_path=reclaim_path,
            zstd_bin=ZSTD_BIN,
        )

    assert wrong.read_bytes() == b"do-not-overwrite"
    assert not list((root / "restore-receipts").glob("restore-*.json"))


def test_restore_requires_absolute_zstd_before_archive_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, local_path, remote_path, raw_paths, _ = _setup(tmp_path)
    plan_path, reclaim_path, _, _ = _historical_reclaim_lineage(
        root=root,
        local_path=local_path,
        remote_path=remote_path,
    )

    archive_validation_called = False

    def unexpected_archive_validation(**_kwargs):
        nonlocal archive_validation_called
        archive_validation_called = True
        raise AssertionError("archive validation ran before zstd validation")

    monkeypatch.setattr(
        reclaim_module,
        "_validate_local_archive",
        unexpected_archive_validation,
    )
    with pytest.raises(DojoHistoricalRawReclaimError, match="absolute path"):
        restore_historical_job_raw(
            run_root=root,
            archive_receipt_path=local_path,
            reclaim_plan_path=plan_path,
            reclaim_receipt_path=reclaim_path,
            zstd_bin="zstd",
        )

    assert archive_validation_called is False
    assert all(not path.exists() for path in raw_paths)
    assert not list((root / "restore-receipts").glob("restore-*.json"))


def test_publish_file_exists_race_is_reverified_once_without_recursion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    stage_root = tmp_path / "stage"
    run_root.mkdir()
    stage_root.mkdir()
    payload = b"content-addressed-restore"
    stage_name = "target.raw"
    (stage_root / stage_name).write_bytes(payload)
    row = {
        "path": "nested/target.raw",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    original_publish = reclaim_module._atomic_rename_between_at_no_replace
    calls = 0

    def concurrent_exact_publish(*args, **kwargs) -> bool:
        nonlocal calls
        calls += 1
        assert original_publish(*args, **kwargs) is True
        return False

    monkeypatch.setattr(
        reclaim_module,
        "_atomic_rename_between_at_no_replace",
        concurrent_exact_publish,
    )
    published = reclaim_module._publish_restore_target(
        run_root=run_root,
        row=row,
        stage_root=stage_root,
        stage_name=stage_name,
    )

    assert published is False
    assert calls == 1
    assert (run_root / row["path"]).read_bytes() == payload
    assert not (stage_root / stage_name).exists()


def test_cleanup_atomic_retirement_preserves_last_moment_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    target = stage_root / "target.raw"
    replacement = stage_root / "replacement.raw"
    original_payload = b"verified-staging-payload"
    replacement_payload = b"attacker-replacement"
    target.write_bytes(original_payload)
    replacement.write_bytes(replacement_payload)
    target_state = target.stat(follow_symlinks=False)
    original_retire = reclaim_module._atomic_rename_at_no_replace
    replaced = False

    def replace_at_retirement(
        directory_fd: int,
        source_name: str,
        destination_name: str,
    ) -> bool:
        nonlocal replaced
        if source_name == "target.raw" and not replaced:
            replaced = True
            reclaim_module.os.replace(replacement, target)
        return original_retire(directory_fd, source_name, destination_name)

    monkeypatch.setattr(
        reclaim_module,
        "_atomic_rename_at_no_replace",
        replace_at_retirement,
    )
    with pytest.raises(DojoHistoricalRawReclaimError, match="retired and preserved"):
        reclaim_module._remove_anchored_verified_file(
            stage_root,
            "target.raw",
            expected_identity=reclaim_module._inode_identity(target_state),
            expected_sha256=hashlib.sha256(original_payload).hexdigest(),
            expected_size_bytes=len(original_payload),
        )

    assert replaced is True
    assert not target.exists()
    anchors = list(stage_root.glob(".retired-*.anchor"))
    assert len(anchors) == 1
    assert anchors[0].read_bytes() == replacement_payload


def test_cleanup_releases_blocks_into_durable_zero_anchor_without_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    target = stage_root / "target.raw"
    payload = b"release-payload-blocks" * (128 * 1024)
    target.write_bytes(payload)
    target_state = target.stat(follow_symlinks=False)

    def reject_unlink(*_args, **_kwargs) -> None:
        raise AssertionError("verified cleanup must never call unlink")

    def reject_hardlink(*_args, **_kwargs) -> None:
        raise AssertionError("verified cleanup must never create a hardlink")

    monkeypatch.setattr(reclaim_module.os, "unlink", reject_unlink)
    monkeypatch.setattr(reclaim_module.os, "link", reject_hardlink)
    allocated = reclaim_module._remove_anchored_verified_file(
        stage_root,
        "target.raw",
        expected_identity=reclaim_module._inode_identity(target_state),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size_bytes=len(payload),
    )

    assert allocated == target_state.st_blocks * 512
    assert not target.exists()
    anchors = list(stage_root.glob(".retired-*.anchor"))
    assert len(anchors) == 1
    assert anchors[0].stat().st_ino == target_state.st_ino
    assert anchors[0].stat().st_size == 0
    assert anchors[0].stat().st_blocks == 0


def test_raw_and_restore_cleanup_source_contains_no_unlink_call() -> None:
    functions = (
        reclaim_module._remove_anchored_verified_file,
        reclaim_module._retire_target,
        reclaim_module._open_stage_output,
        reclaim_module._cleanup_restore_staging,
    )
    for function in functions:
        tree = ast.parse(inspect.getsource(function))
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "unlink"
        ]
    module_source = inspect.getsource(reclaim_module)
    assert "os.unlink" not in module_source
    assert ".unlink(" not in module_source


def test_raw_unlink_and_drivefs_cache_eviction_are_separate_operations(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    drivefs_cache = tmp_path / "DriveFS-local-cache" / "archive.part"
    raw = run_root / "raw" / "target.jsonl"
    payload = b"allowlisted-run-root-raw"
    raw.parent.mkdir(parents=True)
    drivefs_cache.parent.mkdir(parents=True)
    raw.write_bytes(payload)
    drivefs_cache.write_bytes(b"cached-archive-object")
    row = {
        "path": "raw/target.jsonl",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }

    reclaim_module._retire_target(run_root, row)
    boundary = historical_raw_storage_boundary()

    assert not raw.exists()
    retired = list(raw.parent.glob(".retired-*.anchor"))
    assert len(retired) == 1
    assert retired[0].stat().st_size == 0
    assert drivefs_cache.read_bytes() == b"cached-archive-object"
    assert boundary["contract"] == RAW_STORAGE_BOUNDARY_CONTRACT
    assert boundary["raw_reclaim_scope"] == "EXACT_ALLOWLISTED_RUN_ROOT_FILES_ONLY"
    assert boundary["raw_reclaim_mechanism"] == reclaim_module.RECLAIM_MODE
    assert boundary["filesystem_unlink_used"] is False
    assert boundary["archive_object_deletion_allowed"] is False
    assert boundary["drivefs_cache_eviction_included"] is False
    assert (
        boundary["drivefs_cache_eviction_requires_separate_explicit_operation"] is True
    )
    assert boundary["drivefs_cache_eviction_implemented"] is False


def test_raw_retirement_preserves_replacement_at_atomic_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    raw = run_root / "raw" / "target.jsonl"
    replacement = run_root / "raw" / "replacement.jsonl"
    original_payload = b"verified-original-raw"
    replacement_payload = b"new-raw-must-not-be-deleted"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(original_payload)
    replacement.write_bytes(replacement_payload)
    row = {
        "path": "raw/target.jsonl",
        "sha256": hashlib.sha256(original_payload).hexdigest(),
        "size_bytes": len(original_payload),
    }
    original_retire = reclaim_module._atomic_rename_at_no_replace
    raced = False

    def replace_raw_at_retirement(
        directory_fd: int,
        source_name: str,
        destination_name: str,
    ) -> bool:
        nonlocal raced
        if source_name == "target.jsonl" and not raced:
            raced = True
            reclaim_module.os.replace(replacement, raw)
        return original_retire(directory_fd, source_name, destination_name)

    monkeypatch.setattr(
        reclaim_module,
        "_atomic_rename_at_no_replace",
        replace_raw_at_retirement,
    )
    with pytest.raises(DojoHistoricalRawReclaimError, match="retired and preserved"):
        reclaim_module._retire_target(run_root, row)

    assert raced is True
    assert not raw.exists()
    anchors = list(raw.parent.glob(".retired-*.anchor"))
    assert len(anchors) == 1
    assert anchors[0].read_bytes() == replacement_payload


def test_anchored_unlink_rejects_parent_symlink_swap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    (root / "parent").mkdir(parents=True)
    outside.mkdir()
    payload = b"same-hash-is-not-enough"
    original = root / "parent" / "target.raw"
    outside_target = outside / "target.raw"
    original.write_bytes(payload)
    outside_target.write_bytes(payload)
    row = {
        "path": "parent/target.raw",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    (root / "parent").rename(root / "parent-original")
    (root / "parent").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DojoHistoricalRawReclaimError, match="anchored parent"):
        reclaim_module._retire_target(root, row)

    assert outside_target.read_bytes() == payload
    assert (root / "parent-original" / "target.raw").read_bytes() == payload
