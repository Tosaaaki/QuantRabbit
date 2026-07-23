"""Durable, evidence-preserving recovery for an orphaned historical TRAIN claim.

The prepared room generation seals the economic runner and its custody control
plane.  This module deliberately does not rewrite those sealed files.  It is a
small outer recovery control plane that can only resume the one already-active
claim with the generation's sealed runner id.

An interrupted V1 economic transcript cannot be appended safely because the
canonical recorder creates its path with ``O_EXCL``.  Recovery therefore
publishes an immutable intent, renames every incomplete official artifact to a
reclaimable crash-evidence name, publishes an immutable completion receipt,
and replays the sealed source from the beginning under the same claim.  No
partial economics become a result, trainer input, proof, or promotion evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from quant_rabbit import dojo_historical_train_control as control_plane
from quant_rabbit.dojo_long_horizon_economic_runner import _verify_runner_handoff


RECOVERY_CONTRACT: Final = "QR_DOJO_HISTORICAL_CRASH_EVIDENCE_RECOVERY_V1"
RECOVERY_COMPLETION_CONTRACT: Final = (
    "QR_DOJO_HISTORICAL_CRASH_EVIDENCE_RECOVERY_COMPLETION_V1"
)
SCHEMA_VERSION: Final = 1
_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_INTENT_RE: Final = re.compile(r"crash-recovery-([0-9]{4})-([0-9a-f]{64})\.json\Z")
_COMPLETION_RE: Final = re.compile(
    r"crash-recovery-complete-([0-9a-f]{64})-([0-9a-f]{64})\.json\Z"
)
_QUARANTINE_TOKEN_RE: Final = re.compile(r"\.recovery-[0-9]{4}-[0-9a-f]{64}\.")
_TRANSCRIPT_RE: Final = re.compile(r"[0-9a-f]{64}\.economic\.jsonl\Z")
_TRANSCRIPT_ATTESTATION_RE: Final = re.compile(
    r"[0-9a-f]{64}\.economic\.attestation\.json\Z"
)

_AUTHORITY: Final = {
    "automatic_deployment_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
    "promotion_eligible": False,
}


class DojoHistoricalCrashRecoveryError(ValueError):
    """The active claim cannot be recovered without weakening evidence."""


def _module_binding() -> dict[str, Any]:
    path = Path(__file__).resolve(strict=True)
    state = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery implementation must be a single-link regular file"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    current = path.stat(follow_symlinks=False)
    if (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns) != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ):
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery implementation changed while hashed"
        )
    return {
        "relative_path": "src/quant_rabbit/dojo_historical_crash_supervisor.py",
        "size_bytes": state.st_size,
        "sha256": digest.hexdigest(),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_stable_file(path: Path) -> tuple[str, int]:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalCrashRecoveryError(
            f"crash evidence is unavailable: {path.name}"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 1
    ):
        raise DojoHistoricalCrashRecoveryError(
            f"crash evidence must be a nonempty single-link regular file: {path.name}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(descriptor, "rb", closefd=True) as handle:
        opened = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
        opened_after = os.fstat(handle.fileno())
    current = path.stat(follow_symlinks=False)
    identities = {
        (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_nlink)
        for row in (before, opened, opened_after, current)
    }
    if len(identities) != 1 or size != before.st_size:
        raise DojoHistoricalCrashRecoveryError(
            f"crash evidence changed while hashed: {path.name}"
        )
    return digest.hexdigest(), size


def _official_artifact_name(name: str, *, job_sha256: str, claim_sha256: str) -> bool:
    if _QUARANTINE_TOKEN_RE.search(name):
        return False
    if _TRANSCRIPT_RE.fullmatch(name) or _TRANSCRIPT_ATTESTATION_RE.fullmatch(name):
        return True
    prefix = f"{job_sha256}.{claim_sha256}."
    return name in {
        prefix + "fixed-denominator-attestation.json",
        prefix + "fixed-decision-request.json",
        prefix + "fixed-decision-attestation.json",
    }


def _quarantine_name(name: str, *, ordinal: int, inventory_sha256: str) -> str:
    token = f".recovery-{ordinal:04d}-{inventory_sha256}"
    if name.endswith(".economic.jsonl"):
        return name.removesuffix(".economic.jsonl") + token + ".economic.jsonl"
    if name.endswith(".economic.attestation.json"):
        return (
            name.removesuffix(".economic.attestation.json")
            + token
            + ".economic.attestation.json"
        )
    if not name.endswith(".json"):
        raise DojoHistoricalCrashRecoveryError(
            f"crash evidence filename is outside the allowlist: {name}"
        )
    return name.removesuffix(".json") + token + ".json"


def _read_receipt(path: Path, *, field: str) -> dict[str, Any]:
    try:
        return control_plane._read_json(path, field=field)
    except control_plane.DojoHistoricalTrainControlError as exc:
        raise DojoHistoricalCrashRecoveryError(str(exc)) from exc


def _validate_intent(
    path: Path,
    *,
    job_sha256: str,
    claim_sha256: str,
    runner_id: str,
    runner_handoff_sha256: str,
    control_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _read_receipt(path, field="crash recovery intent")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    digest = control_plane.canonical_sha256(body)
    match = _INTENT_RE.fullmatch(path.name)
    files = receipt.get("files")
    expected_keys = {
        "contract",
        "schema_version",
        "job_sha256",
        "claim_sha256",
        "attempt_ordinal",
        "runner_id",
        "runner_handoff_sha256",
        "control_manifest_sha256",
        "sealed_custody_control_plane_binding_sha256",
        "recovery_implementation",
        "recovery_ordinal",
        "reason",
        "restart_policy",
        "file_count",
        "total_bytes",
        "files",
        "files_sha256",
        "same_claim_required",
        "partial_economics_reported",
        "trainer_input_allowed",
        "historical_train_is_proof",
        "automatic_deployment_allowed",
        "broker_mutation_allowed",
        "live_permission",
        "order_authority",
        "promotion_eligible",
        "receipt_sha256",
    }
    if (
        match is None
        or set(receipt) != expected_keys
        or receipt.get("contract") != RECOVERY_CONTRACT
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("job_sha256") != job_sha256
        or receipt.get("claim_sha256") != claim_sha256
        or receipt.get("attempt_ordinal") != 1
        or receipt.get("runner_id") != runner_id
        or receipt.get("runner_handoff_sha256") != runner_handoff_sha256
        or receipt.get("control_manifest_sha256")
        != control_manifest.get("manifest_sha256")
        or receipt.get("sealed_custody_control_plane_binding_sha256")
        != control_manifest.get("custody_control_plane_binding", {}).get(
            "binding_sha256"
        )
        or receipt.get("recovery_implementation") != _module_binding()
        or receipt.get("recovery_ordinal") != int(match.group(1))
        or receipt.get("reason") != "ORPHANED_ACTIVE_CLAIM_AFTER_PROCESS_LEASE_RELEASE"
        or receipt.get("restart_policy")
        != "REPLAY_SEALED_SOURCE_FROM_GENESIS_WITH_SAME_CLAIM"
        or receipt.get("receipt_sha256") != digest
        or match.group(2) != digest
        or not isinstance(files, list)
        or not files
        or receipt.get("file_count") != len(files)
        or receipt.get("total_bytes")
        != sum(row.get("size_bytes", -1) for row in files if isinstance(row, Mapping))
        or receipt.get("files_sha256") != control_plane.canonical_sha256(files)
        or receipt.get("same_claim_required") is not True
        or receipt.get("partial_economics_reported") is not False
        or receipt.get("trainer_input_allowed") is not False
        or receipt.get("historical_train_is_proof") is not False
        or any(receipt.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoHistoricalCrashRecoveryError("crash recovery intent is invalid")
    seen_original: set[str] = set()
    seen_quarantine: set[str] = set()
    originals = []
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "original_name",
            "quarantine_name",
            "sha256",
            "size_bytes",
        }:
            raise DojoHistoricalCrashRecoveryError("crash recovery file row is invalid")
        original = row["original_name"]
        quarantine = row["quarantine_name"]
        if (
            not isinstance(original, str)
            or not isinstance(quarantine, str)
            or "/" in original
            or "/" in quarantine
            or original in seen_original
            or quarantine in seen_quarantine
            or not _official_artifact_name(
                original, job_sha256=job_sha256, claim_sha256=claim_sha256
            )
            or _QUARANTINE_TOKEN_RE.search(quarantine) is None
            or not isinstance(row["sha256"], str)
            or _SHA_RE.fullmatch(row["sha256"]) is None
            or isinstance(row["size_bytes"], bool)
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 1
        ):
            raise DojoHistoricalCrashRecoveryError(
                "crash recovery file mapping is invalid"
            )
        seen_original.add(original)
        seen_quarantine.add(quarantine)
        originals.append(
            {
                "original_name": original,
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
        )
    if [row["original_name"] for row in originals] != sorted(seen_original):
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery inventory order is not canonical"
        )
    inventory_sha256 = control_plane.canonical_sha256(originals)
    for row in files:
        if row["quarantine_name"] != _quarantine_name(
            row["original_name"],
            ordinal=receipt["recovery_ordinal"],
            inventory_sha256=inventory_sha256,
        ):
            raise DojoHistoricalCrashRecoveryError(
                "crash recovery quarantine name is not content-bound"
            )
    return receipt


def _completion_for_intent(evidence_root: Path, receipt_sha256: str) -> Path | None:
    matches = sorted(
        evidence_root.glob(f"crash-recovery-complete-{receipt_sha256}-*.json")
    )
    if len(matches) > 1:
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery intent has multiple completion receipts"
        )
    return matches[0] if matches else None


def _validate_completion(path: Path, *, intent: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _read_receipt(path, field="crash recovery completion")
    body = {key: value for key, value in receipt.items() if key != "completion_sha256"}
    digest = control_plane.canonical_sha256(body)
    match = _COMPLETION_RE.fullmatch(path.name)
    expected_keys = {
        "contract",
        "schema_version",
        "job_sha256",
        "claim_sha256",
        "recovery_ordinal",
        "recovery_receipt_sha256",
        "moved_file_count",
        "moved_total_bytes",
        "quarantine_files_sha256",
        "same_claim_preserved",
        "partial_economics_reported",
        "trainer_input_allowed",
        "automatic_deployment_allowed",
        "broker_mutation_allowed",
        "live_permission",
        "order_authority",
        "promotion_eligible",
        "completion_sha256",
    }
    if (
        match is None
        or set(receipt) != expected_keys
        or receipt.get("contract") != RECOVERY_COMPLETION_CONTRACT
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("job_sha256") != intent["job_sha256"]
        or receipt.get("claim_sha256") != intent["claim_sha256"]
        or receipt.get("recovery_ordinal") != intent["recovery_ordinal"]
        or receipt.get("recovery_receipt_sha256") != intent["receipt_sha256"]
        or match.group(1) != intent["receipt_sha256"]
        or receipt.get("moved_file_count") != intent["file_count"]
        or receipt.get("moved_total_bytes") != intent["total_bytes"]
        or receipt.get("quarantine_files_sha256")
        != control_plane.canonical_sha256(
            [row["quarantine_name"] for row in intent["files"]]
        )
        or receipt.get("same_claim_preserved") is not True
        or receipt.get("partial_economics_reported") is not False
        or receipt.get("trainer_input_allowed") is not False
        or receipt.get("completion_sha256") != digest
        or match.group(2) != digest
        or any(receipt.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery completion receipt is invalid"
        )
    return receipt


def _verify_quarantined_files(
    evidence_root: Path, *, intent: Mapping[str, Any]
) -> None:
    for row in intent["files"]:
        path = evidence_root / row["quarantine_name"]
        digest, size = _hash_stable_file(path)
        if digest != row["sha256"] or size != row["size_bytes"]:
            raise DojoHistoricalCrashRecoveryError(
                f"quarantined crash evidence drifted: {path.name}"
            )


def _complete_intent(
    evidence_root: Path,
    *,
    intent_path: Path,
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    existing_completion = _completion_for_intent(
        evidence_root, intent["receipt_sha256"]
    )
    if existing_completion is not None:
        completion = _validate_completion(existing_completion, intent=intent)
        _verify_quarantined_files(evidence_root, intent=intent)
        return completion
    for row in intent["files"]:
        original = evidence_root / row["original_name"]
        quarantine = evidence_root / row["quarantine_name"]
        original_exists = original.exists()
        quarantine_exists = quarantine.exists()
        if original_exists == quarantine_exists:
            raise DojoHistoricalCrashRecoveryError(
                "pending crash recovery mapping is ambiguous: " + row["original_name"]
            )
        if original_exists:
            digest, size = _hash_stable_file(original)
            if digest != row["sha256"] or size != row["size_bytes"]:
                raise DojoHistoricalCrashRecoveryError(
                    f"pending crash evidence drifted: {original.name}"
                )
            original.rename(quarantine)
            _fsync_directory(evidence_root)
        digest, size = _hash_stable_file(quarantine)
        if digest != row["sha256"] or size != row["size_bytes"]:
            raise DojoHistoricalCrashRecoveryError(
                f"quarantined crash evidence drifted: {quarantine.name}"
            )
    body = {
        "contract": RECOVERY_COMPLETION_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": intent["job_sha256"],
        "claim_sha256": intent["claim_sha256"],
        "recovery_ordinal": intent["recovery_ordinal"],
        "recovery_receipt_sha256": intent["receipt_sha256"],
        "moved_file_count": intent["file_count"],
        "moved_total_bytes": intent["total_bytes"],
        "quarantine_files_sha256": control_plane.canonical_sha256(
            [row["quarantine_name"] for row in intent["files"]]
        ),
        "same_claim_preserved": True,
        "partial_economics_reported": False,
        "trainer_input_allowed": False,
        **_AUTHORITY,
    }
    completion = {
        **body,
        "completion_sha256": control_plane.canonical_sha256(body),
    }
    completion_path = evidence_root / (
        "crash-recovery-complete-"
        f"{intent['receipt_sha256']}-{completion['completion_sha256']}.json"
    )
    control_plane._write_once(completion_path, completion)
    _fsync_directory(evidence_root)
    return completion


def quarantine_orphaned_evidence(
    *,
    evidence_root: Path,
    handoff: Mapping[str, Any],
    control_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Retain incomplete official files and free their deterministic names."""

    job = handoff["job"]
    claim = handoff["claim"]
    job_sha256 = job["job_sha256"]
    claim_sha256 = claim["claim_sha256"]
    if claim.get("attempt_ordinal") != 1:
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery only supports the sealed single attempt"
        )
    try:
        root_state = evidence_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalCrashRecoveryError(
            "economic evidence root is unavailable"
        ) from exc
    if evidence_root.is_symlink() or not stat.S_ISDIR(root_state.st_mode):
        raise DojoHistoricalCrashRecoveryError(
            "economic evidence root must be a real directory"
        )

    intents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(evidence_root.glob("crash-recovery-*.json")):
        if path.name.startswith("crash-recovery-complete-"):
            continue
        intent = _validate_intent(
            path,
            job_sha256=job_sha256,
            claim_sha256=claim_sha256,
            runner_id=claim["runner_id"],
            runner_handoff_sha256=handoff["runner_handoff_sha256"],
            control_manifest=control_manifest,
        )
        intents.append((path, intent))
    ordinals = [intent["recovery_ordinal"] for _, intent in intents]
    if ordinals != list(range(1, len(ordinals) + 1)):
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery ordinals are not contiguous"
        )
    completed_pending: dict[str, Any] | None = None
    for path, intent in intents:
        completion_path = _completion_for_intent(
            evidence_root, intent["receipt_sha256"]
        )
        if completion_path is None:
            if completed_pending is not None:
                raise DojoHistoricalCrashRecoveryError(
                    "multiple pending crash recovery intents exist"
                )
            completed_pending = _complete_intent(
                evidence_root, intent_path=path, intent=intent
            )
        else:
            _validate_completion(completion_path, intent=intent)
            _verify_quarantined_files(evidence_root, intent=intent)
    if completed_pending is not None:
        return completed_pending

    known_names = {
        path.name
        for path, intent in intents
        for path in (
            path,
            _completion_for_intent(evidence_root, intent["receipt_sha256"]),
        )
        if path is not None
    }
    known_names.update(
        row["quarantine_name"] for _, intent in intents for row in intent["files"]
    )
    official_paths: list[Path] = []
    for path in sorted(evidence_root.iterdir()):
        state = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
            raise DojoHistoricalCrashRecoveryError(
                f"economic evidence contains an unsafe entry: {path.name}"
            )
        if path.name in known_names:
            continue
        if not _official_artifact_name(
            path.name, job_sha256=job_sha256, claim_sha256=claim_sha256
        ):
            raise DojoHistoricalCrashRecoveryError(
                f"economic evidence contains an unknown file: {path.name}"
            )
        official_paths.append(path)
    if not official_paths:
        return None

    inventory = []
    for path in official_paths:
        digest, size = _hash_stable_file(path)
        inventory.append(
            {"original_name": path.name, "sha256": digest, "size_bytes": size}
        )
    inventory_sha256 = control_plane.canonical_sha256(inventory)
    ordinal = len(intents) + 1
    files = [
        {
            **row,
            "quarantine_name": _quarantine_name(
                row["original_name"],
                ordinal=ordinal,
                inventory_sha256=inventory_sha256,
            ),
        }
        for row in inventory
    ]
    if len({row["quarantine_name"] for row in files}) != len(files):
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery quarantine names collide"
        )
    body = {
        "contract": RECOVERY_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "job_sha256": job_sha256,
        "claim_sha256": claim_sha256,
        "attempt_ordinal": claim["attempt_ordinal"],
        "runner_id": claim["runner_id"],
        "runner_handoff_sha256": handoff["runner_handoff_sha256"],
        "control_manifest_sha256": control_manifest["manifest_sha256"],
        "sealed_custody_control_plane_binding_sha256": control_manifest[
            "custody_control_plane_binding"
        ]["binding_sha256"],
        "recovery_implementation": _module_binding(),
        "recovery_ordinal": ordinal,
        "reason": "ORPHANED_ACTIVE_CLAIM_AFTER_PROCESS_LEASE_RELEASE",
        "restart_policy": "REPLAY_SEALED_SOURCE_FROM_GENESIS_WITH_SAME_CLAIM",
        "file_count": len(files),
        "total_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
        "files_sha256": control_plane.canonical_sha256(files),
        "same_claim_required": True,
        "partial_economics_reported": False,
        "trainer_input_allowed": False,
        "historical_train_is_proof": False,
        **_AUTHORITY,
    }
    intent = {**body, "receipt_sha256": control_plane.canonical_sha256(body)}
    intent_path = evidence_root / (
        f"crash-recovery-{ordinal:04d}-{intent['receipt_sha256']}.json"
    )
    control_plane._write_once(intent_path, intent)
    _fsync_directory(evidence_root)
    completion = _complete_intent(evidence_root, intent_path=intent_path, intent=intent)
    return completion


def _verified_source_failure(
    path: Path, *, job_sha256: str, claim_sha256: str
) -> dict[str, Any]:
    evidence = control_plane._read_json(path, field="historical source failure")
    body = {
        key: value
        for key, value in evidence.items()
        if key != "failure_evidence_sha256"
    }
    if (
        evidence.get("contract") != "QR_DOJO_HISTORICAL_SOURCE_FAILURE_V1"
        or evidence.get("schema_version") != control_plane.SCHEMA_VERSION
        or evidence.get("job_sha256") != job_sha256
        or evidence.get("claim_sha256") != claim_sha256
        or evidence.get("stage") != "SPARSE_SOURCE_MATERIALIZATION"
        or not isinstance(evidence.get("error_type"), str)
        or not evidence["error_type"]
        or not isinstance(evidence.get("error"), str)
        or not evidence["error"]
        or len(evidence["error"]) > 4096
        or evidence.get("failure_evidence_sha256")
        != control_plane.canonical_sha256(body)
        or any(evidence.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoHistoricalCrashRecoveryError(
            "historical source failure evidence is invalid"
        )
    return evidence


def _publish_source_failure_completion(
    *,
    root: Path,
    schedule: Mapping[str, Any],
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
    evidence: Mapping[str, Any],
    mutation_guard: Any,
) -> dict[str, Any]:
    job = handoff["job"]
    claim = handoff["claim"]
    cells = [
        control_plane.build_long_horizon_coordinate_result(
            job=job,
            claim=claim,
            coordinate_id=coordinate_id,
            status="FAILED",
            failure={
                "code": "SOURCE_SLICE_MATERIALIZATION_FAILURE",
                "retryable": False,
                "evidence_sha256": evidence["failure_evidence_sha256"],
            },
        )
        for coordinate_id in handoff["runnable_coordinate_ids"]
    ]
    if cells:
        mutation_guard()
        control_plane.record_long_horizon_coordinate_results(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            claim_sha256=claim["claim_sha256"],
            results=cells,
        )
    mutation_guard()
    terminal = control_plane.seal_long_horizon_attempt(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
    )
    terminal_cells = terminal["terminal_manifest"]["cells"]
    source_failure_count = sum(
        row["status"] == "FAILED"
        and row["failure"]["code"] == "SOURCE_SLICE_MATERIALIZATION_FAILURE"
        and row["failure"]["evidence_sha256"] == evidence["failure_evidence_sha256"]
        for row in terminal_cells
    )
    predecessor_failure_count = sum(
        row["status"] == "FAILED" and row["failure"]["code"] == "PREDECESSOR_FAILED"
        for row in terminal_cells
    )
    if source_failure_count + predecessor_failure_count != len(terminal_cells):
        raise DojoHistoricalCrashRecoveryError(
            "source-failure terminal mixes an unrelated coordinate result"
        )
    complete_coordinate_count = sum(
        row["status"] == "COMPLETE" for row in terminal_cells
    )
    failed_coordinate_count = len(terminal_cells) - complete_coordinate_count
    completion_body = {
        "contract": control_plane.JOB_COMPLETION_CONTRACT,
        "schema_version": control_plane.SCHEMA_VERSION,
        "job_sha256": job["job_sha256"],
        "claim_sha256": claim["claim_sha256"],
        "source_binding_id": job["source_binding_id"],
        "month": job["month"],
        "intrabar_path": job["intrabar_path"],
        "coordinate_result_count": len(terminal_cells),
        "new_source_failure_coordinate_count": source_failure_count,
        "predecessor_failure_coordinate_count": predecessor_failure_count,
        "complete_coordinate_count": complete_coordinate_count,
        "failed_coordinate_count": failed_coordinate_count,
        "job_status": "FAILED_SOURCE",
        "economic_job_result_sha256": None,
        "terminal_sha256": terminal["terminal_manifest"]["terminal_sha256"],
        "free_disk_bytes_after": shutil.disk_usage(root).free,
        **control_plane._AUTHORITY,
    }
    completion = {
        **completion_body,
        "completion_sha256": control_plane.canonical_sha256(completion_body),
    }
    mutation_guard()
    control_plane._write_once(
        root / "jobs" / job["job_sha256"] / "completion.json", completion
    )
    mutation_guard()
    return {
        "status": "JOB_FAILED_SOURCE_AFTER_SAME_CLAIM_RECOVERY",
        "output_root": str(root),
        "job": {
            "job_sha256": job["job_sha256"],
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
        },
        "claim_sha256": claim["claim_sha256"],
        "same_claim_recovered": True,
        "coordinate_result_count": len(terminal_cells),
        "new_source_failure_coordinate_count": source_failure_count,
        "predecessor_failure_coordinate_count": predecessor_failure_count,
        "complete_coordinate_count": complete_coordinate_count,
        "failed_coordinate_count": failed_coordinate_count,
        "failure_evidence_sha256": evidence["failure_evidence_sha256"],
        "trainer_milestone": control_plane._milestone_status(root, schedule),
        **_AUTHORITY,
    }


def _resume_active_job_locked(
    *,
    repo_root: Path,
    root: Path,
    control: Mapping[str, Any],
    plan: Mapping[str, Any],
    schedule: Mapping[str, Any],
    runtime_seal: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    catalog_wrapper: Mapping[str, Any],
    mutation_guard: Any,
    lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the sealed controller tail for exactly the already-active claim."""

    if (
        lifecycle["state"] != "RUNNING"
        or lifecycle["execution"]["active_job_count"] != 1
    ):
        raise DojoHistoricalCrashRecoveryError(
            "crash recovery requires exactly one active historical claim"
        )
    mutation_guard()
    current_handoff = control_plane.claim_next_long_horizon_job(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        runner_id=control_plane._runner_id(control["fixed_inputs"]),
    )
    job = current_handoff["job"]
    claim = current_handoff["claim"]
    job_sha = job["job_sha256"]
    if job_sha not in {
        row["job_sha256"]
        for row in lifecycle["job_states"]
        if row["state"] == "RUNNING"
    }:
        raise DojoHistoricalCrashRecoveryError(
            "resumed handoff is not the lifecycle's active job"
        )
    job_dir = root / "jobs" / job_sha
    mutation_guard()
    job_dir.mkdir(parents=True, exist_ok=True)
    stored_handoff_path = job_dir / "runner-handoff.json"
    if stored_handoff_path.exists():
        initial_handoff = _verify_runner_handoff(
            control_plane._read_json(
                stored_handoff_path, field="initial runner handoff"
            )
        )
        if (
            initial_handoff["job"]["job_sha256"] != job_sha
            or initial_handoff["claim"]["claim_sha256"] != claim["claim_sha256"]
        ):
            raise DojoHistoricalCrashRecoveryError(
                "initial runner handoff names another active claim"
            )
    else:
        initial_handoff = current_handoff
        mutation_guard()
        control_plane._write_once(stored_handoff_path, initial_handoff)
    result_path = root / "job-results" / f"{job_sha}.json"
    source_failure_path = job_dir / "source-failure.json"
    if source_failure_path.exists():
        if result_path.exists():
            raise DojoHistoricalCrashRecoveryError(
                "active claim mixes source failure and economic result evidence"
            )
        evidence = _verified_source_failure(
            source_failure_path,
            job_sha256=job_sha,
            claim_sha256=claim["claim_sha256"],
        )
        return _publish_source_failure_completion(
            root=root,
            schedule=schedule,
            plan=plan,
            handoff=current_handoff,
            evidence=evidence,
            mutation_guard=mutation_guard,
        )
    if (
        lifecycle["execution"].get("active_recorded_coordinate_count")
        and not result_path.exists()
    ):
        raise DojoHistoricalCrashRecoveryError(
            "active claim has coordinate cells without a recoverable job result"
        )
    source_root = root / "source-slices"
    relative = Path(
        job["source_binding_id"],
        job["month"],
        f"{job['intrabar_path']}-{job_sha}.jsonl",
    )
    receipt_path = job_dir / "source-slice-receipt.json"
    if not receipt_path.exists():
        raise DojoHistoricalCrashRecoveryError(
            "active claim has no immutable source-slice receipt"
        )
    receipt = control_plane.validate_sparse_month_source_slice_receipt(
        control_plane._read_json(receipt_path, field="source slice receipt"),
        job=job,
        source_manifest=source_manifest,
    )
    if receipt["relative_path"] != relative.as_posix():
        raise DojoHistoricalCrashRecoveryError("active claim source-slice path drifted")
    repo = Path(repo_root).resolve(strict=True)
    registry = control_plane._read_json(
        repo / control["fixed_inputs"]["registry_path"], field="G2 registry"
    )
    catalog = catalog_wrapper.get("worker_catalog")
    if not isinstance(catalog, list):
        raise DojoHistoricalCrashRecoveryError("worker catalog is invalid")
    runtimes = control_plane._coordinate_runtimes(
        handoff=initial_handoff,
        plan=plan,
        catalog=catalog,
        registry=registry,
        control=control,
    )
    mutation_guard()
    control_plane._write_once(
        job_dir / "coordinate-runtimes.json", {"coordinate_runtimes": runtimes}
    )
    carries = control_plane._carry_inputs(root, initial_handoff)
    if carries:
        mutation_guard()
        control_plane._write_once(
            job_dir / "carry-states-input.json",
            {"economic_carry_states_by_slot": carries},
        )
    evidence_root = job_dir / "economic-evidence"
    mutation_guard()
    evidence_root.mkdir(parents=True, exist_ok=True)
    recovery: dict[str, Any] | None = None
    if result_path.exists():
        result = control_plane._verified_job_result(
            control_plane._read_json(result_path, field="economic job result"),
            handoff=initial_handoff,
        )
    else:
        manifest = control_plane._read_json(
            root / "control-manifest.json", field="control manifest"
        )
        mutation_guard()
        recovery = quarantine_orphaned_evidence(
            evidence_root=evidence_root,
            handoff=initial_handoff,
            control_manifest=manifest,
        )
        runtime_factory = control_plane.build_tuned_strategy_runtime_factory(
            runtime_seal, repo_root=repo
        )
        mutation_guard()
        result = control_plane.run_long_horizon_economic_job(
            runner_handoff=initial_handoff,
            plan=plan,
            source_root=source_root,
            source_manifest=source_manifest,
            source_slice_receipt=receipt,
            economic_evidence_root=evidence_root,
            worker_catalog=catalog,
            coordinate_runtimes=runtimes,
            worker_runtime_factory=runtime_factory,
            worker_runtime_binding_sha256=runtime_seal["runtime_binding_sha256"],
            worker_runtime_seal=runtime_seal,
            worker_runtime_repo_root=repo,
            carry_states_by_slot=carries,
        )
        result = control_plane._verified_job_result(result, handoff=initial_handoff)
        mutation_guard()
        control_plane._write_once(result_path, result)
    mutation_guard()
    control_plane._publish_carries(root, result)
    mutation_guard()
    control_plane.record_long_horizon_coordinate_results(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
        results=result["coordinate_results"],
    )
    mutation_guard()
    terminal = control_plane.seal_long_horizon_attempt(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        claim_sha256=claim["claim_sha256"],
    )
    completion_body = {
        "contract": control_plane.JOB_COMPLETION_CONTRACT,
        "schema_version": control_plane.SCHEMA_VERSION,
        "job_sha256": job_sha,
        "claim_sha256": claim["claim_sha256"],
        "source_binding_id": job["source_binding_id"],
        "month": job["month"],
        "intrabar_path": job["intrabar_path"],
        "coordinate_result_count": result["coordinate_result_count"],
        "complete_coordinate_count": result["complete_coordinate_count"],
        "failed_coordinate_count": result["failed_coordinate_count"],
        "job_status": result["job_status"],
        "economic_job_result_sha256": result["economic_job_result_sha256"],
        "terminal_sha256": terminal["terminal_manifest"]["terminal_sha256"],
        "free_disk_bytes_after": shutil.disk_usage(root).free,
        **control_plane._AUTHORITY,
    }
    completion = {
        **completion_body,
        "completion_sha256": control_plane.canonical_sha256(completion_body),
    }
    mutation_guard()
    control_plane._write_once(job_dir / "completion.json", completion)
    mutation_guard()
    milestone = control_plane._milestone_status(root, schedule)
    return {
        "status": "JOB_COMPLETE_AFTER_SAME_CLAIM_RECOVERY",
        "output_root": str(root),
        "job": {
            "job_sha256": job_sha,
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
        },
        "claim_sha256": claim["claim_sha256"],
        "same_claim_recovered": True,
        "crash_evidence_recovery": recovery,
        "coordinate_result_count": result["coordinate_result_count"],
        "complete_coordinate_count": result["complete_coordinate_count"],
        "failed_coordinate_count": result["failed_coordinate_count"],
        "archive": None,
        "next_transition": "ARCHIVE_NEXT",
        "trainer_milestone": milestone,
        **_AUTHORITY,
    }


def resume_active_historical_job(
    *, repo_root: Path, run_control_path: Path
) -> dict[str, Any]:
    """Recover and execute exactly one orphaned active claim under all leases."""

    prelock_generation = control_plane._load_generation(
        repo_root=repo_root, run_control_path=run_control_path
    )
    control, root, _, _, _, _, _ = prelock_generation
    (
        run_lock_descriptor,
        global_lock_descriptor,
        conflicting_lock_descriptors,
    ) = control_plane._acquire_historical_operation_locks(root=root, control=control)
    try:
        locked_generation = control_plane._reload_generation_under_lock(
            repo_root=repo_root,
            run_control_path=run_control_path,
            prelock_generation=prelock_generation,
        )
        (
            control,
            locked_root,
            plan,
            schedule,
            runtime_seal,
            source_manifest,
            catalog_wrapper,
        ) = locked_generation
        if locked_root != root:
            raise DojoHistoricalCrashRecoveryError(
                "recovery generation changed after operation locks were acquired"
            )
        lifecycle = control_plane.evaluate_historical_lifecycle(
            root=root, control=control, plan=plan, schedule=schedule
        )
        if lifecycle["state"] != "RUNNING":
            raise DojoHistoricalCrashRecoveryError(
                "active-claim recovery is no longer the selected state"
            )
        control_plane._assert_dynamic_machine_capacity(
            control,
            current_root=root,
            locked_descriptors=conflicting_lock_descriptors,
        )
        control_plane._deep_verify_completed_job_custody(root=root, control=control)
        planned_coordinate_count = max(
            job["coordinate_count"] for job in schedule["jobs"]
        )
        (
            estimated_raw_bytes,
            estimated_archive_upper_bytes,
            estimated_peak_bytes,
        ) = control_plane._estimated_next_job_bytes(
            control=control,
            archive_root=control_plane._archive_root(control),
            planned_coordinate_count=planned_coordinate_count,
        )
        control_plane._assert_disk_capacity(
            control_plane._disk_capacity_snapshot(
                root=root,
                control=control,
                estimated_raw_bytes=estimated_raw_bytes,
                estimated_archive_upper_bytes=estimated_archive_upper_bytes,
                estimated_peak_bytes=estimated_peak_bytes,
            )
        )

        def mutation_guard() -> None:
            control_plane._assert_historical_operation_lock_identities(
                run_lock_descriptor=run_lock_descriptor,
                global_lock_descriptor=global_lock_descriptor,
                conflicting_lock_descriptors=conflicting_lock_descriptors,
            )

        return _resume_active_job_locked(
            repo_root=repo_root,
            root=root,
            control=control,
            plan=plan,
            schedule=schedule,
            runtime_seal=runtime_seal,
            source_manifest=source_manifest,
            catalog_wrapper=catalog_wrapper,
            mutation_guard=mutation_guard,
            lifecycle=lifecycle,
        )
    finally:
        control_plane._release_historical_operation_locks(
            run_lock_descriptor=run_lock_descriptor,
            global_lock_descriptor=global_lock_descriptor,
            conflicting_lock_descriptors=conflicting_lock_descriptors,
        )


def publish_unpublished_terminal_completion(
    *, repo_root: Path, run_control_path: Path
) -> dict[str, Any]:
    """Finish the sole completion publication after a durable terminal crash."""

    prelock_generation = control_plane._load_generation(
        repo_root=repo_root, run_control_path=run_control_path
    )
    control, root, _, _, _, _, _ = prelock_generation
    (
        run_lock_descriptor,
        global_lock_descriptor,
        conflicting_lock_descriptors,
    ) = control_plane._acquire_historical_operation_locks(root=root, control=control)
    try:
        locked_generation = control_plane._reload_generation_under_lock(
            repo_root=repo_root,
            run_control_path=run_control_path,
            prelock_generation=prelock_generation,
        )
        control, locked_root, plan, schedule, _, _, _ = locked_generation
        if locked_root != root:
            raise DojoHistoricalCrashRecoveryError(
                "terminal recovery generation changed after locks were acquired"
            )
        lifecycle = control_plane.evaluate_historical_lifecycle(
            root=root, control=control, plan=plan, schedule=schedule
        )
        unpublished = [
            row
            for row in lifecycle["job_states"]
            if row["state"] == "TERMINAL_UNPUBLISHED"
        ]
        if lifecycle["state"] != "TERMINAL_UNPUBLISHED" or len(unpublished) != 1:
            raise DojoHistoricalCrashRecoveryError(
                "terminal recovery requires exactly one unpublished terminal"
            )
        job_sha = unpublished[0]["job_sha256"]
        jobs = {job["job_sha256"]: job for job in schedule["jobs"]}
        job = jobs.get(job_sha)
        if job is None:
            raise DojoHistoricalCrashRecoveryError(
                "unpublished terminal job is absent from the sealed schedule"
            )

        def mutation_guard() -> None:
            control_plane._assert_historical_operation_lock_identities(
                run_lock_descriptor=run_lock_descriptor,
                global_lock_descriptor=global_lock_descriptor,
                conflicting_lock_descriptors=conflicting_lock_descriptors,
            )

        terminal_paths = sorted(
            (root / "execution-state" / "terminals" / job_sha).glob("attempt-*.json")
        )
        if len(terminal_paths) != 1:
            raise DojoHistoricalCrashRecoveryError(
                "unpublished job does not have exactly one terminal manifest"
            )
        raw_terminal = control_plane._read_json(
            terminal_paths[0], field="unpublished terminal manifest"
        )
        raw_claim = raw_terminal.get("claim")
        if not isinstance(raw_claim, Mapping) or not isinstance(
            raw_claim.get("claim_sha256"), str
        ):
            raise DojoHistoricalCrashRecoveryError(
                "unpublished terminal claim binding is invalid"
            )
        claim_sha = raw_claim["claim_sha256"]
        mutation_guard()
        sealed = control_plane.seal_long_horizon_attempt(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            claim_sha256=claim_sha,
        )["terminal_manifest"]
        if sealed != raw_terminal:
            raise DojoHistoricalCrashRecoveryError(
                "unpublished terminal differs from its sealed cell denominator"
            )
        job_dir = root / "jobs" / job_sha
        source_failure_path = job_dir / "source-failure.json"
        result_path = root / "job-results" / f"{job_sha}.json"
        if source_failure_path.exists():
            if result_path.exists():
                raise DojoHistoricalCrashRecoveryError(
                    "unpublished terminal mixes source failure and economic result"
                )
            evidence = _verified_source_failure(
                source_failure_path,
                job_sha256=job_sha,
                claim_sha256=claim_sha,
            )
            return _publish_source_failure_completion(
                root=root,
                schedule=schedule,
                plan=plan,
                handoff={
                    "job": job,
                    "claim": raw_claim,
                    "runnable_coordinate_ids": [],
                },
                evidence=evidence,
                mutation_guard=mutation_guard,
            )
        if not result_path.exists():
            raise DojoHistoricalCrashRecoveryError(
                "unpublished economic terminal has no immutable job result"
            )
        initial_handoff = _verify_runner_handoff(
            control_plane._read_json(
                job_dir / "runner-handoff.json", field="initial runner handoff"
            )
        )
        result = control_plane._verified_job_result(
            control_plane._read_json(result_path, field="economic job result"),
            handoff=initial_handoff,
        )
        if (
            initial_handoff["job"]["job_sha256"] != job_sha
            or initial_handoff["claim"]["claim_sha256"] != claim_sha
            or result["coordinate_results"] != sealed["cells"]
            or result["coordinate_result_count"] != sealed["coordinate_count"]
            or result["complete_coordinate_count"]
            != sealed["complete_coordinate_count"]
            or result["failed_coordinate_count"] != sealed["failed_coordinate_count"]
            or (result["job_status"] == "COMPLETE")
            != (sealed["terminal_status"] == "COMPLETE")
        ):
            raise DojoHistoricalCrashRecoveryError(
                "unpublished terminal differs from its immutable economic result"
            )
        mutation_guard()
        control_plane._publish_carries(root, result)
        completion_body = {
            "contract": control_plane.JOB_COMPLETION_CONTRACT,
            "schema_version": control_plane.SCHEMA_VERSION,
            "job_sha256": job_sha,
            "claim_sha256": claim_sha,
            "source_binding_id": job["source_binding_id"],
            "month": job["month"],
            "intrabar_path": job["intrabar_path"],
            "coordinate_result_count": result["coordinate_result_count"],
            "complete_coordinate_count": result["complete_coordinate_count"],
            "failed_coordinate_count": result["failed_coordinate_count"],
            "job_status": result["job_status"],
            "economic_job_result_sha256": result["economic_job_result_sha256"],
            "terminal_sha256": sealed["terminal_sha256"],
            "free_disk_bytes_after": shutil.disk_usage(root).free,
            **control_plane._AUTHORITY,
        }
        completion = {
            **completion_body,
            "completion_sha256": control_plane.canonical_sha256(completion_body),
        }
        mutation_guard()
        control_plane._write_once(job_dir / "completion.json", completion)
        mutation_guard()
        return {
            "status": "TERMINAL_COMPLETION_RECOVERED",
            "output_root": str(root),
            "job": {
                "job_sha256": job_sha,
                "source_binding_id": job["source_binding_id"],
                "month": job["month"],
                "intrabar_path": job["intrabar_path"],
            },
            "claim_sha256": claim_sha,
            "same_claim_recovered": True,
            "coordinate_result_count": result["coordinate_result_count"],
            "complete_coordinate_count": result["complete_coordinate_count"],
            "failed_coordinate_count": result["failed_coordinate_count"],
            "archive": None,
            "next_transition": "ARCHIVE_NEXT",
            "trainer_milestone": control_plane._milestone_status(root, schedule),
            **_AUTHORITY,
        }
    finally:
        control_plane._release_historical_operation_locks(
            run_lock_descriptor=run_lock_descriptor,
            global_lock_descriptor=global_lock_descriptor,
            conflicting_lock_descriptors=conflicting_lock_descriptors,
        )


def advance_one_supervised_transition(
    *, repo_root: Path, run_control_path: Path
) -> dict[str, Any]:
    """Advance one transition, including an orphaned active-claim recovery."""

    control, root, plan, schedule, _, _, _ = control_plane._load_generation(
        repo_root=repo_root, run_control_path=run_control_path, operation="custody"
    )
    lifecycle = control_plane.evaluate_historical_lifecycle(
        root=root, control=control, plan=plan, schedule=schedule
    )
    if lifecycle["state"] == "TERMINAL_UNPUBLISHED":
        operation = publish_unpublished_terminal_completion(
            repo_root=repo_root, run_control_path=run_control_path
        )
        selected_transition = "PUBLISH_TERMINAL_COMPLETION"
    elif lifecycle["state"] == "RUNNING":
        operation = resume_active_historical_job(
            repo_root=repo_root, run_control_path=run_control_path
        )
        selected_transition = "RECOVER_ACTIVE_JOB"
    else:
        return control_plane.advance_one_historical_transition(
            repo_root=repo_root, run_control_path=run_control_path
        )
    return {
        **operation,
        "heartbeat_step": {
            "selected_transition": selected_transition,
            "transition_execution_count": 1,
            "fallthrough_allowed": False,
            "lifecycle_before_sha256": lifecycle["lifecycle_sha256"],
            "operation_revalidated_under_lock": True,
        },
    }


__all__ = [
    "DojoHistoricalCrashRecoveryError",
    "RECOVERY_COMPLETION_CONTRACT",
    "RECOVERY_CONTRACT",
    "advance_one_supervised_transition",
    "publish_unpublished_terminal_completion",
    "quarantine_orphaned_evidence",
    "resume_active_historical_job",
]
