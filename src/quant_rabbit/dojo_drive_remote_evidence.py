"""Typed, fail-closed consistency checks for claimed Drive archive metadata.

The archive automation is intentionally the only component which talks to
Google Drive.  This pure module checks caller-supplied canonical JSON against a
verified terminal handoff receipt, its candidate-lineage tip, and local archive
plan/finalization receipts.  It performs no I/O and therefore cannot attest
that any bytes or metadata actually came from Google Drive.

An actual Google Drive connector/API fetch plus independent readback gate is
still mandatory.  Public metadata-consistency outputs keep
``remote_verified=false`` and ``external_readback_attested=false`` and cannot
unblock the AI trainer or the conveyor resource gate.  Requiring canonical
bytes merely prevents one local metadata claim from having multiple textual
representations.

The private trainer-readback boundary in this module exists only for an
authenticated connector adapter to call with metadata and download bytes it
fetched in the same process.  No JSON/file CLI is allowed to mint its typed
capability: without a connector call, local files cannot prove remote custody.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Any, Final

from quant_rabbit.dojo_candidate_lineage_registry import CandidateLineageSnapshot
from quant_rabbit.dojo_terminal_handoff import (
    DojoTerminalHandoffError,
    verify_handoff_receipt,
)


REMOTE_RECEIPT_CONTRACT: Final = "QR_DOJO_DRIVE_REMOTE_METADATA_CONSISTENCY_RECEIPT_V1"
REMOTE_INDEX_CONTRACT: Final = "QR_DOJO_DRIVE_REMOTE_METADATA_CONSISTENCY_INDEX_V1"
SCHEMA_VERSION: Final = 1
MAX_RECEIPT_BYTES: Final = 512 * 1024
MAX_INDEX_BYTES: Final = 8 * 1024 * 1024
TRAINER_READBACK_CONTRACT: Final = "QR_DOJO_DRIVE_TRAINER_READBACK_V1"
TRAINER_READBACK_KINDS: Final = (
    "RUN",
    "EVALUATION",
    "CELLS",
    "SEALED_STUDY",
    "TERMINAL_HANDOFF",
)

_HEX64: Final = re.compile(r"[0-9a-f]{64}\Z")
_MD5: Final = re.compile(r"[0-9a-f]{32}\Z")
_DRIVE_ID: Final = re.compile(r"[A-Za-z0-9_-]{10,200}\Z")
_IDENTIFIER: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}\Z")
_CHUNK_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._|=+-]{0,199}\Z")
_DECIMAL: Final = re.compile(r"[1-9][0-9]{0,39}\Z")

_TRAINER_READBACK_MARKER = object()
_DRIVE_READBACK_METADATA_KEYS: Final = frozenset(
    {
        "id",
        "name",
        "size",
        "md5Checksum",
        "modifiedTime",
        "parents",
        "trashed",
        "version",
        "headRevisionId",
    }
)
_TRAINER_READBACK_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "classification",
        "run_sha256",
        "study_sha256",
        "evaluation_sha256",
        "terminal_handoff_receipt_sha256",
        "lineage_tip_sha256",
        "fixed_denominator",
        "expected_artifact_kinds",
        "downloaded_artifact_count",
        "readback_at_utc",
        "objects",
        "download_bytes_match_local_artifacts",
        "drive_metadata_revision_bound",
        "drive_parents_bound",
        "drive_trashed_false",
        "external_readback_attested",
        "remote_verified",
        "trainer_packet_eligible",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "resource_gate_unblock_allowed",
        "readback_sha256",
    }
)
_TRAINER_READBACK_OBJECT_KEYS: Final = frozenset(
    {
        "artifact_kind",
        "drive_file_id",
        "drive_parent_id",
        "drive_file_name",
        "content_sha256",
        "content_size_bytes",
        "md5_checksum",
        "modified_time",
        "version",
        "head_revision_id",
        "trashed",
    }
)

_AUTHORITY = {
    "source_deleted": False,
    "proof_eligible": False,
    "promotion_eligible": False,
    "live_permission": False,
    "order_authority": "NONE",
    "broker_mutation_allowed": False,
    "trainer_unblock_allowed": False,
    "resource_gate_unblock_allowed": False,
}
_RUN_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "study_sha256",
        "status",
        "corpus",
        "fixed_denominator",
        "coordinates",
        "cells_path",
        "evaluation_path",
        "evaluation_sha256",
        "classification",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "run_sha256",
    }
)
_RUN_DENOMINATOR_KEYS: Final = frozenset(
    {
        "expected_cell_count",
        "observed_cell_count",
        "failed_cell_count",
        "dropped_cell_count",
        "coordinate_receipts_complete",
        "execution_success_complete",
    }
)
_COORDINATE_KEYS: Final = frozenset(
    {
        "candidate_id",
        "intrabar",
        "cost_arm",
        "status",
        "main_session_dir",
        "main_error",
        "lopo_replay_complete",
        "lopo",
        "cell_sha256",
    }
)
_LINEAGE_KEYS: Final = frozenset(
    {
        "registry_id",
        "lineage_prefix",
        "attempt_ordinal",
        "result_event_sequence",
        "result_event_sha256",
        "lineage_tip_sequence",
        "lineage_tip_sha256",
    }
)
_LINEAGE_EVENT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "registry_id",
        "sequence",
        "event_type",
        "event_at_utc",
        "previous_event_sha256",
        "body",
        "external_witness_status",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "event_sha256",
    }
)
_HANDOFF_BINDING_KEYS: Final = frozenset(
    {
        "receipt_sha256",
        "binding_timing_classification",
        "run_semantic_sha256",
        "fixed_denominator",
        "terminal_run_artifact_sha256",
        "terminal_run_artifact_size_bytes",
        "terminal_evaluation_artifact_sha256",
        "terminal_evaluation_artifact_size_bytes",
        "terminal_cells_artifact_sha256",
        "terminal_cells_artifact_size_bytes",
        "lineage_tip_sequence",
        "lineage_tip_sha256",
        "result_event_sequence",
        "result_event_sha256",
    }
)
_RECEIPT_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "classification",
        "run_sha256",
        "study_sha256",
        "evaluation_sha256",
        "lineage_binding",
        "terminal_handoff_binding",
        "coordinate",
        "local_archive",
        "drive_archive",
        "checked_at_utc",
        "remote_metadata_consistent",
        "external_readback_required",
        "external_readback_attested",
        "remote_verified",
        *_AUTHORITY,
        "receipt_sha256",
    }
)
_CELL_ID_KEYS: Final = frozenset(
    {"candidate_id", "intrabar", "cost_arm", "cell_sha256"}
)
_LOCAL_ARCHIVE_KEYS: Final = frozenset(
    {
        "plan_sha256",
        "finalization_sha256",
        "content_tree_sha256",
        "archive_sha256",
        "archive_size_bytes",
        "archive_md5_checksum",
        "archive_name",
    }
)
_DRIVE_ARCHIVE_KEYS: Final = frozenset(
    {
        "file_id",
        "parent_id",
        "name",
        "size_bytes",
        "md5_checksum",
        "content_sha256",
        "modified_time",
    }
)
_INDEX_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "classification",
        "run_sha256",
        "study_sha256",
        "evaluation_sha256",
        "lineage_binding",
        "terminal_handoff_binding",
        "expected_cell_count",
        "remote_receipt_count",
        "entries",
        "remote_metadata_consistent",
        "external_readback_required",
        "external_readback_attested",
        "remote_verified",
        *_AUTHORITY,
        "index_sha256",
    }
)
_INDEX_ENTRY_KEYS: Final = frozenset(
    {
        "coordinate",
        "plan_sha256",
        "finalization_sha256",
        "archive_sha256",
        "archive_size_bytes",
        "archive_md5_checksum",
        "remote_receipt_sha256",
        "remote_receipt_content_sha256",
        "remote_receipt_size_bytes",
        "drive_file_id",
        "drive_parent_id",
        "drive_file_name",
        "drive_size_bytes",
        "drive_md5_checksum",
        "drive_content_sha256",
    }
)

# Local archiver V1 surfaces.  Optional ``plan_path``/``receipt_path`` are
# returned by the Python API but are not persisted inside the sealed objects.
_PLAN_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "source_run_root",
        "destination_root",
        "chunk_kind",
        "chunk_id",
        "terminal_run",
        "file_count",
        "total_source_bytes",
        "content_tree_sha256",
        "files",
        "archive_format",
        "archive_member_prefix",
        "source_deletion_allowed",
        "source_deleted",
        "remote_verification",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "plan_sha256",
    }
)
_FINALIZATION_KEYS: Final = frozenset(
    {
        "contract",
        "schema_version",
        "plan_path",
        "plan_sha256",
        "content_tree_sha256",
        "chunk_kind",
        "chunk_id",
        "archive_path",
        "archive_sha256",
        "archive_size_bytes",
        "file_count",
        "total_source_bytes",
        "local_payload_verified",
        "atomic_publish_complete",
        "source_deletion_allowed",
        "source_deleted",
        "remote_verification",
        "proof_eligible",
        "promotion_eligible",
        "live_permission",
        "order_authority",
        "broker_mutation_allowed",
        "finalization_sha256",
    }
)
_NOT_REMOTE = {
    "status": "NOT_REQUESTED",
    "remote_verified": False,
    "metadata_receipt_sha256": None,
}
_PLAN_TERMINAL_KEYS: Final = frozenset(
    {
        "contract",
        "status",
        "run_sha256",
        "study_sha256",
        "evaluation_sha256",
        "classification",
        "fixed_denominator",
    }
)
_REMOTE_KEYS: Final = frozenset(
    {"status", "remote_verified", "metadata_receipt_sha256"}
)


class DojoDriveRemoteEvidenceError(ValueError):
    """Drive metadata claims are partial, foreign, noncanonical, or unsafe."""


@dataclass(frozen=True, order=True)
class CellCoordinate:
    candidate_id: str
    intrabar: str
    cost_arm: str
    cell_sha256: str

    @property
    def chunk_id(self) -> str:
        return f"{self.candidate_id}|{self.intrabar}|{self.cost_arm}"

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "intrabar": self.intrabar,
            "cost_arm": self.cost_arm,
            "cell_sha256": self.cell_sha256,
        }


@dataclass(frozen=True)
class LocalArchiveBinding:
    """Expected immutable identity derived from local archiver V1 receipts."""

    coordinate: CellCoordinate
    run_sha256: str
    study_sha256: str
    evaluation_sha256: str
    plan_sha256: str
    finalization_sha256: str
    content_tree_sha256: str
    archive_sha256: str
    archive_size_bytes: int
    archive_md5_checksum: str
    archive_name: str


@dataclass(frozen=True)
class DriveMetadataConsistencyResult:
    """A complete local consistency result, never a remote-readback attestation."""

    index: Mapping[str, Any]
    receipts: tuple[Mapping[str, Any], ...]
    index_content_sha256: str
    index_content_size_bytes: int
    remote_metadata_consistent: bool = True
    external_readback_required: bool = True
    external_readback_attested: bool = False
    remote_verified: bool = False
    proof_eligible: bool = False
    promotion_eligible: bool = False
    live_permission: bool = False
    order_authority: str = "NONE"
    broker_mutation_allowed: bool = False
    trainer_unblock_allowed: bool = False
    resource_gate_unblock_allowed: bool = False


@dataclass(frozen=True)
class _DriveDownloadedArtifact:
    """One connector-fetched Drive file resource plus its downloaded bytes.

    ``metadata`` must be the selected Google Drive file-resource fields from
    the same readback operation as ``downloaded_bytes``.  A metadata-only JSON
    claim is deliberately insufficient: the verifier recomputes both hashes
    and requires the downloaded bytes to equal the independently opened local
    terminal artifact byte-for-byte.
    """

    artifact_kind: str
    metadata: Mapping[str, Any]
    downloaded_bytes: bytes


@dataclass(frozen=True)
class _VerifiedDriveTrainerReadback:
    """Exact five-artifact Drive readback admitted for one trainer packet.

    The private marker makes this an in-process validator result rather than a
    deserializable capability.  Packet construction must call
    :func:`_trainer_packet_drive_evidence_refs`, which rechecks the marker, seal,
    fixed denominator, terminal hashes and current lineage tip.
    """

    evidence: Mapping[str, Any]
    _validation_marker: object = field(repr=False, compare=False)


def _verify_authenticated_drive_trainer_readback(
    *,
    run: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    sealed_study: Mapping[str, Any],
    terminal_handoff_receipt: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
    expected_parent_id: str,
    local_artifact_bytes: Mapping[str, bytes],
    readbacks: Sequence[_DriveDownloadedArtifact],
    readback_at_utc: datetime | str,
) -> _VerifiedDriveTrainerReadback:
    """Private adapter seam for exact authenticated Drive downloads.

    The fixed denominator is rebuilt from ``sealed_study`` + ``cells`` and the
    evaluation is reconstructed before any Drive metadata is considered.  All
    five required artifacts must then have exact local/download byte equality,
    matching Drive MD5/size, one expected parent, ``trashed=false``, and both a
    monotonically identified Drive version and head revision.

    This private boundary is not an attestation source by itself.  Its caller
    must be an authenticated Drive connector adapter that obtained ``metadata``
    and ``downloaded_bytes`` in the same process and operation.  It is kept
    private so a JSON/file CLI cannot turn self-authored metadata into
    ``remote_verified=true``.  Unlike legacy metadata-consistency receipts, no
    caller-authored ``remote_verified`` field is accepted as input.
    """

    # Lazy import avoids a module cycle: the trainer packet imports the typed
    # result class but the Drive verifier reuses the trainer's exact reducer.
    from quant_rabbit.dojo_ai_trainer_packet import (
        DojoAITrainerPacketError,
        validate_terminal_result_bundle,
    )
    from quant_rabbit.dojo_bot_trainer import DojoBotTrainerError, verify_sealed_study

    try:
        verified_study = verify_sealed_study(
            sealed_study, sealed_study["source_digests"]
        )
        normalized_run, normalized_evaluation, normalized_cells = (
            validate_terminal_result_bundle(
                run=run,
                evaluation=evaluation,
                cells=cells,
                sealed_study=verified_study,
            )
        )
    except (KeyError, ValueError, DojoBotTrainerError, DojoAITrainerPacketError) as exc:
        raise DojoDriveRemoteEvidenceError(
            f"terminal readback denominator is invalid: {exc}"
        ) from exc

    lineage_binding = _lineage_binding(lineage, normalized_run)
    handoff_binding = _handoff_binding(
        terminal_handoff_receipt, normalized_run, lineage_binding
    )
    parent_id = _drive_id(expected_parent_id, "expected Drive parent id")
    observed_at = _utc_text(readback_at_utc, "readback_at_utc")
    observed_instant = _utc(observed_at, "readback_at_utc")
    handoff_instant = _utc(
        terminal_handoff_receipt.get("recorded_at_utc"),
        "terminal handoff recorded_at_utc",
    )
    if observed_instant < handoff_instant:
        raise DojoDriveRemoteEvidenceError("Drive readback predates terminal handoff")

    if not isinstance(local_artifact_bytes, Mapping):
        raise DojoDriveRemoteEvidenceError("local artifact bytes must be an object")
    if set(local_artifact_bytes) != set(TRAINER_READBACK_KINDS):
        raise DojoDriveRemoteEvidenceError(
            "local Drive readback denominator must contain all five artifact kinds"
        )
    expected_values: dict[str, Any] = {
        "RUN": normalized_run,
        "EVALUATION": normalized_evaluation,
        "CELLS": _clone(cells),
        "SEALED_STUDY": verified_study,
        "TERMINAL_HANDOFF": verify_handoff_receipt(terminal_handoff_receipt),
    }
    local_bytes: dict[str, bytes] = {}
    for kind in TRAINER_READBACK_KINDS:
        raw = local_artifact_bytes[kind]
        if not isinstance(raw, bytes) or not raw:
            raise DojoDriveRemoteEvidenceError(
                f"local {kind} artifact must be nonempty exact bytes"
            )
        parsed = _parse_strict_json_bytes(raw, f"local {kind} artifact")
        if parsed != expected_values[kind]:
            raise DojoDriveRemoteEvidenceError(
                f"local {kind} bytes diverge from the verified terminal bundle"
            )
        local_bytes[kind] = raw

    if not isinstance(readbacks, Sequence) or isinstance(
        readbacks, (str, bytes, bytearray)
    ):
        raise DojoDriveRemoteEvidenceError("Drive readbacks must be an array")
    if len(readbacks) != len(TRAINER_READBACK_KINDS):
        raise DojoDriveRemoteEvidenceError(
            "Drive readback denominator omits or adds a terminal artifact"
        )
    by_kind: dict[str, _DriveDownloadedArtifact] = {}
    for item in readbacks:
        if not isinstance(item, _DriveDownloadedArtifact):
            raise DojoDriveRemoteEvidenceError("Drive readback has the wrong type")
        kind = item.artifact_kind
        if kind not in TRAINER_READBACK_KINDS or kind in by_kind:
            raise DojoDriveRemoteEvidenceError(
                "Drive readback artifact kind is duplicate or unsupported"
            )
        by_kind[kind] = item
    if set(by_kind) != set(TRAINER_READBACK_KINDS):
        raise DojoDriveRemoteEvidenceError(
            "Drive readback denominator omits a terminal artifact"
        )

    objects: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    seen_revisions: set[tuple[str, str]] = set()
    for kind in TRAINER_READBACK_KINDS:
        item = by_kind[kind]
        downloaded = item.downloaded_bytes
        if not isinstance(downloaded, bytes) or not downloaded:
            raise DojoDriveRemoteEvidenceError(
                f"downloaded {kind} artifact must be nonempty exact bytes"
            )
        if downloaded != local_bytes[kind]:
            raise DojoDriveRemoteEvidenceError(
                f"downloaded {kind} bytes differ from the local terminal artifact"
            )
        metadata = _normalize_drive_readback_metadata(
            item.metadata,
            expected_parent_id=parent_id,
            downloaded_bytes=downloaded,
        )
        modified = _utc(metadata["modified_time"], f"{kind} modified_time")
        if modified < handoff_instant or modified > observed_instant:
            raise DojoDriveRemoteEvidenceError(
                f"Drive {kind} modified time is outside handoff/readback bounds"
            )
        if (
            metadata["drive_file_id"] in seen_ids
            or metadata["drive_file_name"] in seen_names
            or (metadata["drive_file_id"], metadata["version"]) in seen_revisions
        ):
            raise DojoDriveRemoteEvidenceError(
                "Drive readback reuses a file, name, or revision"
            )
        seen_ids.add(metadata["drive_file_id"])
        seen_names.add(metadata["drive_file_name"])
        seen_revisions.add((metadata["drive_file_id"], metadata["version"]))
        objects.append({"artifact_kind": kind, **metadata})

    body = {
        "contract": TRAINER_READBACK_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "run_sha256": normalized_run["run_sha256"],
        "study_sha256": verified_study["study_sha256"],
        "evaluation_sha256": normalized_evaluation["evaluation_sha256"],
        "terminal_handoff_receipt_sha256": handoff_binding["receipt_sha256"],
        "lineage_tip_sha256": lineage_binding["lineage_tip_sha256"],
        "fixed_denominator": _clone(normalized_run["fixed_denominator"]),
        "expected_artifact_kinds": list(TRAINER_READBACK_KINDS),
        "downloaded_artifact_count": len(objects),
        "readback_at_utc": observed_at,
        "objects": objects,
        "download_bytes_match_local_artifacts": True,
        "drive_metadata_revision_bound": True,
        "drive_parents_bound": True,
        "drive_trashed_false": True,
        "external_readback_attested": True,
        "remote_verified": True,
        "trainer_packet_eligible": True,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
        "resource_gate_unblock_allowed": False,
    }
    evidence = {**body, "readback_sha256": canonical_json_sha256(body)}
    _verify_trainer_readback_evidence(evidence)
    return _VerifiedDriveTrainerReadback(
        evidence=_clone(evidence),
        _validation_marker=_TRAINER_READBACK_MARKER,
    )


def _trainer_packet_drive_evidence_refs(
    value: _VerifiedDriveTrainerReadback,
    *,
    expected_run_sha256: str,
    expected_study_sha256: str,
    expected_evaluation_sha256: str,
    expected_lineage_tip_sha256: str,
    expected_cell_count: int,
) -> list[dict[str, Any]]:
    """Reduce a same-process verified readback into packet-safe references."""

    if (
        not isinstance(value, _VerifiedDriveTrainerReadback)
        or value._validation_marker is not _TRAINER_READBACK_MARKER
    ):
        raise DojoDriveRemoteEvidenceError(
            "trainer packet requires the typed Drive readback validator output"
        )
    evidence = _verify_trainer_readback_evidence(value.evidence)
    denominator = evidence["fixed_denominator"]
    if (
        evidence["run_sha256"] != _sha(expected_run_sha256, "expected run SHA-256")
        or evidence["study_sha256"]
        != _sha(expected_study_sha256, "expected study SHA-256")
        or evidence["evaluation_sha256"]
        != _sha(expected_evaluation_sha256, "expected evaluation SHA-256")
        or evidence["lineage_tip_sha256"]
        != _sha(expected_lineage_tip_sha256, "expected lineage tip SHA-256")
        or denominator["expected_cell_count"]
        != _positive_integer(expected_cell_count, "expected cell count")
        or denominator["observed_cell_count"] != expected_cell_count
    ):
        raise DojoDriveRemoteEvidenceError(
            "typed Drive readback is foreign to the current trainer result"
        )
    return [
        {
            "artifact_kind": item["artifact_kind"],
            "drive_file_id": item["drive_file_id"],
            "drive_parent_id": item["drive_parent_id"],
            "content_sha256": item["content_sha256"],
            "content_size_bytes": item["content_size_bytes"],
            "version": item["version"],
            "head_revision_id": item["head_revision_id"],
            "readback_sha256": evidence["readback_sha256"],
            "remote_verified": True,
        }
        for item in evidence["objects"]
    ]


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic finite JSON bytes without a trailing newline."""

    _validate_json(value, "value")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoDriveRemoteEvidenceError("value is not canonical JSON") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_artifact_bytes(value: Any) -> bytes:
    """Return the sole accepted Drive JSON artifact representation."""

    return canonical_json_bytes(value) + b"\n"


def local_archive_binding_from_receipts(
    *,
    plan: Mapping[str, Any],
    finalization: Mapping[str, Any],
    archive_md5_checksum: str,
) -> LocalArchiveBinding:
    """Derive a typed expectation from sealed local archiver V1 objects.

    The caller computes the MD5 from the already finalized local archive.  This
    function does not open the archive or trust Drive metadata to supply it.
    """

    plan_row = _api_mapping(plan, _PLAN_KEYS, "archive plan", "plan_path")
    final_row = _api_mapping(
        finalization, _FINALIZATION_KEYS, "archive finalization", "receipt_path"
    )
    _verify_seal(plan_row, "plan_sha256", "archive plan")
    _verify_seal(final_row, "finalization_sha256", "archive finalization")
    if (
        plan_row["contract"] != "QR_DOJO_DRIVE_ARCHIVE_PLAN_V1"
        or plan_row["schema_version"] != 1
        or plan_row["chunk_kind"] != "cell"
        or plan_row["archive_format"] != "POSIX_PAX_TAR_ZSTD"
        or plan_row["archive_member_prefix"] != "run/"
    ):
        raise DojoDriveRemoteEvidenceError("unsupported local archive plan")
    chunk_id = plan_row["chunk_id"]
    if not isinstance(chunk_id, str) or not _CHUNK_ID.fullmatch(chunk_id):
        raise DojoDriveRemoteEvidenceError("local archive chunk id is invalid")
    parts = chunk_id.split("|")
    if len(parts) != 3 or not all(parts):
        raise DojoDriveRemoteEvidenceError("cell chunk id is not a coordinate")
    terminal = _exact(
        plan_row["terminal_run"], _PLAN_TERMINAL_KEYS, "plan terminal run"
    )
    run_sha = _sha(terminal.get("run_sha256"), "plan run SHA-256")
    study_sha = _sha(terminal.get("study_sha256"), "plan study SHA-256")
    evaluation_sha = _sha(terminal.get("evaluation_sha256"), "plan evaluation SHA-256")
    _require_local_no_authority(plan_row, "archive plan")
    plan_remote = _exact(
        plan_row["remote_verification"], _REMOTE_KEYS, "plan remote verification"
    )
    if plan_remote != _NOT_REMOTE:
        raise DojoDriveRemoteEvidenceError("local plan self-asserts remote custody")

    if (
        final_row["contract"] != "QR_DOJO_DRIVE_ARCHIVE_FINALIZATION_V1"
        or final_row["schema_version"] != 1
        or final_row["plan_sha256"] != plan_row["plan_sha256"]
        or final_row["content_tree_sha256"] != plan_row["content_tree_sha256"]
        or final_row["chunk_kind"] != "cell"
        or final_row["chunk_id"] != chunk_id
        or final_row["file_count"] != plan_row["file_count"]
        or final_row["total_source_bytes"] != plan_row["total_source_bytes"]
        or final_row["local_payload_verified"] is not True
        or final_row["atomic_publish_complete"] is not True
    ):
        raise DojoDriveRemoteEvidenceError(
            "local finalization does not bind the archive plan"
        )
    _require_local_no_authority(final_row, "archive finalization")
    final_remote = _exact(
        final_row["remote_verification"],
        _REMOTE_KEYS,
        "finalization remote verification",
    )
    if final_remote != _NOT_REMOTE:
        raise DojoDriveRemoteEvidenceError(
            "local finalization self-asserts remote custody"
        )
    size = _positive_integer(final_row["archive_size_bytes"], "archive size")
    archive_sha = _sha(final_row["archive_sha256"], "archive SHA-256")
    archive_md5 = _md5(archive_md5_checksum, "local archive MD5")
    archive_path = final_row["archive_path"]
    if not isinstance(archive_path, str) or not archive_path:
        raise DojoDriveRemoteEvidenceError("archive path is invalid")
    archive_name = PurePath(archive_path).name
    expected_name = f"cell-{chunk_id}-{plan_row['plan_sha256']}.tar.zst"
    if archive_name != expected_name:
        raise DojoDriveRemoteEvidenceError("archive filename is not plan-derived")
    return LocalArchiveBinding(
        coordinate=CellCoordinate(parts[0], parts[1], parts[2], ""),
        run_sha256=run_sha,
        study_sha256=study_sha,
        evaluation_sha256=evaluation_sha,
        plan_sha256=_sha(plan_row["plan_sha256"], "plan SHA-256"),
        finalization_sha256=_sha(
            final_row["finalization_sha256"], "finalization SHA-256"
        ),
        content_tree_sha256=_sha(
            plan_row["content_tree_sha256"], "content tree SHA-256"
        ),
        archive_sha256=archive_sha,
        archive_size_bytes=size,
        archive_md5_checksum=archive_md5,
        archive_name=archive_name,
    )


def verify_remote_receipt_bytes(
    raw: bytes,
    *,
    expected_run: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
    terminal_handoff_receipt: Mapping[str, Any],
    local_archive: LocalArchiveBinding,
    expected_archive_parent_id: str,
) -> dict[str, Any]:
    """Check one metadata claim against handoff/local truth without Drive I/O."""

    run, coordinates = _verify_run(expected_run)
    lineage_binding = _lineage_binding(lineage, run)
    handoff_binding = _handoff_binding(terminal_handoff_receipt, run, lineage_binding)
    parent_id = _drive_id(expected_archive_parent_id, "expected Drive parent id")
    expected_coordinate = coordinates.get(local_archive.coordinate.chunk_id)
    if expected_coordinate is None:
        raise DojoDriveRemoteEvidenceError("local archive is foreign to current run")
    bound_archive = _bind_local_to_run(local_archive, run, expected_coordinate)
    value = _parse_canonical_artifact(raw, MAX_RECEIPT_BYTES, "remote receipt")
    receipt = _exact(value, _RECEIPT_KEYS, "remote receipt")
    _verify_seal(receipt, "receipt_sha256", "remote receipt")
    if (
        receipt["contract"] != REMOTE_RECEIPT_CONTRACT
        or receipt["schema_version"] != SCHEMA_VERSION
        or receipt["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY"
        or receipt["run_sha256"] != run["run_sha256"]
        or receipt["study_sha256"] != run["study_sha256"]
        or receipt["evaluation_sha256"] != run["evaluation_sha256"]
        or receipt["lineage_binding"] != lineage_binding
        or receipt["terminal_handoff_binding"] != handoff_binding
        or receipt["remote_metadata_consistent"] is not True
        or receipt["external_readback_required"] is not True
        or receipt["external_readback_attested"] is not False
        or receipt["remote_verified"] is not False
    ):
        raise DojoDriveRemoteEvidenceError(
            "metadata receipt is inconsistent, externally attested, or foreign"
        )
    _require_no_authority(receipt, "remote receipt")
    coordinate = _coordinate(receipt["coordinate"], "remote receipt coordinate")
    if coordinate != expected_coordinate:
        raise DojoDriveRemoteEvidenceError("remote receipt cell binding drifted")
    local = _exact(
        receipt["local_archive"], _LOCAL_ARCHIVE_KEYS, "receipt local archive"
    )
    if local != bound_archive:
        raise DojoDriveRemoteEvidenceError(
            "remote receipt does not bind the local archive"
        )
    drive = _verify_drive_archive(
        receipt["drive_archive"],
        expected=bound_archive,
        expected_parent_id=parent_id,
    )
    checked = _utc(receipt["checked_at_utc"], "checked_at_utc")
    modified = _utc(drive["modified_time"], "Drive modified_time")
    result_at = _utc(lineage.latest_event_at_utc, "lineage latest event time")
    if modified > checked or modified < result_at or checked < result_at:
        raise DojoDriveRemoteEvidenceError(
            "remote receipt timestamps predate result or move backward"
        )
    return _clone(receipt)


def verify_remote_index_bundle(
    *,
    index_bytes: bytes,
    receipt_bytes: Sequence[bytes],
    expected_run: Mapping[str, Any],
    lineage: CandidateLineageSnapshot,
    terminal_handoff_receipt: Mapping[str, Any],
    local_archives: Sequence[LocalArchiveBinding],
    expected_archive_parent_id: str,
) -> DriveMetadataConsistencyResult:
    """Check a complete metadata index; never attest an external readback."""

    run, expected_coordinates = _verify_run(expected_run)
    lineage_binding = _lineage_binding(lineage, run)
    handoff_binding = _handoff_binding(terminal_handoff_receipt, run, lineage_binding)
    parent_id = _drive_id(expected_archive_parent_id, "expected Drive parent id")
    if not isinstance(local_archives, Sequence) or isinstance(
        local_archives, (str, bytes, bytearray)
    ):
        raise DojoDriveRemoteEvidenceError("local archives must be an array")
    archive_map: dict[str, LocalArchiveBinding] = {}
    for item in local_archives:
        if not isinstance(item, LocalArchiveBinding):
            raise DojoDriveRemoteEvidenceError(
                "local archive expectation has the wrong type"
            )
        coordinate = expected_coordinates.get(item.coordinate.chunk_id)
        if coordinate is None or item.coordinate.chunk_id in archive_map:
            raise DojoDriveRemoteEvidenceError(
                "local archive denominator is duplicate or foreign"
            )
        _bind_local_to_run(item, run, coordinate)
        archive_map[item.coordinate.chunk_id] = item
    if set(archive_map) != set(expected_coordinates):
        raise DojoDriveRemoteEvidenceError(
            "local archive denominator omits a current-run cell"
        )

    if not isinstance(receipt_bytes, Sequence) or isinstance(
        receipt_bytes, (str, bytes, bytearray)
    ):
        raise DojoDriveRemoteEvidenceError("remote receipt bytes must be an array")
    if len(receipt_bytes) != len(expected_coordinates):
        raise DojoDriveRemoteEvidenceError(
            "remote receipt denominator omits or adds a cell"
        )
    receipts: dict[str, tuple[dict[str, Any], bytes]] = {}
    seen_receipt_seals: set[str] = set()
    seen_receipt_content: set[str] = set()
    seen_drive_ids: set[str] = set()
    seen_drive_names: set[str] = set()
    for raw in receipt_bytes:
        preview = _parse_canonical_artifact(raw, MAX_RECEIPT_BYTES, "remote receipt")
        preview_row = _exact(preview, _RECEIPT_KEYS, "remote receipt")
        coordinate = _coordinate(preview_row["coordinate"], "remote receipt coordinate")
        chunk_id = coordinate.chunk_id
        local = archive_map.get(chunk_id)
        if local is None or chunk_id in receipts:
            raise DojoDriveRemoteEvidenceError(
                "remote receipt coordinate is duplicate or foreign"
            )
        verified = verify_remote_receipt_bytes(
            raw,
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=terminal_handoff_receipt,
            local_archive=local,
            expected_archive_parent_id=parent_id,
        )
        seal = verified["receipt_sha256"]
        content_sha = hashlib.sha256(raw).hexdigest()
        drive = verified["drive_archive"]
        if (
            seal in seen_receipt_seals
            or content_sha in seen_receipt_content
            or drive["file_id"] in seen_drive_ids
            or drive["name"] in seen_drive_names
        ):
            raise DojoDriveRemoteEvidenceError(
                "duplicate remote receipt or Drive object identity"
            )
        seen_receipt_seals.add(seal)
        seen_receipt_content.add(content_sha)
        seen_drive_ids.add(drive["file_id"])
        seen_drive_names.add(drive["name"])
        receipts[chunk_id] = (verified, raw)
    if set(receipts) != set(expected_coordinates):
        raise DojoDriveRemoteEvidenceError(
            "remote receipt set omits a current-run cell"
        )

    index_value = _parse_canonical_artifact(
        index_bytes, MAX_INDEX_BYTES, "remote index"
    )
    index = _exact(index_value, _INDEX_KEYS, "remote index")
    _verify_seal(index, "index_sha256", "remote index")
    expected_count = len(expected_coordinates)
    if (
        index["contract"] != REMOTE_INDEX_CONTRACT
        or index["schema_version"] != SCHEMA_VERSION
        or index["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY"
        or index["run_sha256"] != run["run_sha256"]
        or index["study_sha256"] != run["study_sha256"]
        or index["evaluation_sha256"] != run["evaluation_sha256"]
        or index["lineage_binding"] != lineage_binding
        or index["terminal_handoff_binding"] != handoff_binding
        or index["expected_cell_count"] != expected_count
        or isinstance(index["expected_cell_count"], bool)
        or index["remote_receipt_count"] != expected_count
        or isinstance(index["remote_receipt_count"], bool)
        or index["remote_metadata_consistent"] is not True
        or index["external_readback_required"] is not True
        or index["external_readback_attested"] is not False
        or index["remote_verified"] is not False
    ):
        raise DojoDriveRemoteEvidenceError(
            "metadata index is partial, externally attested, or foreign"
        )
    _require_no_authority(index, "remote index")
    entries = index["entries"]
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise DojoDriveRemoteEvidenceError("remote index cell denominator is partial")
    normalized_entries = [
        _verify_index_entry(item, receipts=receipts) for item in entries
    ]
    ordered = sorted(
        normalized_entries,
        key=lambda item: _coordinate_sort_key(item["coordinate"]),
    )
    if normalized_entries != ordered:
        raise DojoDriveRemoteEvidenceError("remote index entries are not canonical")
    entry_coordinates = [
        _coordinate(item["coordinate"], "remote index coordinate").chunk_id
        for item in normalized_entries
    ]
    if len(set(entry_coordinates)) != expected_count or set(entry_coordinates) != set(
        expected_coordinates
    ):
        raise DojoDriveRemoteEvidenceError(
            "remote index duplicates or omits a current-run cell"
        )
    ordered_receipts = tuple(
        _clone(receipts[chunk_id][0]) for chunk_id in sorted(receipts)
    )
    return DriveMetadataConsistencyResult(
        index=_clone(index),
        receipts=ordered_receipts,
        index_content_sha256=hashlib.sha256(index_bytes).hexdigest(),
        index_content_size_bytes=len(index_bytes),
    )


def _verify_index_entry(
    value: Any, *, receipts: Mapping[str, tuple[Mapping[str, Any], bytes]]
) -> dict[str, Any]:
    entry = _exact(value, _INDEX_ENTRY_KEYS, "remote index entry")
    coordinate = _coordinate(entry["coordinate"], "remote index coordinate")
    bound = receipts.get(coordinate.chunk_id)
    if bound is None:
        raise DojoDriveRemoteEvidenceError("remote index entry is foreign")
    receipt, raw = bound
    local = receipt["local_archive"]
    drive = receipt["drive_archive"]
    expected = {
        "coordinate": coordinate.as_dict(),
        "plan_sha256": local["plan_sha256"],
        "finalization_sha256": local["finalization_sha256"],
        "archive_sha256": local["archive_sha256"],
        "archive_size_bytes": local["archive_size_bytes"],
        "archive_md5_checksum": local["archive_md5_checksum"],
        "remote_receipt_sha256": receipt["receipt_sha256"],
        "remote_receipt_content_sha256": hashlib.sha256(raw).hexdigest(),
        "remote_receipt_size_bytes": len(raw),
        "drive_file_id": drive["file_id"],
        "drive_parent_id": drive["parent_id"],
        "drive_file_name": drive["name"],
        "drive_size_bytes": drive["size_bytes"],
        "drive_md5_checksum": drive["md5_checksum"],
        "drive_content_sha256": drive["content_sha256"],
    }
    if entry != expected:
        raise DojoDriveRemoteEvidenceError(
            "remote index entry diverges from fetched receipt bytes"
        )
    return expected


def _verify_run(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, CellCoordinate]]:
    run = _exact(value, _RUN_KEYS, "terminal run")
    _verify_seal(run, "run_sha256", "terminal run")
    if (
        run["contract"] != "QR_DOJO_BOT_TRAINER_RUN_V1"
        or run["schema_version"] != 1
        or run["status"] not in {"COMPLETE", "COMPLETE_WITH_FAILED_CELLS"}
        or run["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY"
    ):
        raise DojoDriveRemoteEvidenceError("run is not a terminal TRAIN result")
    _require_no_authority(
        run,
        "terminal run",
        include_source_deleted=False,
        include_unblock_fields=False,
    )
    _sha(run["study_sha256"], "run study SHA-256")
    _sha(run["evaluation_sha256"], "run evaluation SHA-256")
    denominator = _exact(
        run["fixed_denominator"], _RUN_DENOMINATOR_KEYS, "run fixed denominator"
    )
    expected = _positive_integer(
        denominator["expected_cell_count"], "expected cell count"
    )
    observed = _integer(denominator["observed_cell_count"], "observed cell count")
    failed = _integer(denominator["failed_cell_count"], "failed cell count")
    dropped = _integer(denominator["dropped_cell_count"], "dropped cell count")
    coordinate_complete = denominator["coordinate_receipts_complete"]
    execution_complete = denominator["execution_success_complete"]
    if (
        observed != expected
        or not 0 <= failed <= expected
        or dropped != 0
        or coordinate_complete is not True
        or not isinstance(execution_complete, bool)
        or execution_complete != (failed == 0)
        or (run["status"] == "COMPLETE") != (failed == 0)
    ):
        raise DojoDriveRemoteEvidenceError("run fixed denominator is incomplete")
    rows = run["coordinates"]
    if not isinstance(rows, list) or len(rows) != expected:
        raise DojoDriveRemoteEvidenceError("run coordinate denominator is partial")
    coordinates: dict[str, CellCoordinate] = {}
    for index, raw in enumerate(rows):
        row = _exact(raw, _COORDINATE_KEYS, f"run coordinate {index}")
        coordinate = CellCoordinate(
            _identifier(row["candidate_id"], "candidate_id"),
            _identifier(row["intrabar"], "intrabar"),
            _identifier(row["cost_arm"], "cost_arm"),
            _sha(row["cell_sha256"], "cell SHA-256"),
        )
        if coordinate.chunk_id in coordinates:
            raise DojoDriveRemoteEvidenceError("run coordinate is duplicated")
        coordinates[coordinate.chunk_id] = coordinate
    return _clone(run), coordinates


def _lineage_binding(
    lineage: CandidateLineageSnapshot, run: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(lineage, CandidateLineageSnapshot):
        raise DojoDriveRemoteEvidenceError("lineage must be a verified snapshot")
    if (
        not lineage.results
        or len(lineage.studies) != len(lineage.results)
        or lineage.external_witness_status != "ABSENT"
        or lineage.proof_eligible is not False
        or lineage.promotion_eligible is not False
        or lineage.live_permission is not False
        or lineage.order_authority != "NONE"
        or lineage.broker_mutation_allowed is not False
    ):
        raise DojoDriveRemoteEvidenceError(
            "lineage is not at a safe RESULT_BOUND boundary"
        )
    result = lineage.results[-1]
    attempt = _positive_integer(result.get("attempt_ordinal"), "attempt ordinal")
    if (
        result.get("study_sha256") != run["study_sha256"]
        or result.get("evaluation_sha256") != run["evaluation_sha256"]
    ):
        raise DojoDriveRemoteEvidenceError("lineage result is foreign to current run")
    matches = [
        event
        for event in lineage.events
        if isinstance(event, Mapping)
        and event.get("event_type") == "RESULT_BOUND"
        and event.get("body") == result
    ]
    if len(matches) != 1:
        raise DojoDriveRemoteEvidenceError("lineage lacks one exact RESULT_BOUND event")
    event = _exact(matches[0], _LINEAGE_EVENT_KEYS, "lineage RESULT_BOUND event")
    event_sha = _sha(event.get("event_sha256"), "result event SHA-256")
    event_sequence = _integer(event.get("sequence"), "result event sequence")
    event_body = {key: item for key, item in event.items() if key != "event_sha256"}
    if canonical_json_sha256(event_body) != event_sha:
        raise DojoDriveRemoteEvidenceError("lineage RESULT_BOUND event seal drifted")
    if (
        event["contract"] != "QR_DOJO_CANDIDATE_LINEAGE_EVENT_V1"
        or event["schema_version"] != 1
        or event["registry_id"] != lineage.registry_id
        or event["event_type"] != "RESULT_BOUND"
        or event["external_witness_status"] != "ABSENT"
        or event["proof_eligible"] is not False
        or event["promotion_eligible"] is not False
        or event["live_permission"] is not False
        or event["order_authority"] != "NONE"
        or event["broker_mutation_allowed"] is not False
        or _utc_text(event["event_at_utc"], "result event time")
        != lineage.latest_event_at_utc
        or event_sequence != lineage.latest_sequence
        or event_sha != lineage.latest_event_sha256
        or lineage.event_count != len(lineage.events)
        or lineage.latest_sequence != lineage.event_count - 1
    ):
        raise DojoDriveRemoteEvidenceError(
            "lineage RESULT_BOUND event is not the current tip"
        )
    return {
        "registry_id": _identifier(lineage.registry_id, "registry_id"),
        "lineage_prefix": _identifier(lineage.lineage_prefix, "lineage_prefix"),
        "attempt_ordinal": attempt,
        "result_event_sequence": event_sequence,
        "result_event_sha256": event_sha,
        "lineage_tip_sequence": lineage.latest_sequence,
        "lineage_tip_sha256": _sha(lineage.latest_event_sha256, "lineage tip SHA-256"),
    }


def _handoff_binding(
    value: Mapping[str, Any],
    run: Mapping[str, Any],
    lineage_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the Drive claim to a separately verified terminal handoff receipt."""

    try:
        handoff = verify_handoff_receipt(value)
    except DojoTerminalHandoffError as exc:
        raise DojoDriveRemoteEvidenceError(
            f"terminal handoff receipt is invalid: {exc}"
        ) from exc
    terminal = handoff["terminal_bundle"]
    after = handoff["lineage_after"]
    denominator = _exact(
        run["fixed_denominator"], _RUN_DENOMINATOR_KEYS, "run fixed denominator"
    )
    if (
        terminal["run_sha256"] != run["run_sha256"]
        or terminal["study_sha256"] != run["study_sha256"]
        or terminal["evaluation_sha256"] != run["evaluation_sha256"]
        or terminal["status"] != run["status"]
        or terminal["expected_cell_count"] != denominator["expected_cell_count"]
        or terminal["observed_cell_count"] != denominator["observed_cell_count"]
        or terminal["failed_cell_count"] != denominator["failed_cell_count"]
    ):
        raise DojoDriveRemoteEvidenceError(
            "terminal handoff does not bind the exact run semantic SHA/denominator"
        )
    if (
        after["registry_id"] != lineage_binding["registry_id"]
        or after["lineage_prefix"] != lineage_binding["lineage_prefix"]
        or after["attempt_ordinal"] != lineage_binding["attempt_ordinal"]
        or after["study_sha256"] != run["study_sha256"]
        or after["evaluation_sha256"] != run["evaluation_sha256"]
        or after["latest_sequence"] != lineage_binding["lineage_tip_sequence"]
        or after["latest_event_sha256"] != lineage_binding["lineage_tip_sha256"]
        or after["result_event_sequence"] != lineage_binding["result_event_sequence"]
        or after["result_event_sha256"] != lineage_binding["result_event_sha256"]
        or terminal["evaluation"]["artifact_sha256"]
        != after["evaluation_artifact_sha256"]
        or terminal["evaluation"]["artifact_size_bytes"]
        != after["evaluation_artifact_size_bytes"]
    ):
        raise DojoDriveRemoteEvidenceError(
            "terminal handoff does not bind the current lineage RESULT_BOUND tip"
        )
    return {
        "receipt_sha256": _sha(
            handoff["receipt_sha256"], "terminal handoff receipt SHA-256"
        ),
        "binding_timing_classification": _identifier(
            handoff["binding_timing_classification"],
            "handoff binding timing classification",
        ),
        "run_semantic_sha256": _sha(run["run_sha256"], "run semantic SHA-256"),
        "fixed_denominator": _clone(denominator),
        "terminal_run_artifact_sha256": _sha(
            terminal["run"]["artifact_sha256"], "terminal run artifact SHA-256"
        ),
        "terminal_run_artifact_size_bytes": _positive_integer(
            terminal["run"]["artifact_size_bytes"], "terminal run artifact size"
        ),
        "terminal_evaluation_artifact_sha256": _sha(
            terminal["evaluation"]["artifact_sha256"],
            "terminal evaluation artifact SHA-256",
        ),
        "terminal_evaluation_artifact_size_bytes": _positive_integer(
            terminal["evaluation"]["artifact_size_bytes"],
            "terminal evaluation artifact size",
        ),
        "terminal_cells_artifact_sha256": _sha(
            terminal["cells"]["artifact_sha256"],
            "terminal cells artifact SHA-256",
        ),
        "terminal_cells_artifact_size_bytes": _positive_integer(
            terminal["cells"]["artifact_size_bytes"], "terminal cells artifact size"
        ),
        "lineage_tip_sequence": lineage_binding["lineage_tip_sequence"],
        "lineage_tip_sha256": lineage_binding["lineage_tip_sha256"],
        "result_event_sequence": lineage_binding["result_event_sequence"],
        "result_event_sha256": lineage_binding["result_event_sha256"],
    }


def _bind_local_to_run(
    value: LocalArchiveBinding,
    run: Mapping[str, Any],
    expected_coordinate: CellCoordinate,
) -> dict[str, Any]:
    coordinate = value.coordinate
    if (
        coordinate.candidate_id != expected_coordinate.candidate_id
        or coordinate.intrabar != expected_coordinate.intrabar
        or coordinate.cost_arm != expected_coordinate.cost_arm
        or coordinate.cell_sha256 not in {"", expected_coordinate.cell_sha256}
        or value.run_sha256 != run["run_sha256"]
        or value.study_sha256 != run["study_sha256"]
        or value.evaluation_sha256 != run["evaluation_sha256"]
    ):
        raise DojoDriveRemoteEvidenceError(
            "local archive binding is foreign to current run"
        )
    archive_name = _filename(value.archive_name, "local archive name")
    expected_name = (
        f"cell-{coordinate.chunk_id}-{_sha(value.plan_sha256, 'plan SHA-256')}.tar.zst"
    )
    if archive_name != expected_name:
        raise DojoDriveRemoteEvidenceError("local archive filename drifted")
    return {
        "plan_sha256": _sha(value.plan_sha256, "plan SHA-256"),
        "finalization_sha256": _sha(value.finalization_sha256, "finalization SHA-256"),
        "content_tree_sha256": _sha(value.content_tree_sha256, "content tree SHA-256"),
        "archive_sha256": _sha(value.archive_sha256, "archive SHA-256"),
        "archive_size_bytes": _positive_integer(
            value.archive_size_bytes, "archive size"
        ),
        "archive_md5_checksum": _md5(value.archive_md5_checksum, "archive MD5"),
        "archive_name": archive_name,
    }


def _verify_drive_archive(
    value: Any,
    *,
    expected: Mapping[str, Any],
    expected_parent_id: str,
) -> dict[str, Any]:
    drive = _exact(value, _DRIVE_ARCHIVE_KEYS, "Drive archive metadata")
    normalized = {
        "file_id": _drive_id(drive["file_id"], "Drive file id"),
        "parent_id": _drive_id(drive["parent_id"], "Drive parent id"),
        "name": _filename(drive["name"], "Drive filename"),
        "size_bytes": _positive_integer(drive["size_bytes"], "Drive size"),
        "md5_checksum": _md5(drive["md5_checksum"], "Drive MD5"),
        "content_sha256": _sha(drive["content_sha256"], "Drive content SHA-256"),
        "modified_time": _utc_text(drive["modified_time"], "Drive modified_time"),
    }
    if normalized != drive:
        raise DojoDriveRemoteEvidenceError("Drive archive metadata is not canonical")
    if (
        normalized["parent_id"] != expected_parent_id
        or normalized["name"] != expected["archive_name"]
        or normalized["size_bytes"] != expected["archive_size_bytes"]
        or normalized["md5_checksum"] != expected["archive_md5_checksum"]
        or normalized["content_sha256"] != expected["archive_sha256"]
    ):
        raise DojoDriveRemoteEvidenceError(
            "Drive metadata does not match the local archive"
        )
    return normalized


def _coordinate(value: Any, label: str) -> CellCoordinate:
    row = _exact(value, _CELL_ID_KEYS, label)
    coordinate = CellCoordinate(
        _identifier(row["candidate_id"], f"{label} candidate_id"),
        _identifier(row["intrabar"], f"{label} intrabar"),
        _identifier(row["cost_arm"], f"{label} cost_arm"),
        _sha(row["cell_sha256"], f"{label} cell SHA-256"),
    )
    if coordinate.as_dict() != row:
        raise DojoDriveRemoteEvidenceError(f"{label} is not canonical")
    return coordinate


def _coordinate_sort_key(value: Any) -> tuple[str, str, str]:
    coordinate = _coordinate(value, "coordinate")
    return (coordinate.candidate_id, coordinate.intrabar, coordinate.cost_arm)


def _normalize_drive_readback_metadata(
    value: Any,
    *,
    expected_parent_id: str,
    downloaded_bytes: bytes,
) -> dict[str, Any]:
    """Normalize selected Drive v3 file fields against the downloaded payload."""

    row = _exact(value, _DRIVE_READBACK_METADATA_KEYS, "Drive readback metadata")
    parents = row["parents"]
    if (
        not isinstance(parents, list)
        or len(parents) != 1
        or _drive_id(parents[0], "Drive readback parent") != expected_parent_id
    ):
        raise DojoDriveRemoteEvidenceError(
            "Drive readback parents do not equal the expected single parent"
        )
    if row["trashed"] is not False:
        raise DojoDriveRemoteEvidenceError("Drive readback object is trashed")
    size = _decimal_integer(row["size"], "Drive readback size")
    if size != len(downloaded_bytes):
        raise DojoDriveRemoteEvidenceError(
            "Drive readback size differs from downloaded bytes"
        )
    md5 = hashlib.md5(downloaded_bytes, usedforsecurity=False).hexdigest()
    if _md5(row["md5Checksum"], "Drive readback MD5") != md5:
        raise DojoDriveRemoteEvidenceError(
            "Drive readback MD5 differs from downloaded bytes"
        )
    version = _decimal_text(row["version"], "Drive readback version")
    head_revision = _drive_id(row["headRevisionId"], "Drive readback head revision id")
    return {
        "drive_file_id": _drive_id(row["id"], "Drive readback file id"),
        "drive_parent_id": expected_parent_id,
        "drive_file_name": _filename(row["name"], "Drive readback filename"),
        "content_sha256": hashlib.sha256(downloaded_bytes).hexdigest(),
        "content_size_bytes": size,
        "md5_checksum": md5,
        "modified_time": _utc_text(row["modifiedTime"], "Drive modifiedTime"),
        "version": version,
        "head_revision_id": head_revision,
        "trashed": False,
    }


def _verify_trainer_readback_evidence(value: Any) -> dict[str, Any]:
    row = _exact(value, _TRAINER_READBACK_KEYS, "Drive trainer readback evidence")
    claimed = _sha(row["readback_sha256"], "Drive trainer readback SHA-256")
    body = {key: item for key, item in row.items() if key != "readback_sha256"}
    if canonical_json_sha256(body) != claimed:
        raise DojoDriveRemoteEvidenceError("Drive trainer readback SHA-256 mismatch")
    denominator = _exact(
        row["fixed_denominator"],
        _RUN_DENOMINATOR_KEYS,
        "Drive trainer fixed denominator",
    )
    expected = _positive_integer(
        denominator["expected_cell_count"], "Drive trainer expected cell count"
    )
    failed = _nonnegative_integer(
        denominator["failed_cell_count"],
        "Drive trainer failed cell count",
    )
    execution_complete = denominator["execution_success_complete"]
    if (
        _positive_integer(
            denominator["observed_cell_count"],
            "Drive trainer observed cell count",
        )
        != expected
        or _nonnegative_integer(
            denominator["dropped_cell_count"],
            "Drive trainer dropped cell count",
        )
        != 0
        or denominator["coordinate_receipts_complete"] is not True
        or not isinstance(execution_complete, bool)
        or execution_complete is not (failed == 0)
        or row["contract"] != TRAINER_READBACK_CONTRACT
        or row["schema_version"] != SCHEMA_VERSION
        or row["classification"] != "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY"
        or row["expected_artifact_kinds"] != list(TRAINER_READBACK_KINDS)
        or row["downloaded_artifact_count"] != len(TRAINER_READBACK_KINDS)
        or isinstance(row["downloaded_artifact_count"], bool)
        or row["download_bytes_match_local_artifacts"] is not True
        or row["drive_metadata_revision_bound"] is not True
        or row["drive_parents_bound"] is not True
        or row["drive_trashed_false"] is not True
        or row["external_readback_attested"] is not True
        or row["remote_verified"] is not True
        or row["trainer_packet_eligible"] is not True
        or row["proof_eligible"] is not False
        or row["promotion_eligible"] is not False
        or row["live_permission"] is not False
        or row["order_authority"] != "NONE"
        or row["broker_mutation_allowed"] is not False
        or row["resource_gate_unblock_allowed"] is not False
    ):
        raise DojoDriveRemoteEvidenceError(
            "Drive trainer readback contract or authority boundary drifted"
        )
    for field_name in (
        "run_sha256",
        "study_sha256",
        "evaluation_sha256",
        "terminal_handoff_receipt_sha256",
        "lineage_tip_sha256",
    ):
        _sha(row[field_name], f"Drive trainer {field_name}")
    _utc_text(row["readback_at_utc"], "Drive trainer readback_at_utc")
    objects = row["objects"]
    if not isinstance(objects, list) or len(objects) != len(TRAINER_READBACK_KINDS):
        raise DojoDriveRemoteEvidenceError(
            "Drive trainer readback object denominator is incomplete"
        )
    kinds: list[str] = []
    ids: set[str] = set()
    names: set[str] = set()
    parents: set[str] = set()
    revisions: set[tuple[str, str]] = set()
    for raw in objects:
        item = _exact(raw, _TRAINER_READBACK_OBJECT_KEYS, "Drive trainer object")
        kind = item["artifact_kind"]
        if kind not in TRAINER_READBACK_KINDS:
            raise DojoDriveRemoteEvidenceError(
                "Drive trainer readback object kind is unsupported"
            )
        kinds.append(kind)
        file_id = _drive_id(item["drive_file_id"], "Drive trainer file id")
        name = _filename(item["drive_file_name"], "Drive trainer filename")
        if file_id in ids or name in names:
            raise DojoDriveRemoteEvidenceError(
                "Drive trainer readback reuses file identity"
            )
        ids.add(file_id)
        names.add(name)
        parents.add(_drive_id(item["drive_parent_id"], "Drive trainer parent id"))
        _sha(item["content_sha256"], "Drive trainer content SHA-256")
        _positive_integer(item["content_size_bytes"], "Drive trainer content size")
        _md5(item["md5_checksum"], "Drive trainer MD5")
        _utc_text(item["modified_time"], "Drive trainer modified time")
        version = _decimal_text(item["version"], "Drive trainer version")
        head_revision = _drive_id(
            item["head_revision_id"], "Drive trainer head revision"
        )
        revision = (file_id, f"{version}:{head_revision}")
        if revision in revisions:
            raise DojoDriveRemoteEvidenceError(
                "Drive trainer readback reuses a file revision"
            )
        revisions.add(revision)
        if item["trashed"] is not False:
            raise DojoDriveRemoteEvidenceError("Drive trainer object is trashed")
    if kinds != list(TRAINER_READBACK_KINDS):
        raise DojoDriveRemoteEvidenceError(
            "Drive trainer readback object order/denominator drifted"
        )
    if len(parents) != 1:
        raise DojoDriveRemoteEvidenceError(
            "Drive trainer readback objects do not share one parent"
        )
    return _clone(row)


def _parse_strict_json_bytes(raw: bytes, label: str) -> Any:
    if not isinstance(raw, bytes) or not raw:
        raise DojoDriveRemoteEvidenceError(f"{label} must be nonempty exact bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except DojoDriveRemoteEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DojoDriveRemoteEvidenceError(f"{label} is not strict JSON") from exc
    _validate_json(value, label)
    return value


def _parse_canonical_artifact(raw: bytes, maximum: int, label: str) -> Any:
    if not isinstance(raw, bytes):
        raise DojoDriveRemoteEvidenceError(f"{label} must be exact bytes")
    if not raw or len(raw) > maximum:
        raise DojoDriveRemoteEvidenceError(f"{label} is empty or exceeds byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DojoDriveRemoteEvidenceError(f"{label} is not strict JSON") from exc
    if raw != canonical_artifact_bytes(value):
        raise DojoDriveRemoteEvidenceError(f"{label} bytes are not canonical")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DojoDriveRemoteEvidenceError(
                f"duplicate JSON key is forbidden: {key}"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DojoDriveRemoteEvidenceError(f"non-finite JSON constant: {value}")


def _validate_json(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoDriveRemoteEvidenceError(f"{label} contains non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoDriveRemoteEvidenceError(f"{label} contains a non-string key")
            _validate_json(item, f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_json(item, f"{label}[{index}]")
        return
    raise DojoDriveRemoteEvidenceError(f"{label} contains a non-JSON value")


def _verify_seal(value: Mapping[str, Any], field: str, label: str) -> None:
    claimed = _sha(value.get(field), f"{label} {field}")
    body = {key: item for key, item in value.items() if key != field}
    if canonical_json_sha256(body) != claimed:
        raise DojoDriveRemoteEvidenceError(f"{label} SHA-256 mismatch")


def _require_no_authority(
    value: Mapping[str, Any],
    label: str,
    *,
    include_source_deleted: bool = True,
    include_unblock_fields: bool = True,
) -> None:
    expected = dict(_AUTHORITY)
    if not include_source_deleted:
        expected.pop("source_deleted")
    if not include_unblock_fields:
        expected.pop("trainer_unblock_allowed")
        expected.pop("resource_gate_unblock_allowed")
    if any(value.get(key) != item for key, item in expected.items()):
        raise DojoDriveRemoteEvidenceError(f"{label} authority boundary is invalid")


def _require_local_no_authority(value: Mapping[str, Any], label: str) -> None:
    if (
        value.get("source_deletion_allowed") is not False
        or value.get("source_deleted") is not False
        or value.get("proof_eligible") is not False
        or value.get("promotion_eligible") is not False
        or value.get("live_permission") is not False
        or value.get("order_authority") != "NONE"
        or value.get("broker_mutation_allowed") is not False
    ):
        raise DojoDriveRemoteEvidenceError(f"{label} authority boundary is invalid")


def _api_mapping(
    value: Mapping[str, Any], expected: frozenset[str], label: str, api_extra: str
) -> dict[str, Any]:
    row = _mapping(value, label)
    keys = set(row)
    if keys == set(expected) | {api_extra}:
        row = {key: item for key, item in row.items() if key != api_extra}
    return _exact(row, expected, label)


def _exact(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    row = _mapping(value, label)
    actual = set(row)
    if actual != set(expected):
        raise DojoDriveRemoteEvidenceError(
            f"{label} schema mismatch: missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )
    return row


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DojoDriveRemoteEvidenceError(f"{label} must be a JSON object")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid")
    return value


def _md5(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _MD5.fullmatch(value):
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid")
    return value


def _drive_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DRIVE_ID.fullmatch(value):
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid")
    return value


def _filename(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 255
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DojoDriveRemoteEvidenceError(f"{label} must be an integer")
    return value


def _positive_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise DojoDriveRemoteEvidenceError(f"{label} must be positive")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise DojoDriveRemoteEvidenceError(f"{label} must be nonnegative")
    return result


def _decimal_text(value: Any, label: str) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str) or _DECIMAL.fullmatch(value) is None:
        raise DojoDriveRemoteEvidenceError(f"{label} must be a positive decimal")
    return value


def _decimal_integer(value: Any, label: str) -> int:
    return int(_decimal_text(value, label))


def _utc_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid")
    _utc(value, label)
    return value


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DojoDriveRemoteEvidenceError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DojoDriveRemoteEvidenceError(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc)


def _clone(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


__all__ = [
    "CellCoordinate",
    "DojoDriveRemoteEvidenceError",
    "LocalArchiveBinding",
    "REMOTE_INDEX_CONTRACT",
    "REMOTE_RECEIPT_CONTRACT",
    "SCHEMA_VERSION",
    "DriveMetadataConsistencyResult",
    "TRAINER_READBACK_CONTRACT",
    "TRAINER_READBACK_KINDS",
    "canonical_artifact_bytes",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "local_archive_binding_from_receipts",
    "verify_remote_index_bundle",
    "verify_remote_receipt_bytes",
]
