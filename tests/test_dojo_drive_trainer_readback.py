from __future__ import annotations

import copy
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from quant_rabbit.dojo_candidate_lineage_registry import verify_registry
from quant_rabbit.dojo_drive_remote_evidence import (
    TRAINER_READBACK_KINDS,
    DojoDriveRemoteEvidenceError,
    GoogleDriveV3TrainerReadbackConnector,
    _AuthenticatedDriveReadbackConnector,
    _DriveConnectorArtifactReadback,
    _VerifiedDriveTrainerReadback,
    canonical_json_sha256,
    _trainer_packet_drive_evidence_refs,
    _verify_authenticated_drive_trainer_readback,
)
from quant_rabbit.dojo_terminal_handoff import canonical_json_bytes
from test_dojo_terminal_handoff import (
    _coordinate_existing,
    _preseal_lineage,
    _terminal_fixture,
)


RUN_PARENT_ID = "driveTrainerRunParent123"
MANIFEST_PARENT_ID = "driveTrainerManifestParent123"


def _parent_id(kind: str) -> str:
    return (
        RUN_PARENT_ID
        if kind in {"RUN", "EVALUATION", "CELLS"}
        else MANIFEST_PARENT_ID
    )


def _parent_ids() -> dict[str, str]:
    return {kind: _parent_id(kind) for kind in TRAINER_READBACK_KINDS}


class _FakeDriveRequest:
    def __init__(self, result: Any) -> None:
        self.result = result

    def execute(self) -> Any:
        return copy.deepcopy(self.result)


class _FakeDriveFilesResource:
    def __init__(self, service: "_FakeDriveV3Service") -> None:
        self.service = service

    def get(self, **kwargs: Any) -> _FakeDriveRequest:
        self.service.calls.append(("files.get", copy.deepcopy(kwargs)))
        file_id = kwargs["fileId"]
        count = self.service.file_get_counts.get(file_id, 0) + 1
        self.service.file_get_counts[file_id] = count
        metadata = copy.deepcopy(self.service.metadata[file_id])
        if self.service.cas_drift_file_id == file_id and count >= 3:
            metadata["version"] = "999"
            metadata["headRevisionId"] = "driveRevisionDrifted99"
        return _FakeDriveRequest(metadata)


class _FakeDriveRevisionsResource:
    def __init__(self, service: "_FakeDriveV3Service") -> None:
        self.service = service

    def get(self, **kwargs: Any) -> _FakeDriveRequest:
        self.service.calls.append(("revisions.get", copy.deepcopy(kwargs)))
        key = (kwargs["fileId"], kwargs["revisionId"])
        return _FakeDriveRequest(self.service.revisions_by_id[key])


class _FakeDriveV3Service:
    def __init__(
        self,
        *,
        metadata: dict[str, dict[str, Any]],
        revisions_by_id: dict[tuple[str, str], Any],
        cas_drift_file_id: str | None = None,
    ) -> None:
        self.metadata = metadata
        self.revisions_by_id = revisions_by_id
        self.cas_drift_file_id = cas_drift_file_id
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.file_get_counts: dict[str, int] = {}

    def files(self) -> _FakeDriveFilesResource:
        return _FakeDriveFilesResource(self)

    def revisions(self) -> _FakeDriveRevisionsResource:
        return _FakeDriveRevisionsResource(self)


def _drive_v3_service_fixture() -> tuple[
    _FakeDriveV3Service, dict[str, str], dict[str, bytes]
]:
    metadata: dict[str, dict[str, Any]] = {}
    revisions: dict[tuple[str, str], bytes] = {}
    file_ids: dict[str, str] = {}
    payloads: dict[str, bytes] = {}
    for index, kind in enumerate(TRAINER_READBACK_KINDS, start=1):
        file_id = f"driveConcreteFile{index:02d}"
        revision_id = f"driveConcreteRevision{index:02d}"
        raw = f'{{"artifact_kind":"{kind}"}}\n'.encode()
        metadata[file_id] = {
            "id": file_id,
            "name": f"{index:02d}-{kind.lower()}.json",
            "size": str(len(raw)),
            "md5Checksum": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
            "modifiedTime": f"2026-07-20T01:0{index}:00Z",
            "parents": [_parent_id(kind)],
            "trashed": False,
            "version": str(index),
            "headRevisionId": revision_id,
        }
        revisions[(file_id, revision_id)] = raw
        file_ids[kind] = file_id
        payloads[kind] = raw
    return (
        _FakeDriveV3Service(metadata=metadata, revisions_by_id=revisions),
        file_ids,
        payloads,
    )


class _FakeAuthenticatedConnector(_AuthenticatedDriveReadbackConnector):
    def __init__(
        self,
        readbacks: dict[str, _DriveConnectorArtifactReadback],
        *,
        completed_at: str = "2026-07-20T01:10:00Z",
    ) -> None:
        self.readbacks = readbacks
        self.completed_at = completed_at
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def read_revision(
        self,
        *,
        artifact_kind: str,
        drive_file_id: str,
        metadata_fields: tuple[str, ...],
    ) -> _DriveConnectorArtifactReadback:
        self.calls.append((artifact_kind, drive_file_id, metadata_fields))
        return self.readbacks[artifact_kind]

    def completed_at_utc(self) -> str:
        return self.completed_at


def _readback_fixture(root: Path) -> dict:
    fixture = _terminal_fixture(root)
    pending = _preseal_lineage(fixture)
    handoff = _coordinate_existing(fixture, pending.latest_event_sha256)
    lineage = verify_registry(fixture["lineage"], artifact_root=fixture["root"])
    local = {
        "RUN": (fixture["terminal"] / "run.json").read_bytes(),
        "EVALUATION": (fixture["terminal"] / "evaluation.json").read_bytes(),
        "CELLS": (fixture["terminal"] / "cells.json").read_bytes(),
        "SEALED_STUDY": fixture["study_path"].read_bytes(),
        "TERMINAL_HANDOFF": canonical_json_bytes(handoff) + b"\n",
    }
    readbacks = {}
    drive_file_ids = {}
    for index, kind in enumerate(TRAINER_READBACK_KINDS, start=1):
        raw = local[kind]
        file_id = f"driveTrainerFile{index:02d}"
        metadata = {
            "id": file_id,
            "name": f"{index:02d}-{kind.lower()}.json",
            "size": str(len(raw)),
            "md5Checksum": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
            "modifiedTime": f"2026-07-20T01:0{index + 2}:00Z",
            "parents": [_parent_id(kind)],
            "trashed": False,
            "version": str(100 + index),
            "headRevisionId": f"driveRevision{index:02d}",
        }
        drive_file_ids[kind] = file_id
        readbacks[kind] = _DriveConnectorArtifactReadback(
            artifact_kind=kind,
            metadata_before=copy.deepcopy(metadata),
            metadata_after=copy.deepcopy(metadata),
            downloaded_bytes=raw,
        )
    connector = _FakeAuthenticatedConnector(readbacks)
    return {
        **fixture,
        "handoff": handoff,
        "lineage_snapshot": lineage,
        "local": local,
        "readbacks": readbacks,
        "drive_file_ids": drive_file_ids,
        "connector": connector,
    }


def _verify(fixture: dict) -> _VerifiedDriveTrainerReadback:
    # Unit-test seam for the future authenticated connector adapter.  No
    # serialized CLI input is permitted to call this private boundary.
    return _verify_authenticated_drive_trainer_readback(
        run=fixture["run"],
        evaluation=fixture["evaluation"],
        cells=fixture["cells"],
        sealed_study=fixture["sealed"],
        terminal_handoff_receipt=fixture["handoff"],
        lineage=fixture["lineage_snapshot"],
        expected_parent_ids=_parent_ids(),
        local_artifact_bytes=fixture["local"],
        drive_file_ids=fixture["drive_file_ids"],
        connector=fixture["connector"],
    )


def test_exact_downloaded_terminal_artifacts_create_typed_packet_evidence(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "valid")
    verified = _verify(fixture)

    assert verified.evidence["remote_verified"] is True
    assert verified.evidence["external_readback_attested"] is True
    assert verified.evidence["fixed_denominator"]["expected_cell_count"] == 4
    assert verified.evidence["downloaded_artifact_count"] == 5
    assert verified.evidence["proof_eligible"] is False
    assert verified.evidence["live_permission"] is False
    refs = _trainer_packet_drive_evidence_refs(
        verified,
        expected_run_sha256=fixture["run"]["run_sha256"],
        expected_study_sha256=fixture["sealed"]["study_sha256"],
        expected_evaluation_sha256=fixture["evaluation"]["evaluation_sha256"],
        expected_lineage_tip_sha256=fixture["lineage_snapshot"].latest_event_sha256,
        expected_cell_count=4,
    )
    assert {row["artifact_kind"] for row in refs} == set(TRAINER_READBACK_KINDS)
    assert len({row["readback_sha256"] for row in refs}) == 1
    parents = {
        row["artifact_kind"]: row["drive_parent_id"] for row in refs
    }
    assert parents == _parent_ids()
    assert [call[0] for call in fixture["connector"].calls] == list(
        TRAINER_READBACK_KINDS
    )


def test_typed_readback_rejects_post_verification_nested_mutation(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "post-verification-mutation")
    verified = _verify(fixture)
    verified.evidence["objects"][0]["drive_file_id"] = "substituteDriveFile999"
    body = {
        key: value
        for key, value in verified.evidence.items()
        if key != "readback_sha256"
    }
    verified.evidence["readback_sha256"] = canonical_json_sha256(body)

    with pytest.raises(
        DojoDriveRemoteEvidenceError,
        match="changed after connector verification",
    ):
        _trainer_packet_drive_evidence_refs(
            verified,
            expected_run_sha256=fixture["run"]["run_sha256"],
            expected_study_sha256=fixture["sealed"]["study_sha256"],
            expected_evaluation_sha256=fixture["evaluation"]["evaluation_sha256"],
            expected_lineage_tip_sha256=(
                fixture["lineage_snapshot"].latest_event_sha256
            ),
            expected_cell_count=4,
        )


def test_concrete_drive_v3_connector_reads_exact_head_revisions_and_cas_rechecks(
) -> None:
    service, file_ids, payloads = _drive_v3_service_fixture()
    connector = GoogleDriveV3TrainerReadbackConnector(service)

    for kind in TRAINER_READBACK_KINDS:
        readback = connector.read_revision(
            artifact_kind=kind,
            drive_file_id=file_ids[kind],
            metadata_fields=tuple(
                sorted(
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
            ),
        )
        assert readback.downloaded_bytes == payloads[kind]
        assert readback.metadata_before == readback.metadata_after

    completed_at = connector.completed_at_utc()
    assert datetime.fromisoformat(completed_at.replace("Z", "+00:00")).tzinfo
    assert not hasattr(connector, "create")
    assert not hasattr(connector, "update")
    assert not hasattr(connector, "delete")
    for index, kind in enumerate(TRAINER_READBACK_KINDS):
        first, revision, second = service.calls[index * 3 : index * 3 + 3]
        assert first[0] == second[0] == "files.get"
        assert first[1] == second[1]
        assert first[1]["fileId"] == file_ids[kind]
        assert first[1]["supportsAllDrives"] is True
        assert revision == (
            "revisions.get",
            {
                "fileId": file_ids[kind],
                "revisionId": service.metadata[file_ids[kind]]["headRevisionId"],
                "alt": "media",
            },
        )
    assert [name for name, _ in service.calls[15:]] == ["files.get"] * 5


def test_concrete_drive_v3_connector_mints_only_the_typed_research_readback(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "concrete-integration")
    metadata: dict[str, dict[str, Any]] = {}
    revisions: dict[tuple[str, str], bytes] = {}
    for kind in TRAINER_READBACK_KINDS:
        item = fixture["readbacks"][kind]
        row = copy.deepcopy(item.metadata_before)
        file_id = fixture["drive_file_ids"][kind]
        metadata[file_id] = row
        revisions[(file_id, row["headRevisionId"])] = item.downloaded_bytes
    fixture["connector"] = GoogleDriveV3TrainerReadbackConnector(
        _FakeDriveV3Service(metadata=metadata, revisions_by_id=revisions)
    )

    verified = _verify(fixture)

    assert verified.evidence["remote_verified"] is True
    assert verified.evidence["trainer_packet_eligible"] is True
    assert verified.evidence["proof_eligible"] is False
    assert verified.evidence["promotion_eligible"] is False
    assert verified.evidence["live_permission"] is False
    assert verified.evidence["order_authority"] == "NONE"
    assert verified.evidence["broker_mutation_allowed"] is False
    assert verified.evidence["resource_gate_unblock_allowed"] is False


def test_concrete_connector_rejects_caller_selected_metadata_fields_without_io(
) -> None:
    service, file_ids, _ = _drive_v3_service_fixture()
    connector = GoogleDriveV3TrainerReadbackConnector(service)

    with pytest.raises(ValueError, match="metadata field set is fixed"):
        connector.read_revision(
            artifact_kind="RUN",
            drive_file_id=file_ids["RUN"],
            metadata_fields=("id", "headRevisionId"),
        )
    assert service.calls == []


def test_parent_denominator_must_bind_exact_two_fixed_groups_before_io(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "parent-denominator")
    with pytest.raises(
        DojoDriveRemoteEvidenceError,
        match="two distinct fixed run/manifest groups",
    ):
        _verify_authenticated_drive_trainer_readback(
            run=fixture["run"],
            evaluation=fixture["evaluation"],
            cells=fixture["cells"],
            sealed_study=fixture["sealed"],
            terminal_handoff_receipt=fixture["handoff"],
            lineage=fixture["lineage_snapshot"],
            expected_parent_ids={
                kind: RUN_PARENT_ID for kind in TRAINER_READBACK_KINDS
            },
            local_artifact_bytes=fixture["local"],
            drive_file_ids=fixture["drive_file_ids"],
            connector=fixture["connector"],
        )
    assert fixture["connector"].calls == []


def test_readback_rejects_manifest_artifact_in_run_parent(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "manifest-in-run-parent")
    first = fixture["readbacks"]["SEALED_STUDY"]
    metadata = copy.deepcopy(first.metadata_before)
    metadata["parents"] = [RUN_PARENT_ID]
    fixture["readbacks"]["SEALED_STUDY"] = _DriveConnectorArtifactReadback(
        artifact_kind="SEALED_STUDY",
        metadata_before=copy.deepcopy(metadata),
        metadata_after=copy.deepcopy(metadata),
        downloaded_bytes=first.downloaded_bytes,
    )
    with pytest.raises(DojoDriveRemoteEvidenceError, match="parents do not equal"):
        _verify(fixture)


def test_concrete_connector_rejects_nonbyte_revision_media() -> None:
    service, file_ids, _ = _drive_v3_service_fixture()
    run_id = file_ids["RUN"]
    revision_id = service.metadata[run_id]["headRevisionId"]
    service.revisions_by_id[(run_id, revision_id)] = {"self_asserted": "bytes"}
    connector = GoogleDriveV3TrainerReadbackConnector(service)

    with pytest.raises(DojoDriveRemoteEvidenceError, match="exact bytes"):
        connector.read_revision(
            artifact_kind="RUN",
            drive_file_id=run_id,
            metadata_fields=tuple(sorted(service.metadata[run_id])),
        )


def test_concrete_connector_coordinator_recheck_rejects_late_revision_drift(
) -> None:
    service, file_ids, _ = _drive_v3_service_fixture()
    service.cas_drift_file_id = file_ids["RUN"]
    connector = GoogleDriveV3TrainerReadbackConnector(service)
    fields = tuple(sorted(service.metadata[file_ids["RUN"]]))
    for kind in TRAINER_READBACK_KINDS:
        connector.read_revision(
            artifact_kind=kind,
            drive_file_id=file_ids[kind],
            metadata_fields=fields,
        )

    with pytest.raises(
        DojoDriveRemoteEvidenceError,
        match="revision changed before batch completion",
    ):
        connector.completed_at_utc()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("download", "downloaded RUN bytes differ"),
        ("parent", "parents do not equal"),
        ("trashed", "is trashed"),
        ("version", "positive decimal"),
        ("revision", "head revision id"),
    ],
)
def test_download_metadata_revision_parent_and_trashed_are_fail_closed(
    tmp_path: Path, mutation: str, match: str
) -> None:
    fixture = _readback_fixture(tmp_path / mutation)
    first = fixture["readbacks"]["RUN"]
    metadata = copy.deepcopy(first.metadata_before)
    downloaded = first.downloaded_bytes
    if mutation == "download":
        downloaded += b"x"
    elif mutation == "parent":
        metadata["parents"] = ["foreignDriveParent123"]
    elif mutation == "trashed":
        metadata["trashed"] = True
    elif mutation == "version":
        metadata["version"] = "0"
    else:
        metadata["headRevisionId"] = "short"
    fixture["readbacks"]["RUN"] = _DriveConnectorArtifactReadback(
        artifact_kind=first.artifact_kind,
        metadata_before=copy.deepcopy(metadata),
        metadata_after=copy.deepcopy(metadata),
        downloaded_bytes=downloaded,
    )

    with pytest.raises(DojoDriveRemoteEvidenceError, match=match):
        _verify(fixture)


def test_partial_resealed_run_cannot_shrink_the_drive_denominator(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "partial")
    partial = copy.deepcopy(fixture["run"])
    partial["coordinates"] = partial["coordinates"][:1]
    partial["fixed_denominator"]["expected_cell_count"] = 1
    partial["fixed_denominator"]["observed_cell_count"] = 1
    body = {key: value for key, value in partial.items() if key != "run_sha256"}
    partial["run_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    fixture["run"] = partial

    with pytest.raises(
        DojoDriveRemoteEvidenceError, match="terminal readback denominator is invalid"
    ):
        _verify(fixture)


def test_plain_json_shape_cannot_replace_authenticated_connector(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "json-connector")
    fixture["connector"] = {
        "remote_verified": True,
        "readbacks": fixture["readbacks"],
    }

    with pytest.raises(
        DojoDriveRemoteEvidenceError,
        match="authenticated Drive connector adapter capability is required",
    ):
        _verify(fixture)


def test_bracketing_metadata_revision_drift_is_rejected(tmp_path: Path) -> None:
    fixture = _readback_fixture(tmp_path / "revision-drift")
    first = fixture["readbacks"]["RUN"]
    metadata_after = copy.deepcopy(first.metadata_after)
    metadata_after["version"] = "999"
    metadata_after["headRevisionId"] = "driveRevisionChanged99"
    fixture["readbacks"]["RUN"] = _DriveConnectorArtifactReadback(
        artifact_kind="RUN",
        metadata_before=first.metadata_before,
        metadata_after=metadata_after,
        downloaded_bytes=first.downloaded_bytes,
    )

    with pytest.raises(DojoDriveRemoteEvidenceError, match="revision changed"):
        _verify(fixture)


def test_pre_handoff_source_artifacts_are_valid_but_handoff_must_not_predate_it(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "timing")
    for index, kind in enumerate(TRAINER_READBACK_KINDS[:-1], start=1):
        item = fixture["readbacks"][kind]
        metadata = copy.deepcopy(item.metadata_before)
        metadata["modifiedTime"] = f"2026-07-19T23:5{index}:00Z"
        fixture["readbacks"][kind] = _DriveConnectorArtifactReadback(
            artifact_kind=kind,
            metadata_before=copy.deepcopy(metadata),
            metadata_after=copy.deepcopy(metadata),
            downloaded_bytes=item.downloaded_bytes,
        )
    terminal_item = fixture["readbacks"]["TERMINAL_HANDOFF"]
    terminal_metadata = copy.deepcopy(terminal_item.metadata_before)
    terminal_metadata["modifiedTime"] = "2026-07-20T01:00:02Z"
    fixture["readbacks"]["TERMINAL_HANDOFF"] = _DriveConnectorArtifactReadback(
        artifact_kind="TERMINAL_HANDOFF",
        metadata_before=copy.deepcopy(terminal_metadata),
        metadata_after=copy.deepcopy(terminal_metadata),
        downloaded_bytes=terminal_item.downloaded_bytes,
    )

    verified = _verify(fixture)
    assert verified.evidence["trainer_packet_eligible"] is True
    assert verified.evidence["proof_eligible"] is False
    assert verified.evidence["promotion_eligible"] is False
    assert verified.evidence["live_permission"] is False
    assert verified.evidence["order_authority"] == "NONE"
    assert verified.evidence["resource_gate_unblock_allowed"] is False

    terminal_metadata["modifiedTime"] = "2026-07-20T01:00:01Z"
    fixture["readbacks"]["TERMINAL_HANDOFF"] = _DriveConnectorArtifactReadback(
        artifact_kind="TERMINAL_HANDOFF",
        metadata_before=copy.deepcopy(terminal_metadata),
        metadata_after=copy.deepcopy(terminal_metadata),
        downloaded_bytes=terminal_item.downloaded_bytes,
    )
    with pytest.raises(
        DojoDriveRemoteEvidenceError,
        match="TERMINAL_HANDOFF modified time predates terminal handoff",
    ):
        _verify(fixture)


def test_any_drive_artifact_modified_after_readback_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "future-modified")
    item = fixture["readbacks"]["RUN"]
    metadata = copy.deepcopy(item.metadata_before)
    metadata["modifiedTime"] = "2026-07-20T01:10:01Z"
    fixture["readbacks"]["RUN"] = _DriveConnectorArtifactReadback(
        artifact_kind="RUN",
        metadata_before=copy.deepcopy(metadata),
        metadata_after=copy.deepcopy(metadata),
        downloaded_bytes=item.downloaded_bytes,
    )

    with pytest.raises(
        DojoDriveRemoteEvidenceError,
        match="RUN modified time postdates Drive readback",
    ):
        _verify(fixture)


def test_drive_readback_completion_must_follow_terminal_handoff(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "readback-before-handoff")
    fixture["connector"].completed_at = "2026-07-20T01:00:01Z"

    with pytest.raises(
        DojoDriveRemoteEvidenceError,
        match="Drive readback predates terminal handoff",
    ):
        _verify(fixture)


def test_self_constructed_typed_value_is_not_a_validator_capability(
    tmp_path: Path,
) -> None:
    fixture = _readback_fixture(tmp_path / "forged")
    verified = _verify(fixture)
    forged = _VerifiedDriveTrainerReadback(
        evidence=verified.evidence,
        _validation_marker=object(),
    )
    with pytest.raises(DojoDriveRemoteEvidenceError, match="typed Drive readback"):
        _trainer_packet_drive_evidence_refs(
            forged,
            expected_run_sha256=fixture["run"]["run_sha256"],
            expected_study_sha256=fixture["sealed"]["study_sha256"],
            expected_evaluation_sha256=fixture["evaluation"]["evaluation_sha256"],
            expected_lineage_tip_sha256=fixture["lineage_snapshot"].latest_event_sha256,
            expected_cell_count=4,
        )
