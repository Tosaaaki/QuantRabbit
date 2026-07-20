from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from quant_rabbit.dojo_candidate_lineage_registry import verify_registry
from quant_rabbit.dojo_drive_remote_evidence import (
    TRAINER_READBACK_KINDS,
    DojoDriveRemoteEvidenceError,
    _DriveDownloadedArtifact,
    _VerifiedDriveTrainerReadback,
    _trainer_packet_drive_evidence_refs,
    _verify_authenticated_drive_trainer_readback,
)
from quant_rabbit.dojo_terminal_handoff import canonical_json_bytes
from test_dojo_terminal_handoff import (
    _coordinate_existing,
    _preseal_lineage,
    _terminal_fixture,
)


PARENT_ID = "driveTrainerParent123"


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
    readbacks = []
    for index, kind in enumerate(TRAINER_READBACK_KINDS, start=1):
        raw = local[kind]
        readbacks.append(
            _DriveDownloadedArtifact(
                artifact_kind=kind,
                metadata={
                    "id": f"driveTrainerFile{index:02d}",
                    "name": f"{index:02d}-{kind.lower()}.json",
                    "size": str(len(raw)),
                    "md5Checksum": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
                    "modifiedTime": f"2026-07-20T01:0{index + 2}:00Z",
                    "parents": [PARENT_ID],
                    "trashed": False,
                    "version": str(100 + index),
                    "headRevisionId": f"driveRevision{index:02d}",
                },
                downloaded_bytes=raw,
            )
        )
    return {
        **fixture,
        "handoff": handoff,
        "lineage_snapshot": lineage,
        "local": local,
        "readbacks": readbacks,
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
        expected_parent_id=PARENT_ID,
        local_artifact_bytes=fixture["local"],
        readbacks=fixture["readbacks"],
        readback_at_utc="2026-07-20T01:10:00Z",
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
    first = fixture["readbacks"][0]
    metadata = copy.deepcopy(first.metadata)
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
    fixture["readbacks"][0] = _DriveDownloadedArtifact(
        artifact_kind=first.artifact_kind,
        metadata=metadata,
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
