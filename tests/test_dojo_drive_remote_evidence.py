from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from quant_rabbit.dojo_candidate_lineage_registry import CandidateLineageSnapshot
from quant_rabbit.dojo_drive_remote_evidence import (
    CellCoordinate,
    DojoDriveRemoteEvidenceError,
    LocalArchiveBinding,
    REMOTE_INDEX_CONTRACT,
    REMOTE_RECEIPT_CONTRACT,
    canonical_artifact_bytes,
    canonical_json_sha256,
    local_archive_binding_from_receipts,
    verify_remote_index_bundle,
    verify_remote_receipt_bytes,
)


PARENT_ID = "driveParent123"
RESULT_AT = "2026-07-20T00:00:00+00:00"


def _sealed(body: dict, field: str) -> dict:
    return {**body, field: canonical_json_sha256(body)}


def _run() -> dict:
    coordinates = []
    for index, (candidate, intrabar, arm) in enumerate(
        (("C1", "OHLC", "BASE"), ("C1", "OLHC", "STRESS")), start=1
    ):
        coordinates.append(
            {
                "candidate_id": candidate,
                "intrabar": intrabar,
                "cost_arm": arm,
                "status": "COMPLETE",
                "main_session_dir": f"/run/session-{index}",
                "main_error": None,
                "lopo_replay_complete": True,
                "lopo": [],
                "cell_sha256": f"{index}" * 64,
            }
        )
    body = {
        "contract": "QR_DOJO_BOT_TRAINER_RUN_V1",
        "schema_version": 1,
        "study_sha256": "a" * 64,
        "status": "COMPLETE",
        "corpus": {},
        "fixed_denominator": {
            "expected_cell_count": 2,
            "observed_cell_count": 2,
            "failed_cell_count": 0,
            "dropped_cell_count": 0,
            "coordinate_receipts_complete": True,
            "execution_success_complete": True,
        },
        "coordinates": coordinates,
        "cells_path": "/run/cells.json",
        "evaluation_path": "/run/evaluation.json",
        "evaluation_sha256": "b" * 64,
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return _sealed(body, "run_sha256")


def _lineage(run: dict) -> CandidateLineageSnapshot:
    result = {
        "attempt_ordinal": 1,
        "study_sha256": run["study_sha256"],
        "evaluation_sha256": run["evaluation_sha256"],
        "evaluation_artifact_relpath": "attempt-1/evaluation.json",
        "evaluation_artifact_sha256": "c" * 64,
        "evaluation_artifact_size_bytes": 1234,
        "verified_trainer_evaluation": True,
    }
    event_body = {
        "contract": "QR_DOJO_CANDIDATE_LINEAGE_EVENT_V1",
        "schema_version": 1,
        "registry_id": "registry-1",
        "sequence": 2,
        "event_type": "RESULT_BOUND",
        "event_at_utc": RESULT_AT,
        "previous_event_sha256": "d" * 64,
        "body": result,
        "external_witness_status": "ABSENT",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    event = _sealed(event_body, "event_sha256")
    return CandidateLineageSnapshot(
        registry_id="registry-1",
        lineage_prefix="C",
        event_count=3,
        latest_sequence=2,
        latest_event_sha256=event["event_sha256"],
        latest_event_at_utc=RESULT_AT,
        studies=({"attempt_ordinal": 1, "study_sha256": run["study_sha256"]},),
        results=(result,),
        cumulative_unique_config_sha256s=("e" * 64,),
        cumulative_unique_proposal_sha256s=("f" * 64,),
        events=({}, {}, event),
    )


def _lineage_binding(lineage: CandidateLineageSnapshot) -> dict:
    return {
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "attempt_ordinal": 1,
        "result_event_sequence": lineage.latest_sequence,
        "result_event_sha256": lineage.latest_event_sha256,
        "lineage_tip_sequence": lineage.latest_sequence,
        "lineage_tip_sha256": lineage.latest_event_sha256,
    }


def _archives(run: dict) -> list[LocalArchiveBinding]:
    result = []
    for index, coordinate in enumerate(run["coordinates"], start=1):
        typed = CellCoordinate(
            coordinate["candidate_id"],
            coordinate["intrabar"],
            coordinate["cost_arm"],
            coordinate["cell_sha256"],
        )
        plan_sha = f"{index + 1}" * 64
        result.append(
            LocalArchiveBinding(
                coordinate=typed,
                run_sha256=run["run_sha256"],
                study_sha256=run["study_sha256"],
                evaluation_sha256=run["evaluation_sha256"],
                plan_sha256=plan_sha,
                finalization_sha256=f"{index + 3}" * 64,
                content_tree_sha256=f"{index + 5}" * 64,
                archive_sha256=f"{index + 7}" * 64,
                archive_size_bytes=1000 + index,
                archive_md5_checksum=f"{index + 1}" * 32,
                archive_name=f"cell-{typed.chunk_id}-{plan_sha}.tar.zst",
            )
        )
    return result


def _handoff(run: dict, lineage: CandidateLineageSnapshot) -> dict:
    def artifact(relpath: str, digest: str, size: int) -> dict:
        return {
            "artifact_relpath": relpath,
            "artifact_sha256": digest,
            "artifact_size_bytes": size,
        }

    terminal = {
        "terminal_dir_relpath": "terminal",
        "run": artifact("terminal/run.json", "1" * 64, 1001),
        "evaluation": artifact("terminal/evaluation.json", "2" * 64, 1002),
        "cells": artifact("terminal/cells.json", "3" * 64, 1003),
        "run_sha256": run["run_sha256"],
        "evaluation_sha256": run["evaluation_sha256"],
        "study_sha256": run["study_sha256"],
        "status": run["status"],
        "expected_cell_count": run["fixed_denominator"]["expected_cell_count"],
        "observed_cell_count": run["fixed_denominator"]["observed_cell_count"],
        "failed_cell_count": run["fixed_denominator"]["failed_cell_count"],
    }
    after = {
        "registry_id": lineage.registry_id,
        "lineage_prefix": lineage.lineage_prefix,
        "latest_sequence": lineage.latest_sequence,
        "latest_event_sha256": lineage.latest_event_sha256,
        "attempt_ordinal": 1,
        "study_sha256": run["study_sha256"],
        "evaluation_sha256": run["evaluation_sha256"],
        "evaluation_artifact_sha256": terminal["evaluation"]["artifact_sha256"],
        "evaluation_artifact_size_bytes": terminal["evaluation"]["artifact_size_bytes"],
        "result_event_sha256": lineage.latest_event_sha256,
        "result_event_sequence": lineage.latest_sequence,
        "result_binding_sha256": "4" * 64,
    }
    body = {
        "contract": "QR_DOJO_TERMINAL_HANDOFF_RECEIPT_V1",
        "schema_version": 1,
        "sequence": 0,
        "recorded_at_utc": RESULT_AT,
        "previous_receipt_sha256": None,
        "binding_timing_classification": (
            "LINEAGE_PRESENT_AT_HANDOFF_NO_PREREGISTRATION_CLAIM"
        ),
        "lineage_branch": "EXISTING_PENDING_STUDY_BOUND",
        "terminal_bundle": terminal,
        "sealed_study": {
            **artifact("study.json", "5" * 64, 1005),
            "study_sha256": run["study_sha256"],
            "attempt_ordinal": 1,
        },
        "lineage_before": {
            "state": "VERIFIED",
            "registry_id": lineage.registry_id,
            "lineage_prefix": lineage.lineage_prefix,
            "latest_sequence": 1,
            "latest_event_sha256": "d" * 64,
            "study_count": 1,
            "result_count": 0,
        },
        "lineage_after": after,
        "checks": {
            "terminal_bundle_complete": True,
            "run_failure_absent": True,
            "terminal_status_valid": True,
            "fixed_denominator_complete": True,
            "study_sha_consistent": True,
            "authority_research_only": True,
            "evaluation_rebuilt_from_cells": True,
            "full_terminal_bundle_required": True,
            "lineage_result_bound": True,
        },
        "limitations": [
            "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
            "LOCAL_FIRST_WRITE_ONLY_NO_EXTERNAL_MONOTONIC_WITNESS",
            "RETROSPECTIVE_ADMIN_BINDING_NEVER_PROVES_PREREGISTRATION",
            "LINEAGE_PRESENT_AT_HANDOFF_DOES_NOT_PROVE_PREREGISTRATION",
            "NO_MODEL_RUNNER_DRIVE_BROKER_ORDER_OR_LIVE_AUTHORITY",
            "FULL_TERMINAL_BUNDLE_REQUIRED_EVALUATION_ONLY_BINDING_FORBIDDEN",
        ],
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return _sealed(body, "receipt_sha256")


def _handoff_binding(
    run: dict, lineage: CandidateLineageSnapshot, handoff: dict
) -> dict:
    terminal = handoff["terminal_bundle"]
    return {
        "receipt_sha256": handoff["receipt_sha256"],
        "binding_timing_classification": handoff["binding_timing_classification"],
        "run_semantic_sha256": run["run_sha256"],
        "fixed_denominator": run["fixed_denominator"],
        "terminal_run_artifact_sha256": terminal["run"]["artifact_sha256"],
        "terminal_run_artifact_size_bytes": terminal["run"]["artifact_size_bytes"],
        "terminal_evaluation_artifact_sha256": terminal["evaluation"][
            "artifact_sha256"
        ],
        "terminal_evaluation_artifact_size_bytes": terminal["evaluation"][
            "artifact_size_bytes"
        ],
        "terminal_cells_artifact_sha256": terminal["cells"]["artifact_sha256"],
        "terminal_cells_artifact_size_bytes": terminal["cells"]["artifact_size_bytes"],
        "lineage_tip_sequence": lineage.latest_sequence,
        "lineage_tip_sha256": lineage.latest_event_sha256,
        "result_event_sequence": lineage.latest_sequence,
        "result_event_sha256": lineage.latest_event_sha256,
    }


def _receipt(
    run: dict,
    lineage: CandidateLineageSnapshot,
    handoff: dict,
    archive: LocalArchiveBinding,
    index: int,
) -> tuple[dict, bytes]:
    body = {
        "contract": REMOTE_RECEIPT_CONTRACT,
        "schema_version": 1,
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "run_sha256": run["run_sha256"],
        "study_sha256": run["study_sha256"],
        "evaluation_sha256": run["evaluation_sha256"],
        "lineage_binding": _lineage_binding(lineage),
        "terminal_handoff_binding": _handoff_binding(run, lineage, handoff),
        "coordinate": archive.coordinate.as_dict(),
        "local_archive": {
            "plan_sha256": archive.plan_sha256,
            "finalization_sha256": archive.finalization_sha256,
            "content_tree_sha256": archive.content_tree_sha256,
            "archive_sha256": archive.archive_sha256,
            "archive_size_bytes": archive.archive_size_bytes,
            "archive_md5_checksum": archive.archive_md5_checksum,
            "archive_name": archive.archive_name,
        },
        "drive_archive": {
            "file_id": f"driveFile{index:04d}",
            "parent_id": PARENT_ID,
            "name": archive.archive_name,
            "size_bytes": archive.archive_size_bytes,
            "md5_checksum": archive.archive_md5_checksum,
            "content_sha256": archive.archive_sha256,
            "modified_time": f"2026-07-20T00:0{index}:00Z",
        },
        "checked_at_utc": f"2026-07-20T00:1{index}:00Z",
        "remote_metadata_consistent": True,
        "external_readback_required": True,
        "external_readback_attested": False,
        "remote_verified": False,
        "source_deleted": False,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
        "trainer_unblock_allowed": False,
        "resource_gate_unblock_allowed": False,
    }
    receipt = _sealed(body, "receipt_sha256")
    return receipt, canonical_artifact_bytes(receipt)


def _index(
    run: dict,
    lineage: CandidateLineageSnapshot,
    handoff: dict,
    receipts: list[tuple[dict, bytes]],
) -> tuple[dict, bytes]:
    entries = []
    for receipt, raw in receipts:
        local = receipt["local_archive"]
        drive = receipt["drive_archive"]
        entries.append(
            {
                "coordinate": receipt["coordinate"],
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
        )
    entries.sort(
        key=lambda row: (
            row["coordinate"]["candidate_id"],
            row["coordinate"]["intrabar"],
            row["coordinate"]["cost_arm"],
        )
    )
    body = {
        "contract": REMOTE_INDEX_CONTRACT,
        "schema_version": 1,
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "run_sha256": run["run_sha256"],
        "study_sha256": run["study_sha256"],
        "evaluation_sha256": run["evaluation_sha256"],
        "lineage_binding": _lineage_binding(lineage),
        "terminal_handoff_binding": _handoff_binding(run, lineage, handoff),
        "expected_cell_count": len(entries),
        "remote_receipt_count": len(entries),
        "entries": entries,
        "remote_metadata_consistent": True,
        "external_readback_required": True,
        "external_readback_attested": False,
        "remote_verified": False,
        "source_deleted": False,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
        "trainer_unblock_allowed": False,
        "resource_gate_unblock_allowed": False,
    }
    index = _sealed(body, "index_sha256")
    return index, canonical_artifact_bytes(index)


@pytest.fixture
def bundle():
    run = _run()
    lineage = _lineage(run)
    handoff = _handoff(run, lineage)
    archives = _archives(run)
    receipts = [
        _receipt(run, lineage, handoff, archive, index)
        for index, archive in enumerate(archives, start=1)
    ]
    index, index_bytes = _index(run, lineage, handoff, receipts)
    return run, lineage, handoff, archives, receipts, index, index_bytes


def _reseal(value: dict, field: str) -> bytes:
    body = {key: item for key, item in value.items() if key != field}
    return canonical_artifact_bytes(_sealed(body, field))


def test_complete_bundle_binds_every_cell_and_keeps_all_authority_false(bundle) -> None:
    run, lineage, handoff, archives, receipts, index, index_bytes = bundle
    verified = verify_remote_index_bundle(
        index_bytes=index_bytes,
        receipt_bytes=[raw for _, raw in reversed(receipts)],
        expected_run=run,
        lineage=lineage,
        terminal_handoff_receipt=handoff,
        local_archives=archives,
        expected_archive_parent_id=PARENT_ID,
    )

    assert verified.index == index
    assert len(verified.receipts) == 2
    assert verified.index_content_sha256 == hashlib.sha256(index_bytes).hexdigest()
    assert verified.index_content_size_bytes == len(index_bytes)
    assert (
        verified.index["terminal_handoff_binding"]["fixed_denominator"]
        == run["fixed_denominator"]
    )
    assert verified.index["remote_metadata_consistent"] is True
    assert verified.index["external_readback_attested"] is False
    assert verified.index["remote_verified"] is False
    assert verified.remote_metadata_consistent is True
    assert verified.external_readback_required is True
    assert verified.external_readback_attested is False
    assert verified.remote_verified is False
    assert verified.proof_eligible is False
    assert verified.promotion_eligible is False
    assert verified.live_permission is False
    assert verified.order_authority == "NONE"
    assert verified.broker_mutation_allowed is False
    assert verified.trainer_unblock_allowed is False
    assert verified.resource_gate_unblock_allowed is False


def test_one_receipt_verifies_exact_drive_and_lineage_binding(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, _ = bundle
    receipt = verify_remote_receipt_bytes(
        receipts[0][1],
        expected_run=run,
        lineage=lineage,
        terminal_handoff_receipt=handoff,
        local_archive=archives[0],
        expected_archive_parent_id=PARENT_ID,
    )
    assert receipt["receipt_sha256"] == receipts[0][0]["receipt_sha256"]
    assert receipt["drive_archive"]["content_sha256"] == archives[0].archive_sha256


@pytest.mark.parametrize("raw", [b"", b"{}", b'{"x":NaN}\n'])
def test_empty_noncanonical_and_nonfinite_receipt_bytes_fail_closed(
    bundle, raw
) -> None:
    run, lineage, handoff, archives, _, _, _ = bundle
    with pytest.raises(DojoDriveRemoteEvidenceError):
        verify_remote_receipt_bytes(
            raw,
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archive=archives[0],
            expected_archive_parent_id=PARENT_ID,
        )


def test_pretty_printed_and_duplicate_key_receipt_bytes_are_rejected(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, _ = bundle
    pretty = json.dumps(receipts[0][0], indent=2).encode() + b"\n"
    duplicate = b'{"contract":"a","contract":"b"}\n'
    for raw in (pretty, duplicate):
        with pytest.raises(DojoDriveRemoteEvidenceError):
            verify_remote_receipt_bytes(
                raw,
                expected_run=run,
                lineage=lineage,
                terminal_handoff_receipt=handoff,
                local_archive=archives[0],
                expected_archive_parent_id=PARENT_ID,
            )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("remote_verified",), True),
        (("external_readback_attested",), True),
        (("proof_eligible",), True),
        (("run_sha256",), "f" * 64),
        (("drive_archive", "parent_id"), "foreignParent9"),
        (("drive_archive", "size_bytes"), 99),
        (("drive_archive", "md5_checksum"), "f" * 32),
        (("drive_archive", "content_sha256"), "f" * 64),
    ],
)
def test_resealed_unverified_foreign_or_tampered_receipt_is_rejected(
    bundle, path, value
) -> None:
    run, lineage, handoff, archives, receipts, _, _ = bundle
    mutated = deepcopy(receipts[0][0])
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    raw = _reseal(mutated, "receipt_sha256")
    with pytest.raises(DojoDriveRemoteEvidenceError):
        verify_remote_receipt_bytes(
            raw,
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archive=archives[0],
            expected_archive_parent_id=PARENT_ID,
        )


def test_unresealed_receipt_tamper_is_rejected_by_canonical_seal(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, _ = bundle
    tampered = deepcopy(receipts[0][0])
    tampered["checked_at_utc"] = "2026-07-20T01:00:00Z"
    with pytest.raises(DojoDriveRemoteEvidenceError, match="SHA-256 mismatch"):
        verify_remote_receipt_bytes(
            canonical_artifact_bytes(tampered),
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archive=archives[0],
            expected_archive_parent_id=PARENT_ID,
        )


def test_omitted_or_duplicate_receipt_fails_before_index_acceptance(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, index_bytes = bundle
    for raw_receipts in ([receipts[0][1]], [receipts[0][1], receipts[0][1]]):
        with pytest.raises(DojoDriveRemoteEvidenceError):
            verify_remote_index_bundle(
                index_bytes=index_bytes,
                receipt_bytes=raw_receipts,
                expected_run=run,
                lineage=lineage,
                terminal_handoff_receipt=handoff,
                local_archives=archives,
                expected_archive_parent_id=PARENT_ID,
            )


def test_duplicate_drive_file_id_is_rejected_even_with_distinct_receipt_seals(
    bundle,
) -> None:
    run, lineage, handoff, archives, receipts, _, index_bytes = bundle
    second = deepcopy(receipts[1][0])
    second["drive_archive"]["file_id"] = receipts[0][0]["drive_archive"]["file_id"]
    second_raw = _reseal(second, "receipt_sha256")
    with pytest.raises(DojoDriveRemoteEvidenceError, match="duplicate"):
        verify_remote_index_bundle(
            index_bytes=index_bytes,
            receipt_bytes=[receipts[0][1], second_raw],
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archives=archives,
            expected_archive_parent_id=PARENT_ID,
        )


def test_omitted_reordered_or_tampered_index_entry_is_rejected(bundle) -> None:
    run, lineage, handoff, archives, receipts, index, _ = bundle
    variants = []
    omitted = deepcopy(index)
    omitted["entries"].pop()
    omitted["expected_cell_count"] = 1
    omitted["remote_receipt_count"] = 1
    variants.append(_reseal(omitted, "index_sha256"))
    reversed_index = deepcopy(index)
    reversed_index["entries"].reverse()
    variants.append(_reseal(reversed_index, "index_sha256"))
    tampered = deepcopy(index)
    tampered["entries"][0]["drive_size_bytes"] += 1
    variants.append(_reseal(tampered, "index_sha256"))
    for index_bytes in variants:
        with pytest.raises(DojoDriveRemoteEvidenceError):
            verify_remote_index_bundle(
                index_bytes=index_bytes,
                receipt_bytes=[raw for _, raw in receipts],
                expected_run=run,
                lineage=lineage,
                terminal_handoff_receipt=handoff,
                local_archives=archives,
                expected_archive_parent_id=PARENT_ID,
            )


def test_unverified_and_authority_forged_index_is_rejected(bundle) -> None:
    run, lineage, handoff, archives, receipts, index, _ = bundle
    for field, value in (("remote_verified", True), ("live_permission", True)):
        mutated = deepcopy(index)
        mutated[field] = value
        with pytest.raises(DojoDriveRemoteEvidenceError):
            verify_remote_index_bundle(
                index_bytes=_reseal(mutated, "index_sha256"),
                receipt_bytes=[raw for _, raw in receipts],
                expected_run=run,
                lineage=lineage,
                terminal_handoff_receipt=handoff,
                local_archives=archives,
                expected_archive_parent_id=PARENT_ID,
            )


def test_index_bytes_and_schema_have_one_canonical_representation(bundle) -> None:
    run, lineage, handoff, archives, receipts, index, _ = bundle
    extra = deepcopy(index)
    extra["unexpected"] = None
    variants = (
        json.dumps(index, indent=2).encode() + b"\n",
        _reseal(extra, "index_sha256"),
    )
    for index_bytes in variants:
        with pytest.raises(DojoDriveRemoteEvidenceError):
            verify_remote_index_bundle(
                index_bytes=index_bytes,
                receipt_bytes=[raw for _, raw in receipts],
                expected_run=run,
                lineage=lineage,
                terminal_handoff_receipt=handoff,
                local_archives=archives,
                expected_archive_parent_id=PARENT_ID,
            )


def test_local_archive_denominator_cannot_omit_or_duplicate_a_cell(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, index_bytes = bundle
    for local in ([archives[0]], [archives[0], archives[0]]):
        with pytest.raises(DojoDriveRemoteEvidenceError):
            verify_remote_index_bundle(
                index_bytes=index_bytes,
                receipt_bytes=[raw for _, raw in receipts],
                expected_run=run,
                lineage=lineage,
                terminal_handoff_receipt=handoff,
                local_archives=local,
                expected_archive_parent_id=PARENT_ID,
            )


def test_lineage_result_must_be_the_current_tip(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, index_bytes = bundle
    stale = CandidateLineageSnapshot(
        **{
            **lineage.__dict__,
            "event_count": 4,
            "latest_sequence": 3,
            "latest_event_sha256": "9" * 64,
            "events": (*lineage.events, {}),
        }
    )
    with pytest.raises(DojoDriveRemoteEvidenceError, match="current tip"):
        verify_remote_index_bundle(
            index_bytes=index_bytes,
            receipt_bytes=[raw for _, raw in receipts],
            expected_run=run,
            lineage=stale,
            terminal_handoff_receipt=handoff,
            local_archives=archives,
            expected_archive_parent_id=PARENT_ID,
        )


def test_run_reseal_cannot_add_or_omit_a_coordinate(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, index_bytes = bundle
    partial = deepcopy(run)
    partial["coordinates"].pop()
    body = {key: value for key, value in partial.items() if key != "run_sha256"}
    partial = _sealed(body, "run_sha256")
    with pytest.raises(DojoDriveRemoteEvidenceError, match="coordinate denominator"):
        verify_remote_index_bundle(
            index_bytes=index_bytes,
            receipt_bytes=[raw for _, raw in receipts],
            expected_run=partial,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archives=archives,
            expected_archive_parent_id=PARENT_ID,
        )


def test_self_rehashed_shrunken_run_cannot_escape_terminal_handoff_denominator(
    bundle,
) -> None:
    run, lineage, handoff, archives, receipts, _, index_bytes = bundle
    shrunken = deepcopy(run)
    shrunken["coordinates"] = shrunken["coordinates"][:1]
    shrunken["fixed_denominator"] = {
        **shrunken["fixed_denominator"],
        "expected_cell_count": 1,
        "observed_cell_count": 1,
    }
    shrunken = _sealed(
        {key: value for key, value in shrunken.items() if key != "run_sha256"},
        "run_sha256",
    )

    with pytest.raises(
        DojoDriveRemoteEvidenceError, match="exact run semantic SHA/denominator"
    ):
        verify_remote_index_bundle(
            index_bytes=index_bytes,
            receipt_bytes=[raw for _, raw in receipts],
            expected_run=shrunken,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archives=archives[:1],
            expected_archive_parent_id=PARENT_ID,
        )


def test_terminal_handoff_receipt_and_its_lineage_tip_are_mandatory(bundle) -> None:
    run, lineage, handoff, archives, receipts, _, index_bytes = bundle
    with pytest.raises(
        DojoDriveRemoteEvidenceError, match="handoff receipt is invalid"
    ):
        verify_remote_index_bundle(
            index_bytes=index_bytes,
            receipt_bytes=[raw for _, raw in receipts],
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt={},
            local_archives=archives,
            expected_archive_parent_id=PARENT_ID,
        )

    foreign_tip = deepcopy(handoff)
    foreign_tip["lineage_after"]["latest_event_sha256"] = "9" * 64
    foreign_tip["lineage_after"]["result_event_sha256"] = "9" * 64
    foreign_tip = json.loads(_reseal(foreign_tip, "receipt_sha256"))
    with pytest.raises(DojoDriveRemoteEvidenceError, match="current lineage"):
        verify_remote_index_bundle(
            index_bytes=index_bytes,
            receipt_bytes=[raw for _, raw in receipts],
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=foreign_tip,
            local_archives=archives,
            expected_archive_parent_id=PARENT_ID,
        )


def test_caller_cannot_manufacture_remote_verification_by_self_resealing(
    bundle,
) -> None:
    run, lineage, handoff, archives, receipts, index, _ = bundle
    forged_receipt = deepcopy(receipts[0][0])
    forged_receipt["remote_verified"] = True
    forged_receipt["external_readback_attested"] = True
    with pytest.raises(DojoDriveRemoteEvidenceError, match="externally attested"):
        verify_remote_receipt_bytes(
            _reseal(forged_receipt, "receipt_sha256"),
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archive=archives[0],
            expected_archive_parent_id=PARENT_ID,
        )

    forged_index = deepcopy(index)
    forged_index["remote_verified"] = True
    forged_index["external_readback_attested"] = True
    with pytest.raises(DojoDriveRemoteEvidenceError, match="externally attested"):
        verify_remote_index_bundle(
            index_bytes=_reseal(forged_index, "index_sha256"),
            receipt_bytes=[raw for _, raw in receipts],
            expected_run=run,
            lineage=lineage,
            terminal_handoff_receipt=handoff,
            local_archives=archives,
            expected_archive_parent_id=PARENT_ID,
        )


def test_local_binding_is_derived_from_exact_archiver_plan_and_finalization(
    bundle,
) -> None:
    run, _, _, archives, _, _, _ = bundle
    expected = archives[0]
    files = [{"path": "run.json", "size_bytes": 10, "sha256": "a" * 64}]
    plan_body = {
        "contract": "QR_DOJO_DRIVE_ARCHIVE_PLAN_V1",
        "schema_version": 1,
        "source_run_root": "/run",
        "destination_root": "/archive",
        "chunk_kind": "cell",
        "chunk_id": expected.coordinate.chunk_id,
        "terminal_run": {
            "contract": run["contract"],
            "status": run["status"],
            "run_sha256": run["run_sha256"],
            "study_sha256": run["study_sha256"],
            "evaluation_sha256": run["evaluation_sha256"],
            "classification": run["classification"],
            "fixed_denominator": run["fixed_denominator"],
        },
        "file_count": 1,
        "total_source_bytes": 10,
        "content_tree_sha256": canonical_json_sha256(files),
        "files": files,
        "archive_format": "POSIX_PAX_TAR_ZSTD",
        "archive_member_prefix": "run/",
        "source_deletion_allowed": False,
        "source_deleted": False,
        "remote_verification": {
            "status": "NOT_REQUESTED",
            "remote_verified": False,
            "metadata_receipt_sha256": None,
        },
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    plan = _sealed(plan_body, "plan_sha256")
    archive_name = f"cell-{expected.coordinate.chunk_id}-{plan['plan_sha256']}.tar.zst"
    final_body = {
        "contract": "QR_DOJO_DRIVE_ARCHIVE_FINALIZATION_V1",
        "schema_version": 1,
        "plan_path": "/archive/plans/plan.json",
        "plan_sha256": plan["plan_sha256"],
        "content_tree_sha256": plan["content_tree_sha256"],
        "chunk_kind": "cell",
        "chunk_id": expected.coordinate.chunk_id,
        "archive_path": f"/archive/archives/{archive_name}",
        "archive_sha256": expected.archive_sha256,
        "archive_size_bytes": expected.archive_size_bytes,
        "file_count": 1,
        "total_source_bytes": 10,
        "local_payload_verified": True,
        "atomic_publish_complete": True,
        "source_deletion_allowed": False,
        "source_deleted": False,
        "remote_verification": {
            "status": "NOT_REQUESTED",
            "remote_verified": False,
            "metadata_receipt_sha256": None,
        },
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    finalization = _sealed(final_body, "finalization_sha256")

    binding = local_archive_binding_from_receipts(
        plan=plan,
        finalization=finalization,
        archive_md5_checksum=expected.archive_md5_checksum,
    )

    assert binding.coordinate.chunk_id == expected.coordinate.chunk_id
    assert binding.coordinate.cell_sha256 == ""
    assert binding.run_sha256 == run["run_sha256"]
    assert binding.plan_sha256 == plan["plan_sha256"]
    assert binding.finalization_sha256 == finalization["finalization_sha256"]
    assert binding.archive_name == archive_name

    forged = deepcopy(finalization)
    forged["remote_verification"]["remote_verified"] = True
    forged["remote_verification"]["status"] = "REMOTE_VERIFIED"
    with pytest.raises(DojoDriveRemoteEvidenceError, match="self-asserts"):
        local_archive_binding_from_receipts(
            plan=plan,
            finalization=json.loads(_reseal(forged, "finalization_sha256")),
            archive_md5_checksum=expected.archive_md5_checksum,
        )
