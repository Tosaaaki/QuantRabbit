from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_rabbit.dojo_candidate_lineage_registry import (
    bind_result,
    initialize_registry,
    seal_study_attempt,
    verify_registry,
)
from quant_rabbit.dojo_terminal_handoff import (
    ABSENT_CAS_TOKEN,
    LINEAGE_PRESENT_AT_HANDOFF,
    RETROSPECTIVE_ADMIN_BINDING,
    DojoTerminalHandoffError,
    canonical_json_bytes,
    coordinate_terminal_handoff,
    receipt_store_status,
    verify_handoff_receipt,
    verify_receipt_store,
)
from test_dojo_ai_trainer_packet import (
    _canonical_sha,
    _cells,
    _evaluation,
    _run,
    _sealed_study,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run-dojo-terminal-handoff.py"


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _terminal_fixture(root: Path) -> dict:
    root.mkdir(parents=True)
    sealed = _sealed_study()
    cells = _cells(sealed)
    evaluation = _evaluation(sealed)
    run = _run(sealed, evaluation, cells)
    study_path = _write_json(root / "artifacts" / "study.json", sealed)
    terminal = root / "terminal"
    _write_json(terminal / "run.json", run)
    _write_json(terminal / "evaluation.json", evaluation)
    _write_json(terminal / "cells.json", cells)
    return {
        "root": root,
        "sealed": sealed,
        "cells": cells,
        "evaluation": evaluation,
        "run": run,
        "study_path": study_path,
        "terminal": terminal,
        "lineage": root / "lineage-events",
        "receipts": root / "handoff-events",
    }


def _preseal_lineage(fixture: dict) -> object:
    snapshot = initialize_registry(
        fixture["lineage"],
        artifact_root=fixture["root"],
        registry_id="qr-terminal-handoff",
        lineage_prefix="qr-",
        created_by="pytest",
        event_at_utc="2026-07-20T01:00:00Z",
    )
    return seal_study_attempt(
        fixture["lineage"],
        artifact_root=fixture["root"],
        sealed_study_path=fixture["study_path"],
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-20T01:00:01Z",
    )


def _coordinate_existing(fixture: dict, tip: str) -> dict:
    return coordinate_terminal_handoff(
        terminal_dir=fixture["terminal"],
        sealed_study_path=fixture["study_path"],
        lineage_events_dir=fixture["lineage"],
        artifact_root=fixture["root"],
        receipt_events_dir=fixture["receipts"],
        expected_lineage_tip_sha256=tip,
        expected_receipt_tip_sha256=ABSENT_CAS_TOKEN,
        binding_timing_classification=LINEAGE_PRESENT_AT_HANDOFF,
        event_at_utc="2026-07-20T01:00:02Z",
    )


def test_existing_pending_lineage_binds_only_after_full_bundle_validation(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path / "existing")
    pending = _preseal_lineage(fixture)

    receipt = _coordinate_existing(fixture, pending.latest_event_sha256)

    assert receipt["binding_timing_classification"] == LINEAGE_PRESENT_AT_HANDOFF
    assert receipt["lineage_branch"] == "EXISTING_PENDING_STUDY_BOUND"
    assert receipt["terminal_bundle"]["expected_cell_count"] == 4
    assert receipt["terminal_bundle"]["observed_cell_count"] == 4
    assert receipt["terminal_bundle"]["failed_cell_count"] == 4
    assert all(receipt["checks"].values())
    assert receipt["proof_eligible"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["live_permission"] is False
    assert receipt["order_authority"] == "NONE"
    assert receipt["broker_mutation_allowed"] is False
    assert (
        "LINEAGE_PRESENT_AT_HANDOFF_DOES_NOT_PROVE_PREREGISTRATION"
        in receipt["limitations"]
    )
    snapshot = verify_registry(fixture["lineage"], artifact_root=fixture["root"])
    assert len(snapshot.studies) == len(snapshot.results) == 1
    assert (
        snapshot.results[0]["evaluation_sha256"]
        == fixture["evaluation"]["evaluation_sha256"]
    )
    receipts = verify_receipt_store(fixture["receipts"])
    assert receipts == [receipt]
    receipt_path = fixture["receipts"] / "000000.json"
    assert receipt_path.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    assert receipt_store_status(fixture["receipts"])["receipt_count"] == 1


def test_uninitialized_after_start_lineage_is_explicitly_retrospective(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path / "retrospective")

    receipt = coordinate_terminal_handoff(
        terminal_dir=fixture["terminal"],
        sealed_study_path=fixture["study_path"],
        lineage_events_dir=fixture["lineage"],
        artifact_root=fixture["root"],
        receipt_events_dir=fixture["receipts"],
        expected_lineage_tip_sha256=ABSENT_CAS_TOKEN,
        expected_receipt_tip_sha256=ABSENT_CAS_TOKEN,
        binding_timing_classification=RETROSPECTIVE_ADMIN_BINDING,
        event_at_utc="2026-07-20T02:00:00Z",
        registry_id="qr-first-wave-admin",
        lineage_prefix="qr-",
        created_by="terminal-handoff-admin",
    )

    assert receipt["binding_timing_classification"] == (RETROSPECTIVE_ADMIN_BINDING)
    assert receipt["lineage_branch"] == (
        "UNINITIALIZED_LINEAGE_CREATED_SEALED_AND_BOUND"
    )
    assert receipt["lineage_before"]["state"] == "ABSENT"
    assert receipt["proof_eligible"] is False
    assert (
        "RETROSPECTIVE_ADMIN_BINDING_NEVER_PROVES_PREREGISTRATION"
        in receipt["limitations"]
    )
    snapshot = verify_registry(fixture["lineage"], artifact_root=fixture["root"])
    assert [event["event_type"] for event in snapshot.events] == [
        "GENESIS",
        "STUDY_SEALED",
        "RESULT_BOUND",
    ]


def test_genesis_only_retry_stays_retrospective_and_exact_bound_retry_is_explicit(
    tmp_path: Path,
) -> None:
    genesis_fixture = _terminal_fixture(tmp_path / "genesis-retry")
    genesis = initialize_registry(
        genesis_fixture["lineage"],
        artifact_root=genesis_fixture["root"],
        registry_id="qr-genesis-retry",
        lineage_prefix="qr-",
        created_by="pytest",
        event_at_utc="2026-07-20T02:10:00Z",
    )
    receipt = coordinate_terminal_handoff(
        terminal_dir=genesis_fixture["terminal"],
        sealed_study_path=genesis_fixture["study_path"],
        lineage_events_dir=genesis_fixture["lineage"],
        artifact_root=genesis_fixture["root"],
        receipt_events_dir=genesis_fixture["receipts"],
        expected_lineage_tip_sha256=genesis.latest_event_sha256,
        expected_receipt_tip_sha256=ABSENT_CAS_TOKEN,
        binding_timing_classification=RETROSPECTIVE_ADMIN_BINDING,
        event_at_utc="2026-07-20T02:10:01Z",
    )
    assert receipt["lineage_branch"] == "EXISTING_GENESIS_SEALED_AND_BOUND"
    assert receipt["binding_timing_classification"] == RETROSPECTIVE_ADMIN_BINDING

    bound_fixture = _terminal_fixture(tmp_path / "already-bound-retry")
    pending = _preseal_lineage(bound_fixture)
    bound = bind_result(
        bound_fixture["lineage"],
        artifact_root=bound_fixture["root"],
        evaluation_path=bound_fixture["terminal"] / "evaluation.json",
        expected_tip_sha256=pending.latest_event_sha256,
        event_at_utc="2026-07-20T01:00:02Z",
    )
    receipt = coordinate_terminal_handoff(
        terminal_dir=bound_fixture["terminal"],
        sealed_study_path=bound_fixture["study_path"],
        lineage_events_dir=bound_fixture["lineage"],
        artifact_root=bound_fixture["root"],
        receipt_events_dir=bound_fixture["receipts"],
        expected_lineage_tip_sha256=bound.latest_event_sha256,
        expected_receipt_tip_sha256=ABSENT_CAS_TOKEN,
        binding_timing_classification=LINEAGE_PRESENT_AT_HANDOFF,
        event_at_utc="2026-07-20T01:00:03Z",
    )
    assert receipt["lineage_branch"] == "EXISTING_EXACT_RESULT_REUSED"
    assert receipt["lineage_after"]["latest_event_sha256"] == (
        bound.latest_event_sha256
    )


def test_uninitialized_lineage_cannot_be_silently_called_existing(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path / "missing-lineage")

    with pytest.raises(
        DojoTerminalHandoffError,
        match="uninitialized lineage requires RETROSPECTIVE_ADMIN_BINDING",
    ):
        _coordinate_existing(fixture, ABSENT_CAS_TOKEN)

    assert not fixture["lineage"].exists()
    assert verify_receipt_store(fixture["receipts"]) == []


@pytest.mark.parametrize("missing", ["run.json", "evaluation.json", "cells.json"])
def test_missing_any_terminal_artifact_forbids_result_binding(
    tmp_path: Path, missing: str
) -> None:
    fixture = _terminal_fixture(tmp_path / missing.replace(".", "-"))
    pending = _preseal_lineage(fixture)
    (fixture["terminal"] / missing).unlink()

    with pytest.raises(DojoTerminalHandoffError, match="terminal bundle is incomplete"):
        _coordinate_existing(fixture, pending.latest_event_sha256)

    snapshot = verify_registry(fixture["lineage"], artifact_root=fixture["root"])
    assert len(snapshot.studies) == 1
    assert len(snapshot.results) == 0
    assert verify_receipt_store(fixture["receipts"]) == []


def test_run_failure_coexistence_forbids_even_otherwise_valid_bundle(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path / "failure-coexists")
    pending = _preseal_lineage(fixture)
    _write_json(
        fixture["terminal"] / "run_failure.json",
        {"status": "RUN_ABORTED_FAIL_CLOSED"},
    )

    with pytest.raises(DojoTerminalHandoffError, match="run_failure.json coexists"):
        _coordinate_existing(fixture, pending.latest_event_sha256)

    snapshot = verify_registry(fixture["lineage"], artifact_root=fixture["root"])
    assert len(snapshot.results) == 0
    assert verify_receipt_store(fixture["receipts"]) == []


@pytest.mark.parametrize("mutation", ["partial_cells", "authority", "study_sha"])
def test_partial_authority_or_foreign_study_bundle_never_binds(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _terminal_fixture(tmp_path / mutation)
    pending = _preseal_lineage(fixture)
    if mutation == "partial_cells":
        _write_json(fixture["terminal"] / "cells.json", fixture["cells"][:-1])
    elif mutation == "authority":
        run = copy.deepcopy(fixture["run"])
        run["live_permission"] = True
        run["run_sha256"] = _canonical_sha(
            {key: value for key, value in run.items() if key != "run_sha256"}
        )
        _write_json(fixture["terminal"] / "run.json", run)
    else:
        run = copy.deepcopy(fixture["run"])
        run["study_sha256"] = "f" * 64
        run["run_sha256"] = _canonical_sha(
            {key: value for key, value in run.items() if key != "run_sha256"}
        )
        _write_json(fixture["terminal"] / "run.json", run)

    with pytest.raises(
        DojoTerminalHandoffError, match="terminal bundle validation failed"
    ):
        _coordinate_existing(fixture, pending.latest_event_sha256)

    snapshot = verify_registry(fixture["lineage"], artifact_root=fixture["root"])
    assert len(snapshot.results) == 0
    assert verify_receipt_store(fixture["receipts"]) == []


def test_lineage_and_receipt_cas_are_fail_closed_and_receipt_is_tamper_evident(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path / "cas")
    pending = _preseal_lineage(fixture)

    with pytest.raises(DojoTerminalHandoffError, match="stale lineage tip"):
        _coordinate_existing(fixture, "0" * 64)
    receipt = _coordinate_existing(fixture, pending.latest_event_sha256)

    with pytest.raises(DojoTerminalHandoffError, match="stale receipt-store tip"):
        coordinate_terminal_handoff(
            terminal_dir=fixture["terminal"],
            sealed_study_path=fixture["study_path"],
            lineage_events_dir=fixture["lineage"],
            artifact_root=fixture["root"],
            receipt_events_dir=fixture["receipts"],
            expected_lineage_tip_sha256=receipt["lineage_after"]["latest_event_sha256"],
            expected_receipt_tip_sha256=ABSENT_CAS_TOKEN,
            binding_timing_classification=LINEAGE_PRESENT_AT_HANDOFF,
            event_at_utc="2026-07-20T01:00:03Z",
        )

    path = fixture["receipts"] / "000000.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["proof_eligible"] = True
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DojoTerminalHandoffError, match="research-only authority"):
        verify_receipt_store(fixture["receipts"])


def _invoke(*args: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_retrospective_branch_and_evaluation_only_rejection(
    tmp_path: Path,
) -> None:
    fixture = _terminal_fixture(tmp_path / "cli")
    result = _invoke(
        "bind",
        "--terminal-dir",
        fixture["terminal"],
        "--sealed-study",
        fixture["study_path"],
        "--lineage-events",
        fixture["lineage"],
        "--artifact-root",
        fixture["root"],
        "--receipt-events",
        fixture["receipts"],
        "--expected-lineage-tip-sha256",
        ABSENT_CAS_TOKEN,
        "--expected-receipt-tip-sha256",
        ABSENT_CAS_TOKEN,
        "--binding-timing-classification",
        RETROSPECTIVE_ADMIN_BINDING,
        "--registry-id",
        "qr-cli-retrospective",
        "--lineage-prefix",
        "qr-",
        "--created-by",
        "pytest-cli",
        "--event-at-utc",
        "2026-07-20T03:00:00Z",
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert verify_handoff_receipt(receipt) == receipt
    assert receipt["binding_timing_classification"] == RETROSPECTIVE_ADMIN_BINDING

    only_evaluation = fixture["root"] / "evaluation-only"
    only_evaluation.mkdir()
    _write_json(
        only_evaluation / "evaluation.json",
        fixture["evaluation"],
    )
    rejected = _invoke(
        "bind",
        "--terminal-dir",
        only_evaluation,
        "--sealed-study",
        fixture["study_path"],
        "--lineage-events",
        fixture["lineage"],
        "--artifact-root",
        fixture["root"],
        "--receipt-events",
        fixture["root"] / "rejected-receipts",
        "--expected-lineage-tip-sha256",
        receipt["lineage_after"]["latest_event_sha256"],
        "--expected-receipt-tip-sha256",
        ABSENT_CAS_TOKEN,
        "--binding-timing-classification",
        LINEAGE_PRESENT_AT_HANDOFF,
        "--event-at-utc",
        "2026-07-20T03:00:01Z",
    )
    assert rejected.returncode == 2
    error = json.loads(rejected.stderr)
    assert "terminal bundle is incomplete" in error["error"]
    assert error["live_permission"] is False
    assert error["order_authority"] == "NONE"
    assert not (fixture["root"] / "rejected-receipts" / "000000.json").exists()


def test_receipt_file_is_content_addressed_to_artifact_bytes(tmp_path: Path) -> None:
    fixture = _terminal_fixture(tmp_path / "artifact-hashes")
    pending = _preseal_lineage(fixture)
    receipt = _coordinate_existing(fixture, pending.latest_event_sha256)

    for name in ("run", "evaluation", "cells"):
        artifact = receipt["terminal_bundle"][name]
        raw = fixture["root"].joinpath(artifact["artifact_relpath"]).read_bytes()
        assert artifact["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
        assert artifact["artifact_size_bytes"] == len(raw)
