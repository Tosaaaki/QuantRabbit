from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from quant_rabbit.dojo_ai_trainer_packet import (
    canonical_packet_bytes,
    verify_trainer_packet,
)
from quant_rabbit.dojo_ai_tuning_state import (
    initialize_state_store,
    initialize_tuning_state,
)
from quant_rabbit.dojo_candidate_lineage_registry import (
    initialize_registry,
    seal_study_attempt,
    verify_registry,
)
from quant_rabbit.dojo_terminal_handoff import (
    ABSENT_CAS_TOKEN,
    LINEAGE_PRESENT_AT_HANDOFF,
    coordinate_terminal_handoff,
)
from tests.test_dojo_ai_trainer_packet import (
    _canonical_sha,
    _cells,
    _evaluation,
    _run,
    _sealed_study,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run-dojo-ai-trainer-packet.py"


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


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


def _accepted(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    value = json.loads(result.stdout)
    assert (
        result.stdout
        == json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    assert value["live_permission"] is False
    assert value["order_authority"] == "NONE"
    assert value["broker_mutation_allowed"] is False
    return value


def _rejected(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 2
    assert result.stdout == ""
    value = json.loads(result.stderr)
    assert value["status"] == "REJECTED"
    assert value["live_permission"] is False
    assert value["order_authority"] == "NONE"
    assert value["broker_mutation_allowed"] is False
    return value


def _fixture(root: Path) -> dict:
    root.mkdir(parents=True)
    sealed = _sealed_study()
    cells = _cells(sealed)
    evaluation = _evaluation(sealed)
    run = _run(sealed, evaluation, cells)
    study_path = _write_json(root / "artifacts" / "study.json", sealed)
    run_path = _write_json(root / "packet-inputs" / "run.json", run)
    evaluation_path = _write_json(
        root / "packet-inputs" / "evaluation.json", evaluation
    )
    cells_path = _write_json(root / "packet-inputs" / "cells.json", cells)
    lineage_events = root / "lineage-events"
    snapshot = initialize_registry(
        lineage_events,
        artifact_root=root,
        registry_id="qr-packet-cli-lineage",
        lineage_prefix="qr-",
        created_by="pytest",
        event_at_utc="2026-07-20T00:00:00Z",
    )
    snapshot = seal_study_attempt(
        lineage_events,
        artifact_root=root,
        sealed_study_path=study_path,
        expected_tip_sha256=snapshot.latest_event_sha256,
        event_at_utc="2026-07-20T00:00:01Z",
    )
    handoff_receipts = root / "handoff-receipts"
    handoff = coordinate_terminal_handoff(
        terminal_dir=run_path.parent,
        sealed_study_path=study_path,
        lineage_events_dir=lineage_events,
        artifact_root=root,
        receipt_events_dir=handoff_receipts,
        expected_lineage_tip_sha256=snapshot.latest_event_sha256,
        expected_receipt_tip_sha256=ABSENT_CAS_TOKEN,
        binding_timing_classification=LINEAGE_PRESENT_AT_HANDOFF,
        event_at_utc="2026-07-20T00:00:02Z",
    )
    lineage = verify_registry(lineage_events, artifact_root=root)
    tuning_state = initialize_tuning_state(
        lineage_events,
        artifact_root=root,
        sealed_study=sealed,
    )
    store = root / "tuning-state-events"
    stored = initialize_state_store(store, tuning_state)
    inputs = {
        "run": run,
        "evaluation": evaluation,
        "cells": cells,
        "lineage_events_dir": lineage_events,
        "artifact_root": root,
        "tuning_state": tuning_state,
    }
    drive_refs = [
        {
            "artifact_kind": "REPORT",
            "drive_file_id": "driveFile123",
            "content_sha256": "6" * 64,
            "content_size_bytes": 321,
            "remote_verified": True,
            "metadata_receipt_sha256": "7" * 64,
        }
    ]
    drive_path = _write_json(root / "packet-inputs" / "drive-refs.json", drive_refs)
    return {
        "root": root,
        "inputs": inputs,
        "store": store,
        "store_snapshot": stored,
        "lineage": lineage,
        "handoff_receipts": handoff_receipts,
        "handoff": handoff,
        "run_path": run_path,
        "evaluation_path": evaluation_path,
        "cells_path": cells_path,
        "drive_path": drive_path,
    }


def _arguments(fixture: dict, output: Path) -> tuple[object, ...]:
    stored = fixture["store_snapshot"]
    return (
        "--run-artifact",
        fixture["run_path"],
        "--evaluation-artifact",
        fixture["evaluation_path"],
        "--cells-artifact",
        fixture["cells_path"],
        "--lineage-events",
        fixture["inputs"]["lineage_events_dir"],
        "--artifact-root",
        fixture["root"],
        "--tuning-state-events",
        fixture["store"],
        "--expected-lineage-tip-sha256",
        fixture["lineage"].latest_event_sha256,
        "--expected-tuning-tip-event-sha256",
        stored["latest_event_sha256"],
        "--expected-tuning-state-sha256",
        stored["latest_state"]["state_sha256"],
        "--handoff-receipt-events",
        fixture["handoff_receipts"],
        "--expected-handoff-receipt-tip-sha256",
        fixture["handoff"]["receipt_sha256"],
        "--drive-evidence-refs-artifact",
        fixture["drive_path"],
        "--output",
        output,
    )


def test_cli_writes_exact_canonical_packet_exclusively(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "valid")
    output = fixture["root"] / "trainer-packet.json"
    receipt = _accepted(_invoke(*_arguments(fixture, output)))

    packet = json.loads(output.read_bytes())
    assert output.read_bytes() == canonical_packet_bytes(packet) + b"\n"
    assert verify_trainer_packet(packet) == packet
    assert receipt["packet_sha256"] == packet["packet_sha256"]
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["output_size_bytes"] == output.stat().st_size
    assert receipt["classification"] == "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY"
    assert (
        receipt["terminal_handoff_receipt_sha256"]
        == fixture["handoff"]["receipt_sha256"]
    )
    serialized = output.read_text(encoding="utf-8")
    assert "ledger.jsonl" not in serialized
    assert "cells_path" not in serialized

    original = output.read_bytes()
    rejected = _rejected(_invoke(*_arguments(fixture, output)))
    assert "refusing to overwrite" in rejected["error"]
    assert output.read_bytes() == original


def test_partial_authority_and_unverified_drive_inputs_fail_before_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "rejections")

    partial = fixture["inputs"]["cells"][:-1]
    fixture["cells_path"] = _write_json(
        fixture["root"] / "packet-inputs" / "partial-cells.json", partial
    )
    partial_output = fixture["root"] / "partial-packet.json"
    rejected = _rejected(_invoke(*_arguments(fixture, partial_output)))
    assert "does not bind exact cells artifact bytes" in rejected["error"]
    assert not partial_output.exists()

    fixture["cells_path"] = _write_json(
        fixture["root"] / "packet-inputs" / "cells-restored.json",
        fixture["inputs"]["cells"],
    )
    authority_run = copy.deepcopy(fixture["inputs"]["run"])
    authority_run["live_permission"] = True
    authority_run["run_sha256"] = _canonical_sha(
        {key: value for key, value in authority_run.items() if key != "run_sha256"}
    )
    fixture["run_path"] = _write_json(
        fixture["root"] / "packet-inputs" / "authority-run.json", authority_run
    )
    authority_output = fixture["root"] / "authority-packet.json"
    rejected = _rejected(_invoke(*_arguments(fixture, authority_output)))
    assert "does not bind exact run artifact bytes" in rejected["error"]
    assert not authority_output.exists()

    fixture["run_path"] = _write_json(
        fixture["root"] / "packet-inputs" / "run-restored.json",
        fixture["inputs"]["run"],
    )
    drive_refs = json.loads(fixture["drive_path"].read_text(encoding="utf-8"))
    drive_refs[0]["remote_verified"] = False
    fixture["drive_path"] = _write_json(
        fixture["root"] / "packet-inputs" / "unverified-drive.json", drive_refs
    )
    drive_output = fixture["root"] / "drive-packet.json"
    rejected = _rejected(_invoke(*_arguments(fixture, drive_output)))
    assert "not remotely verified" in rejected["error"]
    assert not drive_output.exists()

    fixture["drive_path"] = _write_json(
        fixture["root"] / "packet-inputs" / "empty-drive.json", []
    )
    empty_drive_output = fixture["root"] / "empty-drive-packet.json"
    rejected = _rejected(_invoke(*_arguments(fixture, empty_drive_output)))
    assert "at least one remotely verified Drive evidence" in rejected["error"]
    assert not empty_drive_output.exists()


def test_cli_rejects_stale_bindings_duplicate_json_and_symlink_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "safety")
    output = fixture["root"] / "stale-packet.json"
    stale_args = list(_arguments(fixture, output))
    state_index = stale_args.index("--expected-tuning-state-sha256") + 1
    stale_args[state_index] = "0" * 64
    rejected = _rejected(_invoke(*stale_args))
    assert "stale or forked tuning parent state" in rejected["error"]
    assert not output.exists()

    duplicate_cells = fixture["root"] / "packet-inputs" / "duplicate-cells.json"
    duplicate_cells.write_text('[{"x":1,"x":2}]\n', encoding="utf-8")
    duplicate_args = list(_arguments(fixture, output))
    cells_index = duplicate_args.index("--cells-artifact") + 1
    duplicate_args[cells_index] = duplicate_cells
    rejected = _rejected(_invoke(*duplicate_args))
    assert "duplicate JSON key" in rejected["error"]
    assert not output.exists()

    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite", encoding="utf-8")
    output.symlink_to(outside)
    rejected = _rejected(_invoke(*_arguments(fixture, output)))
    assert "already exists" in rejected["error"]
    assert outside.read_text(encoding="utf-8") == "do not overwrite"


def test_result_bound_without_handoff_receipt_never_generates_packet(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "missing-handoff")
    output = fixture["root"] / "forbidden-packet.json"
    empty_store = fixture["root"] / "empty-handoff-store"
    empty_store.mkdir()
    args = list(_arguments(fixture, output))
    events_index = args.index("--handoff-receipt-events") + 1
    args[events_index] = empty_store

    rejected = _rejected(_invoke(*args))

    assert "has no full-bundle hand-off receipt" in rejected["error"]
    assert not output.exists()
    lineage = verify_registry(
        fixture["inputs"]["lineage_events_dir"], artifact_root=fixture["root"]
    )
    assert lineage.events[-1]["event_type"] == "RESULT_BOUND"


def test_cli_requires_current_handoff_tip_and_exact_evaluation_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "handoff-cas")
    output = fixture["root"] / "handoff-cas-packet.json"
    stale_args = list(_arguments(fixture, output))
    tip_index = stale_args.index("--expected-handoff-receipt-tip-sha256") + 1
    stale_args[tip_index] = "0" * 64
    rejected = _rejected(_invoke(*stale_args))
    assert "stale or forked terminal hand-off receipt tip" in rejected["error"]
    assert not output.exists()

    evaluation = json.loads(fixture["evaluation_path"].read_text(encoding="utf-8"))
    alternate_evaluation = (
        fixture["root"] / "packet-inputs" / "alternate-evaluation.json"
    )
    alternate_evaluation.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fixture["evaluation_path"] = alternate_evaluation
    rejected = _rejected(_invoke(*_arguments(fixture, output)))
    assert "does not bind exact evaluation artifact bytes" in rejected["error"]
    assert not output.exists()
