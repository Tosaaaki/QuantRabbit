from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from quant_rabbit.dojo_drive_archive import (
    DojoDriveArchiveError,
    canonical_sha256,
    finalize_archive,
    plan_archive,
    validate_terminal_run,
    verify_finalized_archive,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(path: Path, seal_field: str, value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != seal_field}
    sealed = {**body, seal_field: canonical_sha256(body)}
    _write_json(path, sealed)
    return sealed


def _terminal_run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    root.mkdir(parents=True)
    sessions: list[tuple[str, str | None]] = [
        ("sessions/main", None),
        ("sessions/lopo-EUR_USD", "EUR_USD"),
        ("sessions/lopo-USD_JPY", "USD_JPY"),
    ]
    for relative, held_out in sessions:
        session = root / relative
        session.mkdir(parents=True)
        (session / "ledger.jsonl").write_text(
            json.dumps({"held_out_pair": held_out, "pnl": 1.0}) + "\n",
            encoding="utf-8",
        )
    cell_body = {
        "contract": "QR_DOJO_BOT_TRAINER_CELL_V1",
        "schema_version": 1,
        "study_sha256": "a" * 64,
        "candidate_id": "C1",
        "proposal_sha256": "b" * 64,
        "intrabar": "OHLC",
        "cost_arm": "BASE",
        "execution_status": "SUCCESS",
        "failure_code": None,
        "metrics": {
            "pair_pnl_jpy": {"EUR_USD": 0.5, "USD_JPY": 0.5},
            "leave_one_pair_out_net_jpy": {"EUR_USD": 0.5, "USD_JPY": 0.5},
        },
        "ledger_evidence": {},
    }
    cell = {**cell_body, "cell_sha256": canonical_sha256(cell_body)}
    cells = [cell]
    _write_json(root / "cells.json", cells)
    evaluation_body = {
        "contract": "QR_DOJO_BOT_TRAINER_EVALUATION_V1",
        "schema_version": 1,
        "study_sha256": "a" * 64,
        "fixed_denominator": {
            "expected_cell_count": 1,
            "observed_cell_count": 1,
            "coordinate_receipts_complete": True,
        },
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    evaluation = {
        **evaluation_body,
        "evaluation_sha256": canonical_sha256(evaluation_body),
    }
    _write_json(root / "evaluation.json", evaluation)
    coordinate = {
        "candidate_id": "C1",
        "intrabar": "OHLC",
        "cost_arm": "BASE",
        "status": "COMPLETE",
        "main_session_dir": str(root / "sessions/main"),
        "main_error": None,
        "lopo_replay_complete": True,
        "lopo": [
            {
                "held_out_pair": held_out,
                "status": "VALID_COUNTERFACTUAL_REPLAY",
                "terminal_net_jpy": 1.0,
                "session_dir": str(root / relative),
                "ledger_path": str(root / relative / "ledger.jsonl"),
                "corpus_sha256": "d" * 64,
            }
            for relative, held_out in sessions[1:]
        ],
        "cell_sha256": cell["cell_sha256"],
    }
    run_body = {
        "contract": "QR_DOJO_BOT_TRAINER_RUN_V1",
        "schema_version": 1,
        "study_sha256": "a" * 64,
        "status": "COMPLETE",
        "corpus": {
            "corpus_sha256": "d" * 64,
            "sparse_m1_coverage": {
                "feed_pairs": ["EUR_USD", "USD_JPY"],
                "first_epoch": 1748736000,
                "last_epoch": 1751327940,
            },
        },
        "fixed_denominator": {
            "expected_cell_count": 1,
            "observed_cell_count": 1,
            "failed_cell_count": 0,
            "dropped_cell_count": 0,
            "coordinate_receipts_complete": True,
            "execution_success_complete": True,
        },
        "coordinates": [coordinate],
        "cells_path": str(root / "cells.json"),
        "evaluation_path": str(root / "evaluation.json"),
        "evaluation_sha256": evaluation["evaluation_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    run = {**run_body, "run_sha256": canonical_sha256(run_body)}
    _write_json(root / "run.json", run)
    return root


def test_terminal_run_and_cell_plan_are_strict_and_content_addressed(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    assert validate_terminal_run(root)["status"] == "COMPLETE"
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="cell",
        chunk_id="C1|OHLC|BASE",
    )
    assert plan["file_count"] == 6
    assert [row["path"] for row in plan["files"]] == sorted(
        [
            "cells.json",
            "evaluation.json",
            "run.json",
            "sessions/lopo-EUR_USD/ledger.jsonl",
            "sessions/lopo-USD_JPY/ledger.jsonl",
            "sessions/main/ledger.jsonl",
        ]
    )
    assert plan["content_tree_sha256"] == canonical_sha256(plan["files"])
    assert plan["remote_verification"]["remote_verified"] is False
    repeated = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="cell",
        chunk_id="C1|OHLC|BASE",
    )
    assert repeated["plan_sha256"] == plan["plan_sha256"]


def test_month_plan_covers_entire_terminal_run(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    (root / "report.md").write_text("complete\n", encoding="utf-8")
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    assert "report.md" in {row["path"] for row in plan["files"]}
    assert plan["total_source_bytes"] == sum(row["size_bytes"] for row in plan["files"])


def test_month_plan_rejects_a_label_outside_the_corpus_month(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    with pytest.raises(DojoDriveArchiveError, match="complete corpus UTC month"):
        plan_archive(
            source_run=root,
            destination=tmp_path / "archive",
            chunk_kind="month",
            chunk_id="2025-07",
        )


def test_plan_rejects_nonterminal_and_failure_ambiguity(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    _write_json(root / "run_failure.json", {"status": "RUN_ABORTED_FAIL_CLOSED"})
    with pytest.raises(DojoDriveArchiveError, match="run_failure"):
        plan_archive(
            source_run=root,
            destination=tmp_path / "archive",
            chunk_kind="month",
            chunk_id="2025-06",
        )


def test_plan_rejects_symlinks_and_overlapping_destination(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    os.symlink(root / "report-target", root / "unsafe-link")
    with pytest.raises(DojoDriveArchiveError, match="symlink"):
        plan_archive(
            source_run=root,
            destination=tmp_path / "archive",
            chunk_kind="month",
            chunk_id="2025-06",
        )
    (root / "unsafe-link").unlink()
    with pytest.raises(DojoDriveArchiveError, match="overlap"):
        plan_archive(
            source_run=root,
            destination=root / "archive",
            chunk_kind="month",
            chunk_id="2025-06",
        )


def test_plan_rejects_unsafe_chunk_identifier(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    with pytest.raises(DojoDriveArchiveError, match="safe identifier"):
        plan_archive(
            source_run=root,
            destination=tmp_path / "archive",
            chunk_kind="cell",
            chunk_id="../escape",
        )


def test_terminal_rejects_coordinate_cell_and_lopo_binding_attacks(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    run = _read_json(root / "run.json")
    run["coordinates"][0]["cell_sha256"] = "f" * 64
    _reseal(root / "run.json", "run_sha256", run)
    with pytest.raises(DojoDriveArchiveError, match="sealed cell"):
        validate_terminal_run(root)

    root = _terminal_run(tmp_path / "lopo")
    run = _read_json(root / "run.json")
    run["coordinates"][0]["lopo"][1]["held_out_pair"] = "EUR_USD"
    _reseal(root / "run.json", "run_sha256", run)
    with pytest.raises(DojoDriveArchiveError, match="LOPO identity"):
        validate_terminal_run(root)


def test_terminal_rejects_duplicate_coordinate_even_when_all_hashes_are_resealed(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    cells = json.loads((root / "cells.json").read_text(encoding="utf-8"))
    cells.append(deepcopy(cells[0]))
    _write_json(root / "cells.json", cells)
    evaluation = _read_json(root / "evaluation.json")
    evaluation["fixed_denominator"]["expected_cell_count"] = 2
    evaluation["fixed_denominator"]["observed_cell_count"] = 2
    evaluation = _reseal(root / "evaluation.json", "evaluation_sha256", evaluation)
    run = _read_json(root / "run.json")
    run["fixed_denominator"]["expected_cell_count"] = 2
    run["fixed_denominator"]["observed_cell_count"] = 2
    run["coordinates"].append(deepcopy(run["coordinates"][0]))
    run["evaluation_sha256"] = evaluation["evaluation_sha256"]
    _reseal(root / "run.json", "run_sha256", run)
    with pytest.raises(DojoDriveArchiveError, match="duplicate coordinate"):
        validate_terminal_run(root)


def test_cell_plan_rejects_an_intermediate_directory_symlink(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    external = tmp_path / "external-sessions"
    (root / "sessions").rename(external)
    os.symlink(external, root / "sessions")
    with pytest.raises(DojoDriveArchiveError, match="intermediate symlink"):
        plan_archive(
            source_run=root,
            destination=tmp_path / "archive",
            chunk_kind="cell",
            chunk_id="C1|OHLC|BASE",
        )


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is unavailable")
def test_finalize_is_atomic_locally_verified_and_idempotent(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    receipt = finalize_archive(plan_path=plan["plan_path"])
    archive = Path(receipt["archive_path"])
    before = archive.stat().st_mtime_ns
    assert receipt["local_payload_verified"] is True
    assert receipt["source_deleted"] is False
    assert receipt["remote_verification"] == {
        "status": "NOT_REQUESTED",
        "remote_verified": False,
        "metadata_receipt_sha256": None,
    }
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt["archive_sha256"]
    repeated = finalize_archive(plan_path=plan["plan_path"])
    assert repeated["finalization_sha256"] == receipt["finalization_sha256"]
    assert archive.stat().st_mtime_ns == before
    verified = verify_finalized_archive(plan_path=plan["plan_path"])
    assert verified["archive_sha256"] == receipt["archive_sha256"]
    assert root.exists()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is unavailable")
def test_finalize_rebuilds_inventory_and_rejects_a_resealed_shrunken_plan(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    (root / "extra-evidence.json").write_text("{}\n", encoding="utf-8")
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    forged = _read_json(Path(plan["plan_path"]))
    forged["files"] = [
        row for row in forged["files"] if row["path"] != "extra-evidence.json"
    ]
    forged["file_count"] = len(forged["files"])
    forged["total_source_bytes"] = sum(row["size_bytes"] for row in forged["files"])
    forged["content_tree_sha256"] = canonical_sha256(forged["files"])
    body = {key: item for key, item in forged.items() if key != "plan_sha256"}
    forged["plan_sha256"] = canonical_sha256(body)
    forged_path = (
        tmp_path / "archive/plans" / f"month-2025-06-{forged['plan_sha256']}.json"
    )
    _write_json(forged_path, forged)
    with pytest.raises(DojoDriveArchiveError, match="source-derived chunk"):
        finalize_archive(plan_path=forged_path)


def test_finalize_rejects_a_self_hashed_authority_forged_plan(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    forged = _read_json(Path(plan["plan_path"]))
    forged["proof_eligible"] = True
    body = {key: item for key, item in forged.items() if key != "plan_sha256"}
    forged["plan_sha256"] = canonical_sha256(body)
    forged_path = (
        tmp_path / "archive/plans" / f"month-2025-06-{forged['plan_sha256']}.json"
    )
    _write_json(forged_path, forged)
    with pytest.raises(DojoDriveArchiveError, match="authority boundary"):
        finalize_archive(plan_path=forged_path)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is unavailable")
def test_verify_rejects_a_self_hashed_authority_forged_receipt(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    receipt = finalize_archive(plan_path=plan["plan_path"])
    forged = _read_json(Path(receipt["receipt_path"]))
    forged["local_payload_verified"] = False
    forged["live_permission"] = True
    _reseal(Path(receipt["receipt_path"]), "finalization_sha256", forged)
    with pytest.raises(DojoDriveArchiveError, match="binding is invalid"):
        verify_finalized_archive(plan_path=plan["plan_path"])


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is unavailable")
def test_parallel_finalize_is_serialized_and_idempotent(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(finalize_archive, plan_path=plan["plan_path"])
            for _ in range(2)
        ]
        receipts = [future.result(timeout=30) for future in futures]
    assert receipts[0]["finalization_sha256"] == receipts[1]["finalization_sha256"]
    assert receipts[0]["archive_sha256"] == receipts[1]["archive_sha256"]


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is unavailable")
def test_finalize_recovers_an_incomplete_part_without_touching_source(
    tmp_path: Path,
) -> None:
    root = _terminal_run(tmp_path)
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    stem = f"month-2025-06-{plan['plan_sha256']}"
    archive_dir = tmp_path / "archive" / "archives"
    archive_dir.mkdir()
    (archive_dir / f"{stem}.tar.zst.part").write_bytes(b"interrupted")
    receipt = finalize_archive(plan_path=plan["plan_path"])
    assert Path(receipt["archive_path"]).is_file()
    assert not (archive_dir / f"{stem}.tar.zst.part").exists()
    assert (root / "run.json").is_file()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is unavailable")
def test_finalize_rejects_source_mutation_after_plan(tmp_path: Path) -> None:
    root = _terminal_run(tmp_path)
    plan = plan_archive(
        source_run=root,
        destination=tmp_path / "archive",
        chunk_kind="month",
        chunk_id="2025-06",
    )
    (root / "sessions/main/ledger.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(DojoDriveArchiveError, match="source-derived chunk"):
        finalize_archive(plan_path=plan["plan_path"])
