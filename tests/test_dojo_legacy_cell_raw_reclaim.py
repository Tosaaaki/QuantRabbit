from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import quant_rabbit.dojo_legacy_cell_raw_reclaim as reclaim_module
from quant_rabbit.dojo_drive_archive import (
    canonical_sha256,
    finalize_archive,
    plan_archive,
)
from quant_rabbit.dojo_legacy_cell_raw_reclaim import (
    DojoLegacyCellRawReclaimError,
    reclaim_legacy_cell_raw,
    verify_legacy_cell_raw_reclaim,
)


ZSTD = shutil.which("zstd")
pytestmark = pytest.mark.skipif(ZSTD is None, reason="zstd is unavailable")
DRIVE_PARENT_ID = "drive-parent-id-12345"
RAW_FILENAMES = {"broker_snapshot.json", "ledger.jsonl", "state.json"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hashes(path: Path) -> tuple[str, str, int]:
    payload = path.read_bytes()
    return (
        hashlib.sha256(payload).hexdigest(),
        hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        len(payload),
    )


def _terminal_run(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    root = tmp_path / "run"
    root.mkdir()
    study_sha = "a" * 64
    corpus_sha = "d" * 64
    session_paths: dict[str, list[str]] = {}
    cells: list[dict] = []
    coordinates: list[dict] = []
    for candidate in ("C1", "C2"):
        sessions = [
            (f"sessions/{candidate}-main", None),
            (f"sessions/{candidate}-lopo-EUR_USD", "EUR_USD"),
            (f"sessions/{candidate}-lopo-USD_JPY", "USD_JPY"),
        ]
        session_paths[candidate] = [relative for relative, _ in sessions]
        for relative, held_out in sessions:
            session = root / relative
            session.mkdir(parents=True)
            _write_json(
                session / "broker_snapshot.json",
                {"candidate": candidate, "held_out": held_out, "balance": 1},
            )
            (session / "ledger.jsonl").write_text(
                json.dumps(
                    {"candidate": candidate, "held_out_pair": held_out, "pnl": 1.0}
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(
                session / "state.json",
                {"candidate": candidate, "held_out": held_out, "status": "DONE"},
            )
        cell_body = {
            "contract": "QR_DOJO_BOT_TRAINER_CELL_V1",
            "schema_version": 1,
            "study_sha256": study_sha,
            "candidate_id": candidate,
            "proposal_sha256": ("b" if candidate == "C1" else "c") * 64,
            "intrabar": "OHLC",
            "cost_arm": "BASE",
            "execution_status": "SUCCESS",
            "failure_code": None,
            "metrics": {
                "pair_pnl_jpy": {"EUR_USD": 0.5, "USD_JPY": 0.5},
                "leave_one_pair_out_net_jpy": {
                    "EUR_USD": 0.5,
                    "USD_JPY": 0.5,
                },
            },
            "ledger_evidence": {},
        }
        cell = {**cell_body, "cell_sha256": canonical_sha256(cell_body)}
        cells.append(cell)
        coordinates.append(
            {
                "candidate_id": candidate,
                "intrabar": "OHLC",
                "cost_arm": "BASE",
                "status": "COMPLETE",
                "main_session_dir": str(root / sessions[0][0]),
                "main_error": None,
                "lopo_replay_complete": True,
                "lopo": [
                    {
                        "held_out_pair": held_out,
                        "status": "VALID_COUNTERFACTUAL_REPLAY",
                        "terminal_net_jpy": 1.0,
                        "session_dir": str(root / relative),
                        "ledger_path": str(root / relative / "ledger.jsonl"),
                        "corpus_sha256": corpus_sha,
                    }
                    for relative, held_out in sessions[1:]
                ],
                "cell_sha256": cell["cell_sha256"],
            }
        )
    _write_json(root / "cells.json", cells)
    evaluation_body = {
        "contract": "QR_DOJO_BOT_TRAINER_EVALUATION_V1",
        "schema_version": 1,
        "study_sha256": study_sha,
        "fixed_denominator": {
            "expected_cell_count": 2,
            "observed_cell_count": 2,
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
    run_body = {
        "contract": "QR_DOJO_BOT_TRAINER_RUN_V1",
        "schema_version": 1,
        "study_sha256": study_sha,
        "status": "COMPLETE",
        "corpus": {
            "corpus_sha256": corpus_sha,
            "sparse_m1_coverage": {
                "feed_pairs": ["EUR_USD", "USD_JPY"],
                "first_epoch": 1748736000,
                "last_epoch": 1751327940,
            },
        },
        "fixed_denominator": {
            "expected_cell_count": 2,
            "observed_cell_count": 2,
            "failed_cell_count": 0,
            "dropped_cell_count": 0,
            "coordinate_receipts_complete": True,
            "execution_success_complete": True,
        },
        "coordinates": coordinates,
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
    _write_json(
        root / "run.json", {**run_body, "run_sha256": canonical_sha256(run_body)}
    )
    return root, session_paths


def _remote_receipt(
    *,
    root: Path,
    plan: dict,
    finalization: dict,
    remote_dir: Path,
    legacy: bool = False,
) -> Path:
    run = _read_json(root / "run.json")
    archive_path = Path(finalization["archive_path"])
    archive_sha, archive_md5, archive_size = _file_hashes(archive_path)
    run_artifact_sha, _, _ = _file_hashes(root / "run.json")
    evaluation_artifact_sha, _, _ = _file_hashes(root / "evaluation.json")
    candidate_id, intrabar, cost_arm = plan["chunk_id"].split("|")
    body = {
        "archive_local_md5": archive_md5,
        "archive_local_path": str(archive_path),
        "archive_local_size_bytes": archive_size,
        "archive_name": archive_path.name,
        "archive_remote_md5": archive_md5,
        "archive_remote_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "broker_mutation_allowed": False,
        "checked_at_utc": "2026-07-22T02:00:00Z",
        "classification": run["classification"],
        "content_tree_sha256": plan["content_tree_sha256"],
        "coordinate": {
            "candidate_id": candidate_id,
            "cost_arm": cost_arm,
            "intrabar": intrabar,
        },
        "drive_file_id": f"drive-file-{candidate_id}-12345",
        "drive_file_name": archive_path.name,
        "drive_modified_time": "2026-07-22T01:59:00Z",
        "drive_parent_id": DRIVE_PARENT_ID,
        "drive_parent_name": "archives",
        "evaluation_artifact_sha256": evaluation_artifact_sha,
        "evaluation_sha256": run["evaluation_sha256"],
        "finalization_sha256": finalization["finalization_sha256"],
        "live_permission": False,
        "local_payload_verified": True,
        "order_authority": "NONE",
        "plan_sha256": plan["plan_sha256"],
        "promotion_eligible": False,
        "proof_eligible": False,
        "remote_verified": True,
        "source_deleted": False,
        "source_run_artifact_sha256": run_artifact_sha,
        "source_run_dir": str(root),
        "source_run_sha256": run["run_sha256"],
        "status": "REMOTE_VERIFIED",
        "study_sha256": run["study_sha256"],
    }
    if not legacy:
        body = {
            **body,
            "contract": "QR_DOJO_DRIVE_ARCHIVE_REMOTE_VERIFIED_RECEIPT_V1",
            "schema_version": 1,
        }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    remote_dir.mkdir(parents=True, exist_ok=True)
    path = remote_dir / (
        f"REMOTE_VERIFIED__cell-{plan['chunk_id']}__"
        f"{receipt['receipt_sha256']}.json"
    )
    _write_json(path, receipt)
    return path


def _fixture(
    tmp_path: Path, *, legacy: bool = False
) -> tuple[Path, Path, Path, dict[str, list[str]]]:
    root, session_paths = _terminal_run(tmp_path)
    archive_root = tmp_path / "archive"
    remote_dir = tmp_path / "remote-receipts"
    finalized: dict[str, tuple[dict, dict]] = {}
    for candidate in ("C1", "C2"):
        plan = plan_archive(
            source_run=root,
            destination=archive_root,
            chunk_kind="cell",
            chunk_id=f"{candidate}|OHLC|BASE",
        )
        finalization = finalize_archive(plan_path=plan["plan_path"], zstd_bin=ZSTD)
        finalized[candidate] = (plan, finalization)
    _remote_receipt(
        root=root,
        plan=finalized["C1"][0],
        finalization=finalized["C1"][1],
        remote_dir=remote_dir,
        legacy=legacy,
    )
    return root, archive_root, remote_dir, session_paths


def _options(root: Path, archive_root: Path, remote_dir: Path) -> dict:
    return {
        "source_run": root,
        "archive_root": archive_root,
        "remote_receipts_dir": remote_dir,
        "expected_drive_parent_id": DRIVE_PARENT_ID,
        "zstd_bin": ZSTD,
    }


def _raw_paths(root: Path, sessions: list[str]) -> set[Path]:
    return {
        root / session / filename for session in sessions for filename in RAW_FILENAMES
    }


def test_verify_is_read_only_and_excludes_unverified_local_archive(
    tmp_path: Path,
) -> None:
    root, archive_root, remote_dir, sessions = _fixture(tmp_path)
    before = {
        path.relative_to(root).as_posix(): _file_hashes(path)[0]
        for path in root.rglob("*")
        if path.is_file()
    }

    result = verify_legacy_cell_raw_reclaim(**_options(root, archive_root, remote_dir))

    assert result["status"] == "LEGACY_CELL_RAW_RECLAIM_VERIFIED_NOT_EXECUTED"
    plan = result["plan"]
    assert plan["verified_cell_count"] == 1
    assert plan["unverified_cell_count"] == 1
    assert plan["unverified_coordinate_ids"] == ["C2|OHLC|BASE"]
    assert plan["target_count"] == 9
    assert plan["unverified_raw_file_count"] == 9
    assert {row["coordinate_id"] for row in plan["targets"]} == {"C1|OHLC|BASE"}
    assert not {row["path"] for row in plan["targets"]} & {
        path.relative_to(root).as_posix() for path in _raw_paths(root, sessions["C2"])
    }
    after = {
        path.relative_to(root).as_posix(): _file_hashes(path)[0]
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not (root / "legacy-cell-reclaim-receipts").exists()


def test_exact_legacy_v0_remote_receipt_is_supported(tmp_path: Path) -> None:
    root, archive_root, remote_dir, _ = _fixture(tmp_path, legacy=True)

    result = verify_legacy_cell_raw_reclaim(**_options(root, archive_root, remote_dir))

    assert result["plan"]["verified_cells"][0]["receipt_contract"].endswith("LEGACY_V0")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("drive_parent_id", "wrong-parent-id-12345"),
        ("archive_remote_md5", "0" * 32),
        ("drive_file_name", "wrong.tar.zst"),
        ("finalization_sha256", "0" * 64),
    ],
)
def test_remote_metadata_or_lineage_tamper_fails_before_plan(
    tmp_path: Path, field: str, value: str
) -> None:
    root, archive_root, remote_dir, _ = _fixture(tmp_path)
    old_path = next(remote_dir.iterdir())
    receipt = _read_json(old_path)
    receipt[field] = value
    body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = canonical_sha256(body)
    new_path = remote_dir / (
        f"REMOTE_VERIFIED__cell-C1|OHLC|BASE__{receipt['receipt_sha256']}.json"
    )
    old_path.unlink()
    _write_json(new_path, receipt)

    with pytest.raises(
        DojoLegacyCellRawReclaimError,
        match="Drive metadata or lineage is invalid",
    ):
        reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir))

    assert not (root / "legacy-cell-reclaim-receipts").exists()


def test_reclaim_deletes_only_verified_raw_and_is_idempotent(tmp_path: Path) -> None:
    root, archive_root, remote_dir, sessions = _fixture(tmp_path)
    verified = _raw_paths(root, sessions["C1"])
    unverified = _raw_paths(root, sessions["C2"])

    receipt = reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir))

    assert receipt["status"] == "LEGACY_CELL_RAW_RECLAIMED"
    assert receipt["deleted_file_count"] == 9
    assert all(not path.exists() for path in verified)
    assert all(path.is_file() for path in unverified)
    assert all(
        (root / name).is_file()
        for name in ("run.json", "evaluation.json", "cells.json")
    )
    assert (
        reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir)) == receipt
    )


def test_reclaim_resumes_same_sealed_plan_after_partial_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, archive_root, remote_dir, sessions = _fixture(tmp_path)
    real_unlink = reclaim_module._unlink_target
    calls = 0

    def interrupted_unlink(run_root: Path, row: dict) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DojoLegacyCellRawReclaimError("simulated interruption")
        return real_unlink(run_root, row)

    monkeypatch.setattr(reclaim_module, "_unlink_target", interrupted_unlink)
    with pytest.raises(DojoLegacyCellRawReclaimError, match="simulated interruption"):
        reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir))
    plans = list((root / "legacy-cell-reclaim-receipts").glob("plan-*.json"))
    assert len(plans) == 1
    assert not list((root / "legacy-cell-reclaim-receipts").glob("reclaim-*.json"))

    monkeypatch.setattr(reclaim_module, "_unlink_target", real_unlink)
    receipt = reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir))

    assert receipt["deleted_file_count"] == 9
    assert all(not path.exists() for path in _raw_paths(root, sessions["C1"]))
    assert all(path.is_file() for path in _raw_paths(root, sessions["C2"]))


def test_reclaim_rejects_a_held_run_lock_without_writing_a_plan(
    tmp_path: Path,
) -> None:
    root, archive_root, remote_dir, _ = _fixture(tmp_path)
    lock_path = root / ".dojo-legacy-cell-reclaim.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DojoLegacyCellRawReclaimError, match="run lock is held"):
            reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not (root / "legacy-cell-reclaim-receipts").exists()


def test_cli_requires_explicit_reclaim_confirmation(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/run-dojo-legacy-cell-raw-reclaim.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "reclaim",
            "--source-run",
            str(tmp_path / "missing"),
            "--archive-root",
            str(tmp_path / "missing-archive"),
            "--remote-receipts-dir",
            str(tmp_path / "missing-remote"),
            "--expected-drive-parent-id",
            DRIVE_PARENT_ID,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--confirm-reclaim" in completed.stderr
