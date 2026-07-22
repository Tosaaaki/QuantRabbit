from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

import quant_rabbit.dojo_legacy_cell_raw_reclaim as reclaim_module
import quant_rabbit.dojo_legacy_cell_raw_reclaim_v2 as reclaim_v2_module
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
from quant_rabbit.dojo_legacy_cell_raw_reclaim_v2 import (
    ATTESTATION_CONTRACT,
    DojoLegacyCellRawReclaimV2Error,
    build_v2_attestation_body_candidate,
    build_v2_candidate_plan,
    enroll_v2_attestation_public_key,
    load_v2_plan,
    publish_v2_plan,
    reclaim_generation_2_raw,
    restore_raw_from_v2_plan,
    verify_signed_attestations,
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
            (session / "inbox" / "processed").mkdir(parents=True)
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


def test_files_under_accepts_only_the_empty_runtime_inbox(tmp_path: Path) -> None:
    root = tmp_path / "run"
    session = root / "sessions" / "session-1"
    (session / "inbox" / "processed").mkdir(parents=True)
    for filename in RAW_FILENAMES:
        (session / filename).write_text("evidence", encoding="utf-8")

    rows = reclaim_module._files_under(root, "sessions/session-1")

    assert {Path(row).name for row in rows} == RAW_FILENAMES
    (session / "inbox" / "processed" / "foreign.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(
        DojoLegacyCellRawReclaimError,
        match="processed inbox is not an empty real directory",
    ):
        reclaim_module._files_under(root, "sessions/session-1")


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
        verify_legacy_cell_raw_reclaim(**_options(root, archive_root, remote_dir))

    assert not (root / "legacy-cell-reclaim-receipts").exists()


def test_v1_reclaim_is_disabled_and_preserves_all_raw(tmp_path: Path) -> None:
    root, archive_root, remote_dir, sessions = _fixture(tmp_path)
    verified = _raw_paths(root, sessions["C1"])
    unverified = _raw_paths(root, sessions["C2"])

    with pytest.raises(
        DojoLegacyCellRawReclaimError,
        match="V1_REMOTE_RECEIPTS_UNSIGNED_RECLAIM_DISABLED",
    ):
        reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir))

    assert all(path.is_file() for path in verified)
    assert all(path.is_file() for path in unverified)
    assert all(
        (root / name).is_file()
        for name in ("run.json", "evaluation.json", "cells.json")
    )
    assert not (root / "legacy-cell-reclaim-receipts").exists()


def test_v1_reclaim_never_creates_a_partial_unlink(tmp_path: Path) -> None:
    root, archive_root, remote_dir, sessions = _fixture(tmp_path)
    with pytest.raises(DojoLegacyCellRawReclaimError):
        reclaim_legacy_cell_raw(**_options(root, archive_root, remote_dir))
    assert not (root / "legacy-cell-reclaim-receipts").exists()
    assert all(path.is_file() for path in _raw_paths(root, sessions["C1"]))
    assert all(path.is_file() for path in _raw_paths(root, sessions["C2"]))


def test_v1_reclaim_rejects_before_consulting_a_run_lock(
    tmp_path: Path,
) -> None:
    root, archive_root, remote_dir, _ = _fixture(tmp_path)
    lock_path = root / ".dojo-legacy-cell-reclaim.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            DojoLegacyCellRawReclaimError,
            match="V1_REMOTE_RECEIPTS_UNSIGNED_RECLAIM_DISABLED",
        ):
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


def _v1_lineage_after_reclaim(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, list[str]]]:
    root, archive_root, remote_dir, sessions = _fixture(tmp_path)
    plan = verify_legacy_cell_raw_reclaim(**_options(root, archive_root, remote_dir))[
        "plan"
    ]
    receipt_dir = root / "legacy-cell-reclaim-receipts"
    receipt_dir.mkdir()
    plan_path = receipt_dir / (
        f"plan-{plan['source_run_sha256']}-{plan['remote_receipt_set_sha256']}.json"
    )
    _write_json(plan_path, plan)
    for row in plan["targets"]:
        (root / row["path"]).unlink()
    receipt_body = {
        "contract": reclaim_module.RECLAIM_RECEIPT_CONTRACT,
        "schema_version": 1,
        "status": "LEGACY_CELL_RAW_RECLAIMED",
        "source_run_sha256": plan["source_run_sha256"],
        "remote_receipt_set_sha256": plan["remote_receipt_set_sha256"],
        "reclaim_plan_sha256": plan["reclaim_plan_sha256"],
        "completed_at_utc": "2026-07-22T02:10:00+00:00",
        "verified_cell_count": plan["verified_cell_count"],
        "unverified_cell_count": plan["unverified_cell_count"],
        "deleted_file_count": plan["target_count"],
        "deleted_files": plan["targets"],
        "reclaimed_logical_bytes": plan["target_bytes"],
        "reclaimed_allocated_bytes_observed": 0,
        "free_disk_bytes_before": 0,
        "free_disk_bytes_after": 0,
        "remote_unverified_cells_excluded": True,
        "restore_requires_verified_cell_archives": True,
        "historical_train_is_proof": False,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    receipt = {
        **receipt_body,
        "reclaim_receipt_sha256": reclaim_module._sha256(receipt_body),
    }
    receipt_path = receipt_dir / f"reclaim-{receipt['reclaim_receipt_sha256']}.json"
    _write_json(receipt_path, receipt)
    return root, archive_root, remote_dir, plan_path, receipt_path, sessions


def _v2_plan(tmp_path: Path) -> tuple[dict, Path, object, dict[str, list[str]]]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root, archive_root, remote_dir, prior_plan, prior_receipt, sessions = (
        _v1_lineage_after_reclaim(tmp_path)
    )
    private_key = Ed25519PrivateKey.generate()
    public_hex = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    fingerprint = hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()
    authority = enroll_v2_attestation_public_key(
        source_run=root,
        attestation_public_key_hex=public_hex,
        expected_public_key_sha256=fingerprint,
    )
    plan = build_v2_candidate_plan(
        source_run=root,
        archive_root=archive_root,
        prior_plan_path=prior_plan,
        prior_receipt_path=prior_receipt,
        prior_remote_receipts_dir=remote_dir,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=ZSTD,
        attestation_authority_seal_path=Path(authority["enrollment_seal_path"]),
    )
    return plan, publish_v2_plan(plan), private_key, sessions


def _attestation_observation(plan: dict, cell: dict, index: int, now: object) -> dict:
    from datetime import timedelta

    file_id = f"signed-drive-file-{index}-12345"
    revision = f"currentRevisionId+{index}="
    metadata = {
        "drive_file_id": file_id,
        "drive_file_name": Path(cell["archive_path"]).name,
        "drive_parent_ids": [DRIVE_PARENT_ID],
        "content_size_bytes": cell["archive_size_bytes"],
        "md5_checksum": cell["archive_md5"],
        "mime_type": "application/zstd",
        "modified_time": now.isoformat(),
        "version": str(index + 1),
        "head_revision_id": revision,
        "trashed": False,
    }
    return {
        "coordinate_id": cell["coordinate_id"],
        "files_get_before": deepcopy(metadata),
        "files_get_after": deepcopy(metadata),
        "revisions_list": {
            "drive_file_id": file_id,
            "current_revision_id": revision,
            "listed_revision_ids": [f"previousRevisionId+{index}=", revision],
        },
        "independent_readback": {
            "drive_file_id": file_id,
            "requested_revision_id": revision,
            "resolved_revision_id": revision,
            "content_size_bytes": cell["archive_size_bytes"],
            "md5_checksum": cell["archive_md5"],
            "sha256": cell["archive_sha256"],
        },
        "drivefs_file_id_xattr": file_id,
        "drivefs_md5_field48": cell["archive_md5"],
        "drivefs_revision_id_field78": revision,
        "drivefs_version_field57": str(index + 1),
        "attestation_method": "DRIVE_API_DOWNLOAD_SHA256",
        "observed_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(minutes=10)).isoformat(),
        "issuer": "pytest-ephemeral-operator",
    }


def _signed_attestations(tmp_path: Path, plan: dict, private_key: object) -> Path:
    import base64
    from datetime import datetime, timezone

    directory = tmp_path / "signed-attestations"
    directory.mkdir()
    now = datetime.now(timezone.utc)
    for index, cell in enumerate(plan["cells"]):
        body = build_v2_attestation_body_candidate(
            plan=plan,
            observation=_attestation_observation(plan, cell, index, now),
            now=now,
        )["body"]
        signature = private_key.sign(reclaim_module._canonical_bytes(body))
        envelope_body = {
            "contract": ATTESTATION_CONTRACT,
            "schema_version": 2,
            "body": body,
            "signature_algorithm": "ED25519",
            "signing_public_key_sha256": plan["attestation_authority"][
                "public_key_sha256"
            ],
            "signature_base64": base64.b64encode(signature).decode("ascii"),
        }
        envelope = {
            **envelope_body,
            "attestation_sha256": reclaim_module._sha256(envelope_body),
        }
        _write_json(directory / f"{index}.json", envelope)
    return directory


def test_v2_unsigned_drive_json_never_authorizes_reclaim(tmp_path: Path) -> None:
    plan, _, _, sessions = _v2_plan(tmp_path)
    directory = tmp_path / "unsigned"
    directory.mkdir()
    for index, cell in enumerate(plan["cells"]):
        _write_json(
            directory / f"{index}.json",
            {"body": {"coordinate_id": cell["coordinate_id"]}},
        )

    with pytest.raises(
        DojoLegacyCellRawReclaimV2Error, match="signed attestation schema"
    ):
        verify_signed_attestations(plan=plan, attestations_dir=directory)

    root = Path(plan["source_run_root"])
    assert all(path.is_file() for path in _raw_paths(root, sessions["C2"]))


def test_v2_signed_reclaim_and_atomic_roundtrip_restore(tmp_path: Path) -> None:
    plan, plan_path, private_key, sessions = _v2_plan(tmp_path)
    attestations = _signed_attestations(tmp_path, plan, private_key)

    receipt = reclaim_generation_2_raw(
        plan_path=plan_path,
        attestations_dir=attestations,
        expected_plan_sha256=plan["reclaim_plan_sha256"],
        expected_target_count=plan["generation_2_target_count"],
        expected_target_bytes=plan["generation_2_target_bytes"],
    )

    root = Path(plan["source_run_root"])
    assert receipt["status"] == "GENERATION_2_RAW_RECLAIMED"
    assert all(not path.exists() for path in _raw_paths(root, sessions["C2"]))
    assert all(
        path.stat().st_size == 0 for path in root.rglob(".dojo-retired-v2-*.anchor")
    )
    destination = tmp_path / "restored"
    restore = restore_raw_from_v2_plan(
        plan_path=plan_path,
        destination=destination,
        scope="all",
        expected_plan_sha256=plan["reclaim_plan_sha256"],
    )
    assert restore["roundtrip_verified"] is True
    assert restore["target_count"] == plan["cumulative_target_count"]
    assert (destination / ".dojo-legacy-cell-raw-reclaim-v2.lock").is_file()
    for row in plan["cumulative_targets"]:
        assert _file_hashes(destination / row["path"])[0] == row["sha256"]


def test_v2_exact_confirmation_and_restore_no_overwrite(tmp_path: Path) -> None:
    plan, plan_path, private_key, _ = _v2_plan(tmp_path)
    attestations = _signed_attestations(tmp_path, plan, private_key)
    with pytest.raises(DojoLegacyCellRawReclaimV2Error, match="exact plan"):
        reclaim_generation_2_raw(
            plan_path=plan_path,
            attestations_dir=attestations,
            expected_plan_sha256=plan["reclaim_plan_sha256"],
            expected_target_count=plan["generation_2_target_count"] + 1,
            expected_target_bytes=plan["generation_2_target_bytes"],
        )
    destination = tmp_path / "restore-conflict"
    conflict = destination / plan["generation_2_targets"][0]["path"]
    conflict.parent.mkdir(parents=True)
    conflict.write_text("different", encoding="utf-8")
    with pytest.raises(DojoLegacyCellRawReclaimV2Error, match="different bytes"):
        restore_raw_from_v2_plan(
            plan_path=plan_path,
            destination=destination,
            scope="generation2",
            expected_plan_sha256=plan["reclaim_plan_sha256"],
        )
    assert conflict.read_text(encoding="utf-8") == "different"


def test_v2_signed_attestation_rejects_tamper_staleness_and_md5_only(
    tmp_path: Path,
) -> None:
    import base64
    import copy
    from datetime import datetime, timedelta

    plan, _, private_key, _ = _v2_plan(tmp_path)
    directory = _signed_attestations(tmp_path, plan, private_key)
    path = sorted(directory.iterdir())[0]
    original = _read_json(path)

    def rejected(mutated: dict, match: str, *, verification_now=None) -> None:
        body = mutated["body"]
        mutated["signature_base64"] = base64.b64encode(
            private_key.sign(reclaim_module._canonical_bytes(body))
        ).decode("ascii")
        envelope_body = {
            key: value for key, value in mutated.items() if key != "attestation_sha256"
        }
        mutated["attestation_sha256"] = reclaim_module._sha256(envelope_body)
        _write_json(path, mutated)
        with pytest.raises(DojoLegacyCellRawReclaimV2Error, match=match):
            verify_signed_attestations(
                plan=plan,
                attestations_dir=directory,
                now=verification_now,
            )

    wrong_revision = copy.deepcopy(original)
    wrong_revision["body"]["drivefs_revision_id_field78"] = "differentRevision+="
    rejected(wrong_revision, "DriveFS binding")

    wrong_version = copy.deepcopy(original)
    wrong_version["body"]["drivefs_version_field57"] = "999"
    rejected(wrong_version, "DriveFS binding")

    modified_after_observation = copy.deepcopy(original)
    observed_at = datetime.fromisoformat(
        modified_after_observation["body"]["observed_at_utc"]
    )
    future_modified = (observed_at + timedelta(seconds=1)).isoformat()
    modified_after_observation["body"]["files_get_before"]["modified_time"] = (
        future_modified
    )
    modified_after_observation["body"]["files_get_after"]["modified_time"] = (
        future_modified
    )
    rejected(modified_after_observation, "files.get before/after")

    changed_during_readback = copy.deepcopy(original)
    changed_during_readback["body"]["files_get_after"]["version"] = "999"
    rejected(changed_during_readback, "files.get before/after")

    duplicate_revision = copy.deepcopy(original)
    revision = duplicate_revision["body"]["revisions_list"]["current_revision_id"]
    duplicate_revision["body"]["revisions_list"]["listed_revision_ids"] = [
        revision,
        revision,
    ]
    rejected(duplicate_revision, "uniquely bind")

    missing_head = copy.deepcopy(original)
    missing_head["body"]["revisions_list"]["listed_revision_ids"] = [
        "differentRevisionId+="
    ]
    rejected(missing_head, "uniquely bind")

    wrong_current = copy.deepcopy(original)
    wrong_current["body"]["revisions_list"]["current_revision_id"] = (
        "differentRevisionId+="
    )
    rejected(wrong_current, "uniquely bind")

    wrong_readback_revision = copy.deepcopy(original)
    wrong_readback_revision["body"]["independent_readback"]["requested_revision_id"] = (
        "differentRevisionId+="
    )
    rejected(wrong_readback_revision, "bound to the current Drive revision")

    wrong_resolved_revision = copy.deepcopy(original)
    wrong_resolved_revision["body"]["independent_readback"]["resolved_revision_id"] = (
        "differentRevisionId+="
    )
    rejected(wrong_resolved_revision, "bound to the current Drive revision")

    wrong_readback_bytes = copy.deepcopy(original)
    wrong_readback_bytes["body"]["independent_readback"]["sha256"] = "0" * 64
    rejected(wrong_readback_bytes, "bound to the current Drive revision")

    md5_only = copy.deepcopy(original)
    md5_only["body"]["attestation_method"] = "DRIVE_API_MD5_PLUS_DRIVEFS_PROVENANCE"
    rejected(md5_only, "structured files.get/revisions.list")

    signed_flat_self_report = copy.deepcopy(original)
    signed_flat_self_report["body"].pop("files_get_before")
    signed_flat_self_report["body"].pop("files_get_after")
    signed_flat_self_report["body"].pop("revisions_list")
    signed_flat_self_report["body"].pop("independent_readback")
    signed_flat_self_report["body"]["drive_file_id"] = "self-report-file-id-12345"
    signed_flat_self_report["body"]["drive_revision_id"] = "selfRevisionId+="
    signed_flat_self_report["body"]["remote_readback_sha256"] = original["body"][
        "archive_sha256"
    ]
    rejected(signed_flat_self_report, "body schema")

    stale = copy.deepcopy(original)
    enrolled = datetime.fromisoformat(
        plan["attestation_authority"]["enrollment_created_at_utc"]
    )
    pre_enrollment = copy.deepcopy(original)
    pre_enrollment["body"]["observed_at_utc"] = (
        enrolled - timedelta(seconds=1)
    ).isoformat()
    pre_enrollment["body"]["expires_at_utc"] = (
        enrolled + timedelta(minutes=1)
    ).isoformat()
    rejected(pre_enrollment, "predates public-key enrollment")

    stale["body"]["observed_at_utc"] = (enrolled + timedelta(seconds=1)).isoformat()
    stale["body"]["expires_at_utc"] = (enrolled + timedelta(minutes=1)).isoformat()
    rejected(stale, "stale", verification_now=enrolled + timedelta(minutes=2))

    bad_signature = copy.deepcopy(original)
    signature = bytearray(base64.b64decode(bad_signature["signature_base64"]))
    signature[0] ^= 1
    bad_signature["signature_base64"] = base64.b64encode(signature).decode("ascii")
    envelope_body = {
        key: value
        for key, value in bad_signature.items()
        if key != "attestation_sha256"
    }
    bad_signature["attestation_sha256"] = reclaim_module._sha256(envelope_body)
    _write_json(path, bad_signature)
    with pytest.raises(DojoLegacyCellRawReclaimV2Error, match="signature"):
        verify_signed_attestations(plan=plan, attestations_dir=directory)


def test_v2_plan_without_enrolled_public_key_cannot_be_published(
    tmp_path: Path,
) -> None:
    (
        root,
        archive_root,
        remote_dir,
        prior_plan,
        prior_receipt,
        _,
    ) = _v1_lineage_after_reclaim(tmp_path)
    candidate = build_v2_candidate_plan(
        source_run=root,
        archive_root=archive_root,
        prior_plan_path=prior_plan,
        prior_receipt_path=prior_receipt,
        prior_remote_receipts_dir=remote_dir,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=ZSTD,
    )
    with pytest.raises(
        DojoLegacyCellRawReclaimV2Error,
        match="SIGNED_ATTESTATION_PUBLIC_KEY_NOT_CONFIGURED",
    ):
        publish_v2_plan(candidate)


def test_v2_concurrent_keys_and_target_fork_publish_exactly_one(
    tmp_path: Path,
) -> None:
    from threading import Barrier

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root, archive_root, remote_dir, prior_plan, prior_receipt, _ = (
        _v1_lineage_after_reclaim(tmp_path)
    )
    private_keys = [Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()]
    public_keys = [
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
        for key in private_keys
    ]
    barrier = Barrier(2)

    def enroll(public_hex: str):
        barrier.wait()
        try:
            return enroll_v2_attestation_public_key(
                source_run=root,
                attestation_public_key_hex=public_hex,
                expected_public_key_sha256=hashlib.sha256(
                    bytes.fromhex(public_hex)
                ).hexdigest(),
            )
        except DojoLegacyCellRawReclaimV2Error as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(enroll, public_keys))
    enrolled = [result for result in results if isinstance(result, dict)]
    assert len(enrolled) == 1
    assert (
        len(
            list(
                (root / "legacy-cell-reclaim-v2/attestation-authority").glob(
                    "key-*.json"
                )
            )
        )
        == 1
    )

    plan = build_v2_candidate_plan(
        source_run=root,
        archive_root=archive_root,
        prior_plan_path=prior_plan,
        prior_receipt_path=prior_receipt,
        prior_remote_receipts_dir=remote_dir,
        expected_drive_parent_id=DRIVE_PARENT_ID,
        zstd_bin=ZSTD,
        attestation_authority_seal_path=Path(enrolled[0]["enrollment_seal_path"]),
    )
    fork = deepcopy(plan)
    target_path = fork["generation_2_targets"][0]["path"]
    fork["generation_2_targets"][0]["sha256"] = "f" * 64
    for row in fork["cumulative_targets"]:
        if row["path"] == target_path:
            row["sha256"] = "f" * 64
    fork["generation_2_target_set_sha256"] = reclaim_module._sha256(
        fork["generation_2_targets"]
    )
    fork["cumulative_target_set_sha256"] = reclaim_module._sha256(
        fork["cumulative_targets"]
    )
    fork_body = {
        key: value for key, value in fork.items() if key != "reclaim_plan_sha256"
    }
    fork["reclaim_plan_sha256"] = reclaim_module._sha256(fork_body)
    plan_barrier = Barrier(2)

    def publish(value: dict):
        plan_barrier.wait()
        try:
            return publish_v2_plan(value)
        except DojoLegacyCellRawReclaimV2Error as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        published = list(pool.map(publish, [plan, fork]))
    assert len([result for result in published if isinstance(result, Path)]) == 1
    generation = root / "legacy-cell-reclaim-v2/generation-000002"
    assert len(list(generation.glob("plan-*.json"))) == 1


def test_v2_explicit_path_fork_cannot_be_concealed(tmp_path: Path) -> None:
    plan, plan_path, _, _ = _v2_plan(tmp_path)
    concealed = plan_path.parent / f"plan-{'0' * 64}.json"
    _write_json(concealed, plan)

    with pytest.raises(DojoLegacyCellRawReclaimV2Error, match="exactly one bound plan"):
        load_v2_plan(plan_path)
    with pytest.raises(DojoLegacyCellRawReclaimV2Error, match="exactly one bound plan"):
        verify_signed_attestations(
            plan=plan, attestations_dir=tmp_path / "does-not-matter"
        )


def test_v2_reclaim_resumes_a_full_retirement_anchor_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path, private_key, sessions = _v2_plan(tmp_path)
    attestations = _signed_attestations(tmp_path, plan, private_key)
    real_ftruncate = os.ftruncate
    failed = False

    def crash_once(descriptor: int, length: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated crash before descriptor truncation")
        real_ftruncate(descriptor, length)

    monkeypatch.setattr(reclaim_v2_module.os, "ftruncate", crash_once)
    with pytest.raises(OSError, match="simulated crash"):
        reclaim_generation_2_raw(
            plan_path=plan_path,
            attestations_dir=attestations,
            expected_plan_sha256=plan["reclaim_plan_sha256"],
            expected_target_count=plan["generation_2_target_count"],
            expected_target_bytes=plan["generation_2_target_bytes"],
        )
    monkeypatch.setattr(reclaim_v2_module.os, "ftruncate", real_ftruncate)

    receipt = reclaim_generation_2_raw(
        plan_path=plan_path,
        attestations_dir=attestations,
        expected_plan_sha256=plan["reclaim_plan_sha256"],
        expected_target_count=plan["generation_2_target_count"],
        expected_target_bytes=plan["generation_2_target_bytes"],
    )
    assert receipt["status"] == "GENERATION_2_RAW_RECLAIMED"
    root = Path(plan["source_run_root"])
    assert all(not path.exists() for path in _raw_paths(root, sessions["C2"]))


def test_v2_crash_resume_excludes_same_root_restore_until_reclaim_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Event

    plan, plan_path, private_key, sessions = _v2_plan(tmp_path)
    attestations = _signed_attestations(tmp_path, plan, private_key)
    real_retire = reclaim_v2_module._retire_target

    def crash_before_retire(root_fd: int, row: dict, before_irreversible) -> int:
        del root_fd, row, before_irreversible
        raise OSError("simulated crash after authorization")

    monkeypatch.setattr(reclaim_v2_module, "_retire_target", crash_before_retire)
    with pytest.raises(OSError, match="after authorization"):
        reclaim_generation_2_raw(
            plan_path=plan_path,
            attestations_dir=attestations,
            expected_plan_sha256=plan["reclaim_plan_sha256"],
            expected_target_count=plan["generation_2_target_count"],
            expected_target_bytes=plan["generation_2_target_bytes"],
        )
    assert list(plan_path.parent.glob("authorization-*.json"))
    assert not list(plan_path.parent.glob("reclaim-*.json"))

    entered = Event()
    release = Event()

    def blocked_retire(root_fd: int, row: dict, before_irreversible) -> int:
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5)
        return real_retire(root_fd, row, before_irreversible)

    monkeypatch.setattr(reclaim_v2_module, "_retire_target", blocked_retire)
    options = {
        "plan_path": plan_path,
        "attestations_dir": attestations,
        "expected_plan_sha256": plan["reclaim_plan_sha256"],
        "expected_target_count": plan["generation_2_target_count"],
        "expected_target_bytes": plan["generation_2_target_bytes"],
    }
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(reclaim_generation_2_raw, **options)
        assert entered.wait(timeout=5)
        with pytest.raises(DojoLegacyCellRawReclaimV2Error, match="run lock is held"):
            restore_raw_from_v2_plan(
                plan_path=plan_path,
                destination=Path(plan["source_run_root"]),
                scope="generation2",
                expected_plan_sha256=plan["reclaim_plan_sha256"],
            )
        release.set()
        assert future.result()["status"] == "GENERATION_2_RAW_RECLAIMED"

    monkeypatch.setattr(reclaim_v2_module, "_retire_target", real_retire)
    restored = restore_raw_from_v2_plan(
        plan_path=plan_path,
        destination=Path(plan["source_run_root"]),
        scope="generation2",
        expected_plan_sha256=plan["reclaim_plan_sha256"],
    )
    assert restored["roundtrip_verified"] is True
    root = Path(plan["source_run_root"])
    assert all(path.is_file() for path in _raw_paths(root, sessions["C2"]))


def test_v2_reclaim_rejects_lock_path_rename_before_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path, private_key, sessions = _v2_plan(tmp_path)
    attestations = _signed_attestations(tmp_path, plan, private_key)
    real_authorize = reclaim_v2_module._load_or_create_authorization

    def replace_lock(**kwargs):
        result = real_authorize(**kwargs)
        root = Path(plan["source_run_root"])
        lock = root / ".dojo-legacy-cell-raw-reclaim-v2.lock"
        lock.rename(root / ".replaced-original-lock")
        lock.write_bytes(b"replacement")
        return result

    monkeypatch.setattr(
        reclaim_v2_module, "_load_or_create_authorization", replace_lock
    )
    with pytest.raises(
        DojoLegacyCellRawReclaimV2Error, match="lock pathname was replaced"
    ):
        reclaim_generation_2_raw(
            plan_path=plan_path,
            attestations_dir=attestations,
            expected_plan_sha256=plan["reclaim_plan_sha256"],
            expected_target_count=plan["generation_2_target_count"],
            expected_target_bytes=plan["generation_2_target_bytes"],
        )
    root = Path(plan["source_run_root"])
    assert all(path.is_file() for path in _raw_paths(root, sessions["C2"]))


def test_v2_restore_never_removes_a_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, plan_path, _, _ = _v2_plan(tmp_path)
    replacement = b"concurrent replacement must survive"

    def destination_appears(
        directory_fd: int, source_name: str, destination_name: str
    ) -> bool:
        del source_name
        descriptor = os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, replacement)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return False

    monkeypatch.setattr(
        reclaim_v2_module, "_atomic_rename_at_no_replace", destination_appears
    )
    destination = tmp_path / "restore-race"
    with pytest.raises(DojoLegacyCellRawReclaimV2Error, match="appeared concurrently"):
        restore_raw_from_v2_plan(
            plan_path=plan_path,
            destination=destination,
            scope="generation2",
            expected_plan_sha256=plan["reclaim_plan_sha256"],
        )
    assert any(
        path.read_bytes() == replacement
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert all(
        path.stat().st_size == 0 for path in destination.rglob("*.restore-*.tmp")
    )


def test_v2_cli_runs_every_subcommand_through_main(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root, archive_root, remote_dir, prior_plan, prior_receipt, _ = (
        _v1_lineage_after_reclaim(tmp_path)
    )
    private_key = Ed25519PrivateKey.generate()
    public_hex = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    fingerprint = hashlib.sha256(bytes.fromhex(public_hex)).hexdigest()
    public_file = tmp_path / "public-key.hex"
    public_file.write_text(public_hex + "\n", encoding="ascii")
    script = (
        Path(__file__).parents[1] / "scripts/run-dojo-legacy-cell-raw-reclaim-v2.py"
    )

    def cli(*arguments: str) -> dict:
        completed = subprocess.run(
            [sys.executable, str(script), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    authority = cli(
        "enroll-key",
        "--source-run",
        str(root),
        "--attestation-public-key-file",
        str(public_file),
        "--confirm-public-key-sha256",
        fingerprint,
    )
    lineage = [
        "--source-run",
        str(root),
        "--archive-root",
        str(archive_root),
        "--prior-plan",
        str(prior_plan),
        "--prior-receipt",
        str(prior_receipt),
        "--prior-remote-receipts-dir",
        str(remote_dir),
        "--expected-drive-parent-id",
        DRIVE_PARENT_ID,
        "--zstd-bin",
        str(ZSTD),
        "--attestation-authority-seal",
        authority["enrollment_seal_path"],
    ]
    verified = cli("verify", *lineage)
    assert verified["source_deletion_allowed"] is False
    planned = cli("plan", *lineage, "--confirm-public-key-sha256", fingerprint)
    plan = planned["plan"]
    plan_path = Path(planned["path"])
    observation_path = tmp_path / "observation.json"
    _write_json(
        observation_path,
        _attestation_observation(plan, plan["cells"][0], 0, datetime.now(timezone.utc)),
    )
    unsigned_body = cli(
        "build-attestation-body",
        "--plan",
        str(plan_path),
        "--observation-json",
        str(observation_path),
    )
    assert unsigned_body["status"] == "OPERATOR_CANDIDATE_ATTESTATION_BODY_NOT_SIGNED"
    assert unsigned_body["source_deletion_allowed"] is False
    assert (
        unsigned_body["body"]["files_get_before"]
        == unsigned_body["body"]["files_get_after"]
    )
    attestations = _signed_attestations(tmp_path, plan, private_key)
    signed = cli(
        "verify-attestations",
        "--plan",
        str(plan_path),
        "--attestations-dir",
        str(attestations),
    )
    assert signed["attestation_count"] == plan["required_attestation_count"]
    reclaimed = cli(
        "reclaim",
        "--plan",
        str(plan_path),
        "--attestations-dir",
        str(attestations),
        "--expected-plan-sha256",
        plan["reclaim_plan_sha256"],
        "--expected-target-count",
        str(plan["generation_2_target_count"]),
        "--expected-target-bytes",
        str(plan["generation_2_target_bytes"]),
    )
    assert reclaimed["status"] == "GENERATION_2_RAW_RECLAIMED"
    restored = cli(
        "restore",
        "--plan",
        str(plan_path),
        "--destination",
        str(tmp_path / "cli-restored"),
        "--scope",
        "all",
        "--expected-plan-sha256",
        plan["reclaim_plan_sha256"],
    )
    assert restored["roundtrip_verified"] is True
