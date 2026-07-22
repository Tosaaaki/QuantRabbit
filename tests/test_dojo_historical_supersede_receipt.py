from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_rabbit.dojo_historical_supersede_receipt import (
    DojoHistoricalSupersedeReceiptError,
    create_historical_supersede_receipt,
    verify_historical_supersede_receipt,
    verify_historical_supersede_receipt_file,
    verify_historical_supersede_receipt_locked,
)
from quant_rabbit.dojo_long_horizon_execution import (
    CELL_CONTRACT,
    claim_next_long_horizon_job,
    initialize_long_horizon_execution_state,
)
from quant_rabbit.dojo_long_horizon_plan import (
    IMPLEMENTATION_DIGEST_KEYS,
    RAPID_2025H1_PROFILE,
    SOURCE_BINDING_IDS,
    build_long_horizon_train_plan,
    canonical_sha256,
)
from quant_rabbit.dojo_long_horizon_schedule import build_long_horizon_stream_schedule


FAMILIES = ("breakout", "range_fade")
WORKERS = (
    {"worker_id": "breakout-v1", "family_id": "breakout", "config_sha256": "a" * 64},
    {"worker_id": "range-v1", "family_id": "range_fade", "config_sha256": "b" * 64},
)
RUNNER = {
    "runner_contract": "QR_TEST_BROKER_FREE_RUNNER_V1",
    "runner_code_sha256": "d" * 64,
    "result_contract": CELL_CONTRACT,
}
RESOURCES = {
    "max_resident_coordinates": 816,
    "max_rss_bytes": 2_147_483_648,
    "max_open_files": 256,
    "min_free_disk_bytes": 5_368_709_120,
    "max_checkpoint_bytes": 8_388_608,
    "max_terminal_bytes": 2_097_152,
    "max_parallel_jobs": 1,
}


def _digests(keys: tuple[str, ...], offset: int) -> dict[str, str]:
    return {key: f"{index + offset:064x}" for index, key in enumerate(keys, 1)}


def _artifacts(offset: int) -> tuple[dict, dict]:
    plan = build_long_horizon_train_plan(
        portfolio_families=FAMILIES,
        source_digests=_digests(SOURCE_BINDING_IDS, offset),
        corpus_digests=_digests(SOURCE_BINDING_IDS, offset + 10),
        implementation_digests=_digests(IMPLEMENTATION_DIGEST_KEYS, offset + 20),
        study_profile=RAPID_2025H1_PROFILE,
    )
    return plan, build_long_horizon_stream_schedule(plan, worker_bindings=WORKERS)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _old(tmp_path: Path, *, active: bool = True) -> Path:
    root = tmp_path / "old"
    root.mkdir()
    plan, schedule = _artifacts(0)
    _write(root / "plan.json", plan)
    _write(root / "schedule.json", schedule)
    initialize_long_horizon_execution_state(
        root / "execution-state",
        schedule=schedule,
        plan=plan,
        runner_binding=RUNNER,
        resource_policy=RESOURCES,
    )
    (root / ".historical-train.lock").touch(mode=0o600)
    if active:
        claim_next_long_horizon_job(
            root / "execution-state",
            schedule=schedule,
            plan=plan,
            runner_id="orphaned-test-runner",
        )
    return root


def _new(tmp_path: Path, offset: int = 100) -> Path:
    root = tmp_path / f"new-{offset}"
    root.mkdir()
    plan, schedule = _artifacts(offset)
    _write(root / "plan.json", plan)
    _write(root / "schedule.json", schedule)
    body = {
        "contract": "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1",
        "schema_version": 1,
        "generation": f"g2-room-v{offset}",
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    _write(
        root / "control-manifest.json",
        {**body, "manifest_sha256": canonical_sha256(body)},
    )
    return root


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _receipt(new: Path) -> Path:
    paths = list((new / "transition-receipts").iterdir())
    assert len(paths) == 1
    return paths[0]


def test_create_verify_and_locked_verify_leave_old_root_unchanged(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    before = _hashes(old)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    assert _hashes(old) == before
    assert receipt["old_generation"]["active_job_count"] == 1
    assert receipt["transition"]["old_execution_state_preserved"] is True
    assert (
        receipt["transition"]["old_non_state_evidence_preservation_asserted"] is False
    )
    assert receipt["transition"]["old_resume_authorized"] is False
    assert (
        verify_historical_supersede_receipt(receipt, old_root=old, new_root=new)
        == receipt
    )
    assert (
        verify_historical_supersede_receipt_file(
            _receipt(new), old_root=old, new_root=new
        )
        == receipt
    )
    descriptor = os.open(old / ".historical-train.lock", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (
            verify_historical_supersede_receipt_locked(
                receipt,
                old_root=old,
                new_root=new,
                old_lock_descriptor=descriptor,
            )
            == receipt
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_requires_exactly_one_active_claim_and_unowned_regular_lock(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    with pytest.raises(DojoHistoricalSupersedeReceiptError, match="exactly one"):
        create_historical_supersede_receipt(old_root=old, new_root=new)
    plan = json.loads((old / "plan.json").read_text())
    schedule = json.loads((old / "schedule.json").read_text())
    claim_next_long_horizon_job(
        old / "execution-state", schedule=schedule, plan=plan, runner_id="orphan"
    )
    descriptor = os.open(old / ".historical-train.lock", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DojoHistoricalSupersedeReceiptError, match="still owns"):
            create_historical_supersede_receipt(old_root=old, new_root=new)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_rejects_unknown_state_tree_symlink_and_later_drift(tmp_path: Path) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    (old / "execution-state" / "unknown").mkdir()
    with pytest.raises(DojoHistoricalSupersedeReceiptError, match="top-level tree"):
        create_historical_supersede_receipt(old_root=old, new_root=new)
    (old / "execution-state" / "unknown").rmdir()
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    outside = tmp_path / "outside.json"
    _write(outside, {"x": 1})
    (old / "execution-state" / "escape.json").symlink_to(outside)
    with pytest.raises(DojoHistoricalSupersedeReceiptError):
        verify_historical_supersede_receipt(receipt, old_root=old, new_root=new)


def test_rejects_successor_fork_and_unknown_store_entry(tmp_path: Path) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    create_historical_supersede_receipt(old_root=old, new_root=new)
    fork = _new(tmp_path, 200)
    (fork / "transition-receipts").mkdir()
    (fork / "transition-receipts" / "unknown.txt").write_text("x")
    with pytest.raises(DojoHistoricalSupersedeReceiptError, match="unknown entry"):
        create_historical_supersede_receipt(old_root=old, new_root=fork)
    (fork / "transition-receipts" / "unknown.txt").unlink()
    # One common append-only store is the successor registry; copying the prior
    # receipt makes a different successor for the same predecessor fail closed.
    (fork / "transition-receipts" / _receipt(new).name).write_bytes(
        _receipt(new).read_bytes()
    )
    with pytest.raises(DojoHistoricalSupersedeReceiptError, match="no longer matches"):
        create_historical_supersede_receipt(old_root=old, new_root=fork)


def test_cli_derives_generation_from_new_control_manifest(tmp_path: Path) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    script = (
        Path(__file__).parents[1] / "scripts/run-dojo-historical-supersede-receipt.py"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "create",
            "--old-root",
            str(old),
            "--new-root",
            str(new),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    output = json.loads(result.stdout)
    assert output["status"] == "SUPERSEDE_RECEIPT_CREATED_OR_REOPENED"
    assert output["receipt"]["new_generation"]["generation_id"].startswith(
        "g2-room-v100:"
    )
