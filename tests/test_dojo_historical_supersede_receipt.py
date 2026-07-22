from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import quant_rabbit.dojo_historical_supersede_receipt as supersede_module
from quant_rabbit.dojo_historical_supersede_receipt import (
    CONTRACT,
    LINEAGE_REGISTRY_DIRECTORY,
    DojoHistoricalSupersedeReceiptError,
    create_historical_supersede_receipt,
    verify_historical_supersede_receipt,
    verify_historical_supersede_receipt_file,
    verify_historical_supersede_receipt_locked,
    verify_historical_supersede_receipt_store_locked,
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
REPO_ROOT = Path(__file__).parents[1]
ROOM_CONTROL_PATH = REPO_ROOT / "config" / "dojo_g2_parallel_rooms_run_control_v1.json"


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
    generation = f"g2-room-v{offset}"
    implementation_digests = plan["implementation_binding"]["digests"]
    implementation_body = {
        "contract": "QR_DOJO_IMPLEMENTATION_DIGEST_MANIFEST_V1",
        "schema_version": 1,
        "implementation_digests": implementation_digests,
        "implementation_digests_sha256": canonical_sha256(implementation_digests),
        "automatic_deployment_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
        "promotion_eligible": False,
    }
    implementation = {
        **implementation_body,
        "implementation_manifest_sha256": canonical_sha256(implementation_body),
    }
    implementation_path = root / "sealed-inputs" / "implementation-manifest.json"
    _write(implementation_path, implementation)
    run_control = {
        "contract": "QR_DOJO_G2_HISTORICAL_RUN_CONTROL_V1",
        "schema_version": 1,
        "fixed_inputs": {"generation": generation},
        "execution": {"output_root": str(root)},
        "authority": {
            "automatic_deployment_allowed": False,
            "broker_mutation_allowed": False,
            "historical_replay_process_start_allowed": True,
            "live_permission": False,
            "order_authority": "NONE",
            "promotion_eligible": False,
            "research_filesystem_write_allowed": True,
        },
    }
    run_control_path = root / "sealed-inputs" / "run-control.json"
    _write(run_control_path, run_control)
    sealed_rows = []
    for artifact_id, path in (
        ("IMPLEMENTATION_MANIFEST", implementation_path),
        ("RUN_CONTROL", run_control_path),
    ):
        raw = path.read_bytes()
        sealed_rows.append(
            {
                "artifact_id": artifact_id,
                "relative_path": path.relative_to(root).as_posix(),
                "file_size_bytes": len(raw),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    body = {
        "contract": "QR_DOJO_HISTORICAL_TRAIN_GENERATION_MANIFEST_V1",
        "schema_version": 1,
        "generation": generation,
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "run_control_sha256": hashlib.sha256(run_control_path.read_bytes()).hexdigest(),
        "sealed_input_artifacts": sealed_rows,
        "sealed_input_artifacts_sha256": canonical_sha256(sealed_rows),
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
    paths = [
        path
        for path in (new / "transition-receipts").iterdir()
        if path.suffix == ".json"
        and json.loads(path.read_text())["contract"] == CONTRACT
    ]
    assert len(paths) == 1
    return paths[0]


def _assert_final_has_durable_anchor(final: Path) -> None:
    final_state = final.stat(follow_symlinks=False)
    anchors = list(final.parent.glob(f".{final.name}.pending*"))
    assert anchors
    assert any(
        (state.st_dev, state.st_ino) == (final_state.st_dev, final_state.st_ino)
        for state in (
            anchor.stat(follow_symlinks=False)
            for anchor in anchors
            if anchor.is_file() and not anchor.is_symlink()
        )
    )


def _write_legacy_v1_receipt(old: Path, new: Path) -> Path:
    predecessor_state_sha256 = canonical_sha256(
        supersede_module._stable_old_snapshot(old)
    )
    body = {
        "contract": "QR_DOJO_HISTORICAL_GENERATION_SUPERSEDE_RECEIPT_V1",
        "schema_version": 1,
        "transition_identity_sha256": "1" * 64,
        "predecessor_state_sha256": predecessor_state_sha256,
        "old_generation": {"root": str(old)},
        "new_generation": {"new_root": str(new)},
        "transition": {"reason": "LEGACY_ONLY"},
        "authority": {"order_authority": "NONE"},
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    path = (
        new
        / "transition-receipts"
        / (
            f"supersede-{receipt['transition_identity_sha256']}-"
            f"{receipt['receipt_sha256']}.json"
        )
    )
    _write(path, receipt)
    return path


def _reseal_new_inputs(new: Path) -> None:
    manifest_path = new / "control-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["sealed_input_artifacts"]:
        path = new / row["relative_path"]
        raw = path.read_bytes()
        row["file_size_bytes"] = len(raw)
        row["file_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["sealed_input_artifacts_sha256"] = canonical_sha256(
        manifest["sealed_input_artifacts"]
    )
    manifest["run_control_sha256"] = hashlib.sha256(
        (new / "sealed-inputs" / "run-control.json").read_bytes()
    ).hexdigest()
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(body)
    _write(manifest_path, manifest)


def test_create_verify_and_locked_verify_leave_old_root_unchanged(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    before = _hashes(old)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    assert create_historical_supersede_receipt(old_root=old, new_root=new) == receipt
    assert _hashes(old) == before
    assert receipt["old_generation"]["active_job_count"] == 1
    assert receipt["transition"]["old_execution_state_preserved"] is True
    assert (
        receipt["transition"]["old_non_state_evidence_preservation_asserted"] is False
    )
    assert receipt["transition"]["old_resume_authorized"] is False
    assert receipt["new_generation"]["new_root"] == str(new)
    assert len(receipt["new_generation"]["implementation_manifest_sha256"]) == 64
    lineage = old.parent / LINEAGE_REGISTRY_DIRECTORY
    assert len(list(lineage.glob("*.json"))) == 1
    _assert_final_has_durable_anchor(next(lineage.glob("*.json")))
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


def test_allows_zero_active_claims_and_requires_unowned_regular_lock(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    assert receipt["old_generation"]["active_job_count"] == 0
    descriptor = os.open(old / ".historical-train.lock", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DojoHistoricalSupersedeReceiptError, match="still owns"):
            create_historical_supersede_receipt(old_root=old, new_root=new)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_rejects_two_active_claims(tmp_path: Path) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    plan = json.loads((old / "plan.json").read_text())
    schedule = json.loads((old / "schedule.json").read_text())
    claim_next_long_horizon_job(
        old / "execution-state",
        schedule=schedule,
        plan=plan,
        runner_id="different-runner-creates-second-active-claim",
    )

    with pytest.raises(DojoHistoricalSupersedeReceiptError, match="at most one"):
        create_historical_supersede_receipt(old_root=old, new_root=new)


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


def test_rejects_valid_looking_unowned_state_paths(tmp_path: Path) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    schedule = json.loads((old / "schedule.json").read_text())
    job_sha = schedule["jobs"][0]["job_sha256"]
    extra_reducer = old / "execution-state" / "reducers" / job_sha / f"{'c' * 64}.json"
    _write(extra_reducer, {"looks": "canonical"})

    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="reducer path binding",
    ):
        create_historical_supersede_receipt(old_root=old, new_root=new)


def test_detects_status_neutral_known_directory_drift(tmp_path: Path) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    schedule = json.loads((old / "schedule.json").read_text())
    unclaimed_job_sha = schedule["jobs"][1]["job_sha256"]
    (old / "execution-state" / "claims" / unclaimed_job_sha).mkdir()

    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="no longer matches",
    ):
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
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="different successor",
    ):
        create_historical_supersede_receipt(old_root=old, new_root=fork)
    assert not list((fork / "transition-receipts").iterdir())


def test_locked_verifier_rejects_lock_inode_replacement_and_missing_lock(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    lock_path = old / ".historical-train.lock"
    descriptor = os.open(lock_path, os.O_RDONLY)
    detached = old / ".historical-train.lock.detached"
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_path.rename(detached)
        lock_path.touch(mode=0o600)
        with pytest.raises(
            DojoHistoricalSupersedeReceiptError,
            match="does not bind",
        ):
            verify_historical_supersede_receipt_locked(
                receipt,
                old_root=old,
                new_root=new,
                old_lock_descriptor=descriptor,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        detached.rename(lock_path)

    lock_path.unlink()
    with pytest.raises(DojoHistoricalSupersedeReceiptError, match="unavailable"):
        verify_historical_supersede_receipt(receipt, old_root=old, new_root=new)


def test_rejects_symlink_lock_and_unknown_lineage_entry(tmp_path: Path) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    lineage = old.parent / LINEAGE_REGISTRY_DIRECTORY
    (lineage / "unknown.txt").write_text("not a receipt", encoding="utf-8")
    with pytest.raises(DojoHistoricalSupersedeReceiptError, match="unknown entry"):
        verify_historical_supersede_receipt(receipt, old_root=old, new_root=new)
    (lineage / "unknown.txt").unlink()

    lock_path = old / ".historical-train.lock"
    real_lock = old / ".historical-train.lock.real"
    lock_path.rename(real_lock)
    lock_path.symlink_to(real_lock)
    try:
        with pytest.raises(
            DojoHistoricalSupersedeReceiptError,
            match="regular file",
        ):
            verify_historical_supersede_receipt(receipt, old_root=old, new_root=new)
    finally:
        lock_path.unlink()
        real_lock.rename(lock_path)


def test_rejects_sealed_run_control_and_implementation_drift(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path), _new(tmp_path)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    implementation_path = new / "sealed-inputs" / "implementation-manifest.json"
    implementation = json.loads(implementation_path.read_text())
    implementation["promotion_eligible"] = True
    _write(implementation_path, implementation)

    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="bytes drifted",
    ):
        verify_historical_supersede_receipt(receipt, old_root=old, new_root=new)


def test_rejects_fully_resealed_wrong_root_and_unsafe_implementation(
    tmp_path: Path,
) -> None:
    old = _old(tmp_path)
    wrong_root = _new(tmp_path)
    run_control_path = wrong_root / "sealed-inputs" / "run-control.json"
    run_control = json.loads(run_control_path.read_text())
    run_control["execution"]["output_root"] = str(tmp_path / "another-root")
    _write(run_control_path, run_control)
    _reseal_new_inputs(wrong_root)
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="does not bind",
    ):
        create_historical_supersede_receipt(old_root=old, new_root=wrong_root)

    unsafe = _new(tmp_path, 200)
    implementation_path = unsafe / "sealed-inputs" / "implementation-manifest.json"
    implementation = json.loads(implementation_path.read_text())
    implementation["promotion_eligible"] = True
    implementation_body = {
        key: value
        for key, value in implementation.items()
        if key != "implementation_manifest_sha256"
    }
    implementation["implementation_manifest_sha256"] = canonical_sha256(
        implementation_body
    )
    _write(implementation_path, implementation)
    _reseal_new_inputs(unsafe)
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="implementation manifest",
    ):
        create_historical_supersede_receipt(old_root=old, new_root=unsafe)


def test_v1_receipt_is_ignored_not_trusted_and_v2_can_be_added(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    legacy_path = _write_legacy_v1_receipt(old, new)
    descriptor = os.open(old / ".historical-train.lock", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            DojoHistoricalSupersedeReceiptError,
            match="does not contain one exact predecessor",
        ):
            verify_historical_supersede_receipt_store_locked(
                old_root=old,
                new_root=new,
                old_lock_descriptor=descriptor,
            )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    assert receipt["contract"] == CONTRACT
    assert legacy_path.is_file()
    assert len(list((new / "transition-receipts").glob("*.json"))) == 2
    assert (
        verify_historical_supersede_receipt_file(
            _receipt(new), old_root=old, new_root=new
        )
        == receipt
    )


def test_atomic_publish_recovers_from_crash_before_local_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    original = supersede_module._atomic_link_no_replace

    def crash_before(source: Path, destination: Path, *, directory_fd: int) -> bool:
        if destination.parent == new / "transition-receipts":
            raise DojoHistoricalSupersedeReceiptError("simulated crash before publish")
        return original(source, destination, directory_fd=directory_fd)

    monkeypatch.setattr(supersede_module, "_atomic_link_no_replace", crash_before)
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError, match="crash before publish"
    ):
        create_historical_supersede_receipt(old_root=old, new_root=new)

    store = new / "transition-receipts"
    assert not list(store.glob("*.json"))
    pending = list(store.glob(".*.pending"))
    assert len(pending) == 1
    pending_value = json.loads(pending[0].read_text())
    assert pending_value["contract"] == CONTRACT
    displaced_anchor = tmp_path / "displaced-valid-anchor"
    pending[0].rename(displaced_anchor)
    pending[0].write_text("{", encoding="utf-8")
    replacement_state = pending[0].stat(follow_symlinks=False)

    monkeypatch.setattr(supersede_module, "_atomic_link_no_replace", original)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    final = _receipt(new)
    assert json.loads(final.read_text()) == receipt
    assert pending[0].read_text(encoding="utf-8") == "{"
    preserved_state = pending[0].stat(follow_symlinks=False)
    assert (preserved_state.st_dev, preserved_state.st_ino) == (
        replacement_state.st_dev,
        replacement_state.st_ino,
    )
    assert json.loads(displaced_anchor.read_text())["contract"] == CONTRACT
    _assert_final_has_durable_anchor(final)


def test_atomic_publish_recovers_from_crash_before_lineage_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    original = supersede_module._atomic_link_no_replace

    def crash_before(source: Path, destination: Path, *, directory_fd: int) -> bool:
        if destination.parent.name == LINEAGE_REGISTRY_DIRECTORY:
            raise DojoHistoricalSupersedeReceiptError(
                "simulated lineage crash before publish"
            )
        return original(source, destination, directory_fd=directory_fd)

    monkeypatch.setattr(supersede_module, "_atomic_link_no_replace", crash_before)
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError, match="lineage crash before publish"
    ):
        create_historical_supersede_receipt(old_root=old, new_root=new)

    registry = old.parent / LINEAGE_REGISTRY_DIRECTORY
    assert not list(registry.glob("*.json"))
    assert len(list(registry.glob(".*.pending"))) == 1
    assert not list((new / "transition-receipts").iterdir())

    monkeypatch.setattr(supersede_module, "_atomic_link_no_replace", original)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    lineage_final = next(registry.glob("*.json"))
    assert json.loads(_receipt(new).read_text()) == receipt
    _assert_final_has_durable_anchor(lineage_final)


def test_atomic_publish_recovers_from_crash_immediately_after_local_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    original = supersede_module._atomic_link_no_replace

    def crash_after(source: Path, destination: Path, *, directory_fd: int) -> bool:
        published = original(source, destination, directory_fd=directory_fd)
        if destination.parent == new / "transition-receipts" and published:
            raise DojoHistoricalSupersedeReceiptError("simulated crash after publish")
        return published

    monkeypatch.setattr(supersede_module, "_atomic_link_no_replace", crash_after)
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError, match="crash after publish"
    ):
        create_historical_supersede_receipt(old_root=old, new_root=new)

    store = new / "transition-receipts"
    finals = list(store.glob("*.json"))
    assert len(finals) == 1
    assert json.loads(finals[0].read_text())["contract"] == CONTRACT
    _assert_final_has_durable_anchor(finals[0])

    monkeypatch.setattr(supersede_module, "_atomic_link_no_replace", original)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    assert json.loads(_receipt(new).read_text()) == receipt


def test_replaced_durable_anchor_is_preserved_and_repaired_without_unlink(
    tmp_path: Path,
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)
    final = _receipt(new)
    anchor = next(final.parent.glob(f".{final.name}.pending*"))
    displaced = tmp_path / "displaced-linked-anchor"
    anchor.rename(displaced)
    anchor.write_bytes(b"replacement must not be deleted")
    replacement_state = anchor.stat(follow_symlinks=False)

    with pytest.raises(
        DojoHistoricalSupersedeReceiptError, match="lacks its durable anchor"
    ):
        verify_historical_supersede_receipt_file(final, old_root=old, new_root=new)

    assert create_historical_supersede_receipt(old_root=old, new_root=new) == receipt
    preserved_state = anchor.stat(follow_symlinks=False)
    assert (preserved_state.st_dev, preserved_state.st_ino) == (
        replacement_state.st_dev,
        replacement_state.st_ino,
    )
    assert anchor.read_bytes() == b"replacement must not be deleted"
    assert json.loads(displaced.read_text())["contract"] == CONTRACT
    _assert_final_has_durable_anchor(final)
    assert (
        verify_historical_supersede_receipt_file(final, old_root=old, new_root=new)
        == receipt
    )


def test_transition_receipt_directory_swap_is_detected_and_replacement_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    store = new / "transition-receipts"
    detached = tmp_path / "detached-transition-receipts"
    original = supersede_module._atomic_link_no_replace
    swapped = False

    def publish_then_swap_parent(
        source: Path, destination: Path, *, directory_fd: int
    ) -> bool:
        nonlocal swapped
        published = original(source, destination, directory_fd=directory_fd)
        if destination.parent == store and published and not swapped:
            swapped = True
            store.rename(detached)
            store.mkdir()
            (store / "replacement-sentinel.txt").write_text(
                "preserve replacement", encoding="utf-8"
            )
        return published

    monkeypatch.setattr(
        supersede_module, "_atomic_link_no_replace", publish_then_swap_parent
    )
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="parent directory changed while in use",
    ):
        create_historical_supersede_receipt(old_root=old, new_root=new)

    assert (store / "replacement-sentinel.txt").read_text() == "preserve replacement"
    assert not list(store.glob("*.json"))
    assert len(list(detached.glob("*.json"))) == 1
    _assert_final_has_durable_anchor(next(detached.glob("*.json")))


def test_lineage_registry_directory_swap_is_detected_and_replacement_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new = _old(tmp_path, active=False), _new(tmp_path)
    registry = old.parent / LINEAGE_REGISTRY_DIRECTORY
    detached = tmp_path / "detached-lineage-registry"
    original = supersede_module._atomic_link_no_replace
    swapped = False

    def publish_then_swap_parent(
        source: Path, destination: Path, *, directory_fd: int
    ) -> bool:
        nonlocal swapped
        published = original(source, destination, directory_fd=directory_fd)
        if destination.parent == registry and published and not swapped:
            swapped = True
            registry.rename(detached)
            registry.mkdir()
            (registry / "replacement-sentinel.txt").write_text(
                "preserve replacement", encoding="utf-8"
            )
        return published

    monkeypatch.setattr(
        supersede_module, "_atomic_link_no_replace", publish_then_swap_parent
    )
    with pytest.raises(
        DojoHistoricalSupersedeReceiptError,
        match="parent directory changed while in use",
    ):
        create_historical_supersede_receipt(old_root=old, new_root=new)

    assert (registry / "replacement-sentinel.txt").read_text() == "preserve replacement"
    assert not list(registry.glob("*.json"))
    assert len(list(detached.glob("*.json"))) == 1
    _assert_final_has_durable_anchor(next(detached.glob("*.json")))
    assert not list((new / "transition-receipts").iterdir())


def test_prepared_generation_accepts_active_zero_predecessor_and_emits_v2(
    tmp_path: Path,
) -> None:
    from quant_rabbit.dojo_historical_train_control import prepare_generation

    control = json.loads(ROOM_CONTROL_PATH.read_text())
    new = tmp_path / "prepared-r8"
    control["execution"]["output_root"] = str(new)
    control["execution"]["archive_root"] = str(tmp_path / "archive")
    control["execution"]["global_heavy_lock_path"] = str(tmp_path / "global.lock")
    control["execution"]["conflicting_execution_roots"] = []
    control["execution"]["conflicting_run_lock_paths"] = []
    control_path = tmp_path / "room-control.json"
    control_path.write_text(json.dumps(control), encoding="utf-8")
    prepared = prepare_generation(repo_root=REPO_ROOT, run_control_path=control_path)
    old = _old(tmp_path, active=False)

    receipt = create_historical_supersede_receipt(old_root=old, new_root=new)

    assert prepared["status"] == "PREPARED"
    assert receipt["contract"] == CONTRACT
    assert receipt["schema_version"] == 2
    assert receipt["old_generation"]["active_job_count"] == 0
    assert receipt["new_generation"]["new_root"] == str(new)
    assert (
        verify_historical_supersede_receipt(receipt, old_root=old, new_root=new)
        == receipt
    )


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
