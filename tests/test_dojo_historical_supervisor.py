from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quant_rabbit.dojo_historical_crash_supervisor import (
    DojoHistoricalCrashRecoveryError,
    advance_one_supervised_transition,
    quarantine_orphaned_evidence,
)
from quant_rabbit.dojo_historical_supervisor import (
    _open_launch_lease,
    _release_launch_lease,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_SCRIPT = REPO_ROOT / "scripts" / "run-dojo-historical-supervisor.py"


def _handoff() -> dict:
    job_sha = "1" * 64
    claim_sha = "2" * 64
    return {
        "job": {"job_sha256": job_sha},
        "claim": {
            "claim_sha256": claim_sha,
            "attempt_ordinal": 1,
            "runner_id": "dojo-g2-historical-train-v1",
        },
        "runner_handoff_sha256": "3" * 64,
    }


def _manifest() -> dict:
    return {
        "manifest_sha256": "4" * 64,
        "custody_control_plane_binding": {"binding_sha256": "5" * 64},
    }


def _quarantine_paths(evidence_root: Path) -> list[Path]:
    return sorted(evidence_root.glob("*.recovery-*.economic.jsonl"))


def test_crash_evidence_is_retained_and_same_name_is_freed(tmp_path: Path) -> None:
    evidence_root = tmp_path / "economic-evidence"
    evidence_root.mkdir()
    original = evidence_root / f"{'a' * 64}.economic.jsonl"
    original.write_bytes(b"partial append-only evidence\n")

    completion = quarantine_orphaned_evidence(
        evidence_root=evidence_root,
        handoff=_handoff(),
        control_manifest=_manifest(),
    )

    assert completion is not None
    assert completion["same_claim_preserved"] is True
    assert completion["partial_economics_reported"] is False
    assert not original.exists()
    quarantined = _quarantine_paths(evidence_root)
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"partial append-only evidence\n"
    # A completed receipt is idempotent and does not quarantine its own evidence.
    assert (
        quarantine_orphaned_evidence(
            evidence_root=evidence_root,
            handoff=_handoff(),
            control_manifest=_manifest(),
        )
        is None
    )


def test_pending_multi_file_recovery_finishes_after_mid_rename_crash(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "economic-evidence"
    evidence_root.mkdir()
    originals = [
        evidence_root / f"{'a' * 64}.economic.jsonl",
        evidence_root / f"{'b' * 64}.economic.jsonl",
    ]
    for index, path in enumerate(originals):
        path.write_bytes(f"partial-{index}\n".encode())
    quarantine_orphaned_evidence(
        evidence_root=evidence_root,
        handoff=_handoff(),
        control_manifest=_manifest(),
    )
    completion_path = next(evidence_root.glob("crash-recovery-complete-*.json"))
    completion_path.unlink()
    quarantined = _quarantine_paths(evidence_root)
    # Simulate a crash after only the first mapping became durable.
    quarantined[1].rename(originals[1])

    completion = quarantine_orphaned_evidence(
        evidence_root=evidence_root,
        handoff=_handoff(),
        control_manifest=_manifest(),
    )

    assert completion is not None
    assert completion["moved_file_count"] == 2
    assert all(not path.exists() for path in originals)
    assert len(_quarantine_paths(evidence_root)) == 2


@pytest.mark.parametrize("unsafe_kind", ["unknown", "symlink", "hardlink"])
def test_crash_recovery_rejects_unaccounted_or_aliased_evidence(
    tmp_path: Path, unsafe_kind: str
) -> None:
    evidence_root = tmp_path / "economic-evidence"
    evidence_root.mkdir()
    original = evidence_root / f"{'a' * 64}.economic.jsonl"
    original.write_bytes(b"partial\n")
    if unsafe_kind == "unknown":
        (evidence_root / "unbound.txt").write_text("unknown", encoding="utf-8")
    elif unsafe_kind == "symlink":
        (evidence_root / f"{'b' * 64}.economic.jsonl").symlink_to(original)
    else:
        os.link(original, evidence_root / f"{'b' * 64}.economic.jsonl")

    with pytest.raises(DojoHistoricalCrashRecoveryError):
        quarantine_orphaned_evidence(
            evidence_root=evidence_root,
            handoff=_handoff(),
            control_manifest=_manifest(),
        )


def test_supervised_transition_routes_running_to_same_claim_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    lifecycle = {"state": "RUNNING", "lifecycle_sha256": "6" * 64}
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane._load_generation",
        lambda **kwargs: ({}, root, {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane.evaluate_historical_lifecycle",
        lambda **kwargs: lifecycle,
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane.advance_one_historical_transition",
        lambda **kwargs: pytest.fail("RUNNING must not use the fresh-claim step"),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.resume_active_historical_job",
        lambda **kwargs: {"status": "RECOVERED"},
    )

    result = advance_one_supervised_transition(
        repo_root=REPO_ROOT, run_control_path=tmp_path / "control.json"
    )

    assert result["status"] == "RECOVERED"
    assert result["heartbeat_step"]["selected_transition"] == "RECOVER_ACTIVE_JOB"
    assert result["heartbeat_step"]["transition_execution_count"] == 1


def test_supervised_transition_preserves_existing_nonrunning_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane._load_generation",
        lambda **kwargs: ({}, root, {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane.evaluate_historical_lifecycle",
        lambda **kwargs: {"state": "READY_TO_CLAIM"},
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.resume_active_historical_job",
        lambda **kwargs: pytest.fail("fresh work must not use crash recovery"),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane.advance_one_historical_transition",
        lambda **kwargs: {"status": "FRESH_STEP"},
    )

    result = advance_one_supervised_transition(
        repo_root=REPO_ROOT, run_control_path=tmp_path / "control.json"
    )

    assert result == {"status": "FRESH_STEP"}


def test_supervised_transition_publishes_unpublished_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    lifecycle = {"state": "TERMINAL_UNPUBLISHED", "lifecycle_sha256": "7" * 64}
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane._load_generation",
        lambda **kwargs: ({}, root, {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane.evaluate_historical_lifecycle",
        lambda **kwargs: lifecycle,
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.control_plane.advance_one_historical_transition",
        lambda **kwargs: pytest.fail("unpublished terminal must not remain a no-op"),
    )
    monkeypatch.setattr(
        "quant_rabbit.dojo_historical_crash_supervisor.publish_unpublished_terminal_completion",
        lambda **kwargs: {"status": "PUBLISHED"},
    )

    result = advance_one_supervised_transition(
        repo_root=REPO_ROOT, run_control_path=tmp_path / "control.json"
    )

    assert result["status"] == "PUBLISHED"
    assert (
        result["heartbeat_step"]["selected_transition"] == "PUBLISH_TERMINAL_COMPLETION"
    )


def test_kernel_lease_releases_after_forced_child_death(tmp_path: Path) -> None:
    supervisor_root = tmp_path / "supervisor"
    supervisor_root.mkdir()
    lease_path = supervisor_root / ".historical-supervisor.lock"
    code = """
import fcntl, os, sys, time
path = sys.argv[1]
fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print('READY', flush=True)
time.sleep(30)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(lease_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "READY"
        with pytest.raises(BlockingIOError):
            _open_launch_lease(supervisor_root)
        child.kill()
        assert child.wait(timeout=5) < 0
        descriptor, observed_path = _open_launch_lease(supervisor_root)
        try:
            assert observed_path == lease_path
        finally:
            _release_launch_lease(descriptor)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_supervisor_script_uses_detached_single_owner_launch() -> None:
    source = (
        REPO_ROOT / "src" / "quant_rabbit" / "dojo_historical_supervisor.py"
    ).read_text(encoding="utf-8")
    script_source = SUPERVISOR_SCRIPT.read_text(encoding="utf-8")

    assert "start_new_session=True" in source
    assert "stdin=subprocess.DEVNULL" in source
    assert "pass_fds=(lease_descriptor,)" in source
    assert "SUPERVISOR_ALREADY_RUNNING" in source
    assert "advance_one_supervised_transition" in source
    assert "trainer_action_allowed" in source
    assert "_child" in script_source
    assert "dojo_g2_parallel_rooms_run_control_v6.json" in script_source


def test_supervisor_script_is_valid_json_on_status_rejection(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR_SCRIPT),
            "status",
            "--run-control",
            str(tmp_path / "missing.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    payload = json.loads(result.stderr)
    assert payload["status"] == "REJECTED"
    assert payload["partial_economics_reported"] is False
    assert payload["trainer_action_allowed"] is False
