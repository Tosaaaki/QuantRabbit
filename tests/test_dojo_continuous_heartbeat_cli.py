from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from quant_rabbit.dojo_continuous_heartbeat import (
    OBSERVATION_CONTRACT,
    canonical_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-dojo-continuous-heartbeat.py"
POLICY = ROOT / "config" / "dojo_continuous_heartbeat_policy_v1.json"
PROBE = ROOT / "config" / "dojo_continuous_heartbeat_local_probe_v1.json"
IDLE_STATUS = ROOT / "config" / "dojo_continuous_heartbeat_idle_run_status_v1.json"


def _authority() -> dict:
    return {
        "model_invocation_allowed": False,
        "drive_access_allowed": False,
        "process_start_allowed": False,
        "runner_invocation_allowed": False,
        "filesystem_delete_allowed": False,
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
    }


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _idle_observation(observed_at: str) -> dict:
    return {
        "contract": OBSERVATION_CONTRACT,
        "schema_version": 1,
        "observed_at_utc": observed_at,
        "run": {
            "process_state": "IDLE",
            "run_id": None,
            "process_identity_sha256": None,
            "expected_coordinate_count": 0,
            "completed_coordinate_count": 0,
            "checkpoint_sha256": None,
            "terminal_bundle_sha256": None,
            "failure_artifact_sha256": None,
            "economics_fields_present": False,
        },
        "resources": {
            "active_trainer_count": 0,
            "remote_unverified_generation_count": 0,
            "compression_upload_active_count": 0,
            "free_bytes": 100 * 1024**3,
            "predicted_next_output_bytes": 1024,
            "archive_temp_bytes": 0,
        },
        "authority": _authority(),
    }


def test_tick_semantic_noop_has_stable_status_and_no_event(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    initialized = _run(
        "init",
        "--policy",
        str(POLICY),
        "--state-dir",
        str(state_dir),
        "--event-at-utc",
        "2026-07-22T00:00:00Z",
    )
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr
    assert json.loads(initialized.stdout)["status"] == "INITIALIZED"

    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps(_idle_observation("2026-07-22T00:01:00Z")))
    first = _run(
        "tick",
        "--policy",
        str(POLICY),
        "--state-dir",
        str(state_dir),
        "--observation",
        str(observation),
        "--event-at-utc",
        "2026-07-22T00:01:00Z",
    )
    first_body = json.loads(first.stdout)
    assert first.returncode == 0
    assert first_body["status"] == "UPDATED"
    assert first_body["event_appended"] is True

    observation.write_text(json.dumps(_idle_observation("2026-07-22T00:02:00Z")))
    second = _run(
        "tick",
        "--policy",
        str(POLICY),
        "--state-dir",
        str(state_dir),
        "--observation",
        str(observation),
        "--event-at-utc",
        "2026-07-22T00:02:00Z",
    )
    second_body = json.loads(second.stdout)
    assert second.returncode == 0
    assert second_body["status"] == "NO_CHANGE"
    assert second_body["event_appended"] is False
    assert len(list((state_dir / "events").iterdir())) == 2
    assert second_body["authority"] == {
        "broker_mutation_allowed": False,
        "live_permission": False,
        "order_authority": "NONE",
    }


def test_nonblocking_single_lease_returns_75_without_writing(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    initialized = _run(
        "init",
        "--policy",
        str(POLICY),
        "--state-dir",
        str(state_dir),
        "--event-at-utc",
        "2026-07-22T00:00:00Z",
    )
    assert initialized.returncode == 0
    lock = state_dir / ".heartbeat.lock"
    with lock.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = _run(
            "status",
            "--policy",
            str(POLICY),
            "--state-dir",
            str(state_dir),
        )
    body = json.loads(blocked.stdout)
    assert blocked.returncode == 75
    assert body["status"] == "LEASE_BUSY"
    assert body["error_code"] == "SINGLE_HEARTBEAT_LEASE_BUSY"
    assert len(list((state_dir / "events").iterdir())) == 1


def test_observe_local_emits_a_sealed_tick_compatible_observation(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    marker_names = (
        "active-trainers",
        "remote-unverified-generations",
        "compression-upload-active",
    )
    for name in marker_names:
        (runtime / name).mkdir()
    status_path = runtime / "run-status.json"
    status_path.write_text(IDLE_STATUS.read_text())

    probe = json.loads(PROBE.read_text())
    probe.update(
        {
            "state_root": str(runtime),
            "run_status_path": str(status_path),
            "active_trainer_marker_directory": str(runtime / marker_names[0]),
            "remote_unverified_generation_marker_directory": str(
                runtime / marker_names[1]
            ),
            "compression_upload_marker_directory": str(runtime / marker_names[2]),
            "storage_path": str(tmp_path),
        }
    )
    probe.pop("probe_sha256")
    probe["probe_sha256"] = canonical_sha256(probe)
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe))

    policy = json.loads(POLICY.read_text())
    policy["local_probe_manifest_sha256"] = probe["probe_sha256"]
    policy.pop("policy_sha256")
    policy["policy_sha256"] = canonical_sha256(policy)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    observed = _run(
        "observe-local",
        "--policy",
        str(policy_path),
        "--probe",
        str(probe_path),
        "--observed-at-utc",
        "2026-07-22T01:00:00Z",
    )
    assert observed.returncode == 0, observed.stdout + observed.stderr
    body = json.loads(observed.stdout)
    assert body["contract"] == OBSERVATION_CONTRACT
    assert body["observation_sha256"]
    assert body["resources"]["active_trainer_count"] == 0
    assert body["authority"] == _authority()


def test_tick_local_observes_and_applies_without_a_temporary_file(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in (
        "active-trainers",
        "remote-unverified-generations",
        "compression-upload-active",
    ):
        (runtime / name).mkdir()
    status_path = runtime / "run-status.json"
    status_path.write_text(IDLE_STATUS.read_text())

    probe = json.loads(PROBE.read_text())
    probe.update(
        {
            "state_root": str(runtime),
            "run_status_path": str(status_path),
            "active_trainer_marker_directory": str(runtime / "active-trainers"),
            "remote_unverified_generation_marker_directory": str(
                runtime / "remote-unverified-generations"
            ),
            "compression_upload_marker_directory": str(
                runtime / "compression-upload-active"
            ),
            "storage_path": str(tmp_path),
        }
    )
    probe.pop("probe_sha256")
    probe["probe_sha256"] = canonical_sha256(probe)
    probe_path = tmp_path / "probe.json"
    probe_path.write_text(json.dumps(probe))

    policy = json.loads(POLICY.read_text())
    policy["local_probe_manifest_sha256"] = probe["probe_sha256"]
    policy.pop("policy_sha256")
    policy["policy_sha256"] = canonical_sha256(policy)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    initialized = _run(
        "init",
        "--policy",
        str(policy_path),
        "--state-dir",
        str(runtime),
        "--event-at-utc",
        "2026-07-22T00:00:00Z",
    )
    assert initialized.returncode == 0
    first = _run(
        "tick-local",
        "--policy",
        str(policy_path),
        "--state-dir",
        str(runtime),
        "--probe",
        str(probe_path),
        "--event-at-utc",
        "2026-07-22T00:01:00Z",
    )
    second = _run(
        "tick-local",
        "--policy",
        str(policy_path),
        "--state-dir",
        str(runtime),
        "--probe",
        str(probe_path),
        "--event-at-utc",
        "2026-07-22T00:02:00Z",
    )

    assert first.returncode == 0
    assert json.loads(first.stdout)["status"] == "UPDATED"
    assert second.returncode == 0
    second_body = json.loads(second.stdout)
    assert second_body["status"] == "NO_CHANGE"
    assert second_body["event_appended"] is False
    assert len(list((runtime / "events").iterdir())) == 2
