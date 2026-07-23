"""Session-independent launcher for one historical DOJO lifecycle transition."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from quant_rabbit import dojo_historical_train_control as control_plane
from quant_rabbit.dojo_historical_crash_supervisor import (
    advance_one_supervised_transition,
)
from quant_rabbit.dojo_historical_train_control import (
    DojoHistoricalTrainControlError,
)


LAUNCH_REQUEST_CONTRACT: Final = "QR_DOJO_HISTORICAL_SUPERVISOR_LAUNCH_REQUEST_V1"
LAUNCH_STARTED_CONTRACT: Final = "QR_DOJO_HISTORICAL_SUPERVISOR_LAUNCH_STARTED_V1"
LAUNCH_RESULT_CONTRACT: Final = "QR_DOJO_HISTORICAL_SUPERVISOR_LAUNCH_RESULT_V1"
SCHEMA_VERSION: Final = 1
START_HANDSHAKE_TIMEOUT_SECONDS: Final = 5.0

_AUTHORITY: Final = {
    "automatic_deployment_allowed": False,
    "broker_mutation_allowed": False,
    "live_permission": False,
    "order_authority": "NONE",
    "promotion_eligible": False,
}


class DojoHistoricalSupervisorError(ValueError):
    """A durable launch lease or append-only supervisor receipt is unsafe."""


def _sha_file(path: Path) -> dict[str, Any]:
    provided = Path(path)
    provided_state = provided.stat(follow_symlinks=False)
    if provided.is_symlink() or not stat.S_ISREG(provided_state.st_mode):
        raise DojoHistoricalSupervisorError(
            f"supervisor implementation path is unsafe: {provided}"
        )
    candidate = provided.resolve(strict=True)
    state = candidate.stat(follow_symlinks=False)
    if candidate.is_symlink() or not stat.S_ISREG(state.st_mode) or state.st_nlink != 1:
        raise DojoHistoricalSupervisorError(
            f"supervisor implementation is unsafe: {candidate}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    current = candidate.stat(follow_symlinks=False)
    if (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns) != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ):
        raise DojoHistoricalSupervisorError(
            f"supervisor implementation changed while hashed: {candidate}"
        )
    return {
        "absolute_path": str(candidate),
        "size_bytes": state.st_size,
        "sha256": digest.hexdigest(),
    }


def _supervisor_root(run_root: Path) -> Path:
    path = run_root / "supervisor"
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    state = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISDIR(state.st_mode):
        raise DojoHistoricalSupervisorError(
            "historical supervisor root must be a real directory"
        )
    return path


def _open_launch_lease(supervisor_root: Path) -> tuple[int, Path]:
    path = supervisor_root / ".historical-supervisor.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not stat.S_ISREG(named.st_mode)
            or named.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise DojoHistoricalSupervisorError(
                "historical supervisor lease is not a stable single-link file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise DojoHistoricalSupervisorError(
                "historical supervisor lease was replaced while acquired"
            )
        return descriptor, path
    except BaseException:
        os.close(descriptor)
        raise


def _release_launch_lease(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _validate_inherited_lease(descriptor: int, path: Path) -> dict[str, int]:
    try:
        opened = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DojoHistoricalSupervisorError(
            "inherited historical supervisor lease is unavailable"
        ) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or not stat.S_ISREG(named.st_mode)
        or named.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise DojoHistoricalSupervisorError(
            "inherited historical supervisor lease identity drifted"
        )
    return {"device": opened.st_dev, "inode": opened.st_ino}


def _read_request(path: Path) -> dict[str, Any]:
    try:
        request = control_plane._read_json(path, field="supervisor launch request")
    except control_plane.DojoHistoricalTrainControlError as exc:
        raise DojoHistoricalSupervisorError(str(exc)) from exc
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    expected = control_plane.canonical_sha256(body)
    expected_keys = {
        "contract",
        "schema_version",
        "nonce",
        "requested_at_utc",
        "parent_pid",
        "repo_root",
        "run_control_path",
        "output_root",
        "run_control_sha256",
        "control_manifest_sha256",
        "lease_path",
        "python_executable",
        "child_script",
        "supervisor_module",
        "allowed_transition_count",
        "partial_economics_reported",
        "trainer_action_allowed",
        "automatic_deployment_allowed",
        "broker_mutation_allowed",
        "live_permission",
        "order_authority",
        "promotion_eligible",
        "request_sha256",
    }
    if (
        set(request) != expected_keys
        or request.get("contract") != LAUNCH_REQUEST_CONTRACT
        or request.get("schema_version") != SCHEMA_VERSION
        or request.get("request_sha256") != expected
        or path.name != f"request-{expected}.json"
        or not isinstance(request.get("nonce"), str)
        or len(request["nonce"]) != 64
        or any(character not in "0123456789abcdef" for character in request["nonce"])
        or request.get("python_executable") != sys.executable
        or request.get("allowed_transition_count") != 1
        or request.get("partial_economics_reported") is not False
        or request.get("trainer_action_allowed") is not False
        or any(request.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoHistoricalSupervisorError("supervisor launch request is invalid")
    return request


def _validate_started(path: Path, *, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        started = control_plane._read_json(path, field="supervisor start receipt")
    except control_plane.DojoHistoricalTrainControlError as exc:
        raise DojoHistoricalSupervisorError(str(exc)) from exc
    body = {key: value for key, value in started.items() if key != "started_sha256"}
    expected_keys = {
        "contract",
        "schema_version",
        "request_sha256",
        "child_pid",
        "process_group_id",
        "session_id",
        "lease_identity",
        "detached_session",
        "partial_economics_reported",
        "trainer_action_allowed",
        "automatic_deployment_allowed",
        "broker_mutation_allowed",
        "live_permission",
        "order_authority",
        "promotion_eligible",
        "started_sha256",
    }
    pid = started.get("child_pid")
    lease = started.get("lease_identity")
    if (
        set(started) != expected_keys
        or started.get("contract") != LAUNCH_STARTED_CONTRACT
        or started.get("schema_version") != SCHEMA_VERSION
        or started.get("request_sha256") != request["request_sha256"]
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid < 1
        or started.get("process_group_id") != pid
        or started.get("session_id") != pid
        or started.get("detached_session") is not True
        or not isinstance(lease, Mapping)
        or set(lease) != {"device", "inode"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in lease.values()
        )
        or started.get("partial_economics_reported") is not False
        or started.get("trainer_action_allowed") is not False
        or started.get("started_sha256") != control_plane.canonical_sha256(body)
        or any(started.get(key) != value for key, value in _AUTHORITY.items())
    ):
        raise DojoHistoricalSupervisorError("supervisor start receipt is invalid")
    return started


def _result_receipt(
    *, request: Mapping[str, Any], status: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    body = {
        "contract": LAUNCH_RESULT_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "request_sha256": request["request_sha256"],
        "status": status,
        "payload": dict(payload),
        "partial_economics_reported": False,
        "trainer_action_allowed": False,
        **_AUTHORITY,
    }
    return {**body, "result_sha256": control_plane.canonical_sha256(body)}


def run_supervisor_child(
    *,
    repo_root: Path,
    run_control_path: Path,
    request_path: Path,
    lease_path: Path,
    lease_descriptor: int,
) -> int:
    """Child entrypoint; it owns the inherited lease for the whole transition."""

    request = _read_request(request_path)
    expected_request_path = (
        Path(request["output_root"])
        / "supervisor"
        / "requests"
        / f"request-{request['request_sha256']}.json"
    )
    expected_lease_path = (
        Path(request["output_root"]) / "supervisor" / ".historical-supervisor.lock"
    )
    if (
        request["repo_root"] != str(repo_root.resolve(strict=True))
        or request["run_control_path"] != str(run_control_path.resolve(strict=True))
        or request["lease_path"] != str(lease_path)
        or request_path != expected_request_path
        or lease_path != expected_lease_path
        or request["run_control_sha256"]
        != hashlib.sha256(run_control_path.read_bytes()).hexdigest()
        or request["child_script"]
        != _sha_file(repo_root / "scripts" / "run-dojo-historical-supervisor.py")
        or request["supervisor_module"] != _sha_file(Path(__file__))
    ):
        raise DojoHistoricalSupervisorError(
            "supervisor child arguments differ from the launch request"
        )
    lease_identity = _validate_inherited_lease(lease_descriptor, lease_path)
    supervisor_root = request_path.parent.parent
    started_body = {
        "contract": LAUNCH_STARTED_CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "request_sha256": request["request_sha256"],
        "child_pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "session_id": os.getsid(0),
        "lease_identity": lease_identity,
        "detached_session": os.getsid(0) == os.getpid(),
        "partial_economics_reported": False,
        "trainer_action_allowed": False,
        **_AUTHORITY,
    }
    started = {
        **started_body,
        "started_sha256": control_plane.canonical_sha256(started_body),
    }
    started_path = (
        supervisor_root / "started" / (f"started-{request['request_sha256']}.json")
    )
    control_plane._write_once(started_path, started)
    try:
        operation = advance_one_supervised_transition(
            repo_root=repo_root, run_control_path=run_control_path
        )
        receipt = _result_receipt(
            request=request, status="TRANSITION_FINISHED", payload=operation
        )
        exit_code = 0
    except Exception as exc:
        receipt = _result_receipt(
            request=request,
            status="TRANSITION_REJECTED",
            payload={"error_type": type(exc).__name__, "error": str(exc)},
        )
        exit_code = 2
    result_path = (
        supervisor_root
        / "results"
        / (f"result-{request['request_sha256']}-{receipt['result_sha256']}.json")
    )
    control_plane._write_once(result_path, receipt)
    print(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return exit_code


def launch_supervised_transition(
    *,
    repo_root: Path,
    run_control_path: Path,
    child_script: Path,
    handshake_timeout_seconds: float = START_HANDSHAKE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Launch one detached child and return after its durable start handshake."""

    control, run_root, _, _, _, _, _ = control_plane._load_generation(
        repo_root=repo_root, run_control_path=run_control_path, operation="custody"
    )
    supervisor_root = _supervisor_root(run_root)
    for child in ("requests", "started", "results", "logs"):
        path = supervisor_root / child
        path.mkdir(mode=0o700, exist_ok=True)
        state = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISDIR(state.st_mode):
            raise DojoHistoricalSupervisorError(
                f"supervisor artifact directory is unsafe: {child}"
            )
    try:
        lease_descriptor, lease_path = _open_launch_lease(supervisor_root)
    except BlockingIOError:
        return {
            "status": "SUPERVISOR_ALREADY_RUNNING",
            "output_root": str(run_root),
            "kernel_lease_proves_live_owner": True,
            "new_process_started": False,
            "partial_economics_reported": False,
            "trainer_action_allowed": False,
            **_AUTHORITY,
        }
    transferred = False
    try:
        expected_child_script = (
            Path(repo_root) / "scripts" / "run-dojo-historical-supervisor.py"
        ).resolve(strict=True)
        if child_script.resolve(strict=True) != expected_child_script:
            raise DojoHistoricalSupervisorError(
                "detached historical supervisor child script is not canonical"
            )
        executable = sys.executable
        request_body = {
            "contract": LAUNCH_REQUEST_CONTRACT,
            "schema_version": SCHEMA_VERSION,
            "nonce": secrets.token_hex(32),
            "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_pid": os.getpid(),
            "repo_root": str(Path(repo_root).resolve(strict=True)),
            "run_control_path": str(Path(run_control_path).resolve(strict=True)),
            "output_root": str(run_root),
            "run_control_sha256": hashlib.sha256(
                Path(run_control_path).read_bytes()
            ).hexdigest(),
            "control_manifest_sha256": control_plane._read_json(
                run_root / "control-manifest.json", field="control manifest"
            )["manifest_sha256"],
            "lease_path": str(lease_path),
            "python_executable": executable,
            "child_script": _sha_file(child_script),
            "supervisor_module": _sha_file(Path(__file__)),
            "allowed_transition_count": 1,
            "partial_economics_reported": False,
            "trainer_action_allowed": False,
            **_AUTHORITY,
        }
        request = {
            **request_body,
            "request_sha256": control_plane.canonical_sha256(request_body),
        }
        request_path = (
            supervisor_root / "requests" / (f"request-{request['request_sha256']}.json")
        )
        control_plane._write_once(request_path, request)
        log_path = supervisor_root / "logs" / f"{request['request_sha256']}.log"
        log_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        log_descriptor = os.open(log_path, log_flags, 0o600)
        command: Sequence[str] = (
            executable,
            str(child_script.resolve(strict=True)),
            "_child",
            "--run-control",
            str(Path(run_control_path).resolve(strict=True)),
            "--request",
            str(request_path),
            "--lease-path",
            str(lease_path),
            "--lease-fd",
            str(lease_descriptor),
        )
        try:
            with os.fdopen(log_descriptor, "ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=Path(repo_root).resolve(strict=True),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(lease_descriptor,),
                )
        except BaseException:
            try:
                log_path.unlink()
            except OSError:
                pass
            raise
        # Do not unlock this shared open file description.  Closing only the
        # parent's duplicate transfers the kernel lease to the detached child.
        os.close(lease_descriptor)
        transferred = True
        started_path = (
            supervisor_root / "started" / (f"started-{request['request_sha256']}.json")
        )
        deadline = time.monotonic() + handshake_timeout_seconds
        while time.monotonic() < deadline:
            if started_path.is_file():
                started = _validate_started(started_path, request=request)
                return {
                    "status": "SUPERVISOR_LAUNCHED",
                    "output_root": str(run_root),
                    "request_sha256": request["request_sha256"],
                    "child_pid": started["child_pid"],
                    "detached_session": started["detached_session"],
                    "log_path": str(log_path),
                    "new_process_started": True,
                    "partial_economics_reported": False,
                    "trainer_action_allowed": False,
                    **_AUTHORITY,
                }
            return_code = process.poll()
            if return_code is not None:
                raise DojoHistoricalSupervisorError(
                    "detached supervisor exited before its start handshake: "
                    f"exit_code={return_code}, log={log_path}"
                )
            time.sleep(0.05)
        raise DojoHistoricalSupervisorError(
            "detached supervisor did not publish its start handshake in time: "
            f"pid={process.pid}, log={log_path}"
        )
    finally:
        if not transferred:
            _release_launch_lease(lease_descriptor)


def supervisor_status(*, repo_root: Path, run_control_path: Path) -> dict[str, Any]:
    """Report kernel lease state plus the compact sealed generation status."""

    _, run_root, _, _, _, _, _ = control_plane._load_generation(
        repo_root=repo_root, run_control_path=run_control_path, operation="custody"
    )
    supervisor_root = _supervisor_root(run_root)
    try:
        descriptor, _ = _open_launch_lease(supervisor_root)
    except BlockingIOError:
        running = True
    else:
        running = False
        _release_launch_lease(descriptor)
    try:
        generation = control_plane.generation_status(
            repo_root=repo_root, run_control_path=run_control_path
        )
    except DojoHistoricalTrainControlError as exc:
        if not running or "old historical run still owns its lock" not in str(exc):
            raise
        generation = {
            "status": "RUNNING",
            "output_root": str(run_root),
            "status_probe_deferred_to_live_child": True,
            "status_probe_error_type": type(exc).__name__,
            "status_probe_error": str(exc),
        }
    return {
        "status": "SUPERVISOR_RUNNING" if running else "SUPERVISOR_IDLE",
        "kernel_lease_owned": running,
        "generation": generation,
        "partial_economics_reported": False,
        "trainer_action_allowed": False,
        **_AUTHORITY,
    }


__all__ = [
    "DojoHistoricalSupervisorError",
    "launch_supervised_transition",
    "run_supervisor_child",
    "supervisor_status",
]
