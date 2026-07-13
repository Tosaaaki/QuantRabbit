from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


_PROCESS_BIRTH_RE = re.compile(
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[ \t]+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[ \t]+"
    r"[0-9]{1,2}[ \t]+[0-9]{2}:[0-9]{2}:[0-9]{2}[ \t]+[0-9]{4}"
)


@dataclass(frozen=True)
class LiveLockInspection:
    status: str
    pid: int | None
    command: str | None
    active: bool
    reapable: bool


class LiveLockAlreadyHeld(RuntimeError):
    def __init__(self, inspection: LiveLockInspection) -> None:
        self.inspection = inspection
        detail = f" pid={inspection.pid}" if inspection.pid is not None else ""
        super().__init__(f"live runtime lock is busy{detail}; status={inspection.status}")


def generation_guard_path(lock_dir: Path) -> Path:
    return Path(f"{lock_dir}.acquire.guard")


@contextmanager
def live_lock_generation_guard(lock_dir: Path) -> Iterator[None]:
    """Serialize one lock-directory generation transition across processes."""

    guard_path = generation_guard_path(lock_dir)
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(guard_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def write_live_lock_owner(lock_dir: Path, command: str) -> str:
    token = f"{os.getpid()}:{uuid.uuid4().hex}"
    (lock_dir / "pid").write_text(f"{os.getpid()}\n")
    (lock_dir / "token").write_text(f"{token}\n")
    (lock_dir / "command").write_text(f"{command}\n")
    process_started_at = _process_started_at(os.getpid())
    if process_started_at is not None:
        (lock_dir / "process_started_at").write_text(f"{process_started_at}\n")
    (lock_dir / "started_at_utc").write_text(
        f"{datetime.now(timezone.utc).isoformat()}\n"
    )
    return token


def inspect_live_lock(lock_dir: Path) -> LiveLockInspection:
    pid_text = _read_one_line(lock_dir / "pid")
    command = _read_one_line(lock_dir / "command")
    if pid_text is None or not pid_text.isdigit():
        return LiveLockInspection("MISSING_OWNER_METADATA", None, command, False, True)

    pid = int(pid_text)
    state = _process_state(pid)
    if state is not None and state.startswith("Z"):
        return LiveLockInspection("DEFUNCT_OWNER", pid, command, False, True)
    if not _pid_exists(pid):
        return LiveLockInspection("DEAD_OWNER", pid, command, False, True)

    recorded_birth = _read_one_line(lock_dir / "process_started_at")
    current_birth = _process_started_at(pid)
    recorded_valid = _process_birth_is_valid(recorded_birth)
    current_valid = _process_birth_is_valid(current_birth)
    if recorded_valid and current_valid:
        recorded_birth = _canonical_process_birth(recorded_birth)
        current_birth = _canonical_process_birth(current_birth)
    if recorded_valid and current_valid and recorded_birth != current_birth:
        return LiveLockInspection("RECYCLED_PID", pid, command, False, True)
    if not recorded_valid or not current_valid:
        return LiveLockInspection("ACTIVE_IDENTITY_UNAVAILABLE", pid, command, True, False)
    return LiveLockInspection("ACTIVE", pid, command, True, False)


def acquire_live_lock_owner(
    lock_dir: Path,
    command: str,
    *,
    init_grace_seconds: float | None = None,
) -> str:
    grace = _initialization_grace_seconds(init_grace_seconds)
    missing_owner_observed = False
    lock_dir.parent.mkdir(parents=True, exist_ok=True)

    while True:
        should_wait = False
        with live_lock_generation_guard(lock_dir):
            try:
                lock_dir.mkdir()
            except FileExistsError as exc:
                inspection = inspect_live_lock(lock_dir)
                if inspection.active or not inspection.reapable:
                    raise LiveLockAlreadyHeld(inspection) from exc
                if inspection.status == "MISSING_OWNER_METADATA" and not missing_owner_observed:
                    should_wait = True
                else:
                    shutil.rmtree(lock_dir, ignore_errors=True)
                    try:
                        lock_dir.mkdir()
                    except FileExistsError as retry_exc:
                        raise LiveLockAlreadyHeld(inspect_live_lock(lock_dir)) from retry_exc
            if not should_wait:
                return write_live_lock_owner(lock_dir, command)
        missing_owner_observed = True
        time.sleep(grace)


def inherited_live_lock_is_valid(lock_dir: Path, inherited_token: str | None = None) -> bool:
    with live_lock_generation_guard(lock_dir):
        inspection = inspect_live_lock(lock_dir)
        if not inspection.active or inspection.pid not in {os.getpid(), os.getppid()}:
            return False
        if inherited_token is None:
            return True
        return _read_one_line(lock_dir / "token") == inherited_token


def release_live_lock_owner(lock_dir: Path, token: str, *, owner_pid: int | None = None) -> bool:
    """Remove only the exact lock generation acquired by this caller."""

    expected_pid = os.getpid() if owner_pid is None else owner_pid
    with live_lock_generation_guard(lock_dir):
        try:
            current_token = (lock_dir / "token").read_text().strip()
            current_pid = int((lock_dir / "pid").read_text().strip())
        except (OSError, TypeError, ValueError):
            return False
        if current_token != token or current_pid != expected_pid:
            return False
        shutil.rmtree(lock_dir)
        return True


def _initialization_grace_seconds(explicit: float | None) -> float:
    value: object = explicit
    if value is None:
        value = os.environ.get("QR_LIVE_LOCK_INIT_GRACE_SECONDS", "1")
    try:
        grace = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid live lock initialization grace: {value!r}") from exc
    if not 0.0 <= grace <= 10.0:
        raise ValueError(f"invalid live lock initialization grace: {value!r}")
    return grace


def _read_one_line(path: Path) -> str | None:
    try:
        value = path.read_text()
    except (OSError, UnicodeError):
        return None
    if "\x00" in value or "\r" in value:
        return None
    if value.endswith("\n"):
        value = value[:-1]
    if "\n" in value:
        return None
    value = value.strip()
    return value or None


def _process_birth_is_valid(value: str | None) -> bool:
    return value is not None and _PROCESS_BIRTH_RE.fullmatch(value) is not None


def _canonical_process_birth(value: str) -> str:
    return " ".join(value.split())


def _process_started_at(pid: int) -> str | None:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if result.returncode != 0 or not _process_birth_is_valid(value):
        return None
    return _canonical_process_birth(value)


def _process_state(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    state = result.stdout.strip()
    return state if result.returncode == 0 and state else None


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
