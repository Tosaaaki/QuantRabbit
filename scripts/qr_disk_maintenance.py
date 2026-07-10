#!/usr/bin/env python3
"""Bound QuantRabbit runtime disk growth without deleting trading evidence.

The maintenance surface is deliberately small and auditable:

* verify and gzip old OANDA replay ``.jsonl`` files;
* remove superseded atomic-write debris below runtime ``data/docs/logs``;
* remove old, process-owned profiling database copies directly below
  ``/private/tmp``;
* rotate only the launchd stdout/stderr files named in this module; and
* expire only regenerable files in the dedicated guardian cache/temp roots.

Broker/runtime state, evidence databases, forecast/projection artifacts,
guardian sessions, plugins, skills, and Codex state databases are never cleanup
targets. The command never trades or grants trade permission.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import stat as stat_module
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_HISTORY_DIR = Path("logs/replay/oanda_history")
DEFAULT_REPORT_PATH = Path("logs/disk_maintenance_report.json")
SYSTEM_TEMP_ROOT = Path("/private/tmp")

# These are host-operability thresholds, not market/risk parameters. Five GiB
# leaves room for a full evidence refresh and atomic report writes; two GiB is
# the existing QuantRabbit P0 floor at which normal cycle writes have failed.
GIB = 1024**3
OPERATING_FLOOR_BYTES = 5 * GIB
P0_FREE_BYTES = 2 * GIB

# Report arrays are bounded so the maintenance report itself cannot become a
# disk-growth source during a debris incident.
MAX_REPORT_ITEMS = 200

PROJECT_ATOMIC_ROOTS = (Path("data"), Path("docs"), Path("logs"))
PROJECT_ATOMIC_SUFFIXES = (".tmp", ".partial", ".bak")
PROJECT_SCAN_PRUNE_NAMES = {".git", "__pycache__", "codex_guardian_home"}
# A .bak may be deliberate rollback material rather than an interrupted atomic
# write, so it keeps a full seven-day recovery horizon even during P0 cleanup.
PROJECT_BACKUP_MIN_AGE = timedelta(days=7)

# Only these exact launchd files may be rotated. Strategy journals and trading
# logs are intentionally absent because they are execution evidence.
KNOWN_LAUNCHD_LOGS = (
    "guardian_wake_dispatcher.launchd.log",
    "guardian_wake_dispatcher.launchd.err",
    "position_guardian.launchd.log",
    "position_guardian.launchd.err",
    "qr_trader_run_watchdog.launchd.log",
    "qr_trader_run_watchdog.launchd.err",
)

# External deletion is limited to direct children of /private/tmp, current-user
# ownership, qr_ prefix, and an explicit diagnostic/profiling database token.
EXTERNAL_DIAGNOSTIC_RE = re.compile(
    r"^qr_[A-Za-z0-9_.-]+\.(?:db|sqlite|sqlite3)(?:-(?:wal|shm))?$",
    re.IGNORECASE,
)
EXTERNAL_DIAGNOSTIC_TOKENS = {"profile", "profiling", "diagnostic", "benchmark"}

GUARDIAN_HOME = Path("data/codex_guardian_home")
# These trees are reproducible by the read-only wake CLI. Guardian sessions are
# deliberately excluded: they can be useful audit evidence and are not proven
# redundant with the durable receipt/review artifacts.
GUARDIAN_EPHEMERAL_POLICIES = {
    Path("cache"): "cache",
    Path(".tmp"): "temp",
    Path("tmp"): "temp",
    Path("shell_snapshots"): "cache",
}


@dataclass(frozen=True)
class FilePlan:
    path: Path
    allowed_root: Path
    size_bytes: int
    age_seconds: float
    device: int
    inode: int
    mtime_ns: int
    category: str
    proof_path: Path | None = None
    proof_device: int | None = None
    proof_inode: int | None = None
    proof_size_bytes: int | None = None
    proof_mtime_ns: int | None = None


@dataclass(frozen=True)
class ContentStats:
    size_bytes: int
    row_count: int
    sha256: str


class MaintenanceSkip(RuntimeError):
    """A safe no-op caused by a race, open file, or allowlist check."""


class OpenFileInspector:
    """Best-effort lsof guard which fails closed when availability is unknown."""

    def __init__(self, executable: str | None = None) -> None:
        if executable is None:
            executable = shutil.which("lsof")
            if executable is None and Path("/usr/sbin/lsof").is_file():
                executable = "/usr/sbin/lsof"
        self.executable = executable

    @property
    def available(self) -> bool:
        return bool(self.executable)

    def state(self, path: Path) -> str:
        if not self.executable:
            return "UNKNOWN"
        try:
            result = subprocess.run(
                [self.executable, "-t", "--", str(path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "UNKNOWN"
        if result.returncode == 0:
            return "OPEN"
        if result.returncode == 1:
            return "CLOSED"
        return "UNKNOWN"


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"maintenance root is not a directory: {root}", file=sys.stderr)
        return 2

    try:
        lock_path = _resolve_under_root(root, Path("logs/.qr_disk_maintenance.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("disk maintenance already running")
        return 75
    except (OSError, ValueError) as exc:
        print(f"cannot acquire maintenance lock: {exc}", file=sys.stderr)
        return 2

    try:
        return _run_locked(args, root)
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def _run_locked(args: argparse.Namespace, root: Path) -> int:
    now = datetime.now(timezone.utc)
    before = _disk_usage(root)
    emergency_mode = before["free_bytes"] < P0_FREE_BYTES
    inspector = OpenFileInspector()

    # Emergency horizons are still conservative: they cover files older than
    # the current/previous runtime cycles, never current state or evidence.
    atomic_age = timedelta(
        hours=6 if emergency_mode else float(args.prune_atomic_hours)
    )
    diagnostic_age = timedelta(
        minutes=30 if emergency_mode else float(args.prune_diagnostic_hours) * 60
    )
    guardian_cache_age = timedelta(
        hours=24 if emergency_mode else float(args.guardian_cache_hours)
    )
    guardian_temp_age = timedelta(
        hours=1 if emergency_mode else float(args.guardian_temp_hours)
    )
    replay_temp_age = timedelta(
        hours=6 if emergency_mode else float(args.prune_temp_days) * 24
    )
    log_max_bytes = int(
        (2.0 if emergency_mode else float(args.log_max_mb)) * 1024 * 1024
    )

    report: dict[str, Any] = {
        "generated_at_utc": _iso(now),
        "root": str(root),
        "apply": bool(args.apply),
        "emergency_mode": emergency_mode,
        "operating_floor_bytes": OPERATING_FLOOR_BYTES,
        "p0_free_bytes": P0_FREE_BYTES,
        "pressure_before": _pressure_state(before["free_bytes"]),
        "min_age_minutes": float(args.min_age_minutes),
        "min_size_mb": float(args.min_size_mb),
        "history_dirs": [str(item) for item in args.history_dir],
        "effective_policy": {
            "project_atomic_age_seconds": atomic_age.total_seconds(),
            "external_diagnostic_age_seconds": diagnostic_age.total_seconds(),
            "external_diagnostic_cleanup_enabled": bool(args.prune_external_diagnostics),
            "guardian_cache_age_seconds": guardian_cache_age.total_seconds(),
            "guardian_temp_age_seconds": guardian_temp_age.total_seconds(),
            "replay_temp_age_seconds": replay_temp_age.total_seconds(),
            "guardian_sessions": "PRESERVED_AUDIT_EVIDENCE",
            "launchd_log_max_bytes": log_max_bytes,
        },
        "disk_before": before,
        "open_file_check_available": inspector.available,
        "compressed": [],
        "compression_candidates": [],
        "removed_stale_temp": [],
        "stale_temp_candidates": [],
        "removed_project_atomic": [],
        "project_atomic_candidates": [],
        "removed_external_diagnostics": [],
        "external_diagnostic_candidates": [],
        "removed_guardian_ephemeral": [],
        "guardian_ephemeral_candidates": [],
        "rotated_logs": [],
        "log_rotation_candidates": [],
        "skipped_open": [],
        "skipped_safety": [],
        "errors": [],
        "item_counts": {},
        "byte_totals": {},
        "truncated_items": {},
    }

    try:
        replay_candidates = _compression_candidates(
            root,
            args.history_dir,
            now=now,
            min_age=timedelta(minutes=float(args.min_age_minutes)),
            min_size_bytes=int(float(args.min_size_mb) * 1024 * 1024),
        )
        replay_temp_candidates = _stale_temp_candidates(
            root,
            args.history_dir,
            now=now,
            min_age=replay_temp_age,
        )
        project_atomic_candidates = _project_atomic_candidates(
            root, now=now, min_age=atomic_age, report=report
        )
        external_diagnostic_candidates = (
            _external_diagnostic_candidates(
                SYSTEM_TEMP_ROOT,
                now=now,
                min_age=diagnostic_age,
                trusted_temp_root=SYSTEM_TEMP_ROOT,
                report=report,
            )
            if args.prune_external_diagnostics
            else []
        )
        guardian_candidates = _guardian_ephemeral_candidates(
            root,
            now=now,
            cache_age=guardian_cache_age,
            temp_age=guardian_temp_age,
            report=report,
        )
        log_candidates = _log_rotation_candidates(
            root, now=now, max_size_bytes=log_max_bytes, report=report
        )
    except (OSError, ValueError) as exc:
        _record(
            report,
            "errors",
            {"stage": "candidate_scan", "error": f"{type(exc).__name__}: {exc}"},
        )
        replay_candidates = []
        replay_temp_candidates = []
        project_atomic_candidates = []
        external_diagnostic_candidates = []
        guardian_candidates = []
        log_candidates = []

    _process_replay_compression(
        replay_candidates, args.apply, root, inspector, report
    )
    _process_unlinks(
        replay_temp_candidates,
        args.apply,
        root,
        inspector,
        report,
        applied_key="removed_stale_temp",
        candidate_key="stale_temp_candidates",
    )
    _process_unlinks(
        project_atomic_candidates,
        args.apply,
        root,
        inspector,
        report,
        applied_key="removed_project_atomic",
        candidate_key="project_atomic_candidates",
    )
    _process_unlinks(
        external_diagnostic_candidates,
        args.apply,
        root,
        inspector,
        report,
        applied_key="removed_external_diagnostics",
        candidate_key="external_diagnostic_candidates",
    )
    _process_unlinks(
        guardian_candidates,
        args.apply,
        root,
        inspector,
        report,
        applied_key="removed_guardian_ephemeral",
        candidate_key="guardian_ephemeral_candidates",
    )
    _process_log_rotation(log_candidates, args.apply, root, inspector, report)

    after = _disk_usage(root)
    report["disk_after"] = after
    report["pressure_after"] = _pressure_state(after["free_bytes"])
    report["floor_met_after"] = after["free_bytes"] >= OPERATING_FLOOR_BYTES
    report["bytes_to_operating_floor"] = max(
        0, OPERATING_FLOOR_BYTES - after["free_bytes"]
    )
    report["summary"] = _summary(report)

    try:
        report_path = _resolve_under_root(root, args.report_path)
        _write_report_atomic(report_path, report)
    except (OSError, ValueError) as exc:
        print(f"failed to write maintenance report: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    print(f"wrote {report_path}")
    if _count(report, "errors"):
        return 1
    if (
        args.apply
        and report["pressure_after"] == "P0"
        and not args.allow_p0_no_reclaim
    ):
        return 3
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--history-dir",
        type=Path,
        action="append",
        default=[DEFAULT_HISTORY_DIR],
        help="history directory below --root",
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--min-age-minutes", type=_nonnegative_float, default=30.0)
    parser.add_argument("--min-size-mb", type=_nonnegative_float, default=1.0)
    parser.add_argument("--prune-temp-days", type=_nonnegative_float, default=2.0)
    parser.add_argument("--prune-atomic-hours", type=_nonnegative_float, default=48.0)
    parser.add_argument("--prune-diagnostic-hours", type=_nonnegative_float, default=6.0)
    parser.add_argument(
        "--prune-external-diagnostics",
        action="store_true",
        help="opt in to strict current-user qr_* profiling DB cleanup in /private/tmp",
    )
    parser.add_argument("--guardian-cache-hours", type=_nonnegative_float, default=168.0)
    parser.add_argument("--guardian-temp-hours", type=_nonnegative_float, default=24.0)
    parser.add_argument("--log-max-mb", type=_positive_float, default=8.0)
    parser.add_argument("--apply", action="store_true", help="perform maintenance; otherwise dry-run")
    parser.add_argument(
        "--allow-p0-no-reclaim",
        action="store_true",
        help="test/operator override: return zero when P0 has no safe reclaim target",
    )
    return parser.parse_args()


def _compression_candidates(
    root: Path,
    history_dirs: list[Path],
    *,
    now: datetime,
    min_age: timedelta,
    min_size_bytes: int,
) -> list[FilePlan]:
    out: list[FilePlan] = []
    for history_dir in history_dirs:
        base = _resolve_under_root(root, history_dir)
        if not base.exists():
            continue
        if not base.is_dir():
            raise ValueError(f"history path is not a directory: {base}")
        for path in _walk_regular_files(base):
            if not path.name.endswith(".jsonl") or _is_temp_path(path):
                continue
            file_stat = _safe_lstat(path)
            if file_stat is None:
                continue
            gz_path = path.with_name(f"{path.name}.gz")
            if gz_path.exists() and _safe_lstat(gz_path) is None:
                continue
            age_seconds = _age_seconds(file_stat.st_mtime, now)
            if age_seconds < min_age.total_seconds() or file_stat.st_size < min_size_bytes:
                continue
            out.append(_plan(path, base, file_stat, age_seconds, "replay_jsonl"))
    return sorted(out, key=lambda item: str(item.path))


def _stale_temp_candidates(
    root: Path,
    history_dirs: list[Path],
    *,
    now: datetime,
    min_age: timedelta,
) -> list[FilePlan]:
    out: list[FilePlan] = []
    for history_dir in history_dirs:
        base = _resolve_under_root(root, history_dir)
        if not base.exists():
            continue
        for path in _walk_regular_files(base):
            if not path.name.endswith((".tmp", ".partial")):
                continue
            file_stat = _safe_lstat(path)
            if file_stat is None:
                continue
            age_seconds = _age_seconds(file_stat.st_mtime, now)
            if age_seconds < min_age.total_seconds():
                continue
            out.append(_plan(path, base, file_stat, age_seconds, "replay_temp"))
    return sorted(out, key=lambda item: str(item.path))


def _project_atomic_candidates(
    root: Path,
    *,
    now: datetime,
    min_age: timedelta,
    report: dict[str, Any],
) -> list[FilePlan]:
    out: list[FilePlan] = []
    for relative_base in PROJECT_ATOMIC_ROOTS:
        base = _resolve_under_root(root, relative_base)
        if not base.exists():
            continue
        for path in _walk_regular_files(base, prune_names=PROJECT_SCAN_PRUNE_NAMES):
            if _is_within(path, root / GUARDIAN_HOME) or _is_within(path, root / DEFAULT_HISTORY_DIR):
                continue
            suffix = next(
                (item for item in PROJECT_ATOMIC_SUFFIXES if path.name.endswith(item)),
                None,
            )
            if suffix is None:
                continue
            file_stat = _safe_lstat(path)
            if file_stat is None:
                _record(report, "skipped_safety", {"path": _rel(path, root), "reason": "NOT_REGULAR_OR_SYMLINK"})
                continue
            age_seconds = _age_seconds(file_stat.st_mtime, now)
            required_age = max(PROJECT_BACKUP_MIN_AGE, min_age) if suffix == ".bak" else min_age
            if age_seconds < required_age.total_seconds():
                continue
            target_name = path.name[: -len(suffix)]
            target = path.with_name(target_name)
            if not target_name or _protected_atomic_target(target):
                _record(report, "skipped_safety", {"path": _rel(path, root), "reason": "PROTECTED_RUNTIME_EVIDENCE"})
                continue
            target_stat = _safe_lstat(target)
            if target_stat is None or target_stat.st_mtime_ns <= file_stat.st_mtime_ns:
                _record(report, "skipped_safety", {"path": _rel(path, root), "reason": "NO_NEWER_CANONICAL_TARGET"})
                continue
            out.append(
                _plan(
                    path,
                    base,
                    file_stat,
                    age_seconds,
                    "project_atomic",
                    proof_path=target,
                    proof_stat=target_stat,
                )
            )
    return sorted(out, key=lambda item: str(item.path))


def _external_diagnostic_candidates(
    temp_root: Path,
    *,
    now: datetime,
    min_age: timedelta,
    trusted_temp_root: Path,
    report: dict[str, Any],
) -> list[FilePlan]:
    resolved = temp_root.resolve()
    trusted = trusted_temp_root.resolve()
    if resolved != trusted:
        raise ValueError(f"external diagnostic root is not trusted: {resolved}")
    if not resolved.is_dir():
        return []

    out: list[FilePlan] = []
    for entry in os.scandir(resolved):
        path = Path(entry.path)
        if not _external_diagnostic_name_allowed(entry.name):
            continue
        file_stat = _safe_lstat(path)
        if file_stat is None:
            _record(report, "skipped_safety", {"path": str(path), "reason": "EXTERNAL_NOT_REGULAR_OR_SYMLINK"})
            continue
        if file_stat.st_uid != os.getuid():
            _record(report, "skipped_safety", {"path": str(path), "reason": "EXTERNAL_NOT_CURRENT_USER_OWNED"})
            continue
        age_seconds = _age_seconds(file_stat.st_mtime, now)
        if age_seconds < min_age.total_seconds():
            continue
        out.append(_plan(path, resolved, file_stat, age_seconds, "external_diagnostic"))
    return sorted(out, key=lambda item: str(item.path))


def _external_diagnostic_name_allowed(name: str) -> bool:
    if not EXTERNAL_DIAGNOSTIC_RE.fullmatch(name):
        return False
    base = re.sub(r"-(?:wal|shm)$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r"\.(?:db|sqlite|sqlite3)$", "", base, flags=re.IGNORECASE)
    tokens = {token.lower() for token in re.split(r"[_.-]+", stem[3:]) if token}
    return bool(tokens & EXTERNAL_DIAGNOSTIC_TOKENS)


def _guardian_ephemeral_candidates(
    root: Path,
    *,
    now: datetime,
    cache_age: timedelta,
    temp_age: timedelta,
    report: dict[str, Any],
) -> list[FilePlan]:
    guardian_home = _resolve_under_root(root, GUARDIAN_HOME)
    if not guardian_home.exists():
        return []
    out: list[FilePlan] = []
    for relative_base, policy in GUARDIAN_EPHEMERAL_POLICIES.items():
        base = _resolve_under_root(guardian_home, relative_base)
        if not base.exists():
            continue
        min_age = temp_age if policy == "temp" else cache_age
        for path in _walk_regular_files(base):
            file_stat = _safe_lstat(path)
            if file_stat is None:
                _record(report, "skipped_safety", {"path": _rel(path, root), "reason": "GUARDIAN_NOT_REGULAR_OR_SYMLINK"})
                continue
            age_seconds = _age_seconds(file_stat.st_mtime, now)
            if age_seconds < min_age.total_seconds():
                continue
            out.append(_plan(path, base, file_stat, age_seconds, f"guardian_{policy}"))
    return sorted(out, key=lambda item: str(item.path))


def _log_rotation_candidates(
    root: Path,
    *,
    now: datetime,
    max_size_bytes: int,
    report: dict[str, Any],
) -> list[FilePlan]:
    logs = _resolve_under_root(root, Path("logs"))
    if not logs.exists():
        return []
    out: list[FilePlan] = []
    for name in KNOWN_LAUNCHD_LOGS:
        path = logs / name
        if not path.exists():
            continue
        file_stat = _safe_lstat(path)
        if file_stat is None:
            _record(report, "skipped_safety", {"path": _rel(path, root), "reason": "LOG_NOT_REGULAR_OR_SYMLINK"})
            continue
        if file_stat.st_size <= max_size_bytes:
            continue
        out.append(
            _plan(path, logs, file_stat, _age_seconds(file_stat.st_mtime, now), "launchd_log")
        )
    return out


def _process_replay_compression(
    plans: Iterable[FilePlan],
    apply: bool,
    root: Path,
    inspector: OpenFileInspector,
    report: dict[str, Any],
) -> None:
    for plan in plans:
        item = _plan_item(plan, root)
        if not apply:
            _record(report, "compression_candidates", item)
            continue
        try:
            item.update(_compress_jsonl(plan, inspector))
            _record(report, "compressed", item)
        except MaintenanceSkip as exc:
            item["reason"] = str(exc)
            target = "skipped_open" if str(exc).startswith("OPEN_FILE") or str(exc).startswith("OPEN_CHECK") else "skipped_safety"
            _record(report, target, item)
        except Exception as exc:  # noqa: BLE001 - report and preserve source.
            item["error"] = f"{type(exc).__name__}: {exc}"
            _record(report, "errors", item)


def _process_unlinks(
    plans: Iterable[FilePlan],
    apply: bool,
    root: Path,
    inspector: OpenFileInspector,
    report: dict[str, Any],
    *,
    applied_key: str,
    candidate_key: str,
) -> None:
    for plan in plans:
        item = _plan_item(plan, root)
        if not apply:
            _record(report, candidate_key, item)
            continue
        try:
            _safe_unlink(plan, inspector)
            item["reclaimed_bytes"] = plan.size_bytes
            _record(report, applied_key, item)
        except MaintenanceSkip as exc:
            item["reason"] = str(exc)
            target = "skipped_open" if str(exc).startswith("OPEN_FILE") or str(exc).startswith("OPEN_CHECK") else "skipped_safety"
            _record(report, target, item)
        except OSError as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            _record(report, "errors", item)


def _process_log_rotation(
    plans: Iterable[FilePlan],
    apply: bool,
    root: Path,
    inspector: OpenFileInspector,
    report: dict[str, Any],
) -> None:
    for plan in plans:
        item = _plan_item(plan, root)
        if not apply:
            _record(report, "log_rotation_candidates", item)
            continue
        try:
            item.update(_rotate_log(plan, inspector))
            _record(report, "rotated_logs", item)
        except MaintenanceSkip as exc:
            item["reason"] = str(exc)
            target = "skipped_open" if str(exc).startswith("OPEN_FILE") or str(exc).startswith("OPEN_CHECK") else "skipped_safety"
            _record(report, target, item)
        except OSError as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            _record(report, "errors", item)


def _compress_jsonl(plan: FilePlan, inspector: OpenFileInspector) -> dict[str, Any]:
    _validate_plan(plan)
    _require_closed(plan.path, inspector)
    path = plan.path
    gz_path = path.with_name(f"{path.name}.gz")
    if gz_path.exists():
        gz_stat = _safe_lstat(gz_path)
        if gz_stat is None:
            raise MaintenanceSkip("UNSAFE_EXISTING_GZIP")
        _require_closed(gz_path, inspector)
        source_stats, gzip_stats = _verify_gzip_matches_source(path, gz_path)
        current_gz_stat = _safe_lstat(gz_path)
        if current_gz_stat is None or not _same_file_version(gz_stat, current_gz_stat):
            raise MaintenanceSkip("EXISTING_GZIP_CHANGED_DURING_VERIFICATION")
        _require_closed(gz_path, inspector)
        _require_closed(path, inspector)
        _validate_plan(plan)
        path.unlink()
        _fsync_directory(path.parent)
        return {
            "status": "removed_verified_plain_duplicate",
            "gz_path": str(gz_path),
            "gz_size_bytes": gz_path.stat().st_size,
            "source_sha256": source_stats.sha256,
            "verified_sha256": gzip_stats.sha256,
            "row_count": source_stats.row_count,
            "gzip_crc_verified": True,
            "reclaimed_bytes": plan.size_bytes,
        }

    tmp_path = path.with_name(f"{path.name}.gz.tmp")
    if tmp_path.exists() or tmp_path.is_symlink():
        raise MaintenanceSkip("GZIP_TEMP_ALREADY_EXISTS")
    published_stat: os.stat_result | None = None
    try:
        source_stats = _write_verified_gzip(path, tmp_path)
        _validate_plan(plan)
        gzip_stats = _content_stats_gzip(tmp_path)
        if gzip_stats != source_stats:
            raise OSError("gzip verification mismatch")
        os.utime(tmp_path, ns=(plan.mtime_ns, plan.mtime_ns))
        # Publish without replacing a concurrently-created gzip. A hard link
        # in the same directory gives us create-if-absent semantics on macOS.
        try:
            os.link(tmp_path, gz_path, follow_symlinks=False)
        except FileExistsError as exc:
            raise MaintenanceSkip("GZIP_CREATED_CONCURRENTLY") from exc
        published_stat = _safe_lstat(gz_path)
        if published_stat is None:
            raise MaintenanceSkip("PUBLISHED_GZIP_NOT_REGULAR")
        tmp_path.unlink()
        _fsync_directory(path.parent)
        _require_closed(path, inspector)
        _validate_plan(plan)
        path.unlink()
        _fsync_directory(path.parent)
        gz_size = gz_path.stat().st_size
        return {
            "status": "compressed_verified",
            "gz_path": str(gz_path),
            "gz_size_bytes": gz_size,
            "source_sha256": source_stats.sha256,
            "verified_sha256": gzip_stats.sha256,
            "row_count": source_stats.row_count,
            "gzip_crc_verified": True,
            "reclaimed_bytes": max(0, plan.size_bytes - gz_size),
        }
    except Exception:
        tmp_path.unlink(missing_ok=True)
        if published_stat is not None:
            current = _safe_lstat(gz_path)
            if current is not None and _same_file_version(published_stat, current):
                try:
                    _require_closed(gz_path, inspector)
                except MaintenanceSkip:
                    pass
                else:
                    gz_path.unlink(missing_ok=True)
                    _fsync_directory(path.parent)
        raise


def _write_verified_gzip(source: Path, destination: Path) -> ContentStats:
    digest = hashlib.sha256()
    size_bytes = 0
    newlines = 0
    last_byte = b""
    with source.open("rb") as src, destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6) as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size_bytes += len(chunk)
                newlines += chunk.count(b"\n")
                last_byte = chunk[-1:]
                dst.write(chunk)
        raw.flush()
        os.fsync(raw.fileno())
    return ContentStats(
        size_bytes=size_bytes,
        row_count=newlines + (1 if size_bytes and last_byte != b"\n" else 0),
        sha256=digest.hexdigest(),
    )


def _verify_gzip_matches_source(source: Path, gz_path: Path) -> tuple[ContentStats, ContentStats]:
    source_stats = _content_stats_plain(source)
    try:
        gzip_stats = _content_stats_gzip(gz_path)
    except (OSError, EOFError) as exc:
        raise MaintenanceSkip(f"EXISTING_GZIP_INVALID:{type(exc).__name__}") from exc
    if source_stats != gzip_stats:
        raise MaintenanceSkip("EXISTING_GZIP_CONTENT_MISMATCH")
    return source_stats, gzip_stats


def _content_stats_plain(path: Path) -> ContentStats:
    with path.open("rb") as handle:
        return _content_stats_stream(handle)


def _content_stats_gzip(path: Path) -> ContentStats:
    with gzip.open(path, "rb") as handle:
        return _content_stats_stream(handle)


def _content_stats_stream(handle: Any) -> ContentStats:
    digest = hashlib.sha256()
    size_bytes = 0
    newlines = 0
    last_byte = b""
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size_bytes += len(chunk)
        newlines += chunk.count(b"\n")
        last_byte = chunk[-1:]
    return ContentStats(
        size_bytes=size_bytes,
        row_count=newlines + (1 if size_bytes and last_byte != b"\n" else 0),
        sha256=digest.hexdigest(),
    )


def _safe_unlink(plan: FilePlan, inspector: OpenFileInspector) -> None:
    _validate_plan(plan)
    if plan.proof_path is not None:
        _require_closed(plan.proof_path, inspector)
    _require_closed(plan.path, inspector)
    _validate_plan(plan)
    plan.path.unlink()
    _fsync_directory(plan.path.parent)


def _rotate_log(plan: FilePlan, inspector: OpenFileInspector) -> dict[str, Any]:
    _validate_plan(plan)
    _require_closed(plan.path, inspector)
    generation = plan.path.with_name(f"{plan.path.name}.1.gz")
    previous_generation = plan.path.with_name(f"{plan.path.name}.2.gz")
    temp = plan.path.with_name(f"{plan.path.name}.1.gz.tmp")
    if generation.is_symlink() or previous_generation.is_symlink() or temp.is_symlink():
        raise MaintenanceSkip("UNSAFE_LOG_GENERATION_PATH")
    generation_plan = _existing_file_plan(generation, plan.allowed_root, "launchd_log_generation")
    previous_plan = _existing_file_plan(
        previous_generation,
        plan.allowed_root,
        "launchd_log_previous_generation",
    )
    for existing_plan in (generation_plan, previous_plan):
        if existing_plan is None:
            continue
        _require_closed(existing_plan.path, inspector)
        _validate_plan(existing_plan)

    content_stats: ContentStats | None = None
    if temp.exists():
        temp_plan = _existing_file_plan(temp, plan.allowed_root, "launchd_log_temp")
        if temp_plan is None:
            raise MaintenanceSkip("UNSAFE_LOG_GENERATION_PATH")
        _require_closed(temp, inspector)
        try:
            source_stats, temp_stats = _verify_gzip_matches_source(plan.path, temp)
            _validate_plan(temp_plan)
            if source_stats != temp_stats:
                raise MaintenanceSkip("LOG_TEMP_CONTENT_MISMATCH")
            content_stats = source_stats
        except MaintenanceSkip:
            # A killed prior rotation cannot have truncated the source before
            # publishing its verified generation.  A mismatched closed temp is
            # therefore a redundant crash artifact, never the sole log copy.
            _safe_unlink(temp_plan, inspector)

    published = False
    try:
        if content_stats is None:
            content_stats = _write_verified_gzip(plan.path, temp)
        _validate_plan(plan)
        verified = _content_stats_gzip(temp)
        if verified != content_stats:
            raise OSError("log gzip verification mismatch")

        for existing_plan in (generation_plan, previous_plan):
            if existing_plan is None:
                continue
            _require_closed(existing_plan.path, inspector)
            _validate_plan(existing_plan)
        if generation_plan is not None:
            os.replace(generation, previous_generation)
            _fsync_directory(plan.path.parent)
        temp.replace(generation)
        published = True
        _fsync_directory(plan.path.parent)
        # Preserve the original inode/path expected by launchd; truncate only
        # after a stable-stat and closed-file check, never unlink the live log.
        _require_closed(plan.path, inspector)
        _validate_plan(plan)
        with plan.path.open("r+b") as handle:
            current = os.fstat(handle.fileno())
            if not _stat_matches(plan, current):
                raise MaintenanceSkip("FILE_CHANGED_DURING_LOG_ROTATION")
            handle.truncate(0)
            handle.flush()
            os.fsync(handle.fileno())
        gz_size = generation.stat().st_size
        return {
            "status": "rotated_verified_gzip",
            "generation_path": str(generation),
            "previous_generation_path": str(previous_generation)
            if previous_generation.exists()
            else None,
            "generation_size_bytes": gz_size,
            "source_sha256": content_stats.sha256,
            "verified_sha256": verified.sha256,
            "row_count": content_stats.row_count,
            "gzip_crc_verified": True,
            "reclaimed_bytes": max(0, plan.size_bytes - gz_size),
        }
    except Exception:
        if not published:
            temp.unlink(missing_ok=True)
        raise


def _existing_file_plan(path: Path, allowed_root: Path, category: str) -> FilePlan | None:
    if not path.exists():
        return None
    file_stat = _safe_lstat(path)
    if file_stat is None:
        raise MaintenanceSkip("UNSAFE_LOG_GENERATION_PATH")
    return _plan(path, allowed_root, file_stat, 0.0, category)


def _require_closed(path: Path, inspector: OpenFileInspector) -> None:
    state = inspector.state(path)
    if state == "OPEN":
        raise MaintenanceSkip("OPEN_FILE_SKIPPED")
    if state != "CLOSED":
        raise MaintenanceSkip("OPEN_CHECK_UNAVAILABLE_FAIL_CLOSED")


def _validate_plan(plan: FilePlan) -> os.stat_result:
    _ensure_within(plan.path, plan.allowed_root)
    current = _safe_lstat(plan.path)
    if current is None:
        raise MaintenanceSkip("NOT_REGULAR_OR_SYMLINK")
    if not _stat_matches(plan, current):
        raise MaintenanceSkip("FILE_CHANGED_SINCE_SCAN")
    if plan.proof_path is not None:
        proof = _safe_lstat(plan.proof_path)
        if proof is None or not _proof_stat_matches(plan, proof):
            raise MaintenanceSkip("CANONICAL_TARGET_CHANGED_SINCE_SCAN")
    return current


def _stat_matches(plan: FilePlan, current: os.stat_result) -> bool:
    return (
        current.st_dev == plan.device
        and current.st_ino == plan.inode
        and current.st_size == plan.size_bytes
        and current.st_mtime_ns == plan.mtime_ns
    )


def _proof_stat_matches(plan: FilePlan, current: os.stat_result) -> bool:
    return (
        current.st_dev == plan.proof_device
        and current.st_ino == plan.proof_inode
        and current.st_size == plan.proof_size_bytes
        and current.st_mtime_ns == plan.proof_mtime_ns
    )


def _same_file_version(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )


def _plan(
    path: Path,
    allowed_root: Path,
    file_stat: os.stat_result,
    age_seconds: float,
    category: str,
    *,
    proof_path: Path | None = None,
    proof_stat: os.stat_result | None = None,
) -> FilePlan:
    return FilePlan(
        path=path,
        allowed_root=allowed_root.resolve(),
        size_bytes=file_stat.st_size,
        age_seconds=age_seconds,
        device=file_stat.st_dev,
        inode=file_stat.st_ino,
        mtime_ns=file_stat.st_mtime_ns,
        category=category,
        proof_path=proof_path,
        proof_device=proof_stat.st_dev if proof_stat is not None else None,
        proof_inode=proof_stat.st_ino if proof_stat is not None else None,
        proof_size_bytes=proof_stat.st_size if proof_stat is not None else None,
        proof_mtime_ns=proof_stat.st_mtime_ns if proof_stat is not None else None,
    )


def _plan_item(plan: FilePlan, root: Path) -> dict[str, Any]:
    return {
        "path": _rel(plan.path, root),
        "category": plan.category,
        "size_bytes": plan.size_bytes,
        "age_seconds": round(plan.age_seconds, 3),
    }


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    reclaimed = 0
    candidate_bytes = 0
    byte_totals = report.get("byte_totals") or {}
    for key in (
        "compressed",
        "removed_stale_temp",
        "removed_project_atomic",
        "removed_external_diagnostics",
        "removed_guardian_ephemeral",
        "rotated_logs",
    ):
        reclaimed += int(byte_totals.get(key) or 0)
    for key in (
        "compression_candidates",
        "stale_temp_candidates",
        "project_atomic_candidates",
        "external_diagnostic_candidates",
        "guardian_ephemeral_candidates",
        "log_rotation_candidates",
    ):
        candidate_bytes += int(byte_totals.get(key) or 0)

    pressure = report.get("pressure_after") or "UNKNOWN"
    if _count(report, "errors"):
        status = "ERROR"
    elif pressure == "P0" and reclaimed == 0 and report.get("apply"):
        status = "P0_NO_SAFE_RECLAIM"
    elif pressure == "P0":
        status = "P0_LOW_SPACE"
    elif pressure == "BELOW_OPERATING_FLOOR":
        status = "BELOW_OPERATING_FLOOR"
    else:
        status = "OK"

    disk_after = report.get("disk_after") or {}
    return {
        "status": status,
        "pressure_state": pressure,
        "emergency_mode": bool(report.get("emergency_mode")),
        "compressed_files": _count(report, "compressed"),
        "compression_candidates": _count(report, "compression_candidates"),
        "removed_stale_temp_files": _count(report, "removed_stale_temp"),
        "stale_temp_candidates": _count(report, "stale_temp_candidates"),
        "removed_project_atomic_files": _count(report, "removed_project_atomic"),
        "removed_external_diagnostics": _count(report, "removed_external_diagnostics"),
        "removed_guardian_ephemeral_files": _count(report, "removed_guardian_ephemeral"),
        "rotated_logs": _count(report, "rotated_logs"),
        "skipped_open": _count(report, "skipped_open"),
        "skipped_safety": _count(report, "skipped_safety"),
        "errors": _count(report, "errors"),
        "reclaimed_bytes_estimate": reclaimed,
        "candidate_bytes": candidate_bytes,
        "free_bytes_after": disk_after.get("free_bytes"),
        "floor_met_after": bool(report.get("floor_met_after")),
        "bytes_to_operating_floor": report.get("bytes_to_operating_floor"),
    }


def _record(report: dict[str, Any], key: str, item: dict[str, Any]) -> None:
    counts = report.setdefault("item_counts", {})
    counts[key] = int(counts.get(key) or 0) + 1
    byte_totals = report.setdefault("byte_totals", {})
    measured_bytes = (
        item.get("reclaimed_bytes")
        if "reclaimed_bytes" in item
        else item.get("size_bytes")
    )
    byte_totals[key] = int(byte_totals.get(key) or 0) + int(measured_bytes or 0)
    values = report.setdefault(key, [])
    if len(values) < MAX_REPORT_ITEMS:
        values.append(item)
    else:
        truncated = report.setdefault("truncated_items", {})
        truncated[key] = int(truncated.get(key) or 0) + 1


def _count(report: dict[str, Any], key: str) -> int:
    return int((report.get("item_counts") or {}).get(key) or len(report.get(key) or []))


def _disk_usage(root: Path) -> dict[str, int]:
    usage = shutil.disk_usage(root)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _pressure_state(free_bytes: int) -> str:
    if free_bytes < P0_FREE_BYTES:
        return "P0"
    if free_bytes < OPERATING_FLOOR_BYTES:
        return "BELOW_OPERATING_FLOOR"
    return "OK"


def _resolve_under_root(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    candidate = path if path.is_absolute() else resolved_root / path
    resolved = candidate.resolve(strict=False)
    _ensure_within(resolved, resolved_root)
    return resolved


def _ensure_within(path: Path, allowed_root: Path) -> None:
    resolved_path = path.resolve(strict=False)
    resolved_root = allowed_root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes allowlisted root: {path} not below {resolved_root}") from exc


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _safe_lstat(path: Path) -> os.stat_result | None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return None
    if stat_module.S_ISLNK(file_stat.st_mode) or not stat_module.S_ISREG(file_stat.st_mode):
        return None
    return file_stat


def _walk_regular_files(base: Path, *, prune_names: set[str] | None = None) -> Iterable[Path]:
    prune_names = prune_names or set()
    for current, dirs, files in os.walk(base, followlinks=False):
        current_path = Path(current)
        safe_dirs: list[str] = []
        for name in dirs:
            child = current_path / name
            if name in prune_names or child.is_symlink():
                continue
            safe_dirs.append(name)
        dirs[:] = safe_dirs
        for name in files:
            path = current_path / name
            if _safe_lstat(path) is not None:
                yield path


def _protected_atomic_target(path: Path) -> bool:
    lower_name = path.name.lower()
    lower_parts = [part.lower() for part in path.parts]
    if re.search(r"\.(?:db|sqlite|sqlite3)(?:[-.].*)?$", lower_name):
        return True
    if any("forecast" in part or "projection" in part for part in lower_parts):
        return True
    evidence_tokens = (
        "execution_ledger",
        "verification_ledger",
        "trader_journal",
        "audit_history",
        "s_hunt_ledger",
        "live_trade_log",
    )
    if any(token in part for token in evidence_tokens for part in lower_parts):
        return True
    if "logs" in lower_parts and lower_name.endswith(".jsonl"):
        return True
    return False


def _write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp")
    if temp.is_symlink() or path.is_symlink():
        raise OSError("unsafe report symlink")
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    with temp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _age_seconds(mtime: float, now: datetime) -> float:
    return max(0.0, now.timestamp() - mtime)


def _is_temp_path(path: Path) -> bool:
    return path.name.endswith((".tmp", ".partial"))


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


if __name__ == "__main__":
    sys.exit(main())
