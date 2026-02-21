#!/usr/bin/env python3
"""Scheduled worker wrapper for replay quality gate runs.

This worker executes `scripts/replay_quality_gate.py`, stores the latest status
snapshot, appends run history, and trims old run artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _resolve_path(text: str, *, base_dir: Path) -> Path:
    path = Path(str(text).strip())
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _tail(text: str | None, *, limit: int = 12000) -> str:
    raw = str(text or "")
    return raw[-limit:]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        tmp_path = Path(fh.name)
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


def _iter_report_paths(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    rows: list[Path] = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        report = child / "quality_gate_report.json"
        if report.exists():
            rows.append(report)
    rows.sort(key=lambda p: (p.stat().st_mtime, p.as_posix()), reverse=True)
    return rows


def _extract_report_path_from_stdout(stdout: str, *, out_dir: Path) -> Path | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate or not candidate.endswith("quality_gate_report.json"):
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.exists():
            return path
    reports = _iter_report_paths(out_dir)
    if reports:
        return reports[0]
    return None


def _cleanup_old_runs(out_dir: Path, *, keep_runs: int, protected: Path | None = None) -> list[str]:
    if keep_runs <= 0 or not out_dir.exists():
        return []
    reports = _iter_report_paths(out_dir)
    protected_dir = protected.parent.resolve() if protected is not None else None

    keep_dirs: set[Path] = set()
    for report in reports[:keep_runs]:
        keep_dirs.add(report.parent.resolve())
    if protected_dir is not None:
        keep_dirs.add(protected_dir)

    removed: list[str] = []
    for report in reports[keep_runs:]:
        run_dir = report.parent.resolve()
        if run_dir in keep_dirs:
            continue
        shutil.rmtree(run_dir, ignore_errors=True)
        removed.append(run_dir.name)
    return removed


@dataclass(frozen=True)
class WorkerConfig:
    config_path: Path
    out_dir: Path
    state_path: Path
    history_path: Path
    timeout_sec: int
    keep_runs: int
    strict: bool
    ticks_glob: str
    workers: str
    backend: str


def _default_config() -> WorkerConfig:
    return WorkerConfig(
        config_path=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_CONFIG", "config/replay_quality_gate.yaml"),
            base_dir=REPO_ROOT,
        ),
        out_dir=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_OUT_DIR", "tmp/replay_quality_gate"),
            base_dir=REPO_ROOT,
        ),
        state_path=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_STATE_PATH", "logs/replay_quality_gate_latest.json"),
            base_dir=REPO_ROOT,
        ),
        history_path=_resolve_path(
            os.getenv("REPLAY_QUALITY_GATE_HISTORY_PATH", "logs/replay_quality_gate_history.jsonl"),
            base_dir=REPO_ROOT,
        ),
        timeout_sec=max(30, _env_int("REPLAY_QUALITY_GATE_TIMEOUT_SEC", 1800)),
        keep_runs=max(1, _env_int("REPLAY_QUALITY_GATE_KEEP_RUNS", 24)),
        strict=_env_bool("REPLAY_QUALITY_GATE_STRICT", False),
        ticks_glob=str(os.getenv("REPLAY_QUALITY_GATE_TICKS_GLOB", "")).strip(),
        workers=str(os.getenv("REPLAY_QUALITY_GATE_WORKERS", "")).strip(),
        backend=str(os.getenv("REPLAY_QUALITY_GATE_BACKEND", "")).strip(),
    )


def parse_args() -> argparse.Namespace:
    default = _default_config()
    ap = argparse.ArgumentParser(description="Replay quality gate scheduled worker")
    ap.add_argument("--config", type=Path, default=default.config_path)
    ap.add_argument("--out-dir", type=Path, default=default.out_dir)
    ap.add_argument("--state-path", type=Path, default=default.state_path)
    ap.add_argument("--history-path", type=Path, default=default.history_path)
    ap.add_argument("--timeout-sec", type=int, default=default.timeout_sec)
    ap.add_argument("--keep-runs", type=int, default=default.keep_runs)
    ap.add_argument("--strict", action="store_true", default=default.strict)
    ap.add_argument("--ticks-glob", default=default.ticks_glob)
    ap.add_argument("--workers", default=default.workers)
    ap.add_argument(
        "--backend",
        choices=("exit_workers_groups", "exit_workers_main", ""),
        default=default.backend,
    )
    return ap.parse_args()


def _build_command(cfg: WorkerConfig) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "replay_quality_gate.py"),
        "--config",
        str(cfg.config_path),
        "--out-dir",
        str(cfg.out_dir),
        "--timeout-sec",
        str(max(30, int(cfg.timeout_sec))),
    ]
    if cfg.strict:
        cmd.append("--strict")
    if cfg.ticks_glob:
        cmd.extend(["--ticks-glob", cfg.ticks_glob])
    if cfg.workers:
        cmd.extend(["--workers", cfg.workers])
    if cfg.backend:
        cmd.extend(["--backend", cfg.backend])
    return cmd


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=max(30, int(timeout_sec)),
        check=False,
    )


def _summarize_report(report_payload: dict[str, Any] | None) -> tuple[str, list[str], int]:
    if not isinstance(report_payload, dict):
        return "unknown", [], 0
    overall = report_payload.get("overall") if isinstance(report_payload.get("overall"), dict) else {}
    meta = report_payload.get("meta") if isinstance(report_payload.get("meta"), dict) else {}
    status = str(overall.get("status") or "unknown")
    failing_workers = overall.get("failing_workers")
    if not isinstance(failing_workers, list):
        failing_workers = []
    failing = [str(v) for v in failing_workers if str(v).strip()]
    folds = 0
    try:
        folds = int(meta.get("fold_count") or 0)
    except Exception:
        folds = 0
    return status, failing, folds


def _build_config_from_args(args: argparse.Namespace) -> WorkerConfig:
    return WorkerConfig(
        config_path=Path(args.config).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        state_path=Path(args.state_path).resolve(),
        history_path=Path(args.history_path).resolve(),
        timeout_sec=max(30, int(args.timeout_sec)),
        keep_runs=max(1, int(args.keep_runs)),
        strict=bool(args.strict),
        ticks_glob=str(args.ticks_glob or "").strip(),
        workers=str(args.workers or "").strip(),
        backend=str(args.backend or "").strip(),
    )


def run_once(
    cfg: WorkerConfig,
    *,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
) -> int:
    started = datetime.now(timezone.utc)
    cmd = _build_command(cfg)
    executor = runner or (lambda c: _run_subprocess(c, cwd=REPO_ROOT, timeout_sec=cfg.timeout_sec))
    proc = executor(cmd)
    finished = datetime.now(timezone.utc)

    report_path = _extract_report_path_from_stdout(proc.stdout or "", out_dir=cfg.out_dir)
    report_payload = _read_json(report_path) if report_path is not None else None
    gate_status, failing_workers, fold_count = _summarize_report(report_payload)
    md_path = report_path.with_name("quality_gate_report.md") if report_path else None
    removed_dirs = _cleanup_old_runs(cfg.out_dir, keep_runs=cfg.keep_runs, protected=report_path)

    payload = {
        "worker": "replay_quality_gate_worker",
        "generated_at": finished.isoformat(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": (finished - started).total_seconds(),
        "returncode": int(proc.returncode),
        "strict": bool(cfg.strict),
        "gate_status": gate_status,
        "failing_workers": failing_workers,
        "fold_count": int(fold_count),
        "config_path": str(cfg.config_path),
        "out_dir": str(cfg.out_dir),
        "report_json_path": str(report_path) if report_path else "",
        "report_md_path": str(md_path) if md_path and md_path.exists() else "",
        "cleaned_run_dirs": removed_dirs,
        "command": cmd,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }
    _write_json_atomic(cfg.state_path, payload)
    _append_jsonl(cfg.history_path, payload)

    print(
        f"[replay-quality-gate-worker] rc={proc.returncode} gate_status={gate_status} "
        f"report={payload['report_json_path'] or '-'}"
    )
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    cfg = _build_config_from_args(args)
    return run_once(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
