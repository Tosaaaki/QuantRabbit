#!/usr/bin/env python3
"""Run local analysis/feedback workers on bounded intervals.

This bridges the gap between the local V2 watchdog/autorecover lane and the
analysis timers that existed only in systemd-based deployments. The cycle is
safe to invoke frequently: it keeps per-job intervals/state and skips overlap
with a lock.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_BIN = (
    str((REPO_ROOT / ".venv" / "bin" / "python").resolve())
    if (REPO_ROOT / ".venv" / "bin" / "python").exists()
    else sys.executable
)
LOG_DIR = Path(os.getenv("LOCAL_FEEDBACK_CYCLE_LOG_DIR", REPO_ROOT / "logs"))
STATE_FILE = Path(
    os.getenv("LOCAL_FEEDBACK_CYCLE_STATE_FILE", LOG_DIR / "local_feedback_cycle.state.json")
)
LATEST_OUTPUT = Path(
    os.getenv("LOCAL_FEEDBACK_CYCLE_OUTPUT", LOG_DIR / "local_feedback_cycle_latest.json")
)
HISTORY_OUTPUT = Path(
    os.getenv("LOCAL_FEEDBACK_CYCLE_HISTORY", LOG_DIR / "local_feedback_cycle_history.jsonl")
)
RUN_LOG_DIR = Path(
    os.getenv("LOCAL_FEEDBACK_CYCLE_RUN_LOG_DIR", LOG_DIR / "local_feedback_cycle")
)
LOCK_DIR = Path(
    os.getenv("LOCAL_FEEDBACK_CYCLE_LOCK_DIR", LOG_DIR / "local_feedback_cycle.lock")
)


@dataclass(frozen=True)
class JobConfig:
    name: str
    enabled: bool
    interval_sec: int
    command: tuple[str, ...]
    env_files: tuple[Path, ...]
    output_paths: tuple[Path, ...]
    timeout_sec: int
    retry_count: int
    retry_delay_sec: float


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except Exception:
        return default
    return max(minimum, value)


def _resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _parse_csv_paths(raw: str | None) -> tuple[Path, ...]:
    if not raw:
        return ()
    out: list[Path] = []
    for token in str(raw).split(","):
        cleaned = token.strip()
        if cleaned:
            out.append(_resolve_path(cleaned))
    return tuple(out)


def _parse_command(raw: str) -> tuple[str, ...]:
    parts = tuple(shlex.split(raw))
    if not parts:
        raise ValueError("empty command")
    return parts


def _default_job_command(job_name: str, python_bin: str) -> tuple[str, ...]:
    if job_name == "strategy_feedback":
        return (python_bin, "-m", "analysis.strategy_feedback_worker")
    if job_name == "replay_quality_gate":
        return (python_bin, "-m", "analysis.replay_quality_gate_worker")
    if job_name == "dynamic_alloc":
        return (
            python_bin,
            "scripts/dynamic_alloc_worker.py",
            "--limit",
            "2400",
            "--lookback-days",
            "7",
            "--min-trades",
            "16",
            "--pf-cap",
            "2.0",
            "--target-use",
            "0.88",
        )
    if job_name == "pattern_book":
        return (
            python_bin,
            "scripts/pattern_book_worker.py",
            "--lookback-days",
            "180",
            "--batch-size",
            "4000",
            "--max-backfill-rows",
            "500000",
            "--min-samples-soft",
            "30",
            "--min-samples-block",
            "120",
        )
    if job_name == "trade_counterfactual":
        return (python_bin, "-m", "analysis.trade_counterfactual_worker")
    raise KeyError(f"unsupported job: {job_name}")


def _default_job_env_files(job_name: str) -> tuple[Path, ...]:
    if job_name == "strategy_feedback":
        return (
            _resolve_path("ops/env/quant-v2-runtime.env"),
            _resolve_path("ops/env/quant-strategy-feedback.env"),
        )
    if job_name == "replay_quality_gate":
        return (
            _resolve_path("ops/env/quant-v2-runtime.env"),
            _resolve_path("ops/env/quant-replay-quality-gate.env"),
        )
    if job_name == "trade_counterfactual":
        return (
            _resolve_path("ops/env/quant-v2-runtime.env"),
            _resolve_path("ops/env/quant-trade-counterfactual.env"),
        )
    return ()


def _default_job_outputs(job_name: str) -> tuple[Path, ...]:
    if job_name == "strategy_feedback":
        return (_resolve_path("logs/strategy_feedback.json"),)
    if job_name == "replay_quality_gate":
        return (
            _resolve_path("logs/replay_quality_gate_latest.json"),
            _resolve_path("logs/replay_quality_gate_history.jsonl"),
            _resolve_path("config/worker_reentry.yaml"),
            _resolve_path("logs/replay_auto_improve_state.json"),
        )
    if job_name == "dynamic_alloc":
        return (_resolve_path("config/dynamic_alloc.json"),)
    if job_name == "pattern_book":
        return (
            _resolve_path("logs/patterns.db"),
            _resolve_path("config/pattern_book.json"),
            _resolve_path("config/pattern_book_deep.json"),
        )
    if job_name == "trade_counterfactual":
        return (
            _resolve_path("logs/trade_counterfactual_latest.json"),
            _resolve_path("logs/trade_counterfactual_history.jsonl"),
        )
    raise KeyError(f"unsupported job: {job_name}")


def _build_job(job_name: str, python_bin: str) -> JobConfig:
    env_prefix = f"LOCAL_FEEDBACK_CYCLE_{job_name.upper()}"
    enabled_defaults = {
        "strategy_feedback": False,
        "replay_quality_gate": True,
        "dynamic_alloc": True,
        "pattern_book": True,
        "trade_counterfactual": True,
    }
    enabled = _env_bool(f"{env_prefix}_ENABLED", enabled_defaults[job_name])
    interval_defaults = {
        "strategy_feedback": 600,
        "replay_quality_gate": 10800,
        "dynamic_alloc": 120,
        "pattern_book": 300,
        "trade_counterfactual": 1200,
    }
    timeout_defaults = {
        "strategy_feedback": 180,
        "replay_quality_gate": 1800,
        "dynamic_alloc": 180,
        "pattern_book": 420,
        "trade_counterfactual": 180,
    }
    retry_defaults = {
        "strategy_feedback": 0,
        "replay_quality_gate": 0,
        "dynamic_alloc": 0,
        "pattern_book": 0,
        "trade_counterfactual": 1,
    }
    interval_sec = _env_int(
        f"{env_prefix}_INTERVAL_SEC",
        interval_defaults[job_name],
        minimum=10,
    )
    timeout_sec = _env_int(
        f"{env_prefix}_TIMEOUT_SEC",
        timeout_defaults[job_name],
        minimum=5,
    )
    retry_count = _env_int(
        f"{env_prefix}_RETRY_COUNT",
        retry_defaults[job_name],
        minimum=0,
    )
    retry_delay_sec = float(os.getenv(f"{env_prefix}_RETRY_DELAY_SEC", "2.0") or 2.0)
    command_raw = os.getenv(f"{env_prefix}_CMD")
    command = (
        _parse_command(command_raw)
        if command_raw
        else _default_job_command(job_name, python_bin)
    )
    env_files_raw = os.getenv(f"{env_prefix}_ENV_FILES")
    output_paths_raw = os.getenv(f"{env_prefix}_OUTPUTS")
    env_files = (
        _parse_csv_paths(env_files_raw)
        if env_files_raw is not None
        else _default_job_env_files(job_name)
    )
    output_paths = (
        _parse_csv_paths(output_paths_raw)
        if output_paths_raw is not None
        else _default_job_outputs(job_name)
    )
    return JobConfig(
        name=job_name,
        enabled=enabled,
        interval_sec=interval_sec,
        command=command,
        env_files=env_files,
        output_paths=output_paths,
        timeout_sec=timeout_sec,
        retry_count=retry_count,
        retry_delay_sec=max(0.0, retry_delay_sec),
    )


def _job_order(python_bin: str) -> list[JobConfig]:
    return [
        _build_job("dynamic_alloc", python_bin),
        _build_job("pattern_book", python_bin),
        _build_job("strategy_feedback", python_bin),
        _build_job("trade_counterfactual", python_bin),
        _build_job("replay_quality_gate", python_bin),
    ]


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return env
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if value == "/home/tossaki/QuantRabbit":
            value = str(REPO_ROOT.resolve())
        elif value.startswith("/home/tossaki/QuantRabbit/"):
            value = str(REPO_ROOT.resolve() / value.removeprefix("/home/tossaki/QuantRabbit/"))
        if key:
            env[key] = value
    return env


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"jobs": {}}
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"jobs": {}}
    if not isinstance(payload, dict):
        return {"jobs": {}}
    jobs = payload.get("jobs")
    if not isinstance(jobs, dict):
        payload["jobs"] = {}
    return payload


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


def _path_mtime(path: Path) -> float | None:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return None


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def _job_due(job: JobConfig, state: dict[str, Any], now_epoch: float, force: bool) -> tuple[bool, float]:
    if force:
        return True, 0.0
    jobs_state = state.get("jobs")
    if not isinstance(jobs_state, dict):
        return True, 0.0
    job_state = jobs_state.get(job.name)
    if not isinstance(job_state, dict):
        return True, 0.0
    last_run_epoch = job_state.get("last_run_epoch")
    try:
        last_run = float(last_run_epoch)
    except Exception:
        return True, 0.0
    elapsed = max(0.0, now_epoch - last_run)
    remaining = max(0.0, float(job.interval_sec) - elapsed)
    return elapsed >= float(job.interval_sec), remaining


def _summarize_output_changes(output_paths: tuple[Path, ...], before: dict[Path, float | None]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for path in output_paths:
        after = _path_mtime(path)
        updated = after is not None and before.get(path) != after
        changes.append(
            {
                "path": _relative_path(path),
                "exists": path.exists(),
                "updated": updated,
                "mtime": after,
            }
        )
    return changes


def _run_job(job: JobConfig, *, dry_run: bool) -> dict[str, Any]:
    now_epoch = time.time()
    before = {path: _path_mtime(path) for path in job.output_paths}
    result: dict[str, Any] = {
        "job": job.name,
        "status": "ok",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_epoch)),
        "command": list(job.command),
        "interval_sec": job.interval_sec,
        "timeout_sec": job.timeout_sec,
        "retry_count": job.retry_count,
        "env_files": [_relative_path(path) for path in job.env_files],
    }
    if dry_run:
        result["status"] = "dry_run"
        result["duration_sec"] = 0.0
        result["outputs"] = _summarize_output_changes(job.output_paths, before)
        return result

    env = os.environ.copy()
    for env_file in job.env_files:
        env.update(_load_env_file(env_file))

    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_LOG_DIR / f"{job.name}.log"
    started = time.perf_counter()
    attempt_logs: list[str] = []
    last_stdout = ""
    last_stderr = ""
    attempts_used = 0
    for attempt in range(1, job.retry_count + 2):
        attempts_used = attempt
        try:
            proc = subprocess.run(
                job.command,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=job.timeout_sec,
                check=False,
            )
            last_stdout = proc.stdout or ""
            last_stderr = proc.stderr or ""
            attempt_logs.append(
                f"[attempt] {attempt}\n"
                f"[command] {' '.join(shlex.quote(part) for part in job.command)}\n"
                f"[exit_code] {proc.returncode}\n"
                f"[stdout]\n{last_stdout}\n"
                f"[stderr]\n{last_stderr}\n"
            )
            if proc.returncode == 0:
                result["exit_code"] = 0
                result["status"] = "ok"
                break
            result["exit_code"] = proc.returncode
            result["status"] = "error"
        except subprocess.TimeoutExpired as exc:
            last_stdout = exc.stdout or ""
            last_stderr = exc.stderr or ""
            attempt_logs.append(
                f"[attempt] {attempt}\n"
                f"[command] {' '.join(shlex.quote(part) for part in job.command)}\n"
                f"[timeout_sec] {job.timeout_sec}\n"
                f"[stdout]\n{last_stdout}\n"
                f"[stderr]\n{last_stderr}\n"
            )
            result["status"] = "timeout"
        if attempt < job.retry_count + 1:
            time.sleep(job.retry_delay_sec)
    duration_sec = round(time.perf_counter() - started, 3)
    log_path.write_text("".join(attempt_logs), encoding="utf-8")
    result["attempts"] = attempts_used
    result["duration_sec"] = duration_sec
    result["log_path"] = _relative_path(log_path)
    result["stdout_tail"] = last_stdout.strip().splitlines()[-5:]
    result["stderr_tail"] = last_stderr.strip().splitlines()[-5:]
    result["outputs"] = _summarize_output_changes(job.output_paths, before)
    return result


def _acquire_lock() -> tuple[bool, str | None]:
    LOCK_DIR.parent.mkdir(parents=True, exist_ok=True)
    pid_file = LOCK_DIR / "pid"
    try:
        LOCK_DIR.mkdir()
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        return True, None
    except FileExistsError:
        owner_pid: str | None = None
        try:
            owner_pid = pid_file.read_text(encoding="utf-8").strip() or None
        except Exception:
            owner_pid = None
        if owner_pid and owner_pid.isdigit():
            try:
                os.kill(int(owner_pid), 0)
                return False, owner_pid
            except OSError:
                pass
        try:
            for child in LOCK_DIR.iterdir():
                child.unlink(missing_ok=True)
            LOCK_DIR.rmdir()
        except Exception:
            return False, owner_pid
        LOCK_DIR.mkdir()
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        return True, owner_pid


def _release_lock() -> None:
    pid_file = LOCK_DIR / "pid"
    try:
        pid_file.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        LOCK_DIR.rmdir()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local analysis/feedback jobs with interval guards.")
    parser.add_argument("--force", action="store_true", help="ignore interval guards")
    parser.add_argument("--dry-run", action="store_true", help="report due jobs without executing them")
    parser.add_argument(
        "--job",
        action="append",
        default=[],
        help="restrict execution to one or more job names",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_bin = os.getenv("LOCAL_FEEDBACK_CYCLE_PYTHON_BIN", DEFAULT_PYTHON_BIN)
    jobs = _job_order(python_bin)
    if args.job:
        wanted = {item.strip() for item in args.job if item.strip()}
        jobs = [job for job in jobs if job.name in wanted]

    acquired, owner_pid = _acquire_lock()
    if not acquired:
        payload = {
            "status": "skipped",
            "reason": "lock_held",
            "owner_pid": owner_pid,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "jobs": [],
        }
        _write_json_atomic(LATEST_OUTPUT, payload)
        _append_jsonl(HISTORY_OUTPUT, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    signal.signal(signal.SIGTERM, lambda *_args: (_release_lock(), sys.exit(1)))
    try:
        state = _load_state()
        jobs_state = state.get("jobs")
        if not isinstance(jobs_state, dict):
            jobs_state = {}
            state["jobs"] = jobs_state

        now_epoch = time.time()
        job_results: list[dict[str, Any]] = []
        any_executed = False
        any_error = False

        for job in jobs:
            if not job.enabled:
                job_results.append({"job": job.name, "status": "disabled"})
                continue
            due, remaining_sec = _job_due(job, state, now_epoch, args.force)
            if not due:
                job_results.append(
                    {
                        "job": job.name,
                        "status": "skipped",
                        "reason": "interval_not_elapsed",
                        "remaining_sec": round(remaining_sec, 3),
                        "interval_sec": job.interval_sec,
                    }
                )
                continue
            any_executed = True
            job_result = _run_job(job, dry_run=args.dry_run)
            job_results.append(job_result)
            jobs_state[job.name] = {
                "last_run_epoch": time.time(),
                "last_status": job_result.get("status"),
            }
            if job_result.get("status") in {"error", "timeout"}:
                any_error = True

        payload = {
            "status": "error" if any_error else ("ok" if any_executed else "skipped"),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dry_run": bool(args.dry_run),
            "force": bool(args.force),
            "jobs": job_results,
        }
        _write_json_atomic(STATE_FILE, state)
        _write_json_atomic(LATEST_OUTPUT, payload)
        _append_jsonl(HISTORY_OUTPUT, payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    finally:
        _release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
