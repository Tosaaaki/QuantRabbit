from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CYCLE_SCRIPT = REPO_ROOT / "scripts" / "run_brain_autopdca_cycle.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _prepare_fake_tools(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    benchmark_script = tmp_path / "fake_benchmark.py"
    apply_script = tmp_path / "fake_apply.py"
    stack_script = tmp_path / "fake_stack.sh"
    env_profile = tmp_path / "brain-ollama.env"
    stack_calls = tmp_path / "stack_calls.log"

    env_profile.write_text("BRAIN_OLLAMA_MODEL=old-model\n", encoding="utf-8")

    _write_executable(
        benchmark_script,
        """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args, _ = parser.parse_known_args()
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
  'status': 'ok',
  'selected_source': 'orders_preflight',
  'sample_summary': {'total_samples': 4},
  'ranking': [
    {'rank': 1, 'name': 'fast', 'model': 'qwen2.5:7b', 'score': 1.1}
  ]
}, ensure_ascii=True, sort_keys=True), encoding='utf-8')
print(json.dumps({'status': 'ok', 'output': str(out)}, ensure_ascii=True))
""",
    )

    _write_executable(
        apply_script,
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', required=True)
parser.add_argument('--env-profile', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--dry-run', action='store_true')
args, _ = parser.parse_known_args()

change = os.environ.get('TEST_APPLY_CHANGE', '0') == '1'
env_path = Path(args.env_profile)
if (not args.dry_run) and change:
    existing = env_path.read_text(encoding='utf-8') if env_path.exists() else ''
    env_path.write_text(existing + 'BRAIN_TIMEOUT_SEC=8\\n', encoding='utf-8')

result = {
  'preflight_model': 'qwen2.5:7b',
  'autotune_model': 'gpt-oss:20b',
  'preflight_timeout_sec': 8,
  'env_changed': bool((not args.dry_run) and change),
}
if not args.dry_run:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True), encoding='utf-8')
print(json.dumps(result, ensure_ascii=True, sort_keys=True))
""",
    )

    _write_executable(
        stack_script,
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${STACK_CALLS_FILE:?missing STACK_CALLS_FILE}"
printf '[fake-stack] %s\n' "$*"
""",
    )

    return (
        logs_dir,
        benchmark_script,
        apply_script,
        stack_script,
        env_profile,
        stack_calls,
    )


def _run_cycle(
    tmp_path: Path,
    *,
    apply_change: bool,
    force: bool = True,
    interval_sec: int | None = None,
) -> tuple[dict, str]:
    (
        logs_dir,
        benchmark_script,
        apply_script,
        stack_script,
        env_profile,
        stack_calls,
    ) = _prepare_fake_tools(tmp_path)

    cycle_latest = logs_dir / "cycle_latest.json"
    summary_latest = logs_dir / "cycle_latest.log"

    env = os.environ.copy()
    env.update(
        {
            "PY_BIN": sys.executable,
            "BRAIN_AUTOPDCA_LOG_DIR": str(logs_dir),
            "BRAIN_AUTOPDCA_BENCHMARK_SCRIPT": str(benchmark_script),
            "BRAIN_AUTOPDCA_APPLY_SCRIPT": str(apply_script),
            "BRAIN_AUTOPDCA_ENV_PROFILE": str(env_profile),
            "BRAIN_AUTOPDCA_STACK_SCRIPT": str(stack_script),
            "BRAIN_AUTOPDCA_STACK_ENV": str(tmp_path / "stack.env"),
            "BRAIN_AUTOPDCA_CYCLE_OUTPUT": str(cycle_latest),
            "BRAIN_AUTOPDCA_SUMMARY_LOG": str(summary_latest),
            "STACK_CALLS_FILE": str(stack_calls),
            "TEST_APPLY_CHANGE": "1" if apply_change else "0",
        }
    )

    cmd = [str(CYCLE_SCRIPT)]
    if force:
        cmd.append("--force")
    if interval_sec is not None:
        cmd.extend(["--interval-sec", str(interval_sec)])

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert cycle_latest.exists()

    payload = json.loads(cycle_latest.read_text(encoding="utf-8"))
    stack_text = stack_calls.read_text(encoding="utf-8") if stack_calls.exists() else ""
    return payload, stack_text


def test_run_brain_autopdca_cycle_restarts_only_when_env_changes(
    tmp_path: Path,
) -> None:
    changed_payload, changed_stack = _run_cycle(tmp_path / "changed", apply_change=True)
    assert changed_payload["status"] == "ok"
    assert changed_payload["apply_result"]["env_changed"] is True
    assert changed_payload["apply_result"]["restart_performed"] is True
    assert "restart --env" in changed_stack
    assert "--services quant-order-manager" in changed_stack

    stable_payload, stable_stack = _run_cycle(tmp_path / "stable", apply_change=False)
    assert stable_payload["status"] == "ok"
    assert stable_payload["apply_result"]["env_changed"] is False
    assert stable_payload["apply_result"]["restart_performed"] is False
    assert stable_stack == ""


def test_run_brain_autopdca_cycle_respects_interval_without_force(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "interval"

    first_payload, first_stack = _run_cycle(run_dir, apply_change=True, force=True)
    assert first_payload["status"] == "ok"
    assert first_payload["apply_result"]["restart_performed"] is True
    first_calls = [line for line in first_stack.splitlines() if line.strip()]
    assert len(first_calls) == 1

    skipped_payload, skipped_stack = _run_cycle(
        run_dir,
        apply_change=False,
        force=False,
        interval_sec=3600,
    )
    assert skipped_payload["status"] == "skipped"
    assert skipped_payload["reason"] == "interval_not_elapsed"
    skipped_calls = [line for line in skipped_stack.splitlines() if line.strip()]
    assert len(skipped_calls) == len(first_calls)
