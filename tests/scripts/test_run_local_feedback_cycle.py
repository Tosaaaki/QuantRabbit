from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CYCLE_SCRIPT = REPO_ROOT / "scripts" / "run_local_feedback_cycle.py"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _prepare_fake_job(tmp_path: Path) -> Path:
    job_script = tmp_path / "fake_job.py"
    _write_executable(
        job_script,
        """#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--env-key", required=True)
parser.add_argument("--marker", required=True)
args = parser.parse_args()

payload = {
    "marker": args.marker,
    "env_value": os.getenv(args.env_key),
}
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=True))
""",
    )
    return job_script


def _base_env(tmp_path: Path, fake_job: Path) -> dict[str, str]:
    logs_dir = tmp_path / "logs"
    outputs_dir = tmp_path / "outputs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    dyn_env = tmp_path / "dynamic.env"
    dyn_env.write_text("DYN_MARKER=dynamic-env\n", encoding="utf-8")
    feedback_env = tmp_path / "feedback.env"
    feedback_env.write_text("FEEDBACK_MARKER=feedback-env\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "LOCAL_FEEDBACK_CYCLE_LOG_DIR": str(logs_dir),
            "LOCAL_FEEDBACK_CYCLE_STATE_FILE": str(logs_dir / "state.json"),
            "LOCAL_FEEDBACK_CYCLE_OUTPUT": str(logs_dir / "latest.json"),
            "LOCAL_FEEDBACK_CYCLE_HISTORY": str(logs_dir / "history.jsonl"),
            "LOCAL_FEEDBACK_CYCLE_RUN_LOG_DIR": str(logs_dir / "run_logs"),
            "LOCAL_FEEDBACK_CYCLE_LOCK_DIR": str(logs_dir / "lock"),
            "LOCAL_FEEDBACK_CYCLE_PYTHON_BIN": sys.executable,
            "LOCAL_FEEDBACK_CYCLE_DYNAMIC_ALLOC_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_DYNAMIC_ALLOC_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'dynamic.json'} "
                "--env-key DYN_MARKER --marker dynamic"
            ),
            "LOCAL_FEEDBACK_CYCLE_DYNAMIC_ALLOC_ENV_FILES": str(dyn_env),
            "LOCAL_FEEDBACK_CYCLE_DYNAMIC_ALLOC_OUTPUTS": str(outputs_dir / "dynamic.json"),
            "LOCAL_FEEDBACK_CYCLE_DYNAMIC_ALLOC_INTERVAL_SEC": "120",
            "LOCAL_FEEDBACK_CYCLE_PATTERN_BOOK_ENABLED": "0",
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'feedback.json'} "
                "--env-key FEEDBACK_MARKER --marker feedback"
            ),
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_ENV_FILES": str(feedback_env),
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_OUTPUTS": str(outputs_dir / "feedback.json"),
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_INTERVAL_SEC": "600",
            "LOCAL_FEEDBACK_CYCLE_TRADE_COUNTERFACTUAL_ENABLED": "0",
        }
    )
    return env


def _run_cycle(tmp_path: Path, *, force: bool, env: dict[str, str]) -> dict:
    cmd = [sys.executable, str(CYCLE_SCRIPT)]
    if force:
        cmd.append("--force")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    return json.loads((tmp_path / "logs" / "latest.json").read_text(encoding="utf-8"))


def test_run_local_feedback_cycle_updates_jobs_and_env_files(tmp_path: Path) -> None:
    fake_job = _prepare_fake_job(tmp_path)
    env = _base_env(tmp_path, fake_job)

    payload = _run_cycle(tmp_path, force=True, env=env)
    assert payload["status"] == "ok"
    jobs = {item["job"]: item for item in payload["jobs"] if item.get("job")}

    dynamic = jobs["dynamic_alloc"]
    assert dynamic["status"] == "ok"
    assert any(output["updated"] for output in dynamic["outputs"])

    feedback = jobs["strategy_feedback"]
    assert feedback["status"] == "ok"
    assert any(output["updated"] for output in feedback["outputs"])

    dynamic_output = json.loads((tmp_path / "outputs" / "dynamic.json").read_text(encoding="utf-8"))
    assert dynamic_output == {"marker": "dynamic", "env_value": "dynamic-env"}

    feedback_output = json.loads((tmp_path / "outputs" / "feedback.json").read_text(encoding="utf-8"))
    assert feedback_output == {"marker": "feedback", "env_value": "feedback-env"}


def test_run_local_feedback_cycle_respects_intervals_without_force(tmp_path: Path) -> None:
    fake_job = _prepare_fake_job(tmp_path)
    env = _base_env(tmp_path, fake_job)

    first = _run_cycle(tmp_path, force=True, env=env)
    assert first["status"] == "ok"

    second = _run_cycle(tmp_path, force=False, env=env)
    assert second["status"] == "skipped"
    jobs = {item["job"]: item for item in second["jobs"] if item.get("job")}
    assert jobs["dynamic_alloc"]["status"] == "skipped"
    assert jobs["dynamic_alloc"]["reason"] == "interval_not_elapsed"
    assert jobs["strategy_feedback"]["status"] == "skipped"
    assert jobs["strategy_feedback"]["reason"] == "interval_not_elapsed"
