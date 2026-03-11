from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CYCLE_SCRIPT = REPO_ROOT / "scripts" / "run_local_feedback_cycle.py"
_CYCLE_SPEC = importlib.util.spec_from_file_location("run_local_feedback_cycle_test", CYCLE_SCRIPT)
assert _CYCLE_SPEC is not None and _CYCLE_SPEC.loader is not None
run_local_feedback_cycle = importlib.util.module_from_spec(_CYCLE_SPEC)
sys.modules.setdefault("run_local_feedback_cycle_test", run_local_feedback_cycle)
_CYCLE_SPEC.loader.exec_module(run_local_feedback_cycle)


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
    entry_path_env = tmp_path / "entry_path.env"
    entry_path_env.write_text("ENTRY_PATH_MARKER=entry-path-env\n", encoding="utf-8")
    participation_env = tmp_path / "participation.env"
    participation_env.write_text("PARTICIPATION_MARKER=participation-env\n", encoding="utf-8")
    market_context_env = tmp_path / "market_context.env"
    market_context_env.write_text("MARKET_CONTEXT_MARKER=market-context-env\n", encoding="utf-8")
    macro_news_env = tmp_path / "macro_news.env"
    macro_news_env.write_text("MACRO_NEWS_MARKER=macro-news-env\n", encoding="utf-8")
    feedback_env = tmp_path / "feedback.env"
    feedback_env.write_text("FEEDBACK_MARKER=feedback-env\n", encoding="utf-8")
    forecast_env = tmp_path / "forecast.env"
    forecast_env.write_text("FORECAST_MARKER=forecast-env\n", encoding="utf-8")
    replay_env = tmp_path / "replay.env"
    replay_env.write_text("REPLAY_MARKER=replay-env\n", encoding="utf-8")
    loser_cluster_env = tmp_path / "loser_cluster.env"
    loser_cluster_env.write_text("LOSER_CLUSTER_MARKER=loser-cluster-env\n", encoding="utf-8")
    auto_canary_env = tmp_path / "auto_canary.env"
    auto_canary_env.write_text("AUTO_CANARY_MARKER=auto-canary-env\n", encoding="utf-8")
    draft_env = tmp_path / "trade_findings_draft.env"
    draft_env.write_text("TRADE_FINDINGS_DRAFT_MARKER=trade-findings-draft-env\n", encoding="utf-8")

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
            "LOCAL_FEEDBACK_CYCLE_ENTRY_PATH_AGGREGATOR_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_ENTRY_PATH_AGGREGATOR_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'entry_path.json'} "
                "--env-key ENTRY_PATH_MARKER --marker entry-path"
            ),
            "LOCAL_FEEDBACK_CYCLE_ENTRY_PATH_AGGREGATOR_ENV_FILES": str(entry_path_env),
            "LOCAL_FEEDBACK_CYCLE_ENTRY_PATH_AGGREGATOR_OUTPUTS": str(outputs_dir / "entry_path.json"),
            "LOCAL_FEEDBACK_CYCLE_ENTRY_PATH_AGGREGATOR_INTERVAL_SEC": "300",
            "LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'participation.json'} "
                "--env-key PARTICIPATION_MARKER --marker participation"
            ),
            "LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_ENV_FILES": str(participation_env),
            "LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_OUTPUTS": str(outputs_dir / "participation.json"),
            "LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_INTERVAL_SEC": "300",
            "LOCAL_FEEDBACK_CYCLE_MARKET_CONTEXT_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_MARKET_CONTEXT_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'market_context.json'} "
                "--env-key MARKET_CONTEXT_MARKER --marker market-context"
            ),
            "LOCAL_FEEDBACK_CYCLE_MARKET_CONTEXT_ENV_FILES": str(market_context_env),
            "LOCAL_FEEDBACK_CYCLE_MARKET_CONTEXT_OUTPUTS": str(outputs_dir / "market_context.json"),
            "LOCAL_FEEDBACK_CYCLE_MARKET_CONTEXT_INTERVAL_SEC": "300",
            "LOCAL_FEEDBACK_CYCLE_MACRO_NEWS_CONTEXT_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_MACRO_NEWS_CONTEXT_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'macro_news.json'} "
                "--env-key MACRO_NEWS_MARKER --marker macro-news"
            ),
            "LOCAL_FEEDBACK_CYCLE_MACRO_NEWS_CONTEXT_ENV_FILES": str(macro_news_env),
            "LOCAL_FEEDBACK_CYCLE_MACRO_NEWS_CONTEXT_OUTPUTS": str(outputs_dir / "macro_news.json"),
            "LOCAL_FEEDBACK_CYCLE_MACRO_NEWS_CONTEXT_INTERVAL_SEC": "300",
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'feedback.json'} "
                "--env-key FEEDBACK_MARKER --marker feedback"
            ),
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_ENV_FILES": str(feedback_env),
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_OUTPUTS": str(outputs_dir / "feedback.json"),
            "LOCAL_FEEDBACK_CYCLE_STRATEGY_FEEDBACK_INTERVAL_SEC": "600",
            "LOCAL_FEEDBACK_CYCLE_TRADE_COUNTERFACTUAL_ENABLED": "0",
            "LOCAL_FEEDBACK_CYCLE_FORECAST_IMPROVEMENT_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_FORECAST_IMPROVEMENT_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'forecast.json'} "
                "--env-key FORECAST_MARKER --marker forecast"
            ),
            "LOCAL_FEEDBACK_CYCLE_FORECAST_IMPROVEMENT_ENV_FILES": str(forecast_env),
            "LOCAL_FEEDBACK_CYCLE_FORECAST_IMPROVEMENT_OUTPUTS": str(outputs_dir / "forecast.json"),
            "LOCAL_FEEDBACK_CYCLE_FORECAST_IMPROVEMENT_INTERVAL_SEC": "3600",
            "LOCAL_FEEDBACK_CYCLE_REPLAY_QUALITY_GATE_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_REPLAY_QUALITY_GATE_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'replay.json'} "
                "--env-key REPLAY_MARKER --marker replay"
            ),
            "LOCAL_FEEDBACK_CYCLE_REPLAY_QUALITY_GATE_ENV_FILES": str(replay_env),
            "LOCAL_FEEDBACK_CYCLE_REPLAY_QUALITY_GATE_OUTPUTS": str(outputs_dir / "replay.json"),
            "LOCAL_FEEDBACK_CYCLE_REPLAY_QUALITY_GATE_INTERVAL_SEC": "10800",
            "LOCAL_FEEDBACK_CYCLE_LOSER_CLUSTER_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_LOSER_CLUSTER_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'loser_cluster.json'} "
                "--env-key LOSER_CLUSTER_MARKER --marker loser-cluster"
            ),
            "LOCAL_FEEDBACK_CYCLE_LOSER_CLUSTER_ENV_FILES": str(loser_cluster_env),
            "LOCAL_FEEDBACK_CYCLE_LOSER_CLUSTER_OUTPUTS": str(outputs_dir / "loser_cluster.json"),
            "LOCAL_FEEDBACK_CYCLE_LOSER_CLUSTER_INTERVAL_SEC": "1200",
            "LOCAL_FEEDBACK_CYCLE_AUTO_CANARY_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_AUTO_CANARY_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'auto_canary.json'} "
                "--env-key AUTO_CANARY_MARKER --marker auto-canary"
            ),
            "LOCAL_FEEDBACK_CYCLE_AUTO_CANARY_ENV_FILES": str(auto_canary_env),
            "LOCAL_FEEDBACK_CYCLE_AUTO_CANARY_OUTPUTS": str(outputs_dir / "auto_canary.json"),
            "LOCAL_FEEDBACK_CYCLE_AUTO_CANARY_INTERVAL_SEC": "1200",
            "LOCAL_FEEDBACK_CYCLE_TRADE_FINDINGS_DRAFT_ENABLED": "1",
            "LOCAL_FEEDBACK_CYCLE_TRADE_FINDINGS_DRAFT_CMD": (
                f"{sys.executable} {fake_job} --output {outputs_dir / 'trade_findings_draft.json'} "
                "--env-key TRADE_FINDINGS_DRAFT_MARKER --marker trade-findings-draft"
            ),
            "LOCAL_FEEDBACK_CYCLE_TRADE_FINDINGS_DRAFT_ENV_FILES": str(draft_env),
            "LOCAL_FEEDBACK_CYCLE_TRADE_FINDINGS_DRAFT_OUTPUTS": str(outputs_dir / "trade_findings_draft.json"),
            "LOCAL_FEEDBACK_CYCLE_TRADE_FINDINGS_DRAFT_INTERVAL_SEC": "600",
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


def test_entry_path_aggregator_job_defaults_are_wired() -> None:
    job = run_local_feedback_cycle._build_job("entry_path_aggregator", sys.executable)

    assert job.enabled is True
    assert job.interval_sec == 120
    assert job.timeout_sec == 180
    assert job.command == (
        sys.executable,
        "scripts/entry_path_aggregator.py",
        "--lookback-hours",
        "6",
        "--limit",
        "6000",
        "--top-k",
        "8",
    )
    assert tuple(path.relative_to(REPO_ROOT).as_posix() for path in job.output_paths) == (
        "logs/entry_path_summary_latest.json",
        "logs/entry_path_summary_history.jsonl",
    )


def test_dynamic_alloc_job_defaults_are_wired() -> None:
    job = run_local_feedback_cycle._build_job("dynamic_alloc", sys.executable)

    assert job.enabled is True
    assert job.interval_sec == 120
    assert job.timeout_sec == 180
    assert job.command == (
        sys.executable,
        "scripts/dynamic_alloc_worker.py",
        "--limit",
        "2400",
        "--lookback-days",
        "3",
        "--min-trades",
        "12",
        "--setup-min-trades",
        "4",
        "--pf-cap",
        "2.0",
        "--target-use",
        "0.88",
        "--half-life-hours",
        "18",
    )


def test_participation_allocator_job_defaults_are_wired() -> None:
    job = run_local_feedback_cycle._build_job("participation_allocator", sys.executable)

    assert job.enabled is True
    assert job.interval_sec == 120
    assert job.timeout_sec == 180
    assert job.command == (
        sys.executable,
        "scripts/participation_allocator.py",
        "--lookback-hours",
        "6",
        "--min-attempts",
        "12",
        "--setup-min-attempts",
        "2",
        "--max-units-cut",
        "0.22",
        "--max-units-boost",
        "0.24",
        "--max-probability-boost",
        "0.10",
    )


def test_forecast_improvement_job_defaults_are_wired() -> None:
    job = run_local_feedback_cycle._build_job("forecast_improvement", sys.executable)

    assert job.enabled is True
    assert job.interval_sec == 3600
    assert job.timeout_sec == 1200
    assert job.retry_count == 0
    assert job.command == (sys.executable, "-m", "analysis.forecast_improvement_worker")
    assert tuple(path.relative_to(REPO_ROOT).as_posix() for path in job.env_files) == (
        "ops/env/quant-v2-runtime.env",
        "ops/env/quant-forecast-improvement-audit.env",
    )
    assert tuple(path.relative_to(REPO_ROOT).as_posix() for path in job.output_paths) == (
        "logs/forecast_improvement_latest.json",
        "logs/forecast_improvement_history.jsonl",
        "logs/reports/forecast_improvement/latest.md",
    )


def test_trade_findings_draft_job_defaults_are_wired() -> None:
    job = run_local_feedback_cycle._build_job("trade_findings_draft", sys.executable)

    assert job.enabled is True
    assert job.interval_sec == 600
    assert job.timeout_sec == 120
    assert job.retry_count == 0
    assert job.command == (sys.executable, "scripts/trade_findings_diary_draft.py")
    assert job.env_files == ()
    assert tuple(path.relative_to(REPO_ROOT).as_posix() for path in job.output_paths) == (
        "logs/trade_findings_draft_latest.json",
        "logs/trade_findings_draft_history.jsonl",
        "logs/trade_findings_draft_latest.md",
    )


def test_run_local_feedback_cycle_updates_jobs_and_env_files(tmp_path: Path) -> None:
    fake_job = _prepare_fake_job(tmp_path)
    env = _base_env(tmp_path, fake_job)

    payload = _run_cycle(tmp_path, force=True, env=env)
    assert payload["status"] == "ok"
    jobs = {item["job"]: item for item in payload["jobs"] if item.get("job")}

    dynamic = jobs["dynamic_alloc"]
    assert dynamic["status"] == "ok"
    assert any(output["updated"] for output in dynamic["outputs"])

    entry_path = jobs["entry_path_aggregator"]
    assert entry_path["status"] == "ok"
    assert any(output["updated"] for output in entry_path["outputs"])

    participation = jobs["participation_allocator"]
    assert participation["status"] == "ok"
    assert any(output["updated"] for output in participation["outputs"])

    market_context = jobs["market_context"]
    assert market_context["status"] == "ok"
    assert any(output["updated"] for output in market_context["outputs"])

    macro_news = jobs["macro_news_context"]
    assert macro_news["status"] == "ok"
    assert any(output["updated"] for output in macro_news["outputs"])

    feedback = jobs["strategy_feedback"]
    assert feedback["status"] == "ok"
    assert any(output["updated"] for output in feedback["outputs"])

    forecast = jobs["forecast_improvement"]
    assert forecast["status"] == "ok"
    assert any(output["updated"] for output in forecast["outputs"])

    dynamic_output = json.loads((tmp_path / "outputs" / "dynamic.json").read_text(encoding="utf-8"))
    assert dynamic_output == {"marker": "dynamic", "env_value": "dynamic-env"}

    feedback_output = json.loads((tmp_path / "outputs" / "feedback.json").read_text(encoding="utf-8"))
    assert feedback_output == {"marker": "feedback", "env_value": "feedback-env"}

    forecast_output = json.loads((tmp_path / "outputs" / "forecast.json").read_text(encoding="utf-8"))
    assert forecast_output == {"marker": "forecast", "env_value": "forecast-env"}

    replay_output = json.loads((tmp_path / "outputs" / "replay.json").read_text(encoding="utf-8"))
    assert replay_output == {"marker": "replay", "env_value": "replay-env"}

    loser_cluster_output = json.loads((tmp_path / "outputs" / "loser_cluster.json").read_text(encoding="utf-8"))
    assert loser_cluster_output == {"marker": "loser-cluster", "env_value": "loser-cluster-env"}

    auto_canary_output = json.loads((tmp_path / "outputs" / "auto_canary.json").read_text(encoding="utf-8"))
    assert auto_canary_output == {"marker": "auto-canary", "env_value": "auto-canary-env"}

    draft = jobs["trade_findings_draft"]
    assert draft["status"] == "ok"
    assert any(output["updated"] for output in draft["outputs"])

    draft_output = json.loads((tmp_path / "outputs" / "trade_findings_draft.json").read_text(encoding="utf-8"))
    assert draft_output == {"marker": "trade-findings-draft", "env_value": "trade-findings-draft-env"}


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
    assert jobs["entry_path_aggregator"]["status"] == "skipped"
    assert jobs["entry_path_aggregator"]["reason"] == "interval_not_elapsed"
    assert jobs["participation_allocator"]["status"] == "skipped"
    assert jobs["participation_allocator"]["reason"] == "interval_not_elapsed"
    assert jobs["market_context"]["status"] == "skipped"
    assert jobs["market_context"]["reason"] == "interval_not_elapsed"
    assert jobs["macro_news_context"]["status"] == "skipped"
    assert jobs["macro_news_context"]["reason"] == "interval_not_elapsed"
    assert jobs["strategy_feedback"]["status"] == "skipped"
    assert jobs["strategy_feedback"]["reason"] == "interval_not_elapsed"
    assert jobs["forecast_improvement"]["status"] == "skipped"
    assert jobs["forecast_improvement"]["reason"] == "interval_not_elapsed"
    assert jobs["replay_quality_gate"]["status"] == "skipped"
    assert jobs["replay_quality_gate"]["reason"] == "interval_not_elapsed"
    assert jobs["loser_cluster"]["status"] == "skipped"
    assert jobs["loser_cluster"]["reason"] == "interval_not_elapsed"
    assert jobs["auto_canary"]["status"] == "skipped"
    assert jobs["auto_canary"]["reason"] == "interval_not_elapsed"
    assert jobs["trade_findings_draft"]["status"] == "skipped"
    assert jobs["trade_findings_draft"]["reason"] == "interval_not_elapsed"
