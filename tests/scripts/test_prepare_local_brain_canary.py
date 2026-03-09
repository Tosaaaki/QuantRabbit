from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_local_brain_canary.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def _write_benchmark(path: Path, *, generated_at: str) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "selected_source": "brain",
                "ranking": [
                    {
                        "rank": 1,
                        "name": "gpt-oss20b",
                        "model": "gpt-oss:20b",
                        "score": 1.245,
                        "parse_pass_rate": 1.0,
                        "alignment_coverage": 1.0,
                        "latency_p95_ms": 27791.581,
                        "outcome_score": 0.7,
                        "outcome_scored_trades": 10,
                    },
                    {
                        "rank": 2,
                        "name": "qwen2.5-7b",
                        "model": "qwen2.5:7b",
                        "score": 1.231,
                        "parse_pass_rate": 1.0,
                        "alignment_coverage": 1.0,
                        "latency_p95_ms": 3640.751,
                        "outcome_score": 0.66,
                        "outcome_scored_trades": 10,
                    },
                ],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _write_safe_profile(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "BRAIN_ENABLED=1",
                "ORDER_MANAGER_BRAIN_GATE_ENABLED=1",
                "ORDER_MANAGER_BRAIN_GATE_MODE=shadow",
                "ORDER_MANAGER_SERVICE_WORKERS=1",
                "BRAIN_OLLAMA_URL=http://127.0.0.1:11434/api/chat",
                "BRAIN_OLLAMA_MODEL=old:model",
                "BRAIN_POCKET_ALLOWLIST=micro",
                "BRAIN_STRATEGY_ALLOWLIST=MomentumBurst,MicroLevelReactor,MicroRangeBreak,MicroTrendRetest",
                "BRAIN_SAMPLE_RATE=0.35",
                "BRAIN_FAIL_POLICY=allow",
                "BRAIN_PROMPT_AUTO_TUNE_ENABLED=0",
                "BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED=0",
                "BRAIN_TIMEOUT_SEC=6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_local_brain_canary_syncs_safe_profile(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    selection_path = tmp_path / "selection.json"
    env_path = tmp_path / "brain-ollama-safe.env"
    output_path = tmp_path / "readiness.json"

    fresh_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_benchmark(benchmark_path, generated_at=fresh_ts)
    _write_safe_profile(env_path)

    _run(
        "--benchmark",
        str(benchmark_path),
        "--selection-output",
        str(selection_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
        "--timeout-cap-sec",
        "4",
        "--skip-ollama",
        "--skip-market",
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "BRAIN_OLLAMA_MODEL=qwen2.5:7b" in env_text
    assert "BRAIN_TIMEOUT_SEC=4" in env_text

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["checks"]["selection_sync_ok"] is True
    assert report["checks"]["benchmark_fresh"] is True
    assert report["checks"]["selection_fresh"] is True
    assert report["checks"]["profile_safe"] is True
    assert report["ready"]["enable_recommended"] is True
    assert report["required_models"] == ["qwen2.5:7b"]


def test_prepare_local_brain_canary_blocks_on_stale_benchmark(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    selection_path = tmp_path / "selection.json"
    env_path = tmp_path / "brain-ollama-safe.env"
    output_path = tmp_path / "readiness.json"

    stale_ts = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(timespec="seconds")
    _write_benchmark(benchmark_path, generated_at=stale_ts)
    _write_safe_profile(env_path)

    _run(
        "--benchmark",
        str(benchmark_path),
        "--selection-output",
        str(selection_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
        "--timeout-cap-sec",
        "4",
        "--max-benchmark-age-hours",
        "24",
        "--skip-ollama",
        "--skip-market",
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["checks"]["benchmark_fresh"] is False
    assert report["checks"]["profile_safe"] is True
    assert report["ready"]["enable_recommended"] is False
    assert "benchmark_fresh" in report["ready"]["blockers"]


def test_prepare_local_brain_canary_blocks_on_unsafe_profile(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    selection_path = tmp_path / "selection.json"
    env_path = tmp_path / "brain-ollama-safe.env"
    output_path = tmp_path / "readiness.json"

    fresh_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_benchmark(benchmark_path, generated_at=fresh_ts)
    env_path.write_text(
        "\n".join(
            [
                "BRAIN_ENABLED=1",
                "ORDER_MANAGER_BRAIN_GATE_ENABLED=1",
                "BRAIN_OLLAMA_URL=http://127.0.0.1:11434/api/chat",
                "BRAIN_OLLAMA_MODEL=old:model",
                "BRAIN_POCKET_ALLOWLIST=micro,scalp",
                "BRAIN_SAMPLE_RATE=0.90",
                "BRAIN_FAIL_POLICY=reduce",
                "BRAIN_PROMPT_AUTO_TUNE_ENABLED=1",
                "BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED=1",
                "BRAIN_TIMEOUT_SEC=9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run(
        "--benchmark",
        str(benchmark_path),
        "--selection-output",
        str(selection_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
        "--timeout-cap-sec",
        "4",
        "--skip-ollama",
        "--skip-market",
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["checks"]["profile_safe"] is False
    assert report["ready"]["enable_recommended"] is False
    assert "profile_safe" in report["ready"]["blockers"]


def test_prepare_local_brain_canary_accepts_profit_mode_profile(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    selection_path = tmp_path / "selection.json"
    env_path = tmp_path / "brain-ollama-profit.env"
    output_path = tmp_path / "readiness.json"

    fresh_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_benchmark(benchmark_path, generated_at=fresh_ts)
    env_path.write_text(
        "\n".join(
            [
                "BRAIN_ENABLED=1",
                "BRAIN_PROFILE_MODE=profit",
                "ORDER_MANAGER_BRAIN_GATE_ENABLED=1",
                "ORDER_MANAGER_BRAIN_GATE_MODE=shadow",
                "ORDER_MANAGER_SERVICE_WORKERS=1",
                "BRAIN_OLLAMA_URL=http://127.0.0.1:11434/api/chat",
                "BRAIN_OLLAMA_MODEL=old:model",
                "BRAIN_POCKET_ALLOWLIST=micro",
                "BRAIN_SAMPLE_RATE=1.0",
                "BRAIN_FAIL_POLICY=allow",
                "BRAIN_PROMPT_AUTO_TUNE_ENABLED=1",
                "BRAIN_RUNTIME_PARAM_AUTO_TUNE_ENABLED=1",
                "BRAIN_TIMEOUT_SEC=4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run(
        "--benchmark",
        str(benchmark_path),
        "--selection-output",
        str(selection_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
        "--timeout-cap-sec",
        "4",
        "--skip-ollama",
        "--skip-market",
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["checks"]["profile_safe"] is True
    assert report["ready"]["enable_recommended"] is True
