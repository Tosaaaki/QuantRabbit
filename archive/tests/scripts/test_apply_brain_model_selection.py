from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "apply_brain_model_selection.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCRIPT_PATH), *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_apply_brain_model_selection_updates_env_and_report(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    env_path = tmp_path / "brain-ollama.env"
    output_path = tmp_path / "selection.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-03-05T05:55:00+00:00",
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
                    {
                        "rank": 3,
                        "name": "llama3.1-8b",
                        "model": "llama3.1:8b",
                        "score": 1.217,
                        "parse_pass_rate": 1.0,
                        "alignment_coverage": 1.0,
                        "latency_p95_ms": 4250.294,
                        "outcome_score": 0.62,
                        "outcome_scored_trades": 10,
                    },
                ],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "BRAIN_OLLAMA_MODEL=old:model",
                "BRAIN_PROMPT_AUTO_TUNE_MODEL=old:model",
                "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=old:model",
                "BRAIN_TIMEOUT_SEC=6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run(
        "--benchmark",
        str(benchmark_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "BRAIN_OLLAMA_MODEL=qwen2.5:7b" in env_text
    assert "BRAIN_PROMPT_AUTO_TUNE_MODEL=gpt-oss:20b" in env_text
    assert "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=gpt-oss:20b" in env_text
    assert "BRAIN_TIMEOUT_SEC=8" in env_text

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["preflight_model"] == "qwen2.5:7b"
    assert report["autotune_model"] == "gpt-oss:20b"
    assert int(report["preflight_timeout_sec"]) == 8
    assert report["env_changed"] is True
    assert set(report["changed_keys"]) == {
        "BRAIN_OLLAMA_MODEL",
        "BRAIN_PROMPT_AUTO_TUNE_MODEL",
        "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL",
        "BRAIN_TIMEOUT_SEC",
    }


def test_apply_brain_model_selection_dry_run_keeps_env(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    env_path = tmp_path / "brain-ollama.env"
    output_path = tmp_path / "selection.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "ranking": [
                    {
                        "rank": 1,
                        "name": "gemma3-4b",
                        "model": "gemma3:4b",
                        "score": 1.05,
                        "parse_pass_rate": 0.82,
                        "alignment_coverage": 0.6,
                        "latency_p95_ms": 2200.0,
                    }
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    before = "BRAIN_OLLAMA_MODEL=old:model\n"
    env_path.write_text(before, encoding="utf-8")

    proc = _run(
        "--benchmark",
        str(benchmark_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
        "--dry-run",
    )

    assert env_path.read_text(encoding="utf-8") == before
    assert not output_path.exists()
    stdout = json.loads(proc.stdout)
    assert stdout["dry_run"] is True
    assert stdout["env_changed"] is True


def test_apply_brain_model_selection_reports_no_change(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    env_path = tmp_path / "brain-ollama.env"
    output_path = tmp_path / "selection.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "ranking": [
                    {
                        "rank": 1,
                        "name": "qwen2.5-7b",
                        "model": "qwen2.5:7b",
                        "score": 1.0,
                        "parse_pass_rate": 1.0,
                        "alignment_coverage": 1.0,
                        "latency_p95_ms": 3000.0,
                        "outcome_score": 0.6,
                        "outcome_scored_trades": 10,
                    },
                    {
                        "rank": 2,
                        "name": "gpt-oss20b",
                        "model": "gpt-oss:20b",
                        "score": 1.2,
                        "parse_pass_rate": 1.0,
                        "alignment_coverage": 1.0,
                        "latency_p95_ms": 25000.0,
                        "outcome_score": 0.7,
                        "outcome_scored_trades": 10,
                    },
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "BRAIN_OLLAMA_MODEL=qwen2.5:7b",
                "BRAIN_PROMPT_AUTO_TUNE_MODEL=gpt-oss:20b",
                "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=gpt-oss:20b",
                "BRAIN_TIMEOUT_SEC=6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run(
        "--benchmark",
        str(benchmark_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["env_changed"] is False
    assert report["changed_keys"] == []


def test_apply_brain_model_selection_caps_timeout_when_requested(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    env_path = tmp_path / "brain-ollama.env"
    output_path = tmp_path / "selection.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "ranking": [
                    {
                        "rank": 1,
                        "name": "qwen2.5-7b",
                        "model": "qwen2.5:7b",
                        "score": 1.0,
                        "parse_pass_rate": 1.0,
                        "alignment_coverage": 1.0,
                        "latency_p95_ms": 3640.751,
                        "outcome_score": 0.66,
                        "outcome_scored_trades": 10,
                    },
                    {
                        "rank": 2,
                        "name": "gpt-oss20b",
                        "model": "gpt-oss:20b",
                        "score": 1.2,
                        "parse_pass_rate": 1.0,
                        "alignment_coverage": 1.0,
                        "latency_p95_ms": 27791.581,
                        "outcome_score": 0.7,
                        "outcome_scored_trades": 10,
                    },
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "BRAIN_OLLAMA_MODEL=old:model",
                "BRAIN_PROMPT_AUTO_TUNE_MODEL=old:model",
                "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=old:model",
                "BRAIN_TIMEOUT_SEC=6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run(
        "--benchmark",
        str(benchmark_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
        "--timeout-cap-sec",
        "4",
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "BRAIN_TIMEOUT_SEC=4" in env_text
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert int(report["preflight_timeout_sec_raw"]) == 8
    assert int(report["preflight_timeout_sec"]) == 4
    assert float(report["timeout_cap_sec"]) == 4.0


def test_apply_brain_model_selection_uses_fallback_when_quality_is_too_low(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    env_path = tmp_path / "brain-ollama.env"
    output_path = tmp_path / "selection.json"

    benchmark_path.write_text(
        json.dumps(
            {
                "ranking": [
                    {
                        "rank": 1,
                        "name": "gpt-oss20b-default",
                        "model": "gpt-oss:20b",
                        "score": 0.3125,
                        "parse_pass_rate": 0.3125,
                        "alignment_coverage": 0.0,
                        "latency_p95_ms": 6436.184,
                    },
                    {
                        "rank": 2,
                        "name": "gpt-oss20b-trade-improve",
                        "model": "gpt-oss:20b",
                        "score": 0.0625,
                        "parse_pass_rate": 0.0625,
                        "alignment_coverage": 0.0,
                        "latency_p95_ms": 4706.583,
                    },
                ]
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "BRAIN_OLLAMA_MODEL=old:model",
                "BRAIN_PROMPT_AUTO_TUNE_MODEL=old:model",
                "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=old:model",
                "BRAIN_TIMEOUT_SEC=6",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _run(
        "--benchmark",
        str(benchmark_path),
        "--env-profile",
        str(env_path),
        "--output",
        str(output_path),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "BRAIN_OLLAMA_MODEL=qwen2.5:7b" in env_text
    assert "BRAIN_PROMPT_AUTO_TUNE_MODEL=gpt-oss:20b" in env_text
    assert "BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=gpt-oss:20b" in env_text

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["preflight_model"] == "qwen2.5:7b"
    assert report["autotune_model"] == "gpt-oss:20b"
