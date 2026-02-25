from __future__ import annotations

import json
from pathlib import Path
import subprocess

from analysis import forecast_improvement_worker as worker


def _completed(stdout: str, stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout=stdout, stderr=stderr)


def _cfg(tmp_path: Path) -> worker.WorkerConfig:
    return worker.WorkerConfig(
        env_file=tmp_path / "runtime.env",
        out_dir=tmp_path / "forecast_improvement",
        state_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
        latest_md_path=tmp_path / "latest.md",
        patterns="logs/candles_M1*.json",
        steps="1,5,10",
        horizons=("1m", "5m", "10m"),
        max_bars=1000,
        timeout_sec=120,
        keep_runs=24,
        hit_degrade_threshold=-0.002,
        mae_degrade_threshold=0.020,
        range_cov_degrade_threshold=-0.030,
    )


def test_build_eval_command_applies_runtime_overrides(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    out_json = tmp_path / "eval.json"

    monkeypatch.setenv("FORECAST_TECH_FEATURE_EXPANSION_GAIN", "0.04")
    monkeypatch.setenv("FORECAST_TECH_BREAKOUT_ADAPTIVE_WEIGHT_MAP", "1m=0.12,5m=0.20,10m=0.30")
    monkeypatch.setenv("FORECAST_TECH_DYNAMIC_WEIGHT_ENABLED", "true")

    cmd = worker._build_eval_command(cfg, eval_json_path=out_json)

    joined = " ".join(cmd)
    assert "--feature-expansion-gain 0.04" in joined
    assert "--breakout-adaptive-weight-map 1m=0.12,5m=0.20,10m=0.30" in joined
    assert "--dynamic-weight-enabled 1" in joined


def test_run_once_writes_state_history_and_report(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    def fake_runner(cmd: list[str], _timeout: int) -> subprocess.CompletedProcess[str]:
        script_name = Path(cmd[1]).name
        if script_name == "vm_forecast_snapshot.py":
            snapshot_rows = [
                ["1m", {"p_up": 0.55, "edge": 0.12, "expected_pips": 1.2, "feature_ts": "2026-02-25T00:00:00Z"}],
                ["5m", {"p_up": 0.48, "edge": -0.02, "expected_pips": -0.4, "feature_ts": "2026-02-25T00:00:00Z"}],
                ["10m", {"p_up": 0.53, "edge": 0.06, "expected_pips": 2.1, "feature_ts": "2026-02-25T00:00:00Z"}],
            ]
            return _completed(stdout=json.dumps(snapshot_rows, ensure_ascii=False), returncode=0)

        if script_name == "eval_forecast_before_after.py":
            out_idx = cmd.index("--json-out")
            out_path = Path(cmd[out_idx + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "summary": {
                    "bars": 1000,
                    "from_ts": "2026-02-20T00:00:00+00:00",
                    "to_ts": "2026-02-25T00:00:00+00:00",
                },
                "rows": [
                    {
                        "horizon": "1m",
                        "hit_before": 0.50,
                        "hit_after": 0.51,
                        "hit_delta": 0.01,
                        "mae_before": 2.2,
                        "mae_after": 2.1,
                        "mae_delta": -0.1,
                        "range_coverage_before": 0.62,
                        "range_coverage_after": 0.64,
                        "range_coverage_delta": 0.02,
                    },
                    {
                        "horizon": "5m",
                        "hit_before": 0.49,
                        "hit_after": 0.48,
                        "hit_delta": -0.01,
                        "mae_before": 3.8,
                        "mae_after": 3.85,
                        "mae_delta": 0.05,
                        "range_coverage_before": 0.58,
                        "range_coverage_after": 0.54,
                        "range_coverage_delta": -0.04,
                    },
                    {
                        "horizon": "10m",
                        "hit_before": 0.50,
                        "hit_after": 0.50,
                        "hit_delta": 0.0,
                        "mae_before": 5.0,
                        "mae_after": 4.99,
                        "mae_delta": -0.01,
                        "range_coverage_before": 0.61,
                        "range_coverage_after": 0.61,
                        "range_coverage_delta": 0.0,
                    },
                ],
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return _completed(stdout=f"json_out={out_path}\n", returncode=0)

        raise AssertionError(f"unexpected command: {cmd}")

    rc = worker.run_once(cfg, runner=fake_runner)

    assert rc == 0
    latest = worker._read_json(cfg.state_path)
    assert latest is not None
    assert latest["verdict"] == "degraded"
    assert latest["degraded_horizons"] == ["5m"]
    assert cfg.history_path.exists()
    assert len(cfg.history_path.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert cfg.latest_md_path.exists()
    assert "degraded_horizons: `5m`" in cfg.latest_md_path.read_text(encoding="utf-8")
