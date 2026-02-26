from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from analysis import replay_quality_gate_worker as worker
import pytest


def _completed(stdout: str, stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout=stdout, stderr=stderr)


def _base_cfg(tmp_path: Path) -> worker.WorkerConfig:
    return worker.WorkerConfig(
        config_path=tmp_path / "cfg.yaml",
        out_dir=tmp_path / "replay_quality_gate",
        state_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
        timeout_sec=60,
        keep_runs=4,
        strict=False,
        ticks_glob="",
        workers="",
        backend="",
    )


def test_extract_report_path_from_stdout(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260221_010203"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_dir / "quality_gate_report.json"
    report.write_text("{}", encoding="utf-8")

    stdout = f"hello\n{report}\n"
    resolved = worker._extract_report_path_from_stdout(stdout, out_dir=tmp_path)

    assert resolved == report


def test_cleanup_old_runs_keeps_newest_and_protected(tmp_path: Path) -> None:
    run_dirs = []
    for idx in range(5):
        run_dir = tmp_path / f"20260221_00000{idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        report = run_dir / "quality_gate_report.json"
        report.write_text("{}", encoding="utf-8")
        stamp = 1_700_000_000 + idx
        os.utime(report, (stamp, stamp))
        run_dirs.append(report)

    protected = run_dirs[-1]
    removed = worker._cleanup_old_runs(tmp_path, keep_runs=2, protected=protected)

    remaining = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert len(remaining) == 2
    assert protected.parent.name in remaining
    assert removed


def test_run_once_writes_state_and_history(tmp_path: Path) -> None:
    out_dir = tmp_path / "replay_quality_gate"
    run_dir = out_dir / "20260221_010203"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_json = run_dir / "quality_gate_report.json"
    report_md = run_dir / "quality_gate_report.md"
    report_json.write_text(
        '{"meta":{"fold_count":3},"overall":{"status":"fail","failing_workers":["session_open"]}}',
        encoding="utf-8",
    )
    report_md.write_text("# report\n", encoding="utf-8")

    cfg = replace(_base_cfg(tmp_path), out_dir=out_dir)

    def fake_runner(_cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(stdout=f"{report_json}\n", returncode=0)

    rc = worker.run_once(cfg, runner=fake_runner)

    assert rc == 0
    latest = worker._read_json(cfg.state_path)
    assert latest is not None
    assert latest["gate_status"] == "fail"
    assert latest["failing_workers"] == ["session_open"]
    assert latest["report_json_path"] == str(report_json)
    assert latest["report_md_path"] == str(report_md)
    assert cfg.history_path.exists()
    assert len(cfg.history_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_collect_auto_improve_strategies_filters_virtual_workers(tmp_path: Path) -> None:
    cfg = replace(_base_cfg(tmp_path), auto_improve_scope="failing")
    report = {
        "meta": {
            "workers": ["session_open", "pocket:scalp", "__overall__"],
        },
        "overall": {
            "failing_workers": ["session_open", "pocket:scalp", "__overall__", "session_open"],
        },
    }

    assert worker._collect_auto_improve_strategies(report, cfg) == ["session_open"]


def test_apply_reentry_updates_sets_block_hours(tmp_path: Path) -> None:
    if worker.yaml is None:
        pytest.skip("PyYAML is not available")

    reentry_path = tmp_path / "worker_reentry.yaml"
    reentry_path.write_text(
        "defaults:\n  block_jst_hours: []\nstrategies:\n  session_open:\n    block_jst_hours: []\n",
        encoding="utf-8",
    )

    result = worker._apply_reentry_updates(
        reentry_path=reentry_path,
        strategy_hours={"session_open": [5, 2, 5, 31, -1]},
    )

    assert result["applied"] is True
    payload = worker._load_yaml_dict(reentry_path)
    strategies = payload.get("strategies")
    assert isinstance(strategies, dict)
    session_open = strategies.get("session_open")
    assert isinstance(session_open, dict)
    assert session_open.get("block_jst_hours") == [2, 5]


def test_run_once_auto_improve_updates_reentry(tmp_path: Path) -> None:
    if worker.yaml is None:
        pytest.skip("PyYAML is not available")

    out_dir = tmp_path / "replay_quality_gate"
    run_dir = out_dir / "20260221_010203"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_json = run_dir / "quality_gate_report.json"
    report_md = run_dir / "quality_gate_report.md"
    report_json.write_text(
        '{"meta":{"fold_count":2,"workers":["session_open"]},"overall":{"status":"fail","failing_workers":["session_open"]}}',
        encoding="utf-8",
    )
    report_md.write_text("# report\n", encoding="utf-8")

    reentry_path = tmp_path / "worker_reentry.yaml"
    reentry_path.write_text("defaults:\n  block_jst_hours: []\nstrategies: {}\n", encoding="utf-8")

    cfg = replace(
        _base_cfg(tmp_path),
        out_dir=out_dir,
        auto_improve_enabled=True,
        auto_improve_scope="failing",
        auto_improve_min_trades=1,
        auto_improve_counterfactual_out_dir=tmp_path / "cf_out",
        auto_improve_reentry_config_path=reentry_path,
        auto_improve_apply_reentry=True,
    )

    def fake_runner(_cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(stdout=f"{report_json}\n", returncode=0)

    def fake_counterfactual_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out_path = Path(cmd[cmd.index("--out-path") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            '{"summary":{"trades":12,"stuck_trade_ratio":0.5},"policy_hints":{"block_jst_hours":[3,6]}}',
            encoding="utf-8",
        )
        return _completed(stdout=str(out_path), returncode=0)

    rc = worker.run_once(cfg, runner=fake_runner, counterfactual_runner=fake_counterfactual_runner)

    assert rc == 0
    latest = worker._read_json(cfg.state_path)
    assert latest is not None
    auto = latest.get("auto_improve")
    assert isinstance(auto, dict)
    assert auto.get("status") == "applied"
    payload = worker._load_yaml_dict(reentry_path)
    assert payload.get("strategies", {}).get("session_open", {}).get("block_jst_hours") == [3, 6]


def test_run_once_auto_improve_respects_apply_cooldown(tmp_path: Path) -> None:
    if worker.yaml is None:
        pytest.skip("PyYAML is not available")

    out_dir = tmp_path / "replay_quality_gate"
    run_dir = out_dir / "20260221_010203"
    run_dir.mkdir(parents=True, exist_ok=True)
    report_json = run_dir / "quality_gate_report.json"
    report_md = run_dir / "quality_gate_report.md"
    report_json.write_text(
        '{"meta":{"fold_count":2,"workers":["session_open"]},"overall":{"status":"fail","failing_workers":["session_open"]}}',
        encoding="utf-8",
    )
    report_md.write_text("# report\n", encoding="utf-8")

    reentry_path = tmp_path / "worker_reentry.yaml"
    reentry_path.write_text(
        "defaults:\n  block_jst_hours: []\nstrategies:\n  session_open:\n    block_jst_hours: [1]\n",
        encoding="utf-8",
    )
    apply_state_path = tmp_path / "replay_auto_improve_state.json"
    apply_state_path.write_text(
        '{"applied_at":"' + datetime.now(timezone.utc).isoformat() + '","updates":{"session_open":[1]}}',
        encoding="utf-8",
    )

    cfg = replace(
        _base_cfg(tmp_path),
        out_dir=out_dir,
        auto_improve_enabled=True,
        auto_improve_scope="failing",
        auto_improve_min_trades=1,
        auto_improve_counterfactual_out_dir=tmp_path / "cf_out",
        auto_improve_reentry_config_path=reentry_path,
        auto_improve_apply_reentry=True,
        auto_improve_min_apply_interval_sec=3600,
        auto_improve_apply_state_path=apply_state_path,
    )

    def fake_runner(_cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(stdout=f"{report_json}\n", returncode=0)

    def fake_counterfactual_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        out_path = Path(cmd[cmd.index("--out-path") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            '{"summary":{"trades":12,"stuck_trade_ratio":0.5},"policy_hints":{"block_jst_hours":[3,6]}}',
            encoding="utf-8",
        )
        return _completed(stdout=str(out_path), returncode=0)

    rc = worker.run_once(cfg, runner=fake_runner, counterfactual_runner=fake_counterfactual_runner)

    assert rc == 0
    latest = worker._read_json(cfg.state_path)
    assert latest is not None
    auto = latest.get("auto_improve")
    assert isinstance(auto, dict)
    assert auto.get("status") == "analyzed"
    assert auto.get("reason") == "reentry_apply_cooldown"
    payload = worker._load_yaml_dict(reentry_path)
    assert payload.get("strategies", {}).get("session_open", {}).get("block_jst_hours") == [1]
