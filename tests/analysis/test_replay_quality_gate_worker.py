from __future__ import annotations

import os
from pathlib import Path
import subprocess

from analysis import replay_quality_gate_worker as worker


def _completed(stdout: str, stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout=stdout, stderr=stderr)


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

    cfg = worker.WorkerConfig(
        config_path=tmp_path / "cfg.yaml",
        out_dir=out_dir,
        state_path=tmp_path / "latest.json",
        history_path=tmp_path / "history.jsonl",
        timeout_sec=60,
        keep_runs=4,
        strict=False,
        ticks_glob="",
        workers="",
        backend="",
    )

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
