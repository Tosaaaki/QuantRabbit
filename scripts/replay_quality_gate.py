#!/usr/bin/env python3
"""Run replay workers across multiple tick files and apply quality gates.

This script standardizes internal replay validation using walk-forward folds.
Supported execution backends:
  - scripts/replay_exit_workers_groups.py (worker-group replay)
  - scripts/replay_exit_workers.py (main replay)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from glob import glob
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytics.replay_quality_gate import (
    GateThreshold,
    build_walk_forward_folds,
    compute_trade_metrics,
    evaluate_fold_gate,
    summarize_worker_folds,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out):
        return float(default)
    return out


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


def _resolve_workers(config: Mapping[str, Any], cli_workers: str | None) -> list[str]:
    if cli_workers:
        return [w.strip() for w in cli_workers.split(",") if w.strip()]
    raw = config.get("workers")
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    return []


def _build_threshold(
    *,
    base: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> GateThreshold:
    merged: dict[str, Any] = dict(base)
    if override:
        merged.update(dict(override))
    return GateThreshold(
        min_train_trades=max(0, _to_int(merged.get("min_train_trades"), 0)),
        min_test_trades=max(0, _to_int(merged.get("min_test_trades"), 0)),
        min_test_pf=_to_float(merged.get("min_test_pf"), 0.0),
        min_test_win_rate=_to_float(merged.get("min_test_win_rate"), 0.0),
        min_test_total_pips=_to_float(merged.get("min_test_total_pips"), -1.0e9),
        max_test_drawdown_pips=_to_float(merged.get("max_test_drawdown_pips"), 1.0e9),
        min_pf_stability_ratio=_to_float(merged.get("min_pf_stability_ratio"), 0.0),
    )


def _build_replay_command_groups(
    *,
    ticks_path: Path,
    workers: list[str],
    out_dir: Path,
    replay_cfg: Mapping[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "replay_exit_workers_groups.py"),
        "--ticks",
        str(ticks_path),
        "--workers",
        ",".join(workers),
        "--out-dir",
        str(out_dir),
    ]
    if _to_bool(replay_cfg.get("no_hard_sl"), default=True):
        cmd.append("--no-hard-sl")
    if _to_bool(replay_cfg.get("no_hard_tp"), default=False):
        cmd.append("--no-hard-tp")
    if _to_bool(replay_cfg.get("exclude_end_of_replay"), default=True):
        cmd.append("--exclude-end-of-replay")
    if _to_bool(replay_cfg.get("tune"), default=False):
        cmd.append("--tune")
    if _to_bool(replay_cfg.get("realistic"), default=False):
        cmd.append("--realistic")

    resample_sec = _to_float(replay_cfg.get("resample_sec"), 0.0)
    if resample_sec > 0:
        cmd.extend(["--resample-sec", str(resample_sec)])

    for key, flag in (
        ("slip_base_pips", "--slip-base-pips"),
        ("slip_spread_coef", "--slip-spread-coef"),
        ("slip_atr_coef", "--slip-atr-coef"),
        ("slip_latency_coef", "--slip-latency-coef"),
        ("latency_ms", "--latency-ms"),
    ):
        if key in replay_cfg:
            cmd.extend([flag, str(replay_cfg[key])])

    fill_mode = str(replay_cfg.get("fill_mode") or "").strip().lower()
    if fill_mode in {"lko", "next_tick"}:
        cmd.extend(["--fill-mode", fill_mode])
    return cmd


def _build_replay_command_main(
    *,
    ticks_path: Path,
    out_path: Path,
    replay_cfg: Mapping[str, Any],
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "replay_exit_workers.py"),
        "--ticks",
        str(ticks_path),
        "--out",
        str(out_path),
        "--no-stdout",
    ]
    if _to_bool(replay_cfg.get("no_hard_sl"), default=True):
        cmd.append("--no-hard-sl")
    if _to_bool(replay_cfg.get("no_hard_tp"), default=False):
        cmd.append("--no-hard-tp")
    if _to_bool(replay_cfg.get("exclude_end_of_replay"), default=True):
        cmd.append("--exclude-end-of-replay")
    if _to_bool(replay_cfg.get("fast_only"), default=False):
        cmd.append("--fast-only")
    if _to_bool(replay_cfg.get("sp_only"), default=False):
        cmd.append("--sp-only")
    if _to_bool(replay_cfg.get("sp_live_entry"), default=False):
        cmd.append("--sp-live-entry")
    if _to_bool(replay_cfg.get("disable_macro"), default=False):
        cmd.append("--disable-macro")
    if _to_bool(replay_cfg.get("no_main_strategies"), default=False):
        cmd.append("--no-main-strategies")
    if _to_bool(replay_cfg.get("main_only"), default=False):
        cmd.append("--main-only")
    start = str(replay_cfg.get("start") or "").strip()
    if start:
        cmd.extend(["--start", start])
    end = str(replay_cfg.get("end") or "").strip()
    if end:
        cmd.extend(["--end", end])
    if not start and not end:
        date_match = re.search(r"(20\d{6})", ticks_path.name)
        if date_match:
            ymd = date_match.group(1)
            yyyy_mm_dd = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
            intraday_start = str(replay_cfg.get("intraday_start_utc") or "").strip()
            intraday_end = str(replay_cfg.get("intraday_end_utc") or "").strip()
            if intraday_start:
                cmd.extend(["--start", f"{yyyy_mm_dd}T{intraday_start}+00:00"])
            if intraday_end:
                cmd.extend(["--end", f"{yyyy_mm_dd}T{intraday_end}+00:00"])
    instrument = str(replay_cfg.get("instrument") or "").strip()
    if instrument:
        cmd.extend(["--instrument", instrument])

    realistic = _to_bool(replay_cfg.get("realistic"), default=False)
    slip_values: dict[str, Any] = {}
    if realistic:
        slip_values.update(
            {
                "latency_ms": 180.0,
                "slip_base_pips": 0.02,
                "slip_spread_coef": 0.15,
                "slip_atr_coef": 0.02,
                "slip_latency_coef": 0.0006,
            }
        )
    for key in (
        "slip_base_pips",
        "slip_spread_coef",
        "slip_atr_coef",
        "slip_latency_coef",
        "latency_ms",
    ):
        if key in replay_cfg:
            slip_values[key] = replay_cfg[key]
    for key, flag in (
        ("slip_base_pips", "--slip-base-pips"),
        ("slip_spread_coef", "--slip-spread-coef"),
        ("slip_atr_coef", "--slip-atr-coef"),
        ("slip_latency_coef", "--slip-latency-coef"),
        ("latency_ms", "--latency-ms"),
    ):
        if key in slip_values:
            cmd.extend([flag, str(slip_values[key])])

    fill_mode = str(replay_cfg.get("fill_mode") or ("next_tick" if realistic else "")).strip().lower()
    if fill_mode in {"lko", "next_tick"}:
        cmd.extend(["--fill-mode", fill_mode])
    return cmd


def _load_worker_trades(
    *,
    run_dir: Path,
    workers: list[str],
    exclude_end_of_replay: bool,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for worker in workers:
        path = run_dir / f"replay_exit_{worker}_base.json"
        if not path.exists():
            out[worker] = []
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            out[worker] = []
            continue
        trades = payload.get("trades") or []
        if not isinstance(trades, list):
            out[worker] = []
            continue
        typed_trades = [t for t in trades if isinstance(t, dict)]
        if exclude_end_of_replay:
            typed_trades = [t for t in typed_trades if str(t.get("reason") or "") != "end_of_replay"]
        out[worker] = typed_trades
    return out


def _load_worker_trades_main(
    *,
    out_path: Path,
    workers: list[str],
    exclude_end_of_replay: bool,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {w: [] for w in workers}
    if not out_path.exists():
        return out
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    trades = payload.get("trades") or []
    if not isinstance(trades, list):
        return out
    typed_trades = [t for t in trades if isinstance(t, dict)]
    if exclude_end_of_replay:
        typed_trades = [t for t in typed_trades if str(t.get("reason") or "") != "end_of_replay"]

    for worker in workers:
        if worker == "__overall__":
            out[worker] = list(typed_trades)
            continue
        if worker.startswith("pocket:"):
            pocket = worker.split(":", 1)[1].strip().lower()
            out[worker] = [t for t in typed_trades if str(t.get("pocket") or "").lower() == pocket]
            continue
        if worker.startswith("source:"):
            source = worker.split(":", 1)[1].strip().lower()
            out[worker] = [t for t in typed_trades if str(t.get("source") or "").lower() == source]
            continue
        key = worker.strip().lower()
        out[worker] = [
            t
            for t in typed_trades
            if str(t.get("strategy_tag") or "").lower() == key or str(t.get("strategy") or "").lower() == key
        ]
    return out


def _render_markdown(report: Mapping[str, Any]) -> str:
    meta = report.get("meta") or {}
    overall = report.get("overall") or {}
    worker_results = report.get("worker_results") or {}

    lines = [
        "# Replay Quality Gate",
        "",
        "## Summary",
        f"- generated_at: {meta.get('generated_at')}",
        f"- tick_files: {meta.get('tick_file_count')}",
        f"- folds: {meta.get('fold_count')}",
        f"- status: {overall.get('status')}",
        f"- failing_workers: {', '.join(overall.get('failing_workers') or []) or '-'}",
        "",
        "## Worker Results",
        "| worker | folds | passed | pass_rate | median_test_pf | median_test_win_rate | median_test_max_dd | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for worker in sorted(worker_results.keys()):
        summary = (worker_results[worker] or {}).get("summary") or {}
        lines.append(
            "| {worker} | {folds} | {passed} | {pass_rate:.2f} | {pf:.3f} | {wr:.3f} | {dd:.3f} | {status} |".format(
                worker=worker,
                folds=int(summary.get("folds") or 0),
                passed=int(summary.get("passed_folds") or 0),
                pass_rate=float(summary.get("pass_rate") or 0.0),
                pf=float(summary.get("median_test_pf") or 0.0),
                wr=float(summary.get("median_test_win_rate") or 0.0),
                dd=float(summary.get("median_test_max_drawdown_pips") or 0.0),
                status=str(summary.get("status") or "fail"),
            )
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay walk-forward quality gate")
    ap.add_argument("--config", type=Path, default=Path("config/replay_quality_gate.yaml"))
    ap.add_argument("--ticks-glob", default=None, help="Glob pattern for replay tick JSONL files.")
    ap.add_argument("--workers", default=None, help="Comma separated worker names.")
    ap.add_argument("--backend", default=None, choices=("exit_workers_groups", "exit_workers_main"))
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/replay_quality_gate"))
    ap.add_argument("--train-files", type=int, default=None)
    ap.add_argument("--test-files", type=int, default=None)
    ap.add_argument("--step-files", type=int, default=None)
    ap.add_argument("--min-fold-pass-rate", type=float, default=None)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--strict", action="store_true", help="Return non-zero if gate fails.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_yaml(args.config)

    workers = _resolve_workers(config, args.workers)
    if not workers:
        print("No workers specified. Use --workers or config.replay_quality_gate.yaml:workers", file=sys.stderr)
        return 2

    ticks_glob = args.ticks_glob or str(config.get("ticks_glob") or "")
    if not ticks_glob:
        print("No ticks_glob specified. Use --ticks-glob or config file.", file=sys.stderr)
        return 2

    tick_files = [Path(p) for p in sorted(glob(ticks_glob)) if Path(p).is_file()]
    if not tick_files:
        print(f"No tick files matched: {ticks_glob}", file=sys.stderr)
        return 2

    wf_cfg = config.get("walk_forward") if isinstance(config.get("walk_forward"), dict) else {}
    train_files = max(1, int(args.train_files or wf_cfg.get("train_files") or 4))
    test_files = max(1, int(args.test_files or wf_cfg.get("test_files") or 1))
    step_files = max(1, int(args.step_files or wf_cfg.get("step_files") or 1))
    min_fold_pass_rate = _to_float(
        args.min_fold_pass_rate if args.min_fold_pass_rate is not None else wf_cfg.get("min_fold_pass_rate"),
        1.0,
    )

    tick_keys = [p.name for p in tick_files]
    folds = build_walk_forward_folds(
        tick_keys,
        train_files=train_files,
        test_files=test_files,
        step_files=step_files,
    )
    if not folds:
        print(
            f"Insufficient tick files for walk-forward: files={len(tick_files)} train={train_files} test={test_files}",
            file=sys.stderr,
        )
        return 2

    replay_cfg = config.get("replay") if isinstance(config.get("replay"), dict) else {}
    backend = str(args.backend or replay_cfg.get("backend") or "exit_workers_groups").strip().lower()
    if backend not in {"exit_workers_groups", "exit_workers_main"}:
        print(f"Unsupported replay backend: {backend}", file=sys.stderr)
        return 2
    exclude_end = _to_bool(replay_cfg.get("exclude_end_of_replay"), default=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir / run_id
    run_root = out_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    commands_log: list[dict[str, Any]] = []
    trades_by_file: dict[str, dict[str, list[dict[str, Any]]]] = {}

    print(
        f"[INFO] replay quality gate start backend={backend} workers={','.join(workers)} files={len(tick_files)} folds={len(folds)}"
    )
    for idx, tick_path in enumerate(tick_files, start=1):
        run_dir = run_root / tick_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        main_out_path = run_dir / "replay_exit_workers.json"
        if backend == "exit_workers_main":
            cmd = _build_replay_command_main(
                ticks_path=tick_path,
                out_path=main_out_path,
                replay_cfg=replay_cfg,
            )
        else:
            cmd = _build_replay_command_groups(
                ticks_path=tick_path,
                workers=workers,
                out_dir=run_dir,
                replay_cfg=replay_cfg,
            )
        print(f"[INFO] ({idx}/{len(tick_files)}) replay {tick_path.name}")
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=max(30, int(args.timeout_sec)),
            check=False,
        )
        commands_log.append(
            {
                "tick_file": str(tick_path),
                "run_dir": str(run_dir),
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-8000:],
                "stderr_tail": (proc.stderr or "")[-8000:],
            }
        )
        if proc.returncode != 0:
            print(f"[ERROR] replay failed for {tick_path.name}", file=sys.stderr)
            (out_root / "commands.json").write_text(
                json.dumps(_sanitize_json(commands_log), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return proc.returncode

        if backend == "exit_workers_main":
            trades_by_file[tick_path.name] = _load_worker_trades_main(
                out_path=main_out_path,
                workers=workers,
                exclude_end_of_replay=exclude_end,
            )
        else:
            trades_by_file[tick_path.name] = _load_worker_trades(
                run_dir=run_dir,
                workers=workers,
                exclude_end_of_replay=exclude_end,
            )

    (out_root / "commands.json").write_text(
        json.dumps(_sanitize_json(commands_log), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gates_cfg = config.get("gates") if isinstance(config.get("gates"), dict) else {}
    default_gate = gates_cfg.get("default") if isinstance(gates_cfg.get("default"), dict) else {}
    worker_gates = gates_cfg.get("workers") if isinstance(gates_cfg.get("workers"), dict) else {}

    worker_results: dict[str, Any] = {}
    failing_workers: list[str] = []

    for worker in workers:
        threshold = _build_threshold(
            base=default_gate,
            override=worker_gates.get(worker) if isinstance(worker_gates.get(worker), dict) else None,
        )
        fold_records: list[dict[str, Any]] = []

        for fold_idx, fold in enumerate(folds, start=1):
            train_trades: list[dict[str, Any]] = []
            test_trades: list[dict[str, Any]] = []
            for key in fold["train"]:
                train_trades.extend(trades_by_file.get(key, {}).get(worker, []))
            for key in fold["test"]:
                test_trades.extend(trades_by_file.get(key, {}).get(worker, []))

            train_metrics = compute_trade_metrics(train_trades)
            test_metrics = compute_trade_metrics(test_trades)
            gate = evaluate_fold_gate(train_metrics, test_metrics, threshold)

            fold_records.append(
                {
                    "fold_id": fold_idx,
                    "train_files": list(fold["train"]),
                    "test_files": list(fold["test"]),
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "gate": gate,
                }
            )

        summary = summarize_worker_folds(fold_records, min_fold_pass_rate=min_fold_pass_rate)
        worker_results[worker] = {
            "threshold": threshold.__dict__,
            "summary": summary,
            "folds": fold_records,
        }
        if summary.get("status") != "pass":
            failing_workers.append(worker)

    overall_status = "pass" if not failing_workers else "fail"
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(args.config),
            "ticks_glob": ticks_glob,
            "backend": backend,
            "tick_file_count": len(tick_files),
            "workers": workers,
            "fold_count": len(folds),
            "train_files": train_files,
            "test_files": test_files,
            "step_files": step_files,
            "min_fold_pass_rate": min_fold_pass_rate,
            "out_root": str(out_root),
        },
        "overall": {
            "status": overall_status,
            "failing_workers": failing_workers,
            "passing_workers": [w for w in workers if w not in set(failing_workers)],
        },
        "worker_results": worker_results,
    }

    json_path = out_root / "quality_gate_report.json"
    md_path = out_root / "quality_gate_report.md"
    json_path.write_text(json.dumps(_sanitize_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(str(json_path))
    if args.strict and overall_status != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
