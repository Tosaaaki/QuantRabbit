#!/usr/bin/env python3
"""Evaluate JPY/hour gate thresholds from an existing replay quality report.

This script reuses fold metrics in quality_gate_report.json and performs:
  - threshold sweep for min_test_jpy_per_hour
  - +target_jpy_per_hour feasibility back-calculation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out):
        return float(default)
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _sanitize(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    return value


def _parse_thresholds(raw: str) -> list[float]:
    out: list[float] = []
    for token in str(raw or "").split(","):
        text = token.strip()
        if not text:
            continue
        try:
            out.append(float(text))
        except Exception:
            continue
    deduped: list[float] = []
    for value in out:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _resolve_worker(report: dict[str, Any], worker_arg: str | None) -> str:
    worker_results = report.get("worker_results")
    if not isinstance(worker_results, dict) or not worker_results:
        raise ValueError("worker_results is missing or empty in report")
    if worker_arg:
        key = str(worker_arg).strip()
        if key not in worker_results:
            raise ValueError(f"worker not found in report: {key}")
        return key
    return str(next(iter(worker_results.keys())))


def _gate_ok(
    fold: dict[str, Any],
    *,
    min_test_jpy_per_hour: float,
) -> bool:
    gate = fold.get("gate") if isinstance(fold.get("gate"), dict) else {}
    checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
    for item in checks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name == "test_jpy_per_hour":
            actual = _safe_float(item.get("actual"), 0.0)
            if actual < min_test_jpy_per_hour:
                return False
            continue
        if not bool(item.get("ok")):
            return False
    return True


def _calc_target_feasibility(folds: list[dict[str, Any]], target_jpy_per_hour: float) -> dict[str, Any]:
    total_jpy = 0.0
    total_hours = 0.0
    total_trades = 0.0
    per_fold: list[dict[str, Any]] = []

    for row in folds:
        tm = row.get("test_metrics") if isinstance(row.get("test_metrics"), dict) else {}
        fold_id = _safe_int(row.get("fold_id"), 0)
        test_files = row.get("test_files") if isinstance(row.get("test_files"), list) else []
        test_file = str(test_files[0]) if test_files else ""
        trade_count = _safe_float(tm.get("trade_count"), 0.0)
        duration_hours = _safe_float(tm.get("duration_hours"), 0.0)
        total_jpy_fold = _safe_float(tm.get("total_jpy"), 0.0)
        avg_jpy = _safe_float(tm.get("avg_jpy"), 0.0)
        jpy_per_hour = _safe_float(tm.get("jpy_per_hour"), 0.0)
        trades_per_hour = (trade_count / duration_hours) if duration_hours > 0.0 else 0.0
        required_jpy_per_trade = (target_jpy_per_hour / trades_per_hour) if trades_per_hour > 0.0 else float("inf")
        uplift_vs_avg_jpy = (
            (required_jpy_per_trade / avg_jpy)
            if avg_jpy > 0.0 and not math.isinf(required_jpy_per_trade)
            else float("inf")
        )
        per_fold.append(
            {
                "fold_id": fold_id,
                "test_file": test_file,
                "trade_count": trade_count,
                "duration_hours": duration_hours,
                "total_jpy": total_jpy_fold,
                "jpy_per_hour": jpy_per_hour,
                "trades_per_hour": trades_per_hour,
                "avg_jpy_per_trade": avg_jpy,
                "required_jpy_per_trade_for_target": required_jpy_per_trade,
                "uplift_vs_current_avg_jpy": uplift_vs_avg_jpy,
            }
        )
        total_jpy += total_jpy_fold
        total_hours += max(0.0, duration_hours)
        total_trades += max(0.0, trade_count)

    achieved_jpy_per_hour = (total_jpy / total_hours) if total_hours > 0.0 else 0.0
    achieved_jpy_per_trade = (total_jpy / total_trades) if total_trades > 0.0 else 0.0
    achieved_trades_per_hour = (total_trades / total_hours) if total_hours > 0.0 else 0.0
    required_trades_per_hour = (
        (target_jpy_per_hour / achieved_jpy_per_trade)
        if achieved_jpy_per_trade > 0.0
        else float("inf")
    )
    required_jpy_per_trade = (
        (target_jpy_per_hour / achieved_trades_per_hour)
        if achieved_trades_per_hour > 0.0
        else float("inf")
    )

    return {
        "target_jpy_per_hour": float(target_jpy_per_hour),
        "aggregate": {
            "total_jpy": total_jpy,
            "total_hours": total_hours,
            "total_trades": total_trades,
            "achieved_jpy_per_hour": achieved_jpy_per_hour,
            "achieved_trades_per_hour": achieved_trades_per_hour,
            "achieved_jpy_per_trade": achieved_jpy_per_trade,
            "required_trades_per_hour_at_current_ev": required_trades_per_hour,
            "required_jpy_per_trade_at_current_freq": required_jpy_per_trade,
            "hourly_gap_to_target": target_jpy_per_hour - achieved_jpy_per_hour,
        },
        "per_fold": per_fold,
    }


def _render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Replay JPY/Hour Sweep")
    lines.append("")
    lines.append("## Target Feasibility")
    tf = result.get("target_feasibility") if isinstance(result.get("target_feasibility"), dict) else {}
    agg = tf.get("aggregate") if isinstance(tf.get("aggregate"), dict) else {}
    lines.append(f"- target_jpy_per_hour: {tf.get('target_jpy_per_hour')}")
    lines.append(f"- achieved_jpy_per_hour: {agg.get('achieved_jpy_per_hour')}")
    lines.append(f"- achieved_trades_per_hour: {agg.get('achieved_trades_per_hour')}")
    lines.append(f"- achieved_jpy_per_trade: {agg.get('achieved_jpy_per_trade')}")
    lines.append(f"- required_trades_per_hour_at_current_ev: {agg.get('required_trades_per_hour_at_current_ev')}")
    lines.append(f"- required_jpy_per_trade_at_current_freq: {agg.get('required_jpy_per_trade_at_current_freq')}")
    lines.append("")
    lines.append("## Threshold Sweep")
    lines.append("| min_test_jpy_per_hour | folds | passed | pass_rate | status |")
    lines.append("|---:|---:|---:|---:|---|")
    for row in result.get("sweep", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {thr:.2f} | {folds} | {passed} | {pass_rate:.3f} | {status} |".format(
                thr=_safe_float(row.get("min_test_jpy_per_hour"), 0.0),
                folds=_safe_int(row.get("folds"), 0),
                passed=_safe_int(row.get("passed_folds"), 0),
                pass_rate=_safe_float(row.get("pass_rate"), 0.0),
                status=str(row.get("status") or "fail"),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep min_test_jpy_per_hour on existing replay report")
    ap.add_argument("--report", type=Path, required=True, help="quality_gate_report.json path")
    ap.add_argument("--worker", default="", help="worker key (default: first in report)")
    ap.add_argument(
        "--thresholds",
        default="150,300,500,2000",
        help="CSV of min_test_jpy_per_hour thresholds",
    )
    ap.add_argument("--target-jpy-per-hour", type=float, default=2000.0)
    ap.add_argument("--required-pass-rate", type=float, default=1.0)
    ap.add_argument("--out-json", type=Path, default=Path("tmp/replay_jpy_hour_sweep/result.json"))
    ap.add_argument("--out-md", type=Path, default=Path("tmp/replay_jpy_hour_sweep/result.md"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.report.exists():
        raise SystemExit(f"report not found: {args.report}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise SystemExit("invalid report payload")
    worker = _resolve_worker(report, args.worker)
    worker_result = report.get("worker_results", {}).get(worker)
    if not isinstance(worker_result, dict):
        raise SystemExit(f"worker result missing: {worker}")
    folds = worker_result.get("folds")
    if not isinstance(folds, list) or not folds:
        raise SystemExit(f"worker folds missing: {worker}")

    thresholds = _parse_thresholds(args.thresholds)
    if not thresholds:
        raise SystemExit("no valid thresholds")

    sweep_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        total = len(folds)
        passed = 0
        failed_fold_ids: list[int] = []
        for fold in folds:
            if not isinstance(fold, dict):
                continue
            if _gate_ok(fold, min_test_jpy_per_hour=threshold):
                passed += 1
            else:
                failed_fold_ids.append(_safe_int(fold.get("fold_id"), 0))
        pass_rate = (passed / total) if total > 0 else 0.0
        sweep_rows.append(
            {
                "min_test_jpy_per_hour": float(threshold),
                "folds": int(total),
                "passed_folds": int(passed),
                "pass_rate": float(pass_rate),
                "required_pass_rate": float(args.required_pass_rate),
                "status": "pass" if total > 0 and pass_rate >= float(args.required_pass_rate) else "fail",
                "failed_fold_ids": failed_fold_ids,
            }
        )

    target_feasibility = _calc_target_feasibility(folds, float(args.target_jpy_per_hour))
    result = {
        "source_report": str(args.report),
        "worker": worker,
        "required_pass_rate": float(args.required_pass_rate),
        "sweep": sweep_rows,
        "target_feasibility": target_feasibility,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(_sanitize(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(result), encoding="utf-8")
    print(str(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

