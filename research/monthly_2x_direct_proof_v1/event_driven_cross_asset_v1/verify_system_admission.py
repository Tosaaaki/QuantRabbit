#!/usr/bin/env python3
"""Independent fail-closed oracle for the System Admission artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    report_path = HERE / "SYSTEM_ADMISSION_V1.json"
    report = json.loads(report_path.read_text())
    calendar = json.loads((ROOT / "data/economic_calendar.json").read_text())
    context = json.loads((ROOT / "data/context_asset_charts.json").read_text())
    coverage = json.loads(
        (ROOT / "research/decision_time_execution_evidence/2026-08-10/coverage_report_v1.json").read_text()
    )

    events = calendar.get("events") or []
    required = {"SPX500_USD", "XAU_USD", "USB02Y_USD", "USB10Y_USD"}
    charts = {row.get("pair"): row for row in context.get("charts") or []}
    sided = all(
        view.get("recent_candles")
        and all("bid" in candle and "ask" in candle for candle in view["recent_candles"])
        for pair in required
        for view in (charts.get(pair) or {}).get("views") or []
    )
    stage = coverage["overall_stage_coverage"]
    checks = {
        "calendar_actuals_are_absent": sum(row.get("actual") is not None for row in events) == 0,
        "calendar_receipt_time_is_absent": all("provider_received_at_utc" not in row for row in events),
        "context_is_not_side_aware": sided is False,
        "strict_cost_coverage_is_zero": stage["slippage_fee_financing"] == 0,
        "strict_margin_coverage_is_zero": stage["margin_exposure_concurrency"] == 0,
        "strict_unwind_coverage_is_zero": stage["exit_unwind"] == 0,
        "report_stopped_before_replay": report["replay"]["started"] is False,
        "report_classifies_not_evaluable": report["classification"] == "NOT_EVALUABLE",
        "holdout_unread": report["inspection_boundary"]["holdout_read"] is False,
        "no_external_execution": report["inspection_boundary"]["live_paper_broker_order_deploy"] is False,
    }
    payload = {
        "oracle_id": "EVENT_DRIVEN_CROSS_ASSET_SYSTEM_ADMISSION_ORACLE_V1",
        "checks": checks,
        "all_pass": all(checks.values()),
        "verified_report_sha256": sha(report_path),
        "classification": "NOT_EVALUABLE" if all(checks.values()) else "ORACLE_CONFLICT",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check and args.check.read_text() != rendered:
        raise SystemExit(f"oracle regeneration mismatch: {args.check}")
    print(rendered, end="")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
