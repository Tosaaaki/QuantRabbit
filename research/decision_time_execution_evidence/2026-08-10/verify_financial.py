#!/usr/bin/env python3
"""Independent arithmetic oracle; does not import the ledger builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else None


def drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in values:
        equity += value
        if equity > peak:
            peak = equity
        if peak - equity > worst:
            worst = peak - equity
    return worst


def main() -> None:
    episodes = {
        row["episode_id"]: row
        for row in read_jsonl(REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl")
        if row.get("label_status") == "ACTUAL_AFTER_COST"
    }
    payload = json.loads((REPO / "research/python_ecosystem_audit/2026-08-10/real_shadow_payload.json").read_text(encoding="utf-8"))
    validation_ids = [
        row["episode_id"] for row in payload["episode_records"]
        if row["window"] == "QUADRUPLE_64D" and row["method"] == "ALL_TRADES" and row["split"] == "VALIDATION"
    ]
    values = [float(episodes[episode_id]["net_jpy"]) for episode_id in validation_ids]
    report = json.loads((HERE / "coverage_report_v1.json").read_text(encoding="utf-8"))
    prior = json.loads((REPO / "research/system_utilization_rca/2026-08-10/utilization_report_v1.json").read_text(encoding="utf-8"))
    net = sum(values)
    pf = profit_factor(values)
    dd = drawdown(values)
    prior_metrics = prior["fusion"]["QUADRUPLE_64D"]["candidates"]["calibrated_weighted_vote"]["validation"]
    checks = {
        "validation_count_101": len(values) == 101,
        "validation_net_matches_prior": abs(net - float(prior_metrics["all_trades_net_jpy"])) < 1e-9,
        "validation_dd_matches_prior": abs(dd - float(prior_metrics["all_trades_max_drawdown_jpy"])) < 1e-9,
        "strict_available_zero": report["evaluation"]["QUADRUPLE_64D"]["all_trades_same_eligible_cohort"]["available"] == 0,
        "strict_pf_unknown_not_zero": report["evaluation"]["QUADRUPLE_64D"]["all_trades_same_eligible_cohort"]["profit_factor"] is None,
        "strict_lcb_unknown_not_zero": report["evaluation"]["QUADRUPLE_64D"]["fusion_same_eligible_cohort"]["paired_bootstrap_lcb_jpy"] is None,
    }
    if not all(checks.values()):
        raise SystemExit(json.dumps(checks, sort_keys=True))
    output = {
        "contract": "DECISION_TIME_EXECUTION_EVIDENCE_FINANCIAL_ORACLE_V1",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "quadruple_64d_validation_all_trades": {
            "trades": len(values),
            "net_jpy": net,
            "profit_factor": pf,
            "max_drawdown_jpy": dd,
        },
        "strict_eligible": {
            "trades": 0,
            "net_jpy": None,
            "profit_factor": None,
            "max_drawdown_jpy": None,
            "paired_lcb_jpy": None,
            "margin_coverage": None,
            "status": "NOT_EVALUABLE"
        },
        "coverage_report_sha256": hashlib.sha256((HERE / "coverage_report_v1.json").read_bytes()).hexdigest(),
    }
    (HERE / "financial_oracle_v1.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
