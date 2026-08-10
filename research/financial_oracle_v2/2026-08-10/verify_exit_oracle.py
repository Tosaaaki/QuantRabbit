#!/usr/bin/env python3
"""Independent aggregation oracle for exit replay outputs."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    rows = read_jsonl(HERE / "exit_replay_rows_v1.jsonl")
    cashflows = {row["episode_id"]: row for row in read_jsonl(HERE / "trade_cashflows_v2.jsonl")}
    payload = json.loads((REPO / "research/python_ecosystem_audit/2026-08-10/real_shadow_payload.json").read_text())
    memberships = {(row["window"], row["split"], row["episode_id"]) for row in payload["episode_records"] if row["episode_id"] in cashflows}
    report = json.loads((HERE / "exit_report_v1.json").read_text())
    checks: list[dict] = []
    baseline = [row for row in rows if row["exit_policy"] == "BASELINE"]
    checks.append({"check": "one_baseline_per_membership", "pass": len(baseline) == len(memberships)})
    for row in baseline:
        expected = float(cashflows[row["episode_id"]]["corrected_net_jpy"])
        checks.append({"check": f"baseline:{row['window']}:{row['split']}:{row['episode_id']}", "pass": row["candidate_actual_after_cost_net_jpy"] == expected})
    validation = [row for row in baseline if row["window"] == "QUADRUPLE_64D" and row["split"] == "VALIDATION"]
    checks.extend([
        {"check": "64d_validation_count", "pass": len(validation) == 101},
        {"check": "64d_validation_net", "pass": abs(sum(row["candidate_actual_after_cost_net_jpy"] for row in validation) - 11706.0523) < 1e-7},
        {"check": "changed_after_cost_null", "pass": all(row["candidate_actual_after_cost_net_jpy"] is None for row in rows if row["changed"] is True)},
        {"check": "not_evaluable_after_cost_null", "pass": all(row["candidate_actual_after_cost_net_jpy"] is None for row in rows if row["admission_status"].startswith("NOT_EVALUABLE"))},
        {"check": "strict_path_five", "pass": len({row["episode_id"] for row in rows if row["path_complete"]}) == 5},
        {"check": "holdout_unused", "pass": report["holdout_used"] is False},
    ])
    for window, splits in report["summary"].items():
        for split, arms in splits.items():
            for arm, saved in arms.items():
                selected = [row for row in rows if row["window"] == window and row["split"] == split and row["exit_policy"] == arm]
                checks.append({"check": f"count:{window}:{split}:{arm}", "pass": saved["episodes"] == len(selected) and saved["changed"] == sum(row["changed"] is True for row in selected)})
    failed = [row for row in checks if not row["pass"]]
    result = {"contract": "EXIT_POLICY_PAIRED_REPLAY_INDEPENDENT_ORACLE_V1", "status": "PASS" if not failed else "FAIL", "checks": len(checks), "passed": len(checks) - len(failed), "failed": failed, "holdout_used": False}
    (HERE / "exit_independent_oracle_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
