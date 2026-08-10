#!/usr/bin/env python3
"""Independent arithmetic readback for generated NO_FIXED_SL artifacts.

This intentionally does not import the replay implementation.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parent
START_EQUITY = 254_209.0185
TOLERANCE = 1e-6


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=TOLERANCE)


def main() -> int:
    comparison = json.loads((ROOT / "comparison_v1.json").read_text(encoding="utf-8"))
    cohort = json.loads((ROOT / "frozen_cohort_v1.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (ROOT / "decision_results_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identity_failures: list[str] = []
    for row in rows:
        groups[f"{row['arm']}::{row['account_mode']}"].append(row)
        if row.get("executed"):
            parts = [
                float(row.get("entry_cash_delta_jpy") or 0.0),
                float(row.get("original_realized_jpy") or 0.0),
                float(row.get("hedge_realized_jpy") or 0.0),
                float(row.get("original_mtm_jpy") or 0.0),
                float(row.get("hedge_mtm_jpy") or 0.0),
            ]
            if not close(math.fsum(parts), float(row["terminal_contribution_jpy"])):
                identity_failures.append(str(row["decision_id"]) + "::" + str(row["arm"]))
    expected_ids = {row["decision_id"] for row in cohort["decisions"]}
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for key, group in sorted(groups.items()):
        actual_ids = {row["decision_id"] for row in group}
        if actual_ids != expected_ids or len(group) != len(expected_ids):
            failures.append(f"{key}:decision_id_cohort")
        executed = [row for row in group if row.get("executed")]
        values = [float(row["terminal_contribution_jpy"]) for row in executed]
        gains = math.fsum(value for value in values if value > 0.0)
        losses = -math.fsum(value for value in values if value < 0.0)
        oracle = {
            "scheduled_decisions": len(group),
            "executed": len(executed),
            "after_cost_net_pre_financing_jpy": math.fsum(values),
            "after_cost_terminal_equity_pre_financing_jpy": START_EQUITY + math.fsum(values),
            "profit_factor_pre_financing": gains / losses if losses else None,
            "expectancy_pre_financing_jpy": fmean(values) if values else 0.0,
            "margin_closeout_count": sum(bool(row.get("margin_closeout")) for row in executed),
            "unresolved_inventory_count": sum(bool(row.get("original_open") or row.get("hedge_open")) for row in executed),
            "unknown_financing_count": sum(row.get("financing_jpy") is None for row in executed),
            "original_realized_jpy": math.fsum(float(row.get("original_realized_jpy") or 0.0) for row in executed),
            "hedge_realized_jpy": math.fsum(float(row.get("hedge_realized_jpy") or 0.0) for row in executed),
            "original_terminal_mtm_jpy": math.fsum(float(row.get("original_mtm_jpy") or 0.0) for row in executed),
            "hedge_terminal_mtm_jpy": math.fsum(float(row.get("hedge_mtm_jpy") or 0.0) for row in executed),
        }
        reported = comparison["comparisons"][key]
        for field, value in oracle.items():
            reported_value = reported[field]
            if isinstance(value, float):
                if reported_value is None or not close(value, float(reported_value)):
                    failures.append(f"{key}:{field}")
            elif value != reported_value:
                failures.append(f"{key}:{field}")
        checks[key] = oracle
    if identity_failures:
        failures.append("terminal_contribution_identity")
    result = {
        "contract": "NO_FORCED_LOSS_CLOSE_INDEPENDENT_ORACLE_V1",
        "pass": not failures,
        "groups": len(groups),
        "rows": len(rows),
        "cohort_decision_ids": len(expected_ids),
        "terminal_contribution_identity_failures": identity_failures,
        "failures": failures,
        "checks": checks,
    }
    (ROOT / "independent_oracle_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("pass", "groups", "rows", "cohort_decision_ids", "failures")}, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
