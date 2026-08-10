#!/usr/bin/env python3
"""Independent arithmetic readback for frozen V2 labels and growth artifacts."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> None:
    payload = json.loads((ROOT / "research/python_ecosystem_audit/2026-08-10/real_shadow_payload.json").read_text())
    labels = {
        row["episode_id"]: row
        for row in map(json.loads, (ROOT / "research/financial_oracle_v2/2026-08-10/trade_cashflows_v2.jsonl").open())
    }
    membership = [
        row for row in payload["episode_records"]
        if row["method"] == "ALL_TRADES"
        and row["window"] == "QUADRUPLE_64D"
        and row["split"] == "VALIDATION"
    ]
    corrected = sum(float(labels[row["episode_id"]]["corrected_net_jpy"]) for row in membership)
    checks = {
        "v2_64d_validation_count_101": len(membership) == 101,
        "v2_64d_validation_net_11706_0523": abs(corrected - 11706.0523) < 1e-6,
        "monthly_3x_fixed_200_trade_expectancy_2000": abs((600000.0 - 200000.0) / 200 - 2000.0) < 1e-12,
        "monthly_3x_compound_identity": abs((1 + (3 ** (1 / 200) - 1)) ** 200 - 3) < 1e-10,
    }
    grid = [json.loads(line) for line in (HERE / "growth_grid_v1.jsonl").read_text().splitlines()]
    checks["all_growth_rows_conserve_cashflow"] = all(
        abs(row["ending_equity_jpy"] - 200000.0 - row["after_cost_net_jpy"]) < 1e-6
        for row in grid
    )
    checks["no_validation_success_without_train_plateau"] = all(
        not row["validation_success"] or row["validation_admission_candidate"]
        for row in grid
    )
    result = {
        "contract": "MONTHLY_3X_GROWTH_ENGINE_INDEPENDENT_ORACLE_V1",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "corrected_64d_validation_net_jpy": corrected,
        "source_membership_sha256": hashlib.sha256(json.dumps(sorted(row["episode_id"] for row in membership)).encode()).hexdigest(),
    }
    (HERE / "independent_oracle_v1.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
