#!/usr/bin/env python3
"""Independent readback of row-level operator-alpha receipts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path


getcontext().prec = 28
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "independent_oracle_v1.json"


def _load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _canonical_hash(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _max_dd(values: list[Decimal]) -> Decimal:
    equity = peak = maximum = Decimal("0")
    for value in values:
        equity += value
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return maximum


def _metrics(rows: list[dict]) -> dict:
    pnls = [Decimal(str(row["after_cost_net_jpy"])) for row in rows]
    selected = [row for row in rows if row["selected"]]
    selected_pnls = [Decimal(str(row["after_cost_net_jpy"])) for row in selected]
    profit = sum((max(value, Decimal("0")) for value in selected_pnls), Decimal("0"))
    loss = -sum((min(value, Decimal("0")) for value in selected_pnls), Decimal("0"))
    return {
        "net": sum(pnls, Decimal("0")),
        "pf": profit / loss if loss else None,
        "dd": _max_dd(pnls),
        "selected": len(selected),
        "turnover": sum(int(row["units"]) for row in selected),
        "max_hold": max((Decimal(str(row["holding_seconds"])) for row in selected), default=Decimal("0")),
    }


def verify() -> dict:
    report = _load("comparison_report_v1.json")
    reconstruction = _load("trade_reconstruction_v1.json")
    fusion = _load("fusion_table_v1.json")
    contract = _load("acquisition_contract_v1.json")
    source_transactions = _load("source_transactions_v1.json")
    receipts = [json.loads(line) for line in (ROOT / "arm_receipts_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]
    checks: list[dict] = []

    def check(name: str, passed: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    expected_ids = [f"{a}->{b}" for a, b in contract["frozen_cohort"]["entry_close_fill_pairs"]]
    arms = ["BASELINE_ACTUAL", "OPERATOR_ALPHA", "X_STRUCTURE", "X_OPERATOR_INTERACTION"]
    check("exact_six_trade_reconstruction", [row["cohort_id"] for row in reconstruction["trades"]] == expected_ids, expected_ids)
    check("exact_four_arms", sorted({row["arm"] for row in receipts}) == sorted(arms), arms)
    check("exact_24_receipts", len(receipts) == 24, len(receipts))

    bad_hashes = []
    for row in receipts:
        receipt = dict(row)
        expected = receipt.pop("receipt_sha256")
        if _canonical_hash(receipt) != expected:
            bad_hashes.append(row["arm"] + ":" + row["cohort_id"])
    check("receipt_hashes", not bad_hashes, bad_hashes)

    oracle_metrics = {}
    for arm in arms:
        rows = [row for row in receipts if row["arm"] == arm]
        check(f"same_cohort_{arm}", [row["cohort_id"] for row in rows] == expected_ids, [row["cohort_id"] for row in rows])
        metric = _metrics(rows)
        expected = report["arms"][arm]["metrics"]
        comparisons = {
            "net": abs(metric["net"] - Decimal(str(expected["after_cost_net_jpy"]))) <= Decimal("0.000001"),
            "dd": abs(metric["dd"] - Decimal(str(expected["max_drawdown_jpy"]))) <= Decimal("0.000001"),
            "selected": metric["selected"] == expected["executed_or_retained"],
            "turnover": metric["turnover"] == expected["turnover_units"],
        }
        if metric["pf"] is None:
            comparisons["pf"] = expected["profit_factor"] is None
        else:
            comparisons["pf"] = abs(metric["pf"] - Decimal(str(expected["profit_factor"]))) <= Decimal("0.000001")
        check(f"metrics_{arm}", all(comparisons.values()), comparisons)
        oracle_metrics[arm] = {key: str(value) if isinstance(value, Decimal) else value for key, value in metric.items()}

    wins = [row for row in reconstruction["trades"] if row["label"].startswith("manual_win")]
    win_total = sum((Decimal(str(row["realized_after_cost_jpy"])) for row in wins), Decimal("0"))
    check("four_win_total", win_total == Decimal("5052.0833"), str(win_total))
    margin_losses = [Decimal(str(row["realized_after_cost_jpy"])) for row in reconstruction["trades"] if "margin_closeout" in row["label"]]
    check("two_margin_closeouts", margin_losses == [Decimal("-45720.0"), Decimal("-30480.0")], [str(x) for x in margin_losses])

    tx_by_id = {row["id"]: row for row in source_transactions["transactions"]}
    broken_order_links = []
    for trade in reconstruction["trades"]:
        for fill_id in (trade["entry_fill_id"], trade["close_fill_id"]):
            fill = tx_by_id.get(fill_id, {})
            order = tx_by_id.get(str(fill.get("orderID") or ""), {})
            if order.get("type") != "MARKET_ORDER" or order.get("time") != fill.get("time"):
                broken_order_links.append(fill_id)
    check("transaction_order_fill_timeline", not broken_order_links, broken_order_links)

    max_hold = Decimal(str(report["operator_parameters"]["max_hold_seconds"]))
    operator_rows = [row for row in receipts if row["arm"] == "OPERATOR_ALPHA"]
    causal_issues = []
    for row in operator_rows:
        entry = datetime.fromisoformat(row["entry_time_utc"].replace("Z", "+00:00"))
        exit_at = datetime.fromisoformat(row["exit_time_utc"].replace("Z", "+00:00"))
        elapsed = Decimal(str((exit_at - entry).total_seconds()))
        if elapsed <= 0 or elapsed > max_hold or abs(elapsed - Decimal(str(row["holding_seconds"]))) > Decimal("0.000001"):
            causal_issues.append(row["cohort_id"])
    check("operator_exits_post_entry_within_cap", not causal_issues, causal_issues)
    check(
        "operator_contains_observed_failure_shape",
        any(row["exit_reason"] == "LOSS_BUDGET" for row in operator_rows),
        [row["exit_reason"] for row in operator_rows],
    )
    check(
        "operator_allows_valid_profit_shape",
        any(row["exit_reason"] == "PROFIT_HARVEST" and row["after_cost_net_jpy"] > 0 for row in operator_rows),
        [(row["cohort_id"], row["exit_reason"], row["after_cost_net_jpy"]) for row in operator_rows],
    )
    check(
        "open_boundary_no_touch",
        reconstruction["open_boundary"]["action"] == "NO_TOUCH" and reconstruction["open_boundary"]["entry_fill_id"] == "473207",
        reconstruction["open_boundary"],
    )
    check(
        "live_fusion_waits",
        all(not row["live_permission"] and row["live_answer"] == "WAIT_EVIDENCE_INCOMPLETE" for row in fusion["rows"]),
        [row["live_answer"] for row in fusion["rows"]],
    )
    with (ROOT / "canonical_decision_table_v1.csv").open(encoding="utf-8", newline="") as handle:
        decision_rows = list(csv.DictReader(handle))
    check("six_state_cycles", len(decision_rows) == 36, len(decision_rows))
    check("decision_table_no_live_permission", all(row["live_permission"] == "false" for row in decision_rows), None)

    result = {
        "contract": "OPERATOR_ALPHA_INDEPENDENT_ORACLE_V1",
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "oracle_metrics": oracle_metrics,
        "artifact_readback_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "source_manifest_v1.json",
                "source_transactions_v1.json",
                "source_candles_v1.json",
                "trade_reconstruction_v1.json",
                "canonical_decision_table_v1.csv",
                "fusion_table_v1.json",
                "arm_receipts_v1.jsonl",
                "comparison_report_v1.json",
                "target_arithmetic_v1.json",
                "verdict_v1.md",
            )
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
