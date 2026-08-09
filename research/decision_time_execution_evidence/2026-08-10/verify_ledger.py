#!/usr/bin/env python3
"""Independent, deliberately small oracle for the generated evidence ledger."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def utc(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        plus = suffix.find("+")
        fraction, offset = suffix[:plus], suffix[plus:]
        normalized = f"{prefix}.{fraction[:6]}{offset}"
    return datetime.fromisoformat(normalized).astimezone(timezone.utc)


def main() -> None:
    ledger = rows("evidence_ledger_v1.jsonl")
    decisions = rows("fused_decisions_rerun_v1.jsonl")
    report = json.loads((HERE / "coverage_report_v1.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    checks["unique_251"] = len(ledger) == len({row["decision_id"] for row in ledger}) == 251
    checks["candidate_251"] = sum(row["candidate_order"]["coverage"] for row in ledger) == 251
    checks["pricing_154"] = sum(row["pricing"]["coverage"] for row in ledger) == 154
    checks["fillability_153"] = sum(row["fillability"]["coverage"] for row in ledger) == 153
    checks["cost_complete_zero"] = sum(row["costs"]["coverage"] for row in ledger) == 0
    checks["margin_complete_zero"] = sum(row["portfolio_margin"]["coverage"] for row in ledger) == 0
    checks["unwind_complete_zero"] = sum(row["exit_unwind"]["coverage"] for row in ledger) == 0
    checks["strict_zero"] = sum(row["strict_eligible"] for row in ledger) == report["strict_eligible"] == 0
    checks["no_future"] = all(
        not row["pricing"]["coverage"] or utc(row["pricing"]["value"]["watermark"]) <= utc(row["decision_time"])
        for row in ledger
    )
    checks["bid_ask"] = all(
        not row["pricing"]["coverage"] or row["pricing"]["value"]["bid"] <= row["pricing"]["value"]["ask"]
        for row in ledger
    )
    checks["no_trade"] = Counter(row["action"] for row in decisions).get("TRADE", 0) == 0
    checks["holdout_unread"] = report["holdout_read"] is False and all(row["holdout_read"] is False for row in decisions)
    if not all(checks.values()):
        raise SystemExit(json.dumps({"checks": checks}, sort_keys=True))
    result = {
        "contract": "DECISION_TIME_EXECUTION_EVIDENCE_LEDGER_V1_INDEPENDENT_ORACLE",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "ledger_sha256": hashlib.sha256((HERE / "evidence_ledger_v1.jsonl").read_bytes()).hexdigest(),
    }
    (HERE / "independent_oracle_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
