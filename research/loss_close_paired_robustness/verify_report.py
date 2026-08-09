#!/usr/bin/env python3
"""Independent arithmetic readback for the robustness report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def profit_factor(values: list[float]) -> float | None:
    gains = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    if losses == 0:
        return math.inf if gains else None
    return gains / losses


def drawdown(values: list[float]) -> float:
    running = high = worst = 0.0
    for value in values:
        running += value
        high = max(high, running)
        worst = max(worst, high - running)
    return worst


def close(a: float | None, b: float | None, tolerance: float = 1e-7) -> bool:
    if a is None or b is None:
        return a is b
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def verify(report: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for window in report["windows"]:
        window_id = window["id"]
        for arm, split_reports in window["arms"].items():
            for split in ("TRAIN", "VALIDATION"):
                rows = [
                    row for row in report["events"]
                    if row.get("status") == "CALCULATED_DIAGNOSTIC"
                    and row.get("window_splits", {}).get(window_id) == split
                    and arm in row.get("arms", {})
                ]
                values = [float(row["arms"][arm]["net_jpy"]) for row in rows]
                expected = split_reports[split]
                computed = {
                    "trades": len(values), "net_jpy": sum(values),
                    "expectancy_jpy": mean(values) if values else None,
                    "profit_factor": profit_factor(values), "max_drawdown_jpy": drawdown(values),
                    "cost_breakdown_jpy": {
                        key: sum(float(row["arms"][arm]["cost_breakdown_jpy"][key]) for row in rows)
                        for key in ("intrinsic_spread_estimate", "explicit_fee", "slippage_stress")
                    },
                    "peak_broker_margin_jpy": max((float(row["arms"][arm]["peak_broker_margin_jpy"]) for row in rows), default=0.0),
                    "peak_double_gross_margin_jpy": max((float(row["arms"][arm]["peak_double_gross_margin_jpy"]) for row in rows), default=0.0),
                }
                fields_ok = {
                    "trades": computed["trades"] == expected["trades"],
                    "net_jpy": close(computed["net_jpy"], expected["net_jpy"]),
                    "expectancy_jpy": close(computed["expectancy_jpy"], expected["expectancy_jpy"]),
                    "profit_factor": close(computed["profit_factor"], expected["profit_factor"]),
                    "max_drawdown_jpy": close(computed["max_drawdown_jpy"], expected["max_drawdown_jpy"]),
                    "peak_broker_margin_jpy": close(computed["peak_broker_margin_jpy"], expected["peak_broker_margin_jpy"]),
                    "peak_double_gross_margin_jpy": close(computed["peak_double_gross_margin_jpy"], expected["peak_double_gross_margin_jpy"]),
                    "cost_breakdown_jpy": all(close(value, expected["cost_breakdown_jpy"][key]) for key, value in computed["cost_breakdown_jpy"].items()),
                }
                checks.append({"window": window_id, "arm": arm, "split": split, "passed": all(fields_ok.values()), "fields": fields_ok})
    return {"contract": "loss_close_robustness_independent_readback_v1", "checks": len(checks), "failed": sum(not c["passed"] for c in checks), "passed": all(c["passed"] for c in checks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.report.read_text()))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": result["checks"], "failed": result["failed"], "passed": result["passed"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
