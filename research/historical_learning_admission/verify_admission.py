#!/usr/bin/env python3
"""Independent standard-library oracle for reported selected-return metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def parse_time(value: str):
    from datetime import datetime
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    if "." in text:
        head, tail = text.split(".", 1)
        digits, zone = tail.split("+", 1)
        text = head + "." + digits[:6] + "+" + zone
    return datetime.fromisoformat(text)


def pf(values: list[float]) -> float | str | None:
    gain = sum(v for v in values if v > 0)
    loss = -sum(v for v in values if v < 0)
    return gain / loss if loss else "Infinity" if gain else None


def dd(values: list[float]) -> float:
    equity = high = worst = 0.0
    for value in values:
        equity += value
        high = max(high, equity)
        worst = max(worst, high - equity)
    return worst


def equal(a: Any, b: Any) -> bool:
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    return abs(float(a) - float(b)) <= 1e-7 * max(1.0, abs(float(a)), abs(float(b)))


def verify(report: dict[str, Any], episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for window in report["windows"]:
        start, end = parse_time(window["from_utc"]), parse_time(window["to_utc"])
        available = [
            row for row in episodes.values()
            if row["label_status"] == "ACTUAL_AFTER_COST" and start <= parse_time(row["feature_at_utc"]) <= end
        ]
        available.sort(key=lambda row: parse_time(row["feature_at_utc"]))
        cut = max(1, math.floor(len(available) * .60))
        validation = available[cut:]
        baseline = {row["episode_id"]: float(row["net_jpy"]) for row in validation}
        for model, metrics in window["models"].items():
            if "selected_episode_ids" not in metrics:
                continue
            chosen = set(metrics["selected_episode_ids"])
            values = [value if episode_id in chosen else 0.0 for episode_id, value in baseline.items()]
            computed = {"trades_selected": len(chosen), "net_jpy": sum(values), "profit_factor": pf(values), "max_drawdown_jpy": dd(values), "expectancy_per_available_episode_jpy": mean(values) if values else None}
            fields = {name: equal(value, metrics[name]) for name, value in computed.items()}
            checks.append({"window": window["id"], "model": model, "passed": all(fields.values()), "fields": fields})
    return {"contract": "historical_learning_independent_oracle_v1", "checks": len(checks), "failed": sum(not c["passed"] for c in checks), "passed": all(c["passed"] for c in checks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    episodes = {row["episode_id"]: row for row in (json.loads(line) for line in args.episodes.read_text().splitlines() if line)}
    result = verify(report, episodes)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": result["checks"], "failed": result["failed"], "passed": result["passed"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
