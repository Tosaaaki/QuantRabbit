#!/usr/bin/env python3
"""Independent arithmetic oracle over emitted prediction rows."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def main() -> None:
    report = json.loads((ROOT / "selection_rca_report_v1.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(ROOT / "selection_predictions_v1.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["window_id"]].append(row)
    checks = []
    for window in report["windows"]:
        if window["status"] != "EVALUATED":
            continue
        rows = grouped[window["id"]]
        baseline = sum(float(row["actual_net_jpy"]) for row in rows)
        checks.append((f'{window["id"]}:baseline', close(baseline, float(window["ALL_TRADES"]["net_jpy"]))))
        definitions = {
            "FROZEN_HGB": "frozen_hgb_selected",
            "A_COVERAGE_BINDING": "coverage_binding_selected",
            "B_COST_AWARE_ABSTAIN": "cost_aware_selected",
            "C_PAIR_SIDE_CALIBRATION": "pair_side_calibration_selected",
        }
        for candidate, field in definitions.items():
            selected_net = sum(float(row["actual_net_jpy"]) for row in rows if row[field])
            expected = float(window["candidates"][candidate]["net_jpy"])
            checks.append((f'{window["id"]}:{candidate}:net', close(selected_net, expected)))
            checks.append((f'{window["id"]}:{candidate}:incremental', close(selected_net - baseline, float(window["candidates"][candidate]["incremental_net_jpy"]))))
        hgb = window["candidates"]["FROZEN_HGB"]["attribution"]
        missed = sum(float(row["actual_net_jpy"]) for row in rows if not row["frozen_hgb_selected"] and float(row["actual_net_jpy"]) > 0)
        avoided = -sum(float(row["actual_net_jpy"]) for row in rows if not row["frozen_hgb_selected"] and float(row["actual_net_jpy"]) < 0)
        checks.append((f'{window["id"]}:hgb:missed_winners', close(missed, float(hgb["false_negative_winner_jpy"]))))
        checks.append((f'{window["id"]}:hgb:avoided_losers', close(avoided, float(hgb["avoided_loser_magnitude_jpy"]))))
        checks.append((f'{window["id"]}:hgb:identity', close(avoided - missed, float(window["candidates"]["FROZEN_HGB"]["incremental_net_jpy"]))))
    failed = [name for name, passed in checks if not passed]
    output = {"contract": "selection_rca_independent_oracle_v1", "checks": len(checks), "passed": len(checks) - len(failed), "failed": failed}
    (ROOT / "independent_oracle_v1.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
