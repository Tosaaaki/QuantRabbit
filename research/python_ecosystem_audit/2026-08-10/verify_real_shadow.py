#!/usr/bin/env python3
"""Stdlib-only independent readback for the real-cohort adapter shadow."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ANCHOR = "2026-07-09T07:46:03.151624347Z"
WINDOWS = (
    ("INITIAL_16D", "2026-06-23T07:46:03.151624347Z", ANCHOR),
    ("DOUBLE_32D", "2026-06-07T07:46:03.151624347Z", ANCHOR),
    ("QUADRUPLE_64D", "2026-05-06T07:46:03.151624347Z", ANCHOR),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(value: str) -> datetime:
    normalized = str(value).replace("Z", "+00:00")
    normalized = re.sub(r"(\.\d{6})\d+([+-])", r"\1\2", normalized)
    parsed = datetime.fromisoformat(normalized)
    return parsed.astimezone(timezone.utc)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def drawdown(values: list[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def pf(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else None


def verify() -> dict[str, Any]:
    prereg = json.loads((HERE / "preregister_real_shadow_v1.json").read_text(encoding="utf-8"))
    report = json.loads((HERE / "real_shadow_report.json").read_text(encoding="utf-8"))
    adapters = json.loads((HERE / "real_adapter_report.json").read_text(encoding="utf-8"))
    payload = json.loads((HERE / "real_shadow_payload.json").read_text(encoding="utf-8"))
    selection = json.loads((REPO / "research/historical_learning_selection_rca/selection_rca_report_v1.json").read_text(encoding="utf-8"))
    selection_windows = {row["id"]: row for row in selection["windows"]}
    episodes = [row for row in rows(REPO / prereg["frozen_inputs"]["episodes_path"]) if row.get("label_status") == "ACTUAL_AFTER_COST"]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed})

    for key, value in prereg["frozen_inputs"].items():
        if key.endswith("_path") and key.replace("_path", "_sha256") in prereg["frozen_inputs"]:
            expected = prereg["frozen_inputs"][key.replace("_path", "_sha256")]
            check(f"hash:{key}", sha(REPO / value) == expected, sha(REPO / value))
    check("labeled_episodes_251", len(episodes) == 251, len(episodes))
    for window, start_text, end_text in WINDOWS:
        scoped = sorted(
            [row for row in episodes if parse_time(start_text) <= parse_time(row["feature_at_utc"]) <= parse_time(end_text)],
            key=lambda row: parse_time(row["feature_at_utc"]),
        )
        cut = max(1, math.floor(len(scoped) * 0.60))
        validation = scoped[cut:]
        start = parse_time(validation[0]["feature_at_utc"])
        train = [row for row in scoped[:cut] if parse_time(row["close_at_utc"]) < start - timedelta(hours=1)]
        purged = cut - len(train)
        frozen = selection_windows[window]
        values = [float(row["net_jpy"]) for row in validation]
        check(f"{window}:split", (len(train), len(validation), purged) == (frozen["train_events"], frozen["validation_events"], frozen["purged_train_events"]), [len(train), len(validation), purged])
        check(f"{window}:net", abs(sum(values) - float(frozen["ALL_TRADES"]["net_jpy"])) < 1e-9, sum(values))
        check(f"{window}:pf", abs(float(pf(values)) - float(frozen["ALL_TRADES"]["profit_factor"])) < 1e-12, pf(values))
        check(f"{window}:dd", abs(drawdown(values) - float(frozen["ALL_TRADES"]["max_drawdown_jpy"])) < 1e-9, drawdown(values))
    long_rows = payload["long_rows"]
    dims = payload["cube_axes"] + ["metric"]
    keys = [tuple(row[name] for name in dims) for row in long_rows]
    check("long_keys_unique", len(keys) == len(set(keys)), len(keys))
    check("long_missing_preserved", any(row["value"] is None for row in long_rows), sum(row["value"] is None for row in long_rows))
    check("financial_exact", payload["financial_invariants"]["exact_with_tolerance_1e_9"] is True)
    check("holdout_unread", payload["holdout_read"] is False and report["holdout_read"] is False)
    check("no_external_side_effect", report["live_paper_broker_order_deploy_touched"] is False)
    check("adapter_set", set(adapters) == {"xarray", "salib", "pymoo", "mapie"}, sorted(adapters))
    check("adapter_oracles", all(row["financial_oracle_unchanged"] for row in adapters.values()))
    check("adapter_repeat", all(row["adapter"]["deterministic_repeat"] for row in adapters.values()))
    check("zero_attributed_increment", report["profitability_increment_jpy_attributed_to_adapters"] == 0.0)
    check("dowhy_held", report["decisions"]["dowhy"]["decision"] == "HOLD_UNCHANGED_NOT_RUN")
    check("river_held", report["decisions"]["river"]["decision"] == "HOLD_UNCHANGED_NOT_RUN")
    for pair, source in report["source_lineage"]["sources"].items():
        check(f"lineage:{pair}", sha(REPO / source["path"]) == source["sha256"], source["sha256"])
    failed = [row["name"] for row in checks if not row["passed"]]
    return {
        "contract": "python_ecosystem_real_shadow_independent_oracle_v1",
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "holdout_read": False,
        "stdlib_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    result = verify()
    if args.capture:
        (HERE / "independent_oracle_real_shadow.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
