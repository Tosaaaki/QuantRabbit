#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
prereg = json.loads((HERE / "preregister_v1.json").read_text())
report = json.loads((HERE / "report_v1.json").read_text())
refine = json.loads((HERE / "signal_quality_report_v2.json").read_text())
grid = [json.loads(line) for line in (HERE / "grid_v1.jsonl").open()]
quality_grid = [json.loads(line) for line in (HERE / "signal_quality_grid_v2.jsonl").open()]

source_hashes = {
    pair: hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
    for pair, source in prereg["sources"].items()
}
checks = {
    "source_hashes_match": all(source_hashes[pair] == source["sha256"] for pair, source in prereg["sources"].items()),
    "v1_grid_count": len(grid) == 5670 == report["grid_rows"],
    "v2_grid_count": len(quality_grid) == 2430 == refine["grid_rows"],
    "holdout_sealed": report["holdout_used"] is False and refine["holdout_used"] is False,
    "no_hidden_v1_validation_pass": not any(row["validation_pass"] for row in grid),
    "no_hidden_v2_validation_pass": not any(row["validation_pass"] for row in quality_grid),
    "monthly_target_not_mislabeled": report["monthly_3x_pass_count"] == 0 and report["conclusion"] == "MONTHLY_3X_NOT_PROVED",
    "refine_not_mislabeled": not refine["stable_multiwindow_candidates"] and refine["conclusion"] == "NO_STABLE_EDGE",
}
out = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_passed": all(checks.values())}
(HERE / "independent_oracle_v1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
if not out["all_passed"]:
    raise SystemExit(json.dumps(out, ensure_ascii=False))
