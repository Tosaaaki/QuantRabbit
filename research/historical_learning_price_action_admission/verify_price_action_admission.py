#!/usr/bin/env python3
"""Independent fail-closed readback for the S5 feature-admission result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    prereg = json.loads((ROOT / "preregister_v1.json").read_text())
    report = json.loads((ROOT / "report_v1.json").read_text())
    checks = []
    checks.append(("holdout_sealed", report["holdout_used"] is False))
    checks.append(("coverage_partition", sum(report["feature_coverage"]["reasons"].values()) == report["labeled_episodes"]))
    checks.append(("zero_features", report["feature_coverage"]["available"] == 0))
    checks.append(("decision_reject", report["overall_decision"] == "REJECT"))
    checks.append(("sweep_not_opened", report["multidimensional_sweep"] == "NOT_OPENED_FIXED_FEATURE_ADMISSION_FAILED"))
    checks.append(("all_windows_fail_closed", all(window["status"] == "NOT_FIT_ADMISSION_GATE" for window in report["windows"])))
    for source in prereg["s5_bidask_inputs"]:
        checks.append((f'hash:{source["pair"]}', sha256(REPO / source["path"]) == source["sha256"]))
        checks.append((f'invalid_rows:{source["pair"]}', report["s5_sources"][source["pair"]]["invalid_rows"] == 0))
        checks.append((f'gaps_observed:{source["pair"]}', report["s5_sources"][source["pair"]]["non_5s_gaps"] > 0))
    failed = [name for name, passed in checks if not passed]
    output = {"contract": "price_action_admission_independent_oracle_v1", "checks": len(checks), "passed": len(checks) - len(failed), "failed": failed}
    (ROOT / "independent_oracle_v1.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
