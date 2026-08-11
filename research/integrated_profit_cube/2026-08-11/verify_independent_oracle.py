#!/usr/bin/env python3
"""Stdlib-only independent readback for INTEGRATED_PROFIT_CUBE_V1."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def close(actual: float, expected: float, tolerance: float = 1e-8) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def main() -> int:
    checks: list[dict] = []

    def check(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    prereg = json.loads((HERE / "preregister_v1.json").read_text())
    manifest = json.loads((HERE / "run_manifest_v1.json").read_text())
    report = json.loads((HERE / "report_v1.json").read_text())
    cube = read_jsonl(HERE / "canonical_decision_cube_v1.jsonl")
    grid = read_jsonl(HERE / "candidate_grid_v1.jsonl")

    for name, source in prereg["frozen_sources"].items():
        actual = sha256(ROOT / source["path"])
        check(f"source_hash:{name}", actual == source["sha256"], actual)
    for name, expected in manifest["outputs"].items():
        check(f"output_hash:{name}", sha256(HERE / name) == expected, expected)

    labels = read_jsonl(ROOT / prereg["frozen_sources"]["financial_labels_v2"]["path"])
    evidence = read_jsonl(ROOT / prereg["frozen_sources"]["execution_evidence"]["path"])
    label_by_id = {row["episode_id"]: row for row in labels}
    evidence_by_id = {row["decision_id"]: row for row in evidence}
    check("unique_v2_decisions", len(label_by_id) == len(labels) == 251, len(label_by_id))
    check("evidence_ids_match", set(label_by_id) == set(evidence_by_id), len(evidence_by_id))

    expected_counts = {
        ("INITIAL_16D", "TRAIN"): 13,
        ("INITIAL_16D", "VALIDATION"): 12,
        ("DOUBLE_32D", "TRAIN"): 43,
        ("DOUBLE_32D", "VALIDATION"): 31,
        ("QUADRUPLE_64D", "TRAIN"): 145,
        ("QUADRUPLE_64D", "VALIDATION"): 101,
    }
    observed = defaultdict(int)
    for row in evidence:
        for window, split in row["splits"].items():
            if split in {"TRAIN", "VALIDATION"}:
                observed[(window, split)] += 1
    serializable_observed = {f"{window}:{split}": count for (window, split), count in observed.items()}
    check("split_counts", dict(observed) == expected_counts, serializable_observed)

    validation_ids = sorted({
        row["decision_id"]
        for row in evidence
        if row["splits"]["QUADRUPLE_64D"] == "VALIDATION"
    })
    validation_values = [float(label_by_id[episode_id]["corrected_net_jpy"]) for episode_id in validation_ids]
    close(sum(validation_values), 11706.0523)
    gains = sum(value for value in validation_values if value > 0)
    losses = -sum(value for value in validation_values if value < 0)
    close(gains / losses, 1.4469329373747661)
    check("v2_64d_baseline", True, {"net": sum(validation_values), "pf": gains / losses})

    nonzero_financing = sum(abs(float(row["daily_financing_jpy"])) > 0 for row in labels)
    partial = sum(abs(float(row["partial_reduction_jpy"])) > 0 for row in labels)
    check("daily_financing_retained", nonzero_financing == 58, nonzero_financing)
    check("partial_reduction_retained", partial == 1, partial)

    check("cube_nulls_not_zero", all(row["missing_not_zero"] == (row["value"] is None) for row in cube), len(cube))
    incomplete_changed = [
        row
        for row in cube
        if row["stage"] in {"EXIT", "HEDGE"}
        and row["candidate_actual_after_cost_net_jpy"] is None
        and row["value"] is not None
    ]
    check("incomplete_changed_not_monetized", not incomplete_changed, len(incomplete_changed))
    check("holdout_unused", not report["holdout_used"] and all(not row["holdout_used"] for row in grid), report["holdout_used"])

    train_by_id = defaultdict(list)
    for row in grid:
        if row["split"] == "TRAIN":
            train_by_id[row["parameter_id"]].append(row)
    independently_selected = max(
        train_by_id,
        key=lambda key: (
            min(row["incremental_net_jpy"] for row in train_by_id[key]),
            sum(row["incremental_net_jpy"] for row in train_by_id[key]),
            -sum(row["realized_max_drawdown_jpy"] for row in train_by_id[key]),
            key,
        ),
    )
    check("champion_is_train_only", independently_selected == report["train_only_champion"], independently_selected)

    champion_validation = report["validation_evaluations"][independently_selected]
    primary = champion_validation["QUADRUPLE_64D"]
    close(primary["incremental_net_jpy"], 879.602592321733)
    close(primary["realized_max_drawdown_jpy"], 7724.000594137935)
    check(
        "champion_not_overclaimed",
        primary["paired_bootstrap_lcb_one_sided_95_jpy_per_episode"] <= 0
        and not primary["account_margin_evaluable"]
        and not report["strict_pass_candidates"],
        {
            "lcb": primary["paired_bootstrap_lcb_one_sided_95_jpy_per_episode"],
            "account_margin": primary["account_margin_evaluable"],
        },
    )
    check("xarray_preserved_nulls", report["xarray"]["null_preserved"], report["xarray"])
    check("library_consumers_ran", all(name in report for name in ("xarray", "salib", "pymoo", "mapie")), "all present")
    check("final_conclusion", report["conclusion"] == "BASELINE_POSITIVE_INTEGRATED_IMPROVEMENT_NOT_YET_ADMISSIBLE", report["conclusion"])

    result = {"contract": "INTEGRATED_PROFIT_CUBE_INDEPENDENT_ORACLE_V1", "passed": len(checks), "failed": 0, "checks": checks}
    (HERE / "independent_oracle_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": len(checks), "failed": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
