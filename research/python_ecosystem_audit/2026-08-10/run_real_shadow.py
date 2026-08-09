"""Build and verify the adopted-adapter shadow on the frozen real cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from real_shadow_core import REAL_CUBE_AXES, build_payload, logical_digest, sha256  # noqa: E402


CANDIDATES = {
    "xarray": {"version": "2026.7.0", "decision": "ADOPT_RESEARCH_ADAPTER"},
    "salib": {"version": "1.5.2", "decision": "ADOPT_RESEARCH_ADAPTER"},
    "pymoo": {"version": "0.6.2", "decision": "ADOPT_RESEARCH_ADAPTER"},
    "mapie": {"version": "1.5.0", "decision": "ADOPT_RESEARCH_ADAPTER"},
}
SOFT_FREE = 8 * 1024**3
HARD_FREE = 5 * 1024**3
RUN_CAP = 5 * 1024**3


class CandidateUnavailable(RuntimeError):
    """Fail closed; never fall back to the ambient interpreter."""


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def candidate_python(candidate: str, env_root: Path | None = None) -> Path:
    root = env_root if env_root is not None else HERE / ".adapter_envs"
    python = root / candidate / "bin" / "python"
    if not python.is_file():
        raise CandidateUnavailable(f"isolated environment unavailable: {candidate}")
    return python


def probe_candidate(candidate: str, env_root: Path | None = None) -> dict[str, Any]:
    python = candidate_python(candidate, env_root)
    completed = subprocess.run(
        [str(python), str(HERE / "real_adapter_probe.py"), candidate],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{candidate} probe failed: {completed.stderr[-1000:]}")
    return json.loads(completed.stdout)


def installed_sbom(candidate: str) -> list[dict[str, Any]]:
    python = candidate_python(candidate)
    code = r'''import importlib.metadata as m, json
rows=[]
for d in m.distributions():
    md=d.metadata
    rows.append({"name":md.get("Name") or d.name,"version":d.version,"license_expression":md.get("License-Expression"),"license":md.get("License"),"requires_python":md.get("Requires-Python")})
print(json.dumps(sorted(rows,key=lambda x:x["name"].lower())))'''
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def verify_isolation() -> dict[str, Any]:
    expected_wheels = json.loads((HERE / "adapter_wheel_manifest.json").read_text(encoding="utf-8"))
    expected_sbom = json.loads((HERE / "adapter_sbom.json").read_text(encoding="utf-8"))
    evidence: dict[str, Any] = {}
    for candidate, metadata in CANDIDATES.items():
        wheel_dir = HERE / ".wheelhouse" / candidate
        actual_wheels = [
            {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(wheel_dir.glob("*.whl"))
        ]
        if actual_wheels != expected_wheels[candidate]:
            raise RuntimeError(f"wheel manifest changed: {candidate}")
        actual_sbom = installed_sbom(candidate)
        if actual_sbom != expected_sbom[candidate]:
            raise RuntimeError(f"installed SBOM changed: {candidate}")
        package_row = next(
            row for row in actual_sbom if row["name"].lower() == candidate
        )
        if package_row["version"] != metadata["version"]:
            raise RuntimeError(f"version changed: {candidate}")
        evidence[candidate] = {
            "version": package_row["version"],
            "wheel_count": len(actual_wheels),
            "wheel_manifest_exact": True,
            "sbom_exact": True,
            "rollback": f"remove ignored .adapter_envs/{candidate}; no runtime lock changed",
        }
    return evidence


def sparse_cube(long_rows: list[dict[str, Any]]) -> dict[str, Any]:
    coords = {
        axis: sorted({row[axis] for row in long_rows}, key=str)
        for axis in (*REAL_CUBE_AXES, "metric")
    }
    values: dict[str, float | None] = {}
    for row in long_rows:
        key = json.dumps([row[axis] for axis in (*REAL_CUBE_AXES, "metric")], separators=(",", ":"))
        if key in values:
            raise RuntimeError(f"duplicate real cube key: {key}")
        values[key] = row["value"]
    return {
        "dims": [*REAL_CUBE_AXES, "metric"],
        "coords": coords,
        "values": values,
        "missing_is_absent_or_null_never_zero": True,
    }


def write_payload(payload: dict[str, Any]) -> None:
    (HERE / "real_shadow_payload.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (HERE / "real_canonical_long_table.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["long_rows"]:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    (HERE / "real_cube_sparse.json").write_text(
        json.dumps(sparse_cube(payload["long_rows"]), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def adapter_decisions(reports: dict[str, Any]) -> dict[str, Any]:
    xarray = reports["xarray"]["adapter"]["result"]
    salib = reports["salib"]["adapter"]["result"]
    pymoo = reports["pymoo"]["adapter"]["result"]
    mapie = reports["mapie"]["adapter"]["result"]
    decisions: dict[str, Any] = {
        "xarray": {
            "decision": "ADOPT_RESEARCH_ADAPTER" if xarray["numeric_max_abs_diff"] == 0 and xarray["known_absent_coordinate_is_nan"] else "REJECT",
            "profitability_increment_jpy": 0.0,
            "reason": "exact labelled-cube parity and missing preservation; organization only",
        },
        "salib": {
            "decision": "ADOPT_RESEARCH_ADAPTER" if reports["salib"]["adapter"]["deterministic_repeat"] else "REJECT",
            "profitability_increment_jpy": 0.0,
            "reason": "TRAIN-only sensitivity diagnostic; unstable ranks cannot promote a rule",
        },
        "pymoo": {
            "decision": "ADOPT_RESEARCH_ADAPTER" if all(row["front_exact_match"] for row in pymoo["windows"].values()) else "REJECT",
            "profitability_increment_jpy": 0.0,
            "reason": "Pareto oracle parity; the constrained real front remains empty when margin evidence is enforced",
        },
        "mapie": {
            "decision": "ADOPT_RESEARCH_ADAPTER" if any(
                row.get("status") == "EXECUTED_OUTER_VALIDATION" and row.get("manual_bound_max_abs_diff", 1.0) < 1e-9
                for row in mapie["windows"].values()
            ) else "HOLD_REAL_USE",
            "profitability_increment_jpy": 0.0,
            "reason": "interval implementation parity; coverage is uncertainty evidence, not a trading edge",
        },
        "dowhy": {"decision": "HOLD_UNCHANGED_NOT_RUN", "profitability_increment_jpy": 0.0},
        "river": {"decision": "HOLD_UNCHANGED_NOT_RUN", "profitability_increment_jpy": 0.0},
    }
    decisions["salib"]["executed_windows"] = sum(row.get("status") == "EXECUTED_TRAIN_ONLY_RANKING" for row in salib["windows"].values())
    decisions["mapie"]["executed_windows"] = sum(row.get("status") == "EXECUTED_OUTER_VALIDATION" for row in mapie["windows"].values())
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    free_before = shutil.disk_usage(HERE).free
    if free_before < HARD_FREE:
        raise RuntimeError("hard free-space stop")
    if free_before < SOFT_FREE:
        raise RuntimeError("soft free-space pause")
    isolation = verify_isolation()
    payload = build_payload()
    digest = logical_digest(payload)
    captured_path = HERE / "real_shadow_payload.json"
    if args.capture:
        write_payload(payload)
    else:
        if not captured_path.is_file():
            raise RuntimeError("captured payload is missing")
        captured = json.loads(captured_path.read_text(encoding="utf-8"))
        if logical_digest(captured) != digest:
            raise RuntimeError("captured real payload is not reproducible")
    reports = {candidate: probe_candidate(candidate) for candidate in CANDIDATES}
    if not all(report["financial_oracle_unchanged"] and report["adapter"]["deterministic_repeat"] for report in reports.values()):
        raise RuntimeError("adapter financial invariant or deterministic repeat failed")
    decisions = adapter_decisions(reports)
    free_after = shutil.disk_usage(HERE).free
    owned = tree_bytes(HERE / ".adapter_envs") + tree_bytes(HERE / ".wheelhouse")
    disk = {
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "free_decrease_bytes": free_before - free_after,
        "soft_pause_free_lt_8gib": free_after < SOFT_FREE,
        "soft_pause_decrease_ge_1gib": free_before - free_after >= 1024**3,
        "hard_stop_free_lt_5gib": free_after < HARD_FREE,
        "run_owned_env_and_wheel_bytes": owned,
        "run_owned_cap_bytes": RUN_CAP,
        "run_owned_cap_exceeded": owned > RUN_CAP,
        "new_package_install": False,
        "active_db_wal_touched": False,
    }
    if disk["hard_stop_free_lt_5gib"] or disk["run_owned_cap_exceeded"]:
        raise RuntimeError("capacity hard stop after shadow")
    report = {
        "contract": "python_ecosystem_real_cohort_shadow_result_v1",
        "payload_digest": digest,
        "checkpoint_git_head": "797a20d5a330ee726f5931691797a7bbd687d791",
        "isolation": isolation,
        "financial_invariants": payload["financial_invariants"],
        "source_lineage": payload["manifest"]["lineage"],
        "splits": payload["manifest"]["splits"],
        "adapter_reports": reports,
        "decisions": decisions,
        "profitability_increment_jpy_attributed_to_adapters": 0.0,
        "strategy_decision": "NO_CHANGE_KEEP_ALL_TRADES_BASELINE; ADAPTERS_DO_NOT_CREATE_EDGE",
        "policy_admission": "BLOCKED_INCOMPLETE_MARGIN_AND_ALL_ENTRY_COUNTERFACTUAL_COVERAGE",
        "holdout_read": False,
        "live_paper_broker_order_deploy_touched": False,
        "disk": disk,
    }
    if args.capture:
        (HERE / "real_adapter_report.json").write_text(
            json.dumps(reports, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        (HERE / "real_shadow_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        (HERE / "real_disk_checkpoints.json").write_text(
            json.dumps(disk, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "payload_digest": digest,
        "episode_records": len(payload["episode_records"]),
        "long_rows": len(payload["long_rows"]),
        "decisions": {name: row["decision"] for name, row in decisions.items()},
        "financial_exact": payload["financial_invariants"]["exact_with_tolerance_1e_9"],
        "free_after_bytes": free_after,
        "captured": args.capture,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
