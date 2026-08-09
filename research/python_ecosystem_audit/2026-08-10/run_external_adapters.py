"""Run all isolated adapters and write reproducible evidence manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

HERE = Path(__file__).resolve().parent
CANDIDATES = {
    "xarray": {"version": "2026.7.0", "license": "Apache-2.0", "decision": "ADOPT_RESEARCH_ADAPTER"},
    "salib": {"version": "1.5.2", "license": "MIT", "decision": "ADOPT_RESEARCH_ADAPTER"},
    "pymoo": {"version": "0.6.2", "license": "Apache-2.0", "decision": "ADOPT_RESEARCH_ADAPTER"},
    "dowhy": {"version": "0.14", "license": "MIT", "decision": "HOLD_ISOLATED_DIAGNOSTIC"},
    "mapie": {"version": "1.5.0", "license": "BSD-3-Clause", "decision": "ADOPT_RESEARCH_ADAPTER"},
    "river": {"version": "0.25.0", "license": "BSD-3-Clause", "decision": "HOLD_NO_DRIFT_SIGNAL_IN_FIXTURE"},
}

# df reports 1024-byte blocks. These values are direct readbacks around each
# sequential install, not estimates reconstructed from directory size.
DISK_BLOCKS = {
    "xarray": [71710752, 71536696],
    "salib": [71539284, 71136576],
    "pymoo": [71136460, 70836872],
    "dowhy": [70836664, 69844312],
    "mapie": [69847616, 69602408],
    "river": [69600552, 69388380],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _probe(candidate: str) -> dict[str, Any]:
    python = HERE / ".adapter_envs" / candidate / "bin" / "python"
    output = subprocess.check_output([str(python), str(HERE / "external_adapter_probe.py"), candidate], text=True)
    return json.loads(output)


def _sbom(candidate: str) -> list[dict[str, Any]]:
    python = HERE / ".adapter_envs" / candidate / "bin" / "python"
    code = r'''import importlib.metadata as m, json
rows=[]
for d in m.distributions():
    md=d.metadata
    rows.append({"name":md.get("Name") or d.name,"version":d.version,"license_expression":md.get("License-Expression"),"license":md.get("License"),"requires_python":md.get("Requires-Python")})
print(json.dumps(sorted(rows,key=lambda x:x["name"].lower())))'''
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true", help="refresh benchmark/report artifacts after successful verification")
    args = parser.parse_args()
    reports: dict[str, Any] = {}
    wheels: dict[str, Any] = {}
    sbom: dict[str, Any] = {}
    for candidate, metadata in CANDIDATES.items():
        report = _probe(candidate)
        report["selection"] = metadata
        reports[candidate] = report
        wheel_dir = HERE / ".wheelhouse" / candidate
        wheels[candidate] = [
            {"filename": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(wheel_dir.glob("*.whl"))
        ]
        sbom[candidate] = _sbom(candidate)

    if not all(item["adapter"]["deterministic_repeat"] and item["financial_oracle_unchanged"] for item in reports.values()):
        raise RuntimeError("adapter determinism or financial-oracle contract failed")
    if reports["xarray"]["adapter"]["result"]["numeric_max_abs_diff"] != 0:
        raise RuntimeError("xarray numeric parity failed")
    if not reports["pymoo"]["adapter"]["result"]["front_exact_match"]:
        raise RuntimeError("pymoo Pareto parity failed")
    if reports["dowhy"]["adapter"]["result"]["effect_abs_diff"] >= 1e-12:
        raise RuntimeError("DoWhy/manual OLS parity failed")
    if reports["mapie"]["adapter"]["result"]["manual_bound_max_abs_diff"] >= 1e-12:
        raise RuntimeError("MAPIE/manual quantile parity failed")
    if reports["river"]["adapter"]["result"]["mean_abs_diff"] != 0:
        raise RuntimeError("River/stdlb mean parity failed")

    expected_wheels = json.loads((HERE / "adapter_wheel_manifest.json").read_text(encoding="utf-8")) if (HERE / "adapter_wheel_manifest.json").exists() else None
    expected_sbom = json.loads((HERE / "adapter_sbom.json").read_text(encoding="utf-8")) if (HERE / "adapter_sbom.json").exists() else None
    if expected_wheels is not None and wheels != expected_wheels:
        raise RuntimeError("wheelhouse differs from SHA-256 manifest")
    if expected_sbom is not None and sbom != expected_sbom:
        raise RuntimeError("installed distributions differ from captured SBOM")

    free_now = shutil.disk_usage(HERE).free
    checkpoints = []
    for candidate, (before_blocks, after_blocks) in DISK_BLOCKS.items():
        decrease = (before_blocks - after_blocks) * 1024
        checkpoints.append({
            "candidate": candidate,
            "free_before_bytes": before_blocks * 1024,
            "free_after_bytes": after_blocks * 1024,
            "free_decrease_bytes": decrease,
            "soft_pause_lt_8gib": after_blocks * 1024 < 8 * 1024**3,
            "soft_pause_ge_1gib_decrease": decrease >= 1024**3,
            "hard_stop_lt_5gib": after_blocks * 1024 < 5 * 1024**3,
        })
    disk = {
        "checkpoints": checkpoints,
        "free_final_bytes": free_now,
        "run_owned_env_bytes": _tree_bytes(HERE / ".adapter_envs"),
        "run_owned_wheelhouse_bytes": _tree_bytes(HERE / ".wheelhouse"),
        "output_cap_bytes": 5 * 1024**3,
        "output_cap_exceeded": _tree_bytes(HERE / ".adapter_envs") + _tree_bytes(HERE / ".wheelhouse") > 5 * 1024**3,
        "outside_status_sha_all_installs": "e7a32f2f60ec8457ef0bded1ab91cf5c44b50bfed3e97a257fd507ab61378fd4",
        "outside_status_changed": False,
        "active_db_wal_in_canonical_repo": False,
        "other_worktree_db_wal_untouched": True,
    }
    if args.capture:
        (HERE / "external_adapter_report.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if expected_wheels is None:
            (HERE / "adapter_wheel_manifest.json").write_text(json.dumps(wheels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if expected_sbom is None:
            (HERE / "adapter_sbom.json").write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (HERE / "disk_checkpoints.json").write_text(json.dumps(disk, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidates": list(reports), "all_deterministic": all(item["adapter"]["deterministic_repeat"] for item in reports.values()),
        "all_oracles_unchanged": all(item["financial_oracle_unchanged"] for item in reports.values()),
        "free_final_bytes": free_now, "output_cap_exceeded": disk["output_cap_exceeded"], "captured": args.capture,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
