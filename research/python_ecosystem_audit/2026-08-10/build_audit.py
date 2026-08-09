"""Build the bounded long-table/cube proof and research-only adapter report."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from audit_core import (  # noqa: E402
    build_cube,
    drift_refutation,
    pairwise_interactions,
    pareto_front,
    placebo_refutation,
    records_to_long,
    simpson_candidates,
    source_sha,
)


def fixture_records() -> list[dict[str, object]]:
    """Causal toy cohort: no mid-price, one deliberate sparse missing cell."""
    fixture_digest = "fixture-sha-qr-causal-v1"
    records: list[dict[str, object]] = []
    counter = 0
    for split in ("TRAIN", "VALIDATION"):
        for regime in ("TREND", "RANGE"):
            for method in ("baseline", "cube_shadow"):
                for cost in ("observed_bid_ask", "stress_plus_1pip"):
                    for risk in ("margin_cap_92", "margin_cap_70"):
                        for exit_policy in ("SL", "TIMEOUT"):
                            # Deliberate absent cell: sparse means unknown, not zero.
                            if (split, regime, method, cost, risk, exit_policy) == (
                                "VALIDATION", "RANGE", "cube_shadow", "stress_plus_1pip", "margin_cap_70", "TIMEOUT"
                            ):
                                continue
                            pair = "EUR_USD" if regime == "TREND" else "AUD_JPY"
                            timeframe = "M5" if method == "baseline" else "M15"
                            for local in range(4):
                                counter += 1
                                decision = f"2026-01-{1 + (counter % 20):02d}T00:{counter % 60:02d}:00Z"
                                base = 22.0 if regime == "TREND" else -8.0
                                base += 4.0 if method == "cube_shadow" and regime == "TREND" else 0.0
                                base -= 3.0 if method == "cube_shadow" and regime == "RANGE" else 0.0
                                base -= 7.0 if cost == "stress_plus_1pip" else 0.0
                                base += 1.0 if exit_policy == "TIMEOUT" else 0.0
                                net = base + (local - 1.5) * 2.0
                                # One known unresolved label proves missing is retained.
                                if (split, regime, method, cost, risk, exit_policy, local) == (
                                    "VALIDATION", "RANGE", "baseline", "observed_bid_ask", "margin_cap_92", "SL", 3
                                ):
                                    net_value = None
                                else:
                                    net_value = round(net, 6)
                                margin_feasible = 1.0 if risk == "margin_cap_92" or local < 3 else 0.0
                                records.append({
                                    "episode_id": f"fx-{counter:04d}", "source_sha": fixture_digest,
                                    "decision_time": decision, "pair": pair, "timeframe": timeframe,
                                    "regime": regime, "strategy": method, "parameter_set": f"{method}_v1",
                                    "cost_scenario": cost, "exposure_state": risk, "exit_policy": exit_policy,
                                    "split": split, "method": method, "cost": cost, "risk": risk,
                                    "exit": exit_policy, "net_jpy": net_value, "units": 1000.0,
                                    "margin_feasible": margin_feasible, "fill_valid": 1.0,
                                    "unwind_valid": 1.0, "admission_status": "DIAGNOSTIC_ONLY",
                                })
    return records


def _candidate_summaries(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], dict[str, object]] = defaultdict(dict)
    for row in long_rows:
        key = tuple(row.get(name) for name in ("split", "timeframe", "pair", "regime", "method", "cost", "risk", "exit"))
        groups[key][str(row["metric"])] = row.get("value")
        for name in ("split", "timeframe", "pair", "regime", "method", "cost", "risk", "exit"):
            groups[key][name] = row.get(name)
    return list(groups.values())


def main() -> int:
    records = fixture_records()
    long_rows = records_to_long(records)
    cube = build_cube(long_rows)
    candidates = _candidate_summaries(long_rows)
    analyses = {
        "interaction_contrasts": pairwise_interactions(long_rows),
        "simpson_candidates": simpson_candidates(long_rows),
        "pareto_validation": pareto_front(candidates),
        "fallback_proofs": {
            "causal_placebo": placebo_refutation([float(r["net_jpy"]) for r in records if r["net_jpy"] is not None]),
            "conformal_interval": __import__("audit_core").deterministic_conformal([float(r["net_jpy"]) for r in records if r["net_jpy"] is not None]),
            "streaming_drift": drift_refutation([float(r["net_jpy"]) for r in records if r["net_jpy"] is not None]),
        },
        "external_adapter_status": {
            package: __import__("audit_core").optional_status(package)
            for package in ("dowhy", "mapie", "river", "SALib", "pymoo", "xarray")
        },
    }
    (HERE / "fixture_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "canonical_long_table.jsonl").open("w", encoding="utf-8") as handle:
        for row in long_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (HERE / "cube.json").write_text(json.dumps(cube.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (HERE / "analysis.json").write_text(json.dumps(analyses, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    (HERE / "run_manifest.json").write_text(json.dumps({
        "fixture_record_count": len(records), "long_row_count": len(long_rows),
        "source_sha": source_sha(HERE / "fixture_records.json"),
        "holdout_read": False, "live_paper_broker_order_deploy_touched": False,
        "missing_cell_policy": "absent/null, never zero",
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "long_rows": len(long_rows), "pareto": len(analyses["pareto_validation"]), "interactions": len(analyses["interaction_contrasts"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

