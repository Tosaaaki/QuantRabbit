from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from causal_composite_indicators_v3 import ALL_FEATURES, CURRENT_FEATURES
from indicator_factory_v3 import canonical, digest, metrics, period, qvalue


WORKERS = ("IMMEDIATE_ESCAPE", "ACCEPTED_ESCAPE", "CONFIRMED_REJECTION", "SWEEP_RECOVERY")
HORIZONS = (3, 6, 12, 24)
CONDITIONS = (("LE_Q20", .20, "le"), ("LE_Q30", .30, "le"),
              ("GE_Q70", .70, "ge"), ("GE_Q80", .80, "ge"))
FEATURES_BY_WORKER = {
    "IMMEDIATE_ESCAPE": CURRENT_FEATURES,
    "ACCEPTED_ESCAPE": ALL_FEATURES,
    "CONFIRMED_REJECTION": ALL_FEATURES,
    "SWEEP_RECOVERY": ALL_FEATURES,
}


def load(path: Path):
    events = {}
    scores = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["type"] == "RAW_EVENT":
                values = {feature: float(row[feature]) for feature in ALL_FEATURES}
                if all(math.isfinite(value) for value in values.values()):
                    events[row["signal_id"]] = values
                continue
            key = (row["worker"], int(row["horizon"]), row["signal_id"])
            item = scores.setdefault(key, {
                "signal_id": row["signal_id"], "pair": row["pair"],
                "fill_time": row["fill_time"], "direction": int(row["direction"]),
                "returns": {},
            })
            item["returns"][row["arm"]] = float(
                row["gross_return"] if row["arm"] == "RAW_SIGNAL" else row["net_return"]
            )
    scores = {key: value for key, value in scores.items()
              if value["signal_id"] in events and set(value["returns"]) == {
                  "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
              }}
    return events, scores


def build_rules(events, ids, tuning_ids, features):
    rules = []
    seen = set()
    for feature in features:
        tuning_values = [events[sid][feature] for sid in tuning_ids]
        for _, quantile, op in CONDITIONS:
            threshold = qvalue(tuning_values, quantile)
            members = {
                sid for sid in ids
                if (events[sid][feature] <= threshold if op == "le"
                    else events[sid][feature] >= threshold)
            }
            identity = (feature, op, threshold, frozenset(members))
            if identity in seen:
                continue
            seen.add(identity)
            rules.append({"feature": feature, "op": op, "threshold": threshold, "members": members})
    return rules


def evaluate(ledger: Path, output: Path, timeframe_minutes: int, top_n: int) -> dict:
    events, scores = load(ledger)
    prepared = []
    for worker in WORKERS:
        for horizon in HORIZONS:
            group = {sid: row for (w, h, sid), row in scores.items() if w == worker and h == horizon}
            ids = set(group) & set(events)
            tuning_ids = {sid for sid in ids if period(group[sid]["fill_time"]) == "TUNING"}
            if len(tuning_ids) < 20:
                continue
            rules = build_rules(events, ids, tuning_ids, FEATURES_BY_WORKER[worker])
            prepared.append((worker, horizon, group, ids, rules))
    family_tests = sum(len(item[4]) for item in prepared)
    if not family_tests:
        raise ValueError("no testable composite family")
    z = statistics.NormalDist().inv_cdf(1 - .05 / (2 * family_tests))
    results = []
    rejected = 0
    for worker, horizon, group, ids, rules in prepared:
        for rule in rules:
            rows_by_period = {
                p: [group[sid] for sid in rule["members"] if period(group[sid]["fill_time"]) == p]
                for p in ("TUNING", "WALK_FORWARD", "OPENED_DIAGNOSTIC")
            }
            if len(rows_by_period["TUNING"]) < 20 or len(rows_by_period["WALK_FORWARD"]) < 20:
                rejected += 1
                continue
            public_rule = {key: rule[key] for key in ("feature", "op", "threshold")}
            periods = {
                p: {arm: metrics(rows, arm, z) for arm in (
                    "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
                )} for p, rows in rows_by_period.items()
            }
            corrected = [
                periods["TUNING"]["RAW_SIGNAL"]["family_corrected_cluster_lower_bps"],
                periods["WALK_FORWARD"]["EXECUTABLE_BASE"]["family_corrected_cluster_lower_bps"],
                periods["WALK_FORWARD"]["ADVERSE_STRESS"]["family_corrected_cluster_lower_bps"],
            ]
            uncorrected = [
                periods["TUNING"]["RAW_SIGNAL"]["cluster_lower_bps"],
                periods["WALK_FORWARD"]["EXECUTABLE_BASE"]["cluster_lower_bps"],
                periods["WALK_FORWARD"]["ADVERSE_STRESS"]["cluster_lower_bps"],
            ]
            results.append({
                "candidate_id": digest({"worker": worker, "horizon": horizon, "rule": public_rule})[:24],
                "worker": worker,
                "timeframe_minutes": timeframe_minutes,
                "horizon_bars": horizon,
                "horizon_hours": horizon * timeframe_minutes / 60,
                "rule": public_rule,
                "periods": periods,
                "development_admitted": all(x is not None and x > 0 for x in corrected),
                "ranking_floor_bps": min((x for x in uncorrected if x is not None), default=-math.inf),
            })
    results.sort(key=lambda x: (x["development_admitted"], x["ranking_floor_bps"]), reverse=True)
    payload = {
        "factory_id": "FX_CAUSAL_COMPOSITE_FACTORY_V3",
        "evidence_class": "opened_development_not_future_holdout",
        "source_ledger": str(ledger),
        "source_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "timeframe_minutes": timeframe_minutes,
        "features_by_worker": {key: list(value) for key, value in FEATURES_BY_WORKER.items()},
        "candidate_complexity": "single_fixed_composite_only",
        "family_tests": family_tests,
        "family_corrected_z": z,
        "evaluated": len(results),
        "rejected_for_min_samples": rejected,
        "development_admitted_count": sum(item["development_admitted"] for item in results),
        "final_admitted_count": 0,
        "top_candidates": results[:top_n],
        "final_admission_blockers": [
            "opened development data are not a future holdout",
            "portfolio inventory and terminal MTM are not evaluated in this signal factory",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeframe-minutes", type=int, required=True)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    result = evaluate(args.ledger, args.output, args.timeframe_minutes, args.top)
    print(canonical({key: result[key] for key in (
        "family_tests", "evaluated", "development_admitted_count", "final_admitted_count"
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
