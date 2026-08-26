from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


FEATURES = (
    "rail_escape_energy",
    "boundary_acceptance",
    "rejection_curvature",
    "geodesic_efficiency",
    "abs_price_spread_loop_area",
    "session_spread_strain",
    "boundary_crowding",
    "pre_break_compression",
    "wick_rejection_ratio",
    "tick_volume_shock",
    "liquidity_sweep_geometry",
    "currency_propagation",
    "abs_currency_propagation",
    "currency_breadth",
    "currency_propagation_concentration",
)
QUANTILE_CONDITIONS = (("LE_Q20", 0.20, "le"), ("LE_Q30", 0.30, "le"),
                       ("GE_Q70", 0.70, "ge"), ("GE_Q80", 0.80, "ge"))
WORKERS = ("IMMEDIATE_ESCAPE", "ACCEPTED_ESCAPE", "CONFIRMED_REJECTION", "SWEEP_RECOVERY")
HORIZONS = (3, 6, 12, 24)

# rejection_curvature contains the next completed bar by construction.  It is
# legal only for workers whose decision occurs after that bar and whose fill is
# the following executable open.  Keeping this table beside the factory makes
# feature-time authority explicit instead of relying on strategy prose.
FEATURES_BY_WORKER = {
    "IMMEDIATE_ESCAPE": tuple(x for x in FEATURES if x != "rejection_curvature"),
    "ACCEPTED_ESCAPE": FEATURES,
    "CONFIRMED_REJECTION": FEATURES,
    "SWEEP_RECOVERY": FEATURES,
}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def qvalue(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty quantile input")
    xs = sorted(values)
    return xs[int((len(xs) - 1) * q)]


def transform_features(raw: dict) -> dict[str, float] | None:
    graph = raw.get("currency_propagation")
    if graph is None:
        return None
    result = {k: float(raw[k]) for k in FEATURES if not k.startswith("abs_")}
    result["abs_price_spread_loop_area"] = abs(float(raw["price_spread_loop_area"]))
    result["abs_currency_propagation"] = abs(float(graph))
    if not all(math.isfinite(v) for v in result.values()):
        return None
    return result


def load_ledger(path: Path):
    events: dict[str, dict] = {}
    scores: dict[tuple[str, int, str], dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["type"] == "RAW_EVENT":
                transformed = transform_features(row)
                if transformed is not None:
                    events[row["signal_id"]] = {**row, "features": transformed}
                continue
            key = (row["worker"], int(row["horizon"]), row["signal_id"])
            item = scores.setdefault(key, {
                "signal_id": row["signal_id"], "pair": row["pair"],
                "fill_time": row["fill_time"], "direction": int(row["direction"]),
                "returns": {},
            })
            if row["arm"] == "RAW_SIGNAL":
                item["returns"]["RAW_SIGNAL"] = float(row["gross_return"])
            else:
                item["returns"][row["arm"]] = float(row["net_return"])
    complete = {
        k: v for k, v in scores.items()
        if v["signal_id"] in events and set(v["returns"]) == {"RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"}
    }
    return events, complete


def period(stamp: str) -> str:
    if stamp < "2026-05-01":
        return "TUNING"
    if stamp < "2026-07-01":
        return "WALK_FORWARD"
    return "OPENED_DIAGNOSTIC"


def week_key(stamp: str) -> str:
    # ISO dates are sufficient for deterministic seven-day research clusters.
    # Anchor at the Unix epoch date without accepting local timezone state.
    import datetime as dt
    day = dt.date.fromisoformat(stamp[:10])
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def lower_bound(values_by_week: dict[str, list[float]], z: float) -> float | None:
    weekly = [statistics.fmean(xs) for xs in values_by_week.values() if xs]
    if len(weekly) < 3:
        return None
    mean = statistics.fmean(weekly)
    return mean - z * statistics.stdev(weekly) / math.sqrt(len(weekly))


def currency_exposure_diagnostics(rows: list[dict]) -> dict:
    """Count effective currency-time bets without treating tickets as independent.

    Each unit ticket contributes +1/-1 to the two currency nodes implied by its
    direction.  This is a diagnostic only; it never gates RAW signal creation.
    """
    gross_by_currency: dict[str, float] = defaultdict(float)
    net_by_currency: dict[str, float] = defaultdict(float)
    gross_by_currency_day: dict[tuple[str, str], float] = defaultdict(float)
    for row in rows:
        base, quote = row["pair"].split("_")
        direction = 1.0 if row["direction"] > 0 else -1.0
        day = row["fill_time"][:10]
        for currency, signed in ((base, direction), (quote, -direction)):
            gross_by_currency[currency] += 1.0
            net_by_currency[currency] += signed
            gross_by_currency_day[(day, currency)] += 1.0

    def concentration(values) -> tuple[float | None, float | None]:
        weights = [abs(float(value)) for value in values if value]
        total = sum(weights)
        if total == 0:
            return None, None
        hhi = sum((value / total) ** 2 for value in weights)
        return hhi, 1.0 / hhi

    gross_hhi, effective_currency_nodes = concentration(gross_by_currency.values())
    net_hhi, effective_net_currency_nodes = concentration(net_by_currency.values())
    currency_time_hhi, effective_currency_time_bets = concentration(gross_by_currency_day.values())
    gross_total = sum(gross_by_currency.values())
    return {
        "gross_currency_hhi": gross_hhi,
        "net_currency_hhi": net_hhi,
        "effective_currency_nodes": effective_currency_nodes,
        "effective_net_currency_nodes": effective_net_currency_nodes,
        "effective_currency_time_bets": effective_currency_time_bets,
        "max_gross_currency_share": (
            max(gross_by_currency.values()) / gross_total if gross_total else None
        ),
        "net_currency_exposure_units": dict(sorted(net_by_currency.items())),
    }


def metrics(rows: list[dict], arm: str, z: float) -> dict:
    values = [row["returns"][arm] * 10000 for row in rows]
    by_week: dict[str, list[float]] = defaultdict(list)
    by_month: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values):
        by_week[week_key(row["fill_time"])].append(value)
        by_month[row["fill_time"][:7]].append(value)
    return {
        "n": len(values),
        "mean_bps": statistics.fmean(values) if values else None,
        "median_bps": statistics.median(values) if values else None,
        "positive_rate": sum(v > 0 for v in values) / len(values) if values else None,
        "week_clusters": len(by_week),
        "cluster_lower_bps": lower_bound(by_week, 1.959963984540054),
        "family_corrected_cluster_lower_bps": lower_bound(by_week, z),
        "monthly_mean_bps": {m: statistics.fmean(xs) for m, xs in sorted(by_month.items())},
        "currency_exposure": currency_exposure_diagnostics(rows),
    }


def build_conditions(
    events: dict[str, dict], tuning_ids: set[str], all_ids: set[str],
    features: tuple[str, ...] = FEATURES,
):
    conditions = []
    seen = set()
    for feature in features:
        tuning_values = [events[sid]["features"][feature] for sid in tuning_ids]
        for label, q, op in QUANTILE_CONDITIONS:
            threshold = qvalue(tuning_values, q)
            if op == "le":
                members = {sid for sid in all_ids if events[sid]["features"][feature] <= threshold}
            else:
                members = {sid for sid in all_ids if events[sid]["features"][feature] >= threshold}
            # Ties can make Q20 and Q30 (or Q70 and Q80) the exact same rule.
            # Collapse them before any outcome is inspected so FWER counts only
            # distinct, pre-outcome semantic hypotheses.
            identity = (feature, op, threshold, frozenset(members))
            if identity in seen:
                continue
            seen.add(identity)
            conditions.append({
                "id": f"{feature}:{label}", "feature": feature, "op": op,
                "threshold": threshold, "members": members,
            })
    return conditions


def candidate_specs(conditions: list[dict]):
    for condition in conditions:
        yield (condition,)
    for i, left in enumerate(conditions):
        for right in conditions[i + 1:]:
            if left["feature"] == right["feature"]:
                continue
            yield (left, right)


def prepare_groups(events: dict[str, dict], scores: dict) -> list[dict]:
    groups = []
    for worker in WORKERS:
        for horizon in HORIZONS:
            group = {sid: row for (w, h, sid), row in scores.items() if w == worker and h == horizon}
            all_ids = set(group) & set(events)
            tuning_ids = {sid for sid in all_ids if period(group[sid]["fill_time"]) == "TUNING"}
            if len(tuning_ids) < 20:
                continue
            conditions = build_conditions(
                events, tuning_ids, all_ids, FEATURES_BY_WORKER[worker]
            )
            specs = list(candidate_specs(conditions))
            groups.append({
                "worker": worker, "horizon": horizon, "rows": group,
                "all_ids": all_ids, "conditions": conditions, "specs": specs,
            })
    return groups


def evaluate(ledger: Path, output: Path, top_n: int = 100, timeframe_minutes: int = 240) -> dict:
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")
    events, scores = load_ledger(ledger)
    groups = prepare_groups(events, scores)
    family_tests = sum(len(group["specs"]) for group in groups)
    if not family_tests:
        raise ValueError("no testable candidate family")
    alpha = 0.05
    z_corrected = statistics.NormalDist().inv_cdf(1 - alpha / (2 * family_tests))
    evaluated = []
    rejected_sample = 0
    seen_candidate_ids = set()
    for prepared in groups:
            worker = prepared["worker"]
            horizon = prepared["horizon"]
            group = prepared["rows"]
            all_ids = prepared["all_ids"]
            for spec in prepared["specs"]:
                members = set.intersection(*(c["members"] for c in spec)) & all_ids
                rows_by_period = {
                    p: [group[sid] for sid in members if period(group[sid]["fill_time"]) == p]
                    for p in ("TUNING", "WALK_FORWARD", "OPENED_DIAGNOSTIC")
                }
                if len(rows_by_period["TUNING"]) < 20 or len(rows_by_period["WALK_FORWARD"]) < 20:
                    rejected_sample += 1
                    continue
                rule = [{k: c[k] for k in ("feature", "op", "threshold")} for c in spec]
                candidate_id = digest({"worker": worker, "horizon": horizon, "rule": rule})[:24]
                if candidate_id in seen_candidate_ids:
                    raise AssertionError(f"duplicate candidate identity: {candidate_id}")
                seen_candidate_ids.add(candidate_id)
                result = {
                    "candidate_id": candidate_id,
                    "worker": worker,
                    "timeframe_minutes": timeframe_minutes,
                    "horizon_bars": horizon,
                    "horizon_hours": horizon * timeframe_minutes / 60,
                    "rule": rule,
                    "periods": {},
                }
                for p, rows in rows_by_period.items():
                    result["periods"][p] = {
                        arm: metrics(rows, arm, z_corrected)
                        for arm in ("RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS")
                    }
                tune = result["periods"]["TUNING"]["RAW_SIGNAL"]
                walk_base = result["periods"]["WALK_FORWARD"]["EXECUTABLE_BASE"]
                walk_adv = result["periods"]["WALK_FORWARD"]["ADVERSE_STRESS"]
                corrected = [tune["family_corrected_cluster_lower_bps"],
                             walk_base["family_corrected_cluster_lower_bps"],
                             walk_adv["family_corrected_cluster_lower_bps"]]
                uncorrected = [tune["cluster_lower_bps"], walk_base["cluster_lower_bps"], walk_adv["cluster_lower_bps"]]
                result["development_admitted"] = all(x is not None and x > 0 for x in corrected)
                result["ranking_floor_bps"] = min((x for x in uncorrected if x is not None), default=-math.inf)
                evaluated.append(result)
    evaluated.sort(key=lambda x: (x["development_admitted"], x["ranking_floor_bps"]), reverse=True)
    payload = {
        "factory_id": "FX_INDICATOR_FACTORY_V3",
        "evidence_class": "opened_development_not_future_holdout",
        "source_ledger": str(ledger),
        "source_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "features": list(FEATURES),
        "features_by_worker": {k: list(v) for k, v in FEATURES_BY_WORKER.items()},
        "workers": list(WORKERS),
        "timeframe_minutes": timeframe_minutes,
        "horizons_bars": list(HORIZONS),
        "horizons_hours": [h * timeframe_minutes / 60 for h in HORIZONS],
        "family_tests": family_tests,
        "semantic_duplicate_conditions_removed": (
            sum(len(FEATURES_BY_WORKER[group["worker"]]) for group in groups)
            * len(QUANTILE_CONDITIONS)
            - sum(len(group["conditions"]) for group in groups)
        ),
        "family_wise_alpha": alpha,
        "family_corrected_z": z_corrected,
        "evaluated_with_min_samples": len(evaluated),
        "rejected_for_min_samples": rejected_sample,
        "development_admitted_count": sum(x["development_admitted"] for x in evaluated),
        "final_admitted_count": 0,
        "top_candidates": evaluated[:top_n],
        "final_admission_blockers": [
            "candidate family was developed on already-opened data",
            "new future holdout has not elapsed",
            "portfolio inventory, terminal MTM, financing and equity multiple are not in this signal factory",
        ],
        "live_authority": False,
        "external_orders": 0
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--timeframe-minutes", type=int, required=True)
    args = parser.parse_args()
    result = evaluate(args.ledger, args.output, args.top, args.timeframe_minutes)
    print(canonical({
        "family_tests": result["family_tests"],
        "evaluated": result["evaluated_with_min_samples"],
        "development_admitted": result["development_admitted_count"],
        "final_admitted": result["final_admitted_count"],
        "output": str(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
