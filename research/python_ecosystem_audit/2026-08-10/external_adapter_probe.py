"""Execute one optional OSS adapter against the shared bounded fixture.

Run this file with the candidate-specific isolated interpreter.  The
QuantRabbit long table remains the financial oracle; adapters may organise or
analyse its after-cost outputs but never recompute fills, costs, or PnL.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import tracemalloc
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
DIMS = ("split", "timeframe", "pair", "regime", "method", "cost", "risk", "exit")


def _load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = json.loads((HERE / "fixture_records.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (HERE / "canonical_long_table.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return records, rows


def _digest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["metric"] in {"after_cost_net_jpy", "lcb_jpy"}]
    payload = [(tuple(row.get(dim) for dim in DIMS), row["metric"], row["value"]) for row in selected]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "after_cost_sum": sum(float(row["value"]) for row in selected if row["metric"] == "after_cost_net_jpy" and row["value"] is not None),
        "lcb_sum": sum(float(row["value"]) for row in selected if row["metric"] == "lcb_jpy" and row["value"] is not None),
        "null_values": sum(row["value"] is None for row in selected),
    }


def _bench(operation: Callable[[], dict[str, Any]], repeats: int = 3) -> dict[str, Any]:
    first = operation()
    second = operation()
    deterministic = json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
    durations: list[float] = []
    peaks: list[int] = []
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - started)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    return {
        "result": first,
        "deterministic_repeat": deterministic,
        "median_seconds": statistics.median(durations),
        "peak_python_bytes": max(peaks),
        "memory_scope": "tracemalloc Python allocations only; native allocator RSS excluded",
    }


def _xarray(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import xarray as xr

    metrics = sorted({str(row["metric"]) for row in rows})
    coords = {dim: sorted({row[dim] for row in rows}, key=str) for dim in DIMS}
    indexes = {dim: {value: index for index, value in enumerate(values)} for dim, values in coords.items()}
    metric_index = {value: index for index, value in enumerate(metrics)}
    shape = tuple(len(coords[dim]) for dim in DIMS) + (len(metrics),)

    def operation() -> dict[str, Any]:
        values = np.full(shape, np.nan, dtype=float)
        for row in rows:
            index = tuple(indexes[dim][row[dim]] for dim in DIMS) + (metric_index[str(row["metric"])],)
            if row["value"] is not None:
                values[index] = float(row["value"])
        data = xr.DataArray(values, dims=DIMS + ("metric",), coords={**coords, "metric": metrics})
        differences = []
        for row in rows:
            actual = data.sel({**{dim: row[dim] for dim in DIMS}, "metric": row["metric"]}).item()
            if row["value"] is not None:
                differences.append(abs(float(row["value"]) - float(actual)))
        absent = data.sel({
            "split": "VALIDATION", "timeframe": "M15", "pair": "AUD_JPY", "regime": "RANGE",
            "method": "cube_shadow", "cost": "stress_plus_1pip", "risk": "margin_cap_70",
            "exit": "TIMEOUT", "metric": "after_cost_net_jpy",
        }).item()
        source_after_cost = sum(float(row["value"]) for row in rows if row["metric"] == "after_cost_net_jpy" and row["value"] is not None)
        cube_after_cost = float(data.sel(metric="after_cost_net_jpy").sum(skipna=True).item())
        source_lcb = sum(float(row["value"]) for row in rows if row["metric"] == "lcb_jpy" and row["value"] is not None)
        cube_lcb = float(data.sel(metric="lcb_jpy").sum(skipna=True).item())
        return {
            "numeric_max_abs_diff": max(differences, default=0.0),
            "after_cost_sum_diff": cube_after_cost - source_after_cost,
            "lcb_sum_diff": cube_lcb - source_lcb,
            "known_absent_is_nan": bool(np.isnan(absent)),
            "dense_nan_count": int(np.isnan(values).sum()),
            "observed_non_null_count": sum(row["value"] is not None for row in rows),
            "dims": list(data.dims),
        }

    return _bench(operation)


def _salib(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    factors = ("regime", "method", "cost", "risk", "exit")
    levels = {name: sorted({record[name] for record in records}, key=str) for name in factors}
    train = [record for record in records if record["split"] == "TRAIN" and record["net_jpy"] is not None]
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in train:
        grouped[tuple(record[name] for name in factors)].append(float(record["net_jpy"]))
    lookup = {key: statistics.fmean(values) for key, values in grouped.items()}
    problem = {"num_vars": len(factors), "names": list(factors), "bounds": [[0.0, 1.0]] * len(factors)}

    def evaluate(matrix: Any) -> Any:
        return np.asarray([
            lookup[tuple(levels[name][int(value >= 0.5)] for name, value in zip(factors, sample))]
            for sample in matrix
        ], dtype=float)

    def operation() -> dict[str, Any]:
        samples = sobol_sample.sample(problem, 128, calc_second_order=True, seed=19)
        outputs = evaluate(samples)
        result = sobol_analyze.analyze(problem, outputs, calc_second_order=True, num_resamples=64, seed=19, print_to_console=False)
        factorial = np.asarray([[float(levels[name].index(key[index])) for index, name in enumerate(factors)] for key in lookup])
        factorial = factorial / 1.0
        max_lookup_diff = max(abs(predicted - lookup[key]) for predicted, key in zip(evaluate(factorial), lookup))
        second = {}
        for left in range(len(factors)):
            for right in range(left + 1, len(factors)):
                value = float(result["S2"][left, right])
                second[f"{factors[left]}:{factors[right]}"] = None if math.isnan(value) else round(value, 12)
        return {
            "factorial_lookup_max_abs_diff": float(max_lookup_diff),
            "first_order": {name: round(float(value), 12) for name, value in zip(factors, result["S1"])},
            "total_order": {name: round(float(value), 12) for name, value in zip(factors, result["ST"])},
            "second_order": second,
            "sample_count": int(len(outputs)),
            "training_only": True,
            "missing_outcomes_excluded_not_zero": sum(record["net_jpy"] is None for record in records),
        }

    return _bench(operation)


def _candidate_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[dim] for dim in DIMS)
        grouped[key].update({dim: row[dim] for dim in DIMS})
        grouped[key][row["metric"]] = row["value"]
    return list(grouped.values())


def _pymoo(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from audit_core import pareto_front

    candidates = _candidate_summaries(rows)
    labels = lambda row: "|".join(str(row[dim]) for dim in DIMS)

    def operation() -> dict[str, Any]:
        custom = pareto_front(candidates)
        feasible = [row for row in candidates if row["split"] == "VALIDATION"
                    and row.get("sample_coverage", 0) >= 0.8 and row.get("margin_coverage", 0) >= 0.8
                    and row.get("fill_validity", 0) >= 1.0 and row.get("unwind_validity", 0) >= 1.0]
        objectives = np.asarray([[-float(row["after_cost_net_jpy"]), -float(row["lcb_jpy"]),
                                  -float(row["profit_factor"]), float(row["max_drawdown_jpy"]),
                                  -float(row["margin_coverage"])] for row in feasible])
        indexes = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
        external_labels = sorted(labels(feasible[int(index)]) for index in indexes)
        custom_labels = sorted(labels(row) for row in custom)
        return {
            "front_exact_match": external_labels == custom_labels,
            "external_front": external_labels,
            "custom_front": custom_labels,
            "feasible_candidates": len(feasible),
            "missing_candidates_excluded_not_zero": sum(any(row.get(metric) is None for metric in ("after_cost_net_jpy", "lcb_jpy")) for row in candidates),
            "validation_only": True,
        }

    return _bench(operation)


def _dowhy(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from dowhy import CausalModel

    complete = [record for record in records if record["net_jpy"] is not None]
    frame = pd.DataFrame({
        "treatment": [int(record["method"] == "cube_shadow") for record in complete],
        "outcome": [float(record["net_jpy"]) for record in complete],
        "regime": [int(record["regime"] == "TREND") for record in complete],
        "cost": [int(record["cost"] == "stress_plus_1pip") for record in complete],
        "risk": [int(record["risk"] == "margin_cap_70") for record in complete],
        "exit": [int(record["exit"] == "TIMEOUT") for record in complete],
    })
    common_causes = ["regime", "cost", "risk", "exit"]
    design = np.column_stack([np.ones(len(frame)), frame["treatment"].to_numpy(), *[frame[name].to_numpy() for name in common_causes]])
    manual_effect = float(np.linalg.lstsq(design, frame["outcome"].to_numpy(), rcond=None)[0][1])

    def operation() -> dict[str, Any]:
        model = CausalModel(data=frame, treatment="treatment", outcome="outcome", common_causes=common_causes)
        estimand = model.identify_effect(proceed_when_unidentifiable=False)
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        placebo = model.refute_estimate(
            estimand, estimate, method_name="placebo_treatment_refuter", placebo_type="permute",
            num_simulations=20, random_seed=17, show_progress_bar=False,
        )
        return {
            "effect": round(float(estimate.value), 12),
            "manual_ols_effect": round(manual_effect, 12),
            "effect_abs_diff": abs(float(estimate.value) - manual_effect),
            "placebo_new_effect": round(float(placebo.new_effect), 12),
            "placebo_p_value": round(float(placebo.refutation_result["p_value"]), 12),
            "complete_rows": len(frame),
            "missing_outcomes_excluded_not_zero": sum(record["net_jpy"] is None for record in records),
            "causal_assumptions_are_not_proven": True,
        }

    return _bench(operation, repeats=1)


def _features(records: list[dict[str, Any]]) -> list[list[float]]:
    return [[
        float(record["regime"] == "TREND"), float(record["method"] == "cube_shadow"),
        float(record["cost"] == "stress_plus_1pip"), float(record["risk"] == "margin_cap_70"),
        float(record["exit"] == "TIMEOUT"),
    ] for record in records]


def _mapie(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from mapie.regression import SplitConformalRegressor
    from sklearn.linear_model import LinearRegression

    train = sorted([record for record in records if record["split"] == "TRAIN" and record["net_jpy"] is not None], key=lambda row: row["decision_time"])
    validation_all = sorted([record for record in records if record["split"] == "VALIDATION"], key=lambda row: row["decision_time"])
    validation = [record for record in validation_all if record["net_jpy"] is not None]
    cut = len(train) // 2
    fit_rows, conform_rows = train[:cut], train[cut:]
    x_fit, y_fit = np.asarray(_features(fit_rows)), np.asarray([record["net_jpy"] for record in fit_rows], dtype=float)
    x_conform, y_conform = np.asarray(_features(conform_rows)), np.asarray([record["net_jpy"] for record in conform_rows], dtype=float)
    x_validation = np.asarray(_features(validation))
    y_validation = np.asarray([record["net_jpy"] for record in validation], dtype=float)

    def operation() -> dict[str, Any]:
        adapter = SplitConformalRegressor(estimator=LinearRegression(), confidence_level=0.9, prefit=False)
        adapter.fit(x_fit, y_fit)
        adapter.conformalize(x_conform, y_conform)
        predicted, intervals = adapter.predict_interval(x_validation)
        residuals = np.abs(y_conform - adapter.predict(x_conform))
        quantile_index = min(len(residuals) - 1, math.ceil((len(residuals) + 1) * 0.9) - 1)
        radius = float(np.sort(residuals)[quantile_index])
        manual_lower, manual_upper = predicted - radius, predicted + radius
        lower, upper = intervals[:, 0, 0], intervals[:, 1, 0]
        coverage = float(np.mean((y_validation >= lower) & (y_validation <= upper)))
        return {
            "manual_bound_max_abs_diff": float(max(np.max(np.abs(lower - manual_lower)), np.max(np.abs(upper - manual_upper)))),
            "validation_coverage": coverage,
            "mean_interval_width": float(np.mean(upper - lower)),
            "fit_count": len(fit_rows), "conformal_count": len(conform_rows), "validation_count": len(validation),
            "missing_validation_excluded_not_zero": len(validation_all) - len(validation),
            "chronological_split": True, "holdout_read": False,
        }

    return _bench(operation)


def _river(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    from river import drift, stats

    ordered = sorted(records, key=lambda row: (row["decision_time"], row["episode_id"]))
    values = [float(record["net_jpy"]) for record in ordered if record["net_jpy"] is not None]

    def operation() -> dict[str, Any]:
        online_mean = stats.Mean()
        detector = drift.ADWIN(delta=0.01, clock=1, min_window_length=5, grace_period=10)
        changes = []
        for index, value in enumerate(values):
            online_mean.update(value)
            detector.update(value)
            if detector.drift_detected:
                changes.append(index)
        return {
            "online_mean": float(online_mean.get()),
            "stdlib_mean": statistics.fmean(values),
            "mean_abs_diff": abs(float(online_mean.get()) - statistics.fmean(values)),
            "change_indexes": changes,
            "event_order_fixed": True,
            "missing_outcomes_skipped_not_zero": sum(record["net_jpy"] is None for record in ordered),
            "holdout_read": False,
        }

    return _bench(operation)


ADAPTERS = {"xarray": _xarray, "salib": _salib, "pymoo": _pymoo, "dowhy": _dowhy, "mapie": _mapie, "river": _river}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=sorted(ADAPTERS))
    args = parser.parse_args()
    records, rows = _load()
    before = _digest(rows)
    result = ADAPTERS[args.candidate](records, rows)
    after = _digest(rows)
    payload = {
        "candidate": args.candidate,
        "fixture_records": len(records), "long_rows": len(rows),
        "financial_oracle_before": before, "financial_oracle_after": after,
        "financial_oracle_unchanged": before == after,
        "adapter": result,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
