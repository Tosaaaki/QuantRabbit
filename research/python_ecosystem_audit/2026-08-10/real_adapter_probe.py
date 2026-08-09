"""Execute one adopted package against the frozen real-cohort payload."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import tracemalloc
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
SEED = 20_260_809
TARGET_COVERAGE = 0.90


def _load() -> dict[str, Any]:
    return json.loads((HERE / "real_shadow_payload.json").read_text(encoding="utf-8"))


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if hasattr(value, "item"):
        return _canonical(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 12)
    return value


def _bench(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for _ in range(2):
        tracemalloc.start()
        started = time.perf_counter_ns()
        result = _canonical(operation())
        elapsed = time.perf_counter_ns() - started
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        outputs.append(result)
        samples.append({"elapsed_ns": elapsed, "python_current_bytes": current, "python_peak_bytes": peak})
    return {
        "result": outputs[0],
        "deterministic_repeat": outputs[0] == outputs[1],
        "repeat_digest": hashlib.sha256(json.dumps(outputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "benchmark": {
            "samples": samples,
            "median_elapsed_ns": int(statistics.median(item["elapsed_ns"] for item in samples)),
            "max_python_peak_bytes": max(item["python_peak_bytes"] for item in samples),
            "cross_semantic_speedup_claimed": False,
        },
    }


def _financial_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload["financial_invariants"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def xarray_probe(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import xarray as xr

    rows = payload["long_rows"]
    dims = payload["cube_axes"] + ["metric"]

    def operation() -> dict[str, Any]:
        frame = pd.DataFrame(rows)
        if frame.duplicated(dims).any():
            duplicates = int(frame.duplicated(dims).sum())
            raise RuntimeError(f"canonical long keys are not unique: {duplicates}")
        series = frame.set_index(dims)["value"].astype(float)
        cube = xr.DataArray.from_series(series, sparse=False)
        observed = cube.to_series().reindex(series.index)
        populated = series.notna()
        numeric_error = float(np.max(np.abs(observed[populated].to_numpy() - series[populated].to_numpy()))) if populated.any() else 0.0
        null_preserved = bool(observed[~populated].isna().all())
        expanded = cube.reindex(pair=list(cube.coords["pair"].values) + ["__KNOWN_ABSENT__"])
        absent_is_nan = bool(expanded.sel(pair="__KNOWN_ABSENT__").isnull().all())
        return {
            "numeric_max_abs_diff": numeric_error,
            "input_populated_values": int(populated.sum()),
            "input_null_values": int((~populated).sum()),
            "input_null_preserved": null_preserved,
            "known_absent_coordinate_is_nan": absent_is_nan,
            "dims": list(cube.dims),
            "shape": list(cube.shape),
            "missing_is_nan_not_zero": True,
            "financial_cells_changed": 0,
        }

    return _bench(operation)


def _rank_agreement(left: list[str], right: list[str]) -> float | None:
    common = [name for name in left if name in right]
    if len(common) < 2:
        return None
    positions_left = {name: index for index, name in enumerate(left)}
    positions_right = {name: index for index, name in enumerate(right)}
    n = len(common)
    squared = sum((positions_left[name] - positions_right[name]) ** 2 for name in common)
    return 1.0 - 6.0 * squared / (n * (n * n - 1))


def salib_probe(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from SALib.analyze import delta
    from scipy.stats import spearmanr

    records = payload["episode_records"]

    def operation() -> dict[str, Any]:
        windows: dict[str, Any] = {}
        for window in ("INITIAL_16D", "DOUBLE_32D", "QUADRUPLE_64D"):
            train = [
                row for row in records
                if row["window"] == window and row["split"] == "TRAIN"
                and row["method"] == "ALL_TRADES" and row["price_action_features"] is not None
            ]
            validation = [
                row for row in records
                if row["window"] == window and row["split"] == "VALIDATION"
                and row["method"] == "ALL_TRADES" and row["price_action_features"] is not None
            ]
            if len(train) < 40:
                windows[window] = {
                    "status": "INSUFFICIENT_TRAIN_FEATURE_ROWS", "train_n": len(train),
                    "validation_n": len(validation), "validation_labels_used_for_ranking": False,
                }
                continue
            names = sorted(train[0]["price_action_features"])
            usable = [
                name for name in names
                if len({round(float(row["price_action_features"][name]), 15) for row in train}) > 1
            ]
            x_train = np.asarray([[float(row["price_action_features"][name]) for name in usable] for row in train], dtype=float)
            y_train = np.asarray([float(row["actual_net_jpy"]) for row in train], dtype=float)
            bounds = [[float(x_train[:, index].min()), float(x_train[:, index].max())] for index in range(len(usable))]
            problem = {"num_vars": len(usable), "names": usable, "bounds": bounds}
            result = delta.analyze(
                problem, x_train, y_train, num_resamples=100,
                conf_level=0.95, print_to_console=False, seed=SEED,
            )
            delta_values = {name: float(value) for name, value in zip(usable, result["delta"])}
            salib_rank = sorted(usable, key=lambda name: (-delta_values[name], name))
            custom_scores: dict[str, float] = {}
            for index, name in enumerate(usable):
                coefficient = float(spearmanr(x_train[:, index], y_train).statistic)
                custom_scores[name] = 0.0 if math.isnan(coefficient) else abs(coefficient)
            custom_rank = sorted(usable, key=lambda name: (-custom_scores[name], name))
            validation_scores: dict[str, float] = {}
            if validation:
                y_validation = np.asarray([float(row["actual_net_jpy"]) for row in validation], dtype=float)
                for name in usable:
                    values = np.asarray([float(row["price_action_features"][name]) for row in validation], dtype=float)
                    coefficient = float(spearmanr(values, y_validation).statistic) if len(set(values.tolist())) > 1 else 0.0
                    validation_scores[name] = 0.0 if math.isnan(coefficient) else abs(coefficient)
            validation_rank = sorted(usable, key=lambda name: (-validation_scores.get(name, 0.0), name))
            windows[window] = {
                "status": "EXECUTED_TRAIN_ONLY_RANKING",
                "train_n": len(train), "validation_n": len(validation),
                "salib_delta": delta_values,
                "salib_rank": salib_rank,
                "custom_abs_spearman": custom_scores,
                "custom_rank": custom_rank,
                "train_rank_agreement": _rank_agreement(salib_rank, custom_rank),
                "validation_abs_spearman_for_frozen_train_features": validation_scores,
                "validation_rank": validation_rank,
                "train_to_validation_rank_agreement": _rank_agreement(salib_rank, validation_rank),
                "validation_labels_used_for_ranking": False,
                "validation_used_for_evaluation_only": True,
                "missing_feature_rows_excluded_not_zero": sum(
                    row["window"] == window and row["split"] == "TRAIN" and row["method"] == "ALL_TRADES"
                    and row["price_action_features"] is None for row in records
                ),
            }
        return {
            "windows": windows,
            "ranking_fixed_on_train": True,
            "holdout_read": False,
            "financial_cells_changed": 0,
        }

    return _bench(operation)


def _dominates(a: list[float], b: list[float]) -> bool:
    return all(left <= right for left, right in zip(a, b)) and any(left < right for left, right in zip(a, b))


def _custom_front(objectives: list[list[float]]) -> list[int]:
    return [index for index, row in enumerate(objectives) if not any(_dominates(other, row) for other_index, other in enumerate(objectives) if other_index != index)]


def _pf(values: list[float]) -> float:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    return gains / losses if losses else (1e12 if gains > 0 else 0.0)


def _dd(values: list[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def pymoo_probe(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    summaries = payload["candidate_summaries"]
    records = payload["episode_records"]

    def operation() -> dict[str, Any]:
        output: dict[str, Any] = {}
        for window in ("INITIAL_16D", "DOUBLE_32D", "QUADRUPLE_64D"):
            candidates = [row for row in summaries if row["window"] == window]
            complete = [
                row for row in candidates
                if all(row.get(name) is not None for name in (
                    "after_cost_net_jpy", "lcb_jpy", "profit_factor",
                    "max_drawdown_jpy", "margin_coverage", "sample_coverage",
                    "fill_validity", "unwind_validity",
                ))
            ]
            diagnostic = [row for row in complete if float(row["sample_coverage"]) >= 0.8]
            feasible = [
                row for row in diagnostic
                if float(row["margin_coverage"]) >= 1.0
                and float(row["fill_validity"]) >= 1.0
                and float(row["unwind_validity"]) >= 1.0
            ]
            def objectives(items: list[dict[str, Any]]) -> list[list[float]]:
                return [[
                    -float(row["after_cost_net_jpy"]), -float(row["lcb_jpy"]),
                    -float(row["profit_factor"]), float(row["max_drawdown_jpy"]),
                    -float(row["margin_coverage"]),
                ] for row in items]
            diagnostic_objectives = objectives(diagnostic)
            if diagnostic:
                external_indexes = NonDominatedSorting().do(np.asarray(diagnostic_objectives), only_non_dominated_front=True).tolist()
                custom_indexes = _custom_front(diagnostic_objectives)
            else:
                external_indexes = custom_indexes = []
            external_labels = sorted(diagnostic[index]["method"] for index in external_indexes)
            custom_labels = sorted(diagnostic[index]["method"] for index in custom_indexes)
            feasible_objectives = objectives(feasible)
            constrained_labels = sorted(
                feasible[index]["method"] for index in (
                    NonDominatedSorting().do(np.asarray(feasible_objectives), only_non_dominated_front=True).tolist()
                    if feasible else []
                )
            )
            stability: dict[str, float] = {}
            if len(diagnostic) >= 2:
                rng = np.random.default_rng(SEED)
                counts = Counter()
                source_by_method = {
                    method: [row for row in records if row["window"] == window and row["split"] == "VALIDATION" and row["method"] == method]
                    for method in (row["method"] for row in diagnostic)
                }
                n = min(len(items) for items in source_by_method.values())
                for _ in range(200):
                    indexes = rng.integers(0, n, size=n)
                    boot_objectives = []
                    labels = []
                    for method, items in sorted(source_by_method.items()):
                        scoped = items[:n]
                        applied = [float(scoped[int(index)]["actual_net_jpy"]) if scoped[int(index)]["selected"] else 0.0 for index in indexes]
                        baseline = [float(scoped[int(index)]["actual_net_jpy"]) for index in indexes]
                        deltas = [value - base for value, base in zip(applied, baseline)]
                        mean_delta = statistics.fmean(deltas)
                        lcb = mean_delta - (1.96 * statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0)
                        margin_known = sum(scoped[int(index)]["margin_evidence_known"] and scoped[int(index)]["selected"] for index in indexes)
                        selected = sum(scoped[int(index)]["selected"] for index in indexes)
                        coverage = margin_known / selected if selected else 0.0
                        boot_objectives.append([-sum(applied), -lcb, -_pf(applied), _dd(applied), -coverage])
                        labels.append(method)
                    front = NonDominatedSorting().do(np.asarray(boot_objectives), only_non_dominated_front=True).tolist()
                    counts.update(labels[int(index)] for index in front)
                stability = {method: counts[method] / 200.0 for method in sorted(counts)}
            output[window] = {
                "candidate_count": len(candidates),
                "complete_candidate_count": len(complete),
                "diagnostic_front": external_labels,
                "custom_diagnostic_front": custom_labels,
                "front_exact_match": external_labels == custom_labels,
                "constrained_front": constrained_labels,
                "constrained_front_empty_due_to_margin": not constrained_labels and bool(diagnostic),
                "bootstrap_front_inclusion_rate": stability,
                "stable_members_ge_0_80": sorted(method for method, rate in stability.items() if rate >= 0.8),
                "validation_used_for_evaluation_only": True,
            }
        return {
            "windows": output,
            "single_objective_collapse": False,
            "policy_selection_changed": False,
            "holdout_read": False,
            "financial_cells_changed": 0,
        }

    return _bench(operation)


def _parse_time(value: str) -> datetime:
    text = str(value).replace("Z", "+00:00")
    if "." in text:
        head, tail = text.split(".", 1)
        fraction, zone = tail.split("+", 1) if "+" in tail else (tail, "00:00")
        text = f"{head}.{fraction[:6]}+{zone}"
    parsed = datetime.fromisoformat(text)
    return parsed.astimezone(timezone.utc)


def mapie_probe(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from mapie.regression import SplitConformalRegressor
    from sklearn.linear_model import LinearRegression

    records = payload["episode_records"]

    def operation() -> dict[str, Any]:
        output: dict[str, Any] = {}
        for window in ("INITIAL_16D", "DOUBLE_32D", "QUADRUPLE_64D"):
            train_all = sorted([
                row for row in records if row["window"] == window and row["split"] == "TRAIN"
                and row["method"] == "ALL_TRADES"
            ], key=lambda row: row["decision_time"])
            validation_all = sorted([
                row for row in records if row["window"] == window and row["split"] == "VALIDATION"
                and row["method"] == "ALL_TRADES"
            ], key=lambda row: row["decision_time"])
            train = [row for row in train_all if row["price_action_features"] is not None]
            validation = [row for row in validation_all if row["price_action_features"] is not None]
            if len(train) < 50 or len(validation) < 20:
                output[window] = {
                    "status": "INSUFFICIENT_PRICE_FEATURE_ROWS", "train_n": len(train),
                    "validation_n": len(validation),
                    "missing_train_excluded_not_zero": len(train_all) - len(train),
                    "missing_validation_excluded_not_zero": len(validation_all) - len(validation),
                }
                continue
            split_index = max(30, math.floor(len(train) * 0.60))
            conformal_start = _parse_time(train[split_index]["decision_time"])
            fit = [row for row in train[:split_index] if _parse_time(row["close_time"]) < conformal_start - timedelta(hours=1)]
            conformal = train[split_index:]
            if len(fit) < 30 or len(conformal) < 20:
                output[window] = {
                    "status": "INSUFFICIENT_INNER_PURGED_SPLIT", "train_n": len(train),
                    "fit_n": len(fit), "conformal_n": len(conformal), "validation_n": len(validation),
                    "inner_embargo_seconds": 3600,
                }
                continue
            names = sorted(train[0]["price_action_features"])
            matrix = lambda rows: np.asarray([[float(row["price_action_features"][name]) for name in names] for row in rows], dtype=float)
            target = lambda rows: np.asarray([float(row["actual_net_jpy"]) for row in rows], dtype=float)
            x_fit, y_fit = matrix(fit), target(fit)
            x_conformal, y_conformal = matrix(conformal), target(conformal)
            x_validation, y_validation = matrix(validation), target(validation)
            adapter = SplitConformalRegressor(
                estimator=LinearRegression(), confidence_level=TARGET_COVERAGE, prefit=False,
            )
            adapter.fit(x_fit, y_fit)
            adapter.conformalize(x_conformal, y_conformal)
            prediction, intervals = adapter.predict_interval(x_validation)
            residuals = np.abs(y_conformal - adapter.predict(x_conformal))
            quantile_index = min(len(residuals) - 1, math.ceil((len(residuals) + 1) * TARGET_COVERAGE) - 1)
            radius = float(np.sort(residuals)[quantile_index])
            lower, upper = intervals[:, 0, 0], intervals[:, 1, 0]
            manual_lower, manual_upper = prediction - radius, prediction + radius
            coverage = float(np.mean((y_validation >= lower) & (y_validation <= upper)))
            output[window] = {
                "status": "EXECUTED_OUTER_VALIDATION",
                "fit_n": len(fit), "conformal_n": len(conformal), "validation_n": len(validation),
                "outer_train_n": len(train_all), "outer_validation_n": len(validation_all),
                "missing_train_excluded_not_zero": len(train_all) - len(train),
                "missing_validation_excluded_not_zero": len(validation_all) - len(validation),
                "inner_embargo_seconds": 3600,
                "manual_bound_max_abs_diff": float(max(np.max(np.abs(lower - manual_lower)), np.max(np.abs(upper - manual_upper)))),
                "target_coverage": TARGET_COVERAGE,
                "validation_coverage": coverage,
                "coverage_gap": coverage - TARGET_COVERAGE,
                "mean_interval_width_jpy": float(np.mean(upper - lower)),
                "mean_prediction_jpy": float(np.mean(prediction)),
                "mean_lower_bound_jpy": float(np.mean(lower)),
                "validation_labels_used_for_fit_or_conformal": False,
                "validation_used_for_evaluation_only": True,
            }
        return {
            "windows": output,
            "holdout_read": False,
            "policy_selection_changed": False,
            "incremental_net_jpy_attributed_to_adapter": 0.0,
            "financial_cells_changed": 0,
        }

    return _bench(operation)


PROBES = {"xarray": xarray_probe, "salib": salib_probe, "pymoo": pymoo_probe, "mapie": mapie_probe}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", choices=sorted(PROBES))
    args = parser.parse_args()
    payload = _load()
    before = _financial_digest(payload)
    adapter = PROBES[args.candidate](payload)
    after = _financial_digest(payload)
    output = {
        "candidate": args.candidate,
        "real_episode_records": len(payload["episode_records"]),
        "real_long_rows": len(payload["long_rows"]),
        "financial_oracle_before": before,
        "financial_oracle_after": after,
        "financial_oracle_unchanged": before == after,
        "adapter": adapter,
    }
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
