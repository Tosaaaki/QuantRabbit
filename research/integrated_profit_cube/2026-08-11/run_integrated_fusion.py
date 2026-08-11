#!/usr/bin/env python3
"""Research-only integrated profit cube on the frozen V2 cohort.

This program never imports broker/live gateways.  It rebases every economic
comparison to TRADE_CASHFLOW_FINANCIAL_ORACLE_V2, keeps incomplete exit/hedge
cells null, and uses decision-time systems as bounded size modifiers rather
than winner-discarding hard vetoes.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np
import xarray as xr
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
from mapie.regression import SplitConformalRegressor
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREREG = HERE / "preregister_v1.json"
SEED = 20260811
BOOTSTRAPS = 4000
WINDOWS = ("INITIAL_16D", "DOUBLE_32D", "QUADRUPLE_64D")
COMPONENTS = (
    "GROUP_RELATIVE_SIZE",
    "PRICE_ACTION_RIDGE_SIZE",
    "CAUSAL_FORECAST_RANK",
    "CAUSAL_PRICE_ACTION_RANK",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def load_growth_module() -> Any:
    path = ROOT / "research/monthly_3x_growth_engine/2026-08-10/run_growth_engine.py"
    spec = importlib.util.spec_from_file_location("qr_growth_v1_for_integrated_cube", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profit_factor(values: Iterable[float]) -> float | None:
    values = list(values)
    gain = sum(value for value in values if value > 0)
    loss = -sum(value for value in values if value < 0)
    if loss == 0:
        return None if gain == 0 else math.inf
    return gain / loss


def percentile_rank(values: list[float], value: float) -> float:
    ordered = np.asarray(sorted(values), dtype=float)
    if not len(ordered):
        return 0.5
    left = int(np.searchsorted(ordered, value, side="left"))
    right = int(np.searchsorted(ordered, value, side="right"))
    return (left + right) / (2.0 * len(ordered))


def bootstrap_lcb(deltas: list[float], key: str, q: float = 0.05) -> float | None:
    if not deltas:
        return None
    seed = SEED ^ int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(BOOTSTRAPS))
    return float(means[int(q * (len(means) - 1))])


def screening_lcb(deltas: list[float]) -> float | None:
    """Fast TRAIN-grid screen; final candidates receive the bootstrap oracle."""
    if not deltas:
        return None
    array = np.asarray(deltas, dtype=float)
    if len(array) == 1:
        return float(array[0])
    return float(array.mean() - 1.6448536269514722 * array.std(ddof=1) / math.sqrt(len(array)))


def simplex_weights(step_units: int = 4) -> list[tuple[float, float, float, float]]:
    result = []
    for first in range(step_units + 1):
        for second in range(step_units + 1 - first):
            for third in range(step_units + 1 - first - second):
                fourth = step_units - first - second - third
                result.append(tuple(value / step_units for value in (first, second, third, fourth)))
    return result


def concurrency_at_entry(episodes: list[Any]) -> dict[str, int]:
    ordered = sorted(episodes, key=lambda episode: (episode.fill_at, episode.episode_id))
    result: dict[str, int] = {}
    for episode in ordered:
        result[episode.episode_id] = sum(
            other.fill_at < episode.fill_at < other.close_at for other in ordered
        )
    return result


def make_parameter_id(weights: tuple[float, ...], strength: float, overlay: str, concurrency: str | int) -> str:
    payload = {"weights": weights, "strength": strength, "overlay": overlay, "concurrency": concurrency}
    return "ipc:" + logical_sha(payload)[:16]


def parameter_neighbors(left: dict[str, Any], right: dict[str, Any]) -> bool:
    differences = 0
    if left["strength"] != right["strength"]:
        differences += 1
    if left["overlay"] != right["overlay"]:
        differences += 1
    if left["concurrency"] != right["concurrency"]:
        differences += 1
    weight_distance = sum(abs(a - b) for a, b in zip(left["weights"], right["weights"]))
    if weight_distance:
        if abs(weight_distance - 0.5) > 1e-12:
            return False
        differences += 1
    return differences == 1


def connected_ids(rows: list[dict[str, Any]]) -> set[str]:
    by_id = {row["parameter_id"]: row for row in rows}
    graph = {key: set() for key in by_id}
    keys = sorted(by_id)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            if parameter_neighbors(by_id[left], by_id[right]):
                graph[left].add(right)
                graph[right].add(left)
    accepted: set[str] = set()
    unseen = set(keys)
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = {start}
        while stack:
            current = stack.pop()
            for neighbor in graph[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= 3:
            accepted.update(component)
    return accepted


def main() -> int:
    prereg = json.loads(PREREG.read_text())
    for source in prereg["frozen_sources"].values():
        path = ROOT / source["path"]
        actual = sha256(path)
        if actual != source["sha256"]:
            raise RuntimeError(f"source hash mismatch: {path}: {actual}")

    growth = load_growth_module()
    labels = {
        row["episode_id"]: row
        for row in read_jsonl(ROOT / prereg["frozen_sources"]["financial_labels_v2"]["path"])
    }
    paths = {
        row["episode_id"]: row
        for row in read_jsonl(ROOT / prereg["frozen_sources"]["path_metrics"]["path"])
    }
    evidence = {
        row["decision_id"]: row
        for row in read_jsonl(ROOT / prereg["frozen_sources"]["execution_evidence"]["path"])
    }
    payload = json.loads((ROOT / prereg["frozen_sources"]["split_and_features"]["path"]).read_text())
    inference_rows = read_jsonl(ROOT / prereg["frozen_sources"]["system_inference"]["path"])
    multiplier_rows = read_jsonl(ROOT / prereg["frozen_sources"]["decision_multipliers"]["path"])
    exit_rows = read_jsonl(ROOT / prereg["frozen_sources"]["exit_replay"]["path"])

    if set(labels) != set(paths) or set(labels) != set(evidence) or len(labels) != 251:
        raise RuntimeError("V2 labels/path/evidence decision ids must match exactly at 251")

    inference: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in inference_rows:
        inference[row["episode_id"]][row["system_id"]] = row

    multiplier = {
        (row["window"], row["split"], row["episode_id"], row["policy"]): float(row["decision_multiplier"])
        for row in multiplier_rows
    }

    feature_record: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload["episode_records"]:
        if row["method"] == "ALL_TRADES":
            feature_record[(row["window"], row["split"], row["episode_id"])] = row

    membership: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for episode_id, label in labels.items():
        for window in WINDOWS:
            split = evidence[episode_id]["splits"][window]
            if split not in {"TRAIN", "VALIDATION"}:
                continue
            record = feature_record[(window, split, episode_id)]
            path = paths[episode_id]
            membership[(window, split)].append(
                growth.Episode(
                    episode_id=episode_id,
                    fill_at=growth.utc(label["fill_at_utc"]),
                    close_at=growth.utc(label["close_at_utc"]),
                    pair=label["pair"],
                    side=label["side"],
                    regime=record.get("regime") or "MISSING",
                    units=float(label["units"]),
                    corrected_net_jpy=float(label["corrected_net_jpy"]),
                    initial_margin_jpy=float(path["entry_actual_initial_margin_jpy"]),
                    price_action_features={
                        key: float(value)
                        for key, value in (record.get("price_action_features") or {}).items()
                        if value is not None
                    },
                )
            )

    expected_counts = {
        ("INITIAL_16D", "TRAIN"): 13,
        ("INITIAL_16D", "VALIDATION"): 12,
        ("DOUBLE_32D", "TRAIN"): 43,
        ("DOUBLE_32D", "VALIDATION"): 31,
        ("QUADRUPLE_64D", "TRAIN"): 145,
        ("QUADRUPLE_64D", "VALIDATION"): 101,
    }
    if {key: len(value) for key, value in membership.items()} != expected_counts:
        raise RuntimeError("split membership differs from preregistered counts")

    component_values: dict[tuple[str, str, str], dict[str, float | None]] = {}
    cube_rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        train = membership[(window, "TRAIN")]
        train_signal_values: dict[str, list[float]] = {}
        for system_id in ("forecast", "price_action"):
            train_signal_values[system_id] = sorted(
                float(inference[episode.episode_id][system_id]["probability_or_score"])
                for episode in train
                if inference[episode.episode_id].get(system_id, {}).get("probability_or_score") is not None
            )
        for split in ("TRAIN", "VALIDATION"):
            for episode in membership[(window, split)]:
                values: dict[str, float | None] = {
                    "GROUP_RELATIVE_SIZE": multiplier[(window, split, episode.episode_id, "GROUP_RELATIVE_SIZE")],
                    "PRICE_ACTION_RIDGE_SIZE": multiplier[(window, split, episode.episode_id, "PRICE_ACTION_RIDGE_SIZE")],
                }
                for system_id, component in (
                    ("forecast", "CAUSAL_FORECAST_RANK"),
                    ("price_action", "CAUSAL_PRICE_ACTION_RANK"),
                ):
                    row = inference[episode.episode_id].get(system_id)
                    score = row.get("probability_or_score") if row else None
                    values[component] = (
                        0.5 + percentile_rank(train_signal_values[system_id], float(score))
                        if score is not None
                        else None
                    )
                component_values[(window, split, episode.episode_id)] = values
                for component, value in values.items():
                    cube_rows.append(
                        {
                            "decision_id": episode.episode_id,
                            "window": window,
                            "split": split,
                            "pair": episode.pair,
                            "side": episode.side,
                            "regime": episode.regime,
                            "system_id": component,
                            "stage": "ENTRY_OR_SIZING",
                            "value": value,
                            "missing_not_zero": value is None,
                            "baseline_actual_after_cost_net_jpy": episode.corrected_net_jpy,
                            "candidate_actual_after_cost_net_jpy": episode.corrected_net_jpy,
                            "admission_status": "PASSTHROUGH_WHEN_MISSING",
                        }
                    )

    # Explicit null cells keep exit/hedge evidence gaps visible in the same table.
    exit_by_key = {(row["window"], row["split"], row["episode_id"], row["arm"]): row for row in exit_rows}
    for window in WINDOWS:
        for split in ("TRAIN", "VALIDATION"):
            for episode in membership[(window, split)]:
                for arm in prereg["exit_and_hedge"]["exit_arms"]:
                    row = exit_by_key[(window, split, episode.episode_id, arm)]
                    cube_rows.append(
                        {
                            "decision_id": episode.episode_id,
                            "window": window,
                            "split": split,
                            "pair": episode.pair,
                            "side": episode.side,
                            "regime": episode.regime,
                            "system_id": arm,
                            "stage": "EXIT",
                            "value": row["candidate_actual_after_cost_net_jpy"],
                            "missing_not_zero": row["candidate_actual_after_cost_net_jpy"] is None,
                            "baseline_actual_after_cost_net_jpy": episode.corrected_net_jpy,
                            "candidate_actual_after_cost_net_jpy": row["candidate_actual_after_cost_net_jpy"],
                            "admission_status": row["admission_status"],
                        }
                    )
                for arm in prereg["exit_and_hedge"]["hedge_arms"]:
                    cube_rows.append(
                        {
                            "decision_id": episode.episode_id,
                            "window": window,
                            "split": split,
                            "pair": episode.pair,
                            "side": episode.side,
                            "regime": episode.regime,
                            "system_id": arm,
                            "stage": "HEDGE",
                            "value": None,
                            "missing_not_zero": True,
                            "baseline_actual_after_cost_net_jpy": episode.corrected_net_jpy,
                            "candidate_actual_after_cost_net_jpy": None,
                            "admission_status": "NOT_EVALUABLE_DUAL_LEG_COST_MARGIN_UNWIND_MISSING",
                        }
                    )

    weights_grid = simplex_weights()
    parameter_rows = [
        {
            "weights": list(weights),
            "strength": strength,
            "overlay": overlay,
            "concurrency": concurrency,
            "parameter_id": make_parameter_id(weights, strength, overlay, concurrency),
        }
        for weights in weights_grid
        for strength in (0.5, 1.0)
        for overlay in ("NONE", "INVENTORY", "TECHNICAL", "MEAN")
        for concurrency in ("NONE", 3, 6, 9)
    ]

    grid_rows: list[dict[str, Any]] = []
    delta_cache: dict[tuple[str, str, str], list[float]] = {}
    decision_rows: list[dict[str, Any]] = []
    decision_multiplier_cache: dict[tuple[str, str, str, str], float] = {}
    for window in WINDOWS:
        for split in ("TRAIN", "VALIDATION"):
            episodes = membership[(window, split)]
            baseline_metrics, baseline_scaled = growth.simulate(
                episodes, {episode.episode_id: 1.0 for episode in episodes}, 1.0, 0.75
            )
            open_count = concurrency_at_entry(episodes)
            for parameter in parameter_rows:
                applied: dict[str, float] = {}
                for episode in episodes:
                    values = component_values[(window, split, episode.episode_id)]
                    components = [1.0 if values[name] is None else float(values[name]) for name in COMPONENTS]
                    combined = sum(weight * value for weight, value in zip(parameter["weights"], components))
                    result = 1.0 + float(parameter["strength"]) * (combined - 1.0)
                    if parameter["overlay"] != "NONE":
                        inventory = multiplier[(window, split, episode.episode_id, "INVENTORY_CAP_V2_RELABELED")]
                        technical = multiplier[(window, split, episode.episode_id, "TECHNICAL_DISSENT_CAP_V3_RELABELED")]
                        overlay = {
                            "INVENTORY": inventory,
                            "TECHNICAL": technical,
                            "MEAN": 0.5 * (inventory + technical),
                        }[parameter["overlay"]]
                        result = 0.5 * (result + overlay)
                    if parameter["concurrency"] != "NONE" and open_count[episode.episode_id] >= int(parameter["concurrency"]):
                        result *= 0.5
                    result = min(1.5, max(0.5, result))
                    applied[episode.episode_id] = result
                    decision_multiplier_cache[(window, split, parameter["parameter_id"], episode.episode_id)] = result
                metrics, scaled = growth.simulate(episodes, applied, 1.0, 0.75)
                deltas = [scaled[episode.episode_id] - baseline_scaled[episode.episode_id] for episode in episodes]
                delta_cache[(window, split, parameter["parameter_id"])] = deltas
                row = {
                    **parameter,
                    "window": window,
                    "split": split,
                    "episodes": len(episodes),
                    "changed": sum(abs(applied[episode.episode_id] - 1.0) > 1e-12 for episode in episodes),
                    "baseline_after_cost_net_jpy": baseline_metrics["after_cost_net_jpy"],
                    "after_cost_net_jpy": metrics["after_cost_net_jpy"],
                    "incremental_net_jpy": sum(deltas),
                    "screening_lcb_normal_95_jpy_per_episode": screening_lcb(deltas),
                    "paired_bootstrap_lcb_one_sided_95_jpy_per_episode": None,
                    "profit_factor": metrics["profit_factor"],
                    "baseline_profit_factor": baseline_metrics["profit_factor"],
                    "realized_max_drawdown_jpy": metrics["realized_max_drawdown_jpy"],
                    "baseline_realized_max_drawdown_jpy": baseline_metrics["realized_max_drawdown_jpy"],
                    "cohort_margin_peak_jpy": metrics["cohort_margin_peak_jpy"],
                    "baseline_cohort_margin_peak_jpy": baseline_metrics["cohort_margin_peak_jpy"],
                    "account_margin_evaluable": False,
                    "exit_evaluable_changed": 0,
                    "hedge_evaluable_changed": 0,
                    "holdout_used": False,
                }
                grid_rows.append(row)

    # TRAIN-only feasibility and plateau. Validation never changes the selected ids.
    train_feasible_by_window: dict[str, list[dict[str, Any]]] = {}
    train_plateau_by_window: dict[str, set[str]] = {}
    for window in WINDOWS:
        rows = [row for row in grid_rows if row["window"] == window and row["split"] == "TRAIN"]
        feasible = [
            row
            for row in rows
            if row["after_cost_net_jpy"] > 0
            and row["profit_factor"] is not None
            and row["profit_factor"] > 1
            and row["incremental_net_jpy"] > 0
            and row["screening_lcb_normal_95_jpy_per_episode"] is not None
            and row["screening_lcb_normal_95_jpy_per_episode"] > 0
            and row["realized_max_drawdown_jpy"] <= row["baseline_realized_max_drawdown_jpy"]
            and row["changed"] >= 10
        ]
        train_feasible_by_window[window] = feasible
        train_plateau_by_window[window] = connected_ids(feasible)

    frozen_ids = set.intersection(*(train_plateau_by_window[window] for window in WINDOWS))

    # Deterministic TRAIN-only champion even when the strict plateau is empty.
    train_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grid_rows:
        if row["split"] == "TRAIN":
            train_by_id[row["parameter_id"]].append(row)
    champion_id = max(
        train_by_id,
        key=lambda key: (
            min(row["incremental_net_jpy"] for row in train_by_id[key]),
            sum(row["incremental_net_jpy"] for row in train_by_id[key]),
            -sum(row["realized_max_drawdown_jpy"] for row in train_by_id[key]),
            key,
        ),
    )
    evaluated_ids = frozen_ids or {champion_id}

    # The expensive paired bootstrap is reserved for parameters frozen before
    # validation readout.  This preserves the preregistered acceptance oracle
    # without using validation to choose which grid points receive inference.
    for row in grid_rows:
        if row["parameter_id"] in evaluated_ids:
            row["paired_bootstrap_lcb_one_sided_95_jpy_per_episode"] = bootstrap_lcb(
                delta_cache[(row["window"], row["split"], row["parameter_id"])],
                f"{row['window']}:{row['split']}:{row['parameter_id']}",
            )

    for row in grid_rows:
        row["train_feasible"] = row["parameter_id"] in {
            item["parameter_id"] for item in train_feasible_by_window[row["window"]]
        }
        row["train_connected_plateau"] = row["parameter_id"] in train_plateau_by_window[row["window"]]
        row["frozen_cross_window_candidate"] = row["parameter_id"] in frozen_ids
        row["train_only_champion"] = row["parameter_id"] == champion_id
        row["primary_evaluation_candidate"] = row["parameter_id"] in evaluated_ids

    # xarray is a real cube consumer and proves null cells are not converted to zero.
    cube_decisions = sorted({row["decision_id"] for row in cube_rows})
    cube_systems = sorted({row["system_id"] for row in cube_rows})
    cube_windows = list(WINDOWS)
    cube_splits = ["TRAIN", "VALIDATION"]
    values = np.full(
        (len(cube_windows), len(cube_splits), len(cube_decisions), len(cube_systems)),
        np.nan,
        dtype=float,
    )
    indexes = {
        "window": {value: index for index, value in enumerate(cube_windows)},
        "split": {value: index for index, value in enumerate(cube_splits)},
        "decision": {value: index for index, value in enumerate(cube_decisions)},
        "system": {value: index for index, value in enumerate(cube_systems)},
    }
    for row in cube_rows:
        if row["value"] is not None:
            values[
                indexes["window"][row["window"]],
                indexes["split"][row["split"]],
                indexes["decision"][row["decision_id"]],
                indexes["system"][row["system_id"]],
            ] = float(row["value"])
    data_array = xr.DataArray(
        values,
        dims=("window", "split", "decision_id", "system_id"),
        coords={
            "window": cube_windows,
            "split": cube_splits,
            "decision_id": cube_decisions,
            "system_id": cube_systems,
        },
    )
    xarray_report = {
        "dims": {name: int(size) for name, size in data_array.sizes.items()},
        "nan_cells": int(np.isnan(data_array.values).sum()),
        "non_null_source_rows": sum(row["value"] is not None for row in cube_rows),
        "non_null_cube_cells": int(np.isfinite(data_array.values).sum()),
        "null_preserved": int(np.isfinite(data_array.values).sum()) == sum(row["value"] is not None for row in cube_rows),
    }

    # SALib uses only 64d TRAIN and is diagnostic; it never selects from VALIDATION.
    train64 = membership[("QUADRUPLE_64D", "TRAIN")]
    salib_problem = {
        "num_vars": 4,
        "names": list(COMPONENTS),
        "bounds": [[0.0, 1.0]] * 4,
    }
    salib_samples = sobol_sample.sample(salib_problem, 64, calc_second_order=False, seed=SEED)
    salib_outputs = []
    for sample in salib_samples:
        normalized = np.asarray(sample, dtype=float)
        normalized = normalized / normalized.sum() if normalized.sum() else np.full(4, 0.25)
        applied = {}
        for episode in train64:
            components = component_values[("QUADRUPLE_64D", "TRAIN", episode.episode_id)]
            vector = np.asarray([1.0 if components[name] is None else components[name] for name in COMPONENTS])
            applied[episode.episode_id] = float(np.clip(normalized @ vector, 0.5, 1.5))
        metrics, _ = growth.simulate(train64, applied, 1.0, 0.75)
        salib_outputs.append(metrics["after_cost_net_jpy"])
    salib_result = sobol_analyze.analyze(
        salib_problem,
        np.asarray(salib_outputs),
        calc_second_order=False,
        num_resamples=128,
        seed=SEED,
        print_to_console=False,
    )
    salib_report = {
        "training_only": True,
        "samples": len(salib_outputs),
        "first_order": {name: float(value) for name, value in zip(COMPONENTS, salib_result["S1"])},
        "total_order": {name: float(value) for name, value in zip(COMPONENTS, salib_result["ST"])},
    }

    # pymoo consumes TRAIN aggregate candidates only; ids are frozen before validation readout.
    aggregate_train = []
    for key, rows in train_by_id.items():
        aggregate_train.append(
            {
                "parameter_id": key,
                "incremental_net_jpy": sum(row["incremental_net_jpy"] for row in rows),
                "max_drawdown_jpy": max(row["realized_max_drawdown_jpy"] for row in rows),
                "profit_factor_floor": min(float(row["profit_factor"] or 0.0) for row in rows),
            }
        )
    objectives = np.asarray(
        [
            [-row["incremental_net_jpy"], row["max_drawdown_jpy"], -row["profit_factor_floor"]]
            for row in aggregate_train
        ],
        dtype=float,
    )
    pareto_indexes = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    pareto_ids = sorted(aggregate_train[int(index)]["parameter_id"] for index in pareto_indexes)

    # MAPIE fits/calibrates on TRAIN only. It diagnoses interval width/coverage on 64d validation.
    def causal_vector(window: str, split: str, episode: Any) -> list[float]:
        values = component_values[(window, split, episode.episode_id)]
        return [1.0 if values[name] is None else float(values[name]) for name in COMPONENTS]

    train_sorted = sorted(train64, key=lambda episode: episode.fill_at)
    validation64 = sorted(membership[("QUADRUPLE_64D", "VALIDATION")], key=lambda episode: episode.fill_at)
    cut = max(1, int(len(train_sorted) * 0.6))
    fit_rows, conform_rows = train_sorted[:cut], train_sorted[cut:]
    model = SplitConformalRegressor(estimator=Ridge(alpha=10.0), confidence_level=0.9, prefit=False)
    model.fit(
        np.asarray([causal_vector("QUADRUPLE_64D", "TRAIN", episode) for episode in fit_rows]),
        np.asarray([episode.corrected_net_jpy / episode.units * 1000.0 for episode in fit_rows]),
    )
    model.conformalize(
        np.asarray([causal_vector("QUADRUPLE_64D", "TRAIN", episode) for episode in conform_rows]),
        np.asarray([episode.corrected_net_jpy / episode.units * 1000.0 for episode in conform_rows]),
    )
    point, intervals = model.predict_interval(
        np.asarray([causal_vector("QUADRUPLE_64D", "VALIDATION", episode) for episode in validation64])
    )
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    actual = np.asarray([episode.corrected_net_jpy / episode.units * 1000.0 for episode in validation64])
    mapie_report = {
        "training_only_fit_and_calibration": True,
        "fit_n": len(fit_rows),
        "conformal_n": len(conform_rows),
        "validation_n": len(validation64),
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_interval_width_jpy_per_1000u": float(np.mean(upper - lower)),
        "mean_prediction_jpy_per_1000u": float(np.mean(point)),
        "validation_used_for_evaluation_only": True,
    }

    validation_evaluations = {
        parameter_id: {
            window: next(
                row
                for row in grid_rows
                if row["parameter_id"] == parameter_id and row["window"] == window and row["split"] == "VALIDATION"
            )
            for window in WINDOWS
        }
        for parameter_id in sorted(evaluated_ids)
    }
    strict_pass = []
    for parameter_id, windows in validation_evaluations.items():
        primary = windows["QUADRUPLE_64D"]
        if (
            all(row["after_cost_net_jpy"] > 0 and row["incremental_net_jpy"] > 0 for row in windows.values())
            and primary["paired_bootstrap_lcb_one_sided_95_jpy_per_episode"] is not None
            and primary["paired_bootstrap_lcb_one_sided_95_jpy_per_episode"] > 0
            and primary["realized_max_drawdown_jpy"] <= primary["baseline_realized_max_drawdown_jpy"]
            and primary["account_margin_evaluable"]
            and primary["exit_evaluable_changed"] >= 30
        ):
            strict_pass.append(parameter_id)

    exit64 = [
        row
        for row in exit_rows
        if row["window"] == "QUADRUPLE_64D" and row["split"] == "VALIDATION" and row["arm"] != "BASELINE"
    ]
    exit_summary = {
        arm: {
            "rows": sum(row["arm"] == arm for row in exit64),
            "changed": sum(row["arm"] == arm and row["changed"] is True for row in exit64),
            "after_cost_evaluable": sum(
                row["arm"] == arm and row["candidate_actual_after_cost_net_jpy"] is not None for row in exit64
            ),
            "admission": "NOT_EVALUABLE_STRICT_PATH_COST_MARGIN_UNWIND",
        }
        for arm in prereg["exit_and_hedge"]["exit_arms"]
    }

    raw_baseline64 = sum(
        episode.corrected_net_jpy for episode in membership[("QUADRUPLE_64D", "VALIDATION")]
    )
    baseline_cap64 = next(
        row
        for row in grid_rows
        if row["window"] == "QUADRUPLE_64D"
        and row["split"] == "VALIDATION"
        and row["parameter_id"] == parameter_rows[0]["parameter_id"]
    )["baseline_after_cost_net_jpy"]
    report = {
        "contract": prereg["contract"],
        "preregister_sha256": sha256(PREREG),
        "holdout_used": False,
        "source_hashes_verified": True,
        "decision_ids": len(labels),
        "cube_rows": len(cube_rows),
        "candidate_grid_rows": len(grid_rows),
        "v2_baseline": {
            "raw_64d_validation_after_cost_net_jpy": raw_baseline64,
            "cohort_margin_cap_75pct_64d_validation_after_cost_net_jpy": baseline_cap64,
            "raw_64d_validation_profit_factor": profit_factor(
                episode.corrected_net_jpy for episode in membership[("QUADRUPLE_64D", "VALIDATION")]
            ),
        },
        "xarray": xarray_report,
        "salib": salib_report,
        "pymoo": {"training_only": True, "pareto_candidate_count": len(pareto_ids), "pareto_ids": pareto_ids},
        "mapie": mapie_report,
        "train_feasible_counts": {window: len(train_feasible_by_window[window]) for window in WINDOWS},
        "train_plateau_counts": {window: len(train_plateau_by_window[window]) for window in WINDOWS},
        "cross_window_frozen_candidates": sorted(frozen_ids),
        "train_only_champion": champion_id,
        "validation_evaluations": validation_evaluations,
        "strict_pass_candidates": strict_pass,
        "exit_summary_64d_validation": exit_summary,
        "hedge_status": "NOT_EVALUABLE_DUAL_LEG_COST_MARGIN_UNWIND_MISSING",
        "account_margin_status": "NOT_EVALUABLE_ACCOUNT_NETTING_AND_EXTERNAL_INVENTORY_MISSING",
        "conclusion": "ADOPTABLE_INCREMENTAL_EDGE" if strict_pass else "BASELINE_POSITIVE_INTEGRATED_IMPROVEMENT_NOT_YET_ADMISSIBLE",
        "reason": (
            "V2 baseline is positive, but no candidate can satisfy the preregistered incremental LCB, "
            "cross-window plateau, full changed-row path/cost/margin/unwind coverage, and account margin gates together."
        ),
    }

    for row in cube_rows:
        row["row_sha256"] = logical_sha(row)
    for row in grid_rows:
        row["row_sha256"] = logical_sha(row)
    write_jsonl(HERE / "canonical_decision_cube_v1.jsonl", cube_rows)
    write_jsonl(HERE / "candidate_grid_v1.jsonl", grid_rows)
    write_json(HERE / "report_v1.json", report)
    manifest = {
        "contract": "INTEGRATED_PROFIT_CUBE_RUN_MANIFEST_V1",
        "preregister_sha256": sha256(PREREG),
        "outputs": {
            filename: sha256(HERE / filename)
            for filename in ("canonical_decision_cube_v1.jsonl", "candidate_grid_v1.jsonl", "report_v1.json")
        },
    }
    write_json(HERE / "run_manifest_v1.json", manifest)
    print(json.dumps({"conclusion": report["conclusion"], "champion": champion_id, "strict_pass": strict_pass}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
