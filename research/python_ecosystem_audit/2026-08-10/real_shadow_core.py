"""Frozen real-cohort builder for the research-only OSS adapter shadow.

QuantRabbit owns all chronology, fills, costs, margin coverage and P/L.  The
optional packages consume the derived payload and never become a financial
oracle.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEED = 20_260_809
BOOTSTRAPS = 10_000
EMBARGO_SECONDS = 3_600
ANCHOR = "2026-07-09T07:46:03.151624347Z"
WINDOWS = (
    ("INITIAL_16D", "2026-06-23T07:46:03.151624347Z", ANCHOR),
    ("DOUBLE_32D", "2026-06-07T07:46:03.151624347Z", ANCHOR),
    ("QUADRUPLE_64D", "2026-05-06T07:46:03.151624347Z", ANCHOR),
)
FROZEN = {
    "research/historical_learning_admission/all_entry_episodes_v1.jsonl": "efcf6b0fb675050d6a08efc0119065e0874e50e1c51373a0c0fb61bb6ebd815e",
    "research/historical_learning_selection_rca/preregister_v1.json": "fc23c257bb96084c806f6b280fbe8b27742414637aeec74b2cac481d4891816f",
    "research/historical_learning_selection_rca/selection_rca_report_v1.json": "ca666943d3b206935d3eef7525ec1713f227d4ddbff09e8829e568b874c05fe5",
    "research/historical_learning_selection_rca/selection_predictions_v1.jsonl": "fe17aae2ea119dbe7af9c7e2dd671f7d566a9f3f2f3da41aa31b8ae020a306a5",
    "research/historical_learning_gapless_truth/report_v2.json": "12087b6e66c54ef9d307e27aa22943511a6cca9d289683d50022e6be2507fb43",
    "research/historical_learning_gapless_truth/episode_coverage_v2.jsonl": "5873ab0367c468111de8392f499280c0edf6c9d508abd742004a04fa9e61f56f",
}
METHOD_FIELDS = {
    "FROZEN_HGB": "frozen_hgb_selected",
    "A_COVERAGE_BINDING": "coverage_binding_selected",
    "B_COST_AWARE_ABSTAIN": "cost_aware_selected",
    "C_PAIR_SIDE_CALIBRATION": "pair_side_calibration_selected",
}
LONG_REQUIRED = (
    "episode_id", "source_sha", "decision_time", "pair", "timeframe",
    "regime", "strategy", "parameter_set", "cost_scenario",
    "exposure_state", "exit_policy", "viewpoint", "metric", "value",
    "uncertainty", "sample_count", "admission_status",
)
REAL_CUBE_AXES = (
    "window", "split", "timeframe", "pair", "regime", "method",
    "cost", "risk", "exit",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN.items():
        actual = sha256(REPO / relative)
        if actual != expected:
            raise RuntimeError(f"frozen input changed: {relative}: {actual} != {expected}")
        observed[relative] = actual
    prereg = json.loads((HERE / "preregister_real_shadow_v1.json").read_text(encoding="utf-8"))
    if prereg["checkpoint_git_head"] != "797a20d5a330ee726f5931691797a7bbd687d791":
        raise RuntimeError("checkpoint binding changed")
    return observed


def combined_source_sha(bindings: dict[str, str]) -> str:
    encoded = json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _price_action_features(episodes: list[dict[str, Any]], parse_time: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads((REPO / "research/historical_learning_gapless_truth/report_v2.json").read_text(encoding="utf-8"))
    if report["holdout_used"] is not False:
        raise RuntimeError("gapless report opened holdout")
    if report["source_boundary"] != {
        "cross_source_same_truth_claimed": False,
        "features": "DUKASCOPY_DATAFEED_TICK",
        "fills_and_labels": "OANDA_ACTUAL_AFTER_COST",
    }:
        raise RuntimeError("source boundary changed")
    pa = import_module(
        "real_shadow_price_action",
        REPO / "research/historical_learning_price_action_admission/run_price_action_admission.py",
    )
    pair_bars: dict[str, list[Any]] = {}
    lineage: dict[str, Any] = {}
    for pair, source in sorted(report["s5_sources"].items()):
        path = Path(source["path"]).resolve()
        if REPO.resolve() not in path.parents:
            raise RuntimeError(f"feature path escaped canonical repo: {pair}")
        actual = sha256(path)
        if actual != source["sha256"]:
            raise RuntimeError(f"Dukascopy derived source changed: {pair}")
        bars, audit = pa.load_bars(path, pair, parse_time)
        pair_bars[pair] = bars
        lineage[pair] = {
            "path": str(path.relative_to(REPO)),
            "sha256": actual,
            "complete_m5_bars": audit["complete_m5_bars"],
            "feature_only": True,
        }
    enriched, reasons = pa.attach_features(episodes, pair_bars, parse_time)
    return enriched, {
        "feature_source": "DUKASCOPY_DATAFEED_TICK",
        "execution_source": "OANDA_ACTUAL_AFTER_COST",
        "cross_source_fill_substitution": False,
        "coverage_reasons": reasons,
        "sources": lineage,
    }


def _bootstrap_mean_ci(values: list[float], seed_offset: int = 0) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(SEED + seed_offset)
    samples = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(BOOTSTRAPS))
    return [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]


def _drawdown(values: Iterable[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def _profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if losses:
        return gains / losses
    return None


def metrics(rows: list[dict[str, Any]], *, seed_offset: int = 0) -> dict[str, Any]:
    if not rows:
        return {
            "after_cost_net_jpy": None, "paired_lcb_jpy": None,
            "paired_ucb_jpy": None, "profit_factor": None,
            "max_drawdown_jpy": None, "margin_coverage": None,
            "turnover_units": None, "fill_validity": None,
            "unwind_validity": None, "sample_coverage": None,
            "trades_available": 0, "trades_selected": 0,
            "incremental_net_jpy": None,
        }
    applied = [float(row["actual_net_jpy"]) if row["selected"] else 0.0 for row in rows]
    baseline = [float(row["actual_net_jpy"]) for row in rows]
    deltas = [candidate - base for candidate, base in zip(applied, baseline)]
    ci = _bootstrap_mean_ci(deltas, seed_offset)
    selected = [row for row in rows if row["selected"]]
    margin_known = [row for row in selected if row["margin_evidence_known"]]
    return {
        "after_cost_net_jpy": sum(applied),
        "paired_lcb_jpy": ci[0],
        "paired_ucb_jpy": ci[1],
        "profit_factor": _profit_factor(applied),
        "max_drawdown_jpy": _drawdown(applied),
        "margin_coverage": len(margin_known) / len(selected) if selected else None,
        "turnover_units": sum(abs(float(row["units"])) for row in selected),
        "fill_validity": 1.0 if selected else None,
        "unwind_validity": 1.0 if selected else None,
        "sample_coverage": len(selected) / len(rows),
        "trades_available": len(rows),
        "trades_selected": len(selected),
        "incremental_net_jpy": sum(deltas),
    }


def _candidate_selection_maps() -> dict[tuple[str, str], dict[str, bool]]:
    output: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in read_jsonl(REPO / "research/historical_learning_selection_rca/selection_predictions_v1.jsonl"):
        key = (str(row["window_id"]), str(row["episode_id"]))
        for method, field in METHOD_FIELDS.items():
            output[key][method] = bool(row[field])
    gap_report = json.loads((REPO / "research/historical_learning_gapless_truth/report_v2.json").read_text(encoding="utf-8"))
    for row in gap_report["prediction_rows"]:
        key = (str(row["window_id"]), str(row["episode_id"]))
        output[key]["COVERAGE_MATCHED_METADATA_HGB"] = bool(row["metadata_selected"])
        output[key]["PRICE_ACTION_HGB"] = bool(row["price_action_selected"])
    return output


def _episode_record(window: str, split: str, method: str, row: dict[str, Any], selected: bool, source: str) -> dict[str, Any]:
    feature = row.get("price_action_features")
    regime = str(row.get("forecast_direction") or "MISSING")
    margin_known = bool(str(row["pair"]).endswith("_JPY") and row.get("intended_price") is not None)
    return {
        "window": window,
        "split": split,
        "episode_id": row["episode_id"],
        "decision_time": row["feature_at_utc"],
        "close_time": row["close_at_utc"],
        "pair": row["pair"],
        "side": row["side"],
        "regime": regime,
        "timeframe": "M5" if feature is not None else "DECISION_METADATA_ONLY",
        "method": method,
        "selected": selected,
        "actual_net_jpy": float(row["net_jpy"]),
        "financing_jpy": float(row.get("financing_jpy") or 0.0),
        "units": float(row["units"]),
        "margin_evidence_known": margin_known,
        "fill_valid": True,
        "unwind_valid": True,
        "cost_completeness": row["cost_completeness"],
        "feature_source": "DUKASCOPY_DATAFEED_TICK" if feature is not None else None,
        "execution_source": "OANDA_ACTUAL_AFTER_COST",
        "source_boundary_preserved": True,
        "price_action_features": feature,
        "source_sha": source,
    }


def build_episode_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bindings = verify_frozen()
    source = combined_source_sha(bindings)
    admission = import_module(
        "real_shadow_admission",
        REPO / "research/historical_learning_admission/run_admission.py",
    )
    episodes = [
        row for row in read_jsonl(REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl")
        if row.get("label_status") == "ACTUAL_AFTER_COST"
        and admission.parse_time(row["feature_at_utc"]) <= admission.parse_time(ANCHOR)
    ]
    if len(episodes) != 251:
        raise RuntimeError(f"expected 251 labeled episodes, observed {len(episodes)}")
    enriched, lineage = _price_action_features(episodes, admission.parse_time)
    selections = _candidate_selection_maps()
    records: list[dict[str, Any]] = []
    splits: dict[str, Any] = {}
    for window, start_text, end_text in WINDOWS:
        start, end = admission.parse_time(start_text), admission.parse_time(end_text)
        scoped = [row for row in enriched if start <= admission.parse_time(row["feature_at_utc"]) <= end]
        train, validation, purged = admission.split_rows(scoped)
        splits[window] = {
            "scoped": len(scoped), "train": len(train), "validation": len(validation),
            "purged": purged, "embargo_seconds": EMBARGO_SECONDS,
            "validation_start_utc": validation[0]["feature_at_utc"] if validation else None,
        }
        records.extend(_episode_record(window, "TRAIN", "ALL_TRADES", row, True, source) for row in train)
        for row in validation:
            records.append(_episode_record(window, "VALIDATION", "ALL_TRADES", row, True, source))
            for method, selected in sorted(selections.get((window, row["episode_id"]), {}).items()):
                records.append(_episode_record(window, "VALIDATION", method, row, selected, source))
    return records, {
        "bindings": bindings,
        "combined_source_sha": source,
        "lineage": lineage,
        "splits": splits,
        "holdout_read": False,
    }


def _metric_rows(group: list[dict[str, Any]], viewpoint: str, *, seed_offset: int) -> list[dict[str, Any]]:
    first = group[0]
    # Overall cells must reproduce the frozen QR financial oracle exactly.
    # Stratified diagnostic cells receive a stable offset so their bootstrap
    # streams are independent without changing the admitted comparison.
    effective_seed_offset = 0 if viewpoint == "overall" else seed_offset
    result = metrics(group, seed_offset=effective_seed_offset)
    risk = "MARGIN_COMPLETE" if result["margin_coverage"] == 1.0 else "MARGIN_INCOMPLETE"
    status = "EVALUATED" if first["split"] == "VALIDATION" and len(group) >= 30 else "DIAGNOSTIC_OR_INSUFFICIENT"
    shared = {
        "episode_id": "__aggregate__",
        "source_sha": first["source_sha"],
        "decision_time": max(row["decision_time"] for row in group),
        "pair": first["cube_pair"],
        "timeframe": first["cube_timeframe"],
        "regime": first["cube_regime"],
        "strategy": first["method"],
        "parameter_set": f"{first['method']}|{first['window']}|frozen",
        "cost_scenario": "OANDA_ACTUAL_AFTER_COST",
        "exposure_state": risk,
        "exit_policy": "ACTUAL_BROKER_CLOSE_OR_COUNTERFACTUAL_SKIP",
        "viewpoint": viewpoint,
        "uncertainty": {
            "method": "paired_episode_bootstrap_mean_delta",
            "samples": BOOTSTRAPS,
            "seed": SEED + effective_seed_offset,
            "paired_lcb_jpy_per_episode": result["paired_lcb_jpy"],
            "paired_ucb_jpy_per_episode": result["paired_ucb_jpy"],
            "opportunity_cost": "missing",
        },
        "sample_count": result["trades_available"],
        "admission_status": status,
        "window": first["window"],
        "split": first["split"],
        "method": first["method"],
        "cost": "OANDA_ACTUAL_AFTER_COST",
        "risk": risk,
        "exit": "ACTUAL_OR_SKIP",
    }
    values = {
        "after_cost_net_jpy": result["after_cost_net_jpy"],
        "lcb_jpy": result["paired_lcb_jpy"],
        "profit_factor": result["profit_factor"],
        "max_drawdown_jpy": result["max_drawdown_jpy"],
        "margin_coverage": result["margin_coverage"],
        "turnover": result["turnover_units"],
        "fill_validity": result["fill_validity"],
        "unwind_validity": result["unwind_validity"],
        "sample_coverage": result["sample_coverage"],
        "incremental_net_jpy": result["incremental_net_jpy"],
    }
    return [dict(shared, metric=metric, value=value) for metric, value in values.items()]


def build_long_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    # Overall cells are the only cells admitted to Pareto/financial comparison.
    for row in records:
        overall = dict(row, cube_pair="ALL", cube_regime="ALL", cube_timeframe="MIXED")
        groups[(row["window"], row["split"], row["method"], "ALL", "ALL", "MIXED", "overall")].append(overall)
        # Stratified cells expose pair/regime interactions without treating missing as zero.
        stratified = dict(row, cube_pair=row["pair"], cube_regime=row["regime"], cube_timeframe=row["timeframe"])
        groups[(row["window"], row["split"], row["method"], row["pair"], row["regime"], row["timeframe"], "pair_regime")].append(stratified)
    output: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))):
        output.extend(_metric_rows(group, key[-1], seed_offset=index + 1))
    for index, row in enumerate(output):
        missing = [field for field in LONG_REQUIRED if field not in row]
        if missing:
            raise RuntimeError(f"long row {index} missing {missing}")
        if row["value"] is not None and (not isinstance(row["value"], (int, float)) or not math.isfinite(float(row["value"]))):
            raise RuntimeError(f"long row {index} has non-finite value")
    return output


def candidate_summaries(long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for row in long_rows:
        if row["viewpoint"] != "overall" or row["split"] != "VALIDATION":
            continue
        key = (row["window"], row["method"])
        grouped[key].update({"window": row["window"], "split": row["split"], "method": row["method"]})
        grouped[key][row["metric"]] = row["value"]
        grouped[key]["sample_count"] = row["sample_count"]
        grouped[key]["risk"] = row["risk"]
    return list(grouped.values())


def financial_invariants(records: list[dict[str, Any]]) -> dict[str, Any]:
    selection_report = json.loads((REPO / "research/historical_learning_selection_rca/selection_rca_report_v1.json").read_text(encoding="utf-8"))
    report_windows = {row["id"]: row for row in selection_report["windows"]}
    gap_report = json.loads((REPO / "research/historical_learning_gapless_truth/report_v2.json").read_text(encoding="utf-8"))
    gap_windows = {row["id"]: row for row in gap_report["windows"]}
    comparisons: list[dict[str, Any]] = []
    method_sources = {
        "FROZEN_HGB": (report_windows, "candidates"),
        "A_COVERAGE_BINDING": (report_windows, "candidates"),
        "B_COST_AWARE_ABSTAIN": (report_windows, "candidates"),
        "C_PAIR_SIDE_CALIBRATION": (report_windows, "candidates"),
        "COVERAGE_MATCHED_METADATA_HGB": (gap_windows, None),
        "PRICE_ACTION_HGB": (gap_windows, None),
    }
    for window, _, _ in WINDOWS:
        window_records = [row for row in records if row["window"] == window and row["split"] == "VALIDATION"]
        methods = sorted({row["method"] for row in window_records})
        for method in methods:
            actual = metrics([row for row in window_records if row["method"] == method])
            if method == "ALL_TRADES":
                expected = report_windows[window]["ALL_TRADES"]
            else:
                source, nested = method_sources[method]
                if nested is None:
                    expected = source[window][method]
                else:
                    expected = source[window][nested][method]
            comparisons.append({
                "window": window,
                "method": method,
                "net_diff_jpy": actual["after_cost_net_jpy"] - float(expected["net_jpy"]),
                "retention_diff": actual["sample_coverage"] - float(expected["retention_ratio"]),
                "lcb_diff_jpy": None if expected.get("paired_lcb_jpy") is None else actual["paired_lcb_jpy"] - float(expected["paired_lcb_jpy"]),
            })
    max_net = max(abs(row["net_diff_jpy"]) for row in comparisons)
    max_retention = max(abs(row["retention_diff"]) for row in comparisons)
    max_lcb = max(abs(row["lcb_diff_jpy"]) for row in comparisons if row["lcb_diff_jpy"] is not None)
    return {
        "comparisons": comparisons,
        "max_abs_net_diff_jpy": max_net,
        "max_abs_retention_diff": max_retention,
        "max_abs_lcb_diff_jpy": max_lcb,
        "exact_with_tolerance_1e_9": max(max_net, max_retention, max_lcb) < 1e-9,
        "financial_oracle": "OANDA_ACTUAL_AFTER_COST",
        "opportunity_cost": "missing",
        "holdout_read": False,
    }


def build_payload() -> dict[str, Any]:
    records, manifest = build_episode_records()
    long_rows = build_long_rows(records)
    invariants = financial_invariants(records)
    if not invariants["exact_with_tolerance_1e_9"]:
        raise RuntimeError("financial invariant differs from frozen reports")
    return {
        "contract": "python_ecosystem_real_cohort_shadow_payload_v1",
        "preregister_sha256": sha256(HERE / "preregister_real_shadow_v1.json"),
        "manifest": manifest,
        "episode_records": records,
        "long_rows": long_rows,
        "candidate_summaries": candidate_summaries(long_rows),
        "financial_invariants": invariants,
        "cube_axes": list(REAL_CUBE_AXES),
        "holdout_read": False,
        "permissions": {"live": False, "paper": False, "broker_order": False, "deploy": False},
    }


def logical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
