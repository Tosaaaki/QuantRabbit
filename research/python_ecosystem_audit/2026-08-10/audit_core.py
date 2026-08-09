"""Small, dependency-light adapters for the QR ecosystem audit.

The module intentionally owns the truth boundary. Optional OSS packages are
probes/adapters only; their default fills, PnL, or time handling are never
used as the financial oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_FIELDS = (
    "episode_id", "source_sha", "decision_time", "pair", "timeframe",
    "regime", "strategy", "parameter_set", "cost_scenario",
    "exposure_state", "exit_policy", "viewpoint", "metric", "value",
    "uncertainty", "sample_count", "admission_status",
)
CUBE_AXES = ("split", "timeframe", "pair", "regime", "method", "cost", "risk", "exit")
METRICS = (
    "after_cost_net_jpy", "lcb_jpy", "profit_factor", "max_drawdown_jpy",
    "margin_coverage", "turnover", "fill_validity", "unwind_validity",
    "sample_coverage",
)


def _json_key(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def validate_long_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate schema and preserve missing values as ``None``."""
    out: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {index} missing required fields: {missing}")
        if row["sample_count"] is None or int(row["sample_count"]) < 0:
            raise ValueError(f"row {index} has invalid sample_count")
        if row["value"] is not None and not isinstance(row["value"], (int, float)):
            raise ValueError(f"row {index} value must be numeric or null")
        out.append(row)
    return out


@dataclass(frozen=True)
class LabeledCube:
    dims: tuple[str, ...]
    coords: dict[str, list[Any]]
    values: dict[str, dict[str, float | None]]
    sample_counts: dict[str, dict[str, int]]

    def value(self, metric: str, **coords: Any) -> float | None:
        key = _json_key([coords[dim] for dim in self.dims])
        return self.values.get(metric, {}).get(key)

    def to_json(self) -> dict[str, Any]:
        return {
            "dims": list(self.dims),
            "coords": self.coords,
            "values": self.values,
            "sample_counts": self.sample_counts,
            "missing_is_null_not_zero": True,
        }


def build_cube(rows: Iterable[Mapping[str, Any]]) -> LabeledCube:
    """Build a labelled sparse cube without materialising missing cells."""
    checked = validate_long_rows(rows)
    coords = {dim: sorted({row.get(dim) for row in checked}, key=lambda x: str(x)) for dim in CUBE_AXES}
    values: dict[str, dict[str, float | None]] = {metric: {} for metric in METRICS}
    sample_counts: dict[str, dict[str, int]] = {metric: {} for metric in METRICS}
    for row in checked:
        key = _json_key([row.get(dim) for dim in CUBE_AXES])
        metric = str(row["metric"])
        values.setdefault(metric, {})[key] = None if row["value"] is None else float(row["value"])
        sample_counts.setdefault(metric, {})[key] = int(row["sample_count"])
    return LabeledCube(CUBE_AXES, coords, values, sample_counts)


def _profit_factor(values: Sequence[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if not values:
        return None
    return math.inf if losses == 0 and gains > 0 else (gains / losses if losses else 0.0)


def _max_drawdown(values: Sequence[float]) -> float | None:
    if not values:
        return None
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return abs(drawdown)


def _normal_lcb(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, None
    stdev = statistics.stdev(values)
    return mean, mean - 1.96 * stdev / math.sqrt(len(values))


def _summary_rows(group: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    first = group[0]
    nets = [float(row["net_jpy"]) for row in group if row.get("net_jpy") is not None]
    denominator = len(group)
    mean, lcb = _normal_lcb(nets)
    margin = [float(row["margin_feasible"]) for row in group if row.get("margin_feasible") is not None]
    fills = [float(row["fill_valid"]) for row in group if row.get("fill_valid") is not None]
    unwinds = [float(row["unwind_valid"]) for row in group if row.get("unwind_valid") is not None]
    turnover = sum(abs(float(row.get("units", 0.0))) for row in group) / max(1.0, denominator)
    shared = {
        "episode_id": "__aggregate__",
        "source_sha": first["source_sha"],
        "decision_time": first["decision_time"],
        "pair": first["pair"], "timeframe": first["timeframe"], "regime": first["regime"],
        "strategy": first["strategy"], "parameter_set": first["parameter_set"],
        "cost_scenario": first["cost_scenario"], "exposure_state": first["exposure_state"],
        "exit_policy": first["exit_policy"], "viewpoint": "multi_dimensional_cube",
        "uncertainty": {"mean_lcb": lcb, "method": "normal_approximation_fixture_only"},
        "sample_count": len(nets), "admission_status": first["admission_status"],
        "split": first["split"], "method": first["method"], "cost": first["cost"],
        "risk": first["risk"], "exit": first["exit"],
    }
    metrics: dict[str, float | None] = {
        "after_cost_net_jpy": sum(nets) if nets else None,
        "lcb_jpy": lcb,
        "profit_factor": _profit_factor(nets),
        "max_drawdown_jpy": _max_drawdown(nets),
        "margin_coverage": statistics.fmean(margin) if margin else None,
        "turnover": turnover if group else None,
        "fill_validity": statistics.fmean(fills) if fills else None,
        "unwind_validity": statistics.fmean(unwinds) if unwinds else None,
        "sample_coverage": len(nets) / denominator if denominator else None,
    }
    return [dict(shared, metric=metric, value=value) for metric, value in metrics.items()]


def records_to_long(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate only within identical causal coordinates."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(name) for name in ("split", "timeframe", "pair", "regime", "method", "cost", "risk", "exit"))
        groups.setdefault(key, []).append(record)
    return validate_long_rows(row for group in groups.values() for row in _summary_rows(group))


def pairwise_interactions(rows: Sequence[Mapping[str, Any]], metric: str = "after_cost_net_jpy") -> list[dict[str, Any]]:
    """Return two-factor interaction contrasts on non-missing metric rows."""
    selected = [row for row in rows if row.get("metric") == metric and row.get("value") is not None]
    if not selected:
        return []
    factors = ("method", "regime", "cost", "risk", "exit")
    out: list[dict[str, Any]] = []
    for left_index, left in enumerate(factors):
        for right in factors[left_index + 1:]:
            cells: dict[tuple[Any, Any], list[float]] = {}
            for row in selected:
                cells.setdefault((row.get(left), row.get(right)), []).append(float(row["value"]))
            if len(cells) < 4:
                continue
            left_levels = sorted({key[0] for key in cells}, key=str)
            right_levels = sorted({key[1] for key in cells}, key=str)
            if len(left_levels) < 2 or len(right_levels) < 2:
                continue
            def avg(a: Any, b: Any) -> float | None:
                values = cells.get((a, b), [])
                return statistics.fmean(values) if values else None
            a, b = left_levels[:2]
            c, d = right_levels[:2]
            corners = (avg(a, c), avg(a, d), avg(b, c), avg(b, d))
            if any(value is None for value in corners):
                continue
            contrast = (corners[3] - corners[2]) - (corners[1] - corners[0])
            out.append({"factor_a": left, "factor_b": right, "contrast": contrast, "corners": corners})
    return out


def pareto_front(candidates: Sequence[Mapping[str, Any]], *, split: str = "VALIDATION") -> list[dict[str, Any]]:
    """Keep feasible non-dominated summaries; no scalar PnL collapse."""
    selected = [dict(row) for row in candidates if row.get("split") == split and row.get("holdout") is not True]
    feasible = [row for row in selected if row.get("sample_coverage", 0) >= 0.8
                and row.get("margin_coverage", 0) >= 0.8
                and row.get("fill_validity", 0) >= 1.0
                and row.get("unwind_validity", 0) >= 1.0]
    def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
        greater = (
            float(a["after_cost_net_jpy"]) >= float(b["after_cost_net_jpy"]),
            float(a["lcb_jpy"]) >= float(b["lcb_jpy"]),
            float(a["profit_factor"]) >= float(b["profit_factor"]),
            float(a["max_drawdown_jpy"]) <= float(b["max_drawdown_jpy"]),
            float(a["margin_coverage"]) >= float(b["margin_coverage"]),
        )
        strict = (
            float(a["after_cost_net_jpy"]) > float(b["after_cost_net_jpy"]),
            float(a["lcb_jpy"]) > float(b["lcb_jpy"]),
            float(a["profit_factor"]) > float(b["profit_factor"]),
            float(a["max_drawdown_jpy"]) < float(b["max_drawdown_jpy"]),
            float(a["margin_coverage"]) > float(b["margin_coverage"]),
        )
        return all(greater) and any(strict)
    return [row for row in feasible if not any(dominates(other, row) for other in feasible if other is not row)]


def simpson_candidates(rows: Sequence[Mapping[str, Any]], metric: str = "after_cost_net_jpy") -> list[dict[str, Any]]:
    """Flag aggregate/stratified sign reversals without claiming causality."""
    selected = [row for row in rows if row.get("metric") == metric and row.get("value") is not None]
    if len(selected) < 2:
        return []
    overall = statistics.fmean(float(row["value"]) for row in selected)
    groups: dict[Any, list[float]] = {}
    for row in selected:
        groups.setdefault(row.get("regime"), []).append(float(row["value"]))
    group_means = {str(key): statistics.fmean(values) for key, values in groups.items() if values}
    if overall == 0 or not group_means:
        return []
    if all(value * overall < 0 for value in group_means.values()):
        return [{"overall_mean": overall, "group_means": group_means, "status": "SIMpson_candidate_requires_review"}]
    return []


def deterministic_conformal(values: Sequence[float], calibration_fraction: float = 0.5) -> dict[str, Any]:
    """Split conformal interval adapter used when MAPIE is unavailable."""
    if len(values) < 4:
        return {"status": "INSUFFICIENT_EVIDENCE", "coverage": None, "width": None}
    cut = max(2, int(len(values) * calibration_fraction))
    calibration = list(values[:cut])
    test = list(values[cut:])
    centre = statistics.fmean(calibration)
    residuals = sorted(abs(value - centre) for value in calibration)
    index = min(len(residuals) - 1, math.ceil((len(residuals) + 1) * 0.9) - 1)
    radius = residuals[index]
    coverage = sum(abs(value - centre) <= radius for value in test) / len(test)
    return {"status": "EXECUTED_FALLBACK", "coverage": coverage, "width": 2 * radius, "train_count": len(calibration), "validation_count": len(test)}


def placebo_refutation(values: Sequence[float], seed: int = 17) -> dict[str, Any]:
    """Deterministic placebo permutation, not a replacement for DoWhy."""
    if len(values) < 4:
        return {"status": "INSUFFICIENT_EVIDENCE", "observed_mean": None, "placebo_mean": None}
    observed = statistics.fmean(values)
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    placebo = statistics.fmean(shuffled[:len(values) // 2])
    return {"status": "EXECUTED_FALLBACK", "observed_mean": observed, "placebo_mean": placebo, "seed": seed}


def drift_refutation(values: Sequence[float], min_segment: int = 3) -> dict[str, Any]:
    """Causal-order rolling mean shift used when River is unavailable."""
    if len(values) < min_segment * 2:
        return {"status": "INSUFFICIENT_EVIDENCE", "change_index": None}
    best = max(range(min_segment, len(values) - min_segment + 1), key=lambda i: abs(statistics.fmean(values[:i]) - statistics.fmean(values[i:])))
    return {"status": "EXECUTED_FALLBACK", "change_index": best, "mean_before": statistics.fmean(values[:best]), "mean_after": statistics.fmean(values[best:])}


def optional_status(package: str) -> dict[str, Any]:
    module = {"scikit-learn": "sklearn", "ta-lib": "talib", "pandas-ta": "pandas_ta"}.get(package, package)
    return {"package": package, "installed": importlib.util.find_spec(module) is not None}


def bid_ask_after_cost(
    *, side: str, entry_bid: float, entry_ask: float, exit_bid: float, exit_ask: float,
    units: float, fee_jpy: float = 0.0, financing_jpy: float = 0.0,
    slippage_jpy: float = 0.0, opportunity_cost_jpy: float = 0.0,
) -> dict[str, float]:
    """QR-owned side-aware fill oracle for the comparison fixture."""
    if side == "LONG":
        entry_fill, exit_fill = entry_ask, exit_bid
        gross = (exit_fill - entry_fill) * units
    elif side == "SHORT":
        entry_fill, exit_fill = entry_bid, exit_ask
        gross = (entry_fill - exit_fill) * units
    else:
        raise ValueError("side must be LONG or SHORT")
    total_cost = fee_jpy + financing_jpy + slippage_jpy + opportunity_cost_jpy
    return {"entry_fill": entry_fill, "exit_fill": exit_fill, "gross_jpy": gross, "cost_jpy": total_cost, "net_jpy": gross - total_cost}


def source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
