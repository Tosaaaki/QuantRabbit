#!/usr/bin/env python3
"""Audit forecast-directed passive LIMIT entries on local OANDA bid/ask M1.

This is a broad causal screen, not a live-permission generator.  It compares
predeclared fixed horizons, keeps forecast target/invalidation geometry frozen,
and separates a chronological validation tail.  Any surviving cohort still
requires exact S5 replay and a newly locked forward holdout before promotion.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (SRC, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import oanda_history_replay_validate as replay  # noqa: E402
from train_forecast_orientation_model import _file_sha256  # noqa: E402

from quant_rabbit.forecast_passive_limit_replay import (  # noqa: E402
    PASSIVE_LIMIT_REPLAY_CONTRACT,
    replay_metrics,
    select_independent_forecasts,
    simulate_passive_limit,
)
from quant_rabbit.forecast_technical_reconstruction import (  # noqa: E402
    FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT,
    TECHNICAL_FEATURE_FIELDS,
    reconstruct_missing_technical_features,
)


AUDIT_CONTRACT = "QR_FORECAST_PASSIVE_LIMIT_BROAD_SCREEN_V1"
COHORT_DIMENSIONS = (
    ("pair",),
    ("direction",),
    ("technical_regime",),
    ("technical_atr_band",),
    ("technical_spread_band",),
    ("technical_range_location_24h",),
    ("technical_structure_alignment",),
    ("technical_situation",),
    ("technical_selected_method",),
    ("technical_family_direction_alignment",),
    ("technical_regime", "technical_selected_method"),
    ("technical_regime", "technical_family_direction_alignment"),
)
PROMOTION_BLOCKERS = (
    "BROAD_M1_SCREEN_REQUIRES_EXACT_S5_REPLAY",
    "MULTI_HORIZON_SELECTION_REQUIRES_NEW_FORWARD_HOLDOUT",
    "PASSIVE_LIMIT_FORWARD_HOLDOUT_REQUIRED",
    "STOP_GAP_SLIPPAGE_FORWARD_PROOF_REQUIRED",
)


def main() -> int:
    args = _parse_args()
    time_from = replay._parse_required_time(args.forecast_from, flag="--from")
    time_to = replay._parse_required_time(args.forecast_to, flag="--to")
    if time_from >= time_to:
        raise ValueError("--from must be earlier than --to")
    horizons = _parse_horizons(args.horizons)
    history_dirs = replay._history_dirs(
        args.history_dir,
        granularity="M1",
        auto_min_days=args.auto_history_min_days,
    )
    rows, load_stats = replay._load_forecasts(
        args.forecast_history,
        pairs=replay._parse_pair_filter(args.pairs),
        time_from=time_from,
        time_to=time_to,
        min_confidence=None,
        confidence_field="calibrated",
    )
    windows = _combined_windows(
        rows,
        lookback_hours=args.technical_lookback_hours,
        maximum_horizon_min=max(horizons),
    )
    candles, candle_stats = replay._load_candles(
        history_dirs,
        granularity="M1",
        windows_by_pair=windows,
    )
    technical_rows, reconstruction_stats = reconstruct_missing_technical_features(
        [_technical_seed(row) for row in rows],
        candles,
        granularity="M1",
        lookback_hours=args.technical_lookback_hours,
    )
    technical_by_source = {
        int(row["source_index"]): {
            field: row.get(field, "MISSING") for field in TECHNICAL_FEATURE_FIELDS
        }
        for row in technical_rows
    }
    candle_times = {
        pair: [candle.timestamp_utc for candle in pair_candles]
        for pair, pair_candles in candles.items()
    }

    by_horizon: dict[str, dict[str, Any]] = {}
    all_candidates: list[dict[str, Any]] = []
    for horizon in horizons:
        selected = select_independent_forecasts(rows, horizon_min=horizon)
        outcomes = [
            simulate_passive_limit(
                row,
                candles.get(row.pair, ()),
                horizon_min=horizon,
                candle_times=candle_times.get(row.pair, ()),
                technical_features=technical_by_source.get(row.source_index),
            )
            for row in selected
        ]
        training, validation, split_at = _chronological_split(
            outcomes,
            train_fraction=args.train_fraction,
        )
        cohorts = _cohort_reports(
            training,
            validation,
            minimum_train_fills=args.minimum_train_fills,
            minimum_validation_fills=args.minimum_validation_fills,
        )
        candidates = [row for row in cohorts if row["broad_screen_candidate"]]
        for candidate in candidates:
            all_candidates.append(
                {
                    "horizon_min": horizon,
                    **candidate,
                }
            )
        key = _horizon_key(horizon)
        by_horizon[key] = {
            "horizon_min": horizon,
            "selected_independent_forecasts": len(selected),
            "skipped_overlapping_forecasts": len(rows) - len(selected),
            "split_at_utc": split_at,
            "status_counts": dict(
                sorted(collections.Counter(row.get("status") for row in outcomes).items())
            ),
            "full_metrics": replay_metrics(outcomes),
            "training_metrics": replay_metrics(training),
            "validation_metrics": replay_metrics(validation),
            "cohort_count": len(cohorts),
            "broad_screen_candidates": candidates,
            "all_cohorts": cohorts,
        }

    all_candidates.sort(
        key=lambda row: (
            _metric(row, "validation_metrics", "mean_conservative_pips"),
            _metric(row, "validation_metrics", "fills"),
        ),
        reverse=True,
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": AUDIT_CONTRACT,
        "vehicle_contract": PASSIVE_LIMIT_REPLAY_CONTRACT,
        "promotion_allowed": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "broad_screen_candidate_count": len(all_candidates),
        "broad_screen_candidates_ranked": all_candidates,
        "truth_semantics": (
            "LONG joins first complete M1 bid and fills only if ask reaches it; "
            "SHORT joins ask and fills only if bid reaches it; attached TP exits "
            "on bid/ask executable side; unresolved and same-bar ambiguity are "
            "charged as full forecast risk; no market entry or time close"
        ),
        "horizons_min": horizons,
        "short_term_horizons_min": [value for value in horizons if value <= 30.0],
        "long_term_horizons_min": [value for value in horizons if value >= 60.0],
        "train_fraction": args.train_fraction,
        "minimum_train_fills": args.minimum_train_fills,
        "minimum_validation_fills": args.minimum_validation_fills,
        "forecast_history_path": str(args.forecast_history.resolve()),
        "forecast_history_sha256": _file_sha256(args.forecast_history),
        "history_dirs": [str(path.resolve()) for path in history_dirs],
        "truth_candles_sha256": replay._truth_candles_digest(candles),
        "evaluator_sha256": hashlib.sha256(
            Path(sys.modules[simulate_passive_limit.__module__].__file__).read_bytes()
        ).hexdigest(),
        "technical_reconstruction_contract": (
            FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT
        ),
        **load_stats,
        **candle_stats,
        **reconstruction_stats,
        "by_horizon": by_horizon,
    }
    _write_json(args.report_output, report)
    _write_text(args.report_output.with_suffix(".md"), _markdown(report))
    print(f"wrote {args.report_output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-history", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument("--from", dest="forecast_from", required=True)
    parser.add_argument("--to", dest="forecast_to", required=True)
    parser.add_argument("--pairs", default="")
    parser.add_argument("--horizons", default="5,15,30,60,240")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--technical-lookback-hours", type=float, default=24.0)
    parser.add_argument("--minimum-train-fills", type=int, default=30)
    parser.add_argument("--minimum-validation-fills", type=int, default=20)
    parser.add_argument("--auto-history-min-days", type=float, default=0.0)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "forecast_passive_limit_broad_screen_latest.json",
    )
    return parser.parse_args()


def _parse_horizons(value: str) -> list[float]:
    output: list[float] = []
    for raw in str(value or "").split(","):
        if not raw.strip():
            continue
        parsed = float(raw)
        if not 0.0 < parsed <= 1440.0:
            raise ValueError("each horizon must be inside (0, 1440]")
        if parsed not in output:
            output.append(parsed)
    if not output:
        raise ValueError("at least one horizon is required")
    return output


def _combined_windows(
    rows: Sequence[replay.ForecastRow],
    *,
    lookback_hours: float,
    maximum_horizon_min: float,
) -> dict[str, list[tuple[datetime, datetime]]]:
    lookback = timedelta(hours=float(lookback_hours))
    forward = timedelta(minutes=float(maximum_horizon_min) + 1.0)
    windows: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in rows:
        windows.setdefault(row.pair, []).append(
            (row.timestamp_utc - lookback, row.timestamp_utc + forward)
        )
    return {
        pair: replay._merge_windows(pair_windows)
        for pair, pair_windows in windows.items()
    }


def _technical_seed(row: replay.ForecastRow) -> dict[str, Any]:
    context = row.technical_context_v1
    return {
        "source_index": row.source_index,
        "timestamp_utc": row.timestamp_utc.isoformat(),
        "pair": row.pair,
        "direction": row.direction,
        "forecast_direction": row.direction,
        "technical_context_sha256": replay._technical_context_digest(context),
        "technical_regime": replay._technical_context_label(
            context, "regime", "primary"
        ),
        "technical_atr_band": replay._technical_context_label(
            context, "volatility", "primary_atr_band"
        ),
        "technical_spread_band": replay._technical_context_label(
            context, "execution", "spread_band"
        ),
        "technical_range_location_24h": replay._technical_context_label(
            context, "location", "range_location_24h"
        ),
        "technical_structure_alignment": replay._technical_structure_alignment(
            context,
            forecast_direction=row.direction,
        ),
        "technical_situation": row.technical_situation,
        "technical_selected_method": row.technical_selected_method,
        "technical_family_direction_alignment": (
            row.technical_family_direction_alignment
        ),
    }


def _chronological_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str | None]:
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be inside (0, 1)")
    ordered = sorted(rows, key=lambda row: str(row.get("timestamp_utc") or ""))
    if len(ordered) < 2:
        return list(ordered), [], None
    split_index = min(len(ordered) - 1, max(1, int(len(ordered) * train_fraction)))
    split_at = str(ordered[split_index].get("timestamp_utc") or "")
    training = [row for row in ordered if str(row.get("timestamp_utc") or "") < split_at]
    validation = [row for row in ordered if str(row.get("timestamp_utc") or "") >= split_at]
    return training, validation, split_at or None


def _cohort_reports(
    training: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    minimum_train_fills: int,
    minimum_validation_fills: int,
) -> list[dict[str, Any]]:
    definitions: list[tuple[tuple[str, ...], tuple[str, ...]]] = [((), ())]
    for dimensions in COHORT_DIMENSIONS:
        keys = sorted({tuple(_feature(row, field) for field in dimensions) for row in training})
        definitions.extend((dimensions, key) for key in keys)
    reports: list[dict[str, Any]] = []
    for dimensions, key in definitions:
        train_rows = [row for row in training if _matches(row, dimensions, key)]
        validation_rows = [row for row in validation if _matches(row, dimensions, key)]
        train_metrics = replay_metrics(train_rows)
        validation_metrics = replay_metrics(validation_rows)
        enough = (
            int(train_metrics["fills"]) >= minimum_train_fills
            and int(validation_metrics["fills"]) >= minimum_validation_fills
        )
        train_positive = enough and _positive_metrics(train_metrics)
        validation_positive = enough and _positive_metrics(validation_metrics)
        statistically_positive = (
            train_positive
            and validation_positive
            and _lower_bound_positive(train_metrics)
            and _lower_bound_positive(validation_metrics)
        )
        reports.append(
            {
                "cohort": "ALL"
                if not dimensions
                else ",".join(
                    f"{field}={value}" for field, value in zip(dimensions, key)
                ),
                "dimensions": list(dimensions),
                "values": list(key),
                "training_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "minimum_fill_evidence_met": enough,
                "training_positive": train_positive,
                "validation_positive": validation_positive,
                "broad_screen_candidate": train_positive and validation_positive,
                "one_sided_95_positive_both": statistically_positive,
            }
        )
    reports.sort(
        key=lambda row: (
            bool(row["broad_screen_candidate"]),
            _metric(row, "validation_metrics", "mean_conservative_pips"),
            _metric(row, "validation_metrics", "fills"),
        ),
        reverse=True,
    )
    return reports


def _feature(row: Mapping[str, Any], field: str) -> str:
    if field in row:
        return str(row.get(field) or "MISSING").upper()
    technical = row.get("technical_features")
    if isinstance(technical, Mapping):
        return str(technical.get(field) or "MISSING").upper()
    return "MISSING"


def _matches(
    row: Mapping[str, Any],
    dimensions: tuple[str, ...],
    key: tuple[str, ...],
) -> bool:
    return not dimensions or tuple(_feature(row, field) for field in dimensions) == key


def _positive_metrics(metrics: Mapping[str, Any]) -> bool:
    mean = metrics.get("mean_conservative_pips")
    factor = metrics.get("conservative_profit_factor")
    return (
        isinstance(mean, (int, float))
        and math.isfinite(float(mean))
        and float(mean) > 0.0
        and isinstance(factor, (int, float))
        and float(factor) > 1.0
    )


def _lower_bound_positive(metrics: Mapping[str, Any]) -> bool:
    value = metrics.get("one_sided_95_mean_lower_pips")
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value > 0.0


def _metric(row: Mapping[str, Any], section: str, field: str) -> float:
    value = row.get(section)
    value = value.get(field) if isinstance(value, Mapping) else None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return -math.inf
    return result if math.isfinite(result) else result


def _horizon_key(value: float) -> str:
    return f"{value:g}m".replace(".", "p")


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Forecast passive LIMIT broad screen",
        "",
        "M1 broad-screen evidence only. Live promotion is disabled until exact S5 and a new forward holdout pass.",
        "",
        "| horizon | validation fills | conservative pips/trade | PF | candidates |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_horizon = report.get("by_horizon")
    for key, row in (by_horizon.items() if isinstance(by_horizon, Mapping) else []):
        metrics = row.get("validation_metrics") if isinstance(row, Mapping) else {}
        metrics = metrics if isinstance(metrics, Mapping) else {}
        candidates = row.get("broad_screen_candidates") if isinstance(row, Mapping) else []
        lines.append(
            f"| {key} | {metrics.get('fills')} | {metrics.get('mean_conservative_pips')} "
            f"| {metrics.get('conservative_profit_factor')} | {len(candidates or [])} |"
        )
    lines.extend(
        [
            "",
            f"Broad-screen candidates: `{report.get('broad_screen_candidate_count')}`",
            "",
            "A candidate is only a route to exact replay; it is not live authorization.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
