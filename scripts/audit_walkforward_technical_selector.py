#!/usr/bin/env python3
"""Replay a causal rolling technical-rule tuner on exact M5 bid/ask truth.

At each scheduled decision, the tuner ranks predeclared trend and mean-revert
rules using only outcomes whose full horizon already resolved. It then applies
the selected rule to the strongest current cross-pair opportunities. No fixed
March-trained model survives into July unchanged.

This evaluates forecast direction, not a live exit vehicle. Any positive
horizon must still pass a separately locked exact-S5 TP/SL replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (SRC, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import oanda_history_replay_validate as replay  # noqa: E402
from audit_market_phase_technical_selector import (  # noqa: E402
    RULES as CONTEXT_RULES,
    _phase_and_rule_frame,
)
from train_causal_technical_forecaster import (  # noqa: E402
    _label_frame,
    _pair_feature_frame,
)

from quant_rabbit.technical_forecast_evaluation import (  # noqa: E402
    TECHNICAL_FORECAST_EVALUATION_CONTRACT,
    directional_metrics,
)


SELECTOR_CONTRACT = "QR_WALKFORWARD_TECHNICAL_RULE_SELECTOR_V1"
RULES = (
    "return_3",
    "return_12",
    "return_48",
    "return_96",
    "return_288",
    "ema_gap_5_24",
    "ema_gap_12_48",
    "ema_gap_24_96",
    "ema_gap_48_288",
)
HORIZON_CONFIG = {
    15: {"lookback_days": 3, "entry_interval_minutes": 15},
    30: {"lookback_days": 5, "entry_interval_minutes": 30},
    60: {"lookback_days": 7, "entry_interval_minutes": 60},
    240: {"lookback_days": 21, "entry_interval_minutes": 240},
    1440: {"lookback_days": 30, "entry_interval_minutes": 1440},
}
PROMOTION_BLOCKERS = (
    "DIRECTION_EDGE_REQUIRES_LOCKED_S5_TP_SL_VEHICLE",
    "M5_FIXED_HORIZON_RETURN_IS_NOT_A_LIVE_EXIT_POLICY",
    "FORWARD_LIVE_SHADOW_REQUIRED",
)
CONTEXT_SCOPES = (
    ("PAIR_PHASE_SESSION", ("pair", "market_phase", "utc_session_bucket")),
    ("PAIR_PHASE", ("pair", "market_phase")),
    ("PAIR_SESSION", ("pair", "utc_session_bucket")),
    ("PAIR", ("pair",)),
    ("PHASE_SESSION", ("market_phase", "utc_session_bucket")),
    ("PHASE", ("market_phase",)),
    ("GLOBAL", ()),
)
CONTEXT_THRESHOLD_QUANTILES = (0.0, 0.50, 0.70, 0.85)


def main() -> int:
    args = _parse_args()
    np, pd = _research_dependencies()
    horizons = _parse_horizons(args.horizons)
    pair_filter = _parse_pairs(args.pairs)
    history_dirs = replay._history_dirs(
        args.history_dir,
        granularity="M5",
        auto_min_days=0.0,
    )
    candles, candle_stats = replay._load_candles(history_dirs, granularity="M5")
    if pair_filter:
        missing_pairs = sorted(pair_filter - set(candles))
        if missing_pairs:
            raise ValueError(
                "requested pairs missing from M5 history: "
                + ",".join(missing_pairs)
            )
        candles = {
            pair: pair_candles
            for pair, pair_candles in candles.items()
            if pair in pair_filter
        }
        candle_stats = {
            **candle_stats,
            "history_pairs": len(candles),
            "history_candles": sum(len(rows) for rows in candles.values()),
        }
    pair_codes = {pair: index for index, pair in enumerate(sorted(candles))}
    feature_frames = [
        _phase_and_rule_frame(
            _pair_feature_frame(
            pair,
            pair_candles,
            pair_code=pair_codes[pair],
            np=np,
            pd=pd,
            ),
            np=np,
            pd=pd,
        )
        for pair, pair_candles in sorted(candles.items())
    ]
    args.prediction_output_dir.mkdir(parents=True, exist_ok=True)

    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        config = HORIZON_CONFIG[horizon]
        dataset = pd.concat(
            [
                _label_frame(frame, horizon_min=horizon, np=np, pd=pd)
                for frame in feature_frames
            ]
        ).sort_index()
        scheduled = _scheduled_rows(
            dataset,
            entry_interval_minutes=int(config["entry_interval_minutes"]),
        )
        timestamps = sorted(scheduled.index.unique())
        evaluation_from = timestamps[int(len(timestamps) * args.evaluation_fraction)]
        holdout_from = timestamps[int(len(timestamps) * args.holdout_fraction)]
        if args.contextual:
            selections = _walkforward_contextual_select(
                scheduled,
                horizon_min=horizon,
                lookback_days=args.context_lookback_days,
                evaluation_from=evaluation_from,
                spread_cap_pips=args.spread_cap_pips,
                minimum_context_rows=args.minimum_context_rows,
                admission=args.context_admission,
                np=np,
                pd=pd,
            )
        else:
            selections = _walkforward_select(
                scheduled,
                horizon_min=horizon,
                lookback_days=int(config["lookback_days"]),
                evaluation_from=evaluation_from,
                spread_cap_pips=args.spread_cap_pips,
                opportunity_fraction=args.opportunity_fraction,
                minimum_resolved_rows=args.minimum_resolved_rows,
                np=np,
                pd=pd,
            )
        validation = [
            row
            for row in selections
            if _parse_time(row["timestamp_utc"]) < holdout_from
        ]
        holdout = [
            row
            for row in selections
            if _parse_time(row["timestamp_utc"]) >= holdout_from
        ]
        validation_metrics = directional_metrics(validation)
        holdout_metrics = directional_metrics(holdout)
        direction_candidate = _direction_candidate(
            validation_metrics,
            holdout_metrics,
            minimum_validation_trades=args.minimum_validation_trades,
            minimum_holdout_trades=args.minimum_holdout_trades,
        )
        key = f"{horizon}m"
        validation_path = (
            args.prediction_output_dir / f"walkforward_{key}_validation.jsonl"
        )
        holdout_path = args.prediction_output_dir / f"walkforward_{key}_holdout.jsonl"
        _write_jsonl(validation_path, validation)
        _write_jsonl(holdout_path, holdout)
        by_horizon[key] = {
            "horizon_min": horizon,
            "lookback_days": (
                args.context_lookback_days
                if args.contextual
                else config["lookback_days"]
            ),
            "entry_interval_minutes": config["entry_interval_minutes"],
            "evaluation_from_utc": evaluation_from.isoformat(),
            "holdout_from_utc": holdout_from.isoformat(),
            "scheduled_rows": len(scheduled),
            "validation_metrics": validation_metrics,
            "holdout_metrics": holdout_metrics,
            "selected_rule_counts": dict(
                sorted(Counter(row["selected_rule"] for row in selections).items())
            ),
            "selected_orientation_counts": dict(
                sorted(Counter(row["orientation"] for row in selections).items())
            ),
            "selected_context_scope_counts": dict(
                sorted(
                    Counter(
                        row.get("context_scope", "GLOBAL")
                        for row in selections
                    ).items()
                )
            ),
            "direction_edge_candidate": direction_candidate,
            "promotion_allowed": False,
            "promotion_blockers": list(PROMOTION_BLOCKERS),
            "validation_predictions_path": str(validation_path.resolve()),
            "validation_predictions_sha256": _file_sha256(validation_path),
            "holdout_predictions_path": str(holdout_path.resolve()),
            "holdout_predictions_sha256": _file_sha256(holdout_path),
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": SELECTOR_CONTRACT,
        "evaluation_contract": TECHNICAL_FORECAST_EVALUATION_CONTRACT,
        "promotion_allowed": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "history_dirs": [str(path.resolve()) for path in history_dirs],
        "history_candles_sha256": replay._truth_candles_digest(candles),
        "history_granularity": "M5",
        "source_pair_filter": sorted(pair_filter),
        "rules": list(CONTEXT_RULES if args.contextual else RULES),
        "horizon_configuration": HORIZON_CONFIG,
        "spread_cap_pips": args.spread_cap_pips,
        "opportunity_fraction": args.opportunity_fraction,
        "minimum_resolved_rows": args.minimum_resolved_rows,
        "contextual": args.contextual,
        "context_lookback_days": args.context_lookback_days,
        "minimum_context_rows": args.minimum_context_rows,
        "context_admission": args.context_admission,
        "tuner_semantics": (
            "each scheduled decision ranks predeclared rule/orientation pairs "
            "from only recent already-resolved executable outcomes; the "
            "strongest current cross-pair fraction is then selected"
        ),
        **candle_stats,
        "by_horizon": by_horizon,
    }
    _write_json(args.report_output, report)
    _write_text(args.report_output.with_suffix(".md"), _markdown(report))
    print(f"wrote {args.report_output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--pairs",
        default="",
        help="optional comma-separated pair universe used for tuning and selection",
    )
    parser.add_argument("--horizons", default="60,240,1440")
    parser.add_argument("--spread-cap-pips", type=float, default=2.0)
    parser.add_argument("--opportunity-fraction", type=float, default=0.30)
    parser.add_argument("--minimum-resolved-rows", type=int, default=100)
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--context-lookback-days", type=int, default=14)
    parser.add_argument("--minimum-context-rows", type=int, default=12)
    parser.add_argument(
        "--context-admission",
        choices=("POINT_POSITIVE", "SHRUNK_POSITIVE"),
        default="POINT_POSITIVE",
    )
    parser.add_argument("--evaluation-fraction", type=float, default=0.60)
    parser.add_argument("--holdout-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-validation-trades", type=int, default=30)
    parser.add_argument("--minimum-holdout-trades", type=int, default=30)
    parser.add_argument(
        "--prediction-output-dir",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "walkforward_technical_predictions",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "walkforward_technical_selector_latest.json",
    )
    return parser.parse_args()


def _research_dependencies():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("walk-forward audit requires numpy and pandas") from exc
    return np, pd


def _scheduled_rows(dataset, *, entry_interval_minutes: int):
    """Schedule on executable next-open timestamps, never candle labels."""

    interval = int(entry_interval_minutes)
    if interval < 5 or interval > 1440 or interval % 5:
        raise ValueError("entry interval must be a 5-minute multiple up to 1440")
    entry = dataset["entry_timestamp_utc"]
    minute_of_day = entry.dt.hour * 60 + entry.dt.minute
    return dataset.loc[(minute_of_day % interval) == 0].copy()


def _walkforward_select(
    dataset,
    *,
    horizon_min: int,
    lookback_days: int,
    evaluation_from,
    spread_cap_pips: float,
    opportunity_fraction: float,
    minimum_resolved_rows: int,
    np,
    pd,
) -> list[dict[str, Any]]:
    if not 0.0 < opportunity_fraction <= 1.0:
        raise ValueError("opportunity_fraction must be inside (0, 1]")
    selections: list[dict[str, Any]] = []
    timestamps = sorted(dataset.index.unique())
    for now in timestamps:
        if now < evaluation_from:
            continue
        history = dataset.loc[
            (dataset.index >= now - pd.Timedelta(days=lookback_days))
            & (dataset["future_timestamp_utc"] <= now)
            & (dataset["entry_spread_pips"] <= spread_cap_pips)
        ]
        current = dataset.loc[
            (dataset.index == now)
            & (dataset["entry_spread_pips"] <= spread_cap_pips)
        ]
        if len(history) < minimum_resolved_rows or current.empty:
            continue
        best: tuple[tuple[float, float], str, int, int, float, float] | None = None
        long_history = history["long_pips"].to_numpy()
        short_history = history["short_pips"].to_numpy()
        for rule in RULES:
            raw_score = history[rule].to_numpy()
            for orientation in (1, -1):
                score = raw_score * orientation
                result = np.where(score >= 0.0, long_history, short_history)
                mean = float(result.mean())
                standard_error = float(result.std(ddof=1) / math.sqrt(len(result)))
                # One standard-error shrinkage is a ranking score only, not a
                # statistical proof claim. Final evidence uses the stricter
                # one-sided 95% metrics from directional_metrics.
                rank = (mean - standard_error, mean)
                candidate = (
                    rank,
                    rule,
                    orientation,
                    len(result),
                    mean,
                    standard_error,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        assert best is not None
        _rank, rule, orientation, resolved_rows, recent_mean, recent_se = best
        score = current[rule].to_numpy() * orientation
        count = max(1, int(math.ceil(len(current) * opportunity_fraction)))
        chosen_indices = np.argsort(np.abs(score))[-count:]
        long_current = current["long_pips"].to_numpy()
        short_current = current["short_pips"].to_numpy()
        for index in chosen_indices:
            direction = "UP" if score[index] >= 0.0 else "DOWN"
            row = current.iloc[int(index)]
            selections.append(
                {
                    "timestamp_utc": now.isoformat(),
                    "entry_timestamp_utc": row["entry_timestamp_utc"].isoformat(),
                    "future_timestamp_utc": row["future_timestamp_utc"].isoformat(),
                    "pair": str(row["pair"]),
                    "horizon_min": horizon_min,
                    "selected_rule": rule,
                    "orientation": "DIRECT" if orientation == 1 else "INVERSE",
                    "technical_score": round(float(score[index]), 6),
                    "predicted_pips": round(float(score[index]), 6),
                    "predicted_direction": direction,
                    "spread_pips_at_forecast": round(
                        float(row["spread_pips"]), 6
                    ),
                    "entry_spread_pips": round(
                        float(row["entry_spread_pips"]), 6
                    ),
                    "exit_spread_pips": round(
                        float(row["exit_spread_pips"]), 6
                    ),
                    "roundtrip_spread_cost_pips": round(
                        float(row["roundtrip_spread_cost_pips"]), 6
                    ),
                    "gross_directional_pips": round(
                        float(
                            row["target_mid_pips"]
                            if direction == "UP"
                            else -row["target_mid_pips"]
                        ),
                        6,
                    ),
                    "long_pips": round(float(long_current[index]), 6),
                    "short_pips": round(float(short_current[index]), 6),
                    "executed_pips": round(
                        float(
                            long_current[index]
                            if direction == "UP"
                            else short_current[index]
                        ),
                        6,
                    ),
                    "tuning_lookback_days": lookback_days,
                    "tuning_resolved_rows": resolved_rows,
                    "tuning_recent_mean_pips": round(recent_mean, 6),
                    "tuning_recent_standard_error_pips": round(recent_se, 6),
                    "selector_contract": SELECTOR_CONTRACT,
                }
            )
    return selections


def _walkforward_contextual_select(
    dataset,
    *,
    horizon_min: int,
    lookback_days: int,
    evaluation_from,
    spread_cap_pips: float,
    minimum_context_rows: int,
    admission: str,
    np,
    pd,
) -> list[dict[str, Any]]:
    """Retune one rule per pair/state/session from resolved recent outcomes."""

    if minimum_context_rows < 2:
        raise ValueError("minimum_context_rows must be at least two")
    admission = str(admission).upper()
    if admission not in {"POINT_POSITIVE", "SHRUNK_POSITIVE"}:
        raise ValueError("unsupported contextual admission")
    selections: list[dict[str, Any]] = []
    accepted_until: dict[str, Any] = {}
    timestamps = sorted(dataset.index.unique())
    for now in timestamps:
        if now < evaluation_from:
            continue
        history = dataset.loc[
            (dataset.index >= now - pd.Timedelta(days=lookback_days))
            & (dataset["future_timestamp_utc"] <= now)
            & (dataset["entry_spread_pips"] <= spread_cap_pips)
        ]
        current = dataset.loc[
            (dataset.index == now)
            & (dataset["entry_spread_pips"] <= spread_cap_pips)
        ]
        if history.empty or current.empty:
            continue
        for _, row in current.iterrows():
            pair = str(row["pair"])
            if now < accepted_until.get(pair, now):
                continue
            context, context_scope = _context_history(
                history,
                row,
                minimum_context_rows=minimum_context_rows,
            )
            if context.empty:
                continue
            selected = _best_context_rule(
                context,
                row,
                minimum_qualified_rows=max(4, minimum_context_rows // 2),
                np=np,
            )
            if selected is None:
                continue
            recent_edge = (
                selected["mean_pips"] - selected["standard_error_pips"]
                if admission == "SHRUNK_POSITIVE"
                else selected["mean_pips"]
            )
            if recent_edge <= 0.0:
                continue
            score = float(row[selected["rule"]]) * int(selected["orientation"])
            direction = "UP" if score >= 0.0 else "DOWN"
            long_pips = float(row["long_pips"])
            short_pips = float(row["short_pips"])
            gross_directional = (
                float(row["target_mid_pips"])
                if direction == "UP"
                else -float(row["target_mid_pips"])
            )
            selections.append(
                {
                    "timestamp_utc": now.isoformat(),
                    "entry_timestamp_utc": row["entry_timestamp_utc"].isoformat(),
                    "future_timestamp_utc": row["future_timestamp_utc"].isoformat(),
                    "pair": pair,
                    "market_phase": str(row["market_phase"]),
                    "utc_session_bucket": str(row["utc_session_bucket"]),
                    "context_scope": context_scope,
                    "horizon_min": horizon_min,
                    "selected_rule": selected["rule"],
                    "orientation": (
                        "DIRECT"
                        if int(selected["orientation"]) == 1
                        else "INVERSE"
                    ),
                    "minimum_absolute_score": round(
                        float(selected["threshold"]), 9
                    ),
                    "technical_score": round(score, 6),
                    "predicted_pips": round(score, 6),
                    "predicted_direction": direction,
                    "spread_pips_at_forecast": round(
                        float(row["spread_pips"]), 6
                    ),
                    "entry_spread_pips": round(
                        float(row["entry_spread_pips"]), 6
                    ),
                    "exit_spread_pips": round(
                        float(row["exit_spread_pips"]), 6
                    ),
                    "roundtrip_spread_cost_pips": round(
                        float(row["roundtrip_spread_cost_pips"]), 6
                    ),
                    "gross_directional_pips": round(gross_directional, 6),
                    "long_pips": round(long_pips, 6),
                    "short_pips": round(short_pips, 6),
                    "executed_pips": round(
                        long_pips if direction == "UP" else short_pips,
                        6,
                    ),
                    "tuning_lookback_days": lookback_days,
                    "tuning_context_rows": len(context),
                    "tuning_resolved_rows": selected["qualified_rows"],
                    "tuning_recent_mean_pips": round(
                        float(selected["mean_pips"]), 6
                    ),
                    "tuning_recent_standard_error_pips": round(
                        float(selected["standard_error_pips"]), 6
                    ),
                    "tuning_admission": admission,
                    "selector_contract": SELECTOR_CONTRACT,
                }
            )
            accepted_until[pair] = now + pd.Timedelta(minutes=horizon_min)
    return selections


def _context_history(history, current, *, minimum_context_rows: int):
    for scope_name, columns in CONTEXT_SCOPES:
        scoped = history
        for column in columns:
            scoped = scoped.loc[scoped[column] == current[column]]
        if len(scoped) >= minimum_context_rows:
            return scoped, scope_name
    return history.iloc[0:0], "NONE"


def _best_context_rule(
    history,
    current,
    *,
    minimum_qualified_rows: int,
    np,
) -> dict[str, Any] | None:
    long_history = history["long_pips"].to_numpy()
    short_history = history["short_pips"].to_numpy()
    best: tuple[tuple[float, float, int], dict[str, Any]] | None = None
    for rule in CONTEXT_RULES:
        raw = history[rule].to_numpy()
        absolute = np.abs(raw)
        current_absolute = abs(float(current[rule]))
        thresholds = sorted(
            {
                0.0
                if quantile == 0.0
                else round(float(np.quantile(absolute, quantile)), 9)
                for quantile in CONTEXT_THRESHOLD_QUANTILES
            }
        )
        for threshold in thresholds:
            if current_absolute < threshold:
                continue
            qualified = absolute >= threshold
            qualified_rows = int(qualified.sum())
            if qualified_rows < minimum_qualified_rows:
                continue
            for orientation in (1, -1):
                score = raw[qualified] * orientation
                outcomes = np.where(
                    score >= 0.0,
                    long_history[qualified],
                    short_history[qualified],
                )
                mean = float(outcomes.mean())
                standard_error = float(
                    outcomes.std(ddof=1) / math.sqrt(qualified_rows)
                )
                candidate = {
                    "rule": rule,
                    "orientation": orientation,
                    "threshold": threshold,
                    "qualified_rows": qualified_rows,
                    "mean_pips": mean,
                    "standard_error_pips": standard_error,
                }
                rank = (mean - standard_error, mean, qualified_rows)
                if best is None or rank > best[0]:
                    best = (rank, candidate)
    return best[1] if best is not None else None


def _direction_candidate(
    validation: Mapping[str, Any],
    holdout: Mapping[str, Any],
    *,
    minimum_validation_trades: int,
    minimum_holdout_trades: int,
) -> bool:
    validation_mean = validation.get("mean_pips")
    holdout_lower = holdout.get("one_sided_95_mean_lower_pips")
    holdout_factor = holdout.get("profit_factor")
    return bool(
        int(validation.get("trades") or 0) >= minimum_validation_trades
        and int(holdout.get("trades") or 0) >= minimum_holdout_trades
        and isinstance(validation_mean, (int, float))
        and float(validation_mean) > 0.0
        and isinstance(holdout_lower, (int, float))
        and float(holdout_lower) > 0.0
        and isinstance(holdout_factor, (int, float))
        and float(holdout_factor) > 1.0
        and float(holdout.get("positive_day_rate") or 0.0) > 0.5
    )


def _parse_horizons(value: str) -> list[int]:
    output: list[int] = []
    for raw in str(value or "").split(","):
        if not raw.strip():
            continue
        parsed = int(raw)
        if parsed not in HORIZON_CONFIG:
            raise ValueError(f"supported horizons are {sorted(HORIZON_CONFIG)}")
        if parsed not in output:
            output.append(parsed)
    if not output:
        raise ValueError("at least one horizon is required")
    return output


def _parse_pairs(value: str) -> set[str]:
    output = {
        token.strip().upper().replace("/", "_")
        for token in str(value or "").split(",")
        if token.strip()
    }
    invalid = sorted(
        pair
        for pair in output
        if len(pair) != 7
        or pair[3] != "_"
        or not pair.replace("_", "").isalpha()
    )
    if invalid:
        raise ValueError("invalid pair filter: " + ",".join(invalid))
    return output


def _parse_time(value: Any):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text(
        path,
        "".join(
            json.dumps(_json_safe(dict(row)), ensure_ascii=False, sort_keys=True)
            + "\n"
            for row in rows
        ),
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
        "# Walk-forward technical rule selector",
        "",
        "Direction evidence only. Exact S5 TP/SL vehicle replay is still mandatory.",
        "",
        "| horizon | validation trades | validation mean | holdout trades | holdout mean | holdout lower | candidate |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    by_horizon = report.get("by_horizon")
    for key, row in (by_horizon.items() if isinstance(by_horizon, Mapping) else []):
        validation = row.get("validation_metrics") or {}
        holdout = row.get("holdout_metrics") or {}
        lines.append(
            f"| {key} | {validation.get('trades')} | {validation.get('mean_pips')} | "
            f"{holdout.get('trades')} | {holdout.get('mean_pips')} | "
            f"{holdout.get('one_sided_95_mean_lower_pips')} | "
            f"{row.get('direction_edge_candidate')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
