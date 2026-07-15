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
    60: {"lookback_days": 7, "entry_interval_hours": 1},
    240: {"lookback_days": 21, "entry_interval_hours": 4},
    1440: {"lookback_days": 30, "entry_interval_hours": 24},
}
PROMOTION_BLOCKERS = (
    "DIRECTION_EDGE_REQUIRES_LOCKED_S5_TP_SL_VEHICLE",
    "M5_FUTURE_CLOSE_IS_NOT_A_LIVE_EXIT_POLICY",
    "FORWARD_LIVE_SHADOW_REQUIRED",
)


def main() -> int:
    args = _parse_args()
    np, pd = _research_dependencies()
    horizons = _parse_horizons(args.horizons)
    history_dirs = replay._history_dirs(
        args.history_dir,
        granularity="M5",
        auto_min_days=0.0,
    )
    candles, candle_stats = replay._load_candles(history_dirs, granularity="M5")
    pair_codes = {pair: index for index, pair in enumerate(sorted(candles))}
    feature_frames = [
        _pair_feature_frame(
            pair,
            pair_candles,
            pair_code=pair_codes[pair],
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
            entry_interval_hours=int(config["entry_interval_hours"]),
        )
        timestamps = sorted(scheduled.index.unique())
        evaluation_from = timestamps[int(len(timestamps) * args.evaluation_fraction)]
        holdout_from = timestamps[int(len(timestamps) * args.holdout_fraction)]
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
            "lookback_days": config["lookback_days"],
            "entry_interval_hours": config["entry_interval_hours"],
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
        "rules": list(RULES),
        "horizon_configuration": HORIZON_CONFIG,
        "spread_cap_pips": args.spread_cap_pips,
        "opportunity_fraction": args.opportunity_fraction,
        "minimum_resolved_rows": args.minimum_resolved_rows,
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
    parser.add_argument("--horizons", default="60,240,1440")
    parser.add_argument("--spread-cap-pips", type=float, default=2.0)
    parser.add_argument("--opportunity-fraction", type=float, default=0.30)
    parser.add_argument("--minimum-resolved-rows", type=int, default=100)
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


def _scheduled_rows(dataset, *, entry_interval_hours: int):
    # Forecast at xx:55 uses the complete candle ending on the exact entry hour.
    minute_match = dataset.index.minute == 55
    if entry_interval_hours == 1:
        return dataset.loc[minute_match].copy()
    if entry_interval_hours == 4:
        return dataset.loc[minute_match & (dataset.index.hour % 4 == 3)].copy()
    if entry_interval_hours == 24:
        return dataset.loc[minute_match & (dataset.index.hour == 23)].copy()
    raise ValueError("supported entry intervals are 1, 4, and 24 hours")


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
            & (dataset["spread_pips"] <= spread_cap_pips)
        ]
        current = dataset.loc[
            (dataset.index == now)
            & (dataset["spread_pips"] <= spread_cap_pips)
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
