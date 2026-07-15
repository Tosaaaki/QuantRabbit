#!/usr/bin/env python3
"""Train direct future-return models from causal M5 technical features.

The model sees only complete candles available at forecast time. Entry is the
next exact M5 bid/ask open and truth is the exact future executable close. A
chronological validation block chooses the prediction-strength threshold once;
the final holdout is then evaluated with pair-local non-overlapping positions.

This script proves forecast direction only. It cannot authorize live orders:
surviving directions still need a separately locked TP/SL vehicle replay on
exact S5 bid/ask data and the normal verifier/risk/gateway chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
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

from quant_rabbit.instruments import instrument_pip_factor  # noqa: E402
from quant_rabbit.technical_forecast_evaluation import (  # noqa: E402
    TECHNICAL_FORECAST_EVALUATION_CONTRACT,
    choose_validation_threshold,
    directional_metrics,
    select_non_overlapping_predictions,
)


TRAINING_CONTRACT = "QR_CAUSAL_M5_TECHNICAL_FORECAST_TRAINING_V1"
MODEL_FEATURES = (
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "return_48",
    "return_96",
    "return_288",
    "ema_gap_5_24",
    "ema_gap_12_48",
    "ema_gap_24_96",
    "ema_gap_48_288",
    "atr_14",
    "atr_48",
    "atr_288",
    "volatility_14",
    "volatility_48",
    "volatility_288",
    "range_location_12",
    "range_location_48",
    "range_location_288",
    "spread_pips",
    "rsi_14",
    "body_atr_ratio",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "pair_code",
)
PROMOTION_BLOCKERS = (
    "DIRECTION_MODEL_REQUIRES_LOCKED_S5_TP_SL_VEHICLE",
    "MODEL_ARTIFACT_REQUIRES_PURE_RUNTIME_EXPORT",
    "FORWARD_LIVE_SHADOW_REQUIRED",
)


def main() -> int:
    args = _parse_args()
    np, pd, estimator_class, joblib = _research_dependencies()
    horizons = _parse_horizons(args.horizons)
    history_dirs = replay._history_dirs(
        args.history_dir,
        granularity="M5",
        auto_min_days=0.0,
    )
    candles, candle_stats = replay._load_candles(
        history_dirs,
        granularity="M5",
    )
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
    args.model_output_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_output_dir.mkdir(parents=True, exist_ok=True)

    horizon_reports: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        labelled = [
            _label_frame(frame, horizon_min=horizon, np=np, pd=pd)
            for frame in feature_frames
        ]
        dataset = pd.concat(labelled).sort_index()
        training, validation, holdout, split = _chronological_blocks(
            dataset,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            np=np,
        )
        stride = max(1, int((horizon / 5.0) // 12.0))
        training_fit = training.loc[training["pair_row"] % stride == 0]
        model = estimator_class(
            loss="squared_error",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            categorical_features=["pair_code"],
            random_state=args.random_seed,
        )
        model.fit(training_fit[list(MODEL_FEATURES)], training_fit["target_mid_pips"])
        validation_predictions = model.predict(validation[list(MODEL_FEATURES)])
        holdout_predictions = model.predict(holdout[list(MODEL_FEATURES)])
        validation_rows = _prediction_rows(
            validation,
            validation_predictions,
        )
        holdout_rows = _prediction_rows(
            holdout,
            holdout_predictions,
        )
        thresholds = _prediction_thresholds(validation_predictions, np=np)
        threshold_selection = choose_validation_threshold(
            validation_rows,
            horizon_min=horizon,
            thresholds_pips=thresholds,
            minimum_trades=args.minimum_validation_trades,
            minimum_active_days=args.minimum_validation_days,
        )
        selected_threshold = threshold_selection.get("selected") or {}
        threshold_pips = float(selected_threshold.get("threshold_pips") or 0.0)
        selected_validation = select_non_overlapping_predictions(
            validation_rows,
            horizon_min=horizon,
            minimum_absolute_prediction_pips=threshold_pips,
        )
        selected_holdout = select_non_overlapping_predictions(
            holdout_rows,
            horizon_min=horizon,
            minimum_absolute_prediction_pips=threshold_pips,
        )
        validation_metrics = directional_metrics(selected_validation)
        holdout_metrics = directional_metrics(selected_holdout)
        direction_candidate = _holdout_passes(
            holdout_metrics,
            minimum_trades=args.minimum_holdout_trades,
            minimum_days=args.minimum_holdout_days,
            minimum_pairs=args.minimum_holdout_pairs,
        )

        key = _horizon_key(horizon)
        model_path = args.model_output_dir / f"causal_technical_{key}.joblib"
        joblib.dump(model, model_path)
        validation_path = (
            args.prediction_output_dir / f"causal_technical_{key}_validation.jsonl"
        )
        holdout_path = (
            args.prediction_output_dir / f"causal_technical_{key}_holdout.jsonl"
        )
        _write_jsonl(validation_path, selected_validation)
        _write_jsonl(holdout_path, selected_holdout)
        horizon_reports[key] = {
            "horizon_min": horizon,
            "training_rows": len(training),
            "training_fit_rows": len(training_fit),
            "validation_rows": len(validation),
            "holdout_rows": len(holdout),
            "training_stride_rows": stride,
            **split,
            "threshold_selection": threshold_selection,
            "locked_threshold_pips": threshold_pips,
            "validation_metrics": validation_metrics,
            "holdout_metrics": holdout_metrics,
            "direction_candidate": direction_candidate,
            "promotion_allowed": False,
            "promotion_blockers": list(PROMOTION_BLOCKERS),
            "model_path": str(model_path.resolve()),
            "model_sha256": _file_sha256(model_path),
            "validation_predictions_path": str(validation_path.resolve()),
            "validation_predictions_sha256": _file_sha256(validation_path),
            "holdout_predictions_path": str(holdout_path.resolve()),
            "holdout_predictions_sha256": _file_sha256(holdout_path),
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": TRAINING_CONTRACT,
        "evaluation_contract": TECHNICAL_FORECAST_EVALUATION_CONTRACT,
        "promotion_allowed": False,
        "promotion_blockers": list(PROMOTION_BLOCKERS),
        "history_dirs": [str(path.resolve()) for path in history_dirs],
        "history_candles_sha256": replay._truth_candles_digest(candles),
        "history_granularity": "M5",
        "history_price_component": "BID_ASK",
        "feature_causality": (
            "complete M5 candles through forecast close only; next exact M5 "
            "bid/ask open entry; exact future executable bid/ask close truth"
        ),
        "feature_names": list(MODEL_FEATURES),
        "pair_codes": pair_codes,
        "horizons_min": horizons,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        **candle_stats,
        "by_horizon": horizon_reports,
    }
    _write_json(args.report_output, report)
    _write_text(args.report_output.with_suffix(".md"), _markdown(report))
    print(f"wrote {args.report_output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument("--horizons", default="15,60,240,1440")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-validation-trades", type=int, default=40)
    parser.add_argument("--minimum-validation-days", type=int, default=12)
    parser.add_argument("--minimum-holdout-trades", type=int, default=40)
    parser.add_argument("--minimum-holdout-days", type=int, default=12)
    parser.add_argument("--minimum-holdout-pairs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=3.0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=ROOT / "logs" / "models" / "causal_technical_forecast",
    )
    parser.add_argument(
        "--prediction-output-dir",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "causal_technical_predictions",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "causal_technical_forecast_latest.json",
    )
    return parser.parse_args()


def _research_dependencies():
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError(
            "research training requires numpy, pandas, scikit-learn, and joblib"
        ) from exc
    return np, pd, HistGradientBoostingRegressor, joblib


def _pair_feature_frame(pair: str, candles, *, pair_code: int, np, pd):
    pip_factor = instrument_pip_factor(pair)
    timestamps = pd.DatetimeIndex([candle.timestamp_utc for candle in candles])
    bid_open = np.asarray([candle.bid.o for candle in candles], dtype=float)
    ask_open = np.asarray([candle.ask.o for candle in candles], dtype=float)
    bid_close = np.asarray([candle.bid.c for candle in candles], dtype=float)
    ask_close = np.asarray([candle.ask.c for candle in candles], dtype=float)
    mid_open = (bid_open + ask_open) / 2.0
    mid_close = (bid_close + ask_close) / 2.0
    mid_high = np.asarray(
        [(candle.bid.h + candle.ask.h) / 2.0 for candle in candles],
        dtype=float,
    )
    mid_low = np.asarray(
        [(candle.bid.l + candle.ask.l) / 2.0 for candle in candles],
        dtype=float,
    )
    frame = pd.DataFrame(index=timestamps)
    frame["pair"] = pair
    frame["pair_code"] = pair_code
    frame["pair_row"] = np.arange(len(frame))
    frame["bid_open"] = bid_open
    frame["ask_open"] = ask_open
    frame["bid_close"] = bid_close
    frame["ask_close"] = ask_close
    frame["mid_close"] = mid_close
    frame["spread_pips"] = (ask_close - bid_close) * pip_factor
    for lag in (1, 3, 6, 12, 24, 48, 96, 288):
        frame[f"return_{lag}"] = (
            frame["mid_close"] - frame["mid_close"].shift(lag)
        ) * pip_factor
    for span in (5, 12, 24, 48, 96, 288):
        frame[f"ema_{span}"] = frame["mid_close"].ewm(
            span=span,
            adjust=False,
        ).mean()
    for fast, slow in ((5, 24), (12, 48), (24, 96), (48, 288)):
        frame[f"ema_gap_{fast}_{slow}"] = (
            frame[f"ema_{fast}"] - frame[f"ema_{slow}"]
        ) * pip_factor
    delta = frame["mid_close"].diff() * pip_factor
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean()
    frame["rsi_14"] = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
    previous = frame["mid_close"].shift(1)
    true_range = pd.concat(
        [
            pd.Series((mid_high - mid_low) * pip_factor, index=timestamps),
            (pd.Series(mid_high, index=timestamps) - previous).abs() * pip_factor,
            (pd.Series(mid_low, index=timestamps) - previous).abs() * pip_factor,
        ],
        axis=1,
    ).max(axis=1)
    for window in (14, 48, 288):
        frame[f"atr_{window}"] = true_range.rolling(window).mean()
        frame[f"volatility_{window}"] = delta.rolling(window).std()
    for window in (12, 48, 288):
        low = frame["mid_close"].rolling(window).min()
        high = frame["mid_close"].rolling(window).max()
        frame[f"range_location_{window}"] = (
            frame["mid_close"] - low
        ) / (high - low).replace(0.0, np.nan)
    frame["body_atr_ratio"] = (
        pd.Series((mid_close - mid_open) * pip_factor, index=timestamps)
        / frame["atr_14"]
    )
    frame["hour_sin"] = np.sin(2.0 * np.pi * timestamps.hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * timestamps.hour / 24.0)
    frame["weekday_sin"] = np.sin(2.0 * np.pi * timestamps.dayofweek / 7.0)
    frame["weekday_cos"] = np.cos(2.0 * np.pi * timestamps.dayofweek / 7.0)
    frame["lookback_span_hours"] = (
        frame.index.to_series() - frame.index.to_series().shift(288)
    ).dt.total_seconds() / 3600.0
    return frame


def _label_frame(frame, *, horizon_min: int, np, pd):
    horizon_bars = int(horizon_min / 5)
    pip_factor = instrument_pip_factor(str(frame["pair"].iloc[0]))
    labelled = frame.copy()
    timestamp_series = labelled.index.to_series()
    labelled["entry_timestamp_utc"] = timestamp_series.shift(-1)
    labelled["future_timestamp_utc"] = timestamp_series.shift(-horizon_bars)
    entry_mid = (
        labelled["bid_open"].shift(-1) + labelled["ask_open"].shift(-1)
    ) / 2.0
    labelled["target_mid_pips"] = (
        labelled["mid_close"].shift(-horizon_bars) - entry_mid
    ) * pip_factor
    labelled["long_pips"] = (
        labelled["bid_close"].shift(-horizon_bars)
        - labelled["ask_open"].shift(-1)
    ) * pip_factor
    labelled["short_pips"] = (
        labelled["bid_open"].shift(-1)
        - labelled["ask_close"].shift(-horizon_bars)
    ) * pip_factor
    exact_entry = (
        labelled["entry_timestamp_utc"] - timestamp_series
    ).dt.total_seconds() == 300.0
    exact_truth = (
        labelled["future_timestamp_utc"] - timestamp_series
    ).dt.total_seconds() == float(horizon_min * 60)
    complete_lookback = labelled["lookback_span_hours"] <= 26.0
    required = list(MODEL_FEATURES) + [
        "target_mid_pips",
        "long_pips",
        "short_pips",
        "future_timestamp_utc",
    ]
    return labelled.loc[exact_entry & exact_truth & complete_lookback].dropna(
        subset=required
    )


def _chronological_blocks(
    dataset,
    *,
    train_fraction: float,
    validation_fraction: float,
    np,
):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be inside (0, 1)")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction must leave a positive holdout")
    timestamps = np.asarray(sorted(dataset.index.unique()))
    validation_at = timestamps[int(len(timestamps) * train_fraction)]
    holdout_at = timestamps[
        int(len(timestamps) * (train_fraction + validation_fraction))
    ]
    training = dataset.loc[
        (dataset.index < validation_at)
        & (dataset["future_timestamp_utc"] < validation_at)
    ]
    validation = dataset.loc[
        (dataset.index >= validation_at)
        & (dataset.index < holdout_at)
        & (dataset["future_timestamp_utc"] < holdout_at)
    ]
    holdout = dataset.loc[dataset.index >= holdout_at]
    return training, validation, holdout, {
        "validation_from_utc": validation_at.isoformat(),
        "holdout_from_utc": holdout_at.isoformat(),
    }


def _prediction_rows(frame, predictions: Sequence[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for timestamp, row, prediction in zip(
        frame.index,
        frame.itertuples(index=False),
        predictions,
    ):
        rows.append(
            {
                "timestamp_utc": timestamp.isoformat(),
                "entry_timestamp_utc": row.entry_timestamp_utc.isoformat(),
                "future_timestamp_utc": row.future_timestamp_utc.isoformat(),
                "pair": str(row.pair),
                "predicted_pips": round(float(prediction), 6),
                "long_pips": round(float(row.long_pips), 6),
                "short_pips": round(float(row.short_pips), 6),
            }
        )
    return rows


def _prediction_thresholds(predictions: Sequence[float], *, np) -> list[float]:
    absolute = np.abs(predictions)
    values = {0.0}
    for quantile in (0.25, 0.50, 0.70, 0.80, 0.90, 0.95):
        values.add(round(float(np.quantile(absolute, quantile)), 6))
    return sorted(values)


def _holdout_passes(
    metrics: Mapping[str, Any],
    *,
    minimum_trades: int,
    minimum_days: int,
    minimum_pairs: int,
) -> bool:
    trade_lower = metrics.get("one_sided_95_mean_lower_pips")
    daily_lower = metrics.get("one_sided_95_daily_lower_pips")
    profit_factor = metrics.get("profit_factor")
    pair_count = len(metrics.get("by_pair") or {})
    return bool(
        int(metrics.get("trades") or 0) >= minimum_trades
        and int(metrics.get("active_days") or 0) >= minimum_days
        and pair_count >= minimum_pairs
        and isinstance(trade_lower, (int, float))
        and math.isfinite(float(trade_lower))
        and float(trade_lower) > 0.0
        and isinstance(daily_lower, (int, float))
        and math.isfinite(float(daily_lower))
        and float(daily_lower) > 0.0
        and isinstance(profit_factor, (int, float))
        and float(profit_factor) > 1.0
        and float(metrics.get("positive_day_rate") or 0.0) >= 0.55
    )


def _parse_horizons(value: str) -> list[int]:
    horizons: list[int] = []
    for raw in str(value or "").split(","):
        if not raw.strip():
            continue
        parsed = int(raw)
        if parsed <= 0 or parsed > 1440 or parsed % 5 != 0:
            raise ValueError("horizons must be M5 multiples inside [5, 1440]")
        if parsed not in horizons:
            horizons.append(parsed)
    if not horizons:
        raise ValueError("at least one horizon is required")
    return horizons


def _horizon_key(value: int) -> str:
    return f"{value}m"


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
        "# Causal technical forecast training",
        "",
        "Forecast-direction evidence only. Live promotion remains disabled pending an exact S5 TP/SL vehicle and forward shadow.",
        "",
        "| horizon | threshold pips | validation trades | validation mean | holdout trades | holdout mean | holdout lower | candidate |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    by_horizon = report.get("by_horizon")
    for key, row in (by_horizon.items() if isinstance(by_horizon, Mapping) else []):
        validation = row.get("validation_metrics") or {}
        holdout = row.get("holdout_metrics") or {}
        lines.append(
            f"| {key} | {row.get('locked_threshold_pips')} | "
            f"{validation.get('trades')} | {validation.get('mean_pips')} | "
            f"{holdout.get('trades')} | {holdout.get('mean_pips')} | "
            f"{holdout.get('one_sided_95_mean_lower_pips')} | "
            f"{row.get('direction_candidate')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
