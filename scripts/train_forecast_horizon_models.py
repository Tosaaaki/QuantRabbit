#!/usr/bin/env python3
"""Compare causal forecast-orientation models at fixed future horizons.

One directional forecast is re-scored at several predeclared holding horizons
against the same local OANDA bid/ask truth.  Each horizon gets its own
non-overlapping chronological cohort and untouched validation tail.  The
result is research evidence only; choosing the best horizon after reading this
report does not by itself authorize live promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (SRC, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import oanda_history_replay_validate as replay  # noqa: E402
from train_forecast_orientation_model import (  # noqa: E402
    _file_sha256,
    _forecast_technical_windows,
    _load_technical_candles_as_m1,
)

from quant_rabbit.forecast_learning import train_forecast_orientation_model  # noqa: E402
from quant_rabbit.forecast_technical_reconstruction import (  # noqa: E402
    FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT,
    reconstruct_missing_technical_features,
)


def main() -> int:
    args = _parse_args()
    time_from = replay._parse_required_time(args.forecast_from, flag="--from")
    time_to = replay._parse_required_time(args.forecast_to, flag="--to")
    horizons = _parse_horizons(args.horizons)
    history_dirs = replay._history_dirs(
        args.history_dir,
        granularity=args.granularity,
        auto_min_days=args.auto_history_min_days,
    )
    technical_history_dirs = replay._history_dirs(
        args.technical_history_dir,
        granularity=args.technical_granularity,
        auto_min_days=args.auto_history_min_days,
    )
    base_rows, load_stats = replay._load_forecasts(
        args.forecast_history,
        pairs=set(),
        time_from=time_from,
        time_to=time_to,
        min_confidence=None,
        confidence_field="calibrated",
    )

    selected_by_horizon: dict[float, list[replay.ForecastRow]] = {}
    independence_by_horizon: dict[str, dict[str, object]] = {}
    all_selected: list[replay.ForecastRow] = []
    for horizon in horizons:
        relabelled = [
            replace(
                row,
                horizon_min=horizon,
                target_price=None,
                invalidation_price=None,
            )
            for row in base_rows
        ]
        selected, stats = replay._select_independent_forecasts(relabelled)
        selected_by_horizon[horizon] = selected
        all_selected.extend(selected)
        independence_by_horizon[_horizon_key(horizon)] = stats

    truth_windows = _merged_truth_windows(all_selected)
    candles, candle_stats = replay._load_candles(
        history_dirs,
        granularity=args.granularity,
        windows_by_pair=truth_windows,
    )
    technical_candles, technical_candle_stats = _load_technical_candles_as_m1(
        technical_history_dirs,
        source_granularity=args.technical_granularity,
        windows_by_pair=_forecast_technical_windows(
            all_selected,
            lookback_hours=args.technical_lookback_hours,
        ),
    )
    truth_digest = replay._truth_candles_digest(candles)
    technical_digest = replay._truth_candles_digest(technical_candles)
    forecast_history_file_size = args.forecast_history.stat().st_size
    forecast_history_file_digest = _file_sha256(args.forecast_history)

    horizon_reports: dict[str, dict[str, object]] = {}
    args.model_output_dir.mkdir(parents=True, exist_ok=True)
    for horizon in horizons:
        key = _horizon_key(horizon)
        selected = selected_by_horizon[horizon]
        direct, score_stats, no_market, pending = replay._score_forecasts(
            selected,
            candles,
            granularity=args.granularity,
        )
        direct, reconstruction_stats = reconstruct_missing_technical_features(
            direct,
            technical_candles,
            granularity="M1",
            lookback_hours=args.technical_lookback_hours,
        )
        inverse = [
            item
            for item in (replay._contrarian_row(row) for row in direct)
            if item is not None
        ]
        provenance = {
            "forecast_history_path": str(args.forecast_history.resolve()),
            "forecast_history_file_size_bytes": forecast_history_file_size,
            "forecast_history_file_sha256": forecast_history_file_digest,
            "forecast_from_utc_inclusive": time_from.isoformat(),
            "forecast_to_utc_exclusive": time_to.isoformat(),
            "fixed_label_horizon_min": horizon,
            "forecast_rows_sha256": replay._forecast_rows_digest(selected),
            "truth_candles_sha256": truth_digest,
            "technical_history_candles_sha256": technical_digest,
            "technical_reconstruction_contract": (
                FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT
            ),
            "replay_evaluator_sha256": hashlib.sha256(
                Path(replay.__file__).read_bytes()
            ).hexdigest(),
        }
        source = (
            f"{args.granularity.upper()} OANDA bid/ask fixed {horizon:g}m "
            "chronological independent forecast replay"
        )
        model = train_forecast_orientation_model(
            direct,
            inverse,
            train_fraction=args.train_fraction,
            source=source,
            provenance=provenance,
        )
        horizon_reports[key] = {
            "fixed_label_horizon_min": horizon,
            **independence_by_horizon[key],
            **score_stats,
            **reconstruction_stats,
            "unscorable_no_market_rows": len(no_market),
            "pending_future_truth_rows": len(pending),
            "direct_rows": len(direct),
            "inverse_rows": len(inverse),
            "model": model,
        }
        _write_json(args.model_output_dir / f"forecast_orientation_{key}.json", model)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "QR_FORECAST_FIXED_HORIZON_COMPARISON_V1",
        "promotion_allowed": False,
        "promotion_blocker": "MULTI_HORIZON_SELECTION_REQUIRES_NEW_FORWARD_HOLDOUT",
        "horizons_min": horizons,
        "forecast_history_path": str(args.forecast_history.resolve()),
        "history_dirs": [str(path) for path in history_dirs],
        "technical_history_dirs": [str(path) for path in technical_history_dirs],
        **load_stats,
        **candle_stats,
        "technical_candle_load": technical_candle_stats,
        "truth_candles_sha256": truth_digest,
        "technical_history_candles_sha256": technical_digest,
        "by_horizon": horizon_reports,
    }
    _write_json(args.report_output, report)
    _write_text(args.report_output.with_suffix(".md"), _markdown(report))
    print(f"wrote {args.report_output}")
    return 0


def _merged_truth_windows(
    rows: list[replay.ForecastRow],
) -> dict[str, list[tuple[datetime, datetime]]]:
    by_pair: dict[str, list[tuple[datetime, datetime]]] = {}
    for pair, windows in replay._forecast_truth_windows(rows).items():
        by_pair.setdefault(pair, []).extend(windows)
    return {
        pair: replay._merge_windows(windows)
        for pair, windows in by_pair.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-history", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument("--granularity", default="S5")
    parser.add_argument(
        "--technical-history-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--technical-granularity", default="S5")
    parser.add_argument("--technical-lookback-hours", type=float, default=24.0)
    parser.add_argument("--horizons", default="5,15,30,60,240")
    parser.add_argument("--from", dest="forecast_from", required=True)
    parser.add_argument("--to", dest="forecast_to", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--auto-history-min-days", type=float, default=30.0)
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=ROOT / "logs" / "reports" / "forecast_improvement" / "horizon_models",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "forecast_horizon_learning_latest.json",
    )
    return parser.parse_args()


def _parse_horizons(value: str) -> list[float]:
    horizons: list[float] = []
    for raw in str(value or "").split(","):
        if not raw.strip():
            continue
        try:
            parsed = float(raw)
        except ValueError as exc:
            raise ValueError("horizons must be comma-separated minutes") from exc
        if not 0.0 < parsed <= 1440.0:
            raise ValueError("each horizon must be inside (0, 1440]")
        if parsed not in horizons:
            horizons.append(parsed)
    if len(horizons) < 2:
        raise ValueError("at least two fixed horizons are required")
    return horizons


def _horizon_key(value: float) -> str:
    return f"{value:g}m".replace(".", "p")


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


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


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Fixed-horizon forecast learning",
        "",
        "Promotion is disabled until a newly locked forward holdout confirms a preselected horizon.",
        "",
        "| horizon | status | validation n | avg pips | hit rate | orientation accuracy |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    by_horizon = report.get("by_horizon")
    for key, row in (by_horizon.items() if isinstance(by_horizon, dict) else []):
        model = row.get("model") if isinstance(row, dict) else {}
        validation = model.get("validation_metrics") if isinstance(model, dict) else {}
        validation = validation if isinstance(validation, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                (
                    str(key),
                    str(model.get("status")),
                    str(validation.get("n")),
                    str(validation.get("selected_avg_final_pips")),
                    str(validation.get("selected_hit_rate")),
                    str(validation.get("orientation_accuracy")),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
