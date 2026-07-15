#!/usr/bin/env python3
"""Train the forecast keep/invert model from local OANDA bid/ask history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import oanda_history_replay_validate as replay

from quant_rabbit.forecast_learning import train_forecast_orientation_model


def main() -> int:
    args = _parse_args()
    time_from = replay._parse_required_time(args.forecast_from, flag="--from")
    time_to = replay._parse_required_time(args.forecast_to, flag="--to")
    history_dirs = replay._history_dirs(
        args.history_dir,
        granularity=args.granularity,
        auto_min_days=args.auto_history_min_days,
    )
    rows, load_stats = replay._load_forecasts(
        args.forecast_history,
        pairs=set(),
        time_from=time_from,
        time_to=time_to,
        min_confidence=None,
        confidence_field="calibrated",
    )
    rows, independence_stats = replay._select_independent_forecasts(rows)
    candles, candle_stats = replay._load_candles(
        history_dirs,
        granularity=args.granularity,
        windows_by_pair=replay._forecast_truth_windows(rows),
    )
    direct, score_stats, no_market, pending = replay._score_forecasts(
        rows,
        candles,
        granularity=args.granularity,
    )
    inverse = [
        item
        for item in (replay._contrarian_row(row) for row in direct)
        if item is not None
    ]
    source = f"{args.granularity} OANDA bid/ask chronological independent forecast replay"
    provenance = {
        "granularity": str(args.granularity).upper(),
        "forecast_from_utc_inclusive": time_from.isoformat(),
        "forecast_to_utc_exclusive": time_to.isoformat(),
        "forecast_rows_sha256": replay._forecast_rows_digest(rows),
        "truth_candles_sha256": replay._truth_candles_digest(candles),
        "replay_evaluator_sha256": hashlib.sha256(
            Path(replay.__file__).read_bytes()
        ).hexdigest(),
    }
    model = train_forecast_orientation_model(
        direct,
        inverse,
        train_fraction=args.train_fraction,
        source=source,
        provenance=provenance,
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "training_provenance": provenance,
        "history_dirs": [str(path) for path in history_dirs],
        "truth_semantics": (
            "UP ask-entry/bid-exit; DOWN bid-entry/ask-exit; observed OANDA rows "
            "must be ordered on whole cadence multiples and omitted no-tick buckets "
            "remain absent"
        ),
        **load_stats,
        **independence_stats,
        **candle_stats,
        **score_stats,
        "unscorable_no_market_rows": len(no_market),
        "pending_future_truth_rows": len(pending),
        "direct_rows": len(direct),
        "inverse_rows": len(inverse),
        "model": model,
    }
    _write_json(args.model_output, model)
    _write_json(args.report_output, report)
    markdown = _markdown(report)
    _write_text(args.report_output.with_suffix(".md"), markdown)
    print(f"wrote {args.model_output}")
    print(f"wrote {args.report_output}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-history", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument("--granularity", default="S5")
    parser.add_argument("--from", dest="forecast_from", required=True)
    parser.add_argument("--to", dest="forecast_to", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--auto-history-min-days", type=float, default=30.0)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "config" / "forecast_orientation_model_v1.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT
        / "logs"
        / "reports"
        / "forecast_improvement"
        / "forecast_learning_latest.json",
    )
    return parser.parse_args()


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
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    training = model.get("training_metrics") if isinstance(model, dict) else {}
    validation = model.get("validation_metrics") if isinstance(model, dict) else {}
    return "\n".join(
        [
            "# Forecast orientation learning",
            "",
            f"- status: `{model.get('status')}`",
            f"- ordinary correction enabled: `{model.get('ordinary_forecast_correction_enabled')}`",
            f"- scored bid/ask forecasts: `{report.get('direct_rows')}`",
            f"- training selected avg pips: `{training.get('selected_avg_final_pips')}`",
            f"- validation direct avg pips: `{validation.get('direct_avg_final_pips')}`",
            f"- validation always-invert avg pips: `{validation.get('inverse_avg_final_pips')}`",
            f"- validation learned avg pips: `{validation.get('selected_avg_final_pips')}`",
            f"- validation learned hit rate: `{validation.get('selected_hit_rate')}`",
            f"- validation orientation accuracy: `{validation.get('orientation_accuracy')}`",
            f"- technical feature training status: `{model.get('technical_feature_training_status')}`",
            "",
            "The selector always returns DIRECT or INVERSE. It never creates a no-trade answer.",
        ]
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
