#!/usr/bin/env python3
"""Train the forecast keep/invert model from local OANDA bid/ask history."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import oanda_history_replay_validate as replay

from quant_rabbit.forecast_learning import train_forecast_orientation_model
from quant_rabbit.forecast_technical_reconstruction import (
    FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT,
    reconstruct_missing_technical_features,
)


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
    technical_history_dirs: list[Path] = []
    technical_candle_stats: dict[str, object] = {}
    technical_reconstruction_stats: dict[str, object] = {
        "technical_reconstruction_contract": None,
        "technical_reconstruction_enabled": False,
        "reconstructed_rows": 0,
    }
    technical_candles: dict[str, list[replay.QuoteCandle]] = {}
    if args.technical_history_dir:
        technical_history_dirs = replay._history_dirs(
            args.technical_history_dir,
            granularity=args.technical_granularity,
            auto_min_days=0.0,
        )
        technical_candles, technical_candle_stats = _load_technical_candles_as_m1(
            technical_history_dirs,
            source_granularity=args.technical_granularity,
            windows_by_pair=_forecast_technical_windows(
                rows,
                lookback_hours=args.technical_lookback_hours,
            ),
        )
        direct, technical_reconstruction_stats = (
            reconstruct_missing_technical_features(
                direct,
                technical_candles,
                granularity="M1",
                lookback_hours=args.technical_lookback_hours,
            )
        )
        technical_reconstruction_stats = {
            **technical_reconstruction_stats,
            "technical_reconstruction_enabled": True,
        }
    inverse = [
        item
        for item in (replay._contrarian_row(row) for row in direct)
        if item is not None
    ]
    source = f"{args.granularity} OANDA bid/ask chronological independent forecast replay"
    provenance = {
        "granularity": str(args.granularity).upper(),
        "forecast_history_path": str(args.forecast_history.resolve()),
        "forecast_history_file_size_bytes": args.forecast_history.stat().st_size,
        "forecast_history_file_sha256": _file_sha256(args.forecast_history),
        "forecast_from_utc_inclusive": time_from.isoformat(),
        "forecast_to_utc_exclusive": time_to.isoformat(),
        "forecast_rows_sha256": replay._forecast_rows_digest(rows),
        "truth_candles_sha256": replay._truth_candles_digest(candles),
        "replay_evaluator_sha256": hashlib.sha256(
            Path(replay.__file__).read_bytes()
        ).hexdigest(),
    }
    if technical_candles:
        provenance.update(
            {
                "technical_reconstruction_contract": (
                    FORECAST_TECHNICAL_RECONSTRUCTION_CONTRACT
                ),
                "technical_history_source_granularity": str(
                    args.technical_granularity
                ).upper(),
                "technical_reconstruction_granularity": "M1",
                "technical_history_candles_sha256": (
                    replay._truth_candles_digest(technical_candles)
                ),
                "technical_reconstruction_evaluator_sha256": hashlib.sha256(
                    Path(
                        sys.modules[
                            reconstruct_missing_technical_features.__module__
                        ].__file__
                    ).read_bytes()
                ).hexdigest(),
            }
        )
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
        "technical_history_dirs": [str(path) for path in technical_history_dirs],
        "truth_semantics": (
            "UP ask-entry/bid-exit; DOWN bid-entry/ask-exit; observed OANDA rows "
            "must be ordered on whole cadence multiples and omitted no-tick buckets "
            "remain absent"
        ),
        **load_stats,
        **independence_stats,
        **candle_stats,
        **score_stats,
        "technical_candle_load": technical_candle_stats,
        **technical_reconstruction_stats,
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
        "--technical-history-dir",
        type=Path,
        action="append",
        help=(
            "optional local OANDA bid/ask history used only to causally "
            "reconstruct missing legacy technical features"
        ),
    )
    parser.add_argument("--technical-granularity", default="M1")
    parser.add_argument("--technical-lookback-hours", type=float, default=24.0)
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


def _forecast_technical_windows(
    rows: list[replay.ForecastRow],
    *,
    lookback_hours: float,
) -> dict[str, list[tuple[datetime, datetime]]]:
    lookback = timedelta(hours=float(lookback_hours))
    by_pair: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in rows:
        by_pair.setdefault(row.pair, []).append(
            (row.timestamp_utc - lookback, row.timestamp_utc)
        )
    return {
        pair: replay._merge_windows(windows)
        for pair, windows in by_pair.items()
    }


def _load_technical_candles_as_m1(
    history_dirs: list[Path],
    *,
    source_granularity: str,
    windows_by_pair: dict[str, list[tuple[datetime, datetime]]],
) -> tuple[dict[str, list[replay.QuoteCandle]], dict[str, object]]:
    """Stream S5/M1 truth into causal M1 technical bars.

    The normal replay loader retains every source candle because exits need
    their native cadence. Technical reconstruction only needs M1 OHLC, so
    downsampling while reading avoids keeping tens of millions of S5 objects
    in memory. Each file is aggregated independently before duplicate minute
    bars are reconciled across overlapping history fetches.
    """

    granularity = str(source_granularity or "").strip().upper()
    if granularity not in {"S5", "M1"}:
        raise ValueError("technical history source granularity must be S5 or M1")
    by_pair: dict[str, dict[datetime, replay.QuoteCandle]] = collections.defaultdict(dict)
    window_starts = {
        pair: [start for start, _end in windows]
        for pair, windows in windows_by_pair.items()
    }
    conflict_keys: set[tuple[str, datetime]] = set()
    files = raw_rows = filtered = skipped = duplicate_minutes = conflicting_minutes = 0

    def store(candle: replay.QuoteCandle) -> None:
        nonlocal duplicate_minutes, conflicting_minutes
        key = (candle.pair, candle.timestamp_utc)
        if key in conflict_keys:
            conflicting_minutes += 1
            return
        existing = by_pair[candle.pair].get(candle.timestamp_utc)
        if existing is None:
            by_pair[candle.pair][candle.timestamp_utc] = candle
        elif existing == candle:
            duplicate_minutes += 1
        else:
            conflicting_minutes += 1
            conflict_keys.add(key)
            del by_pair[candle.pair][candle.timestamp_utc]

    for history_dir in history_dirs:
        for path in sorted(history_dir.glob(f"*/*_{granularity}_BA_*.jsonl*")):
            if not replay._is_supported_history_file(path):
                continue
            pair = path.parent.name.upper()
            if pair not in windows_by_pair:
                continue
            files += 1
            bucket_items: list[replay.QuoteCandle] = []
            bucket_at: datetime | None = None
            with replay._open_history_text(path) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        candle = replay._candle_from_payload(payload)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        skipped += 1
                        continue
                    if candle is None or candle.pair != pair:
                        skipped += 1
                        continue
                    raw_rows += 1
                    if not replay._timestamp_in_windows(
                        candle.timestamp_utc,
                        windows_by_pair[pair],
                        window_starts[pair],
                    ):
                        filtered += 1
                        continue
                    minute = candle.timestamp_utc.astimezone(timezone.utc).replace(
                        second=0,
                        microsecond=0,
                    )
                    if bucket_at is not None and minute != bucket_at:
                        store(_aggregate_quote_minute(bucket_items, timestamp_utc=bucket_at))
                        bucket_items = []
                    bucket_at = minute
                    bucket_items.append(candle)
            if bucket_at is not None and bucket_items:
                store(_aggregate_quote_minute(bucket_items, timestamp_utc=bucket_at))
    sorted_by_pair = {
        pair: [items[timestamp] for timestamp in sorted(items)]
        for pair, items in by_pair.items()
    }
    return sorted_by_pair, {
        "technical_history_source_granularity": granularity,
        "technical_history_output_granularity": "M1",
        "technical_history_files": files,
        "technical_history_raw_rows": raw_rows,
        "technical_history_filtered_rows": filtered,
        "technical_history_skipped_rows": skipped,
        "technical_history_duplicate_minutes": duplicate_minutes,
        "technical_history_conflicting_minutes": conflicting_minutes,
        "technical_history_pairs": len(sorted_by_pair),
        "technical_history_m1_candles": sum(
            len(items) for items in sorted_by_pair.values()
        ),
    }


def _aggregate_quote_minute(
    candles: list[replay.QuoteCandle],
    *,
    timestamp_utc: datetime,
) -> replay.QuoteCandle:
    ordered = sorted(candles, key=lambda candle: candle.timestamp_utc)

    def side(name: str) -> replay.Ohlc:
        values = [getattr(candle, name) for candle in ordered]
        return replay.Ohlc(
            o=values[0].o,
            h=max(value.h for value in values),
            l=min(value.l for value in values),
            c=values[-1].c,
        )

    return replay.QuoteCandle(
        timestamp_utc=timestamp_utc,
        pair=ordered[0].pair,
        bid=side("bid"),
        ask=side("ask"),
    )


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    training = (model.get("training_metrics") or {}) if isinstance(model, dict) else {}
    validation = (model.get("validation_metrics") or {}) if isinstance(model, dict) else {}
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
            f"- reconstructed technical rows: `{report.get('reconstructed_rows', 0)}`",
            f"- technical reconstruction contract: `{report.get('technical_reconstruction_contract')}`",
            "",
            "The selector always returns DIRECT or INVERSE. It never creates a no-trade answer.",
        ]
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
