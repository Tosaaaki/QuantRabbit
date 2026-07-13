#!/usr/bin/env python3
"""Replay forecast_history against locally fetched OANDA bid/ask candles.

This is an audit-only validator for data produced by ``oanda_history_fetch.py``.
It never calls broker write endpoints. UP forecasts enter at ask and exit on bid;
DOWN forecasts enter at bid and exit on ask, so spread cost is included.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import fcntl
import gzip
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_rabbit.instruments import instrument_pip_factor
from quant_rabbit.strategy.forecast_technical_context import (
    verify_forecast_technical_context,
)


DIRECTIONAL = {"UP", "DOWN"}
DEFAULT_TP_GRID_PIPS = (2.0, 5.0, 10.0)
DEFAULT_SL_GRID_PIPS = (2.0, 4.0, 7.0)
DEFAULT_EDGE_MIN_SAMPLES = 30
DEFAULT_EDGE_MIN_DIRECTIONAL_HIT_RATE = 0.60
DEFAULT_EDGE_MIN_AVG_FINAL_PIPS = 0.0
DEFAULT_EDGE_MIN_AVG_REALIZED_PIPS = 0.5
DEFAULT_EDGE_MIN_WIN_RATE = 0.55
DEFAULT_EDGE_MIN_PROFIT_FACTOR = 1.5
DEFAULT_NEGATIVE_MIN_SAMPLES = 30
DEFAULT_NEGATIVE_MAX_DIRECTIONAL_HIT_RATE = 0.45
DEFAULT_NEGATIVE_MAX_AVG_FINAL_PIPS = 0.0
DEFAULT_NEGATIVE_MAX_AVG_REALIZED_PIPS = -0.5
DEFAULT_NEGATIVE_MAX_WIN_RATE = 0.40
DEFAULT_NEGATIVE_MAX_PROFIT_FACTOR = 0.75
# Daily-stability gates are audit quality controls, not market thresholds:
# a replay edge must be seen on multiple JST campaign days and must not be
# dominated by one day before it can be called a stable daily route.
DEFAULT_STABLE_MIN_ACTIVE_DAYS = 3
DEFAULT_STABLE_MAX_DAILY_SAMPLE_SHARE = 0.70
DEFAULT_STABLE_MIN_POSITIVE_DAY_RATE = 2.0 / 3.0
# Auto history discovery should prefer multi-week/month replay datasets when
# they exist. The value is an audit coverage floor, not a market edge threshold.
DEFAULT_AUTO_HISTORY_MIN_DAYS = 30.0
JST = timezone(timedelta(hours=9), "JST")
NEW_YORK = ZoneInfo("America/New_York")
_HISTORY_FILE_WINDOW_RE = re.compile(
    r"_(?P<granularity>[A-Z0-9]+)_BA_"
    r"(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)\.jsonl(?:\.gz)?$"
)


@dataclass(frozen=True)
class Ohlc:
    o: float
    h: float
    l: float
    c: float


@dataclass(frozen=True)
class QuoteCandle:
    timestamp_utc: datetime
    pair: str
    bid: Ohlc
    ask: Ohlc


@dataclass(frozen=True)
class ForecastRow:
    source_index: int
    timestamp_utc: datetime
    pair: str
    direction: str
    confidence: float | None
    current_price: float | None
    target_price: float | None
    invalidation_price: float | None
    horizon_min: float
    cycle_id: str | None
    raw_confidence: float | None = None
    calibration_multiplier: float | None = None
    up_score: float | None = None
    down_score: float | None = None
    range_score: float | None = None
    driver_families: tuple[str, ...] = ()
    drivers_against_families: tuple[str, ...] = ()
    technical_context_v1: dict[str, Any] | None = None
    technical_context_status: str = "MISSING"

    @property
    def score_margin(self) -> float | None:
        return _score_margin(
            self.direction,
            self.up_score,
            self.down_score,
            self.range_score,
        )

    @property
    def range_competition(self) -> str:
        return _range_competition(self.up_score, self.down_score, self.range_score)

    @property
    def utc_session_bucket(self) -> str:
        return _utc_session_bucket(self.timestamp_utc)


def main() -> int:
    args = _parse_args()
    with ExitStack() as resources:
        return _run(args, resources=resources)


def _run(args: argparse.Namespace, *, resources: ExitStack) -> int:
    forecast_from = _parse_required_time(args.forecast_from, flag="--from") if args.forecast_from else None
    forecast_to = _parse_required_time(args.forecast_to, flag="--to") if args.forecast_to else None
    if forecast_from is not None and forecast_to is not None and forecast_from >= forecast_to:
        raise ValueError("--from must be earlier than --to")
    history_dirs = _history_dirs(
        args.history_dir,
        granularity=args.granularity,
        auto_min_days=args.auto_history_min_days,
    )
    pair_filter = _parse_pair_filter(args.pairs)
    rows, load_stats = _load_forecasts(
        args.forecast_history,
        pairs=pair_filter,
        time_from=forecast_from,
        time_to=forecast_to,
        min_confidence=args.min_confidence,
        confidence_field=args.confidence_field,
    )
    if args.independent_non_overlap:
        rows, independence_stats = _select_independent_forecasts(rows)
    else:
        independence_stats = {
            "independent_non_overlap": False,
            "independent_input_rows": len(rows),
            "independent_selected_rows": len(rows),
            "skipped_overlapping_rows": 0,
        }
    candles_by_pair, candle_stats = _load_candles(
        history_dirs,
        granularity=args.granularity,
        windows_by_pair=_forecast_truth_windows(rows),
    )
    if args.allow_repeat_experiment and not str(args.repeat_reason or "").strip():
        raise ValueError("--allow-repeat-experiment requires --repeat-reason")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = out_dir / f"oanda_history_replay_validate_{run_ts}.json"
    md_out = out_dir / f"oanda_history_replay_validate_{run_ts}.md"
    latest_json = out_dir / "oanda_history_replay_validate_latest.json"
    latest_md = out_dir / "oanda_history_replay_validate_latest.md"
    experiment = _experiment_identity(
        args=args,
        rows=rows,
        candles_by_pair=candles_by_pair,
        history_dirs=history_dirs,
        forecast_from=forecast_from,
        forecast_to=forecast_to,
        load_stats=load_stats,
        candle_stats=candle_stats,
    )
    experiment_dir = out_dir / "experiments" / experiment["experiment_id"]
    canonical_json = experiment_dir / "report.json"
    canonical_md = experiment_dir / "report.md"
    manifest_path = experiment_dir / "manifest.json"
    repeat_reason = str(args.repeat_reason or "").strip() or None
    if args.independent_non_overlap:
        resources.enter_context(_acquire_experiment_lock(experiment_dir))
        json_out = canonical_json
        md_out = canonical_md
        if (
            _experiment_is_complete(
                canonical_json,
                canonical_md,
                manifest_path,
                experiment_id=experiment["experiment_id"],
            )
            and not args.allow_repeat_experiment
        ):
            print(f"experiment already evaluated: {canonical_json}")
            print("use --allow-repeat-experiment only when an intentional rerun is required")
            return 0
        pending_manifest = {
            **experiment,
            "status": "PENDING",
            "repeat_reason": repeat_reason,
        }
        _write_text_atomic(
            manifest_path,
            json.dumps(pending_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        )
    validation_now = datetime.now(timezone.utc)
    results, score_stats, unscorable_no_market_rows, pending_future_truth_rows = _score_forecasts(
        rows,
        candles_by_pair,
        now_utc=validation_now,
        granularity=args.granularity,
    )
    contrarian_results = [
        item for item in (_contrarian_row(row) for row in results)
        if item is not None
    ]
    exit_grid = _exit_grid(results, tp_grid=args.tp_grid_pips, sl_grid=args.sl_grid_pips)
    segment_exit_grids = {
        "by_pair_direction": _segment_exit_grids(
            results,
            ("pair", "direction"),
            tp_grid=args.tp_grid_pips,
            sl_grid=args.sl_grid_pips,
            min_n=args.min_group_samples,
        ),
        "by_pair_forecast_direction_trade_direction": _segment_exit_grids(
            contrarian_results,
            ("pair", "forecast_direction", "direction"),
            tp_grid=args.tp_grid_pips,
            sl_grid=args.sl_grid_pips,
            min_n=args.min_group_samples,
        ),
        "by_pair_forecast_direction_trade_direction_horizon": _segment_exit_grids(
            contrarian_results,
            ("pair", "forecast_direction", "direction", "horizon_bucket"),
            tp_grid=args.tp_grid_pips,
            sl_grid=args.sl_grid_pips,
            min_n=args.min_group_samples,
        ),
        "by_pair_forecast_direction_trade_direction_confidence": _segment_exit_grids(
            contrarian_results,
            ("pair", "forecast_direction", "direction", "confidence_bucket"),
            tp_grid=args.tp_grid_pips,
            sl_grid=args.sl_grid_pips,
            min_n=args.min_group_samples,
        ),
        "by_pair_forecast_direction_trade_direction_horizon_confidence": _segment_exit_grids(
            contrarian_results,
            ("pair", "forecast_direction", "direction", "horizon_bucket", "confidence_bucket"),
            tp_grid=args.tp_grid_pips,
            sl_grid=args.sl_grid_pips,
            min_n=args.min_group_samples,
        ),
    }
    split = _train_validation_exit_selection(
        results,
        tp_grid=args.tp_grid_pips,
        sl_grid=args.sl_grid_pips,
        train_fraction=args.train_fraction,
        min_train_samples=args.min_train_samples,
        min_validation_samples=args.min_validation_samples,
    )
    proof_eligibility = _proof_eligibility(
        args=args,
        forecast_from=forecast_from,
        forecast_to=forecast_to,
        load_stats=load_stats,
        candle_stats=candle_stats,
        score_stats=score_stats,
        evaluated_rows=len(results),
        split=split,
    )

    precision_rules = _bidask_precision_rules(
        segment_exit_grids["by_pair_direction"],
        contrarian_segment_rows=[
            row
            for name in (
                "by_pair_forecast_direction_trade_direction",
                "by_pair_forecast_direction_trade_direction_horizon",
                "by_pair_forecast_direction_trade_direction_confidence",
                "by_pair_forecast_direction_trade_direction_horizon_confidence",
            )
            for row in segment_exit_grids[name]
        ],
        granularity=args.granularity,
        audit_report=str(json_out),
        edge_min_samples=args.edge_min_samples,
        edge_min_directional_hit_rate=args.edge_min_directional_hit_rate,
        edge_min_avg_final_pips=args.edge_min_avg_final_pips,
        edge_min_avg_realized_pips=args.edge_min_avg_realized_pips,
        edge_min_win_rate=args.edge_min_win_rate,
        edge_min_profit_factor=args.edge_min_profit_factor,
        negative_min_samples=args.negative_min_samples,
        negative_max_directional_hit_rate=args.negative_max_directional_hit_rate,
        negative_max_avg_final_pips=args.negative_max_avg_final_pips,
        negative_max_avg_realized_pips=args.negative_max_avg_realized_pips,
        negative_max_win_rate=args.negative_max_win_rate,
        negative_max_profit_factor=args.negative_max_profit_factor,
        stable_min_active_days=args.stable_min_active_days,
        stable_max_daily_sample_share=args.stable_max_daily_sample_share,
        stable_min_positive_day_rate=args.stable_min_positive_day_rate,
    )
    if not proof_eligibility["eligible"]:
        precision_rules = _block_unverified_positive_rules(
            precision_rules,
            blockers=proof_eligibility["blockers"],
        )
    sample_coverage = _forecast_sample_coverage(
        rows,
        results,
        unscorable_no_market_rows=unscorable_no_market_rows,
        pending_future_truth_rows=pending_future_truth_rows,
        min_directional_samples=args.edge_min_samples,
        min_active_days=args.stable_min_active_days,
    )
    price_truth_coverage = _price_truth_coverage(
        load_stats=load_stats,
        candle_stats=candle_stats,
        score_stats=score_stats,
        sample_coverage=sample_coverage,
        granularity=args.granularity,
        edge_min_samples=args.edge_min_samples,
        now_utc=validation_now,
    )

    report = {
        "generated_at_utc": _iso(datetime.now(timezone.utc)),
        "source": str(args.forecast_history),
        "history_dirs": [str(path) for path in history_dirs],
        "truth_source": (
            f"local OANDA {args.granularity} bid/ask candles; UP entry=ask/exit=bid, "
            "DOWN entry=bid/exit=ask; same-candle TP+SL ambiguity counts as stop-first loss"
        ),
        "granularity": args.granularity,
        "selection_contract": {
            "audit_mode": args.audit_mode,
            "proof_eligible": proof_eligibility["eligible"],
            "proof_blockers": proof_eligibility["blockers"],
            "forecast_from_utc_inclusive": _iso(forecast_from) if forecast_from is not None else None,
            "forecast_to_utc_exclusive": _iso(forecast_to) if forecast_to is not None else None,
            "min_confidence": args.min_confidence,
            "confidence_field": args.confidence_field,
            "confidence_filter_before_non_overlap": True,
            "independent_non_overlap_per_pair": bool(args.independent_non_overlap),
            "independence_limit": (
                "same-pair horizons do not overlap; cross-pair currency correlation is not removed"
            ),
        },
        "segment_contract": {
            "min_group_samples": args.min_group_samples,
            "default_inclusion_rule": "group n >= min_group_samples",
            "exceptions": [
                "by_direction is global and includes every observed direction",
                "by_confidence is global and includes every observed calibrated-confidence bucket",
            ],
            "diagnostic_only": True,
        },
        "exit_grid_config": {
            "take_profit_pips": list(args.tp_grid_pips),
            "stop_loss_pips": list(args.sl_grid_pips),
            "train_fraction": args.train_fraction,
            "min_train_samples": args.min_train_samples,
            "min_validation_samples": args.min_validation_samples,
        },
        **load_stats,
        **independence_stats,
        **candle_stats,
        **score_stats,
        "summary": _summary(results),
        "max_evaluated_horizon_min": (
            max(float(row.get("horizon_min") or 0.0) for row in results)
            if results
            else None
        ),
        "price_truth_coverage": price_truth_coverage,
        "forecast_sample_coverage": sample_coverage,
        "segments": {
            "by_direction": _group(results, ("direction",)),
            "by_pair": _group(results, ("pair",), min_n=args.min_group_samples),
            "by_pair_direction": _group(results, ("pair", "direction"), min_n=args.min_group_samples),
            "by_horizon": _group(results, ("horizon_bucket",), min_n=args.min_group_samples),
            "by_confidence": _group(
                results,
                ("confidence_bucket",),
                min_n=1,
            ),
            "by_raw_confidence": _group(
                results,
                ("raw_confidence_bucket",),
                min_n=args.min_group_samples,
            ),
            "by_score_margin": _group(
                results,
                ("score_margin_bucket",),
                min_n=args.min_group_samples,
            ),
            "by_range_competition": _group(
                results,
                ("range_competition",),
                min_n=args.min_group_samples,
            ),
            "by_session": _group(
                results,
                ("utc_session_bucket",),
                min_n=args.min_group_samples,
            ),
            "by_technical_context_completeness": _group(
                results,
                ("technical_context_complete",),
                min_n=1,
            ),
            "by_technical_regime": _group(
                results,
                ("technical_regime",),
                min_n=args.min_group_samples,
            ),
            "by_technical_atr_band": _group(
                results,
                ("technical_atr_band",),
                min_n=args.min_group_samples,
            ),
            "by_technical_spread_band": _group(
                results,
                ("technical_spread_band",),
                min_n=args.min_group_samples,
            ),
            "by_technical_range_location_24h": _group(
                results,
                ("technical_range_location_24h",),
                min_n=args.min_group_samples,
            ),
            "by_technical_structure_alignment": _group(
                results,
                ("technical_structure_alignment",),
                min_n=args.min_group_samples,
            ),
            "by_month": _group(results, ("month",), min_n=args.min_group_samples),
            "by_pair_direction_confidence": _group(
                results,
                ("pair", "direction", "confidence_bucket"),
                min_n=args.min_group_samples,
            ),
            "by_primary_driver_family": _group(
                results,
                ("primary_driver_family",),
                min_n=args.min_group_samples,
            ),
            "by_primary_driver_family_direction": _group(
                results,
                ("primary_driver_family", "direction"),
                min_n=args.min_group_samples,
            ),
            "by_driver_family_presence": _group_driver_family_presence(
                results,
                min_n=args.min_group_samples,
            ),
            "by_against_driver_family_presence": _group_against_driver_family_presence(
                results,
                min_n=args.min_group_samples,
            ),
        },
        "exit_grid": exit_grid,
        "segment_exit_grids": segment_exit_grids,
        "precision_rules": precision_rules,
        "train_validation_exit_selection": split,
    }
    completed_experiment = {
        **experiment,
        "status": "COMPLETE" if args.independent_non_overlap else "UNLOCKED_DIAGNOSTIC",
        "repeat_reason": repeat_reason,
    }
    report["experiment"] = completed_experiment
    report_json = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    report_markdown = _markdown(report)
    _write_text_atomic(json_out, report_json)
    _write_text_atomic(md_out, report_markdown)
    if args.independent_non_overlap:
        completed_manifest = {
            **completed_experiment,
            "report_json_sha256": hashlib.sha256(report_json.encode("utf-8")).hexdigest(),
            "report_md_sha256": hashlib.sha256(report_markdown.encode("utf-8")).hexdigest(),
        }
        _write_text_atomic(
            manifest_path,
            json.dumps(completed_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        )
    # Latest pointers move only after the canonical report is COMPLETE. A
    # crash may leave latest stale, but never pointing at an unverified run.
    _write_text_atomic(latest_json, report_json)
    _write_text_atomic(latest_md, report_markdown)
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-history", type=Path, default=Path("data/forecast_history.jsonl"))
    parser.add_argument(
        "--pairs",
        default="",
        help="optional comma-separated pair filter, e.g. EUR_USD,GBP_JPY",
    )
    parser.add_argument("--history-dir", type=Path, action="append")
    parser.add_argument("--granularity", default="S5")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/reports/forecast_improvement"))
    parser.add_argument("--from", dest="forecast_from", help="inclusive forecast timestamp in UTC/ISO-8601")
    parser.add_argument("--to", dest="forecast_to", help="exclusive forecast timestamp in UTC/ISO-8601")
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument(
        "--audit-mode",
        choices=("DIAGNOSTIC", "LOCKED_HOLDOUT", "FORWARD"),
        default="DIAGNOSTIC",
        help="distinguish exploratory diagnostics from pre-locked proof cohorts",
    )
    parser.add_argument(
        "--confidence-field",
        choices=("calibrated", "raw"),
        default="calibrated",
        help="field filtered by --min-confidence; filtering happens before non-overlap selection",
    )
    parser.add_argument(
        "--independent-non-overlap",
        action="store_true",
        help="keep only the first forecast per pair until its horizon expires",
    )
    parser.add_argument(
        "--allow-repeat-experiment",
        action="store_true",
        help="intentionally rerun an identical independent experiment",
    )
    parser.add_argument("--repeat-reason", help="required reason for --allow-repeat-experiment")
    parser.add_argument("--auto-history-min-days", type=float, default=DEFAULT_AUTO_HISTORY_MIN_DAYS)
    parser.add_argument("--tp-grid-pips", type=_parse_float_csv, default=DEFAULT_TP_GRID_PIPS)
    parser.add_argument("--sl-grid-pips", type=_parse_float_csv, default=DEFAULT_SL_GRID_PIPS)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--min-train-samples", type=int, default=20)
    parser.add_argument("--min-validation-samples", type=int, default=10)
    parser.add_argument("--min-group-samples", type=int, default=5)
    parser.add_argument("--edge-min-samples", type=int, default=DEFAULT_EDGE_MIN_SAMPLES)
    parser.add_argument(
        "--edge-min-directional-hit-rate",
        type=float,
        default=DEFAULT_EDGE_MIN_DIRECTIONAL_HIT_RATE,
    )
    parser.add_argument("--edge-min-avg-final-pips", type=float, default=DEFAULT_EDGE_MIN_AVG_FINAL_PIPS)
    parser.add_argument("--edge-min-avg-realized-pips", type=float, default=DEFAULT_EDGE_MIN_AVG_REALIZED_PIPS)
    parser.add_argument("--edge-min-win-rate", type=float, default=DEFAULT_EDGE_MIN_WIN_RATE)
    parser.add_argument("--edge-min-profit-factor", type=float, default=DEFAULT_EDGE_MIN_PROFIT_FACTOR)
    parser.add_argument("--negative-min-samples", type=int, default=DEFAULT_NEGATIVE_MIN_SAMPLES)
    parser.add_argument(
        "--negative-max-directional-hit-rate",
        type=float,
        default=DEFAULT_NEGATIVE_MAX_DIRECTIONAL_HIT_RATE,
    )
    parser.add_argument("--negative-max-avg-final-pips", type=float, default=DEFAULT_NEGATIVE_MAX_AVG_FINAL_PIPS)
    parser.add_argument("--negative-max-avg-realized-pips", type=float, default=DEFAULT_NEGATIVE_MAX_AVG_REALIZED_PIPS)
    parser.add_argument("--negative-max-win-rate", type=float, default=DEFAULT_NEGATIVE_MAX_WIN_RATE)
    parser.add_argument("--negative-max-profit-factor", type=float, default=DEFAULT_NEGATIVE_MAX_PROFIT_FACTOR)
    parser.add_argument("--stable-min-active-days", type=int, default=DEFAULT_STABLE_MIN_ACTIVE_DAYS)
    parser.add_argument("--stable-max-daily-sample-share", type=float, default=DEFAULT_STABLE_MAX_DAILY_SAMPLE_SHARE)
    parser.add_argument("--stable-min-positive-day-rate", type=float, default=DEFAULT_STABLE_MIN_POSITIVE_DAY_RATE)
    return parser.parse_args()


def _parse_pair_filter(value: str | Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    return {
        part.strip().upper()
        for part in parts
        if part.strip()
    }


def _history_dirs(
    explicit: Sequence[Path] | None,
    *,
    granularity: str = "S5",
    auto_min_days: float = DEFAULT_AUTO_HISTORY_MIN_DAYS,
) -> list[Path]:
    if explicit:
        return _explicit_history_dirs(
            explicit,
            granularity=str(granularity or "").upper(),
            min_days=float(auto_min_days),
        )
    root = Path("logs/replay/oanda_history")
    multi_month = _discover_multi_month_history_dirs(
        root,
        granularity=str(granularity or "").upper(),
        min_days=float(auto_min_days),
    )
    if multi_month:
        selected: list[Path] = []
        seen: set[Path] = set()
        for item in multi_month:
            _append_unique_path(selected, seen, item)
        for item in _discover_history_summary_dirs(root, granularity=str(granularity or "").upper()):
            _append_unique_path(selected, seen, item)
        return selected
    latest = Path("logs/replay/oanda_history/latest_summary.json")
    if not latest.exists():
        raise FileNotFoundError("missing logs/replay/oanda_history/latest_summary.json; pass --history-dir")
    payload = json.loads(latest.read_text(encoding="utf-8"))
    output_dir = payload.get("output_dir")
    if not output_dir:
        raise RuntimeError("latest_summary.json has no output_dir; pass --history-dir")
    return [Path(str(output_dir))]


def _explicit_history_dirs(
    explicit: Sequence[Path],
    *,
    granularity: str,
    min_days: float,
) -> list[Path]:
    selected: list[Path] = []
    seen: set[Path] = set()
    for raw_path in explicit:
        path = Path(raw_path)
        matched = False
        discovered = _discover_multi_month_history_dirs(
            path,
            granularity=granularity,
            min_days=min_days,
        )
        if discovered:
            for item in discovered:
                _append_unique_path(selected, seen, item)
            matched = True
        summary_dirs = _discover_history_summary_dirs(path, granularity=granularity)
        if summary_dirs:
            for item in summary_dirs:
                _append_unique_path(selected, seen, item)
            matched = True
        if matched:
            continue
        latest = path / "latest_summary.json"
        if latest.exists():
            try:
                payload = json.loads(latest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
            output_dir = payload.get("output_dir")
            if output_dir:
                _append_unique_path(selected, seen, Path(str(output_dir)))
                continue
        _append_unique_path(selected, seen, path)
    return selected


def _append_unique_path(selected: list[Path], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    selected.append(path)


def _discover_multi_month_history_dirs(
    root: Path,
    *,
    granularity: str,
    min_days: float,
) -> list[Path]:
    selected: list[Path] = []
    seen: set[Path] = set()
    if not root.exists():
        return selected
    for summary_path in sorted(root.glob("**/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        granularities = {str(item).upper() for item in payload.get("granularities") or []}
        if granularity not in granularities:
            continue
        window = payload.get("window") or {}
        start = _parse_time(str(window.get("from") or "")) if window.get("from") else None
        end = _parse_time(str(window.get("to") or "")) if window.get("to") else None
        if start is None or end is None:
            continue
        if (end - start).total_seconds() / 86400.0 < min_days:
            continue
        output_dir = Path(str(payload.get("output_dir") or summary_path.parent))
        if not output_dir.exists():
            continue
        resolved = output_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append(output_dir)
    for candle_path in sorted(root.glob(f"**/*_{granularity}_BA_*.jsonl*")):
        days = _history_file_window_days(candle_path, granularity=granularity)
        if days is None or days < min_days:
            continue
        output_dir = candle_path.parent.parent
        if not output_dir.exists():
            continue
        resolved = output_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append(output_dir)
    return sorted(selected, key=lambda path: str(path))


def _discover_history_summary_dirs(root: Path, *, granularity: str) -> list[Path]:
    selected: list[Path] = []
    seen: set[Path] = set()
    if not root.exists():
        return selected
    for summary_path in sorted(root.glob("**/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        granularities = {str(item).upper() for item in payload.get("granularities") or []}
        if granularity not in granularities:
            continue
        output_dir = Path(str(payload.get("output_dir") or summary_path.parent))
        if not output_dir.exists():
            continue
        _append_unique_path(selected, seen, output_dir)
    return sorted(selected, key=lambda path: str(path))


def _history_file_window_days(path: Path, *, granularity: str) -> float | None:
    match = _HISTORY_FILE_WINDOW_RE.search(path.name)
    if not match:
        return None
    if match.group("granularity").upper() != granularity.upper():
        return None
    start = datetime.strptime(match.group("start"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    end = datetime.strptime(match.group("end"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if end <= start:
        return None
    return (end - start).total_seconds() / 86400.0


def _parse_float_csv(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    out = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not out:
        raise argparse.ArgumentTypeError("grid must contain at least one numeric pip value")
    return out


def _load_candles(
    history_dirs: Sequence[Path],
    *,
    granularity: str,
    windows_by_pair: dict[str, list[tuple[datetime, datetime]]] | None = None,
) -> tuple[dict[str, list[QuoteCandle]], dict[str, Any]]:
    by_pair: dict[str, dict[datetime, QuoteCandle]] = collections.defaultdict(dict)
    window_starts_by_pair = {
        pair: [start for start, _end in windows]
        for pair, windows in (windows_by_pair or {}).items()
    }
    files = 0
    rows = 0
    filtered = 0
    skipped = 0
    duplicate_candles = 0
    conflicting_candles = 0
    conflict_keys: set[tuple[str, datetime]] = set()
    for history_dir in history_dirs:
        for path in sorted(history_dir.glob(f"*/*_{granularity}_BA_*.jsonl*")):
            if not _is_supported_history_file(path):
                continue
            path_pair = path.parent.name.upper()
            if windows_by_pair is not None and path_pair not in windows_by_pair:
                continue
            files += 1
            with _open_history_text(path) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        candle = _candle_from_payload(payload)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        skipped += 1
                        continue
                    if candle is None:
                        skipped += 1
                        continue
                    if candle.pair != path_pair:
                        skipped += 1
                        continue
                    rows += 1
                    if windows_by_pair is not None and not _timestamp_in_windows(
                        candle.timestamp_utc,
                        windows_by_pair.get(candle.pair, []),
                        window_starts_by_pair.get(candle.pair, []),
                    ):
                        filtered += 1
                        continue
                    key = (candle.pair, candle.timestamp_utc)
                    if key in conflict_keys:
                        conflicting_candles += 1
                        continue
                    existing = by_pair[candle.pair].get(candle.timestamp_utc)
                    if existing is None:
                        by_pair[candle.pair][candle.timestamp_utc] = candle
                    elif existing == candle:
                        duplicate_candles += 1
                    else:
                        conflicting_candles += 1
                        conflict_keys.add(key)
                        del by_pair[candle.pair][candle.timestamp_utc]
    sorted_by_pair = {
        pair: [items[key] for key in sorted(items)]
        for pair, items in by_pair.items()
    }
    return sorted_by_pair, {
        "history_files": files,
        "history_raw_rows": rows,
        "history_filtered_rows": filtered,
        "history_skipped_rows": skipped,
        "history_duplicate_candles": duplicate_candles,
        "history_conflicting_candles": conflicting_candles,
        "history_pairs": len(sorted_by_pair),
        "history_candles": sum(len(items) for items in sorted_by_pair.values()),
    }


def _is_supported_history_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


def _open_history_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _forecast_truth_windows(rows: Sequence[ForecastRow]) -> dict[str, list[tuple[datetime, datetime]]]:
    """Build compact candle windows needed to score forecasts.

    The one-minute pad is a clock-alignment margin for local candle timestamps,
    not a trading threshold.
    """

    pad = timedelta(minutes=1)
    by_pair: dict[str, list[tuple[datetime, datetime]]] = collections.defaultdict(list)
    for row in rows:
        by_pair[row.pair].append(
            (
                row.timestamp_utc - pad,
                row.timestamp_utc + timedelta(minutes=row.horizon_min) + pad,
            )
        )
    return {
        pair: _merge_windows(windows)
        for pair, windows in by_pair.items()
    }


def _merge_windows(windows: Sequence[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _timestamp_in_windows(
    timestamp: datetime,
    windows: Sequence[tuple[datetime, datetime]],
    starts: Sequence[datetime],
) -> bool:
    if not windows or not starts:
        return False
    idx = bisect.bisect_right(starts, timestamp) - 1
    if idx < 0:
        return False
    start, end = windows[idx]
    return start <= timestamp <= end


def _candle_from_payload(payload: dict[str, Any]) -> QuoteCandle | None:
    if payload.get("complete") is False:
        return None
    pair = str(payload.get("pair") or "").upper()
    timestamp = _parse_time(payload.get("time"))
    bid = _ohlc_from_payload(payload.get("bid"))
    ask = _ohlc_from_payload(payload.get("ask"))
    if not pair or timestamp is None or bid is None or ask is None:
        return None
    if any(
        getattr(ask, field) <= getattr(bid, field)
        for field in ("o", "h", "l", "c")
    ):
        return None
    return QuoteCandle(timestamp_utc=timestamp, pair=pair, bid=bid, ask=ask)


def _ohlc_from_payload(payload: object) -> Ohlc | None:
    if not isinstance(payload, dict):
        return None
    try:
        value = Ohlc(
            o=float(payload["o"]),
            h=float(payload["h"]),
            l=float(payload["l"]),
            c=float(payload["c"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    prices = (value.o, value.h, value.l, value.c)
    if not all(math.isfinite(price) and price > 0.0 for price in prices):
        return None
    if value.h < max(value.o, value.c, value.l):
        return None
    if value.l > min(value.o, value.c, value.h):
        return None
    return value


def _load_forecasts(
    path: Path,
    *,
    pairs: set[str] | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    min_confidence: float | None = None,
    confidence_field: str = "calibrated",
) -> tuple[list[ForecastRow], dict[str, Any]]:
    pair_filter = {str(item).upper() for item in (pairs or set()) if str(item).strip()}
    if confidence_field not in {"calibrated", "raw"}:
        raise ValueError("confidence_field must be calibrated or raw")
    if min_confidence is not None and not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    canonical_by_key: dict[tuple[Any, ...], ForecastRow] = {}
    conflict_keys: set[tuple[Any, ...]] = set()
    conflict_members: dict[tuple[Any, ...], list[ForecastRow]] = collections.defaultdict(list)
    raw_directional = 0
    skipped_pair_filter = 0
    skipped_time_filter = 0
    skipped_confidence_filter = 0
    skipped_duplicate = 0
    skipped_invalid = 0
    raw_confidence_missing_rows = 0
    calibrated_confidence_missing_rows = 0
    technical_context_missing_rows = 0
    technical_context_invalid_rows = 0
    technical_context_incomplete_rows = 0
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue
            if not isinstance(payload, dict):
                skipped_invalid += 1
                continue
            direction = str(payload.get("direction") or "").upper()
            if direction not in DIRECTIONAL:
                continue
            raw_directional += 1
            timestamp = _parse_time(payload.get("timestamp_utc"))
            pair = str(payload.get("pair") or "").upper().strip()
            if timestamp is None or not pair:
                skipped_invalid += 1
                continue
            confidence = _safe_float(payload.get("confidence"))
            raw_confidence = _safe_float(payload.get("raw_confidence"))
            calibration_multiplier = _safe_float(payload.get("calibration_multiplier"))
            target = _safe_float(payload.get("target_price"))
            invalidation = _safe_float(payload.get("invalidation_price"))
            current_price = _safe_float(payload.get("current_price"))
            cycle_id = str(payload.get("cycle_id") or "").strip() or None
            technical_context_raw = payload.get("technical_context_v1")
            technical_context_valid, technical_context_error = verify_forecast_technical_context(
                technical_context_raw,
                pair=pair,
                current_price=current_price,
            )
            technical_context = (
                dict(technical_context_raw)
                if technical_context_valid and isinstance(technical_context_raw, dict)
                else None
            )
            technical_context_status = (
                "VALID"
                if technical_context_valid
                else str(technical_context_error or "TECHNICAL_CONTEXT_INVALID")
            )
            key = _dedupe_key(
                cycle_id=cycle_id,
                pair=pair,
                timestamp=timestamp,
                direction=direction,
                confidence=confidence,
                target=target,
                invalidation=invalidation,
            )
            candidate = ForecastRow(
                source_index=idx,
                timestamp_utc=timestamp,
                pair=pair,
                direction=direction,
                confidence=confidence,
                raw_confidence=raw_confidence,
                calibration_multiplier=calibration_multiplier,
                up_score=_safe_float(payload.get("up_score")),
                down_score=_safe_float(payload.get("down_score")),
                range_score=_safe_float(payload.get("range_score")),
                current_price=current_price,
                target_price=target,
                invalidation_price=invalidation,
                horizon_min=_safe_horizon(payload.get("horizon_min")),
                cycle_id=cycle_id,
                driver_families=_forecast_driver_families(payload.get("drivers_for")),
                drivers_against_families=_forecast_driver_families(payload.get("drivers_against")),
                technical_context_v1=technical_context,
                technical_context_status=technical_context_status,
            )
            if key in conflict_keys:
                conflict_members[key].append(candidate)
                continue
            existing = canonical_by_key.get(key)
            if existing is None:
                canonical_by_key[key] = candidate
            elif _forecast_rows_equivalent_for_dedupe(existing, candidate):
                skipped_duplicate += 1
            else:
                conflict_keys.add(key)
                conflict_members[key].extend((existing, candidate))
                del canonical_by_key[key]
    canonical_rows = sorted(canonical_by_key.values(), key=lambda row: row.source_index)
    scoped_conflicting_forecast_groups = sum(
        1
        for members in conflict_members.values()
        if any(
            (not pair_filter or row.pair in pair_filter)
            and (time_from is None or row.timestamp_utc >= time_from)
            and (time_to is None or row.timestamp_utc < time_to)
            for row in members
        )
    )
    # Canonicalize duplicate emissions before any policy filter. Otherwise a
    # later duplicate can be selectively admitted merely because an earlier
    # copy failed the requested confidence/time/pair cohort.
    rows: list[ForecastRow] = []
    for row in canonical_rows:
        if pair_filter and row.pair not in pair_filter:
            skipped_pair_filter += 1
            continue
        if time_from is not None and row.timestamp_utc < time_from:
            skipped_time_filter += 1
            continue
        if time_to is not None and row.timestamp_utc >= time_to:
            skipped_time_filter += 1
            continue
        if row.confidence is None:
            calibrated_confidence_missing_rows += 1
        selected_confidence = row.confidence
        if confidence_field == "raw":
            selected_confidence = row.raw_confidence
            if row.raw_confidence is None:
                raw_confidence_missing_rows += 1
        if min_confidence is not None and (
            selected_confidence is None or selected_confidence < min_confidence
        ):
            skipped_confidence_filter += 1
            continue
        if row.technical_context_status == "TECHNICAL_CONTEXT_MISSING":
            technical_context_missing_rows += 1
        elif row.technical_context_status != "VALID":
            technical_context_invalid_rows += 1
        elif not bool(((row.technical_context_v1 or {}).get("completeness") or {}).get("complete")):
            technical_context_incomplete_rows += 1
        rows.append(row)
    return rows, {
        "raw_directional_rows": raw_directional,
        "canonical_directional_rows": len(canonical_rows),
        "pair_filter": sorted(pair_filter),
        "skipped_pair_filter_rows": skipped_pair_filter,
        "forecast_time_from_utc": _iso(time_from) if time_from is not None else None,
        "forecast_time_to_utc": _iso(time_to) if time_to is not None else None,
        "confidence_filter_field": confidence_field,
        "min_confidence_filter": min_confidence,
        "skipped_time_filter_rows": skipped_time_filter,
        "skipped_confidence_filter_rows": skipped_confidence_filter,
        "raw_confidence_missing_rows": raw_confidence_missing_rows,
        "calibrated_confidence_missing_rows": calibrated_confidence_missing_rows,
        "technical_context_missing_rows": technical_context_missing_rows,
        "technical_context_invalid_rows": technical_context_invalid_rows,
        "technical_context_incomplete_rows": technical_context_incomplete_rows,
        "deduped_directional_rows": len(rows),
        "skipped_duplicate_rows": skipped_duplicate,
        "skipped_conflicting_forecast_rows": scoped_conflicting_forecast_groups,
        "global_conflicting_forecast_groups": len(conflict_members),
        "skipped_invalid_rows": skipped_invalid,
    }


def _select_independent_forecasts(
    rows: Sequence[ForecastRow],
) -> tuple[list[ForecastRow], dict[str, Any]]:
    """Select non-overlapping forecast windows after all policy filters.

    Repeated forecasts inside one unresolved horizon share most of the same
    future candles. Counting them as separate trials inflates precision and
    lets one market move dominate calibration. Different pairs retain their
    own clocks so a forecast on EUR_USD does not suppress an unrelated pair.
    """

    selected: list[ForecastRow] = []
    accepted_until: dict[str, datetime] = {}
    skipped = 0
    for row in sorted(rows, key=lambda item: (item.timestamp_utc, item.source_index)):
        current_until = accepted_until.get(row.pair)
        if current_until is not None and row.timestamp_utc < current_until:
            skipped += 1
            continue
        selected.append(row)
        accepted_until[row.pair] = _forecast_truth_end(row)
    return selected, {
        "independent_non_overlap": True,
        "independent_input_rows": len(rows),
        "independent_selected_rows": len(selected),
        "skipped_overlapping_rows": skipped,
    }


def _forecast_driver_families(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    families: list[str] = []
    for item in value:
        family = _driver_family(str(item or ""))
        if family not in families:
            families.append(family)
    return tuple(families)


def _technical_context_digest(value: dict[str, Any] | None) -> str | None:
    if not isinstance(value, dict):
        return None
    valid, _error = verify_forecast_technical_context(value)
    if not valid:
        return None
    stored = str(value.get("context_sha256") or "").strip().lower()
    return stored if len(stored) == 64 else None


def _technical_context_value(value: dict[str, Any] | None, *path: str) -> object:
    current: object = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _technical_context_label(value: dict[str, Any] | None, *path: str) -> str:
    raw = _technical_context_value(value, *path)
    text = str(raw or "").strip().upper()
    return text or "MISSING"


def _technical_structure_alignment(
    value: dict[str, Any] | None,
    *,
    forecast_direction: str,
) -> str:
    structure_direction = _technical_context_label(value, "structure", "primary_direction")
    if structure_direction not in DIRECTIONAL:
        return "MISSING"
    return "ALIGNED" if structure_direction == forecast_direction else "OPPOSED"


def _driver_family(text: str) -> str:
    lower = text.lower()
    if "wick-only" in lower and "trap fade" in lower:
        return "WICK_TRAP_FADE"
    if "reversal-from-extreme" in lower:
        return "EXTREME_REVERSAL"
    if "range breakout pending" in lower:
        return "RANGE_BREAKOUT_PENDING"
    if "range breakout confirmed" in lower:
        return "RANGE_BREAKOUT_CONFIRMED"
    if "divergence" in lower:
        return "DIVERGENCE"
    if ("equal-high" in lower or "equal-low" in lower) and "fade" in lower:
        return "LIQUIDITY_SWEEP_FADE"
    if "hvn" in lower or "price magnet" in lower:
        return "HVN_MAGNET"
    if "aroon" in lower or "momentum" in lower:
        return "MOMENTUM"
    if any(
        marker in lower
        for marker in (
            "inside bar",
            "morning star",
            "evening star",
            "shooting star",
            "white soldiers",
            "black crows",
            "engulf",
            "doji",
        )
    ):
        return "CANDLE_PATTERN"
    if "market location" in lower:
        return "MARKET_LOCATION"
    return "OTHER"


def _dedupe_key(
    *,
    cycle_id: str | None,
    pair: str,
    timestamp: datetime,
    direction: str,
    confidence: float | None,
    target: float | None,
    invalidation: float | None,
) -> tuple[Any, ...]:
    if cycle_id:
        return ("cycle", cycle_id, pair)
    return (
        "bucket",
        pair,
        timestamp.replace(microsecond=0).isoformat(),
        direction,
        round(confidence or 0.0, 6),
        target,
        invalidation,
    )


def _forecast_rows_equivalent_for_dedupe(first: ForecastRow, second: ForecastRow) -> bool:
    """Compare duplicate emissions at the one-second ledger resolution.

    Legacy writers emitted the same forecast several times a few milliseconds
    apart. Their canonical key intentionally buckets timestamps to one second;
    sub-second drift alone is therefore not a conflict. Any material forecast
    field change is quarantined instead of selecting the more favorable copy.
    """

    def payload(row: ForecastRow) -> tuple[Any, ...]:
        return (
            row.timestamp_utc.replace(microsecond=0),
            row.pair,
            row.direction,
            row.confidence,
            row.raw_confidence,
            row.calibration_multiplier,
            row.up_score,
            row.down_score,
            row.range_score,
            row.current_price,
            row.target_price,
            row.invalidation_price,
            row.horizon_min,
            row.cycle_id,
            row.driver_families,
            row.drivers_against_families,
            _technical_context_digest(row.technical_context_v1),
            row.technical_context_status,
        )

    return payload(first) == payload(second)


def _score_forecasts(
    rows: Sequence[ForecastRow],
    candles_by_pair: dict[str, list[QuoteCandle]],
    *,
    now_utc: datetime | None = None,
    granularity: str = "S5",
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ForecastRow], list[ForecastRow]]:
    results: list[dict[str, Any]] = []
    skipped_no_pair = 0
    skipped_no_window = 0
    missing_windows: list[ForecastRow] = []
    unscorable_no_market: list[ForecastRow] = []
    pending_future_truth: list[ForecastRow] = []
    incomplete_truth_rows: list[ForecastRow] = []
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    candle_delta = _granularity_delta(granularity)
    for row in rows:
        if _forecast_truth_end(row) > now:
            pending_future_truth.append(row)
            continue
        if _is_likely_fx_no_market_window(row):
            unscorable_no_market.append(row)
            continue
        candles = candles_by_pair.get(row.pair)
        if not candles:
            skipped_no_pair += 1
            missing_windows.append(row)
            continue
        times = [c.timestamp_utc for c in candles]
        start = bisect.bisect_left(times, row.timestamp_utc)
        last_complete_open = _forecast_truth_end(row) - candle_delta
        end = bisect.bisect_right(times, last_complete_open)
        if start >= len(candles) or end <= start:
            skipped_no_window += 1
            missing_windows.append(row)
            continue
        window = candles[start:end]
        if not _truth_window_complete(
            window,
            candle_delta=candle_delta,
            window_start=row.timestamp_utc,
            window_end=_forecast_truth_end(row),
        ):
            incomplete_truth_rows.append(row)
            continue
        scored = _score_one(row, window, candle_delta=candle_delta)
        if scored is not None:
            results.append(scored)
    return (
        results,
        {
            "evaluated_rows": len(results),
            "evaluated_confidence_missing_rows": sum(
                1 for item in results if item.get("confidence") is None
            ),
            "skipped_no_pair_candles": skipped_no_pair,
            "skipped_no_price_window": skipped_no_window,
            "missing_price_window_groups": _missing_price_window_groups(missing_windows),
            "unscorable_no_market_rows": len(unscorable_no_market),
            "unscorable_no_market_window_groups": _missing_price_window_groups(unscorable_no_market),
            "pending_future_truth_rows": len(pending_future_truth),
            "pending_future_truth_window_groups": _missing_price_window_groups(pending_future_truth),
            "skipped_incomplete_truth_window_rows": len(incomplete_truth_rows),
            "incomplete_truth_window_groups": _missing_price_window_groups(incomplete_truth_rows),
        },
        unscorable_no_market,
        pending_future_truth,
    )


def _granularity_delta(granularity: str) -> timedelta:
    seconds = {
        "S5": 5,
        "S10": 10,
        "S15": 15,
        "S30": 30,
        "M1": 60,
        "M2": 120,
        "M4": 240,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H2": 7200,
        "H3": 10800,
        "H4": 14400,
        "H6": 21600,
        "H8": 28800,
        "H12": 43200,
        "D": 86400,
    }.get(str(granularity or "").upper())
    if seconds is None:
        raise ValueError(f"unsupported replay granularity {granularity!r}")
    return timedelta(seconds=seconds)


def _truth_window_complete(
    window: Sequence[QuoteCandle],
    *,
    candle_delta: timedelta,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
) -> bool:
    if not window:
        return False
    expected = candle_delta.total_seconds()
    if expected <= 0:
        return False
    if window_start is not None:
        leading_gap = (window[0].timestamp_utc - window_start).total_seconds()
        # The first executable candle may start after an unaligned forecast,
        # but it must be the immediately following candle. A full interval or
        # more means the leading truth candle is missing.
        if leading_gap < 0 or leading_gap >= expected:
            return False
    if window_end is not None:
        last_close = window[-1].timestamp_utc + candle_delta
        trailing_gap = (window_end - last_close).total_seconds()
        # Only fully closed candles are scored. Less than one residual candle
        # is expected for an unaligned horizon; a full interval means trailing
        # truth is missing.
        if trailing_gap < 0 or trailing_gap >= expected:
            return False
    for previous, current in zip(window, window[1:]):
        gap = (current.timestamp_utc - previous.timestamp_utc).total_seconds()
        if abs(gap - expected) > 1e-6:
            return False
    return True


def _is_likely_fx_no_market_window(row: ForecastRow) -> bool:
    """Identify forecast windows where OANDA has no FX candles to fetch.

    This is a broker-session data-quality classification, not a trading edge
    threshold. Retail FX is normally closed from Friday 17:00 New York time
    until Sunday 17:00 New York time, so forecasts emitted during that interval
    should not keep generating missing-candle fetch work.
    """

    start = row.timestamp_utc.astimezone(timezone.utc)
    end = _forecast_truth_end(row)
    if end <= start:
        return False
    local_start = start.astimezone(NEW_YORK)
    local_end = end.astimezone(NEW_YORK)
    date = local_start.date() - timedelta(days=7)
    last_date = local_end.date() + timedelta(days=7)
    while date <= last_date:
        if date.weekday() == 4:
            close_start = datetime.combine(date, time(17, 0), tzinfo=NEW_YORK).astimezone(timezone.utc)
            close_end = datetime.combine(
                date + timedelta(days=2),
                time(17, 0),
                tzinfo=NEW_YORK,
            ).astimezone(timezone.utc)
            if start < close_end and end > close_start:
                return True
        date += timedelta(days=1)
    return False


def _forecast_truth_end(row: ForecastRow) -> datetime:
    return row.timestamp_utc.astimezone(timezone.utc) + timedelta(minutes=row.horizon_min)


def _forecast_sample_coverage(
    rows: Sequence[ForecastRow],
    results: Sequence[dict[str, Any]],
    *,
    unscorable_no_market_rows: Sequence[ForecastRow] | None = None,
    pending_future_truth_rows: Sequence[ForecastRow] | None = None,
    min_directional_samples: int,
    min_active_days: int,
) -> dict[str, Any]:
    """Publish pair/direction evidence gaps before precision-rule selection."""

    forecast_counts: collections.Counter[tuple[str, str]] = collections.Counter(
        (row.pair, row.direction) for row in rows
    )
    forecast_days: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in rows:
        forecast_days[(row.pair, row.direction)].add(row.timestamp_utc.astimezone(JST).date().isoformat())
    no_market_rows = list(unscorable_no_market_rows or [])
    no_market_counts: collections.Counter[tuple[str, str]] = collections.Counter(
        (row.pair, row.direction) for row in no_market_rows
    )
    no_market_pair_counts: collections.Counter[str] = collections.Counter(row.pair for row in no_market_rows)
    pending_rows = list(pending_future_truth_rows or [])
    pending_counts: collections.Counter[tuple[str, str]] = collections.Counter(
        (row.pair, row.direction) for row in pending_rows
    )
    pending_pair_counts: collections.Counter[str] = collections.Counter(row.pair for row in pending_rows)

    evaluated_counts: collections.Counter[tuple[str, str]] = collections.Counter(
        (str(row.get("pair") or "").upper(), str(row.get("direction") or "").upper())
        for row in results
        if row.get("pair") and row.get("direction")
    )
    evaluated_days: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for row in results:
        pair = str(row.get("pair") or "").upper()
        direction = str(row.get("direction") or "").upper()
        if not pair or direction not in DIRECTIONAL:
            continue
        evaluated_days[(pair, direction)].add(_campaign_day_jst(row))

    pair_counts: collections.Counter[str] = collections.Counter(row.pair for row in rows)
    pair_evaluated_counts: collections.Counter[str] = collections.Counter(
        str(row.get("pair") or "").upper() for row in results if row.get("pair")
    )
    pair_rows = [
        {
            "pair": pair,
            "forecast_samples": count,
            "evaluated_samples": pair_evaluated_counts.get(pair, 0),
            "unscorable_no_market_samples": no_market_pair_counts.get(pair, 0),
            "pending_future_truth_samples": pending_pair_counts.get(pair, 0),
            "missing_price_truth_samples": max(
                0,
                count
                - pair_evaluated_counts.get(pair, 0)
                - no_market_pair_counts.get(pair, 0)
                - pending_pair_counts.get(pair, 0),
            ),
            "missing_evaluated_samples_to_min_directional": max(
                0,
                min_directional_samples - pair_evaluated_counts.get(pair, 0),
            ),
        }
        for pair, count in pair_counts.items()
    ]
    pair_rows.sort(key=lambda item: (-int(item["evaluated_samples"]), item["pair"]))

    under_sampled: list[dict[str, Any]] = []
    all_keys = sorted(set(forecast_counts) | set(evaluated_counts))
    for pair, direction in all_keys:
        forecast_samples = forecast_counts.get((pair, direction), 0)
        evaluated_samples = evaluated_counts.get((pair, direction), 0)
        forecast_active_days = len(forecast_days.get((pair, direction), set()))
        evaluated_active_days = len(evaluated_days.get((pair, direction), set()))
        unscorable_no_market_samples = no_market_counts.get((pair, direction), 0)
        pending_future_truth_samples = pending_counts.get((pair, direction), 0)
        gaps: list[str] = []
        if evaluated_samples < min_directional_samples:
            gaps.append("INSUFFICIENT_EVALUATED_SAMPLES")
        if evaluated_active_days < min_active_days:
            gaps.append("INSUFFICIENT_ACTIVE_DAYS")
        missing_price_truth_samples = max(
            0,
            forecast_samples
            - evaluated_samples
            - unscorable_no_market_samples
            - pending_future_truth_samples,
        )
        if missing_price_truth_samples > 0 and gaps:
            gaps.append("PRICE_TRUTH_WINDOW_MISSING")
        if unscorable_no_market_samples > 0:
            gaps.append("NO_MARKET_SESSION_UNSCORABLE")
        if pending_future_truth_samples > 0:
            gaps.append("PENDING_FUTURE_TRUTH_WINDOW")
        if not gaps:
            continue
        under_sampled.append(
            {
                "pair": pair,
                "direction": direction,
                "forecast_samples": forecast_samples,
                "forecast_active_days": forecast_active_days,
                "evaluated_samples": evaluated_samples,
                "evaluated_active_days": evaluated_active_days,
                "unscorable_no_market_samples": unscorable_no_market_samples,
                "pending_future_truth_samples": pending_future_truth_samples,
                "missing_price_truth_samples": missing_price_truth_samples,
                "missing_evaluated_samples": max(0, min_directional_samples - evaluated_samples),
                "missing_active_days": max(0, min_active_days - evaluated_active_days),
                "coverage_gap_reasons": gaps,
            }
        )
    under_sampled.sort(
        key=lambda item: (
            -int(item["missing_evaluated_samples"]),
            -int(item["missing_active_days"]),
            item["pair"],
            item["direction"],
        )
    )
    return {
        "min_directional_samples_for_precision_rule": min_directional_samples,
        "min_active_days_for_daily_stability": min_active_days,
        "pair_count": len(pair_counts),
        "pair_direction_count": len(all_keys),
        "unscorable_no_market_samples": len(no_market_rows),
        "pending_future_truth_samples": len(pending_rows),
        "pairs": pair_rows,
        "under_sampled_pair_directions": under_sampled,
    }


def _price_truth_coverage(
    *,
    load_stats: dict[str, Any],
    candle_stats: dict[str, Any],
    score_stats: dict[str, Any],
    sample_coverage: dict[str, Any],
    granularity: str,
    edge_min_samples: int,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Summarize whether replay has enough bid/ask truth to be adoption evidence."""

    raw_rows = _int_metric(load_stats.get("raw_directional_rows"))
    deduped_rows = _int_metric(load_stats.get("deduped_directional_rows"))
    history_files = _int_metric(candle_stats.get("history_files"))
    history_candles = _int_metric(candle_stats.get("history_candles"))
    evaluated_rows = _int_metric(score_stats.get("evaluated_rows"))
    missing_groups = list(score_stats.get("missing_price_window_groups") or [])
    incomplete_groups = list(score_stats.get("incomplete_truth_window_groups") or [])
    fetchable_gap_groups = _merge_price_truth_window_groups(missing_groups, incomplete_groups)
    no_market_groups = list(score_stats.get("unscorable_no_market_window_groups") or [])
    no_market_rows = _int_metric(score_stats.get("unscorable_no_market_rows"))
    pending_future_groups = list(score_stats.get("pending_future_truth_window_groups") or [])
    pending_future_rows = _int_metric(score_stats.get("pending_future_truth_rows"))
    pair_rows = list(sample_coverage.get("pairs") or [])
    under_sampled = list(sample_coverage.get("under_sampled_pair_directions") or [])
    missing_truth_samples = sum(
        max(0, _int_metric(item.get("missing_price_truth_samples")))
        for item in pair_rows
    )
    missing_pairs = sorted(
        {
            str(item.get("pair") or "").upper()
            for item in pair_rows
            if _int_metric(item.get("missing_price_truth_samples")) > 0
        }
    )
    missing_pair_directions = [
        f"{item.get('pair')}:{item.get('direction')}"
        for item in under_sampled
        if "PRICE_TRUTH_WINDOW_MISSING" in (item.get("coverage_gap_reasons") or [])
    ]
    under_sampled_pair_directions = [
        f"{item.get('pair')}:{item.get('direction')}"
        for item in under_sampled
        if item.get("pair") and item.get("direction")
    ]
    under_sampled_missing_samples = sum(
        max(0, _int_metric(item.get("missing_evaluated_samples")))
        for item in under_sampled
    )
    sample_coverage_status = "UNDER_SAMPLED" if under_sampled else "OK"

    if raw_rows <= 0 or deduped_rows <= 0:
        status = "NO_DIRECTIONAL_FORECAST_ROWS"
        reason = "forecast_history has no deduped UP/DOWN rows to replay."
    elif evaluated_rows <= 0 and no_market_rows >= deduped_rows and missing_truth_samples <= 0:
        status = "NO_SCORABLE_MARKET_FORECAST_ROWS"
        reason = "Directional forecast rows exist only in broker no-market windows."
    elif evaluated_rows <= 0 and no_market_rows + pending_future_rows >= deduped_rows and missing_truth_samples <= 0:
        status = "NO_SCORABLE_MATURE_MARKET_FORECAST_ROWS"
        reason = "Directional forecast rows are only broker no-market windows or still waiting for future truth."
    elif history_files <= 0:
        status = "NO_PRICE_HISTORY_FILES"
        reason = "No local OANDA bid/ask candle files matched the forecast pairs, granularity, and windows."
    elif history_candles <= 0:
        status = "NO_PRICE_CANDLES_LOADED"
        reason = "History files were found, but no bid/ask candles survived parsing and window filtering."
    elif evaluated_rows <= 0:
        status = "NO_EVALUATED_PRICE_TRUTH"
        reason = "Forecast rows exist, but none could be scored against local bid/ask candles."
    elif evaluated_rows < int(edge_min_samples):
        status = "INSUFFICIENT_EVALUATED_SAMPLES"
        reason = "Evaluated rows are below the precision-rule sample floor."
    elif missing_truth_samples > 0 or fetchable_gap_groups:
        status = "PARTIAL_PRICE_TRUTH"
        reason = "Some forecast pairs or windows still lack matching bid/ask candle truth."
    else:
        status = "PRICE_TRUTH_OK"
        reason = "All loaded forecast samples have local bid/ask candle truth for this replay window."

    candidate_blocked = status in {
        "NO_DIRECTIONAL_FORECAST_ROWS",
        "NO_PRICE_HISTORY_FILES",
        "NO_PRICE_CANDLES_LOADED",
        "NO_EVALUATED_PRICE_TRUTH",
        "NO_SCORABLE_MATURE_MARKET_FORECAST_ROWS",
        "INSUFFICIENT_EVALUATED_SAMPLES",
    }
    global_blocked = status != "PRICE_TRUTH_OK" or bool(under_sampled)
    if candidate_blocked:
        adoption_level = "NO_REPLAY_EVIDENCE"
    elif global_blocked:
        adoption_level = "PAIR_LOCAL_RANK_ONLY"
    else:
        adoption_level = "FULL_REPLAY_READY"
    blockers: list[str] = []
    if candidate_blocked:
        blockers.append("DO_NOT_PROMOTE_PRECISION_RULE")
    if global_blocked:
        blockers.append("DO_NOT_CLAIM_ALL_CURRENCY_VALIDATION")
    if missing_truth_samples > 0:
        blockers.append("FETCH_MISSING_PRICE_TRUTH")
    if under_sampled and missing_truth_samples <= 0:
        blockers.append("COLLECT_MORE_FORECAST_SAMPLES")
    warnings: list[str] = []
    if no_market_rows > 0:
        warnings.append("FORECAST_ROWS_DURING_BROKER_NO_MARKET_WINDOW")
    if pending_future_rows > 0:
        warnings.append("FORECAST_ROWS_WITH_PENDING_FUTURE_TRUTH_WINDOW")
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    history_fetch_commands = _history_fetch_commands(
        fetchable_gap_groups,
        granularity,
        now_utc=now,
    )

    return {
        "status": status,
        "reason": reason,
        "adoption_level": adoption_level,
        "candidate_rule_validation_blocked": candidate_blocked,
        "global_currency_validation_blocked": global_blocked,
        "blockers": blockers,
        "warnings": warnings,
        "raw_directional_rows": raw_rows,
        "deduped_directional_rows": deduped_rows,
        "evaluated_rows": evaluated_rows,
        "unscorable_no_market_rows": no_market_rows,
        "pending_future_truth_rows": pending_future_rows,
        "required_min_evaluated_rows": int(edge_min_samples),
        "history_files": history_files,
        "history_candles": history_candles,
        "missing_price_truth_samples": missing_truth_samples,
        "missing_price_window_group_count": len(missing_groups),
        "incomplete_price_window_group_count": len(incomplete_groups),
        "fetchable_price_window_group_count": len(fetchable_gap_groups),
        "unscorable_no_market_window_group_count": len(no_market_groups),
        "unscorable_no_market_window_groups": no_market_groups[:12],
        "pending_future_truth_window_group_count": len(pending_future_groups),
        "pending_future_truth_window_groups": pending_future_groups[:12],
        "future_price_truth_window_group_count": _future_missing_window_group_count(
            fetchable_gap_groups,
            now_utc=now,
        ),
        "missing_pairs": missing_pairs,
        "missing_pair_directions": missing_pair_directions[:24],
        "all_currency_sample_coverage_status": sample_coverage_status,
        "under_sampled_pair_direction_count": len(under_sampled_pair_directions),
        "under_sampled_pair_directions": under_sampled_pair_directions[:24],
        "under_sampled_missing_evaluated_samples": under_sampled_missing_samples,
        "history_fetch_command_mode": "WINDOWED",
        "history_fetch_command_count": len(history_fetch_commands),
        "history_fetch_commands": history_fetch_commands,
        "history_fetch_command": _history_fetch_command(
            fetchable_gap_groups,
            granularity,
            now_utc=now,
        ),
    }


def _merge_price_truth_window_groups(
    *group_sets: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for groups in group_sets:
        for group in groups:
            date = str(group.get("date") or "").strip()
            start = _parse_time(group.get("needed_from_utc"))
            end = _parse_time(group.get("needed_to_utc"))
            if not date or start is None or end is None:
                continue
            item = merged.setdefault(
                date,
                {
                    "date": date,
                    "count": 0,
                    "needed_from_utc": start,
                    "needed_to_utc": end,
                    "pairs": set(),
                    "pair_directions": set(),
                },
            )
            item["count"] += _int_metric(group.get("count"))
            item["needed_from_utc"] = min(item["needed_from_utc"], start)
            item["needed_to_utc"] = max(item["needed_to_utc"], end)
            item["pairs"].update(str(value).upper() for value in group.get("pairs") or [])
            item["pair_directions"].update(
                str(value).upper() for value in group.get("pair_directions") or []
            )
    return [
        {
            **item,
            "needed_from_utc": _iso(item["needed_from_utc"]),
            "needed_to_utc": _iso(item["needed_to_utc"]),
            "pairs": sorted(value for value in item["pairs"] if value),
            "pair_directions": sorted(value for value in item["pair_directions"] if value),
        }
        for _date, item in sorted(merged.items())
    ]


def _history_fetch_commands(
    missing_groups: Sequence[dict[str, Any]],
    granularity: str,
    *,
    now_utc: datetime | None = None,
) -> list[dict[str, Any]]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    commands: list[dict[str, Any]] = []
    for group in missing_groups:
        start = _parse_time(group.get("needed_from_utc"))
        end = _parse_time(group.get("needed_to_utc"))
        pairs = sorted(
            {
                str(pair).upper()
                for pair in group.get("pairs") or []
                if str(pair).strip()
            }
        )
        if start is None or end is None or start >= now or not pairs:
            continue
        clamped_end = min(end, now)
        command = _format_history_fetch_command(
            pairs=pairs,
            granularity=granularity,
            start=start,
            end=clamped_end,
        )
        commands.append(
            {
                "date": group.get("date"),
                "forecast_rows_missing_truth": _int_metric(group.get("count")),
                "pairs": pairs,
                "from_utc": _iso(start),
                "to_utc": _iso(clamped_end),
                "clamped_to_now": end > now,
                "command": command,
            }
        )
    commands.sort(key=lambda item: (str(item.get("from_utc") or ""), str(item.get("date") or "")))
    return commands


def _history_fetch_command(
    missing_groups: Sequence[dict[str, Any]],
    granularity: str,
    *,
    now_utc: datetime | None = None,
) -> str | None:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    pairs: set[str] = set()
    starts: list[datetime] = []
    ends: list[datetime] = []
    for group in missing_groups:
        start = _parse_time(group.get("needed_from_utc"))
        end = _parse_time(group.get("needed_to_utc"))
        if start is None or end is None or start >= now:
            continue
        pairs.update(str(pair).upper() for pair in group.get("pairs") or [] if str(pair).strip())
        starts.append(start)
        ends.append(min(end, now))
    if not pairs or not starts or not ends:
        return None
    return _format_history_fetch_command(
        pairs=sorted(pairs),
        granularity=granularity,
        start=min(starts),
        end=max(ends),
    )


def _format_history_fetch_command(
    *,
    pairs: Sequence[str],
    granularity: str,
    start: datetime,
    end: datetime,
) -> str:
    return (
        "PYTHONPATH=src python3 scripts/oanda_history_fetch.py "
        f"--pairs {','.join(pairs)} "
        f"--granularities {str(granularity or '').upper()} "
        "--price BA "
        f"--from {_iso(start)} "
        f"--to {_iso(end)} "
        "--output-dir logs/replay/oanda_history"
    )


def _future_missing_window_group_count(
    missing_groups: Sequence[dict[str, Any]],
    *,
    now_utc: datetime | None = None,
) -> int:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    count = 0
    for group in missing_groups:
        end = _parse_time(group.get("needed_to_utc"))
        if end is not None and end > now:
            count += 1
    return count


def _int_metric(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _missing_price_window_groups(rows: Sequence[ForecastRow]) -> list[dict[str, Any]]:
    buckets: dict[str, list[ForecastRow]] = collections.defaultdict(list)
    for row in rows:
        buckets[row.timestamp_utc.date().isoformat()].append(row)
    out: list[dict[str, Any]] = []
    for date, items in buckets.items():
        needed_from = min(row.timestamp_utc for row in items) - timedelta(minutes=5)
        needed_to = max(row.timestamp_utc + timedelta(minutes=row.horizon_min) for row in items) + timedelta(minutes=5)
        pairs = sorted({row.pair for row in items})
        pair_directions = sorted({f"{row.pair}:{row.direction}" for row in items})
        out.append(
            {
                "date": date,
                "count": len(items),
                "needed_from_utc": _iso(needed_from),
                "needed_to_utc": _iso(needed_to),
                "pairs": pairs,
                "pair_directions": pair_directions,
            }
        )
    out.sort(key=lambda item: item["date"])
    return out


def _score_one(
    row: ForecastRow,
    window: Sequence[QuoteCandle],
    *,
    candle_delta: timedelta | None = None,
) -> dict[str, Any] | None:
    return _score_direction(
        row,
        window,
        direction=row.direction,
        forecast_direction=row.direction,
        candle_delta=candle_delta,
    )


def _score_direction(
    row: ForecastRow,
    window: Sequence[QuoteCandle],
    *,
    direction: str,
    forecast_direction: str,
    candle_delta: timedelta | None = None,
) -> dict[str, Any] | None:
    if not window:
        return None
    direction = str(direction or "").upper()
    forecast_direction = str(forecast_direction or "").upper()
    if direction not in DIRECTIONAL or forecast_direction not in DIRECTIONAL:
        return None
    pip_factor = instrument_pip_factor(row.pair)
    first = window[0]
    last = window[-1]
    effective_delta = candle_delta
    if effective_delta is None and len(window) >= 2:
        effective_delta = window[1].timestamp_utc - window[0].timestamp_utc
    last_close = last.timestamp_utc + effective_delta if effective_delta is not None else last.timestamp_utc
    if direction == "UP":
        entry = first.ask.o
        final_pips = (last.bid.c - entry) * pip_factor
        mfe_pips = max(0.0, (max(c.bid.h for c in window) - entry) * pip_factor)
        mae_pips = max(0.0, (entry - min(c.bid.l for c in window)) * pip_factor)
    else:
        entry = first.bid.o
        final_pips = (entry - last.ask.c) * pip_factor
        mfe_pips = max(0.0, (entry - min(c.ask.l for c in window)) * pip_factor)
        mae_pips = max(0.0, (max(c.ask.h for c in window) - entry) * pip_factor)
    geometry_row = row if direction == row.direction else replace(
        row,
        direction=direction,
        target_price=None,
        invalidation_price=None,
    )
    target_reward_side = _target_is_reward_side(geometry_row, entry)
    invalidation_adverse_side = _invalidation_is_adverse_side(geometry_row, entry)
    target_touch, invalidation_touch, target_first = _target_invalidation_order(
        geometry_row,
        window,
        target_reward_side=target_reward_side,
        invalidation_adverse_side=invalidation_adverse_side,
    )
    reward_pips = (
        abs(float(geometry_row.target_price) - entry) * pip_factor
        if target_reward_side and geometry_row.target_price is not None
        else None
    )
    risk_pips = (
        abs(float(geometry_row.invalidation_price) - entry) * pip_factor
        if invalidation_adverse_side and geometry_row.invalidation_price is not None
        else None
    )
    realized_r = None
    geometry_outcome = "MISSING_GEOMETRY"
    if reward_pips is not None and risk_pips is not None and risk_pips > 0:
        if target_first is True:
            realized_r = reward_pips / risk_pips
            geometry_outcome = "TARGET_FIRST"
        elif target_first is False:
            realized_r = -1.0
            geometry_outcome = "INVALIDATION_FIRST_OR_SAME_BAR"
        else:
            realized_r = final_pips / risk_pips
            geometry_outcome = "NO_TOUCH_MARK_TO_HORIZON"
    return {
        "source_index": row.source_index,
        "timestamp_utc": _iso(row.timestamp_utc),
        "entry_timestamp_utc": _iso(first.timestamp_utc),
        "last_timestamp_utc": _iso(last.timestamp_utc),
        "entry_delay_seconds": (first.timestamp_utc - row.timestamp_utc).total_seconds(),
        "effective_holding_min": (last_close - first.timestamp_utc).total_seconds() / 60.0,
        "unobserved_horizon_tail_seconds": max(
            0.0,
            (_forecast_truth_end(row) - last_close).total_seconds(),
        ),
        "pair": row.pair,
        "direction": direction,
        "forecast_direction": forecast_direction,
        "contrarian": direction != forecast_direction,
        "confidence": row.confidence,
        "raw_confidence": row.raw_confidence,
        "calibration_multiplier": row.calibration_multiplier,
        "up_score": row.up_score,
        "down_score": row.down_score,
        "range_score": row.range_score,
        "score_margin": row.score_margin,
        "score_margin_bucket": _score_margin_bucket(row.score_margin),
        "range_competition": row.range_competition,
        "confidence_bucket": _confidence_bucket(row.confidence),
        "raw_confidence_bucket": _confidence_bucket(row.raw_confidence),
        "driver_families": row.driver_families,
        "drivers_against_families": row.drivers_against_families,
        "primary_driver_family": row.driver_families[0] if row.driver_families else "MISSING",
        "horizon_min": row.horizon_min,
        "horizon_bucket": _horizon_bucket(row.horizon_min),
        "month": row.timestamp_utc.strftime("%Y-%m"),
        "utc_session_bucket": row.utc_session_bucket,
        "technical_context_sha256": _technical_context_digest(row.technical_context_v1),
        "technical_context_complete": bool(
            ((row.technical_context_v1 or {}).get("completeness") or {}).get("complete")
        ),
        "technical_regime": _technical_context_label(
            row.technical_context_v1, "regime", "primary"
        ),
        "technical_atr_band": _technical_context_label(
            row.technical_context_v1, "volatility", "primary_atr_band"
        ),
        "technical_spread_band": _technical_context_label(
            row.technical_context_v1, "execution", "spread_band"
        ),
        "technical_range_location_24h": _technical_context_label(
            row.technical_context_v1, "location", "range_location_24h"
        ),
        "technical_structure_direction": _technical_context_label(
            row.technical_context_v1, "structure", "primary_direction"
        ),
        "technical_structure_alignment": _technical_structure_alignment(
            row.technical_context_v1,
            forecast_direction=forecast_direction,
        ),
        "entry_price": entry,
        "final_pips": final_pips,
        "final_direction_hit": final_pips > 0.0,
        "mfe_pips": mfe_pips,
        "mae_pips": mae_pips,
        "target_touch": target_touch,
        "invalidation_touch": invalidation_touch,
        "target_before_invalidation": target_first,
        "target_reward_side": target_reward_side,
        "invalidation_adverse_side": invalidation_adverse_side,
        "reward_pips": reward_pips,
        "risk_pips": risk_pips,
        "realized_r": realized_r,
        "geometry_outcome": geometry_outcome,
        "_window": window,
    }


def _contrarian_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Re-score the same bid/ask window as a trade fading the forecast side."""

    window = row.get("_window")
    if not isinstance(window, (list, tuple)) or not window:
        return None
    forecast_direction = str(row.get("forecast_direction") or row.get("direction") or "").upper()
    trade_direction = _opposite_direction(forecast_direction)
    pair = str(row.get("pair") or "").upper()
    if not pair or trade_direction is None:
        return None
    pip_factor = instrument_pip_factor(pair)
    first = window[0]
    last = window[-1]
    if trade_direction == "UP":
        entry = first.ask.o
        final_pips = (last.bid.c - entry) * pip_factor
        mfe_pips = max(0.0, (max(c.bid.h for c in window) - entry) * pip_factor)
        mae_pips = max(0.0, (entry - min(c.bid.l for c in window)) * pip_factor)
    else:
        entry = first.bid.o
        final_pips = (entry - last.ask.c) * pip_factor
        mfe_pips = max(0.0, (entry - min(c.ask.l for c in window)) * pip_factor)
        mae_pips = max(0.0, (max(c.ask.h for c in window) - entry) * pip_factor)
    out = dict(row)
    out.update(
        {
            "direction": trade_direction,
            "forecast_direction": forecast_direction,
            "contrarian": True,
            "entry_price": entry,
            "final_pips": final_pips,
            "final_direction_hit": final_pips > 0.0,
            "mfe_pips": mfe_pips,
            "mae_pips": mae_pips,
            "target_touch": None,
            "invalidation_touch": None,
            "target_before_invalidation": None,
            "target_reward_side": None,
            "invalidation_adverse_side": None,
            "reward_pips": None,
            "risk_pips": None,
            "realized_r": None,
            "geometry_outcome": "CONTRARIAN_NO_SOURCE_GEOMETRY",
            "source_direction": forecast_direction,
            "source_final_pips": row.get("final_pips"),
            "source_direction_hit": row.get("final_direction_hit"),
            "source_mfe_pips": row.get("mfe_pips"),
            "source_mae_pips": row.get("mae_pips"),
        }
    )
    return out


def _opposite_direction(direction: str) -> str | None:
    if direction == "UP":
        return "DOWN"
    if direction == "DOWN":
        return "UP"
    return None


def _target_is_reward_side(row: ForecastRow, entry: float) -> bool | None:
    if row.target_price is None:
        return None
    if row.direction == "UP":
        return row.target_price > entry
    return row.target_price < entry


def _invalidation_is_adverse_side(row: ForecastRow, entry: float) -> bool | None:
    if row.invalidation_price is None:
        return None
    if row.direction == "UP":
        return row.invalidation_price < entry
    return row.invalidation_price > entry


def _target_invalidation_order(
    row: ForecastRow,
    window: Sequence[QuoteCandle],
    *,
    target_reward_side: bool | None,
    invalidation_adverse_side: bool | None,
) -> tuple[bool | None, bool | None, bool | None]:
    target = row.target_price if target_reward_side else None
    invalidation = row.invalidation_price if invalidation_adverse_side else None
    if target is None and invalidation is None:
        return None, None, None
    target_touch = False if target is not None else None
    invalidation_touch = False if invalidation is not None else None
    first: str | None = None
    for candle in window:
        target_hit = False
        invalidation_hit = False
        if target is not None:
            if row.direction == "UP":
                target_hit = candle.bid.h >= target
            else:
                target_hit = candle.ask.l <= target
        if invalidation is not None:
            if row.direction == "UP":
                invalidation_hit = candle.bid.l <= invalidation
            else:
                invalidation_hit = candle.ask.h >= invalidation
        if target_hit:
            target_touch = True
        if invalidation_hit:
            invalidation_touch = True
        if first is None:
            if target_hit and invalidation_hit:
                first = "invalidation"
            elif target_hit:
                first = "target"
            elif invalidation_hit:
                first = "invalidation"
    target_first = None if first is None else first == "target"
    return target_touch, invalidation_touch, target_first


def _exit_grid(
    results: Sequence[dict[str, Any]],
    *,
    tp_grid: Sequence[float],
    sl_grid: Sequence[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tp in tp_grid:
        for sl in sl_grid:
            realized = [_simulate_exit(row, take_profit_pips=tp, stop_loss_pips=sl) for row in results]
            out.append(_exit_summary(realized, take_profit_pips=tp, stop_loss_pips=sl))
    out.sort(key=lambda item: (item["avg_realized_pips"], item["profit_factor"]), reverse=True)
    return out


def _simulate_exit(row: dict[str, Any], *, take_profit_pips: float, stop_loss_pips: float) -> dict[str, Any]:
    window = row["_window"]
    direction = row["direction"]
    pair = row["pair"]
    entry = float(row["entry_price"])
    pip_factor = instrument_pip_factor(pair)
    pip = 1.0 / pip_factor
    for candle in window:
        if direction == "UP":
            tp_hit = candle.bid.h >= entry + take_profit_pips * pip
            sl_hit = candle.bid.l <= entry - stop_loss_pips * pip
        else:
            tp_hit = candle.ask.l <= entry - take_profit_pips * pip
            sl_hit = candle.ask.h >= entry + stop_loss_pips * pip
        if sl_hit:
            return {"pips": -float(stop_loss_pips), "reason": "SL"}
        if tp_hit:
            return {"pips": float(take_profit_pips), "reason": "TP"}
    return {"pips": float(row["final_pips"]), "reason": "TIMEOUT"}


def _train_validation_exit_selection(
    results: Sequence[dict[str, Any]],
    *,
    tp_grid: Sequence[float],
    sl_grid: Sequence[float],
    train_fraction: float,
    min_train_samples: int,
    min_validation_samples: int,
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: item["timestamp_utc"])
    if len(ordered) < min_train_samples + min_validation_samples:
        return {
            "status": "INSUFFICIENT_SAMPLES",
            "n": len(ordered),
            "min_required": min_train_samples + min_validation_samples,
        }
    split_at = min(max(int(len(ordered) * train_fraction), min_train_samples), len(ordered) - min_validation_samples)
    validation_start = _parse_time(ordered[split_at].get("timestamp_utc"))
    if validation_start is None:
        return {"status": "INVALID_TIMESTAMPS", "n": len(ordered)}
    # Keep a timestamp cluster wholly on one side and purge any training
    # forecast whose truth horizon reaches into validation. This prevents the
    # optimizer from learning on outcomes that occur during the holdout.
    train_candidates = [
        row
        for row in ordered
        if (_parse_time(row.get("timestamp_utc")) or validation_start) < validation_start
    ]
    validation = [
        row
        for row in ordered
        if (_parse_time(row.get("timestamp_utc")) or validation_start) >= validation_start
    ]
    train = [
        row
        for row in train_candidates
        if (
            (_parse_time(row.get("timestamp_utc")) or validation_start)
            + timedelta(minutes=float(row.get("horizon_min") or 0.0))
        ) <= validation_start
    ]
    purged_train_rows = len(train_candidates) - len(train)
    if len(train) < min_train_samples or len(validation) < min_validation_samples:
        return {
            "status": "INSUFFICIENT_SAMPLES_AFTER_PURGE",
            "n": len(ordered),
            "train_n": len(train),
            "validation_n": len(validation),
            "purged_train_rows": purged_train_rows,
            "validation_start_utc": _iso(validation_start),
            "min_train_samples": min_train_samples,
            "min_validation_samples": min_validation_samples,
        }
    train_grid = _exit_grid(train, tp_grid=tp_grid, sl_grid=sl_grid)
    selected = train_grid[0]
    validation_realized = [
        _simulate_exit(
            row,
            take_profit_pips=selected["take_profit_pips"],
            stop_loss_pips=selected["stop_loss_pips"],
        )
        for row in validation
    ]
    validation_summary = _exit_summary(
        validation_realized,
        take_profit_pips=selected["take_profit_pips"],
        stop_loss_pips=selected["stop_loss_pips"],
    )
    return {
        "status": "OK",
        "train_n": len(train),
        "validation_n": len(validation),
        "purged_train_rows": purged_train_rows,
        "validation_start_utc": _iso(validation_start),
        "selected_by_train": selected,
        "validation": validation_summary,
    }


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    target_known = [r for r in rows if r.get("target_touch") is not None]
    invalidation_known = [r for r in rows if r.get("invalidation_touch") is not None]
    target_first_known = [r for r in rows if r.get("target_before_invalidation") is not None]
    target_geometry_known = [r for r in rows if r.get("target_reward_side") is not None]
    invalidation_geometry_known = [r for r in rows if r.get("invalidation_adverse_side") is not None]
    hits = sum(1 for row in rows if row["final_direction_hit"])
    wilson_lower, wilson_upper = _wilson_interval(hits, len(rows))
    r_rows = [r for r in rows if r.get("realized_r") is not None]
    outcomes = collections.Counter(str(r.get("geometry_outcome") or "UNKNOWN") for r in rows)
    return {
        "n": len(rows),
        "hit_rate": hits / len(rows),
        "hit_wilson95_lower": wilson_lower,
        "hit_wilson95_upper": wilson_upper,
        "avg_final_pips": _mean(r["final_pips"] for r in rows),
        "total_final_pips": sum(float(r["final_pips"]) for r in rows),
        "median_final_pips": _median(r["final_pips"] for r in rows),
        "avg_mfe_pips": _mean(r["mfe_pips"] for r in rows),
        "avg_mae_pips": _mean(r["mae_pips"] for r in rows),
        "median_entry_delay_seconds": _median(
            r["entry_delay_seconds"] for r in rows if r.get("entry_delay_seconds") is not None
        ),
        "median_effective_holding_min": _median(
            r["effective_holding_min"] for r in rows if r.get("effective_holding_min") is not None
        ),
        "median_unobserved_horizon_tail_seconds": _median(
            r["unobserved_horizon_tail_seconds"]
            for r in rows
            if r.get("unobserved_horizon_tail_seconds") is not None
        ),
        "target_touch_rate": _rate(r["target_touch"] for r in target_known) if target_known else None,
        "invalidation_touch_rate": _rate(r["invalidation_touch"] for r in invalidation_known) if invalidation_known else None,
        "target_before_invalidation_rate": (
            _rate(r["target_before_invalidation"] for r in target_first_known) if target_first_known else None
        ),
        "reward_side_target_rate": (
            _rate(r["target_reward_side"] for r in target_geometry_known) if target_geometry_known else None
        ),
        "adverse_side_invalidation_rate": (
            _rate(r["invalidation_adverse_side"] for r in invalidation_geometry_known)
            if invalidation_geometry_known
            else None
        ),
        "geometry_outcomes": dict(sorted(outcomes.items())),
        "realized_r_n": len(r_rows),
        "avg_realized_r": _mean(float(r["realized_r"]) for r in r_rows) if r_rows else None,
        "total_realized_r": sum(float(r["realized_r"]) for r in r_rows) if r_rows else None,
        "positive_realized_r_rate": (
            _rate(float(r["realized_r"]) > 0.0 for r in r_rows) if r_rows else None
        ),
        "avg_reward_pips": _mean(float(r["reward_pips"]) for r in r_rows) if r_rows else None,
        "avg_risk_pips": _mean(float(r["risk_pips"]) for r in r_rows) if r_rows else None,
    }


def _exit_summary(
    realized: Sequence[dict[str, Any]],
    *,
    take_profit_pips: float,
    stop_loss_pips: float,
) -> dict[str, Any]:
    pips = [float(item["pips"]) for item in realized]
    wins = [value for value in pips if value > 0]
    losses = [-value for value in pips if value < 0]
    return {
        "take_profit_pips": take_profit_pips,
        "stop_loss_pips": stop_loss_pips,
        "n": len(pips),
        "avg_realized_pips": _mean(pips),
        "win_rate": _rate(value > 0 for value in pips) if pips else None,
        "profit_factor": (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else (math.inf if wins else 0.0),
        "timeout_rate": _rate(item["reason"] == "TIMEOUT" for item in realized) if realized else None,
        "tp_rate": _rate(item["reason"] == "TP" for item in realized) if realized else None,
        "sl_rate": _rate(item["reason"] == "SL" for item in realized) if realized else None,
    }


def _group(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    min_n: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field) for field in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        if len(items) < min_n:
            continue
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(_summary(items))
        out.append(payload)
    out.sort(key=lambda item: (item["n"], item.get("avg_final_pips") or -999.0), reverse=True)
    return out


def _group_driver_family_presence(
    rows: Sequence[dict[str, Any]],
    *,
    min_n: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        for family in row.get("driver_families") or ("MISSING",):
            buckets[str(family)].append(row)
    out: list[dict[str, Any]] = []
    for family, items in buckets.items():
        if len(items) < min_n:
            continue
        payload = {"driver_family": family}
        payload.update(_summary(items))
        out.append(payload)
    out.sort(key=lambda item: (item["n"], item.get("avg_final_pips") or -999.0), reverse=True)
    return out


def _group_against_driver_family_presence(
    rows: Sequence[dict[str, Any]],
    *,
    min_n: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        for family in row.get("drivers_against_families") or ("MISSING",):
            buckets[str(family)].append(row)
    out: list[dict[str, Any]] = []
    for family, items in buckets.items():
        if len(items) < min_n:
            continue
        payload = {"against_driver_family": family}
        payload.update(_summary(items))
        out.append(payload)
    out.sort(key=lambda item: (item["n"], item.get("avg_final_pips") or -999.0), reverse=True)
    return out


def _segment_exit_grids(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
    *,
    tp_grid: Sequence[float],
    sl_grid: Sequence[float],
    min_n: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(field) for field in fields)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        if len(items) < min_n:
            continue
        grid = _exit_grid(items, tp_grid=tp_grid, sl_grid=sl_grid)
        best = grid[0] if grid else None
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(items),
                "summary": _summary(items),
                "best_exit": best,
                "exit_grid": grid,
            }
        )
        if best:
            payload["daily_stability"] = _daily_exit_stability(
                items,
                take_profit_pips=float(best["take_profit_pips"]),
                stop_loss_pips=float(best["stop_loss_pips"]),
            )
        source_summary = _source_summary(items)
        if source_summary is not None:
            payload["source_summary"] = source_summary
        out.append(payload)
    out.sort(
        key=lambda item: (
            float((item.get("best_exit") or {}).get("avg_realized_pips") or -999.0),
            int(item.get("n") or 0),
        ),
        reverse=True,
    )
    return out


def _daily_exit_stability(
    rows: Sequence[dict[str, Any]],
    *,
    take_profit_pips: float,
    stop_loss_pips: float,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        buckets[_campaign_day_jst(row)].append(row)
    if not buckets:
        return {
            "campaign_timezone": "Asia/Tokyo",
            "active_days": 0,
            "daily_summaries": [],
        }
    daily_summaries: list[dict[str, Any]] = []
    for day, items in sorted(buckets.items()):
        realized = [
            _simulate_exit(item, take_profit_pips=take_profit_pips, stop_loss_pips=stop_loss_pips)
            for item in items
        ]
        pips = [float(item["pips"]) for item in realized]
        daily_summaries.append(
            {
                "date": day,
                "samples": len(items),
                "realized_pips": _round(sum(pips)),
                "avg_realized_pips": _round(_mean(pips)),
                "win_rate": _round(_rate(value > 0.0 for value in pips)),
                "tp_rate": _round(_rate(item["reason"] == "TP" for item in realized)),
                "sl_rate": _round(_rate(item["reason"] == "SL" for item in realized)),
                "timeout_rate": _round(_rate(item["reason"] == "TIMEOUT" for item in realized)),
            }
        )
    total = len(rows)
    samples_by_day = [int(item["samples"]) for item in daily_summaries]
    realized_by_day = [float(item["realized_pips"] or 0.0) for item in daily_summaries]
    active_days = len(daily_summaries)
    positive_days = sum(1 for value in realized_by_day if value > 0.0)
    negative_days = sum(1 for value in realized_by_day if value < 0.0)
    flat_days = active_days - positive_days - negative_days
    return {
        "campaign_timezone": "Asia/Tokyo",
        "active_days": active_days,
        "first_day": daily_summaries[0]["date"],
        "last_day": daily_summaries[-1]["date"],
        "min_daily_samples": min(samples_by_day),
        "max_daily_samples": max(samples_by_day),
        "avg_daily_samples": _round(total / active_days),
        "max_daily_sample_share": _round(max(samples_by_day) / total),
        "positive_days": positive_days,
        "negative_days": negative_days,
        "flat_days": flat_days,
        "positive_day_rate": _round(positive_days / active_days),
        "avg_daily_realized_pips": _round(_mean(realized_by_day)),
        "worst_daily_realized_pips": _round(min(realized_by_day)),
        "best_daily_realized_pips": _round(max(realized_by_day)),
        "daily_summaries": daily_summaries,
    }


def _campaign_day_jst(row: dict[str, Any]) -> str:
    timestamp = _parse_time(row.get("timestamp_utc"))
    if timestamp is None:
        return "unknown"
    return timestamp.astimezone(JST).date().isoformat()


def _daily_stability_status(
    daily: dict[str, Any],
    *,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
) -> str:
    active_days = int(daily.get("active_days") or 0)
    if active_days < min_active_days:
        return "INSUFFICIENT_ACTIVE_DAYS"
    share = _safe_metric(daily.get("max_daily_sample_share"))
    if share is None or share > max_daily_sample_share:
        return "DAILY_SAMPLE_CONCENTRATED"
    positive_rate = _safe_metric(daily.get("positive_day_rate"))
    if positive_rate is None or positive_rate < min_positive_day_rate:
        return "DAILY_PNL_UNSTABLE"
    return "DAILY_STABLE"


def _daily_stability_payload(daily: dict[str, Any], status: str) -> dict[str, Any]:
    keys = (
        "campaign_timezone",
        "active_days",
        "first_day",
        "last_day",
        "min_daily_samples",
        "max_daily_samples",
        "avg_daily_samples",
        "max_daily_sample_share",
        "positive_days",
        "negative_days",
        "flat_days",
        "positive_day_rate",
        "avg_daily_realized_pips",
        "worst_daily_realized_pips",
        "best_daily_realized_pips",
    )
    payload = {key: daily[key] for key in keys if key in daily}
    payload["daily_stability_status"] = status
    return payload


def _daily_stability_gap(
    daily: dict[str, Any],
    status: str,
    *,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
) -> dict[str, Any]:
    active_days = int(daily.get("active_days") or 0)
    positive_day_rate = _safe_metric(daily.get("positive_day_rate")) or 0.0
    positive_days = int(daily.get("positive_days") or 0)
    required_active_days = max(active_days, int(min_active_days))
    required_positive_days = math.ceil(float(min_positive_day_rate) * required_active_days)
    max_share = _safe_metric(daily.get("max_daily_sample_share"))
    reasons: list[str] = []
    missing_active_days = max(0, int(min_active_days) - active_days)
    missing_positive_days = max(0, required_positive_days - positive_days)
    if missing_active_days:
        reasons.append("NEEDS_MORE_ACTIVE_DAYS")
    if max_share is None:
        reasons.append("NEEDS_DAILY_SAMPLE_DISTRIBUTION")
    elif max_share > float(max_daily_sample_share):
        reasons.append("NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION")
    if positive_day_rate < float(min_positive_day_rate):
        reasons.append("NEEDS_HIGHER_POSITIVE_DAY_RATE")
    return {
        "status": status,
        "reasons": reasons,
        "missing_active_days": missing_active_days,
        "missing_positive_days_at_current_requirement": missing_positive_days,
        "required_active_days": int(min_active_days),
        "required_positive_days_at_current_requirement": required_positive_days,
        "required_positive_day_rate": _round(float(min_positive_day_rate)),
        "current_positive_day_rate": _round(positive_day_rate),
        "max_allowed_daily_sample_share": _round(float(max_daily_sample_share)),
        "current_max_daily_sample_share": _round(max_share),
    }


def _precision_rule_adoption_payload(
    daily: dict[str, Any],
    status: str,
    *,
    min_active_days: int,
    max_daily_sample_share: float,
    min_positive_day_rate: float,
) -> dict[str, Any]:
    gap = _daily_stability_gap(
        daily,
        status,
        min_active_days=min_active_days,
        max_daily_sample_share=max_daily_sample_share,
        min_positive_day_rate=min_positive_day_rate,
    )
    if status == "DAILY_STABLE":
        return {
            "adoption_status": "LIVE_GRADE_DAILY_STABLE",
            "live_grade": True,
            "adoption_blockers": [],
            "daily_stability_gap": gap,
        }
    blockers = [status]
    blockers.extend(reason for reason in gap["reasons"] if reason not in blockers)
    return {
        "adoption_status": "RANK_ONLY_NOT_DAILY_STABLE",
        "live_grade": False,
        "adoption_blockers": blockers,
        "daily_stability_gap": gap,
    }


def _precision_rule_negative_adoption_payload() -> dict[str, Any]:
    return {
        "adoption_status": "LIVE_BLOCK_NEGATIVE_EXPECTANCY",
        "live_grade": False,
        "adoption_blockers": ["NEGATIVE_EXPECTANCY"],
    }


def _precision_adoption_summary(
    *,
    edge_rules: Sequence[dict[str, Any]],
    daily_stable_edge_rules: Sequence[dict[str, Any]],
    contrarian_edge_rules: Sequence[dict[str, Any]],
    daily_stable_contrarian_edge_rules: Sequence[dict[str, Any]],
    negative_rules: Sequence[dict[str, Any]],
    rejected_daily_stability: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    live_grade_count = len(daily_stable_edge_rules) + len(daily_stable_contrarian_edge_rules)
    rank_only_count = (
        len(edge_rules)
        + len(contrarian_edge_rules)
        - live_grade_count
    )
    blockers = collections.Counter(
        blocker
        for row in rejected_daily_stability
        for blocker in row.get("adoption_blockers", [])
    )
    return {
        "live_grade_support_rules": live_grade_count,
        "rank_only_support_rules": rank_only_count,
        "negative_block_rules": len(negative_rules),
        "has_live_grade_support": live_grade_count > 0,
        "has_rank_only_support": rank_only_count > 0,
        "rank_only_blocker_counts": dict(sorted(blockers.items())),
    }


def _bidask_precision_rules(
    segment_rows: Sequence[dict[str, Any]],
    *,
    contrarian_segment_rows: Sequence[dict[str, Any]] | None = None,
    granularity: str,
    audit_report: str,
    edge_min_samples: int,
    edge_min_directional_hit_rate: float,
    edge_min_avg_final_pips: float,
    edge_min_avg_realized_pips: float,
    edge_min_win_rate: float,
    edge_min_profit_factor: float,
    negative_min_samples: int,
    negative_max_directional_hit_rate: float,
    negative_max_avg_final_pips: float,
    negative_max_avg_realized_pips: float,
    negative_max_win_rate: float,
    negative_max_profit_factor: float,
    stable_min_active_days: int = DEFAULT_STABLE_MIN_ACTIVE_DAYS,
    stable_max_daily_sample_share: float = DEFAULT_STABLE_MAX_DAILY_SAMPLE_SHARE,
    stable_min_positive_day_rate: float = DEFAULT_STABLE_MIN_POSITIVE_DAY_RATE,
) -> dict[str, Any]:
    selection = {
        "edge_min_samples": edge_min_samples,
        "edge_min_directional_hit_rate": edge_min_directional_hit_rate,
        "edge_min_avg_final_pips": edge_min_avg_final_pips,
        "edge_min_avg_realized_pips": edge_min_avg_realized_pips,
        "edge_min_win_rate": edge_min_win_rate,
        "edge_min_profit_factor": edge_min_profit_factor,
        "negative_min_samples": negative_min_samples,
        "negative_max_directional_hit_rate": negative_max_directional_hit_rate,
        "negative_max_avg_final_pips": negative_max_avg_final_pips,
        "negative_max_avg_realized_pips": negative_max_avg_realized_pips,
        "negative_max_win_rate": negative_max_win_rate,
        "negative_max_profit_factor": negative_max_profit_factor,
        "stable_min_active_days": stable_min_active_days,
        "stable_max_daily_sample_share": stable_max_daily_sample_share,
        "stable_min_positive_day_rate": stable_min_positive_day_rate,
    }
    edge_rules: list[dict[str, Any]] = []
    daily_stable_edge_rules: list[dict[str, Any]] = []
    negative_rules: list[dict[str, Any]] = []
    contrarian_edge_rules: list[dict[str, Any]] = []
    daily_stable_contrarian_edge_rules: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_contrarian: list[dict[str, Any]] = []
    rejected_daily_stability: list[dict[str, Any]] = []
    for row in segment_rows:
        summary = row.get("summary") or {}
        best = row.get("best_exit") or {}
        pair = str(row.get("pair") or "").upper()
        direction = str(row.get("direction") or "").upper()
        n = int(row.get("n") or 0)
        hit_rate = _safe_metric(summary.get("hit_rate"))
        avg_final = _safe_metric(summary.get("avg_final_pips"))
        avg_mfe = _safe_metric(summary.get("avg_mfe_pips"))
        avg_mae = _safe_metric(summary.get("avg_mae_pips"))
        median_final = _safe_metric(summary.get("median_final_pips"))
        avg_realized = _safe_metric(best.get("avg_realized_pips"))
        win_rate = _safe_metric(best.get("win_rate"))
        profit_factor = _safe_metric(best.get("profit_factor"))
        take_profit = _safe_metric(best.get("take_profit_pips"))
        stop_loss = _safe_metric(best.get("stop_loss_pips"))
        daily = row.get("daily_stability") or {}
        daily_status = _daily_stability_status(
            daily,
            min_active_days=stable_min_active_days,
            max_daily_sample_share=stable_max_daily_sample_share,
            min_positive_day_rate=stable_min_positive_day_rate,
        )
        if not pair or direction not in DIRECTIONAL or take_profit is None or stop_loss is None:
            continue
        common = {
            "pair": pair,
            "side": _side_for_direction(direction),
            "direction": direction,
            "granularity": granularity,
            "samples": n,
            "directional_hit_rate": _round(hit_rate),
            "avg_final_pips": _round(avg_final),
            "median_final_pips": _round(median_final),
            "avg_mfe_pips": _round(avg_mfe),
            "avg_mae_pips": _round(avg_mae),
            "optimized_take_profit_pips": _round(take_profit),
            "optimized_stop_loss_pips": _round(stop_loss),
            "optimized_avg_realized_pips": _round(avg_realized),
            "optimized_win_rate": _round(win_rate),
            "optimized_profit_factor": _round(profit_factor),
            "audit_report": audit_report,
            **_daily_stability_payload(daily, daily_status),
            **_precision_rule_adoption_payload(
                daily,
                daily_status,
                min_active_days=stable_min_active_days,
                max_daily_sample_share=stable_max_daily_sample_share,
                min_positive_day_rate=stable_min_positive_day_rate,
            ),
        }
        if (
            n >= edge_min_samples
            and hit_rate is not None
            and hit_rate >= edge_min_directional_hit_rate
            and avg_final is not None
            and avg_final > edge_min_avg_final_pips
            and avg_realized is not None
            and avg_realized >= edge_min_avg_realized_pips
            and win_rate is not None
            and win_rate >= edge_min_win_rate
            and profit_factor is not None
            and profit_factor >= edge_min_profit_factor
        ):
            min_target, max_target = _target_bounds(take_profit)
            rule = {
                "name": (
                    f"{pair}_{direction}_{granularity}_BIDASK_HARVEST_"
                    f"TP{_rule_number(take_profit)}_SL{_rule_number(stop_loss)}"
                ),
                **common,
                "min_target_pips": min_target,
                "max_target_pips": max_target,
                "max_stop_pips": _round(stop_loss + 0.2),
            }
            edge_rules.append(rule)
            if daily_status == "DAILY_STABLE":
                daily_stable_edge_rules.append(rule)
            else:
                rejected_daily_stability.append(
                    {
                        "name": rule["name"],
                        "pair": pair,
                        "direction": direction,
                        "samples": n,
                        "optimized_avg_realized_pips": _round(avg_realized),
                        "optimized_win_rate": _round(win_rate),
                        "optimized_profit_factor": _round(profit_factor),
                        **_daily_stability_payload(daily, daily_status),
                        **_precision_rule_adoption_payload(
                            daily,
                            daily_status,
                            min_active_days=stable_min_active_days,
                            max_daily_sample_share=stable_max_daily_sample_share,
                            min_positive_day_rate=stable_min_positive_day_rate,
                        ),
                    }
                )
            continue
        if (
            n >= negative_min_samples
            and hit_rate is not None
            and hit_rate <= negative_max_directional_hit_rate
            and avg_final is not None
            and avg_final <= negative_max_avg_final_pips
            and avg_realized is not None
            and avg_realized <= negative_max_avg_realized_pips
            and win_rate is not None
            and win_rate <= negative_max_win_rate
            and profit_factor is not None
            and profit_factor <= negative_max_profit_factor
        ):
            negative_rules.append(
                {
                    "name": f"{pair}_{direction}_{granularity}_BIDASK_NEGATIVE_EXPECTANCY",
                    **common,
                    **_precision_rule_negative_adoption_payload(),
                    "blocks_live_support": True,
                }
            )
            continue
        if n >= min(edge_min_samples, negative_min_samples):
            rejected.append(
                {
                    "pair": pair,
                    "direction": direction,
                    "samples": n,
                    "directional_hit_rate": _round(hit_rate),
                    "avg_final_pips": _round(avg_final),
                    "optimized_avg_realized_pips": _round(avg_realized),
                    "optimized_win_rate": _round(win_rate),
                    "optimized_profit_factor": _round(profit_factor),
                }
            )
    for row in contrarian_segment_rows or ():
        summary = row.get("summary") or {}
        source_summary = row.get("source_summary") or {}
        best = row.get("best_exit") or {}
        pair = str(row.get("pair") or "").upper()
        forecast_direction = str(row.get("forecast_direction") or "").upper()
        direction = str(row.get("direction") or "").upper()
        horizon_bucket = str(row.get("horizon_bucket") or "").strip()
        confidence_bucket = str(row.get("confidence_bucket") or "").strip()
        if _opposite_direction(forecast_direction) != direction:
            continue
        n = int(row.get("n") or 0)
        hit_rate = _safe_metric(summary.get("hit_rate"))
        avg_final = _safe_metric(summary.get("avg_final_pips"))
        avg_mfe = _safe_metric(summary.get("avg_mfe_pips"))
        avg_mae = _safe_metric(summary.get("avg_mae_pips"))
        median_final = _safe_metric(summary.get("median_final_pips"))
        source_hit_rate = _safe_metric(source_summary.get("hit_rate"))
        source_avg_final = _safe_metric(source_summary.get("avg_final_pips"))
        source_avg_mfe = _safe_metric(source_summary.get("avg_mfe_pips"))
        source_avg_mae = _safe_metric(source_summary.get("avg_mae_pips"))
        avg_realized = _safe_metric(best.get("avg_realized_pips"))
        win_rate = _safe_metric(best.get("win_rate"))
        profit_factor = _safe_metric(best.get("profit_factor"))
        take_profit = _safe_metric(best.get("take_profit_pips"))
        stop_loss = _safe_metric(best.get("stop_loss_pips"))
        daily = row.get("daily_stability") or {}
        daily_status = _daily_stability_status(
            daily,
            min_active_days=stable_min_active_days,
            max_daily_sample_share=stable_max_daily_sample_share,
            min_positive_day_rate=stable_min_positive_day_rate,
        )
        if not pair or direction not in DIRECTIONAL or take_profit is None or stop_loss is None:
            continue
        clears = (
            n >= edge_min_samples
            and source_hit_rate is not None
            and source_hit_rate <= negative_max_directional_hit_rate
            and source_avg_final is not None
            and source_avg_final <= negative_max_avg_final_pips
            and hit_rate is not None
            and hit_rate >= edge_min_directional_hit_rate
            and avg_final is not None
            and avg_final > edge_min_avg_final_pips
            and avg_realized is not None
            and avg_realized >= edge_min_avg_realized_pips
            and win_rate is not None
            and win_rate >= edge_min_win_rate
            and profit_factor is not None
            and profit_factor >= edge_min_profit_factor
        )
        if clears:
            min_target, max_target = _target_bounds(take_profit)
            bucket_fields = {
                key: value
                for key, value in (
                    ("horizon_bucket", horizon_bucket),
                    ("confidence_bucket", confidence_bucket),
                )
                if value
            }
            bucket_name = _contrarian_bucket_name(bucket_fields)
            rule = {
                "name": (
                    f"{pair}_{forecast_direction}{bucket_name}_FADE_TO_{direction}_{granularity}_"
                    f"BIDASK_CONTRARIAN_HARVEST_TP{_rule_number(take_profit)}_"
                    f"SL{_rule_number(stop_loss)}"
                ),
                "pair": pair,
                "side": _side_for_direction(direction),
                "direction": direction,
                "forecast_direction": forecast_direction,
                "faded_direction": forecast_direction,
                "contrarian_edge": True,
                **bucket_fields,
                "granularity": granularity,
                "samples": n,
                "source_directional_hit_rate": _round(source_hit_rate),
                "source_avg_final_pips": _round(source_avg_final),
                "source_avg_mfe_pips": _round(source_avg_mfe),
                "source_avg_mae_pips": _round(source_avg_mae),
                "directional_hit_rate": _round(hit_rate),
                "avg_final_pips": _round(avg_final),
                "median_final_pips": _round(median_final),
                "avg_mfe_pips": _round(avg_mfe),
                "avg_mae_pips": _round(avg_mae),
                "optimized_take_profit_pips": _round(take_profit),
                "optimized_stop_loss_pips": _round(stop_loss),
                "optimized_avg_realized_pips": _round(avg_realized),
                "optimized_win_rate": _round(win_rate),
                "optimized_profit_factor": _round(profit_factor),
                "min_target_pips": min_target,
                "max_target_pips": max_target,
                "max_stop_pips": _round(stop_loss + 0.2),
                "audit_report": audit_report,
                **_daily_stability_payload(daily, daily_status),
                **_precision_rule_adoption_payload(
                    daily,
                    daily_status,
                    min_active_days=stable_min_active_days,
                    max_daily_sample_share=stable_max_daily_sample_share,
                    min_positive_day_rate=stable_min_positive_day_rate,
                ),
            }
            contrarian_edge_rules.append(rule)
            if daily_status == "DAILY_STABLE":
                daily_stable_contrarian_edge_rules.append(rule)
            else:
                rejected_daily_stability.append(
                    {
                        "name": rule["name"],
                        "pair": pair,
                        "forecast_direction": forecast_direction,
                        "direction": direction,
                        **bucket_fields,
                        "samples": n,
                        "optimized_avg_realized_pips": _round(avg_realized),
                        "optimized_win_rate": _round(win_rate),
                        "optimized_profit_factor": _round(profit_factor),
                        **_daily_stability_payload(daily, daily_status),
                        **_precision_rule_adoption_payload(
                            daily,
                            daily_status,
                            min_active_days=stable_min_active_days,
                            max_daily_sample_share=stable_max_daily_sample_share,
                            min_positive_day_rate=stable_min_positive_day_rate,
                        ),
                    }
                )
            continue
        if n >= edge_min_samples:
            rejected_contrarian.append(
                {
                    "pair": pair,
                    "forecast_direction": forecast_direction,
                    "direction": direction,
                    **{
                        key: value
                        for key, value in (
                            ("horizon_bucket", horizon_bucket),
                            ("confidence_bucket", confidence_bucket),
                        )
                        if value
                    },
                    "samples": n,
                    "source_directional_hit_rate": _round(source_hit_rate),
                    "source_avg_final_pips": _round(source_avg_final),
                    "directional_hit_rate": _round(hit_rate),
                    "avg_final_pips": _round(avg_final),
                    "optimized_avg_realized_pips": _round(avg_realized),
                    "optimized_win_rate": _round(win_rate),
                    "optimized_profit_factor": _round(profit_factor),
                }
            )
    edge_rules.sort(
        key=lambda item: (
            item["optimized_profit_factor"],
            item["optimized_avg_realized_pips"],
            item["samples"],
        ),
        reverse=True,
    )
    negative_rules.sort(
        key=lambda item: (
            item["samples"],
            -float(item["optimized_avg_realized_pips"] or 0.0),
        ),
        reverse=True,
    )
    contrarian_edge_rules.sort(
        key=lambda item: (
            item["optimized_profit_factor"],
            item["optimized_avg_realized_pips"],
            item["samples"],
            int(bool(item.get("horizon_bucket"))) + int(bool(item.get("confidence_bucket"))),
        ),
        reverse=True,
    )
    daily_stable_edge_rules.sort(
        key=lambda item: (
            item["optimized_profit_factor"],
            item["optimized_avg_realized_pips"],
            item["samples"],
        ),
        reverse=True,
    )
    daily_stable_contrarian_edge_rules.sort(
        key=lambda item: (
            item["optimized_profit_factor"],
            item["optimized_avg_realized_pips"],
            item["samples"],
            int(bool(item.get("horizon_bucket"))) + int(bool(item.get("confidence_bucket"))),
        ),
        reverse=True,
    )
    return {
        "selection": selection,
        "edge_rules": edge_rules,
        "daily_stable_edge_rules": daily_stable_edge_rules,
        "contrarian_edge_rules": contrarian_edge_rules,
        "daily_stable_contrarian_edge_rules": daily_stable_contrarian_edge_rules,
        "negative_rules": negative_rules,
        "adoption_summary": _precision_adoption_summary(
            edge_rules=edge_rules,
            daily_stable_edge_rules=daily_stable_edge_rules,
            contrarian_edge_rules=contrarian_edge_rules,
            daily_stable_contrarian_edge_rules=daily_stable_contrarian_edge_rules,
            negative_rules=negative_rules,
            rejected_daily_stability=rejected_daily_stability,
        ),
        "rejected_sampled_segments": rejected,
        "rejected_contrarian_segments": rejected_contrarian,
        "rejected_daily_stability_segments": rejected_daily_stability,
    }


def _block_unverified_positive_rules(
    precision: dict[str, Any],
    *,
    blockers: Sequence[str],
) -> dict[str, Any]:
    """Keep diagnostic candidates visible without publishing them as live edge.

    Positive rules are optimized on the replay cohort. They cannot enter the
    packageable rule sections until a pre-registered cohort and validation
    contract are verified. Negative rules remain available as conservative
    trade blockers.
    """

    out = dict(precision)
    positive_sections = (
        "edge_rules",
        "daily_stable_edge_rules",
        "contrarian_edge_rules",
        "daily_stable_contrarian_edge_rules",
    )
    for section in positive_sections:
        candidates = []
        for raw_candidate in out.get(section) or []:
            candidate = dict(raw_candidate)
            candidate["live_grade"] = False
            candidate["adoption_status"] = "DIAGNOSTIC_ONLY_NOT_ADOPTABLE"
            candidate["adoption_blockers"] = list(
                dict.fromkeys(
                    [
                        *(candidate.get("adoption_blockers") or []),
                        *blockers,
                    ]
                )
            )
            candidates.append(candidate)
        out[f"diagnostic_{section}"] = candidates
        out[section] = []
    out["positive_rule_adoption_blocked"] = True
    out["positive_rule_adoption_blockers"] = list(blockers)
    out["adoption_summary"] = _precision_adoption_summary(
        edge_rules=[],
        daily_stable_edge_rules=[],
        contrarian_edge_rules=[],
        daily_stable_contrarian_edge_rules=[],
        negative_rules=list(out.get("negative_rules") or []),
        rejected_daily_stability=list(out.get("rejected_daily_stability_segments") or []),
    )
    return out


def _contrarian_bucket_name(fields: dict[str, str]) -> str:
    parts: list[str] = []
    horizon = fields.get("horizon_bucket")
    if horizon:
        parts.append(f"H{_rule_text_token(horizon)}")
    confidence = fields.get("confidence_bucket")
    if confidence:
        parts.append(f"C{_rule_text_token(confidence)}")
    return "" if not parts else "_" + "_".join(parts)


def _rule_text_token(value: str) -> str:
    return (
        str(value)
        .replace(">=", "GE")
        .replace("<=", "LE")
        .replace(">", "GT")
        .replace("<", "LT")
        .replace(".", "p")
        .replace("-", "_")
        .replace(" ", "")
    )


def _source_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    sourced = [row for row in rows if "source_final_pips" in row]
    if not sourced:
        return None
    return {
        "n": len(sourced),
        "hit_rate": _rate(row.get("source_direction_hit") for row in sourced),
        "avg_final_pips": _mean(
            float(row["source_final_pips"])
            for row in sourced
            if row.get("source_final_pips") is not None
        ),
        "median_final_pips": _median(
            float(row["source_final_pips"])
            for row in sourced
            if row.get("source_final_pips") is not None
        ),
        "avg_mfe_pips": _mean(
            float(row["source_mfe_pips"])
            for row in sourced
            if row.get("source_mfe_pips") is not None
        ),
        "avg_mae_pips": _mean(
            float(row["source_mae_pips"])
            for row in sourced
            if row.get("source_mae_pips") is not None
        ),
    }


def _target_bounds(take_profit_pips: float) -> tuple[float, float]:
    return (
        _round(max(0.1, float(take_profit_pips) - 0.2)),
        _round(float(take_profit_pips) + max(0.5, float(take_profit_pips) * 0.1)),
    )


def _side_for_direction(direction: str) -> str:
    return "LONG" if direction == "UP" else "SHORT"


def _rule_number(value: float) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric).replace(".", "p")


def _safe_metric(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    split = report["train_validation_exit_selection"]
    selection = report.get("selection_contract") or {}
    experiment = report.get("experiment") or {}
    lines = [
        "# OANDA History Replay Validate",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        f"- source: {report['source']}",
        f"- history_dirs: {', '.join(report['history_dirs'])}",
        f"- truth_source: {report['truth_source']}",
        f"- rows: raw_directional={report['raw_directional_rows']} deduped={report['deduped_directional_rows']} evaluated={report['evaluated_rows']}",
        f"- confidence_missing: calibrated_loaded={report.get('calibrated_confidence_missing_rows', 0)} evaluated={report.get('evaluated_confidence_missing_rows', 0)} raw_filter={report.get('raw_confidence_missing_rows', 0)}",
        f"- technical_context: missing={report.get('technical_context_missing_rows', 0)} invalid={report.get('technical_context_invalid_rows', 0)} incomplete={report.get('technical_context_incomplete_rows', 0)}",
        f"- history: files={report['history_files']} candles={report['history_candles']} skipped={report['history_skipped_rows']}",
        f"- price_truth_coverage: status={(report.get('price_truth_coverage') or {}).get('status')} adoption={(report.get('price_truth_coverage') or {}).get('adoption_level')}",
        f"- audit_mode: {selection.get('audit_mode')}",
        f"- proof_eligible: {selection.get('proof_eligible')}",
        f"- proof_blockers: {', '.join(selection.get('proof_blockers') or []) or 'none'}",
        f"- confidence_policy: field={selection.get('confidence_field')} min={selection.get('min_confidence')}",
        f"- independent_non_overlap_per_pair: {selection.get('independent_non_overlap_per_pair')}",
        f"- experiment_id: {experiment.get('experiment_id')}",
        f"- segment_min_group_samples: {(report.get('segment_contract') or {}).get('min_group_samples')}",
        "",
        "## Summary",
        "",
        _summary_line(summary),
        "",
    ]
    truth = report.get("price_truth_coverage") or {}
    lines.extend(
        [
            "## Price Truth Coverage",
            "",
            f"- status: {truth.get('status')} ({truth.get('reason')})",
            f"- adoption_level: {truth.get('adoption_level')}",
            f"- candidate_rule_validation_blocked: {truth.get('candidate_rule_validation_blocked')}",
            f"- global_currency_validation_blocked: {truth.get('global_currency_validation_blocked')}",
            f"- evaluated_rows: {truth.get('evaluated_rows')} / required_min={truth.get('required_min_evaluated_rows')}",
            f"- unscorable_no_market_rows: {truth.get('unscorable_no_market_rows')}",
            f"- missing_price_truth_samples: {truth.get('missing_price_truth_samples')}",
            f"- missing_price_window_groups: {truth.get('missing_price_window_group_count')}",
            f"- incomplete_price_window_groups: {truth.get('incomplete_price_window_group_count')}",
            f"- fetchable_price_window_groups: {truth.get('fetchable_price_window_group_count')}",
            f"- unscorable_no_market_window_groups: {truth.get('unscorable_no_market_window_group_count')}",
            f"- future_price_truth_window_groups: {truth.get('future_price_truth_window_group_count')}",
            f"- missing_pairs: {', '.join(truth.get('missing_pairs') or []) or 'none'}",
            f"- all_currency_sample_coverage_status: {truth.get('all_currency_sample_coverage_status')}",
            f"- under_sampled_pair_direction_count: {truth.get('under_sampled_pair_direction_count')}",
            f"- under_sampled_missing_evaluated_samples: {truth.get('under_sampled_missing_evaluated_samples')}",
            f"- history_fetch_command_mode: {truth.get('history_fetch_command_mode')}",
            f"- history_fetch_command_count: {truth.get('history_fetch_command_count')}",
        ]
    )
    fetch_commands = truth.get("history_fetch_commands") or []
    for item in fetch_commands[:12]:
        lines.append(
            f"  - {item.get('date')}: missing_rows={item.get('forecast_rows_missing_truth')} "
            f"pairs={len(item.get('pairs') or [])} "
            f"from={item.get('from_utc')} to={item.get('to_utc')} "
            f"clamped_to_now={item.get('clamped_to_now')}"
        )
        if item.get("command"):
            lines.append(f"    `{item.get('command')}`")
    fetch_command = truth.get("history_fetch_command")
    if fetch_command:
        lines.append(f"- broad_history_fetch_command_fallback: `{fetch_command}`")
    blockers = truth.get("blockers") or []
    lines.append(f"- blockers: {', '.join(blockers) if blockers else 'none'}")
    warnings = truth.get("warnings") or []
    lines.append(f"- warnings: {', '.join(warnings) if warnings else 'none'}")
    lines.append("")
    coverage = report.get("forecast_sample_coverage") or {}
    under_sampled = coverage.get("under_sampled_pair_directions") or []
    lines.extend(
        [
            "## Forecast Sample Coverage",
            "",
            f"- pair_count: {coverage.get('pair_count')}",
            f"- pair_direction_count: {coverage.get('pair_direction_count')}",
            f"- under_sampled_pair_directions: {len(under_sampled)}",
        ]
    )
    for item in under_sampled[:12]:
        lines.append(
            f"  - {item['pair']} {item['direction']}: "
            f"forecast={item['forecast_samples']} evaluated={item['evaluated_samples']} "
            f"days={item['evaluated_active_days']} "
            f"no_market={item.get('unscorable_no_market_samples', 0)} "
            f"missing_truth={item['missing_price_truth_samples']} "
            f"missing_samples={item['missing_evaluated_samples']} "
            f"missing_days={item['missing_active_days']} "
            f"gap={','.join(item.get('coverage_gap_reasons') or [])}"
        )
    lines.extend(["", "## Train/Validation Exit Selection", ""])
    if split.get("status") != "OK":
        lines.append(f"- status: {split.get('status')} n={split.get('n')} min_required={split.get('min_required')}")
    else:
        selected = split["selected_by_train"]
        validation = split["validation"]
        lines.append(
            "- selected_by_train: "
            f"TP={selected['take_profit_pips']} SL={selected['stop_loss_pips']} "
            f"train_avg={_fmt(selected['avg_realized_pips'])} win={_pct(selected['win_rate'])} PF={_fmt(selected['profit_factor'])}"
        )
        lines.append(
            "- validation: "
            f"n={validation['n']} avg={_fmt(validation['avg_realized_pips'])} "
            f"win={_pct(validation['win_rate'])} PF={_fmt(validation['profit_factor'])} timeout={_pct(validation['timeout_rate'])}"
        )
    lines.extend(["", "## Top Exit Grid", ""])
    for item in report["exit_grid"][:10]:
        lines.append(
            f"- TP={item['take_profit_pips']} SL={item['stop_loss_pips']} "
            f"n={item['n']} avg={_fmt(item['avg_realized_pips'])} "
            f"win={_pct(item['win_rate'])} PF={_fmt(item['profit_factor'])} timeout={_pct(item['timeout_rate'])}"
        )
    precision = report.get("precision_rules") or {}
    lines.extend(["", "## Precision Rule Candidates", ""])
    edge_rules = precision.get("edge_rules") or []
    contrarian_rules = precision.get("contrarian_edge_rules") or []
    daily_stable_edge_rules = precision.get("daily_stable_edge_rules") or []
    daily_stable_contrarian_rules = precision.get("daily_stable_contrarian_edge_rules") or []
    negative_rules = precision.get("negative_rules") or []
    adoption = precision.get("adoption_summary") or {}
    lines.append(f"- edge_rules: {len(edge_rules)}")
    lines.append(f"- daily_stable_edge_rules: {len(daily_stable_edge_rules)}")
    lines.append(
        "- adoption_summary: "
        f"live_grade={adoption.get('live_grade_support_rules', 0)} "
        f"rank_only={adoption.get('rank_only_support_rules', 0)} "
        f"negative_blocks={adoption.get('negative_block_rules', 0)} "
        f"has_live_grade={adoption.get('has_live_grade_support', False)}"
    )
    lines.append(
        "- positive_rule_adoption_blocked: "
        f"{precision.get('positive_rule_adoption_blocked', False)} "
        f"blockers={','.join(precision.get('positive_rule_adoption_blockers') or []) or 'none'}"
    )
    lines.append(
        "- diagnostic_positive_candidates: "
        f"edge={len(precision.get('diagnostic_edge_rules') or [])} "
        f"contrarian={len(precision.get('diagnostic_contrarian_edge_rules') or [])}"
    )
    for rule in edge_rules[:12]:
        lines.append(
            f"  - {rule['name']}: n={rule['samples']} hit={_pct(rule.get('directional_hit_rate'))} "
            f"avg_final={_fmt(rule.get('avg_final_pips'))} "
            f"TP={rule.get('optimized_take_profit_pips')} SL={rule.get('optimized_stop_loss_pips')} "
            f"realized={_fmt(rule.get('optimized_avg_realized_pips'))} "
            f"win={_pct(rule.get('optimized_win_rate'))} PF={_fmt(rule.get('optimized_profit_factor'))} "
            f"adoption={rule.get('adoption_status')} "
            f"blockers={','.join(rule.get('adoption_blockers') or []) or 'none'} "
            f"daily={rule.get('daily_stability_status')} days={rule.get('active_days')} "
            f"max_share={_pct(rule.get('max_daily_sample_share'))}"
        )
    lines.append(f"- contrarian_edge_rules: {len(contrarian_rules)}")
    lines.append(f"- daily_stable_contrarian_edge_rules: {len(daily_stable_contrarian_rules)}")
    for rule in contrarian_rules[:12]:
        lines.append(
            f"  - {rule['name']}: fade={rule.get('faded_direction')} trade={rule.get('direction')} "
            f"n={rule['samples']} source_hit={_pct(rule.get('source_directional_hit_rate'))} "
            f"hit={_pct(rule.get('directional_hit_rate'))} "
            f"avg_final={_fmt(rule.get('avg_final_pips'))} "
            f"TP={rule.get('optimized_take_profit_pips')} SL={rule.get('optimized_stop_loss_pips')} "
            f"realized={_fmt(rule.get('optimized_avg_realized_pips'))} "
            f"win={_pct(rule.get('optimized_win_rate'))} PF={_fmt(rule.get('optimized_profit_factor'))} "
            f"adoption={rule.get('adoption_status')} "
            f"blockers={','.join(rule.get('adoption_blockers') or []) or 'none'} "
            f"daily={rule.get('daily_stability_status')} days={rule.get('active_days')} "
            f"max_share={_pct(rule.get('max_daily_sample_share'))}"
        )
    lines.append(f"- negative_rules: {len(negative_rules)}")
    for rule in negative_rules[:12]:
        lines.append(
            f"  - {rule['name']}: n={rule['samples']} hit={_pct(rule.get('directional_hit_rate'))} "
            f"avg_final={_fmt(rule.get('avg_final_pips'))} "
            f"best_realized={_fmt(rule.get('optimized_avg_realized_pips'))} "
            f"win={_pct(rule.get('optimized_win_rate'))} PF={_fmt(rule.get('optimized_profit_factor'))} "
            f"adoption={rule.get('adoption_status')}"
        )
    rejected_daily = precision.get("rejected_daily_stability_segments") or []
    lines.append(f"- rejected_daily_stability_segments: {len(rejected_daily)}")
    for row in rejected_daily[:12]:
        gap = row.get("daily_stability_gap") or {}
        lines.append(
            f"  - {row.get('name')}: adoption={row.get('adoption_status')} "
            f"blockers={','.join(row.get('adoption_blockers') or []) or 'none'} "
            f"days={row.get('active_days')} missing_days={gap.get('missing_active_days')} "
            f"positive_day_rate={_pct(row.get('positive_day_rate'))} "
            f"max_share={_pct(row.get('max_daily_sample_share'))} "
            f"PF={_fmt(row.get('optimized_profit_factor'))}"
        )
    lines.extend(["", "## Missing Price Windows", ""])
    missing_groups = report.get("missing_price_window_groups") or []
    if not missing_groups:
        lines.append("- none")
    for group in missing_groups[:12]:
        pairs = ",".join(group.get("pairs") or [])
        lines.append(
            f"- {group.get('date')}: count={group.get('count')} "
            f"from={group.get('needed_from_utc')} to={group.get('needed_to_utc')} "
            f"pairs={pairs}"
        )
    lines.extend(["", "## Segment Exit Grids", ""])
    for name, rows in report.get("segment_exit_grids", {}).items():
        lines.append(f"### {name}")
        if not rows:
            lines.append("- no segment cleared sample floor")
        for row in rows[:12]:
            best = row.get("best_exit") or {}
            keys = ", ".join(str(row.get(k)) for k in row if k not in {"summary", "best_exit", "exit_grid", "n"})
            lines.append(
                f"- {keys} n={row.get('n')}: "
                f"best TP={best.get('take_profit_pips')} SL={best.get('stop_loss_pips')} "
                f"avg={_fmt(best.get('avg_realized_pips'))} win={_pct(best.get('win_rate'))} "
                f"PF={_fmt(best.get('profit_factor'))} timeout={_pct(best.get('timeout_rate'))}; "
                f"daily={((row.get('daily_stability') or {}).get('active_days'))}d "
                f"max_share={_pct((row.get('daily_stability') or {}).get('max_daily_sample_share'))}; "
                f"{_summary_line(row.get('summary') or {})}"
            )
        lines.append("")
    lines.extend(["", "## Segments", ""])
    for name, rows in report["segments"].items():
        lines.append(f"### {name}")
        if not rows:
            lines.append("- no segment cleared sample floor")
        for row in rows[:12]:
            keys = ", ".join(f"{k}={row[k]}" for k in row if k not in _SUMMARY_KEYS)
            lines.append(f"- {keys}: {_summary_line(row)}")
        lines.append("")
    return "\n".join(lines)


_SUMMARY_KEYS = {
    "n",
    "hit_rate",
    "hit_wilson95_lower",
    "hit_wilson95_upper",
    "avg_final_pips",
    "total_final_pips",
    "median_final_pips",
    "avg_mfe_pips",
    "avg_mae_pips",
    "median_entry_delay_seconds",
    "median_effective_holding_min",
    "median_unobserved_horizon_tail_seconds",
    "target_touch_rate",
    "invalidation_touch_rate",
    "target_before_invalidation_rate",
    "reward_side_target_rate",
    "adverse_side_invalidation_rate",
    "geometry_outcomes",
    "realized_r_n",
    "avg_realized_r",
    "total_realized_r",
    "positive_realized_r_rate",
    "avg_reward_pips",
    "avg_risk_pips",
}


def _summary_line(summary: dict[str, Any]) -> str:
    return (
        f"n={summary.get('n', 0)} hit={_pct(summary.get('hit_rate'))} "
        f"wilson95=[{_pct(summary.get('hit_wilson95_lower'))},{_pct(summary.get('hit_wilson95_upper'))}] "
        f"avg_final={_fmt(summary.get('avg_final_pips'))} "
        f"median_final={_fmt(summary.get('median_final_pips'))} "
        f"avg_mfe={_fmt(summary.get('avg_mfe_pips'))} avg_mae={_fmt(summary.get('avg_mae_pips'))} "
        f"entry_delay_s={_fmt(summary.get('median_entry_delay_seconds'))} "
        f"effective_hold_m={_fmt(summary.get('median_effective_holding_min'))} "
        f"horizon_tail_s={_fmt(summary.get('median_unobserved_horizon_tail_seconds'))} "
        f"reward_target={_pct(summary.get('reward_side_target_rate'))} "
        f"adverse_inv={_pct(summary.get('adverse_side_invalidation_rate'))} "
        f"target={_pct(summary.get('target_touch_rate'))} invalidation={_pct(summary.get('invalidation_touch_rate'))} "
        f"target_first={_pct(summary.get('target_before_invalidation_rate'))} "
        f"avg_R={_fmt(summary.get('avg_realized_r'))} total_R={_fmt(summary.get('total_realized_r'))}"
    )


def _parse_required_time(value: object, *, flag: str) -> datetime:
    parsed = _parse_time(value)
    if parsed is None:
        raise ValueError(f"{flag} must be a valid ISO-8601 timestamp")
    return parsed


def _parse_time(value: object) -> datetime | None:
    text = str(value or "")
    if not text:
        return None
    if text.endswith("Z"):
        core = text[:-1]
        if "." in core:
            head, frac = core.split(".", 1)
            text = f"{head}.{frac[:6]}+00:00"
        else:
            text = f"{core}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_horizon(value: object) -> float:
    parsed = _safe_float(value)
    if parsed is None or parsed <= 0:
        return 60.0
    return parsed


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "missing"
    if value < 0.50:
        return "<0.50"
    if value < 0.65:
        return "0.50-0.65"
    if value < 0.75:
        return "0.65-0.75"
    if value < 0.90:
        return "0.75-0.90"
    return ">=0.90"


def _score_margin(
    direction: str,
    up_score: float | None,
    down_score: float | None,
    range_score: float | None,
) -> float | None:
    if up_score is None or down_score is None or range_score is None:
        return None
    normalized_direction = str(direction or "").upper()
    if normalized_direction == "UP":
        return float(up_score) - max(float(down_score), float(range_score))
    if normalized_direction == "DOWN":
        return float(down_score) - max(float(up_score), float(range_score))
    return None


def _score_margin_bucket(value: float | None) -> str:
    # Fixed audit bins make repeated reports comparable; they are descriptive
    # cohort labels, not forecast or live-entry thresholds.
    if value is None:
        return "missing"
    if value < 0.0:
        return "<0"
    if value < 5.0:
        return "0-5"
    if value < 10.0:
        return "5-10"
    if value < 20.0:
        return "10-20"
    return ">=20"


def _range_competition(
    up_score: float | None,
    down_score: float | None,
    range_score: float | None,
) -> str:
    if up_score is None or down_score is None or range_score is None:
        return "missing"
    directional_leader = max(float(up_score), float(down_score))
    directional_margin = abs(float(up_score) - float(down_score))
    if float(range_score) >= directional_leader:
        return "RANGE_LEADS_OR_TIES"
    if float(range_score) >= directional_margin:
        return "RANGE_COMPETES_WITH_DIRECTIONAL_MARGIN"
    return "DIRECTIONAL_MARGIN_DOMINATES"


def _utc_session_bucket(value: datetime) -> str:
    # Use fixed UTC bands so the same forecast is never relabelled by DST.
    # These are diagnostic time cohorts, not execution-session permissions.
    hour = value.astimezone(timezone.utc).hour
    if hour < 8:
        return "UTC_00_08"
    if hour < 13:
        return "UTC_08_13"
    if hour < 17:
        return "UTC_13_17"
    if hour < 22:
        return "UTC_17_22"
    return "UTC_22_24"


def _horizon_bucket(value: float) -> str:
    if value <= 15:
        return "<=15m"
    if value <= 30:
        return "16-30m"
    if value <= 60:
        return "31-60m"
    if value <= 240:
        return "61-240m"
    return ">240m"


def _rate(values: Iterable[object]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return sum(1 for value in vals if bool(value)) / len(vals)


def _wilson_interval(hits: int, total: int, *, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = hits / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _proof_eligibility(
    *,
    args: argparse.Namespace,
    forecast_from: datetime | None,
    forecast_to: datetime | None,
    load_stats: dict[str, Any],
    candle_stats: dict[str, Any],
    score_stats: dict[str, Any],
    evaluated_rows: int,
    split: dict[str, Any],
    cohort_lock_verified: bool = False,
) -> dict[str, Any]:
    """Return conservative blockers for calling a cohort proof-grade.

    CLI mode names are declarations, not evidence that a holdout was locked
    before outcomes were inspected. Until a durable pre-evaluation cohort lock
    is supplied and verified, historical runs remain diagnostic evidence.
    """

    blockers: list[str] = []
    if args.audit_mode not in {"LOCKED_HOLDOUT", "FORWARD"}:
        blockers.append("AUDIT_MODE_IS_DIAGNOSTIC")
    if forecast_from is None or forecast_to is None:
        blockers.append("EXPLICIT_COHORT_WINDOW_REQUIRED")
    if not args.independent_non_overlap:
        blockers.append("SAME_PAIR_NON_OVERLAP_REQUIRED")
    if args.min_confidence is None:
        blockers.append("CONFIDENCE_POLICY_MUST_BE_PREDECLARED")
    if not cohort_lock_verified:
        blockers.append("PRE_EVALUATION_COHORT_LOCK_NOT_VERIFIED")
    # Global train/validation output cannot validate a pair/side rule that was
    # optimized on the full cohort. Keep proof closed until each candidate
    # segment has an untouched holdout result of its own.
    blockers.append("SEGMENT_HOLDOUT_RULE_VALIDATION_NOT_IMPLEMENTED")
    if int(load_stats.get("skipped_conflicting_forecast_rows") or 0) > 0:
        blockers.append("CONFLICTING_FORECAST_ROWS")
    if int(load_stats.get("technical_context_missing_rows") or 0) > 0:
        blockers.append("TECHNICAL_CONTEXT_MISSING")
    if int(load_stats.get("technical_context_invalid_rows") or 0) > 0:
        blockers.append("TECHNICAL_CONTEXT_INVALID")
    if int(load_stats.get("technical_context_incomplete_rows") or 0) > 0:
        blockers.append("TECHNICAL_CONTEXT_INCOMPLETE")
    if int(candle_stats.get("history_conflicting_candles") or 0) > 0:
        blockers.append("CONFLICTING_PRICE_TRUTH")
    if int(score_stats.get("skipped_no_pair_candles") or 0) > 0:
        blockers.append("MISSING_PAIR_PRICE_TRUTH")
    if int(score_stats.get("skipped_no_price_window") or 0) > 0:
        blockers.append("MISSING_PRICE_WINDOW")
    if int(score_stats.get("skipped_incomplete_truth_window_rows") or 0) > 0:
        blockers.append("INCOMPLETE_PRICE_WINDOW")
    if int(score_stats.get("pending_future_truth_rows") or 0) > 0:
        blockers.append("PENDING_FUTURE_TRUTH")
    if evaluated_rows < int(args.edge_min_samples):
        blockers.append("MINIMUM_SAMPLE_FLOOR_NOT_MET")
    if str(split.get("status") or "").upper() != "OK":
        blockers.append("TRAIN_VALIDATION_SPLIT_NOT_AVAILABLE")
    else:
        validation = split.get("validation") if isinstance(split.get("validation"), dict) else {}
        validation_avg = _safe_metric((validation or {}).get("avg_realized_pips"))
        validation_pf = _safe_metric((validation or {}).get("profit_factor"))
        if validation_avg is None or validation_avg <= 0.0:
            blockers.append("VALIDATION_EXPECTANCY_NOT_POSITIVE")
        if validation_pf is None or validation_pf <= 1.0:
            blockers.append("VALIDATION_PROFIT_FACTOR_NOT_ABOVE_ONE")
    return {"eligible": not blockers, "blockers": blockers}


def _experiment_identity(
    *,
    args: argparse.Namespace,
    rows: Sequence[ForecastRow],
    candles_by_pair: dict[str, list[QuoteCandle]],
    history_dirs: Sequence[Path],
    forecast_from: datetime | None,
    forecast_to: datetime | None,
    load_stats: dict[str, Any] | None = None,
    candle_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantics = {
        "version": "oanda-bidask-independent-v3",
        "entry": "first complete candle at_or_after forecast; UP ask open; DOWN bid open",
        "exit": "last fully closed candle inside signal horizon; UP bid close; DOWN ask close",
        "truth_completeness": "leading, internal, and trailing candle intervals required",
        "same_bar_tp_sl": "stop_first",
        "non_overlap": "per_pair timestamp >= previous timestamp+horizon",
        "filter_order": "canonical_dedupe,policy_filters,confidence,non_overlap,truth,score",
    }
    forecast_digest = _forecast_rows_digest(rows)
    truth_digest = _truth_candles_digest(candles_by_pair)
    evaluator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract = {
        "schema_version": 1,
        "audit_mode": args.audit_mode,
        "semantics": semantics,
        "evaluator_sha256": evaluator_digest,
        "granularity": str(args.granularity).upper(),
        "forecast_from_utc_inclusive": _iso(forecast_from) if forecast_from is not None else None,
        "forecast_to_utc_exclusive": _iso(forecast_to) if forecast_to is not None else None,
        "pairs": sorted(_parse_pair_filter(args.pairs)),
        "confidence_field": args.confidence_field,
        "min_confidence": args.min_confidence,
        "independent_non_overlap": bool(args.independent_non_overlap),
        "tp_grid_pips": [float(item) for item in args.tp_grid_pips],
        "sl_grid_pips": [float(item) for item in args.sl_grid_pips],
        "train_fraction": float(args.train_fraction),
        "min_train_samples": int(args.min_train_samples),
        "min_validation_samples": int(args.min_validation_samples),
        "min_group_samples": int(args.min_group_samples),
        "edge_min_samples": int(args.edge_min_samples),
        "edge_min_directional_hit_rate": float(args.edge_min_directional_hit_rate),
        "edge_min_avg_final_pips": float(args.edge_min_avg_final_pips),
        "edge_min_avg_realized_pips": float(args.edge_min_avg_realized_pips),
        "edge_min_win_rate": float(args.edge_min_win_rate),
        "edge_min_profit_factor": float(args.edge_min_profit_factor),
        "negative_min_samples": int(args.negative_min_samples),
        "negative_max_directional_hit_rate": float(args.negative_max_directional_hit_rate),
        "negative_max_avg_final_pips": float(args.negative_max_avg_final_pips),
        "negative_max_avg_realized_pips": float(args.negative_max_avg_realized_pips),
        "negative_max_win_rate": float(args.negative_max_win_rate),
        "negative_max_profit_factor": float(args.negative_max_profit_factor),
        "stable_min_active_days": int(args.stable_min_active_days),
        "stable_max_daily_sample_share": float(args.stable_max_daily_sample_share),
        "stable_min_positive_day_rate": float(args.stable_min_positive_day_rate),
        "auto_history_min_days": float(args.auto_history_min_days),
        "cohort_input_diagnostics": {
            "calibrated_confidence_missing_rows": int(
                (load_stats or {}).get("calibrated_confidence_missing_rows") or 0
            ),
            "raw_confidence_missing_rows": int(
                (load_stats or {}).get("raw_confidence_missing_rows") or 0
            ),
            "technical_context_missing_rows": int(
                (load_stats or {}).get("technical_context_missing_rows") or 0
            ),
            "technical_context_invalid_rows": int(
                (load_stats or {}).get("technical_context_invalid_rows") or 0
            ),
            "technical_context_incomplete_rows": int(
                (load_stats or {}).get("technical_context_incomplete_rows") or 0
            ),
            "conflicting_forecast_groups": int(
                (load_stats or {}).get("skipped_conflicting_forecast_rows") or 0
            ),
            "duplicate_truth_candles": int(
                (candle_stats or {}).get("history_duplicate_candles") or 0
            ),
            "conflicting_truth_candles": int(
                (candle_stats or {}).get("history_conflicting_candles") or 0
            ),
        },
        "forecast_rows_sha256": forecast_digest,
        "truth_candles_sha256": truth_digest,
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    experiment_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        **contract,
        "experiment_id": experiment_id,
        "forecast_rows": len(rows),
        "truth_candles": sum(len(items) for items in candles_by_pair.values()),
        "history_sources": len(history_dirs),
    }


def _forecast_rows_digest(rows: Sequence[ForecastRow]) -> str:
    digest = hashlib.sha256()
    payloads: list[str] = []
    for row in rows:
        payload = (
            row.timestamp_utc.astimezone(timezone.utc).isoformat(),
            row.pair,
            row.direction,
            row.confidence,
            row.raw_confidence,
            row.calibration_multiplier,
            row.up_score,
            row.down_score,
            row.range_score,
            row.current_price,
            row.target_price,
            row.invalidation_price,
            row.horizon_min,
            row.cycle_id,
            row.driver_families,
            row.drivers_against_families,
            _technical_context_digest(row.technical_context_v1),
            row.technical_context_status,
        )
        payloads.append(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    for payload in sorted(payloads):
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _truth_candles_digest(candles_by_pair: dict[str, list[QuoteCandle]]) -> str:
    digest = hashlib.sha256()
    for pair in sorted(candles_by_pair):
        for candle in sorted(candles_by_pair[pair], key=lambda item: item.timestamp_utc):
            payload = (
                pair,
                candle.timestamp_utc.astimezone(timezone.utc).isoformat(),
                candle.bid.o,
                candle.bid.h,
                candle.bid.l,
                candle.bid.c,
                candle.ask.o,
                candle.ask.h,
                candle.ask.l,
                candle.ask.c,
            )
            digest.update(json.dumps(payload, separators=(",", ":")).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _experiment_is_complete(
    report_json: Path,
    report_md: Path,
    manifest_path: Path,
    *,
    experiment_id: str,
) -> bool:
    if not report_json.is_file() or not report_md.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report_payload = json.loads(report_json.read_text(encoding="utf-8"))
        report_md_text = report_md.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return False
    report_experiment = report_payload.get("experiment") if isinstance(report_payload, dict) else None
    if not isinstance(report_experiment, dict) or not report_md_text.strip():
        return False
    if manifest.get("experiment_id") != experiment_id or manifest.get("status") != "COMPLETE":
        return False
    if report_experiment.get("experiment_id") != experiment_id:
        return False
    if report_experiment.get("status") != "COMPLETE":
        return False
    return (
        manifest.get("report_json_sha256") == hashlib.sha256(report_json.read_bytes()).hexdigest()
        and manifest.get("report_md_sha256") == hashlib.sha256(report_md.read_bytes()).hexdigest()
    )


def _acquire_experiment_lock(experiment_dir: Path):
    experiment_dir.mkdir(parents=True, exist_ok=True)
    handle = (experiment_dir / ".evaluation.lock").open(mode="a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"experiment evaluation already in progress: {experiment_dir}") from exc
    return handle


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, mode="w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values]
    if not vals:
        return None
    return statistics.mean(vals)


def _median(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values]
    if not vals:
        return None
    return statistics.median(vals)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isinf(numeric):
        return "inf"
    return f"{numeric:.2f}"


def _pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
