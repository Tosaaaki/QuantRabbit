#!/usr/bin/env python3
"""Audit passive RANGE-forecast LIMIT vehicles on local OANDA bid/ask truth.

This validator is intentionally read-only.  It evaluates only information that
existed when a RANGE forecast was emitted: the forecast box, current price,
confidence, and horizon.  A single nearest rail is selected, forecasts for the
same pair are spaced so overlapping persistence is not counted as independent
evidence, and all fills/exits use executable bid/ask sides.

The output is hypothesis evidence, never live permission.  Runtime adoption
still requires the predictive-SCOUT gateway, loss memory, broker freshness,
spread, margin, event, manual-position, and forward-proof gates.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
for path in (SRC_ROOT, SCRIPT_ROOT):
    path_text = str(path)
    sys.path[:] = [item for item in sys.path if item != path_text]
    sys.path.insert(0, path_text)

from oanda_history_replay_validate import (  # noqa: E402
    QuoteCandle,
    _candle_from_payload,
    _history_dirs,
    _is_likely_fx_no_market_window,
    _load_candles,
    _open_history_text,
    _parse_float_csv,
    _parse_pair_filter,
    _parse_time,
    _safe_float,
)
from oanda_range_scout_truth_fetch import (  # noqa: E402
    OANDA_MAX_CANDLES_PER_REQUEST,
    PRODUCTION_OANDA_BASE_URL,
    RANGE_FETCH_SUMMARY_SCHEMA,
    RANGE_FETCH_TASK_SCHEMA,
    RANGE_TRUTH_RECEIPT_FILE,
    _iter_time_chunks,
    _validate_range_truth_receipt_chain,
    expected_dependency_records,
    task_manifest_sha256,
)
from quant_rabbit.instruments import instrument_pip_factor  # noqa: E402


DEFAULT_TP_GRID_PIPS = (3.0, 5.0, 7.0, 10.0)
DEFAULT_SL_GRID_PIPS = (5.0, 7.0, 10.0, 15.0)
DEFAULT_TTL_MINUTES = 90
DEFAULT_DEDUPE_MINUTES = 90
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_MIN_TRAIN_FILLS = 20
DEFAULT_MIN_VALIDATION_FILLS = 10
DEFAULT_MIN_TOTAL_FILLS = 30
MIN_TRUTH_WINDOW_COVERAGE_RATE = 0.95
JST = timezone(timedelta(hours=9), "JST")
S5_CANDLE_INTERVAL = timedelta(seconds=5)
_FETCH_SUMMARY_KEYS = {
    "schema_version",
    "generated_at_utc",
    "window",
    "pairs",
    "granularities",
    "price",
    "max_candles_per_request",
    "output_root",
    "output_dir",
    "tasks",
    "errors",
    "total_rows",
    "total_requests",
    "dry_run",
    "include_incomplete",
}
_FETCH_TASK_KEYS = {
    "schema_version",
    "pair",
    "granularity",
    "price",
    "from",
    "to",
    "path",
    "partial_path",
    "published",
    "windows",
    "requests",
    "rows",
    "errors",
    "dry_run",
    "compressed",
    "max_candles_per_request",
    "include_incomplete",
    "range_truth_acquisition_receipt_sha256",
}
_FETCHED_S5_BA_ROW_KEYS = {
    "time",
    "pair",
    "granularity",
    "price",
    "complete",
    "volume",
    "bid",
    "ask",
}
# One-sided 95% Student-t critical values for df=1..30.  These are
# distributional confidence constants, not market-tuned thresholds.  Above 30
# df the standard Cornish-Fisher expansion converges toward the normal 95%
# quantile without adding a SciPy runtime dependency.
ONE_SIDED_T95_BY_DF = (
    math.inf,
    6.3138,
    2.9200,
    2.3534,
    2.1318,
    2.0150,
    1.9432,
    1.8946,
    1.8595,
    1.8331,
    1.8125,
    1.7959,
    1.7823,
    1.7709,
    1.7613,
    1.7531,
    1.7459,
    1.7396,
    1.7341,
    1.7291,
    1.7247,
    1.7207,
    1.7171,
    1.7139,
    1.7109,
    1.7081,
    1.7056,
    1.7033,
    1.7011,
    1.6991,
    1.6973,
)


@dataclass(frozen=True)
class RangeForecastRow:
    source_index: int
    timestamp_utc: datetime
    pair: str
    confidence: float
    current_price: float
    range_low_price: float
    range_high_price: float
    horizon_min: float
    cycle_id: str | None


@dataclass(frozen=True)
class RangeSignal:
    row: RangeForecastRow
    side: str
    entry: float
    range_width_pips: float
    box_position: float


@dataclass(frozen=True)
class VerifiedSparseS5Truth:
    coverage_by_pair: dict[str, tuple[tuple[datetime, datetime], ...]]
    file_sha256_by_path: dict[Path, str]
    provenance_sha256_by_path: dict[Path, str]
    summary_count: int
    file_count: int
    row_count: int
    receipt_count: int


def main() -> int:
    args = _parse_args()
    pairs = _parse_pair_filter(args.pairs)
    rows, load_stats = load_range_forecasts(
        args.forecast_history,
        pairs=pairs,
        dedupe_minutes=args.dedupe_minutes,
    )
    now = datetime.now(timezone.utc)
    mature_rows = [
        row
        for row in rows
        if raw_truth_end(row, ttl_minutes=args.ttl_minutes) <= now
    ]
    no_market_rows = [
        row
        for row in mature_rows
        if range_truth_is_no_market(row, ttl_minutes=args.ttl_minutes)
    ]
    no_market_row_set = set(no_market_rows)
    scorable_rows = [row for row in mature_rows if row not in no_market_row_set]
    history_dirs = _range_history_dirs(
        args.history_dir,
        granularity=args.granularity,
        auto_min_days=args.auto_history_min_days,
    )
    load_windows = range_truth_windows(scorable_rows, ttl_minutes=args.ttl_minutes)
    exact_truth_windows = range_exact_truth_windows(
        scorable_rows,
        ttl_minutes=args.ttl_minutes,
    )
    verified_truth = validate_sparse_s5_truth_provenance(
        history_dirs,
        required_windows_by_pair=exact_truth_windows,
        granularity=args.granularity,
    )
    candles_by_pair, candle_stats = _load_candles(
        history_dirs,
        granularity=args.granularity,
        windows_by_pair=load_windows,
    )
    _require_lossless_verified_candle_load(candle_stats, verified_truth)
    verify_sparse_s5_truth_snapshot_unchanged(verified_truth)
    results, score_stats = score_range_forecasts(
        scorable_rows,
        candles_by_pair,
        ttl_minutes=args.ttl_minutes,
        tp_grid=args.tp_grid_pips,
        sl_grid=args.sl_grid_pips,
        candle_interval=granularity_interval(args.granularity),
        verified_sparse_s5_coverage_by_pair=verified_truth.coverage_by_pair,
    )
    selections = train_validation_selections(
        results,
        train_fraction=args.train_fraction,
        min_train_fills=args.min_train_fills,
        min_validation_fills=args.min_validation_fills,
        min_total_fills=args.min_total_fills,
        family_truth_coverage={
            (str(item["pair"]), str(item["side"])): item
            for item in score_stats.get("pair_side_truth_coverage", [])
        },
    )
    report = {
        "generated_at_utc": _iso(now),
        "kind": "oanda_range_scout_replay_validate",
        "read_only": True,
        "live_permission_allowed": False,
        "adoption_blockers": [
            "RANGE_SCOUT_TTL_CLOSE_LIVE_CONTRACT_MISMATCH",
            "RANGE_SCOUT_ZERO_LATENCY_ENTRY_ASSUMPTION",
            "RANGE_SCOUT_LIVE_GATE_REPLAY_INCOMPLETE",
        ],
        "source": str(args.forecast_history),
        "history_dirs": [str(path) for path in history_dirs],
        "truth_acquisition_provenance": {
            "status": (
                "VERIFIED_SPARSE_S5_NO_TICK_COVERAGE"
                if verified_truth.receipt_count > 0
                else "NO_SCORABLE_TRUTH_WINDOWS"
            ),
            "summary_count": verified_truth.summary_count,
            "file_count": verified_truth.file_count,
            "row_count": verified_truth.row_count,
            "receipt_count": verified_truth.receipt_count,
            "synthetic_flat_candles_added": 0,
        },
        "truth_source": (
            f"local OANDA {args.granularity} bid/ask; LONG LIMIT fills on ask low and exits on bid, "
            "SHORT LIMIT fills on bid high and exits on ask; a fill-candle stop is conservative, "
            "while an ordering-ambiguous fill-candle target "
            "is ignored until a later bar (or the same close) proves a post-fill re-cross"
        ),
        "vehicle": {
            "selector": "NEAREST_RANGE_RAIL_ONE_SIDE",
            "granularity": args.granularity.upper(),
            "entry": "forecast range_low for LONG or range_high for SHORT",
            "entry_eligibility_assumption": (
                "order is active at the first complete S5 boundary at/after forecast emission "
                "with zero additional verifier/gateway latency (0 to <5 seconds of boundary delay)"
            ),
            "research_clock_contract": {
                "forecast_emission": "raw persisted forecast timestamp is never rounded in source data",
                "order_activation": "ceil forecast emission to the first S5 boundary at/after emission",
                "ttl_boundary": (
                    "floor raw forecast TTL/horizon end to the last S5 boundary at/before it; "
                    "no post-horizon candle is read"
                ),
                "boundary_adjustment_seconds": (
                    "each boundary changes by 0 to <5 seconds; effective activation-to-expiry "
                    "holding duration can be 0 to <10 seconds shorter than the raw TTL"
                ),
            },
            "ttl_minutes": args.ttl_minutes,
            "candle_interval_seconds": granularity_interval(args.granularity).total_seconds(),
            "same_pair_dedupe_minutes": args.dedupe_minutes,
            "take_profit_pips": list(args.tp_grid_pips),
            "stop_loss_pips": list(args.sl_grid_pips),
            "tp_must_not_cross_midpoint": True,
            "unfilled_is_not_a_trade": True,
            "timeout_exit_included": True,
            "live_contract_alignment": {
                "status": "MISMATCH",
                "simulation_exit": "TP, SL, or executable bid/ask TTL_CLOSE after fill",
                "current_live_exit": "attached broker TP/SL only; GTD expires only an unfilled entry order",
                "required_before_adoption": (
                    "either implement and audit a gateway-controlled filled-position time exit, "
                    "or rerun with the actual attached-TP/SL holding contract"
                ),
            },
            "minimum_pair_side_truth_window_coverage_rate": MIN_TRUTH_WINDOW_COVERAGE_RATE,
        },
        **load_stats,
        "mature_rows": len(mature_rows),
        "pending_future_truth_rows": len(rows) - len(mature_rows),
        "unscorable_no_market_rows": len(no_market_rows),
        **candle_stats,
        **score_stats,
        "comparison_period": {
            "forecast_start_utc": _iso(min(row.timestamp_utc for row in scorable_rows))
            if scorable_rows
            else None,
            "forecast_end_utc": _iso(max(truth_end(row, ttl_minutes=args.ttl_minutes) for row in scorable_rows))
            if scorable_rows
            else None,
        },
        "groups": group_metrics(results),
        "train_validation_selections": selections,
        "evaluated_oos_family_count": len(selections),
        "powered_oos_family_count": sum(
            item.get("oos_powered") is True for item in selections
        ),
        "tested_pair_side_family_count": len(
            {(item["pair"], item["side"]) for item in results}
        ),
        "tested_vehicle_count": len(
            {
                (
                    item["pair"],
                    item["side"],
                    item["take_profit_pips"],
                    item["stop_loss_pips"],
                )
                for item in results
            }
        ),
        "multiple_testing_caveat": (
            "all visible pair/side and TP/SL grids are searched; chronological validation and "
            "Student-t lower bounds reduce but do not eliminate family-wise selection risk"
        ),
        "candidate_rules": [item for item in selections if item.get("hypothesis_candidate") is True],
        "status": _report_status(selections),
    }
    # Recheck immediately before publishing: scoring/report assembly can be
    # long enough for a concurrently replaced ledger, summary, or candle file
    # to otherwise leave a report citing bytes that no longer exist.
    verify_sparse_s5_truth_snapshot_unchanged(verified_truth)
    write_report(report, args.output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-history", type=Path, default=Path("data/forecast_history.jsonl"))
    parser.add_argument("--history-dir", type=Path, action="append", default=None)
    parser.add_argument("--granularity", default="S5")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/reports/forecast_improvement/range_scout"))
    parser.add_argument("--auto-history-min-days", type=float, default=30.0)
    parser.add_argument("--ttl-minutes", type=int, default=DEFAULT_TTL_MINUTES)
    parser.add_argument("--dedupe-minutes", type=int, default=DEFAULT_DEDUPE_MINUTES)
    parser.add_argument("--tp-grid-pips", type=_parse_float_csv, default=DEFAULT_TP_GRID_PIPS)
    parser.add_argument("--sl-grid-pips", type=_parse_float_csv, default=DEFAULT_SL_GRID_PIPS)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--min-train-fills", type=int, default=DEFAULT_MIN_TRAIN_FILLS)
    parser.add_argument("--min-validation-fills", type=int, default=DEFAULT_MIN_VALIDATION_FILLS)
    parser.add_argument("--min-total-fills", type=int, default=DEFAULT_MIN_TOTAL_FILLS)
    args = parser.parse_args()
    args.granularity = str(args.granularity).upper()
    if args.granularity != "S5":
        parser.error("--granularity must be S5 for the canonical RANGE SCOUT bid/ask vehicle")
    if args.ttl_minutes <= 0 or args.ttl_minutes > DEFAULT_TTL_MINUTES:
        parser.error(f"--ttl-minutes must be in 1..{DEFAULT_TTL_MINUTES}")
    if args.dedupe_minutes < args.ttl_minutes:
        parser.error("--dedupe-minutes must be >= --ttl-minutes so one pair has no overlapping signal")
    if not 0.5 <= args.train_fraction < 1.0:
        parser.error("--train-fraction must be in [0.5, 1.0)")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in (*args.tp_grid_pips, *args.sl_grid_pips)
    ):
        parser.error("--tp-grid-pips and --sl-grid-pips must contain finite positive values")
    if min(args.min_train_fills, args.min_validation_fills, args.min_total_fills) <= 0:
        parser.error("minimum fill counts must be positive")
    if args.min_total_fills < args.min_train_fills + args.min_validation_fills:
        parser.error("--min-total-fills must cover train plus validation minimums")
    return args


def _range_history_dirs(
    explicit: Sequence[Path] | None,
    *,
    granularity: str,
    auto_min_days: float,
    default_root: Path = Path("logs/replay/oanda_range_scout_truth"),
) -> list[Path]:
    if explicit:
        # Multiple/non-overlapping acquisitions are an explicit operator
        # choice.  The provenance validator below still rejects overlap,
        # partials, missing receipts, and uncovered required windows.
        return _history_dirs(
            explicit,
            granularity=granularity,
            auto_min_days=auto_min_days,
        )
    latest_path = default_root / "latest_summary.json"
    if not latest_path.is_file():
        raise FileNotFoundError(
            f"missing {latest_path}; run the RANGE truth fetch or pass --history-dir"
        )
    latest = _strict_json_object(
        latest_path.read_bytes(),
        label="latest RANGE truth summary",
    )
    output_dir = latest.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("latest RANGE truth summary has no output_dir")
    resolved_root = default_root.resolve(strict=True)
    resolved_output = _strict_absolute_path(
        output_dir,
        label="latest RANGE truth output_dir",
        must_exist=True,
    )
    if not resolved_output.is_dir():
        raise ValueError("latest RANGE truth output_dir is not a directory")
    _require_path_inside(
        resolved_output,
        resolved_root,
        label="latest RANGE truth output_dir",
    )
    return [resolved_output]


def _report_status(selections: Sequence[dict[str, Any]]) -> str:
    powered = [item for item in selections if item.get("oos_powered") is True]
    if not powered:
        return "INSUFFICIENT_OOS_SAMPLE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT"
    if any(item.get("hypothesis_candidate") is True for item in powered):
        return "RANGE_SCOUT_RESEARCH_HYPOTHESES_EXIT_CONTRACT_MISMATCH"
    return "NO_OOS_EDGE_UNDER_TTL_CLOSE_RESEARCH_CONTRACT"


def load_range_forecasts(
    path: Path,
    *,
    pairs: set[str] | None = None,
    dedupe_minutes: int = DEFAULT_DEDUPE_MINUTES,
) -> tuple[list[RangeForecastRow], dict[str, Any]]:
    pair_filter = {str(item).upper() for item in (pairs or set()) if str(item).strip()}
    parsed: list[RangeForecastRow] = []
    seen: set[tuple[Any, ...]] = set()
    raw_range_rows = 0
    skipped_duplicate = 0
    skipped_invalid = 0
    skipped_pair_filter = 0
    with path.open(encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue
            if str(payload.get("direction") or "").upper() != "RANGE":
                continue
            raw_range_rows += 1
            pair = str(payload.get("pair") or "").upper().strip()
            if pair_filter and pair not in pair_filter:
                skipped_pair_filter += 1
                continue
            timestamp = _parse_time(payload.get("timestamp_utc"))
            current = _safe_float(payload.get("current_price"))
            low = _safe_float(payload.get("range_low_price"))
            high = _safe_float(payload.get("range_high_price"))
            if timestamp is None or not pair or current is None or low is None or high is None or not low < high:
                skipped_invalid += 1
                continue
            confidence = _safe_float(payload.get("confidence")) or 0.0
            horizon = _safe_float(payload.get("horizon_min")) or 0.0
            if horizon <= 0:
                skipped_invalid += 1
                continue
            cycle_id = str(payload.get("cycle_id") or "").strip() or None
            key = (
                ("cycle", cycle_id, pair)
                if cycle_id
                else (
                    "bucket",
                    pair,
                    timestamp.replace(microsecond=0).isoformat(),
                    round(confidence, 6),
                    low,
                    high,
                )
            )
            if key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            parsed.append(
                RangeForecastRow(
                    source_index=source_index,
                    timestamp_utc=timestamp,
                    pair=pair,
                    confidence=confidence,
                    current_price=current,
                    range_low_price=low,
                    range_high_price=high,
                    horizon_min=horizon,
                    cycle_id=cycle_id,
                )
            )

    deduped: list[RangeForecastRow] = []
    last_by_pair: dict[str, datetime] = {}
    persistence_deduped = 0
    for row in sorted(parsed, key=lambda item: (item.timestamp_utc, item.pair, item.source_index)):
        if not row.range_low_price <= row.current_price <= row.range_high_price:
            continue
        last = last_by_pair.get(row.pair)
        if last is not None and row.timestamp_utc - last < timedelta(minutes=dedupe_minutes):
            persistence_deduped += 1
            continue
        last_by_pair[row.pair] = row.timestamp_utc
        deduped.append(row)
    return deduped, {
        "raw_range_rows": raw_range_rows,
        "valid_unique_range_rows": len(parsed),
        "deduped_range_rows": len(deduped),
        "skipped_duplicate_rows": skipped_duplicate,
        "skipped_invalid_rows": skipped_invalid,
        "skipped_pair_filter_rows": skipped_pair_filter,
        "skipped_persistence_rows": persistence_deduped,
        "pair_filter": sorted(pair_filter),
    }


def raw_truth_end(row: RangeForecastRow, *, ttl_minutes: int) -> datetime:
    ttl = min(float(ttl_minutes), row.horizon_min)
    return row.timestamp_utc + timedelta(minutes=ttl)


def range_order_activation_at(row: RangeForecastRow) -> datetime:
    """First S5 boundary at/after forecast emission (never before it)."""

    value = row.timestamp_utc.astimezone(timezone.utc)
    floor_epoch = math.floor(value.timestamp() / 5.0) * 5
    floor_boundary = datetime.fromtimestamp(floor_epoch, tz=timezone.utc)
    return floor_boundary if value == floor_boundary else floor_boundary + S5_CANDLE_INTERVAL


def truth_end(row: RangeForecastRow, *, ttl_minutes: int) -> datetime:
    """Last S5 boundary that does not cross the raw forecast TTL/horizon."""

    raw_end = raw_truth_end(row, ttl_minutes=ttl_minutes).astimezone(timezone.utc)
    floor_epoch = math.floor(raw_end.timestamp() / 5.0) * 5
    return datetime.fromtimestamp(floor_epoch, tz=timezone.utc)


def range_truth_is_no_market(row: RangeForecastRow, *, ttl_minutes: int) -> bool:
    activation = range_order_activation_at(row)
    effective_end = truth_end(row, ttl_minutes=ttl_minutes)
    return _is_likely_fx_no_market_window(
        replace(
            row,
            timestamp_utc=activation,
            horizon_min=(effective_end - activation).total_seconds() / 60.0,
        )
    )


def range_truth_windows(
    rows: Sequence[RangeForecastRow],
    *,
    ttl_minutes: int,
) -> dict[str, list[tuple[datetime, datetime]]]:
    pad = timedelta(minutes=1)
    by_pair: dict[str, list[tuple[datetime, datetime]]] = collections.defaultdict(list)
    for row in rows:
        activation = range_order_activation_at(row)
        by_pair[row.pair].append(
            (activation - pad, truth_end(row, ttl_minutes=ttl_minutes) + pad)
        )
    return {pair: merge_windows(windows) for pair, windows in by_pair.items()}


def range_exact_truth_windows(
    rows: Sequence[RangeForecastRow],
    *,
    ttl_minutes: int,
) -> dict[str, list[tuple[datetime, datetime]]]:
    by_pair: dict[str, list[tuple[datetime, datetime]]] = collections.defaultdict(list)
    for row in rows:
        by_pair[row.pair].append(
            (
                range_order_activation_at(row),
                truth_end(row, ttl_minutes=ttl_minutes),
            )
        )
    return {pair: merge_windows(windows) for pair, windows in by_pair.items()}


def validate_sparse_s5_truth_provenance(
    history_dirs: Sequence[Path],
    *,
    required_windows_by_pair: Mapping[
        str,
        Sequence[tuple[datetime, datetime]],
    ],
    granularity: str,
) -> VerifiedSparseS5Truth:
    """Prove that missing S5 rows are OANDA no-tick gaps, not fetch gaps.

    Sparse-candle treatment is deliberately unavailable to legacy files.  It
    requires the canonical fetch summary, its exact receipt-chain link, and
    immutable published bytes for every pair/window used by this replay.
    """

    if str(granularity or "").upper() != "S5":
        raise ValueError("verified sparse no-tick coverage is defined only for S5")
    required = _normalize_required_windows(required_windows_by_pair)
    if not required:
        return VerifiedSparseS5Truth(
            coverage_by_pair={},
            file_sha256_by_path={},
            provenance_sha256_by_path={},
            summary_count=0,
            file_count=0,
            row_count=0,
            receipt_count=0,
        )

    resolved_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for raw_dir in history_dirs:
        resolved = Path(raw_dir).resolve(strict=True)
        if not resolved.is_dir() or resolved in seen_dirs:
            raise ValueError("history directories must be unique existing directories")
        seen_dirs.add(resolved)
        resolved_dirs.append(resolved)
    if not resolved_dirs:
        raise ValueError("verified sparse S5 coverage requires a history directory")

    fetch_script = (SCRIPT_ROOT / "oanda_range_scout_truth_fetch.py").resolve(strict=True)
    fetch_script_sha256 = _sha256_path(fetch_script)
    receipt_ledgers: dict[Path, tuple[list[dict[str, Any]], Path, str]] = {}
    file_sha256_by_path: dict[Path, str] = {}
    provenance_sha256_by_path: dict[Path, str] = {}
    selected_receipt_shas: set[str] = set()
    selected_paths: set[Path] = set()
    coverage: dict[str, list[tuple[datetime, datetime]]] = collections.defaultdict(list)
    seen_candles: dict[tuple[str, datetime], str] = {}
    total_rows = 0

    for history_dir in resolved_dirs:
        partials = sorted(history_dir.rglob("*.partial"))
        if partials:
            raise ValueError("history summary directory contains partial candle output")
        summary_path = history_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError("legacy history without summary.json cannot prove sparse S5 gaps")
        summary_bytes = summary_path.read_bytes()
        summary = _strict_json_object(summary_bytes, label="history fetch summary")
        if set(summary) != _FETCH_SUMMARY_KEYS:
            raise ValueError("history fetch summary schema invalid")
        if summary.get("schema_version") != RANGE_FETCH_SUMMARY_SCHEMA:
            raise ValueError("legacy or unknown history fetch summary schema")
        if summary.get("dry_run") is not False:
            raise ValueError("dry-run history cannot prove sparse S5 gaps")
        if summary.get("include_incomplete") is not False:
            raise ValueError("incomplete-candle acquisition cannot prove sparse S5 gaps")
        if summary.get("errors") != []:
            raise ValueError("history fetch summary contains errors")
        _strict_utc_timestamp(summary.get("generated_at_utc"), label="summary generated_at_utc")

        output_root = _strict_absolute_path(
            summary.get("output_root"),
            label="summary output_root",
            must_exist=True,
        )
        output_dir = _strict_absolute_path(
            summary.get("output_dir"),
            label="summary output_dir",
            must_exist=True,
        )
        if output_dir != history_dir:
            raise ValueError("summary output_dir does not identify its own directory")
        _require_path_inside(output_dir, output_root, label="summary output_dir")
        _require_path_inside(summary_path.resolve(), output_root, label="summary path")
        provenance_sha256_by_path[summary_path.resolve()] = _sha256_bytes(summary_bytes)

        summary_window = summary.get("window")
        if not isinstance(summary_window, dict) or set(summary_window) != {"from", "to"}:
            raise ValueError("history fetch summary window schema invalid")
        summary_start = _strict_utc_timestamp(summary_window.get("from"), label="summary from")
        summary_end = _strict_utc_timestamp(summary_window.get("to"), label="summary to")
        if summary_start >= summary_end:
            raise ValueError("history fetch summary window is not positive")
        if any(
            boundary.microsecond != 0 or boundary.timestamp() % 5 != 0
            for boundary in (summary_start, summary_end)
        ):
            raise ValueError("history fetch summary window is not aligned to S5 boundaries")
        generated_at = _strict_utc_timestamp(
            summary.get("generated_at_utc"),
            label="summary generated_at_utc",
        )
        if generated_at < summary_end:
            raise ValueError("history fetch summary predates truth-window maturity")

        pairs = summary.get("pairs")
        if (
            not isinstance(pairs, list)
            or not pairs
            or any(not isinstance(pair, str) or pair != pair.upper() or not pair for pair in pairs)
            or len(set(pairs)) != len(pairs)
        ):
            raise ValueError("history fetch summary pairs invalid")
        if summary.get("granularities") != ["S5"] or summary.get("price") != "BA":
            raise ValueError("sparse RANGE replay requires an exact S5/BA fetch summary")
        max_per_request = _strict_nonnegative_int(
            summary.get("max_candles_per_request"),
            label="max_candles_per_request",
        )
        if not 1 <= max_per_request <= OANDA_MAX_CANDLES_PER_REQUEST:
            raise ValueError("history max_candles_per_request exceeds broker contract")

        tasks = summary.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != len(pairs):
            raise ValueError("history fetch summary task cardinality invalid")
        expected_windows = len(
            list(
                _iter_time_chunks(
                    summary_start,
                    summary_end,
                    granularity="S5",
                    max_candles_per_request=max_per_request,
                )
            )
        )
        all_task_rows = 0
        all_task_requests = 0
        task_pairs: list[str] = []
        selected_in_dir: set[Path] = set()
        for task in tasks:
            if not isinstance(task, dict) or set(task) != _FETCH_TASK_KEYS:
                raise ValueError("history fetch task schema invalid")
            if task.get("schema_version") != RANGE_FETCH_TASK_SCHEMA:
                raise ValueError("legacy or unknown history fetch task schema")
            pair = task.get("pair")
            if not isinstance(pair, str) or pair not in pairs:
                raise ValueError("history fetch task pair invalid")
            task_pairs.append(pair)
            if task.get("granularity") != "S5" or task.get("price") != "BA":
                raise ValueError("history fetch task is not exact S5/BA")
            if task.get("published") is not True or task.get("dry_run") is not False:
                raise ValueError("unpublished or dry-run history task cannot prove coverage")
            if task.get("partial_path") is not None or task.get("errors") != []:
                raise ValueError("partial or errored history task cannot prove coverage")
            if not isinstance(task.get("compressed"), bool):
                raise ValueError("history fetch task compression flag invalid")
            if task.get("include_incomplete") is not False:
                raise ValueError("history fetch task includes incomplete candles")
            if (
                _strict_nonnegative_int(
                    task.get("max_candles_per_request"),
                    label="task max_candles_per_request",
                )
                != max_per_request
            ):
                raise ValueError("history fetch task request bound differs from summary")
            task_start = _strict_utc_timestamp(task.get("from"), label="task from")
            task_end = _strict_utc_timestamp(task.get("to"), label="task to")
            if (task_start, task_end) != (summary_start, summary_end):
                raise ValueError("history fetch task window differs from summary")
            task_windows = _strict_nonnegative_int(task.get("windows"), label="task windows")
            task_requests = _strict_nonnegative_int(task.get("requests"), label="task requests")
            task_rows = _strict_nonnegative_int(task.get("rows"), label="task rows")
            if task_windows != expected_windows or task_requests != task_windows:
                raise ValueError("history fetch did not complete every bounded request window")
            all_task_rows += task_rows
            all_task_requests += task_requests
            if pair not in required:
                continue

            candle_path = _strict_absolute_path(
                task.get("path"),
                label="task candle path",
                must_exist=True,
            )
            if not candle_path.is_file():
                raise ValueError("published candle path is not a regular file")
            _require_path_inside(candle_path, history_dir, label="published candle path")
            _require_path_inside(candle_path, output_root, label="published candle path")
            if candle_path.parent.name != pair:
                raise ValueError("published candle path pair directory mismatch")
            expected_name = (
                f"{pair}_S5_BA_{_stamp_utc(task_start)}_{_stamp_utc(task_end)}.jsonl"
                + (".gz" if task.get("compressed") is True else "")
            )
            if candle_path.name != expected_name:
                raise ValueError("published candle filename/window contract mismatch")
            if candle_path in selected_paths:
                raise ValueError("duplicate published candle path selected")

            receipt_path = output_root / RANGE_TRUTH_RECEIPT_FILE
            if output_root not in receipt_ledgers:
                if not receipt_path.is_file():
                    raise ValueError("legacy history without acquisition receipt cannot prove gaps")
                receipt_bytes = receipt_path.read_bytes()
                _strict_jsonl_objects(receipt_bytes, label="truth acquisition receipt ledger")
                receipts = _validate_range_truth_receipt_chain(receipt_bytes)
                if not receipts:
                    raise ValueError("truth acquisition receipt ledger is empty")
                for receipt in receipts:
                    _validate_receipt_primitive_schema(receipt)
                receipt_ledgers[output_root] = (
                    receipts,
                    receipt_path.resolve(),
                    _sha256_bytes(receipt_bytes),
                )
                provenance_sha256_by_path[receipt_path.resolve()] = _sha256_bytes(receipt_bytes)
            receipts = receipt_ledgers[output_root][0]
            receipt_sha = _strict_sha256(
                task.get("range_truth_acquisition_receipt_sha256"),
                label="task receipt sha256",
            )
            matches = [item for item in receipts if item.get("receipt_sha256") == receipt_sha]
            if len(matches) != 1:
                raise ValueError("task receipt SHA does not select exactly one chained receipt")
            receipt = matches[0]
            same_path_receipts = [
                item for item in receipts if item.get("candle_path") == str(candle_path)
            ]
            if len(same_path_receipts) != 1 or receipt_sha in selected_receipt_shas:
                raise ValueError("duplicate acquisition receipt selected")
            selected_receipt_shas.add(receipt_sha)
            _validate_selected_receipt(
                receipt,
                output_root=output_root,
                candle_path=candle_path,
                pair=pair,
                start=task_start,
                end=task_end,
                rows=task_rows,
                task_manifest_sha256_value=task_manifest_sha256(task),
                fetch_script=fetch_script,
                fetch_script_sha256=fetch_script_sha256,
            )

            file_sha = _sha256_path(candle_path)
            if receipt.get("candle_sha256") != file_sha:
                raise ValueError("published candle file SHA drifted from receipt")
            loaded_rows = _validate_published_s5_ba_file(
                candle_path,
                pair=pair,
                start=task_start,
                end=task_end,
                seen_candles=seen_candles,
            )
            if loaded_rows != task_rows or loaded_rows != receipt.get("rows"):
                raise ValueError("published candle row count differs from summary/receipt")
            if any(
                existing_start < task_end and task_start < existing_end
                for existing_start, existing_end in coverage[pair]
            ):
                raise ValueError("overlapping receipted task windows cannot prove no-tick gaps")
            selected_paths.add(candle_path)
            selected_in_dir.add(candle_path)
            file_sha256_by_path[candle_path] = file_sha
            total_rows += loaded_rows
            coverage[pair].append((task_start, task_end))

        if task_pairs != pairs or len(set(task_pairs)) != len(task_pairs):
            raise ValueError("history fetch task order/pair identity mismatch")
        if _strict_nonnegative_int(summary.get("total_rows"), label="summary total_rows") != all_task_rows:
            raise ValueError("history fetch summary total_rows mismatch")
        if (
            _strict_nonnegative_int(summary.get("total_requests"), label="summary total_requests")
            != all_task_requests
        ):
            raise ValueError("history fetch summary total_requests mismatch")
        actual_selected = {
            path.resolve()
            for path in history_dir.glob("*/*_S5_BA_*.jsonl*")
            if (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"))
            and path.parent.name.upper() in required
        }
        if actual_selected != selected_in_dir:
            raise ValueError("requested-pair candle files are unlisted or missing from summary")

    merged_coverage = {
        pair: tuple(merge_windows(windows))
        for pair, windows in coverage.items()
    }
    for pair, windows in required.items():
        for start, end in windows:
            if not _window_is_covered(merged_coverage.get(pair, ()), start, end):
                raise ValueError(f"verified S5/BA acquisition does not cover {pair} truth window")
    return VerifiedSparseS5Truth(
        coverage_by_pair=merged_coverage,
        file_sha256_by_path=file_sha256_by_path,
        provenance_sha256_by_path=provenance_sha256_by_path,
        summary_count=len(resolved_dirs),
        file_count=len(file_sha256_by_path),
        row_count=total_rows,
        receipt_count=len(selected_receipt_shas),
    )


def verify_sparse_s5_truth_snapshot_unchanged(provenance: VerifiedSparseS5Truth) -> None:
    for path, expected in {
        **provenance.provenance_sha256_by_path,
        **provenance.file_sha256_by_path,
    }.items():
        if not path.is_file() or _sha256_path(path) != expected:
            raise ValueError("S5 truth provenance changed while replay was loading")


def _require_lossless_verified_candle_load(
    candle_stats: Mapping[str, Any],
    provenance: VerifiedSparseS5Truth,
) -> None:
    if any(
        _strict_nonnegative_int(candle_stats.get(field), label=field) != 0
        for field in (
            "history_skipped_rows",
            "history_duplicate_candles",
            "history_conflicting_candles",
        )
    ):
        raise ValueError("verified S5 replay parser skipped, duplicated, or conflicted candles")
    if _strict_nonnegative_int(candle_stats.get("history_files"), label="history_files") != provenance.file_count:
        raise ValueError("verified S5 replay loaded a different file set")
    if _strict_nonnegative_int(candle_stats.get("history_raw_rows"), label="history_raw_rows") != provenance.row_count:
        raise ValueError("verified S5 replay loaded a different row count")


def _normalize_required_windows(
    value: Mapping[str, Sequence[tuple[datetime, datetime]]],
) -> dict[str, tuple[tuple[datetime, datetime], ...]]:
    normalized: dict[str, tuple[tuple[datetime, datetime], ...]] = {}
    for raw_pair, raw_windows in value.items():
        pair = str(raw_pair or "").upper()
        if not pair or pair != raw_pair:
            raise ValueError("required truth pair must be canonical uppercase")
        windows: list[tuple[datetime, datetime]] = []
        for start, end in raw_windows:
            if (
                not isinstance(start, datetime)
                or not isinstance(end, datetime)
                or start.tzinfo is None
                or end.tzinfo is None
            ):
                raise ValueError("required truth windows must be timezone-aware datetimes")
            start_utc = start.astimezone(timezone.utc)
            end_utc = end.astimezone(timezone.utc)
            if start_utc >= end_utc:
                raise ValueError("required truth windows must be positive")
            windows.append((start_utc, end_utc))
        if windows:
            normalized[pair] = tuple(merge_windows(windows))
    return normalized


def _validate_selected_receipt(
    receipt: Mapping[str, Any],
    *,
    output_root: Path,
    candle_path: Path,
    pair: str,
    start: datetime,
    end: datetime,
    rows: int,
    task_manifest_sha256_value: str,
    fetch_script: Path,
    fetch_script_sha256: str,
) -> None:
    if receipt.get("output_root") != str(output_root):
        raise ValueError("receipt output_root mismatch")
    if receipt.get("candle_path") != str(candle_path):
        raise ValueError("receipt candle_path mismatch")
    if receipt.get("pair") != pair:
        raise ValueError("receipt pair mismatch")
    if receipt.get("granularity") != "S5" or receipt.get("price_component") != "BA":
        raise ValueError("receipt is not exact S5/BA")
    if receipt.get("source_base_url") != PRODUCTION_OANDA_BASE_URL:
        raise ValueError("receipt source endpoint is not production OANDA")
    window = receipt.get("window")
    if not isinstance(window, dict) or set(window) != {"from_utc", "to_utc"}:
        raise ValueError("receipt window schema invalid")
    if (
        _strict_utc_timestamp(window.get("from_utc"), label="receipt from") != start
        or _strict_utc_timestamp(window.get("to_utc"), label="receipt to") != end
    ):
        raise ValueError("receipt window mismatch")
    if _strict_nonnegative_int(receipt.get("rows"), label="receipt rows") != rows:
        raise ValueError("receipt rows mismatch")
    if receipt.get("task_manifest_sha256") != task_manifest_sha256_value:
        raise ValueError("receipt task-manifest SHA mismatch")
    recorded = _strict_utc_timestamp(receipt.get("recorded_at_utc"), label="receipt recorded_at")
    if recorded < end or recorded > datetime.now(timezone.utc):
        raise ValueError("receipt was not recorded after the complete requested window")
    if receipt.get("fetch_script_path") != str(fetch_script):
        raise ValueError("receipt fetch-script path mismatch")
    if receipt.get("fetch_script_sha256") != fetch_script_sha256:
        raise ValueError("receipt fetch-script SHA drifted")
    if receipt.get("dependencies") != expected_dependency_records():
        raise ValueError("receipt acquisition dependency SHA drifted")
    _strict_sha256(receipt.get("candle_sha256"), label="receipt candle sha256")


def _validate_receipt_primitive_schema(receipt: Mapping[str, Any]) -> None:
    _strict_nonnegative_int(receipt.get("sequence"), label="receipt sequence")
    if receipt.get("sequence") < 1:
        raise ValueError("receipt sequence must be positive")
    _strict_nonnegative_int(receipt.get("rows"), label="receipt rows")
    for field in (
        "recorded_at_utc",
        "output_root",
        "candle_path",
        "pair",
        "granularity",
        "price_component",
        "source_base_url",
        "fetch_script_path",
    ):
        if not isinstance(receipt.get(field), str) or not receipt.get(field):
            raise ValueError(f"receipt {field} type invalid")
    recorded = _strict_utc_timestamp(
        receipt.get("recorded_at_utc"),
        label="receipt recorded_at_utc",
    )
    if recorded > datetime.now(timezone.utc):
        raise ValueError("receipt recorded_at_utc is in the future")
    if (
        receipt.get("pair") != str(receipt.get("pair")).upper()
        or receipt.get("granularity") != "S5"
        or receipt.get("price_component") != "BA"
        or receipt.get("source_base_url") != PRODUCTION_OANDA_BASE_URL
    ):
        raise ValueError("receipt source identity is invalid")
    for field in ("output_root", "candle_path", "fetch_script_path"):
        if not Path(str(receipt.get(field))).is_absolute():
            raise ValueError(f"receipt {field} is not absolute")
    window = receipt.get("window")
    if not isinstance(window, dict) or set(window) != {"from_utc", "to_utc"}:
        raise ValueError("receipt window schema invalid")
    window_start = _strict_utc_timestamp(window.get("from_utc"), label="receipt from")
    window_end = _strict_utc_timestamp(window.get("to_utc"), label="receipt to")
    if window_start >= window_end or recorded < window_end:
        raise ValueError("receipt window maturity invalid")
    if any(
        boundary.microsecond != 0 or boundary.timestamp() % 5 != 0
        for boundary in (window_start, window_end)
    ):
        raise ValueError("receipt window is not aligned to S5 boundaries")
    for field in (
        "candle_sha256",
        "task_manifest_sha256",
        "fetch_script_sha256",
        "receipt_sha256",
    ):
        _strict_sha256(receipt.get(field), label=f"receipt {field}")
    previous = receipt.get("previous_receipt_sha256")
    if previous is not None:
        _strict_sha256(previous, label="receipt previous sha256")
    dependencies = receipt.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise ValueError("receipt dependency list invalid")
    dependency_roles = [
        dependency.get("role")
        for dependency in dependencies
        if isinstance(dependency, dict)
    ]
    if dependency_roles != ["BASE_HISTORY_FETCH", "OANDA_READ_CLIENT"]:
        raise ValueError("receipt dependency roles invalid")
    for dependency in dependencies:
        if not isinstance(dependency, dict) or set(dependency) != {"role", "path", "sha256"}:
            raise ValueError("receipt dependency schema invalid")
        if not isinstance(dependency.get("role"), str) or not dependency.get("role"):
            raise ValueError("receipt dependency role invalid")
        _strict_absolute_path(
            dependency.get("path"),
            label="receipt dependency path",
            must_exist=True,
        )
        _strict_sha256(dependency.get("sha256"), label="receipt dependency sha256")


def _validate_published_s5_ba_file(
    path: Path,
    *,
    pair: str,
    start: datetime,
    end: datetime,
    seen_candles: dict[tuple[str, datetime], str],
) -> int:
    count = 0
    previous_time: datetime | None = None
    with _open_history_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank row in published candle file at line {line_number}")
            payload = _strict_json_object(line.encode("utf-8"), label="published candle row")
            if set(payload) != _FETCHED_S5_BA_ROW_KEYS:
                raise ValueError("published S5/BA row schema invalid")
            if (
                payload.get("pair") != pair
                or payload.get("granularity") != "S5"
                or payload.get("price") != "BA"
                or payload.get("complete") is not True
            ):
                raise ValueError("published S5/BA row identity invalid")
            raw_timestamp = payload.get("time")
            if not isinstance(raw_timestamp, str) or not raw_timestamp.endswith("Z"):
                raise ValueError("published S5/BA row timestamp schema invalid")
            volume = payload.get("volume")
            if not isinstance(volume, int) or isinstance(volume, bool) or volume < 0:
                raise ValueError("published S5/BA row volume invalid")
            if not _strict_ohlc_payload(payload.get("bid")) or not _strict_ohlc_payload(
                payload.get("ask")
            ):
                raise ValueError("published S5/BA OHLC schema invalid")
            candle = _candle_from_payload(payload)
            if candle is None or candle.pair != pair:
                raise ValueError("published S5/BA row cannot be parsed losslessly")
            timestamp = candle.timestamp_utc
            if not start <= timestamp < end:
                raise ValueError("published S5/BA row lies outside receipted coverage")
            epoch_seconds = timestamp.timestamp()
            if timestamp.microsecond != 0 or epoch_seconds % 5 != 0:
                raise ValueError("published S5 row timestamp is not on an S5 boundary")
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("published S5 rows are duplicate or out of order")
            previous_time = timestamp
            row_digest = _sha256_bytes(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            key = (pair, timestamp)
            if key in seen_candles:
                raise ValueError("duplicate or conflicting S5 candle across published files")
            seen_candles[key] = row_digest
            count += 1
    return count


def _strict_ohlc_payload(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"o", "h", "l", "c"}:
        return False
    numbers: dict[str, float] = {}
    for key in ("o", "h", "l", "c"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return False
        number = float(raw)
        if not math.isfinite(number) or number <= 0.0:
            return False
        numbers[key] = number
    return (
        numbers["l"] <= numbers["o"] <= numbers["h"]
        and numbers["l"] <= numbers["c"] <= numbers["h"]
    )


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_no_duplicate_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is malformed or has duplicate keys") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _strict_jsonl_objects(payload: bytes, *, label: str) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows = text.splitlines()
    if not rows or any(not row.strip() for row in rows):
        raise ValueError(f"{label} is empty or contains blank rows")
    for row in rows:
        _strict_json_object(row.encode("utf-8"), label=label)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_utc_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    return parsed.astimezone(timezone.utc)


def _strict_nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_absolute_path(value: object, *, label: str, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        resolved = Path(value).resolve(strict=must_exist)
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if str(resolved) != value:
        raise ValueError(f"{label} must be canonical")
    return resolved


def _require_path_inside(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its acquisition root") from exc


def _window_is_covered(
    coverage: Sequence[tuple[datetime, datetime]],
    start: datetime,
    end: datetime,
) -> bool:
    return any(covered_start <= start and end <= covered_end for covered_start, covered_end in coverage)


def _stamp_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def granularity_interval(granularity: str) -> timedelta:
    value = str(granularity or "").strip().upper()
    units = {"S": 1, "M": 60, "H": 60 * 60, "D": 24 * 60 * 60}
    prefix = value[:1]
    try:
        count = int(value[1:]) if prefix != "D" else int(value[1:] or "1")
    except ValueError as exc:
        raise ValueError(f"unsupported candle granularity: {granularity}") from exc
    if prefix not in units or count <= 0:
        raise ValueError(f"unsupported candle granularity: {granularity}")
    return timedelta(seconds=count * units[prefix])


def merge_windows(windows: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(windows):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def range_signal(row: RangeForecastRow) -> RangeSignal:
    pip = 1.0 / instrument_pip_factor(row.pair)
    width_pips = (row.range_high_price - row.range_low_price) / pip
    midpoint = (row.range_low_price + row.range_high_price) / 2.0
    side = "LONG" if row.current_price <= midpoint else "SHORT"
    entry = row.range_low_price if side == "LONG" else row.range_high_price
    box_position = (row.current_price - row.range_low_price) / (row.range_high_price - row.range_low_price)
    return RangeSignal(row=row, side=side, entry=entry, range_width_pips=width_pips, box_position=box_position)


def score_range_forecasts(
    rows: Sequence[RangeForecastRow],
    candles_by_pair: dict[str, list[QuoteCandle]],
    *,
    ttl_minutes: int,
    tp_grid: Sequence[float],
    sl_grid: Sequence[float],
    candle_interval: timedelta,
    verified_sparse_s5_coverage_by_pair: Mapping[
        str,
        Sequence[tuple[datetime, datetime]],
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    eligible_signals = 0
    complete_truth_windows = 0
    filled_signals: set[tuple[int, str]] = set()
    skipped_no_window = 0
    skipped_incomplete_truth_window = 0
    skipped_not_passive = 0
    skipped_unfilled = 0
    ambiguous_fill_bar_target_cases = 0
    pair_side_truth_counts: dict[tuple[str, str], collections.Counter[str]] = (
        collections.defaultdict(collections.Counter)
    )
    pair_side_forecast_rows_by_day: dict[
        tuple[str, str], collections.Counter[str]
    ] = collections.defaultdict(collections.Counter)
    pair_side_complete_windows_by_day: dict[
        tuple[str, str], collections.Counter[str]
    ] = collections.defaultdict(collections.Counter)
    times_by_pair = {
        pair: [item.timestamp_utc for item in candles]
        for pair, candles in candles_by_pair.items()
    }
    for row in rows:
        signal = range_signal(row)
        family = (row.pair, signal.side)
        forecast_day = row.timestamp_utc.date().isoformat()
        truth_counts = pair_side_truth_counts[family]
        truth_counts["forecast_rows"] += 1
        pair_side_forecast_rows_by_day[family][forecast_day] += 1
        candles = candles_by_pair.get(row.pair) or []
        if not candles:
            skipped_no_window += 1
            continue
        times = times_by_pair[row.pair]
        activation_at = range_order_activation_at(row)
        forecast_end = truth_end(row, ttl_minutes=ttl_minutes)
        if forecast_end <= activation_at:
            skipped_incomplete_truth_window += 1
            continue
        start = bisect.bisect_left(times, activation_at)
        # OANDA candle timestamps mark interval start.  Requiring the candle's
        # full interval to end by the truth boundary prevents its high/low or
        # close from reading price action after forecast/TTL expiry.
        last_complete_start = forecast_end - candle_interval
        end = bisect.bisect_right(times, last_complete_start)
        if start >= len(candles) or end <= start:
            skipped_no_window += 1
            continue
        window_candles = candles[start:end]
        sparse_coverage_verified = (
            _window_is_covered(
                verified_sparse_s5_coverage_by_pair.get(row.pair, ()),
                activation_at,
                forecast_end,
            )
            if verified_sparse_s5_coverage_by_pair is not None
            else False
        )
        if (
            verified_sparse_s5_coverage_by_pair is not None
            and not sparse_coverage_verified
        ):
            skipped_incomplete_truth_window += 1
            continue
        if not complete_truth_window(
            window_candles,
            forecast_start=activation_at,
            forecast_end=forecast_end,
            candle_interval=candle_interval,
            verified_sparse_no_tick_coverage=sparse_coverage_verified,
        ):
            skipped_incomplete_truth_window += 1
            continue
        complete_truth_windows += 1
        truth_counts["complete_truth_windows"] += 1
        pair_side_complete_windows_by_day[family][forecast_day] += 1
        pip = 1.0 / instrument_pip_factor(row.pair)
        first = candles[start]
        leading_verified_no_tick_gap = (
            sparse_coverage_verified and first.timestamp_utc > activation_at
        )
        if signal.side == "LONG":
            passive_at_first_quote = signal.entry < first.ask.o
            first_quote_crossed_limit = first.ask.o <= signal.entry
        else:
            passive_at_first_quote = signal.entry > first.bid.o
            first_quote_crossed_limit = first.bid.o >= signal.entry
        # A receipted leading no-tick gap has no executable activation quote.
        # If the first later quote has already crossed the passive limit, that
        # quote fills the live order; treating it as "not passive" would erase
        # exactly the gap-crossing outcomes (including losses).  Outside that
        # independently verified gap, retain the original strict passivity
        # requirement.
        if not passive_at_first_quote and not (
            leading_verified_no_tick_gap and first_quote_crossed_limit
        ):
            skipped_not_passive += 1
            continue
        eligible_signals += 1
        fill_index = next(
            (
                idx
                for idx in range(start, end)
                if (signal.side == "LONG" and candles[idx].ask.l <= signal.entry)
                or (signal.side == "SHORT" and candles[idx].bid.h >= signal.entry)
            ),
            None,
        )
        if fill_index is None:
            skipped_unfilled += 1
            continue
        filled_signals.add((row.source_index, signal.side))
        for take_profit_pips in tp_grid:
            # A rail reversion TP must remain at or before the forecast midpoint.
            if take_profit_pips > signal.range_width_pips / 2.0 + 1e-9:
                continue
            for stop_loss_pips in sl_grid:
                outcome = simulate_filled_signal(
                    signal,
                    candles[fill_index:end],
                    take_profit_pips=float(take_profit_pips),
                    stop_loss_pips=float(stop_loss_pips),
                    pip=pip,
                    candle_interval=candle_interval,
                    timeout_at_utc=forecast_end,
                )
                if outcome.get("fill_bar_target_ambiguous") is True:
                    ambiguous_fill_bar_target_cases += 1
                results.append(
                    {
                        "source_index": row.source_index,
                        "timestamp_utc": _iso(row.timestamp_utc),
                        "campaign_day_jst": row.timestamp_utc.astimezone(JST).date().isoformat(),
                        "granularity": "S5",
                        "pair": row.pair,
                        "side": signal.side,
                        "confidence": round(row.confidence, 6),
                        "confidence_bucket": confidence_bucket(row.confidence),
                        "range_width_pips": round(signal.range_width_pips, 6),
                        "range_width_bucket": range_width_bucket(signal.range_width_pips),
                        "box_position": round(signal.box_position, 6),
                        "take_profit_pips": float(take_profit_pips),
                        "stop_loss_pips": float(stop_loss_pips),
                        **outcome,
                    }
                )
    return results, {
        "eligible_signals": eligible_signals,
        "filled_signals": len(filled_signals),
        "fill_rate": round(len(filled_signals) / eligible_signals, 6) if eligible_signals else 0.0,
        "skipped_no_price_window": skipped_no_window,
        "skipped_incomplete_truth_window": skipped_incomplete_truth_window,
        "skipped_not_passive": skipped_not_passive,
        "skipped_unfilled": skipped_unfilled,
        "ambiguous_fill_bar_target_cases": ambiguous_fill_bar_target_cases,
        "complete_truth_windows": complete_truth_windows,
        "truth_window_coverage_rate": round(
            complete_truth_windows
            / (complete_truth_windows + skipped_no_window + skipped_incomplete_truth_window),
            6,
        )
        if complete_truth_windows + skipped_no_window + skipped_incomplete_truth_window
        else 0.0,
        "pair_side_truth_coverage": [
            {
                "pair": pair,
                "side": side,
                "forecast_rows": int(counts["forecast_rows"]),
                "complete_truth_windows": int(counts["complete_truth_windows"]),
                "coverage_rate": round(
                    counts["complete_truth_windows"] / counts["forecast_rows"],
                    6,
                )
                if counts["forecast_rows"]
                else 0.0,
                "forecast_rows_by_day": dict(
                    sorted(pair_side_forecast_rows_by_day[(pair, side)].items())
                ),
                "complete_truth_windows_by_day": dict(
                    sorted(pair_side_complete_windows_by_day[(pair, side)].items())
                ),
            }
            for (pair, side), counts in sorted(pair_side_truth_counts.items())
        ],
    }


def complete_truth_window(
    candles: Sequence[QuoteCandle],
    *,
    forecast_start: datetime,
    forecast_end: datetime,
    candle_interval: timedelta,
    verified_sparse_no_tick_coverage: bool = False,
) -> bool:
    if not candles:
        return False
    if any(
        current.timestamp_utc <= previous.timestamp_utc
        for previous, current in zip(candles, candles[1:])
    ):
        return False
    if verified_sparse_no_tick_coverage:
        # OANDA does not synthesize a flat S5 bar when no tick occurred.  Once
        # the independently verified acquisition window proves that the whole
        # interval was requested successfully, leading/internal absence is
        # natural no-tick truth.  The first later bar is the first executable
        # quote at which passivity/fill can be tested; requiring a synthetic
        # activation-boundary bar would selectively discard real later losses.
        # The TTL boundary still needs a real bar because carrying an old close
        # across a sparse tail would invent an executable TTL price/timestamp.
        return (
            forecast_start <= candles[0].timestamp_utc < forecast_end
            and candles[-1].timestamp_utc + candle_interval == forecast_end
        )
    if candles[0].timestamp_utc - forecast_start >= candle_interval:
        return False
    if candles[-1].timestamp_utc + candle_interval != forecast_end:
        return False
    return all(
        current.timestamp_utc - previous.timestamp_utc <= candle_interval
        for previous, current in zip(candles, candles[1:])
    )


def simulate_filled_signal(
    signal: RangeSignal,
    candles: Sequence[QuoteCandle],
    *,
    take_profit_pips: float,
    stop_loss_pips: float,
    pip: float,
    candle_interval: timedelta = timedelta(seconds=5),
    timeout_at_utc: datetime | None = None,
) -> dict[str, Any]:
    if (
        not math.isfinite(float(take_profit_pips))
        or not math.isfinite(float(stop_loss_pips))
        or take_profit_pips <= 0.0
        or stop_loss_pips <= 0.0
    ):
        raise ValueError("RANGE SCOUT TP/SL distances must be finite and positive")
    fill_bar_target_ambiguous = False
    for index, candle in enumerate(candles):
        if signal.side == "LONG":
            target_price = signal.entry + take_profit_pips * pip
            stop_hit = candle.bid.l <= signal.entry - stop_loss_pips * pip
            target_hit = candle.bid.h >= target_price
            close_proves_post_fill_target = candle.bid.c >= target_price
        else:
            target_price = signal.entry - take_profit_pips * pip
            stop_hit = candle.ask.h >= signal.entry + stop_loss_pips * pip
            target_hit = candle.ask.l <= target_price
            close_proves_post_fill_target = candle.ask.c <= target_price
        if stop_hit:
            exit_at = candle.timestamp_utc + candle_interval
            return {
                "realized_pips": -stop_loss_pips,
                "exit_reason": "STOP_LOSS",
                "same_candle_stop_first": target_hit,
                "scorable": True,
                "fill_bar_target_ambiguous": (
                    fill_bar_target_ambiguous or bool(index == 0 and target_hit)
                ),
                "fill_at_utc": _iso(candles[0].timestamp_utc),
                "exit_at_utc": _iso(exit_at),
                "resolved_day_utc": exit_at.date().isoformat(),
            }
        # The fill and target may have occurred in the opposite order inside
        # the same S5 OHLC bar.  If the executable close remains beyond target,
        # price necessarily crossed the target after the passive fill and the
        # TP is proved.  Otherwise ignore that first target and continue from
        # the next complete bar; this scores the adverse target-before-fill
        # ordering instead of deleting the sample from the evidence set.
        if target_hit and index == 0:
            fill_bar_target_ambiguous = True
            if not close_proves_post_fill_target:
                continue
        if target_hit:
            exit_at = candle.timestamp_utc + candle_interval
            return {
                "realized_pips": take_profit_pips,
                "exit_reason": "TAKE_PROFIT",
                "same_candle_stop_first": False,
                "scorable": True,
                "fill_bar_target_ambiguous": fill_bar_target_ambiguous,
                "fill_at_utc": _iso(candles[0].timestamp_utc),
                "exit_at_utc": _iso(exit_at),
                "resolved_day_utc": exit_at.date().isoformat(),
            }
    last = candles[-1]
    if (
        timeout_at_utc is not None
        and last.timestamp_utc + candle_interval != timeout_at_utc
    ):
        raise ValueError("TTL_CLOSE requires a real candle ending at the TTL boundary")
    executable_close = last.bid.c if signal.side == "LONG" else last.ask.c
    realized = (
        (executable_close - signal.entry) / pip
        if signal.side == "LONG"
        else (signal.entry - executable_close) / pip
    )
    exit_at = timeout_at_utc or (last.timestamp_utc + candle_interval)
    return {
        "realized_pips": round(realized, 6),
        "exit_reason": "TTL_CLOSE",
        "same_candle_stop_first": False,
        "scorable": True,
        "fill_bar_target_ambiguous": fill_bar_target_ambiguous,
        "fill_at_utc": _iso(candles[0].timestamp_utc),
        "exit_at_utc": _iso(exit_at),
        "resolved_day_utc": exit_at.date().isoformat(),
    }


def group_metrics(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for item in results:
        key = (item["pair"], item["side"], item["take_profit_pips"], item["stop_loss_pips"])
        grouped[key].append(item)
    rows = []
    for (pair, side, tp, sl), items in grouped.items():
        rows.append({"pair": pair, "side": side, "take_profit_pips": tp, "stop_loss_pips": sl, **metrics(items)})
    rows.sort(
        key=lambda item: (
            -float(item["mean_pips"] if item.get("mean_pips") is not None else -999.0),
            -int(item["fills"]),
            item["pair"],
            item["side"],
        )
    )
    return rows


def train_validation_selections(
    results: Sequence[dict[str, Any]],
    *,
    train_fraction: float,
    min_train_fills: int,
    min_validation_fills: int,
    min_total_fills: int,
    family_truth_coverage: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    by_vehicle: dict[tuple[str, str, float, float], list[dict[str, Any]]] = collections.defaultdict(list)
    for item in results:
        by_vehicle[(item["pair"], item["side"], item["take_profit_pips"], item["stop_loss_pips"])].append(item)
    by_family: dict[tuple[str, str], list[tuple[tuple[str, str, float, float], list[dict[str, Any]]]]] = collections.defaultdict(list)
    for key, items in by_vehicle.items():
        items.sort(key=lambda item: (item["timestamp_utc"], item["source_index"]))
        by_family[(key[0], key[1])].append((key, items))

    selections: list[dict[str, Any]] = []
    for family, vehicles in sorted(by_family.items()):
        result_days = {
            str(item["timestamp_utc"])[:10]
            for _key, items in vehicles
            for item in items
        }
        if family_truth_coverage is None:
            coverage = {
                "pair": family[0],
                "side": family[1],
                "forecast_rows": len(result_days),
                "complete_truth_windows": len(result_days),
                "coverage_rate": 1.0,
                "forecast_rows_by_day": {day: 1 for day in result_days},
                "complete_truth_windows_by_day": {day: 1 for day in result_days},
            }
        else:
            coverage = dict((family_truth_coverage or {}).get(family) or {})
        coverage_days = {
            str(day)
            for day, count in (
                coverage.get("forecast_rows_by_day") or {}
            ).items()
            if int(count or 0) > 0
        }
        family_days = sorted(coverage_days or result_days)
        if len(family_days) < 2:
            continue
        day_cut = max(
            1,
            min(len(family_days) - 1, int(len(family_days) * train_fraction)),
        )
        train_days = set(family_days[:day_cut])
        validation_days = set(family_days[day_cut:])
        train_truth_coverage = _truth_coverage_for_days(coverage, train_days)
        validation_truth_coverage = _truth_coverage_for_days(
            coverage,
            validation_days,
        )
        candidates: list[
            tuple[
                float,
                float,
                int,
                tuple[str, str, float, float],
                list[dict[str, Any]],
            ]
        ] = []
        for key, items in vehicles:
            train = [
                item
                for item in items
                if str(item["timestamp_utc"])[:10] in train_days
            ]
            train_metrics = metrics(train)
            if len(train) < min_train_fills:
                continue
            candidates.append(
                (
                    float(train_metrics["mean_pips"]),
                    _profit_factor_value(train_metrics.get("profit_factor")),
                    len(train),
                    key,
                    items,
                )
            )
        if not candidates:
            continue
        _mean, _pf, _n, key, items = max(candidates)
        train = [
            item
            for item in items
            if str(item["timestamp_utc"])[:10] in train_days
        ]
        validation = [
            item
            for item in items
            if str(item["timestamp_utc"])[:10] not in train_days
        ]
        full_metrics = metrics(items)
        train_metrics = metrics(train)
        validation_metrics = metrics(validation)
        truth_coverage_rate = float(coverage.get("coverage_rate") or 0.0)
        oos_powered = bool(
            len(items) >= min_total_fills
            and len(train) >= min_train_fills
            and len(validation) >= min_validation_fills
            and int(train_metrics["active_days"]) >= 5
            and int(validation_metrics["active_days"]) >= 5
            and truth_coverage_rate >= MIN_TRUTH_WINDOW_COVERAGE_RATE
            and train_truth_coverage["coverage_rate"]
            >= MIN_TRUTH_WINDOW_COVERAGE_RATE
            and validation_truth_coverage["coverage_rate"]
            >= MIN_TRUTH_WINDOW_COVERAGE_RATE
        )
        hypothesis_candidate = bool(
            oos_powered
            and float(train_metrics["mean_pips"]) > 0.0
            and _profit_factor_value(train_metrics.get("profit_factor")) >= 1.2
            and float(validation_metrics["mean_pips"]) > 0.0
            and _profit_factor_value(validation_metrics.get("profit_factor")) >= 1.2
            and float(train_metrics.get("one_sided_95_mean_lower_pips") or 0.0) > 0.0
            and float(validation_metrics.get("one_sided_95_mean_lower_pips") or 0.0) > 0.0
            and float(full_metrics.get("one_sided_95_mean_lower_pips") or 0.0) > 0.0
            and float(train_metrics.get("one_sided_95_daily_mean_lower_pips") or 0.0) > 0.0
            and float(validation_metrics.get("one_sided_95_daily_mean_lower_pips") or 0.0) > 0.0
            and float(full_metrics.get("one_sided_95_daily_mean_lower_pips") or 0.0) > 0.0
            and int(train_metrics["active_days"]) >= 5
            and int(validation_metrics["active_days"]) >= 5
            and int(full_metrics["active_days"]) >= 5
            and float(train_metrics["positive_day_rate"]) >= 2.0 / 3.0
            and float(validation_metrics["positive_day_rate"]) >= 2.0 / 3.0
            and float(full_metrics["positive_day_rate"]) >= 2.0 / 3.0
            and float(train_metrics["max_daily_sample_share"]) <= 0.70
            and float(validation_metrics["max_daily_sample_share"]) <= 0.70
            and float(full_metrics["max_daily_sample_share"]) <= 0.70
        )
        selections.append(
            {
                "pair": family[0],
                "side": family[1],
                "take_profit_pips": key[2],
                "stop_loss_pips": key[3],
                "train": train_metrics,
                "validation": validation_metrics,
                "full": full_metrics,
                "truth_coverage": {
                    **coverage,
                    "train": train_truth_coverage,
                    "validation": validation_truth_coverage,
                },
                "minimum_truth_window_coverage_rate": MIN_TRUTH_WINDOW_COVERAGE_RATE,
                "oos_powered": oos_powered,
                "hypothesis_candidate": hypothesis_candidate,
                "live_permission_allowed": False,
            }
        )
    return selections


def _truth_coverage_for_days(
    coverage: dict[str, Any],
    days: set[str],
) -> dict[str, Any]:
    forecast_by_day = coverage.get("forecast_rows_by_day") or {}
    complete_by_day = coverage.get("complete_truth_windows_by_day") or {}
    forecast_rows = sum(int(forecast_by_day.get(day) or 0) for day in days)
    complete_windows = sum(int(complete_by_day.get(day) or 0) for day in days)
    return {
        "days": sorted(days),
        "forecast_rows": forecast_rows,
        "complete_truth_windows": complete_windows,
        "coverage_rate": round(complete_windows / forecast_rows, 6)
        if forecast_rows
        else 0.0,
    }


def metrics(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "fills": 0,
            "mean_pips": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "active_days": 0,
            "positive_day_rate": 0.0,
            "max_daily_sample_share": 0.0,
            "one_sided_95_mean_lower_pips": None,
            "one_sided_95_daily_mean_lower_pips": None,
            "take_profit_rate": 0.0,
            "stop_loss_rate": 0.0,
            "ttl_close_rate": 0.0,
        }
    pnls = [float(item["realized_pips"]) for item in items]
    gross_profit = sum(value for value in pnls if value > 0.0)
    gross_loss = -sum(value for value in pnls if value < 0.0)
    by_day: dict[str, float] = collections.defaultdict(float)
    day_counts: collections.Counter[str] = collections.Counter()
    for item in items:
        day = str(
            item.get("resolved_day_utc")
            or str(item.get("timestamp_utc") or "")[:10]
            or item.get("campaign_day_jst")
            or ""
        )
        if not day:
            continue
        by_day[day] += float(item["realized_pips"])
        day_counts[day] += 1
    mean = statistics.mean(pnls)
    lower = None
    if len(pnls) >= 2:
        lower = mean - _one_sided_t95_critical(len(pnls) - 1) * statistics.stdev(
            pnls
        ) / math.sqrt(len(pnls))
    daily_pnls = list(by_day.values())
    daily_lower = None
    if len(daily_pnls) >= 2:
        daily_lower = statistics.mean(daily_pnls) - _one_sided_t95_critical(
            len(daily_pnls) - 1
        ) * statistics.stdev(daily_pnls) / math.sqrt(len(daily_pnls))
    return {
        "fills": len(items),
        "mean_pips": round(mean, 6),
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss > 0.0 else None,
        "win_rate": round(sum(value > 0.0 for value in pnls) / len(pnls), 6),
        "active_days": len(by_day),
        "positive_day_rate": round(
            sum(value > 0.0 for value in by_day.values()) / len(by_day),
            6,
        )
        if by_day
        else 0.0,
        "max_daily_sample_share": round(max(day_counts.values()) / len(items), 6)
        if day_counts
        else 0.0,
        "one_sided_95_mean_lower_pips": round(lower, 6) if lower is not None else None,
        "one_sided_95_daily_mean_lower_pips": (
            round(daily_lower, 6) if daily_lower is not None else None
        ),
        "take_profit_rate": round(sum(item["exit_reason"] == "TAKE_PROFIT" for item in items) / len(items), 6),
        "stop_loss_rate": round(sum(item["exit_reason"] == "STOP_LOSS" for item in items) / len(items), 6),
        "ttl_close_rate": round(sum(item["exit_reason"] == "TTL_CLOSE" for item in items) / len(items), 6),
    }


def _one_sided_t95_critical(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return math.inf
    if degrees_of_freedom < len(ONE_SIDED_T95_BY_DF):
        return ONE_SIDED_T95_BY_DF[degrees_of_freedom]
    z = statistics.NormalDist().inv_cdf(0.95)
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def _profit_factor_value(value: Any) -> float:
    """Treat a no-loss slice as unbounded PF without writing Infinity to JSON."""

    return math.inf if value is None else float(value)


def confidence_bucket(confidence: float) -> str:
    if confidence < 0.65:
        return "<0.65"
    if confidence < 0.75:
        return "0.65-0.75"
    if confidence < 0.90:
        return "0.75-0.90"
    return ">=0.90"


def range_width_bucket(width_pips: float) -> str:
    if width_pips < 10.0:
        return "<10"
    if width_pips < 20.0:
        return "10-20"
    if width_pips < 40.0:
        return "20-40"
    return ">=40"


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"oanda_range_scout_replay_validate_{run_ts}.json"
    markdown_path = output_dir / f"oanda_range_scout_replay_validate_{run_ts}.md"
    latest_json = output_dir / "oanda_range_scout_replay_validate_latest.json"
    latest_markdown = output_dir / "oanda_range_scout_replay_validate_latest.md"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    json_path.write_text(encoded, encoding="utf-8")
    latest_json.write_text(encoded, encoding="utf-8")
    markdown = markdown_report(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    latest_markdown.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": report.get("status"),
        "candidate_rules": len(report.get("candidate_rules") or []),
        "filled_signals": report.get("filled_signals"),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }, ensure_ascii=False, indent=2, sort_keys=True))


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# OANDA RANGE SCOUT Replay Validation",
        "",
        f"- Generated at UTC: `{report.get('generated_at_utc')}`",
        f"- Status: `{report.get('status')}`",
        f"- RANGE rows: raw={report.get('raw_range_rows')} deduped={report.get('deduped_range_rows')}",
        f"- Eligible signals: {report.get('eligible_signals')}",
        f"- Filled signals: {report.get('filled_signals')} (rate={report.get('fill_rate')})",
        f"- Candidate rules: {len(report.get('candidate_rules') or [])}",
        "- Live permission: `false`",
        f"- Adoption blockers: {', '.join(f'`{item}`' for item in report.get('adoption_blockers') or [])}",
        f"- Complete truth windows: {report.get('complete_truth_windows')} (coverage={report.get('truth_window_coverage_rate')})",
        f"- Ordering-ambiguous fill-bar target cases conservatively continued: {report.get('ambiguous_fill_bar_target_cases')}",
        "",
        "## Chronological Train / Validation",
        "",
        "| Pair | Side | TP | SL | Train PF | Validation PF | Validation mean | Candidate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("train_validation_selections") or []:
        train = item.get("train") or {}
        validation = item.get("validation") or {}
        lines.append(
            f"| {item.get('pair')} | {item.get('side')} | {item.get('take_profit_pips')} | "
            f"{item.get('stop_loss_pips')} | {train.get('profit_factor')} | "
            f"{validation.get('profit_factor')} | {validation.get('mean_pips')} | "
            f"{item.get('hypothesis_candidate')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Forecast persistence is deduplicated per pair for the full pending TTL.",
        "- Spread is included through executable bid/ask fills and exits.",
        "- Only candles whose full interval ends by forecast expiry are read.",
        "- Observation starts at the first complete S5 boundary at/after forecast emission; no pre-forecast OHLC range is consumed.",
        "- The best train grid is judged on the later chronological validation slice.",
        f"- Pair/side truth-window coverage must be at least {MIN_TRUTH_WINDOW_COVERAGE_RATE:.0%} before an OOS family is called powered.",
        "- Both trade-level and UTC resolved-day Student-t lower bounds must remain above zero for a research candidate.",
        "- Replay TTL_CLOSE after fill does not match current live SCOUT, whose GTD expires only unfilled entries; adoption is blocked until the exit contracts match.",
        "- A candidate is only a bounded forward hypothesis; it is not positive-expectancy proof or live permission.",
        "",
    ])
    return "\n".join(lines)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
