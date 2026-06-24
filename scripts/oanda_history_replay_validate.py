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
import gzip
import json
import math
import re
import statistics
import sys
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


def main() -> int:
    args = _parse_args()
    history_dirs = _history_dirs(
        args.history_dir,
        granularity=args.granularity,
        auto_min_days=args.auto_history_min_days,
    )
    pair_filter = _parse_pair_filter(args.pairs)
    rows, load_stats = _load_forecasts(args.forecast_history, pairs=pair_filter)
    candles_by_pair, candle_stats = _load_candles(
        history_dirs,
        granularity=args.granularity,
        windows_by_pair=_forecast_truth_windows(rows),
    )
    validation_now = datetime.now(timezone.utc)
    results, score_stats, unscorable_no_market_rows, pending_future_truth_rows = _score_forecasts(
        rows,
        candles_by_pair,
        now_utc=validation_now,
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

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = out_dir / f"oanda_history_replay_validate_{run_ts}.json"
    md_out = out_dir / f"oanda_history_replay_validate_{run_ts}.md"
    latest_json = out_dir / "oanda_history_replay_validate_latest.json"
    latest_md = out_dir / "oanda_history_replay_validate_latest.md"
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
        "exit_grid_config": {
            "take_profit_pips": list(args.tp_grid_pips),
            "stop_loss_pips": list(args.sl_grid_pips),
            "train_fraction": args.train_fraction,
            "min_train_samples": args.min_train_samples,
            "min_validation_samples": args.min_validation_samples,
        },
        **load_stats,
        **candle_stats,
        **score_stats,
        "summary": _summary(results),
        "price_truth_coverage": price_truth_coverage,
        "forecast_sample_coverage": sample_coverage,
        "segments": {
            "by_direction": _group(results, ("direction",)),
            "by_pair": _group(results, ("pair",), min_n=args.min_group_samples),
            "by_pair_direction": _group(results, ("pair", "direction"), min_n=args.min_group_samples),
            "by_horizon": _group(results, ("horizon_bucket",), min_n=args.min_group_samples),
            "by_confidence": _group(results, ("confidence_bucket",), min_n=args.min_group_samples),
        },
        "exit_grid": exit_grid,
        "segment_exit_grids": segment_exit_grids,
        "precision_rules": precision_rules,
        "train_validation_exit_selection": split,
    }
    json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_json.write_text(json_out.read_text(encoding="utf-8"), encoding="utf-8")
    md_out.write_text(_markdown(report), encoding="utf-8")
    latest_md.write_text(md_out.read_text(encoding="utf-8"), encoding="utf-8")
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
                    rows += 1
                    if windows_by_pair is not None and not _timestamp_in_windows(
                        candle.timestamp_utc,
                        windows_by_pair.get(candle.pair, []),
                        window_starts_by_pair.get(candle.pair, []),
                    ):
                        filtered += 1
                        continue
                    by_pair[candle.pair][candle.timestamp_utc] = candle
    sorted_by_pair = {
        pair: [items[key] for key in sorted(items)]
        for pair, items in by_pair.items()
    }
    return sorted_by_pair, {
        "history_files": files,
        "history_raw_rows": rows,
        "history_filtered_rows": filtered,
        "history_skipped_rows": skipped,
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
    pair = str(payload.get("pair") or "").upper()
    timestamp = _parse_time(payload.get("time"))
    bid = _ohlc_from_payload(payload.get("bid"))
    ask = _ohlc_from_payload(payload.get("ask"))
    if not pair or timestamp is None or bid is None or ask is None:
        return None
    return QuoteCandle(timestamp_utc=timestamp, pair=pair, bid=bid, ask=ask)


def _ohlc_from_payload(payload: object) -> Ohlc | None:
    if not isinstance(payload, dict):
        return None
    try:
        return Ohlc(
            o=float(payload["o"]),
            h=float(payload["h"]),
            l=float(payload["l"]),
            c=float(payload["c"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_forecasts(path: Path, *, pairs: set[str] | None = None) -> tuple[list[ForecastRow], dict[str, Any]]:
    pair_filter = {str(item).upper() for item in (pairs or set()) if str(item).strip()}
    rows: list[ForecastRow] = []
    seen: set[tuple[Any, ...]] = set()
    raw_directional = 0
    skipped_pair_filter = 0
    skipped_duplicate = 0
    skipped_invalid = 0
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
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
            if pair_filter and pair not in pair_filter:
                skipped_pair_filter += 1
                continue
            confidence = _safe_float(payload.get("confidence"))
            target = _safe_float(payload.get("target_price"))
            invalidation = _safe_float(payload.get("invalidation_price"))
            cycle_id = str(payload.get("cycle_id") or "").strip() or None
            key = _dedupe_key(
                cycle_id=cycle_id,
                pair=pair,
                timestamp=timestamp,
                direction=direction,
                confidence=confidence,
                target=target,
                invalidation=invalidation,
            )
            if key in seen:
                skipped_duplicate += 1
                continue
            seen.add(key)
            rows.append(
                ForecastRow(
                    source_index=idx,
                    timestamp_utc=timestamp,
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    current_price=_safe_float(payload.get("current_price")),
                    target_price=target,
                    invalidation_price=invalidation,
                    horizon_min=_safe_horizon(payload.get("horizon_min")),
                    cycle_id=cycle_id,
                )
            )
    return rows, {
        "raw_directional_rows": raw_directional,
        "pair_filter": sorted(pair_filter),
        "skipped_pair_filter_rows": skipped_pair_filter,
        "deduped_directional_rows": len(rows),
        "skipped_duplicate_rows": skipped_duplicate,
        "skipped_invalid_rows": skipped_invalid,
    }


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


def _score_forecasts(
    rows: Sequence[ForecastRow],
    candles_by_pair: dict[str, list[QuoteCandle]],
    *,
    now_utc: datetime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ForecastRow], list[ForecastRow]]:
    results: list[dict[str, Any]] = []
    skipped_no_pair = 0
    skipped_no_window = 0
    missing_windows: list[ForecastRow] = []
    unscorable_no_market: list[ForecastRow] = []
    pending_future_truth: list[ForecastRow] = []
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
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
        end = bisect.bisect_right(times, _forecast_truth_end(row))
        if start >= len(candles) or end <= start:
            skipped_no_window += 1
            missing_windows.append(row)
            continue
        window = candles[start:end]
        scored = _score_one(row, window)
        if scored is not None:
            results.append(scored)
    return (
        results,
        {
            "evaluated_rows": len(results),
            "skipped_no_pair_candles": skipped_no_pair,
            "skipped_no_price_window": skipped_no_window,
            "missing_price_window_groups": _missing_price_window_groups(missing_windows),
            "unscorable_no_market_rows": len(unscorable_no_market),
            "unscorable_no_market_window_groups": _missing_price_window_groups(unscorable_no_market),
            "pending_future_truth_rows": len(pending_future_truth),
            "pending_future_truth_window_groups": _missing_price_window_groups(pending_future_truth),
        },
        unscorable_no_market,
        pending_future_truth,
    )


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
    elif missing_truth_samples > 0 or missing_groups:
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
        missing_groups,
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
        "unscorable_no_market_window_group_count": len(no_market_groups),
        "unscorable_no_market_window_groups": no_market_groups[:12],
        "pending_future_truth_window_group_count": len(pending_future_groups),
        "pending_future_truth_window_groups": pending_future_groups[:12],
        "future_price_truth_window_group_count": _future_missing_window_group_count(
            missing_groups,
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
            missing_groups,
            granularity,
            now_utc=now,
        ),
    }


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


def _score_one(row: ForecastRow, window: Sequence[QuoteCandle]) -> dict[str, Any] | None:
    return _score_direction(row, window, direction=row.direction, forecast_direction=row.direction)


def _score_direction(
    row: ForecastRow,
    window: Sequence[QuoteCandle],
    *,
    direction: str,
    forecast_direction: str,
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
    return {
        "source_index": row.source_index,
        "timestamp_utc": _iso(row.timestamp_utc),
        "entry_timestamp_utc": _iso(first.timestamp_utc),
        "last_timestamp_utc": _iso(last.timestamp_utc),
        "pair": row.pair,
        "direction": direction,
        "forecast_direction": forecast_direction,
        "contrarian": direction != forecast_direction,
        "confidence": row.confidence,
        "confidence_bucket": _confidence_bucket(row.confidence),
        "horizon_min": row.horizon_min,
        "horizon_bucket": _horizon_bucket(row.horizon_min),
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
    train = ordered[:split_at]
    validation = ordered[split_at:]
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
    return {
        "n": len(rows),
        "hit_rate": _rate(r["final_direction_hit"] for r in rows),
        "avg_final_pips": _mean(r["final_pips"] for r in rows),
        "median_final_pips": _median(r["final_pips"] for r in rows),
        "avg_mfe_pips": _mean(r["mfe_pips"] for r in rows),
        "avg_mae_pips": _mean(r["mae_pips"] for r in rows),
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
    lines = [
        "# OANDA History Replay Validate",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        f"- source: {report['source']}",
        f"- history_dirs: {', '.join(report['history_dirs'])}",
        f"- truth_source: {report['truth_source']}",
        f"- rows: raw_directional={report['raw_directional_rows']} deduped={report['deduped_directional_rows']} evaluated={report['evaluated_rows']}",
        f"- history: files={report['history_files']} candles={report['history_candles']} skipped={report['history_skipped_rows']}",
        f"- price_truth_coverage: status={(report.get('price_truth_coverage') or {}).get('status')} adoption={(report.get('price_truth_coverage') or {}).get('adoption_level')}",
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
    "avg_final_pips",
    "median_final_pips",
    "avg_mfe_pips",
    "avg_mae_pips",
    "target_touch_rate",
    "invalidation_touch_rate",
    "target_before_invalidation_rate",
    "reward_side_target_rate",
    "adverse_side_invalidation_rate",
}


def _summary_line(summary: dict[str, Any]) -> str:
    return (
        f"n={summary.get('n', 0)} hit={_pct(summary.get('hit_rate'))} "
        f"avg_final={_fmt(summary.get('avg_final_pips'))} "
        f"median_final={_fmt(summary.get('median_final_pips'))} "
        f"avg_mfe={_fmt(summary.get('avg_mfe_pips'))} avg_mae={_fmt(summary.get('avg_mae_pips'))} "
        f"reward_target={_pct(summary.get('reward_side_target_rate'))} "
        f"adverse_inv={_pct(summary.get('adverse_side_invalidation_rate'))} "
        f"target={_pct(summary.get('target_touch_rate'))} invalidation={_pct(summary.get('invalidation_touch_rate'))} "
        f"target_first={_pct(summary.get('target_before_invalidation_rate'))}"
    )


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
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


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
