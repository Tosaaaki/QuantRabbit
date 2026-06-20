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
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from quant_rabbit.instruments import instrument_pip_factor


DIRECTIONAL = {"UP", "DOWN"}
DEFAULT_TP_GRID_PIPS = (2.0, 5.0, 10.0)
DEFAULT_SL_GRID_PIPS = (2.0, 4.0, 7.0)


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
    history_dirs = _history_dirs(args.history_dir)
    candles_by_pair, candle_stats = _load_candles(history_dirs, granularity=args.granularity)
    rows, load_stats = _load_forecasts(args.forecast_history)
    results, score_stats = _score_forecasts(rows, candles_by_pair)
    exit_grid = _exit_grid(results, tp_grid=args.tp_grid_pips, sl_grid=args.sl_grid_pips)
    segment_exit_grids = {
        "by_pair_direction": _segment_exit_grids(
            results,
            ("pair", "direction"),
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
        "segments": {
            "by_direction": _group(results, ("direction",)),
            "by_pair": _group(results, ("pair",), min_n=args.min_group_samples),
            "by_pair_direction": _group(results, ("pair", "direction"), min_n=args.min_group_samples),
            "by_horizon": _group(results, ("horizon_bucket",), min_n=args.min_group_samples),
            "by_confidence": _group(results, ("confidence_bucket",), min_n=args.min_group_samples),
        },
        "exit_grid": exit_grid,
        "segment_exit_grids": segment_exit_grids,
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
    parser.add_argument("--history-dir", type=Path, action="append")
    parser.add_argument("--granularity", default="S5")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/reports/forecast_improvement"))
    parser.add_argument("--tp-grid-pips", type=_parse_float_csv, default=DEFAULT_TP_GRID_PIPS)
    parser.add_argument("--sl-grid-pips", type=_parse_float_csv, default=DEFAULT_SL_GRID_PIPS)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--min-train-samples", type=int, default=20)
    parser.add_argument("--min-validation-samples", type=int, default=10)
    parser.add_argument("--min-group-samples", type=int, default=5)
    return parser.parse_args()


def _history_dirs(explicit: Sequence[Path] | None) -> list[Path]:
    if explicit:
        return list(explicit)
    latest = Path("logs/replay/oanda_history/latest_summary.json")
    if not latest.exists():
        raise FileNotFoundError("missing logs/replay/oanda_history/latest_summary.json; pass --history-dir")
    payload = json.loads(latest.read_text(encoding="utf-8"))
    output_dir = payload.get("output_dir")
    if not output_dir:
        raise RuntimeError("latest_summary.json has no output_dir; pass --history-dir")
    return [Path(str(output_dir))]


def _parse_float_csv(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(float(item) for item in value)
    out = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not out:
        raise argparse.ArgumentTypeError("grid must contain at least one numeric pip value")
    return out


def _load_candles(history_dirs: Sequence[Path], *, granularity: str) -> tuple[dict[str, list[QuoteCandle]], dict[str, Any]]:
    by_pair: dict[str, dict[datetime, QuoteCandle]] = collections.defaultdict(dict)
    files = 0
    rows = 0
    skipped = 0
    for history_dir in history_dirs:
        for path in sorted(history_dir.glob(f"*/*_{granularity}_BA_*.jsonl")):
            files += 1
            with path.open(encoding="utf-8") as handle:
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
                    by_pair[candle.pair][candle.timestamp_utc] = candle
                    rows += 1
    sorted_by_pair = {
        pair: [items[key] for key in sorted(items)]
        for pair, items in by_pair.items()
    }
    return sorted_by_pair, {
        "history_files": files,
        "history_raw_rows": rows,
        "history_skipped_rows": skipped,
        "history_pairs": len(sorted_by_pair),
        "history_candles": sum(len(items) for items in sorted_by_pair.values()),
    }


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


def _load_forecasts(path: Path) -> tuple[list[ForecastRow], dict[str, Any]]:
    rows: list[ForecastRow] = []
    seen: set[tuple[Any, ...]] = set()
    raw_directional = 0
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    skipped_no_pair = 0
    skipped_no_window = 0
    for row in rows:
        candles = candles_by_pair.get(row.pair)
        if not candles:
            skipped_no_pair += 1
            continue
        times = [c.timestamp_utc for c in candles]
        start = bisect.bisect_left(times, row.timestamp_utc)
        end = bisect.bisect_right(times, row.timestamp_utc + timedelta(minutes=row.horizon_min))
        if start >= len(candles) or end <= start:
            skipped_no_window += 1
            continue
        window = candles[start:end]
        scored = _score_one(row, window)
        if scored is not None:
            results.append(scored)
    return results, {
        "evaluated_rows": len(results),
        "skipped_no_pair_candles": skipped_no_pair,
        "skipped_no_price_window": skipped_no_window,
    }


def _score_one(row: ForecastRow, window: Sequence[QuoteCandle]) -> dict[str, Any] | None:
    if not window:
        return None
    pip_factor = instrument_pip_factor(row.pair)
    first = window[0]
    last = window[-1]
    if row.direction == "UP":
        entry = first.ask.o
        final_pips = (last.bid.c - entry) * pip_factor
        mfe_pips = max(0.0, (max(c.bid.h for c in window) - entry) * pip_factor)
        mae_pips = max(0.0, (entry - min(c.bid.l for c in window)) * pip_factor)
    else:
        entry = first.bid.o
        final_pips = (entry - last.ask.c) * pip_factor
        mfe_pips = max(0.0, (entry - min(c.ask.l for c in window)) * pip_factor)
        mae_pips = max(0.0, (max(c.ask.h for c in window) - entry) * pip_factor)
    target_reward_side = _target_is_reward_side(row, entry)
    invalidation_adverse_side = _invalidation_is_adverse_side(row, entry)
    target_touch, invalidation_touch, target_first = _target_invalidation_order(
        row,
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
        "direction": row.direction,
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
        payload = {field: value for field, value in zip(fields, key)}
        payload.update(
            {
                "n": len(items),
                "summary": _summary(items),
                "best_exit": grid[0] if grid else None,
                "exit_grid": grid,
            }
        )
        out.append(payload)
    out.sort(
        key=lambda item: (
            float((item.get("best_exit") or {}).get("avg_realized_pips") or -999.0),
            int(item.get("n") or 0),
        ),
        reverse=True,
    )
    return out


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
        "",
        "## Summary",
        "",
        _summary_line(summary),
        "",
        "## Train/Validation Exit Selection",
        "",
    ]
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
