#!/usr/bin/env python3
"""Validate forecast_history UP/DOWN rows against historical M1 candles.

This script is deliberately read-only against broker state. It re-fetches OANDA
M1 mid candles for each forecast window, scores the direction from forecast
time to horizon, and writes an auditable JSON/Markdown report under
logs/reports/forecast_improvement.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_rabbit.analysis.candles import Candle, fetch_candles_between
from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.instruments import instrument_pip_factor


DIRECTIONAL = {"UP", "DOWN"}


@dataclass(frozen=True)
class ForecastRow:
    source_index: int
    timestamp: datetime
    pair: str
    direction: str
    confidence: float | None
    entry: float | None
    target: float | None
    invalidation: float | None
    horizon_min: float
    cycle_id: str | None


def main() -> int:
    args = _parse_args()
    data_root = args.data_root
    source = data_root / "forecast_history.jsonl"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = out_dir / f"forecast_direction_candle_truth_{run_ts}.json"
    md_out = out_dir / f"forecast_direction_candle_truth_{run_ts}.md"
    latest_json = out_dir / "forecast_direction_candle_truth_latest.json"
    latest_md = out_dir / "forecast_direction_candle_truth_latest.md"

    rows, load_stats = _load_forecasts(source)
    candles_by_pair, fetch_stats = _fetch_truth(rows, max_chunk_days=args.max_chunk_days)
    results, score_stats = _score_rows(rows, candles_by_pair)

    wide_target = [
        item
        for item in results
        if (item.get("target_pips") or 0.0) >= args.min_target_pips
    ]
    segments = {
        "by_direction": _group(results, ("direction",)),
        "by_horizon": _group(results, ("horizon_bucket",)),
        "by_confidence": _group(results, ("confidence_bucket",)),
        "by_entry_source": _group(results, ("entry_source",)),
        "by_pair_direction": _group(results, ("pair", "direction"), min_n=5),
        "by_pair_direction_confidence": _group(
            results,
            ("pair", "direction", "confidence_bucket"),
            min_n=5,
        ),
        "by_pair_direction_horizon": _group(
            results,
            ("pair", "direction", "horizon_bucket"),
            min_n=5,
        ),
        "wide_by_pair_direction": _group(wide_target, ("pair", "direction"), min_n=5),
    }
    strict = _strict_candidates(
        wide_target,
        min_samples=args.min_samples,
        min_wilson_lower=args.min_wilson_lower,
    )
    exploratory = _exploratory_candidates(wide_target)
    report = {
        "generated_at_utc": _iso(datetime.now(timezone.utc)),
        "source": str(source),
        "truth_source": (
            "OANDA M1 mid candles via read-only API; first partial forecast "
            "minute excluded; missing current_price uses next full M1 open "
            "proxy; same-M1 target+invalidation ambiguity counted as miss"
        ),
        **load_stats,
        **fetch_stats,
        **score_stats,
        "summary": _summarize(results),
        "wide_target_summary": _summarize(wide_target),
        "strict_90_wilson_candidates": strict,
        "exploratory_top_buckets": exploratory[:50],
        "segments": segments,
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
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/reports/forecast_improvement"),
    )
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-wilson-lower", type=float, default=0.90)
    parser.add_argument("--min-target-pips", type=float, default=2.0)
    parser.add_argument("--max-chunk-days", type=float, default=3.0)
    return parser.parse_args()


def _load_forecasts(path: Path) -> tuple[list[ForecastRow], dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    rows: list[ForecastRow] = []
    seen: set[tuple[Any, ...]] = set()
    raw_directional = 0
    skipped_duplicate = 0
    skipped_invalid = 0
    entry_present = 0
    entry_missing = 0
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
            horizon = _safe_horizon(payload.get("horizon_min"))
            if timestamp + timedelta(minutes=horizon) > now_utc - timedelta(minutes=2):
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
            entry = _safe_float(payload.get("current_price"))
            if entry is not None and entry > 0.0:
                entry_present += 1
            else:
                entry = None
                entry_missing += 1
            rows.append(
                ForecastRow(
                    source_index=idx,
                    timestamp=timestamp,
                    pair=pair,
                    direction=direction,
                    confidence=confidence,
                    entry=entry,
                    target=target,
                    invalidation=invalidation,
                    horizon_min=horizon,
                    cycle_id=cycle_id,
                )
            )
    return rows, {
        "raw_directional_rows": raw_directional,
        "deduped_directional_rows": len(rows),
        "entry_price_present_rows": entry_present,
        "entry_price_proxy_needed_rows": entry_missing,
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
        "row",
        pair,
        timestamp.replace(microsecond=0).isoformat(),
        direction,
        round(confidence or 0.0, 6),
        round(target or 0.0, 5),
        round(invalidation or 0.0, 5),
    )


def _fetch_truth(
    rows: list[ForecastRow],
    *,
    max_chunk_days: float,
) -> tuple[dict[str, list[Candle]], dict[str, Any]]:
    windows = _merged_fetch_windows(rows)
    client = OandaReadOnlyClient()
    candles: dict[str, list[Candle]] = collections.defaultdict(list)
    fetch_errors: list[dict[str, str]] = []
    for pair in sorted(windows):
        for start, end in windows[pair]:
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(end, chunk_start + timedelta(days=max_chunk_days))
                for attempt in range(3):
                    try:
                        candles[pair].extend(
                            fetch_candles_between(
                                pair,
                                "M1",
                                time_from=chunk_start,
                                time_to=chunk_end,
                                client=client,
                            )
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - report evidence plumbing errors.
                        if attempt == 2:
                            fetch_errors.append(
                                {
                                    "pair": pair,
                                    "from": _iso(chunk_start),
                                    "to": _iso(chunk_end),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                        else:
                            time.sleep(0.8 * (attempt + 1))
                chunk_start = chunk_end
    deduped: dict[str, list[Candle]] = {}
    for pair, values in candles.items():
        by_ts = {c.timestamp_utc.astimezone(timezone.utc): c for c in values}
        deduped[pair] = [by_ts[k] for k in sorted(by_ts)]
    return deduped, {
        "fetch_pairs": len(windows),
        "fetch_windows": sum(len(items) for items in windows.values()),
        "fetched_candles": sum(len(items) for items in deduped.values()),
        "fetch_errors": fetch_errors[:50],
    }


def _merged_fetch_windows(rows: list[ForecastRow]) -> dict[str, list[tuple[datetime, datetime]]]:
    intervals: dict[str, list[tuple[datetime, datetime]]] = collections.defaultdict(list)
    for row in rows:
        start = _ceil_minute(row.timestamp) - timedelta(minutes=1)
        end = row.timestamp + timedelta(minutes=row.horizon_min) + timedelta(minutes=1)
        intervals[row.pair].append((start, end))
    merged: dict[str, list[tuple[datetime, datetime]]] = {}
    for pair, values in intervals.items():
        values.sort()
        out: list[tuple[datetime, datetime]] = []
        for start, end in values:
            if not out or start > out[-1][1] + timedelta(minutes=2):
                out.append((start, end))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], end))
        merged[pair] = out
    return merged


def _score_rows(
    rows: list[ForecastRow],
    candles_by_pair: dict[str, list[Candle]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    skipped_no_candles = 0
    proxy_scored = 0
    for row in rows:
        start = _ceil_minute(row.timestamp)
        end = row.timestamp + timedelta(minutes=row.horizon_min)
        candles = _candle_window(candles_by_pair, row.pair, start, end)
        if not candles:
            skipped_no_candles += 1
            continue
        entry = row.entry
        entry_source = "forecast_current_price"
        if entry is None or entry <= 0.0:
            entry = candles[0].open
            entry_source = "next_full_m1_open_proxy"
            proxy_scored += 1
        results.append(_score_one(row, candles, entry=entry, entry_source=entry_source))
    return results, {
        "scored_rows": len(results),
        "entry_price_proxy_scored_rows": proxy_scored,
        "skipped_no_candles_rows": skipped_no_candles,
    }


def _score_one(
    row: ForecastRow,
    candles: list[Candle],
    *,
    entry: float,
    entry_source: str,
) -> dict[str, Any]:
    factor = instrument_pip_factor(row.pair)
    final_close = candles[-1].close
    if row.direction == "UP":
        final_pips = (final_close - entry) * factor
        mfe_pips = (max(c.high for c in candles) - entry) * factor
        mae_pips = max(0.0, (entry - min(c.low for c in candles)) * factor)
        final_hit = final_close > entry
    else:
        final_pips = (entry - final_close) * factor
        mfe_pips = (entry - min(c.low for c in candles)) * factor
        mae_pips = max(0.0, (max(c.high for c in candles) - entry) * factor)
        final_hit = final_close < entry
    first_touch, target_first, touch_evidence = _target_first_touch(row, entry, candles)
    target_touch = _target_touch(row, entry, candles)
    return {
        "source_index": row.source_index,
        "timestamp_utc": _iso(row.timestamp),
        "truth_start_utc": _iso(_ceil_minute(row.timestamp)),
        "truth_end_utc": _iso(row.timestamp + timedelta(minutes=row.horizon_min)),
        "pair": row.pair,
        "direction": row.direction,
        "confidence": row.confidence,
        "confidence_bucket": _confidence_bucket(row.confidence),
        "entry_price": entry,
        "entry_source": entry_source,
        "target_price": row.target,
        "invalidation_price": row.invalidation,
        "target_pips": abs(row.target - entry) * factor if row.target is not None else None,
        "invalidation_pips": (
            abs(row.invalidation - entry) * factor if row.invalidation is not None else None
        ),
        "horizon_min": row.horizon_min,
        "horizon_bucket": _horizon_bucket(row.horizon_min),
        "final_close": final_close,
        "final_pips": final_pips,
        "final_direction_hit": bool(final_hit),
        "mfe_pips": mfe_pips,
        "mae_pips": mae_pips,
        "mfe_ge_1pip": bool(mfe_pips >= 1.0),
        "mfe_ge_2pip": bool(mfe_pips >= 2.0),
        "mfe_ge_5pip": bool(mfe_pips >= 5.0),
        "target_touch_hit": target_touch,
        "first_touch": first_touch,
        "target_before_invalidation_hit": target_first,
        "touch_evidence": touch_evidence,
        "cycle_id": row.cycle_id,
    }


def _target_touch(row: ForecastRow, entry: float, candles: list[Candle]) -> bool | None:
    if row.target is None:
        return None
    if row.direction == "UP":
        if row.target <= entry:
            return None
        return any(c.high >= row.target for c in candles)
    if row.target >= entry:
        return None
    return any(c.low <= row.target for c in candles)


def _target_first_touch(
    row: ForecastRow,
    entry: float,
    candles: list[Candle],
) -> tuple[str | None, bool | None, str]:
    if row.target is None or row.invalidation is None:
        return None, None, "missing_or_invalid_geometry"
    if row.direction == "UP":
        if row.target <= entry or row.invalidation >= entry:
            return None, None, "missing_or_invalid_geometry"
    elif row.target >= entry or row.invalidation <= entry:
        return None, None, "missing_or_invalid_geometry"
    for candle in candles:
        if row.direction == "UP":
            target_hit = candle.high >= row.target
            invalid_hit = candle.low <= row.invalidation
        else:
            target_hit = candle.low <= row.target
            invalid_hit = candle.high >= row.invalidation
        if target_hit and invalid_hit:
            return (
                "AMBIGUOUS_SAME_M1",
                False,
                f"{_iso(candle.timestamp_utc)} target and invalidation both inside same M1 candle",
            )
        if target_hit:
            return "TARGET_FIRST", True, f"{_iso(candle.timestamp_utc)} target touched first"
        if invalid_hit:
            return "INVALIDATION_FIRST", False, f"{_iso(candle.timestamp_utc)} invalidation touched first"
    return "TIMEOUT", False, "neither target nor invalidation touched inside horizon"


def _candle_window(
    candles_by_pair: dict[str, list[Candle]],
    pair: str,
    start: datetime,
    end: datetime,
) -> list[Candle]:
    return [
        candle
        for candle in candles_by_pair.get(pair, [])
        if start <= candle.timestamp_utc.astimezone(timezone.utc) <= end and candle.complete
    ]


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    final_hits = sum(1 for item in items if item["final_direction_hit"])
    mfe1 = sum(1 for item in items if item["mfe_ge_1pip"])
    mfe2 = sum(1 for item in items if item["mfe_ge_2pip"])
    mfe5 = sum(1 for item in items if item["mfe_ge_5pip"])
    target_items = [item for item in items if item["target_touch_hit"] is not None]
    target_hits = sum(1 for item in target_items if item["target_touch_hit"])
    first_items = [item for item in items if item["target_before_invalidation_hit"] is not None]
    first_hits = sum(1 for item in first_items if item["target_before_invalidation_hit"])
    final_pips = [float(item["final_pips"]) for item in items]
    mfe_pips = [float(item["mfe_pips"]) for item in items]
    mae_pips = [float(item["mae_pips"]) for item in items]
    return {
        "n": n,
        "final_hits": final_hits,
        "final_hit_rate": final_hits / n if n else None,
        "final_wilson95_lower": _wilson_lower(final_hits, n),
        "mfe_ge_1pip_hits": mfe1,
        "mfe_ge_1pip_rate": mfe1 / n if n else None,
        "mfe_ge_2pip_hits": mfe2,
        "mfe_ge_2pip_rate": mfe2 / n if n else None,
        "mfe_ge_5pip_hits": mfe5,
        "mfe_ge_5pip_rate": mfe5 / n if n else None,
        "target_touch_n": len(target_items),
        "target_touch_hits": target_hits,
        "target_touch_rate": target_hits / len(target_items) if target_items else None,
        "target_touch_wilson95_lower": _wilson_lower(target_hits, len(target_items)),
        "touch_n": len(first_items),
        "target_first_hits": first_hits,
        "target_first_rate": first_hits / len(first_items) if first_items else None,
        "target_first_wilson95_lower": _wilson_lower(first_hits, len(first_items)),
        "avg_final_pips": statistics.fmean(final_pips) if final_pips else None,
        "median_final_pips": statistics.median(final_pips) if final_pips else None,
        "median_mfe_pips": statistics.median(mfe_pips) if mfe_pips else None,
        "median_mae_pips": statistics.median(mae_pips) if mae_pips else None,
    }


def _group(
    items: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    min_n: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        buckets[tuple(item.get(field) for field in fields)].append(item)
    out: list[dict[str, Any]] = []
    for key, values in buckets.items():
        if len(values) < min_n:
            continue
        row = {field: key[index] for index, field in enumerate(fields)}
        row.update(_summarize(values))
        out.append(row)
    out.sort(
        key=lambda row: (
            -(row.get("target_first_wilson95_lower") or 0.0),
            -(row.get("target_touch_wilson95_lower") or 0.0),
            -(row.get("final_wilson95_lower") or 0.0),
            -row["n"],
        )
    )
    return out


def _strict_candidates(
    items: list[dict[str, Any]],
    *,
    min_samples: int,
    min_wilson_lower: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_name, fields in {
        "pair_direction": ("pair", "direction"),
        "pair_direction_confidence": ("pair", "direction", "confidence_bucket"),
    }.items():
        for row in _group(items, fields, min_n=min_samples):
            if (
                row.get("target_first_wilson95_lower") is not None
                and row["target_first_wilson95_lower"] >= min_wilson_lower
            ):
                item = {"group": group_name}
                item.update(row)
                out.append(item)
    return out


def _exploratory_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_name, fields in {
        "pair_direction": ("pair", "direction"),
        "pair_direction_confidence": ("pair", "direction", "confidence_bucket"),
    }.items():
        for row in _group(items, fields, min_n=10):
            item = {"group": group_name}
            item.update(row)
            out.append(item)
    out.sort(
        key=lambda row: (
            -(row.get("target_first_wilson95_lower") or 0.0),
            -(row.get("target_touch_wilson95_lower") or 0.0),
            -row["n"],
        )
    )
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Forecast Direction Candle Truth",
        "",
        f"- generated_at_utc: {report['generated_at_utc']}",
        f"- source: `{report['source']}`",
        f"- truth: {report['truth_source']}",
        f"- raw_directional_rows: {report['raw_directional_rows']}",
        f"- deduped_directional_rows: {report['deduped_directional_rows']}",
        (
            "- entry_price_present/proxy_needed/proxy_scored: "
            f"{report['entry_price_present_rows']} / "
            f"{report['entry_price_proxy_needed_rows']} / "
            f"{report['entry_price_proxy_scored_rows']}"
        ),
        f"- scored_rows: {report['scored_rows']}",
        f"- fetch_pairs/windows/candles: {report['fetch_pairs']} / {report['fetch_windows']} / {report['fetched_candles']}",
        f"- fetch_errors: {len(report['fetch_errors'])}",
        "",
        "## Overall",
        "",
    ]
    lines.extend(
        _table(
            [report["summary"]],
            [
                ("n", "n"),
                ("final hit", "final_hit_rate"),
                ("final Wilson95L", "final_wilson95_lower"),
                ("MFE>=2pip", "mfe_ge_2pip_rate"),
                ("target touch n", "target_touch_n"),
                ("target touch", "target_touch_rate"),
                ("first-touch n", "touch_n"),
                ("target-first", "target_first_rate"),
                ("target-first Wilson95L", "target_first_wilson95_lower"),
            ],
        )
    )
    lines.extend(["", "## Strict 90% Wilson Candidates", ""])
    strict = report["strict_90_wilson_candidates"]
    if strict:
        lines.extend(
            _table(
                strict,
                [
                    ("group", "group"),
                    ("pair", "pair"),
                    ("dir", "direction"),
                    ("conf", "confidence_bucket"),
                    ("n", "n"),
                    ("first-touch n", "touch_n"),
                    ("target-first", "target_first_rate"),
                    ("target-first Wilson95L", "target_first_wilson95_lower"),
                    ("final hit", "final_hit_rate"),
                ],
                limit=20,
            )
        )
    else:
        lines.append("None.")
    lines.extend(["", "## By Confidence", ""])
    lines.extend(
        _table(
            report["segments"]["by_confidence"],
            [
                ("conf", "confidence_bucket"),
                ("n", "n"),
                ("final hit", "final_hit_rate"),
                ("final Wilson95L", "final_wilson95_lower"),
                ("MFE>=2pip", "mfe_ge_2pip_rate"),
                ("target-first", "target_first_rate"),
                ("target-first Wilson95L", "target_first_wilson95_lower"),
            ],
        )
    )
    lines.extend(["", "## By Horizon", ""])
    lines.extend(
        _table(
            report["segments"]["by_horizon"],
            [
                ("horizon", "horizon_bucket"),
                ("n", "n"),
                ("final hit", "final_hit_rate"),
                ("final Wilson95L", "final_wilson95_lower"),
                ("MFE>=2pip", "mfe_ge_2pip_rate"),
                ("target-first", "target_first_rate"),
                ("target-first Wilson95L", "target_first_wilson95_lower"),
                ("avg pips", "avg_final_pips"),
            ],
        )
    )
    lines.extend(["", "## By Pair Direction", ""])
    lines.extend(
        _table(
            report["segments"]["by_pair_direction"],
            [
                ("pair", "pair"),
                ("dir", "direction"),
                ("n", "n"),
                ("final hit", "final_hit_rate"),
                ("final Wilson95L", "final_wilson95_lower"),
                ("target-first", "target_first_rate"),
                ("target-first Wilson95L", "target_first_wilson95_lower"),
                ("avg pips", "avg_final_pips"),
            ],
            limit=24,
        )
    )
    return "\n".join(lines) + "\n"


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]], limit: int = 12) -> list[str]:
    lines = [
        "| " + " | ".join(header for header, _ in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows[:limit]:
        values: list[str] = []
        for _, key in cols:
            value = row.get(key)
            if isinstance(value, float):
                values.append(_pct(value) if "rate" in key or "wilson" in key else f"{value:.2f}")
            elif value is None:
                values.append("n/a")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_horizon(value: Any) -> float:
    parsed = _safe_float(value)
    if parsed is None or parsed <= 0:
        return 60.0
    return min(parsed, 1440.0)


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _ceil_minute(value: datetime) -> datetime:
    value = value.astimezone(timezone.utc)
    base = value.replace(second=0, microsecond=0)
    return base if value == base else base + timedelta(minutes=1)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _wilson_lower(successes: int, trials: int, z: float = 1.96) -> float | None:
    if trials <= 0:
        return None
    p_hat = successes / trials
    denom = 1.0 + z * z / trials
    centre = p_hat + z * z / (2 * trials)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


def _confidence_bucket(confidence: float | None) -> str:
    if confidence is None:
        return "missing"
    if confidence >= 0.90:
        return ">=0.90"
    if confidence >= 0.75:
        return "0.75-0.90"
    if confidence >= 0.55:
        return "0.55-0.75"
    return "<0.55"


def _horizon_bucket(minutes: float) -> str:
    if minutes <= 15:
        return "<=15m"
    if minutes <= 60:
        return "<=60m"
    if minutes <= 240:
        return "<=4h"
    return ">4h"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
