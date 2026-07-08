#!/usr/bin/env python3
"""Fetch large read-only OANDA candle datasets for replay/mining.

The output is JSONL under ``logs/replay/oanda_history``. This script never
places orders; it only calls the OANDA instruments/candles read endpoint.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS_ARG


GRANULARITY_SECONDS: dict[str, int] = {
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
}

DEFAULT_OUTPUT_DIR = Path("logs/replay/oanda_history")


@dataclass(frozen=True)
class FetchTask:
    pair: str
    granularity: str
    start: datetime
    end: datetime
    price: str


def main() -> int:
    args = _parse_args()
    pairs = _parse_csv(args.pairs)
    granularities = _parse_csv(args.granularities)
    _validate_granularities(granularities)
    start, end = _resolve_window(args)
    client = OandaReadOnlyClient()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at_utc": _iso(datetime.now(timezone.utc)),
        "window": {"from": _iso(start), "to": _iso(end)},
        "pairs": pairs,
        "granularities": granularities,
        "price": args.price,
        "max_candles_per_request": args.max_candles_per_request,
        "output_dir": str(run_dir),
        "tasks": [],
        "errors": [],
        "total_rows": 0,
        "total_requests": 0,
        "dry_run": bool(args.dry_run),
    }

    for pair in pairs:
        for granularity in granularities:
            task = FetchTask(pair=pair, granularity=granularity, start=start, end=end, price=args.price)
            task_summary = _fetch_task(
                client,
                task,
                run_dir=run_dir,
                max_candles_per_request=args.max_candles_per_request,
                sleep_seconds=args.sleep_seconds,
                retries=args.retries,
                include_incomplete=args.include_incomplete,
                compress=args.compress,
                dry_run=args.dry_run,
            )
            summary["tasks"].append(task_summary)
            summary["total_rows"] += int(task_summary.get("rows", 0))
            summary["total_requests"] += int(task_summary.get("requests", 0))
            summary["errors"].extend(task_summary.get("errors", []))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not args.dry_run and not summary["errors"]:
        latest_path = args.output_dir / "latest_summary.json"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {summary_path}")
    print(f"rows={summary['total_rows']} requests={summary['total_requests']} errors={len(summary['errors'])}")
    return 1 if summary["errors"] else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=DEFAULT_TRADER_PAIRS_ARG)
    parser.add_argument("--granularities", default="M1")
    parser.add_argument("--price", default="BA", help="OANDA price selector: M, B, A, BA, MBA")
    parser.add_argument("--from", dest="time_from")
    parser.add_argument("--to", dest="time_to")
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-candles-per-request", type=int, default=4500)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--include-incomplete", action="store_true")
    compression = parser.add_mutually_exclusive_group()
    compression.add_argument(
        "--compress",
        dest="compress",
        action="store_true",
        default=True,
        help="write published history files as .jsonl.gz (default)",
    )
    compression.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="write published history files as plain .jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    end = _parse_time(args.time_to) if args.time_to else datetime.now(timezone.utc)
    if args.time_from:
        start = _parse_time(args.time_from)
    else:
        start = end - timedelta(days=float(args.days))
    if start >= end:
        raise ValueError("--from must be earlier than --to")
    return start, end


def _parse_csv(value: str) -> list[str]:
    items = [part.strip().upper() for part in str(value or "").split(",") if part.strip()]
    if not items:
        raise ValueError("comma-separated argument produced no values")
    return items


def _validate_granularities(granularities: Iterable[str]) -> None:
    unsupported = sorted(set(granularities) - set(GRANULARITY_SECONDS))
    if unsupported:
        raise ValueError(f"unsupported granularity {unsupported}; expected one of {sorted(GRANULARITY_SECONDS)}")


def _parse_time(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_task(
    client: OandaReadOnlyClient,
    task: FetchTask,
    *,
    run_dir: Path,
    max_candles_per_request: int,
    sleep_seconds: float,
    retries: int,
    include_incomplete: bool,
    compress: bool = True,
    dry_run: bool,
) -> dict[str, Any]:
    file_dir = run_dir / task.pair
    file_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"{task.pair}_{task.granularity}_{task.price}_"
        f"{_stamp(task.start)}_{_stamp(task.end)}.jsonl"
    )
    if compress:
        file_name += ".gz"
    file_path = file_dir / file_name
    windows = list(
        _iter_time_chunks(
            task.start,
            task.end,
            granularity=task.granularity,
            max_candles_per_request=max_candles_per_request,
        )
    )
    summary: dict[str, Any] = {
        "pair": task.pair,
        "granularity": task.granularity,
        "price": task.price,
        "from": _iso(task.start),
        "to": _iso(task.end),
        "path": str(file_path),
        "partial_path": None,
        "published": False,
        "windows": len(windows),
        "requests": 0,
        "rows": 0,
        "errors": [],
        "dry_run": dry_run,
        "compressed": bool(compress),
    }
    if dry_run:
        return summary

    seen_times: set[str] = set()
    tmp_path = file_path.with_name(f"{file_path.name}.tmp")
    partial_path = file_path.with_name(f"{file_path.name}.partial")
    tmp_path.unlink(missing_ok=True)
    partial_path.unlink(missing_ok=True)
    with _open_jsonl_writer(tmp_path, compress=compress) as handle:
        for window_start, window_end in windows:
            payload = _get_candles_with_retry(
                client,
                task,
                window_start=window_start,
                window_end=window_end,
                retries=retries,
            )
            summary["requests"] += 1
            if payload is None:
                summary["errors"].append(
                    {
                        "pair": task.pair,
                        "granularity": task.granularity,
                        "from": _iso(window_start),
                        "to": _iso(window_end),
                        "error": "request_failed_after_retries",
                    }
                )
                continue
            for row in _rows_from_payload(
                payload,
                pair=task.pair,
                granularity=task.granularity,
                price=task.price,
                include_incomplete=include_incomplete,
            ):
                ts = str(row.get("time") or "")
                if ts in seen_times:
                    continue
                seen_times.add(ts)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                summary["rows"] += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    if summary["errors"]:
        tmp_path.replace(partial_path)
        summary["partial_path"] = str(partial_path)
    else:
        tmp_path.replace(file_path)
        summary["published"] = True
    return summary


def _open_jsonl_writer(path: Path, *, compress: bool):
    if compress:
        return gzip.open(path, mode="wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _get_candles_with_retry(
    client: OandaReadOnlyClient,
    task: FetchTask,
    *,
    window_start: datetime,
    window_end: datetime,
    retries: int,
) -> dict[str, Any] | None:
    query = {
        "granularity": task.granularity,
        "from": _iso(window_start),
        "to": _iso(window_end),
        "price": task.price,
        "includeFirst": "true",
    }
    for attempt in range(max(1, retries)):
        try:
            return client.get_json(f"/v3/instruments/{task.pair}/candles", query)
        except Exception:  # noqa: BLE001 - evidence fetch reports after retries.
            if attempt + 1 >= max(1, retries):
                return None
            time.sleep(0.8 * (attempt + 1))
    return None


def _iter_time_chunks(
    start: datetime,
    end: datetime,
    *,
    granularity: str,
    max_candles_per_request: int,
) -> Iterable[tuple[datetime, datetime]]:
    if max_candles_per_request <= 0:
        raise ValueError("max_candles_per_request must be positive")
    seconds = GRANULARITY_SECONDS[granularity]
    # Keep a one-candle cushion below the OANDA page cap so boundary rounding
    # cannot turn a request into an overfilled page.
    chunk_seconds = max(seconds, seconds * max(1, max_candles_per_request - 1))
    cur = start
    while cur < end:
        nxt = min(end, cur + timedelta(seconds=chunk_seconds))
        yield cur, nxt
        cur = nxt


def _rows_from_payload(
    payload: dict[str, Any],
    *,
    pair: str,
    granularity: str,
    price: str,
    include_incomplete: bool,
) -> Iterable[dict[str, Any]]:
    for candle in payload.get("candles") or []:
        if not isinstance(candle, dict):
            continue
        complete = bool(candle.get("complete", True))
        if not complete and not include_incomplete:
            continue
        ts = str(candle.get("time") or "")
        if not ts:
            continue
        row = {
            "time": ts,
            "pair": pair,
            "granularity": granularity,
            "price": price,
            "complete": complete,
            "volume": int(candle.get("volume") or 0),
        }
        for block_name in ("mid", "bid", "ask"):
            block = candle.get(block_name)
            if isinstance(block, dict):
                parsed = _parse_ohlc(block)
                if parsed is not None:
                    row[block_name] = parsed
        if any(key in row for key in ("mid", "bid", "ask")):
            yield row


def _parse_ohlc(block: dict[str, Any]) -> dict[str, float] | None:
    try:
        return {
            "o": float(block["o"]),
            "h": float(block["h"]),
            "l": float(block["l"]),
            "c": float(block["c"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
