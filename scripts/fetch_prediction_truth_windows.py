#!/usr/bin/env python3
"""Fetch sparse read-only OANDA bid/ask truth around prediction windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for item in (SRC, SCRIPT_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import oanda_history_fetch as fetch  # noqa: E402

from quant_rabbit.broker.oanda import OandaReadOnlyClient  # noqa: E402


CONTRACT = "QR_PREDICTION_SPARSE_TRUTH_FETCH_V1"


def main() -> int:
    args = _parse_args()
    granularity = str(args.granularity).upper()
    fetch._validate_granularities([granularity])
    windows = _prediction_windows(
        args.predictions,
        granularity=granularity,
    )
    client = OandaReadOnlyClient()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        fetch.FetchTask(
            pair=pair,
            granularity=granularity,
            start=start,
            end=end,
            price="BA",
        )
        for pair, pair_windows in sorted(windows.items())
        for start, end in pair_windows
    ]
    task_summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        summary = fetch._fetch_task(
            client,
            task,
            run_dir=run_dir,
            receipt_root=args.output_dir,
            max_candles_per_request=args.max_candles_per_request,
            sleep_seconds=args.sleep_seconds,
            retries=args.retries,
            include_incomplete=False,
            compress=True,
            dry_run=args.dry_run,
        )
        task_summaries.append(summary)
        errors.extend(summary.get("errors") or [])
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT,
        "read_only": True,
        "predictions": [
            {
                "path": str(path.resolve()),
                "sha256": _file_sha256(path),
            }
            for path in args.predictions
        ],
        "granularity": granularity,
        "price": "BA",
        "output_dir": str(run_dir.resolve()),
        "pairs": sorted(windows),
        "merged_windows": sum(len(items) for items in windows.values()),
        "tasks": task_summaries,
        "total_rows": sum(int(item.get("rows") or 0) for item in task_summaries),
        "total_requests": sum(
            int(item.get("requests") or 0) for item in task_summaries
        ),
        "errors": errors,
        "dry_run": bool(args.dry_run),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")
    print(
        f"rows={report['total_rows']} requests={report['total_requests']} "
        f"errors={len(errors)}"
    )
    return 1 if errors else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--granularity", default="S5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs" / "replay" / "oanda_prediction_truth",
    )
    parser.add_argument("--max-candles-per-request", type=int, default=4500)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prediction_windows(
    paths: Sequence[Path],
    *,
    granularity: str,
) -> dict[str, list[tuple[datetime, datetime]]]:
    seconds = fetch.GRANULARITY_SECONDS[str(granularity).upper()]
    pad = timedelta(seconds=seconds)
    raw: dict[str, list[tuple[datetime, datetime]]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                pair = str(payload.get("pair") or "").strip().upper()
                start = _parse_time(payload.get("entry_timestamp_utc"))
                end = _parse_time(payload.get("future_timestamp_utc"))
                if not pair or start is None or end is None or end <= start:
                    raise ValueError(f"invalid prediction truth window in {path}")
                raw.setdefault(pair, []).append((start, end + pad))
    return {pair: _merge_windows(items) for pair, items in raw.items()}


def _merge_windows(
    windows: Sequence[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    output: list[tuple[datetime, datetime]] = []
    for start, end in sorted(windows):
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
