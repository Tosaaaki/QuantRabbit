#!/usr/bin/env python3
"""Validate the bounded OANDA S5 bid/ask extension and its repeat fetch."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent
PRIMARY = ROOT / "oanda_extension/20260810T031843Z"
REPEAT = ROOT / "oanda_extension_repeat/20260810T031857Z"
PAIRS = ("AUD_JPY", "EUR_JPY", "EUR_USD")


def parse_time(value: str) -> datetime:
    head, _, tail = value.partition(".")
    if tail:
        fraction = tail[:-1] if tail.endswith("Z") else tail
        value = f"{head}.{fraction[:6]}Z"
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def compressed_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows.append(json.loads(line))
    return rows, digest.hexdigest()


def data_path(root: Path, pair: str) -> Path:
    matches = list((root / pair).glob(f"{pair}_S5_BA_*.jsonl.gz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pair} file under {root}, got {len(matches)}")
    return matches[0]


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [parse_time(str(row["time"])) for row in rows]
    seconds = [(right - left).total_seconds() for left, right in zip(times, times[1:])]
    issues: list[str] = []
    if times != sorted(times):
        issues.append("NON_MONOTONIC")
    if len(times) != len(set(times)):
        issues.append("DUPLICATE_TIME")
    if any(not row.get("complete") for row in rows):
        issues.append("INCOMPLETE_CANDLE")
    if any(row.get("granularity") != "S5" or row.get("price") != "BA" for row in rows):
        issues.append("PRICE_OR_GRANULARITY_MISMATCH")
    for row in rows:
        for side in ("bid", "ask"):
            bar = row[side]
            values = {key: float(bar[key]) for key in ("o", "h", "l", "c")}
            if not (values["l"] <= values["o"] <= values["h"] and values["l"] <= values["c"] <= values["h"]):
                issues.append("OHLC_INVARIANT")
        if any(float(row["bid"][key]) > float(row["ask"][key]) for key in ("o", "h", "l", "c")):
            issues.append("BID_ABOVE_ASK")
    if any(delta <= 0 or delta % 5 != 0 for delta in seconds):
        issues.append("NON_S5_GRID_DELTA")
    return {
        "rows": len(rows),
        "first_time": rows[0]["time"] if rows else None,
        "last_time": rows[-1]["time"] if rows else None,
        "complete_rows": sum(bool(row.get("complete")) for row in rows),
        "gap_count_over_5s": sum(delta > 5 for delta in seconds),
        "max_gap_seconds": max(seconds, default=0),
        "gap_classification": "CANDLE_ENDPOINT_NO_BAR_INTERVAL_UNRESOLVED_WITHOUT_RAW_TICK_TRUTH",
        "issues": sorted(set(issues)),
    }


def main() -> int:
    per_pair: dict[str, Any] = {}
    all_pass = True
    for pair in PAIRS:
        primary_path = data_path(PRIMARY, pair)
        repeat_path = data_path(REPEAT, pair)
        primary_rows, primary_expanded = read_rows(primary_path)
        repeat_rows, repeat_expanded = read_rows(repeat_path)
        audit = validate_rows(primary_rows)
        repeat_audit = validate_rows(repeat_rows)
        content_match = primary_expanded == repeat_expanded and len(primary_rows) == len(repeat_rows)
        pair_pass = not audit["issues"] and not repeat_audit["issues"] and content_match
        all_pass &= pair_pass
        per_pair[pair] = {
            "source": "OANDA_V20_INSTRUMENT_CANDLES_READ_ONLY",
            "timezone": "UTC",
            "price": "BA",
            "granularity": "S5",
            "primary_path": str(primary_path.relative_to(REPO)),
            "repeat_path": str(repeat_path.relative_to(REPO)),
            "primary_compressed_sha256": compressed_sha(primary_path),
            "repeat_compressed_sha256": compressed_sha(repeat_path),
            "expanded_sha256": primary_expanded,
            "repeat_expanded_sha256": repeat_expanded,
            "repeat_content_match": content_match,
            "primary_bytes": primary_path.stat().st_size,
            "repeat_bytes": repeat_path.stat().st_size,
            "audit": audit,
            "repeat_audit": repeat_audit,
            "status": "PASS" if pair_pass else "FAIL",
        }

    result = {
        "contract": "OANDA_S5_BA_EXTENSION_VALIDATION_V1",
        "requested_from_utc": "2026-07-09T00:39:37Z",
        "requested_to_utc": "2026-07-09T07:46:03Z",
        "source_boundary": "OANDA candle rows are executable bid/ask candle truth. Missing candle timestamps are not raw-tick-proved no-trade intervals and are never filled.",
        "strict_gapless_tick_truth": False,
        "per_pair": per_pair,
        "status": "PASS_CANDLE_TRUTH_WITH_UNRESOLVED_NO_BAR_INTERVALS" if all_pass else "FAIL",
    }
    output = ROOT / "oanda_extension_manifest_v1.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
