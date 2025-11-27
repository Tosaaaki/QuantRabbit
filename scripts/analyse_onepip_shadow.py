#!/usr/bin/env python3
"""
Analyse onepip_maker_s1 shadow log metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional


def _load(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def analyse(path: Path) -> dict:
    records = list(_load(path))
    totals = len(records)
    modes = Counter(rec.get("mode", "unknown") for rec in records)
    dir_counter = Counter(rec.get("direction", "NA") for rec in records)
    spread_values = [float(rec.get("spread_pips", 0.0)) for rec in records if rec.get("spread_pips") is not None]
    imbalance_values = [float(rec.get("imbalance", 0.0)) for rec in records if rec.get("imbalance") is not None]
    drift_values = [float(rec.get("drift_pips", 0.0)) for rec in records if rec.get("drift_pips") is not None]
    latency_values = [float(rec.get("latency_ms", 0.0)) for rec in records if rec.get("latency_ms") is not None]

    result = {
        "total_records": totals,
        "mode_counts": dict(modes),
        "direction_counts": dict(dir_counter),
        "avg_spread_pips": mean(spread_values) if spread_values else None,
        "avg_abs_imbalance": mean(abs(v) for v in imbalance_values) if imbalance_values else None,
        "avg_abs_drift_pips": mean(abs(v) for v in drift_values) if drift_values else None,
        "avg_latency_ms": mean(latency_values) if latency_values else None,
    }

    status_counts = Counter(rec.get("result", "ready") for rec in records if rec.get("mode") == "live")
    if status_counts:
        result["live_results"] = dict(status_counts)

    skip_reasons = Counter()
    for rec in records:
        reason = rec.get("reason")
        if reason:
            skip_reasons[reason] += 1
    if skip_reasons:
        result["reasons"] = dict(skip_reasons)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse onepip_maker_s1 shadow log")
    ap.add_argument("path", type=Path, help="Path to shadow log JSONL")
    args = ap.parse_args()
    summary = analyse(args.path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
