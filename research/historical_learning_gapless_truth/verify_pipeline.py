#!/usr/bin/env python3
"""Independent, read-only oracle for the gapless historical-learning run."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
import random
import statistics
import struct
from typing import Any, Iterable


RECORD = struct.Struct(">3i2f")
EXPECTED_EPISODES = 251
EXPECTED_ALLOWED_EPISODES = 146
EXPECTED_HOURS = 418
PAIRS = {"AUD_JPY", "EUR_JPY", "EUR_USD"}
BOOTSTRAP_SEED = 20260809


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def raw_oracle(repo: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    rows_by_pair: Counter[str] = Counter()
    duplicate_timestamps = exact_duplicates = 0
    seen: set[tuple[str, str]] = set()
    for entry in manifest["entries"]:
        identity = (entry["pair"], entry["utc_from"])
        if identity in seen:
            errors.append(f"duplicate_manifest_identity:{identity}")
        seen.add(identity)
        if not entry["complete"]:
            if not entry["market_closed"]:
                errors.append(f"market_open_incomplete:{identity}")
            continue
        path = repo / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            errors.append(f"hash_or_path:{identity}")
            continue
        try:
            payload = lzma.decompress(path.read_bytes())
        except lzma.LZMAError:
            errors.append(f"lzma:{identity}")
            continue
        if len(payload) % RECORD.size:
            errors.append(f"record_remainder:{identity}")
            continue
        previous_ms = -1
        previous_record: tuple[int, int, int, float, float] | None = None
        count = 0
        for record in RECORD.iter_unpack(payload):
            millis, ask, bid, ask_volume, bid_volume = record
            if (
                not 0 <= millis < 3_600_000 or millis < previous_ms or bid > ask
                or ask <= 0 or bid <= 0
                or not math.isfinite(ask_volume) or not math.isfinite(bid_volume)
                or ask_volume < 0 or bid_volume < 0
            ):
                errors.append(f"schema_or_order:{identity}:{count}")
                break
            duplicate_timestamps += int(millis == previous_ms)
            exact_duplicates += int(record == previous_record)
            previous_ms, previous_record = millis, record
            count += 1
        if count != entry["rows"]:
            errors.append(f"row_count:{identity}:{count}!={entry['rows']}")
        rows_by_pair[entry["pair"]] += count
    return {
        "manifest_entries": len(manifest["entries"]),
        "unique_hours": len(seen),
        "rows_by_pair": dict(sorted(rows_by_pair.items())),
        "duplicate_timestamps": duplicate_timestamps,
        "exact_duplicate_records": exact_duplicates,
        "errors": errors,
        "pass": len(manifest["entries"]) == EXPECTED_HOURS and len(seen) == EXPECTED_HOURS and not errors,
    }


def profit_factor(values: list[float]) -> float | str | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if losses:
        return gains / losses
    if gains:
        return "Infinity"
    return None


def drawdown(values: Iterable[float]) -> float:
    equity = peak = worst = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def independent_lcb(values: list[float]) -> float | None:
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(5_000))
    return means[int(0.025 * (len(means) - 1))]


def metric_oracle(report: dict[str, Any]) -> dict[str, Any]:
    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report["prediction_rows"]:
        predictions[row["window_id"]].append(row)
    windows: list[dict[str, Any]] = []
    passed = True
    for window in report["windows"]:
        if window["status"] != "EVALUATED":
            windows.append({"id": window["id"], "status": window["status"], "exact_match": True})
            continue
        rows = predictions[window["id"]]
        actual = [float(row["actual_net_jpy"]) for row in rows]
        selected = [bool(row["price_action_selected"]) for row in rows]
        values = [value if take else 0.0 for value, take in zip(actual, selected)]
        deltas = [value - baseline for value, baseline in zip(values, actual)]
        oracle = {
            "trades_available": len(rows),
            "trades_selected": sum(selected),
            "net_jpy": sum(values),
            "baseline_net_jpy": sum(actual),
            "incremental_net_jpy": sum(deltas),
            "profit_factor": profit_factor(values),
            "max_drawdown_jpy": drawdown(values),
        }
        stated = window["PRICE_ACTION_HGB"]
        exact_match = all(stated[key] == value for key, value in oracle.items())
        passed = passed and exact_match
        windows.append({
            "id": window["id"],
            "status": window["status"],
            "exact_match": exact_match,
            "oracle": oracle,
            "independent_bootstrap_lcb_jpy": independent_lcb(deltas),
            "reported_bootstrap_lcb_jpy": stated["paired_lcb_jpy"],
        })
    return {"windows": windows, "pass": passed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / "research/historical_learning_gapless_truth"
    manifest = json.loads((root / "cache/manifest_v2.json").read_text())
    report = json.loads((root / "report_v2.json").read_text())
    gap = json.loads((root / "gap_audit_v2.json").read_text())
    raw = raw_oracle(repo, manifest)
    metrics = metric_oracle(report)
    scope_pass = (
        manifest["required_hours"] == EXPECTED_HOURS
        and manifest["episode_scope"]["all_labeled"] == EXPECTED_EPISODES
        and manifest["episode_scope"]["allowed_pair_labeled"] == EXPECTED_ALLOWED_EPISODES
        and gap["episodes_all"] == EXPECTED_EPISODES
        and gap["episodes_allowed_pairs"] == EXPECTED_ALLOWED_EPISODES
    )
    holdout_pass = report["holdout_used"] is False
    double_download_pass = all(row["match"] for row in manifest["double_download_checks"])
    result = {
        "contract": "historical_learning_gapless_truth_independent_oracle_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "raw": raw,
        "metrics": metrics,
        "scope_pass": scope_pass,
        "holdout_sealed": holdout_pass,
        "double_download_pass": double_download_pass,
        "overall_pass": raw["pass"] and metrics["pass"] and scope_pass and holdout_pass and double_download_pass,
    }
    output = root / "independent_oracle_v2.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"overall_pass": result["overall_pass"], "output": str(output.relative_to(repo))}, sort_keys=True))
    return 0 if result["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
