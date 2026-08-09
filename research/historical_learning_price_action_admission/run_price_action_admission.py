#!/usr/bin/env python3
"""Causal S5 bid/ask price-action admission test for the frozen 549 cohort."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


ANCHOR = "2026-07-09T07:46:03.151624347Z"
WINDOWS = (
    ("INITIAL_16D", "2026-06-23T07:46:03.151624347Z", ANCHOR),
    ("DOUBLE_32D", "2026-06-07T07:46:03.151624347Z", ANCHOR),
    ("QUADRUPLE_64D", "2026-05-06T07:46:03.151624347Z", ANCHOR),
)
SOURCES = {
    "AUD_JPY": ("logs/replay/oanda_history/20260709T003937Z/AUD_JPY/AUD_JPY_S5_BA_20260311T003937Z_20260709T003937Z.jsonl.gz", "222e0b7c39dc6fa101654268be19393a41ebbbad2c8c7c54315844aa461ba8f6"),
    "EUR_JPY": ("logs/replay/oanda_history/20260709T003937Z/EUR_JPY/EUR_JPY_S5_BA_20260311T003937Z_20260709T003937Z.jsonl.gz", "26af73c453465260c98205b88d8100a18dbde3e088626a412d0860d8eeb08dac"),
    "EUR_USD": ("logs/replay/oanda_history/20260709T003937Z/EUR_USD/EUR_USD_S5_BA_20260311T003937Z_20260709T003937Z.jsonl.gz", "b634f396889a2b41ae13f5e0957ea705639dad0c16e1c2b3be08ba3ebc98d5ed"),
}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Bar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    median_spread_bps: float


def floor_m5(timestamp: datetime) -> datetime:
    epoch = int(timestamp.timestamp())
    return datetime.fromtimestamp(epoch - epoch % 300, tz=timezone.utc)


def finish_bucket(bucket: dict[str, Any] | None) -> Bar | None:
    if bucket is None:
        return None
    timestamps = bucket["timestamps"]
    if len(timestamps) != 60 or len(set(timestamps)) != 60:
        return None
    if timestamps[0] != bucket["start"] or timestamps[-1] != bucket["start"] + timedelta(seconds=295):
        return None
    if any((right - left).total_seconds() != 5 for left, right in zip(timestamps, timestamps[1:])):
        return None
    return Bar(
        start=bucket["start"],
        open=bucket["open"],
        high=bucket["high"],
        low=bucket["low"],
        close=bucket["close"],
        median_spread_bps=statistics.median(bucket["spread_bps"]),
    )


def load_bars(path: Path, pair: str, parse_time: Any) -> tuple[list[Bar], dict[str, Any]]:
    bars: list[Bar] = []
    bucket: dict[str, Any] | None = None
    rows = invalid_rows = 0
    last_timestamp: datetime | None = None
    gap_seconds: list[float] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            timestamp = parse_time(str(row["time"]))
            if timestamp > parse_time(ANCHOR):
                break
            rows += 1
            if row.get("pair") != pair or row.get("granularity") != "S5" or row.get("complete") is not True:
                invalid_rows += 1
                continue
            bid, ask = row.get("bid"), row.get("ask")
            if not isinstance(bid, dict) or not isinstance(ask, dict):
                invalid_rows += 1
                continue
            if last_timestamp is not None:
                gap = (timestamp - last_timestamp).total_seconds()
                if gap != 5:
                    gap_seconds.append(gap)
            last_timestamp = timestamp
            start = floor_m5(timestamp)
            mid_open = (float(bid["o"]) + float(ask["o"])) / 2
            mid_high = (float(bid["h"]) + float(ask["h"])) / 2
            mid_low = (float(bid["l"]) + float(ask["l"])) / 2
            mid_close = (float(bid["c"]) + float(ask["c"])) / 2
            spread_bps = (float(ask["c"]) - float(bid["c"])) / mid_close * 10_000
            if bucket is None or bucket["start"] != start:
                completed = finish_bucket(bucket)
                if completed is not None:
                    bars.append(completed)
                bucket = {"start": start, "open": mid_open, "high": mid_high, "low": mid_low, "close": mid_close, "timestamps": [timestamp], "spread_bps": [spread_bps]}
            else:
                bucket["high"] = max(bucket["high"], mid_high)
                bucket["low"] = min(bucket["low"], mid_low)
                bucket["close"] = mid_close
                bucket["timestamps"].append(timestamp)
                bucket["spread_bps"].append(spread_bps)
    completed = finish_bucket(bucket)
    if completed is not None:
        bars.append(completed)
    return bars, {
        "path": str(path),
        "rows_to_anchor": rows,
        "invalid_rows": invalid_rows,
        "complete_m5_bars": len(bars),
        "non_5s_gaps": len(gap_seconds),
        "minimum_non_5s_gap_seconds": min(gap_seconds, default=None),
        "maximum_non_5s_gap_seconds": max(gap_seconds, default=None),
        "first_complete_m5_utc": bars[0].start.isoformat().replace("+00:00", "Z") if bars else None,
        "last_complete_m5_utc": bars[-1].start.isoformat().replace("+00:00", "Z") if bars else None,
    }


def structure_features(bars: list[Bar]) -> dict[str, float] | None:
    if len(bars) != 48:
        return None
    if any((right.start - left.start).total_seconds() != 300 for left, right in zip(bars, bars[1:])):
        return None
    closes = np.asarray([bar.close for bar in bars], dtype=float)
    current = closes[-1]

    def ret(period: int) -> float:
        reference = closes[-period - 1] if period < len(closes) else bars[0].open
        return current / reference - 1.0

    def range_features(period: int) -> tuple[float, float, float, float, float]:
        scoped = bars[-period:]
        high = max(bar.high for bar in scoped)
        low = min(bar.low for bar in scoped)
        span = high - low
        position = (current - low) / span if span > 0 else 0.5
        path = sum(abs(right - left) for left, right in zip(closes[-period:], closes[-period + 1 :]))
        efficiency = abs(closes[-1] - closes[-period]) / path if path > 0 else 0.0
        return position, efficiency, (current - high) / current * 10_000, (current - low) / current * 10_000, statistics.median(bar.median_spread_bps for bar in scoped)

    pos12, eff12, high12, low12, spread12 = range_features(12)
    pos48, eff48, high48, low48, spread48 = range_features(48)
    recent3 = max(bar.high for bar in bars[-3:]) - min(bar.low for bar in bars[-3:])
    prior12 = max(bar.high for bar in bars[-12:]) - min(bar.low for bar in bars[-12:])
    return {
        "pa_return_1": ret(1),
        "pa_return_3": ret(3),
        "pa_return_12": ret(12),
        "pa_return_48": ret(48),
        "pa_range_position_12": pos12,
        "pa_range_position_48": pos48,
        "pa_trend_efficiency_12": eff12,
        "pa_trend_efficiency_48": eff48,
        "pa_distance_high_12_bps": high12,
        "pa_distance_low_12_bps": low12,
        "pa_distance_high_48_bps": high48,
        "pa_distance_low_48_bps": low48,
        "pa_median_spread_12_bps": spread12,
        "pa_median_spread_48_bps": spread48,
        "pa_range_expansion_3_over_12": recent3 / prior12 if prior12 > 0 else 0.0,
    }


def attach_features(rows: list[dict[str, Any]], pair_bars: dict[str, list[Bar]], parse_time: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    starts = {pair: [bar.start for bar in bars] for pair, bars in pair_bars.items()}
    enriched = []
    reasons = Counter()
    for source in rows:
        row = dict(source)
        pair = str(row["pair"])
        if pair not in pair_bars:
            row["price_action_features"] = None
            reasons["PAIR_ARCHIVE_MISSING"] += 1
        else:
            feature_at = parse_time(row["feature_at_utc"])
            position = bisect_right(starts[pair], feature_at - timedelta(minutes=5))
            scoped = pair_bars[pair][max(0, position - 48) : position]
            features = structure_features(scoped)
            row["price_action_features"] = features
            reasons["AVAILABLE" if features is not None else "GAP_OR_LOOKBACK_INCOMPLETE"] += 1
        enriched.append(row)
    return enriched, dict(sorted(reasons.items()))


def model_features(parent: Any, row: dict[str, Any], include_price_action: bool) -> dict[str, Any]:
    values = dict(parent.features(row))
    if include_price_action:
        values.update(row["price_action_features"])
    return values


def fit_predict(selection: Any, parent: Any, train: list[dict[str, Any]], validation: list[dict[str, Any]], include_price_action: bool) -> np.ndarray:
    model = selection.pipeline()
    model.fit([model_features(parent, row, include_price_action) for row in train], np.asarray([float(row["net_jpy"]) for row in train]))
    return np.asarray(model.predict([model_features(parent, row, include_price_action) for row in validation]), dtype=float)


def run(repo: Path) -> dict[str, Any]:
    parent = import_module("pa_admission_parent", repo / "research/historical_learning_admission/run_admission.py")
    selection = import_module("pa_selection_parent", repo / "research/historical_learning_selection_rca/run_selection_rca.py")
    episodes_path = repo / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"
    if sha256(episodes_path) != "efcf6b0fb675050d6a08efc0119065e0874e50e1c51373a0c0fb61bb6ebd815e":
        raise RuntimeError("frozen episode input changed")
    pair_bars: dict[str, list[Bar]] = {}
    source_reports = {}
    for pair, (relative, expected) in SOURCES.items():
        path = repo / relative
        if sha256(path) != expected:
            raise RuntimeError(f"S5 source changed: {relative}")
        bars, source_report = load_bars(path, pair, parent.parse_time)
        pair_bars[pair] = bars
        source_report["sha256"] = expected
        source_reports[pair] = source_report
    episodes = selection.read_jsonl(episodes_path)
    labeled = [row for row in episodes if row["label_status"] == "ACTUAL_AFTER_COST"]
    rows, coverage_reasons = attach_features(labeled, pair_bars, parent.parse_time)
    windows = []
    prediction_rows = []
    for window_id, start_text, end_text in WINDOWS:
        start, end = parent.parse_time(start_text), parent.parse_time(end_text)
        scoped = [row for row in rows if start <= parent.parse_time(row["feature_at_utc"]) <= end]
        train, validation, purged = parent.split_rows(scoped)
        train_feature = [row for row in train if row["price_action_features"] is not None]
        validation_feature = [row for row in validation if row["price_action_features"] is not None]
        base = selection.metric(validation, [True] * len(validation))
        window: dict[str, Any] = {
            "id": window_id,
            "labeled_events": len(scoped),
            "train_events": len(train),
            "validation_events": len(validation),
            "purged_train_events": purged,
            "train_price_action_features": len(train_feature),
            "validation_price_action_features": len(validation_feature),
            "validation_feature_coverage": len(validation_feature) / len(validation) if validation else 0.0,
            "ALL_TRADES": base,
        }
        if len(train_feature) < 30 or len(validation) < 30 or not validation_feature:
            window["status"] = "NOT_FIT_ADMISSION_GATE"
            windows.append(window)
            continue
        metadata_prediction = fit_predict(selection, parent, train_feature, validation_feature, False)
        price_prediction = fit_predict(selection, parent, train_feature, validation_feature, True)
        index = {row["episode_id"]: position for position, row in enumerate(validation_feature)}
        metadata_selected = []
        price_selected = []
        for row in validation:
            if row["episode_id"] not in index:
                metadata_selected.append(True)
                price_selected.append(True)
                continue
            position = index[row["episode_id"]]
            metadata_selected.append(bool(metadata_prediction[position] > 0))
            price_selected.append(bool(price_prediction[position] > 0))
        metadata_report = selection.metric(validation, metadata_selected)
        price_report = selection.metric(validation, price_selected)
        pair_delta = [
            (float(row["net_jpy"]) if price else 0.0) - (float(row["net_jpy"]) if metadata else 0.0)
            for row, price, metadata in zip(validation, price_selected, metadata_selected)
        ]
        price_report["incremental_vs_coverage_matched_metadata_jpy"] = sum(pair_delta)
        price_report["paired_lcb_vs_coverage_matched_metadata_jpy"] = selection.bootstrap_mean_ci(pair_delta)[0]
        price_report["gates"] = {
            "minimum_train_feature_rows": len(train_feature) >= 30,
            "minimum_validation_events": len(validation) >= 30,
            "minimum_validation_feature_coverage": len(validation_feature) / len(validation) >= 0.50,
            "net_above_all_trades": price_report["incremental_net_jpy"] > 0,
            "paired_lcb_above_all_trades_positive": price_report["paired_lcb_jpy"] is not None and price_report["paired_lcb_jpy"] > 0,
            "retention_at_least_80pct": price_report["retention_ratio"] >= 0.80,
            "profit_factor_above_one": price_report["profit_factor"] == "Infinity" or (isinstance(price_report["profit_factor"], (int, float)) and price_report["profit_factor"] > 1),
            "margin_complete": price_report["margin_coverage"] == 1.0,
        }
        price_report["accepted"] = all(price_report["gates"].values())
        window["status"] = "EVALUATED"
        window["COVERAGE_MATCHED_METADATA_HGB"] = metadata_report
        window["PRICE_ACTION_HGB"] = price_report
        for row, metadata, price in zip(validation, metadata_selected, price_selected):
            prediction_rows.append({"window_id": window_id, "episode_id": row["episode_id"], "actual_net_jpy": row["net_jpy"], "price_action_features_available": row["price_action_features"] is not None, "metadata_selected": metadata, "price_action_selected": price})
        windows.append(window)
    evaluated = [window for window in windows if window["status"] == "EVALUATED"]
    decision = "ACCEPT" if len(evaluated) >= 2 and all(window["PRICE_ACTION_HGB"]["accepted"] for window in evaluated) else "REJECT"
    return {
        "contract": "historical_learning_price_action_admission_result_v1",
        "preregister_sha256": sha256(repo / "research/historical_learning_price_action_admission/preregister_v1.json"),
        "holdout_used": False,
        "s5_sources": source_reports,
        "labeled_episodes": len(labeled),
        "feature_coverage": {"available": sum(row["price_action_features"] is not None for row in rows), "ratio": sum(row["price_action_features"] is not None for row in rows) / len(rows), "reasons": coverage_reasons},
        "windows": windows,
        "overall_decision": decision,
        "policy_admission": "BLOCKED_ALL_ENTRY_COUNTERFACTUAL_AND_MARGIN_COVERAGE_INCOMPLETE",
        "multidimensional_sweep": "NOT_OPENED_FIXED_FEATURE_ADMISSION_FAILED" if decision != "ACCEPT" else "TRAIN_ONLY_SWEEP_MAY_BE_PREREGISTERED_NEXT",
        "prediction_rows": prediction_rows,
        "permissions": {"read_only_inputs": True, "holdout_used": False, "paper": False, "live": False, "broker_order": False, "deploy": False},
    }


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("research/historical_learning_price_action_admission/report_v1.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = run(repo)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": report["feature_coverage"], "decision": report["overall_decision"], "holdout_used": report["holdout_used"]}, sort_keys=True))


if __name__ == "__main__":
    main()
