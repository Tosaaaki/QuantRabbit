#!/usr/bin/env python3
"""One-axis completed-bar quality refinement after the frozen V1 checkpoint."""

from __future__ import annotations

import json
import math
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "research/monthly_3x_growth_engine/2026-08-10"
sys.path.insert(0, str(BASE))
from run_strategy_expansion import digest, load_bars, metrics, utc  # noqa: E402
from run_strategy_exit_expansion import arrays, replay_config  # noqa: E402


V1 = HERE / "preregister_v1.json"
PREREG = HERE / "preregister_v2_signal_quality.json"


def quality_features(pair_bars: list[dict[str, Any]], time_index: dict, data: dict[str, np.ndarray], entry_time) -> dict[str, float] | None:
    index = time_index.get(entry_time)
    if index is None or index < 52:
        return None
    signal = index - 1
    atr = float(data["atr14"][signal])
    if not math.isfinite(atr) or atr <= 0:
        return None
    body = abs(float(pair_bars[signal]["mid"]["c"]) - float(pair_bars[signal]["mid"]["o"])) / atr
    slope = abs(float(data["sma20"][signal] - data["sma20"][signal - 3])) / atr
    trailing = data["atr14"][signal - 48:signal]
    trailing = trailing[np.isfinite(trailing)]
    atr_ratio = atr / float(np.median(trailing)) if len(trailing) else math.nan
    return {"body_atr": body, "slope3_atr": slope, "atr48_ratio": atr_ratio}


def passes(name: str, feature: dict[str, float]) -> bool:
    if name == "NONE":
        return True
    if name == "BODY_GE_0_5ATR":
        return feature["body_atr"] >= 0.5
    if name == "SLOPE3_GE_0_15ATR":
        return feature["slope3_atr"] >= 0.15
    if name == "ATR48_GE_1_1":
        return feature["atr48_ratio"] >= 1.1
    if name == "BODY_AND_SLOPE":
        return feature["body_atr"] >= 0.5 and feature["slope3_atr"] >= 0.15
    raise ValueError(name)


def main() -> None:
    v1 = json.loads(V1.read_text())
    prereg = json.loads(PREREG.read_text())
    holdout = utc(v1["split"]["holdout_start_utc"])
    bars = {}
    prepared = {}
    for pair, source in v1["sources"].items():
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise SystemExit(f"source SHA mismatch: {path}")
        bars[pair] = load_bars(path, holdout)
        prepared[pair] = arrays(bars[pair])
    cross_eurjpy = {row["time"]: row for row in bars["EUR_JPY"]}
    cross_eurusd = {row["time"]: row for row in bars["EUR_USD"]}
    time_indexes = {pair: {row["time"]: i for i, row in enumerate(rows)} for pair, rows in bars.items()}

    cache = {}
    frozen = prereg["frozen_from_v1"]
    for pair in frozen["pairs"]:
        for family in frozen["families"]:
            for lookback in frozen["lookback"]:
                for session in frozen["sessions"]:
                    raw = replay_config(bars[pair], prepared[pair], family, lookback, session, "TIME_24")
                    converted = []
                    for trade in raw:
                        value = float(trade["pnl"])
                        if pair == "EUR_USD":
                            ej = cross_eurjpy.get(trade["exit_time"])
                            eu = cross_eurusd.get(trade["exit_time"])
                            if ej is None or eu is None:
                                continue
                            value *= float(ej["mid"]["o"]) / float(eu["mid"]["o"]) * 0.999
                        feature = quality_features(bars[pair], time_indexes[pair], prepared[pair], trade["entry_time"])
                        if feature is not None:
                            converted.append({**trade, **feature, "pnl_jpy_per_1000u": value})
                    cache[(pair, family, lookback, session)] = converted

    filters = list(prereg["single_new_axis"]["values"])
    rows = []
    for days in v1["split"]["windows_days"]:
        start = holdout - timedelta(days=days)
        train_end = start + timedelta(days=days * v1["split"]["train_fraction"])
        val_start = train_end + timedelta(hours=v1["split"]["embargo_hours"])
        for key, trades in cache.items():
            for filter_name in filters:
                filtered = [trade for trade in trades if passes(filter_name, trade)]
                for split, left, right in (("TRAIN", start, train_end), ("VALIDATION", val_start, holdout)):
                    selected = [trade for trade in filtered if left <= trade["entry_time"] and trade["exit_time"] < right]
                    values = [float(trade["pnl_jpy_per_1000u"]) for trade in selected]
                    rows.append({
                        "window": f"{days}D", "split": split, "pair": key[0], "family": key[1],
                        "lookback": key[2], "session": key[3], "exit_policy": "TIME_24",
                        "signal_quality": filter_name,
                        **metrics(values, ":".join(map(str, (days, split, *key, filter_name)))),
                    })

    train_pass = set()
    for row in rows:
        if row["split"] == "TRAIN" and row["trades"] >= 20 and (row["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (row["profit_factor"] or 0) > 1 and (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0:
            train_pass.add((row["window"], row["pair"], row["family"], row["session"], row["signal_quality"], row["lookback"]))
    stable = set()
    for window in {row["window"] for row in rows}:
        for pair in frozen["pairs"]:
            for family in frozen["families"]:
                for session in frozen["sessions"]:
                    for quality in filters:
                        passed = {lookback for lookback in frozen["lookback"] if (window, pair, family, session, quality, lookback) in train_pass}
                        if (12 in passed and 24 in passed) or (24 in passed and 48 in passed):
                            stable.add((window, pair, family, session, quality))
    for row in rows:
        row["train_plateau"] = (row["window"], row["pair"], row["family"], row["session"], row["signal_quality"]) in stable
        row["validation_pass"] = bool(row["split"] == "VALIDATION" and row["train_plateau"] and row["trades"] >= 10 and (row["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (row["profit_factor"] or 0) > 1 and (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0)

    index = {(r["window"], r["pair"], r["family"], r["lookback"], r["session"], r["signal_quality"]): r for r in rows if r["split"] == "VALIDATION"}
    candidates = []
    for key, row64 in index.items():
        if key[0] != "64D" or not row64["validation_pass"]:
            continue
        row32 = index.get(("32D", *key[1:]))
        if row32 is not None and row32["validation_pass"]:
            candidates.append({"validation_32d": row32, "validation_64d": row64})
    report = {
        "contract": prereg["contract"], "preregister_sha256": digest(PREREG), "holdout_used": False,
        "grid_rows": len(rows), "train_plateau_rows": sum(r["train_plateau"] for r in rows),
        "validation_pass_rows": sum(r["validation_pass"] for r in rows),
        "stable_multiwindow_candidates": candidates,
        "conclusion": "STABLE_EDGE_FOUND" if candidates else "NO_STABLE_EDGE",
    }
    (HERE / "signal_quality_grid_v2.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    (HERE / "signal_quality_report_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
