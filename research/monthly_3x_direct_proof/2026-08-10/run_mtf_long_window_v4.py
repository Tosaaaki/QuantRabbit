#!/usr/bin/env python3
"""Apply the frozen Q-XFX-MTF-001 rule to the long OANDA M5 cohort."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "research/monthly_3x_growth_engine/2026-08-10"
sys.path.insert(0, str(BASE))
from run_strategy_expansion import digest, load_bars, metrics, signal, utc  # noqa: E402
from run_strategy_exit_expansion import arrays, replay_config  # noqa: E402


V1 = HERE / "preregister_v1.json"
PREREG = HERE / "preregister_v4_mtf_long_window.json"


def completed_h1(bars):
    grouped = defaultdict(list)
    for row in bars:
        hour = row["time"].replace(minute=0, second=0, microsecond=0)
        grouped[hour].append(row)
    result = []
    for hour, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["time"])
        expected = [hour + timedelta(minutes=5 * i) for i in range(12)]
        if [row["time"] for row in rows] != expected:
            continue
        result.append({
            "start": hour, "end": hour + timedelta(hours=1),
            "o": float(rows[0]["mid"]["o"]),
            "h": max(float(row["mid"]["h"]) for row in rows),
            "l": min(float(row["mid"]["l"]) for row in rows),
            "c": float(rows[-1]["mid"]["c"]),
        })
    return result


def opinions_by_signal(bars, h1):
    by_end = {bar["end"]: index for index, bar in enumerate(h1)}
    result = {}
    for row in bars:
        signal_time = row["time"] + timedelta(minutes=5)
        end = signal_time.replace(minute=0, second=0, microsecond=0)
        index = by_end.get(end)
        if index is None or index < 2:
            continue
        a, b, c = h1[index - 2:index + 1]
        if not (a["end"] == b["start"] and b["end"] == c["start"]):
            continue
        opinion = "NEUTRAL"
        if c["c"] > max(a["h"], b["h"]) and a["c"] < b["c"] < c["c"] and c["c"] > c["o"]:
            opinion = "LONG"
        elif c["c"] < min(a["l"], b["l"]) and a["c"] > b["c"] > c["c"] and c["c"] < c["o"]:
            opinion = "SHORT"
        result[row["time"]] = opinion
    return result


def avg_mid(rows, left, right):
    values = [float(row["mid"]["c"]) for row in rows if left <= row["time"] < right]
    return sum(values) / len(values) if values else math.nan


def main():
    v1 = json.loads(V1.read_text())
    holdout = utc(v1["split"]["holdout_start_utc"])
    bars, data, time_index, parent = {}, {}, {}, {}
    for pair, source in v1["sources"].items():
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise SystemExit(f"source SHA mismatch {path}")
        bars[pair] = load_bars(path, holdout)
        data[pair] = arrays(bars[pair])
        time_index[pair] = {row["time"]: i for i, row in enumerate(bars[pair])}
        parent[pair] = opinions_by_signal(bars[pair], completed_h1(bars[pair]))
    ej = {row["time"]: row for row in bars["EUR_JPY"]}
    eu = {row["time"]: row for row in bars["EUR_USD"]}

    grid = v1["signal_grid"]
    cache = {}
    for pair in bars:
        for family in grid["families"]:
            for lookback in grid["lookback"]:
                for session in grid["entry_session_utc"]:
                    for exit_policy in grid["exit_policies"]:
                        raw = replay_config(bars[pair], data[pair], family, int(lookback), session, exit_policy)
                        enriched = []
                        for trade in raw:
                            index = time_index[pair].get(trade["entry_time"])
                            if index is None or index < 1:
                                continue
                            signal_time = bars[pair][index - 1]["time"]
                            side = signal(family, index - 1, int(lookback), data[pair]["close"], data[pair]["high"], data[pair]["low"], data[pair]["sma20"], data[pair]["sma50"])
                            opinion = parent[pair].get(signal_time)
                            if side is None or opinion is None:
                                continue
                            pnl = float(trade["pnl"])
                            if pair == "EUR_USD":
                                cross, base = ej.get(trade["exit_time"]), eu.get(trade["exit_time"])
                                if cross is None or base is None:
                                    continue
                                pnl *= float(cross["mid"]["o"]) / float(base["mid"]["o"]) * 0.999
                            enriched.append({**trade, "signal_time": signal_time, "side": side, "parent_opinion": opinion, "pnl_jpy_per_1000u": pnl})
                        cache[(pair, family, int(lookback), session, exit_policy)] = enriched

    rows = []
    for days in (16, 32, 64):
        start = holdout - timedelta(days=days)
        train_end = start + timedelta(days=days * 0.6)
        val_start = train_end + timedelta(hours=1)
        for key, trades in cache.items():
            for arm in ("BASELINE_ELIGIBLE", "REJECT_OPPOSITE_H1"):
                armed = trades if arm == "BASELINE_ELIGIBLE" else [t for t in trades if t["parent_opinion"] in {"NEUTRAL", t["side"]}]
                for split, left, right in (("TRAIN", start, train_end), ("VALIDATION", val_start, holdout)):
                    selected = [t for t in armed if left <= t["entry_time"] and t["exit_time"] < right]
                    values = [float(t["pnl_jpy_per_1000u"]) for t in selected]
                    rows.append({
                        "window": f"{days}D", "split": split, "arm": arm,
                        "pair": key[0], "family": key[1], "lookback": key[2], "session": key[3], "exit_policy": key[4],
                        **metrics(values, ":".join(map(str, (days, split, arm, *key)))),
                    })

    passed_train = set()
    for row in rows:
        if row["split"] == "TRAIN" and row["arm"] == "REJECT_OPPOSITE_H1" and row["trades"] >= 20 and (row["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (row["profit_factor"] or 0) > 1 and (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0:
            passed_train.add((row["window"], row["pair"], row["family"], row["session"], row["exit_policy"], row["lookback"]))
    plateau = set()
    for row in rows:
        base = (row["window"], row["pair"], row["family"], row["session"], row["exit_policy"])
        passed = {lookback for lookback in grid["lookback"] if (*base, int(lookback)) in passed_train}
        if (12 in passed and 24 in passed) or (24 in passed and 48 in passed):
            plateau.add(base)
    for row in rows:
        row["train_plateau"] = (row["window"], row["pair"], row["family"], row["session"], row["exit_policy"]) in plateau
        row["validation_pass"] = bool(row["split"] == "VALIDATION" and row["arm"] == "REJECT_OPPOSITE_H1" and row["train_plateau"] and row["trades"] >= 10 and (row["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (row["profit_factor"] or 0) > 1 and (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0)

    val = {(r["window"], r["pair"], r["family"], r["lookback"], r["session"], r["exit_policy"]): r for r in rows if r["validation_pass"]}
    target = v1["target"]
    candidates = []
    for key, r64 in val.items():
        if key[0] != "64D":
            continue
        r32 = val.get(("32D", *key[1:]))
        if r32 is None:
            continue
        pair = r64["pair"]
        notional = avg_mid(bars[pair if pair.endswith("_JPY") else "EUR_JPY"], holdout - timedelta(days=64), holdout) * 1000
        scale = math.floor(target["margin_cap_jpy"] / (notional * target["margin_rate"]))
        val_days = 64 * 0.4 - 1 / 24
        trades30 = r64["trades"] * 30 / val_days
        net30 = r64["expectancy_jpy_per_1000u"] * trades30 * scale
        lcb30 = r64["bootstrap_lcb_expectancy_jpy"] * trades30 * scale
        dd = r64["max_drawdown_jpy_per_1000u"] * scale
        candidates.append({"key": key, "validation_32d": r32, "validation_64d": r64, "max_units": scale * 1000, "projected_30d_net_jpy": net30, "projected_30d_lcb_jpy": lcb30, "scaled_dd_jpy": dd, "monthly_3x_pass": net30 >= 400000 and lcb30 >= 400000 and dd <= 80000})
    candidates.sort(key=lambda x: x["projected_30d_lcb_jpy"], reverse=True)
    report = {
        "contract": "MONTHLY_3X_DIRECT_PROOF_MTF_LONG_WINDOW_V4", "holdout_used": False,
        "eligible_trade_rows": sum(r["trades"] for r in rows if r["arm"] == "BASELINE_ELIGIBLE"),
        "train_plateau_count": len(plateau), "stable_32d_64d_count": len(candidates),
        "monthly_3x_pass_count": sum(c["monthly_3x_pass"] for c in candidates),
        "best_candidate": candidates[0] if candidates else None,
        "conclusion": "MONTHLY_3X_PROVED" if any(c["monthly_3x_pass"] for c in candidates) else "MONTHLY_3X_NOT_PROVED",
    }
    (HERE / "mtf_long_window_grid_v4.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (HERE / "mtf_long_window_report_v4.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
