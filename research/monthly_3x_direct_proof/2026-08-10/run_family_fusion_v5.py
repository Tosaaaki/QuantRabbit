#!/usr/bin/env python3
"""Family-normalized technical fusion with one executable decision per M5."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "research/monthly_3x_growth_engine/2026-08-10"
sys.path.insert(0, str(BASE))
from run_strategy_expansion import crosses_financing, digest, in_session, load_bars, metrics, signal, utc  # noqa: E402
from run_strategy_exit_expansion import arrays  # noqa: E402

V1 = HERE / "preregister_v1.json"
PREREG = HERE / "preregister_v5_family_fusion.json"


def family_vote(family, index, data):
    votes = [signal(family, index, lookback, data["close"], data["high"], data["low"], data["sma20"], data["sma50"]) for lookback in (12, 24, 48)]
    counts = Counter(vote for vote in votes if vote is not None)
    if counts["LONG"] >= 2 and counts["SHORT"] == 0:
        return "LONG"
    if counts["SHORT"] >= 2 and counts["LONG"] == 0:
        return "SHORT"
    return None


def fused_side(index, data, families):
    votes = [family_vote(family, index, data) for family in families]
    counts = Counter(vote for vote in votes if vote is not None)
    if counts["LONG"] >= 2 and counts["SHORT"] == 0:
        return "LONG", counts["LONG"]
    if counts["SHORT"] >= 2 and counts["LONG"] == 0:
        return "SHORT", counts["SHORT"]
    return None, 0


def consecutive(times, start, end):
    return start >= 0 and end < len(times) and times[end] - times[start] == timedelta(minutes=5 * (end - start))


def replay(bars, data, families, horizon, session):
    times = [row["time"] for row in bars]
    result, next_free = [], 0
    for index in range(51, len(bars) - horizon - 1):
        entry, exit_ = index + 1, index + 1 + horizon
        if entry < next_free or not in_session(times[entry], session):
            continue
        if not consecutive(times, index - 50, exit_) or crosses_financing(times[entry], times[exit_]):
            continue
        side, support = fused_side(index, data, families)
        if side is None:
            continue
        eb, ea = float(bars[entry]["bid"]["o"]), float(bars[entry]["ask"]["o"])
        xb, xa = float(bars[exit_]["bid"]["o"]), float(bars[exit_]["ask"]["o"])
        entry_extra, exit_extra = 0.5 * (ea - eb), 0.5 * (xa - xb)
        pnl = ((xb - exit_extra) - (ea + entry_extra)) * 1000 if side == "LONG" else ((eb - entry_extra) - (xa + exit_extra)) * 1000
        result.append({"signal_time": times[index], "entry_time": times[entry], "exit_time": times[exit_], "side": side, "supporting_families": support, "pnl": pnl})
        next_free = exit_ + 1
    return result


def avg(rows, left, right):
    values = [float(row["mid"]["c"]) for row in rows if left <= row["time"] < right]
    return float(np.mean(values)) if values else math.nan


def main():
    v1, prereg = json.loads(V1.read_text()), json.loads(PREREG.read_text())
    holdout = utc(v1["split"]["holdout_start_utc"])
    bars, data = {}, {}
    for pair, source in v1["sources"].items():
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise SystemExit("source hash mismatch")
        bars[pair] = load_bars(path, holdout)
        data[pair] = arrays(bars[pair])
    ej, eu = {r["time"]: r for r in bars["EUR_JPY"]}, {r["time"]: r for r in bars["EUR_USD"]}
    cache = {}
    for pair in bars:
        for horizon in prereg["horizon_bars"]:
            for session in prereg["sessions"]:
                raw = replay(bars[pair], data[pair], prereg["families"], horizon, session)
                converted = []
                for trade in raw:
                    pnl = trade["pnl"]
                    if pair == "EUR_USD":
                        cross, base = ej.get(trade["exit_time"]), eu.get(trade["exit_time"])
                        if cross is None or base is None:
                            continue
                        pnl *= float(cross["mid"]["o"]) / float(base["mid"]["o"]) * 0.999
                    converted.append({**trade, "pnl_jpy_per_1000u": pnl})
                cache[(pair, horizon, session)] = converted

    rows = []
    for days in (16, 32, 64):
        start = holdout - timedelta(days=days)
        train_end = start + timedelta(days=days * .6)
        val_start = train_end + timedelta(hours=1)
        for key, trades in cache.items():
            for split, left, right in (("TRAIN", start, train_end), ("VALIDATION", val_start, holdout)):
                picked = [t for t in trades if left <= t["entry_time"] and t["exit_time"] < right]
                rows.append({"window": f"{days}D", "split": split, "pair": key[0], "horizon": key[1], "session": key[2], **metrics([t["pnl_jpy_per_1000u"] for t in picked], f"{days}:{split}:{key}")})
    train_pass = {(r["window"], r["pair"], r["session"], r["horizon"]) for r in rows if r["split"] == "TRAIN" and r["trades"] >= 20 and (r["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (r["profit_factor"] or 0) > 1 and (r["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0}
    plateau = set()
    for row in rows:
        base = (row["window"], row["pair"], row["session"])
        horizons = {h for h in prereg["horizon_bars"] if (*base, h) in train_pass}
        if {6, 12} <= horizons or {12, 24} <= horizons:
            plateau.add(base)
    for r in rows:
        r["train_plateau"] = (r["window"], r["pair"], r["session"]) in plateau
        r["validation_pass"] = bool(r["split"] == "VALIDATION" and r["train_plateau"] and r["trades"] >= 10 and (r["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (r["profit_factor"] or 0) > 1 and (r["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0)
    val = {(r["window"], r["pair"], r["horizon"], r["session"]): r for r in rows if r["validation_pass"]}
    target, candidates = v1["target"], []
    for key, r64 in val.items():
        if key[0] != "64D":
            continue
        r32 = val.get(("32D", *key[1:]))
        if r32 is None:
            continue
        pair = r64["pair"]
        notional = avg(bars[pair if pair.endswith("_JPY") else "EUR_JPY"], holdout-timedelta(days=64), holdout) * 1000
        scale = math.floor(target["margin_cap_jpy"] / (notional * target["margin_rate"]))
        trades30 = r64["trades"] * 30 / (64*.4 - 1/24)
        net, lcb, dd = r64["expectancy_jpy_per_1000u"]*trades30*scale, r64["bootstrap_lcb_expectancy_jpy"]*trades30*scale, r64["max_drawdown_jpy_per_1000u"]*scale
        candidates.append({"key": key, "validation_32d": r32, "validation_64d": r64, "max_units": scale*1000, "projected_30d_net_jpy": net, "projected_30d_lcb_jpy": lcb, "scaled_dd_jpy": dd, "monthly_3x_pass": net >= 400000 and lcb >= 400000 and dd <= 80000})
    candidates.sort(key=lambda c: c["projected_30d_lcb_jpy"], reverse=True)
    report = {"contract": prereg["contract"], "holdout_used": False, "decision_count": sum(len(v) for v in cache.values()), "train_plateau_count": len(plateau), "stable_32d_64d_count": len(candidates), "monthly_3x_pass_count": sum(c["monthly_3x_pass"] for c in candidates), "best_candidate": candidates[0] if candidates else None, "conclusion": "MONTHLY_3X_PROVED" if any(c["monthly_3x_pass"] for c in candidates) else "MONTHLY_3X_NOT_PROVED"}
    (HERE / "family_fusion_grid_v5.jsonl").write_text("".join(json.dumps(r, sort_keys=True)+"\n" for r in rows))
    (HERE / "family_fusion_report_v5.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report))

if __name__ == "__main__":
    main()
