#!/usr/bin/env python3
"""Cross-sectional 28-pair currency rotation; research-only and holdout sealed."""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "research/monthly_3x_growth_engine/2026-08-10"
sys.path.insert(0, str(BASE))
from run_strategy_expansion import crosses_financing, digest, load_bars, metrics, utc  # noqa: E402

PREREG = HERE / "preregister_v6_currency_rotation.json"
SOURCE_DIR = ROOT / "logs/replay/oanda_history/20260715T115624Z"


def source_files():
    return sorted(SOURCE_DIR.glob("*/*_M5_BA_20260311T000000Z_20260715T110000Z.jsonl.gz"))


def manifest_hash(paths):
    rows = [f"{path.parent.name}:{digest(path)}" for path in paths]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def pair_for(c1, c2, pairs):
    direct, reverse = f"{c1}_{c2}", f"{c2}_{c1}"
    if direct in pairs:
        return direct, "LONG"
    if reverse in pairs:
        return reverse, "SHORT"
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", default=str(PREREG))
    parser.add_argument("--output-prefix", default="currency_rotation_v6")
    args = parser.parse_args()
    prereg = json.loads(Path(args.preregister).read_text())
    paths = source_files()
    if len(paths) != prereg["source_count"] or manifest_hash(paths) != prereg["sorted_pair_sha_manifest_sha256"]:
        raise SystemExit("28-pair source manifest mismatch")
    holdout = utc(prereg["holdout_start_utc"])
    bars = {path.parent.name: load_bars(path, holdout) for path in paths}
    maps = {pair: {row["time"]: row for row in rows} for pair, rows in bars.items()}
    common = set.intersection(*(set(index) for index in maps.values()))
    times = sorted(common)
    if not times:
        raise SystemExit("no common decision-time cohort")
    pairs = sorted(bars)
    closes = {pair: np.asarray([float(maps[pair][t]["mid"]["c"]) for t in times]) for pair in pairs}
    currencies = sorted({currency for pair in pairs for currency in pair.split("_")})
    jpy_pairs = {currency: f"{currency}_JPY" for currency in currencies if currency != "JPY"}

    strength = {}
    for lookback in prereg["lookback_bars"]:
        values = []
        for index in range(len(times)):
            scores = defaultdict(list)
            if index >= lookback and times[index] - times[index-lookback] == timedelta(minutes=5*lookback):
                for pair in pairs:
                    base, quote = pair.split("_")
                    ret = math.log(closes[pair][index] / closes[pair][index-lookback])
                    scores[base].append(ret)
                    scores[quote].append(-ret)
            values.append({currency: float(np.mean(scores[currency])) for currency in currencies if len(scores[currency]) >= 4})
        strength[lookback] = values

    def quote_to_jpy(currency, index, pnl):
        if currency == "JPY":
            return 1.0
        row = maps[jpy_pairs[currency]][times[index]]
        return float(row["bid"]["o"] if pnl >= 0 else row["ask"]["o"])

    cache = {}
    for direction in prereg["directions"]:
        for lookback in prereg["lookback_bars"]:
            for horizon in prereg["horizon_bars"]:
                trades, next_free = [], 0
                for index in range(lookback, len(times)-horizon-1):
                    entry, exit_ = index+1, index+1+horizon
                    if entry < next_free or times[exit_] - times[index] != timedelta(minutes=5*(horizon+1)):
                        continue
                    scores = strength[lookback][index]
                    if len(scores) != len(currencies) or crosses_financing(times[entry], times[exit_]):
                        continue
                    strongest, weakest = max(scores, key=scores.get), min(scores, key=scores.get)
                    first, second = (strongest, weakest) if direction == "MOMENTUM" else (weakest, strongest)
                    pair, side = pair_for(first, second, set(pairs))
                    if pair is None:
                        continue
                    row_e, row_x = maps[pair][times[entry]], maps[pair][times[exit_]]
                    eb, ea, xb, xa = *(float(row_e[k]["o"]) for k in ("bid", "ask")), *(float(row_x[k]["o"]) for k in ("bid", "ask"))
                    entry_extra, exit_extra = .5*(ea-eb), .5*(xa-xb)
                    pnl_quote = ((xb-exit_extra)-(ea+entry_extra))*1000 if side == "LONG" else ((eb-entry_extra)-(xa+exit_extra))*1000
                    quote = pair.split("_")[1]
                    pnl_jpy = pnl_quote * quote_to_jpy(quote, exit_, pnl_quote)
                    base = pair.split("_")[0]
                    margin_notional = 1000 * quote_to_jpy(base, entry, 1.0)
                    trades.append({"entry_time": times[entry], "exit_time": times[exit_], "pair": pair, "side": side, "pnl_jpy_per_1000u": pnl_jpy, "notional_jpy_per_1000u": margin_notional, "score_gap": scores[strongest]-scores[weakest]})
                    next_free = exit_+1
                cache[(direction, lookback, horizon)] = trades

    rows = []
    for days in (16, 32, 64):
        start, train_end = holdout-timedelta(days=days), holdout-timedelta(days=days)+timedelta(days=days*.6)
        val_start = train_end+timedelta(hours=1)
        for key, trades in cache.items():
            for split, left, right in (("TRAIN", start, train_end), ("VALIDATION", val_start, holdout)):
                selected = [t for t in trades if left <= t["entry_time"] and t["exit_time"] < right]
                rows.append({"window": f"{days}D", "split": split, "direction": key[0], "lookback": key[1], "horizon": key[2], "max_notional_jpy_per_1000u": max((t["notional_jpy_per_1000u"] for t in selected), default=None), **metrics([t["pnl_jpy_per_1000u"] for t in selected], f"rotation:{days}:{split}:{key}")})
    train_pass = {(r["window"], r["direction"], r["lookback"], r["horizon"]) for r in rows if r["split"] == "TRAIN" and r["trades"] >= 30 and (r["expectancy_jpy_per_1000u"] or -math.inf)>0 and (r["profit_factor"] or 0)>1 and (r["bootstrap_lcb_expectancy_jpy"] or -math.inf)>0}
    plateau = set()
    for row in rows:
        base = (row["window"], row["direction"])
        points = {(lb,h) for lb in prereg["lookback_bars"] for h in prereg["horizon_bars"] if (*base,lb,h) in train_pass}
        stable = any(((a,b) in points and (c,b) in points) for a,c in zip(prereg["lookback_bars"], prereg["lookback_bars"][1:]) for b in prereg["horizon_bars"]) or any(((a,b) in points and (a,c) in points) for a in prereg["lookback_bars"] for b,c in zip(prereg["horizon_bars"], prereg["horizon_bars"][1:]))
        if stable:
            plateau.add(base)
    for r in rows:
        r["train_plateau"] = (r["window"],r["direction"]) in plateau
        r["validation_pass"] = bool(r["split"]=="VALIDATION" and r["train_plateau"] and r["trades"]>=20 and (r["expectancy_jpy_per_1000u"] or -math.inf)>0 and (r["profit_factor"] or 0)>1 and (r["bootstrap_lcb_expectancy_jpy"] or -math.inf)>0)
    val = {(r["window"],r["direction"],r["lookback"],r["horizon"]):r for r in rows if r["validation_pass"]}
    candidates=[]
    for key,r64 in val.items():
        if key[0]!="64D": continue
        r32=val.get(("32D",*key[1:]))
        if r32 is None: continue
        scale=math.floor(150000/(r64["max_notional_jpy_per_1000u"]*.04))
        trades30=r64["trades"]*30/(64*.4-1/24)
        net=r64["expectancy_jpy_per_1000u"]*trades30*scale
        lcb=r64["bootstrap_lcb_expectancy_jpy"]*trades30*scale
        dd=r64["max_drawdown_jpy_per_1000u"]*scale
        candidates.append({"key":key,"validation_32d":r32,"validation_64d":r64,"max_units":scale*1000,"projected_30d_net_jpy":net,"projected_30d_lcb_jpy":lcb,"scaled_dd_jpy":dd,"monthly_3x_pass":net>=400000 and lcb>=400000 and dd<=80000})
    candidates.sort(key=lambda c:c["projected_30d_lcb_jpy"],reverse=True)
    report={"contract":prereg["contract"],"source_count":len(paths),"common_m5_rows":len(times),"holdout_used":False,"decision_count":sum(len(v) for v in cache.values()),"train_plateau_count":len(plateau),"stable_32d_64d_count":len(candidates),"monthly_3x_pass_count":sum(c["monthly_3x_pass"] for c in candidates),"best_candidate":candidates[0] if candidates else None,"conclusion":"MONTHLY_3X_PROVED" if any(c["monthly_3x_pass"] for c in candidates) else "MONTHLY_3X_NOT_PROVED"}
    (HERE/f"{args.output_prefix}_grid.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    (HERE/f"{args.output_prefix}_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report))

if __name__=="__main__": main()
