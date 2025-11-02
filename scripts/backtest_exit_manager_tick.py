#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tick/≤250msバーでの低ボラExit検証ツール

入力:
  --ticks: JSONL or CSV (timestamp,bid,ask[,spread,volume])
  --entries: (optional) CSV/JSONL of precomputed entries (timestamp,direction,long|short,price)
出力:
  --out: CSV per-trade (entry_ts, exit_ts, reason, pips, events, grace_used_ms, hazard_ticks, ...)

Exit順序: GRACE -> EVENT_BUDGET -> HAZARD(debounce) -> UPPER_BOUND
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", required=True, help="JSONL or CSV of ticks (timestamp,bid,ask[,spread])")
    ap.add_argument("--entries", default=None, help="optional entries file (timestamp,direction,price)")
    ap.add_argument("--symbol", default="USD_JPY")
    ap.add_argument("--pip-scale", type=float, default=100.0)
    ap.add_argument("--spread-col", default="spread")
    ap.add_argument("--latency-ms", type=float, default=120.0)
    ap.add_argument("--grace-ms", type=int, default=200)
    ap.add_argument("--event-budget", type=int, default=14)
    ap.add_argument("--hazard-debounce", type=int, default=2)
    ap.add_argument("--upper-bound-sec", type=float, default=4.2)
    ap.add_argument("--momentum-win-ms", type=int, default=400)
    ap.add_argument("--imbalance-win-ms", type=int, default=400)
    ap.add_argument("--auto-entry", action="store_true", help="generate entries by micro momentum if --entries not given")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_ticks(path: str) -> pd.DataFrame:
    infer = os.path.splitext(path)[1].lower()
    if infer == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("ticks require timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "bid" not in df.columns or "ask" not in df.columns:
        if "mid" in df.columns:
            df["bid"] = df["mid"]
            df["ask"] = df["mid"]
        else:
            raise ValueError("need bid/ask or mid")
    if "spread" not in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    return df


def gen_entries(df: pd.DataFrame, pip_scale: float, mom_ms: int) -> pd.DataFrame:
    s = df.copy()
    s["mid"] = (s["bid"] + s["ask"]) / 2.0
    sr = s.set_index("timestamp")["mid"].resample("250ms").last().ffill()
    roll = sr.diff(int(mom_ms / 250)).fillna(0)
    th = roll.rolling(200).std().median() * 0.8
    signals = roll.abs() > th
    entries = []
    for t, flag in signals.items():
        if not flag:
            continue
        direction = "long" if roll.loc[t] > 0 else "short"
        near_idx = s["timestamp"].searchsorted(t)
        near_idx = max(0, min(len(s) - 1, near_idx))
        row = s.iloc[near_idx]
        price = row["ask"] if direction == "long" else row["bid"]
        entries.append({"timestamp": t, "direction": direction, "price": price})
    return pd.DataFrame(entries).drop_duplicates(subset=["timestamp"])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def run_exit(
    df_ticks: pd.DataFrame,
    df_entries: pd.DataFrame,
    pip_scale: float,
    grace_ms: int,
    event_budget: int,
    hazard_debounce: int,
    upper_bound_sec: float,
    latency_ms: float,
) -> pd.DataFrame:
    out = []
    tix = df_ticks.copy()
    tix["mid"] = (tix["bid"] + tix["ask"]) / 2.0
    tix["tick"] = 1
    tix["tick_rate"] = tix["tick"].rolling(4).sum()

    for _, e in df_entries.iterrows():
        entry_ts = pd.Timestamp(e["timestamp"])
        direction = e["direction"]
        side = 1 if direction == "long" else -1
        future = tix[tix["timestamp"] >= entry_ts].head(200)
        if future.empty:
            continue
        entry_row = future.iloc[0]
        entry_price = entry_row["ask"] if side > 0 else entry_row["bid"]
        events = 0
        hazard_ticks = 0
        grace_used_ms = 0.0

        for _, row in future.iterrows():
            now = row["timestamp"]
            elapsed_ms = (now - entry_ts).total_seconds() * 1000.0
            elapsed_sec = elapsed_ms / 1000.0
            bid, ask = row["bid"], row["ask"]
            mid = row["mid"]
            spread = row["spread"]
            tick_rate = float(row.get("tick_rate", 8.0))
            price = bid if side < 0 else ask
            pnl_pips = (price - entry_price) * pip_scale * side

            prev_mid = mid
            if row.name > 0:
                prev_mid = tix.iloc[row.name - 1]["mid"]
            mom = float((mid - prev_mid) * pip_scale)
            imb = math.copysign(min(1.0, abs(mom) / 0.2), mom) if mom else 0.0
            health = 0.5 + 0.4 * math.tanh(mom / 0.3) - 0.3 * (spread / (0.0003 if pip_scale == 100000 else 0.03))

            if elapsed_ms < grace_ms:
                grace_used_ms = elapsed_ms
                if (mom * side) < 0.05 or (imb * side) < 0.2:
                    if abs(pnl_pips) <= 0.3:
                        out.append(
                            dict(
                                entry_ts=entry_ts,
                                exit_ts=now,
                                reason="scratch",
                                pips=float(pnl_pips),
                                events=events,
                                grace_used_ms=grace_used_ms,
                                hazard_ticks=hazard_ticks,
                            )
                        )
                        break
                continue

            events += 1
            if events >= event_budget and health < 0.2:
                out.append(
                    dict(
                        entry_ts=entry_ts,
                        exit_ts=now,
                        reason="event_budget_timeout",
                        pips=float(pnl_pips),
                        events=events,
                        grace_used_ms=grace_used_ms,
                        hazard_ticks=hazard_ticks,
                    )
                )
                break

            p_tp = sigmoid(0.08 * side * mom - 4.0 * spread - 0.003 * latency_ms)
            p_sl = 1.0 - p_tp
            cost_k = 1.0 + (spread / max(0.0003, 0.0003)) + (latency_ms / 300.0)

            if p_tp < p_sl * cost_k:
                hazard_ticks += 1
                if hazard_ticks >= hazard_debounce:
                    out.append(
                        dict(
                            entry_ts=entry_ts,
                            exit_ts=now,
                            reason="hazard_exit",
                            pips=float(pnl_pips),
                            events=events,
                            grace_used_ms=grace_used_ms,
                            hazard_ticks=hazard_ticks,
                        )
                    )
                    break
            else:
                hazard_ticks = 0

            if elapsed_sec > upper_bound_sec:
                out.append(
                    dict(
                        entry_ts=entry_ts,
                        exit_ts=now,
                        reason="hard_timeout",
                        pips=float(pnl_pips),
                        events=events,
                        grace_used_ms=grace_used_ms,
                        hazard_ticks=hazard_ticks,
                    )
                )
                break
        else:
            out.append(
                dict(
                    entry_ts=entry_ts,
                    exit_ts=future.iloc[-1]["timestamp"],
                    reason="window_end",
                    pips=float(pnl_pips),
                    events=events,
                    grace_used_ms=grace_used_ms,
                    hazard_ticks=hazard_ticks,
                )
            )

    return pd.DataFrame(out)


def main():
    args = parse_args()
    ticks = load_ticks(args.ticks)

    if args.entries:
        infer = os.path.splitext(args.entries)[1].lower()
        if infer == ".jsonl":
            entries = pd.read_json(args.entries, lines=True)
        else:
            entries = pd.read_csv(args.entries)
        entries["timestamp"] = pd.to_datetime(entries["timestamp"])
    else:
        entries = gen_entries(ticks, args.pip_scale, args.momentum_win_ms)

    res = run_exit(
        ticks,
        entries,
        args.pip_scale,
        args.grace_ms,
        args.event_budget,
        args.hazard_debounce,
        args.upper_bound_sec,
        args.latency_ms,
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
