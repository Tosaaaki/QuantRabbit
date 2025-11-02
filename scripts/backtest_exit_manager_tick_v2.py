#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tick / ≤250ms バックテスト (v2)
--------------------------------
Grace → EventBudget → Hazard → UpperBound に加え、TP/SL とスクラッチ抑制、
ハザード係数の調整をサポートする。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml


def _default_presets_path() -> Path:
    return Path(__file__).resolve().with_name("tuning_presets.yaml")


def load_preset(name: Optional[str], file_path: Optional[str]) -> Dict[str, float]:
    if not name:
        return {}
    preset_path = Path(file_path) if file_path else _default_presets_path()
    if not preset_path.exists():
        raise FileNotFoundError(f"preset file not found: {preset_path}")
    data = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    presets = data.get("presets", {})
    if name not in presets:
        raise KeyError(f"preset '{name}' not found in {preset_path}")
    return dict(presets[name])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", required=True, help="JSONL or CSV (timestamp,bid,ask[,spread])")
    ap.add_argument("--entries", help="Optional entries file (timestamp,direction,price)")
    ap.add_argument("--symbol", default="USD_JPY")
    ap.add_argument("--pip-scale", type=float, default=100.0)
    ap.add_argument("--spread-col", default="spread")
    ap.add_argument("--latency-ms", type=float, default=120.0)
    ap.add_argument("--grace-ms", type=int, default=200)
    ap.add_argument("--min-grace-before-scratch-ms", type=int, default=0)
    ap.add_argument("--scratch-requires-events", type=int, default=0)
    ap.add_argument("--tp-pips", type=float, default=None)
    ap.add_argument("--sl-pips", type=float, default=None)
    ap.add_argument("--event-budget", type=int, default=14)
    ap.add_argument("--hazard-debounce", type=int, default=2)
    ap.add_argument("--hazard-cost-spread-base", type=float, default=0.3)
    ap.add_argument("--hazard-cost-latency-base-ms", type=float, default=300.0)
    ap.add_argument("--upper-bound-sec", type=float, default=4.2)
    ap.add_argument("--timeout-soft-tp-frac", type=float, default=0.7,
                    help="soft TP triggers if elapsed >= frac*upper_bound and pips >= soft-tp-pips")
    ap.add_argument("--soft-tp-pips", type=float, default=0.3,
                    help="pips threshold for soft TP before hard timeout")
    ap.add_argument("--momentum-win-ms", type=int, default=400)
    ap.add_argument("--auto-entry", action="store_true", help="generate momentum-based entries if not provided")
    ap.add_argument("--preset", help="preset name defined in tuning_presets.yaml (or --preset-file)")
    ap.add_argument("--preset-file", help="optional path to preset yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Apply preset overrides (only if CLI arguments kept default)
    preset_values = load_preset(args.preset, args.preset_file)
    for key, value in preset_values.items():
        cli_value = getattr(args, key, None)
        if cli_value == ap.get_default(key):
            setattr(args, key, value)
    return args


def load_ticks(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
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
            raise ValueError("need bid/ask or mid columns")
    if "spread" not in df.columns:
        df["spread"] = df["ask"] - df["bid"]
    return df


def gen_entries(df: pd.DataFrame, pip_scale: float, mom_ms: int) -> pd.DataFrame:
    s = df.copy()
    s["mid"] = (s["bid"] + s["ask"]) / 2.0
    resampled = s.set_index("timestamp")["mid"].resample("250ms").last().ffill()
    diff = resampled.diff(int(mom_ms / 250)).fillna(0)
    threshold = diff.rolling(200).std().median() * 0.8
    signals = diff.abs() > threshold
    entries: List[Dict[str, object]] = []
    for ts, trigger in signals.items():
        if not trigger:
            continue
        direction = "long" if diff.loc[ts] > 0 else "short"
        idx = s["timestamp"].searchsorted(ts)
        idx = max(0, min(len(s) - 1, idx))
        row = s.iloc[idx]
        price = row["ask"] if direction == "long" else row["bid"]
        entries.append({"timestamp": ts, "direction": direction, "price": price})
    return pd.DataFrame(entries).drop_duplicates(subset=["timestamp"])


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def run_exit(
    ticks: pd.DataFrame,
    entries: pd.DataFrame,
    pip_scale: float,
    grace_ms: int,
    min_grace_before_scratch: int,
    scratch_requires_events: int,
    tp_pips: Optional[float],
    sl_pips: Optional[float],
    event_budget: int,
    hazard_debounce: int,
    timeout_soft_tp_frac: float,
    soft_tp_pips: float,
    hazard_cost_spread_base: float,
    hazard_cost_latency_base: float,
    upper_bound_sec: float,
    latency_ms: float,
) -> pd.DataFrame:
    out = []
    ticks = ticks.copy()
    ticks["mid"] = (ticks["bid"] + ticks["ask"]) / 2.0
    ticks["tick"] = 1
    ticks["tick_rate"] = ticks["tick"].rolling(4).sum()

    for _, entry in entries.iterrows():
        entry_ts = pd.Timestamp(entry["timestamp"])
        direction = entry["direction"]
        side = 1 if direction == "long" else -1
        future = ticks[ticks["timestamp"] >= entry_ts].head(400)
        if future.empty:
            continue
        entry_row = future.iloc[0]
        entry_price = entry_row["ask"] if side > 0 else entry_row["bid"]
        events = 0
        hazard_ticks = 0
        grace_used_ms = 0.0
        scratch_hits = 0
        hazard_reason = None

        for _, row in future.iterrows():
            now = row["timestamp"]
            elapsed_ms = (now - entry_ts).total_seconds() * 1000.0
            elapsed_sec = elapsed_ms / 1000.0
            bid, ask = row["bid"], row["ask"]
            spread = row["spread"]
            tick_rate = float(row.get("tick_rate", 8.0))

            price = bid if side < 0 else ask
            pnl_pips = (price - entry_price) * pip_scale * side

            prev_mid = ticks.iloc[row.name - 1]["mid"] if row.name > 0 else row["mid"]
            mom = (row["mid"] - prev_mid) * pip_scale
            imb = math.copysign(min(1.0, abs(mom) / 0.25), mom) if mom else 0.0
            health = 0.5 + 0.4 * math.tanh(mom / 0.3) - 0.3 * (spread / 0.0003)

            if elapsed_ms < grace_ms:
                grace_used_ms = elapsed_ms
                continue

            # Soft TP before hard timeout
            if (
                elapsed_sec >= timeout_soft_tp_frac * upper_bound_sec
                and pnl_pips >= soft_tp_pips
            ):
                out.append(
                    dict(
                        entry_ts=entry_ts,
                        exit_ts=now,
                        reason="soft_tp_timeout",
                        pips=float(pnl_pips),
                        events=events,
                        grace_used_ms=grace_used_ms,
                        hazard_ticks=hazard_ticks,
                        scratch_hits=scratch_hits,
                    )
                )
                break

            if (
                elapsed_ms >= max(grace_ms, min_grace_before_scratch)
                and events >= scratch_requires_events
            ):
                scratch_conditions = 0
                if mom * side < -0.08:
                    scratch_conditions += 1
                if imb * side < 0.15:
                    scratch_conditions += 1
                if pnl_pips < -0.1:
                    scratch_conditions += 1
                if scratch_conditions >= 2:
                    scratch_hits += 1
                    out.append(
                        dict(
                            entry_ts=entry_ts,
                            exit_ts=now,
                            reason="scratch",
                            pips=float(pnl_pips),
                            events=events,
                            grace_used_ms=grace_used_ms,
                            hazard_ticks=hazard_ticks,
                            scratch_hits=scratch_hits,
                        )
                    )
                    break

            if tp_pips is not None and pnl_pips >= tp_pips:
                out.append(
                    dict(
                        entry_ts=entry_ts,
                        exit_ts=now,
                        reason="tp_hit",
                        pips=float(pnl_pips),
                        events=events,
                        grace_used_ms=grace_used_ms,
                        hazard_ticks=hazard_ticks,
                        scratch_hits=scratch_hits,
                    )
                )
                break

            if sl_pips is not None and pnl_pips <= -sl_pips:
                out.append(
                    dict(
                        entry_ts=entry_ts,
                        exit_ts=now,
                        reason="sl_hit",
                        pips=float(pnl_pips),
                        events=events,
                        grace_used_ms=grace_used_ms,
                        hazard_ticks=hazard_ticks,
                        scratch_hits=scratch_hits,
                    )
                )
                break

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
                        scratch_hits=scratch_hits,
                    )
                )
                break

            p_tp = sigmoid(0.08 * side * mom - 4.0 * spread - 0.003 * latency_ms)
            p_sl = 1.0 - p_tp
            cost_k = 1.0 + (spread / max(hazard_cost_spread_base, 1e-6)) + (
                latency_ms / max(hazard_cost_latency_base, 1.0)
            )

            # optional gap threshold (difference between p_sl and p_tp) from presets
            gap = float(os.environ.get("HAZARD_GAP_THRESH", os.environ.get("hazard_gap_thresh", 0.0)) or 0.0)
            hazard_cond = (p_tp < p_sl * cost_k) or ((p_sl - p_tp) > gap)

            if hazard_cond:
                hazard_ticks += 1
                if hazard_ticks >= hazard_debounce:
                    hazard_reason = "hazard_exit"
                    out.append(
                        dict(
                            entry_ts=entry_ts,
                            exit_ts=now,
                            reason=hazard_reason,
                            pips=float(pnl_pips),
                            events=events,
                            grace_used_ms=grace_used_ms,
                            hazard_ticks=hazard_ticks,
                            scratch_hits=scratch_hits,
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
                        scratch_hits=scratch_hits,
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
                    scratch_hits=scratch_hits,
                )
            )

    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    ticks = load_ticks(args.ticks)

    if args.entries:
        ext = os.path.splitext(args.entries)[1].lower()
        if ext == ".jsonl":
            entries = pd.read_json(args.entries, lines=True)
        else:
            entries = pd.read_csv(args.entries)
        entries["timestamp"] = pd.to_datetime(entries["timestamp"])
    else:
        entries = gen_entries(ticks, args.pip_scale, args.momentum_win_ms)

    res = run_exit(
        ticks=ticks,
        entries=entries,
        pip_scale=args.pip_scale,
        grace_ms=args.grace_ms,
        min_grace_before_scratch=args.min_grace_before_scratch_ms,
        scratch_requires_events=args.scratch_requires_events,
        tp_pips=args.tp_pips,
        sl_pips=args.sl_pips,
        event_budget=args.event_budget,
        hazard_debounce=args.hazard_debounce,
        hazard_cost_spread_base=args.hazard_cost_spread_base,
        hazard_cost_latency_base=args.hazard_cost_latency_base_ms,
        upper_bound_sec=args.upper_bound_sec,
        timeout_soft_tp_frac=args.timeout_soft_tp_frac,
        soft_tp_pips=args.soft_tp_pips,
        latency_ms=args.latency_ms,
    )
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    print(f"[DONE] wrote {len(res)} rows to {out_path}")


if __name__ == "__main__":
    main()
