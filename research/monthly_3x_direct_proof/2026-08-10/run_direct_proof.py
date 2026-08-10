#!/usr/bin/env python3
"""Direct 3x proof gate across frozen price-action and exit families."""

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


PREREG = HERE / "preregister_v1.json"


def avg_price(bars: list[dict[str, Any]], left, right) -> float:
    values = [float(row["mid"]["c"]) for row in bars if left <= row["time"] < right]
    return float(np.mean(values)) if values else math.nan


def pnl_to_jpy(pair: str, pnl: float, exit_time, eurjpy: dict, eurusd: dict) -> float | None:
    if pair.endswith("_JPY"):
        return pnl
    if pair != "EUR_USD":
        return None
    # Deterministic nearest completed M5 cross, never after the exit timestamp.
    cross_row = eurjpy.get(exit_time)
    usd_row = eurusd.get(exit_time)
    if cross_row is None or usd_row is None:
        return None
    eurjpy_mid = float(cross_row["mid"]["o"])
    eurusd_mid = float(usd_row["mid"]["o"])
    if eurusd_mid <= 0:
        return None
    return pnl * (eurjpy_mid / eurusd_mid) * 0.999


def adjacent_plateau(passed: set[int], lookbacks: list[int]) -> bool:
    return any(a in passed and b in passed for a, b in zip(lookbacks, lookbacks[1:]))


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    holdout = utc(prereg["split"]["holdout_start_utc"])
    bars: dict[str, list[dict[str, Any]]] = {}
    prepared = {}
    for pair, source in prereg["sources"].items():
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise SystemExit(f"source SHA mismatch: {path}")
        bars[pair] = load_bars(path, holdout)
        prepared[pair] = arrays(bars[pair])

    grid = prereg["signal_grid"]
    eurjpy_index = {row["time"]: row for row in bars["EUR_JPY"]}
    eurusd_index = {row["time"]: row for row in bars["EUR_USD"]}
    cache: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for pair in bars:
        for family in grid["families"]:
            for lookback in grid["lookback"]:
                for session in grid["entry_session_utc"]:
                    for exit_policy in grid["exit_policies"]:
                        key = (pair, family, int(lookback), session, exit_policy)
                        raw = replay_config(bars[pair], prepared[pair], family, int(lookback), session, exit_policy)
                        converted = []
                        for trade in raw:
                            value = pnl_to_jpy(pair, float(trade["pnl"]), trade["exit_time"], eurjpy_index, eurusd_index)
                            if value is not None:
                                converted.append({**trade, "pnl_jpy_per_1000u": value})
                        cache[key] = converted

    rows: list[dict[str, Any]] = []
    target = prereg["target"]
    for days in prereg["split"]["windows_days"]:
        start = holdout - timedelta(days=days)
        train_end = start + timedelta(days=days * prereg["split"]["train_fraction"])
        val_start = train_end + timedelta(hours=prereg["split"]["embargo_hours"])
        for key, trades in cache.items():
            pair, family, lookback, session, exit_policy = key
            for split, left, right in (("TRAIN", start, train_end), ("VALIDATION", val_start, holdout)):
                selected = [row for row in trades if left <= row["entry_time"] and row["exit_time"] < right]
                values = [float(row["pnl_jpy_per_1000u"]) for row in selected]
                result = metrics(values, ":".join(map(str, (days, split, *key))))
                rows.append({
                    "window": f"{days}D",
                    "split": split,
                    "pair": pair,
                    "family": family,
                    "lookback": lookback,
                    "session": session,
                    "exit_policy": exit_policy,
                    **result,
                })

    plateau_keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row["split"] != "TRAIN":
            continue
        if row["trades"] < 20 or (row["expectancy_jpy_per_1000u"] or -math.inf) <= 0:
            continue
        if (row["profit_factor"] or 0) <= 1 or (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) <= 0:
            continue
        plateau_keys.add((row["window"], row["pair"], row["family"], row["session"], row["exit_policy"], row["lookback"]))

    stable_families: set[tuple[str, str, str, str, str]] = set()
    for window in {row["window"] for row in rows}:
        for pair in bars:
            for family in grid["families"]:
                for session in grid["entry_session_utc"]:
                    for exit_policy in grid["exit_policies"]:
                        passed = {
                            lookback for lookback in grid["lookback"]
                            if (window, pair, family, session, exit_policy, lookback) in plateau_keys
                        }
                        if adjacent_plateau(passed, grid["lookback"]):
                            stable_families.add((window, pair, family, session, exit_policy))

    for row in rows:
        family_key = (row["window"], row["pair"], row["family"], row["session"], row["exit_policy"])
        row["train_plateau"] = family_key in stable_families
        row["validation_pass"] = bool(
            row["split"] == "VALIDATION"
            and row["train_plateau"]
            and row["trades"] >= 10
            and (row["expectancy_jpy_per_1000u"] or -math.inf) > 0
            and (row["profit_factor"] or 0) > 1
            and (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0
        )

    validation_index = {
        (row["window"], row["pair"], row["family"], row["lookback"], row["session"], row["exit_policy"]): row
        for row in rows if row["split"] == "VALIDATION"
    }
    candidates = []
    for key, row64 in validation_index.items():
        if key[0] != "64D" or not row64["validation_pass"]:
            continue
        key32 = ("32D", *key[1:])
        row32 = validation_index.get(key32)
        if row32 is None or not row32["validation_pass"]:
            continue
        pair = row64["pair"]
        val_days = 64 * (1 - prereg["split"]["train_fraction"]) - prereg["split"]["embargo_hours"] / 24
        trades_30d = float(row64["trades"]) * 30.0 / val_days
        if pair == "AUD_JPY":
            notional_jpy_per_1000 = avg_price(bars[pair], holdout - timedelta(days=64), holdout) * 1000
        else:
            notional_jpy_per_1000 = avg_price(bars["EUR_JPY"], holdout - timedelta(days=64), holdout) * 1000
        max_scale = math.floor(target["margin_cap_jpy"] / (notional_jpy_per_1000 * target["margin_rate"]))
        monthly_net = float(row64["expectancy_jpy_per_1000u"]) * trades_30d * max_scale
        monthly_lcb = float(row64["bootstrap_lcb_expectancy_jpy"]) * trades_30d * max_scale
        scaled_dd = float(row64["max_drawdown_jpy_per_1000u"]) * max_scale
        candidates.append({
            "pair": pair,
            "family": row64["family"],
            "lookback": row64["lookback"],
            "session": row64["session"],
            "exit_policy": row64["exit_policy"],
            "validation_32d": row32,
            "validation_64d": row64,
            "trades_per_30d": trades_30d,
            "max_units": max_scale * 1000,
            "margin_cap_jpy": target["margin_cap_jpy"],
            "projected_30d_net_jpy": monthly_net,
            "projected_30d_lcb_jpy": monthly_lcb,
            "scaled_realized_dd_jpy": scaled_dd,
            "target_3x_pass": monthly_net >= target["required_profit_jpy"] and monthly_lcb >= target["required_profit_jpy"] and scaled_dd <= target["max_realized_drawdown_jpy"],
        })

    candidates.sort(key=lambda row: (row["projected_30d_lcb_jpy"], row["projected_30d_net_jpy"]), reverse=True)
    passes = [row for row in candidates if row["target_3x_pass"]]
    report = {
        "contract": prereg["contract"],
        "preregister_sha256": digest(PREREG),
        "holdout_used": False,
        "grid_rows": len(rows),
        "stable_multiwindow_candidate_count": len(candidates),
        "monthly_3x_pass_count": len(passes),
        "monthly_3x_passes": passes,
        "best_candidate": candidates[0] if candidates else None,
        "conclusion": "MONTHLY_3X_PROVED" if passes else "MONTHLY_3X_NOT_PROVED",
    }
    (HERE / "grid_v1.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    (HERE / "report_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
