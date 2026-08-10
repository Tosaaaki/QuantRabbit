#!/usr/bin/env python3
"""One-axis exit-policy refinement for frozen independent strategy signals."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from run_strategy_expansion import (
    ROOT,
    crosses_financing,
    digest,
    in_session,
    load_bars,
    metrics,
    signal,
    utc,
)


HERE = Path(__file__).resolve().parent
PREREG = HERE / "strategy_exit_preregister_v1.json"
SOURCE_PREREG = HERE / "strategy_expansion_preregister_v1.json"


def arrays(bars: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    close = np.asarray([row["mid"]["c"] for row in bars])
    high = np.asarray([row["mid"]["h"] for row in bars])
    low = np.asarray([row["mid"]["l"] for row in bars])
    sma20 = np.full(len(close), np.nan)
    sma50 = np.full(len(close), np.nan)
    sma20[19:] = np.convolve(close, np.ones(20) / 20, mode="valid")
    sma50[49:] = np.convolve(close, np.ones(50) / 50, mode="valid")
    previous = np.roll(close, 1)
    previous[0] = close[0]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    atr14 = np.full(len(close), np.nan)
    atr14[13:] = np.convolve(true_range, np.ones(14) / 14, mode="valid")
    return {"close": close, "high": high, "low": low, "sma20": sma20, "sma50": sma50, "atr14": atr14}


def is_consecutive(times: list[datetime], start: int, end: int) -> bool:
    return start >= 0 and end < len(times) and times[end] - times[start] == timedelta(minutes=5 * (end - start))


def exit_open(bars: list[dict[str, Any]], index: int, side: str, slip: float) -> float:
    bid = float(bars[index]["bid"]["o"])
    ask = float(bars[index]["ask"]["o"])
    extra = slip * (ask - bid)
    return bid - extra if side == "LONG" else ask + extra


def stop_fill(bars: list[dict[str, Any]], index: int, side: str, stop: float, slip: float) -> float:
    bid = float(bars[index]["bid"]["o"])
    ask = float(bars[index]["ask"]["o"])
    extra = slip * (ask - bid)
    if side == "LONG":
        return (bid if bid < stop else stop) - extra
    return (ask if ask > stop else stop) + extra


def target_fill(bars: list[dict[str, Any]], index: int, side: str, target: float, slip: float) -> float:
    bid = float(bars[index]["bid"]["o"])
    ask = float(bars[index]["ask"]["o"])
    extra = slip * (ask - bid)
    if side == "LONG":
        return min(max(bid, target), target) - extra
    return max(min(ask, target), target) + extra


def replay_trade(
    bars: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    entry_index: int,
    side: str,
    exit_policy: str,
    slip: float = 0.5,
) -> tuple[int, float, str]:
    maximum = entry_index + 24
    entry_bid = float(bars[entry_index]["bid"]["o"])
    entry_ask = float(bars[entry_index]["ask"]["o"])
    entry_extra = slip * (entry_ask - entry_bid)
    entry = entry_ask + entry_extra if side == "LONG" else entry_bid - entry_extra
    atr = float(data["atr14"][entry_index - 1])
    direction = 1.0 if side == "LONG" else -1.0
    stop = entry - direction * atr
    if exit_policy == "BRACKET_1ATR_1_5ATR":
        target = entry + direction * 1.5 * atr
    elif exit_policy in {"BRACKET_1ATR_2ATR", "BE_AFTER_1ATR_TP2ATR"}:
        target = entry + direction * 2.0 * atr
    else:
        target = None
    pending_stop: float | None = None
    pending_open_exit = False
    adverse_slope_streak = 0

    for index in range(entry_index + 1, maximum + 1):
        if pending_open_exit:
            return index, exit_open(bars, index, side, slip), "COMPLETED_BAR_EXIT"
        if pending_stop is not None:
            stop = max(stop, pending_stop) if side == "LONG" else min(stop, pending_stop)
            pending_stop = None
        if index == maximum or exit_policy == "TIME_24":
            if exit_policy == "TIME_24" and index < maximum:
                continue
            return index, exit_open(bars, index, side, slip), "TIME_EXIT"

        if side == "LONG":
            stop_touched = float(bars[index]["bid"]["l"]) <= stop
            target_touched = target is not None and float(bars[index]["bid"]["h"]) >= target
        else:
            stop_touched = float(bars[index]["ask"]["h"]) >= stop
            target_touched = target is not None and float(bars[index]["ask"]["l"]) <= target
        if stop_touched:  # conservative STOP_FIRST if both touched
            return index, stop_fill(bars, index, side, stop, slip), "STOP_FIRST_TOUCH"
        if target_touched and target is not None:
            return index, target_fill(bars, index, side, target, slip), "TARGET_FIRST_TOUCH"

        if exit_policy == "BE_AFTER_1ATR_TP2ATR":
            favorable = float(bars[index]["bid"]["h"]) if side == "LONG" else float(bars[index]["ask"]["l"])
            if direction * (favorable - entry) >= atr:
                pending_stop = entry
        elif exit_policy == "ATR_TRAIL":
            completed = float(bars[index]["bid"]["c"]) if side == "LONG" else float(bars[index]["ask"]["c"])
            if direction * (completed - entry) >= atr:
                candidate = completed - direction * 1.5 * atr
                pending_stop = candidate
        elif exit_policy == "SMA_DETERIORATION":
            slope = float(data["sma20"][index] - data["sma20"][index - 1])
            adverse_slope_streak = adverse_slope_streak + 1 if direction * slope < 0 else 0
            if adverse_slope_streak >= 3:
                pending_open_exit = True
        elif exit_policy == "STRUCTURE_BREAK":
            completed = float(data["close"][index])
            if side == "LONG" and completed < float(data["low"][index - 6:index].min()):
                pending_open_exit = True
            elif side == "SHORT" and completed > float(data["high"][index - 6:index].max()):
                pending_open_exit = True
    raise AssertionError("unreachable")


def replay_config(
    bars: list[dict[str, Any]],
    data: dict[str, np.ndarray],
    family: str,
    lookback: int,
    session: str,
    exit_policy: str,
) -> list[dict[str, Any]]:
    times = [row["time"] for row in bars]
    trades: list[dict[str, Any]] = []
    next_free = 0
    start = max(50, lookback) + 1
    for index in range(start, len(bars) - 26):
        entry_index = index + 1
        maximum = entry_index + 24
        if entry_index < next_free or not in_session(times[entry_index], session):
            continue
        if not is_consecutive(times, index - max(50, lookback), maximum):
            continue
        side = signal(family, index, lookback, data["close"], data["high"], data["low"], data["sma20"], data["sma50"])
        if side is None or not math.isfinite(float(data["atr14"][index])) or data["atr14"][index] <= 0:
            continue
        if crosses_financing(times[entry_index], times[maximum]):
            continue
        exit_index, exit_price, terminal = replay_trade(bars, data, entry_index, side, exit_policy)
        entry_bid = float(bars[entry_index]["bid"]["o"])
        entry_ask = float(bars[entry_index]["ask"]["o"])
        entry_extra = 0.5 * (entry_ask - entry_bid)
        entry_price = entry_ask + entry_extra if side == "LONG" else entry_bid - entry_extra
        pnl = (exit_price - entry_price) * 1000.0 if side == "LONG" else (entry_price - exit_price) * 1000.0
        trades.append({"entry_time": times[entry_index], "exit_time": times[exit_index], "pnl": pnl, "terminal": terminal})
        next_free = exit_index + 1
    return trades


def main() -> None:
    prereg = json.loads(PREREG.read_text())
    source_prereg = json.loads(SOURCE_PREREG.read_text())
    holdout = utc(prereg["split_and_gates"]["holdout_start_utc"])
    sources: dict[str, list[dict[str, Any]]] = {}
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for pair, source in source_prereg["sources"].items():
        path = ROOT / source["path"]
        if digest(path) != source["sha256"]:
            raise SystemExit(f"source SHA mismatch: {path}")
        sources[pair] = load_bars(path, holdout)
        prepared[pair] = arrays(sources[pair])

    cache: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for pair, bars in sources.items():
        for family in prereg["signal_grid"]["families"]:
            for lookback in prereg["signal_grid"]["lookback"]:
                for session in prereg["signal_grid"]["entry_session_utc"]:
                    for exit_policy in prereg["exit_policies"]:
                        key = (pair, family, int(lookback), session, exit_policy)
                        cache[key] = replay_config(bars, prepared[pair], family, int(lookback), session, exit_policy)

    rows: list[dict[str, Any]] = []
    for days in prereg["split_and_gates"]["windows_days"]:
        start = holdout - timedelta(days=days)
        train_end = start + timedelta(days=days * prereg["split_and_gates"]["train_fraction"])
        validation_start = train_end + timedelta(hours=prereg["split_and_gates"]["embargo_hours"])
        for key, trades in cache.items():
            for split, left, right in (("TRAIN", start, train_end), ("VALIDATION", validation_start, holdout)):
                selected = [trade for trade in trades if left <= trade["entry_time"] and trade["exit_time"] < right]
                values = [float(trade["pnl"]) for trade in selected]
                terminals: dict[str, int] = {}
                for trade in selected:
                    terminals[trade["terminal"]] = terminals.get(trade["terminal"], 0) + 1
                rows.append({
                    "window": f"{days}D", "split": split, "pair": key[0], "family": key[1],
                    "lookback": key[2], "entry_session_utc": key[3], "exit_policy": key[4],
                    "terminal_counts": terminals,
                    **metrics(values, ":".join(map(str, (days, split, *key)))),
                })

    plateaus: set[tuple[str, str, str, str, str]] = set()
    for days in prereg["split_and_gates"]["windows_days"]:
        window = f"{days}D"
        for pair in sources:
            for family in prereg["signal_grid"]["families"]:
                for session in prereg["signal_grid"]["entry_session_utc"]:
                    for exit_policy in prereg["exit_policies"]:
                        passed = [row for row in rows if row["window"] == window and row["split"] == "TRAIN" and row["pair"] == pair and row["family"] == family and row["entry_session_utc"] == session and row["exit_policy"] == exit_policy and row["trades"] >= 20 and (row["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (row["profit_factor"] or 0) > 1 and (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0]
                        if {row["lookback"] for row in passed} == {24, 48}:
                            plateaus.add((window, pair, family, session, exit_policy))

    for row in rows:
        key = (row["window"], row["pair"], row["family"], row["entry_session_utc"], row["exit_policy"])
        row["train_connected_plateau"] = key in plateaus
        row["validation_pass"] = bool(row["split"] == "VALIDATION" and row["train_connected_plateau"] and row["trades"] >= 10 and (row["expectancy_jpy_per_1000u"] or -math.inf) > 0 and (row["profit_factor"] or 0) > 1 and (row["bootstrap_lcb_expectancy_jpy"] or -math.inf) > 0)

    passed = [row for row in rows if row["validation_pass"]]
    report = {
        "contract": prereg["contract"], "preregister_sha256": digest(PREREG),
        "holdout_used": False, "grid_rows": len(rows), "train_plateaus": len(plateaus),
        "validation_pass_count": len(passed), "validation_pass_rows": passed,
        "conclusion": "EXIT_MANAGED_EDGE_FOUND" if passed else "EXIT_MANAGEMENT_NOT_YET_STABLE",
    }
    (HERE / "strategy_exit_grid_v1.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    (HERE / "strategy_exit_report_v1.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest = {"contract": prereg["contract"], "preregister_sha256": digest(PREREG), "outputs": {name: digest(HERE / name) for name in ("strategy_exit_grid_v1.jsonl", "strategy_exit_report_v1.json")}}
    (HERE / "strategy_exit_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
