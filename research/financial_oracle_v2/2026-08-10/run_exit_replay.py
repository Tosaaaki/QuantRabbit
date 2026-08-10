#!/usr/bin/env python3
"""Run bounded exit-policy diagnostics behind the strict V2 evidence gate."""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import build_path_metrics as bpm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ["BASELINE", "FIXED_BE", "ALL_COST_BE", "PARTIAL_TP_BE", "PURE_ATR_TRAIL", "SMA_DETERIORATION_TRAIL", "STRUCTURE_BREAK_EXIT", "TIME_EXIT"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(bpm.canonical_json(row) + "\n")


def load_s5(strict_rows: list[dict[str, Any]], prereg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in strict_rows:
        target[row["pair"]].append(row)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair, episodes in target.items():
        seen: set[int] = set()
        for source in prereg["source_boundary"]["s5_files"][pair]:
            with gzip.open(REPO / source["path"], "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    candle = json.loads(line)
                    stamp = bpm.parse_ns(candle["time"])
                    if stamp in seen:
                        continue
                    seen.add(stamp)
                    for episode in episodes:
                        start = bpm.parse_ns(episode["s5_full_interval_from_utc"])
                        end = bpm.parse_ns(episode["s5_full_interval_to_utc"])
                        if start <= stamp < end:
                            result[episode["trade_id"]].append({**candle, "ts_ns": stamp})
    for candles in result.values():
        candles.sort(key=lambda row: row["ts_ns"])
    return result


def load_m1() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    base = REPO / "research/continuous_thesis_monitor/2026-08-10/cache/derived"
    for pair in ("AUD_JPY", "EUR_JPY", "EUR_USD"):
        path = base / pair / f"{pair}_M1_BA_HOLDING_PREHOLDOUT.jsonl.gz"
        rows: list[dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append({**row, "end_ns": bpm.parse_ns(row["end_utc"])})
        rows.sort(key=lambda row: row["end_ns"])
        previous_close = None
        true_ranges: list[float] = []
        closes: list[float] = []
        for row in rows:
            high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
            tr = high - low if previous_close is None else max(high - low, abs(high - previous_close), abs(low - previous_close))
            true_ranges.append(tr)
            closes.append(close)
            index = len(closes) - 1
            row["atr14"] = sum(true_ranges[index - 13:index + 1]) / 14 if index >= 13 else None
            row["sma20"] = sum(closes[index - 19:index + 1]) / 20 if index >= 19 else None
            row["sma50"] = sum(closes[index - 49:index + 1]) / 50 if index >= 49 else None
            row["prior5_low"] = min(float(item["low"]) for item in rows[index - 5:index]) if index >= 5 else None
            row["prior5_high"] = max(float(item["high"]) for item in rows[index - 5:index]) if index >= 5 else None
            previous_close = close
        result[pair] = rows
    return result


def quote(candle: dict[str, Any], side: str) -> dict[str, float]:
    key = "bid" if side == "LONG" else "ask"
    return {name: float(candle[key][name]) for name in ("o", "h", "l", "c")}


def adverse_market_price(candle: dict[str, Any], side: str) -> tuple[float, float]:
    spread = float(candle["ask"]["o"]) - float(candle["bid"]["o"])
    half = spread / 2.0
    price = float(candle["bid"]["o"]) - half if side == "LONG" else float(candle["ask"]["o"]) + half
    return price, half


def adverse_stop_price(candle: dict[str, Any], side: str, stop: float) -> tuple[float, float]:
    spread = float(candle["ask"]["o"]) - float(candle["bid"]["o"])
    half = spread / 2.0
    opening = float(candle["bid"]["o"]) if side == "LONG" else float(candle["ask"]["o"])
    raw_price = min(stop, opening) if side == "LONG" else max(stop, opening)
    return (raw_price - half if side == "LONG" else raw_price + half), half


def price_pnl(entry: float, exit_price: float, units: int, side: str) -> float:
    return (exit_price - entry) * units if side == "LONG" else (entry - exit_price) * units


def feature_before(rows: list[dict[str, Any]], stamp: int) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["end_ns"] <= stamp]
    return eligible[-1] if eligible else None


def next_candle(candles: list[dict[str, Any]], stamp: int) -> dict[str, Any] | None:
    return next((row for row in candles if row["ts_ns"] >= stamp), None)


def full_exit_result(arm: str, trade: dict[str, Any], candle: dict[str, Any], reason: str) -> dict[str, Any]:
    price, half = adverse_market_price(candle, trade["side"])
    net = price_pnl(float(trade["entry_price"]), price, int(trade["entry_units"]), trade["side"])
    return {"arm": arm, "changed": True, "action": "EXIT", "action_time": candle["time"], "reason": reason, "diagnostic_exit_price": price, "adverse_half_spread_slippage_price": half, "diagnostic_net_excluding_unknown_fee_financing_jpy": net}


def stop_result(arm: str, trade: dict[str, Any], candle: dict[str, Any], stop: float, reason: str, units: int | None = None) -> dict[str, Any]:
    price, half = adverse_stop_price(candle, trade["side"], stop)
    amount = int(units if units is not None else trade["entry_units"])
    net = price_pnl(float(trade["entry_price"]), price, amount, trade["side"])
    return {"arm": arm, "changed": True, "action": "STOP_EXIT", "action_time": candle["time"], "reason": reason, "diagnostic_exit_price": price, "adverse_half_spread_slippage_price": half, "diagnostic_net_excluding_unknown_fee_financing_jpy": net}


def unchanged(arm: str, trade: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"arm": arm, "changed": False, "action": "BASELINE_PASS_THROUGH", "action_time": None, "reason": reason, "diagnostic_exit_price": None, "adverse_half_spread_slippage_price": 0.0, "diagnostic_net_excluding_unknown_fee_financing_jpy": float(trade["corrected_actual_after_cost_net_jpy"])}


def simulate(trade: dict[str, Any], candles: list[dict[str, Any]], features: list[dict[str, Any]], schedule: list[dict[str, Any]], monitor: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results = {"BASELINE": unchanged("BASELINE", trade, "ACTUAL_CORRECTED_V2")}
    results["ALL_COST_BE"] = {"arm": "ALL_COST_BE", "changed": None, "action": None, "action_time": None, "reason": "CAUSAL_FEE_FINANCING_SCHEDULE_MISSING", "diagnostic_exit_price": None, "adverse_half_spread_slippage_price": None, "diagnostic_net_excluding_unknown_fee_financing_jpy": None}
    if not candles:
        for arm in ARMS[1:]:
            results.setdefault(arm, {"arm": arm, "changed": None, "action": None, "action_time": None, "reason": "STRICT_S5_PATH_MISSING", "diagnostic_exit_price": None, "adverse_half_spread_slippage_price": None, "diagnostic_net_excluding_unknown_fee_financing_jpy": None})
        return results
    side, entry = trade["side"], float(trade["entry_price"])
    tp_events = [row for row in schedule if row.get("protection_kind") == "TP" and row.get("event_kind") == "CREATE"]
    tp = float(tp_events[0]["price"]) if tp_events else None
    favorable_extreme = entry

    # Fixed BE.
    fixed = None
    armed_at = None
    if tp is not None:
        trigger_distance = abs(tp - entry) * 0.60
        for index, candle in enumerate(candles):
            q = quote(candle, side)
            favorable = q["h"] - entry if side == "LONG" else entry - q["l"]
            spread = float(candle["ask"]["o"]) - float(candle["bid"]["o"])
            if armed_at is None and favorable >= max(trigger_distance, spread):
                armed_at = index + 1
                continue
            if armed_at is not None and index >= armed_at:
                touched = q["l"] <= entry if side == "LONG" else q["h"] >= entry
                if touched:
                    fixed = stop_result("FIXED_BE", trade, candle, entry, "BE_STOP_AFTER_60_PERCENT_TP_PROGRESS")
                    break
    results["FIXED_BE"] = fixed or unchanged("FIXED_BE", trade, "NO_CAUSAL_BE_EXIT_BEFORE_BASELINE")

    entry_feature = feature_before(features, bpm.parse_ns(trade["fill_at_utc"]))
    atr = float(entry_feature["atr14"]) if entry_feature and entry_feature.get("atr14") is not None else None

    # Partial + BE.
    partial = None
    if atr is None or int(trade["entry_units"]) < 2000:
        partial = {"arm": "PARTIAL_TP_BE", "changed": None, "action": None, "action_time": None, "reason": "ENTRY_ATR_OR_MINIMUM_UNITS_MISSING", "diagnostic_exit_price": None, "adverse_half_spread_slippage_price": None, "diagnostic_net_excluding_unknown_fee_financing_jpy": None}
    else:
        partial_units = (int(trade["entry_units"]) // 2 // 100) * 100
        runner = int(trade["entry_units"]) - partial_units
        trigger_index = None
        for index, candle in enumerate(candles):
            q = quote(candle, side)
            favorable = q["h"] - entry if side == "LONG" else entry - q["l"]
            if trigger_index is None and favorable >= atr:
                trigger_index = index + 1
                continue
            if trigger_index is not None and index == trigger_index:
                px, half = adverse_market_price(candle, side)
                first_leg = price_pnl(entry, px, partial_units, side)
                runner_exit = None
                for later in candles[index + 1:]:
                    later_q = quote(later, side)
                    touched = later_q["l"] <= entry if side == "LONG" else later_q["h"] >= entry
                    if touched:
                        runner_exit = stop_result("PARTIAL_TP_BE", trade, later, entry, "RUNNER_BE_STOP", runner)
                        break
                runner_net = runner_exit["diagnostic_net_excluding_unknown_fee_financing_jpy"] if runner_exit else float(trade["corrected_actual_after_cost_net_jpy"]) * runner / int(trade["entry_units"])
                partial = {"arm": "PARTIAL_TP_BE", "changed": True, "action": "PARTIAL_AND_RUNNER", "action_time": candle["time"], "reason": "ONE_ATR_PARTIAL_TRIGGER", "diagnostic_exit_price": px, "adverse_half_spread_slippage_price": half, "diagnostic_net_excluding_unknown_fee_financing_jpy": first_leg + runner_net}
                break
        partial = partial or unchanged("PARTIAL_TP_BE", trade, "NO_ONE_ATR_TRIGGER")
    results["PARTIAL_TP_BE"] = partial

    # Pure ATR trail using OANDA favorable extrema and causal feature-only ATR distance.
    trail = None
    stop = None
    last_feature_end = None
    for index, candle in enumerate(candles):
        q = quote(candle, side)
        favorable_extreme = max(favorable_extreme, q["h"]) if side == "LONG" else min(favorable_extreme, q["l"])
        current_feature = feature_before(features, candle["ts_ns"])
        current_atr = float(current_feature["atr14"]) if current_feature and current_feature.get("atr14") is not None else None
        favorable = favorable_extreme - entry if side == "LONG" else entry - favorable_extreme
        if atr is not None and stop is None and favorable >= atr and current_atr is not None:
            stop = favorable_extreme - 1.5 * current_atr if side == "LONG" else favorable_extreme + 1.5 * current_atr
            last_feature_end = current_feature["end_ns"]
            continue
        if stop is not None and current_feature and current_feature["end_ns"] != last_feature_end and current_atr is not None:
            candidate = favorable_extreme - 1.5 * current_atr if side == "LONG" else favorable_extreme + 1.5 * current_atr
            stop = max(stop, candidate) if side == "LONG" else min(stop, candidate)
            last_feature_end = current_feature["end_ns"]
            continue
        if stop is not None:
            touched = q["l"] <= stop if side == "LONG" else q["h"] >= stop
            if touched:
                trail = stop_result("PURE_ATR_TRAIL", trade, candle, stop, "RATCHETED_ATR_STOP")
                break
    results["PURE_ATR_TRAIL"] = trail or (unchanged("PURE_ATR_TRAIL", trade, "NO_ATR_TRAIL_EXIT") if atr is not None else {"arm": "PURE_ATR_TRAIL", "changed": None, "action": None, "action_time": None, "reason": "ENTRY_ATR_MISSING", "diagnostic_exit_price": None, "adverse_half_spread_slippage_price": None, "diagnostic_net_excluding_unknown_fee_financing_jpy": None})

    # Existing completed-M1 monitor hard exit.
    monitor_exit = next((row for row in monitor if row.get("action") == "EXIT" and bpm.parse_ns(row["decision_time"]) >= bpm.parse_ns(trade["fill_at_utc"])), None)
    if monitor_exit:
        action = next_candle(candles, bpm.parse_ns(monitor_exit["decision_time"]))
        results["SMA_DETERIORATION_TRAIL"] = full_exit_result("SMA_DETERIORATION_TRAIL", trade, action, "THREE_BAR_HARD_TECHNICAL_CONTRADICTION") if action else unchanged("SMA_DETERIORATION_TRAIL", trade, "SIGNAL_AFTER_INTERIOR_PATH")
    else:
        results["SMA_DETERIORATION_TRAIL"] = unchanged("SMA_DETERIORATION_TRAIL", trade, "NO_THREE_BAR_HARD_CONTRADICTION")

    # Structure break from completed feature-only M1, executed only on next OANDA S5.
    structure = None
    fill_ns = bpm.parse_ns(trade["fill_at_utc"])
    for feature in features:
        if not fill_ns < feature["end_ns"] < bpm.parse_ns(trade["close_at_utc"]):
            continue
        broken = float(feature["close"]) < float(feature["prior5_low"]) if side == "LONG" and feature.get("prior5_low") is not None else float(feature["close"]) > float(feature["prior5_high"]) if side == "SHORT" and feature.get("prior5_high") is not None else False
        if broken:
            action = next_candle(candles, feature["end_ns"])
            if action:
                structure = full_exit_result("STRUCTURE_BREAK_EXIT", trade, action, "COMPLETED_M1_PRIOR5_STRUCTURE_BREAK")
            break
    results["STRUCTURE_BREAK_EXIT"] = structure or unchanged("STRUCTURE_BREAK_EXIT", trade, "NO_CAUSAL_STRUCTURE_EXIT")

    timed = next_candle(candles, bpm.parse_ns(trade["fill_at_utc"]) + 30 * 60 * bpm.NS)
    results["TIME_EXIT"] = full_exit_result("TIME_EXIT", trade, timed, "FROZEN_30_MINUTE_HORIZON") if timed else unchanged("TIME_EXIT", trade, "BASELINE_CLOSED_BEFORE_30_MINUTES")
    return results


def main() -> int:
    paths = read_jsonl(HERE / "path_metrics_v1.jsonl")
    strict = [row for row in paths if row["path_complete"]]
    path_prereg = json.loads((HERE / "path_preregister_v1.json").read_text())
    candles = load_s5(strict, path_prereg)
    m1 = load_m1()
    schedules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(REPO / "research/active_protection_schedule/2026-08-10/schedule_events_v1.jsonl"):
        schedules[str(row["trade_id"])].append(row)
    monitors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(REPO / "research/continuous_thesis_monitor/2026-08-10/monitor_ledger_v2.jsonl"):
        monitors[str(row["position_id"])].append(row)
    simulated = {row["episode_id"]: simulate(row, candles.get(row["trade_id"], []), m1.get(row["pair"], []), schedules.get(row["trade_id"], []), monitors.get(row["trade_id"], [])) for row in strict}

    payload = json.loads((REPO / path_prereg["split_membership"]["path"]).read_text())
    records: list[dict[str, Any]] = []
    cube: list[dict[str, Any]] = []
    path_by_episode = {row["episode_id"]: row for row in paths}
    seen: set[tuple[str, str, str]] = set()
    for membership in payload["episode_records"]:
        episode_id = membership["episode_id"]
        key = (membership["window"], membership["split"], episode_id)
        if key in seen or episode_id not in path_by_episode:
            continue
        seen.add(key)
        trade = path_by_episode[episode_id]
        for arm in ARMS:
            if arm == "BASELINE":
                result = unchanged(arm, trade, "ACTUAL_CORRECTED_V2")
                status = "EVALUABLE_ACTUAL"
            elif not trade["path_complete"]:
                result = {"arm": arm, "changed": None, "action": None, "action_time": None, "reason": "STRICT_EXECUTABLE_PATH_UNRESOLVED", "diagnostic_exit_price": None, "adverse_half_spread_slippage_price": None, "diagnostic_net_excluding_unknown_fee_financing_jpy": None}
                status = "NOT_EVALUABLE_PATH"
            else:
                result = simulated[episode_id][arm]
                status = "DIAGNOSTIC_ONLY_COST_MARGIN_INCOMPLETE" if result["changed"] is not None else "NOT_EVALUABLE_EVIDENCE"
            after_cost = float(trade["corrected_actual_after_cost_net_jpy"]) if result["changed"] is False or arm == "BASELINE" else None
            record = {
                "episode_id": episode_id, "trade_id": trade["trade_id"], "window": membership["window"], "split": membership["split"],
                "pair": trade["pair"], "side": trade["side"], "fill_at_utc": trade["fill_at_utc"], "close_at_utc": trade["close_at_utc"],
                "exit_policy": arm, "path_complete": trade["path_complete"],
                "admission_status": status, "baseline_actual_after_cost_net_jpy": trade["corrected_actual_after_cost_net_jpy"],
                "candidate_actual_after_cost_net_jpy": after_cost, **result,
            }
            records.append(record)
            for metric, value in (
                ("actual_after_cost_net_jpy", after_cost),
                ("diagnostic_net_excluding_unknown_fee_financing_jpy", result["diagnostic_net_excluding_unknown_fee_financing_jpy"]),
                ("action_changed", 1 if result["changed"] is True else 0 if result["changed"] is False else None),
            ):
                cube.append({
                    "episode_id": episode_id, "source_sha": trade["output_sha256"], "decision_time": trade["fill_at_utc"],
                    "pair": trade["pair"], "timeframe": "S5_EXECUTION_M1_FEATURE", "regime": membership.get("regime"),
                    "strategy": membership.get("method", "ALL_TRADES"), "parameter_set": "EXIT_POLICY_PAIRED_REPLAY_V1",
                    "cost_scenario": "OANDA_SPREAD_PLUS_ADVERSE_HALF_SPREAD_UNKNOWN_FEE_FINANCING",
                    "exposure_state": "GROSS_COHORT_PROXY_ACCOUNT_NETTING_MISSING", "exit_policy": arm,
                    "viewpoint": "SINGLE_AXIS_EXIT_ABLATION", "metric": metric, "value": value, "uncertainty": None,
                    "sample_count": 1, "admission_status": status, "window": membership["window"], "split": membership["split"],
                })
    summary: dict[str, Any] = {}
    for window in sorted({row["window"] for row in records}):
        summary[window] = {}
        for split in ("TRAIN", "VALIDATION"):
            summary[window][split] = {}
            for arm in ARMS:
                rows = [row for row in records if row["window"] == window and row["split"] == split and row["exit_policy"] == arm]
                changed = [row for row in rows if row["changed"] is True]
                evaluable = [row for row in rows if row["candidate_actual_after_cost_net_jpy"] is not None]
                diagnostic = sum((row["diagnostic_net_excluding_unknown_fee_financing_jpy"] if row["changed"] is True else row["baseline_actual_after_cost_net_jpy"]) for row in rows if row["diagnostic_net_excluding_unknown_fee_financing_jpy"] is not None or row["changed"] is not True)
                summary[window][split][arm] = {
                    "episodes": len(rows), "strict_path": sum(row["path_complete"] for row in rows), "changed": len(changed),
                    "after_cost_evaluable": len(evaluable), "diagnostic_full_cohort_net_jpy": diagnostic,
                    "admission": "PASS_BASELINE" if arm == "BASELINE" else "NOT_EVALUABLE_MINIMUM_AND_COST_MARGIN_EVIDENCE",
                }
    report = {
        "contract": "EXIT_POLICY_PAIRED_REPLAY_V1", "status": "BASELINE_PASS_EXIT_ARMS_NOT_EVALUABLE",
        "strict_path_episodes": len(strict), "arms": ARMS, "records": len(records), "cube_rows": len(cube),
        "summary": summary, "holdout_used": False,
        "decisive_blockers": ["STRICT_PATH_VALIDATION_BELOW_20", "CAUSAL_FEE_FINANCING_SCHEDULE_MISSING", "ACCOUNT_MARGIN_AND_NETTING_MISSING", "PARTIAL_FILL_DEPTH_AND_UNWIND_MISSING"],
        "next_cube_phase": "NO_PROMISING_AXIS_ADMITTED_SO_TWO_AXIS_INTERACTIONS_AND_PARETO_NOT_RUN",
    }
    write_jsonl(HERE / "exit_replay_rows_v1.jsonl", records)
    write_jsonl(HERE / "exit_cube_long_v1.jsonl", cube)
    bpm.write_json(HERE / "exit_report_v1.json", report)
    manifest = {
        "contract": "EXIT_POLICY_PAIRED_REPLAY_RUN_MANIFEST_V1",
        "inputs": {
            "exit_preregister": bpm.sha256_path(HERE / "exit_preregister_v1.json"),
            "financial_oracle": bpm.sha256_path(HERE / "financial_oracle_v2.json"),
            "path_metrics": bpm.sha256_path(HERE / "path_metrics_v1.jsonl"),
            "split_membership": bpm.sha256_path(REPO / path_prereg["split_membership"]["path"]),
        },
        "outputs": {
            name: bpm.sha256_path(HERE / name)
            for name in ("exit_replay_rows_v1.jsonl", "exit_cube_long_v1.jsonl", "exit_report_v1.json")
        },
        "holdout_used": False,
    }
    manifest["manifest_sha256"] = bpm.sha256_value(manifest)
    bpm.write_json(HERE / "exit_manifest_v1.json", manifest)
    print(bpm.canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
