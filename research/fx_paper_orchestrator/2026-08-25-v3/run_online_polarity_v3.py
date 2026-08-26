from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import pandas as pd

from indicator_factory_v3 import canonical, metrics, period
from run_v250_partial_holdout_v3 import (
    ROOT, V250, load_common_pair_data, manifest_for_local_source, sha256_file,
)


WORKERS = (
    "M15_DIRECTION_ALL",
    "M15_H1_AGREE",
    "M15_H4_AGREE",
    "MTF_UNANIMOUS",
    "H4_H1_PULLBACK_CONTINUATION",
)
LOOKBACK_DAYS = 84
MIN_DAILY_CLUSTERS = 20
STATE_Z = statistics.NormalDist().inv_cdf(0.95)


def week_start(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.floor("D") - pd.Timedelta(days=timestamp.dayofweek)


def polarity_state(matured: list[dict]) -> dict:
    """Choose a state from source-direction RAW outcomes only.

    The interval is deliberately calculated on daily cluster means rather than
    ticket rows. Costs are absent from this decision, so it cannot erase RAW
    proposals merely because observed execution is expensive.
    """
    if not matured:
        return {"mode": "FREEZE", "daily_clusters": 0, "mean_raw": None,
                "lower": None, "upper": None, "reason": "NO_MATURED_HISTORY"}
    frame = pd.DataFrame({
        "exit_time": [pd.Timestamp(row["exit_time"]) for row in matured],
        "raw": [float(row["returns"]["RAW_SIGNAL"]) for row in matured],
    })
    daily = frame.groupby(frame["exit_time"].dt.floor("D"))["raw"].mean()
    if len(daily) < MIN_DAILY_CLUSTERS:
        return {"mode": "FREEZE", "daily_clusters": int(len(daily)),
                "mean_raw": float(daily.mean()), "lower": None, "upper": None,
                "reason": "INSUFFICIENT_DAILY_CLUSTERS"}
    mean = float(daily.mean())
    std = float(daily.std(ddof=1))
    se = std / math.sqrt(len(daily)) if std > 0 else 0.0
    lower, upper = mean - STATE_Z * se, mean + STATE_Z * se
    if lower > 0:
        mode, reason = "CONTINUE", "RAW_INTERVAL_ABOVE_ZERO"
    elif upper < 0:
        mode, reason = "INVERT", "RAW_INTERVAL_BELOW_ZERO"
    else:
        mode, reason = "FREEZE", "RAW_INTERVAL_CROSSES_ZERO"
    return {"mode": mode, "daily_clusters": int(len(daily)), "mean_raw": mean,
            "lower": lower, "upper": upper, "reason": reason}


def exact_returns(source: dict, direction: int, pair_data: dict[str, pd.DataFrame],
                  scenarios: dict) -> dict[str, float]:
    entry_time, exit_time = pd.Timestamp(source["fill_time"]), pd.Timestamp(source["exit_time"])
    entry = pair_data[source["pair"]].loc[entry_time]
    exit_ = pair_data[source["pair"]].loc[exit_time]
    entry_mid = float((entry["bid_o"] + entry["ask_o"]) * 0.5)
    exit_mid = float((exit_["bid_o"] + exit_["ask_o"]) * 0.5)
    raw = exit_mid / entry_mid - 1.0 if direction > 0 else entry_mid / exit_mid - 1.0
    elapsed = (exit_time - entry_time).total_seconds() / 86400.0
    v245 = V250.V249.V245
    return {
        "RAW_SIGNAL": float(raw),
        "EXECUTABLE_BASE": float(v245.executed_return(
            source["pair"], direction, entry, exit_, elapsed, scenarios["normal"]
        )["net_return"]),
        "ADVERSE_STRESS": float(v245.executed_return(
            source["pair"], direction, entry, exit_, elapsed, scenarios["adverse"]
        )["net_return"]),
    }


def apply_online_policy(source_records: list[dict], pair_data: dict[str, pd.DataFrame],
                        scenarios: dict) -> tuple[list[dict], list[dict]]:
    decisions, states = [], []
    for worker in WORKERS:
        worker_rows = sorted(
            (row for row in source_records if row["worker"] == worker),
            key=lambda row: (row["fill_time"], row["signal_id"]),
        )
        weeks = sorted({week_start(row["fill_time"]) for row in worker_rows})
        state_by_week = {}
        for checkpoint in weeks:
            start = checkpoint - pd.Timedelta(days=LOOKBACK_DAYS)
            matured = [row for row in worker_rows
                       if start <= pd.Timestamp(row["exit_time"]) < checkpoint]
            state = polarity_state(matured)
            state.update({"worker": worker, "checkpoint": checkpoint.isoformat(),
                          "history_start": start.isoformat(),
                          "latest_allowed_exit": checkpoint.isoformat()})
            state_by_week[checkpoint] = state
            states.append(state)
        for source in worker_rows:
            checkpoint = week_start(source["fill_time"])
            state = state_by_week[checkpoint]
            direction = int(source["direction"])
            if state["mode"] == "INVERT":
                direction *= -1
            record = {
                "source_signal_id": source["signal_id"],
                "decision_id": hashlib.sha256(
                    f"POLARITY|{source['signal_id']}|{state['mode']}".encode()
                ).hexdigest()[:24],
                "worker": worker,
                "pair": source["pair"],
                "fill_time": source["fill_time"],
                "exit_time": source["exit_time"],
                "source_direction": int(source["direction"]),
                "selected_direction": direction if state["mode"] != "FREEZE" else None,
                "mode": state["mode"],
                "state_checkpoint": state["checkpoint"],
                "state_reason": state["reason"],
                "source_returns": source["returns"],
                "expected_order": state["mode"] != "FREEZE",
                "cost_used_to_generate_source_signal": False,
                "cost_used_to_choose_mode": False,
            }
            if record["expected_order"]:
                record["returns"] = exact_returns(source, direction, pair_data, scenarios)
            decisions.append(record)
    return decisions, states


def evaluate(source_records: list[dict], decisions: list[dict], states: list[dict],
             output_root: Path, source_audit: dict, source_ledger: Path,
             m15_horizon: int) -> dict:
    family_tests = len(WORKERS) * 2
    z = statistics.NormalDist().inv_cdf(1 - .05 / (2 * family_tests))
    candidates = []
    for worker in WORKERS:
        source = [row for row in source_records if row["worker"] == worker]
        executed = [row for row in decisions if row["worker"] == worker and row["expected_order"]]
        summaries = {}
        for name in ("TUNING", "WALK_FORWARD", "OPENED_DIAGNOSTIC"):
            source_period = [row for row in source if period(row["fill_time"]) == name]
            executed_period = [row for row in executed if period(row["fill_time"]) == name]
            all_decisions = [row for row in decisions if row["worker"] == worker
                             and period(row["fill_time"]) == name]
            summaries[name] = {
                "source_raw": metrics(source_period, "RAW_SIGNAL", z),
                "selected": {arm: metrics(executed_period, arm, z) for arm in (
                    "RAW_SIGNAL", "EXECUTABLE_BASE", "ADVERSE_STRESS"
                )},
                "modes": {mode: sum(row["mode"] == mode for row in all_decisions)
                          for mode in ("CONTINUE", "INVERT", "FREEZE")},
            }
        corrected = [
            summaries["TUNING"]["selected"]["RAW_SIGNAL"]["family_corrected_cluster_lower_bps"],
            summaries["WALK_FORWARD"]["selected"]["EXECUTABLE_BASE"]["family_corrected_cluster_lower_bps"],
            summaries["WALK_FORWARD"]["selected"]["ADVERSE_STRESS"]["family_corrected_cluster_lower_bps"],
        ]
        finite_corrected = [value for value in corrected if value is not None]
        candidates.append({
            "candidate_id": f"FX_ONLINE_POLARITY_{worker}_H{m15_horizon}_V3",
            "worker": worker,
            "holding_m15_bars": m15_horizon,
            "periods": summaries,
            "development_admitted": all(value is not None and value > 0 for value in corrected),
            "ranking_floor_bps": min(finite_corrected) if finite_corrected else None,
        })
    candidates.sort(key=lambda item: (
        item["development_admitted"],
        item["ranking_floor_bps"] if item["ranking_floor_bps"] is not None else -math.inf,
    ), reverse=True)
    output_root.mkdir(parents=True, exist_ok=True)
    decision_path = output_root / "decision_ledger_online_polarity_v3.jsonl"
    state_path = output_root / "state_ledger_online_polarity_v3.jsonl"
    decision_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions), encoding="utf-8")
    state_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in states), encoding="utf-8")
    payload = {
        "experiment": "FX_MATURED_OUTCOME_ONLINE_POLARITY_V3",
        "evidence_class": "opened_development_not_future_holdout",
        "policy": {
            "checkpoint": "Monday 00:00 UTC",
            "lookback_days": LOOKBACK_DAYS,
            "minimum_daily_clusters": MIN_DAILY_CLUSTERS,
            "interval": "two-sided 90 percent normal interval over daily-cluster RAW means",
            "allowed_modes": ["CONTINUE", "INVERT", "FREEZE"],
            "maturity_rule": "exit_time strictly before checkpoint",
            "cost_in_signal_or_state": False,
        },
        "source_signal_rows": len(source_records),
        "expected_order_rows": sum(row["expected_order"] for row in decisions),
        "cost_suppressed_source_signals": 0,
        "same_source_signal_id_all_cost_arms": True,
        "family_tests": family_tests,
        "family_corrected_z": z,
        "candidates": candidates,
        "development_admitted_count": sum(item["development_admitted"] for item in candidates),
        "final_admitted_count": 0,
        "source_audit": source_audit,
        "source_ledger": str(source_ledger),
        "source_ledger_sha256": sha256_file(source_ledger),
        "decision_ledger": str(decision_path),
        "decision_ledger_sha256": sha256_file(decision_path),
        "state_ledger": str(state_path),
        "state_ledger_sha256": sha256_file(state_path),
        "terminal_inventory_mtm_hidden": False,
        "admission_blockers": [
            "policy was added after the 2026 source was opened",
            "portfolio inventory and terminal liquidation are not replayed in this proposal-level branch",
            "future holdout has not elapsed",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    (output_root / "result_online_polarity_v3.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def run(input_root: Path, source_ledger: Path, output_root: Path, m15_horizon: int) -> dict:
    source_records = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    manifest = manifest_for_local_source(input_root)
    pair_data, source_audit = load_common_pair_data(manifest)
    contract = json.loads((ROOT / "research/llm_paper_experiment/2026-08-24-v245/contract_v245.json").read_text())
    decisions, states = apply_online_policy(source_records, pair_data, contract["execution"])
    return evaluate(source_records, decisions, states, output_root, source_audit, source_ledger, m15_horizon)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--m15-horizon", type=int, choices=(8, 16), required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.source_ledger, args.output_root, args.m15_horizon)
    print(json.dumps({
        "source_signal_rows": result["source_signal_rows"],
        "expected_order_rows": result["expected_order_rows"],
        "development_admitted_count": result["development_admitted_count"],
        "result_sha256": result["result_sha256"],
        "top_candidate": result["candidates"][0],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
