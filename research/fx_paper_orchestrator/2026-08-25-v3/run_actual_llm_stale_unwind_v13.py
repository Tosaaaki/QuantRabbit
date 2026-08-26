from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, pip_size, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS, PERIODS, timestamp
from run_causal_harvest_exit_v12 import signed_mid_return
from run_liquid_major_universe_v9 import UNIVERSE


POLICY_ID = "STALE_DIRECTIONAL_IMPULSE_UNWIND_AT_192"


def score_policy(
    bars: list[Bar], fill_index: int, direction: int, target: float,
    arm: str, stale_age: int, impulse_window: int, max_age: int,
) -> dict | None:
    if fill_index + max_age + 1 >= len(bars):
        return None
    entry = bars[fill_index]
    decision_step, exit_reason = max_age, "MAX_AGE"
    for step in range(1, max_age + 1):
        if signed_mid_return(entry, bars[fill_index + step], direction) >= target:
            decision_step, exit_reason = step, "HARVEST"
            break
        if step == stale_age:
            closes = [bars[fill_index + offset].mid_c for offset in range(step - impulse_window, step + 1)]
            impulse = direction * sum(math.log(right / left) for left, right in zip(closes, closes[1:]))
            if impulse <= 0:
                decision_step, exit_reason = step, "STALE_UNWIND"
                break
    exit_bar = bars[fill_index + decision_step + 1]
    gross = signed_mid_return(entry, exit_bar, direction, use_open=True)
    scenario = ARMS[arm]
    if scenario is None:
        net = gross
    else:
        slip = float(scenario["slippage"]) * pip_size(entry.pair)
        if direction > 0:
            net = (exit_bar.bid_o - slip) / (entry.ask_o + slip) - 1.0
        else:
            net = (entry.bid_o - slip) / (exit_bar.ask_o + slip) - 1.0
        elapsed_days = (timestamp(exit_bar.time) - timestamp(entry.time)).total_seconds() / 86400.0
        net -= 2.0 * float(scenario["commission"]) * 1e-4
        net -= float(scenario["financing"]) * 1e-4 * elapsed_days
    return {
        "arm": arm, "gross_return": gross, "net_return": net,
        "exit_reason": exit_reason, "decision_age_m5_bars": decision_step,
        "exit_time": exit_bar.time,
    }


def summarize(rows: list[dict], start: str, end: str) -> dict:
    selected = [row for row in rows if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end]
    arms = {}
    for arm in ARMS:
        values = [row["scores"][arm]["net_return"] for row in selected]
        arms[arm] = {
            "signals": len(values), "mean_return": statistics.fmean(values) if values else None,
            "median_return": statistics.median(values) if values else None,
            "positive_rate": sum(value > 0 for value in values) / len(values) if values else None,
            "additive_return": sum(values),
        }
    reasons = {name: sum(row["exit_reason"] == name for row in selected) for name in (
        "HARVEST", "STALE_UNWIND", "MAX_AGE"
    )}
    return {"start": start, "end": end, "arms": arms, "exit_reason_counts": reasons}


def run(input_root: Path, source_ledger: Path, llm_decision: Path, output_root: Path) -> dict:
    decision = json.loads(llm_decision.read_text())
    policy = decision["structured_decision"]
    if policy["policy_id"] != POLICY_ID or policy["policy_id"] not in decision["allowed_policy_ids"]:
        raise ValueError("actual LLM policy is not the registered allowlisted policy")
    if decision.get("live_authority") is not False or decision.get("external_orders") != 0:
        raise ValueError("actual LLM decision violates zero authority")
    raw_source = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    source_rows = [{key: row[key] for key in (
        "signal_id", "pair", "fill_time", "direction", "target_return"
    )} for row in raw_source]
    corpus = {}
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
    index = {pair: {bar.time: i for i, bar in enumerate(bars)} for pair, bars in corpus.items()}
    rows = []
    for source in source_rows:
        pair = source["pair"]
        fill_index = index[pair].get(source["fill_time"])
        if fill_index is None:
            raise ValueError(f"source fill is absent from corpus: {source['signal_id']}")
        scores = {arm: score_policy(
            corpus[pair], fill_index, int(source["direction"]), float(source["target_return"]), arm,
            int(policy["stale_check_age_m5_bars"]), int(policy["impulse_window_m5_bars"]),
            int(policy["max_age_m5_bars"]),
        ) for arm in ARMS}
        if any(value is None for value in scores.values()):
            continue
        raw = scores["RAW_SIGNAL"]
        rows.append({
            **source, "exit_time": raw["exit_time"], "exit_reason": raw["exit_reason"],
            "decision_age_m5_bars": raw["decision_age_m5_bars"], "scores": scores,
        })
    periods = {name: summarize(rows, *bounds) for name, bounds in PERIODS.items()}
    walk = periods["WALK_FORWARD"]["arms"]
    admitted = walk["RAW_SIGNAL"]["signals"] >= 20 and all(
        walk[arm]["mean_return"] is not None and walk[arm]["mean_return"] > 0 for arm in ARMS
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_actual_llm_stale_unwind_v13.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "experiment": "FX_ACTUAL_LLM_STALE_UNWIND_V13",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "actual_llm_selected_stale_unwind_at_192",
        "source_ledger": str(source_ledger), "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "direction", "target_return"],
        "source_outcome_fields_consumed": False,
        "actual_llm_decision": str(llm_decision), "actual_llm_decision_sha256": sha256_file(llm_decision),
        "actual_llm_policy": policy, "source_signals": len(source_rows), "scored_signals": len(rows),
        "source_audit": source_audit, "cost_suppressed_raw_signals": 0,
        "same_signal_id_all_cost_arms": True, "individual_price_sl": False,
        "period_membership_requires_contained_exit": True,
        "periods": periods, "development_admitted": admitted, "final_admitted": False,
        "proposal_ledger": str(ledger), "proposal_ledger_sha256": sha256_file(ledger),
        "terminal_open_inventory": 0, "terminal_inventory_mtm_hidden": False,
        "live_authority": False, "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_actual_llm_stale_unwind_v13.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--llm-decision", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.source_ledger, args.llm_decision, args.output_root)
    print(json.dumps({
        "actual_llm_policy": result["actual_llm_policy"], "walk_forward": result["periods"]["WALK_FORWARD"],
        "development_admitted": result["development_admitted"], "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
