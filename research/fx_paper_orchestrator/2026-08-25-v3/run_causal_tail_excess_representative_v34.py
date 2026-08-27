"""V34 one-shot paper replay for one cost-independent turnover rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import run_asian_displacement_handoff_fade_v32 as frozen_v32
import run_asian_displacement_handoff_fade_v33 as frozen_v33


CYCLE_ID = "V34"
EXPERIMENT = "FX_CAUSAL_TAIL_EXCESS_REPRESENTATIVE_V34"
PARENT_RESULT_SHA256 = "80ac9cf09680f50aec45eb36a29ed21528246d87bda30e1af049fee8722bd611"
PARENT_LEDGER_SHA256 = "6498f917839ed1bd13beb36e8e9eb650fc5aa972d4c1b9f073f52e624a8b9dd4"
ARMS = frozen_v32.ARMS
_FROZEN_BUILD_EXECUTION_LEDGER = frozen_v32.build_execution_ledger
_FROZEN_SIMULATE_PORTFOLIO = frozen_v32.simulate_portfolio


def causal_tail_excess_score(row: dict) -> float:
    diagnostics = row["diagnostics"]
    return abs(float(diagnostics["native_asian_log_displacement"])) / float(
        diagnostics["training_abs_displacement_q75"]
    )


def apply_rule(rows: list[dict]) -> set[str]:
    """Select one structurally strongest tail displacement per completed UTC day."""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_day[row["utc_day"]].append(row)
    selected = set()
    for day_rows in by_day.values():
        winner = min(day_rows, key=lambda row: (-causal_tail_excess_score(row), row["pair"], row["signal_id"]))
        selected.add(winner["signal_id"])
    return selected


def build_execution_ledger(rows: list[dict], corpus: dict[str, list]) -> list[dict]:
    selected_ids = apply_rule(rows)
    selected_source = [row for row in rows if row["signal_id"] in selected_ids]
    selected_ledger = _FROZEN_BUILD_EXECUTION_LEDGER(selected_source, corpus)
    actions = {row["signal_id"]: row["execution_action"] for row in selected_ledger}
    result = []
    for source in rows:
        row = json.loads(json.dumps(source, sort_keys=True, allow_nan=False))
        selected = row["signal_id"] in selected_ids
        row["execution_selected"] = selected
        row["execution_action"] = actions[row["signal_id"]] if selected \
            else "CASH_NOT_DAILY_MAX_TAIL_EXCESS"
        row["arm_actions"] = {arm: row["execution_action"] for arm in ARMS}
        row["turnover_rule"] = {
            "name": "ONE_DAILY_MAX_NORMALIZED_TAIL_EXCESS_REPRESENTATIVE",
            "score": causal_tail_excess_score(row),
            "selected": selected,
            "cost_inputs": False,
            "outcome_inputs": False,
        }
        result.append(row)
    result.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    return result


def simulate_portfolio(
    corpus: dict[str, list], rows: list[dict], arm: str, start: str, end: str
) -> dict:
    all_period = [row for row in rows if start <= row["fill_time"][:10] < end]
    selected = [row for row in all_period if row["execution_selected"] is True]
    metrics = _FROZEN_SIMULATE_PORTFOLIO(corpus, selected, arm, start, end)
    metrics["source_signals"] = len(all_period)
    metrics["processed_raw_signals"] = len(all_period)
    metrics["executed_signals"] = len(selected)
    metrics["cash_signals"] = len(all_period) - len(selected)
    metrics["execution_selection_rate"] = len(selected) / len(all_period) if all_period else None
    return metrics


def _load_parent(result_path: Path, ledger_path: Path) -> tuple[dict, list[dict]]:
    if frozen_v32.sha256_file(result_path) != PARENT_RESULT_SHA256:
        raise ValueError("sealed V33 result hash mismatch")
    if frozen_v32.sha256_file(ledger_path) != PARENT_LEDGER_SHA256:
        raise ValueError("sealed V33 ledger hash mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("result_sha256") != frozen_v32.frozen_v31.embedded_hash(result, "result_sha256"):
        raise ValueError("sealed V33 embedded result hash mismatch")
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    return result, rows


def run(input_root: Path, parent_result: Path, parent_ledger: Path, output_root: Path) -> dict:
    parent, parent_rows = _load_parent(parent_result, parent_ledger)
    original_builder = frozen_v32.build_execution_ledger
    original_simulator = frozen_v32.simulate_portfolio
    frozen_v32.build_execution_ledger = build_execution_ledger
    frozen_v32.simulate_portfolio = simulate_portfolio
    try:
        payload = frozen_v33.run(input_root, output_root)
    finally:
        frozen_v32.build_execution_ledger = original_builder
        frozen_v32.simulate_portfolio = original_simulator

    old_ledger = output_root / "proposal_ledger_asian_displacement_handoff_fade_v33.jsonl"
    rows = [json.loads(line) for line in old_ledger.read_text(encoding="utf-8").splitlines() if line]
    identity = ("signal_id", "pair", "utc_day", "decision_time", "fill_time", "exit_time", "direction")
    if [[row[field] for field in identity] for row in rows] != [
            [row[field] for field in identity] for row in parent_rows]:
        raise ValueError("V34 changed sealed V33 RAW signal identity")
    ledger = output_root / "proposal_ledger_causal_tail_excess_representative_v34.jsonl"
    old_ledger.replace(ledger)

    payload["cycle_id"] = CYCLE_ID
    payload["experiment"] = EXPERIMENT
    payload["parent_cycle"] = "V33"
    payload["parent_result_sha256"] = frozen_v32.sha256_file(parent_result)
    payload["parent_ledger_sha256"] = frozen_v32.sha256_file(parent_ledger)
    payload["same_parent_signal_id_set"] = True
    payload["same_parent_decision_timestamps"] = True
    payload["same_parent_directions"] = True
    payload["proposal_ledger"] = str(ledger)
    payload["proposal_ledger_sha256"] = frozen_v32.sha256_file(ledger)
    selected_ids = apply_rule(rows)
    mask = [[row["signal_id"], row["signal_id"] in selected_ids] for row in rows]
    payload["same_execution_mask_all_cost_arms"] = True
    payload["execution_mask_sha256"] = hashlib.sha256(
        json.dumps(mask, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    payload["turnover_rule"] = {
        "name": "ONE_DAILY_MAX_NORMALIZED_TAIL_EXCESS_REPRESENTATIVE",
        "score_formula": "abs(native_asian_log_displacement) / frozen_training_abs_displacement_q75",
        "selection_formula": "argmax score within completed UTC day",
        "tie_break": "lexicographically smallest pair then signal_id",
        "selected_signals": len(selected_ids),
        "cash_signals": len(rows) - len(selected_ids),
        "cost_inputs": False,
        "post_entry_outcome_inputs": False,
        "evaluation_month_inputs": False,
    }
    walk = payload["periods"]["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "TAIL_EXCESS_REPRESENTATIVE_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "TAIL_EXCESS_REPRESENTATIVE_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "TAIL_EXCESS_REPRESENTATIVE_ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"
    payload["automatic_rejection"]["reason_code"] = reason
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "result_sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    ).hexdigest()
    old_result = output_root / "result_asian_displacement_handoff_fade_v33.json"
    result = output_root / "result_causal_tail_excess_representative_v34.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    old_result.unlink()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.parent_result, args.parent_ledger, args.output_root)
    print(json.dumps({
        "cycle_id": result["cycle_id"],
        "raw_signals": result["raw_signals"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "automatic_rejection": result["automatic_rejection"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
