"""V35 one-shot paper replay for one global no-overlap admission rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import run_causal_tail_excess_representative_v34 as frozen_v34


CYCLE_ID = "V35"
EXPERIMENT = "FX_GLOBAL_NO_OVERLAP_ADMISSION_V35"
HARD_MAX_AGE_SECONDS = 345600
_FROZEN_V34_APPLY_RULE = frozen_v34.apply_rule


def daily_representatives(rows: list[dict]) -> list[dict]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_day[row["utc_day"]].append(row)
    return sorted(
        (
            min(day_rows, key=lambda row: (
                -frozen_v34.causal_tail_excess_score(row), row["pair"], row["signal_id"]
            ))
            for day_rows in by_day.values()
        ),
        key=lambda row: (frozen_v34.frozen_v32.frozen_v31.ns(row["fill_time"]), row["signal_id"]),
    )


def apply_rule(rows: list[dict]) -> set[str]:
    """Admit a frozen daily winner only after the prior hard-age window ends."""
    selected: set[str] = set()
    next_eligible_ns: int | None = None
    for row in daily_representatives(rows):
        fill_ns = frozen_v34.frozen_v32.frozen_v31.ns(row["fill_time"])
        if next_eligible_ns is not None and fill_ns < next_eligible_ns:
            continue
        selected.add(row["signal_id"])
        next_eligible_ns = fill_ns + HARD_MAX_AGE_SECONDS * 1_000_000_000
    return selected


def run(input_root: Path, parent_result: Path, parent_ledger: Path, output_root: Path) -> dict:
    original = frozen_v34.apply_rule
    frozen_v34.apply_rule = apply_rule
    try:
        payload = frozen_v34.run(input_root, parent_result, parent_ledger, output_root)
    finally:
        frozen_v34.apply_rule = original

    old_ledger = output_root / "proposal_ledger_causal_tail_excess_representative_v34.jsonl"
    rows = [json.loads(line) for line in old_ledger.read_text(encoding="utf-8").splitlines() if line]
    selected_ids = apply_rule(rows)
    for row in rows:
        selected = row["signal_id"] in selected_ids
        row["execution_selected"] = selected
        if not selected:
            row["execution_action"] = "CASH_GLOBAL_NO_OVERLAP"
            row["arm_actions"] = {arm: row["execution_action"] for arm in frozen_v34.ARMS}
        row["turnover_rule"] = {
            "name": "GLOBAL_HARD_AGE_NO_OVERLAP_ADMISSION",
            "daily_rank": "V34_MAX_NORMALIZED_TAIL_EXCESS",
            "selected": selected,
            "hard_spacing_seconds": HARD_MAX_AGE_SECONDS,
            "cost_inputs": False,
            "outcome_inputs": False,
        }
    ledger = output_root / "proposal_ledger_global_no_overlap_admission_v35.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    old_ledger.unlink()

    payload["cycle_id"] = CYCLE_ID
    payload["experiment"] = EXPERIMENT
    payload["failed_predecessor_cycle"] = "V34"
    payload["v34_result_metrics_used"] = False
    payload["proposal_ledger"] = str(ledger)
    payload["proposal_ledger_sha256"] = frozen_v34.frozen_v32.sha256_file(ledger)
    mask = [[row["signal_id"], row["signal_id"] in selected_ids] for row in rows]
    payload["execution_mask_sha256"] = hashlib.sha256(
        json.dumps(mask, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    payload["turnover_rule"] = {
        "name": "GLOBAL_HARD_AGE_NO_OVERLAP_ADMISSION",
        "daily_ranking": "unchanged V34 max normalized tail excess",
        "admission_formula": "admit daily winner iff fill_epoch_ns >= prior_admitted_fill_epoch_ns + 345600e9",
        "hard_spacing_seconds": HARD_MAX_AGE_SECONDS,
        "selected_signals": len(selected_ids),
        "cash_signals": len(rows) - len(selected_ids),
        "cost_inputs": False,
        "post_entry_outcome_inputs": False,
        "v34_result_metrics_used": False,
    }
    walk = payload["periods"]["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "GLOBAL_NO_OVERLAP_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "GLOBAL_NO_OVERLAP_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "GLOBAL_NO_OVERLAP_ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"
    payload["automatic_rejection"]["reason_code"] = reason
    payload["result_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "result_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    old_result = output_root / "result_causal_tail_excess_representative_v34.json"
    result = output_root / "result_global_no_overlap_admission_v35.json"
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
    print(json.dumps({"cycle_id": result["cycle_id"], "raw_signals": result["raw_signals"],
                      "walk_forward": result["periods"]["WALK_FORWARD"],
                      "automatic_rejection": result["automatic_rejection"],
                      "result_sha256": result["result_sha256"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
