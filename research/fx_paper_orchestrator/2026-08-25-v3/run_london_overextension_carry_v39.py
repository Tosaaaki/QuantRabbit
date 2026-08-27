"""V39 one-shot replay changing only V38's finite target carry duration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_london_overextension_fade_v38 as frozen_v38


CYCLE_ID = "V39"
EXPERIMENT = "FX_LONDON_OVEREXTENSION_CARRY_V39"
TARGET_HOLD_SECONDS = 14_100
HARD_MAX_AGE_SECONDS = 345_600


def run(input_root: Path, parent_result: Path, parent_ledger: Path, output_root: Path) -> dict:
    parent_result_sha = frozen_v38.engine.sha256_file(parent_result)
    parent_ledger_sha = frozen_v38.engine.sha256_file(parent_ledger)
    old_planner_hold = frozen_v38.engine.frozen_v31.TARGET_HOLD_SECONDS
    old_engine_hold = frozen_v38.engine.TARGET_HOLD_SECONDS
    frozen_v38.engine.frozen_v31.TARGET_HOLD_SECONDS = TARGET_HOLD_SECONDS
    frozen_v38.engine.TARGET_HOLD_SECONDS = TARGET_HOLD_SECONDS
    try:
        payload = frozen_v38.run(input_root, output_root)
    finally:
        frozen_v38.engine.frozen_v31.TARGET_HOLD_SECONDS = old_planner_hold
        frozen_v38.engine.TARGET_HOLD_SECONDS = old_engine_hold

    old_ledger = output_root / "proposal_ledger_london_overextension_fade_v38.jsonl"
    ledger = output_root / "proposal_ledger_london_overextension_carry_v39.jsonl"
    old_ledger.replace(ledger)
    parent_rows = [json.loads(line) for line in parent_ledger.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    identity = ("signal_id", "pair", "utc_day", "decision_time", "fill_time", "exit_time", "direction")
    if [[row[key] for key in identity] for row in rows] \
            != [[row[key] for key in identity] for row in parent_rows]:
        raise ValueError("V39 changed the sealed V38 RAW signal identity")
    payload.update({
        "cycle_id": CYCLE_ID,
        "experiment": EXPERIMENT,
        "single_changed_variable": "target_hold_seconds_from_172800_to_14100",
        "parent_cycle_id": "V38",
        "parent_result_sha256": parent_result_sha,
        "parent_ledger_sha256": parent_ledger_sha,
        "same_parent_signal_id_set": True,
        "same_parent_decision_fill_raw_exit_direction": True,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": frozen_v38.engine.sha256_file(ledger),
        "carry_rule": {
            "name": "LONDON_LUNCH_FIXED_RAW_HORIZON_CARRY",
            "target_hold_seconds": TARGET_HOLD_SECONDS,
            "hard_max_age_seconds": HARD_MAX_AGE_SECONDS,
            "changed_field_from_v38": "target_hold_seconds",
            "same_signal_ledger": True,
            "cost_or_outcome_inputs": False,
        },
    })
    walk = payload["periods"]["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OVEREXTENSION_CARRY_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OVEREXTENSION_CARRY_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OVEREXTENSION_CARRY_ADVERSE_COST_FRAGILE"
    else:
        reason = "LONDON_OVEREXTENSION_CARRY_MONTHLY_GATE_SHORTFALL"
    payload["automatic_rejection"]["reason_code"] = reason
    payload["result_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "result_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    old_result = output_root / "result_london_overextension_fade_v38.json"
    result = output_root / "result_london_overextension_carry_v39.json"
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
