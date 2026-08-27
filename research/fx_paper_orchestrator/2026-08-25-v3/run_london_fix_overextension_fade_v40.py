"""V40 one-shot paper replay for a causal London-fix overextension fade family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import run_asian_displacement_handoff_fade_v32 as engine
from run_asian_displacement_handoff_fade_v33 import canonical_utc_nine_digits


CYCLE_ID = "V40"
EXPERIMENT = "FX_LONDON_FIX_OVEREXTENSION_FADE_V40"
FAMILY = "FX_LONDON_FIX_MEAN_REVERSION"
REFERENCE_START_MINUTE = 8 * 60
REFERENCE_END_MINUTE = 11 * 60 + 55
MOVE_START_MINUTE = 12 * 60
DECISION_MINUTE = 15 * 60 + 55
FILL_MINUTE = 16 * 60
EXIT_MINUTE = 19 * 60 + 55
TARGET_HOLD_SECONDS = 14_100
EVALUATION_END_EXCLUSIVE = "2026-07-01"


def detect_day_signals(pair_day_bars: dict[str, list]) -> list[dict]:
    if set(pair_day_bars) != set(engine.UNIVERSE):
        return []
    first_pair = min(pair_day_bars)
    if not pair_day_bars[first_pair] \
            or pair_day_bars[first_pair][0].time[:10] >= EVALUATION_END_EXCLUSIVE:
        return []
    rows: list[dict] = []
    common_date = None
    for pair in sorted(engine.UNIVERSE):
        try:
            day, by_stamp = engine._validated_map(pair, pair_day_bars[pair])
        except ValueError:
            return []
        if common_date is None:
            common_date = day.date()
        elif day.date() != common_date:
            return []
        required = [engine.expected_stamp(day, minute)
                    for minute in range(REFERENCE_START_MINUTE, DECISION_MINUTE + 1, 5)]
        if any(stamp not in by_stamp for stamp in required):
            return []
        reference = [by_stamp[engine.expected_stamp(day, minute)]
                     for minute in range(REFERENCE_START_MINUTE, REFERENCE_END_MINUTE + 1, 5)]
        reference_log_range = math.log(
            max(bar.mid_h for bar in reference) / min(bar.mid_l for bar in reference)
        )
        move_open = by_stamp[engine.expected_stamp(day, MOVE_START_MINUTE)].mid_o
        decision_close = by_stamp[engine.expected_stamp(day, DECISION_MINUTE)].mid_c
        fix_displacement = math.log(decision_close / move_open)
        if fix_displacement == 0 or abs(fix_displacement) <= reference_log_range:
            continue
        direction = -1 if fix_displacement > 0 else 1
        rows.append({
            "signal_id": (
                f"LFOF::{day.date().isoformat()}::{pair}::"
                f"{'FADE_LONG' if direction > 0 else 'FADE_SHORT'}"
            ),
            "pair": pair,
            "utc_day": day.date().isoformat(),
            "decision_time": canonical_utc_nine_digits(
                by_stamp[engine.expected_stamp(day, DECISION_MINUTE)].time
            ),
            "fill_time": canonical_utc_nine_digits(
                engine.expected_stamp(day, FILL_MINUTE).isoformat().replace("+00:00", "Z")
            ),
            "exit_time": canonical_utc_nine_digits(
                engine.expected_stamp(day, EXIT_MINUTE).isoformat().replace("+00:00", "Z")
            ),
            "direction": direction,
            "diagnostics": {
                "london_morning_log_range": reference_log_range,
                "pre_fix_log_displacement": fix_displacement,
                "normalized_magnitude": abs(fix_displacement) / reference_log_range,
                "direction_rule": "NEGATIVE_SIGN_OF_COMPLETED_PRE_FIX_DISPLACEMENT",
                "cost_inputs": False,
                "post_entry_outcome_inputs": False,
            },
        })
    return rows


def run(input_root: Path, output_root: Path) -> dict:
    original_detector = engine.detect_day_signals
    original_planner_hold = engine.frozen_v31.TARGET_HOLD_SECONDS
    original_engine_hold = engine.TARGET_HOLD_SECONDS
    engine.detect_day_signals = detect_day_signals
    engine.frozen_v31.TARGET_HOLD_SECONDS = TARGET_HOLD_SECONDS
    engine.TARGET_HOLD_SECONDS = TARGET_HOLD_SECONDS
    try:
        payload = engine.run(input_root, output_root)
    finally:
        engine.detect_day_signals = original_detector
        engine.frozen_v31.TARGET_HOLD_SECONDS = original_planner_hold
        engine.TARGET_HOLD_SECONDS = original_engine_hold
    old_ledger = output_root / "proposal_ledger_asian_displacement_handoff_fade_v32.jsonl"
    ledger = output_root / "proposal_ledger_london_fix_overextension_fade_v40.jsonl"
    old_ledger.replace(ledger)
    payload.update({
        "cycle_id": CYCLE_ID,
        "experiment": EXPERIMENT,
        "family": FAMILY,
        "family_hypotheses": 1,
        "single_changed_variable": "fx_specific_london_fix_overextension_fade_signal_family",
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": engine.sha256_file(ledger),
        "indicator": {
            "reference_range_utc": "08:00-11:55_COMPLETED_MID_HIGH_LOW",
            "pre_fix_displacement_utc": "12:00_OPEN_TO_15:55_COMPLETED_CLOSE",
            "decision_utc": "15:55_COMPLETED",
            "fill_utc": "16:00_EXECUTABLE_OPEN",
            "fixed_raw_exit_utc_bar": "19:55_COMPLETED_CLOSE",
            "eligibility_formula": "abs(log(C1555/O1200)) > log(H_0800_1155/L_0800_1155)",
            "direction_formula": "-sign(log(C1555/O1200))",
            "threshold": "SAME_DAY_COMPLETED_LONDON_MORNING_RANGE_RATIO_ONE_NO_FITTED_PARAMETER",
            "cost_used_for_signal": False,
            "post_entry_outcome_used_for_signal": False,
            "evaluation_month_used_for_threshold": False,
        },
    })
    walk = payload["periods"]["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "LONDON_FIX_FADE_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "LONDON_FIX_FADE_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "LONDON_FIX_FADE_ADVERSE_COST_FRAGILE"
    else:
        reason = "LONDON_FIX_FADE_MONTHLY_GATE_SHORTFALL"
    payload["automatic_rejection"]["reason_code"] = reason
    payload["result_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "result_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    old_result = output_root / "result_asian_displacement_handoff_fade_v32.json"
    result = output_root / "result_london_fix_overextension_fade_v40.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    old_result.unlink()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print(json.dumps({"cycle_id": result["cycle_id"], "raw_signals": result["raw_signals"],
                      "walk_forward": result["periods"]["WALK_FORWARD"],
                      "automatic_rejection": result["automatic_rejection"],
                      "result_sha256": result["result_sha256"]}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
