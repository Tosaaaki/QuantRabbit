"""V41 one-shot paper replay for a causal London-open false-break reclaim family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_asian_displacement_handoff_fade_v32 as engine
from run_asian_displacement_handoff_fade_v33 import canonical_utc_nine_digits


CYCLE_ID = "V41"
EXPERIMENT = "FX_LONDON_OPEN_FALSE_BREAK_RECLAIM_V41"
FAMILY = "FX_LONDON_OPEN_FALSE_BREAK_RECLAIM"
ASIAN_START_MINUTE = 0
ASIAN_END_MINUTE = 5 * 60 + 55
SWEEP_START_MINUTE = 6 * 60
SWEEP_END_MINUTE = 8 * 60 + 30
DECISION_MINUTE = 8 * 60 + 55
FILL_MINUTE = 9 * 60
EXIT_MINUTE = 12 * 60 + 55
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
                    for minute in range(ASIAN_START_MINUTE, DECISION_MINUTE + 1, 5)]
        if any(stamp not in by_stamp for stamp in required):
            return []
        asian = [by_stamp[engine.expected_stamp(day, minute)]
                 for minute in range(ASIAN_START_MINUTE, ASIAN_END_MINUTE + 1, 5)]
        opening = [by_stamp[engine.expected_stamp(day, minute)]
                   for minute in range(SWEEP_START_MINUTE, SWEEP_END_MINUTE + 1, 5)]
        asian_high = max(bar.mid_h for bar in asian)
        asian_low = min(bar.mid_l for bar in asian)
        swept_upper = max(bar.mid_h for bar in opening) > asian_high
        swept_lower = min(bar.mid_l for bar in opening) < asian_low
        decision_bar = by_stamp[engine.expected_stamp(day, DECISION_MINUTE)]
        reclaimed_inside = asian_low < decision_bar.mid_c < asian_high
        if swept_upper == swept_lower or not reclaimed_inside:
            continue
        direction = -1 if swept_upper else 1
        rows.append({
            "signal_id": (
                f"LOFBR::{day.date().isoformat()}::{pair}::"
                f"{'RECLAIM_LONG' if direction > 0 else 'RECLAIM_SHORT'}"
            ),
            "pair": pair,
            "utc_day": day.date().isoformat(),
            "decision_time": canonical_utc_nine_digits(decision_bar.time),
            "fill_time": canonical_utc_nine_digits(
                engine.expected_stamp(day, FILL_MINUTE).isoformat().replace("+00:00", "Z")
            ),
            "exit_time": canonical_utc_nine_digits(
                engine.expected_stamp(day, EXIT_MINUTE).isoformat().replace("+00:00", "Z")
            ),
            "direction": direction,
            "diagnostics": {
                "asian_range_high": asian_high,
                "asian_range_low": asian_low,
                "opening_swept_upper_only": swept_upper and not swept_lower,
                "opening_swept_lower_only": swept_lower and not swept_upper,
                "decision_close_inside_asian_range": reclaimed_inside,
                "direction_rule": "OPPOSITE_OF_SINGLE_SWEPT_ASIAN_RANGE_SIDE",
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
    ledger = output_root / "proposal_ledger_london_open_false_break_reclaim_v41.jsonl"
    old_ledger.replace(ledger)
    payload.update({
        "cycle_id": CYCLE_ID,
        "experiment": EXPERIMENT,
        "family": FAMILY,
        "family_hypotheses": 1,
        "single_changed_variable": "fx_specific_london_open_false_break_reclaim_signal_family",
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": engine.sha256_file(ledger),
        "indicator": {
            "asian_range_utc": "00:00-05:55_COMPLETED_MID_HIGH_LOW",
            "opening_sweep_utc": "06:00-08:30_COMPLETED_MID_HIGH_LOW",
            "decision_utc": "08:55_COMPLETED_CLOSE",
            "fill_utc": "09:00_EXECUTABLE_OPEN",
            "fixed_raw_exit_utc_bar": "12:55_COMPLETED_CLOSE",
            "eligibility_formula": "xor(max(H_0600_0830)>H_asian,min(L_0600_0830)<L_asian) and L_asian<C0855<H_asian",
            "direction_formula": "-1_if_upper_only_sweep_else_+1_if_lower_only_sweep",
            "threshold": "STRICT_RANGE_BOUNDARY_NO_FITTED_PARAMETER",
            "cost_used_for_signal": False,
            "post_entry_outcome_used_for_signal": False,
            "evaluation_month_used_for_threshold": False,
        },
    })
    walk = payload["periods"]["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OPEN_FALSE_BREAK_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OPEN_FALSE_BREAK_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OPEN_FALSE_BREAK_ADVERSE_COST_FRAGILE"
    else:
        reason = "LONDON_OPEN_FALSE_BREAK_MONTHLY_GATE_SHORTFALL"
    payload["automatic_rejection"]["reason_code"] = reason
    payload["result_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "result_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    old_result = output_root / "result_asian_displacement_handoff_fade_v32.json"
    result = output_root / "result_london_open_false_break_reclaim_v41.json"
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
