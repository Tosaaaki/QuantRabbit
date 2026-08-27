"""V38 one-shot paper replay for a causal London overextension fade family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import run_asian_displacement_handoff_fade_v32 as engine
from run_asian_displacement_handoff_fade_v33 import canonical_utc_nine_digits


CYCLE_ID = "V38"
EXPERIMENT = "FX_LONDON_OVEREXTENSION_FADE_V38"
FAMILY = "FX_SESSION_RANGE_NORMALIZED_MEAN_REVERSION"
ASIAN_END_MINUTE = 5 * 60 + 55
LONDON_OPEN_MINUTE = 8 * 60
DECISION_MINUTE = 11 * 60 + 55
FILL_MINUTE = 12 * 60
EXIT_MINUTE = 15 * 60 + 55
EVALUATION_END_EXCLUSIVE = "2026-07-01"


def detect_day_signals(pair_day_bars: dict[str, list]) -> list[dict]:
    """Fade a completed London displacement only when it exceeds Asian range width."""
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
                    for minute in range(0, DECISION_MINUTE + 1, 5)]
        if any(stamp not in by_stamp for stamp in required):
            return []
        asian = [by_stamp[engine.expected_stamp(day, minute)]
                 for minute in range(0, ASIAN_END_MINUTE + 1, 5)]
        asian_log_range = math.log(max(bar.mid_h for bar in asian) / min(bar.mid_l for bar in asian))
        london_open = by_stamp[engine.expected_stamp(day, LONDON_OPEN_MINUTE)].mid_o
        decision_close = by_stamp[engine.expected_stamp(day, DECISION_MINUTE)].mid_c
        london_displacement = math.log(decision_close / london_open)
        if london_displacement == 0 or abs(london_displacement) <= asian_log_range:
            continue
        direction = -1 if london_displacement > 0 else 1
        rows.append({
            "signal_id": (
                f"LOEF::{day.date().isoformat()}::{pair}::"
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
                "asian_log_range": asian_log_range,
                "london_log_displacement": london_displacement,
                "normalized_magnitude": abs(london_displacement) / asian_log_range,
                "direction_rule": "NEGATIVE_SIGN_OF_COMPLETED_LONDON_DISPLACEMENT",
                "cost_inputs": False,
                "post_entry_outcome_inputs": False,
            },
        })
    return rows


def run(input_root: Path, output_root: Path) -> dict:
    original = engine.detect_day_signals
    engine.detect_day_signals = detect_day_signals
    try:
        payload = engine.run(input_root, output_root)
    finally:
        engine.detect_day_signals = original
    old_ledger = output_root / "proposal_ledger_asian_displacement_handoff_fade_v32.jsonl"
    ledger = output_root / "proposal_ledger_london_overextension_fade_v38.jsonl"
    old_ledger.replace(ledger)
    payload.update({
        "cycle_id": CYCLE_ID,
        "experiment": EXPERIMENT,
        "family": FAMILY,
        "family_hypotheses": 1,
        "single_changed_variable": "fx_specific_london_overextension_fade_signal_family",
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": engine.sha256_file(ledger),
        "indicator": {
            "asian_range_utc": "00:00-05:55_COMPLETED_MID_HIGH_LOW",
            "london_displacement_utc": "08:00_OPEN_TO_11:55_COMPLETED_CLOSE",
            "decision_utc": "11:55_COMPLETED",
            "fill_utc": "12:00_EXECUTABLE_OPEN",
            "fixed_raw_exit_utc_bar": "15:55_COMPLETED_CLOSE",
            "eligibility_formula": "abs(log(C1155/O0800)) > log(H_asian/L_asian)",
            "direction_formula": "-sign(log(C1155/O0800))",
            "threshold": "SAME_DAY_COMPLETED_ASIAN_RANGE_RATIO_ONE_NO_FITTED_PARAMETER",
            "cost_used_for_signal": False,
            "post_entry_outcome_used_for_signal": False,
            "evaluation_month_used_for_threshold": False,
        },
    })
    walk = payload["periods"]["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OVEREXTENSION_FADE_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OVEREXTENSION_FADE_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "LONDON_OVEREXTENSION_FADE_ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"
    payload["automatic_rejection"]["reason_code"] = reason
    payload["result_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "result_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    old_result = output_root / "result_asian_displacement_handoff_fade_v32.json"
    result = output_root / "result_london_overextension_fade_v38.json"
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
