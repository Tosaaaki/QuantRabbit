"""V36 one-shot paper replay for one causal London/Asian range breakout family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_asian_displacement_handoff_fade_v32 as engine
from run_asian_displacement_handoff_fade_v33 import canonical_utc_nine_digits


CYCLE_ID = "V36"
EXPERIMENT = "FX_LONDON_ASIAN_RANGE_BREAKOUT_V36"
FAMILY = "FX_SESSION_RANGE_BREAKOUT"
ASIAN_END_MINUTE = 5 * 60 + 55
DECISION_MINUTE = 7 * 60 + 55
FILL_MINUTE = 8 * 60
EXIT_MINUTE = 15 * 60 + 55
_FROZEN_DETECTOR = engine.detect_day_signals


def detect_day_signals(pair_day_bars: dict[str, list]) -> list[dict]:
    """Emit only completed 07:55 closes outside the completed Asian range."""
    if set(pair_day_bars) != set(engine.UNIVERSE):
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
        asian_high = max(bar.mid_h for bar in asian)
        asian_low = min(bar.mid_l for bar in asian)
        decision_close = by_stamp[engine.expected_stamp(day, DECISION_MINUTE)].mid_c
        if decision_close > asian_high:
            direction = 1
        elif decision_close < asian_low:
            direction = -1
        else:
            continue
        decision_time = by_stamp[engine.expected_stamp(day, DECISION_MINUTE)].time
        rows.append({
            "signal_id": (
                f"LARB::{day.date().isoformat()}::{pair}::"
                f"{'BREAK_LONG' if direction > 0 else 'BREAK_SHORT'}"
            ),
            "pair": pair,
            "utc_day": day.date().isoformat(),
            "decision_time": canonical_utc_nine_digits(decision_time),
            "fill_time": canonical_utc_nine_digits(
                engine.expected_stamp(day, FILL_MINUTE).isoformat().replace("+00:00", "Z")
            ),
            "exit_time": canonical_utc_nine_digits(
                engine.expected_stamp(day, EXIT_MINUTE).isoformat().replace("+00:00", "Z")
            ),
            "direction": direction,
            "diagnostics": {
                "asian_mid_high": asian_high,
                "asian_mid_low": asian_low,
                "decision_mid_close": decision_close,
                "direction_rule": "SIGN_OF_COMPLETED_0755_CLOSE_BREAK_OUTSIDE_0000_0555_RANGE",
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
    ledger = output_root / "proposal_ledger_london_asian_range_breakout_v36.jsonl"
    old_ledger.replace(ledger)
    payload.update({
        "cycle_id": CYCLE_ID,
        "experiment": EXPERIMENT,
        "family": FAMILY,
        "family_hypotheses": 1,
        "single_changed_variable": "fx_specific_london_asian_range_breakout_signal_family",
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": engine.sha256_file(ledger),
        "indicator": {
            "asian_range_utc": "00:00-05:55_COMPLETED_MID_HIGH_LOW",
            "observation_window_utc": "06:00-07:55_COMPLETED",
            "decision_utc": "07:55_COMPLETED_CLOSE",
            "fill_utc": "08:00_EXECUTABLE_OPEN",
            "fixed_raw_exit_utc_bar": "15:55_COMPLETED_CLOSE",
            "direction_formula": "sign(mid_close_07:55 outside completed 00:00-05:55 mid range)",
            "threshold": "STRICT_RANGE_BOUNDARY_NO_TUNED_BUFFER",
            "cost_used_for_signal": False,
            "post_entry_outcome_used_for_signal": False,
            "evaluation_month_used_for_threshold": False,
        },
    })
    walk = payload["periods"]["WALK_FORWARD"]
    if walk["RAW_SIGNAL"]["equity_multiple"] <= 1.0:
        reason = "LONDON_ASIAN_RANGE_BREAKOUT_RAW_EDGE_ABSENT"
    elif walk["EXECUTABLE_BASE"]["equity_multiple"] <= 1.0:
        reason = "LONDON_ASIAN_RANGE_BREAKOUT_COST_DOMINANT"
    elif walk["ADVERSE_STRESS"]["equity_multiple"] <= 1.0:
        reason = "LONDON_ASIAN_RANGE_BREAKOUT_ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"
    payload["automatic_rejection"]["reason_code"] = reason
    payload["result_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "result_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    old_result = output_root / "result_asian_displacement_handoff_fade_v32.json"
    result = output_root / "result_london_asian_range_breakout_v36.json"
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
