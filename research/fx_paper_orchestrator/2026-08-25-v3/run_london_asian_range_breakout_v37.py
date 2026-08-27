"""V37 recovery of the unobserved V36 family with fixed evaluation-scope eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_london_asian_range_breakout_v36 as frozen_v36


CYCLE_ID = "V37"
EXPERIMENT = "FX_LONDON_ASIAN_RANGE_BREAKOUT_V37"
EVALUATION_END_EXCLUSIVE = "2026-07-01"
_FROZEN_DETECTOR = frozen_v36.detect_day_signals


def detect_day_signals(pair_day_bars: dict[str, list]) -> list[dict]:
    first_pair = min(pair_day_bars) if pair_day_bars else None
    if first_pair is None or not pair_day_bars[first_pair]:
        return []
    utc_day = pair_day_bars[first_pair][0].time[:10]
    if utc_day >= EVALUATION_END_EXCLUSIVE:
        return []
    return _FROZEN_DETECTOR(pair_day_bars)


def run(input_root: Path, output_root: Path) -> dict:
    original = frozen_v36.detect_day_signals
    frozen_v36.detect_day_signals = detect_day_signals
    try:
        payload = frozen_v36.run(input_root, output_root)
    finally:
        frozen_v36.detect_day_signals = original
    old_ledger = output_root / "proposal_ledger_london_asian_range_breakout_v36.jsonl"
    ledger = output_root / "proposal_ledger_london_asian_range_breakout_v37.jsonl"
    old_ledger.replace(ledger)
    payload.update({
        "cycle_id": CYCLE_ID,
        "experiment": EXPERIMENT,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": frozen_v36.engine.sha256_file(ledger),
        "runtime_compatibility_provenance": {
            "classification": "NON_STRATEGY_EVALUATION_SCOPE_COMPATIBILITY",
            "changed_strategy_variables": 0,
            "same_unobserved_v36_strategy_within_evaluation": True,
            "evaluation_end_exclusive": EVALUATION_END_EXCLUSIVE,
            "v36_rerun_permitted": False,
            "post_evaluation_data_used": False,
        },
    })
    payload["result_sha256"] = hashlib.sha256(
        json.dumps({key: value for key, value in payload.items() if key != "result_sha256"},
                   sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    old_result = output_root / "result_london_asian_range_breakout_v36.json"
    result = output_root / "result_london_asian_range_breakout_v37.json"
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
