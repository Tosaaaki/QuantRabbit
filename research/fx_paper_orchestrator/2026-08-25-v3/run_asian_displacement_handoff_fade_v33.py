"""V33 one-shot recovery of the unobserved V32 strategy.

Only computed scheduled timestamp serialization changes: every decision, fill
and raw-exit timestamp is canonical RFC3339 UTC with exactly nine fractional
digits before it reaches the frozen V31 state planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import run_asian_displacement_handoff_fade_v32 as frozen_v32


CYCLE_ID = "V33"
EXPERIMENT = "FX_ASIAN_DISPLACEMENT_HANDOFF_FADE_V33"
V32_FAILURE_SHA256 = "86b073f7ad7604dc444c074d634c95897ca1859fbfbac092e9de37407c13594d"
_FROZEN_DETECT_DAY_SIGNALS = frozen_v32.detect_day_signals
_UTC = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d{1,9}))?Z$"
)


def canonical_utc_nine_digits(value: str) -> str:
    match = _UTC.fullmatch(value)
    if match is None:
        raise ValueError(f"timestamp is not canonical UTC: {value}")
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return f"{match.group('head')}.{fraction}Z"


def detect_day_signals(pair_day_bars: dict[str, list]) -> list[dict]:
    rows = _FROZEN_DETECT_DAY_SIGNALS(pair_day_bars)
    for row in rows:
        for field in ("decision_time", "fill_time", "exit_time"):
            original = row[field]
            canonical = canonical_utc_nine_digits(original)
            if frozen_v32.frozen_v31.ns(original) != frozen_v32.frozen_v31.ns(canonical):
                raise ValueError("timestamp canonicalization changed the represented instant")
            row[field] = canonical
    return rows


def run(input_root: Path, output_root: Path) -> dict:
    original_detector = frozen_v32.detect_day_signals
    frozen_v32.detect_day_signals = detect_day_signals
    try:
        payload = frozen_v32.run(input_root, output_root)
    finally:
        frozen_v32.detect_day_signals = original_detector

    old_ledger = output_root / "proposal_ledger_asian_displacement_handoff_fade_v32.jsonl"
    ledger = output_root / "proposal_ledger_asian_displacement_handoff_fade_v33.jsonl"
    old_ledger.replace(ledger)
    payload["cycle_id"] = CYCLE_ID
    payload["experiment"] = EXPERIMENT
    payload["proposal_ledger"] = str(ledger)
    payload["proposal_ledger_sha256"] = frozen_v32.sha256_file(ledger)
    payload["runtime_compatibility_provenance"] = {
        "classification": "NON_STRATEGY_RUNTIME_COMPATIBILITY",
        "changed_strategy_variables": 0,
        "same_unobserved_v32_strategy": True,
        "v32_rerun_permitted": False,
        "v32_failure_evidence_sha256": V32_FAILURE_SHA256,
        "single_change": "computed scheduled timestamps canonicalized to exactly nine fractional UTC digits",
        "instant_changed": False,
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "result_sha256"},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    ).hexdigest()
    old_result = output_root / "result_asian_displacement_handoff_fade_v32.json"
    result = output_root / "result_asian_displacement_handoff_fade_v33.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    old_result.unlink()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
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
