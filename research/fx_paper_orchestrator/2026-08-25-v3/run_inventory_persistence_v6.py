from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_counterparty_hourly_netting_v5 import (
    ARMS, PERIODS, SOURCE_CANDIDATE, build_targets, load_corpus, simulate,
)
from run_v250_partial_holdout_v3 import sha256_file


MAX_AGE_HOURS = 12


def run(input_root: Path, decision_ledger: Path, output_root: Path) -> dict:
    decisions = [json.loads(line) for line in decision_ledger.read_text().splitlines() if line]
    targets, target_audit = build_targets(decisions)
    corpus, time_index, source_audit = load_corpus(input_root)
    periods = {
        period_name: {
            arm: simulate(
                corpus, time_index, targets, arm, start, end,
                persistence_hours=MAX_AGE_HOURS,
            )
            for arm in ARMS
        }
        for period_name, (start, end) in PERIODS.items()
    }
    walk = periods["WALK_FORWARD"]
    development_admitted = all(
        walk[arm].get("equity_multiple") is not None and walk[arm]["equity_multiple"] > 1.0
        for arm in ARMS
    )
    payload = {
        "experiment": "FX_CRS_H12_INVENTORY_PERSISTENCE_V6",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "no-signal inventory persistence",
        "max_age_hours": MAX_AGE_HOURS,
        "source_candidate": SOURCE_CANDIDATE,
        "source_decision_ledger": str(decision_ledger),
        "source_decision_ledger_sha256": sha256_file(decision_ledger),
        "target_audit": target_audit,
        "portfolio": {
            "pair_count": len(corpus),
            "weight_per_pair": 1.0 / len(corpus),
            "gross_leverage_cap": 1.0,
            "individual_price_sl": False,
            "finite_max_age": True,
        },
        "periods": periods,
        "development_admitted": development_admitted,
        "final_admitted": False,
        "source_audit": source_audit,
        "cost_suppressed_source_signals": 0,
        "terminal_inventory_mtm_hidden": False,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "result_inventory_persistence_v6.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.decision_ledger, args.output_root)
    print(json.dumps({
        "target_audit": result["target_audit"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
