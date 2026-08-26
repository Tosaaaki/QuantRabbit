from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from run_auction_trap_geometry_v7 import ARMS, PERIODS


UNIVERSE = {"AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict], start: str, end: str) -> dict:
    selected = [
        row for row in rows
        if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end
    ]
    arms = {}
    for arm in ARMS:
        values = [row["scores"][arm]["net_return"] for row in selected]
        arms[arm] = {
            "signals": len(values),
            "mean_return": statistics.fmean(values) if values else None,
            "median_return": statistics.median(values) if values else None,
            "positive_rate": sum(value > 0 for value in values) / len(values) if values else None,
            "additive_return": sum(values),
        }
    return {"start": start, "end": end, "arms": arms}


def run(source_ledger: Path, output_root: Path) -> dict:
    source_rows = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    selected = [row for row in source_rows if row["pair"] in UNIVERSE]
    periods = {name: summarize(selected, *bounds) for name, bounds in PERIODS.items()}
    walk = periods["WALK_FORWARD"]["arms"]
    admitted = walk["RAW_SIGNAL"]["signals"] >= 20 and all(
        walk[arm]["mean_return"] is not None and walk[arm]["mean_return"] > 0 for arm in ARMS
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_liquid_major_universe_v9.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in selected), encoding="utf-8")
    payload = {
        "experiment": "FX_AUCTION_TRAP_H96_LIQUID_MAJORS_V9",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "predefined_7_major_universe",
        "universe": sorted(UNIVERSE),
        "source_ledger": str(source_ledger), "source_ledger_sha256": sha256_file(source_ledger),
        "source_signals": len(source_rows), "raw_signals": len(selected),
        "excluded_outside_universe": len(source_rows) - len(selected),
        "cost_suppressed_raw_signals": 0, "same_signal_id_all_cost_arms": True,
        "periods": periods, "development_admitted": admitted, "final_admitted": False,
        "proposal_ledger": str(ledger), "proposal_ledger_sha256": sha256_file(ledger),
        "terminal_open_inventory": 0, "terminal_inventory_mtm_hidden": False,
        "live_authority": False, "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_liquid_major_universe_v9.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_ledger, args.output_root)
    print(json.dumps({
        "universe": result["universe"], "raw_signals": result["raw_signals"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "development_admitted": result["development_admitted"], "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
