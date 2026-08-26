from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, pip_size, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS, PERIODS, timestamp
from run_liquid_major_universe_v9 import UNIVERSE
from run_major_session_horizon_v10 import summarize


HORIZON = 576


def rescore(bars: list[Bar], fill_index: int, direction: int, arm: str) -> dict | None:
    exit_index = fill_index + HORIZON
    if exit_index >= len(bars):
        return None
    entry, exit_bar = bars[fill_index], bars[exit_index]
    gross = exit_bar.mid_c / entry.mid_o - 1.0 if direction > 0 else entry.mid_o / exit_bar.mid_c - 1.0
    scenario = ARMS[arm]
    if scenario is None:
        net = gross
    else:
        slip = float(scenario["slippage"]) * pip_size(entry.pair)
        if direction > 0:
            net = (exit_bar.bid_c - slip) / (entry.ask_o + slip) - 1.0
        else:
            net = (entry.bid_o - slip) / (exit_bar.ask_c + slip) - 1.0
        elapsed_days = (timestamp(exit_bar.time) - timestamp(entry.time)).total_seconds() / 86400.0
        net -= 2.0 * float(scenario["commission"]) * 1e-4
        net -= float(scenario["financing"]) * 1e-4 * elapsed_days
    return {"arm": arm, "gross_return": gross, "net_return": net}


def run(input_root: Path, source_ledger: Path, output_root: Path) -> dict:
    raw_source = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    source_rows = [{key: row[key] for key in ("signal_id", "pair", "fill_time", "direction")} for row in raw_source]
    corpus = {}
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
    index = {pair: {bar.time: i for i, bar in enumerate(bars)} for pair, bars in corpus.items()}
    rows = []
    for source in source_rows:
        pair = source["pair"]
        fill_index = index[pair].get(source["fill_time"])
        if fill_index is None:
            raise ValueError(f"source fill is absent from corpus: {source['signal_id']}")
        scores = {arm: rescore(corpus[pair], fill_index, int(source["direction"]), arm) for arm in ARMS}
        if any(value is None for value in scores.values()):
            continue
        rows.append({
            **source, "exit_time": corpus[pair][fill_index + HORIZON].time,
            "horizon_m5_bars": HORIZON, "scores": scores,
        })
    periods = {name: summarize(rows, *bounds) for name, bounds in PERIODS.items()}
    walk = periods["WALK_FORWARD"]["arms"]
    admitted = walk["RAW_SIGNAL"]["signals"] >= 20 and all(
        walk[arm]["mean_return"] is not None and walk[arm]["mean_return"] > 0 for arm in ARMS
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_two_day_h576_v14.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "experiment": "FX_AUCTION_TRAP_LIQUID_MAJORS_H576_V14",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "fixed_horizon_384_to_576_m5_bars",
        "family_stop_rule": "stop fixed-horizon extension if adverse mean is not positive",
        "universe": sorted(UNIVERSE), "horizon_m5_bars": HORIZON,
        "source_ledger": str(source_ledger), "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "direction"],
        "source_outcome_fields_consumed": False,
        "source_signals": len(source_rows), "rescored_signals": len(rows), "source_audit": source_audit,
        "cost_suppressed_raw_signals": 0, "same_signal_id_all_cost_arms": True,
        "period_membership_requires_contained_exit": True,
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
    result = output_root / "result_two_day_horizon_v14.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.source_ledger, args.output_root)
    print(json.dumps({
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "development_admitted": result["development_admitted"], "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
