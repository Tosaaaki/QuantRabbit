from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import load_bars, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS
from run_tuning_worker_admission_v17 import simulate_selected


LOOKBACK = 12
HORIZON = 384


def graph_alignment(pair: str, fill_time: str, direction: int, corpus, index) -> tuple[float, int] | None:
    strengths = defaultdict(list)
    used = 0
    for other in sorted(UNIVERSE - {pair}):
        i = index[other].get(fill_time)
        if i is None or i < LOOKBACK:
            continue
        bars = corpus[other]
        closes = [bar.mid_c for bar in bars[i - LOOKBACK:i]]
        steps = [math.log(right / left) for left, right in zip(closes, closes[1:])]
        energy = math.sqrt(sum(value * value for value in steps))
        if energy <= 0:
            continue
        normalized = math.log(closes[-1] / closes[0]) / energy
        base, quote = other.split("_")
        strengths[base].append(normalized)
        strengths[quote].append(-normalized)
        used += 1
    if used < 4:
        return None
    base, quote = pair.split("_")
    base_strength = statistics.fmean(strengths[base]) if strengths[base] else 0.0
    quote_strength = statistics.fmean(strengths[quote]) if strengths[quote] else 0.0
    escape_side = -direction
    return escape_side * (base_strength - quote_strength), used


def run(input_root: Path, source_ledger: Path, output_root: Path) -> dict:
    raw_source = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    source_rows = [{key: row[key] for key in ("signal_id", "pair", "fill_time", "direction")} for row in raw_source if row["pair"] in UNIVERSE]
    corpus, source_audit = {}, []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
    index = {pair: {bar.time: i for i, bar in enumerate(bars)} for pair, bars in corpus.items()}
    rows, unavailable = [], 0
    for source in source_rows:
        pair = source["pair"]
        fill_index = index[pair].get(source["fill_time"])
        if fill_index is None or fill_index + HORIZON >= len(corpus[pair]):
            continue
        measured = graph_alignment(pair, source["fill_time"], int(source["direction"]), corpus, index)
        if measured is None:
            unavailable += 1
            continue
        alignment, graph_pairs = measured
        if alignment > 0:
            continue
        rows.append({
            **source, "exit_time": corpus[pair][fill_index + HORIZON].time,
            "network_alignment": alignment, "graph_pairs": graph_pairs,
        })
    periods = {
        name: {arm: simulate_selected(corpus, rows, set(UNIVERSE), arm, start, end) for arm in ARMS}
        for name, (start, end) in PERIODS.items()
    }
    admitted = all(
        periods[name]["RAW_SIGNAL"]["source_signals"] >= 20
        and periods[name][arm]["equity_multiple"] > 1.0
        for name in PERIODS for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_currency_network_disconfirmation_v18.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "experiment": "FX_CURRENCY_NETWORK_DISCONFIRMATION_V18",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "excluding_self_currency_network_disconfirmation_gate",
        "original_indicator": {"lookback_m5_bars": LOOKBACK, "minimum_other_pairs": 4, "gate": "alignment<=0"},
        "source_ledger": str(source_ledger), "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "direction"],
        "source_outcome_fields_consumed": False,
        "source_major_signals": len(source_rows), "admitted_raw_signals": len(rows),
        "graph_unavailable": unavailable, "cost_suppressed_raw_signals": 0,
        "proposal_ledger": str(ledger), "proposal_ledger_sha256": sha256_file(ledger),
        "periods": periods, "source_audit": source_audit,
        "same_signal_stream_all_cost_arms": True,
        "development_admitted": admitted, "final_admitted": False,
        "terminal_inventory_mtm_hidden": False, "live_authority": False, "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_currency_network_disconfirmation_v18.json"
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
        "source_major_signals": result["source_major_signals"],
        "admitted_raw_signals": result["admitted_raw_signals"],
        "periods": result["periods"], "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
