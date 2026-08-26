from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import load_bars, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS
from run_currency_network_disconfirmation_v18 import HORIZON, LOOKBACK, graph_alignment
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS
from run_tuning_worker_admission_v17 import TUNING, simulate_selected


def tuning_median_positive(measurements: list[dict]) -> tuple[float, int]:
    values = [
        float(row["network_alignment"])
        for row in measurements
        if TUNING[0] <= row["fill_time"][:10] < TUNING[1]
        and float(row["network_alignment"]) > 0.0
    ]
    if not values:
        raise ValueError("no positive tuning network alignments")
    return statistics.median(values), len(values)


def run(input_root: Path, source_ledger: Path, output_root: Path) -> dict:
    raw_source = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    source_rows = [
        {key: row[key] for key in ("signal_id", "pair", "fill_time", "direction")}
        for row in raw_source if row["pair"] in UNIVERSE
    ]
    corpus, source_audit = {}, []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
    index = {pair: {bar.time: i for i, bar in enumerate(bars)} for pair, bars in corpus.items()}
    measured_rows, unavailable = [], 0
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
        measured_rows.append({
            **source,
            "exit_time": corpus[pair][fill_index + HORIZON].time,
            "network_alignment": alignment,
            "graph_pairs": graph_pairs,
        })
    threshold, tuning_positive_count = tuning_median_positive(measured_rows)
    rows = [row for row in measured_rows if float(row["network_alignment"]) >= threshold]
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
    ledger = output_root / "proposal_ledger_currency_network_exhaustion_intensity_v20.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
    )
    payload = {
        "experiment": "FX_CURRENCY_NETWORK_EXHAUSTION_INTENSITY_V20",
        "family": "V18_V20_CURRENCY_NETWORK",
        "family_hypotheses": 3,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "tuning_median_positive_network_alignment_gate",
        "original_indicator": {
            "lookback_m5_bars": LOOKBACK,
            "minimum_other_pairs": 4,
            "gate": "alignment>=tuning_median_positive_alignment",
            "tuning_period": list(TUNING),
            "tuning_positive_count": tuning_positive_count,
            "frozen_threshold": threshold,
            "walk_forward_used_for_threshold": False,
        },
        "source_ledger": str(source_ledger),
        "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "direction"],
        "source_outcome_fields_consumed": False,
        "source_major_signals": len(source_rows),
        "measured_signals": len(measured_rows),
        "admitted_raw_signals": len(rows),
        "graph_unavailable": unavailable,
        "cost_suppressed_raw_signals": 0,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": sha256_file(ledger),
        "periods": periods,
        "source_audit": source_audit,
        "same_signal_stream_all_cost_arms": True,
        "development_admitted": admitted,
        "final_admitted": False,
        "terminal_inventory_mtm_hidden": False,
        "live_authority": False,
        "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "family correction for V18/V19/V20 is not yet complete",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_currency_network_exhaustion_intensity_v20.json"
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
        "admitted_raw_signals": result["admitted_raw_signals"],
        "original_indicator": result["original_indicator"],
        "periods": result["periods"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
