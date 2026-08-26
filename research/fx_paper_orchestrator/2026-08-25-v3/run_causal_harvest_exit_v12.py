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

from fx_original_indicators import Bar, load_bars, pip_size, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS, PERIODS, timestamp
from run_liquid_major_universe_v9 import UNIVERSE


MAX_AGE = 384
TARGET_QUANTILE = 0.35
TUNING_END = "2026-05-01"


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0 < quantile <= 1:
        raise ValueError("nearest-rank quantile requires values and q in (0,1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def signed_mid_return(entry: Bar, current: Bar, direction: int, use_open: bool = False) -> float:
    price = current.mid_o if use_open else current.mid_c
    return price / entry.mid_o - 1.0 if direction > 0 else entry.mid_o / price - 1.0


def maximum_favorable_close(bars: list[Bar], fill_index: int, direction: int) -> float | None:
    if fill_index + MAX_AGE + 1 >= len(bars):
        return None
    entry = bars[fill_index]
    return max(signed_mid_return(entry, bars[fill_index + step], direction) for step in range(1, MAX_AGE + 1))


def freeze_targets(source_rows: list[dict], corpus: dict[str, list[Bar]], index: dict[str, dict[str, int]]) -> dict:
    samples: dict[str, list[float]] = defaultdict(list)
    for source in source_rows:
        if source["fill_time"][:10] >= TUNING_END:
            continue
        pair = source["pair"]
        fill_index = index[pair].get(source["fill_time"])
        if fill_index is None or fill_index + MAX_AGE + 1 >= len(corpus[pair]):
            continue
        if corpus[pair][fill_index + MAX_AGE + 1].time[:10] >= TUNING_END:
            continue
        value = maximum_favorable_close(corpus[pair], fill_index, int(source["direction"]))
        if value is not None:
            samples[pair].append(value)
    pooled = [value for values in samples.values() for value in values]
    global_target = nearest_rank(pooled, TARGET_QUANTILE)
    targets = {
        pair: nearest_rank(samples[pair], TARGET_QUANTILE) if len(samples[pair]) >= 20 else global_target
        for pair in sorted(UNIVERSE)
    }
    return {
        "quantile": TARGET_QUANTILE,
        "max_age_m5_bars": MAX_AGE,
        "global_target": global_target,
        "pair_targets": targets,
        "pair_sample_counts": {pair: len(samples[pair]) for pair in sorted(UNIVERSE)},
        "pooled_sample_count": len(pooled),
        "tuning_end_exclusive": TUNING_END,
    }


def score_dynamic(bars: list[Bar], fill_index: int, direction: int, target: float, arm: str) -> dict | None:
    if fill_index + MAX_AGE + 1 >= len(bars):
        return None
    entry = bars[fill_index]
    decision_step = MAX_AGE
    tp_hit = False
    for step in range(1, MAX_AGE + 1):
        if signed_mid_return(entry, bars[fill_index + step], direction) >= target:
            decision_step = step
            tp_hit = True
            break
    exit_bar = bars[fill_index + decision_step + 1]
    gross = signed_mid_return(entry, exit_bar, direction, use_open=True)
    scenario = ARMS[arm]
    if scenario is None:
        net = gross
    else:
        slip = float(scenario["slippage"]) * pip_size(entry.pair)
        if direction > 0:
            net = (exit_bar.bid_o - slip) / (entry.ask_o + slip) - 1.0
        else:
            net = (entry.bid_o - slip) / (exit_bar.ask_o + slip) - 1.0
        elapsed_days = (timestamp(exit_bar.time) - timestamp(entry.time)).total_seconds() / 86400.0
        net -= 2.0 * float(scenario["commission"]) * 1e-4
        net -= float(scenario["financing"]) * 1e-4 * elapsed_days
    return {
        "arm": arm, "gross_return": gross, "net_return": net,
        "tp_hit": tp_hit, "decision_age_m5_bars": decision_step,
        "exit_time": exit_bar.time,
    }


def summarize(rows: list[dict], start: str, end: str) -> dict:
    selected = [
        row for row in rows
        if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end
    ]
    arms = {}
    for arm in ARMS:
        values = [row["scores"][arm]["net_return"] for row in selected]
        arms[arm] = {
            "signals": len(values), "mean_return": statistics.fmean(values) if values else None,
            "median_return": statistics.median(values) if values else None,
            "positive_rate": sum(value > 0 for value in values) / len(values) if values else None,
            "additive_return": sum(values),
        }
    ages = [row["decision_age_m5_bars"] for row in selected]
    return {
        "start": start, "end": end, "arms": arms,
        "tp_hit_rate": sum(row["tp_hit"] for row in selected) / len(selected) if selected else None,
        "mean_decision_age_m5_bars": statistics.fmean(ages) if ages else None,
        "max_age_liquidations": sum(not row["tp_hit"] for row in selected),
    }


def run(input_root: Path, source_ledger: Path, output_root: Path) -> dict:
    raw_source = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    source_rows = [
        {key: row[key] for key in ("signal_id", "pair", "fill_time", "direction")}
        for row in raw_source
    ]
    corpus = {}
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
    index = {pair: {bar.time: i for i, bar in enumerate(bars)} for pair, bars in corpus.items()}
    target_freeze = freeze_targets(source_rows, corpus, index)
    rows = []
    for source in source_rows:
        pair = source["pair"]
        fill_index = index[pair].get(source["fill_time"])
        if fill_index is None:
            raise ValueError(f"source fill is absent from corpus: {source['signal_id']}")
        target = target_freeze["pair_targets"][pair]
        scores = {arm: score_dynamic(corpus[pair], fill_index, int(source["direction"]), target, arm) for arm in ARMS}
        if any(value is None for value in scores.values()):
            continue
        raw_score = scores["RAW_SIGNAL"]
        rows.append({
            **source, "exit_time": raw_score["exit_time"], "target_return": target,
            "tp_hit": raw_score["tp_hit"], "decision_age_m5_bars": raw_score["decision_age_m5_bars"],
            "scores": scores,
        })
    periods = {name: summarize(rows, *bounds) for name, bounds in PERIODS.items()}
    walk = periods["WALK_FORWARD"]["arms"]
    admitted = walk["RAW_SIGNAL"]["signals"] >= 20 and all(
        walk[arm]["mean_return"] is not None and walk[arm]["mean_return"] > 0 for arm in ARMS
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_causal_harvest_exit_v12.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "experiment": "FX_AUCTION_TRAP_CAUSAL_HARVEST_Q35_V12",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "frozen_q35_causal_harvest_exit_with_384_bar_max_age",
        "universe": sorted(UNIVERSE), "source_ledger": str(source_ledger),
        "source_ledger_sha256": sha256_file(source_ledger), "source_fields_consumed": ["signal_id", "pair", "fill_time", "direction"],
        "source_outcome_fields_consumed": False, "source_signals": len(source_rows), "scored_signals": len(rows),
        "target_freeze": target_freeze, "source_audit": source_audit,
        "cost_suppressed_raw_signals": 0, "same_signal_id_all_cost_arms": True,
        "individual_price_sl": False, "period_membership_requires_contained_exit": True,
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
    result = output_root / "result_causal_harvest_exit_v12.json"
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
        "target_freeze": result["target_freeze"], "walk_forward": result["periods"]["WALK_FORWARD"],
        "development_admitted": result["development_admitted"], "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
