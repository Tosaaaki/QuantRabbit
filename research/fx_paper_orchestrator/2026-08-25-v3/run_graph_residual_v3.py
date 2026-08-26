from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from indicator_factory_v3 import canonical, digest, metrics, period


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))
from fx_original_indicators import (  # noqa: E402
    Bar, aggregate_bars, load_bars, pip_size, sha256_file,
)


TIMEFRAMES = (60, 240)
HORIZONS = (1, 3, 6, 12)
WORKERS = (
    "GRAPH_PROPAGATION",
    "GRAPH_RESIDUAL_REVERSION",
    "GRAPH_COHERENT_MOMENTUM",
    "GRAPH_LAG_CATCHUP",
)
SCENARIOS = {
    "RAW_SIGNAL": {"slippage_pips": 0.0, "commission_bps": 0.0, "financing_bps_day": 0.0},
    "EXECUTABLE_BASE": {"slippage_pips": 0.3, "commission_bps": 0.0, "financing_bps_day": 0.5},
    "ADVERSE_STRESS": {"slippage_pips": 0.9, "commission_bps": 0.4, "financing_bps_day": 1.5},
}


def epoch(stamp: str) -> int:
    return int(datetime.fromisoformat(stamp[:19]).replace(tzinfo=timezone.utc).timestamp())


def returns_at_time(corpus: dict[str, list[Bar]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for pair, bars in corpus.items():
        for previous, current in zip(bars, bars[1:]):
            result[current.time][pair] = math.log(current.mid_c / previous.mid_c)
    return result


def leave_one_out_consensus(pair_returns: dict[str, float], target_pair: str) -> float | None:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for pair, value in pair_returns.items():
        if pair == target_pair:
            continue
        base, quote = pair.split("_")
        sums[base] += value
        sums[quote] -= value
        counts[base] += 1
        counts[quote] += 1
    base, quote = target_pair.split("_")
    if counts[base] < 3 or counts[quote] < 3:
        return None
    return sums[base] / counts[base] - sums[quote] / counts[quote]


def worker_direction(worker: str, target_return: float, consensus: float) -> int | None:
    target_sign = 1 if target_return > 0 else -1 if target_return < 0 else 0
    consensus_sign = 1 if consensus > 0 else -1 if consensus < 0 else 0
    residual = target_return - consensus
    residual_sign = 1 if residual > 0 else -1 if residual < 0 else 0
    if worker == "GRAPH_PROPAGATION":
        return consensus_sign or None
    if worker == "GRAPH_RESIDUAL_REVERSION":
        return -residual_sign if residual_sign else None
    if worker == "GRAPH_COHERENT_MOMENTUM":
        return target_sign if target_sign and target_sign == consensus_sign else None
    if worker == "GRAPH_LAG_CATCHUP":
        return consensus_sign if target_sign and consensus_sign and target_sign != consensus_sign else None
    raise ValueError(worker)


def score(pair: str, direction: int, entry: Bar, exit_bar: Bar, elapsed_days: float) -> dict[str, float]:
    output = {}
    for arm, scenario in SCENARIOS.items():
        slip = scenario["slippage_pips"] * pip_size(pair)
        if arm == "RAW_SIGNAL":
            value = direction * (exit_bar.mid_o / entry.mid_o - 1.0)
        elif direction > 0:
            value = (exit_bar.bid_o - slip) / (entry.ask_o + slip) - 1.0
        else:
            value = (entry.bid_o - slip) / (exit_bar.ask_o + slip) - 1.0
        value -= scenario["commission_bps"] * 1e-4
        value -= scenario["financing_bps_day"] * 1e-4 * elapsed_days
        output[arm] = value
    return output


def generate(corpus: dict[str, list[Bar]], timeframe: int) -> list[dict]:
    indexed = {pair: {bar.time: i for i, bar in enumerate(bars)} for pair, bars in corpus.items()}
    cross = returns_at_time(corpus)
    records = []
    for stamp, pair_returns in sorted(cross.items()):
        if len(pair_returns) < 20:
            continue
        for pair, target_return in pair_returns.items():
            bars = corpus[pair]
            i = indexed[pair][stamp]
            consensus = leave_one_out_consensus(pair_returns, pair)
            if consensus is None:
                continue
            residual = target_return - consensus
            for worker in WORKERS:
                direction = worker_direction(worker, target_return, consensus)
                if direction is None:
                    continue
                for horizon in HORIZONS:
                    fill_i = i + 1
                    exit_i = fill_i + horizon
                    if exit_i >= len(bars):
                        continue
                    expected_gap = timeframe * 60
                    path = bars[i:exit_i + 1]
                    if any(epoch(right.time) - epoch(left.time) != expected_gap for left, right in zip(path, path[1:])):
                        continue
                    entry, exit_bar = bars[fill_i], bars[exit_i]
                    elapsed_days = (epoch(exit_bar.time) - epoch(entry.time)) / 86400.0
                    signal_id = digest({
                        "timeframe": timeframe, "worker": worker, "pair": pair,
                        "decision_bar": stamp, "horizon": horizon,
                    })[:24]
                    records.append({
                        "signal_id": signal_id, "pair": pair, "worker": worker,
                        "timeframe_minutes": timeframe, "horizon_bars": horizon,
                        "decision_bar_start": stamp, "fill_time": entry.time,
                        "exit_time": exit_bar.time, "direction": direction,
                        "target_return": target_return, "loo_consensus": consensus,
                        "graph_residual": residual,
                        "returns": score(pair, direction, entry, exit_bar, elapsed_days),
                    })
    return records


def evaluate(records: list[dict], output: Path, source_audit: dict) -> dict:
    family_tests = len(TIMEFRAMES) * len(WORKERS) * len(HORIZONS)
    z = statistics.NormalDist().inv_cdf(1 - .05 / (2 * family_tests))
    candidates = []
    for timeframe in TIMEFRAMES:
        for worker in WORKERS:
            for horizon in HORIZONS:
                rows = [row for row in records if row["timeframe_minutes"] == timeframe
                        and row["worker"] == worker and row["horizon_bars"] == horizon]
                periods = {
                    p: [row for row in rows if period(row["fill_time"]) == p]
                    for p in ("TUNING", "WALK_FORWARD", "OPENED_DIAGNOSTIC")
                }
                if len(periods["TUNING"]) < 20 or len(periods["WALK_FORWARD"]) < 20:
                    continue
                summaries = {
                    p: {arm: metrics(xs, arm, z) for arm in SCENARIOS}
                    for p, xs in periods.items()
                }
                corrected = [
                    summaries["TUNING"]["RAW_SIGNAL"]["family_corrected_cluster_lower_bps"],
                    summaries["WALK_FORWARD"]["EXECUTABLE_BASE"]["family_corrected_cluster_lower_bps"],
                    summaries["WALK_FORWARD"]["ADVERSE_STRESS"]["family_corrected_cluster_lower_bps"],
                ]
                uncorrected = [
                    summaries["TUNING"]["RAW_SIGNAL"]["cluster_lower_bps"],
                    summaries["WALK_FORWARD"]["EXECUTABLE_BASE"]["cluster_lower_bps"],
                    summaries["WALK_FORWARD"]["ADVERSE_STRESS"]["cluster_lower_bps"],
                ]
                candidates.append({
                    "candidate_id": digest({"timeframe": timeframe, "worker": worker, "horizon": horizon})[:24],
                    "timeframe_minutes": timeframe, "worker": worker,
                    "horizon_bars": horizon, "horizon_hours": horizon * timeframe / 60,
                    "periods": summaries,
                    "development_admitted": all(value is not None and value > 0 for value in corrected),
                    "ranking_floor_bps": min((value for value in uncorrected if value is not None), default=-math.inf),
                })
    candidates.sort(key=lambda item: (item["development_admitted"], item["ranking_floor_bps"]), reverse=True)
    payload = {
        "experiment": "FX_LEAVE_ONE_OUT_CURRENCY_GRAPH_RESIDUAL_V3",
        "evidence_class": "opened_development_not_future_holdout",
        "hypotheses": list(WORKERS),
        "family_tests": family_tests,
        "family_corrected_z": z,
        "raw_signals": len(records),
        "evaluated_candidates": len(candidates),
        "development_admitted_count": sum(item["development_admitted"] for item in candidates),
        "final_admitted_count": 0,
        "top_candidates": candidates,
        "source_audit": source_audit,
        "raw_cost_gate": False,
        "target_pair_excluded_from_graph_consensus": True,
        "admission_blockers": [
            "all data are already opened development evidence",
            "portfolio inventory and terminal MTM are not evaluated",
            "new future holdout has not elapsed",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.input_root.glob("*/*_M5_BA_*.jsonl.gz"))
    if len(files) != 28:
        raise SystemExit(f"expected 28 exact pair files, got {len(files)}")
    source_audit = {path.parent.name: {"path": str(path), "sha256": sha256_file(path)} for path in files}
    all_records = []
    for timeframe in TIMEFRAMES:
        corpus = {path.parent.name: aggregate_bars(load_bars(path), timeframe) for path in files}
        all_records.extend(generate(corpus, timeframe))
    args.output_root.mkdir(parents=True, exist_ok=True)
    ledger = args.output_root / "signal_ledger_graph_v3.jsonl.gz"
    with gzip.open(ledger, "wt", encoding="utf-8") as handle:
        for row in all_records:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    result = evaluate(all_records, args.output_root / "result_graph_v3.json", source_audit)
    result["ledger"] = str(ledger)
    result["ledger_sha256"] = sha256_file(ledger)
    # Write again so the final result seals the exact ledger.
    result["result_sha256"] = hashlib.sha256(canonical({k: v for k, v in result.items() if k != "result_sha256"}).encode()).hexdigest()
    (args.output_root / "result_graph_v3.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "raw_signals": result["raw_signals"],
        "evaluated": result["evaluated_candidates"],
        "development_admitted": result["development_admitted_count"],
        "final_admitted": result["final_admitted_count"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
