from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from indicator_factory_v3 import canonical, lower_bound, period, week_key
from run_graph_residual_v3 import (
    SCENARIOS, aggregate_bars, epoch, leave_one_out_consensus,
    load_bars, pip_size, returns_at_time, sha256_file,
)


EXECUTION_PAIRS = ("AUD_USD", "EUR_USD", "GBP_USD", "NZD_USD", "USD_CAD", "USD_CHF", "USD_JPY")
TOP_K = (2, 4, 7)
REBALANCE_BARS = (1, 3, 6)
TIMEFRAME = 240


def target_weights(pair_returns: dict[str, float], top_k: int) -> dict[str, float]:
    scores = {}
    for pair in EXECUTION_PAIRS:
        if pair not in pair_returns:
            continue
        consensus = leave_one_out_consensus(pair_returns, pair)
        if consensus is not None and consensus != 0:
            scores[pair] = consensus
    chosen = sorted(scores, key=lambda pair: abs(scores[pair]), reverse=True)[:top_k]
    if len(chosen) != top_k:
        return {}
    return {pair: (1.0 if scores[pair] > 0 else -1.0) / top_k for pair in chosen}


def one_way_cost(pair: str, bar, arm: str) -> float:
    if arm == "RAW_SIGNAL":
        return 0.0
    scenario = SCENARIOS[arm]
    mid = bar.mid_o
    half_spread = (bar.ask_o - bar.bid_o) / (2.0 * mid)
    slippage = scenario["slippage_pips"] * pip_size(pair) / mid
    # SCENARIOS stores adverse round-trip commission as 0.4bp.  A rebalance
    # delta is one-way turnover, so charge half here and again on closure.
    commission = scenario["commission_bps"] * 0.5 * 1e-4
    return half_spread + slippage + commission


def interval_gross(pair: str, weight: float, entry, exit_bar) -> float:
    if weight > 0:
        return abs(weight) * (exit_bar.mid_o / entry.mid_o - 1.0)
    return abs(weight) * (entry.mid_o / exit_bar.mid_o - 1.0)


def max_drawdown(returns: list[float]) -> float:
    equity = peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1.0)
    return drawdown


def simulate(targets: list[dict], bar_maps: dict, top_k: int, rebalance_bars: int,
             arm: str, selected_period: str, corrected_z: float) -> dict:
    xs = [item for item in targets if period(item["fill_time"]) == selected_period
          and period(item["exit_time"]) == selected_period]
    if not xs:
        return {"intervals": 0}
    positions: dict[str, float] = {}
    returns = []
    timestamps = []
    turnover = 0.0
    cost_drag = 0.0
    for index, item in enumerate(xs):
        fill, exit_time = item["fill_time"], item["exit_time"]
        target = item[f"target_top_{top_k}"] if index % rebalance_bars == 0 else positions
        if not target:
            continue
        all_pairs = set(positions) | set(target)
        rebalance_cost = 0.0
        for pair in all_pairs:
            delta = target.get(pair, 0.0) - positions.get(pair, 0.0)
            turnover += abs(delta)
            rebalance_cost += abs(delta) * one_way_cost(pair, bar_maps[pair][fill], arm)
        positions = dict(target)
        gross = sum(interval_gross(
            pair, weight, bar_maps[pair][fill], bar_maps[pair][exit_time]
        ) for pair, weight in positions.items())
        elapsed_days = (epoch(exit_time) - epoch(fill)) / 86400.0
        financing = (0.0 if arm == "RAW_SIGNAL" else
                     sum(abs(weight) for weight in positions.values())
                     * SCENARIOS[arm]["financing_bps_day"] * 1e-4 * elapsed_days)
        net = gross - rebalance_cost - financing
        returns.append(net)
        timestamps.append(fill)
        cost_drag += rebalance_cost + financing
    if not returns:
        return {"intervals": 0}
    terminal_time = xs[-1]["exit_time"]
    terminal_cost = sum(
        abs(weight) * one_way_cost(pair, bar_maps[pair][terminal_time], arm)
        for pair, weight in positions.items()
    )
    turnover += sum(abs(weight) for weight in positions.values())
    returns[-1] -= terminal_cost
    cost_drag += terminal_cost
    by_week: dict[str, list[float]] = defaultdict(list)
    by_month: dict[str, list[float]] = defaultdict(list)
    for stamp, value in zip(timestamps, returns):
        by_week[week_key(stamp)].append(value * 10000)
        by_month[stamp[:7]].append(value)
    monthly = {month: math.prod(1.0 + value for value in values)
               for month, values in sorted(by_month.items())}
    return {
        "intervals": len(returns),
        "mean_interval_bps": statistics.fmean(returns) * 10000,
        "cluster_lower_bps": lower_bound(by_week, 1.959963984540054),
        "family_corrected_cluster_lower_bps": lower_bound(by_week, corrected_z),
        "total_multiple": math.prod(1.0 + value for value in returns),
        "monthly_multiples": monthly,
        "worst_month_multiple": min(monthly.values()),
        "months_at_or_above_2x": sum(value >= 2.0 for value in monthly.values()),
        "max_drawdown": max_drawdown(returns),
        "turnover_units": turnover,
        "cost_drag_fraction": cost_drag,
        "terminal_liquidation_cost_fraction": terminal_cost,
        "terminal_inventory_units_before_liquidation": sum(abs(value) for value in positions.values()),
        "terminal_inventory_mtm_hidden": False,
        "final_inventory_units": 0.0,
    }


def build_targets(corpus: dict[str, list], cross_returns: dict) -> tuple[list[dict], dict]:
    bar_maps = {pair: {bar.time: bar for bar in corpus[pair]} for pair in EXECUTION_PAIRS}
    common = set.intersection(*(set(values) for values in bar_maps.values()))
    stamps = sorted(stamp for stamp in common if stamp in cross_returns and len(cross_returns[stamp]) >= 20)
    targets = []
    for decision, fill, exit_time in zip(stamps, stamps[1:], stamps[2:]):
        pair_returns = cross_returns[decision]
        item = {"decision_bar_start": decision, "fill_time": fill, "exit_time": exit_time}
        for top_k in TOP_K:
            item[f"target_top_{top_k}"] = target_weights(pair_returns, top_k)
        if all(item[f"target_top_{top_k}"] for top_k in TOP_K):
            targets.append(item)
    return targets, bar_maps


def run(input_root: Path, output_root: Path) -> dict:
    files = sorted(input_root.glob("*/*_M5_BA_*.jsonl.gz"))
    if len(files) != 28:
        raise ValueError(f"expected 28 source pairs, got {len(files)}")
    corpus = {path.parent.name: aggregate_bars(load_bars(path), TIMEFRAME) for path in files}
    targets, bar_maps = build_targets(corpus, returns_at_time(corpus))
    family_tests = len(TOP_K) * len(REBALANCE_BARS)
    z = statistics.NormalDist().inv_cdf(1 - .05 / (2 * family_tests))
    candidates = []
    for top_k in TOP_K:
        for rebalance in REBALANCE_BARS:
            results = {
                p: {arm: simulate(targets, bar_maps, top_k, rebalance, arm, p, z)
                    for arm in SCENARIOS}
                for p in ("TUNING", "WALK_FORWARD", "OPENED_DIAGNOSTIC")
            }
            corrected = [
                results["TUNING"]["RAW_SIGNAL"].get("family_corrected_cluster_lower_bps"),
                results["WALK_FORWARD"]["EXECUTABLE_BASE"].get("family_corrected_cluster_lower_bps"),
                results["WALK_FORWARD"]["ADVERSE_STRESS"].get("family_corrected_cluster_lower_bps"),
            ]
            candidates.append({
                "candidate_id": f"GRAPH_NET_TOP{top_k}_REB{rebalance}_H4",
                "top_k": top_k, "rebalance_h4_bars": rebalance,
                "gross_leverage": 1.0,
                "results": results,
                "development_admitted": all(value is not None and value > 0 for value in corrected),
                "ranking_floor_bps": min((value for value in corrected if value is not None), default=-math.inf),
            })
    candidates.sort(key=lambda item: (item["development_admitted"], item["ranking_floor_bps"]), reverse=True)
    payload = {
        "experiment": "FX_GRAPH_INTERNAL_NETTING_INVENTORY_V3",
        "evidence_class": "opened_development_not_future_holdout",
        "source_sha256_by_pair": {path.parent.name: sha256_file(path) for path in files},
        "proposal_stream_count": len(targets),
        "same_target_stream_all_cost_arms": True,
        "raw_signal_cost_gate": False,
        "family_tests": family_tests,
        "family_corrected_z": z,
        "candidates": candidates,
        "development_admitted_count": sum(item["development_admitted"] for item in candidates),
        "final_admitted_count": 0,
        "terminal_inventory_mtm_hidden": False,
        "admission_blockers": [
            "the source period is opened development evidence",
            "future holdout has not elapsed",
            "full comparable months do not meet 2x in normal and adverse arms",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result_graph_inventory_v3.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print(json.dumps({
        "proposal_stream_count": result["proposal_stream_count"],
        "development_admitted": result["development_admitted_count"],
        "final_admitted": result["final_admitted_count"],
        "top": result["candidates"][0],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
