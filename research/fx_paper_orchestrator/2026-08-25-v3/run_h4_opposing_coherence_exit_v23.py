from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import timedelta
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS, timestamp
from run_h4_reclaim_exit_only_v22 import simulate_portfolio
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS
from run_tuning_worker_admission_v17 import TUNING


H4_RETURNS = 48
WINDOW_RETURNS = 96
QUANTILE = 0.75


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values or not 0.0 < quantile <= 1.0:
        raise ValueError("nearest-rank quantile requires values and q in (0,1]")
    ordered = sorted(values)
    return ordered[max(1, math.ceil(quantile * len(ordered))) - 1]


def h4_candidate_at(bars: list[Bar], index: int) -> dict | None:
    """Use exactly 96 completed M5 returns ending at an actual UTC H4 close."""
    if index < WINDOW_RETURNS or index + 1 >= len(bars):
        return None
    decision_close = timestamp(bars[index].time) + timedelta(minutes=5)
    if decision_close.minute != 0 or decision_close.hour % 4 != 0:
        return None
    window = bars[index - WINDOW_RETURNS:index + 1]
    if any(
        timestamp(right.time) - timestamp(left.time) != timedelta(minutes=5)
        for left, right in zip(window, window[1:])
    ):
        return None
    closes = [bar.mid_c for bar in window]
    returns = [math.log(right / left) for left, right in zip(closes, closes[1:])]
    if len(returns) != WINDOW_RETURNS:
        raise ValueError("invalid H4 coherence window")
    first = sum(returns[:H4_RETURNS])
    second = sum(returns[H4_RETURNS:])
    if first == 0.0 or second == 0.0 or first * second <= 0.0:
        return None
    energy = math.sqrt(sum(value * value for value in returns))
    if energy <= 0.0:
        return None
    direction = 1 if first > 0.0 else -1
    return {
        "direction": direction,
        "decision_time": bars[index].time,
        "decision_close_time": decision_close.isoformat().replace("+00:00", "Z"),
        "fill_time": bars[index + 1].time,
        "first_h4_log_return": first,
        "second_h4_log_return": second,
        "coherence": abs(sum(returns)) / energy,
    }


def freeze_threshold(candidates: list[dict]) -> tuple[float, int]:
    tuning = [
        float(row["coherence"])
        for row in candidates
        if TUNING[0] <= row["decision_time"][:10] < TUNING[1]
    ]
    return nearest_rank(tuning, QUANTILE), len(tuning)


def run(input_root: Path, source_ledger: Path, output_root: Path) -> dict:
    primary_raw = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    primary_rows = [
        {key: row[key] for key in ("signal_id", "pair", "fill_time", "exit_time", "direction")}
        for row in primary_raw if row["pair"] in UNIVERSE
    ]
    corpus, source_audit, candidates = {}, [], []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        bars = load_bars(matches[0])
        corpus[pair] = bars
        pair_candidates = 0
        for index in range(WINDOW_RETURNS, len(bars) - 1):
            measured = h4_candidate_at(bars, index)
            if measured is None:
                continue
            pair_candidates += 1
            candidates.append({
                "signal_id": f"H4OCE::{pair}::{measured['decision_close_time']}",
                "pair": pair,
                **measured,
            })
        source_audit.append({
            "pair": pair,
            "source_sha256": sha256_file(matches[0]),
            "bars": len(bars),
            "h4_candidates": pair_candidates,
        })
    candidates.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    threshold, tuning_candidate_count = freeze_threshold(candidates)
    auxiliary_rows = [row for row in candidates if float(row["coherence"]) >= threshold]
    periods = {
        name: {
            arm: simulate_portfolio(corpus, primary_rows, auxiliary_rows, arm, start, end)
            for arm in ARMS
        }
        for name, (start, end) in PERIODS.items()
    }
    admitted = all(
        periods[name][arm]["equity_multiple"] > 1.0
        for name in PERIODS for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "auxiliary_ledger_h4_opposing_coherence_exit_v23.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in auxiliary_rows),
        encoding="utf-8",
    )
    payload = {
        "experiment": "FX_H4_OPPOSING_COHERENCE_EXIT_V23",
        "family": "H4_CONTEXT_SHORT_TIMEFRAME_EXECUTION",
        "family_hypotheses": 3,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "completed_h4_two_block_coherence_auxiliary_exit_only",
        "source_ledger": str(source_ledger),
        "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "exit_time", "direction"],
        "source_outcome_fields_consumed": False,
        "indicator": {
            "h4_returns": H4_RETURNS,
            "window_returns": WINDOW_RETURNS,
            "quantile": QUANTILE,
            "tuning_period": list(TUNING),
            "tuning_candidate_count": tuning_candidate_count,
            "frozen_threshold": threshold,
            "walk_forward_used_for_threshold": False,
            "cost_used_for_signal": False,
            "future_outcome_used_for_signal": False,
        },
        "h4_candidates": len(candidates),
        "auxiliary_signals": len(auxiliary_rows),
        "cost_suppressed_raw_signals": 0,
        "auxiliary_ledger": str(ledger),
        "auxiliary_ledger_sha256": sha256_file(ledger),
        "auxiliary_can_open_add_reverse_resize_or_extend": False,
        "same_decision_stream_all_cost_arms": True,
        "portfolio": {"pair_count": 7, "weight_per_pair": 1 / 7, "gross_leverage_cap": 1.0},
        "periods": periods,
        "source_audit": source_audit,
        "development_admitted": admitted,
        "final_admitted": False,
        "terminal_inventory_mtm_hidden": False,
        "live_authority": False,
        "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "family correction for the H4-context family is not yet complete",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_h4_opposing_coherence_exit_v23.json"
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
        "indicator": result["indicator"],
        "h4_candidates": result["h4_candidates"],
        "auxiliary_signals": result["auxiliary_signals"],
        "periods": result["periods"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
