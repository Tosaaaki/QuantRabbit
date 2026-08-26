from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS
from run_asian_box_sweep_reclaim_v24 import (  # noqa: E402
    FIVE_MINUTES,
    expected_stamp,
    path_for_signal,
    raw_path_metrics,
    simulate_portfolio,
    summarize_raw,
    timestamp,
)
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS


DECISION_MINUTE = 5 * 60 + 55
FILL_MINUTE = 6 * 60
EXIT_MINUTE = 11 * 60 + 55
MINIMUM_ALIGNED_PAIRS = 5
USD_BASE = {pair for pair in UNIVERSE if pair.startswith("USD_")}
USD_QUOTE = set(UNIVERSE) - USD_BASE


def _validated_map(pair: str, bars: list[Bar]) -> tuple[object, dict]:
    if not bars:
        raise ValueError(f"empty day for {pair}")
    parsed = [(timestamp(bar.time), bar) for bar in bars]
    if any(bar.pair != pair for _, bar in parsed):
        raise ValueError("pair/day map contains a different pair")
    day = parsed[0][0]
    if any(stamp.date() != day.date() for stamp, _ in parsed):
        raise ValueError("pair/day map spans multiple UTC dates")
    by_stamp = {stamp: bar for stamp, bar in parsed}
    if len(by_stamp) != len(parsed):
        raise ValueError("duplicate timestamp in pair/day map")
    return day, by_stamp


def detect_day_signals(pair_day_bars: dict[str, list[Bar]]) -> list[dict]:
    """Return a coherent basket decision from the completed Asian session only."""
    if set(pair_day_bars) != set(UNIVERSE):
        return []
    maps = {}
    common_date = None
    for pair in sorted(UNIVERSE):
        try:
            day, by_stamp = _validated_map(pair, pair_day_bars[pair])
        except ValueError:
            return []
        if common_date is None:
            common_date = day
        elif day.date() != common_date.date():
            return []
        required = [expected_stamp(day, minute) for minute in range(0, EXIT_MINUTE + 1, 5)]
        if any(stamp not in by_stamp for stamp in required):
            return []
        maps[pair] = (day, by_stamp)

    oriented_returns = {}
    votes = {}
    for pair in sorted(UNIVERSE):
        day, by_stamp = maps[pair]
        first = by_stamp[expected_stamp(day, 0)]
        completed = by_stamp[expected_stamp(day, DECISION_MINUTE)]
        native_return = math.log(completed.mid_c / first.mid_o)
        usd_return = native_return if pair in USD_BASE else -native_return
        oriented_returns[pair] = (native_return, usd_return)
        votes[pair] = 1 if usd_return > 0 else -1 if usd_return < 0 else 0
    positive = sum(vote > 0 for vote in votes.values())
    negative = sum(vote < 0 for vote in votes.values())
    majority_count = max(positive, negative)
    if majority_count < MINIMUM_ALIGNED_PAIRS:
        return []
    usd_direction = 1 if positive > negative else -1

    rows = []
    for pair in sorted(UNIVERSE):
        if votes[pair] != usd_direction:
            continue
        day, by_stamp = maps[pair]
        direction = usd_direction if pair in USD_BASE else -usd_direction
        decision_stamp = expected_stamp(day, DECISION_MINUTE)
        fill_stamp = expected_stamp(day, FILL_MINUTE)
        exit_stamp = expected_stamp(day, EXIT_MINUTE)
        native_return, usd_return = oriented_returns[pair]
        rows.append({
            "signal_id": f"AUCP::{day.date().isoformat()}::{pair}::{'USD_STRONG' if usd_direction > 0 else 'USD_WEAK'}",
            "pair": pair,
            "utc_day": day.date().isoformat(),
            "decision_time": by_stamp[decision_stamp].time,
            "fill_time": by_stamp[fill_stamp].time,
            "exit_time": by_stamp[exit_stamp].time,
            "direction": direction,
            "diagnostics": {
                "native_asian_log_return": native_return,
                "usd_oriented_asian_log_return": usd_return,
                "usd_direction": usd_direction,
                "aligned_pairs": majority_count,
                "positive_usd_votes": positive,
                "negative_usd_votes": negative,
                "coherence_fraction": majority_count / len(UNIVERSE),
            },
        })
    return rows


def summarize_with_independence(rows: list[dict], start: str, end: str) -> dict:
    summary = summarize_raw(rows, start, end)
    selected = [row for row in rows if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end]
    days = {row["utc_day"] for row in selected}
    coherences = [row["diagnostics"]["coherence_fraction"] for row in selected]
    summary.update({
        "effective_bet_days": len(days),
        "tickets_per_effective_bet": len(selected) / len(days) if days else None,
        "mean_coherence_fraction": statistics.fmean(coherences) if coherences else None,
    })
    return summary


def run(input_root: Path, output_root: Path) -> dict:
    corpus = {}
    grouped: dict[str, dict[str, list[Bar]]] = {}
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        bars = load_bars(matches[0])
        corpus[pair] = bars
        grouped[pair] = defaultdict(list)
        for bar in bars:
            grouped[pair][bar.time[:10]].append(bar)
        source_audit.append({
            "pair": pair,
            "source_sha256": sha256_file(matches[0]),
            "bars": len(bars),
        })

    common_days = set.intersection(*(set(grouped[pair]) for pair in sorted(UNIVERSE)))
    rows = []
    for utc_day in sorted(common_days):
        pair_day_bars = {pair: grouped[pair][utc_day] for pair in sorted(UNIVERSE)}
        day_signals = detect_day_signals(pair_day_bars)
        for signal in day_signals:
            signal["raw_path"] = raw_path_metrics(
                path_for_signal(pair_day_bars[signal["pair"]], signal), int(signal["direction"])
            )
            rows.append(signal)
    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))

    periods = {
        name: {
            "raw_diagnostics": summarize_with_independence(rows, start, end),
            **{arm: simulate_portfolio(corpus, rows, arm, start, end) for arm in ARMS},
        }
        for name, (start, end) in PERIODS.items()
    }
    development_admitted = all(
        periods[name]["raw_diagnostics"]["signals"] >= 20
        and periods[name]["raw_diagnostics"]["effective_bet_days"] >= (10 if name == "WALK_FORWARD" else 3)
        and periods[name]["raw_diagnostics"]["mean_gross_return"] is not None
        and periods[name]["raw_diagnostics"]["mean_gross_return"] > 0.0
        and periods[name][arm]["equity_multiple"] > 1.0
        and periods[name][arm]["terminal_open_inventory"] == 0
        for name in PERIODS for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )

    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_asian_usd_coherence_persistence_v25.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "experiment": "FX_ASIAN_USD_COHERENCE_PERSISTENCE_V25",
        "family": "FX_SESSION_CURRENCY_COHERENCE",
        "family_hypotheses": 1,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "cross_pair_usd_oriented_asian_session_coherence_continuation",
        "indicator": {
            "minimum_aligned_pairs": MINIMUM_ALIGNED_PAIRS,
            "measurement_window_utc": "00:00-05:55",
            "fill_utc": "06:00",
            "fixed_exit_utc_bar": "11:55",
            "cost_used_for_signal": False,
            "future_outcome_used_for_signal": False,
        },
        "portfolio": {"pair_count": 7, "weight_per_pair": 1 / 7, "gross_leverage_cap": 1.0},
        "raw_signals": len(rows),
        "effective_bet_days": len({row["utc_day"] for row in rows}),
        "cost_suppressed_raw_signals": 0,
        "same_signal_stream_all_cost_arms": True,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": sha256_file(ledger),
        "periods": periods,
        "source_audit": source_audit,
        "development_admitted": development_admitted,
        "final_admitted": False,
        "terminal_inventory_mtm_hidden": False,
        "live_authority": False,
        "external_orders": 0,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = output_root / "result_asian_usd_coherence_persistence_v25.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.output_root)
    print(json.dumps({
        "raw_signals": result["raw_signals"],
        "effective_bet_days": result["effective_bet_days"],
        "periods": result["periods"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
