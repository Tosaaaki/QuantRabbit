from __future__ import annotations

import argparse
import hashlib
import json
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
from run_portfolio_episode_netting_v15 import PERIODS, Position, roundtrip_return


def simulate_pair(pair, bars, primary_rows, auxiliary_rows, arm, start, end):
    primary = [
        row for row in primary_rows
        if row["pair"] == pair and start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end
    ]
    auxiliary = [
        row for row in auxiliary_rows
        if row["pair"] == pair and start <= row["fill_time"][:10] < end
    ]
    primary_by_fill = {row["fill_time"]: row for row in primary}
    auxiliary_by_fill = defaultdict(list)
    for row in auxiliary:
        auxiliary_by_fill[row["fill_time"]].append(row)
    period_bars = [bar for bar in bars if start <= bar.time[:10] < end]
    if not period_bars:
        raise ValueError(f"no bars for {pair} in {start}..{end}")
    wealth, position = 1.0, None
    marks = {}
    opens = closes = ignored_same = opposite_close_only = auxiliary_exits = terminal_closes = 0
    auxiliary_flat_ignored = auxiliary_same_ignored = 0
    for bar in period_bars:
        exit_events = auxiliary_by_fill.get(bar.time, [])
        directions = {int(row["direction"]) for row in exit_events}
        if len(directions) > 1:
            raise ValueError(f"conflicting auxiliary directions for {pair} at {bar.time}")
        if exit_events:
            auxiliary_direction = next(iter(directions))
            if position is None:
                auxiliary_flat_ignored += 1
            elif position.direction == auxiliary_direction:
                auxiliary_same_ignored += 1
            else:
                wealth *= max(1.0 + roundtrip_return(position.entry_bar, bar, position.direction, arm, True), 1e-12)
                closes += 1
                auxiliary_exits += 1
                position = None
        signal = primary_by_fill.get(bar.time)
        if signal is not None:
            direction = int(signal["direction"])
            if position is None:
                position = Position(direction, bar, signal["exit_time"])
                opens += 1
            elif position.direction == direction:
                ignored_same += 1
            else:
                wealth *= max(1.0 + roundtrip_return(position.entry_bar, bar, position.direction, arm, True), 1e-12)
                closes += 1
                opposite_close_only += 1
                position = None
        if position is not None and position.expiry_time == bar.time:
            wealth *= max(1.0 + roundtrip_return(position.entry_bar, bar, position.direction, arm, False), 1e-12)
            closes += 1
            position = None
        if position is None:
            marks[bar.time] = wealth
        else:
            marks[bar.time] = wealth * max(
                1.0 + roundtrip_return(position.entry_bar, bar, position.direction, arm, False), 1e-12
            )
    if position is not None:
        last = period_bars[-1]
        wealth *= max(1.0 + roundtrip_return(position.entry_bar, last, position.direction, arm, False), 1e-12)
        closes += 1
        terminal_closes += 1
        position = None
        marks[last.time] = wealth
    return marks, {
        "source_signals": len(primary),
        "auxiliary_signals": len(auxiliary),
        "opens": opens,
        "closes": closes,
        "ignored_same_direction": ignored_same,
        "opposite_close_only": opposite_close_only,
        "auxiliary_exit_only": auxiliary_exits,
        "auxiliary_flat_ignored": auxiliary_flat_ignored,
        "auxiliary_same_direction_ignored": auxiliary_same_ignored,
        "terminal_closes": terminal_closes,
        "terminal_open_inventory": int(position is not None),
        "sleeve_equity_multiple": wealth,
    }


def simulate_portfolio(corpus, primary_rows, auxiliary_rows, arm, start, end):
    pair_marks, pair_audit = {}, {}
    for pair in sorted(UNIVERSE):
        pair_marks[pair], pair_audit[pair] = simulate_pair(
            pair, corpus[pair], primary_rows, auxiliary_rows, arm, start, end
        )
    common = set.intersection(*(set(values) for values in pair_marks.values()))
    if not common:
        raise ValueError("pair mark timelines have no common timestamps")
    equity_path = [statistics.fmean(pair_marks[pair][stamp] for pair in sorted(UNIVERSE)) for stamp in sorted(common)]
    peak, max_drawdown = equity_path[0], 0.0
    for value in equity_path:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    opens = sum(item["opens"] for item in pair_audit.values())
    closes = sum(item["closes"] for item in pair_audit.values())
    return {
        "equity_multiple": equity_path[-1],
        "max_drawdown": max_drawdown,
        "source_signals": sum(item["source_signals"] for item in pair_audit.values()),
        "auxiliary_signals": sum(item["auxiliary_signals"] for item in pair_audit.values()),
        "position_opens": opens,
        "position_closes": closes,
        "turnover_nav": (opens + closes) / len(UNIVERSE),
        "ignored_same_direction": sum(item["ignored_same_direction"] for item in pair_audit.values()),
        "opposite_close_only": sum(item["opposite_close_only"] for item in pair_audit.values()),
        "auxiliary_exit_only": sum(item["auxiliary_exit_only"] for item in pair_audit.values()),
        "auxiliary_flat_ignored": sum(item["auxiliary_flat_ignored"] for item in pair_audit.values()),
        "auxiliary_same_direction_ignored": sum(
            item["auxiliary_same_direction_ignored"] for item in pair_audit.values()
        ),
        "terminal_closes": sum(item["terminal_closes"] for item in pair_audit.values()),
        "terminal_open_inventory": sum(item["terminal_open_inventory"] for item in pair_audit.values()),
        "pair_audit": pair_audit,
    }


def run(input_root: Path, source_ledger: Path, auxiliary_ledger: Path, output_root: Path) -> dict:
    primary_raw = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    auxiliary_raw = [json.loads(line) for line in auxiliary_ledger.read_text().splitlines() if line]
    primary_rows = [
        {key: row[key] for key in ("signal_id", "pair", "fill_time", "exit_time", "direction")}
        for row in primary_raw if row["pair"] in UNIVERSE
    ]
    auxiliary_rows = [
        {key: row[key] for key in ("signal_id", "pair", "fill_time", "direction")}
        for row in auxiliary_raw if row["pair"] in UNIVERSE
    ]
    corpus, source_audit = {}, []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
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
    payload = {
        "experiment": "FX_H4_RECLAIM_EXIT_ONLY_V22",
        "family": "H4_CONTEXT_SHORT_TIMEFRAME_EXECUTION",
        "family_hypotheses": 2,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "v21_h4_reclaim_auxiliary_exit_only",
        "source_ledger": str(source_ledger),
        "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "exit_time", "direction"],
        "source_outcome_fields_consumed": False,
        "auxiliary_ledger": str(auxiliary_ledger),
        "auxiliary_ledger_sha256": sha256_file(auxiliary_ledger),
        "auxiliary_fields_consumed": ["signal_id", "pair", "fill_time", "direction"],
        "auxiliary_outcome_fields_consumed": False,
        "auxiliary_can_open_or_add": False,
        "portfolio": {"pair_count": 7, "weight_per_pair": 1 / 7, "gross_leverage_cap": 1.0},
        "periods": periods,
        "source_audit": source_audit,
        "cost_suppressed_raw_signals": 0,
        "same_decision_stream_all_cost_arms": True,
        "development_admitted": admitted,
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
    output_root.mkdir(parents=True, exist_ok=True)
    result = output_root / "result_h4_reclaim_exit_only_v22.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--auxiliary-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.source_ledger, args.auxiliary_ledger, args.output_root)
    print(json.dumps({
        "periods": result["periods"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
