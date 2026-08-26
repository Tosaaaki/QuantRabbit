from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, pip_size, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS, timestamp
from run_liquid_major_universe_v9 import UNIVERSE


PERIODS = {
    "WALK_FORWARD": ("2026-05-01", "2026-07-01"),
    "MONTH_2026_05": ("2026-05-01", "2026-06-01"),
    "MONTH_2026_06": ("2026-06-01", "2026-07-01"),
}


@dataclass
class Position:
    direction: int
    entry_bar: Bar
    expiry_time: str


def roundtrip_return(entry: Bar, exit_bar: Bar, direction: int, arm: str, exit_at_open: bool) -> float:
    exit_mid = exit_bar.mid_o if exit_at_open else exit_bar.mid_c
    if arm == "RAW_SIGNAL":
        return exit_mid / entry.mid_o - 1.0 if direction > 0 else entry.mid_o / exit_mid - 1.0
    scenario = ARMS[arm]
    slip = float(scenario["slippage"]) * pip_size(entry.pair)
    if direction > 0:
        exit_price = (exit_bar.bid_o if exit_at_open else exit_bar.bid_c) - slip
        result = exit_price / (entry.ask_o + slip) - 1.0
    else:
        exit_price = (exit_bar.ask_o if exit_at_open else exit_bar.ask_c) + slip
        result = (entry.bid_o - slip) / exit_price - 1.0
    elapsed_days = (timestamp(exit_bar.time) - timestamp(entry.time)).total_seconds() / 86400.0
    result -= 2.0 * float(scenario["commission"]) * 1e-4
    result -= float(scenario["financing"]) * 1e-4 * elapsed_days
    return result


def simulate_pair(
    pair: str, bars: list[Bar], source_rows: list[dict], arm: str, start: str, end: str,
) -> tuple[dict[str, float], dict]:
    eligible = [
        row for row in source_rows
        if row["pair"] == pair and start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end
    ]
    by_fill = {row["fill_time"]: row for row in eligible}
    period_bars = [bar for bar in bars if start <= bar.time[:10] < end]
    if not period_bars:
        raise ValueError(f"no bars for {pair} in {start}..{end}")
    wealth = 1.0
    position: Position | None = None
    marks: dict[str, float] = {}
    opens = closes = reversals = ignored_same = terminal_closes = 0
    for bar in period_bars:
        signal = by_fill.get(bar.time)
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
                reversals += 1
                position = Position(direction, bar, signal["exit_time"])
                opens += 1
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
        "source_signals": len(eligible), "opens": opens, "closes": closes,
        "reversals": reversals, "ignored_same_direction": ignored_same,
        "terminal_closes": terminal_closes, "terminal_open_inventory": int(position is not None),
        "sleeve_equity_multiple": wealth,
    }


def simulate_portfolio(corpus: dict[str, list[Bar]], rows: list[dict], arm: str, start: str, end: str) -> dict:
    pair_marks = {}
    pair_audit = {}
    for pair in sorted(UNIVERSE):
        pair_marks[pair], pair_audit[pair] = simulate_pair(pair, corpus[pair], rows, arm, start, end)
    common = set.intersection(*(set(values) for values in pair_marks.values()))
    if not common:
        raise ValueError("pair mark timelines have no common timestamps")
    equity_path = [statistics.fmean(pair_marks[pair][stamp] for pair in sorted(UNIVERSE)) for stamp in sorted(common)]
    peak = equity_path[0]
    max_drawdown = 0.0
    for value in equity_path:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    opens = sum(item["opens"] for item in pair_audit.values())
    closes = sum(item["closes"] for item in pair_audit.values())
    return {
        "equity_multiple": equity_path[-1], "max_drawdown": max_drawdown,
        "source_signals": sum(item["source_signals"] for item in pair_audit.values()),
        "position_opens": opens, "position_closes": closes,
        "turnover_nav": (opens + closes) / len(UNIVERSE),
        "reversals": sum(item["reversals"] for item in pair_audit.values()),
        "ignored_same_direction": sum(item["ignored_same_direction"] for item in pair_audit.values()),
        "terminal_closes": sum(item["terminal_closes"] for item in pair_audit.values()),
        "terminal_open_inventory": sum(item["terminal_open_inventory"] for item in pair_audit.values()),
        "pair_audit": pair_audit,
    }


def run(input_root: Path, source_ledger: Path, output_root: Path) -> dict:
    raw_source = [json.loads(line) for line in source_ledger.read_text().splitlines() if line]
    rows = [{key: row[key] for key in ("signal_id", "pair", "fill_time", "exit_time", "direction")} for row in raw_source]
    corpus = {}
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        corpus[pair] = load_bars(matches[0])
        source_audit.append({"pair": pair, "source_sha256": sha256_file(matches[0]), "bars": len(corpus[pair])})
    periods = {
        name: {arm: simulate_portfolio(corpus, rows, arm, start, end) for arm in ARMS}
        for name, (start, end) in PERIODS.items()
    }
    development_admitted = all(
        periods[name][arm]["equity_multiple"] > 1.0
        for name in PERIODS for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )
    payload = {
        "experiment": "FX_AUCTION_TRAP_H384_EPISODE_NETTING_V15",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "same_pair_episode_netting",
        "source_ledger": str(source_ledger), "source_ledger_sha256": sha256_file(source_ledger),
        "source_fields_consumed": ["signal_id", "pair", "fill_time", "exit_time", "direction"],
        "source_outcome_fields_consumed": False,
        "portfolio": {"pair_count": 7, "weight_per_pair": 1 / 7, "gross_leverage_cap": 1.0},
        "periods": periods, "source_audit": source_audit,
        "cost_suppressed_raw_signals": 0, "same_signal_stream_all_cost_arms": True,
        "development_admitted": development_admitted, "final_admitted": False,
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
    output_root.mkdir(parents=True, exist_ok=True)
    result = output_root / "result_portfolio_episode_netting_v15.json"
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
        "periods": result["periods"], "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
