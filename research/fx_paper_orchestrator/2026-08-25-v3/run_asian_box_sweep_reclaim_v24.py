from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, pip_size, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS
from run_liquid_major_universe_v9 import UNIVERSE
from run_portfolio_episode_netting_v15 import PERIODS


BOX_START_MINUTE = 0
BOX_END_MINUTE = 5 * 60 + 55
OBSERVATION_START_MINUTE = 6 * 60
OBSERVATION_END_MINUTE = 11 * 60 + 55
EXIT_MINUTE = 15 * 60 + 55
FIVE_MINUTES = timedelta(minutes=5)


def timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        body = value[:-1]
        if "." in body:
            body, fraction = body.split(".", 1)
            if not fraction or any(character != "0" for character in fraction):
                raise ValueError(f"timestamp is not on an exact bar boundary: {value}")
        parsed = datetime.fromisoformat(body).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"timestamp is not UTC: {value}")
    return parsed.astimezone(timezone.utc)


def expected_stamp(day: datetime, minute: int) -> datetime:
    return day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minute)


def detect_day_signal(pair: str, day_bars: list[Bar]) -> dict | None:
    """Detect the first causal reclaim using only completed bars through decision time.

    Price values in the following fill bar and the fixed exit bar never participate in
    the decision. Missing or duplicate timestamps fail closed.
    """
    if not day_bars:
        return None
    parsed = [(timestamp(bar.time), bar) for bar in day_bars]
    if any(bar.pair != pair for _, bar in parsed):
        raise ValueError("day contains a different pair")
    day = parsed[0][0]
    if any(stamp.date() != day.date() for stamp, _ in parsed):
        raise ValueError("day fixture spans multiple UTC dates")
    by_stamp = {stamp: bar for stamp, bar in parsed}
    if len(by_stamp) != len(parsed):
        return None

    box_stamps = [expected_stamp(day, minute) for minute in range(BOX_START_MINUTE, BOX_END_MINUTE + 1, 5)]
    if any(stamp not in by_stamp for stamp in box_stamps):
        return None
    box = [by_stamp[stamp] for stamp in box_stamps]
    if len(box) != 72:
        return None
    box_high = max(bar.mid_h for bar in box)
    box_low = min(bar.mid_l for bar in box)
    exit_stamp = expected_stamp(day, EXIT_MINUTE)
    if exit_stamp not in by_stamp:
        return None

    ambiguous_bars = 0
    for minute in range(OBSERVATION_START_MINUTE, OBSERVATION_END_MINUTE + 1, 5):
        decision_stamp = expected_stamp(day, minute)
        decision = by_stamp.get(decision_stamp)
        if decision is None:
            continue
        swept_low = decision.mid_l < box_low
        swept_high = decision.mid_h > box_high
        if swept_low and swept_high:
            ambiguous_bars += 1
            continue
        direction = 0
        if swept_low and decision.mid_c > box_low:
            direction = 1
        elif swept_high and decision.mid_c < box_high:
            direction = -1
        if direction == 0:
            continue
        fill_stamp = decision_stamp + FIVE_MINUTES
        if fill_stamp not in by_stamp or fill_stamp >= exit_stamp:
            return None
        required_path = []
        cursor = fill_stamp
        while cursor <= exit_stamp:
            if cursor not in by_stamp:
                return None
            required_path.append(by_stamp[cursor])
            cursor += FIVE_MINUTES
        return {
            "signal_id": f"ABSR::{pair}::{day.date().isoformat()}::{'LONG' if direction > 0 else 'SHORT'}",
            "pair": pair,
            "utc_day": day.date().isoformat(),
            "decision_time": decision.time,
            "fill_time": by_stamp[fill_stamp].time,
            "exit_time": by_stamp[exit_stamp].time,
            "direction": direction,
            "diagnostics": {
                "box_high": box_high,
                "box_low": box_low,
                "decision_high": decision.mid_h,
                "decision_low": decision.mid_l,
                "decision_close": decision.mid_c,
                "ambiguous_bars_preceding_signal": ambiguous_bars,
                "box_completed_bars": len(box),
                "path_completed_bars": len(required_path),
            },
        }
    return None


def raw_path_metrics(path: list[Bar], direction: int) -> dict:
    if not path:
        raise ValueError("raw path is empty")
    entry, exit_bar = path[0], path[-1]
    if direction > 0:
        gross = exit_bar.mid_c / entry.mid_o - 1.0
        mfe = max(bar.mid_h / entry.mid_o - 1.0 for bar in path)
        mae = min(bar.mid_l / entry.mid_o - 1.0 for bar in path)
    elif direction < 0:
        gross = entry.mid_o / exit_bar.mid_c - 1.0
        mfe = max(entry.mid_o / bar.mid_l - 1.0 for bar in path)
        mae = min(entry.mid_o / bar.mid_h - 1.0 for bar in path)
    else:
        raise ValueError("direction must be nonzero")
    return {"gross_return": gross, "mfe_return": mfe, "mae_return": mae}


def path_for_signal(day_bars: list[Bar], signal: dict) -> list[Bar]:
    start = timestamp(signal["fill_time"])
    end = timestamp(signal["exit_time"])
    by_stamp = {timestamp(bar.time): bar for bar in day_bars}
    path = []
    cursor = start
    while cursor <= end:
        if cursor not in by_stamp:
            raise ValueError("accepted signal path became incomplete")
        path.append(by_stamp[cursor])
        cursor += FIVE_MINUTES
    return path


def summarize_raw(rows: list[dict], start: str, end: str) -> dict:
    selected = [row for row in rows if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end]
    gross = [row["raw_path"]["gross_return"] for row in selected]
    mfe = [row["raw_path"]["mfe_return"] for row in selected]
    mae = [row["raw_path"]["mae_return"] for row in selected]
    return {
        "signals": len(selected),
        "mean_gross_return": statistics.fmean(gross) if gross else None,
        "median_gross_return": statistics.median(gross) if gross else None,
        "direction_accuracy": sum(value > 0 for value in gross) / len(gross) if gross else None,
        "mean_mfe_return": statistics.fmean(mfe) if mfe else None,
        "mean_mae_return": statistics.fmean(mae) if mae else None,
        "break_even_roundtrip_cost": statistics.fmean(gross) if gross else None,
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
        "source_signals": len(eligible),
        "opens": opens,
        "closes": closes,
        "reversals": reversals,
        "ignored_same_direction": ignored_same,
        "terminal_closes": terminal_closes,
        "terminal_open_inventory": int(position is not None),
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
        "equity_multiple": equity_path[-1],
        "max_drawdown": max_drawdown,
        "source_signals": sum(item["source_signals"] for item in pair_audit.values()),
        "position_opens": opens,
        "position_closes": closes,
        "turnover_nav": (opens + closes) / len(UNIVERSE),
        "reversals": sum(item["reversals"] for item in pair_audit.values()),
        "ignored_same_direction": sum(item["ignored_same_direction"] for item in pair_audit.values()),
        "terminal_closes": sum(item["terminal_closes"] for item in pair_audit.values()),
        "terminal_open_inventory": sum(item["terminal_open_inventory"] for item in pair_audit.values()),
        "pair_audit": pair_audit,
    }


def run(input_root: Path, output_root: Path) -> dict:
    corpus: dict[str, list[Bar]] = {}
    rows: list[dict] = []
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        bars = load_bars(matches[0])
        corpus[pair] = bars
        by_day: dict[str, list[Bar]] = defaultdict(list)
        for bar in bars:
            by_day[bar.time[:10]].append(bar)
        pair_signals = 0
        for utc_day in sorted(by_day):
            signal = detect_day_signal(pair, by_day[utc_day])
            if signal is None:
                continue
            signal["raw_path"] = raw_path_metrics(path_for_signal(by_day[utc_day], signal), int(signal["direction"]))
            rows.append(signal)
            pair_signals += 1
        source_audit.append({
            "pair": pair,
            "source_sha256": sha256_file(matches[0]),
            "bars": len(bars),
            "signals": pair_signals,
        })

    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    periods = {
        name: {
            "raw_diagnostics": summarize_raw(rows, start, end),
            **{arm: simulate_portfolio(corpus, rows, arm, start, end) for arm in ARMS},
        }
        for name, (start, end) in PERIODS.items()
    }
    development_admitted = all(
        periods[name]["raw_diagnostics"]["signals"] >= 20
        and periods[name]["raw_diagnostics"]["mean_gross_return"] is not None
        and periods[name]["raw_diagnostics"]["mean_gross_return"] > 0.0
        and periods[name][arm]["equity_multiple"] > 1.0
        and periods[name][arm]["terminal_open_inventory"] == 0
        for name in PERIODS for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
    )

    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_asian_box_sweep_reclaim_v24.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "experiment": "FX_ASIAN_BOX_SWEEP_RECLAIM_V24",
        "family": "FX_SESSION_AUCTION_GEOMETRY",
        "family_hypotheses": 1,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "fixed_utc_asian_box_first_sweep_reclaim_entry",
        "indicator": {
            "box_completed_m5_bars": 72,
            "box_window_utc": "00:00-05:55",
            "observation_window_utc": "06:00-11:55",
            "fixed_exit_utc_bar": "15:55",
            "cost_used_for_signal": False,
            "future_outcome_used_for_signal": False,
            "maximum_signals_per_pair_utc_day": 1,
        },
        "portfolio": {"pair_count": 7, "weight_per_pair": 1 / 7, "gross_leverage_cap": 1.0},
        "raw_signals": len(rows),
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
    result = output_root / "result_asian_box_sweep_reclaim_v24.json"
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
        "periods": result["periods"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
