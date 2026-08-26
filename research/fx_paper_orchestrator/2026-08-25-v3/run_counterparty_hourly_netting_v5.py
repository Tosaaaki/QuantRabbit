from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, pip_size, sha256_file  # noqa: E402


SOURCE_CANDIDATE = "FX_CRS_MULTINOMIAL_RESPONSE_H12_V4"
ARMS = {
    "RAW_SIGNAL": {"slippage": 0.0, "commission": 0.0, "financing": 0.0},
    "EXECUTABLE_BASE": {"slippage": 0.3, "commission": 0.0, "financing": 0.5},
    "ADVERSE_STRESS": {"slippage": 0.9, "commission": 0.2, "financing": 1.5},
}
PERIODS = {
    "TUNING": ("2026-03-01", "2026-05-01"),
    "WALK_FORWARD": ("2026-05-01", "2026-07-01"),
    "OPENED_DIAGNOSTIC": ("2026-07-01", "2026-08-01"),
}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def next_hour(value: str) -> datetime:
    stamp = parse_time(value)
    return stamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def build_targets(decisions: list[dict]) -> tuple[dict[datetime, dict[str, int]], dict]:
    votes: dict[tuple[datetime, str], int] = defaultdict(int)
    source_count: dict[tuple[datetime, str], int] = defaultdict(int)
    for row in decisions:
        if row.get("candidate_id") != SOURCE_CANDIDATE or not row.get("expected_order"):
            continue
        checkpoint = next_hour(row["fill_time"])
        key = (checkpoint, row["pair"])
        votes[key] += 1 if int(row["direction"]) > 0 else -1
        source_count[key] += 1
    targets: dict[datetime, dict[str, int]] = defaultdict(dict)
    ties = 0
    for (checkpoint, pair), value in votes.items():
        target = 1 if value > 0 else -1 if value < 0 else 0
        ties += target == 0
        targets[checkpoint][pair] = target
    return dict(targets), {
        "eligible_source_orders": sum(source_count.values()),
        "pair_hour_targets": len(votes),
        "tie_flat_targets": ties,
    }


def load_corpus(input_root: Path) -> tuple[dict[str, list[Bar]], dict[str, list[datetime]], list[dict]]:
    files = sorted(input_root.glob("*/*_M5_BA_*.jsonl.gz"))
    if len(files) != 28:
        raise ValueError(f"expected exact 28-pair corpus, got {len(files)}")
    corpus, time_index, audit = {}, {}, []
    for path in files:
        bars = load_bars(path)
        corpus[bars[0].pair] = bars
        time_index[bars[0].pair] = [parse_time(bar.time) for bar in bars]
        audit.append({"pair": bars[0].pair, "rows": len(bars), "sha256": sha256_file(path)})
    return corpus, time_index, audit


def bar_at_or_after(bars: list[Bar], stamps: list[datetime], checkpoint: datetime) -> Bar | None:
    position = bisect.bisect_left(stamps, checkpoint)
    return bars[position] if position < len(bars) else None


def simulate(
    corpus: dict[str, list[Bar]], time_index: dict[str, list[datetime]],
    targets: dict[datetime, dict[str, int]],
    arm: str, start: str, end: str, persistence_hours: int = 0,
) -> dict:
    scenario = ARMS[arm]
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    checkpoints = sorted(stamp for stamp in targets if start_dt <= stamp < end_dt)
    if len(checkpoints) < 2:
        return {"checkpoints": len(checkpoints), "equity_multiple": None}
    pairs = sorted(corpus)
    weight = 1.0 / len(pairs)
    positions = {pair: 0 for pair in pairs}
    ages = {pair: 0.0 for pair in pairs}
    equity = peak = 1.0
    max_drawdown = 0.0
    turnover = cost_drag = gross_pnl = 0.0
    target_changes = 0
    interval_returns = []
    monthly_start: dict[str, float] = {}
    monthly_end: dict[str, float] = {}
    last_marks: dict[str, Bar] = {}

    for index, checkpoint in enumerate(checkpoints):
        current_marks = {
            pair: bar_at_or_after(corpus[pair], time_index[pair], checkpoint)
            for pair in pairs
        }
        current_marks = {pair: bar for pair, bar in current_marks.items() if bar is not None}
        interval_pnl = 0.0
        if last_marks:
            elapsed_days = (checkpoint - checkpoints[index - 1]).total_seconds() / 86400.0
            for pair, old in positions.items():
                if old == 0 or pair not in last_marks or pair not in current_marks:
                    continue
                previous_mid = last_marks[pair].mid_o
                current_mid = current_marks[pair].mid_o
                # Use the same exact ratio convention as the proposal replay:
                # long wealth follows P_t / P_0, while short wealth follows
                # P_0 / P_t.  The former ``old * (P_t/P_0 - 1)`` shortcut was
                # only a linear approximation for shorts.
                move = (
                    current_mid / previous_mid - 1.0
                    if old > 0
                    else previous_mid / current_mid - 1.0
                )
                pnl = abs(old) * move * weight
                carry = abs(old) * weight * scenario["financing"] * 1e-4 * elapsed_days
                interval_pnl += pnl - carry
                gross_pnl += pnl
                cost_drag += carry

        elapsed_hours = (
            (checkpoint - checkpoints[index - 1]).total_seconds() / 3600.0
            if index else 0.0
        )
        new_targets = {}
        for pair in pairs:
            if pair in targets[checkpoint]:
                new_targets[pair] = int(targets[checkpoint][pair])
                ages[pair] = 0.0
            elif positions[pair] != 0 and persistence_hours > 0 and ages[pair] + elapsed_hours < persistence_hours:
                new_targets[pair] = positions[pair]
                ages[pair] += elapsed_hours
            else:
                new_targets[pair] = 0
                ages[pair] = 0.0
        for pair in pairs:
            old, new = positions[pair], new_targets[pair]
            delta = new - old
            if delta == 0 or pair not in current_marks:
                continue
            target_changes += 1
            turnover += abs(delta) * weight
            mark = current_marks[pair]
            if arm == "RAW_SIGNAL":
                trade_cost = commission = 0.0
            else:
                slip = scenario["slippage"] * pip_size(pair)
                mid = mark.mid_o
                if delta > 0:
                    trade_cost = (mark.ask_o + slip - mid) / mid * abs(delta) * weight
                else:
                    trade_cost = (mid - (mark.bid_o - slip)) / mid * abs(delta) * weight
                commission = scenario["commission"] * 1e-4 * abs(delta) * weight
            interval_pnl -= trade_cost + commission
            cost_drag += trade_cost + commission
            positions[pair] = new

        equity *= max(1.0 + interval_pnl, 1e-12)
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        interval_returns.append(interval_pnl)
        month = checkpoint.strftime("%Y-%m")
        monthly_start.setdefault(month, equity / max(1.0 + interval_pnl, 1e-12))
        monthly_end[month] = equity
        last_marks = current_marks

    terminal_cost = 0.0
    for pair, old in positions.items():
        if old == 0 or pair not in last_marks:
            continue
        mark = last_marks[pair]
        if arm == "RAW_SIGNAL":
            close_cost = 0.0
        else:
            slip = scenario["slippage"] * pip_size(pair)
            mid = mark.mid_o
            if old > 0:
                close_cost = (mid - (mark.bid_o - slip)) / mid * abs(old) * weight
            else:
                close_cost = (mark.ask_o + slip - mid) / mid * abs(old) * weight
            close_cost += scenario["commission"] * 1e-4 * abs(old) * weight
        terminal_cost += close_cost
        turnover += abs(old) * weight
        positions[pair] = 0
    equity *= max(1.0 - terminal_cost, 1e-12)
    cost_drag += terminal_cost
    if checkpoints:
        monthly_end[checkpoints[-1].strftime("%Y-%m")] = equity
    return {
        "checkpoints": len(checkpoints),
        "target_changes": target_changes,
        "turnover_nav": turnover,
        "gross_pnl_nav_additive": gross_pnl,
        "cost_drag_nav_additive": cost_drag,
        "terminal_liquidation_cost_nav": terminal_cost,
        "terminal_open_inventory": sum(abs(value) for value in positions.values()),
        "persistence_hours": persistence_hours,
        "equity_multiple": equity,
        "max_drawdown": max_drawdown,
        "mean_interval_return": sum(interval_returns) / len(interval_returns),
        "monthly_multiples": {
            month: monthly_end[month] / monthly_start[month]
            for month in sorted(monthly_start) if month in monthly_end
        },
    }


def run(input_root: Path, decision_ledger: Path, output_root: Path) -> dict:
    decisions = [json.loads(line) for line in decision_ledger.read_text().splitlines() if line]
    targets, target_audit = build_targets(decisions)
    corpus, time_index, source_audit = load_corpus(input_root)
    periods = {
        period_name: {
            arm: simulate(corpus, time_index, targets, arm, start, end)
            for arm in ARMS
        }
        for period_name, (start, end) in PERIODS.items()
    }
    walk = periods["WALK_FORWARD"]
    development_admitted = all(
        walk[arm].get("equity_multiple") is not None and walk[arm]["equity_multiple"] > 1.0
        for arm in ARMS
    )
    payload = {
        "experiment": "FX_CRS_H12_HOURLY_INTERNAL_NETTING_V5",
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "hourly_same_pair_internal_netting",
        "source_candidate": SOURCE_CANDIDATE,
        "source_decision_ledger": str(decision_ledger),
        "source_decision_ledger_sha256": sha256_file(decision_ledger),
        "target_audit": target_audit,
        "portfolio": {
            "pair_count": len(corpus),
            "weight_per_pair": 1.0 / len(corpus),
            "gross_leverage_cap": 1.0,
            "target_values": [-1, 0, 1],
        },
        "periods": periods,
        "development_admitted": development_admitted,
        "final_admitted": False,
        "source_audit": source_audit,
        "cost_suppressed_source_signals": 0,
        "terminal_inventory_mtm_hidden": False,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout is unavailable",
            "monthly 2.0x normal/adverse acceptance has not been demonstrated",
        ],
        "live_authority": False,
        "external_orders": 0,
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "result_hourly_netting_v5.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.decision_ledger, args.output_root)
    print(json.dumps({
        "target_audit": result["target_audit"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "development_admitted": result["development_admitted"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
