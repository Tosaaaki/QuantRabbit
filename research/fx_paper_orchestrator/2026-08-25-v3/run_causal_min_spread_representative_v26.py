from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


V2_DIR = Path(__file__).resolve().parents[1] / "2026-08-25-v2"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from fx_original_indicators import Bar, load_bars, sha256_file  # noqa: E402
from run_auction_trap_geometry_v7 import ARMS  # noqa: E402
from run_liquid_major_universe_v9 import UNIVERSE  # noqa: E402
from run_portfolio_episode_netting_v15 import PERIODS, roundtrip_return, simulate_portfolio  # noqa: E402


PARENT_RESULT_SHA256 = "eb646f6933d1078e86a025fa28b941fa2ddfd20f1a885a61ced7bbc23bfdef45"
PARENT_LEDGER_SHA256 = "4657f39f444529f8b267d2295485cf8ea66c8506fefdf50bdf1456a1acafc6db"
PARENT_SIGNAL_ID_SET_SHA256 = "4100dd95a74526fddee1a495a8a1bbe0d7568a6a5f5147cb048509a989f23f8e"
PARENT_RAW_SIGNALS = 500
LOOKBACK_BARS = 20
INITIAL_EQUITY_JPY = 200000
AUTHORITY = {
    "paper_only": True,
    "live_authority": False,
    "broker_account_access": False,
    "credential_access": False,
    "order_endpoint": False,
    "external_orders": 0,
    "deploy": False,
    "external_config_mutation": False,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def embedded_hash(payload: dict, field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def signal_id_set_hash(rows: list[dict]) -> str:
    return hashlib.sha256(canonical_bytes(sorted(row["signal_id"] for row in rows))).hexdigest()


def load_parent(parent_result_path: Path, parent_ledger_path: Path) -> tuple[dict, list[dict]]:
    if sha256_file(parent_result_path) != PARENT_RESULT_SHA256:
        raise ValueError("sealed V25 parent result hash mismatch")
    if sha256_file(parent_ledger_path) != PARENT_LEDGER_SHA256:
        raise ValueError("sealed V25 parent ledger hash mismatch")
    parent = json.loads(parent_result_path.read_text(encoding="utf-8"))
    if parent.get("result_sha256") != embedded_hash(parent, "result_sha256"):
        raise ValueError("sealed V25 parent embedded result hash mismatch")
    rows = [json.loads(line) for line in parent_ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != PARENT_RAW_SIGNALS or signal_id_set_hash(rows) != PARENT_SIGNAL_ID_SET_SHA256:
        raise ValueError("sealed V25 parent signal identity mismatch")
    if len({row["signal_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate V25 parent signal id")
    if rows != sorted(rows, key=lambda row: (row["fill_time"], row["signal_id"])):
        raise ValueError("V25 parent ledger is not deterministically ordered")
    return parent, rows


def load_corpus(input_root: Path) -> tuple[dict[str, list[Bar]], list[dict]]:
    corpus = {}
    source_audit = []
    for pair in sorted(UNIVERSE):
        matches = sorted((input_root / pair).glob("*_M5_BA_*.jsonl.gz"))
        if len(matches) != 1:
            raise ValueError(f"expected one source file for {pair}, got {len(matches)}")
        bars = load_bars(matches[0])
        if any(left.time >= right.time for left, right in zip(bars, bars[1:])):
            raise ValueError(f"non-increasing source chronology for {pair}")
        corpus[pair] = bars
        source_audit.append({
            "pair": pair,
            "source_sha256": sha256_file(matches[0]),
            "bars": len(bars),
        })
    return corpus, source_audit


def causal_score(row: dict, bars: list[Bar], time_index: dict[str, int]) -> float:
    index = time_index.get(row["decision_time"])
    if index is None or index < LOOKBACK_BARS - 1:
        raise ValueError(f"missing causal cost lookback for {row['signal_id']}")
    window = bars[index - LOOKBACK_BARS + 1:index + 1]
    if len(window) != LOOKBACK_BARS or any(bar.time > row["decision_time"] for bar in window):
        raise ValueError(f"cost score used a noncausal row for {row['signal_id']}")
    values = [
        2.0 * 10000.0 * (bar.ask_c - bar.bid_c) / ((bar.ask_c + bar.bid_c) / 2.0)
        for bar in window
    ]
    if any(value < 0 for value in values):
        raise ValueError(f"crossed source price for {row['signal_id']}")
    return statistics.median(values)


def apply_rule(parent_rows: list[dict], corpus: dict[str, list[Bar]]) -> list[dict]:
    time_indexes = {pair: {bar.time: index for index, bar in enumerate(bars)} for pair, bars in corpus.items()}
    rows = []
    for parent_row in parent_rows:
        row = json.loads(json.dumps(parent_row, sort_keys=True, allow_nan=False))
        row["causal_roundtrip_spread_proxy_bps"] = causal_score(
            row, corpus[row["pair"]], time_indexes[row["pair"]]
        )
        rows.append(row)
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_day[row["utc_day"]].append(row)
    for day_rows in by_day.values():
        ordered = sorted(day_rows, key=lambda row: (
            row["causal_roundtrip_spread_proxy_bps"], row["signal_id"]
        ))
        selected_id = ordered[0]["signal_id"]
        for rank, row in enumerate(ordered, 1):
            selected = row["signal_id"] == selected_id
            row["execution_rank"] = rank
            row["execution_selected"] = selected
            row["execution_action"] = "EXECUTE_FIXED_ONE_SEVENTH_SLEEVE" if selected else "CASH_NO_POSITION"
            row["arm_actions"] = {arm: row["execution_action"] for arm in ARMS}
    rows.sort(key=lambda row: (row["fill_time"], row["signal_id"]))
    return rows


def period_rows(rows: list[dict], start: str, end: str) -> list[dict]:
    return [row for row in rows if start <= row["fill_time"][:10] < end and row["exit_time"][:10] < end]


def trade_returns(corpus: dict[str, list[Bar]], rows: list[dict], arm: str) -> list[float]:
    indexes = {pair: {bar.time: bar for bar in bars} for pair, bars in corpus.items()}
    values = []
    for row in rows:
        pair_bars = indexes[row["pair"]]
        entry = pair_bars.get(row["fill_time"])
        exit_bar = pair_bars.get(row["exit_time"])
        if entry is None or exit_bar is None:
            raise ValueError(f"missing fill/exit bar for {row['signal_id']}")
        values.append(roundtrip_return(entry, exit_bar, int(row["direction"]), arm, False))
    return values


def arm_metrics(corpus: dict[str, list[Bar]], all_rows: list[dict], executed_rows: list[dict],
                arm: str, start: str, end: str) -> dict:
    simulated = simulate_portfolio(corpus, executed_rows, arm, start, end)
    selected = period_rows(executed_rows, start, end)
    gross = trade_returns(corpus, selected, "RAW_SIGNAL")
    net = trade_returns(corpus, selected, arm)
    ages = [(parse_time(row["exit_time"]) - parse_time(row["fill_time"])).total_seconds() for row in selected]
    raw_count = len(period_rows(all_rows, start, end))
    simulated.update({
        "source_signals": raw_count,
        "executed_signals": len(selected),
        "cash_signals": raw_count - len(selected),
        "gross_edge_bps": statistics.fmean(gross) * 10000.0 if gross else None,
        "realized_cost_bps": statistics.fmean(g - n for g, n in zip(gross, net)) * 10000.0 if net else None,
        "net_edge_bps": statistics.fmean(net) * 10000.0 if net else None,
        "break_even_cost_bps": statistics.fmean(gross) * 10000.0 if gross else None,
        "direction_accuracy": sum(value > 0 for value in gross) / len(gross) if gross else None,
        "N_eff_days": len({row["utc_day"] for row in selected}),
        "max_inventory_age_seconds": max(ages) if ages else 0,
        "terminal_inventory_mtm": 0.0,
        "initial_equity_jpy": INITIAL_EQUITY_JPY,
        "ending_equity_jpy": INITIAL_EQUITY_JPY * simulated["equity_multiple"],
    })
    return simulated


def period_payload(corpus: dict[str, list[Bar]], all_rows: list[dict], selected_rows: list[dict],
                   start: str, end: str) -> dict:
    raw = period_rows(all_rows, start, end)
    selected = period_rows(selected_rows, start, end)
    raw_gross = trade_returns(corpus, raw, "RAW_SIGNAL")
    return {
        "raw_diagnostics": {
            "signals": len(raw),
            "execution_selected_signals": len(selected),
            "cash_signals": len(raw) - len(selected),
            "effective_bet_days": len({row["utc_day"] for row in raw}),
            "N_eff_days": len({row["utc_day"] for row in selected}),
            "all_raw_gross_edge_bps": statistics.fmean(raw_gross) * 10000.0 if raw_gross else None,
            "raw_definition_changed": False,
        },
        **{
            arm: arm_metrics(corpus, all_rows, selected_rows, arm, start, end)
            for arm in ARMS
        },
    }


def comparison_v25(corpus: dict[str, list[Bar]], all_rows: list[dict], v26_periods: dict) -> dict:
    comparison = {}
    for period_name, (start, end) in PERIODS.items():
        comparison[period_name] = {}
        for arm in ARMS:
            v25 = arm_metrics(corpus, all_rows, all_rows, arm, start, end)
            v26 = v26_periods[period_name][arm]
            keys = [
                "gross_edge_bps", "realized_cost_bps", "net_edge_bps", "turnover_nav",
                "break_even_cost_bps", "direction_accuracy", "equity_multiple", "max_drawdown",
                "terminal_inventory_mtm", "max_inventory_age_seconds", "N_eff_days",
            ]
            comparison[period_name][arm] = {
                key: {"V25": v25[key], "V26": v26[key], "delta": v26[key] - v25[key]}
                for key in keys
            }
    return comparison


def rejection(periods: dict) -> dict:
    months = ["MONTH_2026_05", "MONTH_2026_06"]
    normal_pass = all(periods[month]["EXECUTABLE_BASE"]["equity_multiple"] >= 2.0 for month in months)
    adverse_pass = all(periods[month]["ADVERSE_STRESS"]["equity_multiple"] >= 2.0 for month in months)
    raw = periods["WALK_FORWARD"]["RAW_SIGNAL"]["equity_multiple"]
    base = periods["WALK_FORWARD"]["EXECUTABLE_BASE"]["equity_multiple"]
    adverse = periods["WALK_FORWARD"]["ADVERSE_STRESS"]["equity_multiple"]
    if raw <= 1.0:
        reason = "EXECUTION_SUBSET_RAW_EDGE_ABSENT"
    elif base <= 1.0:
        reason = "RAW_EDGE_COST_DOMINANT"
    elif adverse <= 1.0:
        reason = "ADVERSE_COST_FRAGILE"
    else:
        reason = "MONTHLY_2X_AND_UNOPENED_HOLDOUT_NOT_MET"
    return {
        "rejected": not (normal_pass and adverse_pass),
        "reason_code": reason,
        "normal_full_month_2x_pass": normal_pass,
        "adverse_full_month_2x_pass": adverse_pass,
        "holdout_reproduced": False,
        "adoption_authorized": False,
    }


def run(input_root: Path, parent_ledger: Path, parent_result: Path, output_root: Path) -> dict:
    parent, parent_rows = load_parent(parent_result, parent_ledger)
    corpus, source_audit = load_corpus(input_root)
    rows = apply_rule(parent_rows, corpus)
    selected_rows = [row for row in rows if row["execution_selected"]]
    if len(selected_rows) != len({row["utc_day"] for row in rows}):
        raise ValueError("rule did not select exactly one signal per basket")
    if any(len(set(row["arm_actions"].values())) != 1 for row in rows):
        raise ValueError("cost arms do not share the exact execution mask")
    periods = {
        name: period_payload(corpus, rows, selected_rows, start, end)
        for name, (start, end) in PERIODS.items()
    }
    mask = [[row["signal_id"], row["execution_selected"]] for row in rows]
    output_root.mkdir(parents=True, exist_ok=True)
    ledger = output_root / "proposal_ledger_causal_min_spread_representative_v26.jsonl"
    ledger.write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    ), encoding="utf-8")
    payload = {
        "experiment": "FX_CAUSAL_MIN_SPREAD_REPRESENTATIVE_V26",
        "family": "FX_SESSION_CURRENCY_COHERENCE",
        "family_hypotheses": 2,
        "evidence_class": "opened_development_not_future_holdout",
        "single_changed_variable": "one_causal_minimum_spread_representative_execution_per_raw_signal_basket",
        "parent_cycle": "V25",
        "parent_result_sha256": sha256_file(parent_result),
        "parent_ledger_sha256": sha256_file(parent_ledger),
        "parent_signal_id_set_sha256": signal_id_set_hash(parent_rows),
        "raw_signal_definition": parent["indicator"],
        "raw_signals": len(rows),
        "effective_bet_days": len({row["utc_day"] for row in rows}),
        "execution_selected_signals": len(selected_rows),
        "execution_cash_signals": len(rows) - len(selected_rows),
        "selected_pair_counts": dict(sorted(Counter(row["pair"] for row in selected_rows).items())),
        "execution_rule": {
            "name": "CAUSAL_MIN_SPREAD_REPRESENTATIVE",
            "lookback_bars": LOOKBACK_BARS,
            "lookback_minutes": LOOKBACK_BARS * 5,
            "score_units": "roundtrip_quoted_spread_proxy_bps",
            "selected_per_basket": 1,
            "tie_break": "signal_id",
            "unallocated_sleeves": "CASH",
        },
        "execution_mask_sha256": hashlib.sha256(canonical_bytes(mask)).hexdigest(),
        "same_execution_mask_all_cost_arms": True,
        "same_signal_stream_all_cost_arms": True,
        "same_parent_signal_id_set": signal_id_set_hash(rows) == PARENT_SIGNAL_ID_SET_SHA256,
        "same_parent_decision_timestamps": all(
            left["decision_time"] == right["decision_time"]
            for left, right in zip(parent_rows, rows)
        ),
        "cost_suppressed_raw_signals": 0,
        "proposal_ledger": str(ledger),
        "proposal_ledger_sha256": sha256_file(ledger),
        "portfolio": {
            "pair_count": 7,
            "weight_per_pair": 1 / 7,
            "gross_leverage_cap": 1.0,
            "rule_max_gross_leverage": 1 / 7,
            "unallocated_sleeves_reallocated": False,
        },
        "periods": periods,
        "metric_comparison_vs_v25": comparison_v25(corpus, rows, periods),
        "source_audit": source_audit,
        "automatic_rejection": rejection(periods),
        "development_admitted": False,
        "final_admitted": False,
        "terminal_inventory_mtm_hidden": False,
        "holdout": {"label": "FUTURE_FX_HOLDOUT_AFTER_2026_07_15", "state": "UNOPENED"},
        **AUTHORITY,
        "admission_blockers": [
            "opened 2026 data are development evidence",
            "untouched future FX holdout remains unopened",
            "both normal and adverse full-month 2.0x gates are mandatory",
            "strategy adoption remains a separate gate",
        ],
    }
    payload["result_sha256"] = embedded_hash(payload, "result_sha256")
    result = output_root / "result_causal_min_spread_representative_v26.json"
    result.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_root, args.parent_ledger, args.parent_result, args.output_root)
    print(json.dumps({
        "raw_signals": result["raw_signals"],
        "execution_selected_signals": result["execution_selected_signals"],
        "walk_forward": result["periods"]["WALK_FORWARD"],
        "automatic_rejection": result["automatic_rejection"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
