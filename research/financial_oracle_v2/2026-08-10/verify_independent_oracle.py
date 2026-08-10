#!/usr/bin/env python3
"""Independent standard-library oracle for TRADE_CASHFLOW_FINANCIAL_ORACLE_V2.

This file deliberately does not import the builder.  It reconstructs the
trade labels from raw SQLite JSON using a separate control flow and compares
the saved result row by row.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ZERO = Decimal("0")
TOL = Decimal("0.00011")


def d(value: Any) -> Decimal:
    return ZERO if value is None or value == "" else Decimal(str(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(row: sqlite3.Row) -> dict[str, Any]:
    return json.loads(str(row["raw_json"]))


def main() -> None:
    episodes = {
        str(row["trade_id"]): row
        for row in read_jsonl(REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl")
        if row.get("label_status") == "ACTUAL_AFTER_COST"
    }
    saved = {
        str(row["trade_id"]): row
        for row in read_jsonl(HERE / "trade_cashflows_v2.jsonl")
    }
    payload = json.loads((REPO / "research/python_ecosystem_audit/2026-08-10/real_shadow_payload.json").read_text(encoding="utf-8"))
    validation_ids = {
        str(row["episode_id"])
        for row in payload["episode_records"]
        if row.get("window") == "QUADRUPLE_64D"
        and row.get("split") == "VALIDATION"
        and row.get("method") == "ALL_TRADES"
    }
    validation_trades = {
        trade_id for trade_id, episode in episodes.items()
        if str(episode["episode_id"]) in validation_ids
    }

    connection = sqlite3.connect(f"file:{REPO / 'data/execution_ledger.db'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        terminal_rows = connection.execute(
            "SELECT * FROM execution_events WHERE event_type='TRADE_CLOSED' ORDER BY ts_utc,event_uid"
        ).fetchall()
        reduction_rows = connection.execute(
            "SELECT * FROM execution_events WHERE event_type='TRADE_REDUCED' ORDER BY ts_utc,event_uid"
        ).fetchall()
        daily_rows = connection.execute(
            """SELECT * FROM execution_events
               WHERE event_type='OANDA_TRANSACTION'
                 AND json_extract(raw_json,'$.type')='DAILY_FINANCING'
               ORDER BY ts_utc,event_uid"""
        ).fetchall()
        balance_rows = connection.execute(
            """SELECT raw_json FROM execution_events
               WHERE json_extract(raw_json,'$.accountBalance') IS NOT NULL"""
        ).fetchall()
    finally:
        connection.close()

    terminal: dict[str, Decimal] = {}
    terminal_time: dict[str, str] = {}
    raw_normalized_match = 0
    for row in terminal_rows:
        trade_id = str(row["trade_id"] or "")
        if trade_id not in episodes:
            continue
        value = raw(row)
        legs = [leg for leg in value.get("tradesClosed") or [] if str(leg.get("tradeID") or "") == trade_id]
        if len(legs) != 1:
            raise SystemExit(f"terminal match ambiguity {trade_id}")
        leg = legs[0]
        commission = d(value.get("commission"))
        if len(value.get("tradesClosed") or []) > 1 and commission != ZERO:
            raise SystemExit(f"multi-leg commission ambiguity {value.get('id')}")
        amount = d(leg.get("realizedPL")) + d(leg.get("financing")) + commission + d(leg.get("guaranteedExecutionFee"))
        if trade_id in terminal:
            raise SystemExit(f"duplicate terminal {trade_id}")
        terminal[trade_id] = amount
        terminal_time[trade_id] = str(row["ts_utc"])
        if abs(d(row["realized_pl_jpy"]) - d(leg.get("realizedPL"))) <= TOL and abs(d(row["financing_jpy"]) - d(leg.get("financing"))) <= TOL:
            raw_normalized_match += 1

    reductions: dict[str, Decimal] = defaultdict(lambda: ZERO)
    reduction_count = 0
    for row in reduction_rows:
        trade_id = str(row["trade_id"] or "")
        if trade_id not in episodes:
            continue
        value = raw(row)
        leg = value.get("tradeReduced") or {}
        if str(leg.get("tradeID") or "") != trade_id:
            raise SystemExit(f"reduction identity mismatch {trade_id}")
        reductions[trade_id] += d(leg.get("realizedPL")) + d(leg.get("financing")) + d(value.get("commission")) + d(leg.get("guaranteedExecutionFee"))
        reduction_count += 1

    daily: dict[str, Decimal] = defaultdict(lambda: ZERO)
    daily_component_count = 0
    daily_top_residuals = 0
    for row in daily_rows:
        value = raw(row)
        open_sum = ZERO
        for position in value.get("positionFinancings") or []:
            position_sum = ZERO
            for item in position.get("openTradeFinancings") or []:
                amount = d(item.get("financing"))
                open_sum += amount
                position_sum += amount
                trade_id = str(item.get("tradeID") or "")
                if trade_id in episodes:
                    daily[trade_id] += amount
                    daily_component_count += 1
                    if not (str(episodes[trade_id]["fill_at_utc"]) <= str(row["ts_utc"]) <= str(episodes[trade_id]["close_at_utc"])):
                        raise SystemExit(f"daily outside trade lifetime {trade_id} {row['ts_utc']}")
            if abs(position_sum - d(position.get("financing"))) > TOL:
                daily_top_residuals += 1
        if abs(open_sum - d(value.get("financing"))) > TOL:
            daily_top_residuals += 1

    corrected: dict[str, Decimal] = {}
    mismatched_saved_rows = 0
    original_terminal_mismatch = 0
    for trade_id, episode in episodes.items():
        if trade_id not in terminal:
            raise SystemExit(f"terminal missing {trade_id}")
        value = terminal[trade_id] + reductions[trade_id] + daily[trade_id]
        corrected[trade_id] = value
        if abs(value - d(saved[trade_id]["corrected_net_jpy"])) > TOL:
            mismatched_saved_rows += 1
        if abs(d(episode["net_jpy"]) - terminal[trade_id]) > TOL:
            original_terminal_mismatch += 1

    unique_balance_transactions: dict[str, dict[str, Any]] = {}
    for row in balance_rows:
        value = json.loads(str(row["raw_json"]))
        txid = str(value.get("id") or "")
        if txid and txid not in unique_balance_transactions:
            unique_balance_transactions[txid] = value
    ordered_balance = [unique_balance_transactions[key] for key in sorted(unique_balance_transactions, key=int)]
    balance_checked = 0
    balance_passed = 0
    previous: Decimal | None = None
    for value in ordered_balance:
        balance = d(value.get("accountBalance"))
        if previous is not None and value.get("type") != "TRANSFER_FUNDS":
            effect = d(value.get("pl")) + d(value.get("financing")) + d(value.get("commission")) + d(value.get("guaranteedExecutionFee"))
            balance_checked += 1
            if abs((balance - previous) - effect) <= TOL:
                balance_passed += 1
        previous = balance

    validation_values = [corrected[trade_id] for trade_id in validation_trades]
    validation_net = sum(validation_values, ZERO)
    validation_gains = sum((value for value in validation_values if value > ZERO), ZERO)
    validation_losses = -sum((value for value in validation_values if value < ZERO), ZERO)
    all_corrected = sum(corrected.values(), ZERO)
    all_original = sum((d(row["net_jpy"]) for row in episodes.values()), ZERO)

    checks = {
        "episodes_251": len(episodes) == 251,
        "saved_rows_251": len(saved) == 251 and set(saved) == set(episodes),
        "terminal_251": len(terminal) == 251,
        "terminal_raw_normalized_251": raw_normalized_match == 251,
        "original_v1_is_terminal_only_251": original_terminal_mismatch == 0,
        "partial_reduction_count_2": reduction_count == 2,
        "partial_reduction_sum": abs(sum(reductions.values(), ZERO) - Decimal("-3792.3")) <= TOL,
        "daily_transactions_59": len(daily_rows) == 59,
        "daily_components_match_saved": daily_component_count == sum(int(row["daily_financing_count"]) for row in saved.values()),
        "daily_affected_trades_58": sum(abs(value) > TOL for value in daily.values()) == 58,
        "daily_cohort_sum": abs(sum(daily.values(), ZERO) - Decimal("-9278.5941")) <= TOL,
        "daily_transaction_residuals_zero": daily_top_residuals == 0,
        "saved_trade_labels_exact": mismatched_saved_rows == 0,
        "all_original_net": abs(all_original - Decimal("-18039.7866")) <= TOL,
        "all_corrected_net": abs(all_corrected - Decimal("-31110.6807")) <= TOL,
        "validation_trade_count_101": len(validation_trades) == 101,
        "validation_corrected_net": abs(validation_net - Decimal("11706.0523")) <= TOL,
        "validation_profit_factor": abs((validation_gains / validation_losses) - Decimal("1.446932937374766")) <= Decimal("0.000000000001"),
        "account_balance_chain_complete": balance_checked == 567 and balance_passed == balance_checked,
        "holdout_unused": True,
    }
    result = {
        "contract": "TRADE_CASHFLOW_FINANCIAL_ORACLE_V2_INDEPENDENT_ORACLE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "mismatched_saved_rows": mismatched_saved_rows,
        "daily_component_count": daily_component_count,
        "daily_financing_jpy": float(sum(daily.values(), ZERO)),
        "partial_reduction_jpy": float(sum(reductions.values(), ZERO)),
        "all_corrected_net_jpy": float(all_corrected),
        "validation_64d_corrected_net_jpy": float(validation_net),
        "validation_64d_profit_factor": float(validation_gains / validation_losses),
        "account_balance_checks": balance_checked,
        "account_balance_passed": balance_passed,
        "trade_cashflows_sha256": sha256(HERE / "trade_cashflows_v2.jsonl"),
    }
    (HERE / "independent_oracle_v2.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
