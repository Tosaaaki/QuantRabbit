#!/usr/bin/env python3
"""Reconstruct and compare the frozen operator-alpha cohort."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
TRANSACTIONS = ROOT / "source_transactions_v1.json"
CANDLES = ROOT / "source_candles_v1.json"
MANIFEST = ROOT / "source_manifest_v1.json"
RECONSTRUCTION = ROOT / "trade_reconstruction_v1.json"
DECISION_TABLE = ROOT / "canonical_decision_table_v1.csv"
FUSION_TABLE = ROOT / "fusion_table_v1.json"
RECEIPTS = ROOT / "arm_receipts_v1.jsonl"
REPORT = ROOT / "comparison_report_v1.json"
TARGETS = ROOT / "target_arithmetic_v1.json"
VERDICT = ROOT / "verdict_v1.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def midpoint(row: dict[str, Any], field: str = "c") -> float:
    return (float(row["bid"][field]) + float(row["ask"][field])) / 2.0


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def completed_before(rows: list[dict[str, Any]], decision_time: datetime, seconds: int) -> list[dict[str, Any]]:
    return [row for row in rows if parse_time(row["time"]) + timedelta(seconds=seconds) <= decision_time]


def h4_direction_features(rows: list[dict[str, Any]], entry_time: datetime) -> dict[str, Any]:
    causal = completed_before(rows, entry_time, 4 * 3600)
    closes = [midpoint(row) for row in causal]
    if len(closes) < 24:
        return {"complete": False, "reason": "FEWER_THAN_24_COMPLETED_H4_BARS"}
    ema10 = ema(closes, 10)
    ema20 = ema(closes, 20)
    signals = {
        "ema10_vs_ema20": 1 if ema10[-1] > ema20[-1] else -1 if ema10[-1] < ema20[-1] else 0,
        "ema20_slope_3": 1 if ema20[-1] > ema20[-4] else -1 if ema20[-1] < ema20[-4] else 0,
        "close_vs_ema20": 1 if closes[-1] > ema20[-1] else -1 if closes[-1] < ema20[-1] else 0,
    }
    recent = causal[-12:]
    long_reasons = [key for key, value in signals.items() if value > 0]
    short_reasons = [key for key, value in signals.items() if value < 0]
    direction = "LONG" if len(long_reasons) == 3 else "SHORT" if len(short_reasons) == 3 else "CONFLICT"
    return {
        "complete": True,
        "last_completed_h4_start_utc": causal[-1]["time"],
        "last_close_mid": closes[-1],
        "ema10": ema10[-1],
        "ema20": ema20[-1],
        "recent_high_mid": max(midpoint(row, "h") for row in recent),
        "recent_low_mid": min(midpoint(row, "l") for row in recent),
        "signals": signals,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
        "direction": direction,
    }


def m5_context(rows: list[dict[str, Any]], entry_time: datetime, side: str) -> dict[str, Any]:
    causal = completed_before(rows, entry_time, 300)
    closes = [midpoint(row) for row in causal]
    if len(closes) < 12:
        return {"complete": False, "side_confirmed": False, "reason": "FEWER_THAN_12_COMPLETED_M5_BARS"}
    ema10 = ema(closes, 10)
    aligned = closes[-1] < ema10[-1] if side == "SHORT" else closes[-1] > ema10[-1]
    return {
        "complete": True,
        "last_completed_m5_start_utc": causal[-1]["time"],
        "last_close_mid": closes[-1],
        "ema10": ema10[-1],
        "side_confirmed": aligned,
    }


def reconstruct() -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    tx_packet = load_json(TRANSACTIONS)
    candle_packet = load_json(CANDLES)
    manifest = load_json(MANIFEST)
    for name, expected in manifest["files"].items():
        actual = sha(ROOT / name)
        if actual != expected["sha256"]:
            raise RuntimeError(f"source hash mismatch for {name}")
    tx_by_id = {row["id"]: row for row in tx_packet["transactions"]}
    candles = {
        (packet["entry_fill_id"], packet["granularity"]): packet["rows"]
        for packet in candle_packet["packets"]
    }
    trades: list[dict[str, Any]] = []
    for frozen in tx_packet["frozen_trades"]:
        entry = tx_by_id[frozen["entry_fill_id"]]
        close = tx_by_id[frozen["close_fill_id"]]
        units_signed = int(float(entry["units"]))
        side = "LONG" if units_signed > 0 else "SHORT"
        units = abs(units_signed)
        entry_price = float(entry["price"])
        close_price = float(close["price"])
        realized = float(close["pl"]) + float(close.get("financing") or 0.0) + float(close.get("commission") or 0.0)
        quote_pl = (close_price - entry_price) * units * (1 if side == "LONG" else -1)
        conversion = realized / quote_pl if quote_pl else 1.0
        entry_time = parse_time(entry["time"])
        close_time = parse_time(close["time"])
        opened = entry.get("tradeOpened") or {}
        h4 = h4_direction_features(candles[(entry["id"], "H4")], entry_time)
        m5 = m5_context(candles[(entry["id"], "M5")], entry_time, side)
        x_selected = bool(h4.get("complete") and h4.get("direction") == side)
        trades.append(
            {
                "cohort_id": f"{entry['id']}->{close['id']}",
                "label": frozen["label"],
                "entry_fill_id": entry["id"],
                "close_fill_id": close["id"],
                "pair": entry["instrument"],
                "side": side,
                "units": units,
                "entry_time_utc": entry["time"],
                "close_time_utc": close["time"],
                "holding_seconds": (close_time - entry_time).total_seconds(),
                "entry_price": entry_price,
                "close_price": close_price,
                "pip_size": 0.01 if entry["instrument"].endswith("_JPY") else 0.0001,
                "realized_after_cost_jpy": realized,
                "commission_jpy": float(close.get("commission") or 0.0),
                "financing_jpy": float(close.get("financing") or 0.0),
                "entry_half_spread_cost_jpy": float(opened.get("halfSpreadCost") or 0.0),
                "initial_margin_required_jpy": float(opened.get("initialMarginRequired") or 0.0),
                "entry_balance_jpy": float(entry.get("accountBalance") or 0.0),
                "quote_to_jpy_conversion": conversion,
                "terminal_reason": close.get("reason"),
                "h4": h4,
                "m5": m5,
                "x_structure_selected": x_selected,
            }
        )
    trades.sort(key=lambda row: row["entry_time_utc"])
    return trades, candles, tx_by_id


def derive_operator_parameters(trades: list[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in trades if row["label"].startswith("manual_win")]
    profit_floor = min(row["realized_after_cost_jpy"] for row in wins)
    max_hold = math.ceil(max(row["holding_seconds"] for row in wins) / 60.0) * 60
    loss_budget_activation = math.floor(min(row["holding_seconds"] for row in wins) / 60.0) * 60
    return {
        "derivation_cohort_ids": [row["cohort_id"] for row in wins],
        "profit_floor_jpy": profit_floor,
        "max_hold_seconds": max_hold,
        "loss_budget_activation_seconds": loss_budget_activation,
        "loss_budget_fraction_of_entry_equity": 0.0025,
        "slippage_stress": "additional adverse half of observed S5 closing spread",
        "status": "IN_SAMPLE_DESCRIPTION_NOT_VALIDATION",
    }


def operator_exit(
    trade: dict[str, Any], rows: list[dict[str, Any]], parameters: dict[str, Any]
) -> dict[str, Any]:
    entry_time = parse_time(trade["entry_time_utc"])
    deadline = entry_time + timedelta(seconds=parameters["max_hold_seconds"])
    risk_cap = trade["entry_balance_jpy"] * parameters["loss_budget_fraction_of_entry_equity"]
    candidates = []
    for row in rows:
        candle_start = parse_time(row["time"])
        mark_time = candle_start + timedelta(seconds=5)
        if mark_time <= entry_time or mark_time > deadline:
            continue
        bid = float(row["bid"]["c"])
        ask = float(row["ask"]["c"])
        spread = ask - bid
        if trade["side"] == "SHORT":
            exit_price = ask + spread / 2.0
            quote_pl = (trade["entry_price"] - exit_price) * trade["units"]
        else:
            exit_price = bid - spread / 2.0
            quote_pl = (exit_price - trade["entry_price"]) * trade["units"]
        pnl = quote_pl * trade["quote_to_jpy_conversion"]
        candidates.append((mark_time, exit_price, spread, pnl))
        elapsed = (mark_time - entry_time).total_seconds()
        if elapsed >= parameters["loss_budget_activation_seconds"] and pnl <= -risk_cap:
            reason = "LOSS_BUDGET"
            break
        if pnl >= parameters["profit_floor_jpy"]:
            reason = "PROFIT_HARVEST"
            break
    else:
        if not candidates:
            raise RuntimeError(f"no causal S5 exit mark for {trade['cohort_id']}")
        reason = "MAX_HOLD_TIMEOUT"
    if reason == "MAX_HOLD_TIMEOUT":
        mark_time, exit_price, spread, pnl = candidates[-1]
    else:
        mark_time, exit_price, spread, pnl = candidates[-1]
    return {
        "exit_time_utc": iso(mark_time),
        "exit_price_after_slippage_stress": exit_price,
        "exit_spread_price": spread,
        "after_cost_net_jpy": pnl,
        "holding_seconds": (mark_time - entry_time).total_seconds(),
        "reason": reason,
        "loss_budget_jpy": risk_cap,
        "source": "complete OANDA S5 bid/ask close after entry",
    }


def max_drawdown(values: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["selected"]]
    pnls = [float(row["after_cost_net_jpy"]) for row in rows]
    selected_pnls = [float(row["after_cost_net_jpy"]) for row in selected]
    gross_profit = sum(max(value, 0.0) for value in selected_pnls)
    gross_loss = -sum(min(value, 0.0) for value in selected_pnls)
    holds = [float(row["holding_seconds"]) for row in selected]
    return {
        "cohort_decisions": len(rows),
        "executed_or_retained": len(selected),
        "after_cost_net_jpy": sum(pnls),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy_per_cohort_decision_jpy": sum(pnls) / len(rows),
        "expectancy_per_execution_jpy": sum(selected_pnls) / len(selected) if selected else None,
        "max_drawdown_jpy": max_drawdown(pnls),
        "margin_required_peak_proxy_jpy": max((row["initial_margin_required_jpy"] for row in selected), default=0.0),
        "margin_required_peak_to_entry_balance_proxy": max(
            (row["initial_margin_required_jpy"] / row["entry_balance_jpy"] for row in selected if row["entry_balance_jpy"]),
            default=0.0,
        ),
        "holding_time_mean_seconds": statistics.mean(holds) if holds else None,
        "holding_time_median_seconds": statistics.median(holds) if holds else None,
        "holding_time_max_seconds": max(holds) if holds else None,
        "turnover_units": sum(row["units"] for row in selected),
        "wins": sum(value > 0 for value in selected_pnls),
        "losses": sum(value < 0 for value in selected_pnls),
        "max_losing_streak": _max_losing_streak(selected_pnls),
    }


def _max_losing_streak(values: list[float]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def build_arm_rows(
    trades: list[dict[str, Any]], candles: dict[tuple[str, str], list[dict[str, Any]]], parameters: dict[str, Any]
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for trade in trades:
        operator = operator_exit(trade, candles[(trade["entry_fill_id"], "S5")], parameters)
        common = {
            key: trade[key]
            for key in (
                "cohort_id", "label", "pair", "side", "units", "entry_time_utc",
                "entry_balance_jpy", "initial_margin_required_jpy",
            )
        }
        arms = {
            "BASELINE_ACTUAL": {
                "selected": True,
                "after_cost_net_jpy": trade["realized_after_cost_jpy"],
                "holding_seconds": trade["holding_seconds"],
                "exit_reason": trade["terminal_reason"],
                "exit_time_utc": trade["close_time_utc"],
            },
            "OPERATOR_ALPHA": {
                "selected": True,
                "after_cost_net_jpy": operator["after_cost_net_jpy"],
                "holding_seconds": operator["holding_seconds"],
                "exit_reason": operator["reason"],
                "exit_time_utc": operator["exit_time_utc"],
            },
            "X_STRUCTURE": {
                "selected": trade["x_structure_selected"],
                "after_cost_net_jpy": trade["realized_after_cost_jpy"] if trade["x_structure_selected"] else 0.0,
                "holding_seconds": trade["holding_seconds"] if trade["x_structure_selected"] else 0.0,
                "exit_reason": trade["terminal_reason"] if trade["x_structure_selected"] else "X_STRUCTURE_SKIP",
                "exit_time_utc": trade["close_time_utc"] if trade["x_structure_selected"] else None,
            },
            "X_OPERATOR_INTERACTION": {
                "selected": trade["x_structure_selected"],
                "after_cost_net_jpy": operator["after_cost_net_jpy"] if trade["x_structure_selected"] else 0.0,
                "holding_seconds": operator["holding_seconds"] if trade["x_structure_selected"] else 0.0,
                "exit_reason": operator["reason"] if trade["x_structure_selected"] else "X_STRUCTURE_SKIP",
                "exit_time_utc": operator["exit_time_utc"] if trade["x_structure_selected"] else None,
            },
        }
        for arm, result in arms.items():
            receipt = {"arm": arm, **common, **result}
            receipt["receipt_sha256"] = canonical_hash(receipt)
            receipts.append(receipt)
    return receipts


def build_decision_table(
    trades: list[dict[str, Any]], receipts: list[dict[str, Any]], tx_by_id: dict[str, Any]
) -> None:
    operator_by_trade = {
        row["cohort_id"]: row for row in receipts if row["arm"] == "OPERATOR_ALPHA"
    }
    current_boundary = tx_by_id.get("473207")
    rows: list[dict[str, Any]] = []
    for index, trade in enumerate(trades):
        op = operator_by_trade[trade["cohort_id"]]
        events = [
            ("OBSERVE", trade["entry_time_utc"], "completed H4/M5 and S5 bid/ask context"),
            ("DIRECTION", trade["entry_time_utc"], f"actual={trade['side']}; x_h4={trade['h4'].get('direction')}; m5_confirmed={trade['m5'].get('side_confirmed')}"),
            ("ENTRY", trade["entry_time_utc"], f"actual fill {trade['entry_price']} x {trade['units']}u"),
            ("HOLD_MONITOR", op["exit_time_utc"], f"operator exit trigger={op['exit_reason']}"),
            ("EXIT", trade["close_time_utc"], f"actual={trade['terminal_reason']}; pnl_jpy={trade['realized_after_cost_jpy']:.4f}"),
        ]
        if index + 1 < len(trades):
            next_entry = trades[index + 1]
            delay = (parse_time(next_entry["entry_time_utc"]) - parse_time(trade["close_time_utc"])).total_seconds()
            reentry_detail = f"observed next={next_entry['pair']} {next_entry['side']} after {delay:.3f}s"
            reentry_time = next_entry["entry_time_utc"]
        elif current_boundary:
            delay = (parse_time(current_boundary["time"]) - parse_time(trade["close_time_utc"])).total_seconds()
            reentry_detail = f"observed open boundary={current_boundary['instrument']} after {delay:.3f}s; NO_TOUCH"
            reentry_time = current_boundary["time"]
        else:
            reentry_detail = "no observed next entry"
            reentry_time = trade["close_time_utc"]
        events.append(("REENTRY", reentry_time, reentry_detail))
        for state, timestamp, detail in events:
            rows.append(
                {
                    "cohort_id": trade["cohort_id"],
                    "label": trade["label"],
                    "state": state,
                    "timestamp_utc": timestamp,
                    "pair": trade["pair"],
                    "side": trade["side"],
                    "detail": detail,
                    "live_permission": "false",
                    "position_boundary": "NO_TOUCH",
                }
            )
    with DECISION_TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_fusion(trades: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for trade in trades:
        h4_support = trade["h4"].get("short_reasons" if trade["side"] == "SHORT" else "long_reasons", [])
        missing = ["forecast_decision_time_record", "account_margin_available_used", "executable_unwind_packet"]
        rows.append(
            {
                "decision_id": trade["cohort_id"],
                "pair": trade["pair"],
                "actual_side": trade["side"],
                "families": {
                    "technical": {"status": "USED", "h4_support": h4_support, "m5_confirmed": trade["m5"].get("side_confirmed")},
                    "forecast": {"status": "MISSING", "reason": "no causal decision-time record in the frozen packet"},
                    "price_action": {"status": "USED", "source": "completed M5 close versus EMA10"},
                    "continuous_thesis": {"status": "DERIVED_RESEARCH", "entry_frozen_direction": trade["h4"].get("direction"), "monitor": "S5 after-cost and elapsed-time state"},
                    "multidimensional_cube": {"status": "SPARSE", "missing_cells": missing},
                    "hedge": {"status": "NOT_EDGE", "reason": "no explicit executable unwind; cannot grant entry"},
                    "trailing_BE": {"status": "NOT_ACTIVATED_IN_V1", "reason": "separate causal next-quote study required"},
                    "execution": {"status": "USED_RESEARCH", "source": "side-correct S5 bid/ask"},
                    "risk_margin": {"status": "INCOMPLETE", "initial_margin_required_jpy": trade["initial_margin_required_jpy"], "missing": missing[1:]},
                    "operator_alpha": {"status": "USED_RESEARCH", "role": "exit and rotation contract"},
                    "x_method": {"status": "USED_AS_STRUCTURE_ONLY", "selected": trade["x_structure_selected"]},
                },
                "research_answer": "DIAGNOSTIC_CANDIDATE" if trade["x_structure_selected"] else "WAIT_DIRECTION_CONFLICT",
                "live_answer": "WAIT_EVIDENCE_INCOMPLETE",
                "live_permission": False,
                "decisive_constraint": missing,
            }
        )
    return {
        "contract": "OPERATOR_ALPHA_FUSION_TABLE_V1",
        "one_answer_per_decision": True,
        "repository_lineage": {
            "technical_forecast_price_action_fusion": "research/system_utilization_rca/2026-08-10/preregister_v1.json",
            "decision_time_execution": "research/decision_time_execution_evidence/2026-08-10/preregister_v1.json",
            "continuous_thesis_exit_audit": "research/continuous_thesis_monitor/2026-08-10/owner_exit_target_audit_v1.md",
            "active_TP_SL_replacements": "research/active_protection_schedule/2026-08-10/verdict_v1.md",
            "trailing_BE_partial_exit_cube": "research/financial_oracle_v2/2026-08-10/exit_report_v1.json",
            "margin_and_loss_floor": "research/capital_preservation_gate/2026-08-10/preregister_v1.json",
            "multidimensional_cube": "research/python_ecosystem_audit/2026-08-10/canonical_long_table.jsonl"
        },
        "fusion_semantics": {
            "edge_families": ["technical", "forecast", "price_action", "strategy_evidence"],
            "constraint_families": ["execution", "risk_margin", "trailing_BE", "hedge"],
            "operator_alpha_role": "state/exit/rotation consumer; never an independent forecast family",
            "x_role": "measurable checklist only; no edge prior",
            "missing_is_not_zero": True,
            "one_answer": ["TRADE", "WAIT", "SKIP", "MANAGE"]
        },
        "rows": rows,
    }


def build_targets(trades: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    wins = [row for row in trades if row["label"].startswith("manual_win")]
    start = wins[0]["entry_balance_jpy"]
    total = sum(row["realized_after_cost_jpy"] for row in wins)
    mean = total / len(wins)
    goals = {}
    for rate in (0.10, 0.50):
        needed = start * rate
        goals[f"{int(rate * 100)}pct"] = {
            "target_profit_jpy": needed,
            "equivalent_four_win_batches_at_observed_total": needed / total,
            "equivalent_wins_at_observed_mean": needed / mean,
            "required_daily_profit_over_30_days_jpy": needed / 30.0,
            "required_wins_per_day_at_observed_mean": needed / 30.0 / mean,
            "required_average_jpy_by_opportunities_per_day": {
                str(per_day): needed / (30.0 * per_day) for per_day in (1, 2, 4)
            },
        }
    margin_losses = [abs(row["realized_after_cost_jpy"]) for row in trades if "margin_closeout" in row["label"]]
    return {
        "contract": "OBSERVED_PACE_NOT_GUARANTEE_V1",
        "starting_balance_jpy": start,
        "four_win_total_jpy": total,
        "four_win_return_fraction": total / start,
        "mean_win_jpy": mean,
        "goals": goals,
        "break_conditions": {
            "margin_closeout_losses_jpy": margin_losses,
            "combined_margin_closeout_loss_jpy": sum(margin_losses),
            "combined_loss_as_four_win_batches": sum(margin_losses) / total,
            "largest_loss_as_four_win_batches": max(margin_losses) / total,
            "interpretation": "One unlimited hold can erase multiple fast-rotation batches; the observed pace is invalid if maximum hold, loss budget, margin completeness, or opportunity density fails."
        },
        "arm_context": {arm: data["metrics"] for arm, data in report["arms"].items()},
        "guarantee": False,
    }


def run() -> dict[str, Any]:
    trades, candles, tx_by_id = reconstruct()
    parameters = derive_operator_parameters(trades)
    receipts = build_arm_rows(trades, candles, parameters)
    arms: dict[str, Any] = {}
    for arm in ("BASELINE_ACTUAL", "OPERATOR_ALPHA", "X_STRUCTURE", "X_OPERATOR_INTERACTION"):
        rows = [row for row in receipts if row["arm"] == arm]
        cohort_ids = [row["cohort_id"] for row in rows]
        arms[arm] = {"cohort_ids": cohort_ids, "metrics": metrics(rows)}
    cohort_sets = {tuple(value["cohort_ids"]) for value in arms.values()}
    if len(cohort_sets) != 1:
        raise RuntimeError("arm cohort mismatch")
    wins = [row for row in trades if row["label"].startswith("manual_win")]
    observed_total = sum(row["realized_after_cost_jpy"] for row in wins)
    if not math.isclose(observed_total, 5052.0833, abs_tol=1e-7):
        raise RuntimeError(f"four-win total mismatch: {observed_total}")
    reconstruction = {
        "contract": "OPERATOR_ALPHA_TRADE_RECONSTRUCTION_V1",
        "source_manifest_sha256": sha(MANIFEST),
        "operator_parameters": parameters,
        "four_win_summary": {
            "start_balance_jpy": wins[0]["entry_balance_jpy"],
            "after_cost_net_jpy": observed_total,
            "return_fraction": observed_total / wins[0]["entry_balance_jpy"],
            "mean_jpy_per_trade": observed_total / len(wins),
        },
        "open_boundary": {
            "entry_fill_id": "473207",
            "state": "OPEN_AT_TRANSACTION_WATERMARK_473209",
            "ownership": "manual_or_unknown",
            "action": "NO_TOUCH",
        },
        "trades": trades,
    }
    write_json(RECONSTRUCTION, reconstruction)
    RECEIPTS.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts), encoding="utf-8")
    build_decision_table(trades, receipts, tx_by_id)
    fusion = build_fusion(trades)
    write_json(FUSION_TABLE, fusion)
    report = {
        "contract": "OPERATOR_ALPHA_FOUR_ARM_COMPARISON_V1",
        "status": "IN_SAMPLE_DIAGNOSTIC_NOT_ADOPTION_EVIDENCE",
        "same_frozen_cohort": True,
        "cohort_size": len(trades),
        "operator_parameters": parameters,
        "arms": arms,
        "limitations": [
            "six trades, all SHORT, from one account and two pairs",
            "operator thresholds were derived from four rows in the same cohort",
            "X post contributes a structure checklist, not predictive edge",
            "decision-time margin available/used, full inventory, forecast, and executable unwind packets are missing",
            "S5 close ordering is causal but cannot reproduce raw ticks inside a five-second bar",
            "margin metric is required-margin proxy, not broker account margin or netting"
        ],
        "live_permission": False,
        "source_hashes": {
            "transactions": sha(TRANSACTIONS),
            "candles": sha(CANDLES),
            "acquisition_contract": sha(ROOT / "acquisition_contract_v1.json"),
            "operator_contract": sha(ROOT / "operator_alpha_contract_v1.json"),
            "x_contract": sha(ROOT / "x_claims_contract_v1.json"),
        },
    }
    write_json(REPORT, report)
    targets = build_targets(trades, report)
    write_json(TARGETS, targets)
    write_verdict(report, reconstruction, targets)
    return report


def write_verdict(report: dict[str, Any], reconstruction: dict[str, Any], targets: dict[str, Any]) -> None:
    arm_lines = []
    for arm, payload in report["arms"].items():
        m = payload["metrics"]
        pf = "null" if m["profit_factor"] is None else f"{m['profit_factor']:.4f}"
        arm_lines.append(
            f"| {arm} | {m['executed_or_retained']} | {m['after_cost_net_jpy']:.4f} | "
            f"{pf} | {m['expectancy_per_cohort_decision_jpy']:.4f} | {m['max_drawdown_jpy']:.4f} | "
            f"{m['holding_time_mean_seconds'] if m['holding_time_mean_seconds'] is not None else 0:.1f} | {m['turnover_units']} |"
        )
    margin_losses = targets["break_conditions"]["margin_closeout_losses_jpy"]
    text = f"""# Operator-alpha fast rotation verdict

Status: **RESEARCH CONTRACT COMPLETE / LIVE ADOPTION BLOCKED**

The four consecutive manual wins are broker-confirmed at **+5,052.0833 JPY**
(**{reconstruction['four_win_summary']['return_fraction'] * 100:.4f}%** of
{reconstruction['four_win_summary']['start_balance_jpy']:.4f} JPY).  They form a reproducible behavior shape:
pair-specific short direction, executable entry, fast after-cost harvest, close
confirmation, and fresh-evidence rotation.  This is evidence that the operator
performed the behavior, not proof of future expectancy.

| Arm | Selected | Net JPY | PF | Expectancy/decision | DD JPY | Mean hold sec | Turnover units |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(arm_lines)}

`OPERATOR_ALPHA` derives its profit floor ({report['operator_parameters']['profit_floor_jpy']:.4f} JPY)
and maximum holding time ({report['operator_parameters']['max_hold_seconds']} seconds) from the four
wins, then applies an equity-derived 0.25% loss budget and side-correct S5
bid/ask plus adverse half-spread exit stress to all six entries.  The result is
an in-sample diagnostic, not a validation result.  `X_STRUCTURE` adds only the
post's completed-H4/reasons/skip checklist; it does not import the post's
monthly-income or ten-minutes-per-day claims.

The two margin closeouts are broker-confirmed at -{margin_losses[0]:.4f} and
-{margin_losses[1]:.4f} JPY.  Together they equal
{targets['break_conditions']['combined_loss_as_four_win_batches']:.4f} observed four-win batches.
They are classified as contract failures: a fast scalp became a long,
margin-controlled hold.  Margin closeout is never an acceptable exit or a
reason to increase leverage.

## Adoption boundary

The fusion table returns `WAIT_EVIDENCE_INCOMPLETE` for live use because the
frozen packet lacks decision-time margin available/used, full inventory,
forecast lineage, and an executable unwind packet.  The currently open manual
or unknown position at entry fill 473207 remains `NO_TOUCH`.

The 10% and 50% figures in `target_arithmetic_v1.json` are arithmetic scenarios,
not guarantees.  They report required P/L, trade density, and break conditions.
No live, Paper, order, broker mutation, deployment, or holdout action occurred.
"""
    VERDICT.write_text(text, encoding="utf-8")


def main() -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
