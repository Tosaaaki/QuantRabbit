#!/usr/bin/env python3
"""Build corrected trade cash-flow labels from immutable OANDA transaction truth.

The builder is intentionally independent from the historical-learning episode
builder whose last-close-only join is under audit.  It opens SQLite read-only,
uses raw broker transaction identities, and fails closed on every allocation or
arithmetic ambiguity.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREREG = HERE / "preregister_v2.json"
TOLERANCE = Decimal("0.00011")
ZERO = Decimal("0")


def dec(value: Any) -> Decimal:
    if value is None or value == "":
        return ZERO
    return Decimal(str(value))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def raw_payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        value = json.loads(str(row["raw_json"]))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def tx_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    return (int(text), text) if text.isdigit() else (2**63 - 1, text)


def close_component(row: sqlite3.Row, episode_trade_ids: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    raw = raw_payload(row)
    raw_type = str(raw.get("type") or "")
    issues: list[str] = []
    transaction_id = str(raw.get("id") or row["oanda_transaction_id"] or "")
    ts_utc = str(raw.get("time") or row["ts_utc"])
    event_type = str(row["event_type"])
    if raw_type != "ORDER_FILL":
        return [], [f"NON_ORDER_FILL:{row['event_uid']}"]

    if event_type == "TRADE_CLOSED":
        legs = raw.get("tradesClosed") or []
        if not isinstance(legs, list) or not legs:
            return [], [f"TERMINAL_LEG_COUNT:{transaction_id}:{len(legs) if isinstance(legs, list) else 'INVALID'}"]
        normalized_trade_id = str(row["trade_id"] or "")
        matching = [item for item in legs if str(item.get("tradeID") or "") == normalized_trade_id]
        if len(matching) != 1:
            return [], [f"TERMINAL_MATCH_COUNT:{transaction_id}:{normalized_trade_id}:{len(matching)}"]
        leg = matching[0]
        kind = "TERMINAL_CLOSE"
    elif event_type == "TRADE_REDUCED":
        leg = raw.get("tradeReduced")
        if not isinstance(leg, dict):
            return [], [f"REDUCTION_LEG_MISSING:{transaction_id}"]
        legs = [leg]
        kind = "PARTIAL_REDUCTION"
    else:
        return [], [f"UNEXPECTED_EVENT_TYPE:{event_type}"]

    trade_id = str(leg.get("tradeID") or row["trade_id"] or "")
    if trade_id not in episode_trade_ids:
        return [], []
    realized = dec(leg.get("realizedPL"))
    financing = dec(leg.get("financing"))
    commission = dec(raw.get("commission"))
    if len(legs) > 1 and commission != ZERO:
        issues.append(f"MULTI_LEG_COMMISSION_UNALLOCATED:{transaction_id}")
        commission = ZERO
    guaranteed_fee = dec(leg.get("guaranteedExecutionFee"))
    normalized_realized = dec(row["realized_pl_jpy"])
    normalized_financing = dec(row["financing_jpy"])
    if abs(realized - normalized_realized) > TOLERANCE:
        issues.append(f"NORMALIZED_REALIZED_MISMATCH:{transaction_id}:{trade_id}")
    if abs(financing - normalized_financing) > TOLERANCE:
        issues.append(f"NORMALIZED_FINANCING_MISMATCH:{transaction_id}:{trade_id}")
    total_leg_pl = sum((dec(item.get("realizedPL")) for item in legs), ZERO)
    total_leg_financing = sum((dec(item.get("financing")) for item in legs), ZERO)
    total_leg_guaranteed_fee = sum((dec(item.get("guaranteedExecutionFee")) for item in legs), ZERO)
    if abs(dec(raw.get("pl")) - total_leg_pl) > TOLERANCE:
        issues.append(f"TOP_PL_MISMATCH:{transaction_id}")
    if abs(dec(raw.get("financing")) - total_leg_financing) > TOLERANCE:
        issues.append(f"TOP_FINANCING_MISMATCH:{transaction_id}")
    if abs(dec(raw.get("guaranteedExecutionFee")) - total_leg_guaranteed_fee) > TOLERANCE:
        issues.append(f"TOP_GUARANTEED_FEE_MISMATCH:{transaction_id}")

    component = {
        "component_id": f"{transaction_id}:{kind}:{trade_id}:0",
        "component_kind": kind,
        "event_uid": str(row["event_uid"]),
        "transaction_id": transaction_id,
        "ts_utc": ts_utc,
        "trade_id": trade_id,
        "units": abs(int(dec(leg.get("units") or row["units"]))),
        "realized_pl_jpy": float(realized),
        "financing_jpy": float(financing),
        "commission_jpy": float(commission),
        "guaranteed_execution_fee_jpy": float(guaranteed_fee),
        "amount_jpy": float(realized + financing + commission + guaranteed_fee),
        "raw_sha256": hashlib.sha256(str(row["raw_json"]).encode("utf-8")).hexdigest(),
        "allocation_status": "EXACT_TRADE_ID",
    }
    return [component], issues


def daily_components(row: sqlite3.Row, episode_trade_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    raw = raw_payload(row)
    transaction_id = str(raw.get("id") or row["oanda_transaction_id"] or "")
    ts_utc = str(raw.get("time") or row["ts_utc"])
    issues: list[str] = []
    components: list[dict[str, Any]] = []
    all_open_sum = ZERO
    out_of_cohort_sum = ZERO
    ordinal = 0
    positions = raw.get("positionFinancings") or []
    if not isinstance(positions, list):
        return [], {}, [f"POSITION_FINANCINGS_INVALID:{transaction_id}"]
    for position_index, position in enumerate(positions):
        open_rows = position.get("openTradeFinancings") or []
        if not isinstance(open_rows, list):
            issues.append(f"OPEN_TRADE_FINANCINGS_INVALID:{transaction_id}:{position_index}")
            continue
        position_sum = sum((dec(item.get("financing")) for item in open_rows), ZERO)
        if abs(position_sum - dec(position.get("financing"))) > TOLERANCE:
            issues.append(f"POSITION_FINANCING_RESIDUAL:{transaction_id}:{position_index}")
        for item in open_rows:
            trade_id = str(item.get("tradeID") or "")
            amount = dec(item.get("financing"))
            all_open_sum += amount
            if trade_id not in episode_trade_ids:
                out_of_cohort_sum += amount
                continue
            components.append({
                "component_id": f"{transaction_id}:DAILY_FINANCING:{trade_id}:{ordinal}",
                "component_kind": "DAILY_FINANCING",
                "event_uid": str(row["event_uid"]),
                "transaction_id": transaction_id,
                "ts_utc": ts_utc,
                "trade_id": trade_id,
                "units": None,
                "realized_pl_jpy": 0.0,
                "financing_jpy": float(amount),
                "commission_jpy": 0.0,
                "guaranteed_execution_fee_jpy": 0.0,
                "amount_jpy": float(amount),
                "raw_sha256": hashlib.sha256(str(row["raw_json"]).encode("utf-8")).hexdigest(),
                "allocation_status": "EXACT_OPEN_TRADE_FINANCING_TRADE_ID",
            })
            ordinal += 1
    top = dec(raw.get("financing"))
    if abs(all_open_sum - top) > TOLERANCE:
        issues.append(f"DAILY_FINANCING_TOP_RESIDUAL:{transaction_id}")
    audit = {
        "transaction_id": transaction_id,
        "ts_utc": ts_utc,
        "top_financing_jpy": float(top),
        "open_trade_financing_sum_jpy": float(all_open_sum),
        "cohort_financing_jpy": float(sum((dec(item["amount_jpy"]) for item in components), ZERO)),
        "out_of_cohort_financing_jpy": float(out_of_cohort_sum),
        "cohort_component_count": len(components),
        "residual_jpy": float(top - all_open_sum),
    }
    return components, audit, issues


def max_drawdown(values: list[Decimal]) -> Decimal:
    equity = ZERO
    peak = ZERO
    worst = ZERO
    for value in values:
        equity += value
        peak = max(peak, equity)
        worst = max(worst, peak - equity)
    return worst


def profit_factor(values: list[Decimal]) -> float | None:
    gains = sum((value for value in values if value > ZERO), ZERO)
    losses = -sum((value for value in values if value < ZERO), ZERO)
    if losses:
        return float(gains / losses)
    return math.inf if gains else None


def metrics(trades: list[dict[str, Any]], components: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(trades, key=lambda row: (str(row["close_at_utc"]), str(row["episode_id"])))
    values = [dec(row["corrected_net_jpy"]) for row in ordered]
    original = [dec(row["original_v1_net_jpy"]) for row in ordered]
    gains = sum((value for value in values if value > ZERO), ZERO)
    losses = -sum((value for value in values if value < ZERO), ZERO)
    tx_groups: dict[tuple[str, str], Decimal] = defaultdict(lambda: ZERO)
    allowed = {str(row["trade_id"]) for row in trades}
    for component in components:
        if str(component["trade_id"]) in allowed:
            tx_groups[(str(component["ts_utc"]), str(component["transaction_id"]))] += dec(component["amount_jpy"])
    event_values = [value for _, value in sorted(tx_groups.items(), key=lambda item: (item[0][0], tx_key(item[0][1])))]
    return {
        "trades": len(ordered),
        "net_jpy": float(sum(values, ZERO)),
        "original_v1_net_jpy": float(sum(original, ZERO)),
        "paired_delta_vs_v1_jpy": float(sum(values, ZERO) - sum(original, ZERO)),
        "profit_factor": profit_factor(values),
        "gross_gain_jpy": float(gains),
        "gross_loss_jpy": float(losses),
        "expectancy_jpy": float(sum(values, ZERO) / Decimal(len(values))) if values else None,
        "win_rate": sum(value > ZERO for value in values) / len(values) if values else None,
        "episode_terminal_max_drawdown_jpy": float(max_drawdown(values)),
        "event_time_max_drawdown_jpy": float(max_drawdown(event_values)),
        "event_time_cashflow_transactions": len(event_values),
    }


def split_index(payload: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in payload.get("episode_records") or []:
        if row.get("method") != "ALL_TRADES":
            continue
        key = (str(row["window"]), str(row["split"]), str(row["episode_id"]))
        if key in seen:
            continue
        seen.add(key)
        result[(key[0], key[1])].append(key[2])
    return result


def run() -> dict[str, Any]:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    for binding in prereg["source_bindings"].values():
        path = REPO / binding["path"]
        if sha256_path(path) != binding["sha256"]:
            raise RuntimeError(f"SOURCE_HASH_MISMATCH:{binding['path']}")

    episodes = [
        row for row in read_jsonl(REPO / prereg["source_bindings"]["episodes_v1"]["path"])
        if row.get("label_status") == "ACTUAL_AFTER_COST"
    ]
    if len(episodes) != 251:
        raise RuntimeError(f"EPISODE_COUNT:{len(episodes)}")
    episode_by_trade: dict[str, dict[str, Any]] = {}
    episode_by_id: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        trade_id = str(episode.get("trade_id") or "")
        episode_id = str(episode["episode_id"])
        if not trade_id or trade_id in episode_by_trade or episode_id in episode_by_id:
            raise RuntimeError(f"NON_UNIQUE_EPISODE_TRADE:{episode_id}:{trade_id}")
        episode_by_trade[trade_id] = episode
        episode_by_id[episode_id] = episode
    trade_ids = set(episode_by_trade)

    schedule_summaries = {
        str(row["trade_id"]): row
        for row in read_jsonl(REPO / prereg["source_bindings"]["active_protection_summaries"]["path"])
    }
    if set(schedule_summaries) != trade_ids:
        raise RuntimeError("ACTIVE_SCHEDULE_TRADE_SET_MISMATCH")
    if not all(row.get("strict_schedule_eligible") is True for row in schedule_summaries.values()):
        raise RuntimeError("ACTIVE_SCHEDULE_NOT_STRICT")

    db_path = REPO / prereg["source_bindings"]["execution_ledger"]["path"]
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        close_rows = connection.execute(
            """SELECT * FROM execution_events
               WHERE event_type IN ('TRADE_CLOSED','TRADE_REDUCED')
               ORDER BY ts_utc, CAST(oanda_transaction_id AS INTEGER), event_uid"""
        ).fetchall()
        daily_rows = connection.execute(
            """SELECT * FROM execution_events
               WHERE event_type='OANDA_TRANSACTION'
                 AND json_extract(raw_json,'$.type')='DAILY_FINANCING'
               ORDER BY ts_utc, CAST(oanda_transaction_id AS INTEGER), event_uid"""
        ).fetchall()
    finally:
        connection.close()

    components: list[dict[str, Any]] = []
    issues: list[str] = []
    daily_audits: list[dict[str, Any]] = []
    for row in close_rows:
        additions, row_issues = close_component(row, trade_ids)
        components.extend(additions)
        issues.extend(row_issues)
    for row in daily_rows:
        additions, audit, row_issues = daily_components(row, trade_ids)
        components.extend(additions)
        daily_audits.append(audit)
        issues.extend(row_issues)

    component_ids = [str(row["component_id"]) for row in components]
    if len(component_ids) != len(set(component_ids)):
        issues.append("DUPLICATE_COMPONENT_ID")
    components.sort(key=lambda row: (str(row["ts_utc"]), tx_key(row["transaction_id"]), str(row["component_id"])))

    by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        by_trade[str(component["trade_id"])].append(component)

    trade_rows: list[dict[str, Any]] = []
    for trade_id, episode in episode_by_trade.items():
        rows = by_trade.get(trade_id, [])
        terminal = [row for row in rows if row["component_kind"] == "TERMINAL_CLOSE"]
        reductions = [row for row in rows if row["component_kind"] == "PARTIAL_REDUCTION"]
        daily = [row for row in rows if row["component_kind"] == "DAILY_FINANCING"]
        trade_issues: list[str] = []
        if len(terminal) != 1:
            trade_issues.append(f"TERMINAL_COUNT:{len(terminal)}")
        fill_at = str(episode["fill_at_utc"])
        close_at = str(episode["close_at_utc"])
        for row in rows:
            if not (fill_at <= str(row["ts_utc"]) <= close_at):
                trade_issues.append(f"CASHFLOW_OUTSIDE_LIFETIME:{row['component_id']}")
        corrected = sum((dec(row["amount_jpy"]) for row in rows), ZERO) if not trade_issues else None
        original = dec(episode["net_jpy"])
        row = {
            "episode_id": str(episode["episode_id"]),
            "trade_id": trade_id,
            "pair": str(episode["pair"]),
            "side": str(episode["side"]),
            "units": int(episode["units"]),
            "fill_at_utc": fill_at,
            "close_at_utc": close_at,
            "original_v1_net_jpy": float(original),
            "terminal_close_jpy": float(sum((dec(item["amount_jpy"]) for item in terminal), ZERO)),
            "partial_reduction_jpy": float(sum((dec(item["amount_jpy"]) for item in reductions), ZERO)),
            "daily_financing_jpy": float(sum((dec(item["amount_jpy"]) for item in daily), ZERO)),
            "partial_reduction_count": len(reductions),
            "daily_financing_count": len(daily),
            "component_count": len(rows),
            "corrected_net_jpy": float(corrected) if corrected is not None else None,
            "paired_delta_vs_v1_jpy": float(corrected - original) if corrected is not None else None,
            "allocation_status": "PASS" if not trade_issues else "NOT_EVALUABLE",
            "issues": trade_issues,
            "active_protection_status": "STRICT_VERSIONED_PASS",
            "active_terminal_match": schedule_summaries[trade_id].get("terminal_active_match"),
        }
        trade_rows.append(row)
        issues.extend(f"{trade_id}:{issue}" for issue in trade_issues)
    trade_rows.sort(key=lambda row: (str(row["close_at_utc"]), str(row["episode_id"])))

    payload = json.loads((REPO / prereg["source_bindings"]["split_membership"]["path"]).read_text(encoding="utf-8"))
    splits = split_index(payload)
    trade_by_episode = {str(row["episode_id"]): row for row in trade_rows}
    windows: dict[str, dict[str, Any]] = defaultdict(dict)
    for (window, split), episode_ids in sorted(splits.items()):
        missing = [episode_id for episode_id in episode_ids if episode_id not in trade_by_episode]
        scoped = [trade_by_episode[episode_id] for episode_id in episode_ids if episode_id in trade_by_episode]
        if missing or any(row["allocation_status"] != "PASS" for row in scoped):
            windows[window][split] = {"status": "NOT_EVALUABLE", "missing_episode_ids": missing}
        else:
            windows[window][split] = {"status": "PASS", **metrics(scoped, components)}

    affected_daily = {row["trade_id"] for row in trade_rows if abs(dec(row["daily_financing_jpy"])) > TOLERANCE}
    partial_rows = [row for row in components if row["component_kind"] == "PARTIAL_REDUCTION"]
    daily_components_in_cohort = [row for row in components if row["component_kind"] == "DAILY_FINANCING"]
    report = {
        "contract": "TRADE_CASHFLOW_FINANCIAL_ORACLE_V2",
        "status": "PASS" if not issues else "NOT_EVALUABLE",
        "holdout_used": False,
        "episodes": len(trade_rows),
        "strict_allocated": sum(row["allocation_status"] == "PASS" for row in trade_rows),
        "issues": sorted(set(issues)),
        "source_daily_financing_transactions": len(daily_rows),
        "cohort_daily_financing_components": len(daily_components_in_cohort),
        "cohort_trades_with_nonzero_daily_financing": len(affected_daily),
        "cohort_daily_financing_jpy": float(sum((dec(row["amount_jpy"]) for row in daily_components_in_cohort), ZERO)),
        "cohort_partial_reduction_components": len(partial_rows),
        "cohort_partial_reduction_jpy": float(sum((dec(row["amount_jpy"]) for row in partial_rows), ZERO)),
        "all_episodes": metrics(trade_rows, components) if not issues else None,
        "windows": dict(sorted(windows.items())),
        "daily_transaction_audits": daily_audits,
        "active_versioned_protection": {
            "strict_eligible": len(schedule_summaries),
            "terminal_protection_orders": sum(row.get("terminal_order_kind") in {"TP", "SL"} for row in schedule_summaries.values()),
            "terminal_active_matches": sum(row.get("terminal_active_match") is True for row in schedule_summaries.values()),
        },
        "path_replay_gate": {
            "financial_oracle_pass": not issues,
            "unknown_intrabar_order": "UNRESOLVED",
            "exit_arms_evaluated": False,
        },
        "permissions": prereg["permissions"],
    }
    write_jsonl(HERE / "cashflow_components_v2.jsonl", components)
    write_jsonl(HERE / "trade_cashflows_v2.jsonl", trade_rows)
    write_json(HERE / "financial_oracle_v2.json", report)
    manifest = {
        "contract": report["contract"],
        "status": report["status"],
        "preregister_sha256": sha256_path(PREREG),
        "outputs": {
            name: sha256_path(HERE / name)
            for name in ("cashflow_components_v2.jsonl", "trade_cashflows_v2.jsonl", "financial_oracle_v2.json")
        },
    }
    write_json(HERE / "run_manifest_v2.json", manifest)
    return report


def main() -> None:
    report = run()
    print(canonical_json({
        "status": report["status"],
        "episodes": report["episodes"],
        "strict_allocated": report["strict_allocated"],
        "daily_financing_jpy": report["cohort_daily_financing_jpy"],
        "partial_reduction_jpy": report["cohort_partial_reduction_jpy"],
        "validation_64d": report["windows"].get("QUADRUPLE_64D", {}).get("VALIDATION"),
    }))


if __name__ == "__main__":
    main()
