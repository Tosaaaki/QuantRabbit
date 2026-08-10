#!/usr/bin/env python3
"""Build causal executable-path diagnostics for the frozen 251 trade cohort.

OANDA bid/ask S5 candles are used only as observed executable-side wick truth.
Missing candle endpoints and within-S5 ordering remain unresolved.  The script
never forward-fills, substitutes mid/M1, or converts a wick into a proven fill.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREREG = HERE / "path_preregister_v1.json"
DB = REPO / "data/execution_ledger.db"
SCHEDULE = REPO / "research/active_protection_schedule/2026-08-10/schedule_events_v1.jsonl"
TRADES = HERE / "trade_cashflows_v2.jsonl"
NS = 1_000_000_000
BAR_NS = 5 * NS


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def parse_ns(value: str) -> int:
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z", value)
    if not match:
        raise ValueError(f"unsupported UTC timestamp: {value}")
    whole = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
    fraction = (match.group(2) or "")[:9].ljust(9, "0")
    return int(whole.timestamp()) * NS + int(fraction)


def ns_to_utc(value: int) -> str:
    whole, fraction = divmod(value, NS)
    base = datetime.fromtimestamp(whole, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{fraction:09d}Z"


def ceil_bar(value: int) -> int:
    return ((value + BAR_NS - 1) // BAR_NS) * BAR_NS


def raw(row: sqlite3.Row) -> dict[str, Any]:
    value = json.loads(str(row["raw_json"]))
    if not isinstance(value, dict):
        raise ValueError(f"raw transaction is not an object: {row['event_uid']}")
    return value


def exact_leg(payload: dict[str, Any], event_type: str, trade_id: str) -> dict[str, Any]:
    if event_type == "TRADE_REDUCED":
        leg = payload.get("tradeReduced")
        if not isinstance(leg, dict) or str(leg.get("tradeID") or "") != trade_id:
            raise ValueError(f"partial leg identity mismatch: {trade_id}")
        return leg
    legs = payload.get("tradesClosed") or []
    matches = [leg for leg in legs if str(leg.get("tradeID") or "") == trade_id]
    if len(matches) != 1:
        raise ValueError(f"terminal leg identity mismatch: {trade_id}:{len(matches)}")
    return matches[0]


def load_trade_execution(trades: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    trade_ids = {str(row["trade_id"]) for row in trades}
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in trade_ids)
    rows = connection.execute(
        f"""
        SELECT event_uid, ts_utc, event_type, trade_id, units, price,
               oanda_transaction_id, raw_json
        FROM execution_events
        WHERE trade_id IN ({placeholders})
          AND event_type IN ('ORDER_FILLED','TRADE_REDUCED','TRADE_CLOSED')
        ORDER BY ts_utc, CAST(oanda_transaction_id AS INTEGER), event_uid
        """,
        sorted(trade_ids),
    ).fetchall()
    connection.close()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trade_id"])].append(row)

    execution: dict[str, dict[str, Any]] = {}
    margin_events: list[dict[str, Any]] = []
    for frozen in trades:
        trade_id = str(frozen["trade_id"])
        source = grouped.get(trade_id, [])
        entries = [row for row in source if row["event_type"] == "ORDER_FILLED"]
        terminals = [row for row in source if row["event_type"] == "TRADE_CLOSED"]
        reductions = [row for row in source if row["event_type"] == "TRADE_REDUCED"]
        if len(entries) != 1 or len(terminals) != 1:
            raise ValueError(f"non-unique entry/terminal: {trade_id}:{len(entries)}:{len(terminals)}")
        entry_row = entries[0]
        entry_raw = raw(entry_row)
        opened = entry_raw.get("tradeOpened")
        if not isinstance(opened, dict) or str(opened.get("tradeID") or "") != trade_id:
            raise ValueError(f"entry trade identity mismatch: {trade_id}")
        entry_units = abs(int(opened["units"]))
        gain = float(entry_raw["homeConversionFactors"]["gainBaseHome"]["factor"])
        loss = float(entry_raw["homeConversionFactors"]["lossBaseHome"]["factor"])
        base_home_mid = (gain + loss) / 2.0
        actual_initial_margin = float(opened["initialMarginRequired"])
        margin_per_unit = actual_initial_margin / entry_units
        implied_margin_rate = margin_per_unit / base_home_mid
        if not math.isfinite(implied_margin_rate) or implied_margin_rate <= 0:
            raise ValueError(f"invalid implied margin rate: {trade_id}:{implied_margin_rate}")

        reduction_events: list[dict[str, Any]] = []
        reduced_units = 0
        for row in reductions:
            payload = raw(row)
            leg = exact_leg(payload, "TRADE_REDUCED", trade_id)
            units = abs(int(leg["units"]))
            reduced_units += units
            reduction_events.append({
                "transaction_id": str(payload.get("id") or row["oanda_transaction_id"]),
                "ts_utc": str(payload.get("time") or row["ts_utc"]),
                "units": units,
                "price": float(leg["price"]),
            })
        terminal_raw = raw(terminals[0])
        terminal_leg = exact_leg(terminal_raw, "TRADE_CLOSED", trade_id)
        terminal_units = abs(int(terminal_leg["units"]))
        if reduced_units + terminal_units != entry_units:
            raise ValueError(f"unit conservation failed: {trade_id}:{entry_units}:{reduced_units}:{terminal_units}")

        side = str(frozen["side"])
        if side == "LONG" and int(opened["units"]) <= 0:
            raise ValueError(f"LONG entry units are not positive: {trade_id}")
        if side == "SHORT" and int(opened["units"]) >= 0:
            raise ValueError(f"SHORT entry units are not negative: {trade_id}")
        entry = {
            **frozen,
            "entry_price": float(opened["price"]),
            "entry_units": entry_units,
            "entry_transaction_id": str(entry_raw.get("id") or entry_row["oanda_transaction_id"]),
            "entry_base_home_mid": base_home_mid,
            "entry_implied_margin_rate": implied_margin_rate,
            "entry_required_margin_proxy_jpy": actual_initial_margin,
            "entry_actual_initial_margin_jpy": actual_initial_margin,
            "terminal_price": float(terminal_leg["price"]),
            "terminal_units": terminal_units,
            "terminal_transaction_id": str(terminal_raw.get("id") or terminals[0]["oanda_transaction_id"]),
            "partial_reductions": reduction_events,
        }
        execution[trade_id] = entry
        margin_events.append({"ts_utc": frozen["fill_at_utc"], "transaction_id": entry["entry_transaction_id"], "trade_id": trade_id, "kind": "ENTRY", "delta_units": entry_units})
        for event in reduction_events:
            margin_events.append({"ts_utc": event["ts_utc"], "transaction_id": event["transaction_id"], "trade_id": trade_id, "kind": "PARTIAL_REDUCTION", "delta_units": -event["units"]})
        margin_events.append({"ts_utc": frozen["close_at_utc"], "transaction_id": entry["terminal_transaction_id"], "trade_id": trade_id, "kind": "TERMINAL_CLOSE", "delta_units": -terminal_units})
    return execution, margin_events


def build_margin_timeline(execution: dict[str, dict[str, Any]], events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(str(event["ts_utc"]), str(event["transaction_id"]))].append(event)
    remaining = {trade_id: 0 for trade_id in execution}
    timeline: list[dict[str, Any]] = []
    for (ts_utc, transaction_id), group in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]) if item[0][1].isdigit() else 2**63)):
        for event in group:
            trade_id = str(event["trade_id"])
            remaining[trade_id] += int(event["delta_units"])
            if remaining[trade_id] < 0 or remaining[trade_id] > int(execution[trade_id]["entry_units"]):
                raise ValueError(f"invalid remaining units: {trade_id}:{remaining[trade_id]}")
        open_ids = sorted(trade_id for trade_id, units in remaining.items() if units)
        proxy = sum(
            remaining[trade_id]
            * float(execution[trade_id]["entry_actual_initial_margin_jpy"])
            / int(execution[trade_id]["entry_units"])
            for trade_id in open_ids
        )
        timeline.append({
            "ts_utc": ts_utc,
            "transaction_id": transaction_id,
            "event_count": len(group),
            "event_kinds": sorted({str(item["kind"]) for item in group}),
            "trade_ids": sorted(str(item["trade_id"]) for item in group),
            "cohort_open_trades": len(open_ids),
            "cohort_remaining_units": sum(remaining[trade_id] for trade_id in open_ids),
            "cohort_required_margin_proxy_jpy": proxy,
            "account_available_margin_jpy": None,
            "account_margin_coverage": "MISSING",
            "external_inventory_coverage": "MISSING",
        })
    if any(remaining.values()):
        raise ValueError(f"nonzero terminal remaining units: {remaining}")
    peak = max(timeline, key=lambda row: row["cohort_required_margin_proxy_jpy"])
    return timeline, {
        "events": len(timeline),
        "peak_gross_trade_required_margin_proxy_jpy": peak["cohort_required_margin_proxy_jpy"],
        "peak_at_utc": peak["ts_utc"],
        "peak_open_cohort_trades": peak["cohort_open_trades"],
        "account_available_margin_coverage": 0,
        "external_inventory_coverage": 0,
        "admission": "RELATIVE_GROSS_COHORT_PROXY_ONLY_ACCOUNT_NETTING_MISSING",
    }


def initialize_path_states(execution: dict[str, dict[str, Any]], schedule: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    schedule_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in schedule:
        schedule_by_trade[str(event["trade_id"])].append({**event, "ts_ns": parse_ns(str(event["ts_utc"]))})
    for events in schedule_by_trade.values():
        events.sort(key=lambda row: (row["ts_ns"], int(row["transaction_id"]) if str(row["transaction_id"]).isdigit() else 2**63, str(row["event_kind"])))

    states: dict[str, dict[str, Any]] = {}
    for trade_id, trade in execution.items():
        fill_ns = parse_ns(str(trade["fill_at_utc"]))
        close_ns = parse_ns(str(trade["close_at_utc"]))
        full_start = ceil_bar(fill_ns)
        full_end = (close_ns // BAR_NS) * BAR_NS
        expected = max(0, (full_end - full_start) // BAR_NS)
        states[trade_id] = {
            "trade": trade,
            "fill_ns": fill_ns,
            "close_ns": close_ns,
            "full_start_ns": full_start,
            "full_end_ns": full_end,
            "expected_full_bars": expected,
            "observed_starts": set(),
            "duplicate_bars": 0,
            "incomplete_bars": 0,
            "mfe": None,
            "mae": None,
            "mfe_time": None,
            "mae_time": None,
            "first_touch": None,
            "tp_pre_mae": None,
            "tp_pre_mae_time": None,
            "schedule_change_bars": 0,
            "dual_touch_bars": 0,
            "touch_count": 0,
        }
    return states, schedule_by_trade


def active_protection(events: list[dict[str, Any]], start_ns: int) -> tuple[dict[str, Any] | None, bool]:
    before = [event for event in events if event["ts_ns"] <= start_ns]
    state = before[-1] if before else None
    inside = any(start_ns < event["ts_ns"] < start_ns + BAR_NS and event["event_kind"] in {"CREATE", "CANCEL"} for event in events)
    return state, inside


def update_state(state: dict[str, Any], candle: dict[str, Any], start_ns: int, events: list[dict[str, Any]]) -> None:
    if start_ns in state["observed_starts"]:
        state["duplicate_bars"] += 1
        return
    state["observed_starts"].add(start_ns)
    if candle.get("complete") is not True:
        state["incomplete_bars"] += 1
        return
    trade = state["trade"]
    side = str(trade["side"])
    price = float(trade["entry_price"])
    executable = candle["bid"] if side == "LONG" else candle["ask"]
    high = float(executable["h"])
    low = float(executable["l"])
    favorable = high - price if side == "LONG" else price - low
    adverse = price - low if side == "LONG" else high - price
    if state["mfe"] is None or favorable > state["mfe"]:
        state["mfe"], state["mfe_time"] = favorable, ns_to_utc(start_ns)
    if state["mae"] is None or adverse > state["mae"]:
        state["mae"], state["mae_time"] = adverse, ns_to_utc(start_ns)

    protection, changed_inside = active_protection(events, start_ns)
    if changed_inside:
        state["schedule_change_bars"] += 1
    tp = protection.get("active_tp_price") if protection else None
    sl = protection.get("active_sl_price") if protection else None
    tp_touch = tp is not None and (high >= float(tp) if side == "LONG" else low <= float(tp))
    sl_touch = sl is not None and (low <= float(sl) if side == "LONG" else high >= float(sl))
    if not (tp_touch or sl_touch):
        if state["first_touch"] is None and (state["tp_pre_mae"] is None or adverse > state["tp_pre_mae"]):
            state["tp_pre_mae"], state["tp_pre_mae_time"] = adverse, ns_to_utc(start_ns)
        return
    state["touch_count"] += 1
    if tp_touch and sl_touch:
        state["dual_touch_bars"] += 1
    if state["first_touch"] is None:
        if changed_inside:
            status = "UNRESOLVED_PROTECTION_CHANGE_WITHIN_S5"
        elif tp_touch and sl_touch:
            status = "UNRESOLVED_DUAL_TOUCH_WITHIN_S5"
        elif tp_touch:
            status = "TP_WICK_TOUCH_WITHIN_S5_ORDER_UNRESOLVED"
        else:
            status = "SL_WICK_TOUCH_WITHIN_S5_ORDER_UNRESOLVED"
        state["first_touch"] = {
            "bar_start_utc": ns_to_utc(start_ns),
            "status": status,
            "tp_price": tp,
            "sl_price": sl,
            "tp_touched": tp_touch,
            "sl_touched": sl_touch,
            "touch_bar_adverse_excursion_price": adverse,
            "tp_pre_mae_excludes_touch_bar": True,
        }


def stream_paths(states: dict[str, dict[str, Any]], schedule_by_trade: dict[str, list[dict[str, Any]]], prereg: dict[str, Any]) -> dict[str, Any]:
    pair_states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states.values():
        if state["trade"]["pair"] in prereg["source_boundary"]["pairs"]:
            pair_states[str(state["trade"]["pair"])].append(state)
    source_audit: dict[str, Any] = {}
    for pair, entries in pair_states.items():
        entries.sort(key=lambda state: state["full_start_ns"])
        pair_rows = 0
        duplicates = 0
        seen_global: set[int] = set()
        first_time = None
        last_time = None
        for source in prereg["source_boundary"]["s5_files"][pair]:
            path = REPO / source["path"]
            if sha256_path(path) != source["sha256"]:
                raise ValueError(f"S5 source hash mismatch: {path}")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    candle = json.loads(line)
                    start_ns = parse_ns(str(candle["time"]))
                    if start_ns in seen_global:
                        duplicates += 1
                        continue
                    seen_global.add(start_ns)
                    pair_rows += 1
                    first_time = start_ns if first_time is None else min(first_time, start_ns)
                    last_time = start_ns if last_time is None else max(last_time, start_ns)
                    for state in entries:
                        if state["full_start_ns"] <= start_ns < state["full_end_ns"]:
                            update_state(state, candle, start_ns, schedule_by_trade.get(str(state["trade"]["trade_id"]), []))
        source_audit[pair] = {
            "unique_rows": pair_rows,
            "duplicate_source_rows": duplicates,
            "first_time": ns_to_utc(first_time) if first_time is not None else None,
            "last_time": ns_to_utc(last_time) if last_time is not None else None,
        }
    return source_audit


def finalize_states(states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade_id, state in states.items():
        trade = state["trade"]
        observed = len(state["observed_starts"])
        expected = int(state["expected_full_bars"])
        missing = max(0, expected - observed)
        pip = 0.01 if str(trade["pair"]).endswith("_JPY") else 0.0001
        has_source = str(trade["pair"]) in {"AUD_JPY", "EUR_JPY", "EUR_USD"}
        complete = has_source and expected > 0 and missing == 0 and state["duplicate_bars"] == 0 and state["incomplete_bars"] == 0
        first_touch = state["first_touch"]
        reasons: list[str] = []
        if not has_source:
            reasons.append("PAIR_S5_BID_ASK_SOURCE_MISSING")
        if expected == 0:
            reasons.append("NO_FULL_INTERIOR_S5")
        if missing:
            reasons.append("UNRESOLVED_NO_BAR_ENDPOINTS")
        if state["duplicate_bars"]:
            reasons.append("DUPLICATE_S5")
        if state["incomplete_bars"]:
            reasons.append("INCOMPLETE_S5")
        if first_touch is not None:
            reasons.append(first_touch["status"])
        row = {
            "episode_id": trade["episode_id"],
            "trade_id": trade_id,
            "pair": trade["pair"],
            "side": trade["side"],
            "fill_at_utc": trade["fill_at_utc"],
            "close_at_utc": trade["close_at_utc"],
            "entry_price": trade["entry_price"],
            "terminal_price": trade["terminal_price"],
            "entry_units": trade["entry_units"],
            "corrected_actual_after_cost_net_jpy": trade["corrected_net_jpy"],
            "s5_full_interval_from_utc": ns_to_utc(state["full_start_ns"]),
            "s5_full_interval_to_utc": ns_to_utc(state["full_end_ns"]),
            "expected_full_s5_endpoints": expected if has_source else None,
            "observed_full_s5_endpoints": observed if has_source else None,
            "unresolved_no_bar_endpoint_count": missing if has_source else None,
            "path_complete": complete,
            "path_admission_status": "STRICT_PATH_PASS" if complete else "UNRESOLVED",
            "path_reason_codes": reasons,
            "mfe_observed_lower_bound_price": state["mfe"],
            "mfe_observed_lower_bound_pips": state["mfe"] / pip if state["mfe"] is not None else None,
            "mfe_at_utc": state["mfe_time"],
            "mae_observed_lower_bound_price": state["mae"],
            "mae_observed_lower_bound_pips": state["mae"] / pip if state["mae"] is not None else None,
            "mae_at_utc": state["mae_time"],
            "first_active_protection_touch": first_touch,
            "tp_pre_mae_observed_lower_bound_price": state["tp_pre_mae"],
            "tp_pre_mae_observed_lower_bound_pips": state["tp_pre_mae"] / pip if state["tp_pre_mae"] is not None else None,
            "tp_pre_mae_at_utc": state["tp_pre_mae_time"],
            "tp_pre_mae_strict": None,
            "schedule_change_bar_count": state["schedule_change_bars"],
            "dual_touch_bar_count": state["dual_touch_bars"],
            "protection_touch_bar_count": state["touch_count"],
            "entry_required_margin_proxy_jpy": trade["entry_required_margin_proxy_jpy"],
            "entry_actual_initial_margin_jpy": trade["entry_actual_initial_margin_jpy"],
            "account_available_margin_jpy": None,
            "account_margin_evidence": "MISSING",
            "manual_external_inventory_evidence": "MISSING",
        }
        row["output_sha256"] = sha256_value(row)
        rows.append(row)
    return sorted(rows, key=lambda row: (row["fill_at_utc"], row["trade_id"]))


def window_coverage(path_rows: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    by_episode = {str(row["episode_id"]): row for row in path_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for record in payload["episode_records"]:
        episode_id = str(record["episode_id"])
        key = (str(record["window"]), str(record["split"]), episode_id)
        if key in seen or episode_id not in by_episode:
            continue
        seen.add(key)
        grouped[(key[0], key[1])].append(by_episode[episode_id])
    result: dict[str, Any] = {}
    for (window, split), rows in sorted(grouped.items()):
        result.setdefault(window, {})[split] = {
            "episodes": len(rows),
            "s5_source": sum(row["expected_full_s5_endpoints"] is not None for row in rows),
            "observed_mfe_mae": sum(row["mfe_observed_lower_bound_price"] is not None for row in rows),
            "strict_path_pass": sum(row["path_complete"] for row in rows),
            "first_active_protection_touch": sum(row["first_active_protection_touch"] is not None for row in rows),
        }
    return result


def main() -> int:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    financial = HERE / "financial_oracle_v2.json"
    if sha256_path(financial) != prereg["financial_gate"]["sha256"]:
        raise SystemExit("financial oracle gate hash mismatch")
    if json.loads(financial.read_text())["status"] != "PASS":
        raise SystemExit("financial oracle gate did not pass")
    if sha256_path(TRADES) != prereg["trade_cashflows"]["sha256"]:
        raise SystemExit("trade cashflows hash mismatch")
    if sha256_path(SCHEDULE) != prereg["active_protection"]["events_sha256"]:
        raise SystemExit("active protection schedule hash mismatch")

    trades = read_jsonl(TRADES)
    if len(trades) != 251 or len({row["trade_id"] for row in trades}) != 251:
        raise SystemExit("frozen trade identity contract failed")
    execution, margin_events = load_trade_execution(trades)
    margin_timeline, margin_report = build_margin_timeline(execution, margin_events)
    states, schedule_by_trade = initialize_path_states(execution, read_jsonl(SCHEDULE))
    source_audit = stream_paths(states, schedule_by_trade, prereg)
    path_rows = finalize_states(states)
    split_path = REPO / prereg["split_membership"]["path"]
    if sha256_path(split_path) != prereg["split_membership"]["sha256"]:
        raise SystemExit("split membership hash mismatch")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    reason_counts = Counter(reason for row in path_rows for reason in row["path_reason_codes"])
    report = {
        "contract": "EXECUTABLE_PATH_METRICS_V1",
        "status": "PASS_DIAGNOSTIC_PATH_WITH_STRICT_UNRESOLVED_PRESERVED",
        "episodes": len(path_rows),
        "pairs_with_oanda_s5_bid_ask": sorted(prereg["source_boundary"]["pairs"]),
        "episodes_with_s5_source": sum(row["expected_full_s5_endpoints"] is not None for row in path_rows),
        "episodes_with_observed_mfe_mae": sum(row["mfe_observed_lower_bound_price"] is not None and row["mae_observed_lower_bound_price"] is not None for row in path_rows),
        "strict_path_pass": sum(row["path_complete"] for row in path_rows),
        "strict_path_unresolved": sum(not row["path_complete"] for row in path_rows),
        "episodes_with_first_active_protection_touch": sum(row["first_active_protection_touch"] is not None for row in path_rows),
        "reason_counts": dict(sorted(reason_counts.items())),
        "window_coverage": window_coverage(path_rows, split_payload),
        "source_audit": source_audit,
        "margin": margin_report,
        "financial_gate": {
            "status": "PASS",
            "corrected_64d_validation_net_jpy": json.loads(financial.read_text())["windows"]["QUADRUPLE_64D"]["VALIDATION"]["net_jpy"],
        },
        "truth_boundary": {
            "wick": "Observed OANDA executable-side bid/ask extrema only",
            "fill": "No counterfactual fill is asserted from an OHLC touch",
            "missing": "No-bar intervals remain unresolved and are not forward-filled",
            "margin": "Gross trade-level cohort proxy; broker netting, account availability, and external inventory are missing",
        },
    }
    report["output_sha256"] = sha256_value(report)
    write_jsonl(HERE / "path_metrics_v1.jsonl", path_rows)
    write_jsonl(HERE / "concurrent_margin_timeline_v1.jsonl", margin_timeline)
    write_json(HERE / "path_report_v1.json", report)
    manifest = {
        "contract": "EXECUTABLE_PATH_METRICS_RUN_MANIFEST_V1",
        "preregister_sha256": sha256_path(PREREG),
        "source_sha256": {
            "execution_ledger": sha256_path(DB),
            "financial_oracle": sha256_path(financial),
            "trade_cashflows": sha256_path(TRADES),
            "active_protection_schedule": sha256_path(SCHEDULE),
            "split_membership": sha256_path(split_path),
        },
        "outputs": {
            name: sha256_path(HERE / name)
            for name in ("path_metrics_v1.jsonl", "concurrent_margin_timeline_v1.jsonl", "path_report_v1.json")
        },
    }
    manifest["manifest_sha256"] = sha256_value(manifest)
    write_json(HERE / "path_manifest_v1.json", manifest)
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
