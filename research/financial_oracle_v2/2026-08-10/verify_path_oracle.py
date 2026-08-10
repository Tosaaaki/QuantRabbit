#!/usr/bin/env python3
"""Independent raw-data oracle for saved path and gross-margin diagnostics.

This verifier intentionally does not import build_path_metrics.py.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DB = REPO / "data/execution_ledger.db"
NS = 1_000_000_000
BAR = 5 * NS


def parse_ns(value: str) -> int:
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z", value)
    if not match:
        raise ValueError(value)
    whole = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
    return int(whole.timestamp()) * NS + int((match.group(2) or "")[:9].ljust(9, "0"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminal_leg(payload: dict[str, Any], trade_id: str) -> dict[str, Any]:
    matches = [leg for leg in payload.get("tradesClosed") or [] if str(leg.get("tradeID")) == trade_id]
    if len(matches) != 1:
        raise ValueError(f"terminal leg count: {trade_id}:{len(matches)}")
    return matches[0]


def main() -> int:
    prereg = json.loads((HERE / "path_preregister_v1.json").read_text())
    saved = read_jsonl(HERE / "path_metrics_v1.jsonl")
    by_trade = {str(row["trade_id"]): row for row in saved}
    checks: list[dict[str, Any]] = []

    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    ids = sorted(by_trade)
    placeholders = ",".join("?" for _ in ids)
    rows = connection.execute(
        f"SELECT ts_utc,event_type,trade_id,oanda_transaction_id,raw_json FROM execution_events WHERE trade_id IN ({placeholders}) AND event_type IN ('ORDER_FILLED','TRADE_REDUCED','TRADE_CLOSED') ORDER BY ts_utc,CAST(oanda_transaction_id AS INTEGER)",
        ids,
    ).fetchall()
    connection.close()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trade_id"])].append(row)

    unit_events: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    initial_margin: dict[str, float] = {}
    entry_units: dict[str, int] = {}
    for trade_id, source in grouped.items():
        entries = [row for row in source if row["event_type"] == "ORDER_FILLED"]
        terminals = [row for row in source if row["event_type"] == "TRADE_CLOSED"]
        if len(entries) != 1 or len(terminals) != 1:
            raise ValueError(f"nonunique entry/terminal {trade_id}")
        entry_payload = json.loads(entries[0]["raw_json"])
        opened = entry_payload["tradeOpened"]
        units = abs(int(opened["units"]))
        margin = float(opened["initialMarginRequired"])
        entry_units[trade_id] = units
        initial_margin[trade_id] = margin
        unit_events[(str(entries[0]["ts_utc"]), str(entries[0]["oanda_transaction_id"]))].append((trade_id, units))
        reduced = 0
        for row in source:
            payload = json.loads(row["raw_json"])
            if row["event_type"] == "TRADE_REDUCED":
                leg = payload["tradeReduced"]
                amount = abs(int(leg["units"]))
                reduced += amount
                unit_events[(str(row["ts_utc"]), str(row["oanda_transaction_id"]))].append((trade_id, -amount))
        terminal_payload = json.loads(terminals[0]["raw_json"])
        terminal = terminal_leg(terminal_payload, trade_id)
        terminal_units = abs(int(terminal["units"]))
        if reduced + terminal_units != units:
            raise ValueError(f"unit conservation {trade_id}")
        unit_events[(str(terminals[0]["ts_utc"]), str(terminals[0]["oanda_transaction_id"]))].append((trade_id, -terminal_units))
        current = by_trade[trade_id]
        checks.append({
            "check": f"entry_receipt:{trade_id}",
            "pass": current["entry_price"] == float(opened["price"])
            and current["entry_units"] == units
            and current["entry_actual_initial_margin_jpy"] == margin,
        })

    remaining = {trade_id: 0 for trade_id in by_trade}
    peak = 0.0
    peak_open = 0
    for _, changes in sorted(unit_events.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        for trade_id, delta in changes:
            remaining[trade_id] += delta
        gross = sum(remaining[trade_id] * initial_margin[trade_id] / entry_units[trade_id] for trade_id in remaining)
        if gross > peak:
            peak = gross
            peak_open = sum(value > 0 for value in remaining.values())
    report = json.loads((HERE / "path_report_v1.json").read_text())
    checks.extend([
        {"check": "all_units_close_to_zero", "pass": all(value == 0 for value in remaining.values())},
        {"check": "gross_margin_peak", "pass": abs(peak - report["margin"]["peak_gross_trade_required_margin_proxy_jpy"]) < 1e-7},
        {"check": "gross_margin_peak_open_count", "pass": peak_open == report["margin"]["peak_open_cohort_trades"]},
    ])

    sample = [row for row in saved if row["expected_full_s5_endpoints"] not in (None, 0)][:12]
    sample_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_calc: dict[str, dict[str, Any]] = {}
    for row in sample:
        fill = parse_ns(row["fill_at_utc"])
        close = parse_ns(row["close_at_utc"])
        start = ((fill + BAR - 1) // BAR) * BAR
        end = (close // BAR) * BAR
        raw_calc[row["trade_id"]] = {"start": start, "end": end, "seen": set(), "mfe": None, "mae": None}
        sample_by_pair[row["pair"]].append(row)
        checks.append({"check": f"expected_grid:{row['trade_id']}", "pass": row["expected_full_s5_endpoints"] == max(0, (end - start) // BAR)})

    for pair, sample_rows in sample_by_pair.items():
        globally_seen: set[int] = set()
        for source in prereg["source_boundary"]["s5_files"][pair]:
            path = REPO / source["path"]
            checks.append({"check": f"source_hash:{path.name}", "pass": sha256_path(path) == source["sha256"]})
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    candle = json.loads(line)
                    stamp = parse_ns(candle["time"])
                    if stamp in globally_seen:
                        continue
                    globally_seen.add(stamp)
                    for row in sample_rows:
                        calc = raw_calc[row["trade_id"]]
                        if not calc["start"] <= stamp < calc["end"]:
                            continue
                        calc["seen"].add(stamp)
                        quote = candle["bid"] if row["side"] == "LONG" else candle["ask"]
                        high, low, entry = float(quote["h"]), float(quote["l"]), float(row["entry_price"])
                        mfe = high - entry if row["side"] == "LONG" else entry - low
                        mae = entry - low if row["side"] == "LONG" else high - entry
                        calc["mfe"] = mfe if calc["mfe"] is None else max(calc["mfe"], mfe)
                        calc["mae"] = mae if calc["mae"] is None else max(calc["mae"], mae)
    for row in sample:
        calc = raw_calc[row["trade_id"]]
        checks.extend([
            {"check": f"observed_count:{row['trade_id']}", "pass": len(calc["seen"]) == row["observed_full_s5_endpoints"]},
            {"check": f"mfe:{row['trade_id']}", "pass": abs(calc["mfe"] - row["mfe_observed_lower_bound_price"]) < 1e-12},
            {"check": f"mae:{row['trade_id']}", "pass": abs(calc["mae"] - row["mae_observed_lower_bound_price"]) < 1e-12},
        ])

    failed = [row for row in checks if not row["pass"]]
    result = {
        "contract": "EXECUTABLE_PATH_INDEPENDENT_ORACLE_V1",
        "status": "PASS" if not failed else "FAIL",
        "checks": len(checks),
        "passed": len(checks) - len(failed),
        "failed": failed,
        "raw_entry_receipts": len(grouped),
        "raw_path_sample": len(sample),
        "independent_gross_margin_peak_jpy": peak,
        "holdout_used": False,
    }
    (HERE / "path_independent_oracle_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
