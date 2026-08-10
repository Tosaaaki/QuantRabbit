#!/usr/bin/env python3
"""Independent direct-SQL oracle for ACTIVE_PROTECTION_SCHEDULE_V1 outputs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent
DB = REPO / "data/execution_ledger.db"
EPISODES = REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    episodes = [row for row in read_jsonl(EPISODES) if row.get("label_status") == "ACTUAL_AFTER_COST"]
    summaries = {row["trade_id"]: row for row in read_jsonl(ROOT / "trade_summaries_v1.jsonl")}
    coverage = json.loads((ROOT / "coverage_report_v1.json").read_text(encoding="utf-8"))
    trade_ids = {str(row["trade_id"]) for row in episodes}

    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT event_type, order_id, trade_id, oanda_transaction_id, raw_json
            FROM execution_events
            WHERE event_type IN ('PROTECTION_CREATED', 'ORDER_CANCELED', 'TRADE_CLOSED')
            """
        ).fetchall()
    finally:
        connection.close()

    creates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order_to_trade: dict[str, str] = {}
    closes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cancels: list[dict[str, Any]] = []
    duplicate_order_ids: list[str] = []
    normalized_mismatch = 0
    for row in rows:
        raw = json.loads(row["raw_json"])
        if row["event_type"] == "PROTECTION_CREATED":
            trade_id = str(raw["tradeID"])
            if trade_id not in trade_ids:
                continue
            order_id = str(raw["id"])
            if order_id in order_to_trade:
                duplicate_order_ids.append(order_id)
            order_to_trade[order_id] = trade_id
            creates[trade_id].append(raw)
            normalized_mismatch += str(row["order_id"] or "") != order_id
        elif row["event_type"] == "ORDER_CANCELED":
            cancels.append(raw)
        elif str(row["trade_id"] or "") in trade_ids:
            closes[str(row["trade_id"])].append(raw)

    cancels_by_trade: Counter[str] = Counter()
    replacement_links_ok = 0
    replacement_links_bad = 0
    create_by_id = {str(raw["id"]): raw for values in creates.values() for raw in values}
    cancel_by_old = {str(raw.get("orderID") or ""): raw for raw in cancels}
    for old_id, raw_cancel in cancel_by_old.items():
        trade_id = order_to_trade.get(old_id) or str(raw_cancel.get("closedTradeID") or "")
        if trade_id in trade_ids:
            cancels_by_trade[trade_id] += 1
    for values in creates.values():
        for raw_create in values:
            if raw_create.get("reason") != "REPLACEMENT":
                continue
            old_id = str(raw_create.get("replacesOrderID") or "")
            cancel = cancel_by_old.get(old_id)
            valid = bool(
                cancel
                and str(cancel.get("replacedByOrderID") or "") == str(raw_create["id"])
                and str(raw_create.get("cancellingTransactionID") or "") == str(cancel.get("id") or "")
            )
            if valid:
                replacement_links_ok += 1
            else:
                replacement_links_bad += 1

    checks = {
        "episodes_251": len(episodes) == 251,
        "summary_trade_identity_exact": set(summaries) == trade_ids,
        "duplicate_raw_protection_order_ids_zero": not duplicate_order_ids,
        "source_create_count_matches": coverage["source_create_events"] == sum(map(len, creates.values())),
        "source_cancel_count_matches": coverage["source_cancel_events"] == sum(cancels_by_trade.values()),
        "replacement_count_matches": coverage["source_replacement_events"] == replacement_links_ok + replacement_links_bad,
        "replacement_bad_links_zero": replacement_links_bad == 0,
        "normalized_mismatch_matches": coverage["normalized_create_order_id_mismatch"] == normalized_mismatch,
        "per_trade_create_counts_match": all(summaries[trade]["create_count"] == len(creates.get(trade, [])) for trade in trade_ids),
        "per_trade_cancel_counts_match": all(summaries[trade]["cancel_count"] == cancels_by_trade[trade] for trade in trade_ids),
        "per_trade_close_present": all(len(closes.get(trade, [])) >= 1 for trade in trade_ids),
        "source_hash_db_matches_preregister": sha256_path(DB) == "545feb1d62410904bf3f86b4290986caf3932546ef858abec6c3eb27a58b38eb",
        "source_hash_episodes_matches_preregister": sha256_path(EPISODES) == "efcf6b0fb675050d6a08efc0119065e0874e50e1c51373a0c0fb61bb6ebd815e",
    }
    result = {
        "contract": "ACTIVE_PROTECTION_SCHEDULE_INDEPENDENT_ORACLE_V1",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "replacement_links_ok": replacement_links_ok,
        "replacement_links_bad": replacement_links_bad,
        "normalized_create_order_id_mismatch": normalized_mismatch,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    (ROOT / "independent_oracle_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
