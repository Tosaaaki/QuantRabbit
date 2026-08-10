#!/usr/bin/env python3
"""Build an exact, read-only TP/SL schedule from OANDA transaction truth.

The normalized execution_events.order_id is audit-only for protection creates:
replacement rows contain the replaced order id there. Broker raw_json fields are
the identity source, and every replacement link is validated before admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent
DB = REPO / "data/execution_ledger.db"
EPISODES = REPO / "research/historical_learning_admission/all_entry_episodes_v1.jsonl"
PREREG = ROOT / "preregister_v1.json"

KIND_BY_TYPE = {"TAKE_PROFIT_ORDER": "TP", "STOP_LOSS_ORDER": "SL"}
SEVERE_ISSUES = {
    "CREATE_BEFORE_FILL",
    "CREATE_AFTER_CLOSE",
    "CREATE_RAW_INVALID",
    "CREATE_TRADE_MISMATCH",
    "DUPLICATE_PROTECTION_ORDER_ID",
    "MULTIPLE_ACTIVE_SAME_KIND",
    "REPLACEMENT_ACTIVE_OLD_MISMATCH",
    "REPLACEMENT_CANCEL_MISSING",
    "REPLACEMENT_CANCEL_NEW_MISMATCH",
    "REPLACEMENT_CANCEL_TX_MISMATCH",
    "CANCEL_RAW_INVALID",
    "CANCEL_NONACTIVE_ORDER",
    "TERMINAL_ACTIVE_ORDER_MISMATCH",
    "TRADE_CLOSE_EVENT_MISSING",
}


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
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def tx_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    return (int(text), text) if text.isdigit() else (2**63 - 1, text)


def event_sort_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    tx_number, tx_text = tx_key(row.get("transaction_id"))
    return str(row.get("ts_utc") or ""), tx_number, tx_text, str(row.get("event_kind") or "")


def parse_time(value: str) -> datetime:
    # OANDA timestamps carry nanoseconds while datetime stores microseconds.
    normalized = re.sub(r"(\.\d{6})\d+(Z|[+-]\d{2}:\d{2})$", r"\1\2", value)
    return datetime.fromisoformat(normalized.replace("Z", "+00:00"))


def _raw(row: sqlite3.Row) -> dict[str, Any]:
    try:
        value = json.loads(row["raw_json"])
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_source_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    connection.row_factory = sqlite3.Row
    return connection.execute(
        """
        SELECT event_uid, ts_utc, source, event_type, order_id, trade_id,
               oanda_transaction_id, related_transaction_ids_json, raw_json
        FROM execution_events
        WHERE event_type IN ('PROTECTION_CREATED', 'ORDER_CANCELED', 'TRADE_CLOSED')
        ORDER BY ts_utc, CAST(oanda_transaction_id AS INTEGER), event_uid
        """
    ).fetchall()


def normalize_sources(
    rows: list[sqlite3.Row],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[str],
]:
    creates_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    creates_by_order: dict[str, dict[str, Any]] = {}
    closes_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_issues: list[str] = []

    cancel_rows: list[dict[str, Any]] = []
    for row in rows:
        raw = _raw(row)
        raw_sha = hashlib.sha256(str(row["raw_json"]).encode("utf-8")).hexdigest()
        if row["event_type"] == "PROTECTION_CREATED":
            order_id = str(raw.get("id") or "")
            trade_id = str(raw.get("tradeID") or "")
            protection_type = str(raw.get("type") or "")
            price = raw.get("price")
            if not order_id or not trade_id or protection_type not in KIND_BY_TYPE or price is None:
                global_issues.append(f"CREATE_RAW_INVALID:{row['event_uid']}")
                continue
            if order_id in creates_by_order:
                global_issues.append(f"DUPLICATE_PROTECTION_ORDER_ID:{order_id}")
                continue
            event = {
                "event_kind": "CREATE",
                "event_uid": row["event_uid"],
                "ts_utc": row["ts_utc"],
                "transaction_id": str(raw.get("id") or row["oanda_transaction_id"] or ""),
                "trade_id": trade_id,
                "protection_kind": KIND_BY_TYPE[protection_type],
                "protection_type": protection_type,
                "protection_order_id": order_id,
                "price": float(price),
                "reason": str(raw.get("reason") or ""),
                "replaces_order_id": str(raw.get("replacesOrderID") or "") or None,
                "cancelling_transaction_id": str(raw.get("cancellingTransactionID") or "") or None,
                "normalized_order_id": str(row["order_id"] or "") or None,
                "raw_sha256": raw_sha,
            }
            creates_by_trade[trade_id].append(event)
            creates_by_order[order_id] = event
        elif row["event_type"] == "ORDER_CANCELED":
            order_id = str(raw.get("orderID") or "")
            if not order_id:
                global_issues.append(f"CANCEL_RAW_INVALID:{row['event_uid']}")
                continue
            cancel_rows.append(
                {
                    "event_kind": "CANCEL",
                    "event_uid": row["event_uid"],
                    "ts_utc": row["ts_utc"],
                    "transaction_id": str(raw.get("id") or row["oanda_transaction_id"] or ""),
                    "cancelled_order_id": order_id,
                    "reason": str(raw.get("reason") or ""),
                    "replaced_by_order_id": str(raw.get("replacedByOrderID") or "") or None,
                    "closed_trade_id": str(raw.get("closedTradeID") or "") or None,
                    "raw_sha256": raw_sha,
                }
            )
        else:
            trade_id = str(row["trade_id"] or raw.get("tradeID") or "")
            if not trade_id:
                continue
            closes_by_trade[trade_id].append(
                {
                    "event_kind": "TERMINAL",
                    "event_uid": row["event_uid"],
                    "ts_utc": row["ts_utc"],
                    "transaction_id": str(raw.get("id") or row["oanda_transaction_id"] or ""),
                    "trade_id": trade_id,
                    "terminal_order_id": str(raw.get("orderID") or row["order_id"] or "") or None,
                    "terminal_reason": str(raw.get("reason") or ""),
                    "raw_sha256": raw_sha,
                }
            )

    cancels_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cancel in cancel_rows:
        create = creates_by_order.get(cancel["cancelled_order_id"])
        trade_id = create["trade_id"] if create else cancel.get("closed_trade_id")
        if trade_id:
            event = dict(cancel)
            event["trade_id"] = trade_id
            if create:
                event["protection_kind"] = create["protection_kind"]
            cancels_by_trade[trade_id].append(event)
    return creates_by_trade, creates_by_order, cancels_by_trade, closes_by_trade, global_issues


def build_trade_schedule(
    episode: dict[str, Any],
    creates: list[dict[str, Any]],
    creates_by_order: dict[str, dict[str, Any]],
    cancels: list[dict[str, Any]],
    closes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trade_id = str(episode["trade_id"])
    fill_at = str(episode["fill_at_utc"])
    close_at = str(episode["close_at_utc"])
    issues: list[str] = []
    state: dict[str, dict[str, Any] | None] = {"TP": None, "SL": None}
    output_rows: list[dict[str, Any]] = []

    cancel_by_order = {row["cancelled_order_id"]: row for row in cancels}
    cancel_effects: dict[str, str] = {}
    events = sorted([*creates, *cancels, *closes], key=event_sort_key)
    terminal_seen = False
    terminal_order_kind: str | None = None
    terminal_active_match: bool | None = None

    for event in events:
        event = dict(event)
        event_issues: list[str] = []
        ts_utc = str(event["ts_utc"])
        if event["event_kind"] == "CREATE":
            if event["trade_id"] != trade_id:
                event_issues.append("CREATE_TRADE_MISMATCH")
            if ts_utc < fill_at:
                event_issues.append("CREATE_BEFORE_FILL")
            if ts_utc > close_at:
                event_issues.append("CREATE_AFTER_CLOSE")
            kind = event["protection_kind"]
            old = event.get("replaces_order_id")
            if event.get("reason") == "REPLACEMENT" or old:
                cancel = cancel_by_order.get(str(old or ""))
                old_was_active = bool(
                    old
                    and (
                        (state[kind] is not None and state[kind]["protection_order_id"] == old)
                        or cancel_effects.get(str(old)) == "DEACTIVATE"
                    )
                )
                if not old_was_active:
                    event_issues.append("REPLACEMENT_ACTIVE_OLD_MISMATCH")
                if cancel is None:
                    event_issues.append("REPLACEMENT_CANCEL_MISSING")
                else:
                    if cancel.get("replaced_by_order_id") != event["protection_order_id"]:
                        event_issues.append("REPLACEMENT_CANCEL_NEW_MISMATCH")
                    if event.get("cancelling_transaction_id") != cancel.get("transaction_id"):
                        event_issues.append("REPLACEMENT_CANCEL_TX_MISMATCH")
            elif state[kind] is not None and state[kind]["protection_order_id"] != event["protection_order_id"]:
                event_issues.append("MULTIPLE_ACTIVE_SAME_KIND")

            if not any(issue in SEVERE_ISSUES for issue in event_issues):
                state[kind] = {
                    "protection_order_id": event["protection_order_id"],
                    "price": event["price"],
                    "created_at_utc": ts_utc,
                    "transaction_id": event["transaction_id"],
                }
        elif event["event_kind"] == "CANCEL":
            kind = event.get("protection_kind")
            if event.get("reason") == "LINKED_TRADE_CLOSED" and ts_utc >= close_at:
                event["effect"] = "POST_TERMINAL_LINKED_CANCEL"
            elif kind in state and state[kind] is not None and state[kind]["protection_order_id"] == event["cancelled_order_id"]:
                state[kind] = None
                event["effect"] = "DEACTIVATE"
                cancel_effects[event["cancelled_order_id"]] = "DEACTIVATE"
            else:
                event_issues.append("CANCEL_NONACTIVE_ORDER")
        else:
            terminal_seen = True
            terminal_id = event.get("terminal_order_id")
            create = creates_by_order.get(str(terminal_id or ""))
            if create and create.get("trade_id") == trade_id:
                terminal_order_kind = str(create["protection_kind"])
                active = state[terminal_order_kind]
                terminal_active_match = bool(active and active["protection_order_id"] == terminal_id)
                if not terminal_active_match:
                    event_issues.append("TERMINAL_ACTIVE_ORDER_MISMATCH")
            else:
                terminal_active_match = None

        issues.extend(event_issues)
        row = {
            "episode_id": episode["episode_id"],
            "trade_id": trade_id,
            "pair": episode["pair"],
            "side": episode["side"],
            "fill_at_utc": fill_at,
            "close_at_utc": close_at,
            **event,
            "active_tp_order_id": state["TP"]["protection_order_id"] if state["TP"] else None,
            "active_tp_price": state["TP"]["price"] if state["TP"] else None,
            "active_sl_order_id": state["SL"]["protection_order_id"] if state["SL"] else None,
            "active_sl_price": state["SL"]["price"] if state["SL"] else None,
            "event_issues": sorted(set(event_issues)),
            "evidence_kind": "ACTUAL" if not event_issues else "MISSING",
        }
        row["output_sha256"] = sha256_value({key: value for key, value in row.items() if key != "output_sha256"})
        output_rows.append(row)

    if not terminal_seen:
        issues.append("TRADE_CLOSE_EVENT_MISSING")
    unique_issues = sorted(set(issues))
    severe = sorted(issue for issue in unique_issues if issue in SEVERE_ISSUES)
    summary = {
        "episode_id": episode["episode_id"],
        "trade_id": trade_id,
        "pair": episode["pair"],
        "side": episode["side"],
        "fill_at_utc": fill_at,
        "close_at_utc": close_at,
        "feature_at_utc": episode["feature_at_utc"],
        "create_count": len(creates),
        "cancel_count": len(cancels),
        "replacement_count": sum(1 for row in creates if row.get("reason") == "REPLACEMENT"),
        "tp_create_count": sum(1 for row in creates if row.get("protection_kind") == "TP"),
        "sl_create_count": sum(1 for row in creates if row.get("protection_kind") == "SL"),
        "terminal_order_kind": terminal_order_kind,
        "terminal_active_match": terminal_active_match,
        "strict_schedule_eligible": not severe,
        "issues": unique_issues,
        "severe_issues": severe,
    }
    summary["output_sha256"] = sha256_value({key: value for key, value in summary.items() if key != "output_sha256"})
    return output_rows, summary


def window_coverage(summaries: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    embargo = timedelta(hours=1)
    for window in windows:
        selected = sorted(
            [
                row
                for row in summaries
                if window["from_utc"] <= row["feature_at_utc"] <= window["to_utc"]
            ],
            key=lambda row: parse_time(row["feature_at_utc"]),
        )
        if len(selected) < 2:
            train: list[dict[str, Any]] = []
            validation: list[dict[str, Any]] = []
            purged = 0
        else:
            cut = max(1, math.floor(len(selected) * 0.60))
            raw_train = selected[:cut]
            validation = selected[cut:]
            validation_start = parse_time(validation[0]["feature_at_utc"])
            train = [row for row in raw_train if parse_time(row["close_at_utc"]) < validation_start - embargo]
            purged = len(raw_train) - len(train)
        result[window["id"]] = {
            "from_utc": window["from_utc"],
            "to_utc": window["to_utc"],
            "episodes": len(selected),
            "train": len(train),
            "validation": len(validation),
            "purged_train": purged,
            "strict_train": sum(row["strict_schedule_eligible"] for row in train),
            "strict_validation": sum(row["strict_schedule_eligible"] for row in validation),
        }
    return result


def main() -> int:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    expected_db = prereg["frozen_inputs"]["execution_ledger"]["sha256"]
    expected_episodes = prereg["frozen_inputs"]["episodes"]["sha256"]
    actual_hashes = {"execution_ledger": sha256_path(DB), "episodes": sha256_path(EPISODES)}
    if actual_hashes != {"execution_ledger": expected_db, "episodes": expected_episodes}:
        raise SystemExit(f"frozen input hash mismatch: {actual_hashes}")

    episodes = [row for row in read_jsonl(EPISODES) if row.get("label_status") == "ACTUAL_AFTER_COST"]
    trade_ids = [str(row.get("trade_id") or "") for row in episodes]
    if len(episodes) != 251 or not all(trade_ids) or len(set(trade_ids)) != len(trade_ids):
        raise SystemExit("frozen 251 trade identity contract failed")

    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        source_rows = load_source_rows(connection)
    finally:
        connection.close()
    creates_by_trade, creates_by_order, cancels_by_trade, closes_by_trade, global_issues = normalize_sources(source_rows)

    schedule_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for episode in sorted(episodes, key=lambda row: (row["fill_at_utc"], str(row["trade_id"]))):
        trade_id = str(episode["trade_id"])
        rows, summary = build_trade_schedule(
            episode,
            creates_by_trade.get(trade_id, []),
            creates_by_order,
            cancels_by_trade.get(trade_id, []),
            closes_by_trade.get(trade_id, []),
        )
        schedule_rows.extend(rows)
        summaries.append(summary)

    issue_counts = Counter(issue for summary in summaries for issue in summary["issues"])
    coverage = {
        "contract": "ACTIVE_PROTECTION_SCHEDULE_V1",
        "episodes": len(summaries),
        "episodes_with_any_protection": sum(row["create_count"] > 0 for row in summaries),
        "episodes_with_tp": sum(row["tp_create_count"] > 0 for row in summaries),
        "episodes_with_sl": sum(row["sl_create_count"] > 0 for row in summaries),
        "episodes_with_replacement": sum(row["replacement_count"] > 0 for row in summaries),
        "strict_schedule_eligible": sum(row["strict_schedule_eligible"] for row in summaries),
        "strict_schedule_ineligible": sum(not row["strict_schedule_eligible"] for row in summaries),
        "terminal_protection_order": sum(row["terminal_order_kind"] is not None for row in summaries),
        "terminal_protection_active_match": sum(row["terminal_active_match"] is True for row in summaries),
        "terminal_nonprotection_or_missing": sum(row["terminal_active_match"] is None for row in summaries),
        "source_create_events": sum(row["create_count"] for row in summaries),
        "source_cancel_events": sum(row["cancel_count"] for row in summaries),
        "source_replacement_events": sum(row["replacement_count"] for row in summaries),
        "cancel_reason_counts": dict(
            sorted(
                Counter(
                    event["reason"]
                    for events in cancels_by_trade.values()
                    for event in events
                    if event["trade_id"] in set(trade_ids)
                ).items()
            )
        ),
        "normalized_create_order_id_mismatch": sum(
            event["normalized_order_id"] != event["protection_order_id"]
            for events in creates_by_trade.values()
            for event in events
            if event["trade_id"] in set(trade_ids)
        ),
        "issue_counts": dict(sorted(issue_counts.items())),
        "global_issues": sorted(global_issues),
        "window_coverage": window_coverage(summaries, prereg["windows"]),
        "admission_status": "PASS" if not global_issues and all(row["strict_schedule_eligible"] for row in summaries) else "HOLD",
    }
    coverage["output_sha256"] = sha256_value({key: value for key, value in coverage.items() if key != "output_sha256"})

    write_jsonl(ROOT / "schedule_events_v1.jsonl", schedule_rows)
    write_jsonl(ROOT / "trade_summaries_v1.jsonl", summaries)
    write_json(ROOT / "coverage_report_v1.json", coverage)
    manifest = {
        "contract": "ACTIVE_PROTECTION_SCHEDULE_RUN_MANIFEST_V1",
        "source_sha256": actual_hashes,
        "preregister_sha256": sha256_path(PREREG),
        "outputs": {
            name: sha256_path(ROOT / name)
            for name in ("schedule_events_v1.jsonl", "trade_summaries_v1.jsonl", "coverage_report_v1.json")
        },
    }
    manifest["manifest_sha256"] = sha256_value({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    write_json(ROOT / "run_manifest_v1.json", manifest)
    print(canonical_json(coverage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
