from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from quant_rabbit.models import AccountSummary
from quant_rabbit.paths import DEFAULT_EXECUTION_LEDGER_DB, DEFAULT_EXECUTION_LEDGER_REPORT


class TransactionClient(Protocol):
    def account_summary(self, *, now_utc: datetime | None = None) -> AccountSummary: ...

    def transactions_since_id(self, transaction_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExecutionLedgerSummary:
    db_path: Path
    report_path: Path
    status: str
    transactions_seen: int = 0
    transactions_inserted: int = 0
    events_inserted: int = 0
    gateway_receipts_inserted: int = 0
    last_transaction_id: str | None = None
    baseline_transaction_id: str | None = None


class ExecutionLedger:
    """Append-only execution ledger backed by OANDA transaction truth.

    OANDA remains the source of truth. This SQLite database is a durable local
    audit index: it stores every fetched transaction raw, plus normalized
    execution events that are convenient for P/L review and post-trade learning.
    Unknown transaction types are still stored raw and receive a generic event
    so parser gaps are visible instead of silently dropping broker evidence.
    """

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        report_path: Path = DEFAULT_EXECUTION_LEDGER_REPORT,
    ) -> None:
        self.db_path = db_path
        self.report_path = report_path

    def sync_oanda_transactions(
        self,
        client: TransactionClient,
        *,
        since_transaction_id: str | None = None,
    ) -> ExecutionLedgerSummary:
        self._init_db()
        now = _now()
        baseline = None
        with self._connect() as conn:
            start_id = since_transaction_id or _get_state(conn, "last_oanda_transaction_id")
            if not start_id:
                account = client.account_summary(now_utc=datetime.now(timezone.utc))
                baseline = account.last_transaction_id
                if baseline:
                    _set_state(conn, "last_oanda_transaction_id", baseline, now)
                summary = ExecutionLedgerSummary(
                    db_path=self.db_path,
                    report_path=self.report_path,
                    status="BASELINED",
                    baseline_transaction_id=baseline,
                    last_transaction_id=baseline,
                )
                self._write_report(summary)
                return summary

            payload = client.transactions_since_id(str(start_id))
            transactions = [tx for tx in payload.get("transactions", []) or [] if isinstance(tx, dict)]
            inserted_transactions = 0
            inserted_events = 0
            for transaction in transactions:
                if _insert_transaction(conn, transaction, now):
                    inserted_transactions += 1
                for event in _events_from_transaction(transaction, now):
                    if _insert_event(conn, event):
                        inserted_events += 1
                _record_entry_thesis_for_fill(transaction, data_root=self.db_path.parent)
            last_transaction_id = str(payload.get("lastTransactionID") or start_id)
            _set_state(conn, "last_oanda_transaction_id", last_transaction_id, now)

        summary = ExecutionLedgerSummary(
            db_path=self.db_path,
            report_path=self.report_path,
            status="SYNCED",
            transactions_seen=len(transactions),
            transactions_inserted=inserted_transactions,
            events_inserted=inserted_events,
            last_transaction_id=last_transaction_id,
        )
        self._write_report(summary)
        return summary

    def record_gateway_receipt(self, *, kind: str, receipt_path: Path) -> ExecutionLedgerSummary:
        self._init_db()
        if not receipt_path.exists():
            summary = ExecutionLedgerSummary(
                db_path=self.db_path,
                report_path=self.report_path,
                status="NO_RECEIPT",
            )
            self._write_report(summary)
            return summary
        payload = json.loads(receipt_path.read_text())
        now = _now()
        with self._connect() as conn:
            receipt_inserted = _insert_gateway_receipt(conn, kind=kind, path=receipt_path, payload=payload, now=now)
            inserted_events = 0
            for event in _events_from_gateway_receipt(kind=kind, payload=payload, now=now):
                if _insert_event(conn, event):
                    inserted_events += 1
        summary = ExecutionLedgerSummary(
            db_path=self.db_path,
            report_path=self.report_path,
            status="RECORDED",
            gateway_receipts_inserted=1 if receipt_inserted else 0,
            events_inserted=inserted_events,
        )
        self._write_report(summary)
        return summary

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oanda_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    time_utc TEXT,
                    batch_id TEXT,
                    request_id TEXT,
                    raw_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gateway_receipts (
                    receipt_uid TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT,
                    sent INTEGER NOT NULL,
                    lane_id TEXT,
                    lane_ids_json TEXT NOT NULL,
                    path TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS execution_events (
                    event_uid TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    lane_id TEXT,
                    order_id TEXT,
                    trade_id TEXT,
                    client_order_id TEXT,
                    pair TEXT,
                    side TEXT,
                    units INTEGER,
                    price REAL,
                    tp REAL,
                    sl REAL,
                    realized_pl_jpy REAL,
                    financing_jpy REAL,
                    exit_reason TEXT,
                    oanda_transaction_id TEXT,
                    related_transaction_ids_json TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_execution_events_ts ON execution_events(ts_utc);
                CREATE INDEX IF NOT EXISTS idx_execution_events_pair_side ON execution_events(pair, side);
                CREATE INDEX IF NOT EXISTS idx_execution_events_trade_id ON execution_events(trade_id);
                CREATE INDEX IF NOT EXISTS idx_execution_events_order_id ON execution_events(order_id);
                CREATE INDEX IF NOT EXISTS idx_execution_events_type ON execution_events(event_type);
                """
            )
            _backfill_legacy_lane_ids(conn)
            _backfill_legacy_trade_close_ids(conn)

    def _write_report(self, summary: ExecutionLedgerSummary) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Execution Ledger Report",
            "",
            f"- Generated at UTC: `{_now()}`",
            f"- Status: `{summary.status}`",
            f"- DB: `{summary.db_path}`",
            f"- Transactions seen: `{summary.transactions_seen}`",
            f"- Transactions inserted: `{summary.transactions_inserted}`",
            f"- Events inserted: `{summary.events_inserted}`",
            f"- Gateway receipts inserted: `{summary.gateway_receipts_inserted}`",
            f"- Baseline transaction id: `{summary.baseline_transaction_id}`",
            f"- Last transaction id: `{summary.last_transaction_id}`",
            "",
            "## Contract",
            "",
            "- OANDA transactions are broker truth; the database is an append-only local audit index.",
            "- Unknown transaction types are stored raw and recorded as generic `OANDA_TRANSACTION` events.",
            "- First run without `--since-transaction-id` baselines at the current broker `lastTransactionID`; use an explicit since id for historical backfill.",
        ]
        self.report_path.write_text("\n".join(lines) + "\n")


def _events_from_transaction(transaction: dict[str, Any], inserted_at_utc: str) -> list[dict[str, Any]]:
    transaction_id = str(transaction.get("id") or "")
    transaction_type = str(transaction.get("type") or "UNKNOWN")
    ts = str(transaction.get("time") or inserted_at_utc)
    events: list[dict[str, Any]] = []
    if transaction_type in {"MARKET_ORDER", "LIMIT_ORDER", "STOP_ORDER", "MARKET_IF_TOUCHED_ORDER"}:
        events.append(_event(transaction, inserted_at_utc, event_type="ORDER_ACCEPTED", uid_tail=_order_uid_tail(transaction)))
    elif transaction_type.endswith("_REJECT") or transaction_type == "ORDER_CANCEL_REJECT":
        events.append(_event(transaction, inserted_at_utc, event_type="ORDER_REJECTED", uid_tail=transaction_type))
    elif transaction_type == "ORDER_CANCEL":
        events.append(_event(transaction, inserted_at_utc, event_type="ORDER_CANCELED", uid_tail=_order_uid_tail(transaction)))
    elif transaction_type in {"TAKE_PROFIT_ORDER", "STOP_LOSS_ORDER", "TRAILING_STOP_LOSS_ORDER"}:
        events.append(_event(transaction, inserted_at_utc, event_type="PROTECTION_CREATED", uid_tail=_order_uid_tail(transaction)))
    elif transaction_type == "ORDER_FILL":
        events.extend(_events_from_order_fill(transaction, inserted_at_utc))

    if not events:
        events.append(
            {
                **_event(transaction, inserted_at_utc, event_type="OANDA_TRANSACTION", uid_tail=transaction_type),
                "ts_utc": ts,
            }
        )
    return events


def _events_from_order_fill(transaction: dict[str, Any], inserted_at_utc: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    opened = transaction.get("tradeOpened")
    if isinstance(opened, dict):
        events.append(
            _event(
                transaction,
                inserted_at_utc,
                event_type="ORDER_FILLED",
                uid_tail=f"opened:{opened.get('tradeID') or transaction.get('orderID') or ''}",
                trade_id=str(opened.get("tradeID") or ""),
                units=_int(opened.get("units")) or _abs_int(transaction.get("units")),
                price=_float(opened.get("price")) or _float(transaction.get("price")),
                realized_pl_jpy=None,
            )
        )
    reduced = transaction.get("tradeReduced")
    if isinstance(reduced, dict):
        events.append(
            _event(
                transaction,
                inserted_at_utc,
                event_type="TRADE_REDUCED",
                uid_tail=f"reduced:{reduced.get('tradeID') or transaction.get('orderID') or ''}",
                trade_id=str(reduced.get("tradeID") or ""),
                side=_closed_trade_side(transaction),
                units=_abs_int(reduced.get("units")) or _abs_int(transaction.get("units")),
                price=_float(reduced.get("price")) or _float(transaction.get("price")),
                realized_pl_jpy=_float(reduced.get("realizedPL")) or _float(transaction.get("pl")),
            )
        )
    closed_items = transaction.get("tradesClosed")
    if isinstance(closed_items, list):
        for index, closed in enumerate(closed_items):
            if not isinstance(closed, dict):
                continue
            events.append(
                _event(
                    transaction,
                    inserted_at_utc,
                    event_type="TRADE_CLOSED",
                    uid_tail=f"closed:{closed.get('tradeID') or index}",
                    trade_id=str(closed.get("tradeID") or ""),
                    side=_closed_trade_side(transaction),
                    units=_abs_int(closed.get("units")) or _abs_int(transaction.get("units")),
                    price=_float(closed.get("price")) or _float(transaction.get("price")),
                    realized_pl_jpy=_float(closed.get("realizedPL")) or _float(transaction.get("pl")),
                    exit_reason=str(transaction.get("reason") or ""),
                )
            )
    if not events:
        events.append(_event(transaction, inserted_at_utc, event_type="ORDER_FILLED", uid_tail=_order_uid_tail(transaction)))
    return events


def _events_from_gateway_receipt(kind: str, payload: dict[str, Any], now: str) -> list[dict[str, Any]]:
    if kind == "live_order":
        orders = payload.get("orders")
        if isinstance(orders, list):
            return [
                _gateway_live_order_event(item, now=now, index=index)
                for index, item in enumerate(orders)
                if isinstance(item, dict)
            ]
        return [_gateway_live_order_event(payload, now=now, index=0)]
    if kind == "position_execution":
        return [
            _gateway_position_event(action, payload=payload, now=now, index=index)
            for index, action in enumerate(payload.get("actions", []) or [])
            if isinstance(action, dict)
        ]
    return []


def _record_entry_thesis_for_fill(transaction: dict[str, Any], *, data_root: Path) -> None:
    if str(transaction.get("type") or "") != "ORDER_FILL":
        return
    if not isinstance(transaction.get("tradeOpened"), dict):
        return
    try:
        from quant_rabbit.strategy.entry_thesis_ledger import record_entry_thesis_from_order_fill

        record_entry_thesis_from_order_fill(transaction=transaction, data_root=data_root)
    except Exception:
        return


def _gateway_live_order_event(payload: dict[str, Any], *, now: str, index: int) -> dict[str, Any]:
    order = payload.get("order_request") if isinstance(payload.get("order_request"), dict) else {}
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    status = str(payload.get("status") or "").upper()
    event_type = "GATEWAY_ORDER_STAGED"
    if payload.get("sent"):
        event_type = "GATEWAY_ORDER_SENT"
    elif status == "BLOCKED":
        event_type = "GATEWAY_ORDER_BLOCKED"
    elif status == "NO_ACTION" or not order:
        event_type = "GATEWAY_ORDER_NO_ACTION"
    lane_id = _text(payload.get("lane_id"))
    order_id = _response_order_id(response)
    trade_id = _response_trade_id(response)
    uid = f"gateway:live_order:{payload.get('generated_at_utc') or now}:{index}:{event_type}:{lane_id or order_id or ''}"
    units = _int(order.get("units"))
    return {
        "event_uid": uid,
        "ts_utc": str(payload.get("generated_at_utc") or now),
        "source": "gateway",
        "event_type": event_type,
        "lane_id": lane_id,
        "order_id": order_id,
        "trade_id": trade_id,
        "client_order_id": None,
        "pair": _text(order.get("instrument")),
        "side": _side_from_units(units),
        "units": abs(units) if units is not None else None,
        "price": _float(order.get("price")),
        "tp": _nested_price(order, "takeProfitOnFill"),
        "sl": _nested_price(order, "stopLossOnFill"),
        "realized_pl_jpy": None,
        "financing_jpy": None,
        "exit_reason": None,
        "oanda_transaction_id": None,
        "related_transaction_ids_json": json.dumps(_related_ids(response), sort_keys=True),
        "raw_json": _json(payload),
        "inserted_at_utc": now,
    }


def _gateway_position_event(action: dict[str, Any], *, payload: dict[str, Any], now: str, index: int) -> dict[str, Any]:
    request = action.get("request") if isinstance(action.get("request"), dict) else {}
    response = action.get("response") if isinstance(action.get("response"), dict) else {}
    request_type = str(request.get("type") or "none")
    if action.get("sent") and request_type == "CLOSE":
        event_type = "GATEWAY_TRADE_CLOSE_SENT"
    elif action.get("sent") and request_type == "DEPENDENT_ORDER_REPLACE":
        event_type = "GATEWAY_PROTECTION_REPLACE_SENT"
    elif request and payload.get("send_requested") and action.get("issues"):
        event_type = "GATEWAY_POSITION_ACTION_BLOCKED"
    elif request:
        event_type = "GATEWAY_POSITION_ACTION_STAGED"
    else:
        event_type = "GATEWAY_POSITION_NO_ACTION"
    order_request = request.get("order_request") if isinstance(request.get("order_request"), dict) else {}
    uid = (
        f"gateway:position_execution:{payload.get('generated_at_utc') or now}:{index}:"
        f"{event_type}:{action.get('trade_id') or ''}"
    )
    return {
        "event_uid": uid,
        "ts_utc": str(payload.get("generated_at_utc") or now),
        "source": "gateway",
        "event_type": event_type,
        "lane_id": None,
        "order_id": _response_order_id(response),
        "trade_id": _text(action.get("trade_id") or request.get("trade_id")),
        "client_order_id": None,
        "pair": _text(action.get("pair")),
        "side": None,
        "units": None,
        "price": None,
        "tp": _nested_price(order_request, "takeProfit"),
        "sl": _nested_price(order_request, "stopLoss"),
        "realized_pl_jpy": None,
        "financing_jpy": None,
        "exit_reason": _gateway_position_exit_reason(action),
        "oanda_transaction_id": None,
        "related_transaction_ids_json": json.dumps(_related_ids(response), sort_keys=True),
        "raw_json": _json(action),
        "inserted_at_utc": now,
    }


def _gateway_position_exit_reason(action: dict[str, Any]) -> str | None:
    management_action = _text(action.get("management_action"))
    if management_action == "REVIEW_EXIT":
        reason_text = " ".join(str(reason) for reason in action.get("reasons", []) or []).lower()
        if "gpt-close: accepted gpt_trader close receipt passed gate a/b" in reason_text:
            return "GPT_CLOSE"
    return management_action


def _event(
    transaction: dict[str, Any],
    inserted_at_utc: str,
    *,
    event_type: str,
    uid_tail: str,
    trade_id: str | None = None,
    side: str | None = None,
    units: int | None = None,
    price: float | None = None,
    realized_pl_jpy: float | None = None,
    exit_reason: str | None = None,
) -> dict[str, Any]:
    transaction_id = str(transaction.get("id") or "")
    signed_units = _int(transaction.get("units"))
    client_extensions = transaction.get("clientExtensions") if isinstance(transaction.get("clientExtensions"), dict) else {}
    trade_client_extensions = (
        transaction.get("tradeClientExtensions") if isinstance(transaction.get("tradeClientExtensions"), dict) else {}
    )
    nested_extensions = _nested_trade_extensions(transaction)
    raw_trade_id = trade_id or transaction.get("tradeID") or _trade_close_trade_id(transaction)
    pair = _text(transaction.get("instrument"))
    event_side = side if side is not None else _side_from_units(signed_units)
    lane_id = _lane_id(
        transaction,
        client_extensions,
        trade_client_extensions,
        *nested_extensions,
        pair=pair,
        side=event_side,
    )
    client_order_id = _text(
        transaction.get("clientOrderID")
        or client_extensions.get("id")
        or trade_client_extensions.get("id")
        or _first_extension_value(nested_extensions, "id")
    )
    return {
        "event_uid": f"oanda:{transaction_id}:{event_type}:{uid_tail}",
        "ts_utc": str(transaction.get("time") or inserted_at_utc),
        "source": "oanda",
        "event_type": event_type,
        "lane_id": lane_id,
        "order_id": _text(transaction.get("orderID") or transaction.get("replacesOrderID") or transaction_id),
        "trade_id": _text(raw_trade_id),
        "client_order_id": client_order_id,
        "pair": pair,
        "side": event_side,
        "units": units if units is not None else (_abs_int(transaction.get("units"))),
        "price": price if price is not None else _float(transaction.get("price")),
        "tp": _transaction_tp_price(transaction),
        "sl": _transaction_sl_price(transaction),
        "realized_pl_jpy": realized_pl_jpy if realized_pl_jpy is not None else _float(transaction.get("pl")),
        "financing_jpy": _float(transaction.get("financing")),
        "exit_reason": exit_reason if exit_reason is not None else _text(transaction.get("reason")),
        "oanda_transaction_id": transaction_id,
        "related_transaction_ids_json": json.dumps(_related_ids(transaction), sort_keys=True),
        "raw_json": _json(transaction),
        "inserted_at_utc": inserted_at_utc,
    }


def _insert_transaction(conn: sqlite3.Connection, transaction: dict[str, Any], now: str) -> bool:
    transaction_id = str(transaction.get("id") or "")
    if not transaction_id:
        return False
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO oanda_transactions(
            transaction_id, type, time_utc, batch_id, request_id, raw_json, inserted_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transaction_id,
            str(transaction.get("type") or "UNKNOWN"),
            _text(transaction.get("time")),
            _text(transaction.get("batchID")),
            _text(transaction.get("requestID")),
            _json(transaction),
            now,
        ),
    )
    return cur.rowcount > 0


def _insert_gateway_receipt(
    conn: sqlite3.Connection,
    *,
    kind: str,
    path: Path,
    payload: dict[str, Any],
    now: str,
) -> bool:
    lane_ids = payload.get("lane_ids")
    if not isinstance(lane_ids, list):
        lane_ids = [payload.get("lane_id")] if payload.get("lane_id") else []
    uid = f"gateway_receipt:{kind}:{payload.get('generated_at_utc') or now}:{path}"
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO gateway_receipts(
            receipt_uid, ts_utc, kind, status, sent, lane_id, lane_ids_json, path, payload_json, inserted_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uid,
            str(payload.get("generated_at_utc") or now),
            kind,
            _text(payload.get("status")),
            1 if payload.get("sent") else 0,
            _text(payload.get("lane_id")),
            json.dumps(lane_ids, ensure_ascii=False, sort_keys=True),
            str(path),
            _json(payload),
            now,
        ),
    )
    return cur.rowcount > 0


def _insert_event(conn: sqlite3.Connection, event: dict[str, Any]) -> bool:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO execution_events(
            event_uid, ts_utc, source, event_type, lane_id, order_id, trade_id, client_order_id,
            pair, side, units, price, tp, sl, realized_pl_jpy, financing_jpy, exit_reason,
            oanda_transaction_id, related_transaction_ids_json, raw_json, inserted_at_utc
        )
        VALUES (
            :event_uid, :ts_utc, :source, :event_type, :lane_id, :order_id, :trade_id, :client_order_id,
            :pair, :side, :units, :price, :tp, :sl, :realized_pl_jpy, :financing_jpy, :exit_reason,
            :oanda_transaction_id, :related_transaction_ids_json, :raw_json, :inserted_at_utc
        )
        """,
        event,
    )
    return cur.rowcount > 0


def _backfill_legacy_lane_ids(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT event_uid, pair, side, raw_json
        FROM execution_events
        WHERE source = 'oanda'
          AND (lane_id IS NULL OR lane_id = '')
        """
    ).fetchall()
    updated = 0
    for row in rows:
        try:
            transaction = json.loads(str(row["raw_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(transaction, dict):
            continue
        client_extensions = transaction.get("clientExtensions") if isinstance(transaction.get("clientExtensions"), dict) else {}
        trade_client_extensions = (
            transaction.get("tradeClientExtensions")
            if isinstance(transaction.get("tradeClientExtensions"), dict)
            else {}
        )
        lane_id = _lane_id(
            transaction,
            client_extensions,
            trade_client_extensions,
            *_nested_trade_extensions(transaction),
            pair=_text(row["pair"]),
            side=_text(row["side"]),
        )
        if not lane_id:
            continue
        cur = conn.execute(
            "UPDATE execution_events SET lane_id = ? WHERE event_uid = ? AND (lane_id IS NULL OR lane_id = '')",
            (lane_id, row["event_uid"]),
        )
        updated += cur.rowcount
    return updated


def _backfill_legacy_trade_close_ids(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT event_uid, raw_json
        FROM execution_events
        WHERE source = 'oanda'
          AND event_type = 'ORDER_ACCEPTED'
          AND exit_reason = 'TRADE_CLOSE'
          AND (trade_id IS NULL OR trade_id = '')
        """
    ).fetchall()
    updated = 0
    for row in rows:
        try:
            transaction = json.loads(str(row["raw_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(transaction, dict):
            continue
        trade_id = _trade_close_trade_id(transaction)
        if not trade_id:
            continue
        cur = conn.execute(
            "UPDATE execution_events SET trade_id = ? WHERE event_uid = ? AND (trade_id IS NULL OR trade_id = '')",
            (trade_id, row["event_uid"]),
        )
        updated += cur.rowcount
    return updated


def _get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def _set_state(conn: sqlite3.Connection, key: str, value: str, now: str) -> None:
    conn.execute(
        """
        INSERT INTO sync_state(key, value, updated_at_utc)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at_utc = excluded.updated_at_utc
        """,
        (key, value, now),
    )


def _order_uid_tail(transaction: dict[str, Any]) -> str:
    return str(transaction.get("orderID") or transaction.get("id") or transaction.get("clientOrderID") or "")


def _trade_close_trade_id(transaction: dict[str, Any]) -> str | None:
    trade_close = transaction.get("tradeClose")
    if not isinstance(trade_close, dict):
        return None
    for key in ("tradeID", "trade_id", "id"):
        value = trade_close.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _response_order_id(response: dict[str, Any]) -> str | None:
    for key in ("orderCreateTransaction", "orderFillTransaction", "orderCancelTransaction", "orderRejectTransaction"):
        payload = response.get(key)
        if isinstance(payload, dict):
            value = payload.get("orderID") or payload.get("id")
            if value is not None:
                return str(value)
    return None


def _response_trade_id(response: dict[str, Any]) -> str | None:
    fill = response.get("orderFillTransaction")
    if isinstance(fill, dict):
        opened = fill.get("tradeOpened")
        if isinstance(opened, dict) and opened.get("tradeID") is not None:
            return str(opened["tradeID"])
        closed = fill.get("tradesClosed")
        if isinstance(closed, list) and closed and isinstance(closed[0], dict) and closed[0].get("tradeID") is not None:
            return str(closed[0]["tradeID"])
    return None


def _lane_id(
    transaction: dict[str, Any],
    *payloads: dict[str, Any],
    pair: str | None = None,
    side: str | None = None,
) -> str | None:
    for payload in (transaction, *payloads):
        for key in ("lane_id", "laneId"):
            value = payload.get(key)
            if value:
                return str(value)
        lane = _lane_id_from_comment(payload.get("comment"), pair=pair, side=side)
        if lane:
            return lane
    return None


_LEGACY_DESK_METHOD = {
    "trend_trader": "TREND_CONTINUATION",
    "range_trader": "RANGE_ROTATION",
    "failure_trader": "BREAKOUT_FAILURE",
}


def _lane_id_from_comment(value: Any, *, pair: str | None = None, side: str | None = None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    for token in text.split():
        for prefix in ("lane=", "lane_id=", "laneId="):
            if token.startswith(prefix):
                lane = token[len(prefix) :].strip()
                return lane or None
    pair_text = str(pair or "").strip().upper()
    side_text = str(side or "").strip().upper()
    if not pair_text or side_text not in {"LONG", "SHORT"}:
        return None
    if not text.startswith("qr-vnext"):
        return None
    for token in text.split():
        desk = token.strip()
        method = _LEGACY_DESK_METHOD.get(desk)
        if method:
            return f"{desk}:{pair_text}:{side_text}:{method}"
    return None


def _nested_trade_extensions(transaction: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    payloads: list[dict[str, Any]] = []
    for trade_key in ("tradeOpened", "tradeReduced"):
        nested = transaction.get(trade_key)
        if not isinstance(nested, dict):
            continue
        for extension_key in ("clientExtensions", "tradeClientExtensions"):
            extensions = nested.get(extension_key)
            if isinstance(extensions, dict):
                payloads.append(extensions)
    return tuple(payloads)


def _first_extension_value(payloads: tuple[dict[str, Any], ...], key: str) -> Any:
    for payload in payloads:
        value = payload.get(key)
        if value:
            return value
    return None


def _related_ids(payload: dict[str, Any]) -> list[str]:
    ids = payload.get("relatedTransactionIDs")
    if isinstance(ids, list):
        return [str(item) for item in ids]
    related: list[str] = []
    for key in ("id", "orderID", "tradeID", "batchID"):
        value = payload.get(key)
        if value is not None:
            related.append(str(value))
    trade_close = payload.get("tradeClose")
    if isinstance(trade_close, dict):
        trade_id = _trade_close_trade_id(payload)
        if trade_id is not None:
            related.append(trade_id)
    return related


def _nested_price(payload: dict[str, Any], key: str) -> float | None:
    nested = payload.get(key)
    if not isinstance(nested, dict):
        return None
    return _float(nested.get("price"))


def _side_from_units(units: int | None) -> str | None:
    if units is None or units == 0:
        return None
    return "LONG" if units > 0 else "SHORT"


def _closed_trade_side(transaction: dict[str, Any]) -> str | None:
    signed_units = _int(transaction.get("units"))
    if signed_units is None or signed_units == 0:
        return None
    closing_side = _side_from_units(signed_units)
    if closing_side == "LONG":
        return "SHORT"
    if closing_side == "SHORT":
        return "LONG"
    return None


def _transaction_tp_price(transaction: dict[str, Any]) -> float | None:
    nested = _nested_price(transaction, "takeProfitOnFill")
    if nested is not None:
        return nested
    if str(transaction.get("type") or "") == "TAKE_PROFIT_ORDER":
        return _float(transaction.get("price"))
    return None


def _transaction_sl_price(transaction: dict[str, Any]) -> float | None:
    nested = _nested_price(transaction, "stopLossOnFill")
    if nested is not None:
        return nested
    if str(transaction.get("type") or "") == "STOP_LOSS_ORDER":
        return _float(transaction.get("price"))
    return None


def _abs_int(value: Any) -> int | None:
    parsed = _int(value)
    return abs(parsed) if parsed is not None else None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
