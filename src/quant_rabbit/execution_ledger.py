from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from quant_rabbit.models import AccountSummary
from quant_rabbit.paths import DEFAULT_EXECUTION_LEDGER_DB, DEFAULT_EXECUTION_LEDGER_REPORT


GATEWAY_TRADE_CLOSE_RECONCILED = "GATEWAY_TRADE_CLOSE_RECONCILED"
# GPT CLOSE is an immediate market-close gateway handoff. Twenty minutes spans a
# delayed ledger sync or one missed cycle, while preventing a much later manual
# close on the same trade from being attributed back to a stale GPT receipt.
GPT_CLOSE_RECONCILE_MAX_ACCEPT_DELAY_SECONDS = 20 * 60
# Broker and receipt timestamps can differ slightly across local write time,
# OANDA nanosecond timestamps, and transaction fetch order.
GPT_CLOSE_RECONCILE_CLOCK_SKEW_SECONDS = 60
# Claims written before the expiry column existed can only have come from the
# original SCOUT contract, whose broker GTD ceiling is 90 minutes.  This
# compatibility ceiling prevents a legacy claim from occupying a concurrent
# slot forever without inventing a shorter market TTL.
PREDICTIVE_SCOUT_LEGACY_CLAIM_MAX_TTL_MINUTES = 90
# SQLite PRAGMA/schema setup can fail immediately when several fresh worker
# processes open a brand-new ledger together, even with connection busy_timeout.
# These short engineering backoffs serialize that one-time migration without
# weakening the reservation transaction or waiting anywhere near a trade TTL.
SQLITE_SCHEMA_INIT_RETRY_DELAYS_SECONDS = (0.05, 0.10, 0.20, 0.40, 0.80)
# These three compatibility backfills repair rows written by older ledger
# versions.  Current transaction/gateway parsers populate the fields on insert,
# while genuinely manual/tagless broker rows can never acquire a lane/order id.
# Persist a version in sync_state so those irreparable rows are not reparsed on
# every sync and receipt write.  Bump the version only when the migration logic
# itself gains a new repair capability that must be applied to existing rows.
LEGACY_EVENT_BACKFILL_MIGRATION_KEY = (
    "migration:execution_ledger:legacy_event_backfills"
)
LEGACY_EVENT_BACKFILL_MIGRATION_VERSION = "1"


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
    verification_observations_inserted: int = 0
    reconciled_gateway_events_inserted: int = 0
    last_transaction_id: str | None = None
    baseline_transaction_id: str | None = None


@dataclass(frozen=True)
class PredictiveScoutReservationResult:
    reserved: bool
    status: str
    daily_reserved: int
    active_slots: int


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
                reconciled_events = _reconcile_gateway_trade_close_broker_accepts(conn, now=now)
                summary = ExecutionLedgerSummary(
                    db_path=self.db_path,
                    report_path=self.report_path,
                    status="BASELINED",
                    baseline_transaction_id=baseline,
                    last_transaction_id=baseline,
                    reconciled_gateway_events_inserted=reconciled_events,
                )
                self._write_report(summary)
                return summary

            payload = client.transactions_since_id(str(start_id))
            transactions = [tx for tx in payload.get("transactions", []) or [] if isinstance(tx, dict)]
            inserted_transactions = 0
            inserted_events = 0
            opened_trade_ids: list[str] = []
            for transaction in transactions:
                if _insert_transaction(conn, transaction, now):
                    inserted_transactions += 1
                for event in _events_from_transaction(transaction, now):
                    if _insert_event(conn, event):
                        inserted_events += 1
                _record_entry_thesis_for_fill(transaction, data_root=self.db_path.parent)
                opened = transaction.get("tradeOpened")
                if str(transaction.get("type") or "") == "ORDER_FILL" and isinstance(opened, dict):
                    trade_id = str(opened.get("tradeID") or "").strip()
                    if trade_id:
                        opened_trade_ids.append(trade_id)
            last_transaction_id = str(payload.get("lastTransactionID") or start_id)
            _set_state(conn, "last_oanda_transaction_id", last_transaction_id, now)
            reconciled_events = _reconcile_gateway_trade_close_broker_accepts(conn, now=now)

        _backfill_entry_thesis_for_opened_trades(
            db_path=self.db_path,
            data_root=self.db_path.parent,
            trade_ids=opened_trade_ids,
        )

        summary = ExecutionLedgerSummary(
            db_path=self.db_path,
            report_path=self.report_path,
            status="SYNCED",
            transactions_seen=len(transactions),
            transactions_inserted=inserted_transactions,
            events_inserted=inserted_events,
            reconciled_gateway_events_inserted=reconciled_events,
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
        return self.record_gateway_payload(
            kind=kind,
            receipt_path=receipt_path,
            payload=payload,
        )

    def record_gateway_payload(
        self,
        *,
        kind: str,
        receipt_path: Path,
        payload: dict[str, Any],
    ) -> ExecutionLedgerSummary:
        """Durably index a gateway payload without requiring a prior file write."""

        self._init_db()
        now = _now()
        with self._connect() as conn:
            receipt_inserted = _insert_gateway_receipt(
                conn,
                kind=kind,
                path=receipt_path,
                payload=payload,
                now=now,
            )
            inserted_events = 0
            for event in _events_from_gateway_receipt(kind=kind, payload=payload, now=now):
                if _insert_event(conn, event):
                    inserted_events += 1
            inserted_observations = _insert_gateway_close_gate_observations(
                conn,
                kind=kind,
                path=receipt_path,
                payload=payload,
                now=now,
            )
            reconciled_events = _reconcile_gateway_trade_close_broker_accepts(conn, now=now)
        summary = ExecutionLedgerSummary(
            db_path=self.db_path,
            report_path=self.report_path,
            status="RECORDED",
            gateway_receipts_inserted=1 if receipt_inserted else 0,
            events_inserted=inserted_events,
            verification_observations_inserted=inserted_observations,
            reconciled_gateway_events_inserted=reconciled_events,
        )
        self._write_report(summary)
        return summary

    def reserve_predictive_scout_gateway_payload(
        self,
        *,
        kind: str,
        receipt_path: Path,
        payload: dict[str, Any],
        signal_id: str,
        experiment_id: str,
        vehicle_id: str,
        expires_at_utc: str,
        max_daily: int,
        max_concurrent: int,
        broker_active_total: int,
        broker_active_signal_ids: set[str] | None = None,
    ) -> PredictiveScoutReservationResult:
        """Atomically claim one signal plus its shared daily/concurrent slot.

        ``BEGIN IMMEDIATE`` serializes different signal ids as well as duplicate
        ids.  Broker-active vehicles are reconciled against still-live local
        reservations by exact signal id, so a reflected broker order consumes one slot, while an
        in-flight POST that has not reached the broker snapshot also consumes
        one slot.  This closes the stale-snapshot race where two bots could both
        observe one free slot and POST simultaneously.
        """

        if not signal_id or not experiment_id or not vehicle_id:
            raise ValueError("predictive SCOUT reservation identity is incomplete")
        if max_daily <= 0 or max_concurrent <= 0:
            raise ValueError("predictive SCOUT reservation caps must be positive")
        expiry = _parse_utc(expires_at_utc)
        if expiry is None:
            raise ValueError("predictive SCOUT reservation expiry must be timezone-aware UTC")
        broker_active_total = max(0, int(broker_active_total))
        broker_signals = {
            str(signal_id)
            for signal_id in (broker_active_signal_ids or set())
            if str(signal_id)
        }
        self._init_db()
        now = _now()
        now_dt = _parse_utc(now)
        assert now_dt is not None
        if expiry <= now_dt:
            raise ValueError("predictive SCOUT reservation expiry is not in the future")
        day = now_dt.date().isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for broker_signal in broker_signals:
                conn.execute(
                    """
                    UPDATE predictive_scout_signal_claims
                    SET broker_reflected_at_utc = COALESCE(broker_reflected_at_utc, ?)
                    WHERE signal_id = ?
                    """,
                    (now, broker_signal),
                )
            duplicate = conn.execute(
                "SELECT 1 FROM predictive_scout_signal_claims WHERE signal_id = ?",
                (signal_id,),
            ).fetchone()
            daily_budget_keys = {
                f"signal:{str(row[0])}"
                for row in conn.execute(
                    """
                    SELECT signal_id
                    FROM predictive_scout_signal_claims
                    WHERE substr(reserved_at_utc, 1, 10) = ?
                    """,
                    (day,),
                ).fetchall()
                if str(row[0] or "")
            }
            for row in conn.execute(
                """
                SELECT rowid, payload_json
                FROM gateway_receipts
                WHERE sent = 1 AND substr(ts_utc, 1, 10) = ?
                """,
                (day,),
            ).fetchall():
                try:
                    legacy_payload = json.loads(row["payload_json"])
                except (TypeError, json.JSONDecodeError):
                    raise ValueError("predictive SCOUT daily receipt history is unreadable")
                daily_budget_keys.update(
                    _predictive_scout_budget_keys(
                        legacy_payload,
                        fallback=f"legacy-receipt:{row['rowid']}",
                    )
                )
            for row in conn.execute(
                """
                SELECT rowid, event_type, raw_json
                FROM execution_events
                WHERE event_type IN ('GATEWAY_ORDER_SENT', 'GATEWAY_ORDER_STAGED')
                  AND substr(ts_utc, 1, 10) = ?
                """,
                (day,),
            ).fetchall():
                try:
                    legacy_payload = json.loads(row["raw_json"])
                except (TypeError, json.JSONDecodeError):
                    raise ValueError("predictive SCOUT daily event history is unreadable")
                is_reservation = bool(
                    legacy_payload.get("predictive_scout_post_reserved") is True
                    or str(legacy_payload.get("status") or "").upper()
                    == "PREDICTIVE_SCOUT_POST_RESERVED"
                )
                if row["event_type"] != "GATEWAY_ORDER_SENT" and not is_reservation:
                    continue
                daily_budget_keys.update(
                    _predictive_scout_budget_keys(
                        legacy_payload,
                        fallback=f"legacy-event:{row['rowid']}",
                    )
                )
            daily_reserved = len(daily_budget_keys)
            unreflected_claims = 0
            for row in conn.execute(
                """
                SELECT signal_id, reserved_at_utc, expires_at_utc,
                       broker_reflected_at_utc
                FROM predictive_scout_signal_claims
                """
            ).fetchall():
                claim_expiry = _parse_utc(row["expires_at_utc"])
                if claim_expiry is None:
                    reserved_at = _parse_utc(row["reserved_at_utc"])
                    if reserved_at is not None:
                        claim_expiry = reserved_at + timedelta(
                            minutes=PREDICTIVE_SCOUT_LEGACY_CLAIM_MAX_TTL_MINUTES
                        )
                if claim_expiry is None or claim_expiry <= now_dt:
                    continue
                claimed_signal = str(row["signal_id"] or "")
                was_reflected = bool(str(row["broker_reflected_at_utc"] or "").strip())
                if (
                    claimed_signal
                    and claimed_signal not in broker_signals
                    and not was_reflected
                ):
                    unreflected_claims += 1
            active_slots = broker_active_total + unreflected_claims

            if duplicate is not None:
                return PredictiveScoutReservationResult(
                    reserved=False,
                    status="DUPLICATE_SIGNAL",
                    daily_reserved=daily_reserved,
                    active_slots=active_slots,
                )
            if daily_reserved >= max_daily:
                return PredictiveScoutReservationResult(
                    reserved=False,
                    status="DAILY_CAP_REACHED",
                    daily_reserved=daily_reserved,
                    active_slots=active_slots,
                )
            if active_slots >= max_concurrent:
                return PredictiveScoutReservationResult(
                    reserved=False,
                    status="CONCURRENT_CAP_REACHED",
                    daily_reserved=daily_reserved,
                    active_slots=active_slots,
                )

            claim = conn.execute(
                """
                INSERT OR IGNORE INTO predictive_scout_signal_claims(
                    signal_id, experiment_id, vehicle_id, reserved_at_utc,
                    expires_at_utc, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    experiment_id,
                    vehicle_id,
                    now,
                    expiry.isoformat(),
                    _json(payload),
                ),
            )
            if claim.rowcount <= 0:
                return PredictiveScoutReservationResult(
                    reserved=False,
                    status="DUPLICATE_SIGNAL",
                    daily_reserved=daily_reserved,
                    active_slots=active_slots,
                )
            receipt_inserted = _insert_gateway_receipt(
                conn,
                kind=kind,
                path=receipt_path,
                payload=payload,
                now=now,
            )
            inserted_events = 0
            for event in _events_from_gateway_receipt(kind=kind, payload=payload, now=now):
                if _insert_event(conn, event):
                    inserted_events += 1
            inserted_observations = _insert_gateway_close_gate_observations(
                conn,
                kind=kind,
                path=receipt_path,
                payload=payload,
                now=now,
            )
            reconciled_events = _reconcile_gateway_trade_close_broker_accepts(conn, now=now)
        summary = ExecutionLedgerSummary(
            db_path=self.db_path,
            report_path=self.report_path,
            status="PREDICTIVE_SCOUT_SIGNAL_RESERVED",
            gateway_receipts_inserted=1 if receipt_inserted else 0,
            events_inserted=inserted_events,
            verification_observations_inserted=inserted_observations,
            reconciled_gateway_events_inserted=reconciled_events,
        )
        self._write_report(summary)
        return PredictiveScoutReservationResult(
            reserved=True,
            status="RESERVED",
            daily_reserved=daily_reserved + 1,
            active_slots=active_slots + 1,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        for delay in (*SQLITE_SCHEMA_INIT_RETRY_DELAYS_SECONDS, None):
            try:
                self._init_db_once()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or delay is None:
                    raise
                time.sleep(delay)

    def _init_db_once(self) -> None:
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
                CREATE INDEX IF NOT EXISTS idx_gateway_receipts_sent_day
                    ON gateway_receipts(sent, substr(ts_utc, 1, 10));
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
                CREATE TABLE IF NOT EXISTS predictive_scout_signal_claims (
                    signal_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    reserved_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT,
                    broker_reflected_at_utc TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_predictive_scout_signal_claims_vehicle
                    ON predictive_scout_signal_claims(vehicle_id, reserved_at_utc);
                CREATE TABLE IF NOT EXISTS verification_observations (
                    observation_uid TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_path TEXT,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT,
                    check_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT,
                    metric_value REAL,
                    metric_unit TEXT,
                    evidence_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_verification_observations_ts
                    ON verification_observations(ts_utc);
                CREATE INDEX IF NOT EXISTS idx_verification_observations_status
                    ON verification_observations(status, severity);
                CREATE INDEX IF NOT EXISTS idx_verification_observations_subject
                    ON verification_observations(subject_type, subject_id);
                """
            )
            claim_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(predictive_scout_signal_claims)"
                ).fetchall()
            }
            for column_name in ("expires_at_utc", "broker_reflected_at_utc"):
                if column_name in claim_columns:
                    continue
                try:
                    conn.execute(
                        "ALTER TABLE predictive_scout_signal_claims "
                        f"ADD COLUMN {column_name} TEXT"
                    )
                except sqlite3.OperationalError as exc:
                    # Another process may complete the same additive migration
                    # after our PRAGMA read but before our ALTER acquires the
                    # schema lock.  Only accept that precise converged state.
                    refreshed = {
                        str(row[1])
                        for row in conn.execute(
                            "PRAGMA table_info(predictive_scout_signal_claims)"
                        ).fetchall()
                    }
                    if column_name not in refreshed:
                        raise exc
                claim_columns.add(column_name)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictive_scout_signal_claims_expiry "
                "ON predictive_scout_signal_claims(expires_at_utc)"
            )
            # executescript/schema changes must be durable before the migration
            # opens its own IMMEDIATE transaction.  This also makes a failed
            # migration rollback only its row repairs and marker, leaving the
            # additive schema safe for the retry loop.
            conn.commit()
            self._run_legacy_event_backfill_migration(conn)

    @staticmethod
    def _run_legacy_event_backfill_migration(conn: sqlite3.Connection) -> None:
        if (
            _get_state(conn, LEGACY_EVENT_BACKFILL_MIGRATION_KEY)
            == LEGACY_EVENT_BACKFILL_MIGRATION_VERSION
        ):
            return

        # Serialize the check/repair/marker sequence across cold-start workers.
        # The second check is mandatory: another initializer may finish while
        # this connection waits for the write lock.  Marker insertion is in the
        # same transaction as every row update, so an exception rolls both back
        # and the next initializer safely retries the complete migration.
        conn.execute("BEGIN IMMEDIATE")
        try:
            if (
                _get_state(conn, LEGACY_EVENT_BACKFILL_MIGRATION_KEY)
                != LEGACY_EVENT_BACKFILL_MIGRATION_VERSION
            ):
                _backfill_legacy_lane_ids(conn)
                _backfill_legacy_trade_close_ids(conn)
                _backfill_gateway_position_order_ids(conn)
                _set_state(
                    conn,
                    LEGACY_EVENT_BACKFILL_MIGRATION_KEY,
                    LEGACY_EVENT_BACKFILL_MIGRATION_VERSION,
                    _now(),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

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
            f"- Reconciled gateway events inserted: `{summary.reconciled_gateway_events_inserted}`",
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


_UNSET_EVENT_VALUE = object()


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
        reduced_pl = _float(reduced.get("realizedPL"))
        reduced_financing = _float(reduced.get("financing"))
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
                realized_pl_jpy=(
                    reduced_pl if reduced_pl is not None else _float(transaction.get("pl"))
                ),
                financing_jpy=(
                    reduced_financing
                    if reduced_financing is not None
                    else _float(transaction.get("financing"))
                ),
            )
        )
    closed_items = transaction.get("tradesClosed")
    if isinstance(closed_items, list):
        multi_close = sum(1 for item in closed_items if isinstance(item, dict)) > 1
        for index, closed in enumerate(closed_items):
            if not isinstance(closed, dict):
                continue
            closed_pl = _float(closed.get("realizedPL"))
            closed_financing = _float(closed.get("financing"))
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
                    realized_pl_jpy=(
                        closed_pl
                        if closed_pl is not None
                        else (None if multi_close else _float(transaction.get("pl")))
                    ),
                    financing_jpy=(
                        closed_financing
                        if closed_financing is not None
                        else (None if multi_close else _float(transaction.get("financing")))
                    ),
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
    if kind == "gpt_decision":
        return _gateway_gpt_close_events(payload, now=now)
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


def _backfill_entry_thesis_for_opened_trades(
    *,
    db_path: Path,
    data_root: Path,
    trade_ids: list[str],
) -> None:
    if not trade_ids:
        return
    try:
        from quant_rabbit.strategy.entry_thesis_ledger import backfill_entry_theses_from_execution_ledger

        backfill_entry_theses_from_execution_ledger(
            db_path=db_path,
            data_root=data_root,
            trade_ids=trade_ids,
        )
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
    client_extensions = order.get("clientExtensions") if isinstance(order.get("clientExtensions"), dict) else {}
    return {
        "event_uid": uid,
        "ts_utc": str(payload.get("generated_at_utc") or now),
        "source": "gateway",
        "event_type": event_type,
        "lane_id": lane_id,
        "order_id": order_id,
        "trade_id": trade_id,
        "client_order_id": _text(client_extensions.get("id")),
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


def _gateway_gpt_close_events(payload: dict[str, Any], *, now: str) -> list[dict[str, Any]]:
    status = str(payload.get("status") or "").upper()
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else payload
    action = str(decision.get("action") or "").upper() if isinstance(decision, dict) else ""
    if status != "ACCEPTED" or action != "CLOSE":
        return []
    close_trade_ids = decision.get("close_trade_ids") if isinstance(decision.get("close_trade_ids"), list) else []
    ts = str(payload.get("generated_at_utc") or now)
    close_gate_evidence = (
        payload.get("close_gate_evidence")
        if isinstance(payload.get("close_gate_evidence"), list)
        else []
    )
    events: list[dict[str, Any]] = []
    for index, trade_id_value in enumerate(close_trade_ids):
        trade_id = str(trade_id_value or "").strip()
        if not trade_id:
            continue
        trade_close_gate_evidence = [
            evidence
            for evidence in close_gate_evidence
            if isinstance(evidence, dict) and str(evidence.get("trade_id") or "").strip() == trade_id
        ]
        raw = {
            "generated_at_utc": payload.get("generated_at_utc"),
            "status": payload.get("status"),
            "decision": decision,
            "verification_issues": payload.get("verification_issues") or [],
            "close_gate_evidence": trade_close_gate_evidence,
        }
        events.append(
            {
                "event_uid": f"gateway:gpt_decision:{ts}:{index}:GATEWAY_GPT_CLOSE_ACCEPTED:{trade_id}",
                "ts_utc": ts,
                "source": "gateway",
                "event_type": "GATEWAY_GPT_CLOSE_ACCEPTED",
                "lane_id": _text(decision.get("selected_lane_id")),
                "order_id": None,
                "trade_id": trade_id,
                "client_order_id": None,
                "pair": None,
                "side": None,
                "units": None,
                "price": None,
                "tp": None,
                "sl": None,
                "realized_pl_jpy": None,
                "financing_jpy": None,
                "exit_reason": "GPT_CLOSE_ACCEPTED",
                "oanda_transaction_id": None,
                "related_transaction_ids_json": "[]",
                "raw_json": _json(raw),
                "inserted_at_utc": now,
            }
        )
    return events


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
    realized_pl_jpy: float | None | object = _UNSET_EVENT_VALUE,
    financing_jpy: float | None | object = _UNSET_EVENT_VALUE,
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
        "realized_pl_jpy": (
            _float(transaction.get("pl"))
            if realized_pl_jpy is _UNSET_EVENT_VALUE
            else realized_pl_jpy
        ),
        "financing_jpy": (
            _float(transaction.get("financing"))
            if financing_jpy is _UNSET_EVENT_VALUE
            else financing_jpy
        ),
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


def _insert_gateway_close_gate_observations(
    conn: sqlite3.Connection,
    *,
    kind: str,
    path: Path,
    payload: dict[str, Any],
    now: str,
) -> int:
    if kind != "gpt_decision":
        return 0
    close_gate_evidence = (
        payload.get("close_gate_evidence")
        if isinstance(payload.get("close_gate_evidence"), list)
        else []
    )
    if not close_gate_evidence:
        return 0
    ts = str(payload.get("generated_at_utc") or now)
    inserted = 0
    for index, evidence in enumerate(close_gate_evidence):
        if not isinstance(evidence, dict):
            continue
        trade_id = str(evidence.get("trade_id") or index)
        status = _close_gate_evidence_status(evidence)
        observation = {
            "observation_uid": f"execution_ledger:gpt_decision:{ts}:close_gate_evidence:{trade_id}:{index}",
            "ts_utc": ts,
            "source": "gpt_decision",
            "source_path": str(path),
            "subject_type": "close_gate",
            "subject_id": trade_id,
            "check_name": "close_gate_evidence",
            "status": status,
            "severity": "INFO" if status == "PASS" else "BLOCK",
            "metric_value": None,
            "metric_unit": None,
            "evidence_json": _json(evidence),
            "inserted_at_utc": now,
        }
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO verification_observations(
                observation_uid, ts_utc, source, source_path, subject_type, subject_id,
                check_name, status, severity, metric_value, metric_unit, evidence_json,
                inserted_at_utc
            )
            VALUES (
                :observation_uid, :ts_utc, :source, :source_path, :subject_type, :subject_id,
                :check_name, :status, :severity, :metric_value, :metric_unit, :evidence_json,
                :inserted_at_utc
            )
            """,
            observation,
        )
        inserted += int(cur.rowcount > 0)
    return inserted


def _close_gate_evidence_status(evidence: dict[str, Any]) -> str:
    if evidence.get("gate_a_invalidated") is not True:
        return "BLOCK"
    if evidence.get("same_direction_support_conflict"):
        return "BLOCK"
    if evidence.get("hard_timing_gate_required") is True:
        return "BLOCK"
    if (
        evidence.get("explicit_gate_b_required") is True
        and evidence.get("gate_b_explicit_operator_authorized") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("profitability_p0_context_required") is True
        and evidence.get("profitability_p0_context_cited") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("timing_audit_required") is True
        and evidence.get("timing_evidence_cited") is not True
    ):
        return "BLOCK"
    return "PASS"


def _reconcile_gateway_trade_close_broker_accepts(conn: sqlite3.Connection, *, now: str) -> int:
    return _reconcile_gateway_gpt_close_broker_accepts(conn, now=now) + _reconcile_trader_entry_broker_close_accepts(
        conn,
        now=now,
    )


def _reconcile_gateway_gpt_close_broker_accepts(conn: sqlite3.Connection, *, now: str) -> int:
    rows = conn.execute(
        f"""
        SELECT
            g.event_uid AS gpt_event_uid,
            g.ts_utc AS gpt_ts_utc,
            g.lane_id AS gpt_lane_id,
            g.trade_id AS trade_id,
            a.event_uid AS accepted_event_uid,
            a.ts_utc AS accepted_ts_utc,
            a.order_id AS accepted_order_id,
            a.pair AS accepted_pair,
            a.oanda_transaction_id AS accepted_oanda_transaction_id,
            c.event_uid AS close_event_uid,
            c.ts_utc AS close_ts_utc,
            c.pair AS close_pair,
            c.realized_pl_jpy AS close_realized_pl_jpy,
            c.exit_reason AS close_exit_reason,
            c.oanda_transaction_id AS close_oanda_transaction_id
        FROM execution_events g
        INNER JOIN execution_events a
          ON a.event_type = 'ORDER_ACCEPTED'
         AND a.exit_reason = 'TRADE_CLOSE'
         AND a.trade_id = g.trade_id
        INNER JOIN execution_events c
          ON c.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
         AND c.trade_id = g.trade_id
         AND (
             COALESCE(a.order_id, '') = ''
             OR COALESCE(c.order_id, '') = ''
             OR c.order_id = a.order_id
         )
        WHERE g.event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
          AND COALESCE(g.trade_id, '') != ''
          AND NOT EXISTS (
              SELECT 1
              FROM execution_events sent
              WHERE sent.event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                AND sent.trade_id = g.trade_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM execution_events reconciled
              WHERE reconciled.event_type = '{GATEWAY_TRADE_CLOSE_RECONCILED}'
                AND reconciled.trade_id = g.trade_id
          )
        ORDER BY g.ts_utc ASC, a.ts_utc ASC, c.ts_utc ASC
        """
    ).fetchall()
    inserted = 0
    seen_trade_ids: set[str] = set()
    for row in rows:
        trade_id = _text(row["trade_id"])
        if not trade_id or trade_id in seen_trade_ids:
            continue
        gpt_ts = _parse_utc(row["gpt_ts_utc"])
        accepted_ts = _parse_utc(row["accepted_ts_utc"])
        close_ts = _parse_utc(row["close_ts_utc"])
        if not _close_reconcile_timestamps_match(gpt_ts, accepted_ts, close_ts):
            continue
        seen_trade_ids.add(trade_id)
        order_id = _text(row["accepted_order_id"])
        related_ids: list[str] = []
        for item in (
            _text(row["accepted_oanda_transaction_id"]),
            _text(row["close_oanda_transaction_id"]),
            order_id,
        ):
            if item and item not in related_ids:
                related_ids.append(item)
        raw = {
            "reconciled_from": [
                "GATEWAY_GPT_CLOSE_ACCEPTED",
                "ORDER_ACCEPTED:TRADE_CLOSE",
                str(row["close_exit_reason"] or "TRADE_CLOSED"),
            ],
            "gpt_close_event_uid": row["gpt_event_uid"],
            "order_accept_event_uid": row["accepted_event_uid"],
            "close_event_uid": row["close_event_uid"],
            "trade_id": trade_id,
            "order_id": order_id,
            "accepted_ts_utc": row["accepted_ts_utc"],
            "close_ts_utc": row["close_ts_utc"],
            "realized_pl_jpy": row["close_realized_pl_jpy"],
        }
        event = {
            "event_uid": (
                f"ledger_reconcile:gpt_close_broker_accept:{trade_id}:{order_id or ''}:"
                f"{GATEWAY_TRADE_CLOSE_RECONCILED}"
            ),
            "ts_utc": str(row["close_ts_utc"] or row["accepted_ts_utc"] or row["gpt_ts_utc"] or now),
            "source": "ledger_reconcile",
            "event_type": GATEWAY_TRADE_CLOSE_RECONCILED,
            "lane_id": _text(row["gpt_lane_id"]),
            "order_id": order_id,
            "trade_id": trade_id,
            "client_order_id": None,
            "pair": _text(row["close_pair"]) or _text(row["accepted_pair"]),
            "side": None,
            "units": None,
            "price": None,
            "tp": None,
            "sl": None,
            "realized_pl_jpy": None,
            "financing_jpy": None,
            "exit_reason": "GPT_CLOSE_RECONCILED",
            "oanda_transaction_id": _text(row["accepted_oanda_transaction_id"]),
            "related_transaction_ids_json": json.dumps(related_ids, sort_keys=True),
            "raw_json": _json(raw),
            "inserted_at_utc": now,
        }
        if _insert_event(conn, event):
            inserted += 1
    return inserted


def _reconcile_trader_entry_broker_close_accepts(conn: sqlite3.Connection, *, now: str) -> int:
    rows = conn.execute(
        f"""
        WITH gateway_entries AS (
            SELECT trade_id, order_id, lane_id
            FROM execution_events
            WHERE event_type = 'GATEWAY_ORDER_SENT'
              AND COALESCE(lane_id, '') != ''
        ),
        entries AS (
            SELECT
                e.trade_id,
                COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS gateway_lane_id,
                MAX(e.client_order_id) AS client_order_id,
                MAX(e.pair) AS entry_pair,
                CASE
                    WHEN MAX(e.units) > 0 THEN 'LONG'
                    WHEN MIN(e.units) < 0 THEN 'SHORT'
                    ELSE NULL
                END AS entry_side
            FROM execution_events e
            LEFT JOIN gateway_entries g
              ON (
                COALESCE(g.trade_id, '') != ''
                AND g.trade_id = e.trade_id
              )
              OR (
                COALESCE(g.order_id, '') != ''
                AND g.order_id = e.order_id
              )
            WHERE e.event_type = 'ORDER_FILLED'
              AND COALESCE(e.trade_id, '') != ''
            GROUP BY e.trade_id
            HAVING COALESCE(gateway_lane_id, '') != ''
        )
        SELECT
            entries.gateway_lane_id,
            entries.client_order_id,
            entries.entry_pair,
            entries.entry_side,
            a.event_uid AS accepted_event_uid,
            a.ts_utc AS accepted_ts_utc,
            a.order_id AS accepted_order_id,
            a.trade_id AS trade_id,
            a.pair AS accepted_pair,
            a.oanda_transaction_id AS accepted_oanda_transaction_id,
            c.event_uid AS close_event_uid,
            c.ts_utc AS close_ts_utc,
            c.pair AS close_pair,
            c.side AS close_side,
            c.realized_pl_jpy AS close_realized_pl_jpy,
            c.exit_reason AS close_exit_reason,
            c.oanda_transaction_id AS close_oanda_transaction_id
        FROM entries
        INNER JOIN execution_events a
          ON a.event_type = 'ORDER_ACCEPTED'
         AND a.exit_reason = 'TRADE_CLOSE'
         AND a.trade_id = entries.trade_id
        INNER JOIN execution_events c
          ON c.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
         AND c.trade_id = entries.trade_id
         AND (
             COALESCE(a.order_id, '') = ''
             OR COALESCE(c.order_id, '') = ''
             OR c.order_id = a.order_id
         )
        WHERE NOT EXISTS (
              SELECT 1
              FROM execution_events sent
              WHERE sent.event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                AND sent.trade_id = entries.trade_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM execution_events reconciled
              WHERE reconciled.event_type = '{GATEWAY_TRADE_CLOSE_RECONCILED}'
                AND reconciled.trade_id = entries.trade_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM execution_events gpt
              WHERE gpt.event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
                AND gpt.trade_id = entries.trade_id
          )
        ORDER BY a.ts_utc ASC, c.ts_utc ASC
        """
    ).fetchall()
    inserted = 0
    seen_trade_ids: set[str] = set()
    for row in rows:
        trade_id = _text(row["trade_id"])
        if not trade_id or trade_id in seen_trade_ids:
            continue
        accepted_ts = _parse_utc(row["accepted_ts_utc"])
        close_ts = _parse_utc(row["close_ts_utc"])
        if accepted_ts is None:
            continue
        if close_ts is not None:
            close_delay = (close_ts - accepted_ts).total_seconds()
            if close_delay < -GPT_CLOSE_RECONCILE_CLOCK_SKEW_SECONDS:
                continue
            if close_delay > GPT_CLOSE_RECONCILE_MAX_ACCEPT_DELAY_SECONDS:
                continue
        seen_trade_ids.add(trade_id)
        order_id = _text(row["accepted_order_id"])
        related_ids: list[str] = []
        for item in (
            _text(row["accepted_oanda_transaction_id"]),
            _text(row["close_oanda_transaction_id"]),
            order_id,
        ):
            if item and item not in related_ids:
                related_ids.append(item)
        raw = {
            "reconciled_from": [
                "TRADER_ENTRY_LANE_ID",
                "ORDER_ACCEPTED:TRADE_CLOSE",
                str(row["close_exit_reason"] or "TRADE_CLOSED"),
            ],
            "reconcile_reason": "NO_LOCAL_POSITION_EXECUTION_RECEIPT",
            "order_accept_event_uid": row["accepted_event_uid"],
            "close_event_uid": row["close_event_uid"],
            "trade_id": trade_id,
            "order_id": order_id,
            "gateway_lane_id": _text(row["gateway_lane_id"]),
            "client_order_id": _text(row["client_order_id"]),
            "accepted_ts_utc": row["accepted_ts_utc"],
            "close_ts_utc": row["close_ts_utc"],
            "realized_pl_jpy": row["close_realized_pl_jpy"],
        }
        event = {
            "event_uid": (
                f"ledger_reconcile:trader_entry_broker_close:{trade_id}:{order_id or ''}:"
                f"{GATEWAY_TRADE_CLOSE_RECONCILED}"
            ),
            "ts_utc": str(row["close_ts_utc"] or row["accepted_ts_utc"] or now),
            "source": "ledger_reconcile",
            "event_type": GATEWAY_TRADE_CLOSE_RECONCILED,
            "lane_id": _text(row["gateway_lane_id"]),
            "order_id": order_id,
            "trade_id": trade_id,
            "client_order_id": _text(row["client_order_id"]),
            "pair": _text(row["close_pair"]) or _text(row["accepted_pair"]) or _text(row["entry_pair"]),
            "side": _text(row["entry_side"]),
            "units": None,
            "price": None,
            "tp": None,
            "sl": None,
            "realized_pl_jpy": None,
            "financing_jpy": None,
            "exit_reason": "BROKER_TRADE_CLOSE_TRADER_ENTRY_RECONCILED",
            "oanda_transaction_id": _text(row["accepted_oanda_transaction_id"]),
            "related_transaction_ids_json": json.dumps(related_ids, sort_keys=True),
            "raw_json": _json(raw),
            "inserted_at_utc": now,
        }
        if _insert_event(conn, event):
            inserted += 1
    return inserted


def _close_reconcile_timestamps_match(
    gpt_ts: datetime | None,
    accepted_ts: datetime | None,
    close_ts: datetime | None,
) -> bool:
    if gpt_ts is None or accepted_ts is None:
        return False
    accept_delay = (accepted_ts - gpt_ts).total_seconds()
    if accept_delay < -GPT_CLOSE_RECONCILE_CLOCK_SKEW_SECONDS:
        return False
    if accept_delay > GPT_CLOSE_RECONCILE_MAX_ACCEPT_DELAY_SECONDS:
        return False
    if close_ts is not None:
        close_delay = (close_ts - accepted_ts).total_seconds()
        if close_delay < -GPT_CLOSE_RECONCILE_CLOCK_SKEW_SECONDS:
            return False
        if close_delay > GPT_CLOSE_RECONCILE_MAX_ACCEPT_DELAY_SECONDS:
            return False
    return True


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        prefix, suffix = text.split(".", 1)
        tz_index = next((index for index, char in enumerate(suffix) if char in "+-"), None)
        if tz_index is None:
            fraction = suffix
            tz_suffix = ""
        else:
            fraction = suffix[:tz_index]
            tz_suffix = suffix[tz_index:]
        if len(fraction) > 6:
            text = f"{prefix}.{fraction[:6]}{tz_suffix}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def _backfill_gateway_position_order_ids(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT event_uid, raw_json
        FROM execution_events
        WHERE source = 'gateway'
          AND event_type IN (
            'GATEWAY_TRADE_CLOSE_SENT',
            'GATEWAY_PROTECTION_REPLACE_SENT',
            'GATEWAY_POSITION_ACTION_BLOCKED',
            'GATEWAY_POSITION_ACTION_STAGED'
          )
          AND (order_id IS NULL OR order_id = '')
        """
    ).fetchall()
    updated = 0
    for row in rows:
        try:
            payload = json.loads(str(row["raw_json"] or "{}"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        order_id = _response_order_id(response)
        if not order_id:
            continue
        cur = conn.execute(
            "UPDATE execution_events SET order_id = ? WHERE event_uid = ? AND (order_id IS NULL OR order_id = '')",
            (order_id, row["event_uid"]),
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


def _predictive_scout_budget_keys(
    payload: Any,
    *,
    fallback: str,
) -> set[str]:
    """Return one stable daily-budget identity per SCOUT payload.

    Mixed-version ledgers may contain a SENT gateway receipt without a signal
    claim row.  Prefer exact signal, then experiment, then broker client-order
    id so the receipt/event/claim copies of one POST deduplicate.  A payload
    with only the legacy SCOUT marker still consumes a conservative fallback
    slot rather than disappearing from the atomic daily cap.
    """

    scout_dicts: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("predictive_scout") is True:
                scout_dicts.append(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    if not scout_dicts:
        return set()
    signal_ids = {
        str(item.get("predictive_scout_signal_id") or "").strip()
        for item in scout_dicts
        if str(item.get("predictive_scout_signal_id") or "").strip()
    }
    if signal_ids:
        return {f"signal:{signal_id}" for signal_id in signal_ids}
    experiment_ids = {
        str(item.get("predictive_scout_experiment_id") or "").strip()
        for item in scout_dicts
        if str(item.get("predictive_scout_experiment_id") or "").strip()
    }
    if experiment_ids:
        return {f"experiment:{experiment_id}" for experiment_id in experiment_ids}
    client_ids: set[str] = set()

    def collect_client_ids(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("clientExtensions", "tradeClientExtensions"):
                extension = value.get(key)
                if isinstance(extension, dict):
                    client_id = str(extension.get("id") or "").strip()
                    if client_id:
                        client_ids.add(client_id)
            for nested in value.values():
                collect_client_ids(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_client_ids(nested)

    collect_client_ids(payload)
    if client_ids:
        return {f"client:{client_id}" for client_id in client_ids}
    if isinstance(payload, dict):
        generated_at = str(payload.get("generated_at_utc") or "").strip()
        if generated_at:
            lane_id = str(payload.get("lane_id") or "").strip()
            return {f"legacy-generated:{generated_at}:{lane_id}"}
    return {fallback}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
