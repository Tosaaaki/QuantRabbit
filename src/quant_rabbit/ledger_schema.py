from __future__ import annotations

import sqlite3
from dataclasses import dataclass


# SQLite schema setup can fail immediately when several fresh worker processes
# open the same new ledger.  These short engineering backoffs serialize that
# one-time work without waiting anywhere near an order or forecast TTL.
SQLITE_SCHEMA_INIT_RETRY_DELAYS_SECONDS = (0.05, 0.10, 0.20, 0.40, 0.80)


@dataclass(frozen=True)
class _IndexDefinition:
    name: str
    table: str
    required_columns: frozenset[str]
    ddl: str


PROFITABILITY_ACCEPTANCE_INDEX_NAMES = (
    "idx_execution_events_type_trade_ts",
    "idx_execution_events_type_order_trade",
    "idx_verification_close_gate_subject_ts",
)

_PROFITABILITY_ACCEPTANCE_INDEXES = (
    _IndexDefinition(
        name=PROFITABILITY_ACCEPTANCE_INDEX_NAMES[0],
        table="execution_events",
        required_columns=frozenset({"event_type", "trade_id", "ts_utc"}),
        ddl="""
            CREATE INDEX IF NOT EXISTS idx_execution_events_type_trade_ts
            ON execution_events(event_type, trade_id, ts_utc)
        """,
    ),
    _IndexDefinition(
        name=PROFITABILITY_ACCEPTANCE_INDEX_NAMES[1],
        table="execution_events",
        required_columns=frozenset({"event_type", "order_id", "trade_id"}),
        ddl="""
            CREATE INDEX IF NOT EXISTS idx_execution_events_type_order_trade
            ON execution_events(event_type, order_id, trade_id)
        """,
    ),
    _IndexDefinition(
        name=PROFITABILITY_ACCEPTANCE_INDEX_NAMES[2],
        table="verification_observations",
        required_columns=frozenset(
            {"check_name", "subject_id", "ts_utc", "status"}
        ),
        ddl="""
            CREATE INDEX IF NOT EXISTS idx_verification_close_gate_subject_ts
            ON verification_observations(check_name, subject_id, ts_utc, status)
            WHERE check_name = 'close_gate_evidence'
        """,
    ),
)


def ensure_profitability_acceptance_indexes(
    conn: sqlite3.Connection,
) -> tuple[str, ...]:
    """Install the close-audit indexes atomically when their tables support them.

    Execution and verification ledgers share one SQLite database but can be
    initialized by separate processes.  The read-only fast path avoids taking a
    write lock once all applicable indexes exist.  A missing-index migration
    takes one IMMEDIATE transaction, rechecks after acquiring the lock, and
    either commits every missing index or rolls the whole migration back.

    Legacy/unit-test databases may expose only a subset of the current tables or
    columns.  Those definitions are skipped without a durable marker so a later
    full-schema initializer can install them.
    """

    if conn.in_transaction:
        raise sqlite3.OperationalError(
            "profitability acceptance index migration requires a clean transaction"
        )

    applicable = _missing_applicable_indexes(conn)
    if not applicable:
        return ()

    conn.execute("BEGIN IMMEDIATE")
    try:
        # Another initializer may have completed while this connection waited
        # for the write lock.  Rechecking makes concurrent cold starts harmless.
        applicable = _missing_applicable_indexes(conn)
        for definition in applicable:
            conn.execute(definition.ddl)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return tuple(definition.name for definition in applicable)


def _missing_applicable_indexes(
    conn: sqlite3.Connection,
) -> tuple[_IndexDefinition, ...]:
    existing_indexes = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
    }
    tables = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    columns_by_table: dict[str, frozenset[str]] = {}
    missing: list[_IndexDefinition] = []
    for definition in _PROFITABILITY_ACCEPTANCE_INDEXES:
        if definition.name in existing_indexes or definition.table not in tables:
            continue
        columns = columns_by_table.get(definition.table)
        if columns is None:
            columns = frozenset(
                str(row[1])
                for row in conn.execute(
                    f"PRAGMA table_info({definition.table})"
                ).fetchall()
            )
            columns_by_table[definition.table] = columns
        if definition.required_columns.issubset(columns):
            missing.append(definition)
    return tuple(missing)
