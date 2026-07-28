from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS_HASH = "0" * 64


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class LedgerIntegrityError(RuntimeError):
    pass


class CryptoLedger:
    """Append-only, hash-chained decision/order/fill/PnL event ledger."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.verify()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS crypto_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_crypto_events_type "
                "ON crypto_events(event_type, sequence)"
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS crypto_events_no_update
                BEFORE UPDATE ON crypto_events
                BEGIN SELECT RAISE(ABORT, 'crypto ledger is append-only'); END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS crypto_events_no_delete
                BEFORE DELETE ON crypto_events
                BEGIN SELECT RAISE(ABORT, 'crypto ledger is append-only'); END
                """
            )

    def append(
        self,
        event_type: str,
        entity_id: str,
        payload: dict[str, Any],
        *,
        dedupe_key: str,
        event_id: str | None = None,
        created_at: datetime | None = None,
    ) -> tuple[str, bool]:
        payload_json = _canonical(payload)
        payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        event_id = event_id or str(uuid.uuid4())
        created = (created_at or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT event_id, payload_sha256 FROM crypto_events WHERE dedupe_key=?",
                (dedupe_key,),
            ).fetchone()
            if existing:
                if existing["payload_sha256"] != payload_sha:
                    raise LedgerIntegrityError(
                        f"dedupe conflict for {dedupe_key}: payload changed"
                    )
                return str(existing["event_id"]), False
            previous = conn.execute(
                "SELECT event_hash FROM crypto_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            prev_hash = str(previous["event_hash"]) if previous else GENESIS_HASH
            event_hash = hashlib.sha256(
                "|".join(
                    [
                        prev_hash,
                        event_id,
                        dedupe_key,
                        event_type,
                        entity_id,
                        payload_sha,
                        created,
                    ]
                ).encode("utf-8")
            ).hexdigest()
            conn.execute(
                """
                INSERT INTO crypto_events(
                    event_id, dedupe_key, event_type, entity_id, payload_json,
                    payload_sha256, prev_hash, event_hash, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    dedupe_key,
                    event_type,
                    entity_id,
                    payload_json,
                    payload_sha,
                    prev_hash,
                    event_hash,
                    created,
                ),
            )
        return event_id, True

    def events(self, event_type: str | None = None) -> Iterator[dict[str, Any]]:
        query = "SELECT * FROM crypto_events"
        params: tuple[str, ...] = ()
        if event_type:
            query += " WHERE event_type=?"
            params = (event_type,)
        query += " ORDER BY sequence"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            yield item

    def latest_payload(self, event_type: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM crypto_events "
                "WHERE event_type=? ORDER BY sequence DESC LIMIT 1",
                (event_type,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def payload_for_dedupe(self, dedupe_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM crypto_events WHERE dedupe_key=?",
                (dedupe_key,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def metadata_for_dedupe(self, dedupe_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sequence, event_id, event_hash, prev_hash, "
                "created_at_utc FROM crypto_events WHERE dedupe_key=?",
                (dedupe_key,),
            ).fetchone()
        return dict(row) if row else None

    def verify(self) -> dict[str, Any]:
        previous = GENESIS_HASH
        count = 0
        for row in self.events():
            payload_json = _canonical(row["payload"])
            payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            if payload_sha != row["payload_sha256"]:
                raise LedgerIntegrityError(
                    f"payload digest mismatch at sequence={row['sequence']}"
                )
            if row["prev_hash"] != previous:
                raise LedgerIntegrityError(
                    f"chain mismatch at sequence={row['sequence']}"
                )
            expected = hashlib.sha256(
                "|".join(
                    [
                        previous,
                        row["event_id"],
                        row["dedupe_key"],
                        row["event_type"],
                        row["entity_id"],
                        payload_sha,
                        row["created_at_utc"],
                    ]
                ).encode("utf-8")
            ).hexdigest()
            if expected != row["event_hash"]:
                raise LedgerIntegrityError(
                    f"event digest mismatch at sequence={row['sequence']}"
                )
            previous = expected
            count += 1
        return {"valid": True, "event_count": count, "head_hash": previous}
