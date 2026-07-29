from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from .ledger import CryptoLedger

TRADE_COLUMNS = (
    "operation_id",
    "trade_id",
    "run_id",
    "paper_mode",
    "pair",
    "side",
    "opened_at_utc",
    "closed_at_utc",
    "entry_price",
    "exit_price",
    "quantity",
    "entry_notional_jpy",
    "gross_pnl_jpy",
    "fees_jpy",
    "spread_cost_jpy",
    "adverse_cost_jpy",
    "funding_interest_jpy",
    "net_pnl_jpy",
    "holding_ms",
    "exit_reason",
    "strategy",
    "regime",
    "guardian",
    "ledger_sequence",
    "ledger_event_hash",
    "ledger_prev_hash",
    "authority",
    "live_permission",
)

_SENTINEL = object()


def trade_operation_id(trade_id: str) -> str:
    return hashlib.sha256(f"crypto-paper-trade|{trade_id}".encode()).hexdigest()


class AsyncTradeOutbox:
    """Non-blocking producer queue with an append-only JSONL consumer thread."""

    def __init__(self, path: Path, ledger: CryptoLedger) -> None:
        self.path = path
        self.ledger = ledger
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._known, self._last_ledger_sequence = self._load_known()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._queue: queue.SimpleQueue[dict[str, Any] | object] = (
            queue.SimpleQueue()
        )
        self._written = 0
        self._last_error: str | None = None
        self._thread = threading.Thread(
            target=self._consume,
            name=f"crypto-trade-outbox-{path.parent.name}",
            daemon=True,
        )
        self._thread.start()
        self.recover_from_ledger()

    def _load_known(self) -> tuple[set[str], int]:
        known: set[str] = set()
        last_ledger_sequence = 0
        if not self.path.exists():
            return known, last_ledger_sequence
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"malformed trade outbox line {line_number}"
                    ) from exc
                operation_id = str(payload.get("operation_id", ""))
                if not operation_id or operation_id in known:
                    raise RuntimeError(
                        f"invalid trade outbox operation at line {line_number}"
                    )
                try:
                    ledger_sequence = int(payload["ledger_sequence"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"invalid trade outbox ledger sequence at line "
                        f"{line_number}"
                    ) from exc
                if ledger_sequence <= last_ledger_sequence:
                    raise RuntimeError(
                        f"non-increasing trade outbox ledger sequence at line "
                        f"{line_number}"
                    )
                known.add(operation_id)
                last_ledger_sequence = ledger_sequence
        return known, last_ledger_sequence

    def enqueue(self, payload: dict[str, Any]) -> str:
        trade_id = str(payload["trade_id"])
        operation_id = trade_operation_id(trade_id)
        event = {
            **payload,
            "operation_id": operation_id,
            "authority": "NONE",
            "live_permission": False,
        }
        missing = [column for column in TRADE_COLUMNS if column not in event]
        if missing:
            raise ValueError(f"trade outbox missing columns: {missing}")
        with self._lock:
            if operation_id in self._known or operation_id in self._pending:
                return operation_id
            self._pending.add(operation_id)
        self._queue.put(event)
        return operation_id

    def recover_from_ledger(self) -> int:
        recovered = 0
        for row in self.ledger.events_after(
            "PAPER_TRADE_CLOSED",
            self._last_ledger_sequence,
        ):
            payload = dict(row["payload"])
            payload.update(
                {
                    "ledger_sequence": row["sequence"],
                    "ledger_event_hash": row["event_hash"],
                    "ledger_prev_hash": row["prev_hash"],
                }
            )
            operation_id = trade_operation_id(str(payload["trade_id"]))
            with self._lock:
                already_known = (
                    operation_id in self._known
                    or operation_id in self._pending
                )
            if not already_known:
                self.enqueue(payload)
                recovered += 1
        return recovered

    def flush(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.005)
        return False

    def close(self, timeout_sec: float = 5.0) -> bool:
        flushed = self.flush(timeout_sec)
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=max(0.0, timeout_sec))
        return flushed and not self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self.path),
                "known_operations": len(self._known),
                "pending_operations": len(self._pending),
                "written_this_process": self._written,
                "writer_alive": self._thread.is_alive(),
                "last_error": self._last_error,
            }

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                return
            event = dict(item)
            operation_id = str(event["operation_id"])
            try:
                encoded = json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                with self._lock:
                    self._known.add(operation_id)
                    self._last_ledger_sequence = max(
                        self._last_ledger_sequence,
                        int(event["ledger_sequence"]),
                    )
                    self._pending.discard(operation_id)
                    self._written += 1
            except Exception as exc:
                with self._lock:
                    self._last_error = type(exc).__name__
                time.sleep(0.5)
                self._queue.put(event)
