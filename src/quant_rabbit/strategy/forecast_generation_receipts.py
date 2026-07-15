"""Append-only outcome receipts for forecast-to-intent generation cycles."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


RECEIPT_FILENAME = "forecast_generation_receipts.jsonl"
RECEIPT_CONTRACT = "QR_FORECAST_GENERATION_RECEIPT_V1"
RECEIPT_STATUSES = frozenset({"COMMITTED", "ABORTED"})


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_pairs(values: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(value or "").strip().upper()
            for value in values
            if str(value or "").strip()
        }
    )


def forecast_pairs_for_cycle(data_root: Path, cycle_id: str) -> list[str]:
    """Return the distinct history pairs written for one cycle."""

    path = data_root / "forecast_history.jsonl"
    if not path.is_file() or not cycle_id:
        return []
    pairs: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    try:
                        row = json.loads(raw_line)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if not isinstance(row, Mapping):
                        continue
                    if str(row.get("cycle_id") or "") != cycle_id:
                        continue
                    pair = str(row.get("pair") or "").strip().upper()
                    if pair:
                        pairs.add(pair)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, UnicodeDecodeError):
        return []
    return sorted(pairs)


def record_forecast_generation_receipt(
    *,
    data_root: Path,
    cycle_id: str,
    status: str,
    expected_pairs: Iterable[Any],
    order_intents_path: Path,
    error_type: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one content-addressed generation outcome.

    The receipt is diagnostic/calibration lineage only.  It grants no live
    permission and does not alter immutable forecast or projection rows.
    """

    normalized_status = str(status or "").strip().upper()
    if normalized_status not in RECEIPT_STATUSES:
        raise ValueError("forecast generation receipt status is invalid")
    normalized_cycle_id = str(cycle_id or "").strip()
    if not normalized_cycle_id or len(normalized_cycle_id) > 1024:
        raise ValueError("forecast generation receipt cycle_id is invalid")
    recorded_at = now or datetime.now(timezone.utc)
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    else:
        recorded_at = recorded_at.astimezone(timezone.utc)
    output_sha256 = None
    if normalized_status == "COMMITTED":
        output_sha256 = hashlib.sha256(order_intents_path.read_bytes()).hexdigest()
    body = {
        "contract": RECEIPT_CONTRACT,
        "status": normalized_status,
        "cycle_id": normalized_cycle_id,
        "recorded_at_utc": recorded_at.isoformat(),
        "expected_pairs": _normalized_pairs(expected_pairs),
        "recorded_forecast_pairs": forecast_pairs_for_cycle(
            data_root,
            normalized_cycle_id,
        ),
        "order_intents_path": str(order_intents_path),
        "order_intents_sha256": output_sha256,
        "error_type": (
            str(error_type or "").strip()[:128]
            if normalized_status == "ABORTED"
            else None
        ),
        "learning_eligible": normalized_status == "COMMITTED",
        "live_permission": False,
    }
    receipt = {**body, "receipt_sha256": _canonical_json_sha256(body)}
    path = data_root / RECEIPT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return receipt


def forecast_generation_status_by_cycle(data_root: Path) -> dict[str, str]:
    """Return the latest valid generation status for every receipted cycle."""

    path = data_root / RECEIPT_FILENAME
    if not path.is_file():
        return {}
    statuses: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    try:
                        row = json.loads(raw_line)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
                    if not isinstance(row, Mapping):
                        continue
                    material = {
                        str(key): value
                        for key, value in row.items()
                        if key != "receipt_sha256"
                    }
                    if (
                        row.get("contract") != RECEIPT_CONTRACT
                        or row.get("status") not in RECEIPT_STATUSES
                        or row.get("receipt_sha256")
                        != _canonical_json_sha256(material)
                        or row.get("live_permission") is not False
                    ):
                        continue
                    cycle_id = str(row.get("cycle_id") or "").strip()
                    if cycle_id:
                        statuses[cycle_id] = str(row["status"])
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, UnicodeDecodeError, TypeError, ValueError):
        return {}
    return statuses


def aborted_forecast_generation_cycles(data_root: Path) -> frozenset[str]:
    return frozenset(
        cycle_id
        for cycle_id, status in forecast_generation_status_by_cycle(data_root).items()
        if status == "ABORTED"
    )
