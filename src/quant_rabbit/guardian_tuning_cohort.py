"""Canonical forward-only cohorts for guardian bot tuning.

The reviewed candidate is fixed before any eligible entry.  Signal values are
then recovered from the append-only entry-thesis and forecast histories, while
post-cost outcomes and entry units come from one SQLite snapshot.  Callers
cannot supply signal values, outcome rows, or a favorable cutoff.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from quant_rabbit.capture_economics import (
    read_attributed_net_outcomes,
    read_attributed_system_entries,
)
from quant_rabbit.guardian_tuning_evaluator import (
    COHORT_GENERATOR_NAME,
    MIN_FORWARD_SAMPLE_COUNT,
    SUPPORTED_THRESHOLD_PARAMETERS,
    source_semantic_digest,
    validate_source,
)


MAX_SIGNAL_LOG_BYTES = 16 * 1024 * 1024
_COVERAGE_KEY = "oanda_transaction_coverage_start_utc"
_LAST_TRANSACTION_KEY = "last_oanda_transaction_id"


def _utc(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is required")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Python 3.9 accepts microseconds but not OANDA's nanoseconds.
    if "." in text:
        head, tail = text.split(".", 1)
        offset_at = next(
            (index for index, char in enumerate(tail) if char in "+-"),
            len(tail),
        )
        fraction, offset = tail[:offset_at], tail[offset_at:]
        text = f"{head}.{fraction[:6].ljust(6, '0')}{offset}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _rfc3339_utc_order_key(value: object) -> tuple[datetime, int]:
    """Preserve OANDA nanoseconds when ordering the immutable first twenty."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp is required")
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})",
        value.strip(),
    )
    if match is None:
        raise ValueError("timestamp must be RFC3339 with timezone")
    prefix, fraction, offset = match.groups()
    try:
        parsed = datetime.fromisoformat(
            f"{prefix}{'+00:00' if offset == 'Z' else offset}"
        )
    except ValueError:
        raise ValueError("timestamp is invalid") from None
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return (
        parsed.astimezone(timezone.utc).replace(microsecond=0),
        int((fraction or "0").ljust(9, "0")),
    )


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{field} must be numeric") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _bounded_prefix(path: Path, *, prefix_bytes: int | None = None) -> tuple[bytes, int, str]:
    resolved = path.resolve(strict=True)
    size = resolved.stat().st_size
    length = size if prefix_bytes is None else prefix_bytes
    if length <= 0 or length > size or length > MAX_SIGNAL_LOG_BYTES:
        raise ValueError(f"signal log prefix is outside 1..{MAX_SIGNAL_LOG_BYTES} bytes")
    with resolved.open("rb") as handle:
        raw = handle.read(length)
    if len(raw) != length or not raw.endswith(b"\n"):
        raise ValueError("signal log prefix must end at a complete JSONL record")
    return raw, length, hashlib.sha256(raw).hexdigest()


def _jsonl_records(raw: bytes, *, label: str) -> list[tuple[dict[str, Any], str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    records: list[tuple[dict[str, Any], str]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} row {index} is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{label} row {index} must be an object")
        records.append((payload, hashlib.sha256(line.encode("utf-8")).hexdigest()))
    return records


@contextmanager
def _ledger_snapshot(
    ledger_path: Path,
    *,
    max_rowid: int | None = None,
    last_transaction_id: str | None = None,
) -> Iterator[Path]:
    """Yield a stable private SQLite snapshot, optionally cut at a watermark."""

    with tempfile.TemporaryDirectory(prefix="qr-guardian-cohort-") as tmp:
        snapshot = Path(tmp) / "execution_ledger.db"
        source = sqlite3.connect(f"file:{ledger_path.resolve()}?mode=ro", uri=True)
        destination = sqlite3.connect(snapshot)
        try:
            source.backup(destination)
        finally:
            source.close()
        try:
            if max_rowid is not None:
                destination.execute(
                    "DELETE FROM execution_events WHERE rowid > ?",
                    (max_rowid,),
                )
                if last_transaction_id is not None:
                    has_oanda_transactions = destination.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='oanda_transactions'"
                    ).fetchone()
                    if has_oanda_transactions is not None:
                        destination.execute(
                            """
                            DELETE FROM oanda_transactions
                            WHERE CAST(transaction_id AS INTEGER) > CAST(? AS INTEGER)
                            """,
                            (last_transaction_id,),
                        )
                    destination.execute(
                        "UPDATE sync_state SET value=? WHERE key=?",
                        (last_transaction_id, _LAST_TRANSACTION_KEY),
                    )
                destination.commit()
        finally:
            destination.close()
        yield snapshot


def _ledger_identity(ledger_path: Path) -> tuple[int, str, str, str]:
    with sqlite3.connect(f"file:{ledger_path.resolve()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        max_rowid = int(
            conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM execution_events").fetchone()[0]
        )
        if max_rowid <= 0:
            raise ValueError("execution ledger has no canonical events")
        sync = dict(
            conn.execute(
                "SELECT key, value FROM sync_state WHERE key IN (?, ?)",
                (_COVERAGE_KEY, _LAST_TRANSACTION_KEY),
            ).fetchall()
        )
        coverage = str(sync.get(_COVERAGE_KEY) or "")
        last_id = str(sync.get(_LAST_TRANSACTION_KEY) or "")
        _utc(coverage)
        if not last_id.isdigit():
            raise ValueError("execution ledger transaction watermark is unavailable")
        digest = hashlib.sha256()
        columns = [
            str(row[1])
            for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
        ]
        query = "SELECT rowid, " + ", ".join(f'"{name}"' for name in columns)
        query += " FROM execution_events WHERE rowid <= ? ORDER BY rowid"
        for row in conn.execute(query, (max_rowid,)):
            digest.update(
                json.dumps(
                    list(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return max_rowid, digest.hexdigest(), coverage, last_id


def _execution_ledger_anchor_from_identity(
    identity: tuple[int, str, str, str],
    *,
    captured_at_utc: object,
) -> dict[str, Any]:
    rowid, ledger_sha, coverage, last_id = identity
    captured_at = _utc(captured_at_utc)
    coverage_at = _utc(coverage)
    if (
        rowid <= 0
        or len(ledger_sha) != 64
        or any(char not in "0123456789abcdef" for char in ledger_sha)
        or not last_id.isdigit()
    ):
        raise ValueError("execution ledger anchor identity is invalid")
    return {
        "ledger_rowid_watermark": rowid,
        "ledger_prefix_sha256": ledger_sha,
        "execution_ledger_coverage_start_utc": coverage_at.isoformat(),
        "last_oanda_transaction_id": last_id,
        "captured_at_utc": captured_at.isoformat(),
    }


def current_execution_ledger_anchor(*, ledger_path: Path) -> dict[str, Any]:
    """Capture one stable activation boundary from canonical ledger truth."""

    with _ledger_snapshot(ledger_path) as snapshot:
        captured_at = datetime.now(timezone.utc)
        identity = _ledger_identity(snapshot)
    return _execution_ledger_anchor_from_identity(
        identity,
        captured_at_utc=captured_at,
    )


def validate_execution_ledger_anchor(
    *,
    ledger_path: Path,
    anchor: object,
) -> dict[str, Any]:
    """Revalidate an immutable activation prefix against the current ledger."""

    required = {
        "ledger_rowid_watermark",
        "ledger_prefix_sha256",
        "execution_ledger_coverage_start_utc",
        "last_oanda_transaction_id",
        "captured_at_utc",
    }
    if not isinstance(anchor, dict) or set(anchor) != required:
        raise ValueError("activation ledger anchor schema is invalid")
    rowid = anchor.get("ledger_rowid_watermark")
    ledger_sha = str(anchor.get("ledger_prefix_sha256") or "")
    coverage = str(anchor.get("execution_ledger_coverage_start_utc") or "")
    last_id = str(anchor.get("last_oanda_transaction_id") or "")
    captured_at = str(anchor.get("captured_at_utc") or "")
    if (
        isinstance(rowid, bool)
        or not isinstance(rowid, int)
        or rowid <= 0
        or len(ledger_sha) != 64
        or any(char not in "0123456789abcdef" for char in ledger_sha)
        or not last_id.isdigit()
    ):
        raise ValueError("activation ledger anchor values are invalid")
    coverage_at = _utc(coverage)
    _utc(captured_at)

    with _ledger_snapshot(ledger_path) as current_snapshot:
        current_identity = _ledger_identity(current_snapshot)
        current_rowid, _, current_coverage, current_last_id = current_identity
        if (
            current_rowid < rowid
            or int(current_last_id) < int(last_id)
            or _utc(current_coverage) != coverage_at
        ):
            raise ValueError("activation ledger anchor is ahead of current canonical truth")
        with _ledger_snapshot(
            current_snapshot,
            max_rowid=rowid,
            last_transaction_id=last_id,
        ) as prefix_snapshot:
            prefix_identity = _ledger_identity(prefix_snapshot)
    rebuilt = _execution_ledger_anchor_from_identity(
        prefix_identity,
        captured_at_utc=captured_at,
    )
    if rebuilt != anchor:
        raise ValueError("activation ledger anchor prefix no longer matches")
    return dict(anchor)


def current_canonical_forward_source_tip(
    *,
    ledger_path: Path,
    entry_thesis_path: Path,
    forecast_history_path: Path,
) -> dict[str, Any]:
    """Read the authoritative ledger/log tips used only at initial prepare."""

    with _ledger_snapshot(ledger_path) as snapshot:
        rowid, ledger_sha, _, last_id = _ledger_identity(snapshot)
    _, thesis_length, thesis_sha = _bounded_prefix(entry_thesis_path)
    _, forecast_length, forecast_sha = _bounded_prefix(forecast_history_path)
    return {
        "last_oanda_transaction_id": last_id,
        "ledger_rowid_watermark": rowid,
        "ledger_prefix_sha256": ledger_sha,
        "entry_thesis_prefix_bytes": thesis_length,
        "entry_thesis_prefix_sha256": thesis_sha,
        "forecast_history_prefix_bytes": forecast_length,
        "forecast_history_prefix_sha256": forecast_sha,
    }


def build_post_activation_monitor_cohort(
    *,
    ledger_path: Path,
    lane_id: str,
    activated_at_utc: object,
    activation_ledger_anchor: object,
    ledger_rowid_watermark: int | None = None,
    expected_last_transaction_id: str | None = None,
) -> dict[str, Any]:
    """Freeze the first 20 exact-lane entries after activation and their outcomes."""

    pair, side, method, vehicle = _lane_parts(lane_id)
    if vehicle not in {"MARKET", "LIMIT", "STOP"}:
        raise ValueError("post-activation monitor requires an exact five-part lane")
    activated_at = _utc(activated_at_utc)
    activated_at_key = _rfc3339_utc_order_key(str(activated_at_utc or ""))
    anchor = validate_execution_ledger_anchor(
        ledger_path=ledger_path,
        anchor=activation_ledger_anchor,
    )
    anchor_rowid = int(anchor["ledger_rowid_watermark"])
    with _ledger_snapshot(
        ledger_path,
        max_rowid=ledger_rowid_watermark,
        last_transaction_id=expected_last_transaction_id,
    ) as snapshot:
        rowid, ledger_sha, coverage, last_id = _ledger_identity(snapshot)
        entries = read_attributed_system_entries(snapshot)
        outcomes = read_attributed_net_outcomes(snapshot)
        if entries is None or outcomes is None:
            raise ValueError("canonical post-activation ledger is unreadable")
        lane_entries = [
            entry
            for entry in entries
            if entry.pair == pair
            and entry.side == side
            and entry.method == method
            and entry.entry_vehicle == vehicle
            and entry.canonical_lane_id == lane_id
        ]
        if any(
            entry.broker_time_consistent is not True
            or not entry.broker_entry_ts_utc
            for entry in lane_entries
        ):
            raise ValueError("post-activation entry broker timestamp is unverified")
        first_entries = sorted(
            (
                entry
                for entry in lane_entries
                if entry.ledger_rowid > anchor_rowid
                and _rfc3339_utc_order_key(entry.broker_entry_ts_utc)
                > activated_at_key
            ),
            key=lambda entry: (
                _rfc3339_utc_order_key(entry.broker_entry_ts_utc),
                entry.ledger_rowid,
            ),
        )[:MIN_FORWARD_SAMPLE_COUNT]
        identity = {
            "lane_id": lane_id,
            "pair": pair,
            "side": side,
            "method": method,
            "vehicle": vehicle,
            "activated_at_utc": activated_at.isoformat(),
            "activation_ledger_anchor": dict(anchor),
            "ledger_rowid_watermark": rowid,
            "ledger_prefix_sha256": ledger_sha,
            "last_oanda_transaction_id": last_id,
            "execution_ledger_coverage_start_utc": coverage,
            "first_trade_ids": [entry.trade_id for entry in first_entries],
        }
        cohort_id = hashlib.sha256(
            json.dumps(
                identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if len(first_entries) < MIN_FORWARD_SAMPLE_COUNT:
            return {
                "schema_version": 1,
                "status": "WAITING_FOR_FIRST_20_ENTRIES",
                "cohort_id": cohort_id,
                **identity,
                "entry_count": len(first_entries),
                "required_entry_count": MIN_FORWARD_SAMPLE_COUNT,
            }
        samples: list[dict[str, Any]] = []
        unresolved: list[str] = []
        for entry in first_entries:
            matches = [
                outcome
                for outcome in outcomes
                if outcome.trade_id == entry.trade_id
                and outcome.pair == pair
                and outcome.side == side
                and outcome.method == method
                and _lane_matches(
                    lane_id,
                    outcome.lane_id,
                    actual_vehicle=outcome.entry_vehicle,
                )
            ]
            if len(matches) != 1:
                unresolved.append(entry.trade_id)
                continue
            outcome = matches[0]
            if (
                entry.broker_time_consistent is not True
                or outcome.broker_time_consistent is not True
                or not entry.broker_entry_ts_utc
                or not outcome.broker_close_ts_utc
            ):
                raise ValueError("post-activation broker timestamps are unverified")
            entry_at = _rfc3339_utc_order_key(entry.broker_entry_ts_utc)
            closed_at = _rfc3339_utc_order_key(outcome.broker_close_ts_utc)
            if closed_at <= entry_at or not math.isfinite(float(entry.entry_units)):
                raise ValueError("post-activation entry/outcome timing or units are invalid")
            units = abs(float(entry.entry_units))
            if units <= 0.0:
                raise ValueError("post-activation entry units must be positive")
            net_jpy = float(outcome.realized_pl_jpy)
            normalized = net_jpy * 1000.0 / units
            if not math.isfinite(net_jpy) or not math.isfinite(normalized):
                raise ValueError("post-activation canonical outcome is not finite")
            samples.append(
                {
                    "trade_id": entry.trade_id,
                    "order_id": entry.order_id,
                    "entry_at_utc": entry.broker_entry_ts_utc,
                    "closed_at_utc": outcome.broker_close_ts_utc,
                    "entry_units": units,
                    "realized_net_jpy": net_jpy,
                    "net_jpy_per_1000_units": normalized,
                }
            )
        if unresolved:
            return {
                "schema_version": 1,
                "status": "WAITING_FOR_FIRST_20_RESOLUTIONS",
                "cohort_id": cohort_id,
                **identity,
                "entry_count": len(first_entries),
                "resolved_count": len(samples),
                "unresolved_trade_ids": unresolved,
                "required_resolved_count": MIN_FORWARD_SAMPLE_COUNT,
            }
    primary_metric = sum(
        float(sample["net_jpy_per_1000_units"]) for sample in samples
    ) / MIN_FORWARD_SAMPLE_COUNT
    return {
        "schema_version": 1,
        "status": "POST_ACTIVATION_COHORT_COMPLETE",
        "cohort_id": cohort_id,
        **identity,
        "sample_count": len(samples),
        "primary_metric": "net_jpy_per_1000_units_per_opportunity",
        "primary_metric_value": primary_metric,
        "samples": samples,
    }


def validate_post_activation_monitor_cohort(
    payload: dict[str, Any],
    *,
    ledger_path: Path,
) -> dict[str, Any]:
    """Rebuild the sealed prefix and prove current first-20 truth is unchanged."""

    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "POST_ACTIVATION_COHORT_COMPLETE"
    ):
        return {"status": "POST_ACTIVATION_COHORT_SCHEMA_INVALID"}
    try:
        rebuilt = build_post_activation_monitor_cohort(
            ledger_path=ledger_path,
            lane_id=str(payload.get("lane_id") or ""),
            activated_at_utc=payload.get("activated_at_utc"),
            activation_ledger_anchor=payload.get("activation_ledger_anchor"),
            ledger_rowid_watermark=int(payload.get("ledger_rowid_watermark")),
            expected_last_transaction_id=str(
                payload.get("last_oanda_transaction_id") or ""
            ),
        )
    except (OSError, OverflowError, TypeError, ValueError, sqlite3.Error) as exc:
        return {
            "status": "POST_ACTIVATION_COHORT_REBUILD_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if rebuilt != payload:
        return {"status": "POST_ACTIVATION_COHORT_MISMATCH"}
    try:
        current = build_post_activation_monitor_cohort(
            ledger_path=ledger_path,
            lane_id=str(payload.get("lane_id") or ""),
            activated_at_utc=payload.get("activated_at_utc"),
            activation_ledger_anchor=payload.get("activation_ledger_anchor"),
        )
    except (OSError, OverflowError, TypeError, ValueError, sqlite3.Error) as exc:
        return {
            "status": "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
            "current_status": "CURRENT_REBUILD_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }
    current_truth_fields = (
        "first_trade_ids",
        "sample_count",
        "primary_metric",
        "primary_metric_value",
        "samples",
    )
    changed_fields = [
        field
        for field in current_truth_fields
        if current.get(field) != payload.get(field)
    ]
    if (
        current.get("status") != "POST_ACTIVATION_COHORT_COMPLETE"
        or current.get("lane_id") != payload.get("lane_id")
        or current.get("activated_at_utc") != payload.get("activated_at_utc")
        or current.get("activation_ledger_anchor")
        != payload.get("activation_ledger_anchor")
        or changed_fields
    ):
        return {
            "status": "POST_ACTIVATION_COHORT_CURRENT_TRUTH_CHANGED",
            "current_status": current.get("status"),
            "changed_fields": changed_fields,
            "sealed_first_trade_ids": payload.get("first_trade_ids"),
            "current_first_trade_ids": current.get("first_trade_ids"),
        }
    return {
        "status": "VALID",
        "cohort_id": payload.get("cohort_id"),
        "primary_metric_value": payload.get("primary_metric_value"),
    }


def _lane_parts(lane_id: str) -> tuple[str, str, str, str]:
    parts = str(lane_id or "").split(":")
    if len(parts) not in {4, 5} or any(not part.strip() for part in parts):
        raise ValueError("lane id must bind desk, pair, side, and method")
    return (
        str(parts[1]).upper(),
        str(parts[2]).upper(),
        str(parts[3]).upper(),
        str(parts[4]).upper() if len(parts) >= 5 else "UNKNOWN",
    )


def _lane_matches(target: str, actual: str, *, actual_vehicle: str) -> bool:
    target_parts = _lane_parts(target)
    actual_parts = _lane_parts(actual)
    vehicle = str(actual_vehicle or "UNKNOWN").upper()
    return (
        target_parts[:3] == actual_parts[:3]
        and target_parts[3] != "UNKNOWN"
        and target_parts[3] == vehicle
        and actual_parts[3] in {"UNKNOWN", vehicle}
    )


def _review_contract(
    *,
    review: dict[str, Any],
    review_completed_at_utc: object,
    lane_id: str,
) -> dict[str, Any]:
    adjustments = review.get("proposed_adjustments")
    if (
        str(review.get("review_status") or "").upper() != "TEST_REQUIRED"
        or not isinstance(adjustments, list)
        or len(adjustments) != 1
    ):
        raise ValueError("one TEST_REQUIRED adjustment is required")
    adjustment = adjustments[0]
    if not isinstance(adjustment, dict):
        raise ValueError("review adjustment is invalid")
    pair = str(adjustment.get("pair") or "").upper()
    reviewed_lane_id = str(adjustment.get("lane_id") or "").strip()
    family = str(adjustment.get("bot_family") or "").lower()
    parameter = str(adjustment.get("parameter") or "")
    if (
        parameter not in SUPPORTED_THRESHOLD_PARAMETERS
        or not pair
        or not family
        or reviewed_lane_id != lane_id
        or len(reviewed_lane_id.split(":")) != 5
    ):
        raise ValueError("review identity is unsupported")
    lane_pair, _, _, _ = _lane_parts(lane_id)
    if lane_pair != pair:
        raise ValueError("review pair does not match the exact lane")
    completed = _utc(review_completed_at_utc)
    digest = hashlib.sha256(
        json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "pair": pair,
        "bot_family": family,
        "parameter": parameter,
        "lane_id": lane_id,
        "review_completed_at": completed,
        "review_digest_sha256": digest,
    }


def build_canonical_forward_cohort(
    *,
    ledger_path: Path,
    entry_thesis_path: Path,
    forecast_history_path: Path,
    review: dict[str, Any],
    review_completed_at_utc: object,
    lane_id: str,
    ledger_rowid_watermark: int | None = None,
    expected_last_transaction_id: str | None = None,
    entry_thesis_prefix_bytes: int | None = None,
    forecast_history_prefix_bytes: int | None = None,
) -> dict[str, Any]:
    """Freeze the first 20 post-review entries and require all 20 resolutions."""

    contract = _review_contract(
        review=review,
        review_completed_at_utc=review_completed_at_utc,
        lane_id=lane_id,
    )
    thesis_raw, thesis_length, thesis_sha = _bounded_prefix(
        entry_thesis_path,
        prefix_bytes=entry_thesis_prefix_bytes,
    )
    forecast_raw, forecast_length, forecast_sha = _bounded_prefix(
        forecast_history_path,
        prefix_bytes=forecast_history_prefix_bytes,
    )
    theses = _jsonl_records(thesis_raw, label="entry thesis ledger")
    forecasts = _jsonl_records(forecast_raw, label="forecast history")

    with _ledger_snapshot(
        ledger_path,
        max_rowid=ledger_rowid_watermark,
        last_transaction_id=expected_last_transaction_id,
    ) as snapshot:
        rowid, ledger_sha, coverage, last_id = _ledger_identity(snapshot)
        if ledger_rowid_watermark is not None and rowid != ledger_rowid_watermark:
            raise ValueError("ledger rowid watermark is unavailable")
        if expected_last_transaction_id is not None and last_id != expected_last_transaction_id:
            raise ValueError("ledger transaction watermark changed")
        outcomes = read_attributed_net_outcomes(snapshot)
        attributed_entries = read_attributed_system_entries(snapshot)
        if outcomes is None or attributed_entries is None:
            raise ValueError("canonical outcome ledger is unreadable")
        first_entries = [
            entry
            for entry in attributed_entries
            if entry.pair == contract["pair"]
            and entry.canonical_lane_id == lane_id
            and _utc(entry.entry_ts_utc) > contract["review_completed_at"]
        ][:MIN_FORWARD_SAMPLE_COUNT]
        if len(first_entries) < MIN_FORWARD_SAMPLE_COUNT:
            raise ValueError(
                f"forward cohort needs the first {MIN_FORWARD_SAMPLE_COUNT} entries; "
                f"found {len(first_entries)}"
            )
        first_trade_ids = [entry.trade_id for entry in first_entries]
        entries = {
            entry.trade_id: {
                "order_id": entry.order_id,
                "lane_id": entry.canonical_lane_id,
                "pair": entry.pair,
                "side": entry.side,
                "entry_at": _utc(entry.entry_ts_utc),
                "entry_units": entry.entry_units,
                "entry_vehicle": entry.entry_vehicle,
            }
            for entry in first_entries
        }
        outcome_matches = {
            trade_id: [
                outcome
                for outcome in outcomes
                if outcome.trade_id == trade_id
                and outcome.pair == contract["pair"]
                and _lane_matches(
                    lane_id,
                    outcome.lane_id,
                    actual_vehicle=outcome.entry_vehicle,
                )
            ]
            for trade_id in first_trade_ids
        }
        unresolved = [
            trade_id
            for trade_id, matches in outcome_matches.items()
            if len(matches) != 1
        ]
        if unresolved:
            raise ValueError(
                "first forward entry cohort is unresolved or non-canonical: "
                + ",".join(unresolved)
            )
        eligible = [outcome_matches[trade_id][0] for trade_id in first_trade_ids]

    if len(eligible) != MIN_FORWARD_SAMPLE_COUNT:
        raise ValueError(
            f"forward cohort needs exactly {MIN_FORWARD_SAMPLE_COUNT} resolved entries"
        )

    thesis_by_trade: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for payload, digest in theses:
        trade_id = str(payload.get("trade_id") or "")
        if trade_id:
            thesis_by_trade.setdefault(trade_id, []).append((payload, digest))
    forecast_by_key: dict[tuple[str, str, str], list[tuple[dict[str, Any], str]]] = {}
    for payload, digest in forecasts:
        key = (
            str(payload.get("pair") or "").upper(),
            str(payload.get("timestamp_utc") or ""),
            str(payload.get("cycle_id") or ""),
        )
        forecast_by_key.setdefault(key, []).append((payload, digest))

    samples: list[dict[str, Any]] = []
    for outcome in eligible:
        entry = entries[outcome.trade_id]
        matches = thesis_by_trade.get(outcome.trade_id, [])
        if len(matches) != 1:
            raise ValueError(
                f"trade {outcome.trade_id} must have exactly one entry-time thesis"
            )
        thesis, _ = matches[0]
        evidence = thesis.get("context_evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"trade {outcome.trade_id} thesis context is missing")
        if (
            str(thesis.get("pair") or "").upper() != contract["pair"]
            or str(evidence.get("order_id") or "") != entry["order_id"]
            or str(evidence.get("lane_id") or "") != entry["lane_id"]
            or _utc(thesis.get("timestamp_utc")) != entry["entry_at"]
        ):
            raise ValueError(f"trade {outcome.trade_id} thesis does not match entry truth")
        forecast_key = (
            contract["pair"],
            str(evidence.get("forecast_timestamp_utc") or ""),
            str(evidence.get("forecast_cycle_id") or ""),
        )
        forecast_matches = forecast_by_key.get(forecast_key, [])
        if len(forecast_matches) != 1:
            raise ValueError(
                f"trade {outcome.trade_id} needs one exact pre-entry forecast record"
            )
        forecast, forecast_digest = forecast_matches[0]
        signal_at = _utc(forecast.get("timestamp_utc"))
        confidence = _finite(forecast.get("confidence"), field="forecast confidence")
        thesis_confidence = _finite(
            thesis.get("forecast_confidence"),
            field="thesis forecast confidence",
        )
        if (
            signal_at >= entry["entry_at"]
            or not math.isclose(confidence, thesis_confidence, rel_tol=0.0, abs_tol=1e-12)
            or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(f"trade {outcome.trade_id} forecast is not entry-time truth")
        realized = _finite(outcome.realized_pl_jpy, field="realized net JPY")
        units = float(entry["entry_units"])
        samples.append(
            {
                "sample_id": f"trade:{outcome.trade_id}",
                "pair": contract["pair"],
                "bot_family": contract["bot_family"],
                "lane_id": lane_id,
                "trade_id": outcome.trade_id,
                "order_id": entry["order_id"],
                "entry_at_utc": entry["entry_at"].isoformat(),
                "closed_at_utc": _utc(outcome.ts_utc).isoformat(),
                "signal_observed_at_utc": signal_at.isoformat(),
                "signal_record_sha256": forecast_digest,
                "signal_value": confidence,
                "realized_net_jpy": realized,
                "entry_units": units,
                "net_jpy_per_1000_units": realized / abs(units) * 1000.0,
            }
        )

    cutoff = max(_utc(sample["closed_at_utc"]) for sample in samples)
    canonical_digest = hashlib.sha256(
        json.dumps(samples, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    watermark = {
        "selection_cutoff_utc": cutoff.isoformat(),
        "last_oanda_transaction_id": last_id,
        "ledger_rowid_watermark": rowid,
        "ledger_prefix_sha256": ledger_sha,
        "canonical_outcome_set_sha256": canonical_digest,
        "entry_thesis_prefix_bytes": thesis_length,
        "entry_thesis_prefix_sha256": thesis_sha,
        "forecast_history_prefix_bytes": forecast_length,
        "forecast_history_prefix_sha256": forecast_sha,
    }
    validation_contract = {
        "mode": "FORWARD_POST_REVIEW",
        "review_digest_sha256": contract["review_digest_sha256"],
        "review_completed_at_utc": contract["review_completed_at"].isoformat(),
        "minimum_sample_count": MIN_FORWARD_SAMPLE_COUNT,
    }
    cohort_id = "forward-" + hashlib.sha256(
        json.dumps(
            {
                "pair": contract["pair"],
                "bot_family": contract["bot_family"],
                "lane_id": lane_id,
                "parameter": contract["parameter"],
                "validation_contract": validation_contract,
                "canonical_outcome_set_sha256": canonical_digest,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 5,
        "cohort_id": cohort_id,
        "source_watermark": watermark,
        "selection_cutoff_utc": cutoff.isoformat(),
        "pair": contract["pair"],
        "bot_family": contract["bot_family"],
        "lane_id": lane_id,
        "parameter": contract["parameter"],
        "validation_contract": validation_contract,
        "provenance": {
            "generator": COHORT_GENERATOR_NAME,
            "execution_ledger_coverage_start_utc": _utc(coverage).isoformat(),
            "last_oanda_transaction_id": last_id,
            "post_cost_financing_included": True,
        },
        "samples": samples,
    }
    validate_source(payload)
    return payload


def validate_canonical_forward_cohort(
    payload: dict[str, Any],
    *,
    ledger_path: Path,
    entry_thesis_path: Path,
    forecast_history_path: Path,
    review: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a frozen cohort from authoritative sources and compare exactly."""

    try:
        identity = validate_source(payload)
        watermark = payload["source_watermark"]
        rebuilt = build_canonical_forward_cohort(
            ledger_path=ledger_path,
            entry_thesis_path=entry_thesis_path,
            forecast_history_path=forecast_history_path,
            review=review,
            review_completed_at_utc=payload["validation_contract"][
                "review_completed_at_utc"
            ],
            lane_id=identity["lane_id"],
            ledger_rowid_watermark=int(watermark["ledger_rowid_watermark"]),
            expected_last_transaction_id=str(watermark["last_oanda_transaction_id"]),
            entry_thesis_prefix_bytes=int(watermark["entry_thesis_prefix_bytes"]),
            forecast_history_prefix_bytes=int(watermark["forecast_history_prefix_bytes"]),
        )
        if rebuilt != payload:
            return {"status": "CANONICAL_COHORT_MISMATCH"}
        return {
            "status": "VALID",
            "source_semantic_digest": source_semantic_digest(payload),
            "sample_count": len(payload["samples"]),
        }
    except (OSError, OverflowError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "CANONICAL_COHORT_INVALID",
            "error": f"{type(exc).__name__}: {exc}",
        }
