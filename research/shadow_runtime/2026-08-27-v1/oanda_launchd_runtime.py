"""Separated OANDA LIVE zero-order launchd services."""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from oanda_live_feed import OandaLiveRecorder, load_approved_live_credentials, valid_sha256
from shadow_runtime import (
    HashLedger,
    IntegrityError,
    RuntimeLock,
    atomic_json,
    canonical_bytes,
    canonical_hash,
    parse_utc,
    real_dir,
    secure_read,
    sha256_file,
    utc_text,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
SERVICE_ROOT = PACKAGE_ROOT / "runs" / "oanda_live_launchd_v1"
SOURCE_COMMIT = "907195888ae5671d18084d59a4458dc70f3df7c8"
SHARED_RUNTIME_HASH = canonical_hash({
    "source_commit": SOURCE_COMMIT,
    "oanda_contract_sha256": sha256_file(PACKAGE_ROOT / "oanda_live_runtime_contract.json"),
    "topology": "OANDA_LIVE_GET_ONLY_ZERO_ORDER_V1",
})
RUNTIME_SOURCE_PATHS = (
    PACKAGE_ROOT / "oanda_launchd_runtime.py",
    PACKAGE_ROOT / "oanda_live_feed.py",
    PACKAGE_ROOT / "shadow_runtime.py",
    PACKAGE_ROOT / "oanda_live_runtime_contract.json",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.feed-recorder.plist",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.bot-shadow.plist",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.llm-inventory.plist",
    PACKAGE_ROOT / "oanda_launchagents" / "com.quantrabbit.oanda-live.watchdog.plist",
)


def runtime_source_hashes() -> dict[str, str]:
    return {str(path.relative_to(PACKAGE_ROOT)): sha256_file(path) for path in RUNTIME_SOURCE_PATHS}


RUNTIME_SOURCE_HASHES = runtime_source_hashes()
SERVICE_ATTESTATION_HASH = canonical_hash({
    "candidate_runtime_hash": SHARED_RUNTIME_HASH,
    "runtime_source_sha256": RUNTIME_SOURCE_HASHES,
})
LABELS = {
    "feed": "com.quantrabbit.oanda-live.feed-recorder",
    "bot": "com.quantrabbit.oanda-live.bot-shadow",
    "llm": "com.quantrabbit.oanda-live.llm-inventory",
    "watchdog": "com.quantrabbit.oanda-live.watchdog",
}
LLM_MODEL = "gpt-5.6-sol"
LLM_REASONING = "high"
CODEX_BIN = "/Users/tossaki/.local/bin/codex"
ALLOWED_ACTIONS = ("ADD", "FREEZE", "UNWIND", "RESET")
STOP = False
LEGACY_SEGMENT_ID = "LEGACY_UNSEGMENTED"


def request_stop(*_: object) -> None:
    global STOP
    STOP = True


def service_status(counters: dict[str, int], run_state: str = "RUNNING", blocked: bool = False) -> dict[str, Any]:
    return {
        "run_state": run_state,
        "feed_blocked": blocked,
        "runtime_hash": SHARED_RUNTIME_HASH,
        "counters": counters,
    }


def run_feed(max_seconds: float) -> int:
    account_id, token = load_approved_live_credentials()
    root = SERVICE_ROOT / "feed"
    recorder = OandaLiveRecorder(root)
    recorder.mark_approved_credential_file_read()
    status = recorder.run_live(
        account_id,
        token,
        max_seconds,
        runtime_hash=SERVICE_ATTESTATION_HASH,
    )
    return 2 if status["feed_blocked"] else 0


def _segment_identity(
    event: dict[str, Any],
) -> tuple[tuple[str, str | None, str | None] | None, str | None]:
    segment_id = event.get("segment_id")
    segment_started_at = event.get("segment_started_at_utc")
    feed_attestation = event.get("feed_service_attestation_hash")
    provenance_status = event.get("feed_provenance_status")
    if (
        segment_id is None
        and segment_started_at is None
        and feed_attestation is None
        and provenance_status is None
    ):
        return (LEGACY_SEGMENT_ID, None, None), None
    if not isinstance(segment_id, str) or not segment_id:
        return None, "M5_SEGMENT_METADATA_INVALID"
    if not isinstance(segment_started_at, str):
        return None, "M5_SEGMENT_METADATA_INVALID"
    try:
        parse_utc(segment_started_at)
    except Exception:
        return None, "M5_SEGMENT_METADATA_INVALID"
    if provenance_status != "ATTESTED" or not valid_sha256(feed_attestation):
        return None, "M5_FEED_ATTESTATION_INVALID"
    return (segment_id, segment_started_at, feed_attestation), None


def _declared_feed_provenance(control: HashLedger) -> dict[str, tuple[str, str]]:
    declared: dict[str, tuple[str, str]] = {}
    for row in control.rows:
        payload = row["payload"]
        if payload.get("event") != "LIVE_PRICING_CONNECTED":
            continue
        segment_id = payload.get("segment_id")
        segment_started_at = payload.get("segment_started_at_utc")
        feed_attestation = payload.get("feed_service_attestation_hash")
        provenance_status = payload.get("feed_provenance_status")
        if feed_attestation is None and provenance_status is None:
            # Pre-attestation receipts remain readable only as explicit legacy.
            continue
        if (
            not isinstance(segment_id, str)
            or not segment_id
            or not isinstance(segment_started_at, str)
            or provenance_status != "ATTESTED"
            or not valid_sha256(feed_attestation)
        ):
            raise IntegrityError("BOT_FEED_PROVENANCE_INVALID")
        try:
            parse_utc(segment_started_at)
        except Exception as exc:
            raise IntegrityError("BOT_FEED_PROVENANCE_INVALID") from exc
        provenance = (segment_started_at, feed_attestation)
        prior = declared.get(segment_id)
        if prior is not None and prior != provenance:
            raise IntegrityError("BOT_FEED_PROVENANCE_CONFLICT")
        declared[segment_id] = provenance
    return declared


def _register_segment_provenance(
    event: dict[str, Any],
    declared: dict[str, tuple[str, str]],
    observed: dict[str, tuple[str, str]],
) -> tuple[str, str | None, str | None]:
    identity, error = _segment_identity(event)
    if error == "M5_FEED_ATTESTATION_INVALID":
        raise IntegrityError("BOT_FEED_ATTESTATION_INVALID")
    if error is not None or identity is None:
        raise IntegrityError("BOT_SEGMENT_METADATA_INVALID")
    if identity[0] == LEGACY_SEGMENT_ID:
        return identity
    provenance = (identity[1] or "", identity[2] or "")
    if declared.get(identity[0]) != provenance:
        raise IntegrityError("BOT_FEED_PROVENANCE_MISMATCH")
    prior = observed.get(identity[0])
    if prior is not None and prior != provenance:
        raise IntegrityError("BOT_FEED_PROVENANCE_CONFLICT")
    observed[identity[0]] = provenance
    return identity


def _skip_payload(
    instrument: str,
    bucket: int,
    events: list[dict[str, Any]],
    connection_start_buckets: set[int] | None = None,
) -> dict[str, Any] | None:
    identities: set[tuple[str, str | None, str | None]] = set()
    invalid = False
    contains_preconnect_stale_snapshot = False
    for event in events:
        identity, error = _segment_identity(event)
        if error == "M5_FEED_ATTESTATION_INVALID":
            raise IntegrityError("BOT_FEED_ATTESTATION_INVALID")
        if error is not None or identity is None:
            invalid = True
        else:
            identities.add(identity)
            if identity[0] != LEGACY_SEGMENT_ID:
                event_bucket = int(parse_utc(event["event_time_utc"]).timestamp()) // 300 * 300
                start_bucket = int(parse_utc(identity[1] or "").timestamp()) // 300 * 300
                contains_preconnect_stale_snapshot |= event_bucket < start_bucket
    if invalid:
        reason = "M5_SEGMENT_METADATA_INVALID"
    elif len(identities) != 1:
        reason = "M5_SEGMENT_BOUNDARY_WITHIN_BUCKET"
    else:
        identity = next(iter(identities))
        if identity[0] == LEGACY_SEGMENT_ID:
            return None
        segment_start_bucket = int(parse_utc(identity[1] or "").timestamp()) // 300 * 300
        if bucket < segment_start_bucket:
            reason = "M5_PRECONNECT_STALE_SNAPSHOT_BUCKET"
        elif bucket == segment_start_bucket:
            reason = "M5_FIRST_PARTIAL_BUCKET_AFTER_CONNECT"
        elif bucket in (connection_start_buckets or set()):
            reason = "M5_SEGMENT_BOUNDARY_WITHIN_BUCKET"
        else:
            return None
    segment_ids = sorted({
        identity[0]
        for event in events
        for identity, error in [_segment_identity(event)]
        if error is None and identity is not None
    })
    segment_starts = sorted({
        identity[1]
        for event in events
        for identity, error in [_segment_identity(event)]
        if error is None and identity is not None and identity[1] is not None
    })
    feed_attestations = sorted({
        identity[2]
        for event in events
        for identity, error in [_segment_identity(event)]
        if error is None and identity is not None and identity[2] is not None
    })
    return {
        "schema_version": 1,
        "runtime_hash": SHARED_RUNTIME_HASH,
        "event": "M5_CAUSAL_EVIDENCE_SKIPPED",
        "reason": reason,
        "instrument": instrument,
        "timeframe": "M5",
        "start_utc": utc_text(datetime.fromtimestamp(bucket, timezone.utc)),
        "end_utc": utc_text(datetime.fromtimestamp(bucket + 300, timezone.utc)),
        "segment_ids": segment_ids,
        "segment_started_at_utc": segment_starts,
        "feed_service_attestation_hashes": feed_attestations,
        "event_count": len(events),
        "raw_record_set_hash": canonical_hash([x["raw_record_hash"] for x in events]),
        "contains_preconnect_stale_snapshot": contains_preconnect_stale_snapshot,
        "causal_evidence_eligible": False,
        "natural_r5_proposal": False,
        "virtual_fills": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def _bar_from_events(instrument: str, bucket: int, events: list[dict[str, Any]]) -> dict[str, Any]:
    events = sorted(events, key=lambda x: (x["event_time_utc"], x["arrival_time_utc"], x["event_id"]))
    identities: set[tuple[str, str | None, str | None]] = set()
    for event in events:
        identity, error = _segment_identity(event)
        if error is not None or identity is None:
            raise IntegrityError("BOT_SEGMENT_METADATA_INVALID")
        identities.add(identity)
    if len(identities) != 1:
        raise IntegrityError("BOT_SEGMENT_BOUNDARY_WITHIN_BUCKET")
    bar = {
        "schema_version": 1,
        "runtime_hash": SHARED_RUNTIME_HASH,
        "instrument": instrument,
        "timeframe": "M5",
        "start_utc": utc_text(datetime.fromtimestamp(bucket, timezone.utc)),
        "end_utc": utc_text(datetime.fromtimestamp(bucket + 300, timezone.utc)),
        "period_state": "SEALED",
        "event_count": len(events),
        "first_source_time_utc": events[0]["event_time_utc"],
        "last_source_time_utc": events[-1]["event_time_utc"],
        "first_arrival_time_utc": events[0]["arrival_time_utc"],
        "arrival_watermark_utc": max(x["arrival_time_utc"] for x in events),
        "bid_o": events[0]["bid"],
        "bid_h": max(x["bid"] for x in events),
        "bid_l": min(x["bid"] for x in events),
        "bid_c": events[-1]["bid"],
        "ask_o": events[0]["ask"],
        "ask_h": max(x["ask"] for x in events),
        "ask_l": min(x["ask"] for x in events),
        "ask_c": events[-1]["ask"],
        "raw_record_set_hash": canonical_hash([x["raw_record_hash"] for x in events]),
        "strategy_status": "RESEARCH_NOT_ADMITTED",
        "natural_r5_proposal": False,
        "external_orders": 0,
    }
    identity = next(iter(identities))
    if identity[0] != LEGACY_SEGMENT_ID:
        if not valid_sha256(identity[2]) or not valid_sha256(SERVICE_ATTESTATION_HASH):
            raise IntegrityError("BOT_SERVICE_ATTESTATION_INVALID")
        bar.update(
            segment_id=identity[0],
            segment_started_at_utc=identity[1],
            feed_service_attestation_hash=identity[2],
            bot_service_attestation_hash=SERVICE_ATTESTATION_HASH,
            feed_continuity_eligible=True,
        )
    return bar


def _bar_invalidation(bar_row: dict[str, Any], reason: str) -> dict[str, Any]:
    bar = bar_row["payload"]
    return {
        "schema_version": 1,
        "runtime_hash": SHARED_RUNTIME_HASH,
        "event": "BAR_EVIDENCE_INVALIDATED",
        "reason": reason,
        "bar_record_hash": bar_row["record_hash"],
        "instrument": bar["instrument"],
        "timeframe": "M5",
        "start_utc": bar["start_utc"],
        "end_utc": bar["end_utc"],
        "evidence_eligible": False,
        "natural_r5_proposal": False,
        "virtual_fills": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def _legacy_bar_invalidation(bar_row: dict[str, Any]) -> dict[str, Any] | None:
    if bar_row["payload"].get("feed_continuity_eligible") is True:
        return None
    return _bar_invalidation(bar_row, "LEGACY_RAW_WITHOUT_SEGMENT_METADATA")


def _bar_invalidation_record_id(bar_row: dict[str, Any], reason: str) -> str:
    if reason == "LEGACY_RAW_WITHOUT_SEGMENT_METADATA":
        return f"invalidate-bar::{bar_row['record_hash']}"
    return f"invalidate-bar::{bar_row['record_hash']}::{reason}"


def _plan_completed_bar(
    completed: HashLedger,
    bar: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    record_id = f"m5::{bar['instrument']}::{bar['start_utc']}"
    existing = completed.by_id.get(record_id)
    if existing is not None and existing["payload"] != bar:
        provenance_fields = {
            "segment_id",
            "segment_started_at_utc",
            "feed_service_attestation_hash",
            "bot_service_attestation_hash",
            "feed_continuity_eligible",
        }
        expected_core = {
            key: value for key, value in bar.items()
            if key not in provenance_fields
        }
        existing_core = {
            key: value for key, value in existing["payload"].items()
            if key not in provenance_fields
        }
        existing_provenance = existing["payload"]
        existing_segment_id = existing_provenance.get("segment_id")
        if existing_segment_id is not None and (
            not isinstance(existing_segment_id, str) or not existing_segment_id
        ):
            raise IntegrityError("BOT_EXISTING_BAR_PROVENANCE_INVALID")
        existing_segment_start = existing_provenance.get("segment_started_at_utc")
        if existing_segment_start is not None:
            if not isinstance(existing_segment_start, str):
                raise IntegrityError("BOT_EXISTING_BAR_PROVENANCE_INVALID")
            try:
                parse_utc(existing_segment_start)
            except Exception as exc:
                raise IntegrityError("BOT_EXISTING_BAR_PROVENANCE_INVALID") from exc
        for field in ("feed_service_attestation_hash", "bot_service_attestation_hash"):
            if field in existing_provenance and not valid_sha256(existing_provenance[field]):
                raise IntegrityError("BOT_EXISTING_BAR_ATTESTATION_INVALID")
        if (
            "feed_continuity_eligible" in existing_provenance
            and not isinstance(existing_provenance["feed_continuity_eligible"], bool)
        ):
            raise IntegrityError("BOT_EXISTING_BAR_PROVENANCE_INVALID")
        if existing_core != expected_core:
            raise IntegrityError("BOT_COMPLETED_BAR_CONFLICT")
        if not all(
            field in bar for field in provenance_fields
        ):
            raise IntegrityError("BOT_EXISTING_BAR_ATTESTATION_INVALID")
        return existing, True
    planned: list[dict[str, Any]] = []
    row = completed.plan(bar, record_id, planned)
    completed.append_rows(planned)
    return row, False


def _skip_record_id(payload: dict[str, Any]) -> str:
    receipt_id = canonical_hash({
        key: payload[key]
        for key in (
            "instrument", "start_utc", "reason", "segment_ids",
            "segment_started_at_utc", "feed_service_attestation_hashes",
            "raw_record_set_hash",
        )
    })
    return f"skip-m5::{receipt_id}"


def _bot_counts(completed: HashLedger, control: HashLedger, market_events: int) -> dict[str, int]:
    skip_rows = [
        row for row in control.rows
        if row["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"
    ]
    skipped_keys = {
        (row["payload"]["instrument"], row["payload"]["start_utc"])
        for row in skip_rows
    }
    invalidation_rows = [
        row for row in control.rows
        if row["payload"].get("event") == "BAR_EVIDENCE_INVALIDATED"
    ]
    legacy_invalidated = sum(
        row["payload"].get("reason") == "LEGACY_RAW_WITHOUT_SEGMENT_METADATA"
        for row in invalidation_rows
    )
    late_invalidated = sum(
        row["payload"].get("reason") == "LATE_SEGMENT_STALE_SNAPSHOT"
        for row in invalidation_rows
    )
    hybrid_invalidated = sum(
        row["payload"].get("reason") == "ROLLING_HYBRID_ATTESTATION_MIGRATION"
        for row in invalidation_rows
    )
    invalidated_hashes = {
        row["payload"]["bar_record_hash"]
        for row in invalidation_rows
    }
    eligible = sum(
        row["payload"].get("feed_continuity_eligible") is True
        and row["record_hash"] not in invalidated_hashes
        and (row["payload"]["instrument"], row["payload"]["start_utc"]) not in skipped_keys
        for row in completed.rows
    )
    return {
        "market_events": market_events,
        "completed_m5": len(completed.rows),
        "completed_m5_total": len(completed.rows),
        "completed_m5_eligible": eligible,
        "skipped_m5": len(skipped_keys),
        "skip_receipts": len(skip_rows),
        "legacy_invalidated_m5": legacy_invalidated,
        "late_stale_invalidated_m5": late_invalidated,
        "hybrid_invalidated_m5": hybrid_invalidated,
        "natural_r5_proposals": 0,
        "virtual_fills": 0,
        "llm_calls": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
    }


def _closed_bucket_outcomes(
    raw: HashLedger,
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    max_bucket: dict[str, int] = {}
    connection_start_buckets: dict[str, set[int]] = {}
    feed_control = HashLedger(raw.path.parent / "control.jsonl")
    declared_provenance = _declared_feed_provenance(feed_control)
    observed_provenance: dict[str, tuple[str, str]] = {}
    for row in raw.rows:
        event = {**row["payload"], "raw_record_hash": row["record_hash"]}
        stamp = parse_utc(event["event_time_utc"])
        bucket = int(stamp.timestamp()) // 300 * 300
        key = (event["instrument"], bucket)
        grouped.setdefault(key, []).append(event)
        max_bucket[event["instrument"]] = max(bucket, max_bucket.get(event["instrument"], bucket))
        identity = _register_segment_provenance(
            event,
            declared_provenance,
            observed_provenance,
        )
        if identity[0] != LEGACY_SEGMENT_ID:
            start_bucket = int(parse_utc(identity[1] or "").timestamp()) // 300 * 300
            connection_start_buckets.setdefault(event["instrument"], set()).add(start_bucket)
    outcomes = []
    for (instrument, bucket), events in sorted(grouped.items()):
        if bucket >= max_bucket[instrument]:
            continue
        skip = _skip_payload(
            instrument,
            bucket,
            events,
            connection_start_buckets.get(instrument, set()),
        )
        if skip is not None:
            non_stale_events = []
            if skip["contains_preconnect_stale_snapshot"]:
                for event in events:
                    identity, error = _segment_identity(event)
                    if error is not None or identity is None or identity[0] == LEGACY_SEGMENT_ID:
                        non_stale_events.append(event)
                        continue
                    start_bucket = int(parse_utc(identity[1] or "").timestamp()) // 300 * 300
                    if bucket >= start_bucket:
                        non_stale_events.append(event)
            reconstructed = None
            if non_stale_events:
                remaining_skip = _skip_payload(
                    instrument,
                    bucket,
                    non_stale_events,
                    connection_start_buckets.get(instrument, set()),
                )
                if remaining_skip is None:
                    reconstructed = _bar_from_events(instrument, bucket, non_stale_events)
            outcomes.append((reconstructed, skip))
        else:
            outcomes.append((_bar_from_events(instrument, bucket, events), None))
    return outcomes


def _completed_bars(raw: HashLedger) -> list[dict[str, Any]]:
    return [
        bar for bar, skip in _closed_bucket_outcomes(raw)
        if bar is not None
        and skip is None
        and bar.get("feed_continuity_eligible") is True
    ]


class IncrementalBot:
    def __init__(self) -> None:
        self.feed_raw = HashLedger(SERVICE_ROOT / "feed" / "ledgers" / "raw_bbo.jsonl")
        self.feed_control = HashLedger(SERVICE_ROOT / "feed" / "ledgers" / "control.jsonl")
        root = SERVICE_ROOT / "bot"
        real_dir(root)
        real_dir(root / "ledgers")
        self.completed = HashLedger(root / "ledgers" / "completed_m5.jsonl")
        self.control = HashLedger(root / "ledgers" / "control.jsonl")
        self.buckets: dict[str, tuple[int, list[dict[str, Any]]]] = {}
        self.connection_start_buckets: dict[str, set[int]] = {}
        self.declared_provenance = _declared_feed_provenance(self.feed_control)
        self.observed_provenance: dict[str, tuple[str, str]] = {}
        self.processed_rows = 0
        self._consume_new_rows()

    def _record_skip(self, payload: dict[str, Any]) -> None:
        planned: list[dict[str, Any]] = []
        self.control.plan(
            payload,
            _skip_record_id(payload),
            planned,
        )
        self.control.append_rows(planned)

    def _record_bar_invalidation(self, bar_row: dict[str, Any], reason: str) -> None:
        payload = _bar_invalidation(bar_row, reason)
        planned: list[dict[str, Any]] = []
        self.control.plan(
            payload,
            _bar_invalidation_record_id(bar_row, reason),
            planned,
        )
        self.control.append_rows(planned)

    def _record_legacy_invalidation(self, bar_row: dict[str, Any]) -> None:
        if _legacy_bar_invalidation(bar_row) is None:
            return
        self._record_bar_invalidation(bar_row, "LEGACY_RAW_WITHOUT_SEGMENT_METADATA")

    def _seal(self, instrument: str, bucket: int, events: list[dict[str, Any]]) -> None:
        skip = _skip_payload(
            instrument,
            bucket,
            events,
            self.connection_start_buckets.get(instrument, set()),
        )
        if skip is not None:
            self._record_skip(skip)
            return
        bar = _bar_from_events(instrument, bucket, events)
        row, hybrid_collision = _plan_completed_bar(self.completed, bar)
        if hybrid_collision:
            self._record_bar_invalidation(row, "ROLLING_HYBRID_ATTESTATION_MIGRATION")
            return
        self._record_legacy_invalidation(row)
        control_planned: list[dict[str, Any]] = []
        self.control.plan({
            "event": "R5_NATURAL_PROPOSAL_NOT_EMITTED",
            "bar_record_hash": row["record_hash"],
            "candidate_scope": "ACCOUNTING_ONLY_NOT_CAUSAL_SIGNAL_ADMISSION",
            "external_orders": 0,
        }, f"no-proposal::{row['record_hash']}", control_planned)
        self.control.append_rows(control_planned)

    def _consume_new_rows(self) -> None:
        for row in self.feed_raw.rows[self.processed_rows:]:
            event = {**row["payload"], "raw_record_hash": row["record_hash"]}
            instrument = event["instrument"]
            bucket = int(parse_utc(event["event_time_utc"]).timestamp()) // 300 * 300
            identity = _register_segment_provenance(
                event,
                self.declared_provenance,
                self.observed_provenance,
            )
            segment_start_bucket = None
            if identity[0] != LEGACY_SEGMENT_ID:
                segment_start_bucket = int(parse_utc(identity[1] or "").timestamp()) // 300 * 300
                self.connection_start_buckets.setdefault(instrument, set()).add(segment_start_bucket)
            current = self.buckets.get(instrument)
            if current is not None and bucket < current[0]:
                if segment_start_bucket is not None and bucket < segment_start_bucket:
                    record_id = f"m5::{instrument}::{utc_text(datetime.fromtimestamp(bucket, timezone.utc))}"
                    completed = self.completed.by_id.get(record_id)
                    if completed is not None:
                        self._record_bar_invalidation(completed, "LATE_SEGMENT_STALE_SNAPSHOT")
                    else:
                        payload = _skip_payload(
                            instrument,
                            bucket,
                            [event],
                            self.connection_start_buckets.get(instrument, set()),
                        )
                        if payload is None:
                            raise IntegrityError("BOT_STALE_BUCKET_NOT_SKIPPED")
                        self._record_skip(payload)
                    continue
                raise IntegrityError("BOT_SOURCE_BUCKET_REGRESSION")
            if current is None:
                self.buckets[instrument] = (bucket, [event])
            elif bucket == current[0]:
                current[1].append(event)
            elif bucket > current[0]:
                self._seal(instrument, current[0], current[1])
                self.buckets[instrument] = (bucket, [event])
        self.processed_rows = len(self.feed_raw.rows)

    def process_once(self) -> dict[str, int]:
        self.feed_control.refresh()
        self.declared_provenance = _declared_feed_provenance(self.feed_control)
        self.feed_raw.refresh()
        self._consume_new_rows()
        return _bot_counts(self.completed, self.control, len(self.feed_raw.rows))


def bot_process_once() -> dict[str, int]:
    feed_raw = HashLedger(SERVICE_ROOT / "feed" / "ledgers" / "raw_bbo.jsonl")
    root = SERVICE_ROOT / "bot"
    real_dir(root)
    real_dir(root / "ledgers")
    completed = HashLedger(root / "ledgers" / "completed_m5.jsonl")
    control = HashLedger(root / "ledgers" / "control.jsonl")
    for bar, skip in _closed_bucket_outcomes(feed_raw):
        if skip is not None:
            record_id = f"m5::{skip['instrument']}::{skip['start_utc']}"
            existing_bar = completed.by_id.get(record_id)
            if skip["contains_preconnect_stale_snapshot"] and bar is not None:
                existing_bar, hybrid_collision = _plan_completed_bar(completed, bar)
                if hybrid_collision:
                    hybrid_reason = "ROLLING_HYBRID_ATTESTATION_MIGRATION"
                    hybrid_invalidation = _bar_invalidation(existing_bar, hybrid_reason)
                    hybrid_planned: list[dict[str, Any]] = []
                    control.plan(
                        hybrid_invalidation,
                        _bar_invalidation_record_id(existing_bar, hybrid_reason),
                        hybrid_planned,
                    )
                    control.append_rows(hybrid_planned)
            if skip["contains_preconnect_stale_snapshot"] and existing_bar is not None:
                reason = "LATE_SEGMENT_STALE_SNAPSHOT"
                invalidation = _bar_invalidation(existing_bar, reason)
                invalidation_planned: list[dict[str, Any]] = []
                control.plan(
                    invalidation,
                    _bar_invalidation_record_id(existing_bar, reason),
                    invalidation_planned,
                )
                control.append_rows(invalidation_planned)
            else:
                control_planned: list[dict[str, Any]] = []
                control.plan(
                    skip,
                    _skip_record_id(skip),
                    control_planned,
                )
                control.append_rows(control_planned)
            continue
        if bar is None:
            raise IntegrityError("BOT_BUCKET_OUTCOME_MISSING")
        row, hybrid_collision = _plan_completed_bar(completed, bar)
        if hybrid_collision:
            reason = "ROLLING_HYBRID_ATTESTATION_MIGRATION"
            invalidation = _bar_invalidation(row, reason)
            invalidation_planned: list[dict[str, Any]] = []
            control.plan(
                invalidation,
                _bar_invalidation_record_id(row, reason),
                invalidation_planned,
            )
            control.append_rows(invalidation_planned)
            continue
        invalidation = _legacy_bar_invalidation(row)
        if invalidation is not None:
            invalidation_planned: list[dict[str, Any]] = []
            control.plan(
                invalidation,
                _bar_invalidation_record_id(row, invalidation["reason"]),
                invalidation_planned,
            )
            control.append_rows(invalidation_planned)
        control_planned: list[dict[str, Any]] = []
        control.plan({
            "event": "R5_NATURAL_PROPOSAL_NOT_EMITTED",
            "bar_record_hash": row["record_hash"],
            "candidate_scope": "ACCOUNTING_ONLY_NOT_CAUSAL_SIGNAL_ADMISSION",
            "external_orders": 0,
        }, f"no-proposal::{row['record_hash']}", control_planned)
        control.append_rows(control_planned)
    return _bot_counts(completed, control, len(feed_raw.rows))


def run_bot(max_seconds: float) -> int:
    global STOP
    STOP = False
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    root = SERVICE_ROOT / "bot"
    deadline = time.monotonic() + max_seconds
    with RuntimeLock(root, SERVICE_ATTESTATION_HASH) as lock:
        processor = IncrementalBot()
        while not STOP and time.monotonic() < deadline:
            counters = processor.process_once()
            lock.heartbeat(service_status(counters))
            time.sleep(2.0)
        counters = processor.process_once()
        lock.heartbeat(service_status(counters, "STOPPED_GRACEFULLY"))
    return 0


def llm_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
            "currency_cap": {"type": "integer", "minimum": 0, "maximum": 1000000000},
            "mode": {"type": "string", "enum": ["SHADOW_ONLY"]},
            "valid_until": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 240},
        },
        "required": ["action", "currency_cap", "mode", "valid_until", "confidence", "reason"],
        "additionalProperties": False,
    }


def actual_model(prompt: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="qr-oanda-llm-") as tmp:
        schema = Path(tmp) / "schema.json"
        output = Path(tmp) / "output.json"
        schema.write_bytes(canonical_bytes(llm_output_schema()))
        subprocess.run([
            CODEX_BIN, "exec", "-", "--model", LLM_MODEL,
            "-c", 'model_reasoning_effort="high"', "--ephemeral",
            "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
            "--sandbox", "read-only", "--output-schema", str(schema),
            "--output-last-message", str(output),
        ], input=prompt, text=True, capture_output=True, check=True, timeout=180, cwd=tmp)
        return json.loads(output.read_text(encoding="utf-8"))


def process_llm_trigger(runner: Callable[[str], dict[str, Any]] = actual_model) -> dict[str, int]:
    root = SERVICE_ROOT / "llm"
    trigger_path = SERVICE_ROOT / "triggers" / "llm_inventory_request.json"
    real_dir(root)
    real_dir(root / "ledgers")
    receipts = HashLedger(root / "ledgers" / "receipts.jsonl")
    if not trigger_path.exists():
        return {"triggers": 0, "llm_calls": len(receipts.rows), "external_order_attempts": 0, "external_orders": 0}
    trigger = json.loads(secure_read(trigger_path).decode("utf-8", "strict"))
    required = {"trigger_id", "runtime_hash", "inventory_snapshot_hash", "open_inventory_count", "created_at_utc", "evidence_eligible", "profit_evidence", "external_orders"}
    if set(trigger) != required or trigger["runtime_hash"] != SHARED_RUNTIME_HASH:
        raise RuntimeError("LLM_TRIGGER_SCHEMA_MISMATCH")
    if trigger["evidence_eligible"] or trigger["profit_evidence"] or trigger["external_orders"] != 0:
        raise RuntimeError("LLM_TRIGGER_AUTHORITY_MISMATCH")
    record_id = f"llm::{trigger['trigger_id']}"
    if any(row["record_id"] == record_id for row in receipts.rows):
        return {"triggers": 1, "llm_calls": len(receipts.rows), "external_order_attempts": 0, "external_orders": 0}
    request_time = datetime.now(timezone.utc)
    prompt = (
        "Return one JSON inventory decision. Allowed action: ADD/FREEZE/UNWIND/RESET. "
        "mode=SHADOW_ONLY; currency_cap=0..1000000000 JPY microunits; valid_until within 2h. "
        "Do not control order, direction, fill, TP, SL, leverage, cost, or hard guard. External orders=0. "
        f"request_time={utc_text(request_time)} snapshot={canonical_bytes(trigger).decode()}"
    )
    output = runner(prompt)
    if set(output) != {"action", "currency_cap", "mode", "valid_until", "confidence", "reason"}:
        raise RuntimeError("LLM_OUTPUT_SCHEMA_MISMATCH")
    if output["action"] not in ALLOWED_ACTIONS or output["mode"] != "SHADOW_ONLY":
        raise RuntimeError("LLM_OUTPUT_AUTHORITY_MISMATCH")
    if type(output["currency_cap"]) is not int or not 0 <= output["currency_cap"] <= 1000000000:
        raise RuntimeError("LLM_OUTPUT_CAP_MISMATCH")
    if not request_time < parse_utc(output["valid_until"]) <= request_time + timedelta(hours=2):
        raise RuntimeError("LLM_OUTPUT_EXPIRY_MISMATCH")
    decision_time = datetime.now(timezone.utc)
    payload = {
        "kind": "ACTUAL_LLM_INVENTORY_RECEIPT",
        "runtime_hash": SHARED_RUNTIME_HASH,
        "service_attestation_hash": SERVICE_ATTESTATION_HASH,
        "model": LLM_MODEL,
        "reasoning": LLM_REASONING,
        "prompt_full": prompt,
        "prompt_sha256": canonical_hash(prompt),
        "input": trigger,
        "input_sha256": canonical_hash(trigger),
        "output": output,
        "output_sha256": canonical_hash(output),
        "decision_timestamp_utc": utc_text(decision_time),
        "arrival_timestamp_utc": utc_text(datetime.now(timezone.utc)),
        "individual_order_control": False,
        "hard_guard_mutation": False,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    planned: list[dict[str, Any]] = []
    receipts.plan(payload, record_id, planned)
    receipts.append_rows(planned)
    return {"triggers": 1, "llm_calls": len(receipts.rows), "external_order_attempts": 0, "external_orders": 0}


def run_llm() -> int:
    root = SERVICE_ROOT / "llm"
    with RuntimeLock(root, SERVICE_ATTESTATION_HASH) as lock:
        counters = process_llm_trigger()
        lock.heartbeat(service_status(counters, "IDLE_TRIGGER_DRIVEN"))
    return 0


def _verify_heartbeat(path: Path, max_age: float) -> dict[str, Any]:
    row = json.loads(secure_read(path).decode("utf-8", "strict"))
    unsigned = {k: v for k, v in row.items() if k != "heartbeat_hash"}
    if canonical_hash(unsigned) != row.get("heartbeat_hash"):
        raise RuntimeError("CORRUPT_HEARTBEAT")
    if row.get("strategy_hash") != SERVICE_ATTESTATION_HASH:
        raise RuntimeError("RUNTIME_HASH_MISMATCH")
    if (datetime.now(timezone.utc) - parse_utc(row["beat_at_utc"])).total_seconds() > max_age:
        raise RuntimeError("STALE_HEARTBEAT")
    if row["counters"].get("external_orders") != 0 or row["counters"].get("external_order_attempts") != 0:
        raise RuntimeError("ORDER_AUTHORITY_BREACH")
    return row


def run_watchdog() -> int:
    root = SERVICE_ROOT / "watchdog"
    real_dir(root)
    feed = _verify_heartbeat(SERVICE_ROOT / "feed" / "heartbeat.json", 45.0)
    bot = _verify_heartbeat(SERVICE_ROOT / "bot" / "heartbeat.json", 45.0)
    counters = {
        "feed_events": int(feed["counters"].get("market_events_accepted", 0)),
        "bot_bars": int(bot["counters"].get("completed_m5", 0)),
        "bot_bars_total": int(bot["counters"].get("completed_m5_total", 0)),
        "bot_bars_eligible": int(bot["counters"].get("completed_m5_eligible", 0)),
        "bot_bars_skipped": int(bot["counters"].get("skipped_m5", 0)),
        "bot_bars_legacy_invalidated": int(bot["counters"].get("legacy_invalidated_m5", 0)),
        "llm_calls": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
    }
    with RuntimeLock(root, SERVICE_ATTESTATION_HASH) as lock:
        lock.heartbeat(service_status(counters, "HEALTHY"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("feed", "bot", "llm", "watchdog"))
    parser.add_argument("--seconds", type=float, default=86400.0)
    args = parser.parse_args(argv)
    real_dir(SERVICE_ROOT)
    real_dir(SERVICE_ROOT / "triggers")
    real_dir(SERVICE_ROOT / "logs")
    if args.service == "feed":
        return run_feed(args.seconds)
    if args.service == "bot":
        return run_bot(args.seconds)
    if args.service == "llm":
        return run_llm()
    return run_watchdog()


if __name__ == "__main__":
    raise SystemExit(main())
