"""Separated OANDA LIVE zero-order launchd services."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from oanda_live_feed import (
    OandaLiveRecorder,
    fetch_completed_m5_warmup,
    load_approved_live_credentials,
    valid_sha256,
)
from oanda_paper_execution import (
    completed_bar_input_window,
    evaluate_completed_bar_signal,
    executable_bbo_available,
    pip_size,
    pnl_pips,
    quote_pnl,
    validate_paper_config,
    virtual_price,
)
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
SERVICE_ROOT = PACKAGE_ROOT / "runs" / "oanda_live_launchd_v4"
SOURCE_COMMIT = "907195888ae5671d18084d59a4458dc70f3df7c8"
SHARED_RUNTIME_HASH = canonical_hash({
    "source_commit": SOURCE_COMMIT,
    "oanda_contract_sha256": sha256_file(PACKAGE_ROOT / "oanda_live_runtime_contract.json"),
    "topology": "OANDA_LIVE_GET_ONLY_ZERO_ORDER_PAPER_TRADER_V4",
})
RUNTIME_SOURCE_PATHS = (
    PACKAGE_ROOT / "oanda_launchd_runtime.py",
    PACKAGE_ROOT / "oanda_live_feed.py",
    PACKAGE_ROOT / "oanda_paper_execution.py",
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
OANDA_RUNTIME_CONTRACT = json.loads(
    secure_read(PACKAGE_ROOT / "oanda_live_runtime_contract.json").decode("utf-8", "strict")
)
PAPER_CONFIG = OANDA_RUNTIME_CONTRACT["paper_execution"]
HISTORICAL_WARMUP_CONFIG = OANDA_RUNTIME_CONTRACT["historical_warmup"]
LLM_POLICY_CONFIG = OANDA_RUNTIME_CONTRACT["llm_inventory_policy"]
validate_paper_config(PAPER_CONFIG)
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
    for instrument in OANDA_RUNTIME_CONTRACT["symbols"]:
        fetch_completed_m5_warmup(
            account_id,
            token,
            instrument,
            int(HISTORICAL_WARMUP_CONFIG["request_count"]),
            recorder,
        )
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
        "feature_source": "LIVE_ATTESTED_M5",
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
        if not all(field in bar for field in provenance_fields):
            raise IntegrityError("BOT_EXISTING_BAR_ATTESTATION_INVALID")
        existing_has_provenance = all(
            field in existing["payload"] for field in provenance_fields
        )
        if existing_has_provenance:
            for field in (
                "segment_id",
                "segment_started_at_utc",
                "feed_service_attestation_hash",
                "feed_continuity_eligible",
            ):
                if existing["payload"][field] != bar[field]:
                    raise IntegrityError("BOT_EXISTING_BAR_PROVENANCE_CONFLICT")
            # The aggregation result and feed provenance are immutable market
            # evidence.  A later bot release has a different bot attestation,
            # but must reuse the identical prior bar rather than invalidate it.
            return existing, False
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


def _bot_counts(
    completed: HashLedger,
    control: HashLedger,
    market_events: int,
    paper_ledgers: dict[str, HashLedger] | None = None,
    open_inventory_count: int = 0,
    historical_warmup_m5: int = 0,
) -> dict[str, int]:
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
    paper_ledgers = paper_ledgers or {}
    proposals = paper_ledgers.get("proposals")
    expected_orders = paper_ledgers.get("expected_orders")
    fills = paper_ledgers.get("virtual_fills")
    inventory = paper_ledgers.get("inventory")
    pnl = paper_ledgers.get("pnl")
    llm_receipts = paper_ledgers.get("llm_receipts")
    inventory_rows = [] if inventory is None else inventory.rows
    pnl_rows = [] if pnl is None else pnl.rows
    return {
        "market_events": market_events,
        "completed_m5": len(completed.rows),
        "historical_warmup_m5": historical_warmup_m5,
        "completed_m5_total": len(completed.rows),
        "completed_m5_eligible": eligible,
        "skipped_m5": len(skipped_keys),
        "skip_receipts": len(skip_rows),
        "legacy_invalidated_m5": legacy_invalidated,
        "late_stale_invalidated_m5": late_invalidated,
        "hybrid_invalidated_m5": hybrid_invalidated,
        "natural_r5_proposals": 0,
        "natural_paper_proposals": 0 if proposals is None else len(proposals.rows),
        "expected_orders": 0 if expected_orders is None else len(expected_orders.rows),
        "virtual_fills": 0 if fills is None else len(fills.rows),
        "virtual_exits": sum(row["payload"].get("event") == "CLOSE" for row in inventory_rows),
        "open_inventory_count": open_inventory_count,
        "pnl_records": len(pnl_rows),
        "realized_pnl_records": sum(
            row["payload"].get("event") == "REALIZED_PNL" for row in pnl_rows
        ),
        "terminal_mtm_records": sum(
            row["payload"].get("event") == "TERMINAL_MTM" for row in pnl_rows
        ),
        "llm_calls": 0 if llm_receipts is None else len(llm_receipts.rows),
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
        self.historical_warmup = HashLedger(
            SERVICE_ROOT / "feed" / "ledgers" / "historical_warmup_m5.jsonl"
        )
        root = SERVICE_ROOT / "bot"
        real_dir(root)
        real_dir(root / "ledgers")
        real_dir(SERVICE_ROOT / "triggers")
        self.completed = HashLedger(root / "ledgers" / "completed_m5.jsonl")
        self.control = HashLedger(root / "ledgers" / "control.jsonl")
        self.paper_ledgers = {
            name: HashLedger(root / "ledgers" / f"{name}.jsonl")
            for name in ("proposals", "expected_orders", "virtual_fills", "inventory", "pnl")
        }
        self.llm_receipts = HashLedger(SERVICE_ROOT / "llm" / "ledgers" / "receipts.jsonl")
        self.paper_ledgers["llm_receipts"] = self.llm_receipts
        self.buckets: dict[str, tuple[int, list[dict[str, Any]]]] = {}
        self.connection_start_buckets: dict[str, set[int]] = {}
        self.declared_provenance = _declared_feed_provenance(self.feed_control)
        self.observed_provenance: dict[str, tuple[str, str]] = {}
        self.histories: dict[str, list[dict[str, Any]]] = {}
        self.pending_orders: dict[str, dict[str, Any]] = {}
        self.open_positions: dict[str, dict[str, Any]] = {}
        self.last_quotes: dict[str, dict[str, Any]] = {}
        self.llm_policy = {
            "action": "ADD",
            "max_open_positions": PAPER_CONFIG["hard_max_open_positions_total"],
            "source": "BOT_DEFAULT_BEFORE_FIRST_LLM_RECEIPT",
            "valid_until": None,
            "effective_at_utc": None,
        }
        self.llm_unwind_consumed: set[str] = set()
        self.processed_rows = 0
        # Rebuild bars and the open source bucket without retroactively creating
        # signals, fills, or exits.  Only rows appended after this activation are
        # eligible for new paper decisions.
        self._consume_new_rows(replay=True)
        self._rebuild_histories()
        self._reconcile_paper_transactions()
        self._rebuild_paper_state()
        self._replay_durable_execution_events()
        self._refresh_llm_policy()

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

    def _append_paper(self, ledger: str, payload: dict[str, Any], record_id: str) -> dict[str, Any]:
        planned: list[dict[str, Any]] = []
        row = self.paper_ledgers[ledger].plan(payload, record_id, planned)
        self.paper_ledgers[ledger].append_rows(planned)
        return row

    def _eligible_rows(self) -> list[dict[str, Any]]:
        invalidated = {
            row["payload"].get("bar_record_hash")
            for row in self.control.rows
            if row["payload"].get("event") == "BAR_EVIDENCE_INVALIDATED"
        }
        skipped = {
            (row["payload"].get("instrument"), row["payload"].get("start_utc"))
            for row in self.control.rows
            if row["payload"].get("event") == "M5_CAUSAL_EVIDENCE_SKIPPED"
        }
        return [
            row
            for row in self.completed.rows
            if row["payload"].get("feed_continuity_eligible") is True
            and row["record_hash"] not in invalidated
            and (row["payload"].get("instrument"), row["payload"].get("start_utc")) not in skipped
        ]

    def _rebuild_histories(self) -> None:
        self.histories = {}
        seen: dict[tuple[str, str], str] = {}
        for row in sorted(
            self.historical_warmup.rows,
            key=lambda value: (value["payload"]["instrument"], value["payload"]["start_utc"]),
        ):
            payload = row["payload"]
            key = (payload["instrument"], payload["start_utc"])
            if key in seen:
                raise IntegrityError("BOT_WARMUP_DUPLICATE_TIME")
            seen[key] = row["record_hash"]
            self.histories.setdefault(payload["instrument"], []).append(payload)
        for row in sorted(
            self._eligible_rows(),
            key=lambda value: (value["payload"]["instrument"], value["payload"]["start_utc"]),
        ):
            payload = row["payload"]
            key = (payload["instrument"], payload["start_utc"])
            if key in seen:
                raise IntegrityError("BOT_WARMUP_LIVE_OVERLAP")
            seen[key] = row["record_hash"]
            self.histories.setdefault(payload["instrument"], []).append(payload)

    def _position_from_fill(
        self,
        fill_row: dict[str, Any],
        expected: dict[str, Any],
    ) -> dict[str, Any]:
        fill = fill_row["payload"]
        identity_fields = (
            "expected_order_id",
            "signal_id",
            "execution_arm",
            "instrument",
            "direction",
            "units",
        )
        if any(fill.get(field) != expected.get(field) for field in identity_fields):
            raise IntegrityError("PAPER_FILL_EXPECTED_ORDER_MISMATCH")
        fill_source = parse_utc(fill["fill_source_time_utc"])
        return {
            "schema_version": 1,
            "event": "OPEN",
            "position_id": fill["position_id"],
            "expected_order_id": fill["expected_order_id"],
            "signal_id": fill["signal_id"],
            "fill_record_hash": fill_row["record_hash"],
            "execution_arm": fill["execution_arm"],
            "instrument": fill["instrument"],
            "direction": fill["direction"],
            "units": fill["units"],
            "entry_price": fill["virtual_entry_price"],
            "entry_mid": fill["entry_mid"],
            "tp_price": fill["virtual_entry_price"]
            + fill["direction"] * expected["tp_distance_price"],
            "fill_source_time_utc": fill["fill_source_time_utc"],
            "fill_arrival_time_utc": fill["fill_arrival_time_utc"],
            "max_age_at_utc": utc_text(
                fill_source + timedelta(minutes=5 * expected["max_age_bars"])
            ),
            "individual_price_sl": False,
            "external_orders": 0,
        }

    @staticmethod
    def _realized_pnl_from_close(
        position: dict[str, Any],
        close_row: dict[str, Any],
    ) -> dict[str, Any]:
        close = close_row["payload"]
        required = {
            "gross_pips",
            "execution_cost_pips",
            "net_pips",
            "break_even_round_trip_cost_pips",
            "pnl_quote",
            "pnl_jpy",
            "jpy_conversion_status",
            "conversion_bbo_event_id",
            "conversion_source_time_utc",
            "conversion_arrival_time_utc",
            "conversion_bid",
            "conversion_ask",
            "conversion_rate",
            "conversion_side",
            "conversion_quote_age_seconds",
            "conversion_tradeable",
            "conversion_liquidity",
        }
        if required - set(close):
            raise IntegrityError("PAPER_CLOSE_RECOVERY_FIELDS_MISSING")
        return {
            "schema_version": 1,
            "event": "REALIZED_PNL",
            "position_id": position["position_id"],
            "signal_id": position["signal_id"],
            "close_record_hash": close_row["record_hash"],
            "execution_arm": position["execution_arm"],
            "instrument": position["instrument"],
            "reason": close["reason"],
            "gross_pips": close["gross_pips"],
            "execution_cost_pips": close["execution_cost_pips"],
            "net_pips": close["net_pips"],
            "break_even_round_trip_cost_pips": close[
                "break_even_round_trip_cost_pips"
            ],
            "pnl_quote": close["pnl_quote"],
            "pnl_jpy": close["pnl_jpy"],
            "jpy_conversion_status": close["jpy_conversion_status"],
            "conversion_bbo_event_id": close["conversion_bbo_event_id"],
            "conversion_source_time_utc": close["conversion_source_time_utc"],
            "conversion_arrival_time_utc": close["conversion_arrival_time_utc"],
            "conversion_bid": close["conversion_bid"],
            "conversion_ask": close["conversion_ask"],
            "conversion_rate": close["conversion_rate"],
            "conversion_side": close["conversion_side"],
            "conversion_quote_age_seconds": close["conversion_quote_age_seconds"],
            "conversion_tradeable": close["conversion_tradeable"],
            "conversion_liquidity": close["conversion_liquidity"],
            "terminal_mtm_included": True,
            "research_status": "RESEARCH_NOT_ADMITTED",
            "profit_proven": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }

    def _reconcile_paper_transactions(self) -> None:
        """Complete split-ledger transactions from their durable first half."""
        expected = {
            row["payload"]["expected_order_id"]: row["payload"]
            for row in self.paper_ledgers["expected_orders"].rows
        }
        for fill_row in self.paper_ledgers["virtual_fills"].rows:
            fill = fill_row["payload"]
            order = expected.get(fill.get("expected_order_id"))
            if order is None:
                raise IntegrityError("PAPER_FILL_WITHOUT_EXPECTED_ORDER")
            position = self._position_from_fill(fill_row, order)
            record_id = f"inventory-open::{position['position_id']}"
            existing = self.paper_ledgers["inventory"].by_id.get(record_id)
            if existing is None:
                self._append_paper("inventory", position, record_id)
            elif existing["payload"] != position:
                raise IntegrityError("PAPER_FILL_OPEN_RECONCILIATION_MISMATCH")

        opens = {
            row["payload"]["position_id"]: row["payload"]
            for row in self.paper_ledgers["inventory"].rows
            if row["payload"].get("event") == "OPEN"
        }
        for close_row in self.paper_ledgers["inventory"].rows:
            close = close_row["payload"]
            if close.get("event") != "CLOSE":
                continue
            position = opens.get(close["position_id"])
            if position is None:
                raise IntegrityError("PAPER_CLOSE_WITHOUT_OPEN")
            pnl = self._realized_pnl_from_close(position, close_row)
            record_id = f"pnl::{position['position_id']}"
            existing = self.paper_ledgers["pnl"].by_id.get(record_id)
            if existing is None:
                self._append_paper("pnl", pnl, record_id)
            elif existing["payload"] != pnl:
                raise IntegrityError("PAPER_CLOSE_PNL_RECONCILIATION_MISMATCH")

    def _rebuild_paper_state(self) -> None:
        self.open_positions = {}
        self.llm_unwind_consumed = set()
        for row in self.paper_ledgers["inventory"].rows:
            payload = row["payload"]
            if payload.get("event") == "OPEN":
                self.open_positions[payload["position_id"]] = payload
            elif payload.get("event") == "CLOSE":
                self.open_positions.pop(payload["position_id"], None)
            if payload.get("llm_unwind_policy_consumed") is True:
                source = payload.get("llm_policy_source")
                if not isinstance(source, str) or not source:
                    raise IntegrityError("LLM_UNWIND_CONSUMPTION_INVALID")
                self.llm_unwind_consumed.add(source)
        filled = {
            row["payload"]["expected_order_id"]
            for row in self.paper_ledgers["virtual_fills"].rows
        }
        expired = {
            row["payload"].get("expected_order_id")
            for row in self.control.rows
            if row["payload"].get("event") == "VIRTUAL_ORDER_EXPIRED_NO_FILL"
        }
        self.pending_orders = {}
        latency_events: dict[str, set[str]] = {}
        for row in self.control.rows:
            payload = row["payload"]
            if payload.get("event") != "VIRTUAL_ORDER_LATENCY_EVENT_CONSUMED":
                continue
            expected_order_id = payload.get("expected_order_id")
            event_id = payload.get("bbo_event_id")
            if not isinstance(expected_order_id, str) or not isinstance(event_id, str):
                raise IntegrityError("PAPER_LATENCY_RECEIPT_INVALID")
            latency_events.setdefault(expected_order_id, set()).add(event_id)
        for row in self.paper_ledgers["expected_orders"].rows:
            payload = row["payload"]
            if (
                payload.get("status") == "PENDING"
                and payload["expected_order_id"] not in filled
                and payload["expected_order_id"] not in expired
            ):
                configured_latency = PAPER_CONFIG["arms"][payload["execution_arm"]][
                    "entry_latency_events"
                ]
                consumed_event_ids = latency_events.get(payload["expected_order_id"], set())
                if len(consumed_event_ids) > configured_latency:
                    raise IntegrityError("PAPER_LATENCY_RECEIPT_OVERFLOW")
                self.pending_orders[payload["expected_order_id"]] = {
                    **payload,
                    "latency_remaining": configured_latency - len(consumed_event_ids),
                    "latency_consumed_event_ids": set(consumed_event_ids),
                }
    def _replay_durable_execution_events(self) -> None:
        """Recover deterministic exits and fills without recreating signals."""
        self.last_quotes = {}
        for row in self.feed_raw.rows:
            event = row["payload"]
            self._set_llm_policy_at(parse_utc(event["arrival_time_utc"]))
            self.last_quotes[event["instrument"]] = event
            self._close_positions(event)
            self._fill_pending_orders(event)

    def _set_llm_policy_at(self, at: datetime) -> None:
        """Apply the latest receipt that was durably known strictly before at."""
        candidates: list[tuple[datetime, int, dict[str, Any]]] = []
        for row in self.llm_receipts.rows:
            receipt = row["payload"]
            try:
                arrival = parse_utc(receipt["arrival_timestamp_utc"])
            except Exception as exc:
                raise IntegrityError("LLM_RECEIPT_ARRIVAL_INVALID") from exc
            if arrival < at:
                candidates.append((arrival, int(row["sequence_no"]), row))
        if not candidates:
            self.llm_policy = {
                "action": "ADD",
                "max_open_positions": PAPER_CONFIG["hard_max_open_positions_total"],
                "source": "BOT_DEFAULT_BEFORE_FIRST_LLM_RECEIPT",
                "valid_until": None,
                "effective_at_utc": None,
            }
            return
        _, _, receipt_row = max(candidates, key=lambda item: (item[0], item[1]))
        receipt = receipt_row["payload"]
        receipt_identity = receipt_row["record_hash"]
        if (
            receipt.get("kind") != "ACTUAL_LLM_INVENTORY_RECEIPT"
            or receipt.get("runtime_hash") != SHARED_RUNTIME_HASH
            or receipt.get("individual_order_control") is not False
            or receipt.get("hard_guard_mutation") is not False
            or receipt.get("external_orders") != 0
        ):
            self.llm_policy = {
                "action": "FREEZE",
                "max_open_positions": 0,
                "source": receipt_identity,
                "valid_until": None,
                "effective_at_utc": receipt["arrival_timestamp_utc"],
            }
            return
        output = receipt.get("output", {})
        valid_until = output.get("valid_until")
        if not isinstance(valid_until, str):
            valid_until_time = None
        else:
            try:
                valid_until_time = parse_utc(valid_until)
            except Exception:
                valid_until_time = None
        if valid_until_time is None or valid_until_time <= at:
            self.llm_policy = {
                "action": "FREEZE",
                "max_open_positions": 0,
                "source": receipt_identity,
                "valid_until": valid_until,
                "effective_at_utc": receipt["arrival_timestamp_utc"],
            }
            return
        if (
            output.get("action") not in ALLOWED_ACTIONS
            or type(output.get("max_open_positions")) is not int
            or not 0
            <= output["max_open_positions"]
            <= int(LLM_POLICY_CONFIG["hard_max_open_positions"])
        ):
            self.llm_policy = {
                "action": "FREEZE",
                "max_open_positions": 0,
                "source": receipt_identity,
                "valid_until": valid_until,
                "effective_at_utc": receipt["arrival_timestamp_utc"],
            }
            return
        hard_cap = int(LLM_POLICY_CONFIG["hard_max_open_positions"])
        self.llm_policy = {
            "action": output["action"],
            "max_open_positions": min(hard_cap, int(output["max_open_positions"])),
            "source": receipt_identity,
            "valid_until": valid_until,
            "effective_at_utc": receipt["arrival_timestamp_utc"],
        }

    def _refresh_llm_policy(self) -> None:
        self.llm_receipts.refresh()
        self._set_llm_policy_at(datetime.now(timezone.utc))

    def _paper_capacity_reason(self, arm: str, instrument: str) -> str | None:
        relevant_open = [
            position for position in self.open_positions.values()
            if position["execution_arm"] == arm
        ]
        relevant_pending = [
            order for order in self.pending_orders.values()
            if order["execution_arm"] == arm
        ]
        if any(position["instrument"] == instrument for position in relevant_open) or any(
            order["instrument"] == instrument for order in relevant_pending
        ):
            return "HARD_INSTRUMENT_INVENTORY_CAP"
        hard_total = int(PAPER_CONFIG["hard_max_open_positions_total"])
        if len(relevant_open) + len(relevant_pending) >= hard_total:
            return "HARD_TOTAL_INVENTORY_CAP"
        if arm == "ACTUAL_LLM_INVENTORY":
            if self.llm_policy["action"] != "ADD":
                return f"LLM_MODE_{self.llm_policy['action']}"
            if len(relevant_open) + len(relevant_pending) >= self.llm_policy["max_open_positions"]:
                return "LLM_OPEN_POSITION_CAP"
        return None

    def _emit_paper_signal(
        self,
        bar_row: dict[str, Any],
        *,
        decision_arrival: datetime,
    ) -> None:
        bar = bar_row["payload"]
        if decision_arrival <= parse_utc(bar["arrival_watermark_utc"]):
            raise IntegrityError("PAPER_DECISION_ARRIVAL_NOT_AFTER_BAR")
        history = self.histories.setdefault(bar["instrument"], [])
        if not history or history[-1].get("start_utc") != bar["start_utc"]:
            history.append(bar)
        signal = evaluate_completed_bar_signal(history, PAPER_CONFIG)
        if signal is None:
            planned: list[dict[str, Any]] = []
            self.control.plan(
                {
                    "event": "PAPER_RAW_SIGNAL_NOT_EMITTED",
                    "strategy_id": PAPER_CONFIG["strategy_id"],
                    "bar_record_hash": bar_row["record_hash"],
                    "reason": "STRUCTURE_NOT_ALIGNED_OR_WARMUP_INCOMPLETE",
                    "entry_cost_gate_used": False,
                    "external_orders": 0,
                },
                f"paper-no-signal::{bar_row['record_hash']}",
                planned,
            )
            self.control.append_rows(planned)
            return
        minimum = max(
            PAPER_CONFIG["slow_ema_bars"] + 1,
            PAPER_CONFIG["momentum_bars"] + 1,
            PAPER_CONFIG["atr_bars"] + 1,
        )
        input_window = completed_bar_input_window(history, minimum)
        if input_window is None or input_window[-1]["start_utc"] != bar["start_utc"]:
            raise IntegrityError("PAPER_FEATURE_WINDOW_MISMATCH")
        signal_id = "paper-signal::" + canonical_hash(
            {
                "strategy_id": PAPER_CONFIG["strategy_id"],
                "bar_record_hash": bar_row["record_hash"],
                "paper_config_sha256": canonical_hash(PAPER_CONFIG),
            }
        )
        proposal = {
            "schema_version": 1,
            "event": "RAW_SIGNAL",
            "signal_id": signal_id,
            "strategy_id": PAPER_CONFIG["strategy_id"],
            "runtime_hash": SHARED_RUNTIME_HASH,
            "service_attestation_hash": SERVICE_ATTESTATION_HASH,
            "instrument": bar["instrument"],
            "decision_source_time_utc": bar["end_utc"],
            "decision_arrival_watermark_utc": bar["arrival_watermark_utc"],
            "decision_arrival_time_utc": utc_text(decision_arrival),
            "completed_bar_hash": bar_row["record_hash"],
            "feature_input_window_sha256": canonical_hash(input_window),
            "paper_config_sha256": canonical_hash(PAPER_CONFIG),
            **signal,
            "research_status": "RESEARCH_NOT_ADMITTED",
            "shadow_observation": True,
            "profit_proven": False,
            "external_order_attempts": 0,
            "external_orders": 0,
        }
        proposal_row = self._append_paper("proposals", proposal, signal_id)
        decision_time = parse_utc(bar["end_utc"])
        for arm in PAPER_CONFIG["arms"]:
            expected_order_id = f"expected::{signal_id}::{arm}"
            blocked_reason = self._paper_capacity_reason(arm, bar["instrument"])
            expected = {
                "schema_version": 1,
                "event": "EXPECTED_ORDER",
                "expected_order_id": expected_order_id,
                "signal_id": signal_id,
                "proposal_record_hash": proposal_row["record_hash"],
                "execution_arm": arm,
                "instrument": bar["instrument"],
                "direction": signal["direction"],
                "units": PAPER_CONFIG["virtual_units"],
                "decision_source_time_utc": bar["end_utc"],
                "decision_arrival_watermark_utc": bar["arrival_watermark_utc"],
                "decision_arrival_time_utc": utc_text(decision_arrival),
                "order_expires_at_utc": utc_text(
                    decision_time
                    + timedelta(minutes=5 * PAPER_CONFIG["expected_order_ttl_bars"])
                ),
                "tp_distance_price": signal["tp_distance_price"],
                "max_age_bars": PAPER_CONFIG["max_age_bars"],
                "status": "PENDING" if blocked_reason is None else "BLOCKED",
                "blocked_reason": blocked_reason,
                "external_submission_allowed": False,
                "external_order_attempts": 0,
                "external_orders": 0,
            }
            self._append_paper("expected_orders", expected, expected_order_id)
            if blocked_reason is None:
                self.pending_orders[expected_order_id] = {
                    **expected,
                    "latency_remaining": PAPER_CONFIG["arms"][arm]["entry_latency_events"],
                    "latency_consumed_event_ids": set(),
                }

    def _seal(
        self,
        instrument: str,
        bucket: int,
        events: list[dict[str, Any]],
        *,
        replay: bool,
        decision_arrival: datetime | None = None,
    ) -> None:
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
        if not replay and row["payload"].get("feed_continuity_eligible") is True:
            if decision_arrival is None:
                raise IntegrityError("PAPER_DECISION_ARRIVAL_MISSING")
            self._emit_paper_signal(row, decision_arrival=decision_arrival)

    def _quote_to_jpy(
        self,
        instrument: str,
        quote_amount: float,
        valuation_event: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind quote-currency PnL to one causal executable JPY conversion."""
        quote_currency = instrument.split("_", 1)[1]
        if quote_currency == "JPY":
            return {
                "pnl_jpy": quote_amount,
                "jpy_conversion_status": "QUOTE_IS_JPY",
                "conversion_bbo_event_id": None,
                "conversion_source_time_utc": None,
                "conversion_arrival_time_utc": None,
                "conversion_bid": None,
                "conversion_ask": None,
                "conversion_rate": 1.0,
                "conversion_side": "QUOTE_IS_JPY",
                "conversion_quote_age_seconds": 0.0,
                "conversion_tradeable": None,
                "conversion_liquidity": None,
            }
        if quote_currency == "USD":
            usd_jpy = self.last_quotes.get("USD_JPY")
            if usd_jpy is None:
                return self._null_jpy_conversion("USD_JPY_CONVERSION_MISSING")
            valuation_source = parse_utc(valuation_event["event_time_utc"])
            valuation_arrival = parse_utc(valuation_event["arrival_time_utc"])
            conversion_source = parse_utc(usd_jpy["event_time_utc"])
            conversion_arrival = parse_utc(usd_jpy["arrival_time_utc"])
            age_seconds = (valuation_arrival - conversion_arrival).total_seconds()
            side = "BID" if quote_amount >= 0 else "ASK"
            liquidity_field = "bid_liquidity" if side == "BID" else "ask_liquidity"
            liquidity = usd_jpy.get(liquidity_field)
            rate = float(usd_jpy["bid"] if side == "BID" else usd_jpy["ask"])
            rejected = {
                "conversion_bbo_event_id": usd_jpy.get("event_id"),
                "conversion_source_time_utc": usd_jpy.get("event_time_utc"),
                "conversion_arrival_time_utc": usd_jpy.get("arrival_time_utc"),
                "conversion_bid": usd_jpy.get("bid"),
                "conversion_ask": usd_jpy.get("ask"),
                "conversion_rate": rate,
                "conversion_side": side,
                "conversion_quote_age_seconds": age_seconds,
                "conversion_tradeable": usd_jpy.get("tradeable"),
                "conversion_liquidity": liquidity,
            }
            if conversion_source > valuation_source or age_seconds < 0:
                return self._null_jpy_conversion(
                    "USD_JPY_CONVERSION_FUTURE",
                    **rejected,
                )
            if age_seconds > PAPER_CONFIG["jpy_conversion_quote_max_age_seconds"]:
                return self._null_jpy_conversion(
                    "USD_JPY_CONVERSION_STALE",
                    **rejected,
                )
            if (
                usd_jpy.get("tradeable") is not True
                or isinstance(liquidity, bool)
                or not isinstance(liquidity, (int, float))
                or not math.isfinite(float(liquidity))
                or float(liquidity) < PAPER_CONFIG["virtual_units"]
            ):
                return self._null_jpy_conversion(
                    "USD_JPY_CONVERSION_NOT_EXECUTABLE",
                    **rejected,
                )
            return {
                "pnl_jpy": quote_amount * rate,
                "jpy_conversion_status": f"USD_JPY_EXECUTABLE_{side}_AT_VALUATION",
                "conversion_bbo_event_id": usd_jpy["event_id"],
                "conversion_source_time_utc": usd_jpy["event_time_utc"],
                "conversion_arrival_time_utc": usd_jpy["arrival_time_utc"],
                "conversion_bid": usd_jpy["bid"],
                "conversion_ask": usd_jpy["ask"],
                "conversion_rate": rate,
                "conversion_side": side,
                "conversion_quote_age_seconds": age_seconds,
                "conversion_tradeable": True,
                "conversion_liquidity": float(liquidity),
            }
        return self._null_jpy_conversion("UNSUPPORTED_QUOTE_CURRENCY")

    @staticmethod
    def _null_jpy_conversion(status: str, **candidate: Any) -> dict[str, Any]:
        return {
            "pnl_jpy": None,
            "jpy_conversion_status": status,
            "conversion_bbo_event_id": candidate.get("conversion_bbo_event_id"),
            "conversion_source_time_utc": candidate.get("conversion_source_time_utc"),
            "conversion_arrival_time_utc": candidate.get("conversion_arrival_time_utc"),
            "conversion_bid": candidate.get("conversion_bid"),
            "conversion_ask": candidate.get("conversion_ask"),
            "conversion_rate": candidate.get("conversion_rate"),
            "conversion_side": candidate.get("conversion_side"),
            "conversion_quote_age_seconds": candidate.get("conversion_quote_age_seconds"),
            "conversion_tradeable": candidate.get("conversion_tradeable"),
            "conversion_liquidity": candidate.get("conversion_liquidity"),
        }

    def _llm_inventory_summary(self) -> dict[str, Any]:
        positions = []
        for position in sorted(self.open_positions.values(), key=lambda value: value["position_id"]):
            if position["execution_arm"] != "ACTUAL_LLM_INVENTORY":
                continue
            quote = self.last_quotes.get(position["instrument"])
            unrealized_pips = None
            if quote is not None:
                mark = virtual_price(
                    quote,
                    position["direction"],
                    PAPER_CONFIG["arms"][position["execution_arm"]],
                    entry=False,
                )
                unrealized_pips = pnl_pips(
                    position["entry_price"],
                    mark,
                    position["direction"],
                    position["instrument"],
                )
            positions.append(
                {
                    "position_id": position["position_id"],
                    "instrument": position["instrument"],
                    "direction": position["direction"],
                    "opened_at_utc": position["fill_source_time_utc"],
                    "max_age_at_utc": position["max_age_at_utc"],
                    "unrealized_pips": unrealized_pips,
                }
            )
        realized = sum(
            float(row["payload"].get("pnl_jpy") or 0.0)
            for row in self.paper_ledgers["pnl"].rows
            if row["payload"].get("execution_arm") == "ACTUAL_LLM_INVENTORY"
            and row["payload"].get("event") == "REALIZED_PNL"
        )
        return {
            "positions": positions,
            "open_inventory_count": len(positions),
            "realized_pnl_jpy": realized,
            "hard_max_open_positions": LLM_POLICY_CONFIG["hard_max_open_positions"],
            "current_policy": copy.deepcopy(self.llm_policy),
        }

    def _write_llm_trigger(self, event_kind: str, at: datetime) -> None:
        summary = self._llm_inventory_summary()
        snapshot_hash = canonical_hash(summary)
        trigger_id = canonical_hash(
            {
                "event_kind": event_kind,
                "inventory_snapshot_hash": snapshot_hash,
                "inventory_ledger_head": self.paper_ledgers["inventory"].last_hash,
            }
        )
        trigger = {
            "schema_version": 2,
            "trigger_id": trigger_id,
            "runtime_hash": SHARED_RUNTIME_HASH,
            "inventory_snapshot_hash": snapshot_hash,
            "open_inventory_count": summary["open_inventory_count"],
            "created_at_utc": utc_text(at),
            "event_kind": event_kind,
            "inventory_summary": summary,
            "allowed_actions": list(ALLOWED_ACTIONS),
            "hard_guard_mutation_allowed": False,
            "individual_order_control_allowed": False,
            "research_status": "RESEARCH_NOT_ADMITTED",
            "profit_proven": False,
            "external_orders": 0,
        }
        atomic_json(SERVICE_ROOT / "triggers" / "llm_inventory_request.json", trigger)

    def _fill_pending_orders(self, event: dict[str, Any]) -> None:
        event_source = parse_utc(event["event_time_utc"])
        event_arrival = parse_utc(event["arrival_time_utc"])
        for expected_order_id, pending in list(self.pending_orders.items()):
            if pending["instrument"] != event["instrument"]:
                continue
            if (
                event_source < parse_utc(pending["decision_source_time_utc"])
                or event_arrival <= parse_utc(pending["decision_arrival_time_utc"])
            ):
                continue
            if event_source > parse_utc(pending["order_expires_at_utc"]):
                planned: list[dict[str, Any]] = []
                self.control.plan(
                    {
                        "event": "VIRTUAL_ORDER_EXPIRED_NO_FILL",
                        "expected_order_id": expected_order_id,
                        "signal_id": pending["signal_id"],
                        "execution_arm": pending["execution_arm"],
                        "at_utc": event["event_time_utc"],
                        "external_orders": 0,
                    },
                    f"paper-expired::{expected_order_id}",
                    planned,
                )
                self.control.append_rows(planned)
                self.pending_orders.pop(expected_order_id)
                continue
            if not executable_bbo_available(
                event,
                pending["direction"],
                entry=True,
                required_units=pending["units"],
            ):
                continue
            consumed_event_ids = pending["latency_consumed_event_ids"]
            if event["event_id"] in consumed_event_ids:
                continue
            if pending["latency_remaining"] > 0:
                remaining_after = pending["latency_remaining"] - 1
                planned: list[dict[str, Any]] = []
                self.control.plan(
                    {
                        "schema_version": 1,
                        "event": "VIRTUAL_ORDER_LATENCY_EVENT_CONSUMED",
                        "expected_order_id": expected_order_id,
                        "signal_id": pending["signal_id"],
                        "execution_arm": pending["execution_arm"],
                        "bbo_event_id": event["event_id"],
                        "event_source_time_utc": event["event_time_utc"],
                        "event_arrival_time_utc": event["arrival_time_utc"],
                        "remaining_after": remaining_after,
                        "external_order_attempts": 0,
                        "external_orders": 0,
                    },
                    f"paper-latency::{expected_order_id}::{event['event_id']}",
                    planned,
                )
                self.control.append_rows(planned)
                pending["latency_remaining"] -= 1
                consumed_event_ids.add(event["event_id"])
                continue
            arm = pending["execution_arm"]
            arm_config = PAPER_CONFIG["arms"][arm]
            entry_price = virtual_price(
                event,
                pending["direction"],
                arm_config,
                entry=True,
                required_units=pending["units"],
            )
            mid = (float(event["bid"]) + float(event["ask"])) / 2.0
            position_id = f"position::{pending['signal_id']}::{arm}"
            fill = {
                "schema_version": 1,
                "event": "VIRTUAL_FILL",
                "position_id": position_id,
                "expected_order_id": expected_order_id,
                "signal_id": pending["signal_id"],
                "execution_arm": arm,
                "instrument": pending["instrument"],
                "direction": pending["direction"],
                "units": pending["units"],
                "first_executable_bbo_event_id": event["event_id"],
                "fill_source_time_utc": event["event_time_utc"],
                "fill_arrival_time_utc": event["arrival_time_utc"],
                "bid": event["bid"],
                "ask": event["ask"],
                "bid_liquidity": event["bid_liquidity"],
                "ask_liquidity": event["ask_liquidity"],
                "tradeable": True,
                "entry_mid": mid,
                "virtual_entry_price": entry_price,
                "price_mode": arm_config["price_mode"],
                "slippage_pips_per_side": arm_config["slippage_pips_per_side"],
                "latency_events": arm_config["entry_latency_events"],
                "external_submission_allowed": False,
                "external_order_attempts": 0,
                "external_orders": 0,
            }
            fill_row = self._append_paper("virtual_fills", fill, f"fill::{expected_order_id}")
            position = self._position_from_fill(fill_row, pending)
            self._append_paper("inventory", position, f"inventory-open::{position_id}")
            self.open_positions[position_id] = position
            self.pending_orders.pop(expected_order_id)
            if arm == "ACTUAL_LLM_INVENTORY":
                self._write_llm_trigger("INVENTORY_OPENED", event_arrival)

    def _oldest_llm_unwind_target(
        self,
        event_arrival: datetime,
    ) -> tuple[str | None, str | None]:
        source = self.llm_policy.get("source")
        effective_at = self.llm_policy.get("effective_at_utc")
        valid_until = self.llm_policy.get("valid_until")
        if (
            self.llm_policy.get("action") != "UNWIND"
            or not isinstance(source, str)
            or not source
            or source in self.llm_unwind_consumed
            or (
                effective_at is not None
                and event_arrival <= parse_utc(effective_at)
            )
            or (
                valid_until is not None
                and event_arrival >= parse_utc(valid_until)
            )
        ):
            return None, None
        candidates = [
            position
            for position in self.open_positions.values()
            if position["execution_arm"] == "ACTUAL_LLM_INVENTORY"
        ]
        if not candidates:
            return None, source
        oldest = min(
            candidates,
            key=lambda position: (
                parse_utc(position["fill_source_time_utc"]),
                parse_utc(position["fill_arrival_time_utc"]),
                position["position_id"],
            ),
        )
        return oldest["position_id"], source

    def _close_positions(self, event: dict[str, Any]) -> None:
        event_source = parse_utc(event["event_time_utc"])
        event_arrival = parse_utc(event["arrival_time_utc"])
        unwind_target, unwind_source = self._oldest_llm_unwind_target(event_arrival)
        for position_id, position in list(self.open_positions.items()):
            if position["instrument"] != event["instrument"]:
                continue
            arm = position["execution_arm"]
            if (
                event_source < parse_utc(position["fill_source_time_utc"])
                or event_arrival <= parse_utc(position["fill_arrival_time_utc"])
                or not executable_bbo_available(
                    event,
                    position["direction"],
                    entry=False,
                    required_units=position["units"],
                )
            ):
                continue
            exit_price = virtual_price(
                event,
                position["direction"],
                PAPER_CONFIG["arms"][arm],
                entry=False,
                required_units=position["units"],
            )
            tp_hit = (
                exit_price >= position["tp_price"]
                if position["direction"] > 0
                else exit_price <= position["tp_price"]
            )
            llm_unwind = position_id == unwind_target and unwind_source is not None
            aged_out = event_source >= parse_utc(position["max_age_at_utc"])
            if not (tp_hit or llm_unwind or aged_out):
                continue
            reason = "TP" if tp_hit else "LLM_OLDEST_FIRST_UNWIND" if llm_unwind else "MAX_AGE"
            quote_amount = quote_pnl(
                position["entry_price"],
                exit_price,
                position["direction"],
                position["units"],
            )
            conversion = self._quote_to_jpy(
                position["instrument"],
                quote_amount,
                event,
            )
            exit_mid = (float(event["bid"]) + float(event["ask"])) / 2.0
            gross_pips = pnl_pips(
                position["entry_mid"],
                exit_mid,
                position["direction"],
                position["instrument"],
            )
            net_pips = pnl_pips(
                position["entry_price"],
                exit_price,
                position["direction"],
                position["instrument"],
            )
            close = {
                "schema_version": 1,
                "event": "CLOSE",
                "position_id": position_id,
                "signal_id": position["signal_id"],
                "execution_arm": arm,
                "instrument": position["instrument"],
                "direction": position["direction"],
                "units": position["units"],
                "reason": reason,
                "exit_bbo_event_id": event["event_id"],
                "exit_source_time_utc": event["event_time_utc"],
                "exit_arrival_time_utc": event["arrival_time_utc"],
                "virtual_exit_price": exit_price,
                "exit_mid": exit_mid,
                "gross_pips": gross_pips,
                "execution_cost_pips": gross_pips - net_pips,
                "net_pips": net_pips,
                "break_even_round_trip_cost_pips": gross_pips,
                "pnl_quote": quote_amount,
                **conversion,
                "llm_unwind_policy_consumed": llm_unwind,
                "llm_policy_source": unwind_source if llm_unwind else None,
                "external_orders": 0,
            }
            close_row = self._append_paper(
                "inventory",
                close,
                f"inventory-close::{position_id}",
            )
            pnl = self._realized_pnl_from_close(position, close_row)
            self._append_paper("pnl", pnl, f"pnl::{position_id}")
            self.open_positions.pop(position_id)
            if llm_unwind and unwind_source is not None:
                self.llm_unwind_consumed.add(unwind_source)
            if arm == "ACTUAL_LLM_INVENTORY":
                self._write_llm_trigger("INVENTORY_CLOSED", event_arrival)

    def record_terminal_mtm(self) -> None:
        """Persist every still-open virtual position at the latest observed BBO."""
        for position_id, position in self.open_positions.items():
            quote = self.last_quotes.get(position["instrument"])
            if quote is None:
                continue
            exit_price = virtual_price(
                quote,
                position["direction"],
                PAPER_CONFIG["arms"][position["execution_arm"]],
                entry=False,
            )
            quote_amount = quote_pnl(
                position["entry_price"],
                exit_price,
                position["direction"],
                position["units"],
            )
            conversion = self._quote_to_jpy(
                position["instrument"],
                quote_amount,
                quote,
            )
            payload = {
                "schema_version": 1,
                "event": "TERMINAL_MTM",
                "position_id": position_id,
                "signal_id": position["signal_id"],
                "execution_arm": position["execution_arm"],
                "instrument": position["instrument"],
                "mark_bbo_event_id": quote["event_id"],
                "mark_source_time_utc": quote["event_time_utc"],
                "virtual_exit_price": exit_price,
                "unrealized_pips": pnl_pips(
                    position["entry_price"],
                    exit_price,
                    position["direction"],
                    position["instrument"],
                ),
                "unrealized_pnl_quote": quote_amount,
                "unrealized_pnl_jpy": conversion["pnl_jpy"],
                **{
                    key: value
                    for key, value in conversion.items()
                    if key != "pnl_jpy"
                },
                "terminal_mtm_included": True,
                "realized": False,
                "research_status": "RESEARCH_NOT_ADMITTED",
                "profit_proven": False,
                "external_order_attempts": 0,
                "external_orders": 0,
            }
            self._append_paper(
                "pnl",
                payload,
                f"terminal-mtm::{position_id}::{quote['event_id']}",
            )

    def _consume_new_rows(self, *, replay: bool = False) -> None:
        for row in self.feed_raw.rows[self.processed_rows:]:
            event = {**row["payload"], "raw_record_hash": row["record_hash"]}
            instrument = event["instrument"]
            event_arrival = parse_utc(event["arrival_time_utc"])
            if not replay:
                self._set_llm_policy_at(event_arrival)
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
            self.last_quotes[instrument] = event
            if current is None:
                self.buckets[instrument] = (bucket, [event])
            elif bucket == current[0]:
                if not replay:
                    self._close_positions(event)
                    self._fill_pending_orders(event)
                current[1].append(event)
            elif bucket > current[0]:
                if not replay:
                    self._close_positions(event)
                self._seal(
                    instrument,
                    current[0],
                    current[1],
                    replay=replay,
                    decision_arrival=event_arrival if not replay else None,
                )
                self.buckets[instrument] = (bucket, [event])
        self.processed_rows = len(self.feed_raw.rows)

    def process_once(self) -> dict[str, int]:
        warmup_rows = len(self.historical_warmup.rows)
        self.historical_warmup.refresh()
        if len(self.historical_warmup.rows) != warmup_rows:
            self._rebuild_histories()
        self.feed_control.refresh()
        self.declared_provenance = _declared_feed_provenance(self.feed_control)
        self.feed_raw.refresh()
        self._refresh_llm_policy()
        self._consume_new_rows(replay=False)
        return _bot_counts(
            self.completed,
            self.control,
            len(self.feed_raw.rows),
            self.paper_ledgers,
            len(self.open_positions),
            len(self.historical_warmup.rows),
        )


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
        processor.process_once()
        processor.record_terminal_mtm()
        counters = processor.process_once()
        lock.heartbeat(service_status(counters, "STOPPED_GRACEFULLY"))
    return 0


def llm_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
            "max_open_positions": {
                "type": "integer",
                "minimum": 0,
                "maximum": int(LLM_POLICY_CONFIG["hard_max_open_positions"]),
            },
            "mode": {"type": "string", "enum": ["SHADOW_ONLY"]},
            "valid_until": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string", "maxLength": 240},
        },
        "required": [
            "action",
            "max_open_positions",
            "mode",
            "valid_until",
            "confidence",
            "reason",
        ],
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
    required = {
        "schema_version",
        "trigger_id",
        "runtime_hash",
        "inventory_snapshot_hash",
        "open_inventory_count",
        "created_at_utc",
        "event_kind",
        "inventory_summary",
        "allowed_actions",
        "hard_guard_mutation_allowed",
        "individual_order_control_allowed",
        "research_status",
        "profit_proven",
        "external_orders",
    }
    if set(trigger) != required or trigger["runtime_hash"] != SHARED_RUNTIME_HASH:
        raise RuntimeError("LLM_TRIGGER_SCHEMA_MISMATCH")
    if (
        trigger["schema_version"] != 2
        or trigger["allowed_actions"] != list(ALLOWED_ACTIONS)
        or trigger["hard_guard_mutation_allowed"] is not False
        or trigger["individual_order_control_allowed"] is not False
        or trigger["research_status"] != "RESEARCH_NOT_ADMITTED"
        or trigger["profit_proven"] is not False
        or trigger["external_orders"] != 0
        or canonical_hash(trigger["inventory_summary"]) != trigger["inventory_snapshot_hash"]
        or trigger["inventory_summary"].get("open_inventory_count")
        != trigger["open_inventory_count"]
    ):
        raise RuntimeError("LLM_TRIGGER_AUTHORITY_MISMATCH")
    record_id = f"llm::{trigger['trigger_id']}"
    if any(row["record_id"] == record_id for row in receipts.rows):
        return {"triggers": 1, "llm_calls": len(receipts.rows), "external_order_attempts": 0, "external_orders": 0}
    request_time = datetime.now(timezone.utc)
    created_at = parse_utc(trigger["created_at_utc"])
    if created_at > request_time + timedelta(seconds=5) or request_time - created_at > timedelta(minutes=10):
        raise RuntimeError("LLM_TRIGGER_STALE")
    prompt = (
        "Manage only the supplied virtual FX inventory. Return one JSON decision. "
        "Allowed action: ADD/FREEZE/UNWIND/RESET. mode=SHADOW_ONLY; "
        f"max_open_positions=0..{LLM_POLICY_CONFIG['hard_max_open_positions']}; "
        "valid_until within 2h. ADD permits only bot-generated future proposals within the cap. "
        "FREEZE stops adds. UNWIND asks the bot to close oldest inventory deterministically. "
        "RESET is valid only when flat. Do not select an individual order, direction, fill, TP, SL, "
        "leverage, cost, or hard guard. External orders=0. "
        f"request_time={utc_text(request_time)} snapshot={canonical_bytes(trigger).decode()}"
    )
    output = runner(prompt)
    if set(output) != {
        "action",
        "max_open_positions",
        "mode",
        "valid_until",
        "confidence",
        "reason",
    }:
        raise RuntimeError("LLM_OUTPUT_SCHEMA_MISMATCH")
    if output["action"] not in ALLOWED_ACTIONS or output["mode"] != "SHADOW_ONLY":
        raise RuntimeError("LLM_OUTPUT_AUTHORITY_MISMATCH")
    if (
        type(output["max_open_positions"]) is not int
        or not 0
        <= output["max_open_positions"]
        <= int(LLM_POLICY_CONFIG["hard_max_open_positions"])
    ):
        raise RuntimeError("LLM_OUTPUT_CAP_MISMATCH")
    if output["action"] == "RESET" and trigger["open_inventory_count"] != 0:
        raise RuntimeError("LLM_RESET_REQUIRES_FLAT_INVENTORY")
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
        "paper_signals": int(bot["counters"].get("natural_paper_proposals", 0)),
        "virtual_fills": int(bot["counters"].get("virtual_fills", 0)),
        "virtual_exits": int(bot["counters"].get("virtual_exits", 0)),
        "open_inventory_count": int(bot["counters"].get("open_inventory_count", 0)),
        "pnl_records": int(bot["counters"].get("pnl_records", 0)),
        "llm_calls": int(bot["counters"].get("llm_calls", 0)),
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
