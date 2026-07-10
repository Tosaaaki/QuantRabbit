#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from quant_rabbit.guardian_events import (
    DEFAULT_FRESH_ACTION_THROTTLE_SECONDS,
    DEFAULT_THROTTLE_SECONDS,
    GUARDIAN_ACTIONS,
    SEVERITY_RANK,
    GuardianEvent,
    review_guardian_action_receipt,
)


MODEL = "gpt-5.5"
ACTIONABLE_ACTIONS = {"TRADE", "ADD", "HARVEST", "REDUCE", "CANCEL_PENDING"}
THESIS_STATES = {"ALIVE", "WOUNDED", "INVALIDATED", "EMERGENCY"}
DEFAULT_LIVE_ROOT = Path("/Users/tossaki/App/QuantRabbit-live")
DEFAULT_CODEX_APP_BIN = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
LEGACY_CODEX_APP_BIN = Path("/Applications/Codex.app/Contents/Resources/codex")

# Local compatibility floor for GPT-5.5 support: codex-cli 0.25.0 returns
# "gpt-5.5 model requires a newer version of Codex", while 0.142.4 accepts the
# model. This is a CLI compatibility guard, not market/risk logic.
MIN_GPT55_CODEX_VERSION = (0, 142, 0)
CODEX_MODEL_FAILURE_STATUSES = {"CODEX_MODEL_UNSUPPORTED", "CODEX_CLI_VERSION_UNSUPPORTED"}

# Guardian wake runs every 30s from a router that just refreshed broker truth;
# older snapshots are likely from a previous operating window and should wait
# for the normal trader refresh instead of prompting GPT from stale quotes.
BROKER_SNAPSHOT_MAX_AGE_SECONDS = 5 * 60

# Codex wake should finish well inside one event-driven review window. A hung
# local CLI must not hold repeated launchd invocations open indefinitely.
CODEX_TIMEOUT_SECONDS = 8 * 60

# Receipt lifecycle follows the hourly trader cadence with sync/launchd grace.
# It is a runtime artifact retention window, not market or risk geometry.
DEFAULT_RECEIPT_TTL_SECONDS = 75 * 60
TERMINAL_RECEIPT_LIFECYCLES = {"CONSUMED", "SUPERSEDED", "EXPIRED", "REJECTED"}

# Failed wake attempts are not reviews. Keep their retry lifecycle separate so
# a repaired Codex binary or a materially changed event can be retried without
# erasing the last accepted review baseline. The cap prevents a malformed GPT
# response from consuming every 30-second launchd tick forever; the TTL opens a
# fresh bounded series for an event that still needs hourly-trader attention.
DEFAULT_RETRY_BASE_SECONDS = 60
DEFAULT_RETRY_MAX_SECONDS = 15 * 60
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_TTL_SECONDS = 30 * 60
DEFAULT_PENDING_DISPATCH_TTL_SECONDS = 2 * 60 * 60
MAX_PENDING_DISPATCHES = 24
DEFAULT_DISK_P0_FREE_BYTES = 2 * 1024**3
DEFAULT_DISK_WARNING_FREE_BYTES = 5 * 1024**3

_EXPLICIT_DISPATCH_REASONS = {
    "NEW_EVENT",
    "SEVERITY_INCREASE",
    "UNKNOWN_GATEWAY_OUTSIDE_ORDER",
    "UNKNOWN_GATEWAY_OUTSIDE_ORDER_STATE_CHANGE",
    "MARGIN_RISK_THRESHOLD_CROSSED",
    "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
    "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE",
    "PRICE_ENTERED_HARVEST_ZONE",
    "FAILED_DISPATCH_RETRY",
}
_MATERIAL_CHANGE_REASONS = {
    "LARGE_PRICE_DISPLACEMENT_STATE_CHANGE",
    "FAILED_ACCEPTANCE_PRICE_ZONE_CHANGE",
    "TECHNICAL_STATE_CHANGE",
    "REGIME_STATE_CHANGE",
    "VOLATILITY_BUCKET_CHANGE",
    "TECHNICAL_FAMILY_STATE_CHANGE",
    "CLOSED_CANDLE_STRUCTURE_CHANGE",
}
_ACTIVE_EXPOSURE_EVENT_TYPES = {
    "HARVEST_ZONE",
    "THESIS_INVALIDATION",
    "MARGIN_PRESSURE",
    "UNKNOWN_ORDER",
    "UNEXPECTED_PROTECTION_MISSING",
    "CONTRACT_HARVEST_TRIGGER",
    "CONTRACT_WOUNDED_TRIGGER",
    "CONTRACT_INVALIDATION_TRIGGER",
    "CONTRACT_EMERGENCY_TRIGGER",
}


@dataclass(frozen=True)
class DispatcherPaths:
    root: Path
    escalation: Path
    events: Path
    event_state: Path
    broker_snapshot: Path
    daily_target_state: Path
    event_report: Path
    prompt_template: Path
    action_receipt: Path
    action_review: Path
    tuning_work_order: Path
    dispatcher_state: Path
    log: Path
    codex_home: Path
    codex_output: Path
    codex_explicit_output: Path
    live_root: Path
    live_lock: Path

    @classmethod
    def from_root(cls, root: Path, *, live_root: Path | None = None) -> "DispatcherPaths":
        live = live_root or Path(os.environ.get("QR_GUARDIAN_WAKE_LIVE_ROOT", str(DEFAULT_LIVE_ROOT)))
        return cls(
            root=root,
            escalation=root / "data" / "guardian_escalation.json",
            events=root / "data" / "guardian_events.json",
            event_state=root / "data" / "guardian_event_state.json",
            broker_snapshot=root / "data" / "broker_snapshot.json",
            daily_target_state=root / "data" / "daily_target_state.json",
            event_report=root / "docs" / "guardian_event_report.md",
            prompt_template=root / "docs" / "guardian_wake_prompt.md",
            action_receipt=root / "data" / "guardian_action_receipt.json",
            action_review=root / "docs" / "guardian_action_review.md",
            tuning_work_order=root / "data" / "guardian_tuning_work_order.json",
            dispatcher_state=root / "data" / "guardian_wake_dispatcher_state.json",
            log=root / "logs" / "guardian_wake_dispatcher.log",
            codex_home=root / "data" / "codex_guardian_home",
            codex_output=root / "data" / "guardian_wake_codex_last_message.md",
            codex_explicit_output=root / "data" / "guardian_wake_codex_explicit_receipt.json",
            live_root=live,
            live_lock=live / ".quant_rabbit_live.lock",
        )


def run_dispatcher(
    *,
    paths: DispatcherPaths,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
    subprocess_run: Callable[..., Any] = subprocess.run,
    snapshot_refresh_run: Callable[..., Any] | None = None,
    codex_preflight_run: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    clock = _utc(now)
    environ = env if env is not None else os.environ
    paths.log.parent.mkdir(parents=True, exist_ok=True)

    escalation = _load_json(paths.escalation)
    events_payload = _load_json(paths.events)
    event_state = _load_json(paths.event_state)
    broker_snapshot = _load_json(paths.broker_snapshot)
    daily_target_state = _load_json(paths.daily_target_state)
    event_report = _read_text(paths.event_report)
    dispatcher_state = _load_json(paths.dispatcher_state)
    runtime_disk = _runtime_disk_state(paths.root, environ)

    escalation, events_payload, retry_injection = _inject_due_failed_attempts(
        escalation=escalation,
        events_payload=events_payload,
        dispatcher_state=dispatcher_state,
        now=clock,
        env=environ,
    )
    escalation, events_payload, pending_injection = _inject_pending_dispatches(
        escalation=escalation,
        events_payload=events_payload,
        dispatcher_state=dispatcher_state,
        now=clock,
    )

    result_base = {
        "generated_at_utc": clock.isoformat(),
        "model": MODEL,
        "paths": _paths_payload(paths),
        "execution_boundary": _execution_boundary(),
        "runtime_disk": runtime_disk,
    }
    if retry_injection:
        result_base["retry_injection"] = retry_injection
    if pending_injection:
        result_base["pending_injection"] = pending_injection
    # Expiry archival writes a new JSON file. Under P0 disk pressure preserve
    # the current receipt as-is and let the first recovered cycle expire it;
    # the disk guard must run before any optional archive/session growth.
    if runtime_disk["status"] != "RUNTIME_DISK_P0":
        _expire_current_receipt_if_needed(paths.action_receipt, now=clock)

    if escalation.get("wake_gpt") is not True:
        result = {
            **result_base,
            "status": "NO_WAKE",
            "wake_gpt": False,
            "reason": "guardian_escalation wake_gpt is not true",
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_non_accepted_current_receipt(paths.action_receipt)
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
        _append_log(paths.log, result)
        return result

    selected, selection = _select_dispatch_event(
        escalation=escalation,
        events_payload=events_payload,
        dispatcher_state=dispatcher_state,
        now=clock,
        env=environ,
    )
    if selected is None:
        result = {
            **result_base,
            "status": "SUPPRESSED",
            "wake_gpt": True,
            "selection": selection,
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_non_accepted_current_receipt(paths.action_receipt)
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
        _append_log(paths.log, result)
        return result

    if runtime_disk["status"] == "RUNTIME_DISK_P0":
        result = {
            **result_base,
            "status": "RUNTIME_DISK_P0",
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "queued_for_active_trader": True,
            "queue_reason": "runtime free space is below the 2 GiB GPT wake floor",
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="runtime free space is below the GPT wake floor",
        )
        _remove_non_accepted_current_receipt(paths.action_receipt)
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result, attempted_event=selected, env=environ)
        _append_log(paths.log, result)
        return result

    lock = _active_live_lock(paths.live_lock)
    if lock["active"]:
        queued = {
            **result_base,
            "status": "QUEUED_FOR_ACTIVE_TRADER",
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "lock": lock,
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _mark_escalation_queued(paths.escalation, escalation, queued, now=clock, reason="active qr-trader/live gateway lock")
        _write_action_review(paths.action_review, queued, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, queued)
        _append_log(paths.log, queued)
        return queued

    freshness = _broker_snapshot_freshness(broker_snapshot, now=clock, env=environ)
    if not freshness["fresh"]:
        refresh = _refresh_broker_snapshot(
            paths=paths,
            env=environ,
            subprocess_run=snapshot_refresh_run or subprocess.run,
        )
        if refresh["status"] == "REFRESHED":
            broker_snapshot = _load_json(paths.broker_snapshot)
            freshness = _broker_snapshot_freshness(broker_snapshot, now=clock, env=environ)
            freshness["refresh_attempt"] = refresh
        else:
            freshness["refresh_attempt"] = refresh
    if not freshness["fresh"]:
        result = {
            **result_base,
            "status": "BROKER_SNAPSHOT_STALE",
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "broker_snapshot_freshness": freshness,
            "queued_for_active_trader": True,
            "queue_reason": "stale broker snapshot; GPT wake not started because stale broker truth must not enter the prompt",
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="stale broker snapshot; GPT wake not started",
        )
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
        _append_log(paths.log, result)
        return result

    codex_preflight = _run_codex_preflight(
        env=environ,
        subprocess_run=codex_preflight_run or subprocess.run,
    )
    if codex_preflight.get("enabled") and str(codex_preflight.get("status") or "").upper() in CODEX_MODEL_FAILURE_STATUSES:
        result = {
            **result_base,
            "status": codex_preflight["status"],
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "broker_snapshot_freshness": freshness,
            "codex_preflight": codex_preflight,
            "parse": {
                "valid": False,
                "error": codex_preflight["status"],
                "raw_output_excerpt": str(
                    codex_preflight.get("stderr_excerpt") or codex_preflight.get("stdout_excerpt") or ""
                )[:2000],
            },
            "receipt_written": False,
            "action_receipt_path": None,
            "queued_for_active_trader": True,
            "queue_reason": codex_preflight.get("remediation_hint")
            or "Codex CLI/model compatibility failed before guardian wake",
        }
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="Codex CLI/model compatibility failed before guardian wake",
        )
        _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
        _record_state(paths.dispatcher_state, dispatcher_state, result, attempted_event=selected, env=environ)
        _append_log(paths.log, result)
        return result

    _prepare_codex_home(paths.codex_home, live_root=paths.live_root)
    prompt = _build_prompt(
        paths=paths,
        selected_event=selected,
        escalation=escalation,
        events_payload=events_payload,
        event_state=event_state,
        broker_snapshot=broker_snapshot,
        daily_target_state=daily_target_state,
        event_report=event_report,
        dispatcher_state=dispatcher_state,
        runtime_disk=runtime_disk,
    )
    codex = _run_codex(
        paths=paths,
        prompt=prompt,
        env=environ,
        subprocess_run=subprocess_run,
        attempt="initial",
        codex_preflight=codex_preflight,
    )

    parsed = _parse_codex_result(codex)
    codex_attempts = [codex]
    repair_attempted = False
    repair_skipped: dict[str, Any] | None = None
    if not parsed["valid"] and parsed.get("error") not in {
        "CODEX_TIMEOUT",
        "CODEX_AUTH_OR_SANDBOX_FAILURE",
        *CODEX_MODEL_FAILURE_STATUSES,
    }:
        retry_lock = _active_live_lock(paths.live_lock)
        if retry_lock["active"]:
            queued = {
                **result_base,
                "status": "QUEUED_FOR_ACTIVE_TRADER",
                "wake_gpt": True,
                "selected_event": selected,
                "selection": selection,
                "broker_snapshot_freshness": freshness,
                "codex": codex,
                "codex_attempts": codex_attempts,
                "parse": parsed,
                "receipt_written": False,
                "action_receipt_path": None,
                "lock": retry_lock,
                "queue_reason": "active qr-trader/live gateway lock before parse repair retry",
            }
            _mark_escalation_queued(
                paths.escalation,
                escalation,
                queued,
                now=clock,
                reason="active qr-trader/live gateway lock before parse repair retry",
            )
            _write_action_review(paths.action_review, queued, action_receipt_path=paths.action_receipt, now=clock)
            _record_state(paths.dispatcher_state, dispatcher_state, queued, attempted_event=selected, env=environ)
            _append_log(paths.log, queued)
            return queued
        repair_attempted = True
        repair_prompt = (
            prompt.rstrip()
            + "\n\nReturn only the required JSON object. No markdown. No prose.\n"
        )
        repair_codex = _run_codex(
            paths=paths,
            prompt=repair_prompt,
            env=environ,
            subprocess_run=subprocess_run,
            attempt="repair",
            codex_preflight=codex_preflight,
        )
        codex_attempts.append(repair_codex)
        parsed = _parse_codex_result(repair_codex)
        codex = repair_codex
    review_events_payload = dict(events_payload)
    review_event_by_key = {
        str(item.get("dedupe_key") or ""): item
        for item in review_events_payload.get("events", []) or []
        if isinstance(item, dict) and str(item.get("dedupe_key") or "")
    }
    review_event_by_key[str(selected.get("dedupe_key") or "")] = {
        key: value for key, value in selected.items() if key != "wake_reason_codes"
    }
    review_events_payload["events"] = list(review_event_by_key.values())
    events = _events_from_payload(review_events_payload)
    review = None
    receipt_written = False
    tuning_handoff: dict[str, Any] = {"status": "SKIPPED_NO_ACCEPTED_MATERIAL_EVENT"}
    if parsed["valid"]:
        if not parsed["receipt"].get("dedupe_key") and selected.get("dedupe_key"):
            parsed["receipt"]["dedupe_key"] = selected["dedupe_key"]
        review = review_guardian_action_receipt(parsed["receipt"], events=events, selected_event=selected, now=clock)
        receipt_action = str(parsed["receipt"].get("action") or "").upper()
        if review.get("status") == "ACCEPTED" and receipt_action in GUARDIAN_ACTIONS:
            superseded = _supersede_current_receipt(
                paths.action_receipt,
                superseded_by_event_id=str(selected.get("event_id") or ""),
                now=clock,
            )
            payload = {
                **review,
                "source": "guardian_wake_dispatcher",
                "model": MODEL,
                "receipt_status": "ACCEPTED",
                "receipt_lifecycle": "ACTIVE",
                "expires_at_utc": _receipt_expires_at(clock, env=environ),
                "consumed_by_trader": False,
                "superseded_by_event_id": None,
                "no_direct_oanda": True,
                "selected_event": selected,
                "selected_event_id": selected.get("event_id"),
                "selected_event_dedupe_key": selected.get("dedupe_key"),
                "dispatcher_status": "RECEIPT_WRITTEN",
                "receipt_id": parsed["receipt"].get("receipt_id") or review.get("generated_at_utc"),
            }
            if superseded:
                payload["superseded_previous_receipt"] = superseded
            _write_json(paths.action_receipt, payload)
            _archive_receipt(paths.action_receipt, payload)
            receipt_written = True
            tuning_handoff = _maybe_write_tuning_work_order(
                path=paths.tuning_work_order,
                selected_event=selected,
                receipt=parsed["receipt"],
                now=clock,
            )
        else:
            _remove_non_accepted_current_receipt(paths.action_receipt)
    else:
        _remove_non_accepted_current_receipt(paths.action_receipt)

    handoff = _maybe_gateway_handoff(
        receipt_review=review,
        paths=paths,
        env=environ,
        lock=lock,
    )
    tuning_handoff_failed = str(tuning_handoff.get("status") or "").upper() == "WORK_ORDER_WRITE_FAILED"
    status = (
        "TUNING_HANDOFF_FAILED"
        if receipt_written and tuning_handoff_failed
        else "RECEIPT_WRITTEN"
        if receipt_written
        else parsed["error"]
        if not parsed["valid"] and parsed.get("error") in CODEX_MODEL_FAILURE_STATUSES
        else "PARSE_FAILED"
        if not parsed["valid"]
        else "RECEIPT_EVENT_MISMATCH"
        if review is not None and review.get("status") == "RECEIPT_EVENT_MISMATCH"
        else "RECEIPT_REJECTED"
    )
    if repair_attempted and not parsed["valid"]:
        repair_skipped = {"status": "FAILED_AFTER_ONE_RETRY", "max_retries_per_event": 1}
    result = {
        **result_base,
        "status": status,
        "wake_gpt": True,
        "selected_event": selected,
        "selection": selection,
        "broker_snapshot_freshness": freshness,
        "codex_preflight": codex_preflight,
        "codex": codex,
        "codex_attempts": codex_attempts,
        "parse": parsed,
        "repair_attempted": repair_attempted,
        "repair_result": repair_skipped,
        "receipt_written": receipt_written,
        "action_receipt_path": str(paths.action_receipt) if receipt_written else None,
        "gateway_handoff": handoff,
        "tuning_handoff": tuning_handoff,
        "tuning_handoff_failed": tuning_handoff_failed,
    }
    if review is not None:
        result["receipt_review"] = review
    if status == "RECEIPT_EVENT_MISMATCH":
        result["queued_for_active_trader"] = True
        result["queue_reason"] = "guardian wake receipt did not match selected_event"
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="guardian wake receipt did not match selected_event",
        )
    _write_action_review(paths.action_review, result, action_receipt_path=paths.action_receipt, now=clock)
    _record_state(
        paths.dispatcher_state,
        dispatcher_state,
        result,
        reviewed_event=selected if receipt_written and not tuning_handoff_failed else None,
        attempted_event=selected,
        env=environ,
    )
    _append_log(paths.log, result)
    return result


def _runtime_disk_state(root: Path, env: dict[str, str]) -> dict[str, Any]:
    p0_floor = max(1, int(env.get("QR_GUARDIAN_WAKE_DISK_P0_FREE_BYTES", DEFAULT_DISK_P0_FREE_BYTES)))
    warning_floor = max(
        p0_floor,
        int(env.get("QR_GUARDIAN_WAKE_DISK_WARNING_FREE_BYTES", DEFAULT_DISK_WARNING_FREE_BYTES)),
    )
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        return {
            "status": "RUNTIME_DISK_UNKNOWN",
            "path": str(root),
            "p0_free_bytes": p0_floor,
            "warning_free_bytes": warning_floor,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if usage.free < p0_floor:
        status = "RUNTIME_DISK_P0"
    elif usage.free < warning_floor:
        status = "RUNTIME_DISK_WARNING"
    else:
        status = "RUNTIME_DISK_OK"
    return {
        "status": status,
        "path": str(root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": round(usage.free / 1024**3, 3),
        "p0_free_bytes": p0_floor,
        "warning_free_bytes": warning_floor,
        "gpt_wake_allowed": status != "RUNTIME_DISK_P0",
    }


def _inject_due_failed_attempts(
    *,
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    dispatcher_state: dict[str, Any],
    now: datetime,
    env: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    attempts = dispatcher_state.get("dispatch_attempts")
    if not isinstance(attempts, dict) or not attempts:
        return escalation, events_payload, None
    current_events = {
        str(item.get("dedupe_key") or ""): item
        for item in events_payload.get("events", []) or []
        if isinstance(item, dict) and str(item.get("dedupe_key") or "")
    }
    due: list[dict[str, Any]] = []
    for dedupe_key, record in attempts.items():
        if not isinstance(record, dict):
            continue
        stored_event = record.get("event") if isinstance(record.get("event"), dict) else {}
        # Retry the exact failed observation.  A live quote tick changes the
        # current technical event_id/price_zone, but must not reset backoff or
        # the max-attempt budget.  A genuinely new material router escalation
        # is still merged separately and can start its own reviewed series.
        event = dict(stored_event)
        if not event or not event.get("event_id") or not event.get("dedupe_key"):
            continue
        if _retry_suppression(record, event=event, now=now, env=env) is not None:
            continue
        event["wake_reason_codes"] = list(
            dict.fromkeys([*(event.get("wake_reason_codes") or []), "FAILED_DISPATCH_RETRY"])
        )
        due.append(event)
    if not due:
        return escalation, events_payload, None

    payload = dict(escalation)
    review_events = [item for item in payload.get("events_to_review", []) or [] if isinstance(item, dict)]
    review_by_key = {str(item.get("dedupe_key") or ""): item for item in review_events}
    for event in due:
        review_by_key[str(event.get("dedupe_key") or "")] = event
    payload["wake_gpt"] = True
    payload["events_to_review"] = list(review_by_key.values())
    payload["wake_reason_codes"] = sorted(
        {*(str(item) for item in payload.get("wake_reason_codes", []) or []), "FAILED_DISPATCH_RETRY"}
    )

    event_payload = dict(events_payload)
    event_by_key = dict(current_events)
    for event in due:
        event_by_key[str(event.get("dedupe_key") or "")] = {
            key: value for key, value in event.items() if key != "wake_reason_codes"
        }
    event_payload["events"] = list(event_by_key.values())
    return payload, event_payload, {
        "status": "DUE_FAILED_ATTEMPTS_INJECTED",
        "event_ids": [event.get("event_id") for event in due],
        "count": len(due),
    }


def _inject_pending_dispatches(
    *,
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    dispatcher_state: dict[str, Any],
    now: datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    pending = _active_pending_dispatch_records(
        dispatcher_state.get("pending_dispatches"),
        now=now,
    )
    if not pending:
        return escalation, events_payload, None

    payload = dict(escalation)
    review_events = [
        item for item in payload.get("events_to_review", []) or [] if isinstance(item, dict)
    ]
    review_by_key = {
        str(item.get("dedupe_key") or ""): item
        for item in review_events
        if str(item.get("dedupe_key") or "")
    }
    injected: list[dict[str, Any]] = []
    for dedupe_key, record in pending.items():
        if dedupe_key in review_by_key:
            continue
        event = record.get("event") if isinstance(record.get("event"), dict) else {}
        if not event:
            continue
        review_by_key[dedupe_key] = dict(event)
        injected.append(event)
    if not injected:
        return escalation, events_payload, None

    payload["wake_gpt"] = True
    payload["events_to_review"] = list(review_by_key.values())
    payload["wake_reason_codes"] = sorted(
        {
            *(str(item) for item in payload.get("wake_reason_codes", []) or []),
            "DURABLE_PENDING_DISPATCH",
        }
    )
    event_payload = dict(events_payload)
    event_by_key = {
        str(item.get("dedupe_key") or ""): item
        for item in event_payload.get("events", []) or []
        if isinstance(item, dict) and str(item.get("dedupe_key") or "")
    }
    for event in injected:
        event_by_key[str(event.get("dedupe_key") or "")] = {
            key: value for key, value in event.items() if key != "wake_reason_codes"
        }
    event_payload["events"] = list(event_by_key.values())
    return payload, event_payload, {
        "status": "DURABLE_PENDING_DISPATCHES_INJECTED",
        "event_ids": [event.get("event_id") for event in injected],
        "count": len(injected),
    }


def _active_pending_dispatch_records(value: Any, *, now: datetime) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    active: dict[str, dict[str, Any]] = {}
    for dedupe_key, record in value.items():
        if not isinstance(record, dict):
            continue
        event = record.get("event") if isinstance(record.get("event"), dict) else {}
        expires_at = _parse_utc(record.get("expires_at_utc"))
        if not event or expires_at is None or expires_at <= now:
            continue
        active[str(dedupe_key)] = record
    return active


def _select_dispatch_event(
    *,
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    dispatcher_state: dict[str, Any],
    now: datetime,
    env: dict[str, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    review_events = escalation.get("events_to_review")
    if not isinstance(review_events, list) or not review_events:
        return None, {"status": "NO_EVENTS_TO_REVIEW"}

    reviewed = _accepted_review_records(dispatcher_state.get("reviewed_events"))
    attempts = (
        dispatcher_state.get("dispatch_attempts")
        if isinstance(dispatcher_state.get("dispatch_attempts"), dict)
        else {}
    )
    candidates: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    del events_payload
    for raw_event in review_events:
        if not isinstance(raw_event, dict):
            continue
        # events_to_review is the immutable observation that triggered this
        # wake.  Replacing it with a newer raw quote tick would reset retry
        # identity and could evade backoff indefinitely.
        event = dict(raw_event)
        event["wake_reason_codes"] = list(raw_event.get("wake_reason_codes") or [])
        reasons = {str(item).strip().upper() for item in event.get("wake_reason_codes") or [] if str(item).strip()}
        dedupe_key = str(event.get("dedupe_key") or "").strip()
        if not dedupe_key:
            suppressed.append({"event": event, "reason": "MISSING_DEDUPE_KEY"})
            continue
        if not any(_is_dispatch_reason(reason) for reason in reasons):
            suppressed.append({"event": event, "reason": "NO_DISPATCHABLE_STATE_CHANGE_REASON"})
            continue
        retry_suppression = _retry_suppression(
            attempts.get(dedupe_key) if isinstance(attempts, dict) else None,
            event=event,
            now=now,
            env=env,
        )
        if retry_suppression is not None:
            pending_events.append(event)
            suppressed.append({"event": event, **retry_suppression})
            continue
        record = reviewed.get(dedupe_key) if isinstance(reviewed, dict) else None
        throttle = _dispatch_throttle_seconds(event, env)
        if isinstance(record, dict):
            previous_severity = _severity_rank(record.get("severity"))
            current_severity = _severity_rank(event.get("severity"))
            last_reviewed = _parse_utc(record.get("last_reviewed_at_utc"))
            bypass = _event_bypasses_dispatch_throttle(event, reasons, previous_severity=previous_severity)
            if last_reviewed is not None and not bypass:
                age = (now - last_reviewed).total_seconds()
                if age < throttle:
                    suppressed.append(
                        {
                            "event": event,
                            "reason": "THROTTLED",
                            "age_seconds": age,
                            "throttle_seconds": throttle,
                        }
                    )
                    continue
            if current_severity < previous_severity:
                suppressed.append({"event": event, "reason": "SEVERITY_DECREASED_SINCE_ACCEPTED_REVIEW"})
                continue
            if current_severity == previous_severity and not _accepted_baseline_changed(record, event):
                suppressed.append({"event": event, "reason": "DEDUPE_KEY_ALREADY_REVIEWED"})
                continue
        candidates.append(event)
        pending_events.append(event)

    if not candidates:
        return None, {
            "status": "NO_DISPATCHABLE_EVENT",
            "suppressed": suppressed,
            "pending_events": pending_events,
        }
    candidates.sort(key=_dispatch_priority)
    selected = candidates[0]
    return selected, {
        "status": "SELECTED",
        "suppressed": suppressed,
        "candidate_count": len(candidates),
        "pending_events": pending_events,
        "selected_priority": _dispatch_priority_evidence(selected),
    }


def _dispatch_throttle_seconds(event: dict[str, Any], env: dict[str, str]) -> int:
    if str(event.get("action_hint") or "").upper() in {"TRADE", "ADD"} or str(
        event.get("recommended_review_type") or ""
    ).upper() in {"ENTRY_REVIEW", "ADD_REVIEW"}:
        return int(env.get("QR_GUARDIAN_FRESH_ACTION_THROTTLE_SECONDS", DEFAULT_FRESH_ACTION_THROTTLE_SECONDS))
    return int(env.get("QR_GUARDIAN_EVENT_THROTTLE_SECONDS", DEFAULT_THROTTLE_SECONDS))


def _accepted_review_records(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): record
        for key, record in value.items()
        if isinstance(record, dict) and record.get("receipt_written") is True
    }


def _event_bypasses_dispatch_throttle(
    event: dict[str, Any],
    reasons: set[str],
    *,
    previous_severity: int,
) -> bool:
    severity_increased = _severity_rank(event.get("severity")) > previous_severity
    return severity_increased and (
        str(event.get("severity") or "").upper() == "P0" or "SEVERITY_INCREASE" in reasons
    )


def _is_dispatch_reason(reason: str) -> bool:
    code = str(reason or "").strip().upper()
    if code in _EXPLICIT_DISPATCH_REASONS:
        return True
    if _is_thesis_degrade_reason(code) or "HARVEST" in code:
        return True
    if "TECHNICAL_STATE_CHANGE" in code:
        return True
    # Future router contracts can name the exact regime/family transition
    # without requiring a dispatcher release. events_to_review remains the
    # deterministic upstream boundary, so this does not wake on raw indicators.
    return any(token in code for token in ("REGIME", "VOLATILITY", "FAMILY", "STRUCTURE"))


def _is_thesis_degrade_reason(reason: str) -> bool:
    match = re.fullmatch(r"THESIS_([A-Z]+)_TO_([A-Z]+)", str(reason or "").upper())
    if not match:
        return False
    rank = {"UNKNOWN": 0, "ALIVE": 1, "WOUNDED": 2, "INVALIDATED": 3, "EMERGENCY": 4}
    return rank.get(match.group(2), -1) > rank.get(match.group(1), -1)


def _event_material_fingerprint(event: dict[str, Any]) -> str:
    material = {
        key: event.get(key)
        for key in (
            "event_id",
            "event_type",
            "pair",
            "direction",
            "thesis",
            "price_zone",
            "severity",
            "recommended_review_type",
            "action_hint",
            "thesis_state",
            "details",
        )
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _accepted_baseline_changed(record: dict[str, Any], event: dict[str, Any]) -> bool:
    previous_event_id = str(record.get("event_id") or "")
    current_event_id = str(event.get("event_id") or "")
    if previous_event_id and current_event_id and previous_event_id != current_event_id:
        return True
    previous_fingerprint = str(record.get("material_fingerprint") or "")
    # Accepted records written before material baselines were introduced do
    # not carry enough fields for a local comparison. Trust one explicit
    # router material-change reason after the normal review throttle; the new
    # accepted receipt then writes a complete baseline and restores strict
    # fingerprint comparison for subsequent cycles.
    if not previous_fingerprint and _event_is_active_or_material(event):
        return True
    return bool(previous_fingerprint and previous_fingerprint != _event_material_fingerprint(event))


def _retry_suppression(
    record: Any,
    *,
    event: dict[str, Any],
    now: datetime,
    env: dict[str, str],
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    if str(record.get("event_id") or "") != str(event.get("event_id") or ""):
        return None
    recorded_fingerprint = str(record.get("material_fingerprint") or "")
    if recorded_fingerprint and recorded_fingerprint != _event_material_fingerprint(event):
        return None
    if str(record.get("last_status") or "").upper() == "RUNTIME_DISK_P0":
        prior_disk = record.get("runtime_disk") if isinstance(record.get("runtime_disk"), dict) else {}
        disk_path = Path(str(prior_disk.get("path") or "."))
        if _runtime_disk_state(disk_path, env).get("status") != "RUNTIME_DISK_P0":
            return None
    prior_binary = record.get("codex_binary_identity")
    current_binary = _codex_binary_identity(env)
    if isinstance(prior_binary, dict) and _binary_identity_key(prior_binary) != _binary_identity_key(current_binary):
        return None
    expires_at = _parse_utc(record.get("expires_at_utc"))
    if expires_at is not None and now >= expires_at:
        return None
    retry_after = _parse_utc(record.get("retry_after_utc"))
    attempt_count = int(record.get("attempt_count") or 0)
    max_attempts = max(1, int(env.get("QR_GUARDIAN_WAKE_RETRY_MAX_ATTEMPTS", DEFAULT_RETRY_MAX_ATTEMPTS)))
    if attempt_count >= max_attempts:
        return {
            "reason": "RETRY_BUDGET_EXHAUSTED",
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "retry_after_utc": record.get("retry_after_utc"),
            "expires_at_utc": record.get("expires_at_utc"),
        }
    if retry_after is not None and now < retry_after:
        return {
            "reason": "RETRY_BACKOFF",
            "attempt_count": attempt_count,
            "max_attempts": max_attempts,
            "retry_after_utc": retry_after.isoformat(),
            "expires_at_utc": record.get("expires_at_utc"),
        }
    return None


def _codex_binary_identity(env: dict[str, str]) -> dict[str, Any]:
    path = Path(_resolve_codex_bin(env))
    identity: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        return identity
    identity.update(
        {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
    return identity


def _binary_identity_key(identity: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(identity.get(key) for key in ("path", "device", "inode", "size", "mtime_ns"))


def _event_has_open_exposure(event: dict[str, Any]) -> bool:
    if str(event.get("event_type") or "").upper() in _ACTIVE_EXPOSURE_EVENT_TYPES:
        return True
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    stack: list[Any] = [details]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                key_text = str(key).lower()
                if key_text in {"trade_id", "position_id", "open_trade_id"} and str(value or "").strip():
                    return True
                if key_text in {"units", "open_units", "current_units"}:
                    try:
                        if float(value or 0) != 0:
                            return True
                    except (TypeError, ValueError):
                        pass
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _event_is_active_or_material(event: dict[str, Any]) -> bool:
    reasons = {str(item).strip().upper() for item in event.get("wake_reason_codes") or []}
    return _event_has_open_exposure(event) or any(
        reason in _MATERIAL_CHANGE_REASONS
        or _is_thesis_degrade_reason(reason)
        or "HARVEST" in reason
        or "TECHNICAL_STATE_CHANGE" in reason
        or "REGIME" in reason
        or "VOLATILITY" in reason
        or "FAMILY" in reason
        or "STRUCTURE" in reason
        for reason in reasons
    )


def _dispatch_priority(event: dict[str, Any]) -> tuple[Any, ...]:
    open_exposure = _event_has_open_exposure(event)
    p0 = str(event.get("severity") or "").upper() == "P0"
    active_or_material = _event_is_active_or_material(event)
    return (
        -int(open_exposure or p0),
        -int(open_exposure),
        -_severity_rank(event.get("severity")),
        -int(active_or_material),
        str(event.get("pair") or ""),
        str(event.get("event_type") or ""),
        str(event.get("dedupe_key") or ""),
    )


def _dispatch_priority_evidence(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "open_exposure": _event_has_open_exposure(event),
        "p0": str(event.get("severity") or "").upper() == "P0",
        "active_or_material": _event_is_active_or_material(event),
        "severity": event.get("severity"),
        "alphabetical_tiebreak": [event.get("pair"), event.get("event_type"), event.get("dedupe_key")],
    }


def _broker_snapshot_freshness(payload: dict[str, Any], *, now: datetime, env: dict[str, str]) -> dict[str, Any]:
    raw = payload.get("fetched_at_utc") or (payload.get("account") if isinstance(payload.get("account"), dict) else {}).get(
        "fetched_at_utc"
    )
    fetched = _parse_utc(raw)
    max_age = int(env.get("QR_GUARDIAN_WAKE_MAX_SNAPSHOT_AGE_SECONDS", BROKER_SNAPSHOT_MAX_AGE_SECONDS))
    if fetched is None:
        return {"fresh": False, "status": "MISSING_FETCHED_AT", "max_age_seconds": max_age}
    age = (now - fetched).total_seconds()
    return {
        "fresh": age <= max_age,
        "status": "FRESH" if age <= max_age else "STALE",
        "age_seconds": age,
        "max_age_seconds": max_age,
        "fetched_at_utc": fetched.isoformat(),
    }


def _active_live_lock(lock_dir: Path) -> dict[str, Any]:
    if not lock_dir.exists():
        return {"active": False, "path": str(lock_dir)}
    pid = _read_text(lock_dir / "pid").strip()
    label = _read_text(lock_dir / "command").strip()
    started = _read_text(lock_dir / "started_at_utc").strip()
    active = pid.isdigit() and _pid_running(int(pid))
    command = ""
    if active:
        command = _process_command(int(pid))
    return {
        "active": bool(active),
        "path": str(lock_dir),
        "pid": int(pid) if pid.isdigit() else None,
        "label": label or None,
        "started_at_utc": started or None,
        "process_command": command or None,
    }


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_command(pid: int) -> str:
    try:
        proc = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, timeout=3)
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def _mark_escalation_queued(
    path: Path,
    escalation: dict[str, Any],
    result: dict[str, Any],
    *,
    now: datetime,
    reason: str,
) -> None:
    payload = dict(escalation)
    payload["queued_for_active_trader"] = True
    payload["queued_at_utc"] = now.isoformat()
    payload["queue_reason"] = reason
    payload["guardian_wake_dispatcher_status"] = result.get("status")
    payload["guardian_wake_dispatcher_lock"] = result.get("lock")
    payload["guardian_wake_dispatcher_parse_failure"] = result.get("parse_failure")
    _write_json(path, payload)


def _prepare_codex_home(codex_home: Path, *, live_root: Path) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    auth_src = Path.home() / ".codex" / "auth.json"
    auth_dst = codex_home / "auth.json"
    if auth_src.exists() and not auth_dst.exists():
        auth_dst.symlink_to(auth_src)
    config = (
        f'model = "{MODEL}"\n'
        'approval_policy = "never"\n'
        'sandbox_mode = "read-only"\n'
        "model_reasoning_effort = \"high\"\n"
        "\n"
        f'[projects."{live_root}"]\n'
        'trust_level = "trusted"\n'
    )
    config_path = codex_home / "config.toml"
    if not config_path.exists() or config_path.read_text() != config:
        config_path.write_text(config)


def _refresh_broker_snapshot(
    *,
    paths: DispatcherPaths,
    env: dict[str, str],
    subprocess_run: Callable[..., Any],
) -> dict[str, Any]:
    if env.get("QR_GUARDIAN_WAKE_DISABLE_SNAPSHOT_REFRESH", "0") == "1":
        return {"status": "DISABLED", "reason": "QR_GUARDIAN_WAKE_DISABLE_SNAPSHOT_REFRESH=1"}
    python_bin = env.get("QR_PYTHON", "python3")
    cmd = [
        python_bin,
        "-m",
        "quant_rabbit.cli",
        "broker-snapshot",
        "--output",
        str(paths.broker_snapshot),
    ]
    run_env = dict(env)
    run_env["PYTHONPATH"] = str(paths.root / "src")
    timeout = int(env.get("QR_GUARDIAN_WAKE_SNAPSHOT_REFRESH_TIMEOUT_SECONDS", "90"))
    try:
        proc = subprocess_run(cmd, cwd=str(paths.root), capture_output=True, text=True, timeout=timeout, env=run_env)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "TIMEOUT",
            "command": cmd,
            "timeout_seconds": timeout,
            "stdout_tail": _decoded_tail(exc.stdout),
            "stderr_tail": _decoded_tail(exc.stderr),
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "EXCEPTION", "command": cmd, "error": str(exc)}
    return {
        "status": "REFRESHED" if proc.returncode == 0 and paths.broker_snapshot.exists() else "FAILED",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
        "read_only_broker_operation": True,
    }


def _resolve_codex_bin(env: dict[str, str]) -> str:
    configured = str(env.get("QR_GUARDIAN_WAKE_CODEX_BIN") or "").strip()
    if configured:
        return configured
    # Prefer the desktop-bundled CLI: it is upgraded with the app and can
    # support the requested model while a Homebrew/PATH copy remains older.
    # The app was renamed from Codex.app to ChatGPT.app, so keep the legacy
    # location as a compatibility fallback without selecting it when absent.
    for app_bin in (DEFAULT_CODEX_APP_BIN, LEGACY_CODEX_APP_BIN):
        if app_bin.exists():
            return str(app_bin)
    found = shutil.which("codex")
    if found:
        return found
    return "codex"


def _run_codex_preflight(
    *,
    env: dict[str, str],
    subprocess_run: Callable[..., Any],
) -> dict[str, Any]:
    codex_bin = _resolve_codex_bin(env)
    base = {
        "enabled": env.get("QR_GUARDIAN_WAKE_CODEX_PREFLIGHT", "0") == "1",
        "codex_binary_path": codex_bin,
        "requested_model": MODEL,
        "minimum_supported_version": ".".join(str(item) for item in MIN_GPT55_CODEX_VERSION),
    }
    if not base["enabled"]:
        return {**base, "status": "SKIPPED"}
    cmd = [codex_bin, "--version"]
    timeout = int(env.get("QR_GUARDIAN_WAKE_CODEX_PREFLIGHT_TIMEOUT_SECONDS", "10"))
    try:
        proc = subprocess_run(cmd, capture_output=True, text=True, timeout=timeout, env=dict(env))
    except FileNotFoundError as exc:
        return {
            **base,
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "command": cmd,
            "returncode": None,
            "codex_version": None,
            "stdout_excerpt": "",
            "stderr_excerpt": str(exc),
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "command": cmd,
            "returncode": None,
            "codex_version": None,
            "stdout_excerpt": _decoded_tail(exc.stdout),
            "stderr_excerpt": _decoded_tail(exc.stderr),
            "timeout_seconds": timeout,
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "command": cmd,
            "returncode": None,
            "codex_version": None,
            "stdout_excerpt": "",
            "stderr_excerpt": str(exc),
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    stdout = str(getattr(proc, "stdout", "") or "")
    stderr = str(getattr(proc, "stderr", "") or "")
    version = _parse_codex_version(stdout or stderr)
    version_tuple = _codex_version_tuple(version)
    if getattr(proc, "returncode", 1) != 0:
        status = "CODEX_CLI_VERSION_UNSUPPORTED"
    elif version_tuple is None:
        status = "CODEX_CLI_VERSION_UNSUPPORTED"
    elif version_tuple < MIN_GPT55_CODEX_VERSION:
        status = "CODEX_MODEL_UNSUPPORTED"
    else:
        status = "OK"
    result = {
        **base,
        "status": status,
        "command": cmd,
        "returncode": getattr(proc, "returncode", None),
        "codex_version": version,
        "stdout_excerpt": stdout[-1000:],
        "stderr_excerpt": stderr[-1000:],
        "supports_requested_model": status == "OK",
    }
    if status != "OK":
        result["cli_failure_class"] = "CODEX_CLI_VERSION_UNSUPPORTED"
        result["remediation_hint"] = _codex_remediation_hint(status=status, codex_bin=codex_bin)
    return result


def _build_prompt(
    *,
    paths: DispatcherPaths,
    selected_event: dict[str, Any],
    escalation: dict[str, Any],
    events_payload: dict[str, Any],
    event_state: dict[str, Any],
    broker_snapshot: dict[str, Any],
    daily_target_state: dict[str, Any],
    event_report: str,
    dispatcher_state: dict[str, Any],
    runtime_disk: dict[str, Any],
) -> str:
    template = _read_text(paths.prompt_template)
    if not template:
        template = "Return exactly one guardian wake JSON receipt."
    prompt_payload = {
        "selected_event": selected_event,
        "guardian_escalation": _compact_escalation(escalation),
        "guardian_events": events_payload,
        "guardian_event_state": event_state,
        "broker_snapshot": _compact_broker_snapshot(broker_snapshot, pair=str(selected_event.get("pair") or "")),
        "daily_target_state": daily_target_state,
        "dispatcher_state": dispatcher_state,
        "runtime_disk": runtime_disk,
    }
    return (
        template.rstrip()
        + "\n\nIf the final-message capture is unavailable, also return the same JSON object as the final assistant message. "
        + f"If an explicit output file is available, use this path for the JSON object only: {paths.codex_explicit_output}.\n"
        + "\n\n# Dispatcher Context\n\n"
        + "Use only the following local artifacts. Do not request web, OANDA, broker, or API access.\n\n"
        + "```json\n"
        + _json_dumps(prompt_payload, limit=60_000)
        + "\n```\n\n"
        + "# docs/guardian_event_report.md\n\n"
        + event_report[:20_000]
        + "\n"
    )


def _compact_escalation(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "generated_at_utc",
            "wake_gpt",
            "model_target",
            "wake_policy",
            "wake_reason_codes",
            "events_to_review",
            "execution_boundary",
        )
    }


def _compact_broker_snapshot(payload: dict[str, Any], *, pair: str) -> dict[str, Any]:
    quotes = payload.get("quotes") if isinstance(payload.get("quotes"), dict) else {}
    selected_quotes = {}
    if pair and pair in quotes:
        selected_quotes[pair] = quotes[pair]
    if "USD_JPY" in quotes:
        selected_quotes["USD_JPY"] = quotes["USD_JPY"]
    positions = [
        item
        for item in payload.get("positions", []) or []
        if isinstance(item, dict) and (not pair or str(item.get("pair") or item.get("instrument") or "").upper() == pair.upper())
    ]
    orders = [
        item
        for item in payload.get("orders", []) or []
        if isinstance(item, dict) and (not pair or str(item.get("pair") or item.get("instrument") or "").upper() == pair.upper())
    ]
    return {
        "fetched_at_utc": payload.get("fetched_at_utc"),
        "account": payload.get("account"),
        "positions": positions,
        "orders": orders,
        "quotes": selected_quotes,
    }


def _run_codex(
    *,
    paths: DispatcherPaths,
    prompt: str,
    env: dict[str, str],
    subprocess_run: Callable[..., Any],
    attempt: str,
    codex_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths.codex_output.parent.mkdir(parents=True, exist_ok=True)
    paths.codex_output.write_text("")
    paths.codex_explicit_output.parent.mkdir(parents=True, exist_ok=True)
    paths.codex_explicit_output.write_text("")
    codex_preflight = codex_preflight if isinstance(codex_preflight, dict) else {}
    codex_bin = str(codex_preflight.get("codex_binary_path") or _resolve_codex_bin(env))
    codex_version = codex_preflight.get("codex_version")
    timeout = int(env.get("QR_GUARDIAN_WAKE_CODEX_TIMEOUT_SECONDS", CODEX_TIMEOUT_SECONDS))
    cmd = [
        codex_bin,
        "exec",
        "-",
        "-m",
        MODEL,
        "-s",
        "read-only",
        "-C",
        str(paths.live_root),
        "--json",
        "--output-last-message",
        str(paths.codex_output),
    ]
    run_env = dict(env)
    run_env["CODEX_HOME"] = str(paths.codex_home)
    run_env["CODEX_DISABLE_UPDATE_CHECK"] = "1"
    session_before = _latest_session_jsonl(paths.codex_home)
    session_before_mtime = _path_mtime(session_before)
    try:
        proc = subprocess_run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=run_env)
    except subprocess.TimeoutExpired as exc:
        stdout_text = _decode_process_text(exc.stdout)
        stderr_text = _decode_process_text(exc.stderr)
        session_message, session_path = _latest_session_assistant_message(
            paths.codex_home,
            previous=session_before,
            previous_mtime=session_before_mtime,
        )
        explicit_message = _read_text(paths.codex_explicit_output).strip()
        raw_last_message = _read_text(paths.codex_output).strip()
        candidate, source = _first_message_candidate(
            [
                ("output-last-message", raw_last_message),
                ("explicit-output-file", explicit_message),
                ("stdout-assistant", _extract_stdout_assistant_message(stdout_text)),
                ("session-jsonl-assistant", session_message),
            ]
        )
        return {
            "status": "CODEX_TIMEOUT",
            "attempt": attempt,
            "command": cmd,
            "codex_binary_path": codex_bin,
            "codex_version": codex_version,
            "requested_model": MODEL,
            "returncode": None,
            "stderr_tail": stderr_text[-1000:],
            "stdout_tail": stdout_text[-1000:],
            "raw_stdout_excerpt": stdout_text[-2000:],
            "raw_stderr_excerpt": stderr_text[-2000:],
            "output_path": str(paths.codex_output),
            "explicit_output_path": str(paths.codex_explicit_output),
            "session_jsonl_path": str(session_path) if session_path else None,
            "last_message": candidate,
            "last_message_source": source,
            "last_message_file_empty": not raw_last_message,
            "explicit_output_file_empty": not explicit_message,
            "stdout_fallback_used": source == "stdout-assistant",
            "session_jsonl_fallback_used": source == "session-jsonl-assistant",
            "timeout_seconds": timeout,
        }
    except Exception as exc:
        return {
            "status": "CODEX_EXCEPTION",
            "attempt": attempt,
            "command": cmd,
            "codex_binary_path": codex_bin,
            "codex_version": codex_version,
            "requested_model": MODEL,
            "returncode": None,
            "stderr_tail": str(exc),
            "stdout_tail": "",
            "raw_stdout_excerpt": "",
            "raw_stderr_excerpt": str(exc)[-2000:],
            "output_path": str(paths.codex_output),
            "explicit_output_path": str(paths.codex_explicit_output),
            "session_jsonl_path": None,
            "last_message": "",
            "last_message_source": None,
            "last_message_file_empty": True,
            "explicit_output_file_empty": True,
            "stdout_fallback_used": False,
            "session_jsonl_fallback_used": False,
        }
    raw_last_message = _read_text(paths.codex_output)
    explicit_message = _read_text(paths.codex_explicit_output)
    stdout_text = str(getattr(proc, "stdout", "") or "")
    stderr_text = str(getattr(proc, "stderr", "") or "")
    stdout_message = _extract_stdout_assistant_message(stdout_text)
    session_message, session_path = _latest_session_assistant_message(
        paths.codex_home,
        previous=session_before,
        previous_mtime=session_before_mtime,
    )
    last_message_file_empty = not raw_last_message.strip()
    explicit_output_file_empty = not explicit_message.strip()
    last_message, last_message_source = _first_message_candidate(
        [
            ("output-last-message", raw_last_message),
            ("explicit-output-file", explicit_message),
            ("stdout-assistant", stdout_message),
            ("session-jsonl-assistant", session_message),
        ]
    )
    stdout_fallback_used = last_message_source == "stdout-assistant"
    session_jsonl_fallback_used = last_message_source == "session-jsonl-assistant"
    current_invocation_failed = getattr(proc, "returncode", 1) != 0
    model_failure = (
        _codex_model_version_failure(stderr_text=stderr_text, stdout_text=stdout_text, codex_bin=codex_bin)
        if current_invocation_failed
        else None
    )
    if getattr(proc, "returncode", 1) == 0 and last_message:
        status = "OK"
    elif model_failure is not None:
        status = model_failure["status"]
    elif _codex_auth_or_sandbox_failed(getattr(proc, "returncode", 1), stderr_text, stdout_text):
        status = "CODEX_AUTH_OR_SANDBOX_FAILURE"
    elif stdout_text.strip() or session_path is not None:
        status = "CODEX_NO_ASSISTANT_MESSAGE"
    elif getattr(proc, "returncode", 1) == 0 and not last_message:
        status = "CODEX_EMPTY_LAST_MESSAGE"
    else:
        status = "CODEX_FAILED"
    return {
        "status": status,
        "attempt": attempt,
        "command": cmd,
        "codex_binary_path": codex_bin,
        "codex_version": codex_version,
        "requested_model": MODEL,
        "supports_requested_model": status == "OK",
        "remediation_hint": (model_failure or {}).get("remediation_hint"),
        "cli_failure_class": (model_failure or {}).get("cli_failure_class"),
        "returncode": getattr(proc, "returncode", None),
        "stderr_tail": stderr_text[-1000:],
        "stdout_tail": stdout_text[-1000:],
        "raw_stdout_excerpt": stdout_text[-2000:],
        "raw_stderr_excerpt": stderr_text[-2000:],
        "output_path": str(paths.codex_output),
        "explicit_output_path": str(paths.codex_explicit_output),
        "session_jsonl_path": str(session_path) if session_path else None,
        "last_message": last_message,
        "last_message_source": last_message_source,
        "last_message_file_empty": last_message_file_empty,
        "explicit_output_file_empty": explicit_output_file_empty,
        "stdout_fallback_used": stdout_fallback_used,
        "session_jsonl_fallback_used": session_jsonl_fallback_used,
    }


def _parse_codex_receipt(text: str, *, empty_error: str = "CODEX_EMPTY_LAST_MESSAGE") -> dict[str, Any]:
    if not text.strip():
        return {"valid": False, "error": empty_error, "raw_output_excerpt": ""}
    payload = _extract_json_object(text)
    if payload is None:
        return {"valid": False, "error": "CODEX_NO_JSON_RECEIPT", "raw_output_excerpt": text[:2000]}
    if "actions" in payload:
        return {"valid": False, "error": "MULTIPLE_ACTIONS_FIELD_FORBIDDEN", "raw_payload": payload}
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    if not isinstance(receipt, dict):
        return {"valid": False, "error": "RECEIPT_NOT_OBJECT", "raw_payload": payload}
    receipt = dict(receipt)
    issues = _receipt_parse_issues(receipt)
    if issues:
        return {"valid": False, "error": "SCHEMA_INVALID", "issues": issues, "receipt": receipt}
    if "invalidation" not in receipt and "invalidation_evidence" in receipt:
        receipt["invalidation"] = receipt["invalidation_evidence"]
    if "invalidation_evidence" not in receipt and "invalidation" in receipt:
        receipt["invalidation_evidence"] = receipt["invalidation"]
    if "ownership" not in receipt and "manual_system_ownership" in receipt:
        receipt["ownership"] = receipt["manual_system_ownership"]
    receipt["action"] = str(receipt["action"]).upper()
    receipt["thesis_state"] = str(receipt["thesis_state"]).upper()
    receipt["ownership"] = str(receipt.get("ownership") or "UNKNOWN").upper()
    if "side" in receipt:
        receipt["side"] = str(receipt.get("side") or "NONE").upper()
    receipt["gateway_required"] = True
    receipt["no_direct_oanda"] = True
    return {"valid": True, "receipt": receipt}


def _parse_codex_result(codex: dict[str, Any]) -> dict[str, Any]:
    status = str(codex.get("status") or "").upper()
    if status in {"CODEX_TIMEOUT", "CODEX_AUTH_OR_SANDBOX_FAILURE", *CODEX_MODEL_FAILURE_STATUSES}:
        return {
            "valid": False,
            "error": status,
            "raw_output_excerpt": str(codex.get("last_message") or codex.get("raw_stdout_excerpt") or "")[:2000],
            "codex_binary_path": codex.get("codex_binary_path"),
            "codex_version": codex.get("codex_version"),
            "requested_model": codex.get("requested_model") or MODEL,
            "remediation_hint": codex.get("remediation_hint"),
        }
    return _parse_codex_receipt(codex.get("last_message") or "", empty_error=_empty_parse_error(codex))


def _empty_parse_error(codex: dict[str, Any]) -> str:
    status = str(codex.get("status") or "").upper()
    if status in {"CODEX_TIMEOUT", "CODEX_AUTH_OR_SANDBOX_FAILURE", "CODEX_NO_ASSISTANT_MESSAGE", *CODEX_MODEL_FAILURE_STATUSES}:
        return status
    return "CODEX_EMPTY_LAST_MESSAGE"


def _first_message_candidate(candidates: list[tuple[str, str]]) -> tuple[str, str | None]:
    for source, value in candidates:
        text = str(value or "").strip()
        if text:
            return text, source
    return "", None


def _extract_stdout_assistant_message(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and ("action" in payload or "receipt" in payload):
            return stripped
    jsonl_message = _extract_assistant_message_from_jsonl(stripped.splitlines())
    if jsonl_message:
        return jsonl_message
    if stripped.startswith("{") and _extract_json_object(stripped) is not None:
        return stripped
    return _extract_transcript_assistant_message(stripped)


def _latest_session_jsonl(codex_home: Path) -> Path | None:
    sessions = codex_home / "sessions"
    try:
        files = [path for path in sessions.rglob("*.jsonl") if path.is_file()]
    except OSError:
        return None
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def _path_mtime(path: Path | None) -> float | None:
    if path is None:
        return None
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _latest_session_assistant_message(
    codex_home: Path,
    *,
    previous: Path | None = None,
    previous_mtime: float | None = None,
) -> tuple[str, Path | None]:
    path = _latest_session_jsonl(codex_home)
    if path is None:
        return "", None
    if previous is not None:
        try:
            if path == previous and previous_mtime is not None and path.stat().st_mtime <= previous_mtime:
                return "", path
        except OSError:
            return "", path
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "", path
    return _extract_assistant_message_from_jsonl(lines), path


def _extract_assistant_message_from_jsonl(lines: list[str]) -> str:
    messages: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = _assistant_text_from_event(payload)
        if text:
            messages.append(text)
    return messages[-1].strip() if messages else ""


def _assistant_text_from_event(payload: Any) -> str:
    if isinstance(payload, list):
        for item in reversed(payload):
            text = _assistant_text_from_event(item)
            if text:
                return text
        return ""
    if not isinstance(payload, dict):
        return ""
    role = str(payload.get("role") or "").lower()
    event_type = str(payload.get("type") or payload.get("record_type") or "").lower()
    if role == "assistant" or event_type in {"agent_message", "assistant_message", "final_answer"}:
        return _content_text(payload.get("content") or payload.get("message") or payload.get("text"))
    for key in ("message", "item", "delta", "event"):
        text = _assistant_text_from_event(payload.get(key))
        if text:
            return text
    return ""


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "output_text"):
            text = _content_text(value.get(key))
            if text:
                return text
    return ""


def _extract_transcript_assistant_message(text: str) -> str:
    marker = re.search(r"(?im)^\[[^\n\]]+\]\s+(?:assistant|codex)\s*$", text)
    if not marker:
        return ""
    return text[marker.end() :].strip()


def _parse_codex_version(text: str) -> str | None:
    match = re.search(r"codex(?:-cli)?\s+([0-9]+(?:\.[0-9]+){1,3})", str(text or ""), flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\b([0-9]+(?:\.[0-9]+){1,3})\b", str(text or ""))
    return match.group(1) if match else None


def _codex_version_tuple(version: Any) -> tuple[int, int, int] | None:
    if not version:
        return None
    parts = str(version).strip().split(".")
    try:
        numbers = [int(part) for part in parts[:3]]
    except ValueError:
        return None
    while len(numbers) < 3:
        numbers.append(0)
    return tuple(numbers[:3])


def _codex_model_version_failure(*, stderr_text: str, stdout_text: str, codex_bin: str) -> dict[str, str] | None:
    haystack = f"{stderr_text}\n{stdout_text}".lower()
    if not haystack.strip():
        return None
    if MODEL.lower() in haystack and any(token in haystack for token in ("newer version", "upgrade", "requires a newer")):
        return {
            "status": "CODEX_MODEL_UNSUPPORTED",
            "cli_failure_class": "CODEX_CLI_VERSION_UNSUPPORTED",
            "remediation_hint": _codex_remediation_hint(status="CODEX_MODEL_UNSUPPORTED", codex_bin=codex_bin),
        }
    if MODEL.lower() in haystack and any(token in haystack for token in ("unsupported model", "unknown model", "invalid model")):
        return {
            "status": "CODEX_MODEL_UNSUPPORTED",
            "cli_failure_class": "CODEX_MODEL_UNSUPPORTED",
            "remediation_hint": _codex_remediation_hint(status="CODEX_MODEL_UNSUPPORTED", codex_bin=codex_bin),
        }
    if "codex" in haystack and "version" in haystack and any(token in haystack for token in ("unsupported", "too old", "upgrade")):
        return {
            "status": "CODEX_CLI_VERSION_UNSUPPORTED",
            "cli_failure_class": "CODEX_CLI_VERSION_UNSUPPORTED",
            "remediation_hint": _codex_remediation_hint(status="CODEX_CLI_VERSION_UNSUPPORTED", codex_bin=codex_bin),
        }
    return None


def _codex_remediation_hint(*, status: str, codex_bin: str) -> str:
    if status == "CODEX_MODEL_UNSUPPORTED":
        return (
            f"{MODEL} is not supported by Codex binary {codex_bin}. Upgrade that CLI or set "
            "QR_GUARDIAN_WAKE_CODEX_BIN to a GPT-5.5-capable Codex binary such as "
            f"{DEFAULT_CODEX_APP_BIN}."
        )
    return (
        f"Codex binary {codex_bin} could not prove a supported CLI version. Upgrade it or set "
        f"QR_GUARDIAN_WAKE_CODEX_BIN={DEFAULT_CODEX_APP_BIN}."
    )


def _codex_auth_or_sandbox_failed(returncode: int | None, stderr_text: str, stdout_text: str) -> bool:
    if returncode == 0:
        return False
    haystack = f"{stderr_text}\n{stdout_text}".lower()
    return any(
        token in haystack
        for token in (
            "auth",
            "login",
            "not authenticated",
            "authentication",
            "sandbox",
            "permission denied",
            "operation not permitted",
            "not trusted",
            "approval",
        )
    )


def _decode_process_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decoded_tail(value: Any, *, limit: int = 1000) -> str:
    return _decode_process_text(value)[-limit:]


def _receipt_parse_issues(receipt: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    action = str(receipt.get("action") or "").upper()
    if action not in GUARDIAN_ACTIONS:
        issues.append({"code": "BAD_ACTION", "message": "action must be exactly one supported guardian action"})
    required = (
        "event_id",
        "new_information",
        "pair",
        "side",
        "thesis_state",
        "reason",
        "harvest_trigger",
        "margin_state",
        "gateway_required",
        "no_direct_oanda",
    )
    for field in required:
        if field not in receipt:
            issues.append({"code": "FIELD_MISSING", "message": f"missing {field}"})
    if "invalidation" not in receipt and "invalidation_evidence" not in receipt:
        issues.append({"code": "FIELD_MISSING", "message": "missing invalidation evidence"})
    if "ownership" not in receipt and "manual_system_ownership" not in receipt:
        issues.append({"code": "FIELD_MISSING", "message": "missing manual/system ownership"})
    if "new_information" in receipt and not isinstance(receipt.get("new_information"), bool):
        issues.append({"code": "BAD_NEW_INFORMATION", "message": "new_information must be boolean"})
    if receipt.get("gateway_required") is not True:
        issues.append({"code": "GATEWAY_REQUIRED", "message": "gateway_required must be true"})
    if receipt.get("no_direct_oanda") is not True:
        issues.append({"code": "NO_DIRECT_OANDA_REQUIRED", "message": "no_direct_oanda must be true"})
    thesis_state = str(receipt.get("thesis_state") or "").upper()
    if thesis_state and thesis_state not in THESIS_STATES:
        issues.append({"code": "BAD_THESIS_STATE", "message": "thesis_state is not supported"})
    ownership = str(receipt.get("ownership") or receipt.get("manual_system_ownership") or "").upper()
    if ownership and ownership not in {"SYSTEM", "OPERATOR_MANUAL", "UNKNOWN"}:
        issues.append({"code": "BAD_OWNERSHIP", "message": "ownership must be SYSTEM, OPERATOR_MANUAL, or UNKNOWN"})
    side = str(receipt.get("side") or "").upper()
    if side and side not in {"LONG", "SHORT", "NONE", "N/A"}:
        issues.append({"code": "BAD_SIDE", "message": "side must be LONG, SHORT, or NONE"})
    return issues


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", stripped):
        try:
            payload, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            found.append(payload)
    return found[-1] if found else None


def _events_from_payload(payload: dict[str, Any]) -> list[GuardianEvent]:
    events = []
    for item in payload.get("events", []) or []:
        if not isinstance(item, dict):
            continue
        fields = {
            "event_id": str(item.get("event_id") or ""),
            "event_type": str(item.get("event_type") or ""),
            "pair": str(item.get("pair") or ""),
            "direction": item.get("direction"),
            "thesis": str(item.get("thesis") or ""),
            "price_zone": str(item.get("price_zone") or ""),
            "severity": str(item.get("severity") or "P2"),
            "recommended_review_type": str(item.get("recommended_review_type") or ""),
            "dedupe_key": str(item.get("dedupe_key") or ""),
            "action_hint": str(item.get("action_hint") or ""),
            "thesis_state": str(item.get("thesis_state") or "UNKNOWN"),
            "detected_at_utc": str(item.get("detected_at_utc") or ""),
            "details": item.get("details") if isinstance(item.get("details"), dict) else {},
        }
        try:
            events.append(GuardianEvent(**fields))
        except TypeError:
            continue
    return events


def _maybe_gateway_handoff(
    *,
    receipt_review: dict[str, Any] | None,
    paths: DispatcherPaths,
    env: dict[str, str],
    lock: dict[str, Any],
) -> dict[str, Any]:
    enabled = env.get("QR_GUARDIAN_WAKE_GATEWAY_HANDOFF", "0") == "1"
    base = {
        "enabled": enabled,
        "default_off_env": "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF=0",
        "required_gateway": "LiveOrderGateway",
        "required_revalidators": ["guardian-action-cycle", "RiskEngine", "LiveOrderGateway"],
    }
    if not enabled:
        return {**base, "status": "SKIPPED_DEFAULT_OFF"}
    if receipt_review is None or receipt_review.get("status") != "ACCEPTED":
        return {**base, "status": "SKIPPED_RECEIPT_NOT_ACCEPTED"}
    receipt = receipt_review.get("receipt") if isinstance(receipt_review.get("receipt"), dict) else {}
    if receipt.get("gateway_required") is not True:
        return {**base, "status": "SKIPPED_GATEWAY_REQUIRED_FALSE"}
    if receipt.get("new_information") is not True:
        return {**base, "status": "SKIPPED_NO_NEW_INFORMATION"}
    reason = str(receipt.get("reason") or "").lower()
    if "scheduled hour" in reason or "hourly" in reason or receipt.get("schedule_only") is True:
        return {**base, "status": "SKIPPED_SCHEDULED_HOUR_ONLY"}
    if lock.get("active"):
        return {**base, "status": "SKIPPED_LOCK_CONFLICT", "lock": lock}
    live_flags = {
        "QR_LIVE_ENABLED": env.get("QR_LIVE_ENABLED", "0"),
        "QR_GUARDIAN_WAKE_GATEWAY_HANDOFF": env.get("QR_GUARDIAN_WAKE_GATEWAY_HANDOFF", "0"),
        "QR_GUARDIAN_ACTION_EXECUTE": env.get("QR_GUARDIAN_ACTION_EXECUTE", "0"),
    }
    if live_flags["QR_GUARDIAN_ACTION_EXECUTE"] != "1":
        return {**base, "status": "SKIPPED_ACTION_EXECUTE_DISABLED", "live_flags": live_flags}
    if any(value != "1" for value in live_flags.values()):
        return {**base, "status": "SKIPPED_LIVE_FLAGS_DISABLED", "live_flags": live_flags}
    python_bin = env.get("QR_PYTHON", "python3")
    cmd = [python_bin, "-m", "quant_rabbit.cli", "guardian-action-cycle"]
    run_env = dict(env)
    run_env["PYTHONPATH"] = str(paths.root / "src")
    try:
        proc = subprocess.run(cmd, cwd=str(paths.root), capture_output=True, text=True, timeout=180, env=run_env)
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": "ACTION_CYCLE_EXCEPTION", "live_flags": live_flags, "command": cmd, "error": str(exc)}
    return {
        **base,
        "status": "ACTION_CYCLE_CALLED" if proc.returncode == 0 else "ACTION_CYCLE_FAILED",
        "live_flags": live_flags,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
    }


def _write_action_review(
    path: Path,
    payload: dict[str, Any],
    *,
    action_receipt_path: Path | None = None,
    now: datetime | None = None,
) -> None:
    clock = _utc(now)
    latest_receipt = _latest_accepted_receipt_payload(action_receipt_path)
    receipt_written = bool(payload.get("receipt_written", False))
    receipt_exists = bool(latest_receipt)
    receipt_reason = _action_receipt_existence_reason(payload, latest_receipt=latest_receipt)
    lines = [
        "# Guardian Action Review",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Dispatcher status: `{payload.get('status')}`",
        f"- Model: `{payload.get('model') or MODEL}`",
        f"- Receipt exists: `{'yes' if receipt_exists else 'no'}`",
        f"- Receipt reason: {receipt_reason}",
        f"- Receipt written: `{receipt_written}`",
        f"- Action receipt path: `{str(action_receipt_path) if receipt_exists and action_receipt_path else payload.get('action_receipt_path') or 'none'}`",
        "",
        "## Latest Accepted Receipt",
        "",
    ]
    if latest_receipt:
        receipt = latest_receipt.get("receipt") if isinstance(latest_receipt.get("receipt"), dict) else latest_receipt
        lines.extend(
            [
                f"- Receipt status: `{latest_receipt.get('receipt_status') or latest_receipt.get('status')}`",
                f"- Lifecycle: `{latest_receipt.get('receipt_lifecycle') or 'ACTIVE'}`",
                f"- Action: `{receipt.get('action') or 'none'}`",
                f"- Event id: `{latest_receipt.get('selected_event_id') or receipt.get('event_id') or 'none'}`",
                f"- Dedupe key: `{latest_receipt.get('selected_event_dedupe_key') or receipt.get('dedupe_key') or 'none'}`",
                f"- Generated at: `{latest_receipt.get('generated_at_utc') or 'none'}`",
                f"- Expires at: `{latest_receipt.get('expires_at_utc') or 'none'}`",
                f"- Consumed by trader: `{latest_receipt.get('consumed_by_trader', False)}`",
            ]
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Selected Event",
            "",
        ]
    )
    event = payload.get("selected_event") if isinstance(payload.get("selected_event"), dict) else {}
    if event:
        lines.extend(
            [
                f"- Event id: `{event.get('event_id')}`",
                f"- Dedupe key: `{event.get('dedupe_key')}`",
                f"- Type: `{event.get('event_type')}`",
                f"- Pair / side: `{event.get('pair')}` / `{event.get('direction') or event.get('side') or 'N/A'}`",
                f"- Severity: `{event.get('severity')}`",
                f"- Wake reasons: `{', '.join(event.get('wake_reason_codes') or []) or 'none'}`",
            ]
        )
    else:
        lines.append("- none")
    preflight = payload.get("codex_preflight") if isinstance(payload.get("codex_preflight"), dict) else {}
    if preflight:
        lines.extend(
            [
                "",
                "## Codex Preflight",
                "",
                f"- Status: `{preflight.get('status')}` enabled=`{preflight.get('enabled')}`",
                f"- Binary: `{preflight.get('codex_binary_path')}`",
                f"- Version: `{preflight.get('codex_version') or 'unknown'}`",
                f"- Requested model: `{preflight.get('requested_model') or MODEL}`",
                f"- Supports requested model: `{preflight.get('supports_requested_model')}`",
            ]
        )
        if preflight.get("remediation_hint"):
            lines.append(f"- Remediation: {preflight.get('remediation_hint')}")
    attempts = payload.get("codex_attempts") if isinstance(payload.get("codex_attempts"), list) else []
    codex_single = payload.get("codex") if isinstance(payload.get("codex"), dict) else {}
    if attempts or codex_single:
        lines.extend(["", "## Codex Diagnostics", ""])
        for index, attempt in enumerate(attempts or [codex_single], start=1):
            if not isinstance(attempt, dict):
                continue
            lines.extend(
                [
                    f"- Attempt {index}: `{attempt.get('attempt') or 'unknown'}` status=`{attempt.get('status')}` returncode=`{attempt.get('returncode')}`",
                    f"  - binary: `{attempt.get('codex_binary_path') or 'unknown'}` version: `{attempt.get('codex_version') or 'unknown'}` requested model: `{attempt.get('requested_model') or MODEL}`",
                    f"  - output path: `{attempt.get('output_path')}`",
                    f"  - explicit output path: `{attempt.get('explicit_output_path')}`",
                    f"  - session JSONL: `{attempt.get('session_jsonl_path') or 'none'}`",
                    f"  - last-message source: `{attempt.get('last_message_source') or 'none'}`",
                    f"  - last-message empty: `{attempt.get('last_message_file_empty')}` explicit empty: `{attempt.get('explicit_output_file_empty')}` stdout fallback: `{attempt.get('stdout_fallback_used')}` session fallback: `{attempt.get('session_jsonl_fallback_used')}`",
                ]
            )
            stdout_excerpt = str(attempt.get("raw_stdout_excerpt") or attempt.get("stdout_tail") or "").strip()
            stderr_excerpt = str(attempt.get("raw_stderr_excerpt") or attempt.get("stderr_tail") or "").strip()
            if stdout_excerpt:
                lines.append(f"  - stdout excerpt: `{stdout_excerpt[:500]}`")
            if stderr_excerpt:
                lines.append(f"  - stderr excerpt: `{stderr_excerpt[:500]}`")
            if attempt.get("remediation_hint"):
                lines.append(f"  - remediation: {attempt.get('remediation_hint')}")
    lines.extend(["", "## Parse", ""])
    parse = payload.get("parse") if isinstance(payload.get("parse"), dict) else {}
    if parse:
        lines.append(f"- Valid: `{parse.get('valid')}`")
        if parse.get("error"):
            lines.append(f"- Error: `{parse.get('error')}`")
        for issue in parse.get("issues", []) or []:
            lines.append(f"- `{issue.get('code')}` {issue.get('message')}")
    else:
        lines.append("- no Codex output parsed")
    review = payload.get("receipt_review") if isinstance(payload.get("receipt_review"), dict) else {}
    if review:
        lines.extend(["", "## Receipt Review", "", f"- Status: `{review.get('status')}`"])
        receipt = review.get("receipt") if isinstance(review.get("receipt"), dict) else {}
        lines.append(f"- Action: `{receipt.get('action') or 'none'}`")
        lines.append(f"- Event id: `{receipt.get('event_id') or 'none'}`")
        lines.append(f"- Pair / side: `{receipt.get('pair') or 'none'}` / `{receipt.get('side') or 'none'}`")
        lines.append(f"- Dedupe key: `{receipt.get('dedupe_key') or 'none'}`")
        lines.append(f"- Receipt id: `{receipt.get('receipt_id') or review.get('generated_at_utc') or 'none'}`")
        lines.append(f"- Dispatcher status: `{payload.get('status')}`")
        for issue in review.get("issues", []) or []:
            lines.append(f"- `{issue.get('severity')}` `{issue.get('code')}` {issue.get('message')}")
    handoff = payload.get("gateway_handoff") if isinstance(payload.get("gateway_handoff"), dict) else {}
    if handoff:
        lines.extend(["", "## Gateway Handoff", "", f"- Status: `{handoff.get('status')}`"])
        lines.append(f"- Required gateway: `{handoff.get('required_gateway')}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Guardian wake dispatcher never calls OANDA.",
            "- Codex runs read-only and writes at most a review/receipt artifact.",
            "- Live orders, cancels, and closes remain gateway-only.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, "\n".join(lines) + "\n")


def _action_receipt_existence_reason(payload: dict[str, Any], *, latest_receipt: dict[str, Any] | None = None) -> str:
    if payload.get("receipt_written"):
        return "accepted schema-valid receipt is bound to selected_event and was written atomically."
    if latest_receipt:
        receipt = latest_receipt.get("receipt") if isinstance(latest_receipt.get("receipt"), dict) else latest_receipt
        action = str(receipt.get("action") or "UNKNOWN").upper()
        event_id = latest_receipt.get("selected_event_id") or receipt.get("event_id") or "unknown"
        return (
            f"latest accepted `{action}` receipt for event `{event_id}` remains "
            f"`{latest_receipt.get('receipt_lifecycle') or 'ACTIVE'}`; dispatcher status "
            f"`{payload.get('status') or 'UNKNOWN'}` does not invalidate it."
        )
    status = str(payload.get("status") or "")
    parse = payload.get("parse") if isinstance(payload.get("parse"), dict) else {}
    review = payload.get("receipt_review") if isinstance(payload.get("receipt_review"), dict) else {}
    if parse.get("error"):
        return f"no receipt because Codex output parse/classification is `{parse.get('error')}`."
    if review.get("status"):
        codes = ", ".join(str(issue.get("code")) for issue in review.get("issues", []) or []) or "no issue codes"
        return f"no receipt because receipt review status is `{review.get('status')}` ({codes})."
    return f"no receipt because dispatcher status is `{status or 'UNKNOWN'}`."


def _receipt_expires_at(now: datetime, *, env: dict[str, str]) -> str:
    ttl = int(env.get("QR_GUARDIAN_ACTION_RECEIPT_TTL_SECONDS", DEFAULT_RECEIPT_TTL_SECONDS))
    return (now + timedelta(seconds=max(0, ttl))).isoformat()


def _receipt_is_accepted(payload: dict[str, Any]) -> bool:
    return str(payload.get("receipt_status") or payload.get("status") or "").upper() == "ACCEPTED"


def _receipt_lifecycle(payload: dict[str, Any]) -> str:
    return str(payload.get("receipt_lifecycle") or ("ACTIVE" if _receipt_is_accepted(payload) else "")).upper()


def _latest_accepted_receipt_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json(path)
    if not payload or not _receipt_is_accepted(payload):
        return None
    return payload


def _expire_current_receipt_if_needed(path: Path, *, now: datetime) -> None:
    payload = _load_json(path)
    if not payload or not _receipt_is_accepted(payload):
        return
    if _receipt_lifecycle(payload) in TERMINAL_RECEIPT_LIFECYCLES:
        return
    expires_at = _parse_utc(payload.get("expires_at_utc"))
    if expires_at is None or expires_at > now:
        return
    expired = {
        **payload,
        "receipt_lifecycle": "EXPIRED",
        "expired_at_utc": now.isoformat(),
        "lifecycle_reason": "guardian action receipt exceeded expires_at_utc",
    }
    _archive_receipt(path, expired)
    _remove_file(path)


def _supersede_current_receipt(path: Path, *, superseded_by_event_id: str, now: datetime) -> dict[str, Any] | None:
    payload = _load_json(path)
    if not payload or not _receipt_is_accepted(payload):
        return None
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    current_event_id = str(payload.get("selected_event_id") or receipt.get("event_id") or "")
    superseded = {
        **payload,
        "receipt_lifecycle": "SUPERSEDED",
        "superseded_by_event_id": superseded_by_event_id or None,
        "superseded_at_utc": now.isoformat(),
        "consumed_by_trader": bool(payload.get("consumed_by_trader", False)),
    }
    _archive_receipt(path, superseded)
    return {
        "event_id": current_event_id or None,
        "action": receipt.get("action"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "receipt_lifecycle": "SUPERSEDED",
        "archive_path": _archive_path(path, superseded).name,
    }


def _remove_non_accepted_current_receipt(path: Path) -> None:
    payload = _load_json(path)
    if payload and _receipt_is_accepted(payload):
        return
    _remove_file(path)


def _archive_receipt(current_path: Path, payload: dict[str, Any]) -> None:
    archive_path = _archive_path(current_path, payload)
    _write_json(archive_path, payload)


def _archive_path(current_path: Path, payload: dict[str, Any]) -> Path:
    receipt = payload.get("receipt") if isinstance(payload.get("receipt"), dict) else payload
    generated = str(payload.get("generated_at_utc") or datetime.now(timezone.utc).isoformat())
    event_id = str(payload.get("selected_event_id") or receipt.get("event_id") or "unknown")
    lifecycle = str(payload.get("receipt_lifecycle") or "ACTIVE")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{generated}_{event_id}_{lifecycle}").strip("_")
    return current_path.parent / "guardian_action_receipts" / f"{safe}.json"


def _maybe_write_tuning_work_order(
    *,
    path: Path,
    selected_event: dict[str, Any],
    receipt: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    reasons = sorted(
        {
            str(item).strip().upper()
            for item in selected_event.get("wake_reason_codes") or []
            if _is_tuning_handoff_reason(str(item))
        }
    )
    if not reasons:
        return {"status": "SKIPPED_NON_TUNING_EVENT"}
    fingerprint = _event_material_fingerprint(selected_event)
    work_order_id = f"guardian-tuning-{fingerprint[:20]}"
    existing = _load_json(path)
    pending_existing = _pending_tuning_work_order_entries(existing)
    matching_pending = next(
        (
            item
            for item in pending_existing
            if str(item.get("event_fingerprint") or "") == fingerprint
        ),
        None,
    )
    if matching_pending is not None:
        return {
            "status": "UNCHANGED_IDEMPOTENT",
            "work_order_id": matching_pending.get("work_order_id") or work_order_id,
            "path": str(path),
            "event_fingerprint": fingerprint,
            "pending_count": len(pending_existing),
        }
    review_validation = _validate_bot_tuning_review(receipt.get("bot_tuning_review"))
    entry: dict[str, Any] = {
        "generated_at_utc": now.isoformat(),
        "work_order_id": work_order_id,
        "status": "PENDING_HOURLY_AI_REVIEW",
        "source": "guardian_wake_dispatcher",
        "source_receipt_id": receipt.get("receipt_id") or now.isoformat(),
        "selected_event_id": selected_event.get("event_id"),
        "selected_event_dedupe_key": selected_event.get("dedupe_key"),
        "event_fingerprint": fingerprint,
        "material_reason_codes": reasons,
        "selected_event": selected_event,
        "bot_tuning_review_validation": {
            key: value for key, value in review_validation.items() if key != "review"
        },
        "live_permission_allowed": False,
        "no_direct_oanda": True,
        "preserve_blockers": True,
        "execution_boundary": {
            "hourly_ai_review_required": True,
            "work_order_never_grants_live_permission": True,
            "live_gateway_and_risk_gates_unchanged": True,
        },
        "lifecycle_contract": {
            "terminal_statuses": ["CONSUMED", "SUPERSEDED"],
            "consume_only_after_falsifiable_review": True,
            "required_terminal_fields": [
                "consumed_at_utc",
                "consumed_by",
                "experiment_id",
                "experiment_result",
            ],
        },
    }
    if review_validation.get("status") == "VALID":
        entry["bot_tuning_review"] = review_validation.get("review")

    existing_same_fingerprint = (
        str(existing.get("event_fingerprint") or "") == fingerprint
        and not _tuning_work_order_entry_pending(existing)
    )
    if existing_same_fingerprint:
        entry["reopened_count"] = int(existing.get("reopened_count") or 0) + 1
        entry["reopened_from_terminal"] = {
            "work_order_id": existing.get("work_order_id"),
            "status": existing.get("status"),
            "consumed_at_utc": existing.get("consumed_at_utc"),
            "experiment_id": existing.get("experiment_id"),
        }

    entries = [entry, *pending_existing]
    deduped_entries: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for item in entries:
        item_fingerprint = str(item.get("event_fingerprint") or "")
        if not item_fingerprint or item_fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(item_fingerprint)
        deduped_entries.append(item)
        if len(deduped_entries) >= 20:
            break
    payload: dict[str, Any] = {
        **entry,
        "schema_version": 2,
        "work_orders": deduped_entries,
        "pending_count": len(deduped_entries),
    }
    try:
        _write_json(path, payload)
    except OSError as exc:
        return {
            "status": "WORK_ORDER_WRITE_FAILED",
            "work_order_id": work_order_id,
            "path": str(path),
            "event_fingerprint": fingerprint,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "WORK_ORDER_WRITTEN",
        "work_order_id": work_order_id,
        "path": str(path),
        "event_fingerprint": fingerprint,
        "bot_tuning_review_status": review_validation.get("status"),
        "pending_count": len(deduped_entries),
    }


def _pending_tuning_work_order_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("_missing"):
        return []
    raw_entries = payload.get("work_orders")
    entries = [payload]
    if isinstance(raw_entries, list):
        entries.extend(item for item in raw_entries if isinstance(item, dict))
    pending: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in entries:
        fingerprint = str(item.get("event_fingerprint") or item.get("work_order_id") or "")
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        if _tuning_work_order_entry_pending(item):
            pending.append(item)
    return pending


def _tuning_work_order_entry_pending(entry: dict[str, Any]) -> bool:
    return bool(
        str(entry.get("status") or "").upper()
        in {"PENDING_HOURLY_AI_REVIEW", "PENDING", "OPEN"}
        and entry.get("consumed_at_utc") in {None, ""}
        and entry.get("live_permission_allowed") is False
        and entry.get("no_direct_oanda") is True
        and entry.get("preserve_blockers") is True
    )


def _is_tuning_handoff_reason(reason: str) -> bool:
    code = str(reason or "").strip().upper()
    return (
        code in _MATERIAL_CHANGE_REASONS
        or "TECHNICAL_STATE_CHANGE" in code
        or "REGIME" in code
        or "VOLATILITY" in code
        or "FAMILY" in code
        or "STRUCTURE" in code
    )


def _validate_bot_tuning_review(value: Any) -> dict[str, Any]:
    if value is None:
        return {"status": "MISSING", "issues": []}
    if not isinstance(value, dict):
        return {"status": "INVALID", "issues": ["bot_tuning_review must be an object"]}
    issues: list[str] = []
    if "live_permission_allowed" in value and value.get("live_permission_allowed") is not False:
        issues.append("bot_tuning_review live_permission_allowed must be false when present")
    if "no_direct_oanda" in value and value.get("no_direct_oanda") is not True:
        issues.append("bot_tuning_review no_direct_oanda must be true when present")
    if "preserve_blockers" in value and value.get("preserve_blockers") is not True:
        issues.append("bot_tuning_review preserve_blockers must be true when present")
    if issues:
        return {"status": "INVALID_UNSAFE_BOUNDARY", "issues": issues}
    return {"status": "VALID", "issues": [], "review": value}


def _record_state(
    path: Path,
    previous_state: dict[str, Any],
    result: dict[str, Any],
    *,
    reviewed_event: dict[str, Any] | None = None,
    attempted_event: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> None:
    environ = env if env is not None else os.environ
    state = dict(previous_state) if isinstance(previous_state, dict) else {}
    if "reviewed_events" in state:
        state["reviewed_events"] = _accepted_review_records(state.get("reviewed_events"))
    state["generated_at_utc"] = result.get("generated_at_utc")
    state["last_status"] = result.get("status")
    result_time = _parse_utc(result.get("generated_at_utc")) or datetime.now(timezone.utc)
    pending_dispatches = _active_pending_dispatch_records(
        state.get("pending_dispatches"),
        now=result_time,
    )
    selection = result.get("selection") if isinstance(result.get("selection"), dict) else {}
    pending_ttl = max(
        60,
        int(
            environ.get(
                "QR_GUARDIAN_PENDING_DISPATCH_TTL_SECONDS",
                DEFAULT_PENDING_DISPATCH_TTL_SECONDS,
            )
        ),
    )
    for event in selection.get("pending_events", []) or []:
        if not isinstance(event, dict):
            continue
        dedupe_key = str(event.get("dedupe_key") or "").strip()
        if not dedupe_key:
            continue
        prior = pending_dispatches.get(dedupe_key) or {}
        fingerprint = _event_material_fingerprint(event)
        same_observation = str(prior.get("material_fingerprint") or "") == fingerprint
        queued_at = (
            prior.get("queued_at_utc")
            if same_observation and prior.get("queued_at_utc")
            else result_time.isoformat()
        )
        pending_dispatches[dedupe_key] = {
            "event": event,
            "event_id": event.get("event_id"),
            "material_fingerprint": fingerprint,
            "queued_at_utc": queued_at,
            "last_seen_at_utc": result_time.isoformat(),
            "expires_at_utc": (result_time + timedelta(seconds=pending_ttl)).isoformat(),
        }
    state["last_result"] = {
        key: result.get(key)
        for key in (
            "status",
            "generated_at_utc",
            "receipt_written",
            "action_receipt_path",
            "broker_snapshot_freshness",
            "runtime_disk",
            "queued_for_active_trader",
            "queue_reason",
        )
    }
    if isinstance(result.get("selected_event"), dict):
        state["last_result"]["selected_event"] = result.get("selected_event")
    if isinstance(result.get("parse"), dict):
        state["last_result"]["parse"] = result.get("parse")
    if isinstance(result.get("codex_preflight"), dict):
        state["last_result"]["codex_preflight"] = result.get("codex_preflight")
    if isinstance(result.get("codex"), dict):
        codex = result.get("codex") or {}
        state["last_result"]["codex"] = {
            key: codex.get(key)
            for key in (
                "status",
                "attempt",
                "codex_binary_path",
                "codex_version",
                "requested_model",
                "returncode",
                "remediation_hint",
            )
        }
    if isinstance(result.get("parse_failure"), dict):
        state["last_result"]["parse_failure"] = result.get("parse_failure")
    accepted_review = reviewed_event is not None and bool(result.get("receipt_written"))
    if accepted_review:
        reviewed = _accepted_review_records(state.get("reviewed_events"))
        dedupe_key = str(reviewed_event.get("dedupe_key") or "")
        if dedupe_key:
            reviewed[dedupe_key] = {
                "event_id": reviewed_event.get("event_id"),
                "price_zone": reviewed_event.get("price_zone"),
                "details": reviewed_event.get("details") if isinstance(reviewed_event.get("details"), dict) else {},
                "material_fingerprint": _event_material_fingerprint(reviewed_event),
                "severity": reviewed_event.get("severity"),
                "last_reviewed_at_utc": result.get("generated_at_utc"),
                "last_status": result.get("status"),
                "receipt_written": True,
            }
            state["reviewed_events"] = reviewed
            pending_dispatches.pop(dedupe_key, None)
            attempts = state.get("dispatch_attempts") if isinstance(state.get("dispatch_attempts"), dict) else {}
            attempts.pop(dedupe_key, None)
            state["dispatch_attempts"] = attempts
            failures = state.get("parse_failures") if isinstance(state.get("parse_failures"), dict) else {}
            failures.pop(dedupe_key, None)
            state["parse_failures"] = failures
    elif attempted_event is not None:
        failure_code = _failed_attempt_code(result)
        dedupe_key = str(attempted_event.get("dedupe_key") or "")
        if dedupe_key and failure_code:
            attempt_record = _next_failed_attempt_record(
                prior=(state.get("dispatch_attempts") or {}).get(dedupe_key)
                if isinstance(state.get("dispatch_attempts"), dict)
                else None,
                event=attempted_event,
                result=result,
                failure_code=failure_code,
                env=environ,
            )
            attempts = state.get("dispatch_attempts") if isinstance(state.get("dispatch_attempts"), dict) else {}
            attempts[dedupe_key] = attempt_record
            state["dispatch_attempts"] = attempts
            state["last_result"]["dispatch_attempt"] = attempt_record
            if _result_has_parse_failure(result):
                parse_failure = {
                    **{
                        key: attempted_event.get(key)
                        for key in (
                            "event_id",
                            "dedupe_key",
                            "pair",
                            "direction",
                            "event_type",
                            "thesis",
                            "price_zone",
                            "severity",
                        )
                    },
                    "material_fingerprint": attempt_record["material_fingerprint"],
                    "last_failed_at_utc": attempt_record["last_failed_at_utc"],
                    "last_error": failure_code,
                    "consecutive_failures": attempt_record["attempt_count"],
                    "retry_after_utc": attempt_record["retry_after_utc"],
                    "expires_at_utc": attempt_record["expires_at_utc"],
                    "queued_for_active_trader_after_failure": attempt_record["retry_budget_exhausted"],
                }
                failures = state.get("parse_failures") if isinstance(state.get("parse_failures"), dict) else {}
                failures[dedupe_key] = parse_failure
                state["parse_failures"] = failures
                state["last_result"]["parse_failure"] = parse_failure
    ordered_pending = sorted(
        pending_dispatches.items(),
        key=lambda item: (
            _dispatch_priority(
                item[1].get("event") if isinstance(item[1].get("event"), dict) else {}
            ),
            str(item[1].get("queued_at_utc") or ""),
        ),
    )[:MAX_PENDING_DISPATCHES]
    state["pending_dispatches"] = dict(ordered_pending)
    state["last_result"]["pending_dispatch_count"] = len(ordered_pending)
    _write_json(path, state)


def _failed_attempt_code(result: dict[str, Any]) -> str | None:
    status = str(result.get("status") or "").upper()
    if bool(result.get("receipt_written")) and status != "TUNING_HANDOFF_FAILED":
        return None
    parse = result.get("parse") if isinstance(result.get("parse"), dict) else {}
    parse_error = str(parse.get("error") or "").upper()
    if status in {
        "PARSE_FAILED",
        "RECEIPT_EVENT_MISMATCH",
        "RECEIPT_REJECTED",
        "RUNTIME_DISK_P0",
        "TUNING_HANDOFF_FAILED",
        *CODEX_MODEL_FAILURE_STATUSES,
    }:
        return parse_error or status
    # A live lock that appeared after the first malformed response interrupted
    # only the repair retry; the actual GPT attempt still needs backoff state.
    if status == "QUEUED_FOR_ACTIVE_TRADER" and parse.get("valid") is False and result.get("codex"):
        return parse_error or "PARSE_FAILED_BEFORE_QUEUE"
    return None


def _result_has_parse_failure(result: dict[str, Any]) -> bool:
    parse = result.get("parse") if isinstance(result.get("parse"), dict) else {}
    status = str(result.get("status") or "").upper()
    return parse.get("valid") is False and status not in CODEX_MODEL_FAILURE_STATUSES


def _next_failed_attempt_record(
    *,
    prior: Any,
    event: dict[str, Any],
    result: dict[str, Any],
    failure_code: str,
    env: dict[str, str],
) -> dict[str, Any]:
    failed_at = _parse_utc(result.get("generated_at_utc")) or datetime.now(timezone.utc)
    fingerprint = _event_material_fingerprint(event)
    binary_identity = _codex_binary_identity(env)
    same_series = (
        isinstance(prior, dict)
        and str(prior.get("event_id") or "") == str(event.get("event_id") or "")
        and str(prior.get("material_fingerprint") or "") == fingerprint
        and _binary_identity_key(prior.get("codex_binary_identity") or {}) == _binary_identity_key(binary_identity)
        and (_parse_utc(prior.get("expires_at_utc")) or failed_at) > failed_at
    )
    attempt_count = int(prior.get("attempt_count") or 0) + 1 if same_series else 1
    total_failures = int(prior.get("total_failures") or 0) + 1 if isinstance(prior, dict) else 1
    first_failed = _parse_utc(prior.get("first_failed_at_utc")) if same_series else failed_at
    first_failed = first_failed or failed_at
    base = max(1, int(env.get("QR_GUARDIAN_WAKE_RETRY_BASE_SECONDS", DEFAULT_RETRY_BASE_SECONDS)))
    max_backoff = max(base, int(env.get("QR_GUARDIAN_WAKE_RETRY_MAX_SECONDS", DEFAULT_RETRY_MAX_SECONDS)))
    max_attempts = max(1, int(env.get("QR_GUARDIAN_WAKE_RETRY_MAX_ATTEMPTS", DEFAULT_RETRY_MAX_ATTEMPTS)))
    ttl = max(base, int(env.get("QR_GUARDIAN_WAKE_RETRY_TTL_SECONDS", DEFAULT_RETRY_TTL_SECONDS)))
    expires_at = first_failed + timedelta(seconds=ttl)
    backoff = min(max_backoff, base * (2 ** max(0, attempt_count - 1)))
    retry_budget_exhausted = attempt_count >= max_attempts
    retry_after = expires_at if retry_budget_exhausted else min(failed_at + timedelta(seconds=backoff), expires_at)
    return {
        "event_id": event.get("event_id"),
        "dedupe_key": event.get("dedupe_key"),
        "pair": event.get("pair"),
        "event_type": event.get("event_type"),
        "severity": event.get("severity"),
        "event": event,
        "material_fingerprint": fingerprint,
        "codex_binary_identity": binary_identity,
        "first_failed_at_utc": first_failed.isoformat(),
        "last_failed_at_utc": failed_at.isoformat(),
        "retry_after_utc": retry_after.isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "attempt_count": attempt_count,
        "max_attempts": max_attempts,
        "total_failures": total_failures,
        "retry_budget_exhausted": retry_budget_exhausted,
        "last_status": result.get("status"),
        "last_error": failure_code,
        "runtime_disk": result.get("runtime_disk"),
    }


def _paths_payload(paths: DispatcherPaths) -> dict[str, str]:
    return {
        "guardian_escalation": str(paths.escalation),
        "guardian_events": str(paths.events),
        "guardian_event_state": str(paths.event_state),
        "broker_snapshot": str(paths.broker_snapshot),
        "daily_target_state": str(paths.daily_target_state),
        "guardian_event_report": str(paths.event_report),
        "guardian_action_receipt": str(paths.action_receipt),
        "guardian_action_review": str(paths.action_review),
        "guardian_tuning_work_order": str(paths.tuning_work_order),
        "dispatcher_state": str(paths.dispatcher_state),
        "codex_home": str(paths.codex_home),
        "codex_explicit_output": str(paths.codex_explicit_output),
        "live_root": str(paths.live_root),
    }


def _execution_boundary() -> dict[str, bool]:
    return {
        "guardian_never_calls_oanda": True,
        "guardian_never_stages_orders": True,
        "guardian_never_cancels_orders": True,
        "guardian_never_closes_positions": True,
        "guardian_never_edits_broker_state": True,
        "codex_sandbox_read_only": True,
        "live_order_gateway_required_for_execution": True,
    }


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "generated_at_utc": payload.get("generated_at_utc"),
            "status": payload.get("status"),
            "selected_event_id": (payload.get("selected_event") or {}).get("event_id")
            if isinstance(payload.get("selected_event"), dict)
            else None,
            "receipt_written": payload.get("receipt_written", False),
            "runtime_disk_status": (payload.get("runtime_disk") or {}).get("status")
            if isinstance(payload.get("runtime_disk"), dict)
            else None,
            "runtime_disk_free_bytes": (payload.get("runtime_disk") or {}).get("free_bytes")
            if isinstance(payload.get("runtime_disk"), dict)
            else None,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with path.open("a") as handle:
        handle.write(line + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _json_dumps(payload: dict[str, Any], *, limit: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... TRUNCATED ..."


def _parse_utc(raw: Any) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _severity_rank(value: Any) -> int:
    return SEVERITY_RANK.get(str(value or "P2").upper(), SEVERITY_RANK["P2"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Dispatch one Guardian GPT-5.5 wake through Codex CLI.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--live-root", type=Path, default=Path(os.environ.get("QR_GUARDIAN_WAKE_LIVE_ROOT", str(DEFAULT_LIVE_ROOT))))
    parser.add_argument("--escalation", type=Path, default=None)
    parser.add_argument("--events", type=Path, default=None)
    parser.add_argument("--event-state", type=Path, default=None)
    parser.add_argument("--broker-snapshot", type=Path, default=None)
    parser.add_argument("--daily-target-state", type=Path, default=None)
    parser.add_argument("--event-report", type=Path, default=None)
    parser.add_argument("--prompt-template", type=Path, default=None)
    parser.add_argument("--action-receipt", type=Path, default=None)
    parser.add_argument("--action-review", type=Path, default=None)
    parser.add_argument("--tuning-work-order", type=Path, default=None)
    parser.add_argument("--dispatcher-state", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--codex-home", type=Path, default=None)
    parser.add_argument("--codex-output", type=Path, default=None)
    parser.add_argument("--codex-explicit-output", type=Path, default=None)
    parser.add_argument("--live-lock", type=Path, default=None)
    return parser.parse_args(argv)


def paths_from_args(args: argparse.Namespace) -> DispatcherPaths:
    paths = DispatcherPaths.from_root(args.root, live_root=args.live_root)
    return DispatcherPaths(
        root=paths.root,
        escalation=args.escalation or paths.escalation,
        events=args.events or paths.events,
        event_state=args.event_state or paths.event_state,
        broker_snapshot=args.broker_snapshot or paths.broker_snapshot,
        daily_target_state=args.daily_target_state or paths.daily_target_state,
        event_report=args.event_report or paths.event_report,
        prompt_template=args.prompt_template or paths.prompt_template,
        action_receipt=args.action_receipt or paths.action_receipt,
        action_review=args.action_review or paths.action_review,
        tuning_work_order=args.tuning_work_order or paths.tuning_work_order,
        dispatcher_state=args.dispatcher_state or paths.dispatcher_state,
        log=args.log or paths.log,
        codex_home=args.codex_home or paths.codex_home,
        codex_output=args.codex_output or paths.codex_output,
        codex_explicit_output=args.codex_explicit_output or paths.codex_explicit_output,
        live_root=paths.live_root,
        live_lock=args.live_lock or paths.live_lock,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_dispatcher(paths=paths_from_args(args))
    print(json.dumps({k: v for k, v in result.items() if k != "codex"}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"CODEX_EXCEPTION"} else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
