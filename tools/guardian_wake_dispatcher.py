#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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

# Guardian wake runs every 30s from a router that just refreshed broker truth;
# older snapshots are likely from a previous operating window and should wait
# for the normal trader refresh instead of prompting GPT from stale quotes.
BROKER_SNAPSHOT_MAX_AGE_SECONDS = 5 * 60

# Codex wake should finish well inside one event-driven review window. A hung
# local CLI must not hold repeated launchd invocations open indefinitely.
CODEX_TIMEOUT_SECONDS = 8 * 60


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
    dispatcher_state: Path
    log: Path
    codex_home: Path
    codex_output: Path
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
            dispatcher_state=root / "data" / "guardian_wake_dispatcher_state.json",
            log=root / "logs" / "guardian_wake_dispatcher.log",
            codex_home=root / "data" / "codex_guardian_home",
            codex_output=root / "data" / "guardian_wake_codex_last_message.md",
            live_root=live,
            live_lock=live / ".quant_rabbit_live.lock",
        )


def run_dispatcher(
    *,
    paths: DispatcherPaths,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
    subprocess_run: Callable[..., Any] = subprocess.run,
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

    result_base = {
        "generated_at_utc": clock.isoformat(),
        "model": MODEL,
        "paths": _paths_payload(paths),
        "execution_boundary": _execution_boundary(),
    }

    if escalation.get("wake_gpt") is not True:
        result = {
            **result_base,
            "status": "NO_WAKE",
            "wake_gpt": False,
            "reason": "guardian_escalation wake_gpt is not true",
        }
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
        }
        _write_action_review(paths.action_review, result)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
        _append_log(paths.log, result)
        return result

    parse_queue = _same_event_parse_failure_queue(dispatcher_state, selected)
    if parse_queue:
        queued = {
            **result_base,
            "status": "QUEUED_FOR_ACTIVE_TRADER",
            "wake_gpt": True,
            "selected_event": selected,
            "selection": selection,
            "queue_reason": "repeated guardian wake parse failure",
            "parse_failure": parse_queue,
        }
        _mark_escalation_queued(paths.escalation, escalation, queued, now=clock, reason="repeated guardian wake parse failure")
        _write_action_review(paths.action_review, queued)
        _record_state(paths.dispatcher_state, dispatcher_state, queued)
        _append_log(paths.log, queued)
        return queued

    lock = _active_live_lock(paths.live_lock)
    if lock["active"]:
        queued = {
            **result_base,
            "status": "QUEUED_FOR_ACTIVE_TRADER",
            "wake_gpt": True,
            "selected_event": selected,
            "lock": lock,
        }
        _mark_escalation_queued(paths.escalation, escalation, queued, now=clock, reason="active qr-trader/live gateway lock")
        _write_action_review(paths.action_review, queued)
        _record_state(paths.dispatcher_state, dispatcher_state, queued)
        _append_log(paths.log, queued)
        return queued

    freshness = _broker_snapshot_freshness(broker_snapshot, now=clock, env=environ)
    if not freshness["fresh"]:
        result = {
            **result_base,
            "status": "BROKER_SNAPSHOT_STALE",
            "wake_gpt": True,
            "selected_event": selected,
            "broker_snapshot_freshness": freshness,
        }
        _write_action_review(paths.action_review, result)
        _record_state(paths.dispatcher_state, dispatcher_state, result)
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
    )
    codex = _run_codex(
        paths=paths,
        prompt=prompt,
        env=environ,
        subprocess_run=subprocess_run,
        attempt="initial",
    )

    parsed = _parse_codex_receipt(codex.get("last_message") or "")
    codex_attempts = [codex]
    repair_attempted = False
    repair_skipped: dict[str, Any] | None = None
    if not parsed["valid"]:
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
            _remove_file(paths.action_receipt)
            _mark_escalation_queued(
                paths.escalation,
                escalation,
                queued,
                now=clock,
                reason="active qr-trader/live gateway lock before parse repair retry",
            )
            _write_action_review(paths.action_review, queued)
            _record_state(paths.dispatcher_state, dispatcher_state, queued)
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
        )
        codex_attempts.append(repair_codex)
        parsed = _parse_codex_receipt(repair_codex.get("last_message") or "")
        codex = repair_codex
    events = _events_from_payload(events_payload)
    review = None
    receipt_written = False
    if parsed["valid"]:
        if not parsed["receipt"].get("dedupe_key") and selected.get("dedupe_key"):
            parsed["receipt"]["dedupe_key"] = selected["dedupe_key"]
        review = review_guardian_action_receipt(parsed["receipt"], events=events, now=clock)
        receipt_action = str(parsed["receipt"].get("action") or "").upper()
        if review.get("status") == "ACCEPTED" and receipt_action in GUARDIAN_ACTIONS:
            payload = {
                **review,
                "source": "guardian_wake_dispatcher",
                "model": MODEL,
                "no_direct_oanda": True,
                "selected_event": selected,
            }
            _write_json(paths.action_receipt, payload)
            receipt_written = True
        else:
            _remove_file(paths.action_receipt)
    else:
        _remove_file(paths.action_receipt)

    handoff = _maybe_gateway_handoff(
        receipt_review=review,
        paths=paths,
        env=environ,
        lock=lock,
    )
    status = "RECEIPT_WRITTEN" if receipt_written else "PARSE_FAILED" if not parsed["valid"] else "RECEIPT_REJECTED"
    if repair_attempted and not parsed["valid"]:
        repair_skipped = {"status": "FAILED_AFTER_ONE_RETRY", "max_retries_per_event": 1}
    result = {
        **result_base,
        "status": status,
        "wake_gpt": True,
        "selected_event": selected,
        "selection": selection,
        "broker_snapshot_freshness": freshness,
        "codex": codex,
        "codex_attempts": codex_attempts,
        "parse": parsed,
        "repair_attempted": repair_attempted,
        "repair_result": repair_skipped,
        "receipt_written": receipt_written,
        "action_receipt_path": str(paths.action_receipt) if receipt_written else None,
        "gateway_handoff": handoff,
    }
    if review is not None:
        result["receipt_review"] = review
    _write_action_review(paths.action_review, result)
    _record_state(paths.dispatcher_state, dispatcher_state, result, reviewed_event=selected)
    _append_log(paths.log, result)
    return result


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

    reviewed = dispatcher_state.get("reviewed_events") if isinstance(dispatcher_state.get("reviewed_events"), dict) else {}
    candidates: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    current_events = {
        str(item.get("dedupe_key") or ""): item
        for item in events_payload.get("events", []) or []
        if isinstance(item, dict)
    }
    for raw_event in review_events:
        if not isinstance(raw_event, dict):
            continue
        event = dict(current_events.get(str(raw_event.get("dedupe_key") or "")) or raw_event)
        event["wake_reason_codes"] = list(raw_event.get("wake_reason_codes") or event.get("wake_reason_codes") or [])
        reasons = {str(item) for item in event.get("wake_reason_codes") or []}
        dedupe_key = str(event.get("dedupe_key") or "").strip()
        if not dedupe_key:
            suppressed.append({"event": event, "reason": "MISSING_DEDUPE_KEY"})
            continue
        if not ({"NEW_EVENT", "SEVERITY_INCREASE"} & reasons):
            suppressed.append({"event": event, "reason": "NO_NEW_EVENT_OR_SEVERITY_INCREASE"})
            continue
        record = reviewed.get(dedupe_key) if isinstance(reviewed, dict) else None
        throttle = _dispatch_throttle_seconds(event, env)
        if isinstance(record, dict):
            if str(record.get("last_status") or "").upper() == "PARSE_FAILED":
                candidates.append(event)
                continue
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
            if current_severity <= previous_severity:
                suppressed.append({"event": event, "reason": "DEDUPE_KEY_ALREADY_REVIEWED"})
                continue
        candidates.append(event)

    if not candidates:
        return None, {"status": "NO_DISPATCHABLE_EVENT", "suppressed": suppressed}
    candidates.sort(key=lambda item: (-_severity_rank(item.get("severity")), str(item.get("pair") or ""), str(item.get("event_type") or "")))
    return candidates[0], {"status": "SELECTED", "suppressed": suppressed, "candidate_count": len(candidates)}


def _dispatch_throttle_seconds(event: dict[str, Any], env: dict[str, str]) -> int:
    if str(event.get("action_hint") or "").upper() in {"TRADE", "ADD"} or str(
        event.get("recommended_review_type") or ""
    ).upper() in {"ENTRY_REVIEW", "ADD_REVIEW"}:
        return int(env.get("QR_GUARDIAN_FRESH_ACTION_THROTTLE_SECONDS", DEFAULT_FRESH_ACTION_THROTTLE_SECONDS))
    return int(env.get("QR_GUARDIAN_EVENT_THROTTLE_SECONDS", DEFAULT_THROTTLE_SECONDS))


def _event_bypasses_dispatch_throttle(
    event: dict[str, Any],
    reasons: set[str],
    *,
    previous_severity: int,
) -> bool:
    return (
        str(event.get("severity") or "").upper() == "P0"
        and _severity_rank(event.get("severity")) > previous_severity
    ) or "SEVERITY_INCREASE" in reasons


def _same_event_parse_failure_queue(dispatcher_state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
    failures = dispatcher_state.get("parse_failures")
    if not isinstance(failures, dict):
        return None
    dedupe_key = str(event.get("dedupe_key") or "").strip()
    if not dedupe_key:
        return None
    record = failures.get(dedupe_key)
    if not isinstance(record, dict):
        return None
    if str(record.get("event_id") or "") != str(event.get("event_id") or ""):
        return None
    if int(record.get("consecutive_failures") or 0) < 1:
        return None
    return dict(record)


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
    }
    return (
        template.rstrip()
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
) -> dict[str, Any]:
    paths.codex_output.parent.mkdir(parents=True, exist_ok=True)
    paths.codex_output.write_text("")
    codex_bin = env.get("QR_GUARDIAN_WAKE_CODEX_BIN") or shutil.which("codex") or "codex"
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
        "--output-last-message",
        str(paths.codex_output),
    ]
    run_env = dict(env)
    run_env["CODEX_HOME"] = str(paths.codex_home)
    run_env["CODEX_DISABLE_UPDATE_CHECK"] = "1"
    try:
        proc = subprocess_run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=run_env)
    except Exception as exc:
        return {
            "status": "CODEX_EXCEPTION",
            "attempt": attempt,
            "command": cmd,
            "returncode": None,
            "stderr_tail": str(exc),
            "stdout_tail": "",
            "raw_stdout_excerpt": "",
            "raw_stderr_excerpt": str(exc)[-2000:],
            "output_path": str(paths.codex_output),
            "last_message": "",
            "last_message_file_empty": True,
            "stdout_fallback_used": False,
        }
    raw_last_message = _read_text(paths.codex_output)
    stdout_text = str(getattr(proc, "stdout", "") or "")
    stderr_text = str(getattr(proc, "stderr", "") or "")
    last_message_file_empty = not raw_last_message.strip()
    stdout_fallback_used = last_message_file_empty and bool(stdout_text.strip())
    last_message = raw_last_message.strip() or stdout_text.strip()
    if getattr(proc, "returncode", 1) == 0 and last_message:
        status = "OK"
    elif getattr(proc, "returncode", 1) == 0 and not last_message:
        status = "CODEX_EMPTY_LAST_MESSAGE"
    else:
        status = "CODEX_FAILED"
    return {
        "status": status,
        "attempt": attempt,
        "command": cmd,
        "returncode": getattr(proc, "returncode", None),
        "stderr_tail": stderr_text[-1000:],
        "stdout_tail": stdout_text[-1000:],
        "raw_stdout_excerpt": stdout_text[-2000:],
        "raw_stderr_excerpt": stderr_text[-2000:],
        "output_path": str(paths.codex_output),
        "last_message": last_message,
        "last_message_file_empty": last_message_file_empty,
        "stdout_fallback_used": stdout_fallback_used,
    }


def _parse_codex_receipt(text: str) -> dict[str, Any]:
    if not text.strip():
        return {"valid": False, "error": "CODEX_EMPTY_LAST_MESSAGE", "raw_output_excerpt": ""}
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
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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


def _write_action_review(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Guardian Action Review",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Dispatcher status: `{payload.get('status')}`",
        f"- Model: `{payload.get('model') or MODEL}`",
        f"- Receipt written: `{payload.get('receipt_written', False)}`",
        f"- Action receipt path: `{payload.get('action_receipt_path') or 'none'}`",
        "",
        "## Selected Event",
        "",
    ]
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
                    f"  - output path: `{attempt.get('output_path')}`",
                    f"  - last-message empty: `{attempt.get('last_message_file_empty')}` stdout fallback: `{attempt.get('stdout_fallback_used')}`",
                ]
            )
            stdout_excerpt = str(attempt.get("raw_stdout_excerpt") or attempt.get("stdout_tail") or "").strip()
            stderr_excerpt = str(attempt.get("raw_stderr_excerpt") or attempt.get("stderr_tail") or "").strip()
            if stdout_excerpt:
                lines.append(f"  - stdout excerpt: `{stdout_excerpt[:500]}`")
            if stderr_excerpt:
                lines.append(f"  - stderr excerpt: `{stderr_excerpt[:500]}`")
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
    path.write_text("\n".join(lines) + "\n")


def _record_state(
    path: Path,
    previous_state: dict[str, Any],
    result: dict[str, Any],
    *,
    reviewed_event: dict[str, Any] | None = None,
) -> None:
    state = dict(previous_state) if isinstance(previous_state, dict) else {}
    state["generated_at_utc"] = result.get("generated_at_utc")
    state["last_status"] = result.get("status")
    state["last_result"] = {
        key: result.get(key)
        for key in ("status", "generated_at_utc", "receipt_written", "action_receipt_path", "broker_snapshot_freshness")
    }
    if isinstance(result.get("selected_event"), dict):
        state["last_result"]["selected_event"] = result.get("selected_event")
    if isinstance(result.get("parse"), dict):
        state["last_result"]["parse"] = result.get("parse")
    if isinstance(result.get("parse_failure"), dict):
        state["last_result"]["parse_failure"] = result.get("parse_failure")
    if reviewed_event is not None:
        reviewed = state.get("reviewed_events") if isinstance(state.get("reviewed_events"), dict) else {}
        dedupe_key = str(reviewed_event.get("dedupe_key") or "")
        if dedupe_key:
            reviewed[dedupe_key] = {
                "event_id": reviewed_event.get("event_id"),
                "severity": reviewed_event.get("severity"),
                "last_reviewed_at_utc": result.get("generated_at_utc"),
                "last_status": result.get("status"),
                "receipt_written": bool(result.get("receipt_written")),
            }
            state["reviewed_events"] = reviewed
            failures = state.get("parse_failures") if isinstance(state.get("parse_failures"), dict) else {}
            if result.get("status") == "PARSE_FAILED":
                prior = failures.get(dedupe_key) if isinstance(failures.get(dedupe_key), dict) else {}
                same_event = str(prior.get("event_id") or "") == str(reviewed_event.get("event_id") or "")
                consecutive = int(prior.get("consecutive_failures") or 0) + 1 if same_event else 1
                parse = result.get("parse") if isinstance(result.get("parse"), dict) else {}
                failure_record = {
                    "event_id": reviewed_event.get("event_id"),
                    "dedupe_key": dedupe_key,
                    "pair": reviewed_event.get("pair"),
                    "direction": reviewed_event.get("direction"),
                    "event_type": reviewed_event.get("event_type"),
                    "thesis": reviewed_event.get("thesis"),
                    "price_zone": reviewed_event.get("price_zone"),
                    "severity": reviewed_event.get("severity"),
                    "last_failed_at_utc": result.get("generated_at_utc"),
                    "last_error": parse.get("error"),
                    "consecutive_failures": consecutive,
                    "queued_for_active_trader_after_failure": consecutive >= 1,
                }
                failures[dedupe_key] = failure_record
                state["last_result"]["parse_failure"] = failure_record
                state["parse_failures"] = failures
            elif dedupe_key in failures and result.get("receipt_written"):
                failures.pop(dedupe_key, None)
                state["parse_failures"] = failures
    _write_json(path, state)


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
        "dispatcher_state": str(paths.dispatcher_state),
        "codex_home": str(paths.codex_home),
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


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
    parser.add_argument("--dispatcher-state", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--codex-home", type=Path, default=None)
    parser.add_argument("--codex-output", type=Path, default=None)
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
        dispatcher_state=args.dispatcher_state or paths.dispatcher_state,
        log=args.log or paths.log,
        codex_home=args.codex_home or paths.codex_home,
        codex_output=args.codex_output or paths.codex_output,
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
