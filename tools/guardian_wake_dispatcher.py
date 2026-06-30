#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_CODEX_APP_BIN = Path("/Applications/Codex.app/Contents/Resources/codex")

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
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_file(paths.action_receipt)
        _write_action_review(paths.action_review, result)
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
        _remove_file(paths.action_receipt)
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
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_file(paths.action_receipt)
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
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_file(paths.action_receipt)
        _mark_escalation_queued(paths.escalation, escalation, queued, now=clock, reason="active qr-trader/live gateway lock")
        _write_action_review(paths.action_review, queued)
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
            "broker_snapshot_freshness": freshness,
            "queued_for_active_trader": True,
            "queue_reason": "stale broker snapshot; GPT wake not started because stale broker truth must not enter the prompt",
            "receipt_written": False,
            "action_receipt_path": None,
        }
        _remove_file(paths.action_receipt)
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="stale broker snapshot; GPT wake not started",
        )
        _write_action_review(paths.action_review, result)
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
        _remove_file(paths.action_receipt)
        _mark_escalation_queued(
            paths.escalation,
            escalation,
            result,
            now=clock,
            reason="Codex CLI/model compatibility failed before guardian wake",
        )
        _write_action_review(paths.action_review, result)
        _record_state(paths.dispatcher_state, dispatcher_state, result, reviewed_event=selected)
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
            codex_preflight=codex_preflight,
        )
        codex_attempts.append(repair_codex)
        parsed = _parse_codex_result(repair_codex)
        codex = repair_codex
    events = _events_from_payload(events_payload)
    review = None
    receipt_written = False
    if parsed["valid"]:
        if not parsed["receipt"].get("dedupe_key") and selected.get("dedupe_key"):
            parsed["receipt"]["dedupe_key"] = selected["dedupe_key"]
        review = review_guardian_action_receipt(parsed["receipt"], events=events, selected_event=selected, now=clock)
        receipt_action = str(parsed["receipt"].get("action") or "").upper()
        if review.get("status") == "ACCEPTED" and receipt_action in GUARDIAN_ACTIONS:
            payload = {
                **review,
                "source": "guardian_wake_dispatcher",
                "model": MODEL,
                "no_direct_oanda": True,
                "selected_event": selected,
                "selected_event_id": selected.get("event_id"),
                "selected_event_dedupe_key": selected.get("dedupe_key"),
                "dispatcher_status": "RECEIPT_WRITTEN",
                "receipt_id": parsed["receipt"].get("receipt_id") or review.get("generated_at_utc"),
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
    status = (
        "RECEIPT_WRITTEN"
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
    found = shutil.which("codex")
    if found:
        return found
    if DEFAULT_CODEX_APP_BIN.exists():
        return str(DEFAULT_CODEX_APP_BIN)
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


def _write_action_review(path: Path, payload: dict[str, Any]) -> None:
    receipt_written = bool(payload.get("receipt_written", False))
    receipt_reason = _action_receipt_existence_reason(payload)
    lines = [
        "# Guardian Action Review",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Dispatcher status: `{payload.get('status')}`",
        f"- Model: `{payload.get('model') or MODEL}`",
        f"- Receipt exists: `{'yes' if receipt_written else 'no'}`",
        f"- Receipt reason: {receipt_reason}",
        f"- Receipt written: `{receipt_written}`",
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


def _action_receipt_existence_reason(payload: dict[str, Any]) -> str:
    if payload.get("receipt_written"):
        return "accepted schema-valid receipt is bound to selected_event and was written atomically."
    status = str(payload.get("status") or "")
    parse = payload.get("parse") if isinstance(payload.get("parse"), dict) else {}
    review = payload.get("receipt_review") if isinstance(payload.get("receipt_review"), dict) else {}
    if parse.get("error"):
        return f"no receipt because Codex output parse/classification is `{parse.get('error')}`."
    if review.get("status"):
        codes = ", ".join(str(issue.get("code")) for issue in review.get("issues", []) or []) or "no issue codes"
        return f"no receipt because receipt review status is `{review.get('status')}` ({codes})."
    return f"no receipt because dispatcher status is `{status or 'UNKNOWN'}`."


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
        for key in (
            "status",
            "generated_at_utc",
            "receipt_written",
            "action_receipt_path",
            "broker_snapshot_freshness",
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
