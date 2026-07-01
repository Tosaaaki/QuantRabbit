from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


EXPECTED_MODEL = "gpt-5.5"
EXPECTED_REASONING_EFFORT = "high"
EXPECTED_CWD = "/Users/tossaki/App/QuantRabbit-live"
EXPECTED_CADENCE_MINUTES = 60
DEFAULT_GRACE_MINUTES = 15
DEFAULT_LIVE_ROOT = Path(EXPECTED_CWD)
DEFAULT_AUTOMATION_DIR = Path.home() / ".codex" / "automations" / "qr-trader"
DEFAULT_CODEX_LOGS = Path.home() / ".codex" / "logs_2.sqlite"
HIGH_URGENCY_GUARDIAN_ACTIONS = {"REDUCE", "HARVEST", "CANCEL_PENDING"}
LOW_URGENCY_GUARDIAN_ACTIONS = {"HOLD", "NO_ACTION"}
TERMINAL_RECEIPT_LIFECYCLES = {"CONSUMED", "SUPERSEDED", "EXPIRED", "REJECTED"}
RECEIPT_LIFECYCLE_PRECEDENCE = {
    "EXPIRED": 50,
    "CONSUMED": 40,
    "SUPERSEDED": 30,
    "REJECTED": 20,
    "ACTIVE": 10,
}


@dataclass(frozen=True)
class WatchdogPaths:
    root: Path
    automation_toml: Path
    automation_memory: Path
    trader_journal: Path
    autotrade_report: Path
    gpt_decision_report: Path
    decision_response: Path
    guardian_receipt: Path
    guardian_review: Path
    codex_logs: Path
    output_json: Path
    output_report: Path
    output_log: Path

    @classmethod
    def from_root(
        cls,
        root: Path,
        *,
        automation_dir: Path | None = None,
        codex_logs: Path | None = None,
        output_json: Path | None = None,
        output_report: Path | None = None,
        output_log: Path | None = None,
    ) -> "WatchdogPaths":
        automation = automation_dir or DEFAULT_AUTOMATION_DIR
        return cls(
            root=root,
            automation_toml=automation / "automation.toml",
            automation_memory=automation / "memory.md",
            trader_journal=root / "logs" / "trader_journal.jsonl",
            autotrade_report=root / "docs" / "autotrade_cycle_report.md",
            gpt_decision_report=root / "docs" / "gpt_trader_decision_report.md",
            decision_response=root / "data" / "codex_trader_decision_response.json",
            guardian_receipt=root / "data" / "guardian_action_receipt.json",
            guardian_review=root / "docs" / "guardian_action_review.md",
            codex_logs=codex_logs or DEFAULT_CODEX_LOGS,
            output_json=output_json or root / "data" / "qr_trader_run_watchdog.json",
            output_report=output_report or root / "docs" / "qr_trader_run_watchdog_report.md",
            output_log=output_log or root / "logs" / "qr_trader_run_watchdog.log",
        )


def evaluate_watchdog(
    *,
    paths: WatchdogPaths,
    now_utc: datetime | None = None,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    clock = _utc(now_utc or datetime.now(timezone.utc))
    environ = env if env is not None else os.environ
    automation = _automation_config(paths.automation_toml)
    evidence = _latest_run_evidence(paths=paths, now_utc=clock)
    codex_logs = _codex_log_summary(paths.codex_logs, now_utc=clock)
    guardian = _guardian_receipt_status(
        paths.guardian_receipt,
        paths.guardian_review,
        last_trader_run_at=evidence["last_trader_run_at"],
        expected_cadence_minutes=automation.get("cadence_minutes") or EXPECTED_CADENCE_MINUTES,
        grace_minutes=grace_minutes,
        now_utc=clock,
    )

    config_issues = automation["issues"]
    expected_cadence = automation.get("cadence_minutes") or EXPECTED_CADENCE_MINUTES
    threshold_minutes = expected_cadence + grace_minutes
    last_trader_run_at = _parse_utc(evidence.get("last_trader_run_at"))
    minutes_since = None
    missed_expected_window = False
    status = "UNKNOWN"
    issues: list[dict[str, Any]] = []

    issues.extend(config_issues)
    if config_issues:
        status = "BROKEN"
    elif last_trader_run_at is None:
        status = "UNKNOWN"
        issues.append(
            _issue(
                "QR_TRADER_RUN_EVIDENCE_MISSING",
                "P1",
                "No usable trader journal, decision artifact, or automation memory timestamp was found.",
            )
        )
    else:
        minutes_since = round((clock - last_trader_run_at).total_seconds() / 60.0, 3)
        missed_expected_window = minutes_since > threshold_minutes
        if missed_expected_window:
            status = "STALE"
            issues.append(
                _issue(
                    "QR_TRADER_RUN_STALE",
                    "P0",
                    (
                        f"Latest trader run evidence is {minutes_since:.1f} minutes old; "
                        f"expected <= {threshold_minutes} minutes."
                    ),
                )
            )
        else:
            status = "OK"

    issues.extend(guardian["issues"])
    overall_severity = _overall_severity(status, issues)
    suspected_cause = _suspected_cause(status=status, evidence=evidence, codex_logs=codex_logs, guardian=guardian)
    can_wake = str(environ.get("QR_TRADER_WATCHDOG_CAN_WAKE", "0")).strip() == "1"

    payload = {
        "generated_at_utc": clock.isoformat(),
        "status": status,
        "severity": overall_severity,
        "no_live_side_effects": True,
        "codex_exec_enabled": can_wake,
        "broker_writes_enabled": False,
        "missed_expected_window": missed_expected_window,
        "expected_cadence_minutes": expected_cadence,
        "grace_minutes": grace_minutes,
        "threshold_minutes": threshold_minutes,
        "minutes_since_last_run": minutes_since,
        "last_trader_run_at": evidence["last_trader_run_at"],
        "last_journal_at": evidence["last_journal_at"],
        "last_decision_artifact_at": evidence["last_decision_artifact_at"],
        "last_memory_at": evidence["last_memory_at"],
        "automation_config": automation,
        "latest_run_evidence": evidence,
        "guardian_receipt": guardian,
        "codex_logs": codex_logs,
        "suspected_cause": suspected_cause,
        "recommended_operator_action": _recommended_operator_action(status, guardian, config_issues),
        "issues": issues,
        "environment": {
            "QR_TRADER_WATCHDOG_CAN_WAKE": "1" if can_wake else "0",
            "wake_status": (
                "READ_ONLY_DIAGNOSTIC_WAKE_ALLOWED_BUT_NOT_IMPLEMENTED"
                if can_wake
                else "DISABLED_BY_DEFAULT"
            ),
        },
        "execution_boundary": {
            "read_only": True,
            "no_live_side_effects": True,
            "live_side_effects": [],
            "calls_oanda": False,
            "codex_exec_enabled": can_wake,
            "broker_writes_enabled": False,
            "runs_codex_by_default": False,
            "runs_trader_by_default": False,
            "places_orders": False,
            "cancels_orders": False,
            "closes_positions": False,
        },
        "artifact_paths": {
            "automation_toml": str(paths.automation_toml),
            "automation_memory": str(paths.automation_memory),
            "trader_journal": str(paths.trader_journal),
            "autotrade_report": str(paths.autotrade_report),
            "gpt_decision_report": str(paths.gpt_decision_report),
            "decision_response": str(paths.decision_response),
            "guardian_receipt": str(paths.guardian_receipt),
            "guardian_review": str(paths.guardian_review),
            "codex_logs": str(paths.codex_logs),
            "output_json": str(paths.output_json),
            "output_report": str(paths.output_report),
            "output_log": str(paths.output_log),
        },
    }
    return payload


def run_watchdog(
    *,
    paths: WatchdogPaths,
    now_utc: datetime | None = None,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = evaluate_watchdog(paths=paths, now_utc=now_utc, grace_minutes=grace_minutes, env=env)
    paths.output_json.parent.mkdir(parents=True, exist_ok=True)
    paths.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    paths.output_report.parent.mkdir(parents=True, exist_ok=True)
    paths.output_report.write_text(_render_report(payload), encoding="utf-8")
    paths.output_log.parent.mkdir(parents=True, exist_ok=True)
    with paths.output_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "generated_at_utc": payload["generated_at_utc"],
                    "status": payload["status"],
                    "severity": payload["severity"],
                    "last_trader_run_at": payload["last_trader_run_at"],
                    "minutes_since_last_run": payload["minutes_since_last_run"],
                    "issue_codes": [item["code"] for item in payload["issues"]],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return payload


def _automation_config(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    missing = not path.exists()
    if not missing:
        try:
            payload = _load_toml(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - operator evidence should preserve parse failure.
            return {
                "path": str(path),
                "exists": True,
                "parse_error": str(exc),
                "status": None,
                "model": None,
                "reasoning_effort": None,
                "rrule": None,
                "cadence_minutes": None,
                "cwd": None,
                "cwds": [],
                "issues": [_issue("QR_TRADER_AUTOMATION_CONFIG_UNREADABLE", "P0", str(exc))],
            }
    cadence = _cadence_from_rrule(payload.get("rrule"))
    cwds = payload.get("cwds") if isinstance(payload.get("cwds"), list) else []
    cwd = str(cwds[0]) if cwds else str(payload.get("cwd") or "")
    issues: list[dict[str, Any]] = []
    if missing:
        issues.append(_issue("QR_TRADER_AUTOMATION_CONFIG_MISSING", "P0", f"{path} does not exist."))
    if not missing and payload.get("status") != "ACTIVE":
        issues.append(
            _issue(
                "QR_TRADER_AUTOMATION_INACTIVE",
                "P0",
                f"qr-trader status is {payload.get('status')!r}, expected 'ACTIVE'.",
            )
        )
    if not missing and payload.get("model") != EXPECTED_MODEL:
        issues.append(
            _issue(
                "QR_TRADER_AUTOMATION_WRONG_MODEL",
                "P0",
                f"qr-trader model is {payload.get('model')!r}, expected {EXPECTED_MODEL!r}.",
            )
        )
    if not missing and payload.get("reasoning_effort") != EXPECTED_REASONING_EFFORT:
        issues.append(
            _issue(
                "QR_TRADER_AUTOMATION_WRONG_REASONING",
                "P0",
                (
                    f"qr-trader reasoning_effort is {payload.get('reasoning_effort')!r}, "
                    f"expected {EXPECTED_REASONING_EFFORT!r}."
                ),
            )
        )
    if not missing and cadence != EXPECTED_CADENCE_MINUTES:
        issues.append(
            _issue(
                "QR_TRADER_AUTOMATION_WRONG_CADENCE",
                "P0",
                f"qr-trader cadence is {cadence!r} minutes, expected {EXPECTED_CADENCE_MINUTES}.",
            )
        )
    if not missing and EXPECTED_CWD not in [str(item) for item in cwds]:
        issues.append(
            _issue(
                "QR_TRADER_AUTOMATION_WRONG_CWD",
                "P0",
                f"qr-trader cwds is {cwds!r}, expected {EXPECTED_CWD!r}.",
            )
        )
    return {
        "path": str(path),
        "exists": not missing,
        "status": payload.get("status"),
        "model": payload.get("model"),
        "reasoning_effort": payload.get("reasoning_effort"),
        "rrule": payload.get("rrule"),
        "cadence_minutes": cadence,
        "cwd": cwd or None,
        "cwds": cwds,
        "issues": issues,
    }


def _load_toml(text: str) -> dict[str, Any]:
    if tomllib is not None:
        parsed = tomllib.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    parsed: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            parsed[key] = [item.strip().strip('"') for item in value[1:-1].split(",") if item.strip()]
        elif value.startswith('"') and value.endswith('"'):
            parsed[key] = value[1:-1]
        else:
            parsed[key] = value
    return parsed


def _cadence_from_rrule(rrule: Any) -> int | None:
    text = str(rrule or "")
    match = re.search(r"(?:^|;)INTERVAL=(\d+)(?:;|$)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _latest_run_evidence(*, paths: WatchdogPaths, now_utc: datetime) -> dict[str, Any]:
    journal = _journal_evidence(paths.trader_journal)
    decision = _decision_artifact_evidence(
        paths.decision_response,
        paths.gpt_decision_report,
        paths.autotrade_report,
    )
    memory = _memory_evidence(paths.automation_memory)
    candidates = [
        _parse_utc(journal.get("last_at")),
        _parse_utc(decision.get("last_at")),
        _parse_utc(memory.get("last_at")),
    ]
    latest = _max_dt([item for item in candidates if item is not None])
    return {
        "last_trader_run_at": _iso(latest),
        "last_journal_at": journal.get("last_at"),
        "last_decision_artifact_at": decision.get("last_at"),
        "last_memory_at": memory.get("last_at"),
        "journal": journal,
        "decision_artifacts": decision,
        "memory": memory,
        "now_utc": now_utc.isoformat(),
    }


def _journal_evidence(path: Path) -> dict[str, Any]:
    timestamps: list[datetime] = []
    last_status = None
    last_line_excerpt = None
    if path.exists():
        for line in _tail_lines(path, max_bytes=256_000):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                for key in ("ts", "timestamp", "generated_at_utc", "time", "created_at_utc"):
                    parsed = _parse_utc(payload.get(key))
                    if parsed is not None:
                        timestamps.append(parsed)
                        break
                last_status = payload.get("status") or payload.get("gpt_status") or last_status
            last_line_excerpt = stripped[:500]
    mtime = _mtime_utc(path)
    if not timestamps and mtime is not None:
        timestamps.append(mtime)
    latest = _max_dt(timestamps)
    return {
        "path": str(path),
        "exists": path.exists(),
        "last_at": _iso(latest),
        "file_mtime_utc": _iso(mtime),
        "last_status": last_status,
        "last_line_excerpt": last_line_excerpt,
    }


def _decision_artifact_evidence(
    decision_response: Path,
    gpt_report: Path,
    autotrade_report: Path,
) -> dict[str, Any]:
    artifacts = [
        _json_artifact_time(decision_response, "decision_response"),
        _report_artifact_time(gpt_report, "gpt_decision_report"),
        _report_artifact_time(autotrade_report, "autotrade_report"),
    ]
    latest = _max_dt(
        [parsed for parsed in (_parse_utc(item.get("generated_at_utc")) for item in artifacts) if parsed]
    )
    return {
        "last_at": _iso(latest),
        "artifacts": artifacts,
    }


def _json_artifact_time(path: Path, name: str) -> dict[str, Any]:
    payload = _read_json_object(path)
    parsed = None
    status = None
    action = None
    if isinstance(payload, dict):
        for key in ("generated_at_utc", "ts", "timestamp", "created_at_utc"):
            parsed = _parse_utc(payload.get(key))
            if parsed is not None:
                break
        status = payload.get("status")
        action = payload.get("action")
    if parsed is None:
        parsed = _mtime_utc(path)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "generated_at_utc": _iso(parsed),
        "status": status,
        "action": action,
    }


def _report_artifact_time(path: Path, name: str) -> dict[str, Any]:
    text = _read_text(path, limit=16_000)
    parsed = None
    status = None
    if text:
        match = re.search(r"Generated at UTC:\s*`([^`]+)`", text)
        if match:
            parsed = _parse_utc(match.group(1))
        status_match = re.search(r"Status:\s*`([^`]+)`", text)
        if status_match:
            status = status_match.group(1)
    if parsed is None:
        parsed = _mtime_utc(path)
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "generated_at_utc": _iso(parsed),
        "status": status,
    }


def _memory_evidence(path: Path) -> dict[str, Any]:
    text = _read_tail_text(path, max_bytes=128_000)
    timestamps = _timestamps_from_text(text)
    mtime = _mtime_utc(path)
    if mtime is not None:
        timestamps.append(mtime)
    latest = _max_dt(timestamps)
    return {
        "path": str(path),
        "exists": path.exists(),
        "last_at": _iso(latest),
        "file_mtime_utc": _iso(mtime),
        "parsed_timestamp_count": len(timestamps),
    }


def _guardian_receipt_status(
    receipt_path: Path,
    review_path: Path,
    *,
    last_trader_run_at: str | None,
    expected_cadence_minutes: int,
    grace_minutes: int,
    now_utc: datetime,
) -> dict[str, Any]:
    review_text = _read_text(review_path, limit=16_000)
    candidates = _guardian_receipt_candidates(receipt_path)
    groups = _group_guardian_receipts(candidates)
    issues: list[dict[str, Any]] = []
    summaries = [
        _guardian_receipt_group_summary(
            group,
            last_trader_run_at=last_trader_run_at,
            expected_cadence_minutes=expected_cadence_minutes,
            grace_minutes=grace_minutes,
            now_utc=now_utc,
        )
        for group in groups
    ]
    summaries.sort(key=lambda item: item.get("generated_at_utc") or "", reverse=True)
    for summary in summaries:
        issue = _guardian_receipt_issue(summary)
        if issue is not None:
            issues.append(issue)
    current_summary = next((item for item in summaries if item.get("canonical_present")), None)
    if current_summary is None and summaries:
        current_summary = summaries[0]
    archive_dir = receipt_path.parent / "guardian_action_receipts"
    if current_summary is None:
        return {
            "path": str(receipt_path),
            "exists": receipt_path.exists(),
            "active": False,
            "issues": issues,
            "receipts_checked": 0,
            "archive_path": str(archive_dir),
            "archive_exists": archive_dir.exists(),
            "archive_receipts_checked": 0,
            "receipt_summaries": [],
            "review_path": str(review_path),
            "review_excerpt": review_text[:800] if review_text else None,
        }
    return {
        "path": str(receipt_path),
        "exists": receipt_path.exists(),
        "active": current_summary["active"],
        "receipt_status": current_summary["receipt_status"],
        "receipt_lifecycle": current_summary["receipt_lifecycle"],
        "terminal_lifecycle": current_summary["terminal_lifecycle"],
        "action": current_summary["action"],
        "high_urgency_action": current_summary["high_urgency_action"],
        "generated_at_utc": current_summary["generated_at_utc"],
        "expires_at_utc": current_summary["expires_at_utc"],
        "consumed_by_trader": current_summary["consumed_by_trader"],
        "receipt_after_last_trader_run": current_summary["receipt_after_last_trader_run"],
        "next_run_due_utc": current_summary["next_run_due_utc"],
        "expired_before_trader_run": current_summary["expired_before_trader_run"],
        "next_run_window_missed": current_summary["next_run_window_missed"],
        "will_expire_before_next_run": current_summary["will_expire_before_next_run"],
        "dependency_before_next_run": current_summary["dependency_before_next_run"],
        "emergency_or_margin_risk": current_summary["emergency_or_margin_risk"],
        "issues": issues,
        "receipts_checked": len(candidates),
        "archive_path": str(archive_dir),
        "archive_exists": archive_dir.exists(),
        "archive_receipts_checked": len([item for item in candidates if item["source"] == "archive"]),
        "receipt_summaries": summaries,
        "review_path": str(review_path),
        "review_exists": review_path.exists(),
        "review_excerpt": review_text[:800] if review_text else None,
    }


def _guardian_receipt_candidates(receipt_path: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    current = _read_json_object(receipt_path)
    if isinstance(current, dict):
        candidates.append({"source": "canonical", "path": str(receipt_path), "payload": current})
    archive_dir = receipt_path.parent / "guardian_action_receipts"
    if archive_dir.exists():
        for path in sorted(archive_dir.glob("*.json")):
            payload = _read_json_object(path)
            if isinstance(payload, dict):
                candidates.append({"source": "archive", "path": str(path), "payload": payload})
    return candidates


def _group_guardian_receipts(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        identity = _guardian_receipt_identity(candidate["payload"])
        grouped.setdefault(identity, []).append(candidate)
    return list(grouped.values())


def _guardian_receipt_identity(receipt: dict[str, Any]) -> str:
    nested = _nested_receipt(receipt)
    selected = _selected_event(receipt)
    action = _guardian_receipt_action(receipt)
    generated = str(receipt.get("generated_at_utc") or nested.get("generated_at_utc") or "")
    event_id = str(nested.get("event_id") or receipt.get("event_id") or selected.get("event_id") or "")
    dedupe_key = str(nested.get("dedupe_key") or receipt.get("dedupe_key") or selected.get("dedupe_key") or "")
    if event_id:
        return "|".join(["event", event_id, action, generated])
    if dedupe_key:
        return "|".join(["dedupe", dedupe_key, action, generated])
    fallback = json.dumps(
        {
            "action": action,
            "generated_at_utc": generated,
            "expires_at_utc": receipt.get("expires_at_utc") or nested.get("expires_at_utc"),
            "reason": nested.get("reason") or receipt.get("reason"),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return "hash|" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def _guardian_receipt_group_summary(
    group: list[dict[str, Any]],
    *,
    last_trader_run_at: str | None,
    expected_cadence_minutes: int,
    grace_minutes: int,
    now_utc: datetime,
) -> dict[str, Any]:
    payloads = [item["payload"] for item in group]
    canonical_present = any(item["source"] == "canonical" for item in group)
    representative = _guardian_receipt_representative(payloads)
    action = _guardian_receipt_action(representative)
    lifecycle = _guardian_receipt_lifecycle(payloads)
    receipt_status = _guardian_receipt_status_value(payloads)
    generated_at = _guardian_receipt_datetime(payloads, "generated_at_utc")
    expires_at = _guardian_receipt_datetime(payloads, "expires_at_utc")
    consumed = any(_bool_value(payload.get("consumed_by_trader")) for payload in payloads)
    active = receipt_status == "ACCEPTED" and lifecycle == "ACTIVE"
    last_run = _parse_utc(last_trader_run_at)
    next_run_due = None
    if last_run is not None:
        next_run_due = last_run + timedelta(minutes=expected_cadence_minutes + grace_minutes)
    receipt_after_last_run = generated_at is not None and (last_run is None or generated_at > last_run)
    expired_before_trader_run = bool(
        receipt_status == "ACCEPTED"
        and not consumed
        and (
            lifecycle == "EXPIRED"
            or (
                active
                and receipt_after_last_run
                and expires_at is not None
                and expires_at <= now_utc
            )
        )
    )
    next_run_window_missed = bool(
        active
        and not consumed
        and receipt_after_last_run
        and next_run_due is not None
        and now_utc > next_run_due
    )
    will_expire_before_next_run = bool(
        active
        and not consumed
        and receipt_after_last_run
        and expires_at is not None
        and next_run_due is not None
        and expires_at < next_run_due
    )
    dependency_before_next_run = bool(
        active
        and not consumed
        and receipt_after_last_run
        and next_run_due is not None
        and now_utc <= next_run_due
        and not expired_before_trader_run
    )
    emergency_or_margin = any(_receipt_is_emergency_or_margin_risk(payload) for payload in payloads)
    return {
        "identity": _guardian_receipt_identity(representative),
        "canonical_present": canonical_present,
        "source_paths": sorted(item["path"] for item in group),
        "sources": sorted({item["source"] for item in group}),
        "active": active,
        "receipt_status": receipt_status or None,
        "receipt_lifecycle": lifecycle or None,
        "terminal_lifecycle": lifecycle in TERMINAL_RECEIPT_LIFECYCLES,
        "action": action or None,
        "high_urgency_action": action in HIGH_URGENCY_GUARDIAN_ACTIONS,
        "generated_at_utc": _iso(generated_at),
        "expires_at_utc": _iso(expires_at),
        "consumed_by_trader": consumed,
        "receipt_after_last_trader_run": receipt_after_last_run,
        "next_run_due_utc": _iso(next_run_due),
        "expired_before_trader_run": expired_before_trader_run,
        "next_run_window_missed": next_run_window_missed,
        "will_expire_before_next_run": will_expire_before_next_run,
        "dependency_before_next_run": dependency_before_next_run,
        "emergency_or_margin_risk": emergency_or_margin,
    }


def _guardian_receipt_issue(summary: dict[str, Any]) -> dict[str, Any] | None:
    if summary.get("receipt_status") != "ACCEPTED" or summary.get("consumed_by_trader"):
        return None
    if not summary.get("expired_before_trader_run") and not summary.get("next_run_window_missed"):
        return None
    reasons = []
    lifecycle = str(summary.get("receipt_lifecycle") or "")
    if lifecycle == "EXPIRED":
        reasons.append("receipt_lifecycle=EXPIRED while consumed_by_trader=false")
    elif summary.get("expired_before_trader_run"):
        reasons.append("receipt expired before any later trader run consumed or classified it")
    if summary.get("next_run_window_missed"):
        reasons.append("next trader run window passed without consumption/classification")
    message = "; ".join(reasons)
    source_paths = summary.get("source_paths") if isinstance(summary.get("source_paths"), list) else []
    if source_paths:
        message += f"; sources={len(source_paths)}"
    return _issue(
        "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
        _guardian_receipt_issue_severity(summary),
        message,
    )


def _guardian_receipt_issue_severity(summary: dict[str, Any]) -> str:
    if summary.get("emergency_or_margin_risk"):
        return "P0"
    action = str(summary.get("action") or "").upper()
    if action in HIGH_URGENCY_GUARDIAN_ACTIONS:
        return "P1"
    if action in LOW_URGENCY_GUARDIAN_ACTIONS:
        return "WARN"
    return "WARN"


def _guardian_receipt_representative(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        payloads,
        key=lambda payload: (
            RECEIPT_LIFECYCLE_PRECEDENCE.get(str(payload.get("receipt_lifecycle") or "").upper(), 0),
            _parse_utc(payload.get("generated_at_utc")) or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )


def _guardian_receipt_lifecycle(payloads: list[dict[str, Any]]) -> str:
    lifecycles = [str(payload.get("receipt_lifecycle") or "").upper() for payload in payloads]
    lifecycles = [item for item in lifecycles if item]
    if not lifecycles and any(_guardian_receipt_status_value([payload]) == "ACCEPTED" for payload in payloads):
        return "ACTIVE"
    return max(lifecycles, key=lambda item: RECEIPT_LIFECYCLE_PRECEDENCE.get(item, 0), default="")


def _guardian_receipt_status_value(payloads: list[dict[str, Any]]) -> str:
    statuses = [str(payload.get("receipt_status") or payload.get("status") or "").upper() for payload in payloads]
    if "ACCEPTED" in statuses:
        return "ACCEPTED"
    return next((item for item in statuses if item), "")


def _guardian_receipt_datetime(payloads: list[dict[str, Any]], key: str) -> datetime | None:
    parsed = [_parse_utc(payload.get(key)) for payload in payloads]
    parsed = [item for item in parsed if item is not None]
    return _max_dt(parsed)


def _nested_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    nested = receipt.get("receipt")
    return nested if isinstance(nested, dict) else {}


def _selected_event(receipt: dict[str, Any]) -> dict[str, Any]:
    selected = receipt.get("selected_event")
    return selected if isinstance(selected, dict) else {}


def _guardian_receipt_action(receipt: dict[str, Any]) -> str:
    nested = _nested_receipt(receipt)
    return str(nested.get("action") or receipt.get("action") or "").upper()


def _receipt_is_emergency_or_margin_risk(receipt: dict[str, Any]) -> bool:
    nested = _nested_receipt(receipt)
    selected = _selected_event(receipt)
    thesis_states = [
        receipt.get("thesis_state"),
        nested.get("thesis_state"),
        selected.get("thesis_state"),
    ]
    if any(str(item or "").upper() == "EMERGENCY" for item in thesis_states):
        return True
    review_type = str(selected.get("recommended_review_type") or receipt.get("recommended_review_type") or "").upper()
    if "EMERGENCY" in review_type or "MARGIN" in review_type:
        return True
    event_texts = [
        selected.get("event_type"),
        receipt.get("event_type"),
        selected.get("dedupe_key"),
        nested.get("dedupe_key"),
    ]
    if any("MARGIN" in str(item or "").upper() for item in event_texts):
        return True
    margin_state = str(nested.get("margin_state") or receipt.get("margin_state") or "").lower()
    return "margin_pressure=true" in margin_state or "margin pressure=true" in margin_state


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _codex_log_summary(path: Path, *, now_utc: datetime) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "available": False, "entries": []}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error as exc:
        return {"path": str(path), "exists": True, "available": False, "error": str(exc), "entries": []}
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            """
            SELECT id, ts, level, target, substr(coalesce(feedback_log_body, ''), 1, 2000)
            FROM logs
            WHERE feedback_log_body LIKE ?
               OR feedback_log_body LIKE ?
               OR feedback_log_body LIKE ?
            ORDER BY ts DESC
            LIMIT 25
            """,
            (
                "%Automation ID: qr-trader%",
                "%QR vNext Trader%",
                f"%cwd={EXPECTED_CWD}%",
            ),
        ).fetchall()
    except sqlite3.Error as exc:
        return {"path": str(path), "exists": True, "available": False, "error": str(exc), "entries": []}
    finally:
        connection.close()
    entries: list[dict[str, Any]] = []
    scheduler_last_run_at: datetime | None = None
    latest_live_cwd_at: datetime | None = None
    cause_hints: list[str] = []
    for row_id, ts, level, target, body in rows:
        ts_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        text = str(body or "")
        last_run_match = re.search(r"Last run:\s*([0-9T:.\-+Z]+)", text)
        if last_run_match:
            parsed = _parse_utc(last_run_match.group(1))
            if parsed is not None:
                scheduler_last_run_at = _max_dt([item for item in (scheduler_last_run_at, parsed) if item])
        if f"cwd={EXPECTED_CWD}" in text:
            latest_live_cwd_at = _max_dt([item for item in (latest_live_cwd_at, ts_dt) if item])
        hint = _cause_hint_from_log_text(text)
        if hint and hint not in cause_hints:
            cause_hints.append(hint)
        entries.append(
            {
                "id": row_id,
                "ts_utc": ts_dt.isoformat(),
                "level": level,
                "target": target,
                "excerpt": text[:500],
            }
        )
    return {
        "path": str(path),
        "exists": True,
        "available": True,
        "queried_at_utc": now_utc.isoformat(),
        "entry_count": len(entries),
        "scheduler_last_run_at": _iso(scheduler_last_run_at),
        "latest_live_cwd_log_at": _iso(latest_live_cwd_at),
        "cause_hints": cause_hints,
        "entries": entries[:8],
    }


def _cause_hint_from_log_text(text: str) -> str | None:
    lower = text.lower()
    patterns = [
        ("credit", "Codex credit or quota issue appears in recent logs."),
        ("quota", "Codex quota issue appears in recent logs."),
        ("exhaust", "Codex credit/exhaustion issue appears in recent logs."),
        ("model requires a newer version", "Codex CLI model compatibility issue appears in recent logs."),
        ("unsupported", "Unsupported model or capability issue appears in recent logs."),
        ("auth", "Authentication issue appears in recent logs."),
        ("sandbox", "Sandbox/permission issue appears in recent logs."),
        ("timeout", "Timeout appears in recent logs."),
        ("error", "Error text appears in recent logs."),
    ]
    for needle, message in patterns:
        if needle in lower:
            return message
    return None


def _suspected_cause(
    *,
    status: str,
    evidence: dict[str, Any],
    codex_logs: dict[str, Any],
    guardian: dict[str, Any],
) -> str | None:
    hints = codex_logs.get("cause_hints") if isinstance(codex_logs.get("cause_hints"), list) else []
    if status == "BROKEN":
        return "qr-trader automation config does not match the active hourly gpt-5.5 high live-root policy."
    if status == "UNKNOWN":
        return "Run evidence is insufficient; automation config exists but no journal, decision, or memory timestamp was usable."
    if guardian.get("issues"):
        return "Active guardian receipt is waiting on the next trader run to consume or classify it."
    if status == "STALE":
        if hints:
            return str(hints[0])
        scheduler_seen = codex_logs.get("scheduler_last_run_at") or codex_logs.get("latest_live_cwd_log_at")
        if scheduler_seen:
            return (
                "Codex logs show a recent qr-trader/live-root session, but trader artifacts are stale; "
                "inspect the latest Codex thread for preflight, quota, or early-exit failure."
            )
        return "No explicit Codex log cause was visible; latest trader artifacts missed the expected hourly window."
    return None


def _recommended_operator_action(
    status: str,
    guardian: dict[str, Any],
    config_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    if status == "BROKEN":
        return {
            "code": "REPAIR_QR_TRADER_AUTOMATION_CONFIG",
            "requires_explicit_operator_approval": True,
            "command": "open Codex automation qr-trader settings and restore ACTIVE gpt-5.5 high hourly live-root policy",
            "reason": "; ".join(item["code"] for item in config_issues),
        }
    if status == "STALE":
        return {
            "code": "INSPECT_CODEX_QR_TRADER_THREAD",
            "requires_explicit_operator_approval": False,
            "command": "review latest QR vNext Trader Codex thread/logs; do not run live trader manually from the watchdog",
            "reason": "scheduled-run evidence missed the hourly cadence plus grace",
        }
    if guardian.get("issues"):
        return {
            "code": "RESOLVE_GUARDIAN_RECEIPT_IN_NEXT_TRADER_CYCLE",
            "requires_explicit_operator_approval": False,
            "command": "read data/guardian_action_receipt.json and docs/guardian_action_review.md before normal entries",
            "reason": "guardian receipt is active and unconsumed",
        }
    if status == "UNKNOWN":
        return {
            "code": "REBUILD_QR_TRADER_RUN_EVIDENCE",
            "requires_explicit_operator_approval": False,
            "command": "inspect qr-trader memory, live trader journal, and decision reports",
            "reason": "watchdog could not find enough run evidence",
        }
    return {
        "code": "NO_OPERATOR_ACTION_REQUIRED",
        "requires_explicit_operator_approval": False,
        "command": "none",
        "reason": "qr-trader run evidence is within cadence plus grace",
    }


def _overall_severity(status: str, issues: list[dict[str, Any]]) -> str:
    if status in {"BROKEN", "STALE"}:
        return "P0"
    ranks = {"P0": 4, "P1": 3, "WARN": 2, "INFO": 1}
    severity = "OK" if status == "OK" else "P1"
    best = ranks.get(severity, 0)
    for issue in issues:
        issue_severity = str(issue.get("severity") or "")
        if ranks.get(issue_severity, 0) > best:
            severity = issue_severity
            best = ranks[issue_severity]
    return severity


def _issue(code: str, severity: str, message: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message}


def _render_report(payload: dict[str, Any]) -> str:
    automation = payload["automation_config"]
    evidence = payload["latest_run_evidence"]
    guardian = payload["guardian_receipt"]
    codex = payload["codex_logs"]
    lines = [
        "# QR Trader Run Watchdog Report",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}` severity=`{payload['severity']}`",
        f"- Missed expected window: `{payload['missed_expected_window']}`",
        f"- Last trader run evidence: `{payload['last_trader_run_at']}`",
        f"- Minutes since last run: `{payload['minutes_since_last_run']}`",
        f"- Cadence + grace: `{payload['expected_cadence_minutes']} + {payload['grace_minutes']}` minutes",
        f"- Read only: `{payload['execution_boundary']['read_only']}`",
        f"- no_live_side_effects={'true' if payload.get('no_live_side_effects') else 'false'}",
        f"- codex_exec_enabled={'true' if payload.get('codex_exec_enabled') else 'false'}",
        f"- broker_writes_enabled={'true' if payload.get('broker_writes_enabled') else 'false'}",
        f"- Live side effects: `{len(payload['execution_boundary']['live_side_effects'])}`",
        "",
        "## Automation Config",
        "",
        f"- Path: `{automation['path']}` exists=`{automation['exists']}`",
        f"- Status: `{automation.get('status')}`",
        f"- Model: `{automation.get('model')}`",
        f"- Reasoning effort: `{automation.get('reasoning_effort')}`",
        f"- RRULE: `{automation.get('rrule')}` cadence_minutes=`{automation.get('cadence_minutes')}`",
        f"- CWDs: `{automation.get('cwds')}`",
        "",
        "## Latest Run Evidence",
        "",
        f"- Journal: `{payload['last_journal_at']}` path=`{evidence['journal']['path']}`",
        f"- Decision artifacts: `{payload['last_decision_artifact_at']}`",
        f"- Automation memory: `{payload['last_memory_at']}` path=`{evidence['memory']['path']}`",
        "",
        "| Artifact | Generated UTC | Status |",
        "|---|---:|---|",
    ]
    for item in evidence["decision_artifacts"]["artifacts"]:
        lines.append(
            f"| `{item.get('name')}` | `{item.get('generated_at_utc')}` | `{item.get('status')}` |"
        )
    lines.extend(
        [
            "",
            "## Guardian Receipt",
            "",
            f"- Exists: `{guardian.get('exists')}` active=`{guardian.get('active')}`",
            f"- Status/lifecycle/action: `{guardian.get('receipt_status')}` / `{guardian.get('receipt_lifecycle')}` / `{guardian.get('action')}`",
            f"- Generated: `{guardian.get('generated_at_utc')}` expires=`{guardian.get('expires_at_utc')}`",
            f"- Consumed by trader: `{guardian.get('consumed_by_trader')}`",
            f"- Receipt after last trader run: `{guardian.get('receipt_after_last_trader_run')}`",
            f"- Next run due: `{guardian.get('next_run_due_utc')}`",
            f"- Dependency before next run: `{guardian.get('dependency_before_next_run')}`",
            f"- Receipts checked: `{guardian.get('receipts_checked')}` archive=`{guardian.get('archive_receipts_checked')}`",
            "",
            "## Codex Logs",
            "",
            f"- Available: `{codex.get('available')}` path=`{codex.get('path')}`",
            f"- Scheduler last run seen: `{codex.get('scheduler_last_run_at')}`",
            f"- Latest live-root log: `{codex.get('latest_live_cwd_log_at')}`",
            f"- Cause hints: `{codex.get('cause_hints')}`",
            "",
            "## Issues",
            "",
        ]
    )
    if payload["issues"]:
        for issue in payload["issues"]:
            lines.append(f"- `{issue['severity']}` `{issue['code']}`: {issue['message']}")
    else:
        lines.append("- none")
    action = payload["recommended_operator_action"]
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- Suspected cause: {payload['suspected_cause'] or 'none'}",
            f"- Recommended action: `{action['code']}`",
            f"- Command/instruction: `{action['command']}`",
            f"- Requires explicit approval: `{action['requires_explicit_operator_approval']}`",
            "",
            "## Boundary",
            "",
            f"- no_live_side_effects={'true' if payload.get('no_live_side_effects') else 'false'}",
            f"- codex_exec_enabled={'true' if payload.get('codex_exec_enabled') else 'false'}",
            f"- broker_writes_enabled={'true' if payload.get('broker_writes_enabled') else 'false'}",
            "- This watchdog does not call OANDA.",
            "- This watchdog does not run `codex exec` or the trader by default.",
            "- This watchdog does not place, cancel, or close broker orders.",
            f"- QR_TRADER_WATCHDOG_CAN_WAKE: `{payload['environment']['QR_TRADER_WATCHDOG_CAN_WAKE']}` ({payload['environment']['wake_status']})",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_text(path: Path, *, limit: int) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def _read_tail_text(path: Path, *, max_bytes: int) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _tail_lines(path: Path, *, max_bytes: int) -> list[str]:
    return _read_tail_text(path, max_bytes=max_bytes).splitlines()


def _timestamps_from_text(text: str) -> list[datetime]:
    timestamps: list[datetime] = []
    iso_pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})"
    )
    for match in iso_pattern.finditer(text):
        parsed = _parse_utc(match.group(0))
        if parsed is not None:
            timestamps.append(parsed)
    jst_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?::(\d{2}))?\s+JST")
    for match in jst_pattern.finditer(text):
        second = match.group(3) or "00"
        parsed = _parse_utc(f"{match.group(1)}T{match.group(2)}:{second}+09:00")
        if parsed is not None:
            timestamps.append(parsed)
    return timestamps


def _parse_utc(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _utc(parsed)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _mtime_utc(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _max_dt(values: list[datetime]) -> datetime | None:
    return max(values) if values else None


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a read-only qr-trader scheduled-run watchdog report.")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("QR_TRADER_WATCHDOG_ROOT", str(DEFAULT_LIVE_ROOT))))
    parser.add_argument("--automation-dir", type=Path, default=DEFAULT_AUTOMATION_DIR)
    parser.add_argument("--codex-logs", type=Path, default=DEFAULT_CODEX_LOGS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--grace-minutes", type=int, default=int(os.environ.get("QR_TRADER_WATCHDOG_GRACE_MINUTES", DEFAULT_GRACE_MINUTES)))
    parser.add_argument("--now-utc", default=None)
    args = parser.parse_args(argv)
    now = _parse_utc(args.now_utc) if args.now_utc else None
    paths = WatchdogPaths.from_root(
        args.root,
        automation_dir=args.automation_dir,
        codex_logs=args.codex_logs,
        output_json=args.output,
        output_report=args.report,
        output_log=args.log,
    )
    payload = run_watchdog(paths=paths, now_utc=now, grace_minutes=args.grace_minutes)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "severity": payload["severity"],
                "missed_expected_window": payload["missed_expected_window"],
                "last_trader_run_at": payload["last_trader_run_at"],
                "minutes_since_last_run": payload["minutes_since_last_run"],
                "output_json": str(paths.output_json),
                "output_report": str(paths.output_report),
                "issues": payload["issues"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "OK" and payload["severity"] in {"OK", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
