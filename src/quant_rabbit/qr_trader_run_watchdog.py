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

from quant_rabbit.guardian_receipt_consumption import (
    NEEDS_OPERATOR_REVIEW,
    OPERATOR_REVIEW_ISSUE_CODE,
    acknowledgement_for_receipt,
    load_guardian_receipt_consumption,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


EXPECTED_MODEL = "gpt-5.5"
EXPECTED_REASONING_EFFORT = "high"
EXPECTED_CWD = "/Users/tossaki/App/QuantRabbit-live"
EXPECTED_CADENCE_MINUTES = 360
DEFAULT_GRACE_MINUTES = 15
AI_REGIME_SUPERVISION_CONTRACT = "QR_AI_REGIME_SUPERVISION_V1"
DEFAULT_LIVE_ROOT = Path(EXPECTED_CWD)
DEFAULT_AUTOMATION_DIR = Path.home() / ".codex" / "automations" / "qr-trader"
DEFAULT_WEEKEND_STATE = Path.home() / ".codex" / "quant_rabbit_weekend_task_state.json"
DEFAULT_CODEX_LOGS = Path.home() / ".codex" / "logs_2.sqlite"
HIGH_URGENCY_GUARDIAN_ACTIONS = {"REDUCE", "HARVEST", "CANCEL_PENDING"}
LOW_URGENCY_GUARDIAN_ACTIONS = {"HOLD", "NO_ACTION"}
AI_REGIME_SUPERVISION_PAIR_KEYS = {"mode", "reason", "expires_at_utc"}
AI_REGIME_SUPERVISION_MODES = {"GO", "CAUTION", "STOP"}
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
    weekend_state: Path
    trader_journal: Path
    autotrade_report: Path
    gpt_decision_report: Path
    decision_response: Path
    ai_regime_supervision: Path
    guardian_receipt: Path
    guardian_review: Path
    guardian_trigger_contract: Path
    guardian_receipt_consumption: Path
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
        weekend_state: Path | None = None,
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
            weekend_state=weekend_state or DEFAULT_WEEKEND_STATE,
            trader_journal=root / "logs" / "trader_journal.jsonl",
            autotrade_report=root / "docs" / "autotrade_cycle_report.md",
            gpt_decision_report=root / "docs" / "gpt_trader_decision_report.md",
            decision_response=root / "data" / "codex_trader_decision_response.json",
            ai_regime_supervision=root / "data" / "ai_regime_supervision.json",
            guardian_receipt=root / "data" / "guardian_action_receipt.json",
            guardian_review=root / "docs" / "guardian_action_review.md",
            guardian_trigger_contract=root / "data" / "guardian_trigger_contract.json",
            guardian_receipt_consumption=root / "data" / "guardian_receipt_consumption.json",
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
    weekend_pause = _weekend_pause_status(paths.weekend_state, automation, clock)
    if weekend_pause.get("active"):
        automation["issues"] = [
            issue
            for issue in automation.get("issues", [])
            if issue.get("code") != "QR_TRADER_AUTOMATION_INACTIVE"
        ]
    automation["weekend_pause"] = weekend_pause
    evidence = _latest_run_evidence(paths=paths, now_utc=clock)
    codex_logs = _codex_log_summary(paths.codex_logs, now_utc=clock)
    receipt_consumption = load_guardian_receipt_consumption(paths.guardian_receipt_consumption)
    guardian = _guardian_receipt_status(
        paths.guardian_receipt,
        paths.guardian_review,
        paths.guardian_trigger_contract,
        receipt_consumption=receipt_consumption,
        last_trader_run_at=evidence["last_trader_run_at"],
        expected_cadence_minutes=automation.get("cadence_minutes") or EXPECTED_CADENCE_MINUTES,
        grace_minutes=grace_minutes,
        now_utc=clock,
    )

    config_issues = automation["issues"]
    weekend_cadence_paused = bool(weekend_pause.get("active"))
    expected_cadence = automation.get("cadence_minutes") or EXPECTED_CADENCE_MINUTES
    threshold_minutes = expected_cadence + grace_minutes
    last_trader_run_at = _parse_utc(evidence.get("last_trader_run_at"))
    minutes_since = None
    missed_expected_window = False
    status = "UNKNOWN"
    issues: list[dict[str, Any]] = []
    supervision_evidence = evidence.get("ai_regime_supervision", {})
    supervision_invalid = bool(
        isinstance(supervision_evidence, dict)
        and supervision_evidence.get("exists")
        and not supervision_evidence.get("valid_sealed_artifact")
    )

    issues.extend(config_issues)
    if config_issues:
        status = "BROKEN"
    elif supervision_invalid:
        status = "BROKEN"
        issues.append(
            _issue(
                "AI_REGIME_SUPERVISION_INVALID",
                "P1",
                "Existing AI supervision artifact is invalid and cannot be masked by legacy trader evidence: "
                f"{supervision_evidence.get('invalid_reason') or 'unknown validation failure'}",
            )
        )
    elif last_trader_run_at is None:
        if weekend_cadence_paused:
            status = "OK"
        else:
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
        if missed_expected_window and weekend_cadence_paused:
            missed_expected_window = False
            status = "OK"
        elif missed_expected_window:
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
    runtime_status = status
    overall_severity = _overall_severity(runtime_status, issues)
    issue_status = overall_severity
    overall_status = "BLOCKED" if overall_severity in {"P0", "P1"} else runtime_status
    status = "BLOCKED" if runtime_status == "OK" and overall_status == "BLOCKED" else runtime_status
    suspected_cause = _suspected_cause(status=runtime_status, evidence=evidence, codex_logs=codex_logs, guardian=guardian)
    can_wake = str(environ.get("QR_TRADER_WATCHDOG_CAN_WAKE", "0")).strip() == "1"

    payload = {
        "generated_at_utc": clock.isoformat(),
        "status": status,
        "runtime_status": runtime_status,
        "issue_status": issue_status,
        "overall_status": overall_status,
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
        "last_trader_run_source": evidence["last_trader_run_source"],
        "last_trader_run_path": evidence["last_trader_run_path"],
        "accepted_timestamp_candidate": evidence["accepted_timestamp_candidate"],
        "rejected_timestamp_candidates": evidence["rejected_timestamp_candidates"],
        "last_journal_at": evidence["last_journal_at"],
        "last_decision_artifact_at": evidence["last_decision_artifact_at"],
        "last_ai_regime_supervision_at": evidence["last_ai_regime_supervision_at"],
        "last_memory_at": evidence["last_memory_at"],
        "automation_config": automation,
        "weekend_pause": weekend_pause,
        "latest_run_evidence": evidence,
        "guardian_receipt": guardian,
        "codex_logs": codex_logs,
        "suspected_cause": suspected_cause,
        "recommended_operator_action": _recommended_operator_action(runtime_status, guardian, config_issues),
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
            "weekend_state": str(paths.weekend_state),
            "trader_journal": str(paths.trader_journal),
            "autotrade_report": str(paths.autotrade_report),
            "gpt_decision_report": str(paths.gpt_decision_report),
            "decision_response": str(paths.decision_response),
            "ai_regime_supervision": str(paths.ai_regime_supervision),
            "guardian_receipt": str(paths.guardian_receipt),
            "guardian_review": str(paths.guardian_review),
            "guardian_trigger_contract": str(paths.guardian_trigger_contract),
            "guardian_receipt_consumption": str(paths.guardian_receipt_consumption),
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
                    "last_trader_run_source": payload["last_trader_run_source"],
                    "last_trader_run_path": payload["last_trader_run_path"],
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


def _weekend_pause_status(path: Path, automation: dict[str, Any], now_utc: datetime) -> dict[str, Any]:
    payload = _read_json_object(path)
    now_jst = now_utc.astimezone(timezone(timedelta(hours=9), "JST"))
    mode = payload.get("mode") if isinstance(payload, dict) else None
    managed_keys = payload.get("managed_task_keys") if isinstance(payload, dict) else None
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    changes = payload.get("last_changes") if isinstance(payload, dict) else None
    qr_trader_managed = (
        _contains_qr_trader(managed_keys)
        or _contains_qr_trader(tasks.keys() if isinstance(tasks, dict) else [])
        or _changes_include_qr_trader(changes)
    )
    in_pause_window = _is_weekend_pause_window(now_jst)
    automation_paused = str(automation.get("status") or "").upper() == "PAUSED"
    active = bool(
        payload
        and mode == "paused"
        and qr_trader_managed
        and in_pause_window
        and automation_paused
    )
    return {
        "path": str(path),
        "exists": path.exists(),
        "active": active,
        "mode": mode,
        "automation_status": automation.get("status"),
        "qr_trader_managed": qr_trader_managed,
        "in_weekend_pause_window": in_pause_window,
        "now_jst": now_jst.isoformat(),
        "pause_applied_at_utc": payload.get("pause_applied_at_utc") if isinstance(payload, dict) else None,
        "created_at_utc": payload.get("created_at_utc") if isinstance(payload, dict) else None,
        "reason": (
            "qr-trader is intentionally paused by the weekend task switcher"
            if active
            else "no active weekend pause exemption"
        ),
    }


def _contains_qr_trader(values: Any) -> bool:
    if not isinstance(values, (list, tuple, set)):
        return False
    return any(str(item) == "codex:qr-trader" for item in values)


def _changes_include_qr_trader(changes: Any) -> bool:
    if not isinstance(changes, list):
        return False
    return any(isinstance(item, dict) and item.get("key") == "codex:qr-trader" for item in changes)


def _is_weekend_pause_window(now_jst: datetime) -> bool:
    weekday = now_jst.weekday()
    local_minutes = now_jst.hour * 60 + now_jst.minute
    if weekday == 5:
        return local_minutes >= 6 * 60
    if weekday == 6:
        return True
    if weekday == 0:
        return local_minutes < 7 * 60
    return False


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
    supervision = _ai_regime_supervision_evidence(
        paths.ai_regime_supervision,
        now_utc=now_utc,
    )
    journal = _journal_evidence(paths.trader_journal)
    decision = _decision_artifact_evidence(
        paths.decision_response,
        paths.gpt_decision_report,
        paths.autotrade_report,
    )
    memory = _memory_evidence(paths.automation_memory)
    legacy_accepted_candidates: list[dict[str, Any]] = []
    for item in (
        journal.get("accepted_timestamp_candidates"),
        decision.get("accepted_timestamp_candidates"),
        memory.get("accepted_timestamp_candidates"),
    ):
        if isinstance(item, list):
            legacy_accepted_candidates.extend(candidate for candidate in item if isinstance(candidate, dict))
    supervision_candidates = supervision.get("accepted_timestamp_candidates")
    supervision_exists = bool(supervision.get("exists"))
    preferred_candidates = (
        [candidate for candidate in supervision_candidates if isinstance(candidate, dict)]
        if supervision_exists and isinstance(supervision_candidates, list)
        else legacy_accepted_candidates
    )
    accepted_candidates = [
        *(
            [candidate for candidate in supervision_candidates if isinstance(candidate, dict)]
            if isinstance(supervision_candidates, list)
            else []
        ),
        *legacy_accepted_candidates,
    ]
    latest_candidate = _latest_timestamp_candidate(preferred_candidates)
    latest = _parse_utc(latest_candidate.get("timestamp_utc")) if latest_candidate else None
    rejected_candidates: list[dict[str, Any]] = []
    for item in (
        supervision.get("rejected_timestamp_candidates"),
        journal.get("rejected_timestamp_candidates"),
        decision.get("rejected_timestamp_candidates"),
        memory.get("rejected_timestamp_candidates"),
    ):
        if isinstance(item, list):
            rejected_candidates.extend(candidate for candidate in item if isinstance(candidate, dict))
    rejected_candidates.extend(_non_trader_timestamp_candidates(paths))
    return {
        "last_trader_run_at": _iso(latest),
        "last_trader_run_source": latest_candidate.get("source") if latest_candidate else None,
        "last_trader_run_path": latest_candidate.get("path") if latest_candidate else None,
        "accepted_timestamp_candidate": latest_candidate,
        "accepted_timestamp_candidates": accepted_candidates,
        "rejected_timestamp_candidates": rejected_candidates,
        "last_journal_at": journal.get("last_at"),
        "last_decision_artifact_at": decision.get("last_at"),
        "last_ai_regime_supervision_at": supervision.get("last_at"),
        "last_memory_at": memory.get("last_at"),
        "evidence_priority": (
            "AI_REGIME_SUPERVISION"
            if isinstance(supervision_candidates, list) and supervision_candidates
            else (
                "INVALID_AI_REGIME_SUPERVISION_FAIL_CLOSED"
                if supervision_exists
                else "LEGACY_TRADER_ARTIFACT_FALLBACK"
            )
        ),
        "ai_regime_supervision": supervision,
        "journal": journal,
        "decision_artifacts": decision,
        "memory": memory,
        "now_utc": now_utc.isoformat(),
    }


def _ai_regime_supervision_evidence(
    path: Path,
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    payload = _read_json_object(path)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    parsed = _parse_utc(payload.get("generated_at_utc")) if isinstance(payload, dict) else None
    invalid_reason = _ai_regime_supervision_invalid_reason(
        payload,
        now_utc=now_utc,
    )
    if parsed is not None:
        candidate = _timestamp_candidate(
            source="ai_regime_supervision.generated_at_utc",
            path=path,
            timestamp=parsed,
        )
        if invalid_reason is None:
            accepted.append(candidate)
        else:
            rejected.append({**candidate, "rejected_reason": invalid_reason})
    mtime = _mtime_utc(path)
    if mtime is not None:
        rejected.append(
            _timestamp_candidate(
                source="ai_regime_supervision.file_mtime",
                path=path,
                timestamp=mtime,
                rejected_reason=(
                    "file mtime is not explicit sealed AI supervision run evidence"
                    if invalid_reason is None
                    else f"AI supervision artifact is invalid: {invalid_reason}"
                ),
            )
        )
    latest = _latest_timestamp_candidate(accepted)
    return {
        "path": str(path),
        "exists": path.exists(),
        "valid_sealed_artifact": invalid_reason is None,
        "invalid_reason": invalid_reason,
        "last_at": latest.get("timestamp_utc") if latest else None,
        "ai_order_authority": payload.get("ai_order_authority") if isinstance(payload, dict) else None,
        "broker_mutation_allowed": payload.get("broker_mutation_allowed") if isinstance(payload, dict) else None,
        "accepted_timestamp_candidates": accepted,
        "rejected_timestamp_candidates": rejected,
    }


def _ai_regime_supervision_invalid_reason(
    payload: dict[str, Any] | None,
    *,
    now_utc: datetime,
) -> str | None:
    if not isinstance(payload, dict):
        return "artifact is missing or not a JSON object"
    schema_version = payload.get("schema_version")
    if (
        payload.get("contract") != AI_REGIME_SUPERVISION_CONTRACT
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        return "contract or schema_version is invalid"
    stored = str(payload.get("contract_sha256") or "")
    body = {key: item for key, item in payload.items() if key != "contract_sha256"}
    try:
        raw = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return "sealed body is not canonical JSON"
    if len(stored) != 64 or stored != hashlib.sha256(raw).hexdigest():
        return "contract_sha256 does not seal the artifact body"
    generated_at = _strict_aware_utc(payload.get("generated_at_utc"))
    if generated_at is None:
        return "generated_at_utc is missing or invalid"
    if generated_at > now_utc:
        return "generated_at_utc is future-dated"
    last_tuned_at = _strict_aware_utc(payload.get("last_tuned_at_utc"))
    if last_tuned_at is None or last_tuned_at > generated_at:
        return "last_tuned_at_utc is missing, invalid, or later than generation"
    if payload.get("ai_role") != "REGIME_REVIEW_AND_PERIODIC_TUNING_ONLY":
        return "ai_role must be regime review and periodic tuning only"
    if str(payload.get("ai_order_authority") or "").strip().upper() != "NONE":
        return "ai_order_authority must be NONE"
    if payload.get("live_permission") is not False:
        return "live_permission must be false"
    if payload.get("broker_mutation_allowed") is not False:
        return "broker_mutation_allowed must be false"
    if not isinstance(payload.get("pairs"), dict):
        return "pairs must be a JSON object"
    for pair, row in payload["pairs"].items():
        if not str(pair).strip():
            return "pair name must be non-empty"
        if not isinstance(row, dict) or set(row) != AI_REGIME_SUPERVISION_PAIR_KEYS:
            return f"{pair}: pair row must contain exact mode/reason/expires_at_utc keys"
        mode = row.get("mode")
        if not isinstance(mode, str) or mode not in AI_REGIME_SUPERVISION_MODES:
            return f"{pair}: mode must be GO, CAUTION, or STOP"
        reason = row.get("reason")
        if (
            not isinstance(reason, str)
            or reason != reason.strip()
            or not 1 <= len(reason) <= 500
        ):
            return f"{pair}: reason must be non-empty and at most 500 characters"
        expires_at = _strict_aware_utc(row.get("expires_at_utc"))
        if (
            expires_at is None
            or not last_tuned_at < expires_at <= last_tuned_at + timedelta(hours=6)
        ):
            return f"{pair}: expires_at_utc must be aware and within six hours of tuning"
    return None


def _journal_evidence(path: Path) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    last_status = None
    last_line_excerpt = None
    if path.exists():
        for line_number, line in enumerate(_tail_lines(path, max_bytes=256_000), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                parsed = _parse_utc(payload.get("ts"))
                if parsed is not None:
                    accepted.append(
                        _timestamp_candidate(
                            source="trader_journal.ts",
                            path=path,
                            timestamp=parsed,
                            detail=f"jsonl_line={line_number}",
                        )
                    )
                for key in ("timestamp", "generated_at_utc", "time", "created_at_utc"):
                    rejected_ts = _parse_utc(payload.get(key))
                    if rejected_ts is not None:
                        rejected.append(
                            _timestamp_candidate(
                                source=f"trader_journal.{key}",
                                path=path,
                                timestamp=rejected_ts,
                                rejected_reason="trader journal run evidence must use the explicit ts field",
                            )
                        )
                last_status = payload.get("status") or payload.get("gpt_status") or last_status
            last_line_excerpt = stripped[:500]
    mtime = _mtime_utc(path)
    if mtime is not None:
        rejected.append(
            _timestamp_candidate(
                source="trader_journal.file_mtime",
                path=path,
                timestamp=mtime,
                rejected_reason="file mtime is not explicit trader-run evidence",
            )
        )
    latest = _latest_timestamp_candidate(accepted)
    return {
        "path": str(path),
        "exists": path.exists(),
        "last_at": latest.get("timestamp_utc") if latest else None,
        "file_mtime_utc": _iso(mtime),
        "last_status": last_status,
        "last_line_excerpt": last_line_excerpt,
        "accepted_timestamp_candidates": accepted,
        "rejected_timestamp_candidates": rejected,
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
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in artifacts:
        candidate = item.get("timestamp_candidate")
        if isinstance(candidate, dict):
            accepted.append(candidate)
        rejected_items = item.get("rejected_timestamp_candidates")
        if isinstance(rejected_items, list):
            rejected.extend(rejected_item for rejected_item in rejected_items if isinstance(rejected_item, dict))
    latest = _latest_timestamp_candidate(accepted)
    return {
        "last_at": latest.get("timestamp_utc") if latest else None,
        "artifacts": artifacts,
        "accepted_timestamp_candidates": accepted,
        "rejected_timestamp_candidates": rejected,
    }


def _json_artifact_time(path: Path, name: str) -> dict[str, Any]:
    payload = _read_json_object(path)
    parsed = None
    status = None
    action = None
    accepted_candidate = None
    rejected: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        parsed = _parse_utc(payload.get("generated_at_utc"))
        status = payload.get("status")
        action = payload.get("action")
        if parsed is not None:
            if _looks_like_trader_decision_response(payload):
                accepted_candidate = _timestamp_candidate(
                    source=f"{name}.generated_at_utc",
                    path=path,
                    timestamp=parsed,
                )
            else:
                rejected.append(
                    _timestamp_candidate(
                        source=f"{name}.generated_at_utc",
                        path=path,
                        timestamp=parsed,
                        rejected_reason="codex_trader_decision_response is not clearly a trader decision",
                    )
                )
        for key in ("ts", "timestamp", "created_at_utc"):
            rejected_ts = _parse_utc(payload.get(key))
            if rejected_ts is not None:
                rejected.append(
                    _timestamp_candidate(
                        source=f"{name}.{key}",
                        path=path,
                        timestamp=rejected_ts,
                        rejected_reason="decision response run evidence must use generated_at_utc",
                    )
                )
    mtime = _mtime_utc(path)
    if mtime is not None:
        rejected.append(
            _timestamp_candidate(
                source=f"{name}.file_mtime",
                path=path,
                timestamp=mtime,
                rejected_reason="file mtime is not explicit trader-run evidence",
            )
        )
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "generated_at_utc": accepted_candidate.get("timestamp_utc") if accepted_candidate else None,
        "status": status,
        "action": action,
        "timestamp_candidate": accepted_candidate,
        "rejected_timestamp_candidates": rejected,
    }


def _report_artifact_time(path: Path, name: str) -> dict[str, Any]:
    text = _read_text(path, limit=16_000)
    parsed = None
    status = None
    accepted_candidate = None
    rejected: list[dict[str, Any]] = []
    if text:
        match = re.search(r"Generated at UTC:\s*`([^`]+)`", text)
        if match:
            parsed = _parse_utc(match.group(1))
            if parsed is not None:
                if _looks_like_trader_report_text(text, name):
                    accepted_candidate = _timestamp_candidate(
                        source=f"{name}.generated_at_utc",
                        path=path,
                        timestamp=parsed,
                    )
                else:
                    rejected.append(
                        _timestamp_candidate(
                            source=f"{name}.generated_at_utc",
                            path=path,
                            timestamp=parsed,
                            rejected_reason="report is not clearly a qr-trader cycle artifact",
                        )
                    )
        status_match = re.search(r"Status:\s*`([^`]+)`", text)
        if status_match:
            status = status_match.group(1)
    mtime = _mtime_utc(path)
    if mtime is not None:
        rejected.append(
            _timestamp_candidate(
                source=f"{name}.file_mtime",
                path=path,
                timestamp=mtime,
                rejected_reason="file mtime is not explicit trader-run evidence",
            )
        )
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "generated_at_utc": accepted_candidate.get("timestamp_utc") if accepted_candidate else None,
        "status": status,
        "timestamp_candidate": accepted_candidate,
        "rejected_timestamp_candidates": rejected,
    }


def _looks_like_trader_report_text(text: str, name: str) -> bool:
    if name == "gpt_decision_report":
        return bool(re.search(r"^#\s+GPT Trader Decision Report\s*$", text, flags=re.MULTILINE))
    if name == "autotrade_report":
        return bool(re.search(r"^#\s+Autotrade Cycle Report\s*$", text, flags=re.MULTILINE))
    return False


def _memory_evidence(path: Path) -> dict[str, Any]:
    text = _read_tail_text(path, max_bytes=128_000)
    accepted, rejected = _memory_timestamp_candidates(text, path)
    mtime = _mtime_utc(path)
    if mtime is not None:
        rejected.append(
            _timestamp_candidate(
                source="qr_trader_automation_memory.file_mtime",
                path=path,
                timestamp=mtime,
                rejected_reason="automation memory mtime is not an entry timestamp",
            )
        )
    latest = _latest_timestamp_candidate(accepted)
    return {
        "path": str(path),
        "exists": path.exists(),
        "last_at": latest.get("timestamp_utc") if latest else None,
        "file_mtime_utc": _iso(mtime),
        "parsed_timestamp_count": len(accepted)
        + len(
            [
                item
                for item in rejected
                if item.get("source") == "qr_trader_automation_memory.timestamp"
            ]
        ),
        "accepted_timestamp_candidates": accepted,
        "rejected_timestamp_candidates": rejected,
    }


def _memory_timestamp_candidates(text: str, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    in_code_block = False
    lines = text.splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        fence_line = stripped.startswith("```")
        matches = _timestamp_matches_from_text(raw_line)
        if matches:
            rejected_reason = _memory_timestamp_rejected_reason(
                raw_line,
                in_code_block=in_code_block or fence_line,
            )
            if (
                rejected_reason is None
                and not _memory_line_is_run_marker(raw_line)
                and not _memory_section_has_run_marker(lines, line_number - 1)
            ):
                rejected_reason = "automation memory timestamp is not attached to a qr-trader run marker"
            for timestamp in matches:
                if rejected_reason:
                    rejected.append(
                        _timestamp_candidate(
                            source="qr_trader_automation_memory.timestamp",
                            path=path,
                            timestamp=timestamp,
                            detail=_memory_line_detail(line_number, raw_line),
                            rejected_reason=rejected_reason,
                        )
                    )
                else:
                    accepted.append(
                        _timestamp_candidate(
                            source="qr_trader_automation_memory.timestamp",
                            path=path,
                            timestamp=timestamp,
                            detail=_memory_line_detail(line_number, raw_line),
                        )
                    )
        if fence_line:
            in_code_block = not in_code_block
    return accepted, rejected


def _memory_timestamp_rejected_reason(line: str, *, in_code_block: bool) -> str | None:
    stripped = line.strip()
    lower = stripped.lower()
    if in_code_block:
        return "timestamp appears inside a code block"
    if _memory_line_is_json_snippet(stripped):
        return "timestamp appears inside a JSON snippet or quoted JSON block"
    if "expires_at_utc" in lower or (
        "expir" in lower and ("receipt" in lower or "guardian" in lower)
    ):
        return "receipt expiry timestamp is not trader-run evidence"
    if "generated_at_utc" in lower and ("receipt" in lower or "guardian" in lower):
        return "receipt or guardian generated_at timestamp is not trader-run evidence"
    if "guardian action review" in lower or "guardian_action_review" in lower:
        return "guardian action review timestamp is not trader-run evidence"
    if ("receipt_lifecycle" in lower or "receipt lifecycle" in lower) and "receipt" in lower:
        return "guardian receipt lifecycle timestamp is not trader-run evidence"
    if "next_review_deadline_utc" in lower or ("trigger" in lower and "deadline" in lower):
        return "guardian trigger contract deadline timestamp is not trader-run evidence"
    if ("report" in lower or "snippet" in lower) and not _memory_line_is_run_marker(line):
        return "report snippet timestamp is not trader-run evidence"
    if ("issue" in lower or "description" in lower) and not _memory_line_is_run_marker(line):
        return "issue description timestamp is not trader-run evidence"
    return None


def _memory_line_is_run_marker(line: str) -> bool:
    lower = line.strip().lower()
    starts_with_timestamp = _memory_line_starts_with_timestamp(line)
    heading = lower.lstrip("#").strip() != lower
    if starts_with_timestamp and "hourly trader cycle" in lower:
        return True
    if "trader cycle" in lower and (
        "ran one" in lower or "completed" in lower or "attempted" in lower
    ) and (
        "hourly" in lower or "qr-trader" in lower or "qr vnext" in lower
    ):
        return True
    if heading and ("qr-trader" in lower or "hourly trader cycle" in lower) and (
        "run" in lower or "cycle" in lower or "completed" in lower
    ):
        return True
    if "qr-trader" in lower and (
        "run completed" in lower
        or "cycle completed" in lower
        or "automation memory entry" in lower
        or "automation run completed" in lower
        or "completed timestamp" in lower
    ):
        return True
    if "qr-trader" in lower and ("automation run id" in lower or "run_id" in lower) and (
        "complete" in lower or "completed" in lower or "success" in lower
    ):
        return True
    if (
        "latest journal timestamp" in lower
        or "trader_journal.ts" in lower
        or "journal ts" in lower
    ) and ("qr-trader" in lower or "trader run" in lower or "automation memory" in lower):
        return True
    return False


def _memory_section_has_run_marker(lines: list[str], timestamp_index: int) -> bool:
    if timestamp_index < 0 or timestamp_index >= len(lines):
        return False
    if not _memory_line_is_timestamp_heading(lines[timestamp_index]):
        return False
    for raw_line in lines[timestamp_index + 1 :]:
        stripped = raw_line.strip()
        if stripped.startswith("#") and _timestamp_matches_from_text(stripped):
            return False
        if _memory_line_is_run_marker(raw_line):
            return True
    return False


def _memory_line_is_timestamp_heading(line: str) -> bool:
    return line.strip().startswith("#") and _memory_line_starts_with_timestamp(line)


def _memory_line_starts_with_timestamp(line: str) -> bool:
    candidate = line.lstrip(" \t-*#>.")
    return bool(
        re.match(
            r"\d{4}-\d{2}-\d{2}(?:T| )\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2}|\s+JST)",
            candidate,
            flags=re.IGNORECASE,
        )
    )


def _memory_line_is_json_snippet(stripped: str) -> bool:
    text = stripped.lstrip(">")
    text = text.strip()
    if not text:
        return False
    if text.startswith("{") or text.startswith("}") or text.startswith("[") or text.startswith("]"):
        return True
    return bool(re.match(r'"[A-Za-z0-9_:-]+"\s*:', text))


def _memory_line_detail(line_number: int, line: str) -> str:
    compact = " ".join(line.strip().split())
    return f"line={line_number} text={compact[:220]}"


def _timestamp_candidate(
    *,
    source: str,
    path: Path,
    timestamp: datetime,
    detail: str | None = None,
    rejected_reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": source,
        "path": str(path),
        "timestamp_utc": timestamp.astimezone(timezone.utc).isoformat(),
    }
    if detail:
        payload["detail"] = detail
    if rejected_reason:
        payload["rejected_reason"] = rejected_reason
    return payload


def _latest_timestamp_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for candidate in candidates:
        timestamp = _parse_utc(candidate.get("timestamp_utc"))
        if timestamp is not None:
            parsed.append((timestamp, candidate))
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def _looks_like_trader_decision_response(payload: dict[str, Any]) -> bool:
    action_value = payload.get("action")
    decision = payload.get("decision")
    if not action_value and isinstance(decision, dict):
        action_value = decision.get("action")
    action = str(action_value or "").upper()
    if action not in {"TRADE", "WAIT", "CANCEL_PENDING", "PROTECT", "TIGHTEN_SL", "CLOSE", "REQUEST_EVIDENCE"}:
        return False
    trader_markers = {
        "market_read_first",
        "twenty_minute_plan",
        "selected_lane_id",
        "selected_lane_ids",
        "rejected_alternatives",
        "risk_notes",
        "operator_summary",
        "close_trade_ids",
        "cancel_order_ids",
        "confidence",
        "method",
    }
    if any(key in payload for key in trader_markers):
        return True
    decision = payload.get("decision")
    return isinstance(decision, dict) and any(key in decision for key in trader_markers)


def _non_trader_timestamp_candidates(paths: WatchdogPaths) -> list[dict[str, Any]]:
    rejected: list[dict[str, Any]] = []
    rejected.extend(
        _json_timestamp_rejections(
            paths.guardian_receipt,
            keys=("generated_at_utc", "expires_at_utc"),
            source_prefix="guardian_action_receipt",
            reason="guardian receipt timestamps are not trader-run evidence",
        )
    )
    rejected.extend(
        _json_timestamp_rejections(
            paths.guardian_trigger_contract,
            keys=("generated_at_utc", "next_review_deadline_utc"),
            source_prefix="guardian_trigger_contract",
            reason="guardian trigger contract timestamps are not trader-run evidence",
        )
    )
    review_text = _read_text(paths.guardian_review, limit=16_000)
    for timestamp in _timestamps_from_text(review_text):
        rejected.append(
            _timestamp_candidate(
                source="guardian_action_review.timestamp",
                path=paths.guardian_review,
                timestamp=timestamp,
                rejected_reason="guardian action review timestamps are not trader-run evidence",
            )
        )
    rejected.extend(
        _json_timestamp_rejections(
            paths.output_json,
            keys=("generated_at_utc",),
            source_prefix="qr_trader_run_watchdog",
            reason="watchdog generated_at is not trader-run evidence",
        )
    )
    return rejected


def _json_timestamp_rejections(
    path: Path,
    *,
    keys: tuple[str, ...],
    source_prefix: str,
    reason: str,
) -> list[dict[str, Any]]:
    payload = _read_json_object(path)
    if not isinstance(payload, dict):
        return []
    rejected: list[dict[str, Any]] = []
    for key in keys:
        parsed = _parse_utc(payload.get(key))
        if parsed is not None:
            rejected.append(
                _timestamp_candidate(
                    source=f"{source_prefix}.{key}",
                    path=path,
                    timestamp=parsed,
                    rejected_reason=reason,
                )
            )
    for nested_key in ("receipt", "selected_event"):
        nested = payload.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in keys:
            parsed = _parse_utc(nested.get(key))
            if parsed is not None:
                rejected.append(
                    _timestamp_candidate(
                        source=f"{source_prefix}.{nested_key}.{key}",
                        path=path,
                        timestamp=parsed,
                        rejected_reason=reason,
                    )
                )
    return rejected


def _guardian_receipt_status(
    receipt_path: Path,
    review_path: Path,
    trigger_contract_path: Path,
    *,
    receipt_consumption: dict[str, Any],
    last_trader_run_at: str | None,
    expected_cadence_minutes: int,
    grace_minutes: int,
    now_utc: datetime,
) -> dict[str, Any]:
    review_text = _read_text(review_path, limit=16_000)
    _ = trigger_contract_path
    candidates = _guardian_receipt_candidates(receipt_path)
    groups = _group_guardian_receipts(candidates)
    issues: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for group in groups:
        summary = _guardian_receipt_group_summary(
            group,
            last_trader_run_at=last_trader_run_at,
            expected_cadence_minutes=expected_cadence_minutes,
            grace_minutes=grace_minutes,
            now_utc=now_utc,
        )
        acknowledgement = acknowledgement_for_receipt(
            receipt_consumption,
            issue_code="GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER",
            receipt_event_id=summary.get("event_id"),
            receipt_action=summary.get("action"),
            receipt_lifecycle=summary.get("receipt_lifecycle"),
        )
        summary["trader_acknowledgement"] = acknowledgement
        summary["acknowledged_by_trader"] = acknowledgement is not None
        summary["acknowledgement_classification"] = (
            acknowledgement.get("classification") if isinstance(acknowledgement, dict) else None
        )
        summary["normal_routing_allowed_by_acknowledgement"] = (
            acknowledgement.get("normal_routing_allowed") if isinstance(acknowledgement, dict) else None
        )
        summaries.append(summary)
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
            "guardian_receipt_consumption": receipt_consumption,
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
        "event_type": current_summary["event_type"],
        "event_severity": current_summary["event_severity"],
        "event_action_hint": current_summary["event_action_hint"],
        "event_details": current_summary["event_details"],
        "issues": issues,
        "receipts_checked": len(candidates),
        "archive_path": str(archive_dir),
        "archive_exists": archive_dir.exists(),
        "archive_receipts_checked": len([item for item in candidates if item["source"] == "archive"]),
        "receipt_summaries": summaries,
        "guardian_receipt_consumption": receipt_consumption,
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
    event_id = _guardian_receipt_event_id(representative)
    dedupe_key = _guardian_receipt_dedupe_key(representative)
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
    event_types = {_guardian_receipt_event_type(payload) for payload in payloads}
    event_severities = {
        _guardian_receipt_event_severity(payload) for payload in payloads
    }
    event_action_hints = {
        _guardian_receipt_event_action_hint(payload) for payload in payloads
    }
    event_type = (
        next(iter(event_types))
        if len(event_types) == 1 and "" not in event_types
        else None
    )
    event_severity = (
        next(iter(event_severities))
        if len(event_severities) == 1 and "" not in event_severities
        else None
    )
    event_action_hint = (
        next(iter(event_action_hints))
        if len(event_action_hints) == 1 and "" not in event_action_hints
        else None
    )
    event_details = _guardian_receipt_event_details(representative)
    if any(
        _guardian_receipt_event_details(payload) != event_details
        for payload in payloads
    ):
        event_details = {}
    return {
        "identity": _guardian_receipt_identity(representative),
        "event_id": event_id,
        "dedupe_key": dedupe_key,
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
        "event_type": event_type,
        "event_severity": event_severity,
        "event_action_hint": event_action_hint,
        "event_details": event_details,
    }


def _guardian_receipt_issue(summary: dict[str, Any]) -> dict[str, Any] | None:
    if summary.get("receipt_status") != "ACCEPTED" or summary.get("consumed_by_trader"):
        return None
    if not summary.get("expired_before_trader_run") and not summary.get("next_run_window_missed"):
        return None
    acknowledgement = summary.get("trader_acknowledgement")
    if isinstance(acknowledgement, dict):
        classification = str(acknowledgement.get("classification") or "").upper()
        normal_allowed = acknowledgement.get("normal_routing_allowed") is True
        if normal_allowed:
            return None
        if classification == NEEDS_OPERATOR_REVIEW:
            return _issue(
                OPERATOR_REVIEW_ISSUE_CODE,
                _guardian_receipt_issue_severity(summary),
                str(acknowledgement.get("reason") or "guardian receipt needs operator review"),
                receipt_identity=summary.get("identity"),
                receipt_event_id=summary.get("event_id"),
                receipt_action=summary.get("action"),
                receipt_lifecycle=summary.get("receipt_lifecycle"),
                receipt_sources=summary.get("sources"),
                receipt_source_paths=summary.get("source_paths"),
                consumed_by_trader=summary.get("consumed_by_trader"),
                expired_before_trader_run=summary.get("expired_before_trader_run"),
                next_run_window_missed=summary.get("next_run_window_missed"),
                emergency_or_margin_risk=summary.get("emergency_or_margin_risk"),
                event_type=summary.get("event_type"),
                event_severity=summary.get("event_severity"),
                event_action_hint=summary.get("event_action_hint"),
                event_details=summary.get("event_details"),
                acknowledgement_classification=classification,
                normal_routing_allowed=False,
            )
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
        receipt_identity=summary.get("identity"),
        receipt_event_id=summary.get("event_id"),
        receipt_action=summary.get("action"),
        receipt_lifecycle=summary.get("receipt_lifecycle"),
        receipt_sources=summary.get("sources"),
        receipt_source_paths=summary.get("source_paths"),
        consumed_by_trader=summary.get("consumed_by_trader"),
        expired_before_trader_run=summary.get("expired_before_trader_run"),
        next_run_window_missed=summary.get("next_run_window_missed"),
        emergency_or_margin_risk=summary.get("emergency_or_margin_risk"),
        event_type=summary.get("event_type"),
        event_severity=summary.get("event_severity"),
        event_action_hint=summary.get("event_action_hint"),
        event_details=summary.get("event_details"),
        normal_routing_allowed=False,
    )


def _guardian_receipt_issue_severity(summary: dict[str, Any]) -> str:
    event_type = str(summary.get("event_type") or "").upper()
    event_severity = str(summary.get("event_severity") or "").upper()
    if event_type == "MARGIN_PRESSURE":
        # Preserve the Guardian's source severity.  A P1 margin warning is not
        # a P0 emergency; missing/unknown source severity remains fail-closed.
        if event_severity == "P1":
            return "P1"
        return "P0"
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


def _guardian_receipt_event_id(receipt: dict[str, Any]) -> str:
    nested = _nested_receipt(receipt)
    selected = _selected_event(receipt)
    return str(nested.get("event_id") or receipt.get("event_id") or selected.get("event_id") or "")


def _guardian_receipt_dedupe_key(receipt: dict[str, Any]) -> str:
    nested = _nested_receipt(receipt)
    selected = _selected_event(receipt)
    return str(nested.get("dedupe_key") or receipt.get("dedupe_key") or selected.get("dedupe_key") or "")


def _guardian_receipt_event_type(receipt: dict[str, Any]) -> str:
    selected = _selected_event(receipt)
    return str(selected.get("event_type") or receipt.get("event_type") or "").upper()


def _guardian_receipt_event_severity(receipt: dict[str, Any]) -> str:
    selected = _selected_event(receipt)
    return str(selected.get("severity") or receipt.get("event_severity") or "").upper()


def _guardian_receipt_event_action_hint(receipt: dict[str, Any]) -> str:
    selected = _selected_event(receipt)
    return str(
        selected.get("action_hint") or receipt.get("event_action_hint") or ""
    ).upper()


def _guardian_receipt_event_details(receipt: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_event(receipt)
    details = selected.get("details")
    if not isinstance(details, dict):
        details = receipt.get("event_details")
    return dict(details) if isinstance(details, dict) else {}


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
        return "qr-trader automation config does not match the active six-hour gpt-5.5 high supervisor policy."
    if status == "UNKNOWN":
        return "Run evidence is insufficient; automation config exists but no journal, decision, or memory timestamp was usable."
    if guardian.get("issues"):
        return "Guardian receipt consumption issues require the next trader cycle to resolve or classify them before ordinary entries."
    if status == "STALE":
        if hints:
            return str(hints[0])
        scheduler_seen = codex_logs.get("scheduler_last_run_at") or codex_logs.get("latest_live_cwd_log_at")
        if scheduler_seen:
            return (
                "Codex logs show a recent qr-trader/live-root session, but trader artifacts are stale; "
                "inspect the latest Codex thread for preflight, quota, or early-exit failure."
            )
        return "No explicit Codex log cause was visible; latest supervisor artifacts missed the expected six-hour window."
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
            "command": "open Codex automation qr-trader settings and restore the ACTIVE gpt-5.5 high six-hour supervisor policy",
            "reason": "; ".join(item["code"] for item in config_issues),
        }
    if status == "STALE":
        return {
            "code": "INSPECT_CODEX_QR_TRADER_THREAD",
            "requires_explicit_operator_approval": False,
            "command": "review latest QR vNext Trader Codex thread/logs; do not run live trader manually from the watchdog",
            "reason": "scheduled-run evidence missed the six-hour cadence plus grace",
        }
    if guardian.get("issues"):
        return {
            "code": "RESOLVE_GUARDIAN_RECEIPT_IN_NEXT_TRADER_CYCLE",
            "requires_explicit_operator_approval": False,
            "command": "read data/guardian_action_receipt.json and docs/guardian_action_review.md before normal entries",
            "reason": "guardian receipt is unconsumed or expired without trader resolution",
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


def _issue(code: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"code": code, "severity": severity, "message": message}
    payload.update(extra)
    return payload


def _render_report(payload: dict[str, Any]) -> str:
    automation = payload["automation_config"]
    weekend = payload.get("weekend_pause") or {}
    evidence = payload["latest_run_evidence"]
    guardian = payload["guardian_receipt"]
    codex = payload["codex_logs"]
    lines = [
        "# QR Trader Run Watchdog Report",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}` runtime_status=`{payload.get('runtime_status')}` "
        f"overall_status=`{payload.get('overall_status')}` issue_status=`{payload.get('issue_status')}`",
        f"- Severity: `{payload['severity']}`",
        f"- Service health: `{'healthy' if payload.get('runtime_status') == 'OK' else 'attention_required'}`",
        f"- Trading workflow: `{'blocked_by_guardian_or_run_issue' if payload.get('overall_status') == 'BLOCKED' else 'available'}`",
        f"- Missed expected window: `{payload['missed_expected_window']}`",
        f"- Last trader run evidence: `{payload['last_trader_run_at']}`",
        f"- Last trader run source: `{payload.get('last_trader_run_source')}`",
        f"- Last trader run path: `{payload.get('last_trader_run_path')}`",
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
        f"- Weekend pause active: `{weekend.get('active')}` mode=`{weekend.get('mode')}` "
        f"window=`{weekend.get('in_weekend_pause_window')}`",
        f"- Weekend state: `{weekend.get('path')}` qr_trader_managed=`{weekend.get('qr_trader_managed')}`",
        "",
        "## Latest Run Evidence",
        "",
        f"- Evidence priority: `{evidence.get('evidence_priority')}`",
        f"- AI regime supervision: `{payload.get('last_ai_regime_supervision_at')}` "
        f"valid_sealed=`{evidence.get('ai_regime_supervision', {}).get('valid_sealed_artifact')}` "
        f"path=`{evidence.get('ai_regime_supervision', {}).get('path')}`",
        f"- Journal: `{payload['last_journal_at']}` path=`{evidence['journal']['path']}`",
        f"- Decision artifacts: `{payload['last_decision_artifact_at']}`",
        f"- Automation memory: `{payload['last_memory_at']}` path=`{evidence['memory']['path']}`",
        f"- Rejected timestamp candidates: `{len(payload.get('rejected_timestamp_candidates') or [])}`",
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
    return _timestamp_matches_from_text(text)


def _timestamp_matches_from_text(text: str) -> list[datetime]:
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


def _strict_aware_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


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
    parser.add_argument("--weekend-state", type=Path, default=DEFAULT_WEEKEND_STATE)
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
        weekend_state=args.weekend_state,
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
                "last_trader_run_source": payload["last_trader_run_source"],
                "last_trader_run_path": payload["last_trader_run_path"],
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
