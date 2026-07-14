from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.guardian_margin_contract import (
    MARGIN_PRESSURE_WARNING_CAP_FRACTION,
    P1_MARGIN_WARNING_CONTRACT,
    exact_p1_margin_warning_source_event,
    strict_number as _strict_number,
)

from quant_rabbit.guardian_receipt_operator_review import (
    OPERATOR_REVIEW_CLEARS_RECEIPT,
    OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT,
    OPERATOR_REVIEW_REQUIRED_BLOCK_CODE,
    load_guardian_receipt_operator_review,
    operator_review_clearance_status,
    operator_review_durable_clearance_status,
    operator_review_subjects_from_artifacts,
    receipt_requires_operator_review,
)


ISSUE_CODE = "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER"
OPERATOR_REVIEW_ISSUE_CODE = "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW"
BLOCK_NEW_ENTRY_CODE = "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER_BLOCKS_NEW_ENTRY"
WATCHDOG_BLOCK_NEW_ENTRY_CODE = "QR_TRADER_RUN_WATCHDOG_BLOCKS_NEW_ENTRY"
P1_MARGIN_WARNING_OBSERVED_CODE = "GUARDIAN_P1_MARGIN_PRESSURE_OBSERVED"

CONSUMED = "CONSUMED"
EXPIRED_ACKNOWLEDGED = "EXPIRED_ACKNOWLEDGED"
STALE_ACKNOWLEDGED = "STALE_ACKNOWLEDGED"
REJECTED_ACKNOWLEDGED = "REJECTED_ACKNOWLEDGED"
HISTORICAL_ONLY = "HISTORICAL_ONLY"
NEEDS_OPERATOR_REVIEW = "NEEDS_OPERATOR_REVIEW"

ACKNOWLEDGED_CLASSIFICATIONS = {
    CONSUMED,
    EXPIRED_ACKNOWLEDGED,
    STALE_ACKNOWLEDGED,
    REJECTED_ACKNOWLEDGED,
    HISTORICAL_ONLY,
}
CLASSIFICATIONS = ACKNOWLEDGED_CLASSIFICATIONS | {NEEDS_OPERATOR_REVIEW}
PRESERVABLE_REVIEW_LIFECYCLES = {"EXPIRED", "STALE", "REJECTED", "HISTORICAL_ONLY"}
DURABLE_OPERATOR_REVIEW_CLEARANCE_STATUSES = {
    OPERATOR_REVIEW_CLEARS_RECEIPT,
    OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT,
}


def load_guardian_receipt_consumption(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"_missing": True, "_path": str(path), "_unreadable": True}
    if not isinstance(payload, dict):
        return {"_missing": True, "_path": str(path), "_unreadable": True}
    payload.setdefault("_path", str(path))
    return payload


def build_guardian_receipt_consumption(
    watchdog_payload: dict[str, Any] | None,
    *,
    now_utc: datetime | None = None,
    existing: dict[str, Any] | None = None,
    operator_review: dict[str, Any] | None = None,
    broker_snapshot: dict[str, Any] | None = None,
    generated_by: str = "deterministic-preflight",
) -> dict[str, Any]:
    watchdog = watchdog_payload if isinstance(watchdog_payload, dict) else {}
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    existing_rows = _classification_rows(existing)
    existing_by_key = {
        _classification_key(item): item
        for item in existing_rows
        if _has_receipt_key(_classification_key(item))
    }
    issues = _watchdog_guardian_issues(watchdog)
    seen_issue_keys = {_issue_key(issue) for issue in issues}
    for issue in _preserved_existing_issues(existing_rows, seen_issue_keys):
        key = _issue_key(issue)
        issues.append(issue)
        seen_issue_keys.add(key)
    rows: list[dict[str, Any]] = []
    for issue in issues:
        row = _classification_from_issue(
            issue,
            existing_by_key.get(_issue_key(issue)),
            watchdog_payload=watchdog,
            operator_review=operator_review,
            broker_snapshot=broker_snapshot,
            now_utc=now,
            generated_at_utc=now.isoformat(),
            generated_by=generated_by,
        )
        rows.append(row)
    rows_allowed = all(item.get("normal_routing_allowed") is True for item in rows)
    current_p0_p1_blocks = _watchdog_has_uncleared_p0_p1_issue(watchdog, rows)
    normal_allowed = rows_allowed and not current_p0_p1_blocks
    if not rows:
        status = "NO_GUARDIAN_RECEIPT_ISSUES"
        normal_allowed = not current_p0_p1_blocks
        if current_p0_p1_blocks:
            status = "NO_GUARDIAN_RECEIPT_ISSUES_CURRENT_P0_BLOCKS_ROUTING"
    elif rows_allowed and current_p0_p1_blocks:
        status = "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED_CURRENT_P0_BLOCKS_ROUTING"
    elif rows_allowed:
        status = "GUARDIAN_RECEIPT_ISSUES_ACKNOWLEDGED"
    else:
        status = "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
    return {
        "generated_at_utc": now.isoformat(),
        "status": status,
        "normal_routing_allowed": normal_allowed,
        "classifications": rows,
        "unresolved_issue_count": sum(1 for item in rows if item.get("normal_routing_allowed") is not True),
        "current_p0_p1_blocks_routing": current_p0_p1_blocks,
        "watchdog_generated_at_utc": watchdog.get("generated_at_utc"),
        "watchdog_status": watchdog.get("status"),
        "watchdog_path": (
            (watchdog.get("artifact_paths") or {}).get("output_json")
            if isinstance(watchdog.get("artifact_paths"), dict)
            else None
        ),
        "read_only": True,
        "live_side_effects": [],
    }


def write_guardian_receipt_consumption(
    payload: dict[str, Any],
    *,
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_guardian_receipt_consumption_report(payload), encoding="utf-8")


def render_guardian_receipt_consumption_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Guardian Receipt Consumption Report",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Normal routing allowed: `{payload.get('normal_routing_allowed')}`",
        f"- Read only: `{payload.get('read_only')}`",
        f"- Live side effects: `{len(payload.get('live_side_effects') or [])}`",
        "",
        "## Classifications",
        "",
    ]
    rows = _classification_rows(payload)
    if not rows:
        lines.append("- none")
    else:
        lines.extend(
            [
                "| Issue | Event | Action | Lifecycle | Consumed | Classification | Normal Routing | Generated By | Row Generated UTC | Reason |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in rows:
            reason = str(row.get("reason") or "").replace("|", "\\|")
            lines.append(
                f"| `{row.get('issue_code')}` | `{row.get('receipt_event_id')}` | "
                f"`{row.get('receipt_action')}` | `{row.get('receipt_lifecycle')}` | "
                f"`{row.get('consumed_by_trader')}` | `{row.get('classification')}` | "
                f"`{row.get('normal_routing_allowed')}` | `{row.get('generated_by')}` | "
                f"`{row.get('generated_at_utc')}` | {reason} |"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This artifact is a trader classification record only.",
            "- It does not consume the guardian receipt inside guardian-action-cycle.",
            "- It does not place orders, cancel orders, close positions, or change broker state.",
            "",
        ]
    )
    return "\n".join(lines)


def acknowledgement_for_receipt(
    consumption_payload: dict[str, Any],
    *,
    issue_code: str,
    receipt_event_id: str | None,
    receipt_action: str | None,
    receipt_lifecycle: str | None,
) -> dict[str, Any] | None:
    key = _key(issue_code, receipt_event_id, receipt_action, receipt_lifecycle)
    for row in _classification_rows(consumption_payload):
        if _classification_key(row) == key:
            return row
    return None


def consumption_status_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("_missing"):
        return {
            "status": "UNAVAILABLE",
            "normal_routing_allowed": None,
            "classifications": [],
            "unresolved_issue_count": 0,
            "missing": True,
            "path": payload.get("_path") if isinstance(payload, dict) else None,
        }
    rows = _classification_rows(payload)
    return {
        "status": payload.get("status"),
        "normal_routing_allowed": payload.get("normal_routing_allowed"),
        "classifications": rows,
        "unresolved_issue_count": payload.get("unresolved_issue_count"),
        "missing": False,
        "generated_at_utc": payload.get("generated_at_utc"),
        "watchdog_generated_at_utc": payload.get("watchdog_generated_at_utc"),
    }


def guardian_receipt_new_entry_blockers_from_paths(
    *,
    watchdog_path: Path | None = None,
    consumption_path: Path | None = None,
    operator_review_path: Path | None = None,
    broker_snapshot_path: Path | None = None,
    allow_p1_margin_warning: bool = False,
    current_margin_utilization_pct: float | None = None,
    projected_margin_utilization_pct: float | None = None,
    max_margin_utilization_pct: float | None = None,
    margin_available_jpy: float | None = None,
    broker_snapshot_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    watchdog = _read_json_path(_env_path("QR_GUARDIAN_RECEIPT_WATCHDOG_PATH", watchdog_path))
    consumption = load_guardian_receipt_consumption(
        _env_path("QR_GUARDIAN_RECEIPT_CONSUMPTION_PATH", consumption_path)
    )
    operator_review = load_guardian_receipt_operator_review(
        _env_path("QR_GUARDIAN_RECEIPT_OPERATOR_REVIEW_PATH", operator_review_path)
    )
    broker_snapshot = (
        broker_snapshot_payload
        if isinstance(broker_snapshot_payload, dict)
        else _read_json_path(
            _env_path(
                "QR_GUARDIAN_RECEIPT_BROKER_SNAPSHOT_PATH",
                broker_snapshot_path,
            )
        )
    )
    return guardian_receipt_new_entry_blockers(
        watchdog,
        consumption,
        operator_review,
        broker_snapshot,
        allow_p1_margin_warning=allow_p1_margin_warning,
        current_margin_utilization_pct=current_margin_utilization_pct,
        projected_margin_utilization_pct=projected_margin_utilization_pct,
        max_margin_utilization_pct=max_margin_utilization_pct,
        margin_available_jpy=margin_available_jpy,
    )


def guardian_receipt_new_entry_blockers(
    watchdog_payload: dict[str, Any] | None,
    consumption_payload: dict[str, Any] | None,
    operator_review_payload: dict[str, Any] | None = None,
    broker_snapshot_payload: dict[str, Any] | None = None,
    *,
    allow_p1_margin_warning: bool = False,
    current_margin_utilization_pct: float | None = None,
    projected_margin_utilization_pct: float | None = None,
    max_margin_utilization_pct: float | None = None,
    margin_available_jpy: float | None = None,
) -> list[dict[str, Any]]:
    watchdog = watchdog_payload if isinstance(watchdog_payload, dict) else {}
    consumption = consumption_payload if isinstance(consumption_payload, dict) else {}
    operator_review = operator_review_payload if isinstance(operator_review_payload, dict) else {}
    broker_snapshot = broker_snapshot_payload if isinstance(broker_snapshot_payload, dict) else {}
    blockers: list[dict[str, str]] = []
    for issue in operator_review_subjects_from_artifacts(watchdog, consumption):
        acknowledgement = acknowledgement_for_receipt(
            consumption,
            issue_code=_issue_code(issue),
            receipt_event_id=_issue_event_id(issue),
            receipt_action=_issue_action(issue),
            receipt_lifecycle=_issue_lifecycle(issue),
        )
        if _acknowledgement_allows_review_required_routing(acknowledgement):
            continue
        status = operator_review_clearance_status(
            issue,
            operator_review,
            watchdog_payload=watchdog,
            broker_snapshot_payload=broker_snapshot,
        )
        if status.get("normal_routing_allowed") is True:
            continue
        blockers.append(
            {
                "code": OPERATOR_REVIEW_REQUIRED_BLOCK_CODE,
                "severity": "BLOCK",
                "message": (
                    f"operator review required for receipt event={_issue_event_id(issue) or 'UNKNOWN'} "
                    f"action={_issue_action(issue) or 'UNKNOWN'} lifecycle={_issue_lifecycle(issue) or 'UNKNOWN'} "
                    f"status={status.get('status')}: {status.get('reason')}"
                ),
                "receipt_event_id": _issue_event_id(issue),
                "receipt_action": _issue_action(issue),
                "receipt_lifecycle": _issue_lifecycle(issue),
                "classification": NEEDS_OPERATOR_REVIEW,
            }
        )

    issues = [
        issue
        for issue in _watchdog_guardian_issues(watchdog)
        if _issue_code(issue) in {ISSUE_CODE, OPERATOR_REVIEW_ISSUE_CODE}
        and not receipt_requires_operator_review(issue)
    ]
    for issue in issues:
        acknowledgement = acknowledgement_for_receipt(
            consumption,
            issue_code=_issue_code(issue),
            receipt_event_id=_issue_event_id(issue),
            receipt_action=_issue_action(issue),
            receipt_lifecycle=_issue_lifecycle(issue),
        )
        if isinstance(acknowledgement, dict) and acknowledgement.get("normal_routing_allowed") is True:
            continue
        if isinstance(acknowledgement, dict):
            classification = str(acknowledgement.get("classification") or "UNKNOWN")
            reason = str(acknowledgement.get("reason") or "classification does not allow normal routing")
        else:
            classification = "UNCLASSIFIED"
            reason = "no matching guardian_receipt_consumption classification allows normal routing"
        blockers.append(
            {
                "code": BLOCK_NEW_ENTRY_CODE,
                "severity": "BLOCK",
                "message": (
                    f"{_issue_code(issue)} event={_issue_event_id(issue) or 'UNKNOWN'} "
                    f"action={_issue_action(issue) or 'UNKNOWN'} lifecycle={_issue_lifecycle(issue) or 'UNKNOWN'} "
                    f"classification={classification}: {reason}"
                ),
                "receipt_event_id": _issue_event_id(issue),
                "receipt_action": _issue_action(issue),
                "receipt_lifecycle": _issue_lifecycle(issue),
                "classification": classification,
            }
        )
    watchdog_p0_p1_blocks = _watchdog_has_uncleared_p0_p1_issue(watchdog, _classification_rows(consumption))
    if consumption.get("normal_routing_allowed") is False and not blockers:
        if consumption.get("current_p0_p1_blocks_routing") is True or watchdog_p0_p1_blocks:
            blockers.append(
                {
                    "code": WATCHDOG_BLOCK_NEW_ENTRY_CODE,
                    "severity": "BLOCK",
                    "message": (
                        f"guardian_receipt_consumption status={consumption.get('status')} "
                        "has normal_routing_allowed=false because a current P0/P1 watchdog issue blocks routing"
                    ),
                    "receipt_event_id": "",
                    "receipt_action": "",
                    "receipt_lifecycle": "",
                    "classification": "WATCHDOG_P0_P1",
                }
            )
        else:
            blockers.append(
                {
                    "code": OPERATOR_REVIEW_REQUIRED_BLOCK_CODE,
                    "severity": "BLOCK",
                    "message": (
                        f"guardian_receipt_consumption status={consumption.get('status')} "
                        "has normal_routing_allowed=false"
                    ),
                    "receipt_event_id": "",
                    "receipt_action": "",
                    "receipt_lifecycle": "",
                    "classification": str(
                        consumption.get("status") or "NORMAL_ROUTING_FALSE"
                    ),
                }
            )
    elif watchdog_p0_p1_blocks and not blockers:
        blockers.append(
            {
                "code": WATCHDOG_BLOCK_NEW_ENTRY_CODE,
                "severity": "BLOCK",
                "message": (
                    "qr_trader_run_watchdog issue_status/severity is P0/P1; "
                    "ordinary fresh-entry routing remains blocked until watchdog clears"
                ),
                "receipt_event_id": "",
                "receipt_action": "",
                "receipt_lifecycle": "",
                "classification": "WATCHDOG_P0_P1",
            }
        )
    if not allow_p1_margin_warning:
        return blockers
    return _observe_p1_margin_warning_without_relaxing_cap(
        blockers,
        watchdog=watchdog,
        broker_snapshot=broker_snapshot,
        current_margin_utilization_pct=current_margin_utilization_pct,
        projected_margin_utilization_pct=projected_margin_utilization_pct,
        max_margin_utilization_pct=max_margin_utilization_pct,
        margin_available_jpy=margin_available_jpy,
    )


def _observe_p1_margin_warning_without_relaxing_cap(
    blockers: list[dict[str, Any]],
    *,
    watchdog: dict[str, Any],
    broker_snapshot: dict[str, Any],
    current_margin_utilization_pct: float | None,
    projected_margin_utilization_pct: float | None,
    max_margin_utilization_pct: float | None,
    margin_available_jpy: float | None,
) -> list[dict[str, Any]]:
    """Turn only an exact Guardian P1 margin warning into visible WARN evidence.

    This is deliberately narrower than "ignore P1".  The watchdog must carry
    the selected Guardian event type, original event severity, and event
    details; every current P0/P1 issue must be the same P1 MARGIN_PRESSURE
    class.  Current broker truth and the candidate's final projected
    utilization must also remain strictly below the normal hard cap.  Missing
    semantic or numeric evidence leaves the original BLOCK untouched.
    """

    eligibility = _p1_margin_warning_eligibility(
        watchdog,
        broker_snapshot=broker_snapshot,
        current_margin_utilization_pct=current_margin_utilization_pct,
        projected_margin_utilization_pct=projected_margin_utilization_pct,
        max_margin_utilization_pct=max_margin_utilization_pct,
        margin_available_jpy=margin_available_jpy,
    )
    if eligibility.get("status") != "ELIGIBLE_WARNING_ONLY":
        return blockers

    event_ids = set(eligibility.get("event_ids") or [])
    kept: list[dict[str, Any]] = []
    observed_blocker_codes: list[str] = []
    for blocker in blockers:
        code = str(blocker.get("code") or "")
        event_id = str(blocker.get("receipt_event_id") or "")
        action = str(blocker.get("receipt_action") or "").upper()
        generic_watchdog_block = code == WATCHDOG_BLOCK_NEW_ENTRY_CODE and not event_id
        same_margin_receipt = (
            code
            in {
                BLOCK_NEW_ENTRY_CODE,
                OPERATOR_REVIEW_REQUIRED_BLOCK_CODE,
            }
            and event_id in event_ids
            and action in {"HOLD", "NO_ACTION"}
        )
        if generic_watchdog_block or same_margin_receipt:
            observed_blocker_codes.append(code)
            continue
        kept.append(blocker)

    if not observed_blocker_codes:
        return blockers
    kept.append(
        {
            "code": P1_MARGIN_WARNING_OBSERVED_CODE,
            "severity": "WARN",
            "message": (
                "Guardian P1 MARGIN_PRESSURE is observation-only for this candidate: "
                f"current={eligibility['current_margin_utilization_pct']:.6f}% "
                f"projected={eligibility['projected_margin_utilization_pct']:.6f}% "
                f"hard_cap={eligibility['max_margin_utilization_pct']:.6f}%; "
                "RiskEngine and the final gateway must still enforce the hard cap"
            ),
            "contract": P1_MARGIN_WARNING_CONTRACT,
            "classification": "P1_MARGIN_PRESSURE_WARNING_ONLY",
            "receipt_event_id": ",".join(sorted(event_ids)),
            "receipt_action": "HOLD",
            "receipt_lifecycle": "CURRENT_P1_MARGIN_WARNING",
            "current_margin_utilization_pct": eligibility[
                "current_margin_utilization_pct"
            ],
            "projected_margin_utilization_pct": eligibility[
                "projected_margin_utilization_pct"
            ],
            "max_margin_utilization_pct": eligibility[
                "max_margin_utilization_pct"
            ],
            "margin_available_jpy": eligibility["margin_available_jpy"],
            "observed_blocker_codes": sorted(set(observed_blocker_codes)),
        }
    )
    return kept


def _p1_margin_warning_eligibility(
    watchdog: dict[str, Any],
    *,
    broker_snapshot: dict[str, Any],
    current_margin_utilization_pct: float | None,
    projected_margin_utilization_pct: float | None,
    max_margin_utilization_pct: float | None,
    margin_available_jpy: float | None,
) -> dict[str, Any]:
    current, available = _current_margin_values(
        broker_snapshot,
        current_margin_utilization_pct=current_margin_utilization_pct,
        margin_available_jpy=margin_available_jpy,
    )
    projected = _strict_number(projected_margin_utilization_pct)
    cap = _strict_number(max_margin_utilization_pct)
    evidence = {
        "contract": P1_MARGIN_WARNING_CONTRACT,
        "status": "NOT_ELIGIBLE",
        "current_margin_utilization_pct": current,
        "projected_margin_utilization_pct": projected,
        "max_margin_utilization_pct": cap,
        "margin_available_jpy": available,
        "event_ids": [],
    }
    if (
        current is None
        or projected is None
        or cap is None
        or available is None
        or cap <= 0.0
        or cap > 100.0
        or available <= 0.0
        or current < cap * MARGIN_PRESSURE_WARNING_CAP_FRACTION
        or current >= cap
        or projected > cap
    ):
        return evidence

    top_issue_status = str(watchdog.get("issue_status") or "").upper()
    top_severity = str(watchdog.get("severity") or "").upper()
    runtime_status = str(watchdog.get("runtime_status") or "").upper()
    if (
        runtime_status != "OK"
        or top_issue_status != "P1"
        or top_severity != "P1"
    ):
        return evidence

    severe_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for issue in [*_watchdog_issues(watchdog), *_watchdog_guardian_issues(watchdog)]:
        severity = str(issue.get("severity") or "").upper()
        if severity not in {"P0", "P1"}:
            continue
        key = (
            _issue_code(issue),
            _issue_event_id(issue),
            str(issue.get("event_type") or "").upper(),
            f"{severity}:{str(issue.get('event_action_hint') or '').upper()}",
        )
        severe_by_key[key] = issue
    severe = list(severe_by_key.values())
    if not severe:
        return evidence

    event_ids: set[str] = set()
    for issue in severe:
        details = issue.get("event_details")
        if not isinstance(details, dict):
            return evidence
        issue_severity = str(issue.get("severity") or "").upper()
        event_id = _issue_event_id(issue)
        receipt_action = str(issue.get("receipt_action") or "").upper()
        if (
            issue_severity != "P1"
            or not event_id
            or receipt_action not in {"HOLD", "NO_ACTION"}
            or not exact_p1_margin_warning_source_event(
                event_type=issue.get("event_type"),
                event_severity=issue.get("event_severity"),
                action=issue.get("event_action_hint"),
                details=details,
                expected_max_margin_utilization_pct=cap,
            )
        ):
            return evidence
        event_ids.add(event_id)
    evidence["status"] = "ELIGIBLE_WARNING_ONLY"
    evidence["event_ids"] = sorted(event_ids)
    return evidence


def _current_margin_values(
    broker_snapshot: dict[str, Any],
    *,
    current_margin_utilization_pct: float | None,
    margin_available_jpy: float | None,
) -> tuple[float | None, float | None]:
    account = (
        broker_snapshot.get("account")
        if isinstance(broker_snapshot.get("account"), dict)
        else broker_snapshot
    )
    current = _strict_number(current_margin_utilization_pct)
    available = _strict_number(margin_available_jpy)
    if available is None:
        available = _strict_number(account.get("margin_available_jpy"))
    if current is None:
        nav = _strict_number(account.get("nav_jpy"))
        used = _strict_number(account.get("margin_used_jpy"))
        if nav is not None and nav > 0.0 and used is not None and used >= 0.0:
            current = used / nav * 100.0
    return current, available


def _classification_from_issue(
    issue: dict[str, Any],
    existing: dict[str, Any] | None,
    *,
    watchdog_payload: dict[str, Any],
    operator_review: dict[str, Any] | None,
    broker_snapshot: dict[str, Any] | None,
    now_utc: datetime,
    generated_at_utc: str,
    generated_by: str,
) -> dict[str, Any]:
    review_required = receipt_requires_operator_review(issue)
    review_status = (
        _operator_review_consumption_status(
            issue,
            operator_review,
            watchdog_payload=watchdog_payload,
            broker_snapshot_payload=broker_snapshot,
            now_utc=now_utc,
        )
        if review_required
        else None
    )
    if isinstance(existing, dict):
        classification = str(existing.get("classification") or "").upper()
        if classification in CLASSIFICATIONS:
            row = dict(existing)
            row.setdefault("issue_code", _issue_code(issue))
            row.setdefault("receipt_event_id", _issue_event_id(issue))
            row.setdefault("receipt_action", _issue_action(issue))
            row.setdefault("receipt_lifecycle", _issue_lifecycle(issue))
            row.setdefault("event_type", issue.get("event_type"))
            row.setdefault("event_severity", issue.get("event_severity"))
            row.setdefault("event_action_hint", issue.get("event_action_hint"))
            row.setdefault(
                "event_details",
                issue.get("event_details")
                if isinstance(issue.get("event_details"), dict)
                else {},
            )
            row.setdefault("consumed_by_trader", bool(issue.get("consumed_by_trader")))
            row.setdefault("generated_by", generated_by)
            row.setdefault("generated_at_utc", generated_at_utc)
            if review_required:
                review_clears = isinstance(review_status, dict) and review_status.get("normal_routing_allowed") is True
                row["classification"] = (
                    _operator_review_cleared_classification(issue, classification)
                    if review_clears
                    else NEEDS_OPERATOR_REVIEW
                )
                row["operator_review_required"] = True
                row["operator_review_status"] = review_status.get("status") if isinstance(review_status, dict) else None
                row["operator_review_reason"] = review_status.get("reason") if isinstance(review_status, dict) else None
                row["normal_routing_allowed"] = review_clears
                row["reason"] = _classification_reason(issue, row["classification"], review_status=review_status)
            else:
                row["normal_routing_allowed"] = _normal_allowed_for_classification(classification)
            return row

    classification = _automatic_classification(issue)
    normal_allowed = _normal_allowed_for_classification(classification)
    if review_required:
        review_clears = isinstance(review_status, dict) and review_status.get("normal_routing_allowed") is True
        classification = _operator_review_cleared_classification(issue) if review_clears else NEEDS_OPERATOR_REVIEW
        normal_allowed = review_clears
    return {
        "issue_code": _issue_code(issue),
        "receipt_event_id": _issue_event_id(issue),
        "receipt_action": _issue_action(issue),
        "receipt_lifecycle": _issue_lifecycle(issue),
        "receipt_identity": issue.get("receipt_identity"),
        "receipt_source_paths": issue.get("receipt_source_paths") if isinstance(issue.get("receipt_source_paths"), list) else [],
        "event_type": issue.get("event_type"),
        "event_severity": issue.get("event_severity"),
        "event_action_hint": issue.get("event_action_hint"),
        "event_details": (
            issue.get("event_details")
            if isinstance(issue.get("event_details"), dict)
            else {}
        ),
        "consumed_by_trader": bool(issue.get("consumed_by_trader")),
        "classification": classification,
        "reason": _classification_reason(issue, classification, review_status=review_status),
        "normal_routing_allowed": normal_allowed,
        "operator_review_required": review_required,
        "operator_review_status": review_status.get("status") if isinstance(review_status, dict) else None,
        "operator_review_reason": review_status.get("reason") if isinstance(review_status, dict) else None,
        "generated_by": generated_by,
        "generated_at_utc": generated_at_utc,
    }


def _automatic_classification(issue: dict[str, Any]) -> str:
    if issue.get("consumed_by_trader") is True:
        return CONSUMED
    if receipt_requires_operator_review(issue):
        return NEEDS_OPERATOR_REVIEW
    if issue.get("emergency_or_margin_risk") is True:
        return NEEDS_OPERATOR_REVIEW
    lifecycle = _issue_lifecycle(issue)
    sources = [str(item) for item in (issue.get("receipt_sources") or []) if str(item)]
    if sources and "canonical" not in sources:
        return HISTORICAL_ONLY
    if lifecycle == "EXPIRED":
        return EXPIRED_ACKNOWLEDGED
    if lifecycle == "REJECTED":
        return REJECTED_ACKNOWLEDGED
    if issue.get("next_run_window_missed") is True or issue.get("expired_before_trader_run") is True:
        return STALE_ACKNOWLEDGED
    return NEEDS_OPERATOR_REVIEW


def _operator_review_consumption_status(
    issue: dict[str, Any],
    operator_review: dict[str, Any] | None,
    *,
    watchdog_payload: dict[str, Any],
    broker_snapshot_payload: dict[str, Any] | None,
    now_utc: datetime,
) -> dict[str, Any]:
    fresh_status = operator_review_clearance_status(
        issue,
        operator_review,
        watchdog_payload=watchdog_payload,
        broker_snapshot_payload=broker_snapshot_payload,
        now_utc=now_utc,
    )
    if fresh_status.get("normal_routing_allowed") is True:
        return fresh_status
    durable_status = operator_review_durable_clearance_status(
        issue,
        operator_review,
        broker_snapshot_payload=broker_snapshot_payload,
    )
    if durable_status.get("normal_routing_allowed") is True:
        return durable_status
    return fresh_status


def _acknowledgement_allows_review_required_routing(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict) or row.get("normal_routing_allowed") is not True:
        return False
    classification = str(row.get("classification") or "").upper()
    if classification not in ACKNOWLEDGED_CLASSIFICATIONS:
        return False
    if not receipt_requires_operator_review(row):
        return True
    review_status = str(row.get("operator_review_status") or "").upper()
    return review_status in DURABLE_OPERATOR_REVIEW_CLEARANCE_STATUSES


def _classification_reason(
    issue: dict[str, Any],
    classification: str,
    *,
    review_status: dict[str, Any] | None = None,
) -> str:
    lifecycle = _issue_lifecycle(issue) or "UNKNOWN"
    action = _issue_action(issue) or "UNKNOWN"
    event_id = _issue_event_id(issue) or "UNKNOWN"
    if classification in ACKNOWLEDGED_CLASSIFICATIONS and isinstance(review_status, dict):
        return (
            f"Receipt event {event_id} {action} is acknowledged through operator review; "
            f"operator_review_status={review_status.get('status')}, reason={review_status.get('reason')}."
        )
    if classification == CONSUMED:
        return f"Receipt event {event_id} {action} is already marked consumed_by_trader=true."
    if classification == HISTORICAL_ONLY:
        return f"Receipt event {event_id} {action} is archived-only; no current canonical receipt remains."
    if classification == EXPIRED_ACKNOWLEDGED:
        return f"Receipt event {event_id} {action} is lifecycle EXPIRED and was not consumed before expiry."
    if classification == STALE_ACKNOWLEDGED:
        return f"Receipt event {event_id} {action} is stale: lifecycle={lifecycle} and the next trader window was missed."
    if classification == REJECTED_ACKNOWLEDGED:
        return f"Receipt event {event_id} {action} is lifecycle REJECTED and is acknowledged as non-executable."
    if isinstance(review_status, dict):
        return (
            f"Receipt event {event_id} {action} requires operator review before normal new-entry routing; "
            f"operator_review_status={review_status.get('status')}, reason={review_status.get('reason')}."
        )
    return (
        f"Receipt event {event_id} {action} requires operator review before normal new-entry routing; "
        f"lifecycle={lifecycle}, severity={issue.get('severity')}."
    )


def _normal_allowed_for_classification(classification: str) -> bool:
    return classification in ACKNOWLEDGED_CLASSIFICATIONS


def _operator_review_cleared_classification(issue: dict[str, Any], existing_classification: str = "") -> str:
    if existing_classification in ACKNOWLEDGED_CLASSIFICATIONS:
        return existing_classification
    lifecycle = _issue_lifecycle(issue)
    if lifecycle == "REJECTED":
        return REJECTED_ACKNOWLEDGED
    if lifecycle == "STALE":
        return STALE_ACKNOWLEDGED
    return HISTORICAL_ONLY


def _watchdog_issue_status_blocks(payload: dict[str, Any]) -> bool:
    issue_status = str(payload.get("issue_status") or "").upper()
    if issue_status in {"P0", "P1"}:
        return True
    severity = str(payload.get("severity") or "").upper()
    status = str(payload.get("status") or "").upper()
    if severity in {"P0", "P1"} and status in {"BLOCKED", "STALE", "BROKEN"}:
        return True
    if any(str(item.get("severity") or "").upper() in {"P0", "P1"} for item in _watchdog_issues(payload)):
        return True
    return any(str(item.get("severity") or "").upper() in {"P0", "P1"} for item in _watchdog_guardian_issues(payload))


def _watchdog_has_uncleared_p0_p1_issue(payload: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    if not _watchdog_issue_status_blocks(payload):
        return False
    severe_watchdog_issues = [
        item
        for item in _watchdog_issues(payload)
        if str(item.get("severity") or "").upper() in {"P0", "P1"}
    ]
    if any(_issue_code(issue) not in {ISSUE_CODE, OPERATOR_REVIEW_ISSUE_CODE} for issue in severe_watchdog_issues):
        return True
    severe_issues = [
        item
        for item in _watchdog_guardian_issues(payload)
        if str(item.get("severity") or "").upper() in {"P0", "P1"}
    ]
    if not severe_issues:
        return True
    cleared_keys = {
        _classification_key(row)
        for row in rows
        if row.get("normal_routing_allowed") is True
    }
    for issue in severe_issues:
        if _issue_code(issue) not in {ISSUE_CODE, OPERATOR_REVIEW_ISSUE_CODE}:
            return True
        if _issue_key(issue) not in cleared_keys:
            return True
    return False


def _watchdog_guardian_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    compact_issues = payload.get("guardian_receipt_issues")
    if isinstance(compact_issues, list):
        return [item for item in compact_issues if isinstance(item, dict)]
    guardian = payload.get("guardian_receipt") if isinstance(payload.get("guardian_receipt"), dict) else {}
    issues = guardian.get("issues") if isinstance(guardian.get("issues"), list) else []
    return [item for item in issues if isinstance(item, dict)]


def _watchdog_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    issues = payload.get("issues")
    return [item for item in (issues if isinstance(issues, list) else []) if isinstance(item, dict)]


def _classification_row_as_issue(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": row.get("issue_code") or ISSUE_CODE,
        "receipt_event_id": row.get("receipt_event_id"),
        "receipt_action": row.get("receipt_action"),
        "receipt_lifecycle": row.get("receipt_lifecycle"),
        "receipt_identity": row.get("receipt_identity"),
        "receipt_source_paths": row.get("receipt_source_paths") if isinstance(row.get("receipt_source_paths"), list) else [],
        "event_type": row.get("event_type"),
        "event_severity": row.get("event_severity"),
        "event_action_hint": row.get("event_action_hint"),
        "event_details": (
            row.get("event_details")
            if isinstance(row.get("event_details"), dict)
            else {}
        ),
        "consumed_by_trader": row.get("consumed_by_trader"),
        "classification": row.get("classification"),
        "normal_routing_allowed": row.get("normal_routing_allowed"),
    }


def _preserved_existing_issues(
    rows: list[dict[str, Any]],
    seen_issue_keys: set[tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    preserved: list[dict[str, Any]] = []
    for row in rows:
        key = _classification_key(row)
        if not _has_receipt_key(key) or key in seen_issue_keys:
            continue
        classification = str(row.get("classification") or "").upper()
        if (
            receipt_requires_operator_review(row)
            and _issue_lifecycle(row) in PRESERVABLE_REVIEW_LIFECYCLES
        ) or (
            row.get("normal_routing_allowed") is True
            and classification in ACKNOWLEDGED_CLASSIFICATIONS
        ):
            preserved.append(_classification_row_as_issue(row))
            seen_issue_keys.add(key)
    return preserved


def _env_path(name: str, fallback: Path | None) -> Path:
    raw = os.environ.get(name)
    if raw:
        return Path(raw)
    if fallback is not None:
        return fallback
    from quant_rabbit.paths import ROOT

    if name == "QR_GUARDIAN_RECEIPT_WATCHDOG_PATH":
        return ROOT / "data" / "qr_trader_run_watchdog.json"
    if name == "QR_GUARDIAN_RECEIPT_OPERATOR_REVIEW_PATH":
        return ROOT / "data" / "guardian_receipt_operator_review.json"
    if name == "QR_GUARDIAN_RECEIPT_BROKER_SNAPSHOT_PATH":
        return ROOT / "data" / "broker_snapshot.json"
    return ROOT / "data" / "guardian_receipt_consumption.json"


def _read_json_path(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _classification_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("classifications")
    return [item for item in (rows if isinstance(rows, list) else []) if isinstance(item, dict)]


def _classification_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return _key(
        row.get("issue_code"),
        row.get("receipt_event_id"),
        row.get("receipt_action"),
        row.get("receipt_lifecycle"),
    )


def _issue_key(issue: dict[str, Any]) -> tuple[str, str, str, str]:
    return _key(_issue_code(issue), _issue_event_id(issue), _issue_action(issue), _issue_lifecycle(issue))


def _has_receipt_key(key: tuple[str, str, str, str]) -> bool:
    return any(key[1:])


def _key(
    issue_code: Any,
    receipt_event_id: Any,
    receipt_action: Any,
    receipt_lifecycle: Any,
) -> tuple[str, str, str, str]:
    normalized_issue_code = str(issue_code or ISSUE_CODE).upper()
    if normalized_issue_code == OPERATOR_REVIEW_ISSUE_CODE:
        normalized_issue_code = ISSUE_CODE
    return (
        normalized_issue_code,
        str(receipt_event_id or "").strip(),
        str(receipt_action or "").upper(),
        str(receipt_lifecycle or "").upper(),
    )


def _issue_code(issue: dict[str, Any]) -> str:
    return str(issue.get("code") or issue.get("issue_code") or ISSUE_CODE).upper()


def _issue_event_id(issue: dict[str, Any]) -> str:
    return str(issue.get("receipt_event_id") or issue.get("event_id") or "")


def _issue_action(issue: dict[str, Any]) -> str:
    return str(issue.get("receipt_action") or issue.get("action") or "").upper()


def _issue_lifecycle(issue: dict[str, Any]) -> str:
    return str(issue.get("receipt_lifecycle") or "").upper()
