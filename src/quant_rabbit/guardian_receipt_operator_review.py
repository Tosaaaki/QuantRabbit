from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


OPERATOR_REVIEW_REQUIRED_BLOCK_CODE = "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"

OPERATOR_ACKNOWLEDGED_HISTORICAL = "OPERATOR_ACKNOWLEDGED_HISTORICAL"
OPERATOR_CONFIRMED_NO_ACTION = "OPERATOR_CONFIRMED_NO_ACTION"
OPERATOR_CONFIRMED_MANUAL_OWNED = "OPERATOR_CONFIRMED_MANUAL_OWNED"
OPERATOR_REQUESTS_KEEP_BLOCKED = "OPERATOR_REQUESTS_KEEP_BLOCKED"
OPERATOR_REQUESTS_FRESH_REVIEW = "OPERATOR_REQUESTS_FRESH_REVIEW"

OPERATOR_REVIEW_DECISIONS = {
    OPERATOR_ACKNOWLEDGED_HISTORICAL,
    OPERATOR_CONFIRMED_NO_ACTION,
    OPERATOR_CONFIRMED_MANUAL_OWNED,
    OPERATOR_REQUESTS_KEEP_BLOCKED,
    OPERATOR_REQUESTS_FRESH_REVIEW,
}
OPERATOR_REVIEW_CLEARANCE_DECISIONS = {
    OPERATOR_ACKNOWLEDGED_HISTORICAL,
    OPERATOR_CONFIRMED_NO_ACTION,
    OPERATOR_CONFIRMED_MANUAL_OWNED,
}
OPERATOR_REVIEW_BLOCKING_DECISIONS = {
    OPERATOR_REQUESTS_KEEP_BLOCKED,
    OPERATOR_REQUESTS_FRESH_REVIEW,
}
OPERATOR_REVIEW_CLEARS_RECEIPT = "OPERATOR_REVIEW_CLEARS_RECEIPT"
OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT = "OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT"

NEEDS_OPERATOR_REVIEW = "NEEDS_OPERATOR_REVIEW"
OPERATOR_REVIEW_ISSUE_CODE = "GUARDIAN_RECEIPT_NEEDS_OPERATOR_REVIEW"
BASE_RECEIPT_ISSUE_CODE = "GUARDIAN_RECEIPT_NOT_CONSUMED_BY_TRADER"
REVIEW_REQUIRED_ACTIONS = {"REDUCE", "HARVEST", "CANCEL_PENDING"}
HISTORICAL_LIFECYCLES = {"EXPIRED", "STALE", "REJECTED", "HISTORICAL_ONLY"}
P0_P1 = {"P0", "P1"}

# Operator review freshness is an operational handoff window, not a market
# threshold. A review older than one day may describe broker truth that has
# changed across multiple trader cycles, so rows must carry an explicit expiry.
OPERATOR_REVIEW_DEFAULT_TTL_HOURS = 24


def load_guardian_receipt_operator_review(path: Path) -> dict[str, Any]:
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


def build_guardian_receipt_operator_review(
    watchdog_payload: dict[str, Any] | None,
    consumption_payload: dict[str, Any] | None,
    *,
    broker_snapshot_payload: dict[str, Any] | None = None,
    operator_decision_payload: dict[str, Any] | None = None,
    now_utc: datetime | None = None,
    generated_by: str = "guardian-receipt-operator-review",
) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    decision_rows = _decision_rows(operator_decision_payload)
    subjects = _operator_review_subjects(watchdog_payload, consumption_payload)
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        explicit = _matching_decision_row(decision_rows, subject)
        operator_decision = _operator_decision(explicit)
        if explicit is None and operator_decision_payload is None:
            operator_decision = OPERATOR_REQUESTS_KEEP_BLOCKED
        elif explicit is None:
            operator_decision = OPERATOR_REQUESTS_KEEP_BLOCKED
        reason = _operator_reason(explicit) or _default_operator_reason(operator_decision, explicit is not None)
        expires_at = _operator_expires_at(explicit, now)
        row = {
            "receipt_event_id": _issue_event_id(subject),
            "receipt_action": _issue_action(subject),
            "receipt_lifecycle": _issue_lifecycle(subject),
            "original_issue_code": _issue_code(subject),
            "operator_decision": operator_decision,
            "reason": reason,
            "generated_at_utc": now.isoformat(),
            "expires_at_utc": expires_at.isoformat(),
            "normal_routing_allowed": False,
            "no_live_side_effects": True,
            "generated_by": generated_by,
        }
        if explicit is not None:
            for key in (
                "trade_id",
                "pair",
                "side",
                "units",
                "avg_entry",
                "owner",
                "management_intent",
                "operator_confirmation_source",
                "system_pl_counted",
                "same_theme_auto_add_allowed",
                "loss_side_auto_close_allowed",
                "auto_sl_attach_allowed",
                "auto_tp_modify_allowed",
            ):
                if key in explicit:
                    row[key] = explicit[key]
        rows.append(row)

    preview_payload = {
        "generated_at_utc": now.isoformat(),
        "status": "NO_OPERATOR_REVIEW_REQUIRED",
        "normal_routing_allowed": True,
        "classifications": rows,
        "operator_position_reviews": _operator_position_review_rows(operator_decision_payload),
        "read_only": True,
        "no_live_side_effects": True,
        "live_side_effects": [],
    }
    for subject, row in zip(subjects, rows):
        status = operator_review_clearance_status(
            subject,
            preview_payload,
            watchdog_payload=watchdog_payload,
            broker_snapshot_payload=broker_snapshot_payload,
            now_utc=now,
        )
        row["normal_routing_allowed"] = status.get("normal_routing_allowed") is True
        row["clearance_status"] = status.get("status")
        row["clearance_reason"] = status.get("reason")

    review_allowed = all(row.get("normal_routing_allowed") is True for row in rows)
    current_p0_p1_blocks = _has_uncleared_p0_p1_watchdog_issue(watchdog_payload, rows)
    normal_allowed = review_allowed and not current_p0_p1_blocks
    if not rows:
        status = "NO_OPERATOR_REVIEW_REQUIRED"
        normal_allowed = not current_p0_p1_blocks
        if current_p0_p1_blocks:
            status = "NO_OPERATOR_REVIEW_REQUIRED_CURRENT_P0_BLOCKS_ROUTING"
    elif review_allowed and current_p0_p1_blocks:
        status = "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED_CURRENT_P0_BLOCKS_ROUTING"
    elif review_allowed:
        status = "GUARDIAN_RECEIPT_OPERATOR_REVIEW_CLEARED"
    else:
        status = "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED"
    preview_payload.update(
        {
            "status": status,
            "normal_routing_allowed": normal_allowed,
            "unresolved_review_count": sum(1 for row in rows if row.get("normal_routing_allowed") is not True),
            "current_p0_p1_blocks_routing": current_p0_p1_blocks,
            "watchdog_generated_at_utc": (
                watchdog_payload.get("generated_at_utc") if isinstance(watchdog_payload, dict) else None
            ),
            "watchdog_status": watchdog_payload.get("status") if isinstance(watchdog_payload, dict) else None,
            "watchdog_issue_status": watchdog_payload.get("issue_status") if isinstance(watchdog_payload, dict) else None,
        }
    )
    return preview_payload


def write_guardian_receipt_operator_review(
    payload: dict[str, Any],
    *,
    output_path: Path,
    report_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_guardian_receipt_operator_review_report(payload), encoding="utf-8")


def render_guardian_receipt_operator_review_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Guardian Receipt Operator Review Report",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Normal routing allowed: `{payload.get('normal_routing_allowed')}`",
        f"- no_live_side_effects: `{payload.get('no_live_side_effects')}`",
        f"- Live side effects: `{len(payload.get('live_side_effects') or [])}`",
        "",
        "## Classifications",
        "",
    ]
    rows = _review_rows(payload)
    if not rows:
        lines.append("- none")
    else:
        lines.extend(
            [
                "| Event | Action | Lifecycle | Original Issue | Operator Decision | Normal Routing | Generated UTC | Expires UTC | no_live_side_effects | Reason |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in rows:
            reason = str(row.get("reason") or "").replace("|", "\\|")
            lines.append(
                f"| `{row.get('receipt_event_id')}` | `{row.get('receipt_action')}` | "
                f"`{row.get('receipt_lifecycle')}` | `{row.get('original_issue_code')}` | "
                f"`{row.get('operator_decision')}` | `{row.get('normal_routing_allowed')}` | "
                f"`{row.get('generated_at_utc')}` | `{row.get('expires_at_utc')}` | "
                f"`{row.get('no_live_side_effects')}` | {reason} |"
            )
        position_rows = [
            row
            for row in rows
            if any(row.get(key) is not None for key in ("trade_id", "owner", "management_intent"))
        ]
        position_rows.extend(_operator_position_review_rows(payload))
        if position_rows:
            lines.extend(
                [
                    "",
                    "## Operator Position Review",
                    "",
                    "| Trade ID | Pair | Side | Units | Avg Entry | Owner | Management Intent | Source | System P/L Counted | Same-Theme Add | Loss Close | Auto SL | Auto TP |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
                ]
            )
            for row in position_rows:
                lines.append(
                    f"| `{row.get('trade_id')}` | `{row.get('pair')}` | `{row.get('side')}` | "
                    f"`{row.get('units')}` | `{row.get('avg_entry')}` | `{row.get('owner')}` | "
                    f"`{row.get('management_intent')}` | `{row.get('operator_confirmation_source')}` | "
                    f"`{row.get('system_pl_counted')}` | `{row.get('same_theme_auto_add_allowed')}` | "
                    f"`{row.get('loss_side_auto_close_allowed')}` | `{row.get('auto_sl_attach_allowed')}` | "
                    f"`{row.get('auto_tp_modify_allowed')}` |"
                )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This artifact records operator review for expired/historical guardian receipts.",
            "- It does not place orders, cancel orders, close positions, enable execution flags, or modify broker state.",
            "- Receipt clearance requires a fresh explicit operator decision, expired/historical receipt lifecycle, and broker-truth clearance for the same event.",
            "- Top-level normal routing also requires no separate unresolved P0/P1 guardian/watchdog issue.",
            "",
        ]
    )
    return "\n".join(lines)


def operator_review_status_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("_missing"):
        return {
            "status": "UNAVAILABLE",
            "normal_routing_allowed": None,
            "classifications": [],
            "unresolved_review_count": 0,
            "missing": True,
            "path": payload.get("_path") if isinstance(payload, dict) else None,
        }
    return {
        "status": payload.get("status"),
        "normal_routing_allowed": payload.get("normal_routing_allowed"),
        "classifications": _review_rows(payload),
        "unresolved_review_count": payload.get("unresolved_review_count"),
        "missing": False,
        "generated_at_utc": payload.get("generated_at_utc"),
        "watchdog_generated_at_utc": payload.get("watchdog_generated_at_utc"),
        "watchdog_issue_status": payload.get("watchdog_issue_status"),
    }


def receipt_requires_operator_review(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    if _bool_value(record.get("consumed_by_trader")):
        return False
    code = _issue_code(record)
    if code == OPERATOR_REVIEW_ISSUE_CODE:
        return True
    classification = str(record.get("classification") or record.get("acknowledgement_classification") or "").upper()
    if classification == NEEDS_OPERATOR_REVIEW:
        return True
    action = _issue_action(record)
    lifecycle = _issue_lifecycle(record)
    if _bool_value(record.get("emergency_or_margin_risk")) and lifecycle in HISTORICAL_LIFECYCLES:
        return True
    return action in REVIEW_REQUIRED_ACTIONS and lifecycle in HISTORICAL_LIFECYCLES


def operator_review_clearance_status(
    issue: dict[str, Any],
    review_payload: dict[str, Any] | None,
    *,
    watchdog_payload: dict[str, Any] | None = None,
    broker_snapshot_payload: dict[str, Any] | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if not receipt_requires_operator_review(issue):
        return {
            "status": "OPERATOR_REVIEW_NOT_REQUIRED",
            "normal_routing_allowed": True,
            "reason": "receipt does not require operator review",
        }
    if _issue_lifecycle(issue) not in HISTORICAL_LIFECYCLES:
        return _blocked("RECEIPT_NOT_HISTORICAL", "receipt is not expired/historical; operator review cannot clear it")
    row = _matching_review_row(_review_rows(review_payload), issue)
    if row is None:
        return _blocked("OPERATOR_REVIEW_MISSING", "operator review artifact has no matching row")
    if row.get("no_live_side_effects") is not True:
        return _blocked("OPERATOR_REVIEW_SIDE_EFFECT_BOUNDARY_MISSING", "operator review row must set no_live_side_effects=true")
    decision = _operator_decision(row)
    if decision not in OPERATOR_REVIEW_CLEARANCE_DECISIONS:
        return _blocked("OPERATOR_REVIEW_DECISION_BLOCKS", f"operator_decision={decision} does not allow clearance")
    fresh = _review_row_fresh(row, now)
    if fresh is not True:
        return _blocked("OPERATOR_REVIEW_STALE", str(fresh))
    broker_clear = _broker_truth_clears_event(issue, broker_snapshot_payload, decision)
    if broker_clear is not True:
        return _blocked("BROKER_TRUTH_EMERGENCY_NOT_CLEARED", str(broker_clear))
    return {
        "status": OPERATOR_REVIEW_CLEARS_RECEIPT,
        "normal_routing_allowed": True,
        "operator_decision": decision,
        "reason": f"operator_decision={decision} is fresh and broker truth clears the reviewed event",
    }


def operator_review_durable_clearance_status(
    issue: dict[str, Any],
    review_payload: dict[str, Any] | None,
    *,
    broker_snapshot_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not receipt_requires_operator_review(issue):
        return {
            "status": "OPERATOR_REVIEW_NOT_REQUIRED",
            "normal_routing_allowed": True,
            "reason": "receipt does not require operator review",
        }
    if _issue_lifecycle(issue) not in HISTORICAL_LIFECYCLES:
        return _blocked("RECEIPT_NOT_HISTORICAL", "receipt is not expired/historical; durable review cannot clear it")
    row = _matching_review_row(_review_rows(review_payload), issue)
    if row is None:
        return _blocked("OPERATOR_REVIEW_MISSING", "operator review artifact has no matching row")
    if row.get("no_live_side_effects") is not True:
        return _blocked("OPERATOR_REVIEW_SIDE_EFFECT_BOUNDARY_MISSING", "operator review row must set no_live_side_effects=true")
    decision = _operator_decision(row)
    if decision not in OPERATOR_REVIEW_CLEARANCE_DECISIONS:
        return _blocked("OPERATOR_REVIEW_DECISION_BLOCKS", f"operator_decision={decision} does not allow clearance")
    generated_at = _parse_utc(row.get("generated_at_utc"))
    expires_at = _parse_utc(row.get("expires_at_utc"))
    if generated_at is None:
        return _blocked("OPERATOR_REVIEW_NOT_DURABLY_CLEARED", "operator review row missing generated_at_utc")
    if expires_at is None:
        return _blocked("OPERATOR_REVIEW_NOT_DURABLY_CLEARED", "operator review row missing expires_at_utc")
    if expires_at <= generated_at:
        return _blocked("OPERATOR_REVIEW_NOT_DURABLY_CLEARED", "operator review row expiry is not after generation")
    clearance_status = str(row.get("clearance_status") or "").upper()
    if row.get("normal_routing_allowed") is not True or clearance_status != OPERATOR_REVIEW_CLEARS_RECEIPT:
        return _blocked(
            "OPERATOR_REVIEW_NOT_DURABLY_CLEARED",
            "operator review row was not previously generated as a cleared receipt",
        )
    broker_clear = _broker_truth_clears_event(issue, broker_snapshot_payload, decision)
    if broker_clear is not True:
        return _blocked("BROKER_TRUTH_EMERGENCY_NOT_CLEARED", str(broker_clear))
    return {
        "status": OPERATOR_REVIEW_DURABLY_CONSUMED_RECEIPT,
        "normal_routing_allowed": True,
        "operator_decision": decision,
        "reason": (
            f"operator_decision={decision} previously fresh-cleared this historical receipt and "
            "current broker truth still clears the reviewed event"
        ),
    }


def operator_review_subjects_from_artifacts(
    watchdog_payload: dict[str, Any] | None,
    consumption_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return _operator_review_subjects(watchdog_payload, consumption_payload)


def _blocked(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "normal_routing_allowed": False, "reason": reason}


def _operator_review_subjects(
    watchdog_payload: dict[str, Any] | None,
    consumption_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for issue in _watchdog_guardian_issues(watchdog_payload if isinstance(watchdog_payload, dict) else {}):
        if receipt_requires_operator_review(issue):
            candidates.append(issue)
    for row in _consumption_rows(consumption_payload if isinstance(consumption_payload, dict) else {}):
        if receipt_requires_operator_review(row):
            candidates.append(_row_as_issue(row))
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in candidates:
        deduped[_issue_key(item)] = item
    return list(deduped.values())


def _row_as_issue(row: dict[str, Any]) -> dict[str, Any]:
    issue = {
        "code": row.get("issue_code") or BASE_RECEIPT_ISSUE_CODE,
        "receipt_event_id": row.get("receipt_event_id"),
        "receipt_action": row.get("receipt_action"),
        "receipt_lifecycle": row.get("receipt_lifecycle"),
        "receipt_identity": row.get("receipt_identity"),
        "receipt_source_paths": row.get("receipt_source_paths") if isinstance(row.get("receipt_source_paths"), list) else [],
        "consumed_by_trader": row.get("consumed_by_trader"),
        "classification": row.get("classification"),
        "normal_routing_allowed": row.get("normal_routing_allowed"),
    }
    return issue


def _decision_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("decisions")
    if not isinstance(rows, list):
        rows = payload.get("classifications")
    return [item for item in (rows if isinstance(rows, list) else []) if isinstance(item, dict)]


def _operator_position_review_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("operator_position_reviews")
    return [item for item in (rows if isinstance(rows, list) else []) if isinstance(item, dict)]


def _review_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("_missing"):
        return []
    rows = payload.get("classifications")
    return [item for item in (rows if isinstance(rows, list) else []) if isinstance(item, dict)]


def _consumption_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("classifications")
    return [item for item in (rows if isinstance(rows, list) else []) if isinstance(item, dict)]


def _watchdog_guardian_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    compact_issues = payload.get("guardian_receipt_issues")
    if isinstance(compact_issues, list):
        return [item for item in compact_issues if isinstance(item, dict)]
    guardian = payload.get("guardian_receipt") if isinstance(payload.get("guardian_receipt"), dict) else {}
    issues = guardian.get("issues") if isinstance(guardian.get("issues"), list) else []
    return [item for item in issues if isinstance(item, dict)]


def _matching_review_row(rows: list[dict[str, Any]], issue: dict[str, Any]) -> dict[str, Any] | None:
    key = _issue_key(issue)
    for row in rows:
        if _issue_key(row) == key:
            return row
    return None


def _matching_decision_row(rows: list[dict[str, Any]], issue: dict[str, Any]) -> dict[str, Any] | None:
    return _matching_review_row(rows, issue)


def _issue_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (_issue_event_id(record), _issue_action(record), _issue_lifecycle(record))


def _issue_code(record: dict[str, Any]) -> str:
    code = record.get("original_issue_code") or record.get("code") or record.get("issue_code") or BASE_RECEIPT_ISSUE_CODE
    return str(code).upper()


def _issue_event_id(record: dict[str, Any]) -> str:
    return str(record.get("receipt_event_id") or record.get("event_id") or "").strip()


def _issue_action(record: dict[str, Any]) -> str:
    return str(record.get("receipt_action") or record.get("action") or "").upper()


def _issue_lifecycle(record: dict[str, Any]) -> str:
    return str(record.get("receipt_lifecycle") or "").upper()


def _operator_decision(row: dict[str, Any] | None) -> str:
    decision = str((row or {}).get("operator_decision") or "").upper()
    if decision in OPERATOR_REVIEW_DECISIONS:
        return decision
    return OPERATOR_REQUESTS_KEEP_BLOCKED


def _operator_reason(row: dict[str, Any] | None) -> str:
    return str((row or {}).get("reason") or "").strip()


def _default_operator_reason(operator_decision: str, explicit: bool) -> str:
    if explicit:
        return f"Local operator decision file selected {operator_decision} without a reason."
    return "No explicit local operator decision file supplied; keep ordinary fresh-entry routing blocked."


def _operator_expires_at(row: dict[str, Any] | None, now: datetime) -> datetime:
    parsed = _parse_utc((row or {}).get("expires_at_utc"))
    if parsed is not None:
        return parsed
    return now + timedelta(hours=OPERATOR_REVIEW_DEFAULT_TTL_HOURS)


def _review_row_fresh(row: dict[str, Any], now: datetime) -> bool | str:
    generated_at = _parse_utc(row.get("generated_at_utc"))
    expires_at = _parse_utc(row.get("expires_at_utc"))
    if generated_at is None:
        return "operator review row missing generated_at_utc"
    if expires_at is None:
        return "operator review row missing expires_at_utc"
    if generated_at > now + timedelta(seconds=1):
        return "operator review row is generated in the future"
    if expires_at <= now:
        return "operator review row is expired"
    return True


def _has_current_p0_p1_watchdog_issue(watchdog_payload: dict[str, Any] | None) -> bool:
    if not isinstance(watchdog_payload, dict):
        return False
    issue_status = str(watchdog_payload.get("issue_status") or "").upper()
    if issue_status in P0_P1:
        return True
    severity = str(watchdog_payload.get("severity") or "").upper()
    if severity in P0_P1 and str(watchdog_payload.get("status") or "").upper() == "BLOCKED":
        return True
    for item in _watchdog_guardian_issues(watchdog_payload):
        if str(item.get("severity") or "").upper() in P0_P1:
            return True
    return False


def _has_uncleared_p0_p1_watchdog_issue(
    watchdog_payload: dict[str, Any] | None,
    cleared_review_rows: list[dict[str, Any]],
) -> bool:
    if not _has_current_p0_p1_watchdog_issue(watchdog_payload):
        return False
    if not isinstance(watchdog_payload, dict):
        return False
    severe_issues = [
        item
        for item in _watchdog_guardian_issues(watchdog_payload)
        if str(item.get("severity") or "").upper() in P0_P1
    ]
    if not severe_issues:
        return True
    cleared_keys = {
        _issue_key(row)
        for row in cleared_review_rows
        if row.get("normal_routing_allowed") is True
    }
    for issue in severe_issues:
        if _issue_code(issue) not in {OPERATOR_REVIEW_ISSUE_CODE, BASE_RECEIPT_ISSUE_CODE}:
            return True
        if _issue_key(issue) not in cleared_keys:
            return True
    return False


def _broker_truth_clears_event(
    issue: dict[str, Any],
    broker_snapshot_payload: dict[str, Any] | None,
    operator_decision: str,
) -> bool | str:
    if not isinstance(broker_snapshot_payload, dict) or not broker_snapshot_payload:
        return "broker snapshot is missing; cannot prove event cleared"
    identifiers = _event_identifiers(issue)
    if not identifiers:
        if _bool_value(issue.get("emergency_or_margin_risk")):
            return "receipt was emergency/margin risk and no event identifier was available for broker reconciliation"
        return True
    positions = broker_snapshot_payload.get("positions")
    if not isinstance(positions, list):
        positions = broker_snapshot_payload.get("position_summaries")
    if not isinstance(positions, list):
        positions = []
    orders = broker_snapshot_payload.get("orders")
    if not isinstance(orders, list):
        orders = broker_snapshot_payload.get("pending_orders")
    if not isinstance(orders, list):
        orders = []
    live_position_ids = {str(item.get("trade_id") or "") for item in positions if isinstance(item, dict)}
    live_order_ids = {
        str(item.get("order_id") or item.get("id") or item.get("trade_id") or "")
        for item in orders
        if isinstance(item, dict)
    }
    for trade_id in identifiers.get("trade_ids", set()):
        if trade_id in live_position_ids:
            if operator_decision == OPERATOR_CONFIRMED_MANUAL_OWNED and _position_is_operator_owned(positions, trade_id):
                continue
            return f"broker snapshot still contains reviewed trade_id={trade_id}"
    for order_id in identifiers.get("order_ids", set()):
        if order_id in live_order_ids:
            return f"broker snapshot still contains reviewed order_id={order_id}"
    return True


def _position_is_operator_owned(positions: list[Any], trade_id: str) -> bool:
    for item in positions:
        if not isinstance(item, dict) or str(item.get("trade_id") or "") != trade_id:
            continue
        owner = str(item.get("owner") or "").lower()
        return owner in {"operator_manual", "manual"}
    return False


def _event_identifiers(issue: dict[str, Any]) -> dict[str, set[str]]:
    trade_ids: set[str] = set()
    order_ids: set[str] = set()
    _collect_identifiers(issue, trade_ids=trade_ids, order_ids=order_ids)
    for raw_path in issue.get("receipt_source_paths") or []:
        path = Path(str(raw_path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            _collect_identifiers(payload, trade_ids=trade_ids, order_ids=order_ids)
    return {"trade_ids": trade_ids, "order_ids": order_ids}


def _collect_identifiers(payload: Any, *, trade_ids: set[str], order_ids: set[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered == "trade_id" and str(value).strip():
                trade_ids.add(str(value).strip())
            elif lowered == "order_id" and str(value).strip():
                order_ids.add(str(value).strip())
            elif lowered in {"id", "client_order_id"} and str(value).strip() and "order" in lowered:
                order_ids.add(str(value).strip())
            elif isinstance(value, (dict, list)):
                _collect_identifiers(value, trade_ids=trade_ids, order_ids=order_ids)
    elif isinstance(payload, list):
        for item in payload:
            _collect_identifiers(item, trade_ids=trade_ids, order_ids=order_ids)


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
