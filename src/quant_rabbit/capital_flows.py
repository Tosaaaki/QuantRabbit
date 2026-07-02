from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAPITAL_FLOW_TYPES = {"DEPOSIT", "WITHDRAWAL"}
DEFAULT_OPERATOR_DEPOSIT_TIMESTAMP_UTC = "2026-07-02T08:33:11Z"
DEFAULT_OPERATOR_DEPOSIT_AMOUNT_JPY = 100_000
DEFAULT_OPERATOR_DEPOSIT_SOURCE = "operator"
DEFAULT_OPERATOR_DEPOSIT_NOTE = "100,000 JPY operator capital injection"


@dataclass(frozen=True)
class CapitalFlowSummary:
    net_amount_jpy: float
    count: int
    flows: tuple[dict[str, Any], ...]
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapitalFlowArtifactResult:
    flows_path: Path
    report_path: Path
    created: bool
    updated: bool
    deposit_timestamp_utc: str
    flow_count: int
    issues: tuple[str, ...] = ()


def summarize_capital_flows(
    path: Path,
    *,
    start_utc: datetime,
    end_utc: datetime,
) -> CapitalFlowSummary:
    """Return funding flows that raw equity includes but performance excludes."""

    flows, issues = _load_capital_flows(path)
    start = _normalize_utc(start_utc)
    end = _normalize_utc(end_utc)
    included: list[dict[str, Any]] = []
    net = 0.0
    for flow in flows:
        timestamp = _parse_timestamp(flow.get("timestamp_utc"))
        if timestamp is None:
            issues.append("CAPITAL_FLOW_TIMESTAMP_INVALID")
            continue
        if timestamp < start or timestamp > end:
            continue
        if flow.get("included_in_raw_equity") is not True:
            continue
        if flow.get("excluded_from_funding_adjusted_return") is not True:
            continue
        flow_type = str(flow.get("type") or "").upper()
        if flow_type not in CAPITAL_FLOW_TYPES:
            issues.append(f"CAPITAL_FLOW_TYPE_INVALID:{flow_type or 'missing'}")
            continue
        try:
            amount = float(flow.get("amount_jpy"))
        except (TypeError, ValueError):
            issues.append("CAPITAL_FLOW_AMOUNT_INVALID")
            continue
        signed_amount = -abs(amount) if flow_type == "WITHDRAWAL" else amount
        net += signed_amount
        included.append(dict(flow))
    return CapitalFlowSummary(
        net_amount_jpy=round(net, 4),
        count=len(included),
        flows=tuple(included),
        issues=tuple(issues),
    )


def funding_adjusted_equity(current_equity_raw: float, capital_flows_jpy: float) -> float:
    return round(float(current_equity_raw) - float(capital_flows_jpy), 4)


def ensure_operator_deposit_artifact(
    flows_path: Path,
    report_path: Path,
    *,
    target_state_path: Path | None = None,
    generated_at_utc: datetime | None = None,
    amount_jpy: float = DEFAULT_OPERATOR_DEPOSIT_AMOUNT_JPY,
    timestamp_utc: str = DEFAULT_OPERATOR_DEPOSIT_TIMESTAMP_UTC,
    source: str = DEFAULT_OPERATOR_DEPOSIT_SOURCE,
    note: str = DEFAULT_OPERATOR_DEPOSIT_NOTE,
) -> CapitalFlowArtifactResult:
    """Materialize the ignored live capital-flow artifact without broker writes."""

    generated = _normalize_utc(generated_at_utc or datetime.now(timezone.utc))
    generated_text = _format_utc(generated)
    created = not flows_path.exists()
    payload, flows = _read_payload_for_write(flows_path)
    flow, found_index = _find_operator_deposit(flows, amount_jpy=amount_jpy, source=source)
    if flow is None:
        flow = {
            "amount_jpy": amount_jpy,
            "excluded_from_funding_adjusted_return": True,
            "included_in_raw_equity": True,
            "note": note,
            "source": source,
            "timestamp_utc": timestamp_utc,
            "type": "DEPOSIT",
        }
        flows.append(flow)
        updated = True
    else:
        preserved_timestamp = str(flow.get("timestamp_utc") or timestamp_utc)
        repaired = {
            **flow,
            "amount_jpy": amount_jpy,
            "excluded_from_funding_adjusted_return": True,
            "included_in_raw_equity": True,
            "note": note,
            "source": source,
            "timestamp_utc": preserved_timestamp,
            "type": "DEPOSIT",
        }
        updated = repaired != flow
        flows[found_index] = repaired
        flow = repaired

    next_payload = {
        "schema_version": int(payload.get("schema_version") or 1),
        "generated_at_utc": generated_text,
        "capital_flows": flows,
    }
    if created or updated or payload != next_payload:
        flows_path.parent.mkdir(parents=True, exist_ok=True)
        flows_path.write_text(json.dumps(next_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        updated = True

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_capital_flow_report(
            next_payload,
            generated_at_utc=generated_text,
            target_state_path=target_state_path,
        )
    )
    return CapitalFlowArtifactResult(
        flows_path=flows_path,
        report_path=report_path,
        created=created,
        updated=updated,
        deposit_timestamp_utc=str(flow.get("timestamp_utc") or timestamp_utc),
        flow_count=len(flows),
    )


def render_capital_flow_report(
    payload: dict[str, Any],
    *,
    generated_at_utc: str,
    target_state_path: Path | None = None,
) -> str:
    flows = payload.get("capital_flows") if isinstance(payload.get("capital_flows"), list) else []
    lines = [
        "# Capital Flow Report",
        "",
        f"- Generated at UTC: `{generated_at_utc}`",
        "- Scope: accounting/reporting only; no orders, cancels, closes, execution flags, or broker-state writes.",
        "- Source basis: operator statement plus local target state when available; no broker transaction fetch was performed for this record.",
        "",
        "## Recorded Flows",
        "",
        "| timestamp_utc | amount_jpy | type | source | note | included_in_raw_equity | excluded_from_funding_adjusted_return |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for flow in flows:
        if not isinstance(flow, dict):
            continue
        lines.append(
            "| "
            f"`{flow.get('timestamp_utc', '')}` | "
            f"`{flow.get('amount_jpy', '')}` | "
            f"`{flow.get('type', '')}` | "
            f"`{flow.get('source', '')}` | "
            f"`{flow.get('note', '')}` | "
            f"`{str(flow.get('included_in_raw_equity')).lower()}` | "
            f"`{str(flow.get('excluded_from_funding_adjusted_return')).lower()}` |"
        )

    target_state = _load_target_state(target_state_path)
    lines.extend(["", "## 30D Target Accounting", ""])
    if target_state:
        fields = (
            "rolling_30d_start_equity",
            "current_equity_raw",
            "capital_flows_30d",
            "funding_adjusted_equity",
            "rolling_30d_multiplier_raw",
            "rolling_30d_multiplier_funding_adjusted",
            "remaining_to_4x_raw",
            "remaining_to_4x_funding_adjusted",
            "required_calendar_daily_return_funding_adjusted",
            "required_active_day_return_funding_adjusted",
            "performance_basis",
            "sizing_basis",
        )
        lines.extend(["| field | value |", "| --- | --- |"])
        for field in fields:
            lines.append(f"| {field} | `{target_state.get(field, 'n/a')}` |")
    else:
        lines.append("- Target state was not available; run `daily-target-state` to populate current raw/funding-adjusted metrics.")

    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Raw NAV includes deposits and withdrawals because broker equity includes funding flows.",
            "- Funding-adjusted equity excludes deposits and withdrawals from trading performance.",
            "- Rolling 30d 4x performance uses `funding_adjusted_equity`, `rolling_30d_multiplier_funding_adjusted`, and `remaining_to_4x_funding_adjusted`.",
            "- Risk, margin, and sizing use raw broker NAV / `current_equity_raw`.",
            "- A raw NAV increase caused by a deposit must not be described as trading P/L or as the authoritative 30d 4x KPI.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_capital_flows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [], [f"CAPITAL_FLOW_JSON_INVALID:{exc}"]
    if isinstance(payload, list):
        raw_flows = payload
    elif isinstance(payload, dict):
        raw_flows = payload.get("capital_flows") or payload.get("flows") or []
    else:
        return [], ["CAPITAL_FLOW_ROOT_INVALID"]
    if not isinstance(raw_flows, list):
        return [], ["CAPITAL_FLOW_LIST_INVALID"]
    flows = [dict(item) for item in raw_flows if isinstance(item, dict)]
    return flows, []


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _normalize_utc(parsed)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _read_payload_for_write(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        return {}, []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"capital flow artifact is invalid JSON: {path}: {exc}") from exc
    if isinstance(payload, list):
        return {"capital_flows": payload}, [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError(f"capital flow artifact root must be object or list: {path}")
    raw_flows = payload.get("capital_flows") or payload.get("flows") or []
    if not isinstance(raw_flows, list):
        raise ValueError(f"capital flow artifact capital_flows must be a list: {path}")
    return payload, [dict(item) for item in raw_flows if isinstance(item, dict)]


def _find_operator_deposit(
    flows: list[dict[str, Any]],
    *,
    amount_jpy: float,
    source: str,
) -> tuple[dict[str, Any] | None, int]:
    for index, flow in enumerate(flows):
        flow_type = str(flow.get("type") or "").upper()
        flow_source = str(flow.get("source") or "").strip().lower()
        try:
            amount = float(flow.get("amount_jpy"))
        except (TypeError, ValueError):
            amount = None
        if flow_type == "DEPOSIT" and flow_source == source.lower() and amount == float(amount_jpy):
            return flow, index
    return None, -1


def _load_target_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _format_utc(value: datetime) -> str:
    return _normalize_utc(value).isoformat().replace("+00:00", "Z")
