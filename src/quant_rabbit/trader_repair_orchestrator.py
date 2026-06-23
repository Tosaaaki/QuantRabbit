from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
    DEFAULT_TRADER_REPAIR_ORCHESTRATOR_REPORT,
    DEFAULT_TRADER_SUPPORT_BOT,
)
from quant_rabbit.trader_support_bot import (
    REPAIR_AUTOMATION_ALLOWED_ACTIONS,
    REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
    REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
    repair_requests_from_support_payload,
)


CONTRACT_VERSION = "trader_repair_orchestrator_v1"
STATUS_READY = "READY_FOR_CODEX_REPAIR"
STATUS_APPROVAL_REQUIRED = "OPERATOR_APPROVAL_REQUIRED"
STATUS_NO_REQUESTS = "NO_REPAIR_REQUESTS"
STATUS_BLOCKED = "ORCHESTRATOR_BLOCKED"

@dataclass(frozen=True)
class TraderRepairOrchestratorSummary:
    status: str
    output_path: Path
    report_path: Path
    selected_request_code: str | None
    actionable_request_count: int
    approval_required_request_count: int


class TraderRepairOrchestrator:
    """Build a machine-readable repair queue for the Codex work loop.

    This command is deliberately read-only. It translates the support bot's
    repair requests into an execution contract: Codex may edit/test/commit/sync
    code, while order sends, cancels, position closes, and launchd mutations
    remain outside this automation unless an existing gateway or explicit
    operator approval authorizes them.
    """

    def __init__(
        self,
        *,
        support_bot_path: Path = DEFAULT_TRADER_SUPPORT_BOT,
        output_path: Path = DEFAULT_TRADER_REPAIR_ORCHESTRATOR,
        report_path: Path = DEFAULT_TRADER_REPAIR_ORCHESTRATOR_REPORT,
        trader_request: str | None = None,
        now_utc: datetime | None = None,
    ) -> None:
        self.support_bot_path = support_bot_path
        self.output_path = output_path
        self.report_path = report_path
        self.trader_request = trader_request or ""
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> TraderRepairOrchestratorSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        selected = payload.get("selected_request") if isinstance(payload.get("selected_request"), dict) else {}
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        return TraderRepairOrchestratorSummary(
            status=str(payload.get("status") or STATUS_BLOCKED),
            output_path=self.output_path,
            report_path=self.report_path,
            selected_request_code=selected.get("code"),
            actionable_request_count=int(metrics.get("actionable_request_count") or 0),
            approval_required_request_count=int(metrics.get("approval_required_request_count") or 0),
        )

    def build_payload(self) -> dict[str, Any]:
        support = _read_json(self.support_bot_path)
        raw_requests = support.get("repair_requests") if isinstance(support.get("repair_requests"), list) else []
        requests = [item for item in raw_requests if isinstance(item, dict)]
        request_source = "top_level_repair_requests"
        recovered_from_embedded_support = False
        if not requests:
            recovered_requests = repair_requests_from_support_payload(support)
            if recovered_requests:
                requests = recovered_requests
                request_source = "embedded_support_payload"
                recovered_from_embedded_support = True
        queue = [_queue_item(item, trader_request=self.trader_request) for item in requests]
        queue.sort(key=_queue_sort_key)
        actionable = [item for item in queue if item["automation_status"] == "READY_FOR_CODEX_IMPLEMENTATION"]
        approval_required = [item for item in queue if item["automation_status"] == "WAITING_FOR_OPERATOR_APPROVAL"]
        selected = _select_request(actionable, trader_request=self.trader_request)
        if actionable:
            status = STATUS_READY
        elif approval_required:
            status = STATUS_APPROVAL_REQUIRED
        elif requests:
            status = STATUS_BLOCKED
        else:
            status = STATUS_NO_REQUESTS
        payload = {
            "contract_version": CONTRACT_VERSION,
            "generated_at_utc": self.now_utc.isoformat(),
            "status": status,
            "trader_request": self.trader_request,
            "artifact_paths": {
                "trader_support_bot": str(self.support_bot_path),
                "output": str(self.output_path),
                "report": str(self.report_path),
            },
            "selected_request": selected,
            "queue": queue,
            "actionable_requests": actionable,
            "approval_required_requests": approval_required,
            "metrics": {
                "repair_request_count": len(queue),
                "actionable_request_count": len(actionable),
                "approval_required_request_count": len(approval_required),
                "selected_request_code": selected.get("code") if selected else None,
                "support_status": support.get("status"),
                "repair_request_source": request_source,
                "recovered_from_embedded_support": recovered_from_embedded_support,
            },
            "execution_contract": _execution_contract(),
            "read_only": True,
            "live_side_effects": [],
        }
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path), "repair_requests": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _execution_contract() -> dict[str, Any]:
    return {
        "codex_may_execute": list(REPAIR_AUTOMATION_ALLOWED_ACTIONS),
        "code_change_loop": [
            "read_artifacts",
            "inspect_suggested_files",
            "edit_code",
            "edit_tests",
            "edit_runtime_contract_docs",
            "run_targeted_tests",
            "run_required_verification_commands",
            "commit",
            "sync_live_runtime",
        ],
        "commit_and_live_sync_required": True,
        "report_artifacts_are_runtime_drift": True,
        "quant_rabbit_code_may_call_model_api": False,
        "live_side_effects_allowed": [],
        "requires_explicit_operator_approval_for": list(REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS),
        "forbidden_direct_actions": list(REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS),
        "orders_closes_launchd_policy": (
            "Order send, order cancel, position close, and launchd load/reload must go through "
            "explicit operator approval or the existing gateway path. This orchestrator is never live permission."
        ),
    }


def _queue_item(request: dict[str, Any], *, trader_request: str) -> dict[str, Any]:
    contract = request.get("automation_contract") if isinstance(request.get("automation_contract"), dict) else {}
    explicit = bool(request.get("requires_explicit_operator_approval"))
    allowed_actions = [str(item) for item in contract.get("codex_may_execute", []) or []]
    approval_actions = [str(item) for item in contract.get("requires_explicit_operator_approval_for", []) or []]
    forbidden_actions = [str(item) for item in contract.get("forbidden_direct_actions", []) or []]
    if not allowed_actions:
        allowed_actions = list(REPAIR_AUTOMATION_ALLOWED_ACTIONS)
    if not approval_actions:
        approval_actions = list(REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS)
    if not forbidden_actions:
        forbidden_actions = list(REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS)

    automation_status = "WAITING_FOR_OPERATOR_APPROVAL" if explicit else "READY_FOR_CODEX_IMPLEMENTATION"
    verification_commands = _dedupe_strings(request.get("verification_commands"))
    targeted_tests = _targeted_test_commands(request.get("suggested_files"))
    return {
        "code": request.get("code"),
        "priority": request.get("priority"),
        "source_findings": _dedupe_strings(request.get("source_findings")),
        "repair_status": request.get("status"),
        "automation_status": automation_status,
        "match_score": _match_score(request, trader_request),
        "requires_explicit_operator_approval": explicit,
        "problem": request.get("problem"),
        "why_now": request.get("why_now"),
        "clearance_conditions": _dedupe_strings(request.get("clearance_conditions")),
        "verification_commands": verification_commands,
        "suggested_files": _dedupe_strings(request.get("suggested_files")),
        "required_tests": _dedupe_strings(request.get("required_tests")),
        "targeted_test_commands": targeted_tests,
        "final_verification_commands": _dedupe_strings(
            [
                *targeted_tests,
                *verification_commands,
                "PYTHONPATH=src python3 -m unittest discover -s tests -v",
                "git status --short",
                "scripts/sync-live-runtime.sh",
            ]
        ),
        "implementation_loop": (
            [
                "read_support_artifacts",
                "inspect_suggested_files",
                "implement_code_and_tests",
                "run_targeted_tests",
                "run_required_verification_commands",
                "commit_with_codex_attribution",
                "sync_live_runtime",
                "verify_live_head_matches_promoted_commit",
            ]
            if not explicit
            else [
                "run_preflight_read_only_checks",
                "wait_for_explicit_operator_approval",
                "execute_only_the_approved_gateway_or_launchd_action",
                "refresh_support_artifacts",
            ]
        ),
        "automation_contract": {
            "codex_may_execute": allowed_actions,
            "commit_and_live_sync_required": bool(contract.get("commit_and_live_sync_required", True)),
            "quant_rabbit_code_may_call_model_api": False,
            "live_side_effects_allowed": [],
            "requires_explicit_operator_approval_for": approval_actions,
            "forbidden_direct_actions": forbidden_actions,
            "orders_closes_launchd_policy": contract.get("orders_closes_launchd_policy")
            or _execution_contract()["orders_closes_launchd_policy"],
        },
        "read_only": True,
        "live_side_effects": [],
    }


def _select_request(queue: list[dict[str, Any]], *, trader_request: str) -> dict[str, Any]:
    if not queue:
        return {}
    if trader_request.strip():
        return max(queue, key=lambda item: (int(item.get("match_score") or 0), -_priority_rank(item.get("priority"))))
    return queue[0]


def _queue_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    approval_rank = 1 if item.get("requires_explicit_operator_approval") else 0
    priority_rank = _priority_rank(item.get("priority"))
    return (
        approval_rank,
        priority_rank,
        -int(item.get("match_score") or 0),
        str(item.get("code") or ""),
    )


def _priority_rank(value: Any) -> int:
    text = str(value or "").upper()
    if text == "P0":
        return 0
    if text == "P1":
        return 1
    if text == "P2":
        return 2
    return 9


def _match_score(request: dict[str, Any], trader_request: str) -> int:
    terms = _terms(trader_request)
    if not terms:
        return 0
    haystack = " ".join(
        str(part)
        for part in [
            request.get("code"),
            request.get("priority"),
            request.get("status"),
            request.get("problem"),
            request.get("why_now"),
            " ".join(_dedupe_strings(request.get("source_findings"))),
            " ".join(_dedupe_strings(request.get("suggested_files"))),
        ]
        if part
    ).lower()
    return sum(1 for term in terms if term in haystack)


def _terms(text: str) -> list[str]:
    raw = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9_./:-]+|[ぁ-んァ-ン一-龯]+", raw)
    synonyms = {
        "利益": "profit",
        "利確": "profit",
        "決済": "close",
        "修正": "repair",
        "証拠": "evidence",
        "検証": "verification",
        "ガーディアン": "guardian",
    }
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        mapped = synonyms.get(token)
        if mapped:
            expanded.append(mapped)
    return list(dict.fromkeys(expanded))


def _dedupe_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = [value]
    return list(dict.fromkeys(str(item) for item in raw if str(item)))


def _targeted_test_commands(files: Any) -> list[str]:
    commands: list[str] = []
    for item in _dedupe_strings(files):
        path = Path(item)
        if path.parts and path.parts[0] == "tests" and path.suffix == ".py":
            module = ".".join(path.with_suffix("").parts)
            commands.append(f"PYTHONPATH=src python3 -m unittest {module} -v")
    return list(dict.fromkeys(commands))


def _render_report(payload: dict[str, Any]) -> str:
    selected = payload.get("selected_request") if isinstance(payload.get("selected_request"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    contract = payload.get("execution_contract") if isinstance(payload.get("execution_contract"), dict) else {}
    lines = [
        "# Trader Repair Orchestrator Report",
        "",
        f"- Generated at UTC: `{payload.get('generated_at_utc')}`",
        f"- Status: `{payload.get('status')}`",
        f"- Trader request: `{payload.get('trader_request') or ''}`",
        f"- Selected request: `{selected.get('code') if selected else None}`",
        f"- Actionable requests: `{metrics.get('actionable_request_count')}`",
        f"- Approval-required requests: `{metrics.get('approval_required_request_count')}`",
        f"- Repair request source: `{metrics.get('repair_request_source')}`",
        f"- Recovered from embedded support: `{metrics.get('recovered_from_embedded_support')}`",
        f"- Read only: `{payload.get('read_only')}`",
        f"- Live side effects: `{len(payload.get('live_side_effects') or [])}`",
        "",
        "## Execution Contract",
        "",
        f"- Codex may execute: `{', '.join(contract.get('codex_may_execute') or [])}`",
        f"- Approval required for: `{', '.join(contract.get('requires_explicit_operator_approval_for') or [])}`",
        f"- Forbidden direct actions: `{', '.join(contract.get('forbidden_direct_actions') or [])}`",
        f"- Commit and live sync required: `{contract.get('commit_and_live_sync_required')}`",
        f"- QuantRabbit code may call model API: `{contract.get('quant_rabbit_code_may_call_model_api')}`",
        f"- Policy: {contract.get('orders_closes_launchd_policy')}",
        "",
        "## Queue",
        "",
    ]
    queue = payload.get("queue") if isinstance(payload.get("queue"), list) else []
    if queue:
        lines.extend(
            [
                "| Code | Priority | Automation | Match | Verify |",
                "|---|---|---|---:|---|",
            ]
        )
        for item in queue[:12]:
            verify = ", ".join(f"`{command}`" for command in item.get("verification_commands", [])[:2]) or "none"
            lines.append(
                f"| `{item.get('code')}` | `{item.get('priority')}` | "
                f"`{item.get('automation_status')}` | `{item.get('match_score')}` | {verify} |"
            )
    else:
        lines.append("- none")
    if selected:
        lines.extend(
            [
                "",
                "## Selected Request",
                "",
                f"- Code: `{selected.get('code')}`",
                f"- Problem: {selected.get('problem')}",
                f"- Why now: {selected.get('why_now')}",
                f"- Suggested files: `{', '.join(selected.get('suggested_files') or [])}`",
                f"- Targeted tests: `{', '.join(selected.get('targeted_test_commands') or [])}`",
                f"- Final verification: `{', '.join(selected.get('final_verification_commands') or [])}`",
            ]
        )
    return "\n".join(lines) + "\n"
