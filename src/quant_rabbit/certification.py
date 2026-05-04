from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_DRY_RUN_CERTIFICATION,
    DEFAULT_DRY_RUN_CERTIFICATION_REPORT,
    DEFAULT_EXECUTION_REPLAY,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_POST_TRADE_LEARNING,
)


@dataclass(frozen=True)
class DryRunCertificationSummary:
    output_path: Path
    report_path: Path
    status: str
    blockers: int
    checks: int


class DryRunCertifier:
    """Certify that dry-run artifacts are complete enough before live expansion."""

    def __init__(
        self,
        *,
        coverage_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        execution_replay_path: Path = DEFAULT_EXECUTION_REPLAY,
        post_trade_learning_path: Path = DEFAULT_POST_TRADE_LEARNING,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        live_order_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        position_execution_path: Path = DEFAULT_POSITION_EXECUTION,
        gpt_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
        output_path: Path = DEFAULT_DRY_RUN_CERTIFICATION,
        report_path: Path = DEFAULT_DRY_RUN_CERTIFICATION_REPORT,
    ) -> None:
        self.coverage_path = coverage_path
        self.execution_replay_path = execution_replay_path
        self.post_trade_learning_path = post_trade_learning_path
        self.order_intents_path = order_intents_path
        self.live_order_path = live_order_path
        self.position_execution_path = position_execution_path
        self.gpt_decision_path = gpt_decision_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self) -> DryRunCertificationSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        artifacts = {
            "coverage": _load_json(self.coverage_path),
            "execution_replay": _load_json(self.execution_replay_path),
            "post_trade_learning": _load_json(self.post_trade_learning_path),
            "order_intents": _load_json(self.order_intents_path),
            "live_order": _load_json(self.live_order_path),
            "position_execution": _load_json(self.position_execution_path),
            "gpt_decision": _load_json(self.gpt_decision_path),
        }
        checks, blockers = _checks(artifacts)
        payload = {
            "generated_at_utc": generated_at,
            "status": "CERTIFIED" if not blockers else "BLOCKED",
            "artifact_paths": {
                "coverage": str(self.coverage_path),
                "execution_replay": str(self.execution_replay_path),
                "post_trade_learning": str(self.post_trade_learning_path),
                "order_intents": str(self.order_intents_path),
                "live_order": str(self.live_order_path),
                "position_execution": str(self.position_execution_path),
                "gpt_decision": str(self.gpt_decision_path),
            },
            "checks": checks,
            "blockers": blockers,
        }
        self._write_output(payload)
        self._write_report(payload)
        return DryRunCertificationSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=payload["status"],
            blockers=len(blockers),
            checks=len(checks),
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Dry-Run Certification Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Checks: `{len(payload['checks'])}`",
            f"- Blockers: `{len(payload['blockers'])}`",
            "",
            "## Blockers",
            "",
        ]
        if payload["blockers"]:
            lines.extend(f"- {item}" for item in payload["blockers"])
        else:
            lines.append("- none")
        lines.extend(["", "## Checks", ""])
        for item in payload["checks"]:
            lines.append(f"- `{item['status']}` {item['name']}: {item['detail']}")
        lines.extend(
            [
                "",
                "## Certification Contract",
                "",
                "- Certification is dry-run only and does not enable live trading.",
                "- Any artifact showing a live send blocks certification.",
                "- Coverage, replay, and learning receipts must exist before live expansion.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _checks(artifacts: dict[str, dict[str, Any]]) -> tuple[list[dict[str, str]], list[str]]:
    checks: list[dict[str, str]] = []
    blockers: list[str] = []

    coverage = artifacts["coverage"]
    if not coverage:
        _block(checks, blockers, "coverage", "coverage optimization receipt is missing")
    elif coverage.get("blockers"):
        _block(checks, blockers, "coverage", "coverage optimization still has blockers")
    elif coverage.get("status") in {"LIVE_READY_COVERAGE_READY", "TARGET_REACHED_PROTECT"}:
        _pass(checks, "coverage", f"coverage status {coverage.get('status')}")
    else:
        _block(checks, blockers, "coverage", f"coverage status {coverage.get('status')} is not certifiable")

    replay = artifacts["execution_replay"]
    if not replay:
        _block(checks, blockers, "execution_replay", "execution replay receipt is missing")
    elif replay.get("target_jpy") and not replay.get("target_hit"):
        _block(checks, blockers, "execution_replay", "execution replay did not hit target")
    elif replay.get("status") == "BLOCKED":
        _block(checks, blockers, "execution_replay", "execution replay is blocked")
    else:
        _pass(checks, "execution_replay", f"replay status {replay.get('status')}")

    learning = artifacts["post_trade_learning"]
    if not learning:
        _block(checks, blockers, "post_trade_learning", "post-trade learning receipt is missing")
    elif learning.get("status") == "BLOCKED":
        _block(checks, blockers, "post_trade_learning", "post-trade learning has blockers")
    else:
        _pass(checks, "post_trade_learning", f"learning status {learning.get('status')}")

    intents = artifacts["order_intents"]
    live_ready = sum(1 for item in intents.get("results", []) or [] if isinstance(item, dict) and item.get("status") == "LIVE_READY")
    if live_ready <= 0:
        _block(checks, blockers, "order_intents", "no LIVE_READY intents are available")
    elif not _intents_have_contracts(intents):
        _block(checks, blockers, "order_intents", "one or more LIVE_READY intents lack thesis or market_context")
    else:
        _pass(checks, "order_intents", f"{live_ready} LIVE_READY intents have required contracts")

    live_order = artifacts["live_order"]
    if live_order.get("sent") is True or live_order.get("send_requested") is True:
        _block(checks, blockers, "live_order", "dry-run certification found an entry send request or send")
    else:
        _pass(checks, "live_order", "no entry send was requested")

    position_execution = artifacts["position_execution"]
    if position_execution.get("sent") is True or position_execution.get("send_requested") is True:
        _block(checks, blockers, "position_execution", "dry-run certification found a position write request or send")
    else:
        _pass(checks, "position_execution", "no position write was requested")

    gpt = artifacts["gpt_decision"]
    if gpt and gpt.get("status") == "REJECTED":
        _block(checks, blockers, "gpt_decision", "latest GPT trader decision was rejected")
    elif gpt:
        _pass(checks, "gpt_decision", f"GPT status {gpt.get('status')}")
    else:
        _pass(checks, "gpt_decision", "no GPT decision required for this dry-run certification")

    return checks, blockers


def _intents_have_contracts(intents: dict[str, Any]) -> bool:
    for item in intents.get("results", []) or []:
        if not isinstance(item, dict) or item.get("status") != "LIVE_READY":
            continue
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
        if not str(intent.get("thesis") or "").strip():
            return False
        for key in ("regime", "narrative", "chart_story", "method", "invalidation"):
            if not str(context.get(key) or "").strip():
                return False
        metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
        for key in ("risk_jpy", "reward_jpy", "reward_risk", "spread_pips"):
            if metrics.get(key) is None:
                return False
    return True


def _pass(checks: list[dict[str, str]], name: str, detail: str) -> None:
    checks.append({"name": name, "status": "PASS", "detail": detail})


def _block(checks: list[dict[str, str]], blockers: list[str], name: str, detail: str) -> None:
    checks.append({"name": name, "status": "BLOCK", "detail": detail})
    blockers.append(detail)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
