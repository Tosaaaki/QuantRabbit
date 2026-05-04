from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_COMPLETION_STATUS,
    DEFAULT_COMPLETION_STATUS_REPORT,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_DRY_RUN_CERTIFICATION,
    DEFAULT_EXECUTION_REPLAY,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_REPLAY_BACKTEST,
)


@dataclass(frozen=True)
class CompletionStatusSummary:
    output_path: Path
    report_path: Path
    status: str
    blockers: int
    next_actions: int
    live_ready_lanes: int
    remaining_target_jpy: float


class CompletionAuditor:
    """Summarize what still blocks QuantRabbit completion."""

    def __init__(
        self,
        *,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        coverage_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        replay_backtest_path: Path = DEFAULT_REPLAY_BACKTEST,
        execution_replay_path: Path = DEFAULT_EXECUTION_REPLAY,
        dry_run_certification_path: Path = DEFAULT_DRY_RUN_CERTIFICATION,
        live_order_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        output_path: Path = DEFAULT_COMPLETION_STATUS,
        report_path: Path = DEFAULT_COMPLETION_STATUS_REPORT,
    ) -> None:
        self.broker_snapshot_path = broker_snapshot_path
        self.order_intents_path = order_intents_path
        self.target_state_path = target_state_path
        self.coverage_path = coverage_path
        self.replay_backtest_path = replay_backtest_path
        self.execution_replay_path = execution_replay_path
        self.dry_run_certification_path = dry_run_certification_path
        self.live_order_path = live_order_path
        self.output_path = output_path
        self.report_path = report_path

    def run(self) -> CompletionStatusSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        broker = _load_json(self.broker_snapshot_path)
        intents = _load_json(self.order_intents_path)
        target = _load_json(self.target_state_path)
        coverage = _load_json(self.coverage_path)
        replay = _load_json(self.replay_backtest_path)
        execution = _load_json(self.execution_replay_path)
        certification = _load_json(self.dry_run_certification_path)
        live_order = _load_json(self.live_order_path)

        positions = broker.get("positions", []) if isinstance(broker.get("positions"), list) else []
        pending_entries = _pending_entries(broker)
        live_ready = _live_ready_lanes(intents)
        remaining_target = _remaining_target(target, coverage)
        coverage_stale = _coverage_is_stale(intents, coverage, live_ready)
        blockers = _blockers(
            broker=broker,
            positions=positions,
            pending_entries=pending_entries,
            live_ready=live_ready,
            remaining_target=remaining_target,
            coverage=coverage,
            coverage_stale=coverage_stale,
            replay=replay,
            execution=execution,
            certification=certification,
            live_order=live_order,
        )
        next_actions = _next_actions(
            positions=positions,
            pending_entries=pending_entries,
            live_ready=live_ready,
            coverage=coverage,
            coverage_stale=coverage_stale,
            execution=execution,
            live_order=live_order,
        )
        status = "COMPLETE" if not blockers else "BLOCKED"
        payload = {
            "generated_at_utc": generated_at,
            "status": status,
            "artifact_paths": {
                "broker_snapshot": str(self.broker_snapshot_path),
                "order_intents": str(self.order_intents_path),
                "target_state": str(self.target_state_path),
                "coverage": str(self.coverage_path),
                "replay_backtest": str(self.replay_backtest_path),
                "execution_replay": str(self.execution_replay_path),
                "dry_run_certification": str(self.dry_run_certification_path),
                "live_order": str(self.live_order_path),
            },
            "broker": {
                "positions": len(positions),
                "pending_entries": len(pending_entries),
                "orders": len(broker.get("orders", []) or []) if broker else 0,
            },
            "target": {
                "status": target.get("status"),
                "remaining_target_jpy": remaining_target,
                "remaining_risk_budget_jpy": _optional_float(target.get("remaining_risk_budget_jpy")),
            },
            "coverage": {
                "status": coverage.get("status"),
                "stale": coverage_stale,
                "live_ready_reward_jpy": _optional_float(coverage.get("live_ready_reward_jpy")) or 0.0,
                "potential_reward_jpy": _optional_float(coverage.get("potential_reward_jpy")) or 0.0,
            },
            "replay": {
                "historical_target_hits": _nested_int(replay, "summary", "historical_target_hits"),
                "evidence_target_covered": _nested_int(replay, "summary", "evidence_target_covered"),
                "days": _nested_int(replay, "summary", "days"),
            },
            "live_ready_lanes": live_ready,
            "blockers": blockers,
            "next_actions": next_actions,
        }
        self._write_output(payload)
        self._write_report(payload)
        return CompletionStatusSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            blockers=len(blockers),
            next_actions=len(next_actions),
            live_ready_lanes=live_ready,
            remaining_target_jpy=remaining_target,
        )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Completion Status Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Open positions: `{payload['broker']['positions']}`",
            f"- Pending entry orders: `{payload['broker']['pending_entries']}`",
            f"- Remaining target: `{payload['target']['remaining_target_jpy']:.0f} JPY`",
            f"- Live-ready lanes: `{payload['live_ready_lanes']}`",
            f"- Coverage status: `{payload['coverage']['status']}`",
            "",
            "## Blockers",
            "",
        ]
        if payload["blockers"]:
            for item in payload["blockers"]:
                lines.append(f"- `{item['code']}` {item['message']}")
        else:
            lines.append("- none")
        lines.extend(["", "## Next Actions", ""])
        if payload["next_actions"]:
            for item in payload["next_actions"]:
                lines.append(f"- `{item['code']}` {item['message']}")
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Completion Contract",
                "",
                "- Completion requires broker truth, live-ready coverage, execution replay, learning receipts, and dry-run certification to pass together.",
                "- Only unprotected, external/manual, non-trader, over-budget, or pending-entry exposure blocks fresh entries.",
                "- Protected trader-owned exposure may add only through portfolio risk validation.",
                "- The 10% daily target remains a risk-bounded product KPI, not permission to force trades.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _blockers(
    *,
    broker: dict[str, Any],
    positions: list[Any],
    pending_entries: list[dict[str, Any]],
    live_ready: int,
    remaining_target: float,
    coverage: dict[str, Any],
    coverage_stale: bool,
    replay: dict[str, Any],
    execution: dict[str, Any],
    certification: dict[str, Any],
    live_order: dict[str, Any],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not broker:
        blockers.append(_item("BROKER_SNAPSHOT_MISSING", "broker snapshot is missing; run broker-snapshot"))
    blocking_positions = [item for item in positions if isinstance(item, dict) and not _is_layerable_position(item)]
    if blocking_positions:
        summaries = ", ".join(
            f"{item.get('pair')} {item.get('side')} id={item.get('trade_id')}" for item in blocking_positions[:3]
        )
        blockers.append(
            _item(
                "BROKER_EXPOSURE_OPEN",
                f"unprotected, external, or non-trader broker exposure blocks fresh entries: {summaries}",
            )
        )
    if pending_entries:
        summaries = ", ".join(
            f"{item.get('pair')} {item.get('order_type')} id={item.get('order_id')}" for item in pending_entries[:3]
        )
        blockers.append(_item("PENDING_ENTRY_OPEN", f"pending entry orders must be resolved before fresh entries: {summaries}"))
    if remaining_target > 0 and live_ready <= 0:
        blockers.append(_item("NO_LIVE_READY_INTENTS", "no LIVE_READY order intents are available"))
    if coverage_stale:
        blockers.append(
            _item(
                "COVERAGE_STALE",
                "coverage optimization is stale versus current order intents; rerun optimize-coverage before using its blockers",
            )
        )
    else:
        for message in coverage.get("blockers", []) or []:
            blockers.append(_item("COVERAGE_BLOCKER", str(message)))
    replay_summary = replay.get("summary") if isinstance(replay.get("summary"), dict) else {}
    if replay_summary:
        days = int(replay_summary.get("days") or 0)
        covered = int(replay_summary.get("evidence_target_covered") or 0)
        if days and covered < days:
            blockers.append(_item("REPLAY_COVERAGE_GAP", f"legacy replay covers target on {covered}/{days} days"))
    else:
        blockers.append(_item("REPLAY_BACKTEST_MISSING", "replay-backtest receipt is missing"))
    if not execution:
        blockers.append(_item("EXECUTION_REPLAY_MISSING", "execution replay receipt is missing"))
    elif execution.get("status") == "BLOCKED":
        blockers.append(_item("EXECUTION_REPLAY_BLOCKED", "; ".join(str(item) for item in execution.get("blockers", []) or [])))
    if live_order.get("sent") is True or live_order.get("send_requested") is True:
        blockers.append(_item("LIVE_SEND_ARTIFACT_PRESENT", "latest live-order artifact records a send; dry-run certification needs a fresh no-send stage receipt or an archived live audit path"))
    if certification.get("status") != "CERTIFIED":
        for message in certification.get("blockers", []) or ["dry-run certification has not passed"]:
            blockers.append(_item("CERTIFICATION_BLOCKER", str(message)))
    return _dedupe(blockers)


def _next_actions(
    *,
    positions: list[Any],
    pending_entries: list[dict[str, Any]],
    live_ready: int,
    coverage: dict[str, Any],
    coverage_stale: bool,
    execution: dict[str, Any],
    live_order: dict[str, Any],
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    blocking_positions = [item for item in positions if isinstance(item, dict) and not _is_layerable_position(item)]
    if blocking_positions:
        actions.append(
            _item(
                "MANAGE_OPEN_EXPOSURE",
                "let PositionManager/PositionProtectionGateway hold, tighten, close, or repair blocking exposure before new entries",
            )
        )
    if pending_entries:
        actions.append(_item("RESOLVE_PENDING_ENTRIES", "cancel or adopt stale pending entries through the autotrade gateway path"))
    if live_ready <= 0:
        actions.append(_item("BUILD_LIVE_READY_RECEIPTS", "generate risk-valid receipts after broker exposure is flat; promote only receipts that clear risk/profile blockers"))
    if coverage_stale:
        actions.append(_item("RUN_COVERAGE_OPTIMIZATION", "rerun optimize-coverage against the current order intents"))
    else:
        for message in coverage.get("action_items", []) or []:
            actions.append(_item("COVERAGE_ACTION", str(message)))
    if not execution:
        actions.append(_item("RUN_EXECUTION_REPLAY", "run replay-execution with a quote path after at least one LIVE_READY intent exists"))
    if live_order.get("sent") is True or live_order.get("send_requested") is True:
        actions.append(_item("REFRESH_NO_SEND_STAGE", "preserve the live audit, then create a fresh no-send live-order stage artifact for dry-run certification"))
    actions.append(_item("RERUN_CERTIFICATION", "rerun certify-dry-run after coverage, replay, learning, and no-send artifacts pass"))
    return _dedupe(actions)


def _pending_entries(broker: dict[str, Any]) -> list[dict[str, Any]]:
    pending_types = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}
    orders = broker.get("orders", []) if isinstance(broker.get("orders"), list) else []
    return [
        item
        for item in orders
        if isinstance(item, dict)
        and not item.get("trade_id")
        and str(item.get("order_type") or "").upper() in pending_types
    ]


def _is_layerable_position(position: dict[str, Any]) -> bool:
    return (
        str(position.get("owner") or "") == "trader"
        and position.get("take_profit") is not None
        and position.get("stop_loss") is not None
    )


def _live_ready_lanes(intents: dict[str, Any]) -> int:
    return sum(1 for item in intents.get("results", []) or [] if isinstance(item, dict) and item.get("status") == "LIVE_READY")


def _coverage_is_stale(intents: dict[str, Any], coverage: dict[str, Any], live_ready: int) -> bool:
    if not coverage:
        return False
    coverage_lanes = coverage.get("lanes", [])
    if isinstance(coverage_lanes, list) and coverage_lanes:
        coverage_signature = {
            (
                str(lane.get("lane_id") or ""),
                str(lane.get("status") or ""),
                lane.get("counts_live_ready") is True
                or (lane.get("status") == "LIVE_READY" and not lane.get("blockers")),
            )
            for lane in coverage_lanes
            if isinstance(lane, dict)
        }
        intent_signature = {
            (
                str(item.get("lane_id") or ""),
                str(item.get("status") or ""),
                item.get("status") == "LIVE_READY" and not _intent_result_blockers(item),
            )
            for item in intents.get("results", []) or []
            if isinstance(item, dict) and isinstance(item.get("intent"), dict)
        }
        if coverage_signature != intent_signature:
            return True
        coverage_live_ready = sum(
            1
            for lane in coverage_lanes
            if isinstance(lane, dict)
            and (
                lane.get("counts_live_ready") is True
                or (lane.get("status") == "LIVE_READY" and not lane.get("blockers"))
            )
        )
        if coverage_live_ready != live_ready:
            return True
    return False


def _intent_result_blockers(result: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for issue in result.get("risk_issues", []) or []:
        if isinstance(issue, dict) and issue.get("severity") == "BLOCK":
            blockers.append(str(issue.get("message") or issue.get("code") or "risk block"))
    for issue in result.get("strategy_issues", []) or []:
        if isinstance(issue, dict) and issue.get("severity") == "BLOCK":
            blockers.append(str(issue.get("message") or issue.get("code") or "strategy block"))
    blockers.extend(str(item) for item in result.get("live_blockers", []) or [])
    return blockers


def _remaining_target(target: dict[str, Any], coverage: dict[str, Any]) -> float:
    for payload in (target, coverage):
        value = _optional_float(payload.get("remaining_target_jpy"))
        if value is not None:
            return value
    return 0.0


def _nested_int(payload: dict[str, Any], parent: str, key: str) -> int | None:
    nested = payload.get(parent) if isinstance(payload.get(parent), dict) else {}
    value = nested.get(key)
    return int(value) if value is not None else None


def _item(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _dedupe(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = (item["code"], item["message"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
