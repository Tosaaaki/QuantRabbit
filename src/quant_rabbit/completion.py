from __future__ import annotations

import json
import os
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
        close_recommendations = _fresh_close_recommendations_for_completion(
            broker,
            data_root=self.broker_snapshot_path.parent,
        )
        close_authorization_fresh = _close_authorization_fresh(
            data_root=self.broker_snapshot_path.parent,
        )
        close_authorization_available = close_authorization_fresh or _standing_close_recommendation_present(
            close_recommendations
        )
        remaining_target = _remaining_target(target, coverage)
        coverage_stale = _coverage_is_stale(intents, coverage, live_ready)
        certification_stale = _certification_is_stale(
            certification,
            dependencies=(coverage, intents, execution, live_order),
        )
        blockers = _blockers(
            broker=broker,
            positions=positions,
            pending_entries=pending_entries,
            live_ready=live_ready,
            close_recommendations=close_recommendations,
            close_authorization_fresh=close_authorization_fresh,
            remaining_target=remaining_target,
            coverage=coverage,
            coverage_stale=coverage_stale,
            replay=replay,
            execution=execution,
            certification=certification,
            certification_stale=certification_stale,
            live_order=live_order,
        )
        next_actions = _next_actions(
            positions=positions,
            pending_entries=pending_entries,
            live_ready=live_ready,
            remaining_target=remaining_target,
            close_recommendations=close_recommendations,
            close_authorization_fresh=close_authorization_fresh,
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
            "certification": {
                "status": certification.get("status"),
                "stale": certification_stale,
            },
            "replay": {
                "historical_target_hits": _nested_int(replay, "summary", "historical_target_hits"),
                "evidence_target_covered": _nested_int(replay, "summary", "evidence_target_covered"),
                "days": _nested_int(replay, "summary", "days"),
            },
            "live_ready_lanes": live_ready,
            "close_recommendations": {
                "count": len(close_recommendations),
                "gate_b_authorized": close_authorization_available,
                "explicit_gate_b_authorized": close_authorization_fresh,
                "items": close_recommendations,
            },
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
            f"- Close recommendations: `{payload['close_recommendations']['count']}`",
            f"- Close authorization available: `{payload['close_recommendations']['gate_b_authorized']}`",
            f"- Explicit Gate B authorized: `{payload['close_recommendations']['explicit_gate_b_authorized']}`",
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
                "- Only unprotected trader-owned, external, or over-budget exposure blocks fresh entries; manual/tagless operator exposure is TP-managed only.",
                "- Protected trader-owned exposure and trader-owned pending entries may add only through basket portfolio risk validation.",
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
    close_recommendations: list[dict[str, Any]],
    close_authorization_fresh: bool,
    remaining_target: float,
    coverage: dict[str, Any],
    coverage_stale: bool,
    replay: dict[str, Any],
    execution: dict[str, Any],
    certification: dict[str, Any],
    certification_stale: bool,
    live_order: dict[str, Any],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not broker:
        blockers.append(_item("BROKER_SNAPSHOT_MISSING", "broker snapshot is missing; run broker-snapshot"))
    blocking_positions = [
        item
        for item in positions
        if isinstance(item, dict) and not _is_operator_managed_manual(item) and not _is_layerable_position(item)
    ]
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
    blocking_pending_entries = _blocking_pending_entries(pending_entries, remaining_target=remaining_target)
    if blocking_pending_entries:
        summaries = ", ".join(
            f"{item.get('pair')} {item.get('order_type')} id={item.get('order_id')}"
            for item in blocking_pending_entries[:3]
        )
        blockers.append(
            _item(
                "PENDING_ENTRY_BLOCKED",
                "pending entry orders must be resolved before fresh entries: " + summaries,
            )
        )
    if remaining_target > 0 and live_ready <= 0:
        blockers.append(_item("NO_LIVE_READY_INTENTS", "no LIVE_READY order intents are available"))
    if close_recommendations:
        summaries = ", ".join(
            f"{item.get('pair')} {item.get('side')} id={item.get('trade_id')} {item.get('source')}"
            for item in close_recommendations[:3]
        )
        if close_authorization_fresh or _standing_close_recommendation_present(close_recommendations):
            blockers.append(
                _item(
                    "CLOSE_RECEIPT_REQUIRED",
                    "fresh hard Gate A close recommendation(s) or Gate B authorization exist; submit a verified CLOSE receipt before completion: "
                    f"{summaries}",
                )
            )
        else:
            blockers.append(
                _item(
                    "CLOSE_AUTHORIZATION_REQUIRED",
                    "fresh Gate A close recommendation(s) exist but Gate B is not authorized: "
                    f"{summaries}",
                )
            )
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
    if certification_stale:
        blockers.append(
            _item(
                "CERTIFICATION_STALE",
                "dry-run certification is older than current coverage, order-intents, execution replay, or live-order artifacts",
            )
        )
    if certification.get("status") != "CERTIFIED":
        for message in certification.get("blockers", []) or ["dry-run certification has not passed"]:
            blockers.append(_item("CERTIFICATION_BLOCKER", str(message)))
    return _dedupe(blockers)


def _next_actions(
    *,
    positions: list[Any],
    pending_entries: list[dict[str, Any]],
    live_ready: int,
    remaining_target: float,
    close_recommendations: list[dict[str, Any]],
    close_authorization_fresh: bool,
    coverage: dict[str, Any],
    coverage_stale: bool,
    execution: dict[str, Any],
    live_order: dict[str, Any],
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    blocking_positions = [
        item
        for item in positions
        if isinstance(item, dict) and not _is_operator_managed_manual(item) and not _is_layerable_position(item)
    ]
    if blocking_positions:
        actions.append(
            _item(
                "MANAGE_OPEN_EXPOSURE",
                "let PositionManager/PositionProtectionGateway hold, tighten, close, or repair blocking exposure before new entries",
            )
        )
    if pending_entries:
        blocking_pending_entries = _blocking_pending_entries(pending_entries, remaining_target=remaining_target)
        if blocking_pending_entries:
            actions.append(
                _item("RESOLVE_PENDING_ENTRIES", "cancel or adopt stale pending entries through the autotrade gateway path")
            )
        else:
            actions.append(
                _item(
                    "BASKET_VALIDATE_PENDING_ENTRIES",
                    "keep compatible trader-owned pending entries and count them in LiveOrderGateway basket risk/margin validation",
                )
            )
    if live_ready <= 0:
        actions.append(_item("BUILD_LIVE_READY_RECEIPTS", "generate risk-valid receipts after broker exposure is flat; promote only receipts that clear risk/profile blockers"))
    if close_recommendations:
        if close_authorization_fresh or _standing_close_recommendation_present(close_recommendations):
            actions.append(
                _item(
                    "SUBMIT_VERIFIED_CLOSE_RECEIPT",
                    "hard Gate A close evidence or Gate B authorization is present; submit a CLOSE receipt before re-entry",
                )
            )
        else:
            actions.append(
                _item(
                    "AUTHORIZE_OR_REJECT_CLOSE_RECOMMENDATIONS",
                    "operator must either create a fresh close token/override for the recommended loss-cut batch or record a reason to keep holding",
                )
            )
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


def _fresh_close_recommendations_for_completion(
    broker: dict[str, Any],
    *,
    data_root: Path,
) -> list[dict[str, Any]]:
    if not broker:
        return []
    try:
        from quant_rabbit.trader_prompts import _fresh_close_recommendations

        return [dict(item) for item in _fresh_close_recommendations(broker, data_root=data_root)]
    except Exception:
        return []


def _close_authorization_fresh(*, data_root: Path) -> bool:
    try:
        from quant_rabbit.gpt_trader import _operator_close_override_active, _operator_close_token_fresh

        return bool(_operator_close_override_active() or _operator_close_token_fresh(data_root=data_root))
    except Exception:
        return False


def _standing_close_recommendation_present(close_recommendations: list[dict[str, Any]]) -> bool:
    try:
        from quant_rabbit.gpt_trader import _sidecar_close_standing_authorized

        return any(_sidecar_close_standing_authorized(item) for item in close_recommendations)
    except Exception:
        return any(item.get("gate_b_standing_authorized") is True for item in close_recommendations)


def _pending_entries(broker: dict[str, Any]) -> list[dict[str, Any]]:
    pending_types = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}
    orders = broker.get("orders", []) if isinstance(broker.get("orders"), list) else []
    return [
        item
        for item in orders
        if isinstance(item, dict)
        and not item.get("trade_id")
        and not _is_operator_managed_manual(item)
        and str(item.get("order_type") or "").upper() in pending_types
    ]


def _blocking_pending_entries(pending_entries: list[dict[str, Any]], *, remaining_target: float) -> list[dict[str, Any]]:
    # AGENT_CONTRACT §8: trader-owned pending entries are campaign
    # occupancy, not a blanket fresh-entry stop. They block completion
    # only after the target is done, or when the owner is not the trader
    # and cannot be basket-validated as QuantRabbit risk.
    if remaining_target <= 0:
        return list(pending_entries)
    return [item for item in pending_entries if str(item.get("owner") or "").lower() != "trader"]


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _trader_no_broker_tp_runner(position: dict[str, Any]) -> bool:
    return (
        str(position.get("owner") or "") == "trader"
        and position.get("take_profit") is None
        and _trader_sl_repair_disabled()
        and not _missing_tp_repair_enabled()
    )


def _is_layerable_position(position: dict[str, Any]) -> bool:
    # SL-free regime (`QR_TRADER_DISABLE_SL_REPAIR=1`, user directive
    # 「SLいらない」): trader-owned SL=None is intentional, and a missing
    # broker TP is preserved as a no-broker-TP runner unless repair is
    # explicitly enabled. Without this branch, every SL-free
    # trader position gets flagged as BROKER_EXPOSURE_OPEN in completion
    # blockers, freezing fresh entries while the operator-anchored
    # basket is alive (5/7 unblock fix missed this leak).
    if str(position.get("owner") or "") != "trader":
        return False
    if position.get("take_profit") is None and not _trader_no_broker_tp_runner(position):
        return False
    if position.get("stop_loss") is None:
        return _trader_sl_repair_disabled()
    return True


def _is_operator_managed_manual(item: dict[str, Any]) -> bool:
    return str(item.get("owner") or "").lower() in {"manual", "unknown"}


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


def _certification_is_stale(certification: dict[str, Any], *, dependencies: tuple[dict[str, Any], ...]) -> bool:
    certification_ts = _payload_timestamp(certification)
    if certification_ts is None:
        return False
    for payload in dependencies:
        payload_ts = _payload_timestamp(payload)
        if payload_ts is not None and payload_ts > certification_ts:
            return True
    return False


def _payload_timestamp(payload: dict[str, Any]) -> datetime | None:
    if not payload:
        return None
    for key in ("generated_at_utc", "generated_at"):
        raw = payload.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


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
