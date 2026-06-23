from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    repair_replay_contract_from_payload,
)
from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_GUARDIAN_EXECUTION,
    DEFAULT_POSITION_GUARDIAN_HEARTBEAT,
    DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_PROFIT_CAPTURE_BOT,
    DEFAULT_PROFITABILITY_ACCEPTANCE,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_TRADER_SUPPORT_BOT,
    DEFAULT_TRADER_SUPPORT_BOT_REPORT,
)


STATUS_READY = "SUPPORT_READY"
STATUS_BLOCKED = "SUPPORT_BLOCKED"
GUARDIAN_LABEL = "com.quantrabbit.position-guardian"
GUARDIAN_BLOCKER = "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE"
PROFIT_CAPTURE_MISS = "LOSS_CLOSE_PROFIT_CAPTURE_MISSED"
PERSISTENT_DISCIPLINE = "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED"
RANGE_FORECAST_ROTATION_BLOCKER = "RANGE_FORECAST_REQUIRES_RANGE_ROTATION"
RANGE_ROTATION_METHOD = "RANGE_ROTATION"
RANGE_ROTATION_COUNTERPART_MISSING = "RANGE_ROTATION_COUNTERPART_MISSING"
OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED = "OANDA_CAMPAIGN_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED"
OANDA_AUDIT_ONLY_REPLAY_HISTORY_GRANULARITIES = "S5,M5"
REPAIR_EXEMPTION_CODES = {PERSISTENT_DISCIPLINE, "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE"}
GLOBAL_UNLOCK_BLOCKERS = {
    GUARDIAN_BLOCKER,
    "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE",
    "NEGATIVE_EXPECTANCY_REQUIRES_TP_PROVEN_ROTATION",
}
ACCEPTANCE_LEAK_LOOKBACK_DAYS = 7
REPAIR_REQUEST_CONTRACT_VERSION = "trader_support_repair_request_v1"
REPAIR_AUTOMATION_ALLOWED_ACTIONS = [
    "read_artifacts",
    "edit_code",
    "edit_tests",
    "edit_runtime_contract_docs",
    "run_unit_tests",
    "commit",
    "sync_live_runtime",
]
REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS = [
    "order_send",
    "order_cancel",
    "position_close",
    "launchd_load",
    "launchd_reload",
]
REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS = [
    "direct_oanda_order_write",
    "direct_oanda_trade_close",
    "direct_launchd_mutation",
    "model_api_call_from_quantrabbit_code",
]


@dataclass(frozen=True)
class TraderSupportSummary:
    status: str
    output_path: Path
    report_path: Path
    blockers: list[dict[str, Any]]
    operator_actions: list[dict[str, Any]]
    metrics: dict[str, Any]


class TraderSupportBot:
    """Read-only operator panel for live trader support loops.

    The support bot does not send orders and does not load launchd agents. It
    turns existing broker/runtime artifacts into one gate report so the trader
    can see why profit capture, entries, or acceptance are blocked.
    """

    def __init__(
        self,
        *,
        broker_snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        position_management_path: Path = DEFAULT_POSITION_MANAGEMENT,
        position_guardian_management_path: Path = DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
        position_guardian_execution_path: Path = DEFAULT_POSITION_GUARDIAN_EXECUTION,
        position_guardian_heartbeat_path: Path = DEFAULT_POSITION_GUARDIAN_HEARTBEAT,
        self_improvement_audit_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
        profitability_acceptance_path: Path = DEFAULT_PROFITABILITY_ACCEPTANCE,
        execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
        profit_capture_bot_path: Path = DEFAULT_PROFIT_CAPTURE_BOT,
        oanda_rotation_mining_path: Path = DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING,
        output_path: Path = DEFAULT_TRADER_SUPPORT_BOT,
        report_path: Path = DEFAULT_TRADER_SUPPORT_BOT_REPORT,
        now_utc: datetime | None = None,
    ) -> None:
        self.broker_snapshot_path = broker_snapshot_path
        self.order_intents_path = order_intents_path
        self.target_state_path = target_state_path
        self.position_management_path = position_management_path
        self.position_guardian_management_path = position_guardian_management_path
        self.position_guardian_execution_path = position_guardian_execution_path
        self.position_guardian_heartbeat_path = position_guardian_heartbeat_path
        self.self_improvement_audit_path = self_improvement_audit_path
        self.profitability_acceptance_path = profitability_acceptance_path
        self.execution_timing_audit_path = execution_timing_audit_path
        self.profit_capture_bot_path = profit_capture_bot_path
        self.oanda_rotation_mining_path = oanda_rotation_mining_path
        self.output_path = output_path
        self.report_path = report_path
        self.now_utc = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    def run(self) -> TraderSupportSummary:
        payload = self.build_payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload), encoding="utf-8")
        return TraderSupportSummary(
            status=payload["status"],
            output_path=self.output_path,
            report_path=self.report_path,
            blockers=payload["blockers"],
            operator_actions=payload["operator_actions"],
            metrics=payload["metrics"],
        )

    def build_payload(self) -> dict[str, Any]:
        broker = _read_json(self.broker_snapshot_path)
        intents = _read_json(self.order_intents_path)
        target = _read_json(self.target_state_path)
        position_management = _read_json(self.position_management_path)
        guardian_management = _read_json(self.position_guardian_management_path)
        self_improvement = _read_json(self.self_improvement_audit_path)
        profitability = _read_json(self.profitability_acceptance_path)
        timing = _read_json(self.execution_timing_audit_path)
        profit_capture_bot = _read_json(self.profit_capture_bot_path)
        oanda_rotation = _read_json(self.oanda_rotation_mining_path)

        guardian = _guardian_status(
            now_utc=self.now_utc,
            execution_path=self.position_guardian_execution_path,
            heartbeat_path=self.position_guardian_heartbeat_path,
        )
        broker_summary = _broker_summary(broker)
        target_summary = _target_summary(target)
        p0_findings = _p0_findings(self_improvement)
        profit_capture = _profit_capture_summary(self_improvement, timing)
        current_profit_capture = _current_profit_capture_summary(profit_capture_bot)
        entry = _entry_readiness_summary(intents, oanda_rotation=oanda_rotation)
        position = _position_support_summary(position_management, guardian_management, now_utc=self.now_utc)
        acceptance = _acceptance_summary(profitability)

        blockers = _build_blockers(
            guardian=guardian,
            target=target_summary,
            profit_capture=profit_capture,
            p0_findings=p0_findings,
            entry=entry,
            acceptance=acceptance,
        )
        status = STATUS_BLOCKED if blockers else STATUS_READY
        actions = _operator_actions(
            status=status,
            blockers=blockers,
            guardian=guardian,
            profit_capture=profit_capture,
            entry=entry,
            acceptance=acceptance,
        )
        repair_requests = _build_repair_requests(
            guardian=guardian,
            profit_capture=profit_capture,
            entry=entry,
            acceptance=acceptance,
        )
        send_allowed = (
            status == STATUS_READY
            and bool(entry["live_ready_lanes"])
            and (not guardian["required"] or bool(guardian["active"]))
        )
        repair_basket_send_allowed = (
            bool(entry["repair_live_ready"])
            and (not guardian["required"] or bool(guardian["active"]))
        )
        metrics = {
            "send_fresh_entries_allowed": send_allowed,
            "repair_basket_send_allowed": repair_basket_send_allowed,
            "guardian_active": guardian["active"],
            "guardian_heartbeat_fresh": guardian["heartbeat_fresh"],
            "live_ready_lanes": entry["live_ready_lanes"],
            "repair_live_ready_lanes": len(entry["repair_live_ready"]),
            "repair_basket_guardian_recovery_lanes": len(entry["repair_basket_guardian_recovery"]),
            "repair_frontier_lanes": len(entry["repair_frontier"]),
            "repair_frontier_superseded_by_range_forecast_lanes": len(
                entry["repair_frontier_superseded_by_range_forecast"]
            ),
            "oanda_audit_only_local_tp_proof_required_lanes": len(
                entry["oanda_audit_only_local_tp_proof_required"]
            ),
            "oanda_audit_only_with_replay_evidence_lanes": sum(
                1
                for item in entry["oanda_audit_only_local_tp_proof_required"]
                if item.get("oanda_replay_evidence_status") != "MISSING_OANDA_REPLAY_EVIDENCE"
            ),
            "repair_frontier_missing_range_rotation_counterpart_lanes": len(
                entry["repair_frontier_missing_range_rotation_counterpart"]
            ),
            "repair_frontier_after_support_clear_lanes": entry["repair_frontier_after_support_clear_lanes"],
            "repair_frontier_after_support_blocked_lanes": entry["repair_frontier_after_support_blocked_lanes"],
            "repair_frontier_after_support_top_blockers": entry["repair_frontier_remaining_blockers"],
            "global_unlock_frontier_lanes": len(entry["global_unlock_frontier"]),
            "profit_capture_missed_loss_closes": profit_capture["missed_loss_closes"],
            "profit_capture_estimated_gap_jpy": profit_capture["estimated_gap_jpy"],
            "profit_capture_actual_loss_close_pl_jpy": profit_capture["actual_loss_close_pl_jpy"],
            "profit_capture_counterfactual_pl_jpy": profit_capture["counterfactual_profit_capture_pl_jpy"],
            "profit_capture_counterfactual_delta_jpy": profit_capture["counterfactual_profit_capture_delta_jpy"],
            "profit_capture_counterfactual_jpy": profit_capture["counterfactual_profit_capture_jpy"],
            "profit_capture_repair_replay_contract": profit_capture["repair_replay_contract"],
            "profit_capture_repair_replay_contract_present": profit_capture[
                "repair_replay_contract_present"
            ],
            "profit_capture_repair_replay_triggered": profit_capture["repair_replay_triggered"],
            "profit_capture_repair_replay_delta_jpy": profit_capture["repair_replay_delta_jpy"],
            "profit_capture_repair_replay_jpy": profit_capture["repair_replay_jpy"],
            "profit_capture_bankable_positions": current_profit_capture["bankable_positions"],
            "profit_capture_watch_positions": current_profit_capture["watch_positions"],
            "profit_capture_blocked_positions": current_profit_capture["blocked_positions"],
            "open_trader_positions": broker_summary["trader_positions"],
            "target_remaining_jpy": target_summary["remaining_target_jpy"],
            "profitability_status": acceptance["status"],
            "target_firepower_status": acceptance["target_firepower"]["status"],
            "target_firepower_minimum_5pct_estimated_reachable": acceptance["target_firepower"][
                "minimum_5pct_estimated_reachable"
            ],
            "target_firepower_target_10pct_estimated_reachable": acceptance["target_firepower"][
                "target_10pct_estimated_reachable"
            ],
            "acceptance_evidence_collection_count": len(
                acceptance["repair_plan"].get("evidence_collection_items", [])
            ),
            "repair_request_count": len(repair_requests),
            "repair_request_codes": [item["code"] for item in repair_requests],
            "repair_basket_lane_ids": [item["lane_id"] for item in entry["repair_live_ready"]],
            "repair_basket_guardian_recovery_lane_ids": [
                item["lane_id"] for item in entry["repair_basket_guardian_recovery"]
            ],
            "oanda_audit_only_local_tp_proof_required_lane_ids": [
                item["lane_id"] for item in entry["oanda_audit_only_local_tp_proof_required"]
            ],
        }
        generated = self.now_utc.isoformat()
        return {
            "artifact_paths": {
                "broker_snapshot": str(self.broker_snapshot_path),
                "order_intents": str(self.order_intents_path),
                "target_state": str(self.target_state_path),
                "position_management": str(self.position_management_path),
                "position_guardian_management": str(self.position_guardian_management_path),
                "position_guardian_execution": str(self.position_guardian_execution_path),
                "position_guardian_heartbeat": str(self.position_guardian_heartbeat_path),
                "self_improvement_audit": str(self.self_improvement_audit_path),
                "profitability_acceptance": str(self.profitability_acceptance_path),
                "execution_timing_audit": str(self.execution_timing_audit_path),
                "profit_capture_bot": str(self.profit_capture_bot_path),
                "oanda_rotation_mining": str(self.oanda_rotation_mining_path),
            },
            "generated_at_utc": generated,
            "status": status,
            "blockers": blockers,
            "operator_actions": actions,
            "repair_requests": repair_requests,
            "metrics": metrics,
            "guardian": guardian,
            "broker": broker_summary,
            "target": target_summary,
            "profit_capture": profit_capture,
            "current_profit_capture": current_profit_capture,
            "entry_readiness": entry,
            "position_support": position,
            "self_improvement": {
                "status": self_improvement.get("status") if isinstance(self_improvement, dict) else None,
                "p0_findings": p0_findings,
                "p0_count": len(p0_findings),
            },
            "profitability_acceptance": acceptance,
            "read_only": True,
            "live_side_effects": [],
        }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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


def _age_seconds(value: Any, *, now_utc: datetime) -> float | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    return max(0.0, (now_utc - parsed).total_seconds())


def _truthy(raw: Any) -> bool:
    return str(raw).strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _truthy_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _truthy(raw)


def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _guardian_status(*, now_utc: datetime, execution_path: Path, heartbeat_path: Path) -> dict[str, Any]:
    label = os.environ.get("QR_POSITION_GUARDIAN_LABEL", GUARDIAN_LABEL)
    plist = Path(
        os.environ.get(
            "QR_POSITION_GUARDIAN_PLIST",
            str(Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"),
        )
    ).expanduser()
    interval = _int_env("QR_POSITION_GUARDIAN_INTERVAL", default=30, minimum=15)
    max_age = _int_env("QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS", default=interval * 4, minimum=interval)
    heartbeat_required = _truthy_env("QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT", default=True)
    required = _truthy_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", default=True)
    paths = [
        Path(os.environ.get("QR_POSITION_GUARDIAN_EXECUTION", str(execution_path))).expanduser(),
        Path(os.environ.get("QR_POSITION_GUARDIAN_HEARTBEAT", str(heartbeat_path))).expanduser(),
    ]
    heartbeat = _freshest_guardian_heartbeat(paths, now_utc=now_utc, max_age=max_age)
    env_active_raw = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")

    status: dict[str, Any] = {
        "required": required,
        "active": False,
        "active_source": "launchd+heartbeat",
        "label": label,
        "plist_path": str(plist),
        "plist_exists": plist.exists(),
        "launchd_loaded": None,
        "heartbeat_required": heartbeat_required,
        "heartbeat_max_age_seconds": max_age,
        "heartbeat_fresh": heartbeat["fresh"],
        "heartbeat_age_seconds": heartbeat["age_seconds"],
        "heartbeat_generated_at_utc": heartbeat["generated_at_utc"],
        "heartbeat_path": heartbeat["path"],
        "heartbeat_status": heartbeat["status"],
        "heartbeat_candidates": [str(path) for path in paths],
    }
    if env_active_raw is not None:
        env_active = _truthy(env_active_raw)
        status["env_active"] = env_active_raw
        status["active_source"] = "env+heartbeat"
        status["active"] = bool(env_active and (heartbeat["fresh"] or not heartbeat_required))
        return status
    if not plist.exists():
        status["active_source"] = "plist_missing"
        status["launchd_loaded"] = False
        return status
    loaded = _launchd_loaded(label)
    status["launchd_loaded"] = loaded["loaded"]
    if loaded.get("error"):
        status["launchctl_error"] = loaded["error"]
        status["active_source"] = "launchctl_unavailable"
    status["active"] = bool(loaded["loaded"] and (heartbeat["fresh"] or not heartbeat_required))
    if loaded["loaded"] and heartbeat_required and not heartbeat["fresh"]:
        status["active_source"] = "stale_heartbeat"
    return status


def _launchd_loaded(label: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["launchctl", "list", label],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return {"loaded": False, "error": exc.__class__.__name__}
    if proc.returncode == 0:
        return {"loaded": True}
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return {"loaded": False, "error": exc.__class__.__name__}
    return {"loaded": proc.returncode == 0}


def _freshest_guardian_heartbeat(paths: list[Path], *, now_utc: datetime, max_age: int) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        generated = payload.get("generated_at_utc")
        age = _age_seconds(generated, now_utc=now_utc)
        if age is None:
            continue
        item = {
            "path": str(path),
            "generated_at_utc": generated,
            "age_seconds": round(age, 3),
            "fresh": age <= max_age,
            "status": payload.get("status"),
        }
        if best is None or (item["age_seconds"] or 0) < (best["age_seconds"] or 0):
            best = item
    if best is None:
        return {
            "path": None,
            "generated_at_utc": None,
            "age_seconds": None,
            "fresh": False,
            "status": "MISSING",
        }
    return best


def _broker_summary(payload: dict[str, Any]) -> dict[str, Any]:
    positions = payload.get("positions") if isinstance(payload.get("positions"), list) else []
    orders = payload.get("orders") if isinstance(payload.get("orders"), list) else []
    trader_positions = [item for item in positions if str(item.get("owner") or "").lower() == "trader"]
    rows = []
    for item in trader_positions:
        rows.append(
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "units": item.get("units"),
                "unrealized_pl_jpy": _round_optional(item.get("unrealized_pl_jpy"), 3),
                "take_profit": item.get("take_profit"),
                "stop_loss": item.get("stop_loss"),
            }
        )
    account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    return {
        "fetched_at_utc": payload.get("fetched_at_utc") or account.get("fetched_at_utc"),
        "positions": len(positions),
        "orders": len(orders),
        "trader_positions": len(trader_positions),
        "profitable_trader_positions": sum(1 for item in trader_positions if _float(item.get("unrealized_pl_jpy")) > 0),
        "trader_unrealized_pl_jpy": _round_optional(
            sum(_float(item.get("unrealized_pl_jpy")) for item in trader_positions),
            3,
        ),
        "balance_jpy": _round_optional(account.get("balance_jpy"), 3),
        "nav_jpy": _round_optional(account.get("nav_jpy"), 3),
        "margin_available_jpy": _round_optional(account.get("margin_available_jpy"), 3),
        "open_trader_positions": rows,
    }


def _target_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload.get("status"),
        "campaign_day_jst": payload.get("campaign_day_jst"),
        "target_open": str(payload.get("status") or "") == "PURSUE_TARGET",
        "progress_pct": _round_optional(payload.get("progress_pct"), 4),
        "minimum_progress_pct": _round_optional(payload.get("minimum_progress_pct"), 4),
        "remaining_minimum_jpy": _round_optional(payload.get("remaining_minimum_jpy"), 3),
        "remaining_target_jpy": _round_optional(payload.get("remaining_target_jpy"), 3),
        "realized_pl_jpy": _round_optional(payload.get("realized_pl_jpy"), 3),
        "current_equity_jpy": _round_optional(payload.get("current_equity_jpy"), 3),
        "target_trades_per_day": payload.get("target_trades_per_day"),
    }


def _p0_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    return [
        {
            "code": item.get("code"),
            "message": item.get("message"),
            "next_action": item.get("next_action"),
            "evidence": item.get("evidence"),
        }
        for item in findings
        if str(item.get("priority") or "").upper() == "P0"
    ]


def _profit_capture_summary(self_improvement: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    finding = next((item for item in _p0_findings(self_improvement) if item.get("code") == PROFIT_CAPTURE_MISS), None)
    evidence = finding.get("evidence") if isinstance(finding, dict) and isinstance(finding.get("evidence"), dict) else {}
    summary = timing.get("summary") if isinstance(timing.get("summary"), dict) else {}
    repair_replay_contract = repair_replay_contract_from_payload(timing)
    repair_replay_contract_present = repair_replay_contract == TP_PROGRESS_REPAIR_REPLAY_CONTRACT
    missed = evidence.get("loss_closes_profit_capture_missed", summary.get("loss_closes_profit_capture_missed"))
    gap = evidence.get("loss_close_estimated_capture_gap_jpy", summary.get("loss_close_estimated_capture_gap_jpy"))
    actual_pl = evidence.get("loss_close_actual_pl_jpy", summary.get("loss_close_actual_pl_jpy"))
    counterfactual_pl = evidence.get(
        "loss_close_counterfactual_profit_capture_pl_jpy",
        summary.get("loss_close_counterfactual_profit_capture_pl_jpy"),
    )
    counterfactual_delta = evidence.get(
        "loss_close_counterfactual_profit_capture_delta_jpy",
        summary.get("loss_close_counterfactual_profit_capture_delta_jpy"),
    )
    counterfactual_jpy = evidence.get(
        "loss_close_counterfactual_profit_capture_jpy",
        summary.get("loss_close_counterfactual_profit_capture_jpy"),
    )
    repair_replay_triggered = evidence.get(
        "loss_closes_repair_replay_triggered",
        summary.get("loss_closes_repair_replay_triggered"),
    )
    repair_replay_delta = evidence.get(
        "loss_close_repair_replay_delta_jpy",
        summary.get("loss_close_repair_replay_delta_jpy"),
    )
    repair_replay_jpy = evidence.get(
        "loss_close_repair_replay_profit_capture_jpy",
        summary.get("loss_close_repair_replay_profit_capture_jpy"),
    )
    top = evidence.get("top_profit_capture_misses") if isinstance(evidence.get("top_profit_capture_misses"), list) else []
    top_repair = (
        evidence.get("top_repair_replay_triggers")
        if isinstance(evidence.get("top_repair_replay_triggers"), list)
        else []
    )
    if not top_repair:
        rows = timing.get("loss_close_regrets") if isinstance(timing.get("loss_close_regrets"), list) else []
        top_repair = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("repair_replay_triggered_before_loss_close")
        ][:5]
    return {
        "status": "PROFIT_CAPTURE_REPAIR_REQUIRED" if _float(missed) > 0 else "OK",
        "missed_loss_closes": int(_float(missed)),
        "estimated_gap_jpy": _round_optional(gap, 3),
        "actual_loss_close_pl_jpy": _round_optional(actual_pl, 3),
        "counterfactual_profit_capture_pl_jpy": _round_optional(counterfactual_pl, 3),
        "counterfactual_profit_capture_delta_jpy": _round_optional(counterfactual_delta, 3),
        "counterfactual_profit_capture_jpy": _round_optional(counterfactual_jpy, 3),
        "repair_replay_triggered": int(_float(repair_replay_triggered)),
        "repair_replay_contract": repair_replay_contract,
        "repair_replay_contract_present": repair_replay_contract_present,
        "repair_replay_delta_jpy": _round_optional(repair_replay_delta, 3),
        "repair_replay_jpy": _round_optional(repair_replay_jpy, 3),
        "stop_loss_missed": evidence.get("stop_loss_closes_profit_capture_missed", summary.get("stop_loss_closes_profit_capture_missed")),
        "top_misses": top[:5],
        "top_repair_replay_triggers": top_repair[:5],
        "message": finding.get("message") if finding else None,
        "next_action": finding.get("next_action") if finding else None,
        "clearance_condition": (
            "execution-timing-audit reports zero loss_closes_repair_replay_triggered under the "
            "current production-gate replay contract, and position guardian is proven active "
            "before fresh entries resume; raw loss_closes_profit_capture_missed remains "
            "diagnostic unless production-gate replay also triggers"
            if _float(missed) > 0
            else "no missed TP-progress loss close in the active timing audit"
        ),
        "verification_command": "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80",
    }


def _current_profit_capture_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    positions = payload.get("positions") if isinstance(payload.get("positions"), list) else []
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "open_trader_positions": metrics.get("open_trader_positions"),
        "bankable_positions": metrics.get("bankable_positions", 0),
        "watch_positions": metrics.get("watch_positions", 0),
        "blocked_positions": metrics.get("blocked_positions", 0),
        "historical_repair_replay_triggered": metrics.get("historical_repair_replay_triggered"),
        "historical_repair_replay_delta_jpy": metrics.get("historical_repair_replay_delta_jpy"),
        "top_gate_statuses": [
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "gate_status": item.get("gate_status"),
                "tp_progress": item.get("tp_progress"),
                "capture_trigger": item.get("capture_trigger"),
            }
            for item in positions[:5]
            if isinstance(item, dict)
        ],
    }


def _entry_readiness_summary(
    payload: dict[str, Any],
    *,
    oanda_rotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    live_ready = [item for item in results if item.get("status") == "LIVE_READY"]
    oanda_evidence_by_vehicle = _oanda_replay_evidence_by_vehicle(oanda_rotation or {})
    codes = Counter()
    repair_frontier: list[dict[str, Any]] = []
    repair_live_ready: list[dict[str, Any]] = []
    repair_basket_guardian_recovery: list[dict[str, Any]] = []
    repair_frontier_superseded_by_range_forecast: list[dict[str, Any]] = []
    repair_frontier_missing_range_rotation_counterpart: list[dict[str, Any]] = []
    oanda_audit_only_local_tp_proof_required: list[dict[str, Any]] = []
    global_unlock_frontier: list[dict[str, Any]] = []
    range_rotation_counterparts = _range_rotation_counterparts(results)
    for item in results:
        for code in item.get("live_blocker_codes") or []:
            codes[str(code)] += 1
        blocker_codes = [str(code) for code in item.get("live_blocker_codes") or []]
        metadata = _intent_metadata(item)
        if (
            metadata.get("positive_rotation_oanda_campaign_audit_only") is True
            or OANDA_AUDIT_ONLY_LOCAL_TP_PROOF_REQUIRED in blocker_codes
        ):
            intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
            context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
            vehicle_key = (
                metadata.get("positive_rotation_oanda_campaign_matching_vehicle_key")
                or metadata.get("oanda_campaign_vehicle_key")
            )
            replay_evidence = _compact_oanda_replay_evidence(vehicle_key, oanda_evidence_by_vehicle)
            oanda_audit_only_local_tp_proof_required.append(
                {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "pair": intent.get("pair"),
                    "side": intent.get("side"),
                    "method": context.get("method"),
                    "order_type": intent.get("order_type"),
                    "reward_jpy": _intent_reward_jpy(item, metadata),
                    "risk_jpy": _intent_risk_jpy(item, metadata),
                    "capture_take_profit_scope": metadata.get("capture_take_profit_scope"),
                    "capture_take_profit_scope_key": metadata.get("capture_take_profit_scope_key"),
                    "oanda_vehicle_key": vehicle_key,
                    "oanda_replay_evidence_status": replay_evidence["evidence_status"],
                    "oanda_replay_live_permission": replay_evidence["live_permission"],
                    "oanda_replay_evidence": replay_evidence,
                    "remaining_blocker_codes_after_guardian": [
                        code for code in blocker_codes if code != GUARDIAN_BLOCKER
                    ],
                    "local_proof_required": True,
                    "historical_replay_can_clear_local_tp_proof": False,
                    "local_tp_proof_clearance_condition": (
                        "exact PAIR_SIDE_METHOD TAKE_PROFIT_ORDER receipts for this "
                        "pair/side/method must show positive expectancy, zero TP losses, "
                        "and positive Wilson-stressed expectancy; OANDA candle replay can "
                        "rank/mine the candidate but cannot by itself clear the local "
                        "broker-TP proof gate"
                    ),
                }
            )
        tp_proof = _tp_proof_summary(metadata)
        if str(item.get("status") or "") != "LIVE_READY" and blocker_codes:
            remaining_after_global = [code for code in blocker_codes if code not in GLOBAL_UNLOCK_BLOCKERS]
            if not remaining_after_global:
                intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
                context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
                unlock_item = {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "pair": intent.get("pair"),
                    "side": intent.get("side"),
                    "method": context.get("method"),
                    "order_type": intent.get("order_type"),
                    "reward_jpy": _intent_reward_jpy(item, metadata),
                    "risk_jpy": _intent_risk_jpy(item, metadata),
                    "global_blocker_codes": blocker_codes,
                    "remaining_blocker_codes_after_global_unlock": remaining_after_global,
                    "repair_mode": metadata.get("self_improvement_p0_repair_mode")
                    or metadata.get("positive_rotation_mode"),
                }
                if tp_proof:
                    unlock_item["tp_proof"] = tp_proof
                global_unlock_frontier.append(unlock_item)
        if metadata.get("self_improvement_p0_repair_live_ready") is True:
            exempt = set(REPAIR_EXEMPTION_CODES)
            remaining_after_support = [
                code for code in blocker_codes if code != GUARDIAN_BLOCKER and code not in exempt
            ]
            intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
            context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
            repair_item = {
                "lane_id": item.get("lane_id"),
                "status": item.get("status"),
                "pair": intent.get("pair"),
                "side": intent.get("side"),
                "method": context.get("method"),
                "order_type": intent.get("order_type"),
                "reward_jpy": _intent_reward_jpy(item, metadata),
                "risk_jpy": _intent_risk_jpy(item, metadata),
                "repair_mode": metadata.get("self_improvement_p0_repair_mode"),
                "matrix_repair_profile_status": metadata.get("matrix_repair_profile_status"),
                "blocker_codes": blocker_codes,
                "remaining_blocker_codes_after_guardian_and_repair_exemption": remaining_after_support,
            }
            if tp_proof:
                repair_item["tp_proof"] = tp_proof
            if (
                RANGE_FORECAST_ROTATION_BLOCKER in blocker_codes
                and str(context.get("method") or "").upper() != RANGE_ROTATION_METHOD
            ):
                counterpart = range_rotation_counterparts.get((intent.get("pair"), intent.get("side")))
                if counterpart:
                    repair_item["superseded_reason"] = (
                        "RANGE forecast requires RANGE_ROTATION; non-rotation repair candidate is not an executable repair frontier"
                    )
                    repair_item["superseded_by_range_rotation_lane_id"] = counterpart
                    repair_frontier_superseded_by_range_forecast.append(repair_item)
                else:
                    remaining_with_gap = [
                        code
                        for code in remaining_after_support
                        if code != RANGE_FORECAST_ROTATION_BLOCKER
                    ]
                    remaining_with_gap.append(RANGE_ROTATION_COUNTERPART_MISSING)
                    repair_item["missing_range_rotation_counterpart_for"] = [intent.get("pair"), intent.get("side")]
                    repair_item["missing_range_rotation_counterpart_reason"] = (
                        "RANGE forecast blocked a non-rotation repair lane, but order_intents has no same pair/side "
                        "RANGE_ROTATION counterpart to inspect. Treat this as candidate coverage debt, not as a "
                        "superseded executable frontier lane."
                    )
                    repair_item["remaining_blocker_codes_after_guardian_and_repair_exemption"] = remaining_with_gap
                    repair_frontier_missing_range_rotation_counterpart.append(repair_item)
                continue
            repair_frontier.append(repair_item)
            if item.get("status") == "LIVE_READY" and not blocker_codes:
                repair_live_ready.append(repair_item)
            elif GUARDIAN_BLOCKER in blocker_codes and not remaining_after_support:
                repair_basket_guardian_recovery.append(repair_item)
    repair_frontier.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_live_ready.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_basket_guardian_recovery.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_frontier_superseded_by_range_forecast.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    repair_frontier_missing_range_rotation_counterpart.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    oanda_audit_only_local_tp_proof_required.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    global_unlock_frontier.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    remaining_repair_blockers = _repair_frontier_remaining_blockers(repair_frontier)
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "lanes": len(results),
        "live_ready_lanes": len(live_ready),
        "live_ready_lane_ids": [item.get("lane_id") for item in live_ready],
        "top_blockers": [{"code": code, "count": count} for code, count in codes.most_common(12)],
        "guardian_blocked_lanes": codes.get(GUARDIAN_BLOCKER, 0),
        "repair_frontier": repair_frontier[:12],
        "repair_frontier_superseded_by_range_forecast": repair_frontier_superseded_by_range_forecast[:12],
        "repair_frontier_missing_range_rotation_counterpart": repair_frontier_missing_range_rotation_counterpart[:12],
        "oanda_audit_only_local_tp_proof_required": oanda_audit_only_local_tp_proof_required[:12],
        "repair_live_ready": repair_live_ready[:12],
        "repair_basket_guardian_recovery": repair_basket_guardian_recovery[:12],
        "repair_frontier_after_support_clear_lanes": sum(
            1 for item in repair_frontier if not item["remaining_blocker_codes_after_guardian_and_repair_exemption"]
        ),
        "repair_frontier_after_support_blocked_lanes": sum(
            1 for item in repair_frontier if item["remaining_blocker_codes_after_guardian_and_repair_exemption"]
        ),
        "repair_frontier_remaining_blockers": remaining_repair_blockers,
        "global_unlock_frontier": global_unlock_frontier[:12],
    }


def _tp_proof_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "positive_rotation_mode",
        "positive_rotation_pessimistic_expectancy_jpy",
        "capture_take_profit_scope",
        "capture_take_profit_scope_key",
        "capture_take_profit_trades",
        "capture_take_profit_wins",
        "capture_take_profit_losses",
        "capture_take_profit_expectancy_jpy",
        "capture_take_profit_avg_win_jpy",
        "capture_take_profit_avg_loss_jpy",
    )
    summary = {key: metadata.get(key) for key in keys if metadata.get(key) is not None}
    if not summary:
        return {}
    return summary


def _oanda_replay_evidence_by_vehicle(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    firepower = payload.get("campaign_firepower") if isinstance(payload.get("campaign_firepower"), dict) else {}
    sections: list[tuple[int, str, list[Any]]] = []
    high_precision = firepower.get("high_precision") if isinstance(firepower.get("high_precision"), dict) else {}
    evidence_queue = firepower.get("evidence_queue") if isinstance(firepower.get("evidence_queue"), dict) else {}
    sections.append((0, "high_precision", high_precision.get("top_vehicles") or []))
    sections.append((1, "evidence_queue", evidence_queue.get("top_vehicles") or []))
    sections.append((2, "live_grade_evidence_queue", payload.get("live_grade_evidence_queue") or []))
    out: dict[str, dict[str, Any]] = {}
    seen_priority: dict[str, int] = {}
    for priority, section, rows in sections:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            vehicle_key = row.get("vehicle_key")
            if not vehicle_key:
                continue
            key = str(vehicle_key)
            if key in out and seen_priority.get(key, 99) <= priority:
                continue
            compact = dict(row)
            compact["firepower_section"] = row.get("firepower_section") or section
            out[key] = compact
            seen_priority[key] = priority
    return out


def _compact_oanda_replay_evidence(
    vehicle_key: Any,
    evidence_by_vehicle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = str(vehicle_key or "")
    row = evidence_by_vehicle.get(key)
    if not row:
        return {
            "vehicle_key": key or None,
            "evidence_status": "MISSING_OANDA_REPLAY_EVIDENCE",
            "live_permission": False,
            "live_permission_reason": "No matching OANDA candle replay vehicle was found for this intent key.",
        }
    return {
        "vehicle_key": key,
        "firepower_section": row.get("firepower_section"),
        "evidence_status": row.get("evidence_status") or "OANDA_REPLAY_EVIDENCE",
        "validation_n": row.get("validation_n"),
        "validation_win_rate": _round_optional(row.get("validation_win_rate"), 6),
        "validation_win_wilson95_lower": _round_optional(row.get("validation_win_wilson95_lower"), 6),
        "validation_profit_factor": _round_optional(row.get("validation_profit_factor"), 6),
        "validation_avg_realized_pips": _round_optional(row.get("validation_avg_realized_pips"), 6),
        "validation_expectancy_r": _round_optional(row.get("validation_expectancy_r"), 6),
        "active_days": row.get("active_days"),
        "positive_day_rate": _round_optional(row.get("positive_day_rate"), 6),
        "estimated_return_pct_per_active_day_at_observed_frequency": _round_optional(
            row.get("estimated_return_pct_per_active_day_at_observed_frequency"),
            6,
        ),
        "trades_needed_for_minimum_5pct": row.get("trades_needed_for_minimum_5pct"),
        "trades_needed_for_target_10pct": row.get("trades_needed_for_target_10pct"),
        "live_permission": False,
        "live_permission_reason": (
            "OANDA candle replay is audit-only until local broker TP proof, current forecast, "
            "spread, risk, strategy-profile, and gateway checks all pass."
        ),
    }


def _range_rotation_counterparts(results: list[dict[str, Any]]) -> dict[tuple[Any, Any], Any]:
    counterparts: dict[tuple[Any, Any], Any] = {}
    for item in results:
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
        if str(context.get("method") or "").upper() != RANGE_ROTATION_METHOD:
            continue
        key = (intent.get("pair"), intent.get("side"))
        if key not in counterparts:
            counterparts[key] = item.get("lane_id")
    return counterparts


def _repair_frontier_remaining_blockers(repair_frontier: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    reward_by_code: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for item in repair_frontier:
        lane_id = str(item.get("lane_id") or "")
        reward = _float(item.get("reward_jpy"))
        remaining = [
            str(code)
            for code in item.get("remaining_blocker_codes_after_guardian_and_repair_exemption") or []
            if str(code).strip()
        ]
        for code in sorted(set(remaining)):
            counts[code] += 1
            reward_by_code[code] += reward
            if lane_id and len(examples[code]) < 3:
                examples[code].append(lane_id)
    rows = []
    for code, count in counts.most_common(12):
        rows.append(
            {
                "code": code,
                "count": count,
                "reward_jpy": _round_optional(reward_by_code[code], 3),
                "example_lane_ids": examples[code],
            }
        )
    return rows


def _intent_metadata(item: dict[str, Any]) -> dict[str, Any]:
    intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    return metadata


def _intent_reward_jpy(item: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    risk_metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
    return _round_optional(
        metadata.get("sizing_actual_reward_jpy")
        if metadata.get("sizing_actual_reward_jpy") is not None
        else risk_metrics.get("reward_jpy"),
        3,
    )


def _intent_risk_jpy(item: dict[str, Any], metadata: dict[str, Any]) -> float | None:
    risk_metrics = item.get("risk_metrics") if isinstance(item.get("risk_metrics"), dict) else {}
    return _round_optional(
        metadata.get("sizing_actual_risk_jpy")
        if metadata.get("sizing_actual_risk_jpy") is not None
        else risk_metrics.get("risk_jpy"),
        3,
    )


def _position_support_summary(
    position_management: dict[str, Any],
    guardian_management: dict[str, Any],
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    current_positions = (
        position_management.get("positions") if isinstance(position_management.get("positions"), list) else []
    )
    guardian_positions = (
        guardian_management.get("positions") if isinstance(guardian_management.get("positions"), list) else []
    )
    generated = guardian_management.get("generated_at_utc")
    return {
        "position_management_action": position_management.get("action"),
        "position_management_generated_at_utc": position_management.get("generated_at_utc"),
        "managed_positions": _compact_position_actions(current_positions),
        "guardian_management_generated_at_utc": generated,
        "guardian_management_age_seconds": _round_optional(_age_seconds(generated, now_utc=now_utc), 3),
        "guardian_managed_positions": _compact_position_actions(guardian_positions),
    }


def _compact_position_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in rows[:12]:
        compact.append(
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "action": item.get("action"),
                "close_review_action": item.get("close_review_action"),
                "unrealized_pl_jpy": _round_optional(item.get("unrealized_pl_jpy"), 3),
                "remaining_reward_jpy": _round_optional(item.get("remaining_reward_jpy"), 3),
                "remaining_risk_jpy": _round_optional(item.get("remaining_risk_jpy"), 3),
            }
        )
    return compact


def _acceptance_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    capture = metrics.get("capture_economics") if isinstance(metrics.get("capture_economics"), dict) else {}
    firepower = metrics.get("oanda_campaign_firepower") if isinstance(metrics.get("oanda_campaign_firepower"), dict) else {}
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    return {
        "status": payload.get("status"),
        "blockers": blockers[:8],
        "capture_economics": capture,
        "target_firepower": _target_firepower_summary(firepower),
        "repair_plan": _acceptance_repair_plan(payload),
    }


def _acceptance_repair_plan(payload: dict[str, Any]) -> dict[str, Any]:
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    p0_findings = [
        item
        for item in findings
        if isinstance(item, dict) and str(item.get("priority") or "").upper() == "P0"
    ]
    evidence_findings = [
        item
        for item in findings
        if isinstance(item, dict)
        and str(item.get("code") or "") in {
            "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL",
            "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE",
            "BIDASK_CONTRARIAN_EDGE_NOT_DAILY_STABLE",
        }
    ]
    fallback_blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    items: list[dict[str, Any]] = []
    if p0_findings:
        for finding in p0_findings[:12]:
            items.append(_acceptance_repair_item(finding, metrics))
    elif fallback_blockers:
        for blocker in fallback_blockers[:12]:
            code, message = _split_blocker(str(blocker))
            items.append(
                _acceptance_repair_item(
                    {
                        "priority": "P0",
                        "code": code,
                        "message": message,
                        "next_action": "Inspect profitability_acceptance findings for the concrete red invariant.",
                        "evidence": {},
                    },
                    metrics,
                )
            )
    evidence_items = [_acceptance_repair_item(finding, metrics) for finding in evidence_findings[:8]]
    commands: list[str] = []
    seen = set()
    for item in [*items, *evidence_items]:
        command = item.get("verification_command")
        if not command or command in seen:
            continue
        seen.add(command)
        commands.append(str(command))
    return {
        "status": payload.get("status"),
        "p0_count": len(p0_findings) if p0_findings else len(fallback_blockers),
        "items": items,
        "evidence_collection_items": evidence_items,
        "next_verification_commands": commands[:8],
        "loop_breaker": (
            "Rerunning profitability-acceptance alone cannot clear these P0s; the listed proof condition "
            "must change first. Evidence-collection items also require fresh replay/mining output before "
            "they can graduate into live-grade turnover."
            if items
            else "No acceptance P0 repair item is active; work evidence-collection items before claiming "
            "new high-turn firepower."
            if evidence_items
            else "No acceptance P0 repair item is active."
        ),
    }


def _acceptance_repair_item(finding: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    code = str(finding.get("code") or "UNKNOWN_ACCEPTANCE_BLOCKER")
    evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
    condition, command, summary = _acceptance_clearance_for_code(code, evidence, metrics)
    return {
        "code": code,
        "priority": str(finding.get("priority") or "P0"),
        "message": finding.get("message"),
        "next_action": finding.get("next_action"),
        "clearance_condition": condition,
        "verification_command": command,
        "evidence_summary": summary,
    }


def _acceptance_clearance_for_code(
    code: str,
    evidence: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    if code == "SELF_IMPROVEMENT_P0_PRESENT":
        p0_findings = evidence.get("p0_findings") if isinstance(evidence.get("p0_findings"), list) else []
        p0_codes = [
            str(item.get("code"))
            for item in p0_findings
            if isinstance(item, dict) and item.get("code")
        ]
        return (
            "self_improvement_audit has zero P0 findings, or the only remaining discipline finding has "
            "been demoted by verified clean gateway close recovery",
            "PYTHONPATH=src python3 -m quant_rabbit.cli self-improvement-audit",
            {"current_p0_codes": p0_codes[:8], "current_p0_count": len(p0_codes) or None},
        )
    if code == "NEGATIVE_EXPECTANCY_ACTIVE":
        capture = metrics.get("capture_economics") if isinstance(metrics.get("capture_economics"), dict) else {}
        capture = capture or evidence
        overall = capture.get("overall") if isinstance(capture.get("overall"), dict) else capture
        return (
            "capture_economics.status is no longer NEGATIVE_EXPECTANCY, or entries are limited to exact "
            "TP-proven repair/harvest shapes with positive expectancy evidence",
            "PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics",
            {
                "status": capture.get("status"),
                "expectancy_jpy_per_trade": _round_optional(
                    overall.get("expectancy_jpy_per_trade"),
                    3,
                ),
                "net_jpy": _round_optional(overall.get("net_jpy"), 3),
                "trades": overall.get("trades"),
                "profit_factor": _round_optional(overall.get("profit_factor"), 3),
            },
        )
    if code == "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE":
        segments = evidence.get("segments") if isinstance(evidence.get("segments"), list) else []
        return (
            "no TP-proven segment remains net-damaged by MARKET_ORDER_TRADE_CLOSE leakage; preserve broker TP "
            "and guardian capture instead of scaling market-close loss paths",
            "PYTHONPATH=src python3 -m quant_rabbit.cli capture-economics",
            {"damaged_segments": len(segments), "top_segments": segments[:3]},
        )
    if code == "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK":
        latest = evidence.get("latest_loss_close_ts_utc") or _latest_ts_from_examples(evidence)
        return (
            "recent_leak_loss_closes is zero inside the 7-day acceptance window, or each loss-side market "
            "close has contained-risk timing plus durable gateway/GPT close proof",
            "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80",
            {
                "recent_leak_loss_closes": evidence.get("recent_leak_loss_closes"),
                "recent_leak_loss_net_jpy": _round_optional(evidence.get("recent_leak_loss_net_jpy"), 3),
                "latest_loss_close_ts_utc": latest,
                "earliest_auto_clear_if_no_new_leak_utc": _plus_days_iso(
                    latest,
                    ACCEPTANCE_LEAK_LOOKBACK_DAYS,
                ),
                "timing_labels": evidence.get("recent_loss_timing_label_counts"),
            },
        )
    if code == "LOSS_CLOSE_GATE_EVIDENCE_MISSING":
        latest = _latest_ts_from_examples(evidence)
        return (
            "every recent GPT loss-side market close has durable close_gate_evidence in verification_observations, "
            "or the missing-evidence closes age out of the 7-day acceptance window without new leaks",
            "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
            {
                "missing_close_gate_evidence": evidence.get(
                    "recent_close_gate_missing_loss_closes",
                    evidence.get("recent_close_gate_unverified_loss_closes"),
                ),
                "missing_close_gate_net_jpy": _round_optional(
                    evidence.get(
                        "recent_close_gate_missing_loss_net_jpy",
                        evidence.get("recent_close_gate_unverified_loss_net_jpy"),
                    ),
                    3,
                ),
                "not_passing_close_gate_evidence": evidence.get(
                    "recent_close_gate_not_passing_loss_closes"
                ),
                "latest_missing_evidence_ts_utc": latest,
                "earliest_auto_clear_if_no_new_missing_utc": _plus_days_iso(
                    latest,
                    ACCEPTANCE_LEAK_LOOKBACK_DAYS,
                ),
                "example_trade_ids": _example_trade_ids(evidence),
            },
        )
    if code == "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING":
        latest = _latest_ts_from_examples(evidence)
        return (
            "every recent GPT loss-side market close has PASS close_gate_evidence in verification_observations, "
            "or the failed-evidence closes age out of the 7-day acceptance window without new leaks",
            "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
            {
                "not_passing_close_gate_evidence": evidence.get(
                    "recent_close_gate_not_passing_loss_closes",
                    evidence.get("recent_close_gate_unverified_loss_closes"),
                ),
                "not_passing_close_gate_net_jpy": _round_optional(
                    evidence.get(
                        "recent_close_gate_not_passing_loss_net_jpy",
                        evidence.get("recent_close_gate_unverified_loss_net_jpy"),
                    ),
                    3,
                ),
                "missing_close_gate_evidence": evidence.get(
                    "recent_close_gate_missing_loss_closes"
                ),
                "latest_not_passing_evidence_ts_utc": latest,
                "earliest_auto_clear_if_no_new_not_passing_utc": _plus_days_iso(
                    latest,
                    ACCEPTANCE_LEAK_LOOKBACK_DAYS,
                ),
                "example_trade_ids": _example_trade_ids(evidence),
            },
        )
    if code == "TP_PROGRESS_REPLAY_REPAIR_UNPROVED":
        return (
            "execution-timing-audit reports zero loss_closes_repair_replay_triggered under the "
            "current production-gate replay contract after the TP-progress TAKE_PROFIT_MARKET "
            "path and position guardian have had a live window to capture executable plus P/L "
            "before red closes",
            "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80",
            {
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "loss_closes_repair_replay_triggered": evidence.get(
                    "loss_closes_repair_replay_triggered"
                ),
                "counterfactual_profit_capture_delta_jpy": _round_optional(
                    evidence.get("counterfactual_profit_capture_delta_jpy"),
                    3,
                ),
                "counterfactual_profit_capture_jpy": _round_optional(
                    evidence.get("counterfactual_profit_capture_jpy"),
                    3,
                ),
                "example_trade_ids": _example_trade_ids_from_top_misses(evidence),
                "clearance_condition": evidence.get("clearance_condition"),
            },
        )
    if code == "MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED":
        return (
            "the active execution-timing-audit covers at least 720 hours with the current "
            "TP-progress production-gate replay contract before TP-positive / market-close-negative "
            "economics are treated as repaired",
            "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6",
            {
                "take_profit_net_jpy": _round_optional(evidence.get("take_profit_net_jpy"), 3),
                "market_close_net_jpy": _round_optional(evidence.get("market_close_net_jpy"), 3),
                "window_lookback_hours": _round_optional(evidence.get("window_lookback_hours"), 3),
                "required_lookback_hours": _round_optional(
                    evidence.get("required_lookback_hours"),
                    3,
                ),
                "repair_replay_contract": evidence.get("repair_replay_contract"),
                "repair_replay_contract_present": evidence.get("repair_replay_contract_present"),
            },
        )
    if code == "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE":
        residual_groups = (
            evidence.get("top_repair_replay_residual_groups")
            if isinstance(evidence.get("top_repair_replay_residual_groups"), list)
            else []
        )
        return (
            "month-scale production-gate replay is non-negative, or the top residual "
            "pair/side/method groups are removed by close-gate, TP-capture, or entry-selection "
            "changes before turnover is scaled",
            "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --lookback-hours 744 --post-close-hours 6",
            {
                "window_lookback_hours": _round_optional(evidence.get("window_lookback_hours"), 3),
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "loss_closes_repair_replay_triggered": evidence.get(
                    "loss_closes_repair_replay_triggered"
                ),
                "repair_replay_counterfactual_pl_jpy": _round_optional(
                    evidence.get("repair_replay_counterfactual_pl_jpy"),
                    3,
                ),
                "active_counterfactual_profit_capture_pl_jpy": _round_optional(
                    evidence.get("active_counterfactual_profit_capture_pl_jpy"),
                    3,
                ),
                "counterfactual_profit_capture_delta_jpy": _round_optional(
                    evidence.get("counterfactual_profit_capture_delta_jpy"),
                    3,
                ),
                "top_repair_replay_residual_groups": residual_groups[:3],
            },
        )
    if code == "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED":
        return (
            "position guardian is proven active with a fresh heartbeat, then the TP-progress "
            "production-gate replay is rerun after a live window and reports zero "
            "loss_closes_repair_replay_triggered",
            "scripts/install-position-guardian.sh --status",
            {
                "guardian_profit_capture_inactive": evidence.get(
                    "guardian_profit_capture_inactive"
                ),
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "loss_closes_repair_replay_triggered": evidence.get(
                    "loss_closes_repair_replay_triggered"
                ),
                "repair_replay_contract": evidence.get("repair_replay_contract"),
                "example_trade_ids": _example_trade_ids_from_top_misses(evidence),
                "clearance_condition": evidence.get("clearance_condition"),
            },
        )
    if code == "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING":
        return (
            "execution-timing-audit is regenerated by the current runtime and includes "
            f"{TP_PROGRESS_REPAIR_REPLAY_CONTRACT} before TP-progress repair is judged clean",
            "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80",
            {
                "loss_closes_profit_capture_missed": evidence.get(
                    "loss_closes_profit_capture_missed"
                ),
                "repair_replay_contract": evidence.get("repair_replay_contract"),
                "required_repair_replay_contract": evidence.get(
                    "required_repair_replay_contract",
                    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                ),
                "example_trade_ids": _example_trade_ids_from_top_misses(evidence),
                "clearance_condition": evidence.get("clearance_condition"),
            },
        )
    if code in {
        "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL",
        "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE",
        "BIDASK_CONTRARIAN_EDGE_NOT_DAILY_STABLE",
    }:
        bidask = metrics.get("bidask_replay_rules") if isinstance(metrics.get("bidask_replay_rules"), dict) else {}
        bidask = bidask or evidence
        examples = bidask.get("rank_only_examples") if isinstance(bidask.get("rank_only_examples"), list) else []
        validation_command = bidask.get("replay_validation_command") or (
            "python3 scripts/oanda_history_replay_validate.py "
            "--forecast-history data/forecast_history.jsonl "
            "--granularity S5"
        )
        if code == "BIDASK_REPLAY_PRICE_TRUTH_PARTIAL":
            condition = (
                "missing OANDA bid/ask price-truth windows are fetched and the refreshed replay "
                "report either reaches PRICE_TRUTH_OK or proves that any remaining candidate is "
                "only rank/audit evidence"
            )
        elif code == "BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE":
            condition = (
                "at least one bid/ask replay support rule graduates from rank-only to live-grade DAILY_STABLE "
                "after fresh multi-week OANDA BA candle replay; until then these candidates are advisory "
                "ranking evidence and cannot be counted as high-turn daily firepower"
            )
        else:
            condition = (
                "at least one bid/ask contrarian replay rule graduates from rank-only to DAILY_STABLE after "
                "fresh multi-week OANDA BA candle replay; until then weak forecast inversion is advisory ranking "
                "evidence and cannot be counted as live-grade turnover firepower"
            )
        return (
            condition,
            str(validation_command),
            {
                "support_rules": bidask.get("support_rules"),
                "daily_stable_support_rules": bidask.get("daily_stable_support_rules"),
                "rank_only_support_rules": bidask.get("rank_only_support_rules"),
                "edge_rules": bidask.get("edge_rules"),
                "daily_stable_edge_rules": bidask.get("daily_stable_edge_rules"),
                "rank_only_edge_rules": bidask.get("rank_only_edge_rules"),
                "contrarian_edge_rules": bidask.get("contrarian_edge_rules"),
                "daily_stable_contrarian_edge_rules": bidask.get("daily_stable_contrarian_edge_rules"),
                "rank_only_contrarian_edge_rules": bidask.get("rank_only_contrarian_edge_rules"),
                "negative_rules": bidask.get("negative_rules"),
                "price_truth_coverage": bidask.get("price_truth_coverage"),
                "daily_stability_requirements": bidask.get("daily_stability_requirements"),
                "history_fetch_command": bidask.get("history_fetch_command"),
                "history_fetch_command_count": (
                    (bidask.get("price_truth_coverage") or {}).get("history_fetch_command_count")
                    if isinstance(bidask.get("price_truth_coverage"), dict)
                    else None
                ),
                "history_fetch_command_mode": (
                    (bidask.get("price_truth_coverage") or {}).get("history_fetch_command_mode")
                    if isinstance(bidask.get("price_truth_coverage"), dict)
                    else None
                ),
                "missing_price_window_group_count": (
                    (bidask.get("price_truth_coverage") or {}).get("missing_price_window_group_count")
                    if isinstance(bidask.get("price_truth_coverage"), dict)
                    else None
                ),
                "rank_only_examples": examples[:3],
            },
        )
    if code == "EXECUTION_LEDGER_GATEWAY_RECEIPT_STREAM_STALE":
        return (
            "execution_ledger gateway receipt stream is fresh enough to classify recent broker market-close "
            "truth against local GPT/gateway receipts",
            "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
            {
                "gateway_event_stream_lag_minutes": evidence.get("gateway_event_stream_lag_minutes"),
                "latest_gateway_market_close_ts_utc": evidence.get("latest_gateway_market_close_ts_utc"),
            },
        )
    return (
        "the named acceptance P0 finding disappears from profitability_acceptance after its evidence metric changes",
        "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
        {key: evidence.get(key) for key in sorted(evidence)[:6]},
    )


def _split_blocker(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        return raw.strip() or "UNKNOWN_ACCEPTANCE_BLOCKER", raw.strip()
    code, message = raw.split(":", 1)
    return code.strip() or "UNKNOWN_ACCEPTANCE_BLOCKER", message.strip()


def _latest_ts_from_examples(evidence: dict[str, Any]) -> str | None:
    examples = evidence.get("examples") if isinstance(evidence.get("examples"), list) else []
    latest: datetime | None = None
    latest_raw: str | None = None
    for item in examples:
        if not isinstance(item, dict):
            continue
        raw = (
            item.get("ts_utc")
            or item.get("closed_at_utc")
            or item.get("trade_close_ts_utc")
            or item.get("timestamp_utc")
        )
        parsed = _parse_utc(raw)
        if parsed is None:
            continue
        if latest is None or parsed > latest:
            latest = parsed
            latest_raw = parsed.isoformat()
    return latest_raw


def _plus_days_iso(value: Any, days: int) -> str | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    return (parsed + timedelta(days=days)).isoformat()


def _example_trade_ids(evidence: dict[str, Any]) -> list[str]:
    examples = evidence.get("examples") if isinstance(evidence.get("examples"), list) else []
    trade_ids: list[str] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        trade_id = item.get("trade_id")
        if trade_id is None:
            continue
        trade_ids.append(str(trade_id))
    return trade_ids[:8]


def _example_trade_ids_from_top_misses(evidence: dict[str, Any]) -> list[str]:
    top: list[Any] = []
    for key in ("top_profit_capture_misses", "top_repair_replay_triggers"):
        values = evidence.get(key)
        if isinstance(values, list):
            top.extend(values)
    trade_ids: list[str] = []
    for item in top:
        if not isinstance(item, dict):
            continue
        trade_id = item.get("trade_id")
        if trade_id is None:
            continue
        trade_ids.append(str(trade_id))
    trade_ids = list(dict.fromkeys(trade_ids))
    return trade_ids[:8]


def _target_firepower_summary(payload: dict[str, Any]) -> dict[str, Any]:
    high_precision = _firepower_bucket_summary(payload.get("high_precision"))
    evidence_queue = _firepower_bucket_summary(payload.get("evidence_queue"))
    bucket_name, bucket = _best_firepower_bucket(
        high_precision=high_precision,
        evidence_queue=evidence_queue,
    )
    minimum = _float(payload.get("minimum_return_pct")) or 5.0
    target = _float(payload.get("target_return_pct")) or 10.0
    estimated_return = _float(bucket.get("estimated_return_pct_per_active_day_at_observed_frequency"))
    return {
        "status": payload.get("status"),
        "target_open": payload.get("target_open"),
        "minimum_return_pct": _round_optional(minimum, 3),
        "target_return_pct": _round_optional(target, 3),
        "per_trade_risk_pct_lens": _round_optional(payload.get("per_trade_risk_pct_lens"), 3),
        "best_bucket": bucket_name,
        "minimum_5pct_estimated_reachable": bool(estimated_return >= minimum),
        "target_10pct_estimated_reachable": bool(estimated_return >= target),
        "audit_only_no_live_permission": True,
        "high_precision": high_precision,
        "evidence_queue": evidence_queue,
        "contract": payload.get("contract"),
    }


def _firepower_bucket_summary(raw: Any) -> dict[str, Any]:
    bucket = raw if isinstance(raw, dict) else {}
    return {
        "estimated_return_pct_per_active_day_at_observed_frequency": _round_optional(
            bucket.get("estimated_return_pct_per_active_day_at_observed_frequency"),
            6,
        ),
        "weighted_return_pct_per_trade_at_risk_lens": _round_optional(
            bucket.get("weighted_return_pct_per_trade_at_risk_lens"),
            6,
        ),
        "observed_attempts_per_active_day": _round_optional(
            bucket.get("observed_attempts_per_active_day"),
            6,
        ),
        "trades_needed_for_minimum_5pct_at_weighted_expectancy": bucket.get(
            "trades_needed_for_minimum_5pct_at_weighted_expectancy"
        ),
        "trades_needed_for_target_10pct_at_weighted_expectancy": bucket.get(
            "trades_needed_for_target_10pct_at_weighted_expectancy"
        ),
        "pair_count": bucket.get("pair_count"),
        "unique_vehicle_count": bucket.get("unique_vehicle_count"),
        "top_vehicle_keys": list(bucket.get("top_vehicle_keys") or [])[:5],
    }


def _best_firepower_bucket(
    *,
    high_precision: dict[str, Any],
    evidence_queue: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    candidates = [
        ("high_precision", high_precision),
        ("evidence_queue", evidence_queue),
    ]
    candidates.sort(
        key=lambda item: _float(item[1].get("estimated_return_pct_per_active_day_at_observed_frequency")),
        reverse=True,
    )
    name, bucket = candidates[0]
    if _float(bucket.get("estimated_return_pct_per_active_day_at_observed_frequency")) <= 0:
        return None, {}
    return name, bucket


def _build_blockers(
    *,
    guardian: dict[str, Any],
    target: dict[str, Any],
    profit_capture: dict[str, Any],
    p0_findings: list[dict[str, Any]],
    entry: dict[str, Any],
    acceptance: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if guardian["required"] and not guardian["active"]:
        blockers.append(
            {
                "code": "POSITION_GUARDIAN_INACTIVE",
                "severity": "P0",
                "message": (
                    "position guardian is required but not proven active with a fresh heartbeat; "
                    "fresh entries stay blocked because plus P/L can reverse between full cycles"
                ),
            }
        )
    if profit_capture["missed_loss_closes"] > 0:
        production_gate_block = (
            not profit_capture.get("repair_replay_contract_present")
            or int(_float(profit_capture.get("repair_replay_triggered"))) > 0
        )
        blockers.append(
            {
                "code": PROFIT_CAPTURE_MISS,
                "severity": "P0" if production_gate_block else "P1",
                "message": _profit_capture_miss_message(profit_capture),
            }
        )
    if p0_findings:
        blockers.append(
            {
                "code": "SELF_IMPROVEMENT_P0_PRESENT",
                "severity": "P0",
                "message": f"self-improvement audit has {len(p0_findings)} P0 finding(s)",
            }
        )
    if target.get("target_open") and entry["live_ready_lanes"] == 0:
        blockers.append(
            {
                "code": "NO_LIVE_READY_LANES",
                "severity": "P1",
                "message": "daily target is open but no lane is LIVE_READY",
            }
        )
    if str(acceptance.get("status") or "") not in {"", "PROFITABILITY_ACCEPTANCE_PASSED", "PASSED"}:
        blockers.append(
            {
                "code": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                "severity": "P0",
                "message": f"profitability acceptance status is {acceptance.get('status')}",
            }
        )
    return blockers


def _profit_capture_miss_message(profit_capture: dict[str, Any]) -> str:
    message = (
        profit_capture.get("message")
        or f"{profit_capture['missed_loss_closes']} losing close(s) missed TP-progress capture"
    )
    repair_triggers = int(_float(profit_capture.get("repair_replay_triggered")))
    if repair_triggers > 0:
        repair_delta = profit_capture.get("repair_replay_delta_jpy")
        if repair_delta is None:
            return f"{message}; production-gate replay triggers={repair_triggers}"
        return f"{message}; production-gate replay triggers={repair_triggers} delta={repair_delta} JPY"
    delta = profit_capture.get("counterfactual_profit_capture_delta_jpy")
    if delta is None:
        return message
    return f"{message}; conservative candle counterfactual delta={delta} JPY"


def _operator_actions(
    *,
    status: str,
    blockers: list[dict[str, Any]],
    guardian: dict[str, Any],
    profit_capture: dict[str, Any],
    entry: dict[str, Any],
    acceptance: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        {
            "code": "REFRESH_SUPPORT_PANEL",
            "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
            "requires_explicit_operator_approval": False,
            "reason": "read-only status refresh for trader operations",
        }
    ]
    if guardian["required"] and not guardian["active"]:
        actions.extend(
            [
                {
                    "code": "CHECK_POSITION_GUARDIAN_PREFLIGHT",
                    "command": "scripts/install-position-guardian.sh --check",
                    "requires_explicit_operator_approval": False,
                    "reason": "verify live runtime can safely run the fast profit-capture guardian",
                },
                {
                    "code": "CHECK_POSITION_GUARDIAN_STATUS",
                    "command": "scripts/install-position-guardian.sh --status",
                    "requires_explicit_operator_approval": False,
                    "reason": "show plist, launchd, and heartbeat state without changing live services",
                },
                {
                    "code": "LOAD_POSITION_GUARDIAN_ONLY_IF_APPROVED",
                    "command": "scripts/install-position-guardian.sh",
                    "requires_explicit_operator_approval": True,
                    "reason": "loading launchd changes live support services and must be explicit",
                },
            ]
        )
    if profit_capture["missed_loss_closes"] > 0:
        actions.append(
            {
                "code": "RECHECK_TIMING_CAPTURE_MISSES",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80",
                "requires_explicit_operator_approval": False,
                "reason": "recompute TP-progress misses and confirm capture gap is shrinking",
            }
        )
    if entry["live_ready_lanes"] == 0:
        actions.append(
            {
                "code": "REFRESH_EVIDENCE_PACKET",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10",
                "requires_explicit_operator_approval": False,
                "reason": "refresh forecasts, intents, sidecar audits, and route from current broker truth",
            }
        )
    if acceptance.get("status") == "PROFITABILITY_ACCEPTANCE_BLOCKED":
        actions.append(
            {
                "code": "READ_ACCEPTANCE_BLOCKERS",
                "command": "sed -n '1,220p' docs/profitability_acceptance_report.md",
                "requires_explicit_operator_approval": False,
                "reason": "inspect red/green acceptance invariants before increasing turnover",
            }
        )
    repair_plan = acceptance.get("repair_plan") if isinstance(acceptance.get("repair_plan"), dict) else {}
    repair_items = repair_plan.get("items") if isinstance(repair_plan.get("items"), list) else []
    evidence_items = (
        repair_plan.get("evidence_collection_items")
        if isinstance(repair_plan.get("evidence_collection_items"), list)
        else []
    )
    if repair_items:
        actions.append(
            {
                "code": "FOLLOW_ACCEPTANCE_REPAIR_PLAN",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "use profitability_acceptance.repair_plan clearance conditions; rerunning acceptance "
                    "alone will loop until those proof metrics change"
                ),
            }
        )
    if evidence_items:
        first_evidence = evidence_items[0] if isinstance(evidence_items[0], dict) else {}
        summary = (
            first_evidence.get("evidence_summary")
            if isinstance(first_evidence.get("evidence_summary"), dict)
            else {}
        )
        history_fetch = summary.get("history_fetch_command")
        if history_fetch:
            actions.append(
                {
                    "code": "FETCH_BIDASK_REPLAY_HISTORY",
                    "command": str(history_fetch),
                    "requires_explicit_operator_approval": False,
                    "reason": "read-only OANDA BA candle fetch for rank-only bid/ask replay validation",
                }
            )
        validation_command = first_evidence.get("verification_command")
        if validation_command:
            actions.append(
                {
                    "code": "VALIDATE_BIDASK_REPLAY_HISTORY",
                    "command": str(validation_command),
                    "requires_explicit_operator_approval": False,
                    "reason": "rerun historical bid/ask replay before treating replay support as live-grade",
                }
            )
    repair_codes = {str(item.get("code")) for item in repair_items if isinstance(item, dict)}
    if "LOSS_CLOSE_GATE_EVIDENCE_MISSING" in repair_codes:
        actions.append(
            {
                "code": "VERIFY_CLOSE_GATE_EVIDENCE",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli verification-ledger-audit",
                "requires_explicit_operator_approval": False,
                "reason": "confirm future GPT loss-side closes have durable PASS close_gate_evidence",
            }
        )
    if "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK" in repair_codes:
        actions.append(
            {
                "code": "RECHECK_LOSS_CLOSE_LEAK_WINDOW",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80",
                "requires_explicit_operator_approval": False,
                "reason": "verify the 7-day loss-close leak window is shrinking before adding turnover",
            }
        )
    if {
        "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
        "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
        "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING",
        "MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED",
        "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
    } & repair_codes:
        actions.append(
            {
                "code": "VERIFY_TP_PROGRESS_REPLAY_REPAIR",
                "command": (
                    "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit "
                    "--lookback-hours 744 --post-close-hours 6"
                    if {
                        "MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED",
                        "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                    }
                    & repair_codes
                    else "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit --max-events 80"
                ),
                "requires_explicit_operator_approval": False,
                "reason": (
                    "prove the OANDA candle replay TP-progress miss has cleared at the "
                    "required coverage before treating high-turnover profit capture as repaired"
                ),
            }
        )
    if entry.get("global_unlock_frontier"):
        actions.append(
            {
                "code": "WORK_GLOBAL_UNLOCK_FRONTIER",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "some lanes are blocked only by global support/profitability gates; "
                    "clear those gates before adding unrelated indicators"
                ),
            }
        )
    if entry.get("repair_frontier_remaining_blockers"):
        top = entry["repair_frontier_remaining_blockers"][0]
        actions.append(
            {
                "code": "WORK_REPAIR_FRONTIER_REMAINING_BLOCKERS",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "guardian/global repairs are not enough; top remaining repair-frontier blocker is "
                    f"{top.get('code')} across {top.get('count')} lane(s)"
                ),
            }
        )
    if entry.get("oanda_audit_only_local_tp_proof_required"):
        pairs = sorted(
            {
                str(item.get("pair"))
                for item in entry["oanda_audit_only_local_tp_proof_required"]
                if item.get("pair")
            }
        )
        pair_arg = ",".join(pairs) if pairs else "EUR_USD"
        actions.append(
            {
                "code": "MINE_LOCAL_TP_PROOF_FOR_OANDA_AUDIT_ONLY",
                "command": (
                    "PYTHONPATH=src python3 scripts/oanda_history_fetch.py "
                    f"--pairs {pair_arg} --granularities {OANDA_AUDIT_ONLY_REPLAY_HISTORY_GRANULARITIES} --price BA "
                    "--days 120 --output-dir logs/replay/oanda_history"
                ),
                "requires_explicit_operator_approval": False,
                "reason": (
                    "OANDA campaign firepower is audit-only for these lanes; fetch both S5 truth for "
                    "forecast replay and M5 truth for universal-rotation mining before treating the "
                    "candidate as improved evidence"
                ),
            }
        )
        actions.append(
            {
                "code": "VALIDATE_OANDA_AUDIT_ONLY_BIDASK_REPLAY",
                "command": (
                    "PYTHONPATH=src python3 scripts/oanda_history_replay_validate.py "
                    "--history-dir logs/replay/oanda_history --granularity S5"
                ),
                "requires_explicit_operator_approval": False,
                "reason": (
                    "score forecast_history against local OANDA S5 bid/ask candles with spread included; "
                    "this proves whether the prediction side actually made money on historical candles"
                ),
            }
        )
        actions.append(
            {
                "code": "MINE_OANDA_AUDIT_ONLY_CAMPAIGN_FIREPOWER",
                "command": (
                    "PYTHONPATH=src python3 scripts/oanda_universal_rotation_miner.py "
                    "--history-root logs/replay/oanda_history --history-glob '*_M5_BA_*.jsonl' "
                    f"--pairs {pair_arg}"
                ),
                "requires_explicit_operator_approval": False,
                "reason": (
                    "rerun the multi-month M5 bid/ask candle miner for the audit-only pairs so "
                    "improved range/failed-break/pullback vehicles are tested before reinsertion"
                ),
            }
        )
        actions.append(
            {
                "code": "PACKAGE_OANDA_AUDIT_ONLY_FIREPOWER_RULES_AFTER_REVIEW",
                "command": "PYTHONPATH=src python3 scripts/package_oanda_universal_rotation_rules.py",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "packages mined OANDA replay rows into the tracked runtime rule artifact; "
                    "Codex may run this after reviewing mining evidence, then test, commit, and sync "
                    "before live runtime uses the new evidence"
                ),
            }
        )
        actions.append(
            {
                "code": "RERUN_INTENTS_AFTER_OANDA_AUDIT_ONLY_REPLAY",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli cycle-refresh --daily-risk-pct 10",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "rerun the full evidence/intents/acceptance loop after replay mining so improved "
                    "evidence is revalidated instead of leaving the old blocked packet in place"
                ),
            }
        )
    target_firepower = acceptance.get("target_firepower") if isinstance(acceptance.get("target_firepower"), dict) else {}
    if target_firepower.get("minimum_5pct_estimated_reachable"):
        actions.append(
            {
                "code": "WORK_TARGET_FIREPOWER_BLOCKERS",
                "command": "PYTHONPATH=src python3 -m quant_rabbit.cli profitability-acceptance",
                "requires_explicit_operator_approval": False,
                "reason": (
                    "firepower audit estimates enough turnover for the 5% floor, but live permission "
                    "still depends on clearing acceptance, guardian, and lane blockers"
                ),
            }
        )
    if status == STATUS_READY:
        actions.append(
            {
                "code": "RUN_NEXT_TRADER_CYCLE",
                "command": "scripts/run-autotrade-live.sh --send",
                "requires_explicit_operator_approval": True,
                "reason": "support gates are green; live send still requires normal trader/gateway validation",
            }
        )
    # Keep blocker-specific actions unique while preserving order.
    seen = set()
    unique = []
    for action in actions:
        code = action["code"]
        if code in seen:
            continue
        seen.add(code)
        unique.append(action)
    return unique


def _build_repair_requests(
    *,
    guardian: dict[str, Any],
    profit_capture: dict[str, Any],
    entry: dict[str, Any],
    acceptance: dict[str, Any],
) -> list[dict[str, Any]]:
    repair_plan = acceptance.get("repair_plan") if isinstance(acceptance.get("repair_plan"), dict) else {}
    raw_items = repair_plan.get("items") if isinstance(repair_plan.get("items"), list) else []
    items = [item for item in raw_items if isinstance(item, dict)]
    item_by_code = {str(item.get("code")): item for item in items if item.get("code")}
    raw_evidence_items = (
        repair_plan.get("evidence_collection_items")
        if isinstance(repair_plan.get("evidence_collection_items"), list)
        else []
    )
    evidence_items = [item for item in raw_evidence_items if isinstance(item, dict)]
    requests: list[dict[str, Any]] = []

    if "LOSS_CLOSE_GATE_EVIDENCE_MISSING" in item_by_code:
        item = item_by_code["LOSS_CLOSE_GATE_EVIDENCE_MISSING"]
        requests.append(
            _repair_request(
                code="REPAIR_CLOSE_GATE_EVIDENCE_PERSISTENCE",
                priority="P0",
                status="READY_FOR_CODE_REPAIR",
                source_findings=[
                    code
                    for code in (
                        "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                        "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                    )
                    if code in item_by_code
                ],
                problem=(
                    "Loss-side market closes can remain unverified because durable close_gate_evidence "
                    "is missing from the verification ledger."
                ),
                why_now=(
                    "Profitability acceptance will keep blocking high-turn scaling until every recent "
                    "GPT/gateway loss close has PASS close_gate_evidence or ages out."
                ),
                evidence_summary=item.get("evidence_summary"),
                clearance_conditions=[item.get("clearance_condition")],
                verification_commands=[item.get("verification_command")],
                suggested_files=[
                    "src/quant_rabbit/gpt_trader.py",
                    "src/quant_rabbit/verification_ledger.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "tests/test_gpt_trader.py",
                    "tests/test_verification_ledger.py",
                    "tests/test_cli.py",
                ],
                required_tests=[
                    "Regression: a GPT loss-side close without PASS close_gate_evidence remains blocked.",
                    "Positive path: a contained GPT loss-side close with matching PASS close_gate_evidence clears the acceptance blocker.",
                    "Ledger path: close_gate_evidence is written to verification_observations for accepted and rejected CLOSE receipts.",
                ],
            )
        )

    if "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING" in item_by_code:
        item = item_by_code["LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING"]
        requests.append(
            _repair_request(
                code="REVIEW_CLOSE_GATE_EVIDENCE_FAILURES",
                priority="P0",
                status="HISTORICAL_ACCEPTANCE_WINDOW_ACTIVE",
                source_findings=[
                    code
                    for code in (
                        "LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING",
                        "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                    )
                    if code in item_by_code
                ],
                problem=(
                    "Recent GPT loss-side market closes now have durable close_gate_evidence, but that "
                    "evidence is BLOCK rather than PASS."
                ),
                why_now=(
                    "A code-repair loop must not synthesize PASS for historical closes. Acceptance remains "
                    "red until the failed-evidence closes age out or future GPT closes prove Gate A/B with PASS."
                ),
                evidence_summary=item.get("evidence_summary"),
                clearance_conditions=[item.get("clearance_condition")],
                verification_commands=[item.get("verification_command")],
                suggested_files=[
                    "src/quant_rabbit/gpt_trader.py",
                    "src/quant_rabbit/verification_ledger.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "tests/test_gpt_trader.py",
                    "tests/test_verification_ledger.py",
                    "tests/test_cli.py",
                ],
                required_tests=[
                    "Regression: historical BLOCK close_gate_evidence remains a P0 acceptance blocker.",
                    "Positive path: a future contained GPT loss-side close with PASS close_gate_evidence clears this blocker.",
                    "Loop guard: support bot does not label already-persisted BLOCK evidence as a persistence code repair.",
                ],
            )
        )

    tp_codes = [
        code
        for code in (
            "TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
            "TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
            "TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING",
        )
        if code in item_by_code
    ]
    if tp_codes:
        top_item = item_by_code[tp_codes[0]]
        requests.append(
            _repair_request(
                code="REPAIR_TP_PROGRESS_PROFIT_CAPTURE_REPLAY",
                priority="P0",
                status="READY_FOR_CODE_REPAIR",
                source_findings=tp_codes,
                problem=(
                    "Historical losing closes still show executable TP-progress profit that was not "
                    "captured before the close turned red."
                ),
                why_now=(
                    "Adding entry frequency before this capture path is proved clean repeats the "
                    "known TAKE_PROFIT_ORDER plus / MARKET_ORDER_TRADE_CLOSE minus leak."
                ),
                evidence_summary=top_item.get("evidence_summary"),
                clearance_conditions=[
                    item_by_code[code].get("clearance_condition")
                    for code in tp_codes
                    if isinstance(item_by_code.get(code), dict)
                ],
                verification_commands=[
                    item_by_code[code].get("verification_command")
                    for code in tp_codes
                    if isinstance(item_by_code.get(code), dict)
                ],
                suggested_files=[
                    "src/quant_rabbit/profit_capture_bot.py",
                    "src/quant_rabbit/execution_timing_audit.py",
                    "src/quant_rabbit/strategy/position_manager.py",
                    "src/quant_rabbit/strategy/position_protection_gateway.py",
                    "tests/test_execution_timing_audit.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Regression: TP-progress winner that later closes red is surfaced as repair_replay_triggered.",
                    "Positive path: production-gate replay reports zero loss_closes_repair_replay_triggered after the capture repair.",
                    "Safety path: support/profit-capture bots remain read-only and do not close positions directly.",
                ],
            )
        )

    if "MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE" in item_by_code:
        item = item_by_code["MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"]
        requests.append(
            _repair_request(
                code="REPAIR_MONTH_SCALE_RESIDUAL_ENTRY_QUALITY",
                priority="P0",
                status="READY_FOR_CODE_REPAIR",
                source_findings=["MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE"],
                problem=(
                    "Month-scale replay remains net negative after TP-progress repair; residual "
                    "pair/side/method groups must be blocked, reclassified, or given a tested escape."
                ),
                why_now=(
                    "The daily target cannot be scaled by frequency while the 744h production-gate replay "
                    "is still negative for the active residual groups."
                ),
                evidence_summary=item.get("evidence_summary"),
                clearance_conditions=[item.get("clearance_condition")],
                verification_commands=[item.get("verification_command")],
                suggested_files=[
                    "src/quant_rabbit/execution_timing_audit.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "src/quant_rabbit/strategy/intent_generator.py",
                    "tests/test_execution_timing_audit.py",
                    "tests/test_profitability_acceptance_replay_repair.py",
                    "tests/test_intent_generator.py",
                ],
                required_tests=[
                    "Regression: a matching residual pair/side/method cannot become LIVE_READY while month replay is negative.",
                    "Positive path: a non-matching pair/side/method with current evidence is not blocked by the residual group.",
                    "Metric path: TP-progress residuals and entry-quality residuals stay split in acceptance evidence.",
                ],
            )
        )

    if guardian.get("required") and not guardian.get("active"):
        requests.append(
            _repair_request(
                code="RESTORE_POSITION_GUARDIAN_AFTER_PREFLIGHT",
                priority="P0",
                status="OPERATOR_APPROVAL_REQUIRED",
                source_findings=["POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE"],
                problem=(
                    "Fresh entries are blocked because the fast position guardian is not proven active "
                    "with a fresh heartbeat."
                ),
                why_now=(
                    "The guardian is the between-cycle support path that should bank TP-progress profit "
                    "before it reverses into a later market-close loss."
                ),
                evidence_summary={
                    "active": guardian.get("active"),
                    "active_source": guardian.get("active_source"),
                    "launchd_loaded": guardian.get("launchd_loaded"),
                    "heartbeat_fresh": guardian.get("heartbeat_fresh"),
                    "heartbeat_age_seconds": guardian.get("heartbeat_age_seconds"),
                },
                clearance_conditions=[
                    "scripts/install-position-guardian.sh --check passes, then an operator explicitly approves load/reload and heartbeat becomes fresh."
                ],
                verification_commands=[
                    "scripts/install-position-guardian.sh --check",
                    "scripts/install-position-guardian.sh --status",
                ],
                suggested_files=[
                    "scripts/install-position-guardian.sh",
                    "scripts/run-position-guardian-live.sh",
                    "tests/test_position_guardian_install.py",
                ],
                required_tests=[
                    "Preflight blocks unsynced live runtime before launchd can be loaded.",
                    "Support bot shows guardian recovery candidates without loading launchd itself.",
                ],
                requires_explicit_operator_approval=True,
            )
        )

    if evidence_items:
        first = evidence_items[0]
        summary = first.get("evidence_summary") if isinstance(first.get("evidence_summary"), dict) else {}
        requests.append(
            _repair_request(
                code="COLLECT_BIDASK_REPLAY_EVIDENCE",
                priority=str(first.get("priority") or "P1"),
                status="READY_FOR_READ_ONLY_EVIDENCE_COLLECTION",
                source_findings=[str(item.get("code")) for item in evidence_items if item.get("code")],
                problem=(
                    "Bid/ask replay support is still rank-only or partially missing price truth, so it cannot "
                    "be counted as live-grade high-turn firepower."
                ),
                why_now=(
                    "Historical replay may expose usable short-horizon edges, but only after spread-included "
                    "OANDA BA candles prove daily-stable economics."
                ),
                evidence_summary=summary,
                clearance_conditions=[first.get("clearance_condition")],
                verification_commands=[
                    summary.get("history_fetch_command"),
                    first.get("verification_command"),
                ],
                suggested_files=[
                    "scripts/oanda_history_fetch.py",
                    "scripts/oanda_history_replay_validate.py",
                    "scripts/oanda_universal_rotation_miner.py",
                    "src/quant_rabbit/profitability_acceptance.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Read-only fetch/validation commands are surfaced without granting live permission.",
                    "Rank-only replay rules cannot clear profitability acceptance until daily-stable requirements pass.",
                ],
            )
        )

    frontier_blockers = (
        entry.get("repair_frontier_remaining_blockers")
        if isinstance(entry.get("repair_frontier_remaining_blockers"), list)
        else []
    )
    if frontier_blockers:
        top = frontier_blockers[0] if isinstance(frontier_blockers[0], dict) else {}
        code = str(top.get("code") or "UNKNOWN_REPAIR_FRONTIER_BLOCKER")
        requests.append(
            _repair_request(
                code="REPAIR_FRONTIER_LANE_BLOCKER",
                priority="P1",
                status="READY_FOR_CODE_OR_EVIDENCE_REPAIR",
                source_findings=[code],
                problem=(
                    "After global support gates are removed, repair-frontier lanes still have lane-local blockers."
                ),
                why_now=(
                    "This is the next non-guardian blocker that would keep high-turn repair baskets from becoming executable."
                ),
                evidence_summary=top,
                clearance_conditions=[
                    f"{code} disappears from repair_frontier_remaining_blockers or gains a tested downgrade path."
                ],
                verification_commands=[
                    "PYTHONPATH=src python3 -m quant_rabbit.cli trader-support-bot",
                    "PYTHONPATH=src python3 -m quant_rabbit.cli generate-intents --reuse-market-artifacts",
                ],
                suggested_files=[
                    "src/quant_rabbit/strategy/intent_generator.py",
                    "tests/test_intent_generator.py",
                    "tests/test_trader_support_bot.py",
                ],
                required_tests=[
                    "Regression: the original invalid blocker shape remains blocked.",
                    "Positive path: a current valid repair-frontier lane is allowed or explicitly downgraded with an escape condition.",
                ],
            )
        )

    return _unique_repair_requests(requests)


def _repair_request(
    *,
    code: str,
    priority: str,
    status: str,
    source_findings: list[str],
    problem: str,
    why_now: str,
    evidence_summary: Any,
    clearance_conditions: list[Any],
    verification_commands: list[Any],
    suggested_files: list[str],
    required_tests: list[str],
    requires_explicit_operator_approval: bool = False,
) -> dict[str, Any]:
    return {
        "contract_version": REPAIR_REQUEST_CONTRACT_VERSION,
        "code": code,
        "priority": priority,
        "status": status,
        "source_findings": [str(item) for item in source_findings if str(item)],
        "problem": problem,
        "why_now": why_now,
        "evidence_summary": evidence_summary if isinstance(evidence_summary, dict) else {},
        "clearance_conditions": [str(item) for item in clearance_conditions if item],
        "verification_commands": list(dict.fromkeys(str(item) for item in verification_commands if item)),
        "suggested_files": suggested_files,
        "required_tests": required_tests,
        "automation_contract": {
            "codex_may_execute": REPAIR_AUTOMATION_ALLOWED_ACTIONS,
            "commit_and_live_sync_required": True,
            "quant_rabbit_code_may_call_model_api": False,
            "live_side_effects_allowed": [],
            "requires_explicit_operator_approval_for": REPAIR_AUTOMATION_EXPLICIT_APPROVAL_ACTIONS,
            "forbidden_direct_actions": REPAIR_AUTOMATION_FORBIDDEN_DIRECT_ACTIONS,
            "orders_closes_launchd_policy": (
                "Order send, order cancel, position close, and launchd load/reload must go through "
                "explicit operator approval or the existing gateway path; a repair request alone is never live permission."
            ),
        },
        "requires_explicit_operator_approval": requires_explicit_operator_approval,
        "read_only": True,
        "live_side_effects": [],
    }


def _unique_repair_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for request in requests:
        code = str(request.get("code") or "")
        if not code or code in seen:
            continue
        seen.add(code)
        unique.append(request)
    return unique[:12]


def _render_report(payload: dict[str, Any]) -> str:
    guardian = payload["guardian"]
    entry = payload["entry_readiness"]
    profit = payload["profit_capture"]
    current_profit = payload["current_profit_capture"]
    broker = payload["broker"]
    target = payload["target"]
    firepower = payload["profitability_acceptance"].get("target_firepower", {})
    acceptance_repair = payload["profitability_acceptance"].get("repair_plan", {})
    lines = [
        "# Trader Support Bot Report",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Read only: `{payload['read_only']}`",
        f"- Live side effects: `{len(payload['live_side_effects'])}`",
        "",
        "## Support Gates",
        "",
        "| Gate | Value |",
        "|---|---|",
        f"| Fresh entry send allowed | `{payload['metrics']['send_fresh_entries_allowed']}` |",
        f"| Repair basket send allowed | `{payload['metrics']['repair_basket_send_allowed']}` |",
        f"| Guardian active | `{guardian['active']}` source=`{guardian['active_source']}` |",
        f"| Guardian heartbeat fresh | `{guardian['heartbeat_fresh']}` age=`{guardian['heartbeat_age_seconds']}`s |",
        f"| LIVE_READY lanes | `{entry['live_ready_lanes']}` / `{entry['lanes']}` |",
        f"| Repair LIVE_READY lanes | `{len(entry['repair_live_ready'])}` |",
        f"| Repair lanes after guardian recovery | `{len(entry['repair_basket_guardian_recovery'])}` |",
        f"| Repair frontier lanes | `{len(entry['repair_frontier'])}` |",
        f"| RANGE forecast superseded repair lanes | `{len(entry['repair_frontier_superseded_by_range_forecast'])}` |",
        f"| OANDA audit-only local TP proof required | `{len(entry['oanda_audit_only_local_tp_proof_required'])}` |",
        f"| RANGE forecast missing counterpart lanes | `{len(entry['repair_frontier_missing_range_rotation_counterpart'])}` |",
        f"| Repair frontier clear after support | `{entry['repair_frontier_after_support_clear_lanes']}` |",
        f"| Repair frontier blocked after support | `{entry['repair_frontier_after_support_blocked_lanes']}` |",
        f"| Global unlock frontier lanes | `{len(entry['global_unlock_frontier'])}` |",
        f"| Profit-capture misses | `{profit['missed_loss_closes']}` gap=`{profit['estimated_gap_jpy']}` JPY "
        f"counterfactual_delta=`{profit['counterfactual_profit_capture_delta_jpy']}` JPY "
        f"repair_replay_triggered=`{profit['repair_replay_triggered']}` "
        f"repair_delta=`{profit['repair_replay_delta_jpy']}` JPY |",
        f"| Current profit-capture positions | bankable=`{current_profit['bankable_positions']}` watch=`{current_profit['watch_positions']}` blocked=`{current_profit['blocked_positions']}` |",
        f"| Open trader positions | `{broker['trader_positions']}` upl=`{broker['trader_unrealized_pl_jpy']}` JPY |",
        f"| Target remaining | `{target['remaining_target_jpy']}` JPY |",
        f"| Firepower 5% audit estimate | `{firepower.get('minimum_5pct_estimated_reachable')}` best=`{firepower.get('best_bucket')}` |",
        "",
        "## Blockers",
        "",
    ]
    if payload["blockers"]:
        for blocker in payload["blockers"]:
            lines.append(f"- `{blocker['severity']}` `{blocker['code']}`: {blocker['message']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Repair Requests", ""])
    repair_requests = (
        payload.get("repair_requests") if isinstance(payload.get("repair_requests"), list) else []
    )
    if repair_requests:
        lines.extend(
            [
                "| Code | Status | Source | Verify |",
                "|---|---|---|---|",
            ]
        )
        for request in repair_requests[:8]:
            source = ", ".join(f"`{code}`" for code in request.get("source_findings", []) or []) or "none"
            verify = ", ".join(
                f"`{command}`" for command in request.get("verification_commands", [])[:2]
            ) or "none"
            approval = " approval-required" if request.get("requires_explicit_operator_approval") else ""
            lines.append(
                f"| `{request.get('code')}` | `{request.get('status')}`{approval} | {source} | {verify} |"
            )
        lines.append("")
        lines.append(
            "- Automation contract: Codex may edit code/tests/docs, run tests, commit, and sync live; "
            "orders, cancels, closes, and launchd changes require explicit approval or the existing gateway path."
        )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Profit Capture Repair",
            "",
            f"- Status: `{profit['status']}`",
            f"- Actual loss-close PL JPY: `{profit['actual_loss_close_pl_jpy']}`",
            f"- Counterfactual profit-capture PL JPY: `{profit['counterfactual_profit_capture_pl_jpy']}`",
            f"- Counterfactual profit-capture delta JPY: `{profit['counterfactual_profit_capture_delta_jpy']}`",
            f"- Production-gate replay triggered: `{profit['repair_replay_triggered']}`",
            f"- Production-gate replay delta JPY: `{profit['repair_replay_delta_jpy']}`",
            f"- Clearance condition: {profit['clearance_condition']}",
            f"- Verify: `{profit['verification_command']}`",
        ]
    )
    if profit.get("top_repair_replay_triggers"):
        lines.extend(
            [
                "",
                "| Repair Trade | Pair | Side | Exit | Trigger UTC | Repair pips | Noise floor | Repair JPY | Delta JPY |",
                "|---|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in profit["top_repair_replay_triggers"][:5]:
            lines.append(
                f"| `{item.get('trade_id')}` | `{item.get('pair')}` | `{item.get('side')}` | "
                f"`{item.get('exit_reason')}` | `{item.get('repair_trigger_at_utc') or item.get('repair_replay_trigger_at_utc')}` | "
                f"`{_round_optional(item.get('repair_profit_pips') or item.get('repair_replay_profit_pips'), 4)}` | "
                f"`{_round_optional(item.get('repair_noise_floor_pips') or item.get('repair_replay_noise_floor_pips'), 4)}` | "
                f"`{_round_optional(item.get('repair_counterfactual_jpy') or item.get('repair_replay_counterfactual_jpy'), 3)}` | "
                f"`{_round_optional(item.get('repair_counterfactual_delta_jpy') or item.get('repair_replay_counterfactual_net_improvement_jpy'), 3)}` |"
            )
    if profit.get("top_misses"):
        lines.extend(
            [
                "",
                "| Trade | Pair | Side | Exit | MFE JPY | TP progress | Counterfactual JPY | Delta JPY | Realized JPY |",
                "|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in profit["top_misses"][:5]:
            lines.append(
                f"| `{item.get('trade_id')}` | `{item.get('pair')}` | `{item.get('side')}` | "
                f"`{item.get('exit_reason')}` | "
                f"`{_round_optional(item.get('estimated_mfe_jpy_before_loss_close'), 3)}` | "
                f"`{_round_optional(item.get('tp_progress_before_loss_close'), 4)}` | "
                f"`{_round_optional(item.get('profit_capture_counterfactual_jpy'), 3)}` | "
                f"`{_round_optional(item.get('profit_capture_counterfactual_net_improvement_jpy'), 3)}` | "
                f"`{_round_optional(item.get('realized_pl_jpy'), 3)}` |"
            )
    else:
        lines.append("- Missed capture examples: none")
    lines.extend(
        [
            "",
            "## Guardian",
            "",
            f"- Required: `{guardian['required']}`",
            f"- Label: `{guardian['label']}`",
            f"- Plist: `{guardian['plist_path']}` exists=`{guardian['plist_exists']}`",
            f"- Launchd loaded: `{guardian['launchd_loaded']}`",
            f"- Heartbeat path: `{guardian['heartbeat_path']}`",
            f"- Heartbeat generated: `{guardian['heartbeat_generated_at_utc']}`",
            "",
            "## Top Intent Blockers",
            "",
        ]
    )
    if entry["top_blockers"]:
        for item in entry["top_blockers"]:
            lines.append(f"- `{item['code']}`: `{item['count']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Guardian Recovery Candidates", ""])
    if entry["repair_basket_guardian_recovery"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | TP proof | Remaining blockers after guardian |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for item in entry["repair_basket_guardian_recovery"][:8]:
            remaining = ", ".join(
                item["remaining_blocker_codes_after_guardian_and_repair_exemption"]
            ) or "none"
            tp_proof = _format_tp_proof_report_cell(item.get("tp_proof"))
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | {tp_proof} | `{remaining}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Repair Frontier Blockers After Support", ""])
    if entry["repair_frontier_remaining_blockers"]:
        lines.extend(
            [
                "| Blocker | Lanes | Reward JPY | Examples |",
                "|---|---:|---:|---|",
            ]
        )
        for item in entry["repair_frontier_remaining_blockers"][:8]:
            examples = ", ".join(f"`{lane_id}`" for lane_id in item["example_lane_ids"]) or "none"
            lines.append(
                f"| `{item['code']}` | `{item['count']}` | `{item['reward_jpy']}` | {examples} |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Repair Frontier", ""])
    if entry["repair_frontier"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | Remaining blockers after support |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in entry["repair_frontier"][:8]:
            remaining = ", ".join(item["remaining_blocker_codes_after_guardian_and_repair_exemption"]) or "none"
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | `{remaining}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## OANDA Audit-Only Local TP Proof Required", ""])
    if entry["oanda_audit_only_local_tp_proof_required"]:
        lines.append(
            "- Historical replay can rank and mine these candidates, but it does not by itself "
            "clear the local TP proof gate; the escape condition is exact pair/side/method "
            "TAKE_PROFIT_ORDER proof with positive expectancy, zero TP losses, and positive "
            "Wilson-stressed expectancy."
        )
        lines.append("")
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Scope | Vehicle | Replay evidence | Historical clears local proof | Remaining blockers after guardian |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for item in entry["oanda_audit_only_local_tp_proof_required"][:8]:
            remaining = ", ".join(item["remaining_blocker_codes_after_guardian"]) or "none"
            replay = _format_oanda_replay_report_cell(item)
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item.get('capture_take_profit_scope')}` | "
                f"`{item.get('oanda_vehicle_key')}` | {replay} | "
                f"`{item.get('historical_replay_can_clear_local_tp_proof')}` | `{remaining}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## RANGE Forecast Superseded Repair Lanes", ""])
    if entry["repair_frontier_superseded_by_range_forecast"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | Range counterpart |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in entry["repair_frontier_superseded_by_range_forecast"][:8]:
            counterpart = item.get("superseded_by_range_rotation_lane_id") or "missing"
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | `{counterpart}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## RANGE Forecast Missing Counterpart Repair Lanes", ""])
    if entry["repair_frontier_missing_range_rotation_counterpart"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | Missing counterpart |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in entry["repair_frontier_missing_range_rotation_counterpart"][:8]:
            missing_pair_side = item.get("missing_range_rotation_counterpart_for") or []
            missing = ":".join(str(part) for part in missing_pair_side if part) or "unknown"
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | `{missing}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Global Unlock Frontier", ""])
    if entry["global_unlock_frontier"]:
        lines.extend(
            [
                "| Lane | Pair | Side | Method | Reward JPY | TP proof | Global blockers |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for item in entry["global_unlock_frontier"][:8]:
            blockers = ", ".join(item["global_blocker_codes"]) or "none"
            tp_proof = _format_tp_proof_report_cell(item.get("tp_proof"))
            lines.append(
                f"| `{item['lane_id']}` | `{item['pair']}` | `{item['side']}` | "
                f"`{item['method']}` | `{item['reward_jpy']}` | {tp_proof} | `{blockers}` |"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Target Firepower Evidence", ""])
    if firepower.get("status"):
        lines.append(f"- Status: `{firepower.get('status')}`")
        lines.append(f"- Best bucket: `{firepower.get('best_bucket')}`")
        lines.append(f"- 5% estimated reachable: `{firepower.get('minimum_5pct_estimated_reachable')}`")
        lines.append(f"- 10% estimated reachable: `{firepower.get('target_10pct_estimated_reachable')}`")
        lines.append("- Audit only, no live permission grant: `true`")
        for label in ("high_precision", "evidence_queue"):
            bucket = firepower.get(label) if isinstance(firepower.get(label), dict) else {}
            if bucket.get("estimated_return_pct_per_active_day_at_observed_frequency") is None:
                continue
            lines.append(
                f"- `{label}` return/day=`{bucket.get('estimated_return_pct_per_active_day_at_observed_frequency')}` "
                f"trades_needed_5pct=`{bucket.get('trades_needed_for_minimum_5pct_at_weighted_expectancy')}` "
                f"vehicles=`{bucket.get('top_vehicle_keys')}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Acceptance Repair Plan", ""])
    if acceptance_repair.get("items"):
        lines.append(f"- Loop breaker: {acceptance_repair.get('loop_breaker')}")
        lines.extend(
            [
                "",
                "| Code | Clearance condition | Verify |",
                "|---|---|---|",
            ]
        )
        for item in acceptance_repair["items"][:8]:
            lines.append(
                f"| `{item['code']}` | {item['clearance_condition']} | "
                f"`{item['verification_command']}` |"
            )
    else:
        lines.append("- none")
    evidence_items = (
        acceptance_repair.get("evidence_collection_items")
        if isinstance(acceptance_repair.get("evidence_collection_items"), list)
        else []
    )
    lines.extend(["", "## Acceptance Evidence Collection", ""])
    if evidence_items:
        lines.extend(
            [
                "| Code | Clearance condition | Verify |",
                "|---|---|---|",
            ]
        )
        for item in evidence_items[:8]:
            lines.append(
                f"| `{item['code']}` | {item['clearance_condition']} | "
                f"`{item['verification_command']}` |"
            )
            summary = item.get("evidence_summary") if isinstance(item.get("evidence_summary"), dict) else {}
            fetch = summary.get("history_fetch_command")
            if fetch:
                lines.append(f"| `{item['code']}:fetch` | Fetch fresh BA candles | `{fetch}` |")
    else:
        lines.append("- none")
    lines.extend(["", "## Current Profit Capture", ""])
    if current_profit["top_gate_statuses"]:
        for item in current_profit["top_gate_statuses"]:
            lines.append(
                f"- `{item['trade_id']}` `{item['pair']}` `{item['side']}` "
                f"gate=`{item['gate_status']}` progress=`{item['tp_progress']}` trigger=`{item['capture_trigger']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Open Trader Positions", ""])
    if broker["open_trader_positions"]:
        for item in broker["open_trader_positions"]:
            lines.append(
                f"- `{item['trade_id']}` `{item['pair']}` `{item['side']}` "
                f"units=`{item['units']}` upl=`{item['unrealized_pl_jpy']}` TP=`{item['take_profit']}` SL=`{item['stop_loss']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Operator Actions", ""])
    for action in payload["operator_actions"]:
        approval = " approval-required" if action.get("requires_explicit_operator_approval") else ""
        lines.append(f"- `{action['code']}`{approval}: `{action['command']}` — {action['reason']}")
    return "\n".join(lines) + "\n"


def _format_oanda_replay_report_cell(item: dict[str, Any]) -> str:
    evidence = item.get("oanda_replay_evidence") if isinstance(item.get("oanda_replay_evidence"), dict) else {}
    if not evidence or evidence.get("evidence_status") == "MISSING_OANDA_REPLAY_EVIDENCE":
        return "`missing`"
    parts = [
        f"status={evidence.get('evidence_status')}",
        f"n={evidence.get('validation_n')}",
        f"win={evidence.get('validation_win_rate')}",
        f"pf={evidence.get('validation_profit_factor')}",
        f"day%={evidence.get('estimated_return_pct_per_active_day_at_observed_frequency')}",
        f"5%trades={evidence.get('trades_needed_for_minimum_5pct')}",
    ]
    return "`" + " ".join(str(part) for part in parts) + "`"


def _format_tp_proof_report_cell(value: Any) -> str:
    proof = value if isinstance(value, dict) else {}
    if not proof:
        return "`missing`"
    parts: list[str] = []
    mode = proof.get("positive_rotation_mode")
    if mode is not None:
        parts.append(f"mode={mode}")
    scope = proof.get("capture_take_profit_scope")
    if scope is not None:
        parts.append(f"scope={scope}")
    trades = proof.get("capture_take_profit_trades")
    losses = proof.get("capture_take_profit_losses")
    if trades is not None or losses is not None:
        parts.append(f"trades={trades}")
        parts.append(f"losses={losses}")
    expectancy = _round_optional(proof.get("capture_take_profit_expectancy_jpy"), 3)
    if expectancy is not None:
        parts.append(f"tp_exp_jpy={expectancy}")
    pessimistic = _round_optional(proof.get("positive_rotation_pessimistic_expectancy_jpy"), 3)
    if pessimistic is not None:
        parts.append(f"pess_exp_jpy={pessimistic}")
    return "`" + " ".join(parts) + "`" if parts else "`missing`"


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round_optional(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
