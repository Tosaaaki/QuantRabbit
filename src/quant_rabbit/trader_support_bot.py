from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_GUARDIAN_EXECUTION,
    DEFAULT_POSITION_GUARDIAN_HEARTBEAT,
    DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT,
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
REPAIR_EXEMPTION_CODES = {PERSISTENT_DISCIPLINE, "SELF_IMPROVEMENT_P0_PROFITABILITY_DISCIPLINE"}


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

        guardian = _guardian_status(
            now_utc=self.now_utc,
            execution_path=self.position_guardian_execution_path,
            heartbeat_path=self.position_guardian_heartbeat_path,
        )
        broker_summary = _broker_summary(broker)
        target_summary = _target_summary(target)
        p0_findings = _p0_findings(self_improvement)
        profit_capture = _profit_capture_summary(self_improvement, timing)
        entry = _entry_readiness_summary(intents)
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
        send_allowed = (
            status == STATUS_READY
            and bool(entry["live_ready_lanes"])
            and (not guardian["required"] or bool(guardian["active"]))
        )
        metrics = {
            "send_fresh_entries_allowed": send_allowed,
            "guardian_active": guardian["active"],
            "guardian_heartbeat_fresh": guardian["heartbeat_fresh"],
            "live_ready_lanes": entry["live_ready_lanes"],
            "repair_frontier_lanes": len(entry["repair_frontier"]),
            "profit_capture_missed_loss_closes": profit_capture["missed_loss_closes"],
            "profit_capture_estimated_gap_jpy": profit_capture["estimated_gap_jpy"],
            "open_trader_positions": broker_summary["trader_positions"],
            "target_remaining_jpy": target_summary["remaining_target_jpy"],
            "profitability_status": acceptance["status"],
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
            },
            "generated_at_utc": generated,
            "status": status,
            "blockers": blockers,
            "operator_actions": actions,
            "metrics": metrics,
            "guardian": guardian,
            "broker": broker_summary,
            "target": target_summary,
            "profit_capture": profit_capture,
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
    missed = evidence.get("loss_closes_profit_capture_missed", summary.get("loss_closes_profit_capture_missed"))
    gap = evidence.get("loss_close_estimated_capture_gap_jpy", summary.get("loss_close_estimated_capture_gap_jpy"))
    top = evidence.get("top_profit_capture_misses") if isinstance(evidence.get("top_profit_capture_misses"), list) else []
    return {
        "status": "PROFIT_CAPTURE_REPAIR_REQUIRED" if _float(missed) > 0 else "OK",
        "missed_loss_closes": int(_float(missed)),
        "estimated_gap_jpy": _round_optional(gap, 3),
        "stop_loss_missed": evidence.get("stop_loss_closes_profit_capture_missed", summary.get("stop_loss_closes_profit_capture_missed")),
        "top_misses": top[:5],
        "message": finding.get("message") if finding else None,
        "next_action": finding.get("next_action") if finding else None,
    }


def _entry_readiness_summary(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    live_ready = [item for item in results if item.get("status") == "LIVE_READY"]
    codes = Counter()
    repair_frontier: list[dict[str, Any]] = []
    for item in results:
        for code in item.get("live_blocker_codes") or []:
            codes[str(code)] += 1
        metadata = _intent_metadata(item)
        if metadata.get("self_improvement_p0_repair_live_ready") is True:
            blocker_codes = [str(code) for code in item.get("live_blocker_codes") or []]
            exempt = set(REPAIR_EXEMPTION_CODES)
            remaining_after_support = [
                code for code in blocker_codes if code != GUARDIAN_BLOCKER and code not in exempt
            ]
            intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
            context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
            repair_frontier.append(
                {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "pair": intent.get("pair"),
                    "side": intent.get("side"),
                    "method": context.get("method"),
                    "order_type": intent.get("order_type"),
                    "reward_jpy": _round_optional(metadata.get("sizing_actual_reward_jpy"), 3),
                    "risk_jpy": _round_optional(metadata.get("sizing_actual_risk_jpy"), 3),
                    "repair_mode": metadata.get("self_improvement_p0_repair_mode"),
                    "matrix_repair_profile_status": metadata.get("matrix_repair_profile_status"),
                    "blocker_codes": blocker_codes,
                    "remaining_blocker_codes_after_guardian_and_repair_exemption": remaining_after_support,
                }
            )
    repair_frontier.sort(key=lambda item: _float(item.get("reward_jpy")), reverse=True)
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "lanes": len(results),
        "live_ready_lanes": len(live_ready),
        "live_ready_lane_ids": [item.get("lane_id") for item in live_ready],
        "top_blockers": [{"code": code, "count": count} for code, count in codes.most_common(12)],
        "guardian_blocked_lanes": codes.get(GUARDIAN_BLOCKER, 0),
        "repair_frontier": repair_frontier[:12],
    }


def _intent_metadata(item: dict[str, Any]) -> dict[str, Any]:
    intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    return metadata


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
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    return {
        "status": payload.get("status"),
        "blockers": blockers[:8],
        "capture_economics": capture,
    }


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
        blockers.append(
            {
                "code": PROFIT_CAPTURE_MISS,
                "severity": "P0",
                "message": profit_capture.get("message")
                or f"{profit_capture['missed_loss_closes']} losing close(s) missed TP-progress capture",
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


def _render_report(payload: dict[str, Any]) -> str:
    guardian = payload["guardian"]
    entry = payload["entry_readiness"]
    profit = payload["profit_capture"]
    broker = payload["broker"]
    target = payload["target"]
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
        f"| Guardian active | `{guardian['active']}` source=`{guardian['active_source']}` |",
        f"| Guardian heartbeat fresh | `{guardian['heartbeat_fresh']}` age=`{guardian['heartbeat_age_seconds']}`s |",
        f"| LIVE_READY lanes | `{entry['live_ready_lanes']}` / `{entry['lanes']}` |",
        f"| Repair frontier lanes | `{len(entry['repair_frontier'])}` |",
        f"| Profit-capture misses | `{profit['missed_loss_closes']}` gap=`{profit['estimated_gap_jpy']}` JPY |",
        f"| Open trader positions | `{broker['trader_positions']}` upl=`{broker['trader_unrealized_pl_jpy']}` JPY |",
        f"| Target remaining | `{target['remaining_target_jpy']}` JPY |",
        "",
        "## Blockers",
        "",
    ]
    if payload["blockers"]:
        for blocker in payload["blockers"]:
            lines.append(f"- `{blocker['severity']}` `{blocker['code']}`: {blocker['message']}")
    else:
        lines.append("- none")
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
