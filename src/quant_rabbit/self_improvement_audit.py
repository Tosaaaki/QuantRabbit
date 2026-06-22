from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    repair_replay_contract_from_payload,
)
from quant_rabbit.forecast_precision import (
    projection_precision_edge_summary,
    projection_precision_gap_summary,
)
from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ENTRY_THESIS_LEDGER,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_MEMORY_HEALTH,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_SELF_IMPROVEMENT_AUDIT_REPORT,
    DEFAULT_SELF_IMPROVEMENT_HISTORY_DB,
    DEFAULT_TRADER_DECISION,
    DEFAULT_VERIFICATION_LEDGER,
)
from quant_rabbit.risk import DEFAULT_SPECS


STATUS_OK = "SELF_IMPROVEMENT_OK"
STATUS_ACTION_REQUIRED = "SELF_IMPROVEMENT_ACTION_REQUIRED"
STATUS_BLOCKED = "SELF_IMPROVEMENT_BLOCKED"


def _env_nonnegative_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# Match the live-entry telemetry gate's cycle-preflight tolerance
# (intent_generator.PROJECTION_PENDING_EXPIRY_GRACE_SECONDS): one full
# 20-minute scheduler cadence. The consolidated cycle's refresh-to-gateway
# latency was measured at ~10 minutes live (2026-06-11), so rows that cross
# their resolution boundary inside one cadence are next-preflight work, not a
# stale-state defect.
PROJECTION_PENDING_EXPIRY_GRACE_SECONDS = _env_nonnegative_float(
    "QR_PROJECTION_PENDING_EXPIRY_GRACE_SECONDS",
    1200.0,
)

# Scheduled runners and wrappers can retry the same audit after a blocked exit
# code; identical evidence inside this operational retry window is one audit
# observation, not a new trend sample.
AUDIT_HISTORY_DUPLICATE_WINDOW_SECONDS = _env_nonnegative_float(
    "QR_SELF_IMPROVEMENT_HISTORY_DUPLICATE_WINDOW_SECONDS",
    120.0,
)

# Three consecutive non-duplicate audits is an operational trend, not a single
# market outcome. Persistent negative expectancy plus loss/win asymmetry must
# block fresh-risk confidence until repaired or explicitly justified.
PERSISTENT_PROFITABILITY_STREAK_MIN = int(
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_PROFITABILITY_STREAK_MIN", 3.0)
)

# Three accepted entry orders is the smallest sample that proves repeated
# execution behavior rather than one missed quote. This audit does not block
# live trades; it fixes the repair focus when the system sends orders but gets
# no fills.
PENDING_ENTRY_LIFECYCLE_MIN_ACCEPTED = int(
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_PENDING_LIFECYCLE_MIN_ACCEPTED", 3.0)
)
# Majority cancellation before fill means the active bottleneck is execution
# lifecycle, not just lane discovery. 0.5 is a categorical majority boundary,
# not a market price/risk parameter.
PENDING_ENTRY_CANCEL_RATE_WARN_ABOVE = min(
    1.0,
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_PENDING_CANCEL_RATE_WARN_ABOVE", 0.5),
)
# Replacement within this operational audit window means a canceled pending was
# deliberately re-armed nearby in time, not simply abandoned. This is diagnostic
# bookkeeping for cancel churn, not a market/risk threshold.
PENDING_ENTRY_CANCEL_REPLACEMENT_WINDOW_MIN = _env_nonnegative_float(
    "QR_SELF_IMPROVEMENT_PENDING_CANCEL_REPLACEMENT_WINDOW_MIN",
    30.0,
)
# `execution_timing_audit` scores canceled-order regret over the same weekly
# evidence window used by this self-improvement audit. Older regret rows are
# stale process evidence and must not preserve current pending-thesis decisions.
PENDING_CANCEL_REGRET_MAX_AGE_HOURS = _env_nonnegative_float(
    "QR_SELF_IMPROVEMENT_PENDING_CANCEL_REGRET_MAX_AGE_HOURS",
    168.0,
)

# A loss-containment close is positive close-discipline evidence only when it
# avoids at least twice the realized containment cost. The 2x boundary mirrors
# the audit's loss/win asymmetry line: below this, a loss close is still too
# weak to prove the close path is repaired without a winning close in-window.
LOSS_CONTAINMENT_RECOVERY_MIN_AVOIDED_MULT = _env_nonnegative_float(
    "QR_LOSS_CONTAINMENT_RECOVERY_MIN_AVOIDED_MULT",
    2.0,
)

PROFITABILITY_DISCIPLINE_CODES = (
    "NEGATIVE_RECENT_EXPECTANCY",
)

# Same repair finding across three non-duplicate audits is an operator-process
# loop. It must redirect the next cycle to a narrower repair, but it is P1
# because it is not by itself a broker-truth or risk reason to stop a valid
# current lane.
REPEATED_REPAIR_LOOP_STREAK_MIN = int(
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_REPEATED_REPAIR_LOOP_STREAK_MIN", 3.0)
)
REPEATED_REPAIR_LOOP_CODE = "REPEATED_SELF_IMPROVEMENT_LOOP"
# Report the top repeated repair loops together so a persistent P0 does not hide
# a simultaneous code-owned P1 repair surface from root-cause guards.
REPEATED_REPAIR_LOOP_MAX_FINDINGS = 3

# Root-cause focus is an audit summarizer, not a new live gate. It collapses
# repeated symptoms into the smallest repair surface so the next cycle does not
# treat process loops, coverage gaps, and execution churn as unrelated work.
ROOT_CAUSE_CODE = "ROOT_CAUSE_FOCUS"
ROOT_CAUSE_PRIORITY_WEIGHTS = {"P0": 100.0, "P1": 30.0, "P2": 10.0}
ROOT_CAUSE_CODE_BOOSTS = {
    "DIRECTIONAL_FORECAST_ENTRY_GRADE_SAMPLE_SHORTFALL": 65.0,
    "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT": 90.0,
    "DIRECTIONAL_FORECAST_HIT_RATE_WEAK": 70.0,
    "DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK": 55.0,
    "PROJECTION_ECONOMIC_PRECISION_WEAK": 55.0,
    "FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED": 50.0,
    "RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED": 45.0,
    "TARGET_OPEN_LIVE_READY_COVERAGE_SHORTFALL": 60.0,
    "TARGET_OPEN_NO_LIVE_READY_LANES": 60.0,
    "LOSS_CLOSE_PROFIT_CAPTURE_MISSED": 85.0,
    "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE": 90.0,
    "PENDING_ENTRY_FILL_RATE_WEAK": 45.0,
    "PENDING_ENTRY_CANCEL_RATE_HIGH": 40.0,
    "NEGATIVE_RECENT_EXPECTANCY": 35.0,
}

NON_GATEWAY_CLOSE_DRAG_PROVENANCES = (
    "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE",
    "NO_LOCAL_CLOSE_PROVENANCE",
)
_EXTERNAL_BROKER_TRADE_CLOSE_SOURCES = frozenset(
    {
        "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE",
        "NON_TRADER_CLIENT_EXTENSION",
    }
)

_LEARNING_AUDIT_LANE_QUARANTINE_MESSAGES = (
    "risk-increasing learning influence is active while recent effect window is negative",
)


def _external_live_runtime_lock(snapshot_path: Path) -> dict[str, Any] | None:
    """Return active wrapper lock evidence for audits outside that wrapper."""

    if os.environ.get("QR_AUTOTRADE_LOCK_HELD") == "1":
        return None

    configured = os.environ.get("QR_AUTOTRADE_LOCK_DIR")
    if configured:
        lock_dir = Path(configured)
    else:
        runtime_root = snapshot_path.parent.parent if snapshot_path.parent.name == "data" else Path.cwd()
        lock_dir = runtime_root / ".quant_rabbit_live.lock"
    pid_path = lock_dir / "pid"
    if not pid_path.exists():
        return None
    try:
        raw_pid = pid_path.read_text(encoding="utf-8").strip()
        pid = int(raw_pid)
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass

    evidence: dict[str, Any] = {"lock_dir": str(lock_dir), "pid": pid}
    command_path = lock_dir / "command"
    started_path = lock_dir / "started_at_utc"
    try:
        command = command_path.read_text(encoding="utf-8").strip()
    except OSError:
        command = ""
    if command:
        evidence["command"] = command
    try:
        started_at = started_path.read_text(encoding="utf-8").strip()
    except OSError:
        started_at = ""
    if started_at:
        evidence["started_at_utc"] = started_at
        try:
            started_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if started_dt.tzinfo is None:
                started_dt = started_dt.replace(tzinfo=timezone.utc)
            evidence["lock_age_seconds"] = max(
                0.0,
                (datetime.now(timezone.utc) - started_dt.astimezone(timezone.utc)).total_seconds(),
            )
        except ValueError:
            pass
    return evidence

# Forecast-level calibration is the feedback loop for "why did the final
# direction call miss?". Ten samples matches the projection-ledger calibration
# sample floor, and the hit-rate floor is an audit warning, not a trade gate.
FORECAST_CALIBRATION_MIN_SAMPLES = int(
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_FORECAST_CALIBRATION_MIN_SAMPLES", 10.0)
)
FORECAST_HIT_RATE_WARN_BELOW = _env_nonnegative_float(
    "QR_SELF_IMPROVEMENT_FORECAST_HIT_RATE_WARN_BELOW", 0.45
)
# Audit-only majority threshold: if most calibrated directional misses hit
# invalidation before target, the defect is not generic uncertainty. It is an
# adverse-path / reversal-prior problem that needs a different repair queue.
FORECAST_INVALIDATION_FIRST_WARN_ABOVE = min(
    1.0,
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_FORECAST_INVALIDATION_FIRST_WARN_ABOVE", 0.6),
)
# Audit-only majority threshold for full-target no-touch outcomes. These are
# not adverse-path misses, but they show the forecast target/horizon is too
# ambitious for short-horizon opportunity capture.
FORECAST_TARGET_TIMEOUT_WARN_ABOVE = min(
    1.0,
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_FORECAST_TARGET_TIMEOUT_WARN_ABOVE", 0.6),
)
# Audit feedback coverage floor only. Directional calls with mostly TIMEOUT
# outcomes are not learnable enough to justify stronger forecast confidence.
FORECAST_CALIBRATION_MIN_COVERAGE = min(
    1.0,
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_FORECAST_CALIBRATION_MIN_COVERAGE", 0.5),
)
# Use the same confidence floor as the live-entry forecaster. Rows below this
# are watch-only telemetry: important for coverage diagnostics, but not proof
# that entry-grade directional forecasts are weak.
FORECAST_ENTRY_GRADE_CONFIDENCE_MIN = _env_nonnegative_float(
    "QR_FORECAST_ENTRY_CONFIDENCE_MIN",
    0.55,
)
# Mirror the live precision gate so the audit explains the same economic
# precision failure that intent generation and RiskEngine enforce.
PROJECTION_ECONOMIC_PRECISION_MIN_WILSON_LOWER = _env_nonnegative_float(
    "QR_FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER",
    0.90,
)
PROJECTION_ECONOMIC_PRECISION_MIN_SAMPLES = max(
    1,
    int(_env_nonnegative_float("QR_FORECAST_LIVE_PRECISION_MIN_SAMPLES", 30.0)),
)
# Report-limit only; this keeps the live digest readable and is not a market
# threshold.
PROJECTION_ECONOMIC_PRECISION_GAP_LIMIT = int(
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_PROJECTION_ECONOMIC_GAP_LIMIT", 12.0)
)
# Report-limit only. Positive edge rows are evidence for the next mining/ranking
# pass; they do not grant live permission or weaken RiskEngine/Gateway checks.
PROJECTION_ECONOMIC_PRECISION_EDGE_LIMIT = int(
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_PROJECTION_ECONOMIC_EDGE_LIMIT", 12.0)
)


@dataclass(frozen=True)
class SelfImprovementAuditSummary:
    db_path: Path
    history_db_path: Path
    output_path: Path
    report_path: Path
    status: str
    findings: int
    p0_findings: int
    p1_findings: int
    p2_findings: int
    closed_trades: int
    net_jpy: float
    profit_factor: float | None
    expectancy_jpy: float | None
    live_ready_lanes: int
    open_trader_positions: int


class SelfImprovementAuditor:
    """Roll memory/profitability/verifiability gaps into durable action items."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        history_db_path: Path | None = DEFAULT_SELF_IMPROVEMENT_HISTORY_DB,
        output_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
        report_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT_REPORT,
    ) -> None:
        self.db_path = db_path
        self.history_db_path = history_db_path or DEFAULT_SELF_IMPROVEMENT_HISTORY_DB
        self.output_path = output_path
        self.report_path = report_path

    def run(
        self,
        *,
        snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        market_context_matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX,
        memory_health_path: Path = DEFAULT_MEMORY_HEALTH,
        learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
        ai_test_bot_backtest_path: Path = DEFAULT_AI_TEST_BOT_BACKTEST,
        verification_ledger_path: Path = DEFAULT_VERIFICATION_LEDGER,
        attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        entry_thesis_ledger_path: Path = DEFAULT_ENTRY_THESIS_LEDGER,
        gpt_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
        trader_decision_path: Path = DEFAULT_TRADER_DECISION,
        position_management_path: Path = DEFAULT_POSITION_MANAGEMENT,
        position_guardian_management_path: Path = DEFAULT_POSITION_GUARDIAN_MANAGEMENT,
        thesis_evolution_path: Path = Path("data/thesis_evolution_report.json"),
        position_thesis_path: Path = Path("data/position_thesis_report.json"),
        forecast_persistence_path: Path = Path("data/forecast_persistence_report.json"),
        coverage_optimization_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        window_hours: float = 168.0,
        now: datetime | None = None,
    ) -> SelfImprovementAuditSummary:
        clock = _to_utc(now or datetime.now(timezone.utc))
        run_id = clock.isoformat()
        self._init_history_db()

        snapshot_loaded = _read_json(snapshot_path)
        target_loaded = _read_json(target_state_path)
        intents_loaded = _read_json(order_intents_path)
        market_context_matrix_loaded = _read_json(market_context_matrix_path)
        memory_loaded = _read_json(memory_health_path)
        learning_loaded = _read_json(learning_audit_path)
        ai_backtest_loaded = _read_json(ai_test_bot_backtest_path)
        verification_loaded = _read_json(verification_ledger_path)
        attack_advice_loaded = _read_json(attack_advice_path)
        execution_timing_loaded = _read_json(execution_timing_audit_path)
        gpt_loaded = _read_json(gpt_decision_path)
        trader_loaded = _read_json(trader_decision_path)
        position_management_loaded = _read_json(position_management_path)
        position_guardian_management_loaded = (
            _read_json(position_guardian_management_path)
            if position_guardian_management_path.exists()
            else None
        )
        thesis_evolution_loaded = _read_json(thesis_evolution_path)
        position_thesis_loaded = _read_json(position_thesis_path)
        forecast_persistence_loaded = _read_json(forecast_persistence_path)
        coverage_loaded = _read_json(coverage_optimization_path)

        snapshot = snapshot_loaded.payload or {}
        target_state = target_loaded.payload or {}
        intents = intents_loaded.payload or {}
        account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
        snapshot_ts = _parse_utc(snapshot.get("fetched_at_utc") or account.get("fetched_at_utc"))
        target_open = _target_open(target_state)
        live_ready = _live_ready_results(intents)
        active_positions = _active_trader_positions(snapshot)
        pending_entry_orders = _trader_pending_entry_orders(snapshot)
        active_trade_ids = {str(item.get("trade_id") or "") for item in active_positions if item.get("trade_id")}
        active_position_sides = {
            str(item.get("trade_id") or ""): str(item.get("side") or "").upper()
            for item in active_positions
            if item.get("trade_id")
        }
        effect = _effect_metrics(self.db_path, window_hours=window_hours, now=clock)
        effect_24h = _effect_metrics(self.db_path, window_hours=24.0, now=clock)
        close_gate_loss_evidence = _close_gate_loss_evidence(ai_backtest_loaded, ai_test_bot_backtest_path)
        profitability_streak_before = self._history_code_streak(PROFITABILITY_DISCIPLINE_CODES)
        latest_gpt_stale_streak_before = self._history_code_streak(("LATEST_GPT_DECISION_STALE",))
        external_live_lock = _external_live_runtime_lock(snapshot_path)
        pending_entry_lifecycle = _pending_entry_lifecycle_metrics(
            self.db_path,
            window_hours=window_hours,
            now=clock,
        )
        _merge_pending_cancel_timing_regret(
            pending_entry_lifecycle,
            execution_timing_loaded.payload,
            now=clock,
        )
        pending_entry_reconcile = _pending_entry_reconcile_metrics(
            snapshot=snapshot,
            target_state=target_state,
            intents=intents,
            attack_advice=attack_advice_loaded.payload or {},
            target_open=target_open,
        )
        guardian_status = _position_guardian_runtime_status()

        findings: list[dict[str, Any]] = []
        if external_live_lock is not None:
            findings.append(
                _finding(
                    run_id=run_id,
                    priority="P0",
                    layer="runtime",
                    code="LIVE_RUNTIME_UPDATE_IN_PROGRESS",
                    message=(
                        "live runtime wrapper lock is active; broker/order/memory artifacts may be mid-refresh"
                    ),
                    next_action=(
                        "Wait for the live wrapper to release its lock, then rerun self-improvement-audit "
                        "before judging stale memory or order-intent blockers."
                    ),
                    evidence=external_live_lock,
                )
            )
        artifact_loads = {
            "broker_snapshot": (snapshot_loaded, snapshot_path, "runtime"),
            "daily_target_state": (target_loaded, target_state_path, "runtime"),
            "order_intents": (intents_loaded, order_intents_path, "runtime"),
        }
        for label, (loaded, path, layer) in artifact_loads.items():
            if loaded.error is not None:
                findings.append(
                    _finding(
                        run_id=run_id,
                        priority="P0",
                        layer=layer,
                        code=f"{label.upper()}_UNREADABLE",
                        message=f"{label} is unreadable: {path}: {loaded.error}",
                        next_action=f"Regenerate or repair {label} before the next trader route.",
                    )
                )

        if external_live_lock is None:
            findings.extend(
                _memory_findings(
                    run_id=run_id,
                    loaded=memory_loaded,
                    path=memory_health_path,
                    target_open=target_open,
                    snapshot=snapshot,
                    intents=intents,
                )
            )
        findings.extend(_learning_findings(run_id=run_id, loaded=learning_loaded, path=learning_audit_path))
        findings.extend(
            _verification_findings(
                run_id=run_id,
                loaded=verification_loaded,
                path=verification_ledger_path,
            )
        )
        findings.extend(
            _ledger_sync_findings(
                run_id=run_id,
                db_path=self.db_path,
                snapshot=snapshot,
            )
        )
        findings.extend(
            _pending_entry_lifecycle_findings(
                run_id=run_id,
                metrics=pending_entry_lifecycle,
                target_open=target_open,
            )
        )
        profit_capture_miss_findings = _profit_capture_miss_findings(
            run_id=run_id,
            timing_payload=execution_timing_loaded.payload,
            target_open=target_open,
        )
        findings.extend(profit_capture_miss_findings)
        findings.extend(
            _position_guardian_profit_capture_findings(
                run_id=run_id,
                guardian_status=guardian_status,
                target_open=target_open,
                live_ready_lanes=len(live_ready),
                open_trader_positions=len(active_positions),
                open_trader_pending_entries=len(pending_entry_orders),
                profit_capture_miss_active=bool(profit_capture_miss_findings),
            )
        )
        findings.extend(
            _pending_entry_reconcile_findings(
                run_id=run_id,
                metrics=pending_entry_reconcile,
                target_open=target_open,
            )
        )
        findings.extend(
            _market_close_attribution_findings(
                run_id=run_id,
                db_path=self.db_path,
                window_hours=window_hours,
                now=clock,
            )
        )
        findings.extend(
            _forecast_history_findings(
                run_id=run_id,
                path=forecast_history_path,
                target_open=target_open,
            )
        )
        findings.extend(
            _projection_findings(
                run_id=run_id,
                path=projection_ledger_path,
                now=clock,
                target_open=target_open,
                active_positions=active_positions,
            )
        )
        findings.extend(
            _entry_thesis_findings(
                run_id=run_id,
                path=entry_thesis_ledger_path,
                active_trade_ids=active_trade_ids,
            )
        )
        findings.extend(
            _profitability_findings(
                run_id=run_id,
                effect=effect,
                effect_24h=effect_24h,
                snapshot=snapshot,
                min_sample=3,
                close_gate_loss_evidence=close_gate_loss_evidence,
                previous_discipline_streak=profitability_streak_before,
            )
        )
        if external_live_lock is None:
            findings.extend(
                _intent_findings(
                    run_id=run_id,
                    intents=intents,
                    target_state=target_state,
                    target_open=target_open,
                    live_ready=live_ready,
                    active_positions=active_positions,
                    pending_entry_orders=pending_entry_orders,
                    coverage_optimization=coverage_loaded.payload or {},
                )
            )
            findings.extend(
                _order_intent_context_evidence_findings(
                    run_id=run_id,
                    intents=intents,
                    matrix_loaded=market_context_matrix_loaded,
                    matrix_path=market_context_matrix_path,
                    target_open=target_open,
                )
            )
        findings.extend(
            _coverage_findings(
                run_id=run_id,
                loaded=coverage_loaded,
                path=coverage_optimization_path,
                target_open=target_open,
            )
        )
        findings.extend(
            _mechanism_ablation_findings(
                run_id=run_id,
                loaded=ai_backtest_loaded,
                path=ai_test_bot_backtest_path,
            )
        )
        position_sidecars = {
            "position_management": (position_management_loaded, position_management_path),
            "thesis_evolution": (thesis_evolution_loaded, thesis_evolution_path),
            "position_thesis": (position_thesis_loaded, position_thesis_path),
            "forecast_persistence": (forecast_persistence_loaded, forecast_persistence_path),
        }
        if position_guardian_management_loaded is not None:
            position_sidecars["position_guardian_management"] = (
                position_guardian_management_loaded,
                position_guardian_management_path,
            )

        findings.extend(
            _sidecar_findings(
                run_id=run_id,
                snapshot_ts=snapshot_ts,
                active_trade_ids=active_trade_ids,
                gpt_decision=gpt_loaded.payload or {},
                sidecars=position_sidecars,
                defer_stale_judgment=external_live_lock is not None,
            )
        )
        findings.extend(
            _decision_artifact_findings(
                run_id=run_id,
                db_path=self.db_path,
                gpt_loaded=gpt_loaded,
                trader_loaded=trader_loaded,
                gpt_path=gpt_decision_path,
                trader_path=trader_decision_path,
                target_open=target_open,
                active_trade_ids=active_trade_ids,
                active_position_sides=active_position_sides,
                live_ready_lanes=len(live_ready),
                pending_entry_orders=len(pending_entry_orders),
                pending_entry_order_details=pending_entry_orders,
                snapshot_ts=snapshot_ts,
                latest_gpt_stale_streak_before=latest_gpt_stale_streak_before,
                sidecars=position_sidecars,
            )
        )

        findings = _dedupe_findings(findings)
        findings.extend(self._repeated_repair_loop_findings(run_id=run_id, findings=findings))
        findings = _dedupe_findings(findings)
        p0 = sum(1 for item in findings if item["priority"] == "P0")
        p1 = sum(1 for item in findings if item["priority"] == "P1")
        p2 = sum(1 for item in findings if item["priority"] == "P2")
        status = STATUS_BLOCKED if p0 else (STATUS_ACTION_REQUIRED if findings else STATUS_OK)
        runtime_summary = {
            "target_open": target_open,
            "snapshot_fetched_at_utc": snapshot_ts.isoformat() if snapshot_ts else None,
            "open_trader_positions": len(active_positions),
            "open_trader_pending_entries": len(pending_entry_orders),
            "live_ready_lanes": len(live_ready),
            "position_guardian": guardian_status,
            "gpt_status": (gpt_loaded.payload or {}).get("status"),
            "gpt_action": ((gpt_loaded.payload or {}).get("decision") or {}).get("action")
            if isinstance((gpt_loaded.payload or {}).get("decision"), dict)
            else None,
        }
        effect_summary = {
            "window": effect,
            "last_24h": effect_24h,
        }
        execution_quality = {
            "pending_entry_lifecycle": pending_entry_lifecycle,
            "pending_entry_reconcile": pending_entry_reconcile,
        }
        root_cause_focus = _root_cause_focus(
            findings=findings,
            runtime=runtime_summary,
            effect_metrics=effect_summary,
            execution_quality=execution_quality,
        )
        payload = {
            "generated_at_utc": run_id,
            "status": status,
            "window_hours": window_hours,
            "findings_count": len(findings),
            "p0_findings": p0,
            "p1_findings": p1,
            "p2_findings": p2,
            "artifact_paths": {
                "execution_ledger_db": str(self.db_path),
                "audit_history_db": str(self.history_db_path),
                "broker_snapshot": str(snapshot_path),
                "daily_target_state": str(target_state_path),
                "order_intents": str(order_intents_path),
                "ai_attack_advice": str(attack_advice_path),
                "execution_timing_audit": str(execution_timing_audit_path),
                "market_context_matrix": str(market_context_matrix_path),
                "memory_health": str(memory_health_path),
                "learning_audit": str(learning_audit_path),
                "ai_test_bot_backtest": str(ai_test_bot_backtest_path),
                "verification_ledger": str(verification_ledger_path),
                "forecast_history": str(forecast_history_path),
                "projection_ledger": str(projection_ledger_path),
                "entry_thesis_ledger": str(entry_thesis_ledger_path),
                "gpt_decision": str(gpt_decision_path),
                "trader_decision": str(trader_decision_path),
                "position_management": str(position_management_path),
                "position_guardian_management": str(position_guardian_management_path),
                "thesis_evolution": str(thesis_evolution_path),
                "position_thesis": str(position_thesis_path),
                "forecast_persistence": str(forecast_persistence_path),
                "coverage_optimization": str(coverage_optimization_path),
            },
            "runtime": runtime_summary,
            "effect_metrics": effect_summary,
            "execution_quality": execution_quality,
            "root_cause_focus": root_cause_focus,
            "findings": findings,
            "next_actions": _next_actions(findings, root_cause_focus=root_cause_focus),
            "contract": {
                "read_only_live_permission": False,
                "audit_can_write_history_db": True,
                "does_not_grant_trade_permission": True,
                "live_safety_boundaries_are_not_profitability_assumptions": True,
                "offline_ablation_required_before_relaxing_live_gates": True,
                "hard_gates_remain": ["RiskEngine", "IntentGenerator telemetry validation", "LiveOrderGateway", "Gate A/Gate B close discipline"],
            },
        }
        self._write_output(payload)
        self._write_report(payload)
        self._insert_history(payload)
        return SelfImprovementAuditSummary(
            db_path=self.db_path,
            history_db_path=self.history_db_path,
            output_path=self.output_path,
            report_path=self.report_path,
            status=status,
            findings=len(findings),
            p0_findings=p0,
            p1_findings=p1,
            p2_findings=p2,
            closed_trades=int(effect.get("closed_trades") or 0),
            net_jpy=float(effect.get("net_jpy") or 0.0),
            profit_factor=_maybe_float(effect.get("profit_factor")),
            expectancy_jpy=_maybe_float(effect.get("expectancy_jpy")),
            live_ready_lanes=len(live_ready),
            open_trader_positions=len(active_positions),
        )

    def _init_history_db(self) -> None:
        self.history_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.history_db_path) as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS self_improvement_audit_runs (
                    run_uid TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    report_path TEXT NOT NULL,
                    window_hours REAL NOT NULL,
                    findings INTEGER NOT NULL,
                    p0_findings INTEGER NOT NULL,
                    p1_findings INTEGER NOT NULL,
                    p2_findings INTEGER NOT NULL,
                    closed_trades INTEGER NOT NULL,
                    net_jpy REAL NOT NULL,
                    profit_factor REAL,
                    expectancy_jpy REAL,
                    live_ready_lanes INTEGER NOT NULL,
                    open_trader_positions INTEGER NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_self_improvement_audit_runs_ts
                    ON self_improvement_audit_runs(ts_utc);
                CREATE INDEX IF NOT EXISTS idx_self_improvement_audit_runs_status
                    ON self_improvement_audit_runs(status);

                CREATE TABLE IF NOT EXISTS self_improvement_findings (
                    finding_uid TEXT PRIMARY KEY,
                    run_uid TEXT NOT NULL,
                    ts_utc TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    code TEXT NOT NULL,
                    message TEXT NOT NULL,
                    next_action TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_self_improvement_findings_run
                    ON self_improvement_findings(run_uid);
                CREATE INDEX IF NOT EXISTS idx_self_improvement_findings_priority
                    ON self_improvement_findings(priority, code);
                """
            )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        runtime = payload["runtime"]
        window = payload["effect_metrics"]["window"]
        last_24h = payload["effect_metrics"]["last_24h"]
        pending_lifecycle = (
            (payload.get("execution_quality") or {}).get("pending_entry_lifecycle")
            if isinstance(payload.get("execution_quality"), dict)
            else {}
        )
        pending_reconcile = (
            (payload.get("execution_quality") or {}).get("pending_entry_reconcile")
            if isinstance(payload.get("execution_quality"), dict)
            else {}
        )
        guardian = runtime.get("position_guardian") if isinstance(runtime.get("position_guardian"), dict) else {}
        root_focus = payload.get("root_cause_focus") if isinstance(payload.get("root_cause_focus"), dict) else {}
        root_primary = root_focus.get("primary") if isinstance(root_focus.get("primary"), dict) else {}
        lines = [
            "# Self Improvement Audit Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Audit history DB: `{payload['artifact_paths']['audit_history_db']}`",
            "",
            "## Runtime",
            "",
            f"- Target open: `{runtime['target_open']}`",
            f"- Open trader positions: `{runtime['open_trader_positions']}`",
            f"- Open trader pending entries: `{runtime.get('open_trader_pending_entries', 0)}`",
            f"- LIVE_READY lanes: `{runtime['live_ready_lanes']}`",
            f"- Position guardian: required=`{guardian.get('required')}` active=`{guardian.get('active')}` source=`{guardian.get('active_source')}` launchd_loaded=`{guardian.get('launchd_loaded')}`",
            f"- GPT status/action: `{runtime['gpt_status']}` / `{runtime['gpt_action']}`",
            "",
            "## Profitability",
            "",
            f"- Window `{payload['window_hours']}`h: trades `{window['closed_trades']}`, net `{window['net_jpy']:.2f}` JPY, PF `{_fmt_optional(window['profit_factor'])}`, expectancy `{_fmt_optional(window['expectancy_jpy'])}` JPY",
            f"- Last 24h: trades `{last_24h['closed_trades']}`, net `{last_24h['net_jpy']:.2f}` JPY, PF `{_fmt_optional(last_24h['profit_factor'])}`, expectancy `{_fmt_optional(last_24h['expectancy_jpy'])}` JPY",
        ]
        market_close_loss = window.get("market_order_trade_close_loss_provenance_metrics")
        if isinstance(market_close_loss, dict) and market_close_loss:
            lines.append(
                f"- Market-order loss close provenance: `{_fmt_close_provenance_metrics(market_close_loss)}`"
            )
        market_close_loss_24h = last_24h.get("market_order_trade_close_loss_provenance_metrics")
        if isinstance(market_close_loss_24h, dict) and market_close_loss_24h:
            lines.append(
                f"- Last-24h market-order loss close provenance: `{_fmt_close_provenance_metrics(market_close_loss_24h)}`"
            )
        if isinstance(pending_lifecycle, dict) and pending_lifecycle:
            lines.extend(
                [
                    "",
                    "## Execution Quality",
                    "",
                    (
                        "- Pending entry lifecycle: "
                        f"accepted `{pending_lifecycle.get('accepted_entry_orders', 0)}`, "
                        f"filled `{pending_lifecycle.get('filled_entry_orders', 0)}`, "
                        f"canceled_before_fill `{pending_lifecycle.get('canceled_before_fill_orders', 0)}`, "
                        f"open_unfilled `{pending_lifecycle.get('open_unfilled_entry_orders', 0)}`, "
                        f"fill_rate `{_fmt_optional(pending_lifecycle.get('fill_rate'))}`, "
                        f"cancel_before_fill_rate `{_fmt_optional(pending_lifecycle.get('cancel_before_fill_rate'))}`"
                    ),
                ]
            )
            timing_regret = (
                pending_lifecycle.get("timing_regret")
                if isinstance(pending_lifecycle.get("timing_regret"), dict)
                else {}
            )
            if timing_regret:
                lines.append(
                    "- Pending cancel timing regret: "
                    f"audited `{timing_regret.get('canceled_orders_audited', 0)}`, "
                    f"entry_touched `{timing_regret.get('canceled_entry_touched_after_cancel', 0)}`, "
                    f"tp_touched `{timing_regret.get('canceled_tp_touched_after_cancel', 0)}`, "
                    f"missed_mfe_jpy `{_fmt_optional(timing_regret.get('canceled_estimated_missed_mfe_jpy'))}`"
                )
                top_shapes = (
                    timing_regret.get("top_regretted_shapes")
                    if isinstance(timing_regret.get("top_regretted_shapes"), list)
                    else []
                )
                if top_shapes and isinstance(top_shapes[0], dict):
                    top_shape = top_shapes[0]
                    lines.append(
                        "- Top pending cancel regret shape: "
                        f"`{top_shape.get('evidence_ref')}`, "
                        f"priority `{top_shape.get('priority_class')}`, "
                        f"orders `{top_shape.get('orders', 0)}`, "
                        f"missed_mfe_jpy `{_fmt_optional(top_shape.get('estimated_missed_mfe_jpy'))}`"
                    )
        if isinstance(pending_reconcile, dict) and pending_reconcile:
            if "## Execution Quality" not in lines:
                lines.extend(["", "## Execution Quality", ""])
            cancel_ids = pending_reconcile.get("cancel_review_order_ids") or []
            lines.append(
                "- Pending entry reconcile: "
                f"reviewed `{pending_reconcile.get('reviewed_open_pending_orders', 0)}`, "
                f"cancel_review `{pending_reconcile.get('cancel_review_orders', 0)}`, "
                f"ids `{', '.join(str(item) for item in cancel_ids) if cancel_ids else 'none'}`"
            )
        if root_primary:
            lines.extend(
                [
                    "",
                    "## Root Cause Focus",
                    "",
                    (
                        f"- Primary: `{root_primary.get('family')}` "
                        f"score `{_fmt_optional(root_primary.get('score'))}` "
                        f"confidence `{root_primary.get('confidence')}`"
                    ),
                    f"- Why: {root_primary.get('why')}",
                    f"- Goal adjustment: {root_primary.get('goal_adjustment')}",
                    f"- Next: {root_primary.get('next_action')}",
                ]
            )
            support = root_primary.get("supporting_codes")
            if isinstance(support, list) and support:
                lines.append(
                    "- Supporting codes: "
                    + ", ".join(f"`{code}`" for code in support[:8] if str(code).strip())
                )
            symptoms = root_primary.get("downstream_symptoms")
            if isinstance(symptoms, list) and symptoms:
                lines.append(
                    "- Downstream symptoms: "
                    + ", ".join(f"`{item}`" for item in symptoms[:8] if str(item).strip())
                )
        lines.extend(["", "## Findings", ""])
        if payload["findings"]:
            for item in payload["findings"]:
                lines.append(
                    f"- `{item['priority']}` `{item['layer']}` `{item['code']}`: {item['message']} Next: {item['next_action']}"
                )
                lines.extend(_finding_report_details(item))
        else:
            lines.append("- None")
        lines.extend(["", "## Next Actions", ""])
        for item in payload["next_actions"]:
            lines.append(f"- `{item['priority']}` `{item['code']}`: {item['next_action']}")
        lines.extend(
            [
                "",
                "## Contract",
                "",
                "- This audit does not grant permission to trade.",
                "- P0 means the next trader route should repair or explicitly account for the hole before adding risk.",
                "- Live sends remain governed by broker truth, RiskEngine, IntentGenerator telemetry validation, LiveOrderGateway, and Gate A/Gate B close discipline.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _insert_history(self, payload: dict[str, Any]) -> None:
        run_uid = f"self_improvement_audit:{payload['generated_at_utc']}"
        findings = payload["findings"]
        p0 = sum(1 for item in findings if item["priority"] == "P0")
        p1 = sum(1 for item in findings if item["priority"] == "P1")
        p2 = sum(1 for item in findings if item["priority"] == "P2")
        effect = payload["effect_metrics"]["window"]
        runtime = payload["runtime"]
        with sqlite3.connect(self.history_db_path) as conn:
            conn.row_factory = sqlite3.Row
            if _history_has_recent_duplicate(
                conn,
                payload=payload,
                output_path=self.output_path,
                report_path=self.report_path,
            ):
                return
            conn.execute(
                """
                INSERT OR IGNORE INTO self_improvement_audit_runs(
                    run_uid, ts_utc, status, output_path, report_path, window_hours,
                    findings, p0_findings, p1_findings, p2_findings,
                    closed_trades, net_jpy, profit_factor, expectancy_jpy,
                    live_ready_lanes, open_trader_positions, inserted_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_uid,
                    payload["generated_at_utc"],
                    payload["status"],
                    str(self.output_path),
                    str(self.report_path),
                    float(payload["window_hours"]),
                    len(findings),
                    p0,
                    p1,
                    p2,
                    int(effect.get("closed_trades") or 0),
                    float(effect.get("net_jpy") or 0.0),
                    _maybe_float(effect.get("profit_factor")),
                    _maybe_float(effect.get("expectancy_jpy")),
                    int(runtime.get("live_ready_lanes") or 0),
                    int(runtime.get("open_trader_positions") or 0),
                    payload["generated_at_utc"],
                ),
            )
            for item in findings:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO self_improvement_findings(
                        finding_uid, run_uid, ts_utc, priority, layer, code,
                        message, next_action, evidence_json, inserted_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{run_uid}:{item['priority']}:{item['code']}",
                        run_uid,
                        payload["generated_at_utc"],
                        item["priority"],
                        item["layer"],
                        item["code"],
                        item["message"],
                        item["next_action"],
                        json.dumps(item.get("evidence") or {}, ensure_ascii=False, sort_keys=True),
                        payload["generated_at_utc"],
                    ),
                )

    def _history_code_streak(self, required_codes: tuple[str, ...]) -> int:
        """Return consecutive non-duplicate prior audit runs containing every requested code."""
        if not required_codes or not self.history_db_path.exists():
            return 0
        try:
            with sqlite3.connect(f"file:{self.history_db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT run_uid, ts_utc, status, output_path, report_path, window_hours,
                           findings, p0_findings, p1_findings, p2_findings,
                           closed_trades, net_jpy, profit_factor, expectancy_jpy,
                           live_ready_lanes, open_trader_positions
                    FROM self_improvement_audit_runs
                    ORDER BY ts_utc DESC
                    LIMIT 64
                    """
                ).fetchall()
                streak = 0
                previous_signature: tuple[Any, ...] | None = None
                previous_ts: datetime | None = None
                for row in rows:
                    finding_rows = conn.execute(
                        """
                        SELECT priority, layer, code, message, next_action, evidence_json
                        FROM self_improvement_findings
                        WHERE run_uid = ?
                        """,
                        (row["run_uid"],),
                    ).fetchall()
                    row_ts = _parse_utc(row["ts_utc"])
                    signature = (
                        _history_row_signature(row),
                        _history_db_findings_signature(finding_rows),
                    )
                    duplicate_delta = (
                        (previous_ts - row_ts).total_seconds()
                        if previous_ts is not None and row_ts is not None
                        else None
                    )
                    if (
                        previous_signature == signature
                        and duplicate_delta is not None
                        and 0.0 <= duplicate_delta <= AUDIT_HISTORY_DUPLICATE_WINDOW_SECONDS
                    ):
                        continue
                    previous_signature = signature
                    previous_ts = row_ts
                    codes = {
                        str(item["code"] or "")
                        for item in conn.execute(
                            "SELECT code FROM self_improvement_findings WHERE run_uid = ?",
                            (row["run_uid"],),
                        )
                    }
                    if all(code in codes for code in required_codes):
                        streak += 1
                        continue
                    break
                return streak
        except sqlite3.Error:
            return 0

    def _repeated_repair_loop_findings(
        self,
        *,
        run_id: str,
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        repeated: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for item in findings:
            priority = str(item.get("priority") or "").upper()
            if priority not in {"P0", "P1"}:
                continue
            code = str(item.get("code") or "").strip()
            if not code or code == REPEATED_REPAIR_LOOP_CODE or code in seen_codes:
                continue
            seen_codes.add(code)
            prior_streak = self._history_code_streak((code,))
            current_streak = prior_streak + 1
            if current_streak < REPEATED_REPAIR_LOOP_STREAK_MIN:
                continue
            repeated.append(
                {
                    "code": code,
                    "priority": priority,
                    "layer": str(item.get("layer") or ""),
                    "current_streak": current_streak,
                    "previous_streak": prior_streak,
                    "message": item.get("message"),
                    "next_action": item.get("next_action"),
                }
            )
        if not repeated:
            return []
        priority_rank = {"P0": 0, "P1": 1}
        repeated.sort(
            key=lambda item: (
                priority_rank.get(str(item.get("priority") or ""), 9),
                -int(item.get("current_streak") or 0),
                str(item.get("code") or ""),
            )
        )
        repeated = repeated[:REPEATED_REPAIR_LOOP_MAX_FINDINGS]
        primary = repeated[0]
        repeated_code = str(primary.get("code") or "")
        current_streak = int(primary.get("current_streak") or 0)
        previous_streak = int(primary.get("previous_streak") or 0)
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="process",
                code=REPEATED_REPAIR_LOOP_CODE,
                message=(
                    f"same self-improvement finding `{repeated_code}` has persisted for "
                    f"{current_streak} non-duplicate audit run(s)"
                ),
                next_action=(
                    "Stop repeating broad refresh/analysis for this finding. Execute the current finding's "
                    "named next_action as a narrow repair, then verify with its target metric before cycling "
                    "back to the same diagnosis."
                ),
                evidence={
                    "repeated_code": repeated_code,
                    "repeated_priority": primary.get("priority"),
                    "current_streak": current_streak,
                    "previous_streak": previous_streak,
                    "current_message": primary.get("message"),
                    "current_next_action": primary.get("next_action"),
                    "repeated_findings": repeated,
                },
            )
        ]


def _history_has_recent_duplicate(
    conn: sqlite3.Connection,
    *,
    payload: dict[str, Any],
    output_path: Path,
    report_path: Path,
) -> bool:
    generated_at = _parse_utc(payload.get("generated_at_utc"))
    if generated_at is None:
        return False
    window_start = generated_at - timedelta(seconds=AUDIT_HISTORY_DUPLICATE_WINDOW_SECONDS)
    if _history_has_recent_live_lock_duplicate(conn, payload=payload, window_start=window_start, generated_at=generated_at):
        return True
    expected_run = _history_run_signature(payload, output_path=output_path, report_path=report_path)
    expected_findings = _history_findings_signature(payload.get("findings") or [])
    rows = conn.execute(
        """
        SELECT run_uid, ts_utc, status, output_path, report_path, window_hours,
               findings, p0_findings, p1_findings, p2_findings,
               closed_trades, net_jpy, profit_factor, expectancy_jpy,
               live_ready_lanes, open_trader_positions
        FROM self_improvement_audit_runs
        WHERE ts_utc >= ? AND ts_utc <= ?
        ORDER BY ts_utc DESC
        LIMIT 12
        """,
        (window_start.isoformat(), generated_at.isoformat()),
    ).fetchall()
    for row in rows:
        if _history_row_signature(row) != expected_run:
            continue
        finding_rows = conn.execute(
            """
            SELECT priority, layer, code, message, next_action, evidence_json
            FROM self_improvement_findings
            WHERE run_uid = ?
            """,
            (row["run_uid"],),
        ).fetchall()
        if _history_db_findings_signature(finding_rows) == expected_findings:
            return True
    return False


def _history_has_recent_live_lock_duplicate(
    conn: sqlite3.Connection,
    *,
    payload: dict[str, Any],
    window_start: datetime,
    generated_at: datetime,
) -> bool:
    """Collapse short retries while the same live wrapper lock is still active.

    A live wrapper lock means broker/order/memory artifacts may be mutating under
    the audit. Those downstream volatile findings must not become extra trend
    samples for the same lock holder inside the operational retry window.
    """
    current_lock = _history_live_lock_evidence(payload.get("findings") or [])
    if current_lock is None:
        return False
    rows = conn.execute(
        """
        SELECT run_uid
        FROM self_improvement_audit_runs
        WHERE ts_utc >= ? AND ts_utc <= ?
        ORDER BY ts_utc DESC
        LIMIT 12
        """,
        (window_start.isoformat(), generated_at.isoformat()),
    ).fetchall()
    for row in rows:
        finding_rows = conn.execute(
            """
            SELECT code, evidence_json
            FROM self_improvement_findings
            WHERE run_uid = ? AND code = 'LIVE_RUNTIME_UPDATE_IN_PROGRESS'
            """,
            (row["run_uid"],),
        ).fetchall()
        for finding in finding_rows:
            evidence = _history_evidence_from_json(str(finding["evidence_json"] or "{}"))
            if _history_normalized_evidence("LIVE_RUNTIME_UPDATE_IN_PROGRESS", evidence) == current_lock:
                return True
    return False


def _history_live_lock_evidence(findings: list[dict[str, Any]]) -> Any | None:
    for item in findings:
        if str(item.get("code") or "") != "LIVE_RUNTIME_UPDATE_IN_PROGRESS":
            continue
        return _history_normalized_evidence("LIVE_RUNTIME_UPDATE_IN_PROGRESS", item.get("evidence") or {})
    return None


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _truthy_value(raw)


def _truthy_value(raw: Any) -> bool:
    return str(raw).strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _position_guardian_runtime_status() -> dict[str, Any]:
    """Return read-only launchd state used by profit-capture audits."""

    label = os.environ.get("QR_POSITION_GUARDIAN_LABEL", "com.quantrabbit.position-guardian")
    plist = Path(
        os.environ.get(
            "QR_POSITION_GUARDIAN_PLIST",
            str(Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"),
        )
    ).expanduser()
    raw_active = os.environ.get("QR_POSITION_GUARDIAN_ACTIVE")
    interval = _int_env("QR_POSITION_GUARDIAN_INTERVAL", default=30, minimum=15)
    heartbeat_max_age = _int_env(
        "QR_POSITION_GUARDIAN_HEARTBEAT_MAX_AGE_SECONDS",
        default=interval * 4,
        minimum=interval,
    )
    heartbeat_required = _truthy_env("QR_POSITION_GUARDIAN_REQUIRE_HEARTBEAT", default=True)
    heartbeat_paths = [
        _env_path("QR_POSITION_GUARDIAN_EXECUTION", Path("data/position_guardian_execution.json")),
        _env_path("QR_POSITION_GUARDIAN_HEARTBEAT", Path("data/position_guardian.json")),
    ]
    heartbeat = _freshest_guardian_heartbeat(heartbeat_paths, max_age_seconds=heartbeat_max_age)
    status: dict[str, Any] = {
        "required": _truthy_env("QR_REQUIRE_POSITION_GUARDIAN_ACTIVE", default=False),
        "active": False,
        "active_source": "launchd+heartbeat",
        "label": label,
        "plist_path": str(plist),
        "plist_exists": plist.exists(),
        "launchd_loaded": None,
        "heartbeat_required": heartbeat_required,
        "heartbeat_max_age_seconds": heartbeat_max_age,
        "heartbeat_fresh": heartbeat.get("fresh"),
        "heartbeat_age_seconds": heartbeat.get("age_seconds"),
        "heartbeat_generated_at_utc": heartbeat.get("generated_at_utc"),
        "heartbeat_path": heartbeat.get("path"),
        "heartbeat_source": heartbeat.get("source"),
        "heartbeat_status": heartbeat.get("status"),
        "heartbeat_candidates": [str(path) for path in heartbeat_paths],
    }
    if raw_active is not None:
        status["active"] = bool(_truthy_value(raw_active) and (heartbeat.get("fresh") or not heartbeat_required))
        status["active_source"] = "env+heartbeat"
        status["env_active"] = raw_active
        return status
    if not plist.exists():
        status["active_source"] = "plist_missing"
        status["launchd_loaded"] = False
        return status
    try:
        list_proc = subprocess.run(
            ["launchctl", "list", label],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        status["active_source"] = "launchctl_unavailable"
        status["launchd_loaded"] = False
        status["launchctl_error"] = exc.__class__.__name__
        return status
    if list_proc.returncode == 0:
        status["launchd_loaded"] = True
        status["active"] = bool(heartbeat.get("fresh") or not heartbeat_required)
        if heartbeat_required and not heartbeat.get("fresh"):
            status["active_source"] = "stale_heartbeat"
        return status
    try:
        print_proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        status["active_source"] = "launchctl_unavailable"
        status["launchd_loaded"] = False
        status["launchctl_error"] = exc.__class__.__name__
        return status
    loaded = print_proc.returncode == 0
    status["launchd_loaded"] = loaded
    status["active"] = bool(loaded and (heartbeat.get("fresh") or not heartbeat_required))
    if loaded and heartbeat_required and not heartbeat.get("fresh"):
        status["active_source"] = "stale_heartbeat"
    return status


def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw).expanduser() if raw else default
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _freshest_guardian_heartbeat(paths: list[Path], *, max_age_seconds: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    best: dict[str, Any] | None = None
    for path in paths:
        if not path.exists():
            continue
        generated: datetime | None = None
        status: Any = None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            payload = {}
        if isinstance(payload, dict):
            generated = _parse_guardian_heartbeat_time(payload.get("generated_at_utc"))
            status = payload.get("status")
        source = "generated_at_utc"
        if generated is None:
            try:
                generated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            source = "mtime"
        age = (now - generated).total_seconds()
        item = {
            "path": str(path),
            "generated_at_utc": generated.isoformat(),
            "age_seconds": round(age, 3),
            "fresh": -60.0 <= age <= max_age_seconds,
            "source": source,
            "status": status,
        }
        if best is None or item["age_seconds"] < best["age_seconds"]:
            best = item
    if best is None:
        return {
            "path": None,
            "generated_at_utc": None,
            "age_seconds": None,
            "fresh": False,
            "source": None,
            "status": "MISSING",
        }
    return best


def _parse_guardian_heartbeat_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _history_run_signature(
    payload: dict[str, Any],
    *,
    output_path: Path,
    report_path: Path,
) -> tuple[Any, ...]:
    findings = payload.get("findings") or []
    effect = payload["effect_metrics"]["window"]
    runtime = payload["runtime"]
    return (
        payload["status"],
        str(output_path),
        str(report_path),
        _history_float(payload["window_hours"]),
        len(findings),
        sum(1 for item in findings if item["priority"] == "P0"),
        sum(1 for item in findings if item["priority"] == "P1"),
        sum(1 for item in findings if item["priority"] == "P2"),
        int(effect.get("closed_trades") or 0),
        _history_float(effect.get("net_jpy") or 0.0),
        _history_float(_maybe_float(effect.get("profit_factor"))),
        _history_float(_maybe_float(effect.get("expectancy_jpy"))),
        int(runtime.get("live_ready_lanes") or 0),
        int(runtime.get("open_trader_positions") or 0),
    )


def _history_row_signature(row: sqlite3.Row) -> tuple[Any, ...]:
    return (
        row["status"],
        row["output_path"],
        row["report_path"],
        _history_float(row["window_hours"]),
        int(row["findings"] or 0),
        int(row["p0_findings"] or 0),
        int(row["p1_findings"] or 0),
        int(row["p2_findings"] or 0),
        int(row["closed_trades"] or 0),
        _history_float(row["net_jpy"] or 0.0),
        _history_float(row["profit_factor"]),
        _history_float(row["expectancy_jpy"]),
        int(row["live_ready_lanes"] or 0),
        int(row["open_trader_positions"] or 0),
    )


def _history_findings_signature(findings: list[dict[str, Any]]) -> tuple[tuple[str, str, str, str, str, str], ...]:
    return tuple(
        sorted(
            _history_finding_signature(
                priority=str(item.get("priority") or ""),
                layer=str(item.get("layer") or ""),
                code=str(item.get("code") or ""),
                message=str(item.get("message") or ""),
                next_action=str(item.get("next_action") or ""),
                evidence=item.get("evidence") or {},
            )
            for item in findings
        )
    )


def _history_db_findings_signature(rows: list[sqlite3.Row]) -> tuple[tuple[str, str, str, str, str, str], ...]:
    return tuple(
        sorted(
            _history_finding_signature(
                priority=str(row["priority"] or ""),
                layer=str(row["layer"] or ""),
                code=str(row["code"] or ""),
                message=str(row["message"] or ""),
                next_action=str(row["next_action"] or ""),
                evidence=_history_evidence_from_json(str(row["evidence_json"] or "{}")),
            )
            for row in rows
        )
    )


def _history_finding_signature(
    *,
    priority: str,
    layer: str,
    code: str,
    message: str,
    next_action: str,
    evidence: Any,
) -> tuple[str, str, str, str, str, str]:
    normalized_code = str(code or "")
    return (
        priority,
        layer,
        normalized_code,
        _history_normalized_message(normalized_code, message),
        next_action,
        json.dumps(
            _history_normalized_evidence(normalized_code, evidence),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _history_normalized_message(code: str, message: str) -> str:
    if code == REPEATED_REPAIR_LOOP_CODE:
        marker = " has persisted for "
        prefix, sep, suffix = message.partition(marker)
        if not sep:
            return message
        _, _, tail = suffix.partition(" non-duplicate audit run(s)")
        return f"{prefix}{marker}<streak> non-duplicate audit run(s){tail}"
    if code != "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED":
        return message
    marker = " consecutive audit run(s): "
    _, sep, suffix = message.partition(marker)
    if not sep:
        return message
    return f"profitability discipline has failed for <streak>{marker}{suffix}"


def _history_normalized_evidence(code: str, evidence: Any) -> Any:
    if code == "LIVE_RUNTIME_UPDATE_IN_PROGRESS" and isinstance(evidence, dict):
        normalized = dict(evidence)
        normalized.pop("lock_age_seconds", None)
        return normalized
    if code not in HISTORY_VOLATILE_STREAK_CODES or not isinstance(evidence, dict):
        return evidence
    return _history_strip_volatile_streak_fields(evidence)


def _history_strip_volatile_streak_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _history_strip_volatile_streak_fields(item)
            for key, item in value.items()
            if key not in {"current_streak", "previous_streak"}
        }
    if isinstance(value, list):
        return [_history_strip_volatile_streak_fields(item) for item in value]
    return value


HISTORY_VOLATILE_STREAK_CODES = frozenset(
    {
        "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
        "LATEST_GPT_DECISION_STALE",
        REPEATED_REPAIR_LOOP_CODE,
    }
)


def _history_evidence_from_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _history_float(value: Any) -> float | None:
    if value is None:
        return None
    # Nine decimal places preserves audit metrics while ignoring SQLite/Python
    # binary float representation noise during duplicate-run comparison.
    return round(float(value), 9)


@dataclass(frozen=True)
class _LoadedJson:
    payload: dict[str, Any] | None
    error: str | None


def _read_json(path: Path) -> _LoadedJson:
    if not path.exists():
        return _LoadedJson(None, "missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _LoadedJson(None, str(exc))
    if not isinstance(payload, dict):
        return _LoadedJson(None, "not a JSON object")
    return _LoadedJson(payload, None)


def _read_jsonl(path: Path) -> tuple[tuple[dict[str, Any], ...], int, str | None]:
    if not path.exists():
        return (), 0, "missing"
    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if isinstance(item, dict):
                    rows.append(item)
                else:
                    malformed += 1
    except OSError as exc:
        return (), malformed, str(exc)
    return tuple(rows), malformed, None


def _memory_findings(
    *,
    run_id: str,
    loaded: _LoadedJson,
    path: Path,
    target_open: bool,
    snapshot: dict[str, Any],
    intents: dict[str, Any],
) -> list[dict[str, Any]]:
    if loaded.error is not None:
        return [
            _finding(
                run_id=run_id,
                priority="P0" if target_open else "P1",
                layer="memory",
                code="MEMORY_HEALTH_UNREADABLE",
                message=f"memory_health artifact is unreadable: {path}: {loaded.error}",
                next_action="Run memory-health and repair any BLOCK before target-open entry routing.",
            )
        ]
    payload = loaded.payload or {}
    if target_open:
        generated_at = _parse_utc(payload.get("generated_at_utc"))
        if generated_at is None:
            return [
                _finding(
                    run_id=run_id,
                    priority="P0",
                    layer="memory",
                    code="MEMORY_HEALTH_STALE",
                    message=f"memory_health lacks generated_at_utc while target is open: {path}",
                    next_action="Regenerate memory-health before repairing its internal blockers or trusting entry/verify routing.",
                    evidence={"path": str(path), "reason": "missing_generated_at_utc"},
                )
            ]
        stale_refs = _memory_health_stale_refs(
            payload=payload,
            generated_at=generated_at,
            snapshot=snapshot,
            intents=intents,
        )
        if stale_refs:
            return [
                _finding(
                    run_id=run_id,
                    priority="P0",
                    layer="memory",
                    code="MEMORY_HEALTH_STALE",
                    message=(
                        "memory_health predates current routed evidence while target is open: "
                        + "; ".join(
                            f"{item['label']}={item['timestamp_utc']}" for item in stale_refs
                        )
                    ),
                    next_action=(
                        "Run memory-health against the current broker snapshot/order intents before "
                        "repairing old artifact blockers or trusting entry/verify routing."
                    ),
                    evidence={
                        "memory_health_generated_at_utc": generated_at.isoformat(),
                        "stale_against": stale_refs,
                    },
                )
            ]
    status = str(payload.get("status") or "")
    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    if "BLOCK" in status.upper() or blockers:
        return [
            _finding(
                run_id=run_id,
                priority="P0",
                layer="memory",
                code="MEMORY_HEALTH_BLOCKED",
                message=f"memory_health is blocked with {len(blockers) or len(issues)} issue(s)",
                next_action="Repair the first memory-health BLOCK, then rerun trader-prompt-route.",
                evidence={"status": status, "blockers": blockers[:8], "issue_codes": [str(i.get("code")) for i in issues[:12] if isinstance(i, dict)]},
            )
        ]
    if "WARN" in status.upper():
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="memory",
                code="MEMORY_HEALTH_WARN",
                message="memory_health has warnings that can reduce learning reliability",
                next_action="Clear memory-health warnings during the next refresh branch.",
                evidence={"status": status, "warnings": payload.get("warnings", [])[:8] if isinstance(payload.get("warnings"), list) else []},
            )
        ]
    return []


def _memory_health_stale_refs(
    *,
    payload: dict[str, Any],
    generated_at: datetime,
    snapshot: dict[str, Any],
    intents: dict[str, Any],
) -> list[dict[str, str]]:
    """Match trader routing freshness so audit fixes current holes, not old ones."""

    refs: list[tuple[str, datetime]] = []
    account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
    snapshot_ts = _parse_utc(snapshot.get("fetched_at_utc") or account.get("fetched_at_utc"))
    intents_ts = _parse_utc(intents.get("generated_at_utc"))
    if snapshot_ts is not None:
        refs.append(("broker_snapshot", snapshot_ts))
    if intents_ts is not None:
        refs.append(("order_intents", intents_ts))

    stale: list[dict[str, str]] = []
    for label, ref_ts in refs:
        audited_ts = _memory_health_audited_ref_ts(payload, label)
        effective_ts = audited_ts or generated_at
        if effective_ts < ref_ts:
            item = {"label": label, "timestamp_utc": ref_ts.isoformat()}
            if audited_ts is not None:
                item["audited_timestamp_utc"] = audited_ts.isoformat()
            stale.append(item)
    return stale


def _memory_health_audited_ref_ts(payload: dict[str, Any], label: str) -> datetime | None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    runtime = metrics.get("runtime") if isinstance(metrics.get("runtime"), dict) else {}
    if label == "broker_snapshot":
        return _parse_utc(runtime.get("snapshot_fetched_at_utc"))
    if label == "order_intents":
        order_metrics = metrics.get("order_intents") if isinstance(metrics.get("order_intents"), dict) else {}
        return _parse_utc(
            runtime.get("order_intents_generated_at_utc")
            or order_metrics.get("generated_at_utc")
        )
    return None


def _learning_findings(*, run_id: str, loaded: _LoadedJson, path: Path) -> list[dict[str, Any]]:
    if loaded.error is not None:
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="learning",
                code="LEARNING_AUDIT_UNREADABLE",
                message=f"learning_audit artifact is unreadable: {path}: {loaded.error}",
                next_action="Run learning-audit before allowing learning-influenced lane ranking.",
            )
        ]
    payload = loaded.payload or {}
    status = str(payload.get("status") or "")
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if "BLOCK" in status.upper() or blockers:
        if _learning_audit_block_only_quarantines_influenced_lanes(payload):
            influence = payload.get("learning_influence") if isinstance(payload.get("learning_influence"), dict) else {}
            return [
                _finding(
                    run_id=run_id,
                    priority="P1",
                    layer="learning",
                    code="LEARNING_AUDIT_INFLUENCED_LANES_QUARANTINED",
                    message=(
                        "learning_audit blocks only risk-increasing learning influence; "
                        "non-learning live-ready lanes can still be routed"
                    ),
                    next_action=(
                        "Do not select learning-influenced lanes until learning-audit passes; "
                        "continue evaluating clean live-ready lanes through GPT verification."
                    ),
                    evidence={
                        "status": status,
                        "blockers": blockers[:8],
                        "influenced_lanes": int(_maybe_float(influence.get("influenced_lanes")) or 0),
                        "risk_increasing_lanes": int(_maybe_float(influence.get("risk_increasing_lanes")) or 0),
                    },
                )
            ]
        return [
            _finding(
                run_id=run_id,
                priority="P0",
                layer="learning",
                code="LEARNING_AUDIT_BLOCKED",
                message=f"learning_audit is blocked with {len(blockers)} blocker(s)",
                next_action="Remove or quarantine learning influence until learning-audit passes.",
                evidence={"status": status, "blockers": blockers[:8]},
            )
        ]
    if "WARN" in status.upper() or warnings:
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="learning",
                code="LEARNING_AUDIT_WARN",
                message=f"learning_audit has {len(warnings)} warning(s)",
                next_action="Do not increase learning score impact until the effect window improves.",
                evidence={"status": status, "warnings": warnings[:8]},
            )
        ]
    return []


def _learning_audit_block_only_quarantines_influenced_lanes(payload: dict[str, Any]) -> bool:
    influence = payload.get("learning_influence") if isinstance(payload.get("learning_influence"), dict) else {}
    if int(_maybe_float(influence.get("influenced_lanes")) or 0) <= 0:
        return False
    blockers = [str(item) for item in (payload.get("blockers") or []) if str(item).strip()]
    if not blockers:
        return False
    if not all(_is_learning_lane_quarantine_message(message) for message in blockers):
        return False
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    blocking_checks = [
        item
        for item in checks
        if isinstance(item, dict)
        and (
            str(item.get("status") or "").upper() == "BLOCK"
            or str(item.get("severity") or "").upper() == "BLOCK"
        )
    ]
    if not blocking_checks:
        return True
    return all(
        str(item.get("check_name") or "") == "learning_influence_recent_outcome"
        or _is_learning_lane_quarantine_message(str(item.get("message") or ""))
        for item in blocking_checks
    )


def _is_learning_lane_quarantine_message(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in _LEARNING_AUDIT_LANE_QUARANTINE_MESSAGES)


def _verification_findings(
    *,
    run_id: str,
    loaded: _LoadedJson,
    path: Path,
) -> list[dict[str, Any]]:
    if loaded.error is not None:
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="verification",
                code="VERIFICATION_LEDGER_UNREADABLE",
                message=f"verification_ledger artifact is unreadable: {path}: {loaded.error}",
                next_action="Run verification-ledger-audit so cycle evidence is queryable.",
            )
        ]
    payload = loaded.payload or {}
    status = str(payload.get("status") or "")
    blocking = int(_maybe_float(payload.get("blocking_observations")) or 0)
    blocking_evidence = payload.get("blocking_evidence") if isinstance(payload.get("blocking_evidence"), list) else []
    if status == "BLOCKED" or blocking > 0:
        if _only_order_intent_lane_blockers(blocking_evidence):
            return [
                _finding(
                    run_id=run_id,
                    priority="P1",
                    layer="verification",
                    code="VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED",
                    message=(
                        f"verification ledger recorded {blocking} protective order-intent lane blocker(s); "
                        "ledger integrity itself is not the P0"
                    ),
                    next_action="Repair the order_intents top blockers instead of treating the verification ledger as broken.",
                    evidence={
                        "status": status,
                        "blocking_observations": blocking,
                        "blocking_evidence": blocking_evidence[:8],
                    },
                )
            ]
        return [
            _finding(
                run_id=run_id,
                priority="P0",
                layer="verification",
                code="VERIFICATION_LEDGER_BLOCKED",
                message=f"verification ledger is blocked with {blocking} blocking observation(s)",
                next_action="Fix the top verification blocking evidence before the next new-risk cycle.",
                evidence={"status": status, "blocking_observations": blocking, "blocking_evidence": blocking_evidence[:8]},
            )
        ]
    return []


def _ledger_sync_findings(*, run_id: str, db_path: Path, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
    expected = str(account.get("last_transaction_id") or "").strip()
    actual: str | None = None
    error: str | None = None
    if not db_path.exists():
        error = "missing"
    else:
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT value FROM sync_state WHERE key='last_oanda_transaction_id'"
                ).fetchone()
                actual = str(row[0] or "").strip() if row else None
        except sqlite3.Error as exc:
            error = str(exc)
    if error is not None:
        return [
            _finding(
                run_id=run_id,
                priority="P0",
                layer="execution_ledger",
                code="EXECUTION_LEDGER_UNREADABLE",
                message=f"execution ledger is unreadable: {db_path}: {error}",
                next_action="Repair execution_ledger.db or restore the latest valid ledger before live routing.",
            )
        ]
    if expected and (not actual or _transaction_id_is_behind(actual, expected)):
        return [
            _finding(
                run_id=run_id,
                priority="P0",
                layer="execution_ledger",
                code="EXECUTION_LEDGER_STALE",
                message=f"execution ledger last transaction `{actual}` is behind broker snapshot `{expected}`",
                next_action="Run execution-ledger-sync before trusting decision or learning history.",
                evidence={"ledger_last_transaction_id": actual, "broker_last_transaction_id": expected},
            )
        ]
    return []


def _pending_entry_lifecycle_findings(
    *,
    run_id: str,
    metrics: dict[str, Any],
    target_open: bool,
) -> list[dict[str, Any]]:
    if not target_open or metrics.get("error"):
        return []
    accepted = int(metrics.get("accepted_entry_orders") or 0)
    filled = int(metrics.get("filled_entry_orders") or 0)
    canceled_before_fill = int(metrics.get("canceled_before_fill_orders") or 0)
    cancel_rate = _maybe_float(metrics.get("cancel_before_fill_rate"))
    fill_rate = _maybe_float(metrics.get("fill_rate"))
    if (
        accepted < PENDING_ENTRY_LIFECYCLE_MIN_ACCEPTED
        or cancel_rate is None
        or cancel_rate <= PENDING_ENTRY_CANCEL_RATE_WARN_ABOVE
    ):
        return []
    if filled > 0:
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="execution_quality",
                code="PENDING_ENTRY_CANCEL_RATE_HIGH",
                message=(
                    f"{accepted} accepted entry order(s) in the audit window filled {filled} time(s), "
                    f"but {canceled_before_fill} order(s) were canceled before fill"
                ),
                next_action=(
                    "Treat cancel churn as a downstream symptom: separate thesis invalidation from "
                    "entry-distance/TTL failures by pair, then preserve pending entries whose forecast "
                    "and range-rail thesis remain valid."
                ),
                evidence={
                    "accepted_entry_orders": accepted,
                    "filled_entry_orders": filled,
                    "canceled_before_fill_orders": canceled_before_fill,
                    "canceled_before_fill_replaced_orders": int(
                        metrics.get("canceled_before_fill_replaced_orders") or 0
                    ),
                    "canceled_before_fill_orphan_orders": int(
                        metrics.get("canceled_before_fill_orphan_orders") or 0
                    ),
                    "cancel_before_fill_rate": cancel_rate,
                    "cancel_replacement_rate": metrics.get("cancel_replacement_rate"),
                    "fill_rate": fill_rate,
                    "timing_regret": metrics.get("timing_regret") or {},
                    "samples": metrics.get("canceled_before_fill_samples", [])[:8],
                    "canceled_before_fill_orphan_groups": metrics.get(
                        "canceled_before_fill_orphan_groups", []
                    )[:12],
                    "canceled_before_fill_replaced_groups": metrics.get(
                        "canceled_before_fill_replaced_groups", []
                    )[:12],
                },
            )
        ]
    return [
        _finding(
            run_id=run_id,
            priority="P1",
            layer="execution_quality",
            code="PENDING_ENTRY_FILL_RATE_WEAK",
            message=(
                f"{accepted} accepted entry order(s) in the audit window produced {filled} fill(s) "
                f"and {canceled_before_fill} cancellation(s) before fill"
            ),
            next_action=(
                "Treat execution lifecycle as the active repair surface: preserve broker-anchored pending "
                "entries until thesis invalidation, then audit entry distance/TTL by pair instead of issuing "
                "another generic CANCEL_PENDING."
            ),
            evidence={
                "accepted_entry_orders": accepted,
                "filled_entry_orders": filled,
                "canceled_before_fill_orders": canceled_before_fill,
                "canceled_before_fill_replaced_orders": int(
                    metrics.get("canceled_before_fill_replaced_orders") or 0
                ),
                "canceled_before_fill_orphan_orders": int(
                    metrics.get("canceled_before_fill_orphan_orders") or 0
                ),
                "cancel_before_fill_rate": cancel_rate,
                "cancel_replacement_rate": metrics.get("cancel_replacement_rate"),
                "fill_rate": fill_rate,
                "timing_regret": metrics.get("timing_regret") or {},
                "samples": metrics.get("canceled_before_fill_samples", [])[:8],
            },
        )
    ]


def _profit_capture_miss_findings(
    *,
    run_id: str,
    timing_payload: dict[str, Any] | None,
    target_open: bool,
) -> list[dict[str, Any]]:
    if not isinstance(timing_payload, dict):
        return []
    summary = timing_payload.get("summary") if isinstance(timing_payload.get("summary"), dict) else {}
    missed = int(summary.get("loss_closes_profit_capture_missed") or 0)
    if missed <= 0:
        return []
    repair_replay_contract = repair_replay_contract_from_payload(timing_payload)
    repair_replay_contract_present = repair_replay_contract == TP_PROGRESS_REPAIR_REPLAY_CONTRACT
    repair_replay_missed = int(summary.get("loss_closes_repair_replay_triggered") or 0)
    production_gate_p0 = (not repair_replay_contract_present) or repair_replay_missed > 0
    rows = [
        row
        for row in (timing_payload.get("loss_close_regrets") or [])
        if isinstance(row, dict) and row.get("profit_capture_missed_before_loss_close")
    ]
    rows.sort(
        key=lambda row: float(
            row.get("profit_capture_counterfactual_net_improvement_jpy")
            or row.get("estimated_mfe_jpy_before_loss_close")
            or 0.0
        ),
        reverse=True,
    )
    repair_rows = [
        row
        for row in (timing_payload.get("loss_close_regrets") or [])
        if isinstance(row, dict) and row.get("repair_replay_triggered_before_loss_close")
    ]
    repair_rows.sort(
        key=lambda row: float(
            row.get("repair_replay_counterfactual_net_improvement_jpy")
            or row.get("profit_capture_counterfactual_net_improvement_jpy")
            or 0.0
        ),
        reverse=True,
    )
    repair_block_rows = [
        row
        for row in (timing_payload.get("loss_close_regrets") or [])
        if (
            isinstance(row, dict)
            and row.get("profit_capture_missed_before_loss_close")
            and not row.get("repair_replay_triggered_before_loss_close")
        )
    ]
    repair_block_rows.sort(
        key=lambda row: float(
            row.get("profit_capture_counterfactual_net_improvement_jpy")
            or row.get("estimated_mfe_jpy_before_loss_close")
            or 0.0
        ),
        reverse=True,
    )
    if not repair_replay_contract_present:
        message = (
            f"{missed} losing close(s) have raw TP-progress capture evidence, but the "
            "execution-timing-audit sidecar lacks current production-gate replay proof"
        )
        next_action = (
            "Regenerate execution-timing-audit with the current runtime before treating "
            "TP-progress repair as proved or cleared."
        )
    elif repair_replay_missed > 0:
        message = (
            f"{repair_replay_missed} losing close(s) had production-gate replay proof that "
            f"TP-progress capture was executable before closing red ({missed} raw TP-progress miss(es))"
        )
        next_action = (
            "Keep the TP-progress TAKE_PROFIT_MARKET path and position guardian active, then "
            "rerun execution-timing-audit until loss_closes_repair_replay_triggered is zero."
        )
    else:
        message = (
            f"{missed} raw TP-progress miss(es) remain, but production-gate replay found no "
            "executable profit-capture trigger"
        )
        next_action = (
            "Keep these rows as diagnostic only; use tick replay or improved candle ordering to "
            "upgrade them before blocking high-turnover entries."
        )
    return [
        _finding(
            run_id=run_id,
            priority="P0" if target_open and production_gate_p0 else "P1",
            layer="execution_quality",
            code="LOSS_CLOSE_PROFIT_CAPTURE_MISSED",
            message=message,
            next_action=next_action,
            evidence={
                "generated_at_utc": timing_payload.get("generated_at_utc"),
                "loss_closes_audited": int(summary.get("loss_closes_audited") or 0),
                "loss_closes_profit_capture_missed": missed,
                "loss_closes_profit_capture_missed_rate": _maybe_float(
                    summary.get("loss_closes_profit_capture_missed_rate")
                ),
                "stop_loss_closes_profit_capture_missed": int(
                    summary.get("stop_loss_closes_profit_capture_missed") or 0
                ),
                "loss_close_estimated_capture_gap_jpy": _maybe_float(
                    summary.get("loss_close_estimated_capture_gap_jpy")
                ),
                "loss_close_actual_pl_jpy": _maybe_float(summary.get("loss_close_actual_pl_jpy")),
                "loss_close_counterfactual_profit_capture_pl_jpy": _maybe_float(
                    summary.get("loss_close_counterfactual_profit_capture_pl_jpy")
                ),
                "loss_close_counterfactual_profit_capture_delta_jpy": _maybe_float(
                    summary.get("loss_close_counterfactual_profit_capture_delta_jpy")
                ),
                "loss_close_counterfactual_profit_capture_jpy": _maybe_float(
                    summary.get("loss_close_counterfactual_profit_capture_jpy")
                ),
                "repair_replay_contract": repair_replay_contract,
                "repair_replay_contract_present": repair_replay_contract_present,
                "loss_closes_repair_replay_triggered": repair_replay_missed,
                "loss_closes_repair_replay_triggered_rate": _maybe_float(
                    summary.get("loss_closes_repair_replay_triggered_rate")
                ),
                "loss_close_repair_replay_profit_capture_jpy": _maybe_float(
                    summary.get("loss_close_repair_replay_profit_capture_jpy")
                ),
                "loss_close_repair_replay_counterfactual_pl_jpy": _maybe_float(
                    summary.get("loss_close_repair_replay_counterfactual_pl_jpy")
                ),
                "loss_close_repair_replay_delta_jpy": _maybe_float(
                    summary.get("loss_close_repair_replay_delta_jpy")
                ),
                "loss_close_repair_replay_block_reasons": (
                    summary.get("loss_close_repair_replay_block_reasons")
                    if isinstance(summary.get("loss_close_repair_replay_block_reasons"), dict)
                    else {}
                ),
                "top_profit_capture_misses": [
                    {
                        "trade_id": str(row.get("trade_id") or ""),
                        "lane_id": str(row.get("lane_id") or ""),
                        "pair": str(row.get("pair") or ""),
                        "side": str(row.get("side") or ""),
                        "exit_reason": str(row.get("exit_reason") or ""),
                        "realized_pl_jpy": _maybe_float(row.get("realized_pl_jpy")),
                        "mfe_pips_before_loss_close": _maybe_float(row.get("mfe_pips_before_loss_close")),
                        "tp_progress_before_loss_close": _maybe_float(
                            row.get("tp_progress_before_loss_close")
                        ),
                        "estimated_mfe_jpy_before_loss_close": _maybe_float(
                            row.get("estimated_mfe_jpy_before_loss_close")
                        ),
                        "profit_capture_counterfactual_exit": row.get(
                            "profit_capture_counterfactual_exit"
                        ),
                        "profit_capture_counterfactual_pips": _maybe_float(
                            row.get("profit_capture_counterfactual_pips")
                        ),
                        "profit_capture_counterfactual_jpy": _maybe_float(
                            row.get("profit_capture_counterfactual_jpy")
                        ),
                        "profit_capture_counterfactual_net_improvement_jpy": _maybe_float(
                            row.get("profit_capture_counterfactual_net_improvement_jpy")
                        ),
                        "repair_replay_block_reason": row.get("repair_replay_block_reason"),
                        "repair_replay_candidate_profit_pips": _maybe_float(
                            row.get("repair_replay_candidate_profit_pips")
                        ),
                        "repair_replay_candidate_noise_floor_pips": _maybe_float(
                            row.get("repair_replay_candidate_noise_floor_pips")
                        ),
                    }
                    for row in rows[:8]
                ],
                "top_repair_replay_triggers": [
                    {
                        "trade_id": str(row.get("trade_id") or ""),
                        "lane_id": str(row.get("lane_id") or ""),
                        "pair": str(row.get("pair") or ""),
                        "side": str(row.get("side") or ""),
                        "exit_reason": str(row.get("exit_reason") or ""),
                        "realized_pl_jpy": _maybe_float(row.get("realized_pl_jpy")),
                        "repair_replay_exit": row.get("repair_replay_exit"),
                        "repair_replay_trigger_at_utc": row.get(
                            "repair_replay_trigger_at_utc"
                        ),
                        "repair_replay_profit_pips": _maybe_float(
                            row.get("repair_replay_profit_pips")
                        ),
                        "repair_replay_noise_floor_pips": _maybe_float(
                            row.get("repair_replay_noise_floor_pips")
                        ),
                        "repair_replay_counterfactual_jpy": _maybe_float(
                            row.get("repair_replay_counterfactual_jpy")
                        ),
                        "repair_replay_counterfactual_net_improvement_jpy": _maybe_float(
                            row.get("repair_replay_counterfactual_net_improvement_jpy")
                        ),
                    }
                    for row in repair_rows[:8]
                ],
                "top_repair_replay_blocks": [
                    {
                        "trade_id": str(row.get("trade_id") or ""),
                        "lane_id": str(row.get("lane_id") or ""),
                        "pair": str(row.get("pair") or ""),
                        "side": str(row.get("side") or ""),
                        "exit_reason": str(row.get("exit_reason") or ""),
                        "realized_pl_jpy": _maybe_float(row.get("realized_pl_jpy")),
                        "repair_replay_block_reason": row.get("repair_replay_block_reason"),
                        "repair_replay_max_profit_pips": _maybe_float(
                            row.get("repair_replay_max_profit_pips")
                        ),
                        "repair_replay_max_tp_progress": _maybe_float(
                            row.get("repair_replay_max_tp_progress")
                        ),
                        "repair_replay_candidate_profit_pips": _maybe_float(
                            row.get("repair_replay_candidate_profit_pips")
                        ),
                        "repair_replay_candidate_tp_progress": _maybe_float(
                            row.get("repair_replay_candidate_tp_progress")
                        ),
                        "repair_replay_candidate_spread_pips": _maybe_float(
                            row.get("repair_replay_candidate_spread_pips")
                        ),
                        "repair_replay_candidate_m1_atr_pips": _maybe_float(
                            row.get("repair_replay_candidate_m1_atr_pips")
                        ),
                        "repair_replay_candidate_noise_floor_pips": _maybe_float(
                            row.get("repair_replay_candidate_noise_floor_pips")
                        ),
                    }
                    for row in repair_block_rows[:8]
                ],
            },
        )
    ]


def _position_guardian_profit_capture_findings(
    *,
    run_id: str,
    guardian_status: dict[str, Any],
    target_open: bool,
    live_ready_lanes: int,
    open_trader_positions: int,
    open_trader_pending_entries: int,
    profit_capture_miss_active: bool,
) -> list[dict[str, Any]]:
    if not guardian_status.get("required") or guardian_status.get("active"):
        return []
    current_exposure_surface = (
        live_ready_lanes > 0
        or open_trader_positions > 0
        or open_trader_pending_entries > 0
        or profit_capture_miss_active
    )
    if not target_open and not current_exposure_surface:
        return []
    priority = "P0" if target_open and current_exposure_surface else "P1"
    evidence = {
        "guardian": guardian_status,
        "target_open": target_open,
        "live_ready_lanes": live_ready_lanes,
        "open_trader_positions": open_trader_positions,
        "open_trader_pending_entries": open_trader_pending_entries,
        "profit_capture_miss_active": profit_capture_miss_active,
    }
    return [
        _finding(
            run_id=run_id,
            priority=priority,
            layer="execution_quality",
            code="POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE",
            message=(
                "position guardian is required but inactive; TP-progress profit cannot be "
                "captured between full trader cycles"
            ),
            next_action=(
                "Treat this as a profit-capture outage, not an entry-resend problem: run "
                "scripts/install-position-guardian.sh --check, then load/keep the guardian active "
                "only with explicit operator approval; until then, fresh entries remain blocked "
                "because plus P/L can reverse before the next full cycle."
            ),
            evidence=evidence,
        )
    ]


def _pending_entry_reconcile_findings(
    *,
    run_id: str,
    metrics: dict[str, Any],
    target_open: bool,
) -> list[dict[str, Any]]:
    if not target_open or metrics.get("error"):
        return []
    cancel_ids = [str(item) for item in metrics.get("cancel_review_order_ids") or [] if str(item)]
    if not cancel_ids:
        return []
    return [
        _finding(
            run_id=run_id,
            priority="P0",
            layer="execution_quality",
            code="PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
            message=f"{len(cancel_ids)} trader-owned pending entry order(s) need cancel review",
            next_action=(
                "Write a CANCEL_PENDING receipt for these order ids when no current LIVE_READY replacement "
                "exists, or write a TRADE receipt with cancel_order_ids when replacing them with a current "
                "verified basket. Keep only orders whose attached risk and current thesis still match the packet."
            ),
            evidence={
                "cancel_review_order_ids": cancel_ids,
                "groups": metrics.get("cancel_review_groups") or [],
                "orders": metrics.get("orders") or [],
            },
        )
    ]


def _pending_entry_reconcile_metrics(
    *,
    snapshot: dict[str, Any],
    target_state: dict[str, Any],
    intents: dict[str, Any],
    attack_advice: dict[str, Any],
    target_open: bool,
) -> dict[str, Any]:
    orders = _trader_pending_entry_orders(snapshot) if target_open else []
    metrics: dict[str, Any] = {
        "target_open": target_open,
        "reviewed_open_pending_orders": len(orders),
        "cancel_review_orders": 0,
        "cancel_review_order_ids": [],
        "cancel_review_groups": [],
        "orders": [],
        "error": None,
    }
    if not target_open or not orders:
        return metrics

    per_trade_cap = _maybe_float(target_state.get("per_trade_risk_budget_jpy"))
    portfolio_remaining = _pending_entry_portfolio_remaining_jpy(target_state)
    intent_index = _pending_reconcile_intent_index(intents)
    recommended_parents = {
        _parent_lane_id(str(item))
        for item in attack_advice.get("recommended_now_lane_ids") or []
        if str(item).strip()
    }
    attack_generated_at = _parse_utc(attack_advice.get("generated_at_utc"))

    reviews: list[dict[str, Any]] = []
    cumulative_attached_risk = 0.0
    for order in orders:
        review = _pending_entry_order_reconcile(
            order=order,
            snapshot=snapshot,
            per_trade_cap=per_trade_cap,
            portfolio_remaining=portfolio_remaining,
            cumulative_attached_risk=cumulative_attached_risk,
            intent_index=intent_index,
            recommended_parent_lanes=recommended_parents,
            attack_generated_at=attack_generated_at,
        )
        risk_jpy = _maybe_float(review.get("attached_sl_risk_jpy"))
        if risk_jpy is not None:
            cumulative_attached_risk += max(0.0, risk_jpy)
        if review.get("review_reasons"):
            reviews.append(review)

    metrics["orders"] = reviews
    metrics["cancel_review_orders"] = len(reviews)
    metrics["cancel_review_order_ids"] = [item.get("order_id") for item in reviews if item.get("order_id")]
    metrics["cancel_review_groups"] = _pending_reconcile_groups(reviews)
    return metrics


def _pending_entry_order_reconcile(
    *,
    order: dict[str, Any],
    snapshot: dict[str, Any],
    per_trade_cap: float | None,
    portfolio_remaining: float | None,
    cumulative_attached_risk: float,
    intent_index: dict[str, Any],
    recommended_parent_lanes: set[str],
    attack_generated_at: datetime | None,
) -> dict[str, Any]:
    pair = str(order.get("pair") or "")
    side = _pending_order_side(order)
    lane_id = _pending_order_lane_id(order)
    parent_lane_id = _parent_lane_id(lane_id) if lane_id else None
    method = _pending_order_method(order, parent_lane_id)
    order_type = _normalized_pending_order_type(order.get("order_type"))
    created_at = _pending_order_create_time(order)
    risk = _pending_order_attached_sl_risk_jpy(order, snapshot)
    candidates = _pending_reconcile_candidates(
        pair=pair,
        side=side,
        method=method,
        order_type=order_type,
        parent_lane_id=parent_lane_id,
        intent_index=intent_index,
    )

    reasons: list[dict[str, Any]] = []
    portfolio_remaining_before_order = None
    if portfolio_remaining is not None:
        portfolio_remaining_before_order = max(0.0, portfolio_remaining - max(0.0, cumulative_attached_risk))
    risk_cap_basis = "PER_TRADE_CAP"
    if per_trade_cap is not None and per_trade_cap > 0:
        if risk.get("risk_jpy") is None and risk.get("issue_code"):
            reasons.append(
                {
                    "code": risk["issue_code"],
                    "message": risk["message"],
                }
            )
        elif _maybe_float(risk.get("risk_jpy")) is not None and float(risk["risk_jpy"]) > per_trade_cap:
            risk_jpy = float(risk["risk_jpy"])
            if _pending_gateway_tail_risk_allowed(
                order=order,
                risk_jpy=risk_jpy,
                portfolio_remaining_before_order=portfolio_remaining_before_order,
            ):
                risk_cap_basis = "GATEWAY_TAIL_WITHIN_PORTFOLIO_CAP"
            elif portfolio_remaining_before_order is not None and risk_jpy > portfolio_remaining_before_order:
                risk_cap_basis = "PORTFOLIO_CAP"
                reasons.append(
                    {
                        "code": "PENDING_ATTACHED_SL_PORTFOLIO_RISK_EXCEEDS_CAP",
                        "message": (
                            f"attached stop risk {risk_jpy:.2f} JPY exceeds "
                            f"remaining portfolio risk capacity {portfolio_remaining_before_order:.2f} JPY"
                        ),
                    }
                )
            else:
                risk_cap_basis = "PER_TRADE_CAP"
                reasons.append(
                    {
                        "code": "PENDING_ATTACHED_SL_RISK_EXCEEDS_CAP",
                        "message": (
                            f"attached stop risk {risk_jpy:.2f} JPY exceeds "
                            f"per-trade cap {per_trade_cap:.2f} JPY"
                        ),
                    }
                )

    if parent_lane_id and not candidates:
        reasons.append(
            {
                "code": "PENDING_CURRENT_CANDIDATE_MISSING",
                "message": "pending order parent lane is absent from current order_intents",
            }
        )
    elif candidates and not any(str(item.get("status") or "") == "LIVE_READY" for item in candidates):
        reasons.append(
            {
                "code": "PENDING_CURRENT_CANDIDATE_NOT_LIVE_READY",
                "message": "pending order parent lane is present but no current candidate is LIVE_READY",
                "candidate_statuses": sorted({str(item.get("status") or "UNKNOWN") for item in candidates}),
                "candidate_blockers": _pending_candidate_blockers(candidates),
            }
        )

    if (
        parent_lane_id
        and recommended_parent_lanes
        and parent_lane_id not in recommended_parent_lanes
        and attack_generated_at is not None
        and (created_at is None or attack_generated_at >= created_at)
    ):
        reasons.append(
            {
                "code": "PENDING_ATTACK_ADVICE_NOT_CURRENT",
                "message": "current ai_attack_advice no longer recommends the pending order parent lane",
            }
        )

    return {
        "order_id": order.get("order_id"),
        "pair": pair or None,
        "side": side,
        "method": method,
        "order_type": order_type,
        "price": _pending_order_price(order),
        "units": _maybe_float(order.get("units")),
        "lane_id": lane_id,
        "parent_lane_id": parent_lane_id,
        "created_at_utc": created_at.isoformat() if created_at else None,
        "risk_cap_jpy": per_trade_cap,
        "portfolio_risk_remaining_before_order_jpy": portfolio_remaining_before_order,
        "attached_sl_risk_cap_basis": risk_cap_basis,
        "attached_sl": risk.get("stop_loss"),
        "attached_sl_risk_jpy": risk.get("risk_jpy"),
        "current_candidate_count": len(candidates),
        "current_live_ready_candidate_count": sum(1 for item in candidates if str(item.get("status") or "") == "LIVE_READY"),
        "review_reasons": reasons,
    }


def _pending_entry_portfolio_remaining_jpy(target_state: dict[str, Any]) -> float | None:
    remaining = _maybe_float(target_state.get("remaining_risk_budget_jpy"))
    if remaining is not None and remaining >= 0:
        return remaining
    daily_budget = _maybe_float(target_state.get("daily_risk_budget_jpy"))
    if daily_budget is None:
        return None
    open_risk = _maybe_float(target_state.get("open_risk_jpy")) or 0.0
    return max(0.0, daily_budget - max(0.0, open_risk))


def _pending_gateway_tail_risk_allowed(
    *,
    order: dict[str, Any],
    risk_jpy: float,
    portfolio_remaining_before_order: float | None,
) -> bool:
    if portfolio_remaining_before_order is None or risk_jpy > portfolio_remaining_before_order:
        return False
    # LiveOrderGateway deliberately allows SL-free disaster stops to exceed the
    # per-trade expected-invalidation cap, but counts that tail exposure against
    # the portfolio/day capacity. Broker snapshots do not preserve the original
    # intent metadata, so the gateway lane tag is the durable signal that this
    # pending order came through the executable gateway path rather than a
    # hand-written broker order.
    return _pending_order_has_gateway_lane_tag(order)


def _pending_order_has_gateway_lane_tag(order: dict[str, Any]) -> bool:
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    for extension_key in ("clientExtensions", "tradeClientExtensions"):
        extension = raw.get(extension_key)
        if not isinstance(extension, dict):
            continue
        comment = str(extension.get("comment") or "")
        tag = str(extension.get("tag") or "")
        if "lane=" in comment and ("qr-vnext" in comment or tag == "trader"):
            return True
    return False


def _pending_reconcile_intent_index(intents: dict[str, Any]) -> dict[str, Any]:
    by_parent: dict[str, list[dict[str, Any]]] = {}
    by_shape: dict[tuple[str, str | None, str | None, str], list[dict[str, Any]]] = {}
    for item in intents.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        lane_id = str(item.get("lane_id") or "")
        parent = _parent_lane_id(lane_id) if lane_id else ""
        if parent:
            by_parent.setdefault(parent, []).append(item)
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        pair = str(intent.get("pair") or "")
        side = str(intent.get("side") or "").upper() or None
        method = _method_from_lane_id(parent or lane_id)
        order_type = _normalized_pending_order_type(intent.get("order_type"))
        if pair and order_type:
            by_shape.setdefault((pair, side, method, order_type), []).append(item)
            by_shape.setdefault((pair, side, None, order_type), []).append(item)
    return {"by_parent": by_parent, "by_shape": by_shape}


def _pending_reconcile_candidates(
    *,
    pair: str,
    side: str | None,
    method: str | None,
    order_type: str,
    parent_lane_id: str | None,
    intent_index: dict[str, Any],
) -> list[dict[str, Any]]:
    by_parent = intent_index.get("by_parent") if isinstance(intent_index.get("by_parent"), dict) else {}
    if parent_lane_id:
        parent_candidates = by_parent.get(parent_lane_id)
        if isinstance(parent_candidates, list):
            return [item for item in parent_candidates if isinstance(item, dict)]
    by_shape = intent_index.get("by_shape") if isinstance(intent_index.get("by_shape"), dict) else {}
    candidates = by_shape.get((pair, side, method, order_type)) or by_shape.get((pair, side, None, order_type)) or []
    return [item for item in candidates if isinstance(item, dict)]


def _pending_candidate_blockers(candidates: list[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in candidates:
        for raw in item.get("risk_issues") or []:
            if isinstance(raw, dict):
                code = str(raw.get("code") or raw.get("message") or "").strip()
                severity = str(raw.get("severity") or "").upper()
                if code and severity in {"", "BLOCK"}:
                    blockers.append(code)
        for raw in item.get("live_blockers") or []:
            text = str(raw or "").strip()
            if text:
                blockers.append(text)
    return blockers[:8]


def _pending_reconcile_groups(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for review in reviews:
        key = (
            str(review.get("pair") or "UNKNOWN"),
            str(review.get("side") or "UNKNOWN"),
            str(review.get("method") or "UNKNOWN"),
        )
        item = groups.setdefault(
            key,
            {
                "pair": key[0],
                "side": key[1],
                "method": key[2],
                "count": 0,
                "order_ids": [],
                "reason_codes": {},
            },
        )
        item["count"] += 1
        if review.get("order_id"):
            item["order_ids"].append(review.get("order_id"))
        reason_codes = item["reason_codes"]
        for reason in review.get("review_reasons") or []:
            if not isinstance(reason, dict):
                continue
            code = str(reason.get("code") or "UNKNOWN")
            reason_codes[code] = int(reason_codes.get(code) or 0) + 1
    return sorted(groups.values(), key=lambda item: (-int(item["count"]), item["pair"], item["side"], item["method"]))


def _pending_order_attached_sl_risk_jpy(order: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    pair = str(order.get("pair") or "")
    entry = _pending_order_price(order)
    units = _maybe_float(order.get("units"))
    stop = _nested_order_price(order, "stopLossOnFill")
    if not pair or entry is None or units is None or units == 0:
        return {
            "risk_jpy": None,
            "stop_loss": stop,
            "issue_code": "PENDING_ATTACHED_SL_RISK_UNKNOWN",
            "message": "pending order is missing pair, entry price, or units",
        }
    if stop is None:
        return {
            "risk_jpy": None,
            "stop_loss": None,
            "issue_code": "PENDING_ATTACHED_SL_MISSING",
            "message": "pending order has no broker-attached stopLossOnFill",
        }
    spec = DEFAULT_SPECS.get(pair)
    if spec is None:
        return {
            "risk_jpy": None,
            "stop_loss": stop,
            "issue_code": "PENDING_ATTACHED_SL_RISK_UNKNOWN",
            "message": f"pending order pair {pair} is unsupported",
        }
    quote_to_jpy = _snapshot_quote_to_jpy(pair, snapshot)
    if quote_to_jpy is None:
        return {
            "risk_jpy": None,
            "stop_loss": stop,
            "issue_code": "PENDING_ATTACHED_SL_RISK_UNKNOWN",
            "message": f"missing conversion quote for pending order pair {pair}",
        }
    if units > 0:
        loss_pips = (entry - stop) * spec.pip_factor
    else:
        loss_pips = (stop - entry) * spec.pip_factor
    if loss_pips <= 0:
        return {
            "risk_jpy": None,
            "stop_loss": stop,
            "issue_code": "PENDING_ATTACHED_SL_WRONG_SIDE",
            "message": "pending order stopLossOnFill is not on the loss side of the entry",
        }
    jpy_per_pip = (abs(units) / spec.pip_factor) * quote_to_jpy
    return {
        "risk_jpy": round(loss_pips * jpy_per_pip, 4),
        "stop_loss": stop,
        "loss_pips": round(loss_pips, 4),
    }


def _snapshot_quote_to_jpy(pair: str, snapshot: dict[str, Any]) -> float | None:
    if pair.endswith("_JPY"):
        return 1.0
    quote_ccy = pair.split("_")[-1] if "_" in pair else ""
    conversions = snapshot.get("home_conversions") if isinstance(snapshot.get("home_conversions"), dict) else {}
    value = _maybe_float(conversions.get(quote_ccy))
    if value is not None and value > 0:
        return value
    quotes = snapshot.get("quotes") if isinstance(snapshot.get("quotes"), dict) else {}
    direct = quotes.get(f"{quote_ccy}_JPY")
    if isinstance(direct, dict):
        bid = _maybe_float(direct.get("bid"))
        ask = _maybe_float(direct.get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    inverse = quotes.get(f"JPY_{quote_ccy}")
    if isinstance(inverse, dict):
        bid = _maybe_float(inverse.get("bid"))
        ask = _maybe_float(inverse.get("ask"))
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return 1.0 / ((bid + ask) / 2.0)
    return None


def _pending_order_lane_id(order: dict[str, Any]) -> str | None:
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    for key in ("clientExtensions", "tradeClientExtensions"):
        extension = raw.get(key)
        if not isinstance(extension, dict):
            continue
        comment = str(extension.get("comment") or "")
        for token in comment.split():
            if token.startswith("lane="):
                lane_id = token[len("lane=") :].strip()
                return lane_id or None
    return None


def _parent_lane_id(lane_id: str) -> str:
    return lane_id[: -len(":MARKET")] if lane_id.endswith(":MARKET") else lane_id


def _method_from_lane_id(lane_id: str | None) -> str | None:
    if not lane_id:
        return None
    parts = str(lane_id).split(":")
    if len(parts) >= 4:
        return parts[3] or None
    return None


def _pending_order_method(order: dict[str, Any], parent_lane_id: str | None) -> str | None:
    method = _method_from_lane_id(parent_lane_id)
    if method:
        return method
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    for key in ("clientExtensions", "tradeClientExtensions"):
        extension = raw.get(key)
        if not isinstance(extension, dict):
            continue
        comment = str(extension.get("comment") or "")
        for token in comment.split():
            if token.startswith("method=") or token.startswith("desk="):
                value = token.split("=", 1)[1].strip().upper()
                if value:
                    return value
    return None


def _pending_order_side(order: dict[str, Any]) -> str | None:
    side = str(order.get("side") or "").upper()
    if side in {"LONG", "SHORT"}:
        return side
    units = _maybe_float(order.get("units"))
    if units is None or units == 0:
        return None
    return "LONG" if units > 0 else "SHORT"


def _pending_order_price(order: dict[str, Any]) -> float | None:
    value = _maybe_float(order.get("price"))
    if value is not None:
        return value
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    return _maybe_float(raw.get("price"))


def _pending_order_create_time(order: dict[str, Any]) -> datetime | None:
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    return _parse_utc(raw.get("createTime") or raw.get("time") or order.get("createTime") or order.get("time"))


def _nested_order_price(order: dict[str, Any], key: str) -> float | None:
    raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
    nested = raw.get(key)
    if isinstance(nested, dict):
        return _maybe_float(nested.get("price"))
    nested = order.get(key)
    if isinstance(nested, dict):
        return _maybe_float(nested.get("price"))
    return None


def _normalized_pending_order_type(value: Any) -> str:
    text = str(value or "").upper().replace("_", "-")
    if text == "LIMIT-ORDER":
        return "LIMIT"
    if text in {"STOP", "STOP-ORDER", "STOP-ENTRY-ORDER"}:
        return "STOP-ENTRY"
    if text == "MARKET-IF-TOUCHED-ORDER":
        return "MARKET-IF-TOUCHED"
    return text


def _pending_entry_lifecycle_metrics(
    db_path: Path,
    *,
    window_hours: float,
    now: datetime,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "window_hours": window_hours,
        "gateway_sent_orders": 0,
        "accepted_entry_orders": 0,
        "filled_entry_orders": 0,
        "canceled_before_fill_orders": 0,
        "canceled_before_fill_replaced_orders": 0,
        "canceled_before_fill_orphan_orders": 0,
        "open_unfilled_entry_orders": 0,
        "fill_rate": None,
        "cancel_before_fill_rate": None,
        "cancel_replacement_rate": None,
        "canceled_before_fill_samples": [],
        "canceled_before_fill_orphan_groups": [],
        "canceled_before_fill_replaced_groups": [],
        "error": None,
    }
    if not db_path.exists():
        metrics["error"] = "missing"
        return metrics
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            columns = _table_columns(conn, "execution_events")
            required = {"ts_utc", "event_type", "order_id"}
            if not required <= columns:
                metrics["error"] = "execution_events lacks pending lifecycle columns"
                return metrics
            select_columns = ["ts_utc", "event_type", "order_id"]
            for column in ("lane_id", "trade_id", "pair", "side", "units", "price", "exit_reason"):
                select_columns.append(column if column in columns else f"NULL AS {column}")
            rows = list(
                conn.execute(
                    f"""
                    SELECT {', '.join(select_columns)}
                    FROM execution_events
                    WHERE event_type IN (
                        'GATEWAY_ORDER_SENT',
                        'ORDER_ACCEPTED',
                        'ORDER_FILLED',
                        'ORDER_CANCELED'
                    )
                      AND ts_utc >= ?
                    ORDER BY ts_utc ASC
                    """,
                    (cutoff.isoformat(),),
                )
            )
    except sqlite3.Error as exc:
        metrics["error"] = str(exc)
        return metrics

    accepted: dict[str, dict[str, Any]] = {}
    filled_order_ids: set[str] = set()
    canceled: dict[str, sqlite3.Row] = {}
    for row in rows:
        ts = _parse_utc(row["ts_utc"])
        if ts is None:
            continue
        event_type = str(row["event_type"] or "").upper()
        order_id = str(row["order_id"] or "").strip()
        if event_type == "GATEWAY_ORDER_SENT":
            metrics["gateway_sent_orders"] = int(metrics["gateway_sent_orders"]) + 1
            continue
        if not order_id:
            continue
        if event_type == "ORDER_ACCEPTED":
            if not _execution_event_is_entry_order(row):
                continue
            accepted.setdefault(
                order_id,
                {
                    "accepted_ts": ts,
                    "order_id": order_id,
                    "lane_id": _row_text(row, "lane_id"),
                    "method": _method_from_lane_id(_row_text(row, "lane_id")),
                    "pair": _row_text(row, "pair"),
                    "side": _entry_order_side(row),
                    "units": _maybe_float(row["units"]),
                    "price": _maybe_float(row["price"]),
                },
            )
        elif event_type == "ORDER_FILLED":
            filled_order_ids.add(order_id)
        elif event_type == "ORDER_CANCELED":
            canceled[order_id] = row

    canceled_before_fill_samples: list[dict[str, Any]] = []
    filled_count = 0
    canceled_before_fill_count = 0
    canceled_replaced_count = 0
    canceled_orphan_count = 0
    open_unfilled_count = 0
    for order_id, order in accepted.items():
        if order_id in filled_order_ids:
            filled_count += 1
            continue
        cancel_row = canceled.get(order_id)
        if cancel_row is not None:
            canceled_before_fill_count += 1
            cancel_ts = _parse_utc(cancel_row["ts_utc"])
            age_min = (
                (cancel_ts - order["accepted_ts"]).total_seconds() / 60.0
                if cancel_ts is not None
                else None
            )
            replacement = (
                _pending_cancel_replacement(order, accepted, cancel_ts)
                if cancel_ts is not None
                else None
            )
            if replacement is not None:
                canceled_replaced_count += 1
            else:
                canceled_orphan_count += 1
            canceled_before_fill_samples.append(
                {
                    "order_id": order_id,
                    "lane_id": order.get("lane_id"),
                    "method": order.get("method"),
                    "pair": order.get("pair"),
                    "side": order.get("side"),
                    "units": order.get("units"),
                    "price": order.get("price"),
                    "accepted_at_utc": order["accepted_ts"].isoformat(),
                    "canceled_at_utc": cancel_ts.isoformat() if cancel_ts else None,
                    "age_min": round(age_min, 3) if age_min is not None else None,
                    "replaced_with_order_id": replacement["order_id"] if replacement else None,
                    "replacement_after_min": (
                        round(replacement["replacement_after_min"], 3)
                        if replacement
                        else None
                    ),
                }
            )
            continue
        open_unfilled_count += 1

    accepted_count = len(accepted)
    metrics["accepted_entry_orders"] = accepted_count
    metrics["filled_entry_orders"] = filled_count
    metrics["canceled_before_fill_orders"] = canceled_before_fill_count
    metrics["canceled_before_fill_replaced_orders"] = canceled_replaced_count
    metrics["canceled_before_fill_orphan_orders"] = canceled_orphan_count
    metrics["open_unfilled_entry_orders"] = open_unfilled_count
    metrics["fill_rate"] = (filled_count / accepted_count) if accepted_count else None
    metrics["cancel_before_fill_rate"] = (
        canceled_before_fill_count / accepted_count
        if accepted_count
        else None
    )
    metrics["cancel_replacement_rate"] = (
        canceled_replaced_count / canceled_before_fill_count
        if canceled_before_fill_count
        else None
    )
    canceled_before_fill_samples.sort(key=lambda item: str(item.get("canceled_at_utc") or ""), reverse=True)
    metrics["canceled_before_fill_samples"] = canceled_before_fill_samples[:12]
    metrics["canceled_before_fill_orphan_groups"] = _pending_lifecycle_cancel_groups(
        canceled_before_fill_samples,
        replaced=False,
    )
    metrics["canceled_before_fill_replaced_groups"] = _pending_lifecycle_cancel_groups(
        canceled_before_fill_samples,
        replaced=True,
    )
    return metrics


def _merge_pending_cancel_timing_regret(
    metrics: dict[str, Any],
    timing_payload: dict[str, Any] | None,
    *,
    now: datetime,
) -> None:
    """Attach cancel-regret context from execution-timing audit.

    `execution_timing_audit` is the measured feedback loop for pending orders
    that were canceled before fill. Linking it here keeps self-improvement from
    treating all client-request cancels as identical churn when recent post-
    cancel tape shows entry or TP would have touched.
    """

    if not isinstance(timing_payload, dict):
        return
    generated = _parse_utc(timing_payload.get("generated_at_utc"))
    age_hours: float | None = None
    if generated is not None:
        age_hours = (now - generated).total_seconds() / 3600.0
        if age_hours < 0:
            age_hours = 0.0
        if age_hours > PENDING_CANCEL_REGRET_MAX_AGE_HOURS:
            return
    summary = timing_payload.get("summary") if isinstance(timing_payload.get("summary"), dict) else {}
    rows = [row for row in timing_payload.get("canceled_order_regrets", []) or [] if isinstance(row, dict)]
    shape_rollup = (
        timing_payload.get("canceled_order_regret_by_shape")
        if isinstance(timing_payload.get("canceled_order_regret_by_shape"), dict)
        else {}
    )
    regretted_rows = [
        row
        for row in rows
        if row.get("entry_touched_after_cancel")
        and not row.get("sl_touched_after_cancel")
        and (row.get("tp_touched_after_cancel") or (_maybe_float(row.get("mfe_pips_after_cancel_entry")) or 0.0) > 0)
    ]
    regretted_rows.sort(key=lambda row: float(row.get("estimated_missed_mfe_jpy") or 0.0), reverse=True)
    metrics["timing_regret"] = {
        "generated_at_utc": generated.isoformat() if generated else timing_payload.get("generated_at_utc"),
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "canceled_orders_audited": int(summary.get("canceled_orders_audited") or len(rows)),
        "canceled_entry_touched_after_cancel": int(summary.get("canceled_entry_touched_after_cancel") or 0),
        "canceled_entry_touched_after_cancel_rate": _maybe_float(
            summary.get("canceled_entry_touched_after_cancel_rate")
        ),
        "canceled_positive_after_cancel_entry": int(summary.get("canceled_positive_after_cancel_entry") or 0),
        "canceled_tp_touched_after_cancel": int(summary.get("canceled_tp_touched_after_cancel") or 0),
        "canceled_estimated_missed_mfe_jpy": _maybe_float(summary.get("canceled_estimated_missed_mfe_jpy")),
        "top_regretted_cancels": [
            {
                "order_id": str(row.get("order_id") or ""),
                "lane_id": str(row.get("lane_id") or ""),
                "pair": str(row.get("pair") or ""),
                "side": str(row.get("side") or ""),
                "entry_touch_after_cancel_minutes": _maybe_float(row.get("entry_touch_after_cancel_minutes")),
                "tp_touched_after_cancel": bool(row.get("tp_touched_after_cancel")),
                "mfe_pips_after_cancel_entry": _maybe_float(row.get("mfe_pips_after_cancel_entry")),
                "estimated_missed_mfe_jpy": _maybe_float(row.get("estimated_missed_mfe_jpy")),
            }
            for row in regretted_rows[:8]
        ],
        "top_regretted_shapes": _top_regretted_cancel_shapes(shape_rollup),
    }


def _top_regretted_cancel_shapes(shape_rollup: dict[str, Any]) -> list[dict[str, Any]]:
    items = shape_rollup.get("items") if isinstance(shape_rollup.get("items"), list) else []
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        priority = str(item.get("priority_class") or "")
        missed_mfe = _maybe_float(item.get("estimated_missed_mfe_jpy"))
        if priority == "LOW_CANCEL_REGRET" and not missed_mfe:
            continue
        out.append(
            {
                "evidence_ref": str(item.get("evidence_ref") or ""),
                "pair": str(item.get("pair") or ""),
                "side": str(item.get("side") or ""),
                "method": str(item.get("method") or ""),
                "order_type": str(item.get("order_type") or ""),
                "priority_class": priority,
                "orders": int(item.get("orders") or 0),
                "entry_touched_after_cancel": int(item.get("entry_touched_after_cancel") or 0),
                "entry_touch_after_cancel_rate": _maybe_float(item.get("entry_touch_after_cancel_rate")),
                "positive_after_cancel_entry": int(item.get("positive_after_cancel_entry") or 0),
                "positive_after_cancel_entry_rate": _maybe_float(item.get("positive_after_cancel_entry_rate")),
                "tp_touched_after_cancel": int(item.get("tp_touched_after_cancel") or 0),
                "tp_touched_after_cancel_rate": _maybe_float(item.get("tp_touched_after_cancel_rate")),
                "estimated_missed_mfe_jpy": missed_mfe,
                "next_action": str(item.get("next_action") or ""),
            }
        )
    priority_order = {
        "PRESERVE_PENDING_THESIS_TP_TOUCHED": 0,
        "REPRICE_OR_EXTEND_TTL_ENTRY_TOUCHED": 1,
        "ENTRY_TOUCHED_NO_POSITIVE_MFE": 2,
        "LOW_CANCEL_REGRET": 3,
    }
    out.sort(
        key=lambda item: (
            priority_order.get(str(item.get("priority_class") or ""), 99),
            -float(item.get("estimated_missed_mfe_jpy") or 0.0),
            -int(item.get("orders") or 0),
            str(item.get("evidence_ref") or ""),
        )
    )
    return out[:8]


def _pending_cancel_replacement(
    canceled_order: dict[str, Any],
    accepted: dict[str, dict[str, Any]],
    cancel_ts: datetime,
) -> dict[str, Any] | None:
    window = timedelta(minutes=PENDING_ENTRY_CANCEL_REPLACEMENT_WINDOW_MIN)
    lane_id = str(canceled_order.get("lane_id") or "").strip()
    pair = str(canceled_order.get("pair") or "").strip()
    side = str(canceled_order.get("side") or "").strip()
    candidates: list[dict[str, Any]] = []
    for candidate in accepted.values():
        if candidate.get("order_id") == canceled_order.get("order_id"):
            continue
        accepted_ts = candidate.get("accepted_ts")
        if not isinstance(accepted_ts, datetime):
            continue
        delta = accepted_ts - cancel_ts
        if delta <= timedelta(0) or delta > window:
            continue
        candidate_lane_id = str(candidate.get("lane_id") or "").strip()
        if lane_id:
            if candidate_lane_id != lane_id:
                continue
        else:
            if pair and str(candidate.get("pair") or "").strip() != pair:
                continue
            if side and str(candidate.get("side") or "").strip() != side:
                continue
        candidates.append(
            {
                "order_id": candidate.get("order_id"),
                "replacement_after_min": delta.total_seconds() / 60.0,
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["replacement_after_min"]))


def _pending_lifecycle_cancel_groups(samples: list[dict[str, Any]], *, replaced: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for sample in samples:
        has_replacement = bool(sample.get("replaced_with_order_id"))
        if has_replacement is not replaced:
            continue
        key = (
            str(sample.get("pair") or "UNKNOWN"),
            str(sample.get("side") or "UNKNOWN"),
            str(sample.get("method") or "UNKNOWN"),
        )
        item = grouped.setdefault(
            key,
            {"pair": key[0], "side": key[1], "method": key[2], "count": 0, "order_ids": []},
        )
        item["count"] += 1
        if sample.get("order_id"):
            item["order_ids"].append(sample.get("order_id"))
    return sorted(grouped.values(), key=lambda item: (-int(item["count"]), item["pair"], item["side"], item["method"]))


def _execution_event_is_entry_order(row: sqlite3.Row) -> bool:
    exit_reason = str(_row_text(row, "exit_reason") or "").upper()
    if "TRADE_CLOSE" in exit_reason or "POSITION_CLOSE" in exit_reason:
        return False
    units = _maybe_float(row["units"]) if "units" in row.keys() else None
    if units == 0:
        return False
    return True


def _entry_order_side(row: sqlite3.Row) -> str | None:
    side = str(_row_text(row, "side") or "").upper()
    if side in {"LONG", "SHORT"}:
        return side
    units = _maybe_float(row["units"]) if "units" in row.keys() else None
    if units is None or units == 0:
        return None
    return "LONG" if units > 0 else "SHORT"


def _market_close_attribution_findings(
    *,
    run_id: str,
    db_path: Path,
    window_hours: float,
    now: datetime,
) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    rows: list[sqlite3.Row] = []
    accepted_close_rows: list[sqlite3.Row] = []
    stale_close_satisfied_trade_ids: set[str] = set()
    error: str | None = None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            columns = _table_columns(conn, "execution_events")
            required = {
                "event_uid",
                "ts_utc",
                "event_type",
                "lane_id",
                "order_id",
                "trade_id",
                "pair",
                "side",
                "units",
                "realized_pl_jpy",
                "exit_reason",
            }
            if not required <= columns:
                return []
            rows = list(
                conn.execute(
                    """
                    WITH gateway_entries AS (
                        SELECT
                            trade_id,
                            order_id,
                            lane_id
                        FROM execution_events
                        WHERE event_type = 'GATEWAY_ORDER_SENT'
                          AND lane_id IS NOT NULL
                          AND lane_id != ''
                    ),
                    entries AS (
                        SELECT
                            e.trade_id,
                            COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS gateway_lane_id,
                            CASE
                                WHEN MAX(e.units) > 0 THEN 'LONG'
                                WHEN MIN(e.units) < 0 THEN 'SHORT'
                                ELSE NULL
                            END AS position_side
                        FROM execution_events e
                        LEFT JOIN gateway_entries g
                          ON (
                            g.trade_id IS NOT NULL
                            AND g.trade_id != ''
                            AND g.trade_id = e.trade_id
                          )
                          OR (
                            g.order_id IS NOT NULL
                            AND g.order_id != ''
                            AND g.order_id = e.order_id
                          )
                        WHERE e.event_type = 'ORDER_FILLED'
                          AND e.trade_id IS NOT NULL
                          AND e.trade_id != ''
                          AND e.units IS NOT NULL
                        GROUP BY e.trade_id
                        HAVING gateway_lane_id IS NOT NULL
                           AND gateway_lane_id != ''
                    )
                    SELECT
                        e.event_uid,
                        e.ts_utc,
                        e.event_type,
                        e.trade_id,
                        e.order_id,
                        e.pair,
                        e.side,
                        e.realized_pl_jpy,
                        e.exit_reason,
                        entries.gateway_lane_id,
                        entries.position_side,
                        CASE
                            WHEN EXISTS (
                                SELECT 1
                                FROM execution_events c
                                WHERE c.event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
                                  AND c.trade_id = e.trade_id
                            )
                            THEN 1
                            ELSE 0
                        END AS gpt_close_accepted
                    FROM execution_events e
                    INNER JOIN entries ON entries.trade_id = e.trade_id
                    WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                      AND e.exit_reason = 'MARKET_ORDER_TRADE_CLOSE'
                      AND e.realized_pl_jpy IS NOT NULL
                      AND e.trade_id IS NOT NULL
                      AND e.trade_id != ''
                      AND NOT EXISTS (
                          SELECT 1
                          FROM execution_events c
                          WHERE c.event_type IN (
                              'GATEWAY_TRADE_CLOSE_SENT',
                              'GATEWAY_TRADE_CLOSE_RECONCILED'
                          )
                            AND c.trade_id = e.trade_id
                      )
                    ORDER BY e.ts_utc DESC, e.event_uid DESC
                    """
                )
            )
            accepted_close_rows = _broker_trade_close_accept_rows(conn, columns)
            stale_close_satisfied_trade_ids = _stale_gpt_close_satisfied_trade_ids(conn, columns)
    except sqlite3.Error as exc:
        error = str(exc)
    if error is not None:
        return []
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    unattributed: list[sqlite3.Row] = []
    gpt_accepted_without_position_gateway: list[sqlite3.Row] = []
    broker_accept_count = 0
    broker_accept_source_counts: dict[str, int] = {}
    broker_accept_sources_by_trade: dict[str, list[str]] = {}
    gpt_accept_broker_accept_count = 0
    gpt_accept_broker_source_counts: dict[str, int] = {}
    gpt_accept_broker_sources_by_trade: dict[str, list[str]] = {}
    stale_close_satisfied: list[sqlite3.Row] = []
    stale_close_broker_accept_count = 0
    stale_close_broker_source_counts: dict[str, int] = {}
    stale_close_broker_sources_by_trade: dict[str, list[str]] = {}
    for row in rows:
        ts = _parse_utc(row["ts_utc"])
        if ts is None or ts < cutoff:
            continue
        value = _maybe_float(row["realized_pl_jpy"])
        if value is None or value >= 0:
            continue
        trade_id = str(row["trade_id"] or "").strip()
        order_id = str(row["order_id"] or "").strip()
        sources = _broker_trade_close_accept_sources(
            accepted_close_rows,
            trade_id=trade_id,
            order_id=order_id,
        )
        if int(row["gpt_close_accepted"] or 0):
            if sources:
                sources = set(sources)
                if "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE" in sources:
                    sources.discard("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE")
                sources.add("GATEWAY_GPT_CLOSE_ACCEPTED")
            gpt_accepted_without_position_gateway.append(row)
            if sources:
                gpt_accept_broker_accept_count += 1
                gpt_accept_broker_sources_by_trade[trade_id] = sorted(sources)
                for source in sources:
                    gpt_accept_broker_source_counts[source] = gpt_accept_broker_source_counts.get(source, 0) + 1
            continue
        if trade_id and trade_id in stale_close_satisfied_trade_ids:
            stale_close_satisfied.append(row)
            if sources:
                stale_close_broker_accept_count += 1
                stale_close_broker_sources_by_trade[trade_id] = sorted(sources)
                for source in sources:
                    stale_close_broker_source_counts[source] = (
                        stale_close_broker_source_counts.get(source, 0) + 1
                    )
            continue
        unattributed.append(row)
        if sources:
            broker_accept_count += 1
            broker_accept_sources_by_trade[trade_id] = sorted(sources)
            for source in sources:
                broker_accept_source_counts[source] = broker_accept_source_counts.get(source, 0) + 1
    findings: list[dict[str, Any]] = []
    if gpt_accepted_without_position_gateway:
        gpt_net = sum(float(_maybe_float(row["realized_pl_jpy"]) or 0.0) for row in gpt_accepted_without_position_gateway)
        findings.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="execution_ledger",
                code="ACCEPTED_GPT_CLOSE_WITHOUT_POSITION_GATEWAY_RECEIPT",
                message=(
                    f"{len(gpt_accepted_without_position_gateway)} negative bot-attributed market close event(s) "
                    "have an accepted GPT close receipt but lack a matching position-gateway close-sent receipt"
                ),
                next_action=(
                    "Trace why accepted GPT CLOSE receipts did not persist a matching position_execution "
                    "GATEWAY_TRADE_CLOSE_SENT receipt, then keep these separate from direct/manual closes when "
                    "tuning CLOSE Gate A/B."
                ),
                evidence={
                    "window_hours": window_hours,
                    "accepted_gpt_close_without_position_gateway_count": len(gpt_accepted_without_position_gateway),
                    "accepted_gpt_close_without_position_gateway_net_jpy": round(gpt_net, 4),
                    "broker_trade_close_accept_count": gpt_accept_broker_accept_count,
                    "broker_trade_close_accept_source_counts": gpt_accept_broker_source_counts,
                    "examples": [
                        {
                            "ts_utc": str(row["ts_utc"] or ""),
                            "event_uid": str(row["event_uid"] or ""),
                            "event_type": str(row["event_type"] or ""),
                            "trade_id": str(row["trade_id"] or ""),
                            "close_order_id": str(row["order_id"] or ""),
                            "pair": str(row["pair"] or ""),
                            "side": str(row["side"] or row["position_side"] or ""),
                            "original_side": str(row["position_side"] or ""),
                            "gateway_lane_id": str(row["gateway_lane_id"] or ""),
                            "realized_pl_jpy": _maybe_float(row["realized_pl_jpy"]),
                            "exit_reason": str(row["exit_reason"] or ""),
                            "broker_trade_close_accept_sources": gpt_accept_broker_sources_by_trade.get(
                                str(row["trade_id"] or "").strip(),
                                [],
                            ),
                        }
                        for row in gpt_accepted_without_position_gateway[:8]
                    ],
                },
            )
        )
    if stale_close_satisfied:
        stale_net = sum(float(_maybe_float(row["realized_pl_jpy"]) or 0.0) for row in stale_close_satisfied)
        findings.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="execution_ledger",
                code="STALE_GPT_CLOSE_SATISFIED_AFTER_BROKER_CLOSE",
                message=(
                    f"{len(stale_close_satisfied)} negative bot-attributed market close event(s) "
                    "were later recorded as STALE_CLOSE_ALREADY_ABSENT GPT_CLOSE receipts"
                ),
                next_action=(
                    "Keep these separate from sent gateway closes: broker truth already removed the trade before "
                    "PositionProtectionGateway could send, so tune stale decision/receipt freshness without treating "
                    "the broker close as a verified Gate A/B close."
                ),
                evidence={
                    "window_hours": window_hours,
                    "stale_gpt_close_satisfied_count": len(stale_close_satisfied),
                    "stale_gpt_close_satisfied_net_jpy": round(stale_net, 4),
                    "broker_trade_close_accept_count": stale_close_broker_accept_count,
                    "broker_trade_close_accept_source_counts": stale_close_broker_source_counts,
                    "examples": [
                        {
                            "ts_utc": str(row["ts_utc"] or ""),
                            "event_uid": str(row["event_uid"] or ""),
                            "event_type": str(row["event_type"] or ""),
                            "trade_id": str(row["trade_id"] or ""),
                            "close_order_id": str(row["order_id"] or ""),
                            "pair": str(row["pair"] or ""),
                            "side": str(row["side"] or row["position_side"] or ""),
                            "original_side": str(row["position_side"] or ""),
                            "gateway_lane_id": str(row["gateway_lane_id"] or ""),
                            "realized_pl_jpy": _maybe_float(row["realized_pl_jpy"]),
                            "exit_reason": str(row["exit_reason"] or ""),
                            "broker_trade_close_accept_sources": stale_close_broker_sources_by_trade.get(
                                str(row["trade_id"] or "").strip(),
                                [],
                            ),
                        }
                        for row in stale_close_satisfied[:8]
                    ],
                },
            )
        )
    if not unattributed:
        return findings
    net = sum(float(_maybe_float(row["realized_pl_jpy"]) or 0.0) for row in unattributed)
    examples = [
        {
            "ts_utc": str(row["ts_utc"] or ""),
            "event_uid": str(row["event_uid"] or ""),
            "event_type": str(row["event_type"] or ""),
            "trade_id": str(row["trade_id"] or ""),
            "close_order_id": str(row["order_id"] or ""),
            "pair": str(row["pair"] or ""),
            "side": str(row["side"] or row["position_side"] or ""),
            "original_side": str(row["position_side"] or ""),
            "gateway_lane_id": str(row["gateway_lane_id"] or ""),
            "realized_pl_jpy": _maybe_float(row["realized_pl_jpy"]),
            "exit_reason": str(row["exit_reason"] or ""),
            "broker_trade_close_accept_sources": broker_accept_sources_by_trade.get(
                str(row["trade_id"] or "").strip(),
                [],
            ),
        }
        for row in unattributed[:8]
    ]
    if broker_accept_count:
        next_action = (
            "Persist local position-execution receipts or client-extension source tags for broker-accepted "
            "TRADE_CLOSE orders, then separate GPT/Gate A-B closes from operator/manual closes before tuning exits."
        )
    else:
        next_action = (
            "Reconcile broker close provenance for these market-close outcomes, then persist every direct close_trade "
            "path as a position-execution receipt before tuning exits."
        )
    findings.append(
        _finding(
            run_id=run_id,
            priority="P1",
            layer="execution_ledger",
            code="UNATTRIBUTED_MARKET_ORDER_CLOSES",
            message=(
                f"{len(unattributed)} negative bot-attributed market close event(s) lack "
                "a matching gateway close receipt"
            ),
            next_action=next_action,
            evidence={
                "window_hours": window_hours,
                "unattributed_loss_count": len(unattributed),
                "unattributed_loss_net_jpy": round(net, 4),
                "broker_trade_close_accept_count": broker_accept_count,
                "broker_trade_close_accept_source_counts": broker_accept_source_counts,
                "examples": examples,
            },
        )
    )
    return findings


def _broker_trade_close_accept_rows(conn: sqlite3.Connection, columns: set[str]) -> list[dict[str, Any]]:
    select_fields = []
    for column in ("trade_id", "order_id", "lane_id", "exit_reason", "raw_json"):
        if column in columns:
            select_fields.append(column)
        else:
            select_fields.append(f"NULL AS {column}")
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT {', '.join(select_fields)}
            FROM execution_events
            WHERE event_type = 'ORDER_ACCEPTED'
            """
        ).fetchall()
    ]
    if "trade_id" not in columns:
        return rows
    entry_sources = _entry_close_source_rows(conn, columns)
    for row in rows:
        raw = _json_payload(row.get("raw_json"))
        trade_close = raw.get("tradeClose") if isinstance(raw.get("tradeClose"), dict) else {}
        normalized_trade_id = str(row.get("trade_id") or "").strip() or _trade_close_trade_id(trade_close)
        if not normalized_trade_id:
            continue
        row["trade_id"] = normalized_trade_id
        entry_source = entry_sources.get(normalized_trade_id)
        if not entry_source:
            continue
        row["entry_lane_id"] = entry_source.get("lane_id")
        row["entry_raw_json"] = entry_source.get("raw_json")
    return rows


def _entry_close_source_rows(conn: sqlite3.Connection, columns: set[str]) -> dict[str, dict[str, Any]]:
    select_fields = [
        "e.trade_id AS trade_id",
        "COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS lane_id",
    ]
    raw_select = "MAX(e.raw_json) AS raw_json" if "raw_json" in columns else "NULL AS raw_json"
    select_fields.append(raw_select)
    gateway_lane_select = "lane_id" if "lane_id" in columns else "NULL AS lane_id"
    gateway_order_select = "order_id" if "order_id" in columns else "NULL AS order_id"
    gateway_trade_select = "trade_id" if "trade_id" in columns else "NULL AS trade_id"
    fill_order_select = "e.order_id" if "order_id" in columns else "NULL"
    rows = list(
        conn.execute(
            f"""
            WITH gateway_entries AS (
                SELECT {gateway_trade_select}, {gateway_order_select}, {gateway_lane_select}
                FROM execution_events
                WHERE event_type = 'GATEWAY_ORDER_SENT'
                  AND COALESCE(lane_id, '') != ''
            )
            SELECT {', '.join(select_fields)}
            FROM execution_events e
            LEFT JOIN gateway_entries g
              ON (
                COALESCE(g.trade_id, '') != ''
                AND g.trade_id = e.trade_id
              )
              OR (
                COALESCE(g.order_id, '') != ''
                AND g.order_id = {fill_order_select}
              )
            WHERE e.event_type = 'ORDER_FILLED'
              AND COALESCE(e.trade_id, '') != ''
            GROUP BY e.trade_id
            ORDER BY MIN(e.ts_utc) ASC
            """
        ).fetchall()
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        trade_id = str(row["trade_id"] or "").strip()
        if not trade_id:
            continue
        out[trade_id] = {
            "lane_id": str(row["lane_id"] or "").strip() or None,
            "raw_json": row["raw_json"],
        }
    return out

def _broker_trade_close_accept_sources(
    rows: list[dict[str, Any]],
    *,
    trade_id: str,
    order_id: str,
) -> set[str]:
    sources: set[str] = set()
    for row in rows:
        raw = _json_payload(row["raw_json"])
        raw_reason = str(raw.get("reason") or "").strip().upper()
        reason = str(row["exit_reason"] or raw_reason or "").strip().upper()
        trade_close = raw.get("tradeClose") if isinstance(raw.get("tradeClose"), dict) else {}
        close_trade_id = _trade_close_trade_id(trade_close)
        row_trade_id = str(row["trade_id"] or "").strip()
        row_order_id = str(row["order_id"] or "").strip()
        if reason != "TRADE_CLOSE" and not close_trade_id:
            continue
        if not (
            (trade_id and trade_id in {row_trade_id, close_trade_id})
            or (order_id and order_id == row_order_id)
        ):
            continue
        sources.add(_broker_trade_close_accept_source(row, raw))
    return sources


def _broker_trade_close_accept_source(row: dict[str, Any], raw: dict[str, Any]) -> str:
    lane_id = str(row.get("lane_id") or "").strip()
    if lane_id:
        return "LOCAL_LEDGER_LANE_ID"
    entry_lane_id = str(row.get("entry_lane_id") or "").strip()
    if entry_lane_id:
        return "TRADER_ENTRY_LANE_ID"
    client_sources = []
    for key in ("clientExtensions", "tradeClientExtensions"):
        value = raw.get(key)
        if isinstance(value, dict):
            client_sources.append(value)
    haystack = json.dumps(client_sources, ensure_ascii=False, sort_keys=True).lower()
    if "qrv1" in haystack or "qr-vnext" in haystack or '"tag": "trader"' in haystack:
        return "TRADER_CLIENT_EXTENSION"
    entry_raw = _json_payload(row.get("entry_raw_json"))
    if entry_raw:
        entry_haystack = json.dumps(entry_raw, ensure_ascii=False, sort_keys=True).lower()
        if "qrv1" in entry_haystack or "qr-vnext" in entry_haystack or '"tag": "trader"' in entry_haystack:
            return "TRADER_ENTRY_CLIENT_EXTENSION"
    if client_sources:
        return "NON_TRADER_CLIENT_EXTENSION"
    return "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE"


def _trade_close_trade_id(trade_close: Any) -> str:
    if not isinstance(trade_close, dict):
        return ""
    for key in ("tradeID", "trade_id", "id"):
        value = trade_close.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _json_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _forecast_history_findings(
    *,
    run_id: str,
    path: Path,
    target_open: bool,
) -> list[dict[str, Any]]:
    rows, malformed, error = _read_jsonl(path)
    out: list[dict[str, Any]] = []
    if error is not None:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1" if target_open else "P2",
                layer="forecast",
                code="FORECAST_HISTORY_UNREADABLE",
                message=f"forecast history is unreadable: {path}: {error}",
                next_action="Repair forecast_history.jsonl before using forecast before/after metrics or forecast-derived live confidence.",
            )
        )
        return out
    if malformed:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="FORECAST_HISTORY_MALFORMED_ROWS",
                message=f"forecast history has {malformed} malformed row(s)",
                next_action="Repair malformed forecast history rows so forecast quality is measured on a stable sample.",
                evidence={"malformed_rows": malformed},
            )
        )
    if target_open and not rows:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="FORECAST_HISTORY_EMPTY",
                message="target is open but forecast_history has no rows",
                next_action="Record forecast_history before generating live candidates; otherwise prediction accuracy cannot be learned.",
            )
        )
        return out

    cycle_pair_counts: dict[tuple[str, str], int] = {}
    no_cycle_clusters: dict[tuple[str, str, str], int] = {}
    direction_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    with_cycle = 0
    no_cycle = 0
    for row in rows:
        pair = str(row.get("pair") or "UNKNOWN")
        direction = str(row.get("direction") or "UNKNOWN").upper()
        cycle_id = str(row.get("cycle_id") or "").strip()
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if cycle_id:
            with_cycle += 1
            key = (cycle_id, pair)
            cycle_pair_counts[key] = cycle_pair_counts.get(key, 0) + 1
        else:
            no_cycle += 1
            second = str(row.get("timestamp_utc") or "").split(".", maxsplit=1)[0]
            key = (pair, second, direction)
            no_cycle_clusters[key] = no_cycle_clusters.get(key, 0) + 1

    duplicate_cycle_pairs = [
        {"cycle_id": cycle_id, "pair": pair, "count": count}
        for (cycle_id, pair), count in cycle_pair_counts.items()
        if count > 1
    ]
    duplicate_cycle_pairs.sort(key=lambda item: (-int(item["count"]), str(item["pair"]), str(item["cycle_id"])))
    if duplicate_cycle_pairs:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR",
                message=(
                    f"forecast_history has {len(duplicate_cycle_pairs)} duplicate cycle_id/pair group(s); "
                    "forecast quality and projection learning can be double-counted"
                ),
                next_action=(
                    "Fix forecast recording/dedupe so one cycle contributes one forecast row per pair before "
                    "treating before/after forecast improvement as reliable."
                ),
                evidence={
                    "rows": len(rows),
                    "with_cycle": with_cycle,
                    "no_cycle": no_cycle,
                    "duplicate_cycle_pair_groups": len(duplicate_cycle_pairs),
                    "examples": duplicate_cycle_pairs[:8],
                    "direction_counts": dict(sorted(direction_counts.items())),
                    "top_pairs": _top_count_items(pair_counts, limit=8),
                },
            )
        )

    phantom_clusters = [
        {"pair": pair, "timestamp_second": second, "direction": direction, "count": count}
        for (pair, second, direction), count in no_cycle_clusters.items()
        if count >= 3
    ]
    phantom_clusters.sort(key=lambda item: (-int(item["count"]), str(item["pair"]), str(item["timestamp_second"])))
    if phantom_clusters:
        out.append(
            _finding(
                run_id=run_id,
                priority="P2",
                layer="forecast",
                code="FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS",
                message=(
                    f"forecast_history has {len(phantom_clusters)} old no-cycle same-second cluster(s); "
                    "legacy forecast evaluation must dedupe them"
                ),
                next_action=(
                    "Keep legacy no-cycle rows deduped by pair/second/direction/confidence/target/invalidation "
                    "when measuring forecast improvement."
                ),
                evidence={
                    "rows": len(rows),
                    "no_cycle": no_cycle,
                    "phantom_clusters": len(phantom_clusters),
                    "examples": phantom_clusters[:8],
                },
            )
        )
    return out


def _projection_findings(
    *,
    run_id: str,
    path: Path,
    now: datetime,
    target_open: bool,
    active_positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows, malformed, error = _read_jsonl(path)
    if error is not None:
        priority = "P0" if target_open or active_positions else "P1"
        return [
            _finding(
                run_id=run_id,
                priority=priority,
                layer="forecast",
                code="PROJECTION_LEDGER_UNREADABLE",
                message=f"projection ledger is unreadable: {path}: {error}",
                next_action="Repair projection_ledger.jsonl before forecast-dependent live entries.",
            )
        ]
    expired = [row for row in rows if str(row.get("resolution_status") or "PENDING").upper() == "PENDING" and _projection_expired(row, now=now)]
    out: list[dict[str, Any]] = []
    if malformed:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="PROJECTION_LEDGER_MALFORMED_ROWS",
                message=f"projection ledger has {malformed} malformed row(s)",
                next_action="Repair malformed projection rows so forecasts can be scored.",
                evidence={"malformed_rows": malformed},
            )
        )
    if expired:
        out.append(
            _finding(
                run_id=run_id,
                priority="P0",
                layer="forecast",
                code="PROJECTION_LEDGER_EXPIRED_PENDING",
                message=f"projection ledger has {len(expired)} expired PENDING projection(s)",
                next_action="Run verify-projections and learn from HIT/MISS/TIMEOUT before new risk.",
                evidence={"examples": [_projection_ref(row) for row in expired[:8]]},
            )
        )
    out.extend(_directional_forecast_quality_findings(run_id=run_id, rows=rows, now=now))
    out.extend(_projection_economic_precision_findings(run_id=run_id, path=path))
    return out


def _projection_economic_precision_findings(*, run_id: str, path: Path) -> list[dict[str, Any]]:
    try:
        from quant_rabbit.strategy.projection_ledger import compute_hit_rates

        hit_rates = compute_hit_rates(path.parent)
    except Exception:
        return []
    filtered_hit_rates = {
        signal_name: buckets
        for signal_name, buckets in (hit_rates or {}).items()
        if not str(signal_name or "").startswith("directional_forecast")
    }
    weak_buckets = projection_precision_gap_summary(
        filtered_hit_rates,
        min_wilson_lower=PROJECTION_ECONOMIC_PRECISION_MIN_WILSON_LOWER,
        min_samples=PROJECTION_ECONOMIC_PRECISION_MIN_SAMPLES,
        limit=PROJECTION_ECONOMIC_PRECISION_GAP_LIMIT,
    )
    if not weak_buckets:
        return []
    usable_edges = projection_precision_edge_summary(
        filtered_hit_rates,
        min_wilson_lower=PROJECTION_ECONOMIC_PRECISION_MIN_WILSON_LOWER,
        min_samples=PROJECTION_ECONOMIC_PRECISION_MIN_SAMPLES,
        limit=PROJECTION_ECONOMIC_PRECISION_EDGE_LIMIT,
    )
    return [
        _finding(
            run_id=run_id,
            priority="P1",
            layer="forecast",
            code="PROJECTION_ECONOMIC_PRECISION_WEAK",
            message=(
                f"{len(weak_buckets)} projection bucket(s) clear headline Wilson "
                f"{PROJECTION_ECONOMIC_PRECISION_MIN_WILSON_LOWER:.0%} precision but fail "
                "economic precision after TIMEOUT/no-touch penalties"
            ),
            next_action=(
                "Do not use the named projection buckets as 90% high-turn live support. "
                "Mine pair/direction/regime variants or tighten target/horizon geometry until "
                "economic_hit_rate Wilson clears the same live precision floor."
            ),
            evidence={
                "min_wilson_lower": PROJECTION_ECONOMIC_PRECISION_MIN_WILSON_LOWER,
                "min_samples": PROJECTION_ECONOMIC_PRECISION_MIN_SAMPLES,
                "weak_buckets": weak_buckets,
                "usable_edges": usable_edges,
            },
        )
    ]


def _directional_forecast_quality_findings(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    directional = [
        row for row in rows
        if str(row.get("signal_name") or "").strip() == "directional_forecast"
    ]
    if len(directional) < FORECAST_CALIBRATION_MIN_SAMPLES:
        return []
    status_counts: dict[str, int] = {}
    calibrated: list[dict[str, Any]] = []
    target_timeout_rows: list[dict[str, Any]] = []
    for row in directional:
        status = str(row.get("resolution_status") or "PENDING").upper()
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"HIT", "MISS"} and _directional_forecast_has_target_invalidation(row):
            if _directional_forecast_target_timeout_like(row):
                target_timeout_rows.append(row)
                continue
            calibrated.append(row)
    if not calibrated:
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_CALIBRATION_UNRESOLVED",
                message=(
                    f"directional_forecast has {len(directional)} row(s) but no HIT/MISS "
                    "target/invalidation calibration samples"
                ),
                next_action=(
                    "Run verify-projections with candle truth soon after each forecast window and keep "
                    "target/invalidation geometry populated so final direction calls can be learned from."
                ),
                evidence={
                    "rows": len(directional),
                    "status_counts": status_counts,
                    "target_timeout_samples": len(target_timeout_rows),
                    "min_samples": FORECAST_CALIBRATION_MIN_SAMPLES,
                    "examples": [_projection_ref(row) for row in directional[:8]],
                },
            )
        ]
    movement_calibrated = [
        row for row in calibrated
        if _directional_forecast_is_movement_direction(row)
    ]
    range_calibrated = [
        row for row in calibrated
        if str(row.get("direction") or "").upper() == "RANGE"
    ]
    findings: list[dict[str, Any]] = []
    calibration_coverage = len(calibrated) / len(directional)
    target_timeout_count = len(target_timeout_rows)
    timeout_count = int(status_counts.get("TIMEOUT") or 0) + target_timeout_count
    hit_miss_count = (
        int(status_counts.get("HIT") or 0)
        + int(status_counts.get("MISS") or 0)
        - target_timeout_count
    )
    missing_geometry_count = max(0, hit_miss_count - len(calibrated))
    recent_24h_directional = _directional_forecast_rows_since(directional, now=now, window=timedelta(hours=24))
    recent_24h_calibrated = _directional_forecast_rows_since(calibrated, now=now, window=timedelta(hours=24))
    recent_7d_directional = _directional_forecast_rows_since(directional, now=now, window=timedelta(days=7))
    recent_7d_calibrated = _directional_forecast_rows_since(calibrated, now=now, window=timedelta(days=7))
    recent_24h_coverage = (
        len(recent_24h_calibrated) / len(recent_24h_directional)
        if recent_24h_directional
        else 0.0
    )
    recent_7d_coverage = (
        len(recent_7d_calibrated) / len(recent_7d_directional)
        if recent_7d_directional
        else 0.0
    )
    recent_geometry_recovered = (
        len(recent_24h_directional) >= FORECAST_CALIBRATION_MIN_SAMPLES
        and recent_24h_coverage >= FORECAST_CALIBRATION_MIN_COVERAGE
    ) or (
        len(recent_7d_directional) >= FORECAST_CALIBRATION_MIN_SAMPLES
        and recent_7d_coverage >= FORECAST_CALIBRATION_MIN_COVERAGE
    )
    if calibration_coverage < FORECAST_CALIBRATION_MIN_COVERAGE:
        if timeout_count >= FORECAST_CALIBRATION_MIN_SAMPLES and timeout_count >= missing_geometry_count:
            findings.append(
                _finding(
                    run_id=run_id,
                    priority="P1",
                    layer="forecast",
                    code="DIRECTIONAL_FORECAST_CALIBRATION_TIMEOUT_DOMINANT",
                    message=(
                        f"directional_forecast has only {len(calibrated)}/{len(directional)} "
                        "HIT/MISS target/invalidation calibration samples; TIMEOUT dominates"
                    ),
                    next_action=(
                        "Verify projection windows with candle truth quickly enough to resolve HIT/MISS, "
                        "then recalibrate directional confidence before using forecast strength to expand entries."
                    ),
                    evidence={
                        "rows": len(directional),
                        "calibrated_samples": len(calibrated),
                        "calibration_coverage": round(calibration_coverage, 4),
                        "min_coverage": FORECAST_CALIBRATION_MIN_COVERAGE,
                        "missing_geometry_samples": missing_geometry_count,
                        "target_timeout_samples": target_timeout_count,
                        "status_counts": status_counts,
                        "examples": [_projection_ref(row) for row in directional[:8]],
                    },
                )
            )
        elif missing_geometry_count >= FORECAST_CALIBRATION_MIN_SAMPLES:
            findings.append(
                _finding(
                    run_id=run_id,
                    priority="P2" if recent_geometry_recovered else "P1",
                    layer="forecast",
                    code="DIRECTIONAL_FORECAST_CALIBRATION_GEOMETRY_MISSING",
                    message=(
                        f"directional_forecast has only {len(calibrated)}/{len(directional)} "
                        "HIT/MISS target/invalidation calibration samples; resolved legacy samples "
                        "are missing target/invalidation geometry"
                        + (
                            ", but recent geometry coverage has recovered"
                            if recent_geometry_recovered
                            else ""
                        )
                    ),
                    next_action=(
                        "Keep final forecast writes populated with target and invalidation prices, "
                        "and exclude older geometry-less direction samples from confidence expansion."
                        if not recent_geometry_recovered
                        else "Keep current forecast writes populated and treat older geometry-less rows "
                        "as legacy audit debt, not a reason to suppress current calibrated edge discovery."
                    ),
                    evidence={
                        "rows": len(directional),
                        "calibrated_samples": len(calibrated),
                        "missing_geometry_samples": missing_geometry_count,
                        "target_timeout_samples": target_timeout_count,
                        "calibration_coverage": round(calibration_coverage, 4),
                        "min_coverage": FORECAST_CALIBRATION_MIN_COVERAGE,
                        "recent_recovered": recent_geometry_recovered,
                        "recent_24h_rows": len(recent_24h_directional),
                        "recent_24h_calibrated_samples": len(recent_24h_calibrated),
                        "recent_24h_calibration_coverage": round(recent_24h_coverage, 4),
                        "recent_7d_rows": len(recent_7d_directional),
                        "recent_7d_calibrated_samples": len(recent_7d_calibrated),
                        "recent_7d_calibration_coverage": round(recent_7d_coverage, 4),
                        "status_counts": status_counts,
                        "examples": [_projection_ref(row) for row in directional[:8]],
                    },
                )
            )
    entry_grade_movement_calibrated = [
        row for row in movement_calibrated
        if _directional_forecast_entry_grade(row)
    ]
    watch_only_movement_calibrated = [
        row for row in movement_calibrated
        if not _directional_forecast_entry_grade(row)
    ]
    movement_target_timeouts = [
        row for row in target_timeout_rows
        if _directional_forecast_is_movement_direction(row)
        and _directional_forecast_entry_grade(row)
    ]
    touch_or_timeout_samples = len(entry_grade_movement_calibrated) + len(movement_target_timeouts)
    target_timeout_rate = (
        len(movement_target_timeouts) / touch_or_timeout_samples
        if touch_or_timeout_samples
        else 0.0
    )
    if (
        touch_or_timeout_samples >= FORECAST_CALIBRATION_MIN_SAMPLES
        and target_timeout_rate >= FORECAST_TARGET_TIMEOUT_WARN_ABOVE
    ):
        findings.append(
            _finding(
                run_id=run_id,
                priority="P2",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_TARGET_TIMEOUT_DOMINANT",
                message=(
                    "directional_forecast full target was not reached before expiry in "
                    f"{target_timeout_rate:.1%} of {touch_or_timeout_samples} movement sample(s)"
                ),
                next_action=(
                    "Separate full-target forecasting from short-horizon entry capture: tighten target/horizon "
                    "geometry for scalps, or route these setups through RANGE/micro-rotation evidence instead "
                    "of treating target non-arrival as an adverse direction miss."
                ),
                evidence={
                    "samples": touch_or_timeout_samples,
                    "target_timeout_samples": len(movement_target_timeouts),
                    "target_timeout_rate": round(target_timeout_rate, 4),
                    "warn_above": FORECAST_TARGET_TIMEOUT_WARN_ABOVE,
                    "touch_calibrated_samples": len(entry_grade_movement_calibrated),
                    "watch_only_movement_samples_excluded": len(watch_only_movement_calibrated),
                    "examples": [_projection_ref(row) for row in movement_target_timeouts[:8]],
                },
            )
        )
    if (
        len(movement_calibrated) >= FORECAST_CALIBRATION_MIN_SAMPLES
        and len(entry_grade_movement_calibrated) < FORECAST_CALIBRATION_MIN_SAMPLES
        and len(watch_only_movement_calibrated) >= FORECAST_CALIBRATION_MIN_SAMPLES
    ):
        findings.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_ENTRY_GRADE_SAMPLE_SHORTFALL",
                message=(
                    "directional_forecast has too few entry-grade movement samples: "
                    f"{len(entry_grade_movement_calibrated)}/{len(movement_calibrated)} "
                    "movement sample(s) cleared confidence >= "
                    f"{FORECAST_ENTRY_GRADE_CONFIDENCE_MIN:.2f}"
                ),
                next_action=(
                    "Keep watch-only movement forecasts out of live-entry calibration, then improve raw "
                    "forecast priors and geometry until enough target/invalidation-scored samples clear "
                    "the entry confidence floor."
                ),
                evidence={
                    "entry_grade_samples": len(entry_grade_movement_calibrated),
                    "watch_only_movement_samples": len(watch_only_movement_calibrated),
                    "movement_samples": len(movement_calibrated),
                    "confidence_floor": FORECAST_ENTRY_GRADE_CONFIDENCE_MIN,
                    "watch_only_hit_stats": _directional_forecast_hit_stats(
                        watch_only_movement_calibrated
                    ),
                    "watch_only_invalidation_first_stats": (
                        _directional_forecast_invalidation_first_stats(
                            watch_only_movement_calibrated
                        )
                    ),
                    "watch_only_worst_buckets": _directional_forecast_worst_buckets(
                        watch_only_movement_calibrated,
                        min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
                    )[:8],
                    "examples": [_projection_ref(row) for row in watch_only_movement_calibrated[:8]],
                },
            )
        )
    weak_buckets = _directional_forecast_worst_buckets(
        entry_grade_movement_calibrated,
        min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
        warn_below=FORECAST_HIT_RATE_WARN_BELOW,
    )
    recent_24h = _directional_forecast_rows_since(entry_grade_movement_calibrated, now=now, window=timedelta(hours=24))
    recent_7d = _directional_forecast_rows_since(entry_grade_movement_calibrated, now=now, window=timedelta(days=7))
    window_stats = {
        "24h": {"window": "24h", **_directional_forecast_hit_stats(recent_24h)},
        "7d": {"window": "7d", **_directional_forecast_hit_stats(recent_7d)},
        "all": {"window": "all", **_directional_forecast_hit_stats(entry_grade_movement_calibrated)},
    }
    if weak_buckets:
        recent_24h_weak_buckets = _directional_forecast_worst_buckets(
            recent_24h,
            min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
            warn_below=FORECAST_HIT_RATE_WARN_BELOW,
        )
        recent_7d_weak_buckets = _directional_forecast_worst_buckets(
            recent_7d,
            min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
            warn_below=FORECAST_HIT_RATE_WARN_BELOW,
        )
        bucket_gate = window_stats["7d"] if int(window_stats["7d"]["samples"]) >= FORECAST_CALIBRATION_MIN_SAMPLES else window_stats["24h"]
        bucket_recent_recovered = (
            int(bucket_gate["samples"]) >= FORECAST_CALIBRATION_MIN_SAMPLES
            and (
                (bucket_gate["window"] == "7d" and not recent_7d_weak_buckets)
                or (bucket_gate["window"] == "24h" and not recent_24h_weak_buckets)
            )
        )
        findings.append(
            _finding(
                run_id=run_id,
                priority="P2" if bucket_recent_recovered else "P1",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK",
                message=(
                    f"{len(weak_buckets)} directional_forecast pair/direction/regime bucket(s) "
                    f"are below {FORECAST_HIT_RATE_WARN_BELOW:.0%} HIT rate"
                    + (
                        f", but recent {bucket_gate['window']} buckets no longer breach"
                        if bucket_recent_recovered
                        else ""
                    )
                ),
                next_action=(
                    "Dampen or rework the named forecast buckets before treating them as opportunity "
                    "expansion candidates for the 10% campaign."
                    if not bucket_recent_recovered
                    else "Keep weak historical buckets as repair evidence, but avoid suppressing current "
                    "edge discovery when the recent calibrated window has recovered."
                ),
                evidence={
                    "min_samples": FORECAST_CALIBRATION_MIN_SAMPLES,
                    "warn_below": FORECAST_HIT_RATE_WARN_BELOW,
                    "movement_samples": len(entry_grade_movement_calibrated),
                    "watch_only_movement_samples_excluded": len(watch_only_movement_calibrated),
                    "range_samples_excluded": len(range_calibrated),
                    "total_calibrated_samples": len(calibrated),
                    "weak_buckets": weak_buckets,
                    "recent_recovered": bucket_recent_recovered,
                    "recent_24h_weak_buckets": recent_24h_weak_buckets,
                    "recent_7d_weak_buckets": recent_7d_weak_buckets,
                    "window_hit_rates": window_stats,
                },
            )
        )
    samples = len(entry_grade_movement_calibrated)
    hit_count = sum(1 for row in entry_grade_movement_calibrated if str(row.get("resolution_status") or "").upper() == "HIT")
    hit_rate = hit_count / samples if samples else 0.0
    if samples >= FORECAST_CALIBRATION_MIN_SAMPLES and hit_rate < FORECAST_HIT_RATE_WARN_BELOW:
        recent_gate = window_stats["7d"] if int(window_stats["7d"]["samples"]) >= FORECAST_CALIBRATION_MIN_SAMPLES else window_stats["24h"]
        recent_recovered = (
            int(recent_gate["samples"]) >= FORECAST_CALIBRATION_MIN_SAMPLES
            and float(recent_gate["hit_rate"]) >= FORECAST_HIT_RATE_WARN_BELOW
        )
        findings.append(
            _finding(
                run_id=run_id,
                priority="P2" if recent_recovered else "P1",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                message=(
                    f"directional_forecast HIT rate is {hit_rate:.1%} "
                    f"over {samples} calibrated sample(s)"
                    + (
                        f", but recent {recent_gate['window']} recovered to "
                        f"{float(recent_gate['hit_rate']):.1%}"
                        if recent_recovered
                        else ""
                    )
                ),
                next_action=(
                    "Rank the weakest current pair/direction buckets and adjust forecast priors, target "
                    "geometry, or range-location handling before increasing opportunity frequency."
                    if not recent_recovered
                    else "Keep the recent recovered forecast cadence, but do not use stale all-history "
                    "weakness alone to block current edge discovery."
                ),
                evidence={
                    "samples": samples,
                    "hit_count": hit_count,
                    "hit_rate": round(hit_rate, 4),
                    "range_samples_excluded": len(range_calibrated),
                    "target_timeout_samples_excluded": len(movement_target_timeouts),
                    "watch_only_movement_samples_excluded": len(watch_only_movement_calibrated),
                    "total_calibrated_samples": len(calibrated),
                    "warn_below": FORECAST_HIT_RATE_WARN_BELOW,
                    "recent_recovered": recent_recovered,
                    "window_hit_rates": window_stats,
                    "worst_buckets": _directional_forecast_worst_buckets(
                        entry_grade_movement_calibrated,
                        min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
                    )[:8],
                },
            )
        )
    invalidation_first_stats = _directional_forecast_invalidation_first_stats(entry_grade_movement_calibrated)
    if (
        samples >= FORECAST_CALIBRATION_MIN_SAMPLES
        and float(invalidation_first_stats["invalidation_first_rate"]) >= FORECAST_INVALIDATION_FIRST_WARN_ABOVE
    ):
        findings.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT",
                message=(
                    "directional_forecast invalidation was touched before target in "
                    f"{float(invalidation_first_stats['invalidation_first_rate']):.1%} "
                    f"of {samples} calibrated movement sample(s)"
                ),
                next_action=(
                    "Treat this as an adverse-path forecast repair, not as permission to increase entry "
                    "frequency. Rework range-location priors, reversal handling, or target/invalidation "
                    "geometry before using these buckets for live expansion."
                ),
                evidence={
                    **invalidation_first_stats,
                    "warn_above": FORECAST_INVALIDATION_FIRST_WARN_ABOVE,
                    "watch_only_movement_samples_excluded": len(watch_only_movement_calibrated),
                    "worst_buckets": _directional_forecast_invalidation_first_buckets(
                        entry_grade_movement_calibrated,
                        min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
                    )[:8],
                },
            )
        )
    return findings


def _directional_forecast_is_movement_direction(row: dict[str, Any]) -> bool:
    return str(row.get("direction") or "").upper() in {"UP", "DOWN"}


def _directional_forecast_entry_grade(row: dict[str, Any]) -> bool:
    if not _directional_forecast_is_movement_direction(row):
        return True
    if "confidence" not in row:
        return True
    try:
        confidence = float(row.get("confidence"))
    except (TypeError, ValueError):
        return True
    return confidence >= FORECAST_ENTRY_GRADE_CONFIDENCE_MIN


def _directional_forecast_has_target_invalidation(row: dict[str, Any]) -> bool:
    if str(row.get("direction") or "").upper() == "RANGE":
        return (
            row.get("predicted_range_low_price") is not None
            and row.get("predicted_range_high_price") is not None
        )
    return row.get("predicted_target_price") is not None and row.get("predicted_invalidation_price") is not None


def _directional_forecast_target_timeout_like(row: dict[str, Any]) -> bool:
    if str(row.get("resolution_status") or "").upper() != "MISS":
        return False
    if str(row.get("direction") or "").upper() not in {"UP", "DOWN"}:
        return False
    evidence = str(row.get("resolution_evidence") or "").lower()
    if not evidence:
        return False
    if "touched before target" in evidence or "invalidation also touched" in evidence:
        return False
    if "ordering ambiguous" in evidence:
        return False
    if "both untouched" in evidence or "also untouched" in evidence:
        return True
    if "target" in evidence and "not reached before invalidation" in evidence:
        return True
    if "did not reach target" in evidence or "did not reach the target" in evidence:
        return True
    return False


def _directional_forecast_rows_since(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    window: timedelta,
) -> list[dict[str, Any]]:
    cutoff = now - window
    out: list[dict[str, Any]] = []
    for row in rows:
        emitted_at = _parse_utc(row.get("timestamp_emitted_utc"))
        if emitted_at is not None and emitted_at >= cutoff:
            out.append(row)
    return out


def _directional_forecast_hit_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    hit_count = sum(1 for row in rows if str(row.get("resolution_status") or "").upper() == "HIT")
    return {
        "samples": samples,
        "hit_count": hit_count,
        "hit_rate": round(hit_count / samples, 4) if samples else 0.0,
    }


def _directional_forecast_invalidation_first_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    invalidation_first_count = sum(1 for row in rows if _directional_forecast_invalidation_first_like(row))
    return {
        "samples": samples,
        "invalidation_first_count": invalidation_first_count,
        "invalidation_first_rate": round(invalidation_first_count / samples, 4) if samples else 0.0,
    }


def _directional_forecast_invalidation_first_like(row: dict[str, Any]) -> bool:
    if str(row.get("resolution_status") or "").upper() != "MISS":
        return False
    evidence = str(row.get("resolution_evidence") or "").lower()
    if "invalidation" not in evidence:
        return False
    return (
        "before target" in evidence
        or "invalidation also touched" in evidence
    )


def _directional_forecast_invalidation_first_buckets(
    rows: list[dict[str, Any]],
    *,
    min_samples: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        pair = str(row.get("pair") or "UNKNOWN")
        direction = str(row.get("direction") or "UNKNOWN").upper()
        regime = str(row.get("regime_at_emission") or "UNCLEAR").upper()
        item = buckets.setdefault(
            (pair, direction, regime),
            {"pair": pair, "direction": direction, "regime": regime, "samples": 0, "invalidation_first": 0},
        )
        item["samples"] = int(item["samples"]) + 1
        if _directional_forecast_invalidation_first_like(row):
            item["invalidation_first"] = int(item["invalidation_first"]) + 1
    ranked: list[dict[str, Any]] = []
    for item in buckets.values():
        samples = int(item["samples"])
        invalidation_first = int(item["invalidation_first"])
        if samples < min_samples:
            continue
        ranked.append(
            {
                **item,
                "invalidation_first_rate": round(invalidation_first / samples, 4) if samples else 0.0,
            }
        )
    return sorted(
        ranked,
        key=lambda item: (-float(item["invalidation_first_rate"]), -int(item["samples"]), str(item["pair"])),
    )


def _directional_forecast_worst_buckets(
    rows: list[dict[str, Any]],
    *,
    min_samples: int = 1,
    warn_below: float | None = None,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        pair = str(row.get("pair") or "UNKNOWN")
        direction = str(row.get("direction") or "UNKNOWN").upper()
        regime = str(row.get("regime_at_emission") or "UNCLEAR").upper()
        item = buckets.setdefault(
            (pair, direction, regime),
            {"pair": pair, "direction": direction, "regime": regime, "samples": 0, "hits": 0},
        )
        item["samples"] = int(item["samples"]) + 1
        if str(row.get("resolution_status") or "").upper() == "HIT":
            item["hits"] = int(item["hits"]) + 1
    ranked: list[dict[str, Any]] = []
    for item in buckets.values():
        samples = int(item["samples"])
        hits = int(item["hits"])
        hit_rate = hits / samples if samples else 0.0
        if samples < min_samples:
            continue
        if warn_below is not None and hit_rate >= warn_below:
            continue
        ranked.append({**item, "hit_rate": round(hit_rate, 4)})
    return sorted(ranked, key=lambda item: (float(item["hit_rate"]), -int(item["samples"]), str(item["pair"])))[:8]


def _entry_thesis_findings(
    *,
    run_id: str,
    path: Path,
    active_trade_ids: set[str],
) -> list[dict[str, Any]]:
    rows, malformed, error = _read_jsonl(path)
    if error is not None:
        if not active_trade_ids:
            return []
        return [
            _finding(
                run_id=run_id,
                priority="P0",
                layer="position_memory",
                code="ENTRY_THESIS_LEDGER_UNREADABLE",
                message=f"entry thesis ledger is unreadable: {path}: {error}",
                next_action="Restore entry_thesis_ledger or route affected positions through machine-checkable sidecars.",
            )
        ]
    thesis_ids = {str(row.get("trade_id") or "") for row in rows if str(row.get("trade_id") or "")}
    missing = sorted(trade_id for trade_id in active_trade_ids if trade_id not in thesis_ids)
    out: list[dict[str, Any]] = []
    if malformed:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="position_memory",
                code="ENTRY_THESIS_LEDGER_MALFORMED_ROWS",
                message=f"entry thesis ledger has {malformed} malformed row(s)",
                next_action="Repair malformed entry thesis rows before relying on position evolution.",
                evidence={"malformed_rows": malformed},
            )
        )
    if missing:
        out.append(
            _finding(
                run_id=run_id,
                priority="P0",
                layer="position_memory",
                code="ENTRY_THESIS_MISSING_FOR_OPEN_TRADES",
                message=f"{len(missing)} active trader-owned trade(s) lack entry thesis history",
                next_action="Run thesis/position sidecar review and prevent new risk until affected trades are machine-checkable.",
                evidence={"trade_ids": missing[:20]},
            )
        )
    return out


def _profitability_findings(
    *,
    run_id: str,
    effect: dict[str, Any],
    effect_24h: dict[str, Any],
    snapshot: dict[str, Any],
    min_sample: int,
    close_gate_loss_evidence: dict[str, Any] | None = None,
    previous_discipline_streak: int = 0,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    closed = int(effect.get("closed_trades") or 0)
    net = float(effect.get("net_jpy") or 0.0)
    pf = _maybe_float(effect.get("profit_factor"))
    expectancy = _maybe_float(effect.get("expectancy_jpy"))
    if closed < min_sample:
        out.append(
            _finding(
                run_id=run_id,
                priority="P2",
                layer="profitability",
                code="INSUFFICIENT_RECENT_OUTCOME_SAMPLE",
                message=f"only {closed} closed trade(s) in the audit window",
                next_action="Keep collecting outcome data, but do not treat the sample as proof of edge.",
                evidence={"closed_trades": closed, "min_sample": min_sample},
            )
        )
    elif net < 0 or (expectancy is not None and expectancy < 0) or (pf is not None and pf < 1.0):
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="profitability",
                code="NEGATIVE_RECENT_EXPECTANCY",
                message=(
                    f"recent outcome window is not profitable: net={net:.2f} JPY, "
                    f"PF={_fmt_optional(pf)}, expectancy={_fmt_optional(expectancy)}"
                ),
                next_action="Rank pair/side/method losers and open a targeted repair before increasing exposure.",
                evidence={"effect_metrics": effect},
            )
        )
    avg_win = _maybe_float(effect.get("avg_win_jpy"))
    avg_loss = _maybe_float(effect.get("avg_loss_jpy_abs"))
    has_loss_asymmetry = avg_win is not None and avg_win > 0 and avg_loss is not None and avg_loss > avg_win * 2
    if avg_win is not None and avg_win > 0 and avg_loss is not None and avg_loss > avg_win * 2:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="profitability",
                code="SMALL_WIN_LARGE_LOSS_ASYMMETRY",
                message=f"average loss {avg_loss:.2f} JPY is more than 2x average win {avg_win:.2f} JPY",
                next_action="Audit TP capture, giveback, and close discipline for the worst losing segment.",
                evidence={"avg_win_jpy": avg_win, "avg_loss_jpy_abs": avg_loss, "worst_segments": effect.get("worst_segments", [])[:5]},
            )
        )
    has_negative_expectancy = closed >= min_sample and (
        net < 0 or (expectancy is not None and expectancy < 0) or (pf is not None and pf < 1.0)
    )
    current_discipline_streak = previous_discipline_streak + 1
    direct_close_repair = _direct_or_manual_close_repair_evidence(effect, effect_24h)
    gateway_close_bleed = _gateway_close_bleed_observation(effect_24h)
    if has_negative_expectancy and has_loss_asymmetry and direct_close_repair:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="profitability",
                code="DIRECT_OR_MANUAL_CLOSE_DOMINATED_PROFITABILITY_DRAG",
                message=(
                    "historical profitability drag is dominated by non-gateway broker close losses, "
                    "while the last-24h gateway-controllable window is positive"
                ),
                next_action=(
                    "Keep auditing non-gateway close provenance, but do not let repaired non-gateway "
                    "history by itself block fresh risk."
                ),
                evidence=direct_close_repair,
            )
        )
    if (
        has_negative_expectancy
        and not direct_close_repair
        and current_discipline_streak >= max(1, PERSISTENT_PROFITABILITY_STREAK_MIN)
    ):
        recovery_observation = _gateway_close_recovery_observation(effect_24h)
        system_defect_evidence = {
            "profit_factor": pf,
            "expectancy_jpy": expectancy,
            "avg_win_jpy": avg_win,
            "avg_loss_jpy_abs": avg_loss,
            "worst_segments": effect.get("worst_segments", [])[:5],
            "market_order_trade_close_loss_provenance_metrics": effect.get(
                "market_order_trade_close_loss_provenance_metrics", {}
            ),
        }
        if close_gate_loss_evidence:
            system_defect_evidence["ai_backtest_close_gate_loss_evidence"] = close_gate_loss_evidence
        if gateway_close_bleed is not None:
            system_defect_evidence["gateway_close_bleed_observation"] = gateway_close_bleed
        elif not has_loss_asymmetry:
            system_defect_evidence["persistent_negative_expectancy_without_recovery"] = {
                "profit_factor": pf,
                "expectancy_jpy": expectancy,
                "current_streak": current_discipline_streak,
                "last_24h_closed_trades": int(effect_24h.get("closed_trades") or 0),
                "last_24h_profit_factor": _maybe_float(effect_24h.get("profit_factor")),
                "last_24h_expectancy_jpy": _maybe_float(effect_24h.get("expectancy_jpy")),
                "last_24h_gateway_recovery_proven": False,
            }
        if recovery_observation is not None:
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P1",
                    layer="profitability",
                    code="PERSISTENT_PROFITABILITY_DISCIPLINE_RECOVERY",
                    message=(
                        f"trailing profitability discipline is still failed after {current_discipline_streak} "
                        f"consecutive audit run(s) (PF={_fmt_optional(pf)}, expectancy={_fmt_optional(expectancy)}), "
                        "but the last-24h gateway-attributable close window proves repaired discipline: "
                        f"net={recovery_observation['gateway_net_jpy']:.2f} JPY over "
                        f"{recovery_observation['gateway_trades']} close(s) without loss asymmetry"
                    ),
                    next_action=(
                        "Resume risk-validated entries at the normal per-trade budget; this re-escalates to P0 "
                        "on the next 24h gateway close window that turns negative or asymmetric."
                    ),
                    evidence={
                        "current_streak": current_discipline_streak,
                        "previous_streak": previous_discipline_streak,
                        "streak_min": PERSISTENT_PROFITABILITY_STREAK_MIN,
                        "effect_metrics": effect,
                        "last_24h_effect_metrics": effect_24h,
                        "system_defect_evidence": system_defect_evidence,
                        "recovery_observation": recovery_observation,
                    },
                )
            )
        else:
            gateway_bleed_suffix = ""
            if gateway_close_bleed is not None:
                contained_erased_wins = (
                    gateway_close_bleed.get("gateway_bleed_basis") == "contained_loss_erased_wins"
                )
                gateway_bleed_label = "24h_gateway_raw_net" if contained_erased_wins else "24h_gateway_net"
                gateway_bleed_value = (
                    gateway_close_bleed.get("gateway_raw_net_jpy")
                    if contained_erased_wins
                    else gateway_close_bleed.get("gateway_net_jpy")
                )
                if gateway_bleed_value is not None:
                    gateway_bleed_suffix = f", {gateway_bleed_label}={float(gateway_bleed_value):.2f} JPY"
            repair_target = _worst_segment_repair_target(effect)
            repair_target_suffix = f", inspect={repair_target}" if repair_target else ""
            repair_next = (
                f"Inspect {repair_target}, then block new-risk confidence until that segment proves "
                "repaired close discipline or the trader route explicitly justifies the exception."
                if repair_target
                else (
                    "Block new-risk confidence until execution_ledger.db worst segments prove repaired "
                    "close discipline or the trader route explicitly justifies the exception."
                )
            )
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P0",
                    layer="profitability",
                    code="PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
                    message=(
                        f"profitability discipline has failed for {current_discipline_streak} consecutive audit run(s): "
                        f"PF={_fmt_optional(pf)}, expectancy={_fmt_optional(expectancy)}, "
                        f"avg_loss={_fmt_optional(avg_loss)} JPY vs avg_win={_fmt_optional(avg_win)} JPY"
                        + gateway_bleed_suffix
                        + repair_target_suffix
                    ),
                    next_action=repair_next,
                    evidence={
                        "current_streak": current_discipline_streak,
                        "previous_streak": previous_discipline_streak,
                        "streak_min": PERSISTENT_PROFITABILITY_STREAK_MIN,
                        "effect_metrics": effect,
                        "last_24h_effect_metrics": effect_24h,
                        "system_defect_evidence": system_defect_evidence,
                    },
                )
            )
    account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
    unrealized = _maybe_float(account.get("unrealized_pl_jpy"))
    gross_profit_24h = float(effect_24h.get("gross_profit_jpy") or 0.0)
    if unrealized is not None and unrealized < 0 and gross_profit_24h > 0 and abs(unrealized) > gross_profit_24h:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="profitability",
                code="OPEN_LOSS_EXCEEDS_24H_REALIZED_GAIN",
                message=(
                    f"open unrealized loss {unrealized:.2f} JPY exceeds last-24h realized gross profit "
                    f"{gross_profit_24h:.2f} JPY"
                ),
                next_action="Review open-position thesis sidecars; do not let closed-win metrics hide open drawdown.",
                evidence={"unrealized_pl_jpy": unrealized, "gross_profit_24h_jpy": gross_profit_24h},
            )
        )
    return out


def _worst_segment_repair_target(effect: dict[str, Any]) -> str:
    segments = effect.get("worst_segments")
    if not isinstance(segments, list) or not segments:
        return ""
    segment = segments[0]
    if not isinstance(segment, dict):
        return ""
    parts: list[str] = []
    pair = str(segment.get("pair") or "").strip()
    side = str(segment.get("side") or "").strip()
    method = str(segment.get("method") or "").strip()
    if pair:
        parts.append(f"pair={pair}")
    if side:
        parts.append(f"side={side}")
    if method:
        parts.append(f"method={method}")
    trades = segment.get("trades")
    if trades is not None:
        parts.append(f"trades={trades}")
    net = _maybe_float(segment.get("net_jpy"))
    if net is not None:
        parts.append(f"net={net:.2f} JPY")
    ids = [str(item) for item in segment.get("trade_ids", []) or [] if str(item)]
    if ids:
        parts.append(f"trade_ids={','.join(ids[:5])}")
    if not parts:
        return ""
    return "data/execution_ledger.db worst_segment[" + ", ".join(parts) + "]"


def _direct_or_manual_close_repair_evidence(
    effect: dict[str, Any],
    effect_24h: dict[str, Any],
) -> dict[str, Any] | None:
    """Identify repaired profitability drag caused by non-gateway close provenance.

    The 24h window already exists in this audit as the operational recovery
    check. This helper does not create trade permission; it prevents old
    direct/manual broker closes from remaining a P0 once removing that
    non-gateway provenance makes the current audit window non-negative and the
    last-24h gateway-controllable result is positive.
    """
    direct_metric = _combined_close_provenance_metric(
        effect,
        NON_GATEWAY_CLOSE_DRAG_PROVENANCES,
    )
    direct_net = _maybe_float(direct_metric.get("net_jpy"))
    direct_loss_trades = int(direct_metric.get("loss_trades") or 0)
    if direct_net is None or direct_net >= 0.0 or direct_loss_trades <= 0:
        return None

    net = _maybe_float(effect.get("net_jpy"))
    if net is None:
        return None
    net_without_direct = net - direct_net
    if net_without_direct < 0.0:
        return None

    recent_net = _maybe_float(effect_24h.get("net_jpy"))
    recent_expectancy = _maybe_float(effect_24h.get("expectancy_jpy"))
    recent_pf = _maybe_float(effect_24h.get("profit_factor"))
    recent_closed = int(effect_24h.get("closed_trades") or 0)
    if recent_closed <= 0 or recent_net is None or recent_net <= 0.0:
        return None
    if recent_expectancy is not None and recent_expectancy <= 0.0:
        return None
    if recent_pf is not None and recent_pf <= 1.0:
        return None

    recent_direct_market_close_loss = _combined_close_provenance_metric(
        {
            "market_order_trade_close_loss_provenance_metrics": effect_24h.get(
                "market_order_trade_close_loss_provenance_metrics", {}
            )
        },
        NON_GATEWAY_CLOSE_DRAG_PROVENANCES,
        metric_key="market_order_trade_close_loss_provenance_metrics",
    )
    if int(recent_direct_market_close_loss.get("loss_trades") or 0) > 0:
        return None

    return {
        "non_gateway_close_drag_provenances": list(NON_GATEWAY_CLOSE_DRAG_PROVENANCES),
        "non_gateway_close_drag_metric": direct_metric,
        "net_jpy": net,
        "net_without_non_gateway_close_drag_jpy": net_without_direct,
        "last_24h_net_jpy": recent_net,
        "last_24h_profit_factor": recent_pf,
        "last_24h_expectancy_jpy": recent_expectancy,
        "last_24h_non_gateway_market_close_loss_trades": int(
            recent_direct_market_close_loss.get("loss_trades") or 0
        ),
    }


# Close provenances that can neither prove nor disprove the trader's own
# gateway close discipline: operator manual/tagless closes (§9 manual-exposure
# separation — operator-owned outcomes must neither keep the trader blocked
# nor unblock it) and rows whose provenance could not be established.
# Extends NON_GATEWAY_CLOSE_DRAG_PROVENANCES with the same intent.
RECOVERY_EXCLUDED_CLOSE_PROVENANCES = NON_GATEWAY_CLOSE_DRAG_PROVENANCES + (
    "NON_TRADER_CLIENT_EXTENSION",
    "UNKNOWN_CLOSE_PROVENANCE",
)


def _gateway_close_recovery_observation(effect_24h: dict[str, Any]) -> dict[str, Any] | None:
    """Prove repaired close discipline from the last-24h gateway-attributable window.

    The persistent profitability P0 blocks fresh entries "until
    execution_ledger.db worst segments prove repaired close discipline". The
    trailing audit window cannot supply that proof while entries are blocked —
    it only changes as old losers age out — so the P0 deadlocks against its own
    repair demand (observed 2026-06-11: 65 consecutive blocked runs while the
    last-24h gateway window was already positive). 24h is the audit's
    documented operational recovery window, the same boundary
    `_direct_or_manual_close_repair_evidence` uses.

    Recovery requires positive evidence, not absence of evidence: either a
    gateway-attributable winning close, or a material loss-containment close
    that avoided more loss than it realized while leaving active close
    discipline non-negative. The active close window must not show
    small-win/large-loss asymmetry (same 2x boundary as
    SMALL_WIN_LARGE_LOSS_ASYMMETRY). TP fills and gateway closes of held
    positions keep generating this evidence even while fresh entries are
    blocked, so the gate can clear without bypassing risk validation.
    """
    if effect_24h.get("error"):
        return None
    metrics = effect_24h.get("close_provenance_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    net = 0.0
    loss_containment_net = 0.0
    loss_containment_trades = 0
    loss_containment_avoided = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    win_trades = 0
    loss_trades = 0
    for provenance, values in metrics.items():
        if provenance in RECOVERY_EXCLUDED_CLOSE_PROVENANCES:
            continue
        if not isinstance(values, dict):
            continue
        net += float(values.get("net_jpy") or 0.0)
        loss_containment_net += float(values.get("loss_containment_net_jpy") or 0.0)
        loss_containment_trades += int(values.get("loss_containment_trades") or 0)
        loss_containment_avoided += float(values.get("loss_containment_avoided_loss_jpy") or 0.0)
        gross_profit += float(values.get("gross_profit_jpy") or 0.0)
        gross_loss += float(values.get("gross_loss_jpy") or 0.0)
        trades += int(values.get("trades") or 0)
        win_trades += int(values.get("win_trades") or 0)
        loss_trades += int(values.get("loss_trades") or 0)
    close_discipline_net = net - loss_containment_net
    active_loss_trades = max(0, loss_trades - loss_containment_trades)
    active_gross_loss = max(0.0, gross_loss - abs(loss_containment_net))
    material_loss_containment = (
        loss_containment_trades > 0
        and active_loss_trades == 0
        and loss_containment_avoided
        >= abs(loss_containment_net) * LOSS_CONTAINMENT_RECOVERY_MIN_AVOIDED_MULT
    )
    if win_trades > 0 and net < 0.0:
        return None
    if close_discipline_net < 0.0 or (win_trades < 1 and not material_loss_containment):
        return None
    avg_win = gross_profit / win_trades if win_trades else None
    avg_loss = (active_gross_loss / active_loss_trades) if active_loss_trades else None
    if avg_win is not None and avg_loss is not None and avg_loss > avg_win * 2:
        return None
    observation = {
        "window_hours": 24.0,
        "recovery_basis": "material_loss_containment" if win_trades < 1 else "winning_close_window",
        "gateway_net_jpy": close_discipline_net,
        "gateway_raw_net_jpy": net,
        "gateway_trades": trades,
        "gateway_win_trades": win_trades,
        "gateway_loss_trades": active_loss_trades,
        "gateway_avg_win_jpy": avg_win,
        "gateway_avg_loss_jpy_abs": avg_loss,
        "excluded_provenances": list(RECOVERY_EXCLUDED_CLOSE_PROVENANCES),
    }
    if loss_containment_trades:
        observation.update(
            {
                "loss_containment_trades": loss_containment_trades,
                "loss_containment_net_jpy": loss_containment_net,
                "loss_containment_avoided_loss_jpy": loss_containment_avoided,
            }
        )
    return observation


def _gateway_close_bleed_observation(effect_24h: dict[str, Any]) -> dict[str, Any] | None:
    """Return current gateway-attributable close loss evidence.

    Average win/loss asymmetry catches one class of broken close discipline,
    but a sequence can still be system-negative when gateway closes bleed more
    often than TP fills while average loss size is similar to average win size.
    That is still a production discipline defect because it is attributable to
    QuantRabbit's close path, not operator/manual broker actions.
    """
    if effect_24h.get("error"):
        return None
    metrics = effect_24h.get("close_provenance_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    net = 0.0
    loss_containment_net = 0.0
    loss_containment_trades = 0
    loss_containment_avoided = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trades = 0
    win_trades = 0
    loss_trades = 0
    for provenance, values in metrics.items():
        if provenance in RECOVERY_EXCLUDED_CLOSE_PROVENANCES:
            continue
        if not isinstance(values, dict):
            continue
        net += float(values.get("net_jpy") or 0.0)
        loss_containment_net += float(values.get("loss_containment_net_jpy") or 0.0)
        loss_containment_trades += int(values.get("loss_containment_trades") or 0)
        loss_containment_avoided += float(values.get("loss_containment_avoided_loss_jpy") or 0.0)
        gross_profit += float(values.get("gross_profit_jpy") or 0.0)
        gross_loss += float(values.get("gross_loss_jpy") or 0.0)
        trades += int(values.get("trades") or 0)
        win_trades += int(values.get("win_trades") or 0)
        loss_trades += int(values.get("loss_trades") or 0)
    close_discipline_net = net - loss_containment_net
    active_loss_trades = max(0, loss_trades - loss_containment_trades)
    active_gross_loss = max(0.0, gross_loss - abs(loss_containment_net))
    contained_loss_erased_wins = win_trades > 0 and loss_containment_trades > 0 and net < 0.0
    if not contained_loss_erased_wins and (active_loss_trades < 1 or close_discipline_net >= 0.0):
        return None
    observation = {
        "window_hours": 24.0,
        "gateway_net_jpy": close_discipline_net,
        "gateway_raw_net_jpy": net,
        "gateway_bleed_basis": "contained_loss_erased_wins" if contained_loss_erased_wins else "active_close_loss",
        "gateway_trades": trades,
        "gateway_win_trades": win_trades,
        "gateway_loss_trades": active_loss_trades,
        "gateway_avg_win_jpy": (gross_profit / win_trades) if win_trades else None,
        "gateway_avg_loss_jpy_abs": (active_gross_loss / active_loss_trades) if active_loss_trades else None,
        "excluded_provenances": list(RECOVERY_EXCLUDED_CLOSE_PROVENANCES),
    }
    if loss_containment_trades:
        observation.update(
            {
                "loss_containment_trades": loss_containment_trades,
                "loss_containment_net_jpy": loss_containment_net,
                "loss_containment_avoided_loss_jpy": loss_containment_avoided,
            }
        )
    return observation


def _combined_close_provenance_metric(
    effect: dict[str, Any],
    provenances: tuple[str, ...],
    *,
    metric_key: str = "close_provenance_metrics",
) -> dict[str, Any]:
    combined = {
        "gross_loss_jpy": 0.0,
        "gross_profit_jpy": 0.0,
        "loss_trades": 0,
        "net_jpy": 0.0,
        "trades": 0,
        "win_trades": 0,
    }
    found = False
    for provenance in provenances:
        metric = _close_provenance_metric(effect, provenance, metric_key=metric_key)
        if not metric:
            continue
        found = True
        for key in ("gross_loss_jpy", "gross_profit_jpy", "net_jpy"):
            combined[key] = float(combined[key]) + float(metric.get(key) or 0.0)
        for key in ("loss_trades", "trades", "win_trades"):
            combined[key] = int(combined[key]) + int(metric.get(key) or 0)
    return combined if found else {}


def _close_provenance_metric(
    effect: dict[str, Any],
    provenance: str,
    *,
    metric_key: str = "close_provenance_metrics",
) -> dict[str, Any]:
    metrics = effect.get(metric_key)
    if not isinstance(metrics, dict):
        return {}
    metric = metrics.get(provenance)
    return metric if isinstance(metric, dict) else {}


def _intent_findings(
    *,
    run_id: str,
    intents: dict[str, Any],
    target_state: dict[str, Any],
    target_open: bool,
    live_ready: list[dict[str, Any]],
    active_positions: list[dict[str, Any]],
    pending_entry_orders: list[dict[str, Any]],
    coverage_optimization: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    forecast_arbitration = _forecast_arbitration_diagnostics(intents) if target_open else {}
    if target_open and not live_ready:
        entry_path_occupied = bool(active_positions or pending_entry_orders)
        coverage_refresh = _coverage_market_evidence_refresh(coverage_optimization)
        out.append(
            _finding(
                run_id=run_id,
                priority="P0" if not (entry_path_occupied or coverage_refresh) else "P1",
                layer="opportunity",
                code="TARGET_OPEN_NO_LIVE_READY_LANES",
                message="daily target is open but order_intents has no LIVE_READY lanes",
                next_action=(
                    "Refresh broker truth and regenerate intents after quotes/spreads become tradable; "
                    "do not treat market-evidence noise as a strategy expansion defect yet."
                    if coverage_refresh
                    else "Refresh market context and inspect top live blockers instead of ending flat without a named gate."
                ),
                evidence={
                    "active_trader_positions": len(active_positions),
                    "trader_pending_entry_orders": _pending_entry_evidence(pending_entry_orders),
                    "coverage_market_evidence_refresh": coverage_refresh,
                    "opportunity_modes": _coverage_opportunity_mode_summary(coverage_optimization),
                    "runner_candidate_diagnostics": _coverage_runner_candidate_diagnostics(
                        coverage_optimization
                    ),
                    "perspective_alignment_diagnostics": _coverage_perspective_alignment_diagnostics(
                        coverage_optimization
                    ),
                    "status_counts": _intent_status_counts(intents),
                    "top_blockers": _top_intent_blockers(intents),
                    "dry_run_passed_live_readiness_blockers": _top_intent_live_readiness_blockers(
                        intents,
                        statuses={"DRY_RUN_PASSED"},
                    ),
                    "dry_run_passed_forecast_gate_diagnostics": _dry_run_passed_forecast_gate_diagnostics(
                        intents
                    ),
                    "forecast_arbitration_diagnostics": forecast_arbitration,
                    "live_readiness_blocker_families": _intent_live_readiness_family_breakdown(intents),
                    "non_live_ready_live_readiness_blockers": _top_intent_live_readiness_blockers(intents),
                },
            )
        )
    same_side_arbitration = int(
        forecast_arbitration.get("same_side_actionable_repair_lane_count") or 0
    )
    same_side_context_blocked = int(
        forecast_arbitration.get("same_side_context_blocked_lane_count") or 0
    )
    opposite_arbitration = int(
        forecast_arbitration.get("opposite_conflict_lane_count")
        or forecast_arbitration.get("opposite_side_lane_count")
        or 0
    )
    if target_open and same_side_arbitration > 0:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1" if not live_ready else "P2",
                layer="forecast",
                code="FORECAST_ARBITRATION_UNSELECTED_PROJECTION_REPAIR_REQUIRED",
                message=(
                    f"{same_side_arbitration} dry-run passed lane(s) have same-side audited "
                    "directional projection evidence that the pair forecast left unselected"
                ),
                next_action=(
                    "Repair forecast arbitration before forcing live sends: resolve only these same-side "
                    "projection candidates into a pair forecast with target/invalidation geometry, while "
                    "keeping opposite-projection RANGE/UNCLEAR lanes blocked."
                ),
                evidence={"forecast_arbitration_diagnostics": forecast_arbitration},
            )
        )
    if target_open and same_side_context_blocked > 0:
        out.append(
            _finding(
                run_id=run_id,
                priority="P2",
                layer="forecast",
                code="FORECAST_ARBITRATION_SAME_SIDE_CONTEXT_BLOCKED",
                message=(
                    f"{same_side_context_blocked} dry-run passed same-side projection candidate(s) "
                    "also have non-forecast live-readiness blockers"
                ),
                next_action=(
                    "Do not count these as immediately missed entries: repair forecast arbitration only "
                    "after the named chart, strategy-profile, liquidity, or risk-geometry blockers clear."
                ),
                evidence={"forecast_arbitration_diagnostics": forecast_arbitration},
            )
        )
    if target_open and opposite_arbitration > 0:
        out.append(
            _finding(
                run_id=run_id,
                priority="P2",
                layer="forecast",
                code="FORECAST_ARBITRATION_OPPOSITE_PROJECTION_CONFLICTS_ENFORCED",
                message=(
                    f"{opposite_arbitration} dry-run passed lane(s) have opposite or mixed audited "
                    "projection evidence that should remain blocked until the pair forecast resolves"
                ),
                next_action=(
                    "Keep these lanes out of live send; do not count them as missed entries. "
                    "Use them as evidence that the forecast packet is conflicted, not as permission "
                    "to trade through RiskEngine."
                ),
                evidence={"forecast_arbitration_diagnostics": forecast_arbitration},
            )
        )
    if target_open and live_ready:
        coverage_gap = _coverage_live_ready_shortfall(
            coverage_optimization=coverage_optimization,
            target_state=target_state,
            live_ready_count=len(live_ready),
        )
        if coverage_gap:
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P1",
                    layer="opportunity",
                    code="TARGET_OPEN_LIVE_READY_COVERAGE_SHORTFALL",
                    message=(
                        "daily target is open and current LIVE_READY coverage is only "
                        f"{coverage_gap['target_coverage_pct']:.1f}% of remaining target"
                    ),
                    next_action=(
                        "Keep the current tradeable lane managed, then build additional HARVEST and "
                        "RUNNER candidates from the named forecast/strategy blockers before treating "
                        "the campaign as sufficiently covered."
                    ),
                    evidence={
                        **coverage_gap,
                        "active_trader_positions": len(active_positions),
                        "trader_pending_entry_orders": _pending_entry_evidence(pending_entry_orders),
                        "opportunity_modes": _coverage_opportunity_mode_summary(coverage_optimization),
                        "runner_candidate_diagnostics": _coverage_runner_candidate_diagnostics(
                            coverage_optimization
                        ),
                        "perspective_alignment_diagnostics": _coverage_perspective_alignment_diagnostics(
                            coverage_optimization
                        ),
                        "status_counts": _intent_status_counts(intents),
                        "top_blockers": _top_intent_blockers(intents),
                        "live_readiness_blocker_families": _intent_live_readiness_family_breakdown(intents),
                        "non_live_ready_live_readiness_blockers": _top_intent_live_readiness_blockers(intents),
                    },
                )
            )
    low_rr: list[dict[str, Any]] = []
    for result in live_ready:
        intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
        order_type = str(intent.get("order_type") or "").upper()
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        order_timing = str(metadata.get("order_timing") or "").upper()
        rr = _maybe_float((result.get("risk_metrics") or {}).get("reward_risk") if isinstance(result.get("risk_metrics"), dict) else None)
        if (order_type == "MARKET" or order_timing == "NOW_MARKET") and rr is not None and rr < 1.0:
            low_rr.append({"lane_id": result.get("lane_id"), "reward_risk": rr, "status": result.get("status")})
    if low_rr:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="execution_quality",
                code="LIVE_READY_MARKET_RR_BELOW_ONE",
                message=f"{len(low_rr)} LIVE_READY MARKET lane(s) have reward/risk below 1.0",
                next_action="Prefer trigger/limit timing or repair target geometry before accepting MARKET participation.",
                evidence={"lanes": low_rr[:12]},
            )
        )
    return out


def _coverage_live_ready_shortfall(
    *,
    coverage_optimization: dict[str, Any],
    target_state: dict[str, Any],
    live_ready_count: int,
) -> dict[str, Any] | None:
    if live_ready_count <= 0:
        return None
    live_reward = _maybe_float(coverage_optimization.get("live_ready_reward_jpy"))
    if live_reward is None:
        live_reward = _coverage_modes_live_reward(coverage_optimization)
    remaining_target = _maybe_float(coverage_optimization.get("remaining_target_jpy"))
    if remaining_target is None:
        remaining_target = _maybe_float(target_state.get("remaining_target_jpy"))
    remaining_minimum = _maybe_float(target_state.get("remaining_minimum_jpy"))
    if remaining_target is None or remaining_target <= 0:
        return None
    live_reward = max(0.0, float(live_reward or 0.0))
    if live_reward >= remaining_target:
        return None
    minimum_shortfall = (
        max(0.0, remaining_minimum - live_reward)
        if remaining_minimum is not None and remaining_minimum > 0
        else None
    )
    return {
        "live_ready_lanes": live_ready_count,
        "live_ready_reward_jpy": round(live_reward, 4),
        "remaining_target_jpy": round(float(remaining_target), 4),
        "remaining_minimum_jpy": round(float(remaining_minimum), 4)
        if remaining_minimum is not None
        else None,
        "required_additional_reward_jpy": round(max(0.0, float(remaining_target) - live_reward), 4),
        "minimum_floor_shortfall_jpy": round(minimum_shortfall, 4)
        if minimum_shortfall is not None
        else None,
        "target_coverage_pct": round((live_reward / float(remaining_target)) * 100.0, 4),
        "coverage_status": coverage_optimization.get("status"),
    }


def _coverage_modes_live_reward(payload: dict[str, Any]) -> float:
    modes = payload.get("opportunity_modes") if isinstance(payload.get("opportunity_modes"), dict) else {}
    reward = 0.0
    for item in modes.values():
        if isinstance(item, dict):
            reward += float(_maybe_float(item.get("live_ready_reward_jpy")) or 0.0)
    return reward


def _coverage_opportunity_mode_summary(payload: dict[str, Any]) -> dict[str, Any]:
    modes = payload.get("opportunity_modes") if isinstance(payload.get("opportunity_modes"), dict) else {}
    summary: dict[str, Any] = {}
    for mode in ("HARVEST", "RUNNER", "BALANCED"):
        item = modes.get(mode)
        if not isinstance(item, dict):
            continue
        summary[mode] = {
            "lanes": _maybe_int(item.get("lanes")) or 0,
            "live_ready_lanes": _maybe_int(item.get("live_ready_lanes")) or 0,
            "promotion_candidate_lanes": _maybe_int(item.get("promotion_candidate_lanes")) or 0,
            "reward_jpy": _maybe_float(item.get("reward_jpy")) or 0.0,
            "live_ready_reward_jpy": _maybe_float(item.get("live_ready_reward_jpy")) or 0.0,
            "potential_reward_jpy": _maybe_float(item.get("potential_reward_jpy")) or 0.0,
            "coverage_pct": _maybe_float(item.get("coverage_pct")) or 0.0,
            "potential_coverage_pct": _maybe_float(item.get("potential_coverage_pct")) or 0.0,
            "diagnostic_candidate_lanes": _maybe_int(item.get("diagnostic_candidate_lanes")) or 0,
            "demoted_to_harvest_lanes": _maybe_int(item.get("demoted_to_harvest_lanes")) or 0,
            "runner_qualified_lanes": _maybe_int(item.get("runner_qualified_lanes")) or 0,
            "diagnostic_status": str(item.get("diagnostic_status") or ""),
            "top_demotion_reasons": [
                {
                    "reason": str(reason.get("reason") or ""),
                    "count": _maybe_int(reason.get("count")) or 0,
                }
                for reason in (item.get("top_demotion_reasons") or [])[:5]
                if isinstance(reason, dict) and str(reason.get("reason") or "").strip()
            ],
            "top_issue_codes": [
                {
                    "code": str(issue.get("code") or ""),
                    "count": _maybe_int(issue.get("count")) or 0,
                }
                for issue in (item.get("top_issue_codes") or [])[:5]
                if isinstance(issue, dict) and str(issue.get("code") or "").strip()
            ],
            "top_live_blocker_codes": [
                {
                    "code": str(issue.get("code") or ""),
                    "count": _maybe_int(issue.get("count")) or 0,
                }
                for issue in (item.get("top_live_blocker_codes") or [])[:5]
                if isinstance(issue, dict) and str(issue.get("code") or "").strip()
            ],
            "top_blockers": [
                {
                    "label": str(blocker.get("label") or ""),
                    "count": _maybe_int(blocker.get("count")) or 0,
                }
                for blocker in (item.get("top_blockers") or [])[:3]
                if isinstance(blocker, dict) and str(blocker.get("label") or "").strip()
            ],
        }
    runner_diag = _coverage_runner_candidate_diagnostics(payload)
    runner = summary.get("RUNNER")
    if runner and runner_diag:
        for key in ("top_issue_codes", "top_live_blocker_codes", "top_blockers"):
            if runner.get(key):
                continue
            values = runner_diag.get(key)
            if isinstance(values, list):
                runner[key] = values[:5]
    return summary


def _coverage_runner_candidate_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = (
        payload.get("runner_candidate_diagnostics")
        if isinstance(payload.get("runner_candidate_diagnostics"), dict)
        else {}
    )
    if not diagnostics:
        return {}
    return {
        "status": str(diagnostics.get("status") or ""),
        "trend_candidate_lanes": _maybe_int(diagnostics.get("trend_candidate_lanes")) or 0,
        "runner_qualified_lanes": _maybe_int(diagnostics.get("runner_qualified_lanes")) or 0,
        "attached_harvest_lanes": _maybe_int(diagnostics.get("attached_harvest_lanes")) or 0,
        "top_demotion_reasons": [
            {
                "reason": str(item.get("reason") or ""),
                "count": _maybe_int(item.get("count")) or 0,
            }
            for item in (diagnostics.get("top_demotion_reasons") or [])[:5]
            if isinstance(item, dict) and str(item.get("reason") or "").strip()
        ],
        "top_issue_codes": [
            {
                "code": str(item.get("code") or ""),
                "count": _maybe_int(item.get("count")) or 0,
            }
            for item in (diagnostics.get("top_issue_codes") or [])[:5]
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        ],
        "top_live_blocker_codes": [
            {
                "code": str(item.get("code") or ""),
                "count": _maybe_int(item.get("count")) or 0,
            }
            for item in (diagnostics.get("top_live_blocker_codes") or [])[:5]
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        ],
        "top_blockers": [
            {
                "label": str(item.get("label") or ""),
                "count": _maybe_int(item.get("count")) or 0,
            }
            for item in (diagnostics.get("top_blockers") or [])[:5]
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        ],
    }


def _coverage_market_evidence_refresh(payload: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = payload.get("artifact_diagnostics") if isinstance(payload.get("artifact_diagnostics"), dict) else {}
    if not diagnostics.get("requires_market_evidence_refresh"):
        return None
    action_items = payload.get("action_items") if isinstance(payload.get("action_items"), list) else []
    return {
        "coverage_status": payload.get("status"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "requires_market_evidence_refresh": True,
        "all_lanes_spread_blocked": bool(diagnostics.get("all_lanes_spread_blocked")),
        "all_lanes_quote_stale": bool(diagnostics.get("all_lanes_quote_stale")),
        "quote_stale_result_count": int(diagnostics.get("quote_stale_result_count") or 0),
        "spread_normalized_candidate_count": int(diagnostics.get("spread_normalized_candidate_count") or 0),
        "spread_normalized_candidate_reward_jpy": _maybe_float(
            diagnostics.get("spread_normalized_candidate_reward_jpy")
        )
        or 0.0,
        "action_items": [str(item) for item in action_items[:4]],
    }


def _coverage_perspective_alignment_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = (
        payload.get("perspective_alignment_diagnostics")
        if isinstance(payload.get("perspective_alignment_diagnostics"), dict)
        else {}
    )
    if not diagnostics:
        return {}
    return {
        "status": str(diagnostics.get("status") or ""),
        "pair_direction_groups": _maybe_int(diagnostics.get("pair_direction_groups")) or 0,
        "range_forecast_method_mismatch_groups": _maybe_int(
            diagnostics.get("range_forecast_method_mismatch_groups")
        )
        or 0,
        "range_forecast_method_mismatch_lanes": _maybe_int(
            diagnostics.get("range_forecast_method_mismatch_lanes")
        )
        or 0,
        "range_forecast_method_mismatch_top": [
            {
                "pair": str(item.get("pair") or ""),
                "direction": str(item.get("direction") or ""),
                "method_mismatch_lanes": _maybe_int(item.get("method_mismatch_lanes")) or 0,
                "method_mismatch_reward_jpy": _maybe_float(item.get("method_mismatch_reward_jpy")) or 0.0,
                "range_rotation_lanes": _maybe_int(item.get("range_rotation_lanes")) or 0,
                "range_rotation_live_ready_lanes": _maybe_int(
                    item.get("range_rotation_live_ready_lanes")
                )
                or 0,
                "range_rotation_absence_reason": str(item.get("range_rotation_absence_reason") or ""),
                "range_rotation_other_side_lanes": _maybe_int(
                    item.get("range_rotation_other_side_lanes")
                )
                or 0,
                "range_rotation_other_side_directions": [
                    {
                        "code": str(row.get("code") or ""),
                        "count": _maybe_int(row.get("count")) or 0,
                    }
                    for row in (item.get("range_rotation_other_side_directions") or [])[:5]
                    if isinstance(row, dict) and str(row.get("code") or "").strip()
                ],
                "range_rotation_top_live_blocker_codes": [
                    {
                        "code": str(blocker.get("code") or ""),
                        "count": _maybe_int(blocker.get("count")) or 0,
                    }
                    for blocker in (item.get("range_rotation_top_live_blocker_codes") or [])[:5]
                    if isinstance(blocker, dict) and str(blocker.get("code") or "").strip()
                ],
                "range_rotation_other_side_top_live_blocker_codes": [
                    {
                        "code": str(blocker.get("code") or ""),
                        "count": _maybe_int(blocker.get("count")) or 0,
                    }
                    for blocker in (item.get("range_rotation_other_side_top_live_blocker_codes") or [])[:5]
                    if isinstance(blocker, dict) and str(blocker.get("code") or "").strip()
                ],
                "range_rotation_other_side_top_blockers": [
                    {
                        "label": str(blocker.get("label") or ""),
                        "count": _maybe_int(blocker.get("count")) or 0,
                    }
                    for blocker in (item.get("range_rotation_other_side_top_blockers") or [])[:4]
                    if isinstance(blocker, dict) and str(blocker.get("label") or "").strip()
                ],
                "top_live_blocker_codes": [
                    {
                        "code": str(blocker.get("code") or ""),
                        "count": _maybe_int(blocker.get("count")) or 0,
                    }
                    for blocker in (item.get("top_live_blocker_codes") or [])[:5]
                    if isinstance(blocker, dict) and str(blocker.get("code") or "").strip()
                ],
            }
            for item in (diagnostics.get("range_forecast_method_mismatch_top") or [])[:8]
            if isinstance(item, dict) and str(item.get("pair") or "").strip()
        ],
    }


def _order_intent_context_evidence_findings(
    *,
    run_id: str,
    intents: dict[str, Any],
    matrix_loaded: _LoadedJson,
    matrix_path: Path,
    target_open: bool,
) -> list[dict[str, Any]]:
    if not target_open or matrix_loaded.error is not None:
        return []
    matrix = matrix_loaded.payload or {}
    pairs = matrix.get("pairs") if isinstance(matrix.get("pairs"), dict) else {}
    if not pairs:
        return []
    results = [item for item in intents.get("results", []) or [] if isinstance(item, dict)]
    if not results:
        return []
    intents_generated_at = _parse_utc(intents.get("generated_at_utc"))
    matrix_generated_at = _parse_utc(matrix.get("generated_at_utc"))
    with_context = [item for item in results if _intent_has_market_context_evidence(item)]
    if (
        intents_generated_at is not None
        and matrix_generated_at is not None
        and intents_generated_at < matrix_generated_at
    ):
        status_counts = _result_status_counts(results)
        live_ready_count = int(status_counts.get("LIVE_READY") or 0)
        priority = "P0" if live_ready_count > 0 else "P1"
        live_ready_clause = (
            f"; {live_ready_count} LIVE_READY lane(s) are tied to the stale context packet"
            if live_ready_count > 0
            else ""
        )
        return [
            _finding(
                run_id=run_id,
                priority=priority,
                layer="opportunity_context",
                code="ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE",
                message=(
                    "order_intents were generated before the current market_context_matrix, so current "
                    "gold/oil/rates/equity/news context cannot be attributed to these candidates"
                    + live_ready_clause
                ),
                next_action=(
                    "Regenerate generate-intents after the latest market-context-matrix, context-asset, and news "
                    "artifacts; do not trust stale LIVE_READY lanes or judge non-FX/news effect from stale candidates."
                ),
                evidence={
                    "matrix_path": str(matrix_path),
                    "intents_generated_at_utc": intents_generated_at.isoformat(),
                    "matrix_generated_at_utc": matrix_generated_at.isoformat(),
                    "matrix_pairs": len(pairs),
                    "candidate_count": len(results),
                    "live_ready_lanes": live_ready_count,
                    "with_context_refs": len(with_context),
                    "status_counts": status_counts,
                },
            )
        ]
    if with_context:
        return []
    return [
        _finding(
            run_id=run_id,
            priority="P1",
            layer="opportunity_context",
            code="ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_MISSING",
            message=(
                "order_intents candidates lack market_context_matrix/news/context-asset refs "
                "while market_context_matrix is available"
            ),
            next_action=(
                "Regenerate generate-intents after market-context-matrix, context-asset, and news refresh "
                "so live entry_thesis can learn gold/oil/rates/equity/news context; do not treat "
                "non-FX/news prediction as backtest-certified until refs persist on entry receipts."
            ),
            evidence={
                "matrix_path": str(matrix_path),
                "matrix_generated_at_utc": matrix.get("generated_at_utc"),
                "matrix_pairs": len(pairs),
                "candidate_count": len(results),
                "with_context_refs": 0,
                "status_counts": _result_status_counts(results),
            },
        )
    ]


def _intent_has_market_context_evidence(result: dict[str, Any]) -> bool:
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    metadata_sources = []
    if isinstance(result.get("metadata"), dict):
        metadata_sources.append(result["metadata"])
    if isinstance(intent.get("metadata"), dict):
        metadata_sources.append(intent["metadata"])
    context_prefixes = ("matrix:", "context_asset:", "news:")
    for metadata in metadata_sources:
        for key, value in metadata.items():
            key_text = str(key)
            if key_text == "market_context_matrix_ref" and value:
                return True
            if key_text.startswith("matrix_") and value not in (None, "", [], {}):
                return True
            if key_text in {"context_refs", "context_asset_refs", "news_refs"}:
                refs = value if isinstance(value, list) else [value]
                if any(str(ref).startswith(context_prefixes) for ref in refs):
                    return True
            if key_text in {"context_asset_symbols", "news_digest_ref"} and value:
                return True
    return False


def _result_status_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _top_count_items(counts: dict[str, int], *, limit: int) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    ]


def _close_gate_loss_evidence(loaded: _LoadedJson, path: Path) -> dict[str, Any]:
    if loaded.error is not None:
        return {}
    payload = loaded.payload or {}
    mechanism = payload.get("mechanism_ablation") if isinstance(payload.get("mechanism_ablation"), dict) else {}
    close_gate = mechanism.get("close_gate_ab") if isinstance(mechanism.get("close_gate_ab"), dict) else {}
    if not close_gate:
        return {}
    evidence = {
        "ai_test_bot_backtest_path": str(path),
        "status": str(close_gate.get("status") or ""),
        "loss_side_market_close_count": int(_maybe_float(close_gate.get("loss_side_market_close_count")) or 0),
        "loss_side_market_close_net_jpy": _maybe_float(close_gate.get("loss_side_market_close_net_jpy")),
        "broker_trade_close_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("broker_trade_close_loss_side_market_close_count")) or 0
        ),
        "broker_trade_close_loss_side_market_close_source_counts": _dict_or_empty(
            close_gate.get("broker_trade_close_loss_side_market_close_source_counts")
        ),
        "broker_accepted_without_gateway_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("broker_accepted_without_gateway_loss_side_market_close_count")) or 0
        ),
        "broker_accepted_without_gateway_loss_side_market_close_net_jpy": _maybe_float(
            close_gate.get("broker_accepted_without_gateway_loss_side_market_close_net_jpy")
        ),
        "broker_accepted_without_gateway_loss_side_market_close_source_counts": _dict_or_empty(
            close_gate.get("broker_accepted_without_gateway_loss_side_market_close_source_counts")
        ),
        "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": _dict_or_empty(
            close_gate.get("broker_accepted_without_gateway_loss_side_market_close_evidence_counts")
        ),
        "broker_accepted_without_gateway_policy_gap_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_count"))
            or 0
        ),
        "broker_accepted_without_gateway_policy_gap_loss_side_market_close_net_jpy": _maybe_float(
            close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_net_jpy")
        ),
        "broker_accepted_without_gateway_policy_gap_loss_side_market_close_source_counts": _dict_or_empty(
            close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_source_counts")
        ),
        "broker_accepted_without_gateway_policy_gap_loss_side_market_close_evidence_counts": _dict_or_empty(
            close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_evidence_counts")
        ),
        "broker_accepted_without_gateway_external_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("broker_accepted_without_gateway_external_loss_side_market_close_count"))
            or 0
        ),
        "broker_accepted_without_gateway_external_loss_side_market_close_net_jpy": _maybe_float(
            close_gate.get("broker_accepted_without_gateway_external_loss_side_market_close_net_jpy")
        ),
        "broker_accepted_without_gateway_external_loss_side_market_close_source_counts": _dict_or_empty(
            close_gate.get("broker_accepted_without_gateway_external_loss_side_market_close_source_counts")
        ),
        "broker_accepted_without_gateway_external_loss_side_market_close_evidence_counts": _dict_or_empty(
            close_gate.get("broker_accepted_without_gateway_external_loss_side_market_close_evidence_counts")
        ),
        "gateway_gpt_close_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("gateway_gpt_close_loss_side_market_close_count")) or 0
        ),
        "gateway_gpt_close_accepted_without_sent_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("gateway_gpt_close_accepted_without_sent_loss_side_market_close_count")) or 0
        ),
        "gateway_review_exit_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("gateway_review_exit_loss_side_market_close_count")) or 0
        ),
        "gateway_review_exit_loss_side_market_close_net_jpy": _maybe_float(
            close_gate.get("gateway_review_exit_loss_side_market_close_net_jpy")
        ),
        "gateway_review_exit_recent_24h_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("gateway_review_exit_recent_24h_loss_side_market_close_count")) or 0
        ),
        "gateway_review_exit_recent_24h_loss_side_market_close_net_jpy": _maybe_float(
            close_gate.get("gateway_review_exit_recent_24h_loss_side_market_close_net_jpy")
        ),
        "gateway_review_exit_recent_7d_loss_side_market_close_count": int(
            _maybe_float(close_gate.get("gateway_review_exit_recent_7d_loss_side_market_close_count")) or 0
        ),
        "gateway_review_exit_recent_7d_loss_side_market_close_net_jpy": _maybe_float(
            close_gate.get("gateway_review_exit_recent_7d_loss_side_market_close_net_jpy")
        ),
        "gateway_review_exit_latest_loss_side_market_close_ts_utc": close_gate.get(
            "gateway_review_exit_latest_loss_side_market_close_ts_utc"
        ),
    }
    meaningful = [
        evidence["loss_side_market_close_count"],
        evidence["broker_trade_close_loss_side_market_close_count"],
        evidence["broker_accepted_without_gateway_loss_side_market_close_count"],
        evidence["gateway_gpt_close_loss_side_market_close_count"],
        evidence["gateway_gpt_close_accepted_without_sent_loss_side_market_close_count"],
        evidence["gateway_review_exit_loss_side_market_close_count"],
        evidence["broker_trade_close_loss_side_market_close_source_counts"],
        evidence["broker_accepted_without_gateway_loss_side_market_close_source_counts"],
        evidence["broker_accepted_without_gateway_loss_side_market_close_evidence_counts"],
    ]
    return evidence if any(meaningful) else {}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _mechanism_ablation_findings(
    *,
    run_id: str,
    loaded: _LoadedJson,
    path: Path,
) -> list[dict[str, Any]]:
    if loaded.error is not None:
        return []
    payload = loaded.payload or {}
    mechanism = payload.get("mechanism_ablation") if isinstance(payload.get("mechanism_ablation"), dict) else {}
    close_gate = mechanism.get("close_gate_ab") if isinstance(mechanism.get("close_gate_ab"), dict) else {}
    if not close_gate:
        return []
    status = str(close_gate.get("status") or "")
    close_events = int(_maybe_float(close_gate.get("close_events")) or 0)
    bot_attributed = int(_maybe_float(close_gate.get("bot_attributed_close_events")) or 0)
    gateway_sent = int(_maybe_float(close_gate.get("gateway_close_sent_events")) or 0)
    broker_accept = int(_maybe_float(close_gate.get("broker_trade_close_accept_events")) or 0)
    broker_without_gateway = int(
        _maybe_float(close_gate.get("broker_accepted_without_gateway_loss_side_market_close_count")) or 0
    )
    broker_without_gateway_sources = (
        close_gate.get("broker_accepted_without_gateway_loss_side_market_close_source_counts")
        if isinstance(close_gate.get("broker_accepted_without_gateway_loss_side_market_close_source_counts"), dict)
        else {}
    )
    broker_without_gateway_evidence = (
        close_gate.get("broker_accepted_without_gateway_loss_side_market_close_evidence_counts")
        if isinstance(close_gate.get("broker_accepted_without_gateway_loss_side_market_close_evidence_counts"), dict)
        else {}
    )
    broker_without_gateway_policy_gap = int(
        _maybe_float(close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_count")) or 0
    )
    broker_without_gateway_policy_sources = (
        close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_source_counts")
        if isinstance(
            close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_source_counts"),
            dict,
        )
        else {}
    )
    broker_without_gateway_policy_evidence = (
        close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_evidence_counts")
        if isinstance(
            close_gate.get("broker_accepted_without_gateway_policy_gap_loss_side_market_close_evidence_counts"),
            dict,
        )
        else {}
    )
    if broker_without_gateway and not broker_without_gateway_policy_sources:
        broker_without_gateway_policy_sources = {
            source: count
            for source, count in broker_without_gateway_sources.items()
            if source not in _EXTERNAL_BROKER_TRADE_CLOSE_SOURCES
        }
        broker_without_gateway_policy_gap = sum(
            int(_maybe_float(count) or 0) for count in broker_without_gateway_policy_sources.values()
        )
        if broker_without_gateway_policy_gap:
            broker_without_gateway_policy_evidence = broker_without_gateway_evidence
    broker_without_gateway_external = int(
        _maybe_float(close_gate.get("broker_accepted_without_gateway_external_loss_side_market_close_count")) or 0
    )
    broker_without_gateway_external_sources = (
        close_gate.get("broker_accepted_without_gateway_external_loss_side_market_close_source_counts")
        if isinstance(
            close_gate.get("broker_accepted_without_gateway_external_loss_side_market_close_source_counts"),
            dict,
        )
        else {}
    )
    if broker_without_gateway and not broker_without_gateway_external_sources:
        broker_without_gateway_external_sources = {
            source: count
            for source, count in broker_without_gateway_sources.items()
            if source in _EXTERNAL_BROKER_TRADE_CLOSE_SOURCES
        }
        broker_without_gateway_external = sum(
            int(_maybe_float(count) or 0) for count in broker_without_gateway_external_sources.values()
        )
    gateway_review_loss = int(_maybe_float(close_gate.get("gateway_review_exit_loss_side_market_close_count")) or 0)
    gateway_review_loss_net = _maybe_float(close_gate.get("gateway_review_exit_loss_side_market_close_net_jpy")) or 0.0
    gateway_review_recent_24h_loss = int(
        _maybe_float(close_gate.get("gateway_review_exit_recent_24h_loss_side_market_close_count")) or 0
    )
    gateway_review_recent_24h_loss_net = (
        _maybe_float(close_gate.get("gateway_review_exit_recent_24h_loss_side_market_close_net_jpy")) or 0.0
    )
    gateway_review_recent_7d_loss = int(
        _maybe_float(close_gate.get("gateway_review_exit_recent_7d_loss_side_market_close_count")) or 0
    )
    gateway_review_recent_7d_loss_net = (
        _maybe_float(close_gate.get("gateway_review_exit_recent_7d_loss_side_market_close_net_jpy")) or 0.0
    )
    if (
        gateway_review_loss
        and "gateway_review_exit_recent_7d_loss_side_market_close_count" not in close_gate
    ):
        gateway_review_recent_7d_loss = gateway_review_loss
        gateway_review_recent_7d_loss_net = gateway_review_loss_net
    if (
        gateway_review_loss
        and "gateway_review_exit_recent_24h_loss_side_market_close_count" not in close_gate
    ):
        gateway_review_recent_24h_loss = gateway_review_loss
        gateway_review_recent_24h_loss_net = gateway_review_loss_net
    gateway_gpt_loss = int(_maybe_float(close_gate.get("gateway_gpt_close_loss_side_market_close_count")) or 0)
    gateway_gpt_accepted_without_sent_loss = int(
        _maybe_float(close_gate.get("gateway_gpt_close_accepted_without_sent_loss_side_market_close_count")) or 0
    )
    if status != "MEASURED" or not close_events:
        return []
    if (
        bot_attributed
        and gateway_sent
        and not broker_without_gateway_policy_gap
        and not (gateway_review_loss and gateway_review_loss_net < 0)
    ):
        return []
    if bot_attributed and broker_accept and not broker_without_gateway_policy_gap and not (
        gateway_review_loss and gateway_review_loss_net < 0
    ):
        return []
    finding_code = "CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE"
    finding_priority = "P1"
    finding_message = (
        "CLOSE Gate A/B performance is not attributable enough to call the gate policy "
        "verified or disproven"
    )
    if broker_without_gateway_policy_gap:
        trader_entry_without_gateway = sum(
            int(_maybe_float(broker_without_gateway_policy_sources.get(label)) or 0)
            for label in ("TRADER_ENTRY_LANE_ID", "TRADER_ENTRY_CLIENT_EXTENSION")
        )
        direct_without_gateway = int(
            _maybe_float(broker_without_gateway_sources.get("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE")) or 0
        )
        if trader_entry_without_gateway:
            next_action = (
                "Broker accepted TRADE_CLOSE loss events are tied to trader-owned entries but lack local "
                "GATEWAY_TRADE_CLOSE_SENT receipts. Repair close receipt persistence/source tags before "
                "ablation; keep direct/manual broker closes separate from Gate A/B evidence."
            )
            if direct_without_gateway:
                next_action += (
                    f" {direct_without_gateway} residual direct/manual close(s) still need external-source "
                    "separation."
                )
        elif broker_without_gateway_policy_sources.get("GATEWAY_GPT_CLOSE_ACCEPTED"):
            next_action = (
                "Accepted GPT CLOSE loss events are missing matching position_execution SENT receipts. Repair "
                "close receipt persistence and then ablate Gate A/B only on fully attributable GPT-close fills."
            )
        else:
            next_action = (
                "Trace broker accepted TRADE_CLOSE orders back to GPT/gateway/operator source and persist a local "
                "GATEWAY_TRADE_CLOSE_SENT or equivalent source tag before relaxing or hardening live close policy."
            )
    elif broker_without_gateway and not bot_attributed:
        direct_without_gateway = int(
            _maybe_float(broker_without_gateway_external_sources.get("DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE")) or 0
        )
        if direct_without_gateway:
            next_action = (
                "Broker accepted TRADE_CLOSE loss events look direct/manual: no local gateway receipt or trader "
                "client extension identified. Treat them as external/direct exit drag, persist explicit close "
                "source tags, and do not use them as Gate A/B or news-weight evidence."
            )
        else:
            next_action = (
                "Trace broker accepted TRADE_CLOSE orders back to GPT/gateway/operator source and persist a local "
                "GATEWAY_TRADE_CLOSE_SENT or equivalent source tag before relaxing or hardening live close policy."
            )
    elif gateway_review_loss and gateway_review_loss_net < 0:
        if gateway_review_recent_24h_loss and gateway_review_recent_24h_loss_net < 0:
            finding_code = "LEGACY_REVIEW_EXIT_CLOSE_DRAG"
            finding_message = (
                "Legacy REVIEW_EXIT market closes are still firing in the last 24h and must stay separated "
                "from current GPT_CLOSE Gate A/B evidence"
            )
            next_action = (
                "Last-24h legacy REVIEW_EXIT loss closes still exist. Keep plain auto-close blocked and replay "
                "structural REVIEW_EXIT timing before any autonomous REVIEW_EXIT path is trusted."
            )
        else:
            finding_code = "LEGACY_REVIEW_EXIT_HISTORICAL_DRAG"
            finding_priority = "P2"
            finding_message = (
                "Legacy REVIEW_EXIT market-close losses are historical and separated from current GPT_CLOSE "
                "Gate A/B evidence"
            )
            next_action = (
                "Keep the historical REVIEW_EXIT loss cluster as audit evidence, but do not let it occupy the "
                "current close-gate P1 slot unless fresh 24h REVIEW_EXIT losses reappear."
            )
    else:
        next_action = (
            "Link gateway close receipts to filled trades and ablate hard Gate A, soft Gate A, "
            "Gate B, and no-gate exit variants offline before relaxing or hardening live close policy."
        )
    if broker_without_gateway_policy_gap or (broker_without_gateway and not bot_attributed):
        finding_code = "CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE"
    return [
        _finding(
            run_id=run_id,
            priority=finding_priority,
            layer="assumption_ablation",
            code=finding_code,
            message=finding_message,
            next_action=next_action,
            evidence={
                "ai_test_bot_backtest_path": str(path),
                "status": status,
                "close_events": close_events,
                "bot_attributed_close_events": bot_attributed,
                "gateway_close_sent_events": gateway_sent,
                "broker_trade_close_accept_events": broker_accept,
                "gateway_gpt_close_loss_side_market_close_count": gateway_gpt_loss,
                "gateway_gpt_close_accepted_without_sent_loss_side_market_close_count": (
                    gateway_gpt_accepted_without_sent_loss
                ),
                "gateway_review_exit_loss_side_market_close_count": gateway_review_loss,
                "gateway_review_exit_loss_side_market_close_net_jpy": gateway_review_loss_net,
                "gateway_review_exit_recent_24h_loss_side_market_close_count": gateway_review_recent_24h_loss,
                "gateway_review_exit_recent_24h_loss_side_market_close_net_jpy": (
                    gateway_review_recent_24h_loss_net
                ),
                "gateway_review_exit_recent_7d_loss_side_market_close_count": gateway_review_recent_7d_loss,
                "gateway_review_exit_recent_7d_loss_side_market_close_net_jpy": gateway_review_recent_7d_loss_net,
                "gateway_review_exit_latest_loss_side_market_close_ts_utc": close_gate.get(
                    "gateway_review_exit_latest_loss_side_market_close_ts_utc"
                ),
                "loss_side_market_close_count": int(_maybe_float(close_gate.get("loss_side_market_close_count")) or 0),
                "loss_side_market_close_net_jpy": _maybe_float(close_gate.get("loss_side_market_close_net_jpy")),
                "broker_trade_close_loss_side_market_close_count": int(
                    _maybe_float(close_gate.get("broker_trade_close_loss_side_market_close_count")) or 0
                ),
                "broker_trade_close_accept_source_counts": close_gate.get("broker_trade_close_accept_source_counts")
                if isinstance(close_gate.get("broker_trade_close_accept_source_counts"), dict)
                else {},
                "broker_trade_close_loss_side_market_close_source_counts": close_gate.get(
                    "broker_trade_close_loss_side_market_close_source_counts"
                )
                if isinstance(close_gate.get("broker_trade_close_loss_side_market_close_source_counts"), dict)
                else {},
                "broker_accepted_without_gateway_loss_side_market_close_count": broker_without_gateway,
                "broker_accepted_without_gateway_loss_side_market_close_source_counts": broker_without_gateway_sources,
                "broker_accepted_without_gateway_loss_side_market_close_evidence_counts": (
                    broker_without_gateway_evidence
                ),
                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_count": (
                    broker_without_gateway_policy_gap
                ),
                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_source_counts": (
                    broker_without_gateway_policy_sources
                ),
                "broker_accepted_without_gateway_policy_gap_loss_side_market_close_evidence_counts": (
                    broker_without_gateway_policy_evidence
                ),
                "broker_accepted_without_gateway_external_loss_side_market_close_count": (
                    broker_without_gateway_external
                ),
                "broker_accepted_without_gateway_external_loss_side_market_close_source_counts": (
                    broker_without_gateway_external_sources
                ),
                "unattributed_loss_side_market_close_count": int(
                    _maybe_float(close_gate.get("unattributed_loss_side_market_close_count")) or 0
                ),
            },
        )
    ]


def _coverage_findings(
    *,
    run_id: str,
    loaded: _LoadedJson,
    path: Path,
    target_open: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not target_open:
        return out
    if loaded.error is not None:
        return out
    payload = loaded.payload or {}
    diagnostics = payload.get("artifact_diagnostics") if isinstance(payload.get("artifact_diagnostics"), dict) else {}
    perspective_diag = _coverage_perspective_alignment_diagnostics(payload)
    if (
        perspective_diag
        and perspective_diag.get("status") == "RANGE_METHOD_MISMATCH_REPAIR_REQUIRED"
        and int(perspective_diag.get("range_forecast_method_mismatch_lanes") or 0) > 0
    ):
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="RANGE_FORECAST_METHOD_MISMATCH_REPAIR_REQUIRED",
                message=(
                    f"{perspective_diag['range_forecast_method_mismatch_lanes']} lane(s) have RANGE "
                    "forecasts surfaced through directional entry methods"
                ),
                next_action=(
                    "Repair these as RANGE_ROTATION rail/phase quality problems before forcing "
                    "always-on directional entries; keep RANGE forecasts from authorizing "
                    "BREAKOUT_FAILURE or TREND_CONTINUATION sends."
                ),
                evidence={
                    "coverage_path": str(path),
                    "perspective_alignment_diagnostics": perspective_diag,
                },
            )
        )
    bucket_diag = (
        diagnostics.get("profitable_bucket_coverage")
        if isinstance(diagnostics.get("profitable_bucket_coverage"), dict)
        else {}
    )
    if not bucket_diag:
        return out
    blocked_edges = _blocked_profitable_bucket_edges(bucket_diag)
    forecast_gated_edges = [edge for edge in blocked_edges if _edge_is_forecast_gated(edge)]
    strategy_gated_edges = [
        edge for edge in blocked_edges
        if not _edge_is_forecast_gated(edge) and _edge_is_strategy_gated(edge)
    ]
    coverage_repair_edges = [
        edge for edge in blocked_edges
        if not _edge_is_forecast_gated(edge) and not _edge_is_strategy_gated(edge)
    ]
    if coverage_repair_edges:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="opportunity",
                code="PROFITABLE_BACKTEST_EDGE_COVERAGE_GAP",
                message=(
                    f"{len(blocked_edges)} profitable backtest edge(s) are missing or blocked "
                    "in current candidate coverage"
                ),
                next_action=(
                    "Repair the named historical-profitable pair/directions in strategy_profile, "
                    "candidate generation, or live blockers before widening the discovery universe."
                ),
                evidence={
                    "coverage_path": str(path),
                    "source_status": bucket_diag.get("source_status"),
                    "live_permission": bool(bucket_diag.get("live_permission") is True),
                    "positive_pair_directions": bucket_diag.get("positive_pair_directions"),
                    "positive_managed_net_jpy": bucket_diag.get("positive_managed_net_jpy"),
                    "state_counts": bucket_diag.get("state_counts") or {},
                    "blocked_edges": coverage_repair_edges[:8],
                },
            )
        )
    if forecast_gated_edges:
        out.append(
            _finding(
                run_id=run_id,
                priority="P2",
                layer="forecast",
                code="PROFITABLE_BACKTEST_EDGE_FORECAST_GATED",
                message=(
                    f"{len(forecast_gated_edges)} profitable backtest edge(s) are visible, "
                    "but current forecast gates block live expansion"
                ),
                next_action=(
                    "Treat these as forecast-repair evidence, not live coverage expansion; keep "
                    "RiskEngine and forecast-confidence gates intact until the current prediction "
                    "packet supports the side."
                ),
                evidence={
                    "coverage_path": str(path),
                    "forecast_gated_edges": forecast_gated_edges[:8],
                },
            )
        )
    if strategy_gated_edges:
        out.append(
            _finding(
                run_id=run_id,
                priority="P2",
                layer="opportunity",
                code="PROFITABLE_BACKTEST_EDGE_STRATEGY_GATED",
                message=(
                    f"{len(strategy_gated_edges)} profitable backtest edge(s) remain blocked by "
                    "strategy-profile repair gates"
                ),
                next_action=(
                    "Do not amplify these historical buckets until a current risk-resized dry-run "
                    "receipt or new market-structure proof reopens the strategy profile gate."
                ),
                evidence={
                    "coverage_path": str(path),
                    "strategy_gated_edges": strategy_gated_edges[:8],
                },
            )
        )
    context_supported = [
        edge for edge in coverage_repair_edges
        if _edge_has_same_side_matrix_support(edge)
    ]
    if context_supported:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="opportunity",
                code="MARKET_CONTEXT_SUPPORTED_EDGE_NOT_ACTIONABLE",
                message=(
                    f"{len(context_supported)} blocked profitable edge(s) have same-side "
                    "market-context matrix support"
                ),
                next_action=(
                    "Use the matrix-supported edges as the next discovery repair queue, but keep "
                    "forecast confidence, spread, strategy-profile, RiskEngine, and gateway gates intact."
                ),
                evidence={
                    "coverage_path": str(path),
                    "supported_edges": context_supported[:8],
                },
            )
        )
    return out


def _blocked_profitable_bucket_edges(bucket_diag: dict[str, Any]) -> list[dict[str, Any]]:
    rows = bucket_diag.get("blocked_or_missing_top")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        state = str(row.get("coverage_state") or "")
        if state == "SPREAD_NORMALIZED_NO_LIVE_BLOCKER":
            continue
        edge = {
            "pair": str(row.get("pair") or ""),
            "direction": str(row.get("direction") or ""),
            "coverage_state": state,
            "managed_net_jpy": _maybe_float(row.get("managed_net_jpy")),
            "raw_net_jpy": _maybe_float(row.get("raw_net_jpy")),
            "trades": row.get("trades"),
            "days": row.get("days"),
            "current_lane_count": row.get("current_lane_count"),
            "spread_normalized_candidate_count": row.get("spread_normalized_candidate_count"),
            "spread_normalized_no_live_blocker_count": row.get("spread_normalized_no_live_blocker_count"),
            "top_blockers": [str(item) for item in (row.get("top_blockers") or [])[:4]],
            "strategy_profile_status": row.get("strategy_profile_status"),
            "strategy_profile_required_fix": row.get("strategy_profile_required_fix"),
            "strategy_profile_blocks_live": row.get("strategy_profile_blocks_live"),
            "matrix_support_count": int(row.get("matrix_support_count") or 0),
            "matrix_reject_count": int(row.get("matrix_reject_count") or 0),
            "matrix_warning_count": int(row.get("matrix_warning_count") or 0),
            "matrix_strongest_support": row.get("matrix_strongest_support"),
            "matrix_strongest_reject": row.get("matrix_strongest_reject"),
            "matrix_cross_asset_context": [
                str(item) for item in (row.get("matrix_cross_asset_context") or [])[:4] if str(item).strip()
            ],
            "matrix_support_context": [
                str(item) for item in (row.get("matrix_support_context") or [])[:4] if str(item).strip()
            ],
            "matrix_reject_context": [
                str(item) for item in (row.get("matrix_reject_context") or [])[:4] if str(item).strip()
            ],
            "same_side_matrix_context_supported": bool(row.get("same_side_matrix_context_supported")),
        }
        out.append(edge)
    out.sort(
        key=lambda item: (
            -float(item.get("managed_net_jpy") or 0.0),
            str(item.get("pair") or ""),
            str(item.get("direction") or ""),
        )
    )
    return out


def _edge_has_same_side_matrix_support(edge: dict[str, Any]) -> bool:
    if edge.get("same_side_matrix_context_supported") is True:
        return True
    support_count = int(edge.get("matrix_support_count") or 0)
    reject_count = int(edge.get("matrix_reject_count") or 0)
    support_context = edge.get("matrix_support_context")
    has_support_context = isinstance(support_context, list) and bool(support_context)
    return support_count > 0 and support_count > reject_count and has_support_context


def _edge_is_forecast_gated(edge: dict[str, Any]) -> bool:
    blockers = [str(item).upper() for item in edge.get("top_blockers") or []]
    return any(
        "FORECAST_" in blocker
        or " CURRENT PAIR FORECAST" in blocker
        or "FORECAST TELEMETRY" in blocker
        or "WEAK PREDICTION" in blocker
        for blocker in blockers
    )


def _edge_is_strategy_gated(edge: dict[str, Any]) -> bool:
    status = str(edge.get("strategy_profile_status") or "").upper()
    return bool(edge.get("strategy_profile_blocks_live") is True and status == "BLOCK_UNTIL_NEW_EVIDENCE")


def _sidecar_findings(
    *,
    run_id: str,
    snapshot_ts: datetime | None,
    active_trade_ids: set[str],
    gpt_decision: dict[str, Any],
    sidecars: dict[str, tuple[_LoadedJson, Path]],
    defer_stale_judgment: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not active_trade_ids:
        return out
    for name, (loaded, path) in sidecars.items():
        if loaded.error is not None:
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P0" if name == "position_management" else "P1",
                    layer="position_review",
                    code=f"{name.upper()}_UNREADABLE",
                    message=f"{name} sidecar is unreadable for active positions: {path}: {loaded.error}",
                    next_action=f"Regenerate {name} before trusting open-position management.",
                )
            )
            continue
        payload = loaded.payload or {}
        generated_at = _parse_utc(payload.get("generated_at_utc"))
        source_snapshot_at = _parse_utc(payload.get("snapshot_fetched_at_utc"))
        freshness_at = source_snapshot_at or generated_at
        if (
            not defer_stale_judgment
            and snapshot_ts is not None
            and freshness_at is not None
            and freshness_at < snapshot_ts
        ):
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P0" if name == "position_management" else "P1",
                    layer="position_review",
                    code=f"{name.upper()}_STALE",
                    message=f"{name} sidecar predates broker snapshot for active positions",
                    next_action=f"Rerun {name.replace('_', '-')} or route to the refresh branch before new exposure.",
                    evidence={
                        "generated_at_utc": generated_at.isoformat() if generated_at else None,
                        "sidecar_snapshot_fetched_at_utc": source_snapshot_at.isoformat() if source_snapshot_at else None,
                        "snapshot_fetched_at_utc": snapshot_ts.isoformat(),
                    },
                )
            )
    close_refs = _close_recommendations(sidecars, active_trade_ids)
    gpt_action = ""
    if isinstance(gpt_decision.get("decision"), dict):
        gpt_action = str(gpt_decision["decision"].get("action") or "").upper()
    gpt_status = str(gpt_decision.get("status") or "").upper()
    if close_refs and not (gpt_status == "ACCEPTED" and gpt_action == "CLOSE"):
        auth_blocker = _active_close_operator_auth_blocker(gpt_decision, active_trade_ids)
        if auth_blocker:
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P1",
                    layer="position_review",
                    code="OPEN_POSITION_CLOSE_OPERATOR_AUTH_REQUIRED",
                    message=(
                        f"{len(close_refs)} active position close/review signal(s) reached GPT, "
                        "but verifier requires explicit operator Gate B"
                    ),
                    next_action=(
                        "Do not send a live close from automation. Refresh sidecars on the next broker "
                        "snapshot, or provide the explicit operator close token/env only if this loss-side "
                        "close is intentionally authorized."
                    ),
                    evidence={
                        "signals": close_refs[:20],
                        "gpt_status": gpt_status,
                        "gpt_action": gpt_action,
                        "blocking_codes": auth_blocker["codes"],
                        "active_close_trade_ids": auth_blocker["active_close_trade_ids"],
                    },
                )
            )
            return out
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="position_review",
                code="OPEN_POSITION_CLOSE_EVIDENCE_UNRESOLVED",
                message=f"{len(close_refs)} active position close/review signal(s) are unresolved by the latest GPT decision",
                next_action="If evidence is hard Gate A, submit a CLOSE receipt; if soft, keep it explicit and avoid hiding it behind new entries.",
                evidence={"signals": close_refs[:20], "gpt_status": gpt_status, "gpt_action": gpt_action},
            )
        )
    return out


def _active_close_operator_auth_blocker(
    gpt_decision: dict[str, Any],
    active_trade_ids: set[str],
) -> dict[str, Any] | None:
    if str(gpt_decision.get("status") or "").upper() != "REJECTED":
        return None
    decision = gpt_decision.get("decision") if isinstance(gpt_decision.get("decision"), dict) else {}
    if str(decision.get("action") or "").upper() != "CLOSE":
        return None
    close_ids = {str(item) for item in decision.get("close_trade_ids", []) or [] if str(item)}
    active_close_ids = close_ids & active_trade_ids
    if not active_close_ids:
        return None
    issues = gpt_decision.get("verification_issues")
    if not isinstance(issues, list):
        return None
    codes = [
        str(item.get("code") or "")
        for item in issues
        if isinstance(item, dict) and str(item.get("severity") or "").upper() == "BLOCK"
    ]
    if "CLOSE_OPERATOR_AUTH_REQUIRED" not in codes:
        return None
    return {"codes": codes[:12], "active_close_trade_ids": sorted(active_close_ids)}


def _decision_artifact_findings(
    *,
    run_id: str,
    db_path: Path,
    gpt_loaded: _LoadedJson,
    trader_loaded: _LoadedJson,
    gpt_path: Path,
    trader_path: Path,
    target_open: bool,
    active_trade_ids: set[str],
    active_position_sides: dict[str, str] | None,
    live_ready_lanes: int,
    pending_entry_orders: int,
    snapshot_ts: datetime | None,
    pending_entry_order_details: list[dict[str, Any]] | None = None,
    latest_gpt_stale_streak_before: int = 0,
    sidecars: dict[str, tuple[_LoadedJson, Path]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if gpt_loaded.error is not None:
        out.append(
            _finding(
                run_id=run_id,
                priority="P0" if target_open else "P1",
                layer="decision_history",
                code="GPT_DECISION_UNREADABLE",
                message=f"GPT decision audit artifact is unreadable: {gpt_path}: {gpt_loaded.error}",
                next_action="Recreate the decision receipt path before any gateway handoff.",
            )
        )
    # `DEFAULT_TRADER_DECISION` is the legacy deterministic prefilter artifact.
    # The live decision contract now hinges on the GPT/codex receipt; a missing
    # legacy comparison file is only actionable when the current receipt is also
    # unavailable.
    if trader_loaded.error is not None and gpt_loaded.payload is None:
        out.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="decision_history",
                code="TRADER_DECISION_UNREADABLE",
                message=f"deterministic trader decision artifact is unreadable: {trader_path}: {trader_loaded.error}",
                next_action="Regenerate trader_decision so GPT and deterministic paths can be compared.",
            )
        )
    if gpt_loaded.payload is not None:
        payload = gpt_loaded.payload
        generated_at = _parse_utc(payload.get("generated_at_utc"))
        if generated_at is None and isinstance(payload.get("decision"), dict):
            generated_at = _parse_utc(payload["decision"].get("generated_at_utc"))
        issues = payload.get("verification_issues")
        blocking = (
            [
                item
                for item in issues
                if isinstance(item, dict) and str(item.get("severity") or "").upper() == "BLOCK"
            ]
            if isinstance(issues, list)
            else []
        )
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        action = str(decision.get("action") or "").upper()
        close_ids = {
            str(item)
            for item in decision.get("close_trade_ids", []) or []
            if str(item)
        }
        status = str(payload.get("status") or "").upper()
        rejected_inert_receipt = bool(blocking) and status == "REJECTED" and not (
            action == "CLOSE" and close_ids & active_trade_ids
        )
        stale_close_for_closed_trades = bool(blocking) and (
            action == "CLOSE" and bool(close_ids) and close_ids.isdisjoint(active_trade_ids)
        )
        accepted_inert_no_risk_decision = (
            status == "ACCEPTED"
            and action in {"WAIT", "NO_TRADE", "REQUEST_EVIDENCE"}
            and not active_trade_ids
            and live_ready_lanes <= 0
            and pending_entry_orders <= 0
        )
        accepted_trade_consumed_by_pending_entry = (
            status == "ACCEPTED"
            and action == "TRADE"
            and not blocking
            and _accepted_trade_has_current_pending_entry(decision, pending_entry_order_details or [])
        )
        accepted_decision_consumed_by_gateway = (
            status == "ACCEPTED"
            and not blocking
            and _accepted_gpt_decision_consumed_by_gateway(
                db_path=db_path,
                generated_at=generated_at,
                snapshot_ts=snapshot_ts,
                action=action,
                decision=decision,
            )
        )
        if (
            snapshot_ts is not None
            and generated_at is not None
            and generated_at < snapshot_ts
            and not rejected_inert_receipt
            and not stale_close_for_closed_trades
            and not accepted_trade_consumed_by_pending_entry
            and not accepted_decision_consumed_by_gateway
        ):
            priority = (
                "P0"
                if active_trade_ids
                or (
                    target_open
                    and live_ready_lanes <= 0
                    and pending_entry_orders <= 0
                    and not accepted_inert_no_risk_decision
                )
                else "P1"
            )
            out.append(
                _finding(
                    run_id=run_id,
                    priority=priority,
                    layer="decision_history",
                    code="LATEST_GPT_DECISION_STALE",
                    message="latest GPT decision receipt predates the current broker snapshot",
                    next_action=(
                        "Re-verify or rewrite the GPT decision against the current broker snapshot before "
                        "trusting WAIT, CLOSE, CANCEL, or TRADE routing."
                    ),
                    evidence={
                        "generated_at_utc": generated_at.isoformat(),
                        "snapshot_fetched_at_utc": snapshot_ts.isoformat(),
                        "live_ready_lanes": live_ready_lanes,
                        "pending_entry_orders": pending_entry_orders,
                        "current_streak": max(1, latest_gpt_stale_streak_before + 1),
                        "previous_streak": max(0, latest_gpt_stale_streak_before),
                    },
                )
            )
        if blocking:
            if action == "CLOSE" and close_ids and close_ids.isdisjoint(active_trade_ids):
                out.append(
                    _finding(
                        run_id=run_id,
                        priority="P1",
                        layer="decision_history",
                        code="STALE_GPT_CLOSE_BLOCKERS_FOR_CLOSED_TRADES",
                        message=(
                            "latest GPT CLOSE decision has blocking verification issue(s), "
                            "but broker truth no longer shows the requested trader-owned trade(s) as open"
                        ),
                        next_action=(
                            "Do not reuse the stale CLOSE receipt; refresh broker truth and continue with "
                            "current position/entry routing."
                        ),
                        evidence={
                            "codes": [str(item.get("code") or "") for item in blocking[:12]],
                            "close_trade_ids": sorted(close_ids),
                        },
                    )
                )
                return out
            if status == "REJECTED" and not (action == "CLOSE" and close_ids & active_trade_ids):
                out.append(
                    _finding(
                        run_id=run_id,
                        priority="P1",
                        layer="decision_history",
                        code="LATEST_GPT_DECISION_REJECTED_WITH_BLOCKERS",
                        message=(
                            "latest GPT decision was already rejected with blocking verification issue(s); "
                            "it is not an unconsumed live permission"
                        ),
                        next_action=(
                            "Do not reuse the rejected receipt; write and verify a fresh decision against "
                            "the current packet."
                        ),
                        evidence={
                            "status": status,
                            "action": action,
                            "codes": [str(item.get("code") or "") for item in blocking[:12]],
                            "active_close_trade_ids": sorted(close_ids & active_trade_ids),
                        },
                    )
                )
                return out
            active_close_ids = close_ids & active_trade_ids
            if _rejected_close_deferred_by_liquidity(blocking, active_close_ids):
                out.append(
                    _finding(
                        run_id=run_id,
                        priority="P1",
                        layer="decision_history",
                        code="LATEST_GPT_CLOSE_DEFERRED_BY_LIQUIDITY",
                        message=(
                            "latest GPT CLOSE decision was rejected only by deterministic close-spread "
                            "liquidity gates for still-open trader-owned trade(s)"
                        ),
                        next_action=(
                            "Do not reuse or override the rejected CLOSE receipt. Refresh broker truth and "
                            "position sidecars on the next cycle, then verify a fresh CLOSE only if the "
                            "thesis remains broken and close spread has normalized."
                        ),
                        evidence={
                            "codes": [str(item.get("code") or "") for item in blocking[:12]],
                            "active_close_trade_ids": sorted(active_close_ids),
                        },
                    )
                )
                return out
            if _rejected_close_is_nonblocking_soft_advisory(
                blocking,
                active_close_ids,
                active_position_sides or {},
                sidecars or {},
            ):
                out.append(
                    _finding(
                        run_id=run_id,
                        priority="P1",
                        layer="decision_history",
                        code="LATEST_GPT_DECISION_SOFT_CLOSE_ADVISORY_REJECTED",
                        message=(
                            "latest GPT CLOSE decision was rejected from non-blocking soft close advisory "
                            "evidence while same-direction HOLD/EXTEND sidecars still support the open trade"
                        ),
                        next_action=(
                            "Do not reuse the rejected CLOSE receipt; write and verify a fresh entry-branch "
                            "TRADE/CANCEL/WAIT receipt against the current packet."
                        ),
                        evidence={
                            "codes": [str(item.get("code") or "") for item in blocking[:12]],
                            "active_close_trade_ids": sorted(active_close_ids),
                        },
                    )
                )
                return out
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P0",
                    layer="decision_history",
                    code="LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES",
                    message=f"latest GPT decision contains {len(blocking)} blocking verification issue(s)",
                    next_action="Do not reuse the receipt; fix the first blocker and verify a fresh decision.",
                    evidence={
                        "codes": [str(item.get("code") or "") for item in blocking[:12]],
                        "active_close_trade_ids": sorted(close_ids & active_trade_ids),
                    },
                )
            )
    return out


def _rejected_close_deferred_by_liquidity(
    blocking: list[dict[str, Any]],
    active_close_ids: set[str],
) -> bool:
    if not active_close_ids:
        return False
    codes = {
        str(item.get("code") or "")
        for item in blocking
        if isinstance(item, dict) and str(item.get("severity") or "").upper() == "BLOCK"
    }
    if not codes:
        return False
    return codes <= {
        "POSITION_CLOSE_SPREAD_TOO_WIDE",
        "POSITION_CLOSE_FLOW_SPREAD_TOO_WIDE",
    }


def _rejected_close_is_nonblocking_soft_advisory(
    blocking: list[dict[str, Any]],
    active_close_ids: set[str],
    active_position_sides: dict[str, str],
    sidecars: dict[str, tuple[_LoadedJson, Path]],
) -> bool:
    if not active_close_ids:
        return False
    codes = {
        str(item.get("code") or "")
        for item in blocking
        if isinstance(item, dict) and str(item.get("severity") or "").upper() == "BLOCK"
    }
    allowed_codes = {
        "CLOSE_OPERATOR_AUTH_REQUIRED",
        "SOFT_CLOSE_ADVISORY_DOES_NOT_PREEMPT_ENTRY",
    }
    if "CLOSE_OPERATOR_AUTH_REQUIRED" not in codes:
        return False
    if any(code not in allowed_codes for code in codes):
        return False
    return all(
        _trade_has_same_direction_hold_sidecar_support(sidecars, trade_id, active_position_sides.get(trade_id, ""))
        for trade_id in active_close_ids
    )


def _trade_has_same_direction_hold_sidecar_support(
    sidecars: dict[str, tuple[_LoadedJson, Path]],
    trade_id: str,
    active_side: str,
) -> bool:
    if active_side not in {"LONG", "SHORT"}:
        return False
    support_sources: set[str] = set()
    support_tokens = ("HOLD", "EXTEND", "STILL_VALID", "CONTINUE", "CARRY")
    for name, (loaded, _path) in sidecars.items():
        if name == "position_management":
            continue
        payload = loaded.payload or {}
        for key in ("evolutions", "assessments", "verdicts"):
            rows = payload.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("trade_id") or "") != str(trade_id):
                    continue
                if str(row.get("side") or "").upper() != active_side:
                    continue
                status_text = " ".join(
                    str(row.get(field) or "").upper()
                    for field in ("action", "status", "verdict", "reason", "rationale")
                )
                if any(token in status_text for token in support_tokens):
                    support_sources.add(name)
                    break
            if name in support_sources:
                break
    return len(support_sources) >= 2


def _effect_metrics(db_path: Path, *, window_hours: float, now: datetime) -> dict[str, Any]:
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    rows: list[sqlite3.Row] = []
    accepted_close_rows: list[sqlite3.Row] = []
    gateway_close_trade_ids: set[str] = set()
    gateway_close_order_ids: set[str] = set()
    reconciled_close_trade_ids: set[str] = set()
    reconciled_close_order_ids: set[str] = set()
    gpt_close_trade_ids: set[str] = set()
    close_gate_pass_trade_ids: set[str] = set()
    stale_close_satisfied_trade_ids: set[str] = set()
    error: str | None = None
    if db_path.exists():
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                columns = _table_columns(conn, "execution_events")
                has_trade_id = "trade_id" in columns
                has_lane_id = "lane_id" in columns
                if has_trade_id:
                    gateway_close_trade_ids = _event_trade_ids(conn, "GATEWAY_TRADE_CLOSE_SENT")
                    reconciled_close_trade_ids = _event_trade_ids(conn, "GATEWAY_TRADE_CLOSE_RECONCILED")
                    gpt_close_trade_ids = _event_trade_ids(conn, "GATEWAY_GPT_CLOSE_ACCEPTED")
                    if "order_id" in columns:
                        gateway_close_order_ids = _event_order_ids(conn, "GATEWAY_TRADE_CLOSE_SENT")
                        reconciled_close_order_ids = _event_order_ids(conn, "GATEWAY_TRADE_CLOSE_RECONCILED")
                    close_gate_pass_trade_ids = _close_gate_pass_trade_ids(conn)
                    stale_close_satisfied_trade_ids = _stale_gpt_close_satisfied_trade_ids(conn, columns)
                    accepted_close_rows = _broker_trade_close_accept_rows(conn, columns)
                    lane_select = "e.lane_id AS lane_id" if has_lane_id else "NULL AS lane_id"
                    open_lane_expr = "MAX(NULLIF(lane_id, ''))" if has_lane_id else "NULL"
                    entry_price_expr = "MAX(price)" if "price" in columns else "NULL"
                    if "sl" in columns:
                        stop_loss_expr = "MAX(sl)"
                    elif "price" in columns:
                        stop_loss_expr = "MAX(price)"
                    else:
                        stop_loss_expr = "NULL"
                    close_price_select = "e.price AS close_price" if "price" in columns else "NULL AS close_price"
                    query = f"""
                        WITH open_fills AS (
                            SELECT
                                trade_id,
                                {open_lane_expr} AS open_lane_id,
                                {entry_price_expr} AS entry_price
                            FROM execution_events
                            WHERE event_type = 'ORDER_FILLED'
                              AND COALESCE(trade_id, '') <> ''
                            GROUP BY trade_id
                        ),
                        protection_sls AS (
                            SELECT
                                trade_id,
                                {stop_loss_expr} AS stop_loss_price
                            FROM execution_events
                            WHERE event_type = 'PROTECTION_CREATED'
                              AND COALESCE(trade_id, '') <> ''
                            GROUP BY trade_id
                        )
                        SELECT
                            e.ts_utc,
                            e.pair,
                            e.side,
                            e.realized_pl_jpy,
                            e.trade_id,
                            {lane_select},
                            open_fills.open_lane_id,
                            open_fills.entry_price,
                            protection_sls.stop_loss_price,
                            {close_price_select},
                            {_sql_column_or_null(columns, "order_id")},
                            {_sql_column_or_null(columns, "exit_reason")},
                            {_sql_column_or_null(columns, "raw_json")}
                        FROM execution_events e
                        LEFT JOIN open_fills ON open_fills.trade_id = e.trade_id
                        LEFT JOIN protection_sls ON protection_sls.trade_id = e.trade_id
                        WHERE e.event_type = 'TRADE_CLOSED'
                          AND e.realized_pl_jpy IS NOT NULL
                    """
                else:
                    query = f"""
                        SELECT
                            ts_utc,
                            pair,
                            side,
                            realized_pl_jpy,
                            NULL AS trade_id,
                            NULL AS lane_id,
                            NULL AS open_lane_id,
                            NULL AS entry_price,
                            NULL AS stop_loss_price,
                            NULL AS close_price,
                            {_sql_column_or_null(columns, "order_id", qualifier="")},
                            {_sql_column_or_null(columns, "exit_reason", qualifier="")},
                            {_sql_column_or_null(columns, "raw_json", qualifier="")}
                        FROM execution_events
                        WHERE event_type = 'TRADE_CLOSED'
                          AND realized_pl_jpy IS NOT NULL
                    """
                rows = list(conn.execute(query))
        except sqlite3.Error as exc:
            error = str(exc)
    else:
        error = "missing"
    pls: list[float] = []
    segments: dict[tuple[str, str, str], dict[str, Any]] = {}
    close_provenance_metrics: dict[str, dict[str, Any]] = {}
    market_order_trade_close_loss_provenance_metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        ts = _parse_utc(row["ts_utc"])
        if ts is None or ts < cutoff:
            continue
        value = _maybe_float(row["realized_pl_jpy"])
        if value is None:
            continue
        pls.append(value)
        lane_id = _row_text(row, "lane_id") or _row_text(row, "open_lane_id")
        method = _method_from_lane_id(lane_id) or "UNKNOWN"
        key = (str(row["pair"] or "UNKNOWN"), str(row["side"] or "UNKNOWN"), method)
        close_provenance = _close_provenance_for_effect_row(
            row,
            accepted_close_rows=accepted_close_rows,
            gateway_close_trade_ids=gateway_close_trade_ids,
            gateway_close_order_ids=gateway_close_order_ids,
            reconciled_close_trade_ids=reconciled_close_trade_ids,
            reconciled_close_order_ids=reconciled_close_order_ids,
            gpt_close_trade_ids=gpt_close_trade_ids,
            stale_close_satisfied_trade_ids=stale_close_satisfied_trade_ids,
        )
        exit_reason = str(_row_text(row, "exit_reason") or "").strip().upper()
        contained_loss = False
        avoided_loss_jpy: float | None = None
        if exit_reason == "MARKET_ORDER_TRADE_CLOSE" and value < 0:
            trade_id = str(_row_text(row, "trade_id") or "").strip()
            contained_loss, avoided_loss_jpy = _loss_close_containment(
                row,
                value,
                close_provenance,
                close_gate_evidence_passed=trade_id in close_gate_pass_trade_ids,
            )
        _add_close_provenance_metric(
            close_provenance_metrics,
            close_provenance,
            value,
            contained_loss=contained_loss,
            avoided_loss_jpy=avoided_loss_jpy,
        )
        if exit_reason == "MARKET_ORDER_TRADE_CLOSE" and value < 0:
            _add_close_provenance_metric(
                market_order_trade_close_loss_provenance_metrics,
                close_provenance,
                value,
                contained_loss=contained_loss,
                avoided_loss_jpy=avoided_loss_jpy,
            )
        segment = segments.setdefault(
            key,
            {
                "values": [],
                "lane_ids": set(),
                "trade_ids": set(),
                "close_provenance_counts": {},
                "close_provenance_net_jpy": {},
            },
        )
        segment["values"].append(value)
        segment["close_provenance_counts"][close_provenance] = (
            int(segment["close_provenance_counts"].get(close_provenance, 0)) + 1
        )
        segment["close_provenance_net_jpy"][close_provenance] = (
            float(segment["close_provenance_net_jpy"].get(close_provenance, 0.0)) + value
        )
        if lane_id:
            segment["lane_ids"].add(lane_id)
        trade_id = _row_text(row, "trade_id")
        if trade_id:
            segment["trade_ids"].add(trade_id)
    gross_profit = sum(value for value in pls if value > 0)
    losses = [value for value in pls if value < 0]
    gross_loss_abs = abs(sum(losses))
    count = len(pls)
    wins = [value for value in pls if value > 0]
    net = sum(pls)
    worst_segments = [
        {
            "pair": pair,
            "side": side,
            "method": method,
            "trades": len(segment["values"]),
            "net_jpy": sum(segment["values"]),
            "expectancy_jpy": sum(segment["values"]) / len(segment["values"]),
            "lane_ids": sorted(segment["lane_ids"])[:5],
            "trade_ids": sorted(segment["trade_ids"])[:10],
            "close_provenance_counts": dict(sorted(segment["close_provenance_counts"].items())),
            "close_provenance_net_jpy": dict(sorted(segment["close_provenance_net_jpy"].items())),
        }
        for (pair, side, method), segment in segments.items()
    ]
    worst_segments.sort(key=lambda item: float(item["net_jpy"]))
    market_close_loss_net = sum(
        float(item.get("net_jpy") or 0.0)
        for item in market_order_trade_close_loss_provenance_metrics.values()
    )
    market_close_loss_trades = sum(
        int(item.get("trades") or 0)
        for item in market_order_trade_close_loss_provenance_metrics.values()
    )
    return {
        "closed_trades": count,
        "net_jpy": net,
        "gross_profit_jpy": gross_profit,
        "gross_loss_jpy": gross_loss_abs,
        "profit_factor": (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else None,
        "win_rate": (len(wins) / count) if count else None,
        "expectancy_jpy": (net / count) if count else None,
        "avg_win_jpy": (sum(wins) / len(wins)) if wins else None,
        "avg_loss_jpy_abs": (abs(sum(losses)) / len(losses)) if losses else None,
        "worst_segments": worst_segments[:10],
        "close_provenance_metrics": _sorted_close_provenance_metrics(close_provenance_metrics),
        "market_order_trade_close_loss_trades": market_close_loss_trades,
        "market_order_trade_close_loss_net_jpy": market_close_loss_net,
        "market_order_trade_close_loss_provenance_metrics": _sorted_close_provenance_metrics(
            market_order_trade_close_loss_provenance_metrics
        ),
        "error": error,
    }


def _sql_column_or_null(columns: set[str], column: str, *, qualifier: str = "e") -> str:
    if column not in columns:
        return f"NULL AS {column}"
    if qualifier:
        return f"{qualifier}.{column} AS {column}"
    return column


def _event_trade_ids(conn: sqlite3.Connection, event_type: str) -> set[str]:
    try:
        return {
            str(row[0]).strip()
            for row in conn.execute(
                """
                SELECT DISTINCT trade_id
                FROM execution_events
                WHERE event_type = ?
                  AND trade_id IS NOT NULL
                  AND trade_id != ''
                """,
                (event_type,),
            )
            if str(row[0]).strip()
        }
    except sqlite3.Error:
        return set()


def _event_order_ids(conn: sqlite3.Connection, event_type: str) -> set[str]:
    try:
        return {
            str(row[0]).strip()
            for row in conn.execute(
                """
                SELECT DISTINCT order_id
                FROM execution_events
                WHERE event_type = ?
                  AND order_id IS NOT NULL
                  AND order_id != ''
                """,
                (event_type,),
            )
            if str(row[0]).strip()
        }
    except sqlite3.Error:
        return set()


def _close_gate_pass_trade_ids(conn: sqlite3.Connection) -> set[str]:
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
    except sqlite3.Error:
        return set()
    if "verification_observations" not in tables:
        return set()
    try:
        return {
            str(row[0]).strip()
            for row in conn.execute(
                """
                SELECT DISTINCT g.trade_id
                FROM execution_events g
                INNER JOIN verification_observations v
                  ON v.check_name = 'close_gate_evidence'
                 AND v.status = 'PASS'
                 AND v.subject_id = g.trade_id
                WHERE g.event_type IN ('GATEWAY_TRADE_CLOSE_SENT', 'GATEWAY_GPT_CLOSE_ACCEPTED')
                  AND COALESCE(g.trade_id, '') != ''
                  AND (
                      v.ts_utc = g.ts_utc
                      OR EXISTS (
                          SELECT 1
                          FROM execution_events accepted
                          WHERE accepted.event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
                            AND accepted.trade_id = g.trade_id
                            AND accepted.ts_utc = v.ts_utc
                            AND accepted.ts_utc <= g.ts_utc
                            AND NOT EXISTS (
                                SELECT 1
                                FROM execution_events newer_accepted
                                WHERE newer_accepted.event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
                                  AND newer_accepted.trade_id = g.trade_id
                                  AND newer_accepted.ts_utc > accepted.ts_utc
                                  AND newer_accepted.ts_utc <= g.ts_utc
                            )
                      )
                  )
                """
            )
            if str(row[0]).strip()
        }
    except sqlite3.Error:
        return set()


def _stale_gpt_close_satisfied_trade_ids(conn: sqlite3.Connection, columns: set[str]) -> set[str]:
    if "trade_id" not in columns:
        return set()
    select_fields = ["trade_id"]
    for column in ("exit_reason", "raw_json"):
        select_fields.append(column if column in columns else f"NULL AS {column}")
    try:
        rows = conn.execute(
            f"""
            SELECT {', '.join(select_fields)}
            FROM execution_events
            WHERE event_type = 'GATEWAY_POSITION_NO_ACTION'
              AND trade_id IS NOT NULL
              AND trade_id != ''
            """
        ).fetchall()
    except sqlite3.Error:
        return set()
    out: set[str] = set()
    for row in rows:
        trade_id = str(row["trade_id"] or "").strip()
        if not trade_id:
            continue
        raw = _json_payload(row["raw_json"])
        management_action = str(raw.get("management_action") or "").strip().upper()
        exit_reason = str(row["exit_reason"] or "").strip().upper()
        if management_action != "GPT_CLOSE" and exit_reason != "GPT_CLOSE":
            continue
        issue_codes = {
            str(item.get("code") or "").strip().upper()
            for item in raw.get("issues", []) or []
            if isinstance(item, dict)
        }
        if "STALE_CLOSE_ALREADY_ABSENT" in issue_codes:
            out.add(trade_id)
    return out


def _close_provenance_for_effect_row(
    row: sqlite3.Row,
    *,
    accepted_close_rows: list[sqlite3.Row],
    gateway_close_trade_ids: set[str],
    gateway_close_order_ids: set[str],
    reconciled_close_trade_ids: set[str],
    reconciled_close_order_ids: set[str],
    gpt_close_trade_ids: set[str],
    stale_close_satisfied_trade_ids: set[str],
) -> str:
    exit_reason = str(_row_text(row, "exit_reason") or "").strip().upper()
    if exit_reason and exit_reason != "MARKET_ORDER_TRADE_CLOSE":
        return exit_reason
    if exit_reason != "MARKET_ORDER_TRADE_CLOSE":
        return "UNKNOWN_CLOSE_PROVENANCE"
    trade_id = str(_row_text(row, "trade_id") or "").strip()
    order_id = str(_row_text(row, "order_id") or "").strip()
    if (trade_id and trade_id in gateway_close_trade_ids) or (order_id and order_id in gateway_close_order_ids):
        return "GATEWAY_TRADE_CLOSE_SENT"
    if trade_id and trade_id in stale_close_satisfied_trade_ids:
        return "STALE_GPT_CLOSE_SATISFIED"
    if trade_id and trade_id in gpt_close_trade_ids:
        return "GATEWAY_GPT_CLOSE_ACCEPTED"
    if (trade_id and trade_id in reconciled_close_trade_ids) or (
        order_id and order_id in reconciled_close_order_ids
    ):
        return "GATEWAY_TRADE_CLOSE_RECONCILED_UNVERIFIED"
    sources = _broker_trade_close_accept_sources(
        accepted_close_rows,
        trade_id=trade_id,
        order_id=order_id,
    )
    if not sources:
        return "NO_LOCAL_CLOSE_PROVENANCE"
    for label in (
        "LOCAL_LEDGER_LANE_ID",
        "TRADER_CLIENT_EXTENSION",
        "TRADER_ENTRY_LANE_ID",
        "TRADER_ENTRY_CLIENT_EXTENSION",
        "NON_TRADER_CLIENT_EXTENSION",
        "DIRECT_OR_MANUAL_BROKER_TRADE_CLOSE",
    ):
        if label in sources:
            return label
    return "BROKER_TRADE_CLOSE_ACCEPTED:" + "+".join(sorted(sources))


_LOSS_CONTAINMENT_CLOSE_PROVENANCES = frozenset(
    {
        "GATEWAY_TRADE_CLOSE_SENT",
        "GATEWAY_GPT_CLOSE_ACCEPTED",
    }
)


def _loss_close_containment(
    row: sqlite3.Row,
    value: float,
    close_provenance: str,
    *,
    close_gate_evidence_passed: bool,
) -> tuple[bool, float | None]:
    """Detect loss-side gateway closes that reduced broker-side SL loss.

    A realized negative GPT/gateway close can be good close discipline when it
    exits a broken thesis before the attached broker SL. The avoided-loss
    credit is counted only when durable close-gate evidence also passed, so a
    missing Gate A/B receipt cannot look like repaired close discipline.
    """

    if value >= 0 or close_provenance not in _LOSS_CONTAINMENT_CLOSE_PROVENANCES:
        return False, None
    if not close_gate_evidence_passed:
        return False, None
    side = str(_row_text(row, "side") or "").upper()
    entry = _maybe_float(row["entry_price"]) if "entry_price" in row.keys() else None
    stop = _maybe_float(row["stop_loss_price"]) if "stop_loss_price" in row.keys() else None
    close = _maybe_float(row["close_price"]) if "close_price" in row.keys() else None
    if side not in {"LONG", "SHORT"} or entry is None or stop is None or close is None:
        return False, None
    if side == "LONG":
        if not (stop < close < entry):
            return False, None
        planned_loss_distance = entry - stop
        actual_loss_distance = entry - close
    else:
        if not (entry < close < stop):
            return False, None
        planned_loss_distance = stop - entry
        actual_loss_distance = close - entry
    if planned_loss_distance <= 0 or actual_loss_distance <= 0:
        return False, None
    avoided_loss = None
    if planned_loss_distance > actual_loss_distance:
        avoided_loss = abs(value) * ((planned_loss_distance / actual_loss_distance) - 1.0)
    return True, avoided_loss


def _add_close_provenance_metric(
    metrics: dict[str, dict[str, Any]],
    label: str,
    value: float,
    *,
    contained_loss: bool = False,
    avoided_loss_jpy: float | None = None,
) -> None:
    bucket = metrics.setdefault(
        label,
        {
            "trades": 0,
            "net_jpy": 0.0,
            "gross_profit_jpy": 0.0,
            "gross_loss_jpy": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
        },
    )
    bucket["trades"] = int(bucket["trades"]) + 1
    bucket["net_jpy"] = float(bucket["net_jpy"]) + value
    if value > 0:
        bucket["gross_profit_jpy"] = float(bucket["gross_profit_jpy"]) + value
        bucket["win_trades"] = int(bucket["win_trades"]) + 1
    elif value < 0:
        bucket["gross_loss_jpy"] = float(bucket["gross_loss_jpy"]) + abs(value)
        bucket["loss_trades"] = int(bucket["loss_trades"]) + 1
        if contained_loss:
            bucket["loss_containment_trades"] = int(bucket.get("loss_containment_trades") or 0) + 1
            bucket["loss_containment_net_jpy"] = float(bucket.get("loss_containment_net_jpy") or 0.0) + value
            if avoided_loss_jpy is not None:
                bucket["loss_containment_avoided_loss_jpy"] = (
                    float(bucket.get("loss_containment_avoided_loss_jpy") or 0.0) + avoided_loss_jpy
                )


def _sorted_close_provenance_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label, values in sorted(metrics.items()):
        item = {
            "trades": int(values.get("trades") or 0),
            "net_jpy": float(values.get("net_jpy") or 0.0),
            "gross_profit_jpy": float(values.get("gross_profit_jpy") or 0.0),
            "gross_loss_jpy": float(values.get("gross_loss_jpy") or 0.0),
            "win_trades": int(values.get("win_trades") or 0),
            "loss_trades": int(values.get("loss_trades") or 0),
        }
        if int(values.get("loss_containment_trades") or 0):
            item["loss_containment_trades"] = int(values.get("loss_containment_trades") or 0)
            item["loss_containment_net_jpy"] = float(values.get("loss_containment_net_jpy") or 0.0)
            item["loss_containment_avoided_loss_jpy"] = float(
                values.get("loss_containment_avoided_loss_jpy") or 0.0
            )
        out[label] = item
    return out


def _row_text(row: sqlite3.Row, key: str) -> str | None:
    if key not in row.keys():
        return None
    value = row[key]
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _method_from_lane_id(lane_id: str | None) -> str | None:
    parts = [part.strip() for part in str(lane_id or "").split(":") if part.strip()]
    if len(parts) >= 4:
        return parts[3].upper()
    return None


def _lane_pair_side_from_id(lane_id: str | None) -> tuple[str, str] | None:
    parts = [part.strip().upper() for part in str(lane_id or "").split(":") if part.strip()]
    if len(parts) < 3:
        return None
    side = parts[2]
    if side not in {"LONG", "SHORT"}:
        return None
    return parts[1], side


def _accepted_trade_has_current_pending_entry(
    decision: dict[str, Any],
    pending_entry_orders: list[dict[str, Any]],
) -> bool:
    wanted: set[tuple[str, str]] = set()
    for key in ("selected_lane_id", "lane_id"):
        parsed = _lane_pair_side_from_id(decision.get(key))
        if parsed is not None:
            wanted.add(parsed)
    raw_lane_ids = decision.get("selected_lane_ids")
    if isinstance(raw_lane_ids, list):
        for lane_id in raw_lane_ids:
            parsed = _lane_pair_side_from_id(str(lane_id))
            if parsed is not None:
                wanted.add(parsed)
    if not wanted:
        return False

    for order in pending_entry_orders:
        pair = str(order.get("pair") or "").upper()
        units = _maybe_float(order.get("units"))
        if not pair or units is None or units == 0:
            continue
        side = "LONG" if units > 0 else "SHORT"
        if (pair, side) in wanted:
            return True
    return False


def _accepted_gpt_decision_consumed_by_gateway(
    *,
    db_path: Path,
    generated_at: datetime | None,
    snapshot_ts: datetime | None,
    action: str,
    decision: dict[str, Any],
) -> bool:
    if generated_at is None or not db_path.exists():
        return False
    normalized_action = str(action or "").upper()
    event_types = _gateway_consumption_event_types(normalized_action)
    if not event_types:
        return False
    selected_lane_ids = {
        str(item)
        for item in (decision.get("selected_lane_ids") or [])
        if str(item)
    }
    for key in ("selected_lane_id", "lane_id"):
        if decision.get(key):
            selected_lane_ids.add(str(decision.get(key)))
    close_trade_ids = {
        str(item)
        for item in (decision.get("close_trade_ids") or [])
        if str(item)
    }

    try:
        with sqlite3.connect(db_path) as conn:
            columns = _table_columns(conn, "execution_events")
            select_columns = ["ts_utc", "event_type"]
            if "lane_id" in columns:
                select_columns.append("lane_id")
            if "trade_id" in columns:
                select_columns.append("trade_id")
            placeholders = ",".join("?" for _ in event_types)
            rows = conn.execute(
                f"""
                SELECT {", ".join(select_columns)}
                FROM execution_events
                WHERE event_type IN ({placeholders})
                ORDER BY ts_utc DESC
                LIMIT 200
                """,
                tuple(sorted(event_types)),
            ).fetchall()
    except sqlite3.Error:
        return False

    for row in rows:
        event_ts = _parse_utc(row[0])
        if event_ts is None or event_ts < generated_at:
            continue
        if snapshot_ts is not None and event_ts > snapshot_ts:
            continue
        event = {select_columns[idx]: row[idx] for idx in range(len(select_columns))}
        if normalized_action == "TRADE" and selected_lane_ids:
            lane_id = str(event.get("lane_id") or "")
            if lane_id and lane_id not in selected_lane_ids:
                continue
        if normalized_action == "CLOSE" and close_trade_ids:
            trade_id = str(event.get("trade_id") or "")
            if trade_id and trade_id not in close_trade_ids:
                continue
        return True
    return False


def _gateway_consumption_event_types(action: str) -> frozenset[str]:
    if action == "TRADE":
        return frozenset(
            {
                "GATEWAY_ORDER_SENT",
                "GATEWAY_ORDER_BLOCKED",
                "ORDER_ACCEPTED",
                "ORDER_FILLED",
            }
        )
    if action == "CANCEL_PENDING":
        return frozenset({"GATEWAY_ORDER_NO_ACTION", "ORDER_CANCELED"})
    if action in {"CLOSE", "PROTECT", "TIGHTEN_SL"}:
        return frozenset(
            {
                "GATEWAY_TRADE_CLOSE_SENT",
                "GATEWAY_POSITION_NO_ACTION",
                "TRADE_CLOSED",
            }
        )
    if action in {"WAIT", "REQUEST_EVIDENCE"}:
        return frozenset({"GATEWAY_ORDER_NO_ACTION", "GATEWAY_POSITION_NO_ACTION"})
    return frozenset()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")}
    except sqlite3.Error:
        return set()


def _active_trader_positions(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in snapshot.get("positions", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("owner") or "").lower() != "trader":
            continue
        trade_id = str(item.get("trade_id") or "")
        if trade_id:
            out.append(item)
    return out


def _trader_pending_entry_orders(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    entry_order_types = {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}
    for item in snapshot.get("orders", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("owner") or "").lower() != "trader":
            continue
        if item.get("trade_id"):
            continue
        if str(item.get("state") or "").upper() not in {"PENDING", "OPEN"}:
            continue
        if str(item.get("order_type") or "").upper() not in entry_order_types:
            continue
        out.append(item)
    return out


def _pending_entry_evidence(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for order in orders:
        evidence.append(
            {
                "order_id": order.get("order_id"),
                "pair": order.get("pair"),
                "order_type": order.get("order_type"),
                "units": order.get("units"),
                "price": order.get("price"),
                "state": order.get("state"),
            }
        )
    return evidence


def _live_ready_results(intents: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in intents.get("results", []) or []
        if isinstance(item, dict) and str(item.get("status") or "") == "LIVE_READY"
    ]


def _intent_status_counts(intents: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        status = str(result.get("status") or "UNKNOWN").upper()
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _top_intent_blockers(intents: dict[str, Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        seen: set[str] = set()
        for key in ("risk_issues", "live_strategy_issues"):
            for raw in result.get(key) or []:
                if isinstance(raw, dict) and str(raw.get("severity") or "").upper() != "BLOCK":
                    continue
                text = _issue_text(raw)
                if text and text not in seen:
                    seen.add(text)
                    counts[text] = counts.get(text, 0) + 1
        # Backward-compatible fallback for artifacts generated before
        # `live_strategy_issues` existed. Dry-run strategy WARNs are not live
        # blockers, so count only structured BLOCK entries here.
        if "live_strategy_issues" not in result:
            for raw in result.get("strategy_issues") or []:
                if isinstance(raw, dict) and str(raw.get("severity") or "").upper() != "BLOCK":
                    continue
                text = _issue_text(raw)
                if text and text not in seen:
                    seen.add(text)
                    counts[text] = counts.get(text, 0) + 1
        for raw in result.get("live_blockers") or []:
            text = _issue_text(raw)
            if text and text not in seen:
                seen.add(text)
                counts[text] = counts.get(text, 0) + 1
    return [
        {"message": message, "count": count}
        for message, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
    ]


_LIVE_READINESS_WARN_CODES = {
    "CHART_DIRECTION_CONFLICT",
    "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
    "FORECAST_WATCH_ONLY",
}

_FORECAST_LIVE_GATE_CODES = {
    "FORECAST_CONFIDENCE_REQUIRED_FOR_LIVE",
    "FORECAST_NOT_EXECUTABLE_FOR_LIVE",
    "FORECAST_RANGE_UNSELECTED_DIRECTION_CONFLICT",
    "FORECAST_WATCH_ONLY",
    "TELEMETRY_FORECAST_NOT_EXECUTABLE_FOR_LIVE",
}

_FORECAST_ARBITRATION_REPAIR_FAMILY = "forecast"


def _top_intent_live_readiness_blockers(
    intents: dict[str, Any],
    *,
    statuses: set[str] | None = None,
) -> list[dict[str, Any]]:
    status_filter = {item.upper() for item in statuses} if statuses is not None else None
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        status = str(result.get("status") or "").upper()
        if status == "LIVE_READY":
            continue
        if status_filter is not None and status not in status_filter:
            continue

        for issue in _live_readiness_issues(result):
            text = str(issue.get("message") or "")
            if not text:
                continue
            counts[text] = counts.get(text, 0) + 1
            lane_id = str(result.get("lane_id") or "")
            if lane_id and lane_id not in examples.setdefault(text, []):
                examples[text].append(lane_id)

    return [
        {"message": message, "count": count, "example_lanes": examples.get(message, [])[:3]}
        for message, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]


def _intent_live_readiness_family_breakdown(intents: dict[str, Any]) -> dict[str, Any]:
    all_non_live = _live_readiness_family_scope(intents)
    dry_run_passed = _live_readiness_family_scope(intents, statuses={"DRY_RUN_PASSED"})
    return {
        "all_non_live_ready": all_non_live["families"],
        "dry_run_passed": dry_run_passed["families"],
        "nearest_live_ready_candidates": dry_run_passed["nearest_candidates"],
    }


def _live_readiness_family_scope(
    intents: dict[str, Any],
    *,
    statuses: set[str] | None = None,
    family_limit: int = 8,
    blocker_limit: int = 5,
    candidate_limit: int = 8,
) -> dict[str, Any]:
    status_filter = {item.upper() for item in statuses} if statuses is not None else None
    family_lane_counts: dict[str, int] = {}
    family_lane_keys: dict[str, set[str]] = {}
    family_blocker_counts: dict[str, dict[str, int]] = {}
    family_blocker_keys: set[tuple[str, str, str]] = set()
    family_examples: dict[str, list[str]] = {}
    candidates_by_lane: dict[str, dict[str, Any]] = {}
    anonymous_candidate_index = 0

    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        status = str(result.get("status") or "").upper()
        if status == "LIVE_READY":
            continue
        if status_filter is not None and status not in status_filter:
            continue
        issues = _live_readiness_issues(result)
        if not issues:
            continue

        lane_key = str(result.get("lane_id") or "")
        if not lane_key:
            anonymous_candidate_index += 1
            lane_key = f"__anonymous_candidate_{anonymous_candidate_index}"
        families_seen: set[str] = set()
        for issue in issues:
            family = str(issue.get("family") or "other")
            message = str(issue.get("message") or "")
            if not message:
                continue
            family_blocker_counts.setdefault(family, {})
            blocker_key = (family, message, lane_key)
            if blocker_key not in family_blocker_keys:
                family_blocker_keys.add(blocker_key)
                family_blocker_counts[family][message] = (
                    family_blocker_counts[family].get(message, 0) + 1
                )
            if family in families_seen:
                continue
            families_seen.add(family)
            family_lane_keys.setdefault(family, set())
            if lane_key in family_lane_keys[family]:
                continue
            family_lane_keys[family].add(lane_key)
            family_lane_counts[family] = family_lane_counts.get(family, 0) + 1
            if lane_key and lane_key not in family_examples.setdefault(family, []):
                family_examples[family].append(lane_key)

        candidate = _nearest_live_ready_candidate(result, issues)
        previous = candidates_by_lane.get(lane_key)
        if previous is None or _nearest_live_ready_candidate_sort_key(
            candidate
        ) < _nearest_live_ready_candidate_sort_key(previous):
            candidates_by_lane[lane_key] = candidate

    families = []
    for family, lane_count in sorted(
        family_lane_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:family_limit]:
        blockers = [
            {"message": message, "count": count}
            for message, count in sorted(
                family_blocker_counts.get(family, {}).items(),
                key=lambda item: (-item[1], item[0]),
            )[:blocker_limit]
        ]
        families.append(
            {
                "family": family,
                "lane_count": lane_count,
                "top_blockers": blockers,
                "example_lanes": family_examples.get(family, [])[:3],
            }
        )

    candidates = sorted(
        candidates_by_lane.values(),
        key=_nearest_live_ready_candidate_sort_key,
    )
    return {"families": families, "nearest_candidates": candidates[:candidate_limit]}


def _nearest_live_ready_candidate(
    result: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    risk_metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
    families = sorted({str(issue.get("family") or "other") for issue in issues})
    blockers = [_nearest_live_ready_blocker(issue) for issue in issues[:6]]
    return {
        "lane_id": str(result.get("lane_id") or ""),
        "status": str(result.get("status") or ""),
        "pair": str(intent.get("pair") or metadata.get("pair") or ""),
        "side": str(intent.get("side") or ""),
        "method": str(market_context.get("method") or metadata.get("method") or ""),
        "order_type": str(intent.get("order_type") or ""),
        "reward_risk": _maybe_float(risk_metrics.get("reward_risk")),
        "reward_jpy": _maybe_float(risk_metrics.get("reward_jpy")),
        "opportunity_mode": str(metadata.get("opportunity_mode") or ""),
        "opportunity_mode_reason": str(metadata.get("opportunity_mode_reason") or ""),
        "tp_execution_mode": str(metadata.get("tp_execution_mode") or ""),
        "tp_target_intent": str(metadata.get("tp_target_intent") or ""),
        "forecast_direction": str(metadata.get("forecast_direction") or ""),
        "forecast_confidence": _maybe_float(metadata.get("forecast_confidence")),
        "blocker_families": families,
        "blocker_family_count": len(families),
        "blocker_count": len(issues),
        "blockers": blockers,
    }


def _nearest_live_ready_blocker(issue: dict[str, Any]) -> dict[str, Any]:
    blocker = {
        "family": str(issue.get("family") or "other"),
        "message": str(issue.get("message") or ""),
        "source": str(issue.get("source") or ""),
    }
    evidence = issue.get("strategy_profile_evidence")
    if isinstance(evidence, dict):
        blocker["strategy_profile_evidence"] = evidence
    return blocker


def _nearest_live_ready_candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    reward_risk = _maybe_float(item.get("reward_risk"))
    forecast_confidence = _maybe_float(item.get("forecast_confidence"))
    return (
        int(item.get("blocker_family_count") or 0),
        int(item.get("blocker_count") or 0),
        0 if str(item.get("status") or "").upper() == "DRY_RUN_PASSED" else 1,
        1 if reward_risk is None else 0,
        -(reward_risk or 0.0),
        1 if forecast_confidence is None else 0,
        -(forecast_confidence or 0.0),
        str(item.get("lane_id") or ""),
    )


def _live_readiness_issues(result: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    structured_issue_seen = False

    for raw in result.get("risk_issues") or []:
        if not _risk_issue_blocks_live_readiness(raw):
            continue
        structured_issue_seen = _append_live_readiness_issue(
            issues,
            seen,
            raw,
            source="risk_issues",
        ) or structured_issue_seen
    for raw in result.get("live_strategy_issues") or []:
        if not _strategy_issue_blocks_live_readiness(raw):
            continue
        structured_issue_seen = _append_live_readiness_issue(
            issues,
            seen,
            raw,
            source="live_strategy_issues",
        ) or structured_issue_seen
    if "live_strategy_issues" not in result:
        for raw in result.get("strategy_issues") or []:
            if isinstance(raw, dict) and str(raw.get("severity") or "").upper() != "BLOCK":
                continue
            structured_issue_seen = _append_live_readiness_issue(
                issues,
                seen,
                raw,
                source="strategy_issues",
            ) or structured_issue_seen
    if not structured_issue_seen:
        for raw in result.get("live_blockers") or []:
            _append_live_readiness_issue(
                issues,
                seen,
                raw,
                source="live_blockers",
            )
    return issues


def _append_live_readiness_issue(
    issues: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    raw: Any,
    *,
    source: str,
) -> bool:
    text = _issue_text(raw)
    if not text:
        return False
    key = (source, text)
    if key in seen:
        return False
    seen.add(key)
    issues.append(
        _live_readiness_issue_payload(
            raw,
            source=source,
            message=text,
        )
    )
    return True


def _live_readiness_issue_payload(raw: Any, *, source: str, message: str) -> dict[str, Any]:
    payload = {
        "message": message,
        "family": _live_readiness_issue_family(raw, source=source),
        "source": source,
    }
    if isinstance(raw, dict) and isinstance(raw.get("strategy_profile_evidence"), dict):
        payload["strategy_profile_evidence"] = raw["strategy_profile_evidence"]
    return payload


def _live_readiness_issue_family(raw: Any, *, source: str) -> str:
    code = ""
    message = _issue_text(raw)
    if isinstance(raw, dict):
        code = str(raw.get("code") or "")
        message = str(raw.get("message") or message)
    text = f"{code} {message}".upper()

    if any(token in text for token in ("FORECAST", "PROJECTION", "WATCH_ONLY", "TELEMETRY")):
        return "forecast"
    if any(token in text for token in ("STRATEGY", "PROFILE", "METHOD_PROFILE", "ELIGIBLE")):
        return "strategy_profile"
    if any(
        token in text
        for token in (
            "EXHAUSTION",
            "CHASE",
            "CHART_DIRECTION",
            "PATTERN_REVERSAL",
            "BREAKOUT_FAILURE",
            "RANGE_ROTATION",
            "TREND_MARKET",
            "RANGE_MARKET",
            "RAIL",
            "RETEST",
            "PHASE",
            "LOCATION",
        )
    ):
        return "market_structure"
    if any(
        token in text
        for token in (
            "REWARD_RISK",
            "RISK",
            "STOP",
            "SPREAD",
            "MARGIN",
            "UNITS",
            "ATR",
            "DISTANCE",
            "DISASTER",
            "SIZE",
            "TP",
            "SL",
        )
    ):
        return "risk_geometry"
    if any(token in text for token in ("LIQUID", "SESSION", "GATEWAY", "BROKER", "ORDER", "FILL", "SLIPPAGE")):
        return "execution_liquidity"
    if source == "risk_issues":
        return "risk_geometry"
    if source in {"live_strategy_issues", "strategy_issues"}:
        return "strategy_profile"
    return "other"


def _dry_run_passed_forecast_gate_diagnostics(
    intents: dict[str, Any],
    *,
    limit: int = 8,
) -> dict[str, Any]:
    lanes: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        if str(result.get("status") or "").upper() != "DRY_RUN_PASSED":
            continue

        risk_issue_codes = _issue_codes(result.get("risk_issues"))
        live_blocker_text = " ".join(
            _issue_text(raw).upper() for raw in result.get("live_blockers") or []
        )
        if not risk_issue_codes.intersection(_FORECAST_LIVE_GATE_CODES) and "FORECAST" not in live_blocker_text:
            continue

        intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        support = metadata.get("forecast_market_support")
        support = support if isinstance(support, dict) else {}
        support_reason = str(support.get("reason") or "NO_FORECAST_MARKET_SUPPORT_REASON")
        reason_counts[support_reason] = reason_counts.get(support_reason, 0) + 1

        signals = support.get("signals") if isinstance(support.get("signals"), list) else []
        top_signal = next((item for item in signals if isinstance(item, dict)), {})
        lanes.append(
            {
                "lane_id": str(result.get("lane_id") or ""),
                "side": str(intent.get("side") or ""),
                "order_type": str(intent.get("order_type") or ""),
                "opportunity_mode": str(metadata.get("opportunity_mode") or ""),
                "opportunity_mode_reason": str(metadata.get("opportunity_mode_reason") or ""),
                "opportunity_mode_reward_risk": _maybe_float(metadata.get("opportunity_mode_reward_risk")),
                "tp_target_intent": str(metadata.get("tp_target_intent") or ""),
                "forecast_direction": str(metadata.get("forecast_direction") or ""),
                "forecast_confidence": _maybe_float(metadata.get("forecast_confidence")),
                "forecast_raw_confidence": _maybe_float(metadata.get("forecast_raw_confidence")),
                "chart_direction_bias": str(metadata.get("chart_direction_bias") or ""),
                "forecast_market_support_ok": bool(support.get("ok")),
                "forecast_market_support_reason": support_reason,
                "forecast_market_support_best_hit_rate": _maybe_float(support.get("best_hit_rate")),
                "forecast_market_support_best_samples": _maybe_int(support.get("best_samples")),
                "forecast_market_support_aligned_projection_count": _maybe_int(
                    support.get("aligned_projection_count")
                ),
                "forecast_market_support_timing_projection_count": _maybe_int(
                    support.get("timing_projection_count")
                ),
                "forecast_market_support_unselected_reason": str(support.get("unselected_reason") or ""),
                "forecast_market_support_top_signal": {
                    "name": str(top_signal.get("name") or ""),
                    "direction": str(top_signal.get("direction") or ""),
                    "confidence": _maybe_float(top_signal.get("confidence")),
                    "hit_rate": _maybe_float(top_signal.get("hit_rate")),
                    "samples": _maybe_int(top_signal.get("samples")),
                    "timeframe": str(top_signal.get("timeframe") or ""),
                },
                "risk_issue_codes": sorted(risk_issue_codes),
                "live_strategy_issue_codes": sorted(_issue_codes(result.get("live_strategy_issues"))),
            }
        )

    return {
        "reason_counts": [
            {"reason": reason, "count": count}
            for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ],
        "lanes": lanes[:limit],
    }


def _forecast_arbitration_diagnostics(
    intents: dict[str, Any],
    *,
    limit: int = 8,
) -> dict[str, Any]:
    lanes: list[dict[str, Any]] = []
    same_side_lanes: list[dict[str, Any]] = []
    same_side_actionable_repair_lanes: list[dict[str, Any]] = []
    same_side_context_blocked_lanes: list[dict[str, Any]] = []
    opposite_side_lanes: list[dict[str, Any]] = []
    mixed_relation_lanes: list[dict[str, Any]] = []
    signal_counts: dict[str, int] = {}
    direction_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    pair_direction_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    same_side_context_blocker_counts: dict[str, int] = {}
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        if str(result.get("status") or "").upper() != "DRY_RUN_PASSED":
            continue
        intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        support = metadata.get("forecast_market_support")
        support = support if isinstance(support, dict) else {}
        unselected = support.get("unselected_signals")
        if not isinstance(unselected, list) or not unselected:
            continue
        unselected_items = [item for item in unselected if isinstance(item, dict)]
        top = unselected_items[0] if unselected_items else None
        if top is None:
            continue
        signal_name = str(top.get("name") or "projection")
        signal_direction = str(top.get("direction") or "").upper()
        pair = str(intent.get("pair") or metadata.get("pair") or "").upper()
        side = str(intent.get("side") or "").upper()
        if not pair:
            lane_id = str(result.get("lane_id") or "")
            parts = lane_id.split(":")
            if len(parts) >= 3:
                pair = parts[1].upper()
        if not signal_direction:
            signal_direction = "UNKNOWN"
        signal_side = _forecast_signal_side(signal_direction)
        same_side_signal: dict[str, Any] | None = None
        opposite_side_signal: dict[str, Any] | None = None
        for item in unselected_items:
            item_direction = str(item.get("direction") or "").upper()
            item_side = _forecast_signal_side(item_direction)
            if item_side == side:
                if same_side_signal is None:
                    same_side_signal = item
            elif item_side in {"LONG", "SHORT"} and side in {"LONG", "SHORT"} and opposite_side_signal is None:
                opposite_side_signal = item
        relation = (
            "same_side"
            if same_side_signal is not None and opposite_side_signal is None
            else "mixed_with_opposite"
            if same_side_signal is not None and opposite_side_signal is not None
            else "opposite_side"
            if opposite_side_signal is not None
            else "unknown"
        )
        signal_key = f"{signal_name}:{signal_direction}"
        pair_direction_key = f"{pair or 'UNKNOWN'}:{signal_direction}"
        signal_counts[signal_key] = signal_counts.get(signal_key, 0) + 1
        direction_counts[signal_direction] = direction_counts.get(signal_direction, 0) + 1
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        pair_direction_counts[pair_direction_key] = pair_direction_counts.get(pair_direction_key, 0) + 1
        reason = str(support.get("unselected_reason") or support.get("reason") or "")
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        risk_metrics = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
        live_readiness_issues = _live_readiness_issues(result)
        live_readiness_families = sorted(
            {
                str(issue.get("family") or "other")
                for issue in live_readiness_issues
                if str(issue.get("family") or "").strip()
            }
        )
        context_blocker_families = [
            family
            for family in live_readiness_families
            if family != _FORECAST_ARBITRATION_REPAIR_FAMILY
        ]
        lane = {
            "lane_id": str(result.get("lane_id") or ""),
            "pair": pair,
            "side": side,
            "status": str(result.get("status") or ""),
            "order_type": str(intent.get("order_type") or ""),
            "forecast_direction": str(metadata.get("forecast_direction") or ""),
            "forecast_confidence": _maybe_float(metadata.get("forecast_confidence")),
            "forecast_raw_confidence": _maybe_float(metadata.get("forecast_raw_confidence")),
            "chart_direction_bias": str(metadata.get("chart_direction_bias") or ""),
            "forecast_market_support_reason": str(support.get("reason") or ""),
            "forecast_market_support_unselected_reason": reason,
            "top_unselected_signal_side": signal_side,
            "top_unselected_signal_relation": relation,
            "top_unselected_signal": {
                "name": signal_name,
                "direction": signal_direction,
                "confidence": _maybe_float(top.get("confidence")),
                "hit_rate": _maybe_float(top.get("hit_rate")),
                "samples": _maybe_int(top.get("samples")),
                "timeframe": str(top.get("timeframe") or ""),
                "rationale": str(top.get("rationale") or "")[:180],
            },
            "same_side_unselected_signal": _forecast_signal_payload(same_side_signal),
            "opposite_side_unselected_signal": _forecast_signal_payload(opposite_side_signal),
            "unselected_projection_count": _maybe_int(support.get("unselected_projection_count")) or len(unselected),
            "reward_risk": _maybe_float(risk_metrics.get("reward_risk")),
            "reward_jpy": _maybe_float(risk_metrics.get("reward_jpy")),
            "live_readiness_families": live_readiness_families,
            "context_blocker_families": context_blocker_families,
            "risk_issue_codes": sorted(_issue_codes(result.get("risk_issues"))),
            "live_strategy_issue_codes": sorted(_issue_codes(result.get("live_strategy_issues"))),
        }
        lanes.append(lane)
        if relation == "same_side":
            same_side_lanes.append(lane)
            if context_blocker_families:
                same_side_context_blocked_lanes.append(lane)
                for family in context_blocker_families:
                    same_side_context_blocker_counts[family] = (
                        same_side_context_blocker_counts.get(family, 0) + 1
                    )
            else:
                same_side_actionable_repair_lanes.append(lane)
        elif relation == "opposite_side":
            opposite_side_lanes.append(lane)
        elif relation == "mixed_with_opposite":
            mixed_relation_lanes.append(lane)

    def _count_items(counts: dict[str, int], key_name: str) -> list[dict[str, Any]]:
        return [
            {key_name: key, "count": count}
            for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
            if key
        ]

    return {
        "lane_count": len(lanes),
        "same_side_lane_count": len(same_side_lanes),
        "same_side_actionable_repair_lane_count": len(same_side_actionable_repair_lanes),
        "same_side_context_blocked_lane_count": len(same_side_context_blocked_lanes),
        "opposite_side_lane_count": len(opposite_side_lanes),
        "mixed_relation_lane_count": len(mixed_relation_lanes),
        "opposite_conflict_lane_count": len(opposite_side_lanes) + len(mixed_relation_lanes),
        "unknown_relation_lane_count": max(
            0,
            len(lanes) - len(same_side_lanes) - len(opposite_side_lanes) - len(mixed_relation_lanes),
        ),
        "relation_counts": _count_items(relation_counts, "relation"),
        "signal_counts": _count_items(signal_counts, "signal"),
        "direction_counts": _count_items(direction_counts, "direction"),
        "pair_direction_counts": _count_items(pair_direction_counts, "pair_direction"),
        "reason_counts": _count_items(reason_counts, "reason"),
        "same_side_context_blocker_counts": _count_items(same_side_context_blocker_counts, "family"),
        "lanes": lanes[:limit],
        "same_side_lanes": same_side_lanes[:limit],
        "same_side_actionable_repair_lanes": same_side_actionable_repair_lanes[:limit],
        "same_side_context_blocked_lanes": same_side_context_blocked_lanes[:limit],
        "opposite_side_lanes": opposite_side_lanes[:limit],
        "mixed_relation_lanes": mixed_relation_lanes[:limit],
        "opposite_conflict_lanes": (opposite_side_lanes + mixed_relation_lanes)[:limit],
    }


def _forecast_signal_side(direction: str) -> str:
    normalized = str(direction or "").upper()
    if normalized == "UP":
        return "LONG"
    if normalized == "DOWN":
        return "SHORT"
    return "UNKNOWN"


def _forecast_signal_payload(signal: Any) -> dict[str, Any]:
    if not isinstance(signal, dict):
        return {}
    return {
        "name": str(signal.get("name") or "projection"),
        "direction": str(signal.get("direction") or "").upper() or "UNKNOWN",
        "confidence": _maybe_float(signal.get("confidence")),
        "hit_rate": _maybe_float(signal.get("hit_rate")),
        "samples": _maybe_int(signal.get("samples")),
        "timeframe": str(signal.get("timeframe") or ""),
        "rationale": str(signal.get("rationale") or "")[:180],
    }


def _issue_codes(items: Any) -> set[str]:
    codes: set[str] = set()
    for raw in items or []:
        if isinstance(raw, dict):
            code = str(raw.get("code") or "").upper()
            if code:
                codes.add(code)
                continue
        text = _issue_text(raw).strip().upper()
        if text:
            codes.add(text)
    return codes


def _risk_issue_blocks_live_readiness(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return bool(str(raw).strip())
    severity = str(raw.get("severity") or "").upper()
    if severity == "BLOCK":
        return True
    if severity != "WARN":
        return False
    code = str(raw.get("code") or "").upper()
    return code.endswith("_FOR_LIVE") or code in _LIVE_READINESS_WARN_CODES


def _strategy_issue_blocks_live_readiness(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return bool(str(raw).strip())
    return str(raw.get("severity") or "").upper() in {"BLOCK", "WARN"}


def _only_order_intent_lane_blockers(blocking_evidence: list[Any]) -> bool:
    if not blocking_evidence:
        return False
    for item in blocking_evidence:
        if not isinstance(item, dict):
            return False
        if str(item.get("source") or "") != "order_intents":
            return False
        if str(item.get("check_name") or "") != "lane_blockers":
            return False
        if str(item.get("subject_type") or "") != "lane":
            return False
    return True


def _close_recommendations(
    sidecars: dict[str, tuple[_LoadedJson, Path]],
    active_trade_ids: set[str],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for name, (loaded, _path) in sidecars.items():
        payload = loaded.payload or {}
        list_keys = ("positions", "evolutions", "assessments", "verdicts")
        for key in list_keys:
            rows = payload.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                trade_id = str(row.get("trade_id") or "")
                if trade_id not in active_trade_ids:
                    continue
                status_parts = [
                    str(row.get(field) or "")
                    for field in ("action", "status", "verdict", "reason", "close_review_action")
                ]
                reasons = row.get("reasons")
                if isinstance(reasons, list):
                    status_parts.extend(str(item or "") for item in reasons)
                status_text = " ".join(status_parts).upper()
                if any(
                    token in status_text
                    for token in (
                        "REVIEW_CLOSE",
                        "RECOMMEND_CLOSE",
                        "REVIEW_EXIT",
                        "BROKEN",
                        "REQUIRE_THESIS_REPAIR",
                        "UNVERIFIABLE",
                    )
                ):
                    refs.append(
                        {
                            "source": name,
                            "trade_id": trade_id,
                            "pair": row.get("pair"),
                            "side": row.get("side"),
                            "status": row.get("status"),
                            "verdict": row.get("verdict"),
                            "action": row.get("action"),
                            "close_review_action": row.get("close_review_action"),
                        }
                    )
    return refs


def _target_open(target_state: dict[str, Any]) -> bool:
    try:
        remaining = float(target_state.get("remaining_target_jpy") or 0.0)
    except (TypeError, ValueError):
        return False
    return remaining > 0.0 and target_state.get("status") != "TARGET_REACHED_PROTECT"


def _projection_expired(row: dict[str, Any], *, now: datetime) -> bool:
    emitted = _parse_utc(row.get("timestamp_emitted_utc"))
    window_min = _maybe_float(row.get("resolution_window_min"))
    if emitted is None or window_min is None or window_min <= 0:
        return True
    expiry_age_seconds = (now - emitted).total_seconds() - (window_min * 60.0)
    return expiry_age_seconds >= PROJECTION_PENDING_EXPIRY_GRACE_SECONDS


def _projection_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair": row.get("pair"),
        "signal_name": row.get("signal_name"),
        "cycle_id": row.get("cycle_id"),
        "timestamp_emitted_utc": row.get("timestamp_emitted_utc"),
        "resolution_window_min": row.get("resolution_window_min"),
    }


def _finding(
    *,
    run_id: str,
    priority: str,
    layer: str,
    code: str,
    message: str,
    next_action: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "priority": priority,
        "layer": layer,
        "code": code,
        "message": message,
        "next_action": next_action,
    }
    if evidence:
        payload["evidence"] = evidence
    return payload


def _root_cause_focus(
    *,
    findings: list[dict[str, Any]],
    runtime: dict[str, Any],
    effect_metrics: dict[str, Any],
    execution_quality: dict[str, Any],
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    for item in _expand_repeated_repair_findings(findings):
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        repeated_code = ""
        repeated_from_process_loop = False
        if code == REPEATED_REPAIR_LOOP_CODE:
            repeated_code = str(evidence.get("repeated_code") or "").strip()
            family = _root_cause_family_for_code(
                repeated_code,
                layer=str(evidence.get("repeated_layer") or item.get("layer") or ""),
            )
            repeated_from_process_loop = bool(family)
            if not family:
                family = "PROCESS_LOOP"
        else:
            family = _root_cause_family_for_code(code, layer=str(item.get("layer") or ""))
        if not family:
            continue
        candidate = candidates.setdefault(
            family,
            {
                "family": family,
                "score": 0.0,
                "priority": "P2",
                "supporting_codes": [],
                "supporting_findings": [],
                "metrics": {},
                "process_loop_streak": None,
            },
        )
        priority = str(item.get("priority") or "P2").upper()
        candidate["score"] = float(candidate["score"]) + ROOT_CAUSE_PRIORITY_WEIGHTS.get(priority, 5.0)
        candidate["score"] = float(candidate["score"]) + ROOT_CAUSE_CODE_BOOSTS.get(
            repeated_code if repeated_from_process_loop else code,
            0.0,
        )
        if repeated_from_process_loop:
            # A repeated diagnosis is not the essence. It strengthens the
            # family it keeps pointing to, then the primary action can stay
            # narrow instead of becoming "fix the loop" as its own task.
            candidate["score"] = float(candidate["score"]) + 25.0
            candidate["process_loop_streak"] = evidence.get("current_streak")
            support_code = repeated_code
        else:
            support_code = code
        supporting_codes = candidate["supporting_codes"]
        if isinstance(supporting_codes, list) and support_code and support_code not in supporting_codes:
            supporting_codes.append(support_code)
        supporting_findings = candidate["supporting_findings"]
        if isinstance(supporting_findings, list):
            supporting_findings.append(
                {
                    "priority": priority,
                    "code": code,
                    "message": item.get("message"),
                }
            )
        metric_code = code if repeated_from_process_loop else support_code
        _root_cause_merge_metrics(
            candidate,
            code=metric_code,
            finding_evidence=evidence,
            runtime=runtime,
            effect_metrics=effect_metrics,
            execution_quality=execution_quality,
        )

    if not candidates:
        return {
            "status": "NO_ACTIONABLE_ROOT_CAUSE",
            "primary": None,
            "candidates": [],
        }
    finalized = [_root_cause_finalize_candidate(item) for item in candidates.values()]
    finalized.sort(
        key=lambda item: (
            -float(item.get("score") or 0.0),
            _root_cause_family_rank(str(item.get("family") or "")),
        )
    )
    primary = dict(finalized[0])
    primary["downstream_symptoms"] = [
        _root_cause_downstream_label(item)
        for item in finalized[1:]
        if _root_cause_downstream_label(item)
    ][:6]
    return {
        "status": "FOCUSED",
        "primary": primary,
        "candidates": finalized,
    }


def _root_cause_family_for_code(code: str, *, layer: str = "") -> str:
    code = str(code or "").upper()
    layer = str(layer or "").lower()
    if not code:
        return ""
    if code == "VERIFICATION_LEDGER_LANE_BLOCKERS_RECORDED":
        return ""
    if code.startswith("DIRECTIONAL_FORECAST_") or code.startswith("FORECAST_ARBITRATION_"):
        return "FORECAST_ADVERSE_PATH"
    if code.startswith("RANGE_FORECAST_") or code in {
        "PROFITABLE_BACKTEST_EDGE_FORECAST_GATED",
        "FORECAST_HISTORY_DUPLICATE_CYCLE_PAIR",
        "FORECAST_HISTORY_LEGACY_PHANTOM_CLUSTERS",
    }:
        return "FORECAST_ADVERSE_PATH"
    if code in {"TARGET_OPEN_NO_LIVE_READY_LANES", "TARGET_OPEN_LIVE_READY_COVERAGE_SHORTFALL"}:
        return "OPPORTUNITY_COVERAGE"
    if code.startswith("PENDING_ENTRY_") or code == "LIVE_READY_MARKET_RR_BELOW_ONE":
        return "EXECUTION_LIFECYCLE"
    if "PROFIT_CAPTURE" in code or "PROFITABILITY" in code or "EXPECTANCY" in code or "CLOSE_DRAG" in code:
        return "EXIT_AND_PROFIT_CAPTURE"
    if code.startswith("LEGACY_REVIEW_EXIT_") or code.startswith("CLOSE_GATE_"):
        return "EXIT_AND_PROFIT_CAPTURE"
    if code.endswith("_UNREADABLE") or "STALE" in code or "MISSING" in code:
        return "DATA_FRESHNESS"
    if "LEDGER" in code or layer in {"runtime", "memory"}:
        return "DATA_FRESHNESS"
    if code == REPEATED_REPAIR_LOOP_CODE:
        return "PROCESS_LOOP"
    if layer == "forecast":
        return "FORECAST_ADVERSE_PATH"
    if layer == "opportunity":
        return "OPPORTUNITY_COVERAGE"
    if layer == "execution_quality":
        return "EXECUTION_LIFECYCLE"
    if layer == "profitability":
        return "EXIT_AND_PROFIT_CAPTURE"
    return ""


def _root_cause_family_rank(family: str) -> int:
    ranks = {
        "DATA_FRESHNESS": 0,
        "FORECAST_ADVERSE_PATH": 1,
        "EXIT_AND_PROFIT_CAPTURE": 2,
        "EXECUTION_LIFECYCLE": 3,
        "OPPORTUNITY_COVERAGE": 4,
        "PROCESS_LOOP": 5,
    }
    return ranks.get(family, 99)


def _root_cause_merge_metrics(
    candidate: dict[str, Any],
    *,
    code: str,
    finding_evidence: dict[str, Any],
    runtime: dict[str, Any],
    effect_metrics: dict[str, Any],
    execution_quality: dict[str, Any],
) -> None:
    metrics = candidate.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        candidate["metrics"] = metrics
    metrics.setdefault("live_ready_lanes", runtime.get("live_ready_lanes"))
    window = effect_metrics.get("window") if isinstance(effect_metrics.get("window"), dict) else {}
    metrics.setdefault("net_jpy", window.get("net_jpy"))
    metrics.setdefault("profit_factor", window.get("profit_factor"))
    if code == "DIRECTIONAL_FORECAST_HIT_RATE_WEAK":
        metrics["directional_hit_rate"] = finding_evidence.get("hit_rate")
        metrics["directional_samples"] = finding_evidence.get("samples")
    elif code == "DIRECTIONAL_FORECAST_INVALIDATION_FIRST_DOMINANT":
        metrics["invalidation_first_rate"] = finding_evidence.get("invalidation_first_rate")
        metrics["invalidation_first_samples"] = finding_evidence.get("samples")
    elif code == "DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK":
        metrics["weak_forecast_buckets"] = len(finding_evidence.get("weak_buckets") or [])
    elif code == "PROJECTION_ECONOMIC_PRECISION_WEAK":
        weak_buckets = [
            item
            for item in finding_evidence.get("weak_buckets", []) or []
            if isinstance(item, dict)
        ]
        usable_edges = [
            item
            for item in finding_evidence.get("usable_edges", []) or []
            if isinstance(item, dict)
        ]
        metrics["projection_economic_precision_gap_count"] = len(weak_buckets)
        metrics["projection_economic_precision_edge_count"] = len(usable_edges)
        if weak_buckets:
            metrics["projection_worst_economic_wilson_lower"] = weak_buckets[0].get(
                "economic_hit_rate_wilson_lower"
            )
            metrics["projection_worst_timeout_rate"] = weak_buckets[0].get("timeout_rate")
    elif code == "TARGET_OPEN_LIVE_READY_COVERAGE_SHORTFALL":
        metrics["target_coverage_pct"] = finding_evidence.get("target_coverage_pct")
        metrics["remaining_target_jpy"] = finding_evidence.get("remaining_target_jpy")
    elif code in {"PENDING_ENTRY_FILL_RATE_WEAK", "PENDING_ENTRY_CANCEL_RATE_HIGH"}:
        fill_rate = finding_evidence.get("fill_rate")
        cancel_rate = finding_evidence.get("cancel_before_fill_rate")
        if fill_rate is not None:
            metrics["pending_fill_rate"] = fill_rate
        if cancel_rate is not None:
            metrics["pending_cancel_before_fill_rate"] = cancel_rate
    elif code == "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE":
        guardian = finding_evidence.get("guardian") if isinstance(finding_evidence.get("guardian"), dict) else {}
        metrics["position_guardian_required"] = guardian.get("required")
        metrics["position_guardian_active"] = guardian.get("active")
        metrics["position_guardian_active_source"] = guardian.get("active_source")
        metrics["profit_capture_miss_active"] = finding_evidence.get("profit_capture_miss_active")
    pending_lifecycle = (
        execution_quality.get("pending_entry_lifecycle")
        if isinstance(execution_quality.get("pending_entry_lifecycle"), dict)
        else {}
    )
    if pending_lifecycle and candidate.get("family") == "EXECUTION_LIFECYCLE":
        if metrics.get("pending_fill_rate") is None:
            metrics["pending_fill_rate"] = pending_lifecycle.get("fill_rate")
        if metrics.get("pending_cancel_before_fill_rate") is None:
            metrics["pending_cancel_before_fill_rate"] = pending_lifecycle.get("cancel_before_fill_rate")


def _root_cause_finalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    family = str(candidate.get("family") or "")
    meta = _root_cause_family_meta(family)
    priorities = [
        str(item.get("priority") or "P2")
        for item in candidate.get("supporting_findings", [])
        if isinstance(item, dict)
    ]
    priority = "P0" if "P0" in priorities else ("P1" if "P1" in priorities else "P2")
    out = dict(candidate)
    out["score"] = round(float(candidate.get("score") or 0.0), 3)
    out["priority"] = priority
    out["confidence"] = _root_cause_confidence(out)
    out["label"] = meta["label"]
    out["why"] = _root_cause_why(out, meta)
    out["goal_adjustment"] = meta["goal_adjustment"]
    out["next_action"] = meta["next_action"]
    return out


def _root_cause_family_meta(family: str) -> dict[str, str]:
    if family == "FORECAST_ADVERSE_PATH":
        return {
            "label": "forecast path or economic precision is failing before entry expansion",
            "goal_adjustment": (
                "Shift the immediate improvement goal from more entries to reducing invalidation-first "
                "directional forecasts, weak buckets, and timeout-heavy projection precision gaps; only "
                "then expand coverage."
            ),
            "next_action": (
                "Repair the named directional forecast or projection economic-precision buckets, then "
                "verify recent hit-rate, timeout, and invalidation-first metrics before increasing entry "
                "frequency."
            ),
        }
    if family == "OPPORTUNITY_COVERAGE":
        return {
            "label": "campaign coverage is too small for the open target",
            "goal_adjustment": (
                "Shift the goal from single-lane selection to building enough current LIVE_READY coverage "
                "for the remaining floor/target while preserving risk gates."
            ),
            "next_action": (
                "Promote only the nearest named forecast/strategy blockers into additional HARVEST or RUNNER "
                "candidates, then rerun coverage metrics."
            ),
        }
    if family == "EXECUTION_LIFECYCLE":
        return {
            "label": "orders are being accepted but the fill lifecycle is weak",
            "goal_adjustment": (
                "Shift the goal from new candidate search to entry-distance, TTL, and thesis-invalidation "
                "separation for already accepted pending orders."
            ),
            "next_action": (
                "Audit canceled-before-fill orders by pair and cancel reason, preserve still-valid rail "
                "theses, and retune entry distance/TTL only where the thesis survives."
            ),
        }
    if family == "EXIT_AND_PROFIT_CAPTURE":
        return {
            "label": "realized outcome and close/TP capture remain the profit leak",
            "goal_adjustment": (
                "Shift the goal from entry count to realized capture: TP fills, profit-side exits, and "
                "loss-side close discipline must explain the negative expectancy."
            ),
            "next_action": (
                "Rank losing close provenance and TP/giveback segments, then repair the worst segment before "
                "increasing exposure."
            ),
        }
    if family == "DATA_FRESHNESS":
        return {
            "label": "audit data freshness or ledger integrity is the prerequisite defect",
            "goal_adjustment": (
                "Shift the goal to broker-truth/data repair because strategy changes from stale artifacts "
                "will produce false diagnoses."
            ),
            "next_action": (
                "Regenerate the stale or missing artifact, sync execution ledger to broker transaction id, "
                "then rerun self-improvement before strategy edits."
            ),
        }
    return {
        "label": "self-improvement process is looping without a narrower repair",
        "goal_adjustment": (
            "Shift the goal from broad analysis repetition to one measurable repair target."
        ),
        "next_action": (
            "Select the repeated finding's named metric, execute one narrow repair, and do not revisit the "
            "same diagnosis until that metric changes."
        ),
    }


def _root_cause_confidence(candidate: dict[str, Any]) -> str:
    score = float(candidate.get("score") or 0.0)
    support = candidate.get("supporting_codes") if isinstance(candidate.get("supporting_codes"), list) else []
    loop = _maybe_float(candidate.get("process_loop_streak"))
    if loop is not None and loop >= REPEATED_REPAIR_LOOP_STREAK_MIN and score >= 100.0:
        return "HIGH"
    if score >= 150.0 and len(support) >= 2:
        return "HIGH"
    if score >= 80.0:
        return "MEDIUM"
    return "LOW"


def _expand_repeated_repair_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in findings:
        if str(item.get("code") or "") != REPEATED_REPAIR_LOOP_CODE:
            expanded.append(item)
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        repeated = evidence.get("repeated_findings")
        if not isinstance(repeated, list) or not repeated:
            expanded.append(item)
            continue
        for repeated_item in repeated:
            if not isinstance(repeated_item, dict):
                continue
            repeated_code = str(repeated_item.get("code") or "").strip()
            if not repeated_code:
                continue
            repeated_evidence = dict(evidence)
            repeated_evidence.update(
                {
                    "repeated_code": repeated_code,
                    "repeated_priority": repeated_item.get("priority"),
                    "repeated_layer": repeated_item.get("layer"),
                    "current_streak": repeated_item.get("current_streak"),
                    "previous_streak": repeated_item.get("previous_streak"),
                    "current_message": repeated_item.get("message"),
                    "current_next_action": repeated_item.get("next_action"),
                }
            )
            expanded_item = dict(item)
            expanded_item["evidence"] = repeated_evidence
            expanded.append(expanded_item)
    return expanded


def _root_cause_why(candidate: dict[str, Any], meta: dict[str, str]) -> str:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    parts = [meta["label"]]
    if "directional_hit_rate" in metrics:
        parts.append(f"directional_hit_rate={_fmt_optional(metrics.get('directional_hit_rate'))}")
    if "invalidation_first_rate" in metrics:
        parts.append(f"invalidation_first_rate={_fmt_optional(metrics.get('invalidation_first_rate'))}")
    if "target_coverage_pct" in metrics:
        parts.append(f"target_coverage_pct={_fmt_optional(metrics.get('target_coverage_pct'))}")
    if "projection_economic_precision_gap_count" in metrics:
        parts.append(
            "projection_economic_precision_gap_count="
            f"{_fmt_optional(metrics.get('projection_economic_precision_gap_count'))}"
        )
    if "projection_economic_precision_edge_count" in metrics:
        parts.append(
            "projection_economic_precision_edge_count="
            f"{_fmt_optional(metrics.get('projection_economic_precision_edge_count'))}"
        )
    if "projection_worst_economic_wilson_lower" in metrics:
        parts.append(
            "projection_worst_economic_wilson_lower="
            f"{_fmt_optional(metrics.get('projection_worst_economic_wilson_lower'))}"
        )
    if "pending_cancel_before_fill_rate" in metrics:
        parts.append(
            f"pending_cancel_before_fill_rate={_fmt_optional(metrics.get('pending_cancel_before_fill_rate'))}"
        )
    if "position_guardian_active" in metrics:
        source = metrics.get("position_guardian_active_source")
        guardian_text = f"position_guardian_active={metrics.get('position_guardian_active')}"
        if source:
            guardian_text += f" source={source}"
        parts.append(guardian_text)
    if "profit_capture_miss_active" in metrics:
        parts.append(f"profit_capture_miss_active={metrics.get('profit_capture_miss_active')}")
    if "profit_factor" in metrics:
        parts.append(f"profit_factor={_fmt_optional(metrics.get('profit_factor'))}")
    loop = candidate.get("process_loop_streak")
    if loop is not None:
        parts.append(f"same finding repeated {loop} audit run(s)")
    codes = candidate.get("supporting_codes") if isinstance(candidate.get("supporting_codes"), list) else []
    if codes:
        parts.append("codes=" + ",".join(str(code) for code in codes[:5]))
    return "; ".join(parts)


def _root_cause_downstream_label(candidate: dict[str, Any]) -> str:
    family = str(candidate.get("family") or "")
    label_by_family = {
        "OPPORTUNITY_COVERAGE": "coverage shortfall",
        "EXECUTION_LIFECYCLE": "pending-order fill/cancel churn",
        "EXIT_AND_PROFIT_CAPTURE": "realized P/L leak",
        "DATA_FRESHNESS": "data freshness/integrity",
        "PROCESS_LOOP": "process loop",
        "FORECAST_ADVERSE_PATH": "forecast adverse path",
    }
    label = label_by_family.get(family, "")
    if not label:
        return ""
    return f"{label} score={_fmt_optional(candidate.get('score'))}"


def _finding_report_details(item: dict[str, Any]) -> list[str]:
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    details: list[str] = []
    if item.get("code") == "TARGET_OPEN_LIVE_READY_COVERAGE_SHORTFALL":
        details.append(
            "  - live coverage: "
            f"lanes=`{evidence.get('live_ready_lanes')}`, "
            f"reward=`{evidence.get('live_ready_reward_jpy')}` JPY, "
            f"remaining_target=`{evidence.get('remaining_target_jpy')}` JPY, "
            f"remaining_floor=`{evidence.get('remaining_minimum_jpy')}` JPY, "
            f"target_coverage=`{evidence.get('target_coverage_pct')}`%"
        )
    runner_diag = (
        evidence.get("runner_candidate_diagnostics")
        if isinstance(evidence.get("runner_candidate_diagnostics"), dict)
        else {}
    )
    mode_text = _report_opportunity_mode_text(evidence.get("opportunity_modes"))
    if mode_text:
        details.append(f"  - opportunity modes: {mode_text}")
    perspective_text = _report_perspective_alignment_text(evidence.get("perspective_alignment_diagnostics"))
    if perspective_text:
        details.append(f"  - perspective alignment: {perspective_text}")
    family_text = _report_dry_run_family_text(evidence.get("live_readiness_blocker_families"))
    if family_text:
        details.append(f"  - dry-run blocker families: {family_text}")
    nearest_text = _report_nearest_candidate_text(evidence.get("live_readiness_blocker_families"))
    if nearest_text:
        details.append(f"  - nearest dry-run lanes: {nearest_text}")
    forecast_gate_text = _report_forecast_gate_text(evidence.get("dry_run_passed_forecast_gate_diagnostics"))
    if forecast_gate_text:
        details.append(f"  - forecast gate reasons: {forecast_gate_text}")
    arbitration_text = _report_forecast_arbitration_text(evidence.get("forecast_arbitration_diagnostics"))
    if arbitration_text:
        details.append(f"  - forecast arbitration: {arbitration_text}")
    if not runner_diag:
        return details
    reasons = runner_diag.get("top_demotion_reasons")
    reason_text = ", ".join(
        f"{reason.get('reason')}={reason.get('count')}"
        for reason in (reasons if isinstance(reasons, list) else [])[:3]
        if isinstance(reason, dict) and str(reason.get("reason") or "").strip()
    )
    details.append(
        "  - runner candidates: "
        f"status=`{runner_diag.get('status')}`, "
        f"trend=`{runner_diag.get('trend_candidate_lanes')}`, "
        f"runner_qualified=`{runner_diag.get('runner_qualified_lanes')}`, "
        f"attached_harvest=`{runner_diag.get('attached_harvest_lanes')}`, "
        f"demotions=`{reason_text or 'none'}`"
    )
    return details


def _report_opportunity_mode_text(raw: Any) -> str:
    modes = raw if isinstance(raw, dict) else {}
    parts: list[str] = []
    for key in ("HARVEST", "RUNNER"):
        item = modes.get(key) if isinstance(modes.get(key), dict) else None
        if not item:
            continue
        live_blocker_codes = (
            item.get("top_live_blocker_codes")
            if isinstance(item.get("top_live_blocker_codes"), list)
            else []
        )
        live_codes = ", ".join(
            str(issue.get("code"))
            for issue in live_blocker_codes[:3]
            if isinstance(issue, dict) and str(issue.get("code") or "").strip()
        )
        issue_codes = item.get("top_issue_codes") if isinstance(item.get("top_issue_codes"), list) else []
        top_codes = ", ".join(
            str(issue.get("code"))
            for issue in issue_codes[:3]
            if isinstance(issue, dict) and str(issue.get("code") or "").strip()
        )
        parts.append(
            f"{key} lanes=`{item.get('lanes')}` live=`{item.get('live_ready_lanes')}` "
            f"reward=`{item.get('reward_jpy')}` live_codes=`{live_codes or 'none'}` "
            f"codes=`{top_codes or 'none'}`"
        )
    return "; ".join(parts)


def _report_perspective_alignment_text(raw: Any) -> str:
    diagnostics = raw if isinstance(raw, dict) else {}
    if not diagnostics:
        return ""
    status = str(diagnostics.get("status") or "")
    mismatch_lanes = _maybe_int(diagnostics.get("range_forecast_method_mismatch_lanes")) or 0
    mismatch_groups = _maybe_int(diagnostics.get("range_forecast_method_mismatch_groups")) or 0
    rows = diagnostics.get("range_forecast_method_mismatch_top")
    labels: list[str] = []
    for item in _perspective_alignment_report_rows(rows if isinstance(rows, list) else []):
        pair = str(item.get("pair") or "").strip()
        direction = str(item.get("direction") or "").strip()
        if not pair or not direction:
            continue
        blockers = item.get("range_rotation_top_live_blocker_codes")
        codes = [
            str(blocker.get("code") or "")
            for blocker in (blockers if isinstance(blockers, list) else [])[:3]
            if isinstance(blocker, dict) and str(blocker.get("code") or "").strip()
        ]
        other_side_directions = item.get("range_rotation_other_side_directions")
        other_side_codes = item.get("range_rotation_other_side_top_live_blocker_codes")
        other_labels = [
            str(row.get("code") or "")
            for row in (other_side_directions if isinstance(other_side_directions, list) else [])[:3]
            if isinstance(row, dict) and str(row.get("code") or "").strip()
        ]
        other_blocker_codes = [
            str(row.get("code") or "")
            for row in (other_side_codes if isinstance(other_side_codes, list) else [])[:3]
            if isinstance(row, dict) and str(row.get("code") or "").strip()
        ]
        other_text = ""
        if other_labels:
            other_text = f" other_rail={'/'.join(other_labels)}"
            if other_blocker_codes:
                other_text += f" other_blockers={','.join(other_blocker_codes)}"
        labels.append(
            f"{pair} {direction} mismatch={_maybe_int(item.get('method_mismatch_lanes')) or 0} "
            f"range_lanes={_maybe_int(item.get('range_rotation_lanes')) or 0} "
            f"blockers={','.join(codes) if codes else 'none'}"
            f"{other_text}"
        )
    if not (status or mismatch_lanes or labels):
        return ""
    return (
        f"status=`{status}`, groups=`{mismatch_groups}`, lanes=`{mismatch_lanes}`"
        + (", top=" + "; ".join(labels) if labels else "")
    )


def _perspective_alignment_report_rows(rows: list[Any]) -> list[dict[str, Any]]:
    typed_rows = [item for item in rows if isinstance(item, dict)]
    selected = typed_rows[:3]
    if any(_maybe_int(item.get("range_rotation_other_side_lanes")) for item in selected):
        return selected
    for item in typed_rows[3:]:
        if _maybe_int(item.get("range_rotation_other_side_lanes")):
            return [*selected, item]
    return selected


def _report_dry_run_family_text(raw: Any) -> str:
    payload = raw if isinstance(raw, dict) else {}
    families = payload.get("dry_run_passed") if isinstance(payload.get("dry_run_passed"), list) else []
    parts: list[str] = []
    for item in families[:5]:
        if not isinstance(item, dict):
            continue
        family = str(item.get("family") or "")
        if not family:
            continue
        parts.append(f"{family}={item.get('lane_count')}")
    return ", ".join(parts)


def _report_nearest_candidate_text(raw: Any) -> str:
    payload = raw if isinstance(raw, dict) else {}
    candidates = (
        payload.get("nearest_live_ready_candidates")
        if isinstance(payload.get("nearest_live_ready_candidates"), list)
        else []
    )
    parts: list[str] = []
    for item in candidates[:4]:
        if not isinstance(item, dict):
            continue
        lane_id = str(item.get("lane_id") or "")
        if not lane_id:
            continue
        method = str(item.get("method") or "unknown")
        order_type = str(item.get("order_type") or "unknown")
        mode = str(item.get("opportunity_mode") or item.get("tp_target_intent") or "unknown")
        families = item.get("blocker_families") if isinstance(item.get("blocker_families"), list) else []
        family_text = "/".join(str(family) for family in families[:3] if str(family).strip()) or "none"
        parts.append(
            f"`{lane_id}` {method}/{order_type}/{mode} "
            f"rr=`{_fmt_optional(item.get('reward_risk'))}` "
            f"reward=`{_fmt_optional(item.get('reward_jpy'))}` "
            f"conf=`{_fmt_optional(item.get('forecast_confidence'))}` blockers=`{family_text}`"
        )
    return "; ".join(parts)


def _report_forecast_gate_text(raw: Any) -> str:
    payload = raw if isinstance(raw, dict) else {}
    reasons = payload.get("reason_counts") if isinstance(payload.get("reason_counts"), list) else []
    parts: list[str] = []
    for item in reasons[:3]:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "")
        if reason:
            parts.append(f"{reason}={item.get('count')}")
    return "; ".join(parts)


def _report_forecast_arbitration_text(raw: Any) -> str:
    payload = raw if isinstance(raw, dict) else {}
    lane_count = _maybe_int(payload.get("lane_count")) or 0
    if lane_count <= 0:
        return ""
    directions = payload.get("direction_counts") if isinstance(payload.get("direction_counts"), list) else []
    relations = payload.get("relation_counts") if isinstance(payload.get("relation_counts"), list) else []
    signals = payload.get("signal_counts") if isinstance(payload.get("signal_counts"), list) else []
    actionable = _maybe_int(payload.get("same_side_actionable_repair_lane_count")) or 0
    context_blocked = _maybe_int(payload.get("same_side_context_blocked_lane_count")) or 0
    lanes = (
        payload.get("same_side_actionable_repair_lanes")
        if isinstance(payload.get("same_side_actionable_repair_lanes"), list)
        and payload.get("same_side_actionable_repair_lanes")
        else payload.get("lanes")
        if isinstance(payload.get("lanes"), list)
        else []
    )
    direction_text = ", ".join(
        f"{item.get('direction')}={item.get('count')}"
        for item in directions[:3]
        if isinstance(item, dict) and str(item.get("direction") or "").strip()
    )
    relation_text = ", ".join(
        f"{item.get('relation')}={item.get('count')}"
        for item in relations[:3]
        if isinstance(item, dict) and str(item.get("relation") or "").strip()
    )
    signal_text = ", ".join(
        f"{item.get('signal')}={item.get('count')}"
        for item in signals[:3]
        if isinstance(item, dict) and str(item.get("signal") or "").strip()
    )
    lane_labels: list[str] = []
    for item in lanes[:3]:
        if not isinstance(item, dict):
            continue
        top = item.get("top_unselected_signal") if isinstance(item.get("top_unselected_signal"), dict) else {}
        pair = str(item.get("pair") or "").strip()
        side = str(item.get("side") or "").strip()
        signal = str(top.get("name") or "").strip()
        direction = str(top.get("direction") or "").strip()
        if pair and side and signal and direction:
            lane_labels.append(f"{pair} {side}->{signal} {direction}")
    return (
        f"lanes=`{lane_count}`, directions=`{direction_text or 'none'}`, "
        f"relations=`{relation_text or 'none'}`, "
        f"same_side_actionable=`{actionable}`, same_side_context_blocked=`{context_blocked}`, "
        f"signals=`{signal_text or 'none'}`"
        + (", top=" + "; ".join(lane_labels) if lane_labels else "")
    )


def _dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priorities = {"P0": 0, "P1": 1, "P2": 2}
    by_code: dict[str, dict[str, Any]] = {}
    for item in findings:
        code = str(item.get("code") or "")
        existing = by_code.get(code)
        if existing is None or priorities.get(str(item.get("priority")), 9) < priorities.get(str(existing.get("priority")), 9):
            by_code[code] = item
    return sorted(
        by_code.values(),
        key=lambda item: (priorities.get(str(item.get("priority")), 9), str(item.get("layer")), str(item.get("code"))),
    )


def _next_actions(
    findings: list[dict[str, Any]],
    *,
    root_cause_focus: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    primary = (
        root_cause_focus.get("primary")
        if isinstance(root_cause_focus, dict) and isinstance(root_cause_focus.get("primary"), dict)
        else {}
    )
    primary_family = str(primary.get("family") or "")
    if primary_family:
        actions.append(
            {
                "priority": str(primary.get("priority") or "P1"),
                "code": f"{ROOT_CAUSE_CODE}:{primary_family}",
                "next_action": str(primary.get("next_action") or ""),
            }
        )
    for item in findings:
        code = str(item.get("code") or "")
        if any(action["code"] == code for action in actions):
            continue
        actions.append(
            {
                "priority": str(item.get("priority") or ""),
                "code": code,
                "next_action": str(item.get("next_action") or ""),
            }
        )
        if len(actions) >= 3:
            break
    return actions


def _issue_text(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("code") or raw.get("message") or raw.get("reason") or raw)
    return str(raw)


def _transaction_id_is_behind(actual: str, expected: str) -> bool:
    try:
        return int(actual) < int(expected)
    except (TypeError, ValueError):
        return actual < expected


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fmt_optional(value: Any) -> str:
    number = _maybe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}"


def _fmt_close_provenance_metrics(metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    for label, raw in sorted(metrics.items()):
        if not isinstance(raw, dict):
            continue
        trades = int(raw.get("trades") or 0)
        net = _maybe_float(raw.get("net_jpy")) or 0.0
        parts.append(f"{label}: {trades} trade(s), {net:.1f} JPY")
    return "; ".join(parts) if parts else "n/a"


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        offset = ""
        frac = tail
        for marker in ("+", "-"):
            if marker in tail:
                frac, offset_part = tail.split(marker, 1)
                offset = marker + offset_part
                break
        if len(frac) > 6:
            frac = frac[:6]
        text = f"{head}.{frac}{offset}"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _to_utc(dt)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
