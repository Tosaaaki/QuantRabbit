from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ENTRY_THESIS_LEDGER,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_MEMORY_HEALTH,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_POSITION_MANAGEMENT,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_SELF_IMPROVEMENT_AUDIT_REPORT,
    DEFAULT_SELF_IMPROVEMENT_HISTORY_DB,
    DEFAULT_TRADER_DECISION,
    DEFAULT_VERIFICATION_LEDGER,
)


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

PROFITABILITY_DISCIPLINE_CODES = (
    "NEGATIVE_RECENT_EXPECTANCY",
    "SMALL_WIN_LARGE_LOSS_ASYMMETRY",
)

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

# Forecast-level calibration is the feedback loop for "why did the final
# direction call miss?". Ten samples matches the projection-ledger calibration
# sample floor, and the hit-rate floor is an audit warning, not a trade gate.
FORECAST_CALIBRATION_MIN_SAMPLES = int(
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_FORECAST_CALIBRATION_MIN_SAMPLES", 10.0)
)
FORECAST_HIT_RATE_WARN_BELOW = _env_nonnegative_float(
    "QR_SELF_IMPROVEMENT_FORECAST_HIT_RATE_WARN_BELOW", 0.45
)
# Audit feedback coverage floor only. Directional calls with mostly TIMEOUT
# outcomes are not learnable enough to justify stronger forecast confidence.
FORECAST_CALIBRATION_MIN_COVERAGE = min(
    1.0,
    _env_nonnegative_float("QR_SELF_IMPROVEMENT_FORECAST_CALIBRATION_MIN_COVERAGE", 0.5),
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
        forecast_history_path: Path = DEFAULT_FORECAST_HISTORY,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        entry_thesis_ledger_path: Path = DEFAULT_ENTRY_THESIS_LEDGER,
        gpt_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
        trader_decision_path: Path = DEFAULT_TRADER_DECISION,
        position_management_path: Path = DEFAULT_POSITION_MANAGEMENT,
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
        gpt_loaded = _read_json(gpt_decision_path)
        trader_loaded = _read_json(trader_decision_path)
        position_management_loaded = _read_json(position_management_path)
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
        effect = _effect_metrics(self.db_path, window_hours=window_hours, now=clock)
        effect_24h = _effect_metrics(self.db_path, window_hours=24.0, now=clock)
        close_gate_loss_evidence = _close_gate_loss_evidence(ai_backtest_loaded, ai_test_bot_backtest_path)
        profitability_streak_before = self._history_code_streak(PROFITABILITY_DISCIPLINE_CODES)
        latest_gpt_stale_streak_before = self._history_code_streak(("LATEST_GPT_DECISION_STALE",))

        findings: list[dict[str, Any]] = []
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
        findings.extend(
            _intent_findings(
                run_id=run_id,
                intents=intents,
                target_open=target_open,
                live_ready=live_ready,
                active_positions=active_positions,
                pending_entry_orders=pending_entry_orders,
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
        findings.extend(
            _sidecar_findings(
                run_id=run_id,
                snapshot_ts=snapshot_ts,
                active_trade_ids=active_trade_ids,
                gpt_decision=gpt_loaded.payload or {},
                sidecars={
                    "position_management": (position_management_loaded, position_management_path),
                    "thesis_evolution": (thesis_evolution_loaded, thesis_evolution_path),
                    "position_thesis": (position_thesis_loaded, position_thesis_path),
                    "forecast_persistence": (forecast_persistence_loaded, forecast_persistence_path),
                },
            )
        )
        findings.extend(
            _decision_artifact_findings(
                run_id=run_id,
                gpt_loaded=gpt_loaded,
                trader_loaded=trader_loaded,
                gpt_path=gpt_decision_path,
                trader_path=trader_decision_path,
                target_open=target_open,
                active_trade_ids=active_trade_ids,
                live_ready_lanes=len(live_ready),
                pending_entry_orders=len(pending_entry_orders),
                pending_entry_order_details=pending_entry_orders,
                snapshot_ts=snapshot_ts,
                latest_gpt_stale_streak_before=latest_gpt_stale_streak_before,
            )
        )

        findings = _dedupe_findings(findings)
        p0 = sum(1 for item in findings if item["priority"] == "P0")
        p1 = sum(1 for item in findings if item["priority"] == "P1")
        p2 = sum(1 for item in findings if item["priority"] == "P2")
        status = STATUS_BLOCKED if p0 else (STATUS_ACTION_REQUIRED if findings else STATUS_OK)
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
                "thesis_evolution": str(thesis_evolution_path),
                "position_thesis": str(position_thesis_path),
                "forecast_persistence": str(forecast_persistence_path),
                "coverage_optimization": str(coverage_optimization_path),
            },
            "runtime": {
                "target_open": target_open,
                "snapshot_fetched_at_utc": snapshot_ts.isoformat() if snapshot_ts else None,
                "open_trader_positions": len(active_positions),
                "open_trader_pending_entries": len(pending_entry_orders),
                "live_ready_lanes": len(live_ready),
                "gpt_status": (gpt_loaded.payload or {}).get("status"),
                "gpt_action": ((gpt_loaded.payload or {}).get("decision") or {}).get("action")
                if isinstance((gpt_loaded.payload or {}).get("decision"), dict)
                else None,
            },
            "effect_metrics": {
                "window": effect,
                "last_24h": effect_24h,
            },
            "findings": findings,
            "next_actions": _next_actions(findings),
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
        lines.extend(["", "## Findings", ""])
        if payload["findings"]:
            for item in payload["findings"]:
                lines.append(
                    f"- `{item['priority']}` `{item['layer']}` `{item['code']}`: {item['message']} Next: {item['next_action']}"
                )
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
    if code != "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED":
        return message
    marker = " consecutive audit run(s): "
    _, sep, suffix = message.partition(marker)
    if not sep:
        return message
    return f"profitability discipline has failed for <streak>{marker}{suffix}"


def _history_normalized_evidence(code: str, evidence: Any) -> Any:
    if code not in HISTORY_VOLATILE_STREAK_CODES or not isinstance(evidence, dict):
        return evidence
    normalized = dict(evidence)
    normalized.pop("current_streak", None)
    normalized.pop("previous_streak", None)
    return normalized


HISTORY_VOLATILE_STREAK_CODES = frozenset(
    {
        "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
        "LATEST_GPT_DECISION_STALE",
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
        stale_refs = _memory_health_stale_refs(generated_at=generated_at, snapshot=snapshot, intents=intents)
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
        if generated_at < ref_ts:
            stale.append({"label": label, "timestamp_utc": ref_ts.isoformat()})
    return stale


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
    out.extend(_directional_forecast_quality_findings(run_id=run_id, rows=rows))
    return out


def _directional_forecast_quality_findings(
    *,
    run_id: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    directional = [
        row for row in rows
        if str(row.get("signal_name") or "").strip() == "directional_forecast"
    ]
    if len(directional) < FORECAST_CALIBRATION_MIN_SAMPLES:
        return []
    status_counts: dict[str, int] = {}
    calibrated: list[dict[str, Any]] = []
    for row in directional:
        status = str(row.get("resolution_status") or "PENDING").upper()
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"HIT", "MISS"} and _directional_forecast_has_target_invalidation(row):
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
                    "min_samples": FORECAST_CALIBRATION_MIN_SAMPLES,
                    "examples": [_projection_ref(row) for row in directional[:8]],
                },
            )
        ]
    findings: list[dict[str, Any]] = []
    calibration_coverage = len(calibrated) / len(directional)
    timeout_count = int(status_counts.get("TIMEOUT") or 0)
    if calibration_coverage < FORECAST_CALIBRATION_MIN_COVERAGE and timeout_count >= FORECAST_CALIBRATION_MIN_SAMPLES:
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
                    "status_counts": status_counts,
                    "examples": [_projection_ref(row) for row in directional[:8]],
                },
            )
        )
    weak_buckets = _directional_forecast_worst_buckets(
        calibrated,
        min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
        warn_below=FORECAST_HIT_RATE_WARN_BELOW,
    )
    if weak_buckets:
        findings.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_BUCKET_HIT_RATE_WEAK",
                message=(
                    f"{len(weak_buckets)} directional_forecast pair/direction/regime bucket(s) "
                    f"are below {FORECAST_HIT_RATE_WARN_BELOW:.0%} HIT rate"
                ),
                next_action=(
                    "Dampen or rework the named forecast buckets before treating them as opportunity "
                    "expansion candidates for the 10% campaign."
                ),
                evidence={
                    "min_samples": FORECAST_CALIBRATION_MIN_SAMPLES,
                    "warn_below": FORECAST_HIT_RATE_WARN_BELOW,
                    "weak_buckets": weak_buckets,
                },
            )
        )
    samples = len(calibrated)
    hit_count = sum(1 for row in calibrated if str(row.get("resolution_status") or "").upper() == "HIT")
    hit_rate = hit_count / samples if samples else 0.0
    if samples >= FORECAST_CALIBRATION_MIN_SAMPLES and hit_rate < FORECAST_HIT_RATE_WARN_BELOW:
        findings.append(
            _finding(
                run_id=run_id,
                priority="P1",
                layer="forecast",
                code="DIRECTIONAL_FORECAST_HIT_RATE_WEAK",
                message=(
                    f"directional_forecast HIT rate is {hit_rate:.1%} "
                    f"over {samples} calibrated sample(s)"
                ),
                next_action=(
                    "Rank the weakest pair/direction buckets and adjust forecast priors, target geometry, "
                    "or range-location handling before increasing opportunity frequency."
                ),
                evidence={
                    "samples": samples,
                    "hit_count": hit_count,
                    "hit_rate": round(hit_rate, 4),
                    "warn_below": FORECAST_HIT_RATE_WARN_BELOW,
                    "worst_buckets": _directional_forecast_worst_buckets(
                        calibrated,
                        min_samples=FORECAST_CALIBRATION_MIN_SAMPLES,
                    )[:8],
                },
            )
        )
    return findings


def _directional_forecast_has_target_invalidation(row: dict[str, Any]) -> bool:
    return row.get("predicted_target_price") is not None and row.get("predicted_invalidation_price") is not None


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
        and has_loss_asymmetry
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
                    ),
                    next_action=(
                        "Block new-risk confidence until execution_ledger.db worst segments prove repaired "
                        "close discipline or the trader route explicitly justifies the exception."
                    ),
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

    Recovery requires positive evidence, not absence of evidence: at least one
    gateway-attributable winning close, a non-negative gateway-attributable
    net, and no small-win/large-loss asymmetry (same 2x boundary as
    SMALL_WIN_LARGE_LOSS_ASYMMETRY) inside the window. TP fills and gateway
    closes of held positions keep generating this evidence even while fresh
    entries are blocked, so the gate can clear without bypassing risk
    validation; a truly flat blocked book stays blocked until the trailing
    window rolls the old losers out.
    """
    if effect_24h.get("error"):
        return None
    metrics = effect_24h.get("close_provenance_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    net = 0.0
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
        gross_profit += float(values.get("gross_profit_jpy") or 0.0)
        gross_loss += float(values.get("gross_loss_jpy") or 0.0)
        trades += int(values.get("trades") or 0)
        win_trades += int(values.get("win_trades") or 0)
        loss_trades += int(values.get("loss_trades") or 0)
    if win_trades < 1 or net < 0.0:
        return None
    avg_win = gross_profit / win_trades
    avg_loss = (gross_loss / loss_trades) if loss_trades else None
    if avg_loss is not None and avg_loss > avg_win * 2:
        return None
    return {
        "window_hours": 24.0,
        "gateway_net_jpy": net,
        "gateway_trades": trades,
        "gateway_win_trades": win_trades,
        "gateway_loss_trades": loss_trades,
        "gateway_avg_win_jpy": avg_win,
        "gateway_avg_loss_jpy_abs": avg_loss,
        "excluded_provenances": list(RECOVERY_EXCLUDED_CLOSE_PROVENANCES),
    }


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
    target_open: bool,
    live_ready: list[dict[str, Any]],
    active_positions: list[dict[str, Any]],
    pending_entry_orders: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if target_open and not live_ready:
        entry_path_occupied = bool(active_positions or pending_entry_orders)
        out.append(
            _finding(
                run_id=run_id,
                priority="P0" if not entry_path_occupied else "P1",
                layer="opportunity",
                code="TARGET_OPEN_NO_LIVE_READY_LANES",
                message="daily target is open but order_intents has no LIVE_READY lanes",
                next_action="Refresh market context and inspect top live blockers instead of ending flat without a named gate.",
                evidence={
                    "active_trader_positions": len(active_positions),
                    "trader_pending_entry_orders": _pending_entry_evidence(pending_entry_orders),
                    "status_counts": _intent_status_counts(intents),
                    "top_blockers": _top_intent_blockers(intents),
                    "dry_run_passed_live_readiness_blockers": _top_intent_live_readiness_blockers(
                        intents,
                        statuses={"DRY_RUN_PASSED"},
                    ),
                    "dry_run_passed_forecast_gate_diagnostics": _dry_run_passed_forecast_gate_diagnostics(
                        intents
                    ),
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
    bucket_diag = (
        diagnostics.get("profitable_bucket_coverage")
        if isinstance(diagnostics.get("profitable_bucket_coverage"), dict)
        else {}
    )
    if not bucket_diag:
        return out
    blocked_edges = _blocked_profitable_bucket_edges(bucket_diag)
    if blocked_edges:
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
                    "blocked_edges": blocked_edges[:8],
                },
            )
        )
    context_supported = [
        edge for edge in blocked_edges
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


def _sidecar_findings(
    *,
    run_id: str,
    snapshot_ts: datetime | None,
    active_trade_ids: set[str],
    gpt_decision: dict[str, Any],
    sidecars: dict[str, tuple[_LoadedJson, Path]],
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
        if snapshot_ts is not None and freshness_at is not None and freshness_at < snapshot_ts:
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


def _decision_artifact_findings(
    *,
    run_id: str,
    gpt_loaded: _LoadedJson,
    trader_loaded: _LoadedJson,
    gpt_path: Path,
    trader_path: Path,
    target_open: bool,
    active_trade_ids: set[str],
    live_ready_lanes: int,
    pending_entry_orders: int,
    snapshot_ts: datetime | None,
    pending_entry_order_details: list[dict[str, Any]] | None = None,
    latest_gpt_stale_streak_before: int = 0,
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
        request_evidence_no_risk = (
            status == "ACCEPTED"
            and action == "REQUEST_EVIDENCE"
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
        if (
            snapshot_ts is not None
            and generated_at is not None
            and generated_at < snapshot_ts
            and not rejected_inert_receipt
            and not stale_close_for_closed_trades
            and not accepted_trade_consumed_by_pending_entry
        ):
            priority = (
                "P0"
                if active_trade_ids
                or (
                    target_open
                    and live_ready_lanes <= 0
                    and pending_entry_orders <= 0
                    and not request_evidence_no_risk
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


def _effect_metrics(db_path: Path, *, window_hours: float, now: datetime) -> dict[str, Any]:
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    rows: list[sqlite3.Row] = []
    accepted_close_rows: list[sqlite3.Row] = []
    gateway_close_trade_ids: set[str] = set()
    gateway_close_order_ids: set[str] = set()
    reconciled_close_trade_ids: set[str] = set()
    reconciled_close_order_ids: set[str] = set()
    gpt_close_trade_ids: set[str] = set()
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
                    stale_close_satisfied_trade_ids = _stale_gpt_close_satisfied_trade_ids(conn, columns)
                    accepted_close_rows = _broker_trade_close_accept_rows(conn, columns)
                    lane_select = "e.lane_id AS lane_id" if has_lane_id else "NULL AS lane_id"
                    open_lane_expr = "MAX(NULLIF(lane_id, ''))" if has_lane_id else "NULL"
                    query = f"""
                        WITH open_fills AS (
                            SELECT
                                trade_id,
                                {open_lane_expr} AS open_lane_id
                            FROM execution_events
                            WHERE event_type = 'ORDER_FILLED'
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
                            {_sql_column_or_null(columns, "order_id")},
                            {_sql_column_or_null(columns, "exit_reason")},
                            {_sql_column_or_null(columns, "raw_json")}
                        FROM execution_events e
                        LEFT JOIN open_fills ON open_fills.trade_id = e.trade_id
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
        _add_close_provenance_metric(close_provenance_metrics, close_provenance, value)
        exit_reason = str(_row_text(row, "exit_reason") or "").strip().upper()
        if exit_reason == "MARKET_ORDER_TRADE_CLOSE" and value < 0:
            _add_close_provenance_metric(
                market_order_trade_close_loss_provenance_metrics,
                close_provenance,
                value,
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
    if (trade_id and trade_id in reconciled_close_trade_ids) or (
        order_id and order_id in reconciled_close_order_ids
    ):
        return "GATEWAY_TRADE_CLOSE_RECONCILED"
    if trade_id and trade_id in stale_close_satisfied_trade_ids:
        return "STALE_GPT_CLOSE_SATISFIED"
    if trade_id and trade_id in gpt_close_trade_ids:
        return "GATEWAY_GPT_CLOSE_ACCEPTED"
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


def _add_close_provenance_metric(metrics: dict[str, dict[str, Any]], label: str, value: float) -> None:
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


def _sorted_close_provenance_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        label: {
            "trades": int(values.get("trades") or 0),
            "net_jpy": float(values.get("net_jpy") or 0.0),
            "gross_profit_jpy": float(values.get("gross_profit_jpy") or 0.0),
            "gross_loss_jpy": float(values.get("gross_loss_jpy") or 0.0),
            "win_trades": int(values.get("win_trades") or 0),
            "loss_trades": int(values.get("loss_trades") or 0),
        }
        for label, values in sorted(metrics.items())
    }


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

        seen: set[str] = set()
        structured_issue_seen = False
        for raw in result.get("risk_issues") or []:
            if not _risk_issue_blocks_live_readiness(raw):
                continue
            structured_issue_seen = (
                _count_live_readiness_issue(counts, examples, seen, raw, result) or structured_issue_seen
            )
        for raw in result.get("live_strategy_issues") or []:
            if not _strategy_issue_blocks_live_readiness(raw):
                continue
            structured_issue_seen = (
                _count_live_readiness_issue(counts, examples, seen, raw, result) or structured_issue_seen
            )
        if "live_strategy_issues" not in result:
            for raw in result.get("strategy_issues") or []:
                if isinstance(raw, dict) and str(raw.get("severity") or "").upper() != "BLOCK":
                    continue
                structured_issue_seen = (
                    _count_live_readiness_issue(counts, examples, seen, raw, result) or structured_issue_seen
                )
        if not structured_issue_seen:
            for raw in result.get("live_blockers") or []:
                _count_live_readiness_issue(counts, examples, seen, raw, result)

    return [
        {"message": message, "count": count, "example_lanes": examples.get(message, [])[:3]}
        for message, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]


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


def _count_live_readiness_issue(
    counts: dict[str, int],
    examples: dict[str, list[str]],
    seen: set[str],
    raw: Any,
    result: dict[str, Any],
) -> bool:
    text = _issue_text(raw)
    if not text or text in seen:
        return False
    seen.add(text)
    counts[text] = counts.get(text, 0) + 1
    lane_id = str(result.get("lane_id") or "")
    if lane_id and lane_id not in examples.setdefault(text, []):
        examples[text].append(lane_id)
    return True


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
                status_text = " ".join(
                    str(row.get(field) or "").upper()
                    for field in ("action", "status", "verdict", "reason")
                )
                if any(token in status_text for token in ("REVIEW_CLOSE", "RECOMMEND_CLOSE", "REVIEW_EXIT", "BROKEN", "REQUIRE_THESIS_REPAIR", "UNVERIFIABLE")):
                    refs.append(
                        {
                            "source": name,
                            "trade_id": trade_id,
                            "pair": row.get("pair"),
                            "side": row.get("side"),
                            "status": row.get("status"),
                            "verdict": row.get("verdict"),
                            "action": row.get("action"),
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


def _next_actions(findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "priority": str(item.get("priority") or ""),
            "code": str(item.get("code") or ""),
            "next_action": str(item.get("next_action") or ""),
        }
        for item in findings[:3]
    ]


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
