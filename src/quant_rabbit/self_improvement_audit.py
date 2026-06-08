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


# Match the live-entry telemetry gate's cycle-preflight tolerance. Market
# refresh can take a few minutes after projection verification; inside this
# window the projection is not yet a stale-state defect.
PROJECTION_PENDING_EXPIRY_GRACE_SECONDS = _env_nonnegative_float(
    "QR_PROJECTION_PENDING_EXPIRY_GRACE_SECONDS",
    300.0,
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
        history_db_path: Path | None = None,
        output_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
        report_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT_REPORT,
    ) -> None:
        self.db_path = db_path
        self.history_db_path = history_db_path or db_path
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
        active_trade_ids = {str(item.get("trade_id") or "") for item in active_positions if item.get("trade_id")}
        effect = _effect_metrics(self.db_path, window_hours=window_hours, now=clock)
        effect_24h = _effect_metrics(self.db_path, window_hours=24.0, now=clock)
        profitability_streak_before = self._history_code_streak(PROFITABILITY_DISCIPLINE_CODES)

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
                snapshot_ts=snapshot_ts,
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
            f"- LIVE_READY lanes: `{runtime['live_ready_lanes']}`",
            f"- GPT status/action: `{runtime['gpt_status']}` / `{runtime['gpt_action']}`",
            "",
            "## Profitability",
            "",
            f"- Window `{payload['window_hours']}`h: trades `{window['closed_trades']}`, net `{window['net_jpy']:.2f}` JPY, PF `{_fmt_optional(window['profit_factor'])}`, expectancy `{_fmt_optional(window['expectancy_jpy'])}` JPY",
            f"- Last 24h: trades `{last_24h['closed_trades']}`, net `{last_24h['net_jpy']:.2f}` JPY, PF `{_fmt_optional(last_24h['profit_factor'])}`, expectancy `{_fmt_optional(last_24h['expectancy_jpy'])}` JPY",
            "",
            "## Findings",
            "",
        ]
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
        """Return consecutive prior audit runs containing every requested code."""
        if not required_codes or not self.history_db_path.exists():
            return 0
        try:
            with sqlite3.connect(f"file:{self.history_db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT run_uid
                    FROM self_improvement_audit_runs
                    ORDER BY ts_utc DESC
                    LIMIT 64
                    """
                ).fetchall()
                streak = 0
                for row in rows:
                    codes = {
                        str(item[0] or "")
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
            (
                str(item.get("priority") or ""),
                str(item.get("layer") or ""),
                str(item.get("code") or ""),
                str(item.get("message") or ""),
                str(item.get("next_action") or ""),
                json.dumps(item.get("evidence") or {}, ensure_ascii=False, sort_keys=True),
            )
            for item in findings
        )
    )


def _history_db_findings_signature(rows: list[sqlite3.Row]) -> tuple[tuple[str, str, str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(row["priority"] or ""),
                str(row["layer"] or ""),
                str(row["code"] or ""),
                str(row["message"] or ""),
                str(row["next_action"] or ""),
                str(row["evidence_json"] or "{}"),
            )
            for row in rows
        )
    )


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
                        e.pair,
                        e.side,
                        e.realized_pl_jpy,
                        e.exit_reason,
                        entries.gateway_lane_id,
                        entries.position_side
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
                          WHERE c.event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                            AND c.trade_id = e.trade_id
                      )
                    ORDER BY e.ts_utc DESC, e.event_uid DESC
                    """
                )
            )
    except sqlite3.Error as exc:
        error = str(exc)
    if error is not None:
        return []
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    unattributed: list[sqlite3.Row] = []
    for row in rows:
        ts = _parse_utc(row["ts_utc"])
        if ts is None or ts < cutoff:
            continue
        value = _maybe_float(row["realized_pl_jpy"])
        if value is None or value >= 0:
            continue
        unattributed.append(row)
    if not unattributed:
        return []
    net = sum(float(_maybe_float(row["realized_pl_jpy"]) or 0.0) for row in unattributed)
    examples = [
        {
            "ts_utc": str(row["ts_utc"] or ""),
            "event_uid": str(row["event_uid"] or ""),
            "event_type": str(row["event_type"] or ""),
            "trade_id": str(row["trade_id"] or ""),
            "pair": str(row["pair"] or ""),
            "side": str(row["side"] or row["position_side"] or ""),
            "original_side": str(row["position_side"] or ""),
            "gateway_lane_id": str(row["gateway_lane_id"] or ""),
            "realized_pl_jpy": _maybe_float(row["realized_pl_jpy"]),
            "exit_reason": str(row["exit_reason"] or ""),
        }
        for row in unattributed[:8]
    ]
    return [
        _finding(
            run_id=run_id,
            priority="P1",
            layer="execution_ledger",
            code="UNATTRIBUTED_MARKET_ORDER_CLOSES",
            message=(
                f"{len(unattributed)} negative bot-attributed market close event(s) lack "
                "a matching gateway close receipt"
            ),
            next_action=(
                "Persist every direct close_trade path as a position-execution receipt, then separate "
                "manual intervention from strategy/Gate A/B close performance before tuning exits."
            ),
            evidence={
                "window_hours": window_hours,
                "unattributed_loss_count": len(unattributed),
                "unattributed_loss_net_jpy": round(net, 4),
                "examples": examples,
            },
        )
    ]


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
        priority = "P0" if active_trade_ids else "P1"
        return [
            _finding(
                run_id=run_id,
                priority=priority,
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
    if (
        has_negative_expectancy
        and has_loss_asymmetry
        and current_discipline_streak >= max(1, PERSISTENT_PROFITABILITY_STREAK_MIN)
    ):
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
                    "system_defect_evidence": {
                        "profit_factor": pf,
                        "expectancy_jpy": expectancy,
                        "avg_win_jpy": avg_win,
                        "avg_loss_jpy_abs": avg_loss,
                        "worst_segments": effect.get("worst_segments", [])[:5],
                    },
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


def _intent_findings(
    *,
    run_id: str,
    intents: dict[str, Any],
    target_open: bool,
    live_ready: list[dict[str, Any]],
    active_positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if target_open and not live_ready:
        out.append(
            _finding(
                run_id=run_id,
                priority="P0" if not active_positions else "P1",
                layer="opportunity",
                code="TARGET_OPEN_NO_LIVE_READY_LANES",
                message="daily target is open but order_intents has no LIVE_READY lanes",
                next_action="Refresh market context and inspect top live blockers instead of ending flat without a named gate.",
                evidence={"active_trader_positions": len(active_positions), "top_blockers": _top_intent_blockers(intents)},
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
        return [
            _finding(
                run_id=run_id,
                priority="P1",
                layer="opportunity_context",
                code="ORDER_INTENTS_MARKET_CONTEXT_EVIDENCE_STALE",
                message=(
                    "order_intents were generated before the current market_context_matrix, so current "
                    "gold/oil/rates/equity/news context cannot be attributed to these candidates"
                ),
                next_action=(
                    "Regenerate generate-intents after the latest market-context-matrix, context-asset, and news "
                    "artifacts; do not judge non-FX/news effect from stale candidates."
                ),
                evidence={
                    "matrix_path": str(matrix_path),
                    "intents_generated_at_utc": intents_generated_at.isoformat(),
                    "matrix_generated_at_utc": matrix_generated_at.isoformat(),
                    "matrix_pairs": len(pairs),
                    "candidate_count": len(results),
                    "with_context_refs": len(with_context),
                    "status_counts": _result_status_counts(results),
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
    gateway_review_loss = int(_maybe_float(close_gate.get("gateway_review_exit_loss_side_market_close_count")) or 0)
    gateway_review_loss_net = _maybe_float(close_gate.get("gateway_review_exit_loss_side_market_close_net_jpy")) or 0.0
    gateway_gpt_loss = int(_maybe_float(close_gate.get("gateway_gpt_close_loss_side_market_close_count")) or 0)
    if status != "MEASURED" or not close_events:
        return []
    if (
        bot_attributed
        and gateway_sent
        and not broker_without_gateway
        and not (gateway_review_loss and gateway_review_loss_net < 0)
    ):
        return []
    if bot_attributed and broker_accept and not broker_without_gateway and not (
        gateway_review_loss and gateway_review_loss_net < 0
    ):
        return []
    if broker_without_gateway:
        if broker_without_gateway_sources.get("UNLABELED_BROKER_TRADE_CLOSE"):
            next_action = (
                "Broker accepted TRADE_CLOSE loss events are unlabeled by local gateway receipt; persist a local "
                "GATEWAY_TRADE_CLOSE_SENT/GPT_CLOSE source tag before counting them as Gate A/B evidence or "
                "changing live close policy."
            )
        else:
            next_action = (
                "Trace broker accepted TRADE_CLOSE orders back to GPT/gateway/operator source and persist a local "
                "GATEWAY_TRADE_CLOSE_SENT or equivalent source tag before relaxing or hardening live close policy."
            )
    elif gateway_review_loss and gateway_review_loss_net < 0:
        next_action = (
            "Separate legacy REVIEW_EXIT closes from current GPT_CLOSE Gate A/B closes; keep plain auto-close blocked "
            "until structural replay proves REVIEW_EXIT timing positive."
        )
    else:
        next_action = (
            "Link gateway close receipts to filled trades and ablate hard Gate A, soft Gate A, "
            "Gate B, and no-gate exit variants offline before relaxing or hardening live close policy."
        )
    return [
        _finding(
            run_id=run_id,
            priority="P1",
            layer="assumption_ablation",
            code="CLOSE_GATE_ABLATION_NOT_ATTRIBUTABLE",
            message=(
                "CLOSE Gate A/B performance is not attributable enough to call the gate policy "
                "verified or disproven"
            ),
            next_action=next_action,
            evidence={
                "ai_test_bot_backtest_path": str(path),
                "status": status,
                "close_events": close_events,
                "bot_attributed_close_events": bot_attributed,
                "gateway_close_sent_events": gateway_sent,
                "broker_trade_close_accept_events": broker_accept,
                "gateway_gpt_close_loss_side_market_close_count": gateway_gpt_loss,
                "gateway_review_exit_loss_side_market_close_count": gateway_review_loss,
                "gateway_review_exit_loss_side_market_close_net_jpy": gateway_review_loss_net,
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
        generated_at = _parse_utc((loaded.payload or {}).get("generated_at_utc"))
        if snapshot_ts is not None and generated_at is not None and generated_at < snapshot_ts:
            out.append(
                _finding(
                    run_id=run_id,
                    priority="P0" if name == "position_management" else "P1",
                    layer="position_review",
                    code=f"{name.upper()}_STALE",
                    message=f"{name} sidecar predates broker snapshot for active positions",
                    next_action=f"Rerun {name.replace('_', '-')} or route to the refresh branch before new exposure.",
                    evidence={"generated_at_utc": generated_at.isoformat(), "snapshot_fetched_at_utc": snapshot_ts.isoformat()},
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
    snapshot_ts: datetime | None,
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
    if trader_loaded.error is not None:
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
        if snapshot_ts is not None and generated_at is not None and generated_at < snapshot_ts:
            priority = "P0" if target_open or active_trade_ids else "P1"
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
                    },
                )
            )
        issues = gpt_loaded.payload.get("verification_issues")
        if isinstance(issues, list) and issues:
            blocking = [item for item in issues if isinstance(item, dict) and str(item.get("severity") or "").upper() == "BLOCK"]
            if blocking:
                decision = gpt_loaded.payload.get("decision") if isinstance(gpt_loaded.payload.get("decision"), dict) else {}
                action = str(decision.get("action") or "").upper()
                close_ids = {
                    str(item)
                    for item in decision.get("close_trade_ids", []) or []
                    if str(item)
                }
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
                status = str(gpt_loaded.payload.get("status") or "").upper()
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
    error: str | None = None
    if db_path.exists():
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                conn.row_factory = sqlite3.Row
                columns = _table_columns(conn, "execution_events")
                has_trade_id = "trade_id" in columns
                has_lane_id = "lane_id" in columns
                if has_trade_id and has_lane_id:
                    query = """
                        WITH open_fills AS (
                            SELECT
                                trade_id,
                                MAX(NULLIF(lane_id, '')) AS open_lane_id
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
                            e.lane_id,
                            open_fills.open_lane_id
                        FROM execution_events e
                        LEFT JOIN open_fills ON open_fills.trade_id = e.trade_id
                        WHERE e.event_type = 'TRADE_CLOSED'
                          AND e.realized_pl_jpy IS NOT NULL
                    """
                else:
                    query = """
                        SELECT ts_utc, pair, side, realized_pl_jpy
                        FROM execution_events
                        WHERE event_type = 'TRADE_CLOSED'
                          AND realized_pl_jpy IS NOT NULL
                    """
                rows = list(
                    conn.execute(query)
                )
        except sqlite3.Error as exc:
            error = str(exc)
    else:
        error = "missing"
    pls: list[float] = []
    segments: dict[tuple[str, str, str], dict[str, Any]] = {}
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
        segment = segments.setdefault(key, {"values": [], "lane_ids": set(), "trade_ids": set()})
        segment["values"].append(value)
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
        }
        for (pair, side, method), segment in segments.items()
    ]
    worst_segments.sort(key=lambda item: float(item["net_jpy"]))
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
        "error": error,
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


def _live_ready_results(intents: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in intents.get("results", []) or []
        if isinstance(item, dict) and str(item.get("status") or "") == "LIVE_READY"
    ]


def _top_intent_blockers(intents: dict[str, Any]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        for key in ("risk_issues", "strategy_issues", "live_blockers"):
            for raw in result.get(key) or []:
                text = _issue_text(raw)
                if text:
                    counts[text] = counts.get(text, 0) + 1
    return [
        {"message": message, "count": count}
        for message, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
    ]


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


def _fmt_optional(value: Any) -> str:
    number = _maybe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}"


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
