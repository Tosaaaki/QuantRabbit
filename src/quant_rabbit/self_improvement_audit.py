from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ENTRY_THESIS_LEDGER,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_FORECAST_HISTORY,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LEARNING_AUDIT,
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
        memory_health_path: Path = DEFAULT_MEMORY_HEALTH,
        learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
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
        window_hours: float = 168.0,
        now: datetime | None = None,
    ) -> SelfImprovementAuditSummary:
        clock = _to_utc(now or datetime.now(timezone.utc))
        run_id = clock.isoformat()
        self._init_history_db()

        snapshot_loaded = _read_json(snapshot_path)
        target_loaded = _read_json(target_state_path)
        intents_loaded = _read_json(order_intents_path)
        memory_loaded = _read_json(memory_health_path)
        learning_loaded = _read_json(learning_audit_path)
        verification_loaded = _read_json(verification_ledger_path)
        gpt_loaded = _read_json(gpt_decision_path)
        trader_loaded = _read_json(trader_decision_path)
        position_management_loaded = _read_json(position_management_path)
        thesis_evolution_loaded = _read_json(thesis_evolution_path)
        position_thesis_loaded = _read_json(position_thesis_path)
        forecast_persistence_loaded = _read_json(forecast_persistence_path)

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
                "memory_health": str(memory_health_path),
                "learning_audit": str(learning_audit_path),
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
    return out


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
        issues = gpt_loaded.payload.get("verification_issues")
        if isinstance(issues, list) and issues:
            blocking = [item for item in issues if isinstance(item, dict) and str(item.get("severity") or "").upper() == "BLOCK"]
            if blocking:
                out.append(
                    _finding(
                        run_id=run_id,
                        priority="P0",
                        layer="decision_history",
                        code="LATEST_GPT_DECISION_HAS_BLOCKING_ISSUES",
                        message=f"latest GPT decision contains {len(blocking)} blocking verification issue(s)",
                        next_action="Do not reuse the receipt; fix the first blocker and verify a fresh decision.",
                        evidence={"codes": [str(item.get("code") or "") for item in blocking[:12]]},
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
                rows = list(
                    conn.execute(
                        """
                        SELECT ts_utc, pair, side, realized_pl_jpy
                        FROM execution_events
                        WHERE event_type = 'TRADE_CLOSED'
                          AND realized_pl_jpy IS NOT NULL
                        """
                    )
                )
        except sqlite3.Error as exc:
            error = str(exc)
    else:
        error = "missing"
    pls: list[float] = []
    segments: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        ts = _parse_utc(row["ts_utc"])
        if ts is None or ts < cutoff:
            continue
        value = _maybe_float(row["realized_pl_jpy"])
        if value is None:
            continue
        pls.append(value)
        key = (str(row["pair"] or "UNKNOWN"), str(row["side"] or "UNKNOWN"))
        segments.setdefault(key, []).append(value)
    gross_profit = sum(value for value in pls if value > 0)
    losses = [value for value in pls if value < 0]
    gross_loss_abs = abs(sum(losses))
    count = len(pls)
    wins = [value for value in pls if value > 0]
    net = sum(pls)
    worst_segments = [
        {"pair": pair, "side": side, "trades": len(values), "net_jpy": sum(values), "expectancy_jpy": sum(values) / len(values)}
        for (pair, side), values in segments.items()
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
