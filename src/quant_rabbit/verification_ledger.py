from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_AI_TEST_BOT_BACKTEST,
    DEFAULT_BROKER_SNAPSHOT,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_LIVE_ORDER_REQUEST,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_OUTCOME_MART,
    DEFAULT_POSITION_EXECUTION,
    DEFAULT_POST_TRADE_LEARNING,
    DEFAULT_VERIFICATION_LEDGER,
    DEFAULT_VERIFICATION_LEDGER_REPORT,
)


@dataclass(frozen=True)
class VerificationLedgerSummary:
    db_path: Path
    output_path: Path
    report_path: Path
    status: str
    observations_inserted: int
    measurements_inserted: int
    blocking_observations: int
    missing_observations: int
    closed_trades: int
    net_jpy: float
    profit_factor: float | None
    win_rate: float | None
    expectancy_jpy: float | None


class VerificationLedger:
    """Normalize cycle verifiability and outcome metrics into SQLite.

    `ExecutionLedger` stores broker truth and gateway receipts. This layer stores
    the audit state around those receipts: which checks were reproducible,
    which artifacts were missing, and whether recent outcomes improved.
    """

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        output_path: Path = DEFAULT_VERIFICATION_LEDGER,
        report_path: Path = DEFAULT_VERIFICATION_LEDGER_REPORT,
    ) -> None:
        self.db_path = db_path
        self.output_path = output_path
        self.report_path = report_path

    def run(
        self,
        *,
        snapshot_path: Path = DEFAULT_BROKER_SNAPSHOT,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        gpt_decision_path: Path = DEFAULT_GPT_TRADER_DECISION,
        live_order_path: Path = DEFAULT_LIVE_ORDER_REQUEST,
        position_execution_path: Path = DEFAULT_POSITION_EXECUTION,
        thesis_evolution_path: Path = Path("data/thesis_evolution_report.json"),
        position_thesis_path: Path = Path("data/position_thesis_report.json"),
        forecast_persistence_path: Path = Path("data/forecast_persistence_report.json"),
        ai_backtest_path: Path = DEFAULT_AI_TEST_BOT_BACKTEST,
        outcome_mart_path: Path = DEFAULT_OUTCOME_MART,
        post_trade_learning_path: Path = DEFAULT_POST_TRADE_LEARNING,
        ai_attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
        window_hours: float = 168.0,
        now: datetime | None = None,
    ) -> VerificationLedgerSummary:
        self._init_db()
        clock = _to_utc(now or datetime.now(timezone.utc))
        run_id = clock.isoformat()

        payloads = {
            "broker_snapshot": _read_json(snapshot_path),
            "order_intents": _read_json(order_intents_path),
            "gpt_decision": _read_json(gpt_decision_path),
            "live_order": _read_json(live_order_path),
            "position_execution": _read_json(position_execution_path),
            "thesis_evolution": _read_json(thesis_evolution_path),
            "position_thesis": _read_json(position_thesis_path),
            "forecast_persistence": _read_json(forecast_persistence_path),
            "ai_backtest": _read_json(ai_backtest_path),
            "outcome_mart": _read_json(outcome_mart_path),
            "post_trade_learning": _read_json(post_trade_learning_path),
            "ai_attack_advice": _read_json(ai_attack_advice_path),
            "learning_audit": _read_json(learning_audit_path),
        }
        paths = {
            "broker_snapshot": snapshot_path,
            "order_intents": order_intents_path,
            "gpt_decision": gpt_decision_path,
            "live_order": live_order_path,
            "position_execution": position_execution_path,
            "thesis_evolution": thesis_evolution_path,
            "position_thesis": position_thesis_path,
            "forecast_persistence": forecast_persistence_path,
            "ai_backtest": ai_backtest_path,
            "outcome_mart": outcome_mart_path,
            "post_trade_learning": post_trade_learning_path,
            "ai_attack_advice": ai_attack_advice_path,
            "learning_audit": learning_audit_path,
        }

        observations: list[dict[str, Any]] = []
        required_learning_sources = {"ai_backtest", "outcome_mart", "post_trade_learning", "ai_attack_advice", "learning_audit"}
        for source, loaded in payloads.items():
            missing_required_learning = source in required_learning_sources and loaded.payload is None
            observations.append(
                _observation(
                    run_id=run_id,
                    source=source,
                    source_path=paths[source],
                    subject_type="artifact",
                    subject_id=source,
                    check_name="artifact_readable",
                    status="PASS" if loaded.payload is not None else "MISSING",
                    severity="BLOCK" if missing_required_learning else ("INFO" if loaded.payload is not None else "WARN"),
                    evidence={"error": loaded.error} if loaded.error else {},
                )
            )

        observations.extend(_broker_snapshot_observations(run_id, snapshot_path, payloads["broker_snapshot"].payload))
        observations.extend(_order_intent_observations(run_id, order_intents_path, payloads["order_intents"].payload))
        observations.extend(_gpt_decision_observations(run_id, gpt_decision_path, payloads["gpt_decision"].payload))
        observations.extend(_gateway_observations(run_id, live_order_path, "live_order", payloads["live_order"].payload))
        observations.extend(
            _gateway_observations(
                run_id,
                position_execution_path,
                "position_execution",
                payloads["position_execution"].payload,
            )
        )
        observations.extend(
            _thesis_evolution_observations(
                run_id,
                thesis_evolution_path,
                payloads["thesis_evolution"].payload,
            )
        )
        observations.extend(
            _position_thesis_observations(
                run_id,
                position_thesis_path,
                payloads["position_thesis"].payload,
            )
        )
        observations.extend(
            _forecast_persistence_observations(
                run_id,
                forecast_persistence_path,
                payloads["forecast_persistence"].payload,
            )
        )
        observations.extend(
            _ai_backtest_observations(
                run_id,
                ai_backtest_path,
                payloads["ai_backtest"].payload,
            )
        )
        observations.extend(
            _outcome_mart_observations(
                run_id,
                outcome_mart_path,
                payloads["outcome_mart"].payload,
            )
        )
        observations.extend(
            _post_trade_learning_observations(
                run_id,
                post_trade_learning_path,
                payloads["post_trade_learning"].payload,
            )
        )
        observations.extend(
            _ai_attack_advice_observations(
                run_id,
                ai_attack_advice_path,
                payloads["ai_attack_advice"].payload,
            )
        )
        observations.extend(
            _learning_audit_observations(
                run_id,
                learning_audit_path,
                payloads["learning_audit"].payload,
            )
        )
        observations.extend(
            _execution_ledger_sync_observations(
                run_id,
                self.db_path,
                payloads["broker_snapshot"].payload,
            )
        )
        observations.extend(
            _ledger_close_gate_observations(
                self.db_path,
                inserted_at_utc=run_id,
            )
        )
        observations = _dedupe_observations(observations)

        effect = _effect_metrics(self.db_path, window_hours=window_hours, now=clock)
        measurements = _measurements_from_effect(run_id, window_hours=window_hours, effect=effect)

        with self._connect() as conn:
            observations_inserted = sum(_insert_observation(conn, item) for item in observations)
            measurements_inserted = sum(_insert_measurement(conn, item) for item in measurements)

        blocking = [
            item
            for item in observations
            if str(item.get("status")) in {"BLOCK", "UNVERIFIABLE"}
            or str(item.get("severity")) == "BLOCK"
        ]
        missing = [item for item in observations if str(item.get("status")) == "MISSING"]
        summary = VerificationLedgerSummary(
            db_path=self.db_path,
            output_path=self.output_path,
            report_path=self.report_path,
            status="BLOCKED" if blocking else "OK",
            observations_inserted=observations_inserted,
            measurements_inserted=measurements_inserted,
            blocking_observations=len(blocking),
            missing_observations=len(missing),
            closed_trades=int(effect.get("closed_trades") or 0),
            net_jpy=float(effect.get("net_jpy") or 0.0),
            profit_factor=_maybe_float(effect.get("profit_factor")),
            win_rate=_maybe_float(effect.get("win_rate")),
            expectancy_jpy=_maybe_float(effect.get("expectancy_jpy")),
        )
        self._write_output(
            summary,
            observations=observations,
            measurements=measurements,
            window_hours=window_hours,
        )
        self._write_report(
            summary,
            observations=observations,
            measurements=measurements,
            window_hours=window_hours,
        )
        return summary

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS verification_observations (
                    observation_uid TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_path TEXT,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT,
                    check_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT,
                    metric_value REAL,
                    metric_unit TEXT,
                    evidence_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_verification_observations_ts
                    ON verification_observations(ts_utc);
                CREATE INDEX IF NOT EXISTS idx_verification_observations_status
                    ON verification_observations(status, severity);
                CREATE INDEX IF NOT EXISTS idx_verification_observations_subject
                    ON verification_observations(subject_type, subject_id);

                CREATE TABLE IF NOT EXISTS effect_measurements (
                    measurement_uid TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    window_hours REAL NOT NULL,
                    segment TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_unit TEXT,
                    sample_size INTEGER NOT NULL,
                    evidence_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_effect_measurements_ts
                    ON effect_measurements(ts_utc);
                CREATE INDEX IF NOT EXISTS idx_effect_measurements_metric
                    ON effect_measurements(segment, metric_name);
                """
            )

    def _write_report(
        self,
        summary: VerificationLedgerSummary,
        *,
        observations: list[dict[str, Any]],
        measurements: list[dict[str, Any]],
        window_hours: float,
    ) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        blocking = [
            item
            for item in observations
            if str(item.get("status")) in {"BLOCK", "UNVERIFIABLE"}
            or str(item.get("severity")) == "BLOCK"
        ]
        missing = [item for item in observations if str(item.get("status")) == "MISSING"]
        lines = [
            "# Verification Ledger Report",
            "",
            f"- Generated at UTC: `{datetime.now(timezone.utc).isoformat()}`",
            f"- Status: `{summary.status}`",
            f"- DB: `{summary.db_path}`",
            f"- Observations inserted: `{summary.observations_inserted}`",
            f"- Measurements inserted: `{summary.measurements_inserted}`",
            f"- Blocking observations: `{summary.blocking_observations}`",
            f"- Missing observations: `{summary.missing_observations}`",
            "",
            "## Effect Window",
            "",
            f"- Window hours: `{window_hours}`",
            f"- Closed trades: `{summary.closed_trades}`",
            f"- Net JPY: `{summary.net_jpy:.1f}`",
            f"- Profit factor: `{_format_optional(summary.profit_factor)}`",
            f"- Win rate: `{_format_optional(summary.win_rate)}`",
            f"- Expectancy JPY: `{_format_optional(summary.expectancy_jpy)}`",
        ]
        if summary.closed_trades < 30:
            lines.append("- Sample warning: `INSUFFICIENT_SAMPLE_LT_30`")
        lines.extend(["", "## Blocking Evidence", ""])
        if blocking:
            for item in blocking[:20]:
                lines.append(
                    f"- `{item['source']}` `{item['check_name']}` "
                    f"{item.get('subject_type')}={item.get('subject_id')}: `{item['status']}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Missing Artifacts", ""])
        if missing:
            for item in missing[:20]:
                lines.append(f"- `{item['source']}` path=`{item.get('source_path')}`")
        else:
            lines.append("- none")
        learning = [
            item
            for item in observations
            if item["source"] in {"ai_backtest", "outcome_mart", "post_trade_learning", "ai_attack_advice", "learning_audit"}
            and item["check_name"] != "artifact_readable"
        ]
        lines.extend(["", "## Learning Evidence", ""])
        if learning:
            for item in learning[:20]:
                value = ""
                if item.get("metric_value") is not None:
                    value = f" value=`{item['metric_value']}`{item.get('metric_unit') or ''}"
                lines.append(
                    f"- `{item['source']}` `{item['check_name']}` "
                    f"{item.get('subject_type')}={item.get('subject_id')}: `{item['status']}`{value}"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Measurement Contract", ""])
        lines.extend(
            [
                "- Verification observations are append-only DB rows; do not overwrite a prior cycle's evidence.",
                "- Effect metrics are computed from `execution_events` broker/gateway truth for the declared window.",
                "- A metric with fewer than 30 closed trades is tracked but not treated as statistically stable.",
                "- Learning artifacts may influence live-ready lane ranking, but every influenced recommended lane is recorded here and cannot override hard risk/gateway gates.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_output(
        self,
        summary: VerificationLedgerSummary,
        *,
        observations: list[dict[str, Any]],
        measurements: list[dict[str, Any]],
        window_hours: float,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        blocking = [
            item
            for item in observations
            if str(item.get("status")) in {"BLOCK", "UNVERIFIABLE"}
            or str(item.get("severity")) == "BLOCK"
        ]
        missing = [item for item in observations if str(item.get("status")) == "MISSING"]
        learning = [
            item
            for item in observations
            if item["source"] in {"ai_backtest", "outcome_mart", "post_trade_learning", "ai_attack_advice", "learning_audit"}
            and item["check_name"] != "artifact_readable"
        ]
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": summary.status,
            "db_path": str(summary.db_path),
            "report_path": str(summary.report_path),
            "observations_inserted": summary.observations_inserted,
            "measurements_inserted": summary.measurements_inserted,
            "blocking_observations": summary.blocking_observations,
            "missing_observations": summary.missing_observations,
            "effect_metrics": {
                "window_hours": window_hours,
                "closed_trades": summary.closed_trades,
                "net_jpy": summary.net_jpy,
                "profit_factor": summary.profit_factor,
                "win_rate": summary.win_rate,
                "expectancy_jpy": summary.expectancy_jpy,
                "sample_warning": "INSUFFICIENT_SAMPLE_LT_30" if summary.closed_trades < 30 else None,
            },
            "blocking_evidence": [_observation_packet(item) for item in blocking[:20]],
            "missing_artifacts": [_observation_packet(item) for item in missing[:20]],
            "learning_evidence": [_observation_packet(item) for item in learning[:20]],
            "measurements": [_measurement_packet(item) for item in measurements[:20]],
            "contract": {
                "read_only": True,
                "live_permission": False,
                "sqlite_tables": ["verification_observations", "effect_measurements"],
                "json_packet_is_trader_readable": True,
                "markdown_report_is_operator_readable": True,
                "learning_cannot_override_risk_or_gateway_gates": True,
            },
        }
        self.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass(frozen=True)
class _LoadedJson:
    payload: dict[str, Any] | None
    error: str | None = None


def _read_json(path: Path) -> _LoadedJson:
    if not path.exists():
        return _LoadedJson(None, "missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _LoadedJson(None, str(exc))
    if not isinstance(payload, dict):
        return _LoadedJson(None, "json root is not an object")
    return _LoadedJson(payload)


def _observation(
    *,
    run_id: str,
    source: str,
    source_path: Path | None,
    subject_type: str,
    subject_id: str | None,
    check_name: str,
    status: str,
    severity: str = "INFO",
    metric_value: float | None = None,
    metric_unit: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subject = subject_id or ""
    uid = f"verification:{run_id}:{source}:{subject_type}:{subject}:{check_name}"
    return {
        "observation_uid": uid,
        "ts_utc": run_id,
        "source": source,
        "source_path": str(source_path) if source_path else None,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "metric_value": metric_value,
        "metric_unit": metric_unit,
        "evidence_json": _json(evidence or {}),
        "inserted_at_utc": run_id,
    }


def _broker_snapshot_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    positions = payload.get("positions") if isinstance(payload.get("positions"), list) else []
    orders = payload.get("orders") if isinstance(payload.get("orders"), list) else []
    account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
    return [
        _observation(
            run_id=run_id,
            source="broker_snapshot",
            source_path=path,
            subject_type="broker_snapshot",
            subject_id="positions",
            check_name="positions_count",
            status="PASS",
            metric_value=float(len(positions)),
            metric_unit="count",
        ),
        _observation(
            run_id=run_id,
            source="broker_snapshot",
            source_path=path,
            subject_type="broker_snapshot",
            subject_id="orders",
            check_name="orders_count",
            status="PASS",
            metric_value=float(len(orders)),
            metric_unit="count",
        ),
        _observation(
            run_id=run_id,
            source="broker_snapshot",
            source_path=path,
            subject_type="broker_snapshot",
            subject_id="last_transaction_id",
            check_name="last_transaction_id_present",
            status="PASS" if account.get("last_transaction_id") else "MISSING",
            severity="INFO" if account.get("last_transaction_id") else "WARN",
            evidence={"last_transaction_id": account.get("last_transaction_id")},
        ),
    ]


def _order_intent_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    observations = [
        _observation(
            run_id=run_id,
            source="order_intents",
            source_path=path,
            subject_type="order_intents",
            subject_id="live_ready",
            check_name="live_ready_count",
            status="PASS",
            metric_value=float(sum(1 for item in rows if isinstance(item, dict) and item.get("status") == "LIVE_READY")),
            metric_unit="count",
        )
    ]
    for item in rows:
        if not isinstance(item, dict):
            continue
        lane_id = str(item.get("lane_id") or "")
        issues: list[Any] = []
        for key in ("risk_issues", "strategy_issues", "live_blockers"):
            raw = item.get(key)
            if isinstance(raw, list):
                issues.extend(x for x in raw if isinstance(x, (dict, str)) and x)
        blockers = [
            x
            for x in issues
            if (
                isinstance(x, dict)
                and (
                    str(x.get("severity") or "").upper() == "BLOCK"
                    or str(x.get("status") or "").upper() in {"BLOCK", "BLOCKED"}
                )
            )
            or isinstance(x, str)
        ]
        if blockers:
            observations.append(
                _observation(
                    run_id=run_id,
                    source="order_intents",
                    source_path=path,
                    subject_type="lane",
                    subject_id=lane_id,
                    check_name="lane_blockers",
                    status="BLOCK",
                    severity="BLOCK",
                    metric_value=float(len(blockers)),
                    metric_unit="count",
                    evidence={"blockers": blockers[:10]},
                )
            )
    return observations


def _gpt_decision_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    status = str(payload.get("status") or "UNKNOWN")
    issues = payload.get("verification_issues") if isinstance(payload.get("verification_issues"), list) else []
    close_gate_evidence = (
        payload.get("close_gate_evidence")
        if isinstance(payload.get("close_gate_evidence"), list)
        else []
    )
    observations = [
        _observation(
            run_id=run_id,
            source="gpt_decision",
            source_path=path,
            subject_type="gpt_decision",
            subject_id=status,
            check_name="verification_status",
            status="PASS" if status == "ACCEPTED" else ("BLOCK" if status == "REJECTED" else "WARN"),
            severity="INFO" if status == "ACCEPTED" else ("BLOCK" if status == "REJECTED" else "WARN"),
            evidence={"action": (payload.get("decision") or {}).get("action") if isinstance(payload.get("decision"), dict) else None},
        )
    ]
    for idx, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "WARN").upper()
        observations.append(
            _observation(
                run_id=run_id,
                source="gpt_decision",
                source_path=path,
                subject_type="verification_issue",
                subject_id=str(issue.get("code") or idx),
                check_name="verification_issue",
                status="BLOCK" if severity == "BLOCK" else "WARN",
                severity=severity,
                evidence=issue,
            )
        )
    for idx, evidence in enumerate(close_gate_evidence):
        if not isinstance(evidence, dict):
            continue
        observations.append(
            _close_gate_evidence_observation(
                ts_utc=str(payload.get("generated_at_utc") or run_id),
                source="gpt_decision",
                source_path=path,
                trade_id=str(evidence.get("trade_id") or idx),
                index=idx,
                evidence=evidence,
                inserted_at_utc=run_id,
            )
        )
    return observations


def _ledger_close_gate_observations(db_path: Path, *, inserted_at_utc: str) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    collected: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            if "gateway_receipts" in tables:
                collected.extend(_gateway_receipt_close_gate_observations(conn, inserted_at_utc=inserted_at_utc))
            if "execution_events" in tables:
                collected.extend(_execution_event_close_gate_observations(conn, inserted_at_utc=inserted_at_utc))
    except sqlite3.Error:
        return []
    return _dedupe_observations(collected)


def _gateway_receipt_close_gate_observations(
    conn: sqlite3.Connection,
    *,
    inserted_at_utc: str,
) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "gateway_receipts")
    required = {"ts_utc", "kind", "path", "payload_json"}
    if not required.issubset(columns):
        return []
    rows = conn.execute(
        """
        SELECT ts_utc, path, payload_json
        FROM gateway_receipts
        WHERE kind = 'gpt_decision'
        ORDER BY ts_utc ASC
        """
    ).fetchall()
    observations: list[dict[str, Any]] = []
    for row in rows:
        payload = _json_dict(row["payload_json"])
        if payload is None:
            continue
        ts_utc = str(payload.get("generated_at_utc") or row["ts_utc"] or inserted_at_utc)
        source_path = str(row["path"] or "")
        observations.extend(
            _close_gate_observations_from_payload(
                payload,
                ts_utc=ts_utc,
                source="gpt_decision",
                source_path=source_path or None,
                inserted_at_utc=inserted_at_utc,
            )
        )
        observations.extend(
            _missing_close_gate_observations_from_payload(
                payload,
                ts_utc=ts_utc,
                source="gpt_decision",
                source_path=source_path or None,
                inserted_at_utc=inserted_at_utc,
                context={},
            )
        )
    return observations


def _execution_event_close_gate_observations(
    conn: sqlite3.Connection,
    *,
    inserted_at_utc: str,
) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "execution_events")
    if "event_type" not in columns or "ts_utc" not in columns:
        return []
    event_uid_select = "event_uid" if "event_uid" in columns else "NULL AS event_uid"
    trade_id_select = "trade_id" if "trade_id" in columns else "NULL AS trade_id"
    raw_json_select = "raw_json" if "raw_json" in columns else "NULL AS raw_json"
    rows = conn.execute(
        f"""
        SELECT {event_uid_select}, ts_utc, {trade_id_select}, {raw_json_select}
        FROM execution_events
        WHERE event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
        ORDER BY ts_utc ASC
        """
    ).fetchall()
    observations: list[dict[str, Any]] = []
    for row in rows:
        trade_id = str(row["trade_id"] or "").strip()
        payload = _json_dict(row["raw_json"])
        if payload is None:
            payload = {
                "generated_at_utc": row["ts_utc"],
                "status": "ACCEPTED",
                "decision": {"action": "CLOSE", "close_trade_ids": [trade_id] if trade_id else []},
            }
        elif trade_id and not _payload_close_trade_ids(payload):
            payload = dict(payload)
            payload["status"] = payload.get("status") or "ACCEPTED"
            payload["decision"] = {"action": "CLOSE", "close_trade_ids": [trade_id]}
        ts_utc = str(payload.get("generated_at_utc") or row["ts_utc"] or inserted_at_utc)
        context = {"event_uid": row["event_uid"]} if row["event_uid"] else {}
        observations.extend(
            _close_gate_observations_from_payload(
                payload,
                ts_utc=ts_utc,
                source="execution_ledger",
                source_path=None,
                inserted_at_utc=inserted_at_utc,
            )
        )
        observations.extend(
            _missing_close_gate_observations_from_payload(
                payload,
                ts_utc=ts_utc,
                source="execution_ledger",
                source_path=None,
                inserted_at_utc=inserted_at_utc,
                context=context,
            )
        )
    return observations


def _close_gate_observations_from_payload(
    payload: dict[str, Any],
    *,
    ts_utc: str,
    source: str,
    source_path: Path | str | None,
    inserted_at_utc: str,
) -> list[dict[str, Any]]:
    close_gate_evidence = (
        payload.get("close_gate_evidence")
        if isinstance(payload.get("close_gate_evidence"), list)
        else []
    )
    observations: list[dict[str, Any]] = []
    for idx, evidence in enumerate(close_gate_evidence):
        if not isinstance(evidence, dict):
            continue
        observations.append(
            _close_gate_evidence_observation(
                ts_utc=ts_utc,
                source=source,
                source_path=source_path,
                trade_id=str(evidence.get("trade_id") or idx),
                index=idx,
                evidence=evidence,
                inserted_at_utc=inserted_at_utc,
            )
        )
    return observations


def _missing_close_gate_observations_from_payload(
    payload: dict[str, Any],
    *,
    ts_utc: str,
    source: str,
    source_path: Path | str | None,
    inserted_at_utc: str,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    close_trade_ids = _payload_close_trade_ids(payload)
    if not close_trade_ids:
        return []
    close_gate_evidence = (
        payload.get("close_gate_evidence")
        if isinstance(payload.get("close_gate_evidence"), list)
        else []
    )
    evidence_trade_ids = {
        str(item.get("trade_id") or "").strip()
        for item in close_gate_evidence
        if isinstance(item, dict)
    }
    observations: list[dict[str, Any]] = []
    for trade_id in close_trade_ids:
        if trade_id in evidence_trade_ids:
            continue
        observations.append(
            _close_gate_missing_observation(
                ts_utc=ts_utc,
                source=source,
                source_path=source_path,
                trade_id=trade_id,
                inserted_at_utc=inserted_at_utc,
                context=context,
            )
        )
    return observations


def _payload_close_trade_ids(payload: dict[str, Any]) -> list[str]:
    status = str(payload.get("status") or "").upper()
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else payload
    action = str(decision.get("action") or "").upper() if isinstance(decision, dict) else ""
    if status != "ACCEPTED" or action != "CLOSE":
        return []
    close_trade_ids = decision.get("close_trade_ids") if isinstance(decision.get("close_trade_ids"), list) else []
    return list(dict.fromkeys(str(item or "").strip() for item in close_trade_ids if str(item or "").strip()))


def _close_gate_evidence_observation(
    *,
    ts_utc: str,
    source: str,
    source_path: Path | str | None,
    trade_id: str,
    index: int,
    evidence: dict[str, Any],
    inserted_at_utc: str,
) -> dict[str, Any]:
    status_label = _close_gate_evidence_status(evidence)
    subject_id = str(trade_id or index)
    return {
        "observation_uid": f"verification:close_gate_evidence:{ts_utc}:{subject_id}:{index}",
        "ts_utc": ts_utc,
        "source": source,
        "source_path": str(source_path) if source_path else None,
        "subject_type": "close_gate",
        "subject_id": subject_id,
        "check_name": "close_gate_evidence",
        "status": status_label,
        "severity": "INFO" if status_label == "PASS" else "BLOCK",
        "metric_value": None,
        "metric_unit": None,
        "evidence_json": _json(evidence),
        "inserted_at_utc": inserted_at_utc,
    }


def _close_gate_missing_observation(
    *,
    ts_utc: str,
    source: str,
    source_path: Path | str | None,
    trade_id: str,
    inserted_at_utc: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "reason": "close_gate_evidence_missing",
        "trade_id": trade_id,
        "message": "accepted GPT CLOSE receipt lacks durable close_gate_evidence for this trade_id",
    }
    evidence.update({key: value for key, value in context.items() if value})
    return {
        "observation_uid": f"verification:close_gate_evidence_missing:{ts_utc}:{trade_id}",
        "ts_utc": ts_utc,
        "source": source,
        "source_path": str(source_path) if source_path else None,
        "subject_type": "close_gate",
        "subject_id": trade_id,
        "check_name": "close_gate_evidence",
        "status": "BLOCK",
        "severity": "BLOCK",
        "metric_value": None,
        "metric_unit": None,
        "evidence_json": _json(evidence),
        "inserted_at_utc": inserted_at_utc,
    }


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _json_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _close_gate_evidence_status(evidence: dict[str, Any]) -> str:
    if evidence.get("gate_a_invalidated") is not True:
        return "BLOCK"
    if evidence.get("same_direction_support_conflict"):
        return "BLOCK"
    if evidence.get("hard_timing_gate_required") is True:
        return "BLOCK"
    if (
        evidence.get("explicit_gate_b_required") is True
        and evidence.get("gate_b_explicit_operator_authorized") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("profitability_p0_context_required") is True
        and evidence.get("profitability_p0_context_cited") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("timing_audit_required") is True
        and evidence.get("timing_evidence_cited") is not True
    ):
        return "BLOCK"
    if (
        evidence.get("gate_b_standing_authorized") is not True
        and evidence.get("gate_b_explicit_operator_authorized") is not True
    ):
        return "BLOCK"
    return "PASS"


def _dedupe_observations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for item in observations:
        uid = str(item.get("observation_uid") or "")
        if not uid:
            continue
        by_uid[uid] = item
    return list(by_uid.values())


def _gateway_observations(
    run_id: str,
    path: Path,
    source: str,
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if payload is None:
        return []
    status = str(payload.get("status") or "UNKNOWN")
    observations = [
        _observation(
            run_id=run_id,
            source=source,
            source_path=path,
            subject_type=source,
            subject_id=status,
            check_name="gateway_status",
            status="BLOCK" if "GAP" in status or status == "BLOCKED" else "PASS",
            severity="BLOCK" if "GAP" in status else "INFO",
            evidence={"sent": bool(payload.get("sent"))},
        )
    ]
    if source == "live_order":
        observations.extend(_live_order_entry_thesis_observations(run_id, path, payload))
    issues = []
    for key in ("risk_issues", "strategy_issues"):
        raw = payload.get(key)
        if isinstance(raw, list):
            issues.extend(x for x in raw if isinstance(x, dict))
    for idx, issue in enumerate(issues):
        severity = str(issue.get("severity") or "WARN").upper()
        observations.append(
            _observation(
                run_id=run_id,
                source=source,
                source_path=path,
                subject_type="gateway_issue",
                subject_id=str(issue.get("code") or idx),
                check_name="gateway_issue",
                status="BLOCK" if severity == "BLOCK" else "WARN",
                severity=severity,
                evidence=issue,
            )
        )
    return observations


def _live_order_entry_thesis_observations(run_id: str, path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    orders = payload.get("orders") if isinstance(payload.get("orders"), list) else [payload]
    observations: list[dict[str, Any]] = []
    for idx, order in enumerate(orders):
        if not isinstance(order, dict):
            continue
        if not order.get("sent"):
            continue
        record = order.get("entry_thesis_record") if isinstance(order.get("entry_thesis_record"), dict) else None
        record_status = str((record or {}).get("status") or "MISSING")
        ok = record_status in {"RECORDED", "PENDING_RECORDED"}
        observations.append(
            _observation(
                run_id=run_id,
                source="live_order",
                source_path=path,
                subject_type="live_order",
                subject_id=str(order.get("lane_id") or idx),
                check_name="entry_thesis_recorded",
                status="PASS" if ok else "UNVERIFIABLE",
                severity="INFO" if ok else "BLOCK",
                evidence={"entry_thesis_record": record},
            )
        )
    return observations


def _thesis_evolution_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    coverage = payload.get("entry_thesis_coverage") if isinstance(payload.get("entry_thesis_coverage"), dict) else {}
    missing = int(coverage.get("missing") or 0)
    blocking_ids = coverage.get("blocking_trade_ids") if isinstance(coverage.get("blocking_trade_ids"), list) else []
    return [
        _observation(
            run_id=run_id,
            source="thesis_evolution",
            source_path=path,
            subject_type="entry_thesis_coverage",
            subject_id="missing",
            check_name="entry_thesis_coverage",
            status="UNVERIFIABLE" if missing or blocking_ids else "PASS",
            severity="BLOCK" if missing or blocking_ids else "INFO",
            metric_value=float(missing),
            metric_unit="count",
            evidence={"missing_trade_ids": coverage.get("missing_trade_ids"), "blocking_trade_ids": blocking_ids},
        )
    ]


def _position_thesis_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    assessments = payload.get("assessments") if isinstance(payload.get("assessments"), list) else []
    review = [item for item in assessments if isinstance(item, dict) and item.get("verdict") == "REVIEW_CLOSE"]
    return [
        _observation(
            run_id=run_id,
            source="position_thesis",
            source_path=path,
            subject_type="position_thesis",
            subject_id="review_close",
            check_name="review_close_count",
            status="WARN" if review else "PASS",
            severity="WARN" if review else "INFO",
            metric_value=float(len(review)),
            metric_unit="count",
            evidence={"trade_ids": [item.get("trade_id") for item in review]},
        )
    ]


def _forecast_persistence_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    verdicts = payload.get("verdicts") if isinstance(payload.get("verdicts"), list) else []
    review = [item for item in verdicts if isinstance(item, dict) and item.get("verdict") == "RECOMMEND_CLOSE"]
    return [
        _observation(
            run_id=run_id,
            source="forecast_persistence",
            source_path=path,
            subject_type="forecast_persistence",
            subject_id="recommend_close",
            check_name="recommend_close_count",
            status="WARN" if review else "PASS",
            severity="WARN" if review else "INFO",
            metric_value=float(len(review)),
            metric_unit="count",
            evidence={"trade_ids": [item.get("trade_id") for item in review]},
        )
    ]


def _ai_backtest_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    status = str(payload.get("status") or "UNKNOWN")
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    certified = status == "TARGET_COVERAGE_CERTIFIED" and not blockers
    profitable_research = status == "RESEARCH_PROFITABLE_NOT_CERTIFIED"
    if certified:
        audit_status = "PASS"
        severity = "INFO"
    elif profitable_research:
        audit_status = "WARN"
        severity = "WARN"
    else:
        audit_status = "BLOCK"
        severity = "BLOCK"
    selected_trades = _maybe_float(summary.get("selected_trades"))
    observations = [
        _observation(
            run_id=run_id,
            source="ai_backtest",
            source_path=path,
            subject_type="ai_backtest",
            subject_id=status,
            check_name="walk_forward_certification",
            status=audit_status,
            severity=severity,
            metric_value=_maybe_float(summary.get("total_managed_net_jpy")),
            metric_unit="JPY",
            evidence={
                "status": status,
                "blockers": blockers[:10],
                "training_days": payload.get("training_days"),
                "min_train_trades": payload.get("min_train_trades"),
                "max_active_buckets": payload.get("max_active_buckets"),
                "validation_days": summary.get("validation_days"),
                "selected_trades": summary.get("selected_trades"),
                "profit_factor": summary.get("profit_factor"),
            },
        ),
        _observation(
            run_id=run_id,
            source="ai_backtest",
            source_path=path,
            subject_type="ai_backtest",
            subject_id="live_permission",
            check_name="read_only_learning",
            status="BLOCK" if payload.get("live_permission") is True else "PASS",
            severity="BLOCK" if payload.get("live_permission") is True else "INFO",
            evidence={"live_permission": payload.get("live_permission")},
        ),
    ]
    if selected_trades is not None:
        observations.append(
            _observation(
                run_id=run_id,
                source="ai_backtest",
                source_path=path,
                subject_type="ai_backtest",
                subject_id="selected_trades",
                check_name="sample_size",
                status="WARN" if selected_trades < 30 else "PASS",
                severity="WARN" if selected_trades < 30 else "INFO",
                metric_value=selected_trades,
                metric_unit="count",
                evidence={"insufficient_sample_lt_30": selected_trades < 30},
            )
        )
    return observations


def _outcome_mart_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    validation = payload.get("condition_validation") if isinstance(payload.get("condition_validation"), dict) else {}
    coverage = payload.get("source_coverage") if isinstance(payload.get("source_coverage"), dict) else {}
    validated = int(validation.get("validated_outcomes") or 0)
    observations = [
        _observation(
            run_id=run_id,
            source="outcome_mart",
            source_path=path,
            subject_type="outcome_mart",
            subject_id="read_only",
            check_name="read_only_learning",
            status="BLOCK" if payload.get("live_permission") is True or payload.get("read_only") is False else "PASS",
            severity="BLOCK" if payload.get("live_permission") is True or payload.get("read_only") is False else "INFO",
            evidence={"read_only": payload.get("read_only"), "live_permission": payload.get("live_permission")},
        ),
        _observation(
            run_id=run_id,
            source="outcome_mart",
            source_path=path,
            subject_type="outcome_mart",
            subject_id=str(validation.get("status") or "UNKNOWN"),
            check_name="condition_walk_forward",
            status="PASS" if validated > 0 else "WARN",
            severity="INFO" if validated > 0 else "WARN",
            metric_value=float(validated),
            metric_unit="count",
            evidence={
                "status": validation.get("status"),
                "min_prior_outcomes": validation.get("min_prior_outcomes"),
                "eligible_outcomes": validation.get("eligible_outcomes"),
                "validated_outcomes": validation.get("validated_outcomes"),
                "directional_hit_rate_pct": validation.get("directional_hit_rate_pct"),
            },
        ),
        _observation(
            run_id=run_id,
            source="outcome_mart",
            source_path=path,
            subject_type="outcome_mart",
            subject_id="source_coverage",
            check_name="outcome_source_coverage",
            status="PASS" if int(coverage.get("archive_outcomes") or 0) + int(coverage.get("execution_ledger_outcomes") or 0) > 0 else "WARN",
            severity="INFO" if int(coverage.get("archive_outcomes") or 0) + int(coverage.get("execution_ledger_outcomes") or 0) > 0 else "WARN",
            metric_value=float(int(coverage.get("archive_outcomes") or 0) + int(coverage.get("execution_ledger_outcomes") or 0)),
            metric_unit="count",
            evidence=coverage,
        ),
    ]
    return observations


def _post_trade_learning_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    status = str(payload.get("status") or "UNKNOWN")
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    profile_updates = (
        payload.get("profile_update_candidates")
        if isinstance(payload.get("profile_update_candidates"), list)
        else []
    )
    return [
        _observation(
            run_id=run_id,
            source="post_trade_learning",
            source_path=path,
            subject_type="post_trade_learning",
            subject_id=status,
            check_name="learning_review_status",
            status="BLOCK" if status == "BLOCKED" or blockers else ("PASS" if status == "READY_FOR_REVIEW" else "WARN"),
            severity="BLOCK" if status == "BLOCKED" or blockers else ("INFO" if status == "READY_FOR_REVIEW" else "WARN"),
            metric_value=float(len(candidates)),
            metric_unit="count",
            evidence={"status": status, "blockers": blockers[:10], "profile_update_candidates": len(profile_updates)},
        ),
        _observation(
            run_id=run_id,
            source="post_trade_learning",
            source_path=path,
            subject_type="post_trade_learning",
            subject_id="profile_update_candidates",
            check_name="profile_update_requires_review",
            status="WARN" if profile_updates else "PASS",
            severity="WARN" if profile_updates else "INFO",
            metric_value=float(len(profile_updates)),
            metric_unit="count",
            evidence={"candidate_refs": [item.get("source_ref") for item in profile_updates if isinstance(item, dict)]},
        ),
    ]


def _ai_attack_advice_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    lanes = payload.get("lanes") if isinstance(payload.get("lanes"), list) else []
    recommended_ids = {
        str(item)
        for item in (payload.get("recommended_now_lane_ids") if isinstance(payload.get("recommended_now_lane_ids"), list) else [])
        if str(item).strip()
    }
    recommended_lanes = [
        item
        for item in lanes
        if isinstance(item, dict) and str(item.get("lane_id") or "") in recommended_ids
    ]
    influenced = []
    unvalidated = []
    total_learning_score_delta = 0.0
    for lane in recommended_lanes:
        influences = lane.get("learning_influences") if isinstance(lane.get("learning_influences"), list) else []
        influences = [str(item) for item in influences if str(item).strip()]
        score_delta = _maybe_float(lane.get("learning_score_delta")) or 0.0
        details = (
            lane.get("learning_influence_details")
            if isinstance(lane.get("learning_influence_details"), list)
            else []
        )
        if influences:
            total_learning_score_delta += score_delta
            influenced.append(
                {
                    "lane_id": lane.get("lane_id"),
                    "learning_influences": influences,
                    "learning_score_delta": score_delta,
                    "details": details[:8],
                }
            )
        if any("unvalidated" in item for item in influences):
            unvalidated.append(
                {
                    "lane_id": lane.get("lane_id"),
                    "learning_influences": influences,
                    "learning_score_delta": score_delta,
                }
            )
    return [
        _observation(
            run_id=run_id,
            source="ai_attack_advice",
            source_path=path,
            subject_type="ai_attack_advice",
            subject_id="read_only",
            check_name="read_only_learning",
            status="BLOCK" if payload.get("live_permission") is True or payload.get("read_only") is False else "PASS",
            severity="BLOCK" if payload.get("live_permission") is True or payload.get("read_only") is False else "INFO",
            evidence={"read_only": payload.get("read_only"), "live_permission": payload.get("live_permission")},
        ),
        _observation(
            run_id=run_id,
            source="ai_attack_advice",
            source_path=path,
            subject_type="ai_attack_advice",
            subject_id=str(payload.get("status") or "UNKNOWN"),
            check_name="recommended_learning_influence",
            status="WARN" if influenced else "PASS",
            severity="WARN" if influenced else "INFO",
            metric_value=total_learning_score_delta,
            metric_unit="score_delta",
            evidence={
                "recommended_now_lane_ids": sorted(recommended_ids),
                "learning_influenced_lane_count": len(influenced),
                "learning_influenced_lanes": influenced[:20],
                "unvalidated_learning_influenced_lanes": unvalidated[:20],
            },
        ),
    ]


def _learning_audit_observations(run_id: str, path: Path, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    status = str(payload.get("status") or "UNKNOWN")
    influence = payload.get("learning_influence") if isinstance(payload.get("learning_influence"), dict) else {}
    effect = payload.get("effect_metrics") if isinstance(payload.get("effect_metrics"), dict) else {}
    influence_delta = _maybe_float(influence.get("total_learning_score_delta")) or 0.0
    closed_trades = int(_maybe_float(effect.get("closed_trades")) or 0)
    min_effect_sample = int(_maybe_float(payload.get("min_effect_sample")) or 30)
    if status == "LEARNING_AUDIT_BLOCKED":
        audit_status = "BLOCK"
        severity = "BLOCK"
    elif status == "LEARNING_AUDIT_WARN":
        audit_status = "WARN"
        severity = "WARN"
    else:
        audit_status = "PASS"
        severity = "INFO"
    return [
        _observation(
            run_id=run_id,
            source="learning_audit",
            source_path=path,
            subject_type="learning_audit",
            subject_id=status,
            check_name="learning_audit_status",
            status=audit_status,
            severity=severity,
            metric_value=float(len(payload.get("blockers") or [])),
            metric_unit="blockers",
            evidence={
                "warnings": payload.get("warnings", [])[:20] if isinstance(payload.get("warnings"), list) else [],
                "blockers": payload.get("blockers", [])[:20] if isinstance(payload.get("blockers"), list) else [],
            },
        ),
        _observation(
            run_id=run_id,
            source="learning_audit",
            source_path=path,
            subject_type="learning_audit",
            subject_id="learning_influence",
            check_name="learning_influence_score_delta",
            status="WARN" if influence_delta else "PASS",
            severity="WARN" if influence_delta else "INFO",
            metric_value=influence_delta,
            metric_unit="score_delta",
            evidence={"influenced_lanes": influence.get("influenced_lanes"), "lanes": influence.get("lanes", [])[:20] if isinstance(influence.get("lanes"), list) else []},
        ),
        _observation(
            run_id=run_id,
            source="learning_audit",
            source_path=path,
            subject_type="learning_audit",
            subject_id="effect_window",
            check_name="learning_effect_window",
            status="WARN" if closed_trades < min_effect_sample else "PASS",
            severity="WARN" if closed_trades < min_effect_sample else "INFO",
            metric_value=float(closed_trades),
            metric_unit="closed_trades",
            evidence=effect,
        ),
    ]


def _execution_ledger_sync_observations(
    run_id: str,
    db_path: Path,
    broker_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    broker_last = None
    if broker_snapshot and isinstance(broker_snapshot.get("account"), dict):
        broker_last = broker_snapshot["account"].get("last_transaction_id")
    ledger_last = None
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT value FROM sync_state WHERE key='last_oanda_transaction_id'"
                ).fetchone()
                ledger_last = row[0] if row else None
        except sqlite3.Error:
            ledger_last = None
    status = "PASS"
    severity = "INFO"
    if broker_last and (not ledger_last or _int_or_zero(ledger_last) < _int_or_zero(broker_last)):
        status = "BLOCK"
        severity = "BLOCK"
    elif not ledger_last:
        status = "MISSING"
        severity = "WARN"
    return [
        _observation(
            run_id=run_id,
            source="execution_ledger",
            source_path=db_path,
            subject_type="execution_ledger",
            subject_id="last_oanda_transaction_id",
            check_name="ledger_synced_to_broker",
            status=status,
            severity=severity,
            evidence={"broker_last_transaction_id": broker_last, "ledger_last_transaction_id": ledger_last},
        )
    ]


def _effect_metrics(db_path: Path, *, window_hours: float, now: datetime) -> dict[str, Any]:
    rows: list[sqlite3.Row] = []
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = list(
                    conn.execute(
                        """
                        SELECT ts_utc, event_type, pair, side, exit_reason, realized_pl_jpy
                        FROM execution_events
                        WHERE event_type = 'TRADE_CLOSED'
                        """
                    )
                )
        except sqlite3.Error:
            rows = []
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    pls: list[float] = []
    for row in rows:
        ts = _parse_time(row["ts_utc"])
        if ts is None or ts < cutoff:
            continue
        value = _maybe_float(row["realized_pl_jpy"])
        if value is not None:
            pls.append(value)
    gross_profit = sum(value for value in pls if value > 0)
    gross_loss_abs = abs(sum(value for value in pls if value < 0))
    count = len(pls)
    wins = sum(1 for value in pls if value > 0)
    net = sum(pls)
    return {
        "closed_trades": count,
        "net_jpy": net,
        "gross_profit_jpy": gross_profit,
        "gross_loss_jpy": gross_loss_abs,
        "profit_factor": (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else None,
        "win_rate": (wins / count) if count else None,
        "expectancy_jpy": (net / count) if count else None,
    }


def _measurements_from_effect(run_id: str, *, window_hours: float, effect: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sample_size = int(effect.get("closed_trades") or 0)
    for metric_name, unit in (
        ("closed_trades", "count"),
        ("net_jpy", "JPY"),
        ("gross_profit_jpy", "JPY"),
        ("gross_loss_jpy", "JPY"),
        ("profit_factor", "ratio"),
        ("win_rate", "ratio"),
        ("expectancy_jpy", "JPY"),
    ):
        value = _maybe_float(effect.get(metric_name))
        out.append(
            {
                "measurement_uid": f"effect:{run_id}:all:{metric_name}",
                "ts_utc": run_id,
                "window_hours": float(window_hours),
                "segment": "all",
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                "sample_size": sample_size,
                "evidence_json": _json({"insufficient_sample_lt_30": sample_size < 30}),
                "inserted_at_utc": run_id,
            }
        )
    return out


def _insert_observation(conn: sqlite3.Connection, item: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO verification_observations(
            observation_uid, ts_utc, source, source_path, subject_type, subject_id, check_name,
            status, severity, metric_value, metric_unit, evidence_json, inserted_at_utc
        )
        VALUES (
            :observation_uid, :ts_utc, :source, :source_path, :subject_type, :subject_id, :check_name,
            :status, :severity, :metric_value, :metric_unit, :evidence_json, :inserted_at_utc
        )
        """,
        item,
    )
    return int(cur.rowcount > 0)


def _insert_measurement(conn: sqlite3.Connection, item: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO effect_measurements(
            measurement_uid, ts_utc, window_hours, segment, metric_name, metric_value,
            metric_unit, sample_size, evidence_json, inserted_at_utc
        )
        VALUES (
            :measurement_uid, :ts_utc, :window_hours, :segment, :metric_name, :metric_value,
            :metric_unit, :sample_size, :evidence_json, :inserted_at_utc
        )
        """,
        item,
    )
    return int(cur.rowcount > 0)


def _observation_packet(item: dict[str, Any]) -> dict[str, Any]:
    evidence = _parse_evidence_json(item.get("evidence_json"))
    packet = {
        "evidence_ref": _verification_ref(item),
        "source": item.get("source"),
        "source_path": item.get("source_path"),
        "subject_type": item.get("subject_type"),
        "subject_id": item.get("subject_id"),
        "check_name": item.get("check_name"),
        "status": item.get("status"),
        "severity": item.get("severity"),
        "metric_value": item.get("metric_value"),
        "metric_unit": item.get("metric_unit"),
    }
    if evidence:
        packet["evidence"] = evidence
    return packet


def _measurement_packet(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_ref": f"verification:effect:{_ref_part(item.get('segment'))}:{_ref_part(item.get('metric_name'))}",
        "segment": item.get("segment"),
        "metric_name": item.get("metric_name"),
        "metric_value": item.get("metric_value"),
        "metric_unit": item.get("metric_unit"),
        "sample_size": item.get("sample_size"),
        "window_hours": item.get("window_hours"),
        "evidence": _parse_evidence_json(item.get("evidence_json")),
    }


def _verification_ref(item: dict[str, Any]) -> str:
    parts = [
        "verification",
        _ref_part(item.get("source")),
        _ref_part(item.get("check_name")),
    ]
    subject = _ref_part(item.get("subject_id"))
    if subject:
        parts.append(subject)
    return ":".join(parts)


def _ref_part(value: Any) -> str:
    text = str(value or "").strip().replace(" ", "_").replace("/", "_")
    return text[:160]


def _parse_evidence_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # OANDA timestamps may carry nanoseconds; verification only needs
        # microsecond precision for effect-window inclusion.
        match = re.match(r"^(.*\.\d{6})\d+([+-]\d{2}:\d{2})$", text)
        if not match:
            return None
        try:
            parsed = datetime.fromisoformat(match.group(1) + match.group(2))
        except ValueError:
            return None
    return _to_utc(parsed)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _maybe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _int_or_zero(value: Any) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
