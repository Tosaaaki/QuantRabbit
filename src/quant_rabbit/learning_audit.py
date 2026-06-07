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
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_LEARNING_AUDIT_REPORT,
    DEFAULT_OUTCOME_MART,
    DEFAULT_POST_TRADE_LEARNING,
)


DEFAULT_MIN_EFFECT_SAMPLE = 30
DEFAULT_WINDOW_HOURS = 168.0
# Exit-reason diagnostics should wait for repeated occurrences but must still
# catch a bad close mechanism before the full learning sample floor is reached.
# Three observations is an audit repetition floor, not a trading threshold.
MIN_EXIT_REASON_DIAGNOSTIC_SAMPLE = 3

_INFLUENCE_LIMITS = {
    "ai_backtest_certified_positive_edge": 25.0,
    "ai_backtest_research_positive_edge": 8.0,
    "ai_backtest_negative_edge": -25.0,
    "outcome_mart_walk_forward_positive_edge": 15.0,
    "outcome_mart_unvalidated_positive_edge": 4.0,
    "outcome_mart_negative_edge": -15.0,
}


@dataclass(frozen=True)
class LearningAuditSummary:
    output_path: Path
    report_path: Path
    db_path: Path
    status: str
    checks: int
    blockers: int
    warnings: int
    influenced_lanes: int
    total_learning_score_delta: float
    closed_trades: int
    net_jpy: float
    profit_factor: float | None
    expectancy_jpy: float | None


class LearningAuditor:
    """Audit whether learning evidence is safe to influence live-ready ranking."""

    def __init__(
        self,
        *,
        db_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        output_path: Path = DEFAULT_LEARNING_AUDIT,
        report_path: Path = DEFAULT_LEARNING_AUDIT_REPORT,
    ) -> None:
        self.db_path = db_path
        self.output_path = output_path
        self.report_path = report_path

    def run(
        self,
        *,
        ai_backtest_path: Path = DEFAULT_AI_TEST_BOT_BACKTEST,
        outcome_mart_path: Path = DEFAULT_OUTCOME_MART,
        post_trade_learning_path: Path = DEFAULT_POST_TRADE_LEARNING,
        ai_attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        window_hours: float = DEFAULT_WINDOW_HOURS,
        min_effect_sample: int = DEFAULT_MIN_EFFECT_SAMPLE,
        now: datetime | None = None,
    ) -> LearningAuditSummary:
        clock = _to_utc(now or datetime.now(timezone.utc))
        run_id = clock.isoformat()
        self._init_db()

        artifacts = {
            "ai_backtest": _read_json(ai_backtest_path),
            "outcome_mart": _read_json(outcome_mart_path),
            "post_trade_learning": _read_json(post_trade_learning_path),
            "ai_attack_advice": _read_json(ai_attack_advice_path),
        }
        paths = {
            "ai_backtest": ai_backtest_path,
            "outcome_mart": outcome_mart_path,
            "post_trade_learning": post_trade_learning_path,
            "ai_attack_advice": ai_attack_advice_path,
        }
        checks: list[dict[str, Any]] = []
        for source, loaded in artifacts.items():
            checks.append(
                _check(
                    run_id=run_id,
                    source=source,
                    check_name="artifact_readable",
                    status="PASS" if loaded.payload is not None else "BLOCK",
                    severity="INFO" if loaded.payload is not None else "BLOCK",
                    evidence={"path": str(paths[source]), "error": loaded.error},
                )
            )

        effect = _effect_metrics(self.db_path, window_hours=window_hours, now=clock)
        ai_backtest = artifacts["ai_backtest"].payload or {}
        outcome_mart = artifacts["outcome_mart"].payload or {}
        post_trade_learning = artifacts["post_trade_learning"].payload or {}
        ai_attack_advice = artifacts["ai_attack_advice"].payload or {}

        checks.extend(_ai_backtest_checks(run_id, ai_backtest, min_effect_sample=min_effect_sample))
        checks.extend(_outcome_mart_checks(run_id, outcome_mart, min_effect_sample=min_effect_sample))
        checks.extend(_post_trade_learning_checks(run_id, post_trade_learning))
        influence_result = _advice_influence_checks(
            run_id,
            ai_attack_advice,
            ai_backtest=ai_backtest,
            outcome_mart=outcome_mart,
            effect=effect,
            min_effect_sample=min_effect_sample,
        )
        checks.extend(influence_result["checks"])
        checks.extend(_effect_checks(run_id, effect, influence_active=bool(influence_result["influenced_lanes"]), min_effect_sample=min_effect_sample))
        checks.extend(_exit_reason_checks(run_id, effect))

        blockers = [item for item in checks if item["status"] == "BLOCK" or item["severity"] == "BLOCK"]
        warnings = [item for item in checks if item["status"] == "WARN" or item["severity"] == "WARN"]
        status = "LEARNING_AUDIT_BLOCKED" if blockers else ("LEARNING_AUDIT_WARN" if warnings else "LEARNING_AUDIT_PASS")
        payload = {
            "generated_at_utc": run_id,
            "status": status,
            "artifact_paths": {key: str(path) for key, path in paths.items()},
            "window_hours": window_hours,
            "min_effect_sample": min_effect_sample,
            "checks": checks,
            "blockers": [item["message"] for item in blockers],
            "warnings": [item["message"] for item in warnings],
            "learning_influence": {
                "influenced_lanes": influence_result["influenced_lanes"],
                "total_learning_score_delta": influence_result["total_learning_score_delta"],
                "lanes": influence_result["lanes"],
            },
            "effect_metrics": effect,
            "contract": {
                "learning_may_rank_live_ready_lanes": True,
                "learning_cannot_override_hard_gates": True,
                "hard_gates": ["RiskEngine", "LiveOrderGateway", "entry_thesis_blocker", "TP_rebalance_blocker"],
            },
        }
        self._write_output(payload)
        self._write_report(payload)
        self._insert_run(run_id, status, payload, effect, influence_result)
        summary = LearningAuditSummary(
            output_path=self.output_path,
            report_path=self.report_path,
            db_path=self.db_path,
            status=status,
            checks=len(checks),
            blockers=len(blockers),
            warnings=len(warnings),
            influenced_lanes=int(influence_result["influenced_lanes"]),
            total_learning_score_delta=float(influence_result["total_learning_score_delta"]),
            closed_trades=int(effect["closed_trades"]),
            net_jpy=float(effect["net_jpy"]),
            profit_factor=_maybe_float(effect.get("profit_factor")),
            expectancy_jpy=_maybe_float(effect.get("expectancy_jpy")),
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
                CREATE TABLE IF NOT EXISTS learning_audit_runs (
                    run_uid TEXT PRIMARY KEY,
                    ts_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    report_path TEXT NOT NULL,
                    window_hours REAL NOT NULL,
                    checks INTEGER NOT NULL,
                    blockers INTEGER NOT NULL,
                    warnings INTEGER NOT NULL,
                    influenced_lanes INTEGER NOT NULL,
                    total_learning_score_delta REAL NOT NULL,
                    closed_trades INTEGER NOT NULL,
                    net_jpy REAL NOT NULL,
                    profit_factor REAL,
                    expectancy_jpy REAL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_learning_audit_runs_ts
                    ON learning_audit_runs(ts_utc);
                CREATE INDEX IF NOT EXISTS idx_learning_audit_runs_status
                    ON learning_audit_runs(status);

                CREATE TABLE IF NOT EXISTS learning_audit_checks (
                    check_uid TEXT PRIMARY KEY,
                    run_uid TEXT NOT NULL,
                    ts_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    check_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_value REAL,
                    metric_unit TEXT,
                    message TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_learning_audit_checks_run
                    ON learning_audit_checks(run_uid);
                CREATE INDEX IF NOT EXISTS idx_learning_audit_checks_status
                    ON learning_audit_checks(status, severity);
                """
            )

    def _insert_run(
        self,
        run_id: str,
        status: str,
        payload: dict[str, Any],
        effect: dict[str, Any],
        influence: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO learning_audit_runs(
                    run_uid, ts_utc, status, output_path, report_path, window_hours,
                    checks, blockers, warnings, influenced_lanes, total_learning_score_delta,
                    closed_trades, net_jpy, profit_factor, expectancy_jpy, inserted_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"learning_audit:{run_id}",
                    run_id,
                    status,
                    str(self.output_path),
                    str(self.report_path),
                    float(payload["window_hours"]),
                    len(payload["checks"]),
                    len(payload["blockers"]),
                    len(payload["warnings"]),
                    int(influence["influenced_lanes"]),
                    float(influence["total_learning_score_delta"]),
                    int(effect["closed_trades"]),
                    float(effect["net_jpy"]),
                    _maybe_float(effect.get("profit_factor")),
                    _maybe_float(effect.get("expectancy_jpy")),
                    run_id,
                ),
            )
            for idx, item in enumerate(payload["checks"]):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO learning_audit_checks(
                        check_uid, run_uid, ts_utc, source, check_name, status, severity,
                        metric_value, metric_unit, message, evidence_json, inserted_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"learning_audit:{run_id}:{idx}:{item['source']}:{item['check_name']}:{item.get('subject_id') or ''}",
                        f"learning_audit:{run_id}",
                        run_id,
                        item["source"],
                        item["check_name"],
                        item["status"],
                        item["severity"],
                        item.get("metric_value"),
                        item.get("metric_unit"),
                        item["message"],
                        _json(item.get("evidence") or {}),
                        run_id,
                    ),
                )

    def _write_output(self, payload: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, payload: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        influence = payload["learning_influence"]
        effect = payload["effect_metrics"]
        lines = [
            "# Learning Audit Report",
            "",
            f"- Generated at UTC: `{payload['generated_at_utc']}`",
            f"- Status: `{payload['status']}`",
            f"- Checks: `{len(payload['checks'])}`",
            f"- Blockers: `{len(payload['blockers'])}`",
            f"- Warnings: `{len(payload['warnings'])}`",
            f"- Influenced recommended lanes: `{influence['influenced_lanes']}`",
            f"- Total learning score delta: `{influence['total_learning_score_delta']:.1f}`",
            "",
            "## Effect Window",
            "",
            f"- Window hours: `{payload['window_hours']}`",
            f"- Closed trades: `{effect['closed_trades']}`",
            f"- Net JPY: `{effect['net_jpy']:.1f}`",
            f"- Profit factor: `{_format_optional(effect.get('profit_factor'))}`",
            f"- Expectancy JPY: `{_format_optional(effect.get('expectancy_jpy'))}`",
            "",
            "## Exit Reasons",
            "",
        ]
        exit_reasons = effect.get("exit_reason_metrics") if isinstance(effect.get("exit_reason_metrics"), dict) else {}
        if exit_reasons:
            for reason, metrics in sorted(exit_reasons.items(), key=lambda item: float(item[1].get("net_jpy") or 0.0)):
                lines.append(
                    f"- `{reason}` closed=`{int(metrics.get('closed_trades') or 0)}` "
                    f"net=`{float(metrics.get('net_jpy') or 0.0):.1f}` "
                    f"pf=`{_format_optional(metrics.get('profit_factor'))}` "
                    f"expectancy=`{_format_optional(metrics.get('expectancy_jpy'))}`"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Blockers",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in payload["blockers"]) if payload["blockers"] else lines.append("- none")
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in payload["warnings"]) if payload["warnings"] else lines.append("- none")
        lines.extend(["", "## Learning Influence", ""])
        if influence["lanes"]:
            for lane in influence["lanes"][:20]:
                lines.append(
                    f"- `{lane['lane_id']}` delta=`{lane['learning_score_delta']:.1f}` "
                    f"influences=`{', '.join(lane['learning_influences'])}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "## Contract", ""])
        lines.extend(
            [
                "- Learning may rank already-live-ready lanes only.",
                "- Learning cannot override RiskEngine, gateway, entry-thesis, or TP blockers.",
                "- Research-stage positive edges use reduced weight and remain WARN-level audit evidence.",
                "- Every recommended lane influenced by learning must expose `learning_influence_details`.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class _LoadedJson:
    payload: dict[str, Any] | None
    error: str | None = None


def _read_json(path: Path) -> _LoadedJson:
    if not path.exists():
        return _LoadedJson(None, "missing")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return _LoadedJson(None, str(exc))
    return _LoadedJson(payload if isinstance(payload, dict) else None, None if isinstance(payload, dict) else "json root is not an object")


def _check(
    *,
    run_id: str,
    source: str,
    check_name: str,
    status: str,
    severity: str,
    message: str | None = None,
    subject_id: str | None = None,
    metric_value: float | None = None,
    metric_unit: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "source": source,
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "message": message or f"{source}.{check_name} {status}",
        "subject_id": subject_id,
        "metric_value": metric_value,
        "metric_unit": metric_unit,
        "evidence": evidence or {},
    }


def _ai_backtest_checks(run_id: str, payload: dict[str, Any], *, min_effect_sample: int) -> list[dict[str, Any]]:
    if not payload:
        return []
    status = str(payload.get("status") or "UNKNOWN")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    selected = int(_maybe_float(summary.get("selected_trades")) or 0)
    net = _maybe_float(summary.get("total_managed_net_jpy")) or 0.0
    pf = _maybe_float(summary.get("profit_factor")) or 0.0
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    checks = [
        _check(
            run_id=run_id,
            source="ai_backtest",
            check_name="read_only_learning",
            status="BLOCK" if payload.get("live_permission") is True else "PASS",
            severity="BLOCK" if payload.get("live_permission") is True else "INFO",
            message="ai backtest must not grant live permission",
            evidence={"live_permission": payload.get("live_permission")},
        )
    ]
    if status == "TARGET_COVERAGE_CERTIFIED" and not blockers:
        result = ("PASS", "INFO", "AI backtest is certified for full advisory weighting")
    elif status == "RESEARCH_PROFITABLE_NOT_CERTIFIED" and selected >= min_effect_sample and net > 0 and pf > 1.0:
        result = ("WARN", "WARN", "AI backtest is profitable research only; reduced weighting required")
    else:
        result = ("BLOCK", "BLOCK", "AI backtest is not safe to use for positive live ranking")
    checks.append(
        _check(
            run_id=run_id,
            source="ai_backtest",
            check_name="positive_edge_permission",
            status=result[0],
            severity=result[1],
            message=result[2],
            metric_value=float(selected),
            metric_unit="selected_trades",
            evidence={"status": status, "selected_trades": selected, "total_managed_net_jpy": net, "profit_factor": pf, "blockers": blockers[:10]},
        )
    )
    return checks


def _outcome_mart_checks(run_id: str, payload: dict[str, Any], *, min_effect_sample: int) -> list[dict[str, Any]]:
    if not payload:
        return []
    validation = payload.get("condition_validation") if isinstance(payload.get("condition_validation"), dict) else {}
    validated = int(validation.get("validated_outcomes") or 0)
    read_only_ok = payload.get("read_only") is True and payload.get("live_permission") is not True
    return [
        _check(
            run_id=run_id,
            source="outcome_mart",
            check_name="read_only_learning",
            status="PASS" if read_only_ok else "BLOCK",
            severity="INFO" if read_only_ok else "BLOCK",
            message="outcome mart must remain read-only",
            evidence={"read_only": payload.get("read_only"), "live_permission": payload.get("live_permission")},
        ),
        _check(
            run_id=run_id,
            source="outcome_mart",
            check_name="condition_walk_forward",
            status="PASS" if validated >= min_effect_sample else "WARN",
            severity="INFO" if validated >= min_effect_sample else "WARN",
            message="outcome mart walk-forward validation sample",
            metric_value=float(validated),
            metric_unit="validated_outcomes",
            evidence=validation,
        ),
    ]


def _post_trade_learning_checks(run_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not payload:
        return []
    status = str(payload.get("status") or "UNKNOWN")
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    profile_updates = payload.get("profile_update_candidates") if isinstance(payload.get("profile_update_candidates"), list) else []
    return [
        _check(
            run_id=run_id,
            source="post_trade_learning",
            check_name="learning_review_status",
            status="BLOCK" if status == "BLOCKED" or blockers else "PASS",
            severity="BLOCK" if status == "BLOCKED" or blockers else "INFO",
            message="post-trade learning blockers must be resolved before learning expansion",
            metric_value=float(len(blockers)),
            metric_unit="blockers",
            evidence={"status": status, "blockers": blockers[:10]},
        ),
        _check(
            run_id=run_id,
            source="post_trade_learning",
            check_name="profile_update_requires_review",
            status="WARN" if profile_updates else "PASS",
            severity="WARN" if profile_updates else "INFO",
            message="profile update candidates require human/code review before mutation",
            metric_value=float(len(profile_updates)),
            metric_unit="candidates",
            evidence={"candidate_refs": [item.get("source_ref") for item in profile_updates if isinstance(item, dict)]},
        ),
    ]


def _advice_influence_checks(
    run_id: str,
    payload: dict[str, Any],
    *,
    ai_backtest: dict[str, Any],
    outcome_mart: dict[str, Any],
    effect: dict[str, Any],
    min_effect_sample: int,
) -> dict[str, Any]:
    if not payload:
        return {"checks": [], "influenced_lanes": 0, "total_learning_score_delta": 0.0, "lanes": []}
    lanes = payload.get("lanes") if isinstance(payload.get("lanes"), list) else []
    recommended_ids = {str(item) for item in payload.get("recommended_now_lane_ids", []) or [] if str(item).strip()}
    recommended = [lane for lane in lanes if isinstance(lane, dict) and str(lane.get("lane_id") or "") in recommended_ids]
    influenced_lanes: list[dict[str, Any]] = []
    checks = [
        _check(
            run_id=run_id,
            source="ai_attack_advice",
            check_name="read_only_learning",
            status="BLOCK" if payload.get("live_permission") is True or payload.get("read_only") is False else "PASS",
            severity="BLOCK" if payload.get("live_permission") is True or payload.get("read_only") is False else "INFO",
            message="attack advice must rank only and never grant live permission",
            evidence={"read_only": payload.get("read_only"), "live_permission": payload.get("live_permission")},
        )
    ]
    total_delta = 0.0
    for lane in recommended:
        influences = [str(item) for item in (lane.get("learning_influences") or []) if str(item).strip()]
        if not influences:
            continue
        details = lane.get("learning_influence_details") if isinstance(lane.get("learning_influence_details"), list) else []
        lane_delta = _maybe_float(lane.get("learning_score_delta")) or 0.0
        total_delta += lane_delta
        lane_id = str(lane.get("lane_id") or "")
        influenced_lanes.append(
            {
                "lane_id": lane_id,
                "learning_influences": influences,
                "learning_score_delta": lane_delta,
                "details": details,
            }
        )
        checks.extend(
            _lane_influence_checks(
                run_id,
                lane_id,
                influences,
                details,
                lane_delta,
                ai_backtest=ai_backtest,
                outcome_mart=outcome_mart,
                min_effect_sample=min_effect_sample,
            )
        )
        score = abs(_maybe_float(lane.get("score")) or 0.0)
        if score > 0 and abs(lane_delta) > max(25.0, score * 0.5):
            checks.append(
                _check(
                    run_id=run_id,
                    source="ai_attack_advice",
                    subject_id=lane_id,
                    check_name="learning_dominance",
                    status="WARN",
                    severity="WARN",
                    message="learning score delta dominates the ranked lane score",
                    metric_value=lane_delta,
                    metric_unit="score_delta",
                    evidence={"score": lane.get("score"), "learning_score_delta": lane_delta},
                )
            )
    if influenced_lanes:
        checks.append(
            _check(
                run_id=run_id,
                source="ai_attack_advice",
                check_name="recommended_learning_influence",
                status="WARN",
                severity="WARN",
                message="recommended lanes are influenced by learning; audit review required",
                metric_value=total_delta,
                metric_unit="score_delta",
                evidence={"lanes": influenced_lanes[:20]},
            )
        )
    else:
        checks.append(
            _check(
                run_id=run_id,
                source="ai_attack_advice",
                check_name="recommended_learning_influence",
                status="PASS",
                severity="INFO",
                message="no recommended lane is currently influenced by learning",
                metric_value=0.0,
                metric_unit="score_delta",
            )
        )
    if influenced_lanes and int(effect["closed_trades"]) >= min_effect_sample and (float(effect["net_jpy"]) < 0 or (_maybe_float(effect.get("profit_factor")) or 0.0) < 1.0):
        checks.append(
            _check(
                run_id=run_id,
                source="effect_measurement",
                check_name="learning_influence_recent_outcome",
                status="BLOCK",
                severity="BLOCK",
                message="learning-influenced ranking is active while recent effect window is negative",
                evidence=effect,
            )
        )
    return {"checks": checks, "influenced_lanes": len(influenced_lanes), "total_learning_score_delta": round(total_delta, 4), "lanes": influenced_lanes}


def _lane_influence_checks(
    run_id: str,
    lane_id: str,
    influences: list[str],
    details: list[Any],
    lane_delta: float,
    *,
    ai_backtest: dict[str, Any],
    outcome_mart: dict[str, Any],
    min_effect_sample: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not details:
        checks.append(
            _check(
                run_id=run_id,
                source="ai_attack_advice",
                subject_id=lane_id,
                check_name="learning_influence_details_present",
                status="BLOCK",
                severity="BLOCK",
                message="learning-influenced lane lacks learning_influence_details",
                evidence={"learning_influences": influences},
            )
        )
        return checks
    detail_delta = 0.0
    for detail in details:
        if not isinstance(detail, dict):
            continue
        influence = str(detail.get("influence") or "")
        delta = _maybe_float(detail.get("score_delta")) or 0.0
        detail_delta += delta
        limit = _INFLUENCE_LIMITS.get(influence)
        if limit is None or (limit >= 0 and delta > limit) or (limit < 0 and delta < limit):
            checks.append(
                _check(
                    run_id=run_id,
                    source="ai_attack_advice",
                    subject_id=lane_id,
                    check_name="learning_influence_weight",
                    status="BLOCK",
                    severity="BLOCK",
                    message="learning influence has unknown or excessive score delta",
                    metric_value=delta,
                    metric_unit="score_delta",
                    evidence=detail,
                )
            )
        if influence == "ai_backtest_certified_positive_edge" and not _ai_backtest_certified(ai_backtest):
            checks.append(_blocked_influence(run_id, lane_id, "certified AI backtest influence without certified backtest", detail))
        if influence == "ai_backtest_research_positive_edge" and not _ai_backtest_research_allowed(ai_backtest, min_effect_sample=min_effect_sample):
            checks.append(_blocked_influence(run_id, lane_id, "research AI backtest influence without profitable audited research", detail))
        if influence == "outcome_mart_walk_forward_positive_edge" and not _outcome_detail_validated(detail):
            checks.append(_blocked_influence(run_id, lane_id, "outcome mart influence lacks positive walk-forward detail", detail))
        if influence == "outcome_mart_unvalidated_positive_edge":
            checks.append(
                _check(
                    run_id=run_id,
                    source="ai_attack_advice",
                    subject_id=lane_id,
                    check_name="unvalidated_positive_learning_influence",
                    status="WARN",
                    severity="WARN",
                    message="unvalidated positive outcome edge is exploratory and must stay low weight",
                    metric_value=delta,
                    metric_unit="score_delta",
                    evidence=detail,
                )
            )
    if abs(detail_delta - lane_delta) > 0.001:
        checks.append(
            _check(
                run_id=run_id,
                source="ai_attack_advice",
                subject_id=lane_id,
                check_name="learning_score_delta_reconciles",
                status="BLOCK",
                severity="BLOCK",
                message="lane learning_score_delta does not equal detail score deltas",
                metric_value=lane_delta,
                metric_unit="score_delta",
                evidence={"detail_delta": detail_delta, "lane_delta": lane_delta},
            )
        )
    else:
        checks.append(
            _check(
                run_id=run_id,
                source="ai_attack_advice",
                subject_id=lane_id,
                check_name="learning_score_delta_reconciles",
                status="PASS",
                severity="INFO",
                message="lane learning score reconciles with detail rows",
                metric_value=lane_delta,
                metric_unit="score_delta",
            )
        )
    return checks


def _effect_checks(run_id: str, effect: dict[str, Any], *, influence_active: bool, min_effect_sample: int) -> list[dict[str, Any]]:
    sample = int(effect["closed_trades"])
    status = "PASS"
    severity = "INFO"
    message = "effect sample is sufficient for learning audit"
    if sample < min_effect_sample:
        status = "WARN"
        severity = "WARN"
        message = "effect sample is below stability floor"
    elif influence_active and (float(effect["net_jpy"]) < 0 or (_maybe_float(effect.get("profit_factor")) or 0.0) < 1.0):
        status = "BLOCK"
        severity = "BLOCK"
        message = "learning influence is active while recent effect window is negative"
    return [
        _check(
            run_id=run_id,
            source="effect_measurement",
            check_name="recent_learning_effect_window",
            status=status,
            severity=severity,
            message=message,
            metric_value=float(sample),
            metric_unit="closed_trades",
            evidence=effect,
        )
    ]


def _exit_reason_checks(run_id: str, effect: dict[str, Any]) -> list[dict[str, Any]]:
    exit_reasons = effect.get("exit_reason_metrics") if isinstance(effect.get("exit_reason_metrics"), dict) else {}
    market_close = exit_reasons.get("MARKET_ORDER_TRADE_CLOSE") if isinstance(exit_reasons.get("MARKET_ORDER_TRADE_CLOSE"), dict) else None
    if not market_close:
        return []
    closed = int(market_close.get("closed_trades") or 0)
    net = _maybe_float(market_close.get("net_jpy")) or 0.0
    if closed < MIN_EXIT_REASON_DIAGNOSTIC_SAMPLE or net >= 0:
        return []
    return [
        _check(
            run_id=run_id,
            source="effect_measurement",
            check_name="market_order_trade_close_drag",
            status="WARN",
            severity="WARN",
            message="market-order trade closes are negative in the recent effect window; prefer TP/TP-rebalance/profit-side exits unless CLOSE Gate A/B is hard",
            metric_value=net,
            metric_unit="JPY",
            evidence=market_close,
        )
    ]


def _blocked_influence(run_id: str, lane_id: str, message: str, detail: dict[str, Any]) -> dict[str, Any]:
    return _check(
        run_id=run_id,
        source="ai_attack_advice",
        subject_id=lane_id,
        check_name="learning_influence_permission",
        status="BLOCK",
        severity="BLOCK",
        message=message,
        evidence=detail,
    )


def _ai_backtest_certified(payload: dict[str, Any]) -> bool:
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    return payload.get("live_permission") is not True and payload.get("status") == "TARGET_COVERAGE_CERTIFIED" and not blockers


def _ai_backtest_research_allowed(payload: dict[str, Any], *, min_effect_sample: int) -> bool:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return (
        payload.get("live_permission") is not True
        and payload.get("status") == "RESEARCH_PROFITABLE_NOT_CERTIFIED"
        and int(_maybe_float(summary.get("selected_trades")) or 0) >= min_effect_sample
        and (_maybe_float(summary.get("total_managed_net_jpy")) or 0.0) > 0
        and (_maybe_float(summary.get("profit_factor")) or 0.0) > 1.0
    )


def _outcome_detail_validated(detail: dict[str, Any]) -> bool:
    return int(_maybe_float(detail.get("outcomes")) or 0) >= 5 and (_maybe_float(detail.get("actual_net_jpy")) or 0.0) > 0


def _effect_metrics(db_path: Path, *, window_hours: float, now: datetime) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                columns = _table_columns(conn, "execution_events")
                exit_reason_select = ", exit_reason" if "exit_reason" in columns else ""
                rows = list(
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT ts_utc, event_type, realized_pl_jpy{exit_reason_select}
                        FROM execution_events
                        WHERE event_type = 'TRADE_CLOSED'
                        """.format(exit_reason_select=exit_reason_select)
                    )
                )
        except sqlite3.Error:
            rows = []
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    pls: list[float] = []
    pls_by_exit_reason: dict[str, list[float]] = {}
    for row in rows:
        ts = _parse_time(row["ts_utc"])
        if ts is None or ts < cutoff:
            continue
        value = _maybe_float(row["realized_pl_jpy"])
        if value is not None:
            pls.append(value)
            reason = str(row.get("exit_reason") or "UNKNOWN").strip() or "UNKNOWN"
            pls_by_exit_reason.setdefault(reason, []).append(value)
    gross_profit = sum(value for value in pls if value > 0)
    gross_loss = abs(sum(value for value in pls if value < 0))
    net = sum(pls)
    count = len(pls)
    wins = sum(1 for value in pls if value > 0)
    return {
        "closed_trades": count,
        "net_jpy": net,
        "gross_profit_jpy": gross_profit,
        "gross_loss_jpy": gross_loss,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "win_rate": (wins / count) if count else None,
        "expectancy_jpy": (net / count) if count else None,
        "exit_reason_metrics": _exit_reason_metrics(pls_by_exit_reason),
    }


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")}
    except sqlite3.Error:
        return set()


def _exit_reason_metrics(pls_by_exit_reason: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    for reason, pls in sorted(pls_by_exit_reason.items()):
        gross_profit = sum(value for value in pls if value > 0)
        gross_loss = abs(sum(value for value in pls if value < 0))
        net = sum(pls)
        count = len(pls)
        wins = sum(1 for value in pls if value > 0)
        metrics[reason] = {
            "closed_trades": count,
            "net_jpy": round(net, 4),
            "gross_profit_jpy": round(gross_profit, 4),
            "gross_loss_jpy": round(gross_loss, 4),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
            "win_rate": (wins / count) if count else None,
            "expectancy_jpy": (net / count) if count else None,
        }
    return metrics


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # OANDA ledger timestamps can carry nanoseconds; Python accepts only
        # microseconds. Truncate only for audit window math.
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


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _format_optional(value: Any) -> str:
    parsed = _maybe_float(value)
    return "n/a" if parsed is None else f"{parsed:.3f}"
