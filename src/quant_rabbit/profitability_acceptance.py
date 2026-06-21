from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.forecast_precision import (
    DEFAULT_BIDASK_REPLAY_RULES_PATH,
    projection_precision_edge_summary,
    projection_precision_gap_summary,
)
from quant_rabbit.paths import (
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PROFITABILITY_ACCEPTANCE,
    DEFAULT_PROFITABILITY_ACCEPTANCE_REPORT,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
)
from quant_rabbit.risk import (
    FORECAST_LIVE_PRECISION_MIN_SAMPLES,
    FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
)
from quant_rabbit.strategy.projection_ledger import compute_hit_rates


STATUS_PASSED = "PROFITABILITY_ACCEPTANCE_PASSED"
STATUS_ACTION_REQUIRED = "PROFITABILITY_ACCEPTANCE_ACTION_REQUIRED"
STATUS_BLOCKED = "PROFITABILITY_ACCEPTANCE_BLOCKED"


@dataclass(frozen=True)
class ProfitabilityAcceptanceSummary:
    status: str
    output_path: Path
    report_path: Path
    findings: tuple[dict[str, Any], ...]
    blockers: tuple[str, ...]
    metrics: dict[str, Any]


class ProfitabilityAcceptanceAuditor:
    """One red/green gate for the account-level profit invariants."""

    def __init__(
        self,
        *,
        output_path: Path = DEFAULT_PROFITABILITY_ACCEPTANCE,
        report_path: Path = DEFAULT_PROFITABILITY_ACCEPTANCE_REPORT,
    ) -> None:
        self.output_path = output_path
        self.report_path = report_path

    def run(
        self,
        *,
        order_intents_path: Path = DEFAULT_ORDER_INTENTS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        self_improvement_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
        capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        bidask_rules_path: Path = DEFAULT_BIDASK_REPLAY_RULES_PATH,
    ) -> ProfitabilityAcceptanceSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        findings: list[dict[str, Any]] = []

        target = _load_json(target_state_path)
        intents = _load_json(order_intents_path)
        self_improvement = _load_json(self_improvement_path)
        capture = _load_json(capture_economics_path)
        bidask_rules = _load_json(bidask_rules_path)

        order_metrics = _order_intent_metrics(intents)
        target_metrics = _target_metrics(target)
        self_metrics, self_findings = _self_improvement_findings(self_improvement)
        capture_metrics, capture_findings = _capture_economics_findings(capture)
        projection_metrics, projection_findings = _projection_precision_findings(projection_ledger_path)
        bidask_metrics, bidask_findings = _bidask_rule_findings(bidask_rules, bidask_rules_path)

        findings.extend(self_findings)
        findings.extend(capture_findings)
        findings.extend(projection_findings)
        findings.extend(bidask_findings)

        if target_metrics["target_open"] and order_metrics["live_ready_lanes"] <= 0:
            findings.append(
                _finding(
                    priority="P1",
                    code="NO_LIVE_READY_TARGET_COVERAGE",
                    message="daily target is open but there are no LIVE_READY lanes",
                    next_action=(
                        "Do not increase churn blindly. Promote only lanes whose blockers are named "
                        "and whose forecast, spread, TP, risk, and gateway evidence can clear together."
                    ),
                    evidence={
                        "remaining_target_jpy": target_metrics.get("remaining_target_jpy"),
                        "candidate_count": order_metrics.get("candidate_count"),
                        "live_ready_lanes": order_metrics.get("live_ready_lanes"),
                        "top_blockers": order_metrics.get("top_blockers"),
                    },
                )
            )

        p0_findings = [item for item in findings if item.get("priority") == "P0"]
        if p0_findings:
            status = STATUS_BLOCKED
        elif findings:
            status = STATUS_ACTION_REQUIRED
        else:
            status = STATUS_PASSED

        blockers = tuple(
            f"{item.get('code')}: {item.get('message')}"
            for item in findings
            if item.get("priority") == "P0"
        )
        metrics = {
            "generated_at_utc": generated_at,
            "order_intents": order_metrics,
            "target": target_metrics,
            "self_improvement": self_metrics,
            "capture_economics": capture_metrics,
            "projection_precision": projection_metrics,
            "bidask_replay_rules": bidask_metrics,
            "finding_counts": {
                "P0": len(p0_findings),
                "P1": sum(1 for item in findings if item.get("priority") == "P1"),
                "P2": sum(1 for item in findings if item.get("priority") == "P2"),
            },
        }
        payload = {
            "status": status,
            "generated_at_utc": generated_at,
            "findings": findings,
            "blockers": list(blockers),
            "metrics": metrics,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(_render_report(payload))
        return ProfitabilityAcceptanceSummary(
            status=status,
            output_path=self.output_path,
            report_path=self.report_path,
            findings=tuple(findings),
            blockers=blockers,
            metrics=metrics,
        )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _finding(
    *,
    priority: str,
    code: str,
    message: str,
    next_action: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "priority": priority,
        "code": code,
        "message": message,
        "next_action": next_action,
        "evidence": evidence,
    }


def _order_intent_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    live_ready = [item for item in results if str(item.get("status") or "") == "LIVE_READY"]
    blockers: dict[str, int] = {}
    for item in results:
        for issue in item.get("risk_issues") or item.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            code = str(issue.get("code") or "").strip()
            if code:
                blockers[code] = blockers.get(code, 0) + 1
        for code in item.get("live_blockers") or []:
            text = str(code or "").strip()
            if not text:
                continue
            key = text.split(":", 1)[0][:80]
            blockers[key] = blockers.get(key, 0) + 1
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "candidate_count": len(results),
        "live_ready_lanes": len(live_ready),
        "dry_run_blocked_lanes": sum(
            1 for item in results if str(item.get("status") or "") == "DRY_RUN_BLOCKED"
        ),
        "top_blockers": [
            {"code": code, "count": count}
            for code, count in sorted(blockers.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
    }


def _target_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    remaining = _optional_float(payload.get("remaining_target_jpy"))
    status = str(payload.get("status") or "")
    target_open = bool(remaining is not None and remaining > 0) and status not in {
        "TARGET_REACHED",
        "TARGET_DONE",
        "COMPLETE",
    }
    return {
        "status": status or None,
        "remaining_target_jpy": remaining,
        "target_open": target_open,
    }


def _self_improvement_findings(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings = [item for item in payload.get("findings") or [] if isinstance(item, dict)]
    p0_items = [item for item in findings if str(item.get("priority") or "").upper() == "P0"]
    metrics = {
        "status": payload.get("status"),
        "p0_findings": len(p0_items),
        "p1_findings": sum(1 for item in findings if str(item.get("priority") or "").upper() == "P1"),
        "p0_codes": [str(item.get("code") or "") for item in p0_items],
    }
    if not p0_items:
        return metrics, []
    return metrics, [
        _finding(
            priority="P0",
            code="SELF_IMPROVEMENT_P0_PRESENT",
            message=f"self-improvement audit still has {len(p0_items)} P0 finding(s)",
            next_action=(
                "Treat P0 audit findings as acceptance blockers. Repair or explicitly resolve them "
                "before increasing live entry frequency."
            ),
            evidence={
                "status": payload.get("status"),
                "p0_findings": [
                    {
                        "code": item.get("code"),
                        "message": item.get("message"),
                    }
                    for item in p0_items[:5]
                ],
            },
        )
    ]


def _capture_economics_findings(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    by_exit = payload.get("by_exit_reason") if isinstance(payload.get("by_exit_reason"), dict) else {}
    tp_exit = by_exit.get("TAKE_PROFIT_ORDER") if isinstance(by_exit.get("TAKE_PROFIT_ORDER"), dict) else {}
    market_close = (
        by_exit.get("MARKET_ORDER_TRADE_CLOSE")
        if isinstance(by_exit.get("MARKET_ORDER_TRADE_CLOSE"), dict)
        else {}
    )
    priorities = payload.get("segment_repair_priorities")
    priority_items = priorities.get("items") if isinstance(priorities, dict) else []
    priority_items = [item for item in priority_items or [] if isinstance(item, dict)]
    tp_market_close_leaks = [
        item
        for item in priority_items
        if str(item.get("priority_class") or "").upper()
        == "PRESERVE_TP_PROVEN_REPAIR_MARKET_CLOSE_LEAK"
    ]
    status = str(payload.get("status") or "").upper()
    metrics = {
        "status": status or None,
        "overall": {
            "trades": int(_optional_float(overall.get("trades")) or 0),
            "net_jpy": _optional_float(overall.get("net_jpy")),
            "profit_factor": _optional_float(overall.get("profit_factor")),
            "expectancy_jpy_per_trade": _optional_float(overall.get("expectancy_jpy_per_trade")),
            "win_rate": _optional_float(overall.get("win_rate")),
            "payoff_ratio": _optional_float(overall.get("payoff_ratio")),
        },
        "take_profit": {
            "trades": int(_optional_float(tp_exit.get("trades")) or 0),
            "expectancy_jpy_per_trade": _optional_float(tp_exit.get("expectancy_jpy_per_trade")),
            "net_jpy": _optional_float(tp_exit.get("net_jpy")),
        },
        "market_close": {
            "trades": int(_optional_float(market_close.get("trades")) or 0),
            "expectancy_jpy_per_trade": _optional_float(market_close.get("expectancy_jpy_per_trade")),
            "net_jpy": _optional_float(market_close.get("net_jpy")),
        },
        "tp_proven_market_close_leak_segments": len(tp_market_close_leaks),
    }
    findings: list[dict[str, Any]] = []
    if status == "NEGATIVE_EXPECTANCY":
        findings.append(
            _finding(
                priority="P0",
                code="NEGATIVE_EXPECTANCY_ACTIVE",
                message="capture economics is still NEGATIVE_EXPECTANCY",
                next_action=(
                    "Block acceptance until realized expectancy is non-negative or only exact "
                    "TP-proven HARVEST repair shapes are allowed through the entry gate."
                ),
                evidence=metrics["overall"],
            )
        )
    if tp_market_close_leaks:
        findings.append(
            _finding(
                priority="P0",
                code="MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                message=(
                    f"{len(tp_market_close_leaks)} TP-proven segment(s) are still net-damaged "
                    "by MARKET_ORDER_TRADE_CLOSE leakage"
                ),
                next_action=(
                    "Preserve the attached broker-TP shape, but do not scale exposure until the "
                    "loss-side market-close path is repaired or avoided."
                ),
                evidence={
                    "segments": [
                        {
                            "pair": item.get("pair"),
                            "side": item.get("side"),
                            "method": item.get("method"),
                            "take_profit_trades": item.get("take_profit_trades"),
                            "take_profit_expectancy_jpy": item.get("take_profit_expectancy_jpy"),
                            "market_close_net_jpy": item.get("market_close_net_jpy"),
                            "market_close_expectancy_jpy": item.get("market_close_expectancy_jpy"),
                        }
                        for item in tp_market_close_leaks[:5]
                    ]
                },
            )
        )
    return metrics, findings


def _projection_precision_findings(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.exists():
        return {
            "ledger_exists": False,
            "economic_precision_edges": 0,
            "economic_precision_gaps": 0,
        }, [
            _finding(
                priority="P1",
                code="PROJECTION_LEDGER_MISSING",
                message="projection ledger is missing, so 90% prediction evidence cannot be audited",
                next_action="Record and verify projection outcomes before treating any forecast as high precision.",
                evidence={"path": str(path)},
            )
        ]
    try:
        hit_rates = compute_hit_rates(path.parent)
    except Exception as exc:
        return {
            "ledger_exists": True,
            "economic_precision_edges": 0,
            "economic_precision_gaps": 0,
        }, [
            _finding(
                priority="P1",
                code="PROJECTION_PRECISION_UNREADABLE",
                message="projection precision could not be computed",
                next_action="Repair projection ledger parsing before using forecast precision as live support.",
                evidence={"path": str(path), "error": f"{type(exc).__name__}: {exc}"},
            )
        ]
    filtered = {
        signal: buckets
        for signal, buckets in (hit_rates or {}).items()
        if not str(signal or "").startswith("directional_forecast")
    }
    edges = projection_precision_edge_summary(
        filtered,
        min_wilson_lower=FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
        min_samples=FORECAST_LIVE_PRECISION_MIN_SAMPLES,
        limit=20,
    )
    gaps = projection_precision_gap_summary(
        filtered,
        min_wilson_lower=FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
        min_samples=FORECAST_LIVE_PRECISION_MIN_SAMPLES,
        limit=20,
    )
    metrics = {
        "ledger_exists": True,
        "economic_precision_edges": len(edges),
        "economic_precision_gaps": len(gaps),
        "top_edges": edges[:5],
        "top_gaps": gaps[:5],
    }
    findings: list[dict[str, Any]] = []
    if gaps:
        findings.append(
            _finding(
                priority="P1",
                code="PROJECTION_HEADLINE_PRECISION_ECONOMIC_GAP",
                message=(
                    f"{len(gaps)} projection bucket(s) clear headline precision but fail "
                    "economic precision after TIMEOUT/no-touch penalties"
                ),
                next_action=(
                    "Do not use these buckets for high-turn live permission. Tighten detector, "
                    "target, horizon, or pair/regime segmentation until economic Wilson also clears."
                ),
                evidence={"gaps": gaps[:5], "usable_edges": edges[:5]},
            )
        )
    if not edges:
        findings.append(
            _finding(
                priority="P1",
                code="NO_PROJECTION_ECONOMIC_PRECISION_EDGE",
                message="no projection bucket currently clears economic live precision",
                next_action="Mine more pair/direction/regime evidence before claiming 90% forecast support.",
                evidence={
                    "min_wilson_lower": FORECAST_LIVE_PRECISION_MIN_WILSON_LOWER,
                    "min_samples": FORECAST_LIVE_PRECISION_MIN_SAMPLES,
                },
            )
        )
    return metrics, findings


def _bidask_rule_findings(payload: dict[str, Any], path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contrarian_rules = [
        item for item in payload.get("contrarian_edge_rules") or [] if isinstance(item, dict)
    ]
    daily_stable = [
        item
        for item in contrarian_rules
        if str(item.get("daily_stability_status") or "").upper() == "DAILY_STABLE"
    ]
    rank_only = [item for item in contrarian_rules if item not in daily_stable]
    metrics = {
        "path": str(path),
        "contrarian_edge_rules": len(contrarian_rules),
        "daily_stable_contrarian_edge_rules": len(daily_stable),
        "rank_only_contrarian_edge_rules": len(rank_only),
        "rank_only_examples": [
            {
                "name": item.get("name"),
                "pair": item.get("pair"),
                "forecast_direction": item.get("forecast_direction") or item.get("faded_direction"),
                "direction": item.get("direction"),
                "samples": item.get("samples"),
                "active_days": item.get("active_days"),
                "positive_day_rate": item.get("positive_day_rate"),
                "daily_stability_status": item.get("daily_stability_status"),
                "optimized_profit_factor": item.get("optimized_profit_factor"),
            }
            for item in rank_only[:5]
        ],
    }
    if rank_only and not daily_stable:
        return metrics, [
            _finding(
                priority="P1",
                code="BIDASK_CONTRARIAN_EDGE_NOT_DAILY_STABLE",
                message=(
                    f"{len(rank_only)} S5 contrarian replay edge(s) exist, but none are daily-stable "
                    "enough for live-grade inversion"
                ),
                next_action=(
                    "Keep weak forecast inversion as rank-only until multi-day stability, positive-day "
                    "rate, and execution geometry clear the live thresholds."
                ),
                evidence=metrics,
            )
        ]
    return metrics, []


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _render_report(payload: dict[str, Any]) -> str:
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    lines = [
        "# Profitability Acceptance",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Findings: `{len(findings)}`",
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("No acceptance findings.")
    else:
        lines.extend(["| Priority | Code | Message |", "| --- | --- | --- |"])
        for item in findings:
            lines.append(
                f"| `{item.get('priority')}` | `{item.get('code')}` | "
                f"{str(item.get('message') or '').replace('|', '/')} |"
            )
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
