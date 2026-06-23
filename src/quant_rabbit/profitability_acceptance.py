from __future__ import annotations

import json
import math
import shlex
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.execution_timing_contracts import (
    TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
    repair_replay_contract_from_payload,
)
from quant_rabbit.forecast_precision import (
    DEFAULT_BIDASK_REPLAY_RULES_PATH,
    projection_precision_edge_summary,
    projection_precision_gap_summary,
)
from quant_rabbit.paths import (
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING,
    DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES,
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

# Acceptance recency window, not a market-risk threshold: one trading week is
# long enough to catch repeated live close leakage while allowing old repaired
# incidents to age out after a clean operating window.
LOSS_CLOSE_LEAK_LOOKBACK_DAYS = 7

# Monthly repair replay bar: this is an acceptance/audit coverage minimum, not
# a market threshold. A weekly clean window is not enough when capture economics
# still says TP exits are profitable but market-close leakage is net negative.
MONTH_SCALE_LOSS_CLOSE_REPLAY_MIN_HOURS = 24.0 * 30.0

# Engineering freshness guard for the append-only local gateway receipt stream.
# Live cycles run far more often than this; a larger gap means the ledger being
# audited is missing local gateway receipts and broker-close provenance cannot
# be classified as operator/GPT-unverified with confidence.
GATEWAY_RECEIPT_STREAM_STALE_GRACE_MINUTES = 120

# Mirrors the read-only bid/ask replay validator defaults in
# scripts/oanda_history_replay_validate.py. These are audit-quality thresholds,
# not market-risk thresholds: rank-only replay candidates must show multi-day
# coverage before they can be treated as repeatable daily campaign evidence.
BIDASK_REPLAY_STABLE_MIN_ACTIVE_DAYS = 3
BIDASK_REPLAY_STABLE_MAX_DAILY_SAMPLE_SHARE = 0.70
BIDASK_REPLAY_STABLE_MIN_POSITIVE_DAY_RATE = 2.0 / 3.0
BIDASK_REPLAY_AUTO_HISTORY_MIN_DAYS = 30
BIDASK_REPLAY_HISTORY_FETCH_DAYS = 120


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
        execution_ledger_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
        execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        bidask_rules_path: Path = DEFAULT_BIDASK_REPLAY_RULES_PATH,
        oanda_rotation_mining_path: Path = DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING,
    ) -> ProfitabilityAcceptanceSummary:
        generated_at_dt = datetime.now(timezone.utc)
        generated_at = generated_at_dt.isoformat()
        findings: list[dict[str, Any]] = []

        target = _load_json(target_state_path)
        intents = _load_json(order_intents_path)
        self_improvement = _load_json(self_improvement_path)
        capture = _load_json(capture_economics_path)
        bidask_rules = _load_json(bidask_rules_path)
        oanda_rotation_mining_path = _oanda_rotation_mining_effective_path(oanda_rotation_mining_path)
        oanda_rotation_mining = _load_json(oanda_rotation_mining_path)

        order_metrics = _order_intent_metrics(intents)
        target_metrics = _target_metrics(target)
        self_metrics, self_findings = _self_improvement_findings(self_improvement)
        capture_metrics, capture_findings = _capture_economics_findings(capture)
        capture_freshness_metrics, capture_freshness_findings = _order_capture_freshness_findings(
            intents,
            capture,
        )
        ledger_metrics, ledger_findings = _execution_ledger_close_findings(
            execution_ledger_path,
            execution_timing_audit_path=execution_timing_audit_path,
            now_utc=generated_at_dt,
        )
        replay_repair_metrics, replay_repair_findings = _profit_capture_replay_repair_findings(
            ledger_metrics.get("execution_timing_audit")
            if isinstance(ledger_metrics.get("execution_timing_audit"), dict)
            else {},
            self_metrics=self_metrics,
            capture_metrics=capture_metrics,
        )
        projection_metrics, projection_findings = _projection_precision_findings(projection_ledger_path)
        bidask_metrics, bidask_findings = _bidask_rule_findings(bidask_rules, bidask_rules_path)
        firepower_metrics, firepower_findings = _oanda_campaign_firepower_findings(
            oanda_rotation_mining,
            oanda_rotation_mining_path,
            target_open=bool(target_metrics["target_open"]),
        )

        findings.extend(self_findings)
        findings.extend(capture_findings)
        findings.extend(capture_freshness_findings)
        findings.extend(ledger_findings)
        findings.extend(replay_repair_findings)
        findings.extend(projection_findings)
        findings.extend(bidask_findings)
        findings.extend(firepower_findings)

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
            repair_frontier = order_metrics.get("repair_frontier")
            if isinstance(repair_frontier, dict) and int(repair_frontier.get("candidate_count") or 0) > 0:
                findings.append(
                    _finding(
                        priority="P1",
                        code="REPAIR_FRONTIER_BLOCKED",
                        message=(
                            f"{repair_frontier.get('candidate_count')} repair-mode candidate(s) exist, "
                            "but none currently clear live gates"
                        ),
                        next_action=(
                            "Work the repair frontier's top remaining blockers before adding new "
                            "indicators or loosening unrelated gates. A repair candidate must clear "
                            "forecast, spread, strategy, risk, broker-truth, and gateway checks together."
                        ),
                        evidence=repair_frontier,
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
            "order_capture_freshness": capture_freshness_metrics,
            "execution_ledger_close_leak": ledger_metrics,
            "profit_capture_replay_repair": replay_repair_metrics,
            "projection_precision": projection_metrics,
            "bidask_replay_rules": bidask_metrics,
            "oanda_campaign_firepower": firepower_metrics,
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


def _paths_equivalent(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left == right


def _oanda_rotation_mining_effective_path(path: Path) -> Path:
    """Use the tracked packaged OANDA audit when the non-tracked latest is absent."""

    if path.exists():
        return path
    if (
        _paths_equivalent(path, DEFAULT_OANDA_UNIVERSAL_ROTATION_MINING)
        and DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES.exists()
    ):
        return DEFAULT_OANDA_UNIVERSAL_ROTATION_PACKAGED_RULES
    return path


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
        for key in _order_intent_blocker_codes(item):
            blockers[key] = blockers.get(key, 0) + 1
    repair_frontier = _order_intent_repair_frontier(results)
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
        "repair_frontier": repair_frontier,
    }


def _order_intent_blocker_codes(item: dict[str, Any]) -> tuple[str, ...]:
    """Return one canonical blocker-code set for an intent result.

    Current `order_intents.json` publishes `live_blocker_codes` exactly to avoid
    ranking prose fragments as separate blocker categories. Older artifacts did
    not have that field, so the fallback still extracts issue codes and finally
    a bounded legacy label from `live_blockers`.
    """

    if not isinstance(item, dict):
        return ()
    live_codes: list[str] = []
    for raw in item.get("live_blocker_codes") or []:
        if isinstance(raw, dict):
            code = str(raw.get("code") or "").strip()
        else:
            code = str(raw or "").strip()
        if code:
            live_codes.append(code)
    if live_codes:
        return tuple(dict.fromkeys(live_codes))

    fallback: list[str] = []
    for issue in item.get("risk_issues") or item.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "").strip().upper()
        if severity and severity != "BLOCK":
            continue
        code = str(issue.get("code") or "").strip()
        if code:
            fallback.append(code)
    if fallback:
        return tuple(dict.fromkeys(fallback))

    for raw in item.get("live_blockers") or []:
        if isinstance(raw, dict):
            text = str(raw.get("code") or raw.get("message") or "").strip()
        else:
            text = str(raw or "").strip()
        if text:
            fallback.append(text.split(":", 1)[0][:80])
    return tuple(dict.fromkeys(fallback))


_REPAIR_ROTATION_MODES = {
    "TP_PROVEN_HARVEST",
    "TP_PROOF_COLLECTION_HARVEST",
    "OANDA_CAMPAIGN_FIREPOWER_HARVEST",
}


def _order_intent_repair_frontier(results: list[Any]) -> dict[str, Any]:
    """Summarize repair candidates and the blockers still preventing live use.

    This is a debugging invariant, not a live-permission grant. It prevents a
    red acceptance packet from saying only "NO_LIVE_READY" when the actionable
    truth is "repair candidates exist, but these exact gates still stop them".
    """

    candidates: list[dict[str, Any]] = []
    remaining_blockers: dict[str, int] = {}
    live_ready_count = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        repair_mode = _order_intent_repair_mode(metadata)
        if repair_mode is None:
            continue
        blocker_codes = _order_intent_blocker_codes(item)
        status = str(item.get("status") or "")
        is_live_ready = status == "LIVE_READY" and not blocker_codes
        if is_live_ready:
            live_ready_count += 1
        else:
            for code in blocker_codes:
                remaining_blockers[code] = remaining_blockers.get(code, 0) + 1
        market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
        candidates.append(
            {
                "lane_id": item.get("lane_id"),
                "status": status or None,
                "risk_allowed": item.get("risk_allowed"),
                "pair": intent.get("pair"),
                "side": intent.get("side"),
                "method": market_context.get("method") or metadata.get("method"),
                "order_type": intent.get("order_type"),
                "repair_mode": repair_mode,
                "blocker_count": len(blocker_codes),
                "blocker_codes": list(blocker_codes[:8]),
                "positive_rotation_pessimistic_expectancy_jpy": metadata.get(
                    "positive_rotation_pessimistic_expectancy_jpy"
                ),
                "oanda_campaign_matching_vehicle_key": metadata.get(
                    "positive_rotation_oanda_campaign_matching_vehicle_key"
                ),
            }
        )

    examples = sorted(candidates, key=_repair_frontier_example_sort_key)[:8]
    return {
        "candidate_count": len(candidates),
        "live_ready_count": live_ready_count,
        "blocked_count": len(candidates) - live_ready_count,
        "top_remaining_blockers": [
            {"code": code, "count": count}
            for code, count in sorted(
                remaining_blockers.items(),
                key=lambda item: (-item[1], item[0]),
            )[:10]
        ],
        "examples": examples,
    }


def _order_intent_repair_mode(metadata: dict[str, Any]) -> str | None:
    mode = str(metadata.get("positive_rotation_mode") or "").strip().upper()
    if mode in _REPAIR_ROTATION_MODES:
        return mode
    if metadata.get("self_improvement_p0_repair_live_ready") is True:
        repair_mode = str(metadata.get("self_improvement_p0_repair_mode") or "").strip().upper()
        return repair_mode or "TP_HARVEST_REPAIR"
    if metadata.get("self_improvement_pending_execution_repair_live_ready") is True:
        repair_mode = str(metadata.get("self_improvement_pending_execution_repair_mode") or "").strip().upper()
        return repair_mode or "PENDING_EXECUTION_TP_HARVEST_REPAIR"
    return None


def _repair_frontier_example_sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
    status_rank = 0 if item.get("status") == "LIVE_READY" else 1
    blocker_count = int(item.get("blocker_count") or 0)
    return status_rank, blocker_count, str(item.get("lane_id") or "")


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


def _order_capture_freshness_findings(
    intents: dict[str, Any],
    capture: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    intent_generated = _parse_utc(str(intents.get("generated_at_utc") or ""))
    capture_generated = _parse_utc(str(capture.get("generated_at_utc") or ""))
    results = intents.get("results") if isinstance(intents.get("results"), list) else []
    capture_overall = capture.get("overall") if isinstance(capture.get("overall"), dict) else {}
    capture_trades = int(_optional_float(capture_overall.get("trades")) or 0)
    embedded_values: list[int] = []
    embedded_examples: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        intent = item.get("intent") if isinstance(item.get("intent"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        embedded = _optional_float(metadata.get("capture_economics_trades"))
        if embedded is None:
            continue
        embedded_int = int(embedded)
        embedded_values.append(embedded_int)
        if len(embedded_examples) < 5 and embedded_int != capture_trades:
            embedded_examples.append(
                {
                    "lane_id": item.get("lane_id"),
                    "status": item.get("status"),
                    "intent_capture_economics_trades": embedded_int,
                }
            )
    metadata_mismatch = bool(
        capture_trades > 0
        and embedded_values
        and any(value != capture_trades for value in embedded_values)
    )
    timestamp_stale = bool(
        intent_generated is not None
        and capture_generated is not None
        and capture_generated > intent_generated
    )
    metrics = {
        "order_intents_generated_at_utc": intents.get("generated_at_utc"),
        "capture_economics_generated_at_utc": capture.get("generated_at_utc"),
        "capture_generated_after_order_intents": timestamp_stale,
        "capture_trades": capture_trades,
        "intent_capture_economics_trades": sorted(set(embedded_values)),
        "metadata_trade_count_mismatch": metadata_mismatch,
        "mismatch_examples": embedded_examples,
    }
    if not timestamp_stale and not metadata_mismatch:
        return metrics, []
    reasons: list[str] = []
    if timestamp_stale:
        reasons.append("capture_economics generated after order_intents")
    if metadata_mismatch:
        reasons.append("intent metadata embeds stale capture_economics trade count")
    return metrics, [
        _finding(
            priority="P0",
            code="ORDER_INTENTS_CAPTURE_ECONOMICS_STALE",
            message="order_intents were priced from stale capture_economics evidence",
            next_action=(
                "Refresh capture-economics before generate-intents, then regenerate order_intents. "
                "Do not trust live coverage or loss-asymmetry relaxation from this packet."
            ),
            evidence={**metrics, "reasons": reasons},
        )
    ]


def _execution_ledger_close_findings(
    path: Path,
    *,
    execution_timing_audit_path: Path | None = None,
    now_utc: datetime | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    timing_labels, timing_metrics = _execution_timing_loss_close_labels(execution_timing_audit_path)
    audit_now = _normalize_utc(now_utc or datetime.now(timezone.utc))
    metrics: dict[str, Any] = {
        "path": str(path),
        "ledger_exists": path.exists(),
        "lookback_days": LOSS_CLOSE_LEAK_LOOKBACK_DAYS,
        "execution_timing_audit": timing_metrics,
        "gateway_market_closes": 0,
        "recent_gateway_market_closes": 0,
        "recent_loss_closes": 0,
        "recent_loss_net_jpy": 0.0,
        "recent_loss_timing_label_counts": {},
        "recent_leak_loss_closes": 0,
        "recent_leak_loss_net_jpy": 0.0,
        "recent_contained_risk_loss_closes": 0,
        "recent_contained_risk_loss_net_jpy": 0.0,
        "recent_unclassified_loss_closes": 0,
        "recent_premature_loss_closes": 0,
        "recent_unverified_loss_closes": 0,
        "recent_unverified_loss_net_jpy": 0.0,
        "recent_close_gate_unverified_loss_closes": 0,
        "recent_close_gate_unverified_loss_net_jpy": 0.0,
        "recent_close_gate_missing_loss_closes": 0,
        "recent_close_gate_missing_loss_net_jpy": 0.0,
        "recent_close_gate_not_passing_loss_closes": 0,
        "recent_close_gate_not_passing_loss_net_jpy": 0.0,
        "recent_loss_by_lane": [],
        "recent_leak_loss_by_lane": [],
        "latest_gateway_market_close_ts_utc": None,
        "latest_loss_close_ts_utc": None,
        "gateway_event_stream_events": 0,
        "gateway_event_stream_latest_ts_utc": None,
        "gateway_event_stream_lag_minutes": None,
        "gateway_event_stream_market_close_gap_minutes": None,
        "gateway_event_stream_stale": False,
        "recent_loss_examples": [],
        "recent_leak_loss_examples": [],
        "recent_contained_risk_loss_examples": [],
        "recent_unverified_loss_examples": [],
        "recent_close_gate_unverified_loss_examples": [],
        "recent_close_gate_missing_loss_examples": [],
        "recent_close_gate_not_passing_loss_examples": [],
    }
    if not path.exists():
        return metrics, []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(execution_events)").fetchall()
            }
            tables = {
                str(row[0])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            gateway_exit_reason_select = (
                "g.exit_reason AS gateway_exit_reason"
                if "exit_reason" in columns
                else "NULL AS gateway_exit_reason"
            )
            gateway_raw_json_select = (
                "g.raw_json AS gateway_raw_json" if "raw_json" in columns else "NULL AS gateway_raw_json"
            )
            if "verification_observations" in tables:
                close_gate_select = """
                    EXISTS (
                        SELECT 1
                        FROM verification_observations v
                        WHERE v.check_name = 'close_gate_evidence'
                          AND v.subject_id = g.trade_id
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
                    ) AS has_close_gate_evidence,
                    EXISTS (
                        SELECT 1
                        FROM verification_observations v
                        WHERE v.check_name = 'close_gate_evidence'
                          AND v.subject_id = g.trade_id
                          AND v.status = 'PASS'
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
                    ) AS has_passing_close_gate_evidence
                """
            else:
                close_gate_select = """
                    0 AS has_close_gate_evidence,
                    0 AS has_passing_close_gate_evidence
                """
            gateway_event_stream_latest: datetime | None = None
            if "source" in columns:
                stream_row = conn.execute(
                    """
                    SELECT COUNT(*) AS n, MAX(ts_utc) AS latest_ts
                    FROM execution_events
                    WHERE source = 'gateway'
                    """
                ).fetchone()
                if stream_row is not None:
                    metrics["gateway_event_stream_events"] = int(stream_row["n"] or 0)
                    metrics["gateway_event_stream_latest_ts_utc"] = stream_row["latest_ts"]
                    gateway_event_stream_latest = _parse_utc(stream_row["latest_ts"])
            rows = conn.execute(
                f"""
                WITH gateway_closes AS (
                    SELECT *
                    FROM execution_events
                    WHERE event_type IN ('GATEWAY_TRADE_CLOSE_SENT', 'GATEWAY_TRADE_CLOSE_RECONCILED')
                ),
                entry_lanes AS (
                    SELECT
                        trade_id,
                        MAX(NULLIF(lane_id, '')) AS entry_lane_id,
                        MAX(NULLIF(pair, '')) AS entry_pair,
                        MAX(NULLIF(side, '')) AS entry_side
                    FROM execution_events
                    WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_FILLED')
                      AND COALESCE(trade_id, '') != ''
                    GROUP BY trade_id
                )
                SELECT
                    g.ts_utc AS gateway_ts_utc,
                    g.trade_id AS trade_id,
                    g.order_id AS order_id,
                    COALESCE(NULLIF(g.lane_id, ''), NULLIF(c.lane_id, ''), entry.entry_lane_id) AS lane_id,
                    COALESCE(NULLIF(g.pair, ''), NULLIF(c.pair, ''), entry.entry_pair) AS pair,
                    COALESCE(NULLIF(g.side, ''), NULLIF(c.side, ''), entry.entry_side) AS side,
                    c.ts_utc AS close_ts_utc,
                    c.realized_pl_jpy AS realized_pl_jpy,
                    c.exit_reason AS exit_reason,
                    {gateway_exit_reason_select},
                    {gateway_raw_json_select},
                    {close_gate_select},
                    EXISTS (
                        SELECT 1
                        FROM execution_events sent
                        WHERE sent.event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                          AND (
                              (COALESCE(sent.trade_id, '') != '' AND sent.trade_id = g.trade_id)
                              OR (COALESCE(sent.order_id, '') != '' AND sent.order_id = g.order_id)
                          )
                    ) AS has_gateway_close_sent,
                    EXISTS (
                        SELECT 1
                        FROM execution_events accepted
                        WHERE accepted.event_type = 'GATEWAY_GPT_CLOSE_ACCEPTED'
                          AND COALESCE(accepted.trade_id, '') != ''
                          AND accepted.trade_id = g.trade_id
                    ) AS has_gpt_close_accepted
                FROM gateway_closes g
                INNER JOIN execution_events c
                  ON c.event_type = 'TRADE_CLOSED'
                 AND c.trade_id = g.trade_id
                 AND (
                     COALESCE(g.order_id, '') = ''
                     OR COALESCE(c.order_id, '') = ''
                     OR c.order_id = g.order_id
                 )
                LEFT JOIN entry_lanes entry
                  ON entry.trade_id = g.trade_id
                WHERE c.exit_reason = 'MARKET_ORDER_TRADE_CLOSE'
                  AND NOT (
                      g.event_type = 'GATEWAY_TRADE_CLOSE_RECONCILED'
                      AND EXISTS (
                          SELECT 1
                          FROM execution_events direct
                          WHERE direct.event_type = 'GATEWAY_TRADE_CLOSE_SENT'
                            AND (
                                (COALESCE(direct.trade_id, '') != '' AND direct.trade_id = g.trade_id)
                                OR (COALESCE(direct.order_id, '') != '' AND direct.order_id = g.order_id)
                        )
                      )
                  )
                ORDER BY g.ts_utc ASC
                """
            ).fetchall()
    except sqlite3.Error as exc:
        return {
            **metrics,
            "ledger_read_error": f"{type(exc).__name__}: {exc}",
        }, [
            _finding(
                priority="P1",
                code="EXECUTION_LEDGER_CLOSE_LEAK_UNREADABLE",
                message="execution ledger could not be scanned for gateway market-close leakage",
                next_action=(
                    "Repair execution_ledger.db readability before treating profitability acceptance as proven."
                ),
                evidence={"path": str(path), "error": f"{type(exc).__name__}: {exc}"},
            )
        ]

    parsed: list[dict[str, Any]] = []
    latest_ts: datetime | None = None
    for row in rows:
        ts = _parse_utc(row["gateway_ts_utc"]) or _parse_utc(row["close_ts_utc"])
        if ts is None:
            continue
        latest_ts = ts if latest_ts is None or ts > latest_ts else latest_ts
        close_provenance = _reconciled_close_provenance(
            gateway_exit_reason=row["gateway_exit_reason"],
            gateway_raw_json=row["gateway_raw_json"],
            has_gateway_close_sent=bool(row["has_gateway_close_sent"]),
            has_gpt_close_accepted=bool(row["has_gpt_close_accepted"]),
        )
        parsed.append(
            {
                "ts": ts,
                "ts_utc": row["gateway_ts_utc"] or row["close_ts_utc"],
                "trade_id": row["trade_id"],
                "order_id": row["order_id"],
                "lane_id": row["lane_id"],
                "pair": row["pair"],
                "side": row["side"],
                "realized_pl_jpy": _optional_float(row["realized_pl_jpy"]),
                "exit_reason": row["exit_reason"],
                "gateway_exit_reason": row["gateway_exit_reason"],
                "close_provenance": close_provenance,
                "has_close_gate_evidence": bool(row["has_close_gate_evidence"]),
                "has_passing_close_gate_evidence": bool(row["has_passing_close_gate_evidence"]),
            }
        )
    metrics["gateway_market_closes"] = len(parsed)
    if latest_ts is not None:
        metrics["latest_gateway_market_close_ts_utc"] = latest_ts.isoformat()
    gateway_stream_stale = False
    if gateway_event_stream_latest is not None:
        lag_minutes = max(0.0, (audit_now - gateway_event_stream_latest).total_seconds() / 60.0)
        metrics["gateway_event_stream_lag_minutes"] = round(lag_minutes, 3)
        gateway_stream_stale = lag_minutes > GATEWAY_RECEIPT_STREAM_STALE_GRACE_MINUTES
        if latest_ts is not None:
            market_close_gap_minutes = max(
                0.0,
                (latest_ts - gateway_event_stream_latest).total_seconds() / 60.0,
            )
            metrics["gateway_event_stream_market_close_gap_minutes"] = round(
                market_close_gap_minutes,
                3,
            )
            gateway_stream_stale = (
                gateway_stream_stale
                or market_close_gap_minutes > GATEWAY_RECEIPT_STREAM_STALE_GRACE_MINUTES
            )
    elif latest_ts is not None and "source" in columns:
        gateway_stream_stale = True
    metrics["gateway_event_stream_stale"] = gateway_stream_stale
    if not parsed or latest_ts is None:
        return metrics, []

    cutoff = latest_ts - timedelta(days=LOSS_CLOSE_LEAK_LOOKBACK_DAYS)
    recent = [row for row in parsed if row["ts"] >= cutoff]
    recent_losses = [
        row for row in recent if (_optional_float(row.get("realized_pl_jpy")) or 0.0) < 0.0
    ]
    for row in recent_losses:
        label = _loss_close_timing_label(row, timing_labels)
        if label:
            row["timing_path_label"] = label
    recent_contained_losses = [
        row for row in recent_losses if row.get("timing_path_label") == "LOSS_CLOSE_CONTAINED_RISK"
    ]
    recent_leak_losses = [
        row for row in recent_losses if row.get("timing_path_label") != "LOSS_CLOSE_CONTAINED_RISK"
    ]
    recent_premature_losses = [
        row
        for row in recent_losses
        if row.get("timing_path_label") == "LOSS_CLOSE_MAY_HAVE_BEEN_PREMATURE"
    ]
    recent_unclassified_losses = [row for row in recent_losses if not row.get("timing_path_label")]
    recent_unverified_losses = [
        row
        for row in recent_losses
        if row.get("close_provenance") == "GATEWAY_TRADE_CLOSE_RECONCILED_UNVERIFIED"
    ]
    recent_close_gate_unverified_losses = [
        row
        for row in recent_losses
        if str(row.get("gateway_exit_reason") or "").strip().upper() == "GPT_CLOSE"
        and row.get("has_passing_close_gate_evidence") is not True
    ]
    recent_close_gate_missing_losses = [
        row
        for row in recent_close_gate_unverified_losses
        if row.get("has_close_gate_evidence") is not True
    ]
    recent_close_gate_not_passing_losses = [
        row
        for row in recent_close_gate_unverified_losses
        if row.get("has_close_gate_evidence") is True
    ]
    metrics["recent_gateway_market_closes"] = len(recent)
    metrics["recent_loss_closes"] = len(recent_losses)
    metrics["recent_loss_net_jpy"] = round(
        sum(float(row.get("realized_pl_jpy") or 0.0) for row in recent_losses),
        4,
    )
    label_counts: dict[str, int] = {}
    for row in recent_losses:
        label = str(row.get("timing_path_label") or "UNCLASSIFIED")
        label_counts[label] = label_counts.get(label, 0) + 1
    metrics["recent_loss_timing_label_counts"] = dict(sorted(label_counts.items()))
    metrics["recent_leak_loss_closes"] = len(recent_leak_losses)
    metrics["recent_leak_loss_net_jpy"] = round(
        sum(float(row.get("realized_pl_jpy") or 0.0) for row in recent_leak_losses),
        4,
    )
    metrics["recent_contained_risk_loss_closes"] = len(recent_contained_losses)
    metrics["recent_contained_risk_loss_net_jpy"] = round(
        sum(float(row.get("realized_pl_jpy") or 0.0) for row in recent_contained_losses),
        4,
    )
    metrics["recent_unclassified_loss_closes"] = len(recent_unclassified_losses)
    metrics["recent_premature_loss_closes"] = len(recent_premature_losses)
    metrics["recent_unverified_loss_closes"] = len(recent_unverified_losses)
    metrics["recent_unverified_loss_net_jpy"] = round(
        sum(float(row.get("realized_pl_jpy") or 0.0) for row in recent_unverified_losses),
        4,
    )
    metrics["recent_close_gate_unverified_loss_closes"] = len(recent_close_gate_unverified_losses)
    metrics["recent_close_gate_unverified_loss_net_jpy"] = round(
        sum(float(row.get("realized_pl_jpy") or 0.0) for row in recent_close_gate_unverified_losses),
        4,
    )
    metrics["recent_close_gate_missing_loss_closes"] = len(recent_close_gate_missing_losses)
    metrics["recent_close_gate_missing_loss_net_jpy"] = round(
        sum(float(row.get("realized_pl_jpy") or 0.0) for row in recent_close_gate_missing_losses),
        4,
    )
    metrics["recent_close_gate_not_passing_loss_closes"] = len(recent_close_gate_not_passing_losses)
    metrics["recent_close_gate_not_passing_loss_net_jpy"] = round(
        sum(float(row.get("realized_pl_jpy") or 0.0) for row in recent_close_gate_not_passing_losses),
        4,
    )
    metrics["recent_loss_by_lane"] = _loss_close_by_lane(recent_losses)
    metrics["recent_leak_loss_by_lane"] = _loss_close_by_lane(recent_leak_losses)
    if recent_losses:
        latest_loss = max(recent_losses, key=lambda row: row["ts"])
        metrics["latest_loss_close_ts_utc"] = latest_loss["ts"].isoformat()
    metrics["recent_loss_examples"] = [
        {
            "ts_utc": row.get("ts_utc"),
            "trade_id": row.get("trade_id"),
            "order_id": row.get("order_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "lane_id": row.get("lane_id"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "close_provenance": row.get("close_provenance"),
            "timing_path_label": row.get("timing_path_label"),
        }
        for row in sorted(
            recent_losses,
            key=lambda row: float(row.get("realized_pl_jpy") or 0.0),
        )[:5]
    ]
    metrics["recent_leak_loss_examples"] = [
        {
            "ts_utc": row.get("ts_utc"),
            "trade_id": row.get("trade_id"),
            "order_id": row.get("order_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "lane_id": row.get("lane_id"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "close_provenance": row.get("close_provenance"),
            "timing_path_label": row.get("timing_path_label"),
        }
        for row in sorted(
            recent_leak_losses,
            key=lambda row: float(row.get("realized_pl_jpy") or 0.0),
        )[:5]
    ]
    metrics["recent_contained_risk_loss_examples"] = [
        {
            "ts_utc": row.get("ts_utc"),
            "trade_id": row.get("trade_id"),
            "order_id": row.get("order_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "lane_id": row.get("lane_id"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "close_provenance": row.get("close_provenance"),
            "timing_path_label": row.get("timing_path_label"),
        }
        for row in sorted(
            recent_contained_losses,
            key=lambda row: float(row.get("realized_pl_jpy") or 0.0),
        )[:5]
    ]
    metrics["recent_unverified_loss_examples"] = [
        {
            "ts_utc": row.get("ts_utc"),
            "trade_id": row.get("trade_id"),
            "order_id": row.get("order_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "lane_id": row.get("lane_id"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "gateway_exit_reason": row.get("gateway_exit_reason"),
            "close_provenance": row.get("close_provenance"),
            "timing_path_label": row.get("timing_path_label"),
        }
        for row in sorted(
            recent_unverified_losses,
            key=lambda row: float(row.get("realized_pl_jpy") or 0.0),
        )[:5]
    ]
    metrics["recent_close_gate_unverified_loss_examples"] = [
        {
            "ts_utc": row.get("ts_utc"),
            "trade_id": row.get("trade_id"),
            "order_id": row.get("order_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "lane_id": row.get("lane_id"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "gateway_exit_reason": row.get("gateway_exit_reason"),
            "close_provenance": row.get("close_provenance"),
            "timing_path_label": row.get("timing_path_label"),
            "has_close_gate_evidence": row.get("has_close_gate_evidence"),
            "has_passing_close_gate_evidence": row.get("has_passing_close_gate_evidence"),
        }
        for row in sorted(
            recent_close_gate_unverified_losses,
            key=lambda row: float(row.get("realized_pl_jpy") or 0.0),
        )[:5]
    ]
    metrics["recent_close_gate_missing_loss_examples"] = [
        {
            "ts_utc": row.get("ts_utc"),
            "trade_id": row.get("trade_id"),
            "order_id": row.get("order_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "lane_id": row.get("lane_id"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "gateway_exit_reason": row.get("gateway_exit_reason"),
            "close_provenance": row.get("close_provenance"),
            "timing_path_label": row.get("timing_path_label"),
            "has_close_gate_evidence": row.get("has_close_gate_evidence"),
            "has_passing_close_gate_evidence": row.get("has_passing_close_gate_evidence"),
        }
        for row in sorted(
            recent_close_gate_missing_losses,
            key=lambda row: float(row.get("realized_pl_jpy") or 0.0),
        )[:5]
    ]
    metrics["recent_close_gate_not_passing_loss_examples"] = [
        {
            "ts_utc": row.get("ts_utc"),
            "trade_id": row.get("trade_id"),
            "order_id": row.get("order_id"),
            "pair": row.get("pair"),
            "side": row.get("side"),
            "lane_id": row.get("lane_id"),
            "realized_pl_jpy": row.get("realized_pl_jpy"),
            "gateway_exit_reason": row.get("gateway_exit_reason"),
            "close_provenance": row.get("close_provenance"),
            "timing_path_label": row.get("timing_path_label"),
            "has_close_gate_evidence": row.get("has_close_gate_evidence"),
            "has_passing_close_gate_evidence": row.get("has_passing_close_gate_evidence"),
        }
        for row in sorted(
            recent_close_gate_not_passing_losses,
            key=lambda row: float(row.get("realized_pl_jpy") or 0.0),
        )[:5]
    ]
    if not recent_losses:
        return metrics, []
    findings: list[dict[str, Any]] = []
    if recent_leak_losses:
        findings.append(
            _finding(
                priority="P0",
                code="RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
                message=(
                    f"{len(recent_leak_losses)} loss-side gateway MARKET_ORDER_TRADE_CLOSE event(s) "
                    f"remain inside the {LOSS_CLOSE_LEAK_LOOKBACK_DAYS}-day acceptance window "
                    "without contained-risk timing evidence"
                ),
                next_action=(
                    "Keep profitability acceptance red for premature or unclassified loss-side market "
                    "closes. Do not count LOSS_CLOSE_CONTAINED_RISK rows as leak repair proof unless "
                    "the ledger also has durable GPT/gateway close receipts."
                ),
                evidence={
                    "recent_leak_loss_closes": metrics["recent_leak_loss_closes"],
                    "recent_leak_loss_net_jpy": metrics["recent_leak_loss_net_jpy"],
                    "recent_loss_timing_label_counts": metrics["recent_loss_timing_label_counts"],
                    "latest_loss_close_ts_utc": metrics["latest_loss_close_ts_utc"],
                    "by_lane": metrics["recent_leak_loss_by_lane"],
                    "examples": metrics["recent_leak_loss_examples"],
                    "contained_risk_loss_closes": metrics["recent_contained_risk_loss_closes"],
                    "contained_risk_loss_net_jpy": metrics["recent_contained_risk_loss_net_jpy"],
                },
            )
        )
    if gateway_stream_stale:
        findings.append(
            _finding(
                priority="P0",
                code="EXECUTION_LEDGER_GATEWAY_RECEIPT_STREAM_STALE",
                message=(
                    "execution ledger gateway receipt stream is stale relative to broker market-close truth; "
                    "loss-close provenance cannot be classified from this database"
                ),
                next_action=(
                    "Refresh or read the live runtime execution_ledger.db before diagnosing recent loss-side "
                    "market closes as unverified. Do not tune close discipline from a ledger missing local "
                    "GATEWAY_GPT_CLOSE_ACCEPTED / GATEWAY_TRADE_CLOSE_SENT receipts."
                ),
                evidence={
                    "ledger_path": str(path),
                    "latest_gateway_market_close_ts_utc": metrics["latest_gateway_market_close_ts_utc"],
                    "gateway_event_stream_latest_ts_utc": metrics["gateway_event_stream_latest_ts_utc"],
                    "gateway_event_stream_events": metrics["gateway_event_stream_events"],
                    "gateway_event_stream_lag_minutes": metrics["gateway_event_stream_lag_minutes"],
                    "gateway_event_stream_market_close_gap_minutes": metrics[
                        "gateway_event_stream_market_close_gap_minutes"
                    ],
                },
            )
        )
    elif recent_unverified_losses:
        findings.append(
            _finding(
                priority="P0",
                code="UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED",
                message=(
                    f"{len(recent_unverified_losses)} recent loss-side market close(s) were only "
                    "reconciled from broker trade-close truth, without a durable GPT Gate A/B or "
                    "PositionProtectionGateway send receipt"
                ),
                next_action=(
                    "Do not treat these closes as proved loss-cut discipline. Future loss-side market closes "
                    "must persist GATEWAY_GPT_CLOSE_ACCEPTED and/or GATEWAY_TRADE_CLOSE_SENT before the "
                    "broker TRADE_CLOSE, otherwise profitability acceptance stays red."
                ),
                evidence={
                    "recent_unverified_loss_closes": metrics["recent_unverified_loss_closes"],
                    "recent_unverified_loss_net_jpy": metrics["recent_unverified_loss_net_jpy"],
                    "examples": metrics["recent_unverified_loss_examples"],
                },
            )
        )
    if recent_close_gate_missing_losses:
        findings.append(
            _finding(
                priority="P0",
                code="LOSS_CLOSE_GATE_EVIDENCE_MISSING",
                message=(
                    f"{len(recent_close_gate_missing_losses)} recent GPT loss-side market close(s) "
                    "lack durable close_gate_evidence in verification_observations"
                ),
                next_action=(
                    "Persist gpt-trader close_gate_evidence into execution_ledger verification_observations "
                    "for accepted CLOSE receipts. Do not treat missing evidence as proved Gate A/B discipline."
                ),
                evidence={
                    "recent_close_gate_unverified_loss_closes": metrics[
                        "recent_close_gate_unverified_loss_closes"
                    ],
                    "recent_close_gate_unverified_loss_net_jpy": metrics[
                        "recent_close_gate_unverified_loss_net_jpy"
                    ],
                    "recent_close_gate_missing_loss_closes": metrics[
                        "recent_close_gate_missing_loss_closes"
                    ],
                    "recent_close_gate_missing_loss_net_jpy": metrics[
                        "recent_close_gate_missing_loss_net_jpy"
                    ],
                    "recent_close_gate_not_passing_loss_closes": metrics[
                        "recent_close_gate_not_passing_loss_closes"
                    ],
                    "examples": metrics["recent_close_gate_missing_loss_examples"],
                },
            )
        )
    if recent_close_gate_not_passing_losses:
        findings.append(
            _finding(
                priority="P0",
                code="LOSS_CLOSE_GATE_EVIDENCE_NOT_PASSING",
                message=(
                    f"{len(recent_close_gate_not_passing_losses)} recent GPT loss-side market close(s) "
                    "have durable close_gate_evidence, but no PASS close_gate_evidence"
                ),
                next_action=(
                    "Do not synthesize PASS for historical closes. Future loss-side closes must persist "
                    "PASS close_gate_evidence, or these failed-evidence closes must age out of the 7-day "
                    "acceptance window without new leaks."
                ),
                evidence={
                    "recent_close_gate_unverified_loss_closes": metrics[
                        "recent_close_gate_unverified_loss_closes"
                    ],
                    "recent_close_gate_unverified_loss_net_jpy": metrics[
                        "recent_close_gate_unverified_loss_net_jpy"
                    ],
                    "recent_close_gate_not_passing_loss_closes": metrics[
                        "recent_close_gate_not_passing_loss_closes"
                    ],
                    "recent_close_gate_not_passing_loss_net_jpy": metrics[
                        "recent_close_gate_not_passing_loss_net_jpy"
                    ],
                    "recent_close_gate_missing_loss_closes": metrics[
                        "recent_close_gate_missing_loss_closes"
                    ],
                    "examples": metrics["recent_close_gate_not_passing_loss_examples"],
                },
            )
        )
    return metrics, findings


def _reconciled_close_provenance(
    *,
    gateway_exit_reason: Any,
    gateway_raw_json: Any,
    has_gateway_close_sent: bool,
    has_gpt_close_accepted: bool,
) -> str:
    if has_gateway_close_sent:
        return "GATEWAY_TRADE_CLOSE_SENT"
    raw = _json_dict(gateway_raw_json)
    reconciled_from = {
        str(item or "").strip().upper()
        for item in raw.get("reconciled_from", []) or []
        if str(item or "").strip()
    }
    reason = str(gateway_exit_reason or "").strip().upper()
    if (
        has_gpt_close_accepted
        or reason == "GPT_CLOSE_RECONCILED"
        or "GATEWAY_GPT_CLOSE_ACCEPTED" in reconciled_from
    ):
        return "GATEWAY_GPT_CLOSE_RECONCILED"
    return "GATEWAY_TRADE_CLOSE_RECONCILED_UNVERIFIED"


def _execution_timing_loss_close_labels(
    path: Path | None,
) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    metrics: dict[str, Any] = {
        "path": str(path) if path is not None else None,
        "loaded": False,
        "generated_at_utc": None,
        "loss_closes_profit_capture_missed": 0,
        "loss_close_actual_pl_jpy": None,
        "loss_close_counterfactual_profit_capture_pl_jpy": None,
        "loss_close_counterfactual_profit_capture_delta_jpy": None,
        "loss_close_counterfactual_profit_capture_jpy": None,
        "loss_closes_repair_replay_triggered": 0,
        "repair_replay_contract": None,
        "repair_replay_contract_present": False,
        "loss_close_repair_replay_counterfactual_pl_jpy": None,
        "loss_close_repair_replay_delta_jpy": None,
        "loss_close_repair_replay_profit_capture_jpy": None,
        "loss_close_repair_replay_block_reasons": {},
        "window_from_utc": None,
        "window_to_utc": None,
        "window_lookback_hours": None,
        "loss_market_close_rows": 0,
        "label_counts": {},
        "top_profit_capture_misses": [],
        "top_repair_replay_triggers": [],
        "top_repair_replay_blocks": [],
        "top_repair_replay_residual_groups": [],
        "top_tp_progress_repair_residual_groups": [],
        "top_entry_quality_residual_groups": [],
        "top_repair_replay_residual_method_rollups": [],
        "top_tp_progress_repair_residual_method_rollups": [],
        "top_entry_quality_residual_method_rollups": [],
        "read_error": None,
    }
    if path is None:
        return {}, metrics
    if not path.exists():
        return {}, metrics
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        metrics["read_error"] = f"{type(exc).__name__}: {exc}"
        return {}, metrics
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
    metrics["window_from_utc"] = window.get("from_utc")
    metrics["window_to_utc"] = window.get("to_utc")
    metrics["window_lookback_hours"] = _optional_float(window.get("lookback_hours"))
    repair_replay_contract = repair_replay_contract_from_payload(payload)
    metrics["repair_replay_contract"] = repair_replay_contract
    metrics["repair_replay_contract_present"] = (
        repair_replay_contract == TP_PROGRESS_REPAIR_REPLAY_CONTRACT
    )
    metrics["loss_closes_profit_capture_missed"] = int(
        _optional_float(summary.get("loss_closes_profit_capture_missed")) or 0
    )
    metrics["loss_close_actual_pl_jpy"] = _optional_float(summary.get("loss_close_actual_pl_jpy"))
    metrics["loss_close_counterfactual_profit_capture_pl_jpy"] = _optional_float(
        summary.get("loss_close_counterfactual_profit_capture_pl_jpy")
    )
    metrics["loss_close_counterfactual_profit_capture_delta_jpy"] = _optional_float(
        summary.get("loss_close_counterfactual_profit_capture_delta_jpy")
    )
    metrics["loss_close_counterfactual_profit_capture_jpy"] = _optional_float(
        summary.get("loss_close_counterfactual_profit_capture_jpy")
    )
    metrics["loss_closes_repair_replay_triggered"] = int(
        _optional_float(summary.get("loss_closes_repair_replay_triggered")) or 0
    )
    metrics["loss_close_repair_replay_counterfactual_pl_jpy"] = _optional_float(
        summary.get("loss_close_repair_replay_counterfactual_pl_jpy")
    )
    metrics["loss_close_repair_replay_delta_jpy"] = _optional_float(
        summary.get("loss_close_repair_replay_delta_jpy")
    )
    metrics["loss_close_repair_replay_profit_capture_jpy"] = _optional_float(
        summary.get("loss_close_repair_replay_profit_capture_jpy")
    )
    block_reasons = summary.get("loss_close_repair_replay_block_reasons")
    metrics["loss_close_repair_replay_block_reasons"] = (
        dict(block_reasons) if isinstance(block_reasons, dict) else {}
    )
    rows = payload.get("market_close_counterfactuals")
    if not isinstance(rows, list):
        rows = []
    labels: dict[tuple[str, str], str] = {}
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("post_close_path_label") or "").strip().upper()
        if not label.startswith("LOSS_CLOSE_"):
            continue
        metrics["loss_market_close_rows"] = int(metrics["loss_market_close_rows"]) + 1
        counts[label] = counts.get(label, 0) + 1
        for key_name in ("trade_id", "order_id"):
            value = str(row.get(key_name) or "").strip()
            if value:
                labels[(key_name, value)] = label
    regret_rows = payload.get("loss_close_regrets")
    if not isinstance(regret_rows, list):
        regret_rows = []
    top_misses: list[dict[str, Any]] = []
    top_repair_triggers: list[dict[str, Any]] = []
    top_repair_blocks: list[dict[str, Any]] = []
    residual_groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in regret_rows:
        if not isinstance(row, dict):
            continue
        lane_id = row.get("lane_id")
        common = {
            "trade_id": row.get("trade_id"),
            "lane_id": lane_id,
            "pair": row.get("pair"),
            "side": row.get("side"),
            "exit_reason": row.get("exit_reason"),
            "realized_pl_jpy": _optional_float(row.get("realized_pl_jpy")),
            "tp_progress_before_loss_close": _optional_float(
                row.get("tp_progress_before_loss_close")
            ),
        }
        _add_repair_replay_residual_group(residual_groups, row)
        if row.get("profit_capture_missed_before_loss_close"):
            top_misses.append(
                {
                    **common,
                    "counterfactual_exit": row.get("profit_capture_counterfactual_exit"),
                    "counterfactual_jpy": _optional_float(row.get("profit_capture_counterfactual_jpy")),
                    "counterfactual_delta_jpy": _optional_float(
                        row.get("profit_capture_counterfactual_net_improvement_jpy")
                    ),
                    "repair_replay_block_reason": row.get("repair_replay_block_reason"),
                    "repair_replay_max_profit_pips": _optional_float(
                        row.get("repair_replay_max_profit_pips")
                    ),
                    "repair_replay_max_tp_progress": _optional_float(
                        row.get("repair_replay_max_tp_progress")
                    ),
                    "repair_replay_candidate_profit_pips": _optional_float(
                        row.get("repair_replay_candidate_profit_pips")
                    ),
                    "repair_replay_candidate_noise_floor_pips": _optional_float(
                        row.get("repair_replay_candidate_noise_floor_pips")
                    ),
                }
            )
        if (
            row.get("profit_capture_missed_before_loss_close")
            and not row.get("repair_replay_triggered_before_loss_close")
        ):
            top_repair_blocks.append(
                {
                    **common,
                    "repair_replay_block_reason": row.get("repair_replay_block_reason"),
                    "repair_replay_max_profit_pips": _optional_float(
                        row.get("repair_replay_max_profit_pips")
                    ),
                    "repair_replay_max_tp_progress": _optional_float(
                        row.get("repair_replay_max_tp_progress")
                    ),
                    "repair_replay_candidate_profit_pips": _optional_float(
                        row.get("repair_replay_candidate_profit_pips")
                    ),
                    "repair_replay_candidate_tp_progress": _optional_float(
                        row.get("repair_replay_candidate_tp_progress")
                    ),
                    "repair_replay_candidate_spread_pips": _optional_float(
                        row.get("repair_replay_candidate_spread_pips")
                    ),
                    "repair_replay_candidate_m1_atr_pips": _optional_float(
                        row.get("repair_replay_candidate_m1_atr_pips")
                    ),
                    "repair_replay_candidate_noise_floor_pips": _optional_float(
                        row.get("repair_replay_candidate_noise_floor_pips")
                    ),
                }
            )
        if row.get("repair_replay_triggered_before_loss_close"):
            top_repair_triggers.append(
                {
                    **common,
                    "repair_replay_exit": row.get("repair_replay_exit"),
                    "repair_trigger_at_utc": row.get("repair_replay_trigger_at_utc"),
                    "repair_profit_pips": _optional_float(row.get("repair_replay_profit_pips")),
                    "repair_tp_progress": _optional_float(row.get("repair_replay_tp_progress")),
                    "repair_noise_floor_pips": _optional_float(
                        row.get("repair_replay_noise_floor_pips")
                    ),
                    "repair_counterfactual_jpy": _optional_float(
                        row.get("repair_replay_counterfactual_jpy")
                    ),
                    "repair_counterfactual_delta_jpy": _optional_float(
                        row.get("repair_replay_counterfactual_net_improvement_jpy")
                    ),
                }
            )
    metrics["loaded"] = True
    metrics["generated_at_utc"] = payload.get("generated_at_utc") or payload.get("generated_at")
    metrics["label_counts"] = dict(sorted(counts.items()))
    metrics["top_profit_capture_misses"] = top_misses[:5]
    metrics["top_repair_replay_triggers"] = top_repair_triggers[:5]
    metrics["top_repair_replay_blocks"] = top_repair_blocks[:5]
    metrics["top_repair_replay_residual_groups"] = _top_repair_replay_residual_groups(
        residual_groups
    )
    metrics["top_tp_progress_repair_residual_groups"] = _top_repair_replay_residual_groups(
        residual_groups,
        scope_filter={
            "TP_PROGRESS_REPAIR_TRIGGERED",
            "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
        },
    )
    metrics["top_entry_quality_residual_groups"] = _top_repair_replay_residual_groups(
        residual_groups,
        scope_filter={"ENTRY_QUALITY_OR_CLOSE_RESIDUAL"},
    )
    metrics["top_repair_replay_residual_method_rollups"] = (
        _top_repair_replay_residual_method_rollups(residual_groups)
    )
    metrics["top_tp_progress_repair_residual_method_rollups"] = (
        _top_repair_replay_residual_method_rollups(
            residual_groups,
            scope_filter={
                "TP_PROGRESS_REPAIR_TRIGGERED",
                "TP_PROGRESS_DIAGNOSTIC_BLOCKED",
            },
        )
    )
    metrics["top_entry_quality_residual_method_rollups"] = (
        _top_repair_replay_residual_method_rollups(
            residual_groups,
            scope_filter={"ENTRY_QUALITY_OR_CLOSE_RESIDUAL"},
        )
    )
    return labels, metrics


def _add_repair_replay_residual_group(
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]],
    row: dict[str, Any],
) -> None:
    actual = _optional_float(row.get("realized_pl_jpy"))
    if actual is None or actual >= 0.0:
        return
    replay_pl = _optional_float(row.get("repair_replay_counterfactual_pl_jpy"))
    if replay_pl is None:
        replay_pl = actual
    if replay_pl >= 0.0:
        return
    pair = str(row.get("pair") or "UNKNOWN")
    side = str(row.get("side") or "UNKNOWN")
    lane_id = str(row.get("lane_id") or "")
    method = _method_from_lane_id(lane_id) or "UNKNOWN"
    exit_reason = str(row.get("exit_reason") or "UNKNOWN")
    residual_scope = _repair_replay_residual_scope(row)
    key = (residual_scope, pair, side, method, exit_reason)
    item = groups.setdefault(
        key,
        {
            "pair": pair,
            "side": side,
            "method": method,
            "exit_reason": exit_reason,
            "residual_scope": residual_scope,
            "loss_closes": 0,
            "actual_pl_jpy": 0.0,
            "repair_replay_pl_jpy": 0.0,
            "repair_replay_delta_jpy": 0.0,
            "repair_replay_triggered": 0,
            "block_reasons": {},
            "examples": [],
        },
    )
    item["loss_closes"] = int(item["loss_closes"]) + 1
    item["actual_pl_jpy"] = float(item["actual_pl_jpy"]) + actual
    item["repair_replay_pl_jpy"] = float(item["repair_replay_pl_jpy"]) + replay_pl
    item["repair_replay_delta_jpy"] = float(item["repair_replay_delta_jpy"]) + (
        replay_pl - actual
    )
    if row.get("repair_replay_triggered_before_loss_close"):
        item["repair_replay_triggered"] = int(item["repair_replay_triggered"]) + 1
    reason = str(row.get("repair_replay_block_reason") or "NO_REPAIR_REPLAY_TRIGGER")
    reasons = item["block_reasons"]
    if isinstance(reasons, dict):
        reasons[reason] = int(reasons.get(reason) or 0) + 1
    examples = item["examples"]
    if isinstance(examples, list) and len(examples) < 3:
        examples.append(
            {
                "trade_id": row.get("trade_id"),
                "lane_id": lane_id or None,
                "actual_pl_jpy": round(actual, 4),
                "repair_replay_pl_jpy": round(replay_pl, 4),
                "repair_replay_triggered": bool(
                    row.get("repair_replay_triggered_before_loss_close")
                ),
                "repair_replay_block_reason": row.get("repair_replay_block_reason"),
            }
        )


def _repair_replay_residual_scope(row: dict[str, Any]) -> str:
    if row.get("repair_replay_triggered_before_loss_close"):
        return "TP_PROGRESS_REPAIR_TRIGGERED"
    if row.get("profit_capture_missed_before_loss_close"):
        return "TP_PROGRESS_DIAGNOSTIC_BLOCKED"
    return "ENTRY_QUALITY_OR_CLOSE_RESIDUAL"


def _top_repair_replay_residual_groups(
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]],
    *,
    scope_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in groups.values():
        if scope_filter is not None and item.get("residual_scope") not in scope_filter:
            continue
        rows.append(
            {
                **item,
                "actual_pl_jpy": round(float(item.get("actual_pl_jpy") or 0.0), 4),
                "repair_replay_pl_jpy": round(
                    float(item.get("repair_replay_pl_jpy") or 0.0),
                    4,
                ),
                "repair_replay_delta_jpy": round(
                    float(item.get("repair_replay_delta_jpy") or 0.0),
                    4,
                ),
                "block_reasons": dict(
                    sorted(
                        (
                            (str(key), int(value))
                            for key, value in (
                                item.get("block_reasons")
                                if isinstance(item.get("block_reasons"), dict)
                                else {}
                            ).items()
                        ),
                        key=lambda part: (-part[1], part[0]),
                    )
                ),
            }
        )
    rows.sort(
        key=lambda item: (
            float(item.get("repair_replay_pl_jpy") or 0.0),
            float(item.get("actual_pl_jpy") or 0.0),
            str(item.get("pair") or ""),
            str(item.get("side") or ""),
            str(item.get("method") or ""),
        )
    )
    return rows[:10]


def _top_repair_replay_residual_method_rollups(
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]],
    *,
    scope_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    rollups: dict[tuple[str, str], dict[str, Any]] = {}
    for group in groups.values():
        residual_scope = str(group.get("residual_scope") or "UNKNOWN")
        if scope_filter is not None and residual_scope not in scope_filter:
            continue
        method = str(group.get("method") or "UNKNOWN")
        key = (residual_scope, method)
        item = rollups.setdefault(
            key,
            {
                "residual_scope": residual_scope,
                "method": method,
                "group_count": 0,
                "loss_closes": 0,
                "actual_pl_jpy": 0.0,
                "repair_replay_pl_jpy": 0.0,
                "repair_replay_delta_jpy": 0.0,
                "repair_replay_triggered": 0,
                "block_reasons": {},
                "exit_reasons": {},
                "pairs": {},
                "sides": {},
                "examples": [],
            },
        )
        loss_closes = int(group.get("loss_closes") or 0)
        item["group_count"] = int(item["group_count"]) + 1
        item["loss_closes"] = int(item["loss_closes"]) + loss_closes
        item["actual_pl_jpy"] = float(item["actual_pl_jpy"]) + float(
            group.get("actual_pl_jpy") or 0.0
        )
        item["repair_replay_pl_jpy"] = float(item["repair_replay_pl_jpy"]) + float(
            group.get("repair_replay_pl_jpy") or 0.0
        )
        item["repair_replay_delta_jpy"] = float(item["repair_replay_delta_jpy"]) + float(
            group.get("repair_replay_delta_jpy") or 0.0
        )
        item["repair_replay_triggered"] = int(item["repair_replay_triggered"]) + int(
            group.get("repair_replay_triggered") or 0
        )
        pair = str(group.get("pair") or "UNKNOWN")
        side = str(group.get("side") or "UNKNOWN")
        exit_reason = str(group.get("exit_reason") or "UNKNOWN")
        pair_counts = item["pairs"]
        side_counts = item["sides"]
        exit_counts = item["exit_reasons"]
        if isinstance(pair_counts, dict):
            pair_counts[pair] = int(pair_counts.get(pair) or 0) + loss_closes
        if isinstance(side_counts, dict):
            side_counts[side] = int(side_counts.get(side) or 0) + loss_closes
        if isinstance(exit_counts, dict):
            exit_counts[exit_reason] = int(exit_counts.get(exit_reason) or 0) + loss_closes
        source_reasons = (
            group.get("block_reasons") if isinstance(group.get("block_reasons"), dict) else {}
        )
        reasons = item["block_reasons"]
        if isinstance(reasons, dict):
            for reason, count in source_reasons.items():
                reasons[str(reason)] = int(reasons.get(str(reason)) or 0) + int(count or 0)
        examples = item["examples"]
        source_examples = group.get("examples") if isinstance(group.get("examples"), list) else []
        if isinstance(examples, list):
            for example in source_examples:
                if not isinstance(example, dict) or len(examples) >= 3:
                    break
                examples.append({**example, "pair": pair, "side": side, "method": method})

    rows: list[dict[str, Any]] = []
    for item in rollups.values():
        pair_counts = item.get("pairs") if isinstance(item.get("pairs"), dict) else {}
        side_counts = item.get("sides") if isinstance(item.get("sides"), dict) else {}
        exit_counts = item.get("exit_reasons") if isinstance(item.get("exit_reasons"), dict) else {}
        rows.append(
            {
                "residual_scope": item["residual_scope"],
                "method": item["method"],
                "group_count": int(item.get("group_count") or 0),
                "pair_count": len(pair_counts),
                "pairs": sorted(pair_counts),
                "side_count": len(side_counts),
                "sides": sorted(side_counts),
                "exit_reasons": dict(
                    sorted(
                        ((str(reason), int(count)) for reason, count in exit_counts.items()),
                        key=lambda part: (-part[1], part[0]),
                    )
                ),
                "loss_closes": int(item.get("loss_closes") or 0),
                "actual_pl_jpy": round(float(item.get("actual_pl_jpy") or 0.0), 4),
                "repair_replay_pl_jpy": round(
                    float(item.get("repair_replay_pl_jpy") or 0.0),
                    4,
                ),
                "repair_replay_delta_jpy": round(
                    float(item.get("repair_replay_delta_jpy") or 0.0),
                    4,
                ),
                "repair_replay_triggered": int(item.get("repair_replay_triggered") or 0),
                "block_reasons": dict(
                    sorted(
                        (
                            (str(reason), int(count))
                            for reason, count in (
                                item.get("block_reasons")
                                if isinstance(item.get("block_reasons"), dict)
                                else {}
                            ).items()
                        ),
                        key=lambda part: (-part[1], part[0]),
                    )
                ),
                "examples": item.get("examples") if isinstance(item.get("examples"), list) else [],
            }
        )
    rows.sort(
        key=lambda item: (
            float(item.get("repair_replay_pl_jpy") or 0.0),
            -int(item.get("pair_count") or 0),
            str(item.get("residual_scope") or ""),
            str(item.get("method") or ""),
        )
    )
    return rows[:10]


def _profit_capture_replay_repair_findings(
    timing_metrics: dict[str, Any],
    *,
    self_metrics: dict[str, Any] | None = None,
    capture_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    self_metrics = self_metrics or {}
    capture_metrics = capture_metrics or {}
    self_p0_codes = {
        str(code or "").strip().upper()
        for code in self_metrics.get("p0_codes", []) or []
        if str(code or "").strip()
    }
    has_self_profit_capture_context = any(
        "PROFIT_CAPTURE" in code or code == "LOSS_CLOSE_PROFIT_CAPTURE_MISSED"
        for code in self_p0_codes
    )
    guardian_profit_capture_inactive = (
        "POSITION_GUARDIAN_INACTIVE_FOR_PROFIT_CAPTURE" in self_p0_codes
        or "POSITION_GUARDIAN_INACTIVE" in self_p0_codes
    )
    simple_missed = int(_optional_float(timing_metrics.get("loss_closes_profit_capture_missed")) or 0)
    repair_replay_missed = int(
        _optional_float(timing_metrics.get("loss_closes_repair_replay_triggered")) or 0
    )
    repair_replay_contract_present = bool(timing_metrics.get("repair_replay_contract_present"))
    repair_replay_contract_missing = (
        bool(timing_metrics.get("loaded")) and not repair_replay_contract_present
    )
    missed = repair_replay_missed
    counterfactual_delta = _optional_float(
        timing_metrics.get("loss_close_repair_replay_delta_jpy")
    )
    if counterfactual_delta is None:
        counterfactual_delta = _optional_float(
            timing_metrics.get("loss_close_counterfactual_profit_capture_delta_jpy")
        )
    counterfactual_jpy = _optional_float(
        timing_metrics.get("loss_close_repair_replay_profit_capture_jpy")
    )
    if counterfactual_jpy is None:
        counterfactual_jpy = _optional_float(
            timing_metrics.get("loss_close_counterfactual_profit_capture_jpy")
        )
    repair_counterfactual_pl = _optional_float(
        timing_metrics.get("loss_close_repair_replay_counterfactual_pl_jpy")
    )
    raw_counterfactual_pl = _optional_float(
        timing_metrics.get("loss_close_counterfactual_profit_capture_pl_jpy")
    )
    active_counterfactual_pl = (
        repair_counterfactual_pl
        if repair_counterfactual_pl is not None
        else raw_counterfactual_pl
    )
    window_lookback_hours = _optional_float(timing_metrics.get("window_lookback_hours"))
    tp_metrics = (
        capture_metrics.get("take_profit")
        if isinstance(capture_metrics.get("take_profit"), dict)
        else {}
    )
    market_close_metrics = (
        capture_metrics.get("market_close")
        if isinstance(capture_metrics.get("market_close"), dict)
        else {}
    )
    tp_net = _optional_float(tp_metrics.get("net_jpy"))
    market_close_net = _optional_float(market_close_metrics.get("net_jpy"))
    monthly_replay_required = bool(
        tp_net is not None
        and tp_net > 0.0
        and market_close_net is not None
        and market_close_net < 0.0
    )
    month_scale_replay_loaded = bool(
        timing_metrics.get("loaded")
        and repair_replay_contract_present
        and window_lookback_hours is not None
        and window_lookback_hours >= MONTH_SCALE_LOSS_CLOSE_REPLAY_MIN_HOURS
    )
    metrics = {
        "execution_timing_loaded": bool(timing_metrics.get("loaded")),
        "execution_timing_generated_at_utc": timing_metrics.get("generated_at_utc"),
        "execution_timing_window_from_utc": timing_metrics.get("window_from_utc"),
        "execution_timing_window_to_utc": timing_metrics.get("window_to_utc"),
        "execution_timing_window_lookback_hours": window_lookback_hours,
        "month_scale_replay_required": monthly_replay_required,
        "month_scale_replay_loaded": month_scale_replay_loaded,
        "month_scale_replay_min_hours": MONTH_SCALE_LOSS_CLOSE_REPLAY_MIN_HOURS,
        "capture_take_profit_net_jpy": tp_net,
        "capture_market_close_net_jpy": market_close_net,
        "loss_closes_profit_capture_missed": simple_missed,
        "loss_closes_repair_replay_triggered": repair_replay_missed,
        "repair_replay_contract": timing_metrics.get("repair_replay_contract"),
        "repair_replay_contract_present": repair_replay_contract_present,
        "repair_replay_counterfactual_pl_jpy": repair_counterfactual_pl,
        "raw_counterfactual_profit_capture_pl_jpy": raw_counterfactual_pl,
        "active_counterfactual_profit_capture_pl_jpy": active_counterfactual_pl,
        "counterfactual_profit_capture_delta_jpy": counterfactual_delta,
        "counterfactual_profit_capture_jpy": counterfactual_jpy,
        "top_profit_capture_misses": timing_metrics.get("top_profit_capture_misses") or [],
        "top_repair_replay_triggers": timing_metrics.get("top_repair_replay_triggers") or [],
        "top_repair_replay_residual_groups": (
            timing_metrics.get("top_repair_replay_residual_groups") or []
        ),
        "top_tp_progress_repair_residual_groups": (
            timing_metrics.get("top_tp_progress_repair_residual_groups") or []
        ),
        "top_entry_quality_residual_groups": (
            timing_metrics.get("top_entry_quality_residual_groups") or []
        ),
        "top_repair_replay_residual_method_rollups": (
            timing_metrics.get("top_repair_replay_residual_method_rollups") or []
        ),
        "top_tp_progress_repair_residual_method_rollups": (
            timing_metrics.get("top_tp_progress_repair_residual_method_rollups") or []
        ),
        "top_entry_quality_residual_method_rollups": (
            timing_metrics.get("top_entry_quality_residual_method_rollups") or []
        ),
        "loss_close_repair_replay_block_reasons": (
            timing_metrics.get("loss_close_repair_replay_block_reasons") or {}
        ),
        "top_repair_replay_blocks": timing_metrics.get("top_repair_replay_blocks") or [],
        "self_improvement_profit_capture_context": has_self_profit_capture_context,
        "guardian_profit_capture_inactive": guardian_profit_capture_inactive,
        "self_improvement_p0_codes": sorted(self_p0_codes),
        "clearance_condition": (
            "execution-timing-audit must report zero loss_closes_repair_replay_triggered "
            "with the current production-gate replay contract after TP-progress "
            "TAKE_PROFIT_MARKET / guardian repair has run on live broker truth; raw "
            "loss_closes_profit_capture_missed remains diagnostic unless the production gate "
            "also proves an executable profit capture"
        ),
        "replay_repair_proved": (
            bool(timing_metrics.get("loaded"))
            and repair_replay_contract_present
            and repair_replay_missed == 0
            and (
                not monthly_replay_required
                or (
                    month_scale_replay_loaded
                    and (
                        active_counterfactual_pl is None
                        or active_counterfactual_pl >= 0.0
                    )
                )
            )
        ),
    }
    findings: list[dict[str, Any]] = []
    if monthly_replay_required and not month_scale_replay_loaded:
        findings.append(
            _finding(
                priority="P0",
                code="MONTH_SCALE_LOSS_CLOSE_REPLAY_REQUIRED",
                message=(
                    "TAKE_PROFIT_ORDER is net-positive while MARKET_ORDER_TRADE_CLOSE is net-negative, "
                    "but the active execution-timing-audit does not cover a 30-day OANDA candle "
                    "replay with the current TP-progress production gate"
                ),
                next_action=(
                    "Run execution-timing-audit with at least 720 lookback hours and the current "
                    "production-gate replay contract, then use that artifact for profitability "
                    "acceptance before claiming the market-close leak is repaired."
                ),
                evidence={
                    "take_profit_net_jpy": tp_net,
                    "market_close_net_jpy": market_close_net,
                    "execution_timing_loaded": metrics["execution_timing_loaded"],
                    "window_lookback_hours": window_lookback_hours,
                    "required_lookback_hours": MONTH_SCALE_LOSS_CLOSE_REPLAY_MIN_HOURS,
                    "repair_replay_contract": metrics["repair_replay_contract"],
                    "repair_replay_contract_present": repair_replay_contract_present,
                },
            )
        )
    elif (
        monthly_replay_required
        and month_scale_replay_loaded
        and active_counterfactual_pl is not None
        and active_counterfactual_pl < 0.0
    ):
        findings.append(
            _finding(
                priority="P0",
                code="MONTH_SCALE_TP_PROGRESS_REPLAY_STILL_NEGATIVE",
                message=(
                    "30-day OANDA candle replay says the current TP-progress repair improves "
                    "loss-side closes, but the replayed loss-close P/L is still net negative"
                ),
                next_action=(
                    "Do not increase high-turnover entries from this repair alone. Split the "
                    "remaining residuals into TP-progress replay residuals and entry-quality "
                    "residuals, then improve the matching close-gate evidence, entry selection, "
                    "or exit geometry until month-scale production-gate replay is non-negative."
                ),
                evidence={
                    "window_lookback_hours": window_lookback_hours,
                    "loss_closes_profit_capture_missed": simple_missed,
                    "loss_closes_repair_replay_triggered": repair_replay_missed,
                    "repair_replay_counterfactual_pl_jpy": repair_counterfactual_pl,
                    "raw_counterfactual_profit_capture_pl_jpy": raw_counterfactual_pl,
                    "active_counterfactual_profit_capture_pl_jpy": active_counterfactual_pl,
                    "counterfactual_profit_capture_delta_jpy": counterfactual_delta,
                    "counterfactual_profit_capture_jpy": counterfactual_jpy,
                    "top_repair_replay_triggers": metrics["top_repair_replay_triggers"],
                    "top_repair_replay_blocks": metrics["top_repair_replay_blocks"],
                    "top_repair_replay_residual_groups": metrics[
                        "top_repair_replay_residual_groups"
                    ],
                    "top_tp_progress_repair_residual_groups": metrics[
                        "top_tp_progress_repair_residual_groups"
                    ],
                    "top_entry_quality_residual_groups": metrics[
                        "top_entry_quality_residual_groups"
                    ],
                    "top_repair_replay_residual_method_rollups": metrics[
                        "top_repair_replay_residual_method_rollups"
                    ],
                    "top_tp_progress_repair_residual_method_rollups": metrics[
                        "top_tp_progress_repair_residual_method_rollups"
                    ],
                    "top_entry_quality_residual_method_rollups": metrics[
                        "top_entry_quality_residual_method_rollups"
                    ],
                },
            )
        )
    if (
        repair_replay_contract_missing
        and simple_missed > 0
        and has_self_profit_capture_context
    ):
        return metrics, findings + [
            _finding(
                priority="P0",
                code="TP_PROGRESS_REPAIR_REPLAY_CONTRACT_MISSING",
                message=(
                    f"{simple_missed} loss close(s) have TP-progress miss evidence, but the active "
                    "execution-timing-audit sidecar was generated before production-gate replay "
                    "validation and cannot prove the repair"
                ),
                next_action=(
                    "Regenerate execution-timing-audit with the current runtime until the sidecar "
                    f"contains {TP_PROGRESS_REPAIR_REPLAY_CONTRACT}, then evaluate "
                    "loss_closes_repair_replay_triggered and loss_closes_profit_capture_missed."
                ),
                evidence={
                    "loss_closes_profit_capture_missed": simple_missed,
                    "loss_closes_repair_replay_triggered": repair_replay_missed,
                    "repair_replay_contract": metrics["repair_replay_contract"],
                    "required_repair_replay_contract": TP_PROGRESS_REPAIR_REPLAY_CONTRACT,
                    "counterfactual_profit_capture_delta_jpy": counterfactual_delta,
                    "counterfactual_profit_capture_jpy": counterfactual_jpy,
                    "top_profit_capture_misses": metrics["top_profit_capture_misses"],
                    "loss_close_repair_replay_block_reasons": metrics[
                        "loss_close_repair_replay_block_reasons"
                    ],
                    "top_repair_replay_blocks": metrics["top_repair_replay_blocks"],
                    "clearance_condition": metrics["clearance_condition"],
                },
            )
        ]
    if not metrics["execution_timing_loaded"] or not has_self_profit_capture_context:
        return metrics, findings
    if repair_replay_contract_present and missed <= 0:
        return metrics, findings
    if guardian_profit_capture_inactive and repair_replay_contract_present and missed > 0:
        findings.append(
            _finding(
                priority="P0",
                code="TP_PROGRESS_REPAIR_REPLAY_NOT_DEPLOYED",
                message=(
                    f"{missed} loss close(s) replay through the current TP-progress production "
                    "gate, but position guardian is not proven active, so the repaired logic is "
                    "not actually being rerun between full trader cycles"
                ),
                next_action=(
                    "Do not add turnover by merely rerunning reports. First prove guardian "
                    "preflight/status green and load it only with explicit operator approval; "
                    "then give the repaired TP-progress path a live window and rerun "
                    "execution-timing-audit until loss_closes_repair_replay_triggered is zero."
                ),
                evidence={
                    "loss_closes_repair_replay_triggered": repair_replay_missed,
                    "loss_closes_profit_capture_missed": simple_missed,
                    "repair_replay_contract": metrics["repair_replay_contract"],
                    "guardian_profit_capture_inactive": True,
                    "self_improvement_p0_codes": metrics["self_improvement_p0_codes"],
                    "top_repair_replay_triggers": metrics["top_repair_replay_triggers"],
                    "clearance_condition": metrics["clearance_condition"],
                },
            )
        )
    findings.append(
        _finding(
            priority="P0",
            code="TP_PROGRESS_REPLAY_REPAIR_UNPROVED",
            message=(
                f"{missed} loss close(s) have OANDA candle replay evidence that TP-progress "
                "profit capture would have improved realized P/L, but the active audit has not "
                "yet proved the repair clean"
            ),
            next_action=(
                "Do not treat high-turnover trading as repaired by rerunning reports. Keep the "
                "TP-progress TAKE_PROFIT_MARKET path and position guardian active, then rerun "
                "execution-timing-audit until loss_closes_repair_replay_triggered is zero under "
                "the current production-gate replay contract. Raw TP-progress misses without "
                "production-gate triggers stay diagnostic until tick replay upgrades them."
            ),
            evidence={
                "loss_closes_profit_capture_missed": simple_missed,
                "loss_closes_repair_replay_triggered": repair_replay_missed,
                "counterfactual_profit_capture_delta_jpy": counterfactual_delta,
                "counterfactual_profit_capture_jpy": counterfactual_jpy,
                "top_profit_capture_misses": metrics["top_profit_capture_misses"],
                "top_repair_replay_triggers": metrics["top_repair_replay_triggers"],
                "loss_close_repair_replay_block_reasons": metrics[
                    "loss_close_repair_replay_block_reasons"
                ],
                "top_repair_replay_blocks": metrics["top_repair_replay_blocks"],
                "clearance_condition": metrics["clearance_condition"],
            },
        )
    )
    return metrics, findings


def _loss_close_timing_label(
    row: dict[str, Any],
    labels: dict[tuple[str, str], str],
) -> str | None:
    for key_name in ("trade_id", "order_id"):
        value = str(row.get(key_name) or "").strip()
        if not value:
            continue
        label = labels.get((key_name, value))
        if label:
            return label
    return None


def _loss_close_by_lane(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_lane: dict[str, dict[str, Any]] = {}
    for row in rows:
        lane_id = str(row.get("lane_id") or "").strip()
        key = lane_id or f"UNKNOWN:{row.get('pair') or ''}:{row.get('side') or ''}"
        item = by_lane.setdefault(
            key,
            {
                "lane_id": lane_id or None,
                "pair": row.get("pair"),
                "side": row.get("side"),
                "method": _method_from_lane_id(lane_id),
                "loss_closes": 0,
                "net_jpy": 0.0,
            },
        )
        item["loss_closes"] = int(item.get("loss_closes") or 0) + 1
        item["net_jpy"] = float(item.get("net_jpy") or 0.0) + float(
            row.get("realized_pl_jpy") or 0.0
        )
    return [
        {**item, "net_jpy": round(float(item.get("net_jpy") or 0.0), 4)}
        for item in sorted(
            by_lane.values(),
            key=lambda item: (
                float(item.get("net_jpy") or 0.0),
                str(item.get("lane_id") or ""),
            ),
        )[:10]
    ]


def _method_from_lane_id(lane_id: str | None) -> str | None:
    parts = [part.strip() for part in str(lane_id or "").split(":") if part.strip()]
    if len(parts) >= 4:
        return parts[3]
    return None


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        payload = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
    edge_rules = [
        item for item in payload.get("edge_rules") or [] if isinstance(item, dict)
    ]
    contrarian_rules = [
        item for item in payload.get("contrarian_edge_rules") or [] if isinstance(item, dict)
    ]
    negative_rules = [
        item for item in payload.get("negative_rules") or [] if isinstance(item, dict)
    ]
    daily_stable_edges = [item for item in edge_rules if _bidask_rule_is_live_grade(item)]
    daily_stable_contrarian = [
        item for item in contrarian_rules if _bidask_rule_is_live_grade(item)
    ]
    rank_only_edges = [item for item in edge_rules if not _bidask_rule_is_live_grade(item)]
    rank_only_contrarian = [
        item for item in contrarian_rules if not _bidask_rule_is_live_grade(item)
    ]
    support_rules = [*edge_rules, *contrarian_rules]
    daily_stable_support = [*daily_stable_edges, *daily_stable_contrarian]
    rank_only = [*rank_only_edges, *rank_only_contrarian]
    rank_only.sort(
        key=lambda item: (
            float(_optional_float(item.get("optimized_profit_factor")) or 0.0),
            int(_optional_float(item.get("samples")) or 0),
        ),
        reverse=True,
    )
    rank_only_examples = [_bidask_rank_only_example(item) for item in rank_only[:5]]
    fetch_command, validation_command = _bidask_replay_verification_commands(rank_only, payload)
    adoption = payload.get("adoption_summary") if isinstance(payload.get("adoption_summary"), dict) else {}
    truth = (
        payload.get("price_truth_coverage")
        if isinstance(payload.get("price_truth_coverage"), dict)
        else {}
    )
    truth_status = str(truth.get("status") or "").upper()
    metrics = {
        "path": str(path),
        "edge_rules": len(edge_rules),
        "daily_stable_edge_rules": len(daily_stable_edges),
        "rank_only_edge_rules": len(rank_only_edges),
        "contrarian_edge_rules": len(contrarian_rules),
        "daily_stable_contrarian_edge_rules": len(daily_stable_contrarian),
        "rank_only_contrarian_edge_rules": len(rank_only_contrarian),
        "support_rules": len(support_rules),
        "daily_stable_support_rules": len(daily_stable_support),
        "rank_only_support_rules": len(rank_only),
        "negative_rules": len(negative_rules),
        "adoption_summary": adoption,
        "price_truth_coverage": {
            "status": truth.get("status"),
            "adoption_level": truth.get("adoption_level"),
            "evaluated_rows": truth.get("evaluated_rows"),
            "missing_price_truth_samples": truth.get("missing_price_truth_samples"),
            "missing_price_window_group_count": truth.get("missing_price_window_group_count"),
            "history_fetch_command": truth.get("history_fetch_command"),
            "history_fetch_command_count": truth.get("history_fetch_command_count"),
            "history_fetch_command_mode": truth.get("history_fetch_command_mode"),
            "missing_pairs": truth.get("missing_pairs"),
            "missing_pair_directions": truth.get("missing_pair_directions"),
            "all_currency_sample_coverage_status": truth.get("all_currency_sample_coverage_status"),
            "under_sampled_pair_direction_count": truth.get("under_sampled_pair_direction_count"),
            "under_sampled_pair_directions": truth.get("under_sampled_pair_directions"),
            "under_sampled_missing_evaluated_samples": truth.get(
                "under_sampled_missing_evaluated_samples"
            ),
            "global_currency_validation_blocked": truth.get("global_currency_validation_blocked"),
            "warnings": truth.get("warnings"),
        },
        "daily_stability_requirements": {
            "min_active_days": BIDASK_REPLAY_STABLE_MIN_ACTIVE_DAYS,
            "max_daily_sample_share": BIDASK_REPLAY_STABLE_MAX_DAILY_SAMPLE_SHARE,
            "min_positive_day_rate": BIDASK_REPLAY_STABLE_MIN_POSITIVE_DAY_RATE,
        },
        "rank_only_examples": rank_only_examples,
        "history_dirs": _bidask_history_dirs(payload),
        "history_fetch_command": fetch_command,
        "replay_validation_command": validation_command,
    }
    findings: list[dict[str, Any]] = []
    if truth_status and truth_status != "PRICE_TRUTH_OK":
        findings.append(
            _finding(
                priority="P1",
                code="BIDASK_REPLAY_PRICE_TRUTH_PARTIAL",
                message=(
                    "S5 bid/ask replay rules still have partial OANDA price-truth coverage; "
                    "high-turn firepower remains reproducible only for the locally covered "
                    "samples until the missing windows are fetched and validation is rerun"
                ),
                next_action=(
                    "Run the published OANDA read-only history fetch command(s), rerun "
                    "oanda_history_replay_validate, then package the refreshed bid/ask rules."
                ),
                evidence=metrics,
            )
        )
    elif bool(truth.get("global_currency_validation_blocked")):
        findings.append(
            _finding(
                priority="P1",
                code="BIDASK_REPLAY_ALL_CURRENCY_SAMPLE_COVERAGE_THIN",
                message=(
                    "S5 bid/ask replay has price truth for loaded samples, but pair-direction "
                    "sample coverage is too thin to claim all-currency high-turn readiness"
                ),
                next_action=(
                    "Collect more live forecast samples or replay-covered forecast history across the "
                    "under-sampled pair-directions, then rerun oanda_history_replay_validate and package "
                    "the refreshed rules."
                ),
                evidence=metrics,
            )
        )
    if rank_only and not daily_stable_support:
        findings.append(
            _finding(
                priority="P1",
                code="BIDASK_REPLAY_SUPPORT_NOT_DAILY_STABLE",
                message=(
                    f"{len(rank_only)} S5 bid/ask replay support rule(s) exist, but none are "
                    "daily-stable enough for live-grade high-turn firepower"
                ),
                next_action=(
                    "Keep bid/ask replay support as rank-only until multi-day stability, positive-day "
                    "rate, sample distribution, and execution geometry clear the live thresholds."
                ),
                evidence=metrics,
            )
        )
    elif rank_only_contrarian and not daily_stable_contrarian:
        findings.append(
            _finding(
                priority="P1",
                code="BIDASK_CONTRARIAN_EDGE_NOT_DAILY_STABLE",
                message=(
                    f"{len(rank_only_contrarian)} S5 contrarian replay edge(s) exist, but none are "
                    "daily-stable enough for live-grade inversion"
                ),
                next_action=(
                    "Keep weak forecast inversion as rank-only until multi-day stability, positive-day "
                    "rate, and execution geometry clear the live thresholds."
                ),
                evidence=metrics,
            )
        )
    return metrics, findings


def _bidask_rule_is_live_grade(item: dict[str, Any]) -> bool:
    if bool(item.get("live_grade")):
        return True
    if str(item.get("adoption_status") or "").upper() == "LIVE_GRADE_DAILY_STABLE":
        return True
    return str(item.get("daily_stability_status") or "").upper() == "DAILY_STABLE"


def _bidask_rank_only_example(item: dict[str, Any]) -> dict[str, Any]:
    gap = item.get("daily_stability_gap")
    if not isinstance(gap, dict):
        gap = _bidask_daily_stability_gap(item)
    return {
        "name": item.get("name"),
        "pair": item.get("pair"),
        "granularity": item.get("granularity"),
        "forecast_direction": item.get("forecast_direction") or item.get("faded_direction"),
        "direction": item.get("direction"),
        "samples": item.get("samples"),
        "active_days": item.get("active_days"),
        "positive_days": item.get("positive_days"),
        "positive_day_rate": item.get("positive_day_rate"),
        "max_daily_sample_share": item.get("max_daily_sample_share"),
        "daily_stability_status": item.get("daily_stability_status"),
        "adoption_status": item.get("adoption_status"),
        "adoption_blockers": item.get("adoption_blockers"),
        "optimized_profit_factor": item.get("optimized_profit_factor"),
        "optimized_win_rate": item.get("optimized_win_rate"),
        "optimized_take_profit_pips": item.get("optimized_take_profit_pips"),
        "optimized_stop_loss_pips": item.get("optimized_stop_loss_pips"),
        "daily_stability_gap": gap,
    }


def _bidask_daily_stability_gap(item: dict[str, Any]) -> dict[str, Any]:
    active_days = int(item.get("active_days") or 0)
    positive_day_rate = _optional_float(item.get("positive_day_rate")) or 0.0
    positive_days = item.get("positive_days")
    if positive_days is None:
        positive_days = int(round(active_days * positive_day_rate))
    positive_days = int(positive_days or 0)
    required_active_days = max(active_days, BIDASK_REPLAY_STABLE_MIN_ACTIVE_DAYS)
    required_positive_days = math.ceil(
        BIDASK_REPLAY_STABLE_MIN_POSITIVE_DAY_RATE * required_active_days
    )
    max_daily_sample_share = _optional_float(item.get("max_daily_sample_share"))
    reasons: list[str] = []
    missing_active_days = max(0, BIDASK_REPLAY_STABLE_MIN_ACTIVE_DAYS - active_days)
    missing_positive_days = max(0, required_positive_days - positive_days)
    if missing_active_days:
        reasons.append("NEEDS_MORE_ACTIVE_DAYS")
    if max_daily_sample_share is not None and max_daily_sample_share > BIDASK_REPLAY_STABLE_MAX_DAILY_SAMPLE_SHARE:
        reasons.append("NEEDS_LESS_DAILY_SAMPLE_CONCENTRATION")
    if positive_day_rate < BIDASK_REPLAY_STABLE_MIN_POSITIVE_DAY_RATE:
        reasons.append("NEEDS_DAILY_STABILITY_CONFIRMATION")
    return {
        "reasons": reasons,
        "missing_active_days": missing_active_days,
        "missing_positive_days_at_current_requirement": missing_positive_days,
        "required_active_days": BIDASK_REPLAY_STABLE_MIN_ACTIVE_DAYS,
        "required_positive_days_at_current_requirement": required_positive_days,
        "required_positive_day_rate": BIDASK_REPLAY_STABLE_MIN_POSITIVE_DAY_RATE,
        "max_allowed_daily_sample_share": BIDASK_REPLAY_STABLE_MAX_DAILY_SAMPLE_SHARE,
    }


def _bidask_history_dirs(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("history_dirs")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _bidask_replay_verification_commands(
    rank_only: list[dict[str, Any]],
    payload: dict[str, Any],
) -> tuple[str | None, str | None]:
    truth = (
        payload.get("price_truth_coverage")
        if isinstance(payload.get("price_truth_coverage"), dict)
        else {}
    )
    pairs = sorted({str(item.get("pair") or "").upper() for item in rank_only if item.get("pair")})
    granularities = sorted({str(item.get("granularity") or "S5").upper() for item in rank_only})
    if not granularities:
        granularity = str(payload.get("granularity") or "S5").upper()
        granularities = [granularity] if granularity else ["S5"]
    granularities_arg = ",".join(granularities)
    primary_granularity = granularities[0] if granularities else "S5"
    fetch_command = (
        str(truth.get("history_fetch_command"))
        if str(truth.get("history_fetch_command") or "").strip()
        else None
    )
    if pairs:
        pairs_arg = ",".join(pairs)
        fetch_command = fetch_command or (
            "python3 scripts/oanda_history_fetch.py "
            f"--pairs {pairs_arg} --granularities {granularities_arg} --price BA "
            f"--days {BIDASK_REPLAY_HISTORY_FETCH_DAYS} --output-dir logs/replay/oanda_history"
        )
    validation_command = (
        "python3 scripts/oanda_history_replay_validate.py "
        "--forecast-history data/forecast_history.jsonl "
        f"--granularity {primary_granularity} "
        f"{_bidask_history_dir_args(payload)}"
        f"--auto-history-min-days {BIDASK_REPLAY_AUTO_HISTORY_MIN_DAYS} "
        f"--stable-min-active-days {BIDASK_REPLAY_STABLE_MIN_ACTIVE_DAYS} "
        f"--stable-max-daily-sample-share {BIDASK_REPLAY_STABLE_MAX_DAILY_SAMPLE_SHARE} "
        f"--stable-min-positive-day-rate {BIDASK_REPLAY_STABLE_MIN_POSITIVE_DAY_RATE:.10f}"
    )
    return fetch_command, validation_command


def _bidask_history_dir_args(payload: dict[str, Any]) -> str:
    history_dirs = _bidask_history_dirs(payload)
    if not history_dirs:
        return ""
    return "".join(f"--history-dir {shlex.quote(path)} " for path in history_dirs)


_OANDA_FIREPOWER_TARGET_OK_STATUSES = {
    "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED",
    "VERIFIED_TARGET_10_ROUTE_ESTIMATED",
}


def _oanda_campaign_firepower_findings(
    payload: dict[str, Any],
    path: Path,
    *,
    target_open: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    firepower = payload.get("campaign_firepower") if isinstance(payload.get("campaign_firepower"), dict) else {}
    high_precision = firepower.get("high_precision") if isinstance(firepower.get("high_precision"), dict) else {}
    evidence_queue = firepower.get("evidence_queue") if isinstance(firepower.get("evidence_queue"), dict) else {}
    status = str(firepower.get("status") or "").strip().upper()
    metrics = {
        "path": str(path),
        "report_exists": path.exists(),
        "target_open": target_open,
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": status or None,
        "contract": firepower.get("contract"),
        "minimum_return_pct": _optional_float(firepower.get("minimum_return_pct")),
        "target_return_pct": _optional_float(firepower.get("target_return_pct")),
        "per_trade_risk_pct_lens": _optional_float(firepower.get("per_trade_risk_pct_lens")),
        "high_precision": _oanda_firepower_section_metrics(high_precision),
        "evidence_queue": _oanda_firepower_section_metrics(evidence_queue),
    }
    if not target_open:
        return metrics, []

    next_action = (
        "Rerun scripts/oanda_universal_rotation_miner.py with current multi-month OANDA candles. "
        "Keep universal rotation as rank-only until campaign_firepower.status reaches "
        "VERIFIED_MINIMUM_5_ROUTE_ESTIMATED or VERIFIED_TARGET_10_ROUTE_ESTIMATED."
    )
    if not path.exists():
        return metrics, [
            _finding(
                priority="P0",
                code="OANDA_CAMPAIGN_FIREPOWER_REPORT_MISSING",
                message="OANDA universal rotation mining report is missing while the daily target is open",
                next_action=next_action,
                evidence=metrics,
            )
        ]
    if not firepower:
        return metrics, [
            _finding(
                priority="P0",
                code="OANDA_CAMPAIGN_FIREPOWER_NOT_COMPUTED",
                message=(
                    "OANDA universal rotation mining report lacks campaign_firepower; "
                    "daily 5-10% high-turn firepower is unproven"
                ),
                next_action=next_action,
                evidence=metrics,
            )
        ]
    if status in _OANDA_FIREPOWER_TARGET_OK_STATUSES:
        return metrics, []
    if status in {"EVIDENCE_QUEUE_ONLY_NO_VERIFIED_FIREPOWER", "NO_VERIFIED_FIREPOWER"}:
        return metrics, [
            _finding(
                priority="P0",
                code="OANDA_CAMPAIGN_FIREPOWER_UNVERIFIED",
                message=(
                    "OANDA campaign firepower has no high-precision validated vehicle; "
                    "evidence-queue return estimates cannot support high-turn scaling"
                ),
                next_action=next_action,
                evidence=metrics,
            )
        ]
    if status == "VERIFIED_EDGE_BUT_DAILY_TARGET_SHORTFALL":
        return metrics, [
            _finding(
                priority="P0",
                code="OANDA_CAMPAIGN_FIREPOWER_DAILY_TARGET_SHORTFALL",
                message=(
                    "validated OANDA campaign firepower does not reach the daily 5% floor "
                    "at observed frequency"
                ),
                next_action=next_action,
                evidence=metrics,
            )
        ]
    return metrics, [
        _finding(
            priority="P0",
            code="OANDA_CAMPAIGN_FIREPOWER_STATUS_UNKNOWN",
            message=f"OANDA campaign firepower status is not recognized: {status or 'MISSING'}",
            next_action=next_action,
            evidence=metrics,
        )
    ]


def _oanda_firepower_section_metrics(section: dict[str, Any]) -> dict[str, Any]:
    vehicles = section.get("top_vehicles") if isinstance(section.get("top_vehicles"), list) else []
    return {
        "unique_vehicle_count": int(_optional_float(section.get("unique_vehicle_count")) or 0),
        "pair_count": int(_optional_float(section.get("pair_count")) or 0),
        "observed_attempts_per_active_day": _optional_float(
            section.get("observed_attempts_per_active_day")
        ),
        "estimated_return_pct_per_active_day_at_observed_frequency": _optional_float(
            section.get("estimated_return_pct_per_active_day_at_observed_frequency")
        ),
        "weighted_return_pct_per_trade_at_risk_lens": _optional_float(
            section.get("weighted_return_pct_per_trade_at_risk_lens")
        ),
        "trades_needed_for_minimum_5pct_at_weighted_expectancy": _optional_float(
            section.get("trades_needed_for_minimum_5pct_at_weighted_expectancy")
        ),
        "trades_needed_for_target_10pct_at_weighted_expectancy": _optional_float(
            section.get("trades_needed_for_target_10pct_at_weighted_expectancy")
        ),
        "top_vehicle_keys": [
            str(item.get("vehicle_key") or "")
            for item in vehicles[:3]
            if isinstance(item, dict) and item.get("vehicle_key")
        ],
    }


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
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
