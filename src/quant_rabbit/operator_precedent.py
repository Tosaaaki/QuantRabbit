"""Audit the operator's 2025 manual success as machine-readable precedent.

This module deliberately does not grant live permission. It turns the OANDA
read-only manual-history mining artifact into a compact audit packet that the
trader can cite when choosing among already-valid lanes. RiskEngine,
LiveOrderGateway, forecast gates, spread gates, and current broker truth remain
the executable authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from quant_rabbit.paths import (
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_MANUAL_HISTORY_2025,
    DEFAULT_OPERATOR_PRECEDENT_AUDIT,
    DEFAULT_OPERATOR_PRECEDENT_AUDIT_REPORT,
    DEFAULT_ORDER_INTENTS,
)


# This is the operator claim being audited: "1 month / 200%+". It is a proof
# threshold for the historical evidence packet, not a trading target, risk cap,
# lane score, or live gate.
OPERATOR_CLAIM_30D_RETURN_PCT = 200.0

# The manual edge was session-concentrated. Keep the two highest-net positive
# sessions as the advisory match surface so a small positive off-hour bucket
# does not dilute the London/NY operating pattern. This is not a risk limit.
PRIMARY_PRECEDENT_SESSION_COUNT = 2


@dataclass(frozen=True)
class OperatorPrecedentSummary:
    output_path: Path
    report_path: Path
    status: str
    checks: int
    blockers: int
    warnings: int
    best_30d_return_pct: float | None
    live_ready_lanes: int
    aligned_live_ready_lanes: int


def build_operator_precedent_audit(
    *,
    manual_history_path: Path = DEFAULT_MANUAL_HISTORY_2025,
    order_intents_path: Path = DEFAULT_ORDER_INTENTS,
    target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
    output_path: Path = DEFAULT_OPERATOR_PRECEDENT_AUDIT,
    report_path: Path = DEFAULT_OPERATOR_PRECEDENT_AUDIT_REPORT,
    now: datetime | None = None,
) -> OperatorPrecedentSummary:
    clock = now or datetime.now(timezone.utc)
    generated_at = clock.isoformat()
    checks: list[dict[str, Any]] = []
    manual_payload, manual_error = _read_json(manual_history_path)
    intents_payload, intents_error = _read_json(order_intents_path)
    target_payload, target_error = _read_json(target_state_path)

    checks.append(
        _check(
            "manual_history_readable",
            "PASS" if manual_payload is not None else "BLOCK",
            f"manual history artifact readable: {manual_history_path}"
            if manual_payload is not None
            else f"manual history artifact missing/unreadable: {manual_error or manual_history_path}",
            {"path": str(manual_history_path), "error": manual_error},
        )
    )
    checks.append(
        _check(
            "order_intents_readable",
            "PASS" if intents_payload is not None else "WARN",
            f"order intents readable: {order_intents_path}"
            if intents_payload is not None
            else f"order intents missing/unreadable; current alignment unknown: {intents_error or order_intents_path}",
            {"path": str(order_intents_path), "error": intents_error},
            severity="INFO" if intents_payload is not None else "WARN",
        )
    )
    checks.append(
        _check(
            "target_state_readable",
            "PASS" if target_payload is not None else "WARN",
            f"daily target state readable: {target_state_path}"
            if target_payload is not None
            else f"daily target state missing/unreadable; pace comparison unknown: {target_error or target_state_path}",
            {"path": str(target_state_path), "error": target_error},
            severity="INFO" if target_payload is not None else "WARN",
        )
    )

    precedent = _extract_precedent(manual_payload or {})
    checks.extend(_precedent_checks(precedent))
    runtime = _runtime_alignment(
        intents_payload or {},
        target_payload or {},
        precedent=precedent,
    )
    if runtime["live_ready_lanes"] == 0:
        checks.append(
            _check(
                "current_live_ready_alignment",
                "WARN",
                "no LIVE_READY lanes are present, so the manual precedent cannot be expressed in the current basket",
                {"live_ready_lanes": 0},
                severity="WARN",
            )
        )
    elif runtime["aligned_live_ready_lanes"] == 0:
        checks.append(
            _check(
                "current_live_ready_alignment",
                "WARN",
                "LIVE_READY lanes exist, but none match the manual precedent's pair/direction/session shape",
                {
                    "live_ready_lanes": runtime["live_ready_lanes"],
                    "precedent_shape": precedent.get("winning_shape"),
                },
                severity="WARN",
            )
        )
    else:
        checks.append(
            _check(
                "current_live_ready_alignment",
                "PASS",
                "at least one current LIVE_READY lane matches the manual precedent shape",
                {"aligned_live_ready_lanes": runtime["aligned_live_ready_lanes"]},
            )
        )

    blockers = [item for item in checks if item["severity"] == "BLOCK" or item["status"] == "BLOCK"]
    warnings = [item for item in checks if item["severity"] == "WARN" or item["status"] == "WARN"]
    status = "OPERATOR_PRECEDENT_BLOCKED" if blockers else (
        "OPERATOR_PRECEDENT_WARN" if warnings else "OPERATOR_PRECEDENT_PASS"
    )

    payload = {
        "generated_at_utc": generated_at,
        "status": status,
        "artifact_paths": {
            "manual_history": str(manual_history_path),
            "order_intents": str(order_intents_path),
            "target_state": str(target_state_path),
        },
        "operator_claim": {
            "claim": "funding-adjusted 30-calendar-day return exceeded 200%",
            "required_return_pct": OPERATOR_CLAIM_30D_RETURN_PCT,
            "verified": precedent.get("claim_verified") is True,
        },
        "precedent": precedent,
        "runtime_alignment": runtime,
        "checks": checks,
        "blockers": [item["message"] for item in blockers],
        "warnings": [item["message"] for item in warnings],
        "contract": {
            "advisory_only": True,
            "may_rank_current_live_ready_lanes": True,
            "cannot_override": [
                "RiskEngine",
                "LiveOrderGateway",
                "gpt_trader_verifier",
                "fresh_broker_truth",
                "forecast_confidence_gate",
                "spread_and_event_gates",
                "position_close_gate_a_b",
            ],
        },
    }
    _write_json(output_path, payload)
    _write_report(report_path, payload)
    return OperatorPrecedentSummary(
        output_path=output_path,
        report_path=report_path,
        status=status,
        checks=len(checks),
        blockers=len(blockers),
        warnings=len(warnings),
        best_30d_return_pct=_maybe_float(
            ((precedent.get("funding_adjusted_performance") or {}).get("best_30d") or {}).get("return_pct")
        ),
        live_ready_lanes=int(runtime["live_ready_lanes"]),
        aligned_live_ready_lanes=int(runtime["aligned_live_ready_lanes"]),
    )


def _extract_precedent(payload: dict[str, Any]) -> dict[str, Any]:
    analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
    overall = _dict(analysis.get("overall"))
    by_pair = _dict(analysis.get("by_pair"))
    by_side = _dict(analysis.get("by_side"))
    by_session = _dict(analysis.get("by_session_jst"))
    by_close = _dict(analysis.get("by_close_reason"))
    cash = _dict(analysis.get("cash_flows"))
    best_30d = _dict(cash.get("best_30d_funding_adjusted"))

    pair_rank = _rank_by_net(by_pair)
    side_rank = _rank_by_net(by_side)
    session_rank = _rank_by_net(by_session)
    positive_sessions = [item["name"] for item in session_rank if item["net"] > 0]
    primary_sessions = positive_sessions[:PRIMARY_PRECEDENT_SESSION_COUNT]
    negative_sessions = [item["name"] for item in session_rank if item["net"] < 0]
    best_pair = pair_rank[0]["name"] if pair_rank else None
    best_side = side_rank[0]["name"] if side_rank else None
    worst_side = side_rank[-1]["name"] if side_rank else None
    margin_closeout = _dict(by_close.get("MARKET_ORDER_MARGIN_CLOSEOUT"))
    exit_events = int(payload.get("exit_events") or overall.get("trades") or 0)

    return {
        "sample": {
            "transaction_count": int(payload.get("transaction_count") or 0),
            "exit_events": exit_events,
            "closed_trades": int(payload.get("closed_trades") or 0),
            "reduced_trades": int(payload.get("reduced_trades") or 0),
            "window": payload.get("window") or {},
            "active_window": _active_exit_window(payload, exit_events),
        },
        "funding_adjusted_performance": {
            "net_additional_transfers": _maybe_float(cash.get("net_additional_transfers")),
            "peak_profit": _maybe_float(cash.get("transfer_adjusted_peak_profit")),
            "peak_return_pct": _maybe_float(cash.get("transfer_adjusted_peak_return_pct")),
            "end_profit": _maybe_float(cash.get("transfer_adjusted_end_profit")),
            "end_return_pct": _maybe_float(cash.get("transfer_adjusted_end_return_pct")),
            "best_30d": {
                "start_time": best_30d.get("start_time"),
                "end_time": best_30d.get("end_time"),
                "profit": _maybe_float(best_30d.get("profit")),
                "return_pct": _maybe_float(best_30d.get("return_pct")),
                "net_transfers": _maybe_float(best_30d.get("net_transfers")),
            },
        },
        "winning_shape": {
            "primary_pair": best_pair,
            "primary_direction": best_side,
            "primary_sessions": primary_sessions,
            "positive_sessions": positive_sessions,
            "negative_sessions": negative_sessions,
            "median_hold_hours": _maybe_float(overall.get("median_hold_hours")),
            "payoff": _maybe_float(overall.get("payoff")),
            "expectancy_jpy_per_exit": _maybe_float(overall.get("expectancy")),
        },
        "failure_shape": {
            "worst_direction": worst_side,
            "margin_closeout": {
                "trades": int(margin_closeout.get("trades") or 0),
                "net_jpy": _maybe_float(margin_closeout.get("net")),
                "median_hold_hours": _maybe_float(margin_closeout.get("median_hold_hours")),
                "win_rate": _maybe_float(margin_closeout.get("win_rate")),
            },
        },
        "pair_rank": pair_rank,
        "side_rank": side_rank,
        "session_rank": session_rank,
        "claim_verified": (_maybe_float(best_30d.get("return_pct")) or 0.0) >= OPERATOR_CLAIM_30D_RETURN_PCT,
    }


def _precedent_checks(precedent: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    best_30d = (precedent.get("funding_adjusted_performance") or {}).get("best_30d") or {}
    return_pct = _maybe_float(best_30d.get("return_pct"))
    checks.append(
        _check(
            "funding_adjusted_30d_claim",
            "PASS" if return_pct is not None and return_pct >= OPERATOR_CLAIM_30D_RETURN_PCT else "BLOCK",
            f"best funding-adjusted 30d return {return_pct}% verifies the operator 200%+ claim"
            if return_pct is not None and return_pct >= OPERATOR_CLAIM_30D_RETURN_PCT
            else "manual history does not prove the funding-adjusted 30d 200%+ claim",
            {
                "return_pct": return_pct,
                "required_return_pct": OPERATOR_CLAIM_30D_RETURN_PCT,
            },
        )
    )
    transfers = _maybe_float((precedent.get("funding_adjusted_performance") or {}).get("net_additional_transfers"))
    checks.append(
        _check(
            "raw_balance_not_used_as_strategy_pnl",
            "PASS" if transfers is not None else "BLOCK",
            "manual history separates account funding from strategy P/L"
            if transfers is not None
            else "manual history lacks transfer separation; raw balance would overstate strategy performance",
            {"net_additional_transfers": transfers},
        )
    )
    winning_shape = precedent.get("winning_shape") or {}
    if not winning_shape.get("primary_pair") or not winning_shape.get("primary_direction"):
        checks.append(
            _check(
                "winning_shape_extracted",
                "BLOCK",
                "manual history does not expose a primary pair/direction shape",
                {"winning_shape": winning_shape},
            )
        )
    else:
        checks.append(
            _check(
                "winning_shape_extracted",
                "PASS",
                "manual history exposes a primary pair/direction/session shape",
                {"winning_shape": winning_shape},
            )
        )
    margin = (precedent.get("failure_shape") or {}).get("margin_closeout") or {}
    checks.append(
        _check(
            "failure_shape_extracted",
            "PASS" if (margin.get("trades") or 0) > 0 and _maybe_float(margin.get("net_jpy")) is not None else "WARN",
            "manual history exposes the margin-closeout failure mode"
            if (margin.get("trades") or 0) > 0
            else "manual history has no margin-closeout failure mode to compare",
            {"margin_closeout": margin},
            severity="INFO" if (margin.get("trades") or 0) > 0 else "WARN",
        )
    )
    return checks


def _runtime_alignment(
    intents_payload: dict[str, Any],
    target_payload: dict[str, Any],
    *,
    precedent: dict[str, Any],
) -> dict[str, Any]:
    winning = precedent.get("winning_shape") or {}
    primary_pair = str(winning.get("primary_pair") or "")
    primary_direction = str(winning.get("primary_direction") or "").upper()
    match_sessions = winning.get("primary_sessions") or winning.get("positive_sessions") or []
    positive_sessions = {str(item).upper() for item in match_sessions}
    live_ready: list[dict[str, Any]] = []
    aligned: list[dict[str, Any]] = []
    for row in intents_payload.get("results") or []:
        if not isinstance(row, dict) or row.get("status") != "LIVE_READY":
            continue
        intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
        context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        lane = {
            "lane_id": row.get("lane_id"),
            "pair": intent.get("pair"),
            "side": intent.get("side"),
            "method": context.get("method"),
            "order_type": intent.get("order_type"),
            "session": context.get("session") or metadata.get("session_current_tag") or metadata.get("session_bucket"),
        }
        live_ready.append(lane)
        session_text = str(lane.get("session") or "").upper()
        session_matches = not positive_sessions or _session_matches(session_text, positive_sessions)
        if (
            str(lane.get("pair") or "").upper() == primary_pair.upper()
            and str(lane.get("side") or "").upper() == primary_direction
            and session_matches
        ):
            aligned.append(lane)

    target_trades = _maybe_float(target_payload.get("target_trades_per_day"))
    active_window = (precedent.get("sample") or {}).get("active_window") or {}
    manual_cadence = _maybe_float(active_window.get("exit_events_per_calendar_day"))
    return {
        "live_ready_lanes": len(live_ready),
        "aligned_live_ready_lanes": len(aligned),
        "aligned_lanes": aligned[:10],
        "live_ready_sample": live_ready[:10],
        "target_trades_per_day": target_trades,
        "manual_exit_events_per_calendar_day": manual_cadence,
        "alignment_contract": {
            "aligned_precedent_is_advisory": True,
            "absence_of_alignment_is_not_a_trade_blocker": True,
            "current_risk_geometry_remains_authority": True,
        },
    }


def _session_matches(session_text: str, positive_sessions: set[str]) -> bool:
    if not session_text:
        # Missing session tags should not disqualify an otherwise current lane.
        # The audit remains advisory and reports the raw lane for operator review.
        return True
    normalized_session = session_text.replace("-", "_")
    session_parts = set(normalized_session.split("_"))
    for token in positive_sessions:
        normalized_token = token.replace("-", "_")
        if normalized_token in normalized_session:
            return True
        token_parts = set(normalized_token.split("_"))
        if session_parts & token_parts:
            return True
    return False


def _rank_by_net(bucket: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metrics in bucket.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "name": str(name),
                "trades": int(metrics.get("trades") or 0),
                "net": float(metrics.get("net") or 0.0),
                "win_rate": _maybe_float(metrics.get("win_rate")),
                "payoff": _maybe_float(metrics.get("payoff")),
                "median_hold_hours": _maybe_float(metrics.get("median_hold_hours")),
            }
        )
    return sorted(rows, key=lambda item: item["net"], reverse=True)


def _active_exit_window(payload: dict[str, Any], exit_events: int) -> dict[str, Any]:
    close_times: list[datetime] = []
    for trade in payload.get("trades") or []:
        if not isinstance(trade, dict):
            continue
        close_time = _parse_dt(trade.get("close_time"))
        if close_time is not None:
            close_times.append(close_time)
    if not close_times:
        window = payload.get("window") if isinstance(payload.get("window"), dict) else {}
        start = _parse_dt(window.get("from"))
        end = _parse_dt(window.get("to"))
    else:
        start = min(close_times)
        end = max(close_times)
    if start is None or end is None:
        return {}
    days = max((end - start).total_seconds() / 86400.0, 1.0)
    return {
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "calendar_days": round(days, 2),
        "exit_events_per_calendar_day": round(exit_events / days, 2) if exit_events else 0.0,
    }


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        if "." in text:
            head, rest = text.split(".", 1)
            digit_count = 0
            while digit_count < len(rest) and rest[digit_count].isdigit():
                digit_count += 1
            fraction = rest[:digit_count][:6]
            text = f"{head}.{fraction}{rest[digit_count:]}"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    precedent = payload["precedent"]
    perf = precedent["funding_adjusted_performance"]
    best_30d = perf["best_30d"]
    winning = precedent["winning_shape"]
    failure = precedent["failure_shape"]["margin_closeout"]
    runtime = payload["runtime_alignment"]
    lines = [
        "# Operator Precedent Audit",
        "",
        f"- Generated at UTC: `{payload['generated_at_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Best funding-adjusted 30d return: `{best_30d.get('return_pct')}`% (`{best_30d.get('profit')}` JPY)",
        f"- Peak funding-adjusted return: `{perf.get('peak_return_pct')}`%",
        f"- Winning shape: `{winning.get('primary_pair')} {winning.get('primary_direction')}`; primary sessions `{', '.join(winning.get('primary_sessions') or [])}`; median hold `{winning.get('median_hold_hours')}`h",
        f"- Failure shape: margin closeout `{failure.get('trades')}` exits, net `{failure.get('net_jpy')}` JPY, median hold `{failure.get('median_hold_hours')}`h",
        f"- Current LIVE_READY lanes: `{runtime['live_ready_lanes']}`; precedent-aligned: `{runtime['aligned_live_ready_lanes']}`",
        "",
        "## Checks",
        "",
        "| check | status | message |",
        "|---|---|---|",
    ]
    for check in payload["checks"]:
        lines.append(f"| `{check['check_name']}` | `{check['status']}` | {check['message']} |")
    lines += [
        "",
        "## Contract",
        "",
        "- Advisory only: this audit may rank or explain already-current LIVE_READY lanes.",
        "- It cannot override RiskEngine, LiveOrderGateway, forecast, spread, event, broker-truth, or close Gate A/B checks.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check(
    check_name: str,
    status: str,
    message: str,
    evidence: dict[str, Any],
    *,
    severity: str | None = None,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "status": status,
        "severity": severity or ("BLOCK" if status == "BLOCK" else "INFO"),
        "message": message,
        "evidence": evidence,
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "top-level JSON is not an object"
    return payload, None


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _maybe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
