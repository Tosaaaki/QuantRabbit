from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from quant_rabbit.market_read_overlay import (
    CODEX_MARKET_READ_AUTHOR,
    canonical_json_sha256,
    execution_envelope_payload,
    quote_basis_by_pair_from_broker_payload,
    revalidate_codex_market_read_artifacts,
    validate_codex_market_read_provenance,
    validate_market_read_numeric_geometry,
)
from quant_rabbit.market_read_ledger import (
    market_read_feedback_summary,
    record_market_read_prediction,
    refresh_market_read_measurements,
)
from quant_rabbit.market_read_contract import market_read_missing_fields
from quant_rabbit.close_discipline import (
    thesis_evolution_reason_has_hard_close_evidence,
)


def _trader_sl_repair_disabled() -> bool:
    return os.environ.get("QR_TRADER_DISABLE_SL_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # OANDA timestamps may carry nanoseconds while Python 3.10 accepts at
    # most microseconds. Ordering here only needs a conservative UTC anchor;
    # truncate, never round, the fractional part before parsing.
    text = re.sub(r"(\.\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _operator_close_override_active() -> bool:
    """Emergency override for the CLOSE-discipline gate.

    Set `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell when an out-of-
    band close is urgently needed (broker disconnect, regulatory order,
    user-confirmed structural reversal not yet visible to pair_charts).
    Documented so the override is auditable rather than implicit.
    """
    return os.environ.get("QR_OPERATOR_CLOSE_OVERRIDE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


# J (2026-05-13) — Operator close token file. A token-file authorization
# path that the GPT trader receipt cannot self-set. The operator creates
# the file explicitly with `touch data/.operator_close_token`; the verifier
# reads its mtime and rejects if older than the documented freshness window.
#
# 2026-05-12T15:33 UTC mass-close incident proved that
# `operator_close_authorized: true` JSON field is honor-system: the
# trader fills its own receipt. The token file lives in `data/`, a
# directory the trader process can technically write to, but it would
# require the model to identify, name, and `touch` the file — none of
# which match the normal write-this-decision-receipt flow. The
# `operator_close_authorized` JSON field is now treated as advisory
# audit only; it is NOT accepted as Gate B authorization.
OPERATOR_CLOSE_TOKEN_FILENAME = ".operator_close_token"
OPERATOR_CLOSE_TOKEN_FRESH_SECONDS = 300  # 5 minutes documented window


def _projection_pending_expiry_grace() -> timedelta:
    """Same-cycle projection resolution grace used by live telemetry audits."""

    try:
        seconds = max(0.0, float(os.environ.get("QR_PROJECTION_PENDING_EXPIRY_GRACE_SECONDS", "1200")))
    except (TypeError, ValueError):
        seconds = 1200.0
    return timedelta(seconds=seconds)


def _operator_close_token_fresh(data_root: Path | None = None) -> bool:
    """Whether a fresh operator close-authorization token file exists.

    The default location is `data/.operator_close_token` under the
    repo root; tests inject their own path. A token older than
    `OPERATOR_CLOSE_TOKEN_FRESH_SECONDS` is treated as stale and the
    gate fails — operators must explicitly re-authorize before each
    CLOSE batch.
    """
    if data_root is None:
        # Lazy import to avoid module-level cwd dependency in tests.
        from quant_rabbit.paths import ROOT
        data_root = ROOT / "data"
    token = data_root / OPERATOR_CLOSE_TOKEN_FILENAME
    if not token.exists():
        return False
    try:
        mtime = token.stat().st_mtime
    except OSError:
        return False
    age = datetime.now(timezone.utc).timestamp() - mtime
    return age <= OPERATOR_CLOSE_TOKEN_FRESH_SECONDS


def _operator_close_gate_authorized() -> bool:
    """Gate B authorization for loss-side CLOSE decisions."""
    return _operator_close_override_active() or _operator_close_token_fresh()


def _decision_cites_close_timing_evidence(evidence_refs: tuple[str, ...]) -> bool:
    for ref in evidence_refs:
        text = str(ref or "").strip()
        if text in {"timing:audit", "timing:loss_closes", "timing:market_closes"}:
            return True
        if text.startswith("timing:loss_close:") or text.startswith("timing:market_close:"):
            return True
    return False


def _decision_cites_pending_cancel_timing_evidence(evidence_refs: tuple[str, ...]) -> bool:
    for ref in evidence_refs:
        text = str(ref or "").strip()
        if text in {"timing:audit", "timing:canceled_orders"}:
            return True
        if text.startswith("timing:canceled_order:"):
            return True
    return False


def _decision_cites_projection_evidence(evidence_refs: tuple[str, ...]) -> bool:
    for ref in evidence_refs:
        text = str(ref or "").strip()
        if text in {"projection:ledger", "projection:expired_pending"}:
            return True
        if text.startswith("projection:expired_pending:"):
            return True
    return False


def _projection_row_expired(row: dict[str, Any], *, now: datetime) -> bool:
    emitted = _parse_utc(row.get("timestamp_emitted_utc"))
    window_min = _optional_float(row.get("resolution_window_min"))
    if emitted is None or window_min is None or window_min <= 0:
        return True
    return now >= emitted + timedelta(minutes=window_min) + _projection_pending_expiry_grace()


def _projection_row_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pair": row.get("pair"),
        "signal_name": row.get("signal_name"),
        "direction": row.get("direction"),
        "cycle_id": row.get("cycle_id"),
        "timestamp_emitted_utc": row.get("timestamp_emitted_utc"),
        "resolution_window_min": row.get("resolution_window_min"),
    }


def _projection_ledger_packet(path: Path, *, now: datetime | None = None) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "evidence_ref": "projection:ledger",
        "path": str(path),
        "status": "missing",
        "rows": 0,
        "malformed_rows": 0,
        "status_counts": {},
        "expired_pending_count": 0,
        "expired_pending_examples": [],
    }
    if not path.exists():
        return packet
    now = now or datetime.now(timezone.utc)
    status_counts: dict[str, int] = {}
    expired: list[dict[str, Any]] = []
    malformed = 0
    rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                rows += 1
                status = str(row.get("resolution_status") or "PENDING").upper()
                status_counts[status] = status_counts.get(status, 0) + 1
                if status == "PENDING" and _projection_row_expired(row, now=now):
                    expired.append(row)
    except OSError as exc:
        packet["status"] = "unreadable"
        packet["error"] = str(exc)
        return packet

    packet.update(
        {
            "status": "BLOCK" if expired else ("WARN" if malformed else "OK"),
            "rows": rows,
            "malformed_rows": malformed,
            "status_counts": status_counts,
            "expired_pending_count": len(expired),
            "expired_pending_examples": [_projection_row_ref(row) for row in expired[:8]],
        }
    )
    return packet


def _projection_ledger_trade_blockers(packet: dict[str, Any]) -> list[str]:
    projection = packet.get("projection_ledger")
    if not isinstance(projection, dict):
        return []
    blockers: list[str] = []
    expired = _optional_int(projection.get("expired_pending_count")) or 0
    if expired > 0:
        examples = []
        for row in projection.get("expired_pending_examples", []) or []:
            if not isinstance(row, dict):
                continue
            pair = str(row.get("pair") or "").strip()
            signal = str(row.get("signal_name") or "").strip()
            direction = str(row.get("direction") or "").strip()
            if pair or signal or direction:
                examples.append(" ".join(part for part in (pair, signal, direction) if part))
        suffix = f"; examples={', '.join(examples[:3])}" if examples else ""
        blockers.append(
            f"PROJECTION_LEDGER_EXPIRED_PENDING count={expired}{suffix}. "
            "Run verify-projections before taking new risk so forecast HIT/MISS/TIMEOUT is measured."
        )
    if str(projection.get("status") or "").upper() == "UNREADABLE":
        blockers.append(
            "PROJECTION_LEDGER_UNREADABLE: projection telemetry cannot be audited before new risk"
        )
    return blockers


def _news_health_trade_blockers(packet: dict[str, Any]) -> list[str]:
    news = packet.get("news")
    if not isinstance(news, dict):
        return ["NEWS_PACKET_MISSING: current news evidence is absent from the decision packet"]
    health = news.get("health")
    if not isinstance(health, dict):
        return ["NEWS_HEALTH_MISSING: news freshness and market-story sync were not audited"]
    blockers: list[str] = []
    status = str(health.get("status") or "").strip().upper()
    if status in {"", "MISSING", "BLOCK", "ERROR"}:
        blockers.append(
            f"NEWS_HEALTH_{status or 'MISSING'}: news freshness / market-story sync is not tradeable"
        )
    for issue in health.get("issues", []) or []:
        text = str(issue or "").strip()
        if not text:
            continue
        upper = text.upper()
        if upper.startswith("BLOCK") or ":BLOCK" in upper or "BLOCK:" in upper:
            blockers.append(text)
    return blockers


def _user_alpha_continuation_from_overrides(
    payload: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    base = {
        "evidence_ref": "user_alpha:continuation",
        "latest_evidence_ref": "user_alpha:latest",
        "status": "NONE",
        "active": False,
        "required_user_alpha_continuation_block": True,
    }
    if not isinstance(payload, dict) or not payload:
        return base
    expires_at = _parse_utc(payload.get("expires_at_utc"))
    if expires_at is not None and (now or datetime.now(timezone.utc)) >= expires_at:
        expired = dict(base)
        expired["status"] = "EXPIRED"
        expired["expires_at_utc"] = payload.get("expires_at_utc")
        return expired
    continuation = payload.get("user_alpha_continuation")
    if isinstance(continuation, dict) and continuation.get("active"):
        packet = dict(continuation)
    else:
        trades = payload.get("user_alpha_trades")
        latest = trades[-1] if isinstance(trades, list) and trades and isinstance(trades[-1], dict) else None
        if latest is None:
            return base
        packet = {
            "status": "ACTIVE",
            "active": True,
            "edge_source": "USER_ALPHA",
            "latest_trade": latest,
            "five_pct_path_board_candidate": {
                "source": "USER_ALPHA",
                "pair": latest.get("pair"),
                "direction": latest.get("direction"),
                "candidate_roles": ["RELOAD", "SECOND_SHOT"],
                "target_layer": "PACE_5",
            },
            "required_trader_answers": [
                "thesis_alive",
                "reload_candidate",
                "second_shot_candidate",
                "exact_blocker_if_no_continuation",
                "next_trigger",
            ],
            "if_no_continuation_requires_exact_blocker": True,
        }
    packet.setdefault("evidence_ref", "user_alpha:continuation")
    packet.setdefault("latest_evidence_ref", "user_alpha:latest")
    latest_trade = packet.get("latest_trade")
    if isinstance(latest_trade, dict):
        pair = str(latest_trade.get("pair") or "").strip()
        direction = str(latest_trade.get("direction") or "").strip().upper()
        if pair and direction:
            packet["pair_direction_evidence_ref"] = f"user_alpha:{pair}:{direction}"
    return packet


def _active_user_alpha_continuation(packet: dict[str, Any]) -> dict[str, Any] | None:
    continuation = packet.get("user_alpha_continuation")
    if not isinstance(continuation, dict) or not continuation.get("active"):
        return None
    latest = continuation.get("latest_trade")
    if not isinstance(latest, dict):
        return None
    pair = str(latest.get("pair") or "").strip()
    direction = str(latest.get("direction") or "").strip().upper()
    if not pair or direction not in {"LONG", "SHORT"}:
        return None
    return continuation


def _user_alpha_evidence_refs(continuation: dict[str, Any]) -> tuple[str, ...]:
    refs = [
        str(continuation.get("evidence_ref") or "user_alpha:continuation"),
        str(continuation.get("latest_evidence_ref") or "user_alpha:latest"),
    ]
    pair_direction_ref = str(continuation.get("pair_direction_evidence_ref") or "").strip()
    if pair_direction_ref:
        refs.append(pair_direction_ref)
    return tuple(dict.fromkeys(ref for ref in refs if ref))


def _selected_lanes_include_user_alpha(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    continuation: dict[str, Any],
) -> bool:
    latest = continuation.get("latest_trade") if isinstance(continuation, dict) else {}
    pair = str(latest.get("pair") or "").strip()
    direction = str(latest.get("direction") or "").strip().upper()
    if not pair or direction not in {"LONG", "SHORT"}:
        return False
    lanes = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", []) or []
        if isinstance(lane, dict)
    }
    for lane_id in selected_lane_ids:
        lane = lanes.get(lane_id)
        if not lane:
            continue
        if str(lane.get("pair") or "").strip() == pair and str(lane.get("direction") or "").strip().upper() == direction:
            return True
    return False


def _decision_text_blob(decision: "GPTTraderDecision") -> str:
    parts: list[str] = [
        decision.thesis,
        decision.narrative,
        decision.chart_story,
        decision.invalidation,
        decision.operator_summary,
    ]
    parts.extend(decision.rejected_alternatives)
    parts.extend(decision.risk_notes)
    plan = decision.twenty_minute_plan if isinstance(decision.twenty_minute_plan, dict) else {}
    for key in TWENTY_MINUTE_PLAN_TEXT_FIELDS:
        parts.append(str(plan.get(key) or ""))
    return " ".join(part for part in parts if part)


_FORBIDDEN_LOSS_EXIT_REASON_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "unrealized P/L",
        re.compile(r"\bunrealized\s*(?:p/l|pl|profit.?loss)\b", re.IGNORECASE),
    ),
    (
        "negative P/L",
        re.compile(r"\bnegative\s*(?:p/l|pl|profit.?loss)\b|\bred\s+position\b|\btemporarily\s+red\b|\bunderwater\b", re.IGNORECASE),
    ),
    (
        "NEGATIVE_EXPECTANCY",
        re.compile(r"\bnegative[_\s-]*expectancy\b|\bNEGATIVE_EXPECTANCY\b", re.IGNORECASE),
    ),
    (
        "duplicate blocker",
        re.compile(r"\bduplicate\s+(?:blocker|parent|geometry|basket)\b|\bBASKET_DUPLICATE[A-Z0-9_]*\b", re.IGNORECASE),
    ),
    (
        "low LIVE_READY",
        re.compile(r"\b(?:NO[_\s-]*LIVE[_\s-]*READY|LIVE[_\s-]*READY\s*=\s*0|LIVE[_\s-]*READY[^.]{0,40}\blow\b)", re.IGNORECASE),
    ),
    (
        "prior SL template",
        re.compile(r"\b(?:prior|old)\s+SL\s+template\b|\b(?:prior|old)\s+stop\s+template\b|\bSL\s+template\b", re.IGNORECASE),
    ),
)


def _non_invalidation_loss_exit_reason_hits(decision: "GPTTraderDecision") -> tuple[str, ...]:
    text = " ".join((_decision_text_blob(decision), " ".join(decision.evidence_refs)))
    hits: list[str] = []
    for label, pattern in _FORBIDDEN_LOSS_EXIT_REASON_PATTERNS:
        if pattern.search(text):
            hits.append(label)
    return tuple(hits)


def _decision_cites_user_alpha_exact_blocker(
    decision: "GPTTraderDecision",
    continuation: dict[str, Any],
) -> bool:
    latest = continuation.get("latest_trade") if isinstance(continuation, dict) else {}
    pair = str(latest.get("pair") or "").strip().upper()
    direction = str(latest.get("direction") or "").strip().upper()
    text = _decision_text_blob(decision).upper()
    if "BLOCKER" not in text and "BLOCKED" not in text:
        return False
    if pair and pair.upper() not in text:
        return False
    if direction and direction not in text:
        return False
    return True


def _user_alpha_report_lines(
    continuation: dict[str, Any] | None,
    *,
    decision: dict[str, Any] | None = None,
) -> list[str]:
    lines = ["", "## USER ALPHA CONTINUATION", ""]
    if not isinstance(continuation, dict) or not continuation.get("active"):
        return lines + ["- Status: `NONE`"]
    latest = continuation.get("latest_trade") if isinstance(continuation.get("latest_trade"), dict) else {}
    candidate = (
        continuation.get("five_pct_path_board_candidate")
        if isinstance(continuation.get("five_pct_path_board_candidate"), dict)
        else {}
    )
    roles = candidate.get("candidate_roles") if isinstance(candidate.get("candidate_roles"), list) else []
    selected = ", ".join(decision.get("selected_lane_ids") or []) if isinstance(decision, dict) else ""
    lines.extend(
        [
            f"- Status: `{continuation.get('status') or 'ACTIVE'}`",
            (
                "- Latest user alpha: "
                f"`{latest.get('classification') or 'USER_ALPHA'}` "
                f"`{latest.get('pair') or 'unknown'}` `{latest.get('direction') or 'unknown'}` "
                f"realized `{latest.get('realized_pl_jpy')}` JPY"
            ),
            (
                "- Metadata: "
                f"entry=`{latest.get('entry')}` tp=`{latest.get('tp')}` "
                f"mfe=`{latest.get('max_favorable_excursion')}` "
                f"time_to_tp_seconds=`{latest.get('time_to_tp_seconds')}` "
                f"thesis=`{latest.get('thesis') or 'unknown'}`"
            ),
            (
                "- Discovery: "
                f"system_discovered=`{latest.get('system_discovered')}` "
                f"discovered_by=`{latest.get('discovered_by') or 'OPERATOR'}` "
                f"system_tp_managed=`{latest.get('system_tp_managed')}`"
            ),
            (
                "- 5% path board candidate: "
                f"`{candidate.get('pair') or latest.get('pair')}` "
                f"`{candidate.get('direction') or latest.get('direction')}` "
                f"roles=`{', '.join(str(role) for role in roles) or 'RELOAD, SECOND_SHOT'}`"
            ),
            "- Thesis alive: `REQUIRED`",
            "- RELOAD candidate: `REQUIRED`",
            "- SECOND SHOT candidate: `REQUIRED`",
            "- Exact blocker if no continuation: `REQUIRED`",
            f"- Selected basket lanes: `{selected or 'none'}`",
        ]
    )
    return lines


def _decision_cites_profitability_p0(evidence_refs: tuple[str, ...]) -> bool:
    for ref in evidence_refs:
        text = str(ref or "").strip()
        if text in {
            "self_improvement:audit",
            "self_improvement:profitability",
            "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
        }:
            return True
    return False


def _decision_cites_profitability_acceptance(evidence_refs: tuple[str, ...]) -> bool:
    return any(str(ref or "").strip() == "profitability:acceptance" for ref in evidence_refs)


def _profitability_p0_soft_close_blocker(packet: dict[str, Any]) -> dict[str, Any] | None:
    """Return the active profitability-P0 blocker that makes soft closes higher risk."""

    audit = packet.get("self_improvement_audit")
    if not isinstance(audit, dict):
        return None
    blockers = list(audit.get("profitability_blockers", []) or []) + list(
        audit.get("p0_blockers", []) or []
    )
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        if str(blocker.get("code") or "") == "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED":
            return blocker
    return None


def _profitability_acceptance_loss_close_blockers(packet: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return active acceptance P0s proving loss-side market-close leakage."""

    acceptance = packet.get("profitability_acceptance")
    if not isinstance(acceptance, dict):
        return ()
    findings = acceptance.get("p0_findings")
    if not isinstance(findings, list):
        return ()
    leak_codes = {
        "RECENT_GATEWAY_LOSS_MARKET_CLOSE_LEAK",
        "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
        "UNVERIFIED_LOSS_SIDE_MARKET_CLOSE_RECONCILED",
        "LOSS_CLOSE_GATE_EVIDENCE_MISSING",
    }
    blockers: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if str(finding.get("code") or "") in leak_codes:
            blockers.append(finding)
    return tuple(blockers)


def _loss_close_timing_audit_required_trades(
    packet: dict[str, Any],
    decision: "GPTTraderDecision",
    position_by_tid: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    """Require timing-regret evidence after recent premature loss-close proof.

    This is not an automatic HOLD veto. It forces the trader to acknowledge the
    latest execution-timing counterfactual before another underwater market
    close can pass, so H4/sidecar hard evidence can still close a broken thesis
    when the receipt explicitly weighs that timing risk.
    """

    audit = packet.get("execution_timing_audit")
    if not isinstance(audit, dict):
        return ()
    summary = audit.get("summary")
    if not isinstance(summary, dict):
        return ()
    premature = _optional_int(summary.get("loss_market_closes_may_have_been_premature")) or 0
    capture_missed = _timing_profit_capture_missed_count(summary)
    counterfactual_delta = _timing_profit_capture_counterfactual_delta(summary)
    if (
        premature <= 0
        and capture_missed <= 0
        and (counterfactual_delta is None or counterfactual_delta <= 0.0)
    ):
        return ()
    if _decision_cites_close_timing_evidence(decision.evidence_refs):
        return ()

    blocked: list[str] = []
    for tid in decision.close_trade_ids:
        pos = position_by_tid.get(str(tid))
        if pos is None:
            continue
        unrealized = _optional_float(pos.get("unrealized_pl_jpy"))
        if unrealized is not None and unrealized > 0.0:
            continue
        pair = str(pos.get("pair") or "")
        side = str(pos.get("side") or "")
        label = f"{tid}"
        if pair or side:
            label += f" ({pair} {side})".rstrip()
        blocked.append(label)
    return tuple(blocked)


def _timing_profit_capture_missed_count(summary: dict[str, Any]) -> int:
    if _timing_profit_capture_split_present(summary):
        post_missed = (
            _optional_int(
                summary.get("post_repair_live_evidence_loss_closes_profit_capture_missed")
            )
            or 0
        )
        post_replay_missed = (
            _optional_int(
                summary.get("post_repair_live_evidence_loss_closes_repair_replay_triggered")
            )
            or 0
        )
        return max(post_missed, post_replay_missed)
    return _optional_int(summary.get("loss_closes_profit_capture_missed")) or 0


def _timing_profit_capture_split_present(summary: dict[str, Any]) -> bool:
    return "post_repair_live_evidence_loss_closes_profit_capture_missed" in summary


def _timing_profit_capture_counterfactual_delta(summary: dict[str, Any]) -> float | None:
    delta = _optional_float(summary.get("loss_close_counterfactual_profit_capture_delta_jpy"))
    if _timing_profit_capture_split_present(summary) and (
        _timing_profit_capture_missed_count(summary) <= 0
    ):
        return None
    return delta


def _premature_loss_close_timing_guard(packet: dict[str, Any]) -> dict[str, Any] | None:
    """Return timing-regret proof that soft underwater CLOSE must not ignore."""

    audit = packet.get("execution_timing_audit")
    if not isinstance(audit, dict):
        return None
    summary = audit.get("summary")
    if not isinstance(summary, dict):
        return None
    premature = _optional_int(summary.get("loss_market_closes_may_have_been_premature")) or 0
    capture_missed = _timing_profit_capture_missed_count(summary)
    counterfactual_delta = _timing_profit_capture_counterfactual_delta(summary)
    if (
        premature <= 0
        and capture_missed <= 0
        and (counterfactual_delta is None or counterfactual_delta <= 0.0)
    ):
        return None
    audited = _optional_int(summary.get("loss_market_closes_audited"))
    if audited is None:
        audited = _optional_int(summary.get("loss_closes_audited"))
    return {
        "premature": premature,
        "contained": _optional_int(summary.get("loss_market_closes_contained_risk")),
        "audited": audited,
        "followthrough_jpy": _optional_float(
            summary.get("market_close_estimated_followthrough_jpy")
        ),
        "profit_capture_missed": capture_missed,
        "profit_capture_counterfactual_delta_jpy": counterfactual_delta,
        "profit_capture_counterfactual_jpy": _optional_float(
            summary.get("loss_close_counterfactual_profit_capture_jpy")
        ),
        "profit_capture_actual_pl_jpy": _optional_float(
            summary.get("loss_close_actual_pl_jpy")
        ),
        "profit_capture_counterfactual_pl_jpy": _optional_float(
            summary.get("loss_close_counterfactual_profit_capture_pl_jpy")
        ),
    }


def _pending_cancel_timing_audit_required_orders(
    packet: dict[str, Any],
    decision: "GPTTraderDecision",
) -> tuple[str, ...]:
    """Orders whose same-shape cancel-regret evidence must be acknowledged."""

    audit = packet.get("execution_timing_audit")
    if not isinstance(audit, dict):
        return ()
    regrets = audit.get("canceled_order_regrets")
    if not isinstance(regrets, list):
        return ()
    pending_by_id = {
        str(order.get("order_id") or ""): order
        for order in _pending_entry_orders(packet)
        if str(order.get("order_id") or "")
    }
    blocked: list[str] = []
    for order_id in decision.cancel_order_ids:
        order = pending_by_id.get(str(order_id))
        if order is None:
            continue
        pair = str(order.get("pair") or "").strip()
        side = str(order.get("side") or _side_from_order_units(order.get("units")) or "").upper()
        order_type = _normalized_timing_order_type(order.get("order_type"))
        if not pair or not side or not order_type:
            continue
        for row in regrets:
            if not isinstance(row, dict):
                continue
            if str(row.get("pair") or "").strip() != pair:
                continue
            if str(row.get("side") or "").upper() != side:
                continue
            if _normalized_timing_order_type(row.get("order_type")) != order_type:
                continue
            if row.get("sl_touched_after_cancel"):
                continue
            if not row.get("entry_touched_after_cancel"):
                continue
            mfe_pips = _optional_float(row.get("mfe_pips_after_cancel_entry")) or 0.0
            if not row.get("tp_touched_after_cancel") and mfe_pips <= 0.0:
                continue
            prior_id = str(row.get("order_id") or "prior-cancel")
            blocked.append(f"{order_id} ({pair} {side} {order_type}; prior={prior_id})")
            break
    return tuple(blocked)


def _normalized_timing_order_type(order_type: object) -> str:
    text = str(order_type or "").upper().replace("-", "_")
    if text.endswith("_ORDER"):
        text = text[: -len("_ORDER")]
    if text == "STOP_ENTRY":
        return "STOP"
    if text in {"LIMIT", "STOP", "MARKET_IF_TOUCHED"}:
        return text
    return text


# Matches a single per-timeframe segment in chart_reader's chart_story format.
# Example tokens captured:
#   "M15(RANGE, ADX=15.4 ... struct=CHOCH_UP@113.9900)"        (close confirmed)
#   "M15(RANGE, ADX=15.4 ... struct=BOS_UP@114.1460:wick)"     (wick-only)
# Group 1 = timeframe (M1/M5/M15/M30/H1/H4/D), Group 2 = "BOS"|"CHOCH",
# Group 3 = "UP"|"DOWN", Group 4 = numeric price, Group 5 = ":wick" tag or "".
# The optional ":wick" suffix marks a break whose breaking candle closed
# back inside the prior range — Gate A treats it as a stop-hunt that does
# NOT authorize CLOSE on its own (added 2026-05-13, structure.py
# close_confirmed flag).
_STRUCT_EVENT_RE = re.compile(
    r"\b(M1|M5|M15|M30|H1|H4|D)\([^)]*?struct=(BOS|CHOCH)_(UP|DOWN)@([0-9]+\.?[0-9]*)(:wick)?",
)

# Timeframes consulted when deciding whether the operator-thesis behind a
# trader-owned position has been structurally invalidated. M15 catches the
# fast-fail flip; H4 catches the dominant-tape reversal. Lower TFs (M1/M5)
# would whip the gate around session noise; higher TFs (D/W) would lag past
# the per_trade_risk_budget. Both are anchored by chart_reader.structure_events
# rather than a JPY/pip literal, so they're §6-compliant.
CLOSE_DISCIPLINE_TIMEFRAMES: tuple[str, ...] = ("M15", "H4")

# Layers that can represent directional evidence for/against the position
# thesis. Session safety layers such as flow/calendar are intentionally omitted:
# a normal spread or quiet event window is useful context, but it does not prove
# recovery edge for a same-direction position.
CLOSE_DIRECTIONAL_MATRIX_LAYERS: frozenset[str] = frozenset(
    {
        "chart",
        "strength",
        "currency_strength",
        "cross_asset",
        "context_asset_chart",
        "levels",
        "cot",
        "option_skew",
    }
)


def _parse_struct_events(
    chart_story: str,
) -> dict[str, tuple[str, str, float, bool]]:
    """Return {timeframe: (event_type, direction, price, close_confirmed)} parsed
    from chart_story.

    Silently skips tokens whose price does not parse as a float. The
    parser is intentionally tolerant: chart_reader format drift should
    degrade to "no thesis-invalidation evidence" rather than crash the
    verifier and stall the cycle.

    `close_confirmed` is False when chart_reader appended the `:wick`
    suffix — the breaking candle's close did not clear the prior pivot,
    so the high/low was swept but the close held inside the range. Gate
    A treats wick-only breaks as advisory, not as CLOSE authorization.
    """
    if not chart_story:
        return {}
    out: dict[str, tuple[str, str, float, bool]] = {}
    for tf, event_type, direction, price_str, wick_tag in _STRUCT_EVENT_RE.findall(
        chart_story
    ):
        try:
            close_confirmed = wick_tag != ":wick"
            out[tf] = (event_type, direction, float(price_str), close_confirmed)
        except (TypeError, ValueError):
            continue
    return out


def _pair_chart(packet: dict[str, Any], pair: str) -> dict[str, Any]:
    """Pull the full per-pair chart payload from the verifier packet."""
    pairs_block = (
        ((packet.get("market_context") or {}).get("pairs") or {})
        .get(pair) or {}
    )
    chart = pairs_block.get("chart") if isinstance(pairs_block, dict) else None
    if isinstance(chart, dict):
        return chart
    return {}


def _pair_chart_story(packet: dict[str, Any], pair: str) -> str:
    """Pull the per-pair chart_story string from the verifier packet."""
    return str(_pair_chart(packet, pair).get("chart_story") or "")


def _structured_last_event(
    packet: dict[str, Any],
    pair: str,
    timeframe: str,
) -> dict[str, Any]:
    chart = _pair_chart(packet, pair)
    views = chart.get("views") if isinstance(chart.get("views"), dict) else {}
    view = views.get(timeframe) if isinstance(views, dict) else None
    structure = view.get("structure") if isinstance(view, dict) else None
    event = structure.get("last_event") if isinstance(structure, dict) else None
    return event if isinstance(event, dict) else {}


def _position_structure_anchor(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> tuple[datetime | None, str]:
    """Return the latest broker-open / recorded-thesis timestamp for a trade."""

    if not trade_id:
        return None, "trade id is missing, so post-entry structure cannot be proved"
    candidates: list[tuple[datetime, str]] = []
    broker = packet.get("broker_snapshot") if isinstance(packet.get("broker_snapshot"), dict) else {}
    for item in broker.get("position_summaries", []) or []:
        if not isinstance(item, dict) or str(item.get("trade_id") or "") != str(trade_id):
            continue
        if str(item.get("pair") or "") != pair:
            return None, "matching broker position has a missing or mismatched pair"
        if str(item.get("side") or "").upper() != side.upper():
            return None, "matching broker position has a missing or mismatched side"
        raw_opened = item.get("open_time_utc")
        if raw_opened is None or not str(raw_opened).strip():
            continue
        opened = _parse_utc(raw_opened)
        if opened is None:
            return None, (
                "broker openTime is present but invalid; entry-thesis fallback is "
                "not allowed for standing H4 close authority"
            )
        if opened is not None:
            candidates.append((opened, "broker openTime"))

    sidecars = packet.get("protection_sidecars") if isinstance(packet.get("protection_sidecars"), dict) else {}
    for item in sidecars.get("entry_thesis_close_context", []) or []:
        if not isinstance(item, dict) or str(item.get("trade_id") or "") != str(trade_id):
            continue
        if str(item.get("pair") or "") != pair:
            return None, "matching entry thesis has a missing or mismatched pair"
        if str(item.get("side") or "").upper() != side.upper():
            return None, "matching entry thesis has a missing or mismatched side"
        raw_thesis_at = item.get("entry_thesis_timestamp_utc")
        if raw_thesis_at is None or not str(raw_thesis_at).strip():
            continue
        thesis_at = _parse_utc(raw_thesis_at)
        if thesis_at is None:
            return None, (
                "entry-thesis timestamp is present but invalid; broker-open fallback "
                "is not allowed for standing H4 close authority"
            )
        if thesis_at is not None:
            candidates.append((thesis_at, "entry-thesis timestamp"))

    if not candidates:
        return None, (
            "broker openTime and entry-thesis timestamp are missing or invalid; "
            "H4 event remains soft Gate A"
        )
    anchor, label = max(candidates, key=lambda value: value[0])
    return anchor, f"latest {label} anchor={anchor.isoformat()}"


def _h4_structure_postdates_position(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
    event_type: str,
    direction: str,
    price: float,
) -> tuple[bool, str]:
    """Prove that the structured H4 break occurred after this position opened."""

    event = _structured_last_event(packet, pair, "H4")
    expected_kind = f"{event_type}_{direction}".upper()
    if str(event.get("kind") or "").upper() != expected_kind:
        return False, "structured H4 last_event does not match the chart-story event"
    if event.get("close_confirmed") is not True:
        return False, "structured H4 last_event is not close-confirmed"
    structured_price = _optional_float(event.get("broken_pivot_price"))
    if structured_price is None:
        return False, "structured H4 last_event has no broken_pivot_price"
    try:
        # Chart-story prices are decimal text while structured pivots are
        # floats. Half a pip admits only serialization round-off, not a nearby
        # pivot; replace it with a shared structure-event id when one exists.
        price_tolerance = 0.5 / instrument_pip_factor(pair)
    except (KeyError, TypeError, ValueError):
        price_tolerance = 1e-9
    if abs(structured_price - price) > price_tolerance:
        return False, "structured H4 last_event price does not match chart story"
    event_at = _parse_utc(event.get("timestamp"))
    if event_at is None:
        return False, "structured H4 last_event timestamp is missing or invalid"
    snapshot = packet.get("broker_snapshot") if isinstance(packet.get("broker_snapshot"), dict) else {}
    snapshot_at = _parse_utc(snapshot.get("fetched_at_utc"))
    chart_at = _parse_utc(_pair_chart(packet, pair).get("source_generated_at_utc"))
    if snapshot_at is None or chart_at is None:
        return False, (
            "broker snapshot or pair-chart generated timestamp is missing; "
            "H4 event chronology cannot be proved"
        )
    if event_at > snapshot_at or event_at > chart_at:
        return False, (
            f"H4 event timestamp={event_at.isoformat()} is later than its "
            "broker/chart observation boundary"
        )
    anchor, anchor_reason = _position_structure_anchor(
        packet,
        trade_id=trade_id,
        pair=pair,
        side=side,
    )
    if anchor is None:
        return False, anchor_reason
    if event_at <= anchor:
        return False, (
            f"H4 event timestamp={event_at.isoformat()} does not postdate {anchor_reason}"
        )
    return True, (
        f"H4 event timestamp={event_at.isoformat()} postdates {anchor_reason}"
    )


def _close_same_direction_matrix_support(
    packet: dict[str, Any],
    pair: str,
    side: str,
) -> tuple[bool, str]:
    side_upper = str(side or "").upper()
    if side_upper not in {"LONG", "SHORT"}:
        return False, ""
    pair_block = (
        ((packet.get("market_context") or {}).get("pairs") or {})
        .get(pair) or {}
    )
    matrix = pair_block.get("matrix") if isinstance(pair_block, dict) else None
    reading = matrix.get(side_upper) if isinstance(matrix, dict) else None
    if not isinstance(reading, dict):
        return False, ""

    supports = _directional_matrix_observations(reading.get("supports"))
    if not supports:
        return False, ""
    rejects = _directional_matrix_observations(reading.get("rejects"))
    if len(rejects) >= len(supports):
        return False, ""

    support_codes = [
        str(item.get("code") or item.get("layer") or "directional_support")
        for item in supports
    ]
    refs: list[str] = []
    for item in supports:
        for ref in item.get("evidence_refs") or []:
            text = str(ref).strip()
            if text and text not in refs:
                refs.append(text)
    reading_ref = str(reading.get("evidence_ref") or "").strip()
    if reading_ref and reading_ref not in refs:
        refs.insert(0, reading_ref)
    ref_text = f"; refs={', '.join(refs[:5])}" if refs else ""
    return (
        True,
        f"{pair} {side_upper} still has directional matrix support "
        f"({', '.join(support_codes[:4])}){ref_text}",
    )


def _close_same_direction_sidecar_support(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
    allow_single_strong: bool = False,
) -> tuple[bool, str]:
    """Whether fresh position sidecars still support carrying this position.

    The support floor is a categorical majority across independent
    position-review sidecars: position_thesis, thesis_evolution,
    position_management, and forecast_persistence. This is not a market
    threshold; it prevents one M15 internal structure flip from forcing out an
    H1/H4 swing while most of the current position stack still says the thesis
    is alive. Decay-only thesis_evolution / forecast-persistence closes may opt
    into accepting one strong HOLD source because those paths are
    forecast/expiry disagreement, not hard structural invalidation.
    """

    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return False, ""
    support = sidecars.get("position_hold_support")
    if not isinstance(support, list):
        return False, ""

    matched: list[dict[str, Any]] = []
    for rec in support:
        if not isinstance(rec, dict):
            continue
        rec_trade = str(rec.get("trade_id") or "")
        if trade_id is not None and rec_trade != str(trade_id):
            continue
        rec_pair = str(rec.get("pair") or "")
        if pair and rec_pair not in {"", pair}:
            continue
        rec_side = str(rec.get("side") or "").upper()
        if side and rec_side not in {"", str(side).upper()}:
            continue
        matched.append(rec)

    sources = sorted({str(rec.get("source") or "") for rec in matched if str(rec.get("source") or "")})
    if len(sources) < 2:
        if not allow_single_strong or not sources:
            return False, ""
        strong = {"position_thesis", "position_management", "forecast_persistence"}
        if sources[0] not in strong:
            return False, ""
        summary = ", ".join(
            f"{rec.get('source')}:{rec.get('verdict') or rec.get('status')}"
            for rec in matched
            if rec.get("source")
        )
        return True, f"fresh same-direction {sources[0]} supports HOLD/EXTEND ({summary})"
    summary = ", ".join(
        f"{rec.get('source')}:{rec.get('verdict') or rec.get('status')}"
        for rec in matched
        if rec.get("source")
    )
    return True, f"fresh same-direction position sidecars support HOLD/EXTEND ({summary})"


def _directional_matrix_observations(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        layer = str(item.get("layer") or "").strip()
        if layer in CLOSE_DIRECTIONAL_MATRIX_LAYERS:
            out.append(item)
    return out


def _trader_position_lookup(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map trade_id -> position summary for trader-owned open positions."""
    out: dict[str, dict[str, Any]] = {}
    snapshot = packet.get("broker_snapshot") or {}
    for pos in snapshot.get("position_summaries", []) or []:
        if str(pos.get("owner") or "") != "trader":
            continue
        tid = pos.get("trade_id")
        if tid is None:
            continue
        out[str(tid)] = pos
    return out


def _position_close_spread_override_enabled() -> bool:
    return os.environ.get("QR_POSITION_CLOSE_SPREAD_OVERRIDE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _close_spread_session_tag(packet: dict[str, Any], pair: str) -> str | None:
    market_pairs = (packet.get("market_context") or {}).get("pairs") or {}
    pair_context = market_pairs.get(pair, {})
    if not isinstance(pair_context, dict):
        return None
    chart = pair_context.get("chart")
    if isinstance(chart, dict):
        session = chart.get("session")
        if isinstance(session, dict):
            tag = session.get("current_tag") or session.get("session_current_tag") or session.get("session_bucket")
            if tag:
                return str(tag)
    technical = pair_context.get("technical_context")
    if isinstance(technical, dict):
        tag = technical.get("session_current_tag") or technical.get("session_bucket")
        if tag:
            return str(tag)
    return None


def _close_spread_issues(
    packet: dict[str, Any],
    pair: str,
    *,
    trade_id: str,
) -> tuple["VerificationIssue", ...]:
    if _position_close_spread_override_enabled():
        return ()

    normal_spread = NORMAL_SPREAD_PIPS.get(pair)
    if normal_spread is None:
        return (
            VerificationIssue(
                "POSITION_CLOSE_SPREAD_BASELINE_MISSING",
                f"CLOSE rejected for trade {trade_id} {pair}: missing normal spread baseline",
            ),
        )

    max_spread_multiple = RiskPolicy().max_spread_multiple
    session_tag = _close_spread_session_tag(packet, pair)
    session_mult = _spread_session_multiplier_from_tag(session_tag)
    effective_spread_cap_mult = max_spread_multiple * session_mult
    spread_cap = normal_spread * effective_spread_cap_mult
    issues: list[VerificationIssue] = []

    snapshot = packet.get("broker_snapshot") or {}
    quote = (snapshot.get("quotes") or {}).get(pair)
    if not isinstance(quote, dict):
        issues.append(
            VerificationIssue(
                "POSITION_CLOSE_QUOTE_MISSING",
                f"CLOSE rejected for trade {trade_id} {pair}: missing broker quote",
            )
        )
    else:
        bid = _optional_float(quote.get("bid"))
        ask = _optional_float(quote.get("ask"))
        if bid is None or ask is None:
            issues.append(
                VerificationIssue(
                    "POSITION_CLOSE_QUOTE_MISSING",
                    f"CLOSE rejected for trade {trade_id} {pair}: broker quote lacks bid/ask",
                )
            )
        else:
            spread_pips = abs(ask - bid) * instrument_pip_factor(pair)
            if spread_pips > spread_cap:
                issues.append(
                    VerificationIssue(
                        "POSITION_CLOSE_SPREAD_TOO_WIDE",
                        "CLOSE rejected for trade "
                        f"{trade_id} {pair}: broker quote spread {spread_pips:.2f} pips "
                        f"exceeds close cap {spread_cap:.2f} pips "
                        f"({effective_spread_cap_mult:.2f}x normal {normal_spread:.1f}pip; "
                        f"policy={max_spread_multiple:.1f}x, session={session_tag or 'UNKNOWN'}, "
                        f"session_mult={session_mult:.2f})",
                    )
                )

    market_pairs = (packet.get("market_context") or {}).get("pairs") or {}
    pair_context = market_pairs.get(pair, {})
    flow_spread = (
        ((pair_context.get("flow") or {}).get("spread") or {})
        if isinstance(pair_context, dict)
        else {}
    )
    if isinstance(flow_spread, dict):
        flow_current = _optional_float(flow_spread.get("current_pips"))
        if flow_current is not None and flow_current > spread_cap:
            stress_flag = str(flow_spread.get("stress_flag") or "UNKNOWN")
            issues.append(
                VerificationIssue(
                    "POSITION_CLOSE_FLOW_SPREAD_TOO_WIDE",
                    "CLOSE rejected for trade "
                    f"{trade_id} {pair}: flow spread {flow_current:.2f} pips "
                    f"({stress_flag}) exceeds close cap {spread_cap:.2f} pips "
                    f"({effective_spread_cap_mult:.2f}x normal {normal_spread:.1f}pip; "
                    f"policy={max_spread_multiple:.1f}x, session={session_tag or 'UNKNOWN'}, "
                    f"session_mult={session_mult:.2f})",
                )
            )

    return tuple(issues)


def _close_thesis_invalidated(
    packet: dict[str, Any],
    pair: str,
    side: str,
    *,
    trade_id: str | None = None,
    decision: "GPTTraderDecision | None" = None,
) -> tuple[bool, str]:
    invalidated, reason, _standing_authorized = _close_thesis_invalidation(
        packet,
        pair,
        side,
        trade_id=trade_id,
        decision=decision,
    )
    return invalidated, reason


def _close_thesis_invalidation(
    packet: dict[str, Any],
    pair: str,
    side: str,
    *,
    trade_id: str | None = None,
    decision: "GPTTraderDecision | None" = None,
) -> tuple[bool, str, bool]:
    """Check whether the position's thesis has been invalidated.

    Returns `(invalidated, reason, standing_authorized)`.

    Receipt-level invalidation is hard machine evidence. H4 structure is hard
    only when its timestamped structured event postdates this position's
    broker-open / entry-thesis anchor; legacy, missing-time, or pre-entry H4
    evidence remains soft Gate A. The sidecar path may be hard or soft; soft
    reviews still require explicit operator Gate B.

    Acceptance paths (§6-compliant — no JPY/pip/multiplier literals):

      (a) Structural BOS or CHOCH on M15 or H4 printing AGAINST the
          position side. This is the chart_reader.structure_events lens
          that already drives trader_brain price-action scoring; using
          the same signal keeps prefilter and CLOSE-gate aligned. M15 is
          soft Gate A; H4 carries standing authorization only from an exact
          timestamped structured event that occurred after this trade opened.

      (b) The decision receipt's `invalidation_price` + `invalidation_tf`
          fields are populated AND broker-truth quote has cleared the
          level by the configured anti-wick buffer. LONG invalidates
          downward, SHORT upward. Pure prose `invalidation` text alone
          is not enough — the gate requires a machine-checkable price
          hit beyond the buffer plus chart/technical confirmation.

      (c) A fresh position sidecar generated after the current broker
          snapshot marks this trade `REVIEW_CLOSE` / `RECOMMEND_CLOSE`
          from the prediction stack, thesis evolution, or N-cycle
          forecast persistence. This is the machine-checkable
          "no longer likely to recover to plus" path. `thesis_evolution`
          BROKEN / RECOMMEND_CLOSE is hard only when its rationale contains
          price invalidation plus technical confirmation. Structural standing
          authority is verified from the timestamped H4 / position-management
          paths, never from thesis-evolution prose. `position_thesis` no-ledger/adverse technical-loss
          evidence with multi-TF confirmation may also be hard; softer
          sidecars still require env/token Gate B.
    """
    side_upper = str(side or "").upper()
    if side_upper not in {"LONG", "SHORT"}:
        return False, "unknown position side", False

    chart_story = _pair_chart_story(packet, pair)
    structs = _parse_struct_events(chart_story)
    counter_direction = "DOWN" if side_upper == "LONG" else "UP"
    m15_supported_pullback_reason: str | None = None
    m15_soft_structural_reason: str | None = None
    h4_soft_structural_reason: str | None = None
    for tf in CLOSE_DISCIPLINE_TIMEFRAMES:
        event = structs.get(tf)
        if event and event[1] == counter_direction:
            event_type, direction, price, close_confirmed = event
            if not close_confirmed:
                # Wick-only break (the candle that printed the new pivot
                # closed back inside the prior range). Classic stop-hunt
                # / liquidity-sweep — does NOT authorize Gate A on its
                # own. The structural high/low was tagged but the move
                # was rejected.
                continue
            if tf == "H4" and not _counter_structure_reaches_reward_target(
                packet,
                trade_id=trade_id,
                pair=pair,
                side=side_upper,
                structure_price=price,
            ):
                continue
            if tf == "M15":
                supported, support_reason = _close_same_direction_sidecar_support(
                    packet,
                    trade_id=trade_id,
                    pair=pair,
                    side=side_upper,
                )
                if supported:
                    m15_supported_pullback_reason = (
                        f"M15 {event_type}_{direction}@{price:g} prints against "
                        f"{side_upper}, but {support_reason}"
                    )
                    continue
                soft_blocker = _soft_sidecar_blocks_hard_close_authorization(
                    packet,
                    trade_id=trade_id,
                    pair=pair,
                    side=side_upper,
                )
                reason = (
                    f"M15 {event_type}_{direction}@{price:g} prints against "
                    f"{side_upper} thesis (close-confirmed)"
                )
                if soft_blocker:
                    m15_soft_structural_reason = f"{reason}; {soft_blocker}"
                    continue
                m15_soft_structural_reason = (
                    f"{reason}; M15 structure is Gate A evidence but requires explicit Gate B "
                    "unless H4 structure, recorded invalidation, or a hard sidecar also confirms"
                )
                continue
            post_entry, timing_reason = _h4_structure_postdates_position(
                packet,
                trade_id=trade_id,
                pair=pair,
                side=side_upper,
                event_type=event_type,
                direction=direction,
                price=price,
            )
            structural_reason = (
                f"{tf} {event_type}_{direction}@{price:g} prints against "
                f"{side_upper} thesis (close-confirmed)"
            )
            if post_entry:
                return True, f"{structural_reason}; {timing_reason}", True
            h4_soft_structural_reason = (
                f"{structural_reason}; {timing_reason}; old/unanchored H4 structure "
                "is Gate A evidence but requires explicit Gate B"
            )
            continue

    if m15_supported_pullback_reason:
        return False, m15_supported_pullback_reason, False

    if decision is not None and decision.invalidation_price is not None:
        snapshot = packet.get("broker_snapshot") or {}
        quotes = snapshot.get("quotes") or {}
        quote = quotes.get(pair)
        if isinstance(quote, dict):
            bid = _optional_float(quote.get("bid"))
            ask = _optional_float(quote.get("ask"))
            level = decision.invalidation_price
            tf = decision.invalidation_tf or "unspecified-TF"
            price = bid if side_upper == "LONG" else ask
            label = "bid" if side_upper == "LONG" else "ask"
            reason = invalidation_price_hit_reason(
                pair=pair,
                side=side_upper,
                current_price=price,
                invalidation_price=level,
                price_label=label,
            )
            if reason:
                technical_reason = technical_invalidation_confirmation_reason(
                    _pair_chart(packet, pair),
                    side=side_upper,
                )
                if technical_reason:
                    soft_blocker = _soft_sidecar_blocks_hard_close_authorization(
                        packet,
                        trade_id=trade_id,
                        pair=pair,
                        side=side_upper,
                    )
                    receipt_reason = f"{reason}; {technical_reason} on {tf} per receipt"
                    if soft_blocker:
                        return True, f"{receipt_reason}; {soft_blocker}", False
                    hold_conflict = _same_direction_hold_support_conflict(
                        packet,
                        trade_id=trade_id,
                        pair=pair,
                        side=side_upper,
                        evidence_label="receipt-level invalidation hit",
                    )
                    if hold_conflict:
                        return True, f"{receipt_reason}; {hold_conflict}", False
                    return True, receipt_reason, True

    sidecar_ok, sidecar_reason, sidecar_standing_authorized = _position_sidecar_close_recommended(
        packet,
        trade_id=trade_id,
        pair=pair,
        side=side_upper,
    )
    if sidecar_ok:
        return True, sidecar_reason, sidecar_standing_authorized

    if m15_soft_structural_reason:
        return True, m15_soft_structural_reason, False

    if h4_soft_structural_reason:
        return True, h4_soft_structural_reason, False

    return False, "", False


def _counter_structure_reaches_reward_target(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
    structure_price: float,
) -> bool:
    """Whether a counter-structure event is beyond the current reward target.

    A TP-managed position should not be loss-closed from an old higher-timeframe
    break that sits on the reward side of its still-reachable broker TP. For a
    LONG, a DOWN break above the TP has not invalidated the TP path; for a
    SHORT, an UP break below the TP has not invalidated the TP path. Missing TP
    data falls back to the legacy structural gate because runners have no
    machine-checkable reward-side boundary here.
    """

    pos: dict[str, Any] | None = None
    if trade_id is not None:
        pos = _trader_position_lookup(packet).get(str(trade_id))
    if pos is None:
        for candidate in _trader_position_lookup(packet).values():
            if str(candidate.get("pair") or "") == pair and str(candidate.get("side") or "").upper() == side:
                pos = candidate
                break
    if pos is None:
        return True
    tp = _optional_float(pos.get("take_profit"))
    if tp is None:
        return True
    side_upper = str(side or "").upper()
    if side_upper == "LONG":
        return structure_price < tp
    if side_upper == "SHORT":
        return structure_price > tp
    return True


def _position_sidecar_close_recommended(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> tuple[bool, str, bool]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return False, "", False
    recs = sidecars.get("position_close_recommendations")
    if not isinstance(recs, list):
        return False, "", False
    matched = _matching_position_close_sidecars(recs, trade_id=trade_id, pair=pair, side=side)

    if not matched:
        return False, "", False

    # A trade may appear in multiple fresh sidecars in deterministic order
    # (position_thesis before thesis_evolution). Prefer hard evidence so a soft
    # review cannot mask standing structural loss-cut authorization.
    rec = next((item for item in matched if _sidecar_close_standing_authorized(item)), matched[0])
    source = str(rec.get("source") or "position_sidecar")
    verdict = str(rec.get("verdict") or "RECOMMEND_CLOSE")
    reason = str(rec.get("reason") or "prediction no longer supports recovery")
    rec_trade = str(rec.get("trade_id") or "")
    standing_authorized = _sidecar_close_standing_authorized(rec)
    hold_conflict = _sidecar_hold_support_conflict(packet, rec)
    if standing_authorized and hold_conflict:
        standing_authorized = False
        reason = f"{reason}; {hold_conflict}"
    return (
        True,
        f"{source} {verdict} for trade {rec_trade}: {reason}",
        standing_authorized,
    )


def _sidecar_hold_support_conflict(
    packet: dict[str, Any],
    rec: dict[str, Any],
) -> str | None:
    """Downgrade close evidence when current position stack still supports hold.

    `THESIS_EXPIRED` is meant to stop decayed, unsupported theses. If separate
    fresh position sidecars still support the open side, the problem is
    position geometry/repricing, not a hard unattended loss-cut. The same
    conflict applies to recorded-invalidation sidecars unless the reason
    contains a higher-timeframe structural break; a small invalidation hit while
    current forecasts still support the open side is the 2026-06-15 USD_CAD
    loss-close failure mode.
    """

    source = str(rec.get("source") or "").strip()
    reason = str(rec.get("reason") or "")
    if _sidecar_reason_has_h4_structural_break(reason):
        return None
    upper_reason = reason.upper()
    lower_reason = reason.lower()
    if source == "thesis_evolution":
        conflict_label = (
            "THESIS_EXPIRED"
            if "THESIS_EXPIRED" in upper_reason
            else "thesis_evolution close evidence"
        )
        allow_single_strong = not thesis_evolution_reason_has_hard_close_evidence(
            reason,
            expected_side=str(rec.get("side") or ""),
        )
    elif source == "position_thesis" and (
        "invalidation hit:" in lower_reason
        or "technical invalidation confirmed against" in lower_reason
    ):
        conflict_label = "position_thesis invalidation evidence"
        allow_single_strong = False
    elif (
        source in {"position_management", "position_guardian_management"}
        and "entry thesis invalidation hit" in lower_reason
    ):
        conflict_label = f"{source} entry-invalidation evidence"
        allow_single_strong = False
    elif source == "forecast_persistence":
        conflict_label = "forecast_persistence close evidence"
        allow_single_strong = True
    else:
        return None
    return _same_direction_hold_support_conflict(
        packet,
        trade_id=str(rec.get("trade_id") or "") or None,
        pair=str(rec.get("pair") or ""),
        side=str(rec.get("side") or ""),
        evidence_label=conflict_label,
        allow_single_strong_sidecar=allow_single_strong,
    )


def _same_direction_hold_support_conflict(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
    evidence_label: str,
    allow_single_strong_sidecar: bool = False,
) -> str | None:
    supported, support_reason = _close_same_direction_sidecar_support(
        packet,
        trade_id=trade_id,
        pair=pair,
        side=side,
        allow_single_strong=allow_single_strong_sidecar,
    )
    if supported:
        return (
            f"{evidence_label} is downgraded to soft Gate A because {support_reason}; "
            "use HOLD/reprice/TP rebalance unless hard invalidation evidence is present"
        )
    matrix_supported, matrix_reason = _close_same_direction_matrix_support(packet, pair, side)
    h4_supported, h4_reason = _close_same_direction_h4_support(packet, pair, side)
    if not matrix_supported or not h4_supported:
        return None
    return (
        f"{evidence_label} is downgraded to soft Gate A because {matrix_reason}; {h4_reason}; "
        "use HOLD/reprice/TP rebalance unless hard invalidation evidence is present"
    )


def _close_same_direction_h4_support(
    packet: dict[str, Any],
    pair: str,
    side: str,
) -> tuple[bool, str]:
    side_upper = str(side or "").upper()
    if side_upper not in {"LONG", "SHORT"}:
        return False, ""
    same_direction = "UP" if side_upper == "LONG" else "DOWN"
    event = _parse_struct_events(_pair_chart_story(packet, pair)).get("H4")
    if event is None or event[1] != same_direction:
        return False, ""
    event_type, direction, price, close_confirmed = event
    confirmation = "close-confirmed" if close_confirmed else "wick-only"
    return (
        True,
        f"H4 {event_type}_{direction}@{price:g} still supports {side_upper} ({confirmation})",
    )


def _sidecar_reason_has_h4_structural_break(reason: str) -> bool:
    lowered = str(reason or "").lower()
    if "h4" not in lowered:
        return False
    return any(
        token in lowered
        for token in (
            "bos_",
            "choch_",
            "close-confirmed",
            "structural break",
            "order block",
            "ob broken",
        )
    )


def _sidecar_close_standing_authorized(rec: dict[str, Any]) -> bool:
    """Whether a fresh sidecar is strong enough for standing loss-cut auth.

    `thesis_evolution` can emit BROKEN / RECOMMEND_CLOSE for either a proved
    price/technical invalidation or softer forecast-flip, adverse-drift, and
    horizon-expiry evidence. Only the canonical same-side buffered
    price/technical rationale carries standing authorization; the softer cases
    still need explicit Gate B. `position_management` can carry the deterministic
    PositionManager REVIEW_EXIT into this GPT CLOSE route; it is hard only when
    its own reasons are structural loss-cut reasons. `position_thesis` may also
    hard-authorize a legacy/no-ledger trade, but only when it records a
    machine-checkable invalidation hit or structural break plus multi-TF
    confirmation. Score-only, adverse-buffer-only position-thesis, soft
    position-management, and persistence reviews remain softer and need
    explicit operator Gate B.
    """
    source = str(rec.get("source") or "").strip()
    verdict = str(rec.get("verdict") or "").strip().upper()
    if source == "thesis_evolution" and verdict in {"BROKEN", "RECOMMEND_CLOSE"}:
        return thesis_evolution_reason_has_hard_close_evidence(
            str(rec.get("reason") or ""),
            expected_side=str(rec.get("side") or ""),
        )
    if rec.get("gate_b_standing_authorized") is True:
        return True
    if source == "position_thesis" and verdict == "REVIEW_CLOSE":
        reason = str(rec.get("reason") or "").lower()
        has_technical_confirmation = "technical invalidation confirmed against" in reason
        has_invalidation_hit = "invalidation hit:" in reason
        has_structural_break = _position_thesis_structural_break_text(reason)
        return has_technical_confirmation and (has_invalidation_hit or has_structural_break)
    if source in {"position_management", "position_guardian_management"} and verdict == "REVIEW_EXIT":
        reason = str(rec.get("reason") or "").lower()
        return "close-confirmed structural break" in reason or "structural ob broken" in reason
    return False


def _matching_position_close_sidecars(
    recs: Any,
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> list[dict[str, Any]]:
    if not isinstance(recs, list):
        return []
    matched: list[dict[str, Any]] = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        rec_trade = str(rec.get("trade_id") or "")
        if trade_id is not None and rec_trade != str(trade_id):
            continue
        if pair and str(rec.get("pair") or "") not in {"", pair}:
            continue
        if side and str(rec.get("side") or "").upper() not in {"", str(side).upper()}:
            continue
        matched.append(rec)
    return matched


def _soft_sidecar_blocks_hard_close_authorization(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> str | None:
    """Keep no-ledger/entry-buffer close evidence in the soft Gate-B path.

    M15 internal structure and receipt-level `invalidation_price` can confirm a
    recorded thesis level. They must not launder a fresh soft position_thesis
    fallback (`entry thesis lacks invalidation_price` / `entry-buffer`) into
    standing hard loss-cut authorization.
    """

    sidecars = packet.get("protection_sidecars")
    recs = sidecars.get("position_close_recommendations") if isinstance(sidecars, dict) else None
    matched = _matching_position_close_sidecars(recs, trade_id=trade_id, pair=pair, side=side)
    if not matched:
        return None
    if any(_sidecar_close_standing_authorized(rec) for rec in matched):
        return None
    thesis_context = _matching_entry_thesis_close_context(
        packet,
        trade_id=trade_id,
        pair=pair,
        side=side,
    )
    if thesis_context is not None and not bool(thesis_context.get("has_recorded_invalidation_price")):
        recorded = "recorded" if bool(thesis_context.get("recorded")) else "missing"
        return (
            "matching soft close sidecar has "
            f"{recorded} entry_thesis without a recorded invalidation_price; "
            "receipt-level invalidation_price cannot convert it into standing hard Gate A"
        )
    for rec in matched:
        source = str(rec.get("source") or "").strip()
        reason = str(rec.get("reason") or "").lower()
        if source not in {"position_thesis", "position_management"}:
            continue
        if _entry_buffer_or_unrecorded_invalidation_text(reason):
            return (
                "matching soft close sidecar is entry-buffer / unrecorded-invalidation evidence; "
                "M15/receipt evidence cannot convert it into standing hard Gate A"
            )
    return None


def _matching_entry_thesis_close_context(
    packet: dict[str, Any],
    *,
    trade_id: str | None,
    pair: str,
    side: str,
) -> dict[str, Any] | None:
    sidecars = packet.get("protection_sidecars")
    context_rows = sidecars.get("entry_thesis_close_context") if isinstance(sidecars, dict) else None
    if not isinstance(context_rows, list):
        return None
    for row in context_rows:
        if not isinstance(row, dict):
            continue
        if trade_id is not None and str(row.get("trade_id") or "") != str(trade_id):
            continue
        if pair and str(row.get("pair") or "") not in {"", pair}:
            continue
        if side and str(row.get("side") or "").upper() not in {"", str(side).upper()}:
            continue
        return row
    return None


def _entry_buffer_or_unrecorded_invalidation_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in (
            "entry-buffer",
            "entry thesis lacks invalidation_price",
            "no entry thesis",
            "adverse technical loss",
        )
    )


def _position_thesis_structural_break_text(text: str) -> bool:
    lowered = str(text).lower()
    return any(
        token in lowered
        for token in (
            "structural",
            "close-confirmed",
            "order block",
            "ob broken",
        )
    )

from quant_rabbit.analysis.chart_reader import DEFAULT_TIMEFRAMES as DEFAULT_PAIR_CHART_TIMEFRAMES
from quant_rabbit.guardian_receipt_consumption import (
    BLOCK_NEW_ENTRY_CODE,
    build_guardian_receipt_consumption,
    consumption_status_summary,
    guardian_receipt_new_entry_blockers,
    load_guardian_receipt_consumption,
    write_guardian_receipt_consumption,
)
from quant_rabbit.guardian_receipt_operator_review import (
    load_guardian_receipt_operator_review,
    operator_review_status_summary,
)
from quant_rabbit.paths import (
    DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    DEFAULT_ACTIVE_TRADER_CONTRACT,
    DEFAULT_AI_ATTACK_ADVICE,
    DEFAULT_BROKER_INSTRUMENTS,
    DEFAULT_CAMPAIGN_PLAN,
    DEFAULT_CALENDAR_SNAPSHOT,
    DEFAULT_CAPTURE_ECONOMICS,
    DEFAULT_CODEX_MARKET_READ_OVERLAY,
    DEFAULT_CODEX_TRADER_DECISION_RESPONSE,
    DEFAULT_CONTEXT_ASSET_CHARTS,
    DEFAULT_COVERAGE_OPTIMIZATION,
    DEFAULT_COT_SNAPSHOT,
    DEFAULT_CROSS_ASSET_SNAPSHOT,
    DEFAULT_CURRENCY_STRENGTH,
    DEFAULT_DAILY_TARGET_STATE,
    DEFAULT_EXECUTION_LEDGER_DB,
    DEFAULT_EXECUTION_TIMING_AUDIT,
    DEFAULT_FLOW_SNAPSHOT,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION_REPORT,
    DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    DEFAULT_GPT_TRADER_DECISION,
    DEFAULT_GPT_TRADER_DECISION_REPORT,
    DEFAULT_LEVELS_SNAPSHOT,
    DEFAULT_LEARNING_AUDIT,
    DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
    DEFAULT_MARKET_CONTEXT_MATRIX,
    DEFAULT_MARKET_READ_EVIDENCE_PACKET,
    DEFAULT_MARKET_READ_PREDICTIONS,
    DEFAULT_MARKET_READ_SCORE_REPORT,
    DEFAULT_MARKET_STATUS,
    DEFAULT_MARKET_STORY_PROFILE,
    DEFAULT_NEWS_HEALTH,
    DEFAULT_NEWS_SNAPSHOT,
    DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
    DEFAULT_OPTION_SKEW,
    DEFAULT_OPERATOR_PRECEDENT_AUDIT,
    DEFAULT_ORDER_INTENTS,
    DEFAULT_PAIR_CHARTS,
    DEFAULT_PREDICTIVE_LIMIT_ORDERS,
    DEFAULT_PROJECTION_LEDGER,
    DEFAULT_PROFITABILITY_ACCEPTANCE,
    DEFAULT_QR_TRADER_RUN_WATCHDOG,
    DEFAULT_SELF_IMPROVEMENT_AUDIT,
    DEFAULT_STRATEGY_PROFILE,
    DEFAULT_TRADER_DECISION_DRAFT_REPORT,
    DEFAULT_TRADER_DECISION_BASELINE,
    DEFAULT_TRADER_OVERRIDES,
    DEFAULT_VERIFICATION_LEDGER,
    DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR,
)
from quant_rabbit.instruments import (
    DEFAULT_CONTEXT_ASSETS,
    NORMAL_SPREAD_PIPS,
    instrument_pip_factor,
)
from quant_rabbit.market_close_leak_gate import (
    CLOSE_GATE_PROOF_KEYS,
    CONTAINED_RISK_TIMING_EVIDENCE_KEYS,
    MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE,
    TP_PROVEN_EXCEPTION_KEYS,
    market_close_leak_family_payload_issue,
)
from quant_rabbit.month_scale_residual_gate import month_scale_residual_metadata_issue
from quant_rabbit.predictive_scout import (
    predictive_scout_geometry_claimed,
    predictive_scout_metadata_supported,
)
from quant_rabbit.risk import (
    MARGIN_AWARE_BASKET_BUFFER,
    MIN_PRODUCTION_LOT_UNITS,
    RiskPolicy,
    _spread_session_multiplier_from_tag,
)
from quant_rabbit.self_improvement_guards import (
    FORECAST_ADVERSE_PATH_BLOCKER_CODE,
    forecast_adverse_path_exempted_by_tp_harvest_repair,
    forecast_adverse_path_new_risk_blocker,
    intent_matches_profitability_worst_segment,
    oanda_firepower_repair_current_risk_reaches_minimum,
    p0_code_exempted_by_tp_harvest_repair,
    profitability_p0_worst_segment,
)
from quant_rabbit.strategy.entry_thesis_ledger import (
    invalidation_price_hit_reason,
    technical_invalidation_confirmation_reason,
)


ALLOWED_ACTIONS = ("TRADE", "WAIT", "CANCEL_PENDING", "PROTECT", "TIGHTEN_SL", "CLOSE", "REQUEST_EVIDENCE")
ALLOWED_CONFIDENCE = ("LOW", "MEDIUM", "HIGH")
ALLOWED_METHODS = ("TREND_CONTINUATION", "RANGE_ROTATION", "BREAKOUT_FAILURE", "EVENT_RISK", "POSITION_MANAGEMENT")
ALLOWED_SPECIALIST_ROLES = ("macro_news", "indicator", "flow_levels", "risk_audit", "strategy", "portfolio_context")
OPERATOR_PRECEDENT_EVIDENCE_REF = "operator:precedent"
MANUAL_MARKET_CONTEXT_EVIDENCE_REF = "manual:market_context"
PROFITABILITY_ACCEPTANCE_REF_ALIASES = {
    # Historical receipts occasionally swapped "replay" and "repair" in this
    # finding name. Keep the alias explicit so UNKNOWN_EVIDENCE_REF still
    # rejects unrelated invented profitability refs.
    "TP_PROGRESS_REPLAY_REPAIR_UNPROVED": ("TP_PROGRESS_REPAIR_REPLAY_UNPROVED",),
}
FORBIDDEN_SPECIALIST_AUTHORITY_FIELDS = (
    "action",
    "selected_lane_id",
    "selected_lane_ids",
    "cancel_order_ids",
    "units",
    "tp",
    "sl",
    "entry",
    "risk_budget_jpy",
    "daily_risk_budget_jpy",
    "per_trade_risk_budget_jpy",
    "max_loss_jpy",
    "size_multiple",
    "capital_allocation",
    "selected_units",
    "allocation_board_sha256",
    "stage_order",
    "send_order",
    "confirm_live",
)
# Matches the CLI generate-intents breadth used by the scheduled trader. The
# verifier also keeps every LIVE_READY lane even when a smaller cap is passed,
# because the operator may cite any executable lane visible in order_intents.
DEFAULT_GPT_MAX_LANES = 56

# Maximum distinct pairs a verifier-accepted basket should cover when
# ai_attack_advice has recommended lanes across multiple pairs and the
# campaign target is open. Mirrors RiskPolicy.max_portfolio_positions; the
# gateway still re-validates portfolio risk and margin before send.
BASKET_PAIR_COVERAGE_TARGET = 4

# Rank ceiling for "primary attack" lanes when computing basket pair
# coverage. ai_attack_advice sorts recommended_now_lane_ids by descending
# score, so the top N ranks represent the highest-conviction setups for
# the cycle. Pairs whose first appearance in the advised list is below
# this rank are treated as low-conviction repair candidates rather than
# primary-basket coverage requirements; the rank gap is the deterministic
# conviction gate that satisfies AGENT_CONTRACT §5–§6 without forcing the
# trader to paste boilerplate skip-rationale per lower-ranked pair. This
# keeps the bot-grinding defense (single low-conviction lane spam) while
# unblocking high-conviction concentrated attacks (per
# feedback_high_conviction_execution.md).
PRIMARY_ATTACK_RANK_CEILING = 4

# The scheduled trader cadence is one deeper operator decision every 60
# minutes. This is an operational receipt horizon, not a market threshold, JPY
# cap, pip distance, or reward/risk multiplier. If scheduler cadence changes,
# replace this with scheduler config rather than tuning it from trade outcomes.
TRADER_DECISION_HORIZON_MINUTES = 60
ENTRY_DECISION_HORIZON_ACTIONS = ("TRADE", "WAIT", "REQUEST_EVIDENCE")
TWENTY_MINUTE_PLAN_TEXT_FIELDS = (
    "primary_path",
    "failure_path",
    "entry_or_hold_trigger",
    "invalidation_or_cancel_trigger",
    "counterargument",
    "next_cycle_check",
)
SESSION_ONLY_WAIT_PATTERN = re.compile(
    r"(WAIT|PATIENCE|STAY\s+FLAT|HOLD).{0,100}"
    r"(LONDON|NEW\s+YORK|\bNY\b|ASIA|ASIAN|TOKYO|OFF[\s_-]*HOURS|SESSION|KILLZONE)"
    r"|"
    r"(QUIET|THIN|LOW[\s_-]*LIQUIDITY).{0,60}"
    r"(SESSION|ASIA|ASIAN|TOKYO|OFF[\s_-]*HOURS|KILLZONE)"
    r"|"
    r"(SESSION|ASIA|ASIAN|TOKYO|OFF[\s_-]*HOURS|KILLZONE).{0,60}"
    r"(QUIET|THIN|LOW[\s_-]*LIQUIDITY|WAIT|PATIENCE|STAY\s+FLAT)",
    re.IGNORECASE,
)
CONCRETE_WAIT_GATE_PATTERN = re.compile(
    r"\b("
    r"SPREAD|FORECAST|NEWS|EVENT|CPI|FOMC|NFP|RATE|CONFLICT|INVALIDATION|"
    r"MARGIN|REWARD|RR|LIVE_READY|LIVE\s+READY|BOS|CHOCH|ATR|"
    r"VOLATILITY|STRUCTURE|SHELF|BREAK|BROKEN|SUPPORT|RESISTANCE|RETEST|"
    r"PENDING|CAPACITY|TP|SL"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GPTTraderDecision:
    generated_at_utc: str | None
    market_read_first: dict[str, Any]
    action: str
    selected_lane_id: str | None
    selected_lane_ids: tuple[str, ...]
    cancel_order_ids: tuple[str, ...]
    confidence: str
    thesis: str
    method: str
    narrative: str
    chart_story: str
    invalidation: str
    rejected_alternatives: tuple[str, ...]
    risk_notes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    operator_summary: str
    twenty_minute_plan: dict[str, Any] | None = None
    # The scheduled Codex model may replace MARKET_READ_FIRST through the
    # content-addressed overlay tool. It may also veto a deterministic TRADE
    # to WAIT / REQUEST_EVIDENCE or reduce the deterministic lane to a bounded
    # capital-allocation multiple. It cannot upgrade a non-trade receipt,
    # select another lane, exceed the baseline units, expand risk, or grant
    # live permission.
    decision_provenance: dict[str, Any] | None = None
    capital_allocation: dict[str, Any] | None = None
    market_read_review: dict[str, Any] | None = None
    market_read_counterargument: str = ""
    market_read_change_summary: str = ""
    market_read_disposition: str = ""
    market_read_veto_reason: str = ""
    market_read_vetoed_lane_ids: tuple[str, ...] = ()
    # Operator-directed market close on existing trader-owned positions.
    # Used only with action="CLOSE". A loss-cut and a new entry must still
    # have separate receipts: automation ends the close cycle, then the next
    # scheduled cycle must refresh broker truth and require a fresh verified
    # TRADE receipt before any re-entry.
    close_trade_ids: tuple[str, ...] = ()
    strategy_reviews: tuple[dict[str, Any], ...] = ()
    specialist_reviews: tuple[dict[str, Any], ...] = ()
    # CLOSE-action discipline fields (added 2026-05-12, see
    # `feedback_no_unilateral_close.md`; Gate B split 2026-06-04). A CLOSE
    # receipt must pass Gate A market evidence plus the applicable Gate B:
    # hard Gate A carries standing loss-cut authorization, while soft Gate A
    # requires explicit env/token authorization. The `operator_close_authorized`
    # field remains audit text only and is not accepted as authorization,
    # because the trader can set fields in its own receipt.
    invalidation_price: float | None = None
    invalidation_tf: str | None = None
    operator_close_authorized: bool = False
    payload_field_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    severity: str = "BLOCK"


@dataclass(frozen=True)
class CloseGateEvidence:
    trade_id: str
    pair: str
    side: str
    unrealized_pl_jpy: float | None
    loss_side_close: bool
    gate_a_invalidated: bool
    gate_a_reason: str
    gate_b_standing_authorized: bool
    gate_b_explicit_operator_authorized: bool
    explicit_gate_b_required: bool
    profitability_p0_context_required: bool
    profitability_p0_context_cited: bool
    timing_audit_required: bool
    timing_evidence_cited: bool
    hard_timing_gate_required: bool
    same_direction_support_conflict: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    allowed: bool
    issues: tuple[VerificationIssue, ...]
    close_gate_evidence: tuple[CloseGateEvidence, ...] = ()


@dataclass(frozen=True)
class GPTTraderSummary:
    status: str
    output_path: Path
    report_path: Path
    action: str | None
    selected_lane_id: str | None
    selected_lane_ids: tuple[str, ...]
    cancel_order_ids: tuple[str, ...]
    allowed: bool
    issues: int
    close_trade_ids: tuple[str, ...] = ()
    capital_allocation_size_multiple: float | None = None


@dataclass(frozen=True)
class TraderDecisionDraftSummary:
    status: str
    output_path: Path
    report_path: Path
    action: str
    selected_lane_id: str | None
    selected_lane_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    verification_allowed: bool
    verification_issues: tuple[str, ...]


MARKET_READ_BLOCKER_FIELDS = (
    "thesis",
    "narrative",
    "chart_story",
    "invalidation",
    "rejected_alternatives",
    "risk_notes",
    "operator_summary",
)
MARKET_READ_BLOCKER_PATTERN = re.compile(
    r"\b("
    r"BLOCKER|BLOCKED|NO[_\s-]*LIVE[_\s-]*READY|LIVE[_\s-]*READY\s*[=:]\s*0|"
    r"NEGATIVE[_\s-]*EXPECTANCY|EXPOSURE[_\s-]*BLOCK|ENTRY[_\s-]*THESIS|"
    r"NEWS[_\s-]*HEALTH|PROJECTION[_\s-]*LEDGER|SELF[_\s-]*IMPROVEMENT|"
    r"WAIT\s+REJECTED|TRADE\s+REJECTED"
    r")\b",
    re.IGNORECASE,
)
MARKET_READ_LIVE_READY_ZERO_PATTERN = re.compile(
    r"\b(NO[_\s-]*LIVE[_\s-]*READY|LIVE[_\s-]*READY\s*[=:]\s*0)\b",
    re.IGNORECASE,
)
MARKET_READ_NEGATIVE_EXPECTANCY_PATTERN = re.compile(
    r"\bNEGATIVE[_\s-]*EXPECTANCY\b",
    re.IGNORECASE,
)
MARKET_READ_PREDICTION_REF_PATTERN = re.compile(
    r"\b("
    r"30\s*M|30\s*MIN|NEXT[_\s-]*30|NEXT\s+THIRTY|"
    r"2\s*H|2\s*HR|2\s*HOUR|NEXT[_\s-]*2|NEXT\s+TWO"
    r")\b",
    re.IGNORECASE,
)


def _market_read_missing_fields(market_read: dict[str, Any]) -> list[str]:
    return market_read_missing_fields(market_read)


def _decision_contract_text(decision: GPTTraderDecision) -> str:
    values: list[str] = [
        decision.thesis,
        decision.narrative,
        decision.chart_story,
        decision.invalidation,
        decision.operator_summary,
    ]
    values.extend(decision.rejected_alternatives)
    values.extend(decision.risk_notes)
    return "\n".join(str(value or "") for value in values)


def _decision_references_market_prediction(decision: GPTTraderDecision) -> bool:
    return bool(MARKET_READ_PREDICTION_REF_PATTERN.search(_decision_contract_text(decision)))


def _blocker_field_precedes_market_read(decision: GPTTraderDecision) -> str | None:
    field_order = list(decision.payload_field_order)
    if not field_order or "market_read_first" not in field_order:
        return None
    market_read_index = field_order.index("market_read_first")
    for field_name in MARKET_READ_BLOCKER_FIELDS:
        if field_name not in field_order or field_order.index(field_name) > market_read_index:
            continue
        value = getattr(decision, field_name)
        text = "\n".join(str(item) for item in value) if isinstance(value, tuple) else str(value or "")
        if MARKET_READ_BLOCKER_PATTERN.search(text):
            return field_name
    return None


class TraderModelProvider(Protocol):
    def decide(self, input_packet: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]: ...


class StaticTraderProvider:
    def __init__(self, decision: dict[str, Any], *, source_path: Path | None = None) -> None:
        self.decision = decision
        self.source_path = source_path

    def decide(self, input_packet: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        return dict(self.decision)


class GPTTraderBrain:
    """Build a broker-truth packet and verify a Codex-created decision receipt."""

    def __init__(
        self,
        *,
        provider: TraderModelProvider | None = None,
        intents_path: Path = DEFAULT_ORDER_INTENTS,
        campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
        strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
        market_story_profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
        market_status_path: Path = DEFAULT_MARKET_STATUS,
        target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
        pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
        context_asset_charts_path: Path = DEFAULT_CONTEXT_ASSET_CHARTS,
        broker_instruments_path: Path = DEFAULT_BROKER_INSTRUMENTS,
        cross_asset_path: Path = DEFAULT_CROSS_ASSET_SNAPSHOT,
        flow_path: Path = DEFAULT_FLOW_SNAPSHOT,
        currency_strength_path: Path = DEFAULT_CURRENCY_STRENGTH,
        levels_path: Path = DEFAULT_LEVELS_SNAPSHOT,
        market_context_matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX,
        calendar_path: Path = DEFAULT_CALENDAR_SNAPSHOT,
        cot_path: Path = DEFAULT_COT_SNAPSHOT,
        option_skew_path: Path = DEFAULT_OPTION_SKEW,
        attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
        capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
        profitability_acceptance_path: Path = DEFAULT_PROFITABILITY_ACCEPTANCE,
        execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
        coverage_optimization_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
        learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
        verification_ledger_path: Path = DEFAULT_VERIFICATION_LEDGER,
        self_improvement_audit_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
        projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
        operator_precedent_path: Path = DEFAULT_OPERATOR_PRECEDENT_AUDIT,
        manual_market_context_path: Path = DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
        trader_overrides_path: Path = DEFAULT_TRADER_OVERRIDES,
        predictive_limits_path: Path = DEFAULT_PREDICTIVE_LIMIT_ORDERS,
        news_items_path: Path = DEFAULT_NEWS_SNAPSHOT,
        news_health_path: Path = DEFAULT_NEWS_HEALTH,
        qr_trader_run_watchdog_path: Path = DEFAULT_QR_TRADER_RUN_WATCHDOG,
        guardian_receipt_consumption_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
        guardian_receipt_operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
        active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
        active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
        non_eurusd_live_grade_frontier_path: Path = DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
        range_rail_geometry_repair_path: Path = DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR,
        output_path: Path = DEFAULT_GPT_TRADER_DECISION,
        report_path: Path = DEFAULT_GPT_TRADER_DECISION_REPORT,
        market_read_baseline_path: Path = DEFAULT_TRADER_DECISION_BASELINE,
        market_read_evidence_packet_path: Path = DEFAULT_MARKET_READ_EVIDENCE_PACKET,
        market_read_overlay_path: Path = DEFAULT_CODEX_MARKET_READ_OVERLAY,
        market_read_artifact_validation_required: bool = True,
        market_read_evidence_source_overrides: Mapping[str, Path] | None = None,
        market_read_predictions_path: Path | None = None,
        market_read_score_report_path: Path | None = None,
        execution_ledger_path: Path | None = None,
        max_lanes: int = DEFAULT_GPT_MAX_LANES,
    ) -> None:
        self.provider = provider
        self.intents_path = intents_path
        self.campaign_plan_path = campaign_plan_path
        self.strategy_profile_path = strategy_profile_path
        self.market_story_profile_path = market_story_profile_path
        self.market_status_path = market_status_path
        self.target_state_path = target_state_path
        self.pair_charts_path = pair_charts_path
        self.context_asset_charts_path = context_asset_charts_path
        self.broker_instruments_path = broker_instruments_path
        self.cross_asset_path = cross_asset_path
        self.flow_path = flow_path
        self.currency_strength_path = currency_strength_path
        self.levels_path = levels_path
        self.market_context_matrix_path = market_context_matrix_path
        self.calendar_path = calendar_path
        self.cot_path = cot_path
        self.option_skew_path = option_skew_path
        self.attack_advice_path = attack_advice_path
        self.capture_economics_path = capture_economics_path
        self.profitability_acceptance_path = profitability_acceptance_path
        self.execution_timing_audit_path = execution_timing_audit_path
        self.coverage_optimization_path = coverage_optimization_path
        self.learning_audit_path = learning_audit_path
        self.verification_ledger_path = verification_ledger_path
        self.self_improvement_audit_path = self_improvement_audit_path
        self.projection_ledger_path = projection_ledger_path
        self.operator_precedent_path = operator_precedent_path
        self.manual_market_context_path = manual_market_context_path
        self.trader_overrides_path = trader_overrides_path
        self.predictive_limits_path = predictive_limits_path
        self.news_items_path = news_items_path
        self.news_health_path = news_health_path
        self.active_trader_contract_path = active_trader_contract_path
        self.active_opportunity_board_path = active_opportunity_board_path
        self.non_eurusd_live_grade_frontier_path = non_eurusd_live_grade_frontier_path
        self.range_rail_geometry_repair_path = range_rail_geometry_repair_path
        self.output_path = output_path
        self.report_path = report_path
        self.market_read_baseline_path = (
            market_read_baseline_path
            if market_read_baseline_path != DEFAULT_TRADER_DECISION_BASELINE
            or output_path == DEFAULT_GPT_TRADER_DECISION
            else output_path.parent / DEFAULT_TRADER_DECISION_BASELINE.name
        )
        self.market_read_evidence_packet_path = (
            market_read_evidence_packet_path
            if market_read_evidence_packet_path != DEFAULT_MARKET_READ_EVIDENCE_PACKET
            or output_path == DEFAULT_GPT_TRADER_DECISION
            else output_path.parent / DEFAULT_MARKET_READ_EVIDENCE_PACKET.name
        )
        self.market_read_overlay_path = (
            market_read_overlay_path
            if market_read_overlay_path != DEFAULT_CODEX_MARKET_READ_OVERLAY
            or output_path == DEFAULT_GPT_TRADER_DECISION
            else output_path.parent / DEFAULT_CODEX_MARKET_READ_OVERLAY.name
        )
        self.market_read_artifact_validation_required = market_read_artifact_validation_required
        self.market_read_evidence_source_overrides = (
            {
                str(name): Path(path)
                for name, path in market_read_evidence_source_overrides.items()
            }
            if market_read_evidence_source_overrides is not None
            else None
        )
        self.qr_trader_run_watchdog_path = (
            qr_trader_run_watchdog_path
            if qr_trader_run_watchdog_path != DEFAULT_QR_TRADER_RUN_WATCHDOG
            or output_path == DEFAULT_GPT_TRADER_DECISION
            else output_path.parent / DEFAULT_QR_TRADER_RUN_WATCHDOG.name
        )
        self.guardian_receipt_consumption_path = (
            guardian_receipt_consumption_path
            if guardian_receipt_consumption_path != DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION
            or output_path == DEFAULT_GPT_TRADER_DECISION
            else output_path.parent / DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION.name
        )
        self.guardian_receipt_operator_review_path = (
            guardian_receipt_operator_review_path
            if guardian_receipt_operator_review_path != DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW
            or output_path == DEFAULT_GPT_TRADER_DECISION
            else output_path.parent / DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW.name
        )
        self.market_read_predictions_path = (
            market_read_predictions_path
            if market_read_predictions_path is not None
            else output_path.parent / DEFAULT_MARKET_READ_PREDICTIONS.name
        )
        self.market_read_score_report_path = (
            market_read_score_report_path
            if market_read_score_report_path is not None
            else report_path.parent / DEFAULT_MARKET_READ_SCORE_REPORT.name
        )
        self.execution_ledger_path = (
            execution_ledger_path
            if execution_ledger_path is not None
            else (
                DEFAULT_EXECUTION_LEDGER_DB
                if output_path == DEFAULT_GPT_TRADER_DECISION
                else output_path.parent / DEFAULT_EXECUTION_LEDGER_DB.name
            )
        )
        self.max_lanes = max_lanes

    def run(self, *, snapshot_path: Path) -> GPTTraderSummary:
        generated_at = datetime.now(timezone.utc).isoformat()
        packet = self._input_packet(snapshot_path)
        if self.provider is None:
            raise RuntimeError("Codex GPT verifier requires a decision response JSON")
        raw_decision = self.provider.decide(packet, GPT_TRADER_SCHEMA)
        decision = _decision_from_payload(raw_decision)
        artifact_issues: list[tuple[str, str]] = []
        if self.market_read_artifact_validation_required and decision.action in {
            "TRADE",
            "CLOSE",
            "WAIT",
            "REQUEST_EVIDENCE",
        }:
            artifact_issues = revalidate_codex_market_read_artifacts(
                final_payload=raw_decision,
                baseline_path=self.market_read_baseline_path,
                packet_path=self.market_read_evidence_packet_path,
                overlay_path=self.market_read_overlay_path,
                evidence_sources=self._market_read_evidence_sources(snapshot_path),
            )
        verification = DecisionVerifier(
            packet,
            market_read_artifact_issues=artifact_issues,
        ).verify(decision)
        status = "ACCEPTED" if verification.allowed else "REJECTED"
        market_read_prediction = record_market_read_prediction(
            decision,
            packet,
            status=status,
            issues=verification.issues,
            predictions_path=self.market_read_predictions_path,
            report_path=self.market_read_score_report_path,
            pair_charts_path=self.pair_charts_path,
            execution_ledger_path=self.execution_ledger_path,
            now=datetime.now(timezone.utc),
        )
        result = {
            "generated_at_utc": generated_at,
            "status": status,
            "decision": asdict(decision),
            "verification_issues": [asdict(issue) for issue in verification.issues],
            "close_gate_evidence": [asdict(item) for item in verification.close_gate_evidence],
            "market_read_prediction": market_read_prediction,
            "input_packet": packet,
        }
        self._write_result(result)
        self._write_report(result)
        return GPTTraderSummary(
            status=status,
            output_path=self.output_path,
            report_path=self.report_path,
            action=decision.action,
            selected_lane_id=decision.selected_lane_id,
            selected_lane_ids=decision.selected_lane_ids,
            cancel_order_ids=decision.cancel_order_ids,
            allowed=verification.allowed,
            issues=len(verification.issues),
            close_trade_ids=decision.close_trade_ids,
            capital_allocation_size_multiple=_capital_allocation_size_multiple(
                decision.capital_allocation
            ),
        )

    def _market_read_evidence_sources(self, snapshot_path: Path) -> dict[str, Path]:
        """Return the exact source map used by the content-addressed handoff."""

        configured = {
            "broker_snapshot": snapshot_path,
            "order_intents": self.intents_path,
            "campaign_plan": self.campaign_plan_path,
            "strategy_profile": self.strategy_profile_path,
            "market_story_profile": self.market_story_profile_path,
            "market_status": self.market_status_path,
            "daily_target_state": self.target_state_path,
            "pair_charts": self.pair_charts_path,
            "context_asset_charts": self.context_asset_charts_path,
            "broker_instruments": self.broker_instruments_path,
            "cross_asset": self.cross_asset_path,
            "flow": self.flow_path,
            "currency_strength": self.currency_strength_path,
            "levels": self.levels_path,
            "market_context_matrix": self.market_context_matrix_path,
            "calendar": self.calendar_path,
            "cot": self.cot_path,
            "option_skew": self.option_skew_path,
            "attack_advice": self.attack_advice_path,
            "capture_economics": self.capture_economics_path,
            "execution_timing_audit": self.execution_timing_audit_path,
            "coverage_optimization": self.coverage_optimization_path,
            "learning_audit": self.learning_audit_path,
            "verification_ledger": self.verification_ledger_path,
            "self_improvement_audit": self.self_improvement_audit_path,
            "projection_ledger": self.projection_ledger_path,
            "operator_precedent": self.operator_precedent_path,
            "manual_market_context": self.manual_market_context_path,
            "profitability_acceptance": self.profitability_acceptance_path,
            "news_items": self.news_items_path,
            "news_health": self.news_health_path,
            "trader_overrides": self.trader_overrides_path,
            "predictive_limits": self.predictive_limits_path,
            "qr_trader_run_watchdog": self.qr_trader_run_watchdog_path,
            "guardian_receipt_consumption": self.guardian_receipt_consumption_path,
            "guardian_receipt_operator_review": self.guardian_receipt_operator_review_path,
            "active_trader_contract": self.active_trader_contract_path,
            "active_opportunity_board": self.active_opportunity_board_path,
            "non_eurusd_live_grade_frontier": self.non_eurusd_live_grade_frontier_path,
            "range_rail_geometry_repair": self.range_rail_geometry_repair_path,
            "execution_ledger": self.execution_ledger_path,
            "market_read_predictions": self.market_read_predictions_path,
        }
        # The evidence packet records the complete source surface used by the
        # draft command.  Standalone verifier CLI flags expose only a subset;
        # preserve the packet's remaining exact paths, while explicit current
        # runtime paths override them and are re-hashed below.
        packet_sources: dict[str, Path] = {}
        try:
            packet_payload = json.loads(self.market_read_evidence_packet_path.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            packet_payload = {}
        raw_paths = packet_payload.get("source_paths") if isinstance(packet_payload, dict) else None
        if isinstance(raw_paths, dict):
            packet_sources = {
                str(name): Path(path)
                for name, path in raw_paths.items()
                if isinstance(name, str) and isinstance(path, str) and path
            }
        authoritative_sources = (
            configured
            if self.market_read_evidence_source_overrides is None
            else dict(self.market_read_evidence_source_overrides)
        )
        # Broker truth passed to this verifier is never packet-controlled.  A
        # copied fresh snapshot named by a forged packet must not mask the
        # stale/different --snapshot that built the actual verification input.
        authoritative_sources["broker_snapshot"] = snapshot_path
        if not packet_sources:
            packet_sources = dict(configured)
        # Packet paths only fill sources the caller cannot configure.  Every
        # explicit/configured source above wins and is re-hashed.
        packet_sources.update(authoritative_sources)
        return packet_sources

    def _market_read_artifact_contract(self, snapshot_path: Path) -> dict[str, Any]:
        sources = self._market_read_evidence_sources(snapshot_path)
        return {
            "contract": "CODEX_MARKET_READ_ARTIFACT_BINDING_V1",
            "required_for_action": "TRADE_CLOSE_OR_VETO",
            "required_for_actions": ["TRADE", "CLOSE", "WAIT", "REQUEST_EVIDENCE"],
            "validation_required": self.market_read_artifact_validation_required,
            "baseline_path": str(self.market_read_baseline_path),
            "evidence_packet_path": str(self.market_read_evidence_packet_path),
            "overlay_path": str(self.market_read_overlay_path),
            "evidence_source_paths": {
                name: str(path) for name, path in sorted(sources.items())
            },
            "read_only": True,
            "live_permission": False,
            "may_grant_gateway_permission": False,
        }

    def _input_packet(self, snapshot_path: Path) -> dict[str, Any]:
        snapshot = _load_json(snapshot_path)
        intents = _load_json(self.intents_path)
        campaign = _load_json(self.campaign_plan_path)
        strategy = _load_json(self.strategy_profile_path)
        story = _load_json(self.market_story_profile_path)
        market_status = _load_optional_json(self.market_status_path)
        target = _load_json(self.target_state_path) if self.target_state_path.exists() else {}
        lanes = _lane_packet(intents, campaign, strategy, story, max_lanes=self.max_lanes)
        attack_advice = _load_optional_json(self.attack_advice_path)
        capture_economics = _load_optional_json(self.capture_economics_path)
        profitability_acceptance = _load_optional_json(self.profitability_acceptance_path)
        execution_timing_audit = _load_optional_json(self.execution_timing_audit_path)
        coverage_optimization = _load_optional_json(self.coverage_optimization_path)
        learning_audit = _load_optional_json(self.learning_audit_path)
        verification_ledger = _load_optional_json(self.verification_ledger_path)
        self_improvement_audit = _load_optional_json(self.self_improvement_audit_path)
        projection_ledger = _projection_ledger_packet(self.projection_ledger_path)
        operator_precedent = _load_optional_json(self.operator_precedent_path)
        manual_market_context = _load_optional_json(self.manual_market_context_path)
        trader_overrides = _load_optional_json(self.trader_overrides_path)
        user_alpha_continuation = _user_alpha_continuation_from_overrides(trader_overrides)
        predictive_limits = _load_optional_json(self.predictive_limits_path)
        market_context_matrix = _load_optional_json(self.market_context_matrix_path)
        option_skew = _load_optional_json(self.option_skew_path)
        news_items = _load_optional_json(self.news_items_path)
        news_health = _load_optional_json(self.news_health_path)
        qr_trader_run_watchdog = _load_optional_json(self.qr_trader_run_watchdog_path)
        active_trader_contract = _load_optional_json(self.active_trader_contract_path)
        active_opportunity_board = _load_optional_json(self.active_opportunity_board_path)
        non_eurusd_frontier = _load_optional_json(self.non_eurusd_live_grade_frontier_path)
        range_rail_geometry_repair = _load_optional_json(self.range_rail_geometry_repair_path)
        market_read_measurement_refresh = refresh_market_read_measurements(
            predictions_path=self.market_read_predictions_path,
            report_path=self.market_read_score_report_path,
            pair_charts_path=self.pair_charts_path,
            execution_ledger_path=self.execution_ledger_path,
        )
        market_read_feedback = market_read_feedback_summary(self.market_read_predictions_path)
        market_read_feedback["measurement_refresh"] = market_read_measurement_refresh
        active_path = _active_path_packet(
            active_trader_contract=active_trader_contract,
            active_opportunity_board=active_opportunity_board,
            non_eurusd_frontier=non_eurusd_frontier,
            range_rail_geometry_repair=range_rail_geometry_repair,
        )
        guardian_receipt_consumption = consumption_status_summary(
            load_guardian_receipt_consumption(self.guardian_receipt_consumption_path)
        )
        guardian_receipt_operator_review = operator_review_status_summary(
            load_guardian_receipt_operator_review(self.guardian_receipt_operator_review_path)
        )
        pair_set = set(_pairs_from_lanes_and_positions(lanes, snapshot))
        # A no-entry cycle still needs quote-relative 30m/2h forecasts. Do not
        # erase all quote bases merely because no lane or position survived the
        # deterministic filter; retain the broker snapshot's observed pairs for
        # the read-only market analysis and veto artifact.
        raw_quotes = snapshot.get("quotes") if isinstance(snapshot, dict) else {}
        if isinstance(raw_quotes, dict):
            pair_set.update(
                str(pair).strip()
                for pair in raw_quotes
                if isinstance(pair, str) and str(pair).strip()
            )
        for active_lane in _active_path_candidate_lanes(active_path):
            active_pair = str(active_lane.get("pair") or "").strip()
            if active_pair:
                pair_set.add(active_pair)
        pairs = tuple(sorted(pair_set))
        currencies = _currencies_from_pairs(pairs)
        refs = _allowed_refs(
            snapshot=snapshot,
            target=target,
            lanes=lanes,
            attack_advice=attack_advice,
            capture_economics=capture_economics,
            profitability_acceptance=profitability_acceptance,
            execution_timing_audit=execution_timing_audit,
            coverage_optimization=coverage_optimization,
            learning_audit=learning_audit,
            verification_ledger=verification_ledger,
            self_improvement_audit=self_improvement_audit,
            projection_ledger=projection_ledger,
            operator_precedent=operator_precedent,
            manual_market_context=manual_market_context,
            predictive_limits=predictive_limits,
            market_status=market_status,
            market_context_matrix=market_context_matrix,
            option_skew=option_skew,
            news_items=news_items,
            news_health=news_health,
        )
        if user_alpha_continuation.get("active"):
            refs.extend(_user_alpha_evidence_refs(user_alpha_continuation))
        refs.extend(_active_path_evidence_refs(active_path))
        market_read_feedback_ref = str(market_read_feedback.get("evidence_ref") or "").strip()
        if market_read_feedback_ref and market_read_feedback_ref not in refs:
            refs.append(market_read_feedback_ref)
        attack_packet = _attack_advice_packet(attack_advice)
        learning_packet = _learning_audit_packet(learning_audit)
        return {
            "contract": {
                "allowed_actions": list(ALLOWED_ACTIONS),
                "entry_decision_actions": list(ENTRY_DECISION_HORIZON_ACTIONS + ("CANCEL_PENDING",)),
                "trade_requires_live_ready_lane": True,
                "trade_may_select_multiple_live_ready_lanes": False,
                "entry_decisions_require_twenty_minute_plan": True,
                "decision_horizon_minutes": TRADER_DECISION_HORIZON_MINUTES,
                "pending_entries_are_basket_counted_by_gateway": True,
                "protected_trader_position_adds_require_portfolio_validation": True,
                "model_output_is_advisory_until_verified": True,
                "strategy_reviews_must_use_lane_id_not_desk_alias": True,
                "predictive_limits_are_advisory_timing_evidence": True,
                "tp_rebalance_sidecar_blocks_wait": True,
                "entry_thesis_blocker_blocks_trade_and_wait": True,
                "learning_audit_blocks_unsafe_learning_influence": True,
                "verification_ledger_is_read_only_structured_evidence": True,
                "self_improvement_p0_blocks_trade_except_verified_repair_lanes": True,
                "market_status_is_authoritative_calendar_evidence": True,
                "coverage_optimization_is_read_only_gap_evidence": True,
                "operator_precedent_is_advisory_only": True,
                "manual_market_context_gates_only_precedent_usage": True,
                "user_alpha_requires_continuation_or_exact_blocker": True,
                "market_read_feedback_is_read_only_advisory": True,
                "soft_close_advisory_non_blocking": (
                    "position_close_recommendations include blocks_non_close_actions; "
                    "when false, do not choose CLOSE from that advisory on the entry branch "
                    "unless explicit operator Gate B is present"
                ),
            },
            "decision_requirements": {
                "learning_influenced_lane_evidence": _learning_influenced_lane_evidence_requirements(
                    attack_packet,
                    learning_packet,
                ),
            },
            "market_read_artifact_contract": self._market_read_artifact_contract(
                snapshot_path
            ),
            "artifact_timestamps": {
                "daily_target_generated_at_utc": (
                    target.get("generated_at_utc") if isinstance(target, dict) else None
                ),
                "campaign_plan_generated_at_utc": (
                    campaign.get("generated_at_utc") if isinstance(campaign, dict) else None
                ),
                "order_intents_generated_at_utc": intents.get("generated_at_utc"),
                "ai_attack_advice_generated_at_utc": (
                    attack_advice.get("generated_at_utc") if isinstance(attack_advice, dict) else None
                ),
                "market_context_matrix_generated_at_utc": (
                    market_context_matrix.get("generated_at_utc")
                    if isinstance(market_context_matrix, dict)
                    else None
                ),
                "operator_precedent_generated_at_utc": (
                    operator_precedent.get("generated_at_utc")
                    if isinstance(operator_precedent, dict)
                    else None
                ),
                "manual_market_context_generated_at_utc": (
                    manual_market_context.get("generated_at_utc")
                    if isinstance(manual_market_context, dict)
                    else None
                ),
                "trader_overrides_expires_at_utc": (
                    trader_overrides.get("expires_at_utc") if isinstance(trader_overrides, dict) else None
                ),
            },
            "broker_snapshot": _snapshot_packet(snapshot, relevant_pairs=pairs),
            "daily_target": _target_packet(target),
            "lanes": lanes,
            "ai_attack_advice": attack_packet,
            "capture_economics": _capture_economics_packet(capture_economics),
            "profitability_acceptance": _profitability_acceptance_packet(profitability_acceptance),
            "execution_timing_audit": _execution_timing_audit_packet(execution_timing_audit),
            "coverage_optimization": _coverage_optimization_packet(coverage_optimization),
            "learning_audit": learning_packet,
            "verification_ledger": _verification_ledger_packet(verification_ledger),
            "self_improvement_audit": _self_improvement_audit_packet(self_improvement_audit),
            "projection_ledger": projection_ledger,
            "operator_precedent": _operator_precedent_packet(operator_precedent),
            "manual_market_context": _manual_market_context_packet(manual_market_context),
            "user_alpha_continuation": user_alpha_continuation,
            "predictive_limits": _predictive_limits_packet(predictive_limits, pairs=pairs),
            "news": _news_packet(news_items, news_health, pairs=pairs, currencies=currencies),
            "qr_trader_run_watchdog": _qr_trader_run_watchdog_packet(qr_trader_run_watchdog),
            "guardian_receipt_consumption": guardian_receipt_consumption,
            "guardian_receipt_operator_review": guardian_receipt_operator_review,
            "active_path": active_path,
            "market_read_feedback": market_read_feedback,
            "market_status": _market_status_packet(market_status),
            "protection_sidecars": _protection_sidecars_packet(
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                pair_charts_path=self.pair_charts_path,
            ),
            "market_context": _market_context_packet(
                pairs=pairs,
                currencies=currencies,
                pair_charts_path=self.pair_charts_path,
                context_asset_charts_path=self.context_asset_charts_path,
                broker_instruments_path=self.broker_instruments_path,
                cross_asset_path=self.cross_asset_path,
                flow_path=self.flow_path,
                currency_strength_path=self.currency_strength_path,
                levels_path=self.levels_path,
                market_context_matrix_path=self.market_context_matrix_path,
                calendar_path=self.calendar_path,
                cot_path=self.cot_path,
                option_skew_path=self.option_skew_path,
            ),
            "allowed_evidence_refs": refs,
        }

    def _write_result(self, result: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _write_report(self, result: dict[str, Any]) -> None:
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        decision = result["decision"]
        provenance = (
            decision.get("decision_provenance")
            if isinstance(decision.get("decision_provenance"), dict)
            else {}
        )
        lines = [
            "# GPT Trader Decision Report",
            "",
            f"- Generated at UTC: `{result['generated_at_utc']}`",
            f"- Status: `{result['status']}`",
            f"- Action: `{decision.get('action')}`",
            f"- Selected lane: `{decision.get('selected_lane_id')}`",
            f"- Selected basket lanes: `{', '.join(decision.get('selected_lane_ids') or []) or 'none'}`",
            f"- Cancel order ids: `{', '.join(decision.get('cancel_order_ids') or []) or 'none'}`",
            f"- Confidence: `{decision.get('confidence')}`",
            f"- Decision author: `{provenance.get('author_kind') or 'unattributed'}`",
            f"- Market-read disposition: `{decision.get('market_read_disposition') or 'none'}`",
            f"- Market read first: `{decision.get('market_read_first') or 'missing'}`",
            f"- 20m plan: `{decision.get('twenty_minute_plan') or 'missing'}`",
            f"- Specialist reviews: `{len(decision.get('specialist_reviews') or [])}`",
            f"- Market read prediction: `{result.get('market_read_prediction') or 'not recorded'}`",
            f"- Operator summary: {decision.get('operator_summary')}",
        ]
        input_packet = result.get("input_packet") if isinstance(result.get("input_packet"), dict) else {}
        lines.extend(_user_alpha_report_lines(input_packet.get("user_alpha_continuation"), decision=decision))
        lines.extend(["", "## Verification Issues", ""])
        issues = result.get("verification_issues", [])
        if issues:
            for issue in issues:
                lines.append(f"- `{issue['severity']}` {issue['code']}: {issue['message']}")
        else:
            lines.append("- none")
        lines.extend(["", "## Close Gate Evidence", ""])
        close_gate_evidence = result.get("close_gate_evidence", [])
        if close_gate_evidence:
            for item in close_gate_evidence:
                gate_b = "standing" if item.get("gate_b_standing_authorized") else (
                    "explicit" if item.get("gate_b_explicit_operator_authorized") else "missing/soft"
                )
                lines.append(
                    "- "
                    f"`{item.get('trade_id')}` {item.get('pair')} {item.get('side')} "
                    f"GateA={item.get('gate_a_invalidated')} GateB={gate_b} "
                    f"loss_side={item.get('loss_side_close')} "
                    f"reason={item.get('gate_a_reason') or 'none'}"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "## Decision Contract",
                "",
                "- `decision_provenance.author_kind` identifies the decision author; deterministic verification remains the execution gate.",
                "- A fresh schema-v2 `CODEX_MARKET_READ` overlay may accept the deterministic lane at 0.5/0.75/1.0 of its risk-capped units, or veto a baseline `TRADE` to `WAIT` / `REQUEST_EVIDENCE`. It cannot upgrade a non-trade baseline, choose another lane, exceed baseline units, expand risk, or grant live permission.",
                "- `TRADE` requires known `LIVE_READY` lane(s); pending entries are counted by gateway basket validation.",
                "- `TRADE`/`CANCEL_PENDING` cancel ids must be current trader-owned pending entry orders from broker truth.",
                "- Current `ai_attack_advice` recommendations make generic WAIT invalid while the daily target is open, but never grant live permission.",
                "- Learning may only rank already-live-ready lanes. Any learning-influenced selected lane must be covered by a non-blocked `learning_audit` packet and cite `learning:audit` plus `learning:lane:<lane_id>`.",
                "- Active `USER_ALPHA` / `OPERATOR_ALPHA` continuation must cite `user_alpha:continuation` and either continue the same pair/side as RELOAD / SECOND_SHOT / 5% path-board candidate, or name an exact blocker plus next trigger.",
                "- `TRADE` must cite current chart evidence plus `news:health` and `news:items` or `news:current`; blocked news-health is a no-trade gate.",
                f"- `TRADE`, `WAIT`, and `REQUEST_EVIDENCE` receipts must include `twenty_minute_plan` with `horizon_minutes={TRADER_DECISION_HORIZON_MINUTES}`: the next-cycle primary path, failure path, trigger, invalidation/cancel trigger, strongest counterargument, next-cycle check, and known packet refs. This is a receipt-depth gate, not a new market-risk gate.",
                "- `market_status` is deterministic calendar/session evidence only; broker truth still decides prices, positions, and tradability.",
                "- A deterministic `tp-rebalance` sidecar requirement makes WAIT / REQUEST_EVIDENCE invalid until the sidecar is run.",
                "- A deterministic entry-thesis blocker makes TRADE / WAIT invalid until the unverifiable active position is repaired or reviewed.",
                "- Any self-improvement P0 blocks new `TRADE` receipts until the named blocker is repaired or the trader route explicitly justifies the exception.",
                "- The 2025 operator precedent is advisory only and pair-agnostic. A `TRADE` that cites `operator:precedent` must also cite `manual:market_context`, at least one selected lane must match the generalized trade-shape aligned lane set, and that selected lane must not conflict with the bounded manual technical replay buckets; otherwise the receipt must use current deterministic edge instead of precedent-based aggression.",
                "- Evidence refs must come from the input packet; invented refs reject the decision.",
                "- `CLOSE` requires Gate A plus the applicable Gate B. Hard Gate A (an exact timestamped H4 close-confirmed BOS/CHOCH against side that postdates this trade's broker-open / entry-thesis anchor, buffered invalidation_price hit with technical confirmation, fresh thesis_evolution BROKEN/RECOMMEND_CLOSE whose rationale contains that invalidation/technical proof, structural position_management / position_guardian_management REVIEW_EXIT, or position_thesis invalidation-hit/structural-break evidence with multi-TF confirmation) carries standing loss-cut authorization only when it has not been downgraded by fresh same-direction HOLD/EXTEND sidecars. Thesis-evolution structural prose is not market proof. A pre-entry, missing-time, mismatched, or future-dated H4 event and forecast flip, adverse drift, confidence/regime decay, or THESIS_EXPIRED are soft Gate A by themselves. M15 structure is Gate A evidence but not unattended hard Gate B unless post-entry H4 structure, recorded invalidation, or a hard sidecar also confirms; M15 internal structure or receipt-level `invalidation_price` cannot harden a matching soft entry-buffer / unrecorded-invalidation sidecar. `protection_sidecars.position_close_recommendations[].blocks_non_close_actions=false` means the sidecar is advisory for entry routing: do not write CLOSE merely to test the verifier; evaluate current LIVE_READY entries unless a current hard close sidecar separately blocks non-close actions. Softer Gate A still needs `QR_OPERATOR_CLOSE_OVERRIDE=1` or a fresh `data/.operator_close_token` when the trader chooses CLOSE, but operator Gate B does not override fresh same-direction HOLD/EXTEND support. If the same-direction market stack still supports the open position, treat it as TP rebalance / HOLD / profit-side partial / ADD geometry, not loss-side CLOSE plus same-direction re-entry. `TRADE` must not include `close_trade_ids`; automation ends the close cycle, then the next scheduled cycle must refresh broker truth, reprice intents, and require a separate verified `TRADE` receipt. The receipt's `operator_close_authorized` field is advisory only. See AGENT_CONTRACT §10.",
            ]
        )
        self.report_path.write_text("\n".join(lines) + "\n")


def _decision_execution_envelope_sha256(decision: GPTTraderDecision) -> str:
    """Recreate the canonical final envelope from fields present in the receipt."""

    values = asdict(decision)
    values.pop("payload_field_order", None)
    raw_payload = {
        key: values[key]
        for key in decision.payload_field_order
        if key in values
    }
    return canonical_json_sha256(execution_envelope_payload(raw_payload))


def _codex_market_read_veto_active(decision: GPTTraderDecision) -> bool:
    provenance = decision.decision_provenance
    if not isinstance(provenance, dict):
        return False
    if provenance.get("author_kind") != CODEX_MARKET_READ_AUTHOR:
        return False
    if decision.action not in {"WAIT", "REQUEST_EVIDENCE"}:
        return False
    if decision.selected_lane_id is not None or decision.selected_lane_ids:
        return False
    expected_disposition = (
        "VETO_WAIT" if decision.action == "WAIT" else "VETO_REQUEST_EVIDENCE"
    )
    if decision.market_read_disposition != expected_disposition:
        return False
    provenance_issues = validate_codex_market_read_provenance(
        action=decision.action,
        market_read=decision.market_read_first,
        provenance=provenance,
        review=decision.market_read_review,
        counterargument=decision.market_read_counterargument,
        change_summary=decision.market_read_change_summary,
        disposition=decision.market_read_disposition,
        veto_reason=decision.market_read_veto_reason,
        vetoed_lane_ids=decision.market_read_vetoed_lane_ids,
        capital_allocation=decision.capital_allocation,
        execution_envelope_sha256=_decision_execution_envelope_sha256(decision),
    )
    return not provenance_issues


class DecisionVerifier:
    def __init__(
        self,
        input_packet: dict[str, Any],
        *,
        market_read_artifact_issues: Iterable[tuple[str, str]] = (),
    ) -> None:
        self.packet = input_packet
        self.lanes = {str(lane["lane_id"]): lane for lane in input_packet.get("lanes", [])}
        self.allowed_refs = set(str(ref) for ref in input_packet.get("allowed_evidence_refs", []))
        self.close_gate_evidence: list[CloseGateEvidence] = []
        self.market_read_artifact_issues = tuple(market_read_artifact_issues)

    def verify(self, decision: GPTTraderDecision) -> VerificationResult:
        issues: list[VerificationIssue] = []
        self._verify_decision_freshness(decision, issues)
        if decision.action not in ALLOWED_ACTIONS:
            issues.append(VerificationIssue("BAD_ACTION", f"unsupported action {decision.action!r}"))
        if decision.confidence not in ALLOWED_CONFIDENCE:
            issues.append(VerificationIssue("BAD_CONFIDENCE", f"unsupported confidence {decision.confidence!r}"))
        if decision.method not in ALLOWED_METHODS:
            issues.append(VerificationIssue("BAD_METHOD", f"unsupported method {decision.method!r}"))
        if not decision.evidence_refs:
            issues.append(VerificationIssue("MISSING_EVIDENCE_REFS", "decision must cite packet evidence refs"))
        unknown_refs = sorted(set(decision.evidence_refs) - self.allowed_refs)
        if unknown_refs:
            issues.append(VerificationIssue("UNKNOWN_EVIDENCE_REF", f"unknown evidence refs: {', '.join(unknown_refs)}"))
        self._verify_market_read_first(decision, issues)
        self._verify_strategy_reviews(decision, issues)
        self._verify_specialist_reviews(decision, issues)
        self._verify_twenty_minute_plan(decision, issues)

        broker = self.packet.get("broker_snapshot", {})
        positions = int(broker.get("positions") or 0)
        selected_lane_ids = _selected_trade_lane_ids(decision)
        primary_lane_id = decision.selected_lane_id or (selected_lane_ids[0] if selected_lane_ids else None)
        tradeable_lanes = _tradeable_live_ready_lanes(self.packet)
        pace_trade_lanes = _pace_trade_lanes(self.packet, tradeable_lanes)
        attack_lane_ids = _attack_recommended_tradeable_lane_ids(self.packet, tradeable_lanes)
        codex_market_read_veto = _codex_market_read_veto_active(decision)
        exposure_blockers = _trade_exposure_blockers(self.packet)
        entry_thesis_blockers = _entry_thesis_sidecar_reasons(self.packet)
        position_close_reasons = _position_close_sidecar_reasons(self.packet)
        projection_trade_blockers = _projection_ledger_trade_blockers(self.packet)
        news_trade_blockers = _news_health_trade_blockers(self.packet)
        guardian_receipt_trade_blockers = _guardian_receipt_consumption_trade_blockers(self.packet)
        self._verify_user_alpha_continuation(decision, selected_lane_ids, issues)
        self_improvement_trade_blockers = _self_improvement_trade_blockers(
            self.packet,
            decision_generated_at_utc=decision.generated_at_utc,
            resolved_pending_cancel_order_ids=decision.cancel_order_ids
            if decision.action == "TRADE"
            else (),
            selected_lane_ids=selected_lane_ids,
        )
        self_improvement_entry_blockers = _self_improvement_trade_blockers(
            self.packet,
            decision_generated_at_utc=decision.generated_at_utc,
            include_decision_history_stale=False,
            selected_lane_ids=tradeable_lanes,
        )

        if decision.action == "TRADE":
            provenance_lane_ids = (
                decision.decision_provenance.get("baseline_selected_lane_ids")
                if isinstance(decision.decision_provenance, dict)
                and decision.decision_provenance.get("author_kind")
                == CODEX_MARKET_READ_AUTHOR
                else None
            )
            exact_lane_binding = (
                "selected_lane_ids" in decision.payload_field_order
                and len(decision.selected_lane_ids) == 1
                and bool(decision.selected_lane_id)
                and decision.selected_lane_ids[0] == decision.selected_lane_id
                and (
                    provenance_lane_ids is None
                    or provenance_lane_ids == [decision.selected_lane_id]
                )
            )
            if not exact_lane_binding:
                issues.append(
                    VerificationIssue(
                        "AI_MARKET_READ_SINGLE_LANE_REQUIRED",
                        "artifact-bound TRADE requires exactly one selected_lane_ids item equal to "
                        "selected_lane_id and the same deterministic baseline lane",
                    )
                )
            if not selected_lane_ids:
                issues.append(VerificationIssue("LANE_REQUIRED", "TRADE requires selected_lane_id or selected_lane_ids"))
            selected_scout_lane_ids = [
                lane_id
                for lane_id in selected_lane_ids
                if _lane_is_predictive_scout(self.lanes.get(lane_id))
            ]
            if selected_scout_lane_ids and (
                len(selected_scout_lane_ids) != 1 or len(selected_lane_ids) != 1
            ):
                issues.append(
                    VerificationIssue(
                        "PREDICTIVE_SCOUT_SINGLE_LANE_ONLY",
                        "a predictive SCOUT is one bounded forward experiment and cannot be mixed with another lane or basket",
                    )
                )
            for scout_lane_id in selected_scout_lane_ids:
                issues.extend(
                    _predictive_scout_lane_verification_issues(
                        self.lanes.get(scout_lane_id)
                    )
                )
            if decision.close_trade_ids:
                issues.append(
                    VerificationIssue(
                        "CLOSE_REENTRY_SAME_RECEIPT",
                        "TRADE must not include close_trade_ids. Loss-cut first with action=CLOSE, then "
                        "end the close cycle, rerun broker-snapshot / intents on the next scheduled cycle, and re-enter only from a "
                        "separate verified TRADE receipt if a fresh LIVE_READY lane survives.",
                    )
                )
            if position_close_reasons:
                _append_position_close_required_issue(
                    issues,
                    packet=self.packet,
                    action="TRADE",
                    reasons=position_close_reasons,
                )
            if exposure_blockers:
                issues.append(VerificationIssue("EXPOSURE_BLOCKS_TRADE", "; ".join(exposure_blockers[:3])))
            if entry_thesis_blockers:
                issues.append(
                    VerificationIssue(
                        "ENTRY_THESIS_REPAIR_REQUIRED",
                        "TRADE rejected while active trader position(s) have unverifiable entry thesis: "
                        + "; ".join(entry_thesis_blockers[:3]),
                    )
                )
            if self_improvement_trade_blockers:
                issues.append(
                    VerificationIssue(
                        "SELF_IMPROVEMENT_P0_BLOCKS_TRADE",
                        "TRADE rejected while self-improvement audit carries P0 blocker(s): "
                        + "; ".join(self_improvement_trade_blockers[:3]),
                    )
                )
            if projection_trade_blockers:
                issues.append(
                    VerificationIssue(
                        "PROJECTION_LEDGER_EXPIRED_PENDING_BLOCKS_TRADE",
                        "TRADE rejected while projection_ledger has expired PENDING forecast telemetry: "
                        + "; ".join(projection_trade_blockers[:3]),
                    )
                )
            if news_trade_blockers:
                issues.append(
                    VerificationIssue(
                        "NEWS_HEALTH_BLOCKS_TRADE",
                        "TRADE rejected while current news / market-story freshness is blocked: "
                        + "; ".join(news_trade_blockers[:3]),
                    )
                )
            if guardian_receipt_trade_blockers:
                issues.append(
                    VerificationIssue(
                        _guardian_receipt_blocker_issue_code(guardian_receipt_trade_blockers),
                        "TRADE rejected because guardian receipt consumption blocks normal new-entry routing: "
                        + "; ".join(guardian_receipt_trade_blockers[:3]),
                    )
                )
            issues.extend(_learning_audit_trade_issues(self.packet, selected_lane_ids, decision.evidence_refs))
            issues.extend(_manual_precedent_trade_issues(self.packet, selected_lane_ids, decision.evidence_refs))
            if _target_requires_entry(self.packet) and attack_lane_ids and not exposure_blockers:
                selected_attack_lanes = [lane_id for lane_id in selected_lane_ids if lane_id in attack_lane_ids]
                if not selected_attack_lanes:
                    issues.append(
                        VerificationIssue(
                            "ATTACK_ADVICE_IGNORED",
                            "ai_attack_advice recommends current tradeable LIVE_READY lane(s); the selected "
                            "basket must include at least one recommended lane or the advice must be regenerated: "
                            f"{', '.join(attack_lane_ids[:3])}",
                        )
                    )
                else:
                    priority_lane_id = attack_lane_ids[0]
                    if priority_lane_id not in selected_lane_ids:
                        issues.append(
                            VerificationIssue(
                                "ATTACK_PRIORITY_SKIPPED",
                                "ai_attack_advice ranks current tradeable LIVE_READY lanes in execution order; "
                                "the selected basket must include the first-ranked lane instead of skipping "
                                f"to lower-ranked advice: {priority_lane_id}",
                            )
                        )
                    advised_pairs = []
                    for advised_lane_id in attack_lane_ids:
                        pair = _pair_from_lane_id(advised_lane_id)
                        if pair and pair not in advised_pairs:
                            advised_pairs.append(pair)
                    # Restrict primary basket coverage to pairs whose top-ranked
                    # advised lane sits within the rank ceiling. attack_lane_ids
                    # is already sorted by descending score, so a pair whose
                    # first appearance is rank > ceiling has been ranked below
                    # PRIMARY_ATTACK_RANK_CEILING higher-conviction lanes — the
                    # rank gap itself is the deterministic conviction gate, and
                    # bot-grinding the lower-ranked pair would defeat the very
                    # ranking ai_attack_advice exists to express.
                    primary_advised_pairs: list[str] = []
                    for rank, advised_lane_id in enumerate(attack_lane_ids):
                        if rank >= PRIMARY_ATTACK_RANK_CEILING:
                            break
                        pair = _pair_from_lane_id(advised_lane_id)
                        if pair and pair not in primary_advised_pairs:
                            primary_advised_pairs.append(pair)
                    selected_pairs: set[str] = set()
                    for chosen_lane_id in selected_lane_ids:
                        pair = _pair_from_lane_id(chosen_lane_id)
                        if pair:
                            selected_pairs.add(pair)
                    expected_basket_pairs = min(
                        len(primary_advised_pairs),
                        BASKET_PAIR_COVERAGE_TARGET,
                    )
                    if len(selected_pairs) < expected_basket_pairs:
                        skipped_pairs = [
                            pair for pair in primary_advised_pairs if pair not in selected_pairs
                        ][:expected_basket_pairs]
                        issues.append(
                            VerificationIssue(
                                "BASKET_PAIR_COVERAGE_INCOMPLETE",
                                "ai_attack_advice recommends top-ranked tradeable LIVE_READY lanes across "
                                f"{len(primary_advised_pairs)} primary pair(s) "
                                f"({', '.join(primary_advised_pairs)}); basket only covers "
                                f"{len(selected_pairs) or 'no'} pair(s) "
                                f"({', '.join(sorted(selected_pairs)) or 'none'}). Per AGENT_CONTRACT "
                                "§5–§6 campaign exposure occupancy: include one lane per primary "
                                "advised pair (those whose top-ranked lane is within rank "
                                f"{PRIMARY_ATTACK_RANK_CEILING}), or cite a named deterministic gate "
                                f"per skipped primary pair in risk_notes: {', '.join(skipped_pairs)}. "
                                "Pairs whose top advised lane ranks lower than the ceiling are gated "
                                "by the rank/conviction gap itself.",
                                severity="WARN",
                            )
                        )
                    if "attack:advice" not in decision.evidence_refs:
                        issues.append(
                            VerificationIssue(
                                "ATTACK_ADVICE_EVIDENCE_MISSING",
                                "TRADE selecting an ai_attack_advice recommended lane must cite attack:advice",
                            )
                        )
                    for lane_id in selected_attack_lanes:
                        if f"attack:lane:{lane_id}" not in decision.evidence_refs:
                            issues.append(
                                VerificationIssue(
                                    "ATTACK_ADVICE_LANE_EVIDENCE_MISSING",
                                    f"TRADE selecting recommended lane {lane_id} must cite attack:lane:{lane_id}",
                                )
                            )
            if decision.selected_lane_id and decision.selected_lane_id not in selected_lane_ids:
                issues.append(
                    VerificationIssue(
                        "PRIMARY_LANE_NOT_IN_BASKET",
                        "selected_lane_id must also appear in selected_lane_ids",
                    )
                )
            for selected_lane_id in selected_lane_ids:
                selected_lane = self.lanes.get(selected_lane_id)
                if selected_lane is None:
                    issues.append(VerificationIssue("UNKNOWN_LANE", f"selected lane is not in packet: {selected_lane_id}"))
                    continue
                if selected_lane.get("status") != "LIVE_READY":
                    issues.append(VerificationIssue("LANE_NOT_LIVE_READY", f"{selected_lane_id} status is {selected_lane.get('status')}"))
                if selected_lane_id == primary_lane_id and selected_lane.get("method") != decision.method:
                    issues.append(VerificationIssue("METHOD_MISMATCH", "decision method does not match selected primary lane"))
                if selected_lane.get("risk_blockers") or selected_lane.get("strategy_blockers") or selected_lane.get("live_blockers"):
                    issues.append(VerificationIssue("LANE_HAS_BLOCKERS", f"{selected_lane_id} still carries blockers"))
                market_close_leak_issue = market_close_leak_family_payload_issue(selected_lane)
                if market_close_leak_issue is not None:
                    issues.append(
                        VerificationIssue(
                            MARKET_CLOSE_LEAK_FAMILY_BLOCK_CODE,
                            str(market_close_leak_issue.get("message") or ""),
                        )
                    )
                residual_family_issue = month_scale_residual_metadata_issue(selected_lane)
                if residual_family_issue is not None:
                    issues.append(
                        VerificationIssue(
                            str(residual_family_issue.get("code") or ""),
                            str(residual_family_issue.get("message") or ""),
                        )
                    )
                forecast_issue = _lane_forecast_direction_issue(selected_lane)
                if forecast_issue is not None:
                    issues.append(forecast_issue)
                if str(selected_lane.get("evidence_ref") or "") not in decision.evidence_refs:
                    issues.append(
                        VerificationIssue(
                            "SELECTED_LANE_EVIDENCE_MISSING",
                            f"selected lane {selected_lane_id} must be cited in evidence_refs",
                        )
                    )
            for field_name, value in (
                ("thesis", decision.thesis),
                ("narrative", decision.narrative),
                ("chart_story", decision.chart_story),
                ("invalidation", decision.invalidation),
            ):
                if not value.strip():
                    issues.append(VerificationIssue("INCOMPLETE_TRADE_DECISION", f"TRADE missing {field_name}"))
            if "news:health" not in decision.evidence_refs:
                issues.append(
                    VerificationIssue(
                        "NEWS_HEALTH_EVIDENCE_MISSING",
                        "TRADE must cite news:health so the scheduled trader proves it checked news freshness.",
                    )
                )
            if "news:items" not in decision.evidence_refs and "news:current" not in decision.evidence_refs:
                issues.append(
                    VerificationIssue(
                        "NEWS_ITEMS_EVIDENCE_MISSING",
                        "TRADE must cite news:items or news:current so macro/news context is part of the receipt.",
                    )
                )
            self._verify_cancel_order_ids(decision, issues, action="TRADE")
        elif decision.action in {"WAIT", "REQUEST_EVIDENCE"}:
            if decision.selected_lane_id is not None:
                issues.append(VerificationIssue("WAIT_SELECTED_LANE", f"{decision.action} must not select a lane"))
            if decision.action == "WAIT" and _wait_is_session_only(decision):
                issues.append(
                    VerificationIssue(
                        "SESSION_ONLY_WAIT_REJECTED",
                        "WAIT cannot be justified by time-of-day, quiet-session, or London/NY timing alone; "
                        "cite a current spread, forecast, structure, event, broker-truth, close, or risk gate.",
                    )
                )
            if decision.action == "WAIT" and entry_thesis_blockers:
                issues.append(
                    VerificationIssue(
                        "ENTRY_THESIS_REPAIR_REQUIRED",
                        "WAIT rejected while active trader position(s) have unverifiable entry thesis: "
                        + "; ".join(entry_thesis_blockers[:3]),
                    )
                )
            tp_rebalance_reasons = _tp_rebalance_sidecar_reasons(self.packet)
            if tp_rebalance_reasons:
                issues.append(
                    VerificationIssue(
                        "TP_REBALANCE_REQUIRED",
                        "WAIT / REQUEST_EVIDENCE cannot complete while deterministic tp-rebalance has "
                        f"executable adjustment(s): {tp_rebalance_reasons[0]}",
                    )
                )
            if position_close_reasons:
                _append_position_close_required_issue(
                    issues,
                    packet=self.packet,
                    action=decision.action,
                    reasons=position_close_reasons,
                )
            pending_cancel_reasons = _self_improvement_pending_cancel_review_reasons(
                self.packet
            )
            if pending_cancel_reasons:
                issues.append(
                    VerificationIssue(
                        "SELF_IMPROVEMENT_PENDING_CANCEL_REVIEW_REQUIRED",
                        "WAIT / REQUEST_EVIDENCE rejected while self-improvement audit requires "
                        "a pending-entry cancel review. Write CANCEL_PENDING for the named broker "
                        "order ids, or TRADE with cancel_order_ids when replacing them with a "
                        "current verified basket: "
                        + "; ".join(pending_cancel_reasons[:3]),
                    )
                )
            if projection_trade_blockers and not _decision_cites_projection_evidence(decision.evidence_refs):
                issues.append(
                    VerificationIssue(
                        "PROJECTION_LEDGER_EVIDENCE_MISSING",
                        "WAIT / REQUEST_EVIDENCE may pause new risk for expired projection telemetry only when "
                        "the receipt cites projection:ledger, projection:expired_pending, or the relevant "
                        "projection:expired_pending:<pair> evidence ref.",
                    )
                )
            if (
                not position_close_reasons
                and _target_requires_entry(self.packet)
                and not exposure_blockers
                and not self_improvement_entry_blockers
                and not projection_trade_blockers
                and not news_trade_blockers
                and attack_lane_ids
                and not codex_market_read_veto
            ):
                issues.append(
                    VerificationIssue(
                        "ATTACK_ADVICE_REQUIRES_TRADE",
                        "ai_attack_advice recommends current tradeable LIVE_READY lane(s) while the daily target "
                        "is still open. Protected trader exposure is not a no-trade gate; choose TRADE or rerun "
                        f"the advice after a named hard blocker fires: {', '.join(attack_lane_ids[:3])}",
                    )
                )
            if (
                not position_close_reasons
                and _target_requires_entry(self.packet)
                and not exposure_blockers
                and not self_improvement_entry_blockers
                and not projection_trade_blockers
                and not news_trade_blockers
                and pace_trade_lanes
                and not codex_market_read_veto
            ):
                if not _trader_exposure_present(self.packet):
                    issues.append(
                        VerificationIssue(
                            "CAMPAIGN_EXPOSURE_REQUIRED",
                            "rolling 30d pace is open, no trader-owned position or pending entry is active, "
                            "and A/S or attack-recommended LIVE_READY lanes exist; choose TRADE instead of "
                            f"leaving the campaign flat: {', '.join(pace_trade_lanes[:3])}",
                        )
                    )
                cited_live_ready = _cited_live_ready_lanes(decision, pace_trade_lanes)
                if decision.action == "REQUEST_EVIDENCE":
                    issues.append(
                        VerificationIssue(
                            "REQUEST_EVIDENCE_WITH_LIVE_READY_LANES",
                            "REQUEST_EVIDENCE is stale or contradictory because the packet already contains "
                            f"A/S or attack-recommended LIVE_READY lanes: {', '.join(pace_trade_lanes[:3])}",
                        )
                    )
                elif not cited_live_ready:
                    issues.append(
                        VerificationIssue(
                            "WAIT_MISSING_LIVE_READY_REJECTION",
                            "WAIT must cite at least one current A/S or attack-recommended LIVE_READY lane "
                            "evidence ref when clean pace lanes exist and the rolling target is still open",
                        )
                    )
        elif decision.action == "CANCEL_PENDING":
            if decision.selected_lane_id is not None:
                issues.append(VerificationIssue("CANCEL_SELECTED_LANE", "CANCEL_PENDING must not select a trade lane"))
            if not _pending_entry_order_ids(self.packet):
                issues.append(VerificationIssue("NO_PENDING_ENTRY", "CANCEL_PENDING requires a pending entry order"))
            if not decision.cancel_order_ids:
                issues.append(
                    VerificationIssue(
                        "MISSING_CANCEL_ORDER_IDS",
                        "CANCEL_PENDING must name the pending entry order ids to cancel",
                    )
                )
            self._verify_cancel_order_ids(decision, issues, action="CANCEL_PENDING")
            if (
                _target_requires_entry(self.packet)
                and not position_close_reasons
                and not exposure_blockers
                and not tradeable_lanes
            ):
                visible_thesis = _cancel_pending_visible_current_theses(
                    self.packet,
                    decision.cancel_order_ids,
                    exempt_order_ids=_self_improvement_pending_cancel_review_order_ids(
                        self.packet
                    ),
                )
                if visible_thesis:
                    issues.append(
                        VerificationIssue(
                            "CANCEL_PENDING_CURRENT_THESIS_VISIBLE",
                            "CANCEL_PENDING must not clear a broker-anchored pending entry while a current "
                            "same-pair/same-side thesis remains visible in order_intents. Preserve the pending "
                            "entry unless the current packet removes the candidate, or replace it through a "
                            "TRADE receipt with cancel_order_ids: "
                            + "; ".join(visible_thesis[:3]),
                        )
                    )
            if (
                _target_requires_entry(self.packet)
                and not position_close_reasons
                and not exposure_blockers
                and tradeable_lanes
                and not _all_tradeable_lanes_blocked_by_learning_audit(self.packet, tradeable_lanes)
            ):
                issues.append(
                    VerificationIssue(
                        "CANCEL_PENDING_WITH_LIVE_READY_LANES",
                        "CANCEL_PENDING is stale or contradictory because the packet already contains "
                        "tradeable LIVE_READY lane(s). If pending entries must be retired while the daily "
                        "target is still open, choose TRADE with selected lane(s) and optional cancel_order_ids "
                        f"in the same receipt: {', '.join(tradeable_lanes[:3])}",
                    )
                )
        elif decision.action in {"PROTECT", "TIGHTEN_SL", "CLOSE"}:
            if positions <= 0:
                issues.append(VerificationIssue("NO_OPEN_POSITION", f"{decision.action} requires an open position"))
            if decision.action in {"PROTECT", "TIGHTEN_SL"} and position_close_reasons:
                _append_position_close_required_issue(
                    issues,
                    packet=self.packet,
                    action=decision.action,
                    reasons=position_close_reasons,
                )
            if decision.action == "CLOSE":
                if decision.selected_lane_id is not None or decision.selected_lane_ids:
                    issues.append(
                        VerificationIssue(
                            "CLOSE_SELECTED_LANE_FORBIDDEN",
                            "CLOSE is close-only and must not select an entry lane",
                        )
                    )
                if decision.cancel_order_ids:
                    issues.append(
                        VerificationIssue(
                            "CLOSE_CANCEL_ORDER_IDS_FORBIDDEN",
                            "CLOSE is bound to one trade id and must not cancel pending entry orders",
                        )
                    )
                soft_advisory_reasons = _soft_nonblocking_close_advisory_reasons(
                    self.packet,
                    decision.close_trade_ids,
                )
                if (
                    soft_advisory_reasons
                    and _target_requires_entry(self.packet)
                    and not self_improvement_entry_blockers
                    and tradeable_lanes
                ):
                    issues.append(
                        VerificationIssue(
                            "SOFT_CLOSE_ADVISORY_DOES_NOT_PREEMPT_ENTRY",
                            "CLOSE selected a non-blocking soft close advisory while the daily target is open "
                            "and tradeable LIVE_READY lane(s) exist. Treat the advisory as HOLD/reprice/TP "
                            "monitoring unless explicit operator Gate B is present; write an entry-branch "
                            f"TRADE/CANCEL/WAIT receipt instead. Soft advisory: {'; '.join(soft_advisory_reasons[:3])}. "
                            f"Tradeable lane(s): {', '.join(tradeable_lanes[:3])}",
                        )
                    )
                self._verify_close_trade_ids(decision, issues)
                self._verify_close_discipline(decision, issues)
        # A TRADE receipt may not execute close+reentry in one packet,
        # but still validate any supplied close_trade_ids so the report shows
        # every violation instead of hiding bad ids or missing Gate A/B behind
        # the receipt-level reentry blocker.
        if decision.action == "TRADE" and decision.close_trade_ids:
            self._verify_close_trade_ids(decision, issues)
            self._verify_close_discipline(decision, issues)

        return VerificationResult(
            allowed=not any(issue.severity == "BLOCK" for issue in issues),
            issues=tuple(issues),
            close_gate_evidence=tuple(self.close_gate_evidence),
        )

    def _verify_market_read_first(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
    ) -> None:
        market_read = decision.market_read_first if isinstance(decision.market_read_first, dict) else {}
        provenance = (
            decision.decision_provenance
            if isinstance(decision.decision_provenance, dict)
            else None
        )
        execution_sha = (
            _decision_execution_envelope_sha256(decision)
            if provenance and provenance.get("author_kind") == CODEX_MARKET_READ_AUTHOR
            else None
        )
        for code, message in validate_codex_market_read_provenance(
            action=decision.action,
            market_read=market_read,
            provenance=provenance,
            review=decision.market_read_review,
            counterargument=decision.market_read_counterargument,
            change_summary=decision.market_read_change_summary,
            disposition=decision.market_read_disposition,
            veto_reason=decision.market_read_veto_reason,
            vetoed_lane_ids=decision.market_read_vetoed_lane_ids,
            capital_allocation=decision.capital_allocation,
            execution_envelope_sha256=execution_sha,
        ):
            issues.append(VerificationIssue(code, message))
        for code, message in self.market_read_artifact_issues:
            issues.append(VerificationIssue(code, message))
        if provenance and provenance.get("author_kind") == CODEX_MARKET_READ_AUTHOR:
            quote_basis = quote_basis_by_pair_from_broker_payload(
                self.packet.get("broker_snapshot")
                if isinstance(self.packet.get("broker_snapshot"), dict)
                else {}
            )
            for code, message in validate_market_read_numeric_geometry(
                market_read,
                quote_basis_by_pair=quote_basis,
            ):
                issues.append(VerificationIssue(code, message))
        decision_text = _decision_contract_text(decision)
        if not market_read:
            issues.append(
                VerificationIssue(
                    "MARKET_READ_FIRST_MISSING",
                    "decision receipt must begin with market_read_first before blockers, LIVE_READY status, "
                    "negative expectancy, or execution filters are used.",
                )
            )
            if MARKET_READ_LIVE_READY_ZERO_PATTERN.search(decision_text):
                issues.append(
                    VerificationIssue(
                        "LIVE_READY_ZERO_WITHOUT_MARKET_READ",
                        "LIVE_READY=0 / NO_LIVE_READY is an execution-state observation, not a naked market read.",
                    )
                )
            if MARKET_READ_NEGATIVE_EXPECTANCY_PATTERN.search(decision_text):
                issues.append(
                    VerificationIssue(
                        "NEGATIVE_EXPECTANCY_WITHOUT_MARKET_READ",
                        "NEGATIVE_EXPECTANCY cannot replace a current price/path prediction.",
                    )
                )
            return

        missing = _market_read_missing_fields(market_read)
        if missing:
            issues.append(
                VerificationIssue(
                    "MARKET_READ_FIRST_INCOMPLETE",
                    "market_read_first is missing required naked-read / prediction / forced-trade fields: "
                    + ", ".join(missing[:12]),
                )
            )

        blocker_field = _blocker_field_precedes_market_read(decision)
        if blocker_field:
            issues.append(
                VerificationIssue(
                    "BLOCKER_BEFORE_MARKET_READ",
                    f"{blocker_field} explains execution blockers before market_read_first. "
                    "Predict price first, then filter execution.",
                )
            )

        if missing and MARKET_READ_NEGATIVE_EXPECTANCY_PATTERN.search(decision_text):
            issues.append(
                VerificationIssue(
                    "NEGATIVE_EXPECTANCY_WITHOUT_MARKET_READ",
                    "NEGATIVE_EXPECTANCY cannot replace a complete current tape and 30m/2h prediction.",
                )
            )
        if missing and MARKET_READ_LIVE_READY_ZERO_PATTERN.search(decision_text):
            issues.append(
                VerificationIssue(
                    "LIVE_READY_ZERO_WITHOUT_MARKET_READ",
                    "LIVE_READY=0 / NO_LIVE_READY cannot be the conclusion before a complete naked market read.",
                )
            )

        if decision.action in {"TRADE", "WAIT"} and not _decision_references_market_prediction(decision):
            issues.append(
                VerificationIssue(
                    "FINAL_ACTION_MISSING_MARKET_PREDICTION_REF",
                    "final TRADE / WAIT rationale must reference the market_read_first next 30m or next 2h prediction.",
                )
            )
        self._verify_market_read_geometry(decision, market_read, issues)

    def _verify_market_read_geometry(
        self,
        decision: GPTTraderDecision,
        market_read: dict[str, Any],
        issues: list[VerificationIssue],
    ) -> None:
        if decision.action not in ENTRY_DECISION_HORIZON_ACTIONS:
            return
        selected_lane_ids = _selected_trade_lane_ids(decision)
        selected_lanes = [self.lanes.get(lane_id) for lane_id in selected_lane_ids]
        selected_lanes = [lane for lane in selected_lanes if isinstance(lane, dict)]
        naked = market_read.get("naked_read") if isinstance(market_read.get("naked_read"), dict) else {}
        next_30m = market_read.get("next_30m_prediction") if isinstance(market_read.get("next_30m_prediction"), dict) else {}
        next_2h = market_read.get("next_2h_prediction") if isinstance(market_read.get("next_2h_prediction"), dict) else {}
        forced = market_read.get("best_trade_if_forced") if isinstance(market_read.get("best_trade_if_forced"), dict) else {}

        if decision.action == "TRADE" and selected_lanes:
            primary_lane_id = decision.selected_lane_id or selected_lane_ids[0]
            primary_lane = self.lanes.get(primary_lane_id)
            primary_lane = primary_lane if isinstance(primary_lane, dict) else selected_lanes[0]
            primary_pair = str(primary_lane.get("pair") or "").strip()
            selected_pairs = {
                str(lane.get("pair") or "").strip()
                for lane in selected_lanes
                if str(lane.get("pair") or "").strip()
            }
            read_pairs = {
                "naked_read.cleanest_pair_expression": str(
                    naked.get("cleanest_pair_expression") or ""
                ).strip(),
                "next_30m_prediction.pair": str(next_30m.get("pair") or "").strip(),
                "next_2h_prediction.pair": str(next_2h.get("pair") or "").strip(),
                "best_trade_if_forced.pair": str(forced.get("pair") or "").strip(),
            }
            distinct_read_pairs = {pair for pair in read_pairs.values() if pair}
            if (
                len(distinct_read_pairs) != 1
                or not primary_pair
                or distinct_read_pairs != {primary_pair}
                or selected_pairs != {primary_pair}
            ):
                rendered = ", ".join(
                    f"{field}={pair or 'missing'}" for field, pair in read_pairs.items()
                )
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_PAIR_ACTION_CONFLICT",
                        "TRADE market read must describe one primary selected-lane pair across naked/30m/2h/forced sections; "
                        f"primary={primary_pair or 'missing'}, selected={sorted(selected_pairs)}, {rendered}.",
                    )
                )

            primary_side = str(
                primary_lane.get("direction") or primary_lane.get("side") or ""
            ).strip().upper()
            pair_parts = primary_pair.split("_")
            expected_bought = expected_sold = ""
            if len(pair_parts) == 2 and all(pair_parts):
                base, quote = (part.upper() for part in pair_parts)
                if primary_side == "LONG":
                    expected_bought, expected_sold = base, quote
                elif primary_side == "SHORT":
                    expected_bought, expected_sold = quote, base
            actual_bought = str(naked.get("currency_bought") or "").strip().upper()
            actual_sold = str(naked.get("currency_sold") or "").strip().upper()
            if (
                not expected_bought
                or actual_bought != expected_bought
                or actual_sold != expected_sold
            ):
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_CURRENCY_ACTION_CONFLICT",
                        "naked_read currency_bought/currency_sold must match the primary selected pair and side; "
                        f"pair={primary_pair or 'missing'} side={primary_side or 'missing'} "
                        f"expected={expected_bought or 'unknown'}/{expected_sold or 'unknown'} "
                        f"actual={actual_bought or 'missing'}/{actual_sold or 'missing'}.",
                    )
                )
            forced_direction = str(forced.get("direction") or "").strip().upper()
            if (
                primary_side in {"LONG", "SHORT"}
                and forced_direction
                and (
                    _market_read_direction_conflicts_side(forced_direction, primary_side)
                    or forced_direction == "RANGE"
                )
            ):
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_DIRECTION_ACTION_CONFLICT",
                        f"best_trade_if_forced direction {forced_direction} must match primary selected lane "
                        f"{primary_lane_id} side {primary_side}.",
                    )
                )

            horizon_directions = {
                str(next_30m.get("direction") or "").strip().upper(),
                str(next_2h.get("direction") or "").strip().upper(),
            }
            if "RANGE" in horizon_directions:
                primary_method = str(primary_lane.get("method") or decision.method or "").strip().upper()
                selected_methods = {
                    str(lane.get("method") or "").strip().upper()
                    for lane in selected_lanes
                    if str(lane.get("method") or "").strip()
                }
                if (
                    horizon_directions != {"RANGE"}
                    or primary_method != "RANGE_ROTATION"
                    or selected_methods != {"RANGE_ROTATION"}
                ):
                    issues.append(
                        VerificationIssue(
                            "MARKET_READ_RANGE_ACTION_CONFLICT",
                            "RANGE in the 30m/2h market read can authorize TRADE only when both horizons are RANGE "
                            f"and the primary selected lane is RANGE_ROTATION; horizons={sorted(horizon_directions)}, "
                            f"primary_method={primary_method or 'missing'}, selected_methods={sorted(selected_methods)}.",
                        )
                    )

            forced_vehicle = _market_read_vehicle(forced.get("vehicle"))
            primary_vehicle = _market_read_vehicle(primary_lane.get("order_type"))
            if str(primary_lane.get("order_type") or "").strip() and forced_vehicle != primary_vehicle:
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_VEHICLE_ACTION_CONFLICT",
                        f"best_trade_if_forced vehicle {forced_vehicle} must match primary selected lane "
                        f"{primary_lane_id} vehicle {primary_vehicle}.",
                    )
                )
            if decision.market_read_disposition == "ACCEPT_BASELINE":
                geometry_mismatches: list[str] = []
                for field in ("entry", "tp", "sl"):
                    canonical_value = _optional_float(primary_lane.get(field))
                    forced_values = _numbers(str(forced.get(field) or ""))
                    if canonical_value is None:
                        continue
                    if len(forced_values) != 1 or forced_values[0] != canonical_value:
                        geometry_mismatches.append(
                            f"{field}:forced={forced_values or ['missing']} canonical={canonical_value}"
                        )
                if geometry_mismatches:
                    issues.append(
                        VerificationIssue(
                            "MARKET_READ_FORCED_EXECUTION_GEOMETRY_CONFLICT",
                            "ACCEPT_BASELINE requires best_trade_if_forced entry/TP/SL to equal the primary canonical intent: "
                            + "; ".join(geometry_mismatches),
                        )
                    )

            prediction_directions = (
                str(next_30m.get("direction") or "").strip().upper(),
                str(next_2h.get("direction") or "").strip().upper(),
                str(forced.get("direction") or "").strip().upper(),
            )
            direction_conflict_recorded = False
            for lane in selected_lanes:
                lane_side = str(lane.get("direction") or lane.get("side") or "").strip().upper()
                if lane_side not in {"LONG", "SHORT"}:
                    continue
                for direction in prediction_directions:
                    if not direction:
                        continue
                    if _market_read_direction_conflicts_side(direction, lane_side):
                        issues.append(
                            VerificationIssue(
                                "MARKET_READ_DIRECTION_ACTION_CONFLICT",
                                f"market_read_first direction {direction} conflicts with selected {lane_side} lane "
                                f"{lane.get('lane_id') or decision.selected_lane_id}.",
                            )
                        )
                        direction_conflict_recorded = True
                        break
                if direction_conflict_recorded:
                    break

        for horizon_name, prediction in (("next_30m_prediction", next_30m), ("next_2h_prediction", next_2h)):
            pair = str(prediction.get("pair") or "").strip()
            direction = str(prediction.get("direction") or "").strip().upper()
            if not pair or not direction:
                continue
            # Horizon predictions start at broker truth now.  A pending
            # LIMIT/STOP entry is the basis only for the forced trade's own
            # TP/SL geometry, not for a 30m/2h market-direction forecast.
            basis = _current_market_read_price(self.packet, pair)
            if basis is None:
                basis = self._selected_lane_price_basis(pair, selected_lanes)
            if basis is None:
                basis = self._market_read_forced_entry_basis(pair, direction, forced)
            if basis is None:
                continue
            target_numbers = _numbers(str(prediction.get("target_zone") or ""))
            if target_numbers and _market_read_target_on_wrong_side(direction, basis, target_numbers):
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_TARGET_GEOMETRY_CONFLICT",
                        f"{horizon_name} {direction} target_zone is on the wrong side of current {pair} price {basis}.",
                    )
                )
            invalidation_numbers = _numbers(str(prediction.get("invalidation") or ""))
            if invalidation_numbers and _market_read_invalidation_on_wrong_side(direction, basis, invalidation_numbers):
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_INVALIDATION_GEOMETRY_CONFLICT",
                        f"{horizon_name} {direction} invalidation is on the wrong side of current {pair} price {basis}.",
                    )
                )

        forced_direction = str(forced.get("direction") or "").strip().upper()
        forced_entry = _first_number(str(forced.get("entry") or ""))
        if forced_direction and forced_entry is not None:
            forced_tp = _numbers(str(forced.get("tp") or ""))
            forced_sl = _numbers(str(forced.get("sl") or ""))
            if forced_tp and _market_read_target_on_wrong_side(forced_direction, forced_entry, forced_tp):
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_FORCED_TRADE_GEOMETRY_CONFLICT",
                        "best_trade_if_forced TP is on the wrong side for the stated direction.",
                    )
                )
            if forced_sl and _market_read_invalidation_on_wrong_side(forced_direction, forced_entry, forced_sl):
                issues.append(
                    VerificationIssue(
                        "MARKET_READ_FORCED_TRADE_GEOMETRY_CONFLICT",
                        "best_trade_if_forced SL is on the wrong side for the stated direction.",
                    )
                )

    def _selected_lane_price_basis(self, pair: str, selected_lanes: list[dict[str, Any]]) -> float | None:
        for lane in selected_lanes:
            if str(lane.get("pair") or "").strip() != pair:
                continue
            for value in (
                lane.get("entry"),
                (lane.get("risk_metrics") or {}).get("entry_price") if isinstance(lane.get("risk_metrics"), dict) else None,
            ):
                parsed = _optional_float(value)
                if parsed is not None:
                    return parsed
        return None

    def _market_read_forced_entry_basis(
        self,
        pair: str,
        direction: str,
        forced: dict[str, Any],
    ) -> float | None:
        forced_pair = str(forced.get("pair") or "").strip()
        forced_direction = str(forced.get("direction") or "").strip().upper()
        if forced_pair != pair or forced_direction != direction:
            return None
        return _first_number(str(forced.get("entry") or ""))

    def _verify_user_alpha_continuation(
        self,
        decision: GPTTraderDecision,
        selected_lane_ids: tuple[str, ...],
        issues: list[VerificationIssue],
    ) -> None:
        continuation = _active_user_alpha_continuation(self.packet)
        if continuation is None:
            return
        evidence_ref = str(continuation.get("evidence_ref") or "user_alpha:continuation")
        cites_user_alpha = evidence_ref in decision.evidence_refs
        if not cites_user_alpha:
            issues.append(
                VerificationIssue(
                    "USER_ALPHA_CONTINUATION_EVIDENCE_MISSING",
                    "active USER_ALPHA / OPERATOR_ALPHA continuation requires citing user_alpha:continuation "
                    "before choosing TRADE, WAIT, REQUEST_EVIDENCE, or cancel/management actions.",
                )
            )
        if decision.action == "TRADE" and _selected_lanes_include_user_alpha(
            self.packet,
            selected_lane_ids,
            continuation,
        ):
            return
        if not cites_user_alpha or not _decision_cites_user_alpha_exact_blocker(decision, continuation):
            latest = continuation.get("latest_trade") if isinstance(continuation.get("latest_trade"), dict) else {}
            pair = latest.get("pair") or "unknown"
            direction = latest.get("direction") or "unknown"
            issues.append(
                VerificationIssue(
                    "USER_ALPHA_CONTINUATION_UNADDRESSED",
                    f"active USER_ALPHA / OPERATOR_ALPHA {pair} {direction} winner must be answered as "
                    "thesis-alive RELOAD / SECOND_SHOT / 5% path-board continuation, or the receipt must "
                    "cite user_alpha:continuation and name an exact blocker plus next trigger for not continuing.",
                )
            )
        if decision.action in ENTRY_DECISION_HORIZON_ACTIONS:
            plan = decision.twenty_minute_plan if isinstance(decision.twenty_minute_plan, dict) else {}
            plan_refs = tuple(str(ref) for ref in plan.get("evidence_refs", []) or [] if str(ref))
            if evidence_ref not in plan_refs:
                issues.append(
                    VerificationIssue(
                        "USER_ALPHA_TWENTY_MINUTE_PLAN_REF_MISSING",
                        "twenty_minute_plan.evidence_refs must include user_alpha:continuation while an "
                        "active user-led winner is waiting for RELOAD / SECOND_SHOT or an exact blocker.",
                    )
                )

    def _verify_decision_freshness(self, decision: GPTTraderDecision, issues: list[VerificationIssue]) -> None:
        if not decision.generated_at_utc:
            issues.append(
                VerificationIssue(
                    "MISSING_DECISION_TIMESTAMP",
                    "decision receipt must include generated_at_utc so broker-snapshot and market-packet freshness can be verified",
                )
            )
            return
        decision_ts = _parse_utc(decision.generated_at_utc)
        if decision_ts is None:
            issues.append(
                VerificationIssue(
                    "BAD_DECISION_TIMESTAMP",
                    f"generated_at_utc is not parseable: {decision.generated_at_utc}",
                )
            )
            return
        broker = self.packet.get("broker_snapshot", {})
        snapshot_ts = _parse_utc(broker.get("fetched_at_utc") if isinstance(broker, dict) else None)
        freshness_checks = [
            (
                snapshot_ts,
                "broker snapshot",
                "current broker truth, position sidecars, and order intents are rebuilt",
            )
        ]
        artifact_timestamps = self.packet.get("artifact_timestamps")
        if isinstance(artifact_timestamps, dict):
            daily_target_ts = _parse_utc(artifact_timestamps.get("daily_target_generated_at_utc"))
            campaign_plan_ts = _parse_utc(artifact_timestamps.get("campaign_plan_generated_at_utc"))
            order_intents_ts = _parse_utc(artifact_timestamps.get("order_intents_generated_at_utc"))
            attack_advice_ts = _parse_utc(artifact_timestamps.get("ai_attack_advice_generated_at_utc"))
            if (
                decision.action == "TRADE"
                and campaign_plan_ts is not None
                and daily_target_ts is not None
                and campaign_plan_ts < daily_target_ts
            ):
                issues.append(
                    VerificationIssue(
                        "STALE_CAMPAIGN_PLAN_PACKET",
                        "campaign_plan predates the daily_target_state used for verification; rerun "
                        "plan-campaign, then regenerate order_intents before accepting a target-open receipt.",
                    )
                )
            if order_intents_ts is not None:
                if decision.action == "TRADE" and daily_target_ts is not None and order_intents_ts < daily_target_ts:
                    issues.append(
                        VerificationIssue(
                            "STALE_ORDER_INTENTS_PACKET",
                            "order_intents predate the daily_target_state used for verification; rerun "
                            "plan-campaign and generate-intents so lane sizing/firepower reflects the current 5% floor.",
                        )
                    )
                if decision.action == "TRADE" and campaign_plan_ts is not None and order_intents_ts < campaign_plan_ts:
                    issues.append(
                        VerificationIssue(
                            "STALE_ORDER_INTENTS_PACKET",
                            "order_intents predate the campaign_plan used for verification; rerun generate-intents "
                            "from the current campaign plan before accepting a TRADE receipt.",
                        )
                    )
            if order_intents_ts is not None and attack_advice_ts is not None and attack_advice_ts < order_intents_ts:
                issues.append(
                    VerificationIssue(
                        "STALE_ATTACK_ADVICE_PACKET",
                        "ai_attack_advice predates the order_intents used for verification; rerun "
                        "ai-attack-advice after current order intents before accepting TRADE/WAIT.",
                    )
                )
            freshness_checks.extend(
                [
                    (
                        order_intents_ts,
                        "order intents",
                        "current order intents are rebuilt from the latest market packet",
                    ),
                    (
                        attack_advice_ts,
                        "ai_attack_advice",
                        "current attack advice is reflected into the decision receipt",
                    ),
                    (
                        _parse_utc(artifact_timestamps.get("market_context_matrix_generated_at_utc")),
                        "market_context_matrix",
                        "current market_context_matrix is reflected into the order intents and decision receipt",
                    ),
                ]
            )
        for artifact_ts, label, refresh_hint in freshness_checks:
            if artifact_ts is None:
                continue
            if decision_ts < artifact_ts:
                issues.append(
                    VerificationIssue(
                        "STALE_DECISION_RECEIPT",
                        f"decision receipt predates the {label} used for verification; refresh the "
                        f"decision after {refresh_hint}",
                    )
                )

    def _verify_cancel_order_ids(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
        *,
        action: str,
    ) -> None:
        if not decision.cancel_order_ids:
            return
        pending_order_ids = set(_pending_entry_order_ids(self.packet))
        unknown_cancel_ids = sorted(set(decision.cancel_order_ids) - pending_order_ids)
        if unknown_cancel_ids:
            issues.append(
                VerificationIssue(
                    "UNKNOWN_CANCEL_ORDER_ID",
                    f"{action} cancel_order_ids must match current trader-owned pending entry orders: "
                    + ", ".join(unknown_cancel_ids),
                )
            )
        timing_required = _pending_cancel_timing_audit_required_orders(self.packet, decision)
        if timing_required and not _decision_cites_pending_cancel_timing_evidence(decision.evidence_refs):
            issues.append(
                VerificationIssue(
                    "PENDING_CANCEL_TIMING_AUDIT_REQUIRED",
                    f"{action} rejected: execution-timing audit shows same-shape pending cancels "
                    "later touched entry and produced TP or positive MFE without SL. Cite timing:audit, "
                    "timing:canceled_orders, or the relevant timing:canceled_order ref before clearing "
                    "this broker-anchored pending entry: "
                    + ", ".join(timing_required),
                )
            )

    def _verify_twenty_minute_plan(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
    ) -> None:
        if decision.action not in ENTRY_DECISION_HORIZON_ACTIONS:
            return
        plan = decision.twenty_minute_plan if isinstance(decision.twenty_minute_plan, dict) else {}
        if not plan:
            issues.append(
                VerificationIssue(
                    "SHALLOW_DECISION_HORIZON",
                    "TRADE / WAIT / REQUEST_EVIDENCE receipts must include twenty_minute_plan so the "
                    "operator states the next-cycle path before acting.",
                )
            )
            return

        horizon = _optional_float(plan.get("horizon_minutes"))
        if horizon is None or abs(horizon - TRADER_DECISION_HORIZON_MINUTES) > 0.01:
            issues.append(
                VerificationIssue(
                    "BAD_DECISION_HORIZON_MINUTES",
                    f"twenty_minute_plan.horizon_minutes must match the scheduled trader cadence "
                    f"({TRADER_DECISION_HORIZON_MINUTES} minutes), not an invented holding period.",
                )
            )

        missing_fields = [
            field
            for field in TWENTY_MINUTE_PLAN_TEXT_FIELDS
            if not str(plan.get(field) or "").strip()
        ]
        if missing_fields:
            issues.append(
                VerificationIssue(
                    "INCOMPLETE_TWENTY_MINUTE_PLAN",
                    "twenty_minute_plan is missing required reasoning fields: "
                    + ", ".join(missing_fields),
                )
            )

        raw_refs = plan.get("evidence_refs")
        if isinstance(raw_refs, list):
            plan_refs = tuple(str(ref) for ref in raw_refs if str(ref))
        else:
            plan_refs = ()
        if len(plan_refs) < 2:
            issues.append(
                VerificationIssue(
                    "TWENTY_MINUTE_PLAN_REFS_MISSING",
                    "twenty_minute_plan.evidence_refs must cite at least two packet refs used by the next-cycle path.",
                )
            )
        unknown_refs = sorted(set(plan_refs) - self.allowed_refs)
        if unknown_refs:
            issues.append(
                VerificationIssue(
                    "UNKNOWN_TWENTY_MINUTE_PLAN_REF",
                    "twenty_minute_plan uses unknown evidence refs: " + ", ".join(unknown_refs),
                )
            )

        tradeable_lanes = _tradeable_live_ready_lanes(self.packet)
        if (decision.action == "TRADE" or tradeable_lanes) and not any(
            ref.startswith("chart:") for ref in plan_refs
        ):
            issues.append(
                VerificationIssue(
                    "TWENTY_MINUTE_PLAN_CHART_REF_MISSING",
                    "twenty_minute_plan must cite chart evidence when deciding from current tradeable lanes.",
                )
            )

        if decision.action == "TRADE":
            missing_lane_refs = [
                lane_id
                for lane_id in _selected_trade_lane_ids(decision)
                if f"intent:{lane_id}" not in plan_refs
            ]
            if missing_lane_refs:
                issues.append(
                    VerificationIssue(
                        "TWENTY_MINUTE_PLAN_LANE_REF_MISSING",
                        "twenty_minute_plan must cite the selected intent ref(s): "
                        + ", ".join(missing_lane_refs),
                    )
                )

    def _verify_close_trade_ids(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
    ) -> None:
        if not decision.close_trade_ids:
            if decision.action == "CLOSE":
                issues.append(
                    VerificationIssue(
                        "MISSING_CLOSE_TRADE_IDS",
                        "CLOSE must name the trader-owned trade ids to close",
                    )
                )
            return
        if len(decision.close_trade_ids) != 1:
            issues.append(
                VerificationIssue(
                    "CLOSE_SINGLE_TRADE_REQUIRED",
                    "CLOSE must bind exactly one current trader-owned trade id; "
                    "multiple or duplicate targets require separately refreshed receipts",
                )
            )
        snapshot = self.packet.get("broker_snapshot", {}) or {}
        trader_trade_ids: set[str] = set()
        for position in snapshot.get("position_summaries", []) or []:
            if str(position.get("owner") or "") == "trader":
                tid = position.get("trade_id")
                if tid is not None:
                    trader_trade_ids.add(str(tid))
        unknown = sorted(set(decision.close_trade_ids) - trader_trade_ids)
        if unknown:
            issues.append(
                VerificationIssue(
                    "UNKNOWN_CLOSE_TRADE_ID",
                    "close_trade_ids must match current trader-owned open positions: "
                    + ", ".join(unknown),
                )
            )

    def _verify_close_discipline(
        self,
        decision: GPTTraderDecision,
        issues: list[VerificationIssue],
    ) -> None:
        """Two-gate CLOSE discipline (see feedback_no_unilateral_close.md and
        AGENT_CONTRACT §10):

        Gate A — market evidence: every named trade must have its thesis
        invalidated by either a structural BOS/CHOCH on M15 or H4 against
        the position side, OR by an explicit `invalidation_price` +
        `invalidation_tf` that broker truth confirms has been hit. Prose
        `invalidation` text alone does not count.

        Gate B — standing hard loss-cut authorization or explicit operator
        authorization. Hard Gate A is enough for justified loss-cuts; softer
        Gate A still needs one of:

          1. `QR_OPERATOR_CLOSE_OVERRIDE=1` in the operator shell, OR
          2. A fresh `data/.operator_close_token` file (mtime within
             OPERATOR_CLOSE_TOKEN_FRESH_SECONDS = 5 minutes), created
             by `touch data/.operator_close_token` before each softer CLOSE
             batch.

        The `operator_close_authorized` JSON field remains in the
        schema for backward compatibility and audit trail (the verifier
        logs whether the trader claimed authorization) but is treated
        as advisory only.

        Both gates must pass; failing either rejects the receipt.
        """
        if not decision.close_trade_ids:
            return

        position_by_tid = _trader_position_lookup(self.packet)
        timing_required = _loss_close_timing_audit_required_trades(
            self.packet,
            decision,
            position_by_tid,
        )

        # Gate A: thesis-still-valid check, applied to every named trade.
        # Hard Gate A also carries the operator's standing authorization for a
        # justified loss-cut; softer Gate A still needs env/token Gate B.
        still_valid: list[str] = []
        non_invalidation_loss_exit_reasons: list[str] = []
        same_direction_supported: list[str] = []
        needs_explicit_gate_b: list[str] = []
        needs_profitability_p0_context: list[str] = []
        needs_profitability_acceptance_context: list[str] = []
        needs_profitability_acceptance_hard_gate: list[str] = []
        needs_hard_timing_gate: list[str] = []
        profitability_p0_blocker = _profitability_p0_soft_close_blocker(self.packet)
        acceptance_loss_close_blockers = _profitability_acceptance_loss_close_blockers(self.packet)
        premature_timing_guard = _premature_loss_close_timing_guard(self.packet)
        cites_profitability_p0 = _decision_cites_profitability_p0(decision.evidence_refs)
        cites_profitability_acceptance = _decision_cites_profitability_acceptance(decision.evidence_refs)
        cites_timing_evidence = _decision_cites_close_timing_evidence(decision.evidence_refs)
        timing_required_ids = {str(item).split(" ", 1)[0] for item in timing_required}
        explicit_operator_gate_b = _operator_close_gate_authorized()
        for tid in decision.close_trade_ids:
            pos = position_by_tid.get(str(tid))
            if pos is None:
                # `_verify_close_trade_ids` already flagged the unknown id;
                # do not double-report.
                continue
            pair = str(pos.get("pair") or "")
            side = str(pos.get("side") or "")
            unrealized_pl_jpy = _optional_float(pos.get("unrealized_pl_jpy"))
            loss_side_close = unrealized_pl_jpy is None or unrealized_pl_jpy <= 0
            issues.extend(_close_spread_issues(self.packet, pair, trade_id=str(tid)))
            invalidated, _reason, standing_authorized = _close_thesis_invalidation(
                self.packet,
                pair,
                side,
                trade_id=str(tid),
                decision=decision,
            )
            evidence = {
                "trade_id": str(tid),
                "pair": pair,
                "side": side,
                "unrealized_pl_jpy": unrealized_pl_jpy,
                "loss_side_close": loss_side_close,
                "gate_a_invalidated": invalidated,
                "gate_a_reason": _reason or "no reproducible Gate A invalidation evidence",
                "gate_b_standing_authorized": standing_authorized,
                "gate_b_explicit_operator_authorized": explicit_operator_gate_b,
                "explicit_gate_b_required": False,
                "profitability_p0_context_required": False,
                "profitability_p0_context_cited": cites_profitability_p0,
                "timing_audit_required": str(tid) in timing_required_ids,
                "timing_evidence_cited": cites_timing_evidence,
                "hard_timing_gate_required": False,
                "same_direction_support_conflict": None,
            }
            if not invalidated:
                still_valid.append(
                    f"{tid} ({pair} {side})"
                )
                if loss_side_close:
                    reason_hits = _non_invalidation_loss_exit_reason_hits(decision)
                    if reason_hits:
                        non_invalidation_loss_exit_reasons.append(
                            f"{tid} ({pair} {side}: {', '.join(reason_hits)})"
                        )
                self.close_gate_evidence.append(CloseGateEvidence(**evidence))
                continue

            if (
                profitability_p0_blocker is not None
                and loss_side_close
                and not cites_profitability_p0
            ):
                needs_profitability_p0_context.append(f"{tid} ({pair} {side})")
                evidence["profitability_p0_context_required"] = True
            if (
                acceptance_loss_close_blockers
                and loss_side_close
                and not cites_profitability_acceptance
            ):
                needs_profitability_acceptance_context.append(f"{tid} ({pair} {side})")
            if (
                acceptance_loss_close_blockers
                and loss_side_close
                and not standing_authorized
            ):
                needs_profitability_acceptance_hard_gate.append(f"{tid} ({pair} {side})")

            if not standing_authorized:
                sidecar_conflict = _same_direction_hold_support_conflict(
                    self.packet,
                    trade_id=str(tid),
                    pair=pair,
                    side=side,
                    evidence_label="soft close evidence",
                    allow_single_strong_sidecar=True,
                )
                if sidecar_conflict and loss_side_close:
                    same_direction_supported.append(
                        f"{tid} ({sidecar_conflict})"
                    )
                    evidence["same_direction_support_conflict"] = sidecar_conflict
                    self.close_gate_evidence.append(CloseGateEvidence(**evidence))
                    continue
                supported, support_reason = _close_same_direction_matrix_support(
                    self.packet,
                    pair,
                    side,
                )
                if supported and loss_side_close:
                    same_direction_supported.append(
                        f"{tid} ({support_reason})"
                    )
                    evidence["same_direction_support_conflict"] = support_reason
                    self.close_gate_evidence.append(CloseGateEvidence(**evidence))
                    continue
                if premature_timing_guard is not None and (
                    loss_side_close
                ):
                    needs_hard_timing_gate.append(f"{tid} ({pair} {side})")
                    evidence["hard_timing_gate_required"] = True
                needs_explicit_gate_b.append(f"{tid} ({pair} {side})")
                evidence["explicit_gate_b_required"] = True

            self.close_gate_evidence.append(CloseGateEvidence(**evidence))

        if still_valid:
            issues.append(
                VerificationIssue(
                    "CLOSE_THESIS_STILL_VALID",
                    "CLOSE rejected: thesis still valid (no BOS/CHOCH against "
                    "side on M15/H4, no buffered invalidation_price hit with chart/technical confirmation, and no fresh "
                    "position sidecar REVIEW_CLOSE/RECOMMEND_CLOSE) for: "
                    + ", ".join(still_valid),
                )
            )

        if non_invalidation_loss_exit_reasons:
            issues.append(
                VerificationIssue(
                    "THESIS_INVALIDATION_EXIT_REQUIRED",
                    "CLOSE rejected: a loss-side exit must be justified by price/timeframe/structure/"
                    "currency-theme invalidation, margin/emergency invalidation, or explicit operator override "
                    "with matching market evidence. Negative P/L, NEGATIVE_EXPECTANCY, duplicate blockers, "
                    "low LIVE_READY, or old SL templates are not standalone exit triggers while the thesis "
                    "is still valid. Forbidden non-invalidation reason(s) cited for: "
                    + ", ".join(non_invalidation_loss_exit_reasons),
                )
            )

        if same_direction_supported:
            issues.append(
                VerificationIssue(
                    "CLOSE_SAME_DIRECTION_MARKET_SUPPORT",
                    "CLOSE rejected: softer close evidence conflicts with "
                    "same-direction sidecar or directional market_context_matrix support. "
                    "Contract §10 requires HOLD/reprice/TP rebalance while the "
                    "market stack still supports a loss-side open position, "
                    "unless hard invalidation evidence is present. Blocked for: "
                    + ", ".join(same_direction_supported),
                )
            )

        if needs_profitability_p0_context:
            streak = profitability_p0_blocker.get("current_streak") if profitability_p0_blocker else None
            pf = profitability_p0_blocker.get("profit_factor") if profitability_p0_blocker else None
            expectancy = profitability_p0_blocker.get("expectancy_jpy") if profitability_p0_blocker else None
            details = []
            if streak is not None:
                details.append(f"streak={streak}")
            if pf is not None:
                details.append(f"PF={pf}")
            if expectancy is not None:
                details.append(f"expectancy={expectancy}")
            suffix = f" ({', '.join(details)})" if details else ""
            issues.append(
                VerificationIssue(
                    "CLOSE_PROFITABILITY_P0_CONTEXT_REQUIRED",
                    "CLOSE rejected: profitability discipline P0 is active"
                    f"{suffix}, and this underwater CLOSE would realize another loss-side market close. "
                    "The receipt must cite self_improvement:audit, self_improvement:profitability, "
                    "or self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED and explain why "
                    "this loss-side market close repairs rather than repeats the MARKET_ORDER_TRADE_CLOSE leak. "
                    "Missing self-improvement P0 context for: "
                    + ", ".join(needs_profitability_p0_context),
                )
            )

        if needs_profitability_acceptance_context:
            codes = sorted(
                {
                    str(item.get("code") or "")
                    for item in acceptance_loss_close_blockers
                    if str(item.get("code") or "")
                }
            )
            issues.append(
                VerificationIssue(
                    "CLOSE_PROFITABILITY_ACCEPTANCE_P0_REQUIRED",
                    "CLOSE rejected: profitability_acceptance has active P0 loss-side "
                    "market-close leakage"
                    + (f" ({', '.join(codes)})" if codes else "")
                    + ". An underwater CLOSE must cite profitability:acceptance and explain why "
                    "the close repairs the recent MARKET_ORDER_TRADE_CLOSE leak instead of adding "
                    "another red gateway close. Missing profitability acceptance context for: "
                    + ", ".join(needs_profitability_acceptance_context),
                )
            )

        if needs_profitability_acceptance_hard_gate:
            codes = sorted(
                {
                    str(item.get("code") or "")
                    for item in acceptance_loss_close_blockers
                    if str(item.get("code") or "")
                }
            )
            issues.append(
                VerificationIssue(
                    "CLOSE_PROFITABILITY_ACCEPTANCE_HARD_GATE_REQUIRED",
                    "CLOSE rejected: profitability_acceptance has active P0 loss-side "
                    "market-close leakage"
                    + (f" ({', '.join(codes)})" if codes else "")
                    + ". While this red acceptance gate is active, an underwater market CLOSE "
                    "requires standing hard Gate A evidence; soft sidecar evidence plus an operator "
                    "token/citation is not enough because it can repeat the recent gateway "
                    "MARKET_ORDER_TRADE_CLOSE leak. Use HOLD/reprice/TP rebalance, or provide "
                    "post-entry H4 structure, recorded invalidation, thesis_evolution with "
                    "explicit invalidation/technical proof, structural "
                    "position-management REVIEW_EXIT, or equivalent hard sidecar evidence for: "
                    + ", ".join(needs_profitability_acceptance_hard_gate),
                )
            )

        if timing_required:
            summary = self.packet.get("execution_timing_audit", {}).get("summary", {})
            premature = _optional_int(
                summary.get("loss_market_closes_may_have_been_premature")
            ) or 0
            capture_missed = _timing_profit_capture_missed_count(summary)
            labels: list[str] = []
            if premature > 0:
                labels.append(f"{premature} loss-side market close(s) as potentially premature")
            if capture_missed > 0:
                evidence_label = (
                    "post-repair loss close(s)"
                    if _timing_profit_capture_split_present(summary)
                    else "loss close(s)"
                )
                labels.append(
                    f"{capture_missed} {evidence_label} with missed TP-progress profit capture"
                )
            if not labels:
                labels.append(f"{len(timing_required)} loss-side close timing issue(s)")
            followthrough = _optional_float(
                summary.get("market_close_estimated_followthrough_jpy")
            )
            counterfactual_delta = _timing_profit_capture_counterfactual_delta(summary)
            tail_parts: list[str] = []
            if followthrough is not None:
                tail_parts.append(f"estimated follow-through left behind {followthrough:.2f} JPY")
            if counterfactual_delta is not None:
                tail_parts.append(
                    f"profit-capture counterfactual delta {counterfactual_delta:.2f} JPY"
                )
            tail = f"; {'; '.join(tail_parts)}" if tail_parts else ""
            issues.append(
                VerificationIssue(
                    "CLOSE_TIMING_AUDIT_REQUIRED",
                    "CLOSE rejected: recent execution-timing audit marks "
                    f"{' and '.join(labels)}{tail}. "
                    "An underwater CLOSE must cite timing:audit, timing:loss_closes, "
                    "timing:market_closes, or the relevant timing:loss_close/market_close ref "
                    "and explain why current invalidation beats HOLD/reprice/TP evidence. "
                    "Missing timing evidence for: "
                    + ", ".join(timing_required),
                )
            )

        if needs_hard_timing_gate:
            details = []
            if premature_timing_guard is not None:
                premature = premature_timing_guard.get("premature")
                contained = premature_timing_guard.get("contained")
                audited = premature_timing_guard.get("audited")
                followthrough = premature_timing_guard.get("followthrough_jpy")
                if audited is not None:
                    details.append(f"audited={audited}")
                if premature is not None:
                    details.append(f"premature={premature}")
                if contained is not None:
                    details.append(f"contained={contained}")
                if followthrough is not None:
                    details.append(f"followthrough_left={followthrough:.2f} JPY")
                capture_missed = premature_timing_guard.get("profit_capture_missed")
                capture_delta = premature_timing_guard.get(
                    "profit_capture_counterfactual_delta_jpy"
                )
                capture_jpy = premature_timing_guard.get(
                    "profit_capture_counterfactual_jpy"
                )
                actual_pl = premature_timing_guard.get("profit_capture_actual_pl_jpy")
                counterfactual_pl = premature_timing_guard.get(
                    "profit_capture_counterfactual_pl_jpy"
                )
                if capture_missed is not None:
                    details.append(f"profit_capture_missed={capture_missed}")
                if capture_jpy is not None:
                    details.append(f"profit_capture={capture_jpy:.2f} JPY")
                if capture_delta is not None:
                    details.append(f"counterfactual_delta={capture_delta:.2f} JPY")
                if actual_pl is not None:
                    details.append(f"actual_loss_close_pl={actual_pl:.2f} JPY")
                if counterfactual_pl is not None:
                    details.append(f"counterfactual_pl={counterfactual_pl:.2f} JPY")
            suffix = f" ({', '.join(details)})" if details else ""
            issues.append(
                VerificationIssue(
                    "CLOSE_PREMATURE_TIMING_HARD_GATE_REQUIRED",
                    "CLOSE rejected: execution-timing audit still shows premature/profit-capture "
                    f"loss-side close timing leakage{suffix}. A timing reference acknowledges "
                    "that evidence, but softer Gate A plus operator token is not enough for another "
                    "underwater market close while this guard is active. Use HOLD/reprice/TP rebalance "
                    "or provide hard Gate A standing authorization (H4 structure, recorded invalidation, "
                    "or a hard structural sidecar). Blocked for: "
                    + ", ".join(needs_hard_timing_gate),
                )
            )

        # Gate B (repaired 2026-06-04): hard machine-confirmed loss cuts
        # satisfy the operator's standing "妥当な損切りならやっていい" directive.
        # Softer sidecar-only reviews still require explicit env/token
        # authorization. The JSON receipt field remains advisory-only.
        if needs_explicit_gate_b and not explicit_operator_gate_b:
            issues.append(
                VerificationIssue(
                    "CLOSE_OPERATOR_AUTH_REQUIRED",
                    "CLOSE rejected for softer close evidence: requires "
                    "QR_OPERATOR_CLOSE_OVERRIDE=1 in the operator shell OR a fresh data/.operator_close_token "
                    f"(mtime within {OPERATOR_CLOSE_TOKEN_FRESH_SECONDS}s). The "
                    "receipt's `operator_close_authorized` field is advisory "
                    "only and is no longer accepted as authorization "
                    "(2026-05-12T15:33 UTC mass-close incident, "
                    "feedback_no_unilateral_close.md). Standing structural close authorization covered no-token "
                    "hard loss-cuts only; explicit Gate B is still missing for: "
                    + ", ".join(needs_explicit_gate_b),
                )
            )

    def _verify_strategy_reviews(self, decision: GPTTraderDecision, issues: list[VerificationIssue]) -> None:
        for review in decision.strategy_reviews:
            lane_id = str(review.get("lane_id") or "")
            method = str(review.get("method") or "")
            verdict = str(review.get("verdict") or "")
            if not lane_id:
                issues.append(VerificationIssue("STRATEGY_REVIEW_LANE_REQUIRED", "strategy review requires lane_id"))
                continue
            lane = self.lanes.get(lane_id)
            if lane is None:
                issues.append(VerificationIssue("UNKNOWN_STRATEGY_REVIEW_LANE", f"review lane is not in packet: {lane_id}"))
                continue
            if method not in ALLOWED_METHODS:
                issues.append(VerificationIssue("BAD_STRATEGY_REVIEW_METHOD", f"unsupported strategy review method {method!r}"))
            elif lane.get("method") != method:
                issues.append(
                    VerificationIssue(
                        "STRATEGY_REVIEW_METHOD_MISMATCH",
                        f"review method {method} does not match lane {lane_id} method {lane.get('method')}",
                    )
                )
            if verdict and verdict not in {"SUPPORTS", "REJECTS", "BLOCKED", "WATCH"}:
                issues.append(VerificationIssue("BAD_STRATEGY_REVIEW_VERDICT", f"unsupported strategy review verdict {verdict!r}"))

    def _verify_specialist_reviews(self, decision: GPTTraderDecision, issues: list[VerificationIssue]) -> None:
        for review in decision.specialist_reviews:
            role = str(review.get("role") or "")
            verdict = str(review.get("verdict") or "")
            lane_id = str(review.get("lane_id") or "")
            method = str(review.get("method") or "")
            cited_refs = tuple(str(ref) for ref in review.get("cited_evidence_refs", []) or [])
            if role not in ALLOWED_SPECIALIST_ROLES:
                issues.append(VerificationIssue("BAD_SPECIALIST_REVIEW_ROLE", f"unsupported specialist review role {role!r}"))
            if verdict not in {"SUPPORTS", "REJECTS", "BLOCKED", "WATCH"}:
                issues.append(VerificationIssue("BAD_SPECIALIST_REVIEW_VERDICT", f"unsupported specialist review verdict {verdict!r}"))
            if review.get("read_only") is not True:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_NOT_READ_ONLY",
                        "specialist reviews are processed observation only and must declare read_only=true",
                    )
                )
            if review.get("live_permission") is not False:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_LIVE_PERMISSION",
                        "specialist reviews must declare live_permission=false; only the final verified trader receipt can authorize execution",
                    )
                )
            forbidden = sorted(field for field in FORBIDDEN_SPECIALIST_AUTHORITY_FIELDS if field in review)
            if forbidden:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_AUTHORITY_FIELD",
                        "specialist reviews must not carry execution authority fields: " + ", ".join(forbidden),
                    )
                )
            if not cited_refs:
                issues.append(
                    VerificationIssue(
                        "SPECIALIST_REVIEW_REFS_REQUIRED",
                        "specialist reviews must cite packet evidence refs",
                    )
                )
            unknown_refs = sorted(set(cited_refs) - self.allowed_refs)
            if unknown_refs:
                issues.append(
                    VerificationIssue(
                        "UNKNOWN_SPECIALIST_REVIEW_REF",
                        f"specialist review uses unknown evidence refs: {', '.join(unknown_refs)}",
                    )
                )
            if lane_id:
                lane = self.lanes.get(lane_id)
                if lane is None:
                    issues.append(VerificationIssue("UNKNOWN_SPECIALIST_REVIEW_LANE", f"specialist review lane is not in packet: {lane_id}"))
                elif method and method != str(lane.get("method") or ""):
                    issues.append(
                        VerificationIssue(
                            "SPECIALIST_REVIEW_METHOD_MISMATCH",
                            f"specialist review method {method} does not match lane {lane_id} method {lane.get('method')}",
                        )
                    )


GPT_TRADER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "generated_at_utc",
        "market_read_first",
        "action",
        "selected_lane_id",
        "confidence",
        "thesis",
        "method",
        "narrative",
        "chart_story",
        "invalidation",
        "rejected_alternatives",
        "risk_notes",
        "evidence_refs",
        "operator_summary",
        "twenty_minute_plan",
    ],
    "properties": {
        "generated_at_utc": {"type": "string"},
        "market_read_first": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "naked_read",
                "next_30m_prediction",
                "next_2h_prediction",
                "best_trade_if_forced",
            ],
            "properties": {
                "naked_read": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "currency_bought",
                        "currency_sold",
                        "cleanest_pair_expression",
                        "is_cleanest_currency_theme",
                        "location_24h",
                        "h1_h4_alignment",
                        "tape_state",
                        "known_winning_trade_shape_match",
                        "proposed_building_style_allowed",
                        "thesis_state",
                        "what_price_is_trying_to_do_now",
                    ],
                    "properties": {
                        "currency_bought": {"type": "string"},
                        "currency_sold": {"type": "string"},
                        "cleanest_pair_expression": {"type": "string"},
                        "is_cleanest_currency_theme": {"type": "string"},
                        "location_24h": {
                            "type": "string",
                            "enum": ["LOWER", "MIDDLE", "UPPER", "UNKNOWN"],
                        },
                        "h1_h4_alignment": {"type": "string"},
                        "tape_state": {
                            "type": "string",
                            "enum": ["TREND", "RANGE", "SQUEEZE", "FADE", "ROTATION"],
                        },
                        "known_winning_trade_shape_match": {"type": "string"},
                        "proposed_building_style_allowed": {"type": "string"},
                        "thesis_state": {
                            "type": "string",
                            "enum": ["ALIVE", "WOUNDED", "INVALIDATED", "EMERGENCY", "UNKNOWN"],
                        },
                        "what_price_is_trying_to_do_now": {"type": "string"},
                    },
                },
                "next_30m_prediction": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pair", "direction", "expected_path", "target_zone", "invalidation"],
                    "properties": {
                        "pair": {"type": "string"},
                        "direction": {"type": "string"},
                        "expected_path": {"type": "string"},
                        "target_zone": {"type": "string"},
                        "invalidation": {"type": "string"},
                    },
                },
                "next_2h_prediction": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pair", "direction", "expected_path", "target_zone", "invalidation"],
                    "properties": {
                        "pair": {"type": "string"},
                        "direction": {"type": "string"},
                        "expected_path": {"type": "string"},
                        "target_zone": {"type": "string"},
                        "invalidation": {"type": "string"},
                    },
                },
                "best_trade_if_forced": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pair", "direction", "vehicle", "entry", "tp", "sl", "why_this_pays"],
                    "properties": {
                        "pair": {"type": "string"},
                        "direction": {"type": "string"},
                        "vehicle": {"type": "string", "enum": ["MARKET", "LIMIT", "STOP"]},
                        "entry": {"type": "string"},
                        "tp": {"type": "string"},
                        "sl": {"type": "string"},
                        "why_this_pays": {"type": "string"},
                    },
                },
            },
        },
        "decision_provenance": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "integer", "const": 2},
                "author_kind": {
                    "type": "string",
                    "enum": ["DETERMINISTIC_BASELINE", CODEX_MARKET_READ_AUTHOR],
                },
                "generated_by": {"type": "string"},
                "prepared_at_utc": {"type": "string"},
                "model": {"type": "string"},
                "reasoning_effort": {"type": "string"},
                "authored_at_utc": {"type": "string"},
                "applied_at_utc": {"type": "string"},
                "baseline_sha256": {"type": "string"},
                "evidence_packet_sha256": {"type": "string"},
                "overlay_sha256": {"type": "string"},
                "market_read_sha256": {"type": "string"},
                "execution_envelope_sha256": {"type": "string"},
                "baseline_execution_envelope_sha256": {"type": "string"},
                "final_execution_envelope_sha256": {"type": "string"},
                "baseline_action": {"type": "string"},
                "final_action": {"type": "string"},
                "baseline_selected_lane_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "baseline_disposition": {"type": "string"},
                "action_downgrade_only": {"type": "boolean"},
                "capital_allocation_sha256": {"type": "string"},
                "capital_allocation_board_sha256": {"type": "string"},
                "execution_cost_floor_sha256": {
                    "type": ["string", "null"]
                },
                "authorized_size_multiple": {"type": "number"},
                "authorized_units": {"type": "integer"},
                "execution_fields_preserved": {"type": "boolean"},
                "risk_envelope_not_expanded": {"type": "boolean"},
                "live_permission_granted": {"type": "boolean"},
            },
        },
        "capital_allocation": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "decision",
                "lane_id",
                "size_multiple",
                "selected_units",
                "allocation_board_sha256",
                "rationale",
            ],
            "properties": {
                "decision": {"type": "string", "enum": ["ALLOCATE", "NO_TRADE"]},
                "lane_id": {"type": ["string", "null"]},
                "size_multiple": {"type": "number", "enum": [0.0, 0.5, 0.75, 1.0]},
                "selected_units": {"type": "integer"},
                "allocation_board_sha256": {"type": "string"},
                "rationale": {"type": "string"},
            },
        },
        "market_read_review": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "prior_prediction_ids": {"type": "array", "items": {"type": "string"}},
                "what_failed": {"type": "string"},
                "adjustment": {"type": "string"},
                "no_change_reason": {"type": "string"},
            },
        },
        "market_read_counterargument": {"type": "string"},
        "market_read_change_summary": {"type": "string"},
        "market_read_disposition": {
            "type": "string",
            "enum": ["", "ACCEPT_BASELINE", "VETO_WAIT", "VETO_REQUEST_EVIDENCE"],
        },
        "market_read_veto_reason": {"type": "string"},
        "market_read_vetoed_lane_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "action": {"type": "string", "enum": list(ALLOWED_ACTIONS)},
        "selected_lane_id": {"type": ["string", "null"]},
        "selected_lane_ids": {"type": "array", "items": {"type": "string"}},
        "cancel_order_ids": {"type": "array", "items": {"type": "string"}},
        "close_trade_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": list(ALLOWED_CONFIDENCE)},
        "thesis": {"type": "string"},
        "method": {"type": "string", "enum": list(ALLOWED_METHODS)},
        "narrative": {"type": "string"},
        "chart_story": {"type": "string"},
        "invalidation": {"type": "string"},
        "invalidation_price": {"type": ["number", "null"]},
        "invalidation_tf": {"type": ["string", "null"]},
        "operator_close_authorized": {"type": "boolean"},
        "rejected_alternatives": {"type": "array", "items": {"type": "string"}},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "operator_summary": {"type": "string"},
        "twenty_minute_plan": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": [
                "horizon_minutes",
                "primary_path",
                "failure_path",
                "entry_or_hold_trigger",
                "invalidation_or_cancel_trigger",
                "counterargument",
                "next_cycle_check",
                "evidence_refs",
            ],
            "properties": {
                "horizon_minutes": {"type": "number"},
                "primary_path": {"type": "string"},
                "failure_path": {"type": "string"},
                "entry_or_hold_trigger": {"type": "string"},
                "invalidation_or_cancel_trigger": {"type": "string"},
                "counterargument": {"type": "string"},
                "next_cycle_check": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
            },
        },
        "strategy_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["lane_id", "method", "verdict", "summary"],
                "properties": {
                    "lane_id": {"type": "string"},
                    "method": {"type": "string", "enum": list(ALLOWED_METHODS)},
                    "verdict": {"type": "string", "enum": ["SUPPORTS", "REJECTS", "BLOCKED", "WATCH"]},
                    "summary": {"type": "string"},
                },
            },
        },
        "specialist_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "role",
                    "verdict",
                    "summary",
                    "cited_evidence_refs",
                    "read_only",
                    "live_permission",
                ],
                "properties": {
                    "role": {"type": "string", "enum": list(ALLOWED_SPECIALIST_ROLES)},
                    "lane_id": {"type": ["string", "null"]},
                    "method": {"type": ["string", "null"], "enum": [*ALLOWED_METHODS, None]},
                    "verdict": {"type": "string", "enum": ["SUPPORTS", "REJECTS", "BLOCKED", "WATCH"]},
                    "summary": {"type": "string"},
                    "cited_evidence_refs": {"type": "array", "items": {"type": "string"}},
                    "hard_gate_codes": {"type": "array", "items": {"type": "string"}},
                    "read_only": {"type": "boolean"},
                    "live_permission": {"type": "boolean"},
                },
            },
        },
    },
}


def _decision_from_payload(payload: dict[str, Any]) -> GPTTraderDecision:
    selected_lane_id = payload.get("selected_lane_id")
    selected_lane_ids = tuple(
        str(item) for item in payload.get("selected_lane_ids", []) or [] if str(item)
    )
    if not selected_lane_ids and selected_lane_id is not None:
        selected_lane_ids = (str(selected_lane_id),)
    return GPTTraderDecision(
        generated_at_utc=(
            str(payload.get("generated_at_utc")) if payload.get("generated_at_utc") else None
        ),
        market_read_first=(
            dict(payload.get("market_read_first"))
            if isinstance(payload.get("market_read_first"), dict)
            else {}
        ),
        action=str(payload.get("action") or ""),
        selected_lane_id=str(selected_lane_id) if selected_lane_id is not None else None,
        selected_lane_ids=selected_lane_ids,
        cancel_order_ids=tuple(str(item) for item in payload.get("cancel_order_ids", []) or []),
        confidence=str(payload.get("confidence") or ""),
        thesis=str(payload.get("thesis") or ""),
        method=str(payload.get("method") or ""),
        narrative=str(payload.get("narrative") or ""),
        chart_story=str(payload.get("chart_story") or ""),
        invalidation=str(payload.get("invalidation") or ""),
        rejected_alternatives=tuple(str(item) for item in payload.get("rejected_alternatives", []) or []),
        risk_notes=tuple(str(item) for item in payload.get("risk_notes", []) or []),
        evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", []) or []),
        operator_summary=str(payload.get("operator_summary") or ""),
        twenty_minute_plan=(
            dict(payload.get("twenty_minute_plan"))
            if isinstance(payload.get("twenty_minute_plan"), dict)
            else None
        ),
        decision_provenance=(
            dict(payload.get("decision_provenance"))
            if isinstance(payload.get("decision_provenance"), dict)
            else None
        ),
        capital_allocation=(
            dict(payload.get("capital_allocation"))
            if isinstance(payload.get("capital_allocation"), dict)
            else None
        ),
        market_read_review=(
            dict(payload.get("market_read_review"))
            if isinstance(payload.get("market_read_review"), dict)
            else None
        ),
        market_read_counterargument=str(payload.get("market_read_counterargument") or ""),
        market_read_change_summary=str(payload.get("market_read_change_summary") or ""),
        market_read_disposition=str(payload.get("market_read_disposition") or ""),
        market_read_veto_reason=str(payload.get("market_read_veto_reason") or ""),
        market_read_vetoed_lane_ids=tuple(
            str(item) for item in payload.get("market_read_vetoed_lane_ids", []) or []
        ),
        close_trade_ids=tuple(str(item) for item in payload.get("close_trade_ids", []) or []),
        invalidation_price=_optional_float(payload.get("invalidation_price")),
        invalidation_tf=(
            str(payload.get("invalidation_tf")) if payload.get("invalidation_tf") else None
        ),
        operator_close_authorized=bool(payload.get("operator_close_authorized", False)),
        payload_field_order=tuple(str(key) for key in payload.keys()),
        strategy_reviews=tuple(
            dict(item)
            for item in payload.get("strategy_reviews", []) or []
            if isinstance(item, dict)
        ),
        specialist_reviews=tuple(
            dict(item)
            for item in payload.get("specialist_reviews", []) or []
            if isinstance(item, dict)
        ),
    )


def _capital_allocation_size_multiple(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    if str(value.get("decision") or "").strip().upper() != "ALLOCATE":
        return None
    raw_multiple = value.get("size_multiple")
    if isinstance(raw_multiple, bool) or not isinstance(raw_multiple, (int, float)):
        return None
    multiple = _optional_float(raw_multiple)
    if multiple not in {0.5, 0.75, 1.0}:
        return None
    return multiple


def _market_read_longish(direction: str) -> bool:
    return direction.upper() in {"LONG", "BUY", "UP", "BULL", "BULLISH"}


def _market_read_shortish(direction: str) -> bool:
    return direction.upper() in {"SHORT", "SELL", "DOWN", "BEAR", "BEARISH"}


def _market_read_direction_conflicts_side(direction: str, side: str) -> bool:
    if _market_read_longish(direction):
        return side.upper() == "SHORT"
    if _market_read_shortish(direction):
        return side.upper() == "LONG"
    return False


def _market_read_target_on_wrong_side(direction: str, basis: float, prices: list[float]) -> bool:
    if _market_read_longish(direction):
        return min(prices) <= basis
    if _market_read_shortish(direction):
        return max(prices) >= basis
    return False


def _market_read_invalidation_on_wrong_side(direction: str, basis: float, prices: list[float]) -> bool:
    if _market_read_longish(direction):
        return max(prices) >= basis
    if _market_read_shortish(direction):
        return min(prices) <= basis
    return False


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(?<![\d.])[-+]?\d+(?:\.\d+)?", text):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return values


def _first_number(text: str) -> float | None:
    values = _numbers(text)
    return values[0] if values else None


def _current_market_read_price(packet: dict[str, Any], pair: str) -> float | None:
    if not pair:
        return None
    quotes = packet.get("broker_snapshot", {}).get("quotes") if isinstance(packet.get("broker_snapshot"), dict) else {}
    if isinstance(quotes, dict):
        quote = quotes.get(pair)
        if isinstance(quote, dict):
            bid = _optional_float(quote.get("bid"))
            ask = _optional_float(quote.get("ask"))
            if bid is not None and ask is not None:
                return round((bid + ask) / 2.0, 6)
            if bid is not None:
                return bid
            if ask is not None:
                return ask
    for lane in packet.get("lanes", []) or []:
        if not isinstance(lane, dict) or str(lane.get("pair") or "") != pair:
            continue
        for value in (
            (lane.get("technical_context") or {}).get("current_price_mid")
            if isinstance(lane.get("technical_context"), dict)
            else None,
            (lane.get("risk_metrics") or {}).get("entry_price") if isinstance(lane.get("risk_metrics"), dict) else None,
            lane.get("entry"),
        ):
            price = _optional_float(value)
            if price is not None:
                return price
    pair_context = (
        ((packet.get("market_context") or {}).get("pairs") or {}).get(pair)
        if isinstance(packet.get("market_context"), dict)
        else None
    )
    if isinstance(pair_context, dict):
        chart = pair_context.get("chart") if isinstance(pair_context.get("chart"), dict) else {}
        views = chart.get("views") if isinstance(chart.get("views"), dict) else {}
        for granularity in ("M1", "M5", "M15", "M30", "H1"):
            view = views.get(granularity)
            if isinstance(view, dict):
                close = _optional_float(view.get("close"))
                if close is not None:
                    return close
        levels = pair_context.get("levels") if isinstance(pair_context.get("levels"), dict) else {}
        for key in ("last_close", "daily_open", "pdc"):
            price = _optional_float(levels.get(key))
            if price is not None:
                return price
    return None


def draft_trader_decision(
    *,
    snapshot_path: Path,
    intents_path: Path = DEFAULT_ORDER_INTENTS,
    campaign_plan_path: Path = DEFAULT_CAMPAIGN_PLAN,
    strategy_profile_path: Path = DEFAULT_STRATEGY_PROFILE,
    market_story_profile_path: Path = DEFAULT_MARKET_STORY_PROFILE,
    market_status_path: Path = DEFAULT_MARKET_STATUS,
    target_state_path: Path = DEFAULT_DAILY_TARGET_STATE,
    pair_charts_path: Path = DEFAULT_PAIR_CHARTS,
    context_asset_charts_path: Path = DEFAULT_CONTEXT_ASSET_CHARTS,
    broker_instruments_path: Path = DEFAULT_BROKER_INSTRUMENTS,
    cross_asset_path: Path = DEFAULT_CROSS_ASSET_SNAPSHOT,
    flow_path: Path = DEFAULT_FLOW_SNAPSHOT,
    currency_strength_path: Path = DEFAULT_CURRENCY_STRENGTH,
    levels_path: Path = DEFAULT_LEVELS_SNAPSHOT,
    market_context_matrix_path: Path = DEFAULT_MARKET_CONTEXT_MATRIX,
    calendar_path: Path = DEFAULT_CALENDAR_SNAPSHOT,
    cot_path: Path = DEFAULT_COT_SNAPSHOT,
    option_skew_path: Path = DEFAULT_OPTION_SKEW,
    attack_advice_path: Path = DEFAULT_AI_ATTACK_ADVICE,
    capture_economics_path: Path = DEFAULT_CAPTURE_ECONOMICS,
    profitability_acceptance_path: Path = DEFAULT_PROFITABILITY_ACCEPTANCE,
    execution_timing_audit_path: Path = DEFAULT_EXECUTION_TIMING_AUDIT,
    coverage_optimization_path: Path = DEFAULT_COVERAGE_OPTIMIZATION,
    learning_audit_path: Path = DEFAULT_LEARNING_AUDIT,
    verification_ledger_path: Path = DEFAULT_VERIFICATION_LEDGER,
    self_improvement_audit_path: Path = DEFAULT_SELF_IMPROVEMENT_AUDIT,
    projection_ledger_path: Path = DEFAULT_PROJECTION_LEDGER,
    operator_precedent_path: Path = DEFAULT_OPERATOR_PRECEDENT_AUDIT,
    manual_market_context_path: Path = DEFAULT_MANUAL_MARKET_CONTEXT_AUDIT,
    trader_overrides_path: Path = DEFAULT_TRADER_OVERRIDES,
    predictive_limits_path: Path = DEFAULT_PREDICTIVE_LIMIT_ORDERS,
    news_items_path: Path = DEFAULT_NEWS_SNAPSHOT,
    news_health_path: Path = DEFAULT_NEWS_HEALTH,
    qr_trader_run_watchdog_path: Path = DEFAULT_QR_TRADER_RUN_WATCHDOG,
    guardian_receipt_consumption_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION,
    guardian_receipt_consumption_report_path: Path = DEFAULT_GUARDIAN_RECEIPT_CONSUMPTION_REPORT,
    guardian_receipt_operator_review_path: Path = DEFAULT_GUARDIAN_RECEIPT_OPERATOR_REVIEW,
    active_trader_contract_path: Path = DEFAULT_ACTIVE_TRADER_CONTRACT,
    active_opportunity_board_path: Path = DEFAULT_ACTIVE_OPPORTUNITY_BOARD,
    non_eurusd_live_grade_frontier_path: Path = DEFAULT_NON_EURUSD_LIVE_GRADE_FRONTIER,
    range_rail_geometry_repair_path: Path = DEFAULT_RANGE_RAIL_GEOMETRY_REPAIR,
    execution_ledger_path: Path = DEFAULT_EXECUTION_LEDGER_DB,
    output_path: Path = DEFAULT_CODEX_TRADER_DECISION_RESPONSE,
    report_path: Path = DEFAULT_TRADER_DECISION_DRAFT_REPORT,
    max_lanes: int = DEFAULT_GPT_MAX_LANES,
) -> TraderDecisionDraftSummary:
    watchdog_payload = _load_optional_json(qr_trader_run_watchdog_path)
    broker_snapshot_payload = _load_optional_json(snapshot_path)
    operator_review_payload = load_guardian_receipt_operator_review(guardian_receipt_operator_review_path)
    consumption_payload = build_guardian_receipt_consumption(
        watchdog_payload,
        existing=load_guardian_receipt_consumption(guardian_receipt_consumption_path),
        operator_review=operator_review_payload,
        broker_snapshot=broker_snapshot_payload,
    )
    write_guardian_receipt_consumption(
        consumption_payload,
        output_path=guardian_receipt_consumption_path,
        report_path=guardian_receipt_consumption_report_path,
    )
    brain = GPTTraderBrain(
        provider=None,
        intents_path=intents_path,
        campaign_plan_path=campaign_plan_path,
        strategy_profile_path=strategy_profile_path,
        market_story_profile_path=market_story_profile_path,
        market_status_path=market_status_path,
        target_state_path=target_state_path,
        pair_charts_path=pair_charts_path,
        context_asset_charts_path=context_asset_charts_path,
        broker_instruments_path=broker_instruments_path,
        cross_asset_path=cross_asset_path,
        flow_path=flow_path,
        currency_strength_path=currency_strength_path,
        levels_path=levels_path,
        market_context_matrix_path=market_context_matrix_path,
        calendar_path=calendar_path,
        cot_path=cot_path,
        option_skew_path=option_skew_path,
        attack_advice_path=attack_advice_path,
        capture_economics_path=capture_economics_path,
        profitability_acceptance_path=profitability_acceptance_path,
        execution_timing_audit_path=execution_timing_audit_path,
        coverage_optimization_path=coverage_optimization_path,
        learning_audit_path=learning_audit_path,
        verification_ledger_path=verification_ledger_path,
        self_improvement_audit_path=self_improvement_audit_path,
        projection_ledger_path=projection_ledger_path,
        operator_precedent_path=operator_precedent_path,
        manual_market_context_path=manual_market_context_path,
        trader_overrides_path=trader_overrides_path,
        predictive_limits_path=predictive_limits_path,
        news_items_path=news_items_path,
        news_health_path=news_health_path,
        qr_trader_run_watchdog_path=qr_trader_run_watchdog_path,
        guardian_receipt_consumption_path=guardian_receipt_consumption_path,
        guardian_receipt_operator_review_path=guardian_receipt_operator_review_path,
        active_trader_contract_path=active_trader_contract_path,
        active_opportunity_board_path=active_opportunity_board_path,
        non_eurusd_live_grade_frontier_path=non_eurusd_live_grade_frontier_path,
        range_rail_geometry_repair_path=range_rail_geometry_repair_path,
        execution_ledger_path=execution_ledger_path,
        max_lanes=max_lanes,
    )
    packet = brain._input_packet(snapshot_path)
    decision, blockers = _autonomous_decision_from_packet(packet)
    parsed = _decision_from_payload(decision)
    verification = DecisionVerifier(packet).verify(parsed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
    _write_trader_decision_draft_report(
        report_path,
        decision=decision,
        blockers=blockers,
        verification=verification,
        input_packet=packet,
    )
    return TraderDecisionDraftSummary(
        status="DRAFT_ACCEPTED" if verification.allowed else "DRAFT_REQUIRES_OPERATOR_REVIEW",
        output_path=output_path,
        report_path=report_path,
        action=parsed.action,
        selected_lane_id=parsed.selected_lane_id,
        selected_lane_ids=parsed.selected_lane_ids,
        blockers=tuple(blockers),
        verification_allowed=verification.allowed,
        verification_issues=tuple(issue.code for issue in verification.issues),
    )


def _autonomous_decision_from_packet(packet: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    live_ready_lane_ids = _tradeable_live_ready_lanes(packet)
    lanes_by_id = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", [])
        if isinstance(lane, dict)
    }
    candidate_lane_ids = _draft_candidate_lane_ids(packet, live_ready_lane_ids)
    selected_lane_ids = _draft_margin_aware_basket(packet, candidate_lane_ids, lanes_by_id)
    pending_cancel_order_ids = tuple(sorted(_self_improvement_pending_cancel_review_order_ids(packet)))
    pending_cancel_reasons = _self_improvement_pending_cancel_review_reasons(packet)

    blockers: list[str] = []
    position_close_reasons = _position_close_sidecar_reasons(packet)
    close_candidate, close_selection_blockers = _authorized_single_position_close_candidate(packet)
    if close_candidate is not None:
        return _close_decision_draft(packet, close_candidate), position_close_reasons
    blockers.extend(position_close_reasons)
    blockers.extend(close_selection_blockers)
    blockers.extend(_trade_exposure_blockers(packet))
    blockers.extend(_entry_thesis_sidecar_reasons(packet))
    blockers.extend(_projection_ledger_trade_blockers(packet))
    blockers.extend(_news_health_trade_blockers(packet))
    blockers.extend(_guardian_receipt_consumption_trade_blockers(packet))
    if not live_ready_lane_ids:
        blockers.append("NO_LIVE_READY_LANES")
    if live_ready_lane_ids and not selected_lane_ids:
        blockers.append("NO_MARGIN_AWARE_BASKET_FITS")
    if selected_lane_ids:
        blockers.extend(
            _self_improvement_trade_blockers(
                packet,
                decision_generated_at_utc=datetime.now(timezone.utc).isoformat(),
                resolved_pending_cancel_order_ids=pending_cancel_order_ids,
                selected_lane_ids=selected_lane_ids,
            )
        )
    user_alpha = _active_user_alpha_continuation(packet)
    if user_alpha is not None and not (
        selected_lane_ids
        and not blockers
        and _selected_lanes_include_user_alpha(packet, selected_lane_ids, user_alpha)
    ):
        blockers.append(_user_alpha_continuation_blocker_label(user_alpha, selected_lane_ids))

    if selected_lane_ids and not blockers:
        return _trade_decision_draft(
            packet,
            selected_lane_ids,
            lanes_by_id,
            cancel_order_ids=pending_cancel_order_ids,
        ), blockers
    if pending_cancel_order_ids and not live_ready_lane_ids and not position_close_reasons:
        return _cancel_pending_decision_draft(
            packet,
            cancel_order_ids=pending_cancel_order_ids,
            blockers=[*pending_cancel_reasons, *blockers],
        ), blockers
    return _non_trade_decision_draft(packet, blockers, live_ready_lane_ids, lanes_by_id), blockers


def _guardian_receipt_consumption_trade_blockers(packet: dict[str, Any]) -> list[str]:
    blockers = guardian_receipt_new_entry_blockers(
        packet.get("qr_trader_run_watchdog") if isinstance(packet.get("qr_trader_run_watchdog"), dict) else {},
        packet.get("guardian_receipt_consumption")
        if isinstance(packet.get("guardian_receipt_consumption"), dict)
        else {},
        packet.get("guardian_receipt_operator_review")
        if isinstance(packet.get("guardian_receipt_operator_review"), dict)
        else {},
        packet.get("broker_snapshot") if isinstance(packet.get("broker_snapshot"), dict) else {},
    )
    return [
        f"{item.get('code') or BLOCK_NEW_ENTRY_CODE}: {item.get('message') or 'normal new-entry routing blocked'}"
        for item in blockers
    ]


def _guardian_receipt_blocker_issue_code(blockers: list[str]) -> str:
    for blocker in blockers:
        code = str(blocker or "").split(":", 1)[0].strip()
        if code:
            return code
    return BLOCK_NEW_ENTRY_CODE


def _active_path_packet(
    *,
    active_trader_contract: dict[str, Any] | None,
    active_opportunity_board: dict[str, Any] | None,
    non_eurusd_frontier: dict[str, Any] | None,
    range_rail_geometry_repair: dict[str, Any] | None,
) -> dict[str, Any]:
    contract = active_trader_contract if isinstance(active_trader_contract, dict) else {}
    board = active_opportunity_board if isinstance(active_opportunity_board, dict) else {}
    frontier = non_eurusd_frontier if isinstance(non_eurusd_frontier, dict) else {}
    rail = range_rail_geometry_repair if isinstance(range_rail_geometry_repair, dict) else {}
    contract_state = contract.get("current_state") if isinstance(contract.get("current_state"), dict) else {}
    contract_board = (
        contract_state.get("active_opportunity_board")
        if isinstance(contract_state.get("active_opportunity_board"), dict)
        else {}
    )
    contract_frontier = (
        contract_state.get("non_eurusd_live_grade_frontier")
        if isinstance(contract_state.get("non_eurusd_live_grade_frontier"), dict)
        else {}
    )
    board_lane = _first_active_lane(
        contract_board.get("top_lane"),
        board.get("top_lane"),
        contract.get("top_lane"),
    )
    frontier_lane = _frontier_evidence_lane_from_active_path(
        contract_frontier,
        frontier,
    )
    top_lane = _first_active_lane(
        contract.get("top_lane"),
        board_lane,
        frontier_lane,
    )
    next_action = _first_text(
        contract.get("next_trade_enabling_action"),
        contract.get("next_active_path"),
        board.get("next_action"),
        board.get("next_active_path"),
        frontier.get("next_active_path"),
    )
    frontier_action = _first_text(
        frontier_lane.get("next_action"),
        contract_frontier.get("next_active_path"),
        frontier.get("next_active_path"),
    )
    rail_action = _first_text(
        rail.get("next_safe_action"),
        rail.get("next_action"),
        rail.get("next_contract_prompt"),
    )
    return {
        "evidence_ref": "active:contract",
        "contract_status": contract.get("status"),
        "selected_active_path": contract.get("selected_active_path"),
        "board_status": board.get("status"),
        "frontier_status": frontier.get("status"),
        "live_permission_allowed": bool(
            contract.get("live_permission_allowed")
            or board.get("live_permission_allowed")
            or frontier.get("live_permission_allowed")
        ),
        "top_lane": top_lane,
        "active_board_lane": board_lane,
        "frontier_evidence_lane": frontier_lane,
        "next_trade_enabling_action": next_action,
        "frontier_evidence_action": frontier_action,
        "range_rail_action": rail_action,
        "range_rail_status": rail.get("rail_status"),
    }


def _frontier_evidence_lane_from_active_path(
    *artifacts: dict[str, Any],
) -> dict[str, Any]:
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        checks = artifact.get("required_checks") if isinstance(artifact.get("required_checks"), dict) else {}
        lane = _first_active_lane(
            artifact.get("next_evidence_lane"),
            checks.get("next_evidence_lane"),
            artifact.get("top_non_eurusd_lane"),
            artifact.get("top_lane"),
        )
        if lane:
            return lane
    return {}


def _first_active_lane(*candidates: Any) -> dict[str, Any]:
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("lane_id") or "").strip():
            return dict(candidate)
    return {}


def _active_path_evidence_refs(active_path: dict[str, Any]) -> list[str]:
    if not isinstance(active_path, dict) or not active_path:
        return []
    refs = ["active:contract", "active:board", "active:non_eurusd_frontier", "active:range_rail"]
    lane_ids: list[str] = []
    for lane in _active_path_candidate_lanes(active_path):
        lane_id = str(lane.get("lane_id") or "").strip()
        if lane_id and lane_id not in lane_ids:
            lane_ids.append(lane_id)
    refs.extend(f"active:lane:{lane_id}" for lane_id in lane_ids)
    return refs


def _active_path_candidate_lanes(active_path: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(active_path, dict):
        return []
    candidates = [
        active_path.get("top_lane"),
        active_path.get("active_board_lane"),
        active_path.get("frontier_evidence_lane"),
    ]
    lanes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        lane_id = str(candidate.get("lane_id") or "").strip()
        if not lane_id or lane_id in seen:
            continue
        lanes.append(dict(candidate))
        seen.add(lane_id)
    return lanes


def _active_path_ordered_fallback_lanes(active_path: dict[str, Any]) -> list[dict[str, Any]]:
    lanes = _active_path_candidate_lanes(active_path)
    if not lanes or not _active_path_should_focus_frontier(active_path):
        return lanes
    frontier_lane_id = str(
        (active_path.get("frontier_evidence_lane") or {}).get("lane_id") or ""
    ).strip()
    if not frontier_lane_id:
        return lanes
    return sorted(lanes, key=lambda lane: 0 if lane.get("lane_id") == frontier_lane_id else 1)


def _active_path_should_focus_frontier(active_path: dict[str, Any]) -> bool:
    if not isinstance(active_path, dict):
        return False
    frontier = active_path.get("frontier_evidence_lane")
    board = active_path.get("active_board_lane") or active_path.get("top_lane")
    if not isinstance(frontier, dict) or not isinstance(board, dict):
        return False
    frontier_pair = str(frontier.get("pair") or "").upper()
    board_pair = str(board.get("pair") or "").upper()
    if not frontier_pair or frontier_pair == "EUR_USD" or board_pair != "EUR_USD":
        return False
    frontier_lane_id = str(frontier.get("lane_id") or "").strip()
    action_text = " ".join(
        str(active_path.get(key) or "")
        for key in ("next_trade_enabling_action", "frontier_evidence_action")
    )
    return bool(
        frontier_lane_id and (
            frontier_lane_id in action_text
            or "non_eurusd_live_grade_frontier" in action_text
            or "frontier evidence" in action_text
            or "Non-EUR frontier" in action_text
        )
    )


def _active_path_primary_lane(packet: dict[str, Any]) -> dict[str, Any]:
    active_path = packet.get("active_path")
    if not isinstance(active_path, dict):
        return {}
    lane = active_path.get("top_lane")
    return dict(lane) if isinstance(lane, dict) else {}


def _active_path_lane_id(packet: dict[str, Any]) -> str:
    return str(_active_path_primary_lane(packet).get("lane_id") or "").strip()


def _active_path_lane_for_market_read(active_lane: dict[str, Any]) -> dict[str, Any]:
    lane_id = str(active_lane.get("lane_id") or "").strip()
    pair = str(active_lane.get("pair") or "").strip()
    direction = str(active_lane.get("direction") or "").strip().upper()
    strategy = str(active_lane.get("strategy_family") or active_lane.get("method") or "").strip()
    blockers = list(active_lane.get("blockers") or [])
    vehicle = str(active_lane.get("vehicle") or active_lane.get("order_type") or "").strip().upper()
    order_type = {"STOP": "STOP-ENTRY"}.get(vehicle, vehicle)
    return {
        "lane_id": lane_id,
        "evidence_ref": f"active:lane:{lane_id}",
        "status": active_lane.get("status") or "EVIDENCE_ACQUISITION",
        "pair": pair,
        "direction": direction,
        "method": strategy,
        "order_type": order_type or "LIMIT",
        "entry": active_lane.get("entry"),
        "tp": active_lane.get("tp"),
        "sl": active_lane.get("sl"),
        "units": active_lane.get("units"),
        "risk_metrics": {},
        "thesis": f"Active read-only path for {lane_id}",
        "narrative": str(active_lane.get("next_action") or ""),
        "chart_story": f"Active board top lane {lane_id}; execution remains blocked by current evidence gates.",
        "invalidation": "Do not trade until the active path blockers clear and a fresh accepted TRADE receipt exists.",
        "risk_blockers": [],
        "strategy_blockers": [],
        "live_blockers": blockers,
        "forecast": {},
        "opportunity": {},
        "target_path": {},
        "self_improvement": {},
        "market_close_leak_family": {},
        "position_building": {},
        "market_context_matrix": {},
        "technical_context": {},
    }


def _draft_active_fallback_lane_ids(
    packet: dict[str, Any],
    lanes_by_id: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    active_path = packet.get("active_path")
    if not isinstance(active_path, dict):
        return ()
    ordered_lane_ids: list[str] = []
    for active_lane in _active_path_ordered_fallback_lanes(active_path):
        lane_id = str(active_lane.get("lane_id") or "").strip()
        if not lane_id:
            continue
        if lane_id not in lanes_by_id:
            lanes_by_id[lane_id] = _active_path_lane_for_market_read(active_lane)
        if lane_id not in ordered_lane_ids:
            ordered_lane_ids.append(lane_id)
    return tuple(ordered_lane_ids)


def _active_path_blocker_summary(packet: dict[str, Any]) -> str:
    active_path = packet.get("active_path")
    if not isinstance(active_path, dict):
        return ""
    lane = active_path.get("top_lane") if isinstance(active_path.get("top_lane"), dict) else {}
    lane_id = str(lane.get("lane_id") or "").strip()
    next_action = str(active_path.get("next_trade_enabling_action") or "").strip()
    frontier_lane = (
        active_path.get("frontier_evidence_lane")
        if isinstance(active_path.get("frontier_evidence_lane"), dict)
        else {}
    )
    frontier_lane_id = str(frontier_lane.get("lane_id") or "").strip()
    frontier_action = str(active_path.get("frontier_evidence_action") or "").strip()
    rail_action = str(active_path.get("range_rail_action") or "").strip()
    parts: list[str] = []
    if lane_id:
        parts.append(f"ACTIVE_PATH_LANE {lane_id}")
    if next_action:
        parts.append(f"NEXT_ACTIVE_ACTION {next_action}")
    if frontier_lane_id and frontier_lane_id != lane_id:
        parts.append(f"FRONTIER_LANE {frontier_lane_id}")
    if frontier_action:
        parts.append(f"FRONTIER_ACTION {frontier_action}")
    if rail_action:
        parts.append(f"RANGE_RAIL_ACTION {rail_action}")
    return "; ".join(parts)


def _draft_candidate_lane_ids(packet: dict[str, Any], live_ready_lane_ids: list[str]) -> list[str]:
    lane_map = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", []) or []
        if isinstance(lane, dict)
    }
    ordinary_lane_ids = [
        lane_id
        for lane_id in live_ready_lane_ids
        if not _lane_is_predictive_scout(lane_map.get(lane_id))
    ]
    scout_lane_ids = [
        lane_id
        for lane_id in live_ready_lane_ids
        if _lane_is_predictive_scout(lane_map.get(lane_id))
    ]
    candidate_pool = ordinary_lane_ids or scout_lane_ids[:1]
    attack_lane_ids = [
        lane_id
        for lane_id in _attack_recommended_tradeable_lane_ids(packet, candidate_pool)
        if lane_id in candidate_pool
    ]
    ordered = list(attack_lane_ids or candidate_pool)
    for lane_id in candidate_pool:
        if lane_id not in ordered:
            ordered.append(lane_id)
    return ordered


def _lane_is_predictive_scout(lane: dict[str, Any] | None) -> bool:
    if not isinstance(lane, dict):
        return False
    scout = lane.get("predictive_scout")
    return bool(
        isinstance(scout, dict)
        and (
            scout.get("claimed") is True
            or (
                scout.get("enabled") is True
                and scout.get("canonical_rule_supported") is True
            )
        )
    )


def _predictive_scout_lane_verification_issues(
    lane: dict[str, Any] | None,
) -> list[VerificationIssue]:
    if not isinstance(lane, dict):
        return [
            VerificationIssue(
                "PREDICTIVE_SCOUT_PACKET_MISSING",
                "selected predictive SCOUT lane is absent from the verifier packet",
            )
        ]
    scout = lane.get("predictive_scout")
    if not isinstance(scout, dict) or scout.get("canonical_rule_supported") is not True:
        return [
            VerificationIssue(
                "PREDICTIVE_SCOUT_CANONICAL_RULE_REQUIRED",
                "predictive SCOUT must carry a canonical contrarian rule and immutable digest",
            )
        ]
    issues: list[VerificationIssue] = []
    if str(lane.get("order_type") or "").upper() != "LIMIT":
        issues.append(VerificationIssue("PREDICTIVE_SCOUT_LIMIT_ONLY", "predictive SCOUT must use LIMIT"))
    if (_optional_int(lane.get("units")) or 0) < MIN_PRODUCTION_LOT_UNITS:
        issues.append(
            VerificationIssue(
                "PREDICTIVE_SCOUT_MIN_LOT_REQUIRED",
                f"predictive SCOUT must use at least the {MIN_PRODUCTION_LOT_UNITS}u production floor",
            )
        )
    tier = str(scout.get("risk_tier") or "").upper()
    nav_jpy = _optional_float(scout.get("nav_jpy_at_sizing"))
    max_risk_pct = _optional_float(scout.get("max_risk_pct_nav"))
    max_loss_jpy = _optional_float(scout.get("max_loss_jpy"))
    planned_risk_jpy = _optional_float(scout.get("planned_initial_risk_jpy"))
    sizing_digest = str(scout.get("sizing_digest") or "")
    if (
        tier not in {"DISCOVERY", "EMERGING", "ESTABLISHED", "STRONG", "PROVEN"}
        or nav_jpy is None
        or nav_jpy <= 0.0
        or max_risk_pct is None
        or max_risk_pct <= 0.0
        or max_risk_pct > 1.0
        or max_loss_jpy is None
        or max_loss_jpy <= 0.0
        or planned_risk_jpy is None
        or planned_risk_jpy <= 0.0
        or planned_risk_jpy > max_loss_jpy
        or not sizing_digest.startswith("psd-")
    ):
        issues.append(
            VerificationIssue(
                "PREDICTIVE_SCOUT_NAV_RISK_PACKET_INVALID",
                "predictive SCOUT must carry its pre-verifier current-NAV risk tier, cap, planned SL risk, and sizing digest",
            )
        )
    if scout.get("promotion_allowed") is not False:
        issues.append(
            VerificationIssue(
                "PREDICTIVE_SCOUT_AUTO_PROMOTION_FORBIDDEN",
                "predictive SCOUT cannot carry automatic promotion permission",
            )
        )
    if str(scout.get("vehicle_proof_status") or "").upper() != "UNPROVEN_PASSIVE_LIMIT":
        issues.append(
            VerificationIssue(
                "PREDICTIVE_SCOUT_VEHICLE_PROOF_MISSTATED",
                "historical immediate-entry replay must not be presented as proof of the passive LIMIT vehicle",
            )
        )
    if scout.get("rule_is_vehicle_proof") is not False:
        issues.append(
            VerificationIssue(
                "PREDICTIVE_SCOUT_RULE_IS_NOT_VEHICLE_PROOF",
                "the source forecast-failure rule is hypothesis evidence only",
            )
        )
    return issues


def _draft_margin_aware_basket(
    packet: dict[str, Any],
    candidate_lane_ids: list[str],
    lanes_by_id: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    """Select the one exact lane that the hourly market read may authorize.

    The gateway still owns portfolio-wide basket expansion and its independent
    risk/margin checks.  The deterministic baseline, however, carries only one
    ``market_read_first`` thesis, so publishing several pair/side lanes here
    would let one AI read appear to authorize unrelated trades.  A later lane
    therefore needs a fresh receipt and its own market read.
    """

    margin_room = _draft_effective_margin_room(packet)
    for lane_id in candidate_lane_ids:
        lane = lanes_by_id.get(lane_id)
        if not lane:
            continue
        if lane.get("status") != "LIVE_READY":
            continue
        if (
            lane.get("risk_blockers")
            or lane.get("strategy_blockers")
            or lane.get("live_blockers")
        ):
            continue
        margin = _optional_float((lane.get("risk_metrics") or {}).get("estimated_margin_jpy")) or 0.0
        if margin_room is not None and margin > margin_room:
            continue
        return (lane_id,)
    return ()


def _draft_effective_margin_room(packet: dict[str, Any]) -> float | None:
    account = packet.get("broker_snapshot", {}).get("account")
    if not isinstance(account, dict):
        return None
    available = _optional_float(account.get("margin_available_jpy"))
    nav = _optional_float(account.get("nav_jpy"))
    used = _optional_float(account.get("margin_used_jpy"))
    if available is None or nav is None or used is None:
        return None
    utilization_room = nav * (RiskPolicy().max_margin_utilization_pct / 100.0) - used
    base_room = min(available, utilization_room)
    # Same C-4 engineering buffer documented in AGENT_CONTRACT. It absorbs
    # intra-cycle margin drift and is not a market edge threshold.
    return max(0.0, base_room) * MARGIN_AWARE_BASKET_BUFFER


def _draft_market_read_first(
    packet: dict[str, Any],
    lanes_by_id: dict[str, dict[str, Any]],
    *,
    selected_lane_ids: tuple[str, ...] = (),
    fallback_lane_ids: tuple[str, ...] = (),
    focus_pair: str = "",
) -> dict[str, Any]:
    lane = _market_read_primary_lane(lanes_by_id, selected_lane_ids, fallback_lane_ids)
    pair = str(focus_pair or (lane or {}).get("pair") or _market_read_fallback_pair(packet) or "").strip()
    side = str((lane or {}).get("direction") or "").strip().upper()
    if not side:
        side = _market_read_chart_direction(packet, pair)
    forecast = (lane or {}).get("forecast") if isinstance((lane or {}).get("forecast"), dict) else {}
    direction = _market_read_prediction_direction(side, forecast)
    prediction_side = _market_read_side(direction)
    base, quote = _pair_currencies(pair)
    currency_side = prediction_side or side
    bought = base if currency_side == "LONG" else quote
    sold = quote if currency_side == "LONG" else base
    if currency_side not in {"LONG", "SHORT"}:
        bought = base or "UNKNOWN"
        sold = quote or "UNKNOWN"
    current_price = _current_market_read_price(packet, pair)
    lane_aligned = prediction_side is not None and prediction_side == side
    entry_price = (
        _optional_float((lane or {}).get("entry")) if lane_aligned else current_price
    ) or current_price
    geometry_basis = current_price or entry_price
    forecast_target = _market_read_valid_target_price(
        direction,
        geometry_basis,
        _optional_float(forecast.get("forecast_target_price")),
    )
    forecast_invalidation = _market_read_valid_invalidation_price(
        direction,
        geometry_basis,
        _optional_float(forecast.get("forecast_invalidation_price")),
    )
    lane_target = _market_read_valid_target_price(
        direction,
        geometry_basis,
        _optional_float((lane or {}).get("tp")) if lane_aligned else None,
    )
    lane_invalidation = _market_read_valid_invalidation_price(
        direction,
        geometry_basis,
        _optional_float((lane or {}).get("sl")) if lane_aligned else None,
    )
    target_price = forecast_target if forecast_target is not None else lane_target
    invalidation_price = (
        forecast_invalidation if forecast_invalidation is not None else lane_invalidation
    )
    tape_state = _market_read_tape_state(packet, pair, lane)
    expected_path = _market_read_expected_path(packet, pair, direction, current_price, target_price)
    invalidation = _price_sentence(
        invalidation_price,
        fallback=(
            str((lane or {}).get("invalidation") or "structure breaks")
            if lane_aligned
            else "forecast invalidation unavailable; refresh forecast geometry"
        ),
    )
    target_zone = _price_sentence(target_price, fallback=_market_read_target_zone_from_context(packet, pair, direction))
    vehicle = _market_read_vehicle((lane or {}).get("order_type")) if lane_aligned else "MARKET"
    building_answer = _market_read_building_style_allowed(lane)
    return {
        "naked_read": {
            "currency_bought": bought or "UNKNOWN",
            "currency_sold": sold or "UNKNOWN",
            "cleanest_pair_expression": pair or "UNKNOWN_PAIR",
            "is_cleanest_currency_theme": _market_read_cleanest_theme_answer(
                pair,
                bought,
                sold,
                lane if lane_aligned else None,
            ),
            "location_24h": _market_read_location_24h(lane),
            "h1_h4_alignment": _market_read_h1_h4_alignment(lane, currency_side),
            "tape_state": tape_state,
            "known_winning_trade_shape_match": _market_read_shape_match_answer(
                packet,
                lane if lane_aligned else None,
            ),
            "proposed_building_style_allowed": (
                building_answer if lane_aligned else "YES - SINGLE forecast hypothetical only"
            ),
            "thesis_state": _market_read_thesis_state(lane),
            "what_price_is_trying_to_do_now": _market_read_now_sentence(packet, pair, direction, current_price),
        },
        "next_30m_prediction": {
            "pair": pair or "UNKNOWN_PAIR",
            "direction": direction,
            "expected_path": expected_path,
            "target_zone": target_zone,
            "invalidation": invalidation,
        },
        "next_2h_prediction": {
            "pair": pair or "UNKNOWN_PAIR",
            "direction": direction,
            "expected_path": _market_read_two_hour_path(packet, pair, direction, target_zone, invalidation),
            "target_zone": target_zone,
            "invalidation": invalidation,
        },
        "best_trade_if_forced": {
            "pair": pair or "UNKNOWN_PAIR",
            "direction": prediction_side or _market_read_chart_direction(packet, pair),
            "vehicle": vehicle,
            "entry": _price_sentence(entry_price, fallback="current executable price only after broker refresh"),
            "tp": _price_sentence(target_price, fallback=target_zone),
            "sl": _price_sentence(invalidation_price, fallback=invalidation),
            "why_this_pays": _market_read_forced_trade_reason(
                packet,
                lane,
                pair,
                direction,
                lane_aligned=lane_aligned,
            ),
        },
    }


def _market_read_primary_lane(
    lanes_by_id: dict[str, dict[str, Any]],
    selected_lane_ids: tuple[str, ...],
    fallback_lane_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    for lane_id in (*selected_lane_ids, *fallback_lane_ids):
        lane = lanes_by_id.get(lane_id)
        if isinstance(lane, dict) and lane.get("pair"):
            return lane
    for lane in lanes_by_id.values():
        if isinstance(lane, dict) and lane.get("pair"):
            return lane
    return None


def _market_read_cleanest_theme_answer(
    pair: str,
    bought: str,
    sold: str,
    lane: dict[str, Any] | None,
) -> str:
    if pair and lane:
        return f"YES - {pair} is the current cleanest expression for buying {bought or 'UNKNOWN'} and selling {sold or 'UNKNOWN'} in the selected lane."
    if pair:
        return f"UNKNOWN - {pair} is the available expression, but no selected lane proves it is cleanest."
    return "UNKNOWN - no pair expression is available in the packet."


def _market_read_location_24h(lane: dict[str, Any] | None) -> str:
    technical = (lane or {}).get("technical_context")
    if not isinstance(technical, dict):
        return "UNKNOWN"
    percentile = _optional_float(
        technical.get("entry_price_percentile_24h") or technical.get("price_percentile_24h")
    )
    if percentile is None:
        return "UNKNOWN"
    if percentile < (1.0 / 3.0):
        return "LOWER"
    if percentile > (2.0 / 3.0):
        return "UPPER"
    return "MIDDLE"


def _market_read_h1_h4_alignment(lane: dict[str, Any] | None, side: str) -> str:
    technical = (lane or {}).get("technical_context")
    if not isinstance(technical, dict):
        return "H1=UNKNOWN; H4=UNKNOWN"
    h1 = _market_read_tf_alignment(side, technical.get("h1_regime"), "H1")
    h4 = _market_read_tf_alignment(side, technical.get("h4_regime"), "H4")
    return f"H1={h1}; H4={h4}"


def _market_read_tf_alignment(side: str, regime: Any, tf: str) -> str:
    text = str(regime or "").upper()
    if not text or side not in {"LONG", "SHORT"}:
        return f"{tf}_UNKNOWN"
    if "UP" in text:
        direction = "UP"
    elif "DOWN" in text:
        direction = "DOWN"
    else:
        return f"{tf}_UNKNOWN"
    side_direction = "UP" if side == "LONG" else "DOWN"
    relation = "WITH" if direction == side_direction else "AGAINST"
    return f"{relation}_{tf}_TREND"


def _market_read_shape_match_answer(packet: dict[str, Any], lane: dict[str, Any] | None) -> str:
    lane_id = str((lane or {}).get("lane_id") or "")
    pair = str((lane or {}).get("pair") or "")
    operator_precedent = packet.get("operator_precedent")
    engine = (
        operator_precedent.get("trade_shape_engine")
        if isinstance(operator_precedent, dict) and isinstance(operator_precedent.get("trade_shape_engine"), dict)
        else {}
    )
    for item in engine.get("shape_matched_live_ready_lanes", []) or []:
        if isinstance(item, dict) and lane_id and str(item.get("lane_id") or "") == lane_id:
            return f"MATCH - generalized 2025 operator trade shape matches {lane_id}."
    pair_summary = (engine.get("pair_summaries") or {}).get(pair) if isinstance(engine.get("pair_summaries"), dict) else None
    if isinstance(pair_summary, dict):
        status = ((pair_summary.get("precedent_match") or {}).get("status") or "UNKNOWN")
        return f"{status} - generalized 2025 operator trade-shape engine summary for {pair}."
    return "UNKNOWN - operator-precedent trade-shape engine is not available in this packet."


def _market_read_building_style_allowed(lane: dict[str, Any] | None) -> str:
    building = (lane or {}).get("position_building")
    if not isinstance(building, dict):
        return "YES - SINGLE"
    add_type = str(building.get("same_pair_add_type") or "").upper()
    if add_type == "AVERAGE_INTO_ADVERSE":
        return "YES - BOUNDED_ADVERSE_ADD only after current risk/gateway validation"
    if add_type == "PYRAMID_WITH_MOVE":
        return "NO - WITH_MOVE_PYRAMID is blocked by generalized operator precedent"
    if add_type:
        return f"UNKNOWN - {add_type} needs current risk classification"
    return "YES - SINGLE"


def _market_read_thesis_state(lane: dict[str, Any] | None) -> str:
    if not lane:
        return "UNKNOWN"
    status = str(lane.get("status") or "").upper()
    text = " ".join(
        [status]
        + [str(item or "") for item in lane.get("live_blockers", []) or []]
        + [str(item or "") for item in lane.get("risk_blockers", []) or []]
        + [str(item or "") for item in lane.get("strategy_blockers", []) or []]
    ).upper()
    if "EMERGENCY" in text or "MARGIN_CLOSEOUT" in text:
        return "EMERGENCY"
    if "INVALIDATED" in text or "THESIS_BROKEN" in text or "RECOMMEND_CLOSE" in text:
        return "INVALIDATED"
    if status == "LIVE_READY":
        return "ALIVE"
    if "BLOCK" in text or "WATCH_ONLY" in text or "NEGATIVE_EXPECTANCY" in text:
        return "WOUNDED"
    return "ALIVE"


def _market_read_fallback_pair(packet: dict[str, Any]) -> str:
    pairs = ((packet.get("market_context") or {}).get("pairs") or {}) if isinstance(packet.get("market_context"), dict) else {}
    if isinstance(pairs, dict):
        for pair in pairs:
            if str(pair).strip():
                return str(pair)
    for order in _pending_entry_orders(packet):
        pair = str(order.get("pair") or "").strip()
        if pair:
            return pair
    for position in (packet.get("broker_snapshot", {}) or {}).get("position_summaries", []) or []:
        if isinstance(position, dict) and str(position.get("pair") or "").strip():
            return str(position.get("pair"))
    return ""


def _pair_currencies(pair: str) -> tuple[str, str]:
    parts = [part for part in pair.split("_") if part]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def _market_read_prediction_direction(side: str, forecast: dict[str, Any]) -> str:
    raw = str(forecast.get("forecast_direction") or "").strip().upper()
    if raw in {"UP", "LONG", "BUY", "BULL", "BULLISH"}:
        return "LONG"
    if raw in {"DOWN", "SHORT", "SELL", "BEAR", "BEARISH"}:
        return "SHORT"
    if raw:
        return raw
    if side in {"LONG", "SHORT"}:
        return side
    return "RANGE"


def _market_read_side(direction: str) -> str | None:
    if _market_read_longish(direction):
        return "LONG"
    if _market_read_shortish(direction):
        return "SHORT"
    return None


def _market_read_valid_target_price(
    direction: str,
    basis: float | None,
    price: float | None,
) -> float | None:
    if price is None:
        return None
    if basis is not None and _market_read_target_on_wrong_side(direction, basis, [price]):
        return None
    return price


def _market_read_valid_invalidation_price(
    direction: str,
    basis: float | None,
    price: float | None,
) -> float | None:
    if price is None:
        return None
    if basis is not None and _market_read_invalidation_on_wrong_side(direction, basis, [price]):
        return None
    return price


def _market_read_chart_direction(packet: dict[str, Any], pair: str) -> str:
    pair_context = (
        ((packet.get("market_context") or {}).get("pairs") or {}).get(pair)
        if isinstance(packet.get("market_context"), dict)
        else None
    )
    chart = pair_context.get("chart") if isinstance(pair_context, dict) and isinstance(pair_context.get("chart"), dict) else {}
    long_score = _optional_float(chart.get("long_score"))
    short_score = _optional_float(chart.get("short_score"))
    if long_score is not None and short_score is not None:
        if long_score > short_score:
            return "LONG"
        if short_score > long_score:
            return "SHORT"
    regime = str(chart.get("dominant_regime") or "").upper()
    if "DOWN" in regime:
        return "SHORT"
    if "UP" in regime:
        return "LONG"
    return "RANGE"


def _market_read_tape_state(
    packet: dict[str, Any],
    pair: str,
    lane: dict[str, Any] | None,
) -> str:
    pair_context = (
        ((packet.get("market_context") or {}).get("pairs") or {}).get(pair)
        if isinstance(packet.get("market_context"), dict)
        else None
    )
    chart = pair_context.get("chart") if isinstance(pair_context, dict) and isinstance(pair_context.get("chart"), dict) else {}
    pieces = [
        str(chart.get("dominant_regime") or ""),
        str((lane or {}).get("method") or ""),
        str(((lane or {}).get("technical_context") or {}).get("m5_regime") or "")
        if isinstance((lane or {}).get("technical_context"), dict)
        else "",
        str(((lane or {}).get("technical_context") or {}).get("h1_regime") or "")
        if isinstance((lane or {}).get("technical_context"), dict)
        else "",
    ]
    text = " ".join(pieces).upper()
    if "SQUEEZE" in text:
        return "SQUEEZE"
    if "FADE" in text or "FAIL" in text:
        return "FADE"
    if "RANGE" in text:
        return "RANGE"
    if "ROTATION" in text or "CHOP" in text:
        return "ROTATION"
    if "TREND" in text or "UP" in text or "DOWN" in text:
        return "TREND"
    return "ROTATION"


def _market_read_expected_path(
    packet: dict[str, Any],
    pair: str,
    direction: str,
    current_price: float | None,
    target_price: float | None,
) -> str:
    start = f"from {current_price}" if current_price is not None else "from current mid"
    target = f"toward {target_price}" if target_price is not None else "toward the nearest cited level"
    context = _market_read_chart_story(packet, pair)
    return f"Next 30m: {pair or 'UNKNOWN_PAIR'} {direction} should move {start} {target}; {context}"


def _market_read_two_hour_path(
    packet: dict[str, Any],
    pair: str,
    direction: str,
    target_zone: str,
    invalidation: str,
) -> str:
    context = _market_read_chart_story(packet, pair)
    return (
        f"Next 2h: {pair or 'UNKNOWN_PAIR'} {direction} should either extend into {target_zone} "
        f"or fail first at {invalidation}; {context}"
    )


def _market_read_now_sentence(
    packet: dict[str, Any],
    pair: str,
    direction: str,
    current_price: float | None,
) -> str:
    price = f"around {current_price}" if current_price is not None else "around the latest chart close"
    return f"{pair or 'UNKNOWN_PAIR'} price is trying to express {direction} pressure {price} before execution filters are applied."


def _market_read_target_zone_from_context(packet: dict[str, Any], pair: str, direction: str) -> str:
    pair_context = (
        ((packet.get("market_context") or {}).get("pairs") or {}).get(pair)
        if isinstance(packet.get("market_context"), dict)
        else None
    )
    levels = pair_context.get("levels") if isinstance(pair_context, dict) and isinstance(pair_context.get("levels"), dict) else {}
    if _market_read_longish(direction):
        for key in ("pdh", "r1"):
            price = _optional_float((levels.get("standard_pivot") or {}).get(key) if key == "r1" else levels.get(key))
            if price is not None:
                return f"{price}"
    if _market_read_shortish(direction):
        for key in ("pdl", "s1"):
            price = _optional_float((levels.get("standard_pivot") or {}).get(key) if key == "s1" else levels.get(key))
            if price is not None:
                return f"{price}"
    return "nearest cited support/resistance zone from chart and levels packet"


def _market_read_chart_story(packet: dict[str, Any], pair: str) -> str:
    pair_context = (
        ((packet.get("market_context") or {}).get("pairs") or {}).get(pair)
        if isinstance(packet.get("market_context"), dict)
        else None
    )
    chart = pair_context.get("chart") if isinstance(pair_context, dict) and isinstance(pair_context.get("chart"), dict) else {}
    story = str(chart.get("chart_story") or "").strip()
    return story or "chart refs provide the current tape but do not grant execution permission"


def _price_sentence(price: float | None, *, fallback: str) -> str:
    if price is not None:
        return f"{price}"
    return str(fallback or "not specified").strip() or "not specified"


def _market_read_vehicle(order_type: Any) -> str:
    text = str(order_type or "").strip().upper()
    if "STOP" in text:
        return "STOP"
    if "LIMIT" in text or "TOUCHED" in text:
        return "LIMIT"
    return "MARKET"


def _market_read_forced_trade_reason(
    packet: dict[str, Any],
    lane: dict[str, Any] | None,
    pair: str,
    direction: str,
    *,
    lane_aligned: bool = True,
) -> str:
    if lane and not lane_aligned:
        return (
            f"Forced trade is only a {direction} forecast-geometry hypothetical because the active "
            "blocked lane points the other way; execution requires a newly generated aligned intent "
            "and every normal verifier/gateway gate."
        )
    if lane:
        risk = lane.get("risk_metrics") if isinstance(lane.get("risk_metrics"), dict) else {}
        rr = risk.get("reward_risk")
        return (
            f"Forced trade only pays if the naked {direction} read reaches its target before invalidation; "
            f"current lane geometry rr={rr} remains an execution filter, not the market read."
        )
    context = _market_read_chart_story(packet, pair)
    return (
        f"Forced trade is only hypothetical: {context}. Execution still requires a refreshed LIVE_READY lane."
    )


def _market_read_prediction_summary(market_read: dict[str, Any]) -> str:
    next_30m = market_read.get("next_30m_prediction") if isinstance(market_read.get("next_30m_prediction"), dict) else {}
    next_2h = market_read.get("next_2h_prediction") if isinstance(market_read.get("next_2h_prediction"), dict) else {}
    if not next_30m and not next_2h:
        return "MARKET READ FIRST next 30m/next 2h prediction is unavailable in this draft."
    return (
        f"MARKET READ FIRST next 30m {next_30m.get('pair') or 'UNKNOWN_PAIR'} "
        f"{next_30m.get('direction') or 'UNKNOWN'} toward {next_30m.get('target_zone') or 'unknown'}; "
        f"next 2h {next_2h.get('pair') or 'UNKNOWN_PAIR'} {next_2h.get('direction') or 'UNKNOWN'} "
        f"toward {next_2h.get('target_zone') or 'unknown'}."
    )


def _trade_decision_draft(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    lanes_by_id: dict[str, dict[str, Any]],
    *,
    cancel_order_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    primary = lanes_by_id[selected_lane_ids[0]]
    pair = str(primary.get("pair") or "")
    side = str(primary.get("direction") or "")
    method = str(primary.get("method") or "RANGE_ROTATION")
    market_read = _draft_market_read_first(
        packet,
        lanes_by_id,
        selected_lane_ids=selected_lane_ids,
        fallback_lane_ids=selected_lane_ids,
    )
    market_summary = _market_read_prediction_summary(market_read)
    refs = _draft_trade_evidence_refs(packet, selected_lane_ids, lanes_by_id, cancel_order_ids=cancel_order_ids)
    cited_chart_refs = ", ".join(ref for ref in refs if ref.startswith(f"chart:{pair}:")) or "current chart refs"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_read_first": market_read,
        "action": "TRADE",
        "selected_lane_id": selected_lane_ids[0],
        "selected_lane_ids": list(selected_lane_ids),
        "cancel_order_ids": list(cancel_order_ids),
        "close_trade_ids": [],
        "confidence": "HIGH",
        "thesis": f"{market_summary} Execution filter then selects: {_draft_trade_thesis(selected_lane_ids, lanes_by_id)}",
        "method": method,
        "narrative": str(
            f"{market_summary} "
            + str(
                primary.get("narrative")
                or primary.get("thesis")
                or "Selected current LIVE_READY lane(s) from broker-truth intent evidence."
            )
        ),
        "chart_story": str(
            primary.get("chart_story")
            or f"Draft cited current chart evidence for {pair}: {cited_chart_refs}."
        ),
        "invalidation": str(
            primary.get("invalidation")
            or "Selected lane leaves LIVE_READY or broker truth invalidates the quoted structure."
        ),
        "rejected_alternatives": _draft_rejected_alternatives(packet, selected_lane_ids, lanes_by_id),
        "risk_notes": _draft_risk_notes(selected_lane_ids, lanes_by_id, cancel_order_ids=cancel_order_ids),
        "evidence_refs": refs,
        "twenty_minute_plan": _draft_twenty_minute_plan(
            action="TRADE",
            pair=pair,
            side=side,
            selected_lane_ids=selected_lane_ids,
            refs=refs,
            market_read_first=market_read,
        ),
        "strategy_reviews": [],
        "specialist_reviews": _draft_specialist_reviews(primary, refs),
        "operator_summary": (
            f"{market_summary} Autonomous trader draft selected current LIVE_READY lane(s) from the same "
            "broker/market/news packet used by gpt-trader-decision; gateway validation remains final."
        ),
    }


def _close_decision_draft(
    packet: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    trade_id = str(candidate["trade_id"])
    pair = str(candidate["pair"])
    side = str(candidate["side"]).upper()
    recommendations = [
        item
        for item in candidate.get("recommendations", [])
        if isinstance(item, dict)
    ]
    reasons = [
        str(item.get("reason") or "position thesis is invalidated").strip()
        for item in recommendations
    ]
    reason_text = "; ".join(reason for reason in reasons if reason) or "position thesis is invalidated"
    source_text = ", ".join(
        dict.fromkeys(
            str(item.get("source") or "position_sidecar").strip()
            for item in recommendations
        )
    )
    market_read = _draft_market_read_first(packet, {}, focus_pair=pair)
    market_summary = _market_read_prediction_summary(market_read)
    refs = _draft_close_evidence_refs(packet, candidate)
    gate_b = (
        "standing hard Gate B"
        if candidate.get("gate_b_standing_authorized")
        else "fresh explicit operator Gate B"
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_read_first": market_read,
        "action": "CLOSE",
        "selected_lane_id": None,
        "selected_lane_ids": [],
        "cancel_order_ids": [],
        "close_trade_ids": [trade_id],
        "confidence": "HIGH",
        "thesis": (
            f"{market_summary} After that read, close only trader-owned {pair} {side} trade {trade_id}: "
            f"current {source_text or 'position sidecar'} evidence blocks non-CLOSE actions and {gate_b} is satisfied."
        ),
        "method": "POSITION_MANAGEMENT",
        "narrative": (
            f"The exact current position and sidecar identity match trade {trade_id} {pair} {side}. "
            f"Gate A reason: {reason_text}."
        ),
        "chart_story": (
            f"Current {pair} market read and position sidecar evidence are cited in this receipt; "
            "the gateway must refresh the executable quote and spread before sending the close."
        ),
        "invalidation": (
            "The cited current structural exit condition is the invalidation; refresh and fail closed if it no longer holds."
        ),
        "rejected_alternatives": [
            "HOLD rejected because current authorized Gate A evidence explicitly blocks non-CLOSE actions.",
            "TRADE/re-entry rejected for this cycle; broker truth and intents require a fresh next-cycle receipt.",
            "Mass close rejected; this receipt binds exactly one current trader-owned trade id.",
        ],
        "risk_notes": [
            f"Close scope is immutable and limited to trade {trade_id}; no entry or pending-order cancellation is authorized.",
            "Profitability and execution-timing evidence refs are bound when present so this loss-side close cannot bypass the MARKET_ORDER_TRADE_CLOSE leak audit.",
            "If the latest quote, spread cap, position ownership, Gate A, or Gate B no longer matches, fail closed and refresh the full cycle.",
        ],
        "evidence_refs": refs,
        "strategy_reviews": [],
        "specialist_reviews": [],
        "operator_close_authorized": bool(candidate.get("gate_b_explicit_operator_authorized")),
        "operator_summary": (
            f"Close only trader-owned {pair} {side} trade {trade_id}, end the cycle, then refresh broker truth "
            "before any separate entry decision."
        ),
    }


def _non_trade_decision_draft(
    packet: dict[str, Any],
    blockers: list[str],
    live_ready_lane_ids: list[str],
    lanes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    refs = _draft_non_trade_evidence_refs(packet, live_ready_lane_ids, lanes_by_id)
    action = "WAIT" if live_ready_lane_ids else "REQUEST_EVIDENCE"
    active_summary = _active_path_blocker_summary(packet)
    blocker_items = [*blockers[:5]]
    if active_summary:
        blocker_items.append(active_summary)
    blocker_text = "; ".join(blocker_items) if blocker_items else "NO_EXECUTABLE_ENTRY_ROUTE"
    fallback_lane_ids = tuple(live_ready_lane_ids[:1]) or _draft_active_fallback_lane_ids(packet, lanes_by_id)
    market_read = _draft_market_read_first(
        packet,
        lanes_by_id,
        fallback_lane_ids=fallback_lane_ids,
    )
    market_summary = _market_read_prediction_summary(market_read)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_read_first": market_read,
        "action": action,
        "selected_lane_id": None,
        "selected_lane_ids": [],
        "cancel_order_ids": [],
        "close_trade_ids": [],
        "confidence": "HIGH",
        "thesis": f"{market_summary} After that read, do not draft a fresh entry while named blocker(s) remain: {blocker_text}",
        "method": "EVENT_RISK",
        "narrative": f"{market_summary} Autonomous trader draft withheld TRADE instead of converting incomplete evidence into live risk.",
        "chart_story": "Market read is recorded first; current chart/news/broker packet must then be refreshed or cleared before a trade receipt is valid.",
        "invalidation": "Re-run cycle-refresh; if blockers clear and LIVE_READY lanes remain, draft a fresh TRADE receipt.",
        "rejected_alternatives": [
            f"TRADE rejected by autonomous draft blocker: {item}" for item in blockers[:8]
        ] or ["TRADE rejected because no current LIVE_READY lane exists."],
        "risk_notes": [
            "No new risk is authorized by this draft.",
            "The live wrapper may still run existing-position maintenance through the gateway path.",
            active_summary or "No active path summary was available in the current packet.",
        ],
        "evidence_refs": refs,
        "twenty_minute_plan": _draft_twenty_minute_plan(
            action=action,
            pair=_draft_primary_pair(list(fallback_lane_ids), lanes_by_id),
            side="",
            selected_lane_ids=(),
            refs=refs,
            market_read_first=market_read,
        ),
        "strategy_reviews": [],
        "specialist_reviews": _draft_active_path_reviews(packet),
        "operator_summary": f"{market_summary} Autonomous trader draft did not select a trade because required evidence/gates are blocked.",
    }


def _cancel_pending_decision_draft(
    packet: dict[str, Any],
    *,
    cancel_order_ids: tuple[str, ...],
    blockers: list[str],
) -> dict[str, Any]:
    refs = _draft_cancel_pending_evidence_refs(packet)
    pending_by_id = {
        str(order.get("order_id") or ""): order
        for order in _pending_entry_orders(packet)
        if str(order.get("order_id") or "")
    }
    summaries = [
        _draft_pending_order_summary(order_id, pending_by_id.get(order_id) or {})
        for order_id in cancel_order_ids
    ]
    blocker_text = "; ".join(blockers[:5]) if blockers else "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED"
    market_read = _draft_market_read_first(packet, {})
    market_summary = _market_read_prediction_summary(market_read)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_read_first": market_read,
        "action": "CANCEL_PENDING",
        "selected_lane_id": None,
        "selected_lane_ids": [],
        "cancel_order_ids": list(cancel_order_ids),
        "close_trade_ids": [],
        "confidence": "HIGH",
        "thesis": (
            f"{market_summary} "
            "Reconcile trader-owned pending entries before new-risk routing: "
            f"{'; '.join(summaries)}."
        ),
        "method": _draft_cancel_pending_method(pending_by_id, cancel_order_ids),
        "narrative": (
            "Autonomous trader draft selected CANCEL_PENDING because self-improvement "
            "requires pending-entry cancel review and no current LIVE_READY replacement basket exists."
        ),
        "chart_story": (
            "Current broker snapshot still has trader-owned pending entry order(s), while "
            "the current order_intents packet has no tradeable LIVE_READY lane."
        ),
        "invalidation": (
            "Do not cancel if a gateway-refresh broker snapshot no longer contains the order id "
            "as a current trader-owned pending entry."
        ),
        "rejected_alternatives": [
            f"TRADE rejected by autonomous draft blocker: {item}" for item in blockers[:8]
        ] or ["TRADE rejected because no current LIVE_READY lane exists."],
        "risk_notes": [
            "No fresh entry is authorized by this cancel-only receipt.",
            "Cancel only verifier-approved current trader-owned pending entry order ids: "
            + ", ".join(cancel_order_ids),
            "LiveOrderGateway must re-fetch broker truth before canceling.",
        ],
        "evidence_refs": refs,
        "twenty_minute_plan": _draft_twenty_minute_plan(
            action="CANCEL_PENDING",
            pair="",
            side="",
            selected_lane_ids=(),
            refs=refs,
            market_read_first=market_read,
        ),
        "strategy_reviews": [],
        "specialist_reviews": [],
        "operator_summary": (
            f"{market_summary} Autonomous trader draft will clear stale pending exposure through the verified "
            "CANCEL_PENDING gateway path before any new trade decision."
        ),
    }


def _draft_cancel_pending_evidence_refs(packet: dict[str, Any]) -> list[str]:
    refs = [
        "broker:snapshot",
        "target:daily",
        "self_improvement:audit",
        "self_improvement:execution_quality",
        "self_improvement:finding:PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
        "timing:audit",
        "timing:canceled_orders",
    ]
    user_alpha = _active_user_alpha_continuation(packet)
    if user_alpha is not None:
        refs.extend(_user_alpha_evidence_refs(user_alpha))
    return _known_ordered_refs(packet, refs)


def _draft_close_evidence_refs(
    packet: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    trade_id = str(candidate.get("trade_id") or "")
    pair = str(candidate.get("pair") or "")
    side = str(candidate.get("side") or "").upper()
    refs: list[str] = ["broker:snapshot", "target:daily"]
    for item in candidate.get("recommendations", []) or []:
        if isinstance(item, dict):
            refs.append(str(item.get("evidence_ref") or ""))
    refs.extend(_draft_pair_refs(pair, side))
    refs.extend(
        [
            "capture:economics",
            "profitability:acceptance",
            "self_improvement:audit",
            "self_improvement:profitability",
            "self_improvement:finding:PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED",
            "timing:audit",
            "timing:loss_closes",
            "timing:market_closes",
            f"timing:loss_close:{trade_id}",
            f"timing:market_close:{trade_id}",
        ]
    )
    return _known_ordered_refs(packet, refs)


def _user_alpha_continuation_blocker_label(
    continuation: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
) -> str:
    latest = continuation.get("latest_trade") if isinstance(continuation.get("latest_trade"), dict) else {}
    pair = str(latest.get("pair") or "unknown_pair")
    direction = str(latest.get("direction") or "UNKNOWN").upper()
    selected = ", ".join(selected_lane_ids) if selected_lane_ids else "none"
    return (
        f"USER_ALPHA_CONTINUATION_BLOCKER {pair} {direction}: no clean matching "
        f"RELOAD/SECOND_SHOT continuation selected in current basket ({selected}); "
        "next trigger is a current LIVE_READY same-pair/same-direction lane after named blockers clear"
    )


def _draft_pending_order_summary(order_id: str, order: dict[str, Any]) -> str:
    pair = str(order.get("pair") or "unknown_pair")
    side = str(order.get("side") or _side_from_order_units(order.get("units")) or "UNKNOWN").upper()
    order_type = str(order.get("order_type") or "ORDER")
    price = order.get("price")
    units = order.get("units")
    return f"{order_id} {pair} {side} {order_type} units={units} price={price}"


def _draft_cancel_pending_method(
    pending_by_id: dict[str, dict[str, Any]],
    cancel_order_ids: tuple[str, ...],
) -> str:
    methods = {
        _method_from_pending_order(pending_by_id.get(order_id) or {})
        for order_id in cancel_order_ids
    }
    methods.discard("")
    if len(methods) == 1:
        method = next(iter(methods))
        if method in ALLOWED_METHODS:
            return method
    return "POSITION_MANAGEMENT"


def _method_from_pending_order(order: dict[str, Any]) -> str:
    lane = str(order.get("lane_id") or "")
    if not lane:
        raw = order.get("raw") if isinstance(order.get("raw"), dict) else {}
        for key in ("clientExtensions", "tradeClientExtensions"):
            extension = raw.get(key)
            if not isinstance(extension, dict):
                continue
            comment = str(extension.get("comment") or "")
            marker = "lane="
            if marker not in comment:
                continue
            lane = comment.split(marker, 1)[1].split()[0]
            break
    parts = lane.split(":")
    return parts[3] if len(parts) >= 4 else ""


def _draft_trade_evidence_refs(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    lanes_by_id: dict[str, dict[str, Any]],
    *,
    cancel_order_ids: tuple[str, ...] = (),
) -> list[str]:
    refs: list[str] = ["broker:snapshot", "target:daily", "news:health", "news:items"]
    attack_ids = set(_attack_recommended_tradeable_lane_ids(packet, list(lanes_by_id)))
    for lane_id in selected_lane_ids:
        lane = lanes_by_id[lane_id]
        pair = str(lane.get("pair") or "")
        side = str(lane.get("direction") or "")
        refs.extend([f"intent:{lane_id}", f"campaign:{lane_id}", f"strategy:{pair}:{side}", f"story:{pair}"])
        refs.extend(_draft_pair_refs(pair, side))
        if lane_id in attack_ids:
            refs.extend(["attack:advice", f"attack:lane:{lane_id}"])
    for requirement in packet.get("decision_requirements", {}).get("learning_influenced_lane_evidence", []) or []:
        if not isinstance(requirement, dict):
            continue
        if str(requirement.get("lane_id") or "") in selected_lane_ids:
            refs.extend(str(ref) for ref in requirement.get("required_evidence_refs", []) or [])
    if cancel_order_ids:
        refs.extend([
            "self_improvement:audit",
            "self_improvement:execution_quality",
            "self_improvement:finding:PENDING_ENTRY_CANCEL_REVIEW_REQUIRED",
        ])
    user_alpha = _active_user_alpha_continuation(packet)
    if user_alpha is not None and _selected_lanes_include_user_alpha(packet, selected_lane_ids, user_alpha):
        refs.extend(_user_alpha_evidence_refs(user_alpha))
    return _known_ordered_refs(packet, refs)


def _draft_non_trade_evidence_refs(
    packet: dict[str, Any],
    live_ready_lane_ids: list[str],
    lanes_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    refs: list[str] = ["broker:snapshot", "target:daily", "news:health", "news:items"]
    refs.extend(_active_path_evidence_refs(packet.get("active_path") if isinstance(packet.get("active_path"), dict) else {}))
    for lane_id in live_ready_lane_ids[:4]:
        lane = lanes_by_id.get(lane_id)
        if not lane:
            continue
        pair = str(lane.get("pair") or "")
        side = str(lane.get("direction") or "")
        refs.extend([f"intent:{lane_id}", f"campaign:{lane_id}"])
        refs.extend(_draft_pair_refs(pair, side))
    refs.extend(["projection:ledger", "self_improvement:audit", "profitability:acceptance"])
    user_alpha = _active_user_alpha_continuation(packet)
    if user_alpha is not None:
        refs.extend(_user_alpha_evidence_refs(user_alpha))
    return _known_ordered_refs(packet, refs)


def _draft_active_path_reviews(packet: dict[str, Any]) -> list[dict[str, Any]]:
    active_path = packet.get("active_path")
    if not isinstance(active_path, dict):
        return []
    lanes = _active_path_ordered_fallback_lanes(active_path)
    if not lanes:
        return []
    packet_lane_ids = {
        str(item.get("lane_id") or "")
        for item in packet.get("lanes", []) or []
        if isinstance(item, dict)
    }
    summary = _active_path_blocker_summary(packet)
    reviews: list[dict[str, Any]] = []
    for lane in lanes:
        lane_id = str(lane.get("lane_id") or "").strip()
        if not lane_id:
            continue
        hard_codes = [str(code) for code in lane.get("blockers", []) or [] if str(code or "").strip()]
        reviews.append({
            "role": "portfolio_context",
            "lane_id": lane_id if lane_id in packet_lane_ids else None,
            "method": str(lane.get("strategy_family") or ""),
            "verdict": "BLOCKED",
            "summary": summary or "Active path is read-only and not live permission.",
            "cited_evidence_refs": _active_path_evidence_refs(active_path),
            "hard_gate_codes": hard_codes[:12],
            "read_only": True,
            "live_permission": False,
        })
    return reviews


def _draft_pair_refs(pair: str, side: str) -> list[str]:
    if not pair:
        return []
    currencies = [part for part in pair.split("_") if part]
    refs = [
        f"chart:{pair}:M1",
        f"chart:{pair}:M5",
        f"chart:{pair}:M15",
        f"chart:{pair}:M30",
        f"chart:{pair}:H1",
        f"chart:{pair}:H4",
        f"chart:{pair}:D",
        f"chart:{pair}:structure",
        f"matrix:{pair}:{side}",
        f"flow:{pair}",
        f"levels:{pair}",
        f"calendar:{pair}",
        f"strength:{pair}",
        f"cross:correlations:{pair}",
        f"news:{pair}",
    ]
    for currency in currencies:
        refs.extend([f"strength:{currency}", f"calendar:{currency}", f"cot:{currency}", f"news:{currency}"])
    refs.extend(["cross:dxy", "cross:USB10Y_USD"])
    return refs


def _known_ordered_refs(packet: dict[str, Any], refs: list[str]) -> list[str]:
    allowed = set(str(ref) for ref in packet.get("allowed_evidence_refs", []) or [])
    ordered: list[str] = []
    for ref in refs:
        text = str(ref or "").strip()
        if text and text in allowed and text not in ordered:
            ordered.append(text)
    return ordered


def _draft_trade_thesis(selected_lane_ids: tuple[str, ...], lanes_by_id: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for lane_id in selected_lane_ids:
        lane = lanes_by_id[lane_id]
        risk = lane.get("risk_metrics") or {}
        parts.append(
            f"{lane_id} {lane.get('order_type')} units={lane.get('units')} "
            f"entry={lane.get('entry')} tp={lane.get('tp')} sl={lane.get('sl')} "
            f"risk_jpy={risk.get('risk_jpy')} reward_jpy={risk.get('reward_jpy')} rr={risk.get('reward_risk')}"
        )
    return "Current LIVE_READY basket selected from broker-truth intents: " + "; ".join(parts)


def _draft_risk_notes(
    selected_lane_ids: tuple[str, ...],
    lanes_by_id: dict[str, dict[str, Any]],
    *,
    cancel_order_ids: tuple[str, ...] = (),
) -> list[str]:
    notes = [
        "Use only units, entry, TP, SL, and risk metrics already present in current LIVE_READY intents.",
        "LiveOrderGateway must re-fetch broker truth and revalidate portfolio risk, margin, duplicate geometry, and guardian state before send.",
    ]
    if cancel_order_ids:
        notes.append(
            "Cancel only verifier-approved trader-owned pending entry order ids before validating the selected replacement basket: "
            + ", ".join(cancel_order_ids)
        )
    for lane_id in selected_lane_ids:
        risk = lanes_by_id[lane_id].get("risk_metrics") or {}
        notes.append(
            f"{lane_id}: risk_jpy={risk.get('risk_jpy')} reward_jpy={risk.get('reward_jpy')} "
            f"rr={risk.get('reward_risk')} estimated_margin_jpy={risk.get('estimated_margin_jpy')}"
        )
    return notes


def _draft_rejected_alternatives(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    lanes_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    selected = set(selected_lane_ids)
    rejected: list[str] = []
    for lane_id in _tradeable_live_ready_lanes(packet):
        if lane_id in selected:
            continue
        lane = lanes_by_id.get(lane_id) or {}
        pair = str(lane.get("pair") or "")
        reason = (
            "skipped because another selected lane already covers this pair"
            if pair and any(pair == str((lanes_by_id.get(chosen) or {}).get("pair") or "") for chosen in selected)
            else "skipped by margin-aware basket or lower attack-advice rank"
        )
        rejected.append(f"{lane_id}: {reason}")
    return rejected or ["WAIT rejected because current packet contains selected LIVE_READY lane(s) and no named blocker won."]


def _draft_specialist_reviews(primary_lane: dict[str, Any], refs: list[str]) -> list[dict[str, Any]]:
    pair = str(primary_lane.get("pair") or "")
    method = str(primary_lane.get("method") or "")
    cited = [ref for ref in refs if ref.startswith(f"chart:{pair}:")][:4]
    return [
        {
            "role": "portfolio_context",
            "lane_id": primary_lane.get("lane_id"),
            "method": method,
            "verdict": "SUPPORTS",
            "summary": "Read-only draft review; final permission remains with gpt-trader-decision and LiveOrderGateway.",
            "cited_evidence_refs": cited or refs[:2],
            "hard_gate_codes": [],
            "read_only": True,
            "live_permission": False,
        }
    ]


def _draft_twenty_minute_plan(
    *,
    action: str,
    pair: str,
    side: str,
    selected_lane_ids: tuple[str, ...],
    refs: list[str],
    market_read_first: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan_refs = (
        [ref for ref in refs if ref.startswith("chart:")][:2]
        + [f"intent:{lane_id}" for lane_id in selected_lane_ids]
        + [ref for ref in refs if ref.startswith("user_alpha:")]
    )
    if len(plan_refs) < 2:
        plan_refs = refs[:4]
    label = f"{pair} {side}".strip() or "the current packet"
    market_summary = _market_read_prediction_summary(market_read_first or {})
    return {
        "horizon_minutes": TRADER_DECISION_HORIZON_MINUTES,
        "primary_path": (
            f"{market_summary} {label} should respect the selected LIVE_READY trigger before the next scheduled cycle."
            if action == "TRADE"
            else f"{market_summary} Refresh the named blocker(s) and keep broker-truth maintenance active before new risk."
        ),
        "failure_path": (
            "The selected lane leaves LIVE_READY, news-health blocks, or broker truth changes exposure/margin before send."
            if action == "TRADE"
            else "A blocker remains after refresh, so the next cycle must continue repair/evidence work."
        ),
        "entry_or_hold_trigger": (
            "Enter only through the selected current intent(s) after gpt-trader-decision accepts this receipt."
            if action == "TRADE"
            else "Hold new entries until cycle-refresh produces clean LIVE_READY evidence and news-health is non-blocked."
        ),
        "invalidation_or_cancel_trigger": (
            "Cancel the idea if broker truth, selected lane status, news-health, or gateway risk validation changes."
            if action == "TRADE"
            else "Reconsider only after the named blocker code disappears from the refreshed packet."
        ),
        "counterargument": (
            "The strongest counterargument is stale or incomplete market/news context; this draft cites current chart/news refs and the verifier must reject if they are blocked."
            if action == "TRADE"
            else "Campaign pressure argues for taking LIVE_READY lanes, but named blockers outrank discretionary urgency."
        ),
        "next_cycle_check": "First re-check broker snapshot, order_intents, ai_attack_advice, news-health, and selected lane refs.",
        "evidence_refs": _unique_preserve_order(plan_refs),
    }


def _draft_primary_pair(live_ready_lane_ids: list[str], lanes_by_id: dict[str, dict[str, Any]]) -> str:
    if not live_ready_lane_ids:
        return ""
    return str((lanes_by_id.get(live_ready_lane_ids[0]) or {}).get("pair") or "")


def _unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _write_trader_decision_draft_report(
    report_path: Path,
    *,
    decision: dict[str, Any],
    blockers: list[str],
    verification: VerificationResult,
    input_packet: dict[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trader Decision Draft Report",
        "",
        f"- Action: `{decision.get('action')}`",
        f"- Selected lane: `{decision.get('selected_lane_id')}`",
        f"- Selected basket lanes: `{', '.join(decision.get('selected_lane_ids') or []) or 'none'}`",
        f"- Cancel order ids: `{', '.join(decision.get('cancel_order_ids') or []) or 'none'}`",
        f"- Draft blockers: `{'; '.join(blockers) if blockers else 'none'}`",
        f"- Market read first: `{decision.get('market_read_first') or 'missing'}`",
        f"- Verifier precheck: `{'ACCEPTED' if verification.allowed else 'REJECTED'}`",
    ]
    consumption = (
        input_packet.get("guardian_receipt_consumption")
        if isinstance(input_packet.get("guardian_receipt_consumption"), dict)
        else {}
    )
    operator_review = (
        input_packet.get("guardian_receipt_operator_review")
        if isinstance(input_packet.get("guardian_receipt_operator_review"), dict)
        else {}
    )
    lines.extend(
        [
            f"- Guardian receipt consumption: `{consumption.get('status')}` "
            f"normal_routing_allowed=`{consumption.get('normal_routing_allowed')}`",
            f"- Guardian receipt operator review: `{operator_review.get('status')}` "
            f"normal_routing_allowed=`{operator_review.get('normal_routing_allowed')}`",
        ]
    )
    lines.extend(_user_alpha_report_lines(input_packet.get("user_alpha_continuation"), decision=decision))
    lines.extend(["", "## Verification Issues", ""])
    if verification.issues:
        lines.extend(f"- `{issue.severity}` {issue.code}: {issue.message}" for issue in verification.issues)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- Draft generation is read-only and writes only the decision response JSON/report.",
            "- It does not call model APIs, send orders, cancel orders, close positions, or change launchd state.",
            "- `gpt-trader-decision` and `LiveOrderGateway` remain the execution gates.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def _selected_trade_lane_ids(decision: GPTTraderDecision) -> tuple[str, ...]:
    lane_ids = tuple(dict.fromkeys(lane_id for lane_id in decision.selected_lane_ids if lane_id))
    if lane_ids:
        return lane_ids
    return (decision.selected_lane_id,) if decision.selected_lane_id else ()


def _predictive_scout_lane_packet(
    intent: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    if not predictive_scout_geometry_claimed(
        metadata,
        pair=str(intent.get("pair") or ""),
        side=str(intent.get("side") or ""),
        order_type=str(intent.get("order_type") or ""),
        method=str(market_context.get("method") or "") or None,
    ):
        return {}
    rule = (
        metadata.get("bidask_replay_precision_seed_rule")
        if isinstance(metadata.get("bidask_replay_precision_seed_rule"), dict)
        else {}
    )
    return {
        "claimed": True,
        "enabled": metadata.get("predictive_scout") is True,
        "canonical_rule_supported": predictive_scout_metadata_supported(metadata),
        "source": metadata.get("predictive_scout_source"),
        "hypothesis": metadata.get("predictive_scout_hypothesis"),
        "vehicle_proof_status": metadata.get("predictive_scout_vehicle_proof_status"),
        "rule_is_vehicle_proof": metadata.get("predictive_scout_rule_is_vehicle_proof"),
        "rule_name": metadata.get("predictive_scout_rule_name"),
        "rule_digest": metadata.get("predictive_scout_rule_digest"),
        "forecast_cycle_id": metadata.get("forecast_cycle_id"),
        "generated_at_utc": metadata.get("predictive_scout_generated_at_utc"),
        "expires_at_utc": metadata.get("predictive_scout_expires_at_utc"),
        "ttl_minutes": metadata.get("predictive_scout_ttl_minutes"),
        "promotion_allowed": metadata.get("predictive_scout_promotion_allowed"),
        "broker_stop_loss_mode": metadata.get("broker_stop_loss_mode"),
        "risk_tier": metadata.get("predictive_scout_risk_tier"),
        "nav_jpy_at_sizing": metadata.get("predictive_scout_nav_jpy_at_sizing"),
        "max_risk_pct_nav": metadata.get("predictive_scout_max_risk_pct_nav"),
        "max_loss_jpy": metadata.get("predictive_scout_max_loss_jpy"),
        "effective_max_loss_jpy": metadata.get(
            "predictive_scout_effective_max_loss_jpy"
        ),
        "planned_initial_risk_jpy": metadata.get(
            "predictive_scout_planned_initial_risk_jpy"
        ),
        "planned_initial_risk_pct_nav": metadata.get(
            "predictive_scout_planned_initial_risk_pct_nav"
        ),
        "fresh_actual_initial_risk_jpy": metadata.get(
            "predictive_scout_fresh_actual_initial_risk_jpy"
        ),
        "active_initial_risk_jpy": metadata.get(
            "predictive_scout_active_initial_risk_jpy"
        ),
        "aggregate_initial_risk_jpy": metadata.get(
            "predictive_scout_aggregate_initial_risk_jpy"
        ),
        "concurrent_risk_cap_jpy": metadata.get(
            "predictive_scout_concurrent_risk_cap_jpy"
        ),
        "sizing_digest": metadata.get("predictive_scout_sizing_digest"),
        "order_type": intent.get("order_type"),
        "units": intent.get("units"),
        "rule": _small_dict(
            rule,
            (
                "name",
                "pair",
                "side",
                "direction",
                "forecast_direction",
                "contrarian_edge",
                "horizon_bucket",
                "confidence_bucket",
                "samples",
                "active_days",
                "optimized_avg_realized_pips",
                "optimized_profit_factor",
                "positive_day_rate",
                "optimized_take_profit_pips",
                "optimized_stop_loss_pips",
                "daily_stability_status",
                "adoption_status",
            ),
        ),
    }


def _lane_packet(
    intents: dict[str, Any],
    campaign: dict[str, Any],
    strategy: dict[str, Any],
    story: dict[str, Any],
    *,
    max_lanes: int,
) -> list[dict[str, Any]]:
    campaign_index = {f"{lane.get('desk')}:{lane.get('pair')}:{lane.get('direction')}:{lane.get('method')}": lane for lane in campaign.get("lanes", []) or []}
    strategy_index = {(item.get("pair"), item.get("direction")): item for item in strategy.get("profiles", []) or []}
    story_index = {item.get("pair"): item for item in story.get("pair_profiles", []) or []}
    lanes: list[dict[str, Any]] = []
    for result in intents.get("results", []) or []:
        if not isinstance(result, dict) or not isinstance(result.get("intent"), dict):
            continue
        intent = result["intent"]
        lane_id = str(result.get("lane_id") or "")
        pair = str(intent.get("pair") or "")
        direction = str(intent.get("side") or "")
        context = intent.get("market_context") or {}
        metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
        risk_blockers = _block_issues(result.get("risk_issues"))
        strategy_blockers = _block_issues(result.get("strategy_issues"))
        lanes.append(
            {
                "lane_id": lane_id,
                "evidence_ref": f"intent:{lane_id}",
                "status": result.get("status"),
                "pair": pair,
                "direction": direction,
                "method": context.get("method") or "",
                "order_type": intent.get("order_type"),
                "entry": intent.get("entry"),
                "tp": intent.get("tp"),
                "sl": intent.get("sl"),
                "units": intent.get("units"),
                "risk_metrics": _small_dict(
                    result.get("risk_metrics"),
                    (
                        "entry_price",
                        "loss_pips",
                        "reward_pips",
                        "risk_jpy",
                        "reward_jpy",
                        "reward_risk",
                        "spread_pips",
                        "jpy_per_pip",
                        "estimated_margin_jpy",
                        "margin_available_jpy",
                        "margin_budget_jpy",
                        "margin_used_jpy",
                        "margin_utilization_after_pct",
                        "max_margin_utilization_pct",
                    ),
                ),
                "thesis": intent.get("thesis"),
                "narrative": context.get("narrative") or "",
                "chart_story": context.get("chart_story") or "",
                "invalidation": context.get("invalidation") or "",
                "risk_blockers": risk_blockers,
                "strategy_blockers": strategy_blockers,
                "live_blockers": list(result.get("live_blockers", []) or []),
                "campaign": _small_dict(
                    campaign_index.get(lane_id) or campaign_index.get(_parent_lane_id(lane_id)),
                    ("adoption", "campaign_role", "required_receipt"),
                ),
                "strategy": _small_dict(
                    strategy_index.get((pair, direction)),
                    (
                        "status",
                        "pretrade_net_jpy",
                        "live_net_jpy",
                        "live_worst_jpy",
                        "seat_pl_n",
                        "seat_net_jpy",
                        "seat_win_rate_pct",
                        "required_fix",
                    ),
                ),
                "story": _small_dict(story_index.get(pair), ("methods", "themes", "examples")),
                "forecast": _lane_forecast_packet(intent.get("metadata")),
                "opportunity": _small_dict(
                    metadata,
                    (
                        "opportunity_mode",
                        "opportunity_mode_reason",
                        "opportunity_mode_reward_risk",
                        "tp_execution_mode",
                        "tp_target_intent",
                        "tp_target_source",
                    ),
                ),
                "target_path": _small_dict(
                    metadata,
                    (
                        "daily_target_mode",
                        "target_mode",
                        "remaining_to_5pct_yen",
                        "remaining_to_5pct",
                        "remaining_minimum_jpy",
                        "conviction_grade",
                        "grade",
                        "allocation_band",
                        "target_path_role",
                        "path_role",
                        "attack_stack_slot",
                        "valid_as_target_path",
                    ),
                ),
                "self_improvement": _small_dict(
                    metadata,
                    (
                        "self_improvement_p0_repair_live_ready",
                        "self_improvement_p0_repair_mode",
                        "self_improvement_p0_repair_blocker_code",
                        "self_improvement_p0_repair_reason",
                        "self_improvement_forecast_adverse_path_repair_live_ready",
                        "self_improvement_forecast_adverse_path_repair_mode",
                        "self_improvement_forecast_adverse_path_repair_blocker_code",
                        "self_improvement_forecast_adverse_path_repair_reason",
                        "position_intent",
                        "forecast_direction",
                        "attach_take_profit_on_fill",
                        "positive_rotation_live_ready",
                        "positive_rotation_mode",
                        "positive_rotation_pessimistic_expectancy_jpy",
                        "positive_rotation_minimum_floor_reachable",
                        "positive_rotation_minimum_floor_reach_basis",
                        "positive_rotation_oanda_campaign_current_risk_minimum_floor_reachable",
                        "positive_rotation_oanda_campaign_current_risk_estimated_return_pct_per_active_day",
                        "positive_rotation_oanda_campaign_remaining_minimum_pct",
                        "capture_take_profit_scope",
                        "capture_take_profit_scope_key",
                        "capture_take_profit_trades",
                        "capture_take_profit_losses",
                        "capture_take_profit_expectancy_jpy",
                        "tp_execution_mode",
                        "tp_target_intent",
                        "opportunity_mode",
                        "month_scale_residual_loss_repair_blocked",
                        "month_scale_residual_loss_group",
                    ),
                ),
                "predictive_scout": _predictive_scout_lane_packet(intent, metadata),
                "market_close_leak_family": _small_dict(
                    metadata,
                    CLOSE_GATE_PROOF_KEYS
                    + CONTAINED_RISK_TIMING_EVIDENCE_KEYS
                    + TP_PROVEN_EXCEPTION_KEYS
                    + ("trade_id", "broker_trade_id", "classification", "operator_classification"),
                ),
                "position_building": _small_dict(
                    metadata,
                    (
                        "position_intent",
                        "position_fill",
                        "same_pair_add_type",
                        "same_pair_existing_entries",
                        "same_pair_existing_units",
                        "same_pair_existing_avg_entry",
                        "same_pair_add_entry",
                        "same_pair_add_distance_from_avg_pips",
                        "same_pair_adverse_add_pips",
                        "same_pair_with_move_add_pips",
                        "hedge_suppressed_reason",
                    ),
                ),
                "market_context_matrix": _small_dict(
                    metadata,
                    (
                        "market_context_matrix_ref",
                        "matrix_support_count",
                        "matrix_reject_count",
                        "matrix_warning_count",
                        "matrix_missing_count",
                        "strongest_matrix_support",
                        "strongest_matrix_reject",
                    ),
                ),
                "technical_context": _small_dict(
                    metadata,
                    (
                        "session_bucket",
                        "session_current_tag",
                        "entry_price_percentile_24h",
                        "entry_price_percentile_7d",
                        "price_percentile_24h",
                        "price_percentile_7d",
                        "range_24h_sigma_multiple",
                        "tf_agreement_score",
                        "chart_direction_bias",
                        "h1_regime",
                        "h1_adx",
                        "h4_regime",
                        "m5_regime",
                        "m5_regime_quantile",
                        "current_price_mid",
                    ),
                ),
            }
        )
    if max_lanes <= 0 or len(lanes) <= max_lanes:
        return lanes
    capped = lanes[:max_lanes]
    capped_ids = {str(lane.get("lane_id") or "") for lane in capped}
    for lane in lanes[max_lanes:]:
        lane_id = str(lane.get("lane_id") or "")
        if lane_id and lane_id not in capped_ids and lane.get("status") == "LIVE_READY":
            capped.append(lane)
            capped_ids.add(lane_id)
    return capped


def _snapshot_packet(
    snapshot: dict[str, Any],
    *,
    relevant_pairs: set[str] | None = None,
) -> dict[str, Any]:
    orders = snapshot.get("orders", []) or []
    quotes_in = snapshot.get("quotes") or {}
    account = snapshot.get("account") if isinstance(snapshot.get("account"), dict) else {}
    # Scope quotes to pairs we actually have positions on (or are likely
    # to reference in this cycle) so the verifier packet stays compact
    # but the CLOSE-discipline gate has bid/ask available to check
    # `invalidation_price` hits against current broker truth.
    scoped_pairs: set[str] = set(relevant_pairs or set())
    for item in snapshot.get("positions", []) or []:
        if item.get("pair"):
            scoped_pairs.add(str(item["pair"]))
    for order in orders:
        if order.get("pair"):
            scoped_pairs.add(str(order["pair"]))
    scoped_quotes = {
        pair: {
            "bid": q.get("bid"),
            "ask": q.get("ask"),
            "timestamp_utc": q.get("timestamp_utc"),
        }
        for pair, q in quotes_in.items()
        if pair in scoped_pairs and isinstance(q, dict)
    }
    return {
        "evidence_ref": "broker:snapshot",
        "fetched_at_utc": snapshot.get("fetched_at_utc"),
        "account": _small_dict(
            account,
            (
                "balance_jpy",
                "nav_jpy",
                "margin_available_jpy",
                "margin_used_jpy",
                "unrealized_pl_jpy",
                "last_transaction_id",
            ),
        ),
        "positions": len(snapshot.get("positions", []) or []),
        "orders": len(orders),
        "position_summaries": [
            {
                "trade_id": item.get("trade_id"),
                "pair": item.get("pair"),
                "side": item.get("side"),
                "units": item.get("units"),
                "unrealized_pl_jpy": item.get("unrealized_pl_jpy"),
                "take_profit": item.get("take_profit"),
                "stop_loss": item.get("stop_loss"),
                "owner": item.get("owner"),
                "open_time_utc": (
                    (item.get("raw") or {}).get("openTime")
                    if isinstance(item.get("raw"), dict)
                    else None
                ),
            }
            for item in (snapshot.get("positions", []) or [])
        ],
        "pending_orders": _pending_order_packet(orders),
        "quotes": scoped_quotes,
    }


def _pending_order_packet(orders: list[Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(item: Any) -> None:
        if not isinstance(item, dict):
            return
        order_id = str(item.get("order_id") or "")
        if order_id in seen:
            return
        if order_id:
            seen.add(order_id)
        selected.append(
            {
                "order_id": item.get("order_id"),
                "pair": item.get("pair"),
                "order_type": item.get("order_type"),
                "trade_id": item.get("trade_id"),
                "price": item.get("price"),
                "units": item.get("units"),
                "side": item.get("side") or _side_from_order_units(item.get("units")),
                "lane_id": item.get("lane_id") or item.get("client_lane_id"),
                "client_order_id": item.get("client_order_id") or item.get("clientOrderID"),
                "owner": item.get("owner"),
            }
        )

    for item in orders[:5]:
        add(item)
    for item in orders:
        if _is_pending_entry_order_payload(item):
            add(item)
    return selected


def _is_pending_entry_order_payload(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("trade_id"):
        return False
    order_type = str(item.get("order_type") or "").upper()
    return order_type in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}


def _side_from_order_units(units: Any) -> str:
    try:
        value = float(units)
    except (TypeError, ValueError):
        return ""
    if value > 0:
        return "LONG"
    if value < 0:
        return "SHORT"
    return ""


def _parent_lane_id(lane_id: str) -> str:
    if lane_id.endswith(":MARKET"):
        return lane_id[: -len(":MARKET")]
    return lane_id


def _target_packet(target: dict[str, Any]) -> dict[str, Any]:
    if not target:
        return {"evidence_ref": "target:daily", "status": "missing"}
    return {
        "evidence_ref": "target:daily",
        "generated_at_utc": target.get("generated_at_utc"),
        "status": target.get("status"),
        "target_return_pct": target.get("target_return_pct"),
        "target_jpy": target.get("target_jpy"),
        "target_profit_jpy": target.get("target_profit_jpy"),
        "minimum_return_pct": target.get("minimum_return_pct"),
        "minimum_target_jpy": target.get("minimum_target_jpy"),
        "progress_jpy": target.get("progress_jpy"),
        "progress_pct": target.get("progress_pct"),
        "minimum_progress_pct": target.get("minimum_progress_pct"),
        "account_progress_jpy": target.get("account_progress_jpy"),
        "account_progress_pct": target.get("account_progress_pct"),
        "account_unrealized_pl_jpy": target.get("account_unrealized_pl_jpy"),
        "current_equity_jpy": target.get("current_equity_jpy"),
        "rolling_30d_policy": target.get("rolling_30d_policy"),
        "rolling_30d_start_equity": target.get("rolling_30d_start_equity"),
        "current_equity_raw": target.get("current_equity_raw"),
        "capital_flows_30d": target.get("capital_flows_30d"),
        "funding_adjusted_equity": target.get("funding_adjusted_equity"),
        "current_equity": target.get("current_equity"),
        "rolling_30d_multiplier_raw": target.get("rolling_30d_multiplier_raw"),
        "rolling_30d_multiplier_funding_adjusted": target.get("rolling_30d_multiplier_funding_adjusted"),
        "current_30d_multiplier": target.get("current_30d_multiplier"),
        "remaining_to_4x_raw": target.get("remaining_to_4x_raw"),
        "remaining_to_4x_funding_adjusted": target.get("remaining_to_4x_funding_adjusted"),
        "remaining_to_4x": target.get("remaining_to_4x"),
        "required_calendar_daily_return_raw": target.get("required_calendar_daily_return_raw"),
        "required_active_day_return_raw": target.get("required_active_day_return_raw"),
        "required_calendar_daily_return_funding_adjusted": target.get(
            "required_calendar_daily_return_funding_adjusted"
        ),
        "required_active_day_return_funding_adjusted": target.get(
            "required_active_day_return_funding_adjusted"
        ),
        "required_calendar_daily_return": target.get("required_calendar_daily_return"),
        "required_active_day_return": target.get("required_active_day_return"),
        "performance_basis": target.get("performance_basis"),
        "sizing_basis": target.get("sizing_basis"),
        "pace_state": target.get("pace_state"),
        "remaining_minimum_jpy": target.get("remaining_minimum_jpy"),
        "remaining_target_jpy": target.get("remaining_target_jpy"),
        "remaining_risk_budget_jpy": target.get("remaining_risk_budget_jpy"),
        "realized_loss_spent_jpy": target.get("realized_loss_spent_jpy"),
        "daily_loss_capacity_before_open_jpy": target.get("daily_loss_capacity_before_open_jpy"),
        "open_risk_jpy": target.get("open_risk_jpy"),
        "sizing_nav_jpy": target.get("sizing_nav_jpy"),
        "base_per_trade_risk_budget_jpy": target.get("base_per_trade_risk_budget_jpy"),
        "per_trade_risk_budget_jpy": target.get("per_trade_risk_budget_jpy"),
        "per_trade_risk_pct_nav": target.get("per_trade_risk_pct_nav"),
        "target_trades_per_day": target.get("target_trades_per_day"),
        "uncapped_required_trades_per_day": target.get("uncapped_required_trades_per_day"),
        "uncapped_required_trades_per_day_basis_return_pct": target.get(
            "uncapped_required_trades_per_day_basis_return_pct"
        ),
        "selected_basis_uncapped_required_trades_per_day": target.get(
            "selected_basis_uncapped_required_trades_per_day"
        ),
        "selected_basis_return_pct": target.get("selected_basis_return_pct"),
        "operating_pace_trades_per_day": target.get("operating_pace_trades_per_day"),
        "automated_operating_cap_trades_per_day": target.get(
            "automated_operating_cap_trades_per_day"
        ),
        "observed_trades_per_day": target.get("observed_trades_per_day"),
        "observed_expectancy_jpy_per_trade": target.get(
            "observed_expectancy_jpy_per_trade"
        ),
        "frequency_multiple_required": target.get("frequency_multiple_required"),
        "planned_reward_at_operating_pace_jpy": target.get(
            "planned_reward_at_operating_pace_jpy"
        ),
        "stretch_required_minus_operating_gap_trades_per_day": target.get(
            "stretch_required_minus_operating_gap_trades_per_day"
        ),
        "selected_required_minus_operating_gap_trades_per_day": target.get(
            "selected_required_minus_operating_gap_trades_per_day"
        ),
        "trade_pace_feasible_within_operating_pace": target.get(
            "trade_pace_feasible_within_operating_pace"
        ),
        "trade_pace_feasibility": target.get("trade_pace_feasibility"),
    }


def _protection_sidecars_packet(
    *,
    snapshot: dict[str, Any],
    snapshot_path: Path,
    pair_charts_path: Path,
) -> dict[str, Any]:
    from quant_rabbit.trader_prompts import (
        _fresh_close_recommendations,
        _fresh_entry_thesis_blockers,
        _fresh_position_hold_support,
        _tp_rebalance_reasons,
    )

    position_close_recommendations = list(
        _fresh_close_recommendations(snapshot, data_root=snapshot_path.parent)
    )
    position_hold_support = list(
        _fresh_position_hold_support(snapshot, data_root=snapshot_path.parent)
    )
    entry_thesis_blockers = list(
        _fresh_entry_thesis_blockers(snapshot, data_root=snapshot_path.parent)
    )
    entry_thesis_close_context = _entry_thesis_close_context(
        snapshot,
        data_root=snapshot_path.parent,
    )
    if not pair_charts_path.exists():
        sidecars = {
            "tp_rebalance": {
                "required": False,
                "reasons": [],
                "issue": f"missing pair_charts: {pair_charts_path}",
            },
            "position_close_recommendations": position_close_recommendations,
            "position_hold_support": position_hold_support,
            "entry_thesis_blockers": entry_thesis_blockers,
            "entry_thesis_close_context": entry_thesis_close_context,
        }
        sidecars["position_close_recommendations"] = _annotated_position_close_recommendations(
            snapshot,
            sidecars,
        )
        return sidecars
    pair_charts = _load_json(pair_charts_path)

    reasons = _tp_rebalance_reasons(
        snapshot,
        pair_charts,
        snapshot_path=snapshot_path,
    )
    sidecars = {
        "tp_rebalance": {
            "required": bool(reasons),
            "reasons": list(reasons),
        },
        "position_close_recommendations": position_close_recommendations,
        "position_hold_support": position_hold_support,
        "entry_thesis_blockers": entry_thesis_blockers,
        "entry_thesis_close_context": entry_thesis_close_context,
    }
    sidecars["position_close_recommendations"] = _annotated_position_close_recommendations(
        snapshot,
        sidecars,
    )
    return sidecars


def _annotated_position_close_recommendations(
    snapshot: dict[str, Any],
    sidecars: dict[str, Any],
) -> list[dict[str, Any]]:
    packet = {
        "broker_snapshot": _snapshot_packet(snapshot),
        "protection_sidecars": sidecars,
    }
    annotated: list[dict[str, Any]] = []
    for rec in sidecars.get("position_close_recommendations") or []:
        if not isinstance(rec, dict):
            continue
        item = dict(rec)
        blocks_non_close = _position_close_recommendation_blocks_non_close_action(packet, item)
        item["blocks_non_close_actions"] = blocks_non_close
        item["routing_effect"] = (
            "BLOCKS_NON_CLOSE_ACTIONS"
            if blocks_non_close
            else "SOFT_ADVISORY_NON_BLOCKING"
        )
        if not blocks_non_close:
            item["entry_decision_guidance"] = (
                "Do not choose CLOSE from this advisory on the entry branch unless "
                "explicit operator Gate B is present; evaluate current LIVE_READY lanes."
            )
            hold_conflict = _sidecar_hold_support_conflict(packet, item)
            if hold_conflict:
                item["non_blocking_reason"] = hold_conflict
            elif not _operator_close_gate_authorized():
                item["non_blocking_reason"] = (
                    "soft close evidence lacks explicit operator Gate B and therefore "
                    "does not preempt entry routing"
                )
        annotated.append(item)
    return annotated


def _entry_thesis_close_context(snapshot: dict[str, Any], *, data_root: Path) -> list[dict[str, Any]]:
    from quant_rabbit.strategy.entry_thesis_ledger import load_entry_thesis

    rows: list[dict[str, Any]] = []
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict) or str(position.get("owner") or "") != "trader":
            continue
        trade_id = str(position.get("trade_id") or "")
        pair = str(position.get("pair") or "")
        side = str(position.get("side") or "").upper()
        if not trade_id:
            continue
        thesis = load_entry_thesis(trade_id, data_root)
        invalidation = getattr(thesis, "invalidation_price", None) if thesis is not None else None
        rows.append(
            {
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "recorded": thesis is not None,
                "has_recorded_invalidation_price": invalidation is not None,
                "recorded_invalidation_price": invalidation,
                "entry_thesis_timestamp_utc": (
                    getattr(thesis, "timestamp_utc", None)
                    if thesis is not None
                    else None
                ),
            }
        )
    return rows


def _allowed_refs(
    *,
    snapshot: dict[str, Any],
    target: dict[str, Any],
    lanes: list[dict[str, Any]],
    attack_advice: dict[str, Any] | None,
    capture_economics: dict[str, Any] | None,
    profitability_acceptance: dict[str, Any] | None,
    execution_timing_audit: dict[str, Any] | None,
    coverage_optimization: dict[str, Any] | None,
    learning_audit: dict[str, Any] | None,
    verification_ledger: dict[str, Any] | None,
    self_improvement_audit: dict[str, Any] | None,
    projection_ledger: dict[str, Any] | None,
    operator_precedent: dict[str, Any] | None,
    manual_market_context: dict[str, Any] | None,
    predictive_limits: dict[str, Any] | None,
    market_status: dict[str, Any] | None,
    market_context_matrix: dict[str, Any] | None,
    option_skew: dict[str, Any] | None,
    news_items: dict[str, Any] | None,
    news_health: dict[str, Any] | None,
) -> list[str]:
    # Per docs/SKILL_trader.md the playbook prescribes a richer set of evidence
    # refs than the base broker/target/lane triple — the trader is required to
    # cite per-pair charts, cross-asset, flow, levels, currency strength,
    # economic calendar, and COT data. The verifier therefore must accept these
    # refs as known; otherwise every well-formed decision is rejected with
    # UNKNOWN_EVIDENCE_REF and the cycle never reaches the gateway.
    timeframes = DEFAULT_PAIR_CHART_TIMEFRAMES
    structure_keys = ("structure",)
    cross_assets = (
        "dxy",
        "USB10Y_USD",
        "USB02Y_USD",
        "SPX500_USD",
        "XAU_USD",
        "WTICO_USD",
        "BTC_USD",
        "spx",
        "gold",
        "oil",
        "btc",
    )
    refs = ["broker:snapshot", "target:daily", "verification:ledger"]
    pairs: set[str] = set()
    currencies: set[str] = set()
    if isinstance(market_status, dict):
        refs.append(str(market_status.get("evidence_ref") or "market:status"))
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict) or str(position.get("owner") or "") != "trader":
            continue
        pair = str(position.get("pair") or "")
        if pair:
            pairs.add(pair)
            for currency in pair.split("_"):
                if currency:
                    currencies.add(currency)
        trade_id = str(position.get("trade_id") or "")
        if not trade_id:
            continue
        refs.extend(
            [
                f"position:thesis:{trade_id}",
                f"position:evolution:{trade_id}",
                f"position:management:{trade_id}",
                f"position:guardian_management:{trade_id}",
                f"position:persistence:{trade_id}",
            ]
        )
    for lane in lanes:
        lane_id = lane["lane_id"]
        pair = str(lane.get("pair") or "")
        direction = str(lane.get("direction") or "")
        if pair:
            pairs.add(pair)
            for currency in pair.split("_"):
                if currency:
                    currencies.add(currency)
        refs.extend(
            [
                str(lane["evidence_ref"]),
                f"campaign:{lane_id}",
                f"strategy:{pair}:{direction}",
                f"story:{pair}",
                f"intent:{lane_id}",
            ]
        )
    for pair in pairs:
        for tf in timeframes:
            refs.append(f"chart:{pair}:{tf}")
        for key in structure_keys:
            refs.append(f"chart:{pair}:{key}")
        pair_refs = [
            f"flow:{pair}",
            f"levels:{pair}",
            f"calendar:{pair}",
            f"strength:{pair}",
            f"cross:correlations:{pair}",
        ]
        if _option_skew_enabled(option_skew):
            pair_refs.append(f"option:skew:{pair}")
        refs.extend(pair_refs)
        refs.append(f"matrix:{pair}")
        side_map = ((market_context_matrix or {}).get("pairs") or {}).get(pair)
        if isinstance(side_map, dict):
            for side in ("LONG", "SHORT"):
                if isinstance(side_map.get(side), dict):
                    refs.append(str(side_map[side].get("evidence_ref") or f"matrix:{pair}:{side}"))
        else:
            refs.extend([f"matrix:{pair}:LONG", f"matrix:{pair}:SHORT"])
    for currency in currencies:
        refs.append(f"cot:{currency}")
        refs.append(f"strength:{currency}")
        refs.append(f"calendar:{currency}")
    for asset in cross_assets:
        refs.append(f"cross:{asset}")
    refs.extend(["cross:dxy", "cross:correlations"])
    refs.append("broker:instruments")
    refs.extend(f"context_asset:{asset}" for asset in DEFAULT_CONTEXT_ASSETS)
    if _option_skew_enabled(option_skew):
        refs.extend(["option:skew", "option:skew:unknown"])
    if attack_advice:
        refs.append("attack:advice")
        for lane_id in attack_advice.get("recommended_now_lane_ids", []) or []:
            refs.append(f"attack:lane:{lane_id}")
        for lane_id in attack_advice.get("watchlist_lane_ids", []) or []:
            refs.append(f"attack:lane:{lane_id}")
    if capture_economics:
        refs.append("capture:economics")
        by_exit = capture_economics.get("by_exit_reason")
        if isinstance(by_exit, dict):
            for reason in by_exit:
                reason_key = str(reason or "").strip()
                if reason_key:
                    refs.append(f"capture:exit_reason:{reason_key}")
        segment_priorities = capture_economics.get("segment_repair_priorities")
        segment_items = (
            segment_priorities.get("items")
            if isinstance(segment_priorities, dict)
            else []
        )
        for item in segment_items or []:
            if not isinstance(item, dict):
                continue
            ref = str(item.get("evidence_ref") or "").strip()
            if ref:
                refs.append(ref)
    if profitability_acceptance:
        refs.append("profitability:acceptance")
        for finding in profitability_acceptance.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            code = str(finding.get("code") or "").strip()
            if code:
                refs.append(f"profitability:acceptance:{code}")
                for alias in PROFITABILITY_ACCEPTANCE_REF_ALIASES.get(code, ()):
                    refs.append(f"profitability:acceptance:{alias}")
    if execution_timing_audit:
        refs.extend(
            [
                "timing:audit",
                "timing:canceled_orders",
                "timing:loss_closes",
                "timing:market_closes",
            ]
        )
        for row in execution_timing_audit.get("canceled_order_regrets", []) or []:
            if not isinstance(row, dict):
                continue
            order_id = str(row.get("order_id") or "").strip()
            if order_id:
                refs.append(f"timing:canceled_order:{order_id}")
        shape_rollup = execution_timing_audit.get("canceled_order_regret_by_shape")
        shape_items = (
            shape_rollup.get("items")
            if isinstance(shape_rollup, dict)
            else []
        )
        for row in shape_items or []:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("evidence_ref") or "").strip()
            if ref:
                refs.append(ref)
        for row in execution_timing_audit.get("loss_close_regrets", []) or []:
            if not isinstance(row, dict):
                continue
            trade_id = str(row.get("trade_id") or "").strip()
            if trade_id:
                refs.append(f"timing:loss_close:{trade_id}")
        for row in execution_timing_audit.get("market_close_counterfactuals", []) or []:
            if not isinstance(row, dict):
                continue
            trade_id = str(row.get("trade_id") or "").strip()
            if trade_id:
                refs.append(f"timing:market_close:{trade_id}")
    if coverage_optimization:
        refs.append("coverage:optimization")
        diagnostics = (
            coverage_optimization.get("artifact_diagnostics")
            if isinstance(coverage_optimization.get("artifact_diagnostics"), dict)
            else {}
        )
        bucket_diag = (
            diagnostics.get("profitable_bucket_coverage")
            if isinstance(diagnostics.get("profitable_bucket_coverage"), dict)
            else {}
        )
        for edge in bucket_diag.get("top_edges", []) or []:
            if not isinstance(edge, dict):
                continue
            pair = str(edge.get("pair") or "")
            direction = str(edge.get("direction") or "")
            if pair and direction:
                refs.append(f"coverage:profitable_bucket:{pair}:{direction}")
        for edge in bucket_diag.get("matrix_supported_repair_queue", []) or []:
            if not isinstance(edge, dict):
                continue
            pair = str(edge.get("pair") or "")
            direction = str(edge.get("direction") or "")
            if pair and direction:
                refs.append(f"coverage:profitable_bucket:{pair}:{direction}")
    if learning_audit:
        refs.append("learning:audit")
        influence = learning_audit.get("learning_influence") if isinstance(learning_audit.get("learning_influence"), dict) else {}
        for lane in influence.get("lanes", []) or []:
            if not isinstance(lane, dict):
                continue
            lane_id = str(lane.get("lane_id") or "")
            if lane_id:
                refs.append(f"learning:lane:{lane_id}")
        effect = learning_audit.get("effect_metrics") if isinstance(learning_audit.get("effect_metrics"), dict) else {}
        exit_reasons = effect.get("exit_reason_metrics") if isinstance(effect.get("exit_reason_metrics"), dict) else {}
        for reason in exit_reasons:
            reason_key = str(reason or "").strip()
            if reason_key:
                refs.append(f"learning:exit_reason:{reason_key}")
    if verification_ledger:
        refs.extend(["verification:blockers", "verification:effect:all"])
        for key in ("blocking_evidence", "missing_artifacts", "learning_evidence", "measurements"):
            rows = verification_ledger.get(key) if isinstance(verification_ledger.get(key), list) else []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                ref = str(item.get("evidence_ref") or "").strip()
                if ref:
                    refs.append(ref)
    if self_improvement_audit:
        refs.extend(["self_improvement:audit", "self_improvement:profitability"])
        root_focus = (
            self_improvement_audit.get("root_cause_focus")
            if isinstance(self_improvement_audit.get("root_cause_focus"), dict)
            else {}
        )
        primary_root = (
            root_focus.get("primary") if isinstance(root_focus.get("primary"), dict) else {}
        )
        root_family = str(primary_root.get("family") or "").strip()
        if root_family:
            refs.append(f"self_improvement:root_cause:{root_family}")
        for finding in self_improvement_audit.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            layer = str(finding.get("layer") or "").strip()
            if layer:
                refs.append(f"self_improvement:{layer}")
            code = str(finding.get("code") or "").strip()
            if code:
                refs.append(f"self_improvement:finding:{code}")
    if projection_ledger:
        refs.append("projection:ledger")
        expired = _optional_int(projection_ledger.get("expired_pending_count")) or 0
        if expired > 0:
            refs.append("projection:expired_pending")
            for row in projection_ledger.get("expired_pending_examples", []) or []:
                if not isinstance(row, dict):
                    continue
                pair = str(row.get("pair") or "").strip()
                signal = str(row.get("signal_name") or "").strip()
                if pair:
                    refs.append(f"projection:expired_pending:{pair}")
                if pair and signal:
                    refs.append(f"projection:expired_pending:{pair}:{signal}")
    if operator_precedent:
        refs.append(OPERATOR_PRECEDENT_EVIDENCE_REF)
    if manual_market_context:
        refs.append(MANUAL_MARKET_CONTEXT_EVIDENCE_REF)
    if predictive_limits:
        refs.append("predictive:limits")
        for item in predictive_limits.get("orders", []) or []:
            if not isinstance(item, dict):
                continue
            pair = str(item.get("pair") or "")
            side = str(item.get("side") or "")
            if pair and side:
                refs.append(f"predictive:limit:{pair}:{side}")
    refs.extend(["news:items", "news:health"])
    if news_items:
        refs.append("news:current")
    if news_health:
        refs.append("news:freshness")
    for pair in pairs:
        refs.append(f"news:{pair}")
    for currency in currencies:
        refs.append(f"news:{currency}")
    return sorted(set(refs))


def _option_skew_enabled(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("enabled") is False and payload.get("disabled_reason"):
        return False
    return bool(payload.get("readings") or payload.get("issues") or payload.get("provider"))


def _attack_advice_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"evidence_ref": "attack:advice", "status": "missing"}
    recommended_ids = {str(item) for item in payload.get("recommended_now_lane_ids", []) or [] if str(item).strip()}
    lane_summaries: list[dict[str, Any]] = []
    learning_influenced_lane_ids: list[str] = []
    for lane in payload.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("lane_id") or "")
        if not lane_id or lane_id not in recommended_ids:
            continue
        influences = [str(item) for item in (lane.get("learning_influences") or []) if str(item).strip()]
        lane_summary = {
            "lane_id": lane_id,
            "learning_influences": influences,
            "learning_score_delta": lane.get("learning_score_delta"),
        }
        lane_summaries.append(lane_summary)
        if influences:
            learning_influenced_lane_ids.append(lane_id)
    return {
        "evidence_ref": "attack:advice",
        "status": payload.get("status"),
        "read_only": payload.get("read_only"),
        "live_permission": payload.get("live_permission"),
        "coverage_pct": payload.get("coverage_pct"),
        "recommended_now_lane_ids": list(payload.get("recommended_now_lane_ids", []) or []),
        "recommended_now_reward_jpy": payload.get("recommended_now_reward_jpy"),
        "recommended_now_risk_jpy": payload.get("recommended_now_risk_jpy"),
        "required_additional_reward_jpy": payload.get("required_additional_reward_jpy"),
        "recommended_lane_learning": lane_summaries[:20],
        "learning_influenced_lane_ids": learning_influenced_lane_ids[:20],
        "settings_advice": payload.get("settings_advice") if isinstance(payload.get("settings_advice"), dict) else {},
    }


def _capture_economics_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "capture:economics",
            "status": "missing",
            "overall": {},
            "by_exit_reason": {},
        }
    by_exit: dict[str, Any] = {}
    raw_by_exit = payload.get("by_exit_reason")
    if isinstance(raw_by_exit, dict):
        for reason, metrics in raw_by_exit.items():
            if not isinstance(metrics, dict):
                continue
            by_exit[str(reason)] = {
                "evidence_ref": f"capture:exit_reason:{reason}",
                **_small_dict(
                    metrics,
                    (
                        "trades",
                        "wins",
                        "losses",
                        "win_rate",
                        "payoff_ratio",
                        "breakeven_payoff_at_win_rate",
                        "expectancy_jpy_per_trade",
                        "net_jpy",
                        "avg_win_jpy",
                        "avg_loss_jpy",
                    ),
                ),
            }
    segment_repair_items: list[dict[str, Any]] = []
    raw_segment_priorities = payload.get("segment_repair_priorities")
    raw_segment_items = (
        raw_segment_priorities.get("items")
        if isinstance(raw_segment_priorities, dict)
        else []
    )
    for item in raw_segment_items or []:
        if not isinstance(item, dict):
            continue
        segment_repair_items.append(
            {
                "evidence_ref": str(item.get("evidence_ref") or ""),
                **_small_dict(
                    item,
                    (
                        "pair",
                        "side",
                        "method",
                        "priority_class",
                        "next_action",
                        "trades",
                        "win_rate",
                        "expectancy_jpy_per_trade",
                        "net_jpy",
                        "take_profit_trades",
                        "take_profit_proof_gap_trades",
                        "take_profit_proven",
                        "take_profit_expectancy_jpy",
                        "market_close_trades",
                        "market_close_losses",
                        "market_close_expectancy_jpy",
                        "market_close_net_jpy",
                    ),
                ),
            }
        )
    return {
        "evidence_ref": "capture:economics",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "overall": _small_dict(
            payload.get("overall"),
            (
                "trades",
                "wins",
                "losses",
                "win_rate",
                "payoff_ratio",
                "breakeven_payoff_at_win_rate",
                "expectancy_jpy_per_trade",
                "net_jpy",
                "avg_win_jpy",
                "avg_loss_jpy",
            ),
        ),
        "by_exit_reason": by_exit,
        "repair_summary": _small_dict(
            payload.get("repair_summary"),
            (
                "status",
                "payoff_gap_to_breakeven",
                "dominant_loss_exit_reason",
                "dominant_loss_exit_net_jpy",
                "dominant_loss_exit_expectancy_jpy_per_trade",
                "strongest_positive_exit_reason",
                "strongest_positive_exit_net_jpy",
                "top_negative_exit_reasons",
                "top_positive_exit_reasons",
            ),
        ),
        "segment_repair_priorities": {
            "basis": (
                raw_segment_priorities.get("basis")
                if isinstance(raw_segment_priorities, dict)
                else None
            ),
            "scoped_tp_proof_min_exit_trades": (
                raw_segment_priorities.get("scoped_tp_proof_min_exit_trades")
                if isinstance(raw_segment_priorities, dict)
                else None
            ),
            "items": segment_repair_items[:8],
        },
        "action_items": [str(item) for item in (payload.get("action_items") or [])[:6] if str(item).strip()],
    }


def _profitability_acceptance_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "profitability:acceptance",
            "status": "missing",
            "p0_findings": [],
        }
    p0_findings: list[dict[str, Any]] = []
    for finding in payload.get("findings", []) or []:
        if not isinstance(finding, dict):
            continue
        if str(finding.get("priority") or "").upper() != "P0":
            continue
        code = str(finding.get("code") or "").strip()
        aliases = [
            f"profitability:acceptance:{alias}"
            for alias in PROFITABILITY_ACCEPTANCE_REF_ALIASES.get(code, ())
        ]
        p0_findings.append(
            {
                "evidence_ref": f"profitability:acceptance:{code}",
                "evidence_ref_aliases": aliases,
                **_small_dict(
                    finding,
                    (
                        "code",
                        "message",
                        "next_action",
                    ),
                ),
                "evidence": _small_dict(
                    finding.get("evidence"),
                    (
                        "recent_loss_closes",
                        "recent_loss_net_jpy",
                        "latest_loss_close_ts_utc",
                        "examples",
                        "segments",
                    ),
                ),
            }
        )
    return {
        "evidence_ref": "profitability:acceptance",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "blockers": list(payload.get("blockers", []) or [])[:8],
        "p0_findings": p0_findings,
    }


def _execution_timing_audit_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "timing:audit",
            "status": "missing",
            "summary": {},
            "canceled_order_regrets": [],
            "loss_close_regrets": [],
            "market_close_counterfactuals": [],
            "post_stop_thesis_reviews": [],
        }
    post_stop_sources = [
        item
        for item in (
            list(payload.get("loss_close_regrets") or [])
            + list(payload.get("market_close_counterfactuals") or [])
        )
        if isinstance(item, dict)
    ]
    return {
        "evidence_ref": "timing:audit",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "window": _small_dict(
            payload.get("window"),
            ("from_utc", "to_utc", "lookback_hours", "post_cancel_hours", "post_close_hours"),
        ),
        "summary": _small_dict(
            payload.get("summary"),
            (
                "canceled_orders_audited",
                "canceled_entry_touched_after_cancel_rate",
                "canceled_tp_touched_after_cancel_rate",
                "canceled_estimated_missed_mfe_jpy",
                "loss_closes_audited",
                "loss_closes_had_positive_mfe",
                "loss_closes_had_positive_mfe_rate",
                "loss_closes_tp_touched_before_close",
                "loss_closes_tp_touched_before_close_rate",
                "loss_close_estimated_mfe_jpy",
                "loss_closes_profit_capture_missed",
                "loss_closes_profit_capture_missed_rate",
                "stop_loss_closes_profit_capture_missed",
                "tp_progress_repair_live_evidence_boundary_utc",
                "tp_progress_repair_live_evidence_status",
                "pre_repair_historical_loss_closes_profit_capture_missed",
                "pre_repair_historical_loss_closes_repair_replay_triggered",
                "post_repair_live_evidence_loss_closes_audited",
                "post_repair_live_evidence_loss_closes_profit_capture_missed",
                "post_repair_live_evidence_loss_closes_repair_replay_triggered",
                "loss_close_estimated_capture_gap_jpy",
                "loss_close_actual_pl_jpy",
                "loss_close_counterfactual_profit_capture_pl_jpy",
                "loss_close_counterfactual_profit_capture_delta_jpy",
                "loss_close_counterfactual_profit_capture_jpy",
                "avg_decision_lag_minutes_after_first_positive",
                "max_decision_lag_minutes_after_first_positive",
                "market_closes_audited",
                "market_closes_post_close_continued_rate",
                "market_closes_post_close_adverse_rate",
                "market_closes_tp_touched_after_close_rate",
                "market_closes_sl_touched_after_close_rate",
                "market_close_estimated_followthrough_jpy",
                "market_close_estimated_avoided_adverse_jpy",
                "profit_market_closes_left_runner_upside",
                "profit_market_closes_avoided_giveback",
                "loss_market_closes_may_have_been_premature",
                "loss_market_closes_contained_risk",
            ),
        ),
        "canceled_order_regrets": [
            {
                "evidence_ref": f"timing:canceled_order:{item.get('order_id')}",
                **_small_dict(
                    item,
                    (
                        "order_id",
                        "lane_id",
                        "pair",
                        "side",
                        "order_type",
                        "entry_touched_after_cancel",
                        "tp_touched_after_cancel",
                        "sl_touched_after_cancel",
                        "mfe_pips_after_cancel_entry",
                        "estimated_missed_mfe_jpy",
                    ),
                ),
            }
            for item in (payload.get("canceled_order_regrets") or [])[:12]
            if isinstance(item, dict)
        ],
        "canceled_order_regret_by_shape": [
            {
                "evidence_ref": str(item.get("evidence_ref") or ""),
                **_small_dict(
                    item,
                    (
                        "pair",
                        "side",
                        "method",
                        "order_type",
                        "priority_class",
                        "next_action",
                        "orders",
                        "entry_touched_after_cancel",
                        "entry_touch_after_cancel_rate",
                        "positive_after_cancel_entry",
                        "positive_after_cancel_entry_rate",
                        "tp_touched_after_cancel",
                        "tp_touched_after_cancel_rate",
                        "estimated_missed_mfe_jpy",
                        "avg_entry_touch_after_cancel_minutes",
                        "avg_tp_touch_after_cancel_minutes",
                    ),
                ),
            }
            for item in (
                ((payload.get("canceled_order_regret_by_shape") or {}).get("items") or [])
                if isinstance(payload.get("canceled_order_regret_by_shape"), dict)
                else []
            )[:8]
            if isinstance(item, dict)
        ],
        "loss_close_regrets": [
            {
                "evidence_ref": f"timing:loss_close:{item.get('trade_id')}",
                **_small_dict(
                    item,
                    (
                        "trade_id",
                        "lane_id",
                        "pair",
                        "side",
                        "gateway_action",
                        "realized_pl_jpy",
                        "had_positive_mfe_before_loss_close",
                        "tp_touched_before_loss_close",
                        "sl_touched_before_loss_close",
                        "mfe_pips_before_loss_close",
                        "estimated_mfe_jpy_before_loss_close",
                        "tp_progress_before_loss_close",
                        "profit_capture_missed_before_loss_close",
                        "profit_capture_progress_threshold",
                        "profit_capture_counterfactual_exit",
                        "profit_capture_counterfactual_pips",
                        "profit_capture_counterfactual_jpy",
                        "profit_capture_counterfactual_net_improvement_jpy",
                        "profit_capture_counterfactual_pl_jpy",
                        "first_positive_minutes_after_fill",
                        "decision_lag_minutes_after_first_positive",
                    ),
                ),
            }
            for item in (payload.get("loss_close_regrets") or [])[:12]
            if isinstance(item, dict)
        ],
        "market_close_counterfactuals": [
            {
                "evidence_ref": f"timing:market_close:{item.get('trade_id')}",
                **_small_dict(
                    item,
                    (
                        "trade_id",
                        "lane_id",
                        "pair",
                        "side",
                        "gateway_action",
                        "realized_pl_jpy",
                        "post_close_path_label",
                        "post_close_favorable_pips",
                        "estimated_post_close_favorable_jpy",
                        "post_close_adverse_pips",
                        "estimated_post_close_adverse_jpy",
                        "tp_touched_after_market_close",
                        "sl_touched_after_market_close",
                    ),
                ),
            }
            for item in (payload.get("market_close_counterfactuals") or [])[:12]
            if isinstance(item, dict)
        ],
        "post_stop_thesis_reviews": [
            post_stop_thesis_review(item)
            for item in post_stop_sources[:12]
            if _post_stop_review_relevant(item)
        ],
    }


def post_stop_thesis_review(event: dict[str, Any]) -> dict[str, Any]:
    """Answer the post-stop review questions without treating a stop as thesis failure."""

    event = event if isinstance(event, dict) else {}
    lint_codes = _post_stop_lint_codes(event)
    thesis_failure_reasons = _post_stop_thesis_failure_reasons(event)
    thesis_failed = bool(thesis_failure_reasons)
    price_later_moved_intended_direction = _post_stop_favorable_followthrough(event)
    sl_inside_noise_or_battle_zone = bool(
        event.get("sl_inside_noise_or_battle_zone")
        or lint_codes.intersection(
            {
                "SL_LINT_NORMAL_NOISE_BAND",
                "SL_LINT_MAJOR_FIGURE_BATTLE_ZONE",
                "SL_LINT_RECENT_WICK_STOP_RUN_ZONE",
                "SL_LINT_EVENT_INTERVENTION_ZONE",
                "SL_LINT_JPY_THEME_INVALIDATION_REQUIRED",
            }
        )
    )
    stopish = _post_stop_review_relevant(event)
    broker_sl_failure = (
        stopish
        and not thesis_failed
        and price_later_moved_intended_direction
        and (sl_inside_noise_or_battle_zone or bool(lint_codes))
    )
    if broker_sl_failure:
        next_cycle_action = "RE_ENTER"
    elif thesis_failed:
        next_cycle_action = "WAIT"
    elif price_later_moved_intended_direction:
        next_cycle_action = "SCOUT"
    else:
        next_cycle_action = "WAIT_FOR_NEW_EVIDENCE"
    return {
        "evidence_ref": f"post_stop_review:{event.get('trade_id') or event.get('lane_id') or 'unknown'}",
        "trade_id": event.get("trade_id"),
        "lane_id": event.get("lane_id"),
        "pair": event.get("pair"),
        "side": event.get("side"),
        "gateway_action": event.get("gateway_action") or event.get("exit_reason"),
        "realized_pl_jpy": event.get("realized_pl_jpy"),
        "thesis_failed": thesis_failed,
        "thesis_failure_reasons": thesis_failure_reasons,
        "price_later_moved_intended_direction": price_later_moved_intended_direction,
        "broker_sl_failure": broker_sl_failure,
        "sl_inside_noise_or_battle_zone": sl_inside_noise_or_battle_zone,
        "sl_lint_codes": sorted(lint_codes),
        "next_cycle_action": next_cycle_action,
        "review_questions": {
            "did_thesis_fail": thesis_failed,
            "what_proved_thesis_dead": thesis_failure_reasons,
            "did_price_later_move_intended_direction": price_later_moved_intended_direction,
            "was_sl_inside_noise_or_battle_zone": sl_inside_noise_or_battle_zone,
            "was_this_broker_sl_failure": broker_sl_failure,
        },
    }


def _post_stop_review_relevant(event: dict[str, Any]) -> bool:
    text = " ".join(
        str(event.get(key) or "")
        for key in ("gateway_action", "exit_reason", "post_close_path_label")
    ).upper()
    return "STOP" in text or bool(event.get("sl_touched_after_market_close") or event.get("sl_touched_before_loss_close"))


def _post_stop_lint_codes(event: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    for key in ("sl_lint_codes", "sl_lint_issue_codes"):
        raw = event.get(key)
        if isinstance(raw, (list, tuple, set)):
            codes.update(str(item) for item in raw if str(item))
    lint = event.get("sl_lint")
    if isinstance(lint, dict):
        raw_issues = lint.get("issues")
        if isinstance(raw_issues, list):
            for issue in raw_issues:
                if isinstance(issue, dict) and issue.get("code"):
                    codes.add(str(issue["code"]))
        raw_code = lint.get("code")
        if raw_code:
            codes.add(str(raw_code))
    return codes


def _post_stop_thesis_failure_reasons(event: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    fields = (
        ("price_invalidation", "price invalidation"),
        ("timeframe_invalidation", "timeframe invalidation"),
        ("structure_invalidation", "structure invalidation"),
        ("currency_theme_invalidation", "currency-theme invalidation"),
        ("margin_emergency_invalidation", "margin/emergency invalidation"),
        ("operator_override", "operator override"),
        ("gate_a_invalidated", "Gate A invalidation"),
        ("thesis_failed", "explicit thesis_failed"),
    )
    for key, label in fields:
        if event.get(key) is True:
            reasons.append(label)
    for key in ("thesis_failure_reason", "what_proved_thesis_dead", "gate_a_reason"):
        value = str(event.get(key) or "").strip()
        if value:
            reasons.append(value)
    return reasons


def _post_stop_favorable_followthrough(event: dict[str, Any]) -> bool:
    if event.get("tp_touched_after_market_close") or event.get("tp_touched_after_loss_close"):
        return True
    for key in (
        "post_close_favorable_pips",
        "favorable_after_stop_pips",
        "mfe_pips_after_loss_close",
        "profit_capture_counterfactual_pips",
    ):
        value = _optional_float(event.get(key))
        if value is not None and value > 0:
            return True
    return False


def _coverage_optimization_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "coverage:optimization",
            "status": "missing",
            "live_permission": False,
            "profitable_bucket_coverage": {},
            "action_items": [],
        }
    diagnostics = payload.get("artifact_diagnostics") if isinstance(payload.get("artifact_diagnostics"), dict) else {}
    bucket_diag = (
        diagnostics.get("profitable_bucket_coverage")
        if isinstance(diagnostics.get("profitable_bucket_coverage"), dict)
        else {}
    )
    return {
        "evidence_ref": "coverage:optimization",
        "status": payload.get("status"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "live_permission": False,
        **_small_dict(
            payload,
            (
                "remaining_target_jpy",
                "remaining_risk_budget_jpy",
                "live_ready_reward_jpy",
                "live_ready_risk_jpy",
                "potential_reward_jpy",
                "coverage_pct",
                "potential_coverage_pct",
                "sequential_ladder_reward_jpy",
                "sequential_ladder_steps",
            ),
        ),
        "diagnostics": _small_dict(
            diagnostics,
            (
                "intents_artifact_stale",
                "all_lanes_spread_blocked",
                "spread_normalized_candidate_count",
                "spread_normalized_candidate_reward_jpy",
                "spread_normalized_no_live_blocker_count",
                "spread_normalized_no_live_blocker_reward_jpy",
                "market_context_matrix_missing",
            ),
        ),
        "profitable_bucket_coverage": _profitable_bucket_coverage_packet(bucket_diag),
        "opportunity_modes": _opportunity_modes_packet(payload.get("opportunity_modes")),
        "runner_candidate_diagnostics": _runner_candidate_diagnostics_packet(
            payload.get("runner_candidate_diagnostics")
        ),
        "perspective_alignment_diagnostics": _perspective_alignment_diagnostics_packet(
            payload.get("perspective_alignment_diagnostics")
        ),
        "action_items": [str(item) for item in (payload.get("action_items") or [])[:8] if str(item).strip()],
    }


def _perspective_alignment_diagnostics_packet(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "status": payload.get("status"),
        "pair_direction_groups": payload.get("pair_direction_groups"),
        "range_forecast_method_mismatch_groups": payload.get("range_forecast_method_mismatch_groups"),
        "range_forecast_method_mismatch_lanes": payload.get("range_forecast_method_mismatch_lanes"),
        "range_forecast_method_mismatch_top": [
            {
                "pair": str(item.get("pair") or ""),
                "direction": str(item.get("direction") or ""),
                "method_mismatch_lanes": item.get("method_mismatch_lanes"),
                "method_mismatch_reward_jpy": item.get("method_mismatch_reward_jpy"),
                "range_rotation_lanes": item.get("range_rotation_lanes"),
                "range_rotation_live_ready_lanes": item.get("range_rotation_live_ready_lanes"),
                "range_rotation_absence_reason": item.get("range_rotation_absence_reason"),
                "range_rotation_other_side_lanes": item.get("range_rotation_other_side_lanes"),
                "range_rotation_other_side_directions": list(
                    item.get("range_rotation_other_side_directions") or []
                )[:5],
                "method_counts": list(item.get("method_counts") or [])[:5],
                "forecast_direction_counts": list(item.get("forecast_direction_counts") or [])[:5],
                "chart_direction_bias_counts": list(item.get("chart_direction_bias_counts") or [])[:5],
                "range_rotation_top_live_blocker_codes": list(
                    item.get("range_rotation_top_live_blocker_codes") or []
                )[:5],
                "range_rotation_other_side_top_live_blocker_codes": list(
                    item.get("range_rotation_other_side_top_live_blocker_codes") or []
                )[:5],
                "range_rotation_other_side_top_blockers": list(
                    item.get("range_rotation_other_side_top_blockers") or []
                )[:4],
                "top_live_blocker_codes": list(item.get("top_live_blocker_codes") or [])[:5],
                "top_lanes": [
                    {
                        "lane_id": str(lane.get("lane_id") or ""),
                        "status": lane.get("status"),
                        "method": lane.get("method"),
                        "forecast_direction": lane.get("forecast_direction"),
                        "chart_direction_bias": lane.get("chart_direction_bias"),
                        "reward_jpy": lane.get("reward_jpy"),
                        "reward_risk": lane.get("reward_risk"),
                    }
                    for lane in (item.get("top_lanes") or [])[:4]
                    if isinstance(lane, dict) and str(lane.get("lane_id") or "").strip()
                ],
            }
            for item in (payload.get("range_forecast_method_mismatch_top") or [])[:8]
            if isinstance(item, dict) and str(item.get("pair") or "").strip()
        ],
    }


def _runner_candidate_diagnostics_packet(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "status": payload.get("status"),
        "trend_candidate_lanes": payload.get("trend_candidate_lanes"),
        "runner_qualified_lanes": payload.get("runner_qualified_lanes"),
        "attached_harvest_lanes": payload.get("attached_harvest_lanes"),
        "status_counts": payload.get("status_counts") if isinstance(payload.get("status_counts"), dict) else {},
        "top_demotion_reasons": [
            {
                "reason": str(item.get("reason") or ""),
                "count": item.get("count"),
            }
            for item in (payload.get("top_demotion_reasons") or [])[:5]
            if isinstance(item, dict) and str(item.get("reason") or "").strip()
        ],
        "top_issue_codes": [
            {
                "code": str(item.get("code") or ""),
                "count": item.get("count"),
            }
            for item in (payload.get("top_issue_codes") or [])[:5]
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        ],
        "top_live_blocker_codes": [
            {
                "code": str(item.get("code") or ""),
                "count": item.get("count"),
            }
            for item in (payload.get("top_live_blocker_codes") or [])[:5]
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        ],
        "top_lanes": [
            {
                "lane_id": str(item.get("lane_id") or ""),
                "status": item.get("status"),
                "opportunity_mode": item.get("opportunity_mode"),
                "tp_execution_mode": item.get("tp_execution_mode"),
                "tp_attach_reason": item.get("tp_attach_reason"),
                "reward_risk": item.get("reward_risk"),
            }
            for item in (payload.get("top_lanes") or [])[:5]
            if isinstance(item, dict) and str(item.get("lane_id") or "").strip()
        ],
    }


def _opportunity_modes_packet(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for mode in ("HARVEST", "RUNNER", "BALANCED"):
        item = payload.get(mode)
        if not isinstance(item, dict):
            continue
        out[mode] = {
            "lanes": item.get("lanes"),
            "live_ready_lanes": item.get("live_ready_lanes"),
            "promotion_candidate_lanes": item.get("promotion_candidate_lanes"),
            "reward_jpy": item.get("reward_jpy"),
            "live_ready_reward_jpy": item.get("live_ready_reward_jpy"),
            "potential_reward_jpy": item.get("potential_reward_jpy"),
            "coverage_pct": item.get("coverage_pct"),
            "potential_coverage_pct": item.get("potential_coverage_pct"),
            "diagnostic_candidate_lanes": item.get("diagnostic_candidate_lanes"),
            "demoted_to_harvest_lanes": item.get("demoted_to_harvest_lanes"),
            "runner_qualified_lanes": item.get("runner_qualified_lanes"),
            "diagnostic_status": item.get("diagnostic_status"),
            "top_demotion_reasons": list(item.get("top_demotion_reasons") or [])[:5],
            "top_issue_codes": list(item.get("top_issue_codes") or [])[:5],
            "top_live_blocker_codes": list(item.get("top_live_blocker_codes") or [])[:5],
            "top_blockers": list(item.get("top_blockers") or [])[:3],
            "top_lanes": list(item.get("top_lanes") or [])[:3],
        }
    return out


def _profitable_bucket_coverage_packet(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    top_edges: list[dict[str, Any]] = []
    for edge in payload.get("top_edges", []) or []:
        if not isinstance(edge, dict):
            continue
        pair = str(edge.get("pair") or "")
        direction = str(edge.get("direction") or "")
        evidence_ref = f"coverage:profitable_bucket:{pair}:{direction}" if pair and direction else None
        top_edges.append(
            {
                "evidence_ref": evidence_ref,
                "pair": pair,
                "direction": direction,
                "coverage_state": edge.get("coverage_state"),
                "managed_net_jpy": edge.get("managed_net_jpy"),
                "raw_net_jpy": edge.get("raw_net_jpy"),
                "trades": edge.get("trades"),
                "days": edge.get("days"),
                "current_lane_count": edge.get("current_lane_count"),
                "current_status_counts": edge.get("current_status_counts") if isinstance(edge.get("current_status_counts"), dict) else {},
                "current_best_reward_jpy": edge.get("current_best_reward_jpy"),
                "spread_normalized_candidate_count": edge.get("spread_normalized_candidate_count"),
                "spread_normalized_no_live_blocker_count": edge.get("spread_normalized_no_live_blocker_count"),
                "top_blockers": [str(item) for item in (edge.get("top_blockers") or [])[:5] if str(item).strip()],
                "strategy_profile_status": edge.get("strategy_profile_status"),
                "strategy_profile_required_fix": edge.get("strategy_profile_required_fix"),
                "strategy_profile_blocks_live": edge.get("strategy_profile_blocks_live"),
                "strategy_profile_live_net_jpy": edge.get("strategy_profile_live_net_jpy"),
                "strategy_profile_pretrade_net_jpy": edge.get("strategy_profile_pretrade_net_jpy"),
                "strategy_profile_seat_net_jpy": edge.get("strategy_profile_seat_net_jpy"),
                "strategy_profile_seat_win_rate_pct": edge.get("strategy_profile_seat_win_rate_pct"),
                "matrix_ref": edge.get("matrix_ref"),
                "matrix_support_count": edge.get("matrix_support_count"),
                "matrix_reject_count": edge.get("matrix_reject_count"),
                "matrix_warning_count": edge.get("matrix_warning_count"),
                "matrix_strongest_support": edge.get("matrix_strongest_support"),
                "matrix_strongest_reject": edge.get("matrix_strongest_reject"),
                "matrix_cross_asset_context": [
                    str(item) for item in (edge.get("matrix_cross_asset_context") or [])[:4] if str(item).strip()
                ],
            }
        )
        if len(top_edges) >= 12:
            break
    return {
        "source_status": payload.get("source_status"),
        "live_permission": False,
        "positive_pair_directions": payload.get("positive_pair_directions"),
        "positive_managed_net_jpy": payload.get("positive_managed_net_jpy"),
        "positive_trade_count": payload.get("positive_trade_count"),
        "state_counts": payload.get("state_counts") if isinstance(payload.get("state_counts"), dict) else {},
        "top_edges": top_edges,
        "matrix_supported_repair_queue": _matrix_supported_repair_queue_packet(
            payload.get("matrix_supported_repair_queue")
            if isinstance(payload.get("matrix_supported_repair_queue"), list)
            else []
        ),
    }


def _matrix_supported_repair_queue_packet(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        pair = str(item.get("pair") or "")
        direction = str(item.get("direction") or "")
        evidence_ref = f"coverage:profitable_bucket:{pair}:{direction}" if pair and direction else None
        out.append(
            {
                "evidence_ref": evidence_ref,
                "pair": pair,
                "direction": direction,
                "coverage_state": item.get("coverage_state"),
                "managed_net_jpy": item.get("managed_net_jpy"),
                "top_blockers": [str(value) for value in (item.get("top_blockers") or [])[:4] if str(value).strip()],
                "strategy_profile_status": item.get("strategy_profile_status"),
                "matrix_ref": item.get("matrix_ref"),
                "matrix_support_count": item.get("matrix_support_count"),
                "matrix_reject_count": item.get("matrix_reject_count"),
                "matrix_support_layers": [
                    str(value) for value in (item.get("matrix_support_layers") or [])[:6] if str(value).strip()
                ],
                "matrix_support_context": [
                    str(value) for value in (item.get("matrix_support_context") or [])[:4] if str(value).strip()
                ],
            }
        )
        if len(out) >= 8:
            break
    return out


def _learning_audit_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "learning:audit",
            "status": "missing",
            "blockers": [],
            "warnings": [],
            "learning_influence": {
                "influenced_lanes": 0,
                "total_learning_score_delta": 0.0,
                "lanes": [],
            },
        }
    influence = payload.get("learning_influence") if isinstance(payload.get("learning_influence"), dict) else {}
    lanes: list[dict[str, Any]] = []
    for lane in influence.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lanes.append(
            {
                "evidence_ref": f"learning:lane:{lane.get('lane_id')}",
                "lane_id": lane.get("lane_id"),
                "learning_influences": list(lane.get("learning_influences", []) or []),
                "learning_score_delta": lane.get("learning_score_delta"),
            }
        )
    effect = payload.get("effect_metrics") if isinstance(payload.get("effect_metrics"), dict) else {}
    effect_packet = _small_dict(
        effect,
        (
            "closed_trades",
            "net_jpy",
            "profit_factor",
            "expectancy_jpy",
        ),
    )
    exit_reason_metrics = _learning_exit_reason_metrics(effect)
    if exit_reason_metrics:
        effect_packet["exit_reason_metrics"] = exit_reason_metrics
    return {
        "evidence_ref": "learning:audit",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "blockers": list(payload.get("blockers", []) or [])[:12],
        "warnings": list(payload.get("warnings", []) or [])[:12],
        "effect_metrics": effect_packet,
        "learning_influence": {
            "influenced_lanes": influence.get("influenced_lanes", 0),
            "total_learning_score_delta": influence.get("total_learning_score_delta", 0.0),
            "lanes": lanes[:20],
        },
    }


def _learning_influenced_lane_evidence_requirements(
    attack_packet: dict[str, Any],
    learning_packet: dict[str, Any],
) -> list[dict[str, Any]]:
    lane_ids = [str(item) for item in attack_packet.get("learning_influenced_lane_ids", []) or [] if str(item).strip()]
    if not lane_ids:
        return []
    audit_lanes = learning_packet.get("learning_influence", {}).get("lanes") or []
    audit_lane_ids = {
        str(lane.get("lane_id") or "")
        for lane in audit_lanes
        if isinstance(lane, dict) and str(lane.get("lane_id") or "").strip()
    }
    status = str(learning_packet.get("status") or "missing")
    requirements: list[dict[str, Any]] = []
    for lane_id in lane_ids[:20]:
        lane_ref = f"learning:lane:{lane_id}"
        requirements.append(
            {
                "lane_id": lane_id,
                "audit_status": status,
                "covered_by_learning_audit": lane_id in audit_lane_ids,
                "required_evidence_refs": ["learning:audit", lane_ref],
                "verifier_rule": (
                    "TRADE selecting this learning-influenced lane is rejected unless "
                    "decision.evidence_refs includes every required_evidence_refs value."
                ),
            }
        )
    return requirements


def _learning_exit_reason_metrics(effect: dict[str, Any]) -> dict[str, dict[str, Any]]:
    exit_reasons = effect.get("exit_reason_metrics") if isinstance(effect.get("exit_reason_metrics"), dict) else {}
    rows: list[tuple[float, str, dict[str, Any]]] = []
    for reason, metrics in exit_reasons.items():
        reason_key = str(reason or "").strip()
        if not reason_key or not isinstance(metrics, dict):
            continue
        compact = _small_dict(
            metrics,
            (
                "closed_trades",
                "net_jpy",
                "gross_profit_jpy",
                "gross_loss_jpy",
                "profit_factor",
                "win_rate",
                "expectancy_jpy",
            ),
        )
        compact["evidence_ref"] = f"learning:exit_reason:{reason_key}"
        net = _optional_float(metrics.get("net_jpy"))
        rows.append((net if net is not None else 0.0, reason_key, compact))
    return {reason: compact for _net, reason, compact in sorted(rows, key=lambda item: item[0])[:8]}


def _verification_ledger_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "verification:ledger",
            "status": "missing",
            "blocking_observations": 0,
            "missing_observations": 0,
            "effect_metrics": {},
            "blocking_evidence": [],
            "learning_evidence": [],
        }
    effect = payload.get("effect_metrics") if isinstance(payload.get("effect_metrics"), dict) else {}
    return {
        "evidence_ref": "verification:ledger",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "db_path": payload.get("db_path"),
        "report_path": payload.get("report_path"),
        "blocking_observations": payload.get("blocking_observations"),
        "missing_observations": payload.get("missing_observations"),
        "effect_metrics": _small_dict(
            effect,
            (
                "window_hours",
                "closed_trades",
                "net_jpy",
                "profit_factor",
                "win_rate",
                "expectancy_jpy",
                "sample_warning",
            ),
        ),
        "blocking_evidence": _verification_rows(payload.get("blocking_evidence")),
        "missing_artifacts": _verification_rows(payload.get("missing_artifacts")),
        "learning_evidence": _verification_rows(payload.get("learning_evidence")),
        "measurements": _verification_rows(payload.get("measurements")),
        "contract": _small_dict(
            payload.get("contract"),
            (
                "read_only",
                "live_permission",
                "sqlite_tables",
                "json_packet_is_trader_readable",
                "markdown_report_is_operator_readable",
                "learning_cannot_override_risk_or_gateway_gates",
            ),
        ),
    }


def _verification_rows(rows: object) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for item in rows[:12]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "evidence_ref",
                    "source",
                    "subject_type",
                    "subject_id",
                    "check_name",
                    "status",
                    "severity",
                    "metric_value",
                    "metric_unit",
                    "evidence",
                ),
            )
        )
    return out


def _self_improvement_audit_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": "self_improvement:audit",
            "status": "missing",
            "p0_findings": 0,
            "p0_blockers": [],
            "profitability_blockers": [],
            "new_risk_blockers": [],
        }
    blockers: list[dict[str, Any]] = []
    p0_blockers: list[dict[str, Any]] = []
    for finding in payload.get("findings", []) or []:
        if not isinstance(finding, dict):
            continue
        code = str(finding.get("code") or "")
        priority = str(finding.get("priority") or "").upper()
        layer = str(finding.get("layer") or "")
        evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
        if priority == "P0":
            p0_blockers.append(
                {
                    "evidence_ref": f"self_improvement:finding:{code}",
                    "code": code,
                    "layer": layer,
                    "message": finding.get("message"),
                    "next_action": finding.get("next_action"),
                    "current_streak": evidence.get("current_streak"),
                    "count": evidence.get("count"),
                    "cancel_review_order_ids": list(
                        evidence.get("cancel_review_order_ids", []) or []
                    )[:10],
                    "examples": list(evidence.get("examples", []) or [])[:3],
                }
            )
        if priority == "P0" and layer == "profitability" and code == "PERSISTENT_PROFITABILITY_DISCIPLINE_BLOCKED":
            evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
            system_evidence = evidence.get("system_defect_evidence") if isinstance(evidence.get("system_defect_evidence"), dict) else {}
            blockers.append(
                {
                    "evidence_ref": f"self_improvement:finding:{code}",
                    "code": code,
                    "message": finding.get("message"),
                    "next_action": finding.get("next_action"),
                    "current_streak": evidence.get("current_streak"),
                    "profit_factor": system_evidence.get("profit_factor"),
                    "expectancy_jpy": system_evidence.get("expectancy_jpy"),
                    "avg_win_jpy": system_evidence.get("avg_win_jpy"),
                    "avg_loss_jpy_abs": system_evidence.get("avg_loss_jpy_abs"),
                    "worst_segments": list(system_evidence.get("worst_segments", []) or [])[:5],
                }
            )
    new_risk_blockers: list[dict[str, Any]] = []
    forecast_blocker = forecast_adverse_path_new_risk_blocker(payload)
    if forecast_blocker is not None:
        new_risk_blockers.append(forecast_blocker)
    return {
        "evidence_ref": "self_improvement:audit",
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "p0_findings": payload.get("p0_findings"),
        "p1_findings": payload.get("p1_findings"),
        "p2_findings": payload.get("p2_findings"),
        "effect_metrics": _small_dict(
            payload.get("effect_metrics"),
            (
                "closed_trades",
                "net_jpy",
                "profit_factor",
                "expectancy_jpy",
                "avg_win_jpy",
                "avg_loss_jpy_abs",
            ),
        ),
        "p0_blockers": p0_blockers,
        "profitability_blockers": blockers,
        "new_risk_blockers": new_risk_blockers,
    }


def _operator_precedent_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": OPERATOR_PRECEDENT_EVIDENCE_REF,
            "status": "missing",
            "operator_claim": {},
            "winning_shape": {},
            "generalized_trade_shape_precedent": {},
            "runtime_alignment": {},
            "trade_shape_engine": {},
            "warnings": [],
            "blockers": [],
        }
    precedent = payload.get("precedent") if isinstance(payload.get("precedent"), dict) else {}
    runtime = payload.get("runtime_alignment") if isinstance(payload.get("runtime_alignment"), dict) else {}
    return {
        "evidence_ref": OPERATOR_PRECEDENT_EVIDENCE_REF,
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "operator_claim": _small_dict(
            payload.get("operator_claim"),
            ("claim", "required_return_pct", "verified"),
        ),
        "winning_shape": _small_dict(
            precedent.get("winning_shape"),
            (
                "primary_pair",
                "primary_direction",
                "primary_sessions",
                "positive_sessions",
                "negative_sessions",
                "expectancy_jpy_per_exit",
                "median_hold_hours",
                "payoff",
            ),
        ),
        "generalized_trade_shape_precedent": _small_dict(
            payload.get("generalized_trade_shape_precedent"),
            ("id", "source_history_pair", "pair_agnostic", "winning_shape", "operator_memory"),
        ),
        "failure_shape": _small_dict(
            (precedent.get("failure_shape") or {}).get("margin_closeout"),
            ("trades", "net_jpy", "win_rate", "median_hold_hours"),
        ),
        "runtime_alignment": _small_dict(
            runtime,
            (
                "live_ready_lanes",
                "aligned_live_ready_lanes",
                "aligned_lanes",
                "legacy_pair_direction_session_aligned_live_ready_lanes",
                "manual_context_alignment",
                "manual_exit_events_per_calendar_day",
                "target_trades_per_day",
                "alignment_contract",
            ),
        ),
        "trade_shape_engine": _trade_shape_engine_packet(runtime.get("trade_shape_engine")),
        "warnings": list(payload.get("warnings") or [])[:5],
        "blockers": list(payload.get("blockers") or [])[:5],
    }


def _trade_shape_engine_packet(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "status": payload.get("status"),
        "candidate_count": payload.get("candidate_count"),
        "candidate_pairs": list(payload.get("candidate_pairs") or [])[:20],
        "shape_matched_live_ready_count": payload.get("shape_matched_live_ready_count"),
        "shape_matched_live_ready_lanes": list(payload.get("shape_matched_live_ready_lanes") or [])[:10],
        "pair_summaries": {
            str(pair): row
            for pair, row in list((payload.get("pair_summaries") or {}).items())[:20]
            if isinstance(row, dict)
        },
        "contract": _small_dict(
            payload.get("contract"),
            (
                "advisory_only",
                "pair_agnostic_core",
                "pair_specific_overlays_are_adjustments_only",
                "does_not_grant_live_permission",
                "does_not_replace_risk_engine",
                "does_not_force_usd_jpy_only_trading",
            ),
        ),
    }


def _manual_market_context_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "evidence_ref": MANUAL_MARKET_CONTEXT_EVIDENCE_REF,
            "status": "missing",
            "sample": {},
            "guidance": {},
            "bounded_replay_profile": {},
            "excluded_tail_profile": {},
            "position_building": {},
            "warnings": [],
            "blockers": [],
        }
    bounded = payload.get("bounded_replay_profile") if isinstance(payload.get("bounded_replay_profile"), dict) else {}
    excluded = payload.get("excluded_tail_profile") if isinstance(payload.get("excluded_tail_profile"), dict) else {}
    building = payload.get("position_building_profile") if isinstance(payload.get("position_building_profile"), dict) else {}
    return {
        "evidence_ref": MANUAL_MARKET_CONTEXT_EVIDENCE_REF,
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "sample": _small_dict(
            payload.get("sample"),
            ("pair", "manual_trades", "analyzed_trades", "coverage_pct"),
        ),
        "guidance": payload.get("guidance") if isinstance(payload.get("guidance"), dict) else {},
        "bounded_replay_profile": {
            "overall": bounded.get("overall") if isinstance(bounded.get("overall"), dict) else {},
            "by_h1_alignment": _profile_rows(bounded.get("by_h1_alignment")),
            "by_side_h1_alignment": _profile_rows(bounded.get("by_side_h1_alignment")),
            "by_side_entry_location_24h": _profile_rows(bounded.get("by_side_entry_location_24h")),
            "by_session_jst": _profile_rows(bounded.get("by_session_jst")),
        },
        "excluded_tail_profile": {
            "by_hold_bucket": _profile_rows(excluded.get("by_hold_bucket")),
            "by_close_reason": _profile_rows(excluded.get("by_close_reason")),
        },
        "position_building": {
            "basis": building.get("basis"),
            "overall": _small_dict(
                building.get("overall"),
                (
                    "clusters",
                    "multi_entry_clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "max_entries",
                    "adverse_adds",
                    "pyramid_adds",
                    "avg_adverse_add_pips",
                ),
            ),
            "bounded_lt_12h_excluding_margin_closeout": _small_dict(
                building.get("bounded_lt_12h_excluding_margin_closeout"),
                (
                    "clusters",
                    "multi_entry_clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "max_entries",
                    "adverse_adds",
                    "pyramid_adds",
                    "avg_adverse_add_pips",
                ),
            ),
            "adverse_adds": _small_dict(
                building.get("adverse_adds"),
                (
                    "clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "max_entries",
                    "adverse_adds",
                    "avg_adverse_add_pips",
                ),
            ),
            "bounded_by_build_type": _position_building_rows(building.get("bounded_by_build_type")),
            "largest_adverse_add_winners": _position_building_examples(
                ((building.get("examples") or {}).get("largest_adverse_add_winners"))
                if isinstance(building.get("examples"), dict)
                else None
            ),
            "contract": _small_dict(
                building.get("contract"),
                (
                    "advisory_only",
                    "nanpin_is_not_live_permission",
                    "requires_current_basket_risk_validation",
                    "forbidden_to_use_for_unbounded_martingale",
                ),
            ),
        },
        "contract": _small_dict(
            payload.get("contract"),
            (
                "advisory_only",
                "may_gate_use_of_operator_precedent_as_aggression_reason",
                "does_not_override_current_risk_geometry",
                "does_not_grant_live_permission",
            ),
        ),
        "warnings": list(payload.get("warnings") or [])[:5],
        "blockers": list(payload.get("blockers") or [])[:5],
    }


def _position_building_rows(rows: object, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "bucket",
                    "clusters",
                    "multi_entry_clusters",
                    "entries",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "median_entries",
                    "max_entries",
                    "adverse_adds",
                    "pyramid_adds",
                    "avg_adverse_add_pips",
                ),
            )
        )
    return out


def _position_building_examples(rows: object, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "cluster_id",
                    "side",
                    "build_type",
                    "entries",
                    "trade_ids",
                    "session_jst",
                    "hold_hours",
                    "realized_pl",
                    "initial_price",
                    "final_weighted_avg",
                    "adverse_add_count",
                    "pyramid_add_count",
                    "close_reasons",
                ),
            )
        )
    return out


def _profile_rows(rows: object, *, limit: int = 8) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rows[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            _small_dict(
                item,
                (
                    "bucket",
                    "trades",
                    "net_jpy",
                    "win_rate",
                    "expectancy_jpy",
                    "median_hold_hours",
                    "avg_h1_adx",
                    "median_entry_price_percentile_24h",
                ),
            )
        )
    return out


def _predictive_limits_packet(payload: dict[str, Any] | None, *, pairs: set[str]) -> dict[str, Any]:
    if not payload:
        return {"evidence_ref": "predictive:limits", "status": "missing", "orders_count": 0}
    orders_in = [item for item in payload.get("orders", []) or [] if isinstance(item, dict)]
    relevant_pairs = set(pairs)

    def priority(item: dict[str, Any]) -> tuple[int, int]:
        grade = str(item.get("grade") or "").upper()
        pair = str(item.get("pair") or "")
        return (0 if grade == "A" else 1, 0 if pair in relevant_pairs else 1)

    selected = sorted(orders_in, key=priority)[:12]
    return {
        "evidence_ref": "predictive:limits",
        "status": "DRY_RUN" if payload.get("dry_run", True) else "SENT_OR_ATTEMPTED",
        "generated_at_utc": payload.get("generated_at_utc"),
        "dry_run": payload.get("dry_run"),
        "orders_count": len(orders_in),
        "orders": [
            {
                "evidence_ref": f"predictive:limit:{item.get('pair')}:{item.get('side')}",
                "pair": item.get("pair"),
                "side": item.get("side"),
                "grade": item.get("grade"),
                "limit_price": item.get("limit_price"),
                "take_profit_price": item.get("take_profit_price"),
                "units": item.get("units"),
                "source": item.get("source"),
                "gtd_utc": item.get("gtd_utc"),
                "rationale": item.get("rationale"),
            }
            for item in selected
        ],
    }


def _block_issues(items: object) -> list[str]:
    blockers: list[str] = []
    for item in items or []:
        if isinstance(item, dict) and item.get("severity") == "BLOCK":
            blockers.append(str(item.get("message") or item.get("code") or "block"))
    return blockers


def _small_dict(payload: object, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {key: payload.get(key) for key in keys if key in payload}


def _news_packet(
    news_items: dict[str, Any] | None,
    news_health: dict[str, Any] | None,
    *,
    pairs: set[str] | tuple[str, ...],
    currencies: set[str] | tuple[str, ...],
) -> dict[str, Any]:
    pair_set = {str(pair) for pair in pairs if str(pair)}
    currency_set = {str(currency) for currency in currencies if str(currency)}
    health = {
        "evidence_ref": "news:health",
        "status": news_health.get("status") if isinstance(news_health, dict) else "missing",
        "generated_at_utc": news_health.get("generated_at_utc") if isinstance(news_health, dict) else None,
        "market_window": news_health.get("market_window") if isinstance(news_health, dict) else None,
        "issues": list(news_health.get("issues", [])[:8]) if isinstance(news_health, dict) else ["MISSING_NEWS_HEALTH"],
    }
    if not isinstance(news_items, dict):
        return {
            "evidence_ref": "news:items",
            "status": "missing",
            "generated_at_utc": None,
            "health": health,
            "items_count": 0,
            "relevant_items": [],
        }
    rows = [item for item in news_items.get("items", []) or [] if isinstance(item, dict)]
    relevant = [item for item in rows if _news_item_relevant(item, pairs=pair_set, currencies=currency_set)]
    selected = (relevant or rows)[:12]
    return {
        "evidence_ref": "news:items",
        "status": "present",
        "generated_at_utc": news_items.get("generated_at_utc"),
        "health": health,
        "items_count": len(rows),
        "relevant_items_count": len(relevant),
        "relevant_items": [_news_item_packet(item) for item in selected],
    }


def _news_item_relevant(item: dict[str, Any], *, pairs: set[str], currencies: set[str]) -> bool:
    item_pairs = {str(pair) for pair in item.get("pairs", []) or [] if str(pair)}
    item_currencies = {str(currency) for currency in item.get("currencies", []) or [] if str(currency)}
    return bool(item_pairs & pairs or item_currencies & currencies)


def _news_item_packet(item: dict[str, Any]) -> dict[str, Any]:
    pairs = [str(pair) for pair in item.get("pairs", []) or [] if str(pair)]
    currencies = [str(currency) for currency in item.get("currencies", []) or [] if str(currency)]
    evidence_refs = ["news:items"]
    evidence_refs.extend(f"news:{pair}" for pair in pairs)
    evidence_refs.extend(f"news:{currency}" for currency in currencies)
    return {
        "evidence_refs": sorted(set(evidence_refs)),
        "source": item.get("source"),
        "published_at_utc": item.get("published_at_utc"),
        "title": item.get("title"),
        "pairs": pairs,
        "currencies": currencies,
        "topics": list(item.get("topics", [])[:8]) if isinstance(item.get("topics"), list) else [],
        "categories": list(item.get("categories", [])[:8]) if isinstance(item.get("categories"), list) else [],
        "link": item.get("link"),
    }


def _qr_trader_run_watchdog_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"available": False}
    guardian = payload.get("guardian_receipt") if isinstance(payload.get("guardian_receipt"), dict) else {}
    issues = guardian.get("issues") if isinstance(guardian.get("issues"), list) else []
    return {
        "available": True,
        "generated_at_utc": payload.get("generated_at_utc"),
        "status": payload.get("status"),
        "runtime_status": payload.get("runtime_status"),
        "issue_status": payload.get("issue_status"),
        "overall_status": payload.get("overall_status"),
        "severity": payload.get("severity"),
        "last_trader_run_at": payload.get("last_trader_run_at"),
        "last_trader_run_source": payload.get("last_trader_run_source"),
        "last_trader_run_path": payload.get("last_trader_run_path"),
        "minutes_since_last_run": payload.get("minutes_since_last_run"),
        "guardian_receipt_issues": [
            {
                "code": item.get("code"),
                "severity": item.get("severity"),
                "message": item.get("message"),
                "receipt_event_id": item.get("receipt_event_id"),
                "receipt_action": item.get("receipt_action"),
                "receipt_lifecycle": item.get("receipt_lifecycle"),
                "receipt_identity": item.get("receipt_identity"),
                "receipt_source_paths": item.get("receipt_source_paths"),
                "emergency_or_margin_risk": item.get("emergency_or_margin_risk"),
                "consumed_by_trader": item.get("consumed_by_trader"),
                "normal_routing_allowed": item.get("normal_routing_allowed"),
            }
            for item in issues
            if isinstance(item, dict)
        ],
    }


def _lane_forecast_packet(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return _small_dict(
        metadata,
        (
            "forecast_direction",
            "forecast_confidence",
            "forecast_target_price",
            "forecast_invalidation_price",
            "forecast_horizon_min",
            "forecast_rationale",
        ),
    )


def _has_pending_entry_order(packet: dict[str, Any]) -> bool:
    return bool(_pending_entry_order_ids(packet))


def _pending_entry_order_ids(packet: dict[str, Any]) -> list[str]:
    return [
        str(order.get("order_id") or "")
        for order in _pending_entry_orders(packet)
        if str(order.get("order_id") or "")
    ]


def _pending_entry_orders(packet: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = packet.get("broker_snapshot", {})
    orders: list[dict[str, Any]] = []
    for order in snapshot.get("pending_orders", []) or []:
        if _operator_managed_owner(order.get("owner")):
            continue
        if order.get("trade_id"):
            continue
        order_type = str(order.get("order_type") or "").upper()
        order_id = str(order.get("order_id") or "")
        if order_id and order_type in {"LIMIT", "STOP", "MARKET_IF_TOUCHED", "MARKET_IF_TOUCHED_ORDER"}:
            orders.append(order)
    return orders


def _cancel_pending_visible_current_theses(
    packet: dict[str, Any],
    cancel_order_ids: tuple[str, ...],
    *,
    exempt_order_ids: set[str] | None = None,
) -> list[str]:
    cancel_set = {str(order_id) for order_id in cancel_order_ids if str(order_id)}
    cancel_set -= set(exempt_order_ids or set())
    if not cancel_set:
        return []
    out: list[str] = []
    for order in _pending_entry_orders(packet):
        order_id = str(order.get("order_id") or "")
        if order_id not in cancel_set:
            continue
        pair = str(order.get("pair") or "").strip()
        side = str(order.get("side") or _side_from_order_units(order.get("units")) or "").upper()
        if not pair or not side:
            continue
        matching_lanes = [
            str(lane.get("lane_id") or "")
            for lane in packet.get("lanes", []) or []
            if isinstance(lane, dict)
            and str(lane.get("pair") or "").strip() == pair
            and str(lane.get("direction") or "").upper() == side
            and str(lane.get("lane_id") or "")
        ]
        if matching_lanes:
            out.append(f"{order_id} {pair} {side} still has current thesis {matching_lanes[0]}")
    return out


def _trade_exposure_blockers(packet: dict[str, Any]) -> list[str]:
    snapshot = packet.get("broker_snapshot", {})
    blockers: list[str] = []
    sl_free_active = _trader_sl_repair_disabled()
    for position in snapshot.get("position_summaries", []) or []:
        owner = str(position.get("owner") or "")
        if _operator_managed_owner(owner):
            continue
        # SL-free regime: trader-owned SL=None is intentional, and missing
        # broker TP is a no-broker-TP runner unless repair is explicitly
        # enabled. Exposure still reaches LiveOrderGateway for margin,
        # hedging, and portfolio validation.
        sl_ok = position.get("stop_loss") is not None or sl_free_active
        tp_ok = position.get("take_profit") is not None or (
            owner == "trader" and sl_free_active and not _missing_tp_repair_enabled()
        )
        if owner == "trader" and tp_ok and sl_ok:
            continue
        blockers.append(
            f"non-layerable position {position.get('pair')} {position.get('side')} id={position.get('trade_id')}"
        )
    return blockers


def _operator_managed_owner(owner: Any) -> bool:
    normalized = str(owner or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {"manual", "unknown", "operator_manual"}


def _trader_exposure_present(packet: dict[str, Any]) -> bool:
    snapshot = packet.get("broker_snapshot", {})
    for position in snapshot.get("position_summaries", []) or []:
        if str(position.get("owner") or "") == "trader":
            return True
    return _has_pending_entry_order(packet)


def _target_requires_entry(packet: dict[str, Any]) -> bool:
    target = packet.get("daily_target", {})
    remaining = target.get("remaining_target_jpy")
    try:
        return float(remaining or 0.0) > 0.0 and target.get("status") != "TARGET_REACHED_PROTECT"
    except (TypeError, ValueError):
        return False


def _tp_rebalance_sidecar_reasons(packet: dict[str, Any]) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    tp_rebalance = sidecars.get("tp_rebalance")
    if not isinstance(tp_rebalance, dict) or not tp_rebalance.get("required"):
        return []
    return [
        str(reason)
        for reason in (tp_rebalance.get("reasons") or [])
        if str(reason).strip()
    ]


def _entry_thesis_sidecar_reasons(packet: dict[str, Any]) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    blockers = sidecars.get("entry_thesis_blockers")
    if not isinstance(blockers, list):
        return []
    reasons: list[str] = []
    for item in blockers:
        if not isinstance(item, dict):
            continue
        trade_id = str(item.get("trade_id") or "").strip()
        pair = str(item.get("pair") or "").strip()
        side = str(item.get("side") or "").strip()
        reason = str(item.get("reason") or "original entry thesis is not machine-verifiable").strip()
        label = " ".join(part for part in (pair, side, f"id={trade_id}" if trade_id else "") if part)
        reasons.append(f"{label}: {reason}" if label else reason)
    return reasons


def _position_close_sidecar_reasons(
    packet: dict[str, Any],
    *,
    blocking_only: bool = True,
) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    recommendations = sidecars.get("position_close_recommendations")
    if not isinstance(recommendations, list):
        return []
    reasons: list[str] = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        blocks_non_close_action = _position_close_recommendation_blocks_non_close_action(packet, item)
        if blocking_only and not blocks_non_close_action:
            continue
        if not blocking_only and blocks_non_close_action:
            continue
        trade_id = str(item.get("trade_id") or "").strip()
        pair = str(item.get("pair") or "").strip()
        side = str(item.get("side") or "").strip()
        source = str(item.get("source") or "position_sidecar").strip()
        verdict = str(item.get("verdict") or "RECOMMEND_CLOSE").strip()
        reason = str(item.get("reason") or "position recovery edge is broken").strip()
        label = " ".join(part for part in (pair, side, f"id={trade_id}" if trade_id else "") if part)
        prefix = f"{source} {verdict}"
        reasons.append(f"{prefix} {label}: {reason}" if label else f"{prefix}: {reason}")
    return reasons


def _authorized_single_position_close_candidate(
    packet: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Select one exact current trader-owned CLOSE target, or fail closed.

    The deterministic draft may translate a sidecar into CLOSE only when the
    packet already marks it as preempting non-CLOSE actions and the verifier's
    hard/explicit Gate B path is currently available. Recommendations are
    grouped by trade id so multiple independent refs may support one close,
    while two close targets can never become a deterministic mass close.
    """

    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return None, []
    recommendations = sidecars.get("position_close_recommendations")
    if not isinstance(recommendations, list):
        return None, []

    positions = _trader_position_lookup(packet)
    allowed_refs = {
        str(ref)
        for ref in packet.get("allowed_evidence_refs", []) or []
        if str(ref)
    }
    explicit_operator_gate_b = _operator_close_gate_authorized()
    eligible_by_trade_id: dict[str, list[dict[str, Any]]] = {}
    selection_blockers: list[str] = []

    for rec in recommendations:
        if not isinstance(rec, dict) or rec.get("blocks_non_close_actions") is not True:
            continue
        trade_id = str(rec.get("trade_id") or "").strip()
        position = positions.get(trade_id)
        if position is None:
            selection_blockers.append(
                f"CLOSE_BASELINE_POSITION_MISMATCH id={trade_id or 'missing'}: target is not a current trader-owned trade"
            )
            continue
        pair = str(position.get("pair") or "").strip()
        side = str(position.get("side") or "").strip().upper()
        if not pair or not side or str(rec.get("pair") or "").strip() != pair or str(
            rec.get("side") or ""
        ).strip().upper() != side:
            selection_blockers.append(
                f"CLOSE_BASELINE_IDENTITY_MISMATCH id={trade_id}: sidecar pair/side does not match broker truth"
            )
            continue

        standing_authorized = bool(rec.get("gate_b_standing_authorized"))
        if not standing_authorized:
            invalidated, _reason, standing_authorized = _close_thesis_invalidation(
                packet,
                pair,
                side,
                trade_id=trade_id,
                decision=None,
            )
            standing_authorized = bool(invalidated and standing_authorized)
        if not standing_authorized and not explicit_operator_gate_b:
            selection_blockers.append(
                f"CLOSE_BASELINE_GATE_B_MISSING id={trade_id}: blocking sidecar has neither standing hard nor explicit Gate B"
            )
            continue

        evidence_ref = str(rec.get("evidence_ref") or "").strip()
        if not evidence_ref or evidence_ref not in allowed_refs:
            selection_blockers.append(
                f"CLOSE_BASELINE_EVIDENCE_REF_MISSING id={trade_id}: blocking sidecar lacks a current allowed evidence ref"
            )
            continue

        eligible_by_trade_id.setdefault(trade_id, []).append(rec)

    if selection_blockers:
        return None, selection_blockers
    if not eligible_by_trade_id:
        return None, selection_blockers
    if len(eligible_by_trade_id) != 1:
        ids = ", ".join(sorted(eligible_by_trade_id))
        selection_blockers.append(
            "CLOSE_BASELINE_SINGLE_TRADE_REQUIRED: multiple authorized current close targets "
            f"({ids}); require one separately refreshed CLOSE receipt per trade"
        )
        return None, selection_blockers

    trade_id, matched_recommendations = next(iter(eligible_by_trade_id.items()))
    position = positions[trade_id]
    return {
        "trade_id": trade_id,
        "pair": str(position.get("pair") or ""),
        "side": str(position.get("side") or "").upper(),
        "position": position,
        "recommendations": matched_recommendations,
        "gate_b_standing_authorized": any(
            bool(rec.get("gate_b_standing_authorized"))
            or _close_thesis_invalidation(
                packet,
                str(position.get("pair") or ""),
                str(position.get("side") or ""),
                trade_id=trade_id,
                decision=None,
            )[2]
            for rec in matched_recommendations
        ),
        "gate_b_explicit_operator_authorized": explicit_operator_gate_b,
    }, selection_blockers


def _soft_nonblocking_close_advisory_reasons(
    packet: dict[str, Any],
    trade_ids: list[str] | tuple[str, ...],
) -> list[str]:
    sidecars = packet.get("protection_sidecars")
    if not isinstance(sidecars, dict):
        return []
    recommendations = sidecars.get("position_close_recommendations")
    if not isinstance(recommendations, list):
        return []
    selected = {str(trade_id) for trade_id in trade_ids if str(trade_id)}
    reasons: list[str] = []
    for item in recommendations:
        if not isinstance(item, dict):
            continue
        trade_id = str(item.get("trade_id") or "").strip()
        if selected and trade_id not in selected:
            continue
        if _position_close_recommendation_blocks_non_close_action(packet, item):
            continue
        pair = str(item.get("pair") or "").strip()
        side = str(item.get("side") or "").strip()
        source = str(item.get("source") or "position_sidecar").strip()
        verdict = str(item.get("verdict") or "RECOMMEND_CLOSE").strip()
        reason = str(
            item.get("non_blocking_reason")
            or item.get("reason")
            or "soft close evidence does not block non-CLOSE actions"
        ).strip()
        label = " ".join(part for part in (pair, side, f"id={trade_id}" if trade_id else "") if part)
        reasons.append(f"{source} {verdict} {label}: {reason}" if label else f"{source} {verdict}: {reason}")
    return reasons


def _position_close_recommendation_blocks_non_close_action(
    packet: dict[str, Any],
    rec: dict[str, Any],
) -> bool:
    """Whether a close sidecar must preempt TRADE/WAIT/PROTECT.

    Hard Gate A, or soft Gate A with explicit operator Gate B, requires a CLOSE
    receipt first. Soft-only sidecars remain usable Gate A evidence if the
    trader chooses CLOSE, but they do not freeze TP-managed exposure and block
    fresh all-horizon entries while authorization is absent.
    """
    if _sidecar_hold_support_conflict(packet, rec):
        return False
    if bool(rec.get("gate_b_standing_authorized")):
        return True
    trade_id = str(rec.get("trade_id") or "")
    if trade_id:
        pos = _trader_position_lookup(packet).get(trade_id)
        if pos is not None:
            pair = str(pos.get("pair") or rec.get("pair") or "")
            side = str(pos.get("side") or rec.get("side") or "")
            invalidated, _reason, standing_authorized = _close_thesis_invalidation(
                packet,
                pair,
                side,
                trade_id=trade_id,
                decision=None,
            )
            if invalidated and standing_authorized:
                return True
    return _operator_close_gate_authorized()


def _append_position_close_required_issue(
    issues: list[VerificationIssue],
    *,
    packet: dict[str, Any],
    action: str,
    reasons: list[str],
) -> None:
    reason_text = "; ".join(reasons[:3])
    if _operator_close_gate_authorized() or _standing_close_authorization_available(packet):
        issues.append(
            VerificationIssue(
                "POSITION_CLOSE_REQUIRED",
                f"{action} rejected while fresh position sidecar Gate A close evidence exists and close authorization is available. "
                "Emit action=CLOSE for the named trade(s) first; after the accepted CLOSE receipt is handled, "
                "end that cycle and refresh broker truth / intents on the next scheduled cycle before any separate TRADE receipt: "
                f"{reason_text}",
            )
        )
        return
    issues.append(
        VerificationIssue(
            "CLOSE_OPERATOR_AUTH_REQUIRED",
            f"{action} rejected while fresh position sidecar Gate A close evidence exists, but Gate B "
            "operator authorization is missing for the softer close evidence. This is not a recovery-confidence hold; require "
            "QR_OPERATOR_CLOSE_OVERRIDE=1 in the operator shell or a fresh data/.operator_close_token "
            f"before a loss-side CLOSE can be verified: {reason_text}",
        )
    )


def _standing_close_authorization_available(packet: dict[str, Any]) -> bool:
    position_by_tid = _trader_position_lookup(packet)
    sidecars = packet.get("protection_sidecars")
    recs = sidecars.get("position_close_recommendations") if isinstance(sidecars, dict) else None
    if not isinstance(recs, list):
        return False
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        trade_id = str(rec.get("trade_id") or "")
        pos = position_by_tid.get(trade_id)
        if pos is None:
            continue
        pair = str(pos.get("pair") or rec.get("pair") or "")
        side = str(pos.get("side") or rec.get("side") or "")
        invalidated, _reason, standing_authorized = _close_thesis_invalidation(
            packet,
            pair,
            side,
            trade_id=trade_id,
            decision=None,
        )
        if invalidated and standing_authorized:
            return True
    return False


def _tradeable_live_ready_lanes(packet: dict[str, Any]) -> list[str]:
    lanes: list[str] = []
    for lane in packet.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        if lane.get("status") != "LIVE_READY":
            continue
        if lane.get("risk_blockers") or lane.get("strategy_blockers") or lane.get("live_blockers"):
            continue
        lane_id = str(lane.get("lane_id") or "")
        if lane_id:
            lanes.append(lane_id)
    return lanes


def _pace_trade_lanes(packet: dict[str, Any], tradeable_lanes: list[str]) -> list[str]:
    """Lanes strong enough for rolling-policy pace pressure.

    +5% is a pace marker, not a forced churn target. When the rolling policy is
    behind, the verifier may require action only for A/S target-path lanes or
    explicitly recommended attack lanes. B/C LIVE_READY lanes remain tradable if
    selected for independent edge, but they are not forced solely to satisfy the
    daily +5% marker.
    """

    if not _rolling_30d_policy_active(packet):
        return tradeable_lanes
    attack = set(_attack_recommended_tradeable_lane_ids(packet, tradeable_lanes))
    lane_by_id = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", []) or []
        if isinstance(lane, dict)
    }
    forced: list[str] = []
    for lane_id in tradeable_lanes:
        lane = lane_by_id.get(lane_id) or {}
        if lane_id in attack or _lane_target_grade_at_least_a(lane):
            forced.append(lane_id)
    return forced


def _rolling_30d_policy_active(packet: dict[str, Any]) -> bool:
    target = packet.get("daily_target")
    if not isinstance(target, dict):
        return False
    return str(target.get("rolling_30d_policy") or "").upper() == "ROLLING_30D_4X"


def _lane_target_grade_at_least_a(lane: dict[str, Any]) -> bool:
    target_path = lane.get("target_path") if isinstance(lane.get("target_path"), dict) else {}
    grade = str(
        target_path.get("conviction_grade")
        or target_path.get("grade")
        or target_path.get("allocation_band")
        or ""
    ).strip().upper().replace("_", "").replace(" ", "")
    return grade in {"A", "S"}


def _attack_recommended_tradeable_lane_ids(
    packet: dict[str, Any],
    tradeable_lanes: list[str] | None = None,
) -> list[str]:
    advice = packet.get("ai_attack_advice")
    if not isinstance(advice, dict):
        return []
    current = set(tradeable_lanes if tradeable_lanes is not None else _tradeable_live_ready_lanes(packet))
    lane_ids: list[str] = []
    for raw_lane_id in advice.get("recommended_now_lane_ids", []) or []:
        lane_id = str(raw_lane_id or "")
        learning_allowed, _reason = _learning_audit_allows_influenced_lane(packet, lane_id)
        if lane_id and lane_id in current and lane_id not in lane_ids and learning_allowed:
            lane_ids.append(lane_id)
    return lane_ids


def _attack_lane_learning_influenced(packet: dict[str, Any], lane_id: str) -> bool:
    advice = packet.get("ai_attack_advice")
    if not isinstance(advice, dict):
        return False
    influenced = {str(item) for item in advice.get("learning_influenced_lane_ids", []) or []}
    if lane_id in influenced:
        return True
    for lane in advice.get("recommended_lane_learning", []) or []:
        if not isinstance(lane, dict) or str(lane.get("lane_id") or "") != lane_id:
            continue
        return bool([item for item in (lane.get("learning_influences") or []) if str(item).strip()])
    return False


def _learning_audit_lane_ids(packet: dict[str, Any]) -> set[str]:
    audit = packet.get("learning_audit")
    if not isinstance(audit, dict):
        return set()
    influence = audit.get("learning_influence") if isinstance(audit.get("learning_influence"), dict) else {}
    out: set[str] = set()
    for lane in influence.get("lanes", []) or []:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("lane_id") or "")
        if lane_id:
            out.add(lane_id)
    return out


def _learning_audit_allows_influenced_lane(packet: dict[str, Any], lane_id: str) -> tuple[bool, str]:
    if not _attack_lane_learning_influenced(packet, lane_id):
        return True, ""
    audit = packet.get("learning_audit")
    if not isinstance(audit, dict):
        return False, "learning audit packet is missing"
    status = str(audit.get("status") or "missing")
    if status in {"", "missing"}:
        return False, "learning audit is missing"
    if status == "LEARNING_AUDIT_BLOCKED":
        blockers = [str(item) for item in (audit.get("blockers") or []) if str(item).strip()]
        suffix = f": {'; '.join(blockers[:3])}" if blockers else ""
        return False, f"learning audit is blocked{suffix}"
    if lane_id not in _learning_audit_lane_ids(packet):
        return False, f"learning audit does not cover influenced lane {lane_id}"
    return True, ""


def _all_tradeable_lanes_blocked_by_learning_audit(packet: dict[str, Any], lane_ids: list[str]) -> bool:
    if not lane_ids:
        return False
    for lane_id in lane_ids:
        if not _attack_lane_learning_influenced(packet, lane_id):
            return False
        allowed, _reason = _learning_audit_allows_influenced_lane(packet, lane_id)
        if allowed:
            return False
    return True


def _learning_audit_trade_issues(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    refs = set(evidence_refs)
    for lane_id in selected_lane_ids:
        if not _attack_lane_learning_influenced(packet, lane_id):
            continue
        allowed, reason = _learning_audit_allows_influenced_lane(packet, lane_id)
        if not allowed:
            code = "LEARNING_AUDIT_BLOCKED"
            audit = packet.get("learning_audit")
            status = str(audit.get("status") or "") if isinstance(audit, dict) else ""
            if status in {"", "missing"}:
                code = "LEARNING_AUDIT_REQUIRED"
            elif "does not cover" in reason:
                code = "LEARNING_AUDIT_STALE"
            issues.append(
                VerificationIssue(
                    code,
                    f"TRADE rejected for learning-influenced lane {lane_id}: {reason}",
                )
            )
            continue
        if "learning:audit" not in refs:
            issues.append(
                VerificationIssue(
                    "LEARNING_AUDIT_EVIDENCE_MISSING",
                    f"TRADE selecting learning-influenced lane {lane_id} must cite learning:audit",
                )
            )
        lane_ref = f"learning:lane:{lane_id}"
        if lane_ref not in refs:
            issues.append(
                VerificationIssue(
                    "LEARNING_LANE_EVIDENCE_MISSING",
                    f"TRADE selecting learning-influenced lane {lane_id} must cite {lane_ref}",
                )
            )
    return issues


def _manual_precedent_trade_issues(
    packet: dict[str, Any],
    selected_lane_ids: tuple[str, ...],
    evidence_refs: tuple[str, ...],
) -> list[VerificationIssue]:
    refs = set(evidence_refs)
    if OPERATOR_PRECEDENT_EVIDENCE_REF not in refs:
        return []
    issues: list[VerificationIssue] = []
    if MANUAL_MARKET_CONTEXT_EVIDENCE_REF not in refs:
        issues.append(
            VerificationIssue(
                "MANUAL_CONTEXT_EVIDENCE_MISSING",
                "TRADE citing the 2025 operator precedent must also cite "
                f"{MANUAL_MARKET_CONTEXT_EVIDENCE_REF} so the manual technical context is checked.",
            )
        )
    precedent = packet.get("operator_precedent")
    if not isinstance(precedent, dict) or precedent.get("status") == "missing":
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_PACKET_MISSING",
                "TRADE cites operator precedent but the decision packet has no readable operator precedent audit.",
            )
        )
        return issues
    claim = precedent.get("operator_claim") if isinstance(precedent.get("operator_claim"), dict) else {}
    if claim.get("verified") is not True:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_UNVERIFIED",
                "TRADE cites operator precedent but the manual-history claim is not verified in the audit packet.",
            )
        )
    manual_context = packet.get("manual_market_context")
    if not isinstance(manual_context, dict) or manual_context.get("status") == "missing":
        issues.append(
            VerificationIssue(
                "MANUAL_CONTEXT_PACKET_MISSING",
                "TRADE cites operator precedent but the decision packet has no readable manual market-context audit.",
            )
        )
    elif str(manual_context.get("status") or "") != "MANUAL_MARKET_CONTEXT_PASS":
        issues.append(
            VerificationIssue(
                "MANUAL_CONTEXT_NOT_PASSING",
                "TRADE cites operator precedent but manual-market-context audit is not passing: "
                f"{manual_context.get('status')}",
            )
        )

    runtime = precedent.get("runtime_alignment") if isinstance(precedent.get("runtime_alignment"), dict) else {}
    aligned_lane_ids = {
        str(item.get("lane_id") or "")
        for item in (runtime.get("aligned_lanes") or [])
        if isinstance(item, dict) and str(item.get("lane_id") or "").strip()
    }
    if not selected_lane_ids:
        return issues
    if not aligned_lane_ids:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_NO_CURRENT_ALIGNMENT",
                "TRADE cites the 2025 operator precedent, but the current operator-precedent audit has no "
                "LIVE_READY lane aligned to the generalized discretionary trade-shape precedent. Cite current "
                "deterministic edge instead of using the manual precedent as an aggression reason.",
            )
        )
        return issues
    selected_aligned = [lane_id for lane_id in selected_lane_ids if lane_id in aligned_lane_ids]
    if not selected_aligned:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_SELECTED_LANE_NOT_ALIGNED",
                "TRADE cites the 2025 operator precedent, but none of the selected lane(s) are aligned to the "
                "generalized discretionary trade-shape precedent: "
                f"selected={', '.join(selected_lane_ids)} aligned={', '.join(sorted(aligned_lane_ids))}. "
                "Use current forecast/risk/matrix evidence for this trade instead.",
            )
        )
        return issues
    lane_by_id = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", []) or []
        if isinstance(lane, dict) and str(lane.get("lane_id") or "").strip()
    }
    selected_with_move_adds = []
    for lane_id in selected_aligned:
        lane = lane_by_id.get(lane_id) or {}
        building = lane.get("position_building") if isinstance(lane.get("position_building"), dict) else {}
        if str(building.get("same_pair_add_type") or "").upper() == "PYRAMID_WITH_MOVE":
            selected_with_move_adds.append(lane_id)
    if selected_with_move_adds:
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_POSITION_BUILDING_CONFLICT",
                "TRADE cites the 2025 operator precedent, but the selected same-pair add is "
                "PYRAMID_WITH_MOVE. The bounded manual replay supports only selective adverse "
                "retest/add behavior as precedent; cite current deterministic edge instead: "
                f"{', '.join(selected_with_move_adds)}",
            )
        )
    manual_alignment = (
        runtime.get("manual_context_alignment")
        if isinstance(runtime.get("manual_context_alignment"), dict)
        else {}
    )
    conflicting_rows = [
        row
        for row in (manual_alignment.get("conflicting_lanes") or [])
        if isinstance(row, dict) and str(row.get("lane_id") or "").strip()
    ]
    conflicting_lane_ids = {str(row.get("lane_id") or "") for row in conflicting_rows}
    selected_conflicting = [lane_id for lane_id in selected_aligned if lane_id in conflicting_lane_ids]
    if selected_conflicting:
        conflict_details = [
            f"{row.get('lane_id')}:{','.join(str(bucket) for bucket in (row.get('conflicting_buckets') or []))}"
            for row in conflicting_rows
            if str(row.get("lane_id") or "") in selected_conflicting
        ]
        issues.append(
            VerificationIssue(
                "OPERATOR_PRECEDENT_TECHNICAL_CONTEXT_CONFLICT",
                "TRADE cites the 2025 operator precedent, but the selected lane conflicts with the bounded manual "
                "technical replay context. Cite current deterministic edge instead, or choose a lane whose H1/M5/"
                f"24h-location context matches the manual precedent: {'; '.join(conflict_details)}",
            )
        )
    return issues


def _self_improvement_trade_blockers(
    packet: dict[str, Any],
    *,
    decision_generated_at_utc: str | None = None,
    include_decision_history_stale: bool = True,
    resolved_pending_cancel_order_ids: Sequence[str] = (),
    selected_lane_ids: Sequence[str] = (),
) -> list[str]:
    audit = packet.get("self_improvement_audit")
    if not isinstance(audit, dict):
        return []
    audit_generated_at = _parse_utc(audit.get("generated_at_utc"))
    receipt_generated_at = (
        _parse_utc(decision_generated_at_utc) if decision_generated_at_utc else None
    )
    out: list[str] = []
    blockers = (
        list(audit.get("p0_blockers", []) or [])
        + list(audit.get("profitability_blockers", []) or [])
        + list(audit.get("new_risk_blockers", []) or [])
    )
    p0_repair_selected = _all_selected_lanes_are_self_improvement_profitability_repair(
        packet,
        selected_lane_ids,
    )
    forecast_repair_selected = _all_selected_lanes_are_forecast_adverse_path_repair(
        packet,
        selected_lane_ids,
    )
    for blocker in blockers:
        if not isinstance(blocker, dict):
            continue
        code = str(blocker.get("code") or "SELF_IMPROVEMENT_P0")
        if code == FORECAST_ADVERSE_PATH_BLOCKER_CODE and forecast_repair_selected:
            continue
        if p0_code_exempted_by_tp_harvest_repair(
            code,
            p0_repair_selected=p0_repair_selected,
        ):
            continue
        if code == "LATEST_GPT_DECISION_STALE" and not include_decision_history_stale:
            continue
        if (
            code == "LATEST_GPT_DECISION_STALE"
            and receipt_generated_at is not None
            and audit_generated_at is not None
            and receipt_generated_at > audit_generated_at
        ):
            # The audit predates the receipt under verification, so its
            # stale-decision verdict is about an older receipt. Writing one
            # current receipt is exactly the repair that finding demands;
            # rejecting the repair receipt with its own staleness streak
            # would re-create the deadlock the streak exemption documents.
            continue
        if _self_improvement_non_trade_blocker(code, blocker):
            continue
        if code == "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED":
            required_cancel_ids = {
                str(order_id or "").strip()
                for order_id in blocker.get("cancel_review_order_ids", []) or []
                if str(order_id or "").strip()
            }
            receipt_cancel_ids = {
                str(order_id or "").strip()
                for order_id in resolved_pending_cancel_order_ids
                if str(order_id or "").strip()
            }
            if required_cancel_ids and required_cancel_ids <= receipt_cancel_ids:
                continue
        layer = str(blocker.get("layer") or "").strip()
        message = str(blocker.get("message") or "").strip()
        streak = blocker.get("current_streak")
        pf = blocker.get("profit_factor")
        expectancy = blocker.get("expectancy_jpy")
        avg_loss = blocker.get("avg_loss_jpy_abs")
        avg_win = blocker.get("avg_win_jpy")
        count = blocker.get("count")
        details = []
        if layer:
            details.append(f"layer={layer}")
        if streak is not None:
            details.append(f"streak={streak}")
        if count is not None:
            details.append(f"count={count}")
        if pf is not None:
            details.append(f"PF={pf}")
        if expectancy is not None:
            details.append(f"expectancy={expectancy}")
        if avg_loss is not None and avg_win is not None:
            details.append(f"avg_loss={avg_loss} vs avg_win={avg_win}")
        suffix = f" ({', '.join(details)})" if details else ""
        out.append(f"{code}{suffix}: {message or 'self-improvement P0 blocks new risk'}")
    return out


def _all_selected_lanes_are_self_improvement_profitability_repair(
    packet: dict[str, Any],
    selected_lane_ids: Sequence[str],
) -> bool:
    lane_ids = tuple(dict.fromkeys(str(item) for item in selected_lane_ids if str(item)))
    if not lane_ids:
        return False
    worst_segment = profitability_p0_worst_segment(packet.get("self_improvement_audit"))
    lane_map = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", []) or []
        if isinstance(lane, dict) and str(lane.get("lane_id") or "")
    }
    for lane_id in lane_ids:
        lane = lane_map.get(lane_id)
        if not isinstance(lane, dict):
            return False
        repair = lane.get("self_improvement") if isinstance(lane.get("self_improvement"), dict) else {}
        if repair.get("self_improvement_p0_repair_live_ready") is not True:
            return False
        repair_mode = str(repair.get("self_improvement_p0_repair_mode") or "")
        if repair_mode == "PREDICTIVE_SCOUT_FORWARD_EVIDENCE":
            if not _lane_is_predictive_scout(lane):
                return False
            if str(lane.get("order_type") or "").upper() != "LIMIT":
                return False
            if (_optional_int(lane.get("units")) or 0) < MIN_PRODUCTION_LOT_UNITS:
                return False
            if intent_matches_profitability_worst_segment(lane, worst_segment):
                return False
            continue
        if repair_mode != "TP_HARVEST_REPAIR":
            return False
        if not oanda_firepower_repair_current_risk_reaches_minimum(repair):
            return False
        if intent_matches_profitability_worst_segment(lane, worst_segment):
            return False
    return True


def _all_selected_lanes_are_forecast_adverse_path_repair(
    packet: dict[str, Any],
    selected_lane_ids: Sequence[str],
) -> bool:
    lane_ids = tuple(dict.fromkeys(str(item) for item in selected_lane_ids if str(item)))
    if not lane_ids:
        return False
    lane_map = {
        str(lane.get("lane_id") or ""): lane
        for lane in packet.get("lanes", []) or []
        if isinstance(lane, dict) and str(lane.get("lane_id") or "")
    }
    for lane_id in lane_ids:
        lane = lane_map.get(lane_id)
        if not isinstance(lane, dict):
            return False
        repair = lane.get("self_improvement") if isinstance(lane.get("self_improvement"), dict) else {}
        if not forecast_adverse_path_exempted_by_tp_harvest_repair(lane, repair):
            return False
    return True


def _self_improvement_pending_cancel_review_reasons(packet: dict[str, Any]) -> list[str]:
    audit = packet.get("self_improvement_audit")
    if not isinstance(audit, dict):
        return []
    reasons: list[str] = []
    for blocker in audit.get("p0_blockers", []) or []:
        if not isinstance(blocker, dict):
            continue
        if str(blocker.get("code") or "") != "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED":
            continue
        ids = [
            str(order_id)
            for order_id in blocker.get("cancel_review_order_ids", []) or []
            if str(order_id)
        ]
        details: list[str] = []
        if ids:
            details.append("ids=" + ",".join(ids))
        message = str(blocker.get("message") or "").strip()
        if message:
            details.append(message)
        reasons.append("; ".join(details) or "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED")
    return reasons


def _self_improvement_pending_cancel_review_order_ids(packet: dict[str, Any]) -> set[str]:
    audit = packet.get("self_improvement_audit")
    if not isinstance(audit, dict):
        return set()
    order_ids: set[str] = set()
    for blocker in audit.get("p0_blockers", []) or []:
        if not isinstance(blocker, dict):
            continue
        if str(blocker.get("code") or "") != "PENDING_ENTRY_CANCEL_REVIEW_REQUIRED":
            continue
        for order_id in blocker.get("cancel_review_order_ids", []) or []:
            text = str(order_id or "").strip()
            if text:
                order_ids.add(text)
    return order_ids


def _self_improvement_non_trade_blocker(code: str, blocker: dict[str, Any]) -> bool:
    if code not in SELF_IMPROVEMENT_NON_TRADE_BLOCKER_CODES:
        return False
    streak = _optional_int(blocker.get("current_streak"))
    if streak is None:
        return True
    return streak < SELF_IMPROVEMENT_STALE_DECISION_PERSISTENT_STREAK


# A single stale prior GPT decision means "rewrite/verify the receipt against
# the latest packet"; blocking that fresh verifier pass is circular. Once the
# same finding persists across audit runs, it is no longer just repair-in-flight:
# new risk must stop until the route proves the stale decision is cleared.
SELF_IMPROVEMENT_NON_TRADE_BLOCKER_CODES = frozenset({"LATEST_GPT_DECISION_STALE"})
SELF_IMPROVEMENT_STALE_DECISION_PERSISTENT_STREAK = 2


def _lane_forecast_direction_issue(lane: dict[str, Any]) -> VerificationIssue | None:
    forecast = lane.get("forecast")
    if not isinstance(forecast, dict):
        return None
    direction = str(forecast.get("forecast_direction") or "").upper()
    if direction not in {"UP", "DOWN"}:
        return None
    confidence = _optional_float(forecast.get("forecast_confidence"))
    if confidence is None or confidence < _forecast_confidence_min():
        return None
    forecast_side = "LONG" if direction == "UP" else "SHORT"
    lane_side = str(lane.get("direction") or "").upper()
    if lane_side == forecast_side:
        return None
    return VerificationIssue(
        "FORECAST_DIRECTION_CONFLICT",
        (
            f"{lane.get('lane_id')} is {lane_side} but current pair forecast is "
            f"{direction} conf={confidence:.2f}; verifier refuses forecast-opposite TRADE."
        ),
    )


def _forecast_confidence_min() -> float:
    try:
        from quant_rabbit.strategy.directional_forecaster import ENTRY_CONFIDENCE_MIN

        return float(ENTRY_CONFIDENCE_MIN)
    except Exception:
        return 1.0


def _cited_live_ready_lanes(decision: GPTTraderDecision, lane_ids: list[str]) -> list[str]:
    refs = set(decision.evidence_refs)
    return [lane_id for lane_id in lane_ids if f"intent:{lane_id}" in refs]


def _wait_is_session_only(decision: GPTTraderDecision) -> bool:
    text = " ".join(
        part
        for part in (
            decision.thesis,
            decision.narrative,
            decision.chart_story,
            decision.invalidation,
            decision.operator_summary,
            " ".join(decision.rejected_alternatives),
            " ".join(decision.risk_notes),
        )
        if part
    )
    if not SESSION_ONLY_WAIT_PATTERN.search(text):
        return False
    return CONCRETE_WAIT_GATE_PATTERN.search(text) is None


def _pair_from_lane_id(lane_id: str) -> str:
    """Extract the pair token from a `desk:pair:side:method[:MARKET]` lane id."""
    if not lane_id:
        return ""
    parts = lane_id.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return ""


def _pairs_from_lanes(lanes: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(lane.get("pair") or "") for lane in lanes if lane.get("pair")}))


def _pairs_from_lanes_and_positions(lanes: list[dict[str, Any]], snapshot: dict[str, Any]) -> tuple[str, ...]:
    pairs = {str(lane.get("pair") or "") for lane in lanes if lane.get("pair")}
    for position in snapshot.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        pair = str(position.get("pair") or "")
        if pair:
            pairs.add(pair)
    return tuple(sorted(pairs))


def _currencies_from_pairs(pairs: tuple[str, ...]) -> tuple[str, ...]:
    currencies: set[str] = set()
    for pair in pairs:
        for currency in pair.split("_"):
            if currency:
                currencies.add(currency)
    return tuple(sorted(currencies))


def _market_status_packet(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "MISSING",
            "evidence_ref": None,
            "is_fx_open": None,
            "active_sessions": [],
            "issues": ["MISSING_MARKET_STATUS_ARTIFACT"],
        }
    return {
        "status": "AVAILABLE",
        "evidence_ref": str(payload.get("evidence_ref") or "market:status"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "weekday": payload.get("weekday"),
        "weekday_index": payload.get("weekday_index"),
        "is_fx_open": payload.get("is_fx_open"),
        "closed_reason": payload.get("closed_reason"),
        "active_sessions": list(payload.get("active_sessions") or []),
        "minutes_to_next_open": payload.get("minutes_to_next_open"),
        "minutes_to_next_close": payload.get("minutes_to_next_close"),
        "contract": payload.get("contract") if isinstance(payload.get("contract"), dict) else {},
        "issues": [],
    }


def _market_context_packet(
    *,
    pairs: tuple[str, ...],
    currencies: tuple[str, ...],
    pair_charts_path: Path,
    context_asset_charts_path: Path,
    broker_instruments_path: Path,
    cross_asset_path: Path,
    flow_path: Path,
    currency_strength_path: Path,
    levels_path: Path,
    market_context_matrix_path: Path,
    calendar_path: Path,
    cot_path: Path,
    option_skew_path: Path,
) -> dict[str, Any]:
    artifacts = {
        "pair_charts": _load_optional_json(pair_charts_path),
        "context_asset_charts": _load_optional_json(context_asset_charts_path),
        "broker_instruments": _load_optional_json(broker_instruments_path),
        "cross_asset": _load_optional_json(cross_asset_path),
        "flow": _load_optional_json(flow_path),
        "currency_strength": _load_optional_json(currency_strength_path),
        "levels": _load_optional_json(levels_path),
        "market_context_matrix": _load_optional_json(market_context_matrix_path),
        "calendar": _load_optional_json(calendar_path),
        "cot": _load_optional_json(cot_path),
        "option_skew": _load_optional_json(option_skew_path),
    }
    missing = [
        f"MISSING_{name.upper()}_ARTIFACT"
        for name, payload in artifacts.items()
        if payload is None and name != "option_skew"
    ]
    pair_payloads = {
        pair: {
            "chart": _chart_summary(artifacts["pair_charts"], pair),
            "flow": _flow_summary(artifacts["flow"], pair),
            "levels": _levels_summary(artifacts["levels"], pair),
            "matrix": _matrix_pair_summary(artifacts["market_context_matrix"], pair),
            "calendar": _calendar_summary(artifacts["calendar"], pair),
            "option_skew": _option_skew_summary(artifacts["option_skew"], pair),
            "cross_correlations": _cross_correlations(artifacts["cross_asset"], pair),
        }
        for pair in pairs
    }
    issues = list(missing)
    for payload in artifacts.values():
        if isinstance(payload, dict):
            issues.extend(str(issue) for issue in payload.get("issues", [])[:12])
    return {
        "pairs": pair_payloads,
        "context_assets": _context_asset_charts_summary(
            artifacts["context_asset_charts"],
            artifacts["broker_instruments"],
        ),
        "broker_tradeability": _broker_tradeability_summary(artifacts["broker_instruments"]),
        "cross_asset": _cross_asset_summary(artifacts["cross_asset"]),
        "matrix_issues": _matrix_issues(artifacts["market_context_matrix"]),
        "currency_strength": _currency_strength_summary(artifacts["currency_strength"], currencies),
        "cot": _cot_summary(artifacts["cot"], currencies),
        "issues": issues[:40],
    }


def _context_asset_charts_summary(
    payload: dict[str, Any] | None,
    broker_instruments: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "assets": {}, "issues": ["MISSING_CONTEXT_ASSET_CHARTS_ARTIFACT"]}
    tradeable = set()
    not_tradeable = set()
    if isinstance(broker_instruments, dict):
        tradeable = {str(item) for item in broker_instruments.get("context_assets_tradeable", []) or []}
        not_tradeable = {str(item) for item in broker_instruments.get("context_assets_not_tradeable", []) or []}
    assets: dict[str, Any] = {}
    for chart in payload.get("charts", []) or []:
        if not isinstance(chart, dict):
            continue
        instrument = str(chart.get("pair") or "")
        if not instrument:
            continue
        assets[instrument] = {
            "broker_tradeable": instrument in tradeable,
            "broker_tradeability": "TRADEABLE" if instrument in tradeable else ("NOT_TRADEABLE" if instrument in not_tradeable else "UNKNOWN"),
            "evidence_ref": f"context_asset:{instrument}",
            "chart": _chart_summary(payload, instrument),
        }
    return {
        "status": "present",
        "role": payload.get("role"),
        "generated_at_utc": payload.get("generated_at_utc"),
        "assets": assets,
        "issues": [str(issue) for issue in payload.get("issues", [])[:12]],
    }


def _broker_tradeability_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "status": "missing",
            "evidence_ref": "broker:instruments",
            "issues": ["MISSING_BROKER_INSTRUMENTS_ARTIFACT"],
        }
    return {
        "status": payload.get("status"),
        "evidence_ref": "broker:instruments",
        "tradeability_policy": payload.get("tradeability_policy"),
        "tradeable_count": len(payload.get("tradeable_instruments") or []),
        "context_assets_tradeable": list(payload.get("context_assets_tradeable") or []),
        "context_assets_not_tradeable": list(payload.get("context_assets_not_tradeable") or [])[:24],
        "trader_pairs_missing": list(payload.get("trader_pairs_missing") or [])[:24],
        "issues": [str(issue) for issue in payload.get("issues", [])[:12]],
    }


def _chart_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    chart = _first_by_key(payload, "charts", "pair", pair)
    if not chart:
        return {}
    views: dict[str, Any] = {}
    for view in chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        granularity = str(view.get("granularity") or "")
        if not granularity:
            continue
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        regime = view.get("regime_reading") if isinstance(view.get("regime_reading"), dict) else {}
        family = view.get("family_scores") if isinstance(view.get("family_scores"), dict) else {}
        stat = view.get("stat_filters") if isinstance(view.get("stat_filters"), dict) else {}
        structure = view.get("structure") if isinstance(view.get("structure"), dict) else {}
        last_event = structure.get("last_event") if isinstance(structure.get("last_event"), dict) else {}
        views[granularity] = {
            **_small_dict(
                indicators,
                (
                    "atr_pips",
                    "atr_percentile_100",
                    "adx_14",
                    "adx_percentile_100",
                    "rsi_14",
                    "williams_r_14",
                    "mfi_14",
                    "choppiness_14",
                    "bb_width_percentile_100",
                    "hurst_100",
                    "close",
                    "macd_hist",
                    "supertrend_dir",
                    "ichimoku_cloud_pos",
                    "plus_di_14",
                    "minus_di_14",
                ),
            ),
            "regime": view.get("regime"),
            "regime_state": regime.get("state"),
            "regime_confidence": regime.get("confidence"),
            "trend_score": family.get("trend_score"),
            "mean_rev_score": family.get("mean_rev_score"),
            "breakout_score": family.get("breakout_score"),
            "disagreement": family.get("disagreement"),
            "last_jump_bars_ago": stat.get("last_jump_bars_ago"),
            "lag1_autocorr": stat.get("lag1_autocorr"),
            "structure": {
                "last_event": _small_dict(
                    last_event,
                    ("kind", "close_confirmed", "broken_pivot_price", "timestamp"),
                ),
            },
        }
    return {
        "source_generated_at_utc": payload.get("generated_at_utc"),
        "dominant_regime": chart.get("dominant_regime"),
        "long_score": chart.get("long_score"),
        "short_score": chart.get("short_score"),
        "chart_story": chart.get("chart_story"),
        "session": _small_dict(
            chart.get("session"),
            (
                "current_tag",
                "jp_holiday",
                "holiday_name",
                "judas_armed",
                "ny_midnight_open_price",
                "next_killzone",
                "minutes_to_next_killzone",
            ),
        ),
        "views": views,
    }


def _flow_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    spread = _first_by_key(payload, "spreads", "instrument", pair)
    return {"spread": _small_dict(spread, ("current_pips", "median_pips", "p90_pips", "stress_flag", "sample_size"))}


def _levels_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    levels = _first_by_key(payload, "pairs", "pair", pair)
    if not levels:
        return {}
    standard_pivot = None
    for pivot in levels.get("pivots", []) or []:
        if isinstance(pivot, dict) and pivot.get("style") == "STANDARD":
            standard_pivot = pivot
            break
    return {
        **_small_dict(levels, ("daily_open", "weekly_open", "monthly_open", "pdh", "pdl", "pdc", "last_close")),
        "standard_pivot": _small_dict(standard_pivot, ("pp", "r1", "r2", "s1", "s2")),
        "nearest_round_numbers": [
            _small_dict(item, ("price", "distance_pips"))
            for item in (levels.get("round_numbers", []) or [])[:3]
            if isinstance(item, dict)
        ],
    }


def _matrix_pair_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"status": "missing", "evidence_ref": f"matrix:{pair}"}
    side_map = ((payload.get("pairs") or {}).get(pair) or {})
    if not isinstance(side_map, dict):
        return {"status": "missing_pair", "evidence_ref": f"matrix:{pair}"}
    out: dict[str, Any] = {"status": "present", "evidence_ref": f"matrix:{pair}"}
    for side in ("LONG", "SHORT"):
        reading = side_map.get(side) if isinstance(side_map.get(side), dict) else {}
        out[side] = {
            "evidence_ref": reading.get("evidence_ref") or f"matrix:{pair}:{side}",
            "support_count": reading.get("support_count", 0),
            "reject_count": reading.get("reject_count", 0),
            "warning_count": reading.get("warning_count", 0),
            "missing_count": reading.get("missing_count", 0),
            "strongest_support": reading.get("strongest_support"),
            "strongest_reject": reading.get("strongest_reject"),
            "strongest_warning": reading.get("strongest_warning"),
            "supports": _compact_observations(reading.get("supports")),
            "rejects": _compact_observations(reading.get("rejects")),
            "warnings": _compact_observations(reading.get("warnings")),
        }
    return out


def _compact_observations(rows: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        out.append(_small_dict(item, ("code", "layer", "message", "evidence_refs")))
        if len(out) >= limit:
            break
    return out


def _matrix_issues(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return ["MISSING_MARKET_CONTEXT_MATRIX_ARTIFACT"]
    return [str(item) for item in payload.get("issues", [])[:12] if str(item).strip()]


def _calendar_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    window = _first_by_key(payload, "pair_windows", "pair", pair)
    if not window:
        return {}
    next_event = window.get("next_event") if isinstance(window.get("next_event"), dict) else {}
    return {
        "in_window": window.get("in_window"),
        "reason": window.get("reason"),
        "next_event": _small_dict(next_event, ("timestamp_utc", "currency", "impact", "title")),
    }


def _option_skew_summary(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("enabled") is False and payload.get("disabled_reason"):
        return {
            "enabled": False,
            "disabled_reason": payload.get("disabled_reason"),
            "readings": [],
        }
    readings = [
        _small_dict(item, ("tenor", "rr_25d", "atm_iv", "bf_25d", "issue"))
        for item in (payload or {}).get("readings", []) or []
        if isinstance(item, dict) and item.get("pair") == pair
    ]
    return {"enabled": bool(readings), "readings": readings[:3]}


def _cross_asset_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    assets = {}
    for asset in payload.get("assets", []) or []:
        if not isinstance(asset, dict):
            continue
        instrument = str(asset.get("instrument") or "")
        if instrument in {"USB10Y_USD", "USB02Y_USD", "SPX500_USD", "XAU_USD", "WTICO_USD", "BTC_USD"}:
            assets[instrument] = _small_dict(asset, ("last_price", "trend_label", "change_pct_24h", "z_score_60", "issue"))
    return {
        "synthetic_dxy": _small_dict(payload.get("synthetic_dxy"), ("last_value", "change_pct_24h", "change_pct_5d")),
        "yield_spreads": [
            _small_dict(item, ("name", "spread_last", "spread_change_24h", "issue"))
            for item in (payload.get("yield_spreads", []) or [])[:3]
            if isinstance(item, dict)
        ],
        "assets": assets,
    }


def _cross_correlations(payload: dict[str, Any] | None, pair: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    correlations = payload.get("correlations")
    if not isinstance(correlations, dict):
        return {}
    return _small_dict(
        correlations.get(pair),
        ("USB10Y_USD", "USB02Y_USD", "SPX500_USD", "XAU_USD", "WTICO_USD", "BTC_USD"),
    )


def _currency_strength_summary(payload: dict[str, Any] | None, currencies: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    summaries: dict[str, Any] = {
        "strongest_pair_suggestion": payload.get("strongest_pair_suggestion"),
    }
    wanted = set(currencies)
    for item in payload.get("scores", []) or []:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("currency") or "")
        if currency in wanted:
            summaries[currency] = _small_dict(item, ("rank", "score_pct"))
    return summaries


def _cot_summary(payload: dict[str, Any] | None, currencies: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    wanted = set(currencies)
    summaries: dict[str, Any] = {}
    for item in payload.get("reports", []) or []:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("currency") or "")
        if currency in wanted:
            summaries[currency] = _small_dict(
                item,
                ("report_date", "leveraged_net", "week_change_leveraged_net", "asset_mgr_net", "open_interest"),
            )
    return summaries


def _first_by_key(payload: dict[str, Any] | None, list_key: str, item_key: str, value: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for item in payload.get(list_key, []) or []:
        if isinstance(item, dict) and item.get(item_key) == value:
            return item
    return {}


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
