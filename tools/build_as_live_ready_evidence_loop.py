#!/usr/bin/env python3
"""Build read-only A/S LIVE_READY evidence-loop artifacts.

The tool reads local QuantRabbit evidence and the execution ledger, then writes
operator-review JSON/Markdown. It never calls OANDA and never stages, sends,
cancels, closes, or modifies broker orders.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MANUAL_EURUSD_TRADE_ID = "472987"
PROOF_KEYS = (
    "fresh_744h_replay",
    "s5_bidask_spread_included_replay",
    "sample_count_floor",
    "active_day_floor",
    "daily_stability_floor",
    "positive_day_rate_floor",
    "forecast_executable_proof",
    "geometry_proof",
    "attached_tp_proof",
    "market_close_absence_or_close_gate_proof",
    "residual_family_absence",
    "risk_engine_pass",
    "live_order_gateway_pass",
    "gpt_verifier_pass",
    "no_stale_packaged_rule",
    "no_guardian_operator_review_blocker",
)

ATTRIBUTED_REALIZED_SQL = """
WITH gateway_entries AS (
    SELECT trade_id, order_id, lane_id
    FROM execution_events
    WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
      AND lane_id IS NOT NULL AND lane_id != ''
),
entries AS (
    SELECT
        e.trade_id,
        COALESCE(NULLIF(MAX(e.pair), ''), '') AS pair,
        COALESCE(NULLIF(MAX(e.side), ''), '') AS side,
        COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS lane_id
    FROM execution_events e
    LEFT JOIN gateway_entries g
      ON (g.trade_id IS NOT NULL AND g.trade_id != '' AND g.trade_id = e.trade_id)
      OR (g.order_id IS NOT NULL AND g.order_id != '' AND g.order_id = e.order_id)
    WHERE e.event_type = 'ORDER_FILLED'
      AND e.trade_id IS NOT NULL AND e.trade_id != ''
    GROUP BY e.trade_id
    HAVING COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) IS NOT NULL
)
SELECT
    MAX(e.ts_utc) AS ts_utc,
    e.trade_id AS trade_id,
    entries.pair,
    entries.side,
    entries.lane_id,
    (
        SELECT e2.exit_reason FROM execution_events e2
        WHERE e2.trade_id = e.trade_id
          AND e2.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
          AND e2.realized_pl_jpy IS NOT NULL
        ORDER BY e2.ts_utc DESC LIMIT 1
    ) AS exit_reason,
    SUM(e.realized_pl_jpy) AS realized_pl_jpy
FROM execution_events e
INNER JOIN entries ON entries.trade_id = e.trade_id
WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
  AND e.realized_pl_jpy IS NOT NULL
GROUP BY e.trade_id, entries.pair, entries.side, entries.lane_id
HAVING SUM(e.realized_pl_jpy) != 0
ORDER BY MAX(e.ts_utc) ASC
"""

RAW_CLOSED_SQL = """
SELECT
    MAX(ts_utc) AS ts_utc,
    trade_id,
    COALESCE(NULLIF(MAX(pair), ''), 'UNKNOWN') AS pair,
    COALESCE(NULLIF(MAX(side), ''), 'UNKNOWN') AS side,
    COALESCE(NULLIF(MAX(lane_id), ''), '') AS lane_id,
    (
        SELECT e2.exit_reason FROM execution_events e2
        WHERE e2.trade_id = execution_events.trade_id
          AND e2.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
          AND e2.realized_pl_jpy IS NOT NULL
        ORDER BY e2.ts_utc DESC LIMIT 1
    ) AS exit_reason,
    SUM(realized_pl_jpy) AS realized_pl_jpy
FROM execution_events
WHERE event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
  AND realized_pl_jpy IS NOT NULL
  AND trade_id IS NOT NULL
  AND trade_id != ''
GROUP BY trade_id
HAVING SUM(realized_pl_jpy) != 0
ORDER BY MAX(ts_utc) ASC
"""


@dataclass(frozen=True)
class Outcome:
    ts_utc: str
    trade_id: str
    pair: str
    side: str
    lane_id: str
    method: str
    exit_reason: str
    realized_pl_jpy: float


def main() -> int:
    generated_at = _now()
    payloads = build_payloads(generated_at)
    for path, payload in payloads["json"].items():
        _write_json(ROOT / path, payload)
        print(f"wrote {path}")
    for path, text in payloads["markdown"].items():
        _write_text(ROOT / path, text)
        print(f"wrote {path}")
    return 0


def build_payloads(generated_at: str) -> dict[str, dict[Path, Any]]:
    acceptance = _load_json("data/profitability_acceptance.json")
    self_audit = _load_json("data/self_improvement_audit.json")
    capture = _load_json("data/capture_economics.json")
    order_intents = _load_json("data/order_intents.json")
    residual_table = _load_json("data/month_scale_residual_family_table.json")
    market_close_table = _load_json("data/market_close_leak_trade_table.json")
    as_board = _load_json("data/as_lane_candidate_board.json")
    daily = _load_json("data/daily_target_state.json")
    broker = _load_json("data/broker_snapshot.json")

    attributed = _load_outcomes(ROOT / "data/execution_ledger.db", ATTRIBUTED_REALIZED_SQL)
    raw_closed = _load_outcomes(ROOT / "data/execution_ledger.db", RAW_CLOSED_SQL)
    blocked = _blocked_sets(acceptance, residual_table, market_close_table)

    family_repair = _build_post_gate_gap_family_repair_table(
        generated_at=generated_at,
        attributed=attributed,
        blocked=blocked,
    )
    p0_decomposition = _build_p0_decomposition(
        generated_at=generated_at,
        acceptance=acceptance,
        self_audit=self_audit,
        residual_table=residual_table,
        market_close_table=market_close_table,
    )
    post_gate = _build_post_gate_capture(
        generated_at=generated_at,
        raw_closed=raw_closed,
        attributed=attributed,
        blocked=blocked,
        capture=capture,
        family_repair=family_repair,
    )
    firepower = _build_firepower_board(
        generated_at=generated_at,
        order_intents=order_intents,
        daily=daily,
        broker=broker,
        blocked=blocked,
        p0_decomposition=p0_decomposition,
    )
    proof_queue = _build_proof_pack_queue(
        generated_at=generated_at,
        firepower=firepower,
        p0_decomposition=p0_decomposition,
    )
    updated_board = _update_as_board(
        generated_at=generated_at,
        board=as_board,
        daily=daily,
        p0_decomposition=p0_decomposition,
        firepower=firepower,
        proof_queue=proof_queue,
    )

    return {
        "json": {
            Path("data/remaining_profitability_p0_decomposition.json"): p0_decomposition,
            Path("data/post_gate_gap_family_repair_table.json"): family_repair,
            Path("data/post_gate_capture_economics_decomposition.json"): post_gate,
            Path("data/rolling_30d_4x_firepower_board.json"): firepower,
            Path("data/as_proof_pack_queue.json"): proof_queue,
            Path("data/as_lane_candidate_board.json"): updated_board,
        },
        "markdown": {
            Path("docs/remaining_profitability_p0_decomposition.md"): _p0_md(p0_decomposition),
            Path("docs/post_gate_gap_family_repair_table.md"): _family_repair_md(family_repair),
            Path("docs/post_gate_capture_economics_decomposition.md"): _post_gate_md(post_gate),
            Path("docs/rolling_30d_4x_firepower_board.md"): _firepower_md(firepower),
            Path("docs/as_proof_pack_queue.md"): _proof_queue_md(proof_queue),
            Path("docs/as_lane_candidate_board.md"): _as_board_md(updated_board),
        },
    }


def _build_p0_decomposition(
    *,
    generated_at: str,
    acceptance: dict[str, Any],
    self_audit: dict[str, Any],
    residual_table: dict[str, Any],
    market_close_table: dict[str, Any],
) -> dict[str, Any]:
    findings = _findings_by_code(acceptance)
    residual_ids = _residual_trade_ids(residual_table)
    market_ids = _market_close_trade_ids(acceptance, market_close_table)
    rows: list[dict[str, Any]] = []

    self_finding = findings.get("SELF_IMPROVEMENT_P0_PRESENT") or {}
    p0_items = _nested(self_finding, "evidence", "p0_findings") or self_audit.get("p0_findings") or []
    if p0_items:
        for item in p0_items:
            code = str(item.get("code") or "SELF_IMPROVEMENT_P0_PRESENT")
            rows.append(
                _p0_row(
                    blocker_code="SELF_IMPROVEMENT_P0_PRESENT",
                    row_code=code,
                    classification="ACTIVE_BLOCKER",
                    source_artifact="data/self_improvement_audit.json",
                    message=item.get("message") or self_finding.get("message"),
                    trade_ids=[],
                    family_ids=[],
                    residual_ids=residual_ids,
                    market_ids=market_ids,
                    required_proof=_self_improvement_proof(code),
                    proposed_action=item.get("next_action") or _self_improvement_action(code),
                    affects_live_ready=True,
                )
            )
    else:
        rows.append(
            _p0_row(
                blocker_code="SELF_IMPROVEMENT_P0_PRESENT",
                row_code="SELF_IMPROVEMENT_P0_PRESENT",
                classification="STALE_SUPERSEDED",
                source_artifact="data/profitability_acceptance.json",
                message="not present in regenerated self-improvement P0 evidence",
                trade_ids=[],
                family_ids=[],
                residual_ids=residual_ids,
                market_ids=market_ids,
                required_proof="Keep regenerated self-improvement audit free of P0 findings.",
                proposed_action="No clearance action; keep as superseded evidence only.",
                affects_live_ready=False,
            )
        )

    neg = findings.get("NEGATIVE_EXPECTANCY_ACTIVE") or {}
    rows.append(
        _p0_row(
            blocker_code="NEGATIVE_EXPECTANCY_ACTIVE",
            row_code="NEGATIVE_EXPECTANCY_ACTIVE",
            classification="NEGATIVE_EXPECTANCY_REALIZED" if neg else "STALE_SUPERSEDED",
            source_artifact="data/capture_economics.json",
            message=neg.get("message") or "not present in current acceptance",
            trade_ids=[],
            family_ids=[],
            residual_ids=residual_ids,
            market_ids=market_ids,
            required_proof=(
                "Regenerated post-gate realized capture economics must be non-negative without relying "
                "only on excluded blocked historical families. Exact positive attached-TP HARVEST evidence "
                "may enter a proof pack, but cannot create live permission by itself."
            ),
            proposed_action="Repair market-close leakage and residual families, then rerun capture-economics and profitability-acceptance.",
            affects_live_ready=True,
        )
    )

    leak = findings.get("MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE") or {}
    for segment in (_nested(leak, "evidence", "segments") or []):
        trade_ids = _trade_ids_from_segment(segment)
        family = _family_id(segment)
        rows.append(
            _p0_row(
                blocker_code="MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                row_code="MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                classification="MARKET_CLOSE_LEAK_FAMILY",
                source_artifact="data/profitability_acceptance.json",
                message=leak.get("message"),
                trade_ids=trade_ids,
                family_ids=[family],
                residual_ids=residual_ids,
                market_ids=market_ids,
                required_proof=(
                    "TP-proven segment must no longer be net-damaged by MARKET_ORDER_TRADE_CLOSE leakage, "
                    "or every retained loss-side market close must have durable close-gate and contained-risk timing proof."
                ),
                proposed_action="Preserve attached TP/HARVEST shape; block or repair loss-side market-close path.",
                affects_live_ready=True,
            )
        )

    family_block = findings.get("MARKET_CLOSE_LEAK_FAMILY_BLOCKED") or {}
    if family_block:
        ev = family_block.get("evidence") if isinstance(family_block.get("evidence"), dict) else {}
        family = ev.get("family") if isinstance(ev.get("family"), dict) else {}
        rows.append(
            _p0_row(
                blocker_code="MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
                row_code="MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
                classification="MARKET_CLOSE_LEAK_FAMILY",
                source_artifact="data/market_close_leak_trade_table.json",
                message=family_block.get("message"),
                trade_ids=[str(x) for x in ev.get("system_gateway_loss_trade_ids") or []],
                family_ids=[_family_id(family)],
                residual_ids=residual_ids,
                market_ids=market_ids,
                required_proof=(
                    "Exact close-gate proof, contained-risk timing evidence, and TP-proven exception evidence "
                    "must all exist for EUR_USD LONG BREAKOUT_FAILURE before the family can route."
                ),
                proposed_action="Keep the family DRY_RUN_BLOCKED and non-permission until the full exception proof stack exists.",
                affects_live_ready=True,
            )
        )

    summary = {
        "p0_blocker_codes": [
            "SELF_IMPROVEMENT_P0_PRESENT",
            "NEGATIVE_EXPECTANCY_ACTIVE",
            "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
            "MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
        ],
        "rows": len(rows),
        "active_blockers": sum(1 for row in rows if row["classification"] == "ACTIVE_BLOCKER"),
        "negative_expectancy_rows": sum(1 for row in rows if row["classification"] == "NEGATIVE_EXPECTANCY_REALIZED"),
        "market_close_family_rows": sum(1 for row in rows if row["classification"] == "MARKET_CLOSE_LEAK_FAMILY"),
        "can_create_live_permission": False,
        "as_live_ready_can_be_cleared_now": False,
    }
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_remaining_profitability_p0_decomposition",
        "source_artifacts": [
            "data/profitability_acceptance.json",
            "docs/profitability_acceptance_report.md",
            "data/self_improvement_audit.json",
            "docs/self_improvement_audit_report.md",
            "data/capture_economics.json",
            "data/order_intents.json",
            "data/month_scale_residual_family_table.json",
            "docs/month_scale_residual_family_table.md",
            "data/market_close_leak_trade_table.json",
            "docs/market_close_leak_trade_table.md",
            "data/as_lane_candidate_board.json",
            "docs/as_lane_candidate_board.md",
        ],
        "classification_values": [
            "ACTIVE_BLOCKER",
            "STALE_SUPERSEDED",
            "MARKET_CLOSE_LEAK_FAMILY",
            "NEGATIVE_EXPECTANCY_REALIZED",
            "EVIDENCE_GAP",
            "TAXONOMY_DUPLICATE",
        ],
        "summary": summary,
        "rows": rows,
        "dependency_graph": _dependency_graph(rows),
        "live_side_effects": [],
    }


def _p0_row(
    *,
    blocker_code: str,
    row_code: str,
    classification: str,
    source_artifact: str,
    message: Any,
    trade_ids: list[str],
    family_ids: list[str],
    residual_ids: set[str],
    market_ids: set[str],
    required_proof: str,
    proposed_action: str,
    affects_live_ready: bool,
) -> dict[str, Any]:
    clean_trade_ids = [str(x) for x in trade_ids if str(x)]
    return {
        "blocker_code": blocker_code,
        "row_code": row_code,
        "classification": classification,
        "source_artifact": source_artifact,
        "message": message,
        "trade_ids": clean_trade_ids,
        "family_ids": [item for item in family_ids if item and item != "UNKNOWN UNKNOWN UNKNOWN"],
        "residual_family_gate_already_contains_it": bool(set(clean_trade_ids) & residual_ids),
        "market_close_leak_family_already_contains_it": bool(set(clean_trade_ids) & market_ids),
        "can_create_live_permission": False,
        "required_proof_to_clear": required_proof,
        "proposed_action": proposed_action,
        "clearing_would_affect_as_live_ready": affects_live_ready,
        "clearing_alone_creates_live_ready": False,
    }


def _build_post_gate_gap_family_repair_table(
    *,
    generated_at: str,
    attributed: list[Outcome],
    blocked: dict[str, Any],
) -> dict[str, Any]:
    manual_ids = {MANUAL_EURUSD_TRADE_ID}
    base_rows = [
        row
        for row in attributed
        if row.trade_id not in manual_ids | blocked["market_close_trade_ids"] | blocked["residual_trade_ids"]
    ]
    groups: dict[tuple[str, str, str, str], list[Outcome]] = {}
    for row in base_rows:
        groups.setdefault((row.pair, row.side, row.method, row.exit_reason), []).append(row)

    close_gate_proven = _close_gate_proven_trade_ids()
    rows: list[dict[str, Any]] = []
    for (pair, side, method, exit_reason), family_rows in groups.items():
        metrics = _bucket_metrics(family_rows)
        if (metrics.get("net_jpy") or 0.0) >= 0.0:
            continue
        trade_ids = [row.trade_id for row in sorted(family_rows, key=lambda item: item.ts_utc)]
        key = (pair, side, method)
        close_gate_ids = sorted(set(trade_ids) & close_gate_proven)
        market_close_path = exit_reason == "MARKET_ORDER_TRADE_CLOSE"
        market_close_allowed = bool(market_close_path and len(close_gate_ids) == len(trade_ids))
        action = _family_repair_action(
            pair=pair,
            side=side,
            method=method,
            exit_reason=exit_reason,
            market_close_allowed=market_close_allowed,
        )
        rows.append(
            {
                "family_id": f"{pair}|{side}|{method}|{exit_reason}",
                "pair": pair,
                "side": side,
                "method": method,
                "exit_reason": exit_reason,
                "attribution": "SYSTEM_GATEWAY_ATTRIBUTED_ONLY",
                "operator_manual_excluded": True,
                **metrics,
                "trade_ids": trade_ids,
                "trades_detail": [
                    {
                        "ts_utc": row.ts_utc,
                        "trade_id": row.trade_id,
                        "lane_id": row.lane_id,
                        "realized_pl_jpy": _round(row.realized_pl_jpy),
                    }
                    for row in sorted(family_rows, key=lambda item: item.realized_pl_jpy)
                ],
                "existing_residual_family_key_match": key in blocked["residual_keys"],
                "existing_market_close_family_key_match": key == blocked["market_close_family_key"],
                "close_gate_proven_trade_ids": close_gate_ids,
                "market_close_path_allowed": market_close_allowed if market_close_path else None,
                "can_create_live_permission": False,
                "blocker": _family_repair_blocker(
                    pair=pair,
                    side=side,
                    method=method,
                    exit_reason=exit_reason,
                    market_close_allowed=market_close_allowed,
                ),
                "action": action,
                "secondary_actions": _family_secondary_actions(exit_reason, market_close_allowed),
                "exact_clearance_condition": _family_clearance_condition(exit_reason, market_close_allowed),
                "permission_boundary": (
                    "This row may contain or ban a historical loss path only. It cannot create A/S "
                    "permission without fresh LIVE_READY, RiskEngine, LiveOrderGateway, GPT, and guardian proof."
                ),
            }
        )

    rows.sort(key=lambda item: (item["net_jpy"], item["family_id"]))
    filter_trade_ids = {
        trade_id
        for row in rows
        if row["action"] in {"BAN_FAMILY", "MARKET_CLOSE_PATH_BLOCK"}
        for trade_id in row.get("trade_ids") or []
    }
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_post_gate_gap_family_repair_table",
        "source_artifacts": [
            "data/execution_ledger.db",
            "data/post_gate_capture_economics_decomposition.json",
            "data/profitability_acceptance.json",
            "data/month_scale_residual_family_table.json",
            "data/market_close_leak_trade_table.json",
        ],
        "scope": {
            "name": "manual_excluded_plus_existing_market_close_and_residual_filters_remaining_negative_families",
            "definition": (
                "Trader-attributed realized rows excluding EUR_USD 472987, current "
                "MARKET_CLOSE_LEAK_FAMILY_BLOCKED trade IDs, and current month-scale residual trade IDs."
            ),
            "metrics": _bucket_metrics(base_rows),
        },
        "action_values": [
            "BAN_FAMILY",
            "MARKET_CLOSE_PATH_BLOCK",
            "REQUIRE_LOCAL_TP_PROOF",
            "REQUIRE_CLOSE_GATE_PROOF",
            "HISTORICAL_ONLY",
            "TAXONOMY_DUPLICATE",
        ],
        "summary": {
            "negative_family_count": len(rows),
            "market_close_path_block_count": sum(1 for row in rows if row["action"] == "MARKET_CLOSE_PATH_BLOCK"),
            "ban_family_count": sum(1 for row in rows if row["action"] == "BAN_FAMILY"),
            "require_local_tp_proof_count": sum(1 for row in rows if row["action"] == "REQUIRE_LOCAL_TP_PROOF"),
            "can_create_live_permission_count": 0,
            "containment_filter_trade_count": len(filter_trade_ids),
            "largest_remaining_adverse_family": rows[0]["family_id"] if rows else None,
        },
        "containment_filter_trade_ids": sorted(filter_trade_ids),
        "rows": rows,
        "live_side_effects": [],
    }


def _build_post_gate_capture(
    *,
    generated_at: str,
    raw_closed: list[Outcome],
    attributed: list[Outcome],
    blocked: dict[str, Any],
    capture: dict[str, Any],
    family_repair: dict[str, Any],
) -> dict[str, Any]:
    manual_ids = {MANUAL_EURUSD_TRADE_ID}
    market_ids = blocked["market_close_trade_ids"]
    residual_ids = blocked["residual_trade_ids"]
    family_filter_ids = _family_repair_filter_trade_ids(family_repair)
    unknown_market_ids = _unknown_unverified_gateway_close_ids(attributed)
    scopes = [
        (
            "raw_realized_system_ledger",
            attributed,
            "Trader-attributed realized ledger before post-gate exclusions; manual/tagless broker closes are excluded by attribution.",
            True,
            False,
        ),
        (
            "manual_excluded_ledger",
            [row for row in attributed if row.trade_id not in manual_ids],
            "Explicit EUR_USD 472987 manual exclusion applied; no closed 472987 row exists.",
            True,
            False,
        ),
        (
            "manual_excluded_plus_market_close_leak_family_blocked_excluded",
            [row for row in attributed if row.trade_id not in manual_ids | market_ids],
            "Manual exclusion plus current MARKET_CLOSE_LEAK_FAMILY_BLOCKED trade IDs removed.",
            False,
            False,
        ),
        (
            "manual_excluded_plus_residual_family_blocked_excluded",
            [row for row in attributed if row.trade_id not in manual_ids | residual_ids],
            "Manual exclusion plus month-scale residual-family blocked trade IDs removed.",
            False,
            False,
        ),
        (
            "manual_excluded_plus_both_market_close_leak_and_residual_family_filters",
            [row for row in attributed if row.trade_id not in manual_ids | market_ids | residual_ids],
            "Manual exclusion plus both non-permission historical filters.",
            False,
            False,
        ),
        (
            "manual_excluded_plus_existing_filters_plus_new_family_containment",
            [
                row
                for row in attributed
                if row.trade_id not in manual_ids | market_ids | residual_ids | family_filter_ids
            ],
            (
                "Manual exclusion plus existing filters plus this run's BAN_FAMILY / "
                "MARKET_CLOSE_PATH_BLOCK containment rows. Non-negative here is containment, not permission."
            ),
            False,
            False,
        ),
        (
            "attached_tp_only_harvest_segments",
            [row for row in attributed if row.exit_reason == "TAKE_PROFIT_ORDER"],
            "Broker attached-TP exits only. Positive result is proof-pack evidence only, not live permission.",
            False,
            True,
        ),
        (
            "market_order_trade_close_segments",
            [row for row in attributed if row.exit_reason == "MARKET_ORDER_TRADE_CLOSE"],
            "System-attributed MARKET_ORDER_TRADE_CLOSE exits.",
            True,
            False,
        ),
        (
            "unknown_unverified_gateway_close_segments",
            [row for row in attributed if row.trade_id in unknown_market_ids],
            "Market-close rows reconciled without durable local position-execution receipt.",
            True,
            False,
        ),
    ]
    scope_rows = []
    for name, rows, note, can_clear_negative, proof_candidate_only in scopes:
        metrics = _bucket_metrics(rows)
        non_negative = (metrics.get("expectancy_jpy_per_trade") or 0.0) >= 0.0 if metrics.get("trades") else False
        scope_rows.append(
            {
                "scope": name,
                "note": note,
                **metrics,
                "trade_ids_contributing_most_to_loss": _top_loss_ids(rows, 8),
                "can_clear_negative_expectancy_active": bool(can_clear_negative and non_negative),
                "can_create_live_permission": False,
                "can_enter_proof_pack": bool(proof_candidate_only and non_negative and metrics.get("trades", 0) > 0),
                "permission_boundary": (
                    "Non-negative only after excluding blocked historical families cannot create live permission."
                    if not can_clear_negative
                    else "Can clear only if this exact scope is the accepted regenerated economics basis."
                ),
            }
        )
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_post_gate_capture_economics_decomposition",
        "source_artifacts": [
            "data/execution_ledger.db",
            "data/capture_economics.json",
            "data/month_scale_residual_family_table.json",
            "data/market_close_leak_trade_table.json",
            "data/post_gate_gap_family_repair_table.json",
        ],
        "raw_broker_closed_context": _bucket_metrics(raw_closed),
        "capture_economics_status": capture.get("status"),
        "capture_economics_overall": capture.get("overall"),
        "blocked_trade_sets": {
            "manual_trade_ids": sorted(manual_ids),
            "market_close_leak_trade_ids": sorted(market_ids),
            "residual_family_trade_ids": sorted(residual_ids),
            "post_gate_family_repair_filter_trade_ids": sorted(family_filter_ids),
            "unknown_unverified_gateway_close_trade_ids": sorted(unknown_market_ids),
        },
        "scopes": scope_rows,
        "negative_expectancy_active_should_remain": True,
        "reason": (
            "The accepted capture_economics artifact remains NEGATIVE_EXPECTANCY. Positive TP-only "
            "or post-filter scopes are proof-pack inputs only and do not create live permission."
        ),
        "live_side_effects": [],
    }


def _build_firepower_board(
    *,
    generated_at: str,
    order_intents: dict[str, Any],
    daily: dict[str, Any],
    broker: dict[str, Any],
    blocked: dict[str, Any],
    p0_decomposition: dict[str, Any],
) -> dict[str, Any]:
    funding_adjusted = _float(daily.get("funding_adjusted_equity"))
    required_pct = _float(daily.get("required_calendar_daily_return_funding_adjusted"))
    required_daily_jpy = funding_adjusted * required_pct / 100.0 if funding_adjusted and required_pct else None
    rows: list[dict[str, Any]] = []
    hard_excluded: list[dict[str, Any]] = []
    for result in order_intents.get("results") or []:
        row = _candidate_firepower_row(result, daily=daily, broker=broker, blocked=blocked, required_daily_jpy=required_daily_jpy)
        if row["hard_excluded"]:
            hard_excluded.append({k: row.get(k) for k in ("lane_id", "pair", "side", "method", "hard_exclusion_reasons")})
        else:
            rows.append(row)
    rows.sort(key=lambda item: (not item["candidate_daily_expected_return_pct_ge_required"], -(item.get("expected_daily_return_pct_on_funding_adjusted_equity") or -9999), item["proof_gap_count"], item["lane_id"]))
    top = rows[:20]
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_rolling_30d_4x_firepower_board",
        "source_artifacts": [
            "data/order_intents.json",
            "data/daily_target_state.json",
            "data/broker_snapshot.json",
            "data/capture_economics.json",
            "data/month_scale_residual_family_table.json",
            "data/market_close_leak_trade_table.json",
            "data/remaining_profitability_p0_decomposition.json",
        ],
        "target_math": _target_math(daily),
        "summary": {
            "total_order_intent_rows": len(order_intents.get("results") or []),
            "candidate_rows_after_hard_exclusions": len(rows),
            "hard_excluded_rows": len(hard_excluded),
            "rows_meeting_required_daily_return_prefilter": sum(1 for row in rows if row["candidate_daily_expected_return_pct_ge_required"]),
            "can_create_live_permission_rows": 0,
            "can_enter_proof_pack_rows": sum(1 for row in rows if row["can_enter_proof_pack"]),
            "as_live_ready_path_exists": False,
            "normal_routing_status": "BLOCKED",
            "p0_dependency_count": len(p0_decomposition.get("rows") or []),
        },
        "hard_exclusions": hard_excluded,
        "candidates": top,
        "all_candidate_count": len(rows),
        "live_side_effects": [],
    }


def _candidate_firepower_row(
    result: dict[str, Any],
    *,
    daily: dict[str, Any],
    broker: dict[str, Any],
    blocked: dict[str, Any],
    required_daily_jpy: float | None,
) -> dict[str, Any]:
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    metadata = intent.get("metadata") if isinstance(intent.get("metadata"), dict) else {}
    market_context = intent.get("market_context") if isinstance(intent.get("market_context"), dict) else {}
    risk = result.get("risk_metrics") if isinstance(result.get("risk_metrics"), dict) else {}
    lane_id = str(result.get("lane_id") or "")
    pair = str(intent.get("pair") or _lane_part(lane_id, 1) or "UNKNOWN")
    side = str(intent.get("side") or _lane_part(lane_id, 2) or "UNKNOWN").upper()
    method = str(market_context.get("method") or _lane_part(lane_id, 3) or "UNKNOWN")
    order_type = str(intent.get("order_type") or _lane_part(lane_id, 4) or "UNKNOWN")
    exit_shape = _exit_shape(metadata)
    rule = _evidence_rule(metadata)
    sample_count = _int(rule.get("samples")) or _int(metadata.get("positive_rotation_tp_trades")) or _int(metadata.get("capture_take_profit_trades"))
    active_days = _int(rule.get("active_days"))
    win_rate = _float(rule.get("optimized_win_rate")) or _float(metadata.get("capture_take_profit_wins")) and 1.0
    wilson = _float(metadata.get("positive_rotation_tp_win_rate_lower")) or _float(rule.get("hit_rate_wilson_lower")) or _float(rule.get("economic_hit_rate_wilson_lower"))
    expected_jpy = (
        _float(metadata.get("positive_rotation_pessimistic_expectancy_jpy"))
        or _float(metadata.get("capture_take_profit_expectancy_jpy"))
    )
    spread_expectancy = expected_jpy
    avg_win = _float(metadata.get("capture_take_profit_avg_win_jpy")) or _float(metadata.get("capture_avg_win_jpy"))
    avg_loss = _float(metadata.get("capture_take_profit_avg_loss_jpy")) or _float(metadata.get("positive_rotation_loss_proxy_jpy")) or _float(metadata.get("capture_avg_loss_jpy"))
    max_loss = _float(metadata.get("max_loss_jpy")) or _float(risk.get("risk_jpy"))
    trades_per_day = _float(rule.get("avg_daily_samples"))
    if trades_per_day is None and sample_count and active_days:
        trades_per_day = sample_count / active_days
    funding_adjusted = _float(daily.get("funding_adjusted_equity"))
    expected_daily_return_pct = (
        expected_jpy * trades_per_day / funding_adjusted * 100.0
        if expected_jpy is not None and trades_per_day and funding_adjusted
        else None
    )
    required_trades_per_day = (
        required_daily_jpy / expected_jpy if required_daily_jpy and expected_jpy and expected_jpy > 0 else None
    )
    actual_units = _int(metadata.get("sizing_actual_units")) or _int(intent.get("units"))
    realistic_margin = _float(metadata.get("estimated_margin_jpy")) or _float(risk.get("estimated_margin_jpy"))
    margin_min_lot = None
    if realistic_margin is not None and actual_units and actual_units > 0:
        margin_min_lot = realistic_margin / actual_units * 1000.0
    blockers = [str(x) for x in result.get("live_blocker_codes") or []]
    hard_reasons = _hard_exclusion_reasons(pair, side, method, lane_id, blocked)
    proof_gaps = _proof_gaps(
        result=result,
        metadata=metadata,
        rule=rule,
        blockers=blockers,
        hard_reasons=hard_reasons,
        sample_count=sample_count,
        active_days=active_days,
    )
    source_evidence = _source_evidence(metadata, rule)
    can_enter = bool(
        expected_daily_return_pct is not None
        and expected_daily_return_pct > 0
        and not hard_reasons
        and (
            (
                expected_daily_return_pct
                >= (_float(daily.get("required_calendar_daily_return_funding_adjusted")) or math.inf)
                and not source_evidence.get("historical_only")
            )
            or metadata.get("positive_rotation_proof_collection_ready") is True
        )
    )
    return {
        "lane_id": lane_id,
        "pair": pair,
        "side": side,
        "method": method,
        "order_type": order_type,
        "exit_shape": exit_shape,
        "source_evidence": source_evidence,
        "sample_count": sample_count,
        "active_days": active_days,
        "win_rate": round(win_rate, 6) if isinstance(win_rate, float) else win_rate,
        "wilson_lower": wilson,
        "spread_included_expectancy": spread_expectancy,
        "expected_jpy_per_trade": expected_jpy,
        "avg_win_jpy": avg_win,
        "avg_loss_jpy": avg_loss,
        "max_loss_jpy": max_loss,
        "estimated_trades_per_day_available": trades_per_day,
        "required_trades_per_day_to_contribute_to_30d_4x": required_trades_per_day,
        "margin_requirement_min_lot_jpy": _round(margin_min_lot),
        "margin_requirement_realistic_size_jpy": realistic_margin,
        "realistic_units": actual_units,
        "expected_daily_return_pct_on_funding_adjusted_equity": _round(expected_daily_return_pct),
        "candidate_daily_expected_return_pct_ge_required": bool(
            expected_daily_return_pct is not None
            and expected_daily_return_pct >= (_float(daily.get("required_calendar_daily_return_funding_adjusted")) or math.inf)
        ),
        "current_blockers": blockers,
        "exact_proof_gaps": proof_gaps,
        "proof_gap_count": len(proof_gaps),
        "hard_excluded": bool(hard_reasons),
        "hard_exclusion_reasons": hard_reasons,
        "can_create_live_permission": False,
        "can_enter_proof_pack": can_enter,
        "status": result.get("status"),
        "risk_allowed": result.get("risk_allowed"),
        "broker_margin_context": {
            "margin_used_jpy": _nested(broker, "account", "margin_used_jpy"),
            "margin_available_jpy": _nested(broker, "account", "margin_available_jpy"),
            "broker_margin_free_units": metadata.get("broker_margin_free_units"),
        },
    }


def _build_proof_pack_queue(
    *,
    generated_at: str,
    firepower: dict[str, Any],
    p0_decomposition: dict[str, Any],
) -> dict[str, Any]:
    audjpy_fresh_s5 = _load_json("data/audjpy_limit_fresh_s5_bidask_replay.json")
    candidates = [
        row for row in firepower.get("candidates") or []
        if row.get("can_enter_proof_pack") or row.get("candidate_daily_expected_return_pct_ge_required")
    ]
    queue = []
    for row in candidates[:12]:
        missing = _missing_proof_map(row)
        _apply_audjpy_limit_fresh_s5(row, missing, audjpy_fresh_s5)
        current_blockers = _current_blockers_with_audjpy_limit_fresh_s5(row, audjpy_fresh_s5)
        classification = _proof_classification({**row, "current_blockers": current_blockers}, missing)
        queue.append(
            {
                "lane_id": row.get("lane_id"),
                "pair": row.get("pair"),
                "side": row.get("side"),
                "method": row.get("method"),
                "order_type": row.get("order_type"),
                "exit_shape": row.get("exit_shape"),
                "expected_daily_return_pct_on_funding_adjusted_equity": row.get("expected_daily_return_pct_on_funding_adjusted_equity"),
                "proof_classification": classification,
                "missing_proof": missing,
                "current_blockers": current_blockers,
                "can_create_live_permission": False,
                "can_enter_proof_pack": bool(classification in {"EVIDENCE_GAP", "REPAIR_REQUIRED"} and not row.get("hard_excluded")),
                "proof_distance": sum(1 for value in missing.values() if value is not True),
                "notes": _proof_notes(row, classification),
            }
        )
    queue.sort(key=lambda item: (item["proof_distance"], -(_float(item.get("expected_daily_return_pct_on_funding_adjusted_equity")) or -9999), item["lane_id"]))
    return {
        "generated_at_utc": generated_at,
        "mode": "read_only_as_proof_pack_queue",
        "source_artifacts": [
            "data/rolling_30d_4x_firepower_board.json",
            "data/remaining_profitability_p0_decomposition.json",
            "data/order_intents.json",
            "data/audjpy_limit_fresh_s5_bidask_replay.json",
            "docs/audjpy_limit_fresh_s5_bidask_replay.md",
        ],
        "classification_values": ["PROOF_READY", "EVIDENCE_GAP", "REPAIR_REQUIRED", "REJECTED", "HISTORICAL_ONLY"],
        "summary": {
            "queue_count": len(queue),
            "proof_ready_count": sum(1 for item in queue if item["proof_classification"] == "PROOF_READY"),
            "can_create_live_permission_count": 0,
            "as_live_ready_path_exists": False,
            "remaining_p0_rows": len(p0_decomposition.get("rows") or []),
        },
        "queue": queue,
        "live_side_effects": [],
    }


def _update_as_board(
    *,
    generated_at: str,
    board: dict[str, Any],
    daily: dict[str, Any],
    p0_decomposition: dict[str, Any],
    firepower: dict[str, Any],
    proof_queue: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(board)
    queue = proof_queue.get("queue") or []
    closest = next((row for row in queue if row.get("can_enter_proof_pack")), queue[0] if queue else None)
    updated.update(
        {
            "generated_at_utc": generated_at,
            "as_live_ready_path_exists": False,
            "live_ready_lanes": 0,
            "normal_routing_status": "BLOCKED",
            "routing_allowed": False,
            "rolling_30d_4x_target_math": _target_math(daily),
            "firepower_board_summary": firepower.get("summary"),
            "remaining_p0_dependency_graph": p0_decomposition.get("dependency_graph"),
            "closest_candidate_to_proof_pack": closest,
            "exact_blocker_preventing_live_ready": {
                "primary": "PROFITABILITY_ACCEPTANCE_BLOCKED",
                "p0_rows": [row.get("row_code") for row in p0_decomposition.get("rows") or []],
                "global_blockers": [
                    "NEGATIVE_EXPECTANCY_ACTIVE",
                    "MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE",
                    "MARKET_CLOSE_LEAK_FAMILY_BLOCKED",
                    "SELF_IMPROVEMENT_P0_PRESENT",
                    "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED",
                    "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE",
                ],
                "normal_routing_must_remain_blocked": True,
                "as_live_ready_stays_zero": True,
            },
            "new_evidence_loop_artifacts": [
                "data/remaining_profitability_p0_decomposition.json",
                "docs/remaining_profitability_p0_decomposition.md",
                "data/post_gate_capture_economics_decomposition.json",
                "docs/post_gate_capture_economics_decomposition.md",
                "data/post_gate_gap_family_repair_table.json",
                "docs/post_gate_gap_family_repair_table.md",
                "data/rolling_30d_4x_firepower_board.json",
                "docs/rolling_30d_4x_firepower_board.md",
                "data/as_proof_pack_queue.json",
                "docs/as_proof_pack_queue.md",
                "data/post_gate_expectancy_gap_trace.json",
                "docs/post_gate_expectancy_gap_trace.md",
                "data/historical_only_to_fresh_proof_replay.json",
                "docs/historical_only_to_fresh_proof_replay.md",
                "data/audjpy_short_breakout_failure_repair_proof.json",
                "docs/audjpy_short_breakout_failure_repair_proof.md",
                "data/audjpy_short_breakout_failure_limit_proof_pack.json",
                "docs/audjpy_short_breakout_failure_limit_proof_pack.md",
                "data/audjpy_limit_fresh_s5_bidask_replay.json",
                "docs/audjpy_limit_fresh_s5_bidask_replay.md",
                "data/manual_eurusd_tp_replacement_provenance.json",
                "docs/manual_eurusd_tp_replacement_provenance.md",
                "data/profitability_acceptance_blocker_reconciliation.json",
                "docs/profitability_acceptance_blocker_reconciliation.md",
                "data/portfolio_4x_path_planner.json",
                "docs/portfolio_4x_path_planner.md",
            ],
        }
    )
    return updated


def _target_math(daily: dict[str, Any]) -> dict[str, Any]:
    start = _float(daily.get("rolling_30d_start_equity"))
    current_raw = _float(daily.get("current_equity_raw"))
    funding_adjusted = _float(daily.get("funding_adjusted_equity"))
    target_from_start = start * 4.0 if start is not None else None
    prompt_style_target_from_current = funding_adjusted * 4.0 if funding_adjusted is not None else None
    return {
        "rolling_30d_start_equity": start,
        "current_equity_raw": current_raw,
        "capital_flows_30d": _float(daily.get("capital_flows_30d")),
        "funding_adjusted_equity": funding_adjusted,
        "rolling_30d_multiplier_raw": _float(daily.get("rolling_30d_multiplier_raw")),
        "rolling_30d_multiplier_funding_adjusted": _float(daily.get("rolling_30d_multiplier_funding_adjusted")),
        "target_equity_from_rolling_start_4x_jpy": _round(target_from_start),
        "prompt_style_target_equity_current_funding_adjusted_4x_jpy": _round(prompt_style_target_from_current),
        "remaining_to_4x_funding_adjusted": _float(daily.get("remaining_to_4x_funding_adjusted")),
        "remaining_to_4x_raw": _float(daily.get("remaining_to_4x_raw")),
        "required_calendar_daily_return_funding_adjusted_pct": _float(daily.get("required_calendar_daily_return_funding_adjusted")),
        "required_active_day_return_funding_adjusted_pct": _float(daily.get("required_active_day_return_funding_adjusted")),
        "required_calendar_daily_profit_jpy": _round(
            funding_adjusted * (_float(daily.get("required_calendar_daily_return_funding_adjusted")) or 0.0) / 100.0
            if funding_adjusted is not None
            else None
        ),
        "performance_basis": daily.get("performance_basis"),
        "sizing_basis": daily.get("sizing_basis"),
        "pace_state": daily.get("pace_state"),
    }


def _blocked_sets(
    acceptance: dict[str, Any],
    residual_table: dict[str, Any],
    market_close_table: dict[str, Any],
) -> dict[str, Any]:
    residual_ids = _residual_trade_ids(residual_table)
    market_ids = _market_close_trade_ids(acceptance, market_close_table)
    residual_keys = {
        (str(item.get("pair")), str(item.get("side")), str(item.get("method") or item.get("strategy")))
        for item in residual_table.get("families") or []
        if item.get("can_create_live_permission") is False
    }
    family = _nested(_findings_by_code(acceptance).get("MARKET_CLOSE_LEAK_FAMILY_BLOCKED") or {}, "evidence", "family") or {}
    market_key = (
        str(family.get("pair") or "EUR_USD"),
        str(family.get("side") or "LONG"),
        str(family.get("method") or "BREAKOUT_FAILURE"),
    )
    return {
        "residual_trade_ids": residual_ids,
        "market_close_trade_ids": market_ids,
        "residual_keys": residual_keys,
        "market_close_family_key": market_key,
    }


def _load_outcomes(db_path: Path, sql: str) -> list[Outcome]:
    rows: list[Outcome] = []
    if not db_path.exists():
        return rows
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for ts, trade_id, pair, side, lane_id, exit_reason, pl in conn.execute(sql):
            if pl is None:
                continue
            lane = str(lane_id or "")
            rows.append(
                Outcome(
                    ts_utc=str(ts or ""),
                    trade_id=str(trade_id or ""),
                    pair=str(pair or "UNKNOWN"),
                    side=str(side or "UNKNOWN").upper(),
                    lane_id=lane,
                    method=_lane_part(lane, 3) or "UNKNOWN",
                    exit_reason=str(exit_reason or "UNKNOWN"),
                    realized_pl_jpy=float(pl),
                )
            )
    return rows


def _bucket_metrics(rows: list[Outcome]) -> dict[str, Any]:
    wins = [row.realized_pl_jpy for row in rows if row.realized_pl_jpy > 0]
    losses = [row.realized_pl_jpy for row in rows if row.realized_pl_jpy < 0]
    n = len(wins) + len(losses)
    if n == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_jpy": 0.0,
            "expectancy_jpy_per_trade": None,
            "profit_factor": None,
            "avg_win_jpy": None,
            "avg_loss_jpy": None,
            "max_loss_jpy": None,
        }
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "net_jpy": _round(gross_win - gross_loss),
        "expectancy_jpy_per_trade": _round((gross_win - gross_loss) / n),
        "profit_factor": _round(gross_win / gross_loss) if gross_loss else None,
        "win_rate": _round(len(wins) / n),
        "avg_win_jpy": _round(gross_win / len(wins)) if wins else None,
        "avg_loss_jpy": _round(gross_loss / len(losses)) if losses else None,
        "max_loss_jpy": _round(min(losses)) if losses else None,
    }


def _unknown_unverified_gateway_close_ids(rows: list[Outcome]) -> set[str]:
    # In this ledger generation, reconciled broker closes without a local
    # position-execution receipt are exactly the unverified gateway-close proof
    # gap the acceptance layer calls out. Treat MARKET_ORDER_TRADE_CLOSE rows as
    # unverified unless a durable sent receipt exists; the current DB has only
    # one sent close and many reconciled closes.
    sent_ids: set[str] = set()
    with sqlite3.connect(f"file:{ROOT / 'data/execution_ledger.db'}?mode=ro", uri=True) as conn:
        for (trade_id,) in conn.execute(
            "SELECT DISTINCT trade_id FROM execution_events WHERE event_type='GATEWAY_TRADE_CLOSE_SENT'"
        ):
            if trade_id:
                sent_ids.add(str(trade_id))
    return {
        row.trade_id
        for row in rows
        if row.exit_reason == "MARKET_ORDER_TRADE_CLOSE" and row.trade_id not in sent_ids
    }


def _close_gate_proven_trade_ids() -> set[str]:
    proven: set[str] = set()
    db_path = ROOT / "data/execution_ledger.db"
    if not db_path.exists():
        return proven
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for (trade_id,) in conn.execute(
            """
            SELECT DISTINCT trade_id
            FROM execution_events
            WHERE event_type IN ('GATEWAY_GPT_CLOSE_ACCEPTED', 'GATEWAY_TRADE_CLOSE_SENT')
              AND trade_id IS NOT NULL
              AND trade_id != ''
            """
        ):
            proven.add(str(trade_id))
    return proven


def _family_repair_action(
    *,
    pair: str,
    side: str,
    method: str,
    exit_reason: str,
    market_close_allowed: bool,
) -> str:
    if pair == "GBP_USD" and side == "LONG" and method == "BREAKOUT_FAILURE" and exit_reason == "MARKET_ORDER_TRADE_CLOSE":
        return "MARKET_CLOSE_PATH_BLOCK"
    if exit_reason == "MARKET_ORDER_TRADE_CLOSE":
        return "REQUIRE_CLOSE_GATE_PROOF" if market_close_allowed else "MARKET_CLOSE_PATH_BLOCK"
    if exit_reason == "MARKET_ORDER_MARGIN_CLOSEOUT":
        return "BAN_FAMILY"
    if exit_reason in {"STOP_LOSS_ORDER", "TAKE_PROFIT_ORDER"}:
        return "REQUIRE_LOCAL_TP_PROOF"
    return "HISTORICAL_ONLY"


def _family_repair_blocker(
    *,
    pair: str,
    side: str,
    method: str,
    exit_reason: str,
    market_close_allowed: bool,
) -> str:
    if pair == "GBP_USD" and side == "LONG" and method == "BREAKOUT_FAILURE" and exit_reason == "MARKET_ORDER_TRADE_CLOSE":
        return "GBP_USD_LONG_BREAKOUT_FAILURE_MARKET_CLOSE_PATH_BLOCKED"
    if exit_reason == "MARKET_ORDER_TRADE_CLOSE" and not market_close_allowed:
        return "MARKET_CLOSE_PATH_UNVERIFIED_AND_NEGATIVE"
    if exit_reason == "MARKET_ORDER_TRADE_CLOSE":
        return "MARKET_CLOSE_PATH_REQUIRES_CLOSE_GATE_PROOF"
    if exit_reason == "MARKET_ORDER_MARGIN_CLOSEOUT":
        return "MARGIN_CLOSEOUT_FAMILY_BANNED_FROM_A_S"
    if exit_reason == "STOP_LOSS_ORDER":
        return "LOSS_EXIT_REQUIRES_LOCAL_TP_OR_GEOMETRY_PROOF"
    return "NEGATIVE_HISTORICAL_FAMILY_REPAIR_REQUIRED"


def _family_secondary_actions(exit_reason: str, market_close_allowed: bool) -> list[str]:
    actions: list[str] = []
    if exit_reason == "MARKET_ORDER_TRADE_CLOSE":
        actions.append("REQUIRE_CLOSE_GATE_PROOF")
        if not market_close_allowed:
            actions.append("REQUIRE_LOCAL_TP_PROOF")
    elif exit_reason == "MARKET_ORDER_MARGIN_CLOSEOUT":
        actions.append("REQUIRE_LOCAL_TP_PROOF")
    elif exit_reason == "STOP_LOSS_ORDER":
        actions.append("REQUIRE_LOCAL_TP_PROOF")
    return actions


def _family_clearance_condition(exit_reason: str, market_close_allowed: bool) -> str:
    if exit_reason == "MARKET_ORDER_TRADE_CLOSE":
        if market_close_allowed:
            return (
                "Retain only if every matching future close has durable close-gate proof, "
                "contained-risk timing proof, and the regenerated family economics are non-negative."
            )
        return (
            "Block the MARKET_ORDER_TRADE_CLOSE path. A fresh lane may be considered only through "
            "attached local TP/HARVEST proof or exact close-gate proof plus non-negative regenerated economics."
        )
    if exit_reason == "MARKET_ORDER_MARGIN_CLOSEOUT":
        return "Reject as an A/S vehicle until margin-closeout risk disappears from a fresh replay and current gateway sizing proof."
    if exit_reason == "STOP_LOSS_ORDER":
        return "Require exact local TP, geometry, and risk proof showing the loss path no longer dominates the family."
    return "Treat as historical repair context until exact fresh proof regenerates a clean LIVE_READY lane."


def _family_repair_filter_trade_ids(table: dict[str, Any]) -> set[str]:
    return {
        str(trade_id)
        for row in table.get("rows") or []
        if row.get("action") in {"BAN_FAMILY", "MARKET_CLOSE_PATH_BLOCK"}
        for trade_id in row.get("trade_ids") or []
        if str(trade_id)
    }


def _hard_exclusion_reasons(
    pair: str,
    side: str,
    method: str,
    lane_id: str,
    blocked: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    key = (pair, side, method)
    if key in blocked["residual_keys"]:
        reasons.append("MONTH_SCALE_RESIDUAL_FAMILY_BLOCKED")
    if key == blocked["market_close_family_key"]:
        reasons.append("EUR_USD_LONG_BREAKOUT_FAILURE_MARKET_CLOSE_LEAK_FAMILY")
    if pair == "AUD_USD":
        reasons.append("AUD_USD_CURRENT_NEGATIVE_EXPECTANCY_GEOMETRY_REPLAY_BLOCKERS")
    if pair == "USD_JPY" and side == "LONG" and method == "BREAKOUT_FAILURE":
        reasons.append("STALE_PACKAGED_USD_JPY_RULE_WITHOUT_FRESH_TP10_SL7_PROOF")
    if pair == "EUR_JPY" and side == "SHORT":
        reasons.append("EUR_JPY_SHORT_BROAD_NEGATIVE_EVIDENCE")
    if lane_id == f"manual:{MANUAL_EURUSD_TRADE_ID}":
        reasons.append("MANUAL_EUR_USD_472987")
    return reasons


def _proof_gaps(
    *,
    result: dict[str, Any],
    metadata: dict[str, Any],
    rule: dict[str, Any],
    blockers: list[str],
    hard_reasons: list[str],
    sample_count: int | None,
    active_days: int | None,
) -> list[str]:
    gaps: list[str] = []
    if hard_reasons:
        gaps.extend(hard_reasons)
    if not rule:
        gaps.append("S5_BIDASK_SPREAD_INCLUDED_REPLAY_MISSING")
    elif rule.get("live_grade") is not True and rule.get("adoption_status") not in {"LIVE_SUPPORT", "PROOF_COLLECTION"}:
        gaps.append(str(rule.get("adoption_status") or "S5_REPLAY_NOT_LIVE_GRADE"))
    if not sample_count or sample_count < 20:
        gaps.append("SAMPLE_COUNT_FLOOR_NOT_MET")
    if not active_days or active_days < 3:
        gaps.append("ACTIVE_DAY_FLOOR_NOT_MET")
    if "TELEMETRY_FORECAST_QUOTE_STALE_FOR_LIVE" in blockers or "STALE_QUOTE" in blockers:
        gaps.append("FRESH_FORECAST_QUOTE_PROOF_MISSING")
    if "SPREAD_TOO_WIDE" in blockers:
        gaps.append("CURRENT_SPREAD_GATE_BLOCKED")
    if any("NEGATIVE_EXPECTANCY" in code for code in blockers):
        gaps.append("NEGATIVE_EXPECTANCY_ACTIVE")
    if "HARVEST_TP_STRUCTURE_MISSING" in blockers:
        gaps.append("ATTACHED_TP_STRUCTURE_PROOF_MISSING")
    if "GUARDIAN_RECEIPT_OPERATOR_REVIEW_REQUIRED" in blockers:
        gaps.append("GUARDIAN_OPERATOR_REVIEW_BLOCKER")
    if result.get("risk_allowed") is not True:
        gaps.append("RISK_ENGINE_PASS_MISSING")
    if result.get("status") != "LIVE_READY":
        gaps.append("LIVE_ORDER_GATEWAY_PREFLIGHT_MISSING")
    gaps.append("FRESH_GPT_VERIFIER_TRADE_RECEIPT_MISSING")
    return sorted(set(gaps))


def _missing_proof_map(row: dict[str, Any]) -> dict[str, Any]:
    gaps = set(row.get("exact_proof_gaps") or [])
    blockers = set(row.get("current_blockers") or [])
    source = row.get("source_evidence") if isinstance(row.get("source_evidence"), dict) else {}
    bidask_status = str(source.get("bidask_rule_status") or "")
    historical_only = bool(source.get("historical_only"))
    s5_live_support = bool(
        bidask_status in {"LIVE_SUPPORT", "PROOF_COLLECTION"}
        and not historical_only
        and "LIVE_BLOCK_NEGATIVE_EXPECTANCY" not in gaps
    )
    return {
        "fresh_744h_replay": "MONTH_SCALE_RESIDUAL_FAMILY_BLOCKED" not in gaps and not historical_only,
        "s5_bidask_spread_included_replay": s5_live_support,
        "sample_count_floor": "SAMPLE_COUNT_FLOOR_NOT_MET" not in gaps,
        "active_day_floor": "ACTIVE_DAY_FLOOR_NOT_MET" not in gaps,
        "daily_stability_floor": "DAILY_PNL_UNSTABLE" not in gaps,
        "positive_day_rate_floor": "NEEDS_HIGHER_POSITIVE_DAY_RATE" not in gaps,
        "forecast_executable_proof": "FRESH_FORECAST_QUOTE_PROOF_MISSING" not in gaps,
        "geometry_proof": not any("GEOMETRY" in item or "CHASE" in item or "RR" in item for item in gaps | blockers),
        "attached_tp_proof": "ATTACHED_TP_STRUCTURE_PROOF_MISSING" not in gaps,
        "market_close_absence_or_close_gate_proof": "MARKET_CLOSE" not in " ".join(gaps),
        "residual_family_absence": "MONTH_SCALE_RESIDUAL_FAMILY_BLOCKED" not in gaps,
        "risk_engine_pass": "RISK_ENGINE_PASS_MISSING" not in gaps,
        "live_order_gateway_pass": "LIVE_ORDER_GATEWAY_PREFLIGHT_MISSING" not in gaps,
        "gpt_verifier_pass": "FRESH_GPT_VERIFIER_TRADE_RECEIPT_MISSING" not in gaps,
        "no_stale_packaged_rule": "STALE_PACKAGED_USD_JPY_RULE_WITHOUT_FRESH_TP10_SL7_PROOF" not in gaps,
        "no_guardian_operator_review_blocker": "GUARDIAN_OPERATOR_REVIEW_BLOCKER" not in gaps,
    }


def _proof_classification(row: dict[str, Any], missing: dict[str, Any]) -> str:
    if row.get("hard_excluded"):
        return "REJECTED"
    if all(value is True for value in missing.values()):
        return "PROOF_READY"
    if row.get("source_evidence", {}).get("historical_only"):
        return "HISTORICAL_ONLY"
    blockers = set(row.get("current_blockers") or [])
    if any("NEGATIVE_EXPECTANCY" in item or "SPREAD_TOO_WIDE" in item or "STALE_QUOTE" in item for item in blockers):
        return "REPAIR_REQUIRED"
    return "EVIDENCE_GAP"


def _apply_audjpy_limit_fresh_s5(
    row: dict[str, Any],
    missing: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    if not _is_audjpy_limit_target(row) or not artifact:
        return
    thresholds = _nested(artifact, "thresholds", "results") or {}
    exact_pass = bool(_nested(artifact, "thresholds", "meets_exact_s5_proof_thresholds"))
    for key in (
        "sample_count_floor",
        "active_day_floor",
        "daily_stability_floor",
        "positive_day_rate_floor",
    ):
        if key in thresholds:
            missing[key] = thresholds.get(key) is True
    missing["s5_bidask_spread_included_replay"] = exact_pass


def _current_blockers_with_audjpy_limit_fresh_s5(
    row: dict[str, Any],
    artifact: dict[str, Any],
) -> list[str]:
    blockers = [str(item) for item in row.get("current_blockers") or []]
    if _is_audjpy_limit_target(row) and artifact:
        for reason in _nested(artifact, "thresholds", "failed_reasons") or []:
            if str(reason) not in blockers:
                blockers.append(str(reason))
    return blockers


def _is_audjpy_limit_target(row: dict[str, Any]) -> bool:
    return (
        row.get("pair") == "AUD_JPY"
        and row.get("side") == "SHORT"
        and row.get("method") == "BREAKOUT_FAILURE"
        and row.get("order_type") == "LIMIT"
        and row.get("exit_shape") == "TP_PROOF_COLLECTION_HARVEST"
    )


def _source_evidence(metadata: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "capture_economics_status": metadata.get("capture_economics_status"),
        "capture_take_profit_scope_key": metadata.get("capture_take_profit_scope_key"),
        "positive_rotation_mode": metadata.get("positive_rotation_mode"),
        "positive_rotation_proof_collection_ready": metadata.get("positive_rotation_proof_collection_ready"),
        "bidask_rule_name": rule.get("name"),
        "bidask_rule_status": rule.get("adoption_status"),
        "bidask_rule_generated_at_utc": rule.get("rule_set_generated_at_utc"),
        "historical_only": metadata.get("positive_rotation_proof_collection_ready") is not True and rule.get("live_grade") is not True,
    }


def _evidence_rule(metadata: dict[str, Any]) -> dict[str, Any]:
    for key in ("bidask_replay_precision_seed_rule", "bidask_replay_precision_negative"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _exit_shape(metadata: dict[str, Any]) -> str:
    if metadata.get("positive_rotation_mode"):
        return str(metadata.get("positive_rotation_mode"))
    if metadata.get("attach_take_profit_on_fill"):
        return "ATTACHED_TECHNICAL_TP"
    return "UNKNOWN"


def _dependency_graph(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    graph = []
    for row in rows:
        graph.append(
            {
                "node": row["row_code"],
                "classification": row["classification"],
                "blocks": ["A/S LIVE_READY", "normal routing"] if row["clearing_would_affect_as_live_ready"] else [],
                "requires": row["required_proof_to_clear"],
                "can_create_live_permission": False,
            }
        )
    return graph


def _self_improvement_proof(code: str) -> str:
    if code == "MEMORY_HEALTH_BLOCKED":
        return "memory-health must PASS after fresh forecast/quote/order-intent evidence; stale short forecast history cannot support live routing."
    if code == "TARGET_OPEN_NO_LIVE_READY_LANES":
        return "regenerated order_intents must contain at least one LIVE_READY lane after profitability, telemetry, guardian, risk, and gateway gates pass."
    return "self-improvement-audit must regenerate without this P0 code."


def _self_improvement_action(code: str) -> str:
    if code == "MEMORY_HEALTH_BLOCKED":
        return "Refresh forecast/market evidence, regenerate order_intents, rerun memory-health and self-improvement."
    if code == "TARGET_OPEN_NO_LIVE_READY_LANES":
        return "Use the firepower board and proof-pack queue to repair exact lane blockers; do not end with optimism."
    return "Repair the named self-improvement P0 and rerun acceptance."


def _trade_ids_from_segment(segment: dict[str, Any]) -> list[str]:
    ids = [str(x) for x in segment.get("market_close_loss_trade_ids") or []]
    for example in segment.get("market_close_loss_examples") or []:
        trade_id = str(example.get("trade_id") or "")
        if trade_id and trade_id not in ids:
            ids.append(trade_id)
    return ids


def _family_id(item: dict[str, Any]) -> str:
    return f"{item.get('pair') or 'UNKNOWN'} {item.get('side') or 'UNKNOWN'} {item.get('method') or item.get('strategy') or 'UNKNOWN'}"


def _residual_trade_ids(table: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in table.get("families") or []:
        ids.update(str(x) for x in item.get("trade_ids") or [] if str(x))
    return ids


def _market_close_trade_ids(acceptance: dict[str, Any], table: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    family = _findings_by_code(acceptance).get("MARKET_CLOSE_LEAK_FAMILY_BLOCKED") or {}
    ids.update(str(x) for x in _nested(family, "evidence", "system_gateway_loss_trade_ids") or [] if str(x))
    blocker = (table.get("blockers") or {}).get("MARKET_CLOSE_LEAK_DOMINATES_TP_EDGE") or {}
    ids.update(str(x) for x in blocker.get("contributing_trade_ids") or [] if str(x))
    for item in table.get("trades") or []:
        trade_id = str(item.get("trade_id") or "")
        if trade_id:
            ids.add(trade_id)
    return ids


def _top_loss_ids(rows: list[Outcome], limit: int) -> list[dict[str, Any]]:
    losses = sorted((row for row in rows if row.realized_pl_jpy < 0), key=lambda row: row.realized_pl_jpy)
    return [
        {
            "trade_id": row.trade_id,
            "pair": row.pair,
            "side": row.side,
            "method": row.method,
            "exit_reason": row.exit_reason,
            "realized_pl_jpy": _round(row.realized_pl_jpy),
        }
        for row in losses[:limit]
    ]


def _proof_notes(row: dict[str, Any], classification: str) -> str:
    if classification == "REPAIR_REQUIRED":
        return "Candidate has firepower or collection value, but current blockers require repair before proof can become permission."
    if classification == "HISTORICAL_ONLY":
        return "Historical evidence can rank repair work only; fresh executable proof is missing."
    if classification == "EVIDENCE_GAP":
        return "Candidate may enter proof work, but missing proof items must be filled exactly."
    if classification == "PROOF_READY":
        return "All proof flags are present, but live permission still requires regenerated LIVE_READY and current verifier/gateway checks."
    return "Rejected by a hard exclusion."


def _findings_by_code(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("code")): item
        for item in payload.get("findings") or []
        if isinstance(item, dict) and item.get("code")
    }


def _load_json(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _nested(payload: Any, *keys: str) -> Any:
    cur = payload
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _lane_part(lane_id: str, index: int) -> str | None:
    parts = [part for part in str(lane_id or "").split(":") if part]
    if len(parts) > index:
        return parts[index]
    return None


def _float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _p0_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Remaining Profitability P0 Decomposition",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Rows: `{payload.get('summary', {}).get('rows')}`",
        f"- Can create live permission: `{payload.get('summary', {}).get('can_create_live_permission')}`",
        "",
        "| blocker | row | class | trades/families | residual gate | market-close gate | live permission | action |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        refs = ",".join(row.get("trade_ids") or row.get("family_ids") or [])
        lines.append(
            f"| `{row.get('blocker_code')}` | `{row.get('row_code')}` | `{row.get('classification')}` | {refs or 'none'} | `{row.get('residual_family_gate_already_contains_it')}` | `{row.get('market_close_leak_family_already_contains_it')}` | `{row.get('can_create_live_permission')}` | {row.get('proposed_action')} |"
        )
    lines.extend(["", "## Dependency Graph", ""])
    for node in payload.get("dependency_graph") or []:
        lines.append(f"- `{node.get('node')}` -> blocks `{', '.join(node.get('blocks') or [])}`; requires: {node.get('requires')}")
    return "\n".join(lines) + "\n"


def _family_repair_md(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    scope = payload.get("scope") or {}
    metrics = scope.get("metrics") or {}
    lines = [
        "# Post-Gate Gap Family Repair Table",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Scope: `{scope.get('name')}`",
        f"- Scope net / expectancy: `{metrics.get('net_jpy')}` / `{metrics.get('expectancy_jpy_per_trade')}` JPY",
        f"- Negative families: `{summary.get('negative_family_count')}`",
        f"- Largest remaining adverse family: `{summary.get('largest_remaining_adverse_family')}`",
        f"- Can create live permission: `{summary.get('can_create_live_permission_count')}`",
        "",
        "| family | trades | net | exp/trade | exit reason | market-close allowed | blocker | action | live permission |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| `{row.get('family_id')}` | {row.get('trades')} | {row.get('net_jpy')} | {row.get('expectancy_jpy_per_trade')} | `{row.get('exit_reason')}` | `{row.get('market_close_path_allowed')}` | `{row.get('blocker')}` | `{row.get('action')}` | `{row.get('can_create_live_permission')}` |"
        )
    lines.extend(["", "## Trade IDs And Required Proof", ""])
    for row in payload.get("rows") or []:
        ids = ", ".join(str(x) for x in row.get("trade_ids") or [])
        secondaries = ", ".join(row.get("secondary_actions") or [])
        lines.append(f"- `{row.get('family_id')}`: trades `{ids}`")
        lines.append(f"  - P/L: `{row.get('net_jpy')}` JPY; attribution `{row.get('attribution')}`; action `{row.get('action')}`; secondary `{secondaries or 'none'}`")
        lines.append(f"  - Clearance: {row.get('exact_clearance_condition')}")
    return "\n".join(lines) + "\n"


def _post_gate_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Post-Gate Capture Economics Decomposition",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Capture economics status: `{payload.get('capture_economics_status')}`",
        f"- NEGATIVE_EXPECTANCY_ACTIVE should remain: `{payload.get('negative_expectancy_active_should_remain')}`",
        "",
        "| scope | trades | wins | losses | net | exp/trade | PF | avg win | avg loss | max loss | clears negative? | live permission | proof pack |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload.get("scopes") or []:
        lines.append(
            f"| `{row.get('scope')}` | {row.get('trades')} | {row.get('wins')} | {row.get('losses')} | {row.get('net_jpy')} | {row.get('expectancy_jpy_per_trade')} | {row.get('profit_factor')} | {row.get('avg_win_jpy')} | {row.get('avg_loss_jpy')} | {row.get('max_loss_jpy')} | `{row.get('can_clear_negative_expectancy_active')}` | `{row.get('can_create_live_permission')}` | `{row.get('can_enter_proof_pack')}` |"
        )
    lines.extend(["", "## Loss Contributors", ""])
    for row in payload.get("scopes") or []:
        ids = row.get("trade_ids_contributing_most_to_loss") or []
        if ids:
            lines.append(f"- `{row.get('scope')}`: " + ", ".join(f"{item.get('trade_id')} ({item.get('realized_pl_jpy')})" for item in ids[:5]))
    return "\n".join(lines) + "\n"


def _firepower_md(payload: dict[str, Any]) -> str:
    target = payload.get("target_math") or {}
    lines = [
        "# Rolling 30D 4X Firepower Board",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Funding-adjusted equity: `{target.get('funding_adjusted_equity')}` JPY",
        f"- Required calendar daily return: `{target.get('required_calendar_daily_return_funding_adjusted_pct')}`%",
        f"- Candidate rows after hard exclusions: `{payload.get('summary', {}).get('candidate_rows_after_hard_exclusions')}`",
        f"- Rows meeting required daily return prefilter: `{payload.get('summary', {}).get('rows_meeting_required_daily_return_prefilter')}`",
        f"- Can create live permission rows: `0`",
        "",
        "| lane | pair | side | method | order | exit | samples/days | exp JPY | trades/day | daily % | required? | proof gaps |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload.get("candidates") or []:
        lines.append(
            f"| `{row.get('lane_id')}` | {row.get('pair')} | {row.get('side')} | {row.get('method')} | {row.get('order_type')} | {row.get('exit_shape')} | {row.get('sample_count')}/{row.get('active_days')} | {row.get('expected_jpy_per_trade')} | {row.get('estimated_trades_per_day_available')} | {row.get('expected_daily_return_pct_on_funding_adjusted_equity')} | `{row.get('candidate_daily_expected_return_pct_ge_required')}` | {', '.join(row.get('exact_proof_gaps') or [])} |"
        )
    lines.extend(["", "## Hard Exclusions", ""])
    for row in payload.get("hard_exclusions") or []:
        lines.append(f"- `{row.get('lane_id')}`: `{', '.join(row.get('hard_exclusion_reasons') or [])}`")
    return "\n".join(lines) + "\n"


def _proof_queue_md(payload: dict[str, Any]) -> str:
    lines = [
        "# A/S Proof-Pack Queue",
        "",
        f"- Generated: `{payload.get('generated_at_utc')}`",
        f"- Queue count: `{payload.get('summary', {}).get('queue_count')}`",
        f"- PROOF_READY: `{payload.get('summary', {}).get('proof_ready_count')}`",
        f"- Can create live permission: `0`",
        "",
        "| lane | class | daily % | distance | can enter proof pack | blockers |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in payload.get("queue") or []:
        lines.append(
            f"| `{row.get('lane_id')}` | `{row.get('proof_classification')}` | {row.get('expected_daily_return_pct_on_funding_adjusted_equity')} | {row.get('proof_distance')} | `{row.get('can_enter_proof_pack')}` | {', '.join(row.get('current_blockers') or [])} |"
        )
    lines.extend(["", "## Missing Proof", ""])
    for row in payload.get("queue") or []:
        missing = [key for key, value in (row.get("missing_proof") or {}).items() if value is not True]
        lines.append(f"- `{row.get('lane_id')}`: {', '.join(missing) or 'none'}")
    return "\n".join(lines) + "\n"


def _as_board_md(board: dict[str, Any]) -> str:
    target = board.get("rolling_30d_4x_target_math") or {}
    closest = board.get("closest_candidate_to_proof_pack") or {}
    blocker = board.get("exact_blocker_preventing_live_ready") or {}
    lines = [
        "# A/S Lane Candidate Board",
        "",
        f"- Generated: `{board.get('generated_at_utc')}`",
        f"- Total lanes: `{board.get('total_lanes')}`",
        f"- LIVE_READY lanes: `{board.get('live_ready_lanes')}`",
        f"- A/S LIVE_READY path exists: `{board.get('as_live_ready_path_exists')}`",
        f"- Normal routing: `{board.get('normal_routing_status')}`",
        "",
        "## 30D 4X Target Math",
        "",
        f"- Rolling 30d start equity: `{target.get('rolling_30d_start_equity')}`",
        f"- Current raw / broker NAV: `{target.get('current_equity_raw')}`",
        f"- Capital flows 30d: `{target.get('capital_flows_30d')}`",
        f"- Funding-adjusted equity: `{target.get('funding_adjusted_equity')}`",
        f"- Funding-adjusted multiplier: `{target.get('rolling_30d_multiplier_funding_adjusted')}`",
        f"- Target from rolling start 4x: `{target.get('target_equity_from_rolling_start_4x_jpy')}`",
        f"- Prompt-style current funding-adjusted 4x target: `{target.get('prompt_style_target_equity_current_funding_adjusted_4x_jpy')}`",
        f"- Remaining to 4x funding-adjusted: `{target.get('remaining_to_4x_funding_adjusted')}`",
        f"- Required calendar daily return: `{target.get('required_calendar_daily_return_funding_adjusted_pct')}`%",
        "",
        "## Firepower Summary",
        "",
    ]
    for key, value in (board.get("firepower_board_summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Remaining P0 Dependency Graph", ""])
    for node in board.get("remaining_p0_dependency_graph") or []:
        lines.append(f"- `{node.get('node')}`: `{node.get('classification')}`; can create permission `{node.get('can_create_live_permission')}`")
    lines.extend(
        [
            "",
            "## Closest Candidate",
            "",
            f"- Lane: `{closest.get('lane_id')}`",
            f"- Classification: `{closest.get('proof_classification')}`",
            f"- Proof distance: `{closest.get('proof_distance')}`",
            f"- Can create live permission: `{closest.get('can_create_live_permission')}`",
            "",
            "## Exact Blocker Preventing LIVE_READY",
            "",
            f"- Primary: `{blocker.get('primary')}`",
            f"- P0 rows: `{', '.join(blocker.get('p0_rows') or [])}`",
            f"- Global blockers: `{', '.join(blocker.get('global_blockers') or [])}`",
            f"- A/S LIVE_READY stays zero: `{blocker.get('as_live_ready_stays_zero')}`",
            "",
            "## New Evidence Loop Artifacts",
            "",
        ]
    )
    for item in board.get("new_evidence_loop_artifacts") or []:
        lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
