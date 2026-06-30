"""Generate `data/trader_overrides.json` from yesterday's P&L.

The `daily-review` CLI command runs this module once per day to convert
the previous day's realized P&L (read from `execution_ledger.db`) into
direction-bias overrides + blocked-lane hints for tomorrow's trader.

This module closes the "daily-review → trader 知見伝達の断絶" called
out in `project_v8_postmortem_20260327.md`. Module C in trader_brain
already reads `trader_overrides.json`; this is the producer side.

Heuristics (deterministic, env-tunable):

- **Direction bias**: when (pair, direction) has ≥ N_TRADES_FOR_BIAS
  realized trade outcomes with net P&L beyond ±BIAS_NET_PL_THRESHOLD, emit a
  bias_override scaled `tanh(net_pl / SATURATION) × MAX_BIAS`.
- **Lane block**: when a specific lane_id has ≥ N_LOSSES_FOR_BLOCK
  losing realized outcomes in the lookback window, add it to blocked_lanes.
- **Expiry**: next JST 00:00 in UTC, so the override is valid for the
  current trading day only.

The output is JSON; the reader (`trader_overrides.load_trader_overrides`)
gracefully ignores expired or malformed entries.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple


DAILY_REVIEW_LOOKBACK_HOURS = float(os.environ.get("QR_DAILY_REVIEW_LOOKBACK_HOURS", "24"))
DAILY_REVIEW_N_TRADES_FOR_BIAS = int(os.environ.get("QR_DAILY_REVIEW_N_TRADES_FOR_BIAS", "3"))
DAILY_REVIEW_BIAS_PL_THRESHOLD = float(os.environ.get("QR_DAILY_REVIEW_BIAS_PL_THRESHOLD", "1000"))
DAILY_REVIEW_BIAS_SATURATION = float(os.environ.get("QR_DAILY_REVIEW_BIAS_SATURATION", "3000"))
DAILY_REVIEW_MAX_BIAS = float(os.environ.get("QR_DAILY_REVIEW_MAX_BIAS", "20.0"))
DAILY_REVIEW_N_LOSSES_FOR_BLOCK = int(os.environ.get("QR_DAILY_REVIEW_N_LOSSES_FOR_BLOCK", "3"))
# Structural review catches repeatable pair/direction underperformance that can
# disappear from a sparse 24h window. The default 0 means "all available ledger
# history"; expiry still remains next JST midnight, so this is recomputed rather
# than becoming a permanent hard block.
DAILY_REVIEW_STRUCTURAL_LOOKBACK_HOURS = float(os.environ.get("QR_DAILY_REVIEW_STRUCTURAL_LOOKBACK_HOURS", "0"))
DAILY_REVIEW_STRUCTURAL_N_TRADES_FOR_BIAS = int(os.environ.get("QR_DAILY_REVIEW_STRUCTURAL_N_TRADES_FOR_BIAS", "10"))


@dataclass
class DailyReviewReport:
    bias_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    blocked_lanes: list[str] = field(default_factory=list)
    narrative_summary: str = ""
    expires_at_utc: str = ""
    source_window_start_utc: str = ""
    source_window_end_utc: str = ""
    structural_window_start_utc: str = ""
    pair_pl_breakdown: Dict[str, float] = field(default_factory=dict)
    structural_pair_pl_breakdown: Dict[str, float] = field(default_factory=dict)
    structural_pair_counts: Dict[str, int] = field(default_factory=dict)
    lane_loss_counts: Dict[str, int] = field(default_factory=dict)
    target_path_live_reviews: list[dict[str, Any]] = field(default_factory=list)
    user_alpha_trades: list[dict[str, Any]] = field(default_factory=list)
    user_alpha_continuation: dict[str, Any] = field(default_factory=dict)
    market_read_review: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "expires_at_utc": self.expires_at_utc,
            "narrative_summary": self.narrative_summary,
            "bias_overrides": self.bias_overrides,
            "blocked_lanes": self.blocked_lanes,
            "target_path_live_reviews": self.target_path_live_reviews,
            "user_alpha_trades": self.user_alpha_trades,
            "user_alpha_continuation": self.user_alpha_continuation,
            "market_read_review": self.market_read_review,
            "_diagnostics": {
                "source_window_start_utc": self.source_window_start_utc,
                "source_window_end_utc": self.source_window_end_utc,
                "structural_window_start_utc": self.structural_window_start_utc,
                "pair_direction_net_pl_jpy": {
                    f"{k[0]}:{k[1]}": v for k, v in self.pair_pl_breakdown.items()
                } if all(isinstance(k, tuple) for k in self.pair_pl_breakdown.keys()) else self.pair_pl_breakdown,
                "structural_pair_direction_net_pl_jpy": {
                    f"{k[0]}:{k[1]}": v for k, v in self.structural_pair_pl_breakdown.items()
                } if all(isinstance(k, tuple) for k in self.structural_pair_pl_breakdown.keys()) else self.structural_pair_pl_breakdown,
                "structural_pair_direction_counts": {
                    f"{k[0]}:{k[1]}": v for k, v in self.structural_pair_counts.items()
                } if all(isinstance(k, tuple) for k in self.structural_pair_counts.keys()) else self.structural_pair_counts,
                "lane_loss_counts": self.lane_loss_counts,
                "target_path_live_review_counts": _target_path_live_review_counts(self.target_path_live_reviews),
                "user_alpha_counts": _user_alpha_counts(self.user_alpha_trades),
                "market_read_review": self.market_read_review,
            },
        }
        return out


def _next_jst_midnight_utc(now: datetime | None = None) -> datetime:
    """Compute the next JST 00:00 in UTC. JST is UTC+9."""
    now = now or datetime.now(timezone.utc)
    jst_now = now + timedelta(hours=9)
    jst_midnight = jst_now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return jst_midnight - timedelta(hours=9)


def _read_recent_closes(
    db_path: Path,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> list[tuple[str, str, float, str | None]]:
    """Return bot-attributed (pair, original_side, realized_pl_jpy, lane_id)
    tuples for realized trade outcomes inside the window."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    rows: list[tuple[str, str, float, str | None]] = []
    try:
        cur = conn.execute(
            """
            WITH gateway_entries AS (
                SELECT
                    trade_id,
                    order_id,
                    lane_id
                FROM execution_events
                WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
                  AND lane_id IS NOT NULL
                  AND lane_id != ''
            ),
            entries AS (
                SELECT
                    e.trade_id,
                    COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS gateway_lane_id,
                    CASE
                        WHEN SUM(CASE WHEN UPPER(COALESCE(e.side, '')) = 'LONG' THEN 1 ELSE 0 END) > 0 THEN 'LONG'
                        WHEN SUM(CASE WHEN UPPER(COALESCE(e.side, '')) = 'SHORT' THEN 1 ELSE 0 END) > 0 THEN 'SHORT'
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
            ),
            realized AS (
                SELECT
                    COALESCE(NULLIF(e.trade_id, ''), e.event_uid) AS outcome_id,
                    e.ts_utc,
                    e.pair,
                    COALESCE(NULLIF(e.lane_id, ''), entries.gateway_lane_id) AS lane_id,
                    entries.position_side AS original_side,
                    e.realized_pl_jpy
                FROM execution_events e
                INNER JOIN entries ON entries.trade_id = e.trade_id
                WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                  AND e.realized_pl_jpy IS NOT NULL
                  AND e.pair IS NOT NULL
                  AND e.ts_utc >= ?
                  AND e.ts_utc <= ?
            )
            SELECT
                pair,
                original_side,
                SUM(realized_pl_jpy) AS realized_pl_jpy,
                lane_id,
                MAX(ts_utc) AS ts_utc
            FROM realized
            WHERE original_side IS NOT NULL
            GROUP BY outcome_id, pair, original_side, lane_id
            ORDER BY ts_utc ASC
            """,
            (
                window_start_utc.isoformat().replace("+00:00", "Z"),
                window_end_utc.isoformat().replace("+00:00", "Z"),
            ),
        )
        for pair, original_side, pl, lane_id, _ts in cur.fetchall():
            rows.append((str(pair), original_side, float(pl), lane_id))
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return rows


def _read_live_target_path_reviews(
    db_path: Path,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> list[dict[str, Any]]:
    """Classify LIVE-LEARNING target-path sends from gateway receipts.

    The gateway receipt is the immutable source for the intended path/size.
    Broker close rows may arrive later, so missing realized P/L is kept as a
    deployment failure instead of being silently dropped from daily review.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    reviews: list[dict[str, Any]] = []
    try:
        rows = conn.execute(
            """
            SELECT
                event_uid,
                ts_utc,
                lane_id,
                order_id,
                trade_id,
                pair,
                side,
                units,
                raw_json
            FROM execution_events
            WHERE event_type = 'GATEWAY_ORDER_SENT'
              AND ts_utc >= ?
              AND ts_utc <= ?
            ORDER BY ts_utc ASC
            """,
            (
                window_start_utc.isoformat().replace("+00:00", "Z"),
                window_end_utc.isoformat().replace("+00:00", "Z"),
            ),
        ).fetchall()
        for row in rows:
            payload = _json_object(row["raw_json"])
            receipt = payload.get("target_path_receipt") if isinstance(payload.get("target_path_receipt"), dict) else None
            if receipt is None:
                continue
            if receipt.get("live_order_sent") is False:
                continue
            outcome = _target_path_live_outcome(
                conn,
                trade_id=_text_value(row["trade_id"]),
                lane_id=_text_value(row["lane_id"]),
                order_id=_text_value(row["order_id"]),
                sent_ts_utc=str(row["ts_utc"] or ""),
                window_end_utc=window_end_utc,
            )
            classification = _classify_live_target_path_review(receipt, outcome)
            reviews.append(
                {
                    "classification": classification,
                    "event_uid": row["event_uid"],
                    "sent_at_utc": row["ts_utc"],
                    "lane_id": row["lane_id"],
                    "order_id": row["order_id"],
                    "trade_id": row["trade_id"],
                    "pair": row["pair"],
                    "side": row["side"],
                    "daily_target_mode": _text_value(receipt.get("daily_target_mode")),
                    "five_pct_path_role": _text_value(receipt.get("five_pct_path_role")),
                    "attack_stack_slot": _text_value(receipt.get("attack_stack_slot")),
                    "grade": _text_value(receipt.get("grade")),
                    "suggested_units": _int_value(receipt.get("suggested_units")),
                    "final_units": _int_value(receipt.get("final_units")),
                    "risk_yen": _float_value(receipt.get("risk_yen")),
                    "risk_pct": _float_value(receipt.get("risk_pct")),
                    "target_yen": _float_value(receipt.get("target_yen")),
                    "contribution_to_5pct": _float_value(receipt.get("contribution_to_5pct")),
                    "remaining_to_5pct": _float_value(receipt.get("remaining_to_5pct")),
                    "live_order_gateway_receipt_id": _text_value(receipt.get("live_order_gateway_receipt_id")),
                    "target_path_live_mode": _text_value(receipt.get("target_path_live_mode") or "LIVE_LEARNING"),
                    "realized_pl_jpy": outcome.get("realized_pl_jpy"),
                    "exit_reason": outcome.get("exit_reason"),
                    "closed_at_utc": outcome.get("closed_at_utc"),
                }
            )
    except (sqlite3.Error, TypeError, ValueError):
        return reviews
    finally:
        conn.close()
    return reviews


def _read_user_alpha_trades(
    db_path: Path,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> list[dict[str, Any]]:
    """Return profitable manual/operator outcomes without bot gateway attribution.

    A user-alpha outcome is deliberately kept out of system P&L bias. It is an
    operator-discovered edge that the trader must either continue or block with
    a named current reason.
    """
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        columns = _table_columns(conn, "execution_events")
        close_rows = conn.execute(
            f"""
            SELECT
                event_uid,
                ts_utc,
                COALESCE(NULLIF(trade_id, ''), event_uid) AS outcome_id,
                trade_id,
                order_id,
                lane_id,
                pair,
                side,
                units,
                realized_pl_jpy,
                {_optional_select_column(columns, "exit_reason")},
                {_optional_select_column(columns, "price")},
                {_optional_select_column(columns, "tp")},
                raw_json
            FROM execution_events
            WHERE event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED', 'GATEWAY_TRADE_CLOSE_RECONCILED')
              AND realized_pl_jpy IS NOT NULL
              AND pair IS NOT NULL
              AND ts_utc >= ?
              AND ts_utc <= ?
            ORDER BY ts_utc ASC
            """,
            (
                window_start_utc.isoformat().replace("+00:00", "Z"),
                window_end_utc.isoformat().replace("+00:00", "Z"),
            ),
        ).fetchall()
        if not close_rows:
            return []
        gateway_trade_ids, gateway_order_ids = _gateway_attribution_ids(conn)
        trade_ids = sorted(
            {
                str(row["trade_id"])
                for row in close_rows
                if _text_value(row["trade_id"])
            }
        )
        entry_rows: dict[str, list[sqlite3.Row]] = {}
        if trade_ids:
            placeholders = ",".join("?" for _ in trade_ids)
            entries = conn.execute(
                f"""
                SELECT
                    event_uid,
                    ts_utc,
                    trade_id,
                    order_id,
                    lane_id,
                    pair,
                    side,
                    units,
                    realized_pl_jpy,
                    {_optional_select_column(columns, "price")},
                    {_optional_select_column(columns, "tp")},
                    raw_json
                FROM execution_events
                WHERE event_type = 'ORDER_FILLED'
                  AND trade_id IN ({placeholders})
                ORDER BY ts_utc ASC
                """,
                trade_ids,
            ).fetchall()
            for row in entries:
                trade_id = _text_value(row["trade_id"])
                if trade_id:
                    entry_rows.setdefault(trade_id, []).append(row)

        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in close_rows:
            outcome_id = str(row["outcome_id"] or row["event_uid"])
            grouped.setdefault(outcome_id, []).append(row)

        trades: list[dict[str, Any]] = []
        for outcome_id, rows in grouped.items():
            trade_id = _text_value(rows[-1]["trade_id"])
            order_ids = {_text_value(row["order_id"]) for row in rows}
            order_ids.discard(None)
            entries = entry_rows.get(trade_id or "", [])
            entry_order_ids = {_text_value(row["order_id"]) for row in entries}
            entry_order_ids.discard(None)
            if trade_id and trade_id in gateway_trade_ids:
                continue
            if order_ids.intersection(gateway_order_ids) or entry_order_ids.intersection(gateway_order_ids):
                continue
            realized = sum(float(row["realized_pl_jpy"] or 0.0) for row in rows)
            if realized <= 0:
                continue
            pair = _text_value(rows[-1]["pair"])
            if pair is None:
                continue
            entry = entries[0] if entries else None
            direction = (
                _original_side_from_entry_row(entry)
                if entry is not None
                else _original_side_from_close_side(_text_value(rows[-1]["side"]))
            )
            if direction not in {"LONG", "SHORT"}:
                continue
            opened_at = _text_value(entry["ts_utc"]) if entry is not None else None
            closed_at = max(str(row["ts_utc"] or "") for row in rows)
            close_payload = _json_object(rows[-1]["raw_json"])
            entry_payload = _json_object(entry["raw_json"]) if entry is not None else {}
            tp = _first_float(
                rows[-1]["tp"],
                entry["tp"] if entry is not None else None,
                close_payload.get("tp"),
                entry_payload.get("tp"),
                _nested_value(entry_payload, ("takeProfitOnFill", "price")),
            )
            exit_reason = _text_value(rows[-1]["exit_reason"]) or _text_value(close_payload.get("exit_reason"))
            thesis = _first_text(
                close_payload.get("thesis"),
                entry_payload.get("thesis"),
                _nested_value(entry_payload, ("market_context", "thesis")),
                _nested_value(close_payload, ("market_context", "thesis")),
            )
            alpha_type = "OPERATOR_ALPHA"
            trade = {
                "edge_source": "USER_ALPHA",
                "classification": alpha_type,
                "discovered_by": "OPERATOR",
                "system_discovered": False,
                "system_tp_managed": _exit_reason_is_tp(exit_reason) or tp is not None,
                "outcome_id": outcome_id,
                "trade_id": trade_id,
                "pair": pair,
                "direction": direction,
                "entry": _first_float(
                    entry["price"] if entry is not None else None,
                    entry_payload.get("price"),
                ),
                "tp": tp,
                "realized_pl_jpy": round(realized, 2),
                "max_favorable_excursion": _first_float(
                    close_payload.get("mfe"),
                    close_payload.get("max_favorable_excursion"),
                    close_payload.get("max_favorable_excursion_jpy"),
                ),
                "time_to_tp_seconds": _elapsed_seconds(opened_at, closed_at),
                "opened_at_utc": opened_at,
                "closed_at_utc": closed_at,
                "exit_reason": exit_reason,
                "thesis": thesis,
                "operator_found_system_tp_managed": True,
                "continuation_required": True,
            }
            trades.append(trade)
        trades.sort(key=lambda item: str(item.get("closed_at_utc") or ""))
        return trades
    except (sqlite3.Error, TypeError, ValueError):
        return []
    finally:
        conn.close()


def _user_alpha_continuation_packet(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "status": "NONE",
            "active": False,
            "required_user_alpha_continuation_block": True,
        }
    latest = trades[-1]
    pair = _text_value(latest.get("pair"))
    direction = _text_value(latest.get("direction"))
    return {
        "status": "ACTIVE",
        "active": True,
        "source": "daily_review",
        "edge_source": "USER_ALPHA",
        "latest_trade": latest,
        "five_pct_path_board_candidate": {
            "source": "USER_ALPHA",
            "pair": pair,
            "direction": direction,
            "candidate_roles": ["RELOAD", "SECOND_SHOT"],
            "target_layer": "PACE_5",
            "reason": "profitable operator-discovered/user-led winner must be evaluated for continuation",
        },
        "required_trader_answers": [
            "thesis_alive",
            "reload_candidate",
            "second_shot_candidate",
            "exact_blocker_if_no_continuation",
            "next_trigger",
        ],
        "if_no_continuation_requires_exact_blocker": True,
        "negative_expectancy_scope": (
            "NEGATIVE_EXPECTANCY on system-generated edge does not erase proven USER_ALPHA; "
            "the trader must cite an explicit current blocker."
        ),
    }


def _target_path_live_outcome(
    conn: sqlite3.Connection,
    *,
    trade_id: str | None,
    lane_id: str | None,
    order_id: str | None,
    sent_ts_utc: str,
    window_end_utc: datetime,
) -> dict[str, Any]:
    conditions: list[str] = []
    params: list[Any] = []
    if trade_id:
        conditions.append("trade_id = ?")
        params.append(trade_id)
    if lane_id:
        conditions.append("lane_id = ?")
        params.append(lane_id)
    if order_id:
        conditions.append("order_id = ?")
        params.append(order_id)
    if not conditions:
        return {}
    params.extend([sent_ts_utc, window_end_utc.isoformat().replace("+00:00", "Z")])
    try:
        row = conn.execute(
            f"""
            SELECT
                SUM(realized_pl_jpy) AS realized_pl_jpy,
                MAX(exit_reason) AS exit_reason,
                MAX(ts_utc) AS closed_at_utc
            FROM execution_events
            WHERE realized_pl_jpy IS NOT NULL
              AND event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED', 'GATEWAY_TRADE_CLOSE_RECONCILED')
              AND ({' OR '.join(conditions)})
              AND ts_utc >= ?
              AND ts_utc <= ?
            """,
            params,
        ).fetchone()
    except sqlite3.Error:
        return {}
    if row is None or row["realized_pl_jpy"] is None:
        return {}
    return {
        "realized_pl_jpy": float(row["realized_pl_jpy"]),
        "exit_reason": _text_value(row["exit_reason"]),
        "closed_at_utc": _text_value(row["closed_at_utc"]),
    }


def _classify_live_target_path_review(receipt: dict[str, Any], outcome: dict[str, Any]) -> str:
    role = _text_value(receipt.get("five_pct_path_role"))
    slot = _text_value(receipt.get("attack_stack_slot"))
    grade = _text_value(receipt.get("grade"))
    if not role or not slot or not grade:
        return "discovery failure"

    suggested_units = _int_value(receipt.get("suggested_units"))
    final_units = _int_value(receipt.get("final_units"))
    contribution = _float_value(receipt.get("contribution_to_5pct"))
    if (
        suggested_units is None
        or suggested_units <= 0
        or final_units is None
        or final_units <= 0
        or final_units < int(suggested_units * 0.5)
        or contribution is None
        or contribution <= 0
    ):
        return "sizing failure"

    realized = _float_value(outcome.get("realized_pl_jpy"))
    if realized is None:
        return "deployment failure"
    if realized >= 0:
        return "good execution"

    risk_yen = _float_value(receipt.get("risk_yen"))
    exit_reason = (_text_value(outcome.get("exit_reason")) or "").upper()
    if (risk_yen is not None and abs(realized) > risk_yen * 1.05) or any(
        token in exit_reason for token in ("STOP", "SL", "MARKET", "CLOSE", "GPT")
    ):
        return "management failure"
    return "vehicle failure"


def _target_path_live_review_counts(reviews: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for review in reviews:
        classification = str(review.get("classification") or "unknown")
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def _user_alpha_counts(trades: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        classification = str(trade.get("classification") or "USER_ALPHA")
        counts[classification] = counts.get(classification, 0) + 1
    return counts


def _market_read_review(
    path: Path | None,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, Any]:
    base = {
        "status": "NO_MARKET_READ_SCORE_PATH",
        "total_predictions": 0,
        "resolved_predictions": 0,
        "pending_predictions": 0,
        "verdict_counts": {},
        "accuracy_30m_pct": None,
        "accuracy_2h_pct": None,
        "full_read_accuracy_pct": None,
        "blocked_but_correct_read_count": 0,
        "wrong_read_traded_count": 0,
        "best_trade_if_forced_correct_count": 0,
        "best_trade_if_forced_wrong_count": 0,
        "codex_vs_operator_manual_trade": "UNKNOWN_NO_OPERATOR_MANUAL_COMPARISON",
        "examples": [],
    }
    if path is None:
        return base
    if not path.exists():
        out = dict(base)
        out["status"] = "MISSING"
        out["path"] = str(path)
        return out

    rows: list[dict[str, Any]] = []
    malformed = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(payload, dict):
                    malformed += 1
                    continue
                generated_at = _parse_utc(payload.get("generated_at_utc"))
                if generated_at is None or generated_at < window_start or generated_at > window_end:
                    continue
                rows.append(payload)
    except OSError as exc:
        out = dict(base)
        out.update({"status": "UNREADABLE", "path": str(path), "error": str(exc)})
        return out

    verdict_counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict") or "PENDING")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    resolved = [row for row in rows if str(row.get("verdict") or "PENDING") != "PENDING"]
    pending = [row for row in rows if str(row.get("verdict") or "PENDING") == "PENDING"]
    resolved_30m = [row for row in rows if str(row.get("thirty_minute_verdict") or "PENDING") != "PENDING"]
    resolved_2h = [row for row in rows if str(row.get("two_hour_verdict") or "PENDING") != "PENDING"]

    blocked_correct = [
        row for row in resolved
        if row.get("verdict") == "CORRECT"
        and (row.get("action") != "TRADE" or row.get("verification_status") != "ACCEPTED")
    ]
    wrong_traded = [
        row for row in resolved
        if row.get("action") == "TRADE"
        and row.get("verification_status") == "ACCEPTED"
        and row.get("verdict") in {"WRONG", "INVALIDATED_FIRST"}
    ]
    forced_correct = [row for row in resolved if row.get("verdict") in {"CORRECT", "MIXED"}]
    forced_wrong = [row for row in resolved if row.get("verdict") in {"WRONG", "INVALIDATED_FIRST"}]

    examples = []
    for row in rows[-8:]:
        examples.append(
            {
                "generated_at_utc": row.get("generated_at_utc"),
                "pair": row.get("pair"),
                "direction": row.get("direction"),
                "action": row.get("action"),
                "verification_status": row.get("verification_status"),
                "verdict": row.get("verdict"),
                "thirty_minute_verdict": row.get("thirty_minute_verdict"),
                "two_hour_verdict": row.get("two_hour_verdict"),
            }
        )

    return {
        "status": "OK" if not malformed else "WARN_MALFORMED_ROWS",
        "path": str(path),
        "malformed_rows": malformed,
        "total_predictions": len(rows),
        "resolved_predictions": len(resolved),
        "pending_predictions": len(pending),
        "verdict_counts": verdict_counts,
        "accuracy_30m_pct": _pct(
            sum(1 for row in resolved_30m if row.get("thirty_minute_verdict") == "CORRECT"),
            len(resolved_30m),
        ),
        "accuracy_2h_pct": _pct(
            sum(1 for row in resolved_2h if row.get("two_hour_verdict") == "CORRECT"),
            len(resolved_2h),
        ),
        "full_read_accuracy_pct": _pct(
            sum(1 for row in resolved if row.get("verdict") == "CORRECT"),
            len(resolved),
        ),
        "blocked_but_correct_read_count": len(blocked_correct),
        "wrong_read_traded_count": len(wrong_traded),
        "best_trade_if_forced_correct_count": len(forced_correct),
        "best_trade_if_forced_wrong_count": len(forced_wrong),
        "codex_vs_operator_manual_trade": "UNKNOWN_NO_OPERATOR_MANUAL_COMPARISON",
        "examples": examples,
    }


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100.0, 1)


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    except sqlite3.Error:
        return set()


def _optional_select_column(columns: set[str], name: str) -> str:
    return name if name in columns else f"NULL AS {name}"


def _gateway_attribution_ids(conn: sqlite3.Connection) -> tuple[set[str], set[str]]:
    try:
        rows = conn.execute(
            """
            SELECT trade_id, order_id
            FROM execution_events
            WHERE event_type IN ('GATEWAY_ORDER_SENT', 'ORDER_ACCEPTED')
              AND lane_id IS NOT NULL
              AND lane_id != ''
            """
        ).fetchall()
    except sqlite3.Error:
        return set(), set()
    trade_ids: set[str] = set()
    order_ids: set[str] = set()
    for row in rows:
        trade_id = _text_value(row["trade_id"] if isinstance(row, sqlite3.Row) else row[0])
        order_id = _text_value(row["order_id"] if isinstance(row, sqlite3.Row) else row[1])
        if trade_id:
            trade_ids.add(trade_id)
        if order_id:
            order_ids.add(order_id)
    return trade_ids, order_ids


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _text_value(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any) -> int | None:
    parsed = _float_value(value)
    return int(parsed) if parsed is not None else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float_value(value)
        if parsed is not None:
            return parsed
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        parsed = _text_value(value)
        if parsed:
            return parsed
    return None


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _parse_utc(value: Any) -> datetime | None:
    text = _text_value(value)
    if text is None:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _elapsed_seconds(start: Any, end: Any) -> float | None:
    start_dt = _parse_utc(start)
    end_dt = _parse_utc(end)
    if start_dt is None or end_dt is None:
        return None
    seconds = (end_dt - start_dt).total_seconds()
    return round(seconds, 3) if seconds >= 0 else None


def _original_side_from_close_side(side: str | None) -> str | None:
    side_upper = str(side or "").upper()
    if side_upper == "LONG":
        return "SHORT"
    if side_upper == "SHORT":
        return "LONG"
    return None


def _original_side_from_entry_row(row: sqlite3.Row | None) -> str | None:
    if row is None:
        return None
    side = str(row["side"] or "").upper()
    if side in {"LONG", "SHORT"}:
        return side
    units = _float_value(row["units"])
    if units is None:
        return None
    return "LONG" if units > 0 else "SHORT" if units < 0 else None


def _exit_reason_is_tp(exit_reason: str | None) -> bool:
    upper = str(exit_reason or "").upper()
    return "TAKE_PROFIT" in upper or upper in {"TP", "TAKE PROFIT"} or " TP" in f" {upper}"


def _aggregate_rows(
    rows: list[tuple[str, str, float, str | None]],
) -> tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], int], Dict[str, int]]:
    pair_pl: Dict[Tuple[str, str], float] = {}
    pair_count: Dict[Tuple[str, str], int] = {}
    lane_losses: Dict[str, int] = {}
    for pair, direction, pl, lane_id in rows:
        key = (pair, direction)
        pair_pl[key] = pair_pl.get(key, 0.0) + pl
        pair_count[key] = pair_count.get(key, 0) + 1
        if lane_id and pl < 0:
            lane_losses[lane_id] = lane_losses.get(lane_id, 0) + 1
    return pair_pl, pair_count, lane_losses


def _structural_window_start(now: datetime) -> datetime:
    if DAILY_REVIEW_STRUCTURAL_LOOKBACK_HOURS <= 0:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return now - timedelta(hours=DAILY_REVIEW_STRUCTURAL_LOOKBACK_HOURS)


def compute_daily_review(
    db_path: Path,
    *,
    now: datetime | None = None,
    lookback_hours: float = DAILY_REVIEW_LOOKBACK_HOURS,
    market_read_score_path: Path | None = None,
) -> DailyReviewReport:
    """Build a DailyReviewReport from recent closed trades."""
    now = now or datetime.now(timezone.utc)
    window_end = now
    window_start = now - timedelta(hours=lookback_hours)

    rows = _read_recent_closes(db_path, window_start, window_end)
    structural_start = _structural_window_start(now)
    structural_rows = _read_recent_closes(db_path, structural_start, window_end)
    target_path_live_reviews = _read_live_target_path_reviews(db_path, window_start, window_end)
    user_alpha_trades = _read_user_alpha_trades(db_path, window_start, window_end)
    user_alpha_continuation = _user_alpha_continuation_packet(user_alpha_trades)
    market_read = _market_read_review(market_read_score_path, window_start, window_end)

    # Aggregate per (pair, direction)
    pair_pl, pair_count, lane_losses = _aggregate_rows(rows)
    structural_pair_pl, structural_pair_count, _structural_lane_losses = _aggregate_rows(structural_rows)

    # Direction bias: per (pair, direction) where count ≥ N and |net_pl| beyond threshold
    bias_overrides: Dict[str, Dict[str, float]] = {}
    for (pair, direction), net_pl in pair_pl.items():
        count = pair_count[(pair, direction)]
        if count < DAILY_REVIEW_N_TRADES_FOR_BIAS:
            continue
        if abs(net_pl) < DAILY_REVIEW_BIAS_PL_THRESHOLD:
            continue
        magnitude = DAILY_REVIEW_MAX_BIAS * math.tanh(abs(net_pl) / DAILY_REVIEW_BIAS_SATURATION)
        sign = 1.0 if net_pl > 0 else -1.0
        bias_overrides.setdefault(pair, {})[direction] = round(sign * magnitude, 2)

    # Structural negative bias: if the short tactical window is sparse or quiet,
    # still tell trader_brain about pair/direction loss contributors with enough
    # realized samples. This is a daily expiring score penalty, not a hard block.
    for (pair, direction), net_pl in structural_pair_pl.items():
        if direction in bias_overrides.get(pair, {}):
            continue
        count = structural_pair_count[(pair, direction)]
        if count < DAILY_REVIEW_STRUCTURAL_N_TRADES_FOR_BIAS:
            continue
        if net_pl >= 0:
            continue
        magnitude = DAILY_REVIEW_MAX_BIAS * math.tanh(abs(net_pl) / DAILY_REVIEW_BIAS_SATURATION)
        bias_overrides.setdefault(pair, {})[direction] = round(-magnitude, 2)

    # Blocked lanes
    blocked = [
        lane_id for lane_id, losses in lane_losses.items()
        if losses >= DAILY_REVIEW_N_LOSSES_FOR_BLOCK
    ]

    # Narrative summary (1-2 lines)
    losers = [
        f"{k[0]}:{k[1]} {v:+.0f}JPY"
        for k, v in sorted(pair_pl.items(), key=lambda x: x[1])[:3]
        if v < -DAILY_REVIEW_BIAS_PL_THRESHOLD
    ]
    winners = [
        f"{k[0]}:{k[1]} {v:+.0f}JPY"
        for k, v in sorted(pair_pl.items(), key=lambda x: -x[1])[:3]
        if v > DAILY_REVIEW_BIAS_PL_THRESHOLD
    ]
    structural_losers = [
        f"{k[0]}:{k[1]} {v:+.0f}JPY/{structural_pair_count[k]}trades"
        for k, v in sorted(structural_pair_pl.items(), key=lambda x: x[1])[:3]
        if v < 0 and structural_pair_count[k] >= DAILY_REVIEW_STRUCTURAL_N_TRADES_FOR_BIAS
    ]
    parts: list[str] = []
    if losers:
        parts.append("losing: " + ", ".join(losers))
    if winners:
        parts.append("winning: " + ", ".join(winners))
    if structural_losers:
        parts.append("structural losing: " + ", ".join(structural_losers))
    target_path_counts = _target_path_live_review_counts(target_path_live_reviews)
    if target_path_counts:
        parts.append(
            "target-path live: "
            + ", ".join(f"{name}={count}" for name, count in sorted(target_path_counts.items()))
        )
    if user_alpha_trades:
        latest_alpha = user_alpha_trades[-1]
        parts.append(
            "user-alpha: "
            f"{latest_alpha.get('pair')}:{latest_alpha.get('direction')} "
            f"{float(latest_alpha.get('realized_pl_jpy') or 0.0):+.0f}JPY "
            f"({latest_alpha.get('classification') or 'USER_ALPHA'})"
        )
    if market_read.get("total_predictions"):
        parts.append(
            "market-read: "
            f"{market_read.get('resolved_predictions', 0)}/{market_read.get('total_predictions', 0)} resolved "
            f"full_accuracy={market_read.get('full_read_accuracy_pct')}"
        )
    if not parts:
        parts.append(f"no decisive (pair, direction) signal in last {lookback_hours:.0f}h")
    narrative = "; ".join(parts)

    expires = _next_jst_midnight_utc(now)

    return DailyReviewReport(
        bias_overrides=bias_overrides,
        blocked_lanes=blocked,
        narrative_summary=narrative,
        expires_at_utc=expires.isoformat().replace("+00:00", "Z"),
        source_window_start_utc=window_start.isoformat().replace("+00:00", "Z"),
        source_window_end_utc=window_end.isoformat().replace("+00:00", "Z"),
        structural_window_start_utc=structural_start.isoformat().replace("+00:00", "Z"),
        pair_pl_breakdown={k: round(v, 2) for k, v in pair_pl.items()},
        structural_pair_pl_breakdown={k: round(v, 2) for k, v in structural_pair_pl.items()},
        structural_pair_counts=structural_pair_count,
        lane_loss_counts=lane_losses,
        target_path_live_reviews=target_path_live_reviews,
        user_alpha_trades=user_alpha_trades,
        user_alpha_continuation=user_alpha_continuation,
        market_read_review=market_read,
    )


def write_trader_overrides(
    report: DailyReviewReport,
    output_path: Path,
) -> None:
    """Serialize the report to `data/trader_overrides.json` shape.

    The reader (`strategy/trader_overrides.load_trader_overrides`)
    ignores the `_diagnostics` key, so we keep it for human review
    without affecting trader behavior.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
