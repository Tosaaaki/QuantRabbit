#!/usr/bin/env python3
"""Shared close-discipline helpers for discretionary trader inventory."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "collab_trade" / "memory" / "memory.db"
LOG_PATH = ROOT / "logs" / "live_trade_log.txt"

SOFT_REGIMES = {"RANGE", "SQUEEZE", "QUIET"}
FLOW_NEUTRAL_LABELS = {"", "FLOW_NEUTRAL", "unknown", "UNKNOWN"}


def classify_dead_layer(reason: str | None) -> str:
    text = str(reason or "").lower()
    if not text:
        return "unknown"
    if any(token in text for token in ("stale", "zombie", "recycle", "aged", "wish_distance")):
        return "aging"
    if any(token in text for token in ("spread", "slippage", "friction", "bad_fill", "too wide")):
        return "vehicle"
    if any(token in text for token in ("acceptance_above_entry", "accept", "body break", "structure", "shelf broke")):
        return "structure"
    if any(
        token in text
        for token in (
            "m1_pulse_flip",
            "no_first_confirmation",
            "no_reclaim",
            "shelf_fail",
            "no_rebuy",
            "failed_break",
            "failed_floor",
            "false_break",
            "retest",
            "reclaim",
            "trigger",
        )
    ):
        return "trigger"
    if any(token in text for token in ("macro", "market", "flow flip", "risk-off", "risk on", "usd bid", "usd offer")):
        return "market"
    return "unknown"


def _query_rows(sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def load_pretrade_meta(trade_id: str | int | None) -> dict[str, Any] | None:
    if trade_id is None:
        return None
    rows = _query_rows(
        """
        SELECT trade_id, pair, direction, execution_style, thesis_family, thesis_market,
               thesis_structure, thesis_trigger, thesis_vehicle, thesis_age, allocation_band
        FROM pretrade_outcomes
        WHERE trade_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (str(trade_id),),
    )
    return dict(rows[0]) if rows else None


def load_pretrade_meta_bulk(trade_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not trade_ids:
        return {}
    placeholders = ",".join("?" for _ in trade_ids)
    rows = _query_rows(
        f"""
        SELECT trade_id, pair, direction, execution_style, thesis_family, thesis_market,
               thesis_structure, thesis_trigger, thesis_vehicle, thesis_age, allocation_band
        FROM pretrade_outcomes
        WHERE trade_id IN ({placeholders})
        ORDER BY id DESC
        """,
        tuple(str(tid) for tid in trade_ids),
    )
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = str(row["trade_id"])
        if tid in latest:
            continue
        latest[tid] = dict(row)
    return latest


def parse_meta_shape(meta: dict[str, Any] | None) -> dict[str, str]:
    if not meta:
        return {"archetype": "", "wave": "", "session": "", "regime": ""}

    family = str(meta.get("thesis_family") or "")
    parts = [part for part in family.split("|") if part]
    if len(parts) >= 6:
        return {
            "archetype": parts[2].upper(),
            "wave": parts[3].upper(),
            "session": parts[4].replace("SESSION_", "").upper(),
            "regime": parts[5].replace("REGIME_", "").upper(),
        }

    structure = str(meta.get("thesis_structure") or "").upper().split("|")
    return {
        "archetype": structure[0] if len(structure) >= 1 else "",
        "wave": structure[1] if len(structure) >= 2 else "",
        "session": "",
        "regime": "",
    }


def entry_spread_pips_from_log(trade_id: str | int | None) -> float | None:
    if trade_id is None or not LOG_PATH.exists():
        return None
    tid = str(trade_id)
    pattern = re.compile(rf"\bid={re.escape(tid)}\b.*?\bSp=(?P<spread>[\d.]+)pip\b")
    for line in LOG_PATH.read_text().splitlines():
        if "ENTRY " not in line and "ENTRY_ORDER " not in line:
            continue
        match = pattern.search(line)
        if match:
            try:
                return float(match.group("spread"))
            except ValueError:
                return None
    return None


def suggested_half_units(units: int) -> int:
    abs_units = abs(int(units or 0))
    if abs_units <= 1000:
        return abs_units
    rounded = max(1000, int(round(abs_units / 2 / 1000.0) * 1000))
    return min(abs_units - 1, rounded) if rounded >= abs_units else rounded


def inventory_family_summary(
    current_trade_id: str | int | None,
    direction: str,
    current_meta: dict[str, Any] | None,
    open_trades: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not current_meta or not open_trades:
        return None
    thesis_market = str(current_meta.get("thesis_market") or "")
    if thesis_market in FLOW_NEUTRAL_LABELS:
        return None

    open_ids = [str(trade.get("id")) for trade in open_trades if trade.get("id") is not None]
    bulk = load_pretrade_meta_bulk(open_ids)

    siblings: list[dict[str, Any]] = []
    for trade in open_trades:
        tid = str(trade.get("id"))
        if tid == str(current_trade_id):
            continue
        side = "LONG" if float(trade.get("currentUnits", 0) or 0) > 0 else "SHORT"
        if side != direction:
            continue
        meta = bulk.get(tid)
        if not meta:
            continue
        if str(meta.get("thesis_market") or "") != thesis_market:
            continue
        siblings.append(
            {
                "trade_id": tid,
                "pair": trade.get("instrument", ""),
                "units": abs(int(float(trade.get("currentUnits", 0) or 0))),
                "upl": float(trade.get("unrealizedPL", 0) or 0.0),
            }
        )

    if not siblings:
        return None
    return {
        "thesis_market": thesis_market,
        "count": len(siblings) + 1,
        "siblings": siblings,
        "pairs": sorted({str(item["pair"]) for item in siblings}),
    }


def decide_close_discipline(
    trade: dict[str, Any],
    *,
    reason: str,
    requested_units: int | None = None,
    open_trades: list[dict[str, Any]] | None = None,
    current_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trade_id = str(trade.get("id") or "")
    pair = str(trade.get("instrument") or "")
    direction = "LONG" if float(trade.get("currentUnits", 0) or 0) > 0 else "SHORT"
    current_units = abs(int(float(trade.get("currentUnits", 0) or 0)))
    requested = current_units if requested_units is None else abs(int(requested_units))
    full_close = requested >= current_units
    upl = float(trade.get("unrealizedPL", 0) or 0.0)

    meta = current_meta or load_pretrade_meta(trade_id)
    shape = parse_meta_shape(meta)
    dead_layer = classify_dead_layer(reason)
    execution_style = str((meta or {}).get("execution_style") or "").upper()
    allocation_band = str((meta or {}).get("allocation_band") or "").upper()
    archetype = shape["archetype"]
    regime = shape["regime"]
    entry_spread = entry_spread_pips_from_log(trade_id)
    inventory = inventory_family_summary(trade_id, direction, meta, open_trades)

    bad_fill = False
    if entry_spread is not None:
        hard_abs = 6.0 if pair.endswith("JPY") else 3.0
        bad_fill = entry_spread >= hard_abs

    dirty_counter = archetype == "COUNTER_REVERSAL" or "COUNTER" in archetype or allocation_band == "B-"
    soft_limit_stop = execution_style in {"LIMIT", "STOP-ENTRY", "STOP"} and regime in SOFT_REGIMES

    allow_full_close = True
    recommended_action = "FULL_CLOSE"
    problems: list[str] = []
    notes: list[str] = []

    if not reason.strip():
        allow_full_close = False
        recommended_action = "WRITE_REASON_FIRST"
        problems.append("full close without --reason is not allowed for trader inventory")

    if not full_close or upl >= 0:
        return {
            "allow_full_close": allow_full_close,
            "recommended_action": recommended_action if full_close else "PARTIAL_CLOSE_OK",
            "dead_layer": dead_layer,
            "execution_style": execution_style or "UNKNOWN",
            "archetype": archetype or "UNKNOWN",
            "regime": regime or "UNKNOWN",
            "inventory_group": inventory,
            "suggested_units": suggested_half_units(current_units),
            "problems": problems,
            "notes": notes,
            "bad_fill": bad_fill,
        }

    if bad_fill:
        notes.append(f"entry spread was pathological at {entry_spread:.1f}pip; kill the seat, not the thesis")
    if dirty_counter:
        notes.append("dirty counter / reversal seat should die faster than a core trend inventory leg")
    if inventory:
        sibling_pairs = ", ".join(inventory["pairs"])
        notes.append(
            f"same market thesis still has {inventory['count']} live expressions ({sibling_pairs or pair}); manage as inventory, not isolated micro-close"
        )

    if dead_layer in {"aging", "market", "structure"}:
        recommended_action = "FULL_CLOSE"
    elif bad_fill or dirty_counter:
        recommended_action = "FULL_CLOSE"
    elif dead_layer in {"trigger", "vehicle", "unknown"} and soft_limit_stop:
        recommended_action = "HALF_OR_HOLD"
        allow_full_close = False
        problems.append(
            f"{execution_style or 'LIMIT/STOP'} {regime or 'soft-regime'} seat is trigger/vehicle damage only; do not flatten on first wobble"
        )
    elif dead_layer in {"trigger", "vehicle", "unknown"}:
        recommended_action = "HALF_OR_HOLD"
        allow_full_close = False
        problems.append("trigger/vehicle damage alone is not enough for reflex full close")

    if inventory and dead_layer in {"trigger", "vehicle", "unknown"}:
        recommended_action = "MANAGE_AS_INVENTORY"
        allow_full_close = False
        problems.append("same market thesis still has sibling inventory; do not micro-close one expression")

    if allow_full_close and not problems:
        notes.append("full close is consistent with the dead-layer branch")

    return {
        "allow_full_close": allow_full_close,
        "recommended_action": recommended_action,
        "dead_layer": dead_layer,
        "execution_style": execution_style or "UNKNOWN",
        "archetype": archetype or "UNKNOWN",
        "regime": regime or "UNKNOWN",
        "inventory_group": inventory,
        "suggested_units": suggested_half_units(current_units),
        "problems": problems,
        "notes": notes,
        "bad_fill": bad_fill,
    }
