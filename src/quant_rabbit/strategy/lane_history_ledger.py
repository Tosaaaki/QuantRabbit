"""Recent-trade P&L bias modifiers for trader_brain scoring.

Reads `data/execution_ledger.db` (already populated by `execution-ledger-sync`
each cycle) and computes a per-(pair, direction) bias modifier from the last
N realized trade outcomes. The modifier is added to lane scores so:

- Pair × direction with recent realized losses → negative modifier → score downweight
- Pair × direction with recent realized wins → positive modifier → score upweight
- Pair × direction with no history → 0 modifier → no change

This breaks the 2026-05-12/13 pattern where the trader kept entering
GBP_USD LONG and AUD_JPY SHORT despite consecutive losses on those
exact lane shapes (post-mortem in feedback_broker_sl_noise_hunt.md +
project_micro_adverse_within_macro_correct.md).

The modifier is BOUNDED so a single bad streak can't blackhole a lane:
- Saturates at ±LANE_HISTORY_MAX_MODIFIER (default ±25.0, same magnitude
  as the existing direction-conflict penalty and attack-veto penalty).
- Saturation P&L is `LANE_HISTORY_SATURATION_PL_JPY` (default 3000 JPY)
  — small enough to detect a 2-3 trade losing streak, large enough that
  a single -500 JPY trade is barely felt.
- Modifier function is `tanh(net_pl / saturation)` so the curve is smooth
  near zero and saturates gracefully.
"""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


# Tunable knobs. Env-overridable so a sandbox/test can override without
# code edits, and so a future regime-switch can broaden or narrow the
# response curve without re-deploying.
LANE_HISTORY_LOOKBACK_TRADES = int(os.environ.get("QR_LANE_HISTORY_LOOKBACK", "5"))
LANE_HISTORY_MAX_MODIFIER = float(os.environ.get("QR_LANE_HISTORY_MAX_MODIFIER", "25.0"))
LANE_HISTORY_SATURATION_PL_JPY = float(os.environ.get("QR_LANE_HISTORY_SATURATION_PL", "3000"))


@dataclass(frozen=True)
class LaneHistorySnapshot:
    """Per-(pair, direction) snapshot of recent realized P&L."""

    pair: str
    direction: str  # "LONG" | "SHORT"
    sample_size: int
    net_pl_jpy: float
    modifier: float  # bounded score delta to add in _score_lane


def compute_lane_history(db_path: Path) -> Dict[Tuple[str, str], LaneHistorySnapshot]:
    """Read execution_ledger.db, aggregate the last N realized outcomes per
    (pair, direction), and emit bias modifiers.

    Returns an empty dict on any database error so the trader degrades
    gracefully — never blocks scoring on a missing/locked DB.
    """
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return {}
    try:
        # TRADE_CLOSED and TRADE_REDUCED events are authoritative realized
        # P&L, but OANDA close transactions do not reliably carry the trader
        # lane id. Attribute outcomes through the original gateway-sent entry
        # (or an entry fill that already carries lane_id) before using them for
        # live scoring; manual/tagless outcomes must not bias the bot.
        cur = conn.execute(
            """
            WITH gateway_entries AS (
                SELECT
                    trade_id,
                    order_id,
                    lane_id
                FROM execution_events
                WHERE event_type = 'GATEWAY_ORDER_SENT'
                  AND lane_id IS NOT NULL
                  AND lane_id != ''
            ),
            entries AS (
                SELECT
                    e.trade_id,
                    COALESCE(NULLIF(MAX(e.lane_id), ''), MAX(g.lane_id)) AS gateway_lane_id,
                    CASE
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
                    entries.position_side AS original_side,
                    e.realized_pl_jpy
                FROM execution_events e
                INNER JOIN entries ON entries.trade_id = e.trade_id
                WHERE e.event_type IN ('TRADE_CLOSED', 'TRADE_REDUCED')
                  AND e.realized_pl_jpy IS NOT NULL
                  AND e.pair IS NOT NULL
            )
            SELECT
                pair,
                original_side,
                SUM(realized_pl_jpy) AS realized_pl_jpy,
                MAX(ts_utc) AS ts_utc
            FROM realized
            WHERE original_side IS NOT NULL
            GROUP BY outcome_id, pair, original_side
            ORDER BY ts_utc DESC
            LIMIT 500
            """
        )
        rows = cur.fetchall()
    except sqlite3.Error:
        conn.close()
        return {}
    conn.close()

    grouped: Dict[Tuple[str, str], list] = {}
    for pair, original_side, pl, ts in rows:
        if pl is None or pair is None or original_side is None:
            continue
        key = (str(pair), original_side)
        bucket = grouped.setdefault(key, [])
        if len(bucket) < LANE_HISTORY_LOOKBACK_TRADES:
            bucket.append(float(pl))

    snapshots: Dict[Tuple[str, str], LaneHistorySnapshot] = {}
    for key, pls in grouped.items():
        if not pls:
            continue
        net = sum(pls)
        modifier = LANE_HISTORY_MAX_MODIFIER * math.tanh(net / LANE_HISTORY_SATURATION_PL_JPY)
        snapshots[key] = LaneHistorySnapshot(
            pair=key[0],
            direction=key[1],
            sample_size=len(pls),
            net_pl_jpy=net,
            modifier=modifier,
        )
    return snapshots


def lane_history_modifier(
    snapshots: Dict[Tuple[str, str], LaneHistorySnapshot],
    pair: str,
    direction: str,
) -> tuple[float, str | None]:
    """Look up a single (pair, direction) modifier and a one-line rationale.

    Returns (modifier, rationale) where rationale is None when the lane
    has no history yet (so caller can skip appending a noise line).
    """
    key = (pair, "LONG" if direction.upper() == "LONG" else "SHORT")
    snap = snapshots.get(key)
    if snap is None or snap.sample_size == 0:
        return 0.0, None
    sign = "+" if snap.modifier >= 0 else ""
    rationale = (
        f"lane history {pair}:{direction} last {snap.sample_size}t "
        f"net={snap.net_pl_jpy:+.0f}JPY → score {sign}{snap.modifier:.1f}"
    )
    return snap.modifier, rationale
