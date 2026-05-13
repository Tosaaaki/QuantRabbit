"""Recent-trade P&L bias modifiers for trader_brain scoring.

Reads `data/execution_ledger.db` (already populated by `execution-ledger-sync`
each cycle) and computes a per-(pair, direction) bias modifier from the last
N closed trades. The modifier is added to lane scores so:

- Pair × direction with recent losses → negative modifier → score downweight
- Pair × direction with recent wins → positive modifier → score upweight
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
    """Read execution_ledger.db, aggregate the last N closed trades per
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
        # TRADE_CLOSED events are the authoritative record of realized
        # P&L per closed trade. Their `side` field records the CLOSING
        # transaction direction, which is the OPPOSITE of the original
        # position direction (closing a LONG position issues a SELL).
        # We invert below so that the (pair, direction) key reflects the
        # position's original side, not the close fill side.
        cur = conn.execute(
            """
            SELECT pair, side, realized_pl_jpy, ts_utc
            FROM execution_events
            WHERE event_type = 'TRADE_CLOSED'
              AND realized_pl_jpy IS NOT NULL
              AND pair IS NOT NULL
              AND side IS NOT NULL
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
    for pair, close_side, pl, ts in rows:
        if pl is None or pair is None or close_side is None:
            continue
        # Invert: CLOSE direction is opposite of original position direction.
        original_side = "SHORT" if str(close_side).upper() == "LONG" else "LONG"
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
