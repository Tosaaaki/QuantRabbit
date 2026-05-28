"""Entry timing gate — confirm M5 momentum before SEND_ENTRY.

The 2026-05-13 incident showed the trader entering at the WRONG moment
even when the H4/D thesis was correct: the position bled for 30+ min
before the trend resumed, hitting SL on noise. A discretionary trader
waits for M5/M15 momentum to align with the thesis before pulling the
trigger. This module gates SEND_ENTRY behind that confirmation.

Read M5 chart from `pair_charts.json` and check:
1. Last 3 M5 candles close direction vs intent direction
2. M5 momentum class (HIGH/MODERATE/LOW from existing
   `_short_term_momentum_class`)

Gate result is one of:
- "ALIGNED" — last 3 M5 candles agree with entry direction → no penalty
- "MIXED" — at least 1 candle disagrees → small penalty
- "AGAINST" — all 3 candles disagree → big penalty (likely top/bottom)

The module returns an additive score signal and state. `trader_brain`
uses the state to hard-block live `MARKET` entries only when the last
three M5 candles are fully against the lane; pending rail geometry can
still wait for its trigger. This keeps "fine timing" separate from
"blatant top-buying / bottom-selling".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


ENTRY_TIMING_AGAINST_PENALTY = float(os.environ.get("QR_ENTRY_TIMING_AGAINST_PENALTY", "20.0"))
ENTRY_TIMING_MIXED_PENALTY = float(os.environ.get("QR_ENTRY_TIMING_MIXED_PENALTY", "8.0"))
ENTRY_TIMING_ALIGNED_BONUS = float(os.environ.get("QR_ENTRY_TIMING_ALIGNED_BONUS", "5.0"))


@dataclass(frozen=True)
class EntryTimingResult:
    state: str  # "ALIGNED" | "MIXED" | "AGAINST" | "UNKNOWN"
    score_delta: float
    rationale: str | None


def _m5_recent_closes(pair_chart: Dict[str, Any], count: int = 3) -> list[tuple[float, float]]:
    """Extract last N (open, close) pairs from M5 candle view."""
    views = pair_chart.get("views") or []
    for v in views:
        if not isinstance(v, dict):
            continue
        if str(v.get("timeframe") or v.get("tf") or v.get("granularity") or "").upper() != "M5":
            continue
        candles = v.get("candles") or v.get("bars") or v.get("recent_candles") or []
        # Take the last `count` candles, preserving chronological order.
        recent = candles[-count:] if isinstance(candles, list) else []
        out: list[tuple[float, float]] = []
        for c in recent:
            if not isinstance(c, dict):
                continue
            o = c.get("open") or c.get("o")
            cl = c.get("close") or c.get("c")
            try:
                out.append((float(o), float(cl)))
            except (TypeError, ValueError):
                continue
        return out
    return []


def check_entry_timing(
    pair_chart: Optional[Dict[str, Any]],
    intent_direction: str,
) -> EntryTimingResult:
    """Check whether the last 3 M5 candles favor the entry direction.

    Returns UNKNOWN (zero delta) when the chart payload doesn't include
    M5 candles — degrades to the existing scoring without blocking.
    """
    if not pair_chart:
        return EntryTimingResult(state="UNKNOWN", score_delta=0.0, rationale=None)

    closes = _m5_recent_closes(pair_chart, count=3)
    if len(closes) < 3:
        return EntryTimingResult(state="UNKNOWN", score_delta=0.0, rationale=None)

    direction_up = intent_direction.upper() == "LONG"
    # Per-candle direction: up if close > open.
    candle_dirs = [c > o for o, c in closes]
    aligned_count = sum(1 for d in candle_dirs if d == direction_up)

    if aligned_count == 3:
        return EntryTimingResult(
            state="ALIGNED",
            score_delta=ENTRY_TIMING_ALIGNED_BONUS,
            rationale=f"entry timing ALIGNED: last 3 M5 closes all favor {intent_direction} (+{ENTRY_TIMING_ALIGNED_BONUS:.1f})",
        )
    if aligned_count == 0:
        return EntryTimingResult(
            state="AGAINST",
            score_delta=-ENTRY_TIMING_AGAINST_PENALTY,
            rationale=(
                f"entry timing AGAINST: last 3 M5 closes all opposite to {intent_direction} "
                f"(-{ENTRY_TIMING_AGAINST_PENALTY:.1f}) — likely buying-top / selling-bottom"
            ),
        )
    # Mixed (1 or 2 aligned).
    return EntryTimingResult(
        state="MIXED",
        score_delta=-ENTRY_TIMING_MIXED_PENALTY,
        rationale=(
            f"entry timing MIXED: {aligned_count}/3 M5 closes favor {intent_direction} "
            f"(-{ENTRY_TIMING_MIXED_PENALTY:.1f})"
        ),
    )
