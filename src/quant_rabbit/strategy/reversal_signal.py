"""Reversal-from-extreme detector — override history penalties at turning points.

The 2026-05-12/13 P&L review surfaced an asymmetry: the trader system
correctly DOWNWEIGHTS losing directions via lane_history (Module B),
trader_overrides (Module C), and news_themes. But that creates a new
failure mode: when price hits an extreme low and reverses, the system
CANNOT take the bottom-buy LONG because all three modules have
accumulated penalties against that direction (from yesterday's losses
on the SAME direction).

User feedback 2026-05-13:
  「今、ポンドドル、下げ否定して、上げ始めてるよね？こういうときに
   追撃するんじゃないの？安く買って、高く売る、高く売って、安く買う。
   こういう基本的なことができてない」

This module detects:
1. **Bottom-buy LONG**: price_percentile_24h ≤ LOW_THRESHOLD AND
   structure prints BOS_UP / CHOCH_UP on M5 or M15 (confirming the
   downside is being rejected).
2. **Top-fade SHORT**: price_percentile_24h ≥ HIGH_THRESHOLD AND
   structure prints BOS_DOWN / CHOCH_DOWN on M5 or M15.

When the signal fires, `_score_lane` adds REVERSAL_BONUS (default +40)
which is large enough to offset the typical accumulated history-based
penalty (-25 lane_history + -20 trader_overrides + -10 news ≈ -55).
The bonus's MAGNITUDE is intentionally large so that at confirmed
reversal points, the system PREFERS the contrarian direction over the
trend-aligned one — that's the "buy low, sell high" primitive.

Safety:
- Only fires when both conditions are met (extreme percentile AND
  structural confirmation). Pure-percentile bottoms without structure
  confirmation are NOT enough — could be a continuing trend leg.
- Requires `close_confirmed` structural break (not wick-only, per
  feedback_structure_close_vs_wick.md).
- Magnitude is env-tunable; kill switch `QR_DISABLE_REVERSAL_SIGNAL=1`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


REVERSAL_LOW_PERCENTILE_THRESHOLD = float(
    os.environ.get("QR_REVERSAL_LOW_PCTILE", "0.15")
)
REVERSAL_HIGH_PERCENTILE_THRESHOLD = float(
    os.environ.get("QR_REVERSAL_HIGH_PCTILE", "0.85")
)
# 7-day percentile is checked in parallel with 24h; either qualifying
# extreme suffices. 7d catches "we're at a weekly low even though
# today's range hasn't been deep yet" — that's the swing-low bottom
# the user wants to buy.
REVERSAL_LOW_PCTILE_7D_THRESHOLD = float(
    os.environ.get("QR_REVERSAL_LOW_PCTILE_7D", "0.15")
)
REVERSAL_HIGH_PCTILE_7D_THRESHOLD = float(
    os.environ.get("QR_REVERSAL_HIGH_PCTILE_7D", "0.85")
)
REVERSAL_BONUS = float(os.environ.get("QR_REVERSAL_BONUS", "40.0"))


# Match M1/M5/M15 BOS_UP/DOWN or CHOCH_UP/DOWN. Optionally with the
# `:wick` suffix; we EXCLUDE wick-confirmed breaks per
# feedback_structure_close_vs_wick.md.
# M1 is included because at fresh reversal points the M1 BOS often
# prints BEFORE M5 confirms — and the user's directive 2026-05-13 is
# explicitly to "追撃 (chase) the rejection as it forms", not to wait
# for slow-TF confirmation.
_STRUCT_RE = re.compile(
    r"\b(M1|M5|M15)\([^)]*?struct=(BOS|CHOCH)_(UP|DOWN)@[0-9.]+(:wick)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReversalSignal:
    pair: str
    side: str  # "LONG" | "SHORT"
    percentile_24h: float
    struct_tf: str  # "M5" | "M15"
    struct_kind: str  # "BOS" | "CHOCH"
    bonus: float
    rationale: str


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_REVERSAL_SIGNAL", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def detect_reversal(
    pair_chart: Optional[Dict[str, Any]],
    intent_direction: str,
) -> Optional[ReversalSignal]:
    """Detect a reversal-from-extreme setup for the given intent direction.

    Returns None when:
    - module is kill-switched
    - chart data unavailable
    - percentile not at extreme
    - no close-confirmed structural reversal in M5/M15 against the
      previous direction
    """
    if _is_disabled():
        return None
    if not pair_chart:
        return None

    confluence = pair_chart.get("confluence") or {}

    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    pctile_24h = _to_float(confluence.get("price_percentile_24h"))
    pctile_7d = _to_float(confluence.get("price_percentile_7d"))

    side = intent_direction.upper()
    pair = str(pair_chart.get("pair") or "")

    # Percentile + direction compatibility check. Either 24h OR 7d extreme qualifies.
    pctile_chosen: float
    pctile_kind: str
    if side == "LONG":
        candidates: list[tuple[float, str]] = []
        if pctile_24h is not None and pctile_24h <= REVERSAL_LOW_PERCENTILE_THRESHOLD:
            candidates.append((pctile_24h, "24h"))
        if pctile_7d is not None and pctile_7d <= REVERSAL_LOW_PCTILE_7D_THRESHOLD:
            candidates.append((pctile_7d, "7d"))
        if not candidates:
            return None
        pctile_chosen, pctile_kind = min(candidates, key=lambda x: x[0])
        expected_struct_dir = "UP"
    elif side == "SHORT":
        candidates = []
        if pctile_24h is not None and pctile_24h >= REVERSAL_HIGH_PERCENTILE_THRESHOLD:
            candidates.append((pctile_24h, "24h"))
        if pctile_7d is not None and pctile_7d >= REVERSAL_HIGH_PCTILE_7D_THRESHOLD:
            candidates.append((pctile_7d, "7d"))
        if not candidates:
            return None
        pctile_chosen, pctile_kind = max(candidates, key=lambda x: x[0])
        expected_struct_dir = "DOWN"
    else:
        return None

    # Scan chart_story for M5/M15 BOS_UP/DOWN close-confirmed (not wick).
    chart_story = str(pair_chart.get("chart_story") or "")
    matched: Optional[tuple[str, str]] = None  # (tf, kind)
    for m in _STRUCT_RE.finditer(chart_story):
        tf = m.group(1).upper()
        kind = m.group(2).upper()
        struct_dir = m.group(3).upper()
        is_wick = bool(m.group(4))
        if is_wick:
            continue  # wick-confirmed sweeps don't count
        if struct_dir == expected_struct_dir:
            matched = (tf, kind)
            break

    if matched is None:
        return None

    tf, kind = matched
    rationale = (
        f"reversal-from-extreme: {pair} pctile_{pctile_kind}={pctile_chosen:.2f} "
        f"(extreme {'low' if side == 'LONG' else 'high'}) + {tf} "
        f"{kind}_{expected_struct_dir} close-confirmed → +{REVERSAL_BONUS:.0f} for {side}"
    )
    return ReversalSignal(
        pair=pair,
        side=side,
        percentile_24h=pctile_chosen,
        struct_tf=tf,
        struct_kind=kind,
        bonus=REVERSAL_BONUS,
        rationale=rationale,
    )
