"""Multi-pair correlation matrix prediction.

User directive 2026-05-14:「Multi-pair correlation matrix prediction
(DXY/yields/SPX → individual pairs)」.

This module computes rolling correlations between pairs from
`pair_charts.json` close-series (the recent_candles field shipped
2026-05-14). When a leading pair has moved meaningfully but a
correlated lagging pair hasn't yet, the lagging pair has a directional
bias for catch-up.

Example:
- EUR_USD and GBP_USD typically correlate ~0.85 (both USD-quote)
- If GBP_USD made a +25pip move while EUR_USD is flat, EUR_USD is
  likely to follow UP (catch-up trade)
- Generate `correlation_lag` signal: direction=UP, lead_time ≈ 30 min

The DXY case is special — already covered by `_detect_cross_asset_lag`
in forward_projection.py via cross_asset_snapshot.json. This module
covers PAIR-TO-PAIR correlations (intra-FX), which the cross_asset
file doesn't.

Algorithm:
1. For each pair in pair_charts, extract recent_candles closes (last N
   values, typically 30).
2. Compute pairwise rolling Pearson correlation across all pairs.
3. For each (pair_A, pair_B) with |correlation| ≥ 0.7:
   - Compute recent move (last K bars %change) for both
   - If pair_A moved ≥ MOVE_THRESHOLD% but pair_B moved <
     MOVE_THRESHOLD% / 3, project pair_B catch-up
4. Direction:
   - Positive correlation: same direction (pair_A up → pair_B up)
   - Negative correlation: opposite direction
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


CORRELATION_MIN_ABS = float(os.environ.get("QR_CORRELATION_MIN_ABS", "0.7"))
CORRELATION_LOOKBACK_BARS = int(os.environ.get("QR_CORRELATION_LOOKBACK_BARS", "30"))
LEADING_MOVE_THRESHOLD_PCT = float(os.environ.get("QR_LEADING_MOVE_THRESHOLD_PCT", "0.15"))
LAG_RATIO_MAX = float(os.environ.get("QR_LAG_RATIO_MAX", "0.33"))
CORRELATION_LAG_BONUS = float(os.environ.get("QR_CORRELATION_LAG_BONUS", "13.0"))


@dataclass(frozen=True)
class CorrelationLagSignal:
    pair: str
    leader_pair: str
    correlation: float
    leader_move_pct: float
    lagger_move_pct: float
    direction: str  # "UP" | "DOWN"
    confidence: float
    bonus_magnitude: float
    rationale: str


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_CORRELATION_PREDICTOR", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 5:
        return None
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return num / (sx * sy)


def _extract_closes(view: Dict[str, Any]) -> List[float]:
    """Pull close prices from recent_candles."""
    candles = view.get("recent_candles") or []
    out: List[float] = []
    for c in candles:
        if not isinstance(c, dict):
            continue
        try:
            out.append(float(c.get("c", c.get("close"))))
        except (TypeError, ValueError):
            continue
    return out


def _find_m15_view(pair_chart: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for v in pair_chart.get("views") or []:
        if not isinstance(v, dict):
            continue
        if str(v.get("granularity") or "").upper() == "M15":
            return v
    return None


def build_correlation_map(
    pair_charts: Dict[str, Dict[str, Any]],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise correlations across all pairs in pair_charts.

    Returns sparse dict {(pair_A, pair_B): correlation} where
    |correlation| ≥ CORRELATION_MIN_ABS. A→B and B→A both stored.
    """
    closes_by_pair: Dict[str, List[float]] = {}
    for pair, chart in pair_charts.items():
        view = _find_m15_view(chart)
        if view is None:
            continue
        cl = _extract_closes(view)
        if len(cl) >= 10:
            closes_by_pair[pair] = cl[-CORRELATION_LOOKBACK_BARS:]
    pairs = sorted(closes_by_pair.keys())
    out: Dict[Tuple[str, str], float] = {}
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            xs = closes_by_pair[a]; ys = closes_by_pair[b]
            n = min(len(xs), len(ys))
            corr = _pearson(xs[-n:], ys[-n:])
            if corr is None:
                continue
            if abs(corr) >= CORRELATION_MIN_ABS:
                out[(a, b)] = corr
                out[(b, a)] = corr
    return out


def detect_correlation_lag(
    target_pair: str,
    pair_charts: Dict[str, Dict[str, Any]],
    correlation_map: Optional[Dict[Tuple[str, str], float]] = None,
) -> List[CorrelationLagSignal]:
    """For target_pair, find leader pairs with strong correlation that
    have made a meaningful move ahead of target. Project target catch-up."""
    if _is_disabled():
        return []
    if correlation_map is None:
        correlation_map = build_correlation_map(pair_charts)

    # Get target pair closes
    target_chart = pair_charts.get(target_pair)
    if target_chart is None:
        return []
    target_view = _find_m15_view(target_chart)
    if target_view is None:
        return []
    target_closes = _extract_closes(target_view)
    if len(target_closes) < 10:
        return []
    # Recent move %: compare last close vs N bars ago
    recent_window = min(5, len(target_closes) - 1)
    target_move_pct = (target_closes[-1] - target_closes[-1 - recent_window]) / target_closes[-1 - recent_window] * 100.0

    out: List[CorrelationLagSignal] = []
    for (a, b), corr in correlation_map.items():
        if a != target_pair:
            continue
        leader = b
        leader_chart = pair_charts.get(leader)
        if leader_chart is None:
            continue
        leader_view = _find_m15_view(leader_chart)
        if leader_view is None:
            continue
        leader_closes = _extract_closes(leader_view)
        if len(leader_closes) < 10:
            continue
        leader_move_pct = (leader_closes[-1] - leader_closes[-1 - recent_window]) / leader_closes[-1 - recent_window] * 100.0

        # Need leader to have moved meaningfully
        if abs(leader_move_pct) < LEADING_MOVE_THRESHOLD_PCT:
            continue
        # And target to be lagging
        if abs(target_move_pct) >= abs(leader_move_pct) * LAG_RATIO_MAX:
            continue
        # Project direction
        if corr > 0:
            # Positive correlation: same direction
            direction = "UP" if leader_move_pct > 0 else "DOWN"
        else:
            direction = "DOWN" if leader_move_pct > 0 else "UP"

        out.append(CorrelationLagSignal(
            pair=target_pair,
            leader_pair=leader,
            correlation=round(corr, 3),
            leader_move_pct=round(leader_move_pct, 3),
            lagger_move_pct=round(target_move_pct, 3),
            direction=direction,
            confidence=min(1.0, abs(corr) * (abs(leader_move_pct) / LEADING_MOVE_THRESHOLD_PCT) * 0.5),
            bonus_magnitude=CORRELATION_LAG_BONUS,
            rationale=(
                f"corr({leader},{target_pair})={corr:+.2f}; "
                f"{leader} moved {leader_move_pct:+.2f}%, "
                f"{target_pair} {target_move_pct:+.2f}% — catch-up {direction}"
            ),
        ))
    return out


def aggregate_correlation_lag_score(
    signals: List[CorrelationLagSignal],
    intent_direction: str,
) -> Tuple[float, List[str]]:
    """Sum aligned correlation lag signals. Multiple leaders pointing
    the same way reinforce the score (no special multiplier — confluence
    is reflected through having multiple signals)."""
    intent_up = intent_direction.upper() == "LONG"
    total = 0.0
    rationales: List[str] = []
    for s in signals:
        contrib = s.bonus_magnitude * s.confidence
        signal_up = s.direction.upper() == "UP"
        if signal_up == intent_up:
            total += contrib
            rationales.append(f"+{contrib:.1f} {s.rationale}")
        else:
            total -= contrib * 0.5
            rationales.append(f"-{contrib * 0.5:.1f} AGAINST {s.rationale}")
    return round(total, 2), rationales
