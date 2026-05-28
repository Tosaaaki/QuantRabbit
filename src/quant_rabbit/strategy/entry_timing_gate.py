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

The module returns additive score signals and states. `trader_brain`
uses `check_entry_timing` to hard-block live `MARKET` entries only when
the last three M5 candles are fully against the lane; pending rail
geometry can still wait through that micro noise. `check_operating_tf_momentum`
is stricter: if M5/M15/M30 are already walking the opposite way, even a
resting rail LIMIT must wait for a later close-confirmed rejection instead
of being pre-armed into the impulse. This keeps "fine timing" separate
from "blatant top-buying / bottom-selling".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


ENTRY_TIMING_AGAINST_PENALTY = float(os.environ.get("QR_ENTRY_TIMING_AGAINST_PENALTY", "20.0"))
ENTRY_TIMING_MIXED_PENALTY = float(os.environ.get("QR_ENTRY_TIMING_MIXED_PENALTY", "8.0"))
ENTRY_TIMING_ALIGNED_BONUS = float(os.environ.get("QR_ENTRY_TIMING_ALIGNED_BONUS", "5.0"))
OPERATING_TF_MOMENTUM_AGAINST_PENALTY = float(
    os.environ.get("QR_OPERATING_TF_MOMENTUM_AGAINST_PENALTY", "35.0")
)
OPERATING_TF_MOMENTUM_MIN_ADX = float(os.environ.get("QR_OPERATING_TF_MOMENTUM_MIN_ADX", "20.0"))
OPERATING_TF_MOMENTUM_STRONG_ADX = float(os.environ.get("QR_OPERATING_TF_MOMENTUM_STRONG_ADX", "25.0"))
OPERATING_TF_MOMENTUM_TFS = ("M5", "M15", "M30")


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


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _view_timeframe(view: Dict[str, Any]) -> str:
    return str(view.get("timeframe") or view.get("tf") or view.get("granularity") or "").upper()


def _regime_direction(regime: Any) -> str | None:
    text = str(regime or "").upper()
    if "TREND_UP" in text or "IMPULSE_UP" in text or text in {"BULL", "BULLISH"}:
        return "UP"
    if "TREND_DOWN" in text or "IMPULSE_DOWN" in text or text in {"BEAR", "BEARISH"}:
        return "DOWN"
    return None


def check_operating_tf_momentum(
    pair_chart: Optional[Dict[str, Any]],
    intent_direction: str,
) -> EntryTimingResult:
    """Check whether the M5/M15/M30 operating stack is strongly against entry.

    Unlike `check_entry_timing`, this is not a three-candle fine-timing
    read. It asks whether the actual setup stack is already walking the
    opposite way. A resting range LIMIT can wait through noisy M5 candles,
    but it should not be armed into a multi-timeframe impulse without a
    later rejection confirmation.
    """
    if not pair_chart:
        return EntryTimingResult(state="UNKNOWN", score_delta=0.0, rationale=None)
    intent_side = intent_direction.upper()
    if intent_side not in {"LONG", "SHORT"}:
        return EntryTimingResult(state="UNKNOWN", score_delta=0.0, rationale=None)
    intent_move = "UP" if intent_side == "LONG" else "DOWN"

    views = pair_chart.get("views") or []
    if not isinstance(views, list):
        return EntryTimingResult(state="UNKNOWN", score_delta=0.0, rationale=None)

    observed = 0
    opposed: list[tuple[str, str, str, float | None, bool]] = []
    aligned: list[tuple[str, str, str, float | None, bool]] = []
    for view in views:
        if not isinstance(view, dict):
            continue
        tf = _view_timeframe(view)
        if tf not in OPERATING_TF_MOMENTUM_TFS:
            continue
        observed += 1
        regime = str(view.get("regime") or "")
        move = _regime_direction(regime)
        if move is None:
            continue
        indicators = view.get("indicators") if isinstance(view.get("indicators"), dict) else {}
        adx = _float_or_none((indicators or {}).get("adx_14") or (indicators or {}).get("adx"))
        if adx is None or adx < OPERATING_TF_MOMENTUM_MIN_ADX:
            continue
        strong = adx >= OPERATING_TF_MOMENTUM_STRONG_ADX or (
            "IMPULSE" in regime.upper() and adx >= OPERATING_TF_MOMENTUM_MIN_ADX
        )
        item = (tf, regime, move, adx, strong)
        if move == intent_move:
            aligned.append(item)
        else:
            opposed.append(item)

    if observed == 0:
        return EntryTimingResult(state="UNKNOWN", score_delta=0.0, rationale=None)

    strong_opposed = sum(1 for *_unused, strong in opposed if strong)
    any_strong_opposed = strong_opposed > 0
    if len(opposed) >= 2 and (strong_opposed >= 2 or any_strong_opposed):
        details = ", ".join(f"{tf} {regime} ADX={adx:.1f}" for tf, regime, _move, adx, _strong in opposed)
        return EntryTimingResult(
            state="AGAINST",
            score_delta=-OPERATING_TF_MOMENTUM_AGAINST_PENALTY,
            rationale=(
                f"operating TF momentum AGAINST: {details} oppose {intent_side} "
                f"(-{OPERATING_TF_MOMENTUM_AGAINST_PENALTY:.1f})"
            ),
        )

    strong_aligned = sum(1 for *_unused, strong in aligned if strong)
    if len(aligned) >= 2 and strong_aligned >= 1:
        return EntryTimingResult(
            state="ALIGNED",
            score_delta=0.0,
            rationale=None,
        )
    return EntryTimingResult(state="MIXED", score_delta=0.0, rationale=None)
