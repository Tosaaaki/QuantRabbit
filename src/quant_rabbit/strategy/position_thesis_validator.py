"""Position thesis validator — apply prediction stack to OPEN positions.

User insight 2026-05-14:「その予測はエントリー時のみ？ポジについても
精査してくれるの？」.

The 17-layer prediction stack (pattern_signals, forward_projection,
correlation_predictor, path_projection, reversal_signal, ...) was
originally wired only to `trader_brain._score_lane`, which evaluates
NEW entry candidates. Existing open positions were managed by
`PositionManager._adaptive_tp_action` using a much smaller signal
set (micro/macro classification from chart_story). This created a
blind spot: a position with a thesis that the prediction layer has
since invalidated would continue HOLD because the position-side
logic didn't consult predictions.

This module runs the same prediction stack against each open
trader-owned position in the position's direction. The aggregate
score tells the trader:

- **score ≥ THESIS_EXTEND_THRESHOLD** (default +20):
  predictions reinforce the position direction → safe to extend TP /
  hold longer / consider adding to position
- **score ≤ THESIS_EXIT_THRESHOLD** (default -25):
  predictions are now AGAINST the position → flag for thesis review,
  candidate for CLOSE (via Gate A/B in gpt_trader path)
- **between thresholds**:
  HOLD — predictions don't strongly support or contradict

Importantly this module does NOT auto-close. It produces a SCORE +
RATIONALE that gets written to `data/position_thesis_report.json` for
the trader to consume. CLOSE decisions still require Gate A/B (J
hardening) — gpt_trader can use this score as evidence but the gates
must still pass.

The module respects the SL-free invariant by never modifying SL,
never modifying TP directly (tp_rebalancer handles that), and never
calling broker write APIs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


THESIS_EXIT_THRESHOLD = float(os.environ.get("QR_THESIS_EXIT_THRESHOLD", "-25.0"))
THESIS_EXTEND_THRESHOLD = float(os.environ.get("QR_THESIS_EXTEND_THRESHOLD", "20.0"))


@dataclass(frozen=True)
class PositionThesisAssessment:
    trade_id: str
    pair: str
    side: str
    pattern_score: float
    projection_score: float
    correlation_score: float
    path_score: float
    reversal_against: bool
    aggregate_score: float
    verdict: str  # "EXTEND" | "HOLD" | "REVIEW_CLOSE"
    rationale_lines: tuple[str, ...]
    context_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "side": self.side,
            "pattern_score": self.pattern_score,
            "projection_score": self.projection_score,
            "correlation_score": self.correlation_score,
            "path_score": self.path_score,
            "reversal_against": self.reversal_against,
            "aggregate_score": self.aggregate_score,
            "verdict": self.verdict,
            "rationale_lines": list(self.rationale_lines),
            "context_notes": list(self.context_notes),
        }

    def with_verdict(self, verdict: str, *notes: str) -> "PositionThesisAssessment":
        return PositionThesisAssessment(
            trade_id=self.trade_id,
            pair=self.pair,
            side=self.side,
            pattern_score=self.pattern_score,
            projection_score=self.projection_score,
            correlation_score=self.correlation_score,
            path_score=self.path_score,
            reversal_against=self.reversal_against,
            aggregate_score=self.aggregate_score,
            verdict=verdict,
            rationale_lines=self.rationale_lines,
            context_notes=self.context_notes + tuple(n for n in notes if n),
        )


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_THESIS_VALIDATOR", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def assess_position(
    *,
    trade_id: str,
    pair: str,
    side: str,
    current_price: float,
    pair_chart: Dict[str, Any],
    pair_charts_full: Dict[str, Dict[str, Any]],
    cot_payload: Optional[Dict[str, Any]] = None,
    option_skew_payload: Optional[Dict[str, Any]] = None,
    calendar_path: Optional[Path] = None,
    cross_asset_path: Optional[Path] = None,
    hit_rates: Optional[Dict[str, Dict[str, Any]]] = None,
    regime: Optional[str] = None,
) -> PositionThesisAssessment:
    """Run the full prediction stack against ONE open position.

    Returns a `PositionThesisAssessment` with per-layer scores and
    aggregate verdict. Direction is the position's own direction
    (not flipped) — we ask "do the predictions still support this
    position's side?".
    """
    # Lazy imports to avoid circulars and keep the module standalone.
    from quant_rabbit.strategy.pattern_signals import (
        aggregate_pattern_score,
        detect_pattern_signals,
    )
    from quant_rabbit.strategy.forward_projection import (
        aggregate_projection_score,
        detect_forward_projections,
    )
    from quant_rabbit.strategy.correlation_predictor import (
        aggregate_correlation_lag_score,
        detect_correlation_lag,
    )
    from quant_rabbit.strategy.path_projection import (
        aggregate_path_score,
        detect_paths,
    )
    from quant_rabbit.strategy.reversal_signal import detect_reversal
    from quant_rabbit.strategy.directional_forecaster import synthesize_forecast
    from quant_rabbit.strategy.forecast_persistence_tracker import assess_position as _persistence_assess

    rationale_lines: List[str] = []
    side_up = side.upper()

    if _is_disabled():
        return PositionThesisAssessment(
            trade_id=trade_id, pair=pair, side=side_up,
            pattern_score=0.0, projection_score=0.0,
            correlation_score=0.0, path_score=0.0,
            reversal_against=False, aggregate_score=0.0,
            verdict="HOLD",
            rationale_lines=("validator disabled",),
        )

    # 1. Pattern signals (past confirmation)
    pattern_signals = detect_pattern_signals(
        pair_chart,
        cot_payload=cot_payload,
        option_skew_payload=option_skew_payload,
    )
    pattern_score, pattern_rat = aggregate_pattern_score(pattern_signals, side_up)
    if pattern_score != 0:
        rationale_lines.append(f"patterns {pattern_score:+.1f}: " + "; ".join(pattern_rat[:2]))

    # 2. Forward projection (future)
    projection_signals = detect_forward_projections(
        pair_chart,
        pair=pair,
        current_price=current_price,
        calendar_path=calendar_path,
        cross_asset_path=cross_asset_path,
    )
    projection_score, proj_rat = aggregate_projection_score(
        projection_signals, side_up,
        hit_rates=hit_rates, pair=pair, regime=regime,
    )
    if projection_score != 0:
        rationale_lines.append(f"forward-proj {projection_score:+.1f}: " + "; ".join(proj_rat[:2]))

    # 3. Cross-pair correlation lag
    correlation_score = 0.0
    if pair_charts_full and pair in pair_charts_full:
        corr_signals = detect_correlation_lag(pair, pair_charts_full)
        correlation_score, corr_rat = aggregate_correlation_lag_score(corr_signals, side_up)
        if correlation_score != 0:
            rationale_lines.append(f"corr-lag {correlation_score:+.1f}: " + "; ".join(corr_rat[:1]))

    # 4. Multi-step path projection
    paths = detect_paths(pair_chart, side_up, current_price)
    path_score, path_rat = aggregate_path_score(paths, side_up)
    if path_score != 0:
        rationale_lines.append(f"path-proj {path_score:+.1f}: " + "; ".join(path_rat[:1]))

    # 5. Reversal signal — but inverted: a reversal signal for the
    # OPPOSITE direction means the market is about to turn AGAINST
    # this position. That's the relevant signal for thesis validation.
    opposite_dir = "SHORT" if side_up == "LONG" else "LONG"
    reversal_against_pos = detect_reversal(pair_chart, opposite_dir)
    reversal_against = reversal_against_pos is not None
    reversal_score = 0.0
    if reversal_against and reversal_against_pos:
        reversal_score = -reversal_against_pos.bonus  # negative for position
        rationale_lines.append(
            f"reversal-AGAINST {reversal_score:+.1f}: {reversal_against_pos.rationale[:80]}"
        )

    aggregate = pattern_score + projection_score + correlation_score + path_score + reversal_score

    if aggregate <= THESIS_EXIT_THRESHOLD:
        verdict = "REVIEW_CLOSE"
    elif aggregate >= THESIS_EXTEND_THRESHOLD:
        verdict = "EXTEND"
    else:
        verdict = "HOLD"

    return PositionThesisAssessment(
        trade_id=trade_id, pair=pair, side=side_up,
        pattern_score=round(pattern_score, 2),
        projection_score=round(projection_score, 2),
        correlation_score=round(correlation_score, 2),
        path_score=round(path_score, 2),
        reversal_against=reversal_against,
        aggregate_score=round(aggregate, 2),
        verdict=verdict,
        rationale_lines=tuple(rationale_lines),
    )


def assess_all_positions(
    positions: List[Any],
    *,
    quotes_by_pair: Dict[str, Dict[str, float]],
    pair_charts_full: Dict[str, Dict[str, Any]],
    cot_payload: Optional[Dict[str, Any]] = None,
    option_skew_payload: Optional[Dict[str, Any]] = None,
    calendar_path: Optional[Path] = None,
    cross_asset_path: Optional[Path] = None,
    hit_rates: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[PositionThesisAssessment]:
    """Loop trader-owned positions and assess each."""
    out: List[PositionThesisAssessment] = []
    for position in positions:
        owner = getattr(position, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if owner_str.lower() != "trader":
            continue
        pair = str(getattr(position, "pair", ""))
        if not pair or pair not in pair_charts_full:
            continue
        side = getattr(position, "side", None)
        side_val = side.value if hasattr(side, "value") else str(side or "")
        side_up = side_val.upper()
        if side_up not in ("LONG", "SHORT"):
            continue
        # Current price = bid for LONG exit reference, ask for SHORT
        q = quotes_by_pair.get(pair) or {}
        if side_up == "LONG":
            cur_price = float(q.get("bid") or 0.0)
        else:
            cur_price = float(q.get("ask") or 0.0)
        if cur_price <= 0:
            continue
        chart = pair_charts_full[pair]
        # Extract regime label from chart confluence
        regime = None
        conf = chart.get("confluence") or {}
        regime_raw = str(conf.get("dominant_regime") or "").upper()
        if "TREND" in regime_raw:
            regime = "TREND"
        elif "RANGE" in regime_raw:
            regime = "RANGE"

        try:
            assessment = assess_position(
                trade_id=str(getattr(position, "trade_id", "")),
                pair=pair, side=side_up,
                current_price=cur_price,
                pair_chart=chart,
                pair_charts_full=pair_charts_full,
                cot_payload=cot_payload,
                option_skew_payload=option_skew_payload,
                calendar_path=calendar_path,
                cross_asset_path=cross_asset_path,
                hit_rates=hit_rates,
                regime=regime,
            )
            out.append(assessment)
        except Exception:
            continue
    return _reconcile_same_pair_hedges(out, positions)


def _reconcile_same_pair_hedges(
    assessments: List[PositionThesisAssessment],
    positions: List[Any],
) -> List[PositionThesisAssessment]:
    """Suppress contradictory EXTEND verdicts on same-pair opposite legs.

    This module scores each position in isolation. A same-pair LONG and SHORT
    can both receive positive detector scores when the market is ranging or
    compressing, but that is not a portfolio-level instruction to extend both
    legs. Hedge intent and unwind discipline live in the entry/gateway path; a
    position-thesis sidecar that lacks entry-thesis metadata must surface the
    conflict and defer to hedge/unwind review.
    """

    trader_units: dict[str, dict[str, int]] = {}
    for position in positions:
        owner = getattr(position, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if owner_str.lower() != "trader":
            continue
        pair = str(getattr(position, "pair", ""))
        side = getattr(position, "side", None)
        side_up = (side.value if hasattr(side, "value") else str(side or "")).upper()
        if not pair or side_up not in {"LONG", "SHORT"}:
            continue
        try:
            units = abs(int(getattr(position, "units", 0) or 0))
        except (TypeError, ValueError):
            units = 0
        trader_units.setdefault(pair, {"LONG": 0, "SHORT": 0})[side_up] += units

    by_pair: dict[str, list[PositionThesisAssessment]] = {}
    for assessment in assessments:
        by_pair.setdefault(assessment.pair, []).append(assessment)

    reconciled: list[PositionThesisAssessment] = []
    for assessment in assessments:
        units = trader_units.get(assessment.pair) or {}
        long_units = int(units.get("LONG") or 0)
        short_units = int(units.get("SHORT") or 0)
        if long_units <= 0 or short_units <= 0:
            reconciled.append(assessment)
            continue

        pair_assessments = by_pair.get(assessment.pair, [])
        opposite_side = "SHORT" if assessment.side == "LONG" else "LONG"
        opposite_extends = any(
            other.side == opposite_side and other.verdict == "EXTEND"
            for other in pair_assessments
        )
        context = (
            "same-pair trader hedge context: "
            f"long_units={long_units}, short_units={short_units}, "
            f"net_units={long_units - short_units}; "
            "position-thesis scores are independent and do not authorize both legs to extend"
        )
        if assessment.verdict == "EXTEND" and opposite_extends:
            reconciled.append(assessment.with_verdict(
                "HOLD",
                context,
                "EXTEND suppressed until the hedge/unwind plan identifies the active leg",
            ))
        else:
            reconciled.append(assessment.with_verdict(assessment.verdict, context))
    return reconciled
