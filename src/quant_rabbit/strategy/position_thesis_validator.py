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
hardening) — gpt_trader can use a fresh REVIEW_CLOSE as Gate A
evidence, but Gate B operator authorization must still pass.

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

# Position thesis is not an entry signal; it decides whether an already-open
# leg still has technical support. The score-gap prior mirrors the forecast
# layer, while the operating-TF cap keeps M5/M15/M30/H1 agreement large enough
# to overrule one micro exhaustion pattern but not large enough to force a close
# without the separate entry-buffer / Gate-A checks below.
THESIS_OPERATING_TF_ALIGNMENT_GAIN = float(os.environ.get("QR_THESIS_OPERATING_TF_ALIGNMENT_GAIN", "6.0"))
THESIS_OPERATING_TF_ALIGNMENT_CAP = float(os.environ.get("QR_THESIS_OPERATING_TF_ALIGNMENT_CAP", "18.0"))
THESIS_LOCATION_PRIOR_CAP = float(os.environ.get("QR_THESIS_LOCATION_PRIOR_CAP", "10.0"))
THESIS_STRONG_INVALIDATION_TF_COUNT = int(os.environ.get("QR_THESIS_STRONG_INVALIDATION_TF_COUNT", "3"))


@dataclass(frozen=True)
class PositionThesisAssessment:
    trade_id: str
    pair: str
    side: str
    pattern_score: float
    projection_score: float
    correlation_score: float
    path_score: float
    technical_score: float
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
            "technical_score": self.technical_score,
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
            technical_score=self.technical_score,
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
            correlation_score=0.0, path_score=0.0, technical_score=0.0,
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

    # 4b. Broad chart alignment. Pattern/projection detectors are allowed to
    # spot early fades, but they must not outrank a full M5/M15/M30/H1 panel
    # that is currently against the position.
    technical_score, technical_rat = _chart_alignment_score(pair_chart, side_up)
    if technical_score != 0:
        rationale_lines.append(f"chart-tech {technical_score:+.1f}: " + "; ".join(technical_rat[:3]))

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

    aggregate = pattern_score + projection_score + correlation_score + path_score + technical_score + reversal_score

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
        technical_score=round(technical_score, 2),
        reversal_against=reversal_against,
        aggregate_score=round(aggregate, 2),
        verdict=verdict,
        rationale_lines=tuple(rationale_lines),
    )


def _chart_alignment_score(pair_chart: Dict[str, Any], side: str) -> tuple[float, list[str]]:
    """Score whether the full chart panel still supports the position side.

    Micro exhaustion can mark the start of a turn, but an open-position thesis
    must still account for the current M5/M15/M30/H1 panel and pair-level score
    gap. This layer is deliberately bounded; it can stop a bad defer/extend, but
    the separate entry-buffer and Gate-A checks still decide REVIEW_CLOSE.
    """

    try:
        from quant_rabbit.strategy.directional_forecaster import (
            FORECAST_MARKET_LOCATION_EXTREME,
            FORECAST_SCORE_GAP_PRIOR_CAP,
            FORECAST_SCORE_GAP_PRIOR_GAIN,
            RANGE_PHASE_TIMEFRAMES,
        )
    except Exception:
        FORECAST_MARKET_LOCATION_EXTREME = 0.15
        FORECAST_SCORE_GAP_PRIOR_CAP = 30.0
        FORECAST_SCORE_GAP_PRIOR_GAIN = 35.0
        RANGE_PHASE_TIMEFRAMES = {"M5", "M15", "M30", "H1"}

    side_up = side.upper()
    if side_up not in {"LONG", "SHORT"}:
        return 0.0, []
    confluence = pair_chart.get("confluence") if isinstance(pair_chart.get("confluence"), dict) else {}
    score = 0.0
    rationales: list[str] = []

    balance = str((confluence or {}).get("score_balance") or "").upper()
    score_gap = _to_float((confluence or {}).get("score_gap"))
    if score_gap is not None and balance in {"LONG_LEAN", "SHORT_LEAN"}:
        aligned_balance = "LONG_LEAN" if side_up == "LONG" else "SHORT_LEAN"
        magnitude = min(FORECAST_SCORE_GAP_PRIOR_CAP, abs(score_gap) * FORECAST_SCORE_GAP_PRIOR_GAIN)
        signed = magnitude if balance == aligned_balance else -magnitude
        score += signed
        rationales.append(f"pair_chart {balance} score_gap={score_gap:+.3f} -> {signed:+.1f}")

    tf_score = 0.0
    tf_bits: list[str] = []
    for view in pair_chart.get("views") or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or view.get("timeframe") or "").upper()
        if tf not in RANGE_PHASE_TIMEFRAMES:
            continue
        long_bias = _to_float(view.get("long_bias"))
        short_bias = _to_float(view.get("short_bias"))
        if long_bias is None or short_bias is None:
            continue
        directional_bias = (long_bias - short_bias) if side_up == "LONG" else (short_bias - long_bias)
        if abs(directional_bias) < 1e-9:
            continue
        tf_score += directional_bias * THESIS_OPERATING_TF_ALIGNMENT_GAIN
        tf_bits.append(f"{tf} bias={directional_bias:+.2f}")
    tf_score = _clamp(tf_score, -THESIS_OPERATING_TF_ALIGNMENT_CAP, THESIS_OPERATING_TF_ALIGNMENT_CAP)
    if tf_score:
        score += tf_score
        rationales.append(f"operating TF {' '.join(tf_bits[:4])} -> {tf_score:+.1f}")

    location_score, location_note = _range_location_alignment_score(
        confluence,
        side_up,
        extreme=FORECAST_MARKET_LOCATION_EXTREME,
    )
    if location_score:
        score += location_score
        rationales.append(location_note)

    return round(score, 2), rationales


def _range_location_alignment_score(
    confluence: Dict[str, Any],
    side: str,
    *,
    extreme: float,
) -> tuple[float, str]:
    p24 = _to_float((confluence or {}).get("price_percentile_24h"))
    p7 = _to_float((confluence or {}).get("price_percentile_7d"))
    sigma = _to_float((confluence or {}).get("range_24h_sigma_multiple"))
    lower = []
    upper = []
    if p24 is not None and p24 <= extreme:
        lower.append(f"24h_pct={p24:.2f}")
    if p7 is not None and p7 <= extreme:
        lower.append(f"7d_pct={p7:.2f}")
    upper_cutoff = 1.0 - extreme
    if p24 is not None and p24 >= upper_cutoff:
        upper.append(f"24h_pct={p24:.2f}")
    if p7 is not None and p7 >= upper_cutoff:
        upper.append(f"7d_pct={p7:.2f}")

    labels = lower if side == "LONG" else upper
    against_labels = upper if side == "LONG" else lower
    if labels:
        boost = 1.0 if sigma is not None and sigma >= 2.0 else 0.7
        score = THESIS_LOCATION_PRIOR_CAP * boost
        return score, f"market location {' '.join(labels)} supports {side} mean-reversion -> {score:+.1f}"
    if against_labels:
        boost = 1.0 if sigma is not None and sigma >= 2.0 else 0.7
        score = -THESIS_LOCATION_PRIOR_CAP * boost
        return score, f"market location {' '.join(against_labels)} opposes {side} continuation -> {score:+.1f}"
    return 0.0, ""


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
    data_root: Path = Path("data"),
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
    out = _apply_entry_invalidation_overrides(
        out,
        positions,
        quotes_by_pair=quotes_by_pair,
        pair_charts_full=pair_charts_full,
        data_root=data_root,
    )
    return _reconcile_same_pair_hedges(out, positions)


def _apply_entry_invalidation_overrides(
    assessments: List[PositionThesisAssessment],
    positions: List[Any],
    *,
    quotes_by_pair: Dict[str, Dict[str, Any]],
    pair_charts_full: Dict[str, Dict[str, Any]],
    data_root: Path,
) -> List[PositionThesisAssessment]:
    """Promote trader positions whose recorded thesis invalidation is hit."""

    try:
        from quant_rabbit.strategy.entry_thesis_ledger import (
            load_entry_thesis,
            technical_invalidation_confirmation_reason,
            thesis_invalidation_hit_reason,
        )
    except Exception:
        return assessments

    by_trade_id = {str(getattr(p, "trade_id", "")): p for p in positions}
    out: list[PositionThesisAssessment] = []
    for assessment in assessments:
        position = by_trade_id.get(assessment.trade_id)
        if position is None:
            out.append(assessment)
            continue
        thesis = load_entry_thesis(assessment.trade_id, data_root)

        q = quotes_by_pair.get(assessment.pair) or {}
        price = None
        label = "bid" if assessment.side == "LONG" else "ask"
        try:
            price = float(q.get(label))
        except (TypeError, ValueError):
            price = None
        reason = None
        if thesis is not None:
            reason = thesis_invalidation_hit_reason(
                thesis,
                side=assessment.side,
                current_price=price,
                price_label=label,
            )
        if reason:
            technical_reason = technical_invalidation_confirmation_reason(
                pair_charts_full.get(assessment.pair),
                side=assessment.side,
            )
            if technical_reason:
                out.append(assessment.with_verdict(
                    "REVIEW_CLOSE",
                    reason,
                    technical_reason,
                    _post_loss_cut_reentry_note(),
                ))
            else:
                out.append(assessment.with_verdict(
                    assessment.verdict,
                    f"{reason}; waiting for chart/technical confirmation",
                ))
        else:
            adverse_reason = _entry_buffer_adverse_loss_reason(
                position=position,
                assessment=assessment,
                current_price=price,
                price_label=label,
                thesis=thesis,
            )
            if adverse_reason:
                technical_reason = technical_invalidation_confirmation_reason(
                    pair_charts_full.get(assessment.pair),
                    side=assessment.side,
                )
                if technical_reason:
                    deferred = _defer_missing_invalidation_loss_cut_for_recovery_support(
                        assessment,
                        adverse_reason=adverse_reason,
                        technical_reason=technical_reason,
                    )
                    if deferred is not None:
                        out.append(deferred)
                    else:
                        out.append(assessment.with_verdict(
                            "REVIEW_CLOSE",
                            adverse_reason,
                            technical_reason,
                            _post_loss_cut_reentry_note(),
                        ))
                else:
                    out.append(assessment.with_verdict(
                        assessment.verdict,
                        f"{adverse_reason}; waiting for chart/technical confirmation",
                    ))
            else:
                out.append(assessment)
    return out


def _post_loss_cut_reentry_note() -> str:
    return (
        "post-close re-entry discipline: close the broken recovery edge first; "
        "do not re-enter in the same receipt; refresh broker truth / intents "
        "and require a fresh LIVE_READY lane before entering again"
    )


def _defer_missing_invalidation_loss_cut_for_recovery_support(
    assessment: PositionThesisAssessment,
    *,
    adverse_reason: str,
    technical_reason: str,
) -> Optional[PositionThesisAssessment]:
    """Avoid turning a no-ledger fallback into a contradictory panic cut.

    Recorded entry invalidation still wins above. This guard applies only to
    legacy/no-invalidation positions where the fallback inferred a review from
    entry-buffer distance plus current technicals. If the same prediction stack
    that feeds position review still strongly supports the position direction,
    the correct outcome is urgent HOLD/recheck, not REVIEW_CLOSE. This keeps
    "cut early when thesis breaks" separate from "cut into a supported recovery
    just because the ledger was incomplete".
    """

    if assessment.aggregate_score < THESIS_EXTEND_THRESHOLD:
        return None
    if _strong_invalidation_tf_count(technical_reason) >= THESIS_STRONG_INVALIDATION_TF_COUNT:
        return None
    return assessment.with_verdict(
        "HOLD",
        adverse_reason,
        technical_reason,
        (
            "loss-cut deferred: current prediction stack still supports "
            f"{assessment.side} (aggregate_score={assessment.aggregate_score:.1f} >= "
            f"THESIS_EXTEND_THRESHOLD={THESIS_EXTEND_THRESHOLD:.1f}); "
            "wait for recovery support to fail or use recorded invalidation / "
            "structure Gate A"
        ),
    )


def _strong_invalidation_tf_count(technical_reason: str) -> int:
    seen: set[str] = set()
    for tf in ("M5", "M15", "M30", "H1", "H4", "D"):
        if f"{tf}:" in technical_reason:
            seen.add(tf)
    return len(seen)


def _entry_buffer_adverse_loss_reason(
    *,
    position: Any,
    assessment: PositionThesisAssessment,
    current_price: Optional[float],
    price_label: str,
    thesis: Any,
) -> Optional[str]:
    """Fallback Gate-A evidence when the entry thesis lacks invalidation geometry.

    A missing ledger or missing `invalidation_price` must not make an underwater
    position harder to review than a well-recorded one. This fallback still
    requires both adverse P/L and an entry-distance buffer; the caller adds the
    multi-timeframe technical confirmation so a wick alone cannot trigger it.
    """

    if thesis is not None and getattr(thesis, "invalidation_price", None) is not None:
        return None
    try:
        upl = float(getattr(position, "unrealized_pl_jpy", 0.0) or 0.0)
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        price = float(current_price) if current_price is not None else 0.0
    except (TypeError, ValueError):
        return None
    if upl >= 0.0 or entry_price <= 0.0 or price <= 0.0:
        return None

    try:
        from quant_rabbit.strategy.entry_thesis_ledger import (
            invalidation_buffer_price,
            thesis_invalidation_buffer_pips,
        )
    except Exception:
        return None

    side_up = assessment.side.upper()
    buffer_pips = thesis_invalidation_buffer_pips()
    buffer_price = invalidation_buffer_price(assessment.pair, buffer_pips)
    source = "no entry thesis" if thesis is None else "entry thesis lacks invalidation_price"
    if side_up == "LONG":
        trigger = entry_price - buffer_price
        if price <= trigger:
            return (
                f"adverse technical loss: {source}; current {price_label} {price:.5f} <= "
                f"entry-buffer {trigger:.5f} (entry {entry_price:.5f}, "
                f"buffer {buffer_pips:.1f}p, upl {upl:.1f} JPY)"
            )
    elif side_up == "SHORT":
        trigger = entry_price + buffer_price
        if price >= trigger:
            return (
                f"adverse technical loss: {source}; current {price_label} {price:.5f} >= "
                f"entry-buffer {trigger:.5f} (entry {entry_price:.5f}, "
                f"buffer {buffer_pips:.1f}p, upl {upl:.1f} JPY)"
            )
    return None


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
