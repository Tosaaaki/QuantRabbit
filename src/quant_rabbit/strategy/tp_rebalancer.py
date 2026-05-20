"""Dynamic TP rebalancing on open positions.

AGENT_CONTRACT §10 historically said "Existing TP is not moved by the
protection gateway." That rule was the conservative default to stop
the protection layer from silently shrinking a planned TP. But static
entry-time TP ignores live market conditions — when the regime
expands, the original TP is too tight; when it contracts, the
original TP becomes unreachable.

User directive 2026-05-13:
  「エントリー時だけではなく、途中で伸び縮みできるようにしてほしい。
   市況によって。それって当たり前だよね？」

This module reads each profit-managed position's current pair_charts
context, recomputes the market-derived reward_risk (same function as
intent_generator), derives a fresh TP distance from current ATR, and
adjusts the broker's TP order if the change exceeds the hysteresis.

Invariants:
- TP must remain on the correct side of entry (LONG: above; SHORT:
  below) and at least `MIN_TP_TO_MARKET_PIPS` from current price so
  the rebalance never fires the TP accidentally on the same tick.
- Trader-owned plus operator-managed manual / unknown-owner positions
  are eligible for TP-only profit capture. External positions are skipped.
- Missing broker TP is not auto-created by default. A manually deleted TP is
  treated as an operator decision to run without a broker cap. Exception:
  when a runner is already in profit and the latest forecast / technical
  stack cannot justify carrying it into the next session, the rebalancer may
  place an insurance TP close to current reward-side price. Set
  `QR_ENABLE_MISSING_TP_REPAIR=1` to restore the old unconditional repair
  behavior for manual / unknown-owner positions.
- SL is NEVER touched here. This module is TP-only. The SL-free
  invariant `stop_loss is None` is respected by skipping any
  attempt to read/write SL.
- Hysteresis: ignore changes smaller than `HYSTERESIS_PIPS` to avoid
  noise-driven broker churn.
- Kill switch: `QR_DISABLE_TP_REBALANCE=1` short-circuits the
  rebalancer to a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


HYSTERESIS_PIPS = float(os.environ.get("QR_TP_REBALANCE_HYSTERESIS_PIPS", "10"))
MIN_TP_TO_MARKET_PIPS = float(os.environ.get("QR_TP_REBALANCE_MIN_TP_TO_MARKET", "5"))
MAX_TP_DISTANCE_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_MAX_DISTANCE_ATR", "10"))

# Contract-mode tuning (2026-05-13 second iteration after user feedback).
# When a position is ≥ ADVERSE_ATR_MULT × ATR underwater AND no reversal
# signal is firing, the trader pulls TP closer to entry to lock in a
# small bounce-back profit instead of waiting for the original wide
# target that may never be hit. User directive:「下げ基調のとき TP を狭める
# のはいい。下げを否定し始めたときに TP 広げて利鞘を稼ぐ」.
ADVERSE_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_ADVERSE_ATR_MULT", "1.0"))
MIN_LOCK_IN_PIPS = float(os.environ.get("QR_TP_REBALANCE_MIN_LOCK_IN_PIPS", "8"))
LOCK_IN_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_LOCK_IN_ATR_MULT", "0.5"))

# Trailing-TP tuning (2026-05-14): when a position is in profit ≥
# TRAILING_TRIGGER_ATR_MULT × ATR, the rebalancer also considers a
# "trailing" candidate anchored on CURRENT PRICE (not entry):
# `current_price ± TRAILING_LOCK_BEHIND_ATR_MULT × ATR`. The new TP is
# the MAX of (entry-anchored desired, trailing candidate, existing) so
# it only moves further away. As price advances into profit, the
# trailing branch becomes binding and pushes TP ahead, extending the
# planned win. User feedback 2026-05-14:「平均勝ち +460 が小さい、
# 利鞘を稼げてない」.
TRAILING_TRIGGER_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_TRAILING_TRIGGER_ATR_MULT", "1.0"))
TRAILING_LOCK_BEHIND_ATR_MULT = float(os.environ.get("QR_TP_REBALANCE_TRAILING_LOCK_BEHIND_ATR_MULT", "1.5"))


@dataclass(frozen=True)
class TPAdjustment:
    trade_id: str
    pair: str
    side: str
    entry_price: float
    current_tp: Optional[float]
    new_tp: float
    distance_pips_old: float
    distance_pips_new: float
    rationale: str


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _round_price(pair: str, price: float) -> float:
    return round(price, 3 if pair.endswith("_JPY") else 5)


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_TP_REBALANCE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _missing_tp_repair_enabled() -> bool:
    return os.environ.get("QR_ENABLE_MISSING_TP_REPAIR", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _insurance_tp_disabled() -> bool:
    return os.environ.get("QR_DISABLE_INSURANCE_TP", "").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _profit_take_owner_allowed(owner: str) -> bool:
    return owner.strip().lower() in {"trader", "manual", "unknown"}


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _forecast_direction_for_side(side: str) -> str:
    return "UP" if side.upper() == "LONG" else "DOWN"


def _forecast_runner_drag_reasons(
    *,
    side: str,
    latest_forecast: Optional[Dict[str, Any]],
) -> list[str]:
    """Return forecast reasons that weaken carrying a runner further."""
    if not latest_forecast:
        return []
    aligned_direction = _forecast_direction_for_side(side)
    direction = str(latest_forecast.get("direction") or "UNCLEAR").upper()
    confidence = _optional_float(latest_forecast.get("confidence")) or 0.0
    if direction in {"RANGE", "UNCLEAR"}:
        return [f"forecast {direction} has no directional runner edge"]
    if direction != aligned_direction:
        return [f"forecast {direction} opposes {side.upper()} runner"]
    try:
        from quant_rabbit.strategy.directional_forecaster import ENTRY_CONFIDENCE_MIN
    except Exception:
        ENTRY_CONFIDENCE_MIN = 0.55
    if confidence < ENTRY_CONFIDENCE_MIN:
        return [
            f"forecast {direction} confidence {confidence:.2f} below runner threshold {ENTRY_CONFIDENCE_MIN:.2f}"
        ]
    return []


def _insurance_tp_reasons(
    *,
    side: str,
    profit_pips: float,
    atr_pips: float,
    latest_forecast: Optional[Dict[str, Any]],
    chart_context: Optional[Dict[str, Any]],
) -> list[str]:
    """Return reasons to add an insurance TP to a TP-less runner.

    This is intentionally stricter than missing-TP repair: it only acts on
    positions already in profit, because the purpose is MFE capture / session
    handoff insurance, not capping a losing position's recovery path.
    """
    if _insurance_tp_disabled() or profit_pips <= 0:
        return []
    milestone_reasons: list[str] = []
    forecast_reasons: list[str] = []
    session_reasons: list[str] = []

    try:
        from quant_rabbit.strategy.dynamic_position_policy import trailing_trigger_mult

        trigger_mult, trigger_reasons = trailing_trigger_mult(chart_context)
    except Exception:
        trigger_mult, trigger_reasons = None, []
    if trigger_mult is not None and atr_pips > 0 and profit_pips >= trigger_mult * atr_pips:
        reason = f"profit {profit_pips:.1f}pip reached runner milestone {trigger_mult:.2f}×ATR"
        if trigger_reasons:
            reason += " (" + "; ".join(trigger_reasons[:2]) + ")"
        milestone_reasons.append(reason)

    forecast_reasons.extend(
        _forecast_runner_drag_reasons(side=side, latest_forecast=latest_forecast)
    )
    if latest_forecast:
        horizon_min = _optional_float(latest_forecast.get("horizon_min")) or 0.0
        session = (chart_context or {}).get("session") if isinstance(chart_context, dict) else {}
        next_minutes = _optional_float((session or {}).get("minutes_to_next_killzone"))
        if next_minutes is not None and (horizon_min <= 0 or horizon_min < next_minutes):
            session_reasons.append(
                f"forecast horizon {horizon_min:.0f}m does not cover next session in {next_minutes:.0f}m"
            )

    technical_reasons = _technical_harvest_pressure(side=side, chart_context=chart_context)
    technical_stack_reasons = technical_reasons if len(technical_reasons) >= 2 else []
    if not technical_stack_reasons and not (forecast_reasons and (milestone_reasons or session_reasons)):
        return []
    return (milestone_reasons + forecast_reasons + session_reasons + technical_stack_reasons)[:8]


def _existing_tp_harvest_reasons(
    *,
    side: str,
    profit_pips: float,
    latest_forecast: Optional[Dict[str, Any]],
    chart_context: Optional[Dict[str, Any]],
) -> list[str]:
    """Return reasons to contract an existing profitable TP for MFE capture.

    Existing broker TPs are normally runners. Contracting them requires both:
    the forecast no longer supports directional carry, and the technical stack
    says current MFE is at risk. That keeps the 2026-05-13 expand-only invariant
    for ordinary winners while still allowing the 2026-05-20 "do not sit on a
    stale full-distance hedge TP" repair.
    """
    if _insurance_tp_disabled() or profit_pips <= 0:
        return []
    forecast_reasons = _forecast_runner_drag_reasons(
        side=side,
        latest_forecast=latest_forecast,
    )
    if not forecast_reasons:
        return []
    technical_reasons = _technical_harvest_pressure(
        side=side,
        chart_context=chart_context,
    )
    # Two independent technical warnings are the same stack boundary used by
    # TP-less runner insurance; one isolated overbought/oversold print is noise.
    if len(technical_reasons) < 2:
        return []
    return (forecast_reasons + technical_reasons)[:8]


def _technical_harvest_pressure(*, side: str, chart_context: Optional[Dict[str, Any]]) -> list[str]:
    if not chart_context:
        return []
    side_up = side.upper()
    confluence = chart_context.get("confluence") or {}
    reasons: list[str] = []

    price_pct_24h = _optional_float(confluence.get("price_percentile_24h"))
    price_pct_7d = _optional_float(confluence.get("price_percentile_7d"))
    if side_up == "LONG":
        if price_pct_24h is not None and price_pct_24h >= 0.95:
            reasons.append(f"LONG at 24h price percentile {price_pct_24h:.2f} (upper extreme)")
        if price_pct_7d is not None and price_pct_7d >= 0.95:
            reasons.append(f"LONG at 7d price percentile {price_pct_7d:.2f} (upper extreme)")
    else:
        if price_pct_24h is not None and price_pct_24h <= 0.05:
            reasons.append(f"SHORT at 24h price percentile {price_pct_24h:.2f} (lower extreme)")
        if price_pct_7d is not None and price_pct_7d <= 0.05:
            reasons.append(f"SHORT at 7d price percentile {price_pct_7d:.2f} (lower extreme)")

    tf_agreement = _optional_float(confluence.get("tf_agreement_score"))
    # 2/3 is the documented majority boundary on the M15/M30/H1 panel.
    if tf_agreement is not None and tf_agreement < (2.0 / 3.0):
        reasons.append(f"TF agreement {tf_agreement:.2f} below majority")
    sigma_24h = _optional_float(confluence.get("range_24h_sigma_multiple"))
    # 2σ is the existing exhaustion boundary used by intent generation.
    if sigma_24h is not None and sigma_24h >= 2.0:
        reasons.append(f"24h range {sigma_24h:.2f}σ exhausted")

    for tf in ("M15", "M30", "H1"):
        indicators = (chart_context.get("indicators_by_tf") or {}).get(tf) or {}
        rsi = _optional_float(indicators.get("rsi_14"))
        stoch_rsi = _optional_float(indicators.get("stoch_rsi"))
        williams = _optional_float(indicators.get("williams_r_14"))
        close = _optional_float(indicators.get("close"))
        bb_upper = _optional_float(indicators.get("bb_upper"))
        bb_lower = _optional_float(indicators.get("bb_lower"))
        donchian_high = _optional_float(indicators.get("donchian_high"))
        donchian_low = _optional_float(indicators.get("donchian_low"))

        # RSI 70/30, StochRSI 0.8/0.2, and Williams %R -20/-80 are
        # standard overbought/oversold technical boundaries, not tuned
        # QuantRabbit profit targets.
        if side_up == "LONG":
            if rsi is not None and rsi >= 70:
                reasons.append(f"{tf} RSI {rsi:.1f} overbought")
            if stoch_rsi is not None and stoch_rsi >= 0.8:
                reasons.append(f"{tf} StochRSI {stoch_rsi:.2f} overbought")
            if williams is not None and williams >= -20:
                reasons.append(f"{tf} Williams %R {williams:.1f} overbought")
            if close is not None and bb_upper is not None and close >= bb_upper:
                reasons.append(f"{tf} close at/above BB upper")
            if close is not None and donchian_high is not None and close >= donchian_high:
                reasons.append(f"{tf} close at/above Donchian high")
        else:
            if rsi is not None and rsi <= 30:
                reasons.append(f"{tf} RSI {rsi:.1f} oversold")
            if stoch_rsi is not None and stoch_rsi <= 0.2:
                reasons.append(f"{tf} StochRSI {stoch_rsi:.2f} oversold")
            if williams is not None and williams <= -80:
                reasons.append(f"{tf} Williams %R {williams:.1f} oversold")
            if close is not None and bb_lower is not None and close <= bb_lower:
                reasons.append(f"{tf} close at/below BB lower")
            if close is not None and donchian_low is not None and close <= donchian_low:
                reasons.append(f"{tf} close at/below Donchian low")
    return reasons[:6]


def _structural_existing_tp_harvest_candidate(
    *,
    pair: str,
    side: str,
    entry_price: float,
    current_price: float,
    current_tp: float,
    pair_chart: Optional[Dict[str, Any]],
    pip_factor: int,
) -> tuple[Optional[float], str]:
    """Pick a closer market-derived TP from the structural HARVEST ladder."""
    if not pair_chart:
        return None, "no pair_chart structural ladder"
    from quant_rabbit.strategy.price_action import structural_tp_target

    side_up = side.upper()
    safety_margin = MIN_TP_TO_MARKET_PIPS / pip_factor
    skipped: list[str] = []
    for intent in ("HARVEST", "EXTEND"):
        candidate, anchor_reason = structural_tp_target(
            pair_chart,
            side=side,
            current_price=current_price,
            pip_factor=float(pip_factor),
            intent=intent,
        )
        if candidate is None:
            skipped.append(f"{intent}: {anchor_reason}")
            continue
        candidate_tp = _round_price(pair, float(candidate))
        if side_up == "LONG":
            if candidate_tp >= current_tp:
                skipped.append(f"{intent}: {anchor_reason} not closer than current TP")
                continue
            if candidate_tp < current_price + safety_margin:
                skipped.append(f"{intent}: {anchor_reason} inside market safety margin")
                continue
            if candidate_tp <= entry_price:
                skipped.append(f"{intent}: {anchor_reason} not on reward side")
                continue
        else:
            if candidate_tp <= current_tp:
                skipped.append(f"{intent}: {anchor_reason} not closer than current TP")
                continue
            if candidate_tp > current_price - safety_margin:
                skipped.append(f"{intent}: {anchor_reason} inside market safety margin")
                continue
            if candidate_tp >= entry_price:
                skipped.append(f"{intent}: {anchor_reason} not on reward side")
                continue
        change_pips = round(abs(candidate_tp - current_tp) * pip_factor, 1)
        if change_pips < HYSTERESIS_PIPS:
            skipped.append(f"{intent}: {anchor_reason} change {change_pips:.1f}pip below hysteresis")
            continue
        if intent == "EXTEND":
            anchor_label = anchor_reason.removeprefix("EXTEND→").replace(" (skip first)", "")
            anchor_reason = f"next HARVEST anchor after unsafe nearest ({anchor_label})"
        return candidate_tp, anchor_reason
    return None, "; ".join(skipped[:4]) or "no usable structural anchor"


def compute_tp_adjustment(
    *,
    trade_id: str,
    pair: str,
    side: str,
    entry_price: float,
    current_tp: Optional[float],
    current_price: float,
    atr_pips: float,
    reward_risk: float,
    is_reversal_firing: bool = False,
    owner: str = "trader",
    chart_context: Optional[Dict[str, Any]] = None,
    latest_forecast: Optional[Dict[str, Any]] = None,
    pair_chart: Optional[Dict[str, Any]] = None,
) -> Optional[TPAdjustment]:
    """Compute a new TP for one position.

    Core modes (2026-05-13 iteration after the 471029 regression):

    1. **expand_reversal** — reversal signal fires for this side:
       use the full `reward_risk × ATR` distance from entry, letting
       the bounce run further. May expand or be a no-op (never
       contract here — reversal means we want MORE room).
    2. **contract_adverse** — position is ≥ `ADVERSE_ATR_MULT × ATR`
       underwater AND no reversal signal: pull TP to
       `entry + max(MIN_LOCK_IN_PIPS, atr × LOCK_IN_ATR_MULT)` (LONG;
       mirrored for SHORT). User directive 2026-05-13:「下げ基調の
       とき TP を狭めるのはいい。下げを否定し始めたときに TP 広げて
       利鞘を稼がないといけないんじゃないの？」 — settle for a small
       bounce-back profit instead of waiting for the original wide
       target that may never be hit.
    3. **expand_only** — otherwise (in profit, or barely adverse, no
       reversal signal): TP can only move FURTHER from entry. The
       original "let winners run" rule.

    4. **forecast_harvest** — profitable existing TP with forecast drag and
       a technical exhaustion stack: move TP closer only to a structural
       HARVEST anchor from pair_charts. No structural anchor = HOLD.

    Returns None when the position should not be adjusted (non-managed
    owner, trader-owned missing TP, change below hysteresis, safety violation).
    """
    if _is_disabled():
        return None
    owner_normalized = owner.strip().lower()
    if not _profit_take_owner_allowed(owner_normalized):
        return None
    manual_missing_tp_repair = (
        current_tp is None
        and owner_normalized in {"manual", "unknown"}
        and _missing_tp_repair_enabled()
    )
    if atr_pips <= 0 or reward_risk <= 0:
        return None

    pip_factor = _pip_factor(pair)
    pip_size = 1.0 / pip_factor
    side_up = side.upper()
    if side_up not in ("LONG", "SHORT"):
        return None

    if side_up == "LONG":
        profit_pips = (current_price - entry_price) * pip_factor
    else:
        profit_pips = (entry_price - current_price) * pip_factor
    insurance_reasons = (
        _insurance_tp_reasons(
            side=side_up,
            profit_pips=profit_pips,
            atr_pips=atr_pips,
            latest_forecast=latest_forecast,
            chart_context=chart_context,
        )
        if current_tp is None
        else []
    )
    existing_tp_harvest_reasons = (
        _existing_tp_harvest_reasons(
            side=side_up,
            profit_pips=profit_pips,
            latest_forecast=latest_forecast,
            chart_context=chart_context,
        )
        if current_tp is not None
        else []
    )
    insurance_missing_tp = current_tp is None and bool(insurance_reasons)
    if current_tp is None and not (manual_missing_tp_repair or insurance_missing_tp):
        return None

    distance_old = abs(current_tp - entry_price) * pip_factor if current_tp is not None else 0.0
    desired_distance_pips = min(reward_risk * atr_pips, MAX_TP_DISTANCE_ATR_MULT * atr_pips)

    # Adverse detection (only matters for contract_adverse mode).
    if side_up == "LONG":
        is_adverse = current_price < entry_price
        adverse_pips = (entry_price - current_price) * pip_factor
    else:
        is_adverse = current_price > entry_price
        adverse_pips = (current_price - entry_price) * pip_factor
    is_significant_adverse = is_adverse and adverse_pips >= ADVERSE_ATR_MULT * atr_pips

    # Pick mode.
    harvest_anchor_reason: Optional[str] = None
    if insurance_missing_tp:
        mode = "insurance_tp"
        lock_in_pips = max(MIN_LOCK_IN_PIPS, atr_pips * LOCK_IN_ATR_MULT)
        if side_up == "LONG":
            if profit_pips > 0:
                candidate_tp = current_price + MIN_TP_TO_MARKET_PIPS * pip_size
            else:
                candidate_tp = entry_price + lock_in_pips * pip_size
        else:
            if profit_pips > 0:
                candidate_tp = current_price - MIN_TP_TO_MARKET_PIPS * pip_size
            else:
                candidate_tp = entry_price - lock_in_pips * pip_size
    elif manual_missing_tp_repair:
        mode = "manual_tp_repair"
        if side_up == "LONG":
            entry_anchored = entry_price + desired_distance_pips * pip_size
            market_safe = current_price + MIN_TP_TO_MARKET_PIPS * pip_size
            candidate_tp = max(entry_anchored, market_safe)
        else:
            entry_anchored = entry_price - desired_distance_pips * pip_size
            market_safe = current_price - MIN_TP_TO_MARKET_PIPS * pip_size
            candidate_tp = min(entry_anchored, market_safe)
    elif is_reversal_firing:
        mode = "expand_reversal"
        if side_up == "LONG":
            candidate_tp = entry_price + desired_distance_pips * pip_size
        else:
            candidate_tp = entry_price - desired_distance_pips * pip_size
        # In reversal mode we never want to contract; if the desired
        # distance is shorter than the old TP, leave it alone.
        if desired_distance_pips <= distance_old:
            return None
    elif existing_tp_harvest_reasons:
        mode = "forecast_harvest"
        candidate_tp, harvest_anchor_reason = _structural_existing_tp_harvest_candidate(
            pair=pair,
            side=side_up,
            entry_price=entry_price,
            current_price=current_price,
            current_tp=float(current_tp),
            pair_chart=pair_chart,
            pip_factor=pip_factor,
        )
        if candidate_tp is None:
            return None
    elif is_significant_adverse:
        mode = "contract_adverse"
        lock_in_pips = max(MIN_LOCK_IN_PIPS, atr_pips * LOCK_IN_ATR_MULT)
        if side_up == "LONG":
            candidate_tp = entry_price + lock_in_pips * pip_size
        else:
            candidate_tp = entry_price - lock_in_pips * pip_size
        # Only fire if the lock-in TP is actually CLOSER than the
        # existing TP (we're contracting). If it would EXPAND, fall
        # through to expand_only mode which has stricter rules.
        if abs(candidate_tp - entry_price) >= distance_old:
            mode = "expand_only"
            if desired_distance_pips <= distance_old:
                return None
            if side_up == "LONG":
                candidate_tp = entry_price + desired_distance_pips * pip_size
            else:
                candidate_tp = entry_price - desired_distance_pips * pip_size
    else:
        mode = "expand_only"
        # Entry-anchored candidate.
        if side_up == "LONG":
            entry_anchored = entry_price + desired_distance_pips * pip_size
        else:
            entry_anchored = entry_price - desired_distance_pips * pip_size

        # Trailing candidate: anchored on CURRENT PRICE so TP advances
        # as price moves into profit. Trigger AND lock-behind both
        # market-derived from chart_context (AGENT_CONTRACT §3.5).
        from quant_rabbit.strategy.dynamic_position_policy import (
            trailing_trigger_mult,
            trailing_lock_behind_mult,
        )
        trigger_mult, _ = trailing_trigger_mult(chart_context)
        lock_mult, _ = trailing_lock_behind_mult(chart_context)
        profit_pips: float
        if side_up == "LONG":
            profit_pips = (current_price - entry_price) * pip_factor
        else:
            profit_pips = (entry_price - current_price) * pip_factor
        trailing_eligible = profit_pips >= trigger_mult * atr_pips
        if trailing_eligible:
            trail_distance_pips = lock_mult * atr_pips
            if side_up == "LONG":
                trailing_anchored = current_price + trail_distance_pips * pip_size
            else:
                trailing_anchored = current_price - trail_distance_pips * pip_size
        else:
            trailing_anchored = entry_anchored  # neutralize

        # Pick the candidate that is FURTHEST from entry (= the most
        # aggressive let-winners-run target).
        if side_up == "LONG":
            candidate_tp = max(entry_anchored, trailing_anchored)
        else:
            candidate_tp = min(entry_anchored, trailing_anchored)

        # Now apply the expand-only invariant.
        candidate_distance = abs(candidate_tp - entry_price) * pip_factor
        if candidate_distance <= distance_old:
            return None
        if trailing_eligible and candidate_tp == trailing_anchored:
            mode = "trailing"

    # Safety: TP must not fire immediately. Keep at least
    # MIN_TP_TO_MARKET_PIPS distance from current price. TP must remain
    # on the correct side of entry (avoid locking in a loss).
    safety_margin = MIN_TP_TO_MARKET_PIPS * pip_size
    if side_up == "LONG":
        if candidate_tp < current_price + safety_margin:
            return None
        if candidate_tp <= entry_price:
            return None
    else:
        if candidate_tp > current_price - safety_margin:
            return None
        if candidate_tp >= entry_price:
            return None

    new_tp = _round_price(pair, candidate_tp)
    # Round to 1 decimal place before comparison so float precision
    # noise (e.g., 9.99999 vs 10.0) doesn't block legitimate adjustments.
    change_pips = round(abs(new_tp - current_tp) * pip_factor, 1) if current_tp is not None else round(abs(new_tp - entry_price) * pip_factor, 1)
    if current_tp is not None and change_pips < HYSTERESIS_PIPS:
        return None

    distance_new = abs(new_tp - entry_price) * pip_factor
    direction_label = "expanded" if distance_new > distance_old else "contracted"
    rationale = (
        f"TP {direction_label} {distance_old:.1f}→{distance_new:.1f}pip "
        f"(mode={mode}, reward_risk={reward_risk:.2f}, atr={atr_pips:.1f}pip, "
        f"change={change_pips:.1f}pip)"
    )
    if insurance_missing_tp:
        rationale += "; insurance: " + "; ".join(insurance_reasons[:4])
    if mode == "forecast_harvest":
        rationale += "; harvest: " + "; ".join(existing_tp_harvest_reasons[:4])
        if harvest_anchor_reason:
            rationale += f"; anchor: {harvest_anchor_reason}"
    return TPAdjustment(
        trade_id=trade_id,
        pair=pair,
        side=side_up,
        entry_price=entry_price,
        current_tp=current_tp,
        new_tp=new_tp,
        distance_pips_old=distance_old,
        distance_pips_new=distance_new,
        rationale=rationale,
    )


def compute_all_tp_adjustments(
    *,
    positions: Iterable[Any],
    quotes: Dict[str, Dict[str, float]],
    pair_charts: Dict[str, Dict[str, Any]],
    market_reward_risk_fn,
    latest_forecasts_by_pair: Optional[Dict[str, Dict[str, Any]]] = None,
) -> list[TPAdjustment]:
    """Loop profit-managed positions and compute TP adjustments.

    `quotes` is the per-pair bid/ask dict from broker_snapshot.
    `pair_charts` is the keyed-by-pair dict (same shape as
    `trader_brain._load_full_pair_charts_for_brain`'s return value).
    `market_reward_risk_fn(chart_context)` is injected so the caller
    can use the same dynamic reward_risk computation as
    intent_generator (avoids a circular import).
    """
    if _is_disabled():
        return []
    adjustments: list[TPAdjustment] = []
    for position in positions:
        owner = getattr(position, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if not _profit_take_owner_allowed(owner_str):
            continue
        pair = getattr(position, "pair", None)
        if not pair or pair not in quotes:
            continue
        # Current price for the side's exit:
        # LONG exits with sell at bid; SHORT exits with buy at ask.
        side = getattr(position, "side", None)
        side_value = side.value if hasattr(side, "value") else str(side or "")
        side_up = side_value.upper()
        quote = quotes.get(pair) or {}
        if side_up == "LONG":
            current_price = float(quote.get("bid") or 0.0)
        elif side_up == "SHORT":
            current_price = float(quote.get("ask") or 0.0)
        else:
            continue
        if current_price <= 0:
            continue

        chart = pair_charts.get(pair) or {}
        # Compute fresh reward_risk via injected fn (intent_generator's
        # `_market_derived_reward_risk` returns (rr, rationale_list)).
        try:
            reward_risk_value, _ = market_reward_risk_fn(_chart_context_from_chart(chart))
        except Exception:
            continue

        atr_pips = _extract_atr_pips(chart, pair)
        if atr_pips is None or atr_pips <= 0:
            continue

        # Check reversal signal for the position's direction. If firing,
        # tp_rebalancer enters expand_reversal mode and the contract
        # branch is short-circuited.
        try:
            from quant_rabbit.strategy.reversal_signal import detect_reversal as _detect_reversal
            reversal = _detect_reversal(chart, side_up)
        except Exception:
            reversal = None

        adj = compute_tp_adjustment(
            trade_id=str(getattr(position, "trade_id", "")),
            pair=pair,
            side=side_up,
            entry_price=float(getattr(position, "entry_price", 0.0)),
            current_tp=_optional_float(getattr(position, "take_profit", None)),
            current_price=current_price,
            atr_pips=atr_pips,
            reward_risk=float(reward_risk_value),
            is_reversal_firing=(reversal is not None),
            owner=owner_str.lower(),
            chart_context=_chart_context_from_chart(chart),
            latest_forecast=(latest_forecasts_by_pair or {}).get(pair),
            pair_chart=chart,
        )
        if adj is not None:
            adjustments.append(adj)
    return adjustments


def _chart_context_from_chart(chart: Dict[str, Any]) -> Dict[str, Any]:
    """Project the full pair_chart entry to the flat context shape that
    `_market_derived_reward_risk` expects.

    pair_charts views use `granularity` (not `timeframe`) and put
    indicators under `indicators.{adx_14, atr_pips, ...}`.
    """
    context: Dict[str, Any] = {}
    context["confluence"] = chart.get("confluence") or {}
    context["session"] = chart.get("session") if isinstance(chart.get("session"), dict) else {}
    context["indicators_by_tf"] = {}
    for view in chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or view.get("timeframe") or view.get("tf") or "").upper()
        indicators = view.get("indicators") or {}
        if tf:
            context["indicators_by_tf"][tf] = indicators
        adx = indicators.get("adx_14") or indicators.get("adx") or indicators.get("ADX")
        if adx is None:
            continue
        if tf == "H1":
            context["h1_adx"] = adx
        elif tf == "H4":
            context["h4_adx"] = adx
    context["session_current_tag"] = (
        chart.get("session_current_tag")
        or chart.get("session_bucket")
        or chart.get("session")
        or (chart.get("confluence") or {}).get("session_current_tag")
    )
    return context


def _extract_atr_pips(chart: Dict[str, Any], pair: str) -> Optional[float]:
    """Pull current ATR (in pips) from chart.

    Preference order (pair_charts schema 2026-05-13):
    1. confluence.h4_atr_pips (intent_generator pipeline projection)
    2. confluence.atr_pips
    3. views[granularity=H4].indicators.atr_pips
    4. views[granularity=H1].indicators.atr_pips
    5. Any view's indicators.atr_pips
    """
    confluence = chart.get("confluence") or {}
    for key in ("h4_atr_pips", "h1_atr_pips", "atr_pips"):
        raw = confluence.get(key)
        if raw is None:
            continue
        try:
            v = float(raw)
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue

    # Preferred per-granularity lookup.
    preference = ("H4", "H1", "M30", "M15", "M5", "M1", "D")
    by_gran: Dict[str, float] = {}
    for view in chart.get("views", []) or []:
        if not isinstance(view, dict):
            continue
        tf = str(view.get("granularity") or view.get("timeframe") or view.get("tf") or "").upper()
        indicators = view.get("indicators") or {}
        raw = indicators.get("atr_pips")
        if raw is None:
            continue
        try:
            v = float(raw)
            if v > 0:
                by_gran[tf] = v
        except (TypeError, ValueError):
            continue
    for tf in preference:
        if tf in by_gran:
            return by_gran[tf]
    return None


def apply_tp_adjustments(
    adjustments: Iterable[TPAdjustment],
    broker_client: Any,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Send the TP modify requests to the broker.

    Returns a list of result dicts (one per adjustment, including
    failed calls — broker exceptions are logged into the dict, never
    raised, so a single failure doesn't block other adjustments).
    """
    results: list[dict] = []
    for adj in adjustments:
        entry = {
            "trade_id": adj.trade_id,
            "pair": adj.pair,
            "side": adj.side,
            "current_tp": adj.current_tp,
            "new_tp": adj.new_tp,
            "distance_pips_old": adj.distance_pips_old,
            "distance_pips_new": adj.distance_pips_new,
            "rationale": adj.rationale,
            "sent": False,
            "error": None,
        }
        if dry_run:
            results.append(entry)
            continue
        try:
            broker_client.replace_trade_dependent_orders(
                adj.trade_id,
                {"takeProfit": {"price": f"{adj.new_tp:.5f}".rstrip("0").rstrip(".") if not adj.pair.endswith("_JPY") else f"{adj.new_tp:.3f}", "timeInForce": "GTC"}},
            )
            entry["sent"] = True
        except Exception as exc:  # noqa: BLE001 — keep loop running
            entry["error"] = str(exc)
        results.append(entry)
    return results
