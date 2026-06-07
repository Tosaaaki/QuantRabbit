"""Adverse-continuation partial close — free margin without abandoning the thesis.

User feedback 2026-05-14:
  「保有中の管理が空白。margin 91% 詰まりで攻めれない」

When a position is significantly underwater AND no reversal signal is
firing AND the macro context doesn't support the side, the trader has
three bad choices under SL-free mode:

1. Wait for TP (may never fire if direction is wrong)
2. Wait for thesis-invalidation manual close (Gate A requires M15/H4
   BOS — slow; meanwhile margin stays locked)
3. Manual close at a loss (Gate B requires operator token)

This module adds a fourth, milder option: **partial close** (e.g.,
50% of units) to:
- Free up margin so other lanes can be funded
- Keep some exposure if the thesis eventually plays out
- Lock in a portion of the loss without abandoning the trade

This is NOT a full CLOSE so the AGENT_CONTRACT §10 two-gate (A+B)
requirements don't apply — those guard against accidental full-close
of a still-valid thesis. Partial close for margin management is a
risk-control action, not a thesis decision.

Trigger conditions (all must be true):
- Position is trader-owned (manual positions skipped)
- Currently underwater ≥ `ADVERSE_PARTIAL_TRIGGER_ATR_MULT × ATR`
- No reversal_signal firing for this side
- Position units >= `MIN_POSITION_UNITS_FOR_PARTIAL` (don't partial
  a position that's already too small to halve meaningfully)

Kill switch: `QR_DISABLE_ADVERSE_PARTIAL_CLOSE=1`.
CLI live sends are dry-run by default and require `--send --confirm-live`
plus `QR_LIVE_ENABLED=1`; this module must not become an implicit close path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


ADVERSE_PARTIAL_TRIGGER_ATR_MULT = float(
    os.environ.get("QR_ADVERSE_PARTIAL_TRIGGER_ATR_MULT", "1.5")
)
PARTIAL_CLOSE_FRACTION = float(os.environ.get("QR_ADVERSE_PARTIAL_FRACTION", "0.5"))
MIN_POSITION_UNITS_FOR_PARTIAL = int(
    os.environ.get("QR_ADVERSE_PARTIAL_MIN_UNITS", "2000")
)


@dataclass(frozen=True)
class PartialCloseAction:
    trade_id: str
    pair: str
    side: str
    original_units: int
    close_units: int  # ALWAYS POSITIVE — broker close API takes absolute units
    remaining_units: int
    adverse_pips: float
    atr_pips: float
    rationale: str


def _pip_factor(pair: str) -> int:
    return 100 if pair.endswith("_JPY") else 10000


def _is_disabled() -> bool:
    return os.environ.get("QR_DISABLE_ADVERSE_PARTIAL_CLOSE", "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }


def compute_partial_close(
    *,
    trade_id: str,
    pair: str,
    side: str,
    units: int,
    entry_price: float,
    current_price: float,
    atr_pips: float,
    is_reversal_firing: bool,
    owner: str = "trader",
    chart_context: Optional[Dict[str, Any]] = None,
) -> Optional[PartialCloseAction]:
    """Compute a partial-close action for one position.

    Returns None when no action should be taken (kill-switched,
    wrong owner, not enough adverse, reversal firing, position too
    small to halve).
    """
    if _is_disabled():
        return None
    if owner != "trader":
        return None
    if atr_pips <= 0:
        return None
    if is_reversal_firing:
        # Reversal could be playing out → don't reduce exposure now.
        return None

    pip_factor = _pip_factor(pair)
    side_up = side.upper()
    if side_up == "LONG":
        adverse_pips = (entry_price - current_price) * pip_factor
    elif side_up == "SHORT":
        adverse_pips = (current_price - entry_price) * pip_factor
    else:
        return None

    # Market-derived threshold & fraction (AGENT_CONTRACT §3.5).
    # Higher-TF AGAINST → close earlier + bigger chunk.
    # Higher-TF WITH → give room + keep more on.
    from quant_rabbit.strategy.dynamic_position_policy import (
        adverse_trigger_mult,
        partial_close_fraction,
    )
    trigger_mult, _ = adverse_trigger_mult(chart_context, side_up)
    fraction, _ = partial_close_fraction(chart_context, side_up)
    threshold_pips = trigger_mult * atr_pips

    if adverse_pips < threshold_pips:
        return None

    absolute_units = abs(int(units))
    if absolute_units < MIN_POSITION_UNITS_FOR_PARTIAL:
        return None

    close_units_raw = int(absolute_units * fraction)
    close_units = (close_units_raw // 100) * 100
    if close_units < 100:
        return None
    if close_units >= absolute_units:
        return None

    remaining = absolute_units - close_units
    rationale = (
        f"adverse {adverse_pips:.1f}pip ≥ {trigger_mult:.2f}×ATR "
        f"({threshold_pips:.1f}pip), no reversal → close "
        f"{close_units}/{absolute_units} ({fraction:.0%}), remaining {remaining}"
    )
    return PartialCloseAction(
        trade_id=trade_id,
        pair=pair,
        side=side_up,
        original_units=absolute_units,
        close_units=close_units,
        remaining_units=remaining,
        adverse_pips=adverse_pips,
        atr_pips=atr_pips,
        rationale=rationale,
    )


def compute_all_partial_closes(
    *,
    positions: Iterable[Any],
    quotes: Dict[str, Dict[str, float]],
    pair_charts: Dict[str, Dict[str, Any]],
) -> list[PartialCloseAction]:
    """Loop trader-owned positions and compute partial-close actions."""
    if _is_disabled():
        return []
    # Lazy imports to avoid circulars + keep this module standalone.
    from quant_rabbit.strategy.reversal_signal import detect_reversal
    from quant_rabbit.strategy.tp_rebalancer import (
        _extract_atr_pips,
        _chart_context_from_chart,
    )

    actions: list[PartialCloseAction] = []
    for position in positions:
        owner = getattr(position, "owner", None)
        owner_str = owner.value if hasattr(owner, "value") else str(owner or "")
        if owner_str.lower() != "trader":
            continue
        pair = getattr(position, "pair", None)
        if not pair or pair not in quotes:
            continue
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
        atr_pips = _extract_atr_pips(chart, pair)
        if atr_pips is None or atr_pips <= 0:
            continue
        try:
            reversal = detect_reversal(chart, side_up)
        except Exception:
            reversal = None
        action = compute_partial_close(
            trade_id=str(getattr(position, "trade_id", "")),
            pair=pair,
            side=side_up,
            units=int(getattr(position, "units", 0)),
            entry_price=float(getattr(position, "entry_price", 0.0)),
            current_price=current_price,
            atr_pips=atr_pips,
            is_reversal_firing=(reversal is not None),
            owner=owner_str.lower(),
            chart_context=_chart_context_from_chart(chart),
        )
        if action is not None:
            actions.append(action)
    return actions


def apply_partial_closes(
    actions: Iterable[PartialCloseAction],
    broker_client: Any,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """Send partial-close requests via broker_client.close_trade(trade_id, units=<n>).

    Exceptions are captured per-action so one failure doesn't block
    the others.
    """
    results: list[dict] = []
    for a in actions:
        entry = {
            "trade_id": a.trade_id,
            "pair": a.pair,
            "side": a.side,
            "original_units": a.original_units,
            "close_units": a.close_units,
            "remaining_units": a.remaining_units,
            "adverse_pips": a.adverse_pips,
            "rationale": a.rationale,
            "sent": False,
            "error": None,
            "response": None,
        }
        if dry_run:
            results.append(entry)
            continue
        try:
            response = broker_client.close_trade(a.trade_id, units=str(a.close_units))
            entry["sent"] = True
            entry["response"] = response
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        results.append(entry)
    return results
