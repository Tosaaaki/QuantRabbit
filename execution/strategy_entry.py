"""Strategy-facing entry helpers.

Strategies submit orders through this module. Before dispatching, it coordinates
entry intent via order_manager's blackboard to keep strategy-level intent and avoid
duplicate cross-strategy overexposure.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from execution.order_manager import cancel_order, close_trade, set_trade_protections
from execution import order_manager


def get_last_order_status_by_client_id(
    client_order_id: Optional[str],
) -> Optional[dict[str, object]]:
    """Compatibility wrapper retained for existing strategy imports."""
    return order_manager.get_last_order_status_by_client_id(client_order_id)


def _resolve_strategy_tag(
    strategy_tag: Optional[str],
    client_order_id: Optional[str],
    entry_thesis: Optional[dict],
) -> Optional[str]:
    resolved = strategy_tag
    if not resolved:
        resolved = order_manager._strategy_tag_from_client_id(client_order_id)
    if not resolved and isinstance(entry_thesis, dict):
        resolved = order_manager._strategy_tag_from_thesis(entry_thesis)
    return resolved


def _resolve_entry_probability(
    entry_thesis: Optional[dict],
    confidence: Optional[float],
) -> Optional[float]:
    if not isinstance(entry_thesis, dict):
        return _entry_probability_value(confidence) if confidence is not None else None
    for key in ("entry_probability", "confidence"):
        if key not in entry_thesis:
            continue
        raw = entry_thesis.get(key)
        try:
            probability = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(probability) or math.isinf(probability):
            continue
        if probability <= 1.0:
            return max(0.0, min(1.0, probability))
        return max(0.0, min(1.0, probability / 100.0))
    return (
        _entry_probability_value(confidence)
        if confidence is not None
        else None
    )


def _entry_probability_value(raw: Optional[float]) -> Optional[float]:
    if raw is None:
        return None
    try:
        probability = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(probability) or math.isinf(probability):
        return None
    if probability <= 1.0:
        return max(0.0, min(1.0, probability))
    return max(0.0, min(1.0, probability / 100.0))


async def _coordinate_entry_units(
    *,
    instrument: str,
    pocket: str,
    strategy_tag: Optional[str],
    units: int,
    reduce_only: bool,
    entry_probability: Optional[float],
    client_order_id: Optional[str],
) -> int:
    if not units:
        return units
    if reduce_only:
        return units
    if not strategy_tag:
        return units
    min_units = order_manager.min_units_for_pocket(pocket)
    final_units, reason, _ = await order_manager.coordinate_entry_intent(
        instrument=instrument,
        pocket=pocket,
        strategy_tag=strategy_tag,
        side=1 if units > 0 else -1,
        raw_units=units,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
        min_units=min_units,
    )
    if not final_units and reason in {"reject", "scaled", "rejected", None}:
        return 0
    return int(final_units)


async def market_order(
    instrument: str,
    units: int,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp", "scalp_fast", "manual"],
    *,
    client_order_id: Optional[str] = None,
    strategy_tag: Optional[str] = None,
    reduce_only: bool = False,
    entry_thesis: Optional[dict] = None,
    meta: Optional[dict] = None,
    confidence: Optional[int] = None,
    stage_index: Optional[int] = None,
    arbiter_final: bool = False,
) -> Optional[str]:
    resolved_strategy_tag = _resolve_strategy_tag(strategy_tag, client_order_id, entry_thesis)
    entry_probability = _resolve_entry_probability(entry_thesis, confidence)
    coordinated_units = await _coordinate_entry_units(
        instrument=instrument,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        units=units,
        reduce_only=reduce_only,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
    )
    if not coordinated_units:
        return None
    if coordinated_units != units:
        units = coordinated_units
    return await order_manager.market_order(
        instrument=instrument,
        units=units,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=pocket,
        client_order_id=client_order_id,
        strategy_tag=resolved_strategy_tag,
        reduce_only=reduce_only,
        entry_thesis=entry_thesis,
        meta=meta,
        confidence=confidence,
        stage_index=stage_index,
        arbiter_final=arbiter_final,
    )


async def limit_order(
    instrument: str,
    units: int,
    price: float,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: Literal["micro", "macro", "scalp"],
    *,
    current_bid: Optional[float] = None,
    current_ask: Optional[float] = None,
    require_passive: bool = True,
    client_order_id: Optional[str] = None,
    strategy_tag: Optional[str] = None,
    reduce_only: bool = False,
    ttl_ms: float = 800.0,
    entry_thesis: Optional[dict] = None,
    confidence: Optional[int] = None,
    meta: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    resolved_strategy_tag = _resolve_strategy_tag(strategy_tag, client_order_id, entry_thesis)
    entry_probability = _resolve_entry_probability(entry_thesis, confidence)
    coordinated_units = await _coordinate_entry_units(
        instrument=instrument,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        units=units,
        reduce_only=reduce_only,
        entry_probability=entry_probability,
        client_order_id=client_order_id,
    )
    if not coordinated_units:
        return None, None
    if coordinated_units != units:
        units = coordinated_units
    return await order_manager.limit_order(
        instrument=instrument,
        units=units,
        price=price,
        sl_price=sl_price,
        tp_price=tp_price,
        pocket=pocket,
        current_bid=current_bid,
        current_ask=current_ask,
        require_passive=require_passive,
        client_order_id=client_order_id,
        reduce_only=reduce_only,
        ttl_ms=ttl_ms,
        entry_thesis=entry_thesis,
        confidence=confidence,
        meta=meta,
    )
