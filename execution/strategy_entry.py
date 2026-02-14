"""Strategy-facing entry helpers.

This module is intentionally passthrough-only: strategy code calls into this
module and the intent is forwarded directly to execution.order_manager.
"""

from __future__ import annotations

from typing import Literal, Optional

from execution.order_manager import cancel_order, close_trade, set_trade_protections
from execution import order_manager


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
