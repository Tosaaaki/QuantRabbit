"""Strategy-facing entry helpers.

Strategies submit orders through this module. Before dispatching, it coordinates
entry intent via order_manager's blackboard to keep strategy-level intent and avoid
duplicate cross-strategy overexposure.
"""

from __future__ import annotations

import math
import os
from typing import Literal, Optional

from analysis.technique_engine import evaluate_entry_techniques
from execution.order_manager import cancel_order, close_trade, set_trade_protections
from execution import order_manager


def get_last_order_status_by_client_id(
    client_order_id: Optional[str],
) -> Optional[dict[str, object]]:
    """Compatibility wrapper retained for existing strategy imports."""
    return order_manager.get_last_order_status_by_client_id(client_order_id)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_ENTRY_TECH_CONTEXT_ENABLED = _env_bool("ENTRY_TECH_CONTEXT_ENABLED", True)


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


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_entry_side(units: int) -> str:
    return "long" if units >= 0 else "short"


def _resolve_entry_price(
    units: int,
    entry_thesis: Optional[dict],
    *,
    limit_price: Optional[float] = None,
) -> Optional[float]:
    if limit_price is not None:
        resolved = _to_float(limit_price)
        if resolved is not None:
            return resolved
    if isinstance(entry_thesis, dict):
        for key in ("entry_price", "price", "mid", "current_mid"):
            value = _to_float(entry_thesis.get(key))
            if value and value > 0:
                return value
        side = _resolve_entry_side(units)
        for key in ("current_ask", "ask", "current_bid", "bid"):
            if side == "long" and key in ("current_bid", "bid"):
                continue
            value = _to_float(entry_thesis.get(key))
            if value is not None and value > 0:
                return value
        for key in ("current_bid", "bid", "current_ask", "ask"):
            if side == "short" and key in ("current_ask", "ask"):
                continue
            value = _to_float(entry_thesis.get(key))
            if value is not None and value > 0:
                return value
    try:
        from market_data import tick_window
    except Exception:
        return None
    try:
        latest = tick_window.summarize(seconds=2.5)
        if not latest:
            return None
        side = _resolve_entry_side(units)
        if side == "long":
            value = _to_float(latest.get("latest_ask"))
            if value is not None and value > 0:
                return value
        value = _to_float(latest.get("latest_bid"))
        if value is not None and value > 0:
            return value
        return _to_float(latest.get("latest_mid"))
    except Exception:
        return None


def _inject_entry_technical_context(
    *,
    units: int,
    pocket: str,
    strategy_tag: Optional[str],
    entry_thesis: Optional[dict],
    entry_price: Optional[float],
) -> Optional[dict]:
    if not _ENTRY_TECH_CONTEXT_ENABLED:
        return entry_thesis
    if entry_price is None:
        return entry_thesis
    if not isinstance(entry_thesis, dict):
        return entry_thesis
    try:
        decision = evaluate_entry_techniques(
            entry_price=entry_price,
            side=_resolve_entry_side(units),
            pocket=pocket,
            strategy_tag=strategy_tag,
            entry_thesis=entry_thesis,
        )
        context = {
            "enabled": True,
            "entry_price": entry_price,
            "side": _resolve_entry_side(units),
            "result": {
                "allowed": bool(decision.allowed),
                "reason": decision.reason,
                "score": decision.score,
                "coverage": decision.coverage,
                "size_mult": decision.size_mult,
            },
            "debug": decision.debug,
        }
    except Exception as exc:
        context = {
            "enabled": True,
            "entry_price": entry_price,
            "error": str(exc),
        }
    prev = entry_thesis.get("technical_context")
    if isinstance(prev, dict):
        merged = dict(prev)
        merged.update(context)
        entry_thesis["technical_context"] = merged
    else:
        entry_thesis["technical_context"] = context
    return entry_thesis


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
    if entry_thesis is None:
        entry_thesis = {}
    entry_thesis = _inject_entry_technical_context(
        units=units,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        entry_thesis=entry_thesis,
        entry_price=_resolve_entry_price(units, entry_thesis),
    )
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
    if entry_thesis is None:
        entry_thesis = {}
    entry_thesis = _inject_entry_technical_context(
        units=units,
        pocket=pocket,
        strategy_tag=resolved_strategy_tag,
        entry_thesis=entry_thesis,
        entry_price=_resolve_entry_price(units, entry_thesis, limit_price=price),
    )
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
