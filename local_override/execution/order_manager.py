from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def market_order(
    instrument: str,
    units: int,
    sl_price: Optional[float],
    tp_price: Optional[float],
    pocket: str,
    *,
    client_order_id: Optional[str] = None,
    reduce_only: bool = False,
    stage_index: Optional[int] = None,
) -> Tuple[Optional[str], Optional[float]]:
    # offline stub: pretend order is filled at None (use close price upstream)
    trade_id = f"sim-{_now_iso()}"
    return trade_id, None


async def close_trade(trade_id: str, units: int) -> bool:
    return True


async def update_dynamic_protections(trade_id: str, sl_price=None, tp_price=None):
    return None


async def set_trade_protections(trade_id: str, sl_price=None, tp_price=None):
    return None


def plan_partial_reductions(
    open_positions: dict,
    fac_m1: dict,
    *,
    range_mode: bool = False,
    stage_state: Optional[dict] = None,
    pocket_profiles: Optional[dict] = None,
    now: Optional[datetime] = None,
    threshold_overrides: Optional[dict] = None,
) -> List[tuple[str, str, int]]:
    return []
