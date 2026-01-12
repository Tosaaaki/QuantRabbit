from __future__ import annotations

from execution.order_manager import close_trade as _close_trade
from workers.common.exit_emergency import should_allow_negative_close


async def close_trade(
    trade_id: str,
    units: int | None = None,
    client_order_id: str | None = None,
    allow_negative: bool = False,
) -> bool:
    if not allow_negative:
        allow_negative = should_allow_negative_close()
    return await _close_trade(
        trade_id,
        units,
        client_order_id=client_order_id,
        allow_negative=allow_negative,
    )
