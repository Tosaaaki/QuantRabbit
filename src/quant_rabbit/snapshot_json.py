from __future__ import annotations

from typing import Any


# Persist only the broker order fields needed to recognize a carried GTC
# thesis and its dependent exit geometry. Full OANDA order raw may include
# account-level identifiers that are not needed in reusable market artifacts.
ORDER_RAW_SNAPSHOT_KEYS = (
    "createTime",
    "time",
    "clientExtensions",
    "tradeClientExtensions",
    "takeProfitOnFill",
    "stopLossOnFill",
    "timeInForce",
    "gtdTime",
    "triggerCondition",
    "triggerMode",
    "type",
)


def snapshot_order_raw(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {key: raw[key] for key in ORDER_RAW_SNAPSHOT_KEYS if key in raw}


def snapshot_payload_order_raw(item: object) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return snapshot_order_raw(item.get("raw"))
