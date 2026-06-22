from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TP_PROGRESS_REPAIR_REPLAY_CONTRACT = "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1"
TP_PROGRESS_REPAIR_REPLAY_FIELD = "profit_capture_repair_replay_contract"


def repair_replay_contract_from_payload(payload: Mapping[str, Any]) -> str | None:
    precision = payload.get("precision")
    if not isinstance(precision, Mapping):
        return None
    value = precision.get(TP_PROGRESS_REPAIR_REPLAY_FIELD)
    if value is None:
        return None
    return str(value)


def has_current_repair_replay_contract(payload: Mapping[str, Any]) -> bool:
    return repair_replay_contract_from_payload(payload) == TP_PROGRESS_REPAIR_REPLAY_CONTRACT
