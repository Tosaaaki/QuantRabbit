from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TP_PROGRESS_REPAIR_REPLAY_CONTRACT = "TP_PROGRESS_PRODUCTION_GATE_REPLAY_V1"
TP_PROGRESS_REPAIR_REPLAY_FIELD = "profit_capture_repair_replay_contract"

# Contracted production replay window for TP-progress profit-capture repair.
# 744h is the month-scale coverage used by cycle-refresh; the 80 event cap
# limits report rows only and must not shorten the replay window.
MONTH_SCALE_REPAIR_REPLAY_LOOKBACK_HOURS = 744
MONTH_SCALE_REPAIR_REPLAY_POST_CLOSE_HOURS = 6
MONTH_SCALE_REPAIR_REPLAY_MAX_EVENTS = 80
MONTH_SCALE_EXECUTION_TIMING_AUDIT_COMMAND = (
    "PYTHONPATH=src python3 -m quant_rabbit.cli execution-timing-audit "
    f"--lookback-hours {MONTH_SCALE_REPAIR_REPLAY_LOOKBACK_HOURS} "
    f"--post-close-hours {MONTH_SCALE_REPAIR_REPLAY_POST_CLOSE_HOURS} "
    f"--max-events {MONTH_SCALE_REPAIR_REPLAY_MAX_EVENTS}"
)


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
