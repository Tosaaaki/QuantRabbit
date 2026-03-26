from __future__ import annotations

from typing import Any, Dict, Optional

from analysis.local_decider import heuristic_decision

_LAST_DECISION: Optional[Dict[str, Any]] = None


async def get_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _LAST_DECISION
    decision = heuristic_decision(payload, _LAST_DECISION)
    if isinstance(decision, dict):
        _LAST_DECISION = decision
    return decision


def fallback_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
    return heuristic_decision(payload, _LAST_DECISION)
