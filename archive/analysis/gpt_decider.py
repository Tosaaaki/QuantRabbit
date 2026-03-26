"""
analysis.gpt_decider
~~~~~~~~~~~~~~~~~~~~
Local-only decision wrapper (LLM removed).
"""

from __future__ import annotations

from typing import Dict, Optional

from analysis.local_decider import heuristic_decision

_LAST_DECISION: Optional[Dict[str, object]] = None


async def get_decision(payload: Dict) -> Dict:
    """Return a local heuristic decision without any LLM calls."""
    global _LAST_DECISION
    decision = heuristic_decision(payload, _LAST_DECISION)
    if isinstance(decision, dict):
        _LAST_DECISION = decision
    return decision


def get_decision_sync(payload: Dict) -> Dict:
    """Synchronous helper for local-only decision."""
    global _LAST_DECISION
    decision = heuristic_decision(payload, _LAST_DECISION)
    if isinstance(decision, dict):
        _LAST_DECISION = decision
    return decision


__all__ = ["get_decision", "get_decision_sync"]
