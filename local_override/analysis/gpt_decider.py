from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple


_FALLBACK = {
    "focus_tag": "hybrid",
    "weight_macro": 0.5,
    "weight_scalp": 0.15,
    "ranked_strategies": [
        "TrendMA",
        "H1Momentum",
        "Donchian55",
        "BB_RSI",
    ],
    "reason": "fallback_override",
}


async def get_decision(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str]:
    """Return a deterministic fallback decision for offline replay.

    Returns: (decision_dict, tokens_used, model_used)
    """
    await asyncio.sleep(0)
    return dict(_FALLBACK), 0, "offline-fallback"


def fallback_decision(payload: Dict[str, Any]) -> Dict[str, Any]:
    return dict(_FALLBACK)
