"""
signals.pocket_allocator
~~~~~~~~~~~~~~~~~~~~~~~~
GPT が返す weight_macro と、risk_guard が計算した
口座許容 lot を入力し、各 Pocket の lot を算出。
"""

from __future__ import annotations
from typing import Dict


def alloc(total_lot: float, weight_macro: float) -> Dict[str, float]:
    """
    Parameters
    ----------
    total_lot   : 口座全体で許容される lot
    weight_macro: 0‑1

    Returns
    -------
    {"micro": 0.02, "macro": 0.04}
    """
    macro = round(total_lot * weight_macro, 3)
    micro = round(total_lot - macro, 3)
    return {"micro": micro, "macro": macro}
