"""
signals.pocket_allocator
~~~~~~~~~~~~~~~~~~~~~~~~

Lot の総量と macro 重みから、pocket ごとの配分を返す。
"""

from __future__ import annotations

from typing import Dict


def alloc(total_lot: float, weight_macro: float | None) -> Dict[str, float]:
    """総ロットと macro 重みから配分を決める。

    - weight_macro: [0,1]。None の場合は 0.5 とみなす
    - 戻り値: {"macro": lot, "micro": lot}
    """
    if weight_macro is None:
        w = 0.5
    else:
        try:
            w = max(0.0, min(1.0, float(weight_macro)))
        except Exception:
            w = 0.5

    macro = round(total_lot * w, 3)
    micro = round(max(total_lot - macro, 0.0), 3)
    return {"macro": macro, "micro": micro}

