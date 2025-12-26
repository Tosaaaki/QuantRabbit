"""Shared helpers for TP比スケールと仮想SLフロア適用."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class TPScaleConfig:
    tp_floor_ratio: float = 1.0
    trail_from_tp_ratio: float = 0.82
    lock_from_tp_ratio: float = 0.45
    virtual_sl_ratio: float = 0.72
    min_profit_take: float = 1.0


def apply_tp_virtual_floor(
    profit_take: float,
    trail_start: float,
    lock_buffer: float,
    stop_loss: float,
    state: Any,
    cfg: TPScaleConfig | None = None,
) -> Tuple[float, float, float, float]:
    """
    Ensure TPを基準にしたトレール/ロック/仮想SLの下限を揃える。
    - tp_hint があれば profit_take を底上げし、trail/lock を比率で拡張
    - 仮想SL は TP 比率（virtual_sl_ratio）でフロア設定
    """
    if cfg is None:
        cfg = TPScaleConfig()

    pt = max(float(profit_take), cfg.min_profit_take)
    ts = float(trail_start)
    lb = float(lock_buffer)
    sl = float(stop_loss)

    tp_hint_val = None
    try:
        tp_hint_val = float(getattr(state, "tp_hint", None))
    except Exception:
        tp_hint_val = None

    if tp_hint_val is not None:
        pt = max(pt, max(cfg.min_profit_take, tp_hint_val * cfg.tp_floor_ratio))
        ts = max(ts, max(cfg.min_profit_take, pt * cfg.trail_from_tp_ratio))
        lb = max(lb, pt * cfg.lock_from_tp_ratio)

    sl = max(sl, pt * cfg.virtual_sl_ratio)
    ts = max(ts, pt * cfg.trail_from_tp_ratio)
    lb = max(lb, pt * cfg.lock_from_tp_ratio)

    return pt, ts, lb, sl
