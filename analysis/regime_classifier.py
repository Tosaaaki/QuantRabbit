"""
analysis.regime_classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~
最新テクニカル因子から
  Trend / Range / Breakout / Event / Mixed
の 5 つのレジームを返すシンプルな判定器。
"""

from __future__ import annotations
from typing import Dict


THRESH_ADX_TREND = 25.0
THRESH_MA_SLOPE  = 0.0003       # 0.03%
THRESH_BBW_RANGE = 0.25


def classify(factors: Dict[str, float], *, event_mode: bool = False) -> str:
    """
    Parameters
    ----------
    factors : 指標 dict (calc_core.compute の戻り値)
    event_mode : 経済指標 30 分前後なら True

    Returns
    -------
    str : "Trend" | "Range" | "Breakout" | "Event" | "Mixed"
    """

    if event_mode:
        return "Event"

    adx   = factors.get("adx", 0.0)
    bbw   = factors.get("bbw", 1.0)   # ボリン幅 / middle
    ma10  = factors.get("ma10", 0.0)
    ma20  = factors.get("ma20", 0.0)

    ma_slope = abs(ma20 - ma10) / ma10 if ma10 else 0.0

    # --- Trend 判定 ---
    if adx >= THRESH_ADX_TREND and ma_slope >= THRESH_MA_SLOPE:
        return "Trend"

    # --- Range 判定 ---
    if bbw <= THRESH_BBW_RANGE and adx < (THRESH_ADX_TREND - 5):
        return "Range"

    # --- Breakout 判定 ---
    # ADX が立ち上がり途中 & BB幅が膨らんでいる状態
    if adx >= 20 and bbw > THRESH_BBW_RANGE and ma_slope >= THRESH_MA_SLOPE / 2:
        return "Breakout"

    return "Mixed"