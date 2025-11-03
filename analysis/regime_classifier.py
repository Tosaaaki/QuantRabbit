"""
analysis.regime_classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~
最新テクニカル因子から
  Trend / Range / Breakout / Event / Mixed
の 5 つのレジームを返すシンプルな判定器。
"""

from __future__ import annotations
from typing import Dict, Literal

TimeFrame = Literal["M1", "H1", "H4"]

THRESH_ADX_TREND = {"M1": 25.0, "H1": 24.0, "H4": 22.0}
THRESH_MA_SLOPE = {"M1": 0.0003, "H1": 0.00055, "H4": 0.001}
THRESH_BBW_RANGE = {"M1": 0.25, "H1": 0.30, "H4": 0.35}


def classify(
    factors: Dict[str, float], tf: TimeFrame, *, event_mode: bool = False
) -> str:
    """
    Parameters
    ----------
    factors : 指標 dict (calc_core.compute の戻り値)
    tf: "M1", "H1" or "H4"
    event_mode : 経済指標 30 分前後なら True

    Returns
    -------
    str : "Trend" | "Range" | "Breakout" | "Event" | "Mixed"
    """

    if event_mode:
        return "Event"

    adx = factors.get("adx", 0.0)
    bbw = factors.get("bbw", 1.0)  # ボリン幅 / middle
    ma10 = factors.get("ma10", 0.0)
    ma20 = factors.get("ma20", 0.0)

    ma_slope = abs(ma20 - ma10) / ma10 if ma10 else 0.0

    adx_th = THRESH_ADX_TREND[tf]
    slope_th = THRESH_MA_SLOPE[tf]
    bbw_th = THRESH_BBW_RANGE[tf]

    # --- Trend 判定 ---
    if adx >= adx_th and ma_slope >= slope_th:
        return "Trend"

    # --- Range 判定 ---
    if bbw <= bbw_th and adx < (adx_th - 5):
        return "Range"

    # --- Breakout 判定 ---
    # ADX が立ち上がり途中 & BB幅が膨らんでいる状態
    if adx >= (adx_th - 5) and bbw > bbw_th and ma_slope >= slope_th / 2:
        return "Breakout"

    return "Mixed"
