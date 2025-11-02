"""
analysis.regime_classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~
最新テクニカル因子から
  Trend / Range / Breakout / Event / Mixed
の 5 つのレジームを返すシンプルな判定器。
"""

from __future__ import annotations
import os
from typing import Dict, Literal

TimeFrame = Literal["M1", "H4"]

THRESH_ADX_TREND = {"M1": 25.0, "H4": 22.0}
THRESH_MA_SLOPE = {"M1": 0.0003, "H4": 0.001}
THRESH_BBW_RANGE = {"M1": 0.25, "H4": 0.35}
SCALP_TREND_VELOCITY = float(os.getenv("SCALP_REGIME_TREND_VELOCITY", "1.4"))
SCALP_BREAKOUT_RANGE = float(os.getenv("SCALP_REGIME_BREAKOUT_RANGE", "2.2"))
SCALP_RANGE_MAX_VELOCITY = float(os.getenv("SCALP_REGIME_RANGE_MAX_VELOCITY", "0.6"))
SCALP_RANGE_MAX_RANGE = float(os.getenv("SCALP_REGIME_RANGE_MAX_RANGE", "1.0"))
SCALP_ATR_SCALE = float(os.getenv("SCALP_REGIME_ATR_SCALE", "100.0"))
SCALP_TREND_MOMENTUM = float(os.getenv("SCALP_REGIME_TREND_MOMENTUM", "0.8"))
SCALP_BREAKOUT_MOMENTUM = float(os.getenv("SCALP_REGIME_BREAKOUT_MOMENTUM", "1.0"))


def classify(
    factors: Dict[str, float], tf: TimeFrame, *, event_mode: bool = False
) -> str:
    """
    Parameters
    ----------
    factors : 指標 dict (calc_core.compute の戻り値)
    tf: "M1" or "H4"
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


def classify_scalp(factors: Dict[str, float]) -> str:
    """
    Lightweight regime classifier for the scalp pocket using tick metrics.

    Returns: "Trend" | "Range" | "Breakout" | "Mixed"
    """

    velocity = abs(float(factors.get("tick_velocity_30s") or 0.0))
    tick_range = float(factors.get("tick_range_30s") or 0.0)
    mom5 = abs(float(factors.get("tick_momentum_5") or 0.0))
    mom10 = abs(float(factors.get("tick_momentum_10") or 0.0))
    atr = float(factors.get("atr") or 0.0) * SCALP_ATR_SCALE

    if velocity >= SCALP_TREND_VELOCITY and (mom5 >= SCALP_TREND_MOMENTUM or mom10 >= SCALP_TREND_MOMENTUM):
        return "Trend"

    if tick_range >= SCALP_BREAKOUT_RANGE and velocity >= (SCALP_TREND_VELOCITY * 0.8):
        if mom5 >= SCALP_BREAKOUT_MOMENTUM or atr >= 6.0:
            return "Breakout"

    if velocity <= SCALP_RANGE_MAX_VELOCITY and tick_range <= SCALP_RANGE_MAX_RANGE and atr <= 4.0:
        return "Range"

    return "Mixed"
