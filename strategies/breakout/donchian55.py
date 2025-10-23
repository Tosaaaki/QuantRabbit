import pandas as pd
from typing import Dict
from analysis.ma_projection import compute_donchian_projection


class Donchian55:
    name = "Donchian55"
    pocket = "macro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        candles = fac.get("candles")
        if candles is None or len(candles) < 56:
            return None

        df = pd.DataFrame(candles)[-56:]
        high55 = df["high"][:-1].max()
        low55 = df["low"][:-1].min()
        close = df["close"].iloc[-1]
        if any(val is None for val in (high55, low55, close)):
            return None
        range_span = max(1e-6, high55 - low55)
        breakout_strength = abs(close - (high55 + low55) / 2) / range_span
        # ブレイクへの近さ（pips）で発火加減を調整
        proj = compute_donchian_projection(candles, lookback=55)
        near_pips = proj.nearest_pips if proj else None
        distance_bonus = 0.0
        if near_pips is not None:
            distance_bonus = max(0.0, min(10.0, (8.0 - min(8.0, near_pips)) * 1.2))
        confidence = int(max(45.0, min(95.0, 54.0 + breakout_strength * 42.0 + distance_bonus)))

        if close > high55:
            return {
                "action": "OPEN_LONG",
                "sl_pips": 55,
                "tp_pips": 110,
                "confidence": confidence,
                "tag": f"{Donchian55.name}-breakout-up",
            }
        if close < low55:
            return {
                "action": "OPEN_SHORT",
                "sl_pips": 55,
                "tp_pips": 110,
                "confidence": confidence,
                "tag": f"{Donchian55.name}-breakout-down",
            }
        return None
