import pandas as pd
from indicators.factor_cache import _CANDLES, get

class Donchian55:
    name = "Donchian55"
    pocket = "macro"

    @staticmethod
    def check():
        if len(_CANDLES) < 56:
            return None
        df = pd.DataFrame(_CANDLES)[-56:]
        high55 = df["high"][:-1].max()
        low55  = df["low"][:-1].min()
        close  = df["close"].iloc[-1]

        if close > high55:
            return {"action":"buy","sl_pips":55,"tp_pips":110}
        if close < low55:
            return {"action":"sell","sl_pips":55,"tp_pips":110}
        return None