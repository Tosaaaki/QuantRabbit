import pandas as pd
from typing import Dict

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
        low55  = df["low"][:-1].min()
        close  = df["close"].iloc[-1]

        if close > high55:
            return {"action":"buy","sl_pips":55,"tp_pips":110}
        if close < low55:
            return {"action":"sell","sl_pips":55,"tp_pips":110}
        return None