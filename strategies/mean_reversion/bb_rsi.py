from typing import Dict


class BBRsi:
    name = "BB_RSI"
    pocket = "micro"

    @staticmethod
    def check(fac: Dict) -> Dict | None:
        rsi = fac.get("rsi")
        bbw = fac.get("bbw")
        ma = fac.get("ma20")
        if not all([rsi, bbw, ma]):
            return None

        price = fac.get("close", ma)
        upper = ma + (ma * bbw / 2)
        lower = ma - (ma * bbw / 2)

        if price < lower and rsi < 30:
            # 逆張りは短時間決着＆安全側に建値繰上げを早める
            return {"action": "buy", "sl_pips": 10, "tp_pips": 15, "ttl_sec": 1800, "be_rr": 0.8, "trail_at_rr": 1.2, "trail_atr_mult": 1.0}
        if price > upper and rsi > 70:
            return {"action": "sell", "sl_pips": 10, "tp_pips": 15, "ttl_sec": 1800, "be_rr": 0.8, "trail_at_rr": 1.2, "trail_atr_mult": 1.0}
        return None
