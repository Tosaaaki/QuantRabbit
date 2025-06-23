from indicators.factor_cache import get

class BBRsi:
    name = "BB_RSI"
    pocket = "micro"

    @staticmethod
    def check():
        rsi = get("rsi")
        bbw = get("bbw")
        ma  = get("ma20")
        if not all([rsi, bbw, ma]):
            return None

        price = get("close", ma)
        upper = ma * (1 + bbw/2)
        lower = ma * (1 - bbw/2)

        if price < lower and rsi < 30:
            return {"action":"buy","sl_pips":10,"tp_pips":15}
        if price > upper and rsi > 70:
            return {"action":"sell","sl_pips":10,"tp_pips":15}
        return None