import asyncio, datetime
from market_data.candle_fetcher import start_candle_stream
from indicators.factor_cache import on_candle, all_factors
from analysis.regime_classifier import classify
from analysis.focus_decider import decide_focus
from analysis.gpt_decider import get_decision
from signals.pocket_allocator import alloc
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp
from execution.order_manager import market_order
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi

STRATEGIES = {
    "TrendMA": MovingAverageCross,
    "Donchian55": Donchian55,
    "BB_RSI": BBRsi,
}

EQUITY = 10000.0   # ← 実際は REST から取得

async def candle_handler(key, cndl):
    await on_candle(key, cndl)

async def logic_loop():
    while True:
        fac = all_factors()
        macro = classify(fac)
        micro = classify(fac)
        focus, w_macro = decide_focus(macro, micro)

        payload = {
            "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "reg_macro": macro,
            "reg_micro": micro,
            "factors": fac,
        }
        gpt = await get_decision(payload)
        weight = gpt["weight_macro"]

        lot_total = allowed_lot(EQUITY, sl_pips=20)
        lots = alloc(lot_total, weight)

        for sname in gpt["ranked_strategies"]:
            cls = STRATEGIES.get(sname)
            if not cls:
                continue
            sig = cls.check()
            if not sig:
                continue
            pocket = cls.pocket
            if not can_trade(pocket):
                continue
            lot = lots[pocket]
            if lot <= 0:
                continue
            units = int(lot * 100000) * (1 if sig["action"]=="buy" else -1)
            price = fac.get("close", fac.get("ma20", 0))
            sl, tp = clamp_sl_tp(price,
                                 price - sig["sl_pips"]/100,
                                 price + sig["tp_pips"]/100,
                                 sig["action"]=="buy")
            ticket = market_order("USD_JPY", units, sl, tp, pocket)
            print("ORDER", ticket, cls.name, sig)
            break   # 1 取引/ループ
        await asyncio.sleep(60)

async def main():
    await asyncio.gather(
        start_candle_stream("USD_JPY", candle_handler),
        logic_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())