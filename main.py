import asyncio, datetime
from market_data.candle_fetcher import start_candle_stream, Candle
from indicators.factor_cache import on_candle, all_factors, TimeFrame
from analysis.regime_classifier import classify
from analysis.focus_decider import decide_focus
from analysis.gpt_decider import get_decision
from analysis.perf_monitor import snapshot as get_perf
from analysis.summary_ingestor import get_latest_news, check_event_soon
from signals.pocket_allocator import alloc
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp, check_global_drawdown
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal

STRATEGIES = {
    "TrendMA": MovingAverageCross,
    "Donchian55": Donchian55,
    "BB_RSI": BBRsi,
    "NewsSpikeReversal": NewsSpikeReversal,
}

EQUITY = 10000.0   # ← 実際は REST から取得

async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)

async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)

async def logic_loop():
    pos_manager = PositionManager()
    perf_cache = {}
    news_cache = {}
    last_update_time = datetime.datetime.min
    last_heartbeat_time = datetime.datetime.min # Add this line

    try:
        while True:
            now = datetime.datetime.utcnow()

            # Heartbeat logging
            if (now - last_heartbeat_time).total_seconds() >= 300: # Every 5 minutes
                print(f"[HEARTBEAT] System is alive at {now.isoformat(timespec='seconds')}")
                last_heartbeat_time = now

            # --- 1. 状況分析 ---
            factors = all_factors()
            fac_m1 = factors.get("M1")
            fac_h4 = factors.get("H4")

            # 両方のタイムフレームのデータが揃うまで待機
            if not fac_m1 or not fac_h4 or not fac_m1.get("close") or not fac_h4.get("close"):
                print("[WAIT] Waiting for M1/H4 factor data...")
                await asyncio.sleep(5)
                continue

            # 5分ごとにパフォーマンスとニュースを更新
            if (now - last_update_time).total_seconds() >= 300:
                perf_cache = get_perf()
                news_cache = get_latest_news()
                last_update_time = now
                print(f"[PERF] Updated: {perf_cache}")
                print(f"[NEWS] Updated: {news_cache}")

            event_soon = check_event_soon(within_minutes=30, min_impact=3)
            global_drawdown_exceeded = check_global_drawdown()

            if global_drawdown_exceeded:
                print("[STOP] Global drawdown limit exceeded. Stopping new trades.")
                await asyncio.sleep(60)
                continue

            macro_regime = classify(fac_h4, "H4", event_mode=event_soon)
            micro_regime = classify(fac_m1, "M1", event_mode=event_soon)
            focus, w_macro = decide_focus(macro_regime, micro_regime, perf=perf_cache)

            # --- 2. GPT判断 ---
            payload = {
                "ts": now.isoformat(timespec="seconds"),
                "reg_macro": macro_regime,
                "reg_micro": micro_regime,
                "factors_m1": {k: v for k, v in fac_m1.items() if k != 'candles'},
                "factors_h4": {k: v for k, v in fac_h4.items() if k != 'candles'},
                "perf": perf_cache,
                "news_short": news_cache.get("short", []),
                "news_long": news_cache.get("long", []),
                "event_soon": event_soon
            }
            gpt = await get_decision(payload)
            weight = gpt.get("weight_macro", w_macro)

            # --- 3. 発注準備 ---
            lot_total = allowed_lot(EQUITY, sl_pips=20) # sl_pipsは仮
            lots = alloc(lot_total, weight)

            # --- 4. 戦略実行ループ ---
            for sname in gpt.get("ranked_strategies", []):
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                
                # 全ての戦略はM1の指標で判断
                # NewsSpikeReversal は news_short も必要
                if sname == "NewsSpikeReversal":
                    sig = cls.check(fac_m1, news_cache.get("short", []))
                else:
                    sig = cls.check(fac_m1)

                if not sig:
                    continue
                
                pocket = cls.pocket
                if event_soon and pocket == "micro":
                    print(f"[SKIP] Event soon, skipping micro pocket trade.")
                    continue

                if not can_trade(pocket):
                    print(f"[SKIP] DD limit for {pocket} pocket.")
                    continue
                
                lot = lots.get(pocket, 0)
                if lot <= 0:
                    continue
                
                units = int(lot * 100000) * (1 if sig["action"]=="buy" else -1)
                price = fac_m1.get("close")
                sl, tp = clamp_sl_tp(price,
                                     price - sig["sl_pips"]/100,
                                     price + sig["tp_pips"]/100,
                                     sig["action"]=="buy")
                
                trade_id = market_order("USD_JPY", units, sl, tp, pocket)
                if trade_id:
                    print(f"[ORDER] {trade_id} | {cls.name} | {sig['action']} | {lot} lot | SL={sl}, TP={tp}")
                else:
                    print(f"[ORDER FAILED] {cls.name}")
                
                break # 1取引/ループ

            # --- 5. 決済済み取引の同期 ---
            pos_manager.sync_trades()

            await asyncio.sleep(60)
    except Exception as e:
        print(f"[ERROR] An unhandled exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pos_manager.close()
        print("PositionManager closed.")

async def main():
    handlers = [
        ("M1", m1_candle_handler),
        ("H4", h4_candle_handler)
    ]
    await asyncio.gather(
        start_candle_stream("USD_JPY", handlers),
        logic_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())