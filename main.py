import asyncio
import datetime
import logging
import traceback

from market_data.candle_fetcher import (
    Candle,
    start_candle_stream,
    initialize_history,
)
from indicators.factor_cache import all_factors, on_candle
from analysis.regime_classifier import classify
from analysis.focus_decider import decide_focus
from analysis.gpt_decider import get_decision
from analysis.perf_monitor import snapshot as get_perf
from analysis.summary_ingestor import check_event_soon, get_latest_news
# バックグラウンドでニュース取得と要約を実行するためのインポート
from market_data.news_fetcher import fetch_loop as news_fetch_loop
from analysis.summary_ingestor import ingest_loop as summary_ingest_loop
from analysis.kaizen import audit_loop as kaizen_loop
from signals.pocket_allocator import alloc
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    check_global_drawdown,
)
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.exit_manager import exit_loop
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from analysis.learning import re_rank_strategies, risk_multiplier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Application started!")

STRATEGIES = {
    "TrendMA": MovingAverageCross,
    "Donchian55": Donchian55,
    "BB_RSI": BBRsi,
    "NewsSpikeReversal": NewsSpikeReversal,
}

EQUITY = 10000.0  # ← 実際は REST から取得


async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)


async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)


async def logic_loop():
    pos_manager = PositionManager()
    perf_cache = {}
    news_cache = {}
    last_update_time = datetime.datetime.min
    last_heartbeat_time = datetime.datetime.min  # Add this line

    try:
        while True:
            now = datetime.datetime.utcnow()

            # Heartbeat logging
            if (now - last_heartbeat_time).total_seconds() >= 300:  # Every 5 minutes
                logging.info(
                    f"[HEARTBEAT] System is alive at {now.isoformat(timespec='seconds')}"
                )
                last_heartbeat_time = now

            # 5分ごとにパフォーマンスとニュースを更新
            if (now - last_update_time).total_seconds() >= 300:
                perf_cache = get_perf()
                news_cache = get_latest_news()
                last_update_time = now
                logging.info(f"[PERF] Updated: {perf_cache}")
                logging.info(f"[NEWS] Updated: {news_cache}")

            # --- 1. 状況分析 ---
            factors = all_factors()
            fac_m1 = factors.get("M1")
            fac_h4 = factors.get("H4")

            # 両方のタイムフレームのデータが揃うまで待機
            if (
                not fac_m1
                or not fac_h4
                or not fac_m1.get("close")
                or not fac_h4.get("close")
            ):
                logging.info("[WAIT] Waiting for M1/H4 factor data for trading logic...")
                await asyncio.sleep(5)
                continue

            event_soon = check_event_soon(within_minutes=30, min_impact=3)
            global_drawdown_exceeded = check_global_drawdown()

            if global_drawdown_exceeded:
                logging.warning(
                    "[STOP] Global drawdown limit exceeded. Stopping new trades."
                )
                await asyncio.sleep(60)
                continue

            macro_regime = classify(fac_h4, "H4", event_mode=event_soon)
            micro_regime = classify(fac_m1, "M1", event_mode=event_soon)
            # PF は {pocket: {pf:..}} の形。存在しなければ None
            macro_pf = (perf_cache.get("macro") or {}).get("pf") if isinstance(perf_cache, dict) else None
            micro_pf = (perf_cache.get("micro") or {}).get("pf") if isinstance(perf_cache, dict) else None
            focus, w_macro = decide_focus(
                macro_regime,
                micro_regime,
                event_soon=event_soon,
                macro_pf=macro_pf,
                micro_pf=micro_pf,
            )

            # --- 2. GPT判断 ---
            # M1/H4 の移動平均・RSI などの指標をまとめて送信
            payload = {
                "ts": now.isoformat(timespec="seconds"),
                "reg_macro": macro_regime,
                "reg_micro": micro_regime,
                "factors_m1": {k: v for k, v in fac_m1.items() if k != "candles"},
                "factors_h4": {k: v for k, v in fac_h4.items() if k != "candles"},
                "perf": perf_cache,
                "news_short": news_cache.get("short", []),
                "news_long": news_cache.get("long", []),
                "event_soon": event_soon,
            }
            gpt = await get_decision(payload)
            weight = gpt.get("weight_macro", w_macro)
            strategy_directives = gpt.get("strategy_directives", {}) or {}

            def _is_enabled(name: str) -> bool:
                cfg = strategy_directives.get(name) or {}
                return bool(cfg.get("enabled", True))

            def _risk_bias(name: str) -> float:
                cfg = strategy_directives.get(name) or {}
                try:
                    return float(cfg.get("risk_bias", 1.0))
                except (TypeError, ValueError):
                    return 1.0

            # GPT 候補から未実装戦略を除外し、学習スコアで再ランク
            gpt_list = [
                s
                for s in gpt.get("ranked_strategies", [])
                if s in ("TrendMA", "Donchian55", "BB_RSI", "NewsSpikeReversal") and _is_enabled(s)
            ]
            ranked = re_rank_strategies(gpt_list, macro_regime, micro_regime) if gpt_list else []
            if not ranked:
                fallback_order = [s for s in ("TrendMA", "Donchian55", "BB_RSI", "NewsSpikeReversal") if _is_enabled(s)]
                ranked = fallback_order

            # --- 3. 発注準備 ---
            lot_total = allowed_lot(EQUITY, sl_pips=20)  # sl_pipsは仮
            lots = alloc(lot_total, weight)

            # --- 4. 戦略実行ループ ---
            # ヘルパー: ATRベース/ピップスベースの SL/TP を計算
            def _calc_sl_tp_from_signal(sig: dict, action: str, price: float, pocket: str) -> tuple[float, float]:
                # ピップス指定
                if "sl_pips" in sig and "tp_pips" in sig:
                    sl_pips = float(sig["sl_pips"])
                    tp_pips = float(sig["tp_pips"])
                    pip = 0.01  # USD/JPY
                    if action == "buy":
                        sl = price - sl_pips * pip
                        tp = price + tp_pips * pip
                    else:
                        sl = price + sl_pips * pip
                        tp = price - tp_pips * pip
                    return clamp_sl_tp(price, sl, tp, action == "buy")
                # ATR倍率指定（macro=H4優先、無ければM1）
                if "sl_atr_mult" in sig and "tp_atr_mult" in sig:
                    use_h4 = pocket == "macro"
                    atr_src = fac_h4 if use_h4 and fac_h4 else fac_m1
                    atr = float(atr_src.get("atr", 0.0) or 0.0)
                    sl_d = float(sig["sl_atr_mult"]) * atr
                    tp_d = float(sig["tp_atr_mult"]) * atr
                    if action == "buy":
                        sl = price - sl_d
                        tp = price + tp_d
                    else:
                        sl = price + sl_d
                        tp = price - tp_d
                    return clamp_sl_tp(price, sl, tp, action == "buy")
                # デフォルト: 20pips 相当
                pip = 0.01
                if action == "buy":
                    return clamp_sl_tp(price, price - 20 * pip, price + 20 * pip, True)
                else:
                    return clamp_sl_tp(price, price + 20 * pip, price - 20 * pip, False)

            for sname in ranked:
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                if not _is_enabled(sname):
                    logging.info(f"[SKIP] {sname} disabled by GPT directive")
                    continue

                # 戦略ごとに必要な入力を渡す
                if sname == "NewsSpikeReversal":
                    sig = cls.check(fac_m1, news_cache.get("short", []))
                elif sname in ("TrendMA", "Donchian55"):
                    sig = cls.check(fac_m1, fac_h4)
                else:  # BB_RSI
                    sig = cls.check(fac_m1)

                if not sig:
                    continue

                pocket = cls.pocket
                # Event モード中は micro を原則禁止。ただし NewsSpikeReversal は例外で許可。
                if event_soon and pocket == "micro" and sname != "NewsSpikeReversal":
                    logging.info("[SKIP] Event soon, skipping non-news micro trade.")
                    continue

                if not can_trade(pocket):
                    logging.info(f"[SKIP] DD limit for {pocket} pocket.")
                    continue

                lot = lots.get(pocket, 0)
                # 学習済みの戦略成績からリスク係数を適用（0.7x〜1.3x）
                try:
                    lot = round(lot * float(risk_multiplier(pocket, cls.name)), 3)
                except Exception:
                    pass
                directive_bias = _risk_bias(sname)
                if directive_bias != 1.0:
                    lot = round(lot * directive_bias, 3)
                if lot <= 0:
                    continue

                units = int(lot * 100000) * (1 if sig["action"] == "buy" else -1)
                price = float(fac_m1.get("close"))
                sl, tp = _calc_sl_tp_from_signal(sig, sig["action"], price, pocket)

                order_result = await market_order(
                    "USD_JPY",
                    units,
                    sl,
                    tp,
                    pocket,
                    strategy=cls.name,
                    macro_regime=macro_regime,
                    micro_regime=micro_regime,
                )
                if order_result.get("success"):
                    logging.info(
                        "[ORDER] %s | %s | %s lot | SL=%s TP=%s",
                        order_result.get("trade_id"),
                        cls.name,
                        lot,
                        sl,
                        tp,
                    )
                else:
                    logging.error(
                        "[ORDER FAILED] %s | error=%s",
                        cls.name,
                        order_result.get("error"),
                    )

                break  # 1取引/ループ

            # --- 5. 決済済み取引の同期 ---
            pos_manager.sync_trades()

            await asyncio.sleep(60)
    except Exception as e:
        logging.error(f"[ERROR] An unhandled exception occurred: {e}")
        logging.error(traceback.format_exc())
    finally:
        pos_manager.close()
        logging.info("PositionManager closed.")


async def main():
    handlers = [("M1", m1_candle_handler), ("H4", h4_candle_handler)]
    await initialize_history("USD_JPY")
    # 複数の無限ループを並列で実行する。
    # - start_candle_stream: Tick データとローソク足生成
    # - logic_loop: トレーディングロジック
    # - news_fetch_loop: 経済指標 RSS 取得
    # - summary_ingest_loop: GCS summary/ から DB への取り込み
    await asyncio.gather(
        start_candle_stream("USD_JPY", handlers),
        logic_loop(),
        exit_loop(),
        news_fetch_loop(),
        summary_ingest_loop(),
        kaizen_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
