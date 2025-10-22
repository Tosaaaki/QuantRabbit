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
from signals.pocket_allocator import alloc, DEFAULT_SCALP_SHARE
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    check_global_drawdown,
)
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from strategies.scalping.m1_scalper import M1Scalper

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
    "M1Scalper": M1Scalper,
}

EQUITY = 10000.0  # ← 実際は REST から取得

STAGE_RATIOS = {
    "macro": (0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03),
    "micro": (0.025, 0.02, 0.015, 0.015, 0.012, 0.01, 0.008, 0.008),
    "scalp": (0.02, 0.015, 0.012, 0.01, 0.008, 0.006),
}


def _stage_conditions_met(
    pocket: str,
    stage_idx: int,
    action: str,
    fac_m1: dict[str, float],
    fac_h4: dict[str, float],
    open_info: dict[str, float],
) -> bool:
    if stage_idx == 0:
        return True

    price = fac_m1.get("close")
    avg_price = open_info.get("avg_price", price or 0.0)
    rsi = fac_m1.get("rsi", 50.0)
    adx_h4 = fac_h4.get("adx", 0.0)
    slope_h4 = abs(fac_h4.get("ma20", 0.0) - fac_h4.get("ma10", 0.0))

    if pocket == "macro":
        # Require trend strength to increase with each stage
        if adx_h4 < 20 + stage_idx * 2 or slope_h4 < 0.0005:
            logging.info(
                "[STAGE] Macro gating failed (ADX %.2f, slope %.5f) for stage %d.",
                adx_h4,
                slope_h4,
                stage_idx,
            )
            return False
        if price is not None and avg_price:
            if action == "buy" and price < avg_price - 0.02:
                logging.info(
                    "[STAGE] Macro buy gating: price %.3f below avg %.3f.", price, avg_price
                )
                return False
            if action == "sell" and price > avg_price + 0.02:
                logging.info(
                    "[STAGE] Macro sell gating: price %.3f above avg %.3f.", price, avg_price
                )
                return False
        # RSI-based re-entry gates
        if action == "buy":
            threshold = 60 - stage_idx * 5
            if rsi > threshold:
                logging.info(
                    "[STAGE] Macro buy gating: RSI %.1f > %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        else:
            threshold = 40 + stage_idx * 5
            if rsi < threshold:
                logging.info(
                    "[STAGE] Macro sell gating: RSI %.1f < %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        return True

    if pocket == "micro":
        # mean reversion pocket requires RSI extremes to persist
        if action == "buy":
            threshold = 45 - min(stage_idx * 5, 15)
            if rsi > threshold:
                logging.info(
                    "[STAGE] Micro buy gating: RSI %.1f > %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        else:
            threshold = 55 + min(stage_idx * 5, 15)
            if rsi < threshold:
                logging.info(
                    "[STAGE] Micro sell gating: RSI %.1f < %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        return True

    if pocket == "scalp":
        atr = fac_m1.get("atr", 0.0) * 100
        if atr < 1.5:
            logging.info("[STAGE] Scalp gating: ATR %.2f too low for stage %d.", atr, stage_idx)
            return False
        momentum = (fac_m1.get("close") or 0.0) - (fac_m1.get("ema20") or 0.0)
        if action == "buy" and momentum > 0:
            logging.info(
                "[STAGE] Scalp buy gating: momentum %.4f positive (stage %d).",
                momentum,
                stage_idx,
            )
            return False
        if action == "sell" and momentum < 0:
            logging.info(
                "[STAGE] Scalp sell gating: momentum %.4f negative (stage %d).",
                momentum,
                stage_idx,
            )
            return False
        if action == "buy":
            if rsi > 55 - min(stage_idx * 4, 12):
                logging.info(
                    "[STAGE] Scalp buy gating: RSI %.1f too high (stage %d).",
                    rsi,
                    stage_idx,
                )
                return False
        else:
            if rsi < 45 + min(stage_idx * 4, 12):
                logging.info(
                    "[STAGE] Scalp sell gating: RSI %.1f too low (stage %d).",
                    rsi,
                    stage_idx,
                )
                return False
        return True

    return True


def compute_stage_lot(
    pocket: str,
    total_lot: float,
    open_units_same_dir: int,
    action: str,
    fac_m1: dict[str, float],
    fac_h4: dict[str, float],
    open_info: dict[str, float],
) -> float:
    """段階的エントリーの次ロットを返す（lot単位）。0 の場合は追加不要。"""
    plan = STAGE_RATIOS.get(pocket, (1.0,))
    current_lot = round(max(open_units_same_dir, 0) / 100000, 3)
    cumulative = 0.0
    for stage_idx, fraction in enumerate(plan):
        cumulative = round(cumulative + fraction, 6)
        stage_target = round(total_lot * cumulative, 3)
        if current_lot + 1e-3 < stage_target:
            if not _stage_conditions_met(
                pocket, stage_idx, action, fac_m1, fac_h4, open_info
            ):
                return 0.0
            next_lot = max(stage_target - current_lot, 0.0)
            logging.info(
                "[STAGE] %s pocket total=%.3f current=%.3f -> next=%.3f (stage %d)",
                pocket,
                stage_target,
                current_lot,
                next_lot,
                stage_idx,
            )
            return round(next_lot, 3)

    logging.info(
        "[STAGE] %s pocket already filled %.3f / %.3f lots. No additional entry.",
        pocket,
        current_lot,
        total_lot,
    )
    return 0.0


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
            focus, w_macro = decide_focus(
                macro_regime,
                micro_regime,
                event_soon=event_soon,
                macro_pf=perf_cache.get("macro", {}).get("pf")
                if perf_cache
                else None,
                micro_pf=perf_cache.get("micro", {}).get("pf")
                if perf_cache
                else None,
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
            # GPT判断（フォールバックなし）。失敗時はこのループをスキップ。
            try:
                gpt = await get_decision(payload)
            except Exception as e:
                logging.warning(f"[SKIP] GPT decision unavailable: {e}")
                await asyncio.sleep(5)
                continue
            logging.info(
                "[GPT] focus=%s weight_macro=%.2f strategies=%s",
                gpt.get("focus_tag"),
                gpt.get("weight_macro", 0.0),
                gpt.get("ranked_strategies"),
            )
            ranked_strategies = list(gpt.get("ranked_strategies", []))

            weight = gpt.get("weight_macro", w_macro)

            # --- 3. 発注準備 ---
            lot_total = allowed_lot(EQUITY, sl_pips=20)  # sl_pipsは仮
            requested_pockets = {
                STRATEGIES[s].pocket
                for s in ranked_strategies
                if STRATEGIES.get(s)
            }
            scalp_share = DEFAULT_SCALP_SHARE if "scalp" in requested_pockets else 0.0
            lots = alloc(lot_total, weight, scalp_share=scalp_share)
            if "micro" not in requested_pockets:
                lots["micro"] = 0.0
            if "macro" not in requested_pockets:
                lots["macro"] = 0.0
            if "scalp" not in requested_pockets:
                lots["scalp"] = 0.0
            open_positions = pos_manager.get_open_positions()
            net_units = int(open_positions.get("__net__", {}).get("units", 0))

            executed_pockets: set[str] = set()
            # --- 4. 戦略実行ループ ---
            for sname in ranked_strategies:
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
                logging.info("[SIGNAL] %s -> %s", cls.name, sig)

                pocket = cls.pocket
                if event_soon and pocket in {"micro", "scalp"}:
                    logging.info("[SKIP] Event soon, skipping %s pocket trade.", pocket)
                    continue
                if pocket in executed_pockets:
                    logging.info("[SKIP] %s pocket already traded this loop.", pocket)
                    continue

                if not can_trade(pocket):
                    logging.info(f"[SKIP] DD limit for {pocket} pocket.")
                    continue

                total_lot_for_pocket = lots.get(pocket, 0)
                if total_lot_for_pocket <= 0:
                    continue

                open_info = open_positions.get(pocket, {})
                price = fac_m1.get("close")
                if sig["action"] == "buy":
                    open_units = int(open_info.get("long_units", 0))
                    ref_price = open_info.get("long_avg_price")
                else:
                    open_units = int(open_info.get("short_units", 0))
                    ref_price = open_info.get("short_avg_price")
                stage_context = dict(open_info) if open_info else {}
                if ref_price is None or (ref_price == 0.0 and open_units == 0):
                    ref_price = price
                if ref_price is not None:
                    stage_context["avg_price"] = ref_price
                staged_lot = compute_stage_lot(
                    pocket,
                    total_lot_for_pocket,
                    open_units,
                    sig["action"],
                    fac_m1,
                    fac_h4,
                    stage_context,
                )
                if staged_lot <= 0:
                    continue

                units = int(round(staged_lot * 100000)) * (
                    1 if sig["action"] == "buy" else -1
                )
                if units == 0:
                    logging.info(
                        "[SKIP] Stage lot %.3f produced 0 units. Skipping.", staged_lot
                    )
                    continue

                price = fac_m1.get("close")
                sl, tp = clamp_sl_tp(
                    price,
                    price - sig["sl_pips"] / 100,
                    price + sig["tp_pips"] / 100,
                    sig["action"] == "buy",
                )

                trade_id = await market_order("USD_JPY", units, sl, tp, pocket)
                if trade_id:
                    logging.info(
                        f"[ORDER] {trade_id} | {cls.name} | {staged_lot} lot | SL={sl}, TP={tp}"
                    )
                    pos_manager.register_open_trade(trade_id, pocket)
                    info = open_positions.setdefault(
                        pocket,
                        {
                            "units": 0,
                            "avg_price": price or 0.0,
                            "trades": 0,
                            "long_units": 0,
                            "long_avg_price": 0.0,
                            "short_units": 0,
                            "short_avg_price": 0.0,
                        },
                    )
                    info["units"] = info.get("units", 0) + units
                    info["trades"] = info.get("trades", 0) + 1
                    if price is not None:
                        info["avg_price"] = price
                        if units > 0:
                            prev_units = info.get("long_units", 0)
                            new_units = prev_units + units
                            if new_units > 0:
                                if prev_units == 0:
                                    info["long_avg_price"] = price
                                else:
                                    info["long_avg_price"] = (
                                        info.get("long_avg_price", price) * prev_units
                                        + price * units
                                    ) / new_units
                            info["long_units"] = new_units
                        else:
                            trade_size = abs(units)
                            prev_units = info.get("short_units", 0)
                            new_units = prev_units + trade_size
                            if new_units > 0:
                                if prev_units == 0:
                                    info["short_avg_price"] = price
                                else:
                                    info["short_avg_price"] = (
                                        info.get("short_avg_price", price) * prev_units
                                        + price * trade_size
                                    ) / new_units
                            info["short_units"] = new_units
                    net_units += units
                    open_positions.setdefault("__net__", {})["units"] = net_units
                    executed_pockets.add(pocket)
                else:
                    logging.error(f"[ORDER FAILED] {cls.name}")

                if len(executed_pockets) >= len(STAGE_RATIOS):
                    break

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
        news_fetch_loop(),
        summary_ingest_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
