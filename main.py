import asyncio
import datetime
import logging
import traceback
import time
from typing import Optional

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
from analytics.insight_client import InsightClient
from signals.pocket_allocator import alloc, DEFAULT_SCALP_SHARE, dynamic_scalp_share
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    check_global_drawdown,
)
from execution.order_manager import (
    market_order,
    close_trade,
    update_dynamic_protections,
    plan_partial_reductions,
)
from execution.exit_manager import ExitManager
from execution.position_manager import PositionManager
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from strategies.scalping.m1_scalper import M1Scalper
from utils.oanda_account import get_account_snapshot

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

FALLBACK_EQUITY = 10000.0  # REST失敗時のフォールバック

STAGE_RATIOS = {
    "macro": (0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03),
    "micro": (0.025, 0.02, 0.015, 0.015, 0.012, 0.01, 0.008, 0.008),
    "scalp": (0.5, 0.3, 0.15, 0.05),
}

MIN_SCALP_STAGE_LOT = 0.01  # 1000 units baseline so micro/macro bias does not null scalps


def build_client_order_id(focus_tag: Optional[str], strategy_tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    focus_part = (focus_tag or "hybrid")[:6]
    clean_tag = "".join(ch for ch in strategy_tag if ch.isalnum())[:9] or "sig"
    return f"qr-{ts_ms}-{focus_part}-{clean_tag}"


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
            if action == "OPEN_LONG" and price < avg_price - 0.02:
                logging.info(
                    "[STAGE] Macro buy gating: price %.3f below avg %.3f.", price, avg_price
                )
                return False
            if action == "OPEN_SHORT" and price > avg_price + 0.02:
                logging.info(
                    "[STAGE] Macro sell gating: price %.3f above avg %.3f.", price, avg_price
                )
                return False
        # RSI-based re-entry gates
        if action == "OPEN_LONG":
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
        if action == "OPEN_LONG":
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
        if action == "OPEN_LONG" and momentum > 0:
            logging.info(
                "[STAGE] Scalp buy gating: momentum %.4f positive (stage %d).",
                momentum,
                stage_idx,
            )
            return False
        if action == "OPEN_SHORT" and momentum < 0:
            logging.info(
                "[STAGE] Scalp sell gating: momentum %.4f negative (stage %d).",
                momentum,
                stage_idx,
            )
            return False
        if action == "OPEN_LONG":
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
    current_lot = max(open_units_same_dir, 0) / 100000.0
    cumulative = 0.0
    for stage_idx, fraction in enumerate(plan):
        cumulative += fraction
        stage_target = total_lot * cumulative
        if current_lot + 1e-4 < stage_target:
            if not _stage_conditions_met(
                pocket, stage_idx, action, fac_m1, fac_h4, open_info
            ):
                return 0.0
            next_lot = max(stage_target - current_lot, 0.0)
            remaining = max(total_lot - current_lot, 0.0)
            if pocket == "scalp" and remaining > 0:
                floor = min(MIN_SCALP_STAGE_LOT, remaining)
                next_lot = max(next_lot, floor)
            if remaining > 0:
                next_lot = min(next_lot, remaining)
            logging.info(
                "[STAGE] %s pocket total=%.3f current=%.3f -> next=%.3f (stage %d)",
                pocket,
                round(stage_target, 4),
                round(current_lot, 4),
                round(next_lot, 4),
                stage_idx,
            )
            return round(next_lot, 4)

    logging.info(
        "[STAGE] %s pocket already filled %.3f / %.3f lots. No additional entry.",
        pocket,
        round(current_lot, 4),
        round(total_lot, 4),
    )
    return 0.0


async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)


async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)


async def logic_loop():
    pos_manager = PositionManager()
    exit_manager = ExitManager()
    perf_cache = {}
    news_cache = {}
    insight = InsightClient()
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
                try:
                    insight.refresh()
                except Exception:
                    pass
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

            atr_pips = (fac_m1.get("atr") or 0.0) * 100
            ema20 = fac_m1.get("ema20") or fac_m1.get("ma20")
            close_px = fac_m1.get("close")
            momentum = 0.0
            if ema20 is not None and close_px is not None:
                momentum = close_px - ema20

            scalp_ready = atr_pips >= 2.2 and abs(momentum) >= 0.0015
            if not scalp_ready and fac_m1.get("vol_5m"):
                scalp_ready = atr_pips >= 2.0 and fac_m1["vol_5m"] >= 1.2
            if scalp_ready and "M1Scalper" not in ranked_strategies:
                ranked_strategies.append("M1Scalper")
                logging.info(
                    "[SCALP] Auto-added M1Scalper (ATR %.2f, momentum %.4f, vol5m %.2f).",
                    atr_pips,
                    momentum,
                    fac_m1.get("vol_5m", 0.0),
                )

            focus_tag = gpt.get("focus_tag") or focus
            weight = gpt.get("weight_macro", w_macro)

            evaluated_signals: list[dict] = []
            for sname in ranked_strategies:
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                if sname == "NewsSpikeReversal":
                    raw_signal = cls.check(fac_m1, news_cache.get("short", []))
                else:
                    raw_signal = cls.check(fac_m1)
                if not raw_signal:
                    continue
                signal = {
                    "strategy": sname,
                    "pocket": cls.pocket,
                    "action": raw_signal.get("action"),
                    "confidence": int(raw_signal.get("confidence", 50) or 50),
                    "sl_pips": raw_signal.get("sl_pips"),
                    "tp_pips": raw_signal.get("tp_pips"),
                    "tag": raw_signal.get("tag", cls.name),
                }
                evaluated_signals.append(signal)
                logging.info("[SIGNAL] %s -> %s", cls.name, signal)

            open_positions = pos_manager.get_open_positions()
            try:
                update_dynamic_protections(open_positions, fac_m1, fac_h4)
            except Exception as exc:
                logging.warning("[PROTECTION] update failed: %s", exc)
            try:
                partials = plan_partial_reductions(open_positions, fac_m1)
            except Exception as exc:
                logging.warning("[PARTIAL] planning failed: %s", exc)
                partials = []
            partial_closed = False
            for pocket, trade_id, reduce_units in partials:
                ok = await close_trade(trade_id, reduce_units)
                if ok:
                    logging.info(
                        "[PARTIAL] trade=%s pocket=%s units=%s",
                        trade_id,
                        pocket,
                        reduce_units,
                    )
                    partial_closed = True
            if partial_closed:
                open_positions = pos_manager.get_open_positions()
            net_units = int(open_positions.get("__net__", {}).get("units", 0))

            exit_decisions = exit_manager.plan_closures(
                open_positions, evaluated_signals, fac_m1, fac_h4, event_soon
            )

            executed_pockets: set[str] = set()
            for decision in exit_decisions:
                pocket = decision.pocket
                remaining = abs(decision.units)
                target_side = "long" if decision.units < 0 else "short"
                trades = (open_positions.get(pocket, {}) or {}).get("open_trades", [])
                trades = [t for t in trades if t.get("side") == target_side]
                for tr in trades:
                    if remaining <= 0:
                        break
                    trade_units = abs(int(tr.get("units", 0) or 0))
                    if trade_units == 0:
                        continue
                    close_amount = min(remaining, trade_units)
                    sign = 1 if target_side == "long" else -1
                    trade_id = tr.get("trade_id")
                    if not trade_id:
                        continue
                    ok = await close_trade(trade_id, sign * close_amount)
                    if ok:
                        logging.info(
                            "[EXIT] trade=%s pocket=%s units=%s reason=%s",
                            trade_id,
                            pocket,
                            sign * close_amount,
                            decision.reason,
                        )
                        remaining -= close_amount
                if remaining > 0:
                    client_id = build_client_order_id(focus_tag, decision.tag)
                    fallback_units = -remaining if decision.units < 0 else remaining
                    trade_id = await market_order(
                        "USD_JPY",
                        fallback_units,
                        None,
                        None,
                        pocket,
                        client_order_id=client_id,
                        reduce_only=True,
                    )
                    if trade_id:
                        logging.info(
                            "[EXIT] %s pocket=%s units=%s reason=%s client_id=%s",
                            trade_id,
                            pocket,
                            fallback_units,
                            decision.reason,
                            client_id,
                        )
                        remaining = 0
                    else:
                        logging.error(
                            "[EXIT FAILED] pocket=%s units=%s reason=%s",
                            pocket,
                            decision.units,
                            decision.reason,
                        )
                if remaining <= 0 and not decision.allow_reentry:
                    executed_pockets.add(pocket)

            if not evaluated_signals:
                logging.info("[WAIT] No actionable entry signals this cycle.")

            sl_values = [
                s["sl_pips"] for s in evaluated_signals if s.get("sl_pips") is not None
            ]
            avg_sl = sum(sl_values) / len(sl_values) if sl_values else 20.0

            try:
                account_snapshot = get_account_snapshot()
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "[ACCOUNT] Failed to load snapshot, fallback equity %.0f: %s",
                    FALLBACK_EQUITY,
                    exc,
                )
                account_snapshot = None

            account_equity = FALLBACK_EQUITY
            margin_available = None
            margin_rate = None
            scalp_buffer = None
            scalp_free_ratio = None

            if account_snapshot:
                account_equity = account_snapshot.nav or account_snapshot.balance or FALLBACK_EQUITY
                margin_available = account_snapshot.margin_available
                margin_rate = account_snapshot.margin_rate
                scalp_buffer = account_snapshot.health_buffer
                scalp_free_ratio = account_snapshot.free_margin_ratio
                if scalp_buffer is not None and scalp_buffer < 0.06:
                    logging.warning(
                        "[ACCOUNT] Margin health critical buffer=%.3f free=%.1f%%",
                        scalp_buffer,
                        scalp_free_ratio * 100 if scalp_free_ratio is not None else 0.0,
                    )

            lot_total = allowed_lot(
                account_equity,
                sl_pips=max(1.0, avg_sl),
                margin_available=margin_available,
                price=fac_m1.get("close"),
                margin_rate=margin_rate,
            )
            requested_pockets = {
                STRATEGIES[s].pocket
                for s in ranked_strategies
                if STRATEGIES.get(s)
            }
            scalp_share = 0.0
            if "scalp" in requested_pockets:
                if account_snapshot:
                    scalp_share = dynamic_scalp_share(account_snapshot, DEFAULT_SCALP_SHARE)
                    logging.info(
                        "[SCALP] share=%.3f buffer=%.3f free=%.1f%%",
                        scalp_share,
                        scalp_buffer if scalp_buffer is not None else -1.0,
                        (scalp_free_ratio * 100) if scalp_free_ratio is not None else -1.0,
                    )
                else:
                    scalp_share = DEFAULT_SCALP_SHARE
            lots = alloc(lot_total, weight, scalp_share=scalp_share)
            active_pockets = {sig["pocket"] for sig in evaluated_signals}
            for key in list(lots):
                if key not in active_pockets:
                    lots[key] = 0.0

            for signal in evaluated_signals:
                pocket = signal["pocket"]
                action = signal.get("action")
                if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                    continue
                if event_soon and pocket in {"micro", "scalp"}:
                    logging.info("[SKIP] Event soon, skipping %s pocket trade.", pocket)
                    continue
                if pocket in executed_pockets:
                    logging.info("[SKIP] %s pocket already handled this loop.", pocket)
                    continue
                if not can_trade(pocket):
                    logging.info(f"[SKIP] DD limit for {pocket} pocket.")
                    continue

                total_lot_for_pocket = lots.get(pocket, 0.0)
                if total_lot_for_pocket <= 0:
                    continue

                confidence = max(0, min(100, signal.get("confidence", 50)))
                confidence_factor = max(0.2, confidence / 100.0)
                confidence_target = round(total_lot_for_pocket * confidence_factor, 3)
                if confidence_target <= 0:
                    continue

                # Apply lot multiplier from insights per pocket side
                side = "LONG" if action == "OPEN_LONG" else "SHORT"
                try:
                    m = float(insight.get_multiplier(pocket, side))
                except Exception:
                    m = 1.0
                if abs(m - 1.0) > 1e-3:
                    adj = round(confidence_target * m, 3)
                    logging.info(
                        "[INSIGHT] pocket=%s side=%s multiplier=%.3f confTarget %.3f -> %.3f",
                        pocket,
                        side,
                        m,
                        confidence_target,
                        adj,
                    )
                    confidence_target = adj

                open_info = open_positions.get(pocket, {})
                price = fac_m1.get("close")
                if action == "OPEN_LONG":
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
                    confidence_target,
                    open_units,
                    action,
                    fac_m1,
                    fac_h4,
                    stage_context,
                )
                if staged_lot <= 0:
                    continue

                units = int(round(staged_lot * 100000)) * (
                    1 if action == "OPEN_LONG" else -1
                )
                if units == 0:
                    logging.info(
                        "[SKIP] Stage lot %.3f produced 0 units. Skipping.", staged_lot
                    )
                    continue

                sl_pips = signal.get("sl_pips")
                tp_pips = signal.get("tp_pips")
                if sl_pips is None or tp_pips is None:
                    logging.info("[SKIP] Missing SL/TP for %s.", signal["strategy"])
                    continue

                sl, tp = clamp_sl_tp(
                    price,
                    price - sl_pips / 100,
                    price + tp_pips / 100,
                    action == "OPEN_LONG",
                )

                client_id = build_client_order_id(focus_tag, signal["tag"])
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl,
                    tp,
                    pocket,
                    client_order_id=client_id,
                )
                if trade_id:
                    logging.info(
                        "[ORDER] %s | %s | %.3f lot | SL=%.3f | TP=%.3f | conf=%d | client_id=%s",
                        trade_id,
                        signal["tag"],
                        staged_lot,
                        sl,
                        tp,
                        confidence,
                        client_id,
                    )
                    pos_manager.register_open_trade(trade_id, pocket, client_id)
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
                    logging.error(f"[ORDER FAILED] {signal['strategy']}")

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
