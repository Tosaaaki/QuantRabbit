import asyncio
import datetime
import logging
import traceback
try:
    from google.cloud import firestore  # type: ignore
    fs = firestore.Client()
except Exception:
    firestore = None  # type: ignore
    class _NoopFS:
        def collection(self, *a, **kw):
            class _NoopDoc:
                def document(self, *a, **kw):
                    class _NoopRef:
                        def set(self, *a, **kw):
                            return None
                    return _NoopRef()
            return _NoopDoc()
    fs = _NoopFS()

from market_data.candle_fetcher import (
    Candle,
    start_candle_stream,
    initialize_history,
    fetch_historical_candles,
)
from indicators.factor_cache import all_factors, on_candle
from analysis.regime_classifier import classify, THRESH_ADX_TREND, THRESH_MA_SLOPE, THRESH_BBW_RANGE
from analysis.focus_decider import decide_focus
from analysis.gpt_decider import get_decision
from analysis.perf_monitor import snapshot as get_perf
from analysis.summary_ingestor import check_event_soon, get_latest_news
from analysis.instrument_selector import rank_by_atr, is_resource_currency_pair
# バックグラウンドでニュース取得と要約を実行するためのインポート
from market_data.news_fetcher import fetch_loop as news_fetch_loop
from analysis.summary_ingestor import ingest_loop as summary_ingest_loop
# from signals.pocket_allocator import alloc
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    check_global_drawdown,
    recent_loss_streak,
)
from execution.order_manager import market_order
from execution.position_manager import PositionManager
from execution.trade_actions import close_trade, update_trade_orders
from strategies.trend.ma_cross import MovingAverageCross
from strategies.trend.ma_rsi_macd import MaRsiMacd
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
    "MA_RSI_MACD": MaRsiMacd,
    "Donchian55": Donchian55,
    "BB_RSI": BBRsi,
    "NewsSpikeReversal": NewsSpikeReversal,
}

EQUITY = 10000.0  # ← 実際は REST から取得
BASE_INSTRUMENT = "USD_JPY"
WATCHLIST = ["USD_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY"]


async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)


async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)


async def logic_loop():
    pos_manager = PositionManager()
    active_trades: dict[str, dict] = {}
    last_trade_time: dict[str, datetime.datetime] = {}
    cooldown_sec_by_strategy = {
        "TrendMA": 15 * 60,
        "Donchian55": 30 * 60,
        "BB_RSI": 5 * 60,
        "NewsSpikeReversal": 20 * 60,
    }
    max_active_by_pocket = {"micro": 3, "macro": 3}
    perf_cache = {}
    news_cache = {}
    last_update_time = datetime.datetime.min
    last_heartbeat_time = datetime.datetime.min  # Add this line
    # Instrument selection cache
    selected_instruments: list[str] = [BASE_INSTRUMENT]
    last_select_time = datetime.datetime.min
    # ---- GPTコールのレート制御／レジーム変化検知用の簡易状態 ----
    last_gpt_call_time = datetime.datetime.min
    prev_adx_m1 = None
    prev_adx_h4 = None
    prev_bbw_m1 = None
    prev_bbw_h4 = None
    prev_slope_m1 = None
    prev_slope_h4 = None
    last_macro_regime = None
    last_micro_regime = None
    pending_macro = None
    pending_micro = None
    pending_macro_count = 0
    pending_micro_count = 0

    try:
        while True:
            now = datetime.datetime.utcnow()

            # Heartbeat logging (and lightweight status update)
            if (now - last_heartbeat_time).total_seconds() >= 300:  # Every 5 minutes
                logging.info(
                    f"[HEARTBEAT] System is alive at {now.isoformat(timespec='seconds')}"
                )
                try:
                    import os as _os
                    loop_sec = int(_os.environ.get("LOOP_SEC", "60"))
                except Exception:
                    loop_sec = 60
                try:
                    hb = {
                        "ts": now.isoformat(timespec="seconds"),
                        "mode": "VM",
                        "heartbeat": now.isoformat(timespec="seconds"),
                        "loop_sec": loop_sec,
                    }
                    fs.collection("status").document("trader").set(hb, merge=True)
                except Exception:
                    pass
                last_heartbeat_time = now

            # 5分ごとにパフォーマンスとニュースを更新
            if (now - last_update_time).total_seconds() >= 300:
                perf_cache = get_perf()
                news_cache = get_latest_news()
                last_update_time = now
                logging.info(f"[PERF] Updated: {perf_cache}")
                logging.info(f"[NEWS] Updated: {news_cache}")
                # 15分ごとにボラ上位の銘柄を選定（JPYクロス）
                if (now - last_select_time).total_seconds() >= 900:
                    try:
                        selected_instruments = await rank_by_atr(WATCHLIST, top_k=2)
                        if BASE_INSTRUMENT not in selected_instruments:
                            selected_instruments.insert(0, BASE_INSTRUMENT)
                        logging.info(f"[SELECT] Instruments: {selected_instruments}")
                    except Exception as e:
                        logging.error(f"[SELECT] instrument ranking failed: {e}")
                        selected_instruments = [BASE_INSTRUMENT]
                    last_select_time = now

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
            # perf_cache からPFを抽出して渡す
            focus, w_macro = decide_focus(
                macro_regime,
                micro_regime,
                macro_pf=(perf_cache.get("macro", {}) or {}).get("pf"),
                micro_pf=(perf_cache.get("micro", {}) or {}).get("pf"),
                event_soon=event_soon,
            )

            # --- 既存ポジ管理（BE/トレーリング/イベント前フラット）---
            try:
                cur_price = float(fac_m1.get("close"))
                atr_m1 = float(fac_m1.get("atr", 0.0) or 0.0)
                atr_h4 = float(fac_h4.get("atr", 0.0) or 0.0)
                to_remove: list[str] = []
                # イベント前は micro を原則クローズ（NewsSpikeReversal は例外）
                if event_soon:
                    for tid, st in list(active_trades.items()):
                        if st.get("pocket") == "micro" and st.get("strategy") != "NewsSpikeReversal":
                            if close_trade(tid):
                                logging.info(f"[EVENT FLAT] closed micro trade={tid}")
                                to_remove.append(tid)
                for tid, st in list(active_trades.items()):
                    if tid in to_remove:
                        continue
                    side = st.get("side")  # 'buy' or 'sell'
                    entry = float(st.get("entry", 0.0))
                    sl = float(st.get("sl", 0.0))
                    tp = float(st.get("tp", 0.0))
                    pocket = st.get("pocket")
                    be_applied = bool(st.get("be_applied"))
                    be_rr = float(st.get("be_rr", 1.0))
                    trail_at_rr = float(st.get("trail_at_rr", 1.5))
                    trail_mult = float(st.get("trail_mult", 2.0 if pocket == "macro" else 1.2))

                    # RR 計算
                    risk = abs(entry - sl)
                    if risk <= 0:
                        continue
                    reward = (cur_price - entry) if side == "buy" else (entry - cur_price)
                    rr = reward / risk

                    # 部分利確（RR >= 1.0 で 50%）
                    if not st.get("partial_done") and rr >= 1.0 and st.get("units", 0) > 1:
                        half = int(max(st.get("units", 0) // 2, 1))
                        if close_trade(tid, units=half):
                            st["units"] = st.get("units", 0) - half
                            st["partial_done"] = True
                            logging.info(f"[PARTIAL] trade={tid} closed {half} units")

                    # BE 移動
                    if not be_applied and rr >= be_rr:
                        new_sl = entry
                        ok = update_trade_orders(tid, sl_price=new_sl)
                        if ok:
                            st["sl"] = new_sl
                            st["be_applied"] = True
                            logging.info(f"[BE] trade={tid} -> SL to BE @ {new_sl:.3f}")

                    # トレーリング（ATR距離、緩め）
                    if rr >= trail_at_rr:
                        atr = atr_h4 if pocket == "macro" else atr_m1
                        dist = max(atr * trail_mult, 0.02)  # 最小距離: 2pips相当
                        if side == "buy":
                            target_sl = max(cur_price - dist, st.get("sl", cur_price))
                            if target_sl > st.get("sl", 0.0) + 1e-6:
                                if update_trade_orders(tid, sl_price=target_sl):
                                    st["sl"] = target_sl
                                    logging.info(f"[TRAIL] trade={tid} SL->{target_sl:.3f}")
                        else:
                            target_sl = min(cur_price + dist, st.get("sl", cur_price))
                            if target_sl < st.get("sl", 1e9) - 1e-6:
                                if update_trade_orders(tid, sl_price=target_sl):
                                    st["sl"] = target_sl
                                    logging.info(f"[TRAIL] trade={tid} SL->{target_sl:.3f}")

                for tid in to_remove:
                    active_trades.pop(tid, None)
            except Exception as e:
                logging.error(f"[MANAGE] error: {e}")

            # --- 2. GPT判断（条件付きで呼ぶ） ---
            # ルール: イベント接近 OR レジーム変化クロス検出 OR 15分クールダウン経過
            cooldown_sec = 15 * 60
            call_due_to_cooldown = (now - last_gpt_call_time).total_seconds() >= cooldown_sec

            # ヒステリシス付きしきい値
            adx_enter_m1 = THRESH_ADX_TREND["M1"]
            adx_enter_h4 = THRESH_ADX_TREND["H4"]
            adx_exit_m1 = adx_enter_m1 - 3.0
            adx_exit_h4 = adx_enter_h4 - 3.0
            bbw_th_m1 = THRESH_BBW_RANGE["M1"]
            bbw_th_h4 = THRESH_BBW_RANGE["H4"]
            bbw_exit_m1 = bbw_th_m1 + 0.05
            bbw_exit_h4 = bbw_th_h4 + 0.05
            slope_enter_m1 = THRESH_MA_SLOPE["M1"]
            slope_enter_h4 = THRESH_MA_SLOPE["H4"]
            slope_exit_m1 = slope_enter_m1 / 2.0
            slope_exit_h4 = slope_enter_h4 / 2.0

            # 現在の値
            cur_adx_m1 = fac_m1.get("adx", 0.0)
            cur_adx_h4 = fac_h4.get("adx", 0.0)
            cur_bbw_m1 = fac_m1.get("bbw", 1.0)
            cur_bbw_h4 = fac_h4.get("bbw", 1.0)
            ma10_m1, ma20_m1 = fac_m1.get("ma10", 0.0), fac_m1.get("ma20", 0.0)
            ma10_h4, ma20_h4 = fac_h4.get("ma10", 0.0), fac_h4.get("ma20", 0.0)
            cur_slope_m1 = abs(ma20_m1 - ma10_m1) / ma10_m1 if ma10_m1 else 0.0
            cur_slope_h4 = abs(ma20_h4 - ma10_h4) / ma10_h4 if ma10_h4 else 0.0

            # 3バー連続で新レジームが続いたら切替確定（デバウンス）
            regime_switched = False
            # Macro
            if last_macro_regime is None:
                last_macro_regime = macro_regime
            elif macro_regime == last_macro_regime:
                pending_macro = None
                pending_macro_count = 0
            else:
                if pending_macro == macro_regime:
                    pending_macro_count += 1
                else:
                    pending_macro = macro_regime
                    pending_macro_count = 1
                if pending_macro_count >= 3:
                    last_macro_regime = macro_regime
                    pending_macro = None
                    pending_macro_count = 0
                    regime_switched = True
            # Micro
            if last_micro_regime is None:
                last_micro_regime = micro_regime
            elif micro_regime == last_micro_regime:
                pending_micro = None
                pending_micro_count = 0
            else:
                if pending_micro == micro_regime:
                    pending_micro_count += 1
                else:
                    pending_micro = micro_regime
                    pending_micro_count = 1
                if pending_micro_count >= 3:
                    last_micro_regime = micro_regime
                    pending_micro = None
                    pending_micro_count = 0
                    regime_switched = True

            # しきい値クロス検出（前回値がある場合のみ）
            crossed = False
            if prev_adx_m1 is not None:
                crossed |= (prev_adx_m1 < adx_enter_m1 <= cur_adx_m1) or (prev_adx_m1 > adx_exit_m1 >= cur_adx_m1)
            if prev_adx_h4 is not None:
                crossed |= (prev_adx_h4 < adx_enter_h4 <= cur_adx_h4) or (prev_adx_h4 > adx_exit_h4 >= cur_adx_h4)
            if prev_bbw_m1 is not None:
                crossed |= (prev_bbw_m1 > bbw_th_m1 >= cur_bbw_m1) or (prev_bbw_m1 < bbw_exit_m1 <= cur_bbw_m1)
            if prev_bbw_h4 is not None:
                crossed |= (prev_bbw_h4 > bbw_th_h4 >= cur_bbw_h4) or (prev_bbw_h4 < bbw_exit_h4 <= cur_bbw_h4)
            if prev_slope_m1 is not None:
                crossed |= (prev_slope_m1 < slope_enter_m1 <= cur_slope_m1) or (prev_slope_m1 > slope_exit_m1 >= cur_slope_m1)
            if prev_slope_h4 is not None:
                crossed |= (prev_slope_h4 < slope_enter_h4 <= cur_slope_h4) or (prev_slope_h4 > slope_exit_h4 >= cur_slope_h4)

            should_call_gpt = event_soon or regime_switched or crossed or call_due_to_cooldown

            if not should_call_gpt:
                await asyncio.sleep(1)
                # 次ループ用に前回値を更新
                prev_adx_m1, prev_adx_h4 = cur_adx_m1, cur_adx_h4
                prev_bbw_m1, prev_bbw_h4 = cur_bbw_m1, cur_bbw_h4
                prev_slope_m1, prev_slope_h4 = cur_slope_m1, cur_slope_h4
                continue

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
            last_gpt_call_time = now
            # 前回値更新
            prev_adx_m1, prev_adx_h4 = cur_adx_m1, cur_adx_h4
            prev_bbw_m1, prev_bbw_h4 = cur_bbw_m1, cur_bbw_h4
            prev_slope_m1, prev_slope_h4 = cur_slope_m1, cur_slope_h4
            weight = gpt.get("weight_macro", w_macro)
            # 0.05 刻みで丸め、[0,1]にクリップ
            try:
                weight = max(0.0, min(1.0, round(float(weight) / 0.05) * 0.05))
            except Exception:
                pass

            # ダッシュボード用のステータスをFirestoreへ反映
            try:
                status_doc = {
                    "ts": now.isoformat(timespec="seconds"),
                    "mode": "VM",
                    "equity": EQUITY,
                    "macro_regime": macro_regime,
                    "micro_regime": micro_regime,
                    "focus_tag": focus,
                    "weight_macro": weight,
                    "event_soon": event_soon,
                    "gpt_ranked": gpt.get("ranked_strategies", []),
                    "news_avg_sent": None,
                    "news_avg_imp": None,
                    "factors_m1": {k: fac_m1.get(k) for k in ("adx","bbw","ma10","ma20","rsi","close")},
                    "factors_h4": {k: fac_h4.get(k) for k in ("adx","bbw","ma10","ma20","rsi","close")},
                    "selected_instruments": selected_instruments,
                    "diversify_slots": 2,
                    "resource_rollover_utc": now.replace(hour=22, minute=0, second=0, microsecond=0).isoformat(timespec="seconds"),
                    "resource_ttl_sec": max(int(((now.replace(hour=22, minute=0, second=0, microsecond=0) + (datetime.timedelta(days=1) if now.hour>=22 else datetime.timedelta())).replace(tzinfo=None) - now).total_seconds()) - 300, 0),
                }
                fs.collection("status").document("trader").set(status_doc)
            except Exception:
                pass

            # --- 3. 発注準備 ---
            # GPT出力の整合性を確認し、未知の戦略名を除外
            allowed = set(STRATEGIES.keys())
            ranked = [s for s in (gpt.get("ranked_strategies") or []) if s in allowed]
            if not ranked:
                ranked = [
                    s for s in ("TrendMA", "Donchian55", "MA_RSI_MACD", "BB_RSI", "NewsSpikeReversal")
                    if s in allowed
                ]
            # 学習: 直近実績で再ランク
            try:
                ranked = re_rank_strategies(ranked, macro_regime, micro_regime)
            except Exception as e:
                logging.error(f"[LEARN] re-rank failed: {e}")
            gpt["ranked_strategies"] = ranked
            # per-trade で実際のSL距離から lot を再計算する方式に変更

            # --- focus_tag に基づく戦略フィルタ/優先度調整 ---
            macro_names = {name for name, cls in STRATEGIES.items() if getattr(cls, "pocket", "macro") == "macro"}
            micro_names = {name for name, cls in STRATEGIES.items() if getattr(cls, "pocket", "micro") == "micro"}
            ranked0 = list(ranked)
            if focus == "event":
                ranked = [s for s in ranked if s == "NewsSpikeReversal"]
                if not ranked and "NewsSpikeReversal" in STRATEGIES:
                    ranked = ["NewsSpikeReversal"]
            elif focus == "macro":
                ranked = [s for s in ranked if s in macro_names]
                if not ranked:
                    ranked = [s for s in ("TrendMA", "MA_RSI_MACD", "Donchian55") if s in STRATEGIES]
            elif focus == "micro":
                ranked = [s for s in ranked if s in micro_names]
                if not ranked:
                    ranked = [s for s in ("BB_RSI", "NewsSpikeReversal") if s in STRATEGIES]
            else:  # hybrid
                # weight_macro に応じてポケット優先度を与え、安定ソート
                prefer_macro = (gpt.get("weight_macro", w_macro or 0.5) >= 0.5)
                def _prio(name: str) -> int:
                    is_macro = name in macro_names
                    if prefer_macro:
                        return 0 if is_macro else 1
                    return 0 if (name in micro_names) else 1
                ranked = sorted(ranked, key=_prio)
            if ranked != ranked0:
                logging.info(f"[FOCUS FILTER] focus={focus}, ranked={ranked}")

            # --- 4. 戦略実行ループ ---
            # 4. 戦略実行ループ（上位インストゥルメントで試行）
            # 1ループあたりの最大新規建て数（分散エントリー）
            max_new_trades = min(2, len(selected_instruments))
            if focus == "event":
                max_new_trades = 1
            placed = 0

            for sname in gpt.get("ranked_strategies", []):
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                cd = cooldown_sec_by_strategy.get(sname, 600)

                for instrument in selected_instruments:
                    # NewsSpikeReversal は USD_JPY のみに限定
                    if sname == "NewsSpikeReversal" and instrument != "USD_JPY":
                        continue
                    key = f"{sname}@{instrument}"
                    last_ts = last_trade_time.get(key)
                    if last_ts is not None and (now - last_ts).total_seconds() < cd:
                        continue

                    # 参照する指標セットを準備
                    if instrument == BASE_INSTRUMENT:
                        im1, ih4 = fac_m1, fac_h4
                    else:
                        try:
                            h4_c = await fetch_historical_candles(instrument, "H4", 60)
                            m1_c = await fetch_historical_candles(instrument, "M1", 60)
                            if len(h4_c) < 20 or len(m1_c) < 20:
                                continue
                            import pandas as _pd  # local import
                            from indicators.calc_core import IndicatorEngine as _IE
                            ih4 = _IE.compute(_pd.DataFrame(h4_c)[["open","high","low","close"]])
                            im1 = _IE.compute(_pd.DataFrame(m1_c)[["open","high","low","close"]])
                            ih4["close"] = float(h4_c[-1]["close"])  # ensure close present
                            im1["close"] = float(m1_c[-1]["close"])  # ensure close present
                        except Exception:
                            continue

                    # 戦略のシグナル生成
                    if sname == "NewsSpikeReversal":
                        sig = cls.check(im1, news_cache.get("short", []))
                    elif cls.pocket == "macro":
                        sig = cls.check(im1, ih4)
                    else:
                        sig = cls.check(im1)

                    if not sig:
                        continue

                    pocket = cls.pocket
                    # イベント前は micro の新規を停止（NewsSpikeReversal は許可）
                    if event_soon and pocket == "micro" and sname != "NewsSpikeReversal":
                        logging.info("[SKIP] Event soon, skipping non-news micro trade.")
                        continue
                    # 週末クローズ前は新規停止（UTC 金曜 20:00 以降）
                    try:
                        if now.weekday() == 4 and now.hour >= 20:
                            logging.info("[SKIP] Pre-weekend, skipping new trades.")
                            continue
                    except Exception:
                        pass
                    # ポケットごとの同時建玉上限
                    actives_in_pocket = sum(1 for st in active_trades.values() if st.get("pocket") == pocket)
                    if actives_in_pocket >= max_active_by_pocket.get(pocket, 3):
                        logging.info(f"[SKIP] Active cap for {pocket} pocket.")
                        continue
                    if not can_trade(pocket):
                        logging.info(f"[SKIP] DD limit for {pocket} pocket.")
                        continue

                    price = float(im1.get("close"))
                    sl_pips = sig.get("sl_pips")
                    tp_pips = sig.get("tp_pips")
                    # ATR連動
                    if sig.get("sl_atr_mult") or sig.get("tp_atr_mult"):
                        atr_src = ih4 if pocket == "macro" else im1
                        atr_val = float(atr_src.get("atr", 0.0) or 0.0)
                        pip_div = 100 if instrument.endswith("_JPY") else 10000
                        if sig.get("sl_atr_mult"):
                            sl_pips = int(round(atr_val * pip_div * float(sig.get("sl_atr_mult", 0.0))))
                        if sig.get("tp_atr_mult"):
                            tp_pips = int(round(atr_val * pip_div * float(sig.get("tp_atr_mult", 0.0))))
                    if not sl_pips or not tp_pips:
                        logging.info(f"[SKIP] Missing SL/TP pips for {cls.name}")
                        continue

                    # リスク・ロット（分散エントリー時は按分）
                    pocket_equity = EQUITY * (weight if pocket == "macro" else (1.0 - weight))
                    slot = max_new_trades if max_new_trades > 0 else 1
                    pocket_equity = pocket_equity / slot
                    # 学習: 戦略/ポケットに応じてリスク係数調整
                    try:
                        mult = risk_multiplier(pocket, sname)
                    except Exception:
                        mult = 1.0
                    base_risk = 0.02
                    # 連敗中は安全側に低減
                    try:
                        streak = recent_loss_streak(pocket)
                        if streak >= 2:
                            base_risk *= 0.5
                        elif streak >= 1:
                            base_risk *= 0.75
                    except Exception:
                        pass
                    # 極端に小さい配分はスキップ
                    if pocket_equity < 100.0:
                        continue
                    lot = allowed_lot(pocket_equity, sl_pips=float(sl_pips), risk_pct=base_risk * mult)
                    if lot <= 0:
                        continue
                    units = int(lot * 100000) * (1 if sig["action"] == "buy" else -1)

                    # 価格に変換（JPYクロスは 1pip=0.01）
                    pip_div = 100 if instrument.endswith("_JPY") else 10000

                    # スワップ直前の資源国通貨は買いを優先（売りは見送り）
                    if is_resource_currency_pair(instrument):
                        try:
                            rollover_hour = 22
                            target = now.replace(hour=rollover_hour, minute=0, second=0, microsecond=0)
                            if target <= now:
                                target = target + datetime.timedelta(days=1)
                            secs = (target - now).total_seconds()
                            if secs <= 3600 and sig.get("action") == "sell":
                                logging.info(f"[SKIP] Pre-rollover avoid short for resource pair: {instrument}")
                                continue
                        except Exception:
                            pass

                    sl, tp = clamp_sl_tp(
                        price,
                        price - sl_pips / pip_div,
                        price + tp_pips / pip_div,
                        sig["action"] == "buy",
                    )

                    # 資源国通貨は宵越ししない: 次のスワップ時刻前に TTL を設定
                    ttl = sig.get("ttl_sec")
                    if ttl is None and is_resource_currency_pair(instrument):
                        try:
                            # Assume rollover at 22:00 UTC; close 5 minutes before
                            rollover_hour = 22
                            target = now.replace(hour=rollover_hour, minute=0, second=0, microsecond=0)
                            if target <= now:
                                target = target + datetime.timedelta(days=1)
                            ttl = max(int((target - now).total_seconds() - 300), 300)
                        except Exception:
                            ttl = None

                    trade_id = await market_order(
                        instrument, units, sl, tp, pocket,
                        strategy=sname, macro_regime=macro_regime, micro_regime=micro_regime,
                    )
                    if trade_id:
                        logging.info(
                            f"[ORDER] {trade_id} | {instrument} | {cls.name} | {lot} lot | SL={sl}, TP={tp}"
                        )
                        active_trades[trade_id] = {
                            "instrument": instrument,
                            "side": "buy" if units > 0 else "sell",
                            "pocket": pocket,
                            "entry": price,
                            "sl": sl,
                            "tp": tp,
                            "units": abs(units),
                            "strategy": sname,
                            "partial_done": False,
                            "be_applied": False,
                            "be_rr": float(sig.get("be_rr", 1.0)),
                            "trail_at_rr": float(sig.get("trail_at_rr", 1.5)),
                            "trail_mult": float(sig.get("trail_atr_mult", 2.0 if pocket == "macro" else 1.2)),
                        }
                        if ttl:
                            async def _ttl_close(tid: str, sec: int):
                                try:
                                    await asyncio.sleep(int(sec))
                                    ok = close_trade(tid)
                                    if ok:
                                        logging.info(f"[TTL CLOSE] trade={tid} after {sec}s")
                                        active_trades.pop(tid, None)
                                except Exception as e:
                                    logging.error(f"[TTL CLOSE] error for {tid}: {e}")
                            asyncio.create_task(_ttl_close(trade_id, int(ttl)))
                        last_trade_time[key] = now
                        placed += 1
                        if placed >= max_new_trades:
                            break
                        # 次のインストゥルメントへ（同戦略で分散）
                        continue
                    else:
                        logging.error(f"[ORDER FAILED] {cls.name} @ {instrument}")
                else:
                    # inner loop didn't place trade; try next strategy
                    continue
                if placed >= max_new_trades:
                    break

            # --- 5. 決済済み取引の同期 ---
            pos_manager.sync_trades()

            # Loop cadence (seconds). Default 60s; can be lowered on VM.
            import os as _os
            try:
                loop_sec = int(_os.environ.get("LOOP_SEC", "60"))
            except Exception:
                loop_sec = 60
            await asyncio.sleep(max(loop_sec, 1))
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
