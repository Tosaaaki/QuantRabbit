import os
import json
import logging
import asyncio
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask
from google.cloud import firestore

from indicators.calc_core import IndicatorEngine
from analysis.regime_classifier import (
    classify,
    THRESH_ADX_TREND,
    THRESH_BBW_RANGE,
    THRESH_MA_SLOPE,
)
from analysis.focus_decider import decide_focus
from analysis.gpt_decider import get_decision
from market_data.candle_fetcher import fetch_historical_candles
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp, check_global_drawdown
from execution.account_info import get_account_summary
from execution.order_manager import market_order
import httpx


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INSTRUMENT = os.environ.get("INSTRUMENT", "USD_JPY")

app = Flask(__name__)
fs = firestore.Client()


def _df_from_candles(candles):
    return pd.DataFrame(
        [{"open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]} for c in candles]
    )


def _get_latest_news(limit_short=3, limit_long=5):
    news_ref = fs.collection("news").order_by("ts_utc", direction=firestore.Query.DESCENDING).limit(50)
    docs = list(news_ref.stream())
    news_short, news_long = [], []
    for d in docs:
        x = d.to_dict()
        item = {
            "summary": x.get("summary", ""),
            "ts": x.get("ts_utc"),
            "sentiment": x.get("sentiment", 0),
            "impact": x.get("impact", 1),
        }
        if x.get("horizon", "short") == "short" and len(news_short) < limit_short:
            news_short.append(item)
        elif x.get("horizon") == "long" and len(news_long) < limit_long:
            news_long.append(item)
        if len(news_short) >= limit_short and len(news_long) >= limit_long:
            break
    # 集約値（最新shortの平均）
    if news_short:
        avg_sent = sum(n.get("sentiment", 0) for n in news_short) / max(len(news_short), 1)
        avg_imp = sum(n.get("impact", 1) for n in news_short) / max(len(news_short), 1)
    else:
        avg_sent, avg_imp = 0.0, 1.0
    return {"short": news_short, "long": news_long, "avg_sent": avg_sent, "avg_imp": avg_imp}


def _get_spread_pips() -> float:
    """Fetch current bid/ask for USD_JPY and return spread in pips."""
    try:
        from utils.secrets import get_secret
        token = get_secret("oanda_token")
        account = get_secret("oanda_account_id")
        try:
            practice = get_secret("oanda_practice").lower() == "true"
        except Exception:
            practice = True
        host = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        url = f"{host}/v3/accounts/{account}/pricing"
        params = {"instruments": INSTRUMENT.replace("_", "/")}
        headers = {"Authorization": f"Bearer {token}"}
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url, params=params, headers=headers)
            r.raise_for_status()
            prices = r.json().get("prices", [])
            if not prices:
                return 0.0
            bids = prices[0].get("bids", [])
            asks = prices[0].get("asks", [])
            if not bids or not asks:
                return 0.0
            bid = float(bids[0]["price"]) ; ask = float(asks[0]["price"]) 
            return max((ask - bid) * 100, 0.0)  # USDJPY: 1 pip = 0.01
    except Exception:
        return 0.0


def _log_trade(doc: dict):
    try:
        fs.collection("trades").add(doc)
    except Exception:
        pass


def _event_soon(within_minutes=30, min_impact=3) -> bool:
    """Return True if an event with sufficient impact occurs within the window.

    Firestore composite indexes are avoided by querying on time range only and
    filtering impact in application code.
    """
    now_dt = datetime.utcnow()
    now = now_dt.isoformat()
    end = (now_dt + timedelta(minutes=within_minutes)).isoformat()
    # 単一不等号のみ使用（>=）にしてインデックス要件を回避
    q = fs.collection("news").where("event_time", ">=", now).limit(10)
    for d in q.stream():
        x = d.to_dict() or {}
        ev = x.get("event_time") or ""
        if ev <= end and int(x.get("impact", 1) or 1) >= min_impact:
            return True
    return False


def _load_state():
    doc = fs.collection("state").document("trader").get()
    return doc.to_dict() if doc.exists else {}


def _save_state(st):
    fs.collection("state").document("trader").set(st)


@app.route("/", methods=["GET"])  # 1 リクエスト = 1 サイクル
def run_once():
    try:
        # Weekend guard (default on). Skip Sat(5)/Sun(6) in UTC to cut costs.
        if os.environ.get("SKIP_WEEKENDS", "true").lower() == "true":
            if datetime.utcnow().weekday() >= 5:
                return "WEEKEND", 200

        st = _load_state() or {}
        # Daily warm‑up: 初回はWARMUP_MIN分スキップ（流動性/スプレッド対策）
        warmup_min = int(os.environ.get("WARMUP_MIN", "30"))
        today = datetime.utcnow().date().isoformat()
        if st.get("warm_key") != today:
            st["warm_key"] = today
            st["warm_until"] = (datetime.utcnow() + timedelta(minutes=warmup_min)).isoformat(timespec="seconds")
            _save_state(st)
        try:
            if datetime.utcnow() < datetime.fromisoformat(st.get("warm_until")):
                return "WARMUP", 200
        except Exception:
            pass

        # 1) 最新ローソクをRESTで取得
        m1 = asyncio.run(fetch_historical_candles(INSTRUMENT, "M1", 60))
        h4 = asyncio.run(fetch_historical_candles(INSTRUMENT, "H4", 60))
        if len(m1) < 20 or len(h4) < 20:
            logging.info("Not enough candles yet.")
            return "WAIT", 200

        # 2) 因子計算
        fac_m1 = IndicatorEngine.compute(_df_from_candles(m1))
        fac_h4 = IndicatorEngine.compute(_df_from_candles(h4))
        fac_m1.update({"close": float(m1[-1]["close"])})
        fac_h4.update({"close": float(h4[-1]["close"])})

        # 3) ニュースとイベント
        news_cache = _get_latest_news()
        event_soon = _event_soon(within_minutes=30, min_impact=3)

        # 4) レジーム・フォーカス
        macro_regime = classify(fac_h4, "H4", event_mode=event_soon)
        micro_regime = classify(fac_m1, "M1", event_mode=event_soon)
        focus, w_macro = decide_focus(
            macro_regime,
            micro_regime,
            event_soon=event_soon,
            macro_pf=None,
            micro_pf=None,
        )

        # 5) ゲーティング（ヒステリシス + 3バー連続はステートで近似）
        now = datetime.utcnow()
        last_call_ts = st.get("last_gpt_call_time")
        cooldown_min = int(os.environ.get("DECIDER_COOLDOWN_MIN", "30"))
        cooldown = max(cooldown_min, 1) * 60
        call_due_to_cooldown = True
        if last_call_ts:
            try:
                dt = datetime.fromisoformat(last_call_ts)
                call_due_to_cooldown = (now - dt).total_seconds() >= cooldown
            except Exception:
                call_due_to_cooldown = True

        # ヒステリシス閾値
        adx_enter_m1 = THRESH_ADX_TREND["M1"]; adx_exit_m1 = adx_enter_m1 - 3.0
        adx_enter_h4 = THRESH_ADX_TREND["H4"]; adx_exit_h4 = adx_enter_h4 - 3.0
        bbw_th_m1 = THRESH_BBW_RANGE["M1"]; bbw_exit_m1 = bbw_th_m1 + 0.05
        bbw_th_h4 = THRESH_BBW_RANGE["H4"]; bbw_exit_h4 = bbw_th_h4 + 0.05
        slope_enter_m1 = THRESH_MA_SLOPE["M1"]; slope_exit_m1 = slope_enter_m1 / 2.0
        slope_enter_h4 = THRESH_MA_SLOPE["H4"]; slope_exit_h4 = slope_enter_h4 / 2.0

        cur_adx_m1 = fac_m1.get("adx", 0.0); cur_adx_h4 = fac_h4.get("adx", 0.0)
        cur_bbw_m1 = fac_m1.get("bbw", 1.0); cur_bbw_h4 = fac_h4.get("bbw", 1.0)
        ma10_m1, ma20_m1 = fac_m1.get("ma10", 0.0), fac_m1.get("ma20", 0.0)
        ma10_h4, ma20_h4 = fac_h4.get("ma10", 0.0), fac_h4.get("ma20", 0.0)
        cur_slope_m1 = abs(ma20_m1 - ma10_m1) / ma10_m1 if ma10_m1 else 0.0
        cur_slope_h4 = abs(ma20_h4 - ma10_h4) / ma10_h4 if ma10_h4 else 0.0

        # 3バー連続デバウンスの近似（前回レジームと pending カウントをFirestoreに保持）
        last_macro = st.get("last_macro_regime")
        last_micro = st.get("last_micro_regime")
        pm = st.get("pending_macro"); pmc = int(st.get("pending_macro_count", 0))
        pmi = st.get("pending_micro"); pmic = int(st.get("pending_micro_count", 0))

        regime_switched = False
        # Macro
        if not last_macro:
            last_macro = macro_regime
        elif macro_regime == last_macro:
            pm = None; pmc = 0
        else:
            if pm == macro_regime:
                pmc += 1
            else:
                pm, pmc = macro_regime, 1
            if pmc >= 3:
                last_macro = macro_regime; pm = None; pmc = 0; regime_switched = True
        # Micro
        if not last_micro:
            last_micro = micro_regime
        elif micro_regime == last_micro:
            pmi = None; pmic = 0
        else:
            if pmi == micro_regime:
                pmic += 1
            else:
                pmi, pmic = micro_regime, 1
            if pmic >= 3:
                last_micro = micro_regime; pmi = None; pmic = 0; regime_switched = True

        # しきい値クロス
        crossed = False
        for key, prev, cur, en, ex in (
            ("adx_m1", st.get("prev_adx_m1"), cur_adx_m1, adx_enter_m1, adx_exit_m1),
            ("adx_h4", st.get("prev_adx_h4"), cur_adx_h4, adx_enter_h4, adx_exit_h4),
        ):
            if prev is not None:
                crossed |= (float(prev) < en <= cur) or (float(prev) > ex >= cur)
        for key, prev, cur, th, ex in (
            ("bbw_m1", st.get("prev_bbw_m1"), cur_bbw_m1, bbw_th_m1, bbw_exit_m1),
            ("bbw_h4", st.get("prev_bbw_h4"), cur_bbw_h4, bbw_th_h4, bbw_exit_h4),
        ):
            if prev is not None:
                crossed |= (float(prev) > th >= cur) or (float(prev) < ex <= cur)
        for key, prev, cur, en, ex in (
            ("slope_m1", st.get("prev_slope_m1"), cur_slope_m1, slope_enter_m1, slope_exit_m1),
            ("slope_h4", st.get("prev_slope_h4"), cur_slope_h4, slope_enter_h4, slope_exit_h4),
        ):
            if prev is not None:
                crossed |= (float(prev) < en <= cur) or (float(prev) > ex >= cur)

        should_call_gpt = event_soon or regime_switched or crossed or call_due_to_cooldown
        if not should_call_gpt:
            # 状態だけ更新して終了
            st.update({
                "prev_adx_m1": cur_adx_m1, "prev_adx_h4": cur_adx_h4,
                "prev_bbw_m1": cur_bbw_m1, "prev_bbw_h4": cur_bbw_h4,
                "prev_slope_m1": cur_slope_m1, "prev_slope_h4": cur_slope_h4,
                "last_macro_regime": last_macro, "last_micro_regime": last_micro,
                "pending_macro": pm, "pending_macro_count": pmc,
                "pending_micro": pmi, "pending_micro_count": pmic,
            })
            _save_state(st)
            return "SKIP", 200

        # 6) GPT 決定
        payload = {
            "ts": now.isoformat(timespec="seconds"),
            "reg_macro": macro_regime,
            "reg_micro": micro_regime,
            "factors_m1": {k: v for k, v in fac_m1.items()},
            "factors_h4": {k: v for k, v in fac_h4.items()},
            "perf": {},
            "news_short": news_cache.get("short", []),
            "news_long": news_cache.get("long", []),
            "event_soon": event_soon,
        }
        gpt = asyncio.run(get_decision(payload))
        weight = gpt.get("weight_macro", 0.5)

        # 7) Lot 配分と戦略実行（最初の成立戦略のみ）
        if check_global_drawdown():
            logging.info("Global DD exceeded; skip trade")
            decision = "DD_SKIP"
        else:
            # --- Equity / margin from OANDA ---
            acc = asyncio.run(get_account_summary())
            equity = acc.get("NAV", 0.0)
            margin_avail = acc.get("marginAvailable", 0.0)
            margin_rate = max(acc.get("marginRate", 0.0), 1e-6)

            # --- Daily target based risk per trade (with dynamic adjust) ---
            daily_target_pct = float(os.environ.get("DAILY_TARGET_PCT", "0.10"))  # 10%
            trades_per_day = int(os.environ.get("TRADES_PER_DAY", "10"))
            risk_share = float(os.environ.get("RISK_SHARE_OF_TARGET", "0.5"))  # 50% of target budget
            risk_pct_base = daily_target_pct * risk_share / max(trades_per_day, 1)
            risk_cap = 0.02
            risk_floor = 0.0005

            # keep start-of-day NAV in state and compute progress vs target
            today = now.date().isoformat()
            day_start_nav = st.get("day_start_nav")
            if (not day_start_nav) or (st.get("day_key") != today):
                day_start_nav = equity
                st["day_start_nav"] = day_start_nav
                st["day_key"] = today
            try:
                progress_pct = (equity - float(day_start_nav)) / float(day_start_nav) if day_start_nav else 0.0
            except Exception:
                progress_pct = 0.0

            if progress_pct < daily_target_pct:
                remaining_ratio = (daily_target_pct - progress_pct) / daily_target_pct
                scale = 1.0 + 0.5 * max(min(remaining_ratio, 1.5), 0.0)
            else:
                scale = 0.5
            risk_pct = max(min(risk_pct_base * scale, risk_cap), risk_floor)

            # Risk model for USD_JPY: pip value per 1 unit ~= 0.01 JPY
            sl_pips = 20
            risk_amount_jpy = equity * risk_pct
            units_by_risk = int(risk_amount_jpy / (sl_pips * 0.01))
            price = fac_m1.get("close") or 150.0
            units_by_margin = int(margin_avail / (price * margin_rate))
            base_units = max(min(units_by_risk, units_by_margin), 0)

            # ニュース係数: sentiment(-2..2)×impact(1..3) を ±20% で補正
            sent = float(news_cache.get("avg_sent", 0.0))
            imp = float(news_cache.get("avg_imp", 1.0))
            news_mult = 1.0 + max(min((sent / 2.0) * (imp / 3.0) * 0.2, 0.2), -0.2)

            # スプレッドが広いときは縮小（上限×0.5）
            spread = _get_spread_pips()
            spread_max = float(os.environ.get("SPREAD_MAX_PIPS", "2.0"))
            spread_mult = 0.5 if spread > spread_max else 1.0

            base_units = int(base_units * news_mult * spread_mult)

            # Convert to lot fraction for allocator (1 lot = 100k units)
            lot_total = round(base_units / 100000.0, 3)
            lots = {"micro": lot_total * (1 - weight), "macro": lot_total * weight}
            decision = None

            # NewsSpikeReversalのみ実装がニュース依存、他はM1のみ
            ranked = gpt.get("ranked_strategies", [])
            from strategies.trend.ma_cross import MovingAverageCross
            from strategies.breakout.donchian55 import Donchian55
            from strategies.mean_reversion.bb_rsi import BBRsi
            from strategies.news.spike_reversal import NewsSpikeReversal
            STRATS = {
                "TrendMA": MovingAverageCross,
                "Donchian55": Donchian55,
                "BB_RSI": BBRsi,
                "NewsSpikeReversal": NewsSpikeReversal,
            }
            for sname in ranked:
                cls = STRATS.get(sname)
                if not cls:
                    continue
                if event_soon and cls.pocket == "micro":
                    continue
                if sname == "NewsSpikeReversal":
                    sig = cls.check(fac_m1, news_cache.get("short", []))
                elif cls.pocket == "macro":
                    sig = cls.check(fac_m1, fac_h4)
                else:
                    sig = cls.check(fac_m1)
                if not sig:
                    continue
                pocket = cls.pocket
                if not can_trade(pocket):
                    continue
                lot = lots.get(pocket, 0)
                if lot <= 0:
                    continue
                price = fac_m1.get("close")
                # 動的SL/TP（ATR倍率）対応
                sl_pips = sig.get("sl_pips")
                tp_pips = sig.get("tp_pips")
                if sig.get("sl_atr_mult") or sig.get("tp_atr_mult"):
                    atr_src = fac_h4 if pocket == "macro" else fac_m1
                    atr_val = float(atr_src.get("atr", 0.0) or 0.0)
                    if sig.get("sl_atr_mult"):
                        sl_pips = int(round(atr_val * 100 * float(sig.get("sl_atr_mult", 0.0))))
                    if sig.get("tp_atr_mult"):
                        tp_pips = int(round(atr_val * 100 * float(sig.get("tp_atr_mult", 0.0))))
                if not sl_pips or not tp_pips:
                    continue
                units = int(lot * 100000) * (1 if sig["action"] == "buy" else -1)
                sl, tp = clamp_sl_tp(
                    price,
                    price - sl_pips / 100,
                    price + tp_pips / 100,
                    sig["action"] == "buy",
                )
                trade_req = {
                    "ts": now.isoformat(timespec="seconds"),
                    "strategy": sname,
                    "pocket": pocket,
                    "instrument": INSTRUMENT,
                    "action": "BUY" if units>0 else "SELL",
                    "units": units,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "lot_total": lot_total,
                    "news_mult": news_mult,
                    "spread_pips": spread,
                }
                trade_id = asyncio.run(
                    market_order(
                        INSTRUMENT, units, sl, tp, pocket,
                        strategy=sname, macro_regime=macro_regime, micro_regime=micro_regime,
                    )
                )
                trade_req["trade_id"] = trade_id or "FAIL"
                _log_trade(trade_req)
                decision = f"ORDER:{trade_id or 'FAIL'}:{sname}:{pocket}:{lot}"
                break

        # 8) 状態保存
        st.update({
            "last_gpt_call_time": now.isoformat(timespec="seconds"),
            "prev_adx_m1": cur_adx_m1, "prev_adx_h4": cur_adx_h4,
            "prev_bbw_m1": cur_bbw_m1, "prev_bbw_h4": cur_bbw_h4,
            "prev_slope_m1": cur_slope_m1, "prev_slope_h4": cur_slope_h4,
            "last_macro_regime": last_macro, "last_micro_regime": last_micro,
            "pending_macro": pm, "pending_macro_count": pmc,
            "pending_micro": pmi, "pending_micro_count": pmic,
            "equity": acc.get("NAV", 0.0) if 'acc' in locals() else None,
            "margin_avail": acc.get("marginAvailable", 0.0) if 'acc' in locals() else None,
            "risk_pct": locals().get("risk_pct", None),
            "day_start_nav": st.get("day_start_nav"),
            "day_key": st.get("day_key"),
        })
        _save_state(st)

        # 9) Write dashboard status snapshot (Firestore /status/trader)
        try:
            status_doc = {
                "ts": now.isoformat(timespec="seconds"),
                "instrument": INSTRUMENT,
                "equity": st.get("equity"),
                "margin_avail": st.get("margin_avail"),
                "macro_regime": macro_regime,
                "micro_regime": micro_regime,
                "focus_tag": focus,
                "weight_macro": w_macro,
                "event_soon": event_soon,
                "cooldown_min": cooldown_min,
                "gpt_ranked": ranked,
                "decision": decision,
                "lot_total": locals().get("lot_total"),
                "spread_pips": locals().get("spread"),
                "news_avg_sent": news_cache.get("avg_sent"),
                "news_avg_imp": news_cache.get("avg_imp"),
                "factors_m1": {k: fac_m1.get(k) for k in ("adx","bbw","ma10","ma20","rsi","close")},
                "factors_h4": {k: fac_h4.get(k) for k in ("adx","bbw","ma10","ma20","rsi","close")},
                "can_trade_micro": bool(can_trade("micro")),
                "can_trade_macro": bool(can_trade("macro")),
                "global_dd_stop": bool(check_global_drawdown()),
                "day_start_nav": st.get("day_start_nav"),
            }
            fs.collection("status").document("trader").set(status_doc)
        except Exception as _e:
            logging.warning(f"status_write_failed: {_e}")

        logging.info(f"trader_result={decision}")
        return decision or "NOOP", 200
    except Exception as e:
        logging.error(f"Unhandled trader error: {e}")
        return "ERROR", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
