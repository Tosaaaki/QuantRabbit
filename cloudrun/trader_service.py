import os
import logging
import json
import asyncio
import math
from collections import defaultdict
from typing import Any, Dict, List
from datetime import datetime, timedelta, timezone

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
from analysis.focus_decider import FocusDecision, decide_focus
from analysis.gpt_decider import get_decision
from market_data.candle_fetcher import fetch_historical_candles
from execution.risk_guard import allowed_lot, can_trade, clamp_sl_tp, check_global_drawdown
from execution.scalp_engine import get_scalp_state, run_scalp_once
from execution.account_info import get_account_summary
from execution.order_manager import market_order
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from strategies.micro.trend_pullback import MicroTrendPullback
from strategies.range.range_bounce import RangeBounce
from signals.pocket_allocator import alloc
import httpx
from utils.firestore_helpers import apply_filter


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

INSTRUMENT = os.environ.get("INSTRUMENT", "USD_JPY")
RUN_SCALP_ON_REQUEST = os.environ.get("SCALP_RUN_ON_CLOUDRUN", "true").lower() == "true"
MARKET_STALE_AFTER_SEC = int(os.environ.get("MARKET_STALE_AFTER_SEC", "600"))

STRONG_TREND_VELOCITY = float(os.environ.get("STRONG_TREND_VELOCITY_30S", "8.0"))
STRONG_TREND_RANGE = float(os.environ.get("STRONG_TREND_RANGE_30S", "12.0"))
STRONG_TREND_MACRO_SHARE = float(os.environ.get("STRONG_TREND_MACRO_SHARE", "0.7"))
MIN_BASELINE_POCKET_LOT = float(os.environ.get("MIN_BASELINE_POCKET_LOT", "0.0001"))
JST = timezone(timedelta(hours=9))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


MIN_SCALP_WEIGHT = max(0.0, _env_float("MIN_SCALP_WEIGHT", 0.04))
MAX_SCALP_WEIGHT = max(MIN_SCALP_WEIGHT, _env_float("MAX_SCALP_WEIGHT", 0.35))
MACRO_SCALP_CAP = max(0.6, _env_float("MACRO_SCALP_SUM_CAP", 0.9))


def _safe_weight(value, default: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = default
    if not math.isfinite(weight):  # type: ignore[name-defined]
        weight = default
    return max(0.0, min(1.0, weight))


def _safe_scalp_weight(value, default: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = default
    if not math.isfinite(weight):  # type: ignore[name-defined]
        weight = default
    return max(MIN_SCALP_WEIGHT, min(MAX_SCALP_WEIGHT, weight))


def _resolve_regime_bias(macro_regime: str, micro_regime: str, strong_trend: bool) -> str | None:
    if strong_trend or macro_regime == "Trend":
        return "macro_trend"
    if micro_regime == "Breakout":
        return "micro_breakout"
    if micro_regime == "Trend":
        return "micro_trend"
    if macro_regime in ("Range", "Mixed") and micro_regime in ("Range", "Mixed"):
        return "range_reversion"
    return None

app = Flask(__name__)
fs = firestore.Client()

LEARNING_LOOKBACK_DAYS = int(os.environ.get("LEARNING_LOOKBACK_DAYS", "7"))
LEARNING_MAX_DOCS = int(os.environ.get("LEARNING_MAX_DOCS", "500"))
LEARNING_CACHE_SEC = int(os.environ.get("LEARNING_CACHE_SEC", "120"))

_learning_cache: Dict[str, Any] = {"ts": datetime.min, "stats": {}}
_config_cache: Dict[str, Any] = {"ts": datetime.min, "data": {}}
_CONFIG_CACHE_SEC = int(os.environ.get("CONFIG_CACHE_SEC", "60"))


def _df_from_candles(candles):
    return pd.DataFrame(
        [{"open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"]} for c in candles]
    )


def _session_label(dt_obj: datetime) -> str:
    hour = dt_obj.hour
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "europe"
    if 12 <= hour < 20:
        return "us"
    return "late_us"


def _hold_bucket(minutes: float | None) -> str:
    if minutes is None:
        return "unknown"
    if minutes < 30:
        return "<30m"
    if minutes < 120:
        return "30-120m"
    if minutes < 360:
        return "120-360m"
    return ">=360m"


def _vol_bucket(atr_value: float | None) -> str:
    if atr_value is None or atr_value <= 0:
        return "unknown"
    pip_atr = atr_value * 100  # USDJPY pip ≒0.01
    if pip_atr < 5:
        return "low"
    if pip_atr < 10:
        return "medium"
    if pip_atr < 20:
        return "high"
    return "extreme"


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


def _score_row(trades: int, wins: int, losses: int, avg_pips: float) -> float:
    total = max(trades, 1)
    win_rate = wins / total
    confidence = min(total / 20.0, 1.0)
    return (win_rate * 2.0 - 1.0) * 0.5 * confidence + (avg_pips / 20.0) * confidence


def _parse_close_time(value: any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str) and value:
        iso = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            dt_obj = datetime.fromisoformat(iso)
            if dt_obj.tzinfo:
                return dt_obj.astimezone(timezone.utc).replace(tzinfo=None)
            return dt_obj
        except ValueError:
            return None
    return None


def _compute_strategy_stats(now: datetime) -> dict[str, dict]:
    lookback = now - timedelta(days=LEARNING_LOOKBACK_DAYS)
    try:
        query = apply_filter(fs.collection("trades"), "state", "==", "CLOSED").limit(LEARNING_MAX_DOCS)
        docs = list(query.stream())
    except Exception as exc:
        logging.warning("[LEARNING] Firestore fetch failed: %s", exc)
        return {}

    stats: dict[str, dict] = {}
    pip_size = 0.01
    for doc in docs:
        data = doc.to_dict() or {}
        close_dt = _parse_close_time(data.get("close_time") or data.get("fill_time"))
        if close_dt is None or close_dt < lookback:
            continue
        strategy = data.get("strategy")
        pocket = data.get("pocket") or "unknown"
        units = data.get("units")
        entry_price = data.get("price")
        close_price = data.get("close_price") or data.get("fill_price")
        if not strategy or units in (None, 0) or entry_price is None or close_price is None:
            continue
        try:
            units_val = float(units)
            entry_val = float(entry_price)
            close_val = float(close_price)
        except (TypeError, ValueError):
            continue
        if units_val == 0:
            continue
        direction = 1.0 if units_val > 0 else -1.0
        pl_pips = (close_val - entry_val) * direction / pip_size

        strat_stat = stats.setdefault(
            strategy,
            {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "sum_pips": 0.0,
                "pockets": defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0}),
                "sessions": defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0}),
                "directions": defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0}),
                "holds": defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0}),
                "volatility": defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0}),
                "macro": defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0}),
                "micro": defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0}),
            },
        )
        strat_stat["trades"] += 1
        strat_stat["sum_pips"] += pl_pips
        if pl_pips > 0:
            strat_stat["wins"] += 1
        elif pl_pips < 0:
            strat_stat["losses"] += 1

        def _upd(bucket, key):
            entry = bucket.setdefault(key, {"trades": 0, "wins": 0, "losses": 0, "sum_pips": 0.0})
            entry["trades"] += 1
            entry["sum_pips"] += pl_pips
            if pl_pips > 0:
                entry["wins"] += 1
            elif pl_pips < 0:
                entry["losses"] += 1

        _upd(strat_stat["pockets"], pocket)
        entry_dt = _parse_close_time(data.get("fill_time") or data.get("entry_time") or data.get("ts"))
        session = data.get("session") or _session_label(entry_dt or close_dt or now)
        _upd(strat_stat["sessions"], session)
        direction_key = "long" if direction > 0 else "short"
        _upd(strat_stat["directions"], direction_key)
        macro_key = data.get("macro_regime") or "?"
        micro_key = data.get("micro_regime") or "?"
        _upd(strat_stat["macro"], macro_key)
        _upd(strat_stat["micro"], micro_key)
        hold_minutes = None
        entry_for_hold = entry_dt
        if entry_for_hold and close_dt:
            hold_minutes = (close_dt - entry_for_hold).total_seconds() / 60.0
        hold_bucket = _hold_bucket(hold_minutes)
        _upd(strat_stat["holds"], hold_bucket)
        atr_val = data.get("atr_m1") or data.get("atr")
        try:
            atr_val = float(atr_val)
        except (TypeError, ValueError):
            atr_val = None
        vol_bucket = _vol_bucket(atr_val)
        _upd(strat_stat["volatility"], vol_bucket)

    for strat, stat in stats.items():
        total = stat["trades"]
        avg = stat["sum_pips"] / total if total else 0.0
        stat["avg_pips"] = avg
        stat["score"] = _score_row(total, stat["wins"], stat["losses"], avg)
        for key in ("pockets", "sessions", "directions", "holds", "volatility", "macro", "micro"):
            bucket = stat[key]
            if isinstance(bucket, defaultdict):
                bucket = dict(bucket)
                stat[key] = bucket
            for name, bstat in bucket.items():
                b_total = bstat["trades"]
                b_avg = bstat["sum_pips"] / b_total if b_total else 0.0
                bstat["avg_pips"] = b_avg
                bstat["score"] = _score_row(b_total, bstat["wins"], bstat["losses"], b_avg)
    return stats


def _get_strategy_stats(now: datetime) -> dict[str, dict]:
    cache_ts = _learning_cache.get("ts")
    if cache_ts and (now - cache_ts).total_seconds() < LEARNING_CACHE_SEC:
        return _learning_cache.get("stats", {})
    stats = _compute_strategy_stats(now)
    _learning_cache["ts"] = now
    _learning_cache["stats"] = stats
    return stats


def _get_config_params(now: datetime) -> dict[str, Any]:
    cache_ts = _config_cache.get("ts")
    if cache_ts and (now - cache_ts).total_seconds() < _CONFIG_CACHE_SEC:
        return _config_cache.get("data", {})
    try:
        doc = fs.collection("config").document("params").get()
        data = doc.to_dict() or {}
    except Exception as exc:
        logging.warning("[CONFIG] fetch failed: %s", exc)
        data = _config_cache.get("data", {})
    _config_cache["ts"] = now
    _config_cache["data"] = data
    return data


def _rerank_with_stats(candidates: list[str], stats: dict[str, dict]) -> list[str]:
    if not candidates:
        return []
    priors = {s: (len(candidates) - i) * 0.01 for i, s in enumerate(candidates)}
    scored = []
    for s in candidates:
        score = stats.get(s, {}).get("score", 0.0)
        scored.append((-(priors.get(s, 0.0) + score), s))
    scored.sort()
    return [s for _, s in scored]


def _log_event(level: int, event: str, **fields: object) -> None:
    """Emit structured JSON for Cloud Logging."""
    payload = {"event": event, **fields}
    logging.log(level, json.dumps(payload, ensure_ascii=False, default=str))


def _risk_multiplier_from_stats(stats: dict[str, dict], strategy: str, pocket: str) -> float:
    strat = stats.get(strategy)
    if not strat:
        return 1.0
    pocket_stats = strat.get("pockets", {}).get(pocket)
    data = pocket_stats or strat
    score = data.get("score") if data else None
    trades = data.get("trades", 0)
    sum_pips = data.get("sum_pips", strat.get("sum_pips", 0.0)) if data else 0.0
    if score is None:
        return 1.0
    if trades >= 5:
        if sum_pips <= -40 or score < -0.15:
            return 0.4
        if sum_pips <= -20 or score < -0.08:
            return 0.6
        if sum_pips >= 60 or score > 0.2:
            return 1.4
        if sum_pips >= 30 or score > 0.1:
            return 1.2
    if score > 0.08:
        return 1.2
    if score < -0.08:
        return 0.7
    return 1.0


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
        if "state" not in doc:
            doc["state"] = "PLACED"
        doc.setdefault("created_at", datetime.utcnow().isoformat(timespec="seconds"))
        if doc.get("trade_id") is not None:
            doc["trade_id"] = str(doc["trade_id"])
        fs.collection("trades").add(doc)
    except Exception:
        pass


def _event_soon(within_minutes=30, min_impact=3) -> bool:
    """Return True if an event with sufficient impact occurs within the window.

    Firestore composite indexes are avoided by querying on time range only and
    filtering impact in application code.
    """
    now_dt = datetime.utcnow()
    start = (now_dt - timedelta(minutes=within_minutes)).isoformat()
    end = (now_dt + timedelta(minutes=within_minutes)).isoformat()
    # 単一不等号のみ使用（>=）にしてインデックス要件を回避（終端はアプリ側で絞り込み）
    q = apply_filter(fs.collection("news"), "event_time", ">=", start).limit(20)
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


def _is_weekend_window(now_utc: datetime) -> bool:
    """Return True during weekend shutdown window (Sat 06:00 JST -> Mon 06:00 JST)."""

    now_jst = now_utc.astimezone(JST)
    dow = now_jst.weekday()  # Mon=0 ... Sun=6
    hour = now_jst.hour
    minute = now_jst.minute

    if dow == 5:  # Saturday
        return (hour, minute) >= (6, 0)
    if dow == 6:  # Sunday
        return True
    if dow == 0:  # Monday before 06:00
        return (hour, minute) < (6, 0)
    return False


def _last_candle_time(candles: List[Dict[str, Any]]) -> datetime | None:
    if not candles:
        return None
    ts = candles[-1].get("time")
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return None


@app.route("/", methods=["GET"])  # 1 リクエスト = 1 サイクル
def run_once():
    try:
        scalp_result: Dict[str, Any] | None = None

        def _status_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
            data = dict(payload)
            data["scalp_state"] = get_scalp_state()
            if scalp_result is not None:
                data.setdefault("scalp_last", scalp_result)
            return data

        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)

        if os.environ.get("SKIP_WEEKENDS", "true").lower() == "true" and _is_weekend_window(now_utc):
            try:
                fs.collection("status").document("trader").set(
                    _status_payload(
                        {
                            "ts": now_utc.isoformat(timespec="seconds"),
                            "instrument": INSTRUMENT,
                            "macro_regime": None,
                            "micro_regime": None,
                            "focus_tag": "weekend",
                            "weight_macro": 0.0,
                            "mode": "weekend",
                            "selected_instruments": [INSTRUMENT],
                            "diversify_slots": 1,
                            "resource_ttl_sec": None,
                            "event_soon": False,
                        }
                    )
                )
            except Exception:
                pass
            return "WEEKEND", 200

        st = _load_state() or {}
        if "last_order_by_strategy" not in st or not isinstance(st.get("last_order_by_strategy"), dict):
            st["last_order_by_strategy"] = {}
        if "last_order_by_pocket" not in st or not isinstance(st.get("last_order_by_pocket"), dict):
            st["last_order_by_pocket"] = {}
        # Daily warm‑up: 初回はWARMUP_MIN分スキップ（流動性/スプレッド対策）
        warmup_min = int(os.environ.get("WARMUP_MIN", "30"))
        today = datetime.utcnow().date().isoformat()
        if st.get("warm_key") != today:
            st["warm_key"] = today
            st["warm_until"] = (now_utc + timedelta(minutes=warmup_min)).isoformat(timespec="seconds")
            _save_state(st)
        try:
            if now_utc < datetime.fromisoformat(st.get("warm_until")):
                # 早期リターンでもダッシュボードに最新時刻とイベント状態を反映
                try:
                    ttl = int((datetime.fromisoformat(st.get("warm_until")) - now_utc).total_seconds())
                except Exception:
                    ttl = None
                try:
                    fs.collection("status").document("trader").set(
                        _status_payload(
                            {
                                "ts": now_utc.isoformat(timespec="seconds"),
                                "instrument": INSTRUMENT,
                                "macro_regime": None,
                                "micro_regime": None,
                                "focus_tag": "warmup",
                                "weight_macro": 0.0,
                                "mode": "warmup",
                                "selected_instruments": [INSTRUMENT],
                                "diversify_slots": 1,
                                "resource_ttl_sec": ttl,
                                "event_soon": _event_soon(within_minutes=30, min_impact=3),
                            }
                        )
                    )
                except Exception:
                    pass
                return "WARMUP", 200
        except Exception:
            pass

        # 1) 事前のイベント検知（早期リターンでも最新の表示を更新するため先に評価）
        event_soon = _event_soon(within_minutes=30, min_impact=3)

        # 2) 最新ローソクをRESTで取得
        m1 = asyncio.run(fetch_historical_candles(INSTRUMENT, "M1", 60))
        h4 = asyncio.run(fetch_historical_candles(INSTRUMENT, "H4", 60))
        if len(m1) < 20 or len(h4) < 20:
            logging.info("Not enough candles yet.")
            try:
                fs.collection("status").document("trader").set(
                    _status_payload(
                        {
                            "ts": now_utc.isoformat(timespec="seconds"),
                            "instrument": INSTRUMENT,
                            "macro_regime": None,
                            "micro_regime": None,
                            "focus_tag": None,
                            "weight_macro": None,
                            "mode": "init",
                            "selected_instruments": [INSTRUMENT],
                            "diversify_slots": 1,
                            "resource_ttl_sec": None,
                            "event_soon": event_soon,
                        }
                    )
                )
            except Exception:
                pass
            return "WAIT", 200

        last_candle_ts = _last_candle_time(m1)
        if last_candle_ts is None:
            last_age = None
        else:
            last_age = (now_utc - last_candle_ts).total_seconds()

        if last_age is None or last_age > MARKET_STALE_AFTER_SEC:
            try:
                fs.collection("status").document("trader").set(
                    _status_payload(
                        {
                            "ts": now_utc.isoformat(timespec="seconds"),
                            "instrument": INSTRUMENT,
                            "macro_regime": None,
                            "micro_regime": None,
                            "focus_tag": "stale",
                            "weight_macro": 0.0,
                            "mode": "market_closed",
                            "selected_instruments": [INSTRUMENT],
                            "diversify_slots": 1,
                            "resource_ttl_sec": None,
                            "event_soon": event_soon,
                            "last_candle_age_sec": last_age,
                        }
                    )
                )
            except Exception:
                pass
            return "MARKET_CLOSED", 200

        # 2) 因子計算
        fac_m1 = IndicatorEngine.compute(_df_from_candles(m1))
        fac_h4 = IndicatorEngine.compute(_df_from_candles(h4))
        fac_m1.update({
            "close": float(m1[-1]["close"]),
            "candles": [
                {
                    "timestamp": c["time"].isoformat(),
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                }
                for c in m1
            ],
        })
        fac_h4.update({
            "close": float(h4[-1]["close"]),
            "candles": [
                {
                    "timestamp": c["time"].isoformat(),
                    "open": c["open"],
                    "high": c["high"],
                    "low": c["low"],
                    "close": c["close"],
                }
                for c in h4
            ],
        })
        try:
            rsi_m1 = float(fac_m1.get("rsi") or 50.0)
        except (TypeError, ValueError):
            rsi_m1 = 50.0

        if RUN_SCALP_ON_REQUEST:
            try:
                spread_for_scalp = _get_spread_pips()
                scalp_result = run_scalp_once(fac_m1, now_utc, spread_pips=spread_for_scalp)
            except Exception as exc:  # noqa: BLE001
                logging.exception("[SCALP] Cloud Run execution failed: %s", exc)
                scalp_result = {"executed": False, "reason": "error", "error": str(exc)}

        # 3) ニュース（イベントは事前評価済み）
        news_cache = _get_latest_news()

        # 4) レジーム・フォーカス
        macro_regime = classify(fac_h4, "H4", event_mode=event_soon)
        micro_regime = classify(fac_m1, "M1", event_mode=event_soon)
        tick_velocity = abs(float(fac_m1.get("tick_velocity_30s") or 0.0))
        tick_range = float(fac_m1.get("tick_range_30s") or 0.0)
        recent_range_pips = 0.0
        candles_m1 = fac_m1.get("candles") or []
        if len(candles_m1) >= 5:
            try:
                window = candles_m1[-5:]
                high = max(float(c.get("high")) for c in window)
                low = min(float(c.get("low")) for c in window)
                recent_range_pips = (high - low) / 0.01
            except (TypeError, ValueError):
                recent_range_pips = 0.0
        strong_trend = (
            macro_regime == "Trend"
            and micro_regime in ("Trend", "Breakout")
            and (
                tick_velocity >= STRONG_TREND_VELOCITY
                or tick_range >= STRONG_TREND_RANGE
                or recent_range_pips >= 4.0
            )
        )
        high_volatility = (
            tick_velocity >= STRONG_TREND_VELOCITY
            or tick_range >= STRONG_TREND_RANGE
            or recent_range_pips >= 4.0
        )
        focus_decision = decide_focus(
            macro_regime,
            micro_regime,
            event_soon=event_soon,
            macro_pf=None,
            micro_pf=None,
            strong_trend=strong_trend,
            high_volatility=high_volatility,
        )
        focus = focus_decision.focus_tag
        w_macro = focus_decision.weight_macro
        w_micro = focus_decision.weight_micro
        w_scalp = focus_decision.weight_scalp

        # 5) ゲーティング（ヒステリシス + 3バー連続はステートで近似）
        now = datetime.utcnow()
        config_params = _get_config_params(now)
        last_call_ts = st.get("last_gpt_call_time")
        cfg_cooldown = config_params.get("DECIDER_COOLDOWN_MIN")
        try:
            cooldown_min = int(cfg_cooldown)
        except (TypeError, ValueError):
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
            # 状態だけ更新（Firestoreのダッシュボードにも反映）して終了
            st.update({
                "prev_adx_m1": cur_adx_m1, "prev_adx_h4": cur_adx_h4,
                "prev_bbw_m1": cur_bbw_m1, "prev_bbw_h4": cur_bbw_h4,
                "prev_slope_m1": cur_slope_m1, "prev_slope_h4": cur_slope_h4,
                "last_macro_regime": last_macro, "last_micro_regime": last_micro,
                "pending_macro": pm, "pending_macro_count": pmc,
                "pending_micro": pmi, "pending_micro_count": pmic,
            })
            _save_state(st)
            try:
                fs.collection("status").document("trader").set(
                    _status_payload(
                        {
                            "ts": datetime.utcnow().isoformat(timespec="seconds"),
                            "instrument": INSTRUMENT,
                            "macro_regime": last_macro,
                            "micro_regime": last_micro,
                            "focus_tag": None,
                            "weight_macro": None,
                            "mode": "idle",
                            "selected_instruments": [INSTRUMENT],
                            "diversify_slots": 1,
                            "resource_ttl_sec": int(cooldown) if 'cooldown' in locals() else None,
                            "event_soon": event_soon,
                        }
                    )
                )
            except Exception:
                pass
            _log_event(
                logging.INFO,
                "skip_cooldown",
                call_due_to_cooldown=call_due_to_cooldown,
                regime_switched=regime_switched,
                event_soon=event_soon,
                last_order_time=st.get("last_order_time"),
                cooldown_sec=cooldown if "cooldown" in locals() else None,
            )
            return "SKIP", 200

        # 6) GPT 決定
        learning_stats = _get_strategy_stats(now)

        decision_hints: Dict[str, Any] = {}
        regime_bias = _resolve_regime_bias(macro_regime, micro_regime, strong_trend)
        if regime_bias:
            decision_hints["regime_bias"] = regime_bias
        context_flags: list[str] = []
        if strong_trend:
            context_flags.append("strong_trend")
        if high_volatility and not strong_trend:
            context_flags.append("high_volatility")
        if event_soon:
            context_flags.append("event_mode")
        if context_flags:
            decision_hints["context_flags"] = context_flags

        payload = {
            "ts": now.isoformat(timespec="seconds"),
            "reg_macro": macro_regime,
            "reg_micro": micro_regime,
            "factors_m1": {k: v for k, v in fac_m1.items() if k != "candles"},
            "factors_h4": {k: v for k, v in fac_h4.items() if k != "candles"},
            "perf": {},
            "news_short": news_cache.get("short", []),
            "news_long": news_cache.get("long", []),
            "event_soon": event_soon,
            "focus_baseline": {
                "focus": focus,
                "weight_macro": w_macro,
                "weight_scalp": w_scalp,
                "weights": {
                    "macro": w_macro,
                    "micro": w_micro,
                    "scalp": w_scalp,
                },
            },
        }
        if decision_hints:
            payload["decision_hints"] = decision_hints
        gpt = asyncio.run(get_decision(payload))
        allowed = ("TrendMA", "Donchian55", "BB_RSI", "NewsSpikeReversal", "MicroTrendPullback")
        directives = gpt.get("strategy_directives", {}) or {}

        try:
            logging.info(
                "[GPT_DECISION] focus=%s weight_macro=%.2f weight_scalp=%.2f ranked=%s",
                gpt.get("focus_tag"),
                float(gpt.get("weight_macro", 0.5)),
                float(gpt.get("weight_scalp", 0.12)),
                gpt.get("ranked_strategies", []),
            )
        except Exception:
            logging.info("[GPT_DECISION] raw=%s", gpt)

        def _is_enabled(name: str) -> bool:
            cfg = directives.get(name) or {}
            return bool(cfg.get("enabled", True))

        def _risk_bias(name: str) -> float:
            cfg = directives.get(name) or {}
            try:
                return float(cfg.get("risk_bias", 1.0))
            except (TypeError, ValueError):
                return 1.0

        def _ensure_directive_enabled(name: str) -> None:
            cfg = directives.setdefault(name, {"enabled": True, "risk_bias": 1.0})
            cfg["enabled"] = True
            try:
                bias = float(cfg.get("risk_bias", 1.0))
            except (TypeError, ValueError):
                bias = 1.0
            cfg["risk_bias"] = max(0.1, min(bias, 2.0))

        ranked_candidates = [s for s in gpt.get("ranked_strategies", []) if s in allowed and _is_enabled(s)]
        for fallback in allowed:
            if fallback not in ranked_candidates and _is_enabled(fallback):
                ranked_candidates.append(fallback)
        ranked_candidates = list(dict.fromkeys(ranked_candidates))
        learning_stats = _get_strategy_stats(now)
        if learning_stats:
            top_entries = sorted(
                ((k, v.get("score", 0.0)) for k, v in learning_stats.items()),
                key=lambda item: -item[1],
            )[:4]
            learning_snapshot = {k: round(score, 3) for k, score in top_entries}
        else:
            learning_snapshot = {}
        ranked = _rerank_with_stats(ranked_candidates, learning_stats) if ranked_candidates else [s for s in allowed if _is_enabled(s)]
        if not ranked:
            ranked = [allowed[0]]

        def _inject_baseline_strategies(lots: Dict[str, float]) -> None:
            priority: list[str] = []
            if lots.get("macro", 0.0) >= MIN_BASELINE_POCKET_LOT:
                priority.append("TrendMA")
            if lots.get("micro", 0.0) >= MIN_BASELINE_POCKET_LOT:
                for candidate in ("MicroTrendPullback", "BB_RSI", "RangeBounce"):
                    if candidate in allowed:
                        priority.append(candidate)
            if not priority:
                return
            logging.info("[BASELINE] ensure strategies=%s lots=%s", priority, lots)
            for name in reversed(priority):
                _ensure_directive_enabled(name)
                if name in ranked:
                    ranked.remove(name)
                ranked.insert(0, name)

        weight_macro = _safe_weight(gpt.get("weight_macro"), w_macro)
        weight_scalp = _safe_scalp_weight(gpt.get("weight_scalp"), w_scalp)
        if strong_trend:
            weight_scalp = max(MIN_SCALP_WEIGHT, weight_scalp - 0.03)
        elif high_volatility and not strong_trend:
            weight_scalp = min(MAX_SCALP_WEIGHT, weight_scalp + 0.02)
        if event_soon:
            weight_scalp = min(weight_scalp, w_scalp)
        if weight_macro + weight_scalp > MACRO_SCALP_CAP:
            excess = (weight_macro + weight_scalp) - MACRO_SCALP_CAP
            reduce_macro = min(weight_macro - 0.05, excess * 0.7)
            if reduce_macro > 0:
                weight_macro -= reduce_macro
                excess -= reduce_macro
            if excess > 0:
                weight_scalp = max(MIN_SCALP_WEIGHT, weight_scalp - excess)
        micro_share = max(0.0, 1.0 - weight_macro - weight_scalp)
        if not event_soon and micro_share < 0.12 and weight_macro > 0.3:
            uplift = min(0.12 - micro_share, weight_macro * 0.2)
            micro_share += uplift
            weight_macro = max(0.0, weight_macro - uplift)
        # Derive human-friendly mode for dashboard using macro vs (macro+micro)
        denom = max(1e-6, weight_macro + micro_share)
        macro_ratio = weight_macro / denom
        if macro_ratio <= 0.4:
            mode = "micro"
        elif macro_ratio >= 0.6:
            mode = "macro"
        else:
            mode = "hybrid"

        # 7) Lot 配分と戦略実行（最初の成立戦略のみ）
        skip_reasons: list[dict] = []
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
            trades_per_day_cfg = config_params.get("TRADES_PER_DAY")
            try:
                trades_per_day = int(trades_per_day_cfg)
            except (TypeError, ValueError):
                trades_per_day = int(os.environ.get("TRADES_PER_DAY", "10"))
            risk_share_cfg = config_params.get("RISK_SHARE_OF_TARGET")
            try:
                risk_share = float(risk_share_cfg)
            except (TypeError, ValueError):
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

            if progress_pct < 0:
                scale = max(0.5, 1.0 + progress_pct * 2.5)
            elif progress_pct < daily_target_pct:
                remaining_ratio = (daily_target_pct - progress_pct) / daily_target_pct
                scale = 1.0 + 0.4 * max(min(remaining_ratio, 1.0), 0.0)
            else:
                scale = 0.6
            scale_cap = float(os.environ.get("RISK_SCALE_CAP", "1.2"))
            scale = min(scale, scale_cap)
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

            # 拡張: GPT 指示がリスク抑制を推奨する場合、ranked先頭のバイアスを適用
            top_strategy = ranked[0] if ranked else None
            strategy_bias = _risk_bias(top_strategy) if top_strategy else 1.0
            base_units = int(base_units * news_mult * spread_mult * strategy_bias)

            try:
                min_units_cfg = int(os.environ.get("MIN_UNITS", "0"))
            except ValueError:
                min_units_cfg = 0
            if min_units_cfg > 0 and units_by_margin > 0:
                forced_units = max(base_units, min_units_cfg)
                forced_units = min(forced_units, units_by_margin)
                if forced_units > base_units:
                    _log_event(
                        logging.INFO,
                        "base_units_floor_applied",
                        base_units=base_units,
                        forced_units=forced_units,
                        min_units=min_units_cfg,
                        units_by_risk=units_by_risk,
                        units_by_margin=units_by_margin,
                        risk_pct=risk_pct,
                        news_mult=news_mult,
                        spread_mult=spread_mult,
                        strategy_bias=strategy_bias,
                    )
                base_units = forced_units

            # Convert to lot fraction for allocator (1 lot = 100k units)
            lot_total = round(base_units / 100000.0, 3)
            lots = alloc(lot_total, weight_macro, weight_scalp)
            _inject_baseline_strategies(lots)

            # --- Cooldown / gating (never drop signals; scale exposure instead) ---
            soft_cd_min = int(os.environ.get("SOFT_COOLDOWN_MIN", "60"))
            soft_cd_factor = float(os.environ.get("SOFT_COOLDOWN_FACTOR", "0.25"))
            event_size_factor = float(os.environ.get("EVENT_SIZE_FACTOR", "0.15"))
            spread_soft_min = float(os.environ.get("SPREAD_SOFT_MIN", "1.0"))
            spread_hard_max = float(os.environ.get("SPREAD_HARD_MAX", "3.0"))
            daily_soft_stop_pct = float(os.environ.get("DAILY_SOFT_STOP_PCT", "-0.01"))  # -1%
            daily_soft_factor = float(os.environ.get("DAILY_SOFT_FACTOR", "0.10"))
            hard_cd_global = int(os.environ.get("HARD_COOLDOWN_MIN", "15"))
            hard_cd_macro = int(os.environ.get("HARD_COOLDOWN_MACRO_MIN", str(hard_cd_global)))
            hard_cd_micro = int(os.environ.get("HARD_COOLDOWN_MICRO_MIN", "5"))

            last_ot = st.get("last_order_time")
            last_ot_pocket = st.get("last_order_pocket")
            hard_cd_threshold = hard_cd_global
            if last_ot_pocket == "macro":
                hard_cd_threshold = hard_cd_macro
            elif last_ot_pocket == "micro":
                hard_cd_threshold = hard_cd_micro

            hard_cd_active = False
            try:
                if last_ot and (now - datetime.fromisoformat(last_ot)).total_seconds() < hard_cd_threshold * 60:
                    hard_cd_active = True
            except Exception:
                last_ot = None

            exposure_factor_base = 0.3 if hard_cd_active else 1.0
            if hard_cd_active:
                logging.info(
                    "[SKIP] Hard cooldown in effect",
                    extra={"hard_cd_min": hard_cd_threshold, "last_order_time": last_ot, "last_order_pocket": last_ot_pocket},
                )
                skip_reasons.append(
                    {
                        "strategy": None,
                        "reason": "hard_cooldown",
                        "cooldown_min": hard_cd_threshold,
                    }
                )
            # Event window: scale down size instead of forbidding
            if event_soon:
                exposure_factor_base *= max(min(event_size_factor, 1.0), 0.0)
            # Soft cooldown since last actual order
            try:
                if last_ot:
                    elapsed = (now - datetime.fromisoformat(last_ot)).total_seconds() / 60.0
                    if elapsed < soft_cd_min:
                        decay = max(0.5, 1.0 - (soft_cd_min - elapsed) / max(soft_cd_min, 1.0) * (1.0 - soft_cd_factor))
                        exposure_factor_base *= decay
                        logging.info(
                            "[GATE] soft cooldown",
                            extra={"elapsed_min": round(elapsed, 2), "factor": round(decay, 3)},
                        )
            except Exception:
                pass
            # Daily soft stop by PnL percent from start-of-day NAV
            try:
                if progress_pct <= daily_soft_stop_pct:
                    exposure_factor_base *= max(min(daily_soft_factor, 1.0), 0.0)
                elif progress_pct >= 0:
                    exposure_factor_base *= 1.05
            except Exception:
                pass
            # Spread-based linear decay between soft_min..hard_max
            try:
                if spread > spread_soft_min:
                    if spread >= spread_hard_max:
                        exposure_factor_base *= 0.0
                    else:
                        span = max(spread_hard_max - spread_soft_min, 0.0001)
                        decay = max(0.0, 1.0 - (spread - spread_soft_min) / span)
                        exposure_factor_base *= decay
            except Exception:
                pass
            decision = None
            decisions: list[str] = []
            filled_pockets: set[str] = set()
            micro_executed = False

            def _micro_fallback_signal() -> Dict[str, float] | None:
                if event_soon:
                    return None
                price_raw = fac_m1.get("close")
                ma20_raw = fac_m1.get("ma20")
                ma10_raw = fac_m1.get("ma10")
                atr_raw = fac_m1.get("atr")
                if price_raw is None:
                    return None
                try:
                    price_val = float(price_raw)
                    ma20_val = float(ma20_raw) if ma20_raw is not None else price_val
                    ma10_val = float(ma10_raw) if ma10_raw is not None else price_val
                    atr_val = float(atr_raw or 0.0)
                except (TypeError, ValueError):
                    return None

                atr_pips = max(atr_val * 100.0, 1.0)

                if micro_regime in ("Trend", "Breakout"):
                    action = "buy" if ma10_val >= ma20_val else "sell"
                elif micro_regime in ("Range", "Mixed"):
                    action = "buy" if price_val <= ma20_val else "sell"
                else:
                    velocity = float(fac_m1.get("tick_velocity_30s", 0.0) or 0.0)
                    action = "buy" if velocity >= 0 else "sell"

                sl_pips = max(6.0, atr_pips * 0.9)
                tp_pips = max(sl_pips * 1.45, sl_pips + 6.0)
                return {
                    "action": action,
                    "sl_pips": round(sl_pips, 1),
                    "tp_pips": round(tp_pips, 1),
                }
            STRATS = {
                "TrendMA": MovingAverageCross,
                "Donchian55": Donchian55,
                "BB_RSI": BBRsi,
                "NewsSpikeReversal": NewsSpikeReversal,
                "MicroTrendPullback": MicroTrendPullback,
            }
            hard_strat_cd_global = int(os.environ.get("HARD_STRATEGY_COOLDOWN_MIN", str(hard_cd_global)))
            hard_strat_cd_macro = int(os.environ.get("HARD_STRATEGY_COOLDOWN_MIN_MACRO", str(hard_strat_cd_global)))
            hard_strat_cd_micro = int(os.environ.get("HARD_STRATEGY_COOLDOWN_MIN_MICRO", str(hard_cd_micro)))
            last_by_strategy = st.get("last_order_by_strategy", {}) or {}
            last_by_pocket = st.get("last_order_by_pocket", {}) or {}
            adx_h4 = float(fac_h4.get("adx") or 0.0)
            adx_m1 = float(fac_m1.get("adx") or 0.0)
            slope_h4 = abs(float(fac_h4.get("ma_slope") or 0.0))

            for sname in ranked:
                cls = STRATS.get(sname)
                if not cls:
                    continue
                pocket = cls.pocket
                if pocket in filled_pockets:
                    logging.info(
                        "[SKIP] %s pocket %s already filled this request",
                        sname,
                        pocket,
                    )
                    skip_reasons.append(
                        {"strategy": sname, "reason": "pocket_filled", "pocket": pocket}
                    )
                    continue
                strategy_key = f"{sname}:{cls.pocket}"
                last_strat_ts = last_by_strategy.get(strategy_key)
                strategy_cd = hard_strat_cd_micro if cls.pocket == "micro" else hard_strat_cd_macro
                if last_strat_ts:
                    try:
                        if (now - datetime.fromisoformat(last_strat_ts)).total_seconds() < strategy_cd * 60:
                            logging.info(
                                "[SKIP] Strategy cooldown active",
                                extra={"strategy": sname, "pocket": cls.pocket, "cooldown_min": strategy_cd},
                            )
                            skip_reasons.append(
                                {
                                    "strategy": sname,
                                    "pocket": cls.pocket,
                                    "reason": "strategy_cooldown",
                                }
                            )
                            continue
                    except Exception:
                        pass
                # Event モード中も取引は許可（上でサイズを縮小）
                if sname == "BB_RSI":
                    max_adx_h4 = float(os.environ.get("BBRSI_MAX_ADX_H4", "42"))
                    max_adx_m1 = float(os.environ.get("BBRSI_MAX_ADX_M1", "38"))
                    max_slope_h4 = float(os.environ.get("BBRSI_MAX_SLOPE_H4", "0.09"))
                    suppress = False
                    trend_stack = macro_regime == "Trend" and micro_regime == "Trend"
                    if trend_stack and adx_h4 > max_adx_h4 and slope_h4 > max_slope_h4:
                        suppress = True
                    if adx_m1 > max_adx_m1 and trend_stack:
                        suppress = True
                    if suppress:
                        logging.info(
                            "[SKIP] BB_RSI suppressed by trend filter",
                            extra={
                                "macro_regime": macro_regime,
                                "micro_regime": micro_regime,
                                "adx_h4": adx_h4,
                                "adx_m1": adx_m1,
                                "slope_h4": slope_h4,
                            },
                        )
                        continue
                if sname == "NewsSpikeReversal":
                    sig = cls.check(fac_m1, news_cache.get("short", []))
                elif getattr(cls, "requires_h4", False) or cls.pocket == "macro":
                    sig = cls.check(fac_m1, fac_h4)
                else:
                    sig = cls.check(fac_m1)
                if not sig:
                    logging.info(
                        "[SKIP] Strategy returned no signal",
                        extra={"strategy": sname, "macro_regime": macro_regime},
                    )
                    skip_reasons.append(
                        {
                            "strategy": sname,
                            "reason": "no_signal",
                            "macro_regime": macro_regime,
                            "adx_h4": adx_h4,
                            "adx_m1": adx_m1,
                        }
                    )
                    continue
                pocket = cls.pocket
                if not can_trade(pocket):
                    logging.info(
                        "[SKIP] Pocket drawdown guard active",
                        extra={"strategy": sname, "pocket": pocket},
                    )
                    skip_reasons.append(
                        {"strategy": sname, "reason": "pocket_dd", "pocket": pocket}
                    )
                    continue
                lot = lots.get(pocket, 0)
                lot = round(lot * float(_risk_multiplier_from_stats(learning_stats, cls.name, pocket)), 3)
                # Apply soft gating exposure factor (never hard drop the signal here)
                lot = round(lot * exposure_factor_base, 4)
                if exposure_factor_base > 0 and lot_total > 0:
                    if pocket == "micro":
                        min_lot_cfg = float(os.environ.get("MIN_MICRO_LOT", "0.0"))
                    else:
                        min_lot_cfg = float(os.environ.get("MIN_MACRO_LOT", "0.0"))
                    if min_lot_cfg > 0:
                        lot = max(lot, min(min_lot_cfg, lot_total))
                if lot <= 0:
                    logging.info(
                        "[SKIP] lot allocation <= 0",
                        extra={"strategy": sname, "lot": lot, "pocket": pocket},
                    )
                    skip_reasons.append(
                        {"strategy": sname, "reason": "lot_zero", "pocket": pocket}
                    )
                    continue
                price = fac_m1.get("close")
                # 動的SL/TP（ATR倍率）対応
                pip = 0.01
                sl_pips = sig.get("sl_pips")
                tp_pips = sig.get("tp_pips")
                if sig.get("sl_atr_mult") or sig.get("tp_atr_mult"):
                    atr_src = fac_h4 if pocket == "macro" else fac_m1
                    atr_val = float(atr_src.get("atr", 0.0) or 0.0)
                    if sig.get("sl_atr_mult"):
                        sl_pips = float(atr_val * 100 * float(sig.get("sl_atr_mult", 0.0)))
                    if sig.get("tp_atr_mult"):
                        tp_pips = float(atr_val * 100 * float(sig.get("tp_atr_mult", 0.0)))
                if sl_pips is None or tp_pips is None:
                    sl_pips = 20.0
                    tp_pips = 20.0
                sl_cap = float(sig.get("sl_cap_pips", 0.0) or 0.0)
                sl_floor = float(sig.get("sl_floor_pips", 0.0) or 0.0)
                tp_cap = float(sig.get("tp_cap_pips", 0.0) or 0.0)
                tp_floor = float(sig.get("tp_floor_pips", 0.0) or 0.0)
                if sl_floor:
                    sl_pips = max(sl_pips, sl_floor)
                if sl_cap:
                    sl_pips = min(sl_pips, sl_cap)
                if tp_floor:
                    tp_pips = max(tp_pips, tp_floor)
                if tp_cap:
                    tp_pips = min(tp_pips, tp_cap)

                units = int(lot * 100000) * (1 if sig["action"] == "buy" else -1)
                if units == 0:
                    logging.info(
                        "[SKIP] units resolved to zero",
                        extra={"strategy": sname, "lot": lot},
                    )
                    skip_reasons.append(
                        {"strategy": sname, "reason": "units_zero", "lot": lot}
                    )
                    continue
                if sig["action"] == "buy":
                    sl, tp = clamp_sl_tp(price, price - sl_pips * pip, price + tp_pips * pip, True)
                else:
                    sl, tp = clamp_sl_tp(price, price + sl_pips * pip, price - tp_pips * pip, False)
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
                    "macro_regime": macro_regime,
                    "micro_regime": micro_regime,
                    "focus_tag": focus,
                    "weight_macro": weight_macro,
                    "weight_scalp": weight_scalp,
                    "mode": mode,
                    "adx_m1": cur_adx_m1,
                    "adx_h4": cur_adx_h4,
                    "bbw_m1": cur_bbw_m1,
                    "bbw_h4": cur_bbw_h4,
                    "atr_m1": float(fac_m1.get("atr") or 0.0),
                    "atr_h4": float(fac_h4.get("atr") or 0.0),
                    "session": _session_label(now),
                    "direction": "long" if units > 0 else "short",
                    "rsi_m1": rsi_m1,
                }
                order_result = asyncio.run(
                    market_order(
                        INSTRUMENT,
                        units,
                        sl,
                        tp,
                        pocket,
                        strategy=sname,
                        macro_regime=macro_regime,
                        micro_regime=micro_regime,
                    )
                )
                trade_req["trade_id"] = order_result.get("trade_id") or "FAIL"
                if order_result.get("order_id"):
                    trade_req["order_id"] = order_result.get("order_id")
                if not order_result.get("success"):
                    trade_req["order_error"] = order_result.get("error")
                    _log_trade(trade_req)
                    continue
                logging.info(
                    "[ORDER_PLACED] %s | %s | pocket=%s | lot=%s | units=%s | SL=%s TP=%s",
                    trade_req["trade_id"],
                    sname,
                    pocket,
                    lot,
                    units,
                    sl,
                    tp,
                )
                _log_trade(trade_req)
                # mark last_order_time for soft cooldown
                try:
                    if "last_order_by_strategy" not in st or not isinstance(st.get("last_order_by_strategy"), dict):
                        st["last_order_by_strategy"] = {}
                    if "last_order_by_pocket" not in st or not isinstance(st.get("last_order_by_pocket"), dict):
                        st["last_order_by_pocket"] = {}
                    st["last_order_time"] = now.isoformat(timespec="seconds")
                    st["last_order_strategy"] = sname
                    st["last_order_pocket"] = pocket
                    st["last_order_by_strategy"][strategy_key] = now.isoformat(timespec="seconds")
                    st["last_order_by_pocket"][pocket] = now.isoformat(timespec="seconds")
                    _save_state(st)
                except Exception:
                    pass
                decisions.append(f"ORDER:{trade_req['trade_id']}:{sname}:{pocket}:{lot}")
                if pocket == "micro":
                    micro_executed = True
                filled_pockets.add(pocket)

        if (
            not micro_executed
            and not hard_cd_active
            and can_trade("micro")
            and lots.get("micro", 0.0) >= MIN_BASELINE_POCKET_LOT
        ):
            fallback_sig = _micro_fallback_signal()
            if fallback_sig:
                lot = lots.get("micro", 0.0)
                lot = round(lot * float(_risk_multiplier_from_stats(learning_stats, "MicroTrendPullback", "micro")), 3)
                lot = round(lot * exposure_factor_base, 4)
                if lot > 0:
                    units = int(round(lot * 100000))
                    if units != 0:
                        pip = 0.01
                        price = fac_m1.get("close") or 0.0
                        if fallback_sig["action"] == "sell":
                            units = -abs(units)
                            sl, tp = clamp_sl_tp(price, price + fallback_sig["sl_pips"] * pip, price - fallback_sig["tp_pips"] * pip, False)
                        else:
                            units = abs(units)
                            sl, tp = clamp_sl_tp(price, price - fallback_sig["sl_pips"] * pip, price + fallback_sig["tp_pips"] * pip, True)
                        trade_req = {
                            "ts": now.isoformat(timespec="seconds"),
                            "strategy": "MicroFallback",
                            "pocket": "micro",
                            "instrument": INSTRUMENT,
                            "action": "BUY" if units > 0 else "SELL",
                            "units": units,
                            "price": price,
                            "sl": sl,
                            "tp": tp,
                            "lot_total": lot_total,
                            "news_mult": news_mult,
                            "spread_pips": spread,
                            "macro_regime": macro_regime,
                            "micro_regime": micro_regime,
                            "focus_tag": mode,
                            "weight_macro": weight_macro,
                            "mode": mode,
                            "adx_m1": cur_adx_m1,
                            "adx_h4": cur_adx_h4,
                            "bbw_m1": cur_bbw_m1,
                            "bbw_h4": cur_bbw_h4,
                            "atr_m1": float(fac_m1.get("atr") or 0.0),
                            "atr_h4": float(fac_h4.get("atr") or 0.0),
                            "session": _session_label(now),
                            "direction": "long" if units > 0 else "short",
                            "rsi_m1": rsi_m1,
                        }
                        result = asyncio.run(
                            market_order(
                                INSTRUMENT,
                                units,
                                sl,
                                tp,
                                "micro",
                                strategy="MicroFallback",
                                macro_regime=macro_regime,
                                micro_regime=micro_regime,
                            )
                        )
                        trade_req["trade_id"] = result.get("trade_id") or "FAIL"
                        if result.get("order_id"):
                            trade_req["order_id"] = result.get("order_id")
                        if not result.get("success"):
                            trade_req["order_error"] = result.get("error")
                            _log_trade(trade_req)
                        else:
                            logging.info(
                                "[ORDER_FALLBACK] %s | MicroFallback | pocket=micro | lot=%s | units=%s | SL=%s TP=%s",
                                trade_req["trade_id"],
                                lot,
                                units,
                                sl,
                                tp,
                            )
                            _log_trade(trade_req)
                            st.setdefault("last_order_by_strategy", {})["MicroFallback:micro"] = now.isoformat(timespec="seconds")
                            st.setdefault("last_order_by_pocket", {})["micro"] = now.isoformat(timespec="seconds")
                            st["last_order_time"] = now.isoformat(timespec="seconds")
                            st["last_order_strategy"] = "MicroFallback"
                            st["last_order_pocket"] = "micro"
                            _save_state(st)
                            decisions.append(f"ORDER:{trade_req['trade_id']}:MicroFallback:micro:{lot}")
                            filled_pockets.add("micro")
                            micro_executed = True

        if decisions:
            decision = ";".join(decisions)

        # 8) 状態保存
        if decision is None and hard_cd_active:
            decision = "HARD_COOLDOWN"
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
            # TTL until next likely GPT call (cooldown-based approximation)
            try:
                last_ts = st.get("last_gpt_call_time")
                last_dt = datetime.fromisoformat(last_ts) if last_ts else None
                elapsed = (now - last_dt).total_seconds() if last_dt else cooldown_min * 60
                ttl = max(int(cooldown_min * 60 - elapsed), 0)
            except Exception:
                ttl = None

            status_doc = _status_payload({
                "ts": now.isoformat(timespec="seconds"),
                "instrument": INSTRUMENT,
                "equity": st.get("equity"),
                "margin_avail": st.get("margin_avail"),
                "macro_regime": macro_regime,
                "micro_regime": micro_regime,
                "focus_tag": focus,
                "weight_macro": w_macro,
                "weight_micro": w_micro,
                "weight_scalp": w_scalp,
                "mode": mode,
                "selected_instruments": [INSTRUMENT],
                "diversify_slots": 1,
                "resource_ttl_sec": ttl,
                "event_soon": event_soon,
                "cooldown_min": cooldown_min,
                "gpt_ranked": ranked,
                "decision": decision,
                "lot_total": locals().get("lot_total"),
                "spread_pips": locals().get("spread"),
                "exposure_factor": locals().get("exposure_factor_base"),
                "risk_pct": locals().get("risk_pct"),
                "base_units": locals().get("base_units"),
                "news_mult": locals().get("news_mult"),
                "spread_mult": locals().get("spread_mult"),
                "config_risk_share": locals().get("risk_share"),
                "config_cooldown_min": cooldown_min,
                "news_avg_sent": news_cache.get("avg_sent"),
                "news_avg_imp": news_cache.get("avg_imp"),
                "factors_m1": {k: fac_m1.get(k) for k in ("adx","bbw","ma10","ma20","rsi","close")},
                "factors_h4": {k: fac_h4.get(k) for k in ("adx","bbw","ma10","ma20","rsi","close")},
                "can_trade_micro": bool(can_trade("micro")),
                "can_trade_macro": bool(can_trade("macro")),
                "global_dd_stop": bool(check_global_drawdown()),
                "day_start_nav": st.get("day_start_nav"),
                "skip_reasons": skip_reasons[:5] if skip_reasons else [],
                "learning_scores": learning_snapshot,
            })
            fs.collection("status").document("trader").set(status_doc)
        except Exception as _e:
            logging.warning(f"status_write_failed: {_e}")

        decision = decision or "NOOP"
        if decision == "NOOP" and skip_reasons:
            logging.info(
                "[NO_ORDER] decision=NOOP",
                extra={
                    "risk_pct": locals().get("risk_pct"),
                    "base_units": locals().get("base_units"),
                    "lot_total": locals().get("lot_total"),
                    "skip_reasons": skip_reasons,
                    "gpt_ranked": ranked,
                    "event_soon": event_soon,
                },
            )
        logging.info(f"trader_result={decision}")
        return decision, 200
    except Exception as e:
        logging.error(f"Unhandled trader error: {e}")
        return "ERROR", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
