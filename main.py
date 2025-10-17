import asyncio
import datetime
import logging
import math
import os
import time
import traceback
from collections import defaultdict
from functools import partial
from typing import Awaitable, Callable

from market_data.candle_fetcher import (
    Candle,
    start_candle_stream,
    initialize_history,
)
from indicators.factor_cache import all_factors, on_candle
from analysis.regime_classifier import classify
from analysis.focus_decider import FocusDecision, decide_focus
from analysis.gpt_decider import get_decision
from analysis.perf_monitor import snapshot as get_perf
from analysis.summary_ingestor import (
    check_event_soon,
    get_latest_news,
    ingest_loop as summary_ingest_loop,
    bucket as summary_bucket,
)
# バックグラウンドでニュース取得と要約を実行するためのインポート
from market_data.news_fetcher import (
    fetch_loop as news_fetch_loop,
    bucket as news_bucket,
)
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
from strategies.micro.trend_pullback import MicroTrendPullback
from analysis.learning import re_rank_strategies, risk_multiplier
from utils.market_hours import is_market_open
from execution.scalp_engine import scalp_loop

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
    "MicroTrendPullback": MicroTrendPullback,
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


FALLBACK_NOOP_STREAK = max(0, _env_int("FALLBACK_NOOP_STREAK", 3))
NOOP_TRACK_CAP = max(1, _env_int("NOOP_TRACK_CAP", 12))


POCKET_BY_STRATEGY = {
    name: getattr(cls, "pocket", "") for name, cls in STRATEGIES.items()
}

EQUITY = 10000.0  # ← 実際は REST から取得


STRONG_TREND_VELOCITY = float(os.getenv("STRONG_TREND_VELOCITY_30S", "8.0"))
STRONG_TREND_RANGE = float(os.getenv("STRONG_TREND_RANGE_30S", "12.0"))
STRONG_TREND_MACRO_SHARE = float(os.getenv("STRONG_TREND_MACRO_SHARE", "0.7"))
MIN_BASELINE_POCKET_LOT = float(os.getenv("MIN_BASELINE_POCKET_LOT", "0.0001"))

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


MIN_SCALP_WEIGHT = max(0.0, _env_float("MIN_SCALP_WEIGHT", 0.04))
MAX_SCALP_WEIGHT = max(MIN_SCALP_WEIGHT, _env_float("MAX_SCALP_WEIGHT", 0.35))
MACRO_SCALP_CAP = max(0.6, _env_float("MACRO_SCALP_SUM_CAP", 0.9))

SUPERVISOR_RESTART_DELAY_SEC = 5.0


def _pf_value(value) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _safe_weight(value, default: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = default
    if not math.isfinite(weight):
        weight = default
    return max(0.0, min(1.0, weight))


def _safe_scalp_weight(value, default: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = default
    if not math.isfinite(weight):
        weight = default
    return max(MIN_SCALP_WEIGHT, min(MAX_SCALP_WEIGHT, weight))


def _adjust_weight(weight: float, hints: dict) -> float:
    macro_floor = None
    micro_ceiling = None

    loss_recovery = hints.get("loss_recovery") if isinstance(hints, dict) else None
    if isinstance(loss_recovery, list):
        if "macro" in loss_recovery:
            macro_floor = max(macro_floor or 0.0, 0.55)
        if "micro" in loss_recovery:
            micro_ceiling = min(micro_ceiling or 1.0, 0.45)

    bias = hints.get("regime_bias") if isinstance(hints, dict) else None
    if bias == "macro_trend":
        macro_floor = max(macro_floor or 0.0, 0.6)
    elif bias in ("micro_breakout", "micro_trend"):
        micro_ceiling = min(micro_ceiling or 1.0, 0.4)
    elif bias == "range_reversion":
        macro_floor = max(macro_floor or 0.0, 0.4)
        micro_ceiling = min(micro_ceiling or 1.0, 0.6)

    if macro_floor is not None:
        weight = max(weight, macro_floor)
    if micro_ceiling is not None:
        weight = min(weight, micro_ceiling)

    return round(max(0.05, min(weight, 0.95)), 2)


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


def _build_fallback_seed(
    decision_hints: dict,
    *,
    need_macro: bool,
    need_micro: bool,
    need_event_micro: bool,
) -> list[str]:
    seed: list[str] = []
    bias = decision_hints.get("regime_bias") if isinstance(decision_hints, dict) else None

    if need_macro:
        macro_candidates = ["TrendMA", "Donchian55"]
        if bias == "micro_breakout":
            macro_candidates = ["Donchian55", "TrendMA"]
        for name in macro_candidates:
            if name not in seed:
                seed.append(name)

    if need_micro:
        micro_candidates = ["MicroTrendPullback", "BB_RSI"]
        if bias == "range_reversion":
            micro_candidates = ["BB_RSI", "MicroTrendPullback"]
        elif bias in ("micro_breakout", "micro_trend"):
            micro_candidates = ["MicroTrendPullback", "BB_RSI"]
        for name in micro_candidates:
            if name not in seed:
                seed.append(name)

    if need_event_micro and "NewsSpikeReversal" not in seed:
        seed.append("NewsSpikeReversal")

    return seed


def _ensure_directive_enabled(strategy_directives: dict, name: str, hints: dict) -> None:
    cfg = strategy_directives.setdefault(name, {"enabled": True, "risk_bias": 1.0})
    cfg["enabled"] = True
    loss_recovery = hints.get("loss_recovery") if isinstance(hints, dict) else []
    pocket = POCKET_BY_STRATEGY.get(name)
    try:
        current_bias = float(cfg.get("risk_bias", 1.0))
    except (TypeError, ValueError):
        current_bias = 1.0
    if isinstance(loss_recovery, list) and pocket in loss_recovery:
        current_bias = max(current_bias, 1.15)
    cfg["risk_bias"] = round(max(0.1, min(current_bias, 2.0)), 2)


async def _run_with_restart(
    name: str,
    coro_factory: Callable[[], Awaitable[None]],
    *,
    restart_delay: float = SUPERVISOR_RESTART_DELAY_SEC,
) -> None:
    attempt = 0
    while True:
        if attempt == 0:
            logging.info("[SUPERVISOR] starting %s", name)
        else:
            logging.warning(
                "[SUPERVISOR] restarting %s (attempt %d) in %.1fs",
                name,
                attempt + 1,
                restart_delay,
            )
            await asyncio.sleep(restart_delay)
        try:
            await coro_factory()
            logging.warning(
                "[SUPERVISOR] %s exited without error; restarting in %.1fs",
                name,
                restart_delay,
            )
        except asyncio.CancelledError:
            logging.info("[SUPERVISOR] %s cancelled; shutting down", name)
            raise
        except Exception:
            logging.exception(
                "[SUPERVISOR] %s crashed; restarting in %.1fs",
                name,
                restart_delay,
            )
        attempt += 1
        await asyncio.sleep(restart_delay)
async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)


async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)


async def logic_loop():
    pos_manager = PositionManager()
    perf_cache = {}
    news_cache = {}
    last_update_time = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    last_heartbeat_time = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
    market_pause_logged = False
    noop_streak = 0

    try:
        while True:
            now = datetime.datetime.now(datetime.timezone.utc)

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

            market_open, age_sec = is_market_open(fac_m1, now=now)
            if not market_open:
                if not market_pause_logged:
                    if age_sec is None:
                        logging.info(
                            "[PAUSE] Market inactive: missing recent M1 candle. Skipping GPT logic."
                        )
                    else:
                        logging.info(
                            "[PAUSE] Market inactive: last M1 candle %.1f minutes ago. Skipping GPT logic.",
                            age_sec / 60.0,
                        )
                    market_pause_logged = True
                pos_manager.sync_trades()
                wait_sec = 300 if (age_sec or 0) >= 3600 else 60
                await asyncio.sleep(wait_sec)
                continue
            elif market_pause_logged:
                logging.info("[PAUSE] Market activity detected. Resuming GPT logic.")
                market_pause_logged = False

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
            macro_pf_raw = (perf_cache.get("macro") or {}).get("pf") if isinstance(perf_cache, dict) else None
            micro_pf_raw = (perf_cache.get("micro") or {}).get("pf") if isinstance(perf_cache, dict) else None
            macro_pf = _pf_value(macro_pf_raw)
            micro_pf = _pf_value(micro_pf_raw)
            scalp_pf_raw = (perf_cache.get("scalp") or {}).get("pf") if isinstance(perf_cache, dict) else None
            scalp_pf = _pf_value(scalp_pf_raw)
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
                macro_pf=macro_pf,
                micro_pf=micro_pf,
                strong_trend=strong_trend,
                high_volatility=high_volatility,
            )
            focus = focus_decision.focus_tag
            w_macro = focus_decision.weight_macro
            w_scalp = focus_decision.weight_scalp

            # --- 2. GPT判断 ---
            decision_hints: dict[str, object] = {}
            regime_bias = _resolve_regime_bias(macro_regime, micro_regime, strong_trend)
            if regime_bias:
                decision_hints["regime_bias"] = regime_bias

            loss_recovery: list[str] = []
            if macro_pf is not None and macro_pf < 0.95:
                loss_recovery.append("macro")
            if micro_pf is not None and micro_pf < 0.95:
                loss_recovery.append("micro")
            if scalp_pf is not None and scalp_pf < 0.95:
                loss_recovery.append("scalp")
            if loss_recovery:
                decision_hints["loss_recovery"] = loss_recovery
            if strong_trend:
                decision_hints.setdefault("context_flags", []).append("strong_trend")
            if high_volatility and not strong_trend:
                decision_hints.setdefault("context_flags", []).append("high_volatility")
            if event_soon:
                decision_hints.setdefault("context_flags", []).append("event_mode")

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
                "focus_baseline": {
                    "focus": focus,
                    "weight_macro": w_macro,
                    "weight_scalp": w_scalp,
                    "weights": {
                        "macro": w_macro,
                        "micro": focus_decision.weight_micro,
                        "scalp": w_scalp,
                    },
                },
                "noop_streak": noop_streak,
            }
            if decision_hints:
                payload["decision_hints"] = decision_hints
            gpt = await get_decision(payload)
            weight_macro = _safe_weight(gpt.get("weight_macro"), w_macro)
            weight_macro = _adjust_weight(weight_macro, decision_hints)
            weight_scalp = _safe_scalp_weight(gpt.get("weight_scalp"), w_scalp)
            if "scalp" in loss_recovery:
                weight_scalp = min(MAX_SCALP_WEIGHT, max(weight_scalp, w_scalp + 0.02))
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
            all_known = (
                "TrendMA",
                "Donchian55",
                "BB_RSI",
                "NewsSpikeReversal",
                "MicroTrendPullback",
                "RangeBounce",
            )
            gpt_list = [
                s
                for s in gpt.get("ranked_strategies", [])
                if s in all_known and _is_enabled(s)
            ]
            ranked = re_rank_strategies(gpt_list, macro_regime, micro_regime) if gpt_list else []
            if not ranked:
                fallback_order = [s for s in all_known if _is_enabled(s)]
                ranked = fallback_order

            if strong_trend and ranked:
                priority = [s for s in ("TrendMA", "Donchian55", "MicroTrendPullback") if s in ranked]
                ranked = priority + [s for s in ranked if s not in priority]

            original_ranked = list(ranked)

            # --- 3. 発注準備 ---
            lot_total = allowed_lot(EQUITY, sl_pips=20)  # sl_pipsは仮

            def compute_lots(weight_macro_value: float, weight_scalp_value: float) -> dict[str, float]:
                allocated = alloc(lot_total, weight_macro_value, weight_scalp_value)
                scalp_lot = allocated.get("scalp", 0.0)
                if strong_trend and scalp_lot > 0.0:
                    keep = round(scalp_lot * 0.4, 3)
                    realloc = round(max(scalp_lot - keep, 0.0), 3)
                    if realloc > 0:
                        macro_share = max(0.0, min(1.0, STRONG_TREND_MACRO_SHARE))
                        allocated["scalp"] = keep
                        allocated["macro"] = round(
                            allocated.get("macro", 0.0) + realloc * macro_share,
                            3,
                        )
                        allocated["micro"] = round(
                            allocated.get("micro", 0.0) + realloc * (1.0 - macro_share),
                            3,
                        )
                        logging.info(
                            "[REALLOC] strong trend detected (vel=%.1f, range=%.1f). partial scalp realloc=%.3f",
                            tick_velocity,
                            tick_range,
                            realloc,
                        )
                return allocated

            lots = compute_lots(weight_macro, weight_scalp)

            def _inject_baseline_strategies() -> None:
                priority: list[str] = []
                if lots.get("macro", 0.0) >= MIN_BASELINE_POCKET_LOT:
                    priority.append("TrendMA")
                if lots.get("micro", 0.0) >= MIN_BASELINE_POCKET_LOT:
                    for candidate in ("MicroTrendPullback", "BB_RSI", "RangeBounce"):
                        if candidate in STRATEGIES:
                            priority.append(candidate)
                if not priority:
                    return
                logging.info(
                    "[BASELINE] ensure strategies=%s lots=%s", priority, lots
                )
                for name in reversed(priority):
                    if name not in STRATEGIES:
                        continue
                    _ensure_directive_enabled(strategy_directives, name, decision_hints)
                    if name in ranked:
                        ranked.remove(name)
                    ranked.insert(0, name)

            _inject_baseline_strategies()

            has_macro = any(POCKET_BY_STRATEGY.get(s) == "macro" for s in ranked)
            has_micro = any(POCKET_BY_STRATEGY.get(s) == "micro" for s in ranked)
            has_event_micro = "NewsSpikeReversal" in ranked

            need_macro = lots.get("macro", 0.0) > 0.0 and not has_macro
            need_micro = (
                lots.get("micro", 0.0) > 0.0
                and not has_micro
                and not event_soon
            )
            need_event_micro = (
                event_soon
                and lots.get("micro", 0.0) > 0.0
                and not has_event_micro
            )

            fallback_reasons: list[str] = []
            if not ranked:
                fallback_reasons.append("empty_ranked")
            if FALLBACK_NOOP_STREAK and noop_streak >= FALLBACK_NOOP_STREAK:
                fallback_reasons.append(f"noop_streak_{noop_streak}")
            if need_macro:
                fallback_reasons.append("missing_macro")
            if need_micro:
                fallback_reasons.append("missing_micro")
            if need_event_micro:
                fallback_reasons.append("missing_event_micro")

            fallback_need_macro = need_macro
            fallback_need_micro = need_micro
            if (
                FALLBACK_NOOP_STREAK
                and noop_streak >= FALLBACK_NOOP_STREAK
                and lot_total > 0.0
            ):
                fallback_need_macro = fallback_need_macro or lots.get("macro", 0.0) > 0.0
                fallback_need_micro = fallback_need_micro or (
                    lots.get("micro", 0.0) > 0.0 and not event_soon
                )

            if fallback_reasons and (fallback_need_macro or fallback_need_micro or need_event_micro):
                fallback_macro_weight = weight_macro
                if fallback_need_macro:
                    fallback_macro_weight = max(fallback_macro_weight, 0.5)
                if fallback_need_micro and not event_soon:
                    fallback_macro_weight = min(fallback_macro_weight, 0.5)
                if fallback_macro_weight != weight_macro:
                    weight_macro = fallback_macro_weight
                    lots = compute_lots(weight_macro, weight_scalp)

                seed = _build_fallback_seed(
                    decision_hints,
                    need_macro=fallback_need_macro,
                    need_micro=fallback_need_micro,
                    need_event_micro=need_event_micro,
                )
                fallback_ranked = re_rank_strategies(seed, macro_regime, micro_regime) if seed else []
                fallback_ranked = fallback_ranked or seed
                remainder = [s for s in original_ranked if s not in fallback_ranked]
                if fallback_ranked:
                    ranked = fallback_ranked + [s for s in remainder if s not in fallback_ranked]
                    for name in fallback_ranked:
                        _ensure_directive_enabled(strategy_directives, name, decision_hints)
                logging.warning(
                    "[FALLBACK] injecting strategies reason=%s lot_total=%s weights=(macro=%.2f, scalp=%.2f) ranked=%s",
                    fallback_reasons,
                    lot_total,
                    weight_macro,
                    weight_scalp,
                    ranked,
                )

            trade_allowed: dict[str, bool] = {}
            for pocket_name in ("micro", "macro", "scalp"):
                try:
                    trade_allowed[pocket_name] = can_trade(pocket_name)
                except Exception:
                    trade_allowed[pocket_name] = True

            if not trade_allowed.get("micro", True) and lots.get("micro", 0.0):
                lots["macro"] = round(lots.get("macro", 0.0) + lots.get("micro", 0.0), 3)
                lots["micro"] = 0.0

            if not trade_allowed.get("scalp", True) and lots.get("scalp", 0.0):
                lots["macro"] = round(lots.get("macro", 0.0) + lots.get("scalp", 0.0), 3)
                lots["scalp"] = 0.0

            # --- 4. 戦略実行ループ ---
            # ヘルパー: ATRベース/ピップスベースの SL/TP を計算
            def _calc_sl_tp_from_signal(sig: dict, action: str, price: float, pocket: str) -> tuple[float, float]:
                pip = 0.01
                sl_pips = None
                tp_pips = None
                if "sl_pips" in sig and "tp_pips" in sig:
                    sl_pips = float(sig["sl_pips"])
                    tp_pips = float(sig["tp_pips"])
                elif "sl_atr_mult" in sig and "tp_atr_mult" in sig:
                    use_h4 = pocket == "macro"
                    atr_src = fac_h4 if use_h4 and fac_h4 else fac_m1
                    atr = float(atr_src.get("atr", 0.0) or 0.0)
                    sl_pips = max(0.0, float(sig.get("sl_atr_mult", 0.0)) * atr * 100)
                    tp_pips = max(0.0, float(sig.get("tp_atr_mult", 0.0)) * atr * 100)
                else:
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

                if action == "buy":
                    sl = price - sl_pips * pip
                    tp = price + tp_pips * pip
                    return clamp_sl_tp(price, sl, tp, True)
                else:
                    sl = price + sl_pips * pip
                    tp = price - tp_pips * pip
                    return clamp_sl_tp(price, sl, tp, False)

            def _micro_fallback_signal() -> dict | None:
                if event_soon:
                    return None
                price_raw = fac_m1.get("close")
                ma20_raw = fac_m1.get("ma20")
                ma10_raw = fac_m1.get("ma10")
                atr_raw = fac_m1.get("atr")
                if price_raw is None:
                    return None
                try:
                    price = float(price_raw)
                    ma20 = float(ma20_raw) if ma20_raw is not None else price
                    ma10 = float(ma10_raw) if ma10_raw is not None else price
                    atr_val = float(atr_raw or 0.0)
                except (TypeError, ValueError):
                    return None

                atr_pips = max(atr_val * 100.0, 1.0)

                if micro_regime in ("Trend", "Breakout"):
                    action = "buy" if ma10 >= ma20 else "sell"
                elif micro_regime in ("Range", "Mixed"):
                    action = "buy" if price <= ma20 else "sell"
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

            pocket_limits = {"scalp": 1, "micro": 2, "macro": 2}
            pocket_counts: defaultdict[str, int] = defaultdict(int)
            filled_pockets: set[str] = set()
            orders_executed = False
            for sname in ranked:
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                if not _is_enabled(sname):
                    logging.info(f"[SKIP] {sname} disabled by GPT directive")
                    continue

                pocket = cls.pocket
                if pocket in filled_pockets:
                    logging.info(f"[SKIP] {sname} pocket {pocket} already filled this cycle")
                    continue

                # 戦略ごとに必要な入力を渡す
                if sname == "NewsSpikeReversal":
                    sig = cls.check(fac_m1, news_cache.get("short", []))
                elif sname == "BB_RSI":
                    sig = cls.check(fac_m1, fac_h4)
                elif getattr(cls, "requires_h4", False):
                    sig = cls.check(fac_m1, fac_h4)
                elif sname == "RangeBounce":
                    sig = cls.check(fac_m1, fac_h4, micro_regime, macro_regime)
                else:  # default signature
                    sig = cls.check(fac_m1)

                if not sig:
                    continue

                # Event モード中は micro を原則禁止。ただし NewsSpikeReversal は例外で許可。
                if event_soon and pocket == "micro" and sname != "NewsSpikeReversal":
                    logging.info("[SKIP] Event soon, skipping non-news micro trade.")
                    continue

                if not trade_allowed.get(pocket, True):
                    logging.info(f"[SKIP] Guard active for {pocket} pocket.")
                    continue

                limit = pocket_limits.get(pocket, 1)
                if pocket_counts[pocket] >= limit:
                    logging.info(
                        "[SKIP] pocket limit reached pocket=%s count=%s limit=%s",
                        pocket,
                        pocket_counts[pocket],
                        limit,
                    )
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
                        "[ORDER_PLACED] %s | %s | pocket=%s | lot=%s | units=%s | SL=%s TP=%s",
                        order_result.get("trade_id"),
                        cls.name,
                        pocket,
                        lot,
                        units,
                        sl,
                        tp,
                    )
                    logging.info(
                        "[ORDER] %s | %s | %s lot | SL=%s TP=%s",
                        order_result.get("trade_id"),
                        cls.name,
                        lot,
                        sl,
                        tp,
                    )
                    pocket_counts[pocket] += 1
                    filled_pockets.add(pocket)
                    orders_executed = True
                else:
                    logging.error(
                        "[ORDER FAILED] %s | error=%s",
                        cls.name,
                        order_result.get("error"),
                    )

            if (
                not pocket_counts.get("micro")
                and trade_allowed.get("micro", True)
                and lots.get("micro", 0.0) >= MIN_BASELINE_POCKET_LOT
            ):
                micro_candidates = [
                    s for s in ranked if POCKET_BY_STRATEGY.get(s) == "micro"
                ]
                logging.info(
                    "[MICRO_FALLBACK_CHECK] allowed=%s lot=%.4f candidates=%s",
                    trade_allowed.get("micro", True),
                    lots.get("micro", 0.0),
                    micro_candidates[:3],
                )
                fallback_sig = _micro_fallback_signal()
                if fallback_sig:
                    lot = lots.get("micro", 0.0)
                    try:
                        lot = round(lot * float(risk_multiplier("micro", "MicroTrendPullback")), 3)
                    except Exception:
                        lot = round(lot, 3)
                    directive_bias = _risk_bias("MicroTrendPullback")
                    if directive_bias != 1.0:
                        lot = round(lot * directive_bias, 3)
                    if lot > 0:
                        units = int(round(lot * 100000))
                        if units != 0:
                            action = fallback_sig["action"]
                            price = float(fac_m1.get("close"))
                            sl, tp = _calc_sl_tp_from_signal(
                                fallback_sig,
                                action,
                                price,
                                "micro",
                            )
                            if action == "sell" and units > 0:
                                units = -units
                            elif action == "buy" and units < 0:
                                units = -units
                            result = await market_order(
                                "USD_JPY",
                                units,
                                sl,
                                tp,
                                "micro",
                                strategy="MicroFallback",
                                macro_regime=macro_regime,
                                micro_regime=micro_regime,
                            )
                            if result.get("success"):
                                logging.info(
                                    "[ORDER_FALLBACK] %s | MicroFallback | pocket=micro | lot=%s | units=%s | SL=%s TP=%s",
                                    result.get("trade_id"),
                                    lot,
                                    units,
                                    sl,
                                    tp,
                                )
                                pocket_counts["micro"] += 1
                                filled_pockets.add("micro")
                                orders_executed = True
                            else:
                                logging.warning(
                                    "[ORDER_FALLBACK_FAILED] micro | error=%s",
                                    result.get("error"),
                                )
                else:
                    logging.info(
                        "[MICRO_FALLBACK_BYPASS] reason=no_signal event_mode=%s",
                        event_soon,
                    )

            if orders_executed:
                if noop_streak:
                    logging.info(
                        "[NOOP] streak reset after trade (previous=%d)", noop_streak
                    )
                noop_streak = 0
            else:
                noop_streak = min(NOOP_TRACK_CAP, noop_streak + 1)
                if noop_streak == 1 or (
                    FALLBACK_NOOP_STREAK
                    and noop_streak >= FALLBACK_NOOP_STREAK
                ):
                    level = (
                        logging.WARNING
                        if FALLBACK_NOOP_STREAK
                        and noop_streak >= FALLBACK_NOOP_STREAK
                        else logging.INFO
                    )
                    logging.log(
                        level,
                        "[NOOP] streak=%d (threshold=%s)",
                        noop_streak,
                        FALLBACK_NOOP_STREAK or "disabled",
                    )

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
    tasks: list[asyncio.Task[None]] = []

    candle_factory: Callable[[], Awaitable[None]] = partial(
        start_candle_stream,
        "USD_JPY",
        handlers,
    )
    tasks.append(
        asyncio.create_task(_run_with_restart("candle_stream", candle_factory))
    )
    tasks.append(asyncio.create_task(_run_with_restart("logic_loop", logic_loop)))
    tasks.append(asyncio.create_task(_run_with_restart("scalp_loop", scalp_loop)))
    tasks.append(asyncio.create_task(_run_with_restart("exit_loop", exit_loop)))
    tasks.append(asyncio.create_task(_run_with_restart("kaizen_loop", kaizen_loop)))
    tasks.append(
        asyncio.create_task(_run_with_restart("news_fetch_loop", news_fetch_loop))
    )

    if not news_bucket:
        logging.info(
            "[SUPERVISOR] news_fetch_loop running in dry mode (no bucket configured)"
        )

    if summary_bucket:
        tasks.append(
            asyncio.create_task(
                _run_with_restart("summary_ingest_loop", summary_ingest_loop)
            )
        )
    else:
        logging.info(
            "[SUPERVISOR] summary_ingest_loop disabled (no bucket configured)"
        )

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logging.info("[MAIN] KeyboardInterrupt received. Shutting down.")
            break
        except Exception:
            logging.exception("[MAIN] Fatal error; restarting in 10s")
            time.sleep(10)
        else:
            logging.warning("[MAIN] main() exited unexpectedly; restarting in 10s")
            time.sleep(10)
