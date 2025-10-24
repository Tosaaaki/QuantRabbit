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
from market_data import spread_monitor
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
from analysis.range_guard import detect_range_mode
from signals.pocket_allocator import alloc, DEFAULT_SCALP_SHARE, dynamic_scalp_share
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    check_global_drawdown,
    update_dd_context,
)
from execution.order_manager import (
    market_order,
    close_trade,
    update_dynamic_protections,
    plan_partial_reductions,
)
from execution.exit_manager import ExitManager
from execution.position_manager import PositionManager
from execution.stage_tracker import StageTracker
from analytics.realtime_metrics_client import (
    ConfidencePolicy,
    RealtimeMetricsClient,
    StrategyHealth,
)
from strategies.trend.ma_cross import MovingAverageCross
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.news.spike_reversal import NewsSpikeReversal
from strategies.scalping.m1_scalper import M1Scalper
from strategies.scalping.range_fader import RangeFader
from strategies.scalping.pulse_break import PulseBreak
from utils.oanda_account import get_account_snapshot
from utils.secrets import get_secret

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
    "RangeFader": RangeFader,
    "PulseBreak": PulseBreak,
}

POCKET_STRATEGY_MAP = {
    "macro": {"TrendMA", "Donchian55"},
    "micro": {"BB_RSI", "NewsSpikeReversal"},
    "scalp": {"M1Scalper", "RangeFader", "PulseBreak"},
}

FOCUS_POCKETS = {
    "macro": ("macro",),
    "micro": ("micro", "scalp"),
    "hybrid": ("macro", "micro", "scalp"),
    "event": ("macro", "micro"),
}

POCKET_EXIT_COOLDOWNS = {
    "macro": 540,
    "micro": 240,
    "scalp": 180,
}

POCKET_LOSS_COOLDOWNS = {
    "macro": 960,
    "micro": 600,
    "scalp": 360,
}

# 新規エントリー後のクールダウン（再エントリー抑制）
POCKET_ENTRY_MIN_INTERVAL = {
    "macro": 180,  # 3分
    "micro": 120,  # 2分
    "scalp": 60,
}

FALLBACK_EQUITY = 10000.0  # REST失敗時のフォールバック

STAGE_RATIOS = {
    # Fractions per stage sum to 1.0 so pocket lot can fully deploy as conditions allow.
    "macro": (0.2, 0.18, 0.16, 0.14, 0.12, 0.08, 0.06, 0.06),
    "micro": (0.22, 0.18, 0.16, 0.14, 0.12, 0.08, 0.05, 0.05),
    "scalp": (0.5, 0.3, 0.15, 0.05),
}

MIN_SCALP_STAGE_LOT = 0.01  # 1000 units baseline so micro/macro bias does not null scalps
DEFAULT_COOLDOWN_SECONDS = 180
RANGE_COOLDOWN_SECONDS = 420
# In range mode, allow mean‑reversion and light scalping entries
ALLOWED_RANGE_STRATEGIES = {"BB_RSI", "RangeFader", "M1Scalper"}
SOFT_RANGE_SUPPRESS_STRATEGIES = {"TrendMA", "Donchian55"}
LOW_TREND_ADX_THRESHOLD = 18.0
LOW_TREND_SLOPE_THRESHOLD = 0.00035
LOW_TREND_WEIGHT_CAP = 0.35
SOFT_RANGE_SCORE_MIN = 0.58
SOFT_RANGE_COMPRESSION_MIN = 0.55
SOFT_RANGE_VOL_MIN = 0.40
SOFT_RANGE_WEIGHT_CAP = 0.32
SOFT_RANGE_ADX_BUFFER = 6.0
RANGE_ENTRY_CONFIRMATIONS = 2
RANGE_EXIT_CONFIRMATIONS = 3
RANGE_MIN_ACTIVE_SECONDS = 240
RANGE_ENTRY_SCORE_FLOOR = 0.62
RANGE_EXIT_SCORE_CEIL = 0.56
STAGE_RESET_GRACE_SECONDS = 180

try:
    _BASE_RISK_PCT = float(get_secret("risk_pct"))
    if _BASE_RISK_PCT <= 0:
        _BASE_RISK_PCT = 0.02
except Exception:
    _BASE_RISK_PCT = 0.02
try:
    _MAX_RISK_PCT = float(get_secret("risk_pct_max"))
    if _MAX_RISK_PCT < _BASE_RISK_PCT:
        _MAX_RISK_PCT = _BASE_RISK_PCT
    elif _MAX_RISK_PCT > 0.3:
        _MAX_RISK_PCT = 0.3
except Exception:
    _MAX_RISK_PCT = _BASE_RISK_PCT


def _dynamic_risk_pct(signals: list[dict], range_mode: bool, weight_macro: float | None) -> float:
    if range_mode or not signals or _MAX_RISK_PCT <= _BASE_RISK_PCT:
        return _BASE_RISK_PCT
    actionable = [
        s
        for s in signals
        if s.get("action") in {"OPEN_LONG", "OPEN_SHORT"} and s.get("pocket")
    ]
    if not actionable:
        return _BASE_RISK_PCT
    pocket_factor = min(len({s["pocket"] for s in actionable}) / 3.0, 1.0)
    avg_conf = sum(max(0, min(100, s.get("confidence", 0))) for s in actionable)
    avg_conf = (avg_conf / len(actionable)) / 100.0
    bias = 0.0
    if isinstance(weight_macro, (int, float)):
        bias = min(1.0, abs(weight_macro - 0.5) * 2.0)
    score = 0.45 * pocket_factor + 0.45 * avg_conf + 0.1 * bias
    score = max(0.0, min(score, 1.0))
    return _BASE_RISK_PCT + (_MAX_RISK_PCT - _BASE_RISK_PCT) * score


def build_client_order_id(focus_tag: Optional[str], strategy_tag: str) -> str:
    ts_ms = int(time.time() * 1000)
    focus_part = (focus_tag or "hybrid")[:6]
    clean_tag = "".join(ch for ch in strategy_tag if ch.isalnum())[:9] or "sig"
    return f"qr-{ts_ms}-{focus_part}-{clean_tag}"


def _cooldown_for_pocket(pocket: str, range_mode: bool) -> int:
    base = POCKET_EXIT_COOLDOWNS.get(pocket, DEFAULT_COOLDOWN_SECONDS)
    if range_mode:
        base = max(base, RANGE_COOLDOWN_SECONDS)
    return base


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
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        ma10_m1 = fac_m1.get("ma10")
        ma20_m1 = fac_m1.get("ma20")
        ema20_m1 = fac_m1.get("ema20") or ma20_m1
        close_m1 = fac_m1.get("close")

        if action == "OPEN_LONG":
            if ma10_h4 is not None and ma20_h4 is not None and ma10_h4 < ma20_h4:
                logging.info(
                    "[STAGE] Macro buy gating: H4 trend down (ma10 %.3f < ma20 %.3f).",
                    ma10_h4,
                    ma20_h4,
                )
                return False
            if (
                close_m1 is not None
                and ema20_m1 is not None
                and close_m1 < ema20_m1 - 0.005
            ):
                logging.info(
                    "[STAGE] Macro buy gating: M1 close %.3f below ema20 %.3f.",
                    close_m1,
                    ema20_m1,
                )
                return False
            if ma10_m1 is not None and ma20_m1 is not None and ma10_m1 < ma20_m1:
                logging.info(
                    "[STAGE] Macro buy gating: M1 ma10 %.3f < ma20 %.3f.",
                    ma10_m1,
                    ma20_m1,
                )
                return False

        if action == "OPEN_SHORT":
            if ma10_h4 is not None and ma20_h4 is not None and ma10_h4 > ma20_h4:
                logging.info(
                    "[STAGE] Macro sell gating: H4 trend up (ma10 %.3f > ma20 %.3f).",
                    ma10_h4,
                    ma20_h4,
                )
                return False
            if (
                close_m1 is not None
                and ema20_m1 is not None
                and close_m1 > ema20_m1 + 0.005
            ):
                logging.info(
                    "[STAGE] Macro sell gating: M1 close %.3f above ema20 %.3f.",
                    close_m1,
                    ema20_m1,
                )
                return False
            if ma10_m1 is not None and ma20_m1 is not None and ma10_m1 > ma20_m1:
                logging.info(
                    "[STAGE] Macro sell gating: M1 ma10 %.3f > ma20 %.3f.",
                    ma10_m1,
                    ma20_m1,
                )
                return False

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
) -> tuple[float, int]:
    """段階的エントリーの次ロットとステージ番号を返す。"""
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
                return 0.0, stage_idx
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
            return round(next_lot, 4), stage_idx

    logging.info(
        "[STAGE] %s pocket already filled %.3f / %.3f lots. No additional entry.",
        pocket,
        round(current_lot, 4),
        round(total_lot, 4),
    )
    return 0.0, -1


async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)


async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)


async def logic_loop():
    pos_manager = PositionManager()
    metrics_client = RealtimeMetricsClient()
    confidence_policy = ConfidencePolicy()
    exit_manager = ExitManager()
    stage_tracker = StageTracker()
    perf_cache = {}
    news_cache = {}
    insight = InsightClient()
    last_update_time = datetime.datetime.min
    last_heartbeat_time = datetime.datetime.min  # Add this line
    last_metrics_refresh = datetime.datetime.min
    strategy_health_cache: dict[str, StrategyHealth] = {}
    range_active = False
    range_soft_active = False
    last_range_reason = ""
    range_state_since = datetime.datetime.min
    range_entry_counter = 0
    range_exit_counter = 0
    raw_range_active = False
    raw_range_reason = ""
    stage_empty_since: dict[tuple[str, str], datetime.datetime] = {}
    last_risk_pct: float | None = None
    last_spread_gate = False
    last_spread_gate_reason = ""

    try:
        while True:
            now = datetime.datetime.utcnow()
            stage_tracker.clear_expired(now)
            stage_tracker.update_loss_streaks(now=now, cooldown_map=POCKET_LOSS_COOLDOWNS)

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

            spread_blocked, spread_remain, spread_snapshot, spread_reason = spread_monitor.is_blocked()
            spread_gate_reason = ""
            if spread_blocked:
                remain_txt = f"{spread_remain}s"
                base_reason = spread_reason or "spread_threshold"
                spread_gate_reason = f"{base_reason} (remain {remain_txt})"
            elif spread_snapshot:
                if spread_snapshot["age_ms"] > spread_snapshot["max_age_ms"]:
                    spread_gate_reason = (
                        f"spread_stale age={spread_snapshot['age_ms']}ms "
                        f"> {spread_snapshot['max_age_ms']}ms"
                    )
                elif spread_snapshot["spread_pips"] >= spread_snapshot["limit_pips"]:
                    spread_gate_reason = (
                        f"spread_hot {spread_snapshot['spread_pips']:.2f}p "
                        f">= {spread_snapshot['limit_pips']:.2f}p"
                    )
            spread_gate_active = bool(spread_gate_reason)
            if spread_snapshot:
                def _fmt(v: object) -> str:
                    return f"{float(v):.2f}" if isinstance(v, (int, float)) else "NA"

                last_txt = _fmt(spread_snapshot.get("spread_pips"))
                avg_txt = _fmt(spread_snapshot.get("avg_pips"))
                age_ms = spread_snapshot.get("age_ms")
                if spread_snapshot.get("baseline_ready"):
                    baseline_avg = spread_snapshot.get("baseline_avg_pips")
                    baseline_p95 = spread_snapshot.get("baseline_p95_pips")
                    baseline_txt = (
                        f"baseline≈{_fmt(baseline_avg)}p p95={_fmt(baseline_p95)}p"
                        if baseline_avg is not None and baseline_p95 is not None
                        else "baseline_ready"
                    )
                else:
                    baseline_txt = (
                        f"baseline_warmup samples={spread_snapshot.get('baseline_samples', 0)}"
                    )
                spread_log_context = (
                    f"last={last_txt}p avg={avg_txt}p age={age_ms}ms {baseline_txt}"
                )
            else:
                spread_log_context = "no_snapshot"
            if spread_gate_active:
                if (
                    not last_spread_gate
                    or spread_gate_reason != last_spread_gate_reason
                ):
                    logging.info(
                        "[SPREAD] gating entries (%s, %s)",
                        spread_gate_reason,
                        spread_log_context,
                    )
            elif last_spread_gate:
                logging.info("[SPREAD] entries re-enabled (%s)", spread_log_context)
            last_spread_gate = spread_gate_active
            last_spread_gate_reason = spread_gate_reason

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
            soft_range_just_activated = False
            range_ctx = detect_range_mode(fac_m1, fac_h4)
            if range_ctx.active != raw_range_active or range_ctx.reason != raw_range_reason:
                logging.info(
                    "[RANGE] detected active=%s reason=%s score=%.2f metrics=%s",
                    range_ctx.active,
                    range_ctx.reason,
                    range_ctx.score,
                    range_ctx.metrics,
                )
            raw_range_active = range_ctx.active
            raw_range_reason = range_ctx.reason
            metrics = range_ctx.metrics or {}
            compression_ratio = float(metrics.get("compression_ratio", 0.0) or 0.0)
            volatility_ratio = float(metrics.get("volatility_ratio", 0.0) or 0.0)
            effective_adx_m1 = float(
                metrics.get("effective_adx_m1", fac_m1.get("adx", 0.0) or 0.0)
            )
            adx_threshold = float(metrics.get("adx_threshold", 22.0) or 22.0)
            soft_range_prev = range_soft_active
            soft_range_candidate = (
                not raw_range_active
                and range_ctx.score >= SOFT_RANGE_SCORE_MIN
                and compression_ratio >= SOFT_RANGE_COMPRESSION_MIN
                and volatility_ratio >= SOFT_RANGE_VOL_MIN
                and effective_adx_m1 <= (adx_threshold + SOFT_RANGE_ADX_BUFFER)
            )
            if soft_range_candidate != soft_range_prev and not raw_range_active:
                logging.info(
                    "[RANGE] soft=%s score=%.2f eff_adx=%.2f comp=%.2f vol=%.2f",
                    soft_range_candidate,
                    range_ctx.score,
                    effective_adx_m1,
                    compression_ratio,
                    volatility_ratio,
                )
            range_soft_active = (
                soft_range_candidate if not raw_range_active else False
            )
            soft_range_just_activated = (
                range_soft_active and not soft_range_prev and not raw_range_active
            )
            entry_ready = raw_range_active
            exit_ready = (not raw_range_active) and (range_ctx.score <= RANGE_EXIT_SCORE_CEIL)
            if entry_ready:
                range_entry_counter += 1
            else:
                range_entry_counter = 0
            if exit_ready:
                range_exit_counter += 1
            else:
                range_exit_counter = 0
            prev_range_state = range_active
            if (
                not range_active
                and entry_ready
                and range_entry_counter >= RANGE_ENTRY_CONFIRMATIONS
            ):
                range_active = True
                range_state_since = now
                last_range_reason = range_ctx.reason
                range_exit_counter = 0
                logging.info(
                    "[RANGE] latched active (score=%.2f reason=%s)",
                    range_ctx.score,
                    range_ctx.reason,
                )
            elif range_active:
                held = (now - range_state_since).total_seconds() >= RANGE_MIN_ACTIVE_SECONDS
                if exit_ready and held and range_exit_counter >= RANGE_EXIT_CONFIRMATIONS:
                    range_active = False
                    range_state_since = now
                    range_entry_counter = 0
                    last_range_reason = range_ctx.reason
                    logging.info(
                        "[RANGE] released (score=%.2f reason=%s)",
                        range_ctx.score,
                        range_ctx.reason,
                    )
            if prev_range_state and not range_active:
                range_exit_counter = 0
            elif not prev_range_state and range_active:
                range_entry_counter = 0

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

            # Update realtime metrics cache every few minutes
            if (now - last_metrics_refresh).total_seconds() >= 240:
                try:
                    metrics_client.refresh()
                    strategy_health_cache.clear()
                    last_metrics_refresh = now
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[REALTIME] metrics refresh failed: %s", exc)

            atr_pips = (fac_m1.get("atr") or 0.0) * 100
            ema20 = fac_m1.get("ema20") or fac_m1.get("ma20")
            close_px = fac_m1.get("close")
            momentum = 0.0
            if ema20 is not None and close_px is not None:
                momentum = close_px - ema20
            scalp_ready = False
            if not range_active:
                scalp_ready = atr_pips >= 2.2 and abs(momentum) >= 0.0015
                if not scalp_ready and fac_m1.get("vol_5m"):
                    scalp_ready = atr_pips >= 2.0 and fac_m1["vol_5m"] >= 1.2

            focus_tag = gpt.get("focus_tag") or focus
            weight = gpt.get("weight_macro", w_macro)
            if range_active:
                focus_tag = "micro"
                weight = min(weight, 0.15)
            elif range_soft_active and focus_tag == "macro":
                if soft_range_just_activated:
                    logging.info(
                        "[FOCUS] Soft range compression forcing hybrid focus (score=%.2f).",
                        range_ctx.score,
                    )
                focus_tag = "hybrid"
            if not range_active and range_soft_active and weight > SOFT_RANGE_WEIGHT_CAP:
                prev_weight = weight
                weight = min(weight, SOFT_RANGE_WEIGHT_CAP)
                if prev_weight != weight:
                    logging.info(
                        "[MACRO] Soft range compression (score=%.2f eff_adx=%.2f) weight_macro %.2f -> %.2f",
                        range_ctx.score,
                        effective_adx_m1,
                        prev_weight,
                        weight,
                    )
            focus_pockets = set(FOCUS_POCKETS.get(focus_tag, ("macro", "micro", "scalp")))
            if not focus_pockets:
                focus_pockets = {"micro"}
            if range_active and "macro" in focus_pockets:
                focus_pockets.discard("macro")

            ma10_h4 = fac_h4.get("ma10")
            ma20_h4 = fac_h4.get("ma20")
            adx_h4 = fac_h4.get("adx", 0.0)
            slope_gap = abs((ma10_h4 or 0.0) - (ma20_h4 or 0.0))
            low_trend = (
                adx_h4 <= LOW_TREND_ADX_THRESHOLD
                and slope_gap <= LOW_TREND_SLOPE_THRESHOLD
            )
            if low_trend and "macro" in focus_pockets:
                prev_weight = weight
                weight = min(weight, LOW_TREND_WEIGHT_CAP)
                logging.info(
                    "[MACRO] H4 trend weak (ADX %.2f gap %.5f). weight_macro %.2f -> %.2f",
                    adx_h4,
                    slope_gap,
                    prev_weight,
                    weight,
                )

            if (
                scalp_ready
                and not range_active
                and "scalp" in focus_pockets
                and "M1Scalper" not in ranked_strategies
            ):
                ranked_strategies.append("M1Scalper")
                logging.info(
                    "[SCALP] Auto-added M1Scalper (ATR %.2f, momentum %.4f, vol5m %.2f).",
                    atr_pips,
                    momentum,
                    fac_m1.get("vol_5m", 0.0),
                )

            # Range mode: prefer mean-reversion scalping. Ensure RangeFader is present.
            if range_active and "scalp" in focus_pockets and "RangeFader" not in ranked_strategies:
                ranked_strategies.append("RangeFader")
                try:
                    atr_hint = fac_m1.get("atr_pips") or ((fac_m1.get("atr") or 0.0) * 100)
                except Exception:
                    atr_hint = 0.0
                logging.info(
                    "[SCALP] Range mode: auto-added RangeFader (score=%.2f bbw=%.2f atr=%.2f).",
                    range_ctx.score,
                    fac_m1.get("bbw", 0.0) or 0.0,
                    atr_hint,
                )

            evaluated_signals: list[dict] = []
            for sname in ranked_strategies:
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                pocket = cls.pocket
                if (
                    range_soft_active
                    and not range_active
                    and pocket == "macro"
                    and getattr(cls, "name", sname) in SOFT_RANGE_SUPPRESS_STRATEGIES
                ):
                    logging.info(
                        "[RANGE] Soft compression: skip %s until trend strength returns.",
                        sname,
                    )
                    continue
                if pocket not in focus_pockets:
                    logging.info(
                        "[FOCUS] skip %s pocket=%s focus=%s",
                        sname,
                        pocket,
                        focus_tag,
                    )
                    continue
                allowed_names = POCKET_STRATEGY_MAP.get(pocket)
                if allowed_names and sname not in allowed_names:
                    logging.info(
                        "[POCKET] skip %s (pocket=%s) not in whitelist %s",
                        sname,
                        pocket,
                        sorted(allowed_names),
                    )
                    continue
                if range_active and cls.name not in ALLOWED_RANGE_STRATEGIES:
                    logging.info("[RANGE] skip %s in range mode.", sname)
                    continue
                if sname == "NewsSpikeReversal":
                    raw_signal = cls.check(fac_m1, news_cache.get("short", []))
                else:
                    raw_signal = cls.check(fac_m1)
                if not raw_signal:
                    continue

                health = strategy_health_cache.get(sname)
                if not health:
                    health = metrics_client.evaluate(sname, cls.pocket)
                    health = confidence_policy.apply(health)
                    strategy_health_cache[sname] = health

                if not health.allowed:
                    logging.info(
                        "[HEALTH] skip %s pocket=%s reason=%s",
                        sname,
                        cls.pocket,
                        health.reason,
                    )
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
                scaled_conf = int(signal["confidence"] * health.confidence_scale)
                signal["confidence"] = max(0, min(100, scaled_conf))
                if range_active:
                    atr_hint = (
                        fac_m1.get("atr_pips")
                        or ((fac_m1.get("atr") or 0.0) * 100)
                        or 6.0
                    )
                    if signal["pocket"] == "macro":
                        tp_cap = min(2.0, max(1.6, atr_hint * 1.05))
                        signal["tp_pips"] = round(tp_cap, 2)
                        signal["sl_pips"] = round(tp_cap, 2)
                        signal["confidence"] = int(signal["confidence"] * 0.55)
                    else:
                        tp_default = min(1.8, max(1.3, atr_hint * 1.1))
                        signal["tp_pips"] = round(tp_default, 2)
                        signal["sl_pips"] = round(max(1.2, min(tp_default * 1.05, 1.9)), 2)
                signal["health"] = {
                    "win_rate": health.win_rate,
                    "pf": health.profit_factor,
                    "confidence_scale": health.confidence_scale,
                    "drawdown": health.max_drawdown_pips,
                    "losing_streak": health.losing_streak,
                }
                evaluated_signals.append(signal)
                logging.info("[SIGNAL] %s -> %s", cls.name, signal)

            open_positions = pos_manager.get_open_positions()
            try:
                update_dynamic_protections(open_positions, fac_m1, fac_h4)
            except Exception as exc:
                logging.warning("[PROTECTION] update failed: %s", exc)
            try:
                partials = plan_partial_reductions(
                    open_positions, fac_m1, range_mode=range_active, now=now
                )
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

            for pocket, info in open_positions.items():
                if pocket == "__net__":
                    continue
                for direction, key_units in (("long", "long_units"), ("short", "short_units")):
                    units_value = int(info.get(key_units, 0) or 0)
                    tracker_stage = stage_tracker.get_stage(pocket, direction)
                    key = (pocket, direction)
                    if units_value == 0 and tracker_stage > 0:
                        empty_since = stage_empty_since.get(key)
                        if empty_since is None:
                            stage_empty_since[key] = now
                        elif (now - empty_since).total_seconds() >= STAGE_RESET_GRACE_SECONDS:
                            logging.info(
                                "[STAGE] reset pocket=%s direction=%s after %.0fs flat",
                                pocket,
                                direction,
                                (now - empty_since).total_seconds(),
                            )
                            stage_tracker.reset_stage(pocket, direction, now=now)
                            stage_empty_since.pop(key, None)
                    elif units_value != 0:
                        stage_empty_since.pop(key, None)

            exit_decisions = exit_manager.plan_closures(
                open_positions,
                evaluated_signals,
                fac_m1,
                fac_h4,
                event_soon,
                range_active,
                now=now,
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
                if remaining <= 0:
                    if not decision.allow_reentry or decision.reason == "range_cooldown":
                        executed_pockets.add(pocket)
                        cooldown_seconds = _cooldown_for_pocket(pocket, range_active)
                        stage_tracker.set_cooldown(
                            pocket,
                            target_side,
                            reason=decision.reason,
                            seconds=cooldown_seconds,
                            now=now,
                        )
                        # Flip guard: avoid immediate opposite-direction flip after reverse exit
                        if decision.reason == "reverse_signal":
                            opposite = "short" if target_side == "long" else "long"
                            flip_cd = min(240, max(60, cooldown_seconds // 2))
                            stage_tracker.set_cooldown(
                                pocket,
                                opposite,
                                reason="flip_guard",
                                seconds=flip_cd,
                                now=now,
                            )

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

            risk_override = _dynamic_risk_pct(evaluated_signals, range_active, weight)
            if (
                _MAX_RISK_PCT > _BASE_RISK_PCT
                and (last_risk_pct is None or abs(risk_override - last_risk_pct) > 0.001)
            ):
                logging.info(
                    "[RISK] dynamic risk pct=%.3f (base=%.3f, max=%.3f, pockets=%d)",
                    risk_override,
                    _BASE_RISK_PCT,
                    _MAX_RISK_PCT,
                    len({s.get('pocket') for s in evaluated_signals if s.get('action') in {'OPEN_LONG', 'OPEN_SHORT'}}),
                )
            last_risk_pct = risk_override

            lot_total = allowed_lot(
                account_equity,
                sl_pips=max(1.0, avg_sl),
                margin_available=margin_available,
                price=fac_m1.get("close"),
                margin_rate=margin_rate,
                risk_pct_override=risk_override,
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
            update_dd_context(account_equity, weight, scalp_share)
            lots = alloc(lot_total, weight, scalp_share=scalp_share)
            for pocket_key in list(lots.keys()):
                if pocket_key not in focus_pockets:
                    lots[pocket_key] = 0.0
            active_pockets = {sig["pocket"] for sig in evaluated_signals}
            for key in list(lots):
                if key not in active_pockets:
                    lots[key] = 0.0
            if range_active and "macro" in lots:
                lots["macro"] = 0.0

            spread_skip_logged = False
            for signal in evaluated_signals:
                pocket = signal["pocket"]
                action = signal.get("action")
                if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                    continue
                if spread_gate_active:
                    if not spread_skip_logged:
                        logging.info(
                            "[SKIP] Spread guard active (%s, %s).",
                            spread_gate_reason,
                            spread_log_context,
                        )
                        spread_skip_logged = True
                    continue
                if event_soon and pocket in {"micro", "scalp"}:
                    logging.info("[SKIP] Event soon, skipping %s pocket trade.", pocket)
                    continue
                if pocket in executed_pockets:
                    logging.info("[SKIP] %s pocket already handled this loop.", pocket)
                    continue
                if range_active and pocket == "macro":
                    logging.info("[SKIP] Range mode active, skipping macro entry.")
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
                    direction = "long"
                else:
                    open_units = int(open_info.get("short_units", 0))
                    ref_price = open_info.get("short_avg_price")
                    direction = "short"

                size_factor = stage_tracker.size_multiplier(pocket, direction)
                if size_factor < 0.999:
                    logging.info("[SIZE] %s %s factor=%.2f due to streaks", pocket, direction, size_factor)
                confidence_target = round(confidence_target * size_factor, 3)
                if confidence_target <= 0:
                    continue

                blocked, remain_sec, block_reason = stage_tracker.is_blocked(
                    pocket, direction, now
                )
                if blocked:
                    logging.info(
                        "[COOLDOWN] Skip %s %s entry (%ss remaining reason=%s)",
                        pocket,
                        direction,
                        remain_sec,
                        block_reason,
                    )
                    continue

                stage_context = dict(open_info) if open_info else {}
                if ref_price is None or (ref_price == 0.0 and open_units == 0):
                    ref_price = price
                if ref_price is not None:
                    stage_context["avg_price"] = ref_price

                staged_lot, stage_idx = compute_stage_lot(
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
                # Build a lightweight entry thesis for contextual exits
                thesis_type = (
                    "trend_follow" if pocket == "macro" else ("mean_reversion" if pocket == "micro" else "scalp")
                )
                h4_ma10 = fac_h4.get("ma10")
                h4_ma20 = fac_h4.get("ma20")
                entry_thesis = {
                    "type": thesis_type,
                    "strategy": signal.get("strategy"),
                    "tag": signal.get("tag"),
                    "pocket": pocket,
                    "action": action,
                    "entry_ts": now.isoformat(timespec="seconds"),
                    "min_hold_min": 11.0 if pocket == "macro" else (5.0 if pocket == "micro" else 3.0),
                    "factors": {
                        "m1": {
                            "rsi": fac_m1.get("rsi"),
                            "adx": fac_m1.get("adx"),
                            "ema20": fac_m1.get("ema20") or fac_m1.get("ma20"),
                            "ma10": fac_m1.get("ma10"),
                            "ma20": fac_m1.get("ma20"),
                            "atr_pips": fac_m1.get("atr_pips") or ((fac_m1.get("atr") or 0.0) * 100),
                        },
                        "h4": {
                            "ma10": h4_ma10,
                            "ma20": h4_ma20,
                            "adx": fac_h4.get("adx"),
                        },
                    },
                    "invalidation_hints": (
                        ["ema20_cross", "h4_trend_weaken"]
                        if pocket == "macro"
                        else (["rsi_revert", "bb_exit"] if pocket == "micro" else ["momentum_flip"])
                    ),
                }
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl,
                    tp,
                    pocket,
                    client_order_id=client_id,
                    entry_thesis=entry_thesis,
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
                    if stage_idx >= 0:
                        stage_tracker.set_stage(pocket, direction, stage_idx + 1, now=now)
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
                    # 直後の再エントリーを抑制（気迷いトレード対策）
                    entry_cd = POCKET_ENTRY_MIN_INTERVAL.get(pocket, 120)
                    stage_tracker.set_cooldown(
                        pocket,
                        direction,
                        reason="entry_rate_limit",
                        seconds=entry_cd,
                        now=now,
                    )
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
        metrics_client.close()
        stage_tracker.close()
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
