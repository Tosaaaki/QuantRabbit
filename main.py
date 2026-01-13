import asyncio
import datetime
import logging
import traceback
import time
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple, Coroutine, Any, Dict, Sequence
from types import SimpleNamespace
from utils import signal_bus

PIP_VALUE = 0.01  # USD/JPY pip size

# Safe float conversion for optional numeric inputs
def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strategy_conf_multiplier(gpt: Dict[str, object], strategy: str) -> float:
    """Apply GPT mode/bias/pattern hints as gentle confidence multipliers."""
    base = 1.0
    mode = str(gpt.get("mode") or "").upper()
    risk = str(gpt.get("risk_bias") or "").lower()
    liq = str(gpt.get("liquidity_bias") or "").lower()
    rc = _safe_float(gpt.get("range_confidence"), 0.0)
    hints = gpt.get("pattern_hint") or []
    if isinstance(hints, str):
        hints = [hints]

    # Mode-based coefficients
    if mode == "DEFENSIVE":
        base *= 0.88
    elif mode == "TRANSITION":
        base *= 0.94
    elif mode == "TREND_FOLLOW":
        if strategy in TREND_STRATEGIES:
            base *= 1.08
        if strategy in RANGE_STRATEGIES:
            base *= 0.9
    elif mode == "RANGE_SCALP":
        if strategy in RANGE_STRATEGIES:
            base *= 1.1
        if strategy in TREND_STRATEGIES:
            base *= 0.9

    # Risk bias
    if risk == "high":
        base *= 1.08
    elif risk == "low":
        base *= 0.9

    # Liquidity bias
    if liq == "tight":
        base *= 0.92
    elif liq == "loose":
        base *= 1.04

    # Range confidence tilt
    if rc >= 0.65:
        if strategy in RANGE_STRATEGIES:
            base *= 1.08
        if strategy in TREND_STRATEGIES:
            base *= 0.95
    elif rc <= 0.35:
        if strategy in TREND_STRATEGIES:
            base *= 1.05
        if strategy in RANGE_STRATEGIES:
            base *= 0.95

    # Pattern hints
    lower_hints = [h.strip().lower() for h in hints if isinstance(h, str)]
    if any("long_wick" in h or "hammer" in h for h in lower_hints):
        if strategy in RANGE_STRATEGIES:
            base *= 1.05
    if any("bull_flag" in h or "impulse" in h for h in lower_hints):
        if strategy in MOMENTUM_STRATEGIES:
            base *= 1.05
    if any("double_top" in h or "engulfing_bear" in h for h in lower_hints):
        if strategy in TREND_STRATEGIES:
            base *= 0.95

    # Clamp
    return max(0.7, min(1.3, base))


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, None)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

# Trading from main is enabled alongside workers for higher entry density.
# 内蔵ストラテジーはデフォルト停止。動かす場合は環境変数で明示的にONにする。
MAIN_TRADING_ENABLED = _env_bool("MAIN_TRADING_ENABLED", default=False)
# ワーカー発のシグナルを関所で集約・順位付けするか
SIGNAL_GATE_ENABLED = _env_bool("SIGNAL_GATE_ENABLED", default=True)
# 関所キューから一度に取り出す件数
SIGNAL_GATE_FETCH_LIMIT = int(os.getenv("SIGNAL_GATE_FETCH_LIMIT", "120"))
SIGNAL_DIVERSITY_ENABLED = _env_bool("SIGNAL_DIVERSITY_ENABLED", default=True)
SIGNAL_DIVERSITY_DEDUPE = _env_bool("SIGNAL_DIVERSITY_DEDUPE", default=True)
SIGNAL_DIVERSITY_IDLE_SEC = float(os.getenv("SIGNAL_DIVERSITY_IDLE_SEC", "300"))
SIGNAL_DIVERSITY_SCALE_SEC = float(os.getenv("SIGNAL_DIVERSITY_SCALE_SEC", "1200"))
SIGNAL_DIVERSITY_MAX_BONUS = float(os.getenv("SIGNAL_DIVERSITY_MAX_BONUS", "8"))
_SIGNAL_DIVERSITY_LAST_TS: Dict[str, float] = {}

# Aggressive mode: loosen range gatesとマイクロの入口ガードを緩めるフラグ
# デフォルトは安全寄りに OFF
AGGRESSIVE_TRADING = _env_bool("AGGRESSIVE_TRADING", default=False)
# Micro 新規エントリーを緊急停止するフラグ（Exit は動かす）
MICRO_OPENS_DISABLED = _env_bool("MICRO_OPENS_DISABLED", default=False)

# Worker-onlyモード: mainはワーカー起動/データ供給のみ行い、発注/Exitロジックはスキップ
WORKER_ONLY_MODE = _env_bool("WORKER_ONLY_MODE", default=False)
# GPT を完全に無効化するフラグ（誤作動防止用）。設定時はローカル順位付けのみを使用。
GPT_DISABLED = _env_bool("GPT_DISABLED", default=False)

# ---- Dynamic allocation (strategy score / pocket cap) loader ----
_DYNAMIC_ALLOC_PATH = Path("config/dynamic_alloc.json")
_DYNAMIC_ALLOC_MTIME: float | None = None
_DYNAMIC_ALLOC_CACHE: dict | None = None


def load_dynamic_alloc() -> Optional[dict]:
    """Load dynamic allocation JSON if present; cached by mtime."""
    global _DYNAMIC_ALLOC_MTIME, _DYNAMIC_ALLOC_CACHE
    try:
        stat = _DYNAMIC_ALLOC_PATH.stat()
    except FileNotFoundError:
        _DYNAMIC_ALLOC_CACHE = None
        _DYNAMIC_ALLOC_MTIME = None
        return None
    if _DYNAMIC_ALLOC_MTIME == stat.st_mtime and _DYNAMIC_ALLOC_CACHE is not None:
        return _DYNAMIC_ALLOC_CACHE
    try:
        data = json.loads(_DYNAMIC_ALLOC_PATH.read_text())
        _DYNAMIC_ALLOC_CACHE = data
        _DYNAMIC_ALLOC_MTIME = stat.st_mtime
        return data
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[DYN_ALLOC] failed to load %s: %s", _DYNAMIC_ALLOC_PATH, exc)
        _DYNAMIC_ALLOC_CACHE = None
        _DYNAMIC_ALLOC_MTIME = stat.st_mtime
        return None


def _signal_strategy_key(sig: dict) -> str:
    strategy = sig.get("strategy") or sig.get("strategy_tag") or sig.get("tag") or "unknown"
    pocket = sig.get("pocket") or "unknown"
    return f"{pocket}:{strategy}"


def _signal_diversity_bonus(strategy_key: str, now_ts: float) -> float:
    if not SIGNAL_DIVERSITY_ENABLED:
        return 0.0
    last_ts = _SIGNAL_DIVERSITY_LAST_TS.get(strategy_key)
    if last_ts is None:
        return SIGNAL_DIVERSITY_MAX_BONUS
    idle = max(0.0, now_ts - last_ts)
    if idle < SIGNAL_DIVERSITY_IDLE_SEC:
        return 0.0
    scale = max(1.0, SIGNAL_DIVERSITY_SCALE_SEC)
    bonus = (idle - SIGNAL_DIVERSITY_IDLE_SEC) / scale * SIGNAL_DIVERSITY_MAX_BONUS
    return min(SIGNAL_DIVERSITY_MAX_BONUS, bonus)


def apply_dynamic_alloc(signals: list[dict], alloc: Optional[dict]) -> tuple[list[dict], dict, float]:
    """Adjust confidence based on dynamic scores; return (signals, pocket_caps, target_use)."""
    if not alloc:
        return signals, {}, 0.88
    strat_scores = alloc.get("strategies") or {}
    pocket_caps = alloc.get("pocket_caps") or {}
    target_use = float(alloc.get("target_use", 0.88) or 0.88)
    adjusted: list[dict] = []
    for sig in signals:
        strat = sig.get("strategy")
        info = strat_scores.get(strat, {}) if isinstance(strat_scores, dict) else {}
        try:
            score = float(info.get("score", 1.0) or 1.0)
        except Exception:
            score = 1.0
        conf = int(sig.get("confidence", 0) or 0)
        new_conf = conf
        if score < 0.15:
            new_conf = max(0, int(conf * 0.3))
        elif score < 0.3:
            new_conf = max(0, int(conf * 0.6))
        elif score > 0.8:
            new_conf = min(100, int(conf * 1.05))
        if new_conf != conf:
            sig = dict(sig)
            sig["confidence"] = new_conf
        adjusted.append(sig)
    return adjusted, pocket_caps, target_use

from market_data.candle_fetcher import (
    Candle,
    start_candle_stream,
    initialize_history,
)
from market_data import spread_monitor, tick_window
from indicators.factor_cache import all_factors, get_candles_snapshot, on_candle
from analysis.regime_classifier import classify
from analysis.focus_decider import decide_focus
from analysis.local_decider import heuristic_decision
from analysis.gpt_decider import get_decision
from analysis.perf_monitor import snapshot as get_perf
from analytics.insight_client import InsightClient
try:
    from analytics.firestore_strategy_client import (
        FirestoreStrategyClient,
        firestore_strategy_enabled,
    )
    from analytics.level_map_client import LevelMapClient
except Exception:  # pragma: no cover - optional dependency
    logging.warning("[FIRESTORE] firestore_strategy_client not available; exporting disabled")

    class FirestoreStrategyClient:  # type: ignore[override]
        def __init__(self, enable: bool = False):
            self.enable = False

        def export_scores(self, *_, **__):
            return None

    def firestore_strategy_enabled() -> bool:
        return False

    class LevelMapClient:  # type: ignore[override]
        def __init__(self, *_, **__):
            self.enabled = False

        def refresh(self, *_, **__):
            return None

        def nearest(self, *_, **__):
            return None
from analysis.range_guard import RangeContext, detect_range_mode, detect_range_mode_for_tf
from analysis.range_model import compute_range_snapshot
from analysis.param_context import ParamContext, ParamSnapshot
from analysis.chart_story import ChartStory, ChartStorySnapshot
from signals.pocket_allocator import (
    alloc,
    DEFAULT_SCALP_SHARE,
    dynamic_scalp_share,
    MIN_MICRO_WEIGHT,
    MIN_MACRO_WEIGHT,
    MIN_SCALP_WEIGHT,
)
from execution.risk_guard import (
    allowed_lot,
    can_trade,
    clamp_sl_tp,
    check_global_drawdown,
    update_dd_context,
    MAX_MARGIN_USAGE,
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
from strategies.trend.h1_momentum import H1MomentumSwing
from strategies.breakout.donchian55 import Donchian55
from strategies.mean_reversion.bb_rsi import BBRsi
from strategies.scalping.m1_scalper import M1Scalper
from strategies.scalping.range_fader import RangeFader
from strategies.scalping.pulse_break import PulseBreak
from strategies.scalping.impulse_retrace import ImpulseRetraceScalp
from strategies.micro.momentum_burst import MomentumBurstMicro
from strategies.micro.momentum_stack import MicroMomentumStack
from strategies.micro.pullback_ema import MicroPullbackEMA
from strategies.micro.level_reactor import MicroLevelReactor
from strategies.micro.range_break import MicroRangeBreak
from strategies.micro.vwap_bound_revert import MicroVWAPBound
from strategies.micro.trend_momentum import TrendMomentumMicro
from strategies.micro_lowvol.micro_vwap_revert import MicroVWAPRevert
from strategies.micro_lowvol.bb_rsi_fast import BBRsiFast
from strategies.micro_lowvol.vol_compression_break import VolCompressionBreak
from strategies.micro_lowvol.momentum_pulse import MomentumPulse
from utils.oanda_account import get_account_snapshot, get_position_summary
from utils.secrets import get_secret
from utils.metrics_logger import log_metric
from utils.market_hours import is_market_open, seconds_until_open
from advisors.rr_ratio import RRRatioAdvisor
from advisors.exit_advisor import ExitAdvisor
from advisors.strategy_confidence import StrategyConfidenceAdvisor
from advisors.focus_override import FocusOverrideAdvisor
from advisors.volatility_bias import VolatilityBiasAdvisor
from advisors.stage_plan import StagePlanAdvisor
from advisors.partial_reduction import PartialReductionAdvisor
from workers.fast_scalp import FastScalpState, fast_scalp_worker
from workers.fast_scalp.signal import _compute_rsi as _fs_compute_rsi  # reuse RSI helper

# Configure logging (stdout + file)
LOG_PATH = Path("logs/pipeline.log")
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
    force=True,  # enforce INFO even if another logger configured earlier
)

logging.info("Application started!")
logging.info(
    "[CONFIG] main_trading_enabled=%s worker_only_mode=%s signal_gate=%s",
    MAIN_TRADING_ENABLED,
    WORKER_ONLY_MODE,
    SIGNAL_GATE_ENABLED,
)

# Backward-compatible alias (expected by STRATEGIES map and logs)
TrendMA = MovingAverageCross

# 内蔵ストラテジーでも、既に独立ワーカー化されているものはメイン側で発注しない
DISABLE_MAIN_STRATEGIES = {
    "TrendMA",
    "Donchian55",
    "BB_RSI",
    "MicroRangeBreak",
    "M1Scalper",
}

# メイン側では内蔵ストラテジーを発注しない（ワーカーに移行済み）
STRATEGIES: dict[str, object] = {}
TREND_STRATEGIES: set[str] = set()
RANGE_STRATEGIES: set[str] = set()
MOMENTUM_STRATEGIES: set[str] = set()
POCKET_STRATEGY_MAP: dict[str, set[str]] = {}

FOCUS_POCKETS = {
    "macro": ("macro",),
    "micro": ("micro", "scalp"),
    "hybrid": ("macro", "micro", "scalp"),
    "event": ("macro", "micro"),
}

POCKET_EXIT_COOLDOWNS = {
    "macro": 720,
    "micro": 360,
    "scalp": 240,
}

POCKET_LOSS_COOLDOWNS = {
    "macro": 900,
    "micro": 600,
    "scalp": 360,
}

# 新規エントリー後のクールダウン（再エントリー抑制）
POCKET_ENTRY_MIN_INTERVAL = {
    "macro": 180,  # 少し短縮
    "micro": 150,  # 2.5分。micro の連打抑制を強める
    "scalp": 90,   # 1.5分
}

# Gross exposure guard (both sides合計での上限を軽く見る)
GROSS_EXPOSURE_SOFT = float(os.getenv("GROSS_EXPOSURE_SOFT", "1.6") or 1.6)
GROSS_EXPOSURE_HARD = float(os.getenv("GROSS_EXPOSURE_HARD", "1.8") or 1.8)

# reduce_only 用のデフォルトSL/TP（シグナル側で未指定のときに使用）
REDUCE_ONLY_DEFAULT_SL_PIPS = float(os.getenv("REDUCE_ONLY_DEFAULT_SL_PIPS", "5.0") or 5.0)
REDUCE_ONLY_DEFAULT_TP_PIPS = float(os.getenv("REDUCE_ONLY_DEFAULT_TP_PIPS", "5.0") or 5.0)

if os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"", "0", "false", "no"}:
    POCKET_LOSS_COOLDOWNS["scalp"] = 120
    POCKET_ENTRY_MIN_INTERVAL["scalp"] = 25


def _dynamic_entry_cooldown_seconds(
    pocket: str,
    fac_m1: Optional[dict],
    fac_h1: Optional[dict],
    fac_h4: Optional[dict],
    range_active: bool,
) -> int:
    """
    Determine entry cooldown using live technicals to boost throughput when 市況が強い。
    - 強トレンド/高ボラ: クールダウン短縮（エントリー頻度↑）
    - 低ボラ/レンジ圧縮: 伸長
    """
    base = POCKET_ENTRY_MIN_INTERVAL.get(pocket, 120)
    if base <= 0:
        return 0

    m1 = fac_m1 or {}
    h1 = fac_h1 or {}
    h4 = fac_h4 or {}

    adx_m1 = _safe_float(m1.get("adx"), 0.0)
    vol_5m = _safe_float(m1.get("vol_5m"), 1.0)
    atr_m1 = _safe_float(m1.get("atr_pips"), _safe_float(m1.get("atr"), 0.0) * 100.0)
    bbw_m1 = _safe_float(m1.get("bbw"), 0.0)
    gap_m1 = abs(_safe_float(m1.get("ma10"), 0.0) - _safe_float(m1.get("ma20"), 0.0)) / 0.01

    adx_h1 = _safe_float(h1.get("adx"), 0.0)
    adx_h4 = _safe_float(h4.get("adx"), 0.0)
    bbw_h1 = _safe_float(h1.get("bbw"), 0.0)
    atr_h1 = _safe_float(h1.get("atr_pips"), _safe_float(h1.get("atr"), 0.0) * 100.0)

    scale = 1.0
    if pocket == "macro":
        trend_strength = max(adx_h1, adx_h4)
        if trend_strength >= 26.0 or atr_h1 >= 3.2:
            scale *= 0.55
        elif trend_strength >= 22.0:
            scale *= 0.72
        elif trend_strength <= 16.0 or bbw_h1 <= 0.012:
            scale *= 1.15
    else:
        # micro/scalp 系は短期トレンドとボラを優先
        if adx_m1 >= 24.0 and vol_5m >= 0.7 and atr_m1 >= 2.0:
            scale *= 0.45
        elif adx_m1 >= 20.0 and vol_5m >= 0.5:
            scale *= 0.65
        elif adx_m1 <= 13.0 or vol_5m < 0.35:
            scale *= 1.18
        if gap_m1 >= 4.0 and adx_m1 >= 22.0:
            scale *= 0.9
        if range_active and (adx_m1 < 14.5 or bbw_m1 < 0.0016):
            scale *= 1.15

    scale = max(0.38, min(1.35, scale))
    return int(max(20, round(base * scale)))

PIP = 0.01

MACRO_DISABLED = _env_bool("MACRO_DISABLE", False)
MACRO_STRATEGIES = {"TrendMA", "Donchian55"}

POCKET_MAX_ACTIVE_TRADES = {
    "macro": 20,
    "micro": 12,
    "scalp": 8,
}

POCKET_MAX_ACTIVE_TRADES_RANGE = {
    "macro": 20,
    "micro": 12,
    "scalp": 8,
}

POCKET_MAX_DIRECTIONAL_TRADES = {
    "macro": 12,
    "micro": 6,
    "scalp": 6,
}

POCKET_MAX_DIRECTIONAL_TRADES_RANGE = {
    "macro": 12,
    "micro": 6,
    "scalp": 6,
}

try:
    HEDGING_ENABLED = get_secret("oanda_hedging_enabled").lower() == "true"
except Exception:
    HEDGING_ENABLED = False
if HEDGING_ENABLED:
    logging.info("[CONFIG] Hedging enabled; allowing offsetting positions.")

FALLBACK_EQUITY = 10000.0  # REST失敗時のフォールバック

RSI_LONG_FLOOR = {
    "macro": 35.0,
    "micro": 34.0,
    "scalp": 36.0,
}

RSI_SHORT_CEILING = {
    "macro": 65.0,
    "micro": 66.0,
    "scalp": 64.0,
}

POCKET_ATR_MIN_PIPS = {
    "micro": 1.1,
    "scalp": 1.1,
}

try:
    RR_ADVISOR = RRRatioAdvisor()
except Exception as exc:  # pragma: no cover - defensive
    logging.warning("[RR_ADVISOR] init failed: %s", exc)
    RR_ADVISOR = None

try:
    EXIT_ADVISOR = ExitAdvisor()
except Exception as exc:  # pragma: no cover - defensive
    logging.warning("[EXIT_ADVISOR] init failed: %s", exc)
    EXIT_ADVISOR = None

try:
    STRATEGY_CONF_ADVISOR = StrategyConfidenceAdvisor()
except Exception as exc:  # pragma: no cover - defensive
    logging.warning("[STRAT_CONF] init failed: %s", exc)
    STRATEGY_CONF_ADVISOR = None

try:
    FOCUS_ADVISOR = FocusOverrideAdvisor()
except Exception as exc:  # pragma: no cover - defensive
    logging.warning("[FOCUS_ADVISOR] init failed: %s", exc)
    FOCUS_ADVISOR = None

try:
    VOLATILITY_ADVISOR = VolatilityBiasAdvisor()
except Exception as exc:  # pragma: no cover - defensive
    logging.warning("[VOL_ADVISOR] init failed: %s", exc)
    VOLATILITY_ADVISOR = None

try:
    STAGE_PLAN_ADVISOR = StagePlanAdvisor()
except Exception as exc:  # pragma: no cover - defensive
    logging.warning("[STAGE_PLAN_ADVISOR] init failed: %s", exc)
    STAGE_PLAN_ADVISOR = None

try:
    PARTIAL_ADVISOR = PartialReductionAdvisor()
except Exception as exc:  # pragma: no cover - defensive
    logging.warning("[PARTIAL_ADVISOR] init failed: %s", exc)
    PARTIAL_ADVISOR = None

PIP = 0.01
_MACRO_PULLBACK_MIN_RETRACE_PIPS = {
    1: 4.8,
    2: 6.4,
    3: 8.0,
    4: 9.2,
    5: 10.5,
}
_MACRO_PULLBACK_MAX_RETRACE_PIPS = 16.0
_MACRO_PULLBACK_MAX_EMA_GAP_PIPS = 6.5
_MACRO_PULLBACK_MAX_MA_SLACK_PIPS = 2.1
_MACRO_PULLBACK_MIN_ADX = 19.0
_MACRO_LIMIT_TIMEOUT_SEC = 60.0
_MACRO_LIMIT_TIMEOUT_MIN = 35.0
MACRO_LIMIT_WAIT: dict[str, dict[str, float]] = {}
_DIR_BIAS_SCALE_OPPOSE = 0.35
_DIR_BIAS_SCALE_ALIGN = 1.05
# Clamp/L3 reduce-only controls
_CLAMP_L3_REDUCE_FRACTION = 0.25  # fraction of net to unwind per reduce-only order
_CLAMP_L3_MIN_REDUCE_UNITS = 1000

_BASE_STAGE_RATIOS = {
    # Spread staging to smooth adverse price moves while preserving total exposure.
    "macro": (0.32, 0.22, 0.18, 0.14, 0.09, 0.05),
    "micro": (0.22, 0.19, 0.17, 0.15, 0.14, 0.13),
    "scalp": (1.0, 0.35, 0.2),
}

_STAGE_PLAN_OVERRIDES: dict[str, tuple[float, ...]] = {}

def _safe_env_float(key: str, default: float, *, low: float, high: float) -> float:
    value = default
    raw = os.getenv(key)
    if raw is not None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = default
    return max(low, min(high, value))


SCALP_SHADOW_MODE = _env_bool("SCALP_SHADOW_MODE", False)
SCALP_SHADOW_LOG_ONLY = _env_bool("SCALP_SHADOW_LOG_ONLY", False)
SCALP_SHADOW_SPREAD_PIPS = _safe_env_float("SCALP_SHADOW_SPREAD_PIPS", 0.2, low=0.0, high=5.0)
SCALP_SHADOW_MEDIAN_PIPS = _safe_env_float("SCALP_SHADOW_MEDIAN_PIPS", 0.25, low=0.0, high=5.0)
SCALP_SHADOW_LATENCY_MS = _safe_env_float("SCALP_SHADOW_LATENCY_MS", 1200.0, low=0.0, high=20000.0)
SCALP_SHADOW_MIN_ATR_PIPS = _safe_env_float("SCALP_SHADOW_MIN_ATR_PIPS", 0.5, low=0.0, high=20.0)
SCALP_SHADOW_LOG_PATH = Path(os.getenv("SCALP_SHADOW_LOG_PATH", "logs/scalp_shadow.jsonl"))


def _write_scalp_shadow_log(payload: dict) -> None:
    try:
        SCALP_SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SCALP_SHADOW_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        logging.debug("[SCALP_SHADOW] write failed", exc_info=True)


def _scalp_shadow_gate(
    *,
    spread_state: Optional[dict],
    spread_gate_reason: str,
    atr_pips: Optional[float],
    latency_ms: Optional[float],
) -> tuple[bool, str, dict]:
    state = spread_state or {}
    spread_latest = state.get("spread_pips")
    spread_p50 = state.get("baseline_p50_pips") or state.get("median_pips")
    reasons: list[str] = []
    if spread_gate_reason:
        reasons.append(spread_gate_reason)
    if SCALP_SHADOW_SPREAD_PIPS > 0 and spread_latest is not None and spread_latest > SCALP_SHADOW_SPREAD_PIPS:
        reasons.append(f"spread>{SCALP_SHADOW_SPREAD_PIPS:.2f}")
    if SCALP_SHADOW_MEDIAN_PIPS > 0 and spread_p50 is not None and spread_p50 > SCALP_SHADOW_MEDIAN_PIPS:
        reasons.append(f"spread_p50>{SCALP_SHADOW_MEDIAN_PIPS:.2f}")
    if SCALP_SHADOW_LATENCY_MS > 0 and latency_ms is not None and latency_ms > SCALP_SHADOW_LATENCY_MS:
        reasons.append(f"latency>{SCALP_SHADOW_LATENCY_MS:.0f}ms")
    if SCALP_SHADOW_MIN_ATR_PIPS > 0 and atr_pips is not None and atr_pips < SCALP_SHADOW_MIN_ATR_PIPS:
        reasons.append(f"atr<{SCALP_SHADOW_MIN_ATR_PIPS:.2f}")
    passed = len(reasons) == 0
    ctx = {
        "spread_pips": spread_latest,
        "spread_p50": spread_p50,
        "latency_ms": latency_ms,
        "atr_pips": atr_pips,
    }
    return passed, ("ok" if passed else ";".join(reasons)), ctx


SCALP_WEIGHT_FLOOR = _safe_env_float("SCALP_WEIGHT_FLOOR", 0.22, low=0.0, high=0.4)
SCALP_WEIGHT_READY_FLOOR = _safe_env_float(
    "SCALP_WEIGHT_READY_FLOOR", 0.32, low=SCALP_WEIGHT_FLOOR, high=0.45
)
SCALP_AUTO_MIN_WEIGHT = _safe_env_float("SCALP_AUTO_MIN_WEIGHT", 0.12, low=0.0, high=0.3)
SCALP_CONFIDENCE_FLOOR = _safe_env_float("SCALP_CONFIDENCE_FLOOR", 0.74, low=0.4, high=1.0)
# スカルプロットの絶対下限は 0（フロアなし）。環境変数での下限設定も無効化。
SCALP_MIN_ABS_LOT = 0.0
PULSEBREAK_AUTO_MOM_MIN = _safe_env_float("PULSEBREAK_AUTO_MOM_MIN", 0.0018, low=0.0, high=0.02)
PULSEBREAK_AUTO_ATR_MIN = _safe_env_float("PULSEBREAK_AUTO_ATR_MIN", 2.4, low=0.0, high=15.0)
PULSEBREAK_AUTO_VOL_MIN = _safe_env_float("PULSEBREAK_AUTO_VOL_MIN", 1.3, low=0.0, high=5.0)
MACRO_LOT_SHARE_FLOOR = _safe_env_float("MACRO_LOT_SHARE_FLOOR", 0.7, low=0.0, high=0.95)
MACRO_CONFIDENCE_FLOOR = _safe_env_float("MACRO_CONFIDENCE_FLOOR", 1.0, low=0.3, high=1.0)
MACRO_SPREAD_OVERRIDE = _safe_env_float("MACRO_SPREAD_OVERRIDE", 1.4, low=1.0, high=3.0)

GPT_FACTOR_KEYS: Dict[str, tuple[str, ...]] = {
    "M1": (
        "close",
        "ma10",
        "ma20",
        "adx",
        "rsi",
        "atr_pips",
        "vol_5m",
        "bbw",
    ),
    "H4": (
        "close",
        "ma10",
        "ma20",
        "adx",
        "atr_pips",
        "rsi",
    ),
}
# systemd service mapping for worker/exit processes (names must match .service files)
WORKER_SERVICES = {
    # Scalp / S5
    "fast_scalp": "quant-fast-scalp.service",
    "fast_scalp_exit": "quant-fast-scalp-exit.service",
    "impulse_break_s5": "quant-impulse-break-s5.service",
    "impulse_break_s5_exit": "quant-impulse-break-s5-exit.service",
    "impulse_momentum_s5": "quant-impulse-momentum-s5.service",
    "impulse_momentum_s5_exit": "quant-impulse-momentum-s5-exit.service",
    "impulse_retest_s5": "quant-impulse-retest-s5.service",
    "impulse_retest_s5_exit": "quant-impulse-retest-s5-exit.service",
    "pullback_s5": "quant-pullback-s5.service",
    "pullback_s5_exit": "quant-pullback-s5-exit.service",
    "pullback_runner_s5": "quant-pullback-runner-s5.service",
    "pullback_runner_s5_exit": "quant-pullback-runner-s5-exit.service",
    "pullback_scalp": "quant-pullback-scalp.service",
    "pullback_scalp_exit": "quant-pullback-scalp-exit.service",
    "squeeze_break_s5": "quant-squeeze-break-s5.service",
    "squeeze_break_s5_exit": "quant-squeeze-break-s5-exit.service",
    "vwap_magnet_s5": "quant-vwap-magnet-s5.service",
    "vwap_magnet_s5_exit": "quant-vwap-magnet-s5-exit.service",
    "mirror_spike_s5": "quant-mirror-spike-s5.service",
    "mirror_spike_s5_exit": "quant-mirror-spike-s5-exit.service",
    "mirror_spike_tight": "quant-mirror-spike-tight.service",
    "mirror_spike_tight_exit": "quant-mirror-spike-tight-exit.service",
    "mirror_spike": "quant-mirror-spike.service",
    "mirror_spike_exit": "quant-mirror-spike-exit.service",
    "onepip_maker_s1": "quant-onepip-s1.service",
    "onepip_maker_s1_exit": "quant-onepip-s1-exit.service",
    "scalp_multi": "quant-scalp-multi.service",
    "scalp_multi_exit": "quant-scalp-multi-exit.service",
    "m1_scalper": "qr-m1scalper.service",
    "m1_scalper_exit": "quant-m1scalper-exit.service",
    # Micro
    "micro_bbrsi": "qr-micro-bbrsi.service",
    "micro_levelreactor": "qr-micro-level-reactor.service",
    "micro_levelreactor_exit": "qr-micro-level-reactor-exit.service",
    "micro_momentumburst": "qr-micro-momentum-burst.service",
    "micro_momentumburst_exit": "qr-micro-momentum-burst-exit.service",
    "micro_momentumstack": "qr-micro-momentum-stack.service",
    "micro_pullbackema": "qr-micro-pullback-ema.service",
    "micro_pullbackema_exit": "qr-micro-pullback-ema-exit.service",
    "micro_rangebreak": "qr-micro-range-break.service",
    "micro_rangebreak_exit": "qr-micro-range-break-exit.service",
    "micro_trendmomentum": "qr-micro-trend-momentum.service",
    "micro_trendmomentum_exit": "qr-micro-trend-momentum-exit.service",
    "micro_vwapbound": "qr-micro-vwap-bound.service",
    "micro_vwapbound_exit": "qr-micro-vwap-bound-exit.service",
    "micro_multi": "quant-micro-multi.service",
    "micro_multi_exit": "quant-micro-multi-exit.service",
    # Macro
    "macro_trendma": "quant-trendma.service",
    "macro_trendma_exit": "quant-trendma-exit.service",
    "macro_donchian55": "quant-donchian55.service",
    "macro_donchian55_exit": "quant-donchian55-exit.service",
    "macro_h1momentum": "quant-h1momentum.service",
    "macro_h1momentum_exit": "quant-h1momentum-exit.service",
    "macro_trend_h1": "quant-trend-h1.service",
    "macro_trend_h1_exit": "quant-trend-h1-exit.service",
    "macro_london_momentum": "quant-london-momentum.service",
    "macro_london_momentum_exit": "quant-london-momentum-exit.service",
    "macro_manual_swing": "quant-manual-swing.service",
    "macro_manual_swing_exit": "quant-manual-swing-exit.service",
}
WORKER_ALL_SERVICES = set(WORKER_SERVICES.keys())
WORKER_AUTOCONTROL_ENABLED = os.getenv("WORKER_AUTOCONTROL", "1").strip() not in {"", "0", "false", "no"}
# Default to a higher cap so more workers can run in parallel; 0 or negative = no cap
WORKER_AUTOCONTROL_LIMIT = int(os.getenv("WORKER_AUTOCONTROL_LIMIT", "16") or "16")
WORKER_SYSTEMCTL_ENABLED = _env_bool("WORKER_SYSTEMCTL", True)
GPT_PERF_KEYS = ("pf", "win_rate", "avg_pips", "sharpe", "sample")
_GPT_FACTOR_PRECISION = {
    "adx": 2,
    "rsi": 2,
    "atr_pips": 3,
    "vol_5m": 3,
    "bbw": 3,
}
_EXIT_MAIN_DISABLED_POCKETS = {
    p.strip().lower()
    for p in os.getenv("EXIT_MAIN_DISABLE_POCKETS", "").split(",")
    if p.strip()
}
if _env_bool("EXIT_MAIN_DISABLE_MICRO", False):
    _EXIT_MAIN_DISABLED_POCKETS.add("micro")


def _set_stage_plan_overrides(overrides: dict[str, tuple[float, ...]]) -> None:
    global _STAGE_PLAN_OVERRIDES
    _STAGE_PLAN_OVERRIDES = {k: tuple(v) for k, v in overrides.items()}


def _stage_plan(pocket: str) -> tuple[float, ...]:
    if pocket in _STAGE_PLAN_OVERRIDES:
        return _STAGE_PLAN_OVERRIDES[pocket]
    if pocket in _BASE_STAGE_RATIOS:
        return _BASE_STAGE_RATIOS[pocket]
    return (1.0,)


def _normalize_plan(plan: Sequence[float]) -> tuple[float, ...]:
    values = []
    for val in plan:
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        if num <= 0:
            continue
        values.append(num)
    if not values:
        return (1.0,)
    total = sum(values)
    if total <= 0:
        return (1.0,)
    normalized = [v / total for v in values]
    norm_sum = sum(normalized)
    if abs(norm_sum - 1.0) > 1e-6:
        normalized[-1] += 1.0 - norm_sum
    rounded = [round(max(0.0, val), 4) for val in normalized]
    diff = round(1.0 - sum(rounded), 4)
    rounded[-1] = round(max(0.0, rounded[-1] + diff), 4)
    return tuple(rounded)


def _mtf_dir_score(fac: Optional[dict], side: str, adx_floor: float, slope_floor: float) -> float:
    if not fac:
        return 0.0
    try:
        ma_fast = float(fac.get("ma10") or fac.get("ema12") or 0.0)
        ma_slow = float(fac.get("ma20") or fac.get("ema20") or 0.0)
        adx = float(fac.get("adx") or 0.0)
    except Exception:
        return 0.0
    dir_ok = (side == "long" and ma_fast > ma_slow) or (side == "short" and ma_fast < ma_slow)
    if not dir_ok:
        return -0.3
    gap = abs(ma_fast - ma_slow) / PIP
    score = 0.2 + min(0.5, gap * 0.05)
    if adx >= adx_floor:
        score += min(0.4, (adx - adx_floor) * 0.02)
    try:
        ema_fast = float(fac.get("ema12") or ma_fast)
        ema_slow = float(fac.get("ema20") or ma_slow)
        slope = (ema_fast - ema_slow) / PIP
        if (side == "long" and slope > slope_floor) or (side == "short" and -slope > slope_floor):
            score += 0.1
    except Exception:
        pass
    return max(0.0, min(1.0, score))


def _apply_tech_overlays(signal: dict, fac_m1: dict, fac_m5: Optional[dict] = None, fac_h1: Optional[dict] = None, fac_h4: Optional[dict] = None) -> dict:
    """
    Cross-worker tech overlays: apply Ichimoku/cluster/oscillator context to confidence/TP.
    """
    if not signal or not isinstance(signal, dict):
        return signal
    # 共通オーバーレイを使わない場合はそのまま返す（戦略内で個別処理する前提）
    if os.getenv("TECH_OVERLAY_ENABLE", "0").strip().lower() in {"", "0", "false", "no"}:
        return signal
    sig = dict(signal)
    try:
        conf = int(sig.get("confidence", 50) or 50)
    except Exception:
        conf = 50
    try:
        tp = float(sig.get("tp_pips")) if sig.get("tp_pips") is not None else None
    except Exception:
        tp = None
    action = str(sig.get("action", "")).upper()

    def _safe_float(val, default=0.0):
        try:
            return float(val)
        except Exception:
            return default

    cloud_pos = _safe_float(fac_m1.get("ichimoku_cloud_pos"), 0.0)
    span_a_gap = _safe_float(fac_m1.get("ichimoku_span_a_gap"), 0.0)
    span_b_gap = _safe_float(fac_m1.get("ichimoku_span_b_gap"), 0.0)
    cluster_high = _safe_float(fac_m1.get("cluster_high_gap"), 0.0)
    cluster_low = _safe_float(fac_m1.get("cluster_low_gap"), 0.0)
    macd_hist = _safe_float(fac_m1.get("macd_hist"), 0.0)
    dmi_diff = _safe_float(fac_m1.get("plus_di"), 0.0) - _safe_float(fac_m1.get("minus_di"), 0.0)
    stoch = _safe_float(fac_m1.get("stoch_rsi"), 0.5)
    vol5 = _safe_float(fac_m1.get("vol_5m"), 1.0)
    roc5 = _safe_float(fac_m1.get("roc5"), 0.0)
    roc10 = _safe_float(fac_m1.get("roc10"), 0.0)
    bbw = _safe_float(fac_m1.get("bbw"), 0.0)
    cci = _safe_float(fac_m1.get("cci"), 0.0)
    vwap_gap = _safe_float(fac_m1.get("vwap_gap"), 0.0)
    adx_m1 = _safe_float(fac_m1.get("adx"), 0.0)
    vol_5m = _safe_float(fac_m1.get("vol_5m"), 1.0)
    ma10 = _safe_float(fac_m1.get("ma10"), 0.0)
    ma20 = _safe_float(fac_m1.get("ma20"), 0.0)
    slope_pips = (ma10 - ma20) / PIP if PIP else 0.0
    gap_pips = abs(ma10 - ma20) / PIP if PIP else 0.0

    mult_conf = 1.0
    mult_tp = 1.0

    # Ichimoku bias: above/below cloud supports順方向、雲中は弱め
    if action == "OPEN_LONG":
        if cloud_pos > 0.5:
            mult_conf += 0.06
        elif cloud_pos < -0.5:
            mult_conf -= 0.12
            mult_tp -= 0.08
    elif action == "OPEN_SHORT":
        if cloud_pos < -0.5:
            mult_conf += 0.06
        elif cloud_pos > 0.5:
            mult_conf -= 0.12
            mult_tp -= 0.08

    # Span方向に素直な場合は少し強化
    if action == "OPEN_LONG" and span_a_gap > 0 and span_b_gap > 0:
        mult_conf += 0.03
    if action == "OPEN_SHORT" and span_a_gap < 0 and span_b_gap < 0:
        mult_conf += 0.03

    # クラスタ距離が近い側は抑制、遠いなら伸ばす
    if action == "OPEN_LONG":
        dist = cluster_high
    else:
        dist = cluster_low
    if dist > 0:
        if dist < 3.0:
            mult_conf -= 0.08
            mult_tp -= 0.1
        elif dist > 7.0:
            mult_conf += 0.05
            mult_tp += 0.08

    # MACD/DMI 順方向は強化、逆行は抑制
    if action == "OPEN_LONG":
        if macd_hist > 0 or dmi_diff > 2.0:
            mult_conf += 0.04
        elif macd_hist < 0 and dmi_diff < -2.0:
            mult_conf -= 0.08
    elif action == "OPEN_SHORT":
        if macd_hist < 0 or dmi_diff < -2.0:
            mult_conf += 0.04
        elif macd_hist > 0 and dmi_diff > 2.0:
            mult_conf -= 0.08

    # StochRSI極端は抑制、適度なボラは維持
    if stoch >= 0.85 or stoch <= 0.15:
        mult_conf -= 0.05
    if vol5 < 0.4:
        mult_conf -= 0.04

    # ROC/モメ判定: 方向一致で軽く強化（頻度維持のため抑制は緩め）
    if action == "OPEN_LONG" and (roc5 > 0.0 or roc10 > 0.0):
        mult_conf += 0.04
    elif action == "OPEN_SHORT" and (roc5 < 0.0 or roc10 < 0.0):
        mult_conf += 0.04
    elif roc5 == 0 and roc10 == 0:
        mult_conf -= 0.02

    # M1総合スコア（ADX/ROC/スロープ/BBW/vol_5m）
    m1_score = 0.0
    if adx_m1 >= 16.0:
        m1_score += min(0.2, (adx_m1 - 16.0) * 0.01)
    if (action == "OPEN_LONG" and slope_pips > 0.0) or (action == "OPEN_SHORT" and slope_pips < 0.0):
        m1_score += min(0.16, abs(slope_pips) * 0.02)
    if abs(roc5) > 0 or abs(roc10) > 0:
        m1_score += min(0.12, (abs(roc5) + abs(roc10)) * 0.005)
    if bbw <= 0.0016:
        m1_score -= 0.05
    if vol_5m < 0.45:
        m1_score -= 0.06
    mult_conf += m1_score

    # CCI 極端は反動警戒
    if cci >= 140 or cci <= -140:
        mult_conf -= 0.05

    # BBW と VWAP 乖離でリスクリワード調整（頻度は落とさずTP側を調整）
    if bbw <= 0.0016:
        mult_tp -= 0.04
    elif bbw >= 0.004:
        mult_tp += 0.04
    if action == "OPEN_LONG" and vwap_gap > 0.0:
        mult_conf += 0.02
    elif action == "OPEN_SHORT" and vwap_gap < 0.0:
        mult_conf += 0.02

    # セッションバイアス（ロンドン/NYは緩め、アジアは軽く絞る）
    now = datetime.datetime.utcnow()
    hour = now.hour
    session = "asia"
    if 7 <= hour < 17:
        session = "london"
    elif 17 <= hour < 23:
        session = "ny"
    if session in {"london", "ny"}:
        mult_conf += 0.04
        mult_tp += 0.02
    else:
        if abs(vwap_gap) < 0.8 and bbw <= 0.0025:
            mult_conf -= 0.04
        else:
            mult_conf -= 0.01

    # クラスタ/VWAP振り分け: 近接なら逆張り系を優遇、遠いならブレイク系を優遇（拒否せずスケールのみ）
    mean_rev_strats = {"BB_RSI", "RangeFader", "pullback_s5", "pullback_scalp", "pullback_runner_s5"}
    breakout_strats = {"TrendMA", "Donchian55", "LondonMomentum", "SqueezeBreak", "ImpulseBreak", "impulse_break_s5"}
    strat = str(sig.get("strategy") or sig.get("tag") or "").strip()
    cluster_gap = cluster_high if action == "OPEN_LONG" else cluster_low
    if cluster_gap > 0:
        if cluster_gap < 3.0:
            if strat in mean_rev_strats:
                mult_conf += 0.05
            elif strat in breakout_strats:
                mult_conf -= 0.06
        elif cluster_gap > 7.0:
            if strat in breakout_strats:
                mult_conf += 0.06
            elif strat in mean_rev_strats:
                mult_conf -= 0.05

    # MA整列直前チェックでサイズ感（confidence）を微調整
    aligned = (action == "OPEN_LONG" and ma10 > ma20) or (action == "OPEN_SHORT" and ma10 < ma20)
    if aligned and gap_pips >= 3.0 and adx_m1 >= 14.0:
        mult_conf += 0.05
    elif not aligned and gap_pips >= 2.0:
        mult_conf -= 0.08

    # パターンヒント（三角/矩形）はブレイク系を優遇、逆張り系は抑制（拒否はしない）
    pattern_hint = str(sig.get("pattern") or sig.get("pattern_hint") or "").lower()
    if "triangle" in pattern_hint or "sym_triangle" in pattern_hint or "range" in pattern_hint or "box" in pattern_hint:
        if strat in breakout_strats:
            mult_conf += 0.04
        elif strat in mean_rev_strats:
            mult_conf -= 0.04

    # MTF方向コンフルエンス（M5/H1/H4）
    mtf_scores = []
    for fac, adx_floor, slope_floor in (
        (fac_m5, 13.0, 0.5),
        (fac_h1, 16.0, 0.4),
        (fac_h4, 18.0, 0.3),
    ):
        mtf_scores.append(_mtf_dir_score(fac, action, adx_floor=adx_floor, slope_floor=slope_floor))
    mtf_score = sum(mtf_scores) / max(1, len([s for s in mtf_scores if s is not None]))
    if mtf_score > 0.6:
        mult_conf += 0.06
        mult_tp += 0.05
    elif mtf_score < 0.2:
        mult_conf -= 0.08
        mult_tp -= 0.06

    mult_conf = max(0.65, min(1.25, mult_conf))
    mult_tp = max(0.7, min(1.2, mult_tp))

    conf = int(max(0, min(100, conf * mult_conf)))
    sig["confidence"] = conf
    if tp is not None:
        sig["tp_pips"] = round(max(0.5, tp * mult_tp), 2)
    notes = sig.get("notes") or {}
    if not isinstance(notes, dict):
        notes = {}
    notes.update(
        {
            "tech_conf_mult": round(mult_conf, 3),
            "tech_tp_mult": round(mult_tp, 3),
            "ichimoku_pos": round(cloud_pos, 3),
            "cluster_high": round(cluster_high, 3),
            "cluster_low": round(cluster_low, 3),
            "mtf_score": round(mtf_score, 3),
            "roc5": round(roc5, 3),
            "roc10": round(roc10, 3),
            "cci": round(cci, 3),
            "bbw": round(bbw, 4),
            "vwap_gap": round(vwap_gap, 3),
            "m1_score": round(m1_score, 3),
            "session": session,
            "ma_gap": round(gap_pips, 2),
            # Logging-only: wick/hit stats for後分析（シグナル判断には未使用）
            "upper_wick": round(_safe_float(fac_m1.get("upper_wick_avg_pips"), 0.0), 3),
            "lower_wick": round(_safe_float(fac_m1.get("lower_wick_avg_pips"), 0.0), 3),
            "high_hits": round(_safe_float(fac_m1.get("high_hits"), 0.0), 2),
            "low_hits": round(_safe_float(fac_m1.get("low_hits"), 0.0), 2),
        }
    )
    sig["notes"] = notes
    return sig


def _frontload_plan(base_plan: Sequence[float], target_first: float) -> tuple[float, ...]:
    plan = _normalize_plan(base_plan)
    if len(plan) == 1:
        return plan
    target_first = max(0.12, min(target_first, 0.8))
    rest_sum = sum(plan[1:])
    if rest_sum <= 0:
        return (1.0,)
    scale = max(0.0, 1.0 - target_first) / rest_sum
    adjusted = [target_first]
    for frac in plan[1:]:
        adjusted.append(max(0.0, frac * scale))
    total = sum(adjusted)
    if total <= 0:
        return plan
    adjusted = [val / total for val in adjusted]
    diff = 1.0 - sum(adjusted)
    adjusted[-1] = max(0.0, adjusted[-1] + diff)
    rounded = [round(val, 4) for val in adjusted]
    diff_round = round(1.0 - sum(rounded), 4)
    rounded[-1] = round(max(0.0, rounded[-1] + diff_round), 4)
    return tuple(rounded)


MIN_MACRO_STAGE_LOT = 0.01  # smaller floor to let macro trickle-in
MAX_MICRO_STAGE_LOT = 0.02  # cap micro scaling when momentum signals fire repeatedly
MIN_MACRO_STAGE_LOT = 0.01  # ensure初回でも0.01lotは投下
MIN_SCALP_STAGE_LOT = 0.01  # 最低でも0.01lot確保
REENTRY_EXTRA_LOT = {
    "macro": _safe_env_float("STAGE_REENTRY_EXTRA_MACRO", 0.04, low=0.0, high=0.5),
    "micro": _safe_env_float("STAGE_REENTRY_EXTRA_MICRO", 0.03, low=0.0, high=0.3),
    "scalp": _safe_env_float("STAGE_REENTRY_EXTRA_SCALP", 0.06, low=0.0, high=0.6),
}
DEFAULT_COOLDOWN_SECONDS = 180
RANGE_COOLDOWN_SECONDS = 420
GPT_MIN_INTERVAL_SECONDS = 180
MIN_MACRO_TOTAL_LOT = _safe_env_float("MIN_MACRO_TOTAL_LOT", 0.02, low=0.0, high=3.0)
TARGET_MACRO_MARGIN_RATIO = _safe_env_float("TARGET_MACRO_MARGIN_RATIO", 0.7, low=0.0, high=0.95)
MACRO_MARGIN_SAFETY_BUFFER = _safe_env_float("MACRO_MARGIN_SAFETY_BUFFER", 0.1, low=0.0, high=0.5)

FORCE_SCALP_MODE = os.getenv("SCALP_FORCE_ALWAYS", "0").strip().lower() not in {"", "0", "false", "no"}
if FORCE_SCALP_MODE:
    logging.warning("[FORCE_SCALP] mode enabled")

RELAX_GPT_ALLOWLIST = True  # GPT は順位付けのみ利用し、評価フィルタには使わない
DISABLE_WAIT_GUARD = os.getenv("DISABLE_WAIT_GUARD", "0").strip().lower() not in {"", "0", "false", "no"}


def _session_bucket(now: datetime.datetime) -> str:
    """Rough session bucket in UTC."""
    hour = now.hour
    if 7 <= hour < 17:
        return "london"
    if 17 <= hour < 23:
        return "ny"
    return "asia"


def _shock_state(fac_m1: dict) -> dict[str, object]:
    """Compute 1H shock strength from recent M1 candles."""
    candles = fac_m1.get("candles") or []
    if not isinstance(candles, list) or len(candles) < 20:
        return {"strength": 0.0, "down": False, "up": False}
    sample = candles[-60:]
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for c in sample:
        try:
            closes.append(float(c.get("close")))
            highs.append(float(c.get("high", c.get("h", 0.0))))
            lows.append(float(c.get("low", c.get("l", 0.0))))
        except Exception:
            continue
    if len(closes) < 10 or not highs or not lows:
        return {"strength": 0.0, "down": False, "up": False}
    delta = closes[-1] - closes[0]
    rng = max(highs) - min(lows)
    strength = abs(delta) / max(rng, 1e-6)
    return {
        "strength": strength,
        "down": delta < -0.001 and strength >= 0.7,
        "up": delta > 0.001 and strength >= 0.7,
    }


def _strategy_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("trend", "momentum", "impulse")):
        return "trend"
    if any(k in n for k in ("break", "burst", "retest", "donchian")):
        return "breakout"
    if any(k in n for k in ("range", "vwap", "bb", "revert", "mean", "magnet")):
        return "range"
    if any(k in n for k in ("scalp", "micro", "onepip", "s1")):
        return "scalp"
    return "other"


def _local_strategy_ranking(
    *,
    strategies: list[str],
    fac_m1: dict,
    fac_h4: dict,
    range_ctx,
    session_bucket: str,
    last_gpt_mode: str | None,
    last_focus: str | None,
) -> list[str]:
    """Rank strategies locally using regime/volatility/context without GPT."""
    adx_m1 = float(fac_m1.get("adx") or 0.0)
    adx_h4 = float(fac_h4.get("adx") or 0.0)
    atr = float(fac_m1.get("atr_pips") or 0.0)
    vol_5m = float(fac_m1.get("vol_5m") or 0.0)
    bbw = float(fac_m1.get("bbw") or 0.0)
    ma10 = fac_m1.get("ma10")
    ma20 = fac_m1.get("ma20")
    slope = 0.0
    if ma10 is not None and ma20 is not None:
        try:
            slope = float(ma10) - float(ma20)
        except Exception:
            slope = 0.0
    slope_pips = slope / PIP_VALUE
    range_active = bool(getattr(range_ctx, "active", False))
    compression = float((range_ctx.metrics or {}).get("compression_ratio", 0.0) if range_ctx else 0.0)
    scores: dict[str, float] = {}
    for name in strategies:
        cat = _strategy_category(name)
        score = 0.05  # baseline to avoid zero-score drops
        # Base: trend strength
        if cat in {"trend", "breakout"}:
            score += 0.8 * (adx_h4 / 30.0) + 0.6 * (adx_m1 / 30.0)
            score += 0.2 * max(0.0, min(3.0, abs(slope_pips)))
        if cat == "range":
            score += 0.5 if range_active else -0.4
            score += 0.2 * (1.0 - min(1.0, compression))
            score += 0.15 * max(0.0, 0.4 - bbw)
        if cat == "scalp":
            score += 0.4 * (vol_5m / 1.5) + 0.3 * (atr / 2.0)
            score += 0.2 * max(0.0, 0.5 - bbw)
        if cat == "breakout":
            score += 0.4 * (vol_5m / 1.2) + 0.3 * (atr / 2.5)
        if cat == "other":
            score += 0.2 * (atr / 2.0) + 0.2 * (vol_5m / 1.0)
        # Session bias
        if session_bucket in {"london", "ny"} and cat in {"trend", "breakout"}:
            score += 0.3
        if session_bucket == "asia" and cat == "range":
            score += 0.3
        # Range suppression for trends
        if range_active and cat in {"trend", "breakout"}:
            score -= 0.6
        # Last GPT mode/focus hint: keep some continuity
        if last_gpt_mode and str(last_gpt_mode).lower().startswith("range") and cat == "range":
            score += 0.2
        if last_focus and last_focus == "micro" and "micro" in name.lower():
            score += 0.1
        scores[name] = score
    if not scores:
        return strategies
    ranked = sorted(strategies, key=lambda n: scores.get(n, 0.0), reverse=True)
    # If all scores tie/very low, still return full list (top N fallback not needed because we keep all)
    return ranked


def _reset_strategy_registry() -> None:
    """Populate STRATEGIES/POCKET maps soランキングが空にならないようにする。"""
    STRATEGIES.clear()
    TREND_STRATEGIES.clear()
    RANGE_STRATEGIES.clear()
    MOMENTUM_STRATEGIES.clear()
    POCKET_STRATEGY_MAP.clear()

    strategy_classes = [
        TrendMA,
        Donchian55,
        BBRsi,
        BBRsiFast,
        RangeFader,
        M1Scalper,
        PulseBreak,
        ImpulseRetraceScalp,
        MomentumBurstMicro,
        MicroMomentumStack,
        MicroPullbackEMA,
        MicroLevelReactor,
        MicroRangeBreak,
        MicroVWAPBound,
        TrendMomentumMicro,
        MicroVWAPRevert,
        VolCompressionBreak,
        MomentumPulse,
    ]
    for cls in strategy_classes:
        name = getattr(cls, "name", cls.__name__)
        STRATEGIES[name] = cls
        pocket = getattr(cls, "pocket", "")
        if pocket:
            POCKET_STRATEGY_MAP.setdefault(pocket, set()).add(name)
        cat = _strategy_category(name)
        if cat == "range":
            RANGE_STRATEGIES.add(name)
        if cat in {"trend", "breakout"}:
            TREND_STRATEGIES.add(name)
            MOMENTUM_STRATEGIES.add(name)
        elif cat == "scalp":
            MOMENTUM_STRATEGIES.add(name)


# 起動時に一度だけ初期化
_reset_strategy_registry()

def _select_worker_targets(
    fac_m1: dict,
    fac_m5: Optional[dict],
    fac_h1: Optional[dict],
    fac_h4: dict,
    now: datetime.datetime,
) -> set[str]:
    """Score workers based on市況 (multi-TF) and pick top-N."""

    def _atr_pips(factors: Optional[dict]) -> float:
        if not factors:
            return 0.0
        atr_val = factors.get("atr_pips")
        if atr_val is None:
            atr_val = (factors.get("atr") or 0.0) * 100.0  # fallback from raw ATR
        try:
            return float(atr_val or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _vol_5m(factors: Optional[dict]) -> float:
        if not factors:
            return 0.0
        # prefer precomputed value if present
        try:
            vol = float(factors.get("vol_5m") or 0.0)
        except (TypeError, ValueError):
            vol = 0.0
        if vol > 0:
            return vol
        # fallback: compute simple high/low range over last ~5 M1 candles
        candles = factors.get("candles")
        if isinstance(candles, list) and len(candles) >= 5:
            highs = []
            lows = []
            for entry in candles[-5:]:
                try:
                    highs.append(float(entry.get("high")))
                    lows.append(float(entry.get("low")))
                except Exception:
                    continue
            if highs and lows:
                return (max(highs) - min(lows)) * 100.0
        # fallback: use recent tick summary (range over recent window)
        summary = factors.get("recent_tick_summary") or {}
        try:
            high_mid = float(summary.get("high_mid") or 0.0)
            low_mid = float(summary.get("low_mid") or 0.0)
            if high_mid and low_mid:
                return (high_mid - low_mid) * 100.0
        except Exception:
            pass
        return 0.0

    def _ma_gap(factors: Optional[dict]) -> float:
        if not factors:
            return 0.0
        try:
            return abs(float(factors.get("ma10") or 0.0) - float(factors.get("ma20") or 0.0))
        except Exception:
            return 0.0

    atr_m1 = _atr_pips(fac_m1)
    atr_m5 = _atr_pips(fac_m5)
    atr_h1 = _atr_pips(fac_h1)
    atr_pips = atr_m1 or atr_m5 or atr_h1 or 0.0
    vol_5m = _vol_5m(fac_m1)
    vol_m5 = _vol_5m(fac_m5)
    adx_h4 = float(fac_h4.get("adx") or 0.0)
    adx_h1 = float((fac_h1 or {}).get("adx") or 0.0)
    ma_gap_h4 = _ma_gap(fac_h4)
    ma_gap_h1 = _ma_gap(fac_h1)
    session = _session_bucket(now)
    rsi_m1 = float(fac_m1.get("rsi") or 50.0)
    bbw_m1 = float(fac_m1.get("bbw") or 0.0)
    try:
        close_m1 = float(fac_m1.get("close")) if fac_m1.get("close") is not None else None
        ma20_m1 = float(fac_m1.get("ma20")) if fac_m1.get("ma20") is not None else None
    except (TypeError, ValueError):
        close_m1 = None
        ma20_m1 = None
    momentum_m1 = 0.0
    if close_m1 is not None and ma20_m1 is not None:
        momentum_m1 = close_m1 - ma20_m1

    high_vol = max(atr_m1, atr_m5, atr_h1) >= 2.0 or max(vol_5m, vol_m5) >= 1.5
    mid_vol = max(atr_m1, atr_m5, atr_h1) >= 1.3 or max(vol_5m, vol_m5) >= 1.0
    low_vol = max(atr_m1, atr_m5, atr_h1) <= 1.0 and max(vol_5m, vol_m5) <= 0.7
    trending = (adx_h4 >= 18.0 or ma_gap_h4 >= 0.12) or (adx_h1 >= 16.0 or ma_gap_h1 >= 0.07)
    range_like = (adx_h4 <= 15.0 and max(vol_5m, vol_m5) <= 0.9) and adx_h1 <= 14.0
    compression = bbw_m1 <= 0.002 or (vol_5m <= 0.6 and atr_pips <= 1.2)
    spike_like = high_vol or abs(momentum_m1) >= 0.015
    soft_range = (
        (adx_h4 <= 22.0 and adx_h1 <= 20.0)
        and vol_5m <= 1.4
        and bbw_m1 <= 0.004
    ) or (compression and vol_5m <= 1.2)

    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}

    def bump(name: str, score: float, reason: str) -> None:
        if name not in WORKER_SERVICES:
            return
        prev = scores.get(name, 0.0)
        scores[name] = prev + score
        if prev < score:
            reasons[name] = reason

    # Baseline set
    bump("fast_scalp", 0.9, "baseline_scalp")
    bump("mm_lite", 0.3, "baseline_mm_lite")

    # Macro/trend
    if trending:
        bump("trend_h1", 0.5, "trend_confirmed")
    if high_vol and trending:
        bump("mtf_breakout", 0.8, "high_vol_trend")
    if session in {"london", "ny"} and mid_vol:
        bump("london_momentum", 0.7, f"session_{session}")

    # Range/低ボラ
    if low_vol or range_like:
        bump("pullback_scalp", 0.9, "range_low_vol")
        bump("vol_squeeze", 0.45, "range_compression")
        bump("pullback_s5", 0.6, "range_s5")
        bump("pullback_runner_s5", 0.55, "range_runner")
        bump("vwap_magnet_s5", 0.55, "vwap_range")
        bump("onepip_maker_s1", 0.6, "low_vol_onepip")
        if compression:
            bump("squeeze_break_s5", 0.5, "compression_break")
    elif soft_range:
        bump("pullback_scalp", 0.78, "soft_range")
        bump("pullback_s5", 0.65, "soft_range")
        bump("vwap_magnet_s5", 0.6, "soft_range")
        bump("onepip_maker_s1", 0.6, "soft_range")
        bump("vol_squeeze", 0.45, "soft_range")

    # Spike/impulse style entries
    if spike_like:
        bump("impulse_break_s5", 0.45, "impulse_break")
        bump("impulse_momentum_s5", 0.4, "impulse_momentum")
        bump("impulse_retest_s5", 0.35, "impulse_retest")
        bump("stop_run_reversal", 0.35, "stop_run")
        bump("mirror_spike", 0.4, "spike_reversal")
        bump("mirror_spike_s5", 0.35, "spike_reversal_s5")
        bump("mirror_spike_tight", 0.35, "spike_reversal_tight")

    # Session open bias
    if now.minute < 20:
        bump("session_open", 0.3, f"session_{session}_open")

    # Safety: avoid overloading tiny VM
    if WORKER_AUTOCONTROL_LIMIT <= 0:
        limit = len(scores)
    else:
        limit = max(1, min(len(scores), WORKER_AUTOCONTROL_LIMIT))
    picked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    selected = {name for name, _ in picked}

    # Diversity nudge: 低〜中ボラではレンジ系が一つも選ばれていない場合に補充する
    range_workers = {
        "pullback_scalp",
        "pullback_s5",
        "vwap_magnet_s5",
        "onepip_maker_s1",
        "vol_squeeze",
        "pullback_runner_s5",
        "squeeze_break_s5",
    }
    if (low_vol or range_like or soft_range) and not (selected & range_workers):
        candidates = [(name, scores[name]) for name in range_workers if name in scores]
        if candidates:
            best_name, best_score = max(candidates, key=lambda x: x[1])
            if len(selected) < limit:
                selected.add(best_name)
            else:
                lowest_name, lowest_score = min(picked, key=lambda x: x[1])
                # 低スコア枠と入れ替えてレンジ系を確保する
                if best_score >= lowest_score * 0.7:
                    selected.discard(lowest_name)
                    selected.add(best_name)
    logging.info(
        "[WORKER_CTL] plan session=%s atr(m1/m5/h1)=%.2f/%.2f/%.2f vol=%.2f/%.2f adx(h4/h1)=%.1f/%.1f selected=%s reasons=%s",
        session,
        atr_m1,
        atr_m5,
        atr_h1,
        vol_5m,
        vol_m5,
        adx_h4,
        adx_h1,
        ",".join(sorted(selected)),
        {k: reasons.get(k, "") for k in selected},
    )
    return selected


async def _systemctl(action: str, service: str) -> bool:
    if not WORKER_SYSTEMCTL_ENABLED:
        if not getattr(_systemctl, "_warned", False):
            logging.warning("[WORKER_CTL] systemctl disabled (WORKER_SYSTEMCTL=0); skip %s %s", action, service)
            _systemctl._warned = True  # type: ignore[attr-defined]
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "systemctl",
            action,
            service,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("[WORKER_CTL] %s %s failed: %s", action, service, exc)
        return False


async def _reconcile_worker_services(previous: set[str], desired: set[str]) -> None:
    to_start = desired - previous
    to_stop = previous - desired
    if not to_start and not to_stop:
        return
    now = datetime.datetime.utcnow()
    missing_units: list[str] = []
    for svc in to_start:
        unit = WORKER_SERVICES.get(svc)
        if not unit:
            missing_units.append(svc)
            continue
        ok = await _systemctl("start", unit)
        if ok:
            logging.info("[WORKER_CTL] started %s (%s)", svc, unit)
            log_metric("worker_start", 1.0, tags={"service": svc, "unit": unit, "result": "started"}, ts=now)
        else:
            logging.warning("[WORKER_CTL] failed to start %s (%s)", svc, unit)
            log_metric("worker_start", 0.0, tags={"service": svc, "unit": unit, "result": "failed"}, ts=now)
    for svc in to_stop:
        unit = WORKER_SERVICES.get(svc)
        if not unit:
            continue
        ok = await _systemctl("stop", unit)
        if ok:
            logging.info("[WORKER_CTL] stopped %s (%s)", svc, unit)
            log_metric("worker_stop", 1.0, tags={"service": svc, "unit": unit, "result": "stopped"}, ts=now)
        else:
            logging.warning("[WORKER_CTL] failed to stop %s (%s)", svc, unit)
            log_metric("worker_stop", 0.0, tags={"service": svc, "unit": unit, "result": "failed"}, ts=now)
    if missing_units:
        logging.warning("[WORKER_CTL] units missing for selected services: %s", ",".join(sorted(missing_units)))
        for svc in missing_units:
            log_metric("worker_missing_unit", 1.0, tags={"service": svc}, ts=now)


class GPTDecisionState:
    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._latest: Optional[dict] = None
        self._signature: Optional[str] = None
        self._updated_at: Optional[datetime.datetime] = None

    async def update(self, signature: str, decision: dict) -> None:
        async with self._cond:
            self._latest = dict(decision)
            self._latest["model_used"] = decision.get("model_used")
            self._signature = signature
            self._updated_at = datetime.datetime.utcnow()
            self._cond.notify_all()

    async def get_latest(self) -> Tuple[dict, Optional[str], Optional[datetime.datetime]]:
        async with self._cond:
            while self._latest is None:
                await self._cond.wait()
            return dict(self._latest), self._signature, self._updated_at

    async def needs_refresh(self, signature: str, min_interval: int) -> bool:
        async with self._cond:
            if self._signature is None:
                return True
            if self._signature != signature:
                return True
            if not self._updated_at:
                return True
            if (datetime.datetime.utcnow() - self._updated_at).total_seconds() >= min_interval:
                return True
            return False

    async def wait_for_signature(self, signature: str, timeout: Optional[float]) -> dict:
        deadline = None if timeout is None else datetime.datetime.utcnow() + datetime.timedelta(seconds=timeout)
        async with self._cond:
            while True:
                if self._signature == signature and self._latest is not None:
                    return dict(self._latest)
                if deadline is not None:
                    remaining = (deadline - datetime.datetime.utcnow()).total_seconds()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                else:
                    await self._cond.wait()


class GPTRequestManager:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Tuple[str, dict]] = asyncio.Queue()
        self._pending: set[str] = set()
        self._lock = asyncio.Lock()

    async def submit(self, signature: str, payload: dict) -> None:
        async with self._lock:
            if signature in self._pending:
                return
            self._pending.add(signature)
        await self._queue.put((signature, payload))

    async def get(self) -> Tuple[str, dict]:
        return await self._queue.get()

    async def task_done(self, signature: str) -> None:
        async with self._lock:
            self._pending.discard(signature)
        self._queue.task_done()
# In range mode, allow mean‑reversionと軽量スキャルのみを通す
ALLOWED_RANGE_STRATEGIES = {
    "BB_RSI",
    "BB_RSI_Fast",
    "RangeFader",
    "M1Scalper",
    "MicroRangeBreak",
    "MicroVWAPRevert",
    "MicroVWAPBound",
    # レンジ中でもロングを通したいエントリー系を許可
    "MomentumPulse",
    "VolCompressionBreak",
    "MicroPullbackEMA",
    "MicroLevelReactor",
    "MicroMomentumStack",
    # S5/スカルプ系（レンジでも評価を通す）
    "vwap_magnet_s5",
    "pullback_runner_s5",
    "squeeze_break_s5",
    "mirror_spike",
    "mirror_spike_tight",
    "mirror_spike_s5",
    "pullback_s5",
    "pullback_scalp",
    "impulse_break_s5",
    "impulse_momentum_s5",
    "impulse_retest_s5",
    "onepip_maker_s1",
}
SOFT_RANGE_SUPPRESS_STRATEGIES = {"TrendMA", "Donchian55"}
LOW_TREND_ADX_THRESHOLD = 18.0
LOW_TREND_SLOPE_THRESHOLD = 0.00035
LOW_TREND_WEIGHT_CAP = 0.35
SOFT_RANGE_SCORE_MIN = 0.52
SOFT_RANGE_COMPRESSION_MIN = 0.50
SOFT_RANGE_VOL_MIN = 0.35
SOFT_RANGE_WEIGHT_CAP = 0.25
SOFT_RANGE_ADX_BUFFER = 6.0
RANGE_ENTRY_CONFIRMATIONS = 1
RANGE_EXIT_CONFIRMATIONS = 3
RANGE_GUARD_RANGE_CONFIRM = int(os.getenv("RANGE_GUARD_RANGE_CONFIRM", "2"))
RANGE_GUARD_TREND_CONFIRM = int(os.getenv("RANGE_GUARD_TREND_CONFIRM", "2"))
RANGE_GUARD_NEUTRAL_CONFIRM = int(os.getenv("RANGE_GUARD_NEUTRAL_CONFIRM", "2"))
RANGE_GUARD_RANGE_SCORE_MICRO = _safe_env_float(
    "RANGE_GUARD_RANGE_SCORE_MICRO", 0.75, low=0.0, high=1.0
)
RANGE_GUARD_TREND_SCORE_MICRO = _safe_env_float(
    "RANGE_GUARD_TREND_SCORE_MICRO", 0.45, low=0.0, high=1.0
)
RANGE_GUARD_ADX_MICRO = _safe_env_float(
    "RANGE_GUARD_ADX_MICRO", 14.0, low=5.0, high=40.0
)
RANGE_GUARD_BBW_MICRO = _safe_env_float(
    "RANGE_GUARD_BBW_MICRO", 0.14, low=0.0, high=1.0
)
RANGE_GUARD_ATR_MICRO = _safe_env_float(
    "RANGE_GUARD_ATR_MICRO", 1.3, low=0.1, high=50.0
)
RANGE_GUARD_BBW_PIPS_MICRO = _safe_env_float(
    "RANGE_GUARD_BBW_PIPS_MICRO", 4.0, low=0.0, high=50.0
)
RANGE_GUARD_RANGE_SCORE_SCALP = _safe_env_float(
    "RANGE_GUARD_RANGE_SCORE_SCALP", 0.75, low=0.0, high=1.0
)
RANGE_GUARD_TREND_SCORE_SCALP = _safe_env_float(
    "RANGE_GUARD_TREND_SCORE_SCALP", 0.45, low=0.0, high=1.0
)
RANGE_GUARD_ADX_SCALP = _safe_env_float(
    "RANGE_GUARD_ADX_SCALP", 14.0, low=5.0, high=40.0
)
RANGE_GUARD_BBW_SCALP = _safe_env_float(
    "RANGE_GUARD_BBW_SCALP", 0.14, low=0.0, high=1.0
)
RANGE_GUARD_ATR_SCALP = _safe_env_float(
    "RANGE_GUARD_ATR_SCALP", 1.3, low=0.1, high=50.0
)
RANGE_GUARD_BBW_PIPS_SCALP = _safe_env_float(
    "RANGE_GUARD_BBW_PIPS_SCALP", 4.0, low=0.0, high=50.0
)
RANGE_GUARD_RANGE_SCORE_GLOBAL = _safe_env_float(
    "RANGE_GUARD_RANGE_SCORE_GLOBAL", 0.75, low=0.0, high=1.0
)
RANGE_GUARD_TREND_SCORE_GLOBAL = _safe_env_float(
    "RANGE_GUARD_TREND_SCORE_GLOBAL", 0.45, low=0.0, high=1.0
)
RANGE_GUARD_ADX_GLOBAL = _safe_env_float(
    "RANGE_GUARD_ADX_GLOBAL", 14.0, low=5.0, high=40.0
)
RANGE_GUARD_BBW_GLOBAL = _safe_env_float(
    "RANGE_GUARD_BBW_GLOBAL", 0.14, low=0.0, high=1.0
)
RANGE_GUARD_ATR_GLOBAL = _safe_env_float(
    "RANGE_GUARD_ATR_GLOBAL", 1.3, low=0.1, high=50.0
)
RANGE_GUARD_BBW_PIPS_GLOBAL = _safe_env_float(
    "RANGE_GUARD_BBW_PIPS_GLOBAL", 4.0, low=0.0, high=50.0
)
RANGE_MIN_ACTIVE_SECONDS = 120
RANGE_ENTRY_SCORE_FLOOR = 0.62
RANGE_EXIT_SCORE_CEIL = 0.56
RANGE_BREAK_MOMENTUM_RELEASE = 0.015  # 1.5 pips
RANGE_BREAK_ATR_MIN = 3.6
RANGE_BREAK_VOL_SPIKE = 1.35
RANGE_BREAK_RELEASE_SECONDS = 90
RANGE_BREAK_MACRO_HOLD_SECONDS = 180
MICRO_BREAKOUT_WEIGHT_FLOOR = 0.45
STAGE_RESET_GRACE_SECONDS = 180
if os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"", "0", "false", "no"}:
    STAGE_RESET_GRACE_SECONDS = 60
RANGE_SCALP_ATR_MIN = 0.7
RANGE_SCALP_VOL_MIN = 0.4
RANGE_SCALP_MAX_MOMENTUM = 0.0022
RANGE_FADER_MIN_RR = 1.12
RANGE_FADER_MAX_RR = 1.32
RANGE_FADER_RANGE_CONF_SCALE = 0.82

_HARD_BASE_RISK_CAP = 0.15   # 最大ベースリスク 15%（証拠金を積極活用）
_HARD_MAX_RISK_CAP = 0.35    # ダイナミックリスク上限 35%
_RANGE_RISK_CAP = 0.02       # レンジ時も 2% まで許容
_SQUEEZE_RISK_CAP = 0.02     # 個別ドローダウン時の下限を緩めて回転率を確保

try:
    _BASE_RISK_PCT = float(get_secret("risk_pct"))
    if _BASE_RISK_PCT <= 0:
        _BASE_RISK_PCT = 0.1
except Exception:
    _BASE_RISK_PCT = 0.1
_BASE_RISK_PCT = max(_BASE_RISK_PCT, 0.1)
try:
    _MAX_RISK_PCT = float(get_secret("risk_pct_max"))
    if _MAX_RISK_PCT < _BASE_RISK_PCT:
        _MAX_RISK_PCT = _BASE_RISK_PCT
    elif _MAX_RISK_PCT > 0.3:
        _MAX_RISK_PCT = 0.3
except Exception:
    _MAX_RISK_PCT = _BASE_RISK_PCT

_BASE_RISK_PCT = min(_BASE_RISK_PCT, _HARD_BASE_RISK_CAP)
_MAX_RISK_PCT = max(_BASE_RISK_PCT, min(_MAX_RISK_PCT, _HARD_MAX_RISK_CAP))


def _hashable_float(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _compute_stage_base(
    *,
    param_snapshot: ParamSnapshot | None,
    range_active: bool,
    focus_tag: str,
    weight_macro: float,
    weight_scalp: float | None,
) -> dict[str, tuple[float, ...]]:
    """
    Build per-pocket stage plans before bias/Advisor adjustments.
    Front-load the first stage (e.g., 45% / 55%) whenトレンドが強いときだけ許容し、
    レンジや高ボラでは分割して次のエントリー枠を残す。
    """
    base = {pocket: _normalize_plan(plan) for pocket, plan in _BASE_STAGE_RATIOS.items()}
    if param_snapshot is None:
        return base

    trend_score = (param_snapshot.adx_m1_score + param_snapshot.adx_h4_score) / 2.0
    vol_high = param_snapshot.volatility_state == "high" or param_snapshot.vol_high_ratio >= 0.3
    tight_liquidity = param_snapshot.liquidity_state == "wide"
    risk = param_snapshot.risk_appetite

    macro_first = base.get("macro", (0.35,))[0]
    micro_first = base.get("micro", (0.3,))[0]
    scalp_first = base.get("scalp", (0.4,))[0]

    if not range_active and trend_score >= 0.6 and risk >= 0.55:
        macro_first = max(macro_first, 0.45)
    elif range_active or trend_score <= 0.45 or tight_liquidity:
        macro_first = min(macro_first, 0.34)
    else:
        macro_first = max(macro_first, 0.38)
    if not range_active and weight_macro >= 0.55:
        macro_first = max(macro_first, 0.42)
    if weight_macro <= 0.25:
        macro_first = min(macro_first, 0.38)
    if focus_tag == "macro" and not range_active:
        macro_first = max(macro_first, 0.4)
    macro_first = _clamp(macro_first, 0.22, 0.65)

    if range_active:
        micro_first = max(micro_first * 0.95, 0.3)
        micro_first = min(micro_first, 0.42)
    elif trend_score >= 0.6:
        micro_first = max(micro_first, 0.4)
    else:
        micro_first = max(micro_first, 0.34)
    micro_first = _clamp(micro_first, 0.22, 0.6)

    scalp_first = max(scalp_first, 0.35)
    if not vol_high and risk >= 0.5 and not tight_liquidity:
        scalp_first = max(scalp_first, 0.45)
    if range_active or vol_high or tight_liquidity:
        scalp_first = min(scalp_first, 0.4)
    if weight_scalp is not None and weight_scalp < 0.18:
        scalp_first = min(scalp_first, 0.38)
    scalp_first = _clamp(scalp_first, 0.25, 0.6)

    base["macro"] = _frontload_plan(_BASE_STAGE_RATIOS["macro"], macro_first)
    base["micro"] = _frontload_plan(_BASE_STAGE_RATIOS["micro"], micro_first)
    base["scalp"] = _frontload_plan(_BASE_STAGE_RATIOS["scalp"], scalp_first)
    return base


def _gpt_payload_signature(
    payload: dict,
    *,
    range_active: bool,
    range_reason: Optional[str],
    soft_range: bool,
) -> str:
    factors_m1 = payload.get("factors_m1") or {}
    factors_m5 = payload.get("factors_m5") or {}
    factors_h1 = payload.get("factors_h1") or {}
    factors_h4 = payload.get("factors_h4") or {}
    factors_d1 = payload.get("factors_d1") or {}
    perf = payload.get("perf") or {}
    perf_snapshot = {}
    for key, value in perf.items():
        if isinstance(value, (int, float)):
            perf_snapshot[key] = _hashable_float(value, 3)
        elif isinstance(value, dict):
            win = value.get("win_rate")
            pf = value.get("pf")
            perf_snapshot[key] = {
                "win_rate": _hashable_float(win, 3),
                "pf": _hashable_float(pf, 3),
            }
    key_data = {
        "reg_macro": payload.get("reg_macro"),
        "reg_micro": payload.get("reg_micro"),
        "event": payload.get("event_soon"),
        "range_active": range_active,
        "range_reason": range_reason,
        "soft_range": soft_range,
        "m1": {
            "close": _hashable_float(factors_m1.get("close"), 4),
            "adx": _hashable_float(factors_m1.get("adx"), 3),
            "rsi": _hashable_float(factors_m1.get("rsi"), 3),
            "atr": _hashable_float(factors_m1.get("atr_pips"), 3),
            "vol": _hashable_float(factors_m1.get("vol_5m"), 3),
            "ma10": _hashable_float(factors_m1.get("ma10"), 4),
            "ma20": _hashable_float(factors_m1.get("ma20"), 4),
        },
        "m5": {
            "close": _hashable_float(factors_m5.get("close"), 4),
            "adx": _hashable_float(factors_m5.get("adx"), 3),
            "rsi": _hashable_float(factors_m5.get("rsi"), 3),
            "ma10": _hashable_float(factors_m5.get("ma10"), 4),
            "ma20": _hashable_float(factors_m5.get("ma20"), 4),
        },
        "h1": {
            "close": _hashable_float(factors_h1.get("close"), 4),
            "adx": _hashable_float(factors_h1.get("adx"), 3),
            "ma10": _hashable_float(factors_h1.get("ma10"), 4),
            "ma20": _hashable_float(factors_h1.get("ma20"), 4),
        },
        "h4": {
            "close": _hashable_float(factors_h4.get("close"), 4),
            "adx": _hashable_float(factors_h4.get("adx"), 3),
            "ma10": _hashable_float(factors_h4.get("ma10"), 4),
            "ma20": _hashable_float(factors_h4.get("ma20"), 4),
        },
        "d1": {
            "close": _hashable_float(factors_d1.get("close"), 4),
            "adx": _hashable_float(factors_d1.get("adx"), 3),
            "ma10": _hashable_float(factors_d1.get("ma10"), 4),
            "ma20": _hashable_float(factors_d1.get("ma20"), 4),
        },
        "perf": perf_snapshot,
    }
    serialized = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _dynamic_risk_pct(
    signals: list[dict],
    range_mode: bool,
    weight_macro: float | None,
    *,
    context: ParamSnapshot | None = None,
    gpt_bias: Dict[str, object] | None = None,
) -> float:
    if not signals or _MAX_RISK_PCT <= _BASE_RISK_PCT:
        return _BASE_RISK_PCT
    if range_mode:
        if context:
            dampen = 0.45 + context.risk_appetite * 0.3
            if context.vol_high_ratio >= 0.3:
                dampen = min(dampen, 0.35)
            dampen = max(0.25, min(dampen, 0.75))
            range_risk = _BASE_RISK_PCT * dampen
        else:
            range_risk = _BASE_RISK_PCT * 0.5
        return max(0.0005, min(range_risk, _RANGE_RISK_CAP))
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
    if context:
        env_score = context.risk_appetite
        score = (score * 0.65) + (env_score * 0.35)
        if context.volatility_state == "high":
            score -= 0.08
        elif context.volatility_state == "low":
            score += 0.05
        if context.liquidity_state == "wide":
            score -= 0.08
        if context.vol_high_ratio >= 0.3:
            score -= 0.12
        score = max(0.0, min(score, 1.0))
    risk_pct = _BASE_RISK_PCT + (_MAX_RISK_PCT - _BASE_RISK_PCT) * score
    if context:
        if context.risk_appetite < 0.25:
            risk_pct = min(risk_pct, _BASE_RISK_PCT * 0.35, _SQUEEZE_RISK_CAP)
        if context.vol_high_ratio >= 0.3:
            risk_pct = min(risk_pct, _BASE_RISK_PCT * 0.4)
    if gpt_bias:
        mode = str(gpt_bias.get("mode") or "").upper()
        risk_flag = str(gpt_bias.get("risk_bias") or "").lower()
        liq_flag = str(gpt_bias.get("liquidity_bias") or "").lower()
        if mode == "DEFENSIVE":
            risk_pct *= 0.75
        elif mode == "TRANSITION":
            risk_pct *= 0.9
        elif mode == "TREND_FOLLOW":
            risk_pct *= 1.05
        elif mode == "RANGE_SCALP":
            risk_pct *= 0.95
        if risk_flag == "high":
            risk_pct *= 1.08
        elif risk_flag == "low":
            risk_pct *= 0.85
        if liq_flag == "tight":
            risk_pct *= 0.9
        elif liq_flag == "loose":
            risk_pct *= 1.03
    risk_pct = min(risk_pct, _HARD_MAX_RISK_CAP)
    return max(0.0005, risk_pct)


async def gpt_worker(
    gpt_state: GPTDecisionState,
    gpt_requests: GPTRequestManager,
) -> None:
    logging.info("[GPT WORKER] started")
    while True:
        signature, payload = await gpt_requests.get()
        try:
            logging.info("[GPT WORKER] processing signature=%s", signature[:10])
            decision = await get_decision(payload)
            await gpt_state.update(signature, decision)
            logging.info(
                "[GPT WORKER] decision ready signature=%s model=%s",
                signature[:10],
                decision.get("model_used", "unknown"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.error(
                "[GPT WORKER] decision failed signature=%s (%s: %s) - keeping previous decision",
                signature[:10],
                type(exc).__name__,
                str(exc) or "no message",
            )
        finally:
            await gpt_requests.task_done(signature)


async def prime_gpt_decision(
    gpt_state: GPTDecisionState,
    gpt_requests: GPTRequestManager,
    *,
    timeout: float = 45.0,
) -> None:
    """Ensure an initial GPT decision is ready before strategy loop starts."""
    start_time = datetime.datetime.utcnow()
    while True:
        factors = all_factors()
        fac_m1 = factors.get("M1")
        fac_h4 = factors.get("H4")
        if (
            not fac_m1
            or not fac_h4
            or not fac_m1.get("close")
            or not fac_h4.get("close")
        ):
            await asyncio.sleep(1.0)
            if (datetime.datetime.utcnow() - start_time).total_seconds() > timeout:
                logging.warning("[GPT PRIME] waiting for factors timed out; retrying")
                start_time = datetime.datetime.utcnow()
            continue

        perf_cache = get_perf()
        event_soon = False
        payload = {
            "ts": datetime.datetime.utcnow().isoformat(timespec="seconds"),
            "reg_macro": classify(fac_h4, "H4", event_mode=event_soon),
            "reg_micro": classify(fac_m1, "M1", event_mode=event_soon),
            "factors_m1": _compact_factors(fac_m1, GPT_FACTOR_KEYS["M1"]),
            "factors_h4": _compact_factors(fac_h4, GPT_FACTOR_KEYS["H4"]),
            "perf": {
                pocket: {
                    key: val
                    for key, val in (metrics or {}).items()
                    if key in GPT_PERF_KEYS and val not in (None, "")
                }
                for pocket, metrics in (perf_cache or {}).items()
                if metrics
            },
            "event_soon": event_soon,
        }
        signature = _gpt_payload_signature(
            payload,
            range_active=False,
            range_reason=None,
            soft_range=False,
        )
        logging.info("[GPT PRIME] submitting initial decision signature=%s", signature[:10])
        await gpt_requests.submit(signature, payload)
        try:
            decision = await gpt_state.wait_for_signature(signature, timeout=timeout)
            logging.info(
                "[GPT PRIME] initial decision ready model=%s",
                decision.get("model_used", "unknown"),
            )
            return
        except asyncio.TimeoutError:
            logging.warning("[GPT PRIME] timeout waiting for initial decision; retrying")
            await asyncio.sleep(5.0)


async def supervised_runner(name: str, coro: asyncio.coroutines.coroutine) -> None:
    logging.info("[SUPERVISOR] %s started", name)
    try:
        await coro
        raise RuntimeError(f"{name} completed unexpectedly")
    except asyncio.CancelledError:
        logging.info("[SUPERVISOR] %s cancelled", name)
        raise
    except Exception:
        logging.exception("[SUPERVISOR] %s crashed", name)
        raise


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


def _macro_pullback_threshold(stage_idx: int, atr_pips: Optional[float] = None) -> tuple[float, float]:
    base = _MACRO_PULLBACK_MIN_RETRACE_PIPS.get(
        stage_idx,
        _MACRO_PULLBACK_MIN_RETRACE_PIPS[max(_MACRO_PULLBACK_MIN_RETRACE_PIPS)],
    )
    atr = float(atr_pips or 0.0)
    if atr > 0.0:
        base = max(base, min(base + 2.0, atr * 0.6 + 2.0))
    max_depth = max(base + 4.0, min(_MACRO_PULLBACK_MAX_RETRACE_PIPS, atr * 1.7 + 5.0))
    return round(base, 2), round(max_depth, 2)


def _dynamic_macro_pullback_pips(
    atr_pips: float,
    momentum: float,
    pullback_floor: float,
    pullback_cap: float,
) -> float:
    atr = max(0.0, atr_pips)
    if atr <= 1.6:
        base = 1.05 + atr * 0.35
    elif atr <= 3.2:
        base = 1.35 + atr * 0.4
    elif atr <= 5.5:
        base = pullback_floor + atr * 0.42
    else:
        base = pullback_floor + atr * 0.48
    if abs(momentum) >= 0.012:
        base *= 0.88
    if abs(momentum) >= 0.02:
        base *= 0.82
    return max(1.15, min(pullback_cap, round(base, 2)))


def _compact_factors(data: Optional[Dict[str, Any]], keys: tuple[str, ...]) -> Dict[str, Any]:
    if not data:
        return {}
    compact: Dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, (int, float)):
            precision = _GPT_FACTOR_PRECISION.get(key, 4)
            try:
                compact[key] = round(float(value), precision)
            except (TypeError, ValueError):
                continue
        else:
            compact[key] = value
    return compact


def _dir_bias(fac_h1: dict[str, Any], fac_h4: dict[str, Any]) -> tuple[int, int, float, float]:
    def _dir_fast(fac: dict[str, Any]) -> int:
        try:
            ma10 = float(fac.get("ma10") or 0.0)
            ma20 = float(fac.get("ma20") or 0.0)
        except Exception:
            return 0
        if ma10 > ma20:
            return 1
        if ma10 < ma20:
            return -1
        return 0

    def _dir_slow(fac: dict[str, Any]) -> int:
        try:
            ma20 = float(fac.get("ma20") or 0.0)
            ma50 = float(fac.get("ma50") or fac.get("ema50") or 0.0)
        except Exception:
            return 0
        if ma20 > ma50:
            return 1
        if ma20 < ma50:
            return -1
        return 0

    bias_h1 = _dir_fast(fac_h1 or {})
    bias_h4 = _dir_slow(fac_h4 or {})
    try:
        adx_h1 = float(fac_h1.get("adx") or 0.0)
    except Exception:
        adx_h1 = 0.0
    try:
        adx_h4 = float(fac_h4.get("adx") or 0.0)
    except Exception:
        adx_h4 = 0.0
    return bias_h1, bias_h4, adx_h1, adx_h4


def _macro_pullback_ready(
    action: str,
    stage_idx: int,
    fac_m1: dict[str, float],
    fac_h4: dict[str, float],
    open_info: dict[str, float],
    atr_pips: Optional[float],
) -> Optional[float]:
    if action not in {"OPEN_LONG", "OPEN_SHORT"}:
        return None
    price = fac_m1.get("close")
    if price is None:
        return None
    min_retrace, max_retrace = _macro_pullback_threshold(stage_idx, atr_pips)
    ema20_m1 = fac_m1.get("ema20")
    if ema20_m1 is None:
        ema20_m1 = fac_m1.get("ma20")
    ma10_h4 = fac_h4.get("ma10")
    ma20_h4 = fac_h4.get("ma20")
    adx_h4 = float(fac_h4.get("adx", 0.0) or 0.0)
    ma10_m1 = fac_m1.get("ma10")
    ma20_m1 = fac_m1.get("ma20")

    if action == "OPEN_LONG":
        avg_price = open_info.get("long_avg_price") or open_info.get("avg_price")
        if not avg_price:
            return None
        retrace = (avg_price - price) / PIP
        if retrace < min_retrace or retrace > max_retrace:
            return None
        if ma10_h4 is not None and ma20_h4 is not None and ma10_h4 <= ma20_h4:
            return None
        if adx_h4 < _MACRO_PULLBACK_MIN_ADX:
            return None
        if ema20_m1 is not None:
            ema_gap = max(0.0, (ema20_m1 - price) / PIP)
            if ema_gap > _MACRO_PULLBACK_MAX_EMA_GAP_PIPS:
                return None
        if ma10_m1 is not None and ma20_m1 is not None:
            ma_gap = (ma20_m1 - ma10_m1) / PIP
            if ma_gap > _MACRO_PULLBACK_MAX_MA_SLACK_PIPS:
                return None
        return round(retrace, 2)

    avg_price = open_info.get("short_avg_price") or open_info.get("avg_price")
    if not avg_price:
        return None
    retrace = (price - avg_price) / PIP
    if retrace < min_retrace or retrace > max_retrace:
        return None
    if ma10_h4 is not None and ma20_h4 is not None and ma10_h4 >= ma20_h4:
        return None
    if adx_h4 < _MACRO_PULLBACK_MIN_ADX:
        return None
    if ema20_m1 is not None:
        ema_gap = max(0.0, (price - ema20_m1) / PIP)
        if ema_gap > _MACRO_PULLBACK_MAX_EMA_GAP_PIPS:
            return None
    if ma10_m1 is not None and ma20_m1 is not None:
        ma_gap = (ma10_m1 - ma20_m1) / PIP
        if ma_gap > _MACRO_PULLBACK_MAX_MA_SLACK_PIPS:
            return None
    return round(retrace, 2)


def _stage_conditions_met(
    pocket: str,
    stage_idx: int,
    action: str,
    fac_m1: dict[str, float],
    fac_h4: dict[str, float],
    open_info: dict[str, float],
) -> bool:
    price = fac_m1.get("close")
    avg_price = open_info.get("avg_price", price or 0.0)
    rsi = fac_m1.get("rsi", 50.0)
    adx_h4 = fac_h4.get("adx", 0.0)
    slope_h4 = abs(fac_h4.get("ma20", 0.0) - fac_h4.get("ma10", 0.0))
    atr_raw = fac_m1.get("atr_pips")
    if atr_raw is None:
        atr_raw = (fac_m1.get("atr") or 0.0) * 100
    try:
        atr_pips = float(atr_raw or 0.0)
    except (TypeError, ValueError):
        atr_pips = 0.0

    macro_pullback_retrace: Optional[float] = None
    if pocket == "macro" and stage_idx >= 1:
        macro_pullback_retrace = _macro_pullback_ready(
            action,
            stage_idx,
            fac_m1,
            fac_h4,
            open_info,
            atr_pips,
        )

    buy_floor = RSI_LONG_FLOOR.get(pocket)
    if (
        action == "OPEN_LONG"
        and buy_floor is not None
        and rsi < buy_floor
        and not (pocket == "macro" and macro_pullback_retrace is not None)
    ):
        logging.info(
            "[STAGE] %s buy gating: RSI %.1f below floor %.1f (stage %d).",
            pocket,
            rsi,
            buy_floor,
            stage_idx,
        )
        return False

    sell_cap = RSI_SHORT_CEILING.get(pocket)
    if (
        action == "OPEN_SHORT"
        and sell_cap is not None
        and rsi > sell_cap
        and not (pocket == "macro" and macro_pullback_retrace is not None)
    ):
        logging.info(
            "[STAGE] %s sell gating: RSI %.1f above ceiling %.1f (stage %d).",
            pocket,
            rsi,
            sell_cap,
            stage_idx,
        )
        return False

    min_atr = POCKET_ATR_MIN_PIPS.get(pocket)
    if min_atr is not None and atr_pips < min_atr:
        logging.info(
            "[STAGE] %s gating: ATR %.2f pips below %.2f (stage %d).",
            pocket,
            atr_pips,
            min_atr,
            stage_idx,
        )
        return False

    if stage_idx == 0:
        return True

    if pocket == "macro":
        ma10_h4 = fac_h4.get("ma10")
        ma20_h4 = fac_h4.get("ma20")
        ma10_m1 = fac_m1.get("ma10")
        ma20_m1 = fac_m1.get("ma20")
        ema20_m1 = fac_m1.get("ema20")
        if ema20_m1 is None:
            ema20_m1 = ma20_m1
        close_m1 = fac_m1.get("close")

        if action == "OPEN_LONG":
            if macro_pullback_retrace is not None:
                min_req, _ = _macro_pullback_threshold(stage_idx, atr_pips)
                logging.info(
                    "[STAGE] Macro pullback buy allowed: stage %d retrace=%.1fp (min %.1fp).",
                    stage_idx,
                    macro_pullback_retrace,
                    min_req,
                )
                return True
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
            if macro_pullback_retrace is not None:
                min_req, _ = _macro_pullback_threshold(stage_idx, atr_pips)
                logging.info(
                    "[STAGE] Macro pullback sell allowed: stage %d retrace=%.1fp (min %.1fp).",
                    stage_idx,
                    macro_pullback_retrace,
                    min_req,
                )
                return True
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
        if macro_pullback_retrace is None:
            trend_floor = 18.0 + stage_idx * 1.5
            if adx_h4 < trend_floor or slope_h4 < 0.0004:
                logging.info(
                    "[STAGE] Macro gating failed stage %d (ADX %.2f<%.2f, slope %.5f).",
                    stage_idx,
                    adx_h4,
                    trend_floor,
                    slope_h4,
                )
                return False
            if price is not None and avg_price:
                if action == "OPEN_LONG" and price < avg_price - 0.02:
                    logging.info(
                        "[STAGE] Macro buy gating: price %.3f below avg %.3f.",
                        price,
                        avg_price,
                    )
                    return False
                if action == "OPEN_SHORT" and price > avg_price + 0.02:
                    logging.info(
                        "[STAGE] Macro sell gating: price %.3f above avg %.3f.",
                        price,
                        avg_price,
                    )
                    return False
        # RSI-based re-entry gates
        if action == "OPEN_LONG":
            threshold = 65 - stage_idx * 4
            if rsi > threshold:
                logging.info(
                    "[STAGE] Macro buy gating: RSI %.1f > %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        else:
            threshold = 35 + stage_idx * 4
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
            threshold = 48 - min(stage_idx * 4, 12)
            if rsi > threshold:
                logging.info(
                    "[STAGE] Micro buy gating: RSI %.1f > %.1f for stage %d.",
                    rsi,
                    threshold,
                    stage_idx,
                )
                return False
        else:
            threshold = 52 + min(stage_idx * 4, 12)
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
        return True

    return True


def _evaluate_high_impact_context(
    *,
    pocket: str,
    direction: str,
    lose_streak: int,
    win_streak: int,
    fac_m1: dict[str, float],
    fac_h4: dict[str, float],
    story_snapshot: Optional[ChartStorySnapshot],
    macro_regime: Optional[str],
    momentum: float,
    atr_pips: float,
    range_active: bool,
) -> Optional[dict[str, object]]:
    if pocket != "macro":
        return None
    if range_active:
        return None
    if lose_streak < 3:
        return None

    try:
        adx_h4 = float(fac_h4.get("adx", 0.0) or 0.0)
    except (TypeError, ValueError):
        adx_h4 = 0.0
    try:
        slope_h4 = abs(
            float(fac_h4.get("ma20", 0.0) or 0.0)
            - float(fac_h4.get("ma10", 0.0) or 0.0)
        )
    except (TypeError, ValueError):
        slope_h4 = 0.0

    direction_sign = 1.0 if direction == "long" else -1.0
    directional_momentum = direction_sign * float(momentum)

    macro_story = getattr(story_snapshot, "macro_trend", None) if story_snapshot else None
    higher_story = getattr(story_snapshot, "higher_trend", None) if story_snapshot else None

    strong_trend = adx_h4 >= 27.0 and slope_h4 >= 0.00035
    momentum_ok = directional_momentum >= 0.006
    story_ok = (
        (direction == "long" and macro_story == "up")
        or (direction == "short" and macro_story == "down")
    )
    higher_ok = (
        higher_story is None
        or (direction == "long" and higher_story == "up")
        or (direction == "short" and higher_story == "down")
    )

    if not (strong_trend and momentum_ok and story_ok and higher_ok):
        return None

    reason_parts = []
    if strong_trend:
        reason_parts.append(f"adx={adx_h4:.1f}")
    if momentum_ok:
        reason_parts.append(f"momentum={directional_momentum:.4f}")
    if win_streak > 0:
        reason_parts.append(f"win_streak={win_streak}")
    if macro_regime:
        reason_parts.append(f"regime={macro_regime}")

    return {
        "enabled": True,
        "reason": ", ".join(reason_parts) or "trend_breakout",
        "lose_streak": lose_streak,
        "win_streak": win_streak,
        "adx_h4": adx_h4,
        "momentum": directional_momentum,
        "atr_pips": atr_pips,
    }


def _build_entry_context(
    *,
    pocket: str,
    direction: str,
    stage_index: int,
    size_factor: float,
    confidence_target: float,
    range_active: bool,
    macro_regime: Optional[str],
    micro_regime: Optional[str],
    momentum: float,
    atr_pips: float,
    vol_5m: float,
    lose_streak: int,
    win_streak: int,
    high_impact_enabled: bool,
    high_impact_reason: Optional[str],
) -> Dict[str, object]:
    context: Dict[str, object] = {
        "pocket": pocket,
        "direction": direction,
        "stage_index": int(stage_index),
        "size_factor": round(float(size_factor), 4),
        "confidence_target": round(float(confidence_target), 3),
        "range_active": bool(range_active),
        "macro_regime": macro_regime,
        "micro_regime": micro_regime,
        "momentum": round(float(momentum), 6),
        "atr_pips": round(float(atr_pips), 4),
        "vol_5m": round(float(vol_5m), 4),
        "lose_streak": int(lose_streak),
        "win_streak": int(win_streak),
        "high_impact_override": bool(high_impact_enabled),
    }
    if high_impact_reason:
        context["high_impact_reason"] = high_impact_reason
    return context


def _recompute_m1_technicals(fac_m1: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Best-effort recompute of ATR/RSI/ADX from M1 candles when factor fields are missing.
    Returns (atr_pips, rsi, adx) or (None, None, None) if unavailable.
    """
    candles = fac_m1.get("candles") or []
    if not isinstance(candles, list) or len(candles) < 20:
        return None, None, None
    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for c in candles[-120:]:
        try:
            closes.append(float(c.get("c") or c.get("close")))
            highs.append(float(c.get("h") or c.get("high")))
            lows.append(float(c.get("l") or c.get("low")))
        except Exception:
            continue
    if len(closes) < 20 or len(highs) < 20 or len(lows) < 20:
        return None, None, None

    # ATR (simple Wilder smoothing over last 14 bars)
    tr_list: list[float] = []
    prev_close = closes[0]
    for h, l, c in zip(highs[1:], lows[1:], closes[1:]):
        try:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            tr_list.append(tr / PIP)
            prev_close = c
        except Exception:
            continue
    atr_pips = None
    if len(tr_list) >= 14:
        atr_pips = sum(tr_list[-14:]) / 14.0

    # RSI (reuse fast_scalp helper)
    rsi_val = None
    try:
        rsi_val = _fs_compute_rsi(closes[-30:], 14)
    except Exception:
        rsi_val = None

    # ADX (simplified SMA-based DI/ADX over last 14)
    adx_val = None
    try:
        dm_plus: list[float] = []
        dm_minus: list[float] = []
        tr_vals: list[float] = []
        for i in range(1, min(len(highs), len(lows))):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            dm_plus.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            dm_minus.append(down_move if down_move > up_move and down_move > 0 else 0.0)
            tr_vals.append(tr_list[i - 1] if i - 1 < len(tr_list) else 0.0)
        period = 14
        if len(tr_vals) >= period and len(dm_plus) >= period and len(dm_minus) >= period:
            tr_avg = sum(tr_vals[-period:]) / period
            if tr_avg > 0:
                di_plus = (sum(dm_plus[-period:]) / period) / tr_avg * 100
                di_minus = (sum(dm_minus[-period:]) / period) / tr_avg * 100
                dx_vals: list[float] = []
                for i in range(period, len(dm_plus)):
                    tr_win = tr_vals[i - period : i]
                    dm_p_win = dm_plus[i - period : i]
                    dm_m_win = dm_minus[i - period : i]
                    tr_win_avg = sum(tr_win) / period if tr_win else 0.0
                    if tr_win_avg <= 0:
                        continue
                    di_p = (sum(dm_p_win) / period) / tr_win_avg * 100
                    di_m = (sum(dm_m_win) / period) / tr_win_avg * 100
                    dx_vals.append(abs(di_p - di_m) / max(di_p + di_m, 1e-9) * 100)
                dx_series = dx_vals[-5:] if dx_vals else [abs(di_plus - di_minus) / max(di_plus + di_minus, 1e-9) * 100]
                if dx_series:
                    adx_val = sum(dx_series) / len(dx_series)
    except Exception:
        adx_val = None

    return atr_pips, rsi_val, adx_val


def compute_stage_lot(
    pocket: str,
    total_lot: float,
    open_units_same_dir: int,
    action: str,
    fac_m1: dict[str, float],
    fac_h4: dict[str, float],
    open_info: dict[str, float],
    *,
    high_impact_context: Optional[dict[str, object]] = None,
) -> tuple[float, int]:
    """段階的エントリーの次ロットとステージ番号を返す。"""
    override_enabled = bool(high_impact_context and high_impact_context.get("enabled"))
    override_reason = None
    if override_enabled:
        override_reason = str(high_impact_context.get("reason", "high_impact"))
    direction = "long" if action == "OPEN_LONG" else "short"
    plan = _stage_plan(pocket)
    current_lot = max(open_units_same_dir, 0) / 100000.0
    cumulative = 0.0
    for stage_idx, fraction in enumerate(plan):
        cumulative += fraction
        stage_target = total_lot * cumulative
        if current_lot + 1e-4 < stage_target:
            allow_override = (
                override_enabled
                and pocket == "macro"
                and stage_idx == 0
            )
            stage_ok = _stage_conditions_met(
                pocket, stage_idx, action, fac_m1, fac_h4, open_info
            )
            if not stage_ok:
                if allow_override:
                    logging.info(
                        "[STAGE_OVERRIDE] macro high-impact stage=%d direction=%s reason=%s",
                        stage_idx,
                        direction,
                        override_reason,
                    )
                    try:
                        log_metric(
                            "stage_override_high_impact",
                            1.0,
                            tags={
                                "direction": direction,
                                "reason": override_reason or "high_impact",
                                "lose_streak": str(
                                    high_impact_context.get("lose_streak")
                                    if high_impact_context
                                    else ""
                                ),
                            },
                        )
                    except Exception:
                        pass
                else:
                    return 0.0, stage_idx
            next_lot = max(stage_target - current_lot, 0.0)
            remaining = max(total_lot - current_lot, 0.0)
            if pocket == "macro" and remaining > 0:
                floor = min(MIN_MACRO_STAGE_LOT, remaining)
                next_lot = max(next_lot, floor)
            if pocket == "micro" and remaining > 0:
                next_lot = min(next_lot, MAX_MICRO_STAGE_LOT, remaining)
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
    extra = REENTRY_EXTRA_LOT.get(pocket, 0.0)
    if extra > 0.0:
        logging.info(
            "[STAGE] %s pocket allow re-entry override extra=%.3f lots",
            pocket,
            round(extra, 4),
        )
        return round(extra, 4), len(plan) if plan else -1
    return 0.0, -1


def _cluster_directional_units(
    direction: str,
    open_positions: dict[str, dict[str, object]],
    pockets: tuple[str, ...] = ("macro", "micro", "scalp"),
) -> int:
    """
    Aggregate existing exposure for averaging logic.
    Returns gross units in the requested direction across specified pockets.
    """
    key = "long_units" if direction == "long" else "short_units"
    total = 0
    for name in pockets:
        info = open_positions.get(name)
        if not info:
            continue
        try:
            units_val = int(info.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if units_val > 0:
            total += units_val
    return total


def _micro_chart_gate(
    signal: dict[str, object],
    fac_m1: dict[str, object],
    story_snapshot: object | None,
    open_positions: dict[str, dict[str, object]] | None = None,
) -> tuple[bool, str, dict[str, float | str]]:
    """
    Lightweight micro entry guard that uses recent price action instead of blindly stacking.
    Returns (allow, reason, ctx).
    """
    if AGGRESSIVE_TRADING:
        return True, "aggressive_override", {"mode": "aggressive"}
    pocket = signal.get("pocket")
    action = signal.get("action")
    if pocket != "micro" or action not in {"OPEN_LONG", "OPEN_SHORT"}:
        return True, "pass", {}

    candles = fac_m1.get("candles") or []
    if len(candles) < 6:
        return True, "no_candles", {}

    closes: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    for cndl in candles:
        try:
            close_val = _safe_float(cndl.get("close"))
            high_val = _safe_float(cndl.get("high"), close_val)
            low_val = _safe_float(cndl.get("low"), close_val)
        except Exception:
            continue
        closes.append(close_val)
        highs.append(high_val)
        lows.append(low_val)

    if len(closes) < 6:
        return True, "no_candles", {}

    price = closes[-1]
    slope6 = (price - closes[-6]) / PIP
    window = min(12, len(highs))
    high_n = max(highs[-window:])
    low_n = min(lows[-window:])
    range_pips = (high_n - low_n) / PIP if highs and lows else 0.0
    top_gap = (high_n - price) / PIP if highs else 0.0
    bottom_gap = (price - low_n) / PIP if lows else 0.0

    # 攻め方/足場を確認: 直近高値へのアタック回数と安値の切り上げ幅
    high_attacks = 0
    low_attacks = 0
    band = 0.3  # pips許容帯 (動的に調整)
    try:
        atr_pips = float(fac_m1.get("atr_pips") or (fac_m1.get("atr") or 0.0) * 100.0)
    except Exception:
        atr_pips = 0.0
    if atr_pips > 0:
        band = max(0.2, min(0.6, atr_pips * 0.05))
    # ボラと市況（セッション/上位TF）に応じて「何回のタッチで鈍化とみなすか」を動的に決める
    if atr_pips <= 3.0:
        attack_thresh = 2.4
    elif atr_pips >= 8.0:
        attack_thresh = 4.2
    else:
        attack_thresh = 3.0
    if highs:
        recent_high = max(highs[-window:])
        for h in highs[-6:]:
            if (recent_high - h) / PIP <= band:
                high_attacks += 1
    if lows:
        recent_low = min(lows[-window:])
        for l in lows[-6:]:
            if (l - recent_low) / PIP <= band:
                low_attacks += 1
    low_base_rise = 0.0
    if lows:
        first_half = lows[-window:-window // 2] or lows[:-window // 2] or lows
        if first_half:
            try:
                low_base_rise = (lows[-1] - min(first_half)) / PIP
            except Exception:
                low_base_rise = 0.0

    base_rise_thresh = 0.2
    if atr_pips > 0:
        base_rise_thresh = max(0.1, min(0.6, atr_pips * 0.04))

    range_chop_thresh = 4.0
    slope_chop_thresh = 1.4
    top_gap_thresh = 0.6
    bottom_gap_thresh = 0.6
    slope_chase_thresh = 2.0
    slope_stack_soft = 3.0
    slope_stack_hard = 0.8
    nwave_slope_thresh = 1.5
    if atr_pips > 0:
        range_chop_thresh = max(2.5, min(8.0, atr_pips * 1.5))
        slope_chop_thresh = max(1.0, min(3.5, atr_pips * 0.6))
        top_gap_thresh = max(0.3, min(1.2, atr_pips * 0.05))
        bottom_gap_thresh = top_gap_thresh
        slope_chase_thresh = max(1.5, min(4.0, atr_pips * 0.8))
        slope_stack_soft = max(2.0, min(5.0, atr_pips * 0.9))
        slope_stack_hard = max(0.6, min(2.0, atr_pips * 0.3))
        nwave_slope_thresh = max(1.0, min(3.0, atr_pips * 0.5))

    pattern_summary = {}
    if story_snapshot and hasattr(story_snapshot, "pattern_summary"):
        try:
            pattern_summary = dict(getattr(story_snapshot, "pattern_summary") or {})
        except Exception:
            pattern_summary = {}
    n_wave = None
    candlestick = None
    if isinstance(pattern_summary, dict):
        n_wave = pattern_summary.get("n_wave")
        candlestick = pattern_summary.get("candlestick")
    n_wave_direction = None
    if isinstance(n_wave, dict):
        n_wave_direction = n_wave.get("direction") or n_wave.get("bias")

    micro_trend = None
    summary = {}
    if story_snapshot and hasattr(story_snapshot, "micro_trend"):
        micro_trend = getattr(story_snapshot, "micro_trend")
        if hasattr(story_snapshot, "summary"):
            try:
                summary = dict(getattr(story_snapshot, "summary") or {})
            except Exception:
                summary = {}
        # 方向ミスマッチによるブロックは無効化（機会損失防止）

    def _opposes(frame_trend: str | None, side: str) -> bool:
        if frame_trend not in {"up", "down"}:
            return False
        return (side == "OPEN_LONG" and frame_trend == "down") or (
            side == "OPEN_SHORT" and frame_trend == "up"
        )

    m15_trend = summary.get("M15") if isinstance(summary, dict) else None
    h1_trend = summary.get("H1") if isinstance(summary, dict) else None
    vol_state = ""
    if story_snapshot and hasattr(story_snapshot, "volatility_state"):
        try:
            vol_state = str(getattr(story_snapshot, "volatility_state") or "")
        except Exception:
            vol_state = ""
    session = _session_bucket(datetime.datetime.utcnow())

    # 上位TFのトレンド・ボラとセッションに合わせて閾値を微調整
    if action == "OPEN_LONG":
        if m15_trend == "up":
            attack_thresh += 0.4
        if h1_trend == "up":
            attack_thresh += 0.6
        if m15_trend == "down" or h1_trend == "down":
            attack_thresh -= 0.5
    elif action == "OPEN_SHORT":
        if m15_trend == "down":
            attack_thresh += 0.4
        if h1_trend == "down":
            attack_thresh += 0.6
        if m15_trend == "up" or h1_trend == "up":
            attack_thresh -= 0.5
    if vol_state == "low":
        attack_thresh -= 0.3
        base_rise_thresh *= 1.1
    elif vol_state == "high":
        attack_thresh += 0.3
        base_rise_thresh *= 0.9
    if session == "asia":
        attack_thresh -= 0.2
    elif session == "ny":
        attack_thresh += 0.2
    attack_thresh = max(2.0, min(5.0, attack_thresh))

    if action == "OPEN_LONG":
        if micro_trend == "up":
            base_rise_thresh = max(0.08, base_rise_thresh * 0.9)
        elif micro_trend == "down":
            base_rise_thresh = min(0.8, base_rise_thresh * 1.15)
    else:
        if micro_trend == "down":
            base_rise_thresh = max(0.08, base_rise_thresh * 0.9)
        elif micro_trend == "up":
            base_rise_thresh = min(0.8, base_rise_thresh * 1.15)
    opposed_count = int(_opposes(m15_trend, str(action))) + int(_opposes(h1_trend, str(action)))
    if opposed_count >= 2 and abs(slope6) <= 3.5:
        return False, "micro_mtf_opposed", {
            "slope6": round(slope6, 2),
            "range": round(range_pips, 2),
            "trend": micro_trend or "",
            "m15": m15_trend or "",
            "h1": h1_trend or "",
        }

    if action == "OPEN_LONG":
        if high_attacks >= attack_thresh and low_base_rise <= base_rise_thresh and slope6 <= slope_stack_soft:
            return False, "micro_high_attack_flat_base", {
                "slope6": round(slope6, 2),
                "range": round(range_pips, 2),
                "trend": micro_trend or "",
                "m15": m15_trend or "",
                "h1": h1_trend or "",
                "high_attacks": high_attacks,
                "low_rise": round(low_base_rise, 2),
                "attack_thresh": attack_thresh,
                "base_rise_thresh": round(base_rise_thresh, 2),
            }
        if n_wave_direction == "down" and slope6 <= nwave_slope_thresh:
            return False, "micro_nwave_opposed", {
                "slope6": round(slope6, 2),
                "range": round(range_pips, 2),
                "trend": micro_trend or "",
                "m15": m15_trend or "",
                "h1": h1_trend or "",
                "nwave": n_wave,
            }
    if action == "OPEN_SHORT":
        if low_attacks >= attack_thresh and low_base_rise >= -base_rise_thresh and slope6 >= -slope_stack_soft:
            return False, "micro_low_attack_flat_base", {
                "slope6": round(slope6, 2),
                "range": round(range_pips, 2),
                "trend": micro_trend or "",
                "m15": m15_trend or "",
                "h1": h1_trend or "",
                "low_attacks": low_attacks,
                "low_rise": round(low_base_rise, 2),
                "attack_thresh": attack_thresh,
                "base_rise_thresh": round(base_rise_thresh, 2),
            }
        if n_wave_direction == "up" and slope6 >= -nwave_slope_thresh:
            return False, "micro_nwave_opposed", {
                "slope6": round(slope6, 2),
                "range": round(range_pips, 2),
                "trend": micro_trend or "",
                "m15": m15_trend or "",
                "h1": h1_trend or "",
                "nwave": n_wave,
            }

    # H1ローソク足パターンが明確に逆行している場合はワンタッチで抑制（強トレンド時のみ許容）
    if candlestick and isinstance(candlestick, dict):
        try:
            candle_conf = float(candlestick.get("confidence", 0.0) or 0.0)
        except Exception:
            candle_conf = 0.0
        candle_type = str(candlestick.get("type") or "")
        candle_bias = str(candlestick.get("bias") or "")
        opp_long = action == "OPEN_LONG" and candle_bias == "down"
        opp_short = action == "OPEN_SHORT" and candle_bias == "up"
        strong_trend_supports = (action == "OPEN_LONG" and micro_trend == "up") or (
            action == "OPEN_SHORT" and micro_trend == "down"
        )
        if candle_conf >= 0.6 and (opp_long or opp_short) and not strong_trend_supports:
            reason = "micro_candle_opposed"
            return False, reason, {
                "slope6": round(slope6, 2),
                "range": round(range_pips, 2),
                "trend": micro_trend or "",
                "m15": m15_trend or "",
                "h1": h1_trend or "",
                "candle": candle_type,
                "candle_conf": round(candle_conf, 2),
            }

    if range_pips <= range_chop_thresh and abs(slope6) <= slope_chop_thresh:
        return False, "micro_chop_gate", {
            "slope6": round(slope6, 2),
            "range": round(range_pips, 2),
            "trend": micro_trend or "",
            "m15": m15_trend or "",
            "h1": h1_trend or "",
        }

    # 直近天井/底付近のブロックを解除（スカルプの取りこぼし防止）

    micro_info = (open_positions or {}).get("micro", {}) if open_positions else {}
    try:
        long_units = int(micro_info.get("long_units", 0) or 0)
    except Exception:
        long_units = 0
    try:
        short_units = int(micro_info.get("short_units", 0) or 0)
    except Exception:
        short_units = 0
    try:
        avg_price = float(micro_info.get("avg_price", 0.0) or 0.0)
    except Exception:
        avg_price = 0.0
    same_band = abs(price - avg_price) / PIP if avg_price > 0 else 99.0
    if (
        action == "OPEN_LONG"
        and long_units >= 15000
        and same_band <= 1.2  # 1.2pips帯での積み増しを抑制
        and slope6 <= slope_stack_soft
    ):
        return False, "micro_same_band_stack", {
            "slope6": round(slope6, 2),
            "range": round(range_pips, 2),
            "trend": micro_trend or "",
            "m15": m15_trend or "",
            "h1": h1_trend or "",
            "long_units": long_units,
            "band_pips": round(same_band, 2),
        }
    if (
        action == "OPEN_SHORT"
        and short_units >= 15000
        and same_band <= 1.2
        and slope6 >= -slope_stack_soft
    ):
        return False, "micro_same_band_stack_short", {
            "slope6": round(slope6, 2),
            "range": round(range_pips, 2),
            "trend": micro_trend or "",
            "m15": m15_trend or "",
            "h1": h1_trend or "",
            "short_units": short_units,
            "band_pips": round(same_band, 2),
        }
    stack_threshold = 20000
    if action == "OPEN_LONG" and long_units >= stack_threshold and slope6 <= 0.8:
        return False, "micro_stack_guard_long", {
            "slope6": round(slope6, 2),
            "range": round(range_pips, 2),
            "trend": micro_trend or "",
            "long_units": long_units,
        }
    if action == "OPEN_SHORT" and short_units >= stack_threshold and slope6 >= -0.8:
        return False, "micro_stack_guard_short", {
            "slope6": round(slope6, 2),
            "range": round(range_pips, 2),
            "trend": micro_trend or "",
            "short_units": short_units,
            "m15": m15_trend or "",
            "h1": h1_trend or "",
        }

    return True, "ok", {
        "slope6": round(slope6, 2),
        "range": round(range_pips, 2),
        "trend": micro_trend or "",
        "m15": m15_trend or "",
        "h1": h1_trend or "",
    }


async def m1_candle_handler(cndl: Candle):
    await on_candle("M1", cndl)


async def h4_candle_handler(cndl: Candle):
    await on_candle("H4", cndl)


async def m5_candle_handler(cndl: Candle):
    await on_candle("M5", cndl)


async def h1_candle_handler(cndl: Candle):
    await on_candle("H1", cndl)


async def d1_candle_handler(cndl: Candle):
    await on_candle("D1", cndl)


async def worker_only_loop() -> None:
    """Minimal supervisor loop for worker-only runtime."""
    last_worker_plan: set[str] = set()
    last_market_closed: Optional[datetime.datetime] = None
    last_heartbeat_time = datetime.datetime.utcnow()

    while True:
        now = datetime.datetime.utcnow()
        if not is_market_open(now):
            if last_market_closed is None or (now - last_market_closed).total_seconds() >= 900:
                wait_sec = max(60.0, min(3600.0, seconds_until_open(now)))
                logging.info(
                    "[MARKET_CLOSED] Worker-only standby. Next open in ~%.1f min (UTC=%s)",
                    wait_sec / 60.0,
                    now.isoformat(timespec="seconds"),
                )
                last_market_closed = now
            await asyncio.sleep(60)
            continue
        last_market_closed = None

        if WORKER_AUTOCONTROL_ENABLED:
            try:
                desired_workers = WORKER_ALL_SERVICES
                if not desired_workers:
                    logging.debug("[WORKER_CTL] no worker services configured; skip reconcile")
                elif desired_workers != last_worker_plan:
                    await _reconcile_worker_services(last_worker_plan, desired_workers)
                    last_worker_plan = desired_workers
            except Exception as exc:  # pragma: no cover - defensive
                logging.debug("[WORKER_CTL] reconcile failed: %s", exc)

        if (now - last_heartbeat_time).total_seconds() >= 300:
            logging.info(
                "[WORKER_ONLY] heartbeat active_workers=%s",
                ",".join(sorted(last_worker_plan)) if last_worker_plan else "unknown",
            )
            last_heartbeat_time = now

        await asyncio.sleep(10)


async def logic_loop(
    gpt_state: GPTDecisionState,
    gpt_requests: GPTRequestManager,
    fast_scalp_state: FastScalpState | None = None,
    *,
    rr_advisor: RRRatioAdvisor | None = None,
    exit_advisor: ExitAdvisor | None = None,
    strategy_conf_advisor: StrategyConfidenceAdvisor | None = None,
    focus_advisor: FocusOverrideAdvisor | None = None,
    volatility_advisor: VolatilityBiasAdvisor | None = None,
    stage_plan_advisor: StagePlanAdvisor | None = None,
    partial_advisor: PartialReductionAdvisor | None = None,
):
    if WORKER_ONLY_MODE and not SIGNAL_GATE_ENABLED:
        logging.info("[LOGIC_LOOP] disabled (worker-only runtime).")
        return
    if not MAIN_TRADING_ENABLED and not SIGNAL_GATE_ENABLED:
        logging.info("[LOGIC_LOOP] disabled (main trading off, gate off).")
        return
    pos_manager = PositionManager()
    metrics_client = RealtimeMetricsClient()
    confidence_policy = ConfidencePolicy()
    exit_manager = ExitManager()
    stage_tracker = StageTracker()
    param_context = ParamContext()
    chart_story = ChartStory()
    perf_cache = {}
    insight = InsightClient()
    fs_strategy_enabled = firestore_strategy_enabled()
    fs_strategy_client = FirestoreStrategyClient(enable=fs_strategy_enabled) if fs_strategy_enabled else None
    fs_strategy_apply_sltp = _env_bool("FIRESTORE_STRATEGY_APPLY_SLTP", default=False)
    level_map_enabled = _env_bool("LEVEL_MAP_ENABLE", default=False)
    level_map_client = LevelMapClient(
        object_path=os.getenv("LEVEL_MAP_OBJECT_PATH", "analytics/level_map.json"),
        ttl_sec=int(os.getenv("LEVEL_MAP_TTL_SEC", "300")),
    ) if level_map_enabled else None
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
    guard_state: dict[str, dict[str, object]] = {
        "scalp": {"mode": "NEUTRAL", "range_hits": 0, "trend_hits": 0, "neutral_hits": 0},
        "micro": {"mode": "NEUTRAL", "range_hits": 0, "trend_hits": 0, "neutral_hits": 0},
        "global": {"mode": "NEUTRAL", "range_hits": 0, "trend_hits": 0, "neutral_hits": 0},
    }
    guard_contexts: dict[str, RangeContext] = {}
    raw_range_active = False
    raw_range_reason = ""
    range_breakout_release_until = datetime.datetime.min
    range_breakout_reason = ""
    last_macro_regime: Optional[str] = None
    last_micro_regime: Optional[str] = None
    range_override_active_prev = False
    last_range_scalp_ready: Optional[bool] = None
    range_macro_hold_until = datetime.datetime.min
    stage_empty_since: dict[tuple[str, str], datetime.datetime] = {}
    last_risk_pct: float | None = None
    last_spread_gate = False
    last_spread_gate_reason = ""
    last_logged_range_state: Optional[bool] = None
    last_logged_focus: Optional[str] = None
    last_logged_weight: Optional[float] = None
    last_logged_scalp_weight: Optional[float] = None
    recent_profiles: dict[str, dict[str, float]] = {}
    param_snapshot: Optional[ParamSnapshot] = None
    scalp_ready_forced = False
    loop_counter = 0
    tick_empty_counter = 0

    def _factor_snapshot(data: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not data:
            return {}
        snapshot: Dict[str, float] = {}
        for key in ("adx", "rsi", "ma10", "ma20", "atr_pips", "close"):
            val = data.get(key)
            if val is None:
                continue
            try:
                snapshot[key] = round(float(val), 6)
            except (TypeError, ValueError):
                continue
        return snapshot

    def _normalize_bus_signal(raw: dict, price_hint: Optional[float]) -> Optional[dict]:
        if not isinstance(raw, dict):
            return None
        pocket = str(raw.get("pocket") or raw.get("pocket_name") or "").strip() or "micro"
        pocket_lower = pocket.lower()
        if pocket_lower.startswith("scalp"):
            pocket = "scalp"
        elif pocket_lower.startswith("micro"):
            pocket = "micro"
        elif pocket_lower.startswith("macro"):
            pocket = "macro"
        action = (raw.get("action") or raw.get("side") or "").upper()
        if not action and "units" in raw:
            try:
                action = "OPEN_LONG" if float(raw.get("units")) > 0 else "OPEN_SHORT"
            except Exception:
                action = ""
        if action not in {"OPEN_LONG", "OPEN_SHORT", "CLOSE"}:
            return None
        strategy = (
            raw.get("strategy")
            or raw.get("strategy_tag")
            or raw.get("tag")
            or raw.get("profile")
            or ""
        )
        if not strategy:
            return None
        try:
            conf = int(raw.get("confidence", raw.get("conf", 50)) or 50)
        except Exception:
            conf = 50
        conf = max(0, min(100, conf))
        entry_price = _safe_float(
            raw.get("entry_price")
            or raw.get("price")
            or raw.get("mid_price")
            or raw.get("entry_ref")
            or raw.get("entry")
        )
        sl_price = _safe_float(raw.get("sl_price"))
        tp_price = _safe_float(raw.get("tp_price"))
        sl_pips = _safe_float(raw.get("sl_pips"), default=None)
        tp_pips = _safe_float(raw.get("tp_pips"), default=None)
        if sl_pips is None and entry_price is not None and sl_price is not None:
            sl_pips = abs(entry_price - sl_price) / PIP
        if tp_pips is None and entry_price is not None and tp_price is not None:
            tp_pips = abs(entry_price - tp_price) / PIP
        if sl_pips is None and price_hint is not None and sl_price is not None:
            sl_pips = abs(price_hint - sl_price) / PIP
        if tp_pips is None and price_hint is not None and tp_price is not None:
            tp_pips = abs(price_hint - tp_price) / PIP
        reduce_only_val = raw.get("reduce_only")
        reduce_cap_units_raw = raw.get("reduce_cap_units")
        reduce_only = None
        if reduce_only_val is not None:
            if isinstance(reduce_only_val, str):
                reduce_only = reduce_only_val.strip().lower() in {"1", "true", "yes", "on"}
            else:
                reduce_only = bool(reduce_only_val)
        reduce_cap_units = None
        if reduce_cap_units_raw is not None:
            try:
                reduce_cap_units = int(float(reduce_cap_units_raw))
            except Exception:
                reduce_cap_units = None
        sig = {
            "strategy": str(strategy),
            "pocket": pocket,
            "action": action,
            "confidence": conf,
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "tag": raw.get("tag") or strategy,
            "entry_price": entry_price,
            "entry_type": raw.get("entry_type") or raw.get("order_type") or "market",
            "client_order_id": raw.get("client_order_id"),
            "proposed_units": raw.get("proposed_units"),
            "meta": raw.get("meta") or {},
            "source": raw.get("source") or "bus",
        }
        profile = raw.get("profile") or raw.get("strategy_profile")
        if profile:
            sig["profile"] = profile
        if reduce_only is not None:
            sig["reduce_only"] = reduce_only
        if reduce_cap_units is not None:
            sig["reduce_cap_units"] = reduce_cap_units
        if sl_price is not None:
            sig["sl_price"] = sl_price
        if tp_price is not None:
            sig["tp_price"] = tp_price
        entry_thesis = raw.get("entry_thesis")
        if isinstance(entry_thesis, dict):
            sig["entry_thesis"] = entry_thesis
        execution = raw.get("execution")
        if isinstance(execution, dict):
            sig["execution"] = execution
        return sig

    def _market_guarded_skip(signal: dict, price: Optional[float], *, context: str) -> bool:
        if not signal or price is None or price <= 0:
            return False
        if signal.get("reduce_only"):
            return False
        exec_cfg = signal.get("execution")
        if not isinstance(exec_cfg, dict):
            thesis = signal.get("entry_thesis")
            if isinstance(thesis, dict):
                exec_cfg = thesis.get("execution")
        if not isinstance(exec_cfg, dict):
            return False
        if exec_cfg.get("order_policy") != "market_guarded":
            return False
        ideal = _safe_float(exec_cfg.get("ideal_entry"))
        chase_max = _safe_float(exec_cfg.get("chase_max"))
        if ideal is None or chase_max is None:
            return False
        if abs(price - ideal) <= chase_max:
            return False
        logging.info(
            "[MARKET_GUARD] skip=%s strategy=%s pocket=%s price=%.3f ideal=%.3f chase_max=%.3f",
            context,
            signal.get("strategy") or signal.get("tag"),
            signal.get("pocket"),
            price,
            ideal,
            chase_max,
        )
        return True

    MR_SIGNAL_TAGS = {
        "BB_RSI",
        "MicroVWAPBound",
        "MicroVWAPRevert",
        "RangeFader",
        "vwap_magnet_s5",
        "mirror_spike",
        "mirror_spike_tight",
        "mirror_spike_s5",
    }
    MR_OVERLAY_TAGS = {"VolCompressionBreak", "MomentumPulse"}

    def _is_mr_signal(strategy_tag: Optional[str], profile: Optional[str]) -> bool:
        tag = str(strategy_tag or "").strip()
        if not tag:
            return False
        base_tag = tag.split("-", 1)[0]
        if base_tag in MR_SIGNAL_TAGS:
            return True
        tag_lower = tag.lower()
        if tag_lower.startswith("mlr-fade") or tag_lower.startswith("mlr-bounce"):
            return True
        if profile in {"bb_range_reversion", "micro_vwap_bound"}:
            return True
        return False

    def _is_mr_overlay_signal(strategy_tag: Optional[str], profile: Optional[str]) -> bool:
        tag = str(strategy_tag or "").strip()
        if not tag:
            return False
        base_tag = tag.split("-", 1)[0]
        if base_tag in MR_OVERLAY_TAGS:
            return True
        return False

    def _augment_entry_thesis_for_mr(
        entry_thesis: dict,
        *,
        pocket: str,
        atr_entry: float,
        overlay: bool = False,
    ) -> None:
        if not isinstance(entry_thesis, dict):
            return
        if pocket not in {"micro", "scalp"}:
            return
        env_tf, struct_tf = ("H1", "M5") if pocket == "micro" else ("M5", "M1")
        lookback = 20
        hi_pct = 95.0
        lo_pct = 5.0
        entry_thesis.setdefault("env_tf", env_tf)
        entry_thesis.setdefault("struct_tf", struct_tf)
        entry_thesis.setdefault("range_method", "percentile")
        entry_thesis.setdefault("range_lookback", lookback)
        entry_thesis.setdefault("range_hi_pct", hi_pct)
        entry_thesis.setdefault("range_lo_pct", lo_pct)
        if atr_entry and atr_entry > 0:
            entry_thesis.setdefault("atr_entry", float(atr_entry))
        entry_thesis.setdefault("structure_break", {"buffer_atr": 0.10, "confirm_closes": 2})
        if pocket == "scalp":
            z_ext = 0.45
            contraction_min = 0.45
            k_per_z = 2.5
            min_bars = 2
            max_bars = 8
        else:
            z_ext = 0.55
            contraction_min = 0.50
            k_per_z = 3.5
            min_bars = 2
            max_bars = 12
        rf = entry_thesis.setdefault("reversion_failure", {})
        if isinstance(rf, dict):
            rf.setdefault("z_ext", z_ext)
            rf.setdefault("contraction_min", contraction_min)
            bars_budget = rf.setdefault("bars_budget", {})
            if isinstance(bars_budget, dict):
                bars_budget.setdefault("k_per_z", k_per_z)
                bars_budget.setdefault("min", min_bars)
                bars_budget.setdefault("max", max_bars)
            rf.setdefault("trend_takeover", {"require_env_trend_bars": 2})
            trend_bias = bool(entry_thesis.get("trend_bias") or entry_thesis.get("trend_score"))
            if trend_bias:
                rf["z_ext"] = 0.45
                rf["contraction_min"] = 0.60
                if isinstance(bars_budget, dict):
                    bars_budget["k_per_z"] = 2.5
                    bars_budget["max"] = 8
        entry_thesis.setdefault("tp_mode", "soft_zone")
        entry_thesis.setdefault("tp_target", "entry_mean")
        base_pad = _safe_float(os.getenv("MR_TP_PAD_ATR"), 0.05)
        overlay_pad = _safe_float(os.getenv("MR_OVERLAY_TP_PAD_ATR"), 0.06)
        entry_thesis.setdefault("tp_pad_atr", overlay_pad if overlay else base_pad)
        if entry_thesis.get("range_snapshot") and entry_thesis.get("entry_mean") is not None:
            return
        candles = get_candles_snapshot(env_tf, limit=lookback)
        snapshot = compute_range_snapshot(
            candles,
            lookback=lookback,
            method="percentile",
            hi_pct=hi_pct,
            lo_pct=lo_pct,
        )
        if snapshot:
            entry_thesis.setdefault("range_snapshot", snapshot.to_dict())
            entry_thesis.setdefault("entry_mean", snapshot.mid)

    def _confirm_guard_mode(raw_mode: str, state: dict[str, object]) -> str:
        mode = str(raw_mode or "").upper()
        range_hits = int(state.get("range_hits", 0) or 0)
        trend_hits = int(state.get("trend_hits", 0) or 0)
        neutral_hits = int(state.get("neutral_hits", 0) or 0)
        if mode == "RANGE":
            range_hits += 1
            trend_hits = 0
            neutral_hits = 0
            if range_hits >= RANGE_GUARD_RANGE_CONFIRM:
                state["mode"] = "RANGE"
        elif mode == "TREND":
            trend_hits += 1
            range_hits = 0
            neutral_hits = 0
            if trend_hits >= RANGE_GUARD_TREND_CONFIRM:
                state["mode"] = "TREND"
        else:
            neutral_hits += 1
            range_hits = 0
            trend_hits = 0
            if neutral_hits >= RANGE_GUARD_NEUTRAL_CONFIRM:
                state["mode"] = "NEUTRAL"
        state["range_hits"] = range_hits
        state["trend_hits"] = trend_hits
        state["neutral_hits"] = neutral_hits
        return str(state.get("mode") or "NEUTRAL")

    def _mr_guard_snapshot(pocket: str) -> dict[str, object]:
        pocket_key = "scalp" if pocket in {"scalp_fast"} else pocket
        local_ctx = guard_contexts.get(pocket_key)
        global_ctx = guard_contexts.get("global")
        return {
            "local_mode": guard_modes.get(pocket_key, "NEUTRAL"),
            "local_score": getattr(local_ctx, "score", None),
            "local_reason": getattr(local_ctx, "reason", None),
            "local_env_tf": getattr(local_ctx, "env_tf", None),
            "global_mode": global_guard_mode,
            "global_score": getattr(global_ctx, "score", None),
            "global_reason": getattr(global_ctx, "reason", None),
            "global_env_tf": getattr(global_ctx, "env_tf", None),
        }

    def _mr_guard_reject(signal: dict, *, context: str) -> bool:
        tag = signal.get("strategy") or signal.get("strategy_tag") or signal.get("tag")
        profile = signal.get("profile") or signal.get("strategy_profile")
        if not _is_mr_signal(tag, profile):
            return False
        pocket = signal.get("pocket")
        pocket_key = "scalp" if pocket in {"scalp_fast"} else pocket
        local_mode = guard_modes.get(pocket_key, "NEUTRAL")
        if local_mode != "RANGE" or global_guard_mode == "TREND":
            logging.info(
                "[MR_GUARD] skip=%s strategy=%s pocket=%s local=%s global=%s",
                context,
                tag,
                pocket,
                local_mode,
                global_guard_mode,
            )
            return True
        return False

    last_volatility_state: Optional[str] = None
    last_liquidity_state: Optional[str] = None
    last_risk_appetite: Optional[float] = None
    last_vol_high_ratio: Optional[float] = None
    last_stage_biases: dict[str, float] = {}
    last_story_summary: Optional[dict] = None
    last_worker_plan: set[str] = set()
    last_market_closed: Optional[datetime.datetime] = None
    last_rsi_m1: Optional[float] = None
    last_close_m1: Optional[float] = None
    clamp_state: dict[str, object] = {}
    last_local_decision: dict | None = None

    try:
        while True:
            now = datetime.datetime.utcnow()
            loop_start_mono = time.monotonic()
            loop_counter += 1
            scalp_ready_forced = False
            evaluated_signals: list[dict] = []
            logging.info("[LOOP] start loop=%d", loop_counter)
            if FORCE_SCALP_MODE and loop_counter % 5 == 0:
                logging.warning("[FORCE_SCALP] loop=%d", loop_counter)
            stage_tracker.clear_expired(now)
            # Heartbeat logging
            if (now - last_heartbeat_time).total_seconds() >= 300:  # Every 5 minutes
                logging.info(
                    f"[HEARTBEAT] System is alive at {now.isoformat(timespec='seconds')}"
                )
                last_heartbeat_time = now

            # 5分ごとにパフォーマンスを更新
            if (now - last_update_time).total_seconds() >= 300:
                perf_cache = get_perf()
                try:
                    insight.refresh()
                except Exception:
                    pass
                if level_map_client:
                    try:
                        level_map_client.refresh()
                    except Exception:
                        pass
                if fs_strategy_client:
                    try:
                        fs_strategy_client.refresh()
                    except Exception:
                        pass
                last_update_time = now
                logging.info(f"[PERF] Updated: {perf_cache}")

            # 週末クローズ中はロジックとGPTを止めて待機
            if not is_market_open(now):
                if last_market_closed is None or (now - last_market_closed).total_seconds() >= 900:
                    wait_sec = max(60.0, min(3600.0, seconds_until_open(now)))
                    logging.info(
                        "[MARKET_CLOSED] Weekend window active. Next open in ~%.1f min (UTC=%s)",
                        wait_sec / 60.0,
                        now.isoformat(timespec="seconds"),
                    )
                    last_market_closed = now
                await asyncio.sleep(60)
                continue
            else:
                last_market_closed = None

            if WORKER_AUTOCONTROL_ENABLED:
                try:
                    desired_workers = WORKER_ALL_SERVICES
                    if not desired_workers:
                        logging.debug("[WORKER_CTL] no worker services configured; skip reconcile")
                    elif desired_workers != last_worker_plan:
                        await _reconcile_worker_services(last_worker_plan, desired_workers)
                        last_worker_plan = desired_workers
                except Exception as exc:  # pragma: no cover - defensive
                    logging.debug("[WORKER_CTL] reconcile failed: %s", exc)

            if (WORKER_ONLY_MODE or not MAIN_TRADING_ENABLED) and not SIGNAL_GATE_ENABLED:
                if loop_counter % 5 == 1:
                    logging.info(
                        "[WORKER_MODE] trading loop skipped (worker_only=%s main_trading=%s). active_workers=%s",
                        WORKER_ONLY_MODE,
                        MAIN_TRADING_ENABLED,
                        ",".join(sorted(last_worker_plan)) if last_worker_plan else "unknown",
                    )
                await asyncio.sleep(10)
                continue

            # --- 1. 状況分析 ---
            factors = all_factors()
            fac_m1_raw = factors.get("M1")
            fac_m5 = factors.get("M5")
            fac_h1 = factors.get("H1")
            fac_h4 = factors.get("H4")
            fac_d1 = factors.get("D1")

            # 両方のタイムフレームのデータが揃うまで待機
            if (
                not fac_m1_raw
                or not fac_h4
                or not fac_m1_raw.get("close")
                or not fac_h4.get("close")
            ):
                logging.info("[WAIT] Waiting for M1/H4 factor data for trading logic...")
                if FORCE_SCALP_MODE:
                    logging.warning(
                        "[FORCE_SCALP] factors missing m1=%s h4=%s",
                        bool(fac_m1_raw),
                        bool(fac_h4),
                    )
                await asyncio.sleep(5)
                continue

            fac_m1 = dict(fac_m1_raw)
            recent_tick_rows = tick_window.recent_ticks(75.0, limit=180)
            tick_count = len(recent_tick_rows)
            if tick_count == 0:
                tick_empty_counter += 1
                if tick_empty_counter in {1, 6, 12} or tick_empty_counter % 30 == 0:
                    logging.warning(
                        "[TICK] No recent ticks available (empty_count=%d)",
                        tick_empty_counter,
                    )
            else:
                if tick_empty_counter:
                    logging.info(
                        "[TICK] Recent ticks restored count=%d after %d empty cycles",
                        tick_count,
                        tick_empty_counter,
                    )
                    tick_empty_counter = 0
                # Keep spread monitor in sync using the latest tick (netting-aware sizing relies on this).
                try:
                    last_tick = recent_tick_rows[-1]
                    bid = last_tick.get("bid")
                    ask = last_tick.get("ask")
                    if bid is not None and ask is not None:
                        epoch = last_tick.get("epoch")
                        ts = (
                            datetime.fromtimestamp(float(epoch), timezone.utc)
                            if epoch is not None
                            else datetime.utcnow().replace(tzinfo=timezone.utc)
                        )
                        spread_monitor.update_from_tick(
                            SimpleNamespace(bid=bid, ask=ask, time=ts)
                        )
                except Exception:
                    pass
            def _range_pips_from_ticks(ticks: list[dict]) -> float:
                hi = float("-inf")
                lo = float("inf")
                for t in ticks:
                    mid = t.get("mid")
                    if mid is None:
                        bid = t.get("bid")
                        ask = t.get("ask")
                        if bid is not None and ask is not None:
                            try:
                                mid = (float(bid) + float(ask)) / 2.0
                            except (TypeError, ValueError):
                                mid = None
                        elif bid is not None:
                            try:
                                mid = float(bid)
                            except (TypeError, ValueError):
                                mid = None
                        elif ask is not None:
                            try:
                                mid = float(ask)
                            except (TypeError, ValueError):
                                mid = None
                    if mid is None:
                        continue
                    try:
                        mval = float(mid)
                    except (TypeError, ValueError):
                        continue
                    if mval > hi:
                        hi = mval
                    if mval < lo:
                        lo = mval
                if hi == float("-inf") or lo == float("inf"):
                    return 0.0
                return max(0.0, (hi - lo) / 0.01)

            ticks_5m = tick_window.recent_ticks(300.0, limit=2000)
            ticks_15m = tick_window.recent_ticks(900.0, limit=4000)
            sustained_range_5 = _range_pips_from_ticks(ticks_5m)
            sustained_range_15 = _range_pips_from_ticks(ticks_15m)
            if recent_tick_rows:
                fac_m1["recent_ticks"] = recent_tick_rows
                fac_m1["recent_tick_summary"] = tick_window.summarize(75.0)
            else:
                fac_m1["recent_ticks"] = []
                fac_m1["recent_tick_summary"] = {}
            if loop_counter % 5 == 0:
                latest_epoch = None
                if recent_tick_rows:
                    latest_epoch = float(recent_tick_rows[-1].get("epoch", 0.0) or 0.0)
                age_ms = None
                if latest_epoch:
                    age_ms = max(0, int((now.timestamp() - latest_epoch) * 1000))
                log_metric(
                    "tick_window_recent_count",
                    float(tick_count),
                    tags={"age_ms": age_ms if age_ms is not None else "n/a"},
                    ts=now,
                )

            atr_pips = fac_m1.get("atr_pips")
            if atr_pips is None:
                atr_pips = (fac_m1.get("atr") or 0.0) * 100
            atr_pips = float(atr_pips or 0.0)
            fac_m1["range_5m_pips"] = sustained_range_5
            fac_m1["range_15m_pips"] = sustained_range_15
            vol_5m = float(fac_m1.get("vol_5m", 0.0) or 0.0)
            # 低ボラすぎると全スキップになるため、下限を持ち上げて評価を通す
            if vol_5m < 0.5:
                vol_5m = 0.8
                fac_m1["vol_5m"] = vol_5m
            ema20_raw = fac_m1.get("ema20")
            if ema20_raw is None:
                ema20_raw = fac_m1.get("ma20")
            try:
                ema20_value = float(ema20_raw) if ema20_raw is not None else None
            except (TypeError, ValueError):
                ema20_value = None
            close_raw = fac_m1.get("close")
            try:
                close_px_value = float(close_raw) if close_raw is not None else None
            except (TypeError, ValueError):
                close_px_value = None
            momentum = 0.0
            if close_px_value is not None and ema20_value is not None:
                momentum = close_px_value - ema20_value

            event_soon = False
            global_drawdown_exceeded = check_global_drawdown()

            if global_drawdown_exceeded:
                logging.warning(
                    "[STOP] Global drawdown limit exceeded. Stopping new trades."
                )
                await asyncio.sleep(60)
                continue

            spread_blocked, spread_remain, spread_snapshot, spread_reason = spread_monitor.is_blocked()
            spread_gate_reason = ""
            spread_gate_type = ""
            spread_gate_soft_scalp = False
            spread_stale_for = 0.0
            stale_grace = 0.0
            stale_flag = False
            if spread_snapshot:
                try:
                    stale_flag = bool(spread_snapshot.get("stale"))
                    spread_stale_for = float(spread_snapshot.get("stale_for_sec") or 0.0)
                    stale_grace = float(spread_snapshot.get("stale_grace_sec") or 0.0)
                except (TypeError, ValueError):
                    stale_flag = False
                    spread_stale_for = 0.0
                    stale_grace = 0.0
            if spread_blocked:
                remain_txt = f"{spread_remain}s"
                base_reason = spread_reason or "spread_threshold"
                spread_gate_reason = f"{base_reason} (remain {remain_txt})"
                spread_gate_type = "blocked"
            elif spread_snapshot:
                if stale_flag:
                    if stale_grace <= 0.0 or spread_stale_for >= stale_grace:
                        spread_gate_reason = (
                            f"spread_stale age={spread_snapshot['age_ms']}ms "
                            f"> {spread_snapshot['max_age_ms']}ms"
                        )
                        spread_gate_type = "stale"
                        spread_gate_soft_scalp = True
                    else:
                        spread_gate_type = "stale_grace"
                elif spread_snapshot["spread_pips"] >= spread_snapshot["limit_pips"]:
                    spread_gate_reason = (
                        f"spread_hot {spread_snapshot['spread_pips']:.2f}p "
                        f">= {spread_snapshot['limit_pips']:.2f}p"
                    )
                    spread_gate_type = "hot"
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
                if stale_flag:
                    stale_txt = (
                        f" stale_for={spread_stale_for:.1f}s"
                        + (f"/{stale_grace:.1f}s" if stale_grace else "")
                    )
                else:
                    stale_txt = ""
                spread_log_context = (
                    f"last={last_txt}p avg={avg_txt}p age={age_ms}ms {baseline_txt}{stale_txt}"
                )
            else:
                spread_log_context = "no_snapshot"
            tick_bid = None
            tick_ask = None
            if spread_snapshot:
                try:
                    raw_bid = spread_snapshot.get("bid")
                    if raw_bid is not None:
                        tick_bid = float(raw_bid)
                    raw_ask = spread_snapshot.get("ask")
                    if raw_ask is not None:
                        tick_ask = float(raw_ask)
                except (TypeError, ValueError):
                    tick_bid = tick_ask = None
                if spread_gate_active:
                    if (
                        not last_spread_gate
                        or spread_gate_reason != last_spread_gate_reason
                    ):
                        logging.info(
                            "[SPREAD] gating entries (%s, %s type=%s)",
                            spread_gate_reason,
                            spread_log_context,
                            spread_gate_type or "n/a",
                        )
            elif last_spread_gate:
                logging.info("[SPREAD] entries re-enabled (%s)", spread_log_context)
            last_spread_gate = spread_gate_active
            last_spread_gate_reason = spread_gate_reason

            param_snapshot = param_context.update(
                now=now,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
                spread_snapshot=spread_snapshot,
            )
            if volatility_advisor and volatility_advisor.enabled:
                vol_context = param_snapshot.to_dict()
                vol_context.update(
                    {
                        "range_active": range_active,
                        "event_soon": event_soon,
                        "spread_gate": spread_gate_active,
                    }
                )
                try:
                    vol_hint = await volatility_advisor.advise(vol_context)
                except Exception as exc:  # pragma: no cover
                    logging.debug("[VOL_ADVISOR] failed: %s", exc)
                    vol_hint = None
                if vol_hint and vol_hint.confidence >= 0.35:
                    new_risk = _clamp(param_snapshot.risk_appetite + vol_hint.bias, 0.0, 1.0)
                    if new_risk != param_snapshot.risk_appetite:
                        param_snapshot.risk_appetite = new_risk
                        param_snapshot.notes["vol_bias"] = vol_hint.bias
                        log_metric(
                            "volatility_bias",
                            float(vol_hint.bias),
                            tags={"reason": vol_hint.reason or ""},
                            ts=now,
                        )
            if param_snapshot.volatility_state != last_volatility_state:
                logging.info(
                    "[PARAM] volatility_state=%s atr=%.2fp score=%.2f",
                    param_snapshot.volatility_state,
                    param_snapshot.atr_pips,
                    param_snapshot.atr_score,
                )
                last_volatility_state = param_snapshot.volatility_state
            if param_snapshot.liquidity_state != last_liquidity_state:
                logging.info(
                    "[PARAM] liquidity_state=%s spread=%.2fp score=%.2f",
                    param_snapshot.liquidity_state,
                    param_snapshot.spread_pips,
                    param_snapshot.spread_score,
                )
                last_liquidity_state = param_snapshot.liquidity_state
            if (
                last_risk_appetite is None
                or abs(param_snapshot.risk_appetite - last_risk_appetite) >= 0.12
            ):
                logging.info(
                    "[PARAM] risk_appetite=%.2f trend=%.2f vol=%.2f spread=%.2f",
                    param_snapshot.risk_appetite,
                    param_snapshot.notes.get("trend_score", 0.0),
                    param_snapshot.notes.get("vol_state_score", 0.0),
                    param_snapshot.notes.get("spread_state_score", 0.0),
                )
                last_risk_appetite = param_snapshot.risk_appetite
            if (
                last_vol_high_ratio is None
                or abs(param_snapshot.vol_high_ratio - last_vol_high_ratio) >= 0.08
            ):
                logging.info(
                    "[PARAM] vol_high_ratio=%.2f state=%s",
                    param_snapshot.vol_high_ratio,
                    param_snapshot.volatility_state,
                )
                last_vol_high_ratio = param_snapshot.vol_high_ratio

            story_state = "missing"
            story_snapshot = chart_story.update(fac_m1, fac_h4)
            if story_snapshot:
                fac_m1["story_levels"] = story_snapshot.major_levels
            if story_snapshot:
                story_state = "fresh"
                if last_story_summary != story_snapshot.summary:
                    logging.info(
                        "[STORY] macro=%s micro=%s higher=%s volatility=%s summary=%s",
                        story_snapshot.macro_trend,
                        story_snapshot.micro_trend,
                        story_snapshot.higher_trend,
                        story_snapshot.volatility_state,
                        story_snapshot.summary,
                    )
                    last_story_summary = dict(story_snapshot.summary)
            else:
                story_snapshot = chart_story.last_snapshot
                if story_snapshot:
                    story_state = "reuse"

            if story_snapshot:
                log_metric(
                    "chart_story_snapshot",
                    1.0 if story_state == "fresh" else 0.5,
                    tags={
                        "state": story_state,
                        "macro": story_snapshot.macro_trend,
                        "micro": story_snapshot.micro_trend,
                        "higher": story_snapshot.higher_trend,
                        "volatility": story_snapshot.volatility_state,
                    },
                    ts=now,
                )
            else:
                log_metric(
                    "chart_story_snapshot",
                    0.0,
                    tags={"state": "missing"},
                    ts=now,
                )
            logging.info(
                "[FLOW] story macro=%s micro=%s higher=%s rng5=%.2fp rng15=%.2fp vol5=%.2f atr=%.2f mom=%.4f",
                story_snapshot.macro_trend if story_snapshot else "n/a",
                story_snapshot.micro_trend if story_snapshot else "n/a",
                story_snapshot.higher_trend if story_snapshot else "n/a",
                sustained_range_5,
                sustained_range_15,
                vol_5m,
                atr_pips,
                momentum,
            )

            # 既に独立ワーカー化したストラテジーはメインの関所から除外
            if evaluated_signals:
                filtered_for_main = []
                dropped = []
                for sig in evaluated_signals:
                    strat_name = sig.get("strategy") or sig.get("strategy_tag") or ""
                    if strat_name in DISABLE_MAIN_STRATEGIES:
                        dropped.append(strat_name)
                        continue
                    filtered_for_main.append(sig)
                if dropped:
                    logging.info("[MAIN_SKIP] dropped strategies in main loop: %s", ",".join(sorted(set(dropped))))
                evaluated_signals = filtered_for_main

            def _safe_regime(factors: dict | None, tf: str, last: Optional[str]) -> str:
                try:
                    return classify(factors or {}, tf, event_mode=event_soon)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[REGIME] classify failed tf=%s err=%s; fallback=%s", tf, exc, last or "Mixed")
                    return last or "Mixed"

            macro_regime = _safe_regime(fac_h4, "H4", last_macro_regime)
            micro_regime = _safe_regime(fac_m1, "M1", last_micro_regime)
            last_macro_regime = macro_regime
            last_micro_regime = micro_regime
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
            focus_tag = focus
            try:
                weight_macro = float(w_macro)
            except (TypeError, ValueError):
                logging.warning(
                    "[FOCUS] invalid macro weight from focus_decider (%s); defaulting to 0.0",
                    w_macro,
                )
                weight_macro = 0.0
            soft_range_just_activated = False
            range_ctx = detect_range_mode(fac_m1, fac_h4)
            if AGGRESSIVE_TRADING:
                if range_ctx.active:
                    logging.info(
                        "[RANGE] aggressive override: force inactive (reason=%s)",
                        range_ctx.reason,
                    )
                range_ctx = RangeContext(
                    active=False,
                    reason="aggressive_override",
                    score=range_ctx.score,
                    metrics=range_ctx.metrics,
                    mode="TREND",
                    env_tf=range_ctx.env_tf,
                    macro_tf=range_ctx.macro_tf,
                )
                range_active = False
                range_soft_active = False
                range_entry_counter = 0
                range_exit_counter = 0
            if range_ctx.active != raw_range_active or range_ctx.reason != raw_range_reason:
                logging.info(
                    "[RANGE] detected active=%s reason=%s score=%.2f metrics=%s",
                    range_ctx.active,
                    range_ctx.reason,
                    range_ctx.score,
                    range_ctx.metrics,
                )
            prev_guard_modes = {
                "micro": str((guard_state.get("micro") or {}).get("mode") or "NEUTRAL"),
                "scalp": str((guard_state.get("scalp") or {}).get("mode") or "NEUTRAL"),
                "global": str((guard_state.get("global") or {}).get("mode") or "NEUTRAL"),
            }
            guard_modes = {"micro": "NEUTRAL", "scalp": "NEUTRAL"}
            global_guard_mode = "NEUTRAL"
            if AGGRESSIVE_TRADING:
                guard_modes = {"micro": "TREND", "scalp": "TREND"}
                global_guard_mode = "TREND"
                guard_contexts = {
                    "micro": RangeContext(
                        active=False,
                        reason="aggressive_override",
                        score=0.0,
                        metrics={},
                        mode="TREND",
                        env_tf="H1",
                        macro_tf="H4",
                    ),
                    "scalp": RangeContext(
                        active=False,
                        reason="aggressive_override",
                        score=0.0,
                        metrics={},
                        mode="TREND",
                        env_tf="M5",
                        macro_tf="H4",
                    ),
                    "global": RangeContext(
                        active=False,
                        reason="aggressive_override",
                        score=0.0,
                        metrics={},
                        mode="TREND",
                        env_tf="H4",
                        macro_tf="H4",
                    ),
                }
            else:
                guard_micro = detect_range_mode_for_tf(
                    factors,
                    "H1",
                    macro_tf="H4",
                    range_score_threshold=RANGE_GUARD_RANGE_SCORE_MICRO,
                    trend_score_threshold=RANGE_GUARD_TREND_SCORE_MICRO,
                    adx_threshold=RANGE_GUARD_ADX_MICRO,
                    bbw_threshold=RANGE_GUARD_BBW_MICRO,
                    atr_threshold=RANGE_GUARD_ATR_MICRO,
                    bbw_pips_threshold=RANGE_GUARD_BBW_PIPS_MICRO,
                )
                guard_scalp = detect_range_mode_for_tf(
                    factors,
                    "M5",
                    macro_tf="H4",
                    range_score_threshold=RANGE_GUARD_RANGE_SCORE_SCALP,
                    trend_score_threshold=RANGE_GUARD_TREND_SCORE_SCALP,
                    adx_threshold=RANGE_GUARD_ADX_SCALP,
                    bbw_threshold=RANGE_GUARD_BBW_SCALP,
                    atr_threshold=RANGE_GUARD_ATR_SCALP,
                    bbw_pips_threshold=RANGE_GUARD_BBW_PIPS_SCALP,
                )
                guard_global = detect_range_mode_for_tf(
                    factors,
                    "H4",
                    macro_tf="H4",
                    range_score_threshold=RANGE_GUARD_RANGE_SCORE_GLOBAL,
                    trend_score_threshold=RANGE_GUARD_TREND_SCORE_GLOBAL,
                    adx_threshold=RANGE_GUARD_ADX_GLOBAL,
                    bbw_threshold=RANGE_GUARD_BBW_GLOBAL,
                    atr_threshold=RANGE_GUARD_ATR_GLOBAL,
                    bbw_pips_threshold=RANGE_GUARD_BBW_PIPS_GLOBAL,
                )
                guard_contexts = {
                    "micro": guard_micro,
                    "scalp": guard_scalp,
                    "global": guard_global,
                }
                guard_modes["micro"] = _confirm_guard_mode(guard_micro.mode, guard_state["micro"])
                guard_modes["scalp"] = _confirm_guard_mode(guard_scalp.mode, guard_state["scalp"])
                global_guard_mode = _confirm_guard_mode(guard_global.mode, guard_state["global"])
                if guard_modes["micro"] != prev_guard_modes["micro"]:
                    logging.info(
                        "[MR_GUARD] mode change pocket=micro mode=%s score=%.2f reason=%s",
                        guard_modes["micro"],
                        guard_micro.score,
                        guard_micro.reason,
                    )
                    log_metric(
                        "range_guard_mode",
                        1.0,
                        tags={"pocket": "micro", "mode": guard_modes["micro"]},
                        ts=now,
                    )
                if guard_modes["scalp"] != prev_guard_modes["scalp"]:
                    logging.info(
                        "[MR_GUARD] mode change pocket=scalp mode=%s score=%.2f reason=%s",
                        guard_modes["scalp"],
                        guard_scalp.score,
                        guard_scalp.reason,
                    )
                    log_metric(
                        "range_guard_mode",
                        1.0,
                        tags={"pocket": "scalp", "mode": guard_modes["scalp"]},
                        ts=now,
                    )
                if global_guard_mode != prev_guard_modes["global"]:
                    logging.info(
                        "[MR_GUARD] mode change pocket=global mode=%s score=%.2f reason=%s",
                        global_guard_mode,
                        guard_global.score,
                        guard_global.reason,
                    )
                    log_metric(
                        "range_guard_mode",
                        1.0,
                        tags={"pocket": "global", "mode": global_guard_mode},
                        ts=now,
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
            override_release_active = now < range_breakout_release_until
            breakout_reason = None
            if raw_range_active:
                if abs(momentum) >= RANGE_BREAK_MOMENTUM_RELEASE and atr_pips >= RANGE_BREAK_ATR_MIN:
                    breakout_reason = "momentum_spike"
                elif vol_5m >= RANGE_BREAK_VOL_SPIKE and atr_pips >= max(2.5, RANGE_BREAK_ATR_MIN * 0.85):
                    breakout_reason = "volume_spike"
            if breakout_reason:
                new_until = now + datetime.timedelta(seconds=RANGE_BREAK_RELEASE_SECONDS)
                triggered_now = not override_release_active
                if new_until > range_breakout_release_until:
                    range_breakout_release_until = new_until
                hold_until_candidate = now + datetime.timedelta(seconds=RANGE_BREAK_MACRO_HOLD_SECONDS)
                if hold_until_candidate > range_macro_hold_until:
                    range_macro_hold_until = hold_until_candidate
                override_release_active = True
                if range_breakout_reason != breakout_reason:
                    range_breakout_reason = breakout_reason
                    triggered_now = True
                if triggered_now:
                    logging.info(
                        "[RANGE] breakout override armed reason=%s momentum=%.4f atr=%.2f vol5m=%.2f",
                        range_breakout_reason,
                        momentum,
                        atr_pips,
                        vol_5m,
                    )
                    range_entry_counter = 0
                    range_exit_counter = 0
                    if range_active:
                        range_active = False
                        range_state_since = now
                        last_range_reason = range_breakout_reason
            if override_release_active and not range_override_active_prev:
                logging.info(
                    "[RANGE] breakout override active reason=%s until=%s",
                    range_breakout_reason or raw_range_reason or "unknown",
                    range_breakout_release_until.isoformat(timespec="seconds"),
                )
            elif not override_release_active and range_override_active_prev:
                logging.info(
                    "[RANGE] breakout override expired (reason=%s)",
                    range_breakout_reason or raw_range_reason or "unknown",
                )
                range_breakout_reason = ""
                range_macro_hold_until = datetime.datetime.min
                range_breakout_release_until = datetime.datetime.min
            range_override_active_prev = override_release_active
            soft_range_prev = range_soft_active
            soft_range_candidate = (
                not raw_range_active
                and range_ctx.score >= SOFT_RANGE_SCORE_MIN
                and compression_ratio >= SOFT_RANGE_COMPRESSION_MIN
                and volatility_ratio >= SOFT_RANGE_VOL_MIN
                and effective_adx_m1 <= (adx_threshold + SOFT_RANGE_ADX_BUFFER)
            )
            if AGGRESSIVE_TRADING:
                soft_range_candidate = False
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
            entry_ready = raw_range_active and not override_release_active
            exit_ready = (not raw_range_active) and (range_ctx.score <= RANGE_EXIT_SCORE_CEIL)
            if entry_ready:
                range_entry_counter += 1
            else:
                range_entry_counter = 0
            if exit_ready:
                range_exit_counter += 1
            else:
                range_exit_counter = 0
            if override_release_active and range_active:
                range_active = False
                range_state_since = now
                range_entry_counter = 0
                range_exit_counter = 0
                last_range_reason = range_breakout_reason or range_ctx.reason
                logging.info(
                    "[RANGE] forced release (override reason=%s momentum=%.4f atr=%.2f vol5m=%.2f)",
                    range_breakout_reason or range_ctx.reason,
                    momentum,
                    atr_pips,
                    vol_5m,
                )
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
            if FORCE_SCALP_MODE:
                logging.warning("[FORCE_SCALP] entering GPT stage loop=%d", loop_counter)
            # M1/H4 の移動平均・RSI などの指標をまとめて送信
            payload = {
                "ts": now.isoformat(timespec="seconds"),
                "reg_macro": macro_regime,
                "reg_micro": micro_regime,
                "factors_m1": _compact_factors(fac_m1, GPT_FACTOR_KEYS["M1"]),
                "factors_h4": _compact_factors(fac_h4, GPT_FACTOR_KEYS["H4"]),
                "perf": {
                    pocket: {
                        key: val
                        for key, val in (metrics or {}).items()
                        if key in GPT_PERF_KEYS and val not in (None, "")
                    }
                for pocket, metrics in (perf_cache or {}).items()
                if metrics
                },
                "event_soon": event_soon,
            }
            payload_signature = _gpt_payload_signature(
                payload,
                range_active=range_active,
                range_reason=last_range_reason or range_ctx.reason,
                soft_range=range_soft_active,
            )
            if GPT_DISABLED:
                gpt = {}
                reuse_reason = "disabled"
                logging.info("[GPT] disabled; skipping GPT evaluation and using local ranking only")
            else:
                logging.info(
                    "[GPT] decision_trigger signature=%s range=%s",
                    payload_signature[:10],
                    range_active,
                )
                if await gpt_state.needs_refresh(payload_signature, GPT_MIN_INTERVAL_SECONDS):
                    logging.info("[GPT] enqueue signature=%s", payload_signature[:10])
                    await gpt_requests.submit(payload_signature, payload)
                    try:
                        gpt = await gpt_state.wait_for_signature(payload_signature, timeout=30)
                        reuse_reason = gpt.get("reason") or "live_call"
                    except asyncio.TimeoutError:
                        logging.warning(
                            "[GPT] wait for signature %s timed out; using previous decision",
                            payload_signature[:10],
                        )
                        gpt, _, _ = await gpt_state.get_latest()
                        reuse_reason = gpt.get("reason") or "cached_timeout"
                else:
                    gpt, _, _ = await gpt_state.get_latest()
                    reuse_reason = gpt.get("reason") or "cached"
                if not isinstance(gpt, dict):
                    logging.warning(
                        "[GPT] invalid decision payload type=%s; using empty dict", type(gpt)
                    )
                    gpt = {}
            local_decision: dict = {}
            try:
                local_decision = heuristic_decision(payload, last_local_decision)
                if isinstance(local_decision, dict):
                    last_local_decision = local_decision
                else:
                    local_decision = {}
            except Exception as exc:
                logging.warning("[LOCAL_DECIDER] failed: %s", exc)
                local_decision = {}
            raw_weight_scalp = gpt.get("weight_scalp")
            if raw_weight_scalp is None:
                weight_scalp_display = "n/a"
            else:
                try:
                    weight_scalp_display = f"{float(raw_weight_scalp):.2f}"
                except (TypeError, ValueError):
                    weight_scalp_display = "invalid"
            raw_weight_macro = gpt.get("weight_macro")
            weight_macro_override: Optional[float] = None
            if raw_weight_macro is not None:
                try:
                    weight_macro_override = max(0.0, min(1.0, float(raw_weight_macro)))
                except (TypeError, ValueError):
                    logging.warning(
                        "[GPT] invalid weight_macro=%s; keeping focus_decider value %.2f",
                        raw_weight_macro,
                        weight_macro,
                    )
                    weight_macro_override = None
            if MACRO_DISABLED:
                weight_macro_override = 0.0
            logging.info(
                "[GPT] mode=%s risk=%s liq=%s range=%.2f focus=%s weight_macro=%.2f weight_scalp=%s model=%s reason=%s",
                gpt.get("mode"),
                gpt.get("risk_bias"),
                gpt.get("liquidity_bias"),
                _safe_float(gpt.get("range_confidence"), 0.0),
                gpt.get("focus_tag"),
                weight_macro_override if weight_macro_override is not None else weight_macro,
                weight_scalp_display,
                gpt.get("model_used", "unknown"),
                reuse_reason,
            )
            forecast_bias = str(gpt.get("forecast_bias") or "").lower() if gpt else ""
            forecast_conf = float(gpt.get("forecast_confidence") or 0.0) if gpt else 0.0
            forecast_horizon = gpt.get("forecast_horizon_min")
            if forecast_bias not in {"up", "down", "flat"}:
                forecast_bias = ""
                forecast_conf = 0.0
                forecast_horizon = None
            # --- ローカル順位付けに置換（GPTは順位ヒントのみ） ---
            all_strats = list(STRATEGIES.keys())
            gpt_rank = list(local_decision.get("ranked_strategies") or [])
            # MACRO 禁止時は除外
            if MACRO_DISABLED and gpt_rank:
                before = list(gpt_rank)
                gpt_rank = [s for s in gpt_rank if s not in MACRO_STRATEGIES]
                if gpt_rank != before:
                    logging.info("[MACRO_DISABLED] Filtered macro strategies %s -> %s", before, gpt_rank)
            # ローカルスコアリング（レジーム/ATR/vol/MA/セッション/範囲）
            session_bucket = _session_bucket(now)
            ranked_strategies = _local_strategy_ranking(
                strategies=all_strats,
                fac_m1=fac_m1,
                fac_h4=fac_h4,
                range_ctx=range_ctx,
                session_bucket=session_bucket,
                last_gpt_mode=str(gpt.get("mode")) if gpt else None,
                last_focus=gpt.get("focus_tag") if gpt else None,
            )
            # GPT順位をヒントとして微調整（上位に加点）
            if gpt_rank:
                bonus = {name: (len(gpt_rank) - idx) * 0.01 for idx, name in enumerate(gpt_rank)}
                ranked_strategies = sorted(
                    ranked_strategies,
                    key=lambda n: bonus.get(n, 0.0),
                    reverse=True,
                )
            gpt_strategy_allowlist = set(all_strats)  # フィルタしない
            if not ranked_strategies:
                ranked_strategies = all_strats
            logging.info(
                "[STRAT_EVAL_PRE] ranked_strategies(local)=%s focus=%s session=%s range=%s",
                ranked_strategies[:8],
                focus_tag,
                session_bucket,
                range_active,
            )
            auto_injected_strategies: set[str] = set()
            weight_scalp = None
            if raw_weight_scalp is not None:
                try:
                    weight_scalp = max(0.0, min(1.0, float(raw_weight_scalp)))
                except (TypeError, ValueError):
                    weight_scalp = None
            if weight_macro_override is not None:
                weight_macro = weight_macro_override
            focus_tag = gpt.get("focus_tag") or focus_tag
            if MACRO_DISABLED and focus_tag in {"macro", "hybrid"}:
                focus_tag = "micro"
            focus_override_hint = None
            if focus_advisor and focus_advisor.enabled:
                focus_context = {
                    "reg_macro": macro_regime,
                    "reg_micro": micro_regime,
                    "range_active": range_active,
                    "range_reason": last_range_reason or range_ctx.reason,
                    "event_soon": event_soon,
                    "d1": _factor_snapshot(fac_d1),
                    "h4": _factor_snapshot(fac_h4),
                    "gpt_focus": gpt.get("focus_tag"),
                    "gpt_weight_macro": gpt.get("weight_macro"),
                    "gpt_weight_scalp": gpt.get("weight_scalp"),
                }
                try:
                    focus_override_hint = await focus_advisor.advise(focus_context)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.debug("[FOCUS_ADVISOR] failed: %s", exc)
                    focus_override_hint = None

            # Update realtime metrics cache every few minutes
            if (now - last_metrics_refresh).total_seconds() >= 240:
                try:
                    metrics_client.refresh()
                    strategy_health_cache.clear()
                    last_metrics_refresh = now
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[REALTIME] metrics refresh failed: %s", exc)

            if close_px_value is None:
                try:
                    close_px_value = float(fac_m1.get("close") or 0.0)
                except (TypeError, ValueError):
                    close_px_value = None
            if ema20_value is None:
                try:
                    ema_fallback = fac_m1.get("ema20") or fac_m1.get("ma20")
                    ema20_value = float(ema_fallback) if ema_fallback is not None else None
                except (TypeError, ValueError):
                    ema20_value = None
            if (close_px_value is not None and ema20_value is not None) and abs(momentum) < 1e-6:
                momentum = close_px_value - ema20_value
            momentum_abs = abs(momentum)
            scalp_ready = False
            if range_active:
                scalp_ready = (
                    atr_pips >= RANGE_SCALP_ATR_MIN
                    and vol_5m >= RANGE_SCALP_VOL_MIN
                    and momentum_abs <= RANGE_SCALP_MAX_MOMENTUM
                )
                if last_range_scalp_ready is None or scalp_ready != last_range_scalp_ready:
                    logging.info(
                        "[SCALP-MAIN] Range scalp %s momentum=%.4f (≤%.4f) atr=%.2f (≥%.2f) vol5m=%.2f (≥%.2f)",
                        "ready" if scalp_ready else "blocked",
                        momentum,
                        RANGE_SCALP_MAX_MOMENTUM,
                        atr_pips,
                        RANGE_SCALP_ATR_MIN,
                        vol_5m,
                        RANGE_SCALP_VOL_MIN,
                    )
                last_range_scalp_ready = scalp_ready
            else:
                scalp_ready = (
                    (atr_pips >= 0.60 and momentum_abs >= 0.00030)
                    or (atr_pips >= 0.50 and vol_5m and vol_5m >= 0.25)
                )
                if not scalp_ready and momentum_abs >= 0.0007:
                    scalp_ready = atr_pips >= 0.50
                if (
                    not scalp_ready
                    and atr_pips >= 0.50
                    and (
                        sustained_range_5 >= 1.2
                        or sustained_range_15 >= 2.4
                    )
                ):
                    scalp_ready = True
                    logging.info(
                        "[SCALP-MAIN] Sustained range unlock range5=%.2f range15=%.2f atr=%.2f vol5m=%s mom=%.4f",
                        sustained_range_5,
                        sustained_range_15,
                        atr_pips,
                        f"{vol_5m:.2f}" if vol_5m is not None else "n/a",
                        momentum,
                    )
                last_range_scalp_ready = None
            if (
                not scalp_ready
                and weight_scalp is not None
                and weight_scalp >= SCALP_WEIGHT_FLOOR
            ):
                scalp_ready = True
                scalp_ready_forced = True
                logging.info(
                    "[SCALP-MAIN] Forcing readiness (weight=%.2f floor=%.2f atr=%.2f vol5m=%s momentum=%.4f)",
                    weight_scalp,
                    SCALP_WEIGHT_FLOOR,
                    atr_pips,
                    f"{vol_5m:.2f}" if vol_5m is not None else "n/a",
                    momentum,
                )
            if FORCE_SCALP_MODE:
                if not scalp_ready:
                    scalp_ready = True
                    scalp_ready_forced = True
                    logging.info(
                        "[SCALP-MAIN] Force mode -> readiness enabled (atr=%.2f vol5m=%s momentum=%.4f)",
                        atr_pips,
                        f"{vol_5m:.2f}" if vol_5m is not None else "n/a",
                        momentum,
                    )
                target_force_weight = max(SCALP_WEIGHT_READY_FLOOR, SCALP_WEIGHT_FLOOR)
                if weight_scalp is None or weight_scalp < target_force_weight:
                    prev_weight_scalp = weight_scalp
                    weight_scalp = target_force_weight
                    logging.info(
                        "[SCALP-MAIN] Force mode -> weight uplift %s -> %.2f",
                        f"{prev_weight_scalp:.2f}" if prev_weight_scalp is not None else "None",
                        weight_scalp,
                    )
                if weight_macro + (weight_scalp or 0.0) > 1.0:
                    excess = weight_macro + (weight_scalp or 0.0) - 1.0
                    prev_macro = weight_macro
                    weight_macro = max(0.0, round(weight_macro - excess, 3))
                    logging.info(
                        "[SCALP-MAIN] Force mode -> macro weight trimmed %.2f -> %.2f to balance shares",
                        prev_macro,
                        weight_macro,
                    )

            if range_active:
                focus_tag = "micro"
                weight_macro = min(weight_macro, 0.15)
                if weight_scalp is not None:
                    weight_scalp = min(weight_scalp, 0.2)
            elif range_soft_active and focus_tag == "macro":
                if soft_range_just_activated:
                    logging.info(
                        "[FOCUS] Soft range compression forcing hybrid focus (score=%.2f).",
                        range_ctx.score,
                    )
                focus_tag = "hybrid"
            if not range_active and range_soft_active and weight_macro > SOFT_RANGE_WEIGHT_CAP:
                prev_weight = weight_macro
                weight_macro = min(weight_macro, SOFT_RANGE_WEIGHT_CAP)
                if prev_weight != weight_macro:
                    logging.info(
                        "[MACRO] Soft range compression (score=%.2f eff_adx=%.2f) weight_macro %.2f -> %.2f",
                        range_ctx.score,
                        effective_adx_m1,
                        prev_weight,
                        weight_macro,
                    )
            if focus_override_hint and focus_override_hint.confidence >= 0.4:
                if focus_override_hint.focus_tag:
                    focus_tag = focus_override_hint.focus_tag
                if focus_override_hint.weight_macro is not None:
                    weight_macro = _clamp(float(focus_override_hint.weight_macro), 0.0, 1.0)
                if focus_override_hint.weight_scalp is not None:
                    weight_scalp = _clamp(float(focus_override_hint.weight_scalp), 0.0, 1.0)
            override_macro_hold_active = now < range_macro_hold_until
            if override_macro_hold_active:
                prev_focus = focus_tag
                if focus_tag == "macro":
                    focus_tag = "micro"
                prev_macro_weight = weight_macro
                weight_macro = min(weight_macro, 0.18)
                if weight_scalp is not None:
                    weight_scalp = min(max(weight_scalp, 0.0), 0.25)
                micro_weight_est = 1.0 - (weight_macro + (weight_scalp or 0.0))
                if micro_weight_est < MICRO_BREAKOUT_WEIGHT_FLOOR:
                    shortfall = MICRO_BREAKOUT_WEIGHT_FLOOR - micro_weight_est
                    if weight_scalp is not None and weight_scalp > 0.1:
                        give = min(shortfall, weight_scalp - 0.1)
                        if give > 0:
                            weight_scalp = round(weight_scalp - give, 3)
                            shortfall = round(shortfall - give, 3)
                    if shortfall > 0:
                        weight_macro = round(max(0.0, weight_macro - shortfall), 3)
                if weight_macro != prev_macro_weight or focus_tag != prev_focus:
                    logging.info(
                        "[FOCUS] Macro hold active -> focus=%s weight_macro=%.2f (was %.2f) weight_scalp=%s",
                        focus_tag,
                        weight_macro,
                        prev_macro_weight,
                        f"{weight_scalp:.2f}" if weight_scalp is not None else "n/a",
                    )
            if weight_scalp is not None:
                scalp_floor = SCALP_WEIGHT_FLOOR
                if scalp_ready:
                    scalp_floor = max(scalp_floor, SCALP_WEIGHT_READY_FLOOR)
                if focus_tag in {"micro", "hybrid"} or scalp_ready:
                    desired_min = scalp_floor
                else:
                    desired_min = max(SCALP_WEIGHT_FLOOR * 0.5, 0.05)
                cap = max(0.0, 1.0 - weight_macro)
                if desired_min > 0 and desired_min > cap + 1e-6:
                    reduction = min(weight_macro, desired_min - cap)
                    if reduction > 0:
                        prev_macro_weight = weight_macro
                        weight_macro = round(max(0.0, weight_macro - reduction), 3)
                        cap = max(0.0, 1.0 - weight_macro)
                        logging.info(
                            "[FOCUS] Reduced macro weight %.2f -> %.2f to honor scalp floor %.2f (focus=%s)",
                            prev_macro_weight,
                            weight_macro,
                            desired_min,
                            focus_tag,
                        )
                target = min(cap, desired_min)
                if target > 0 and weight_scalp + 1e-6 < target:
                    prev_weight_scalp = weight_scalp
                    weight_scalp = round(target, 3)
                    logging.info(
                        "[FOCUS] Scalp weight floor %.2f -> %.2f (cap=%.2f focus=%s ready=%s)",
                        prev_weight_scalp,
                        weight_scalp,
                        cap,
                        focus_tag,
                        scalp_ready,
                    )
            focus_pockets = set(FOCUS_POCKETS.get(focus_tag, ("macro", "micro", "scalp")))
            if weight_scalp is None and "scalp" in focus_pockets:
                default_scalp = SCALP_WEIGHT_READY_FLOOR if scalp_ready else SCALP_WEIGHT_FLOOR
                weight_scalp = round(default_scalp, 3)
                if weight_macro + weight_scalp > 1.0:
                    excess = weight_macro + weight_scalp - 1.0
                    prev_macro = weight_macro
                    weight_macro = round(max(0.0, weight_macro - excess), 3)
                    logging.info(
                        "[FOCUS] Trimmed macro weight %.2f -> %.2f to fit scalp default %.2f",
                        prev_macro,
                        weight_macro,
                        weight_scalp,
                    )
                logging.info(
                    "[FOCUS] Applied default scalp weight %.2f (ready=%s focus=%s)",
                    weight_scalp,
                    scalp_ready,
                    focus_tag,
                )
            stage_base = _compute_stage_base(
                param_snapshot=param_snapshot,
                range_active=range_active,
                focus_tag=focus_tag,
                weight_macro=weight_macro,
                weight_scalp=weight_scalp,
            )
            stage_overrides, stage_changed, stage_biases = param_context.stage_overrides(
                stage_base,
                range_active=range_active,
            )
            if stage_plan_advisor and stage_plan_advisor.enabled:
                stage_context = {
                    "range_active": range_active,
                    "risk_appetite": param_snapshot.risk_appetite,
                    "stage_bias": stage_biases,
                    "weight_macro": weight_macro,
                    "weight_scalp": weight_scalp if weight_scalp is not None else 0.0,
                }
                try:
                    stage_hint = await stage_plan_advisor.advise(stage_context)
                except Exception as exc:  # pragma: no cover
                    logging.debug("[STAGE_PLAN] failed: %s", exc)
                    stage_hint = None
                if stage_hint and stage_hint.confidence >= 0.4:
                    stage_overrides.update(stage_hint.plans)
                    stage_changed = True
            _set_stage_plan_overrides(stage_overrides)
            vol_ratio = param_snapshot.vol_high_ratio if param_snapshot else -1.0
            if stage_changed:
                log_bias = {k: round(v, 2) for k, v in stage_biases.items()}
                first_stage = {k: round(plan[0], 3) for k, plan in stage_overrides.items()}
                logging.info(
                    "[STAGE] dynamic_plan=%s first=%s bias=%s range=%s vol_high=%.2f",
                    stage_overrides,
                    first_stage,
                    log_bias,
                    range_active,
                    vol_ratio,
                )
            elif any(
                abs(stage_biases.get(k, 1.0) - last_stage_biases.get(k, 1.0)) >= 0.08
                for k in stage_biases
            ):
                log_bias = {k: round(v, 2) for k, v in stage_biases.items()}
                logging.info(
                    "[STAGE] bias_adjust drift=%s range=%s vol_high=%.2f",
                    log_bias,
                    range_active,
                    vol_ratio,
                )
            last_stage_biases = dict(stage_biases)

            # アカウント情報はクランプ閾値にも使うため早めに取得する
            account_equity = FALLBACK_EQUITY
            # クランプ検知のためのオープン玉情報
            clamp_positions = pos_manager.get_open_positions()
            open_scalp_trades = 0
            try:
                open_scalp_trades = len(
                    clamp_positions.get("scalp", {}).get("open_trades", []) or []
                )
            except Exception:
                open_scalp_trades = 0
            atr_m5_val = None
            try:
                if fac_m5 and fac_m5.get("atr_pips") is not None:
                    atr_m5_val = float(fac_m5["atr_pips"])
            except Exception:
                atr_m5_val = None

            stage_tracker.update_loss_streaks(
                now=now,
                cooldown_map=POCKET_LOSS_COOLDOWNS,
                range_active=range_active,
                atr_pips=atr_pips,
                vol_5m=vol_5m,
                adx_m1=_safe_float(fac_m1.get("adx")),
                momentum=momentum,
                nav=account_equity,
                open_scalp_positions=open_scalp_trades,
                atr_m5_pips=atr_m5_val,
            )
            clamp_state = stage_tracker.get_clamp_state(now=now)
            recent_profiles = stage_tracker.get_recent_profiles()
            if last_logged_range_state is None or last_logged_range_state != range_active:
                log_metric(
                    "range_mode_active",
                    1.0 if range_active else 0.0,
                    tags={
                        "reason": last_range_reason or range_ctx.reason or "unknown",
                        "score": round(range_ctx.score, 3),
                    },
                    ts=now,
                )
                last_logged_range_state = range_active
            focus_candidates = set(FOCUS_POCKETS.get(focus_tag, ("macro", "micro", "scalp")))
            if not focus_candidates:
                focus_candidates = {"micro"}
            strategy_pockets = {
                STRATEGIES[s].pocket
                for s in ranked_strategies
                if STRATEGIES.get(s)
            }
            if not ranked_strategies:
                logging.info(
                    "[WAIT] ranked_strategies empty focus=%s pockets=%s range=%s soft=%s",
                    focus_tag,
                    ",".join(sorted(focus_pockets)),
                    range_active,
                    range_soft_active,
                )
            macro_hint = max(min(weight_macro, 1.0), 0.0)
            scalp_hint = 0.0
            if weight_scalp is not None:
                scalp_hint = max(min(weight_scalp, 1.0), 0.0)
            micro_hint = max(1.0 - macro_hint - scalp_hint, 0.0)

            stage_tracker.set_weight_hint("macro", macro_hint)
            stage_tracker.set_weight_hint("micro", micro_hint)
            stage_tracker.set_weight_hint("scalp", scalp_hint if weight_scalp is not None else None)
            # Pocketフィルタは無効化し、全ポケットを常に評価対象にする
            focus_pockets = {"macro", "micro", "scalp"}
            diag_fields = {
                "ready": int(bool(scalp_ready)),
                "forced": int(bool(scalp_ready_forced)),
                "focus": focus_tag,
                "range": range_active,
                "soft": range_soft_active,
                "atr": round(atr_pips, 2) if atr_pips is not None else "n/a",
                "vol5": round(vol_5m, 2) if vol_5m is not None else "n/a",
                "momentum": round(momentum, 4),
                "weight": round(weight_scalp, 3) if weight_scalp is not None else "n/a",
                "rng5": round(sustained_range_5, 2),
                "rng15": round(sustained_range_15, 2),
                "pockets": ",".join(sorted(focus_pockets)),
                "strategies": ",".join(ranked_strategies[:4]) if ranked_strategies else "-",
            }
            logging.info(
                "[SCALP_READY] %s",
                " ".join(f"{k}={v}" for k, v in diag_fields.items()),
            )
            if "scalp" in focus_pockets and "M1Scalper" not in ranked_strategies:
                logging.info(
                    "[SCALP_FLOW] missing_strategy ready=%s weight=%s strategies=%s",
                    scalp_ready,
                    f"{weight_scalp:.3f}" if weight_scalp is not None else "n/a",
                    ranked_strategies,
                )
            if range_active and "macro" in focus_pockets:
                logging.info(
                    "[RANGE_MACRO_KEEP] range_active macro_regime=%s macro pocket kept",
                    macro_regime or "unknown",
                )

            if (
                last_logged_focus != focus_tag
                or last_logged_weight is None
                or abs((last_logged_weight or 0.0) - float(weight_macro or 0.0)) >= 0.05
            ):
                log_metric(
                    "weight_macro",
                    float(weight_macro or 0.0),
                    tags={
                        "focus_tag": focus_tag,
                        "range_active": str(range_active),
                    },
                    ts=now,
                )
                last_logged_focus = focus_tag
                last_logged_weight = float(weight_macro or 0.0)
            if weight_scalp is not None:
                if (
                    last_logged_scalp_weight is None
                    or abs(last_logged_scalp_weight - float(weight_scalp or 0.0)) >= 0.05
                ):
                    log_metric(
                        "weight_scalp",
                        float(weight_scalp or 0.0),
                        tags={
                            "focus_tag": focus_tag,
                            "range_active": str(range_active),
                        },
                        ts=now,
                    )
                    last_logged_scalp_weight = float(weight_scalp or 0.0)

            ma10_h4 = fac_h4.get("ma10")
            ma20_h4 = fac_h4.get("ma20")
            adx_h4 = fac_h4.get("adx", 0.0)
            slope_gap = abs((ma10_h4 or 0.0) - (ma20_h4 or 0.0))
            low_trend = (
                adx_h4 <= LOW_TREND_ADX_THRESHOLD
                and slope_gap <= LOW_TREND_SLOPE_THRESHOLD
            )
            if low_trend and "macro" in focus_pockets:
                prev_weight = weight_macro
                weight_macro = min(weight_macro, LOW_TREND_WEIGHT_CAP)
                logging.info(
                    "[MACRO] H4 trend weak (ADX %.2f gap %.5f). weight_macro %.2f -> %.2f",
                    adx_h4,
                    slope_gap,
                    prev_weight,
                    weight_macro,
                )

            scalp_weight_ok = weight_scalp is None or weight_scalp >= SCALP_AUTO_MIN_WEIGHT
            if FORCE_SCALP_MODE:
                scalp_weight_ok = True
            if (
                scalp_ready
                and "scalp" in focus_pockets
                and "M1Scalper" not in ranked_strategies
                and scalp_weight_ok
            ):
                ranked_strategies.append("M1Scalper")
                auto_injected_strategies.add("M1Scalper")
            if os.getenv("SCALP_TACTICAL", "0").strip().lower() not in {"","0","false","no"}:
                focus_pockets.add("scalp")
                if "M1Scalper" not in ranked_strategies:
                    ranked_strategies.append("M1Scalper")
                    auto_injected_strategies.add("M1Scalper")
                if "BB_RSI" not in ranked_strategies:
                    ranked_strategies.append("BB_RSI")
                    auto_injected_strategies.add("BB_RSI")

                logging.info(
                    "[SCALP-MAIN] Auto-added M1Scalper (mode=%s ATR %.2f momentum %.4f vol5m %.2f).",
                    "range" if range_active else "trend",
                    atr_pips,
                    momentum,
                    vol_5m,
                )

            # Range mode: prefer mean-reversion scalping. Ensure RangeFader is present.
            if (
                range_active
                and "scalp" in focus_pockets
                and "RangeFader" not in ranked_strategies
                and scalp_weight_ok
            ):
                ranked_strategies.append("RangeFader")
                auto_injected_strategies.add("RangeFader")
                logging.info(
                    "[SCALP-MAIN] Range mode: auto-added RangeFader (score=%.2f bbw=%.2f atr=%.2f).",
                    range_ctx.score,
                    fac_m1.get("bbw", 0.0) or 0.0,
                    atr_pips,
                )
            if range_active and "micro" in focus_pockets:
                injected: list[str] = []
                for sname in ("MicroVWAPBound", "MicroRangeBreak", "MicroVWAPRevert", "BB_RSI_Fast"):
                    if sname not in ranked_strategies:
                        ranked_strategies.insert(0, sname)
                        auto_injected_strategies.add(sname)
                        injected.append(sname)
                if injected:
                    logging.info(
                        "[MICRO-RANGE] Auto-added %s (range score=%.2f adx=%.2f bbw=%.2f atr=%.2f).",
                        ",".join(injected),
                        range_ctx.score,
                        fac_m1.get("adx", 0.0) or 0.0,
                        fac_m1.get("bbw", 0.0) or 0.0,
                        atr_pips,
                    )
            if (
                not range_active
                and scalp_ready
                and "scalp" in focus_pockets
                and "PulseBreak" not in ranked_strategies
                and scalp_weight_ok
                and momentum_abs >= PULSEBREAK_AUTO_MOM_MIN
                and atr_pips >= PULSEBREAK_AUTO_ATR_MIN
                and (vol_5m or 0.0) >= PULSEBREAK_AUTO_VOL_MIN
            ):
                ranked_strategies.append("PulseBreak")
                auto_injected_strategies.add("PulseBreak")
                logging.info(
                    "[SCALP-MAIN] Auto-added PulseBreak (mom=%.4f atr=%.2f vol5m=%.2f).",
                    momentum,
                    atr_pips,
                    vol_5m,
                )

            open_positions_snapshot = pos_manager.get_open_positions()
            evaluated_signals: list[dict] = []
            signals: list[dict] = []
            signal_emitted = False
            evaluated_count = 0
            bus_signals: list[dict] = []
            price_hint = _safe_float(fac_m1.get("close")) or _safe_float(fac_m1.get("mid"))
            if SIGNAL_GATE_ENABLED:
                try:
                    raw_bus = signal_bus.fetch(limit=SIGNAL_GATE_FETCH_LIMIT)
                except Exception as exc:
                    raw_bus = []
                    logging.warning("[SIGNAL_GATE] fetch failed: %s", exc)
                for raw_sig in raw_bus:
                    norm = _normalize_bus_signal(raw_sig, price_hint)
                    if norm:
                        if _mr_guard_reject(norm, context="bus"):
                            continue
                        tag = norm.get("strategy") or norm.get("strategy_tag") or norm.get("tag")
                        profile = norm.get("profile") or norm.get("strategy_profile")
                        if _is_mr_signal(tag, profile):
                            guard_note = _mr_guard_snapshot(norm.get("pocket"))
                            norm["mr_guard"] = guard_note
                            norm.setdefault("meta", {})["mr_guard"] = guard_note
                        bus_signals.append(norm)
                if bus_signals:
                    try:
                        logging.info(
                            "[SIGNAL_GATE] bus=%d tags=%s",
                            len(bus_signals),
                            ",".join(
                                sorted(
                                    {
                                        s.get("strategy")
                                        or s.get("strategy_tag")
                                        or s.get("tag")
                                        or "unknown"
                                        for s in bus_signals
                                    }
                                )
                            ),
                        )
                    except Exception:
                        pass
            evaluated_signals = list(bus_signals)
            signals = list(bus_signals)
            evaluated_count = len(bus_signals)
            signal_emitted = bool(bus_signals)
            shock_ctx = _shock_state(fac_m1)
            h4_dir = 0
            try:
                ma10_h4_val = float(fac_h4.get("ma10") or 0.0)
                ma20_h4_val = float(fac_h4.get("ma20") or 0.0)
                if ma10_h4_val and ma20_h4_val:
                    if ma10_h4_val > ma20_h4_val:
                        h4_dir = 1
                    elif ma10_h4_val < ma20_h4_val:
                        h4_dir = -1
            except Exception:
                h4_dir = 0
            adx_h4_val = float(fac_h4.get("adx") or 0.0)
            logging.info(
                "[STRAT_EVAL_BEGIN] ranked=%s pockets=%s allow=%s focus=%s range=%s atr=%.2f vol5=%s momentum=%.4f",
                ranked_strategies,
                ",".join(sorted(focus_pockets)),
                sorted(gpt_strategy_allowlist) if gpt_strategy_allowlist else ["*"],
                focus_tag,
                range_active,
                atr_pips if atr_pips is not None else -1.0,
                f"{vol_5m:.2f}" if vol_5m is not None else "n/a",
                momentum,
            )
            # Safety: if GPT allowlist is empty, fall back to full universe to avoid stalls.
            if not gpt_strategy_allowlist:
                gpt_strategy_allowlist = set(STRATEGIES.keys())
                logging.info("[STRAT_GUARD] GPT allowlist empty; falling back to all strategies.")
            if FORCE_SCALP_MODE:
                logging.warning(
                    "[FORCE_SCALP] ranked_strategies=%s focus=%s pockets=%s ready=%s",
                    ranked_strategies,
                    focus_tag,
                    sorted(focus_pockets),
                    scalp_ready,
                )
            for sname in ranked_strategies:
                cls = STRATEGIES.get(sname)
                if not cls:
                    continue
                evaluated_count += 1
                pocket = cls.pocket
                rsi_val = fac_m1.get("rsi")
                adx_val = fac_m1.get("adx")
                ma10_val = fac_m1.get("ma10")
                ma20_val = fac_m1.get("ma20")
                close_val = fac_m1.get("close")
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
                # ポケット別フィルタは全て無効化
                if range_active and cls.name not in ALLOWED_RANGE_STRATEGIES:
                    logging.info("[RANGE] skip %s in range mode.", sname)
                    continue
                # GPT allowlist は参考のみ。スキップしない。
                raw_signal = cls.check(fac_m1)
                if not raw_signal:
                    logging.info(
                        "[STRAT_SKIP] %s pocket=%s signal=None focus=%s range=%s atr=%.2f vol5=%s momentum=%.4f rsi=%s adx=%s ma10=%s ma20=%s close=%s",
                        sname,
                        pocket,
                        focus_tag,
                        range_active,
                        atr_pips if atr_pips is not None else -1.0,
                        f"{vol_5m:.2f}" if vol_5m is not None else "n/a",
                        momentum,
                        f"{_safe_float(rsi_val):.1f}" if rsi_val is not None else "n/a",
                        f"{_safe_float(adx_val):.1f}" if adx_val is not None else "n/a",
                        f"{_safe_float(ma10_val):.4f}" if ma10_val is not None else "n/a",
                        f"{_safe_float(ma20_val):.4f}" if ma20_val is not None else "n/a",
                        f"{_safe_float(close_val):.4f}" if close_val is not None else "n/a",
                    )
                    if FORCE_SCALP_MODE and pocket == "scalp":
                        logging.warning("[FORCE_SCALP] %s returned None", sname)
                    continue

                strategy_tag = raw_signal.get("tag", cls.name) if isinstance(raw_signal, dict) else cls.name
                strategy_profile = None
                if isinstance(raw_signal, dict):
                    strategy_profile = raw_signal.get("profile") or raw_signal.get("strategy_profile")
                if _is_mr_signal(strategy_tag, strategy_profile):
                    local_mode = guard_modes.get(pocket, "NEUTRAL")
                    if local_mode != "RANGE" or global_guard_mode == "TREND":
                        logging.info(
                            "[MR_GUARD] skip=local strategy=%s pocket=%s local=%s global=%s",
                            sname,
                            pocket,
                            local_mode,
                            global_guard_mode,
                        )
                        continue

                health = strategy_health_cache.get(sname)
                if not health:
                    health = metrics_client.evaluate(sname, cls.pocket)
                    health = confidence_policy.apply(health)
                    strategy_health_cache[sname] = health

                if not health.allowed:
                    if FORCE_SCALP_MODE and pocket == "scalp":
                        logging.warning(
                            "[FORCE_SCALP] overriding health block %s reason=%s",
                            sname,
                            health.reason,
                        )
                        health.allowed = True
                        health.reason = None
                        if health.confidence_scale < 0.75:
                            health.confidence_scale = 0.75
                    else:
                        logging.info(
                            "[HEALTH] skip %s pocket=%s reason=%s conf_scale=%.2f",
                            sname,
                            cls.pocket,
                            health.reason,
                            health.confidence_scale,
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
                if raw_signal.get("profile") or raw_signal.get("strategy_profile"):
                    signal["profile"] = raw_signal.get("profile") or raw_signal.get("strategy_profile")
                entry_thesis_raw = raw_signal.get("entry_thesis")
                if isinstance(entry_thesis_raw, dict):
                    signal["entry_thesis"] = entry_thesis_raw
                execution_raw = raw_signal.get("execution")
                if isinstance(execution_raw, dict):
                    signal["execution"] = execution_raw
                if _is_mr_signal(signal.get("tag"), signal.get("profile")):
                    guard_note = _mr_guard_snapshot(pocket)
                    signal["mr_guard"] = guard_note
                    signal.setdefault("meta", {})["mr_guard"] = guard_note
                # ランクに応じて confidence を微調整（上位ほど少し増やす）
                if isinstance(raw_signal, dict):
                    rank_pos = max(0, ranked_strategies.index(sname)) if sname in ranked_strategies else 0
                    if len(ranked_strategies) > 1:
                        rank_boost = 1.0 + 0.12 * (len(ranked_strategies) - rank_pos - 1) / (len(ranked_strategies) - 1)
                    else:
                        rank_boost = 1.0
                    try:
                        conf = float(raw_signal.get("confidence", 50))
                        raw_signal["confidence"] = int(min(100, conf * rank_boost))
                    except Exception:
                        pass
                    # GPTの方向バイアスを confidence に反映（軽めの重み）
                    if forecast_bias and raw_signal.get("action") in {"OPEN_LONG", "OPEN_SHORT"}:
                        horizon = float(forecast_horizon or 0)
                        aligned = (forecast_bias == "up" and raw_signal["action"] == "OPEN_LONG") or (
                            forecast_bias == "down" and raw_signal["action"] == "OPEN_SHORT"
                        )
                        # 短期( <60min )はスカルプ/マイクロ寄り、長期はマクロ寄りの補正を強める
                        if horizon >= 60 and pocket == "macro":
                            mult = 0.2
                        elif horizon < 60 and pocket != "macro":
                            mult = 0.15
                        else:
                            mult = 0.08
                        bias_factor = 1.0 + (mult * forecast_conf if aligned else -mult * forecast_conf)
                        bias_factor = max(0.7, min(1.3, bias_factor))
                        try:
                            conf2 = float(raw_signal.get("confidence", 50))
                            raw_signal["confidence"] = int(min(100, max(1, conf2 * bias_factor)))
                            if aligned:
                                raw_signal.setdefault("meta", {})["forecast_bias"] = forecast_bias
                            else:
                                raw_signal.setdefault("meta", {})["forecast_bias"] = f"opp:{forecast_bias}"
                        except Exception:
                            pass
                signal_emitted = True
                for extra_key in (
                    "entry_type",
                    "entry_price",
                    "entry_tolerance_pips",
                    "limit_expiry_seconds",
                    "hard_stop_pips",
                    "notes",
                ):
                    if extra_key in raw_signal:
                        signal[extra_key] = raw_signal[extra_key]
                scaled_conf = int(signal["confidence"] * health.confidence_scale)
                signal["confidence"] = max(0, min(100, scaled_conf))
                # Apply GPT mode/bias/pattern multipliers (gentle)
                conf_mult = _strategy_conf_multiplier(gpt, sname)
                if conf_mult != 1.0:
                    signal["confidence"] = max(
                        0, min(100, int(signal["confidence"] * conf_mult))
                    )
                # Tech overlays (Ichimoku/cluster/MACD-DMI/Stoch) applied uniformly across workers
                signal = _apply_tech_overlays(signal, fac_m1, fac_m5, fac_h1, fac_h4)

                # TP extension for trend-runner modes (keep short-T P workers as-is)
                try:
                    tp_val = float(signal.get("tp_pips") or 0.0)
                except Exception:
                    tp_val = 0.0
                if tp_val > 0 and not range_active:
                    adx_h1_val = 0.0
                    try:
                        adx_h1_val = float((fac_h1 or {}).get("adx") or 0.0)
                    except Exception:
                        adx_h1_val = 0.0
                    strong_trend = (adx_h4_val >= 28.0) or (adx_h1_val >= 28.0)
                    if strong_trend and signal.get("strategy") in {"M1Scalper", "ImpulseRetrace", "ImpulseBreak", "ImpulseMomentum"}:
                        base = tp_val
                        boosted = max(base * 1.2, base + 2.0)
                        atr_cap = None
                        try:
                            atr_cap = float((fac_m5 or {}).get("atr_pips") or 0.0) * 6.0
                        except Exception:
                            atr_cap = None
                        cap = 15.0
                        if atr_cap and atr_cap > 0:
                            cap = min(cap, atr_cap)
                        new_tp = round(min(boosted, cap), 2)
                        if abs(new_tp - base) >= 0.01:
                            logging.info(
                                "[TP_BOOST] strategy=%s tp=%.2f->%.2f adx_h4=%.1f adx_h1=%.1f cap=%.2f",
                                signal.get("strategy"),
                                base,
                                new_tp,
                                adx_h4_val,
                                adx_h1_val,
                                cap,
                            )
                            signal["tp_pips"] = new_tp

                # --- Dynamic guards: shock/position/stretch and role-based blocks ---
                action = signal.get("action")
                if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                    continue
                is_buy = action == "OPEN_LONG"

                def _ensure_notes(sig: dict) -> dict:
                    notes = sig.get("notes")
                    if not isinstance(notes, dict):
                        notes = {}
                    sig["notes"] = notes
                    return notes

                def _apply_wait(sig: dict, *, reason: str, price_hint: Optional[float]) -> None:
                    if price_hint is None or DISABLE_WAIT_GUARD:
                        return
                    wait_pips = max(0.12, min(max(atr_pips, 0.8) * 0.6, 1.8))
                    tol = max(
                        0.25,
                        float(sig.get("entry_tolerance_pips") or 0.0),
                        wait_pips * 0.6,
                    )
                    target = (
                        price_hint - wait_pips * PIP if is_buy else price_hint + wait_pips * PIP
                    )
                    sig["entry_type"] = "limit"
                    sig["entry_price"] = round(target, 3)
                    sig["entry_tolerance_pips"] = round(tol, 2)
                    notes = _ensure_notes(sig)
                    notes.setdefault("wait_guard", []).append(reason)

                try:
                    price_now = float(close_val) if close_val is not None else None
                except Exception:
                    price_now = None
                pos_pct = None
                try:
                    c_list = fac_m1.get("candles") or []
                    sample = c_list[-60:] if isinstance(c_list, list) else []
                    highs = [float(c.get("high", c.get("h"))) for c in sample if c.get("high") or c.get("h")]
                    lows = [float(c.get("low", c.get("l"))) for c in sample if c.get("low") or c.get("l")]
                    if highs and lows and price_now is not None:
                        rng = max(highs) - min(lows)
                        if rng > 1e-6:
                            pos_pct = max(0.0, min(1.0, (price_now - min(lows)) / rng))
                except Exception:
                    pos_pct = None
                stretch = None
                try:
                    ema20_h1 = float(fac_h1.get("ema20") or fac_h1.get("ma20") or 0.0) if fac_h1 else 0.0
                    atr_m5 = fac_m5.get("atr_pips") if fac_m5 else None
                    if atr_m5 is None and fac_m5:
                        atr_m5 = (fac_m5.get("atr") or 0.0) * 100.0
                    if price_now is not None and atr_m5:
                        stretch = (price_now - ema20_h1) / max(float(atr_m5), 1e-6)
                except Exception:
                    stretch = None

                # Shock handling: prefer waiting over chasing
                if shock_ctx.get("down") and is_buy:
                    _apply_wait(signal, reason="shock_down", price_hint=price_now)
                if shock_ctx.get("up") and not is_buy:
                    _apply_wait(signal, reason="shock_up", price_hint=price_now)

                # Position (bottom/top) handling
                if pos_pct is not None:
                    if pos_pct < 0.15 and not is_buy:
                        _apply_wait(signal, reason="bottom_short_wait", price_hint=price_now)
                    if pos_pct > 0.85 and is_buy:
                        _apply_wait(signal, reason="top_long_wait", price_hint=price_now)

                # Stretch guard (ema20 vs ATR)
                if stretch is not None:
                    if is_buy and stretch > 1.2:
                        _apply_wait(signal, reason="long_stretch_wait", price_hint=price_now)
                    if (not is_buy) and stretch < -1.2:
                        _apply_wait(signal, reason="short_stretch_wait", price_hint=price_now)

                # Strong trend: drop opposite direction entirely unless reversal is evident
                strong_trend_dir = 0
                strong_trend = False
                try:
                    if bias_h1 != 0 and bias_h1 == bias_h4 and max(adx_h1, adx_h4) >= 25.0:
                        strong_trend_dir = bias_h1
                        strong_trend = True
                except Exception:
                    strong_trend = False
                rsi_val_num = None
                try:
                    rsi_val_num = float(fac_m1.get("rsi"))
                except Exception:
                    rsi_val_num = None
                pocket_val = (signal.get("pocket") or "").lower()
                if strong_trend_dir and ((is_buy and strong_trend_dir < 0) or ((not is_buy) and strong_trend_dir > 0)):
                    allow_reversal = False
                    if rsi_val_num is not None and momentum is not None:
                        if is_buy:
                            allow_reversal = rsi_val_num >= 35.0 and momentum > 0.0
                        else:
                            allow_reversal = rsi_val_num <= 65.0 and momentum < 0.0
                    if not allow_reversal:
                        # スキャルプ/マイクロは逆行でもサイズを落として通す
                        if pocket_val in {"micro", "scalp", "scalp_fast"}:
                            conf_before = signal.get("confidence", 0)
                            signal["confidence"] = int(max(15, conf_before * 0.45))
                            logging.info(
                                "[DIR_GUARD_SOFT] strong trend opposite allowed with reduced confidence strategy=%s dir=%s trend_dir=%s conf=%s->%s",
                                signal.get("strategy"),
                                "long" if is_buy else "short",
                                strong_trend_dir,
                                conf_before,
                                signal.get("confidence"),
                            )
                        else:
                            signal["confidence"] = 0
                            logging.info(
                                "[DIR_GUARD] Opposite to strong trend blocked strategy=%s dir=%s trend_dir=%s adx_max=%.1f rsi=%s mom=%.4f",
                                signal.get("strategy"),
                                "long" if is_buy else "short",
                                strong_trend_dir,
                                max(adx_h1, adx_h4),
                                rsi_val_num,
                                momentum,
                            )

                # ImpulseRetrace: strong downtrend -> block longs unless reversal
                if (
                    signal.get("strategy") == "ImpulseRetrace"
                    and is_buy
                    and h4_dir < 0
                    and adx_h4_val >= 25.0
                ):
                    rsi_v = fac_m1.get("rsi")
                    momentum_v = momentum
                    allow_reversal = False
                    try:
                        allow_reversal = (rsi_v is not None and float(rsi_v) >= 35.0) and momentum_v is not None and float(momentum_v) > 0.0
                    except Exception:
                        allow_reversal = False
                    if not allow_reversal:
                        signal["confidence"] = 0
                        logging.info(
                            "[DIR_GUARD] ImpulseRetrace long blocked (h4_down adx=%.1f rsi=%s mom=%.4f)",
                            adx_h4_val,
                            rsi_v,
                            momentum,
                        )

                if signal.get("confidence", 0) <= 0:
                    logging.info("[SKIP] zero confidence after guards %s", signal.get("strategy"))
                    continue

                allow_micro, gate_reason, gate_ctx = _micro_chart_gate(
                    signal,
                    fac_m1,
                    story_snapshot,
                    open_positions_snapshot,
                )
                if not allow_micro:
                    logging.info(
                        "[CHART_GATE] skip %s pocket=%s action=%s reason=%s ctx=%s",
                        sname,
                        pocket,
                        signal["action"],
                        gate_reason,
                        gate_ctx,
                    )
                    log_metric(
                        "micro_chart_gate_block",
                        1.0,
                        tags={
                            "reason": gate_reason,
                            "strategy": sname,
                            "trend": str(gate_ctx.get("trend", "")),
                        },
                        ts=now,
                    )
                    continue
                if FORCE_SCALP_MODE:
                    logging.warning("[FORCE_SCALP] signal=%s", signal)
                if range_active:
                    profile = recent_profiles.get(signal["pocket"], {})
                    sample_size = int(profile.get("sample_size", 0) or 0)
                    use_profile = sample_size >= 5
                    avg_win = float(profile.get("avg_win_pips", 0.0) or 0.0) if use_profile else 0.0
                    avg_loss = float(profile.get("avg_loss_pips", 0.0) or 0.0) if use_profile else 0.0
                    atr_hint = max(atr_pips, 0.5)
                    if signal["pocket"] == "macro":
                        base_sl = avg_loss * 0.9 if avg_loss > 0.2 else atr_hint * 0.85
                        base_sl = max(1.0, min(base_sl, 2.2))
                        base_tp = avg_win * 0.95 if avg_win > 0.2 else atr_hint * 1.2
                        base_tp = max(base_sl * 1.15, min(base_tp, 2.6))
                        signal["confidence"] = int(signal["confidence"] * 0.6)
                    else:
                        base_sl = avg_loss * 0.85 if avg_loss > 0.15 else atr_hint * 0.65
                        base_sl = max(0.8, min(base_sl, 1.6))
                        base_tp = avg_win * 0.9 if avg_win > 0.15 else atr_hint * 0.95
                        base_tp = max(base_sl * 1.2, min(base_tp, 2.1))
                        signal["confidence"] = int(signal["confidence"] * 0.75)
                    if base_tp <= base_sl:
                        base_tp = base_sl * 1.2
                    signal["sl_pips"] = round(base_sl, 2)
                    signal["tp_pips"] = round(base_tp, 2)
                    signal["confidence"] = max(0, min(100, signal["confidence"]))
                    if use_profile:
                        logging.info(
                            "[RANGE] RR tuned via profile pocket=%s avg_win=%.2f avg_loss=%.2f samples=%d",
                            signal["pocket"],
                            avg_win,
                            avg_loss,
                            sample_size,
                        )
                if range_active and signal["strategy"] == "RangeFader":
                    sl_val = float(signal.get("sl_pips") or 0.0)
                    tp_val = float(signal.get("tp_pips") or 0.0)
                    rr_before = tp_val / sl_val if sl_val > 0 else 0.0
                    tp_adj = tp_val
                    if sl_val > 0 and tp_val > 0:
                        if rr_before > RANGE_FADER_MAX_RR:
                            tp_adj = sl_val * RANGE_FADER_MAX_RR
                        elif rr_before < RANGE_FADER_MIN_RR:
                            tp_adj = sl_val * RANGE_FADER_MIN_RR
                        if tp_adj != tp_val:
                            tp_val = round(tp_adj, 2)
                            signal["tp_pips"] = tp_val
                        rr_after = tp_val / sl_val if sl_val > 0 else 0.0
                        log_metric(
                            "range_fader_rr",
                            rr_after,
                            tags={"range_active": "true"},
                            ts=now,
                        )
                    prev_conf = signal["confidence"]
                    scaled_conf = int(prev_conf * RANGE_FADER_RANGE_CONF_SCALE)
                    signal["confidence"] = max(35, min(80, scaled_conf))
                    log_metric(
                        "range_fader_confidence",
                        float(signal["confidence"]),
                        tags={"range_active": "true"},
                        ts=now,
                    )
                    if prev_conf != signal["confidence"]:
                        rr_logged = (
                            (signal.get("tp_pips") or 0.0) / sl_val if sl_val > 0 else 0.0
                        )
                        logging.info(
                            "[RANGE] RangeFader confidence %d -> %d (SL=%.2f TP=%.2f rr=%.2f)",
                            prev_conf,
                            signal["confidence"],
                            sl_val,
                            signal.get("tp_pips"),
                            rr_logged,
                        )

                if strategy_conf_advisor and strategy_conf_advisor.enabled:
                    conf_context = {
                        "strategy": sname,
                        "pocket": cls.pocket,
                        "reg_macro": macro_regime,
                        "reg_micro": micro_regime,
                        "range_active": range_active,
                        "weight_macro": weight_macro,
                        "health_pf": health.profit_factor,
                        "health_win_rate": health.win_rate,
                        "health_dd": health.max_drawdown_pips,
                    }
                    try:
                        conf_hint = await strategy_conf_advisor.advise(sname, conf_context)
                    except Exception as exc:  # pragma: no cover
                        logging.debug("[STRAT_CONF] failed: %s", exc)
                        conf_hint = None
                    if conf_hint and conf_hint.confidence >= 0.35:
                        prev_conf = signal["confidence"]
                        signal["confidence"] = max(
                            0,
                            min(100, int(signal["confidence"] * conf_hint.scale)),
                        )
                        if prev_conf != signal["confidence"]:
                            log_metric(
                                "strategy_conf_scale",
                                float(conf_hint.scale),
                                tags={"strategy": sname},
                                ts=now,
                            )
                if signal["strategy"] == "PulseBreak":
                    rr = 0.0
                    sl_val = float(signal.get("sl_pips") or 0.0)
                    tp_val = float(signal.get("tp_pips") or 0.0)
                    if sl_val > 0:
                        rr = tp_val / sl_val
                    log_metric(
                        "pulse_break_confidence",
                        float(signal["confidence"]),
                        tags={"range_active": str(range_active)},
                        ts=now,
                    )
                    log_metric(
                        "pulse_break_rr",
                        float(rr),
                        tags={"range_active": str(range_active)},
                        ts=now,
                    )
                signal["health"] = {
                    "win_rate": health.win_rate,
                    "pf": health.profit_factor,
                    "confidence_scale": health.confidence_scale,
                    "drawdown": health.max_drawdown_pips,
                    "losing_streak": health.losing_streak,
                }

                if (
                    rr_advisor
                    and rr_advisor.enabled
                    and signal.get("sl_pips")
                    and signal.get("tp_pips")
                ):
                    try:
                        tp_before_rr = float(signal.get("tp_pips") or 0.0)
                    except Exception:
                        tp_before_rr = 0.0
                    try:
                        rr_context = {
                            "pocket": signal["pocket"],
                            "strategy": signal["strategy"],
                            "sl_pips": float(signal["sl_pips"] or 0.0),
                            "tp_pips": float(signal["tp_pips"] or 0.0),
                            "reg_macro": macro_regime,
                            "reg_micro": micro_regime,
                            "range_active": range_active,
                            "atr_m1": float(atr_pips),
                            "atr_h4": float(fac_h4.get("atr_pips", 0.0) or 0.0),
                            "atr_h1": float((fac_h1 or {}).get("atr_pips", 0.0) or 0.0),
                            "factors_h1": _factor_snapshot(fac_h1),
                            "factors_h4": _factor_snapshot(fac_h4),
                            "factors_d1": _factor_snapshot(fac_d1),
                        }
                        rr_hint = await rr_advisor.advise(rr_context)
                    except Exception as exc:  # pragma: no cover - defensive
                        logging.debug("[RR_ADVISOR] failed: %s", exc)
                        rr_hint = None
                    if rr_hint:
                        sl_val = float(signal["sl_pips"] or 0.0)
                        target_tp = round(max(sl_val * rr_hint.ratio, sl_val * rr_advisor.min_ratio), 2)
                        if target_tp > 0.0:
                            # Donchian55 の TP は戦略側の上限を優先して短く保つ
                            if (
                                (signal.get("strategy") == "Donchian55"
                                 or signal.get("profile") == "macro_breakout_donchian")
                                and tp_before_rr > 0.0
                                and target_tp > tp_before_rr
                            ):
                                logging.info(
                                    "[RR_ADVISOR] cap tp for Donchian55 %.2f->%.2f",
                                    target_tp,
                                    tp_before_rr,
                                )
                                target_tp = tp_before_rr
                            signal["tp_pips"] = target_tp
                            log_metric(
                                "rr_advisor_ratio",
                                float(rr_hint.ratio),
                                tags={
                                    "pocket": signal["pocket"],
                                    "strategy": signal["strategy"],
                                },
                                ts=now,
                            )
                guard_price = _safe_float(fac_m1.get("close") or fac_m1.get("mid"))
                if _market_guarded_skip(signal, guard_price, context="prefilter"):
                    continue
                evaluated_signals.append(signal)
                signals.append(signal)
                logging.info("[SIGNAL] %s -> %s", cls.name, signal)

            open_positions = pos_manager.get_open_positions()
            net_units = 0
            try:
                net_units = int(open_positions.get("__net__", {}).get("units", 0) or 0)
            except Exception:
                net_units = 0
            stage_snapshot: dict[str, dict[str, int]] = {}
            for pocket_name, position_info in open_positions.items():
                if pocket_name == "__net__":
                    continue
                stage_snapshot[pocket_name] = {
                    "long": stage_tracker.get_stage(pocket_name, "long"),
                    "short": stage_tracker.get_stage(pocket_name, "short"),
                }
            try:
                update_dynamic_protections(open_positions, fac_m1, fac_h4)
            except Exception as exc:
                logging.warning("[PROTECTION] update failed: %s", exc)
            partial_threshold_overrides = None
            if partial_advisor and partial_advisor.enabled:
                partial_context = {
                    "range_active": range_active,
                    "atr_pips": atr_pips,
                    "risk_appetite": param_snapshot.risk_appetite,
                    "profile_samples": {
                        pocket: profile.get("sample_size", 0)
                        for pocket, profile in recent_profiles.items()
                    },
                }
                try:
                    partial_hint = await partial_advisor.advise(partial_context)
                except Exception as exc:  # pragma: no cover
                    logging.debug("[PARTIAL_ADVISOR] failed: %s", exc)
                    partial_hint = None
                if partial_hint and partial_hint.confidence >= 0.35:
                    partial_threshold_overrides = partial_hint.thresholds

            try:
                partials = plan_partial_reductions(
                    open_positions,
                    fac_m1,
                    range_mode=range_active,
                    stage_state=stage_snapshot,
                    pocket_profiles=recent_profiles,
                    now=now,
                    threshold_overrides=partial_threshold_overrides,
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
                stage_snapshot = {}
                for pocket_name, position_info in open_positions.items():
                    if pocket_name == "__net__":
                        continue
                    stage_snapshot[pocket_name] = {
                        "long": stage_tracker.get_stage(pocket_name, "long"),
                        "short": stage_tracker.get_stage(pocket_name, "short"),
                    }
            net_units = int(open_positions.get("__net__", {}).get("units", 0))

            for pocket, info in open_positions.items():
                if pocket == "__net__":
                    continue
                for direction, key_units in (("long", "long_units"), ("short", "short_units")):
                    units_value = int(info.get(key_units, 0) or 0)
                    tracker_stage = stage_snapshot.get(pocket, {}).get(direction)
                    if tracker_stage is None:
                        tracker_stage = stage_tracker.get_stage(pocket, direction)
                        stage_snapshot.setdefault(pocket, {})[direction] = tracker_stage
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
                            stage_snapshot.setdefault(pocket, {})[direction] = 0
                            stage_empty_since.pop(key, None)
                    elif units_value != 0:
                        stage_empty_since.pop(key, None)

            advisor_hints = None
            if exit_advisor and exit_advisor.enabled:
                try:
                    advisor_hints = await exit_advisor.build_hints(
                        open_positions,
                        fac_m1=fac_m1,
                        fac_h4=fac_h4,
                        fac_h1=fac_h1,
                        fac_d1=fac_d1,
                        range_active=range_active,
                        now=now,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("[EXIT_ADVISOR] build_hints failed: %s", exc)
                    advisor_hints = None

            signals_for_exit = [
                sig for sig in evaluated_signals if sig.get("pocket") not in _EXIT_MAIN_DISABLED_POCKETS
            ]
            stage_state_for_exit = {
                pocket: state
                for pocket, state in (stage_snapshot or {}).items()
                if pocket not in _EXIT_MAIN_DISABLED_POCKETS
            }
            open_positions_for_exit = {
                pocket: info
                for pocket, info in open_positions.items()
                if pocket == "__net__" or pocket not in _EXIT_MAIN_DISABLED_POCKETS
            }

            exit_decisions = []
            if open_positions_for_exit:
                exit_decisions = exit_manager.plan_closures(
                    open_positions_for_exit,
                    signals_for_exit,
                    fac_m1,
                    fac_h4,
                    fac_h1=fac_h1,
                    fac_m5=fac_m5,
                    event_soon=event_soon,
                    range_mode=range_active,
                    stage_state=stage_state_for_exit,
                    pocket_profiles=recent_profiles,
                    now=now,
                    clamp_state=clamp_state,
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
                        # 動的クールダウン: 時間ガードやMAE系の退出は少し長めにする
                        if decision.reason and decision.reason.startswith("time_guard"):
                            cooldown_seconds = int(max(cooldown_seconds, cooldown_seconds * 1.5, 150))
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
                            # micro/スカルプは短めにして方向転換を許容、macroは従来通り長め
                            if pocket == "micro":
                                flip_cd = 30
                            elif pocket == "scalp":
                                flip_cd = 45
                            else:
                                flip_cd = min(240, max(60, cooldown_seconds // 2))
                            stage_tracker.set_cooldown(
                                pocket,
                                opposite,
                                reason="flip_guard",
                                seconds=flip_cd,
                                now=now,
                            )

            if not evaluated_signals:
                logging.info(
                    "[WAIT] No signals focus=%s pockets=%s range=%s soft=%s atr=%.2f vol5=%s mom=%.4f ranked=%s allow=%s",
                    focus_tag,
                    ",".join(sorted(focus_pockets)),
                    range_active,
                    range_soft_active,
                    atr_pips if atr_pips is not None else -1.0,
                    f"{vol_5m:.2f}" if vol_5m is not None else "n/a",
                    momentum,
                    ranked_strategies,
                    sorted(gpt_strategy_allowlist),
                )
                if evaluated_count == 0:
                    logging.info(
                        "[WAIT] No strategies evaluated focus=%s pockets=%s gpt_allow=%s",
                        focus_tag,
                        ",".join(sorted(focus_pockets)),
                        sorted(gpt_strategy_allowlist),
                    )
            else:
                logging.info(
                    "[STRAT_EVAL_END] evaluated=%d signals=%d focus=%s range=%s",
                    evaluated_count,
                    len(evaluated_signals),
                    focus_tag,
                    range_active,
                )
                if FORCE_SCALP_MODE:
                    logging.warning("[FORCE_SCALP] evaluated_signals empty")

            if FORCE_SCALP_MODE:
                logging.warning("[FORCE_SCALP] evaluated_signals_raw=%s", evaluated_signals)
                log_metric(
                    "force_scalp_signal_count",
                    float(len(evaluated_signals)),
                    tags={"stage": "pre_lot"},
                    ts=now,
                )
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
            margin_usage = None

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

            clamp_level = int(clamp_state.get("level", 0) or 0)
            scalp_conf_scale = float(clamp_state.get("scalp_conf_scale", 1.0) or 1.0)
            impulse_stop_until = clamp_state.get("impulse_stop_until")
            impulse_thin_active = bool(clamp_state.get("impulse_thin_active"))
            impulse_thin_scale = float(clamp_state.get("impulse_thin_scale", 1.0) or 1.0)
            bias_h1, bias_h4, adx_h1, adx_h4 = _dir_bias(fac_h1, fac_h4)
            high_vol_env = (atr_pips is not None and atr_pips > 2.5) or (vol_5m is not None and vol_5m > 1.5)
            low_vol_env = (atr_pips is not None and atr_pips < 1.2) and (vol_5m is not None and vol_5m < 0.7)
            try:
                mid_price = float(fac_m1.get("close") or fac_m1.get("mid") or 0.0)
            except Exception:
                mid_price = 0.0
            open_positions_snapshot = pos_manager.get_open_positions()
            # bot-only net units (manualポケットは除外)
            net_units = 0
            try:
                net_units = int(open_positions_snapshot.get("__net__", {}).get("units", 0) or 0)
            except Exception:
                net_units = 0
            try:
                bot_units = 0
                side_long_units = 0
                side_short_units = 0
                for pk, info in open_positions_snapshot.items():
                    if pk in {"__net__", "__meta__"}:
                        continue
                    try:
                        bot_units += int(info.get("units", 0) or 0)
                    except Exception:
                        continue
                    try:
                        lu = int(info.get("long_units", 0) or 0)
                    except Exception:
                        lu = 0
                    try:
                        su = int(info.get("short_units", 0) or 0)
                    except Exception:
                        su = 0
                    if lu == 0 and su == 0:
                        try:
                            units_raw = int(info.get("units", 0) or 0)
                            if units_raw > 0:
                                lu = units_raw
                            elif units_raw < 0:
                                su = abs(units_raw)
                        except Exception:
                            pass
                    side_long_units += max(0, lu)
                    side_short_units += max(0, su)
                net_units = bot_units
            except Exception:
                pass
            # Account/margin exposure guards（OANDAスナップショットに加え、自前のネット額計算を優先）
            margin_block = False
            margin_warn = False
            if account_snapshot:
                try:
                    m_avail = float(account_snapshot.margin_available or 0.0)
                    m_used = float(account_snapshot.margin_used or 0.0)
                    total_margin = m_avail + m_used
                    if total_margin > 0:
                        margin_usage = m_used / total_margin
                except Exception:
                    margin_usage = None
            # 自前計算: abs(net_units) * mid_price * margin_rate / equity でネット証拠金率を推定
            try:
                if margin_rate and account_equity > 0 and mid_price > 0:
                    usage_est = abs(float(net_units)) * mid_price * margin_rate / account_equity
                    margin_usage = usage_est
            except Exception:
                pass
            if margin_usage is not None:
                # allow利用目標: 82〜88%、警告は 93% 以上。ブロックは方向別判定に任せ、ここでは止めない。
                if margin_usage >= 0.93:
                    margin_warn = True
                    logging.info("[RISK] margin usage elevated %.1f%% (monitoring, no block)", margin_usage * 100)
            exposure_pct = 0.0
            side_exposure_long = 0.0
            side_exposure_short = 0.0
            margin_usage_long = None
            margin_usage_short = None
            if mid_price > 0 and account_equity > 0:
                exposure_pct = abs(net_units) * mid_price / account_equity
                side_exposure_long = side_long_units * mid_price / account_equity
                side_exposure_short = side_short_units * mid_price / account_equity
                if margin_rate:
                    margin_usage_long = side_long_units * mid_price * margin_rate / account_equity
                    margin_usage_short = side_short_units * mid_price * margin_rate / account_equity
            # 露出capは証拠金率に整合させる（例: 4% = 25x → ハード24x、ソフト22x）
            exposure_hard_cap = 0.90
            exposure_soft_cap = 0.82
            try:
                if margin_rate and margin_rate > 0:
                    target_hard_usage = 0.96  # margin_usageブロック閾値に合わせる
                    target_soft_usage = 0.88
                    exposure_hard_cap = max(2.0, min(30.0, target_hard_usage / margin_rate))
                    # ソフトキャップは実質ハードと同等にして方向別マージンの許容量を確保
                    exposure_soft_cap = exposure_hard_cap
            except Exception:
                pass

            filtered_signals: list[dict] = []
            for sig in evaluated_signals:
                reduce_cap_units = 0
                action_dir = 0
                try:
                    if sig.get("action") == "OPEN_LONG":
                        action_dir = 1
                    elif sig.get("action") == "OPEN_SHORT":
                        action_dir = -1
                except Exception:
                    action_dir = 0
                net_reducing = bool(
                    net_units
                    and action_dir
                    and ((net_units > 0 and action_dir < 0) or (net_units < 0 and action_dir > 0))
                )
                side_exposure = 0.0
                if action_dir > 0:
                    side_exposure = side_exposure_long
                elif action_dir < 0:
                    side_exposure = side_exposure_short
                if sig.get("pocket") == "scalp" and clamp_level >= 3:
                    logging.info(
                        "[CLAMP] skip scalp entry level=3 strategy=%s action=%s",
                        sig.get("strategy"),
                        sig.get("action"),
                    )
                    continue
                if margin_block and not sig.get("reduce_only") and not net_reducing:
                    # 方向別に判定し、反対側は通す（別腹扱い）
                    dir_block = False
                    dir_cap = 0.96
                    if action_dir > 0 and margin_usage_long is not None and margin_usage_long >= dir_cap:
                        dir_block = True
                    if action_dir < 0 and margin_usage_short is not None and margin_usage_short >= dir_cap:
                        dir_block = True
                    if dir_block:
                        logging.info(
                            "[RISK] skip new entry (dir margin block) strategy=%s dir=%s usage=%.1f%%",
                            sig.get("strategy"),
                            "long" if action_dir > 0 else "short",
                            (margin_usage_long if action_dir > 0 else margin_usage_short) * 100.0,
                        )
                        continue
                # margin_usage が十分余裕 (<MAX_MARGIN_USAGE) の場合は露出capによるブロックを緩和
                def _can_apply_exposure_cap() -> bool:
                    if margin_usage is None:
                        return True
                    return margin_usage >= MAX_MARGIN_USAGE

                if (
                    action_dir != 0
                    and side_exposure >= exposure_hard_cap
                    and not sig.get("reduce_only")
                    and _can_apply_exposure_cap()
                    and not net_reducing
                ):
                    logging.info(
                        "[RISK] skip new entry (side exposure %.2f >= cap %.2f) strategy=%s dir=%s",
                        side_exposure,
                        exposure_hard_cap,
                        sig.get("strategy"),
                        "long" if action_dir > 0 else "short",
                    )
                    continue
                if (
                    not sig.get("reduce_only")
                    and action_dir != 0
                    and side_exposure >= exposure_soft_cap
                    and _can_apply_exposure_cap()
                    and not net_reducing
                ):
                    logging.info(
                        "[RISK] skip same-direction entry (side exposure %.2f >= soft cap %.2f) strategy=%s dir=%s",
                        side_exposure,
                        exposure_soft_cap,
                        sig.get("strategy"),
                        "long" if action_dir > 0 else "short",
                    )
                    continue
                # L3 reduce-only: allow only net-reducing orders
                if clamp_level >= 3 and action_dir != 0:
                    if net_units == 0:
                        logging.info("[CLAMP] L3 reduce_only skip (flat net).")
                        continue
                    net_dir = 1 if net_units > 0 else -1
                    if action_dir == net_dir:
                        logging.info(
                            "[CLAMP] L3 reduce_only skip same_dir net=%d strategy=%s",
                            net_units,
                            sig.get("strategy"),
                        )
                        continue
                    reduce_cap_units = max(
                        _CLAMP_L3_MIN_REDUCE_UNITS, int(abs(net_units) * _CLAMP_L3_REDUCE_FRACTION)
                    )
                    sig["reduce_only"] = True
                    sig["reduce_cap_units"] = reduce_cap_units
                if action_dir != 0 and (bias_h1 != 0 or bias_h4 != 0):
                    scale = 1.0
                    align = False
                    oppose = False
                    # determine trend strength from H1/H4
                    adx_max = max(adx_h1, adx_h4)
                    reverse_scale = _DIR_BIAS_SCALE_OPPOSE
                    align_scale = _DIR_BIAS_SCALE_ALIGN
                    if adx_max >= 28.0:
                        reverse_scale = min(reverse_scale, 0.25)
                        align_scale = max(align_scale, 1.08)
                    elif adx_max <= 18.0:
                        reverse_scale = max(reverse_scale, 0.6)
                        align_scale = 1.0
                    if bias_h1 != 0 and action_dir != bias_h1:
                        oppose = True
                    if bias_h4 != 0 and action_dir != bias_h4:
                        oppose = True
                    if bias_h1 != 0 and action_dir == bias_h1:
                        align = True
                    if bias_h4 != 0 and action_dir == bias_h4:
                        align = True
                    if oppose and align:
                        # disagreement between H1/H4: weaken bias
                        reverse_scale = min(0.7, reverse_scale * 1.5)
                        align_scale = 1.0
                    if oppose:
                        scale *= reverse_scale
                    elif align:
                        scale *= align_scale
                    if abs(scale - 1.0) > 1e-3:
                        prev_conf = int(sig.get("confidence", 0) or 0)
                        new_conf = max(0, int(prev_conf * scale))
                        sig["confidence"] = new_conf
                        logging.info(
                            "[DIR_BIAS] strategy=%s action=%s conf=%d->%d h1=%d h4=%d",
                            sig.get("strategy"),
                            sig.get("action"),
                            prev_conf,
                            new_conf,
                            bias_h1,
                            bias_h4,
                        )
                # Strategy-specific directional shaping
                strategy_name = str(sig.get("strategy") or "")
                if action_dir != 0 and adx_h1 is not None and adx_h4 is not None:
                    adx_max = max(adx_h1, adx_h4)
                    aligned_h4 = bias_h4 != 0 and bias_h4 == action_dir
                    opposed_h4 = bias_h4 != 0 and bias_h4 != action_dir
                    aligned_h1 = bias_h1 != 0 and bias_h1 == action_dir
                    opposed_h1 = bias_h1 != 0 and bias_h1 != action_dir
                    trend_dir = bias_h4 or bias_h1
                    try:
                        close_val = float(price or 0.0)
                    except Exception:
                        close_val = None
                    try:
                        ema20_h1 = float(fac_h1.get("ema20") or fac_h1.get("ma20") or 0.0)
                    except Exception:
                        ema20_h1 = 0.0
                    try:
                        atr_m5_val = float(fac_m5.get("atr_pips") or (fac_m5.get("atr") or 0.0) * 100.0)
                    except Exception:
                        atr_m5_val = 0.0
                    dist_norm = None
                    if close_val is not None and atr_m5_val > 0:
                        dist_norm = abs(close_val - ema20_h1) / (atr_m5_val * PIP)
                    # Trend-following (macro/core)
                    if strategy_name == "TrendMA":
                        adj = 1.0
                        if opposed_h4:
                            adj *= 0.2
                        elif aligned_h4:
                            adj *= 1.1
                        if opposed_h1 and adx_max >= 22.0:
                            adj *= 0.85
                        if abs(adj - 1.0) > 1e-3:
                            prev_conf = int(sig.get("confidence", 0) or 0)
                            sig["confidence"] = max(0, int(prev_conf * adj))
                            logging.info(
                                "[DIR_STRAT] TrendMA conf=%d->%d h1=%d h4=%d adx_max=%.1f",
                                prev_conf,
                                sig["confidence"],
                                bias_h1,
                                bias_h4,
                                adx_max,
                            )
                    # Mean reversion / range: downscale when trend is strong
                    if strategy_name in {"RangeFader", "BB_RSI"}:
                        if adx_max >= 25.0 and (aligned_h4 or aligned_h1):
                            prev_conf = int(sig.get("confidence", 0) or 0)
                            new_conf = max(0, int(prev_conf * 0.45))
                            if new_conf != prev_conf:
                                sig["confidence"] = new_conf
                                logging.info(
                                    "[DIR_STRAT] Range downscale conf=%d->%d h1=%d h4=%d adx_max=%.1f",
                                    prev_conf,
                                    new_conf,
                                    bias_h1,
                                    bias_h4,
                                    adx_max,
                                )
                    # Impulse系/Scalper: trend強なら順方向強め、逆は薄め
                    if strategy_name in {"ImpulseRe", "ImpulseRetrace", "M1Scalper"}:
                        adj = 1.0
                        # 強トレンドで逆方向のImpulse系はほぼ封印
                        if (
                            strategy_name in {"ImpulseRe", "ImpulseRetrace"}
                            and trend_dir != 0
                            and action_dir != trend_dir
                            and adx_max >= 25.0
                        ):
                            prev_conf = int(sig.get("confidence", 0) or 0)
                            if prev_conf > 0:
                                sig["confidence"] = 0
                                logging.info(
                                    "[DIR_STRAT] %s oppose strong trend -> conf=%d->0 h1=%d h4=%d adx=%.1f",
                                    strategy_name,
                                    prev_conf,
                                    bias_h1,
                                    bias_h4,
                                    adx_max,
                                )
                            continue
                        # 強トレンドでM1Scalper逆方向も大きく抑制
                        if (
                            strategy_name == "M1Scalper"
                            and trend_dir != 0
                            and action_dir != trend_dir
                            and adx_max >= 25.0
                        ):
                            prev_conf = int(sig.get("confidence", 0) or 0)
                            new_conf = max(0, int(prev_conf * 0.25))
                            sig["confidence"] = new_conf
                            logging.info(
                                "[DIR_STRAT] M1Scalper oppose strong trend conf=%d->%d h1=%d h4=%d adx=%.1f",
                                prev_conf,
                                new_conf,
                                bias_h1,
                                bias_h4,
                                adx_max,
                            )
                        # use normalized distance to avoid chasing stretched moves
                        if strategy_name in {"ImpulseRe", "ImpulseRetrace"} and dist_norm is not None:
                            if trend_dir != 0 and action_dir == trend_dir:
                                if 0.2 <= dist_norm <= 0.9:
                                    adj *= 1.12
                                elif dist_norm >= 1.6:
                                    adj *= 0.7
                            elif trend_dir != 0 and action_dir != trend_dir:
                                if dist_norm >= 1.0:
                                    adj *= 0.3
                                else:
                                    adj *= 0.5
                        if strategy_name == "M1Scalper" and dist_norm is not None:
                            if trend_dir != 0 and action_dir == trend_dir:
                                if dist_norm >= 1.5:
                                    adj *= 0.6
                                elif dist_norm <= 0.9:
                                    adj *= 1.05
                            elif trend_dir != 0 and action_dir != trend_dir:
                                if adx_max >= 25.0:
                                    adj *= 0.5
                                elif dist_norm <= 1.0 and adx_max <= 18.0:
                                    adj *= 0.9
                        if adx_max >= 25.0:
                            if aligned_h4 or aligned_h1:
                                adj *= 1.1
                            if opposed_h4 or opposed_h1:
                                adj *= 0.4
                        elif adx_max <= 18.0 and opposed_h4 and opposed_h1:
                            adj *= 0.7
                        if abs(adj - 1.0) > 1e-3:
                            prev_conf = int(sig.get("confidence", 0) or 0)
                            sig["confidence"] = max(0, int(prev_conf * adj))
                            logging.info(
                                "[DIR_STRAT] %s conf=%d->%d h1=%d h4=%d adx_max=%.1f",
                                strategy_name,
                                prev_conf,
                                sig["confidence"],
                                bias_h1,
                                bias_h4,
                                adx_max,
                            )
                macro_hedge_enabled = os.getenv("MACRO_HEDGE_ENABLE", "0").strip().lower() not in {
                    "",
                    "0",
                    "false",
                    "off",
                }
                # Macro hedging during Orange/Red: only net-reducing orders with cap
                if (
                    macro_hedge_enabled
                    and clamp_level >= 2
                    and sig.get("pocket") == "macro"
                    and action_dir != 0
                    and net_units != 0
                ):
                    net_dir = 1 if net_units > 0 else -1
                    if action_dir == net_dir:
                        logging.info(
                            "[MACRO_HEDGE] Skip same-dir macro (clamp>=2) net=%d strategy=%s",
                            net_units,
                            sig.get("strategy"),
                        )
                        continue
                    reduce_cap_units = max(
                        reduce_cap_units,
                        int(abs(net_units) * 0.35),
                        _CLAMP_L3_MIN_REDUCE_UNITS,
                    )
                    sig["reduce_only"] = True
                    sig["reduce_cap_units"] = reduce_cap_units
                # Macroヘッジを常時薄く許可（net逆方向のみ、net超過はしない）
                if (
                    macro_hedge_enabled
                    and sig.get("pocket") == "macro"
                    and action_dir != 0
                    and net_units != 0
                ):
                    net_dir = 1 if net_units > 0 else -1
                    if action_dir != net_dir:
                        hedge_frac = 0.25
                        if clamp_level >= 2:
                            hedge_frac = 0.5
                        reduce_cap_units = max(
                            reduce_cap_units,
                            int(abs(net_units) * hedge_frac),
                            _CLAMP_L3_MIN_REDUCE_UNITS,
                        )
                        sig["reduce_only"] = True
                        sig["reduce_cap_units"] = reduce_cap_units
                        logging.info(
                            "[MACRO_HEDGE] allow reduce-only macro net=%d action_dir=%d cap=%d frac=%.2f",
                            net_units,
                            action_dir,
                            reduce_cap_units,
                            hedge_frac,
                        )
                if sig.get("pocket") == "scalp" and scalp_conf_scale < 0.999:
                    prev_conf = int(sig.get("confidence", 0) or 0)
                    new_conf = max(0, int(prev_conf * scalp_conf_scale))
                    if new_conf != prev_conf:
                        logging.info(
                            "[CLAMP] scalp confidence scale=%.2f strategy=%s action=%s %d->%d level=%s",
                            scalp_conf_scale,
                            sig.get("strategy"),
                            sig.get("action"),
                            prev_conf,
                            new_conf,
                            clamp_level,
                        )
                        sig["confidence"] = new_conf

                if sig.get("strategy") == "ImpulseRe" and sig.get("action") in {"OPEN_LONG", "OPEN_SHORT"}:
                    stop_active = False
                    remain = None
                    rebound_ok = True
                    is_buy = sig.get("action") == "OPEN_LONG"
                    if impulse_stop_until:
                        try:
                            stop_active = now.timestamp() < impulse_stop_until.timestamp()
                            remain = int(max(1, impulse_stop_until.timestamp() - now.timestamp()))
                        except Exception:
                            stop_active = False
                    rsi_val = fac_m1.get("rsi")
                    if stop_active is False:
                        # 停止明けの市況チェック
                        if scalp_buffer is None or scalp_buffer <= 0.1:
                            rebound_ok = False
                        if rsi_val is None:
                            rebound_ok = False
                        if is_buy:
                            if rsi_val is None or rsi_val <= 45.0:
                                rebound_ok = False
                            if momentum < 0:
                                rebound_ok = False
                        else:
                            if rsi_val is None or rsi_val >= 55.0:
                                rebound_ok = False
                            if momentum > 0:
                                rebound_ok = False
                        if not rebound_ok:
                            stop_active = True
                            remain = remain or 30
                    if stop_active:
                        logging.info(
                            "[CLAMP] skip ImpulseRe %s during cooldown remain=%ss level=%s rebound_ok=%s",
                            "BUY" if is_buy else "SELL",
                            remain,
                            clamp_level,
                            rebound_ok,
                        )
                        continue
                    prev_conf = int(sig.get("confidence", 0) or 0)
                    rsi_val = fac_m1.get("rsi")
                    allow_rebound = True
                    rsi_gate = 33.0
                    if high_vol_env and momentum < 0:
                        rsi_gate = 38.0
                    elif low_vol_env:
                        rsi_gate = 33.0
                    if rsi_val is not None and momentum < 0 and rsi_val < 40:
                        rsi_rising = (
                            last_rsi_m1 is not None and rsi_val > last_rsi_m1 and rsi_val >= rsi_gate
                        )
                        price_rising = False
                        if close_px_value is not None and last_close_m1 is not None:
                            try:
                                price_rising = close_px_value > last_close_m1
                            except Exception:
                                price_rising = False
                        if not (rsi_rising or price_rising):
                            allow_rebound = False
                            damped = max(0, int(prev_conf * 0.35))
                            sig["confidence"] = damped
                            logging.info(
                                "[IMPULSE_FILTER] damped ImpulseRe BUY conf=%d->%d rsi=%.2f mom=%.4f rising=%s price_up=%s",
                                prev_conf,
                                damped,
                                rsi_val,
                                momentum,
                                rsi_rising,
                                price_rising,
                            )
                    if not allow_rebound and int(sig.get("confidence", 0) or 0) <= 0:
                        continue
                    if impulse_thin_active:
                        if scalp_buffer is None or scalp_buffer > 0.1:
                            prev_conf = int(sig.get("confidence", 0) or 0)
                            thin_scale = impulse_thin_scale
                            if high_vol_env:
                                thin_scale = max(thin_scale, 0.3)
                            scaled_conf = max(0, int(prev_conf * thin_scale))
                            if scaled_conf != prev_conf:
                                logging.info(
                                    "[CLAMP] ImpulseRe BUY thin scale=%.2f conf=%d->%d level=%s",
                                    thin_scale,
                                    prev_conf,
                                    scaled_conf,
                                    clamp_level,
                                )
                                sig["confidence"] = scaled_conf
                        else:
                            logging.info(
                                "[CLAMP] skip ImpulseRe BUY thin stage buffer=%.3f level=%s",
                                scalp_buffer,
                                clamp_level,
                            )
                            continue
                # 強トレンドの逆張りは反転確認が無ければ大きく減衰（全戦略共通の簡易ガード）
                if action_dir != 0 and bias_h4 != 0 and action_dir != bias_h4 and adx_h4 is not None and adx_h4 >= 25.0:
                    try:
                        rsi_m1_val = float(fac_m1.get("rsi") or 0.0)
                    except Exception:
                        rsi_m1_val = 0.0
                    try:
                        mom_val = float(momentum or 0.0)
                    except Exception:
                        mom_val = 0.0
                    try:
                        close_val = float(fac_m1.get("close") or 0.0)
                        ema20_m1_val = float(fac_m1.get("ema20") or fac_m1.get("ma20") or 0.0)
                    except Exception:
                        close_val = 0.0
                        ema20_m1_val = 0.0
                    allow_reversal = False
                    if bias_h4 < 0 and action_dir > 0:
                        # buy許可: RSI>52, momentum>=0, 直近close>EMA20
                        if rsi_m1_val >= 52.0 and mom_val >= 0 and close_val > ema20_m1_val:
                            allow_reversal = True
                    if bias_h4 > 0 and action_dir < 0:
                        # sell許可: RSI<48, momentum<=0, 直近close<EMA20
                        if rsi_m1_val <= 48.0 and mom_val <= 0 and close_val < ema20_m1_val:
                            allow_reversal = True
                    if not allow_reversal:
                        prev_conf = int(sig.get("confidence", 0) or 0)
                        damped = max(0, int(prev_conf * 0.1))
                        sig["confidence"] = damped
                        logging.info(
                            "[REVERSAL_GUARD] strategy=%s dir=%s trend_h4=%s adx=%.1f rsi=%.1f mom=%.4f ema20=%.3f close=%.3f conf=%d->%d",
                            sig.get("strategy"),
                            "BUY" if action_dir > 0 else "SELL",
                            "up" if bias_h4 > 0 else "down",
                            adx_h4,
                            rsi_m1_val,
                            mom_val,
                            ema20_m1_val,
                            close_val,
                            prev_conf,
                            damped,
                        )
                        if damped <= 0:
                            continue
                filtered_signals.append(sig)
            # Apply dynamic allocation (score-driven confidence trim) if available
            alloc_data = load_dynamic_alloc()
            evaluated_signals, pocket_caps, target_use = apply_dynamic_alloc(filtered_signals, alloc_data)
            # デバッグ: 関所手前のシグナル件数とタグ
            try:
                if not evaluated_signals:
                    logging.info(
                        "[SIGNAL_DEBUG] evaluated=0 filtered=%d raw=%d alloc_data=%s",
                        len(filtered_signals),
                        len(signals),
                        "yes" if alloc_data else "no",
                    )
                else:
                    logging.info(
                        "[SIGNAL_DEBUG] evaluated=%d pockets=%s tags=%s",
                        len(evaluated_signals),
                        ",".join(sorted({s.get('pocket') or 'unknown' for s in evaluated_signals})),
                        ",".join(sorted({s.get('strategy') or s.get('strategy_tag') or s.get('tag') or 'unknown' for s in evaluated_signals})),
                    )
            except Exception:
                pass
            # 同一サイクルで全体からconfidence上位のみを通す（最大3本）。
            # fast_scalp 偏重を避けるため、戦略ごとのブースト/ペナルティを適用し、fast_scalpは原則1本。
            max_signals = 3
            conf_floor = 0  # 両建て/ヘッジ運用ではconfidence下限でロングを落とさない
            if margin_usage is not None and margin_usage >= 0.88:
                max_signals = 2
            if margin_usage is not None and margin_usage >= 0.92:
                max_signals = 1
            if evaluated_signals:
                now_ts = time.time()
                candidates = []
                for s in evaluated_signals:
                    action = (s.get("action") or "").upper()
                    if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                        continue
                    raw_conf = int(s.get("confidence", 0) or 0)
                    tag = s.get("strategy") or s.get("strategy_tag") or s.get("tag") or ""
                    pocket = s.get("pocket") or ""
                    adj = raw_conf
                    # 戦略別バイアス: fast_scalpを強めに抑え、macro/microを押し上げる
                    if tag == "fast_scalp":
                        adj -= 14
                    elif pocket == "scalp":
                        adj -= 8
                    elif s.get("pocket") == "macro":
                        adj += 14
                    elif s.get("pocket") == "micro":
                        adj += 10
                    s["conf_adj"] = float(adj)
                    candidates.append(s)

                if SIGNAL_DIVERSITY_ENABLED:
                    for s in candidates:
                        key = _signal_strategy_key(s)
                        bonus = _signal_diversity_bonus(key, now_ts)
                        if bonus > 0.0:
                            s["conf_adj"] = float(s.get("conf_adj", 0.0)) + bonus
                            s["diversity_bonus"] = round(bonus, 2)

                if SIGNAL_DIVERSITY_DEDUPE:
                    deduped: Dict[str, dict] = {}
                    for s in candidates:
                        key = _signal_strategy_key(s)
                        cur = deduped.get(key)
                        if cur is None or float(s.get("conf_adj", 0.0)) > float(cur.get("conf_adj", 0.0)):
                            deduped[key] = s
                    if len(deduped) != len(candidates):
                        try:
                            logging.info(
                                "[SIGNAL_DEDUPE] before=%d after=%d",
                                len(candidates),
                                len(deduped),
                            )
                        except Exception:
                            pass
                    candidates = list(deduped.values())

                fast_candidates = []
                scalp_candidates = []
                macro_micro_candidates = []
                for s in candidates:
                    tag = s.get("strategy") or s.get("strategy_tag") or s.get("tag") or ""
                    pocket = s.get("pocket") or ""
                    if tag == "fast_scalp":
                        fast_candidates.append(s)
                    elif pocket == "scalp":
                        scalp_candidates.append(s)
                    else:
                        macro_micro_candidates.append(s)
                macro_micro_candidates.sort(
                    key=lambda s: int(s.get("conf_adj", s.get("confidence", 0)) or 0),
                    reverse=True,
                )
                fast_candidates.sort(
                    key=lambda s: int(s.get("conf_adj", s.get("confidence", 0)) or 0),
                    reverse=True,
                )
                scalp_candidates.sort(
                    key=lambda s: int(s.get("conf_adj", s.get("confidence", 0)) or 0),
                    reverse=True,
                )

                fast_limit = 1
                scalp_limit = 1
                selected: list[dict] = []
                # まず macro/micro を優先的に詰める
                for sig in macro_micro_candidates:
                    if len(selected) >= max_signals:
                        break
                    selected.append(sig)
                # 次に通常のscalpを1本まで
                scalp_added = 0
                for sig in scalp_candidates:
                    if len(selected) >= max_signals or scalp_added >= scalp_limit:
                        break
                    selected.append(sig)
                    scalp_added += 1
                # 残枠に fast_scalp を1本だけ許容
                fast_added = 0
                for sig in fast_candidates:
                    if len(selected) >= max_signals or fast_added >= fast_limit:
                        break
                    selected.append(sig)
                    fast_added += 1

                if len(selected) != len(candidates):
                    try:
                        logging.info(
                            "[SIGNAL_SELECT] picked=%d out_of=%d margin=%.2f conf_floor=%s strategies=%s fast_used=%d",
                            len(selected),
                            len(candidates),
                            margin_usage if margin_usage is not None else -1.0,
                            conf_floor if conf_floor > 0 else "none",
                            ",".join(sorted({s.get('strategy') or s.get('strategy_tag') or 'unknown' for s in selected})),
                            fast_added,
                        )
                    except Exception:
                        pass
                try:
                    logging.info(
                        "[SIGNAL_SELECT] selected=%d fast=%d confs=%s tags=%s",
                        len(selected),
                        sum(1 for sig in selected if (sig.get('strategy') or sig.get('strategy_tag') or sig.get('tag')) == 'fast_scalp'),
                        ",".join(str(sig.get("conf_adj", sig.get("confidence"))) for sig in selected),
                        ",".join(sig.get("strategy") or sig.get("strategy_tag") or sig.get("tag") or "unknown" for sig in selected),
                    )
                except Exception:
                    pass
                for sig in selected:
                    try:
                        log_metric(
                            "signal_confidence",
                            float(sig.get("confidence", 0) or 0),
                            tags={
                                "pocket": sig.get("pocket") or "unknown",
                                "strategy": sig.get("strategy") or sig.get("strategy_tag") or sig.get("tag") or "unknown",
                            },
                            ts=now,
                        )
                    except Exception:
                        pass
                if SIGNAL_DIVERSITY_ENABLED:
                    for sig in selected:
                        _SIGNAL_DIVERSITY_LAST_TS[_signal_strategy_key(sig)] = now_ts
                evaluated_signals = selected

            risk_override = _dynamic_risk_pct(
                evaluated_signals,
                range_active,
                weight_macro,
                context=param_snapshot,
                gpt_bias=gpt,
            )
            if alloc_data:
                try:
                    target_scale = max(0.8, min(1.4, target_use / 0.88))
                    risk_override = min(_MAX_RISK_PCT, risk_override * target_scale)
                except Exception:
                    pass
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

            if fast_scalp_state:
                fast_scalp_state.update_from_main(
                    account_equity=account_equity,
                    margin_available=float(margin_available or 0.0),
                    margin_rate=float(margin_rate or 0.0),
                    weight_scalp=weight_scalp,
                    focus_tag=focus_tag,
                    risk_pct_override=risk_override,
                    range_active=range_active,
                    m1_rsi=_safe_float(fac_m1.get("rsi")),
                    m1_rsi_age_sec=None,
                )

            base_price_val = fac_m1.get("close")
            try:
                mid_price = float(base_price_val or 0.0)
            except (TypeError, ValueError):
                mid_price = 0.0
            if tick_bid is not None and tick_ask is not None:
                mid_price = round((tick_bid + tick_ask) / 2, 3)

            def _lot_pattern_multiplier(fac: dict, story: object | None) -> float:
                """
                Composite multiplier fromテクニカル/パターン。
                0.75〜1.25の範囲でlotを調整する（市況が良ければ増、悪ければ減）。
                """
                mult = 1.0
                try:
                    adx = float(fac.get("adx") or 0.0)
                    rsi = float(fac.get("rsi") or 50.0)
                    bbw = float(fac.get("bbw") or 0.0)
                    vol5 = float(fac.get("vol_5m") or 0.0)
                    macd_hist = float(fac.get("macd_hist") or 0.0)
                    stoch = float(fac.get("stoch_rsi") or 0.0)
                    plus_di = float(fac.get("plus_di") or 0.0)
                    minus_di = float(fac.get("minus_di") or 0.0)
                    kc_width = float(fac.get("kc_width") or 0.0)
                    don_width = float(fac.get("donchian_width") or 0.0)
                    chaikin_vol = float(fac.get("chaikin_vol") or 0.0)
                    vwap_gap = float(fac.get("vwap_gap") or 0.0)
                except Exception:
                    adx, rsi, bbw, vol5 = (0.0, 50.0, 0.0, 0.0)
                    macd_hist = stoch = plus_di = minus_di = kc_width = don_width = chaikin_vol = vwap_gap = 0.0
                # トレンド強/適度なボラなら増額、低ボラレンジなら減額
                if adx >= 25:
                    mult += 0.05
                if bbw <= 0.0015 and adx <= 14:
                    mult -= 0.08
                if vol5 >= 1.2:
                    mult += 0.04
                if vol5 <= 0.5:
                    mult -= 0.05
                # RSIが極端なら控えめ
                if rsi <= 25 or rsi >= 75:
                    mult -= 0.03
                # MACDヒスト/DMI順行なら加点、逆行なら減点
                dmi_diff = plus_di - minus_di
                if macd_hist > 0:
                    mult += 0.02
                elif macd_hist < 0:
                    mult -= 0.02
                if dmi_diff > 5:
                    mult += 0.02
                elif dmi_diff < -5:
                    mult -= 0.02
                # StochRSI極端で減額
                if 0.8 <= stoch <= 1.2:
                    mult -= 0.015
                elif 0.0 <= stoch <= 0.2:
                    mult -= 0.015
                # ボラ幅系
                if kc_width > 0.012 or don_width > 0.012 or chaikin_vol > 0.15:
                    mult -= 0.02  # 過剰ボラは控えめ
                if kc_width < 0.006 and don_width < 0.006:
                    mult -= 0.01  # 低ボラレンジも控えめ
                # VWAP乖離が大きければ逆張り余地で少し加点
                if abs(vwap_gap) >= 5.0:
                    mult += 0.01
                patterns = {}
                if story and hasattr(story, "pattern_summary"):
                    try:
                        patterns = dict(getattr(story, "pattern_summary") or {})
                    except Exception:
                        patterns = {}
                n_wave = patterns.get("n_wave") if isinstance(patterns, dict) else None
                candle = patterns.get("candlestick") if isinstance(patterns, dict) else None
                try:
                    if n_wave and float(n_wave.get("confidence", 0.0) or 0.0) >= 0.6:
                        mult += 0.06
                except Exception:
                    pass
                try:
                    if candle and float(candle.get("confidence", 0.0) or 0.0) >= 0.6:
                        mult += 0.03
                except Exception:
                    pass
                return max(0.75, min(1.25, mult))

            long_units = 0.0
            short_units = 0.0
            try:
                long_units, short_units = get_position_summary("USD_JPY", timeout=3.0)
            except Exception:
                long_units, short_units = 0.0, 0.0

            lot_total = allowed_lot(
                account_equity,
                sl_pips=max(1.0, avg_sl),
                margin_available=margin_available,
                margin_used=account_snapshot.margin_used if account_snapshot else None,
                price=mid_price if mid_price > 0 else None,
                margin_rate=margin_rate,
                risk_pct_override=risk_override,
                open_long_units=long_units,
                open_short_units=short_units,
            )
            lot_total = round(lot_total * _lot_pattern_multiplier(fac_m1, story_snapshot), 3)
            if FORCE_SCALP_MODE and lot_total <= 0:
                logging.warning(
                    "[FORCE_SCALP] lot_total %.3f -> forcing floor %.3f",
                    lot_total,
                    MIN_SCALP_STAGE_LOT,
                )
                lot_total = MIN_SCALP_STAGE_LOT
            requested_pockets = {
                STRATEGIES[s].pocket
                for s in ranked_strategies
                if STRATEGIES.get(s)
            }
            scalp_share = 0.0
            if "scalp" in requested_pockets and weight_scalp is None:
                if account_snapshot:
                    scalp_share = dynamic_scalp_share(account_snapshot, DEFAULT_SCALP_SHARE)
                    logging.info(
                        "[SCALP-MAIN] share=%.3f buffer=%.3f free=%.1f%%",
                        scalp_share,
                        scalp_buffer if scalp_buffer is not None else -1.0,
                        (scalp_free_ratio * 100) if scalp_free_ratio is not None else -1.0,
                    )
                else:
                    scalp_share = DEFAULT_SCALP_SHARE
            atr_hint = None
            try:
                atr_hint = float(fac_m1.get("atr_pips") or (fac_m1.get("atr") or 0.0) * 100.0)
            except Exception:
                atr_hint = None
            perf_hint = None
            try:
                perf_summary = pos_manager.get_performance_summary()
                perf_hint = perf_summary.get("pockets")
            except Exception as exc:  # noqa: BLE001
                logging.warning("[PERF] failed to compute pocket PF: %s", exc)
                perf_hint = None
            update_dd_context(
                account_equity,
                weight_macro,
                weight_scalp,
                scalp_share,
                atr_pips=atr_hint,
                free_margin_ratio=scalp_free_ratio,
                perf_hint=perf_hint,
            )
            lots = alloc(
                lot_total,
                weight_macro,
                weight_scalp=weight_scalp,
                scalp_share=scalp_share,
            )
            if FORCE_SCALP_MODE and lot_total > 0:
                scalp_target = round(
                    max(
                        SCALP_MIN_ABS_LOT,
                        MIN_SCALP_STAGE_LOT,
                        lot_total * max(weight_scalp or SCALP_WEIGHT_READY_FLOOR, SCALP_WEIGHT_READY_FLOOR),
                    ),
                    3,
                )
                current_scalp = lots.get("scalp", 0.0)
                if current_scalp + 1e-6 < scalp_target:
                    shortfall = round(scalp_target - current_scalp, 3)
                    lots["scalp"] = round(scalp_target, 3)
                    for donor in ("macro", "micro"):
                        available = max(lots.get(donor, 0.0), 0.0)
                        if available <= 0:
                            continue
                        take = min(shortfall, available)
                        if take <= 0:
                            continue
                        lots[donor] = round(lots.get(donor, 0.0) - take, 3)
                        lots[donor] = max(lots[donor], 0.0)
                        shortfall = round(shortfall - take, 3)
                        if shortfall <= 0:
                            break
                    if shortfall > 0:
                        lots["micro"] = max(round(lots.get("micro", 0.0) - shortfall, 3), 0.0)
                logging.warning("[FORCE_SCALP] lots_alloc=%s total_lot=%.3f", lots, lot_total)
                log_metric(
                    "force_scalp_lot",
                    float(lots.get("scalp", 0.0)),
                    tags={"total": f"{lot_total:.3f}"},
                    ts=now,
                )
            for pocket_key in list(lots.keys()):
                if pocket_key not in focus_pockets:
                    lots[pocket_key] = 0.0
            active_pockets = {sig["pocket"] for sig in evaluated_signals}
            for key in list(lots):
                if key not in active_pockets:
                    lots[key] = 0.0
            if range_active and "macro" in lots:
                # レンジでもmacroを完全にゼロにせず抑制だけする
                lots["macro"] = round(lots["macro"] * 0.4, 3)

            signal_counts: dict[str, int] = {}
            for sig in evaluated_signals:
                pocket_name = sig["pocket"]
                signal_counts[pocket_name] = signal_counts.get(pocket_name, 0) + 1

            if lot_total > 0 and signal_counts:
                min_targets: dict[str, float] = {}
                if signal_counts.get("macro"):
                    min_targets["macro"] = round(lot_total * MIN_MACRO_WEIGHT, 3)
                if signal_counts.get("micro"):
                    min_targets["micro"] = round(lot_total * MIN_MICRO_WEIGHT, 3)
                if signal_counts.get("scalp"):
                    min_targets["scalp"] = round(lot_total * MIN_SCALP_WEIGHT, 3)

                def _boost_lot(pocket: str, target: float) -> None:
                    deficit = round(target - lots.get(pocket, 0.0), 3)
                    if deficit <= 0:
                        return
                    if pocket == "macro":
                        donors = ("micro", "scalp")
                    elif pocket == "micro":
                        donors = ("macro", "scalp")
                    else:
                        donors = ("macro", "micro")
                    for donor in donors:
                        if donor == pocket or donor not in lots:
                            continue
                        available = round(max(lots[donor], 0.0), 3)
                        if available <= 0:
                            continue
                        give = min(deficit, available)
                        if give <= 0:
                            continue
                        lots[pocket] = round(lots.get(pocket, 0.0) + give, 3)
                        lots[donor] = round(lots[donor] - give, 3)
                        deficit = round(target - lots[pocket], 3)
                        if deficit <= 1e-6:
                            break

                for pocket_name, target in min_targets.items():
                    _boost_lot(pocket_name, target)

                if signal_counts.get("scalp"):
                    lots.setdefault("scalp", 0.0)
                    scalp_target_abs = round(min(lot_total, SCALP_MIN_ABS_LOT), 3)
                    if scalp_target_abs > 0 and lots["scalp"] + 1e-6 < scalp_target_abs:
                        deficit = round(scalp_target_abs - lots["scalp"], 3)
                        for donor in ("macro", "micro"):
                            if deficit <= 0:
                                break
                            if donor not in lots:
                                continue
                            available = round(max(lots.get(donor, 0.0), 0.0), 3)
                            if available <= 0:
                                continue
                            give = min(deficit, available)
                            if give <= 0:
                                continue
                            lots["scalp"] = round(lots["scalp"] + give, 3)
                            lots[donor] = round(lots[donor] - give, 3)
                            deficit = round(deficit - give, 3)
                        if deficit > 0:
                            logging.info(
                                "[SCALP-MAIN] Unable to reach absolute lot floor %.3f (remaining=%.3f).",
                                scalp_target_abs,
                                deficit,
                        )

            if (
                not range_active
                and lot_total > 0
                and signal_counts.get("macro")
                and "macro" in lots
            ):
                desired_macro_lot = round(
                    min(
                        lot_total,
                        max(MIN_MACRO_TOTAL_LOT, lot_total * MACRO_LOT_SHARE_FLOOR),
                    ),
                    3,
                )
                current_macro_lot = round(max(lots.get("macro", 0.0), 0.0), 3)
                macro_deficit = round(desired_macro_lot - current_macro_lot, 3)
                reallocated = 0.0
                if macro_deficit > 0:
                    donor_candidates = [
                        pocket_name
                        for pocket_name in ("micro", "scalp")
                        if lots.get(pocket_name, 0.0) > 0
                    ]
                    for donor in sorted(
                        donor_candidates,
                        key=lambda name: lots.get(name, 0.0),
                        reverse=True,
                    ):
                        available = round(max(lots.get(donor, 0.0), 0.0), 3)
                        if available <= 0:
                            continue
                        transfer = min(available, macro_deficit)
                        if transfer <= 0:
                            continue
                        lots[donor] = round(lots[donor] - transfer, 3)
                        if lots[donor] < 1e-6:
                            lots[donor] = 0.0
                        lots["macro"] = round(lots["macro"] + transfer, 3)
                        reallocated = round(reallocated + transfer, 3)
                        macro_deficit = round(desired_macro_lot - lots["macro"], 3)
                        if macro_deficit <= 1e-6:
                            break
                macro_deficit = max(0.0, round(desired_macro_lot - lots["macro"], 3))
                if reallocated > 0:
                    share_pct = (lots["macro"] / lot_total) * 100 if lot_total > 0 else 0.0
                    logging.info(
                        "[ALLOCATION] Macro lot boosted %.3f -> %.3f (target=%.3f share=%.1f%% reallocated=%.3f)",
                        current_macro_lot,
                        lots["macro"],
                        desired_macro_lot,
                        share_pct,
                        reallocated,
                    )
                if macro_deficit > 0:
                    logging.debug(
                        "[ALLOCATION] Macro target unmet: desired=%.3f current=%.3f donors=%s",
                        desired_macro_lot,
                        lots.get("macro", 0.0),
                        {
                            key: round(max(lots.get(key, 0.0), 0.0), 3)
                            for key in ("micro", "scalp")
                        },
                    )

            # できるだけ余力を使う: 信号があるポケットで 92% 以上を配分する
            allocated_total = round(sum(max(v, 0.0) for v in lots.values()), 3)
            target_total = round(lot_total * 0.92, 3)
            if lot_total > 0 and allocated_total + 1e-6 < target_total:
                remaining = target_total - allocated_total
                active_with_signals = [p for p, cnt in signal_counts.items() if cnt > 0 and lots.get(p, 0.0) >= 0.0]
                if active_with_signals:
                    share = round(remaining / len(active_with_signals), 3)
                    for p in active_with_signals:
                        lots[p] = round(max(lots.get(p, 0.0), 0.0) + share, 3)

            if lot_total > 0:
                macro_lot = round(max(lots.get("macro", 0.0), 0.0), 3)
                micro_lot = round(max(lots.get("micro", 0.0), 0.0), 3)
                scalp_lot = round(max(lots.get("scalp", 0.0), 0.0), 3)
                macro_share = (macro_lot / lot_total) * 100 if lot_total > 0 else 0.0
                logging.info(
                    "[LOTS] total=%.3f macro=%.3f (%.1f%%) micro=%.3f scalp=%.3f",
                    lot_total,
                    macro_lot,
                    macro_share,
                    micro_lot,
                    scalp_lot,
                )
            if account_snapshot and margin_rate and margin_rate > 0:
                if MIN_MACRO_TOTAL_LOT > 0 and "macro" in lots:
                    macro_lot = round(max(lots.get("macro", 0.0), 0.0), 3)
                    if macro_lot + 1e-6 < MIN_MACRO_TOTAL_LOT:
                        shortfall = round(MIN_MACRO_TOTAL_LOT - macro_lot, 3)
                        for donor in ("micro", "scalp"):
                            available = round(max(lots.get(donor, 0.0), 0.0), 3)
                            if available <= 0:
                                continue
                            give = min(available, shortfall)
                            if give <= 0:
                                continue
                            lots[donor] = round(lots.get(donor, 0.0) - give, 3)
                            lots["macro"] = round(lots["macro"] + give, 3)
                            shortfall = round(shortfall - give, 3)
                            if shortfall <= 1e-6:
                                break

                if TARGET_MACRO_MARGIN_RATIO > 0 and "macro" in lots:
                    try:
                        margin_available_val = float(account_snapshot.margin_available)
                        margin_used_val = float(account_snapshot.margin_used)
                    except Exception:
                        margin_available_val = account_snapshot.margin_available or 0.0
                        margin_used_val = account_snapshot.margin_used or 0.0
                    total_margin_capacity = max(0.0, margin_available_val + margin_used_val)
                    if total_margin_capacity > 0.0:
                        macro_open_margin = 0.0
                        macro_trades = (open_positions.get("macro", {}) or {}).get("open_trades", [])
                        ref_price = mid_price if mid_price > 0 else base_price_val or 0.0
                        if ref_price <= 0:
                            ref_price = 152.0
                        for tr in macro_trades:
                            try:
                                units_val = abs(float(tr.get("units") or 0.0))
                                price_val = float(tr.get("price") or 0.0) or ref_price
                            except Exception:
                                continue
                            if units_val <= 0 or price_val <= 0:
                                continue
                            macro_open_margin += units_val * price_val * margin_rate

                        target_margin = total_margin_capacity * TARGET_MACRO_MARGIN_RATIO
                        extra_margin_needed = target_margin - macro_open_margin
                        if extra_margin_needed > 0:
                            safety_cap = total_margin_capacity * (1.0 - MACRO_MARGIN_SAFETY_BUFFER)
                            remaining_capacity = max(0.0, safety_cap - margin_used_val)
                            available_margin_headroom = max(0.0, min(margin_available_val, remaining_capacity))
                            extra_margin_needed = min(extra_margin_needed, available_margin_headroom)
                            if extra_margin_needed > 0:
                                desired_units = extra_margin_needed / (margin_rate * max(ref_price, 1e-6))
                                desired_lot = round(desired_units / 100000.0, 3)
                                if desired_lot > 0:
                                    lots["macro"] = round(max(lots.get("macro", 0.0), desired_lot), 3)

            lot_total = round(sum(max(value, 0.0) for value in lots.values()), 3)

            worker_lot_caps: dict[tuple[str, str], float] = {}
            if lot_total > 0:
                per_pocket_signals: dict[str, list[dict[str, object]]] = {}
                for sig in evaluated_signals:
                    if sig.get("action") not in {"OPEN_LONG", "OPEN_SHORT"}:
                        continue
                    pocket_name = sig.get("pocket")
                    strategy_name = sig.get("strategy")
                    if not pocket_name or not strategy_name:
                        continue
                    per_pocket_signals.setdefault(str(pocket_name), []).append(sig)

                for pocket_name, signals in per_pocket_signals.items():
                    pocket_budget = round(max(lots.get(pocket_name, 0.0), 0.0), 3)
                    if pocket_budget <= 0:
                        continue
                    weights: list[float] = []
                    for idx, sig in enumerate(signals):
                        try:
                            conf = max(0.3, min(1.0, (float(sig.get("confidence") or 0.0) / 100.0)))
                        except Exception:
                            conf = 0.3
                        health_scale = 1.0
                        try:
                            health_scale = float((sig.get("health") or {}).get("confidence_scale") or 1.0)
                        except Exception:
                            health_scale = 1.0
                        health_scale = max(0.65, min(1.35, health_scale))
                        rank_bonus = max(0.72, 1.0 - idx * 0.08)
                        weights.append(conf * health_scale * rank_bonus)
                    total_weight = sum(weights)
                    if total_weight <= 0:
                        continue
                    for sig, weight in zip(signals, weights):
                        pocket_key = str(sig.get("pocket"))
                        strategy_key = str(sig.get("strategy"))
                        share = weight / total_weight
                        cap = round(pocket_budget * share, 3)
                        worker_lot_caps[(pocket_key, strategy_key)] = cap

                if worker_lot_caps:
                    logging.info(
                        "[WORKER_ALLOC] per_strategy=%s",
                        {
                            f"{pocket}:{strategy}": lot
                            for (pocket, strategy), lot in worker_lot_caps.items()
                        },
                )

            spread_skip_logged = False
            pocket_limits_map = POCKET_MAX_ACTIVE_TRADES_RANGE if range_active else POCKET_MAX_ACTIVE_TRADES
            direction_limits_map = (
                POCKET_MAX_DIRECTIONAL_TRADES_RANGE if range_active else POCKET_MAX_DIRECTIONAL_TRADES
            )
            active_signal_pockets = {
                sig["pocket"]
                for sig in evaluated_signals
                if sig.get("action") in {"OPEN_LONG", "OPEN_SHORT"}
            }
            donated_pockets: set[str] = set()

            def _reallocate_blocked_lot(from_pocket: str, *, reason: str) -> None:
                if from_pocket in donated_pockets:
                    return
                available = round(max(lots.get(from_pocket, 0.0), 0.0), 3)
                if available <= 0:
                    return
                recipients: list[str] = []
                for sig in evaluated_signals:
                    pocket_name = sig.get("pocket")
                    if pocket_name == from_pocket or pocket_name in recipients:
                        continue
                    if pocket_name not in active_signal_pockets:
                        continue
                    info = open_positions.get(pocket_name, {}) or {}
                    current = int(info.get("trades", 0) or 0)
                    limit = pocket_limits_map.get(pocket_name, 0)
                    if limit <= 0 or current >= limit:
                        continue
                    recipients.append(pocket_name)
                if not recipients:
                    return
                share = round(available / len(recipients), 3)
                for pocket_name in recipients:
                    lots[pocket_name] = round(max(lots.get(pocket_name, 0.0), 0.0) + share, 3)
                lots[from_pocket] = 0.0
                donated_pockets.add(from_pocket)
                logging.info(
                    "[ALLOC] Reallocated %.3f lot from %s (%s) to %s",
                    available,
                    from_pocket,
                    reason,
                    ",".join(recipients),
                )

            entry_mix: dict[str, int] = {}
            for signal in evaluated_signals:
                pocket = signal["pocket"]
                action = signal.get("action")
                if action not in {"OPEN_LONG", "OPEN_SHORT"}:
                    continue
                if story_snapshot and not story_snapshot.is_aligned(pocket, action) and not (FORCE_SCALP_MODE and pocket == "scalp"):
                    # Relaxed: macroのみ厳密チェック。micro/scalpは記録だけ残して通す。
                    if pocket == "macro":
                        logging.info(
                            "[STORY] skip pocket=%s action=%s trend macro=%s micro=%s",
                            pocket,
                            action,
                            story_snapshot.macro_trend,
                            story_snapshot.micro_trend,
                        )
                        continue
                    else:
                        logging.info(
                            "[STORY] micro/scalp override pocket=%s action=%s macro_trend=%s micro_trend=%s higher_trend=%s",
                            pocket,
                            action,
                            story_snapshot.macro_trend,
                            story_snapshot.micro_trend,
                            story_snapshot.higher_trend,
                        )
                is_tactical = os.getenv('SCALP_TACTICAL','0').strip().lower() not in ('','0','false','no')
                direction = "long" if action == "OPEN_LONG" else "short"
                lose_streak, win_streak = stage_tracker.get_loss_profile(pocket, direction)
                high_impact_context = _evaluate_high_impact_context(
                    pocket=pocket,
                    direction=direction,
                    lose_streak=lose_streak,
                    win_streak=win_streak,
                    fac_m1=fac_m1,
                    fac_h4=fac_h4,
                    story_snapshot=story_snapshot,
                    macro_regime=macro_regime,
                    momentum=momentum,
                    atr_pips=atr_pips,
                    range_active=range_active,
                )
                allow_spread_bypass = False
                if spread_gate_active:
                    if pocket == 'scalp' and (is_tactical or spread_gate_soft_scalp):
                        allow_spread_bypass = True
                        if spread_gate_soft_scalp:
                            logging.info(
                                "[SPREAD] stale gating bypassed for scalp (reason=%s context=%s)",
                                spread_gate_reason,
                                spread_log_context,
                            )
                    if not allow_spread_bypass:
                        if (
                            pocket == "macro"
                            and high_impact_context
                            and high_impact_context.get("enabled")
                        ):
                            allow_spread_bypass = True
                            reason = str(high_impact_context.get("reason", "high_impact"))
                            logging.info(
                                "[SPREAD] high-impact macro bypass (reason=%s, gate=%s)",
                                reason,
                                spread_gate_reason,
                            )
                            try:
                                log_metric(
                                    "spread_override_high_impact",
                                    1.0,
                                    tags={
                                        "direction": direction,
                                        "reason": reason,
                                    },
                                )
                            except Exception:
                                pass
                    if not allow_spread_bypass:
                        if (
                            pocket == "macro"
                            and spread_gate_type == "hot"
                            and spread_snapshot
                        ):
                            try:
                                current_spread = float(spread_snapshot.get("spread_pips") or 0.0)
                            except (TypeError, ValueError):
                                current_spread = 0.0
                            if (
                                current_spread > 0.0
                                and current_spread <= MACRO_SPREAD_OVERRIDE
                                and weight_macro >= MIN_MACRO_WEIGHT
                            ):
                                allow_spread_bypass = True
                                logging.info(
                                    "[SPREAD] macro override enabled (spread=%.2fp <= %.2fp weight=%.2f focus=%s)",
                                    current_spread,
                                    MACRO_SPREAD_OVERRIDE,
                                    weight_macro,
                                    focus_tag,
                                )
                        if not allow_spread_bypass:
                            if not spread_skip_logged:
                                logging.info(
                                    "[SKIP] Spread guard active (%s, %s).",
                                    spread_gate_reason,
                                    spread_log_context,
                            )
                            spread_skip_logged = True
                        continue
                if event_soon and pocket in {"micro", "scalp"} and not is_tactical:
                    logging.info("[SKIP] Event soon, skipping %s pocket trade.", pocket)
                    continue
                # Tick watchdog: if scalp_tick loop stops reporting, avoid stale trading
                try:
                    last_tick_ago = (datetime.utcnow() - _last_tick_ts).total_seconds()
                except Exception:
                    last_tick_ago = 0
                if pocket == "scalp" and last_tick_ago > 15:
                    logging.warning("[WATCHDOG] Scalp tick stale %.1fs -> skip scalp trade", last_tick_ago)
                    continue
                if pocket in executed_pockets:
                    logging.info("[SKIP] %s pocket already handled this loop.", pocket)
                    continue
                if (range_active or override_macro_hold_active) and pocket == "macro":
                    reason = "range_active" if range_active else "macro_hold_after_breakout"
                    logging.info(
                        "[SKIP] %s, skipping macro entry (score=%.2f momentum=%.4f atr=%.2f vol5m=%.2f override=%s reason=%s hold_until=%s)",
                        reason,
                        range_ctx.score,
                        momentum,
                        atr_pips,
                        vol_5m,
                        override_release_active,
                        range_breakout_reason or range_ctx.reason,
                        (
                            range_macro_hold_until.isoformat(timespec="seconds")
                            if override_macro_hold_active
                            else "n/a"
                        ),
                    )
                    continue
                if not can_trade(pocket):
                    logging.info(f"[SKIP] DD limit for {pocket} pocket.")
                    continue

                total_lot_for_pocket = lots.get(pocket, 0.0)
                strategy_cap = worker_lot_caps.get((pocket, signal.get("strategy")))
                if strategy_cap is not None:
                    if strategy_cap + 1e-6 < total_lot_for_pocket:
                        logging.info(
                            "[ALLOC] pocket=%s strategy=%s capped %.3f -> %.3f",
                            pocket,
                            signal.get("strategy"),
                            total_lot_for_pocket,
                            strategy_cap,
                        )
                    total_lot_for_pocket = strategy_cap
                if total_lot_for_pocket <= 0:
                    continue

                confidence = max(0, min(100, signal.get("confidence", 50)))
                is_reduce_only = bool(signal.get("reduce_only"))
                raw_conf_factor = max(0.0, min(1.0, confidence / 100.0))
                base_conf_factor = raw_conf_factor if clamp_level > 0 else max(0.3, raw_conf_factor)
                confidence_factor = base_conf_factor
                if pocket == "macro" and not is_reduce_only and clamp_level == 0:
                    boosted_factor = max(confidence_factor, MACRO_CONFIDENCE_FLOOR)
                    if boosted_factor > 1.0:
                        boosted_factor = 1.0
                    if boosted_factor > confidence_factor + 1e-6:
                        logging.info(
                            "[ALLOCATION] Macro confidence factor boost %.2f -> %.2f (conf=%d)",
                            confidence_factor,
                            boosted_factor,
                            confidence,
                        )
                    confidence_factor = boosted_factor
                if pocket == "scalp" and not is_reduce_only and clamp_level == 0:
                    if confidence_factor + 1e-6 < SCALP_CONFIDENCE_FLOOR:
                        logging.info(
                            "[ALLOCATION] Scalp confidence factor floor %.2f -> %.2f (conf=%d)",
                            confidence_factor,
                            SCALP_CONFIDENCE_FLOOR,
                            confidence,
                        )
                    confidence_factor = min(1.0, max(confidence_factor, SCALP_CONFIDENCE_FLOOR))
                confidence_factor = min(confidence_factor, 1.0)
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

                if fs_strategy_client:
                    try:
                        strat_mult = float(
                            fs_strategy_client.get_multiplier(
                                str(signal.get("strategy") or ""),
                                pocket,
                            )
                        )
                    except Exception:
                        strat_mult = 1.0
                    if abs(strat_mult - 1.0) > 1e-3:
                        new_conf_target = round(confidence_target * strat_mult, 3)
                        logging.info(
                            "[FS_STRAT] pocket=%s strategy=%s mult=%.3f target %.3f -> %.3f",
                            pocket,
                            signal.get("strategy"),
                            strat_mult,
                            confidence_target,
                            new_conf_target,
                        )
                        confidence_target = new_conf_target

                open_info = open_positions.get(pocket, {})
                pocket_limits = pocket_limits_map
                per_direction_limits = direction_limits_map
                direction_limit = per_direction_limits.get(pocket, 1)
                if direction_limit <= 0:
                    logging.info(
                        "[SKIP] Pocket %s direction %s blocked by directional limit.",
                        pocket,
                        direction,
                    )
                    _reallocate_blocked_lot(pocket, reason="direction_limit")
                    continue
                open_trades = open_info.get("open_trades", []) or []
                direction_units_positive = direction == "long"
                same_direction_trades = sum(
                    1
                    for trade in open_trades
                    if (trade.get("units", 0) > 0) == direction_units_positive and trade.get("units", 0) != 0
                )
                opposite_direction_trades = sum(
                    1
                    for trade in open_trades
                    if (trade.get("units", 0) > 0) != direction_units_positive and trade.get("units", 0) != 0
                )
                max_trades_allowed = pocket_limits.get(pocket, 1)
                current_trades = int(open_info.get("trades", 0) or 0)
                hedging_override = False
                if HEDGING_ENABLED and current_trades >= max_trades_allowed:
                    if (
                        current_trades == max_trades_allowed
                        and same_direction_trades == 0
                        and opposite_direction_trades > 0
                    ):
                        hedging_override = True
                        logging.info(
                            "[HEDGE] Allowing %s entry in %s pocket beyond cap (%d).",
                            direction,
                            pocket,
                            max_trades_allowed,
                        )
                if not hedging_override and current_trades >= max_trades_allowed:
                    logging.info(
                        "[SKIP] Pocket %s has %d/%d active trades. Skipping new entry.",
                        pocket,
                        current_trades,
                        max_trades_allowed,
                    )
                    _reallocate_blocked_lot(pocket, reason="pocket_cap")
                    continue
                if same_direction_trades >= direction_limit:
                    logging.info(
                        "[SKIP] Pocket %s %s has %d/%d open trades. Skipping.",
                        pocket,
                        direction,
                        same_direction_trades,
                        direction_limit,
                    )
                    _reallocate_blocked_lot(pocket, reason="direction_cap")
                    continue

                try:
                    price = float(mid_price) if mid_price > 0 else float(fac_m1.get("close") or 0.0)
                except (TypeError, ValueError):
                    price = 0.0
                if action == "OPEN_LONG":
                    open_units = int(open_info.get("long_units", 0))
                    ref_price = open_info.get("long_avg_price")
                else:
                    open_units = int(open_info.get("short_units", 0))
                    ref_price = open_info.get("short_avg_price")

                # マクロのステージ計算は自ポケットのみを基準にし、他ポケットの同方向エクスポージャは考慮しない
                if pocket == "macro":
                    cluster_units = open_units
                else:
                    cluster_units = _cluster_directional_units(direction, open_positions)

                is_buy = action == "OPEN_LONG"
                entry_price = price
                if is_buy and tick_ask is not None:
                    entry_price = float(tick_ask)
                elif not is_buy and tick_bid is not None:
                    entry_price = float(tick_bid)
                if _market_guarded_skip(signal, entry_price, context="final"):
                    continue

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
                    ref_price = entry_price
                if ref_price is not None:
                    stage_context["avg_price"] = ref_price
                if pocket == "macro":
                    stage_context["cluster_units"] = cluster_units
                    stage_context["cluster_direction"] = direction
                reduce_only = bool(signal.get("reduce_only"))
                reduce_cap_units = int(signal.get("reduce_cap_units") or 0)
                proposed_units_raw = signal.get("proposed_units")
                proposed_units: Optional[int] = None
                try:
                    if proposed_units_raw is not None:
                        proposed_units = int(float(proposed_units_raw))
                except Exception:
                    proposed_units = None

                high_impact_context = _evaluate_high_impact_context(
                    pocket=pocket,
                    direction=direction,
                    lose_streak=lose_streak,
                    win_streak=win_streak,
                    fac_m1=fac_m1,
                    fac_h4=fac_h4,
                    story_snapshot=story_snapshot,
                    macro_regime=macro_regime,
                    momentum=momentum,
                    atr_pips=atr_pips,
                    range_active=range_active,
                )

                staged_lot, stage_idx = compute_stage_lot(
                    pocket,
                    confidence_target,
                    cluster_units,
                    action,
                    fac_m1,
                    fac_h4,
                    stage_context,
                    high_impact_context=high_impact_context,
                )
                # 負け方向の深いステージは大幅抑制（逆張りを止める/薄くする）
                if staged_lot > 0 and stage_idx is not None:
                    if stage_idx >= 2 and bias_h4 != 0 and action_dir != 0 and action_dir != bias_h4:
                        staged_lot *= 0.4
                        logging.info(
                            "[STAGE_GUARD] downscale lot due to stage=%s against H4 dir=%s->action=%s lot=%.3f",
                            stage_idx,
                            bias_h4,
                            action_dir,
                            staged_lot,
                        )
                    if stage_idx >= 3 and lose_streak >= 2:
                        staged_lot *= 0.5
                        logging.info(
                            "[STAGE_GUARD] downscale lot lose_streak=%s stage=%s lot=%.3f",
                            lose_streak,
                            stage_idx,
                            staged_lot,
                        )
                if staged_lot <= 0:
                    if pocket == "scalp":
                        logging.info(
                            "[SCALP_STAGE] lot_zero stage=%s confidence=%.3f open_units=%s plan=%s ready=%s",
                            stage_idx,
                            confidence_target,
                            open_units,
                            _stage_plan(pocket),
                            scalp_ready,
                        )
                    continue

                units = int(round(staged_lot * 100000)) * (
                    1 if action == "OPEN_LONG" else -1
                )
                # ステージ配分で極小になった場合でも最低1k unitsは確保する
                if units != 0 and abs(units) < 1000:
                    units = 1000 if units > 0 else -1000
                    staged_lot = abs(units) / 100000.0
                if reduce_only and proposed_units:
                    sign = 1 if action == "OPEN_LONG" else -1
                    units = sign * abs(proposed_units)
                    staged_lot = abs(units) / 100000.0
                if units == 0:
                    logging.info(
                        "[SKIP] Stage lot %.3f produced 0 units. Skipping.", staged_lot
                    )
                    continue
                # 同方向でエクスポージャ上限を超える場合はユニットを自動調整（手動ポジは無視）
                if action_dir != 0 and net_units != 0 and account_equity > 0 and mid_price > 0:
                    same_dir = (net_units > 0 and units > 0) or (net_units < 0 and units < 0)
                    if same_dir:
                        current_notional = abs(net_units) * mid_price
                        cap_notional = exposure_hard_cap * account_equity
                        remain_notional = cap_notional - current_notional
                        if remain_notional <= 0:
                            logging.info(
                                "[RISK] auto-adjust skip (no remaining notional) net=%.0f cap=%.2f",
                                net_units,
                                exposure_hard_cap,
                            )
                            continue
                        allowed_units = int(remain_notional / mid_price)
                        if allowed_units < abs(units):
                            adj_units = max(0, allowed_units)
                            if adj_units == 0:
                                logging.info(
                                    "[RISK] auto-adjust produced 0 units (net=%.0f cap=%.2f)",
                                    net_units,
                                    exposure_hard_cap,
                                )
                                continue
                            logging.info(
                                "[RISK] auto-adjust units %d -> %d to fit cap %.2f (net=%.0f)",
                                units,
                                adj_units if units > 0 else -adj_units,
                                exposure_hard_cap,
                                net_units,
                            )
                            units = adj_units if units > 0 else -adj_units
                if reduce_only:
                    net_dir = 0
                    if net_units > 0:
                        net_dir = 1
                    elif net_units < 0:
                        net_dir = -1
                    # allow only net-reducing orders
                    if net_dir != 0 and ((units > 0 and net_dir > 0) or (units < 0 and net_dir < 0)):
                        logging.info(
                            "[CLAMP] Reduce-only skip (same dir) units=%d net=%d",
                            units,
                            net_units,
                        )
                        continue
                    cap_units = abs(net_units)
                    if reduce_cap_units > 0:
                        cap_units = min(cap_units, reduce_cap_units)
                    capped_units = min(abs(units), cap_units)
                    if capped_units <= 0:
                        logging.info("[CLAMP] Reduce-only produced 0 units. Skipping.")
                        continue
                    units = capped_units if units > 0 else -capped_units
                # Gross exposure guard: 両サイド合計が上限を超えないよう調整
                if (
                    not reduce_only
                    and account_equity > 0
                    and mid_price > 0
                    and GROSS_EXPOSURE_HARD > 0
                ):
                    gross_units = side_long_units + side_short_units
                    gross_exposure = gross_units * mid_price / account_equity
                    gross_after_units = gross_units + abs(units)
                    gross_after_exposure = gross_after_units * mid_price / account_equity
                    netting_reduce = net_units != 0 and units != 0 and (net_units * units) < 0
                    # Nettingで総ノッチ（long+short）が増えない方向のオーダーは gross cap を無視する
                    if not netting_reduce and gross_after_exposure >= GROSS_EXPOSURE_HARD:
                        max_units_allowed = int((GROSS_EXPOSURE_HARD * account_equity) / mid_price) - gross_units
                        if max_units_allowed <= 0:
                            logging.info(
                                "[RISK] gross cap hit (%.2f>=%.2f) skip strategy=%s dir=%s",
                                gross_after_exposure,
                                GROSS_EXPOSURE_HARD,
                                signal.get("strategy"),
                                direction,
                            )
                            continue
                        adj_units = min(abs(units), max_units_allowed)
                        if adj_units <= 0:
                            logging.info(
                                "[RISK] gross cap adjust produced 0 units strategy=%s dir=%s",
                                signal.get("strategy"),
                                direction,
                            )
                            continue
                        if adj_units < abs(units):
                            units = adj_units if units > 0 else -adj_units
                            staged_lot = abs(units) / 100000.0
                            logging.info(
                                "[RISK] gross cap adjust units -> %d gross_after=%.2f cap=%.2f strategy=%s dir=%s",
                                units,
                                (gross_units + abs(units)) * mid_price / account_equity,
                                GROSS_EXPOSURE_HARD,
                                signal.get("strategy"),
                                direction,
                            )

                entry_context_payload = _build_entry_context(
                    pocket=pocket,
                    direction=direction,
                    stage_index=stage_idx,
                    size_factor=size_factor,
                    confidence_target=confidence_target,
                    range_active=range_active,
                    macro_regime=macro_regime,
                    micro_regime=micro_regime,
                    momentum=momentum,
                    atr_pips=atr_pips,
                    vol_5m=vol_5m,
                    lose_streak=lose_streak,
                    win_streak=win_streak,
                    high_impact_enabled=bool(
                        high_impact_context and high_impact_context.get("enabled")
                    ),
                    high_impact_reason=(
                        str(high_impact_context.get("reason"))
                        if high_impact_context and high_impact_context.get("enabled")
                        else None
                    ),
                )

                pullback_note = None
                limit_wait_key = None
                entry_type = signal.get("entry_type", "market")
                target_price = signal.get("entry_price")
                tolerance_pips = float(signal.get("entry_tolerance_pips", 0.25))
                if (
                    pocket == "macro"
                    and stage_idx == 0
                    and entry_type == "market"
                    and not reduce_only
                ):
                    atr_hint = fac_m1.get("atr_pips")
                    if atr_hint is None:
                        atr_hint = (fac_m1.get("atr") or 0.0) * 100
                    try:
                        atr_hint = float(atr_hint or 0.0)
                    except (TypeError, ValueError):
                        atr_hint = 0.0
                    pullback_min, pullback_max = _macro_pullback_threshold(1, atr_hint)
                    pullback_pips = _dynamic_macro_pullback_pips(
                        atr_hint,
                        momentum,
                        pullback_min,
                        pullback_max,
                    )
                    limit_wait_key = f"{signal.get('strategy') or 'macro'}:{direction}"
                    if is_buy:
                        target_price = round(entry_price - pullback_pips * PIP, 3)
                    else:
                        target_price = round(entry_price + pullback_pips * PIP, 3)
                    entry_type = "limit"
                    tolerance_pips = max(
                        tolerance_pips,
                        max(
                            0.35,
                            min(
                                1.25,
                                pullback_pips * (0.28 if atr_hint <= 2.0 else 0.33)
                                + (0.18 if abs(momentum) <= 0.008 else 0.08),
                            ),
                        ),
                        min(1.8, 0.42 * atr_hint + 0.28),
                    )
                    pullback_note = pullback_pips
                tolerance_price = tolerance_pips * PIP
                reference_price = entry_price
                if entry_type == "limit":
                    spread_pips = None
                    try:
                        if tick_ask is not None and tick_bid is not None:
                            spread_pips = max(0.0, (float(tick_ask) - float(tick_bid)) / PIP)
                    except Exception:
                        spread_pips = None
                    if target_price is None:
                        logging.info("[LIMIT] Missing entry_price for %s.", signal["strategy"])
                        continue
                    try:
                        reference_price = float(target_price)
                    except (TypeError, ValueError):
                        logging.info("[LIMIT] Invalid entry_price for %s.", signal["strategy"])
                        continue
                    else:
                        target_price = reference_price
                    # limit待ちは廃止して即マーケット化（取り逃し防止）
                    entry_type = "market"
                    reference_price = entry_price
                    target_price = None
                    if limit_wait_key:
                        MACRO_LIMIT_WAIT.pop(limit_wait_key, None)

                # ensure sl/tp price placeholders exist for logging
                sl_price = signal.get("sl_price")
                tp_price = signal.get("tp_price")
                try:
                    sl_price = float(sl_price) if sl_price is not None else None
                except Exception:
                    sl_price = None
                try:
                    tp_price = float(tp_price) if tp_price is not None else None
                except Exception:
                    tp_price = None

                logging.info(
                    "[ENTRY_PLAN] strategy=%s pocket=%s action=%s stage=%s lot=%.3f units=%d reduce_only=%s sl=%.3f tp=%.3f entry_type=%s",
                    signal.get("strategy"),
                    pocket,
                    action,
                    stage_idx,
                    staged_lot,
                    units,
                    reduce_only,
                    sl_price if sl_price is not None else -1.0,
                    tp_price if tp_price is not None else -1.0,
                    entry_type,
                )
                sl_pips = signal.get("sl_pips")
                if sl_pips is None:
                    hard_stop = signal.get("hard_stop_pips")
                    if hard_stop is not None:
                        sl_pips = hard_stop
                tp_pips = signal.get("tp_pips")
                if reduce_only:
                    if sl_pips is None:
                        sl_pips = REDUCE_ONLY_DEFAULT_SL_PIPS
                    if tp_pips is None:
                        tp_pips = REDUCE_ONLY_DEFAULT_TP_PIPS
                if fs_strategy_client and fs_strategy_apply_sltp:
                    try:
                        tp_override, sl_override = fs_strategy_client.get_sltp(
                            str(signal.get("strategy") or ""),
                            pocket,
                            None,
                        )
                        if tp_override is not None:
                            tp_pips = max(1.0, float(tp_override))
                            signal["tp_pips"] = tp_pips
                        if sl_override is not None:
                            sl_pips = max(1.0, float(sl_override))
                            signal["sl_pips"] = sl_pips
                    except Exception as exc:
                        logging.info("[FS_STRAT] sltp override skipped: %s", exc)
                if tp_pips is None and not reduce_only:
                    logging.info("[SKIP] Missing TP for %s.", signal["strategy"])
                    continue
                # ATR/RSI/ADX は必須。欠損時は M1 ローソクから再計算を試みる。
                atr_entry = fac_m1.get("atr_pips")
                if atr_entry is None:
                    atr_entry = (fac_m1.get("atr") or 0.0) * 100
                rsi_entry = fac_m1.get("rsi")
                adx_entry = fac_m1.get("adx")
                missing = False
                try:
                    atr_entry = float(atr_entry)
                    rsi_entry = float(rsi_entry)
                    adx_entry = float(adx_entry)
                except Exception:
                    missing = True
                if missing or atr_entry <= 0.0 or math.isnan(atr_entry) or math.isnan(rsi_entry) or math.isnan(adx_entry):
                    rec_atr, rec_rsi, rec_adx = _recompute_m1_technicals(fac_m1)
                    if rec_atr is not None and rec_rsi is not None and rec_adx is not None and rec_atr > 0.0:
                        atr_entry, rsi_entry, adx_entry = rec_atr, rec_rsi, rec_adx
                        logging.info("[RECOMPUTE] Filled ATR/RSI/ADX from candles atr=%.3f rsi=%.2f adx=%.2f", atr_entry, rsi_entry, adx_entry)
                    else:
                        logging.error("[SKIP] Missing/invalid ATR/RSI/ADX for %s (recompute failed)", signal["strategy"])
                        continue

                base_sl = None
                sl = None
                tp = None
                if not reduce_only:
                    if sl_pips is not None:
                        base_sl = (
                            reference_price - (sl_pips / 100)
                            if is_buy
                            else reference_price + (sl_pips / 100)
                        )
                    base_tp = (
                        reference_price + (tp_pips / 100)
                        if is_buy
                        else reference_price - (tp_pips / 100)
                    )

                    level_map_note = None
                    if level_map_client and level_map_enabled:
                        try:
                            lm = level_map_client.nearest(reference_price)
                            if lm:
                                level_map_note = {
                                    "bucket": lm.get("bucket"),
                                    "hit_count": lm.get("hit_count"),
                                    "p_up_5": lm.get("p_up_5"),
                                    "p_down_5": lm.get("p_down_5"),
                                    "mean_ret_5": lm.get("mean_ret_5"),
                                }
                        except Exception:
                            level_map_note = None

                    sl, tp = clamp_sl_tp(
                        reference_price,
                        base_sl,
                        base_tp,
                        is_buy,
                    )
                else:
                    level_map_note = None

                strategy_tag = (
                    signal.get("tag")
                    or signal.get("strategy")
                    or signal.get("strategy_tag")
                    or signal.get("profile")
                    or "unknown_signal"
                )
                signal["tag"] = strategy_tag

                client_id = build_client_order_id(focus_tag, strategy_tag)
                # Build a lightweight entry thesis for contextual exits
                thesis_type = (
                    "trend_follow" if pocket == "macro" else ("mean_reversion" if pocket == "micro" else "scalp")
                )
                h4_ma10 = fac_h4.get("ma10")
                h4_ma20 = fac_h4.get("ma20")
                # Strategy-provided metaで上書きし、無い場合はポケット別のデフォルトを使う
                def _opt_float(val):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None

                strategy_profile = signal.get("profile") or signal.get("strategy_profile")
                target_tp_hint = _opt_float(signal.get("target_tp_pips"))
                loss_guard_hint = _opt_float(signal.get("loss_guard_pips") or signal.get("loss_grace_pips"))
                min_hold_sec = _opt_float(signal.get("min_hold_sec") or signal.get("min_hold_seconds"))
                if min_hold_sec is None:
                    min_hold_min = 11.0 if pocket == "macro" else (5.0 if pocket == "micro" else 3.0)
                    min_hold_sec = min_hold_min * 60.0
                fast_cut_pips = _opt_float(signal.get("fast_cut_pips"))
                fast_cut_time = None
                try:
                    fast_cut_time = int(float(signal.get("fast_cut_time_sec")))
                except (TypeError, ValueError):
                    fast_cut_time = None
                fast_cut_hard = _opt_float(signal.get("fast_cut_hard_mult"))
                exit_tags = signal.get("exit_tags") or signal.get("tags")
                if fast_cut_pips is None or fast_cut_time is None or fast_cut_hard is None:
                    if pocket == "scalp":
                        fast_cut_pips = _opt_float(fast_cut_pips) or round(
                            max(6.0, atr_entry * (1.0 if not range_active else 0.9)), 2
                        )
                        fast_cut_time = fast_cut_time or int(max(60.0, atr_entry * 15.0))
                        fast_cut_hard = fast_cut_hard or 1.6
                    elif pocket == "macro":
                        fast_cut_pips = _opt_float(fast_cut_pips) or round(max(8.0, atr_entry * 1.5), 2)
                        fast_cut_time = fast_cut_time or int(max(120.0, atr_entry * 18.0))
                        fast_cut_hard = fast_cut_hard or 1.8
                    else:
                        fast_cut_pips = _opt_float(fast_cut_pips) or round(max(6.5, atr_entry * 1.1), 2)
                        fast_cut_time = fast_cut_time or int(max(90.0, atr_entry * 16.0))
                        fast_cut_hard = fast_cut_hard or 1.6
                target_tp = target_tp_hint if target_tp_hint is not None else _opt_float(tp_pips)
                loss_guard = loss_guard_hint if loss_guard_hint is not None else _opt_float(sl_pips)
                draft_thesis = signal.get("entry_thesis")
                if not isinstance(draft_thesis, dict):
                    draft_thesis = None
                base_thesis = {
                    "type": thesis_type,
                    "strategy": signal.get("strategy"),
                    "strategy_tag": strategy_tag,
                    "profile": strategy_profile,
                    "confidence": confidence,
                    "tag": signal.get("tag"),
                    "pocket": pocket,
                    "action": action,
                    "entry_type": entry_type,
                    "entry_ref": reference_price,
                    "limit_target": target_price,
                    "entry_ts": now.isoformat(timespec="seconds"),
                    "sl_pips": sl_pips,
                    "tp_pips": tp_pips,
                    "hard_stop_pips": signal.get("hard_stop_pips"),
                    "target_tp_pips": target_tp,
                    "loss_guard_pips": loss_guard,
                    "min_hold_sec": min_hold_sec,
                    "min_hold_minutes": min_hold_sec / 60.0 if min_hold_sec else None,
                    "factors": {
                        "m1": {
                            "rsi": rsi_entry,
                            "adx": adx_entry,
                            "ema20": fac_m1.get("ema20") or fac_m1.get("ma20"),
                            "ma10": fac_m1.get("ma10"),
                            "ma20": fac_m1.get("ma20"),
                            "atr_pips": atr_entry,
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
                    "story": {
                        "macro": story_snapshot.macro_trend if story_snapshot else None,
                        "micro": story_snapshot.micro_trend if story_snapshot else None,
                        "higher": story_snapshot.higher_trend if story_snapshot else None,
                        "volatility": story_snapshot.volatility_state if story_snapshot else None,
                    },
                    "levels": story_snapshot.major_levels if story_snapshot else None,
                    "context": entry_context_payload,
                    "fast_cut_pips": fast_cut_pips,
                    "fast_cut_time_sec": fast_cut_time,
                    "fast_cut_hard_mult": fast_cut_hard,
                    "kill_switch": True,
                    "exit_tags": exit_tags,
                    "regime": {
                        "range_active": bool(range_active),
                        "macro": macro_regime,
                        "micro": micro_regime,
                    },
                }
                if draft_thesis:
                    entry_thesis = dict(draft_thesis)
                    entry_thesis.update(base_thesis)
                else:
                    entry_thesis = base_thesis
                is_mr_signal = _is_mr_signal(strategy_tag, strategy_profile)
                is_mr_overlay = _is_mr_overlay_signal(strategy_tag, strategy_profile)
                if is_mr_signal or is_mr_overlay:
                    _augment_entry_thesis_for_mr(
                        entry_thesis,
                        pocket=pocket,
                        atr_entry=atr_entry,
                        overlay=is_mr_overlay,
                    )
                    if is_mr_signal:
                        entry_thesis.setdefault("mr_guard", _mr_guard_snapshot(pocket))
                    if is_mr_overlay:
                        entry_thesis.setdefault("mr_overlay", True)
                execution_cfg = signal.get("execution")
                if isinstance(execution_cfg, dict):
                    entry_thesis["execution"] = execution_cfg
                if level_map_note:
                    entry_thesis["level_map"] = level_map_note
                note = signal.get("notes")
                if note:
                    entry_thesis["note"] = note
                if pullback_note is not None:
                    entry_thesis.setdefault("notes_auto", {})[
                        "macro_pullback_pips"
                    ] = pullback_note
                shadow_enabled = pocket == "scalp" and (SCALP_SHADOW_MODE or SCALP_SHADOW_LOG_ONLY)
                if shadow_enabled:
                    latency_ms = max(0.0, (time.monotonic() - loop_start_mono) * 1000.0)
                    shadow_pass, shadow_reason, shadow_ctx = _scalp_shadow_gate(
                        spread_state=spread_snapshot,
                        spread_gate_reason=spread_gate_reason,
                        atr_pips=atr_entry,
                        latency_ms=latency_ms,
                    )
                    shadow_payload = {
                        "ts": datetime.datetime.utcnow().isoformat(),
                        "mode": "shadow" if SCALP_SHADOW_MODE else "log",
                        "pocket": pocket,
                        "strategy": signal.get("strategy"),
                        "tag": signal.get("tag"),
                        "action": action,
                        "units": units,
                        "stage": stage_idx + 1,
                        "lot": staged_lot,
                        "price": price,
                        "sl_pips": sl_pips,
                        "tp_pips": tp_pips,
                        "range": bool(range_active),
                        "event_soon": bool(event_soon),
                        "spread_gate": bool(spread_gate_reason),
                        "spread_reason": spread_gate_reason,
                        "shadow_pass": shadow_pass,
                        "shadow_reason": shadow_reason,
                    }
                    shadow_payload.update(shadow_ctx)
                    _write_scalp_shadow_log(shadow_payload)
                    if SCALP_SHADOW_MODE:
                        logging.info(
                            "[SCALP_SHADOW] skip order strategy=%s pass=%s reason=%s spread=%.3f med=%.3f lat=%.0f atr=%.2f",
                            signal.get("strategy"),
                            shadow_pass,
                            shadow_reason,
                            shadow_ctx.get("spread_pips") or -1.0,
                            shadow_ctx.get("spread_p50") or -1.0,
                            shadow_ctx.get("latency_ms") or -1.0,
                            shadow_ctx.get("atr_pips") or -1.0,
                        )
                        continue
                trade_id = await market_order(
                    "USD_JPY",
                    units,
                    sl,
                    tp,
                    pocket,
                    client_order_id=client_id,
                    strategy_tag=signal.get("tag")
                    or signal.get("strategy")
                    or sname
                    or "unknown",
                    entry_thesis=entry_thesis,
                    arbiter_final=True,
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
                    if not reduce_only:
                        if stage_idx >= 0:
                            stage_tracker.set_stage(pocket, direction, stage_idx + 1, now=now)
                        pos_manager.register_open_trade(trade_id, pocket, client_id)
                        info = open_positions.setdefault(
                            pocket,
                            {
                                "units": 0,
                                "avg_price": entry_price or price or 0.0,
                                "trades": 0,
                                "long_units": 0,
                                "long_avg_price": 0.0,
                                "short_units": 0,
                                "short_avg_price": 0.0,
                            },
                        )
                        info["units"] = info.get("units", 0) + units
                        info["trades"] = info.get("trades", 0) + 1
                        if entry_price is not None:
                            info["avg_price"] = entry_price
                            if units > 0:
                                prev_units = info.get("long_units", 0)
                                new_units = prev_units + units
                                if new_units > 0:
                                    if prev_units == 0:
                                        info["long_avg_price"] = entry_price
                                    else:
                                        info["long_avg_price"] = (
                                            info.get("long_avg_price", entry_price) * prev_units
                                            + entry_price * units
                                        ) / new_units
                                info["long_units"] = new_units
                            else:
                                trade_size = abs(units)
                                prev_units = info.get("short_units", 0)
                                new_units = prev_units + trade_size
                                if new_units > 0:
                                    if prev_units == 0:
                                        info["short_avg_price"] = entry_price
                                    else:
                                        info["short_avg_price"] = (
                                            info.get("short_avg_price", entry_price) * prev_units
                                            + entry_price * trade_size
                                        ) / new_units
                                info["short_units"] = new_units
                    net_units += units
                    open_positions.setdefault("__net__", {})["units"] = net_units
                    executed_pockets.add(pocket)
                    if not reduce_only:
                        key = f"{signal.get('strategy')}@{pocket}"
                        entry_mix[key] = entry_mix.get(key, 0) + 1
                        # 直後の再エントリーを抑制（気迷いトレード対策）
                        entry_cd = _dynamic_entry_cooldown_seconds(
                            pocket,
                            fac_m1,
                            fac_h1,
                            fac_h4,
                            range_active,
                        )
                        stage_tracker.set_cooldown(
                            pocket,
                            direction,
                            reason="entry_rate_limit",
                            seconds=entry_cd,
                            now=now,
                        )
                else:
                    logging.error(f"[ORDER FAILED] {signal['strategy']}")

            if entry_mix:
                logging.info(
                    "[ENTRY_MIX] %s",
                    ", ".join(
                        f"{k}:{v}"
                        for k, v in sorted(entry_mix.items(), key=lambda kv: kv[1], reverse=True)
                    ),
                )

            # --- 5. 決済済み取引の同期 ---
            try:
                if fac_m1.get("rsi") is not None:
                    last_rsi_m1 = float(fac_m1.get("rsi") or last_rsi_m1)
            except Exception:
                pass
            try:
                if close_px_value is not None:
                    last_close_m1 = close_px_value
            except Exception:
                pass
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
    handlers = [
        ("M1", m1_candle_handler),
        ("M5", m5_candle_handler),
        ("H1", h1_candle_handler),
        ("H4", h4_candle_handler),
        ("D1", d1_candle_handler),
    ]
    seeded = await initialize_history("USD_JPY")
    if not seeded:
        logging.warning("[HISTORY] Startup seeding incomplete, continuing with live feed.")

    while True:
        # 周辺コンポーネント（GPT/アドバイザー等）の初期化
        gpt_state = GPTDecisionState()
        gpt_requests = GPTRequestManager()
        fast_scalp_state = FastScalpState()
        rr_advisor = RRRatioAdvisor()
        exit_advisor = ExitAdvisor()
        strategy_conf_advisor = StrategyConfidenceAdvisor()
        focus_advisor = FocusOverrideAdvisor()
        volatility_advisor = VolatilityBiasAdvisor()
        stage_plan_advisor = StagePlanAdvisor()
        partial_advisor = PartialReductionAdvisor()

        tasks = [
            asyncio.create_task(
                supervised_runner(
                    "candle_stream",
                    start_candle_stream("USD_JPY", handlers),
                )
            ),
            asyncio.create_task(
                supervised_runner(
                    "worker_only_loop",
                    worker_only_loop(),
                )
            ),
            asyncio.create_task(
                supervised_runner(
                    "gpt_worker",
                    gpt_worker(gpt_state, gpt_requests),
                )
            ),
        ]

        # 先にGPTの初期決定を温めておく（ロジック開始前のプリム）
        try:
            await prime_gpt_decision(gpt_state, gpt_requests)
        except Exception:
            logging.exception("[MAIN] prime_gpt_decision failed; continuing without primer")

        run_logic = (MAIN_TRADING_ENABLED or SIGNAL_GATE_ENABLED) and (
            not WORKER_ONLY_MODE or SIGNAL_GATE_ENABLED
        )
        if run_logic:
            tasks.append(
                asyncio.create_task(
                    supervised_runner(
                        "logic_loop",
                        logic_loop(
                            gpt_state,
                            gpt_requests,
                            fast_scalp_state=fast_scalp_state,
                            rr_advisor=rr_advisor,
                            exit_advisor=exit_advisor,
                            strategy_conf_advisor=strategy_conf_advisor,
                            focus_advisor=focus_advisor,
                            volatility_advisor=volatility_advisor,
                            stage_plan_advisor=stage_plan_advisor,
                            partial_advisor=partial_advisor,
                        ),
                    )
                )
            )
        else:
            logging.info(
                "[MAIN] logic_loop disabled main_trading_enabled=%s worker_only=%s signal_gate=%s",
                MAIN_TRADING_ENABLED,
                WORKER_ONLY_MODE,
                SIGNAL_GATE_ENABLED,
            )
        try:
            await asyncio.gather(*tasks)
            logging.error("[SUPERVISOR] Task group exited cleanly; restarting in 5s.")
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
        except Exception:
            logging.exception("[SUPERVISOR] Worker-only task group crashed; restarting in 5s.")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
